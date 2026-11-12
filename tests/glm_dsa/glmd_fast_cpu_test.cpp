// tests/glm_dsa/glmd_fast_cpu_test.cpp - the glm-dsa decode attention twins (B4.3-B4.5) on the CPU.
//
// The twins in glmd::twin follow the GPU kernels' algorithms and storage formats (glmd_fast.hpp), so what passes
// here is the arithmetic and the selection logic the kernels run; tests/glm_dsa/glmd_fast_parity.cu then pins the
// kernels to the twins on a GPU.
//   fp16     the host float <-> fp16 conversion: every finite fp16 round-trips, round-to-nearest-even cases;
//   top-k    10^4 random rows, n in {2049, 8192, 131072} (normal scores, heavy ties, +-0, -inf), the selection
//            equals a sort-based oracle (score descending, ties to the lower cell, top 2048) in position order;
//            identity below 2049 cells; per-row n from positions;
//   prep     GLM-5.3 sizes at positions {0, 1, 7, 2047, 2048, 32767}: q_a RMS and its q8_1, latent (FP16 and
//            INT8 records), k_pe rope, index key LayerNorm + rope, q-side rope vs the B3.4 CPU reference:
//            max abs err < 1e-5 beyond the storage rounding (half an fp16 ulp, half an INT8 / q8_1 step);
//   mla2     q_abs, the rope-tail logits at scale 1/16, the chunked softmax, ctx and W_vb vs the reference absorbed
//            attention (cpu::mla_absorb_q + cpu::mla_attention on the same stored values): rel L2 < 1e-4 at
//            n_sel 1, 37 (with padding), 2048, both latent formats; the Flash scale 1/sqrt(192) must fail.
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_ref.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

int g_fail = 0;

void expect(bool ok, const char* what) {
    std::printf("%-72s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

void gate(const char* what, double err, double lim) {
    const bool ok = err < lim;
    std::printf("%-56s %.3e (gate %.1e) %s\n", what, err, lim, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

std::vector<float> randv(std::mt19937& rng, size_t n, float scale = 1.0f, float mean = 0.0f) {
    std::normal_distribution<float> d(mean, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

uint16_t bf16_bits(float f) {   // round to nearest even (finite inputs)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    x += 0x7fffu + ((x >> 16) & 1u);
    return (uint16_t) (x >> 16);
}
float bf16_float(uint16_t v) {
    const uint32_t x = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}
// how far a q8_1 block's dequantized values are beyond the quantizer's own bound: the code is round(v / d32)
// with d32 = amax / 127 in f32, the stored scale d16 is d32 in fp16, so |q d16 - v| <= d32 / 2 + |q| |d16 - d32|
double q81_excess(const glmd::Q81& b, const float* v) {
    float amax = 0.0f;
    for (int l = 0; l < 32; ++l) amax = std::max(amax, std::fabs(v[l]));
    const double d32 = std::min(amax / 127.0f, 65504.0f), d16 = glmd::twin::half_float(b.d);
    double e = 0.0;
    for (int l = 0; l < 32; ++l)
        e = std::max(e, std::fabs(b.qs[l] * d16 - v[l]) - 0.5 * d32 - std::abs(b.qs[l]) * std::fabs(d16 - d32) -
                            1e-6 * std::fabs(v[l]));
    return e;
}
// half an fp16 ulp at |v| (subnormal spacing below 2^-14)
double half_ulp16(double v) {
    const double a = std::fabs(v);
    if (a < std::ldexp(1.0, -14)) return std::ldexp(1.0, -25);
    return std::ldexp(1.0, (int) std::floor(std::log2(a)) - 11);
}

// ---------------------------------------------------------------- fp16
void test_fp16() {
    using glmd::twin::half_bits;
    using glmd::twin::half_float;
    int bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        if (((h >> 10) & 0x1fu) == 31u && (h & 0x3ffu)) continue;   // NaN
        bad += half_bits(half_float((uint16_t) h)) != h;
    }
    expect(bad == 0, "fp16: every non-NaN half round-trips");
    expect(half_bits(1.0f) == 0x3c00 && half_bits(65504.0f) == 0x7bff && half_bits(65519.0f) == 0x7bff &&
               half_bits(65520.0f) == 0x7c00 && half_bits(std::ldexp(1.0f, -24)) == 0x0001 &&
               half_bits(std::ldexp(1.0f, -26)) == 0x0000 && half_bits(std::ldexp(3.0f, -26)) == 0x0001 &&
               half_bits(-2.0f) == 0xc000,
           "fp16: known encodings (max, overflow, subnormal rounding)");
    // ties: 1 + 2^-11 rounds to 1 (even), 1 + 3 * 2^-11 rounds up to 1 + 2^-9
    expect(half_bits(1.0f + std::ldexp(1.0f, -11)) == 0x3c00 && half_bits(1.0f + 3 * std::ldexp(1.0f, -11)) == 0x3c02,
           "fp16: round half to even");
}

// ---------------------------------------------------------------- top-k
// the oracle: sort by (score descending, cell ascending), the first k, in cell order
std::vector<int> oracle_topk(const float* sc, int n, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (n <= k) return idx;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [sc](int a, int b) {
        return sc[a] > sc[b] || (sc[a] == sc[b] && a < b);
    });
    idx.resize(k);
    std::sort(idx.begin(), idx.end());
    return idx;
}

void fill_row(std::mt19937& rng, float* sc, int n, int kind) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> small(0, 15);
    std::uniform_int_distribution<int> coin(0, 99);
    for (int i = 0; i < n; ++i) {
        switch (kind) {
            case 0: sc[i] = nd(rng); break;                                // distinct (almost surely)
            case 1: sc[i] = (float) small(rng); break;                     // heavy ties at the cut
            case 2: sc[i] = coin(rng) < 50 ? 0.0f : (coin(rng) < 50 ? -0.0f : (float) small(rng) - 8.0f); break;
            case 3: sc[i] = coin(rng) < 30 ? -std::numeric_limits<float>::infinity() : nd(rng); break;
            default: sc[i] = std::max(0.0f, nd(rng)) * 1e-3f; break;       // indexer-like: many exact zeros
        }
    }
}

void test_topk() {
    const int K = glmd::kTopK;
    std::mt19937 rng(1234);
    struct Case { int n, rows; };
    const Case cases[] = {{2049, 4500}, {8192, 4500}, {131072, 1000}};
    const auto t0 = std::chrono::steady_clock::now();
    int total = 0;
    for (const Case& c : cases) {
        std::vector<float> sc(c.n);
        std::vector<int> sel(K);
        int bad = 0;
        for (int r = 0; r < c.rows; ++r, ++total) {
            fill_row(rng, sc.data(), c.n, r % 5);
            glmd::twin::topk(sc.data(), c.n, c.n, 1, nullptr, K, sel.data());
            const std::vector<int> want = oracle_topk(sc.data(), c.n, K);
            if (!std::equal(want.begin(), want.end(), sel.begin())) {
                if (bad < 3) std::printf("  top-k mismatch: n %d row %d kind %d\n", c.n, r, r % 5);
                ++bad;
            }
        }
        char what[96];
        std::snprintf(what, sizeof what, "top-k: n %d, %d rows equal the oracle (position order)", c.n, c.rows);
        expect(bad == 0, what);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  (%d rows in %.1f s)\n", total, s);
    // identity while p + 1 <= 2048, and per-row n from positions (the decode and prefill forms)
    {
        std::vector<float> sc(4096);
        fill_row(rng, sc.data(), 4096, 0);
        std::vector<int> sel(K);
        bool ok = true;
        for (int n : {1, 2, 1000, 2047, 2048}) {
            glmd::twin::topk(sc.data(), 4096, n, 1, nullptr, K, sel.data());
            for (int i = 0; i < K; ++i) ok &= sel[i] == (i < n ? i : -1);
        }
        expect(ok, "top-k: identity (cells 0..n-1, then -1) for n <= 2048");
        const int T = 5;
        const int pos[T] = {0, 2047, 2048, 3000, 9999};
        std::vector<float> rows((size_t) T * 4096);
        for (int t = 0; t < T; ++t) fill_row(rng, rows.data() + (size_t) t * 4096, 4096, t % 5);
        std::vector<int> selT((size_t) T * K);
        glmd::twin::topk(rows.data(), 4096, 4096, T, pos, K, selT.data());
        ok = true;
        for (int t = 0; t < T; ++t) {
            const int n = std::min(4096, pos[t] + 1);
            std::vector<int> want = oracle_topk(rows.data() + (size_t) t * 4096, n, K);
            want.resize(K, -1);
            ok &= std::equal(want.begin(), want.end(), selT.begin() + (size_t) t * K);
        }
        expect(ok, "top-k: per-row n = min(n_kv, pos + 1) over 5 rows");
    }
}

// ---------------------------------------------------------------- prep and q-side rope
constexpr int kQLora = 2048, kKvLora = 512, kRope = 64, kIdxDim = 128, kIdxRope = 64, kIdxHeads = 32;
constexpr int kHeads = 64, kNope = 192, kVHead = 256;
constexpr double kTheta = 8e6;

void test_prep() {
    std::mt19937 rng(77);
    std::vector<float> inv(kRope / 2);
    glmd::rope_inv_freq(inv.data(), kRope, kTheta);
    const int max_p = 32767;
    std::vector<uint16_t> lat16((size_t) (max_p + 1) * kKvLora), k_pe((size_t) (max_p + 1) * kRope);
    std::vector<uint8_t> lat8((size_t) (max_p + 1) * glmd::lat8_rec_bytes(kKvLora));
    std::vector<float> ik_cache((size_t) (max_p + 1) * kIdxDim);
    double e_qr = 0, e_qq = 0, e_lat = 0, e_lat8 = 0, e_ik = 0;
    bool kpe_exact = true, rope_q_exact = true, lat_bits = true;
    for (int p : {0, 1, 7, 2047, 2048, max_p}) {
        const std::vector<float> qr_raw = randv(rng, kQLora, 2.0f), q_a_norm = randv(rng, kQLora, 0.1f, 1.0f);
        const std::vector<float> kv_raw = randv(rng, kKvLora + kRope, 3.0f), kv_norm = randv(rng, kKvLora, 0.1f, 1.0f);
        const std::vector<float> ik_raw = randv(rng, kIdxDim, 1.5f, 0.3f), ln_w = randv(rng, kIdxDim, 0.2f, 1.0f),
                                 ln_b = randv(rng, kIdxDim, 0.1f);
        std::vector<float> qr(kQLora);
        std::vector<glmd::Q81> qr_q(kQLora / 32);
        glmd::DsaPrep a;
        a.qr_raw = qr_raw.data(); a.q_a_norm = q_a_norm.data(); a.qr = qr.data(); a.qr_q = qr_q.data();
        a.q_lora = kQLora;
        a.kv_raw = kv_raw.data(); a.kv_norm = kv_norm.data(); a.lat = lat16.data(); a.kv_lora = kKvLora;
        a.k_pe = k_pe.data(); a.qk_rope = kRope; a.inv_freq = inv.data();
        a.ik_raw = ik_raw.data(); a.ik_ln_w = ln_w.data(); a.ik_ln_b = ln_b.data(); a.idx_ln_eps = 1e-6f;
        a.ik_cache = ik_cache.data(); a.idx_dim = kIdxDim; a.idx_rope = kIdxRope; a.idx_inv_freq = inv.data();
        a.p = p;
        a.eps = 1e-5f;
        glmd::twin::dsa_prep(a);
        glmd::DsaPrep a8 = a;
        a8.lat = lat8.data();
        a8.lat_q8 = 1;
        a8.ik_raw = nullptr;   // a shared layer: no index key
        glmd::twin::dsa_prep(a8);

        // the reference: double RMS (llm_build_norm RMS: x / sqrt(mean(x^2) + eps) * w)
        auto rms_ref = [](const float* x, const float* w, int n, double eps, std::vector<double>& y) {
            double ss = 0;
            for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
            const double inv_r = 1.0 / std::sqrt(ss / n + eps);
            y.resize(n);
            for (int i = 0; i < n; ++i) y[i] = x[i] * inv_r * w[i];
        };
        std::vector<double> yq, ykv;
        rms_ref(qr_raw.data(), q_a_norm.data(), kQLora, 1e-5, yq);
        rms_ref(kv_raw.data(), kv_norm.data(), kKvLora, 1e-5, ykv);
        for (int i = 0; i < kQLora; ++i) e_qr = std::max(e_qr, std::fabs(qr[i] - yq[i]));
        for (int b = 0; b < kQLora / 32; ++b) e_qq = std::max(e_qq, q81_excess(qr_q[b], qr.data() + 32 * b));
        for (int c = 0; c < kKvLora; ++c) {
            const uint16_t h = lat16[(size_t) kKvLora * p + c];
            e_lat = std::max(e_lat, std::fabs(glmd::twin::half_float(h) - ykv[c]) - half_ulp16(ykv[c]));
            lat_bits &= h == glmd::twin::half_bits((float) ykv[c]) ||
                        std::fabs(glmd::twin::half_float(h) - ykv[c]) <= 2 * half_ulp16(ykv[c]);
            const uint8_t* rec = lat8.data() + (size_t) glmd::lat8_rec_bytes(kKvLora) * p;
            uint16_t d16;
            std::memcpy(&d16, rec + kKvLora + 2 * (c / 32), 2);
            const double d = glmd::twin::half_float(d16);
            e_lat8 = std::max(e_lat8, std::fabs((int8_t) rec[c] * d - ykv[c]) - 0.5 * d);
        }
        // k_pe: the reference rotation of the f32 tail, then fp16
        std::vector<float> kp(kv_raw.begin() + kKvLora, kv_raw.end());
        glmd::cpu::rope_norm(kp.data(), kRope, 1, 1, 0, kRope, &p, inv.data());
        for (int j = 0; j < kRope; ++j) kpe_exact &= k_pe[(size_t) kRope * p + j] == glmd::twin::half_bits(kp[j]);
        // the index key: LayerNorm (double) + rope
        std::vector<float> ik(ik_raw);
        glmd::cpu::layer_norm(ik.data(), ln_w.data(), ln_b.data(), 1e-6f, kIdxDim, 1);
        glmd::cpu::rope_norm(ik.data(), kIdxDim, 1, 1, 0, kIdxRope, &p, inv.data());
        for (int i = 0; i < kIdxDim; ++i) e_ik = std::max(e_ik, (double) std::fabs(ik_cache[(size_t) kIdxDim * p + i] - ik[i]));

        // q side: MLA q_pe (dims 192..255 of 64 x 256) and the indexer query (dims 0..63 of 32 x 128)
        std::vector<float> q = randv(rng, (size_t) kHeads * (kNope + kRope)), iq = randv(rng, (size_t) kIdxHeads * kIdxDim);
        std::vector<float> q_ref(q), iq_ref(iq);
        glmd::DsaRopeQ r;
        r.q = q.data(); r.n_head = kHeads; r.q_stride = kNope + kRope; r.rot_off = kNope; r.n_rot = kRope;
        r.inv_freq = inv.data(); r.iq = iq.data(); r.idx_heads = kIdxHeads; r.idx_dim = kIdxDim; r.idx_rope = kIdxRope;
        r.idx_inv_freq = inv.data(); r.p = p;
        glmd::twin::dsa_rope_q(r);
        glmd::cpu::rope_norm(q_ref.data(), kNope + kRope, kHeads, 1, kNope, kRope, &p, inv.data());
        glmd::cpu::rope_norm(iq_ref.data(), kIdxDim, kIdxHeads, 1, 0, kIdxRope, &p, inv.data());
        rope_q_exact &= q == q_ref && iq == iq_ref;
    }
    gate("prep: q_a RMS f32 vs reference, max abs err", e_qr, 1e-5);
    gate("prep: q_a q8_1 beyond the quantizer bound, max abs err", std::max(e_qq, 0.0), 1e-5);
    gate("prep: latent FP16 beyond half an ulp, max abs err", std::max(e_lat, 0.0), 1e-5);
    gate("prep: latent INT8 record beyond half a step, max abs err", std::max(e_lat8, 0.0), 1e-5);
    gate("prep: index key LayerNorm + rope vs reference, max abs err", e_ik, 1e-5);
    expect(lat_bits, "prep: latent FP16 is the rounding of the reference (within 1 ulp)");
    expect(kpe_exact, "prep: k_pe = fp16(reference NORM rope of kv_a[512..575]) exactly");
    expect(rope_q_exact, "rope_q: q_pe and indexer query equal the reference rotation exactly");

    // a NEOX rotation (Maya's rope.hpp pairing) must not pass for the k_pe tail
    {
        std::vector<float> kv_raw = randv(rng, kKvLora + kRope, 3.0f), kv_norm(kKvLora, 1.0f), qr_raw(kQLora, 1.0f),
                           qn(kQLora, 1.0f);
        glmd::DsaPrep a;
        a.qr_raw = qr_raw.data(); a.q_a_norm = qn.data(); a.q_lora = kQLora;
        a.kv_raw = kv_raw.data(); a.kv_norm = kv_norm.data(); a.lat = lat16.data(); a.kv_lora = kKvLora;
        a.k_pe = k_pe.data(); a.qk_rope = kRope; a.inv_freq = inv.data(); a.p = 1000;
        glmd::twin::dsa_prep(a);
        std::vector<float> neox(kv_raw.begin() + kKvLora, kv_raw.end());
        for (int i = 0; i < kRope / 2; ++i) {
            float c, s;
            glmd::rope_cs(1000, inv[i], c, s);
            glmd::rope_pair(neox[i], neox[i + kRope / 2], c, s);
        }
        double e = 0;
        for (int j = 0; j < kRope; ++j)
            e = std::max(e, (double) std::fabs(glmd::twin::half_float(k_pe[(size_t) kRope * 1000 + j]) - neox[j]));
        expect(e > 1e-2, "prep: a NEOX reading of k_pe is detected (wrong pairing differs)");
    }
}

// ---------------------------------------------------------------- mla2
double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

void test_mla2() {
    std::mt19937 rng(4242);
    const int n_cache = 4096, q_dim = kKvLora + kRope, q_stride = kNope + kRope;
    // weights in BF16 (as the engine stores attn_k_b / attn_v_b), the reference reads the same values in f32
    std::vector<uint16_t> wk((size_t) kHeads * kKvLora * kNope), wv((size_t) kHeads * kVHead * kKvLora);
    std::vector<float> wk_f(wk.size()), wv_f(wv.size());
    {
        std::normal_distribution<float> d(0.0f, 0.05f);
        for (size_t i = 0; i < wk.size(); ++i) wk_f[i] = bf16_float(wk[i] = bf16_bits(d(rng)));
        for (size_t i = 0; i < wv.size(); ++i) wv_f[i] = bf16_float(wv[i] = bf16_bits(d(rng)));
    }
    // the cache: FP16 latents and the INT8 records of the same rows, FP16 k_pe
    std::vector<uint16_t> lat16((size_t) n_cache * kKvLora), k_pe((size_t) n_cache * kRope);
    std::vector<uint8_t> lat8((size_t) n_cache * glmd::lat8_rec_bytes(kKvLora));
    {
        std::vector<float> inv(kRope / 2), one(kKvLora, 1.0f), qr(kQLora, 1.0f);
        glmd::rope_inv_freq(inv.data(), kRope, kTheta);
        for (int p = 0; p < n_cache; ++p) {
            std::vector<float> kv = randv(rng, q_dim, 1.0f);
            glmd::DsaPrep a;
            a.qr_raw = qr.data(); a.q_a_norm = qr.data(); a.q_lora = kQLora;
            a.kv_raw = kv.data(); a.kv_norm = one.data(); a.lat = lat16.data(); a.kv_lora = kKvLora;
            a.k_pe = k_pe.data(); a.qk_rope = kRope; a.inv_freq = inv.data(); a.p = p;
            glmd::twin::dsa_prep(a);
            a.lat = lat8.data();
            a.lat_q8 = 1;
            glmd::twin::dsa_prep(a);
        }
    }
    // the values the kernel attends to, as reference cache rows [c_kv | k_pe]
    auto kv_rows = [&](bool q8) {
        std::vector<float> kv((size_t) n_cache * q_dim);
        const int rb = glmd::lat8_rec_bytes(kKvLora);
        for (int s = 0; s < n_cache; ++s) {
            for (int c = 0; c < kKvLora; ++c) {
                float v;
                if (q8) {
                    const uint8_t* rec = lat8.data() + (size_t) rb * s;
                    uint16_t d16;
                    std::memcpy(&d16, rec + kKvLora + 2 * (c / 32), 2);
                    // staged in FP16 shared memory by the split kernel
                    v = glmd::twin::half_float(glmd::twin::half_bits((int8_t) rec[c] * glmd::twin::half_float(d16)));
                } else {
                    v = glmd::twin::half_float(lat16[(size_t) kKvLora * s + c]);
                }
                kv[(size_t) s * q_dim + c] = v;
            }
            for (int j = 0; j < kRope; ++j) kv[(size_t) s * q_dim + kKvLora + j] = glmd::twin::half_float(k_pe[(size_t) kRope * s + j]);
        }
        return kv;
    };
    const std::vector<float> kv16 = kv_rows(false), kv8 = kv_rows(true);
    const std::vector<float> q = randv(rng, (size_t) kHeads * q_stride);
    std::vector<float> Q((size_t) kHeads * q_dim);
    glmd::cpu::mla_absorb_q(q.data(), wk_f.data(), kHeads, kNope, kRope, kKvLora, 1, Q.data());

    for (int n_sel : {0, 1, 37, 2048}) {
        // distinct random cells; at 37 every fourth slot is padding
        std::vector<int> perm(n_cache);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<int> cells(perm.begin(), perm.begin() + n_sel);
        if (n_sel == 37)
            for (int i = 0; i < n_sel; i += 4) cells[i] = -1;
        for (int q8 = 0; q8 < 2; ++q8) {
            std::vector<float> out((size_t) kHeads * kVHead), ref(out.size());
            std::vector<glmd::Q81> oq(out.size() / 32);
            glmd::Mla2 m;
            m.q = q.data(); m.q_stride = q_stride; m.wk_b = wk.data(); m.wv_b = wv.data();
            m.lat = q8 ? (const void*) lat8.data() : (const void*) lat16.data(); m.lat_q8 = q8; m.k_pe = k_pe.data();
            m.cells = cells.data(); m.n_sel = n_sel; m.n_head = kHeads; m.qk_nope = kNope; m.qk_rope = kRope;
            m.kv_lora = kKvLora; m.v_head = kVHead; m.scale = glmd::kGlm53KqScale; m.out = out.data();
            m.out_q8_1 = oq.data();
            glmd::twin::mla2(m);
            glmd::cpu::mla_attention(Q.data(), q8 ? kv8.data() : kv16.data(), cells.data(), wv_f.data(), kHeads,
                                     kKvLora, kRope, kVHead, 1, n_sel, glmd::kGlm53KqScale, ref.data());
            char what[96];
            std::snprintf(what, sizeof what, "mla2: n_sel %d, %s latent vs reference, rel L2", n_sel, q8 ? "INT8" : "FP16");
            if (n_sel == 0) {
                double mx = 0;
                for (float v : out) mx = std::max(mx, (double) std::fabs(v));
                gate("mla2: n_sel 0 writes zeros, max |out|", mx, 1e-30);
            } else {
                gate(what, rel_l2(out, ref), 1e-4);
            }
            double eq = 0;
            for (size_t b = 0; b < oq.size(); ++b) eq = std::max(eq, q81_excess(oq[b], out.data() + 32 * b));
            if (n_sel == 2048) gate("mla2: q8_1 output beyond the quantizer bound", std::max(eq, 0.0), 1e-5);
            // the Flash rule 1/sqrt(qk_nope) is a wrong reading for GLM-5.3: it must fail the gate
            if (n_sel == 2048 && !q8) {
                m.scale = 1.0f / std::sqrt((float) kNope);
                glmd::twin::mla2(m);
                expect(rel_l2(out, ref) > 1e-3, "mla2: the Flash scale 1/sqrt(192) is detected");
            }
        }
    }
}

}  // namespace

int main() {
    test_fp16();
    test_prep();
    test_mla2();
    test_topk();
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
