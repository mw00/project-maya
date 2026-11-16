// tests/glm_dsa/glmd_fast_parity.cu - the glm-dsa decode attention kernels (B4.3-B4.5) vs the F32 reference kernels
// (glmd::ref_*) and their CPU twins (glmd::twin), GLM-5.3 sizes.
//   prep     positions {0, 1, 2047, 2048, 40000}: the latent (FP16 and INT8 records) and k_pe equal the twin's
//            bytes within one fp16 ulp / one INT8 step; k_pe and the index key vs ref_rope_norm / ref_indexer_key
//            (identity W_idx_k): max abs err < 1e-5 beyond the fp16 rounding; qr vs the twin < 1e-5;
//   rope_q   q_pe and the indexer query vs ref_rope_norm: max abs err < 2e-6;
//   mla2     vs ref_mla_absorb_q + ref_mla_attention over the stored (dequantized) cache at scale 1/16: rel L2 < 1e-4
//            at n_sel 1 and 2048, FP16 and INT8 latents; also vs the twin; a refused shape is counted;
//   top-k    the GPU selection equals the twin's exactly (same algorithm) on 300 rows of n 2049 / 8192 / 131072
//            and a per-row-position batch, and equals ref_topk (sorted to position order) at n 4096.
// Needs a GPU; exits 77 (ctest SKIP_RETURN_CODE) when there is none.
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_ref.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

int g_fail = 0;

#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(const std::vector<T>& h) : n(h.size()) {
        CK(cudaMalloc(&p, n * sizeof(T)));
        CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    explicit Dev(size_t count) : n(count) {
        CK(cudaMalloc(&p, n * sizeof(T)));
        CK(cudaMemset(p, 0, n * sizeof(T)));
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<T> get() const {
        std::vector<T> h(n);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

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
uint16_t bf16_bits(float f) {
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
double half_ulp16(double v) {
    const double a = std::fabs(v);
    if (a < std::ldexp(1.0, -14)) return std::ldexp(1.0, -25);
    return std::ldexp(1.0, (int) std::floor(std::log2(a)) - 11);
}
double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

constexpr int kQLora = 2048, kKvLora = 512, kRope = 64, kIdxDim = 128, kIdxRope = 64, kIdxHeads = 32;
constexpr int kHeads = 64, kNope = 192, kVHead = 256;

void test_prep_rope(std::mt19937& rng, const float* d_inv, const std::vector<float>& inv) {
    const int max_p = 40000, rb = glmd::lat8_rec_bytes(kKvLora);
    Dev<uint16_t> d_lat16((size_t) (max_p + 1) * kKvLora), d_kpe((size_t) (max_p + 1) * kRope);
    Dev<uint8_t> d_lat8((size_t) (max_p + 1) * rb);
    Dev<float> d_ik((size_t) (max_p + 1) * kIdxDim), d_qr(kQLora);
    Dev<glmd::Q81> d_qq(kQLora / 32);
    std::vector<float> eye((size_t) kIdxDim * kIdxDim, 0.0f);
    for (int i = 0; i < kIdxDim; ++i) eye[(size_t) i * kIdxDim + i] = 1.0f;
    Dev<float> d_eye(eye);
    double e_qr = 0, e_lat = 0, e_kpe = 0, e_ik = 0, e_rq = 0;
    bool lat_close = true, lat8_close = true, qq_close = true;
    const int e0 = glmd::launch_errors();
    for (int p : {0, 1, 2047, 2048, max_p}) {
        const auto qr_raw = randv(rng, kQLora, 2.0f), qn = randv(rng, kQLora, 0.1f, 1.0f);
        const auto kv_raw = randv(rng, kKvLora + kRope, 3.0f), kvn = randv(rng, kKvLora, 0.1f, 1.0f);
        const auto ik_raw = randv(rng, kIdxDim, 1.5f, 0.3f), lnw = randv(rng, kIdxDim, 0.2f, 1.0f),
                   lnb = randv(rng, kIdxDim, 0.1f);
        Dev<float> d_qr_raw(qr_raw), d_qn(qn), d_kv_raw(kv_raw), d_kvn(kvn), d_ik_raw(ik_raw), d_lnw(lnw), d_lnb(lnb);
        glmd::DsaPrep a;
        a.qr_raw = d_qr_raw.p; a.q_a_norm = d_qn.p; a.qr = d_qr.p; a.qr_q = d_qq.p; a.q_lora = kQLora;
        a.kv_raw = d_kv_raw.p; a.kv_norm = d_kvn.p; a.lat = d_lat16.p; a.kv_lora = kKvLora;
        a.k_pe = d_kpe.p; a.qk_rope = kRope; a.inv_freq = d_inv;
        a.ik_raw = d_ik_raw.p; a.ik_ln_w = d_lnw.p; a.ik_ln_b = d_lnb.p; a.idx_ln_eps = 1e-6f; a.ik_cache = d_ik.p;
        a.idx_dim = kIdxDim; a.idx_rope = kIdxRope; a.idx_inv_freq = d_inv; a.p = p; a.eps = 1e-5f;
        expect(glmd::dsa_prep(a, nullptr), "prep: launch (full layer, FP16 latent)");
        glmd::DsaPrep a8 = a;
        a8.lat = d_lat8.p;
        a8.lat_q8 = 1;
        a8.ik_raw = nullptr;
        Dev<int> d_p(std::vector<int>{p});
        a8.p = -1;           // the device position wins
        a8.p_dev = d_p.p;
        expect(glmd::dsa_prep(a8, nullptr), "prep: launch (shared layer, INT8 latent, device position)");

        // the twin on host copies
        std::vector<uint16_t> t_lat((size_t) kKvLora), t_kpe(kRope);
        std::vector<uint8_t> t_lat8(rb);
        std::vector<float> t_ik(kIdxDim), t_qr(kQLora);
        std::vector<glmd::Q81> t_qq(kQLora / 32);
        glmd::DsaPrep h = a;
        h.qr_raw = qr_raw.data(); h.q_a_norm = qn.data(); h.qr = t_qr.data(); h.qr_q = t_qq.data();
        h.kv_raw = kv_raw.data(); h.kv_norm = kvn.data(); h.lat = t_lat.data(); h.k_pe = t_kpe.data();
        h.inv_freq = inv.data(); h.ik_raw = ik_raw.data(); h.ik_ln_w = lnw.data(); h.ik_ln_b = lnb.data();
        h.ik_cache = t_ik.data(); h.idx_inv_freq = inv.data(); h.p = 0;   // twin outputs at row 0
        glmd::twin::dsa_prep(h);
        h.lat = t_lat8.data();
        h.lat_q8 = 1;
        h.ik_raw = nullptr;
        glmd::twin::dsa_prep(h);

        const auto g_lat = d_lat16.get(), g_kpe = d_kpe.get();
        const auto g_lat8 = d_lat8.get();
        const auto g_ik = d_ik.get(), g_qr = d_qr.get();
        const auto g_qq = d_qq.get();
        for (int i = 0; i < kQLora; ++i) e_qr = std::max(e_qr, (double) std::fabs(g_qr[i] - t_qr[i]));
        for (int b = 0; b < kQLora / 32; ++b)
            for (int l = 0; l < 32; ++l) qq_close &= std::abs(g_qq[b].qs[l] - t_qq[b].qs[l]) <= 1;
        for (int c = 0; c < kKvLora; ++c) {
            const float g = glmd::twin::half_float(g_lat[(size_t) kKvLora * p + c]);
            const float t = glmd::twin::half_float(t_lat[c]);
            lat_close &= std::fabs(g - t) <= 2 * half_ulp16(t);
            e_lat = std::max(e_lat, std::fabs(g - t) - 2 * half_ulp16(t));
            lat8_close &= std::abs((int) (int8_t) g_lat8[(size_t) rb * p + c] - (int) (int8_t) t_lat8[c]) <= 1;
        }
        // k_pe and the index key against the GPU reference kernels at the same position
        Dev<int> d_pos(std::vector<int>{p});
        Dev<float> d_ref_kv(kv_raw);
        glmd::ref_rope_norm(d_ref_kv.p, kKvLora + kRope, 1, 1, kKvLora, kRope, d_pos.p, d_inv, nullptr);
        const auto r_kv = d_ref_kv.get();
        for (int j = 0; j < kRope; ++j) {
            const double r = r_kv[kKvLora + j];
            e_kpe = std::max(e_kpe, std::fabs(glmd::twin::half_float(g_kpe[(size_t) kRope * p + j]) - r) - half_ulp16(r));
        }
        Dev<float> d_ref_ik(kIdxDim);
        glmd::ref_indexer_key(d_ik_raw.p, d_eye.p, d_lnw.p, d_lnb.p, 1e-6f, kIdxDim, kIdxDim, 1, kIdxRope, d_pos.p, d_inv,
                              d_ref_ik.p, nullptr);
        const auto r_ik = d_ref_ik.get();
        for (int i = 0; i < kIdxDim; ++i) e_ik = std::max(e_ik, (double) std::fabs(g_ik[(size_t) kIdxDim * p + i] - r_ik[i]));

        // q side
        const auto q = randv(rng, (size_t) kHeads * (kNope + kRope)), iq = randv(rng, (size_t) kIdxHeads * kIdxDim);
        Dev<float> d_q(q), d_iq(iq), d_qref(q), d_iqref(iq);
        glmd::DsaRopeQ r;
        r.q = d_q.p; r.n_head = kHeads; r.q_stride = kNope + kRope; r.rot_off = kNope; r.n_rot = kRope;
        r.inv_freq = d_inv; r.iq = d_iq.p; r.idx_heads = kIdxHeads; r.idx_dim = kIdxDim; r.idx_rope = kIdxRope;
        r.idx_inv_freq = d_inv; r.p = p;
        expect(glmd::dsa_rope_q(r, nullptr), "rope_q: launch");
        glmd::ref_rope_norm(d_qref.p, kNope + kRope, kHeads, 1, kNope, kRope, d_pos.p, d_inv, nullptr);
        glmd::ref_rope_norm(d_iqref.p, kIdxDim, kIdxHeads, 1, 0, kIdxRope, d_pos.p, d_inv, nullptr);
        const auto gq = d_q.get(), rq = d_qref.get(), giq = d_iq.get(), riq = d_iqref.get();
        for (size_t i = 0; i < gq.size(); ++i) e_rq = std::max(e_rq, (double) std::fabs(gq[i] - rq[i]));
        for (size_t i = 0; i < giq.size(); ++i) e_rq = std::max(e_rq, (double) std::fabs(giq[i] - riq[i]));
    }
    gate("prep: qr GPU vs twin, max abs err", e_qr, 1e-5);
    expect(qq_close, "prep: qr q8_1 codes GPU vs twin within 1");
    expect(lat_close, "prep: FP16 latent GPU vs twin within 1 ulp");
    gate("prep: FP16 latent beyond 1 ulp", std::max(e_lat, 0.0), 1e-5);
    expect(lat8_close, "prep: INT8 latent codes GPU vs twin within 1");
    gate("prep: k_pe vs ref_rope_norm beyond fp16 rounding", std::max(e_kpe, 0.0), 1e-5);
    gate("prep: index key vs ref_indexer_key, max abs err", e_ik, 1e-5);
    gate("rope_q: q_pe and indexer query vs ref_rope_norm", e_rq, 2e-6);
    expect(glmd::launch_errors() == e0, "prep / rope_q: no launch error");
}

void test_mla2(std::mt19937& rng) {
    const int n_cache = 4096, q_dim = kKvLora + kRope, q_stride = kNope + kRope, rb = glmd::lat8_rec_bytes(kKvLora);
    std::vector<uint16_t> wk((size_t) kHeads * kKvLora * kNope), wv((size_t) kHeads * kVHead * kKvLora);
    std::vector<float> wk_f(wk.size()), wv_f(wv.size());
    {
        std::normal_distribution<float> d(0.0f, 0.05f);
        for (size_t i = 0; i < wk.size(); ++i) wk_f[i] = bf16_float(wk[i] = bf16_bits(d(rng)));
        for (size_t i = 0; i < wv.size(); ++i) wv_f[i] = bf16_float(wv[i] = bf16_bits(d(rng)));
    }
    // a random cache in the stored formats
    std::vector<uint16_t> lat16((size_t) n_cache * kKvLora), k_pe((size_t) n_cache * kRope);
    std::vector<uint8_t> lat8((size_t) n_cache * rb);
    for (auto& v : lat16) v = glmd::twin::half_bits(std::normal_distribution<float>(0.0f, 1.0f)(rng));
    for (auto& v : k_pe) v = glmd::twin::half_bits(std::normal_distribution<float>(0.0f, 1.0f)(rng));
    for (int s = 0; s < n_cache; ++s) {
        for (int c = 0; c < kKvLora; ++c) lat8[(size_t) rb * s + c] = (uint8_t) (int8_t) (int) (rng() % 255 - 127);
        for (int g = 0; g < kKvLora / 32; ++g) {
            const uint16_t d16 = glmd::twin::half_bits(0.01f + 0.001f * (float) (rng() % 16));
            std::memcpy(&lat8[(size_t) rb * s + kKvLora + 2 * g], &d16, 2);
        }
    }
    // reference cache rows: the values the kernel attends to (INT8 decoded, then staged in fp16)
    auto kv_rows = [&](bool q8) {
        std::vector<float> kv((size_t) n_cache * q_dim);
        for (int s = 0; s < n_cache; ++s) {
            for (int c = 0; c < kKvLora; ++c) {
                float v;
                if (q8) {
                    uint16_t d16;
                    std::memcpy(&d16, &lat8[(size_t) rb * s + kKvLora + 2 * (c / 32)], 2);
                    v = glmd::twin::half_float(
                        glmd::twin::half_bits((int8_t) lat8[(size_t) rb * s + c] * glmd::twin::half_float(d16)));
                } else {
                    v = glmd::twin::half_float(lat16[(size_t) kKvLora * s + c]);
                }
                kv[(size_t) s * q_dim + c] = v;
            }
            for (int j = 0; j < kRope; ++j)
                kv[(size_t) s * q_dim + kKvLora + j] = glmd::twin::half_float(k_pe[(size_t) kRope * s + j]);
        }
        return kv;
    };
    const auto q = randv(rng, (size_t) kHeads * q_stride);
    Dev<float> d_q(q), d_wkf(wk_f), d_wvf(wv_f), d_Q((size_t) kHeads * q_dim);
    Dev<uint16_t> d_wk(wk), d_wv(wv), d_lat16(lat16), d_kpe(k_pe);
    Dev<uint8_t> d_lat8(lat8);
    glmd::ref_mla_absorb_q(d_q.p, d_wkf.p, kHeads, kNope, kRope, kKvLora, 1, d_Q.p, nullptr);
    const size_t wb = glmd::mla2_work_bytes(kHeads, kKvLora, 2048);
    Dev<uint8_t> d_work(wb);
    const int e0 = glmd::launch_errors();
    for (int q8 = 0; q8 < 2; ++q8) {
        const auto kv = kv_rows(q8);
        Dev<float> d_kv(kv);
        for (int n_sel : {1, 2048}) {
            std::vector<int> perm(n_cache);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            std::vector<int> cells(perm.begin(), perm.begin() + n_sel);
            std::sort(cells.begin(), cells.end());   // position order, as the selector writes it
            Dev<int> d_cells(cells);
            Dev<float> d_out((size_t) kHeads * kVHead), d_ref((size_t) kHeads * kVHead);
            Dev<glmd::Q81> d_oq((size_t) kHeads * kVHead / 32);
            glmd::Mla2 m;
            m.q = d_q.p; m.q_stride = q_stride; m.wk_b = d_wk.p; m.wv_b = d_wv.p;
            m.lat = q8 ? (const void*) d_lat8.p : (const void*) d_lat16.p; m.lat_q8 = q8; m.k_pe = d_kpe.p;
            m.cells = d_cells.p; m.n_sel = n_sel; m.n_head = kHeads; m.qk_nope = kNope; m.qk_rope = kRope;
            m.kv_lora = kKvLora; m.v_head = kVHead; m.scale = glmd::kGlm53KqScale; m.work = d_work.p;
            m.work_bytes = wb; m.out = d_out.p; m.out_q8_1 = d_oq.p;
            expect(glmd::mla2(m, nullptr), "mla2: launch");
            glmd::ref_mla_attention(d_Q.p, d_kv.p, d_cells.p, d_wvf.p, kHeads, kKvLora, kRope, kVHead, 1, n_sel,
                                    glmd::kGlm53KqScale, d_ref.p, nullptr);
            const auto out = d_out.get(), ref = d_ref.get();
            char what[96];
            std::snprintf(what, sizeof what, "mla2: n_sel %d, %s latent vs ref_mla_*, rel L2", n_sel, q8 ? "INT8" : "FP16");
            gate(what, rel_l2(out, ref), 1e-4);
            // and against the twin
            std::vector<float> t_out(out.size());
            glmd::Mla2 h = m;
            h.q = q.data(); h.wk_b = wk.data(); h.wv_b = wv.data();
            h.lat = q8 ? (const void*) lat8.data() : (const void*) lat16.data(); h.k_pe = k_pe.data();
            h.cells = cells.data(); h.out = t_out.data(); h.out_q8_1 = nullptr;
            glmd::twin::mla2(h);
            std::snprintf(what, sizeof what, "mla2: n_sel %d, %s latent vs twin, rel L2", n_sel, q8 ? "INT8" : "FP16");
            gate(what, rel_l2(out, t_out), 1e-5);
        }
    }
    expect(glmd::launch_errors() == e0, "mla2: no launch error");
    {
        glmd::Mla2 bad;
        bad.qk_rope = 32;   // only 0 and 64 are instantiated
        expect(!glmd::mla2(bad, nullptr) && glmd::launch_errors() == e0 + 1, "mla2: a refused shape is counted");
    }
    expect(glmd::mla2_smem_bytes(kKvLora, kRope) == 38912, "mla2: split shared memory 38,912 B (no opt-in)");
}

void test_topk(std::mt19937& rng) {
    const int K = glmd::kTopK;
    const int e0 = glmd::launch_errors();
    for (int n : {2049, 8192, 131072}) {
        const int rows = 100;
        std::vector<float> sc((size_t) rows * n);
        std::uniform_int_distribution<int> small(0, 15);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < n; ++i) sc[(size_t) r * n + i] = r % 2 ? (float) small(rng) : nd(rng);
        Dev<float> d_sc(sc);
        Dev<int> d_sel((size_t) rows * K);
        expect(glmd::topk(d_sc.p, n, n, rows, nullptr, K, d_sel.p, nullptr), "top-k: launch");
        std::vector<int> t_sel((size_t) rows * K);
        glmd::twin::topk(sc.data(), n, n, rows, nullptr, K, t_sel.data());
        char what[96];
        std::snprintf(what, sizeof what, "top-k: n %d, %d rows, GPU equals the twin", n, rows);
        expect(d_sel.get() == t_sel, what);
    }
    {   // per-row n from positions, decode and identity rows mixed
        const int n_kv = 9000;
        const std::vector<int> pos = {0, 100, 2047, 2048, 2049, 5000, 8999};
        const int T = (int) pos.size();
        const auto sc = randv(rng, (size_t) T * n_kv);
        Dev<float> d_sc(sc);
        Dev<int> d_pos(pos), d_sel((size_t) T * K);
        glmd::topk(d_sc.p, n_kv, n_kv, T, d_pos.p, K, d_sel.p, nullptr);
        std::vector<int> t_sel((size_t) T * K);
        glmd::twin::topk(sc.data(), n_kv, n_kv, T, pos.data(), K, t_sel.data());
        expect(d_sel.get() == t_sel, "top-k: per-row positions, GPU equals the twin");
    }
    {   // vs the O(n^2) reference selector at n 4096 (set equality: the reference ranks, this one sorts by cell)
        const int n = 4096, T = 4;
        const auto sc = randv(rng, (size_t) T * n);
        Dev<float> d_sc(sc);
        Dev<int> d_sel((size_t) T * K), d_ref((size_t) T * K);
        glmd::topk(d_sc.p, n, n, T, nullptr, K, d_sel.p, nullptr);
        glmd::ref_topk(d_sc.p, n, T, K, d_ref.p, nullptr);
        auto g = d_sel.get(), r = d_ref.get();
        for (int t = 0; t < T; ++t) std::sort(r.begin() + (size_t) t * K, r.begin() + (size_t) (t + 1) * K);
        expect(g == r, "top-k: equals ref_topk as a set (n 4096)");
    }
    expect(glmd::launch_errors() == e0, "top-k: no launch error");
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("glmd_fast_parity: no CUDA device, skipped\n");
        return 77;
    }
    std::vector<float> inv(kRope / 2);
    glmd::rope_inv_freq(inv.data(), kRope, 8e6);
    Dev<float> d_inv(inv);
    std::mt19937 rng(531);
    test_prep_rope(rng, d_inv.p, inv);
    test_mla2(rng);
    test_topk(rng);
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
