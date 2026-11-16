// tests/glm_dsa/glmd_volta_cpu_test.cpp - the Volta glm-dsa decode kernels' twins (glm-dsa B10.2-B10.4, B10.7, B10.8)
// on the CPU: the algorithms the kernels run (tiling, split ranges, k-quarter order, online softmax with fp16 P,
// per-block selection) against the B3.4 CPU reference and sort-based oracles.  GLM-5.3 sizes.
//   splits    the split ranges of mla_fused cover [0, n_sel) in whole 32-cell units; shared memory fits sm_70;
//   headwise  BF16 and native Q8_0 W_kb / W_vb rows vs the dequantized rows in double: rel err < 1e-5 (B10.7);
//   fused     vtwin::mla_fused vs cpu::mla_absorb_q + cpu::mla_attention on the stored cache values: rel L2 < 2e-3
//             at n_sel 1, 37 (padding), 2048 for FP16 / INT8 latents, BF16 / Q8_0 weights, splits 1 / 7 / 32 / 64;
//             split invariance < 5e-4 (P is fp16 against each split's running max); large logits (|S * scale| ~ 120) stay finite and accurate (FA rule);
//             n_sel 0 writes zeros; the Flash scale is detected; mla2's Q8_0 path (twin) vs the same reference;
//   scorer    vtwin::idx_score_simple vs cpu::indexer_score: rel err < 1e-5; vtwin::idx_score_wmma (fp16 inputs,
//             f32 products) top-2048 overlap vs the F32 reference >= 99.9% outside near-ties at n 20000 (decode)
//             and for 3 prefill tokens with per-token positions (B10.3); FP16 keys (B10.8) >= 99.5% at 32K;
//   top-k     vtwin::topk_mb equals twin::topk (the single-block selector) and a sort oracle exactly, n up to
//             131072, heavy ties, -inf, per-row positions (B10.4).
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_ref.hpp"
#include "strata/kernels/glmd_volta.hpp"

#include "../../src/core/glm_dsa_fast_plan.hpp"
#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
constexpr int kHeads = 64, kNope = 192, kRope = 64, kKvLora = 512, kVHead = 256, kQLora = 2048;
constexpr double kTheta = 8.0e6;

void expect(bool ok, const char* what) {
    std::printf("%-72s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}
void gate(const char* what, double err, double lim) {
    const bool ok = err < lim;
    std::printf("%-64s %.3e (gate %.1e) %s\n", what, err, lim, ok ? "ok" : "FAIL");
    g_fail += !ok;
}
std::vector<float> randv(std::mt19937& rng, size_t n, float scale = 1.0f) {
    std::normal_distribution<float> d(0.0f, scale);
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
double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

// ---------------------------------------------------------------- splits / shared memory
void test_splits() {
    bool ok = true;
    for (int n_sel : {1, 31, 32, 33, 37, 64, 100, 2047, 2048, 4096})
        for (int req : {1, 3, 7, 32, 64, 1000}) {
            const int ns = glmd::mla_fused_splits(n_sel, req), cps = glmd::mla_fused_split_cells(n_sel, req);
            ok &= cps % glmd::kFusedUnit == 0 && ns >= 1 && ns <= std::max(1, req) && (ns - 1) * cps < n_sel &&
                  ns * cps >= n_sel;
        }
    ok &= glmd::mla_fused_splits(0, 32) == 0;
    ok &= glmd::mla_fused_splits(2048, 32) == 32 && glmd::mla_fused_split_cells(2048, 32) == 64;
    expect(ok, "fused splits: whole 32-cell units cover [0, n_sel)");
    expect(glmd::mla_fused_smem_bytes(64) + 384 <= 96 * 1024, "fused smem (64 x 584 + 16 x 584 FP16 + S) fits 96 KB");
    std::printf("  fused attention dynamic smem: %zu B (rope 64), %zu B (rope 0)\n", glmd::mla_fused_smem_bytes(64),
                glmd::mla_fused_smem_bytes(0));
    expect(glmd::mla_fused_supported(64, 192, 64, 512, 256, glmd::kWTypeBF16) &&
               glmd::mla_fused_supported(64, 192, 64, 512, 256, glmd::kWTypeQ8_0) &&
               glmd::mla_fused_supported(16, 192, 0, 512, 256, glmd::kWTypeBF16) &&
               !glmd::mla_fused_supported(65, 192, 64, 512, 256, glmd::kWTypeBF16) &&
               !glmd::mla_fused_supported(64, 192, 64, 256, 256, glmd::kWTypeBF16) &&
               !glmd::mla_fused_supported(64, 200, 64, 512, 256, glmd::kWTypeQ8_0),
           "fused supported shapes");
    expect(glmd::head_row_bytes(glmd::kWTypeQ8_0, 192) == 204 && glmd::head_row_bytes(glmd::kWTypeBF16, 192) == 384 &&
               glmd::head_row_bytes(glmd::kWTypeQ8_0, 200) == 0,
           "head_row_bytes: Q8_0 192 -> 204 B, BF16 384 B");
    expect(glmd::topk_mb_work_bytes(3) == (size_t) 3 * (256 + 8 + 4 * glmd::kTopkMbBlocks) * 4, "topk_mb work bytes");
    {   // the WMMA scorer reuses each warp's 16 x (D + 8) FP16 key tile as its 16 x H f32 output scratch
        bool fits = true;
        int admitted = 0;
        for (int H = 1; H <= 64; ++H)
            for (int D = 1; D <= 512; ++D)
                if (glmd::idx_wmma_shape_rule(H, D)) {
                    ++admitted;
                    fits &= (size_t) 16 * H * 4 <= (size_t) 16 * (D + 8) * 2;
                }
        expect(fits && admitted > 0, "idx WMMA shape rule: every admitted shape's f32 scratch fits its key tile");
        expect(glmd::idx_wmma_shape_rule(32, 128) && glmd::idx_wmma_shape_rule(16, 64) &&
                   glmd::idx_wmma_shape_rule(32, 64) && glmd::idx_wmma_shape_rule(16, 32) &&
                   glmd::idx_wmma_shape_rule(32, 256),
               "idx WMMA shape rule admits 32 x 128 (GLM-5.3), 16 x 64, 32 x 64, 16 x 32, 32 x 256");
        expect(!glmd::idx_wmma_shape_rule(32, 16) && !glmd::idx_wmma_shape_rule(32, 32) &&
                   !glmd::idx_wmma_shape_rule(32, 48) && !glmd::idx_wmma_shape_rule(16, 16) &&
                   !glmd::idx_wmma_shape_rule(48, 128) && !glmd::idx_wmma_shape_rule(32, 272) &&
                   !glmd::idx_wmma_shape_rule(8, 128),
               "idx WMMA shape rule refuses 32 x {16, 32, 48}, 16 x 16 (scratch overrun) and 48 / 8 heads, D 272");
    }
}

// ---------------------------------------------------------------- weights
struct Weights {
    int type;
    std::vector<uint8_t> kb, vb;      // stored rows
    std::vector<float> kb_f, vb_f;    // the stored values in f32 (the reference reads these)
};
Weights make_weights(std::mt19937& rng, int type) {
    Weights w;
    w.type = type;
    const size_t nk = (size_t) kHeads * kKvLora * kNope, nv = (size_t) kHeads * kVHead * kKvLora;
    std::vector<float> fk = randv(rng, nk, 0.05f), fv = randv(rng, nv, 0.05f);
    w.kb_f.resize(nk);
    w.vb_f.resize(nv);
    if (type == glmd::kWTypeBF16) {
        w.kb.resize(nk * 2);
        w.vb.resize(nv * 2);
        uint16_t* k16 = (uint16_t*) w.kb.data();
        uint16_t* v16 = (uint16_t*) w.vb.data();
        for (size_t i = 0; i < nk; ++i) w.kb_f[i] = bf16_float(k16[i] = bf16_bits(fk[i]));
        for (size_t i = 0; i < nv; ++i) w.vb_f[i] = bf16_float(v16[i] = bf16_bits(fv[i]));
    } else {
        w.kb.resize(nk / 32 * 34);
        w.vb.resize(nv / 32 * 34);
        glmd::vtwin::q8_0_quantize(fk.data(), (int) nk, w.kb.data());
        glmd::vtwin::q8_0_quantize(fv.data(), (int) nv, w.vb.data());
        glmd::vtwin::q8_0_dequantize(w.kb.data(), (int) nk, w.kb_f.data());
        glmd::vtwin::q8_0_dequantize(w.vb.data(), (int) nv, w.vb_f.data());
    }
    return w;
}

void test_headwise(std::mt19937& rng) {
    for (int type : {glmd::kWTypeBF16, glmd::kWTypeQ8_0}) {
        const int H = 8, R = 64, K = 192, stride = 256;
        std::vector<float> f = randv(rng, (size_t) H * R * K, 0.05f), deq(f.size()), x = randv(rng, (size_t) H * stride);
        std::vector<uint8_t> W;
        if (type == glmd::kWTypeBF16) {
            W.resize(f.size() * 2);
            for (size_t i = 0; i < f.size(); ++i) deq[i] = bf16_float(((uint16_t*) W.data())[i] = bf16_bits(f[i]));
        } else {
            W.resize(f.size() / 32 * 34);
            glmd::vtwin::q8_0_quantize(f.data(), (int) f.size(), W.data());
            glmd::vtwin::q8_0_dequantize(W.data(), (int) f.size(), deq.data());
        }
        std::vector<float> out((size_t) H * R);
        std::vector<glmd::Q81> q8(out.size() / 32);
        glmd::vtwin::headwise(W.data(), type, x.data(), H, R, K, stride, out.data(), q8.data());
        double e = 0, mx = 0;
        for (int h = 0; h < H; ++h)
            for (int r = 0; r < R; ++r) {
                double ref = 0;
                for (int c = 0; c < K; ++c) ref += (double) deq[((size_t) h * R + r) * K + c] * x[(size_t) h * stride + c];
                e = std::max(e, std::fabs(ref - out[(size_t) h * R + r]));
                mx = std::max(mx, std::fabs(ref));
            }
        gate(type == glmd::kWTypeBF16 ? "headwise BF16 vs dequantized rows (max err / max |y|)"
                                      : "headwise Q8_0 vs dequantized rows (max err / max |y|)",
             e / mx, 1e-5);
    }
}

// ---------------------------------------------------------------- the cache
struct Cache {
    int n = 0;
    std::vector<uint16_t> lat16, k_pe;
    std::vector<uint8_t> lat8;
};
Cache make_cache(std::mt19937& rng, int n) {
    Cache c;
    c.n = n;
    c.lat16.resize((size_t) n * kKvLora);
    c.k_pe.resize((size_t) n * kRope);
    c.lat8.resize((size_t) n * glmd::lat8_rec_bytes(kKvLora));
    std::vector<float> inv(kRope / 2), one(kKvLora, 1.0f), qr(kQLora, 1.0f);
    glmd::rope_inv_freq(inv.data(), kRope, kTheta);
    for (int p = 0; p < n; ++p) {
        std::vector<float> kv = randv(rng, kKvLora + kRope, 1.0f);
        glmd::DsaPrep a;
        a.qr_raw = qr.data();
        a.q_a_norm = qr.data();
        a.q_lora = kQLora;
        a.kv_raw = kv.data();
        a.kv_norm = one.data();
        a.lat = c.lat16.data();
        a.kv_lora = kKvLora;
        a.k_pe = c.k_pe.data();
        a.qk_rope = kRope;
        a.inv_freq = inv.data();
        a.p = p;
        glmd::twin::dsa_prep(a);
        a.lat = c.lat8.data();
        a.lat_q8 = 1;
        glmd::twin::dsa_prep(a);
    }
    return c;
}
// reference rows [c_kv | k_pe] of the values the kernels stage (FP16; an INT8 record decoded then rounded to FP16)
std::vector<float> kv_rows(const Cache& c, bool q8) {
    const int qd = kKvLora + kRope, rb = glmd::lat8_rec_bytes(kKvLora);
    std::vector<float> kv((size_t) c.n * qd);
    for (int s = 0; s < c.n; ++s) {
        for (int j = 0; j < kKvLora; ++j) {
            float v;
            if (q8) {
                const uint8_t* rec = c.lat8.data() + (size_t) rb * s;
                uint16_t d16;
                std::memcpy(&d16, rec + kKvLora + 2 * (j / 32), 2);
                v = glmd::twin::half_float(glmd::twin::half_bits((int8_t) rec[j] * glmd::twin::half_float(d16)));
            } else {
                v = glmd::twin::half_float(c.lat16[(size_t) kKvLora * s + j]);
            }
            kv[(size_t) s * qd + j] = v;
        }
        for (int j = 0; j < kRope; ++j) kv[(size_t) s * qd + kKvLora + j] = glmd::twin::half_float(c.k_pe[(size_t) kRope * s + j]);
    }
    return kv;
}

void test_fused(std::mt19937& rng) {
    const int n_cache = 4096, qd = kKvLora + kRope, q_stride = kNope + kRope;
    const Cache cache = make_cache(rng, n_cache);
    const std::vector<float> kv16 = kv_rows(cache, false), kv8 = kv_rows(cache, true);
    for (int wt : {glmd::kWTypeBF16, glmd::kWTypeQ8_0}) {
        const Weights W = make_weights(rng, wt);
        const char* wn = wt == glmd::kWTypeBF16 ? "BF16" : "Q8_0";
        for (float qscale : {1.0f, 12.0f}) {   // 12: |S * scale| up to ~120, the long-context overflow regime
            std::vector<float> q = randv(rng, (size_t) kHeads * q_stride, qscale);
            std::vector<float> Q((size_t) kHeads * qd);
            glmd::cpu::mla_absorb_q(q.data(), W.kb_f.data(), kHeads, kNope, kRope, kKvLora, 1, Q.data());
            for (int n_sel : {0, 1, 37, 2048}) {
                if (qscale != 1.0f && n_sel != 2048) continue;
                std::vector<int> perm(n_cache);
                std::iota(perm.begin(), perm.end(), 0);
                std::shuffle(perm.begin(), perm.end(), rng);
                std::vector<int> cells(perm.begin(), perm.begin() + n_sel);
                if (n_sel == 37)
                    for (int i = 0; i < n_sel; i += 4) cells[i] = -1;
                for (int q8 = 0; q8 < 2; ++q8) {
                    std::vector<float> ref((size_t) kHeads * kVHead);
                    glmd::cpu::mla_attention(Q.data(), q8 ? kv8.data() : kv16.data(), cells.data(), W.vb_f.data(),
                                             kHeads, kKvLora, kRope, kVHead, 1, n_sel, glmd::kGlm53KqScale, ref.data());
                    std::vector<float> base;
                    for (int split : {1, 7, 32, 64}) {
                        if (n_sel != 2048 && split != 32) continue;
                        if (qscale != 1.0f && split != 32) continue;
                        std::vector<float> out((size_t) kHeads * kVHead);
                        glmd::FusedMla f;
                        f.q = q.data();
                        f.q_stride = q_stride;
                        f.wk_b = W.kb.data();
                        f.wv_b = W.vb.data();
                        f.w_type = wt;
                        f.lat = q8 ? (const void*) cache.lat8.data() : (const void*) cache.lat16.data();
                        f.lat_q8 = q8;
                        f.k_pe = cache.k_pe.data();
                        f.cells = cells.data();
                        f.n_sel = n_sel;
                        f.scale = glmd::kGlm53KqScale;
                        f.n_split = split;
                        f.out = out.data();
                        glmd::vtwin::mla_fused(f);
                        char what[128];
                        if (n_sel == 0) {
                            double mx = 0;
                            for (float v : out) mx = std::max(mx, (double) std::fabs(v));
                            std::snprintf(what, sizeof what, "fused %s %s: n_sel 0 writes zeros", wn, q8 ? "INT8" : "FP16");
                            gate(what, mx, 1e-30);
                            continue;
                        }
                        bool finite = true;
                        for (float v : out) finite &= std::isfinite(v);
                        std::snprintf(what, sizeof what, "fused %s %s n_sel %d split %d q x%g vs ref, rel L2", wn,
                                      q8 ? "INT8" : "FP16", n_sel, split, qscale);
                        gate(what, finite ? rel_l2(out, ref) : 1.0, 2e-3);
                        if (base.empty()) {
                            base = out;
                        } else {
                            // fp16 P is rounded against each split's running max, so splits differ by P rounding
                            std::snprintf(what, sizeof what, "fused %s split %d vs split 1 (P rounding only)", wn, split);
                            gate(what, rel_l2(out, base), 5e-4);
                        }
                        if (n_sel == 2048 && split == 32 && !q8 && qscale == 1.0f) {
                            f.scale = 1.0f / std::sqrt((float) kNope);
                            glmd::vtwin::mla_fused(f);
                            expect(rel_l2(out, ref) > 1e-2, "fused: the Flash scale 1/sqrt(192) is detected");
                        }
                    }
                    // mla2 (the default kernel) with the same weights: the B10.7 Q8_0 path of the twin
                    if (n_sel == 2048 && qscale == 1.0f) {
                        std::vector<float> out((size_t) kHeads * kVHead);
                        glmd::Mla2 m;
                        m.q = q.data();
                        m.q_stride = q_stride;
                        m.wk_b = (const uint16_t*) W.kb.data();
                        m.wv_b = (const uint16_t*) W.vb.data();
                        m.w_type = wt;
                        m.lat = q8 ? (const void*) cache.lat8.data() : (const void*) cache.lat16.data();
                        m.lat_q8 = q8;
                        m.k_pe = cache.k_pe.data();
                        m.cells = cells.data();
                        m.n_sel = n_sel;
                        m.scale = glmd::kGlm53KqScale;
                        m.out = out.data();
                        glmd::twin::mla2(m);
                        char what[96];
                        std::snprintf(what, sizeof what, "mla2 %s %s n_sel 2048 vs ref, rel L2", wn, q8 ? "INT8" : "FP16");
                        gate(what, rel_l2(out, ref), 1e-4);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- the indexer scorers
struct Overlap {
    double overlap;
    int ties;
};
// the two top-k sets of one row, mismatches not counted where the reference score is within rel tol of the cut
Overlap overlap(const std::vector<float>& ref, const std::vector<float>& test, int n, int k, double tol) {
    std::vector<int> ia(n), ib(n);
    std::iota(ia.begin(), ia.end(), 0);
    std::iota(ib.begin(), ib.end(), 0);
    auto by = [](const std::vector<float>& v) {
        return [&v](int x, int y) { return v[x] > v[y] || (v[x] == v[y] && x < y); };
    };
    std::partial_sort(ia.begin(), ia.begin() + k, ia.end(), by(ref));
    std::partial_sort(ib.begin(), ib.begin() + k, ib.end(), by(test));
    const double cut = ref[ia[k - 1]];
    std::vector<char> ina(n, 0), inb(n, 0);
    for (int i = 0; i < k; ++i) {
        ina[ia[i]] = 1;
        inb[ib[i]] = 1;
    }
    int miss = 0, ties = 0, cnt = 0;
    for (int s = 0; s < n; ++s) {
        if (ina[s] == inb[s]) {
            cnt += ina[s];
            continue;
        }
        if (std::fabs(ref[s] - cut) <= tol * std::max(1e-30, std::fabs(cut))) {
            ++ties;
            continue;
        }
        ++miss;
    }
    (void) cnt;
    return {1.0 - (double) miss / (2.0 * k), ties};
}

void test_scorer(std::mt19937& rng) {
    const int H = 32, D = 128, k = glmd::kTopK;
    // decode at n 20000 (F32 keys) and 32768 (FP16 keys, B10.8)
    for (int n : {20000, 32768}) {
        std::vector<float> keys = randv(rng, (size_t) n * D), iq = randv(rng, (size_t) H * D),
                           w = randv(rng, H, 1.0f / std::sqrt((float) H * D));
        std::vector<uint16_t> k16(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) k16[i] = glmd::twin::half_bits(keys[i]);
        std::vector<float> ref(n), simple(n), wmma(n), wmma16(n);
        const int last = n - 1;
        glmd::cpu::indexer_score(iq.data(), keys.data(), w.data(), H, D, 1, n, &last, ref.data());
        glmd::IdxScoreWmma a;
        a.iq = iq.data();
        a.w = w.data();
        a.keys = keys.data();
        a.n_kv = n;
        a.score = simple.data();
        glmd::vtwin::idx_score_simple(a);
        double e = 0, mx = 0;
        for (int s = 0; s < n; ++s) {
            e = std::max(e, (double) std::fabs(simple[s] - ref[s]));
            mx = std::max(mx, (double) std::fabs(ref[s]));
        }
        char what[96];
        std::snprintf(what, sizeof what, "scorer simple F32 n %d vs reference (max err / max)", n);
        gate(what, e / mx, 1e-5);
        a.score = wmma.data();
        glmd::vtwin::idx_score_wmma(a);
        Overlap o = overlap(ref, wmma, n, k, 1e-3);
        std::snprintf(what, sizeof what, "scorer WMMA F32 keys n %d: top-2048 miss rate (ties %d)", n, o.ties);
        gate(what, 1.0 - o.overlap, 1e-3);
        a.keys = k16.data();
        a.key_f16 = 1;
        a.score = wmma16.data();
        glmd::vtwin::idx_score_wmma(a);
        o = overlap(ref, wmma16, n, k, 1e-3);
        std::snprintf(what, sizeof what, "scorer WMMA FP16 keys n %d: top-2048 miss rate (ties %d)", n, o.ties);
        gate(what, 1.0 - o.overlap, 5e-3);
        std::vector<float> s16(n);
        a.score = s16.data();
        glmd::vtwin::idx_score_simple(a);
        o = overlap(ref, s16, n, k, 1e-3);
        std::snprintf(what, sizeof what, "scorer simple FP16 keys n %d: top-2048 miss rate (ties %d)", n, o.ties);
        gate(what, 1.0 - o.overlap, 5e-3);
    }
    // prefill: 3 tokens at positions 4100, 6000, 8191 of an 8192-cell cache
    {
        const int n = 8192, T = 3;
        const int pos[T] = {4100, 6000, 8191};
        std::vector<float> keys = randv(rng, (size_t) n * D), iq = randv(rng, (size_t) T * H * D),
                           w = randv(rng, (size_t) T * H, 1.0f / std::sqrt((float) H * D));
        std::vector<float> ref((size_t) n * T), wm((size_t) n * T);
        glmd::cpu::indexer_score(iq.data(), keys.data(), w.data(), H, D, T, n, pos, ref.data());
        glmd::IdxScoreWmma a;
        a.iq = iq.data();
        a.w = w.data();
        a.keys = keys.data();
        a.n_kv = n;
        a.T = T;
        a.pos = pos;
        a.score = wm.data();
        glmd::vtwin::idx_score_wmma(a);
        bool masked = true;
        double worst = 0;
        for (int t = 0; t < T; ++t) {
            for (int s = pos[t] + 1; s < n; ++s) masked &= std::isinf(wm[(size_t) t * n + s]) && wm[(size_t) t * n + s] < 0;
            std::vector<float> r(ref.begin() + (size_t) t * n, ref.begin() + (size_t) t * n + pos[t] + 1),
                tt(wm.begin() + (size_t) t * n, wm.begin() + (size_t) t * n + pos[t] + 1);
            worst = std::max(worst, 1.0 - overlap(r, tt, pos[t] + 1, k, 1e-3).overlap);
        }
        expect(masked, "scorer WMMA prefill: cells past each token's position are -inf");
        gate("scorer WMMA prefill 3 tokens: worst top-2048 miss rate", worst, 1e-3);
    }
}

// ---------------------------------------------------------------- top-k
void fill_row(std::mt19937& rng, float* sc, int n, int kind) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> ud(0, 7);
    for (int i = 0; i < n; ++i) {
        if (kind == 0) sc[i] = nd(rng);
        else if (kind == 1) sc[i] = (float) ud(rng);              // heavy ties
        else sc[i] = (i % 5 == 0) ? -std::numeric_limits<float>::infinity() : (i % 7 == 0 ? -0.0f : nd(rng));
    }
}
void test_topk(std::mt19937& rng) {
    const int k = glmd::kTopK;
    bool same = true, oracle_ok = true;
    int rows = 0;
    for (int n : {2049, 5000, 40000, 131072})
        for (int kind = 0; kind < 3; ++kind)
            for (int rep = 0; rep < (n > 40000 ? 2 : 6); ++rep) {
                std::vector<float> sc(n);
                fill_row(rng, sc.data(), n, kind);
                std::vector<int> a(k), b(k);
                glmd::vtwin::topk_mb(sc.data(), n, n, 1, nullptr, k, a.data());
                glmd::twin::topk(sc.data(), n, n, 1, nullptr, k, b.data());
                same &= a == b;
                std::vector<int> idx(n);
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int x, int y) {
                    const float vx = sc[x] == 0.0f ? 0.0f : sc[x], vy = sc[y] == 0.0f ? 0.0f : sc[y];
                    return vx > vy || (vx == vy && x < y);
                });
                std::sort(idx.begin(), idx.begin() + k);
                oracle_ok &= std::equal(a.begin(), a.end(), idx.begin());
                ++rows;
            }
    std::printf("  top-k rows checked: %d\n", rows);
    expect(same, "topk_mb == single-block topk (exact)");
    expect(oracle_ok, "topk_mb == sort oracle in position order (exact)");
    // per-row positions: identity rows, short rows and a long row in one batch
    {
        const int n = 50000, T = 4;
        const int pos[T] = {10, 2047, 2048, 49999};
        std::vector<float> sc((size_t) n * T);
        for (int t = 0; t < T; ++t) fill_row(rng, sc.data() + (size_t) t * n, n, t % 3);
        std::vector<int> a((size_t) k * T), b((size_t) k * T);
        glmd::vtwin::topk_mb(sc.data(), n, n, T, pos, k, a.data());
        glmd::twin::topk(sc.data(), n, n, T, pos, k, b.data());
        expect(a == b, "topk_mb per-row positions (identity, 2049, 50000) == topk");
        expect(a[11] == -1 && a[10] == 10 && a[k + k - 1] == k - 1, "topk_mb identity rows pad with -1");
    }
}

// ---------------------------------------------------------------- B10.7 binder rule
void test_pick_headwise() {
    namespace gd = strata::core::glmdsa;
    static const char q8[4] = {}, b16[4] = {}, q4[4] = {};
    std::map<std::string, std::pair<int, const void*>> native = {{"blk.0.attn_k_b.weight", {8, q8}},
                                                                  {"blk.1.attn_k_b.weight", {2, q4}}};
    std::map<std::string, const void*> w16 = {{"blk.1.attn_k_b.weight", b16}, {"blk.2.attn_k_b.weight", b16}};
    gd::Tables t;
    t.native = [&](const std::string& n, int& type, const void*& q, const void*& f) {
        const auto it = native.find(n);
        if (it == native.end()) return false;
        type = it->second.first;
        q = it->second.second;
        f = nullptr;
        return true;
    };
    t.bf16 = [&](const std::string& n) -> const void* {
        const auto it = w16.find(n);
        return it == w16.end() ? nullptr : it->second;
    };
    t.f32 = [](const std::string&) -> const void* { return nullptr; };
    t.mv_ok = [](int) { return true; };
    const gd::Pick a = gd::pick_headwise(t, "blk.0.attn_k_b.weight"), b = gd::pick_headwise(t, "blk.1.attn_k_b.weight"),
                   c = gd::pick_headwise(t, "blk.2.attn_k_b.weight"), d = gd::pick_headwise(t, "blk.3.attn_k_b.weight");
    expect(a.src == gd::Src::Native && a.type == glmd::kWTypeQ8_0 && a.w == q8, "pick_headwise: native Q8_0 served");
    expect(b.src == gd::Src::Bf16 && b.type == glmd::kWTypeBF16 && b.w == b16, "pick_headwise: Q4_0 -> the BF16 copy");
    expect(c.src == gd::Src::Bf16 && d.src == gd::Src::Missing, "pick_headwise: BF16 only / missing");
}

}  // namespace

int main() {
    std::mt19937 rng(20261012);
    test_splits();
    test_headwise(rng);
    test_fused(rng);
    test_scorer(rng);
    test_topk(rng);
    test_pick_headwise();
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
