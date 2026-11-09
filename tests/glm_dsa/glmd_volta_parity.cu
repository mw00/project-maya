// tests/glm_dsa/glmd_volta_parity.cu - the Volta glm-dsa decode kernels (glm-dsa B10.2-B10.4, B10.7, B10.8) on a GPU
// vs the F32 reference kernels (glmd::ref_*), the simple glmd kernels and the CPU twins (glmd::vtwin), GLM-5.3 sizes.
//   headwise  BF16 / Q8_0 GPU vs the twin: rel L2 < 1e-6;
//   fused     mla_fused vs ref_mla_absorb_q + ref_mla_attention on the stored values: rel L2 < 2e-3 at n_sel 1, 37,
//             2048 (FP16 / INT8 latents, BF16 / Q8_0 weights, splits 1 / 32 / 64), vs the twin < 1e-3, vs mla2
//             < 2e-3; n_sel 0 writes zeros; mla2 with Q8_0 weights vs the reference < 1e-4; times fused vs mla2
//             at n_sel 2048 (printed; B10.2's 0.35x goal is measured on the test PC, not gated here);
//   scorer    simple GPU == twin within 1e-5 of the scale; WMMA GPU vs twin within 1e-3 of the scale; WMMA
//             top-2048 miss rate vs ref_indexer_score < 1e-3 outside near-ties (decode n 20000, prefill 5 tokens
//             with positions); FP16 keys < 5e-3 at 32K (B10.8); dsa_prep's FP16 key == fp16(F32 key);
//   top-k     topk_mb == topk exactly (n 2049 / 40000 / 131072, ties, -inf, per-row positions), three calls
//             in a row on the same scratch (it must be left zeroed).
// Needs a GPU; exits 77 (ctest SKIP_RETURN_CODE) when there is none.
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_ref.hpp"
#include "strata/kernels/glmd_volta.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace {

#if defined(STRATA_USE_HIP)
// HIP builds have neither nvcuda::wmma kernel (the fused MLA, the WMMA scorer): their cases are skipped, and the WMMA
// scorer refuses every shape there (idx_score_wmma_supported is false); the rest (headwise, mla2 with BF16 / Q8_0
// weights, the CUDA-core scorer, the radix top-k, the FP16 keys) runs as on CUDA
constexpr bool kWmma = false;
#else
constexpr bool kWmma = true;
#endif

int g_fail = 0;
int g_expected_volta_errors = 0;   // refusals the test provokes on purpose (volta_launch_errors counts them)

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
        CK(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
        if (n) CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    explicit Dev(size_t count) : n(count) {
        CK(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
        CK(cudaMemset(p, 0, std::max<size_t>(1, n) * sizeof(T)));
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<T> get() const {
        std::vector<T> h(n);
        CK(cudaDeviceSynchronize());
        if (n) CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

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

constexpr int kQLora = 2048, kKvLora = 512, kRope = 64, kHeads = 64, kNope = 192, kVHead = 256;
constexpr int kIdxDim = 128, kIdxHeads = 32;

// ---------------------------------------------------------------- headwise
void test_headwise(std::mt19937& rng) {
    for (int wt : {glmd::kWTypeBF16, glmd::kWTypeQ8_0}) {
        const int H = 64, R = 512, K = 192, stride = 256;
        std::vector<float> f = randv(rng, (size_t) H * R * K, 0.05f), x = randv(rng, (size_t) H * stride);
        std::vector<uint8_t> W;
        if (wt == glmd::kWTypeBF16) {
            W.resize(f.size() * 2);
            for (size_t i = 0; i < f.size(); ++i) ((uint16_t*) W.data())[i] = bf16_bits(f[i]);
        } else {
            W.resize(f.size() / 32 * 34);
            glmd::vtwin::q8_0_quantize(f.data(), (int) f.size(), W.data());
        }
        Dev<uint8_t> dW(W);
        Dev<float> dx(x), dout((size_t) H * R);
        expect(glmd::headwise(dW.p, wt, dx.p, H, R, K, stride, dout.p, nullptr, nullptr), "headwise: launch");
        std::vector<float> t((size_t) H * R);
        glmd::vtwin::headwise(W.data(), wt, x.data(), H, R, K, stride, t.data(), nullptr);
        gate(wt == glmd::kWTypeBF16 ? "headwise BF16 GPU vs twin, rel L2" : "headwise Q8_0 GPU vs twin, rel L2",
             rel_l2(dout.get(), t), 1e-6);
    }
}

// ---------------------------------------------------------------- fused MLA
void test_fused(std::mt19937& rng) {
    const int n_cache = 4096, qd = kKvLora + kRope, q_stride = kNope + kRope, rb = glmd::lat8_rec_bytes(kKvLora);
    // the cache through the twin's prep (FP16 and INT8 records of the same rows)
    std::vector<uint16_t> lat16((size_t) n_cache * kKvLora), kpe((size_t) n_cache * kRope);
    std::vector<uint8_t> lat8((size_t) n_cache * rb);
    {
        std::vector<float> inv(kRope / 2), one(kKvLora, 1.0f), qr(kQLora, 1.0f);
        glmd::rope_inv_freq(inv.data(), kRope, 8e6);
        for (int p = 0; p < n_cache; ++p) {
            std::vector<float> kv = randv(rng, qd);
            glmd::DsaPrep a;
            a.qr_raw = qr.data(); a.q_a_norm = qr.data(); a.q_lora = kQLora;
            a.kv_raw = kv.data(); a.kv_norm = one.data(); a.lat = lat16.data(); a.kv_lora = kKvLora;
            a.k_pe = kpe.data(); a.qk_rope = kRope; a.inv_freq = inv.data(); a.p = p;
            glmd::twin::dsa_prep(a);
            a.lat = lat8.data();
            a.lat_q8 = 1;
            glmd::twin::dsa_prep(a);
        }
    }
    auto rows = [&](bool q8) {
        std::vector<float> kv((size_t) n_cache * qd);
        for (int s = 0; s < n_cache; ++s) {
            for (int c = 0; c < kKvLora; ++c) {
                float v;
                if (q8) {
                    const uint8_t* rec = lat8.data() + (size_t) rb * s;
                    uint16_t d16;
                    std::memcpy(&d16, rec + kKvLora + 2 * (c / 32), 2);
                    v = glmd::twin::half_float(glmd::twin::half_bits((int8_t) rec[c] * glmd::twin::half_float(d16)));
                } else {
                    v = glmd::twin::half_float(lat16[(size_t) kKvLora * s + c]);
                }
                kv[(size_t) s * qd + c] = v;
            }
            for (int j = 0; j < kRope; ++j) kv[(size_t) s * qd + kKvLora + j] = glmd::twin::half_float(kpe[(size_t) kRope * s + j]);
        }
        return kv;
    };
    Dev<uint16_t> d_lat16(lat16), d_kpe(kpe);
    Dev<uint8_t> d_lat8(lat8);
    Dev<float> d_kv16(rows(false)), d_kv8(rows(true));
    const size_t work_bytes = std::max(glmd::mla2_work_bytes(kHeads, kKvLora, 2048),
                                       glmd::mla_fused_work_bytes(kHeads, kKvLora, kRope, 2048, 64));
    Dev<uint8_t> d_work(work_bytes);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int wt : {glmd::kWTypeBF16, glmd::kWTypeQ8_0}) {
        const char* wn = wt == glmd::kWTypeBF16 ? "BF16" : "Q8_0";
        const size_t nk = (size_t) kHeads * kKvLora * kNope, nv = (size_t) kHeads * kVHead * kKvLora;
        std::vector<float> fk = randv(rng, nk, 0.05f), fv = randv(rng, nv, 0.05f), kf(nk), vf(nv);
        std::vector<uint8_t> kb, vb;
        if (wt == glmd::kWTypeBF16) {
            kb.resize(nk * 2);
            vb.resize(nv * 2);
            for (size_t i = 0; i < nk; ++i) kf[i] = bf16_float(((uint16_t*) kb.data())[i] = bf16_bits(fk[i]));
            for (size_t i = 0; i < nv; ++i) vf[i] = bf16_float(((uint16_t*) vb.data())[i] = bf16_bits(fv[i]));
        } else {
            kb.resize(nk / 32 * 34);
            vb.resize(nv / 32 * 34);
            glmd::vtwin::q8_0_quantize(fk.data(), (int) nk, kb.data());
            glmd::vtwin::q8_0_quantize(fv.data(), (int) nv, vb.data());
            glmd::vtwin::q8_0_dequantize(kb.data(), (int) nk, kf.data());
            glmd::vtwin::q8_0_dequantize(vb.data(), (int) nv, vf.data());
        }
        Dev<uint8_t> d_kb(kb), d_vb(vb);
        Dev<float> d_kf(kf), d_vf(vf);
        const std::vector<float> q = randv(rng, (size_t) kHeads * q_stride);
        Dev<float> d_q(q), d_Q((size_t) kHeads * qd);
        glmd::ref_mla_absorb_q(d_q.p, d_kf.p, kHeads, kNope, kRope, kKvLora, 1, d_Q.p, nullptr);
        for (int n_sel : {0, 1, 37, 2048}) {
            std::vector<int> perm(n_cache);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            std::vector<int> cells(perm.begin(), perm.begin() + n_sel);
            if (n_sel == 37)
                for (int i = 0; i < n_sel; i += 4) cells[i] = -1;
            Dev<int> d_cells(cells);
            for (int q8 = 0; q8 < 2; ++q8) {
                Dev<float> d_ref((size_t) kHeads * kVHead);
                if (n_sel > 0)
                    glmd::ref_mla_attention(d_Q.p, q8 ? d_kv8.p : d_kv16.p, d_cells.p, d_vf.p, kHeads, kKvLora, kRope,
                                            kVHead, 1, n_sel, glmd::kGlm53KqScale, d_ref.p, nullptr);
                const auto ref = d_ref.get();
                for (int split : {1, 32, 64}) {
                    if (!kWmma) break;   // (HIP: no fused kernel)
                    if (n_sel != 2048 && split != 32) continue;
                    Dev<float> d_out((size_t) kHeads * kVHead);
                    glmd::FusedMla f;
                    f.q = d_q.p; f.q_stride = q_stride; f.wk_b = d_kb.p; f.wv_b = d_vb.p; f.w_type = wt;
                    f.lat = q8 ? (const void*) d_lat8.p : (const void*) d_lat16.p; f.lat_q8 = q8; f.k_pe = d_kpe.p;
                    f.cells = d_cells.p; f.n_sel = n_sel; f.scale = glmd::kGlm53KqScale; f.n_split = split;
                    f.work = d_work.p; f.work_bytes = work_bytes; f.out = d_out.p;
                    char what[128];
                    std::snprintf(what, sizeof what, "fused %s %s n_sel %d split %d: launch", wn, q8 ? "INT8" : "FP16",
                                  n_sel, split);
                    expect(glmd::mla_fused(f, nullptr), what);
                    const auto out = d_out.get();
                    if (n_sel == 0) {
                        double mx = 0;
                        for (float v : out) mx = std::max(mx, (double) std::fabs(v));
                        gate("fused: n_sel 0 writes zeros", mx, 1e-30);
                        continue;
                    }
                    std::snprintf(what, sizeof what, "fused %s %s n_sel %d split %d vs ref, rel L2", wn,
                                  q8 ? "INT8" : "FP16", n_sel, split);
                    gate(what, rel_l2(out, ref), 2e-3);
                    // the twin on host copies
                    std::vector<float> t((size_t) kHeads * kVHead);
                    glmd::FusedMla h = f;
                    h.q = q.data(); h.wk_b = kb.data(); h.wv_b = vb.data();
                    h.lat = q8 ? (const void*) lat8.data() : (const void*) lat16.data(); h.k_pe = kpe.data();
                    h.cells = cells.data(); h.out = t.data();
                    glmd::vtwin::mla_fused(h);
                    std::snprintf(what, sizeof what, "fused %s %s n_sel %d split %d vs twin, rel L2", wn,
                                  q8 ? "INT8" : "FP16", n_sel, split);
                    gate(what, rel_l2(out, t), 1e-3);
                }
                if (n_sel == 2048) {   // mla2 (B4.4 kernel, B10.7 Q8_0 path) and the timing
                    Dev<float> d_m2((size_t) kHeads * kVHead), d_fu((size_t) kHeads * kVHead);
                    glmd::Mla2 m;
                    m.q = d_q.p; m.q_stride = q_stride; m.wk_b = (const uint16_t*) d_kb.p;
                    m.wv_b = (const uint16_t*) d_vb.p; m.w_type = wt;
                    m.lat = q8 ? (const void*) d_lat8.p : (const void*) d_lat16.p; m.lat_q8 = q8; m.k_pe = d_kpe.p;
                    m.cells = d_cells.p; m.n_sel = n_sel; m.scale = glmd::kGlm53KqScale; m.work = d_work.p;
                    m.work_bytes = work_bytes; m.out = d_m2.p;
                    expect(glmd::mla2(m, nullptr), "mla2: launch");
                    char what[96];
                    std::snprintf(what, sizeof what, "mla2 %s %s n_sel 2048 vs ref, rel L2", wn, q8 ? "INT8" : "FP16");
                    const auto m2 = d_m2.get();
                    gate(what, rel_l2(m2, ref), 1e-4);
                    if (!kWmma) continue;   // (HIP: no fused kernel to time or compare)
                    glmd::FusedMla f;
                    f.q = d_q.p; f.q_stride = q_stride; f.wk_b = d_kb.p; f.wv_b = d_vb.p; f.w_type = wt;
                    f.lat = m.lat; f.lat_q8 = q8; f.k_pe = d_kpe.p; f.cells = d_cells.p; f.n_sel = n_sel;
                    f.scale = glmd::kGlm53KqScale; f.n_split = 32; f.work = d_work.p; f.work_bytes = work_bytes;
                    f.out = d_fu.p;
                    float t_m2 = 0, t_fu = 0;
                    for (int rep = 0; rep < 2; ++rep) {   // rep 0 warms up
                        CK(cudaEventRecord(e0));
                        for (int i = 0; i < 20; ++i) glmd::mla2(m, nullptr);
                        CK(cudaEventRecord(e1));
                        CK(cudaEventSynchronize(e1));
                        CK(cudaEventElapsedTime(&t_m2, e0, e1));
                        CK(cudaEventRecord(e0));
                        for (int i = 0; i < 20; ++i) glmd::mla_fused(f, nullptr);
                        CK(cudaEventRecord(e1));
                        CK(cudaEventSynchronize(e1));
                        CK(cudaEventElapsedTime(&t_fu, e0, e1));
                    }
                    std::printf("  time n_sel 2048 %s %s: mla2 %.1f us, fused %.1f us (%.2fx)\n", wn,
                                q8 ? "INT8" : "FP16", t_m2 * 50.0f, t_fu * 50.0f, t_fu / std::max(t_m2, 1e-6f));
                    std::snprintf(what, sizeof what, "fused vs mla2 %s %s n_sel 2048, rel L2", wn, q8 ? "INT8" : "FP16");
                    gate(what, rel_l2(d_fu.get(), m2), 2e-3);
                }
            }
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    expect(glmd::volta_launch_errors() == 0, "no launch errors in the Volta kernels");
}

// ---------------------------------------------------------------- scorer
double miss_rate(const std::vector<float>& ref, const std::vector<float>& test, int n, int k, double tol) {
    if (n <= k) return 0.0;
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
    int miss = 0;
    for (int s = 0; s < n; ++s)
        if (ina[s] != inb[s] && std::fabs(ref[s] - cut) > tol * std::max(1e-30, std::fabs(cut))) ++miss;
    return (double) miss / (2.0 * k);
}
double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0, mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isinf(b[i])) {
            if (!(std::isinf(a[i]) && (a[i] < 0) == (b[i] < 0))) return 1.0;
            continue;
        }
        e = std::max(e, (double) std::fabs(a[i] - b[i]));
        mx = std::max(mx, (double) std::fabs(b[i]));
    }
    return mx > 0 ? e / mx : e;
}

void test_scorer(std::mt19937& rng) {
    const int H = kIdxHeads, D = kIdxDim, k = glmd::kTopK;
    for (int n : {20000, 32768}) {
        std::vector<float> keys = randv(rng, (size_t) n * D), iq = randv(rng, (size_t) H * D),
                           w = randv(rng, H, 1.0f / std::sqrt((float) H * D));
        std::vector<uint16_t> k16(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) k16[i] = glmd::twin::half_bits(keys[i]);
        Dev<float> d_keys(keys), d_iq(iq), d_w(w), d_ref(n), d_s(n), d_wm(n), d_wm16(n);
        Dev<uint16_t> d_k16(k16);
        Dev<int> d_last(std::vector<int>{n - 1});
        glmd::ref_indexer_score(d_iq.p, d_keys.p, d_w.p, H, D, 1, n, d_last.p, d_ref.p, nullptr);
        glmd::IdxScoreWmma a;
        a.iq = d_iq.p; a.w = d_w.p; a.keys = d_keys.p; a.n_kv = n; a.score = d_s.p;
        expect(glmd::idx_score_simple(a, nullptr), "scorer simple: launch");
        if (kWmma) {
            a.score = d_wm.p;
            expect(glmd::idx_score_wmma(a, nullptr), "scorer WMMA: launch");
            a.keys = d_k16.p; a.key_f16 = 1; a.score = d_wm16.p;
            expect(glmd::idx_score_wmma(a, nullptr), "scorer WMMA FP16 keys: launch");
        } else {   // (HIP) the CUDA-core scorer on FP16 keys stands in: its top-2048 against the reference
            a.keys = d_k16.p; a.key_f16 = 1; a.score = d_wm16.p;
            expect(glmd::idx_score_simple(a, nullptr), "scorer simple FP16 keys: launch");
        }
        const auto ref = d_ref.get(), s = d_s.get(), wm = d_wm.get(), wm16 = d_wm16.get();
        // twins
        std::vector<float> ts(n), tw(n);
        glmd::IdxScoreWmma h;
        h.iq = iq.data(); h.w = w.data(); h.keys = keys.data(); h.n_kv = n; h.score = ts.data();
        glmd::vtwin::idx_score_simple(h);
        h.score = tw.data();
        glmd::vtwin::idx_score_wmma(h);
        char what[96];
        std::snprintf(what, sizeof what, "scorer simple GPU vs twin n %d (max err / max)", n);
        gate(what, max_rel(s, ts), 1e-5);
        if (!kWmma) {
            std::snprintf(what, sizeof what, "scorer simple F32 keys n %d: top-2048 miss rate", n);
            gate(what, miss_rate(ref, s, n, k, 1e-3), 1e-3);
            std::snprintf(what, sizeof what, "scorer simple FP16 keys n %d: top-2048 miss rate", n);
            gate(what, miss_rate(ref, wm16, n, k, 1e-3), 5e-3);
            continue;
        }
        std::snprintf(what, sizeof what, "scorer WMMA GPU vs twin n %d (max err / max)", n);
        gate(what, max_rel(wm, tw), 1e-3);
        std::snprintf(what, sizeof what, "scorer WMMA F32 keys n %d: top-2048 miss rate", n);
        gate(what, miss_rate(ref, wm, n, k, 1e-3), 1e-3);
        std::snprintf(what, sizeof what, "scorer WMMA FP16 keys n %d: top-2048 miss rate", n);
        gate(what, miss_rate(ref, wm16, n, k, 1e-3), 5e-3);
    }
    if (kWmma) {   // prefill: 5 tokens with positions (the WMMA scorer's prompt form)
        const int n = 8192, T = 5;
        const std::vector<int> pos = {0, 2047, 2048, 6000, 8191};
        std::vector<float> keys = randv(rng, (size_t) n * D), iq = randv(rng, (size_t) T * H * D),
                           w = randv(rng, (size_t) T * H, 1.0f / std::sqrt((float) H * D));
        Dev<float> d_keys(keys), d_iq(iq), d_w(w), d_ref((size_t) n * T), d_wm((size_t) n * T);
        Dev<int> d_pos(pos);
        glmd::ref_indexer_score(d_iq.p, d_keys.p, d_w.p, H, D, T, n, d_pos.p, d_ref.p, nullptr);
        glmd::IdxScoreWmma a;
        a.iq = d_iq.p; a.w = d_w.p; a.keys = d_keys.p; a.n_kv = n; a.T = T; a.pos = d_pos.p; a.score = d_wm.p;
        expect(glmd::idx_score_wmma(a, nullptr), "scorer WMMA prefill: launch");
        const auto ref = d_ref.get(), wm = d_wm.get();
        double worst = 0;
        bool masked = true;
        for (int t = 0; t < T; ++t) {
            for (int s = pos[t] + 1; s < n; ++s) masked &= std::isinf(wm[(size_t) t * n + s]) && wm[(size_t) t * n + s] < 0;
            std::vector<float> r(ref.begin() + (size_t) t * n, ref.begin() + (size_t) t * n + pos[t] + 1),
                tt(wm.begin() + (size_t) t * n, wm.begin() + (size_t) t * n + pos[t] + 1);
            worst = std::max(worst, miss_rate(r, tt, pos[t] + 1, k, 1e-3));
        }
        expect(masked, "scorer WMMA prefill: cells past each position are -inf");
        gate("scorer WMMA prefill 5 tokens: worst top-2048 miss rate", worst, 1e-3);
    }
    {   // the shape rule at its boundary: the smallest admitted D per head count runs against the twin; the overrun
        // shapes (f32 scratch > key tile) are refused by idx_score_wmma and served by the token scorer in idx_score
        const int n = glmd::kIdxWmmaMinDecode + 700;   // idx_score's decode auto mode would pick WMMA here
        for (const auto& hd : std::vector<std::pair<int, int>>{{16, 32}, {32, 64}, {16, 16}, {32, 32}, {32, 48}}) {
            const int h = hd.first, dd = hd.second;
            const bool ok = glmd::idx_score_wmma_supported(h, dd);
            char what[112];
            if (kWmma) {
                std::snprintf(what, sizeof what, "scorer WMMA %d x %d: supported == (2H <= D + 8)", h, dd);
                expect(ok == (2 * h <= dd + 8), what);
            } else {   // (HIP: every shape refused, served by the token scorer)
                std::snprintf(what, sizeof what, "scorer WMMA %d x %d: not supported in a HIP build", h, dd);
                expect(!ok, what);
            }
            std::vector<float> keys = randv(rng, (size_t) n * dd), iq = randv(rng, (size_t) h * dd),
                               w = randv(rng, h, 1.0f / std::sqrt((float) h * dd));
            Dev<float> d_keys(keys), d_iq(iq), d_w(w), d_s(n);
            glmd::IdxScoreWmma a;
            a.iq = d_iq.p; a.w = d_w.p; a.keys = d_keys.p; a.n_kv = n; a.score = d_s.p; a.idx_heads = h;
            a.idx_dim = dd;
            std::vector<float> tw(n);
            glmd::IdxScoreWmma t = a;
            t.iq = iq.data(); t.w = w.data(); t.keys = keys.data(); t.score = tw.data();
            glmd::vtwin::idx_score_wmma(t);
            if (ok) {
                std::snprintf(what, sizeof what, "scorer WMMA %d x %d: launch", h, dd);
                expect(glmd::idx_score_wmma(a, nullptr), what);
                std::snprintf(what, sizeof what, "scorer WMMA %d x %d GPU vs twin (max err / max)", h, dd);
                gate(what, max_rel(d_s.get(), tw), 1e-3);
            } else {
                // the refusal counts once in volta_launch_errors; the token scorer's launch adds nothing
                const int e0 = glmd::volta_launch_errors();
                std::snprintf(what, sizeof what, "scorer WMMA %d x %d: refused, counted once", h, dd);
                expect(!glmd::idx_score_wmma(a, nullptr) && glmd::volta_launch_errors() == e0 + 1, what);
                g_expected_volta_errors += 1;
                std::snprintf(what, sizeof what, "scorer %d x %d: idx_score falls back to the token scorer", h, dd);
                expect(glmd::idx_score(a, nullptr) && glmd::volta_launch_errors() == e0 + 1, what);
                std::vector<float> ts(n);
                t.score = ts.data();
                glmd::vtwin::idx_score_simple(t);
                std::snprintf(what, sizeof what, "scorer %d x %d fallback GPU vs twin (max err / max)", h, dd);
                gate(what, max_rel(d_s.get(), ts), 1e-5);
            }
        }
    }
    {   // B10.8: dsa_prep writes fp16(the F32 key)
        std::vector<float> inv(32);
        glmd::rope_inv_freq(inv.data(), 64, 8e6);
        Dev<float> d_inv(inv);
        const auto qr = randv(rng, kQLora), kv = randv(rng, kKvLora + kRope), ik = randv(rng, kIdxDim);
        std::vector<float> one(kQLora, 1.0f), zero(kIdxDim, 0.0f);
        Dev<float> d_qr(qr), d_kv(kv), d_ik(ik), d_one(one), d_zero(zero), d_ik32((size_t) 8 * kIdxDim);
        Dev<uint16_t> d_lat((size_t) 8 * kKvLora), d_kpe((size_t) 8 * kRope), d_ik16((size_t) 8 * kIdxDim);
        glmd::DsaPrep a;
        a.qr_raw = d_qr.p; a.q_a_norm = d_one.p; a.q_lora = kQLora; a.kv_raw = d_kv.p; a.kv_norm = d_one.p;
        a.lat = d_lat.p; a.kv_lora = kKvLora; a.k_pe = d_kpe.p; a.qk_rope = kRope; a.inv_freq = d_inv.p;
        a.ik_raw = d_ik.p; a.ik_ln_w = d_one.p; a.ik_ln_b = d_zero.p; a.ik_cache = d_ik32.p; a.idx_dim = kIdxDim;
        a.idx_rope = 64; a.idx_inv_freq = d_inv.p; a.p = 5;
        expect(glmd::dsa_prep(a, nullptr), "prep F32 key: launch");
        a.ik_cache = (float*) d_ik16.p;
        a.ik_f16 = 1;
        expect(glmd::dsa_prep(a, nullptr), "prep FP16 key: launch");
        const auto k32 = d_ik32.get();
        const auto k16 = d_ik16.get();
        bool same = true;
        for (int i = 0; i < kIdxDim; ++i) same &= k16[(size_t) 5 * kIdxDim + i] == glmd::twin::half_bits(k32[(size_t) 5 * kIdxDim + i]);
        expect(same, "prep FP16 key == fp16(F32 key) bit for bit (B10.8)");
    }
}

// ---------------------------------------------------------------- top-k
void test_topk(std::mt19937& rng) {
    const int k = glmd::kTopK;
    const int max_T = 4;
    Dev<uint8_t> d_work(glmd::topk_mb_work_bytes(max_T));
    expect(glmd::topk_mb_init(d_work.p, max_T, nullptr), "topk_mb: init");
    bool same = true;
    for (int n : {2049, 40000, 131072})
        for (int kind = 0; kind < 3; ++kind) {
            std::vector<float> sc(n);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::uniform_int_distribution<int> ud(0, 7);
            for (int i = 0; i < n; ++i)
                sc[i] = kind == 0 ? nd(rng) : kind == 1 ? (float) ud(rng)
                        : (i % 5 == 0 ? -std::numeric_limits<float>::infinity() : nd(rng));
            Dev<float> d_sc(sc);
            Dev<int> d_sel(k);
            std::vector<int> t(k);
            glmd::twin::topk(sc.data(), n, n, 1, nullptr, k, t.data());
            for (int rep = 0; rep < 3; ++rep) {   // the scratch must come back zeroed
                expect(glmd::topk_mb(d_sc.p, n, n, 1, nullptr, k, d_sel.p, d_work.p, nullptr), "topk_mb: launch");
                same &= d_sel.get() == t;
            }
        }
    expect(same, "topk_mb == twin topk exactly (n 2049 / 40000 / 131072, ties, -inf, 3 calls each)");
    {
        const int n = 50000, T = 4;
        const std::vector<int> pos = {10, 2047, 2048, 49999};
        std::vector<float> sc = randv(rng, (size_t) n * T);
        Dev<float> d_sc(sc);
        Dev<int> d_pos(pos), d_sel((size_t) k * T), d_one((size_t) k * T);
        std::vector<int> t((size_t) k * T);
        glmd::twin::topk(sc.data(), n, n, T, pos.data(), k, t.data());
        expect(glmd::topk_mb(d_sc.p, n, n, T, d_pos.p, k, d_sel.p, d_work.p, nullptr), "topk_mb rows: launch");
        expect(d_sel.get() == t, "topk_mb per-row device positions == twin topk");
        expect(glmd::topk(d_sc.p, n, n, T, d_pos.p, k, d_one.p, nullptr), "topk rows: launch");
        expect(d_one.get() == t, "topk (single block) per-row positions == twin topk");
        // timing at 131072 (B10.4 goal <= 0.1 ms, measured on the test PC)
        const int nl = 131072;
        Dev<float> d_l(randv(rng, nl));
        Dev<int> d_s(k);
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        float t_mb = 0, t_one = 0;
        for (int rep = 0; rep < 2; ++rep) {
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) glmd::topk_mb(d_l.p, nl, nl, 1, nullptr, k, d_s.p, d_work.p, nullptr);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&t_mb, e0, e1));
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) glmd::topk(d_l.p, nl, nl, 1, nullptr, k, d_s.p, nullptr);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&t_one, e0, e1));
        }
        std::printf("  top-2048 of 131072: multi-block %.1f us, single block %.1f us\n", t_mb * 50.0f, t_one * 50.0f);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
}

}  // namespace

int main() {
#ifdef _WIN32
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // (MSVC's CRT refuses _IOLBF with size 0)
#else
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // in step with the kernels' stderr notes (a refusal next to its check)
#endif
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("glmd_volta_parity: no CUDA device, skipped\n");
        return 77;
    }
    std::mt19937 rng(1012);
    test_headwise(rng);
    test_fused(rng);
    test_scorer(rng);
    test_topk(rng);
    char what[96];
    std::snprintf(what, sizeof what, "no launch errors beyond the %d shape refusals the test provokes",
                  g_expected_volta_errors);
    expect(glmd::volta_launch_errors() == g_expected_volta_errors && glmd::launch_errors() == 0, what);
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
