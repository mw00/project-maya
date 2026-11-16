// tests/glm_dsa/glmd_batch_parity.cu - the glm-dsa batched prompt kernels (glmd_batch.cu) against the token-path
// kernels (glmd_fast.cu) and the CPU twins (glm-dsa B9.2-B9.5, B9.8 clamp).
//   prep     T = 1 byte-identical to glmd::dsa_prep (both latent formats); T = 37, 4096 vs the T-row twin
//            (latent / k_pe / keys rel err < 1e-3);
//   select   index_score_tile + topk_merge_tile over 16K / 4K key blocks and 2048-query sub-batches equal glmd::topk
//            on the whole score row for EVERY row at 2,049 and 9,000 tokens (same scores: identical sets);
//   score    index_score_tile_wmma (CUDA's WMMA tile; on HIP gfx11 / gfx12 the RDNA one) vs the F32 scorer: rel L2
//            < 3e-3 on the causal cells and the tiled selections' mean overlap > 0.97 (F32 and FP16 keys, 9,048 /
//            16,256 cells, a query row past 65504);
//   mla      mla_prompt_f32 vs twin::mla_prompt rel L2 < 1e-4, mla_prompt_wmma and mla_prompt_h64 (CUDA) or
//            mla_prompt_rdna (HIP gfx11 / gfx12) < 2e-3 at 32, 1,032, 2,047, 2,049, a 9,000-, 16,384- and
//            40,016-cell context, the three walks, INT8 and FP16 latents, qk_rope 64 and 0; the WMMA attribute check
//            passes on the device (GLMD_BATCH_BENCH=1 adds kernel times: CUDA WMMA vs H64, HIP F32 vs RDNA;
//            GLMD_BATCH_BENCH=2: the kernel times alone);
//   route    route_rows == glmd::cpu::route_select (HF normalisation) on 512 random rows;
//   clamp    q8_mmq_clamp turns +-inf halves into +-65504 and leaves finite ones.
// Needs a GPU; exits 77 (ctest SKIP_RETURN_CODE) when there is none.
#include "strata/kernels/glmd_batch.hpp"
#include "strata/kernels/glmd_route.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

#if defined(STRATA_USE_HIP)
constexpr bool kWmma = false;   // the WMMA kernels (B9.5, H64, the WMMA score tile) are CUDA's: HIP runs the F32 ones
#else
constexpr bool kWmma = true;
#endif

int g_fail = 0;
void expect(bool ok, const char* what) {
    std::printf("%-80s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

template <typename T>
T* up(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T));
    if (!h.empty()) cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

std::mt19937 rng(17);
std::vector<float> randn(size_t n, float s = 1.0f) {
    std::normal_distribution<float> nd(0.0f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

// ---- prep
void test_prep(int T, bool q8) {
    const int QL = 2048, KV = 512, R = 64, ID = 128, LD = 576, p0 = 1000, ctx = p0 + T;
    auto qr_raw = randn((size_t) T * QL), kv = randn((size_t) T * LD), ik = randn((size_t) T * ID);
    auto qn = randn(QL), kn = randn(KV), lw = randn(ID), lb = randn(ID);
    std::vector<float> inv(R / 2);
    glmd::rope_inv_freq(inv.data(), R, 8.0e6);
    const size_t lat_bytes = q8 ? (size_t) glmd::lat8_rec_bytes(KV) * ctx : (size_t) KV * ctx * 2;
    float *d_qr_raw = up(qr_raw), *d_kv = up(kv), *d_ik = up(ik), *d_qn = up(qn), *d_kn = up(kn), *d_lw = up(lw),
          *d_lb = up(lb), *d_inv = up(inv);
    float *d_qr, *d_ikc, *d_qr1, *d_ikc1;
    uint16_t *d_kpe, *d_kpe1, *d_qr16;
    void *d_lat, *d_lat1;
    cudaMalloc(&d_qr, (size_t) T * QL * 4);
    cudaMalloc(&d_qr1, (size_t) T * QL * 4);
    cudaMalloc(&d_qr16, (size_t) T * QL * 2);
    cudaMalloc(&d_ikc, (size_t) ctx * ID * 4);
    cudaMalloc(&d_ikc1, (size_t) ctx * ID * 4);
    cudaMalloc(&d_kpe, (size_t) ctx * R * 2);
    cudaMalloc(&d_kpe1, (size_t) ctx * R * 2);
    cudaMalloc(&d_lat, lat_bytes);
    cudaMalloc(&d_lat1, lat_bytes);
    for (void* p : {(void*) d_ikc, (void*) d_ikc1, (void*) d_kpe, (void*) d_kpe1, d_lat, d_lat1})
        cudaMemset(p, 0, p == d_ikc || p == d_ikc1 ? (size_t) ctx * ID * 4 : p == d_kpe || p == d_kpe1 ? (size_t) ctx * R * 2 : lat_bytes);
    glmd::DsaPrep a;
    a.q_a_norm = d_qn;
    a.kv_norm = d_kn;
    a.lat_q8 = q8;
    a.inv_freq = d_inv;
    a.ik_ln_w = d_lw;
    a.ik_ln_b = d_lb;
    a.idx_inv_freq = d_inv;
    glmd::DsaPrepRows r;
    r.a = a;
    r.a.qr_raw = d_qr_raw;
    r.a.qr = d_qr;
    r.a.kv_raw = d_kv;
    r.a.lat = d_lat;
    r.a.k_pe = d_kpe;
    r.a.ik_raw = d_ik;
    r.a.ik_cache = d_ikc;
    r.a.p = p0;
    r.T = T;
    r.qr16 = d_qr16;
    glmd::dsa_prep_rows(r, nullptr);
    for (int t = 0; t < T; ++t) {   // the token path, one launch a token
        glmd::DsaPrep b = a;
        b.qr_raw = d_qr_raw + (size_t) t * QL;
        b.qr = d_qr1 + (size_t) t * QL;
        b.kv_raw = d_kv + (size_t) t * LD;
        b.lat = d_lat1;
        b.k_pe = d_kpe1;
        b.ik_raw = d_ik + (size_t) t * ID;
        b.ik_cache = d_ikc1;
        b.p = p0 + t;
        glmd::dsa_prep(b, nullptr);
    }
    cudaDeviceSynchronize();
    const bool same = down((uint8_t*) d_lat, lat_bytes) == down((uint8_t*) d_lat1, lat_bytes) &&
                      down(d_kpe, (size_t) ctx * R) == down(d_kpe1, (size_t) ctx * R) &&
                      down(d_ikc, (size_t) ctx * ID) == down(d_ikc1, (size_t) ctx * ID) &&
                      down(d_qr, (size_t) T * QL) == down(d_qr1, (size_t) T * QL);
    char what[160];
    std::snprintf(what, sizeof what, "prep rows T %d %s: identical bytes to the token path", T, q8 ? "INT8" : "FP16");
    expect(same, what);
    for (void* p : {(void*) d_qr_raw, (void*) d_kv, (void*) d_ik, (void*) d_qn, (void*) d_kn, (void*) d_lw, (void*) d_lb,
                    (void*) d_inv, (void*) d_qr, (void*) d_qr1, (void*) d_qr16, (void*) d_ikc, (void*) d_ikc1,
                    (void*) d_kpe, (void*) d_kpe1, d_lat, d_lat1})
        cudaFree(p);
}

// ---- select: the tiled sub-batched path vs glmd::topk on whole rows, same GPU scores
void test_select(int T, int key_block) {
    const int H = 32, D = 128, top = glmd::kTopK, q_sub = 2048;
    auto iq = randn((size_t) T * H * D), ik = randn((size_t) T * D, 0.3f), w = randn((size_t) T * H, 0.1f);
    float *d_iq = up(iq), *d_ik = up(ik), *d_w = up(w);
    // the token path: one score row of n = t + 1 per token (row stride T), glmd::topk with pos
    float* d_row;
    cudaMalloc(&d_row, (size_t) T * T * 4);
    glmd::index_score_tile(d_iq, d_w, d_ik, H, D, T, 0, T, d_row, T, nullptr);
    std::vector<int> pos((size_t) T);
    for (int t = 0; t < T; ++t) pos[(size_t) t] = t;
    int *d_pos = up(pos), *d_tok, *d_bat;
    cudaMalloc(&d_tok, (size_t) T * top * 4);
    cudaMalloc(&d_bat, (size_t) T * top * 4);
    glmd::topk(d_row, T, T, T, d_pos, top, d_tok, nullptr);
    // the batched path
    float *d_tile, *d_sa, *d_sb;
    int *d_ca, *d_cb, *d_na, *d_nb;
    cudaMalloc(&d_tile, (size_t) q_sub * key_block * 4);
    cudaMalloc(&d_sa, (size_t) q_sub * top * 4);
    cudaMalloc(&d_sb, (size_t) q_sub * top * 4);
    cudaMalloc(&d_ca, (size_t) q_sub * top * 4);
    cudaMalloc(&d_cb, (size_t) q_sub * top * 4);
    cudaMalloc(&d_na, (size_t) q_sub * 4);
    cudaMalloc(&d_nb, (size_t) q_sub * 4);
    for (int s0 = 0; s0 < T; s0 += q_sub) {
        const int tn = std::min(q_sub, T - s0);
        const int64_t n_kv = s0 + tn;
        const int tiles = (int) ((n_kv + key_block - 1) / key_block);
        const int *ci = nullptr, *ni = nullptr;
        const float* si = nullptr;
        for (int k = 0; k < tiles; ++k) {
            const int64_t k0 = (int64_t) k * key_block;
            const int nk = (int) std::min<int64_t>(key_block, n_kv - k0);
            glmd::index_score_tile(d_iq + (size_t) s0 * H * D, d_w + (size_t) s0 * H, d_ik, H, D, tn, k0, nk, d_tile,
                                   key_block, nullptr);
            glmd::TopkMerge m;
            m.score = d_tile;
            m.ld = key_block;
            m.T = tn;
            m.p0 = s0;
            m.k0 = k0;
            m.nk = nk;
            m.in_cell = ci;
            m.in_score = si;
            m.in_n = ni;
            m.out_cell = k == tiles - 1 ? d_bat + (size_t) s0 * top : (k & 1 ? d_cb : d_ca);
            m.out_score = k & 1 ? d_sb : d_sa;
            m.out_n = k & 1 ? d_nb : d_na;
            glmd::topk_merge_tile(m, nullptr);
            ci = m.out_cell;
            si = m.out_score;
            ni = m.out_n;
        }
    }
    cudaDeviceSynchronize();
    const auto a = down(d_tok, (size_t) T * top), b = down(d_bat, (size_t) T * top);
    int bad = 0;
    for (int t = 0; t < T; ++t) bad += std::memcmp(&a[(size_t) t * top], &b[(size_t) t * top], top * 4) != 0;
    char what[160];
    std::snprintf(what, sizeof what, "select T %d key_block %d: tiled == token path on every row (%d differ)", T,
                  key_block, bad);
    expect(bad == 0, what);
    for (void* p : {(void*) d_iq, (void*) d_ik, (void*) d_w, (void*) d_row, (void*) d_pos, (void*) d_tok, (void*) d_bat,
                    (void*) d_tile, (void*) d_sa, (void*) d_sb, (void*) d_ca, (void*) d_cb, (void*) d_na, (void*) d_nb})
        cudaFree(p);
}

// ---- the WMMA scorer vs the F32 one: scores (rel L2 over the causal cells), and the selection both give through the
// tiled merge (mean / min overlap of the 2,048-cell sets); keys F32 or FP16; a query row past 65504 stays finite
void test_score_wmma(int T, int64_t p0, bool kf16, bool big) {
    const int H = 32, D = 128, top = glmd::kTopK, kb = 4096;
    const int64_t n_kv = p0 + T;
    auto iq = randn((size_t) T * H * D), w = randn((size_t) T * H, 0.1f), ik = randn((size_t) n_kv * D, 0.3f);
    if (big) iq[((size_t) 3 * H + 5) * D + 7] = 2.0e5f;   // row 3, head 5: past FP16
    std::vector<uint16_t> ik16((size_t) n_kv * D);
    for (size_t i = 0; i < ik.size(); ++i) {
        ik16[i] = glmd::twin::half_bits(ik[i]);
        if (kf16) ik[i] = glmd::twin::half_float(ik16[i]);   // both scorers read the same key values
    }
    float* d_iq = up(iq);
    float* d_w = up(w);
    const void* d_keys = kf16 ? (const void*) up(ik16) : (const void*) up(ik);
    float *d_t32, *d_t16, *d_s;
    int *d_c, *d_n, *d_sel32, *d_sel16;
    cudaMalloc(&d_t32, (size_t) T * kb * 4);
    cudaMalloc(&d_t16, (size_t) T * kb * 4);
    cudaMalloc(&d_s, (size_t) 2 * T * top * 4);
    cudaMalloc(&d_c, (size_t) 2 * T * top * 4);
    cudaMalloc(&d_n, (size_t) 2 * T * 4);
    cudaMalloc(&d_sel32, (size_t) T * top * 4);
    cudaMalloc(&d_sel16, (size_t) T * top * 4);
    const int tiles = (int) ((n_kv + kb - 1) / kb);
    double num = 0.0, den = 0.0;
    bool finite = true, ran = true;
    for (int pass = 0; pass < 2; ++pass) {
        const int *ci = nullptr, *ni = nullptr;
        const float* si = nullptr;
        for (int k = 0; k < tiles; ++k) {
            const int64_t k0 = (int64_t) k * kb;
            const int nk = (int) std::min<int64_t>(kb, n_kv - k0);
            float* tile = pass ? d_t16 : d_t32;
            glmd::index_score_tile_keys(d_iq, d_w, d_keys, kf16, H, D, T, k0, nk, d_t32, kb, nullptr);
            if (pass) {   // the WMMA tile, compared on the causal cells with the F32 one
                ran = glmd::index_score_tile_wmma(d_iq, d_w, d_keys, kf16, H, D, T, k0, nk, d_t16, kb, nullptr) && ran;
                const auto a = down(d_t32, (size_t) T * kb), b = down(d_t16, (size_t) T * kb);
                for (int t = 0; t < T; ++t)
                    for (int j = 0; j < nk && k0 + j <= p0 + t; ++j) {
                        const double x = a[(size_t) t * kb + j], y = b[(size_t) t * kb + j];
                        finite = finite && std::isfinite(y);
                        num += (x - y) * (x - y);
                        den += x * x;
                    }
            }
            glmd::TopkMerge m;
            m.score = tile;
            m.ld = kb;
            m.T = T;
            m.p0 = p0;
            m.k0 = k0;
            m.nk = nk;
            m.in_cell = ci;
            m.in_score = si;
            m.in_n = ni;
            m.out_cell = k == tiles - 1 ? (pass ? d_sel16 : d_sel32) : d_c + (size_t) (k & 1) * T * top;
            m.out_score = d_s + (size_t) (k & 1) * T * top;
            m.out_n = d_n + (size_t) (k & 1) * T;
            glmd::topk_merge_tile(m, nullptr);
            ci = m.out_cell;
            si = m.out_score;
            ni = m.out_n;
        }
    }
    cudaDeviceSynchronize();
    const auto s32 = down(d_sel32, (size_t) T * top), s16 = down(d_sel16, (size_t) T * top);
    double ov_sum = 0.0, ov_min = 1.0;
    for (int t = 0; t < T; ++t) {
        const int* a = &s32[(size_t) t * top];
        const int* b = &s16[(size_t) t * top];
        int na = 0, common = 0, i = 0, j = 0;
        while (na < top && a[na] >= 0) ++na;
        int nb = 0;
        while (nb < top && b[nb] >= 0) ++nb;
        while (i < na && j < nb) {   // both ascending
            if (a[i] == b[j]) { ++common; ++i; ++j; }
            else if (a[i] < b[j]) ++i;
            else ++j;
        }
        const double ov = na ? (double) common / na : 1.0;
        ov_sum += ov;
        ov_min = std::min(ov_min, ov);
    }
    const double rel = den > 0 ? std::sqrt(num / den) : 0.0, ov_mean = ov_sum / T;
    char what[220];
    std::snprintf(what, sizeof what,
                  "score wmma T %d p0 %lld keys %s%s: rel L2 %.2e (< 3e-3), selection overlap mean %.4f min %.4f (> 0.97)",
                  T, (long long) p0, kf16 ? "FP16" : "F32", big ? " big row" : "", rel, ov_mean, ov_min);
    expect(ran && finite && rel < 3e-3 && ov_mean > 0.97, what);
    for (const void* p : {(const void*) d_iq, (const void*) d_w, d_keys, (const void*) d_t32, (const void*) d_t16,
                          (const void*) d_s, (const void*) d_c, (const void*) d_n, (const void*) d_sel32,
                          (const void*) d_sel16})
        cudaFree((void*) p);
}

// ---- prompt MLA
// synthetic GLM-5.3 caches and a selection for queries at p0 .. p0 + T - 1 (n_kv = p0 + T cells)
struct MlaData {
    std::vector<float> qa, q;
    std::vector<uint16_t> lat16, kpe;
    std::vector<uint8_t> lat8;
    std::vector<int> sel;
    std::vector<uint32_t> mask;
    int64_t words = 0;
};
MlaData mla_data(int T, int64_t p0, float q_sd = 0.5f) {
    const int H = 64, K = 512, R = 64, top = glmd::kTopK;
    const int64_t n_kv = p0 + T;
    MlaData m;
    m.qa = randn((size_t) T * H * K, q_sd);
    m.q = randn((size_t) T * H * 256);
    m.lat16.resize((size_t) n_kv * K);
    m.kpe.resize((size_t) n_kv * R);
    m.lat8.resize((size_t) n_kv * glmd::lat8_rec_bytes(K));
    {
        auto l = randn((size_t) n_kv * K);
        for (size_t i = 0; i < l.size(); ++i) m.lat16[i] = glmd::twin::half_bits(l[i]);
        for (int64_t c = 0; c < n_kv; ++c) {   // INT8 records: scale 1/127 per 32, codes from the FP16 values
            uint8_t* rec = m.lat8.data() + (size_t) c * glmd::lat8_rec_bytes(K);
            for (int g = 0; g < K / 32; ++g) {
                const uint16_t d = glmd::twin::half_bits(3.0f / 127.0f);
                std::memcpy(rec + K + 2 * g, &d, 2);
                for (int j = 0; j < 32; ++j)
                    rec[32 * g + j] = (uint8_t) (int8_t) std::max(-127, std::min(127, (int) std::lround(l[(size_t) c * K + 32 * g + j] * 127.0f / 3.0f)));
            }
        }
        auto kp = randn((size_t) n_kv * R);
        for (size_t i = 0; i < kp.size(); ++i) m.kpe[i] = glmd::twin::half_bits(kp[i]);
    }
    // a selection: random scores through the twin
    m.sel.resize((size_t) T * top);
    struct G { uint32_t s; } g{(uint32_t) (p0 * 7 + T)};
    glmd::twin::topk_tiled(
        [](void* u, int64_t pos, int64_t cell) -> float {
            uint64_t h = ((uint64_t) ((G*) u)->s << 40) ^ ((uint64_t) pos << 20) ^ (uint64_t) cell;
            h *= 0x9e3779b97f4a7c15ULL;
            return (float) (h >> 40);
        },
        &g, T, p0, 4096, top, m.sel.data());
    m.words = (n_kv + 31) / 32;
    m.mask.assign((size_t) T * m.words, 0u);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < top; ++i)
            if (const int c = m.sel[(size_t) t * top + i]; c >= 0) m.mask[(size_t) t * m.words + c / 32] |= 1u << (c % 32);
    return m;
}

void test_mla(int T, int64_t p0, glmd::AttnWalk walk, bool q8, int R = 64) {
    const int H = 64, K = 512, top = glmd::kTopK;
    const int64_t n_kv = p0 + T;
    MlaData m = mla_data(T, p0);
    auto& qa = m.qa;
    auto& q = m.q;
    auto& lat16 = m.lat16;
    auto& lat8 = m.lat8;
    auto& kpe = m.kpe;
    auto& sel = m.sel;
    auto& mask = m.mask;
    const int64_t words = m.words;
    glmd::MlaPrompt a;
    a.walk = walk;
    a.top_k = top;
    a.p0 = p0;
    a.T = T;
    a.n_head = H;
    a.kv_lora = K;
    a.qk_rope = R;
    a.lat_q8 = q8;
    a.mask_words = words;
    std::vector<float> ref((size_t) T * H * K);
    a.q_abs = qa.data();
    a.q = q.data();
    a.lat = q8 ? (const void*) lat8.data() : (const void*) lat16.data();
    a.k_pe = kpe.data();
    a.sel = sel.data();
    a.mask = mask.data();
    a.ctx = ref.data();
    glmd::twin::mla_prompt(a);
    glmd::MlaPrompt d = a;
    d.q_abs = up(qa);
    d.q = up(q);
    d.lat = q8 ? (const void*) up(lat8) : (const void*) up(lat16);
    d.k_pe = up(kpe);
    d.sel = up(sel);
    uint32_t* d_mask;
    cudaMalloc(&d_mask, mask.size() * 4);
    glmd::sel_to_mask(d.sel, top, T, n_kv, d_mask, words, nullptr);
    d.mask = d_mask;
    float* d_ctx;
    cudaMalloc(&d_ctx, ref.size() * 4);
    d.ctx = d_ctx;
    glmd::mla_prompt_f32(d, nullptr);
    cudaDeviceSynchronize();
    const double e32 = rel_l2(down(d_ctx, ref.size()), ref);
    const bool mask_ok = down(d_mask, mask.size()) == mask;
    bool wm = true, h64 = true;
    double e16 = 0.0, e64 = 0.0;
    if (kWmma) {
        cudaMemset(d_ctx, 0, ref.size() * 4);
        wm = glmd::mla_prompt_wmma(d, nullptr);
        cudaDeviceSynchronize();
        e16 = rel_l2(down(d_ctx, ref.size()), ref);
        cudaMemset(d_ctx, 0, ref.size() * 4);
        h64 = glmd::mla_prompt_h64_fits(d) && glmd::mla_prompt_h64(d, nullptr);
        cudaDeviceSynchronize();
        e64 = rel_l2(down(d_ctx, ref.size()), ref);
    }
    // HIP (S7): the RDNA WMMA kernel where the device takes it (gfx11 / gfx12), at the WMMA kernels' bound
    bool rd = true;
    double erd = 0.0;
    const bool rd_dev = !kWmma && glmd::mla_prompt_rdna_ok();
    if (rd_dev) {
        cudaMemset(d_ctx, 0, ref.size() * 4);
        rd = glmd::mla_prompt_rdna(d, nullptr);
        cudaDeviceSynchronize();
        erd = rel_l2(down(d_ctx, ref.size()), ref);
    }
    char what[220];
    if (kWmma)
        std::snprintf(what, sizeof what,
                      "mla T %d p0 %lld walk %d %s R %d: f32 rel L2 %.2e (< 1e-4), wmma %.2e, h64 %.2e (< 2e-3)", T,
                      (long long) p0, (int) walk, q8 ? "INT8" : "FP16", R, e32, e16, e64);
    else if (rd_dev)
        std::snprintf(what, sizeof what, "mla T %d p0 %lld walk %d %s R %d: f32 rel L2 %.2e (< 1e-4), rdna %.2e (< 2e-3)",
                      T, (long long) p0, (int) walk, q8 ? "INT8" : "FP16", R, e32, erd);
    else
        std::snprintf(what, sizeof what, "mla T %d p0 %lld walk %d %s R %d: f32 rel L2 %.2e (< 1e-4) (no WMMA here)",
                      T, (long long) p0, (int) walk, q8 ? "INT8" : "FP16", R, e32);
    expect(mask_ok && e32 < 1e-4 && wm && e16 < 2e-3 && h64 && e64 < 2e-3 && rd && erd < 2e-3, what);
    for (const void* p : {(const void*) d.q_abs, (const void*) d.q, d.lat, (const void*) d.k_pe, (const void*) d.sel,
                          (const void*) d_mask, (const void*) d_ctx})
        cudaFree((void*) p);
}

// GLMD_BATCH_BENCH=1: kernel times of the prompt MLA for the last 2,048 queries of an n_kv context (INT8 cache, the
// GLM-5.3 shape), the 16-head WMMA kernel and H64, each walk the context allows - the B14.12 crossover on this device.
// Not a pass / fail check (ctest does not set the variable).
void bench_mla() {
    const int T = 2048, top = glmd::kTopK;
    std::printf("bench: prompt MLA, %d queries (the last of the context), INT8 latent, ms (best of 3)\n", T);
    for (int64_t n_kv : {2048, 3072, 5120, 8192, 16384, 32768}) {
        const int64_t p0 = n_kv - T;
        MlaData m = mla_data(T, p0);
        glmd::MlaPrompt d;
        d.top_k = top;
        d.p0 = p0;
        d.T = T;
        d.lat_q8 = 1;
        d.mask_words = m.words;
        d.scale = 0.0625f;
        d.q_abs = up(m.qa);
        d.q = up(m.q);
        d.lat = up(m.lat8);
        d.k_pe = up(m.kpe);
        d.sel = up(m.sel);
        d.mask = up(m.mask);
        float* d_ctx;
        cudaMalloc(&d_ctx, (size_t) T * 64 * 512 * 4);
        d.ctx = d_ctx;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const auto ms = [&](int k) {   // 1 the 16-head WMMA kernel, 2 H64, 3 RDNA (HIP), 0 F32
            float best = 1e30f;
            for (int r = 0; r < 4; ++r) {   // the first is a warm-up
                cudaEventRecord(e0);
                if (k == 2) glmd::mla_prompt_h64(d, nullptr);
                else if (k == 1) glmd::mla_prompt_wmma(d, nullptr);
                else if (k == 3) glmd::mla_prompt_rdna(d, nullptr);
                else glmd::mla_prompt_f32(d, nullptr);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float t = 0.0f;
                cudaEventElapsedTime(&t, e0, e1);
                if (r > 0) best = std::min(best, t);
            }
            return best;
        };
        for (glmd::AttnWalk w : {glmd::AttnWalk::Dense, glmd::AttnWalk::DenseMasked, glmd::AttnWalk::Gather}) {
            if ((w == glmd::AttnWalk::Dense) != (n_kv <= top)) continue;
            d.walk = w;
            const char* wn = w == glmd::AttnWalk::Dense ? "dense" : w == glmd::AttnWalk::Gather ? "gather" : "dense-masked";
            if (kWmma) {
                const float t16 = ms(1), t64 = ms(2);
                std::printf("  n_kv %6lld walk %-11s wmma16 %8.1f  h64 %8.1f  (x%.2f)\n", (long long) n_kv, wn, t16, t64,
                            t16 / t64);
            } else {   // HIP: the F32 kernel against the RDNA WMMA one
                const float t32 = ms(0), trd = ms(3);
                std::printf("  n_kv %6lld walk %-11s f32 %8.1f  rdna %8.1f  (x%.2f)\n", (long long) n_kv, wn, t32, trd,
                            t32 / trd);
            }
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        for (const void* p : {(const void*) d.q_abs, (const void*) d.q, d.lat, (const void*) d.k_pe, (const void*) d.sel,
                              (const void*) d.mask, (const void*) d_ctx})
            cudaFree((void*) p);
    }
}

// GLMD_BATCH_BENCH=1: one 16,384-key score tile for 2,048 queries (GLM-5.3's indexer shape), F32 vs WMMA, ms
void bench_score() {
    const int T = 2048, H = 32, D = 128, nk = 16384;
    auto iq = randn((size_t) T * H * D), w = randn((size_t) T * H, 0.1f), ik = randn((size_t) nk * D, 0.3f);
    float *d_iq = up(iq), *d_w = up(w), *d_ik = up(ik), *d_t;
    cudaMalloc(&d_t, (size_t) T * nk * 4);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best[2] = {1e30f, 1e30f};
    for (int r = 0; r < 4; ++r)
        for (int k = 0; k < 2; ++k) {
            cudaEventRecord(e0);
            if (k) glmd::index_score_tile_wmma(d_iq, d_w, d_ik, 0, H, D, T, 0, nk, d_t, nk, nullptr);
            else glmd::index_score_tile_keys(d_iq, d_w, d_ik, 0, H, D, T, 0, nk, d_t, nk, nullptr);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float t = 0.0f;
            cudaEventElapsedTime(&t, e0, e1);
            if (r > 0) best[k] = std::min(best[k], t);
        }
    std::printf("bench: score tile %d queries x %d keys (32 heads x 128): f32 %.2f ms, wmma %.2f ms (x%.2f)\n", T, nk,
                best[0], best[1], best[0] / best[1]);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    for (const void* p : {(const void*) d_iq, (const void*) d_w, (const void*) d_ik, (const void*) d_t}) cudaFree((void*) p);
}

void test_route() {
    const int T = 512, E = 256, K = 8;
    auto lg = randn((size_t) T * E, 2.0f), bias = randn(E, 0.1f);
    float *d_lg = up(lg), *d_b = up(bias), *d_w;
    int* d_ids;
    cudaMalloc(&d_w, (size_t) T * K * 4);
    cudaMalloc(&d_ids, (size_t) T * K * 4);
    glmd::route_rows(d_lg, d_b, E, K, 2.5f, true, glmd::kRouteNormHf, T, d_ids, d_w, nullptr);
    cudaDeviceSynchronize();
    const auto ids = down(d_ids, (size_t) T * K);
    const auto w = down(d_w, (size_t) T * K);
    int bad = 0;
    double maxe = 0.0;
    for (int t = 0; t < T; ++t) {
        int ri[8];
        float rw[8];
        glmd::cpu::route_select(&lg[(size_t) t * E], bias.data(), E, K, 2.5f, true, glmd::kRouteNormHf, ri, rw);
        for (int i = 0; i < K; ++i) {
            bad += ri[i] != ids[(size_t) t * K + i];
            maxe = std::max(maxe, (double) std::fabs(rw[i] - w[(size_t) t * K + i]) / std::max(1e-6f, std::fabs(rw[i])));
        }
    }
    expect(bad == 0 && maxe < 1e-5, "route rows == cpu::route_select (HF): ids exact, weights rel < 1e-5");
    cudaFree(d_lg);
    cudaFree(d_b);
    cudaFree(d_w);
    cudaFree(d_ids);
}

void test_clamp() {
    std::vector<uint16_t> blk(72 * 3, 0x3c00);   // 3 blocks of 144 B; first 8 halves of each are d / sum
    blk[0] = 0x7c00;                             // +inf
    blk[72 + 3] = 0xfc00;                        // -inf
    uint16_t* d = up(blk);
    glmd::q8_mmq_clamp(d, 3, 144, 8, nullptr);
    const auto h = down(d, blk.size());
    expect(h[0] == 0x7bff && h[72 + 3] == 0xfbff && h[1] == 0x3c00 && h[8] == 0x3c00,
           "q8_mmq_clamp: +-inf -> +-65504 in the scale halves, the rest untouched");
    cudaFree(d);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    if (const char* b = std::getenv("GLMD_BATCH_BENCH"); b && b[0] == '2') {   // the benches alone (kernel work)
        if (kWmma || glmd::index_score_wmma_fits(0, 32, 128)) bench_score();
        if (kWmma || glmd::mla_prompt_rdna_ok()) bench_mla();
        return 0;
    }
    test_prep(1, false);
    test_prep(1, true);
    test_prep(37, false);
    test_prep(37, true);
    test_prep(4096, false);
    test_select(2049, 16384);
    test_select(2049, 1024);
    test_select(9000, 4096);
    if (kWmma) {
        test_score_wmma(2048, 7000, false, false);
        test_score_wmma(256, 16000, true, false);
        test_score_wmma(64, 3000, false, true);
        expect(glmd::mla_prompt_wmma_ok(), "mla wmma: opt-in shared memory and attributes check out");
    } else {
        expect(!glmd::mla_prompt_wmma_ok(), "mla wmma: not in a HIP build (the prompt path runs the RDNA or F32 kernel)");
        if (glmd::index_score_wmma_fits(0, 32, 128)) {   // HIP on gfx11 / gfx12: the RDNA scorer (S7), the same bounds
            test_score_wmma(2048, 7000, false, false);
            test_score_wmma(256, 16000, true, false);
            test_score_wmma(64, 3000, false, true);
        }
    }
    test_mla(32, 0, glmd::AttnWalk::Dense, false);
    test_mla(64, 1983, glmd::AttnWalk::Dense, false);
    test_mla(64, 1985, glmd::AttnWalk::DenseMasked, false);
    test_mla(64, 1985, glmd::AttnWalk::Gather, true);
    test_mla(48, 8952, glmd::AttnWalk::Gather, false);
    test_mla(48, 8952, glmd::AttnWalk::DenseMasked, true);
    // GLM-5.3 shapes for the 64-head kernel: 1K dense INT8, no rope part, 16K / 40K gather (INT8, FP16), short
    // tiles (T 37 at p0 0: rows of 1..37 cells)
    test_mla(32, 1000, glmd::AttnWalk::Dense, true);
    test_mla(37, 0, glmd::AttnWalk::Dense, true);
    test_mla(32, 1000, glmd::AttnWalk::Dense, false, 0);
    test_mla(24, 16360, glmd::AttnWalk::Gather, true);
    test_mla(16, 40000, glmd::AttnWalk::Gather, true);
    test_mla(16, 40000, glmd::AttnWalk::Gather, false);
    if (const char* b = std::getenv("GLMD_BATCH_BENCH"); b && b[0] == '1' && (kWmma || glmd::mla_prompt_rdna_ok())) {
        if (kWmma || glmd::index_score_wmma_fits(0, 32, 128)) bench_score();   // (HIP: the RDNA scorer)
        bench_mla();
    }
    test_route();
    test_clamp();
    expect(glmd::batch_launch_errors() == 0, "no launch errors or refused shapes");
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
