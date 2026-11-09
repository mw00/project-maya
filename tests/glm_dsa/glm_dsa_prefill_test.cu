// tests/glm_dsa/glm_dsa_prefill_test.cu - B9.10: the batched glm-dsa prompt layer (glmpf::dsa_layer, the code the
// prompt path runs) against a token-by-token composition of the decode kernels (glmd::dsa_prep, dsa_rope_q, the
// indexer score row, glmd::topk, glmd::mla2) on a synthetic two-layer model (one full-indexer layer, one shared layer
// reusing its selection), GLM-5.3's attention widths (64 heads - the prompt attention runs mla_prompt_h64 -, kv_lora
// 512, qk 192 + 64, idx 128 / rope 64, top-2048) with a small n_embd.  Prompts of 32 and 2,100 tokens (the
// crossover: dense, then dense-masked / gather walks), and a
// prompt in two chunks (1,500 + 600: the second chunk's queries see the first chunk's caches).  Gates: every layer's
// mixer rows rel L2 < 2e-3 (FP16 projections / products vs the F32 token path), the latent / k_pe / index-key caches
// equal within FP16 rounding, every row's selection sharing > 99% of its cells with the token path's (the prompt path's
// WMMA scorer rounds products to FP16: near-ties may swap).
// Needs a GPU; exits 77 (ctest SKIP_RETURN_CODE) when there is none.
#include "glm_prefill_dsa.hpp"
#include "strata/core/glm_state_layout.hpp"
#include "strata/kernels/glmd_batch.hpp"
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_forward.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include "glm_test_env.hpp"   // setenv / unsetenv (<unistd.h> on POSIX; mapped on Windows)

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::core;

namespace {

int g_fail = 0;
void expect(bool ok, const char* what) {
    std::printf("%-96s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

std::mt19937 rng(23);
std::vector<float> randn(size_t n, float s) {
    std::normal_distribution<float> nd(0.0f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}
template <typename T>
T* up(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
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
uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

constexpr int E = 256, H = 64, QL = 64, KV = 512, NOPE = 192, R = 64, QH = 256, VH = 64, IH = 4, ID = 128, TOP = 2048;
constexpr int KVR = KV + R;

struct Weights {
    WSlot q_a, q_b, kv_a, k_b, v_b, out, idx_k, idx_q_b, idx_proj;
    float *q_a_norm, *kv_a_norm, *ik_w, *ik_b;
};

WSlot f32_slot(int n_in, int n_out, float s) {
    WSlot w;
    w.type = 0;
    w.f32 = up(randn((size_t) n_in * n_out, s));
    w.q = w.f32;
    w.n_in = n_in;
    w.n_out = n_out;
    return w;
}
WSlot bf16_slot(int n_in, int n_out, float s) {
    const auto f = randn((size_t) n_in * n_out, s);
    std::vector<uint16_t> b(f.size());
    for (size_t i = 0; i < f.size(); ++i) b[i] = bf16(f[i]);
    WSlot w;
    w.type = 30;
    w.q = up(b);
    w.n_in = n_in;
    w.n_out = n_out;
    return w;
}
Weights make_layer(bool full) {
    Weights w{};
    w.q_a = f32_slot(E, QL, 1.0f / 16.0f);
    w.q_b = f32_slot(QL, H * QH, 1.0f / 8.0f);
    w.kv_a = f32_slot(E, KVR, 1.0f / 16.0f);
    w.k_b = bf16_slot(NOPE, H * KV, 1.0f / 14.0f);
    w.v_b = bf16_slot(KV, H * VH, 1.0f / 22.0f);
    w.out = f32_slot(H * VH, E, 1.0f / 32.0f);
    if (full) {
        w.idx_k = f32_slot(E, ID, 1.0f / 16.0f);
        w.idx_q_b = f32_slot(QL, IH * ID, 1.0f / 8.0f);
        w.idx_proj = f32_slot(E, IH, 1.0f / 16.0f);
        w.ik_w = up(randn(ID, 0.2f));
        w.ik_b = up(randn(ID, 0.1f));
    }
    auto ones = [](int n) {
        std::vector<float> v((size_t) n);
        for (auto& x : v) x = 1.0f + 0.1f * std::normal_distribution<float>(0.0f, 1.0f)(rng);
        return v;
    };
    w.q_a_norm = up(ones(QL));
    w.kv_a_norm = up(ones(KV));
    return w;
}

Glm5Geometry geometry() {
    Glm5Geometry g;
    g.arch = GlmArch::GlmDsa;
    g.n_layers = 2;
    g.nextn = 0;
    g.n_embd = E;
    g.n_head = H;
    g.q_lora = QL;
    g.kv_lora = KV;
    g.qk_nope = NOPE;
    g.qk_rope = R;
    g.v_head = VH;
    g.idx_heads = IH;
    g.idx_key = ID;
    g.idx_rope = 64;
    g.idx_top_k = TOP;
    g.idx_kpool = 1;
    g.idx_select_tail = 0;
    g.hc = 1;
    g.norm_eps = 1e-5f;
    g.idx_ln_eps = 1e-6f;
    g.rope_theta = 8.0e6;
    g.n_head_kv = {1, 1};
    g.indexer_full = {1, 0, 1};
    return g;
}

// y[t][N] = W[N][K] x[t] (F32 cuBLAS, the token path's GEMVs)
void sgemv(cublasHandle_t h, const WSlot& w, int N, int K, const float* x, float* y, float alpha = 1.0f) {
    const float zero = 0.0f;
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K, &alpha, w.f32, K, x, K, &zero, y, N);
}

struct Run {
    std::vector<float> mixer[2];   // per layer, all chunks' rows
    std::vector<int> sel;          // the full layer's selection per token (TOP each)
    std::vector<uint16_t> lat0, kpe0;
    std::vector<float> ik0;
};

// the batched path: dsa_layer per chunk and layer
Run batched(const Glm5Geometry& g, const GlmStateLayout& lay, glmdsa::GlmDsaFast& dsa, const std::vector<float>& x,
            const std::vector<int>& chunks, cublasHandle_t blas) {
    const int Tall = (int) (x.size() / E);
    Run r;
    float* state = nullptr;
    cudaMalloc(&state, lay.bytes);
    cudaMemset(state, 0, lay.bytes);
    int64_t p0 = 0;
    for (int T : chunks) {
        const int sub = std::min(T, glmpf::kSubBatch);
        const size_t ub = glmpf::scratch_bytes(g, (size_t) sub, lay.max_ctx), kb = glmpf::keep_bytes(g, (size_t) T, lay.max_ctx);
        uint8_t *uni, *keep;
        uint16_t* w16;
        float *w32, *dx, *mixer;
        uint16_t* dx16;
        cudaMalloc(&uni, ub);
        cudaMalloc(&keep, kb);
        const int64_t w16n = (int64_t) H * KV * NOPE, w32n = (int64_t) H * KV * NOPE;
        cudaMalloc(&w16, (size_t) w16n * 2);
        cudaMalloc(&w32, (size_t) w32n * 4);
        cudaMalloc(&dx, (size_t) T * E * 4);
        cudaMalloc(&dx16, (size_t) T * E * 2);
        cudaMalloc(&mixer, (size_t) T * E * 4);
        cudaMemcpy(dx, x.data() + (size_t) p0 * E, (size_t) T * E * 4, cudaMemcpyHostToDevice);
        glmd::pack_f16(dx, E, E, 1, E, T, dx16, nullptr);
        glmpf::Io io;
        io.g = &g;
        io.s = nullptr;
        io.blas = blas;
        io.w16 = w16;
        io.w16_elems = w16n;
        io.w32 = w32;
        io.w32_elems = w32n;
        io.uni = uni;
        io.keep = keep;
        io.x = dx;
        io.x16 = dx16;
        io.mixer = mixer;
        io.T = T;
        io.p0 = p0;
        io.state = state;
        io.lay = &lay;
        io.max_ctx = lay.max_ctx;
        io.dsa = &dsa;
        io.sub = sub;
        for (int il = 0; il < 2; ++il) {
            std::string err;
            if (!glmpf::dsa_layer(io, il, err)) std::printf("dsa_layer %d: %s\n", il, err.c_str());
            const auto m = down(mixer, (size_t) T * E);
            r.mixer[il].insert(r.mixer[il].end(), m.begin(), m.end());
            if (il == 0) {
                // the selection rows the shared layer read (identity rows where the walk was dense)
                const auto s = down((const int*) keep, (size_t) T * TOP);
                for (int t = 0; t < T; ++t) {
                    const int64_t n = p0 + t + 1;
                    for (int i = 0; i < TOP; ++i)
                        r.sel.push_back(n <= TOP ? (i < n ? i : -1) : s[(size_t) t * TOP + i]);
                }
            }
        }
        for (void* p : {(void*) uni, (void*) keep, (void*) w16, (void*) w32, (void*) dx, (void*) dx16, (void*) mixer})
            cudaFree(p);
        p0 += T;
    }
    cudaDeviceSynchronize();
    r.lat0 = down((const uint16_t*) (state + lay.dsa_lat[0]), (size_t) Tall * KV);
    r.kpe0 = down((const uint16_t*) (state + lay.dsa_kpe[0]), (size_t) Tall * R);
    r.ik0 = down(state + lay.dsa_ik[0], (size_t) Tall * ID);
    cudaFree(state);
    return r;
}

// the token path: one position at a time through the decode kernels, F32 GEMVs
Run token(const GlmStateLayout& lay, const Weights* W, const std::vector<float>& x, cublasHandle_t blas) {
    const int Tall = (int) (x.size() / E);
    Run r;
    for (auto& m : r.mixer) m.assign((size_t) Tall * E, 0.0f);
    r.sel.assign((size_t) Tall * TOP, -1);
    float* state = nullptr;
    cudaMalloc(&state, lay.bytes);
    cudaMemset(state, 0, lay.bytes);
    std::vector<float> inv(R / 2);
    glmd::rope_inv_freq(inv.data(), R, 8.0e6);
    float* d_inv = up(inv);
    float *dx, *qr_raw, *qr, *kv, *ik, *iw, *q, *iq, *row, *out, *mix;
    int* sel;
    cudaMalloc(&dx, E * 4);
    cudaMalloc(&qr_raw, QL * 4);
    cudaMalloc(&qr, QL * 4);
    cudaMalloc(&kv, KVR * 4);
    cudaMalloc(&ik, ID * 4);
    cudaMalloc(&iw, IH * 4);
    cudaMalloc(&q, H * QH * 4);
    cudaMalloc(&iq, IH * ID * 4);
    cudaMalloc(&row, (size_t) Tall * 4);
    cudaMalloc(&out, H * VH * 4);
    cudaMalloc(&mix, E * 4);
    cudaMalloc(&sel, (size_t) Tall * TOP * 4);
    const size_t wb = glmd::mla2_work_bytes(H, KV, TOP);
    void* work;
    cudaMalloc(&work, wb);
    for (int t = 0; t < Tall; ++t) {
        cudaMemcpy(dx, x.data() + (size_t) t * E, E * 4, cudaMemcpyHostToDevice);
        for (int il = 0; il < 2; ++il) {
            const Weights& w = W[il];
            const bool full = il == 0;
            sgemv(blas, w.q_a, QL, E, dx, qr_raw);
            sgemv(blas, w.kv_a, KVR, E, dx, kv);
            if (full) {
                sgemv(blas, w.idx_k, ID, E, dx, ik);
                sgemv(blas, w.idx_proj, IH, E, dx, iw, 1.0f / std::sqrt((float) (ID * IH)));
            }
            glmd::DsaPrep a;
            a.qr_raw = qr_raw;
            a.q_a_norm = w.q_a_norm;
            a.qr = qr;
            a.q_lora = QL;
            a.kv_raw = kv;
            a.kv_norm = w.kv_a_norm;
            a.lat = state + lay.dsa_lat[(size_t) il];
            a.kv_lora = KV;
            a.k_pe = (uint16_t*) (state + lay.dsa_kpe[(size_t) il]);
            a.qk_rope = R;
            a.inv_freq = d_inv;
            if (full) {
                a.ik_raw = ik;
                a.ik_ln_w = w.ik_w;
                a.ik_ln_b = w.ik_b;
                a.idx_ln_eps = 1e-6f;
                a.ik_cache = state + lay.dsa_ik[(size_t) il];
                a.idx_dim = ID;
                a.idx_rope = 64;
                a.idx_inv_freq = d_inv;
            }
            a.p = t;
            a.eps = 1e-5f;
            glmd::dsa_prep(a, nullptr);
            sgemv(blas, w.q_b, H * QH, QL, qr, q);
            if (full) sgemv(blas, w.idx_q_b, IH * ID, QL, qr, iq);
            glmd::DsaRopeQ rq;
            rq.q = q;
            rq.n_head = H;
            rq.inv_freq = d_inv;
            rq.iq = full ? iq : nullptr;
            rq.idx_heads = IH;
            rq.idx_dim = ID;
            rq.idx_inv_freq = d_inv;
            rq.p = t;
            glmd::dsa_rope_q(rq, nullptr);
            int* st = sel + (size_t) t * TOP;
            if (full) {
                glmd::index_score_tile(iq, iw, state + lay.dsa_ik[0], IH, ID, 1, 0, t + 1, row, Tall, nullptr);
                glmd::topk(row, Tall, t + 1, 1, nullptr, TOP, st, nullptr);
            }
            glmd::Mla2 m;
            m.q = q;
            m.wk_b = (const uint16_t*) w.k_b.q;
            m.wv_b = (const uint16_t*) w.v_b.q;
            m.lat = state + lay.dsa_lat[(size_t) il];
            m.k_pe = (const uint16_t*) (state + lay.dsa_kpe[(size_t) il]);
            m.cells = st;
            m.n_sel = std::min(t + 1, TOP);
            m.n_head = H;
            m.v_head = VH;
            m.scale = glmd::attn_scale(QH);
            m.work = work;
            m.work_bytes = wb;
            m.out = out;
            glmd::mla2(m, nullptr);
            sgemv(blas, w.out, E, H * VH, out, mix);
            cudaMemcpy(r.mixer[il].data() + (size_t) t * E, mix, E * 4, cudaMemcpyDeviceToHost);
        }
    }
    cudaDeviceSynchronize();
    r.sel = down(sel, (size_t) Tall * TOP);
    r.lat0 = down((const uint16_t*) (state + lay.dsa_lat[0]), (size_t) Tall * KV);
    r.kpe0 = down((const uint16_t*) (state + lay.dsa_kpe[0]), (size_t) Tall * R);
    r.ik0 = down(state + lay.dsa_ik[0], (size_t) Tall * ID);
    for (void* p : {(void*) state, (void*) d_inv, (void*) dx, (void*) qr_raw, (void*) qr, (void*) kv, (void*) ik,
                    (void*) iw, (void*) q, (void*) iq, (void*) row, (void*) out, (void*) mix, (void*) sel, work})
        cudaFree(p);
    return r;
}

void compare(const char* label, const Run& b, const Run& t, int Tall) {
    char what[200];
    for (int il = 0; il < 2; ++il) {
        const double e = rel_l2(b.mixer[il], t.mixer[il]);
        std::snprintf(what, sizeof what, "%s layer %d (%s): mixer rel L2 %.2e (< 2e-3)", label, il, il ? "shared" : "full", e);
        expect(e < 2e-3, what);
    }
    std::vector<float> lb(b.lat0.size()), lt(t.lat0.size()), kb(b.kpe0.size()), kt(t.kpe0.size());
    for (size_t i = 0; i < lb.size(); ++i) {
        lb[i] = glmd::twin::half_float(b.lat0[i]);
        lt[i] = glmd::twin::half_float(t.lat0[i]);
    }
    for (size_t i = 0; i < kb.size(); ++i) {
        kb[i] = glmd::twin::half_float(b.kpe0[i]);
        kt[i] = glmd::twin::half_float(t.kpe0[i]);
    }
    const double el = rel_l2(lb, lt), ek = rel_l2(kb, kt), ei = rel_l2(b.ik0, t.ik0);
    std::snprintf(what, sizeof what, "%s caches: latent %.2e, k_pe %.2e (FP16 inputs, < 2e-3), keys %.2e (F32, < 1e-4)",
                  label, el, ek, ei);
    expect(el < 2e-3 && ek < 2e-3 && ei < 1e-4, what);
    // the prompt path scores with the WMMA scorer (FP16 products) where the device takes it, the token path in F32: a
    // row may swap near-tied cells, never more (the F32 scorer's exact equality is glmd_batch_parity's test_select)
    int rows_diff = 0;
    double ov_min = 1.0;
    for (int r = 0; r < Tall; ++r) {
        const int* x = &b.sel[(size_t) r * TOP];
        const int* y = &t.sel[(size_t) r * TOP];
        if (std::memcmp(x, y, TOP * 4) == 0) continue;
        ++rows_diff;
        int nx = 0, ny = 0, common = 0;
        while (nx < TOP && x[nx] >= 0) ++nx;
        while (ny < TOP && y[ny] >= 0) ++ny;
        for (int i = 0, j = 0; i < nx && j < ny;) {   // both ascending
            if (x[i] == y[j]) {
                ++common;
                ++i;
                ++j;
            } else if (x[i] < y[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        ov_min = std::min(ov_min, nx == ny && nx > 0 ? (double) common / nx : 0.0);
    }
    std::snprintf(what, sizeof what, "%s selection: %d of %d rows differ, the worst shares %.4f of its cells (> 0.99)",
                  label, rows_diff, Tall, ov_min);
    expect(ov_min > 0.99, what);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    const Glm5Geometry g = geometry();
    Weights W[2] = {make_layer(true), make_layer(false)};
    glmdsa::GlmDsaFast dsa;
    dsa.L.resize(3);
    for (int il = 0; il < 2; ++il) {
        glmdsa::GlmDsaLayerW& L = dsa.L[(size_t) il];
        L.bound = true;
        L.full = il == 0;
        L.idx_src = 0;
        L.q_a = W[il].q_a;
        L.q_a_norm = W[il].q_a_norm;
        L.q_b = W[il].q_b;
        L.kv_a = W[il].kv_a;
        L.kv_a_norm = W[il].kv_a_norm;
        L.k_b = W[il].k_b;
        L.v_b = W[il].v_b;
        L.out = W[il].out;
        if (L.full) {
            L.idx_k = W[il].idx_k;
            L.idx_q_b = W[il].idx_q_b;
            L.idx_proj = W[il].idx_proj;
            L.idx_k_norm_w = W[il].ik_w;
            L.idx_k_norm_b = W[il].ik_b;
        }
    }
    cublasHandle_t blas;
    cublasCreate(&blas);
    struct Case {
        const char* label;
        std::vector<int> chunks;
        const char* ratio;
    };
    const Case cases[] = {{"32 tokens (dense)", {32}, nullptr},
                          {"2,100 tokens (dense-masked past 2,048, ratio 4)", {2100}, "4"},
                          {"2,100 tokens (gather, ratio 1)", {2100}, "1"},
                          {"1,500 + 600 tokens in two chunks (the default walk)", {1500, 600}, nullptr}};
    for (const Case& c : cases) {
        int Tall = 0;
        for (int t : c.chunks) Tall += t;
        GlmStateLayout lay;
        std::string err;
        if (!glm_state_carve(g, 0, 2, -1, Tall, GlmStateMode::Fp16, 0, lay, err)) {
            std::printf("state carve: %s\n", err.c_str());
            return 1;
        }
        if (c.ratio) setenv("STRATA_GLMD_GATHER_RATIO", c.ratio, 1);
        else unsetenv("STRATA_GLMD_GATHER_RATIO");
        const auto x = randn((size_t) Tall * E, 1.0f);
        const Run b = batched(g, lay, dsa, x, c.chunks, blas);
        const Run t = token(lay, W, x, blas);
        compare(c.label, b, t, Tall);
    }
    expect(glmd::batch_launch_errors() == 0 && glmd::launch_errors() == 0, "no launch errors or refused shapes");
    cublasDestroy(blas);
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
