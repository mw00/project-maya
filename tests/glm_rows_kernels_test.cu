// tests/glm_rows_kernels_test.cu - the batched rows' one-launch kernels (glm_rows.cu, STRATA_GLM_ROWS_BATCH) against
// the per-row kernels the rows take without it, on random inputs (needs a GPU, no model), compared bitwise:
//  - hc_rows over n rows (each its own args, partials and counter) against hc() for each row alone: R_new, pre, post,
//    comb, x and xq, with and without a previous block's output to fold in;
//  - kda_layers over n rows of one layer (each its own conv state, recurrent state, scratch and gated output) against
//    kda_prep + kda_rec for each row alone: the conv outputs, g1 / g2, both states and the q8_1 output;
//  - mv_nt for 2 .. kMaxMvNt activations (past 8: in groups of 8) against mv() for each activation alone, BF16 and
//    Q8_0 weights.
#include "strata/kernels/glm_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace {

int g_fail = 0;
std::mt19937 g_rng(4321);

#define CK(x)                                                                                                    \
    do {                                                                                                         \
        const cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));              \
            std::exit(2);                                                                                        \
        }                                                                                                        \
    } while (0)

template<class T> T* dev(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc((void**) &p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
template<class T> std::vector<T> back(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
template<class T> void put(T* d, const std::vector<T>& h) {
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
std::vector<float> rnd(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (float& x : v) x = u(g_rng);
    return v;
}
std::vector<uint16_t> rnd_bf16(size_t n, float scale) {   // BF16: a float's top 16 bits
    const std::vector<float> f = rnd(n, -scale, scale);
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t b;
        std::memcpy(&b, &f[i], 4);
        v[i] = (uint16_t) (b >> 16);
    }
    return v;
}
template<class T> float* fdev(const std::vector<T>& h) {
    T* d = dev<T>(h.size());
    put(d, h);
    return (float*) d;
}
// bitwise; prints the first difference
void same(const char* what, int n, int row, const void* a, const void* b, size_t bytes) {
    if (std::memcmp(a, b, bytes) == 0) return;
    size_t i = 0;
    while (((const uint8_t*) a)[i] == ((const uint8_t*) b)[i]) ++i;
    std::fprintf(stderr, "FAIL: %s, %d rows: row %d differs from the per-row kernel's (first at byte %zu of %zu)\n", what,
                 n, row, i, bytes);
    ++g_fail;
}

// ---- hc_rows == hc per row
void test_hc(int n, bool fold) {
    constexpr int E = 4096, HC = 4, MIX = 2 * HC + HC * HC;
    const size_t part_n = (size_t) (HC * E / 512) * 26 + 64;
    // the layer's weights, shared by the rows
    uint16_t* w_fn = (uint16_t*) fdev(rnd_bf16((size_t) MIX * HC * E, 0.02f));
    float* w_scale = fdev(rnd(3, 0.5f, 1.5f));
    float* w_base = fdev(rnd(MIX, -0.5f, 0.5f));
    float* norm_w = fdev(rnd(E, 0.5f, 1.5f));
    struct Side {
        float *R_new, *pre, *post, *comb, *x, *part;
        uint8_t* xq;
        unsigned int* counter;
    };
    std::vector<gf::HcArgs> ref((size_t) n), bat((size_t) n);
    std::vector<Side> sr((size_t) n), sb((size_t) n);
    const size_t xq_bytes = (size_t) (E / 32) * 36;
    for (int i = 0; i < n; ++i) {
        // the row's inputs, the same for both sides
        float* R_old = fdev(rnd((size_t) HC * E, -1.0f, 1.0f));
        float* block_out = fold ? fdev(rnd(E, -1.0f, 1.0f)) : nullptr;
        float* post_in = fdev(rnd(HC, 0.0f, 1.0f));
        float* comb_in = fdev(rnd((size_t) HC * HC, 0.0f, 0.5f));
        for (int side = 0; side < 2; ++side) {
            Side& o = (side == 0 ? sr : sb)[(size_t) i];
            o.R_new = dev<float>((size_t) HC * E);
            o.pre = dev<float>(8);
            o.post = dev<float>(8);
            o.comb = dev<float>(16);
            o.x = dev<float>(E);
            o.part = dev<float>(part_n);
            o.xq = dev<uint8_t>(xq_bytes);
            o.counter = (unsigned int*) dev<uint8_t>(64);
            gf::HcArgs a;
            a.block_out = block_out;
            a.R_old = R_old;
            a.R_new = o.R_new;
            a.post_in = post_in;
            a.comb_in = comb_in;
            a.pre = o.pre;
            a.post = o.post;
            a.comb = o.comb;
            a.w_fn = w_fn;
            a.w_scale = w_scale;
            a.w_base = w_base;
            a.norm_w = norm_w;
            a.n_embd = E;
            a.x = o.x;
            a.xq = o.xq;
            a.part = o.part;
            a.counter = o.counter;
            (side == 0 ? ref : bat)[(size_t) i] = a;
        }
    }
    for (int i = 0; i < n; ++i) gf::hc(ref[(size_t) i], nullptr);
    gf::hc_rows(bat.data(), n, nullptr);
    CK(cudaDeviceSynchronize());
    if (gf::launch_errors() != 0) {
        std::fprintf(stderr, "FAIL: hc launches failed\n");
        ++g_fail;
        return;
    }
    char what[64];
    std::snprintf(what, sizeof what, "hc_rows%s", fold ? " (folding a block output)" : "");
    for (int i = 0; i < n; ++i) {
        const Side &r = sr[(size_t) i], &b = sb[(size_t) i];
        if (fold) same(what, n, i, back(r.R_new, (size_t) HC * E).data(), back(b.R_new, (size_t) HC * E).data(), (size_t) HC * E * 4);
        same(what, n, i, back(r.pre, 8).data(), back(b.pre, 8).data(), 8 * 4);
        same(what, n, i, back(r.post, 8).data(), back(b.post, 8).data(), 8 * 4);
        same(what, n, i, back(r.comb, 16).data(), back(b.comb, 16).data(), 16 * 4);
        same(what, n, i, back(r.x, E).data(), back(b.x, E).data(), (size_t) E * 4);
        same(what, n, i, back(r.xq, xq_bytes).data(), back(b.xq, xq_bytes).data(), xq_bytes);
    }
}

// ---- kda_layers over the rows == kda_prep + kda_rec per row
void test_kda(int n) {
    constexpr int NH = 64, HD = 128, DC = 4, DI = NH * HD;
    const size_t conv_n = (size_t) 3 * DI * (DC - 1), st_n = (size_t) NH * HD * HD, q_bytes = (size_t) (DI / 32) * 36;
    // the layer's weights
    float* conv_w[3];
    for (int c = 0; c < 3; ++c) conv_w[c] = fdev(rnd((size_t) DI * DC, -0.5f, 0.5f));
    uint16_t* f_b = (uint16_t*) fdev(rnd_bf16((size_t) DI * HD, 0.05f));
    uint16_t* g_b = (uint16_t*) fdev(rnd_bf16((size_t) DI * HD, 0.05f));
    float* dt_bias = fdev(rnd(DI, -1.0f, 1.0f));
    float* ssm_a = fdev(rnd(NH, -1.0f, 1.0f));
    float* norm_w = fdev(rnd(HD, 0.5f, 1.5f));
    struct Side {
        float *conv_state, *state, *out[3], *g1, *g2;
        uint8_t* q;
    };
    std::vector<Side> sr((size_t) n), sb((size_t) n);
    std::vector<gf::KdaPrepArgs> kp((size_t) n), bp((size_t) n);
    std::vector<gf::KdaRecArgs> br((size_t) n);
    std::vector<float*> beta((size_t) n);
    for (int i = 0; i < n; ++i) {
        float* proj[3];
        for (int c = 0; c < 3; ++c) proj[c] = fdev(rnd(DI, -1.0f, 1.0f));
        float* fa = fdev(rnd(HD, -1.0f, 1.0f));
        float* ga = fdev(rnd(HD, -1.0f, 1.0f));
        beta[(size_t) i] = fdev(rnd(NH, -2.0f, 2.0f));
        const std::vector<float> cs0 = rnd(conv_n, -1.0f, 1.0f), st0 = rnd(st_n, -0.1f, 0.1f);
        for (int side = 0; side < 2; ++side) {
            Side& o = (side == 0 ? sr : sb)[(size_t) i];
            o.conv_state = fdev(cs0);
            o.state = fdev(st0);
            for (int c = 0; c < 3; ++c) o.out[c] = dev<float>(DI);
            o.g1 = dev<float>(DI);
            o.g2 = dev<float>(DI);
            o.q = dev<uint8_t>(q_bytes);
            gf::KdaPrepArgs k;
            for (int c = 0; c < 3; ++c) {
                k.proj[c] = proj[c];
                k.conv_w[c] = conv_w[c];
                k.out[c] = o.out[c];
            }
            k.conv_state = o.conv_state;
            k.fa = fa;
            k.ga = ga;
            k.f_b = f_b;
            k.g_b = g_b;
            k.dt_bias = dt_bias;
            k.ssm_a = ssm_a;
            k.g1 = o.g1;
            k.g2 = o.g2;
            k.n_head = NH;
            k.head_dim = HD;
            k.d_conv = DC;
            (side == 0 ? kp : bp)[(size_t) i] = k;
        }
        gf::KdaRecArgs& r = br[(size_t) i];
        r.q = sb[(size_t) i].out[0];
        r.k = sb[(size_t) i].out[1];
        r.v = sb[(size_t) i].out[2];
        r.g1 = sb[(size_t) i].g1;
        r.beta_raw = beta[(size_t) i];
        r.state = sb[(size_t) i].state;
        r.g2 = sb[(size_t) i].g2;
        r.norm_w = norm_w;
        r.out = sb[(size_t) i].q;
    }
    constexpr float eps = 1e-5f;
    for (int i = 0; i < n; ++i) {
        const Side& o = sr[(size_t) i];
        gf::kda_prep(kp[(size_t) i], nullptr);
        gf::kda_rec(o.out[0], o.out[1], o.out[2], o.g1, beta[(size_t) i], o.state, o.g2, norm_w, eps, NH, HD, o.q, nullptr);
    }
    for (int i0 = 0; i0 < n; i0 += gf::kKdaLayers)
        gf::kda_layers(bp.data() + i0, br.data() + i0, std::min(gf::kKdaLayers, n - i0), eps, nullptr);
    CK(cudaDeviceSynchronize());
    if (gf::launch_errors() != 0) {
        std::fprintf(stderr, "FAIL: kda launches failed\n");
        ++g_fail;
        return;
    }
    for (int i = 0; i < n; ++i) {
        const Side &r = sr[(size_t) i], &b = sb[(size_t) i];
        for (int c = 0; c < 3; ++c) same("kda_layers conv out", n, i, back(r.out[c], DI).data(), back(b.out[c], DI).data(), (size_t) DI * 4);
        same("kda_layers g1", n, i, back(r.g1, DI).data(), back(b.g1, DI).data(), (size_t) DI * 4);
        same("kda_layers g2", n, i, back(r.g2, DI).data(), back(b.g2, DI).data(), (size_t) DI * 4);
        same("kda_layers conv state", n, i, back(r.conv_state, conv_n).data(), back(b.conv_state, conv_n).data(), conv_n * 4);
        same("kda_layers recurrent state", n, i, back(r.state, st_n).data(), back(b.state, st_n).data(), st_n * 4);
        same("kda_layers gated output (q8_1)", n, i, back(r.q, q_bytes).data(), back(b.q, q_bytes).data(), q_bytes);
    }
}

// ---- mv_nt == mv per activation
std::vector<uint8_t> q8_0_rows(int rows, int n_in) {   // random Q8_0 rows (FP16 scale + 32 int8 a block)
    std::vector<uint8_t> v((size_t) rows * (size_t) (n_in / 32) * 34);
    std::uniform_real_distribution<float> sc(0.001f, 0.02f);
    std::uniform_int_distribution<int> q(-127, 127);
    for (size_t b = 0; b < v.size(); b += 34) {
        const __half h = __float2half(sc(g_rng));
        std::memcpy(&v[b], &h, 2);
        for (int i = 0; i < 32; ++i) v[b + 2 + (size_t) i] = (uint8_t) (int8_t) q(g_rng);
    }
    return v;
}
void test_mv_nt(int nt) {
    constexpr int N_IN = 4096, N_OUT = 1024, Q8_0 = 8;
    const size_t xq_row = (size_t) (N_IN / 32) * 36;
    float* xf = fdev(rnd((size_t) nt * N_IN, -1.0f, 1.0f));
    uint8_t* xq = dev<uint8_t>((size_t) nt * xq_row);
    gf::quantize_q8_1(xf, xq, nt * N_IN, nullptr);   // (each activation's blocks follow the one before)
    const void* w[2] = {fdev(rnd_bf16((size_t) N_OUT * N_IN, 0.05f)), fdev(q8_0_rows(N_OUT, N_IN))};
    const int type[2] = {gf::kTypeBF16, Q8_0};
    float* y_ref[2] = {dev<float>((size_t) nt * N_OUT), dev<float>((size_t) nt * N_OUT)};
    float* y_nt[2] = {dev<float>((size_t) nt * N_OUT), dev<float>((size_t) nt * N_OUT)};
    gf::MvJob jobs[2];
    for (int k = 0; k < 2; ++k) {
        jobs[k] = {w[k], xq, xf, y_nt[k], nullptr, 1.0f, type[k], N_IN, N_OUT};
        for (int t = 0; t < nt; ++t) {
            gf::MvJob one = {w[k], xq + (size_t) t * xq_row, xf + (size_t) t * N_IN, y_ref[k] + (size_t) t * N_OUT,
                             nullptr, 1.0f, type[k], N_IN, N_OUT};
            if (!gf::mv(&one, 1, nullptr)) {
                std::fprintf(stderr, "FAIL: mv refused type %d\n", type[k]);
                ++g_fail;
                return;
            }
        }
    }
    if (!gf::mv_nt(jobs, 2, nt, nullptr)) {
        std::fprintf(stderr, "FAIL: mv_nt refused %d activations\n", nt);
        ++g_fail;
        return;
    }
    CK(cudaDeviceSynchronize());
    for (int k = 0; k < 2; ++k)
        for (int t = 0; t < nt; ++t)
            same(k == 0 ? "mv_nt (BF16)" : "mv_nt (Q8_0)", nt, t, back(y_ref[k] + (size_t) t * N_OUT, N_OUT).data(),
                 back(y_nt[k] + (size_t) t * N_OUT, N_OUT).data(), (size_t) N_OUT * 4);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("glm_rows_kernels_test: no GPU, skipped\n");
        return 0;
    }
    for (const int rows : {2, 3, gf::kMaxMvNt}) {
        test_hc(rows, false);
        test_hc(rows, true);
        test_kda(rows);
    }
    for (int nt = 2; nt <= gf::kMaxMvNt; ++nt) test_mv_nt(nt);
    if (g_fail == 0)
        std::printf("glm_rows_kernels_test: ok (hc_rows, kda_layers: 2, 3 and %d rows; mv_nt: 2 .. %d activations)\n",
                    gf::kMaxMvNt, gf::kMaxMvNt);
    return g_fail == 0 ? 0 : 1;
}
