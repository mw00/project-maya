// tests/ling_mixed_gu_test.cu - maya-ling's routed experts whose gate and up are of different types (Maya-M / Maya-L
// give a few layers a larger up), on the shared src/ds4/ds4_iq.cu built as maya-ling builds it (MAYA_LING_SWIGLU: the
// post-SiLU clamp, a negative limit), random blocks, no model:
//   * the token path (iq::gate_up, a plan of experts) and the window path (groups of entries): h against a double
//     reference on ggml's own dequantized rows and the int8 inputs, the clamp off, Ling's post-SiLU clamp, DeepSeek's;
//   * the window's h bit for bit the token path's;
//   * the prompt sets (iq::set_gate_up: the light kernels for experts of a few rows, the tiles for the others), the
//     gate's F32 sums kept in graw between the two passes;
//   * a layer of one type keeps the one-pass kernels: tu = the gate's type gives the bits of tu = 0.
// ling_mixed_gu_test
#include "../src/ds4/ds4_iq.cuh"

#include "ggml.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace iq = maya::ds4::gguf::iq;

namespace {

int g_fail = 0;
std::mt19937 rng(1224);

#define CU(x)                                                                             \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

float frand(float a = 1.0f) { return std::uniform_real_distribution<float>(-a, a)(rng); }

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> h(n);
    CU(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

void expect(const std::string& what, const float* got, const double* want, size_t n, double rtol) {
    double worst = 0, scale = 1e-30;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs(want[i]));
    size_t at = 0;
    bool finite = true;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i])) finite = false;
        const double e = std::fabs(got[i] - want[i]) / scale;
        if (!(e <= worst)) worst = e, at = i;
    }
    const bool ok = finite && worst <= rtol;
    if (!ok) ++g_fail;
    std::printf("%-62s %s  max rel err %.2e (at %zu: %.6g vs %.6g)\n", what.c_str(), ok ? "ok  " : "FAIL", worst, at,
                (double) got[at], want[at]);
}
void same(const std::string& what, const void* a, const void* b, size_t bytes) {
    const bool ok = std::memcmp(a, b, bytes) == 0;
    if (!ok) ++g_fail;
    std::printf("%-62s %s\n", what.c_str(), ok ? "ok    bit for bit" : "FAIL  differs");
}

float h2f(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, 2);
    return __half2float(h);
}
uint16_t f2h(float f) {
    const __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
float f16r(float f) { return h2f(f2h(f)); }

// random blocks of a type (mm3_gguf_test's q_rand): random bytes, the block scales small normal F16s - divided by the
// type's typical code size, so a gate and an up of different types have values of one size (as a real layer's): the
// clamp then bites on both, and no row sums large terms to a small value (F32 sums' cancellation, not the kernels')
float d_div(int type) {
    switch (type) {
        case iq::kIQ4_XS:
        case iq::kQ6_K: return 20.0f;
        case iq::kQ4_K: return 5.0f;
        case iq::kIQ3_S: return 2.0f;
        default: return 1.0f;
    }
}
std::vector<uint8_t> q_rand(int type, int n, int k) {
    const size_t rb = iq::row_bytes(type, k), bb = (size_t) iq::block_bytes(type);
    std::vector<uint8_t> w(rb * n);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    auto put = [&](size_t at, float f) {
        const uint16_t u = f2h(f);
        std::memcpy(&w[at], &u, 2);
    };
    for (size_t o = 0; o < w.size(); o += bb) {
        const float d = (0.001f + 0.002f * std::fabs(frand())) / d_div(type);
        switch (type) {
            case iq::kQ4_K:
            case iq::kQ5_K: put(o, d), put(o + 2, 0.5f * d); break;
            case iq::kQ6_K: put(o + 208, d); break;
            case iq::kIQ1_M: {
                const uint16_t u = f2h(d);
                for (int i = 0; i < 4; ++i) {
                    uint16_t sw;
                    std::memcpy(&sw, &w[o + 48 + 2 * i], 2);
                    sw = (uint16_t) ((sw & 0x0FFFu) | (((u >> (4 * i)) & 0xFu) << 12));
                    std::memcpy(&w[o + 48 + 2 * i], &sw, 2);
                }
                break;
            }
            default: put(o, d); break;
        }
    }
    return w;
}
std::vector<float> deq(int type, const std::vector<uint8_t>& w, size_t n) {
    std::vector<float> v(n);
    ggml_get_type_traits((ggml_type) type)->to_float(w.data(), v.data(), (int64_t) n);
    return v;
}
std::vector<float> q8_round(const float* x, int k) {   // (the light sets' rounding: a scale a 32, max|x| / 127)
    std::vector<float> v(k);
    for (int b = 0; b < k / 32; ++b) {
        float am = 0.0f;
        for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(x[b * 32 + j]));
        const float sc = am / 127.0f;
        for (int j = 0; j < 32; ++j)
            v[b * 32 + j] = am > 0.0f ? (float) ((double) std::nearbyint(x[b * 32 + j] / sc) * sc) : 0.0f;
    }
    return v;
}
// the activation maya-ling's kernels apply (ds4_iq.cu's swiglu under MAYA_LING_SWIGLU), in double
double act_ref(double g, double u, double lim) {
    if (lim > 0) {
        g = std::min(g, lim);
        u = std::min(std::max(u, -lim), lim);
        return g / (1.0 + std::exp(-g)) * u;
    }
    const double a = g / (1.0 + std::exp(-g));
    if (lim < 0) {
        const double L = -lim;
        return std::min(a, L) * std::min(std::max(u, -L), L);
    }
    return a * u;
}
double dot(const float* w, const float* x, int k) {
    double s = 0;
    for (int i = 0; i < k; ++i) s += (double) w[i] * x[i];
    return s;
}

struct Layer {
    int D, F, n_exp, ty[3];
    size_t tr[3] = {0, 0, 0}, blob = 0;
    std::vector<uint8_t> h;
    std::vector<std::vector<float>> w;   // [e * 2 + r]: ggml's values of the gate (r 0) / up (r 1), [F][D]
    uint8_t* d = nullptr;
    const float* row(int e, int r, int f) const { return w[(size_t) e * 2 + r].data() + (size_t) f * D; }
};
Layer make(int D, int F, int n_exp, int tg, int tu, int td) {
    Layer L{D, F, n_exp, {tg, tu, td}};
    size_t off = 0;
    for (int r = 0; r < 3; ++r) {
        L.tr[r] = off;
        off += ((r < 2 ? F : D) * iq::row_bytes(L.ty[r], r < 2 ? D : F) + 255) / 256 * 256;
    }
    L.blob = off;
    L.h.assign(L.blob * n_exp, 0);
    for (int e = 0; e < n_exp; ++e)
        for (int r = 0; r < 3; ++r) {
            const int rows = r < 2 ? F : D, k = r < 2 ? D : F;
            const std::vector<uint8_t> m = q_rand(L.ty[r], rows, k);
            std::memcpy(L.h.data() + (size_t) e * L.blob + L.tr[r], m.data(), m.size());
            if (r < 2) L.w.push_back(deq(L.ty[r], m, (size_t) rows * k));
        }
    L.d = dev(L.h);
    return L;
}

// h of the token path: K experts for one token's int8 input, then the window: T tokens' entries grouped by expert
void test_token_window(const Layer& L, float lim, const std::string& name, cudaStream_t s) {
    const int D = L.D, F = L.F, K = 6, T = 3;
    // int8 inputs with a scale a 32 (the rows the engine quantizes), T rows
    std::vector<int8_t> xq((size_t) T * D);
    std::vector<float> xd((size_t) T * D / 32), xv((size_t) T * D);
    for (auto& v : xq) v = (int8_t) ((int) (rng() % 255) - 127);
    for (auto& v : xd) v = 0.004f + 0.004f * std::fabs(frand());
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = (float) ((double) xq[i] * xd[i / 32]);
    int8_t* dxq = dev(xq);
    float* dxd = dev(xd);
    std::vector<int> pick((size_t) T * 8, 0);
    for (int t = 0; t < T; ++t) {
        std::vector<int> all(L.n_exp);
        for (int i = 0; i < L.n_exp; ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        for (int i = 0; i < K; ++i) pick[(size_t) t * 8 + i] = all[i];
    }
    auto args = [&](float* h) {
        iq::Args a;
        a.n_embd = D;
        a.n_ff = F;
        a.limit = lim;
        a.h = h;
        a.tg = L.ty[0];
        a.tu = L.ty[1] != L.ty[0] ? L.ty[1] : 0;
        a.td = L.ty[2];
        a.act = iq::kActSwiglu;
        for (int r = 0; r < 3; ++r) a.tr[r] = L.tr[r];
        return a;
    };
    // the token path, token by token (plan: K experts, input row 0)
    std::vector<float> tok((size_t) T * 8 * F, 0.0f);
    float* dh = dev(std::vector<float>((size_t) 8 * F));
    for (int t = 0; t < T; ++t) {
        std::vector<unsigned long long> pp(8, 0ull);
        for (int i = 0; i < K; ++i) pp[i] = (unsigned long long) (L.d + (size_t) pick[(size_t) t * 8 + i] * L.blob);
        unsigned long long* dpp = dev(pp);
        iq::Args a = args(dh);
        a.plan_ptr = dpp;
        a.xq = dxq + (size_t) t * D;
        a.xd = dxd + (size_t) t * D / 32;
        if (!iq::gate_up(a, K, 1, s)) {
            ++g_fail;
            std::printf("%-62s FAIL  gate_up refused\n", (name + " token").c_str());
        }
        CU(cudaStreamSynchronize(s));
        const std::vector<float> hh = host(dh, (size_t) K * F);
        std::copy(hh.begin(), hh.end(), tok.begin() + (long) t * 8 * F);
        cudaFree(dpp);
    }
    std::vector<double> want;
    std::vector<float> got;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i) {
            const int e = pick[(size_t) t * 8 + i];
            for (int f = 0; f < F; ++f) {
                want.push_back(act_ref(dot(L.row(e, 0, f), xv.data() + (size_t) t * D, D),
                                       dot(L.row(e, 1, f), xv.data() + (size_t) t * D, D), lim));
                got.push_back(tok[((size_t) t * 8 + i) * F + f]);
            }
        }
    expect(name + " token h", got.data(), want.data(), want.size(), 1e-4);
    // the window: each distinct expert once, its entries (entry t*8+i: input row t)
    std::vector<unsigned long long> gptr;
    std::vector<int> gstart, gent;
    int ne_max = 0;
    for (int e = 0; e < L.n_exp; ++e) {
        std::vector<int> ents;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < K; ++i)
                if (pick[(size_t) t * 8 + i] == e) ents.push_back(t * 8 + i);
        if (ents.empty()) continue;
        gptr.push_back((unsigned long long) (L.d + (size_t) e * L.blob));
        gstart.push_back((int) gent.size());
        for (int x : ents) gent.push_back(x);
        ne_max = std::max(ne_max, (int) ents.size());
    }
    gstart.push_back((int) gent.size());
    std::vector<int> ng(1, (int) gptr.size());
    unsigned long long* dgp = dev(gptr);
    int *dgs = dev(gstart), *dge = dev(gent), *dng = dev(ng);
    float* dwh = dev(std::vector<float>((size_t) T * 8 * F));
    iq::Args w = args(dwh);
    w.grp_ptr = dgp;
    w.grp_start = dgs;
    w.grp_ent = dge;
    w.n_groups = dng;
    w.xq = dxq;
    w.xd = dxd;
    if (!iq::gate_up(w, (int) gptr.size(), ne_max, s)) {
        ++g_fail;
        std::printf("%-62s FAIL  gate_up refused\n", (name + " window").c_str());
    }
    CU(cudaStreamSynchronize(s));
    const std::vector<float> win = host(dwh, (size_t) T * 8 * F);
    std::vector<float> a1, a2;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i)
            for (int f = 0; f < F; ++f) {
                a1.push_back(win[((size_t) t * 8 + i) * F + f]);
                a2.push_back(tok[((size_t) t * 8 + i) * F + f]);
            }
    same(name + " window h = the token path's", a1.data(), a2.data(), a1.size() * 4);
    cudaFree(dxq), cudaFree(dxd), cudaFree(dh), cudaFree(dgp), cudaFree(dgs), cudaFree(dge), cudaFree(dng);
    cudaFree(dwh);
}

// the prompt sets: experts of a few rows (the light kernels) and of more (the tiles); graw between the passes
void test_sets(const Layer& L, float lim, const std::string& name, cudaStream_t s) {
    const int D = L.D, F = L.F;
    for (int R : {50, 400}) {
        const int NT = 30;
        std::vector<int> owner(R), idx(R);
        for (int r = 0; r < R; ++r) owner[r] = (int) (rng() % L.n_exp);
        std::sort(owner.begin(), owner.end());
        std::vector<int> bounds(L.n_exp + 1, 0);
        for (int e = 0, r = 0; e <= L.n_exp; ++e) {
            while (r < R && owner[r] < e) ++r;
            bounds[e] = r;
        }
        int max_rows = 0, n_light = 0;
        for (int e = 0; e < L.n_exp; ++e) {
            const int n = bounds[e + 1] - bounds[e];
            max_rows = std::max(max_rows, n);
            n_light += n <= 16 ? n : 0;
        }
        std::vector<float> xs((size_t) NT * D);
        for (auto& v : xs) v = frand();
        for (int r = 0; r < R; ++r) idx[r] = (int) (rng() % NT);
        float* dxs = dev(xs);
        int *dbd = dev(bounds), *didx = dev(idx);
        uint16_t* dh16 = dev(std::vector<uint16_t>((size_t) R * F));
        float* dy = dev(std::vector<float>((size_t) R * D));   // (the engine's graw: the down output's rows)
        iq::SetArgs a;
        a.wbase = L.d;
        a.stride = L.blob;
        for (int r = 0; r < 3; ++r) a.tr[r] = L.tr[r];
        a.bounds = dbd;
        a.row0 = 0;
        a.max_rows = max_rows;
        a.n_exp = L.n_exp;
        a.n_embd = D;
        a.n_ff = F;
        a.limit = lim;
        a.xf = dxs;
        a.idx = didx;
        a.ldx = D;
        a.h = dh16;
        a.ldo = F;
        a.tg = L.ty[0];
        a.tu = L.ty[1] != L.ty[0] ? L.ty[1] : 0;
        a.td = L.ty[2];
        a.act = iq::kActSwiglu;
        a.graw = a.tu != 0 ? dy : nullptr;
        if (!iq::set_gate_up(a, s)) {
            ++g_fail;
            std::printf("%-62s FAIL  set_gate_up refused\n", (name + " set").c_str());
        }
        CU(cudaStreamSynchronize(s));
        const std::vector<uint16_t> h16 = host(dh16, (size_t) R * F);
        std::vector<float> hg((size_t) R * F);
        std::vector<double> hw((size_t) R * F);
        for (int r = 0; r < R; ++r) {
            const int e = owner[r];
            const bool light = bounds[e + 1] - bounds[e] <= 16;   // (light: int8 inputs per 32; tiles: FP16)
            std::vector<float> xi(xs.begin() + (long) idx[r] * D, xs.begin() + (long) (idx[r] + 1) * D);
            if (light) xi = q8_round(xi.data(), D);
            else
                for (auto& v : xi) v = f16r(v);
            for (int f = 0; f < F; ++f) {
                double g = 0, u = 0;
                for (int i = 0; i < D; ++i) {
                    const float wg = L.row(e, 0, f)[i], wu = L.row(e, 1, f)[i];
                    g += (double) (light ? wg : f16r(wg)) * xi[i];
                    u += (double) (light ? wu : f16r(wu)) * xi[i];
                }
                hw[(size_t) r * F + f] = act_ref(g, u, lim);
                hg[(size_t) r * F + f] = h2f(h16[(size_t) r * F + f]);
            }
        }
        expect(name + " set h (F16), " + std::to_string(R) + " rows, " + std::to_string(n_light) + " light", hg.data(),
               hw.data(), hw.size(), 2e-3);
        cudaFree(dxs), cudaFree(dbd), cudaFree(didx), cudaFree(dh16), cudaFree(dy);
    }
}

// a layer of one type: tu = tg gives the one pass's bits (tu = 0)
void test_one_type(cudaStream_t s) {
    const int D = 512, F = 256;
    Layer L = make(D, F, 4, iq::kIQ2_S, iq::kIQ2_S, iq::kIQ3_S);
    std::vector<int8_t> xq(D);
    std::vector<float> xd(D / 32);
    for (auto& v : xq) v = (int8_t) ((int) (rng() % 255) - 127);
    for (auto& v : xd) v = 0.004f;
    int8_t* dxq = dev(xq);
    float* dxd = dev(xd);
    std::vector<unsigned long long> pp(4);
    for (int i = 0; i < 4; ++i) pp[i] = (unsigned long long) (L.d + (size_t) i * L.blob);
    unsigned long long* dpp = dev(pp);
    std::vector<float> out[2];
    for (int k = 0; k < 2; ++k) {
        float* dh = dev(std::vector<float>((size_t) 4 * F));
        iq::Args a;
        a.plan_ptr = dpp;
        a.n_embd = D;
        a.n_ff = F;
        a.limit = -0.05f;
        a.xq = dxq;
        a.xd = dxd;
        a.h = dh;
        for (int r = 0; r < 3; ++r) a.tr[r] = L.tr[r];
        a.tg = L.ty[0];
        a.tu = k == 0 ? 0 : L.ty[0];
        a.td = L.ty[2];
        a.act = iq::kActSwiglu;
        iq::gate_up(a, 4, 1, s);
        CU(cudaStreamSynchronize(s));
        out[k] = host(dh, (size_t) 4 * F);
        cudaFree(dh);
    }
    same("IQ2_S/IQ2_S: tu = tg is the one pass (tu = 0)", out[0].data(), out[1].data(), out[0].size() * 4);
    cudaFree(dxq), cudaFree(dxd), cudaFree(dpp), cudaFree(L.d);
}

}  // namespace

int main() {
    cudaStream_t s;
    CU(cudaStreamCreate(&s));
    struct Case {
        int tg, tu, td, D, F;
    };
    // (Maya-M's and Maya-L's kinds of layers - an i-quant gate with a larger up - and a K-quant pair; the last at
    //  Ling-3.2-air's own geometry, n_embd 2560, n_ff 768)
    const Case cases[] = {
        {iq::kIQ2_S, iq::kIQ3_S, iq::kIQ3_S, 512, 256},  {iq::kIQ3_S, iq::kIQ4_XS, iq::kIQ4_XS, 512, 256},
        {iq::kIQ2_XXS, iq::kQ4_K, iq::kQ4_K, 512, 256},  {iq::kQ4_K, iq::kQ6_K, iq::kQ6_K, 512, 256},
        {iq::kIQ2_S, iq::kIQ4_XS, iq::kIQ3_S, 2560, 768},
        {iq::kIQ2_S, iq::kIQ2_S, iq::kIQ3_S, 2560, 768},   // (one type, the one pass: the same reference and bounds)
    };
    // (the clamp off, Ling's post-SiLU clamp - tight enough that it bites on both the SiLU and the up - DeepSeek's)
    const float limits[] = {0.0f, -0.05f, 0.08f};
    for (const Case& c : cases) {
        Layer L = make(c.D, c.F, 10, c.tg, c.tu, c.td);
        for (float lim : limits) {
            char nm[96];
            std::snprintf(nm, sizeof nm, "%s/%s/%s %dx%d lim %g", iq::name(c.tg), iq::name(c.tu), iq::name(c.td), c.D,
                          c.F, (double) lim);
            test_token_window(L, lim, nm, s);
            test_sets(L, lim, nm, s);
        }
        cudaFree(L.d);
    }
    test_one_type(s);
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
