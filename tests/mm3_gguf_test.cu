// tests/mm3_gguf_test.cu - maya-mm3's GGUF weights (src/mm3/mm3_gguf.cu over maya-ds4's src/ds4/ds4_iq.cu, the CPU
// lane's mm3_cpu_gg.cpp) on random blocks, no model:
//   * the dense types (Q4_K, Q5_K, Q6_K, Q8_0): the kernels' values, the FP16 rows and the host's rows equal to ggml's
//     dequantize_row_*; the GEMVs (1, 3, 16 activations) against a double reference on the int8 activations; the
//     prompt path's GEMMs (FP16 bands, one and several) against a double reference on the FP16 values;
//   * the routed experts with MiniMax's swigluoai - gate and up of one type, and of two (two passes): the token path
//     (an expert off the device, the CPU lane's part, the shared expert), a verify window (bit for bit the token
//     path's), the prompt sets (light kernels and tensor-core tiles; FP16 rows given or not: the same bits);
//   * the CPU lane's GGUF experts against the device's, and a window's tokens bit for bit each token alone;
//   * mm3_gv's dense GEMVs bit for bit ds4_iq's mv (MiniMax's row lengths, 1 to 16 activations, rows not a multiple of
//     a block's).
// mm3_gguf_test [--dense | --experts | --cpu]
#include "../src/mm3/mm3_cpu.hpp"
#include "../src/mm3/mm3_exl3.hpp"
#include "../src/mm3/mm3_gguf.cuh"
#include "../src/mm3/mm3_gguf_host.hpp"
#include "../src/mm3/mm3_gv.cuh"
#include "../src/ds4/ds4_iq.cuh"

#include "ggml.h"   // (the reference values: ggml's own dequantize_row_*)

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace x3 = strata::mm3::exl3;
namespace iq = strata::ds4::gguf::iq;
namespace gg = mm3::gg;

namespace {

int g_fail = 0;
std::mt19937 rng(4321);

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);      \
            std::exit(1);                                                                      \
        }                                                                                      \
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

void expect(const char* what, const float* got, const double* want, size_t n, double rtol) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            ++g_fail;
            std::printf("%-52s FAIL  not finite (at %zu: %.6g vs %.6g)\n", what, i, (double) got[i], want[i]);
            return;
        }
    double worst = 0, scale = 1e-30;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs(want[i]));
    size_t at = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs(got[i] - want[i]) / scale;
        if (!(e <= worst)) {
            worst = e;
            at = i;
        }
    }
    const bool ok = worst <= rtol;
    if (!ok) ++g_fail;
    std::printf("%-52s %s  max rel err %.2e (at %zu: %.6g vs %.6g)\n", what, ok ? "ok  " : "FAIL", worst, at,
                (double) got[at], want[at]);
}
void same(const char* what, const void* a, const void* b, size_t bytes) {
    const bool ok = std::memcmp(a, b, bytes) == 0;
    if (!ok) ++g_fail;
    std::printf("%-52s %s\n", what, ok ? "ok    bit for bit" : "FAIL  differs");
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

// random blocks of a type (ds4_gguf_test's q_rand): random bytes, the block scales small normal F16s
std::vector<uint8_t> q_rand(int type, int n, int k) {
    const size_t rb = iq::row_bytes(type, k), bb = (size_t) iq::block_bytes(type);
    std::vector<uint8_t> w(rb * n);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    auto put = [&](size_t at, float f) {
        const uint16_t u = f2h(f);
        std::memcpy(&w[at], &u, 2);
    };
    for (size_t o = 0; o < w.size(); o += bb) {
        const float d = 0.001f + 0.002f * std::fabs(frand());
        switch (type) {
            case iq::kQ4_K:
            case iq::kQ5_K: put(o, d), put(o + 2, 0.5f * d); break;
            case iq::kQ6_K: put(o + 208, d); break;
            case iq::kIQ1_M: {   // (d: an F16 in the top nibbles of the 4 scale words, bytes 48..55)
                const uint16_t u = f2h(d);
                for (int i = 0; i < 4; ++i) {
                    uint16_t sw;
                    std::memcpy(&sw, &w[o + 48 + 2 * i], 2);
                    sw = (uint16_t) ((sw & 0x0FFFu) | (((u >> (4 * i)) & 0xFu) << 12));
                    std::memcpy(&w[o + 48 + 2 * i], &sw, 2);
                }
                break;
            }
            default: put(o, d); break;   // (the i-quants and Q8_0: d first)
        }
    }
    return w;
}
std::vector<float> deq(int type, const std::vector<uint8_t>& w, size_t n) {
    std::vector<float> v(n);
    ggml_get_type_traits((ggml_type) type)->to_float(w.data(), v.data(), (int64_t) n);
    return v;
}
// x rounded as the GEMVs (and the light sets) round it: a scale a 32, max|x| / 127
std::vector<float> q8_round(const float* x, int k) {
    std::vector<float> v(k);
    for (int b = 0; b < k / 32; ++b) {
        float am = 0.0f;
        for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(x[b * 32 + j]));
        const float sc = am / 127.0f;
        for (int j = 0; j < 32; ++j) v[b * 32 + j] = am > 0.0f ? (float) ((double) std::nearbyint(x[b * 32 + j] / sc) * sc) : 0.0f;
    }
    return v;
}
// rows rounded by the device as the experts' activations are (gg::act_quant): their values
std::vector<float> act_dev(const float* dx, int rows, int k, cudaStream_t s) {
    int8_t* q = nullptr;
    float* d = nullptr;
    CU(cudaMalloc(&q, (size_t) rows * k));
    CU(cudaMalloc(&d, (size_t) rows * k / 32 * 4));
    gg::act_quant(dx, rows, k, q, d, s);
    CU(cudaStreamSynchronize(s));
    const std::vector<int8_t> hq = host(q, (size_t) rows * k);
    const std::vector<float> hd = host(d, (size_t) rows * k / 32);
    std::vector<float> v((size_t) rows * k);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float) ((double) hq[i] * hd[i / 32]);
    cudaFree(q), cudaFree(d);
    return v;
}
double oai(double g, double u, double lim) {
    g = std::min(g, lim);
    u = std::min(std::max(u, -lim), lim);
    return g / (1.0 + std::exp(-1.702 * g)) * (u + 1.0);
}
double dot(const float* w, const float* x, int k) {
    double s = 0;
    for (int i = 0; i < k; ++i) s += (double) w[i] * x[i];
    return s;
}

// ---------------------------------------------------------------- the dense types
const int kDense[] = {iq::kQ4_K, iq::kQ5_K, iq::kQ6_K, iq::kQ8_0};

void test_dense(cudaStream_t s) {
    for (int t : kDense) {
        const int k = 1024, n = 96;
        const std::vector<uint8_t> w = q_rand(t, n, k);
        const std::vector<float> want = deq(t, w, (size_t) n * k);
        uint8_t* dw = dev(w);
        // the kernels' values (token and tile decoders), the FP16 rows, the host's rows
        float* dsub = dev(std::vector<float>((size_t) n * k, -1.0f));
        float* dtile = dev(std::vector<float>((size_t) n * k, -1.0f));
        uint16_t* d16 = nullptr;
        CU(cudaMalloc(&d16, (size_t) n * k * 2));
        const bool r1 = iq::decode_values(t, dw, k, n, dsub, dtile, s), r2 = iq::f16_rows(t, dw, k, n, d16, k, s);
        CU(cudaStreamSynchronize(s));
        const std::vector<float> gs = host(dsub, (size_t) n * k), gt = host(dtile, (size_t) n * k);
        const std::vector<uint16_t> gh = host(d16, (size_t) n * k);
        std::vector<float> hr((size_t) n * k);
        bool hok = true;
        for (int r = 0; r < n; ++r) hok = hok && gg::host_row(iq::name(t), w.data(), k, r, hr.data() + (size_t) r * k);
        size_t bs = 0, bt = 0, bh = 0, bhost = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            bs += !(gs[i] == want[i]);
            bt += !(gt[i] == want[i]);
            bh += gh[i] != f2h(want[i]) && !((gh[i] | f2h(want[i])) == 0x8000u);   // (a zero's sign aside)
            bhost += !(hr[i] == want[i]);
        }
        const bool ok = r1 && r2 && hok && bs == 0 && bt == 0 && bh == 0 && bhost == 0;
        if (!ok) ++g_fail;
        std::printf("%-52s %s  %d values: kernels %zu, tiles %zu, f16_rows %zu, host %zu differ%s\n",
                    (std::string("rows ") + iq::name(t) + " = ggml's").c_str(), ok ? "ok  " : "FAIL", n * k, bs, bt, bh,
                    bhost, r1 && r2 && hok ? "" : " (not run)");
        // the GEMVs (through mm3_exl3's mv: a GGUF Mat)
        x3::Mat m;
        m.fmt = gg::kFmtGG;
        m.gg_type = t;
        m.raw = dw;
        m.k = k;
        m.n = n;
        m.ld = (int) iq::row_bytes(t, k);
        for (int nc : {1, 3, 16}) {
            const size_t xcs = (size_t) k + 32;
            std::vector<float> x(xcs * nc), bias(n);
            for (auto& v : x) v = frand();
            for (auto& v : bias) v = frand(0.1f);
            float *dx = dev(x), *db = dev(bias), *dy = dev(std::vector<float>((size_t) n * nc));
            x3::MvJob j;
            j.m = &m;
            j.xf = dx;
            j.y = dy;
            j.bias = db;
            j.alpha = 0.5f;
            j.ncol = nc;
            j.xf_cs = xcs;
            j.y_cs = n;
            if (!x3::mv(&j, 1, s)) ++g_fail;
            CU(cudaStreamSynchronize(s));
            const std::vector<float> y = host(dy, (size_t) n * nc);
            std::vector<double> ref((size_t) n * nc);
            for (int c = 0; c < nc; ++c) {
                const std::vector<float> xr = q8_round(x.data() + c * xcs, k);
                for (int r = 0; r < n; ++r)
                    ref[(size_t) c * n + r] = 0.5 * dot(want.data() + (size_t) r * k, xr.data(), k) + bias[r];
            }
            char what[96];
            std::snprintf(what, sizeof what, "mv %s, %d activations", iq::name(t), nc);
            expect(what, y.data(), ref.data(), ref.size(), 2e-5);
            cudaFree(dx), cudaFree(db), cudaFree(dy);
        }
        // the prompt path's GEMM: y = beta y + x W over FP16 rows
        for (float beta : {0.0f, 1.0f}) {
            const int rows = 40, ldx = k + 16, ldy = n + 8;
            std::vector<uint16_t> x16((size_t) rows * ldx);
            for (auto& v : x16) v = f2h(frand());
            std::vector<float> y0((size_t) rows * ldy);
            for (auto& v : y0) v = frand();
            uint16_t* dx = dev(x16);
            float* dy = dev(y0);
            x3::gemm_rows(m, dx, ldx, rows, dy, ldy, beta, s);
            CU(cudaStreamSynchronize(s));
            const std::vector<float> y = host(dy, y0.size());
            std::vector<float> got;
            std::vector<double> ref;
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < n; ++c) {
                    double a = beta * y0[(size_t) r * ldy + c];
                    for (int i = 0; i < k; ++i) a += (double) f16r(want[(size_t) c * k + i]) * h2f(x16[(size_t) r * ldx + i]);
                    ref.push_back(a);
                    got.push_back(y[(size_t) r * ldy + c]);
                }
            char what[96];
            std::snprintf(what, sizeof what, "gemm_rows %s, %d rows, beta %g", iq::name(t), rows, beta);
            expect(what, got.data(), ref.data(), ref.size(), 1e-4);
            cudaFree(dx), cudaFree(dy);
        }
        cudaFree(dw), cudaFree(dsub), cudaFree(dtile), cudaFree(d16);
    }
    {   // a matrix past one FP16 band (64 MB): its rows in several
        const int t = iq::kQ8_0, k = 8192, n = 4608, rows = 6;
        const std::vector<uint8_t> w = q_rand(t, n, k);
        const std::vector<float> want = deq(t, w, (size_t) n * k);
        uint8_t* dw = dev(w);
        x3::Mat m;
        m.fmt = gg::kFmtGG;
        m.gg_type = t;
        m.raw = dw;
        m.k = k;
        m.n = n;
        std::vector<uint16_t> x16((size_t) rows * k);
        for (auto& v : x16) v = f2h(frand());
        uint16_t* dx = dev(x16);
        float* dy = dev(std::vector<float>((size_t) rows * n));
        x3::gemm_rows(m, dx, k, rows, dy, n, 0.0f, s);
        CU(cudaStreamSynchronize(s));
        const std::vector<float> y = host(dy, (size_t) rows * n);
        std::vector<double> ref((size_t) rows * n);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < n; ++c) {
                double a = 0;
                for (int i = 0; i < k; ++i) a += (double) f16r(want[(size_t) c * k + i]) * h2f(x16[(size_t) r * k + i]);
                ref[(size_t) r * n + c] = a;
            }
        expect("gemm_rows Q8_0 8192 x 4608 (two FP16 bands)", y.data(), ref.data(), ref.size(), 1e-4);
        cudaFree(dw), cudaFree(dx), cudaFree(dy);
    }
}

// ---------------------------------------------------------------- the routed experts
struct QE {
    int D, F, n_exp, ty[3];
    x3::ExpertLayout lay;
    size_t blob = 0;
    std::vector<uint8_t> h;
    std::vector<std::vector<float>> w;   // [e * 3 + r]: ggml's values, [rows][k]
    uint8_t* d = nullptr;
    const uint8_t* mat(int e, int r) const { return h.data() + (size_t) e * blob + lay.trellis[r]; }
    const float* row(int e, int r, int i) const { return w[(size_t) e * 3 + r].data() + (size_t) i * (r < 2 ? D : F); }
};
QE make(int D, int F, int n_exp, int tg, int tu, int td) {
    QE E{D, F, n_exp, {tg, tu, td}};
    size_t off = 0;
    for (int r = 0; r < 3; ++r) {
        E.lay.trellis[r] = E.lay.suh[r] = E.lay.svh[r] = off;
        E.lay.gg_type[r] = E.ty[r];
        off += ((r < 2 ? F : D) * iq::row_bytes(E.ty[r], r < 2 ? D : F) + 255) / 256 * 256;
    }
    E.lay.fmt = gg::kFmtGG;
    E.blob = off;
    E.h.assign(E.blob * n_exp, 0);
    for (int e = 0; e < n_exp; ++e)
        for (int r = 0; r < 3; ++r) {
            const int rows = r < 2 ? F : D, k = r < 2 ? D : F;
            const std::vector<uint8_t> m = q_rand(E.ty[r], rows, k);
            std::memcpy(E.h.data() + (size_t) e * E.blob + E.lay.trellis[r], m.data(), m.size());
            E.w.push_back(deq(E.ty[r], m, (size_t) rows * k));
        }
    E.d = dev(E.h);
    return E;
}

void test_experts(int tg, int tu, int td, cudaStream_t s) {
    const int D = 512, F = 256, n_exp = 10, K = 4, T = 3;
    const float lim = 7.0f;
    QE E = make(D, F, n_exp, tg, tu, td);
    char name[64];
    std::snprintf(name, sizeof name, "%s/%s/%s", iq::name(tg), iq::name(tu), iq::name(td));
    // the plans: token 1's third pick off the device, token 2 with the CPU lane's part
    std::vector<int> pick((size_t) T * 8, 0);
    std::vector<float> pw((size_t) T * 8, 0.0f);
    std::vector<unsigned long long> pp((size_t) T * 8, 0ull);
    for (int t = 0; t < T; ++t) {
        std::vector<int> all(n_exp);
        for (int i = 0; i < n_exp; ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        for (int i = 0; i < K; ++i) {
            pick[t * 8 + i] = all[i];
            pw[t * 8 + i] = 0.2f + 0.1f * frand();
            pp[t * 8 + i] = t == 1 && i == 2 ? 0ull : (unsigned long long) (E.d + (size_t) all[i] * E.blob);
        }
    }
    std::vector<float> x((size_t) T * D), sh((size_t) T * D), cpu((size_t) T * D);
    for (auto& v : x) v = frand();
    for (auto& v : sh) v = frand(0.1f);
    for (auto& v : cpu) v = frand(0.1f);
    std::vector<int> cflag(T, 0);
    cflag[2] = 1;
    float *dx = dev(x), *dsh = dev(sh), *dcpu = dev(cpu), *dpw = dev(pw);
    unsigned long long* dpp = dev(pp);
    int* dflag = dev(cflag);
    float* dh = dev(std::vector<float>((size_t) 8 * F));
    float* dout = dev(std::vector<float>(D));
    std::vector<float> tok((size_t) T * D), gh((size_t) T * 8 * F);
    for (int t = 0; t < T; ++t) {
        x3::MoeArgs a;
        a.plan_ptr = dpp + t * 8;
        a.plan_w = dpw + t * 8;
        a.cpu_part = dcpu + (size_t) t * D;
        a.cpu_flag = dflag + t;
        a.lay = E.lay;
        a.k = K;
        a.n_embd = D;
        a.n_ff = F;
        a.limit = lim;
        x3::moe_gate_up(a, dx + (size_t) t * D, dh, s);
        x3::moe_down(a, dh, dsh + (size_t) t * D, dout, s);
        CU(cudaStreamSynchronize(s));
        CU(cudaMemcpy(tok.data() + (size_t) t * D, dout, (size_t) D * 4, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(gh.data() + (size_t) t * 8 * F, dh, (size_t) 8 * F * 4, cudaMemcpyDeviceToHost));
    }
    // the reference: gate / up in double on the device's rounded x, the down on the device's own h rounded the same way
    const std::vector<float> xr = act_dev(dx, T, D, s);
    std::vector<double> want((size_t) T * D, 0.0), hwant;
    std::vector<float> hgot;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < K; ++i) {
            if (pp[t * 8 + i] == 0ull) continue;
            const int e = pick[t * 8 + i];
            const float* h = gh.data() + ((size_t) t * 8 + i) * F;
            for (int f = 0; f < F; ++f) {
                hwant.push_back(oai(dot(E.row(e, 0, f), xr.data() + (size_t) t * D, D),
                                    dot(E.row(e, 1, f), xr.data() + (size_t) t * D, D), lim));
                hgot.push_back(h[f]);
            }
            float* dhi = dev(std::vector<float>(h, h + F));
            const std::vector<float> hr = act_dev(dhi, 1, F, s);
            cudaFree(dhi);
            for (int d = 0; d < D; ++d) want[(size_t) t * D + d] += pw[t * 8 + i] * dot(E.row(e, 2, d), hr.data(), F);
        }
        for (int d = 0; d < D; ++d) want[(size_t) t * D + d] += (cflag[t] ? cpu[(size_t) t * D + d] : 0.0f) + sh[(size_t) t * D + d];
    }
    char what[128];
    std::snprintf(what, sizeof what, "%s token experts: h (swigluoai)", name);
    expect(what, hgot.data(), hwant.data(), hwant.size(), 1e-4);
    std::snprintf(what, sizeof what, "%s token experts: out", name);
    expect(what, tok.data(), want.data(), want.size(), 1e-4);
    // the window: each distinct expert once for its tokens
    std::vector<unsigned long long> gptr;
    std::vector<int> gstart, gent;
    for (int x0 = 0; x0 < n_exp; ++x0) {
        std::vector<int> ents;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < K; ++i)
                if (pick[t * 8 + i] == x0 && pp[t * 8 + i] != 0ull) ents.push_back(t * 8 + i);
        if (ents.empty()) continue;
        gptr.push_back((unsigned long long) (E.d + (size_t) x0 * E.blob));
        gstart.push_back((int) gent.size());
        for (int e : ents) gent.push_back(e);
    }
    gstart.push_back((int) gent.size());
    std::vector<int> ng(1, (int) gptr.size());
    unsigned long long* dgp = dev(gptr);
    int *dgs = dev(gstart), *dge = dev(gent), *dng = dev(ng);
    float* dwh = dev(std::vector<float>((size_t) T * 8 * F));
    float* dpart = dev(std::vector<float>((size_t) T * 8 * D));
    float* dwo = dev(std::vector<float>((size_t) T * D));
    x3::WinArgs w;
    w.grp_ptr = dgp;
    w.grp_start = dgs;
    w.grp_ent = dge;
    w.n_groups = dng;
    w.plan_ptr = dpp;
    w.plan_w = dpw;
    w.cpu_part = dcpu;
    w.cpu_stride = D;
    w.cpu_flag = dflag;
    w.x = dx;
    w.x_stride = D;
    w.h = dwh;
    w.part = dpart;
    w.sh_out = dsh;
    w.out = dwo;
    w.out_stride = D;
    w.lay = E.lay;
    w.T = T;
    w.k = K;
    w.n_embd = D;
    w.n_ff = F;
    w.limit = lim;
    if (!x3::win_experts(w, s)) ++g_fail;
    CU(cudaStreamSynchronize(s));
    const std::vector<float> win = host(dwo, (size_t) T * D);
    std::snprintf(what, sizeof what, "%s window (T %d) = the token path's", name, T);
    same(what, win.data(), tok.data(), win.size() * 4);
    // the prompt sets: experts of a few rows (the light kernels), then of more (the tiles)
    for (int R : {60, 400}) {
        const int NT = 30;
        std::vector<int> owner(R), idx(R);
        for (int r = 0; r < R; ++r) owner[r] = (int) (rng() % n_exp);
        std::sort(owner.begin(), owner.end());
        std::vector<int> bounds(n_exp + 1, 0);
        for (int e = 0, r = 0; e <= n_exp; ++e) {
            while (r < R && owner[r] < e) ++r;
            bounds[e] = r;
        }
        int max_rows = 0;
        for (int e = 0; e < n_exp; ++e) max_rows = std::max(max_rows, bounds[e + 1] - bounds[e]);
        std::vector<float> xs((size_t) NT * D);
        for (auto& v : xs) v = frand();
        std::vector<uint16_t> x16(xs.size());
        for (size_t i = 0; i < xs.size(); ++i) x16[i] = f2h(xs[i]);
        for (int r = 0; r < R; ++r) idx[r] = (int) (rng() % NT);
        float* dxs = dev(xs);
        uint16_t* dx16 = dev(x16);
        int *dbd = dev(bounds), *didx = dev(idx);
        uint16_t* dh16 = dev(std::vector<uint16_t>((size_t) R * F));
        uint16_t* dh16b = dev(std::vector<uint16_t>((size_t) R * F));
        float* dy = dev(std::vector<float>((size_t) R * D));
        float* graw = dev(std::vector<float>((size_t) R * F));
        x3::SetArgs sa;
        sa.wbase = E.d;
        sa.stride = E.blob;
        sa.lay = E.lay;
        sa.n_exp = n_exp;
        sa.bounds = dbd;
        sa.row0 = 0;
        sa.max_rows = max_rows;
        sa.n_embd = D;
        sa.n_ff = F;
        sa.limit = lim;
        sa.graw = graw;
        sa.x16 = dx16;
        x3::set_gate_up(sa, dxs, D, didx, dh16, F, s);
        x3::set_down(sa, dh16, F, dy, D, s);
        x3::SetArgs sb = sa;
        sb.x16 = nullptr;
        x3::set_gate_up(sb, dxs, D, didx, dh16b, F, s);
        CU(cudaStreamSynchronize(s));
        const std::vector<uint16_t> h16 = host(dh16, (size_t) R * F), h16b = host(dh16b, (size_t) R * F);
        const std::vector<float> y = host(dy, (size_t) R * D);
        std::snprintf(what, sizeof what, "%s set gate/up, %d rows: FP16 rows given or not", name, R);
        same(what, h16.data(), h16b.data(), h16.size() * 2);
        std::vector<float> hg((size_t) R * F);
        std::vector<double> hw((size_t) R * F), yw((size_t) R * D);
        int n_light = 0;
        for (int e = 0; e < n_exp; ++e) n_light += bounds[e + 1] - bounds[e] <= 16 ? bounds[e + 1] - bounds[e] : 0;
        for (int r = 0; r < R; ++r) {
            const int e = owner[r];
            const bool light = bounds[e + 1] - bounds[e] <= 16;   // (the light kernels: int8 inputs per 32; the tiles: FP16)
            std::vector<float> xi(xs.begin() + (long) idx[r] * D, xs.begin() + (long) (idx[r] + 1) * D);
            if (light) xi = q8_round(xi.data(), D);
            else
                for (auto& v : xi) v = f16r(v);
            std::vector<float> hr(F);
            for (int f = 0; f < F; ++f) {
                double g = 0, u = 0;
                for (int i = 0; i < D; ++i) {
                    const float wg = E.row(e, 0, f)[i], wu = E.row(e, 1, f)[i];
                    g += (double) (light ? wg : f16r(wg)) * xi[i];
                    u += (double) (light ? wu : f16r(wu)) * xi[i];
                }
                hw[(size_t) r * F + f] = oai(g, u, lim);
                hg[(size_t) r * F + f] = hr[f] = h2f(h16[(size_t) r * F + f]);
            }
            if (light) hr = q8_round(hr.data(), F);
            for (int d = 0; d < D; ++d) {
                double a = 0;
                for (int i = 0; i < F; ++i) a += (double) (light ? E.row(e, 2, d)[i] : f16r(E.row(e, 2, d)[i])) * hr[i];
                yw[(size_t) r * D + d] = a;
            }
        }
        std::snprintf(what, sizeof what, "%s set gate/up (F16 h), %d rows, %d light", name, R, n_light);
        expect(what, hg.data(), hw.data(), hw.size(), 2e-3);
        std::snprintf(what, sizeof what, "%s set down, %d rows, %d light", name, R, n_light);
        expect(what, y.data(), yw.data(), yw.size(), 1e-4);
        cudaFree(dxs), cudaFree(dx16), cudaFree(dbd), cudaFree(didx), cudaFree(dh16), cudaFree(dh16b), cudaFree(dy);
        cudaFree(graw);
    }
    // the CPU lane: token 0's experts (all on the device there, no CPU part) against the device's, then a window of
    // the three tokens bit for bit each token alone
    {
        mm3::GgFmt f;
        std::string err;
        if (!mm3::gg_fmt(tg, tu, td, D, F, f, err)) {
            std::printf("%-52s FAIL  %s\n", "  CPU lane", err.c_str());
            ++g_fail;
        } else {
            mm3::CpuLane lane(3, D, F, 1.702f, lim);
            std::vector<mm3::CpuExpert> ex;
            for (int i = 0; i < K; ++i) {
                mm3::CpuExpert c;
                c.gf = &f;
                c.w = pw[i];
                for (int r = 0; r < 3; ++r) c.gg[r] = E.mat(pick[i], r);
                ex.push_back(c);
            }
            std::vector<float> out(D);
            mm3::gg_run(lane, D, F, lim, ex, x.data(), out.data());
            std::vector<double> dref(D);
            for (int d = 0; d < D; ++d) dref[d] = (double) tok[d] - sh[d];
            std::snprintf(what, sizeof what, "%s CPU lane token = the device's", name);
            expect(what, out.data(), dref.data(), dref.size(), 2e-3);
            // a window: tokens 0..2, each with its own experts (their union), users in slot order
            std::vector<mm3::CpuExpert> wx;
            std::vector<std::vector<std::pair<int, float>>> users;
            std::vector<std::pair<int, int>> order;
            std::vector<int> slot_of(n_exp, -1);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < K; ++i) {
                    const int e = pick[t * 8 + i];
                    if (slot_of[e] < 0) {
                        slot_of[e] = (int) wx.size();
                        mm3::CpuExpert c;
                        c.gf = &f;
                        for (int r = 0; r < 3; ++r) c.gg[r] = E.mat(e, r);
                        wx.push_back(c);
                        users.emplace_back();
                    }
                    users[(size_t) slot_of[e]].push_back({t, pw[t * 8 + i]});
                    order.push_back({slot_of[e], (int) users[(size_t) slot_of[e]].size() - 1});
                }
            std::vector<float> wout((size_t) T * D);
            mm3::gg_run_multi(lane, D, F, lim, wx, users, x.data(), T, wout.data(), &order);
            bool ok = true;
            for (int t = 0; t < T; ++t) {
                std::vector<mm3::CpuExpert> one;
                for (int i = 0; i < K; ++i) {
                    mm3::CpuExpert c = wx[(size_t) slot_of[pick[t * 8 + i]]];
                    c.w = pw[t * 8 + i];
                    one.push_back(c);
                }
                std::vector<float> o1(D);
                mm3::gg_run(lane, D, F, lim, one, x.data() + (size_t) t * D, o1.data());
                ok = ok && std::memcmp(o1.data(), wout.data() + (size_t) t * D, (size_t) D * 4) == 0;
            }
            if (!ok) ++g_fail;
            std::snprintf(what, sizeof what, "%s CPU lane window (T %d) = each token alone", name, T);
            std::printf("%-52s %s\n", what, ok ? "ok    bit for bit" : "FAIL  differs");
        }
    }
    cudaFree(E.d), cudaFree(dx), cudaFree(dsh), cudaFree(dcpu), cudaFree(dpw), cudaFree(dpp), cudaFree(dflag);
    cudaFree(dh), cudaFree(dout), cudaFree(dgp), cudaFree(dgs), cudaFree(dge), cudaFree(dng), cudaFree(dwh);
    cudaFree(dpart), cudaFree(dwo);
}

// gate / up / down types: MiniMax-M3.1's Maya-XS and Maya-S layers (one type, and gate / up of two), and the i-quants
const int kTriples[][3] = {
    {iq::kIQ1_S, iq::kIQ1_S, iq::kIQ1_S},       {iq::kIQ1_S, iq::kIQ1_S, iq::kIQ2_XXS},
    {iq::kIQ2_XXS, iq::kIQ1_S, iq::kIQ2_XXS},   {iq::kIQ2_XS, iq::kIQ2_XXS, iq::kIQ3_XXS},
    {iq::kIQ2_S, iq::kIQ2_S, iq::kIQ3_XXS},     {iq::kIQ3_XXS, iq::kIQ3_XXS, iq::kIQ3_XXS},
    {iq::kIQ2_XXS, iq::kIQ2_XS, iq::kIQ2_XS},   {iq::kIQ4_XS, iq::kIQ3_S, iq::kIQ1_M}};

// mm3_gv's GEMVs against ds4_iq's mv, bit for bit: every dense type, MiniMax's row lengths, 1 .. 16 columns
__global__ void q8s_ref_kernel(const float* x, int k, int8_t* q, float* d, int* sums) {   // (the int8 rows + sums)
    const int r = blockIdx.y, b = blockIdx.x, lane = threadIdx.x;
    const float v = x[(size_t) r * k + (size_t) b * 32 + lane];
    float am = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;
    const int qi = am > 0.0f ? __float2int_rn(v / sc) : 0;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) qi;
    int sm = qi;
    for (int o = 8; o > 0; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
    const int hi = __shfl_sync(0xffffffffu, sm, 16);
    if (lane == 0) {
        d[(size_t) r * (k / 32) + b] = sc;
        sums[(size_t) r * (k / 32) + b] = (sm & 0xffff) | (hi << 16);
    }
}
void test_gv(cudaStream_t s) {
    for (int t : kDense)
        for (int k : {3072, 6144, 8192})
            for (int nc : {1, 2, 5, 7, 16}) {
                const int n = 203;   // (not a multiple of any rows-a-block)
                const std::vector<uint8_t> w = q_rand(t, n, k);
                uint8_t* dw = dev(w);
                std::vector<float> x((size_t) nc * k);
                for (auto& v : x) v = frand(2.0f);
                std::vector<float> bias(n);
                for (auto& v : bias) v = frand(0.1f);
                float *dx = dev(x), *db = dev(bias), *xd = nullptr, *y0 = nullptr, *y1 = nullptr;
                int8_t* xq = nullptr;
                int* xs = nullptr;
                CU(cudaMalloc(&xq, x.size()));
                CU(cudaMalloc(&xd, x.size() / 32 * 4));
                CU(cudaMalloc(&xs, x.size() / 32 * 4));
                CU(cudaMalloc(&y0, (size_t) nc * n * 4));
                CU(cudaMalloc(&y1, (size_t) nc * n * 4));
                CU(cudaMemset(y1, 0xff, (size_t) nc * n * 4));
                q8s_ref_kernel<<<dim3((unsigned) (k / 32), (unsigned) nc), 32, 0, s>>>(dx, k, xq, xd, xs);
                iq::MvJob a;
                a.w = dw, a.ld = iq::row_bytes(t, k), a.type = t, a.k = k, a.n = n, a.ncol = nc, a.xq = xq, a.xd = xd;
                a.y = y0, a.ycs = (size_t) n, a.bias = db, a.alpha = 0.75f;
                mm3::gv::Job b;
                b.w = dw, b.ld = a.ld, b.type = t, b.k = k, b.n = n, b.ncol = nc, b.xq = xq, b.xd = xd, b.xs = xs;
                b.y = y1, b.ycs = (size_t) n, b.bias = db, b.alpha = 0.75f;
                const bool cov = mm3::gv::covered(b);
                const bool r0 = iq::mv(&a, 1, s), r1 = cov && mm3::gv::run(&b, 1, s);
                CU(cudaStreamSynchronize(s));
                const std::vector<float> h0 = host(y0, (size_t) nc * n), h1 = host(y1, (size_t) nc * n);
                char what[96];
                std::snprintf(what, sizeof what, "gv %s k %d, %d activations = ds4_iq's mv", iq::name(t), k, nc);
#if defined(STRATA_USE_HIP)
                const bool hip = true;   // (HIP: the GEMVs stay ds4_iq's - mm3_gv covers nothing there)
#else
                const bool hip = false;
#endif
                if (!cov && (hip || (t == iq::kQ6_K && k % 1024 != 0))) {   // (Q6_K rows of 3072: ds4_iq's mv)
                    std::printf("%-52s ok    not covered (ds4_iq's mv)\n", what);
                } else if (!r0 || !r1) {
                    ++g_fail;
                    std::printf("%-52s FAIL  not run\n", what);
                } else {
                    same(what, h0.data(), h1.data(), h0.size() * 4);
                }
                cudaFree(dw), cudaFree(dx), cudaFree(db), cudaFree(xd), cudaFree(y0), cudaFree(y1), cudaFree(xq),
                    cudaFree(xs);
            }
}

}  // namespace

int main(int argc, char** argv) {
    cudaStream_t s;
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const std::string only = argc >= 2 ? argv[1] : "";
    if (only.empty() || only == "--dense") test_dense(s);
    if (only.empty() || only == "--dense" || only == "--gv") test_gv(s);
    if (only.empty() || only == "--experts" || only == "--cpu")
        for (const auto& tr : kTriples) test_experts(tr[0], tr[1], tr[2], s);
    if (x3::launch_errors() != 0) {
        std::printf("launch errors: %d\n", x3::launch_errors());
        ++g_fail;
    }
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
