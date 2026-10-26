// tests/ds4_gguf_test.cu - maya-ds4's GGUF kernels (src/ds4/ds4_gguf.cu, ds4_gguf_cpu.cpp) against a plain
// double-precision reference of random BF16 matrices and MXFP4 experts: the GEMVs (BF16 / MXFP4, 1..16 activations,
// aligned or not), the token experts (gate / up / swiglu / down / combine, experts off the device, the CPU part, the
// shared expert), a window's grouped experts (bit for bit the token path's), the prompt path's expert sets and the CPU
// lane's rows - for MXFP4 and every low-bit type (IQ1_S .. Q4_K; their decode equal to ggml's dequantize_row_* on
// random blocks); the PCIe lane's streamed experts (bit for bit the VRAM ones'); and the KV caches' packed rows
// (ds4_kernels.cu) against the BF16 rows bit for bit.  Needs a GPU, no model.
// ds4_gguf_test [n_embd n_ff] | --kv (the caches alone) | --lane (the PCIe lane alone)
#include "../src/ds4/ds4_gguf.cuh"
#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "../src/ds4/ds4_iq.cuh"
#include "../src/ds4/ds4_kernels.cuh"
#include "../src/ds4/ds4_pcie.cuh"
#include "../src/ds4/ds4_q6k.cuh"

#include "ggml.h"   // (the low-bit types' reference values: ggml's own dequantize_row_*)

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace x3 = strata::kernels::exl3;

namespace {

int g_fail = 0;
std::mt19937 rng(1234);

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);      \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

float frand(float a = 1.0f) { return std::uniform_real_distribution<float>(-a, a)(rng); }

float bf16f(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f2bf(float f) {   // truncating is fine: the reference reads back the same bits
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
const float kVal[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
double e8m0_half(int e) { return std::ldexp(1.0, e - 128); }

// random MXFP4 rows [n][k]: scales around 2^-6, every code
std::vector<uint8_t> mx_rand(int n, int k) {
    std::vector<uint8_t> w(gg::mx_bytes(n, k));
    for (size_t b = 0; b < w.size() / 17; ++b) {
        w[b * 17] = (uint8_t) (120 + rng() % 6);
        for (int j = 1; j < 17; ++j) w[b * 17 + j] = (uint8_t) (rng() & 0xff);
    }
    return w;
}
double mx_val(const uint8_t* row, int j) {   // value j of a row
    const uint8_t* p = row + (size_t) (j / 32) * 17;
    const int jj = j % 32;
    const int code = jj < 16 ? (p[1 + jj] & 15) : (p[1 + jj - 16] >> 4);
    return kVal[code] * e8m0_half(p[0]);
}
// random Q8_0 rows [n][k], and a value of one
std::vector<uint8_t> q8_rand(int n, int k) {
    std::vector<uint8_t> w(gg::q8_row_bytes(k) * n);
    for (size_t b = 0; b < w.size() / 34; ++b) {
        const __half d = __float2half(0.001f + 0.002f * std::fabs(frand()));
        std::memcpy(&w[b * 34], &d, 2);
        for (int j = 2; j < 34; ++j) w[b * 34 + j] = (uint8_t) (rng() & 0xff);
    }
    return w;
}
double q8_val(const uint8_t* row, int j) {
    const uint8_t* p = row + (size_t) (j / 32) * 34;
    __half d;
    std::memcpy(&d, p, 2);
    return (double) __half2float(d) * (int8_t) p[2 + j % 32];
}
double mx_dot(const uint8_t* w, int k, int row, const float* x) {
    const uint8_t* r = w + (size_t) row * gg::mx_row_bytes(k);
    double s = 0;
    for (int j = 0; j < k; ++j) s += mx_val(r, j) * x[j];
    return s;
}
// x rounded as the kernels' q8_quant: per 32 values sc = max|x| / 127 (F32), q = round-half-even(x / sc)
std::vector<float> q8_round(const float* x, int k) {
    std::vector<float> v(k);
    for (int b = 0; b < k / 32; ++b) {
        float am = 0.0f;
        for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(x[b * 32 + j]));
        const float sc = am / 127.0f;
        for (int j = 0; j < 32; ++j) {
            const float q = am > 0.0f ? std::nearbyint(x[b * 32 + j] / sc) : 0.0f;
            v[b * 32 + j] = (float) ((double) q * sc);
        }
    }
    return v;
}
// x rounded as the low-bit experts' kernels round it (iq_act_quant): ggml's Q8_K per 256 values (max = the signed
// value of the largest magnitude, the first one; iscale = -127 / max; q = min(127, round-half-even(iscale x)), the
// product rounded to F32 first; d = 1 / iscale), or q8_round's per 32 with DS4_IQ_Q8K=0 or k not a multiple of 256
std::vector<float> act_round(const float* x, int k) {
    const char* on = std::getenv("DS4_IQ_Q8K");
    if ((on != nullptr && std::atoi(on) == 0) || k % 256 != 0) return q8_round(x, k);
    std::vector<float> v(k, 0.0f);
    for (int b = 0; b < k / 256; ++b) {
        const float* xb = x + (size_t) b * 256;
        int at = 0;
        for (int j = 1; j < 256; ++j)
            if (std::fabs(xb[j]) > std::fabs(xb[at])) at = j;
        if (xb[at] == 0.0f) continue;
        const float iscale = -127.0f / xb[at], d = 1.0f / iscale;
        for (int j = 0; j < 256; ++j) {
            const float q = std::min(127.0f, std::nearbyint(iscale * xb[j]));
            v[(size_t) b * 256 + j] = (float) ((double) q * d);
        }
    }
    return v;
}
double swiglu(double g, double u, double lim) {
    if (lim > 0) {
        g = std::min(g, lim);
        u = std::min(std::max(u, -lim), lim);
    }
    return g / (1.0 + std::exp(-g)) * u;
}

void expect(const char* what, const float* got, const double* want, size_t n, double rtol) {
    // every value finite first, on both sides: a NaN would drop out of the max below (and the next finite error take
    // its place), an Inf in want would make the scale infinite
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            ++g_fail;
            std::printf("%-44s FAIL  not finite (at %zu: %.6g vs %.6g)\n", what, i, (double) got[i], want[i]);
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
    std::printf("%-44s %s  max rel err %.2e (at %zu: %.6g vs %.6g)\n", what, ok ? "ok  " : "FAIL", worst, at,
                (double) got[at], want[at]);
}

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

void test_mv(cudaStream_t s) {
    for (int fmt : {gg::kFmtBF16, gg::kFmtMXFP4, gg::kFmtQ8_0})
        for (int nc : {1, 3, 16})
            for (int off : {0, 1}) {   // off 1: activations not 16-byte aligned (the scalar kernels)
                const int k = 1280, n = 520;
                std::vector<uint16_t> wb;
                std::vector<uint8_t> wm;
                if (fmt == gg::kFmtBF16) {
                    wb.resize((size_t) n * k);
                    for (auto& v : wb) v = f2bf(frand(0.05f));
                } else {
                    wm = fmt == gg::kFmtMXFP4 ? mx_rand(n, k) : q8_rand(n, k);
                }
                const size_t xcs = (size_t) k + 4 + off;
                std::vector<float> x(xcs * nc + 8), bias(n);
                for (auto& v : x) v = frand();
                for (auto& v : bias) v = frand(0.1f);
                float* dx = dev(x);
                float* db = dev(bias);
                std::vector<float> y0((size_t) n * nc, 0.0f);
                float* dy = dev(y0);
                void* dw = fmt == gg::kFmtBF16 ? (void*) dev(wb) : (void*) dev(wm);
                x3::Mat m;
                m.fmt = fmt;
                m.raw = dw;
                m.k = k;
                m.n = n;
                x3::MvJob j;
                j.m = &m;
                j.xf = dx + off;
                j.y = dy;
                j.bias = db;
                j.alpha = 0.5f;
                j.ncol = nc;
                j.xf_cs = xcs;
                j.y_cs = n;
                if (!gg::mv(&j, 1, s)) ++g_fail;
                CU(cudaStreamSynchronize(s));
                std::vector<float> y = host(dy, (size_t) n * nc);
                std::vector<double> want((size_t) n * nc);
                for (int c = 0; c < nc; ++c)
                    for (int r = 0; r < n; ++r) {
                        const float* xc = x.data() + off + c * xcs;
                        double d = 0;
                        if (fmt == gg::kFmtBF16)
                            for (int i = 0; i < k; ++i) d += (double) bf16f(wb[(size_t) r * k + i]) * xc[i];
                        else if (fmt == gg::kFmtMXFP4)
                            d = mx_dot(wm.data(), k, r, xc);
                        else {
                            const std::vector<float> xr = q8_round(xc, k);
                            for (int i = 0; i < k; ++i) d += q8_val(wm.data() + (size_t) r * gg::q8_row_bytes(k), i) * xr[i];
                        }
                        want[(size_t) c * n + r] = 0.5 * d + bias[r];
                    }
                char what[96];
                std::snprintf(what, sizeof(what), "mv %s, %d activations%s", fmt == gg::kFmtBF16 ? "BF16" : fmt == gg::kFmtMXFP4 ? "MXFP4" : "Q8_0", nc,
                              off ? ", unaligned" : "");
                expect(what, y.data(), want.data(), want.size(), 2e-5);
                // the first activation alone gives the same bits as in the batch
                if (nc > 1) {
                    j.ncol = 1;
                    std::vector<float> y1(n, 0.0f);
                    float* dy1 = dev(y1);
                    j.y = dy1;
                    gg::mv(&j, 1, s);
                    CU(cudaStreamSynchronize(s));
                    y1 = host(dy1, n);
                    const bool same = std::memcmp(y1.data(), y.data(), (size_t) n * 4) == 0;
                    if (!same) ++g_fail;
                    std::printf("%-44s %s\n", "  ... one activation alone, bit for bit", same ? "ok" : "FAIL");
                    cudaFree(dy1);
                }
                cudaFree(dx), cudaFree(db), cudaFree(dy), cudaFree(dw);
            }
    // rows(): BF16 -> F16 of a view
    {
        const int k = 256, n = 384;
        std::vector<uint16_t> wb((size_t) n * k);
        for (auto& v : wb) v = f2bf(frand(2.0f));
        uint16_t* dw = dev(wb);
        x3::Mat m;
        m.fmt = gg::kFmtBF16;
        m.raw = dw;
        m.k = k;
        m.n = n;
        const x3::Mat v = gg::view(m, 128, 128);
        std::vector<uint16_t> z((size_t) 128 * k);
        uint16_t* dd = dev(z);
        gg::rows(v, dd, false, s);
        CU(cudaStreamSynchronize(s));
        z = host(dd, z.size());
        std::vector<float> got(z.size());
        std::vector<double> want(z.size());
        for (size_t i = 0; i < z.size(); ++i) {
            got[i] = __half2float(*(const __half*) &z[i]);
            want[i] = bf16f(wb[(size_t) 128 * k + i]);
        }
        expect("rows BF16 -> F16 (a view)", got.data(), want.data(), want.size(), 1e-3);
        cudaFree(dw), cudaFree(dd);
    }
    {   // rows(): Q8_0 -> F16 of a view
        const int k = 256, n = 384;
        std::vector<uint8_t> wq = q8_rand(n, k);
        uint8_t* dw = dev(wq);
        x3::Mat m;
        m.fmt = gg::kFmtQ8_0;
        m.raw = dw;
        m.k = k;
        m.n = n;
        const x3::Mat v = gg::view(m, 128, 128);
        std::vector<uint16_t> z((size_t) 128 * k);
        uint16_t* dd = dev(z);
        gg::rows(v, dd, false, s);
        CU(cudaStreamSynchronize(s));
        z = host(dd, z.size());
        std::vector<float> got(z.size());
        std::vector<double> want(z.size());
        for (size_t i = 0; i < z.size(); ++i) {
            got[i] = __half2float(*(const __half*) &z[i]);
            want[i] = q8_val(wq.data() + (128 + i / k) * gg::q8_row_bytes(k), (int) (i % k));
        }
        expect("rows Q8_0 -> F16 (a view)", got.data(), want.data(), want.size(), 1e-3);
        cudaFree(dw), cudaFree(dd);
    }
}

struct Experts {
    int D, F, n_exp;
    x3::ExpertLayout lay;
    size_t blob;
    std::vector<uint8_t> host;   // n_exp blobs
    uint8_t* dev = nullptr;
};
Experts make_experts(int D, int F, int n_exp) {
    Experts E{D, F, n_exp, {}, 0, {}, nullptr};
    size_t off = 0;
    for (int r = 0; r < 3; ++r) {
        E.lay.trellis[r] = off;
        off += (gg::mx_bytes(r < 2 ? F : D, r < 2 ? D : F) + 255) / 256 * 256;
    }
    E.lay.fmt = gg::kFmtMXFP4;
    E.blob = off;
    E.host.assign(E.blob * n_exp, 0);
    for (int e = 0; e < n_exp; ++e)
        for (int r = 0; r < 3; ++r) {
            const std::vector<uint8_t> m = mx_rand(r < 2 ? F : D, r < 2 ? D : F);
            std::memcpy(E.host.data() + e * E.blob + E.lay.trellis[r], m.data(), m.size());
        }
    E.dev = dev(E.host);
    return E;
}
const uint8_t* emat(const Experts& E, int e, int r) { return E.host.data() + e * E.blob + E.lay.trellis[r]; }
// reference of one token's expert e: h then the down output
// (x and h rounded to int8 per 32 first, as the dp4a kernels round them)
void ref_expert(const Experts& E, int e, const float* x, double lim, std::vector<float>& h, std::vector<double>& y) {
    const std::vector<float> xr = q8_round(x, E.D);
    h.resize(E.F);
    for (int f = 0; f < E.F; ++f)
        h[f] = (float) swiglu(mx_dot(emat(E, e, 0), E.D, f, xr.data()), mx_dot(emat(E, e, 1), E.D, f, xr.data()), lim);
    const std::vector<float> hr = q8_round(h.data(), E.F);
    y.resize(E.D);
    for (int d = 0; d < E.D; ++d) y[d] = mx_dot(emat(E, e, 2), E.F, d, hr.data());
}

void test_experts(int D, int F, cudaStream_t s) {
    const int n_exp = 12, k = 6, T = 5;
    const float lim = 10.0f;
    Experts E = make_experts(D, F, n_exp);
    // T tokens' plans [T][8]: picks, weights, some picks off the device (ptr 0), token 1 with a CPU part
    std::vector<int> pick((size_t) T * 8, 0);
    std::vector<float> pw((size_t) T * 8, 0.0f);
    std::vector<unsigned long long> pp((size_t) T * 8, 0ull);
    for (int t = 0; t < T; ++t) {
        std::vector<int> all(n_exp);
        for (int i = 0; i < n_exp; ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        for (int i = 0; i < k; ++i) {
            pick[t * 8 + i] = all[i];
            pw[t * 8 + i] = 0.2f + 0.1f * frand();
            const bool on = !(t == 2 && i == 3) && !(t == 4 && i == 0);
            pp[t * 8 + i] = on ? (unsigned long long) (E.dev + (size_t) all[i] * E.blob) : 0ull;
        }
    }
    std::vector<float> x((size_t) T * D), sh((size_t) T * D), cpu((size_t) T * D);
    for (auto& v : x) v = frand();
    for (auto& v : sh) v = frand(0.1f);
    for (auto& v : cpu) v = frand(0.1f);
    std::vector<int> cflag(T, 0);
    cflag[1] = 1;
    float *dx = dev(x), *dsh = dev(sh), *dcpu = dev(cpu), *dpw = dev(pw);
    unsigned long long* dpp = dev(pp);
    int* dflag = dev(cflag);
    // reference outputs
    std::vector<double> want((size_t) T * D, 0.0);
    std::vector<float> hh;
    std::vector<double> yy;
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) want[(size_t) t * D + d] = 0;
        for (int i = 0; i < k; ++i) {
            if (pp[t * 8 + i] == 0ull) continue;
            ref_expert(E, pick[t * 8 + i], x.data() + (size_t) t * D, lim, hh, yy);
            for (int d = 0; d < D; ++d) want[(size_t) t * D + d] += pw[t * 8 + i] * yy[d];
        }
        for (int d = 0; d < D; ++d)
            want[(size_t) t * D + d] += (cflag[t] ? cpu[(size_t) t * D + d] : 0.0f) + sh[(size_t) t * D + d];
    }
    // the token path, token by token
    std::vector<float> tok((size_t) T * D);
    float* dh = dev(std::vector<float>((size_t) 8 * F));
    float* dout = dev(std::vector<float>(D));
    for (int t = 0; t < T; ++t) {
        x3::MoeArgs a;
        a.plan_ptr = dpp + t * 8;
        a.plan_w = dpw + t * 8;
        a.cpu_part = dcpu + (size_t) t * D;
        a.cpu_flag = dflag + t;
        a.lay = E.lay;
        a.k = k;
        a.n_embd = D;
        a.n_ff = F;
        a.limit = lim;
        gg::moe_gate_up(a, dx + (size_t) t * D, dh, s);
        gg::moe_down(a, dh, dsh + (size_t) t * D, dout, s);
        CU(cudaStreamSynchronize(s));
        CU(cudaMemcpy(tok.data() + (size_t) t * D, dout, (size_t) D * 4, cudaMemcpyDeviceToHost));
    }
    char what[96];
    std::snprintf(what, sizeof(what), "token experts (D %d, F %d)", D, F);
    expect(what, tok.data(), want.data(), want.size(), 1e-3);   // (an h value can round to its other int8)
    // the window: groups by expert (the device ones), entries e = t * 8 + i
    std::vector<unsigned long long> gptr;
    std::vector<int> gstart, gent;
    for (int x0 = 0; x0 < n_exp; ++x0) {
        std::vector<int> ents;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < k; ++i)
                if (pick[t * 8 + i] == x0 && pp[t * 8 + i] != 0ull) ents.push_back(t * 8 + i);
        if (ents.empty()) continue;
        gptr.push_back((unsigned long long) (E.dev + (size_t) x0 * E.blob));
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
    w.k = k;
    w.n_embd = D;
    w.n_ff = F;
    w.limit = lim;
    if (!gg::win_experts(w, s)) ++g_fail;
    CU(cudaStreamSynchronize(s));
    std::vector<float> win = host(dwo, (size_t) T * D);
    std::snprintf(what, sizeof(what), "window experts (T %d)", T);
    expect(what, win.data(), want.data(), want.size(), 1e-3);
    const bool same = std::memcmp(win.data(), tok.data(), win.size() * 4) == 0;
    if (!same) ++g_fail;
    std::printf("%-44s %s\n", "  ... bit for bit the token path's", same ? "ok" : "FAIL");

    // the prompt path's set: rows sorted by expert (bounds), each row a token index (row_idx) - experts of a few rows (the
    // light kernels), then of more (the tiles)
    for (int R : {100, 200, 400}) {
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
        const int NT = 40;
        std::vector<float> xs((size_t) NT * D);
        for (auto& v : xs) v = frand();
        for (int r = 0; r < R; ++r) idx[r] = (int) (rng() % NT);
        float* dxs = dev(xs);
        int *dbd = dev(bounds), *didx = dev(idx);
        uint16_t* dh16 = dev(std::vector<uint16_t>((size_t) R * F));
        float* dy = dev(std::vector<float>((size_t) R * D));
        x3::SetArgs sa;
        sa.wbase = E.dev;
        sa.stride = E.blob;
        sa.lay = E.lay;
        sa.n_exp = n_exp;
        sa.bounds = dbd;
        sa.row0 = 0;
        sa.max_rows = max_rows;
        sa.n_embd = D;
        sa.n_ff = F;
        sa.limit = lim;
        gg::set_gate_up(sa, dxs, D, didx, dh16, F, s);
        gg::set_down(sa, dh16, F, dy, D, s);
        CU(cudaStreamSynchronize(s));
        std::vector<uint16_t> h16 = host(dh16, (size_t) R * F);
        std::vector<float> y = host(dy, (size_t) R * D);
        {   // gate / up given the rows in FP16 too (the prompt path's x16: rows_f16 of the same rows) - bit for bit
            std::vector<uint16_t> x16(xs.size());
            for (size_t i = 0; i < xs.size(); ++i) {
                const __half h = __float2half_rn(xs[i]);
                std::memcpy(&x16[i], &h, 2);
            }
            uint16_t* dx16 = dev(x16);
            uint16_t* dh2 = dev(std::vector<uint16_t>((size_t) R * F));
            gg::set_gate_up(sa, dxs, D, didx, dh2, F, s, dx16);
            CU(cudaStreamSynchronize(s));
            const bool same = host(dh2, (size_t) R * F) == h16;
            std::printf("  set gate/up, %d rows, F16 rows given       %s\n", R, same ? "ok    (bit for bit the F32 rows')" : "FAIL");
            if (!same) ++g_fail;
            cudaFree(dx16), cudaFree(dh2);
        }
        std::vector<float> hgot((size_t) R * F);
        std::vector<double> hwant((size_t) R * F), ywant((size_t) R * D);
        // (an expert of at most 16 rows takes the light kernels: its inputs rounded to int8 per 32, as the token path's)
        int n_light = 0;
        for (int e = 0; e < n_exp; ++e) n_light += bounds[e + 1] - bounds[e] <= 16 ? bounds[e + 1] - bounds[e] : 0;
        for (int r = 0; r < R; ++r) {
            const bool light = bounds[owner[r] + 1] - bounds[owner[r]] <= 16;
            const std::vector<float> xq = light ? q8_round(xs.data() + (size_t) idx[r] * D, D) : std::vector<float>();
            const float* xr = light ? xq.data() : xs.data() + (size_t) idx[r] * D;
            std::vector<float> hr(F);
            for (int f = 0; f < F; ++f) {
                hwant[(size_t) r * F + f] =
                    swiglu(mx_dot(emat(E, owner[r], 0), D, f, xr), mx_dot(emat(E, owner[r], 1), D, f, xr), lim);
                hgot[(size_t) r * F + f] = __half2float(*(const __half*) &h16[(size_t) r * F + f]);
                hr[f] = hgot[(size_t) r * F + f];   // the down from the F16 h the kernel kept
            }
            if (light) hr = q8_round(hr.data(), F);
            for (int d = 0; d < D; ++d) ywant[(size_t) r * D + d] = mx_dot(emat(E, owner[r], 2), F, d, hr.data());
        }
        char sw[64];
        std::snprintf(sw, sizeof(sw), "set gate/up (F16 h), %d rows, %d light", R, n_light);
        expect(sw, hgot.data(), hwant.data(), hwant.size(), 2e-3);
        std::snprintf(sw, sizeof(sw), "set down, %d rows, %d light", R, n_light);
        expect(sw, y.data(), ywant.data(), ywant.size(), n_light > 0 ? 1e-4 : 5e-5);
        cudaFree(dxs), cudaFree(dbd), cudaFree(didx), cudaFree(dh16), cudaFree(dy);
    }
    // the CPU lane's rows
    {
        const int n = 3;
        const float* xp[n] = {x.data(), x.data() + D, x.data() + 2 * D};
        std::vector<float> y((size_t) n * 100);
        float* yp[n] = {y.data(), y.data() + 100, y.data() + 200};
        gg::mx_cpu_rows(emat(E, 5, 0), gg::mx_row_bytes(D), D, 7, 107, xp, n, yp);
        std::vector<double> want2((size_t) n * 100);
        for (int i = 0; i < n; ++i)
            for (int r = 7; r < 107; ++r) want2[(size_t) i * 100 + r - 7] = mx_dot(emat(E, 5, 0), D, r, xp[i]);
        expect("CPU lane rows", y.data(), want2.data(), want2.size(), 5e-5);
    }
    // the CPU lane's int8 rows (gg::mx_rows), every tier this CPU runs: the gate and the down shapes, 1..9 tokens (past
    // a call's 8), against the double reference on the q8-rounded activations; the tiers against the scalar one; the
    // activation's bytes (vector and scalar quant, whole and by bands) against q8_round's values
    {
        const int NT = 9, R0 = 5, R1 = 69;
        for (int down = 0; down < 2; ++down) {
            const int k = down ? F : D;
            const uint8_t* w = emat(E, 7, down ? 2 : 0);
            const size_t rb = gg::mx_row_bytes(k), ab = gg::mx_act_bytes(k);
            std::vector<float> xs((size_t) NT * k);
            for (auto& v : xs) v = down ? frand(0.05f) : frand();
            for (int j = 0; j < 32; ++j) xs[(size_t) 2 * k + 32 + j] = 0.0f;   // (an all-zero block: scale 0)
            std::vector<uint8_t> act((size_t) NT * ab), act_s(ab), act_b(ab);
            const void* ap[NT];
            bool q_ok = true;
            for (int i = 0; i < NT; ++i) {
                uint8_t* a = act.data() + (size_t) i * ab;
                gg::mx_quant(xs.data() + (size_t) i * k, k, a);
                gg::mx_quant_blocks_scalar(xs.data() + (size_t) i * k, 0, k / 32, act_s.data(), k);
                for (int b0 = 0; b0 < k / 32; b0 += 2)   // (by bands of 2 blocks, as the lane's jobs write h)
                    gg::mx_quant_blocks(xs.data() + (size_t) i * k + 32 * b0, b0, b0 + 2, act_b.data(), k);
                q_ok = q_ok && std::memcmp(a, act_s.data(), ab) == 0 && std::memcmp(a, act_b.data(), ab) == 0;
                const std::vector<float> xr = q8_round(xs.data() + (size_t) i * k, k);
                for (int b = 0; b < k / 32; ++b)
                    for (int j = 0; j < 32; ++j) {
                        const int q = gg::mx_act_value(a, b, j);
                        const float sc = gg::mx_act_scale(a, k, b);
                        q_ok = q_ok && (float) ((double) q * sc) == xr[(size_t) b * 32 + j];
                    }
                ap[i] = a;
            }
            if (!q_ok) ++g_fail;
            std::printf("%-44s %s\n", down ? "CPU int8 quant (down's k)" : "CPU int8 quant (gate's k)",
                        q_ok ? "ok    vector = scalar = bands = q8_round" : "FAIL");
            std::vector<double> want3((size_t) NT * (R1 - R0));
            for (int i = 0; i < NT; ++i) {
                const std::vector<float> xr = q8_round(xs.data() + (size_t) i * k, k);
                for (int r = R0; r < R1; ++r) want3[(size_t) i * (R1 - R0) + r - R0] = mx_dot(w, k, r, xr.data());
            }
            std::vector<float> ref((size_t) NT * (R1 - R0));
            for (int t = 0; t < gg::kMxTiers; ++t) {
                if (!gg::mx_tier_ok(t)) continue;
                for (int n : {1, 2, 3, 8, 9}) {
                    std::vector<float> y((size_t) n * (R1 - R0));
                    std::vector<float*> yp((size_t) n);
                    for (int i = 0; i < n; ++i) yp[(size_t) i] = y.data() + (size_t) i * (R1 - R0);
                    gg::mx_rows_tier(t, w, rb, k, R0, R1, ap, n, yp.data(), t == gg::kMxScalar ? 0 : 256);
                    char what[96];
                    std::snprintf(what, sizeof what, "CPU int8 rows %s %s n=%d", down ? "down" : "gate",
                                  gg::mx_tier_name(t), n);
                    expect(what, y.data(), want3.data(), y.size(), 5e-5);
                    if (t == gg::kMxScalar && n == NT) ref = y;
                    if (t != gg::kMxScalar && !ref.empty()) {   // (every tier the same bits as the scalar one)
                        const bool same = std::memcmp(y.data(), ref.data(), y.size() * sizeof(float)) == 0;
                        if (!same) ++g_fail;
                        std::printf("  %-42s %s\n", what, same ? "ok    bit-identical to the scalar tier" : "FAIL  differs from the scalar tier");
                    }
                }
            }
        }
        std::printf("CPU int8 rows: this CPU's tier %s\n", gg::mx_tier_name(gg::mx_tier()));
    }
    cudaFree(E.dev), cudaFree(dx), cudaFree(dsh), cudaFree(dcpu), cudaFree(dpw), cudaFree(dpp), cudaFree(dflag);
    cudaFree(dh), cudaFree(dout), cudaFree(dgp), cudaFree(dgs), cudaFree(dge), cudaFree(dng), cudaFree(dwh);
    cudaFree(dpart), cudaFree(dwo);
}

// ---------------------------------------------------------------- the low-bit types (ds4_iq)
// random blocks of a type: random bytes, the block scales small normal values (every grid index, sign and code)
std::vector<uint8_t> q_rand(int type, int n, int k) {
    const size_t rb = gg::type_row_bytes(type, k), bb = (size_t) gg::iq::block_bytes(type);
    std::vector<uint8_t> w(rb * n);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    auto put = [&](size_t at, float f) {
        const __half h = __float2half(f);
        std::memcpy(&w[at], &h, 2);
    };
    for (size_t o = 0; o < w.size(); o += bb) {
        const float d = 0.001f + 0.002f * std::fabs(frand());
        switch (type) {
            case gg::iq::kQ4_K:
            case gg::iq::kQ5_K: put(o, d), put(o + 2, 0.5f * d); break;
            case gg::iq::kQ2_K: put(o + 80, d), put(o + 82, 0.5f * d); break;
            case gg::iq::kQ3_K: put(o + 108, d); break;
            case gg::iq::kIQ1_M: {   // (d: an F16 in the top nibbles of the 4 scale words, bytes 48..55)
                const __half h = __float2half(d);
                uint16_t u;
                std::memcpy(&u, &h, 2);
                for (int i = 0; i < 4; ++i) {
                    uint16_t sw;
                    std::memcpy(&sw, &w[o + 48 + 2 * i], 2);
                    sw = (uint16_t) ((sw & 0x0FFFu) | (((u >> (4 * i)) & 0xFu) << 12));
                    std::memcpy(&w[o + 48 + 2 * i], &sw, 2);
                }
                break;
            }
            default: put(o, d); break;   // (the i-quants: d first)
        }
    }
    return w;
}
// a row's dot with x in double, the weights ggml's dequantize_row_* values
double q_dot(int type, const uint8_t* row, int k, const float* x) {
    static thread_local std::vector<float> w;
    w.resize(k);
    ggml_get_type_traits((ggml_type) type)->to_float(row, w.data(), k);
    double s = 0;
    for (int i = 0; i < k; ++i) s += (double) w[i] * x[i];
    return s;
}
const char* tn(int t) { return gg::type_name(t); }
const int kQTypes[] = {gg::iq::kIQ2_XXS, gg::iq::kIQ2_S, gg::iq::kIQ3_XXS, gg::iq::kIQ3_S, gg::iq::kIQ4_XS,
                       gg::iq::kQ2_K,    gg::iq::kQ3_K, gg::iq::kQ4_K,    gg::iq::kIQ1_S, gg::iq::kIQ1_M,
                       gg::iq::kIQ2_XS,  gg::iq::kQ5_K};   // (the last two: dense weights / other GGUFs' experts)

// n rows of type t (k values each) through the decode kernels and the prompt path's f16_rows against ggml's own
// dequantize_row_* values: the kernels' F32 values equal, f16_rows' halves equal to ggml's rounded to F16
bool check_rows(int t, const std::vector<uint8_t>& w, int k, int n, const char* label, cudaStream_t s) {
    std::vector<float> want((size_t) n * k);
    ggml_get_type_traits((ggml_type) t)->to_float(w.data(), want.data(), (int64_t) n * k);
    uint8_t* dw = dev(w);
    float* dsub = dev(std::vector<float>((size_t) n * k, -1.0f));
    float* dtile = dev(std::vector<float>((size_t) n * k, -1.0f));
    uint16_t* dh = nullptr;
    CU(cudaMalloc(&dh, (size_t) n * k * 2));
    const bool ran = gg::iq::decode_values(t, dw, k, n, dsub, dtile, s);
    const bool ran16 = gg::iq::f16_rows(t, dw, k, n, dh, k, s);
    CU(cudaStreamSynchronize(s));
    const std::vector<float> gs = host(dsub, (size_t) n * k), gt = host(dtile, (size_t) n * k);
    std::vector<uint16_t> gh((size_t) n * k);
    CU(cudaMemcpy(gh.data(), dh, gh.size() * 2, cudaMemcpyDeviceToHost));
    size_t bs = 0, bt = 0, bh = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        bs += !(gs[i] == want[i]);
        bt += !(gt[i] == want[i]);
        const __half wh = __float2half_rn(want[i]);
        uint16_t u;
        std::memcpy(&u, &wh, 2);
        bh += gh[i] != u && !((gh[i] | u) == 0x8000u);   // (a zero's sign aside)
    }
    const bool ok = ran && ran16 && bs == 0 && bt == 0 && bh == 0;
    if (!ok) ++g_fail;
    std::printf("%-52s %s  %d values: kernels %zu, tiles %zu, f16_rows %zu differ%s\n", label, ok ? "ok  " : "FAIL",
                n * k, bs, bt, bh, ran && ran16 ? "" : " (not run)");
    cudaFree(dw), cudaFree(dsub), cudaFree(dtile), cudaFree(dh);
    return ok;
}

// the prompt path's rows (f16_rows) for every type on random blocks
void test_f16_rows(cudaStream_t s) {
    for (int t : kQTypes) {
        char what[96];
        std::snprintf(what, sizeof what, "rows %s = ggml's (kernels, tiles, f16_rows)", tn(t));
        check_rows(t, q_rand(t, 24, 2048), 2048, 24, what, s);
    }
}

// every low-bit type's decode against ggml's own dequantize_row_* on random blocks: the token / window / light
// kernels' codes times their scales, and the prompt tiles' loader - each value equal to ggml's (a zero's sign aside)
void test_decode_values(cudaStream_t s) {
    for (int t : kQTypes) {
        const int k = 2048, n = 40;
        const std::vector<uint8_t> w = q_rand(t, n, k);
        std::vector<float> want((size_t) n * k);
        ggml_get_type_traits((ggml_type) t)->to_float(w.data(), want.data(), (int64_t) n * k);
        uint8_t* dw = dev(w);
        float* dsub = dev(std::vector<float>((size_t) n * k, -1.0f));
        float* dtile = dev(std::vector<float>((size_t) n * k, -1.0f));
        const bool ran = gg::iq::decode_values(t, dw, k, n, dsub, dtile, s);
        CU(cudaStreamSynchronize(s));
        const std::vector<float> gs = host(dsub, (size_t) n * k), gt = host(dtile, (size_t) n * k);
        size_t bs = 0, bt = 0, at = want.size();
        for (size_t i = 0; i < want.size(); ++i) {
            const bool a = !(gs[i] == want[i]), b = !(gt[i] == want[i]);
            bs += a;
            bt += b;
            if ((a || b) && at == want.size()) at = i;
        }
        const bool ok = ran && bs == 0 && bt == 0;
        if (!ok) ++g_fail;
        char what[96];
        std::snprintf(what, sizeof what, "decode %s = ggml's dequantize_row", tn(t));
        if (ok) std::printf("%-44s ok    %d values, kernels and tiles\n", what, n * k);
        else
            std::printf("%-44s FAIL  %s%zu kernel / %zu tile values differ (first at %zu: %.9g / %.9g vs %.9g)\n", what,
                        ran ? "" : "not run; ", bs, bt, at, at < want.size() ? gs[at] : 0.0f,
                        at < want.size() ? gt[at] : 0.0f, at < want.size() ? want[at] : 0.0f);
        cudaFree(dw), cudaFree(dsub), cudaFree(dtile);
    }
}

void test_mv_q(int type, cudaStream_t s) {
    for (int nc : {1, 3, 16}) {
        const int k = 768, n = 136;
        std::vector<uint8_t> wq = q_rand(type, n, k);
        const size_t xcs = (size_t) k + 8;
        std::vector<float> x(xcs * nc), bias(n);
        for (auto& v : x) v = frand();
        for (auto& v : bias) v = frand(0.1f);
        float *dx = dev(x), *db = dev(bias), *dy = dev(std::vector<float>((size_t) n * nc));
        uint8_t* dw = dev(wq);
        x3::Mat m;
        m.fmt = gg::kFmtGG;
        m.gg_type = type;
        m.raw = dw;
        m.k = k;
        m.n = n;
        x3::MvJob j;
        j.m = &m;
        j.xf = dx;
        j.y = dy;
        j.bias = db;
        j.alpha = 0.5f;
        j.ncol = nc;
        j.xf_cs = xcs;
        j.y_cs = n;
        if (!gg::mv(&j, 1, s)) ++g_fail;
        CU(cudaStreamSynchronize(s));
        std::vector<float> y = host(dy, (size_t) n * nc);
        std::vector<double> want((size_t) n * nc);
        for (int c = 0; c < nc; ++c) {
            const std::vector<float> xr = q8_round(x.data() + c * xcs, k);
            for (int r = 0; r < n; ++r)
                want[(size_t) c * n + r] = 0.5 * q_dot(type, wq.data() + r * gg::type_row_bytes(type, k), k, xr.data()) + bias[r];
        }
        char what[96];
        std::snprintf(what, sizeof(what), "mv %s, %d activations", tn(type), nc);
        expect(what, y.data(), want.data(), want.size(), 2e-5);
        cudaFree(dx), cudaFree(db), cudaFree(dy), cudaFree(dw);
    }
}

struct QExperts {
    int D, F, n_exp, tg, td;
    x3::ExpertLayout lay;
    size_t blob;
    std::vector<uint8_t> host;
    uint8_t* dev = nullptr;
    const uint8_t* mat(int e, int r) const { return host.data() + e * blob + lay.trellis[r]; }
    size_t rb(int r) const { return gg::type_row_bytes(r < 2 ? tg : td, r < 2 ? D : F); }
    double dot(int e, int r, int row, const float* x) const {
        return q_dot(r < 2 ? tg : td, mat(e, r) + row * rb(r), r < 2 ? D : F, x);
    }
};
QExperts make_q(int D, int F, int n_exp, int tg, int td) {
    QExperts E{D, F, n_exp, tg, td, {}, 0, {}, nullptr};
    size_t off = 0;
    for (int r = 0; r < 3; ++r) {
        E.lay.trellis[r] = off;
        E.lay.gg_type[r] = r < 2 ? tg : td;
        off += ((r < 2 ? F : D) * gg::type_row_bytes(E.lay.gg_type[r], r < 2 ? D : F) + 255) / 256 * 256;
    }
    E.lay.fmt = gg::kFmtMXFP4;
    E.blob = off;
    E.host.assign(E.blob * n_exp, 0);
    for (int e = 0; e < n_exp; ++e)
        for (int r = 0; r < 3; ++r) {
            const std::vector<uint8_t> m = q_rand(E.lay.gg_type[r], r < 2 ? F : D, r < 2 ? D : F);
            std::memcpy(E.host.data() + e * E.blob + E.lay.trellis[r], m.data(), m.size());
        }
    E.dev = dev(E.host);
    return E;
}

// test_experts' checks for low-bit experts: the token path, the window (bit for bit the token path's), the prompt sets
// and the CPU lane (ggml-cpu's arithmetic: Q8_K activations, so against a reference on those)
void test_experts_q(int D, int F, int tg, int td, cudaStream_t s) {
    const int n_exp = 10, k = 6, T = 4;
    const float lim = 10.0f;
    QExperts E = make_q(D, F, n_exp, tg, td);
    std::vector<int> pick((size_t) T * 8, 0);
    std::vector<float> pw((size_t) T * 8, 0.0f);
    std::vector<unsigned long long> pp((size_t) T * 8, 0ull);
    for (int t = 0; t < T; ++t) {
        std::vector<int> all(n_exp);
        for (int i = 0; i < n_exp; ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        for (int i = 0; i < k; ++i) {
            pick[t * 8 + i] = all[i];
            pw[t * 8 + i] = 0.2f + 0.1f * frand();
            const bool on = !(t == 2 && i == 3);
            pp[t * 8 + i] = on ? (unsigned long long) (E.dev + (size_t) all[i] * E.blob) : 0ull;
        }
    }
    std::vector<float> x((size_t) T * D), sh((size_t) T * D), cpu((size_t) T * D);
    for (auto& v : x) v = frand();
    for (auto& v : sh) v = frand(0.1f);
    for (auto& v : cpu) v = frand(0.1f);
    std::vector<int> cflag(T, 0);
    cflag[1] = 1;
    float *dx = dev(x), *dsh = dev(sh), *dcpu = dev(cpu), *dpw = dev(pw);
    unsigned long long* dpp = dev(pp);
    int* dflag = dev(cflag);
    std::vector<float> tok((size_t) T * D), gh((size_t) T * 8 * F);
    float* dh = dev(std::vector<float>((size_t) 8 * F));
    float* dout = dev(std::vector<float>(D));
    for (int t = 0; t < T; ++t) {
        x3::MoeArgs a;
        a.plan_ptr = dpp + t * 8;
        a.plan_w = dpw + t * 8;
        a.cpu_part = dcpu + (size_t) t * D;
        a.cpu_flag = dflag + t;
        a.lay = E.lay;
        a.k = k;
        a.n_embd = D;
        a.n_ff = F;
        a.limit = lim;
        gg::moe_gate_up(a, dx + (size_t) t * D, dh, s);
        gg::moe_down(a, dh, dsh + (size_t) t * D, dout, s);
        CU(cudaStreamSynchronize(s));
        CU(cudaMemcpy(tok.data() + (size_t) t * D, dout, (size_t) D * 4, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(gh.data() + (size_t) t * 8 * F, dh, (size_t) 8 * F * 4, cudaMemcpyDeviceToHost));
    }
    // the reference: gate / up in double on the int8 x, then the down on the device's own h rounded as the kernels
    // round it (an h within F32 rounding of the double one can still round to the next int8 - a test artefact, not the
    // kernels': the GPU's swiglu is F32), the h checked against the double one apart
    std::vector<double> want((size_t) T * D, 0.0), hwant;
    std::vector<float> hgot;
    for (int t = 0; t < T; ++t) {
        const std::vector<float> xr = act_round(x.data() + (size_t) t * D, D);
        for (int i = 0; i < k; ++i) {
            if (pp[t * 8 + i] == 0ull) continue;
            const int e = pick[t * 8 + i];
            const float* h = gh.data() + ((size_t) t * 8 + i) * F;
            for (int f = 0; f < F; ++f) {
                hwant.push_back(swiglu(E.dot(e, 0, f, xr.data()), E.dot(e, 1, f, xr.data()), lim));
                hgot.push_back(h[f]);
            }
            const std::vector<float> hr = act_round(h, F);
            for (int d = 0; d < D; ++d) want[(size_t) t * D + d] += pw[t * 8 + i] * E.dot(e, 2, d, hr.data());
        }
        for (int d = 0; d < D; ++d)
            want[(size_t) t * D + d] += (cflag[t] ? cpu[(size_t) t * D + d] : 0.0f) + sh[(size_t) t * D + d];
    }
    char what[96];
    std::snprintf(what, sizeof(what), "%s/%s token experts (D %d, F %d)", tn(tg), tn(td), D, F);
    expect(what, tok.data(), want.data(), want.size(), 1e-3);
    // (1e-3 as the output's: on HIP / Windows one token's host-emulated Q8_K x took an int8 apart from the device's -
    // 2e-4 at full size; the device's Q8_K is byte for byte the CPU lane's, test_q8k)
    expect("  ... their h (gate / up, swiglu)", hgot.data(), hwant.data(), hwant.size(), 1e-3);
    // the window
    std::vector<unsigned long long> gptr;
    std::vector<int> gstart, gent;
    for (int x0 = 0; x0 < n_exp; ++x0) {
        std::vector<int> ents;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < k; ++i)
                if (pick[t * 8 + i] == x0 && pp[t * 8 + i] != 0ull) ents.push_back(t * 8 + i);
        if (ents.empty()) continue;
        gptr.push_back((unsigned long long) (E.dev + (size_t) x0 * E.blob));
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
    w.k = k;
    w.n_embd = D;
    w.n_ff = F;
    w.limit = lim;
    if (!gg::win_experts(w, s)) ++g_fail;
    CU(cudaStreamSynchronize(s));
    std::vector<float> win = host(dwo, (size_t) T * D);
    std::snprintf(what, sizeof(what), "  window experts (T %d)", T);
    expect(what, win.data(), want.data(), want.size(), 1e-3);
    const bool same = std::memcmp(win.data(), tok.data(), win.size() * 4) == 0;
    if (!same) ++g_fail;
    std::printf("%-44s %s\n", "  ... bit for bit the token path's", same ? "ok" : "FAIL");
    for (int R : {90, 200, 400}) {   // the prompt path's set: experts of a few rows (the light kernels), then of more
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
        for (int r = 0; r < R; ++r) idx[r] = (int) (rng() % NT);
        float* dxs = dev(xs);
        int *dbd = dev(bounds), *didx = dev(idx);
        uint16_t* dh16 = dev(std::vector<uint16_t>((size_t) R * F));
        float* dy = dev(std::vector<float>((size_t) R * D));
        x3::SetArgs sa;
        sa.wbase = E.dev;
        sa.stride = E.blob;
        sa.lay = E.lay;
        sa.n_exp = n_exp;
        sa.bounds = dbd;
        sa.row0 = 0;
        sa.max_rows = max_rows;
        sa.n_embd = D;
        sa.n_ff = F;
        sa.limit = lim;
        gg::set_gate_up(sa, dxs, D, didx, dh16, F, s);
        gg::set_down(sa, dh16, F, dy, D, s);
        CU(cudaStreamSynchronize(s));
        std::vector<uint16_t> h16 = host(dh16, (size_t) R * F);
        std::vector<float> y = host(dy, (size_t) R * D);
        {   // gate / up given the rows in FP16 too (the prompt path's x16: rows_f16 of the same rows) - bit for bit
            std::vector<uint16_t> x16(xs.size());
            for (size_t i = 0; i < xs.size(); ++i) {
                const __half h = __float2half_rn(xs[i]);
                std::memcpy(&x16[i], &h, 2);
            }
            uint16_t* dx16 = dev(x16);
            uint16_t* dh2 = dev(std::vector<uint16_t>((size_t) R * F));
            gg::set_gate_up(sa, dxs, D, didx, dh2, F, s, dx16);
            CU(cudaStreamSynchronize(s));
            const bool same = host(dh2, (size_t) R * F) == h16;
            std::printf("  set gate/up, %d rows, F16 rows given       %s\n", R, same ? "ok    (bit for bit the F32 rows')" : "FAIL");
            if (!same) ++g_fail;
            cudaFree(dx16), cudaFree(dh2);
        }
        std::vector<float> hgot((size_t) R * F);
        std::vector<double> hwant((size_t) R * F), ywant((size_t) R * D);
        // (the light kernels, experts of <= 16 rows: the inputs rounded to int8 per 32 as the token path rounds them; the
        // GEMM tiles: F16 inputs and weights - a low-bit weight is not always an F16 value)
        int n_light = 0;
        for (int e = 0; e < n_exp; ++e) n_light += bounds[e + 1] - bounds[e] <= 16 ? bounds[e + 1] - bounds[e] : 0;
        for (int r = 0; r < R; ++r) {
            const bool light = bounds[owner[r] + 1] - bounds[owner[r]] <= 16;
            const std::vector<float> xq = light ? q8_round(xs.data() + (size_t) idx[r] * D, D) : std::vector<float>();
            const float* xr = light ? xq.data() : xs.data() + (size_t) idx[r] * D;
            std::vector<float> hr(F);
            for (int f = 0; f < F; ++f) {
                hwant[(size_t) r * F + f] = swiglu(E.dot(owner[r], 0, f, xr), E.dot(owner[r], 1, f, xr), lim);
                hgot[(size_t) r * F + f] = __half2float(*(const __half*) &h16[(size_t) r * F + f]);
                hr[f] = hgot[(size_t) r * F + f];
            }
            if (light) hr = q8_round(hr.data(), F);
            for (int d = 0; d < D; ++d) ywant[(size_t) r * D + d] = E.dot(owner[r], 2, d, hr.data());
        }
        char sw[64];
        std::snprintf(sw, sizeof(sw), "  set gate/up (F16 h), %d rows, %d light", R, n_light);
        expect(sw, hgot.data(), hwant.data(), hwant.size(), n_light == R ? 2e-3 : 8e-3);   // (IQ4_XS: the widest weights in F16)
        std::snprintf(sw, sizeof(sw), "  set down, %d rows, %d light", R, n_light);
        expect(sw, y.data(), ywant.data(), ywant.size(), n_light == R ? 1e-4 : 2e-3);
        cudaFree(dxs), cudaFree(dbd), cudaFree(didx), cudaFree(dh16), cudaFree(dy);
    }
    {   // the CPU lane: 3 tokens of expert 4, gate / up rows [7, 107) then all its down rows from the CPU's own h
        gg::IqCpu fm;
        std::string err;
        if (!gg::iq_cpu_fmt(tg, td, D, F, fm, err)) {
            std::printf("%-44s FAIL  %s\n", "  CPU lane", err.c_str());
            ++g_fail;
        } else {
            const int n = 3, e = 4, r0 = 7, r1 = std::min(107, F);
            std::vector<std::vector<uint8_t>> aq(n, std::vector<uint8_t>(fm.act_bytes)),
                hq(n, std::vector<uint8_t>(fm.h_bytes));
            std::vector<std::vector<float>> ff(n, std::vector<float>(F, 0.0f)), yo(n, std::vector<float>(D, 0.0f));
            const void* ap[n];
            const void* hp[n];
            float* fp[n];
            float* yp[n];
            for (int i = 0; i < n; ++i) {
                gg::iq_cpu_quant(fm.gu_act, x.data() + (size_t) i * D, D, aq[i].data());
                ap[i] = aq[i].data();
                fp[i] = ff[i].data();
            }
            gg::iq_cpu_gate_up(fm, E.mat(e, 0), E.mat(e, 1), ap, n, fp, r0, r1, lim);
            gg::iq_cpu_gate_up(fm, E.mat(e, 0), E.mat(e, 1), ap, n, fp, 0, r0, lim);
            gg::iq_cpu_gate_up(fm, E.mat(e, 0), E.mat(e, 1), ap, n, fp, r1, F, lim);
            for (int i = 0; i < n; ++i) {
                gg::iq_cpu_quant(fm.d_act, ff[i].data(), F, hq[i].data());
                hp[i] = hq[i].data();
                yp[i] = yo[i].data();
            }
            gg::iq_cpu_down(fm, E.mat(e, 2), hp, n, yp, 0, D);
            // the reference on the Q8_K values the CPU took (block_q8_K: F32 d, int8 qs[256], int16 bsums[16])
            auto q8k = [](const std::vector<uint8_t>& q, int kk) {
                std::vector<float> v(kk);
                for (int b = 0; b < kk / 256; ++b) {
                    const uint8_t* p = q.data() + (size_t) b * 292;
                    float d;
                    std::memcpy(&d, p, 4);
                    for (int j = 0; j < 256; ++j) v[b * 256 + j] = d * (float) (int8_t) p[4 + j];
                }
                return v;
            };
            const bool k8 = fm.gu_act == GGML_TYPE_Q8_K && fm.d_act == GGML_TYPE_Q8_K;
            std::vector<float> got_h, got_y;
            std::vector<double> want_h, want_y;
            for (int i = 0; i < n && k8; ++i) {
                const std::vector<float> xa = q8k(aq[i], D), ha = q8k(hq[i], F);
                for (int f = 0; f < F; ++f) {
                    got_h.push_back(ff[i][f]);
                    want_h.push_back(swiglu(E.dot(e, 0, f, xa.data()), E.dot(e, 1, f, xa.data()), lim));
                }
                for (int d = 0; d < D; ++d) {
                    got_y.push_back(yo[i][d]);
                    want_y.push_back(E.dot(e, 2, d, ha.data()));
                }
            }
            if (!k8) {
                std::printf("%-44s FAIL  activation types %d / %d\n", "  CPU lane", fm.gu_act, fm.d_act);
                ++g_fail;
            } else {
                expect("  CPU lane gate/up (Q8_K)", got_h.data(), want_h.data(), want_h.size(), 1e-4);
                expect("  CPU lane down (Q8_K)", got_y.data(), want_y.data(), want_y.size(), 1e-4);
            }
        }
    }
    cudaFree(E.dev), cudaFree(dx), cudaFree(dsh), cudaFree(dcpu), cudaFree(dpw), cudaFree(dpp), cudaFree(dflag);
    cudaFree(dh), cudaFree(dout), cudaFree(dgp), cudaFree(dgs), cudaFree(dge), cudaFree(dng), cudaFree(dwh);
    cudaFree(dpart), cudaFree(dwo);
}

// the role pairs the tests take: every type as gate / up and as down
const std::pair<int, int> kQPairs[] = {
    {gg::iq::kIQ2_XXS, gg::iq::kIQ3_XXS}, {gg::iq::kIQ2_S, gg::iq::kIQ3_S},  {gg::iq::kIQ3_XXS, gg::iq::kIQ3_S},
    {gg::iq::kIQ4_XS, gg::iq::kQ4_K},     {gg::iq::kQ2_K, gg::iq::kQ3_K},    {gg::iq::kQ4_K, gg::iq::kIQ4_XS},
    {gg::iq::kQ3_K, gg::iq::kQ2_K},       {gg::iq::kIQ3_S, gg::iq::kIQ2_XXS}, {gg::iq::kIQ3_S, gg::iq::kIQ2_S},
    {gg::iq::kIQ1_S, gg::iq::kIQ1_M},     {gg::iq::kIQ1_M, gg::iq::kIQ1_S},   {gg::iq::kIQ1_M, gg::iq::kIQ2_XXS}};
constexpr int kQPairIQ1 = 9;   // (the first IQ1 pair)

// a token's 6 low-bit experts at full size: GB/s of the weights read (only: the token experts of the first pair - the
// reference - and of the pairs with a type whose name holds it, e.g. "IQ1"; no prompt sets)
void bench_q(cudaStream_t s, const char* only = nullptr) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    const int D = 5120, F = 2304;
    for (const auto& pr : kQPairs) {
        if (only != nullptr && &pr != &kQPairs[0] && std::strstr(tn(pr.first), only) == nullptr &&
            std::strstr(tn(pr.second), only) == nullptr)
            continue;
        QExperts E = make_q(D, F, 6, pr.first, pr.second);
        // 8 copies of the 6 experts, the token's plan going round them: 48 experts' weights, past the GPU's last-level
        // caches (Strix Halo's 32 MB Infinity Cache would otherwise hold one token's IQ1 experts between the calls)
        constexpr int NS = 8;
        uint8_t* big = nullptr;
        CU(cudaMalloc(&big, (size_t) NS * 6 * E.blob));
        for (int c = 0; c < NS; ++c)
            CU(cudaMemcpy(big + (size_t) c * 6 * E.blob, E.dev, 6 * E.blob, cudaMemcpyDeviceToDevice));
        std::vector<unsigned long long> pp((size_t) 8 * NS, 0ull);
        std::vector<float> pw(8, 0.1f);
        for (int c = 0; c < NS; ++c)
            for (int i = 0; i < 6; ++i) pp[(size_t) c * 8 + i] = (unsigned long long) (big + ((size_t) c * 6 + i) * E.blob);
        unsigned long long* dpp = dev(pp);
        float* dpw = dev(pw);
        float* dx = dev(std::vector<float>(D, 0.01f));
        float* dh = dev(std::vector<float>((size_t) 8 * F));
        float* dout = dev(std::vector<float>(D));
        x3::MoeArgs ma;
        ma.plan_ptr = dpp;
        ma.plan_w = dpw;
        int turn = 0;
        auto next = [&] { ma.plan_ptr = dpp + (size_t) 8 * (turn++ % NS); };
        ma.lay = E.lay;
        ma.k = 6;
        ma.n_embd = D;
        ma.n_ff = F;
        ma.limit = 10.0f;
        const double gu = 6.0 * 2 * F * E.rb(0), dn = 6.0 * D * E.rb(2);
        auto timed = [&](const char* what, double bytes, auto&& fn) {
            fn();
            CU(cudaStreamSynchronize(s));
            cudaEventRecord(a, s);
            for (int i = 0; i < 50; ++i) fn();
            cudaEventRecord(b, s);
            CU(cudaEventSynchronize(b));
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("bench %-40s %8.1f us  %6.0f GB/s\n", what, ms * 1000 / 50, bytes / (ms / 50 * 1e-3) / 1e9);
        };
        char what[96];
        std::snprintf(what, sizeof(what), "6 %s/%s experts (gate/up + down)", tn(pr.first), tn(pr.second));
        timed(what, gu + dn, [&] {
            next();
            gg::moe_gate_up(ma, dx, dh, s);
            gg::moe_down(ma, dh, nullptr, dout, s);
        });
        timed("  gate/up alone", gu, [&] {
            next();
            gg::moe_gate_up(ma, dx, dh, s);
        });
        timed("  down + combine alone", dn, [&] {
            next();
            gg::moe_down(ma, dh, nullptr, dout, s);
        });
        cudaFree(E.dev), cudaFree(big), cudaFree(dpp), cudaFree(dpw), cudaFree(dx), cudaFree(dh), cudaFree(dout);
    }
    if (only != nullptr) return;
    // the prompt path's sets: 32 experts x R rows (128: an 8K chunk's share of 384 experts; 8: a short prompt's - the
    // weights' bandwidth), the MXFP4 ones first
    for (int R : {128, 8}) {
        const int NE = 32;
        for (int pi = -1; pi < (int) (sizeof(kQPairs) / sizeof(kQPairs[0])); ++pi) {
            x3::ExpertLayout lay;
            size_t blob;
            double wbytes;
            uint8_t* wdev;
            char what[96];
            if (pi < 0) {
                Experts M = make_experts(D, F, NE);
                lay = M.lay, blob = M.blob, wdev = M.dev;
                wbytes = 3.0 * gg::mx_bytes(F, D);
                std::snprintf(what, sizeof(what), "set MXFP4, %d experts x %d rows", NE, R);
            } else {
                QExperts E = make_q(D, F, NE, kQPairs[pi].first, kQPairs[pi].second);
                lay = E.lay, blob = E.blob, wdev = E.dev;
                wbytes = 2.0 * F * E.rb(0) + (double) D * E.rb(2);
                std::snprintf(what, sizeof(what), "set %s/%s, %d experts x %d rows", tn(kQPairs[pi].first),
                              tn(kQPairs[pi].second), NE, R);
            }
            std::vector<int> bounds(NE + 1), idx(NE * R);
            for (int e = 0; e <= NE; ++e) bounds[e] = e * R;
            for (int i = 0; i < NE * R; ++i) idx[i] = i;
            int *db = dev(bounds), *di = dev(idx);
            float* xs = dev(std::vector<float>((size_t) NE * R * D, 0.01f));
            uint16_t* h16 = dev(std::vector<uint16_t>((size_t) NE * R * F));
            float* y = dev(std::vector<float>((size_t) NE * R * D));
            x3::SetArgs sa;
            sa.wbase = wdev;
            sa.stride = blob;
            sa.lay = lay;
            sa.n_exp = NE;
            sa.bounds = db;
            sa.max_rows = R;
            sa.n_embd = D;
            sa.n_ff = F;
            sa.limit = 10.0f;
            auto one = [&] {
                gg::set_gate_up(sa, xs, D, di, h16, F, s);
                gg::set_down(sa, h16, F, y, D, s);
            };
            one();
            CU(cudaStreamSynchronize(s));
            cudaEventRecord(a, s);
            for (int i = 0; i < 5; ++i) one();
            cudaEventRecord(b, s);
            CU(cudaEventSynchronize(b));
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            ms /= 5;
            std::printf("bench %-40s %8.2f ms  %6.1f TFLOP/s  %6.0f GB/s of weights\n", what, ms,
                        2.0 * NE * R * 3.0 * D * F / (ms * 1e-3) / 1e12, NE * wbytes / (ms * 1e-3) / 1e9);
            cudaFree(wdev), cudaFree(db), cudaFree(di), cudaFree(xs), cudaFree(h16), cudaFree(y);
        }
    }
}

// the CPU lane's speed: one full-size expert (gate / up rows, then down rows) for 1 and 4 tokens, on `nth` threads (rows
// split between them), GB/s of weights read - MXFP4 (mx_cpu_rows) and the low-bit pairs (ggml-cpu's rows)
void bench_cpu(int nth) {
    const int D = 5120, F = 2304, reps = 6;
    auto run = [&](const char* what, double bytes, const std::function<void(int, int)>& rows, int n_rows) {
        auto pass = [&] {
            std::vector<std::thread> th;
            for (int t = 0; t < nth; ++t)
                th.emplace_back([&, t] { rows(n_rows * t / nth, n_rows * (t + 1) / nth); });
            for (auto& x : th) x.join();
        };
        pass();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) pass();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("cpu %-46s %8.2f ms  %6.1f GB/s (%d threads)\n", what, s * 1e3, bytes / s / 1e9, nth);
    };
    std::vector<float> x((size_t) 4 * D), h((size_t) 4 * F);
    for (auto& v : x) v = frand();
    for (auto& v : h) v = frand(0.2f);
    for (int n : {1, 4}) {
        Experts M = make_experts(D, F, 1);
        const float* xp[4] = {x.data(), x.data() + D, x.data() + 2 * D, x.data() + 3 * D};
        const float* hp[4] = {h.data(), h.data() + F, h.data() + 2 * F, h.data() + 3 * F};
        std::vector<float> yo((size_t) 4 * D), yg((size_t) 4 * F);
        char what[96];
        std::snprintf(what, sizeof(what), "MXFP4 expert, %d token%s", n, n > 1 ? "s" : "");
        run(what, 2.0 * gg::mx_bytes(F, D), [&](int r0, int r1) {
            float* yp[4] = {yg.data(), yg.data() + F, yg.data() + 2 * F, yg.data() + 3 * F};
            for (int i = 0; i < n; ++i) yp[i] += r0;
            gg::mx_cpu_rows(emat(M, 0, 0), gg::mx_row_bytes(D), D, r0, r1, xp, n, yp);
            gg::mx_cpu_rows(emat(M, 0, 1), gg::mx_row_bytes(D), D, r0, r1, xp, n, yp);
        }, F);
        run("  down", (double) gg::mx_bytes(D, F), [&](int r0, int r1) {
            float* yp[4] = {yo.data() + r0, yo.data() + D + r0, yo.data() + 2 * D + r0, yo.data() + 3 * D + r0};
            gg::mx_cpu_rows(emat(M, 0, 2), gg::mx_row_bytes(F), F, r0, r1, hp, n, yp);
        }, D);
        cudaFree(M.dev);
        for (const auto& pr : kQPairs) {
            QExperts E = make_q(D, F, 1, pr.first, pr.second);
            gg::IqCpu fm;
            std::string err;
            if (!gg::iq_cpu_fmt(pr.first, pr.second, D, F, fm, err)) continue;
            std::vector<uint8_t> aq((size_t) 4 * fm.act_bytes), hq((size_t) 4 * fm.h_bytes);
            const void* ap[4];
            const void* hqp[4];
            for (int i = 0; i < 4; ++i) {
                gg::iq_cpu_quant(fm.gu_act, xp[i], D, aq.data() + i * fm.act_bytes);
                gg::iq_cpu_quant(fm.d_act, hp[i], F, hq.data() + i * fm.h_bytes);
                ap[i] = aq.data() + i * fm.act_bytes;
                hqp[i] = hq.data() + i * fm.h_bytes;
            }
            std::snprintf(what, sizeof(what), "%s/%s expert, %d token%s", tn(pr.first), tn(pr.second), n, n > 1 ? "s" : "");
            run(what, 2.0 * F * E.rb(0), [&](int r0, int r1) {
                float* fp[4] = {yg.data(), yg.data() + F, yg.data() + 2 * F, yg.data() + 3 * F};
                gg::iq_cpu_gate_up(fm, E.mat(0, 0), E.mat(0, 1), ap, n, fp, r0, r1, 10.0f);
            }, F);
            run("  down", (double) D * E.rb(2), [&](int r0, int r1) {
                float* yp[4] = {yo.data(), yo.data() + D, yo.data() + 2 * D, yo.data() + 3 * D};
                gg::iq_cpu_down(fm, E.mat(0, 2), hqp, n, yp, r0, r1);
            }, D);
            cudaFree(E.dev);
        }
    }
}

// the decode kernels' speed: GB/s of weights read (a token's dense GEMVs and its experts)
void bench(cudaStream_t s) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    auto timed = [&](const char* what, double bytes, auto&& fn) {
        fn();
        CU(cudaStreamSynchronize(s));
        cudaEventRecord(a, s);
        const int reps = 50;
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(b, s);
        CU(cudaEventSynchronize(b));
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("bench %-40s %8.1f us  %6.0f GB/s\n", what, ms * 1000 / reps, bytes / (ms / reps * 1e-3) / 1e9);
    };
    const std::pair<int, int> shapes[] = {{1280, 32768}, {5120, 1280}, {8192, 5120}, {5120, 129280}};
    for (int fmt : {gg::kFmtBF16, gg::kFmtQ8_0})
        for (const auto& kn : shapes) {
            const int k = kn.first, n = kn.second;
            const size_t bytes = fmt == gg::kFmtBF16 ? (size_t) n * k * 2 : gg::q8_row_bytes(k) * n;
            uint8_t* dw;
            CU(cudaMalloc(&dw, bytes));
            CU(cudaMemset(dw, 0, bytes));
            float *dx, *dy;
            CU(cudaMalloc(&dx, (size_t) k * 4));
            CU(cudaMemset(dx, 0, (size_t) k * 4));
            CU(cudaMalloc(&dy, (size_t) n * 4));
            x3::Mat m;
            m.fmt = fmt;
            m.raw = dw;
            m.k = k;
            m.n = n;
            x3::MvJob j;
            j.m = &m;
            j.xf = dx;
            j.y = dy;
            char what[64];
            std::snprintf(what, sizeof(what), "mv %s %dx%d", fmt == gg::kFmtBF16 ? "BF16" : "Q8_0", n, k);
            timed(what, (double) bytes, [&] { gg::mv(&j, 1, s); });
            cudaFree(dw), cudaFree(dx), cudaFree(dy);
        }
    {   // a token's 6 experts at full size
        const int D = 5120, F = 2304;
        Experts E = make_experts(D, F, 6);
        std::vector<unsigned long long> pp(8, 0ull);
        std::vector<float> pw(8, 0.1f);
        for (int i = 0; i < 6; ++i) pp[i] = (unsigned long long) (E.dev + (size_t) i * E.blob);
        unsigned long long* dpp = dev(pp);
        float* dpw = dev(pw);
        float* dx = dev(std::vector<float>(D, 0.01f));
        float* dh = dev(std::vector<float>((size_t) 8 * F));
        float* dout = dev(std::vector<float>(D));
        x3::MoeArgs ma;
        ma.plan_ptr = dpp;
        ma.plan_w = dpw;
        ma.lay = E.lay;
        ma.k = 6;
        ma.n_embd = D;
        ma.n_ff = F;
        ma.limit = 10.0f;
        timed("6 MXFP4 experts (gate/up + down)", 6.0 * (2 * gg::mx_bytes(F, D) + gg::mx_bytes(D, F)), [&] {
            gg::moe_gate_up(ma, dx, dh, s);
            gg::moe_down(ma, dh, nullptr, dout, s);
        });
        cudaFree(E.dev);
        {   // the prompt path: 32 experts x 128 rows each (an 8K chunk's share of 384 experts)
            const int NE = 32, R = 128;
            Experts S = make_experts(D, F, NE);
            std::vector<int> bounds(NE + 1), idx(NE * R);
            for (int e = 0; e <= NE; ++e) bounds[e] = e * R;
            for (int i = 0; i < NE * R; ++i) idx[i] = i;
            int *db = dev(bounds), *di = dev(idx);
            float* xs = dev(std::vector<float>((size_t) NE * R * D, 0.01f));
            uint16_t* h16 = dev(std::vector<uint16_t>((size_t) NE * R * F));
            float* y = dev(std::vector<float>((size_t) NE * R * D));
            x3::SetArgs sa;
            sa.wbase = S.dev;
            sa.stride = S.blob;
            sa.lay = S.lay;
            sa.n_exp = NE;
            sa.bounds = db;
            sa.max_rows = R;
            sa.n_embd = D;
            sa.n_ff = F;
            sa.limit = 10.0f;
            const double flops = 2.0 * NE * R * 3.0 * D * F;
            cudaEvent_t a0, a1;
            cudaEventCreate(&a0);
            cudaEventCreate(&a1);
            gg::set_gate_up(sa, xs, D, di, h16, F, s);
            gg::set_down(sa, h16, F, y, D, s);
            cudaEventRecord(a0, s);
            for (int i = 0; i < 5; ++i) {
                gg::set_gate_up(sa, xs, D, di, h16, F, s);
                gg::set_down(sa, h16, F, y, D, s);
            }
            cudaEventRecord(a1, s);
            CU(cudaEventSynchronize(a1));
            float ms = 0;
            cudaEventElapsedTime(&ms, a0, a1);
            std::printf("bench %-40s %8.1f ms  %6.1f TFLOP/s\n", "prompt set: 32 experts x 128 rows", ms / 5,
                        flops / (ms / 5 * 1e-3) / 1e12);
            cudaFree(S.dev), cudaFree(db), cudaFree(di), cudaFree(xs), cudaFree(h16), cudaFree(y);
        }
        E.dev = nullptr;
        E = make_experts(D, F, 6);
        for (int i = 0; i < 6; ++i) pp[i] = (unsigned long long) (E.dev + (size_t) i * E.blob);
        CU(cudaMemcpy(dpp, pp.data(), pp.size() * 8, cudaMemcpyHostToDevice));
        timed("  gate/up alone", 6.0 * 2 * gg::mx_bytes(F, D), [&] { gg::moe_gate_up(ma, dx, dh, s); });
        timed("  down + combine alone", 6.0 * gg::mx_bytes(D, F), [&] { gg::moe_down(ma, dh, nullptr, dout, s); });
        cudaFree(E.dev), cudaFree(dpp), cudaFree(dpw), cudaFree(dx), cudaFree(dh), cudaFree(dout);
    }
}

// ---- the caches' packed rows (ds4_kernels.cu): DeepSeek's FP8 / FP4 roundings kept as codes and scales.  Each packed
// row against the BF16 row the same finish writes - a host decode of every value bit for bit - then the readers (the
// window ring and its rollback, the sparse attention over the compressed KV, the indexer's scores) the same bits from
// either, at full size (a 64K context's compressed rows)
namespace dk = strata::ds4;

float h_e2m1(uint32_t c) {
    static const float m[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = m[c & 7u];
    return (c & 8u) ? -v : v;
}
float h_e4m3(uint32_t q) {
    const int e = (int) ((q >> 3) & 15u), m = (int) (q & 7u);
    const float v = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7);
    return (q & 0x80u) ? -v : v;
}
float h_ue8m0(uint32_t b) {
    const uint32_t u = b << 23;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// value i of a packed row of d values
float h_unpack(int fmt, const uint8_t* row, int d, int i) {
    if (fmt == dk::kKvFp8) return h_e4m3(row[i]) * h_ue8m0(row[d + i / 32]);
    const uint32_t c = (uint32_t) row[i / 2] >> (4 * (i & 1));
    return h_e2m1(c) * (fmt == dk::kKvFp4 ? h_e4m3(row[d / 2 + i / 16]) : h_ue8m0(row[d / 2 + i / 32]));
}

// n rows of d values: groups of 16 at scales 2^-30 .. 2^10, some all zero, some with negative zeros, some tiny
std::vector<float> kv_rows(int n, int d) {
    std::vector<float> x((size_t) n * d);
    for (int r = 0; r < n; ++r)
        for (int g0 = 0; g0 < d; g0 += 16) {
            const int kind = (int) (rng() % 16u);
            const float sc = std::ldexp(1.0f, (int) (rng() % 41u) - 30);
            for (int e = g0; e < g0 + 16; ++e) {
                float v = frand(sc);
                if (kind == 0) v = 0.0f;
                else if (kind == 1 && e % 3 == 0) v = -0.0f;
                else if (kind == 2) v = frand(1e-7f);
                else if (kind == 3 && e == g0) v = sc * 1000.0f;   // (one value far above the rest)
                x[(size_t) r * d + e] = v;
            }
        }
    return x;
}

// finish(quant) of n rows of d: the BF16 rows (dst) and the packed rows (pdst), both from the same input
void kv_finish(const float* x, int n, int d, int quant, uint16_t* bf, uint8_t* pk, cudaStream_t s) {
    dk::Finish f;
    f.quant = quant;
    f.dst = bf;
    f.dst_stride = d;
    dk::finish(const_cast<float*>(x), n, d, d, f, s);
    dk::Finish p;
    p.quant = quant;
    p.pdst = pk;
    p.pdst_stride = (long long) dk::kv_row_bytes(quant, d);
    dk::finish(const_cast<float*>(x), n, d, d, p, s);
}

// every packed value decoded on the host against the BF16 row's bits
void kv_check_rows(const char* what, int fmt, const std::vector<uint16_t>& bf, const std::vector<uint8_t>& pk, int n,
                   int d) {
    const size_t rb = dk::kv_row_bytes(fmt, d);
    size_t bad = 0, at = 0;
    for (int r = 0; r < n; ++r)
        for (int i = 0; i < d; ++i) {
            const float v = h_unpack(fmt, pk.data() + (size_t) r * rb, d, i);
            uint32_t u;
            std::memcpy(&u, &v, 4);
            if ((u & 0xffffu) != 0u || (uint16_t) (u >> 16) != bf[(size_t) r * d + i]) {
                if (bad++ == 0) at = (size_t) r * d + i;
            }
        }
    if (bad) ++g_fail;
    std::printf("%-44s %s  %d rows x %d, %zu of %zu values differ (first at %zu), %zu B a row vs %d\n", what,
                bad ? "FAIL" : "ok  ", n, d, bad, (size_t) n * d, at, rb, d * 2);
}

void kv_same(const char* what, const void* a, const void* b, size_t bytes) {
    const bool ok = std::memcmp(a, b, bytes) == 0;
    if (!ok) ++g_fail;
    std::printf("%-44s %s  %zu bytes %s\n", what, ok ? "ok  " : "FAIL", bytes, ok ? "identical" : "DIFFER");
}

// the router's F32 GEMV over a prompt's rows (gemv_f32_rows_kernel, kGemvT tokens a block) against a launch a token -
// bit for bit, so the prompt and the token path route alike (DeepSeek-V4.1's 384 x 5120; odd row and token counts)
void test_router_rows(cudaStream_t s) {
    for (int T : {2, 37, 300}) {
        const int n = 384, k = 5120;
        std::vector<float> w((size_t) n * k), x((size_t) T * k);
        for (auto& v : w) v = frand(0.05f);
        for (auto& v : x) v = frand(2.0f);
        float *dw = dev(w), *dx = dev(x), *y1 = nullptr, *yt = nullptr;
        CU(cudaMalloc(&y1, (size_t) T * 1024 * 4));
        CU(cudaMalloc(&yt, (size_t) T * 1024 * 4));
        strata::ds4::gemv_f32(dw, dx, n, k, T, yt, 1024, s);
        for (int t = 0; t < T; ++t) strata::ds4::gemv_f32(dw, dx + (size_t) t * k, n, k, 1, y1 + (size_t) t * 1024, 1024, s);
        std::vector<float> a((size_t) T * 1024), b((size_t) T * 1024);
        CU(cudaMemcpy(a.data(), yt, a.size() * 4, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost));
        int diff = 0;
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < n; ++r) diff += std::memcmp(&a[(size_t) t * 1024 + r], &b[(size_t) t * 1024 + r], 4) != 0;
        std::printf("router rows, %d tokens vs a launch a token     %s    %d of %d logits differ\n", T,
                    diff ? "FAIL" : "ok  ", diff, T * n);
        if (diff) ++g_fail;
        cudaFree(dw), cudaFree(dx), cudaFree(y1), cudaFree(yt);
    }
}

void test_kv_pack(cudaStream_t s) {
    const int HD = 512, IX = 128, NC = 65536 / 4 + 1, WIN = 128;
    // the compressed KV (FP4, an E4M3 scale per 16) and the index keys (FP4, a power-of-two scale per 32)
    std::vector<float> xc = kv_rows(NC, HD), xi = kv_rows(NC, IX);
    float *d_xc = dev(xc), *d_xi = dev(xi);
    uint16_t *c16 = nullptr, *i16 = nullptr;
    uint8_t *c4 = nullptr, *i4 = nullptr;
    CU(cudaMalloc(&c16, (size_t) NC * HD * 2));
    CU(cudaMalloc(&i16, (size_t) NC * IX * 2));
    CU(cudaMalloc(&c4, (size_t) NC * dk::kv_row_bytes(dk::kKvFp4, HD)));
    CU(cudaMalloc(&i4, (size_t) NC * dk::kv_row_bytes(dk::kKvFp4p, IX)));
    kv_finish(d_xc, NC, HD, dk::kKvFp4, c16, c4, s);
    kv_finish(d_xi, NC, IX, dk::kKvFp4p, i16, i4, s);
    CU(cudaStreamSynchronize(s));
    kv_check_rows("kv pack: compressed KV (FP4 + E4M3 / 16)", dk::kKvFp4, host(c16, (size_t) NC * HD),
                  host(c4, (size_t) NC * dk::kv_row_bytes(dk::kKvFp4, HD)), NC, HD);
    kv_check_rows("kv pack: index keys (FP4 + 2^k / 32)", dk::kKvFp4p, host(i16, (size_t) NC * IX),
                  host(i4, (size_t) NC * dk::kv_row_bytes(dk::kKvFp4p, IX)), NC, IX);
    kv_same("kv pack: finish leaves its input", xc.data(), host(d_xc, xc.size()).data(), xc.size() * 4);

    // the window ring (FP8, a power-of-two scale per 32): batches of 128 rows through the finish's BF16 rows, into the
    // ring (its scales derived from the BF16 values) and back out
    {
        const size_t rb = dk::kv_row_bytes(dk::kKvFp8, HD);
        std::vector<float> xw = kv_rows(WIN, HD);
        float* d_xw = dev(xw);
        uint16_t *ext = nullptr, *ext2 = nullptr;
        uint8_t *ring = nullptr, *pk = nullptr, *bak = nullptr;
        CU(cudaMalloc(&ext, (size_t) 2 * WIN * HD * 2));
        CU(cudaMalloc(&ext2, (size_t) WIN * HD * 2));
        CU(cudaMalloc(&ring, (size_t) WIN * rb));
        CU(cudaMalloc(&pk, (size_t) WIN * rb));
        CU(cudaMalloc(&bak, (size_t) dk::kWin * rb));
        size_t bad = 0, bad_pk = 0, n_vals = 0;
        for (int b = 0; b < 64; ++b) {
            if (b > 0) {
                xw = kv_rows(WIN, HD);
                CU(cudaMemcpy(d_xw, xw.data(), xw.size() * 4, cudaMemcpyHostToDevice));
            }
            kv_finish(d_xw, WIN, HD, dk::kKvFp8, ext + (size_t) WIN * HD, pk, s);
            dk::ext_to_ring(ext, ring, 0, WIN, WIN, HD, s);
            dk::ring_to_ext(ring, ext2, WIN, WIN, HD, s);   // (p0 = n: the rows as they lie)
            CU(cudaStreamSynchronize(s));
            const std::vector<uint16_t> want = host(ext + (size_t) WIN * HD, (size_t) WIN * HD),
                                        got = host(ext2, (size_t) WIN * HD);
            for (size_t i = 0; i < want.size(); ++i) bad += want[i] != got[i];
            const std::vector<uint8_t> hr = host(ring, (size_t) WIN * rb), hp = host(pk, (size_t) WIN * rb);
            for (int r = 0; r < WIN; ++r)   // the ring's codes against the finish's own packing: the same values
                for (int i = 0; i < HD; ++i)
                    bad_pk += h_unpack(dk::kKvFp8, hr.data() + (size_t) r * rb, HD, i) !=
                                  h_unpack(dk::kKvFp8, hp.data() + (size_t) r * rb, HD, i) ||
                              std::signbit(h_unpack(dk::kKvFp8, hr.data() + (size_t) r * rb, HD, i)) !=
                                  std::signbit(h_unpack(dk::kKvFp8, hp.data() + (size_t) r * rb, HD, i));
            n_vals += want.size();
        }
        if (bad || bad_pk) ++g_fail;
        std::printf("%-44s %s  %zu values, %zu differ through the ring, %zu from the finish's packing; %zu B a row vs %d\n",
                    "kv pack: window ring (FP8 + 2^k / 32)", bad || bad_pk ? "FAIL" : "ok  ", n_vals, bad, bad_pk, rb,
                    HD * 2);
        // a speculative window's rollback: 16 rows over the ring at position 1000 (backed up), the first 5 kept
        const std::vector<uint8_t> before = host(ring, (size_t) WIN * rb);
        xw = kv_rows(dk::kWin, HD);
        CU(cudaMemcpy(d_xw, xw.data(), (size_t) dk::kWin * HD * 4, cudaMemcpyHostToDevice));
        kv_finish(d_xw, dk::kWin, HD, dk::kKvFp8, ext + (size_t) WIN * HD, pk, s);
        dk::ext_to_ring(ext, ring, 1000, WIN, dk::kWin, HD, s, bak);
        CU(cudaStreamSynchronize(s));
        std::vector<uint8_t> want = before;
        const std::vector<uint8_t> after = host(ring, (size_t) WIN * rb);
        for (int t = 0; t < 5; ++t) {
            const size_t r = (size_t) ((1000 + t) % WIN);
            std::memcpy(want.data() + r * rb, after.data() + r * rb, rb);
        }
        dk::ring_restore(ring, bak, 1000, WIN, 5, dk::kWin, HD, s);
        CU(cudaStreamSynchronize(s));
        kv_same("kv pack: window ring rollback", want.data(), host(ring, (size_t) WIN * rb).data(), want.size());
        CU(cudaFree(d_xw));
        CU(cudaFree(ext));
        CU(cudaFree(ext2));
        CU(cudaFree(ring));
        CU(cudaFree(pk));
        CU(cudaFree(bak));
    }

    // the sparse attention: the window rows and 512 selected compressed rows each, 64 heads - the BF16 rows' kernel
    // against the packed rows' every way (decoded once a query into the scratch; staged, every number of heads a
    // block), and the time of each (a decode step's query, a few, a verification's 16, chunk slabs of 16 and 256)
    {
        const int TM = 256, NH = 64, SEL = 512, ST = 16;
        std::vector<float> q((size_t) TM * NH * HD), sink(NH);
        for (float& v : q) v = frand(0.2f);
        for (float& v : sink) v = frand(1.0f);
        std::vector<int> sel((size_t) TM * SEL);
        for (int& v : sel) v = (int) (rng() % (unsigned) NC);
        std::vector<float> xw = kv_rows(WIN + TM, HD);
        float *d_q = dev(q), *d_sink = dev(sink), *d_xw = dev(xw), *o1 = nullptr, *o2 = nullptr;
        int* d_sel = dev(sel);
        uint16_t* ext = nullptr;
        uint8_t* pk = nullptr;
        dk::AttnScratch scr;
        scr.max_t = ST;
        scr.cap = SEL;
        CU(cudaMalloc(&scr.rows, (size_t) ST * SEL * HD * 2));
        CU(cudaMalloc(&ext, (size_t) (WIN + TM) * HD * 2));
        CU(cudaMalloc(&pk, (size_t) (WIN + TM) * dk::kv_row_bytes(dk::kKvFp8, HD)));
        CU(cudaMalloc(&o1, q.size() * 4));
        CU(cudaMalloc(&o2, q.size() * 4));
        kv_finish(d_xw, WIN + TM, HD, dk::kKvFp8, ext, pk, s);
        cudaEvent_t e0, e1;
        CU(cudaEventCreate(&e0));
        CU(cudaEventCreate(&e1));
        const float scale = 1.0f / std::sqrt((float) HD);
        const struct {
            int T, gen, iters;
            const char* what;
        } shapes[] = {{1, 0, 400, "1 query"},   {2, 0, 400, "2 queries"},  {4, 0, 200, "4 queries"},
                      {8, 0, 200, "8 queries"}, {16, 0, 100, "16 queries"}, {16, 1, 100, "a chunk slab of 16"},
                      {256, 1, 10, "a chunk slab of 256"}};
        for (const auto& sh : shapes) {
            const int T = sh.T;
            dk::AttnRows ar;
            if (sh.gen) {   // a chunk's rows at 40000.. (ratio 4: every query its 512)
                ar.gen = 1;
                ar.p0 = 40000;
                ar.t0 = 0;
                ar.win = WIN;
                ar.R = 4;
                ar.kcap = SEL;
            } else
                for (int t = 0; t < T; ++t) {
                    ar.win0[t] = t + 1;
                    ar.nwin[t] = WIN;
                    ar.nsel[t] = SEL - 31 * t;
                }
            const size_t n = (size_t) T * NH * HD;
            dk::sparse_attn(d_q, ext, c16, d_sel, SEL, ar, d_sink, NH, HD, T, scale, o1, s, dk::kKvBf16);
            CU(cudaStreamSynchronize(s));
            const std::vector<float> want = host(o1, n);
            bool same = true;
            for (int g : {-1, 0, 1, 2, 4, 8}) {
                CU(cudaMemset(o2, 0xff, n * 4));
                dk::sparse_attn(d_q, ext, c4, d_sel, SEL, ar, d_sink, NH, HD, T, scale, o2, s, dk::kKvFp4, g, &scr);
                CU(cudaStreamSynchronize(s));
                same = same && std::memcmp(want.data(), host(o2, n).data(), n * 4) == 0;
            }
            // the time: BF16 rows; packed as the engine runs it (0), through the scratch (-1), staged (its own G)
            float us[4];
            const int ways[4] = {99, 0, -1, -2};
            for (int w = 0; w < 4; ++w) {
                CU(cudaEventRecord(e0, s));
                for (int it = 0; it < sh.iters; ++it) {
                    if (ways[w] == 99)
                        dk::sparse_attn(d_q, ext, c16, d_sel, SEL, ar, d_sink, NH, HD, T, scale, o1, s, dk::kKvBf16);
                    else
                        dk::sparse_attn(d_q, ext, c4, d_sel, SEL, ar, d_sink, NH, HD, T, scale, o1, s, dk::kKvFp4,
                                        ways[w] == -2 ? 0 : ways[w], ways[w] == -2 ? nullptr : &scr);
                }
                CU(cudaEventRecord(e1, s));
                CU(cudaEventSynchronize(e1));
                float ms = 0.0f;
                CU(cudaEventElapsedTime(&ms, e0, e1));
                us[w] = ms * 1000.0f / sh.iters;
            }
            if (T >= 16) {   // kAttnWmma: the matrix cores (RDNA 3/4; elsewhere the packed kernel)
                CU(cudaMemset(o2, 0xff, n * 4));
                dk::sparse_attn(d_q, ext, c4, d_sel, SEL, ar, d_sink, NH, HD, T, scale, o2, s, dk::kKvFp4, dk::kAttnWmma,
                                &scr);
                CU(cudaStreamSynchronize(s));
                const std::vector<float> got = host(o2, n);
                double e2 = 0, r2 = 0, worst = 0;
                bool finite = true;
                for (size_t i = 0; i < n; ++i) {
                    finite = finite && std::isfinite(got[i]) && std::isfinite(want[i]);
                    const double dv = (double) got[i] - want[i];
                    e2 += dv * dv;
                    r2 += (double) want[i] * want[i];
                    worst = std::max(worst, std::fabs(dv));
                }
                const double rel = std::sqrt(e2 / std::max(r2, 1e-30)), rms = std::sqrt(r2 / (double) n);
                CU(cudaEventRecord(e0, s));
                for (int it = 0; it < sh.iters; ++it)
                    dk::sparse_attn(d_q, ext, c4, d_sel, SEL, ar, d_sink, NH, HD, T, scale, o2, s, dk::kKvFp4,
                                    dk::kAttnWmma, &scr);
                CU(cudaEventRecord(e1, s));
                CU(cudaEventSynchronize(e1));
                float ms = 0.0f;
                CU(cudaEventElapsedTime(&ms, e0, e1));
                const bool ok = finite && rel < 2e-3 && worst < 2e-2 * std::max(rms, 1e-6) * 10;
                if (!ok) ++g_fail;
                std::printf("kv pack: sparse attention, %-22s %s  matrix cores: finite %d, max_abs %.2e, rel %.2e, "
                            "worst/rms %.2e, %.1f us a call\n", sh.what, ok ? "ok  " : "FAIL", (int) finite, worst,
                            rel, worst / std::max(rms, 1e-30), ms * 1000.0f / sh.iters);
            }
            if (!same) ++g_fail;
            char what[96];
            std::snprintf(what, sizeof what, "kv pack: sparse attention, %s", sh.what);
            std::printf("%-44s %s  %s every way; BF16 rows %.1f us a call, packed %.1f (decoded once a query %.1f, "
                        "staged %.1f)\n",
                        what, same ? "ok  " : "FAIL", same ? "identical" : "DIFFER", us[0], us[1], us[2], us[3]);
        }
        // Keep the broad-range quantization fixture above; the extra shapes below use bounded KV values.
        // Its 2^20 spikes can make tiny dot-product differences flip a saturated softmax's winning row.
        // DS4_ATTN_TEST_EXTREME=1 reproduces that numerical stress with the same queries and error limits.
        const bool extreme = std::getenv("DS4_ATTN_TEST_EXTREME") != nullptr;
        if (!extreme) {
            for (float& v : xc) v = std::ldexp(v, -20);
            for (float& v : xw) v = std::ldexp(v, -20);
            CU(cudaMemcpy(d_xc, xc.data(), xc.size() * 4, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(d_xw, xw.data(), xw.size() * 4, cudaMemcpyHostToDevice));
            kv_finish(d_xc, NC, HD, dk::kKvFp4, c16, c4, s);
            kv_finish(d_xw, WIN + TM, HD, dk::kKvFp8, ext, pk, s);
        }
        // The same gfx11 error limits on layout/tile edges and the model's real prompt slab dimensions.
        // Exercise each 16-head group with distinct data, including zero/tiny/large queries and sink domination.
        const auto wmma_case = [&](const char* what, int T, int heads, const dk::AttnRows& ar) {
            const size_t n = (size_t) T * heads * HD;
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < heads; ++h) {
                    const float mag = h % 16 == 0 ? 0.0f : h % 16 == 1 ? 1e-6f : h % 16 == 2 ? 1e3f : 0.2f;
                    for (int k = 0; k < HD; ++k) q[((size_t) t * heads + h) * HD + k] = frand(mag);
                }
            for (int h = 0; h < heads; ++h) sink[h] = h % 16 == 3 ? 1e6f : h % 16 == 4 ? -1e6f : frand();
            CU(cudaMemcpy(d_q, q.data(), n * 4, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(d_sink, sink.data(), heads * 4, cudaMemcpyHostToDevice));
            CU(cudaMemset(o1, 0xff, n * 4));
            CU(cudaMemset(o2, 0xff, n * 4));
            dk::sparse_attn(d_q, ext, c16, d_sel, SEL, ar, d_sink, heads, HD, T, scale, o1, s, dk::kKvBf16);
            dk::sparse_attn(d_q, ext, c4, d_sel, SEL, ar, d_sink, heads, HD, T, scale, o2, s, dk::kKvFp4,
                            dk::kAttnWmma, &scr);
            CU(cudaStreamSynchronize(s));
            const auto want = host(o1, n), got = host(o2, n);
            double e2 = 0, r2 = 0, worst = 0;
            size_t worst_i = 0;
            std::vector<double> head_e2(heads, 0.0), head_r2(heads, 0.0), head_worst(heads, 0.0);
            bool finite = true;
            for (size_t i = 0; i < n; ++i) {
                finite = finite && std::isfinite(got[i]) && std::isfinite(want[i]);
                const double dv = (double) got[i] - want[i];
                e2 += dv * dv;
                r2 += (double) want[i] * want[i];
                if (std::fabs(dv) > worst) worst_i = i;
                worst = std::max(worst, std::fabs(dv));
                const int h = (int) ((i / HD) % heads);
                head_e2[h] += dv * dv;
                head_r2[h] += (double) want[i] * want[i];
                head_worst[h] = std::max(head_worst[h], std::fabs(dv));
            }
            const double rel = std::sqrt(e2 / std::max(r2, 1e-30)), rms = std::sqrt(r2 / (double) n);
            const bool ok = finite && rel < 2e-3 && worst < 2e-2 * std::max(rms, 1e-6) * 10;
            if (!ok) ++g_fail;
            std::printf("kv pack: WMMA %-24s T=%d H=%d %s finite %d max_abs %.2e rel %.2e worst/rms %.2e\n",
                        what, T, heads, ok ? "ok  " : "FAIL", (int) finite, worst, rel, worst / std::max(rms, 1e-30));
            const int worst_t = (int) (worst_i / ((size_t) heads * HD));
            const int worst_h = (int) ((worst_i / HD) % heads), worst_d = (int) (worst_i % HD);
            const float qmag = worst_h % 16 == 0 ? 0.0f : worst_h % 16 == 1 ? 1e-6f : worst_h % 16 == 2 ? 1e3f : 0.2f;
            std::printf("  WMMA worst index %zu token %d head %d dim %d query_mag %.6g ref %.9g got %.9g\n",
                        worst_i, worst_t, worst_h, worst_d, qmag, want[worst_i], got[worst_i]);
            if (!ok)
                for (int h = 0; h < heads; ++h) {
                    const double hrms = std::sqrt(head_r2[h] / ((double) T * HD));
                    std::printf("  WMMA head %d rel %.6e max_abs %.6e worst/rms %.6e\n", h,
                                std::sqrt(head_e2[h] / std::max(head_r2[h], 1e-30)), head_worst[h],
                                head_worst[h] / std::max(hrms, 1e-30));
                }
        };
        dk::AttnRows edges{};
        const int nwin[] = {0, 1, 15, 16, 17, 31, 32, 33, 127, 128, 0, 1, 15, 16, 17, 128};
        const int nsel[] = {0, 0, 0, 0, 0, 1, 15, 16, 17, 31, 32, 33, 511, 512, 1, 512};
        for (int t = 0; t < 16; ++t) {
            edges.win0[t] = t + 1;
            edges.nwin[t] = nwin[t];
            edges.nsel[t] = nsel[t];
        }
        for (int heads : {16, 32, 64}) wmma_case("partial tiles / empty rows", 16, heads, edges);
        // Window-only and R=1/2 (this Maya-S pack), plus R=4/128: 256-row slabs at the causal start and
        // 8K/32K/64K/120K depth. Keep the original stress cohort's order/data for reproducibility.
        // A nonzero t0 also checks slabs taken from the middle of a chunk's extended window buffer.
        for (long long p0 : {0LL, 8192LL, 32768LL, 65536LL, 122880LL})
            for (int ratio : (extreme ? std::vector<int>{4, 128} : std::vector<int>{0, 1, 2, 4, 128})) {
                dk::AttnRows ar{};
                ar.gen = 1; ar.p0 = p0; ar.win = WIN; ar.R = ratio; ar.kcap = SEL;
                char what[64];
                std::snprintf(what, sizeof what, "position %lld ratio %d", p0, ratio);
                wmma_case(what, TM, NH, ar);
            }
        dk::AttnRows offset{};
        offset.gen = 1; offset.p0 = 7; offset.t0 = 7; offset.win = WIN; offset.R = 4; offset.kcap = SEL;
        wmma_case("offset slab / partial rows", 17, NH, offset);
        CU(cudaEventDestroy(e0));
        CU(cudaEventDestroy(e1));
        CU(cudaFree(d_q));
        CU(cudaFree(d_sink));
        CU(cudaFree(d_xw));
        CU(cudaFree(d_sel));
        CU(cudaFree(ext));
        CU(cudaFree(pk));
        CU(cudaFree(o1));
        CU(cudaFree(o2));
        CU(cudaFree(scr.rows));
    }

    // the indexer's scores over every key (32 heads), BF16 vs packed keys
    {
        const int T = 4, NH = 32;
        std::vector<float> q((size_t) T * NH * IX), w((size_t) T * 64);
        for (float& v : q) v = frand(1.0f);
        for (float& v : w) v = frand(0.3f);
        float *d_q = dev(q), *d_w = dev(w), *o1 = nullptr, *o2 = nullptr;
        const long long ss = NC + 64;
        CU(cudaMalloc(&o1, (size_t) T * ss * 4));
        CU(cudaMalloc(&o2, (size_t) T * ss * 4));
        CU(cudaMemset(o1, 0, (size_t) T * ss * 4));
        CU(cudaMemset(o2, 0, (size_t) T * ss * 4));
        dk::Rows lim;
        for (int t = 0; t < T; ++t) {
            lim.n[t] = NC - 3 * t;
            lim.k[t] = 512;
        }
        dk::index_score(d_q, d_w, i16, NH, IX, T, NC, nullptr, 0, 1, lim, o1, ss, s, dk::kKvBf16);
        dk::index_score(d_q, d_w, i4, NH, IX, T, NC, nullptr, 0, 1, lim, o2, ss, s, dk::kKvFp4p);
        CU(cudaStreamSynchronize(s));
        kv_same("kv pack: index scores", host(o1, (size_t) T * ss).data(), host(o2, (size_t) T * ss).data(),
                (size_t) T * ss * 4);
        {   // (the time of each over the 16385 keys, 50 times)
            cudaEvent_t e0, e1;
            CU(cudaEventCreate(&e0));
            CU(cudaEventCreate(&e1));
            float ms[2];
            for (int f = 0; f < 2; ++f) {
                CU(cudaEventRecord(e0, s));
                for (int it = 0; it < 50; ++it)
                    dk::index_score(d_q, d_w, f ? (const void*) i4 : (const void*) i16, NH, IX, T, NC, nullptr, 0, 1, lim,
                                    o1, ss, s, f ? dk::kKvFp4p : dk::kKvBf16);
                CU(cudaEventRecord(e1, s));
                CU(cudaEventSynchronize(e1));
                CU(cudaEventElapsedTime(&ms[f], e0, e1));
            }
            std::printf("%-44s       BF16 keys %.1f us a call, packed %.1f us\n", "kv pack: index score time", ms[0] * 20.0f,
                        ms[1] * 20.0f);
            CU(cudaEventDestroy(e0));
            CU(cudaEventDestroy(e1));
        }
        CU(cudaFree(d_q));
        CU(cudaFree(d_w));
        CU(cudaFree(o1));
        CU(cudaFree(o2));
    }
    CU(cudaFree(d_xc));
    CU(cudaFree(d_xi));
    CU(cudaFree(c16));
    CU(cudaFree(i16));
    CU(cudaFree(c4));
    CU(cudaFree(i4));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("kv pack: %s\n", cudaGetErrorString(e));
        ++g_fail;
    }
}

}  // namespace

// ---------------------------------------------------------------- the decode path's PCIe lane (ds4_pcie.cuh)
// A token's 6 experts, 3 of them streamed: off the plan at first (the CPU lane's, to the token path), their blobs in
// pinned memory (as the host's readers leave them), pulled by stream_fetch into slots elsewhere once their ready flags
// are set (after the kernel started: it waits for them) - or read where they lie (copy 0: an integrated GPU's way) -
// taken into the plan, their gate / up in a second pass: the output bit for bit the one with all 6 in the plan from the
// start (a streamed expert is a VRAM one to the sums); with none streamed, the first pass's alone
// (blobs on the device, hblobs their bytes on the host)
void test_lane(const char* name, const x3::ExpertLayout& lay, const uint8_t* blobs, const uint8_t* hblobs, size_t blob,
               int D, int F, cudaStream_t s) {
    namespace pc = strata::ds4::pcie;
    const int k = 6;
    const float lim = 10.0f;
    std::vector<float> x(D), sh(D), w(8, 0.0f);
    for (auto& v : x) v = frand();
    for (auto& v : sh) v = frand(0.1f);
    for (int i = 0; i < k; ++i) w[i] = 0.1f + 0.05f * (float) i;
    float *dx = dev(x), *dsh = dev(sh), *dw = dev(w);
    float* dh = dev(std::vector<float>((size_t) 8 * F));
    float *da = dev(std::vector<float>(D)), *db = dev(std::vector<float>(D));
    uint8_t *slots = nullptr, *pinned = nullptr;
    CU(cudaMalloc(&slots, 8 * blob));
    CU(cudaHostAlloc(&pinned, 8 * blob, cudaHostAllocMapped));
    pc::Pull* pull = nullptr;
    CU(cudaHostAlloc(&pull, sizeof(pc::Pull), cudaHostAllocMapped));
    std::memset(pull, 0, sizeof(pc::Pull));
    unsigned* dflag = dev(std::vector<unsigned>(pc::kPullFlags, 0u));
    std::vector<unsigned long long> pa(8, 0ull);
    for (int i = 0; i < k; ++i) pa[i] = (unsigned long long) (blobs + (size_t) i * blob);
    unsigned long long *dpa = dev(pa), *dpb = nullptr, *dp2 = dev(std::vector<unsigned long long>(8, 0ull));
    x3::MoeArgs ma;
    ma.plan_w = dw;
    ma.lay = lay;
    ma.k = k;
    ma.n_embd = D;
    ma.n_ff = F;
    ma.limit = lim;
    ma.plan_ptr = dpa;   // all six in the plan
    gg::moe_gate_up(ma, dx, dh, s);
    gg::moe_down(ma, dh, dsh, da, s);
    CU(cudaStreamSynchronize(s));
    const std::vector<float> ya = host(da, D);
    pc::Mats mt;   // (each matrix with the padding after it: within the blob)
    for (int r = 0; r < 3; ++r) {
        mt.off[r] = lay.trellis[r];
        mt.bytes[r] = (r < 2 ? lay.trellis[r + 1] : blob) - lay.trellis[r];
    }
    unsigned seq = 100;
    for (const int copy : {1, 0})
        for (const unsigned mask : {0x1Au, 0u}) {   // picks 1, 3, 4 streamed; none
            ++seq;
            std::vector<unsigned long long> pb = pa;
            for (int i = 0; i < 8; ++i) {
                pull->src[i] = 0ull;
                pull->bad[i] = 0u;
                for (int r = 0; r < 4; ++r) pull->ready[i][r] = 0u;
                if (i < k && ((mask >> i) & 1u)) {
                    pull->src[i] = (unsigned long long) (pinned + (size_t) i * blob);
                    pb[i] = 0ull;
                }
            }
            dpb = dev(pb);
            x3::MoeArgs mb = ma;
            mb.plan_ptr = dpb;
            gg::moe_gate_up(mb, dx, dh, s);
            pc::stream_fetch(pull, seq, dflag, slots, blob, mt, copy != 0, dpb, dp2, k, 4, s);
            x3::MoeArgs m2 = ma;
            m2.plan_ptr = dp2;
            gg::moe_gate_up(m2, dx, dh, s);
            gg::moe_down(mb, dh, dsh, db, s);
            // the host's side, after the device started waiting: the pull's seq, then each expert's matrices read and
            // flagged one by one
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            *(volatile unsigned*) &pull->seq = seq;
            for (int i = 0; i < k; ++i)
                if ((mask >> i) & 1u)
                    for (int r = 0; r < 3; ++r) {
                        std::memcpy(pinned + (size_t) i * blob + mt.off[r], hblobs + (size_t) i * blob + mt.off[r],
                                    mt.bytes[r]);
                        std::atomic_thread_fence(std::memory_order_release);
                        *(volatile unsigned*) &pull->ready[i][r] = seq;
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
            CU(cudaStreamSynchronize(s));
            const std::vector<float> yb = host(db, D);
            const bool same = std::memcmp(ya.data(), yb.data(), (size_t) D * 4) == 0;
            const bool done = *(volatile unsigned*) &pull->done == seq;   // (the pull's end, as the host sees it)
            if (!same || !done) ++g_fail;
            char what[96];
            std::snprintf(what, sizeof what, "lane: %s, %s streamed%s", name, mask ? "3 of 6" : "none",
                          copy ? "" : " (read in place)");
            std::printf("%-44s %s%s\n", what, same ? "ok    bit for bit all in VRAM" : "FAIL  differs from all in VRAM",
                        done ? "" : " - FAIL: the pull's end not seen");
            cudaFree(dpb);
        }
    cudaFree(dx), cudaFree(dsh), cudaFree(dw), cudaFree(dh), cudaFree(da), cudaFree(db), cudaFree(slots), cudaFree(dpa);
    cudaFree(dp2), cudaFree(dflag);
    cudaFreeHost(pinned), cudaFreeHost(pull);
}
// The low-bit experts' activations as the CPU lane rounds them (act_q8k: ggml's Q8_K, the GPU's unless DS4_IQ_Q8K=0):
// the int8 values and scales byte for byte the CPU lane's (iq_cpu_quant to Q8_K), ties and an all-zero block among
// them; then each low-bit pair's expert on the GPU (as this process's DS4_IQ_Q8K has it) against the CPU lane's
// (gg::iq_cpu_*), the largest difference of the output relative to its largest value
void test_q8k(cudaStream_t s) {
    const int k = 1024, rows = 6;
    std::vector<float> x((size_t) rows * k);
    for (auto& v : x) v = frand();
    for (int j = 0; j < 256; ++j) x[(size_t) 1 * k + j] = 0.0f;                     // (an all-zero block)
    x[(size_t) 2 * k + 3] = 2.5f, x[(size_t) 2 * k + 9] = -2.5f;                    // (a tie: the first one's sign)
    for (int j = 0; j < 256; ++j) x[(size_t) 3 * k + 256 + j] = 0.5f * (float) (j % 7) - 1.5f;   // (exact halves)
    float* dx = dev(x);
    int8_t* dq = dev(std::vector<int8_t>((size_t) rows * k));
    float* dd = dev(std::vector<float>((size_t) rows * k / 32));
    gg::act_q8k(dx, rows, k, dq, dd, s);
    CU(cudaStreamSynchronize(s));
    const std::vector<int8_t> q = host(dq, (size_t) rows * k);
    const std::vector<float> d = host(dd, (size_t) rows * k / 32);
    size_t bad = 0;
    std::vector<uint8_t> blk((size_t) (k / 256) * 292);
    for (int r = 0; r < rows; ++r) {
        gg::iq_cpu_quant(15 /* GGML_TYPE_Q8_K */, x.data() + (size_t) r * k, k, blk.data());
        for (int b = 0; b < k / 256; ++b) {
            float cd;
            std::memcpy(&cd, blk.data() + (size_t) b * 292, 4);
            const int8_t* cq = (const int8_t*) (blk.data() + (size_t) b * 292 + 4);
            for (int j = 0; j < 256; ++j) bad += cq[j] != q[(size_t) r * k + b * 256 + j];
            for (int c = 0; c < 8; ++c) bad += std::memcmp(&cd, &d[(size_t) r * (k / 32) + b * 8 + c], 4) != 0;
        }
    }
    if (bad) ++g_fail;
    std::printf("%-44s %s  %zu of %d values / scales differ\n", "q8k: the GPU's Q8_K = the CPU lane's", bad ? "FAIL" : "ok  ",
                bad, rows * k + rows * k / 32);
    cudaFree(dx), cudaFree(dq), cudaFree(dd);
    const char* on = std::getenv("DS4_IQ_Q8K");
    for (const auto& pr : kQPairs) {
        const int D = 1024, F = 512;
        const float lim = 10.0f;
        QExperts Q = make_q(D, F, 1, pr.first, pr.second);
        gg::IqCpu fm;
        std::string err;
        if (!gg::iq_cpu_fmt(pr.first, pr.second, D, F, fm, err)) continue;
        std::vector<float> xs(D);
        for (auto& v : xs) v = frand();
        std::vector<uint8_t> aq((size_t) D * 4), hq((size_t) D * 4);
        std::vector<float> hc(F), yc(D);
        gg::iq_cpu_quant(fm.gu_act, xs.data(), D, aq.data());
        const void* ap[1] = {aq.data()};
        float* hp[1] = {hc.data()};
        gg::iq_cpu_gate_up(fm, Q.mat(0, 0), Q.mat(0, 1), ap, 1, hp, 0, F, lim);
        gg::iq_cpu_quant(fm.d_act, hc.data(), F, hq.data());
        const void* hqp[1] = {hq.data()};
        float* yp[1] = {yc.data()};
        gg::iq_cpu_down(fm, Q.mat(0, 2), hqp, 1, yp, 0, D);
        std::vector<unsigned long long> pp(8, 0ull);
        pp[0] = (unsigned long long) Q.dev;
        std::vector<float> w(8, 0.0f);
        w[0] = 1.0f;
        unsigned long long* dpp = dev(pp);
        float *dw = dev(w), *dxs = dev(xs), *dh = dev(std::vector<float>((size_t) 8 * F));
        float *dsh = dev(std::vector<float>(D, 0.0f)), *dout = dev(std::vector<float>(D));
        x3::MoeArgs a;
        a.plan_ptr = dpp;
        a.plan_w = dw;
        a.lay = Q.lay;
        a.k = 1;
        a.n_embd = D;
        a.n_ff = F;
        a.limit = lim;
        gg::moe_gate_up(a, dxs, dh, s);
        gg::moe_down(a, dh, dsh, dout, s);
        CU(cudaStreamSynchronize(s));
        const std::vector<float> yg = host(dout, D);
        double mx = 0, worst = 0;
        for (int i = 0; i < D; ++i) mx = std::max(mx, (double) std::fabs(yc[i]));
        for (int i = 0; i < D; ++i) worst = std::max(worst, std::fabs((double) yc[i] - (double) yg[i]) / mx);
        char what[96];
        std::snprintf(what, sizeof what, "q8k: %s/%s expert, CPU lane vs GPU (%s)", tn(pr.first), tn(pr.second),
                      on == nullptr || std::atoi(on) != 0 ? "Q8_K" : "per 32");
        std::printf("%-56s max diff %.3g of the largest output\n", what, worst);
        cudaFree(dpp), cudaFree(dw), cudaFree(dxs), cudaFree(dh), cudaFree(dsh), cudaFree(dout), cudaFree(Q.dev);
    }
}
void test_lanes(cudaStream_t s) {   // MXFP4 and every low-bit pair
    Experts M = make_experts(512, 256, 6);
    test_lane("MXFP4", M.lay, M.dev, M.host.data(), M.blob, 512, 256, s);
    cudaFree(M.dev);
    for (const auto& pr : kQPairs) {
        QExperts Q = make_q(512, 256, 6, pr.first, pr.second);
        char nm[64];
        std::snprintf(nm, sizeof nm, "%s/%s", tn(pr.first), tn(pr.second));
        test_lane(nm, Q.lay, Q.dev, Q.host.data(), Q.blob, 512, 256, s);
        cudaFree(Q.dev);
    }
}

// ---------------------------------------------------------------- Engram rows in Q6_K
__global__ void q6k_rows_kernel(const uint8_t* rows, int hd, float* out) {   // (engram_dequant's Q6_K branch)
    const int r = blockIdx.x;
    for (int e = threadIdx.x; e < hd; e += blockDim.x)
        out[(size_t) r * hd + e] =
            strata::ds4::q6k_value(rows + (size_t) r * hd + (size_t) (e / 256) * strata::ds4::kQ6kBytes, e % 256);
}
void test_engram_q6k(cudaStream_t s) {
    const int n = 64, hd = 512, sb = hd / 256;   // rows of two superblocks, each row hd bytes apart (as the engine keeps)
    std::vector<uint8_t> rows((size_t) n * hd, 0), packed((size_t) n * sb * 210);
    for (auto& v : packed) v = (uint8_t) (rng() & 0xff);
    for (size_t b = 0; b < (size_t) n * sb; ++b) {   // d: a small normal F16
        const __half h = __float2half(0.0005f + 0.002f * std::fabs(frand()));
        std::memcpy(&packed[b * 210 + 208], &h, 2);
    }
    for (int r = 0; r < n; ++r) std::memcpy(&rows[(size_t) r * hd], &packed[(size_t) r * sb * 210], (size_t) sb * 210);
    std::vector<float> want((size_t) n * hd);
    ggml_get_type_traits(GGML_TYPE_Q6_K)->to_float(packed.data(), want.data(), (int64_t) n * hd);
    uint8_t* d_rows = dev(rows);
    float* d_out = dev(std::vector<float>((size_t) n * hd));
    q6k_rows_kernel<<<n, 256, 0, s>>>(d_rows, hd, d_out);
    CU(cudaStreamSynchronize(s));
    const std::vector<float> got = host(d_out, (size_t) n * hd);
    const bool same = std::memcmp(got.data(), want.data(), got.size() * 4) == 0;
    if (!same) ++g_fail;
    std::printf("%-44s %s\n", "Engram Q6_K rows = ggml's, bit for bit", same ? "ok" : "FAIL");
    cudaFree(d_rows), cudaFree(d_out);
}

// ---------------------------------------------------------------- Engram rows in IQ4_NL, the Q5_K embedding
__global__ void iq4nl_rows_kernel(const uint8_t* rows, int hd, float* out) {   // (engram_dequant's IQ4_NL branch)
    const int r = blockIdx.x;
    for (int e = threadIdx.x; e < hd; e += blockDim.x)
        out[(size_t) r * hd + e] = strata::ds4::eg_block_value(rows + (size_t) r * hd, e, 2);
}
void test_engram_iq4nl(cudaStream_t s) {
    const int n = 64, hd = 512, nb = hd / 32;   // rows hd bytes apart (as the engine keeps them), 16 blocks of 18 bytes
    std::vector<uint8_t> rows((size_t) n * hd, 0), packed((size_t) n * nb * 18);
    for (auto& v : packed) v = (uint8_t) (rng() & 0xff);
    for (size_t b = 0; b < (size_t) n * nb; ++b) {
        const __half h = __float2half(0.0005f + 0.002f * std::fabs(frand()));
        std::memcpy(&packed[b * 18], &h, 2);
    }
    for (int r = 0; r < n; ++r) std::memcpy(&rows[(size_t) r * hd], &packed[(size_t) r * nb * 18], (size_t) nb * 18);
    std::vector<float> want((size_t) n * hd);
    ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(packed.data(), want.data(), (int64_t) n * hd);
    uint8_t* d_rows = dev(rows);
    float* d_out = dev(std::vector<float>((size_t) n * hd));
    iq4nl_rows_kernel<<<n, 256, 0, s>>>(d_rows, hd, d_out);
    CU(cudaStreamSynchronize(s));
    const std::vector<float> got = host(d_out, (size_t) n * hd);
    const bool same = std::memcmp(got.data(), want.data(), got.size() * 4) == 0;
    if (!same) ++g_fail;
    std::printf("%-52s %s\n", "Engram IQ4_NL rows = ggml's, bit for bit", same ? "ok" : "FAIL");
    cudaFree(d_rows), cudaFree(d_out);
}
void test_q5k_host() {
    const int nb = 512;
    std::vector<uint8_t> w = q_rand(gg::iq::kQ5_K, 1, nb * 256);
    std::vector<float> want((size_t) nb * 256), got((size_t) nb * 256);
    ggml_get_type_traits(GGML_TYPE_Q5_K)->to_float(w.data(), want.data(), (int64_t) nb * 256);
    for (int b = 0; b < nb; ++b) strata::ds4::q5k_values(w.data() + (size_t) b * 176, got.data() + (size_t) b * 256);
    const bool same = std::memcmp(got.data(), want.data(), got.size() * 4) == 0;
    if (!same) ++g_fail;
    std::printf("%-52s %s\n", "Q5_K embedding (host) = ggml's, bit for bit", same ? "ok" : "FAIL");
}

int main(int argc, char** argv) {
    cudaStream_t s;
    CU(cudaStreamCreate(&s));
    if (!gg::prepare(s)) return 1;
    if (argc >= 2 && std::strcmp(argv[1], "--real") == 0) {   // --real <ggml type> <k> <file> <offset> <rows> [label]
        if (argc < 7) return 2;
        const int t = std::atoi(argv[2]), k = std::atoi(argv[3]), n = std::atoi(argv[6]);
        const size_t rb = t == GGML_TYPE_IQ4_NL ? (size_t) k / 32 * 18 : gg::type_row_bytes(t, k);   // (IQ4_NL: Engram only)
        std::vector<uint8_t> w(rb * (size_t) n);
        std::FILE* f = std::fopen(argv[4], "rb");
        if (f == nullptr || fseeko(f, (off_t) std::atoll(argv[5]), SEEK_SET) != 0 || std::fread(w.data(), 1, w.size(), f) != w.size())
            return 3;
        std::fclose(f);
        if (t == GGML_TYPE_IQ4_NL || t == GGML_TYPE_Q5_K && argc >= 8 && std::strstr(argv[7], "embd")) {   // (host / Engram decoders)
            std::vector<float> want((size_t) n * k), got((size_t) n * k);
            ggml_get_type_traits((ggml_type) t)->to_float(w.data(), want.data(), (int64_t) n * k);
            if (t == GGML_TYPE_Q5_K) {
                for (size_t b = 0; b < (size_t) n * k / 256; ++b) strata::ds4::q5k_values(w.data() + b * 176, got.data() + b * 256);
            } else {
                std::vector<uint8_t> rows((size_t) n * k, 0);   // (each row k bytes apart, as the engine keeps them)
                for (int r = 0; r < n; ++r) std::memcpy(&rows[(size_t) r * k], &w[(size_t) r * k / 32 * 18], (size_t) k / 32 * 18);
                uint8_t* d_rows = dev(rows);
                float* d_out = dev(std::vector<float>((size_t) n * k));
                iq4nl_rows_kernel<<<n, 256, 0, s>>>(d_rows, k, d_out);
                CU(cudaStreamSynchronize(s));
                got = host(d_out, (size_t) n * k);
                cudaFree(d_rows), cudaFree(d_out);
            }
            const bool same = std::memcmp(got.data(), want.data(), got.size() * 4) == 0;
            std::printf("%-52s %s  %d values\n", argc >= 8 ? argv[7] : "real rows", same ? "ok  " : "FAIL", n * k);
            return same ? 0 : 1;
        }
        return check_rows(t, w, k, n, argc >= 8 ? argv[7] : "real rows", s) ? 0 : 1;
    }
    if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) {
        bench(s);
        bench_q(s);
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "--bench-q") == 0) {   // [a type name's part: those pairs' token experts]
        bench_q(s, argc >= 3 ? argv[2] : nullptr);
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "--bench-cpu") == 0) {   // [threads]
        bench_cpu(argc >= 3 ? std::atoi(argv[2]) : 1);
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "--q8k") == 0) {   // the low-bit activations' Q8_K rounding (DS4_IQ_Q8K)
        test_q8k(s);
        std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
        return g_fail ? 1 : 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "--lane") == 0) {   // the PCIe lane's streamed experts alone
        test_lanes(s);
        std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
        return g_fail ? 1 : 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "--kv") == 0) {   // the caches' packing alone
        test_kv_pack(s);
        std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
        return g_fail ? 1 : 0;
    }
    test_kv_pack(s);
    test_router_rows(s);
    test_mv(s);
    test_engram_q6k(s);
    test_engram_iq4nl(s);
    test_q5k_host();
    test_experts(512, 256, s);
    test_decode_values(s);
    test_f16_rows(s);
    for (int t : kQTypes) test_mv_q(t, s);
    for (const auto& pr : kQPairs) test_experts_q(512, 256, pr.first, pr.second, s);
    test_lanes(s);
    if (argc >= 3) {
        test_experts(std::atoi(argv[1]), std::atoi(argv[2]), s);
        for (int i : {0, 4, kQPairIQ1, kQPairIQ1 + 1})
            test_experts_q(std::atoi(argv[1]), std::atoi(argv[2]), kQPairs[i].first, kQPairs[i].second, s);
    }
    if (gg::launch_errors() != 0) {
        std::printf("launch errors: %d\n", gg::launch_errors());
        ++g_fail;
    }
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
