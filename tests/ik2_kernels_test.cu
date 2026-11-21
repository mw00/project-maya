// tests/ik2_kernels_test.cu - maya-ik2's kernels (src/ik2/ik2_kernels.cu) on random data, no model, each against a
// double reference written from ref/inkling2.py's ops:
//   * rms, the scaled L2 clip (with and without the full layers' log scaling), the short conv (static and dynamic taps,
//     the history carried over calls of 1, 2, 3 and many tokens = one call over all of them), the attention residual
//     (1 and 3 queries, 1 to 9 states);
//   * a head's attention weights: the relative bias out to the extent, the window, the sink, queries and keys at
//     offsets (a chunk of a prompt against the cache);
//   * the router (the picks as a set and the weights), the softcapped SwiGLU with the experts' scales, the combine;
//   * the decodes (NVFP4, Q8_0, Q5_K, Q6_K): the device's F32 values bit for bit the host's, FP16 = F32 rounded (NVFP4:
//     exact), NVFP4 against the format's definition.  (Q8_0 / Q5_K / Q6_K against ggml's values: not here yet - the
//     real model's seams cover them.)
#include "../src/ik2/ik2_iq_tables.h"
#include "../src/ik2/ik2_kernels.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = ik2::k;

namespace {

int g_fail = 0;
std::mt19937 rng(20261116);

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);      \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

float frand(float a = 1.0f) { return std::uniform_real_distribution<float>(-a, a)(rng); }
std::vector<float> fvec(size_t n, float a = 1.0f) {
    std::vector<float> v(n);
    for (auto& x : v) x = frand(a);
    return v;
}

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T>
T* dev_n(size_t n) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    CU(cudaMemset(p, 0, std::max<size_t>(1, n) * sizeof(T)));
    return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> h(n);
    CU(cudaDeviceSynchronize());
    CU(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// |got - want| <= atol + rtol |want| everywhere
void expect(const char* what, const std::vector<float>& got, const std::vector<double>& want, double rtol,
            double atol = 1e-6) {
    double worst = 0.0;
    size_t bad = 0, at = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double err = std::fabs((double) got[i] - want[i]), lim = atol + rtol * std::fabs(want[i]);
        if (!(err <= lim) || !std::isfinite(got[i])) {
            if (!bad) at = i;
            ++bad;
        }
        worst = std::max(worst, err / (atol + rtol * std::fabs(want[i])));
    }
    if (bad) {
        ++g_fail;
        std::printf("FAIL %-34s %zu of %zu off (first at %zu: got %g want %g)\n", what, bad, want.size(), at,
                    (double) got[at], want[at]);
    } else {
        std::printf("ok   %-34s %zu values, worst %.2f of the tolerance\n", what, want.size(), worst);
    }
}
void expect_true(const char* what, bool ok) {
    if (!ok) ++g_fail;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
}

float h2f_host(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    if (e == 0) return (sign ? -1.0f : 1.0f) * std::ldexp((float) m, -24);
    if (e == 31) return m ? NAN : (sign ? -INFINITY : INFINITY);
    return (sign ? -1.0f : 1.0f) * std::ldexp(1.0f + (float) m / 1024.0f, (int) e - 15);
}
uint16_t f2h_host(float f) { return (uint16_t) __half_as_ushort(__float2half_rn(f)); }

// ---------------------------------------------------------------------------------------------------------------

void test_rms() {
    const int rows = 5, D = 4608;
    auto x = fvec((size_t) rows * D, 3.0f), w = fvec(D);
    float *dx = dev(x), *dw = dev(w), *dy = dev_n<float>((size_t) rows * D);
    for (int with_w = 0; with_w < 2; ++with_w) {
        const float eps = with_w ? 1e-6f : 1e-5f;
        k::rms(dx, with_w ? dw : nullptr, rows, D, eps, dy, 0);
        std::vector<double> want((size_t) rows * D);
        for (int r = 0; r < rows; ++r) {
            double ss = 0;
            for (int d = 0; d < D; ++d) ss += (double) x[(size_t) r * D + d] * x[(size_t) r * D + d];
            const double rs = 1.0 / std::sqrt(ss / D + eps);
            for (int d = 0; d < D; ++d) want[(size_t) r * D + d] = x[(size_t) r * D + d] * rs * (with_w ? w[d] : 1.0);
        }
        expect(with_w ? "rms (weight)" : "rms (no weight)", host(dy, want.size()), want, 2e-5);
    }
    k::rms(dx, nullptr, rows * 36, 128, 1e-6f, dy, 0);   // v's rows: [T][Hkv][128]
    std::vector<double> want((size_t) rows * D);
    for (int r = 0; r < rows * 36; ++r) {
        double ss = 0;
        for (int d = 0; d < 128; ++d) ss += (double) x[(size_t) r * 128 + d] * x[(size_t) r * 128 + d];
        for (int d = 0; d < 128; ++d) want[(size_t) r * 128 + d] = x[(size_t) r * 128 + d] / std::sqrt(ss / 128 + 1e-6);
    }
    expect("rms (rows of 128)", host(dy, want.size()), want, 2e-5);
    cudaFree(dx), cudaFree(dw), cudaFree(dy);
}

void test_l2clip() {
    const int rows = 7, th_cases = 2;
    for (int c = 0; c < 3; ++c) {
        const int G = c == 2 ? 48 : c == 1 ? 8 : 48, D = c == 2 ? 16 : 128;
        const bool scaled = c != 1, tau = c == 0;
        auto x = fvec((size_t) rows * G * D, 4.0f), sc = fvec((size_t) G * D, 2.0f);
        for (size_t i = 0; i < (size_t) G * D; ++i) x[i] *= 0.01f;   // row 0: under the threshold
        std::vector<int> pos(rows);
        for (int r = 0; r < rows; ++r) pos[r] = r == 0 ? 5 : 100000 * r + 77777;   // some past the floor
        float *dx = dev(x), *dsc = dev(sc), *dy = dev_n<float>(x.size());
        int* dpos = dev(pos);
        const float th = c == 2 ? 8.0f : 22.627416610717773f, alpha = 0.1f, n_floor = 128000.0f;
        k::l2clip(dx, scaled ? dsc : nullptr, rows, G, D, th, tau ? dpos : nullptr, alpha, n_floor, dy, 0);
        std::vector<double> want(x.size());
        int clipped = 0;
        for (int r = 0; r < rows; ++r)
            for (int g = 0; g < G; ++g) {
                const size_t o = ((size_t) r * G + g) * D;
                double ss = 0;
                for (int d = 0; d < D; ++d) {
                    const double v = (double) x[o + d] * (scaled ? sc[(size_t) g * D + d] : 1.0);
                    ss += v * v;
                }
                double f = std::min(1.0, th / (std::sqrt(ss) + 1e-6));
                clipped += f < 1.0;
                if (tau) f *= 1.0 + alpha * std::log(std::max(1.0, (pos[r] + 1.0) / n_floor));
                for (int d = 0; d < D; ++d) want[o + d] = (double) x[o + d] * (scaled ? sc[(size_t) g * D + d] : 1.0) * f;
            }
        char name[64];
        std::snprintf(name, sizeof name, "l2clip G%d D%d%s%s (%d clipped)", G, D, scaled ? " scaled" : "",
                      tau ? " tau" : "", clipped);
        expect(name, host(dy, want.size()), want, 2e-5);
        cudaFree(dx), cudaFree(dsc), cudaFree(dy), cudaFree(dpos);
    }
    (void) th_cases;
}

// the reference conv over the whole sequence (ref/inkling2.py's conv)
std::vector<double> conv_ref(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>* dyn,
                             int heads, int T, int C) {
    std::vector<double> y((size_t) T * C);
    auto xp = [&](int r, int c) -> double { return r < 3 ? 0.0 : x[(size_t) (r - 3) * C + c]; };
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c) {
            double a = 0;
            for (int j = 0; j < 4; ++j) a += (double) w[(size_t) c * 4 + j] * xp(t + j, c);
            if (dyn)
                for (int lag = 0; lag < 4; ++lag)
                    a += (double) (*dyn)[(size_t) t * heads * 4 + (size_t) (c / (C / heads)) * 4 + lag] * xp(t + 3 - lag, c);
            y[(size_t) t * C + c] = a;
        }
    return y;
}

void test_conv() {
    const int T = 41, C = 1024, heads = 8;
    auto x = fvec((size_t) T * C), w = fvec((size_t) C * 4), dyn = fvec((size_t) T * heads * 4);
    float *dx = dev(x), *dw = dev(w), *dd = dev(dyn), *dy = dev_n<float>((size_t) T * C);
    for (int with_dyn = 0; with_dyn < 2; ++with_dyn) {
        const auto want = conv_ref(x, w, with_dyn ? &dyn : nullptr, heads, T, C);
        // one call
        float* hist = dev_n<float>((size_t) 3 * C);
        k::conv4(dx, hist, dw, with_dyn ? dd : nullptr, heads, T, C, dy, 0);
        expect(with_dyn ? "conv4 dynamic, one call" : "conv4 static, one call", host(dy, want.size()), want, 2e-5);
        // the same tokens in calls of 1, 2, 3, 5, the rest: the history carries
        CU(cudaMemset(hist, 0, (size_t) 3 * C * sizeof(float)));
        CU(cudaMemset(dy, 0, (size_t) T * C * sizeof(float)));
        const int parts[] = {1, 2, 3, 1, 5, 1, 1, T};
        int t0 = 0;
        for (int p : parts) {
            const int n = std::min(p, T - t0);
            if (n <= 0) break;
            k::conv4(dx + (size_t) t0 * C, hist, dw, with_dyn ? dd + (size_t) t0 * heads * 4 : nullptr, heads, n, C,
                     dy + (size_t) t0 * C, 0);
            t0 += n;
        }
        expect(with_dyn ? "conv4 dynamic, in pieces" : "conv4 static, in pieces", host(dy, want.size()), want, 2e-5);
        for (int tb : {16, 17}) {
            CU(cudaMemset(hist, 0, (size_t) 3 * C * sizeof(float)));
            CU(cudaMemset(dy, 0, (size_t) T * C * sizeof(float)));
            k::conv4(dx, hist, dw, with_dyn ? dd : nullptr, heads, tb, C, dy, 0);
            k::conv4(dx + (size_t) tb * C, hist, dw, with_dyn ? dd + (size_t) tb * heads * 4 : nullptr, heads, T - tb,
                     C, dy + (size_t) tb * C, 0);
            char name[64];
            std::snprintf(name, sizeof name, "conv4 %s, split at %d", with_dyn ? "dyn" : "static", tb);
            expect(name, host(dy, want.size()), want, 2e-5);
        }
        cudaFree(hist);
    }
    cudaFree(dx), cudaFree(dw), cudaFree(dd), cudaFree(dy);
}

void test_attn_res() {
    const int T = 9, C = 4608;
    for (int n : {1, 2, 9})
        for (int nq : {1, 3}) {
            std::vector<std::vector<float>> st(n);
            k::ResStates rs{};
            std::vector<float*> ds(n);
            for (int i = 0; i < n; ++i) st[i] = fvec((size_t) T * C, 1.0f + i), ds[i] = dev(st[i]), rs.st[i] = ds[i];
            auto q = fvec((size_t) nq * C, 3.0f);
            float *dq = dev(q), *dout = dev_n<float>((size_t) nq * T * C);
            k::attn_res(rs, n, dq, nq, T, C, 1e-6f, dout, 0);
            std::vector<double> want((size_t) nq * T * C);
            for (int t = 0; t < T; ++t) {
                std::vector<double> r(n);
                for (int i = 0; i < n; ++i) {
                    double ss = 0;
                    for (int d = 0; d < C; ++d) ss += (double) st[i][(size_t) t * C + d] * st[i][(size_t) t * C + d];
                    r[i] = 1.0 / std::sqrt(ss / C + 1e-6);
                }
                for (int qi = 0; qi < nq; ++qi) {
                    std::vector<double> sc(n);
                    double m = -1e300, sum = 0;
                    for (int i = 0; i < n; ++i) {
                        double dot = 0;
                        for (int d = 0; d < C; ++d) dot += (double) q[(size_t) qi * C + d] * st[i][(size_t) t * C + d] * r[i];
                        sc[i] = dot / std::sqrt((double) C);
                        m = std::max(m, sc[i]);
                    }
                    for (int i = 0; i < n; ++i) sum += (sc[i] = std::exp(sc[i] - m));
                    for (int d = 0; d < C; ++d) {
                        double a = 0;
                        for (int i = 0; i < n; ++i) a += sc[i] / sum * st[i][(size_t) t * C + d] * r[i];
                        want[((size_t) qi * T + t) * C + d] = a;
                    }
                }
            }
            char name[64];
            std::snprintf(name, sizeof name, "attn_res %d states, %d queries", n, nq);
            expect(name, host(dout, want.size()), want, 1e-4, 1e-6);
            for (auto* p : ds) cudaFree(p);
            cudaFree(dq), cudaFree(dout);
        }
}

void test_attn_probs() {
    struct Case {
        int Tq, nk, qpos0, kpos0, extent, window;
        const char* name;
    };
    const int d_rel = 16, H = 48;
    const Case cases[] = {
        {37, 37, 0, 0, 24, 1 << 30, "attn_probs full, from 0"},
        {20, 150, 130, 0, 64, 1 << 30, "attn_probs full, a chunk"},
        {33, 90, 200, 143, 32, 32, "attn_probs sliding window 32"},
        {1, 300, 299, 0, 1024, 1 << 30, "attn_probs one token"},
        {5, 2100, 3000, 905, 1024, 1024, "attn_probs sliding 1024"},
    };
    for (const Case& c : cases) {
        auto lg = fvec((size_t) c.Tq * c.nk, 40.0f);
        auto r = fvec((size_t) c.Tq * H * d_rel, 2.0f);        // [T][H][d_rel]; head 5's rows are used
        auto proj = fvec((size_t) c.extent * d_rel, 1.0f);
        auto sink = fvec((size_t) c.Tq * H, 3.0f);
        const int head = 5;
        const float scale = 1.0f / std::sqrt(128.0f);
        float *dl = dev(lg), *dr = dev(r), *dp = dev(proj), *dsk = dev(sink);
        k::attn_probs(dl, c.Tq, c.nk, c.qpos0, c.kpos0, scale, dr + head * d_rel, H * d_rel, d_rel, dp, c.extent,
                      c.window, dsk + head, H, 0);
        std::vector<double> want((size_t) c.Tq * c.nk);
        for (int i = 0; i < c.Tq; ++i) {
            std::vector<double> l(c.nk + 1);
            double m = -1e300;
            for (int j = 0; j < c.nk; ++j) {
                const long dist = (long) (c.qpos0 + i) - (c.kpos0 + j);
                if (dist < 0 || dist >= c.window) {
                    l[j] = -INFINITY;
                    continue;
                }
                double v = (double) lg[(size_t) i * c.nk + j] * scale;
                if (dist < c.extent)
                    for (int e = 0; e < d_rel; ++e)
                        v += (double) r[((size_t) i * H + head) * d_rel + e] * proj[(size_t) dist * d_rel + e];
                l[j] = v;
                m = std::max(m, v);
            }
            l[c.nk] = sink[(size_t) i * H + head];
            m = std::max(m, l[c.nk]);
            double sum = 0;
            for (int j = 0; j <= c.nk; ++j) sum += std::exp(l[j] - m);
            for (int j = 0; j < c.nk; ++j) want[(size_t) i * c.nk + j] = std::exp(l[j] - m) / sum;
        }
        expect(c.name, host(dl, want.size()), want, 2e-4, 1e-9);
        cudaFree(dl), cudaFree(dr), cudaFree(dp), cudaFree(dsk);
    }

    // Batched attn_probs_f16: [H][Tq][nk] F32 logits + [Tq][H][d_rel] r + [Tq][H] sink -> [H][Tq][nk] FP16
    for (const Case& c : cases) {
        auto lg = fvec((size_t) H * c.Tq * c.nk, 40.0f);
        auto r = fvec((size_t) c.Tq * H * d_rel, 2.0f);
        auto proj = fvec((size_t) c.extent * d_rel, 1.0f);
        auto sink = fvec((size_t) c.Tq * H, 3.0f);
        const float scale = 1.0f / std::sqrt(128.0f);
        float *dl = dev(lg), *dr = dev(r), *dp = dev(proj), *dsk = dev(sink);
        uint16_t* dout = dev_n<uint16_t>((size_t) H * c.Tq * c.nk);
        k::attn_probs_f16(dl, dout, H, c.Tq, c.nk, c.qpos0, c.kpos0, scale, dr, d_rel, dp, c.extent, c.window, dsk, 0);
        const auto raw = host(dout, (size_t) H * c.Tq * c.nk);
        std::vector<float> got(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) got[i] = h2f_host(raw[i]);
        std::vector<double> want(raw.size(), 0.0);
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < c.Tq; ++i) {
                std::vector<double> l(c.nk + 1);
                double m = -1e300;
                for (int j = 0; j < c.nk; ++j) {
                    const long dist = (long) (c.qpos0 + i) - (c.kpos0 + j);
                    if (dist < 0 || dist >= c.window) {
                        l[j] = -INFINITY;
                        continue;
                    }
                    double v = (double) lg[((size_t) h * c.Tq + i) * c.nk + j] * scale;
                    if (dist < c.extent)
                        for (int e = 0; e < d_rel; ++e)
                            v += (double) r[((size_t) i * H + h) * d_rel + e] * proj[(size_t) dist * d_rel + e];
                    l[j] = v;
                    m = std::max(m, v);
                }
                l[c.nk] = sink[(size_t) i * H + h];
                m = std::max(m, l[c.nk]);
                double sum = 0;
                for (int j = 0; j <= c.nk; ++j) sum += std::exp(l[j] - m);
                for (int j = 0; j < c.nk; ++j) want[((size_t) h * c.Tq + i) * c.nk + j] = std::exp(l[j] - m) / sum;
            }
        }
        char nm[64];
        std::snprintf(nm, sizeof nm, "%s (f16)", c.name);
        expect(nm, got, want, 2e-3, 1e-4);
        // the same with the bias as a table [Tq * H][extent] (the prefill's path)
        std::vector<float> bt((size_t) c.Tq * H * c.extent);
        for (int i = 0; i < c.Tq; ++i)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < c.extent; ++d) {
                    double b = 0;
                    for (int e = 0; e < d_rel; ++e)
                        b += (double) r[((size_t) i * H + h) * d_rel + e] * proj[(size_t) d * d_rel + e];
                    bt[((size_t) i * H + h) * c.extent + d] = (float) b;
                }
        float* dbt = dev(bt);
        k::attn_probs_bias_f16(dl, dout, H, c.Tq, c.nk, c.qpos0, c.kpos0, scale, dbt, c.extent, c.window, dsk, 0);
        const auto raw2 = host(dout, (size_t) H * c.Tq * c.nk);
        for (size_t i = 0; i < raw2.size(); ++i) got[i] = h2f_host(raw2[i]);
        std::snprintf(nm, sizeof nm, "%s (f16, bias table)", c.name);
        expect(nm, got, want, 2e-3, 1e-4);
        cudaFree(dl), cudaFree(dr), cudaFree(dp), cudaFree(dsk), cudaFree(dout), cudaFree(dbt);
    }
}

void test_route() {
    const int T = 300, E = 384, K = 8;
    auto lg = fvec((size_t) T * E, 6.0f), bias = fvec(E, 0.2f);
    float *dl = dev(lg), *db = dev(bias), *dw = dev_n<float>((size_t) T * K);
    int* dt = dev_n<int>((size_t) T * K);
    k::route(dl, db, T, E, K, 8.0f, dt, dw, 0);
    const auto top = host(dt, (size_t) T * K);
    const auto w = host(dw, (size_t) T * K);
    bool picks_ok = true;
    std::vector<double> want_w((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        std::vector<std::pair<double, int>> v(E);
        for (int e = 0; e < E; ++e) v[e] = {-(1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + e])) + bias[e]), e};
        std::sort(v.begin(), v.end());
        std::vector<int> pick(K);
        for (int i = 0; i < K; ++i) pick[i] = v[i].second;
        std::sort(pick.begin(), pick.end());
        double sum = 0;
        for (int i = 0; i < K; ++i) sum += 1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + pick[i]]));
        for (int i = 0; i < K; ++i) {
            picks_ok &= top[(size_t) t * K + i] == pick[i];
            want_w[(size_t) t * K + i] = 1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + pick[i]])) / (sum + 1e-20) * 8.0;
        }
    }
    expect_true("route: the picks", picks_ok);
    if (picks_ok) expect("route: the weights", w, want_w, 2e-5);
    // route_token: a token's route and route_gather in one launch, the same values to the bit (ties among them:
    // the logits of the first rows repeated)
    {
        std::vector<const uint8_t*> tab(E);
        for (int e = 0; e < E; ++e) tab[e] = (const uint8_t*) (uintptr_t) (0x1000 + 64 * e);
        auto sgt = fvec(E, 2.0f), sut = fvec(E, 2.0f), sdt = fvec(E, 2.0f);
        const uint8_t** dtab = dev(tab);
        float *dsg = dev(sgt), *dsu = dev(sut), *dsd = dev(sdt);
        bool same = true;
        for (int t = 0; t < 40; ++t) {
            std::vector<float> row(lg.begin() + (size_t) t * E, lg.begin() + (size_t) (t + 1) * E);
            if (t >= 30)   // ties: a handful of equal logits and biases among the largest
                for (int e = 0; e < E; e += 7) row[e] = 5.5f;
            std::vector<float> b2 = bias;
            if (t >= 30)
                for (int e = 0; e < E; e += 7) b2[e] = 0.1f;
            float *dr = dev(row), *db2 = dev(b2);
            int *t1 = dev_n<int>(K), *t2 = dev_n<int>(K), *k1 = dev_n<int>(K), *k2 = dev_n<int>(K), *i1 = dev_n<int>(K),
                *i2 = dev_n<int>(K);
            float *w1 = dev_n<float>(K), *w2 = dev_n<float>(K), *g1 = dev_n<float>(K), *g2 = dev_n<float>(K),
                  *u1 = dev_n<float>(K), *u2 = dev_n<float>(K), *x1 = dev_n<float>(K), *x2 = dev_n<float>(K);
            const uint8_t **b1 = dev_n<const uint8_t*>(K), **bb2 = dev_n<const uint8_t*>(K);
            k::route(dr, db2, 1, E, K, 8.0f, t1, w1, 0);
            k::route_gather(t1, w1, dtab, dsg, dsu, dsd, K, K, b1, k1, i1, g1, u1, x1, 0);
            k::route_token(dr, db2, E, K, 8.0f, dtab, dsg, dsu, dsd, t2, w2, bb2, k2, i2, g2, u2, x2, 0);
            same &= host(t1, K) == host(t2, K) && host(k1, K) == host(k2, K) && host(i1, K) == host(i2, K);
            same &= host(b1, K) == host(bb2, K);
            const auto a1 = host(w1, K), a2 = host(w2, K), c1 = host(g1, K), c2 = host(g2, K), d1 = host(u1, K),
                       d2 = host(u2, K), e1 = host(x1, K), e2 = host(x2, K);
            same &= !std::memcmp(a1.data(), a2.data(), K * 4) && !std::memcmp(c1.data(), c2.data(), K * 4) &&
                    !std::memcmp(d1.data(), d2.data(), K * 4) && !std::memcmp(e1.data(), e2.data(), K * 4);
            for (int* q : {t1, t2, k1, k2, i1, i2}) cudaFree(q);
            for (float* q : {w1, w2, g1, g2, u1, u2, x1, x2, dr, db2}) cudaFree(q);
            cudaFree(b1), cudaFree(bb2);
        }
        expect_true("route_token: route and route_gather's values to the bit (40 tokens, ties among them)", same);
        cudaFree(dtab), cudaFree(dsg), cudaFree(dsu), cudaFree(dsd);
    }
    cudaFree(dl), cudaFree(db), cudaFree(dw), cudaFree(dt);
}

void test_moe_pieces() {
    const int n = 37, F = 2304, C = 4608, T = 11, K = 8;
    // swiglu_cap over FP16 gate | up rows
    std::vector<uint16_t> gu((size_t) n * 2 * F);
    std::vector<float> guf(gu.size());
    {
        auto v = fvec(gu.size(), 12.0f);
        float* dv = dev(v);
        std::vector<int> idx(n * 2 * F / C);
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
        int* di = dev(idx);
        uint16_t* dg = dev_n<uint16_t>(gu.size());
        k::gather_f16(dv, di, (int) idx.size(), C, dg, 0);   // (also gather_f16 with the identity)
        gu = host(dg, gu.size());
        bool ok = true;
        for (size_t i = 0; i < gu.size(); ++i) {
            guf[i] = h2f_host(gu[i]);
            ok &= std::fabs(guf[i] - v[i]) <= 6e-4f * std::fabs(v[i]) + 1e-7f;
        }
        expect_true("gather_f16: FP16 of the rows", ok);
        cudaFree(dv), cudaFree(di), cudaFree(dg);
    }
    auto sg = fvec(n, 2.0f), su = fvec(n, 3.0f);
    uint16_t *dgu = dev(gu), *dh = dev_n<uint16_t>((size_t) n * F);
    float *dsg = dev(sg), *dsu = dev(su);
    k::swiglu_cap(dgu, n, F, dsg, dsu, 4.0f, 25.0f, dh, 0);
    const auto h = host(dh, (size_t) n * F);
    std::vector<float> hf(h.size());
    std::vector<double> want(h.size());
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < F; ++c) {
            const double g = (double) guf[(size_t) r * 2 * F + c] * sg[r], u = (double) guf[(size_t) r * 2 * F + F + c] * su[r];
            want[(size_t) r * F + c] = 4.0 * std::tanh(g / 4.0) / (1.0 + std::exp(-g)) * 25.0 * std::tanh(u / 25.0);
            hf[(size_t) r * F + c] = h2f_host(h[(size_t) r * F + c]);
        }
    expect("swiglu_cap (FP16 out)", hf, want, 6e-4, 1e-6);
    cudaFree(dgu), cudaFree(dh), cudaFree(dsg), cudaFree(dsu);

    // the combine: token t's K slots are rows inv[t][i] of y
    const int rows = T * K;
    std::vector<uint16_t> y((size_t) rows * C);
    std::vector<float> yf(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = (uint16_t) (rng() & 0x7BFF) | (uint16_t) ((rng() & 1) << 15);   // any finite FP16
        if (((y[i] >> 10) & 31) > 20) y[i] &= ~(uint16_t) (8 << 10);          // (keep the sums small)
        yf[i] = h2f_host(y[i]);
    }
    std::vector<int> inv(rows);
    for (int i = 0; i < rows; ++i) inv[i] = i;
    std::shuffle(inv.begin(), inv.end(), rng);
    auto wt = fvec(rows, 2.0f), base = fvec((size_t) T * C, 5.0f);
    uint16_t* dy = dev(y);
    int* dinv = dev(inv);
    float *dwt = dev(wt), *dout = dev(base);
    for (int add = 1; add >= 0; --add) {
        k::moe_combine(dy, dinv, dwt, T, K, C, dout, add, 0);
        std::vector<double> w2((size_t) T * C);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) {
                double a = add ? base[(size_t) t * C + c] : 0.0;
                for (int i = 0; i < K; ++i) a += (double) wt[(size_t) t * K + i] * yf[(size_t) inv[(size_t) t * K + i] * C + c];
                w2[(size_t) t * C + c] = a;
            }
        expect(add ? "moe_combine (add)" : "moe_combine (overwrite)", host(dout, w2.size()), w2, 2e-5, 2e-4);
    }
    cudaFree(dy), cudaFree(dinv), cudaFree(dwt), cudaFree(dout);

    auto lgt = fvec(50000, 90.0f);
    float* dlg = dev(lgt);
    k::softcap(dlg, lgt.size(), 30.0f, 0);
    std::vector<double> w3(lgt.size());
    for (size_t i = 0; i < lgt.size(); ++i) w3[i] = 30.0 * std::tanh((double) lgt[i] / 30.0);
    expect("softcap", host(dlg, lgt.size()), w3, 2e-5);
    auto o = fvec(4096, 3.0f), g = fvec(4096, 8.0f);
    float *dob = dev(o), *dgb = dev(g);
    k::gate_mul(dob, dgb, o.size(), 0);
    std::vector<double> w4(o.size());
    for (size_t i = 0; i < o.size(); ++i) w4[i] = (double) o[i] / (1.0 + std::exp(-(double) g[i]));
    expect("gate_mul", host(dob, o.size()), w4, 2e-5);
    cudaFree(dlg), cudaFree(dob), cudaFree(dgb);
}

// NVFP4 by the format's definition (ggml's dequantize_row_nvfp4): 64 values in 36 bytes - four UE4M3 scales, then for
// each 16 values 8 bytes, value j < 8 the low nibble of byte j and value j + 8 its high one
double nvfp4_def(const uint8_t* blocks, size_t v) {
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const uint8_t* b = blocks + (v / 64) * 36;
    const int i = (int) (v % 64), sb = i / 16, j = i % 16;
    const uint8_t x = b[sb];
    double d = 0.0;
    if (x != 0 && x != 0x7F) {
        const int e = (x >> 3) & 0xF, m = x & 7;
        d = (e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m / 8.0, e - 7)) * 0.5;
    }
    const uint8_t byte = b[4 + sb * 8 + j % 8];
    return kv[j < 8 ? (byte & 0xF) : (byte >> 4)] * d;
}

void test_decode() {
    const int types[] = {k::kNVFP4, k::kQ8_0, k::kQ5_K, k::kQ6_K, k::kIQ3_XXS, k::kIQ4_XS};
    const char* names[] = {"NVFP4", "Q8_0", "Q5_K", "Q6_K", "IQ3_XXS", "IQ4_XS"};
    for (int ti = 0; ti < 6; ++ti) {
        const int type = types[ti], bv = k::block_values(type);
        const size_t bb = k::block_bytes(type), nblk = 777, n = nblk * bv;
        std::vector<uint8_t> raw(nblk * bb);
        for (auto& b : raw) b = (uint8_t) rng();
        // the F16 scales: finite and moderate (random bits would give inf / nan scales)
        for (size_t b = 0; b < nblk; ++b) {
            auto fix = [&](size_t off) {
                uint8_t* p = raw.data() + b * bb + off;
                uint16_t h = (uint16_t) (p[0] | (p[1] << 8));
                h = (uint16_t) ((h & 0x83FF) | ((uint16_t) (8 + (h >> 10) % 8) << 10));
                p[0] = (uint8_t) h, p[1] = (uint8_t) (h >> 8);
            };
            if (type == k::kQ8_0) fix(0);
            if (type == k::kQ5_K) fix(0), fix(2);
            if (type == k::kQ6_K) fix(208);
        if (type == k::kIQ3_XXS || type == k::kIQ4_XS) fix(0);
        }
        std::vector<float> hostv(n);
        k::decode_host(type, raw.data(), n, hostv.data());
        uint8_t* dr = dev(raw);
        float* d32 = dev_n<float>(n);
        uint16_t* d16 = dev_n<uint16_t>(n);
        k::decode_f32(type, dr, n, d32, 0);
        k::decode_f16(type, dr, n, d16, 0);
        const auto g32 = host(d32, n);
        const auto g16 = host(d16, n);
        bool same = std::memcmp(g32.data(), hostv.data(), n * sizeof(float)) == 0;
        char name[80];
        std::snprintf(name, sizeof name, "decode %s: device F32 = host, bit for bit", names[ti]);
        expect_true(name, same);
        bool f16_ok = true, exact = true;
        for (size_t i = 0; i < n; ++i) {
            const float back = h2f_host(g16[i]);
            f16_ok &= std::fabs(back - hostv[i]) <= 4.9e-4f * std::fabs(hostv[i]) + 6e-8f || !std::isfinite(back);
            exact &= back == hostv[i];
        }
        std::snprintf(name, sizeof name, "decode %s: FP16 = the value rounded%s", names[ti],
                      type == k::kNVFP4 ? " (exact)" : "");
        expect_true(name, f16_ok && (type != k::kNVFP4 || exact));
        if (type == k::kNVFP4) {
            bool def = true;
            for (size_t i = 0; i < n; ++i) def &= (double) hostv[i] == nvfp4_def(raw.data(), i);
            expect_true("decode NVFP4: the format's definition", def);
        }
        cudaFree(dr), cudaFree(d32), cudaFree(d16);
    }
}

// random blocks of `type` whose F16 scales are finite and moderate
std::vector<uint8_t> rand_blocks(int type, size_t n_values) {
    const size_t bb = k::block_bytes(type), nb = n_values / k::block_values(type);
    std::vector<uint8_t> raw(nb * bb);
    for (auto& b : raw) b = (uint8_t) rng();
    for (size_t b = 0; b < nb; ++b) {
        auto fix = [&](size_t off) {
            uint8_t* p = raw.data() + b * bb + off;
            uint16_t h = (uint16_t) (p[0] | (p[1] << 8));
            h = (uint16_t) ((h & 0x83FF) | ((uint16_t) (8 + (h >> 10) % 8) << 10));
            p[0] = (uint8_t) h, p[1] = (uint8_t) (h >> 8);
        };
        if (type == k::kQ8_0) fix(0);
        if (type == k::kQ5_K) fix(0), fix(2);
        if (type == k::kQ6_K) fix(208);
        if (type == k::kIQ3_XXS || type == k::kIQ4_XS) fix(0);
    }
    return raw;
}

// ggml's dequantize_row_iq3_xxs / dequantize_row_iq4_xs, transcribed (ksigns_iq2xs[i] = i | parity(i) << 7)
// (ggml's tables themselves: ik2_iq_tables.h reads them from third_party/ggml/ggml-common.h)
static const uint32_t* const g_iq3xxs_grid = ik2_iq::host::iq3xxs_grid;
static const int8_t* const g_iq4nl = ik2_iq::host::kvalues_iq4nl;
void ggml_iq3_xxs(const uint8_t* x, float* y, size_t k) {
    for (size_t i = 0; i < k / 256; ++i) {
        const uint8_t* blk = x + i * 98;
        const float d = h2f_host((uint16_t) (blk[0] | (blk[1] << 8)));
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, 4);
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint32_t s7 = (aux32 >> 7 * l) & 127;
                uint32_t par = s7 ^ (s7 >> 4);   // (its parity, folded: portable)
                par ^= par >> 2;
                par ^= par >> 1;
                const uint8_t signs = (uint8_t) (s7 | (par & 1) << 7);
                const uint8_t* grid1 = (const uint8_t*) (g_iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*) (g_iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & (1 << j) ? -1.f : 1.f);
                    y[j + 4] = db * grid2[j] * (signs & (1 << (j + 4)) ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}
void ggml_iq4_xs(const uint8_t* x, float* y, size_t k) {
    for (size_t i = 0; i < k / 256; ++i) {
        const uint8_t* blk = x + i * 136;
        const float d = h2f_host((uint16_t) (blk[0] | (blk[1] << 8)));
        const uint16_t scales_h = (uint16_t) (blk[2] | (blk[3] << 8));
        const uint8_t* scales_l = blk + 4;
        const uint8_t* qs = blk + 8;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * g_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * g_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}
void test_iq_ref() {
    for (int type : {k::kIQ3_XXS, k::kIQ4_XS}) {
        const size_t n = 256 * 333;
        auto raw = rand_blocks(type, n);
        std::vector<float> want(n), got(n);
        (type == k::kIQ3_XXS ? ggml_iq3_xxs : ggml_iq4_xs)(raw.data(), want.data(), n);
        k::decode_host(type, raw.data(), n, got.data());
        expect(type == k::kIQ3_XXS ? "decode_host IQ3_XXS = ggml" : "decode_host IQ4_XS = ggml", got,
               std::vector<double>(want.begin(), want.end()), 0.0, 0.0);
    }
}

void test_experts_mv() {
    const int C = 512, F = 256, S = 5;
    struct Case {
        int tg, tu, td;
        const char* name;
    };
    const Case cases[] = {{k::kNVFP4, k::kNVFP4, k::kNVFP4, "NVFP4"},
                          {k::kQ5_K, k::kQ5_K, k::kQ6_K, "Q5_K / Q6_K"},
                          {k::kQ8_0, k::kNVFP4, k::kQ6_K, "mixed"},
                          {k::kIQ3_XXS, k::kIQ3_XXS, k::kNVFP4, "IQ3_XXS / NVFP4"},
                          {k::kIQ4_XS, k::kIQ4_XS, k::kQ6_K, "IQ4_XS / Q6_K"}};
    for (const Case& cs : cases) {
        const int E = 3;
        auto g = rand_blocks(cs.tg, (size_t) E * F * C), u = rand_blocks(cs.tu, (size_t) E * F * C),
             d = rand_blocks(cs.td, (size_t) E * C * F);
        const size_t eb0 = g.size() / E, eb1 = u.size() / E, eb2 = d.size() / E;
        // each expert's blob: gate | up | down, 256-byte aligned slots
        const size_t slot = (eb0 + eb1 + eb2 + 255) / 256 * 256;
        std::vector<uint8_t> pool(slot * E);
        for (int e = 0; e < E; ++e) {
            std::memcpy(pool.data() + e * slot, g.data() + e * eb0, eb0);
            std::memcpy(pool.data() + e * slot + eb0, u.data() + e * eb1, eb1);
            std::memcpy(pool.data() + e * slot + eb0 + eb1, d.data() + e * eb2, eb2);
        }
        uint8_t* dpool = dev(pool);
        const int pick[S] = {2, 0, 2, 1, 0}, tok[S] = {0, 1, 1, 2, 0};
        std::vector<const uint8_t*> blob(S);
        for (int i = 0; i < S; ++i) blob[i] = dpool + pick[i] * slot;
        const uint8_t** dblob = nullptr;
        CU(cudaMalloc(&dblob, S * sizeof(void*)));
        CU(cudaMemcpy(dblob, blob.data(), S * sizeof(void*), cudaMemcpyHostToDevice));
        auto x = fvec((size_t) 3 * C, 2.0f);
        float *dx = dev(x), *dgu = dev_n<float>((size_t) S * 2 * F), *dy = dev_n<float>((size_t) S * C);
        int* dtok = dev(std::vector<int>(tok, tok + S));
        k::experts_gate_up(dblob, dtok, S, cs.tg, cs.tu, eb0, dx, C, F, dgu, 0);
        std::vector<uint16_t> h((size_t) S * F);
        std::vector<float> hf(h.size());
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] = (uint16_t) ((rng() & 0x3BFF) | ((rng() & 1) << 15));
            hf[i] = h2f_host(h[i]);
        }
        uint16_t* dh = dev(h);
        k::experts_down(dblob, S, cs.td, eb0 + eb1, dh, C, F, dy, 0);
        std::vector<double> want_gu((size_t) S * 2 * F), want_y((size_t) S * C);
        std::vector<float> row(C > F ? C : F);
        for (int i = 0; i < S; ++i) {
            const int e = pick[i];
            for (int r = 0; r < 2 * F; ++r) {
                const bool up = r >= F;
                const int rr = up ? r - F : r;
                const int ty = up ? cs.tu : cs.tg;
                const uint8_t* w = (up ? u.data() + e * eb1 : g.data() + e * eb0) + (size_t) rr * (eb0 / F);
                if (up) w = u.data() + e * eb1 + (size_t) rr * (eb1 / F);
                k::decode_host(ty, w, C, row.data());
                double a = 0;
                for (int c = 0; c < C; ++c) a += (double) row[c] * x[(size_t) tok[i] * C + c];
                want_gu[(size_t) i * 2 * F + r] = a;
            }
            for (int r = 0; r < C; ++r) {
                k::decode_host(cs.td, d.data() + e * eb2 + (size_t) r * (eb2 / C), F, row.data());
                double a = 0;
                for (int c = 0; c < F; ++c) a += (double) row[c] * hf[(size_t) i * F + c];
                want_y[(size_t) i * C + r] = a;
            }
        }
        char name[64];
        std::snprintf(name, sizeof name, "experts_gate_up %s", cs.name);
        expect(name, host(dgu, want_gu.size()), want_gu, 2e-4, 2e-3);
        std::snprintf(name, sizeof name, "experts_down %s", cs.name);
        expect(name, host(dy, want_y.size()), want_y, 2e-4, 2e-3);
        // S = 1 path (<4, 2> launch config)
        k::experts_gate_up(dblob, dtok, 1, cs.tg, cs.tu, eb0, dx, C, F, dgu, 0);
        k::experts_down(dblob, 1, cs.td, eb0 + eb1, dh, C, F, dy, 0);
        std::snprintf(name, sizeof name, "experts_gate_up S=1 %s", cs.name);
        expect(name, host(dgu, 2 * F), std::vector<double>(want_gu.begin(), want_gu.begin() + 2 * F), 2e-4, 2e-3);
        std::snprintf(name, sizeof name, "experts_down S=1 %s", cs.name);
        expect(name, host(dy, C), std::vector<double>(want_y.begin(), want_y.begin() + C), 2e-4, 2e-3);
        cudaFree(dpool), cudaFree(dblob), cudaFree(dx), cudaFree(dgu), cudaFree(dy), cudaFree(dtok), cudaFree(dh);
    }
}

// the grouped kernels against the per-slot ones: 8 tokens, variable group sizes (1, 2, 3, 5, 8) over all quant types
void test_experts_grouped() {
    const int C = 512, F = 256, E = 5, T = 8, K = 8, S = T * K;
    const int types[][3] = {{k::kNVFP4, k::kNVFP4, k::kNVFP4},
                            {k::kQ5_K, k::kQ5_K, k::kQ6_K},
                            {k::kQ8_0, k::kQ8_0, k::kQ8_0},
                            {k::kQ8_0, k::kNVFP4, k::kQ6_K},
                            {k::kIQ3_XXS, k::kIQ3_XXS, k::kNVFP4},
                            {k::kIQ4_XS, k::kIQ4_XS, k::kQ6_K}};
    const char* tnames[] = {"NVFP4", "Q5_K/Q6_K", "Q8_0", "Q8_0/NVFP4/Q6_K", "IQ3_XXS/NVFP4", "IQ4_XS/Q6_K"};
    for (int ti = 0; ti < 6; ++ti) {
        const auto& ty = types[ti];
        auto g = rand_blocks(ty[0], (size_t) E * F * C), u = rand_blocks(ty[1], (size_t) E * F * C),
             d = rand_blocks(ty[2], (size_t) E * C * F);
        const size_t eb0 = g.size() / E, eb1 = u.size() / E, eb2 = d.size() / E, slot = (eb0 + eb1 + eb2 + 255) / 256 * 256;
        std::vector<uint8_t> pool(slot * E);
        for (int e = 0; e < E; ++e) {
            std::memcpy(pool.data() + e * slot, g.data() + e * eb0, eb0);
            std::memcpy(pool.data() + e * slot + eb0, u.data() + e * eb1, eb1);
            std::memcpy(pool.data() + e * slot + eb0 + eb1, d.data() + e * eb2, eb2);
        }
        uint8_t* dpool = dev(pool);
        // Assign experts so group sizes hit 1, 2, 3, 5, 8 across the 5 experts (distinct within each token)
        const int want_ng[E] = {1, 2, 3, 5, 8};
        std::vector<const uint8_t*> blob(S);
        std::vector<int> tok(S), gn, gtok, grow;
        std::vector<const uint8_t*> gb;
        std::vector<int> pick(S, -1);
        for (int e = 0; e < E; ++e)
            for (int t = 0; t < want_ng[e]; ++t) pick[t * K + e] = e;
        std::vector<int> idx;   // the slots in use
        for (int i = 0; i < S; ++i)
            if (pick[i] >= 0) idx.push_back(i);
        const int n = (int) idx.size();
        std::vector<int> gi(E, -1);
        for (int r = 0; r < n; ++r) {
            const int i = idx[r], e = pick[i];
            blob[r] = dpool + e * slot;
            tok[r] = i / K;
            if (gi[e] < 0) {
                gi[e] = (int) gb.size();
                gb.push_back(dpool + e * slot);
                gn.push_back(0);
                gtok.resize(gtok.size() + k::kGroupMax);
                grow.resize(grow.size() + k::kGroupMax);
            }
            const int gg = gi[e];
            gtok[gg * k::kGroupMax + gn[gg]] = i / K;
            grow[gg * k::kGroupMax + gn[gg]] = r;
            ++gn[gg];
        }
        const uint8_t** dblob = nullptr;
        const uint8_t** dgb = nullptr;
        CU(cudaMalloc(&dblob, n * sizeof(void*)));
        CU(cudaMalloc(&dgb, gb.size() * sizeof(void*)));
        CU(cudaMemcpy(dblob, blob.data(), n * sizeof(void*), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dgb, gb.data(), gb.size() * sizeof(void*), cudaMemcpyHostToDevice));
        int *dtok = dev(tok), *dgn = dev(gn), *dgt = dev(gtok), *dgr = dev(grow);
        auto x = fvec((size_t) T * C, 2.0f);
        std::vector<uint16_t> h((size_t) n * F);
        for (auto& v : h) v = (uint16_t) ((rng() & 0x3BFF) | ((rng() & 1) << 15));
        float *dx = dev(x), *gu1 = dev_n<float>((size_t) n * 2 * F), *gu2 = dev_n<float>((size_t) n * 2 * F);
        float *y1 = dev_n<float>((size_t) n * C), *y2 = dev_n<float>((size_t) n * C);
        uint16_t* dh = dev(h);
        k::experts_gate_up(dblob, dtok, n, ty[0], ty[1], eb0, dx, C, F, gu1, 0);
        k::experts_gate_up_g(dgb, dgn, dgt, dgr, (int) gb.size(), ty[0], ty[1], eb0, dx, C, F, gu2, 0);
        k::experts_down(dblob, n, ty[2], eb0 + eb1, dh, C, F, y1, 0);
        k::experts_down_g(dgb, dgn, dgr, (int) gb.size(), ty[2], eb0 + eb1, dh, C, F, y2, 0);
        const auto a1 = host(gu1, (size_t) n * 2 * F), b1 = host(y1, (size_t) n * C);
        std::vector<double> wa(a1.begin(), a1.end()), wb(b1.begin(), b1.end());
        char name[80];
        std::snprintf(name, sizeof name, "experts_gate_up_g (%zu groups) %s", gb.size(), tnames[ti]);
        // (two F32 orders of the same sums: random NVFP4 blocks make terms of 1e3 - 1e4 that cancel to a few units,
        //  so the absolute part of the tolerance follows the terms, not the result)
        expect(name, host(gu2, wa.size()), wa, 1e-4, 2e-2);
        std::snprintf(name, sizeof name, "experts_down_g %s", tnames[ti]);
        // (the same two orders over F terms: 1 of 9728 sums went 2e-3 off at a value of 6.5, IQ4_XS / Q6_K on H200)
        expect(name, host(y2, wb.size()), wb, 1e-4, 5e-3);
        for (void* q : {(void*) dpool, (void*) dblob, (void*) dgb, (void*) dtok, (void*) dgn, (void*) dgt, (void*) dgr,
                        (void*) dx, (void*) gu1, (void*) gu2, (void*) y1, (void*) y2, (void*) dh})
            cudaFree(q);
    }
}

// several queries against attn_decode a query at a time
void test_attn_multi() {
    const int H = 48, KV = 8, D = 128, d_rel = 16, ctx = 3100, nq = 8;
    auto kc = fvec((size_t) ctx * KV * D), vc = fvec((size_t) ctx * KV * D);
    auto q = fvec((size_t) nq * H * D, 3.0f), r = fvec((size_t) nq * H * d_rel, 2.0f), sink = fvec((size_t) nq * H, 3.0f),
         proj = fvec((size_t) 1024 * d_rel);
    float *dk = dev(kc), *dv = dev(vc), *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
    float *scratch = dev_n<float>((size_t) nq * H * (ctx + 32 * 130)), *out = dev_n<float>((size_t) nq * H * D),
          *one = dev_n<float>((size_t) H * D);
    const float scale = 1.0f / std::sqrt(128.0f);
    struct Case { int pos0, window; const char* name; };
    const Case cases[] = {{3000, 1 << 30, "attn_multi full, 8 queries at 3000"}, {2500, 1024, "attn_multi sliding 1024"},
                          {0, 1 << 30, "attn_multi from position 0"}, {100, 1 << 30, "attn_multi at 100 (one run)"}};
    for (const Case& cs : cases) {
        const int k0 = cs.window < (1 << 30) ? std::max(0, cs.pos0 - cs.window + 1) : 0, nk = cs.pos0 + nq - k0;
        k::attn_multi(dq, dr, ds, dk, dv, nq, H, KV, k0, nk, cs.pos0, scale, dp, d_rel, 1024, cs.window, scratch, out, 0);
        const auto got = host(out, (size_t) nq * H * D);
        std::vector<double> want((size_t) nq * H * D);
        for (int qi = 0; qi < nq; ++qi) {
            const int qp = cs.pos0 + qi, kq = cs.window < (1 << 30) ? std::max(0, qp - cs.window + 1) : 0;
            k::attn_decode(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H, dk, dv, H, KV, D,
                           kq, qp + 1 - kq, qp, scale, dp, d_rel, 1024, cs.window, scratch, one, 0);
            const auto o1 = host(one, (size_t) H * D);
            for (size_t j = 0; j < o1.size(); ++j) want[(size_t) qi * H * D + j] = o1[j];
        }
        expect(cs.name, got, want, 1e-4, 1e-6);
    }
    for (float* p : {dk, dv, dq, dr, ds, dp, scratch, out, one}) cudaFree(p);
}

void test_mv_blocks() {
    struct Case {
        int type, N, K;
        const char* name;
    };
    const Case cases[] = {
        {k::kQ8_0, 384, 4608, "Q8_0 N=384"},
        {k::kQ8_0, 1031, 6144, "Q8_0 N=1031"},
        {k::kQ8_0, 2307, 4608, "Q8_0 N=2307"},
        {k::kQ8_0, 33, 96, "Q8_0 odd nb"},
        {k::kNVFP4, 1031, 6144, "NVFP4 N=1031"},
        {k::kQ6_K, 1031, 6144, "Q6_K N=1031"},
        {k::kQ5_K, 513, 4608, "Q5_K N=513"},
        {k::kIQ3_XXS, 1031, 4608, "IQ3_XXS N=1031"},
        {k::kIQ4_XS, 513, 4608, "IQ4_XS N=513"},
    };
    for (const Case& cs : cases) {
        const int N = cs.N, K = cs.K;
        auto raw = rand_blocks(cs.type, (size_t) N * K);
        auto x = fvec(K, 2.0f);
        uint8_t* dw = dev(raw);
        float *dx = dev(x), *dy = dev_n<float>(N);
        k::mv_blocks(cs.type, dw, N, K, dx, dy, 0);
        std::vector<double> want(N);
        std::vector<float> row(K);
        const size_t rb = raw.size() / N;
        for (int r = 0; r < N; ++r) {
            k::decode_host(cs.type, raw.data() + (size_t) r * rb, K, row.data());
            double a = 0;
            for (int c = 0; c < K; ++c) a += (double) row[c] * x[c];
            want[r] = a;
        }
        char name[64];
        std::snprintf(name, sizeof name, "mv_blocks %s", cs.name);
        // Cancellation over K=4608..6144 random blocks leaves ~1e-2 F32 rounding residual (same as test_experts_grouped).
        expect(name, host(dy, N), want, 2e-4, K > 2048 ? 3e-2 : 2e-3);
        cudaFree(dw), cudaFree(dx), cudaFree(dy);
    }
}

void test_attn_decode() {
    const int H = 48, KV = 8, D = 128, d_rel = 16, ctx = 3000;
    struct Case {
        int k0, nk, qpos, extent, window;
        const char* name;
    };
    const Case cases[] = {{0, 1, 0, 1024, 1 << 30, "attn_decode first token"},
                          {0, 2500, 2499, 1024, 1 << 30, "attn_decode full, 2500 keys"},
                          {1476, 1024, 2499, 1024, 1024, "attn_decode sliding 1024"},
                          {0, 200, 199, 1024, 1 << 30, "attn_decode full, 200 keys (one run)"},
                          {0, 2999, 2998, 64, 1 << 30, "attn_decode full, 2999 keys, extent 64"}};
    auto kc = fvec((size_t) ctx * KV * D), vc = fvec((size_t) ctx * KV * D);
    auto q = fvec((size_t) H * D, 3.0f), r = fvec((size_t) H * d_rel, 2.0f), sink = fvec(H, 3.0f),
         proj = fvec((size_t) 1024 * d_rel);
    float *dk = dev(kc), *dv = dev(vc), *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
    float *scratch = dev_n<float>((size_t) H * ctx + (size_t) H * 32 * 130), *dout = dev_n<float>((size_t) H * D);
    const float scale = 1.0f / std::sqrt(128.0f);
    for (const Case& cs : cases) {
        k::attn_decode(dq, dr, ds, dk, dv, H, KV, D, cs.k0, cs.nk, cs.qpos, scale, dp, d_rel, cs.extent, cs.window,
                       scratch, dout, 0);
        std::vector<double> want((size_t) H * D);
        for (int h = 0; h < H; ++h) {
            const int g = h / (H / KV);
            std::vector<double> l(cs.nk);
            double m = sink[h];
            for (int j = 0; j < cs.nk; ++j) {
                const int dist = cs.qpos - (cs.k0 + j);
                if (dist < 0 || dist >= cs.window) {
                    l[j] = -INFINITY;
                    continue;
                }
                double a = 0;
                for (int d = 0; d < D; ++d) a += (double) q[h * D + d] * kc[((size_t) (cs.k0 + j) * KV + g) * D + d];
                a *= scale;
                if (dist < cs.extent)
                    for (int e = 0; e < d_rel; ++e) a += (double) r[h * d_rel + e] * proj[(size_t) dist * d_rel + e];
                l[j] = a;
                m = std::max(m, a);
            }
            double sum = std::exp(sink[h] - m);
            for (int j = 0; j < cs.nk; ++j) sum += std::exp(l[j] - m);
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int j = 0; j < cs.nk; ++j)
                    o += std::exp(l[j] - m) / sum * vc[((size_t) (cs.k0 + j) * KV + g) * D + d];
                want[(size_t) h * D + d] = o;
            }
        }
        expect(cs.name, host(dout, want.size()), want, 2e-4, 1e-6);
    }
    for (float* p : {dk, dv, dq, dr, ds, dp, scratch, dout}) cudaFree(p);
}

void test_argmax() {
    // argmax_sum: multi-block (200058), single-block (777), null b/c, unaligned, and tie-breaking (lower index)
    {
        const int n = 200058;
        auto a = fvec(n, 5.0f), b = fvec(n, 5.0f), c = fvec(n, 5.0f);
        // Set two exact ties across blocks (500 and 150000) above the random range
        a[500] = 20.0f, b[500] = 10.0f, c[500] = 5.0f;
        a[150000] = 20.0f, b[150000] = 10.0f, c[150000] = 5.0f;
        float *da = dev(a), *db = dev(b), *dc = dev(c);
        int* dout = dev_n<int>(1);
        k::argmax_sum(da, db, dc, n, dout, 0);
        expect_true("argmax_sum (a+b+c, tie)", host(dout, 1)[0] == 500);
        a[199999] = 50.0f;
        CU(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        k::argmax_sum(da, nullptr, nullptr, n, dout, 0);
        expect_true("argmax_sum (a only, multi)", host(dout, 1)[0] == 199999);
        k::argmax_sum(da + 1, db + 1, nullptr, n - 1, dout, 0);
        int want_unaligned = 0;
        float best_u = -1e30f;
        for (int i = 0; i < n - 1; ++i) {
            const float v = a[1 + i] + b[1 + i];
            if (v > best_u) best_u = v, want_unaligned = i;
        }
        expect_true("argmax_sum (unaligned)", host(dout, 1)[0] == want_unaligned);
        k::argmax_sum(da, db, dc, 777, dout, 0);
        expect_true("argmax_sum (single block)", host(dout, 1)[0] == 500);
        cudaFree(da), cudaFree(db), cudaFree(dc), cudaFree(dout);
    }
    // argmax_rows: 8 rows x 200058 cols with ld = 201024, plus single-block case
    {
        const int rows = 8, cols = 200058, ld = 201024;
        auto x = fvec((size_t) rows * ld, 5.0f);
        x[3 * (size_t) ld + 1234] = 30.0f;
        x[3 * (size_t) ld + 180000] = 30.0f;
        float* dx = dev(x);
        int* dout = dev_n<int>(rows);
        k::argmax_rows(dx, ld, rows, cols, dout, 0);
        const auto got = host(dout, rows);
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            int w = 0;
            for (int c = 1; c < cols; ++c)
                if (x[(size_t) r * ld + c] > x[(size_t) r * ld + w]) w = c;
            ok &= got[r] == w;
        }
        expect_true("argmax_rows (8 x 200058, tie)", ok);
        k::argmax_rows(dx, ld, 1, 500, dout, 0);
        int w1 = 0;
        for (int c = 1; c < 500; ++c)
            if (x[c] > x[w1]) w1 = c;
        expect_true("argmax_rows (1 x 500)", host(dout, 1)[0] == w1);
        cudaFree(dx), cudaFree(dout);
    }
    // argmax_gemv_f16: W [R][ld] FP16, x [R] F32, base [n] F32, bias_f16 [n] FP16
    {
        const int R = 256, n = 200058, ld = 201024;
        std::vector<uint16_t> wh((size_t) R * ld), bh(n);
        for (auto& v : wh) v = f2h_host(frand(0.5f));
        for (auto& v : bh) v = f2h_host(frand(1.0f));
        auto x = fvec(R, 0.5f), base = fvec(n, 2.0f);
        uint16_t *dw = dev(wh), *dbias = dev(bh);
        float *dx = dev(x), *dbase = dev(base);
        int* dout = dev_n<int>(1);
        k::argmax_gemv_f16(dw, ld, dx, dbase, dbias, R, n, dout, 0);
        auto ref_gemv = [&](int off_w, int w_ld, int nn, bool use_base, bool use_bias) {
            int best_i = 0;
            float best_v = -INFINITY;
            for (int v = 0; v < nn; ++v) {
                float acc = 0.0f;
                for (int r = 0; r < R; ++r) acc = std::fma(h2f_host(wh[(size_t) r * w_ld + off_w + v]), x[r], acc);
                if (use_base) acc += base[v];
                if (use_bias) acc += h2f_host(bh[v]);
                if (acc > best_v) best_v = acc, best_i = v;
            }
            return best_i;
        };
        expect_true("argmax_gemv_f16 (200058)", host(dout, 1)[0] == ref_gemv(0, ld, n, true, true));
        k::argmax_gemv_f16(dw, ld, dx, nullptr, nullptr, R, 200057, dout, 0);
        expect_true("argmax_gemv_f16 (no base/bias, tail)", host(dout, 1)[0] == ref_gemv(0, ld, 200057, false, false));
        k::argmax_gemv_f16(dw + 1, ld, dx, dbase, dbias, R, 777, dout, 0);
        expect_true("argmax_gemv_f16 (unaligned, single)", host(dout, 1)[0] == ref_gemv(1, ld, 777, true, true));
        cudaFree(dw), cudaFree(dbias), cudaFree(dx), cudaFree(dbase), cudaFree(dout);
    }
}

void test_attn_f16() {
    const int H = 48, KV = 8, D = 128, d_rel = 16, ctx = 3100, kv_cap = 1056, window = 1024, nq = 8;
    auto kc_f = fvec((size_t) ctx * KV * D), vc_f = fvec((size_t) ctx * KV * D);
    std::vector<uint16_t> kc_lin(kc_f.size()), vc_lin(vc_f.size());
    for (size_t i = 0; i < kc_f.size(); ++i) {
        kc_lin[i] = f2h_host(kc_f[i]);
        vc_lin[i] = f2h_host(vc_f[i]);
    }
    // Ring-buffer copy holding positions [1476 .. 2507] wrapped mod kv_cap
    std::vector<uint16_t> kc_ring((size_t) kv_cap * KV * D, 0), vc_ring((size_t) kv_cap * KV * D, 0);
    for (int pos = 1476; pos < 2510; ++pos) {
        const int slot = pos % kv_cap;
        std::memcpy(kc_ring.data() + (size_t) slot * KV * D, kc_lin.data() + (size_t) pos * KV * D,
                    (size_t) KV * D * sizeof(uint16_t));
        std::memcpy(vc_ring.data() + (size_t) slot * KV * D, vc_lin.data() + (size_t) pos * KV * D,
                    (size_t) KV * D * sizeof(uint16_t));
    }
    auto q = fvec((size_t) nq * H * D, 3.0f), r = fvec((size_t) nq * H * d_rel, 2.0f), sink = fvec((size_t) nq * H, 3.0f),
         proj = fvec((size_t) 1024 * d_rel);
    uint16_t *dk_lin = dev(kc_lin), *dv_lin = dev(vc_lin), *dk_ring = dev(kc_ring), *dv_ring = dev(vc_ring);
    float *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
    float *scratch = dev_n<float>((size_t) nq * H * (ctx + 32 * 130)), *out = dev_n<float>((size_t) nq * H * D),
          *one = dev_n<float>((size_t) H * D);
    const float scale = 1.0f / std::sqrt(128.0f);

    // 1. attn_decode_f16 linear (kv_cap=0, full) and ring-buffer (kv_cap=1056, sliding) vs double reference
    for (int ring = 0; ring < 2; ++ring) {
        const int k0 = ring ? 1476 : 0, nk = ring ? 1024 : 2500, qpos = 2499, w = ring ? window : (1 << 30);
        k::attn_decode_f16(dq, dr, ds, ring ? dk_ring : dk_lin, ring ? dv_ring : dv_lin, ring ? kv_cap : 0, H, KV, D,
                           k0, nk, qpos, scale, dp, d_rel, 1024, w, scratch, one, 0);
        std::vector<double> want((size_t) H * D);
        for (int h = 0; h < H; ++h) {
            const int g = h / (H / KV);
            std::vector<double> l(nk);
            double m = sink[h];
            for (int j = 0; j < nk; ++j) {
                const int dist = qpos - (k0 + j);
                if (dist < 0 || dist >= w) {
                    l[j] = -INFINITY;
                    continue;
                }
                double a = 0;
                for (int d = 0; d < D; ++d)
                    a += (double) q[h * D + d] * h2f_host(kc_lin[((size_t) (k0 + j) * KV + g) * D + d]);
                a *= scale;
                if (dist < 1024)
                    for (int e = 0; e < d_rel; ++e) a += (double) r[h * d_rel + e] * proj[(size_t) dist * d_rel + e];
                l[j] = a;
                m = std::max(m, a);
            }
            double sum = std::exp(sink[h] - m);
            for (int j = 0; j < nk; ++j) sum += std::exp(l[j] - m);
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int j = 0; j < nk; ++j)
                    o += std::exp(l[j] - m) / sum * h2f_host(vc_lin[((size_t) (k0 + j) * KV + g) * D + d]);
                want[(size_t) h * D + d] = o;
            }
        }
        expect(ring ? "attn_decode_f16 ring 1056" : "attn_decode_f16 linear 2500", host(one, want.size()), want, 2e-4, 1e-6);
    }

    // 2. attn_multi_f16 ring-buffer (pos0 = 2500, k0 = 1477, nk = 1031) vs attn_decode_f16
    {
        const int pos0 = 2500, k0 = pos0 - window + 1, nk = pos0 + nq - k0;
        k::attn_multi_f16(dq, dr, ds, dk_ring, dv_ring, kv_cap, nq, H, KV, k0, nk, pos0, scale, dp, d_rel, 1024,
                          window, scratch, out, 0);
        const auto got = host(out, (size_t) nq * H * D);
        std::vector<double> want((size_t) nq * H * D);
        for (int qi = 0; qi < nq; ++qi) {
            const int qp = pos0 + qi, kq = qp - window + 1;
            k::attn_decode_f16(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H, dk_ring,
                               dv_ring, kv_cap, H, KV, D, kq, qp + 1 - kq, qp, scale, dp, d_rel, 1024, window,
                               scratch, one, 0);
            const auto o1 = host(one, (size_t) H * D);
            for (size_t j = 0; j < o1.size(); ++j) want[(size_t) qi * H * D + j] = o1[j];
        }
        expect("attn_multi_f16 ring 1056", got, want, 1e-4, 1e-6);
    }

    for (uint16_t* p : {dk_lin, dv_lin, dk_ring, dv_ring}) cudaFree(p);
    for (float* p : {dq, dr, ds, dp, scratch, out, one}) cudaFree(p);
}

// The resident token path's kernels: the planar Q8 product (with a short conv on some rows, with rows that take
// another input) and the FP16 one against a double reference; the launches that stand for several - rms_multi,
// attn_res_jobs, moe_combine_conv, the sinks in the clips' launch - against the ones they replace, to the bit; a
// looped layer's message gate, a token's kernels and a chunk's, against a double reference.
void test_token_path() {
    const int C = 2304;
    {   // mv_q8p and mv_q8p_conv
        const int N = 300, K = C, nb = K / 32;
        std::vector<int8_t> q((size_t) N * K);
        for (auto& v : q) v = (int8_t) ((int) (rng() % 255) - 127);
        std::vector<uint16_t> d((size_t) N * nb);
        for (auto& v : d) v = f2h_host(0.001f + 0.01f * std::fabs(frand()));
        auto x = fvec(K, 2.0f), xa = fvec(K, 2.0f), xb = fvec(K, 2.0f);
        int8_t* dq = dev(q);
        uint16_t* dd = dev(d);
        float *dx = dev(x), *dxa = dev(xa), *dxb = dev(xb), *dy = dev_n<float>(N);
        auto dot = [&](int row, const std::vector<float>& in) {
            double a = 0;
            for (int b = 0; b < nb; ++b) {
                double sb = 0;
                for (int j = 0; j < 32; ++j) sb += (double) q[(size_t) row * K + b * 32 + j] * in[b * 32 + j];
                a += (double) h2f_host(d[(size_t) row * nb + b]) * sb;
            }
            return a;
        };
        k::mv_q8p(dq, dd, N, K, dx, dy, 0);
        std::vector<double> want(N);
        for (int r = 0; r < N; ++r) want[r] = dot(r, x);
        expect("mv_q8p", host(dy, N), want, 2e-4, 1e-3);
        // rows 40..99 through a conv with dynamic taps and a residual, rows 200..259 through a plain one; rows
        // 100..149 take xa, 150..199 xb
        const int n0 = 60, n1 = 60, heads = 4;
        auto h0 = fvec(3 * n0), w0 = fvec(4 * n0), dyn = fvec(heads * 4), res = fvec(n0), h1 = fvec(3 * n1), w1 = fvec(4 * n1);
        float *dh0 = dev(h0), *dw0 = dev(w0), *ddyn = dev(dyn), *dres = dev(res), *dh1 = dev(h1), *dw1 = dev(w1);
        float *dy0 = dev_n<float>(n0), *dy1 = dev_n<float>(n1);
        CU(cudaMemset(dy, 0, N * sizeof(float)));
        k::mv_q8p_conv(dq, dd, N, K, dx, dy, k::MvConv{40, n0, dh0, dw0, ddyn, heads, dres, dy0},
                       k::MvConv{200, n1, dh1, dw1, nullptr, 0, nullptr, dy1}, 0, k::MvAlt{dxa, 100, 50},
                       k::MvAlt{dxb, 150, 50});
        const auto gy = host(dy, N), gy0 = host(dy0, n0), gy1 = host(dy1, n1), gh0 = host(dh0, 3 * n0);
        std::vector<float> got;
        std::vector<double> ref;
        for (int r = 0; r < N; ++r) {
            const double v = dot(r, r >= 100 && r < 150 ? xa : r >= 150 && r < 200 ? xb : x);
            if (r >= 40 && r < 40 + n0) {
                const int c = r - 40;
                const float* dc = dyn.data() + (size_t) (c / (n0 / heads)) * 4;
                double a = (double) w0[c * 4] * h0[c] + (double) w0[c * 4 + 1] * h0[n0 + c] +
                           (double) w0[c * 4 + 2] * h0[2 * n0 + c] + (double) w0[c * 4 + 3] * v;
                a += dc[0] * v + (double) dc[1] * h0[2 * n0 + c] + (double) dc[2] * h0[n0 + c] + (double) dc[3] * h0[c];
                got.push_back(gy0[c]), ref.push_back(a + res[c]);
                got.push_back(gh0[2 * n0 + c]), ref.push_back(v);         // (the history takes the row)
                got.push_back(gh0[c]), ref.push_back(h0[n0 + c]);
            } else if (r >= 200 && r < 200 + n1) {
                const int c = r - 200;
                got.push_back(gy1[c]);
                ref.push_back((double) w1[c * 4] * h1[c] + (double) w1[c * 4 + 1] * h1[n1 + c] +
                              (double) w1[c * 4 + 2] * h1[2 * n1 + c] + (double) w1[c * 4 + 3] * v);
            } else {
                got.push_back(gy[r]), ref.push_back(v);
            }
        }
        expect("mv_q8p_conv: convs and other inputs", got, ref, 2e-4, 2e-3);
        // mv_f16
        const int Nf = 40;
        std::vector<uint16_t> wf((size_t) Nf * K);
        for (auto& v : wf) v = f2h_host(frand());
        uint16_t* dwf = dev(wf);
        float* dyf = dev_n<float>(Nf);
        k::mv_f16(dwf, Nf, K, dx, dyf, 0);
        std::vector<double> wantf(Nf);
        for (int r = 0; r < Nf; ++r)
            for (int i = 0; i < K; ++i) wantf[r] += (double) h2f_host(wf[(size_t) r * K + i]) * x[i];
        expect("mv_f16", host(dyf, Nf), wantf, 2e-4, 1e-3);
        // the sinks in the clips' launch: mv_f16's values to the bit
        {
            const int H = 24, KV = 8, D = 128, d_rel = 16;
            auto qv = fvec((size_t) H * D), kv = fvec((size_t) KV * D), vv = fvec((size_t) KV * D), rv = fvec((size_t) H * d_rel);
            float *dqv = dev(qv), *dkv = dev(kv), *dvv = dev(vv), *drv = dev(rv), *dsk = dev_n<float>(H);
            uint16_t *dkc = dev_n<uint16_t>((size_t) 4 * KV * D), *dvc = dev_n<uint16_t>((size_t) 4 * KV * D);
            std::vector<int> pos = {2};
            int* dpos = dev(pos);
            k::prep_qkv_r_f16_at(dqv, nullptr, dkv, dkc, dvv, dvc, drv, nullptr, H, KV, D, d_rel, 22.6f, 8.0f, 1e-6f, dpos,
                                 false, 0.1f, 128000.0f, 0, nullptr, 0, dwf, dx, K, dsk);
            const auto a = host(dsk, H), b = host(dyf, H);
            expect_true("prep_qkv_r_f16_at: the sinks are mv_f16's to the bit", !std::memcmp(a.data(), b.data(), H * 4));
            for (float* p : {dqv, dkv, dvv, drv, dsk}) cudaFree(p);
            cudaFree(dkc), cudaFree(dvc), cudaFree(dpos);
        }
        cudaFree(dq), cudaFree(dd), cudaFree(dwf);
        for (float* p : {dx, dxa, dxb, dy, dh0, dw0, ddyn, dres, dh1, dw1, dy0, dy1, dyf}) cudaFree(p);
    }
    {   // rms_multi against rms; moe_combine_conv against moe_combine_f32 + conv4_res; attn_res_jobs against attn_res
        auto x3 = fvec((size_t) 3 * C, 3.0f), w3 = fvec((size_t) 3 * C);
        float *dx3 = dev(x3), *dw3 = dev(w3), *ya = dev_n<float>((size_t) 3 * C), *yb = dev_n<float>((size_t) 3 * C);
        for (int j = 0; j < 3; ++j) k::rms(dx3 + (size_t) j * C, dw3 + (size_t) j * C, 1, C, 1e-6f, ya + (size_t) j * C, 0);
        k::rms_multi(k::RmsJobs{{dx3, dx3 + C, dx3 + 2 * C}, {dw3, dw3 + C, dw3 + 2 * C}, {yb, yb + C, yb + 2 * C}}, 3, C,
                     1e-6f, 0);
        const auto ra = host(ya, (size_t) 3 * C), rb = host(yb, (size_t) 3 * C);
        expect_true("rms_multi: rms's values to the bit", !std::memcmp(ra.data(), rb.data(), ra.size() * 4));

        const int K = 8;
        auto y = fvec((size_t) K * C), wt = fvec(K), hist = fvec((size_t) 3 * C), tw = fvec((size_t) 4 * C), res = fvec(C);
        std::vector<int> inv = {3, 1, 0, 2, 7, 5, 6, 4};
        float *dyk = dev(y), *dwt = dev(wt), *dha = dev(hist), *dhb = dev(hist), *dtw = dev(tw), *dres = dev(res);
        float *tmp = dev_n<float>(C), *oa = dev_n<float>(C), *ob = dev_n<float>(C);
        int* dinv = dev(inv);
        k::moe_combine_f32(dyk, dinv, dwt, 1, K, C, tmp, false, 0);
        k::conv4_res(tmp, dha, dtw, nullptr, 0, 1, C, dres, oa, 0);
        k::moe_combine_conv(dyk, dinv, dwt, K, C, dhb, dtw, dres, ob, 0);
        const auto a1 = host(oa, C), b1 = host(ob, C), a2 = host(dha, (size_t) 3 * C), b2 = host(dhb, (size_t) 3 * C);
        expect_true("moe_combine_conv: moe_combine_f32 and conv4_res's values to the bit",
                    !std::memcmp(a1.data(), b1.data(), C * 4) && !std::memcmp(a2.data(), b2.data(), a2.size() * 4));

        const int n = 5;
        auto st = fvec((size_t) n * C, 2.0f), qq = fvec((size_t) 4 * C);
        float *dst = dev(st), *dqq = dev(qq), *ra4 = dev_n<float>((size_t) 4 * C), *rb4 = dev_n<float>((size_t) 4 * C);
        k::ResStates states{};
        for (int i = 0; i < n; ++i) states.st[i] = dst + (size_t) i * C;
        k::attn_res(states, n, dqq, 2, 1, C, 1e-6f, ra4, 0);
        k::attn_res(states, n, dqq + (size_t) 2 * C, 2, 1, C, 1e-6f, ra4 + (size_t) 2 * C, 0);
        k::attn_res_jobs(states, n, dqq, 2, 2, C, 1e-6f, rb4, 0);
        const auto a4 = host(ra4, (size_t) 4 * C), b4 = host(rb4, (size_t) 4 * C);
        expect_true("attn_res_jobs: two attn_res's values to the bit", !std::memcmp(a4.data(), b4.data(), a4.size() * 4));
        for (float* p : {dx3, dw3, ya, yb, dyk, dwt, dha, dhb, dtw, dres, tmp, oa, ob, dst, dqq, ra4, rb4}) cudaFree(p);
        cudaFree(dinv);
    }
    {   // a looped layer's gate: y = (1 - tanh(F g)) x + tanh(G g) m, g = D [rms(x); rms(m)]
        const int T = 3, R = 64;
        const float eps = 1e-6f;
        auto x = fvec((size_t) T * C, 2.0f), m = fvec((size_t) T * C, 2.0f), D = fvec((size_t) R * 2 * C, 0.05f),
             F = fvec((size_t) R * C, 0.3f), G = fvec((size_t) R * C, 0.3f);
        std::vector<double> want((size_t) T * C), a1((size_t) T * R), a2((size_t) T * R), fw((size_t) T * C), gw((size_t) T * C);
        for (int t = 0; t < T; ++t) {
            double sx = 0, sm = 0;
            for (int c = 0; c < C; ++c) sx += (double) x[(size_t) t * C + c] * x[(size_t) t * C + c], sm += (double) m[(size_t) t * C + c] * m[(size_t) t * C + c];
            const double rx = 1.0 / std::sqrt(sx / C + eps), rm = 1.0 / std::sqrt(sm / C + eps);
            std::vector<double> g(R);
            for (int r = 0; r < R; ++r) {
                double p1 = 0, p2 = 0;
                for (int c = 0; c < C; ++c) {
                    p1 += (double) D[(size_t) r * 2 * C + c] * x[(size_t) t * C + c];
                    p2 += (double) D[(size_t) r * 2 * C + C + c] * m[(size_t) t * C + c];
                }
                a1[(size_t) t * R + r] = p1, a2[(size_t) t * R + r] = p2;
                g[r] = rx * p1 + rm * p2;
            }
            for (int c = 0; c < C; ++c) {
                double f = 0, gn = 0;
                for (int r = 0; r < R; ++r) f += (double) F[(size_t) r * C + c] * g[r], gn += (double) G[(size_t) r * C + c] * g[r];
                fw[(size_t) t * C + c] = f, gw[(size_t) t * C + c] = gn;
                want[(size_t) t * C + c] = (1.0 - std::tanh(f)) * x[(size_t) t * C + c] + std::tanh(gn) * m[(size_t) t * C + c];
            }
        }
        float *dx = dev(x), *dm = dev(m), *dD = dev(D), *dF = dev(F), *dG = dev(G), *dy = dev_n<float>((size_t) T * C);
        float* dsc = dev_n<float>((size_t) 3 * T * (2 * R + 2));
        const k::LoopGate gt{dx, dm, dD, dF, dG, dy};
        k::loop_gates(&gt, 1, T, C, R, eps, dsc, 0);
        expect("loop_gates (a token's kernels)", host(dy, want.size()), want, 2e-4, 2e-4);
        // a chunk's: the products given (here the reference's), the two kernels around them
        std::vector<float> a1f(a1.begin(), a1.end()), a2f(a2.begin(), a2.end());
        float *da1 = dev(a1f), *da2 = dev(a2f), *dg = dev_n<float>((size_t) T * R);
        k::loop_gate_mix(dx, dm, da1, da2, T, C, R, eps, dg, 0);
        const auto gg = host(dg, (size_t) T * R);
        std::vector<float> ff((size_t) T * C), gf((size_t) T * C);   // f = g F, gn = g G from the kernel's g
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) {
                double f = 0, gn = 0;
                for (int r = 0; r < R; ++r) f += (double) F[(size_t) r * C + c] * gg[(size_t) t * R + r], gn += (double) G[(size_t) r * C + c] * gg[(size_t) t * R + r];
                ff[(size_t) t * C + c] = (float) f, gf[(size_t) t * C + c] = (float) gn;
            }
        float *df = dev(ff), *dgn = dev(gf);
        CU(cudaMemset(dy, 0, (size_t) T * C * sizeof(float)));
        k::loop_gate_blend(dx, dm, df, dgn, (size_t) T * C, dy, 0);
        expect("loop_gate_mix, loop_gate_blend (a chunk's)", host(dy, want.size()), want, 2e-4, 2e-4);
        for (float* p : {dx, dm, dD, dF, dG, dy, dsc, da1, da2, dg, df, dgn}) cudaFree(p);
    }
}

// cand_topk: every one of the K largest values is among the gathered, each gathered value is its token's, and the
// count says when the buffer was too small (a flat row)
void test_cand_topk() {
    const int n = 200058, K = 1000, cap = 4096;
    int* dscr = dev_n<int>(k::cand_scratch(n));
    int* dout = dev_n<int>(2 + 2 * cap);
    for (int kind = 0; kind < 3; ++kind) {
        std::vector<float> x(n);
        std::normal_distribution<float> nd(0.0f, kind == 0 ? 3.0f : 0.5f);
        for (auto& v : x) v = kind == 2 ? 1.25f : std::max(-29.9f, std::min(29.9f, nd(rng)));
        if (kind == 1)   // (ties at the cut and above it)
            for (int i = 0; i < n; i += 97) x[i] = 1.5f;
        float* dx = dev(x);
        k::cand_topk(dx, n, -30.0f, 30.0f, K, cap, dscr, dout, 0);
        const auto out = host(dout, 2 + 2 * cap);
        const int cnt = out[0];
        if (kind == 2) {
            expect_true("cand_topk: a flat row reports more values than the buffer holds", cnt == n);
            cudaFree(dx);
            continue;
        }
        bool ok = cnt >= K && cnt <= cap;
        std::vector<char> got(n, 0);
        for (int i = 0; i < std::min(cnt, cap) && ok; ++i) {
            const int id = out[2 + i];
            float v;
            std::memcpy(&v, &out[2 + cap + i], 4);
            ok = id >= 0 && id < n && !got[id] && v == x[id];
            if (ok) got[id] = 1;
        }
        if (ok) {   // nothing left out is larger than the K-th largest value
            std::vector<float> sorted = x;
            std::nth_element(sorted.begin(), sorted.begin() + (K - 1), sorted.end(), std::greater<float>());
            const float kth = sorted[K - 1];
            for (int i = 0; i < n && ok; ++i) ok = got[i] || x[i] < kth;
        }
        char label[96];
        std::snprintf(label, sizeof label, "cand_topk: the %d largest of %d are among its %d (%s)", K, n, cnt,
                      kind ? "ties" : "spread");
        expect_true(label, ok);
        cudaFree(dx);
    }
    cudaFree(dscr), cudaFree(dout);
}

float e4m3fn_to_f32_host(uint8_t q);
// attn_decode_mma_{f16,fp8} (one token's attention, a warp straight from the cache) against a double reference:
// 1 to 8 heads a K / V head, linear and ring caches, a window, the bias within and past its extent, the sink, a
// first token, a context shorter than a step, splits with and without keys; the FP8 cache with every finite byte
// (the subnormals among them).
void test_attn_decode_mma() {
    const int D = 128, d_rel = 16;
    struct Case {
        int H, KV, qpos, window, extent, kv_cap, n_split;
        const char* name;
    };
    const Case cases[] = {
        {12, 4, 0, 1 << 30, 64, 0, 1, "first token"},
        {12, 4, 9, 1 << 30, 64, 0, 3, "10 keys, 3 blocks"},
        {24, 8, 2499, 1 << 30, 1024, 0, 5, "2500 keys, 3 heads a K/V head"},
        {8, 1, 1000, 1 << 30, 200, 0, 2, "1001 keys, 8 heads a K/V head"},
        {4, 4, 333, 1 << 30, 64, 0, 7, "334 keys, 1 head a K/V head"},
        {12, 2, 2000, 256, 256, 400, 2, "ring 400, window 256, 6 heads"},
        {10, 2, 5000, 1024, 1024, 1056, 4, "ring 1056, window 1024, 5 heads"},
    };
    for (int fp8 = 0; fp8 < 2; ++fp8)
        for (const Case& c : cases) {
            const int H = c.H, KV = c.KV, n_pos = c.qpos + 1, rows = c.kv_cap > 0 ? c.kv_cap : n_pos;
            const float k_inv = fp8 ? 1.0f / 16.0f : 1.0f, v_inv = fp8 ? 1.0f / 32.0f : 1.0f;
            // the cache's values as the kernel reads them: kf, vf [n_pos][KV * D] (before the scales)
            std::vector<float> kf((size_t) n_pos * KV * D), vf((size_t) n_pos * KV * D);
            std::vector<uint16_t> k16, v16;
            std::vector<uint8_t> k8, v8;
            if (fp8) k8.assign((size_t) rows * KV * D, 0x7e), v8.assign((size_t) rows * KV * D, 0x7e);
            else k16.assign((size_t) rows * KV * D, 0x7e00), v16.assign((size_t) rows * KV * D, 0x7e00);   // (NaN: unused)
            for (int pos = 0; pos < n_pos; ++pos) {
                const size_t pr = (size_t) (c.kv_cap > 0 ? pos % c.kv_cap : pos) * KV * D;
                for (int x = 0; x < KV * D; ++x) {
                    const size_t i = (size_t) pos * KV * D + x;
                    if (fp8) {
                        uint8_t bk = (uint8_t) (rng() & 0xFFu), bv = (uint8_t) (rng() & 0xFFu);
                        if ((bk & 0x7F) >= 0x60) bk &= 0xDFu;   // (magnitudes under 32, as a clipped K times 16)
                        if ((bv & 0x7F) >= 0x60) bv &= 0xDFu;
                        k8[pr + x] = bk, v8[pr + x] = bv;
                        kf[i] = e4m3fn_to_f32_host(bk), vf[i] = e4m3fn_to_f32_host(bv);
                    } else {
                        k16[pr + x] = f2h_host(frand(1.0f)), v16[pr + x] = f2h_host(frand(1.0f));
                        kf[i] = h2f_host(k16[pr + x]), vf[i] = h2f_host(v16[pr + x]);
                    }
                }
            }
            auto qf = fvec((size_t) H * D, fp8 ? 2.0f : 4.0f);
            std::vector<uint16_t> q16(qf.size());
            for (size_t i = 0; i < qf.size(); ++i) q16[i] = f2h_host(qf[i]);
            auto r = fvec((size_t) H * d_rel, 2.0f), proj = fvec((size_t) c.extent * d_rel), sink = fvec((size_t) H, 3.0f);
            auto gate = fvec((size_t) H * D, 3.0f);   // (the FP8 runs go through the gate and take the FP16 copy)
            const float scale = 1.0f / std::sqrt((float) D);
            uint16_t* dq = dev(q16);
            float *dr = dev(r), *dp = dev(proj), *ds = dev(sink), *dout = dev_n<float>((size_t) H * D), *dg = dev(gate);
            uint16_t* dout16 = dev_n<uint16_t>((size_t) H * D);
            float* dscr = dev_n<float>(k::attn_decode_mma_scratch(H, c.n_split));
            std::vector<int> pos_h = {c.qpos};
            int* dpos = dev(pos_h);
            if (fp8) {
                uint8_t *dk = dev(k8), *dv = dev(v8);
                k::attn_decode_mma_fp8(dq, dr, ds, dk, dv, c.kv_cap, H, KV, dpos, -1, c.n_split, scale * k_inv, v_inv, dp,
                                       d_rel, c.extent, c.window, dscr, dout, dg, dout16, 0);
                CU(cudaDeviceSynchronize());
                cudaFree(dk), cudaFree(dv);
            } else {
                uint16_t *dk = dev(k16), *dv = dev(v16);
                k::attn_decode_mma_f16(dq, dr, ds, dk, dv, c.kv_cap, H, KV, nullptr, c.qpos, c.n_split, scale, dp, d_rel,
                                       c.extent, c.window, dscr, dout, nullptr, nullptr, 0);
                CU(cudaDeviceSynchronize());
                cudaFree(dk), cudaFree(dv);
            }
            const int lo = std::max(0, c.qpos - (c.window < (1 << 30) ? c.window : c.qpos + 1) + 1);
            std::vector<double> want((size_t) H * D);
            double mag = 0;   // the largest sum of |terms| of an output (the tolerance below)
            for (int h = 0; h < H; ++h) {
                const int g = h / (H / KV);
                std::vector<double> l(c.qpos + 1 - lo);
                double m = sink[h];
                for (int kp = lo; kp <= c.qpos; ++kp) {
                    double a = 0;
                    for (int d = 0; d < D; ++d)
                        a += (double) h2f_host(q16[(size_t) h * D + d]) * kf[((size_t) kp * KV + g) * D + d];
                    a *= (double) (scale * k_inv);
                    if (c.qpos - kp < c.extent)
                        for (int e = 0; e < d_rel; ++e)
                            a += (double) r[h * d_rel + e] * proj[(size_t) (c.qpos - kp) * d_rel + e];
                    l[kp - lo] = a;
                    m = std::max(m, a);
                }
                double sum = std::exp(sink[h] - m);
                for (double a : l) sum += std::exp(a - m);
                for (int d = 0; d < D; ++d) {
                    double o = 0, oa = 0;
                    for (int kp = lo; kp <= c.qpos; ++kp) {
                        const double t = std::exp(l[kp - lo] - m) / sum * vf[((size_t) kp * KV + g) * D + d];
                        o += t, oa += std::fabs(t);
                    }
                    const double gs = fp8 ? 1.0 / (1.0 + std::exp(-(double) gate[(size_t) h * D + d])) : 1.0;
                    want[(size_t) h * D + d] = o * (double) v_inv * gs;
                    mag = std::max(mag, oa * (double) v_inv);
                }
            }
            char label[96];
            std::snprintf(label, sizeof label, "mma %s %s", fp8 ? "fp8" : "f16", c.name);
            // (P is FP16 in the P V product: 2^-11 of a term)
            const auto got = host(dout, want.size());
            expect(label, got, want, 1e-2, std::max(1e-6, 1e-3 * mag));
            if (fp8) {   // the FP16 copy: the output's rounding
                const auto h16 = host(dout16, want.size());
                bool same = true;
                for (size_t i = 0; i < got.size(); ++i) same &= h16[i] == f2h_host(got[i]);
                if (!same) expect_true("mma fp8: the FP16 copy is the output rounded", false);
            }
            cudaFree(dq), cudaFree(dpos), cudaFree(dout16);
            for (float* p : {dr, dp, ds, dout, dscr, dg}) cudaFree(p);
        }
}

// attn_flash_f16 (the fused prefill attention) against a double reference: linear and ring caches, a window, the
// relative bias table, the sink, a chunk from 0 and one at an offset, 1 to 7 key splits (some of them empty).
void test_attn_flash() {
    const int H = 12, KV = 2, D = 128;
    struct Case {
        int Tq, qpos0, window, extent, kv_cap, n_split;
        const char* name;
    };
    const Case cases[] = {
        {130, 0, 1 << 30, 64, 0, 1, "attn_flash from 0"},
        {100, 300, 1 << 30, 64, 0, 3, "attn_flash at 300, 3 splits"},
        {77, 1500, 1 << 30, 200, 0, 7, "attn_flash at 1500, 7 splits"},
        {70, 2000, 256, 256, 400, 2, "attn_flash ring 400, window 256"},
        {9, 40, 1 << 30, 64, 0, 7, "attn_flash 9 queries, empty splits"},
    };
    for (const Case& c : cases) {
        const int n_pos = c.qpos0 + c.Tq;   // positions written so far
        const int rows = c.kv_cap > 0 ? c.kv_cap : n_pos;
        auto kf = fvec((size_t) n_pos * KV * D), vf = fvec((size_t) n_pos * KV * D);
        std::vector<uint16_t> kc((size_t) rows * KV * D, 0x7e00), vc((size_t) rows * KV * D, 0x7e00);   // (NaN: unused)
        for (int pos = 0; pos < n_pos; ++pos) {
            const int pr = c.kv_cap > 0 ? pos % c.kv_cap : pos;
            for (int x = 0; x < KV * D; ++x) {
                kc[(size_t) pr * KV * D + x] = f2h_host(kf[(size_t) pos * KV * D + x]);
                vc[(size_t) pr * KV * D + x] = f2h_host(vf[(size_t) pos * KV * D + x]);
            }
        }
        const int q_ld = H * D + 64;
        auto qf = fvec((size_t) c.Tq * q_ld, 4.0f);
        std::vector<uint16_t> q16(qf.size());
        for (size_t i = 0; i < qf.size(); ++i) q16[i] = f2h_host(qf[i]);
        auto bias = fvec((size_t) c.Tq * H * c.extent, 2.0f), sink = fvec((size_t) c.Tq * H, 3.0f);
        const float scale = 1.0f / std::sqrt((float) D);
        const int o_ld = H * D + 32;
        uint16_t *dq = dev(q16), *dk = dev(kc), *dv = dev(vc);
        float *db = dev(bias), *ds = dev(sink), *dout = dev_n<float>((size_t) c.Tq * o_ld);
        const size_t scr = (size_t) 8 * c.Tq * H * (D + 2);
        float* dscr = dev_n<float>(scr);
        k::attn_flash_f16(dq, q_ld, dk, dv, c.kv_cap, H, KV, D, c.Tq, c.qpos0, scale, db, c.extent, c.window, ds, dout,
                          o_ld, dscr, scr, c.n_split, 0);
        const auto out = host(dout, (size_t) c.Tq * o_ld);
        std::vector<float> got;
        std::vector<double> want;
        for (int i = 0; i < c.Tq; ++i) {
            const int qp = c.qpos0 + i, lo = std::max(0, qp - c.window + 1);
            for (int h = 0; h < H; ++h) {
                const int g = h / (H / KV);
                std::vector<double> l(qp + 1 - lo);
                double m = sink[(size_t) i * H + h];
                for (int kp = lo; kp <= qp; ++kp) {
                    double a = 0;
                    for (int d = 0; d < D; ++d)
                        a += (double) h2f_host(q16[(size_t) i * q_ld + h * D + d]) *
                             h2f_host(kc[((size_t) (c.kv_cap > 0 ? kp % c.kv_cap : kp) * KV + g) * D + d]);
                    a *= scale;
                    if (qp - kp < c.extent) a += bias[((size_t) i * H + h) * c.extent + (qp - kp)];
                    l[kp - lo] = a;
                    m = std::max(m, a);
                }
                double sum = std::exp(sink[(size_t) i * H + h] - m);
                for (double a : l) sum += std::exp(a - m);
                for (int d = 0; d < D; ++d) {
                    double o = 0;
                    for (int kp = lo; kp <= qp; ++kp)
                        o += std::exp(l[kp - lo] - m) / sum * h2f_host(vc[((size_t) (c.kv_cap > 0 ? kp % c.kv_cap : kp) * KV + g) * D + d]);
                    want.push_back(o);
                    got.push_back(out[(size_t) i * o_ld + h * D + d]);
                }
            }
        }
        expect(c.name, got, want, 1e-2, 2e-3);
        for (uint16_t* p : {dq, dk, dv}) cudaFree(p);
        for (float* p : {db, ds, dout, dscr}) cudaFree(p);
    }
}

void test_round2_fused() {
    // 1. rms_f16: (with/without w) x (y16+y32, y16-only, y32-only) + unaligned/non-mult4 tail
    {
        struct Case {
            int rows, D, off;
            bool with_w, w16, w32;
            const char* name;
        };
        const Case cases[] = {
            {5, 4608, 0, true, true, true, "rms_f16 w y16+y32"},
            {5, 4608, 0, false, true, false, "rms_f16 no-w y16-only"},
            {5, 128, 0, true, false, true, "rms_f16 w y32-only"},
            {4, 130, 1, true, true, true, "rms_f16 unaligned tail"},
        };
        for (const Case& cs : cases) {
            auto x = fvec((size_t) cs.rows * cs.D + cs.off, 3.0f), w = fvec(cs.D + cs.off);
            float *dx = dev(x), *dw = dev(w), *dy32 = dev_n<float>((size_t) cs.rows * cs.D + cs.off);
            uint16_t* dy16 = dev_n<uint16_t>((size_t) cs.rows * cs.D + cs.off);
            const float eps = 1e-6f;
            k::rms_f16(dx + cs.off, cs.with_w ? dw + cs.off : nullptr, cs.rows, cs.D, eps,
                       cs.w16 ? dy16 + cs.off : nullptr, cs.w32 ? dy32 + cs.off : nullptr, 0);
            std::vector<double> want((size_t) cs.rows * cs.D);
            for (int r = 0; r < cs.rows; ++r) {
                double ss = 0;
                for (int d = 0; d < cs.D; ++d) {
                    const double v = x[cs.off + (size_t) r * cs.D + d];
                    ss += v * v;
                }
                const double rs = 1.0 / std::sqrt(ss / cs.D + eps);
                for (int d = 0; d < cs.D; ++d)
                    want[(size_t) r * cs.D + d] = x[cs.off + (size_t) r * cs.D + d] * rs * (cs.with_w ? w[cs.off + d] : 1.0);
            }
            if (cs.w32) {
                const auto got32 = host(dy32 + cs.off, want.size());
                char nm[64];
                std::snprintf(nm, sizeof nm, "%s (f32)", cs.name);
                expect(nm, got32, want, 2e-5);
            }
            if (cs.w16) {
                const auto raw16 = host(dy16 + cs.off, want.size());
                std::vector<float> got16(want.size());
                for (size_t i = 0; i < want.size(); ++i) got16[i] = h2f_host(raw16[i]);
                char nm[64];
                std::snprintf(nm, sizeof nm, "%s (f16)", cs.name);
                expect(nm, got16, want, 2e-3, 1e-4);
            }
            cudaFree(dx), cudaFree(dw), cudaFree(dy32), cudaFree(dy16);
        }
    }

    // 2. gate_mul_f16: f16-only vs f16+f32, non-mult4 sizes, n < 4, and unaligned o16
    {
        struct Case {
            size_t n;
            int off16;
            bool write_f32;
            const char* name;
        };
        const Case cases[] = {
            {4096, 0, false, "gate_mul_f16 f16-only"},
            {4099, 0, true, "gate_mul_f16 tail f16+f32"},
            {3, 0, true, "gate_mul_f16 n=3"},
            {1031, 1, false, "gate_mul_f16 unaligned o16"},
        };
        for (const Case& cs : cases) {
            auto o = fvec(cs.n, 3.0f), g = fvec(cs.n, 4.0f);
            float *do_ = dev(o), *dg = dev(g);
            uint16_t* do16 = dev_n<uint16_t>(cs.n + cs.off16);
            k::gate_mul_f16(do_, dg, cs.n, do16 + cs.off16, cs.write_f32, 0);
            std::vector<double> want(cs.n), orig(o.begin(), o.end());
            for (size_t i = 0; i < cs.n; ++i) want[i] = (double) o[i] * (1.0 / (1.0 + std::exp(-(double) g[i])));
            const auto raw16 = host(do16 + cs.off16, cs.n);
            std::vector<float> got16(cs.n);
            for (size_t i = 0; i < cs.n; ++i) got16[i] = h2f_host(raw16[i]);
            expect(cs.name, got16, want, 2e-3, 1e-4);
            const auto got32 = host(do_, cs.n);
            expect(cs.write_f32 ? "gate_mul_f16 f32 updated" : "gate_mul_f16 f32 untouched", got32,
                   cs.write_f32 ? want : orig, 1e-5);
            cudaFree(do_), cudaFree(dg), cudaFree(do16);
        }
    }

    // 3. prep_qkv_r_f16: sliding (pos=nullptr, kv_cap wraparound) and full (pos!=nullptr, kv_cap=0)
    {
        const int T = 5, H = 48, KV = 8, D = 128, d_rel = 16, kv_cap = 1056;
        const float qk_clip = 22.627416610717773f, rel_clip = 8.0f, eps = 1e-6f, alpha = 0.1f, n_floor = 128000.0f;
        for (int full = 0; full < 2; ++full) {
            const int cap = full ? 0 : kv_cap, pos0 = full ? 100 : 1054;
            const int kc_rows = cap > 0 ? cap : pos0 + T;
            auto q = fvec((size_t) T * H * D, 4.0f), qc = fvec((size_t) H * D, 2.0f),
                 k_in = fvec((size_t) T * KV * D, 4.0f), v_in = fvec((size_t) T * KV * D, 3.0f),
                 r = fvec((size_t) T * H * d_rel, 4.0f), rc = fvec((size_t) H * d_rel, 2.0f);
            for (size_t i = 0; i < (size_t) H * D; ++i) q[i] *= 0.01f;
            std::vector<int> pos = {5, 130000, 200000, 127999, 300000};
            float *dq = dev(q), *dqc = dev(qc), *dk = dev(k_in), *dv = dev(v_in), *dr = dev(r), *drc = dev(rc);
            uint16_t *dkc = dev_n<uint16_t>((size_t) kc_rows * KV * D),
                     *dvc = dev_n<uint16_t>((size_t) kc_rows * KV * D);
            int* dpos = dev(pos);
            k::prep_qkv_r_f16(dq, dqc, dk, dkc, dv, dvc, dr, drc, T, H, KV, D, d_rel, qk_clip, rel_clip, eps,
                              full ? dpos : nullptr, alpha, n_floor, pos0, cap, 0);
            std::vector<double> wq(q.size()), wr(r.size()), wk((size_t) T * KV * D), wv((size_t) T * KV * D);
            for (int t = 0; t < T; ++t) {
                const double tau =
                    full ? (1.0 + alpha * std::log(std::max(1.0, ((double) pos[t] + 1.0) / n_floor))) : 1.0;
                for (int h = 0; h < H; ++h) {
                    double ss = 0;
                    for (int d = 0; d < D; ++d) {
                        const double v = (double) q[((size_t) t * H + h) * D + d] * qc[(size_t) h * D + d];
                        ss += v * v;
                    }
                    const double f = std::min(1.0, qk_clip / (std::sqrt(ss) + 1e-6)) * tau;
                    for (int d = 0; d < D; ++d)
                        wq[((size_t) t * H + h) * D + d] = (double) q[((size_t) t * H + h) * D + d] * qc[(size_t) h * D + d] * f;

                    double sr = 0;
                    for (int d = 0; d < d_rel; ++d) {
                        const double v = (double) r[((size_t) t * H + h) * d_rel + d] * rc[(size_t) h * d_rel + d];
                        sr += v * v;
                    }
                    const double fr = std::min(1.0, rel_clip / (std::sqrt(sr) + 1e-6)) * tau;
                    for (int d = 0; d < d_rel; ++d)
                        wr[((size_t) t * H + h) * d_rel + d] =
                            (double) r[((size_t) t * H + h) * d_rel + d] * rc[(size_t) h * d_rel + d] * fr;
                }
                for (int kv = 0; kv < KV; ++kv) {
                    double sk = 0, sv = 0;
                    for (int d = 0; d < D; ++d) {
                        const double kv_v = k_in[((size_t) t * KV + kv) * D + d];
                        const double vv_v = v_in[((size_t) t * KV + kv) * D + d];
                        sk += kv_v * kv_v;
                        sv += vv_v * vv_v;
                    }
                    const double fk = std::min(1.0, qk_clip / (std::sqrt(sk) + 1e-6));
                    const double rs = 1.0 / std::sqrt(sv / D + eps);
                    for (int d = 0; d < D; ++d) {
                        wk[((size_t) t * KV + kv) * D + d] = k_in[((size_t) t * KV + kv) * D + d] * fk;
                        wv[((size_t) t * KV + kv) * D + d] = v_in[((size_t) t * KV + kv) * D + d] * rs;
                    }
                }
            }
            const auto kc_all = host(dkc, (size_t) kc_rows * KV * D);
            const auto vc_all = host(dvc, (size_t) kc_rows * KV * D);
            std::vector<float> got_k((size_t) T * KV * D), got_v((size_t) T * KV * D);
            for (int t = 0; t < T; ++t) {
                const int slot = cap > 0 ? (pos0 + t) % cap : (pos0 + t);
                for (size_t j = 0; j < (size_t) KV * D; ++j) {
                    got_k[(size_t) t * KV * D + j] = h2f_host(kc_all[(size_t) slot * KV * D + j]);
                    got_v[(size_t) t * KV * D + j] = h2f_host(vc_all[(size_t) slot * KV * D + j]);
                }
            }
            expect(full ? "prep_qkv_r_f16 full q" : "prep_qkv_r_f16 ring q", host(dq, wq.size()), wq, 2e-5);
            expect(full ? "prep_qkv_r_f16 full r" : "prep_qkv_r_f16 ring r", host(dr, wr.size()), wr, 2e-5);
            expect(full ? "prep_qkv_r_f16 full k" : "prep_qkv_r_f16 ring k", got_k, wk, 2e-3, 1e-4);
            expect(full ? "prep_qkv_r_f16 full v" : "prep_qkv_r_f16 ring v", got_v, wv, 2e-3, 1e-4);
            for (float* p : {dq, dqc, dk, dv, dr, drc}) cudaFree(p);
            cudaFree(dkc), cudaFree(dvc), cudaFree(dpos);
        }
    }

    // 4. experts_gate_up_swiglu vs experts_gate_up + swiglu_cap_f32
    {
        const int C = 512, F = 256, E = 3;
        struct Case {
            int tg, tu, off;
            bool with_scale;
            const char* name;
        };
        const Case cases[] = {
            {k::kNVFP4, k::kNVFP4, 0, true, "NVFP4 scaled"},
            {k::kQ5_K, k::kQ5_K, 0, true, "Q5_K scaled"},
            {k::kQ6_K, k::kQ6_K, 0, false, "Q6_K unscaled"},
            {k::kQ8_0, k::kQ8_0, 0, true, "Q8_0 aligned"},
            {k::kQ8_0, k::kQ8_0, 2, true, "Q8_0 unaligned"},
            {k::kQ8_0, k::kNVFP4, 0, true, "Q8_0/NVFP4 mixed"},
            {k::kIQ3_XXS, k::kIQ3_XXS, 0, true, "IQ3_XXS scaled"},
            {k::kIQ4_XS, k::kIQ4_XS, 0, false, "IQ4_XS unscaled"},
        };
        for (const Case& cs : cases) {
            auto g = rand_blocks(cs.tg, (size_t) E * F * C), u = rand_blocks(cs.tu, (size_t) E * F * C);
            const size_t eb0 = g.size() / E, eb1 = u.size() / E, slot = (eb0 + eb1 + 255) / 256 * 256;
            std::vector<uint8_t> pool(slot * E + cs.off, 0);
            for (int e = 0; e < E; ++e) {
                std::memcpy(pool.data() + cs.off + e * slot, g.data() + e * eb0, eb0);
                std::memcpy(pool.data() + cs.off + e * slot + eb0, u.data() + e * eb1, eb1);
            }
            uint8_t* dpool = dev(pool);
            const int S_max = 5, pick[S_max] = {2, 0, 2, 1, 0}, tok[S_max] = {0, 1, 1, 2, 0};
            std::vector<const uint8_t*> blob(S_max);
            for (int i = 0; i < S_max; ++i) blob[i] = dpool + cs.off + pick[i] * slot;
            const uint8_t** dblob = nullptr;
            CU(cudaMalloc(&dblob, S_max * sizeof(void*)));
            CU(cudaMemcpy(dblob, blob.data(), S_max * sizeof(void*), cudaMemcpyHostToDevice));
            auto x = fvec((size_t) 3 * C, 1.5f), sg = fvec(S_max, 0.5f), su = fvec(S_max, 0.5f);
            std::vector<float> ones(S_max, 1.0f);
            for (int i = 0; i < S_max; ++i) {
                sg[i] = 0.7f + 0.1f * i;
                su[i] = 0.8f + 0.05f * i;
            }
            float *dx = dev(x), *dgu = dev_n<float>((size_t) S_max * 2 * F), *dsg = dev(sg), *dsu = dev(su),
                  *dones = dev(ones);
            int* dtok = dev(std::vector<int>(tok, tok + S_max));
            uint16_t *dh_ref = dev_n<uint16_t>((size_t) S_max * F), *dh_got = dev_n<uint16_t>((size_t) S_max * F);
            const float gcap = 20.0f, ucap = 20.0f;
            for (int S : {1, 5}) {
                k::experts_gate_up(dblob, dtok, S, cs.tg, cs.tu, eb0, dx, C, F, dgu, 0);
                k::swiglu_cap_f32(dgu, S, F, cs.with_scale ? dsg : dones, cs.with_scale ? dsu : dones, gcap, ucap,
                                  dh_ref, 0);
                k::experts_gate_up_swiglu(dblob, dtok, S, cs.tg, cs.tu, eb0, dx, C, F, cs.with_scale ? dsg : nullptr,
                                          cs.with_scale ? dsu : nullptr, gcap, ucap, dh_got, 0);
                const auto r16 = host(dh_ref, (size_t) S * F), g16 = host(dh_got, (size_t) S * F);
                std::vector<double> want((size_t) S * F);
                std::vector<float> got((size_t) S * F);
                for (size_t i = 0; i < want.size(); ++i) {
                    want[i] = h2f_host(r16[i]);
                    got[i] = h2f_host(g16[i]);
                }
                char nm[64];
                std::snprintf(nm, sizeof nm, "experts_gu_swiglu S=%d %s", S, cs.name);
                expect(nm, got, want, 3e-3, 1e-2);
            }
            cudaFree(dpool), cudaFree(dblob), cudaFree(dx), cudaFree(dgu), cudaFree(dsg), cudaFree(dsu), cudaFree(dones),
                cudaFree(dtok), cudaFree(dh_ref), cudaFree(dh_got);
        }
    }

    // 5. experts_gate_up_swiglu_g vs experts_gate_up_g + swiglu_cap_f32
    {
        const int C = 512, F = 256, E = 5, T = 8, K = 8, S = T * K;
        const int types[][2] = {{k::kNVFP4, k::kNVFP4},
                                {k::kQ5_K, k::kQ5_K},
                                {k::kQ6_K, k::kQ6_K},
                                {k::kQ8_0, k::kQ8_0},
                                {k::kQ8_0, k::kNVFP4},
                                {k::kIQ3_XXS, k::kIQ3_XXS},
                                {k::kIQ4_XS, k::kIQ4_XS}};
        const char* tnames[] = {"NVFP4", "Q5_K", "Q6_K", "Q8_0", "Q8_0/NVFP4", "IQ3_XXS", "IQ4_XS"};
        const int want_ng[E] = {1, 2, 3, 5, 8};
        for (int ti = 0; ti < 7; ++ti) {
            const int tg = types[ti][0], tu = types[ti][1];
            auto g = rand_blocks(tg, (size_t) E * F * C), u = rand_blocks(tu, (size_t) E * F * C);
            const size_t eb0 = g.size() / E, eb1 = u.size() / E, slot = (eb0 + eb1 + 255) / 256 * 256;
            std::vector<uint8_t> pool(slot * E);
            for (int e = 0; e < E; ++e) {
                std::memcpy(pool.data() + e * slot, g.data() + e * eb0, eb0);
                std::memcpy(pool.data() + e * slot + eb0, u.data() + e * eb1, eb1);
            }
            uint8_t* dpool = dev(pool);
            std::vector<int> pick(S, -1), gn, gtok, grow, idx;
            std::vector<const uint8_t*> gb;
            for (int e = 0; e < E; ++e)
                for (int t = 0; t < want_ng[e]; ++t) pick[t * K + e] = e;
            for (int i = 0; i < S; ++i)
                if (pick[i] >= 0) idx.push_back(i);
            const int n = (int) idx.size();
            std::vector<int> gi(E, -1);
            for (int r = 0; r < n; ++r) {
                const int i = idx[r], e = pick[i];
                if (gi[e] < 0) {
                    gi[e] = (int) gb.size();
                    gb.push_back(dpool + e * slot);
                    gn.push_back(0);
                    gtok.resize(gtok.size() + k::kGroupMax);
                    grow.resize(grow.size() + k::kGroupMax);
                }
                const int gg = gi[e];
                gtok[gg * k::kGroupMax + gn[gg]] = i / K;
                grow[gg * k::kGroupMax + gn[gg]] = r;
                ++gn[gg];
            }
            const uint8_t** dgb = nullptr;
            CU(cudaMalloc(&dgb, gb.size() * sizeof(void*)));
            CU(cudaMemcpy(dgb, gb.data(), gb.size() * sizeof(void*), cudaMemcpyHostToDevice));
            int *dgn = dev(gn), *dgt = dev(gtok), *dgr = dev(grow);
            auto x = fvec((size_t) T * C, 1.5f), sg = fvec(n), su = fvec(n);
            std::vector<float> ones(n, 1.0f);
            for (int i = 0; i < n; ++i) {
                sg[i] = 0.6f + 0.03f * i;
                su[i] = 0.9f + 0.02f * i;
            }
            float *dx = dev(x), *dgu = dev_n<float>((size_t) n * 2 * F), *dsg = dev(sg), *dsu = dev(su),
                  *dones = dev(ones);
            uint16_t *dh_ref = dev_n<uint16_t>((size_t) n * F), *dh_got = dev_n<uint16_t>((size_t) n * F);
            const float gcap = 20.0f, ucap = 20.0f;
            for (int with_scale = 0; with_scale < 2; ++with_scale) {
                k::experts_gate_up_g(dgb, dgn, dgt, dgr, (int) gb.size(), tg, tu, eb0, dx, C, F, dgu, 0);
                k::swiglu_cap_f32(dgu, n, F, with_scale ? dsg : dones, with_scale ? dsu : dones, gcap, ucap, dh_ref,
                                  0);
                k::experts_gate_up_swiglu_g(dgb, dgn, dgt, dgr, (int) gb.size(), tg, tu, eb0, dx, C, F,
                                            with_scale ? dsg : nullptr, with_scale ? dsu : nullptr, gcap, ucap, dh_got,
                                            0);
                const auto r16 = host(dh_ref, (size_t) n * F), g16 = host(dh_got, (size_t) n * F);
                std::vector<double> want((size_t) n * F);
                std::vector<float> got((size_t) n * F);
                for (size_t i = 0; i < want.size(); ++i) {
                    want[i] = h2f_host(r16[i]);
                    got[i] = h2f_host(g16[i]);
                }
                char nm[64];
                std::snprintf(nm, sizeof nm, "experts_gu_swiglu_g %s%s", tnames[ti], with_scale ? "" : " (no scale)");
                expect(nm, got, want, 3e-3, 1e-2);
            }
            for (void* p : {(void*) dpool, (void*) dgb, (void*) dgn, (void*) dgt, (void*) dgr, (void*) dx, (void*) dgu,
                            (void*) dsg, (void*) dsu, (void*) dones, (void*) dh_ref, (void*) dh_got})
                cudaFree(p);
        }
    }
}

float e4m3fn_to_f32_host(uint8_t q) {
    const uint32_t mag = q & 0x7Fu;
    const float sgn = (q & 0x80u) ? -1.0f : 1.0f;
    if (mag == 0x7Fu) return NAN;
    const uint32_t e = (mag >> 3) & 15u, m = mag & 7u;
    if (e == 0) return sgn * std::ldexp((float) m, -9);
    return sgn * std::ldexp(1.0f + (float) m / 8.0f, (int) e - 7);
}

void test_fp8_kv() {
    // 1. All 254 finite E4M3FN byte values -> unpack_kv_fp8_to_f16 bit-for-bit equality vs exact FP16
    {
        std::vector<uint8_t> all_b(256);
        for (int i = 0; i < 256; ++i) all_b[i] = (uint8_t) i;
        uint8_t* db = dev(all_b);
        uint16_t *dk16 = dev_n<uint16_t>(256), *dv16 = dev_n<uint16_t>(256);
        k::unpack_kv_fp8_to_f16(db, db, 0, 0, 1, 256, 1.0f, 1.0f, dk16, dv16, 0);
        const auto hk = host(dk16, 256), hv = host(dv16, 256);
        bool finite_exact = true, nan_ok = true;
        for (int b = 0; b < 256; ++b) {
            if ((b & 0x7F) == 0x7F) {
                nan_ok &= ((hk[b] & 0x7FFF) > 0x7C00) && ((hv[b] & 0x7FFF) > 0x7C00);
            } else {
                const uint16_t want_bits = f2h_host(e4m3fn_to_f32_host((uint8_t) b));
                finite_exact &= (hk[b] == want_bits) && (hv[b] == want_bits);
            }
        }
        expect_true("e4m3fn_to_f16: all 254 finite bytes bit-exact", finite_exact);
        expect_true("e4m3fn_to_f16: 0x7F/0xFF NaN", nan_ok);
        cudaFree(db), cudaFree(dk16), cudaFree(dv16);
    }

    // 2. prep_qkv_r_fp8 + unpack_kv_fp8_to_f16 (ring wraparound, linear, and scalar KD=10 tail)
    {
        const int T = 5, H = 48, KV = 8, D = 128, d_rel = 16, kv_cap = 1056;
        const float qk_clip = 22.627416610717773f, rel_clip = 8.0f, eps = 1e-6f, alpha = 0.1f, n_floor = 128000.0f;
        const float k_scale = 16.0f, v_scale = 32.0f, k_inv = 1.0f / k_scale, v_inv = 1.0f / v_scale;
        for (int full = 0; full < 2; ++full) {
            const int cap = full ? 0 : kv_cap, pos0 = full ? 100 : 1054;
            const int kc_rows = cap > 0 ? cap : pos0 + T;
            auto q = fvec((size_t) T * H * D, 4.0f), qc = fvec((size_t) H * D, 2.0f),
                 k_in = fvec((size_t) T * KV * D, 4.0f), v_in = fvec((size_t) T * KV * D, 3.0f),
                 r = fvec((size_t) T * H * d_rel, 4.0f), rc = fvec((size_t) H * d_rel, 2.0f);
            for (size_t i = 0; i < (size_t) H * D; ++i) q[i] *= 0.01f;
            std::vector<int> pos = {5, 130000, 200000, 127999, 300000};
            float *dq = dev(q), *dqc = dev(qc), *dk = dev(k_in), *dv = dev(v_in), *dr = dev(r), *drc = dev(rc);
            uint8_t *dkc = dev_n<uint8_t>((size_t) kc_rows * KV * D),
                    *dvc = dev_n<uint8_t>((size_t) kc_rows * KV * D);
            int* dpos = dev(pos);
            k::prep_qkv_r_fp8(dq, dqc, dk, dkc, dv, dvc, dr, drc, T, H, KV, D, d_rel, qk_clip, rel_clip, eps,
                              full ? dpos : nullptr, alpha, n_floor, pos0, cap, k_scale, v_scale, 0);
            std::vector<double> wq(q.size()), wr(r.size()), wk((size_t) T * KV * D), wv((size_t) T * KV * D);
            for (int t = 0; t < T; ++t) {
                const double tau =
                    full ? (1.0 + alpha * std::log(std::max(1.0, ((double) pos[t] + 1.0) / n_floor))) : 1.0;
                for (int h = 0; h < H; ++h) {
                    double ss = 0;
                    for (int d = 0; d < D; ++d) {
                        const double v = (double) q[((size_t) t * H + h) * D + d] * qc[(size_t) h * D + d];
                        ss += v * v;
                    }
                    const double f = std::min(1.0, qk_clip / (std::sqrt(ss) + 1e-6)) * tau;
                    for (int d = 0; d < D; ++d)
                        wq[((size_t) t * H + h) * D + d] =
                            (double) q[((size_t) t * H + h) * D + d] * qc[(size_t) h * D + d] * f;

                    double sr = 0;
                    for (int d = 0; d < d_rel; ++d) {
                        const double v = (double) r[((size_t) t * H + h) * d_rel + d] * rc[(size_t) h * d_rel + d];
                        sr += v * v;
                    }
                    const double fr = std::min(1.0, rel_clip / (std::sqrt(sr) + 1e-6)) * tau;
                    for (int d = 0; d < d_rel; ++d)
                        wr[((size_t) t * H + h) * d_rel + d] =
                            (double) r[((size_t) t * H + h) * d_rel + d] * rc[(size_t) h * d_rel + d] * fr;
                }
                for (int kv = 0; kv < KV; ++kv) {
                    double sk = 0, sv = 0;
                    for (int d = 0; d < D; ++d) {
                        const double kv_v = k_in[((size_t) t * KV + kv) * D + d];
                        const double vv_v = v_in[((size_t) t * KV + kv) * D + d];
                        sk += kv_v * kv_v;
                        sv += vv_v * vv_v;
                    }
                    const double fk = std::min(1.0, qk_clip / (std::sqrt(sk) + 1e-6));
                    const double rs = 1.0 / std::sqrt(sv / D + eps);
                    for (int d = 0; d < D; ++d) {
                        wk[((size_t) t * KV + kv) * D + d] = k_in[((size_t) t * KV + kv) * D + d] * fk;
                        wv[((size_t) t * KV + kv) * D + d] = v_in[((size_t) t * KV + kv) * D + d] * rs;
                    }
                }
            }
            // Also check unpack_kv_fp8_to_f16 across wrapped slots [pos0 .. pos0+T)
            uint16_t *uk16 = dev_n<uint16_t>((size_t) T * KV * D), *uv16 = dev_n<uint16_t>((size_t) T * KV * D);
            k::unpack_kv_fp8_to_f16(dkc, dvc, cap, pos0, T, KV * D, k_inv, v_inv, uk16, uv16, 0);
            const auto hk_un = host(uk16, (size_t) T * KV * D), hv_un = host(uv16, (size_t) T * KV * D);
            const auto kc_raw = host(dkc, (size_t) kc_rows * KV * D), vc_raw = host(dvc, (size_t) kc_rows * KV * D);
            std::vector<float> got_k((size_t) T * KV * D), got_v((size_t) T * KV * D);
            std::vector<double> wk_exact((size_t) T * KV * D), wv_exact((size_t) T * KV * D);
            bool no_nan = true;
            for (int t = 0; t < T; ++t) {
                const int slot = cap > 0 ? (pos0 + t) % cap : (pos0 + t);
                for (size_t j = 0; j < (size_t) KV * D; ++j) {
                    const uint8_t bk = kc_raw[(size_t) slot * KV * D + j], bv = vc_raw[(size_t) slot * KV * D + j];
                    no_nan &= ((bk & 0x7F) != 0x7F) && ((bv & 0x7F) != 0x7F);
                    got_k[(size_t) t * KV * D + j] = h2f_host(hk_un[(size_t) t * KV * D + j]);
                    got_v[(size_t) t * KV * D + j] = h2f_host(hv_un[(size_t) t * KV * D + j]);
                    wk_exact[(size_t) t * KV * D + j] = (double) e4m3fn_to_f32_host(bk) * k_inv;
                    wv_exact[(size_t) t * KV * D + j] = (double) e4m3fn_to_f32_host(bv) * v_inv;
                }
            }
            expect_true(full ? "prep_qkv_r_fp8 full no NaN" : "prep_qkv_r_fp8 ring no NaN", no_nan);
            expect(full ? "prep_qkv_r_fp8 full q" : "prep_qkv_r_fp8 ring q", host(dq, wq.size()), wq, 2e-5);
            expect(full ? "prep_qkv_r_fp8 full r" : "prep_qkv_r_fp8 ring r", host(dr, wr.size()), wr, 2e-5);
            // E4M3 quantization accuracy (~1/16 max relative step)
            expect(full ? "prep_qkv_r_fp8 full k (e4m3)" : "prep_qkv_r_fp8 ring k (e4m3)", got_k, wk, 0.065, 2e-3);
            expect(full ? "prep_qkv_r_fp8 full v (e4m3)" : "prep_qkv_r_fp8 ring v (e4m3)", got_v, wv, 0.065, 2e-3);
            // unpack_kv_fp8_to_f16 exact FP16 decode of stored FP8 bytes
            expect(full ? "unpack_kv_fp8 full k" : "unpack_kv_fp8 ring k", got_k, wk_exact, 1e-3, 1e-5);
            expect(full ? "unpack_kv_fp8 full v" : "unpack_kv_fp8 ring v", got_v, wv_exact, 1e-3, 1e-5);
            for (float* p : {dq, dqc, dk, dv, dr, drc}) cudaFree(p);
            cudaFree(dkc), cudaFree(dvc), cudaFree(dpos), cudaFree(uk16), cudaFree(uv16);
        }

        // Scalar fallback for unpack_kv_fp8_to_f16 (KD=10, non-mult4, ring wraparound)
        const int nk = 7, KD_s = 10, cap_s = 5, k0_s = 3;
        std::vector<uint8_t> ks((size_t) cap_s * KD_s), vs((size_t) cap_s * KD_s);
        for (size_t i = 0; i < ks.size(); ++i) {
            ks[i] = (uint8_t) ((rng() & 0x7Eu) | ((rng() & 1u) << 7));
            vs[i] = (uint8_t) ((rng() & 0x7Eu) | ((rng() & 1u) << 7));
            if ((ks[i] & 0x7F) == 0x7F) ks[i] = 0x7E;
            if ((vs[i] & 0x7F) == 0x7F) vs[i] = 0x7E;
        }
        uint8_t *dks = dev(ks), *dvs = dev(vs);
        uint16_t *uk = dev_n<uint16_t>((size_t) nk * KD_s), *uv = dev_n<uint16_t>((size_t) nk * KD_s);
        k::unpack_kv_fp8_to_f16(dks, dvs, cap_s, k0_s, nk, KD_s, k_inv, v_inv, uk, uv, 0);
        const auto hk = host(uk, (size_t) nk * KD_s), hv = host(uv, (size_t) nk * KD_s);
        std::vector<float> gk(hk.size()), gv(hv.size());
        std::vector<double> wk(hk.size()), wv(hv.size());
        for (int j = 0; j < nk; ++j) {
            const int slot = (k0_s + j) % cap_s;
            for (int d = 0; d < KD_s; ++d) {
                const size_t idx = (size_t) j * KD_s + d;
                gk[idx] = h2f_host(hk[idx]);
                gv[idx] = h2f_host(hv[idx]);
                wk[idx] = (double) e4m3fn_to_f32_host(ks[(size_t) slot * KD_s + d]) * k_inv;
                wv[idx] = (double) e4m3fn_to_f32_host(vs[(size_t) slot * KD_s + d]) * v_inv;
            }
        }
        expect("unpack_kv_fp8 scalar k", gk, wk, 1e-3, 1e-5);
        expect("unpack_kv_fp8 scalar v", gv, wv, 1e-3, 1e-5);
        cudaFree(dks), cudaFree(dvs), cudaFree(uk), cudaFree(uv);
    }

    // 3. attn_decode_fp8 & attn_multi_fp8 (linear & ring buffer) vs double reference
    {
        const int H = 48, KV = 8, D = 128, d_rel = 16, ctx = 3100, kv_cap = 1056, window = 1024, nq = 8;
        const float k_inv = 1.0f / 16.0f, v_inv = 1.0f / 32.0f, scale = 1.0f / std::sqrt(128.0f);
        std::vector<uint8_t> kc_lin((size_t) ctx * KV * D), vc_lin((size_t) ctx * KV * D);
        for (size_t i = 0; i < kc_lin.size(); ++i) {
            uint8_t bk = (uint8_t) (rng() & 0xFFu), bv = (uint8_t) (rng() & 0xFFu);
            if ((bk & 0x7F) >= 0x7E) bk = (bk & 0x80u) | 0x60u;
            if ((bv & 0x7F) >= 0x7E) bv = (bv & 0x80u) | 0x60u;
            kc_lin[i] = bk;
            vc_lin[i] = bv;
        }
        std::vector<uint8_t> kc_ring((size_t) kv_cap * KV * D, 0), vc_ring((size_t) kv_cap * KV * D, 0);
        for (int pos = 1476; pos < 2510; ++pos) {
            const int slot = pos % kv_cap;
            std::memcpy(kc_ring.data() + (size_t) slot * KV * D, kc_lin.data() + (size_t) pos * KV * D,
                        (size_t) KV * D);
            std::memcpy(vc_ring.data() + (size_t) slot * KV * D, vc_lin.data() + (size_t) pos * KV * D,
                        (size_t) KV * D);
        }
        auto q = fvec((size_t) nq * H * D, 3.0f), r = fvec((size_t) nq * H * d_rel, 2.0f),
             sink = fvec((size_t) nq * H, 3.0f), proj = fvec((size_t) 1024 * d_rel);
        uint8_t *dk_lin = dev(kc_lin), *dv_lin = dev(vc_lin), *dk_ring = dev(kc_ring), *dv_ring = dev(vc_ring);
        float *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
        float *scratch = dev_n<float>((size_t) nq * H * (ctx + 32 * 130)), *out = dev_n<float>((size_t) nq * H * D),
              *one = dev_n<float>((size_t) H * D);

        for (int ring = 0; ring < 2; ++ring) {
            const int k0 = ring ? 1476 : 0, nk = ring ? 1024 : 2500, qpos = 2499, w = ring ? window : (1 << 30);
            k::attn_decode_fp8(dq, dr, ds, ring ? dk_ring : dk_lin, ring ? dv_ring : dv_lin, ring ? kv_cap : 0, H, KV,
                               D, k0, nk, qpos, scale * k_inv, v_inv, dp, d_rel, 1024, w, scratch, one, 0);
            std::vector<double> want((size_t) H * D);
            double mag = 0;   // the largest sum of |terms| of an output (the tolerance below)
            for (int h = 0; h < H; ++h) {
                const int g = h / (H / KV);
                std::vector<double> l(nk);
                double m = sink[h];
                for (int j = 0; j < nk; ++j) {
                    const int dist = qpos - (k0 + j);
                    if (dist < 0 || dist >= w) {
                        l[j] = -INFINITY;
                        continue;
                    }
                    double a = 0;
                    for (int d = 0; d < D; ++d)
                        a += (double) q[h * D + d] *
                             (double) e4m3fn_to_f32_host(kc_lin[((size_t) (k0 + j) * KV + g) * D + d]);
                    a *= (double) (scale * k_inv);
                    if (dist < 1024)
                        for (int e = 0; e < d_rel; ++e) a += (double) r[h * d_rel + e] * proj[(size_t) dist * d_rel + e];
                    l[j] = a;
                    m = std::max(m, a);
                }
                double sum = std::exp(sink[h] - m);
                for (int j = 0; j < nk; ++j) sum += std::exp(l[j] - m);
                for (int d = 0; d < D; ++d) {
                    double o = 0, oa = 0;
                    for (int j = 0; j < nk; ++j) {
                        const double t = std::exp(l[j] - m) / sum *
                                         (double) e4m3fn_to_f32_host(vc_lin[((size_t) (k0 + j) * KV + g) * D + d]);
                        o += t;
                        oa += std::fabs(t);
                    }
                    want[(size_t) h * D + d] = o * (double) v_inv;
                    mag = std::max(mag, oa * (double) v_inv);
                }
            }
            // (V spans E4M3's whole range, so an output is terms up to ~448 that cancel - F32 rounds each at ~6e-8,
            //  over up to 2500 of them ~sqrt(2500) x 6e-8: the absolute part follows the terms' size, 3e-6 of it)
            expect(ring ? "attn_decode_fp8 ring 1056" : "attn_decode_fp8 linear 2500", host(one, want.size()), want,
                   2e-4, std::max(1e-6, 3e-6 * mag));
        }

        const int pos0 = 2500, k0 = pos0 - window + 1, nk = pos0 + nq - k0;
        k::attn_multi_fp8(dq, dr, ds, dk_ring, dv_ring, kv_cap, nq, H, KV, k0, nk, pos0, scale * k_inv, v_inv, dp,
                           d_rel, 1024, window, scratch, out, 0);
        const auto got = host(out, (size_t) nq * H * D);
        std::vector<double> want((size_t) nq * H * D);
        for (int qi = 0; qi < nq; ++qi) {
            const int qp = pos0 + qi, kq = qp - window + 1;
            k::attn_decode_fp8(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H, dk_ring,
                               dv_ring, kv_cap, H, KV, D, kq, qp + 1 - kq, qp, scale * k_inv, v_inv, dp, d_rel, 1024,
                               window, scratch, one, 0);
            const auto o1 = host(one, (size_t) H * D);
            for (size_t j = 0; j < o1.size(); ++j) want[(size_t) qi * H * D + j] = o1[j];
        }
        expect("attn_multi_fp8 ring 1056", got, want, 1e-4, 1e-6);

        for (uint8_t* p : {dk_lin, dv_lin, dk_ring, dv_ring}) cudaFree(p);
        for (float* p : {dq, dr, ds, dp, scratch, out, one}) cudaFree(p);
    }

    // 4. attn_decode_fp8 with nk > 16384 (exercises P = 128 splits and kv_cap = 0 full-layer path)
    for (int unit_kv = 0; unit_kv < 2; ++unit_kv) {
        const int H = 48, KV = 8, D = 128, d_rel = 16, nk = 16640, qpos = nk - 1, w = 1 << 30;
        const float k_inv = unit_kv ? 1.0f : 1.0f / 16.0f, v_inv = unit_kv ? 1.0f : 1.0f / 32.0f;
        const float scale = 1.0f / std::sqrt(128.0f);
        std::vector<uint8_t> kc((size_t) nk * KV * D), vc((size_t) nk * KV * D);
        for (size_t i = 0; i < kc.size(); ++i) {
            uint8_t bk = (uint8_t) (rng() & 0xFFu), bv = (uint8_t) (rng() & 0xFFu);
            if (unit_kv) {
                // Realistic unit-scale KV magnitudes (<= 4.0) without 1e4x random-walk cancellation across 16640 keys.
                bk = (bk & 0x80u) | (uint8_t) (rng() % 0x49u);
                bv = (bv & 0x80u) | (uint8_t) (rng() % 0x49u);
            } else {
                if ((bk & 0x7F) >= 0x7E) bk = (bk & 0x80u) | 0x60u;
                if ((bv & 0x7F) >= 0x7E) bv = (bv & 0x80u) | 0x60u;
            }
            kc[i] = bk;
            vc[i] = bv;
        }
        auto q = fvec((size_t) H * D, 3.0f), r = fvec((size_t) H * d_rel, 2.0f), sink = fvec((size_t) H, 3.0f),
             proj = fvec((size_t) 1024 * d_rel);
        uint8_t *dk = dev(kc), *dv = dev(vc);
        float *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
        float *scratch = dev_n<float>((size_t) H * (nk + 128 * 130)), *one = dev_n<float>((size_t) H * D);
        k::attn_decode_fp8(dq, dr, ds, dk, dv, 0, H, KV, D, 0, nk, qpos, scale * k_inv, v_inv, dp, d_rel, 1024, w,
                           scratch, one, 0);
        std::vector<double> want((size_t) H * D);
        for (int h = 0; h < H; ++h) {
            const int g = h / (H / KV);
            std::vector<double> l(nk);
            double m = sink[h];
            for (int j = 0; j < nk; ++j) {
                const int dist = qpos - j;
                double a = 0;
                for (int d = 0; d < D; ++d)
                    a += (double) q[h * D + d] *
                         (double) e4m3fn_to_f32_host(kc[((size_t) j * KV + g) * D + d]);
                a *= (double) (scale * k_inv);
                if (dist < 1024)
                    for (int e = 0; e < d_rel; ++e) a += (double) r[h * d_rel + e] * proj[(size_t) dist * d_rel + e];
                l[j] = a;
                m = std::max(m, a);
            }
            double sum = std::exp(sink[h] - m);
            for (int j = 0; j < nk; ++j) sum += std::exp(l[j] - m);
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int j = 0; j < nk; ++j)
                    o += std::exp(l[j] - m) / sum *
                         (double) e4m3fn_to_f32_host(vc[((size_t) j * KV + g) * D + d]);
                want[(size_t) h * D + d] = o * (double) v_inv;
            }
        }
        expect(unit_kv ? "attn_decode_fp8 P=128 unit (nk=16640)" : "attn_decode_fp8 P=128 raw (nk=16640)",
               host(one, want.size()), want, 2e-4, unit_kv ? 1e-6 : 5e-6);
        for (uint8_t* p : {dk, dv}) cudaFree(p);
        for (float* p : {dq, dr, ds, dp, scratch, one}) cudaFree(p);
    }
}

void test_round6() {
    // 1. attn_res_rms_f16: C=4608 (vectorized float4/576-thread) and odd C=517 (scalar fallback), with optional w / y32
    for (int C : {4608, 517}) {
        const int T = 7, n = 9;
        for (int nq : {1, 3}) {
            std::vector<std::vector<float>> st(n);
            k::ResStates rs{};
            std::vector<float*> ds(n);
            for (int i = 0; i < n; ++i) st[i] = fvec((size_t) T * C, 1.0f + 0.5f * i), ds[i] = dev(st[i]), rs.st[i] = ds[i];
            auto q = fvec((size_t) nq * C, 2.5f);
            float* dq = dev(q);
            std::vector<std::vector<float>> w(nq);
            std::vector<float*> dw(nq, nullptr), dy32(nq, nullptr);
            std::vector<uint16_t*> dy16(nq, nullptr);
            k::RmsOuts outs{};
            for (int qi = 0; qi < nq; ++qi) {
                if (!(nq == 3 && qi == 1)) {   // leave qi==1 weight null in nq==3 case to exercise null-weight path
                    w[qi] = fvec(C, 2.0f);
                    dw[qi] = dev(w[qi]);
                }
                outs.w[qi] = dw[qi];
                dy16[qi] = dev_n<uint16_t>((size_t) T * C);
                outs.y16[qi] = dy16[qi];
                if (!(nq == 3 && qi == 2)) {   // leave qi==2 y32 null in nq==3 case
                    dy32[qi] = dev_n<float>((size_t) T * C);
                    outs.y32[qi] = dy32[qi];
                }
            }
            const float res_eps = 1e-6f, rms_eps = 1e-5f;
            k::attn_res_rms_f16(rs, n, dq, nq, T, C, res_eps, rms_eps, outs, 0);
            for (int qi = 0; qi < nq; ++qi) {
                std::vector<double> want_y((size_t) T * C);
                for (int t = 0; t < T; ++t) {
                    std::vector<double> r(n);
                    for (int i = 0; i < n; ++i) {
                        double ss = 0;
                        for (int d = 0; d < C; ++d) ss += (double) st[i][(size_t) t * C + d] * st[i][(size_t) t * C + d];
                        r[i] = 1.0 / std::sqrt(ss / C + res_eps);
                    }
                    std::vector<double> sc(n);
                    double m = -1e300, sum = 0;
                    for (int i = 0; i < n; ++i) {
                        double dot = 0;
                        for (int d = 0; d < C; ++d) dot += (double) q[(size_t) qi * C + d] * st[i][(size_t) t * C + d] * r[i];
                        sc[i] = dot / std::sqrt((double) C);
                        m = std::max(m, sc[i]);
                    }
                    for (int i = 0; i < n; ++i) sum += (sc[i] = std::exp(sc[i] - m));
                    std::vector<double> o(C);
                    double oo = 0;
                    for (int d = 0; d < C; ++d) {
                        double a = 0;
                        for (int i = 0; i < n; ++i) a += sc[i] / sum * st[i][(size_t) t * C + d] * r[i];
                        o[d] = a;
                        oo += a * a;
                    }
                    const double ir = 1.0 / std::sqrt(oo / C + rms_eps);
                    for (int d = 0; d < C; ++d)
                        want_y[(size_t) t * C + d] = o[d] * ir * (dw[qi] ? (double) w[qi][d] : 1.0);
                }
                const auto h16 = host(dy16[qi], want_y.size());
                std::vector<float> got16(want_y.size());
                for (size_t j = 0; j < want_y.size(); ++j) got16[j] = h2f_host(h16[j]);
                char label[96];
                std::snprintf(label, sizeof label, "attn_res_rms_f16 C=%d nq=%d qi=%d y16", C, nq, qi);
                expect(label, got16, want_y, 2e-3, 1e-4);
                if (dy32[qi]) {
                    std::snprintf(label, sizeof label, "attn_res_rms_f16 C=%d nq=%d qi=%d y32", C, nq, qi);
                    expect(label, host(dy32[qi], want_y.size()), want_y, 1e-4, 5e-6);
                }
            }
            for (auto* p : ds) cudaFree(p);
            for (int qi = 0; qi < nq; ++qi) cudaFree(dw[qi]), cudaFree(dy16[qi]), cudaFree(dy32[qi]);
            cudaFree(dq);
        }
    }

    // 2. conv4_res: static/dynamic with and without residual across T=1 and T=41
    for (int T : {1, 41}) {
        const int C = 1024, heads = 8;
        auto x = fvec((size_t) T * C), w = fvec((size_t) C * 4), dyn = fvec((size_t) T * heads * 4),
             res = fvec((size_t) T * C, 2.0f);
        float *dx = dev(x), *dw = dev(w), *dd = dev(dyn), *dres = dev(res), *dy = dev_n<float>((size_t) T * C);
        for (int with_dyn : {0, 1}) {
            const auto base = conv_ref(x, w, with_dyn ? &dyn : nullptr, heads, T, C);
            for (int with_res : {0, 1}) {
                std::vector<double> want(base.size());
                for (size_t i = 0; i < base.size(); ++i) want[i] = base[i] + (with_res ? (double) res[i] : 0.0);
                float* hist = dev_n<float>((size_t) 3 * C);
                k::conv4_res(dx, hist, dw, with_dyn ? dd : nullptr, heads, T, C, with_res ? dres : nullptr, dy, 0);
                char label[80];
                std::snprintf(label, sizeof label, "conv4_res T=%d %s %s", T, with_dyn ? "dyn" : "static",
                              with_res ? "+res" : "no-res");
                expect(label, host(dy, want.size()), want, 2e-5);
                cudaFree(hist);
            }
        }
        cudaFree(dx), cudaFree(dw), cudaFree(dd), cudaFree(dres), cudaFree(dy);
    }

    // 3. attn_decode_* & attn_multi_* across split boundaries nk = 10, 64, 65, 200, 1056, 2500, 16640 (F32, F16, FP8)
    {
        const int H = 48, KV = 8, D = 128, d_rel = 16, ctx = 16640, nq = 4;
        const float scale = 1.0f / std::sqrt(128.0f);
        auto kc32 = fvec((size_t) ctx * KV * D, 1.5f), vc32 = fvec((size_t) ctx * KV * D, 1.5f);
        std::vector<uint16_t> kc16(kc32.size()), vc16(vc32.size());
        std::vector<uint8_t> kc8(kc32.size()), vc8(vc32.size());
        for (size_t i = 0; i < kc32.size(); ++i) {
            kc16[i] = f2h_host(kc32[i]);
            vc16[i] = f2h_host(vc32[i]);
            kc8[i] = (uint8_t) ((rng() & 0x80u) | (uint8_t) (rng() % 0x49u));   // unit-scale finite E4M3 <= 4.0
            vc8[i] = (uint8_t) ((rng() & 0x80u) | (uint8_t) (rng() % 0x49u));
        }
        auto q = fvec((size_t) nq * H * D, 2.0f), r = fvec((size_t) nq * H * d_rel, 1.5f),
             sink = fvec((size_t) nq * H, 2.5f), proj = fvec((size_t) 1024 * d_rel, 1.0f);
        float *dk32 = dev(kc32), *dv32 = dev(vc32), *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
        uint16_t *dk16 = dev(kc16), *dv16 = dev(vc16);
        uint8_t *dk8 = dev(kc8), *dv8 = dev(vc8);
        float *scratch = dev_n<float>((size_t) nq * H * (ctx + 128 * 130)), *one = dev_n<float>((size_t) H * D),
              *multi = dev_n<float>((size_t) nq * H * D);

        for (int nk : {10, 64, 65, 200, 1056, 2500, 16640}) {
            const int qpos = nk - 1, w = nk == 1056 ? 512 : (1 << 30);   // exercise sliding mask on nk=1056 too
            for (int kind = 0; kind < 3; ++kind) {
                if (kind == 0)
                    k::attn_decode(dq, dr, ds, dk32, dv32, H, KV, D, 0, nk, qpos, scale, dp, d_rel, 1024, w, scratch,
                                   one, 0);
                else if (kind == 1)
                    k::attn_decode_f16(dq, dr, ds, dk16, dv16, 0, H, KV, D, 0, nk, qpos, scale, dp, d_rel, 1024, w,
                                       scratch, one, 0);
                else
                    k::attn_decode_fp8(dq, dr, ds, dk8, dv8, 0, H, KV, D, 0, nk, qpos, scale, 1.0f, dp, d_rel, 1024, w,
                                       scratch, one, 0);
                std::vector<double> want((size_t) H * D);
                for (int h = 0; h < H; ++h) {
                    const int g = h / (H / KV);
                    std::vector<double> l(nk);
                    double m = sink[h];
                    for (int j = 0; j < nk; ++j) {
                        const int dist = qpos - j;
                        if (dist < 0 || dist >= w) {
                            l[j] = -INFINITY;
                            continue;
                        }
                        double a = 0;
                        for (int d = 0; d < D; ++d) {
                            const size_t idx = ((size_t) j * KV + g) * D + d;
                            const double kv = kind == 0 ? kc32[idx]
                                            : kind == 1 ? h2f_host(kc16[idx])
                                                        : e4m3fn_to_f32_host(kc8[idx]);
                            a += (double) q[h * D + d] * kv;
                        }
                        a *= scale;
                        if (dist < 1024)
                            for (int e = 0; e < d_rel; ++e) a += (double) r[h * d_rel + e] * proj[(size_t) dist * d_rel + e];
                        l[j] = a;
                        m = std::max(m, a);
                    }
                    double sum = std::exp(sink[h] - m);
                    for (int j = 0; j < nk; ++j) sum += std::exp(l[j] - m);
                    for (int d = 0; d < D; ++d) {
                        double o = 0;
                        for (int j = 0; j < nk; ++j) {
                            const size_t idx = ((size_t) j * KV + g) * D + d;
                            const double vv = kind == 0 ? vc32[idx]
                                            : kind == 1 ? h2f_host(vc16[idx])
                                                        : e4m3fn_to_f32_host(vc8[idx]);
                            o += std::exp(l[j] - m) / sum * vv;
                        }
                        want[(size_t) h * D + d] = o;
                    }
                }
                const char* knames[] = {"f32", "f16", "fp8"};
                char label[64];
                std::snprintf(label, sizeof label, "attn_decode_%s nk=%d", knames[kind], nk);
                expect(label, host(one, want.size()), want, 2e-4, 1e-6);
            }
        }
        for (int nk : {64, 65, 1056, 2500}) {
            const int pos0 = nk - nq, w = nk == 1056 ? 512 : (1 << 30);
            for (int kind = 0; kind < 3; ++kind) {
                if (kind == 0)
                    k::attn_multi(dq, dr, ds, dk32, dv32, nq, H, KV, 0, nk, pos0, scale, dp, d_rel, 1024, w, scratch,
                                  multi, 0);
                else if (kind == 1)
                    k::attn_multi_f16(dq, dr, ds, dk16, dv16, 0, nq, H, KV, 0, nk, pos0, scale, dp, d_rel, 1024, w,
                                      scratch, multi, 0);
                else
                    k::attn_multi_fp8(dq, dr, ds, dk8, dv8, 0, nq, H, KV, 0, nk, pos0, scale, 1.0f, dp, d_rel, 1024, w,
                                      scratch, multi, 0);
                const auto got = host(multi, (size_t) nq * H * D);
                std::vector<double> want((size_t) nq * H * D);
                for (int qi = 0; qi < nq; ++qi) {
                    const int qp = pos0 + qi;
                    if (kind == 0)
                        k::attn_decode(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H,
                                       dk32, dv32, H, KV, D, 0, qp + 1, qp, scale, dp, d_rel, 1024, w, scratch, one, 0);
                    else if (kind == 1)
                        k::attn_decode_f16(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H,
                                           dk16, dv16, 0, H, KV, D, 0, qp + 1, qp, scale, dp, d_rel, 1024, w, scratch,
                                           one, 0);
                    else
                        k::attn_decode_fp8(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H,
                                           dk8, dv8, 0, H, KV, D, 0, qp + 1, qp, scale, 1.0f, dp, d_rel, 1024, w,
                                           scratch, one, 0);
                    const auto o1 = host(one, (size_t) H * D);
                    for (size_t j = 0; j < o1.size(); ++j) want[(size_t) qi * H * D + j] = o1[j];
                }
                const char* knames[] = {"f32", "f16", "fp8"};
                char label[64];
                std::snprintf(label, sizeof label, "attn_multi_%s nk=%d", knames[kind], nk);
                expect(label, got, want, 1e-4, 1e-6);
            }
        }
        for (float* p : {dk32, dv32, dq, dr, ds, dp, scratch, one, multi}) cudaFree(p);
        cudaFree(dk16), cudaFree(dv16), cudaFree(dk8), cudaFree(dv8);
    }

    // 4. route: E=384 (route_warp_kernel) and E=600 (fallback route_kernel), with exact score ties
    for (int E : {384, 600}) {
        const int T = 64, K = 8;
        auto lg = fvec((size_t) T * E, 6.0f), bias = fvec(E, 0.2f);
        // Inject 10 exact duplicate top-scoring ties so K=8 must deterministically pick the 8 lowest indices
        for (int e : {5, 37, 101, 199, 255, 320, 350, 360, 370, 383}) bias[e] = 5.0f;
        for (int t = 0; t < T; ++t) {
            const float shared_lg = 2.0f + 0.05f * (t & 7);
            for (int e : {5, 37, 101, 199, 255, 320, 350, 360, 370, 383}) lg[(size_t) t * E + e] = shared_lg;
        }
        float *dl = dev(lg), *db = dev(bias), *dw = dev_n<float>((size_t) T * K);
        int* dt = dev_n<int>((size_t) T * K);
        k::route(dl, db, T, E, K, 8.0f, dt, dw, 0);
        const auto top = host(dt, (size_t) T * K);
        const auto w = host(dw, (size_t) T * K);
        bool picks_ok = true;
        std::vector<double> want_w((size_t) T * K);
        for (int t = 0; t < T; ++t) {
            std::vector<std::pair<float, int>> v(E);
            for (int e = 0; e < E; ++e) {
                const float sg = 1.0f / (1.0f + std::exp(-lg[(size_t) t * E + e]));
                v[e] = {-(sg + bias[e]), e};
            }
            std::sort(v.begin(), v.end());
            std::vector<int> pick(K);
            for (int i = 0; i < K; ++i) pick[i] = v[i].second;
            std::sort(pick.begin(), pick.end());
            double sum = 0;
            for (int i = 0; i < K; ++i) sum += 1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + pick[i]]));
            for (int i = 0; i < K; ++i) {
                picks_ok &= top[(size_t) t * K + i] == pick[i];
                want_w[(size_t) t * K + i] =
                    1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + pick[i]])) / (sum + 1e-20) * 8.0;
            }
        }
        char label[64];
        std::snprintf(label, sizeof label, "route ties E=%d picks", E);
        expect_true(label, picks_ok);
        if (picks_ok) {
            std::snprintf(label, sizeof label, "route ties E=%d weights", E);
            expect(label, w, want_w, 2e-5);
        }
        cudaFree(dl), cudaFree(db), cudaFree(dw), cudaFree(dt);
    }

    // 5. experts_gate_up_swiglu & experts_down (row_dot_2 + odd C=513 + launch thresholds S=1,3,5,8)
    {
        const int C = 513, F = 256, E = 3, S_max = 8;   // odd C exercises tail row in row_dot_2
        const int types[][3] = {{k::kNVFP4, k::kNVFP4, k::kNVFP4},
                                {k::kQ8_0, k::kQ8_0, k::kQ8_0},
                                {k::kQ5_K, k::kQ5_K, k::kQ5_K},
                                {k::kQ6_K, k::kQ6_K, k::kQ6_K},
                                {k::kIQ3_XXS, k::kIQ3_XXS, k::kNVFP4},
                                {k::kIQ4_XS, k::kIQ4_XS, k::kQ6_K}};
        const char* tnames[] = {"NVFP4", "Q8_0", "Q5_K", "Q6_K", "IQ3_XXS/NVFP4", "IQ4_XS/Q6_K"};
        for (int ti = 0; ti < 6; ++ti) {
            const int tg = types[ti][0], tu = types[ti][1], td = types[ti][2];
            const int Cin = 512;   // weight row width for gate/up (multiple of 256 for Q5_K/Q6_K)
            auto g = rand_blocks(tg, (size_t) E * F * Cin), u = rand_blocks(tu, (size_t) E * F * Cin),
                 d = rand_blocks(td, (size_t) E * C * F);
            const size_t eb0 = g.size() / E, eb1 = u.size() / E, eb2 = d.size() / E;
            const size_t slot = (eb0 + eb1 + eb2 + 255) / 256 * 256;
            std::vector<uint8_t> pool(slot * E);
            for (int e = 0; e < E; ++e) {
                std::memcpy(pool.data() + e * slot, g.data() + e * eb0, eb0);
                std::memcpy(pool.data() + e * slot + eb0, u.data() + e * eb1, eb1);
                std::memcpy(pool.data() + e * slot + eb0 + eb1, d.data() + e * eb2, eb2);
            }
            uint8_t* dpool = dev(pool);
            const int pick[S_max] = {2, 0, 2, 1, 0, 1, 2, 0}, tok[S_max] = {0, 1, 1, 2, 0, 2, 1, 0};
            std::vector<const uint8_t*> blob(S_max);
            for (int i = 0; i < S_max; ++i) blob[i] = dpool + pick[i] * slot;
            const uint8_t** dblob = nullptr;
            CU(cudaMalloc(&dblob, S_max * sizeof(void*)));
            CU(cudaMemcpy(dblob, blob.data(), S_max * sizeof(void*), cudaMemcpyHostToDevice));
            auto x = fvec((size_t) 3 * Cin, 1.5f), sg = fvec(S_max, 0.5f), su = fvec(S_max, 0.5f);
            float *dx = dev(x), *dgu = dev_n<float>((size_t) S_max * 2 * F), *dsg = dev(sg), *dsu = dev(su),
                  *dy = dev_n<float>((size_t) S_max * C);
            int* dtok = dev(std::vector<int>(tok, tok + S_max));
            std::vector<uint16_t> h((size_t) S_max * F);
            std::vector<float> hf(h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                h[i] = (uint16_t) ((rng() & 0x3BFF) | ((rng() & 1) << 15));
                hf[i] = h2f_host(h[i]);
            }
            uint16_t *dh_in = dev(h), *dh_ref = dev_n<uint16_t>((size_t) S_max * F),
                     *dh_got = dev_n<uint16_t>((size_t) S_max * F);
            std::vector<double> want_y((size_t) S_max * C);
            std::vector<float> row(F);
            for (int i = 0; i < S_max; ++i) {
                const int e = pick[i];
                for (int r = 0; r < C; ++r) {
                    k::decode_host(td, d.data() + (size_t) e * eb2 + (size_t) r * (eb2 / C), F, row.data());
                    double a = 0;
                    for (int c = 0; c < F; ++c) a += (double) row[c] * hf[(size_t) i * F + c];
                    want_y[(size_t) i * C + r] = a;
                }
            }
            for (int S : {1, 3, 5, 8}) {
                k::experts_gate_up(dblob, dtok, S, tg, tu, eb0, dx, Cin, F, dgu, 0);
                k::swiglu_cap_f32(dgu, S, F, dsg, dsu, 20.0f, 20.0f, dh_ref, 0);
                k::experts_gate_up_swiglu(dblob, dtok, S, tg, tu, eb0, dx, Cin, F, dsg, dsu, 20.0f, 20.0f, dh_got, 0);
                const auto r16 = host(dh_ref, (size_t) S * F), g16 = host(dh_got, (size_t) S * F);
                std::vector<float> fg(g16.size());
                std::vector<double> wr(r16.size());
                for (size_t j = 0; j < g16.size(); ++j) fg[j] = h2f_host(g16[j]), wr[j] = h2f_host(r16[j]);
                char label[80];
                std::snprintf(label, sizeof label, "r6 experts_gu_swiglu S=%d %s", S, tnames[ti]);
                expect(label, fg, wr, 2e-3, 1e-3);

                k::experts_down(dblob, S, td, eb0 + eb1, dh_in, C, F, dy, 0);
                std::snprintf(label, sizeof label, "r6 experts_down C=513 S=%d %s", S, tnames[ti]);
                expect(label, host(dy, (size_t) S * C),
                       std::vector<double>(want_y.begin(), want_y.begin() + (size_t) S * C), 2e-4, 2e-3);
            }
            for (void* p : {(void*) dpool, (void*) dblob, (void*) dx, (void*) dgu, (void*) dsg, (void*) dsu,
                            (void*) dy, (void*) dtok, (void*) dh_in, (void*) dh_ref, (void*) dh_got})
                cudaFree(p);
        }
    }
}

}  // namespace

// One token's NVFP4 experts through the tensor cores against the scalar kernels over the same blocks (every scale
// byte, the subnormal ones and 0x7F with them): gate | up with the input as two FP16 limbs, down over the FP16 h.
void test_experts_mma() {
    constexpr int C = 256, F = 128, S = 5;
    const int ty = k::kNVFP4;
    const size_t matrix_bytes = (size_t) C * F / k::block_values(ty) * k::block_bytes(ty);
    auto pool = rand_blocks(ty, (size_t) S * 3 * C * F);
    for (size_t off = 0, i = 0; off < pool.size(); off += 36, ++i)   // a scale of each kind somewhere
        if (i % 7 == 0) pool[off + i % 4] = (uint8_t) (i % 13 == 0 ? 0x7F : i % 8);
    auto* device_pool = dev(pool);
    std::vector<const uint8_t*> ptrs;
    for (int j = 0; j < S; ++j) ptrs.push_back(device_pool + 3 * j * matrix_bytes);
    auto** blobs = dev(ptrs);
    auto* input = dev(fvec(C, 3.0f));
    auto* tokens = dev(std::vector<int>(S, 0));
    auto sgv = fvec(S), suv = fvec(S);
    auto *sg = dev(sgv), *su = dev(suv);
    auto* limbs = dev_n<uint16_t>(2 * C);
    auto *ref = dev_n<uint16_t>(S * F), *got = dev_n<uint16_t>(S * F);
    auto *y_ref = dev_n<float>(S * C), *y_got = dev_n<float>(S * C);
    k::experts_gate_up_swiglu(blobs, tokens, S, ty, ty, matrix_bytes, input, C, F, sg, su, 20, 20, ref, 0);
    k::to_f16_limbs(input, C, limbs, 0);
    k::experts_gate_up_swiglu_mma(blobs, S, matrix_bytes, limbs, C, F, sg, su, 20, 20, got, 0);
    const auto h_ref = host(ref, S * F), h_got = host(got, S * F);
    std::vector<double> expected(h_ref.size());
    std::vector<float> actual(h_ref.size());
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = h2f_host(h_ref[i]), actual[i] = h2f_host(h_got[i]);
    expect("experts_gate_up_swiglu_mma", actual, expected, 1e-3, 1e-3);
    k::experts_down(blobs, S, ty, 2 * matrix_bytes, ref, C, F, y_ref, 0);
    k::experts_down_mma(blobs, S, 2 * matrix_bytes, ref, C, F, y_got, 0);
    const auto want = host(y_ref, S * C);
    expect("experts_down_mma", host(y_got, S * C), std::vector<double>(want.begin(), want.end()), 2e-5, 0.25);   // (random scales reach 240: sums of terms of 1e4 and more that cancel)
    expect_true("experts_mma_fits", k::experts_mma_fits(ty, ty, ty, 2304, 1152) && !k::experts_mma_fits(ty, k::kQ8_0, ty, 2304, 1152));
}

int main() {
    int dev_count = 0;
    CU(cudaGetDeviceCount(&dev_count));
    cudaDeviceProp p{};
    CU(cudaGetDeviceProperties(&p, 0));
    std::printf("ik2_kernels_test on %s (sm_%d%d)\n", p.name, p.major, p.minor);
    test_rms();
    test_l2clip();
    test_conv();
    test_attn_res();
    test_attn_probs();
    test_route();
    test_moe_pieces();
    test_decode();
    test_iq_ref();
    test_experts_mv();
    test_attn_decode();
    test_mv_blocks();
    test_experts_grouped();
    test_attn_multi();
    test_argmax();
    test_attn_f16();
    test_round2_fused();
    test_fp8_kv();
    test_round6();
    test_attn_flash();   // (last: its random data leaves the others' as they were)
    test_attn_decode_mma();
    test_cand_topk();
    test_token_path();
    test_experts_mma();
    std::printf(g_fail ? "FAILED: %d check(s)\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
