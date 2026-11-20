// tests/hy4_kernels_test.cu - maya-hy4's own kernels (src/hy4/hy4_kernels.cu) against double-precision references on the
// host (no model): the hyper-connections' pre-mix (with its norm folded in), post-mix and head; RoPE and the latent
// cache rows; the MLA attention with sinks over a window ring and over a key list; the router; the indexer's pooled
// keys, scores and picks (ties included); Engram's rows and mix; a row's top token (row_top).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../src/hy4/hy4_kernels.cuh"

namespace {

int fails = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                          \
        }                                                                     \
    } while (0)

std::mt19937 rng(42);
std::vector<float> randn(size_t n, float sd = 1.0f) {
    std::normal_distribution<float> d(0.0f, sd);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}
template <typename T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(1, h.size()) * sizeof(T));
    if (!h.empty()) cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <typename T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
double rel(const std::vector<float>& a, const std::vector<double>& r) {
    double d2 = 0, r2 = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d2 += (a[i] - r[i]) * (a[i] - r[i]);
        r2 += r[i] * r[i];
    }
    return std::sqrt(d2 / std::max(r2, 1e-300));
}
double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
// rel of what was added to x: (a - x) against (r - x)
double rel_delta(const std::vector<float>& a, const std::vector<double>& r, const std::vector<float>& x) {
    std::vector<float> da(a.size());
    std::vector<double> dr(r.size());
    for (size_t i = 0; i < a.size(); ++i) {
        da[i] = (float) ((double) a[i] - x[i]);
        dr[i] = r[i] - x[i];
    }
    return rel(da, dr);
}
// ggml's NORM rope of a pair i of n_rot at position p
void rope_ref(double& a, double& b, double p, int i, int n_rot, double base) {
    const double th = p * std::pow(base, -2.0 * i / n_rot), c = std::cos(th), s = std::sin(th);
    const double x0 = a, x1 = b;
    a = x0 * c - x1 * s;
    b = x0 * s + x1 * c;
}
float h2f(__half h) { return __half2float(h); }

void test_hc() {
    const int T = 3, hc = 4, E = 3072, HE = hc * E;
    auto x = randn((size_t) T * HE), fn = randn((size_t) 2 * hc * HE, 0.02f), base = randn(2 * hc, 0.5f),
         scale = randn(2, 0.5f), nw = randn(E, 0.3f), y = randn((size_t) T * E), pw = randn(E, 0.3f);
    for (float& v : nw) v += 1.0f;
    for (float& v : pw) v += 1.0f;
    const float eps = 1e-5f, hc_eps = 1e-6f, mag = 2.0f;
    float *dx = dev(x), *dfn = dev(fn), *db = dev(base), *ds = dev(scale), *dnw = dev(nw), *dpw = dev(pw), *dy = dev(y);
    float *dcur, *dpost, *dx2;
    cudaMalloc(&dcur, sizeof(float) * T * E);
    cudaMalloc(&dpost, sizeof(float) * T * hc);
    hy4::k::hc_pre(dx, T, hc, E, dfn, ds, db, eps, hc_eps, mag, dcur, dpost, 0, dnw, eps);
    std::vector<double> rcur((size_t) T * E), rpost((size_t) T * hc), rpre((size_t) T * hc);
    for (int t = 0; t < T; ++t) {
        const float* xt = x.data() + (size_t) t * HE;
        double ss = 0;
        for (int i = 0; i < HE; ++i) ss += (double) xt[i] * xt[i];
        const double r = 1.0 / std::sqrt(ss / HE + eps);
        double mix[8];
        for (int j = 0; j < 2 * hc; ++j) {
            double a = 0;
            for (int i = 0; i < HE; ++i) a += (double) fn[(size_t) j * HE + i] * xt[i];
            mix[j] = a * r;
        }
        for (int j = 0; j < hc; ++j) {
            rpre[(size_t) t * hc + j] = sigm(mix[j] * scale[0] + base[j]) + hc_eps;
            rpost[(size_t) t * hc + j] = sigm(mix[hc + j] * scale[1] + base[hc + j]) * mag + hc_eps;
        }
        std::vector<double> c((size_t) E);
        double s2 = 0;
        for (int d = 0; d < E; ++d) {
            double v = 0;
            for (int j = 0; j < hc; ++j) v += rpre[(size_t) t * hc + j] * xt[(size_t) j * E + d];
            c[(size_t) d] = v;
            s2 += v * v;
        }
        const double r2 = 1.0 / std::sqrt(s2 / E + eps);
        for (int d = 0; d < E; ++d) rcur[(size_t) t * E + d] = c[(size_t) d] * r2 * nw[(size_t) d];
    }
    const double e1 = rel(host(dcur, (size_t) T * E), rcur), e2 = rel(host(dpost, (size_t) T * hc), rpost);
    std::printf("  hc_pre (+ norm): cur rel %.2e, post rel %.2e\n", e1, e2);
    CHECK(e1 < 1e-5 && e2 < 1e-6);
    // post-mix with the post norm
    dx2 = dev(x);
    hy4::k::hc_post_norm(dx2, T, hc, E, dy, dpost, dpw, eps, 0);
    std::vector<double> rx((size_t) T * HE);
    const std::vector<float> post = host(dpost, (size_t) T * hc);
    for (int t = 0; t < T; ++t) {
        double ss = 0;
        for (int d = 0; d < E; ++d) ss += (double) y[(size_t) t * E + d] * y[(size_t) t * E + d];
        const double r = 1.0 / std::sqrt(ss / E + eps);
        for (int j = 0; j < hc; ++j)
            for (int d = 0; d < E; ++d)
                rx[((size_t) t * hc + j) * E + d] =
                    x[((size_t) t * hc + j) * E + d] + y[(size_t) t * E + d] * r * pw[(size_t) d] * post[(size_t) t * hc + j];
    }
    const double e3 = rel_delta(host(dx2, (size_t) T * HE), rx, x);
    std::printf("  hc_post_norm: rel %.2e\n", e3);
    CHECK(e3 < 1e-6);
    cudaFree(dx);
    cudaFree(dfn);
    cudaFree(db);
    cudaFree(ds);
    cudaFree(dnw);
    cudaFree(dpw);
    cudaFree(dy);
    cudaFree(dcur);
    cudaFree(dpost);
    cudaFree(dx2);
}

// the latent rows and the attention over them: a window ring (positions mod R) and a key list
void test_attention() {
    const int T = 5, H = 32, KL = 512, R = 64, KR = KL + R, ring = 40, n_swa = 16;
    const float eps = 1e-5f, base = 1e4f, scale = 1.0f / 16.0f;
    const int p0 = 50;   // (positions 50 .. 54; the ring holds 40, the window 16)
    std::vector<int> pos(T);
    for (int t = 0; t < T; ++t) pos[(size_t) t] = p0 + t;
    // the cache: rows of positions p0 - 30 .. p0 + T - 1 written by kv_store (earlier ones a chunk before)
    const int first = p0 - 30, n_all = p0 + T - first;
    auto kv = randn((size_t) n_all * KR), w = randn(KL, 0.2f);
    for (float& v : w) v += 1.0f;
    std::vector<int> all_pos((size_t) n_all);
    for (int i = 0; i < n_all; ++i) all_pos[(size_t) i] = first + i;
    uint16_t* cache;
    cudaMalloc(&cache, sizeof(uint16_t) * ring * KR);
    float *dkv = dev(kv), *dw = dev(w);
    int* dpos_all = dev(all_pos);
    hy4::k::kv_store(dkv, n_all, KL, R, dw, eps, dpos_all, base, cache, ring, 0);
    // the reference rows (as F16, the cache's)
    std::vector<double> rows((size_t) n_all * KR);
    for (int i = 0; i < n_all; ++i) {
        const float* x = kv.data() + (size_t) i * KR;
        double ss = 0;
        for (int d = 0; d < KL; ++d) ss += (double) x[d] * x[d];
        const double r = 1.0 / std::sqrt(ss / KL + eps);
        for (int d = 0; d < KL; ++d) rows[(size_t) i * KR + d] = h2f(__float2half_rn((float) (x[d] * r * w[(size_t) d])));
        for (int k = 0; k < R / 2; ++k) {
            double a = x[KL + 2 * k], b = x[KL + 2 * k + 1];
            rope_ref(a, b, all_pos[(size_t) i], k, R, base);
            rows[(size_t) i * KR + KL + 2 * k] = h2f(__float2half_rn((float) a));
            rows[(size_t) i * KR + KL + 2 * k + 1] = h2f(__float2half_rn((float) b));
        }
    }
    {   // the cache's rows against them
        std::vector<uint16_t> c = host(cache, (size_t) ring * KR);
        double md = 0;
        for (int i = std::max(0, n_all - ring); i < n_all; ++i)
            for (int d = 0; d < KR; ++d) {
                __half hv;
                std::memcpy(&hv, &c[(size_t) (all_pos[(size_t) i] % ring) * KR + d], 2);
                md = std::max(md, std::fabs((double) h2f(hv) - rows[(size_t) i * KR + d]));
            }
        std::printf("  kv_store: max diff %.2e\n", md);
        CHECK(md < 2e-3);
    }
    auto q = randn((size_t) T * H * KR, 0.1f), sinks = randn(H, 1.0f);
    float *dq = dev(q), *dsink = dev(sinks), *out, *ws;
    cudaMalloc(&out, sizeof(float) * T * H * KL);
    cudaMalloc(&ws, sizeof(float) * hy4::k::attn_ws_floats(T, H, KL, 64));
    auto ref = [&](const std::vector<std::vector<int>>& keys) {
        std::vector<double> o((size_t) T * H * KL, 0.0);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h) {
                const float* qh = q.data() + ((size_t) t * H + h) * KR;
                std::vector<double> s;
                for (int p : keys[(size_t) t]) {
                    const double* row = rows.data() + (size_t) (p - first) * KR;
                    double v = 0;
                    for (int d = 0; d < KR; ++d) v += qh[d] * row[d];
                    s.push_back(v * scale);
                }
                double m = sinks[(size_t) h];
                for (double v : s) m = std::max(m, v);
                double den = std::exp(sinks[(size_t) h] - m);
                for (double v : s) den += std::exp(v - m);
                for (size_t k = 0; k < s.size(); ++k) {
                    const double pk = std::exp(s[k] - m) / den;
                    const double* row = rows.data() + (size_t) (keys[(size_t) t][k] - first) * KR;
                    for (int d = 0; d < KL; ++d) o[((size_t) t * H + h) * KL + d] += pk * row[d];
                }
            }
        return o;
    };
    {   // the window: the last n_swa positions up to each token's own, from the ring
        std::vector<int> lo(T), nk(T);
        std::vector<std::vector<int>> keys(T);
        for (int t = 0; t < T; ++t) {
            lo[(size_t) t] = std::max(0, pos[(size_t) t] - n_swa + 1);
            nk[(size_t) t] = pos[(size_t) t] - lo[(size_t) t] + 1;
            for (int p = lo[(size_t) t]; p <= pos[(size_t) t]; ++p) keys[(size_t) t].push_back(p);
        }
        int *dlo = dev(lo), *dnk = dev(nk);
        hy4::k::attention(dq, T, H, KL, R, cache, ring, nullptr, 0, dlo, dnk, n_swa, scale, dsink, ws, out, 0);
        const double e = rel(host(out, (size_t) T * H * KL), ref(keys));
        std::printf("  attention, window ring: rel %.2e\n", e);
        CHECK(e < 1e-5);
        cudaFree(dlo);
        cudaFree(dnk);
    }
    {   // a key list: scattered positions in the ring's range (ring 0 reads row = position: the full cache's way)
        uint16_t* full;
        cudaMalloc(&full, sizeof(uint16_t) * (size_t) (p0 + T) * KR);
        hy4::k::kv_store(dkv, n_all, KL, R, dw, eps, dpos_all, base, full, 0, 0);
        const int ldk = 24;
        std::vector<int> kl((size_t) T * ldk, 0), nk(T);
        std::vector<std::vector<int>> keys(T);
        std::uniform_int_distribution<int> pick(first, p0 - 1);
        for (int t = 0; t < T; ++t) {
            std::vector<int> v;
            for (int k = 0; k < 17 + t; ++k) v.push_back(pick(rng));
            v.push_back(pos[(size_t) t]);
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            nk[(size_t) t] = (int) v.size();
            for (size_t k = 0; k < v.size(); ++k) kl[(size_t) t * ldk + k] = v[k];
            keys[(size_t) t] = v;
        }
        int *dk = dev(kl), *dnk = dev(nk);
        hy4::k::attention(dq, T, H, KL, R, full, 0, dk, ldk, nullptr, dnk, ldk, scale, dsink, ws, out, 0);
        const double e = rel(host(out, (size_t) T * H * KL), ref(keys));
        std::printf("  attention, key list: rel %.2e\n", e);
        CHECK(e < 1e-5);
        cudaFree(full);
        cudaFree(dk);
        cudaFree(dnk);
    }
    {   // the 8-bit cache (--kv q8): the latent within half a step of its values, the rotated part F16's, read back by
        // kv_deq; the attention over them the reference's over the dequantized rows (window ring and key list), and
        // near the F16 cache's
        const size_t rb = hy4::k::kv_q8_row_bytes(KL, R);
        CHECK(rb == 672);
        uint16_t *q8r, *q8f, *deq;
        cudaMalloc(&q8r, rb * ring);
        cudaMalloc(&q8f, rb * (size_t) (p0 + T));
        cudaMalloc(&deq, sizeof(uint16_t) * (size_t) n_all * KR);
        hy4::k::kv_store(dkv, n_all, KL, R, dw, eps, dpos_all, base, q8r, ring, 0, true);
        hy4::k::kv_store(dkv, n_all, KL, R, dw, eps, dpos_all, base, q8f, 0, 0, true);
        // the ring's rows (the last `ring` positions) and the full cache's, as F16
        const int nr = std::min(ring, n_all), r0 = first + n_all - nr;
        hy4::k::kv_deq(q8r, ring, KL, R, r0, nr, deq, 0);
        std::vector<uint16_t> dr = host(deq, (size_t) nr * KR);
        hy4::k::kv_deq(q8f, 0, KL, R, first, n_all, deq, 0);
        std::vector<uint16_t> df = host(deq, (size_t) n_all * KR);
        std::vector<double> rows8((size_t) n_all * KR);
        double worst = 0, same = 0, rope = 0;
        for (int i = 0; i < n_all; ++i)
            for (int b = 0; b < KR / 32; ++b) {
                double am = 0;
                for (int d = 32 * b; d < 32 * b + 32; ++d) am = std::max(am, std::fabs(rows[(size_t) i * KR + d]));
                for (int d = 32 * b; d < 32 * b + 32; ++d) {
                    __half hv;
                    std::memcpy(&hv, &df[(size_t) i * KR + d], 2);
                    rows8[(size_t) i * KR + d] = h2f(hv);
                    if (d >= KL) rope = std::max(rope, std::fabs(h2f(hv) - rows[(size_t) i * KR + d]));
                    // (the F16 reference row's own rounding: a 2^-11 of the value more)
                    worst = std::max(worst, std::fabs(h2f(hv) - rows[(size_t) i * KR + d]) /
                                                (am / 127.0 * 0.5 + std::fabs(rows[(size_t) i * KR + d]) / 1024.0 + 1e-6));
                    if (i >= n_all - nr) {
                        __half hr;
                        std::memcpy(&hr, &dr[(size_t) (i - (n_all - nr)) * KR + d], 2);
                        same = std::max(same, std::fabs((double) h2f(hr) - h2f(hv)));
                    }
                }
            }
        std::printf("  kv_store q8: worst %.2f of half a step; the rotated part F16's (max diff %.1e); ring rows = full "
                    "rows (max diff %.1e)\n", worst, rope, same);
        CHECK(worst <= 1.05 && rope < 2e-3 && same == 0.0);
        auto ref8 = [&](const std::vector<std::vector<int>>& keys) {
            std::vector<double> o((size_t) T * H * KL, 0.0);
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < H; ++h) {
                    const float* qh = q.data() + ((size_t) t * H + h) * KR;
                    std::vector<double> sc;
                    for (int p : keys[(size_t) t]) {
                        const double* row = rows8.data() + (size_t) (p - first) * KR;
                        double v = 0;
                        for (int d = 0; d < KR; ++d) v += qh[d] * row[d];
                        sc.push_back(v * scale);
                    }
                    double m = sinks[(size_t) h];
                    for (double v : sc) m = std::max(m, v);
                    double den = std::exp(sinks[(size_t) h] - m);
                    for (double v : sc) den += std::exp(v - m);
                    for (size_t k = 0; k < sc.size(); ++k) {
                        const double pk = std::exp(sc[k] - m) / den;
                        const double* row = rows8.data() + (size_t) (keys[(size_t) t][k] - first) * KR;
                        for (int d = 0; d < KL; ++d) o[((size_t) t * H + h) * KL + d] += pk * row[d];
                    }
                }
            return o;
        };
        std::vector<int> lo(T), nk(T);
        std::vector<std::vector<int>> keys(T);
        for (int t = 0; t < T; ++t) {
            lo[(size_t) t] = std::max(0, pos[(size_t) t] - n_swa + 1);
            nk[(size_t) t] = pos[(size_t) t] - lo[(size_t) t] + 1;
            for (int p = lo[(size_t) t]; p <= pos[(size_t) t]; ++p) keys[(size_t) t].push_back(p);
        }
        int *dlo = dev(lo), *dnk = dev(nk);
        hy4::k::attention(dq, T, H, KL, R, q8r, ring, nullptr, 0, dlo, dnk, n_swa, scale, dsink, ws, out, 0, true);
        const auto got = host(out, (size_t) T * H * KL);
        const double e8 = rel(got, ref8(keys)), e16 = rel(got, ref(keys));
        std::printf("  attention q8, window ring: rel %.2e to its rows, %.2e to the F16 cache's\n", e8, e16);
        CHECK(e8 < 1e-5 && e16 < 2e-2);
        const int ldk = n_all;   // (a key list: every second position of the full cache)
        std::vector<int> kl((size_t) T * ldk, 0), nk2(T);
        std::vector<std::vector<int>> keys2(T);
        for (int t = 0; t < T; ++t) {
            for (int p = first + (t & 1); p <= pos[(size_t) t]; p += 2) keys2[(size_t) t].push_back(p);
            nk2[(size_t) t] = (int) keys2[(size_t) t].size();
            for (size_t k = 0; k < keys2[(size_t) t].size(); ++k) kl[(size_t) t * ldk + k] = keys2[(size_t) t][k];
        }
        int *dk = dev(kl), *dnk2 = dev(nk2);
        float* ws2;
        cudaMalloc(&ws2, sizeof(float) * hy4::k::attn_ws_floats(T, H, KL, ldk));
        hy4::k::attention(dq, T, H, KL, R, q8f, 0, dk, ldk, nullptr, dnk2, ldk, scale, dsink, ws2, out, 0, true);
        const double ek = rel(host(out, (size_t) T * H * KL), ref8(keys2));
        std::printf("  attention q8, key list: rel %.2e\n", ek);
        CHECK(ek < 1e-5);
        cudaFree(ws2);
        cudaFree(dk);
        cudaFree(dnk2);
        cudaFree(dlo);
        cudaFree(dnk);
        cudaFree(q8r);
        cudaFree(q8f);
        cudaFree(deq);
    }
    cudaFree(cache);
    cudaFree(dkv);
    cudaFree(dw);
    cudaFree(dpos_all);
    cudaFree(dq);
    cudaFree(dsink);
    cudaFree(out);
    cudaFree(ws);
}

void test_router() {
    const int T = 4, NE = 512, NU = 16;
    auto lg = randn((size_t) T * NE, 2.0f), bias = randn(NE, 0.1f);
    for (int i = 0; i < 40; ++i) lg[(size_t) i] = lg[(size_t) (i + 100)];   // (ties among the logits)
    float *dl = dev(lg), *db = dev(bias), *dw;
    int* di;
    cudaMalloc(&di, sizeof(int) * T * NU);
    cudaMalloc(&dw, sizeof(float) * T * NU);
    hy4::k::router(dl, T, NE, db, NU, true, 4.0f, di, dw, 0);
    const std::vector<int> ids = host(di, (size_t) T * NU);
    const std::vector<float> w = host(dw, (size_t) T * NU);
    int bad = 0;
    double werr = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<int> o((size_t) NE);
        std::iota(o.begin(), o.end(), 0);
        auto sel = [&](int e) { return sigm(lg[(size_t) t * NE + e]) + bias[(size_t) e]; };
        std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return sel(a) > sel(b); });
        double sum = 0;
        for (int j = 0; j < NU; ++j) sum += sigm(lg[(size_t) t * NE + o[(size_t) j]]);
        std::vector<int> got(ids.begin() + (long) t * NU, ids.begin() + (long) (t + 1) * NU);
        std::vector<int> want(o.begin(), o.begin() + NU);
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        bad += got != want;
        for (int j = 0; j < NU; ++j) {
            const int e = ids[(size_t) t * NU + j];
            werr = std::max(werr, std::fabs(w[(size_t) t * NU + j] - sigm(lg[(size_t) t * NE + e]) / sum * 4.0));
        }
    }
    std::printf("  router: %d tokens with other picks, max weight diff %.2e\n", bad, werr);
    {   // (its time: a token, and a window of 4)
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int tt : {1, 4}) {
            for (int r = 0; r < 20; ++r) hy4::k::router(dl, tt, NE, db, NU, true, 4.0f, di, dw, 0);
            cudaEventRecord(e0);
            for (int r = 0; r < 200; ++r) hy4::k::router(dl, tt, NE, db, NU, true, 4.0f, di, dw, 0);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            std::printf("  router: %d tokens, %.1f us a call\n", tt, ms * 1e3 / 200);
        }
        {   // (a launch's own cost, for scale: a 16-value add)
            for (int r = 0; r < 20; ++r) hy4::k::add(dw, dw, 16, 0);
            cudaEventRecord(e0);
            for (int r = 0; r < 200; ++r) hy4::k::add(dw, dw, 16, 0);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            std::printf("  (a 16-value add: %.1f us a call)\n", ms * 1e3 / 200);
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    CHECK(bad == 0 && werr < 1e-5);
    cudaFree(dl);
    cudaFree(db);
    cudaFree(di);
    cudaFree(dw);
}

// the indexer: pooled keys of new groups, the scores, the picks and the tail
void test_indexer() {
    const int D = 128, KP = 4, R = 64, IH = 16, ring = 64;
    const float eps = 1e-5f, base = 1e7f;
    const int n_pos = 4 * 300;   // (positions 0 .. 1199: 300 groups)
    // every position's value | gate in a ring the size of the positions (a ring of `ring` rows suffices for a window;
    // the test pools all groups at once)
    auto ivg = randn((size_t) n_pos * 2 * D), ape = randn((size_t) KP * D, 0.3f), w = randn(D, 0.2f);
    for (float& v : w) v += 1.0f;
    float *divg = dev(ivg), *dape = dev(ape), *dw = dev(w), *pooled;
    cudaMalloc(&pooled, sizeof(float) * (n_pos / KP) * D);
    hy4::k::ix_pool(divg, n_pos, 0, n_pos / KP, KP, D, dape, dw, eps, R, base, pooled, 0);
    (void) ring;
    std::vector<double> rp((size_t) (n_pos / KP) * D);
    for (int gi = 0; gi < n_pos / KP; ++gi) {
        std::vector<double> v((size_t) D);
        for (int c = 0; c < D; ++c) {
            double m = -1e300, l[8];
            for (int k = 0; k < KP; ++k) {
                l[k] = ivg[(size_t) (gi * KP + k) * 2 * D + D + c] + ape[(size_t) k * D + c];
                m = std::max(m, l[k]);
            }
            double den = 0, num = 0;
            for (int k = 0; k < KP; ++k) {
                const double e = std::exp(l[k] - m);
                den += e;
                num += e * ivg[(size_t) (gi * KP + k) * 2 * D + c];
            }
            v[(size_t) c] = num / den;
        }
        double ss = 0;
        for (double x : v) ss += x * x;
        const double r = 1.0 / std::sqrt(ss / D + eps);
        for (int c = 0; c < D; ++c) v[(size_t) c] *= r * w[(size_t) c];
        for (int k = 0; k < R / 2; ++k) rope_ref(v[(size_t) (D - R + 2 * k)], v[(size_t) (D - R + 2 * k + 1)], gi * KP, k, R, base);
        for (int c = 0; c < D; ++c) rp[(size_t) gi * D + c] = v[(size_t) c];
    }
    const std::vector<float> pk = host(pooled, rp.size());
    const double ep = rel(pk, rp);
    std::printf("  ix_pool: rel %.2e\n", ep);
    CHECK(ep < 1e-5);
    // scores and picks for tokens past the all-pools range (nv > top)
    const int T = 3, top = 40;
    std::vector<int> pos = {n_pos - 1, n_pos - 3, 4 * 30 + 1};   // (the last one sees 30 pools: fewer than top)
    auto q = randn((size_t) T * IH * D, 0.2f), wt = randn((size_t) T * IH, 0.3f);
    float *dq = dev(q), *dwt = dev(wt), *score;
    int* dpos = dev(pos);
    const int n_pool = n_pos / KP, ld = n_pool + 8;
    cudaMalloc(&score, sizeof(float) * T * ld);
    hy4::k::ix_score(dq, dwt, T, IH, D, pooled, dpos, KP, n_pool, score, ld, 0);
    std::vector<float> sc = host(score, (size_t) T * ld);
    // the reference scores (from the device's pooled keys: the picks are compared, not the keys again)
    double emax = 0;
    for (int t = 0; t < T; ++t)
        for (int gi = 0; gi < pos[(size_t) t] / KP; ++gi) {
            double s = 0;
            for (int h = 0; h < IH; ++h) {
                double d = 0;
                for (int c = 0; c < D; ++c) d += (double) q[((size_t) t * IH + h) * D + c] * pk[(size_t) gi * D + c];
                s += std::max(d, 0.0) * wt[(size_t) t * IH + h];
            }
            emax = std::max(emax, std::fabs(s - sc[(size_t) t * ld + gi]) / std::max(1.0, std::fabs(s)));
        }
    std::printf("  ix_score: max rel diff %.2e\n", emax);
    CHECK(emax < 1e-4);
    {   // ix_reduce: the heads' products as a GEMM leaves them (column (t, h), row g - g0), in two slabs -> ix_score's
        // scores, -inf from each token's nv on
        const int cut = n_pool / 3 / 8 * 8;
        float* sr;
        cudaMalloc(&sr, sizeof(float) * T * ld);
        double er = 0;
        int bad_inf = 0;
        for (int part = 0; part < 2; ++part) {
            const int g0 = part ? cut : 0, n = part ? n_pool - cut : cut, ldS = (n + 7) / 8 * 8;
            std::vector<float> S((size_t) T * IH * ldS, 0.0f);
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < IH; ++h)
                    for (int j = 0; j < n; ++j) {
                        double d = 0;
                        for (int c = 0; c < D; ++c)
                            d += (double) q[((size_t) t * IH + h) * D + c] * pk[(size_t) (g0 + j) * D + c];
                        S[(size_t) (t * IH + h) * ldS + j] = (float) d;
                    }
            float* dS = dev(S);
            hy4::k::ix_reduce(dS, (size_t) ldS, dwt, T, IH, dpos, KP, g0, n, sr, ld, 0);
            cudaDeviceSynchronize();
            cudaFree(dS);
        }
        const std::vector<float> s3 = host(sr, (size_t) T * ld);
        for (int t = 0; t < T; ++t)
            for (int gi = 0; gi < n_pool; ++gi) {
                const float a = s3[(size_t) t * ld + gi], b = sc[(size_t) t * ld + gi];
                if (gi >= pos[(size_t) t] / KP) bad_inf += !(std::isinf(a) && a < 0 && std::isinf(b) && b < 0);
                else er = std::max(er, (double) std::fabs(a - b) / std::max(1.0, (double) std::fabs(b)));
            }
        std::printf("  ix_reduce: max rel diff to ix_score %.2e, %d beyond a token's pools not -inf\n", er, bad_inf);
        CHECK(er < 1e-4 && bad_inf == 0);
        cudaFree(sr);
    }
    {   // --kv q8's F16 rows (value | gate ring, pooled keys): the pooled values within F16's rounding, the scores
        // within its effect on them
        float *p16, *s16, *iv16;
        cudaMalloc(&p16, sizeof(uint16_t) * (n_pos / KP) * D);
        cudaMalloc(&s16, sizeof(float) * T * ld);
        cudaMalloc(&iv16, sizeof(uint16_t) * (size_t) n_pos * 2 * D);
        std::vector<int> all_p((size_t) n_pos);
        for (int i = 0; i < n_pos; ++i) all_p[(size_t) i] = i;
        int* dall = dev(all_p);
        hy4::k::store_rows(divg, n_pos, 2 * D, iv16, n_pos, dall, 0, true);   // (the F32 rows as the ring's F16)
        hy4::k::ix_pool(iv16, n_pos, 0, n_pos / KP, KP, D, dape, dw, eps, R, base, p16, 0, true);
        hy4::k::ix_score(dq, dwt, T, IH, D, p16, dpos, KP, n_pool, s16, ld, 0, true);
        const std::vector<uint16_t> h16 = host((const uint16_t*) p16, (size_t) (n_pos / KP) * D);
        std::vector<float> pf(h16.size());
        for (size_t i = 0; i < h16.size(); ++i) {
            __half hv;
            std::memcpy(&hv, &h16[i], 2);
            pf[i] = h2f(hv);
        }
        const std::vector<float> s2 = host(s16, (size_t) T * ld);
        double es = 0;
        for (int t = 0; t < T; ++t)
            for (int gi = 0; gi < pos[(size_t) t] / KP; ++gi)
                es = std::max(es, (double) std::fabs(s2[(size_t) t * ld + gi] - sc[(size_t) t * ld + gi]) /
                                      std::max(1.0, (double) std::fabs(sc[(size_t) t * ld + gi])));
        const double e16 = rel(pf, rp);
        std::printf("  ix_pool f16: rel %.2e; ix_score over them: max rel diff %.2e\n", e16, es);
        CHECK(e16 < 2e-3 && es < 1e-2);
        cudaFree(p16);
        cudaFree(s16);
        cudaFree(iv16);
        cudaFree(dall);
    }
    // ties: the second token's first 20 scores copied to pools 100 .. 119 (equal scores, some near the threshold)
    for (int gi = 0; gi < 20; ++gi) sc[(size_t) 1 * ld + gi + 100] = sc[(size_t) 1 * ld + gi];
    cudaMemcpy(score, sc.data(), sc.size() * sizeof(float), cudaMemcpyHostToDevice);
    const int ldk = top * KP + KP;
    int *keys, *nk;
    cudaMalloc(&keys, sizeof(int) * T * ldk);
    cudaMalloc(&nk, sizeof(int) * T);
    hy4::k::ix_select(score, ld, T, dpos, KP, top, keys, ldk, nk, 0);
    const std::vector<int> hk = host(keys, (size_t) T * ldk), hn = host(nk, (size_t) T);
    // each token: nsel = min(nv, top) whole pools (all kpool positions each), no pool scored below one left out, then
    // its own group's positions up to itself (a tie at the threshold may go either way)
    int bad = 0;
    for (int t = 0; t < T; ++t) {
        const int p = pos[(size_t) t], nv = p / KP, nsel = std::min(nv, top);
        std::vector<int> got(hk.begin() + (long) t * ldk, hk.begin() + (long) t * ldk + hn[(size_t) t]);
        std::sort(got.begin(), got.end());
        std::vector<int> cnt((size_t) nv, 0), tail;
        bool ok = got.size() == (size_t) (nsel * KP + p % KP + 1);
        for (int k : got) {
            if (k >= nv * KP) tail.push_back(k);
            else if (k >= 0) ++cnt[(size_t) (k / KP)];
            else ok = false;
        }
        std::vector<int> want_tail;
        for (int k = 0; k <= p % KP; ++k) want_tail.push_back(nv * KP + k);
        ok = ok && tail == want_tail;
        float lo_in = INFINITY, hi_out = -INFINITY;
        int n_in = 0;
        for (int gi = 0; gi < nv; ++gi) {
            const float v = sc[(size_t) t * ld + gi];
            if (cnt[(size_t) gi] == KP) {
                ++n_in;
                lo_in = std::min(lo_in, v);
            } else if (cnt[(size_t) gi] == 0) {
                hi_out = std::max(hi_out, v);
            } else {
                ok = false;   // (a pool in part)
            }
        }
        ok = ok && n_in == nsel && (nsel == nv || lo_in >= hi_out);
        if (!ok) {
            ++bad;
            std::fprintf(stderr, "  token %d: %zu keys (%d pools, lowest %g, highest left out %g), %d pools expected\n",
                         t, got.size(), n_in, lo_in, hi_out, nsel);
        }
    }
    std::printf("  ix_select: %d tokens with other keys\n", bad);
    CHECK(bad == 0);
    {   // the prompt path past the dense span: the pools the tokens' keys name (ix_union), their columns' mask
        // (dsa_mask_list), their rows gathered from a cache (kv_gather's list form), the softmax by the mask alone
        const int npl = n_pos / KP + 1, kend = n_pos - 2, row = 16;
        int *col, *list, *dn;
        cudaMalloc(&col, sizeof(int) * npl);
        cudaMalloc(&list, sizeof(int) * npl);
        cudaMalloc(&dn, sizeof(int));
        hy4::k::ix_union(keys, ldk, nk, T, KP, npl, col, list, dn, 0);
        const std::vector<int> hc = host(col, (size_t) npl), hl = host(list, (size_t) npl), hn1 = host(dn, 1);
        std::vector<int> want((size_t) npl, 0);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < hn[(size_t) t]; ++i) want[(size_t) (hk[(size_t) t * ldk + i] / KP)] = 1;
        int ub = 0, r = 0;
        for (int gi = 0; gi < npl; ++gi) {
            if (want[(size_t) gi]) {
                ub += hc[(size_t) gi] != r || hl[(size_t) r] != gi;
                ++r;
            } else {
                ub += hc[(size_t) gi] != -1;
            }
        }
        ub += hn1[0] != r;
        const int ncol = r * KP;
        float* mask;
        cudaMalloc(&mask, sizeof(float) * T * ncol);
        hy4::k::dsa_mask_list(keys, ldk, nk, T, col, KP, ncol, mask, (size_t) ncol, 0);
        const std::vector<float> hm = host(mask, (size_t) T * ncol);
        int mb = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<char> in((size_t) ncol, 0);
            for (int i = 0; i < hn[(size_t) t]; ++i) {
                const int p = hk[(size_t) t * ldk + i];
                in[(size_t) (hc[(size_t) (p / KP)] * KP + p % KP)] = 1;
            }
            for (int j = 0; j < ncol; ++j) {
                const float v = hm[(size_t) t * ncol + j];
                mb += in[(size_t) j] ? v != 0.0f : !(std::isinf(v) && v < 0);
            }
        }
        // a cache whose row q holds q + 1 in every value: the list's rows in order, zeros from kend on
        std::vector<uint16_t> cache((size_t) (npl * KP) * row);
        for (size_t i = 0; i < cache.size(); ++i) cache[i] = (uint16_t) (i / row + 1);
        uint16_t *dc = dev(cache), *gat;
        cudaMalloc(&gat, sizeof(uint16_t) * ncol * row);
        hy4::k::kv_gather(dc, 0, row, 0, ncol, gat, 0, list, KP, kend);
        const std::vector<uint16_t> hg = host((const uint16_t*) gat, (size_t) ncol * row);
        int gb = 0;
        for (int j = 0; j < ncol; ++j) {
            const int q = hl[(size_t) (j / KP)] * KP + j % KP;
            for (int c = 0; c < row; ++c) gb += hg[(size_t) j * row + c] != (q < kend ? (uint16_t) (q + 1) : 0);
        }
        // the softmax by the mask alone = the one over positions when every column is a position the tokens see
        const int Hh = 2, ldS = (ncol + 7) / 8 * 8;
        auto S = randn((size_t) T * Hh * ldS, 1.0f);
        std::vector<int> far((size_t) T, ncol);
        float* dS = dev(S);
        int* dfar = dev(far);
        uint16_t *P1, *P2;
        cudaMalloc(&P1, sizeof(uint16_t) * T * Hh * ldS);
        cudaMalloc(&P2, sizeof(uint16_t) * T * Hh * ldS);
        cudaMemset(P1, 0, sizeof(uint16_t) * T * Hh * ldS);
        cudaMemset(P2, 0, sizeof(uint16_t) * T * Hh * ldS);
        hy4::k::attn_softmax(dS, ldS, T, Hh, ncol, 0, dfar, 0, mask, (size_t) ncol, 0.3f, nullptr, P1, ldS, 0);
        hy4::k::attn_softmax(dS, ldS, T, Hh, ncol, 0, dpos, -1, mask, (size_t) ncol, 0.3f, nullptr, P2, ldS, 0);
        const bool same_p = host((const uint16_t*) P1, (size_t) T * Hh * ldS) ==
                            host((const uint16_t*) P2, (size_t) T * Hh * ldS);
        std::printf("  ix_union: %d pools, %d wrong; dsa_mask_list: %d wrong; kv_gather by the list: %d wrong; the "
                    "softmax by the mask alone: %s\n", r, ub, mb, gb, same_p ? "the same" : "DIFFERENT");
        CHECK(ub == 0 && mb == 0 && gb == 0 && same_p);
        {   // tensor parallel's exchange of the picks: packed into mapped host rows, unpacked over cleared keys
            const int ldw = (top + 7) / 8 * 8;
            uint16_t* wire = nullptr;
            cudaHostAlloc(&wire, sizeof(uint16_t) * T * ldw, cudaHostAllocMapped | cudaHostAllocPortable);
            int *k2, *n2;
            cudaMalloc(&k2, sizeof(int) * T * ldk);
            cudaMalloc(&n2, sizeof(int) * T);
            cudaMemset(k2, 0xff, sizeof(int) * T * ldk);
            cudaMemset(n2, 0, sizeof(int) * T);
            hy4::k::ix_pack(keys, ldk, T, dpos, KP, top, wire, ldw, 0);
            hy4::k::ix_unpack(wire, ldw, T, dpos, KP, top, k2, ldk, n2, 0);
            const std::vector<int> hk2 = host(k2, (size_t) T * ldk), hn2 = host(n2, (size_t) T);
            int xb = 0;
            for (int t = 0; t < T; ++t) {
                xb += hn2[(size_t) t] != hn[(size_t) t];
                for (int i = 0; i < hn[(size_t) t]; ++i) xb += hk2[(size_t) t * ldk + i] != hk[(size_t) t * ldk + i];
            }
            std::printf("  ix_pack / ix_unpack: %d keys or counts other than ix_select's\n", xb);
            CHECK(xb == 0);
            cudaFree(k2);
            cudaFree(n2);
            cudaFreeHost(wire);
        }
        for (int* p : {col, list, dn, dfar}) cudaFree(p);
        for (uint16_t* p : {dc, gat, P1, P2}) cudaFree(p);
        cudaFree(mask);
        cudaFree(dS);
    }
    for (float* p : {divg, dape, dw, pooled, dq, dwt, score}) cudaFree(p);
    for (int* p : {dpos, keys, nk}) cudaFree(p);
}

void test_engram() {
    const int T = 3, cols = 16, dim = 96, hc = 4, E = 3072;
    const float eps = 1.1920929e-07f;
    std::vector<uint8_t> rows((size_t) T * cols * dim);
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint8_t& b : rows) {
        b = (uint8_t) byte(rng);
        if ((b & 0x7f) == 0x7f) b ^= 1;   // (no NaN codes)
    }
    std::vector<float> valid = {1.0f, 0.0f, 1.0f};
    auto ln = randn((size_t) cols * dim, 0.2f);
    for (float& v : ln) v += 1.0f;
    uint8_t* drows = dev(rows);
    float *dvalid = dev(valid), *dln = dev(ln), *r;
    cudaMalloc(&r, sizeof(float) * T * cols * dim);
    hy4::k::engram_rows(drows, dvalid, T, cols, dim, dln, eps, r, 0);
    auto f8 = [](uint8_t x) {
        const int ax = x & 0x7f, e = (ax >> 3) & 0xf, m = ax & 7;
        const double v = e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m / 8.0, e - 7);
        return (x & 0x80) ? -v : v;
    };
    std::vector<double> rr((size_t) T * cols * dim);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < cols; ++c) {
            double ss = 0;
            for (int i = 0; i < dim; ++i) ss += f8(rows[((size_t) t * cols + c) * dim + i]) * f8(rows[((size_t) t * cols + c) * dim + i]);
            const double s = 1.0 / std::sqrt(ss / dim + eps);
            for (int i = 0; i < dim; ++i)
                rr[((size_t) t * cols + c) * dim + i] = f8(rows[((size_t) t * cols + c) * dim + i]) * s * ln[(size_t) c * dim + i] * valid[(size_t) t];
        }
    const double e1 = rel(host(r, rr.size()), rr);
    std::printf("  engram_rows: rel %.2e\n", e1);
    CHECK(e1 < 1e-6);
    {   // the rows' own scales (a table of per-row scaled codes), at the BF16 run's eps: values = codes * scale
        const float eps2 = 0.0078125f;
        std::vector<float> sc((size_t) T * cols);
        std::uniform_real_distribution<float> us(1e-5f, 3e-4f);
        for (float& v : sc) v = us(rng);
        float* dsc = dev(sc);
        hy4::k::engram_rows(drows, dvalid, T, cols, dim, dln, eps2, r, 0, dsc);
        std::vector<double> rs((size_t) T * cols * dim);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < cols; ++c) {
                const size_t b = ((size_t) t * cols + c) * dim;
                const double sv = sc[(size_t) t * cols + c];
                double ss = 0;
                for (int i = 0; i < dim; ++i) ss += f8(rows[b + i]) * sv * f8(rows[b + i]) * sv;
                const double s = 1.0 / std::sqrt(ss / dim + eps2);
                for (int i = 0; i < dim; ++i) rs[b + i] = f8(rows[b + i]) * sv * s * ln[(size_t) c * dim + i] * valid[(size_t) t];
            }
        const double e1s = rel(host(r, rs.size()), rs);
        std::printf("  engram_rows (row scales, eps %g): rel %.2e\n", (double) eps2, e1s);
        CHECK(e1s < 1e-6);
        cudaFree(dsc);
    }
    // the mix
    auto x = randn((size_t) T * hc * E), kv = randn((size_t) T * (hc + 1) * E), kvb = randn((size_t) (hc + 1) * E, 0.1f),
         eg = randn((size_t) hc * E, 0.5f), gam = randn(E, 0.3f), hcs = randn((size_t) hc * E, 0.1f);
    float *dx = dev(x), *dkv = dev(kv), *dkvb = dev(kvb), *deg = dev(eg), *dgam = dev(gam), *dhcs = dev(hcs);
    hy4::k::engram_mix(dx, dkv, dkvb, T, hc, E, deg, dgam, dhcs, dvalid, eps, 0);
    std::vector<double> rx(x.begin(), x.end());
    for (int t = 0; t < T; ++t) {
        if (valid[(size_t) t] == 0.0f) continue;
        auto rms = [&](const double* v) {
            double ss = 0;
            for (int d = 0; d < E; ++d) ss += v[d] * v[d];
            return 1.0 / std::sqrt(ss / E + eps);
        };
        std::vector<double> key((size_t) (hc + 1) * E);
        for (int j = 0; j < (hc + 1) * E; ++j) key[(size_t) j] = (double) kv[(size_t) t * (hc + 1) * E + j] + kvb[(size_t) j];
        double gate[8];
        for (int s = 0; s < hc; ++s) {
            std::vector<double> xs(x.begin() + (long) ((size_t) t * hc + s) * E, x.begin() + (long) ((size_t) t * hc + s + 1) * E);
            const double rx2 = rms(xs.data()), rk = rms(key.data() + (size_t) s * E);
            double dot = 0;
            for (int d = 0; d < E; ++d) dot += xs[(size_t) d] * rx2 * key[(size_t) s * E + d] * rk * eg[(size_t) s * E + d];
            dot /= std::sqrt((double) E);
            const double mag = std::sqrt(std::max(std::fabs(dot), 1e-6));
            gate[s] = sigm(dot < 0 ? -mag : mag);
        }
        const double rv = rms(key.data() + (size_t) hc * E);
        for (int d = 0; d < E; ++d) {
            const double sig = key[(size_t) hc * E + d] * rv * gam[(size_t) d];
            for (int s = 0; s < hc; ++s) rx[((size_t) t * hc + s) * E + d] += sig * (gate[s] + hcs[(size_t) s * E + d]);
        }
    }
    const double e2 = rel_delta(host(dx, rx.size()), rx, x);
    std::printf("  engram_mix: rel %.2e\n", e2);
    CHECK(e2 < 1e-6);
    cudaFree(drows);
    for (float* p : {dvalid, dln, r, dx, dkv, dkvb, deg, dgam, dhcs}) cudaFree(p);
}

// row_top: each row's largest value's first index (the host's walk), the value beside it; -1 for a row with a NaN or
// nothing above -inf - over a row's part (an offset i0, rows ld apart)
void test_row_top() {
    const int rows = 6, n = 13424, ld = 13440, i0 = 26848;
    std::vector<float> x = randn((size_t) rows * ld);
    // row 1: its largest value twice (the lower index wins), at a block-stride apart; row 2: equal values everywhere;
    // row 3: a NaN; row 4: all -inf; row 5: the largest in the last column
    x[(size_t) 1 * ld + 700] = 9.0f;
    x[(size_t) 1 * ld + 700 + 256] = 9.0f;
    x[(size_t) 1 * ld + 5000] = 9.0f;
    for (int i = 0; i < n; ++i) x[(size_t) 2 * ld + i] = 0.5f;
    x[(size_t) 3 * ld + 4321] = std::nanf("");
    for (int i = 0; i < n; ++i) x[(size_t) 4 * ld + i] = -INFINITY;
    x[(size_t) 5 * ld + n - 1] = 99.0f;
    float* dx = dev(x);
    float* dout = nullptr;
    cudaMalloc(&dout, sizeof(float) * 2 * rows);
    hy4::k::row_top(dx, rows, n, (size_t) ld, i0, dout, 0);
    const std::vector<float> out = host(dout, (size_t) 2 * rows);
    int bad = 0;
    for (int r = 0; r < rows; ++r) {
        const float* row = x.data() + (size_t) r * ld;
        int b = 0;
        bool nan = false;
        for (int i = 0; i < n; ++i) {
            nan |= std::isnan(row[i]);
            if (row[i] > row[b]) b = i;
        }
        const bool none = nan || !(row[b] > -INFINITY);
        int id;
        std::memcpy(&id, &out[(size_t) 2 * r], 4);
        const int want = none ? -1 : i0 + b;
        if (id != want || (!none && out[(size_t) 2 * r + 1] != row[b])) {
            std::fprintf(stderr, "  row_top row %d: %d, the walk %d\n", r, id, want);
            ++bad;
        }
    }
    std::printf("  row_top: %d of %d rows as the walk (ties, a NaN, -inf)\n", rows - bad, rows);
    CHECK(bad == 0);
    cudaFree(dx);
    cudaFree(dout);
}

}  // namespace

// KV streaming: a VRAM ring with tags and every row in (mapped) host memory read as the cache with every row in VRAM -
// the rows themselves, the spans in order, the picked rows (side by side, and each once), the attention over them:
// the same bits, F16 rows and 8-bit rows
void test_kv_streaming() {
    const int n = 200, H = 32, KL = 512, R = 64, KR = KL + R, ring = 48, T = 5, ldk = 24;
    const float eps = 1e-5f, base = 1e4f, scale = 1.0f / 16.0f;
    auto kv = randn((size_t) n * KR), w = randn(KL, 0.2f), q = randn((size_t) T * H * KR), sink = randn(H);
    std::vector<int> all((size_t) n);
    std::iota(all.begin(), all.end(), 0);
    float *dkv = dev(kv), *dw = dev(w), *dq = dev(q), *dsink = dev(sink);
    int* dall = dev(all);
    // the keys: scattered positions, most of them out of the ring by the end (the newest 48 are in it)
    std::vector<int> kl((size_t) T * ldk, 0), nk(T);
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int t = 0; t < T; ++t) {
        std::vector<int> v;
        for (int k = 0; k < 16 + t; ++k) v.push_back(pick(rng));
        v.push_back(n - 1 - t);
        v.push_back(7);   // (a position every token picks)
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        nk[(size_t) t] = (int) v.size();
        for (size_t k = 0; k < v.size(); ++k) kl[(size_t) t * ldk + k] = v[k];
    }
    int *dk = dev(kl), *dnk = dev(nk);
    std::vector<int> lo(T), minus((size_t) n, -1);
    for (int t = 0; t < T; ++t) lo[(size_t) t] = t * ldk;
    int *dlo = dev(lo), *slot = dev(minus), *count, *list, *keys2;
    cudaMalloc(&count, sizeof(int));
    cudaMalloc(&list, sizeof(int) * T * ldk);
    cudaMalloc(&keys2, sizeof(int) * T * ldk);
    float *ws, *out;
    cudaMalloc(&ws, sizeof(float) * hy4::k::attn_ws_floats(T, H, KL, ldk));
    cudaMalloc(&out, sizeof(float) * T * H * KL);
    for (int q8 = 0; q8 < 2; ++q8) {
        const size_t rb = q8 ? hy4::k::kv_q8_row_bytes(KL, R) : (size_t) KR * 2;
        uint16_t *full, *rc, *hrows, *picked, *span, *span0;
        int* tag;
        cudaMalloc(&full, rb * n);
        cudaMalloc(&rc, rb * ring);
        cudaMalloc(&tag, sizeof(int) * ring);
        cudaMalloc(&picked, rb * T * ldk);
        cudaMalloc(&span, sizeof(uint16_t) * n * KR);
        cudaMalloc(&span0, sizeof(uint16_t) * n * KR);
        cudaMemset(tag, 0xff, sizeof(int) * ring);
        CHECK(cudaHostAlloc((void**) &hrows, rb * n, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
        std::memset(hrows, 0x5a, rb * n);
        const hy4::k::KvHost hs{tag, hrows};
        hy4::k::kv_store(dkv, n, KL, R, dw, eps, dall, base, full, 0, 0, q8 != 0);
        for (int p0 = 0; p0 < n; p0 += 40)   // (windows of 40 positions: the ring of 48 wraps four times)
            hy4::k::kv_store(dkv + (size_t) p0 * KR, std::min(40, n - p0), KL, R, dw, eps, dall + p0, base, rc, ring, 0,
                             q8 != 0, hs);
        cudaDeviceSynchronize();
        const auto hf = host((const uint8_t*) full, rb * n);
        const bool rows_same = std::memcmp(hf.data(), hrows, rb * n) == 0;
        const auto tg = host(tag, (size_t) ring);
        bool tags_ok = true;
        for (int p = n - ring; p < n; ++p) tags_ok = tags_ok && tg[(size_t) (p % ring)] == p;
        // a span in order (the prompt path's dense read): [0, n) from the ring and the host = from the full cache
        if (q8) {
            hy4::k::kv_deq(rc, ring, KL, R, 0, n, span, 0, nullptr, 1, 0, hs);
            hy4::k::kv_deq(full, 0, KL, R, 0, n, span0, 0);
        } else {
            hy4::k::kv_gather(rc, ring, KR, 0, n, span, 0, nullptr, 1, 0, hs);
            hy4::k::kv_gather(full, 0, KR, 0, n, span0, 0);
        }
        const bool span_same = host(span, (size_t) n * KR) == host(span0, (size_t) n * KR);
        // the attention over the key list: the full cache's, the picked rows', the rows' picked once
        hy4::k::attention(dq, T, H, KL, R, full, 0, dk, ldk, nullptr, dnk, ldk, scale, dsink, ws, out, 0, q8 != 0);
        const auto o0 = host(out, (size_t) T * H * KL);
        hy4::k::kv_pick(rc, ring, rb, dk, ldk, dnk, T, picked, 0, hs);
        hy4::k::attention(dq, T, H, KL, R, picked, 0, nullptr, 0, dlo, dnk, ldk, scale, dsink, ws, out, 0, q8 != 0);
        const auto o1 = host(out, (size_t) T * H * KL);
        cudaMemset(picked, 0, rb * T * ldk);
        hy4::k::kv_pick_union(rc, ring, rb, dk, ldk, dnk, T, slot, count, list, keys2, picked, 0, hs);
        const int n_union = host(count, 1)[0];
        std::vector<int> seen;
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < nk[(size_t) t]; ++k) seen.push_back(kl[(size_t) t * ldk + k]);
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        hy4::k::attention(dq, T, H, KL, R, picked, 0, keys2, ldk, nullptr, dnk, ldk, scale, dsink, ws, out, 0, q8 != 0);
        const auto o2 = host(out, (size_t) T * H * KL);
        const bool slot_clean = host(slot, (size_t) n) == minus;
        const bool a1 = std::memcmp(o0.data(), o1.data(), sizeof(float) * o0.size()) == 0;
        const bool a2 = std::memcmp(o0.data(), o2.data(), sizeof(float) * o0.size()) == 0;
        std::printf("  kv streaming, %s rows: host rows %s, tags %s, span %s, picked attention %s, union (%d rows of %zu "
                    "picks) %s, slots %s\n", q8 ? "8-bit" : "F16", rows_same ? "same" : "DIFFER", tags_ok ? "ok" : "BAD",
                    span_same ? "same" : "DIFFERS", a1 ? "same bits" : "DIFFERS", n_union, seen.size(),
                    a2 ? "same bits" : "DIFFERS", slot_clean ? "clean" : "LEFT");
        CHECK(rows_same && tags_ok && span_same && a1 && a2 && slot_clean && n_union == (int) seen.size());
        cudaFree(full);
        cudaFree(rc);
        cudaFree(tag);
        cudaFree(picked);
        cudaFree(span);
        cudaFree(span0);
        cudaFreeHost(hrows);
    }
    for (void* d : {(void*) dkv, (void*) dw, (void*) dq, (void*) dsink, (void*) dall, (void*) dk, (void*) dnk, (void*) dlo,
                    (void*) slot, (void*) count, (void*) list, (void*) keys2, (void*) ws, (void*) out})
        cudaFree(d);
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("hy4_kernels_test: no GPU, skipped\n");
        return 0;
    }
    test_hc();
    test_attention();
    test_kv_streaming();
    test_router();
    test_indexer();
    test_engram();
    test_row_top();
    CHECK(hy4::k::errors() == 0);
    std::printf("hy4_kernels_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
