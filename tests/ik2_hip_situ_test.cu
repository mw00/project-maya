// tests/ik2_hip_situ_test.cu - Inkling 2's expert activation (SITU: ds4_iq.cuh's kActSitu, llama.cpp's LLM_FFN_SITU)
// on maya-ds4's ggml block-type kernels and the CPU lane, no model:
//   a = beta tanh(g / beta) sigmoid(g) * (lb > 0 ? lb tanh(u / lb) : u)      (Inkling 2: beta 4, lb 25)
// Random experts at Inkling 2 Small's shapes (n_embd 4608, n_ff 2304; the GGUF's fused gate_up tensor of an expert -
// rows 0..2303 the gate, 2304..4607 the up - in ds4_iq's blob layout: tr[1] = tr[0] + 2304 row_bytes, then the down),
// quantized by ggml to its layers' types (layers 0-39: gate_up IQ2_XS, down IQ3_XXS; 40-47: IQ3_XXS, IQ4_XS), 24
// experts, 8 picks a token (a skewed routing: some experts of many rows, some of few).  Each form maya-ik2 takes -
// the token form (a plan: a decode step), the window form (grouped entries: a verify window, a few prompt rows), the
// prompt form (the sets: the light experts of <= 16 rows and the GEMM tiles) for 1 / 8 / 16 / 64 rows (the sets also
// 160) and the CPU lane (iq_cpu_gate_up / iq_cpu_down with IqCpu::act) - against a double-precision reference over
// ggml's dequantized weights:
//   h    the activation itself against the same inputs the form rounds (int8 per 32 values on the token / window /
//        light forms, the CPU lane's Q8_K) - only F32 sums apart: the epilogue's SITU checked to ~1e-6
//   y    each token's routed output against the exact reference (unrounded inputs): the forms' own roundings
// and the window form's entries bit for bit the token form's (as with SwiGLU), every form again with other parameters
// (beta 2.5, lb 0: not the structs' defaults), iq::situ_f16 against the reference, and what the dispatch refuses.
// The environment picks the paths a run takes: MAYA_DS4_NO_TC=1 the F32 tiles, MAYA_DS4_NO_LIGHT=1 no light experts,
// DS4_IQ_TC_RM=4 the tensor-core tiles of 256 rows, DS4_IQ_OWN=0 the CPU lane's ggml rows (cmake/ik2_hip_kernels.cmake
// runs each).
//
// --dump FILE / --check FILE: the existing activations (DeepSeek's clamped SwiGLU, MiniMax's swigluoai with gate and up
// of one type and of two) through every form and type, written / compared bit for bit - built as ik2_hip_situ_dump
// (IK2_HIP_SITU_LEGACY_ONLY: this file against ds4_iq's API before kActSitu), run before and after a change to ds4_iq.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ggml.h"
#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "../src/ds4/ds4_iq.cuh"

namespace iq = strata::ds4::gguf::iq;
namespace gg = strata::ds4::gguf;

namespace {

int fails = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                          \
        }                                                                     \
    } while (0)
#define CU(x) CHECK((x) == cudaSuccess)

constexpr int kNU = 8;   // picks a token (the window form's entry e reads input row e >> 3)
constexpr float kBeta = 4.0f, kLb = 25.0f;

int n_threads() {
    const unsigned hw = std::thread::hardware_concurrency();
    return (int) std::max(1u, std::min(24u, hw / 2));
}
template <class Fn>
void par(int n, const Fn& fn) {   // fn(i) for i < n over the threads
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    const int nt = std::max(1, std::min(n, n_threads()));
    for (int w = 0; w < nt; ++w)
        th.emplace_back([&] {
            for (int i; (i = next.fetch_add(1)) < n;) fn(i);
        });
    for (auto& t : th) t.join();
}

float h2f(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, 2);
    return __half2float(h);
}

// x rounded to int8 per 32 values as the device rounds its activations (ds4_gguf.cu's q8_quant, light_q8_kernel): sc =
// max|x| / 127, q = round-half-even(x / sc), the same F32 operations
void q8_host(const float* x, int k, int8_t* q, float* d) {
    for (int b = 0; b < k / 32; ++b) {
        float am = 0.0f;
        for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(x[b * 32 + i]));
        const float sc = am / 127.0f;
        for (int i = 0; i < 32; ++i) q[b * 32 + i] = (int8_t) (am > 0.0f ? (int) std::nearbyint(x[b * 32 + i] / sc) : 0);
        d[b] = sc;
    }
}

double situ_ref(double g, double u, double beta = kBeta, double lb = kLb) {
    const double a = beta * std::tanh(g / beta) / (1.0 + std::exp(-g));
    return a * (lb > 0.0 ? lb * std::tanh(u / lb) : u);
}
double swiglu_ref(double g, double u) { return g / (1.0 + std::exp(-g)) * u; }

template <class T>
struct DBuf {   // a device buffer, zeroed
    T* p = nullptr;
    size_t n = 0;
    explicit DBuf(size_t n_) : n(n_) {
        CU(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
        CU(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
    }
    ~DBuf() { (void) cudaFree(p); }
    DBuf(const DBuf&) = delete;
    DBuf& operator=(const DBuf&) = delete;
    void put(const T* h) { CU(cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice)); }
    void get(T* h) const { CU(cudaMemcpy(h, p, n * sizeof(T), cudaMemcpyDeviceToHost)); }
};

// ---------------------------------------------------------------- the experts
// a matrix pool: n rows of k values drawn N(0, sigma), quantized by ggml to `type` (q: rows rb apart) and decoded back by
// ggml's dequantize_row_* (w: the values the kernels take)
struct Pool {
    int type = 0, k = 0, n = 0;
    size_t rb = 0;
    std::vector<uint8_t> q;
    std::vector<float> w;
};
void make_pool(Pool& p, int type, int k, int n, float sigma, uint64_t seed) {
    p.type = type;
    p.k = k;
    p.n = n;
    p.rb = ggml_row_size((ggml_type) type, k);
    CHECK(p.rb == iq::row_bytes(type, k));
    p.q.assign(p.rb * n, 0);
    p.w.assign((size_t) n * k, 0.0f);
    ggml_quantize_init((ggml_type) type);
    const std::vector<float> im((size_t) k, 1.0f);   // (a uniform importance: IQ2_XXS / IQ2_XS / IQ1_S need one)
    constexpr int kChunk = 16;
    par((n + kChunk - 1) / kChunk, [&](int c) {
        const int r0 = c * kChunk, nr = std::min(kChunk, n - r0);
        std::mt19937_64 rng(seed * 1000003ull + (uint64_t) c);
        std::normal_distribution<float> nd(0.0f, sigma);
        std::vector<float> x((size_t) nr * k);
        for (float& v : x) v = nd(rng);
        ggml_quantize_chunk((ggml_type) type, x.data(), p.q.data() + (size_t) r0 * p.rb, 0, nr, k, im.data());
        const auto* tt = ggml_get_type_traits((ggml_type) type);
        for (int r = 0; r < nr; ++r)
            tt->to_float(p.q.data() + (size_t) (r0 + r) * p.rb, p.w.data() + (size_t) (r0 + r) * k, k);
    });
}

// A layer's experts in ds4_iq's blob layout as maya-ik2 places Inkling's: an expert's GGUF gate_up tensor (rows
// 0..F-1 the gate, F..2F-1 the up: tr[1] = tr[0] + F row_bytes) then its down; expert e's rows are the pools' rows in
// an order of its own (a rotation by e), so every expert differs
struct Layer {
    int tg = 0, tu = 0, td = 0, n_exp = 0, E = 0, F = 0;
    size_t tr[3] = {0, 0, 0}, stride = 0;
    Pool gate, up, down;
    std::vector<uint8_t> host;
    uint8_t* dev = nullptr;
    int g_row(int e, int r) const { return (r + 131 * e) % F; }
    int u_row(int e, int r) const { return (r + 577 * e + 7) % F; }
    int d_row(int e, int r) const { return (r + 1009 * e + 3) % E; }
    Layer() = default;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    ~Layer() {
        if (dev != nullptr) (void) cudaFree(dev);
    }
};
void make_layer(Layer& L, int tg, int tu, int td, int E, int F, int n_exp, float sg, float su, float sd, uint64_t seed) {
    L.tg = tg;
    L.tu = tu;
    L.td = td;
    L.E = E;
    L.F = F;
    L.n_exp = n_exp;
    make_pool(L.gate, tg, E, F, sg, seed);
    make_pool(L.up, tu, E, F, su, seed + 1);
    make_pool(L.down, td, F, E, sd, seed + 2);
    L.tr[0] = 0;
    L.tr[1] = (size_t) F * L.gate.rb;
    L.tr[2] = L.tr[1] + (size_t) F * L.up.rb;
    L.stride = (L.tr[2] + (size_t) E * L.down.rb + 255) / 256 * 256;
    L.host.assign(L.stride * n_exp, 0);
    for (int e = 0; e < n_exp; ++e) {
        uint8_t* b = L.host.data() + (size_t) e * L.stride;
        for (int r = 0; r < F; ++r) {
            std::memcpy(b + L.tr[0] + (size_t) r * L.gate.rb, L.gate.q.data() + (size_t) L.g_row(e, r) * L.gate.rb,
                        L.gate.rb);
            std::memcpy(b + L.tr[1] + (size_t) r * L.up.rb, L.up.q.data() + (size_t) L.u_row(e, r) * L.up.rb, L.up.rb);
        }
        for (int r = 0; r < E; ++r)
            std::memcpy(b + L.tr[2] + (size_t) r * L.down.rb, L.down.q.data() + (size_t) L.d_row(e, r) * L.down.rb,
                        L.down.rb);
    }
    CU(cudaMalloc(&L.dev, L.host.size()));
    CU(cudaMemcpy(L.dev, L.host.data(), L.host.size(), cudaMemcpyHostToDevice));
}

// T tokens' picks: kNU distinct experts a token, drawn by a skewed popularity (rank i: 1 / (i + 1)^0.9, the ranks in a
// shuffled order of the experts), router weights in [0.2, 1.5]
struct Route {
    int T = 0;
    std::vector<int> ids;
    std::vector<float> w;
};
Route make_route(int T, int n_exp, uint64_t seed) {
    Route r;
    r.T = T;
    r.ids.resize((size_t) T * kNU);
    r.w.resize((size_t) T * kNU);
    std::mt19937_64 rng(seed);
    std::vector<int> perm((size_t) n_exp);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_real_distribution<float> uw(0.2f, 1.5f);
    for (int t = 0; t < T; ++t) {
        std::vector<double> p((size_t) n_exp);
        for (int i = 0; i < n_exp; ++i) p[(size_t) i] = 1.0 / std::pow(i + 1.0, 0.9);
        for (int k = 0; k < kNU; ++k) {
            double tot = 0;
            for (double v : p) tot += v;
            double u = u01(rng) * tot;
            int i = 0;
            while (i < n_exp - 1 && (u -= p[(size_t) i]) > 0) ++i;
            while (p[(size_t) i] == 0.0) i = (i + 1) % n_exp;   // (rounding at the end: the next one left)
            p[(size_t) i] = 0.0;
            r.ids[(size_t) t * kNU + k] = perm[(size_t) i];
            r.w[(size_t) t * kNU + k] = uw(rng);
        }
    }
    return r;
}

// the activation a run asks for
struct Act {
    int act = 0;   // iq::kActSwiglu / kActOai / kActSitu
    float limit = 0.0f;
    int pexp = 0;
    float beta = kBeta, lb = kLb;   // (kActSitu)
};
iq::Args base_args(const Layer& L, const Act& c) {
    iq::Args a;
    for (int i = 0; i < 3; ++i) a.tr[i] = L.tr[i];
    a.n_embd = L.E;
    a.n_ff = L.F;
    a.limit = c.limit;
    a.tg = L.tg;
    a.td = L.td;
    a.tu = L.tu == L.tg ? 0 : L.tu;
    a.act = c.act;
    a.pexp = c.pexp;
#if !defined(IK2_HIP_SITU_LEGACY_ONLY)
    a.situ_beta = c.beta;
    a.situ_lb = c.lb;
#endif
    return a;
}

// ---------------------------------------------------------------- the forms
// One token's kNU picks through the token form (a plan: group g = pick g, NE 1): h [kNU][F], part [kNU][E], y [E]
bool token_form(const Layer& L, const Act& c, const float* x, const int* ids, const float* w, cudaStream_t s,
                std::vector<float>& h, std::vector<float>& part, std::vector<float>& y) {
    const int E = L.E, F = L.F;
    std::vector<unsigned long long> plan(kNU);
    for (int k = 0; k < kNU; ++k) plan[(size_t) k] = (unsigned long long) (uintptr_t) (L.dev + (size_t) ids[k] * L.stride);
    std::vector<int8_t> xq((size_t) E), hq((size_t) kNU * F);
    std::vector<float> xd((size_t) E / 32), hd((size_t) kNU * F / 32);
    q8_host(x, E, xq.data(), xd.data());
    DBuf<unsigned long long> dplan(kNU);
    DBuf<int8_t> dxq((size_t) E), dhq((size_t) kNU * F);
    DBuf<float> dxd((size_t) E / 32), dh((size_t) kNU * F), dpart((size_t) kNU * E), dhd((size_t) kNU * F / 32);
    dplan.put(plan.data());
    dxq.put(xq.data());
    dxd.put(xd.data());
    iq::Args a = base_args(L, c);
    a.plan_ptr = dplan.p;
    a.xq = dxq.p;
    a.xd = dxd.p;
    a.h = dh.p;
    a.part = dpart.p;
    bool ok = iq::gate_up(a, kNU, 1, s);
    CU(cudaStreamSynchronize(s));
    h.resize((size_t) kNU * F);
    dh.get(h.data());
    for (int k = 0; k < kNU; ++k) q8_host(h.data() + (size_t) k * F, F, hq.data() + (size_t) k * F, hd.data() + (size_t) k * F / 32);
    dhq.put(hq.data());
    dhd.put(hd.data());
    a.hq = dhq.p;
    a.hd = dhd.p;
    ok = ok && iq::down(a, kNU, 1, s);
    CU(cudaStreamSynchronize(s));
    part.resize((size_t) kNU * E);
    dpart.get(part.data());
    y.assign((size_t) E, 0.0f);
    for (int k = 0; k < kNU; ++k)
        for (int i = 0; i < E; ++i) y[(size_t) i] += w[k] * part[(size_t) k * E + i];
    return ok;
}

// The window form (grouped: a group an expert, its entries t kNU + k; entry e's input row e >> 3): h [T kNU][F], part
// [T kNU][E], y [T][E]
bool window_form(const Layer& L, const Act& c, const std::vector<float>& x, const Route& rt, int T, cudaStream_t s,
                 std::vector<float>& h, std::vector<float>& part, std::vector<float>& y) {
    const int E = L.E, F = L.F, n = T * kNU;
    std::map<int, std::vector<int>> by;
    for (int j = 0; j < n; ++j) by[rt.ids[(size_t) j]].push_back(j);
    const int G = (int) by.size();
    std::vector<unsigned long long> gptr((size_t) G);
    std::vector<int> gstart((size_t) G + 1), gent;
    int ne_max = 0, g = 0;
    for (const auto& [e, v] : by) {
        gptr[(size_t) g] = (unsigned long long) (uintptr_t) (L.dev + (size_t) e * L.stride);
        gstart[(size_t) g++] = (int) gent.size();
        gent.insert(gent.end(), v.begin(), v.end());
        ne_max = std::max(ne_max, (int) v.size());
    }
    gstart[(size_t) G] = (int) gent.size();
    std::vector<int8_t> xq((size_t) T * E), hq((size_t) n * F);
    std::vector<float> xd((size_t) T * E / 32), hd((size_t) n * F / 32);
    for (int t = 0; t < T; ++t)
        q8_host(x.data() + (size_t) t * E, E, xq.data() + (size_t) t * E, xd.data() + (size_t) t * E / 32);
    DBuf<unsigned long long> dgp((size_t) G);
    DBuf<int> dgs((size_t) G + 1), dge((size_t) n), dng(1);
    DBuf<int8_t> dxq((size_t) T * E), dhq((size_t) n * F);
    DBuf<float> dxd((size_t) T * E / 32), dh((size_t) n * F), dpart((size_t) n * E), dhd((size_t) n * F / 32);
    dgp.put(gptr.data());
    dgs.put(gstart.data());
    dge.put(gent.data());
    dng.put(&G);
    dxq.put(xq.data());
    dxd.put(xd.data());
    iq::Args a = base_args(L, c);
    a.grp_ptr = dgp.p;
    a.grp_start = dgs.p;
    a.grp_ent = dge.p;
    a.n_groups = dng.p;
    a.xq = dxq.p;
    a.xd = dxd.p;
    a.h = dh.p;
    a.part = dpart.p;
    bool ok = iq::gate_up(a, G, ne_max, s);
    CU(cudaStreamSynchronize(s));
    h.resize((size_t) n * F);
    dh.get(h.data());
    for (int j = 0; j < n; ++j) q8_host(h.data() + (size_t) j * F, F, hq.data() + (size_t) j * F, hd.data() + (size_t) j * F / 32);
    dhq.put(hq.data());
    dhd.put(hd.data());
    a.hq = dhq.p;
    a.hd = dhd.p;
    ok = ok && iq::down(a, G, ne_max, s);
    CU(cudaStreamSynchronize(s));
    part.resize((size_t) n * E);
    dpart.get(part.data());
    y.assign((size_t) T * E, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < kNU; ++k)
            for (int i = 0; i < E; ++i)
                y[(size_t) t * E + i] += rt.w[(size_t) t * kNU + k] * part[((size_t) t * kNU + k) * E + i];
    return ok;
}

// The prompt form: the entries sorted by expert (bounds), the sets' gate / up (F32 x rows -> F16 h) and down (-> F32
// rows), each token's weighted sum: h16 [rows][F] (sorted), row_of[t kNU + k], counts[e], y [T][E]
bool set_form(const Layer& L, const Act& c, const std::vector<float>& x, const Route& rt, int T, cudaStream_t s,
              std::vector<uint16_t>& h16, std::vector<int>& row_of, std::vector<int>& counts, std::vector<float>& y,
              int* max_rows_out = nullptr) {
    const int E = L.E, F = L.F, n = T * kNU;
    std::vector<int> bounds((size_t) L.n_exp + 1), idx((size_t) n);
    row_of.assign((size_t) n, 0);
    counts.assign((size_t) L.n_exp, 0);
    int at = 0, max_rows = 0;
    for (int e = 0; e < L.n_exp; ++e) {
        bounds[(size_t) e] = at;
        for (int j = 0; j < n; ++j)
            if (rt.ids[(size_t) j] == e) {
                idx[(size_t) at] = j / kNU;
                row_of[(size_t) j] = at++;
            }
        counts[(size_t) e] = at - bounds[(size_t) e];
        max_rows = std::max(max_rows, counts[(size_t) e]);
    }
    bounds[(size_t) L.n_exp] = at;
    if (max_rows_out != nullptr) *max_rows_out = max_rows;
    DBuf<float> dx((size_t) T * E), dy((size_t) n * E), dgraw((size_t) n * F);
    DBuf<int> dbounds(bounds.size()), didx((size_t) n);
    DBuf<uint16_t> dh16((size_t) n * F);
    dx.put(x.data());
    dbounds.put(bounds.data());
    didx.put(idx.data());
    iq::SetArgs sa;
    sa.wbase = L.dev;
    sa.stride = L.stride;
    for (int i = 0; i < 3; ++i) sa.tr[i] = L.tr[i];
    sa.bounds = dbounds.p;
    sa.row0 = 0;
    sa.max_rows = max_rows;
    sa.n_exp = L.n_exp;
    sa.n_embd = E;
    sa.n_ff = F;
    sa.limit = c.limit;
    sa.xf = dx.p;
    sa.idx = didx.p;
    sa.ldx = E;
    sa.h = dh16.p;
    sa.ldo = F;
    sa.tg = L.tg;
    sa.td = L.td;
    sa.act = c.act;
    sa.tu = L.tu == L.tg ? 0 : L.tu;
    sa.graw = sa.tu != 0 ? dgraw.p : nullptr;
#if !defined(IK2_HIP_SITU_LEGACY_ONLY)
    sa.situ_beta = c.beta;
    sa.situ_lb = c.lb;
#endif
    bool ok = iq::set_gate_up(sa, s);
    CU(cudaStreamSynchronize(s));
    h16.resize((size_t) n * F);
    dh16.get(h16.data());
    iq::SetArgs sd = sa;
    sd.xf = nullptr;
    sd.idx = nullptr;
    sd.xh = dh16.p;
    sd.ldx = F;
    sd.h = nullptr;
    sd.y = dy.p;
    sd.ldo = E;
    ok = ok && iq::set_down(sd, s);
    CU(cudaStreamSynchronize(s));
    std::vector<float> yr((size_t) n * E);
    dy.get(yr.data());
    y.assign((size_t) T * E, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < kNU; ++k) {
            const float* r = yr.data() + (size_t) row_of[(size_t) t * kNU + k] * E;
            for (int i = 0; i < E; ++i) y[(size_t) t * E + i] += rt.w[(size_t) t * kNU + k] * r[i];
        }
    return ok;
}

// The CPU lane (the experts in host memory, as GGUF tensors: gate and up rows of one tensor): an expert's entries up to
// 16 a call (iq_cpu_gate_up's nt), the rows in pieces (r0 > 0 too) - h [T kNU][F], y [T][E]
void cpu_form(const Layer& L, const gg::IqCpu& f, float limit, const std::vector<float>& x, const Route& rt, int T,
              std::vector<float>& h, std::vector<float>& y) {
    const int E = L.E, F = L.F, n = T * kNU;
    std::map<int, std::vector<int>> by;
    for (int j = 0; j < n; ++j) by[rt.ids[(size_t) j]].push_back(j);
    std::vector<std::pair<int, std::vector<int>>> groups;
    for (const auto& [e, v] : by)
        for (size_t a = 0; a < v.size(); a += 16)
            groups.push_back({e, std::vector<int>(v.begin() + (long) a, v.begin() + (long) std::min(v.size(), a + 16))});
    h.assign((size_t) n * F, 0.0f);
    std::vector<float> ye((size_t) n * E, 0.0f);
    par((int) groups.size(), [&](int gi) {
        const auto& [e, ents] = groups[(size_t) gi];
        const uint8_t* b = L.host.data() + (size_t) e * L.stride;
        const int nt = (int) ents.size();
        std::vector<uint8_t> abuf((size_t) nt * f.act_bytes), hbuf((size_t) nt * f.h_bytes);
        std::vector<const void*> ap((size_t) nt), hp((size_t) nt);
        std::vector<float*> fp((size_t) nt), op((size_t) nt);
        for (int k = 0; k < nt; ++k) {
            const int t = ents[(size_t) k] / kNU;
            gg::iq_cpu_quant(f.gu_act, x.data() + (size_t) t * E, E, abuf.data() + (size_t) k * f.act_bytes);
            ap[(size_t) k] = abuf.data() + (size_t) k * f.act_bytes;
            fp[(size_t) k] = h.data() + (size_t) ents[(size_t) k] * F;
        }
        for (int r0 = 0; r0 < F; r0 += 576)
            gg::iq_cpu_gate_up(f, b + L.tr[0], b + L.tr[1], ap.data(), nt, fp.data(), r0, std::min(F, r0 + 576), limit);
        for (int k = 0; k < nt; ++k) {
            gg::iq_cpu_quant(f.d_act, fp[(size_t) k], F, hbuf.data() + (size_t) k * f.h_bytes);
            hp[(size_t) k] = hbuf.data() + (size_t) k * f.h_bytes;
            op[(size_t) k] = ye.data() + (size_t) ents[(size_t) k] * E;
        }
        for (int r0 = 0; r0 < E; r0 += 1152)
            gg::iq_cpu_down(f, b + L.tr[2], hp.data(), nt, op.data(), r0, std::min(E, r0 + 1152));
    });
    y.assign((size_t) T * E, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < kNU; ++k)
            for (int i = 0; i < E; ++i)
                y[(size_t) t * E + i] += rt.w[(size_t) t * kNU + k] * ye[((size_t) t * kNU + k) * E + i];
}

// ---------------------------------------------------------------- --dump / --check: the existing activations
struct Dump {
    std::vector<std::pair<std::string, std::vector<uint8_t>>> items;
    template <class T>
    void add(const std::string& name, const std::vector<T>& v) {
        std::vector<uint8_t> b(v.size() * sizeof(T));
        if (!b.empty()) std::memcpy(b.data(), v.data(), b.size());
        items.emplace_back(name, std::move(b));
    }
};

// Every form of the activations ds4_iq had before kActSitu, on small random experts (n_embd 1536, n_ff 512, 12
// experts, 8 picks): the token form (2 tokens), the window form (5 and 16 tokens), the sets (1 / 5 / 16 / 80 tokens -
// the light experts and the tiles, one expert of > 64 rows), the CPU lane (SwiGLU) and swiglu_f16
void legacy_cases(Dump& d, cudaStream_t s) {
    const int E = 1536, F = 512, N = 12;
    struct Cfg {
        const char* name;
        int tg, tu, td, act;
        float limit;
        int pexp;
    };
    std::vector<Cfg> cfgs;
    const int all[] = {iq::kIQ2_XXS, iq::kIQ2_XS, iq::kQ5_K, iq::kIQ2_S, iq::kIQ3_XXS, iq::kIQ3_S, iq::kIQ4_XS, iq::kQ4_K,
                       iq::kQ2_K, iq::kQ3_K, iq::kIQ1_S, iq::kIQ1_M, iq::kIQ4_NL, iq::kQ6_K, iq::kQ8_0};
    for (int t : all) cfgs.push_back({"swiglu", t, t, t, iq::kActSwiglu, 7.0f, 0});
    cfgs.push_back({"swiglu-nolimit", iq::kIQ2_XS, iq::kIQ2_XS, iq::kIQ3_XXS, iq::kActSwiglu, 0.0f, 0});
    cfgs.push_back({"swiglu-pexp", iq::kIQ2_XS, iq::kIQ2_XS, iq::kIQ3_XXS, iq::kActSwiglu, 7.0f, 1});
    cfgs.push_back({"swiglu-pexp", iq::kQ4_K, iq::kQ4_K, iq::kQ6_K, iq::kActSwiglu, 7.0f, 1});
    for (int t : {iq::kIQ2_XS, iq::kQ4_K, iq::kIQ1_M, iq::kIQ4_NL, iq::kIQ3_XXS})
        cfgs.push_back({"oai", t, t, iq::kIQ4_XS, iq::kActOai, 7.0f, 0});
    cfgs.push_back({"oai-split", iq::kIQ2_XS, iq::kIQ3_XXS, iq::kIQ4_XS, iq::kActOai, 7.0f, 0});
    cfgs.push_back({"oai-split", iq::kQ4_K, iq::kQ5_K, iq::kQ6_K, iq::kActOai, 7.0f, 0});
    uint64_t seed = 1000;
    for (const Cfg& c : cfgs) {
        Layer L;
        // (gate and up sums of about N(0, 3): the limit clamps some)
        make_layer(L, c.tg, c.tu, c.td, E, F, N, 3.0f / std::sqrt((float) E), 3.0f / std::sqrt((float) E),
                   1.0f / std::sqrt((float) F), seed);
        const Route rt = make_route(80, N, seed + 7);
        std::vector<float> x((size_t) 80 * E);
        {
            std::mt19937_64 rng(seed + 9);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (float& v : x) v = nd(rng);
        }
        seed += 100;
        const Act a{c.act, c.limit, c.pexp};
        const std::string tag = std::string(c.name) + "/" + iq::name(c.tg) + "/" + iq::name(c.tu) + "/" + iq::name(c.td);
        std::vector<float> h, part, y;
        for (int t = 0; t < 2; ++t) {
            CHECK(token_form(L, a, x.data() + (size_t) t * E, rt.ids.data() + (size_t) t * kNU,
                             rt.w.data() + (size_t) t * kNU, s, h, part, y));
            d.add(tag + "/token" + std::to_string(t) + "/h", h);
            d.add(tag + "/token" + std::to_string(t) + "/y", y);
        }
        for (int T : {5, 16}) {
            CHECK(window_form(L, a, x, rt, T, s, h, part, y));
            d.add(tag + "/window" + std::to_string(T) + "/h", h);
            d.add(tag + "/window" + std::to_string(T) + "/y", y);
        }
        for (int T : {1, 5, 16, 80}) {
            std::vector<uint16_t> h16;
            std::vector<int> row_of, counts;
            CHECK(set_form(L, a, x, rt, T, s, h16, row_of, counts, y));
            d.add(tag + "/set" + std::to_string(T) + "/h", h16);
            d.add(tag + "/set" + std::to_string(T) + "/y", y);
        }
        gg::IqCpu f;
        std::string err;
        if (c.act == iq::kActSwiglu && c.tu == c.tg && gg::iq_cpu_fmt(c.tg, c.td, E, F, f, err))
            for (int T : {1, 5}) {
                cpu_form(L, f, c.limit, x, rt, T, h, y);
                d.add(tag + "/cpu" + std::to_string(T) + "/h", h);
                d.add(tag + "/cpu" + std::to_string(T) + "/y", y);
            }
    }
    // swiglu_f16 (the busy experts' path): F32 gate | up rows -> F16
    {
        const int n = 37, ff = 512, ld = 2 * ff + 64, ldh = ff + 32;
        std::vector<float> gu((size_t) n * ld);
        std::mt19937_64 rng(77);
        std::normal_distribution<float> nd(0.0f, 4.0f);
        for (float& v : gu) v = nd(rng);
        DBuf<float> dgu(gu.size());
        DBuf<uint16_t> dh((size_t) n * ldh);
        dgu.put(gu.data());
        for (float lim : {0.0f, 7.0f}) {
            iq::swiglu_f16(dgu.p, ld, n, ff, lim, dh.p, ldh, s);
            CU(cudaStreamSynchronize(s));
            std::vector<uint16_t> h((size_t) n * ldh);
            dh.get(h.data());
            d.add("swiglu_f16/limit" + std::to_string((int) lim), h);
        }
    }
}

bool write_dump(const Dump& d, const char* path) {
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) return false;
    for (const auto& [name, b] : d.items) {
        const uint32_t nl = (uint32_t) name.size();
        const uint64_t bl = b.size();
        std::fwrite(&nl, 4, 1, f);
        std::fwrite(name.data(), 1, nl, f);
        std::fwrite(&bl, 8, 1, f);
        if (bl) std::fwrite(b.data(), 1, bl, f);
    }
    std::fclose(f);
    return true;
}
bool read_dump(Dump& d, const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    for (;;) {
        uint32_t nl = 0;
        uint64_t bl = 0;
        if (std::fread(&nl, 4, 1, f) != 1) break;
        std::string name(nl, '\0');
        bool ok = std::fread(&name[0], 1, nl, f) == nl && std::fread(&bl, 8, 1, f) == 1;
        std::vector<uint8_t> b(bl);
        ok = ok && (bl == 0 || std::fread(b.data(), 1, bl, f) == bl);
        if (!ok) {
            std::fclose(f);
            return false;
        }
        d.items.emplace_back(std::move(name), std::move(b));
    }
    std::fclose(f);
    return true;
}
int dump_main(bool check, const char* path) {
    cudaStream_t s;
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    Dump d;
    legacy_cases(d, s);
    size_t bytes = 0;
    for (const auto& it : d.items) bytes += it.second.size();
    if (!check) {
        if (!write_dump(d, path)) {
            std::fprintf(stderr, "cannot write %s\n", path);
            return 2;
        }
        std::printf("ik2_hip_situ: %zu outputs (%zu bytes) of the existing activations written to %s%s\n",
                    d.items.size(), bytes, path, fails ? " - with FAILURES" : "");
        return fails ? 1 : 0;
    }
    Dump ref;
    if (!read_dump(ref, path)) {
        std::fprintf(stderr, "cannot read %s\n", path);
        return 2;
    }
    std::map<std::string, const std::vector<uint8_t>*> by;
    for (const auto& it : ref.items) by[it.first] = &it.second;
    int differ = 0, missing = 0;
    for (const auto& [name, b] : d.items) {
        const auto it = by.find(name);
        if (it == by.end()) {
            ++missing;
            std::printf("  missing in %s: %s\n", path, name.c_str());
            continue;
        }
        if (*it->second != b) {
            ++differ;
            size_t nd = 0;
            for (size_t i = 0; i < std::min(b.size(), it->second->size()); ++i) nd += b[i] != (*it->second)[i];
            std::printf("  DIFFERS: %s (%zu of %zu bytes)\n", name.c_str(), nd, b.size());
        }
    }
    std::printf("ik2_hip_situ: %zu outputs (%zu bytes) of the existing activations against %s: %d differ, %d missing\n",
                d.items.size(), bytes, path, differ, missing);
    return differ || missing || fails ? 1 : 0;
}

#if !defined(IK2_HIP_SITU_LEGACY_ONLY)
// ---------------------------------------------------------------- SITU against the double reference
// The pools' rows dotted with every token's input, in double: exact (x as given), with x as the token / window / light
// forms round it (int8 per 32) and as the CPU lane does (its Q8_K); and each token's routed output, exact
struct Ref {
    int T = 0, T8 = 0;
    std::vector<double> G, U, G8, U8, GK, UK;   // [t][F]: expert e's gate row r . x_t = G[t][g_row(e, r)], ...
    std::vector<double> y;                      // [T][E]
    double sw_dist = 0;                         // |SITU - SwiGLU| / |SITU| over the entries' activations (exact)
    bool have_k = false;
};
double dotd(const float* w, const double* x, int k) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < k; i += 4) {
        s0 += (double) w[i] * x[i];
        s1 += (double) w[i + 1] * x[i + 1];
        s2 += (double) w[i + 2] * x[i + 2];
        s3 += (double) w[i + 3] * x[i + 3];
    }
    return (s0 + s1) + (s2 + s3);
}
// the CPU lane's activation for x (its Q8_K: an F32 scale a 256 values, then 256 int8 and 16 int16 sums), decoded
bool q8k_decode(int act_type, const float* x, int k, double* out) {
    if (act_type != (int) GGML_TYPE_Q8_K) return false;
    std::vector<uint8_t> b(ggml_row_size(GGML_TYPE_Q8_K, k));
    gg::iq_cpu_quant(act_type, x, k, b.data());
    for (int blk = 0; blk < k / 256; ++blk) {
        const uint8_t* p = b.data() + (size_t) blk * 292;
        float d;
        std::memcpy(&d, p, 4);
        for (int i = 0; i < 256; ++i) out[blk * 256 + i] = (double) d * (double) (int8_t) p[4 + i];
    }
    return true;
}
Ref reference(const Layer& L, const std::vector<float>& x, const Route& rt, int T8, int cpu_act) {
    const int E = L.E, F = L.F, T = rt.T;
    Ref R;
    R.T = T;
    R.T8 = T8;
    std::vector<double> xe((size_t) T * E), x8((size_t) T8 * E), xk((size_t) T8 * E);
    for (size_t i = 0; i < xe.size(); ++i) xe[i] = x[i];
    R.have_k = true;
    for (int t = 0; t < T8; ++t) {
        std::vector<int8_t> q((size_t) E);
        std::vector<float> d((size_t) E / 32);
        q8_host(x.data() + (size_t) t * E, E, q.data(), d.data());
        for (int i = 0; i < E; ++i) x8[(size_t) t * E + i] = (double) d[(size_t) i / 32] * q[(size_t) i];
        R.have_k = R.have_k && q8k_decode(cpu_act, x.data() + (size_t) t * E, E, xk.data() + (size_t) t * E);
    }
    R.G.assign((size_t) T * F, 0);
    R.U.assign((size_t) T * F, 0);
    R.G8.assign((size_t) T8 * F, 0);
    R.U8.assign((size_t) T8 * F, 0);
    R.GK.assign((size_t) T8 * F, 0);
    R.UK.assign((size_t) T8 * F, 0);
    par(F, [&](int p) {
        const float* wg = L.gate.w.data() + (size_t) p * E;
        const float* wu = L.up.w.data() + (size_t) p * E;
        for (int t = 0; t < T; ++t) {
            R.G[(size_t) t * F + p] = dotd(wg, xe.data() + (size_t) t * E, E);
            R.U[(size_t) t * F + p] = dotd(wu, xe.data() + (size_t) t * E, E);
        }
        for (int t = 0; t < T8; ++t) {
            R.G8[(size_t) t * F + p] = dotd(wg, x8.data() + (size_t) t * E, E);
            R.U8[(size_t) t * F + p] = dotd(wu, x8.data() + (size_t) t * E, E);
            if (R.have_k) {
                R.GK[(size_t) t * F + p] = dotd(wg, xk.data() + (size_t) t * E, E);
                R.UK[(size_t) t * F + p] = dotd(wu, xk.data() + (size_t) t * E, E);
            }
        }
    });
    const int n = T * kNU;
    std::vector<double> ye((size_t) n * E), d2((size_t) n), s2((size_t) n);
    std::vector<float> wd((size_t) E * F);   // (the down's rows in double-friendly order: as they are)
    par(n, [&](int j) {
        const int t = j / kNU, e = rt.ids[(size_t) j];
        std::vector<double> a((size_t) F);
        double dd = 0, ss = 0;
        for (int r = 0; r < F; ++r) {
            const double g = R.G[(size_t) t * F + L.g_row(e, r)], u = R.U[(size_t) t * F + L.u_row(e, r)];
            a[(size_t) r] = situ_ref(g, u);
            const double sw = swiglu_ref(g, u);
            dd += (a[(size_t) r] - sw) * (a[(size_t) r] - sw);
            ss += a[(size_t) r] * a[(size_t) r];
        }
        for (int i = 0; i < E; ++i) ye[(size_t) j * E + i] = dotd(L.down.w.data() + (size_t) L.d_row(e, i) * F, a.data(), F);
        d2[(size_t) j] = dd;
        s2[(size_t) j] = ss;
    });
    R.y.assign((size_t) T * E, 0);
    double D = 0, S = 0;
    for (int j = 0; j < n; ++j) {
        D += d2[(size_t) j];
        S += s2[(size_t) j];
        for (int i = 0; i < E; ++i) R.y[(size_t) (j / kNU) * E + i] += (double) rt.w[(size_t) j] * ye[(size_t) j * E + i];
    }
    R.sw_dist = std::sqrt(D / std::max(S, 1e-300));
    return R;
}
// entry (t, e)'s reference activation [F]: which = 0 exact, 1 int8 x, 2 the CPU lane's Q8_K x
void href(const Ref& R, const Layer& L, int which, int t, int e, double* out, double beta = kBeta, double lb = kLb) {
    const std::vector<double>& G = which == 0 ? R.G : which == 1 ? R.G8 : R.GK;
    const std::vector<double>& U = which == 0 ? R.U : which == 1 ? R.U8 : R.UK;
    for (int r = 0; r < L.F; ++r)
        out[r] = situ_ref(G[(size_t) t * L.F + L.g_row(e, r)], U[(size_t) t * L.F + L.u_row(e, r)], beta, lb);
}

// an output's error against a reference: rel = |a - r|_2 / |r|_2, max = max |a - r| / max |r|
struct Err {
    double d2 = 0, r2 = 0, dmax = 0, rmax = 0;
    void add(double a, double r) {
        d2 += (a - r) * (a - r);
        r2 += r * r;
        dmax = std::max(dmax, std::fabs(a - r));
        rmax = std::max(rmax, std::fabs(r));
    }
    double rel() const { return std::sqrt(d2 / std::max(r2, 1e-300)); }
    double max() const { return dmax / std::max(rmax, 1e-300); }
};
Err y_err(const std::vector<float>& y, const Ref& R, int T, int E) {
    Err e;
    for (size_t i = 0; i < (size_t) T * E; ++i) e.add(y[i], R.y[i]);
    return e;
}

// the worst errors a layer's forms show (the summary)
struct Worst {
    double h = 0, y = 0;
    void see(const Err& eh, const Err& ey) {
        h = std::max(h, eh.max());
        y = std::max(y, ey.max());
    }
};

void situ_layer(int tg, int td, const char* label, cudaStream_t s, std::map<std::string, Worst>& worst) {
    const int E = 4608, F = 2304, N = 24, Tmax = 160, T8 = 64;
    const bool light = !(std::getenv("MAYA_DS4_NO_LIGHT") != nullptr && std::getenv("MAYA_DS4_NO_LIGHT")[0] == '1');
    std::printf("%s: gate_up %s, down %s - n_embd %d, n_ff %d, %d experts, %d picks a token (SITU beta %g, lb %g)\n",
                label, iq::name(tg), iq::name(td), E, F, N, kNU, kBeta, kLb);
    // gate sums of about N(0, 6), up of N(0, 30): both tanh's well into their bends (the SITU is not SwiGLU there)
    Layer L;
    make_layer(L, tg, tg, td, E, F, N, 6.0f / std::sqrt((float) E), 30.0f / std::sqrt((float) E),
               0.05f / std::sqrt((float) F), 0x1A2Bull + (uint64_t) tg * 977 + (uint64_t) td);
    const Route rt = make_route(Tmax, N, 0x5EEDull + (uint64_t) tg);
    std::vector<float> x((size_t) Tmax * E);
    {
        std::mt19937_64 rng(0xC0FFEEull + (uint64_t) tg);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : x) v = nd(rng);
    }
    gg::IqCpu f;
    std::string err;
    const bool cpu_ok = gg::iq_cpu_fmt(tg, td, E, F, f, err);
    if (!cpu_ok) std::printf("  CPU lane: not in this build (%s)\n", err.c_str());
    CHECK(cpu_ok);
    f.act = gg::kIqActSitu;
    f.situ_beta = kBeta;
    f.situ_lb = kLb;
    const Ref R = reference(L, x, rt, T8, cpu_ok ? f.gu_act : -1);
    std::printf("  reference: |SITU - SwiGLU| / |SITU| of the entries' activations = %.3f (the test tells them apart)\n",
                R.sw_dist);
    CHECK(R.sw_dist > 0.2);
    const Act a{iq::kActSitu, 0.0f, 0};
    std::vector<double> hr((size_t) F);
    // the token form: tokens 0..15, each a decode step
    std::vector<std::vector<float>> tok_h(16), tok_part(16);
    {
        Err eh, ey;
        std::vector<float> h, part, y;
        for (int t = 0; t < 16; ++t) {
            CHECK(token_form(L, a, x.data() + (size_t) t * E, rt.ids.data() + (size_t) t * kNU,
                             rt.w.data() + (size_t) t * kNU, s, h, part, y));
            for (int k = 0; k < kNU; ++k) {
                href(R, L, 1, t, rt.ids[(size_t) t * kNU + k], hr.data());
                for (int r = 0; r < F; ++r) eh.add(h[(size_t) k * F + r], hr[(size_t) r]);
            }
            for (int i = 0; i < E; ++i) ey.add(y[(size_t) i], R.y[(size_t) t * E + i]);
            tok_h[(size_t) t] = h;
            tok_part[(size_t) t] = part;
        }
        std::printf("  token form   16 tokens one at a time: h max %.2e (int8 x) | y rel %.2e max %.2e\n", eh.max(),
                    ey.rel(), ey.max());
        CHECK(eh.max() < 1e-4);
        CHECK(ey.rel() < 3e-2 && ey.max() < 6e-2);
        worst["token"].see(eh, ey);
    }
    // the window form
    for (int T : {1, 8, 16, 64}) {
        std::vector<float> h, part, y;
        CHECK(window_form(L, a, x, rt, T, s, h, part, y));
        Err eh;
        for (int j = 0; j < T * kNU; ++j) {
            href(R, L, 1, j / kNU, rt.ids[(size_t) j], hr.data());
            for (int r = 0; r < F; ++r) eh.add(h[(size_t) j * F + r], hr[(size_t) r]);
        }
        const Err ey = y_err(y, R, T, E);
        // its entries against the token form's: the same kernel and order, the same bits
        int diff = 0, cmp = 0;
        for (int t = 0; t < std::min(T, 16); ++t)
            for (int k = 0; k < kNU; ++k) {
                diff += std::memcmp(h.data() + ((size_t) t * kNU + k) * F, tok_h[(size_t) t].data() + (size_t) k * F,
                                    sizeof(float) * F) != 0;
                diff += std::memcmp(part.data() + ((size_t) t * kNU + k) * E, tok_part[(size_t) t].data() + (size_t) k * E,
                                    sizeof(float) * E) != 0;
                cmp += 2;
            }
        std::printf("  window form  %3d rows: h max %.2e (int8 x) | y rel %.2e max %.2e | vs token form: %d of %d rows "
                    "differ\n", T, eh.max(), ey.rel(), ey.max(), diff, cmp);
        CHECK(eh.max() < 1e-4);
        CHECK(ey.rel() < 3e-2 && ey.max() < 6e-2);
        CHECK(diff == 0);
        worst["window"].see(eh, ey);
    }
    // the prompt form: light experts (<= 16 rows: int8 x, as the token form) and the tiles (F16 / F32 x)
    for (int T : {1, 8, 16, 64, 160}) {
        std::vector<uint16_t> h16;
        std::vector<int> row_of, counts;
        std::vector<float> y;
        int mr = 0;
        CHECK(set_form(L, a, x, rt, T, s, h16, row_of, counts, y, &mr));
        Err el, et;   // the light experts' rows against the int8 reference, the tiles' against the exact one
        int nl = 0, ntl = 0;
        for (int j = 0; j < T * kNU; ++j) {
            const int t = j / kNU, e = rt.ids[(size_t) j];
            const bool lt = light && counts[(size_t) e] <= 16;
            if (lt && t >= T8) continue;   // (the int8 reference: the first T8 tokens)
            href(R, L, lt ? 1 : 0, t, e, hr.data());
            const uint16_t* hp = h16.data() + (size_t) row_of[(size_t) j] * F;
            for (int r = 0; r < F; ++r) (lt ? el : et).add(h2f(hp[r]), hr[(size_t) r]);
            (lt ? nl : ntl) += 1;
        }
        const Err ey = y_err(y, R, T, E);
        std::printf("  prompt form  %3d rows (%3d an expert at most): h max %.2e (%d light rows, int8 x) %.2e (%d tile "
                    "rows, exact x) | y rel %.2e max %.2e\n", T, mr, el.max(), nl, et.max(), ntl, ey.rel(), ey.max());
        CHECK(el.max() < 2e-3);
        CHECK(et.max() < 1e-2);
        CHECK(ey.rel() < 3e-2 && ey.max() < 6e-2);
        Err eh = el;
        eh.dmax = std::max(el.max(), et.max());
        eh.rmax = 1.0;
        worst["prompt"].see(eh, ey);
    }
    // the CPU lane
    if (cpu_ok)
        for (int T : {1, 8, 16, 64}) {
            std::vector<float> h, y;
            cpu_form(L, f, 0.0f, x, rt, T, h, y);
            Err eh;
            if (R.have_k)
                for (int j = 0; j < T * kNU; ++j) {
                    href(R, L, 2, j / kNU, rt.ids[(size_t) j], hr.data());
                    for (int r = 0; r < F; ++r) eh.add(h[(size_t) j * F + r], hr[(size_t) r]);
                }
            const Err ey = y_err(y, R, T, E);
            std::printf("  CPU lane     %3d rows: h max %.2e (its Q8_K x) | y rel %.2e max %.2e  (own rows: %s / %s)\n", T,
                        eh.max(), ey.rel(), ey.max(), gg::iq_cpu_own_rows(tg) ? "yes" : "no",
                        gg::iq_cpu_own_rows(td) ? "yes" : "no");
            CHECK(R.have_k);
            CHECK(eh.max() < 1e-4);
            CHECK(ey.rel() < 4e-2 && ey.max() < 8e-2);
            worst["cpu"].see(eh, ey);
        }
    // the parameters reach every form (not Inkling's defaults the structs hold): beta 2.5, lb 0 (the up as it is) - the
    // activations against the reference of those (the token form, the window form, the sets' light experts and
    // tiles, the CPU lane)
    {
        Act b = a;
        b.beta = 2.5f;
        b.lb = 0.0f;
        Err et, ew, el, ec, es;
        std::vector<float> h, part, y;
        CHECK(token_form(L, b, x.data(), rt.ids.data(), rt.w.data(), s, h, part, y));
        for (int k = 0; k < kNU; ++k) {
            href(R, L, 1, 0, rt.ids[(size_t) k], hr.data(), b.beta, b.lb);
            for (int r = 0; r < F; ++r) et.add(h[(size_t) k * F + r], hr[(size_t) r]);
        }
        CHECK(window_form(L, b, x, rt, 8, s, h, part, y));
        for (int j = 0; j < 8 * kNU; ++j) {
            href(R, L, 1, j / kNU, rt.ids[(size_t) j], hr.data(), b.beta, b.lb);
            for (int r = 0; r < F; ++r) ew.add(h[(size_t) j * F + r], hr[(size_t) r]);
        }
        for (int T : {8, 160}) {
            std::vector<uint16_t> h16;
            std::vector<int> row_of, counts;
            CHECK(set_form(L, b, x, rt, T, s, h16, row_of, counts, y));
            for (int j = 0; j < T * kNU; ++j) {
                const int t = j / kNU, e = rt.ids[(size_t) j];
                const bool lt = light && counts[(size_t) e] <= 16;
                if (lt && t >= T8) continue;
                href(R, L, lt ? 1 : 0, t, e, hr.data(), b.beta, b.lb);
                for (int r = 0; r < F; ++r) (lt ? el : es).add(h2f(h16[(size_t) row_of[(size_t) j] * F + r]), hr[(size_t) r]);
            }
        }
        if (cpu_ok && R.have_k) {
            gg::IqCpu fb = f;
            fb.situ_beta = b.beta;
            fb.situ_lb = b.lb;
            cpu_form(L, fb, 0.0f, x, rt, 8, h, y);
            for (int j = 0; j < 8 * kNU; ++j) {
                href(R, L, 2, j / kNU, rt.ids[(size_t) j], hr.data(), b.beta, b.lb);
                for (int r = 0; r < F; ++r) ec.add(h[(size_t) j * F + r], hr[(size_t) r]);
            }
        }
        std::printf("  beta %.1f, lb %.0f: h max token %.2e, window %.2e, light %.2e, tiles %.2e, CPU %.2e\n", b.beta,
                    b.lb, et.max(), ew.max(), el.max(), es.max(), ec.max());
        CHECK(et.max() < 1e-4 && ew.max() < 1e-4 && ec.max() < 1e-4);
        CHECK((!light || el.max() < 2e-3) && es.max() < 1e-2);
    }
}

// iq::situ_f16 (the busy experts' path: the caller's F32 gate | up rows -> F16 h) against the double reference
void situ_f16_case(cudaStream_t s) {
    const int n = 37, ff = 2304, ld = 2 * ff + 64, ldh = ff + 32;
    std::vector<float> gu((size_t) n * ld);
    std::mt19937_64 rng(78);
    std::normal_distribution<float> ng(0.0f, 6.0f), nu(0.0f, 30.0f);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < ld; ++c) gu[(size_t) r * ld + c] = c < ff ? ng(rng) : nu(rng);
    DBuf<float> dgu(gu.size());
    DBuf<uint16_t> dh((size_t) n * ldh);
    dgu.put(gu.data());
    // (the up through lb tanh(u / lb) at 25; as it is at lb 0 - then the values reach the hundreds: F16's step there)
    for (const auto& [beta, lb] : {std::pair<float, float>{kBeta, kLb}, std::pair<float, float>{2.5f, 0.0f}}) {
        iq::situ_f16(dgu.p, ld, n, ff, beta, lb, dh.p, ldh, s);
        CU(cudaStreamSynchronize(s));
        std::vector<uint16_t> h((size_t) n * ldh);
        dh.get(h.data());
        double worst = 0;
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < ff; ++c) {
                const double ref = situ_ref(gu[(size_t) r * ld + c], gu[(size_t) r * ld + ff + c], beta, lb);
                worst = std::max(worst, std::fabs(h2f(h[(size_t) r * ldh + c]) - ref) / (std::fabs(ref) + 1e-3));
            }
        std::printf("situ_f16 (beta %.1f, lb %.0f): %d rows of %d: max |h - ref| / (|ref| + 1e-3) = %.2e (F16)\n", beta,
                    lb, n, ff, worst);
        CHECK(worst < 1e-3);
    }
}

// what the dispatch refuses (false, nothing launched): a SITU without a positive beta, gate and up of two types
void situ_refusals(cudaStream_t s) {
    iq::Args a;
    a.n_embd = 4608;
    a.n_ff = 2304;
    a.tg = iq::kIQ2_XS;
    a.td = iq::kIQ3_XXS;
    a.act = iq::kActSitu;
    a.situ_beta = 0.0f;
    CHECK(!iq::gate_up(a, 1, 1, s));
    a.situ_beta = kBeta;
    a.tu = iq::kIQ3_XXS;
    CHECK(!iq::gate_up(a, 1, 1, s));
    iq::SetArgs sa;
    sa.n_embd = 4608;
    sa.n_ff = 2304;
    sa.n_exp = 1;
    sa.tg = iq::kIQ2_XS;
    sa.td = iq::kIQ3_XXS;
    sa.act = iq::kActSitu;
    sa.situ_beta = -1.0f;
    CHECK(!iq::set_gate_up(sa, s));
    CHECK(!iq::light_gate_up(sa, s));
    sa.situ_beta = kBeta;
    sa.tu = iq::kIQ2_XXS;
    CHECK(!iq::set_gate_up(sa, s));
    sa.tu = 0;
    sa.tg = sa.td = 39;   // (MXFP4: ds4_gguf.cu's experts in the light form - SwiGLU only)
    CHECK(!iq::light_gate_up(sa, s));
    CU(cudaStreamSynchronize(s));
    std::printf("refusals (beta <= 0, gate / up of two types, MXFP4's light form): %s\n", fails ? "FAILED" : "ok");
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc >= 2 ? argv[1] : "";
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("ik2_hip_situ_test: no GPU, skipped\n");
        return 77;
    }
    if (mode == "--dump" || mode == "--check") {
        if (argc < 3) {
            std::fprintf(stderr, "usage: %s --dump|--check FILE\n", argv[0]);
            return 2;
        }
        return dump_main(mode == "--check", argv[2]);
    }
#if defined(IK2_HIP_SITU_LEGACY_ONLY)
    std::fprintf(stderr, "usage: %s --dump|--check FILE (this build: the existing activations only)\n", argv[0]);
    return 2;
#else
    static_assert(gg::kIqActSwiglu == iq::kActSwiglu && gg::kIqActSitu == iq::kActSitu, "the CPU lane's activation ids");
    cudaStream_t s;
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaDeviceProp prop{};
    CU(cudaGetDeviceProperties(&prop, 0));
#if defined(STRATA_USE_HIP)
    std::printf("ik2_hip_situ_test on %s (%s)", prop.name, prop.gcnArchName);
#else
    std::printf("ik2_hip_situ_test on %s (sm_%d%d)", prop.name, prop.major, prop.minor);
#endif
    const char* envs[] = {"MAYA_DS4_NO_TC", "MAYA_DS4_NO_LIGHT", "DS4_IQ_TC_RM", "DS4_IQ_OWN"};
    for (const char* e : envs)
        if (const char* v = std::getenv(e)) std::printf(" %s=%s", e, v);
    std::printf("\n");
    std::map<std::string, Worst> worst;
    situ_layer(iq::kIQ2_XS, iq::kIQ3_XXS, "layers 0-39", s, worst);
    situ_layer(iq::kIQ3_XXS, iq::kIQ4_XS, "layers 40-47", s, worst);
    situ_f16_case(s);
    situ_refusals(s);
    std::printf("worst (max |err| / max |ref|): ");
    for (const auto& [k, w] : worst) std::printf(" %s h %.2e y %.2e;", k.c_str(), w.h, w.y);
    std::printf("\nik2_hip_situ_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
#endif
}
