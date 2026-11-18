// tests/k31_kda_test.cu - the K3.1 KDA layer adapter (src/k31/k31_kda.hpp) held to ref/kimi_k31.py's kda_layer and
// to a float64 reference, plus the mutations, each of which must move the layer output by >= 10x its bound.
//
//   k31_kda_test --selftest      K3.1's shape (96 heads x 128, conv 4), one head with tiny q/k (the l2 eps shows),
//                                two calls vs one, a 4096-token run with slow decay (the float32 state's drift),
//                                the refusals
//   k31_kda_test --dumps DIR     every KDA layer of the fixture (python tools/kimi_k31_kda_dump.py --out DIR)
//
// The composed path is the engine's: kda_layer_launch (GLM conv x3, GDN l2 norm x2, k31 prep, GLM gate, recurrence,
// output gate), then attn_output as a cuBLAS SGEMM (CUBLAS_DEFAULT_MATH, never TF32).  Errors are max |d| / rms(ref)
// per seam.  The float64 reference here mirrors the oracle's arithmetic (float32 seams, float64 inside, float64
// state) and carries the wrong readings as flags: the gate's lower bound -1, the qwen softplus gate, dt_bias dropped,
// A_log's zero tail read for the heads (ssm_a = -1) or the heads shifted by one, the conv taps reversed, no l2 norm,
// an rms (mean) l2 norm, the rms eps for the l2 eps, no delta correction, beta without its sigmoid, no 1/sqrt(dk),
// a SiLU output gate, and the state or the conv history reset between two calls.
#include "../src/k31/k31_kda.hpp"
#include "strata/kernels/glm_kda.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Vec = std::vector<float>;
constexpr double kMinNormal = 1.17549435082228751e-38;

// bounds: ~3x the measured gap (2026-11-05, RTX 3090: fixture / K3.1-shape selftest / the 4096-token run)
constexpr double kBoundConv = 5e-6;     // silu(conv), v: float32 taps vs float64 (measured 1.4e-6 / 1.8e-6)
constexpr double kBoundQK = 5e-6;       // the l2-normalised q, k (6.7e-7 / 1.4e-6)
constexpr double kBoundG1 = 1e-6;       // the gate (2.3e-7 / 1.7e-7)
constexpr double kBoundBetaUlp = 1.0;   // the k31 float64 sigmoid (0 ULP; numpy's exp may round one float32 apart)
constexpr double kBoundScan = 6e-5;     // the recurrence output and the state: float32 state and sums against the
                                        // oracle's float64 (4.1e-6 / 2.2e-5 / 1.4e-5, no growth over 4096 tokens;
                                        // precise expf/sqrtf in glm_kda.cu measured no better: 4.5e-6 / 1.8e-5 / 2.3e-5)
constexpr double kBoundOut = 5e-5;      // gated and kda_out (4.1e-6 / 1.2e-5 / 1.5e-5)
constexpr double kBoundRef = 1e-6;      // this file's float64 reference against the oracle, through `gated` (measured 0:
                                        // it reproduces every seam bit for bit); kda_out is held to kBoundOut, as the
                                        // oracle's attn_output is a float32 GEMM
const double kNeed = 10.0 * kBoundOut;

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
void blas_ok(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s: cublas %d\n", what, (int) s); std::exit(2); }
}
double ulp_unit(double r) {
    const double ar = std::fabs(r);
    if (ar < kMinNormal) return kMinNormal;
    const float f = (float) ar;
    return std::max((double) std::nextafter(f, std::numeric_limits<float>::infinity()) - (double) f, kMinNormal);
}
double max_ulp(const Vec& a, const Vec& r) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double) a[i] - (double) r[i]) / ulp_unit(r[i]));
    return m;
}
double rel_err(const Vec& a, const Vec& r) {
    if (a.size() != r.size()) return 1e300;
    double num = 0, ss = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) r[i]);
        num = std::max(num, d != d ? 1e300 : d);
        ss += (double) r[i] * r[i];
    }
    return num / std::max(std::sqrt(ss / std::max<size_t>(r.size(), 1)), 1e-30);
}
bool same_bits(const Vec& a, const Vec& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0; }

std::vector<void*> g_dev;
template <class T>
T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    cuda_ok(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)), "malloc");
    if (!h.empty()) cuda_ok(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    g_dev.push_back(d);
    return d;
}
float* dzeros(size_t n) {
    float* d = nullptr;
    cuda_ok(cudaMalloc(&d, std::max<size_t>(n, 1) * 4), "malloc");
    cuda_ok(cudaMemset(d, 0, std::max<size_t>(n, 1) * 4), "memset");
    g_dev.push_back(d);
    return d;
}
Vec host(const float* d, size_t n) {
    Vec h(n);
    cuda_ok(cudaDeviceSynchronize(), "sync");
    cuda_ok(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost), "d2h");
    return h;
}
void free_all() {
    for (void* p : g_dev) cudaFree(p);
    g_dev.clear();
}

struct Layer {
    int T = 0, H = 0, dk = 0, dc = 0, n_embd = 0;
    float lb = -5.0f, eps_l2 = 1e-6f, eps_rms = 1e-5f;
    Vec xq, xk, xv, f_pre, beta_pre, g2, conv[3], dt_bias, ssm_a, norm_w, wo;
    int d() const { return H * dk; }
};
struct Out {
    Vec q, k, v, g1, beta, o, gated, cs[3], S;
};

// the engine: kda_layer_launch on the whole sequence, or as two calls split at `split`, the state carried
Out run_engine(const Layer& L, int split) {
    const int d = L.d(), T = L.T, H = L.H;
    const float *xq = dev(L.xq), *xk = dev(L.xk), *xv = dev(L.xv), *fp = dev(L.f_pre), *bp = dev(L.beta_pre), *g2 = dev(L.g2);
    const float *cw[3] = {dev(L.conv[0]), dev(L.conv[1]), dev(L.conv[2])};
    const float *dtb = dev(L.dt_bias), *sa = dev(L.ssm_a), *nw = dev(L.norm_w);
    const size_t nh = (size_t) d * (L.dc - 1), nS = (size_t) H * L.dk * L.dk, nT = (size_t) T * d;
    float* cs[3] = {dzeros(nh), dzeros(nh), dzeros(nh)};
    float* S = dzeros(nS);
    float *q = dzeros(nT), *k = dzeros(nT), *v = dzeros(nT), *g1 = dzeros(nT), *beta = dzeros((size_t) T * H), *o = dzeros(nT),
          *gated = dzeros(nT);
    auto call = [&](int t0, int n) {
        strata::k31::KdaLayerArgs a;
        a.n_tokens = n; a.n_head = H; a.head_dim = L.dk; a.d_conv = L.dc;
        a.lower_bound = L.lb; a.eps_l2 = L.eps_l2; a.eps_rms = L.eps_rms;
        const size_t o1 = (size_t) t0 * d, ob = (size_t) t0 * H;
        a.xq = xq + o1; a.xk = xk + o1; a.xv = xv + o1; a.f_pre = fp + o1; a.beta_pre = bp + ob; a.g2 = g2 + o1;
        for (int s = 0; s < 3; ++s) { a.conv_w[s] = cw[s]; a.conv_state[s] = cs[s]; }
        a.dt_bias = dtb; a.ssm_a = sa; a.norm_w = nw; a.ssm_state = S;
        a.q = q + o1; a.k = k + o1; a.v = v + o1; a.g1 = g1 + o1; a.beta = beta + ob; a.o = o + o1; a.gated = gated + o1;
        if (!strata::k31::kda_layer_launch(a, nullptr)) { std::fprintf(stderr, "kda_layer_launch refused\n"); std::exit(2); }
    };
    if (split > 0 && split < T) { call(0, split); call(split, T - split); }
    else call(0, T);
    Out r;
    r.q = host(q, nT); r.k = host(k, nT); r.v = host(v, nT); r.g1 = host(g1, nT); r.beta = host(beta, (size_t) T * H);
    r.o = host(o, nT); r.gated = host(gated, nT);
    for (int s = 0; s < 3; ++s) r.cs[s] = host(cs[s], nh);
    r.S = host(S, nS);
    free_all();
    return r;
}

Vec run_wo(cublasHandle_t hb, const Vec& gated, const Layer& L) {
    const float *dg = dev(gated), *dw = dev(L.wo);
    float* dout = dzeros((size_t) L.T * L.n_embd);
    const float one = 1.0f, zero = 0.0f;
    blas_ok(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, L.n_embd, L.T, L.d(), &one, dw, L.n_embd, dg, L.d(), &zero, dout, L.n_embd), "wo");
    Vec out = host(dout, (size_t) L.T * L.n_embd);
    free_all();
    return out;
}
Vec host_wo(const Vec& gated, const Layer& L) {   // float64 attn_output for the mutated references
    Vec out((size_t) L.T * L.n_embd);
    std::vector<double> acc(L.n_embd);
    for (int t = 0; t < L.T; ++t) {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int c = 0; c < L.d(); ++c) {
            const double g = gated[(size_t) t * L.d() + c];
            const float* w = &L.wo[(size_t) c * L.n_embd];
            for (int n = 0; n < L.n_embd; ++n) acc[n] += (double) w[n] * g;
        }
        for (int n = 0; n < L.n_embd; ++n) out[(size_t) t * L.n_embd + n] = (float) acc[n];
    }
    return out;
}

enum Mut {
    M_NONE, M_LB1, M_SOFTPLUS, M_NO_DT, M_ALOG_TAIL, M_ALOG_SHIFT, M_CONV_REV, M_NO_L2, M_L2_MEAN, M_L2_EPS_RMS, M_NO_DELTA,
    M_NO_BETA_SIG, M_NO_SCALE, M_GATE_SILU, M_STATE_RESET, M_CONV_RESET, M_COUNT
};
const char* kMutName[M_COUNT] = {"none", "gate lower bound -1", "qwen softplus gate", "dt_bias dropped",
                                 "A_log zero tail (ssm_a -1)", "A_log heads shifted by one", "conv taps reversed", "no l2 norm",
                                 "l2 as rms (mean)", "rms eps as the l2 eps", "no delta correction", "beta without sigmoid",
                                 "no 1/sqrt(dk)", "SiLU output gate", "state reset between calls", "conv history reset between calls"};

double sig64(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// float64 reference of the layer (the oracle's arithmetic), with one wrong reading `m`; `split` is the call
// boundary the two reset mutations act on
Out ref(const Layer& L, Mut m, int split) {
    const int T = L.T, H = L.H, dk = L.dk, dc = L.dc, d = L.d();
    Out r;
    const Vec* x[3] = {&L.xq, &L.xk, &L.xv};
    Vec* cv[3] = {&r.q, &r.k, &r.v};
    for (int s = 0; s < 3; ++s) {
        cv[s]->resize((size_t) T * d);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < d; ++c) {
                double acc = 0;
                for (int tap = 0; tap < dc; ++tap) {
                    const int src = t - (dc - 1) + tap;
                    if (src < 0 || (m == M_CONV_RESET && t >= split && src < split)) continue;
                    const int wt = m == M_CONV_REV ? dc - 1 - tap : tap;
                    acc += (double) L.conv[s][(size_t) c * dc + wt] * (double) (*x[s])[(size_t) src * d + c];
                }
                const float y = (float) acc;
                (*cv[s])[(size_t) t * d + c] = y * (float) sig64(y);
            }
        r.cs[s].resize((size_t) d * (dc - 1));
        for (int c = 0; c < d; ++c)
            for (int j = 0; j < dc - 1; ++j) {
                const int src = T - (dc - 1) + j;
                r.cs[s][(size_t) c * (dc - 1) + j] = src >= 0 ? (*x[s])[(size_t) src * d + c] : 0.0f;
            }
    }
    if (m != M_NO_L2)
        for (Vec* qk : {&r.q, &r.k})
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < H; ++h) {
                    float* p = &(*qk)[(size_t) t * d + (size_t) h * dk];
                    double ss = 0;
                    for (int e = 0; e < dk; ++e) ss += (double) p[e] * p[e];
                    if (m == M_L2_MEAN) ss /= dk;
                    const double den = std::sqrt(ss + (m == M_L2_EPS_RMS ? (double) L.eps_rms : (double) L.eps_l2));
                    for (int e = 0; e < dk; ++e) p[e] = (float) ((double) p[e] / den);
                }
    r.g1.resize((size_t) T * d);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < d; ++c) {
            const int h = c / dk;
            const float pre = L.f_pre[(size_t) t * d + c] + (m == M_NO_DT ? 0.0f : L.dt_bias[c]);
            const int ha = m == M_ALOG_SHIFT ? (h + 1) % H : h;
            const double a = m == M_ALOG_TAIL ? -1.0 : (double) L.ssm_a[ha];
            float g;
            if (m == M_SOFTPLUS) {
                const double sp = pre > 20.0f ? (double) pre : std::log1p(std::exp((double) pre));
                g = (float) (sp * a);
            } else {
                const float arg = (float) ((double) pre * -a);
                g = (float) ((m == M_LB1 ? -1.0 : (double) L.lb) * (double) (float) sig64(arg));
            }
            r.g1[(size_t) t * d + c] = g;
        }
    r.beta.resize((size_t) T * H);
    for (size_t i = 0; i < r.beta.size(); ++i) r.beta[i] = m == M_NO_BETA_SIG ? L.beta_pre[i] : (float) sig64(L.beta_pre[i]);
    const double scale = m == M_NO_SCALE ? 1.0 : 1.0 / std::sqrt((double) dk);
    r.o.resize((size_t) T * d);
    r.S.resize((size_t) H * dk * dk);
    std::vector<double> S((size_t) dk * dk), u(dk), del(dk);
    for (int h = 0; h < H; ++h) {
        std::fill(S.begin(), S.end(), 0.0);
        for (int t = 0; t < T; ++t) {
            if (m == M_STATE_RESET && t == split) std::fill(S.begin(), S.end(), 0.0);
            const size_t b = (size_t) t * d + (size_t) h * dk;
            const double bt = r.beta[(size_t) t * H + h];
            for (int i = 0; i < dk; ++i) {
                const double dec = std::exp((double) r.g1[b + i]);
                for (int j = 0; j < dk; ++j) S[(size_t) i * dk + j] *= dec;
            }
            for (int j = 0; j < dk; ++j) {
                double acc = 0;
                for (int i = 0; i < dk; ++i) acc += S[(size_t) i * dk + j] * (double) r.k[b + i];
                u[j] = acc;
            }
            for (int j = 0; j < dk; ++j) del[j] = ((double) r.v[b + j] - (m == M_NO_DELTA ? 0.0 : u[j])) * bt;
            for (int i = 0; i < dk; ++i)
                for (int j = 0; j < dk; ++j) S[(size_t) i * dk + j] += (double) r.k[b + i] * del[j];
            for (int j = 0; j < dk; ++j) {
                double acc = 0;
                for (int i = 0; i < dk; ++i) acc += S[(size_t) i * dk + j] * (double) r.q[b + i];
                r.o[b + j] = (float) (acc * scale);
            }
        }
        for (size_t i = 0; i < S.size(); ++i) r.S[(size_t) h * dk * dk + i] = (float) S[i];
    }
    r.gated.resize((size_t) T * d);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            const size_t b = (size_t) t * d + (size_t) h * dk;
            double ms = 0;
            for (int e = 0; e < dk; ++e) ms += (double) r.o[b + e] * r.o[b + e];
            const double den = std::sqrt(ms / dk + (double) L.eps_rms);
            for (int e = 0; e < dk; ++e) {
                const float y = (float) ((double) r.o[b + e] / den);
                const float normed = y * L.norm_w[e];
                const float gv = L.g2[b + e];
                const float sg = m == M_GATE_SILU ? gv * (float) sig64(gv) : (float) sig64(gv);
                r.gated[b + e] = normed * sg;
            }
        }
    return r;
}

uint64_t g_s = 0x9E3779B97F4A7C15ull;
double urand() { g_s = g_s * 6364136223846793005ull + 1442695040888963407ull; return (double) (g_s >> 11) / 9007199254740992.0; }
double nrand() { const double u = std::max(urand(), 1e-300), v = urand(); return std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v); }

// a synthetic layer: `slow` puts the gate deep in the sigmoid's tail (decay ~0.999, long memory); `tiny_head0`
// shrinks head 0's q, k inputs until |q|^2 is ~1e-7 and the l2 eps decides the norm
Layer synth(int T, int H, int dk, bool slow, bool tiny_head0) {
    Layer L;
    L.T = T; L.H = H; L.dk = dk; L.dc = strata::k31::kKdaConv; L.n_embd = 64;
    const int d = L.d();
    auto fill = [&](Vec& v, size_t n, double s, double mu = 0) { v.resize(n); for (auto& x : v) x = (float) (mu + s * nrand()); };
    fill(L.xq, (size_t) T * d, 0.6); fill(L.xk, (size_t) T * d, 0.6); fill(L.xv, (size_t) T * d, 0.6);
    for (int s = 0; s < 3; ++s) fill(L.conv[s], (size_t) d * L.dc, 0.5);
    fill(L.f_pre, (size_t) T * d, slow ? 0.3 : 1.0, slow ? -3.0 : 0.0);
    fill(L.dt_bias, d, slow ? 0.2 : 0.5);
    L.ssm_a.resize(H);
    for (auto& a : L.ssm_a) a = (float) -std::exp(slow ? 1.0 + 0.1 * nrand() : 0.5 * nrand());
    fill(L.beta_pre, (size_t) T * H, 1.5);
    fill(L.g2, (size_t) T * d, 2.0);
    fill(L.norm_w, dk, 0.1, 1.0);
    fill(L.wo, (size_t) d * L.n_embd, 1.0 / std::sqrt((double) d));
    if (tiny_head0)
        for (Vec* x : {&L.xq, &L.xk})
            for (int t = 0; t < T; ++t)
                for (int e = 0; e < dk; ++e) (*x)[(size_t) t * d + e] *= 1e-4f;
    return L;
}

struct Errs {
    double q = 0, k = 0, v = 0, g1 = 0, beta_ulp = 0, o = 0, S = 0, gated = 0, out = 0;
    void max_with(const Errs& e) {
        q = std::max(q, e.q); k = std::max(k, e.k); v = std::max(v, e.v); g1 = std::max(g1, e.g1); beta_ulp = std::max(beta_ulp, e.beta_ulp);
        o = std::max(o, e.o); S = std::max(S, e.S); gated = std::max(gated, e.gated); out = std::max(out, e.out);
    }
    bool ok() const {
        return q <= kBoundQK && k <= kBoundQK && v <= kBoundConv && g1 <= kBoundG1 && beta_ulp <= kBoundBetaUlp && o <= kBoundScan &&
               S <= kBoundScan && gated <= kBoundOut && out <= kBoundOut;
    }
    void print(const char* tag) const {
        std::printf("    %s: q %.2e k %.2e v %.2e g1 %.2e beta %.1f ULP  scan %.2e state %.2e  gated %.2e out %.2e\n", tag, q, k, v, g1,
                    beta_ulp, o, S, gated, out);
    }
};
Errs compare(const Out& e, const Out& r, const Vec& eout, const Vec& rout) {
    Errs x;
    x.q = rel_err(e.q, r.q); x.k = rel_err(e.k, r.k); x.v = rel_err(e.v, r.v); x.g1 = rel_err(e.g1, r.g1);
    x.beta_ulp = max_ulp(e.beta, r.beta); x.o = rel_err(e.o, r.o); x.S = rel_err(e.S, r.S); x.gated = rel_err(e.gated, r.gated);
    x.out = rel_err(eout, rout);
    return x;
}

// the conv history after the call must be the last d_conv-1 inputs exactly; two calls must equal one bit for bit
int carry_checks(const Layer& L, const Out& one, int split) {
    int fails = 0;
    const Out ref0 = ref(L, M_NONE, 0);
    bool hist = true;
    for (int s = 0; s < 3; ++s) hist = hist && same_bits(one.cs[s], ref0.cs[s]);
    const Out two = run_engine(L, split);
    bool same = same_bits(one.o, two.o) && same_bits(one.gated, two.gated) && same_bits(one.S, two.S) && same_bits(one.q, two.q) &&
                same_bits(one.g1, two.g1);
    for (int s = 0; s < 3; ++s) same = same && same_bits(one.cs[s], two.cs[s]);
    std::printf("    conv history = the last %d inputs: %s;  calls [0,%d) + [%d,%d) == one call, bit for bit: %s\n", L.dc - 1,
                hist ? "ok" : "FAIL", split, split, L.T, same ? "ok" : "FAIL");
    fails += !hist + !same;
    return fails;
}

int selftest(cublasHandle_t hb) {
    int fails = 0;
    std::printf("k31_kda_test --selftest\n");
    {   // K3.1's shape, a head with tiny q/k
        const Layer L = synth(40, strata::k31::kKdaHeads, strata::k31::kKdaHeadDim, false, true);
        std::printf("  K3.1 shape: %d heads x %d, conv %d, T %d (head 0's q, k inputs x1e-4)\n", L.H, L.dk, L.dc, L.T);
        const Out e = run_engine(L, 0), r = ref(L, M_NONE, 0);
        const Vec rout = host_wo(r.gated, L);
        const Errs x = compare(e, r, run_wo(hb, e.gated, L), rout);
        x.print("engine vs float64");
        fails += !x.ok();
        fails += carry_checks(L, e, 17);
        for (int m = M_LB1; m < M_COUNT; ++m) {
            const double eff = rel_err(host_wo(ref(L, (Mut) m, 17).gated, L), rout);
            const bool caught = eff >= kNeed;
            std::printf("    mutation %-34s effect on out %.2e (needs >= %.0e)  %s\n", kMutName[m], eff, kNeed, caught ? "caught" : "NOT CAUGHT");
            fails += !caught;
        }
    }
    {   // long memory: the float32 state's drift over 4096 tokens
        const Layer L = synth(4096, 2, strata::k31::kKdaHeadDim, true, false);
        const Out e = run_engine(L, 0), r = ref(L, M_NONE, 0);
        double gmin = 0, gmax = -1e30;
        for (float g : r.g1) { gmin = std::min(gmin, (double) g); gmax = std::max(gmax, (double) g); }
        Errs x = compare(e, r, Vec(), Vec());
        // the drift at the end: the last 256 tokens' output
        const size_t tail = (size_t) 256 * L.d();
        const Vec eo(e.o.end() - tail, e.o.end()), ro(r.o.end() - tail, r.o.end());
        const double end = rel_err(eo, ro);
        std::printf("  long memory: 2 heads x 128, T %d, g1 in [%.2e, %.2e] (decay >= %.4f)\n", L.T, gmin, gmax, std::exp(gmin));
        std::printf("    scan %.2e (last 256 tokens %.2e), state %.2e, gated %.2e (bounds %.0e / %.0e)  %s\n", x.o, end, x.S, x.gated,
                    kBoundScan, kBoundOut, (x.o <= kBoundScan && x.S <= kBoundScan && x.gated <= kBoundOut) ? "ok" : "FAIL");
        fails += !(x.o <= kBoundScan && x.S <= kBoundScan && x.gated <= kBoundOut);
    }
    {   // refusals (and a pending error is refused, not blamed)
        auto z = [] { return dzeros(256); };   // every buffer its own (the contract forbids overlaps)
        strata::k31::KdaLayerArgs a;
        a.n_tokens = 1; a.n_head = 1; a.head_dim = 8; a.d_conv = 4; a.eps_rms = 1e-5f;
        a.xq = z(); a.xk = z(); a.xv = z(); a.f_pre = z(); a.beta_pre = z(); a.g2 = z(); a.dt_bias = z(); a.ssm_a = z(); a.norm_w = z();
        for (int s = 0; s < 3; ++s) { a.conv_w[s] = z(); a.conv_state[s] = z(); }
        a.ssm_state = z(); a.q = z(); a.k = z(); a.v = z(); a.g1 = z(); a.beta = z(); a.o = z(); a.gated = z();
        int refused = 0, n = 0;
        auto expect_refuse = [&](strata::k31::KdaLayerArgs b) { ++n; refused += !strata::k31::kda_layer_launch(b, nullptr); };
        auto b = a; b.n_head = 0; expect_refuse(b);
        b = a; b.head_dim = strata::k31::kKdaMaxHeadDim + 1; expect_refuse(b);
        b = a; b.d_conv = 1; expect_refuse(b);
        b = a; b.d_conv = strata::k31::kKdaMaxConv + 1; expect_refuse(b);
        b = a; b.lower_bound = 0.0f; expect_refuse(b);
        b = a; b.eps_l2 = 0.0f; expect_refuse(b);
        b = a; b.n_tokens = -1; expect_refuse(b);
        b = a; b.gated = nullptr; expect_refuse(b);
        b = a; b.conv_state[2] = nullptr; expect_refuse(b);
        void* huge = nullptr;
        const bool pend = cudaMalloc(&huge, (size_t) 1 << 50) != cudaSuccess;   // leaves a (non-sticky) error pending
        if (pend) expect_refuse(a);
        (void) cudaGetLastError();
        const bool fine = strata::k31::kda_layer_launch(a, nullptr) && cudaDeviceSynchronize() == cudaSuccess;
        b = a; b.n_tokens = 0;
        const bool noop = strata::k31::kda_layer_launch(b, nullptr);
        free_all();
        std::printf("  refusals: %d/%d (0 heads, head_dim 257, d_conv 1 and 18, lb 0, eps_l2 0, T -1, null out, null state, pending error);"
                    " a valid call %s; T 0 %s\n", refused, n, fine ? "runs" : "FAILS", noop ? "is a no-op" : "FAILS");
        fails += (refused != n) + !fine + !noop;
    }
    return fails;
}

bool read_raw(const std::string& path, Vec& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize) (out.size() * 4));
    return (size_t) f.gcount() == out.size() * 4 && f.peek() == std::char_traits<char>::eof();
}

int dumps(const std::string& dir, cublasHandle_t hb) {
    std::stringstream jss;
    jss << std::ifstream(dir + "/kda.json").rdbuf();
    const std::string js = jss.str();
    if (js.empty()) { std::printf("%s/kda.json: missing (python tools/kimi_k31_kda_dump.py --out %s)  FAIL\n", dir.c_str(), dir.c_str()); return 1; }
    auto num = [&](const char* key) {
        const size_t p = js.find(std::string("\"") + key + "\"");
        if (p == std::string::npos) { std::fprintf(stderr, "kda.json: no %s\n", key); std::exit(2); }
        return std::strtod(js.c_str() + js.find(':', p) + 1, nullptr);
    };
    Layer base;
    base.H = (int) num("n_head"); base.dk = (int) num("head_dim"); base.dc = (int) num("d_conv"); base.T = (int) num("T");
    base.n_embd = (int) num("n_embd"); base.lb = (float) num("lower_bound"); base.eps_l2 = (float) num("eps_l2"); base.eps_rms = (float) num("eps_rms");
    std::vector<int> layers;
    {
        const size_t a = js.find('[', js.find("\"layers\"")), b = js.find(']', a);
        std::stringstream ls(js.substr(a + 1, b - a - 1));
        std::string tok;
        while (std::getline(ls, tok, ',')) layers.push_back(std::atoi(tok.c_str()));
    }
    const int T = base.T, d = base.d(), split = T / 2 - 3;
    std::printf("k31_kda_test --dumps %s: layers %zu, %d heads x %d, conv %d, T %d, n_embd %d, lb %g, eps l2 %g rms %g\n", dir.c_str(),
                layers.size(), base.H, base.dk, base.dc, T, base.n_embd, base.lb, base.eps_l2, base.eps_rms);
    int fails = 0;
    Errs worst, worst_ref;
    double worst_conv = 0;
    std::vector<double> eff(M_COUNT, 1e300);
    for (int il : layers) {
        Layer L = base;
        const std::string p = dir + "/kda." + std::to_string(il) + ".";
        const size_t nT = (size_t) T * d, nb = (size_t) T * L.H;
        auto sized = [](size_t n) { return Vec(n); };
        L.xq = sized(nT); L.xk = sized(nT); L.xv = sized(nT); L.f_pre = sized(nT); L.g2 = sized(nT); L.beta_pre = sized(nb);
        for (int s = 0; s < 3; ++s) L.conv[s] = sized((size_t) d * L.dc);
        L.dt_bias = sized(d); L.ssm_a = sized(L.H); L.norm_w = sized(L.dk); L.wo = sized((size_t) d * L.n_embd);
        Out o;
        o.q = sized(nT); o.k = sized(nT); o.v = sized(nT); o.g1 = sized(nT); o.beta = sized(nb); o.o = sized(nT); o.gated = sized(nT);
        o.S = sized((size_t) L.H * L.dk * L.dk);
        Vec qc = sized(nT), kc = sized(nT), rout = sized((size_t) T * L.n_embd);
        bool ok = read_raw(p + "xq.f32", L.xq) && read_raw(p + "xk.f32", L.xk) && read_raw(p + "xv.f32", L.xv) &&
                  read_raw(p + "conv_q.f32", L.conv[0]) && read_raw(p + "conv_k.f32", L.conv[1]) && read_raw(p + "conv_v.f32", L.conv[2]) &&
                  read_raw(p + "f_pre.f32", L.f_pre) && read_raw(p + "dt_bias.f32", L.dt_bias) && read_raw(p + "ssm_a.f32", L.ssm_a) &&
                  read_raw(p + "beta_pre.f32", L.beta_pre) && read_raw(p + "g2.f32", L.g2) && read_raw(p + "norm_w.f32", L.norm_w) &&
                  read_raw(p + "wo.f32", L.wo) && read_raw(p + "q_conv.f32", qc) && read_raw(p + "k_conv.f32", kc) &&
                  read_raw(p + "v_conv.f32", o.v) && read_raw(p + "g1.f32", o.g1) && read_raw(p + "beta.f32", o.beta) &&
                  read_raw(p + "q_l2.f32", o.q) && read_raw(p + "k_l2.f32", o.k) && read_raw(p + "scan_out.f32", o.o) &&
                  read_raw(p + "gated.f32", o.gated) && read_raw(p + "out.f32", rout) && read_raw(p + "state.f32", o.S);
        if (!ok) { std::printf("  layer %d: unreadable or the wrong size  FAIL\n", il); ++fails; continue; }
        // the conv seams straight from glm_kda_conv (the adapter l2-normalises q and k in place)
        double ce = 0;
        for (int s = 0; s < 2; ++s) {
            const float* dx = dev(s ? L.xk : L.xq);
            const float* dw = dev(L.conv[s]);
            float *dst = dzeros((size_t) d * (L.dc - 1)), *dout = dzeros(nT);
            strata::kernels::glm_kda_conv(dx, dw, dst, d, L.dc, T, dout, nullptr);
            ce = std::max(ce, rel_err(host(dout, nT), s ? kc : qc));
            free_all();
        }
        worst_conv = std::max(worst_conv, ce);
        const Out e = run_engine(L, 0);
        const Errs x = compare(e, o, run_wo(hb, e.gated, L), rout);
        const Out r = ref(L, M_NONE, split);
        const Errs xr = compare(r, o, host_wo(r.gated, L), rout);
        worst.max_with(x);
        worst_ref.max_with(xr);
        const bool refok = xr.q <= kBoundRef && xr.k <= kBoundRef && xr.v <= kBoundRef && xr.g1 <= kBoundRef && xr.beta_ulp <= kBoundBetaUlp &&
                           xr.o <= kBoundRef && xr.S <= kBoundRef && xr.gated <= kBoundRef && xr.out <= kBoundOut;
        const bool lok = x.ok() && ce <= kBoundConv && refok;
        std::printf("  layer %d: conv %.2e  q %.2e  g1 %.2e  beta %.1f ULP  scan %.2e  state %.2e  gated %.2e  out %.2e  (float64 ref: seams %.2e, out %.2e)  %s\n",
                    il, ce, x.q, x.g1, x.beta_ulp, x.o, x.S, x.gated, x.out, std::max({xr.q, xr.k, xr.v, xr.g1, xr.o, xr.S, xr.gated}), xr.out,
                    lok ? "ok" : "FAIL");
        fails += !lok;
        fails += carry_checks(L, e, split);
        for (int m = M_LB1; m < M_COUNT; ++m) eff[m] = std::min(eff[m], rel_err(host_wo(ref(L, (Mut) m, split).gated, L), rout));
    }
    std::printf("  worst over layers (engine vs the oracle):  conv %.2e (bound %.0e)\n", worst_conv, kBoundConv);
    worst.print("engine   ");
    worst_ref.print("float64  ");
    std::printf("    bounds: q/k %.0e, v %.0e, g1 %.0e, beta %.0f ULP, scan/state %.0e, gated/out %.0e; the float64 reference %.0e (out %.0e)\n",
                kBoundQK, kBoundConv, kBoundG1, kBoundBetaUlp, kBoundScan, kBoundOut, kBoundRef, kBoundOut);
    for (int m = M_LB1; m < M_COUNT; ++m) {
        // the l2 eps cannot show at the fixture's |q|^2 (it is the selftest's tiny head that catches it)
        const bool fixture_blind = m == M_L2_EPS_RMS;
        const bool caught = eff[m] >= kNeed && eff[m] < 1e299;
        std::printf("  mutation %-34s smallest effect on kda_out %.2e (needs >= %.0e)  %s\n", kMutName[m], eff[m], kNeed,
                    caught ? "caught" : fixture_blind ? "below the fixture's reach (the selftest catches it)" : "NOT CAUGHT");
        fails += !caught && !fixture_blind;
    }
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    cublasHandle_t hb;
    blas_ok(cublasCreate(&hb), "cublasCreate");
    blas_ok(cublasSetMathMode(hb, CUBLAS_DEFAULT_MATH), "math mode");
    int fails = 0;
    bool any = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") { fails += selftest(hb); any = true; }
        else if (a == "--dumps" && i + 1 < argc) { fails += dumps(argv[++i], hb); any = true; }
        else { std::fprintf(stderr, "usage: %s [--selftest] [--dumps DIR]\n", argv[0]); return 2; }
    }
    if (!any) fails += selftest(hb);
    cublasDestroy(hb);
    std::printf("k31_kda_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
