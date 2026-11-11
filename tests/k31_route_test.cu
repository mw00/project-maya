// tests/k31_route_test.cu - the K3.1 router (src/k31/k31_route.hpp) held to ref/kimi_k31.py's routing and to
// tools/kimi_k31_golden.py's vectors, plus the wrong readings, each of which must fail the gate.
//
//   k31_route_test --selftest       built-in cases at E = 896, k = 16 (and the fixture's 24 / 10, E = 1024, k = 1)
//   k31_route_test --golden DIR     the golden vectors (python tools/kimi_k31_golden.py --out DIR)
//
// The gate (the golden's check_route): per token the selected set equals the reference's, or - only when the
// reference's cut margin is > 0 - both are k distinct ids and every id in their symmetric difference scores within 4
// float32 ULPs of the cut (a rounding near-tie; bitwise ties must match, since equal inputs give equal scores);
// weights, matched by expert id, within the budget (8 ULPs, measured from a plain float32 kernel) of the float64
// weights of the reference's float32 probs; the picks in rank order (descending score, ties to the lower id, NaN
// after -inf), allowing a swap only between neighbours within 4 ULPs; every id in range; the device probs equal the
// float32 of the float64 sigmoid bit for bit.  The kernel computes its sigmoid like the reference, so in practice
// the ids match in order and the weights bit for bit; both are reported.
// Wrong readings: softmax, weights from the biased score, no renormalisation, a 2^-14 clamp, higher-id ties, picks by
// the unbiased score, 8 picks, the scale dropped, renormalisation over all experts, no eps, max(sum, eps); and the
// device kernel run with k = 8 (the "experts truncated to 8" mutation).  The denormal case (every sigmoid below
// 2^-126) is what fails a flush-to-zero (--use_fast_math) build; eps_dom (sum(p) ~ 1e-21) is what pins the + 1e-20.
#include "../src/k31/k31_route.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kMinNormal = 1.17549435082228751e-38;
double g_budget = 8.0;

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

enum Read { kRight, kSoftmax, kBiasedW, kNoRenorm, kClamp, kHighTies, kUnbiasedPicks, kK8, kNoScale, kRenormAll, kEpsZero,
            kEpsClamp, kNRead };
const char* const kReadName[kNRead] = {"right", "softmax", "biased_weights", "no_renorm", "clamp_2m14", "high_id_ties",
                                       "unbiased_picks", "k8", "no_scale", "renorm_all", "no_eps", "max_sum_eps"};

// the kernel's total order on scores: equal floats equal (+0 == -0), NaN below -inf
uint32_t order_key(float x) {
    if (x != x) return 0u;
    uint32_t b = 0;
    if (x != 0.0f) std::memcpy(&b, &x, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// the float32 spacing at |x| (for x = 0 the smallest denormal)
double f32_spacing(float x) {
    const float a = std::fabs(x);
    return (double) std::nextafter(a, std::numeric_limits<float>::infinity()) - (double) a;
}

struct Case {
    std::string name;
    int T = 0, E = 0, k = 16;
    float scale = 1.0f;
    bool norm = true;
    double eps = 1e-20;
    std::vector<float> logits, bias;   // [T][E], [E]; an empty bias is passed as null (sel = p)
};

struct Ref {
    int k = 0;
    std::vector<int> ids;            // [T][k]
    std::vector<float> w;            // [T][k] float32 (rounded once)
    std::vector<double> w64;         // [T][k]
    std::vector<float> biased;       // [T][E]
    std::vector<double> margin;      // [T]
};

// ref/kimi_k31.py moe_ffn routing, and the wrong readings
Ref reference(const Case& c, Read rd = kRight) {
    Ref r;
    const int E = c.E, T = c.T, k = rd == kK8 ? 8 : c.k;
    r.k = k;
    r.ids.assign((size_t) T * k, 0);
    r.w.assign((size_t) T * k, 0);
    r.w64.assign((size_t) T * k, 0);
    r.biased.assign((size_t) T * E, 0);
    r.margin.assign(T, 0);
    std::vector<float> p(E), sel(E);
    for (int t = 0; t < T; ++t) {
        const float* l = c.logits.data() + (size_t) t * E;
        if (rd == kSoftmax) {
            double mx = -1e300, s = 0;
            for (int e = 0; e < E; ++e) mx = std::max(mx, (double) l[e]);
            for (int e = 0; e < E; ++e) s += std::exp((double) l[e] - mx);
            for (int e = 0; e < E; ++e) p[e] = std::isfinite(mx) ? (float) (std::exp((double) l[e] - mx) / s) : 0.0f;
        } else {
            for (int e = 0; e < E; ++e) p[e] = (float) (1.0 / (1.0 + std::exp(-(double) l[e])));
        }
        for (int e = 0; e < E; ++e) sel[e] = c.bias.empty() ? p[e] : p[e] + c.bias[e];
        std::copy(sel.begin(), sel.end(), r.biased.begin() + (size_t) t * E);
        const std::vector<float>& key = rd == kUnbiasedPicks ? p : sel;
        std::vector<int> ord(E);
        std::iota(ord.begin(), ord.end(), 0);
        auto kd = [&](int a) { return order_key(key[a]); };
        if (rd == kHighTies) std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return kd(a) > kd(b) || (kd(a) == kd(b) && a > b); });
        else std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return kd(a) > kd(b); });
        if (E > c.k) {   // the true order's cut margin (no near-tie allowance unless finite and > 0)
            std::vector<int> o2(E);
            std::iota(o2.begin(), o2.end(), 0);
            std::stable_sort(o2.begin(), o2.end(), [&](int a, int b) { return order_key(sel[a]) > order_key(sel[b]); });
            const double m = (double) sel[o2[c.k - 1]] - (double) sel[o2[c.k]];
            r.margin[t] = std::isfinite(m) ? m : 0.0;
        } else {
            r.margin[t] = 1.0;
        }
        double sum = 0, all = 0;
        for (int i = 0; i < k; ++i) sum += (double) (rd == kBiasedW ? sel[ord[i]] : p[ord[i]]);
        for (int e = 0; e < E; ++e) all += (double) p[e];
        for (int i = 0; i < k; ++i) {
            const double x = (double) (rd == kBiasedW ? sel[ord[i]] : p[ord[i]]);
            double v;
            switch (rd) {
                case kNoRenorm: v = x; break;
                case kClamp: v = x / std::max(sum, 6.103515625e-5); break;
                case kRenormAll: v = x / (all + c.eps); break;
                case kEpsZero: v = x / sum; break;
                case kEpsClamp: v = x / std::max(sum, c.eps); break;
                default: v = c.norm ? x / (sum + c.eps) : x;
            }
            v *= rd == kNoScale ? 1.0 : (double) c.scale;
            r.ids[(size_t) t * k + i] = ord[i];
            r.w64[(size_t) t * k + i] = v;
            r.w[(size_t) t * k + i] = (float) v;
        }
    }
    return r;
}

double ulp_err(double got, double ref) {
    const double ar = std::fabs(ref);
    double unit = kMinNormal;
    if (ar >= kMinNormal) {
        const float f = (float) ar;
        unit = std::max((double) std::nextafter(f, std::numeric_limits<float>::infinity()) - (double) f, kMinNormal);
    }
    return std::fabs(got - ref) / unit;
}

double tie_tol(double score) {
    const float f = (float) std::fabs(score);
    return 4.0 * ((double) std::nextafter(f, std::numeric_limits<float>::infinity()) - (double) f);
}

struct Gate {
    long bad_sets = 0, bad_order = 0, near = 0, exact_order = 0, bit_w = 0, n_w = 0;
    double worst = 0;
    bool ok() const { return bad_sets == 0 && bad_order == 0 && worst <= g_budget; }
};

Gate check(const Case& c, const Ref& gold, const std::vector<int>& ids, const std::vector<float>& w, int kk) {
    Gate g;
    const int k = gold.k, E = c.E;
    for (int t = 0; t < c.T; ++t) {
        std::vector<int> a(gold.ids.begin() + (size_t) t * k, gold.ids.begin() + (size_t) (t + 1) * k);
        std::vector<int> b(ids.begin() + (size_t) t * kk, ids.begin() + (size_t) (t + 1) * kk);
        if ((int) b.size() != k || std::any_of(b.begin(), b.end(), [&](int e) { return e < 0 || e >= E; })) { ++g.bad_sets; continue; }
        std::set<int> sa(a.begin(), a.end()), sb(b.begin(), b.end());
        if ((int) sb.size() != k) { ++g.bad_sets; continue; }
        const float* col = gold.biased.data() + (size_t) t * E;
        if (sa != sb) {
            bool ok = gold.margin[t] > 0;
            if (ok) {
                std::vector<int> o2(E);
                std::iota(o2.begin(), o2.end(), 0);
                std::stable_sort(o2.begin(), o2.end(), [&](int x, int y) { return order_key(col[x]) > order_key(col[y]); });
                const double cut = col[o2[k - 1]], tol = tie_tol(cut);
                for (int e : sa) if (!sb.count(e) && std::fabs(col[e] - cut) > tol) ok = false;
                for (int e : sb) if (!sa.count(e) && std::fabs(col[e] - cut) > tol) ok = false;
            }
            if (!ok) { ++g.bad_sets; continue; }
            ++g.near;
        }
        // rank order: each neighbour pair descends (ties to the lower id), or is a rounding near-tie
        for (int i = 0; i + 1 < k; ++i) {
            const int x = b[i], y = b[i + 1];
            const uint32_t kx = order_key(col[x]), ky = order_key(col[y]);
            const bool near = std::isfinite(col[x]) && std::isfinite(col[y]) && kx != ky &&
                              std::fabs((double) col[x] - (double) col[y]) <= tie_tol(col[x]);
            if (!(kx > ky || (kx == ky && x < y) || near)) { ++g.bad_order; break; }
        }
        if (a == b) ++g.exact_order;
        for (int i = 0; i < k; ++i) {
            const auto it = std::find(a.begin(), a.end(), b[i]);
            if (it == a.end()) continue;
            const size_t gi = (size_t) t * k + (it - a.begin());
            g.worst = std::max(g.worst, ulp_err(w[(size_t) t * kk + i], gold.w64[gi]));
            g.bit_w += std::memcmp(&w[(size_t) t * kk + i], &gold.w[gi], 4) == 0;
            ++g.n_w;
        }
    }
    return g;
}

bool run_device(const Case& c, int k, std::vector<int>& ids, std::vector<float>& w, std::vector<float>* probs = nullptr) {
    float *dl = nullptr, *db = nullptr, *dw = nullptr, *dp = nullptr;
    int32_t* di = nullptr;
    cuda_ok(cudaMalloc(&dl, c.logits.size() * 4), "malloc");
    if (!c.bias.empty()) cuda_ok(cudaMalloc(&db, c.bias.size() * 4), "malloc");
    cuda_ok(cudaMalloc(&di, (size_t) c.T * k * 4), "malloc");
    cuda_ok(cudaMalloc(&dw, (size_t) c.T * k * 4), "malloc");
    if (probs) cuda_ok(cudaMalloc(&dp, (size_t) c.T * c.E * 4), "malloc");
    cuda_ok(cudaMemcpy(dl, c.logits.data(), c.logits.size() * 4, cudaMemcpyHostToDevice), "h2d");
    if (db) cuda_ok(cudaMemcpy(db, c.bias.data(), c.bias.size() * 4, cudaMemcpyHostToDevice), "h2d");
    strata::k31::RouteArgs a;
    a.logits = dl; a.bias = db; a.n_tokens = c.T; a.n_expert = c.E; a.k = k; a.scale = c.scale; a.norm = c.norm; a.eps = c.eps;
    a.ids = di; a.weights = dw; a.probs = dp;
    const bool ok = strata::k31::route_launch(a, nullptr);
    if (ok) {
        cuda_ok(cudaDeviceSynchronize(), "sync");
        ids.resize((size_t) c.T * k);
        w.resize((size_t) c.T * k);
        cuda_ok(cudaMemcpy(ids.data(), di, ids.size() * 4, cudaMemcpyDeviceToHost), "d2h");
        cuda_ok(cudaMemcpy(w.data(), dw, w.size() * 4, cudaMemcpyDeviceToHost), "d2h");
        if (probs) { probs->resize((size_t) c.T * c.E); cuda_ok(cudaMemcpy(probs->data(), dp, probs->size() * 4, cudaMemcpyDeviceToHost), "d2h"); }
    }
    cudaFree(dl); cudaFree(db); cudaFree(di); cudaFree(dw); cudaFree(dp);
    return ok;
}

// returns failures; `wrong` collects, per wrong reading, whether some case caught it
int run_case(const Case& c, std::vector<int>& caught) {
    int fails = 0;
    const Ref gold = reference(c);
    std::vector<int> ids;
    std::vector<float> w, probs;
    if (!run_device(c, c.k, ids, w, &probs)) { std::printf("  %-10s launch refused  FAIL\n", c.name.c_str()); return 1; }
    const Gate g = check(c, gold, ids, w, c.k);
    long pbit = 0;
    for (int t = 0; t < c.T; ++t)
        for (int e = 0; e < c.E; ++e) {
            const float want = (float) (1.0 / (1.0 + std::exp(-(double) c.logits[(size_t) t * c.E + e]))), got = probs[(size_t) t * c.E + e];
            pbit += std::memcmp(&got, &want, 4) == 0 || (got != got && want != want);
        }
    const bool pok = pbit == (long) c.T * c.E;
    std::printf("  %-10s E %4d T %3d k %2d scale %.1f%s  sets bad %ld near %ld  order bad %ld, exact %ld/%d  w worst %.2f ULP, bit-equal %ld/%ld  "
                "probs bit-equal %ld/%ld  %s\n", c.name.c_str(), c.E, c.T, c.k, c.scale, c.norm ? "" : " (no norm)", g.bad_sets, g.near,
                g.bad_order, g.exact_order, c.T, g.worst, g.bit_w, g.n_w, pbit, (long) c.T * c.E, g.ok() && pok ? "ok" : "FAIL");
    fails += !g.ok() || !pok;
    for (int rd = 1; rd < kNRead; ++rd) {
        const Ref wr = reference(c, (Read) rd);
        const Gate gw = check(c, gold, wr.ids, wr.w, wr.k);
        if (!gw.ok()) caught[rd] = 1;
    }
    if (c.k == 16) {   // the device kernel itself truncated to 8 picks
        std::vector<int> i8;
        std::vector<float> w8;
        if (run_device(c, 8, i8, w8)) {
            std::vector<int> pad((size_t) c.T * 16);
            std::vector<float> wpad((size_t) c.T * 16, 0.0f);
            for (int t = 0; t < c.T; ++t)
                for (int i = 0; i < 16; ++i) { pad[(size_t) t * 16 + i] = i8[(size_t) t * 8 + (i % 8)]; wpad[(size_t) t * 16 + i] = w8[(size_t) t * 8 + (i % 8)]; }
            if (!check(c, gold, pad, wpad, 16).ok()) caught[0] = 1;   // slot 0 = the device mutation
        }
    }
    return fails;
}

uint64_t g_s = 0x5DEECE66Dull;
double urand() { g_s = g_s * 6364136223846793005ull + 1442695040888963407ull; return (double) (g_s >> 11) / 9007199254740992.0; }
double nrand() { double u = std::max(urand(), 1e-300), v = urand(); return std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v); }

std::vector<Case> builtin_cases() {
    std::vector<Case> v;
    auto mk = [&](const char* n, int T, int E, int k, float scale, double lsd, double bsd, double lmean = 0) {
        Case c; c.name = n; c.T = T; c.E = E; c.k = k; c.scale = scale;
        c.logits.resize((size_t) T * E); c.bias.resize(E);
        for (auto& x : c.logits) x = (float) (lmean + lsd * nrand());
        for (auto& x : c.bias) x = (float) (bsd * nrand());
        return c;
    };
    v.push_back(mk("rand", 64, 896, 16, 1.0f, 2.0, 0.3));
    v.push_back(mk("decode1", 1, 896, 16, 1.0f, 2.0, 0.3));
    v.push_back(mk("window16", 16, 896, 16, 1.0f, 1.0, 0.05));
    {   // ties: a 9-way bitwise-equal block straddling the cut; a 20-way tie at the top; all 896 equal
        Case c = mk("ties", 10, 896, 16, 1.0f, 0.5, 0.0, -3.0);
        for (int t = 0; t < 8; ++t) {
            std::vector<int> perm(896);
            std::iota(perm.begin(), perm.end(), 0);
            for (int i = 895; i > 0; --i) std::swap(perm[i], perm[(int) (urand() * (i + 1))]);
            for (int i = 0; i < 12; ++i) c.logits[(size_t) t * 896 + perm[i]] = (float) (4.0 + urand());
            for (int i = 12; i < 21; ++i) c.logits[(size_t) t * 896 + perm[i]] = 1.25f;
        }
        for (int e = 0; e < 896; ++e) c.logits[(size_t) 8 * 896 + e] = 0.5f;
        for (int i = 0; i < 20; ++i) c.logits[(size_t) 8 * 896 + (i * 37) % 896] = 3.0f;
        for (int e = 0; e < 896; ++e) c.logits[(size_t) 9 * 896 + e] = 0.7f;
        v.push_back(c);
    }
    {   // bias decides the set
        Case c = mk("bias_flip", 16, 896, 16, 1.0f, 1.0, 0.0);
        for (int i = 0; i < 40; ++i) c.bias[(i * 53) % 896] = 2.0f;
        v.push_back(c);
    }
    {   // every selected sigmoid tiny: +1e-20 vs the 2^-14 clamp
        Case c = mk("tiny", 12, 896, 16, 1.0f, 0.0, 1e-12);
        for (auto& x : c.logits) x = (float) (-30.0 + 9.0 * urand());
        v.push_back(c);
    }
    {   // sigmoid == 1.0f: the bias alone orders, with ties
        Case c = mk("saturate", 8, 896, 16, 1.0f, 0.0, 0.0);
        for (auto& x : c.logits) x = (float) (18.0 + 42.0 * urand());
        for (auto& x : c.bias) x = (float) ((int) (urand() * 6) * 0.125);
        v.push_back(c);
    }
    {   // +-inf logits
        Case c = mk("inf", 6, 896, 16, 1.0f, 2.0, 0.1);
        for (int t = 0; t < 6; ++t) {
            for (int i = 0; i < 5 + 2 * t; ++i) c.logits[(size_t) t * 896 + (i * 97 + t) % 896] = std::numeric_limits<float>::infinity();
            for (int i = 0; i < 50; ++i) c.logits[(size_t) t * 896 + (i * 31 + 7 * t + 1) % 896] = -std::numeric_limits<float>::infinity();
        }
        v.push_back(c);
    }
    v.push_back(mk("scaled", 8, 896, 16, 2.5f, 2.0, 0.3));
    {   // NaN logits must not break the selection: ids stay distinct and in range (checked in run_nan)
        Case c = mk("nan", 2, 896, 16, 1.0f, 2.0, 0.3);
        for (int i = 0; i < 30; ++i) c.logits[(size_t) (i % 2) * 896 + (i * 29) % 896] = std::numeric_limits<float>::quiet_NaN();
        v.push_back(c);
    }
    {   // every sigmoid denormal (logits in [-103.9, -88]), null bias: a flush-to-zero build routes by id, weights 0
        Case c = mk("denormal", 8, 896, 16, 1.0f, 0.0, 0.0);
        for (auto& x : c.logits) x = (float) (-103.9 + 15.9 * urand());
        c.bias.clear();
        v.push_back(c);
    }
    {   // sum(p) ~ 1e-21, under the eps: the + 1e-20 decides every weight (no eps or a clamp moves them a lot)
        Case c = mk("eps_dom", 8, 896, 16, 1.0f, 0.0, 1e-24);
        for (auto& x : c.logits) x = (float) (-52.0 + 4.0 * urand());
        v.push_back(c);
    }
    {   // a non-default eps (1e-3 against sum(p) ~ 5e-3)
        Case c = mk("eps_1e-3", 8, 896, 16, 1.0f, 0.0, 0.01);
        for (auto& x : c.logits) x = (float) (-8.5 + urand());
        c.eps = 1e-3;
        v.push_back(c);
    }
    v.push_back(mk("k_eq_e", 6, 32, 32, 1.0f, 2.0, 0.3));
    {   Case c = mk("no_norm", 8, 896, 16, 2.5f, 2.0, 0.3); c.norm = false; v.push_back(c); }
    {   // NaN after -inf: 16 finite scores, 8 with a -inf bias, 8 NaN logits; k = 20 takes 4 of the -inf ones (lowest ids)
        Case c = mk("nan_order", 3, 32, 20, 1.0f, 2.0, 0.3);
        for (int t = 0; t < 3; ++t)
            for (int i = 0; i < 8; ++i) c.logits[(size_t) t * 32 + 3 + 3 * i + (t % 2)] = std::numeric_limits<float>::quiet_NaN();
        for (int i = 0; i < 8; ++i) c.bias[2 + 4 * i - (i % 2)] = -std::numeric_limits<float>::infinity();
        v.push_back(c);
    }
    v.push_back(mk("fixture", 32, 24, 10, 1.0f, 2.0, 0.3));
    v.push_back(mk("e1024", 4, 1024, 16, 1.0f, 2.0, 0.3));
    v.push_back(mk("k1", 4, 896, 1, 1.0f, 2.0, 0.3));
    return v;
}

int report_wrong(const std::vector<int>& caught) {
    int fails = 0;
    for (int rd = 1; rd < kNRead; ++rd) {
        std::printf("  wrong reading %-15s %s\n", kReadName[rd], caught[rd] ? "caught" : "NOT CAUGHT");
        fails += !caught[rd];
    }
    std::printf("  device kernel truncated to 8 picks  %s\n", caught[0] ? "caught" : "NOT CAUGHT");
    fails += !caught[0];
    return fails;
}

int run_nan(const Case& c) {
    std::vector<int> ids;
    std::vector<float> w;
    if (!run_device(c, c.k, ids, w)) { std::printf("  nan        launch refused  FAIL\n"); return 1; }
    long bad = 0;
    for (int t = 0; t < c.T; ++t) {
        std::set<int> s;
        for (int i = 0; i < c.k; ++i) {
            const int e = ids[(size_t) t * c.k + i];
            bad += e < 0 || e >= c.E || std::isnan(c.logits[(size_t) t * c.E + e]);
            s.insert(e);
        }
        bad += (int) s.size() != c.k;
    }
    std::printf("  nan        NaN logits: ids distinct, in range, none NaN: %s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}

int selftest() {
    std::printf("k31_route_test --selftest: budget %.0f ULP\n", g_budget);
    std::vector<int> caught(kNRead, 0);
    int fails = 0;
    for (const auto& c : builtin_cases()) fails += run_case(c, caught) + (c.name == "nan" ? run_nan(c) : 0);
    fails += report_wrong(caught);
    // the contract's refusals
    strata::k31::RouteArgs bad;
    int32_t* di = nullptr; float* dw = nullptr; float* dl = nullptr;
    cuda_ok(cudaMalloc(&di, 64), "m"); cuda_ok(cudaMalloc(&dw, 64), "m"); cuda_ok(cudaMalloc(&dl, 8192), "m");
    strata::k31::RouteArgs good;
    good.logits = dl; good.ids = di; good.weights = dw; good.n_tokens = 1; good.n_expert = 8; good.k = 4;
    int refused = 0, n = 0;
    auto refuse = [&](strata::k31::RouteArgs b) { ++n; refused += !strata::k31::route_launch(b, nullptr); };
    bad = good; bad.n_expert = 1025; bad.k = 16; refuse(bad);
    bad = good; bad.n_expert = 896; bad.k = 33; refuse(bad);
    bad = good; bad.k = 9; refuse(bad);
    bad = good; bad.k = 0; refuse(bad);
    bad = good; bad.n_expert = 0; bad.k = 0; refuse(bad);
    bad = good; bad.n_tokens = -1; refuse(bad);
    bad = good; bad.logits = nullptr; refuse(bad);
    bad = good; bad.ids = nullptr; refuse(bad);
    bad = good; bad.weights = nullptr; refuse(bad);
    void* huge = nullptr;
    if (cudaMalloc(&huge, (size_t) 1 << 50) != cudaSuccess) refuse(good);   // a (non-sticky) error left pending
    (void) cudaGetLastError();
    bad = good; bad.n_tokens = 0;
    const bool noop = strata::k31::route_launch(bad, nullptr);
    const bool runs = strata::k31::route_launch(good, nullptr) && cudaDeviceSynchronize() == cudaSuccess;
    cudaFree(di); cudaFree(dw); cudaFree(dl);
    std::printf("  refusals (E 1025, k 33, k > E, k 0, E 0, T -1, null logits/ids/weights, a pending error): %d/%d; T 0 %s; a valid call %s  %s\n",
                refused, n, noop ? "is a no-op" : "FAILS", runs ? "runs" : "FAILS", refused == n && noop && runs ? "ok" : "FAIL");
    fails += refused != n || !noop || !runs;
    return fails;
}

// ---- golden files
bool read_raw(const std::string& path, size_t n, void* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(static_cast<char*>(out), (std::streamsize) (n * 4));
    return (size_t) f.gcount() == n * 4;
}

// the [open, close] of the JSON object that starts at the first '{' at or after `from` (npos, npos if none)
std::pair<size_t, size_t> object_span(const std::string& js, size_t from) {
    const size_t a = from == std::string::npos ? from : js.find('{', from);
    if (a == std::string::npos) return {std::string::npos, std::string::npos};
    int depth = 0;
    bool str = false;
    for (size_t i = a; i < js.size(); ++i) {
        const char ch = js[i];
        if (str) { if (ch == '\\') ++i; else if (ch == '"') str = false; continue; }
        if (ch == '"') str = true;
        else if (ch == '{') ++depth;
        else if (ch == '}' && --depth == 0) return {a, i};
    }
    return {std::string::npos, std::string::npos};
}

// `key`'s value inside the object span, its own keys only (nested objects skipped); true/false as 1/0, a quoted
// number ("inf") parsed
double json_num(const std::string& js, std::pair<size_t, size_t> span, const std::string& key, double dflt) {
    if (span.first == std::string::npos) return dflt;
    const std::string pat = "\"" + key + "\"";
    for (size_t b = js.find(pat, span.first); b != std::string::npos && b < span.second; b = js.find(pat, b + 1)) {
        int depth = 0;
        for (size_t i = span.first + 1; i < b; ++i) depth += js[i] == '{' ? 1 : js[i] == '}' ? -1 : 0;
        if (depth != 0) continue;
        const size_t c = js.find(':', b);
        if (c == std::string::npos || c > span.second) return dflt;
        const char* s = js.c_str() + c + 1;
        while (*s == ' ' || *s == '\n') ++s;
        if (!std::strncmp(s, "true", 4)) return 1.0;
        if (!std::strncmp(s, "false", 5)) return 0.0;
        if (*s == '"') return std::strtod(s + 1, nullptr);
        return std::strtod(s, nullptr);
    }
    return dflt;
}

int golden(const std::string& dir) {
    std::ifstream mf(dir + "/manifest.txt");
    if (!mf) { std::printf("%s/manifest.txt: missing (python tools/kimi_k31_golden.py --out %s)  FAIL\n", dir.c_str(), dir.c_str()); return 1; }
    std::stringstream jss;
    jss << std::ifstream(dir + "/golden.json").rdbuf();
    const std::string js = jss.str();
    const size_t router = js.find("\"router\"");
    const auto rspan = object_span(js, router);
    g_budget = json_num(js, rspan, "budget_ulp", g_budget);
    std::printf("k31_route_test --golden %s: budget %.0f ULP\n", dir.c_str(), g_budget);
    std::vector<std::pair<std::string, std::vector<size_t>>> man;
    std::string line;
    while (std::getline(mf, line)) {
        std::istringstream ls(line);
        std::string name;
        ls >> name;
        std::vector<size_t> d;
        size_t x;
        while (ls >> x) d.push_back(x);
        man.push_back({name, d});
    }
    std::vector<int> caught(kNRead, 0);
    int fails = 0, cases = 0;
    for (const auto& [name, dims] : man) {
        if (name.rfind("route.", 0) != 0 || name.size() < 7 || name.substr(name.size() - 7) != ".logits") continue;
        const std::string cname = name.substr(6, name.size() - 13);
        Case c;
        c.name = cname;
        c.T = (int) dims[0];
        c.E = (int) dims[1];
        const size_t cpos = js.find("\"" + cname + "\"", router);
        const auto cspan = cpos != std::string::npos && cpos < rspan.second ? object_span(js, cpos) : std::make_pair(std::string::npos, std::string::npos);
        if (cspan.first == std::string::npos) { std::printf("  %s: no entry in golden.json's router  FAIL\n", cname.c_str()); ++fails; continue; }
        c.k = (int) json_num(js, cspan, "k", 16);
        c.scale = (float) json_num(js, cspan, "scale", 1.0);
        c.norm = json_num(js, cspan, "norm", 1.0) != 0.0;
        c.eps = json_num(js, cspan, "eps", 1e-20);
        c.logits.resize((size_t) c.T * c.E);
        c.bias.resize(c.E);
        std::vector<int32_t> gids((size_t) c.T * c.k);
        std::vector<float> gw((size_t) c.T * c.k), gprobs((size_t) c.T * c.E);
        const std::string p = dir + "/route." + cname + ".";
        if (!read_raw(p + "logits.f32", c.logits.size(), c.logits.data()) || !read_raw(p + "bias.f32", c.bias.size(), c.bias.data()) ||
            !read_raw(p + "ids.f32", gids.size(), gids.data()) || !read_raw(p + "w.f32", gw.size(), gw.data()) ||
            !read_raw(p + "probs.f32", gprobs.size(), gprobs.data())) {
            std::printf("  %s: unreadable  FAIL\n", cname.c_str());
            ++fails;
            continue;
        }
        // the golden file must agree with this harness's reference (the same oracle semantics, written twice)
        const Ref ref = reference(c);
        // probs: numpy's float64 exp is not glibc's (= CUDA's) and rounds one float32 logit (0x37260000) apart, so a
        // golden prob may sit 1 ULP off there: counted, and only > 1 ULP fails
        long id_diff = 0, w_diff = 0, p_diff = 0, p_far = 0;
        for (size_t i = 0; i < gids.size(); ++i) { id_diff += gids[i] != ref.ids[i]; w_diff += std::memcmp(&gw[i], &ref.w[i], 4) != 0; }
        for (int t = 0; t < c.T; ++t)
            for (int e = 0; e < c.E; ++e) {
                const float want = (float) (1.0 / (1.0 + std::exp(-(double) c.logits[(size_t) t * c.E + e]))), got = gprobs[(size_t) t * c.E + e];
                if (std::memcmp(&got, &want, 4) == 0 || (got != got && want != want)) continue;
                ++p_diff;
                p_far += !(std::fabs((double) got - (double) want) <= f32_spacing(want));
            }
        std::printf("  %-10s golden file vs this harness's reference: ids %ld, weights %ld differ; probs %ld differ (%ld by > 1 ULP)  %s\n",
                    cname.c_str(), id_diff, w_diff, p_diff, p_far, id_diff + w_diff + p_far ? "FAIL" : "ok");
        fails += (id_diff + w_diff + p_far) != 0;
        fails += run_case(c, caught);
        ++cases;
    }
    if (!cases) { std::printf("  no route.* cases in %s  FAIL\n", dir.c_str()); return 1; }
    fails += report_wrong(caught);
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    int fails = 0;
    bool any = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") { fails += selftest(); any = true; }
        else if (a == "--golden" && i + 1 < argc) { fails += golden(argv[++i]); any = true; }
        else { std::fprintf(stderr, "usage: %s [--selftest] [--golden DIR]\n", argv[0]); return 2; }
    }
    if (!any) fails += selftest();
    std::printf("k31_route_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
