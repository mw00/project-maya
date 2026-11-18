// tests/k31_index_test.cu - the K3.1 lightning indexer (src/k31/k31_index.hpp) held to ref/kimi_k31.py's indexer()
// and to a float64 reference, plus the mutations, each of which must fail the gate.
//
//   k31_index_test --selftest     K3.1's geometry (32 heads x 128, top 2048): a 2200-token prefill and a 64-token
//                                 continuation, tied keys at the cut, a row of zero weights, low-variance keys (the
//                                 LayerNorm eps shows), the refusals
//   k31_index_test --dumps DIR    the fixture's MLA layers (12 x 32, top 6) and a 2200 + 40-token call at top 2048
//                                 (python tools/kimi_k31_indexer_dump.py --out DIR)
//
// The gate, per call: the float32 keys within 1 ULP of the reference (BF16 cache rows = the float32 keys rounded);
// scores max |d| / rms over the visible keys within the bound, -inf exactly where the reference masks; every row's
// selection agrees with the reference's up to ties (ref/kimi_k31.py topk_sets_agree, tolerance 2x the call's
// measured max |score error|); the selection exactly the top-k of the engine's OWN scores (a stable host sort: the
// picks above the k-th in position order, then its ties in position order, then -1), so the radix select is checked
// bit for bit whatever the scores' precision; chunked calls equal one call bit for bit.
// Mutations (each must move the scores by >= 10x the bound or break a row's set): no relu, no weights, no
// prescale, ties to the higher position, a future key let in, top-k / 2 (1024 of 2048), no LayerNorm bias, an RMS
// norm for the LayerNorm, the LayerNorm eps 1e-6 for 1e-5.
#include "../src/k31/k31_index.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

using strata::k31::IdxKeyFmt;
using Vec = std::vector<float>;
using DVec = std::vector<double>;
using IVec = std::vector<int32_t>;
constexpr double kMinNormal = 1.17549435082228751e-38;
const double kInf = std::numeric_limits<double>::infinity();

// bounds, ~3x the measured gap (2026-11-05, RTX 3090: the fixture's 3 layers / the 2240-token call / the selftest)
constexpr double kBoundKeyUlp = 1.0;     // the float32 keys: float64 LayerNorm, the oracle's two roundings (measured 0)
double g_bound_score = 5e-6;             // scores with float32 keys, max |d| / rms: float32 sums vs float64 (6.7e-7 /
                                         // 1.4e-6 / 1.6e-6; abs 4.1e-7 on the fixture)
double g_bound_score_bf16 = 5e-2;        // scores with BF16 keys (7.2e-3 / 1.6e-2 / 9.4e-3; abs 4.2e-3 on the fixture,
                                         // 14x the plan's M1 "indexer scores max <= 3e-4": the parity runs need F32 keys)
constexpr double kNeedFactor = 10.0;

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
double ulp_unit(double r) {
    const double ar = std::fabs(r);
    if (ar < kMinNormal) return kMinNormal;
    const float f = (float) ar;
    return std::max((double) std::nextafter(f, std::numeric_limits<float>::infinity()) - (double) f, kMinNormal);
}
uint16_t bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

std::vector<void*> g_dev;
template <class T>
T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    cuda_ok(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)), "malloc");
    if (!h.empty()) cuda_ok(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    g_dev.push_back(d);
    return d;
}
void* dzeros(size_t bytes) {
    void* d = nullptr;
    cuda_ok(cudaMalloc(&d, std::max<size_t>(bytes, 4)), "malloc");
    cuda_ok(cudaMemset(d, 0, std::max<size_t>(bytes, 4)), "memset");
    g_dev.push_back(d);
    return d;
}
template <class T>
std::vector<T> host(const void* d, size_t n) {
    std::vector<T> h(n);
    cuda_ok(cudaDeviceSynchronize(), "sync");
    cuda_ok(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}
void free_all() {
    for (void* p : g_dev) cudaFree(p);
    g_dev.clear();
}

// one indexer call of a sequence
struct Call {
    std::string name;
    int T = 0, H = 0, D = 0, n_kv = 0, k = 0;
    long long p0 = 0;
    double eps = 1e-5, prescale = 0;
    Vec q, kpre, w, ln_w, ln_b;   // [T][H][D], [T][D], [T][H], [D], [D]
    // the oracle's (dumps mode)
    Vec k_ref;                    // [T][D]
    DVec score_ref;               // [T][n_kv]
    IVec top_ref;                 // [T][top_cols]
    int top_cols = 0;
};
struct EngineOut {
    Vec keys;                     // [T][D] float32 seam
    std::vector<uint16_t> cache_bf16;   // this call's cache rows (BF16 runs)
    Vec score;                    // [T][n_kv]
    IVec sel;                     // [T][k]
};

// the engine over a sequence of calls sharing one key cache
std::vector<EngineOut> run_engine(const std::vector<Call>& seq, IdxKeyFmt fmt) {
    const int D = seq[0].D;
    const int n_ctx = seq.back().n_kv;
    const size_t eb = fmt == IdxKeyFmt::kF32 ? 4 : 2;
    void* cache = dzeros((size_t) n_ctx * D * eb);
    std::vector<EngineOut> outs;
    for (const Call& c : seq) {
        const float *q = dev(c.q), *kp = dev(c.kpre), *w = dev(c.w), *lw = dev(c.ln_w), *lb = dev(c.ln_b);
        float* kout = (float*) dzeros((size_t) c.T * D * 4);
        float* sc = (float*) dzeros((size_t) c.T * c.n_kv * 4);
        int32_t* sel = (int32_t*) dzeros((size_t) c.T * c.k * 4);
        strata::k31::IndexKeyArgs ka;
        ka.kpre = kp; ka.ln_w = lw; ka.ln_b = lb; ka.eps = c.eps; ka.n_tokens = c.T; ka.dim = D; ka.p0 = c.p0; ka.fmt = fmt;
        ka.cache = cache; ka.k_out = kout;
        strata::k31::IndexScoreArgs sa;
        sa.q = q; sa.w = w; sa.keys = cache; sa.fmt = fmt; sa.n_tokens = c.T; sa.n_head = c.H; sa.dim = D; sa.n_kv = c.n_kv;
        sa.p0 = c.p0; sa.prescale = c.prescale; sa.score = sc; sa.score_stride = c.n_kv;
        strata::k31::IndexTopkArgs ta;
        ta.score = sc; ta.score_stride = c.n_kv; ta.n_tokens = c.T; ta.n_kv = c.n_kv; ta.k = c.k; ta.p0 = c.p0; ta.sel = sel; ta.sel_stride = c.k;
        if (!strata::k31::index_key_launch(ka, nullptr) || !strata::k31::index_score_launch(sa, nullptr) ||
            !strata::k31::index_topk_launch(ta, nullptr)) {
            std::fprintf(stderr, "%s: a launch was refused\n", c.name.c_str());
            std::exit(2);
        }
        EngineOut o;
        o.keys = host<float>(kout, (size_t) c.T * D);
        o.score = host<float>(sc, (size_t) c.T * c.n_kv);
        o.sel = host<int32_t>(sel, (size_t) c.T * c.k);
        if (fmt == IdxKeyFmt::kBf16) {
            o.cache_bf16.resize((size_t) c.T * D);
            cuda_ok(cudaMemcpy(o.cache_bf16.data(), (const uint16_t*) cache + (size_t) c.p0 * D, o.cache_bf16.size() * 2,
                               cudaMemcpyDeviceToHost), "d2h");
        }
        outs.push_back(std::move(o));
    }
    free_all();
    return outs;
}

enum Mut { M_NONE, M_NO_RELU, M_NO_WEIGHTS, M_NO_PRESCALE, M_HIGH_TIES, M_FUTURE, M_TOP_HALF, M_NO_LN_BIAS, M_RMS_NORM,
           M_LN_EPS6, M_COUNT };
const char* kMutName[M_COUNT] = {"none", "no relu", "no weights", "no prescale", "ties to the higher position",
                                 "a future key let in", "top-k / 2", "no LayerNorm bias", "RMS norm for the LayerNorm",
                                 "LayerNorm eps 1e-6 for 1e-5"};
bool score_mut(int m) { return m == M_NO_RELU || m == M_NO_WEIGHTS || m == M_NO_PRESCALE || m == M_NO_LN_BIAS || m == M_RMS_NORM || m == M_LN_EPS6; }

// the oracle's layer_norm on one row: float64 mean/variance, float32 y, float32 y*w, float32 + b
void ref_key_row(const float* x, int D, const Vec& lw, const Vec& lb, double eps, int m, float* out) {
    double mu = 0;
    for (int i = 0; i < D; ++i) mu += x[i];
    mu /= D;
    if (m == M_RMS_NORM) mu = 0;
    double v = 0;
    for (int i = 0; i < D; ++i) v += ((double) x[i] - mu) * ((double) x[i] - mu);
    const double den = std::sqrt(v / D + (m == M_LN_EPS6 ? 1e-6 : eps));
    for (int i = 0; i < D; ++i) {
        const float y = (float) (((double) x[i] - mu) / den);
        const float yw = y * lw[i];
        out[i] = m == M_NO_LN_BIAS ? yw : yw + lb[i];
    }
}

// float64 scores of the rows `rows` of call c (or all), keys = every key of the sequence so far ([n_kv][D])
DVec ref_scores(const Call& c, const Vec& keys, int m, const std::vector<int>& rows) {
    const int H = c.H, D = c.D;
    DVec s((size_t) c.T * c.n_kv, -kInf);
    for (int t : rows) {
        const long long p = c.p0 + t;
        const long long vis = std::min<long long>(c.n_kv - 1, m == M_FUTURE ? p + 1 : p);
        for (long long j = 0; j <= vis; ++j) {
            double tot = 0;
            for (int h = 0; h < H; ++h) {
                const float* q = &c.q[((size_t) t * H + h) * D];
                const float* kk = &keys[(size_t) j * D];
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += (double) q[d] * (double) kk[d];
                if (m != M_NO_RELU) dot = std::max(dot, 0.0);
                const double wv = m == M_NO_WEIGHTS ? 1.0 : (double) c.w[(size_t) t * H + h];
                tot += dot * wv * (m == M_NO_PRESCALE ? 1.0 : c.prescale);
            }
            s[(size_t) t * c.n_kv + j] = tot;
        }
    }
    return s;
}

// the top-k of one row of scores (double): stable descending, ties to the lower position (or the higher), finite or
// -inf entries past the visible end excluded
std::vector<int> select_row(const double* sc, int n_vis, int k, bool high_ties) {
    std::vector<int> o(n_vis);
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return sc[a] > sc[b] || (high_ties && sc[a] == sc[b] && a > b); });
    o.resize(std::min(k, n_vis));
    return o;
}

// exact ties at the cut: when the reference's k-th score is shared by a group of positions the cut splits, the picks
// inside that group must be its lowest positions (topk_sets_agree accepts any swap inside a tie: this is the
// lower-position rule).  Returns false on a violation; `splits` counts the rows where a tie straddles the cut.
bool tie_prefix_ok(const double* sc, int n, const std::vector<int>& want, const std::vector<int>& got, long& splits) {
    if ((int) want.size() >= n || want.empty()) return true;
    double cut = kInf;
    for (int i : want) cut = std::min(cut, sc[i]);
    std::vector<int> grp;
    for (int i = 0; i < n; ++i) if (sc[i] == cut) grp.push_back(i);
    int in_want = 0;
    for (int i : want) in_want += sc[i] == cut;
    if (in_want == (int) grp.size()) return true;   // the cut does not split the group
    ++splits;
    std::set<int> g(got.begin(), got.end());
    int m = 0;
    for (int i : grp) m += g.count(i);
    for (int j = 0; j < (int) grp.size(); ++j) if ((j < m) != (g.count(grp[j]) > 0)) return false;
    return true;
}

// ref/kimi_k31.py topk_sets_agree
bool sets_agree(const double* sc, const std::vector<int>& a_, const std::vector<int>& b_, double tol) {
    std::set<int> a(a_.begin(), a_.end()), b(b_.begin(), b_.end());
    if (a == b) return true;
    if (a.size() != b.size()) return false;
    for (int i : a) if (!std::isfinite(sc[i])) return false;
    for (int i : b) if (!std::isfinite(sc[i])) return false;
    double ma = kInf, mb = kInf;
    for (int i : a) ma = std::min(ma, sc[i]);
    for (int i : b) mb = std::min(mb, sc[i]);
    const double cut = std::max(ma, mb);
    for (int i : a) if (!b.count(i) && std::fabs(sc[i] - cut) > tol) return false;
    for (int i : b) if (!a.count(i) && std::fabs(sc[i] - cut) > tol) return false;
    return true;
}

struct ScoreErr {
    double rel = 0, abs = 0;
    long mask_bad = 0;
};
// over rows `rows`: max |e - r| over the visible (finite reference) entries, / rms of those; masked entries must be -inf
ScoreErr score_err(const Call& c, const Vec& e, const DVec& r, const std::vector<int>& rows) {
    ScoreErr x;
    double ss = 0;
    long n = 0;
    for (int t : rows)
        for (int j = 0; j < c.n_kv; ++j) {
            const size_t i = (size_t) t * c.n_kv + j;
            if (!std::isfinite(r[i])) { x.mask_bad += !(std::isinf(e[i]) && e[i] < 0); continue; }
            const double d = std::fabs((double) e[i] - r[i]);
            x.abs = std::max(x.abs, d != d ? kInf : d);
            ss += r[i] * r[i];
            ++n;
        }
    x.rel = x.abs / std::max(std::sqrt(ss / std::max<long>(n, 1)), 1e-30);
    return x;
}

std::vector<int> sel_row(const IVec& sel, int k, int t) {
    std::vector<int> v;
    for (int i = 0; i < k; ++i) if (sel[(size_t) t * k + i] >= 0) v.push_back(sel[(size_t) t * k + i]);
    return v;
}

// the selection must be exactly the top-k of the engine's own scores, in the kernel's layout
long select_exact_bad(const Call& c, const EngineOut& o) {
    long bad = 0;
    for (int t = 0; t < c.T; ++t) {
        const int n = (int) std::min<long long>(c.p0 + t + 1, c.n_kv), kk = std::min(c.k, n);
        const float* s = &o.score[(size_t) t * c.n_kv];
        std::vector<int> o2(n);
        std::iota(o2.begin(), o2.end(), 0);
        std::stable_sort(o2.begin(), o2.end(), [&](int a, int b) { return s[a] > s[b]; });
        std::vector<int> want;
        if (kk >= n) {
            for (int i = 0; i < n; ++i) want.push_back(i);
        } else {
            const float kth = s[o2[kk - 1]];
            for (int i = 0; i < n; ++i) if (s[i] > kth) want.push_back(i);
            for (int i = 0; i < n && (int) want.size() < kk; ++i) if (s[i] == kth) want.push_back(i);
        }
        while ((int) want.size() < c.k) want.push_back(-1);
        bad += !std::equal(want.begin(), want.end(), o.sel.begin() + (size_t) t * c.k);
    }
    return bad;
}

// all the reference keys of a sequence up to and including call `upto` ([n_kv][D]), with mutation m
Vec ref_keys_seq(const std::vector<Call>& seq, size_t upto, int m) {
    const Call& last = seq[upto];
    Vec keys((size_t) last.n_kv * last.D, 0.0f);
    for (size_t ci = 0; ci <= upto; ++ci) {
        const Call& c = seq[ci];
        for (int t = 0; t < c.T; ++t) ref_key_row(&c.kpre[(size_t) t * c.D], c.D, c.ln_w, c.ln_b, c.eps, m, &keys[(size_t) (c.p0 + t) * c.D]);
    }
    return keys;
}

struct SeqResult {
    int fails = 0;
    double worst_rel = 0, worst_abs = 0, worst_rel_bf16 = 0, worst_abs_bf16 = 0;
    long tie_rows = 0;
    std::vector<int> caught = std::vector<int>(M_COUNT, 0);
};

// gate one sequence; `oracle` = the calls carry the oracle's keys/scores/tops; rows: the query rows to hold to the
// float64 reference (all when empty); engine checks always cover every row
void gate_seq(const std::vector<Call>& seq, bool oracle, const std::vector<std::vector<int>>& rows_in, SeqResult& R) {
    const auto e32 = run_engine(seq, IdxKeyFmt::kF32), e16 = run_engine(seq, IdxKeyFmt::kBf16);
    for (size_t ci = 0; ci < seq.size(); ++ci) {
        const Call& c = seq[ci];
        std::vector<int> rows = rows_in.empty() || rows_in[ci].empty() ? std::vector<int>() : rows_in[ci];
        if (rows.empty()) { rows.resize(c.T); std::iota(rows.begin(), rows.end(), 0); }
        // the reference: the oracle's, or this file's float64
        const Vec keys = ref_keys_seq(seq, ci, M_NONE);
        DVec rs = oracle ? c.score_ref : ref_scores(c, keys, M_NONE, rows);
        Vec kref = oracle ? c.k_ref : Vec(keys.begin() + (size_t) c.p0 * c.D, keys.begin() + (size_t) (c.p0 + c.T) * c.D);
        // keys
        double kulp = 0;
        long bf_bad = 0;
        for (size_t i = 0; i < kref.size(); ++i) {
            kulp = std::max(kulp, std::fabs((double) e32[ci].keys[i] - (double) kref[i]) / ulp_unit(kref[i]));
            bf_bad += e16[ci].cache_bf16[i] != bf16_rne(e16[ci].keys[i]);
        }
        bool kbits = std::memcmp(e32[ci].keys.data(), e16[ci].keys.data(), kref.size() * 4) == 0;
        // scores and sets
        const ScoreErr s32 = score_err(c, e32[ci].score, rs, rows), s16 = score_err(c, e16[ci].score, rs, rows);
        const double tol32 = 2.0 * s32.abs, tol16 = 2.0 * s16.abs;
        long set_bad = 0, set_bad16 = 0, rows_equal = 0, rows_equal16 = 0, over = 0, splits = 0, tie_bad = 0, dummy = 0;
        for (int t : rows) {
            const int n = (int) std::min<long long>(c.p0 + t + 1, c.n_kv);
            over += n > c.k;
            const double* sc = &rs[(size_t) t * c.n_kv];
            const std::vector<int> want = oracle ? sel_row(c.top_ref, c.top_cols, t) : select_row(sc, n, c.k, false);
            const std::vector<int> g32 = sel_row(e32[ci].sel, c.k, t), g16 = sel_row(e16[ci].sel, c.k, t);
            set_bad += !sets_agree(sc, want, g32, tol32);
            set_bad16 += !sets_agree(sc, want, g16, tol16);
            tie_bad += !tie_prefix_ok(sc, n, want, g32, splits) + !tie_prefix_ok(sc, n, want, g16, dummy);
            rows_equal += std::set<int>(want.begin(), want.end()) == std::set<int>(g32.begin(), g32.end());
            rows_equal16 += std::set<int>(want.begin(), want.end()) == std::set<int>(g16.begin(), g16.end());
        }
        const long ex32 = select_exact_bad(c, e32[ci]), ex16 = select_exact_bad(c, e16[ci]);
        const bool ok = kulp <= kBoundKeyUlp && bf_bad == 0 && kbits && s32.rel <= g_bound_score && s16.rel <= g_bound_score_bf16 &&
                        s32.mask_bad == 0 && s16.mask_bad == 0 && set_bad == 0 && set_bad16 == 0 && ex32 == 0 && ex16 == 0 && tie_bad == 0;
        std::printf("  %-8s T %4d p0 %4lld n_kv %4d k %4d (%ld rows past k): keys %.1f ULP, bf16 rows %s;  scores f32 %.2e (abs %.2e) "
                    "bf16 %.2e (abs %.2e), masks %s;  sets agree f32 %s bf16 %s (equal %ld / %ld of %zu);  ties at the cut %ld rows, "
                    "lowest positions %s;  select exact %s  %s\n",
                    c.name.c_str(), c.T, c.p0, c.n_kv, c.k, over, kulp, bf_bad ? "WRONG" : "= rne(f32)", s32.rel, s32.abs, s16.rel,
                    s16.abs, s32.mask_bad + s16.mask_bad ? "WRONG" : "ok", set_bad ? "NO" : "yes", set_bad16 ? "NO" : "yes", rows_equal,
                    rows_equal16, rows.size(), splits, tie_bad ? "NO" : "yes", ex32 + ex16 ? "NO" : "yes", ok ? "ok" : "FAIL");
        R.tie_rows += splits;
        R.fails += !ok;
        R.worst_rel = std::max(R.worst_rel, s32.rel); R.worst_abs = std::max(R.worst_abs, s32.abs);
        R.worst_rel_bf16 = std::max(R.worst_rel_bf16, s16.rel); R.worst_abs_bf16 = std::max(R.worst_abs_bf16, s16.abs);
        // mutations: a float64 reference with the wrong reading, through the same gate as the engine (float32 keys)
        const double need = kNeedFactor * g_bound_score;
        for (int m = 1; m < M_COUNT; ++m) {
            bool caught = false;
            if (score_mut(m)) {
                const Vec km = ref_keys_seq(seq, ci, m);
                const DVec sm = ref_scores(c, km, m, rows);
                Vec smf(sm.size());
                for (size_t i = 0; i < sm.size(); ++i) smf[i] = (float) sm[i];
                caught = score_err(c, smf, rs, rows).rel >= need;
                for (int t : rows) {
                    if (caught) break;
                    const int n = (int) std::min<long long>(c.p0 + t + 1, c.n_kv);
                    const double* sc = &rs[(size_t) t * c.n_kv];
                    const std::vector<int> want = oracle ? sel_row(c.top_ref, c.top_cols, t) : select_row(sc, n, c.k, false);
                    caught = !sets_agree(sc, want, select_row(&sm[(size_t) t * c.n_kv], n, c.k, false), tol32);
                }
            } else {
                for (int t : rows) {
                    if (caught) break;
                    const int n = (int) std::min<long long>(c.p0 + t + 1, c.n_kv);
                    const double* sc = &rs[(size_t) t * c.n_kv];
                    const std::vector<int> want = oracle ? sel_row(c.top_ref, c.top_cols, t) : select_row(sc, n, c.k, false);
                    std::vector<int> got;
                    if (m == M_HIGH_TIES) got = select_row(sc, n, c.k, true);
                    else if (m == M_TOP_HALF) got = select_row(sc, n, std::max(1, c.k / 2), false);
                    else {   // M_FUTURE: the key at p + 1 visible (its float64 score from this file's reference)
                        if (c.p0 + t + 1 >= c.n_kv) continue;
                        DVec row(sc, sc + c.n_kv);
                        const DVec fut = ref_scores(c, keys, M_FUTURE, {t});
                        row[n] = fut[(size_t) t * c.n_kv + n];
                        got = select_row(row.data(), n + 1, c.k, false);
                    }
                    long sp = 0;
                    caught = !sets_agree(sc, want, got, tol32) || !tie_prefix_ok(sc, n, want, got, sp);
                }
            }
            R.caught[m] |= caught;
        }
    }
}

int report(const SeqResult& R, const char* where, bool fixture) {
    int fails = R.fails;
    std::printf("  worst: scores f32 keys %.2e (abs %.2e, bound %.0e), bf16 keys %.2e (abs %.2e, bound %.0e); rows with a tie at the cut %ld\n",
                R.worst_rel, R.worst_abs, g_bound_score, R.worst_rel_bf16, R.worst_abs_bf16, g_bound_score_bf16, R.tie_rows);
    for (int m = 1; m < M_COUNT; ++m) {
        const bool blind = fixture && m == M_LN_EPS6;   // the fixture's keys have unit variance: the eps cannot show
        std::printf("  mutation %-30s %s\n", kMutName[m], R.caught[m] ? "caught" : blind ? "below the dumps' reach (the selftest catches it)" : "NOT CAUGHT");
        fails += !R.caught[m] && !blind;
    }
    (void) where;
    return fails;
}

uint64_t g_s = 0x2545F4914F6CDD1Dull;
double urand() { g_s = g_s * 6364136223846793005ull + 1442695040888963407ull; return (double) (g_s >> 11) / 9007199254740992.0; }
double nrand() { const double u = std::max(urand(), 1e-300), v = urand(); return std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v); }

Call synth_call(const char* name, int T, long long p0, int n_kv, int H, int D, int k, const Vec& lw, const Vec& lb) {
    Call c;
    c.name = name; c.T = T; c.p0 = p0; c.n_kv = n_kv; c.H = H; c.D = D; c.k = k; c.eps = 1e-5; c.prescale = 1.0 / std::sqrt((double) H * D);
    c.ln_w = lw; c.ln_b = lb;
    c.q.resize((size_t) T * H * D); c.kpre.resize((size_t) T * D); c.w.resize((size_t) T * H);
    for (auto& x : c.q) x = (float) (nrand() / std::sqrt((double) D));
    for (auto& x : c.kpre) x = (float) (0.5 + nrand());
    for (auto& x : c.w) x = (float) nrand();
    return c;
}

int selftest() {
    std::printf("k31_index_test --selftest (K3.1: %d heads x %d, top %d)\n", strata::k31::kIdxHeads, strata::k31::kIdxDim, strata::k31::kIdxTopK);
    const int H = strata::k31::kIdxHeads, D = strata::k31::kIdxDim, k = strata::k31::kIdxTopK, T1 = 2200, T2 = 64;
    Vec lw(D), lb(D);
    for (auto& x : lw) x = (float) (1.0 + 0.1 * nrand());
    for (auto& x : lb) x = (float) (0.1 * nrand());
    std::vector<Call> seq = {synth_call("pre", T1, 0, T1 + T2, H, D, k, lw, lb), synth_call("cont", T2, T1, T1 + T2, H, D, k, lw, lb)};
    Call& a = seq[0];
    for (int t = 600; t < 900; ++t) std::copy(&a.kpre[(size_t) 599 * D], &a.kpre[(size_t) 600 * D], &a.kpre[(size_t) t * D]);   // 300 tied keys
    for (int t = 1000; t < 1040; ++t)   // low-variance keys: variance ~1e-6, where eps 1e-5 vs 1e-6 decides the key
        for (int d = 0; d < D; ++d) a.kpre[(size_t) t * D + d] = (float) (0.3 + 1e-3 * nrand());
    for (int h = 0; h < H; ++h) seq[1].w[(size_t) 5 * H + h] = 0.0f;   // a row of zero weights: every score 0, positions 0..2047
    for (int t = 20; t < 30; ++t) std::copy(&a.kpre[(size_t) 599 * D], &a.kpre[(size_t) 600 * D], &seq[1].kpre[(size_t) t * D]);
    // the float64 reference on a sample of rows (every row checked by the engine-side rules)
    std::vector<int> r0, r1;
    for (int t = 0; t < T1; t += 157) r0.push_back(t);
    for (int t = 2040; t < T1; t += 9) r0.push_back(t);
    for (int t = 0; t < T2; t += 3) r1.push_back(t);
    if (std::find(r1.begin(), r1.end(), 5) == r1.end()) r1.push_back(5);
    SeqResult R;
    gate_seq(seq, false, {r0, r1}, R);
    int fails = report(R, "selftest", false);
    {   // chunked == one call: the prefill and the continuation as one 2264-token call
        Call one = seq[0];
        one.name = "one"; one.T = T1 + T2;
        one.q.insert(one.q.end(), seq[1].q.begin(), seq[1].q.end());
        one.kpre.insert(one.kpre.end(), seq[1].kpre.begin(), seq[1].kpre.end());
        one.w.insert(one.w.end(), seq[1].w.begin(), seq[1].w.end());
        const auto e1 = run_engine({one}, IdxKeyFmt::kBf16), e2 = run_engine(seq, IdxKeyFmt::kBf16);
        const size_t off = (size_t) T1 * one.n_kv, offs = (size_t) T1 * k;
        const bool same = std::memcmp(e1[0].score.data() + off, e2[1].score.data(), e2[1].score.size() * 4) == 0 &&
                          std::memcmp(e1[0].sel.data() + offs, e2[1].sel.data(), e2[1].sel.size() * 4) == 0;
        const bool pre_same = std::memcmp(e1[0].score.data(), e2[0].score.data(), off * 4) == 0;
        std::printf("  prefill + continuation == one call, bit for bit: scores and picks %s, prefill rows %s\n", same ? "ok" : "FAIL", pre_same ? "ok" : "FAIL");
        fails += !same + !pre_same;
        // the zero-weight row: positions 0..2047 exactly
        bool zero_ok = true;
        for (int i = 0; i < k; ++i) zero_ok = zero_ok && e2[1].sel[(size_t) 5 * k + i] == i;
        std::printf("  a row of zero weights picks positions 0..%d: %s\n", k - 1, zero_ok ? "ok" : "FAIL");
        fails += !zero_ok;
    }
    {   // refusals
        float* d = (float*) dzeros(1 << 16);
        int32_t* di = (int32_t*) dzeros(1 << 16);
        strata::k31::IndexScoreArgs s;
        s.q = d; s.w = d; s.keys = d; s.score = d; s.n_tokens = 1; s.n_head = 4; s.dim = 32; s.n_kv = 8; s.score_stride = 8; s.prescale = 0.1;
        strata::k31::IndexTopkArgs t;
        t.score = d; t.score_stride = 8; t.n_tokens = 1; t.n_kv = 8; t.k = 4; t.sel = di; t.sel_stride = 4;
        strata::k31::IndexKeyArgs ka;
        ka.kpre = d; ka.ln_w = d; ka.ln_b = d; ka.n_tokens = 1; ka.dim = 32; ka.cache = d;
        int n = 0, refused = 0;
        auto rs = [&](strata::k31::IndexScoreArgs b) { ++n; refused += !strata::k31::index_score_launch(b, nullptr); };
        auto rt = [&](strata::k31::IndexTopkArgs b) { ++n; refused += !strata::k31::index_topk_launch(b, nullptr); };
        auto rk = [&](strata::k31::IndexKeyArgs b) { ++n; refused += !strata::k31::index_key_launch(b, nullptr); };
        auto b = s; b.n_head = 33; rs(b);
        b = s; b.dim = 96; rs(b);
        b = s; b.p0 = 8; rs(b);                 // a query past the cache
        b = s; b.score_stride = 7; rs(b);
        b = s; b.prescale = 0; rs(b);
        b = s; b.keys = nullptr; rs(b);
        auto c = t; c.k = 0; rt(c);
        c = t; c.sel_stride = 3; rt(c);
        c = t; c.p0 = -1; rt(c);
        auto e = ka; e.eps = 0; rk(e);
        e = ka; e.dim = 0; rk(e);
        e = ka; e.cache = nullptr; rk(e);
        void* huge = nullptr;
        if (cudaMalloc(&huge, (size_t) 1 << 50) != cudaSuccess) { rs(s); }   // a pending error
        (void) cudaGetLastError();
        const bool runs = strata::k31::index_score_launch(s, nullptr) && strata::k31::index_topk_launch(t, nullptr) &&
                          strata::k31::index_key_launch(ka, nullptr) && cudaDeviceSynchronize() == cudaSuccess;
        free_all();
        std::printf("  refusals: %d/%d (33 heads, dim 96, a query past the cache, stride < n_kv, prescale 0, null keys, k 0, "
                    "sel stride < k, p0 -1, eps 0, dim 0, null cache, a pending error); valid calls %s  %s\n", refused, n,
                    runs ? "run" : "FAIL", refused == n && runs ? "ok" : "FAIL");
        fails += refused != n || !runs;
    }
    return fails;
}

// ---- the dumps
bool read_raw(const std::string& path, size_t n, void* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(static_cast<char*>(out), (std::streamsize) (n * 4));
    return (size_t) f.gcount() == n * 4 && f.peek() == std::char_traits<char>::eof();
}
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
std::string obj_str(const std::string& js, std::pair<size_t, size_t> sp, const std::string& key) {
    const size_t b = js.find("\"" + key + "\"", sp.first);
    if (b == std::string::npos || b > sp.second) return "";
    const size_t q1 = js.find('"', js.find(':', b) + 1), q2 = js.find('"', q1 + 1);
    return js.substr(q1 + 1, q2 - q1 - 1);
}
double obj_num(const std::string& js, std::pair<size_t, size_t> sp, const std::string& key) {
    const size_t b = js.find("\"" + key + "\"", sp.first);
    if (b == std::string::npos || b > sp.second) { std::fprintf(stderr, "index.json: no %s\n", key.c_str()); std::exit(2); }
    return std::strtod(js.c_str() + js.find(':', b) + 1, nullptr);
}

int dumps(const std::string& dir) {
    std::stringstream jss;
    jss << std::ifstream(dir + "/index.json").rdbuf();
    const std::string js = jss.str();
    if (js.empty()) { std::printf("%s/index.json: missing (python tools/kimi_k31_indexer_dump.py --out %s)  FAIL\n", dir.c_str(), dir.c_str()); return 1; }
    const auto top = object_span(js, 0);
    const int H = (int) obj_num(js, top, "H"), D = (int) obj_num(js, top, "D");
    const double eps = obj_num(js, top, "eps"), prescale = obj_num(js, top, "prescale");
    std::vector<Call> calls;
    const size_t arr = js.find('[', js.find("\"calls\"")), arr_end = js.find(']', arr);   // (call objects hold no arrays)
    for (size_t at = js.find('{', arr); at != std::string::npos && at < arr_end;) {
        const auto sp = object_span(js, at);
        if (sp.first == std::string::npos) break;
        Call c;
        c.name = obj_str(js, sp, "name"); c.T = (int) obj_num(js, sp, "T"); c.p0 = (long long) obj_num(js, sp, "p0");
        c.n_kv = (int) obj_num(js, sp, "n_kv"); c.top_cols = (int) obj_num(js, sp, "k"); c.k = (int) obj_num(js, sp, "top_k");
        c.H = H; c.D = D; c.eps = eps; c.prescale = prescale;
        const int layer = (int) obj_num(js, sp, "layer");
        const std::string p = dir + "/idx." + c.name + ".";
        c.q.resize((size_t) c.T * H * D); c.kpre.resize((size_t) c.T * D); c.w.resize((size_t) c.T * H); c.k_ref.resize((size_t) c.T * D);
        c.ln_w.resize(D); c.ln_b.resize(D); c.top_ref.resize((size_t) c.T * c.top_cols);
        Vec hi((size_t) c.T * c.n_kv), lo((size_t) c.T * c.n_kv);
        const std::string ln = dir + "/idx.ln." + std::to_string(layer) + ".";
        if (!read_raw(p + "q.f32", c.q.size(), c.q.data()) || !read_raw(p + "kpre.f32", c.kpre.size(), c.kpre.data()) ||
            !read_raw(p + "w.f32", c.w.size(), c.w.data()) || !read_raw(p + "k.f32", c.k_ref.size(), c.k_ref.data()) ||
            !read_raw(p + "score_hi.f32", hi.size(), hi.data()) || !read_raw(p + "score_lo.f32", lo.size(), lo.data()) ||
            !read_raw(p + "top.f32", c.top_ref.size(), c.top_ref.data()) || !read_raw(ln + "w.f32", D, c.ln_w.data()) ||
            !read_raw(ln + "b.f32", D, c.ln_b.data())) {
            std::printf("  %s: unreadable or the wrong size  FAIL\n", c.name.c_str());
            return 1;
        }
        c.score_ref.resize(hi.size());
        for (size_t i = 0; i < hi.size(); ++i) c.score_ref[i] = std::isfinite(hi[i]) ? (double) hi[i] + (double) lo[i] : (double) hi[i];
        calls.push_back(std::move(c));
        at = js.find('{', sp.second + 1);
    }
    std::printf("k31_index_test --dumps %s: %d heads x %d, eps %g, %zu calls\n", dir.c_str(), H, D, eps, calls.size());
    SeqResult R;
    std::vector<Call> longseq;
    for (const Call& c : calls) {
        if (c.name.rfind("long.", 0) == 0) { longseq.push_back(c); continue; }
        gate_seq({c}, true, {}, R);
        // this file's float64 reference must reproduce the oracle (keys bit for bit, scores to float64 rounding)
        const Vec keys = ref_keys_seq({c}, 0, M_NONE);
        std::vector<int> all(c.T);
        std::iota(all.begin(), all.end(), 0);
        const DVec rs = ref_scores(c, keys, M_NONE, all);
        double md = 0;
        for (size_t i = 0; i < rs.size(); ++i) if (std::isfinite(c.score_ref[i])) md = std::max(md, std::fabs(rs[i] - c.score_ref[i]));
        const bool kok = std::memcmp(keys.data(), c.k_ref.data(), c.k_ref.size() * 4) == 0;
        std::printf("    the float64 reference vs the oracle: keys %s, scores max |d| %.1e  %s\n", kok ? "bit for bit" : "DIFFER", md, kok && md < 1e-12 ? "ok" : "FAIL");
        R.fails += !(kok && md < 1e-12);
    }
    if (!longseq.empty()) {
        gate_seq(longseq, true, {}, R);
        // chunked == one call on the engine (the 2200 + 40 calls as one 2240-token call)
        Call one = longseq[0];
        one.name = "one"; one.T = longseq.back().p0 + longseq.back().T; one.n_kv = longseq.back().n_kv;
        for (size_t i = 1; i < longseq.size(); ++i) {
            one.q.insert(one.q.end(), longseq[i].q.begin(), longseq[i].q.end());
            one.kpre.insert(one.kpre.end(), longseq[i].kpre.begin(), longseq[i].kpre.end());
            one.w.insert(one.w.end(), longseq[i].w.begin(), longseq[i].w.end());
        }
        std::vector<Call> seq2 = longseq;
        for (auto& c : seq2) c.n_kv = one.n_kv;   // the same score-row width, so the rows compare bit for bit
        const auto e1 = run_engine({one}, IdxKeyFmt::kBf16), e2 = run_engine(seq2, IdxKeyFmt::kBf16);
        bool same = true;
        size_t row = 0;
        for (size_t ci = 0; ci < seq2.size(); ++ci) {
            same = same && std::memcmp(e1[0].score.data() + row * one.n_kv, e2[ci].score.data(), e2[ci].score.size() * 4) == 0 &&
                   std::memcmp(e1[0].sel.data() + row * one.k, e2[ci].sel.data(), e2[ci].sel.size() * 4) == 0;
            row += seq2[ci].T;
        }
        std::printf("  long: %zu calls == one %d-token call, bit for bit (scores, picks): %s\n", seq2.size(), one.T, same ? "ok" : "FAIL");
        R.fails += !same;
    }
    return report(R, "dumps", true);
}

}  // namespace

int main(int argc, char** argv) {
    int fails = 0;
    bool any = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") { fails += selftest(); any = true; }
        else if (a == "--dumps" && i + 1 < argc) { fails += dumps(argv[++i]); any = true; }
        else { std::fprintf(stderr, "usage: %s [--selftest] [--dumps DIR]\n", argv[0]); return 2; }
    }
    if (!any) fails += selftest();
    std::printf("k31_index_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
