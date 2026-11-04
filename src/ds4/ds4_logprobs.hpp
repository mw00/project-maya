// src/ds4/ds4_logprobs.hpp - maya-ds4 --serve's log-probabilities (GEN logprobs=<n>, the server's OpenAI `logprobs` /
// `top_logprobs`): for each token a request writes, the log-probability of that token and the n most likely tokens of
// the row it came from.  The distribution is the model's softmax over its logits after the request's penalties
// (llama.cpp's repeat / frequency / presence, as the sampler counts them), at temperature 1 and before top_k, top_p and
// min_p - OpenAI's convention: the same numbers whatever the request's temperature and cuts, the penalties included
// because they are part of the request's model of the next token.  A token the draw could never pick (its penalized
// logit -inf) has logprob -inf.  The device kernel (ds4_kernels.cu: logprob_rows) and the host reference below share
// the penalty arithmetic and the order (higher logit first, the lower id on a tie); tests/ds4_logprobs_test.cpp checks
// the host side and the protocol line.
#pragma once

#include <math.h>   // (expf, fmaxf: the host's and the device's alike)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DS4_LP_HD __host__ __device__
#else
#define DS4_LP_HD
#endif

namespace strata::ds4 {

constexpr int kTopLogprobs = 20;   // OpenAI's top_logprobs at most

/// One written token's row: its own logprob and the row's top n (most likely first).  Plain data: the kernel writes
/// it on the device, a copy brings it to the host.
struct LogprobRow {
    int32_t id = -1;     // the token written
    float lp = 0.0f;     // its logprob
    int32_t n = 0;       // alternatives below (n <= kTopLogprobs; fewer where the row has fewer finite logits)
    int32_t top[kTopLogprobs] = {};
    float top_lp[kTopLogprobs] = {};
};

/// A logit after the penalties for a token seen `count` times in the window (the sampler's apply_penalties: the
/// repeat penalty divides a positive logit and multiplies the others; presence counts once)
DS4_LP_HD inline float lp_penalized(float x, int count, float repeat, float freq, float present) {
    if (count <= 0) return x;
    if (x <= 0.0f) x *= repeat;
    else x /= repeat;
    return x - ((float) count * freq + present);
}

/// The order of the top list: a before b (the higher logit, the lower id on a tie)
DS4_LP_HD inline bool lp_before(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

/// The kernel's pieces (one thread's stride of a row; tests/ds4_logprobs_test.cpp runs them on the host): a running
/// log-sum-exp (m, s: the sum of exp(x - m) over the values pushed) and the k best values in order.  push() takes the
/// values in ascending id, so of equal values the lower id stays first; -inf (never drawn) and NaN are skipped.
struct LpStride {
    float m = -INFINITY, s = 0.0f;
    float tv[kTopLogprobs];
    int32_t ti[kTopLogprobs];
    int tn = 0;
    DS4_LP_HD void push(float x, int v, int k) {
        if (!(x > -INFINITY)) return;
        if (x > m) {
            s = s * expf(m - x) + 1.0f;
            m = x;
        } else
            s += expf(x - m);
        if (tn < k || (k > 0 && x > tv[k - 1])) {
            int p = tn < k ? tn++ : k - 1;
            for (; p > 0 && x > tv[p - 1]; --p) tv[p] = tv[p - 1], ti[p] = ti[p - 1];
            tv[p] = x, ti[p] = v;
        }
    }
};
/// (m, s) += (m2, s2): two strides' log-sum-exp together
DS4_LP_HD inline void lse_merge(float& m, float& s, float m2, float s2) {
    const float mx = fmaxf(m, m2);
    if (mx == -INFINITY) return;   // (both empty)
    s = (m == -INFINITY ? 0.0f : s * expf(m - mx)) + (m2 == -INFINITY ? 0.0f : s2 * expf(m2 - mx));
    m = mx;
}

/// The host reference: `row` (V logits), the penalty window `hist` (h ids, -1 for none; ids outside [0, V) ignored),
/// the token written `id` and the top k (0 .. kTopLogprobs) -> `out`.  NaN logits never count; the sum in double.
inline void logprob_row_host(const float* row, int V, const int32_t* hist, int h, float repeat, float freq,
                             float present, int id, int k, LogprobRow& out) {
    std::vector<int> cnt;
    if (hist != nullptr && h > 0) {
        cnt.assign((size_t) V, 0);
        for (int i = 0; i < h; ++i)
            if (hist[i] >= 0 && hist[i] < V) ++cnt[(size_t) hist[i]];
    }
    const auto val = [&](int v) { return cnt.empty() ? row[v] : lp_penalized(row[v], cnt[(size_t) v], repeat, freq, present); };
    float mx = -INFINITY;
    for (int v = 0; v < V; ++v) mx = std::max(mx, val(v));   // (std::max keeps mx over a NaN)
    double s = 0.0;
    for (int v = 0; v < V; ++v) {
        const float x = val(v);
        if (x > -INFINITY) s += std::exp((double) x - (double) mx);
    }
    const double lse = mx > -INFINITY ? (double) mx + std::log(s) : (double) -INFINITY;
    out = LogprobRow();
    out.id = id;
    out.lp = id >= 0 && id < V ? (float) ((double) val(id) - lse) : NAN;
    k = std::max(0, std::min(k, kTopLogprobs));
    std::vector<int> ord;
    for (int v = 0; v < V; ++v)
        if (val(v) > -INFINITY) ord.push_back(v);
    const size_t kk = std::min(ord.size(), (size_t) k);
    std::partial_sort(ord.begin(), ord.begin() + (long) kk, ord.end(),
                      [&](int a, int b) { return lp_before(val(a), a, val(b), b); });
    for (size_t j = 0; j < kk; ++j) {
        out.top[j] = ord[j];
        out.top_lp[j] = (float) ((double) val(ord[j]) - lse);
    }
    out.n = (int32_t) kk;
}

/// The protocol line after the token's `T` line: `LP [<rid> ]<id> <logprob> <id>:<logprob> ...` (rid >= 0: --parallel's
/// tag, as on its T lines); -inf and nan as printf spells them
inline std::string lp_line(long long rid, const LogprobRow& r) {
    std::string s = "LP";
    char b[48];
    if (rid >= 0) {
        std::snprintf(b, sizeof b, " %lld", rid);
        s += b;
    }
    std::snprintf(b, sizeof b, " %d %.7g", (int) r.id, (double) r.lp);
    s += b;
    for (int j = 0; j < r.n && j < kTopLogprobs; ++j) {
        std::snprintf(b, sizeof b, " %d:%.7g", (int) r.top[j], (double) r.top_lp[j]);
        s += b;
    }
    return s;
}

}  // namespace strata::ds4
