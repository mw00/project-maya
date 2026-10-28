// maya-ds4 --serve's log-probabilities (src/ds4/ds4_logprobs.hpp): the host reference against a plain log-softmax in
// double, the penalties as the sampler applies them, the order of the top list (ties: the lower id), the rows a draw
// can never pick left out; the kernel's own pieces (LpStride, lse_merge) run as its threads run them, against the
// reference; and the protocol's LP line.  No GPU, no model: rows are made up here.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ds4/ds4_logprobs.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using strata::ds4::kTopLogprobs;
using strata::ds4::logprob_row_host;
using strata::ds4::LogprobRow;
using strata::ds4::lp_before;
using strata::ds4::lp_line;
using strata::ds4::lp_penalized;
using strata::ds4::LpStride;
using strata::ds4::lse_merge;

namespace {

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * (1.0 + std::fabs(b)); }

// log-softmax of the penalized row in double, the plain way
std::vector<double> log_softmax(const std::vector<float>& row, const std::vector<int32_t>& hist, float rep, float fr,
                                float pr) {
    std::vector<double> x(row.size());
    for (size_t v = 0; v < row.size(); ++v) {
        int c = 0;
        for (int32_t h : hist) c += h == (int32_t) v;
        x[v] = c > 0 ? (double) lp_penalized(row[v], c, rep, fr, pr) : (double) row[v];
    }
    double mx = -INFINITY;
    for (double a : x)
        if (a > mx) mx = a;
    double s = 0.0;
    for (double a : x)
        if (a > -INFINITY) s += std::exp(a - mx);
    for (double& a : x) a = a - (mx + std::log(s));
    return x;
}

// the kernel's arithmetic on the host: nt threads, thread i the ids i, i + nt, ...; the sums merged, then k rounds of
// the best head of the threads' lists (each winner's thread moves to its next)
LogprobRow kernel_like(const std::vector<float>& row, const std::vector<int32_t>& hist, float rep, float fr, float pr,
                       int id, int k, int nt) {
    const int V = (int) row.size();
    std::vector<int> cnt((size_t) V, 0);
    for (int32_t h : hist)
        if (h >= 0 && h < V) ++cnt[(size_t) h];
    const auto val = [&](int v) { return cnt[(size_t) v] > 0 ? lp_penalized(row[(size_t) v], cnt[(size_t) v], rep, fr, pr) : row[(size_t) v]; };
    std::vector<LpStride> st((size_t) nt);
    for (int t = 0; t < nt; ++t)
        for (int v = t; v < V; v += nt) st[(size_t) t].push(val(v), v, k);
    float m = -INFINITY, s = 0.0f;
    for (const LpStride& x : st) lse_merge(m, s, x.m, x.s);
    const float lse = m == -INFINITY ? -INFINITY : m + logf(s);
    LogprobRow o;
    o.id = id;
    o.lp = val(id) - lse;
    std::vector<int> hp((size_t) nt, 0);
    int n = 0;
    for (; n < k; ++n) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bt = -1;
        for (int t = 0; t < nt; ++t) {
            const LpStride& x = st[(size_t) t];
            if (hp[(size_t) t] < x.tn && lp_before(x.tv[hp[(size_t) t]], x.ti[hp[(size_t) t]], bv, bi))
                bv = x.tv[hp[(size_t) t]], bi = x.ti[hp[(size_t) t]], bt = t;
        }
        if (bt < 0) break;
        o.top[n] = bi, o.top_lp[n] = bv - lse;
        ++hp[(size_t) bt];
    }
    o.n = n;
    return o;
}

std::vector<float> random_row(int V, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, 4.0f);
    std::vector<float> r((size_t) V);
    for (float& x : r) x = d(g);
    return r;
}

// the LP line read back as the server reads it: [rid] id logprob id:logprob ...
struct Parsed {
    long long rid = -1;
    int id = -1;
    double lp = 0;
    std::vector<std::pair<int, double>> top;
};
Parsed parse(const std::string& line, bool tagged) {
    Parsed p;
    std::istringstream in(line);
    std::string w;
    in >> w;
    assert(w == "LP");
    if (tagged) in >> p.rid;
    in >> p.id >> w;
    p.lp = std::strtod(w.c_str(), nullptr);
    while (in >> w) {
        const size_t c = w.find(':');
        assert(c != std::string::npos);
        p.top.emplace_back(std::atoi(w.substr(0, c).c_str()), std::strtod(w.c_str() + c + 1, nullptr));
    }
    return p;
}

}  // namespace

int main() {
    {   // a random row: the reference against the plain log-softmax, its top k in order, exp of the top k summing to at
        // most 1, the chosen token's own logprob; no penalties
        const int V = 5000;
        const std::vector<float> row = random_row(V, 1);
        const std::vector<double> ls = log_softmax(row, {}, 1.0f, 0.0f, 0.0f);
        LogprobRow r;
        logprob_row_host(row.data(), V, nullptr, 0, 1.0f, 0.0f, 0.0f, 123, 20, r);
        assert(r.id == 123 && r.n == 20);
        assert(near(r.lp, ls[123], 1e-5));
        double tot = 0.0;
        for (int j = 0; j < r.n; ++j) {
            assert(near(r.top_lp[j], ls[(size_t) r.top[j]], 1e-5));
            if (j > 0) assert(lp_before(row[(size_t) r.top[j - 1]], r.top[j - 1], row[(size_t) r.top[j]], r.top[j]));
            tot += std::exp((double) r.top_lp[j]);
        }
        assert(tot <= 1.0 + 1e-6);
        // nothing outside the list is above its last
        for (int v = 0; v < V; ++v) {
            bool in = false;
            for (int j = 0; j < r.n; ++j) in = in || r.top[j] == v;
            if (!in) assert(!lp_before(row[(size_t) v], v, row[(size_t) r.top[r.n - 1]], r.top[r.n - 1]));
        }
        // the whole distribution: exp over every token sums to 1
        double all = 0.0;
        for (double a : ls) all += std::exp(a);
        assert(near(all, 1.0, 1e-9));
        // a greedy token is its row's first
        int am = 0;
        for (int v = 1; v < V; ++v)
            if (row[(size_t) v] > row[(size_t) am]) am = v;
        LogprobRow g;
        logprob_row_host(row.data(), V, nullptr, 0, 1.0f, 0.0f, 0.0f, am, 3, g);
        assert(g.n == 3 && g.top[0] == am && g.top_lp[0] == g.lp && g.lp < 0.0f);
    }
    {   // the penalties: as the sampler applies them (a positive logit divided, a negative one multiplied, frequency per
        // count, presence once), over the window's ids; an id outside the vocabulary or -1 counts for nothing
        assert(lp_penalized(2.0f, 1, 2.0f, 0.0f, 0.0f) == 1.0f);
        assert(lp_penalized(-2.0f, 1, 2.0f, 0.0f, 0.0f) == -4.0f);
        assert(lp_penalized(3.0f, 2, 1.0f, 0.5f, 0.25f) == 3.0f - 1.25f);
        assert(lp_penalized(3.0f, 0, 2.0f, 0.5f, 0.25f) == 3.0f);
        std::vector<float> row = {1.0f, 3.0f, 2.9f, -1.0f, 0.5f};
        const std::vector<int32_t> hist = {-1, -1, 1, 1, 3, 99};
        LogprobRow r;
        logprob_row_host(row.data(), 5, hist.data(), (int) hist.size(), 1.3f, 0.2f, 0.1f, 1, 5, r);
        const std::vector<double> ls = log_softmax(row, {1, 1, 3}, 1.3f, 0.2f, 0.1f);
        assert(near(r.lp, ls[1], 1e-6));
        assert(r.n == 5 && r.top[0] == 2);   // (3.0 / 1.3 - 0.4 - 0.1 = 1.81 falls below 2.9)
        for (int j = 0; j < r.n; ++j) assert(near(r.top_lp[j], ls[(size_t) r.top[j]], 1e-6));
        // without the window the same row puts 1 first
        logprob_row_host(row.data(), 5, nullptr, 0, 1.3f, 0.2f, 0.1f, 1, 5, r);
        assert(r.top[0] == 1);
    }
    {   // ties: the lower id first, in every place of the list; -inf never listed, its logprob -inf; NaN never counts
        std::vector<float> row(64, 0.0f);
        row[40] = row[7] = row[23] = 5.0f;
        row[3] = -INFINITY;
        row[9] = NAN;
        LogprobRow r;
        logprob_row_host(row.data(), 64, nullptr, 0, 1.0f, 0.0f, 0.0f, 3, 6, r);
        assert(r.n == 6 && r.top[0] == 7 && r.top[1] == 23 && r.top[2] == 40);
        assert(r.top[3] == 0 && r.top[4] == 1 && r.top[5] == 2);   // (the zeros: 0, 1, 2 - never 3 or 9)
        assert(std::isinf(r.lp) && r.lp < 0.0f);
        assert(r.top_lp[0] == r.top_lp[1] && r.top_lp[1] == r.top_lp[2]);
        // k larger than the finite values: only those (62 of 64 here when k is 20: all fit)
        std::vector<float> few(8, -INFINITY);
        few[5] = 1.0f, few[2] = 1.0f;
        logprob_row_host(few.data(), 8, nullptr, 0, 1.0f, 0.0f, 0.0f, 5, 20, r);
        assert(r.n == 2 && r.top[0] == 2 && r.top[1] == 5);
        assert(near(r.lp, std::log(0.5), 1e-6));
        // k 0: the token's own logprob alone
        logprob_row_host(few.data(), 8, nullptr, 0, 1.0f, 0.0f, 0.0f, 2, 0, r);
        assert(r.n == 0 && near(r.lp, std::log(0.5), 1e-6));
        // an id outside the row: NaN (the server reads it as no number)
        logprob_row_host(few.data(), 8, nullptr, 0, 1.0f, 0.0f, 0.0f, 8, 2, r);
        assert(std::isnan(r.lp) && r.n == 2);
    }
    {   // the kernel's pieces as its threads run them (1, 3, 32 and 256 threads; with and without penalties, with ties
        // and -inf in the row) give the reference's list and logprobs
        const int V = 129280 / 16;   // (a 16th of the vocabulary: the same code, a quicker test)
        std::vector<float> row = random_row(V, 7);
        for (int v = 100; v < V; v += 977) row[(size_t) v] = 9.5f;   // (ties at the top)
        for (int v = 50; v < V; v += 501) row[(size_t) v] = -INFINITY;
        std::vector<int32_t> hist;
        for (int i = 0; i < 300; ++i) hist.push_back((int32_t) ((i * 7919) % V));
        hist.push_back(100);
        hist.push_back(-1);
        for (int pen = 0; pen < 2; ++pen)
            for (int nt : {1, 3, 32, 256})
                for (int k : {0, 1, 5, 20}) {
                    const std::vector<int32_t> h = pen ? hist : std::vector<int32_t>();
                    LogprobRow ref;
                    logprob_row_host(row.data(), V, h.empty() ? nullptr : h.data(), (int) h.size(), 1.2f, 0.1f, 0.3f, 1077,
                                     k, ref);
                    const LogprobRow got = kernel_like(row, h, pen ? 1.2f : 1.0f, pen ? 0.1f : 0.0f, pen ? 0.3f : 0.0f,
                                                       1077, k, nt);
                    assert(got.n == ref.n && got.id == ref.id);
                    assert(near(got.lp, ref.lp, 2e-5));
                    for (int j = 0; j < ref.n; ++j) {
                        assert(got.top[j] == ref.top[j]);
                        assert(near(got.top_lp[j], ref.top_lp[j], 2e-5));
                    }
                }
    }
    {   // the protocol's line: one conversation's and --parallel's (the rid after LP), read back; -inf as printf spells it
        LogprobRow r;
        r.id = 42, r.lp = -0.25f, r.n = 2;
        r.top[0] = 42, r.top_lp[0] = -0.25f;
        r.top[1] = 7, r.top_lp[1] = -1.6094379f;
        const std::string a = lp_line(-1, r), b = lp_line(12, r);
        assert(a == "LP 42 -0.25 42:-0.25 7:-1.609438");
        assert(b == "LP 12 42 -0.25 42:-0.25 7:-1.609438");
        const Parsed p = parse(b, true);
        assert(p.rid == 12 && p.id == 42 && p.lp == -0.25 && p.top.size() == 2 && p.top[1].first == 7);
        assert(near(p.top[1].second, -1.6094379, 1e-6));
        r.n = 0, r.lp = -INFINITY;
        assert(lp_line(-1, r) == "LP 42 -inf");
        r.lp = -1.0e-7f;
        assert(parse(lp_line(3, r), true).lp < 0.0 && near(parse(lp_line(3, r), true).lp, -1.0e-7, 1e-6));
        static_assert(kTopLogprobs == 20, "OpenAI's top_logprobs at most");
    }
    std::printf("ds4_logprobs_test: ok\n");
    return 0;
}
