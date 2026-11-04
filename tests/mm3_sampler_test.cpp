// tests/mm3_sampler_test.cpp - maya-mm3's sampler (src/mm3/mm3_sample.hpp) on the CPU:
//   1. a row's distribution from its candidates (host_cands with cands_for's kc and extra ids, the device's topk_rows
//      semantics) is the whole row's, bit for bit - temperature, top_k, top_p, min_p, penalties, ties;
//   2. speculative sampling with sampled drafts (keep d ~ q with min(1, p(d) / q(d)), else draw_residual) gives p.
//     g++ -O2 -std=c++17 -Isrc/mm3 -Iinclude tests/mm3_sampler_test.cpp -o build/mm3_sampler_test
#include <cstdio>
#include <random>

#include "mm3_cands.hpp"
#include "mm3_sample.hpp"

using namespace mm3::sample;

int main() {
    std::mt19937_64 rng(7);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    const int V = 5000;
    int bad = 0, cases = 0;
    for (int it = 0; it < 4000; ++it) {
        std::vector<float> row((size_t) V);
        for (float& x : row) x = nd(rng);
        for (int i = 0; i < 40; ++i) row[(size_t) (rng() % V)] = row[(size_t) (rng() % V)];   // (ties)
        if (it % 7 == 0) row[(size_t) (rng() % V)] = 0.0f, row[(size_t) (rng() % V)] = -0.0f;
        if (it % 11 == 0) {   // (many ties at the top)
            const float top = *std::max_element(row.begin(), row.end());
            for (int i = 0; i < 30; ++i) row[(size_t) (rng() % V)] = top;
        }
        Req r;
        const float temps[] = {0.0f, 0.6f, 1.0f, 1.3f};
        const int ks[] = {1, 5, 40, 64};
        const float ps[] = {1.0f, 0.95f, 0.8f};
        r.temperature = temps[rng() % 4];
        r.top_k = ks[rng() % 4];
        r.top_p = ps[rng() % 3];
        r.min_p = rng() % 3 == 0 ? 0.05f : 0.0f;
        if (rng() % 2) {
            r.pen_n = rng() % 2 ? 8 : 64;
            const float rep[] = {1.0f, 1.1f, 0.9f}, fr[] = {0.0f, 0.2f}, pr[] = {0.0f, 0.4f, -0.3f};
            r.pen_repeat = rep[rng() % 3];
            r.pen_freq = fr[rng() % 2];
            r.pen_present = pr[rng() % 3];
        }
        // a history (some of the row's best ids in it) and the window's tokens after it
        std::vector<int> order((size_t) V);
        for (int i = 0; i < V; ++i) order[(size_t) i] = i;
        std::partial_sort(order.begin(), order.begin() + 100, order.end(),
                          [&](int a, int b) { return row[(size_t) a] > row[(size_t) b]; });
        std::vector<int> hist, more;
        for (int i = 0; i < 300; ++i) hist.push_back(rng() % 3 ? order[(size_t) (rng() % 100)] : (int) (rng() % V));
        for (int i = 0; i < (int) (rng() % 8); ++i) more.push_back(order[(size_t) (rng() % 100)]);
        std::vector<int> full_hist = hist;
        full_hist.insert(full_hist.end(), more.begin(), more.end());
        mm3::Cands c;
        if (!cands_for(r, hist, more, c)) continue;
        std::vector<std::pair<int, float>> top;
        std::vector<float> ex;
        mm3::host_cands(row.data(), V, c.kc, c.extra, top, ex);
        // every row of the window: its history the window's tokens before it
        for (size_t w = 0; w <= more.size(); ++w) {
            std::vector<int> h(hist);
            h.insert(h.end(), more.begin(), more.begin() + (long) w);
            const auto a = dist(row.data(), V, r, h);
            const auto b = dist_c(top, c.extra, ex, r, h);
            ++cases;
            if (a != b) {
                if (++bad <= 5)
                    std::printf("MISMATCH case %d: T %.1f top_k %d top_p %.2f min_p %.2f pen %d/%.1f/%.1f/%.1f: %zu vs %zu "
                                "entries, first %d / %d\n", it, r.temperature, r.top_k, r.top_p, r.min_p, r.pen_n,
                                r.pen_repeat, r.pen_freq, r.pen_present, a.size(), b.size(), a[0].first, b[0].first);
            }
        }
    }
    std::printf("candidates vs the whole row: %d of %d distributions differ\n", bad, cases);

    // speculative sampling with a sampled draft: the output's distribution against p (total variation)
    double worst = 0;
    for (int trial = 0; trial < 6; ++trial) {
        std::vector<float> rp((size_t) V), rq((size_t) V);
        for (int i = 0; i < V; ++i) {
            rp[(size_t) i] = nd(rng);
            rq[(size_t) i] = rp[(size_t) i] + 0.8f * nd(rng);   // (a draft that is close, not equal)
        }
        Req r;
        r.temperature = trial % 2 ? 1.0f : 0.7f;
        r.top_k = 64;
        r.top_p = trial < 3 ? 0.95f : 1.0f;
        const auto p = dist(rp.data(), V, r, {});
        const auto qd = dist(rq.data(), V, r, {});
        std::vector<std::pair<int, float>> q;
        for (const auto& [t, x] : qd) q.push_back({t, (float) x});
        std::unordered_map<int, long> cnt;
        const long N = 2000000;
        long kept = 0;
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (long n = 0; n < N; ++n) {
            // x ~ q (the device's draw: the first whose running sum passes u)
            const double u = U(rng);
            double c2 = 0;
            int x = q.back().first;
            for (const auto& [t, qt] : q) {
                c2 += qt;
                if (u < c2) {
                    x = t;
                    break;
                }
            }
            double px = 0, qx = 1;
            for (const auto& [t, pt] : p)
                if (t == x) px = pt;
            for (const auto& [t, qt] : q)
                if (t == x) qx = qt;
            int out;
            if (U(rng) * qx < px) {
                out = x;
                ++kept;
            } else {
                out = draw_residual(p, q, rng);
            }
            ++cnt[out];
        }
        double tv = 0;
        for (const auto& [t, pt] : p) tv += std::fabs(pt - (double) cnt[t] / N);
        for (const auto& [t, n] : cnt)
            if (std::none_of(p.begin(), p.end(), [&](const auto& e) { return e.first == t; })) tv += (double) n / N;
        tv /= 2;
        worst = std::max(worst, tv);
        std::printf("speculative sampling trial %d: |p| %zu |q| %zu, kept %.3f, total variation to p %.5f\n", trial,
                    p.size(), q.size(), (double) kept / N, tv);
    }
    const bool ok = bad == 0 && worst < 0.004;
    std::printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
