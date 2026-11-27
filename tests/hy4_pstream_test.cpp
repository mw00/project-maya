// maya-hy4's HY4_PSTREAM bookkeeping (src/hy4/hy4_pstream.hpp; no GPU):
//  - ps_plan: every expert outside VRAM in exactly one pass, no pass larger than the buffer, the last pass not empty
//  - ps_may_copy: a run of passes as the engine queues them (copies while it may, then a pass's kernels) - a copy never
//    goes into a buffer whose pass has not been run, no copy is queued twice, and every pass is copied before it runs
//  - ps_swap: random HY4_HOT swaps - each pinned slot keeps one expert outside VRAM, each of those one slot
#include "../src/hy4/hy4_pstream.hpp"

#include <cstdio>
#include <random>
#include <set>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    using namespace hy4;
    // the passes
    for (int cap : {1, 2, 7, 96, 128}) {
        for (int m = 0; m <= 700; ++m) {
            const PsPlan p = ps_plan(m, cap);
            if (m == 0) {
                CHECK(p.passes == 0);
                continue;
            }
            CHECK(p.passes == (m + cap - 1) / cap);
            std::vector<char> seen((size_t) m, 0);
            for (int i = 0; i < p.passes; ++i) {
                const int n = p.count(m, i);
                CHECK(n >= 1 && n <= cap && n <= p.per);
                for (int k = 0; k < n; ++k) {
                    CHECK(i * p.per + k < m && !seen[(size_t) (i * p.per + k)]);
                    seen[(size_t) (i * p.per + k)] = 1;
                }
            }
            for (char c : seen) CHECK(c);
        }
    }
    // the order
    std::mt19937 rng(7);
    for (int t = 0; t < 2000; ++t) {
        const size_t total = 1 + rng() % 40;
        size_t copied = 0, run = 0;
        std::vector<char> in_buf(2, 0);   // (the pass whose data buffer b holds and has not run: 1 + pass, 0: free)
        std::vector<size_t> holds(2, 0);
        while (run < total) {
            while (copied < total && ps_may_copy(copied, run)) {
                const size_t b = copied & 1;
                // the buffer's earlier pass (copied - 2) must have been run
                CHECK(copied < 2 || copied - 2 < run);
                CHECK(!(in_buf[b] && holds[b] >= run));   // (no unrun pass overwritten)
                in_buf[b] = 1;
                holds[b] = copied++;
            }
            CHECK(run < copied && copied - run <= 2);   // (at most two copies ahead; this pass's is queued)
            CHECK(holds[run & 1] == run);
            ++run;
        }
        CHECK(copied == total);
    }
    // the swaps
    for (int t = 0; t < 200; ++t) {
        const int n = 60 + (int) (rng() % 40), vram = n / 2 + (int) (rng() % 5);
        std::vector<int> of((size_t) n, -1), ex;
        std::set<int> lane;
        for (int e = vram; e < n; ++e) {
            of[(size_t) e] = (int) ex.size();
            ex.push_back(e);
            lane.insert(e);
        }
        std::set<int> res;
        for (int e = 0; e < vram; ++e) res.insert(e);
        for (int s = 0; s < 400; ++s) {
            auto it = lane.begin();
            std::advance(it, rng() % lane.size());
            const int in = *it;
            auto jt = res.begin();
            std::advance(jt, rng() % res.size());
            const int out = *jt;
            const int q = ps_swap(of, ex, out, in);
            CHECK(q >= 0 && q == of[(size_t) out]);
            CHECK(ps_swap(of, ex, out, in) == -1);   // (`in` is in VRAM now: nothing)
            lane.erase(in);
            lane.insert(out);
            res.erase(out);
            res.insert(in);
            std::set<int> seen;
            for (int q = 0; q < (int) ex.size(); ++q) {
                CHECK(of[(size_t) ex[(size_t) q]] == q && lane.count(ex[(size_t) q]) && seen.insert(ex[(size_t) q]).second);
            }
            for (int e : res) CHECK(of[(size_t) e] < 0);
        }
    }
    std::printf("hy4_pstream_test: ok\n");
    return 0;
}
