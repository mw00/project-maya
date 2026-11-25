// src/spec/suffix_drafter_test.cpp - plan v0.3 P6: suffix-lookup drafter tests and an offline simulation.
//
//   suffix_drafter_test                      unit tests
//   suffix_drafter_test --simulate [--k K] FILE.ids ...
//
// The simulation replays text as if a model had produced it: each step drafts up to K tokens, accepts the
// prefix that matches the true next tokens, and commits that prefix plus one model token (the verify pass's
// bonus). tokens/step is the lookup drafter's ceiling speedup on that text when a K+1-token verify costs the same
// as a 1-token step. Two regimes per file:
//   continue  history = first half of the prompt; generate its second half (natural continuation)
//   copy      history = the whole prompt; generate its middle third again (output quoting input: edits)
// No model runs; this measures the drafter on text, not on the model's own outputs.
#include "maya/spec/suffix_drafter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using maya::spec::SuffixDrafter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

std::vector<int32_t> read_ids(const char* path) {
    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream in(text);
    std::vector<int32_t> ids;
    for (int32_t v; in >> v;) ids.push_back(v);
    return ids;
}

struct Sim { long steps = 0, tokens = 0, proposals = 0, accepted = 0; };

Sim simulate(const std::vector<int32_t>& history, const std::vector<int32_t>& target, int K) {
    SuffixDrafter d(3, 32, history.size() + target.size() + 16);
    d.append(history.data(), history.size());
    Sim s;
    std::vector<int32_t> draft(K);
    size_t i = 0;
    while (i < target.size()) {
        const int n = d.propose(K, draft.data());
        int a = 0;
        while (a < n && i + a < target.size() && draft[a] == target[i + a]) ++a;
        if (n > 0) { ++s.proposals; s.accepted += a; }
        const size_t commit = std::min(target.size() - i, (size_t) a + 1);
        d.append(&target[i], commit);
        i += commit;
        s.tokens += (long) commit;
        ++s.steps;
    }
    return s;
}

void unit_tests() {
    {   // a repeated passage is proposed verbatim
        SuffixDrafter d;
        const std::vector<int32_t> doc = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
        d.append(doc.data(), doc.size());
        const int32_t again[] = {10, 11, 12};
        d.append(again, 3);
        int32_t out[8];
        const int n = d.propose(5, out);
        check(n == 5 && out[0] == 13 && out[4] == 17, "repeat proposes the continuation");
        check(d.last_match() == 3, "match length is the shared suffix");
    }
    {   // no earlier occurrence: nothing proposed
        SuffixDrafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 4, 5, 6, 7};
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) == 0, "no match proposes nothing");
    }
    {   // two occurrences of the trigram: the longer match wins even though it is older
        SuffixDrafter d;
        const std::vector<int32_t> doc = {7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        const int n = d.propose(2, out);
        check(n == 2 && out[0] == 100 && out[1] == 101, "longest match preferred over most recent");
        check(d.last_match() == 5, "longest match length 5");
    }
    {   // a match shorter than min_match is ignored (only a bigram in common)
        SuffixDrafter d(4);
        const std::vector<int32_t> doc = {1, 2, 3, 9, 5, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        check(d.propose(2, out) == 0, "min_match respected");
    }
    {   // periodic text: the continuation may run into the current suffix
        SuffixDrafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 1, 2, 3, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[6];
        const int n = d.propose(6, out);
        check(n >= 3 && out[0] == 1 && out[1] == 2 && out[2] == 3, "periodic continuation");
    }
    {   // bounded memory: appending beyond the nominal capacity does not crash and still works
        SuffixDrafter d(3, 32, 1024);
        std::vector<int32_t> doc;
        for (int i = 0; i < 5000; ++i) doc.push_back(i % 997);
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) > 0, "propose after overflow of nominal capacity");
    }
    {   // restart (DS4_LOOKUP_KEEP): the kept index is the one a reset and an append of the whole history build -
        // a history that extends the old one (kept), one that differs inside it, a shorter one, the same one, an
        // empty one; past the nominal capacity too (a table that filled)
        uint64_t x = 88172645463325252ull;
        auto rnd = [&](int mod) {
            x ^= x << 13, x ^= x >> 7, x ^= x << 17;
            return (int32_t) (x % (uint64_t) mod);
        };
        for (const size_t cap : {size_t(1) << 14, size_t(300)}) {
            SuffixDrafter kept(3, 64, cap);
            std::vector<int32_t> h;
            for (int i = 0; i < 4000; ++i) h.push_back(rnd(50));   // (a small vocabulary: slots with several ways)
            auto rebuilt_same = [&](const std::vector<int32_t>& t) {
                SuffixDrafter fresh(3, 64, cap);
                fresh.reset();
                fresh.append(t.data(), t.size());
                return kept.same(fresh);
            };
            check(kept.restart(h.data(), h.size()) == 0 && rebuilt_same(h), "restart: first history built");
            for (int i = 0; i < 37; ++i) {   // (a decode's tokens, one at a time, with proposals between)
                int32_t out[8];
                kept.propose(8, out);
                h.push_back(rnd(50));
                kept.append(h.back());
            }
            std::vector<int32_t> t = h;   // append: the next turn
            for (int i = 0; i < 500; ++i) t.push_back(rnd(50));
            check(kept.restart(t.data(), t.size()) == h.size() && rebuilt_same(t), "restart: extended history kept, same index");
            check(kept.restart(t.data(), t.size()) == t.size() && rebuilt_same(t), "restart: the same history kept, same index");
            std::vector<int32_t> d = t;   // diverge: one token inside differs
            d[d.size() - 300] += 1;
            check(kept.restart(d.data(), d.size()) == 0 && rebuilt_same(d), "restart: a history that differs is rebuilt");
            std::vector<int32_t> sh(d.begin(), d.begin() + 1000);   // shorter: a prefix of the old one
            check(kept.restart(sh.data(), sh.size()) == 0 && rebuilt_same(sh), "restart: a shorter history is rebuilt");
            std::vector<int32_t> dl = sh;   // diverge at the last token of the old history, and longer
            dl.back() += 1;
            for (int i = 0; i < 100; ++i) dl.push_back(rnd(50));
            check(kept.restart(dl.data(), dl.size()) == 0 && rebuilt_same(dl), "restart: a different last token is rebuilt");
            check(kept.restart(nullptr, 0) == 0 && rebuilt_same({}), "restart: an empty history");
            check(kept.restart(sh.data(), sh.size()) == 0 && rebuilt_same(sh), "restart: after an empty history");
            int32_t o1[8], o2[8];   // (and the proposals agree)
            SuffixDrafter fresh(3, 64, cap);
            fresh.append(sh.data(), sh.size());
            const int n1 = kept.propose(8, o1), n2 = fresh.propose(8, o2);
            check(n1 == n2 && std::memcmp(o1, o2, (size_t) n1 * 4) == 0 && kept.last_match() == fresh.last_match(),
                  "restart: the same proposal");
        }
    }
    std::printf("suffix_drafter unit tests: %s\n", g_fail ? "FAILED" : "OK");
}

// --bench: a request's Lookup::start as the DeepSeek serve loop runs it (min match 3, max 64, the table sized for a
// 1M context + 4096): reset + append of the whole context against restart() of a context that grew by a turn
void bench() {
    uint64_t x = 88172645463325252ull;
    auto rnd = [&]() {
        x ^= x << 13, x ^= x >> 7, x ^= x << 17;
        return (int32_t) (x % 150000u);
    };
    for (const size_t n : {size_t(30000), size_t(150000), size_t(350000)}) {
        std::vector<int32_t> ctx;
        for (size_t i = 0; i < n; ++i) ctx.push_back(rnd());
        SuffixDrafter a(3, 64, 1052672), b(3, 64, 1052672);
        a.append(ctx.data(), ctx.size());
        b.restart(ctx.data(), ctx.size());
        double t_full = 1e30, t_keep = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            for (int i = 0; i < 300; ++i) ctx.push_back(rnd());   // (a turn: 300 tokens more)
            auto t0 = std::chrono::steady_clock::now();
            a.reset();
            a.append(ctx.data(), ctx.size());
            auto t1 = std::chrono::steady_clock::now();
            b.restart(ctx.data(), ctx.size());
            auto t2 = std::chrono::steady_clock::now();
            t_full = std::min(t_full, std::chrono::duration<double, std::milli>(t1 - t0).count());
            t_keep = std::min(t_keep, std::chrono::duration<double, std::milli>(t2 - t1).count());
            check(a.same(b), "bench: kept index equals the rebuilt one");
        }
        std::printf("context %7zu tokens + 300: reset + append %8.2f ms, restart %7.3f ms (same index: %s)\n", n, t_full,
                    t_keep, g_fail ? "NO" : "yes");
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--simulate") == 0) {
        int K = 8, first = 2;
        if (argc > 3 && std::strcmp(argv[2], "--k") == 0) { K = std::atoi(argv[3]); first = 4; }
        std::printf("%-28s %8s | %-44s | %-44s\n", "prompt", "tokens", "continue: tok/step  proposals  acc/proposal",
                    "copy: tok/step  proposals  acc/proposal");
        for (int f = first; f < argc; ++f) {
            const std::vector<int32_t> ids = read_ids(argv[f]);
            if (ids.size() < 64) continue;
            const size_t half = ids.size() / 2;
            const Sim c = simulate({ids.begin(), ids.begin() + half}, {ids.begin() + half, ids.end()}, K);
            const size_t a = ids.size() / 3, b = 2 * ids.size() / 3;
            const Sim p = simulate(ids, {ids.begin() + a, ids.begin() + b}, K);
            const char* name = std::strrchr(argv[f], '/') ? std::strrchr(argv[f], '/') + 1 : argv[f];
            std::printf("%-28s %8zu | %8.2f %10.1f%% %10.2f              | %8.2f %10.1f%% %10.2f\n", name, ids.size(),
                        (double) c.tokens / c.steps, 100.0 * c.proposals / c.steps,
                        c.proposals ? (double) c.accepted / c.proposals : 0.0, (double) p.tokens / p.steps,
                        100.0 * p.proposals / p.steps, p.proposals ? (double) p.accepted / p.proposals : 0.0);
        }
        return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        bench();
        return g_fail ? 1 : 0;
    }
    unit_tests();
    return g_fail ? 1 : 0;
}
