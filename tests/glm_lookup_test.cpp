// tests/glm_lookup_test.cpp - MAYA_GLM_LOOKUP (include/maya/core/glm_lookup.hpp): the lookup's history with
// pictures, the index kept across requests, the choice between the drafter's block and the lookup's run, and
// decode_dflash's round rule replayed on the host (a draft stands only when it is the token the model picks): the
// same tokens with and without the lookup, fewer rounds where the answer repeats the prompt.  No GPU.
//   g++ -std=c++17 -O2 -Iinclude tests/glm_lookup_test.cpp src/spec/suffix_drafter.cpp -o glm_lookup_test
#include "maya/core/glm_lookup.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using maya::core::GlmLookup;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}
uint64_t g_x = 88172645463325252ull;
int32_t rnd(int mod) {
    g_x ^= g_x << 13, g_x ^= g_x >> 7, g_x ^= g_x << 17;
    return (int32_t) (g_x % (uint64_t) mod);
}
bool is(const GlmLookup::Choice& c, GlmLookup::Source s, int n, int nd, const int32_t* out,
        const std::vector<int32_t>& want) {
    return c.src == s && c.n == n && c.n_drafter == nd && (int) want.size() == n &&
           std::memcmp(out, want.data(), (size_t) n * 4) == 0;
}

void test_choose() {
    int32_t out[32];
    const int32_t d[] = {1, 2, 3, 4}, l[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const int32_t x[] = {1, 9, 3, 4}, z[] = {7, 2, 3};
    const int trust = 16;
    // no run: the block (an empty one too)
    check(is(GlmLookup::choose(d, 4, l, 0, 0, trust, out), GlmLookup::kDrafter, 4, 4, out, {1, 2, 3, 4}), "no run");
    check(is(GlmLookup::choose(d, 0, l, 0, 0, trust, out), GlmLookup::kDrafter, 0, 0, out, {}), "no run, no block");
    // a short match: the block is the run's start -> the run goes on past it by `match` drafts at most
    check(is(GlmLookup::choose(d, 4, l, 12, 3, trust, out), GlmLookup::kExtend, 7, 4, out, {1, 2, 3, 4, 5, 6, 7}),
          "short match extends the block by the match length");
    check(is(GlmLookup::choose(d, 4, l, 12, 10, trust, out), GlmLookup::kExtend, 12, 4, out,
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
          "short match extends to the run's end");
    check(is(GlmLookup::choose(d, 0, l, 12, 5, trust, out), GlmLookup::kExtend, 5, 0, out, {1, 2, 3, 4, 5}),
          "no block: the run's first `match` drafts");
    // a short match, the two differ: the block
    check(is(GlmLookup::choose(x, 4, l, 12, 8, trust, out), GlmLookup::kDrafter, 4, 4, out, {1, 9, 3, 4}),
          "short match, different second draft: the block");
    check(is(GlmLookup::choose(z, 3, l, 12, 15, trust, out), GlmLookup::kDrafter, 3, 3, out, {7, 2, 3}),
          "short match, different first draft: the block");
    // a trusted match: the run whatever the block says
    check(is(GlmLookup::choose(z, 3, l, 12, 16, trust, out), GlmLookup::kLookup, 12, 0, out,
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
          "trusted match over a different block");
    check(is(GlmLookup::choose(d, 4, l, 12, 40, trust, out), GlmLookup::kLookup, 12, 0, out,
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
          "trusted match over an agreeing block");
    check(is(GlmLookup::choose(d, 0, l, 12, 40, trust, out), GlmLookup::kLookup, 12, 0, out,
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
          "trusted match, no block");
    // the block holds the whole run (and more): the block, at any match
    check(is(GlmLookup::choose(d, 4, l, 2, 40, trust, out), GlmLookup::kDrafter, 4, 4, out, {1, 2, 3, 4}),
          "the block holds the run (trusted)");
    check(is(GlmLookup::choose(d, 4, l, 4, 3, trust, out), GlmLookup::kDrafter, 4, 4, out, {1, 2, 3, 4}),
          "the block is the run");
    // a trusted run shorter than a block that differs: the run
    check(is(GlmLookup::choose(x, 4, l, 3, 20, trust, out), GlmLookup::kLookup, 3, 0, out, {1, 2, 3}),
          "trusted short run over a different block");
    // the choice reads nothing else: the same inputs, the same answer
    const GlmLookup::Choice a = GlmLookup::choose(d, 4, l, 12, 3, trust, out);
    const GlmLookup::Choice b = GlmLookup::choose(d, 4, l, 12, 3, trust, out);
    check(a.src == b.src && a.n == b.n && a.n_drafter == b.n_drafter, "choose is a function of its inputs");
}

void test_history() {
    // a prompt with two pictures (runs of the placeholder id 900 at the listed positions) and the same text around
    std::vector<int32_t> p = {5, 6, 7, 900, 900, 900, 8, 9, 10, 11, 900, 900, 8, 9, 10, 11, 12, 13};
    std::vector<int64_t> img = {3, 4, 5, 10, 11};
    std::vector<int32_t> h;
    GlmLookup::history(p, img, h);
    const std::vector<int32_t> want = {5, 6, 7, -1, 8, 9, 10, 11, -2, 8, 9, 10, 11, 12, 13};
    check(h == want, "history: a mark a run of image positions, a different one each run");
    GlmLookup::history(p, {}, h);
    check(h == p, "history: no pictures, the prompt");
    // two pictures side by side with no text between are one run; a picture at the start and at the end
    std::vector<int32_t> q = {900, 900, 1, 2, 3, 900};
    GlmLookup::history(q, {0, 1, 5}, h);
    check(h == std::vector<int32_t>({-1, 1, 2, 3, -2}), "history: pictures at the ends");

    // a run ends at a picture's mark: history ... A B C D [img] E ... A B C D -> D's continuation is the mark: nothing
    GlmLookup lk(1, 3, 16, 1 << 20, 4096);
    std::vector<int32_t> t = {1, 2, 3, 4, 5, 900, 900, 6, 7, 8, 9, 2, 3, 4};
    GlmLookup::history(t, {5, 6}, h);
    lk.begin(h);
    int32_t out[16];
    int match = 0;
    int n = lk.propose(15, out, match);
    check(n == 1 && out[0] == 5 && match == 3, "propose: the run stops before a picture's mark");
    lk.wrote(5);
    n = lk.propose(15, out, match);
    check(n == 0 && match == 0, "propose: nothing where the continuation is a picture");
    // the text after two different pictures does not match through them: [img1] 6 7 8 / [img2] 6 7 8 matches on the
    // three tokens only (the marks differ)
    std::vector<int32_t> u = {40, 41, 900, 6, 7, 8, 50, 51, 52, 53, 900, 6, 7, 8};
    GlmLookup::history(u, {2, 10}, h);
    lk.begin(h);
    n = lk.propose(15, out, match);
    check(n > 0 && out[0] == 50 && match == 3, "propose: a match ends at the pictures' different marks");
    // MAYA_GLM_LOOKUP_MAX caps a run
    GlmLookup cap(1, 3, 16, 2, 4096);
    std::vector<int32_t> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3};
    cap.begin(v);
    check(cap.propose(15, out, match) == 2 && out[0] == 4 && out[1] == 5, "propose: the cap");
    check(cap.propose(0, out, match) == 0, "propose: no room");
}

void test_keep() {
    // a conversation: request 2's prompt is request 1's prompt + its answer + a new turn -> the index is kept and is
    // the one a fresh build of the new prompt gives; a prompt that differs inside is rebuilt
    GlmLookup lk(1, 3, 16, 1 << 20, 1 << 14);
    std::vector<int32_t> p1;
    for (int i = 0; i < 3000; ++i) p1.push_back(rnd(60));
    lk.begin(p1);
    check(lk.kept_index == 0 && lk.index_at_start == p1.size(), "begin: the first prompt built");
    std::vector<int32_t> p2 = p1;
    for (int i = 0; i < 200; ++i) {   // the answer, a token at a time, proposals between
        int32_t out[16];
        int match;
        lk.propose(15, out, match);
        const int32_t t = rnd(60);
        lk.wrote(t);
        p2.push_back(t);
    }
    for (int i = 0; i < 400; ++i) p2.push_back(rnd(60));   // the next turn
    lk.begin(p2);
    GlmLookup fresh(1, 3, 16, 1 << 20, 1 << 14);
    fresh.begin(p2);
    check(lk.kept_index == p1.size() + 200 && lk.sfx.same(fresh.sfx), "begin: an extended prompt keeps the index");
    std::vector<int32_t> p3 = p2;
    p3[p3.size() - 100] += 1;   // (the answer came back in other ids)
    lk.begin(p3);
    GlmLookup fresh3(1, 3, 16, 1 << 20, 1 << 14);
    fresh3.begin(p3);
    check(lk.kept_index == 0 && lk.sfx.same(fresh3.sfx), "begin: a prompt that differs is rebuilt");
    check(lk.proposals == 0 && lk.rounds[0] + lk.rounds[1] + lk.rounds[2] == 0, "begin: the request's counts cleared");
}

// decode_dflash's rounds on the host: `truth` is what the model picks after each context (a fixed continuation, as
// a greedy decode's or one seed's draws); the drafter's block is right for a geometric run (p a draft) and wrong
// after; a round keeps the longest prefix of its drafts equal to the picks, then the pick after it.
struct Sim {
    std::vector<int32_t> written;
    long long rounds = 0, drafter_starts = 0;
};
Sim decode(const std::vector<int32_t>& prompt, const std::vector<int32_t>& truth, GlmLookup* lk, int block, int p_pct,
           uint64_t seed) {
    Sim s;
    uint64_t x = seed;
    const auto r100 = [&] {
        x ^= x << 13, x ^= x >> 7, x ^= x << 17;
        return (int) (x % 100);
    };
    const int cap = 15;   // kPipeMax - 1
    int32_t lbuf[16], win[16];
    int ln = 0, lmatch = 0;
    bool alone = false;
    size_t at = 0;   // truth[at]: the next pick
    if (lk != nullptr) lk->begin(prompt);
    const auto wrote = [&](int32_t t) {
        s.written.push_back(t);
        if (lk != nullptr) lk->wrote(t);
    };
    const auto next = [&] {
        ln = lmatch = 0;
        alone = false;
        if (lk == nullptr) return;
        ln = lk->propose(cap, lbuf, lmatch);
        alone = lk->alone(ln, lmatch);
    };
    wrote(truth[at++]);
    next();
    while (at < truth.size()) {
        std::vector<int32_t> d;
        if (!alone) {
            ++s.drafter_starts;
            for (int i = 0; i < block - 1 && at + (size_t) i < truth.size(); ++i) {
                const bool right = r100() < p_pct;
                d.push_back(right ? truth[at + (size_t) i] : -7);
                if (!right) break;   // (drafts after a wrong one: wrong too - cut here as the confidence cut would)
            }
        } else {
            for (int i = 0; i < block - 1; ++i) r100();   // (the same draws either way: the drafter's later blocks alike)
        }
        GlmLookup::Choice c;
        c.n = c.n_drafter = (int) d.size();
        const int first = ln > 0 ? lbuf[0] : -1;
        if (ln > 0) {
            c = GlmLookup::choose(d.data(), (int) d.size(), lbuf, ln, lmatch, lk->trust, win);
            d.assign(win, win + c.n);
        }
        int acc = 0;
        while (acc < (int) d.size() && at + (size_t) acc < truth.size() && d[(size_t) acc] == truth[at + (size_t) acc]) ++acc;
        if (lk != nullptr) {
            if (first >= 0) lk->score_first(first == truth[at]);
            lk->observe(c, acc, alone);
        }
        ++s.rounds;
        for (int i = 0; i < acc; ++i) wrote(truth[at++]);
        if (at >= truth.size()) break;
        wrote(truth[at++]);   // the pick after the kept drafts: the next anchor
        next();
    }
    return s;
}

void test_rounds() {
    // an edit: the prompt holds a 600-token file; the answer writes it again with every 97th token changed
    std::vector<int32_t> prompt, file, truth;
    for (int i = 0; i < 300; ++i) prompt.push_back(rnd(5000));
    for (int i = 0; i < 600; ++i) file.push_back(5000 + rnd(5000));
    prompt.insert(prompt.end(), file.begin(), file.end());
    for (int i = 0; i < 40; ++i) prompt.push_back(rnd(5000));
    for (int i = 0; i < 30; ++i) truth.push_back(10000 + rnd(5000));   // the answer's own start
    for (int i = 0; i < 600; ++i) truth.push_back(i % 97 == 96 ? 20000 + i : file[(size_t) i]);
    for (int i = 0; i < 30; ++i) truth.push_back(10000 + rnd(5000));
    for (const int mode : {1, 2}) {
        GlmLookup lk(mode, 3, 16, 1 << 20, 1 << 14);
        const Sim base = decode(prompt, truth, nullptr, 8, 55, 99);
        const Sim with = decode(prompt, truth, &lk, 8, 55, 99);
        check(base.written == truth && with.written == truth, "rounds: the same tokens with and without the lookup");
        check(with.rounds * 2 < base.rounds, "rounds: an edit takes under half the rounds with the lookup");
        check(lk.kept_lookup > 400 && lk.rounds[GlmLookup::kLookup] > 0, "rounds: the lookup's drafts kept");
        check(lk.off_drafter >= lk.kept_drafter && lk.off_lookup >= lk.kept_lookup &&
              (lk.kept_drafter + lk.kept_lookup + with.rounds + 1 == (long long) truth.size() ||   // (the last round
               lk.kept_drafter + lk.kept_lookup + with.rounds == (long long) truth.size()),         //  may end on a draft)
              "rounds: every token is the first, a kept draft or a round's pick");
        check(mode == 2 ? lk.alone_rounds > 0 && with.drafter_starts < with.rounds
                        : lk.alone_rounds == 0 && with.drafter_starts == with.rounds,
              "rounds: the drafter is skipped only by MAYA_GLM_LOOKUP=2's trusted runs");
        std::printf("edit, MAYA_GLM_LOOKUP=%d: %lld rounds without, %lld with (%.2f -> %.2f tokens a round), drafter "
                    "started %lld times\n  ", mode, base.rounds, with.rounds, (double) truth.size() / base.rounds,
                    (double) truth.size() / with.rounds, with.drafter_starts);
        lk.report(stdout);
    }
    {   // text that repeats nothing (every token new): no proposal, the drafter's rounds as without the lookup
        std::vector<int32_t> p, t;
        for (int i = 0; i < 2000; ++i) p.push_back(i);
        for (int i = 0; i < 500; ++i) t.push_back(100000 + i);
        GlmLookup lk(2, 3, 16, 1 << 20, 1 << 14);
        const Sim base = decode(p, t, nullptr, 8, 55, 7), with = decode(p, t, &lk, 8, 55, 7);
        check(with.written == t && with.rounds == base.rounds && lk.proposals == 0 && lk.off_lookup == 0,
              "rounds: nothing to look up, the same rounds");
    }
    {   // text with short accidental repeats (a 40-token vocabulary: 3-token matches whose continuation is wrong):
        // the same tokens; the drafter's kept drafts are not displaced (a short match only goes on past its block)
        std::vector<int32_t> p, t;
        for (int i = 0; i < 4000; ++i) p.push_back(rnd(40));
        for (int i = 0; i < 2000; ++i) t.push_back(rnd(40));
        GlmLookup lk(2, 3, 16, 1 << 20, 1 << 14);
        const Sim base = decode(p, t, nullptr, 8, 55, 7), with = decode(p, t, &lk, 8, 55, 7);
        check(with.written == t, "noise: the same tokens");
        check(with.rounds <= base.rounds, "noise: never more rounds than without the lookup");
        check(lk.rounds[GlmLookup::kLookup] == 0 && lk.alone_rounds == 0, "noise: no trusted run");
        std::printf("noise: %lld rounds without, %lld with\n  ", base.rounds, with.rounds);
        lk.report(stdout);
    }
    {   // a trusted run that stops holding: MAYA_GLM_LOOKUP=2 stops skipping the drafter (alone() reads the score)
        GlmLookup lk(2, 3, 16, 1 << 20, 4096);
        for (int i = 0; i < 20; ++i) lk.score_first(true);
        check(lk.alone(5, 20) && !lk.alone(5, 15) && !lk.alone(0, 20), "alone: a trusted match once its drafts hold");
        for (int i = 0; i < 6; ++i) lk.score_first(false);
        check(!lk.alone(5, 20), "alone: not after its first drafts failed");
        GlmLookup one(1, 3, 16, 1 << 20, 4096);
        for (int i = 0; i < 20; ++i) one.score_first(true);
        check(!one.alone(5, 20), "alone: never with MAYA_GLM_LOOKUP=1");
    }
}
}  // namespace

int main() {
    test_choose();
    test_history();
    test_keep();
    test_rounds();
    std::printf("glm_lookup tests: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
