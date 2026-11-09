// tests/glm_dsa/glm_draft_adaptive_test.cpp - glm-dsa B15.9 (CPU): the adaptive draft depth controller
// (include/strata/core/glm_draft_adaptive.hpp) and its use in NgramSession (STRATA_GLM_DRAFT_ADAPTIVE).
//   1. AdaptiveDepth: the climb thresholds (2, 4, the 3 -> 4 barrier of 10), the drop pressure max(5d, 20), floor and
//      cap, no pressure at the floor, reset;
//   2. ProposalTracker: full accept, first-miss resolution, one round at a time;
//   3. rounds over a code-like stream (a repeated block: the n-gram continuation is exact) and a prose-like one
//      (random tokens with short accidental repeats), adaptive vs fixed depths 1 / 4 / 8: the adaptive depth reaches
//      the cap on code (>= 90% of fixed-8's accepted tokens) and stays shallow on prose (far fewer rejected verify
//      rows than fixed-8); the numbers are printed;
//   4. NgramSession: off -> depth() is STRATA_GLM_NGRAM_MAX and nothing is tracked; on -> the proposals are cut to
//      depth(), the depth climbs on a repetitive conversation, a floor of 2 starts at 2, and it resets per session.
#include "strata/core/glm_draft_adaptive.hpp"
#include "strata/core/glm_ngram_draft.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

using namespace strata::core::glmng;

void test_controller() {
    std::printf("[1. AdaptiveDepth]\n");
    AdaptiveDepth a;
    a.reset(8, 1);
    check(a.depth() == 1, "starts at the floor");
    a.update(1, 1, 8, 1);
    check(a.depth() == 1, "one full accept at depth 1 is not enough");
    a.update(1, 1, 8, 1);
    check(a.depth() == 2, "two climb to 2");
    for (int i = 0; i < 4; ++i) a.update(2, 2, 8, 1);
    check(a.depth() == 3, "four more climb to 3");
    for (int i = 0; i < 9; ++i) a.update(3, 3, 8, 1);
    check(a.depth() == 3, "nine full accepts at 3 do not pass the barrier");
    a.update(3, 2, 8, 1);   // a miss resets the streak
    for (int i = 0; i < 9; ++i) a.update(3, 3, 8, 1);
    check(a.depth() == 3, "a miss restarts the streak");
    a.update(3, 3, 8, 1);
    check(a.depth() == 4, "the tenth consecutive full accept climbs to 4");
    for (int i = 0; i < 4; ++i) a.update(4, 0, 8, 1);
    check(a.depth() == 4, "pressure 16 < 20 keeps depth 4");
    a.update(4, 0, 8, 1);
    check(a.depth() == 3, "pressure 20 drops to 3");
    AdaptiveDepth b;
    b.reset(2, 1);
    for (int i = 0; i < 50; ++i) b.update(b.depth(), b.depth(), 2, 1);
    check(b.depth() == 2, "never above the cap");
    AdaptiveDepth c;
    c.reset(8, 3);
    for (int i = 0; i < 200; ++i) c.update(3, 0, 8, 3);
    check(c.depth() == 3 && c.n_drop == 0, "never below the floor, no pressure there");
    c.reset(8, 3);
    check(c.depth() == 3 && c.n_climb == 0 && c.n_drop == 0, "reset");
    check(AdaptiveDepth::drop_pressure(2) == 20 && AdaptiveDepth::drop_pressure(6) == 30, "drop_pressure = max(5d, 20)");
}

void test_tracker() {
    std::printf("[2. ProposalTracker]\n");
    ProposalTracker t;
    const int32_t p[3] = {5, 6, 7};
    int nd = 0, na = 0;
    check(t.issue(p, 3) && !t.issue(p, 3), "one open round at a time");
    const bool r1 = t.push(5, nd, na), r2 = t.push(6, nd, na), r3 = t.push(7, nd, na);
    check(!r1 && !r2 && r3 && nd == 3 && na == 3, "every token matched: (3, 3)");
    t.issue(p, 3);
    const bool s1 = t.push(5, nd, na), s2 = t.push(9, nd, na);
    check(!s1 && s2 && nd == 3 && na == 1, "the first miss resolves: (3, 1)");
    check(!t.push(4, nd, na) && !t.open(), "nothing open: nothing resolves");
}

// rounds: at each step the matcher proposes up to `depth` tokens; the round consumes accepted + 1 tokens of the stream
struct Rounds {
    uint64_t rounds = 0, drafted = 0, accepted = 0;
    int final_depth = 0;
};
Rounds run_rounds(const std::vector<int32_t>& s, int fixed, int n_max) {
    NgramConfig cfg;
    cfg.on = true;
    cfg.n_max = n_max;
    cfg.min_match = 3;
    NgramMatcher m(cfg);
    AdaptiveDepth a;
    a.reset(n_max, 1);
    Rounds r;
    size_t i = 0;
    m.append(s.data(), 16);
    i = 16;
    while (i < s.size()) {
        const int depth = fixed > 0 ? fixed : a.depth();
        int32_t buf[kMaxDraft];
        const int n = m.propose(buf, depth);
        if (n <= 0) {
            m.append(s[i++]);
            continue;
        }
        int acc = 0;
        while (acc < n && i + (size_t) acc < s.size() && buf[acc] == s[i + (size_t) acc]) ++acc;
        ++r.rounds;
        r.drafted += (uint64_t) n;
        r.accepted += (uint64_t) acc;
        if (fixed <= 0) a.update(n, acc, n_max, 1);
        const size_t take = std::min(s.size() - i, (size_t) acc + 1);   // the accepted tokens and the correction
        m.append(s.data() + i, take);
        i += take;
    }
    r.final_depth = fixed > 0 ? fixed : a.depth();
    return r;
}

uint64_t rng(uint64_t& x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

void test_rounds() {
    std::printf("[3. rounds: adaptive vs fixed depth (n_max 8)]\n");
    std::vector<int32_t> code, prose;
    uint64_t x = 88172645463325252ull;
    std::vector<int32_t> block;
    for (int i = 0; i < 48; ++i) block.push_back((int32_t) (rng(x) % 5000));
    for (int rep = 0; rep < 60; ++rep) {   // an edit session: the block again with an occasional changed token
        for (int32_t t : block) code.push_back(t);
        block[(size_t) (rng(x) % block.size())] = (int32_t) (rng(x) % 5000);
    }
    for (int i = 0; i < 3000; ++i) prose.push_back((int32_t) (rng(x) % 40));   // short accidental repeats only
    struct Line {
        const char* name;
        const std::vector<int32_t>* s;
    };
    Rounds ad_code{}, ad_prose{}, f8_code{}, f8_prose{};
    for (const Line& l : {Line{"code-like", &code}, Line{"prose-like", &prose}}) {
        for (int fixed : {0, 1, 4, 8}) {
            const Rounds r = run_rounds(*l.s, fixed, 8);
            std::printf("    %-10s %-9s rounds %5llu, drafted %6llu, accepted %6llu, rejected verify rows %6llu, "
                        "accepted/round %.2f, final depth %d\n", l.name,
                        fixed ? ("fixed " + std::to_string(fixed)).c_str() : "adaptive", (unsigned long long) r.rounds,
                        (unsigned long long) r.drafted, (unsigned long long) r.accepted,
                        (unsigned long long) (r.drafted - r.accepted), r.rounds ? (double) r.accepted / r.rounds : 0.0,
                        r.final_depth);
            if (l.s == &code && fixed == 0) ad_code = r;
            if (l.s == &code && fixed == 8) f8_code = r;
            if (l.s == &prose && fixed == 0) ad_prose = r;
            if (l.s == &prose && fixed == 8) f8_prose = r;
        }
    }
    check(ad_code.final_depth == 8, "code-like: the depth reaches the cap");
    check(ad_code.accepted * 10 >= f8_code.accepted * 9, "code-like: >= 90% of fixed-8's accepted tokens");
    check(ad_prose.final_depth <= 2, "prose-like: the depth stays shallow (" + std::to_string(ad_prose.final_depth) + ")");
    const uint64_t rej_ad = ad_prose.drafted - ad_prose.accepted, rej_8 = f8_prose.drafted - f8_prose.accepted;
    check(rej_ad * 3 <= rej_8, "prose-like: at most a third of fixed-8's rejected verify rows (" + std::to_string(rej_ad) +
                                   " vs " + std::to_string(rej_8) + ")");
}

void test_session() {
    std::printf("[4. NgramSession]\n");
    setenv("STRATA_GLM_NGRAM_DRAFT", "1", 1);
    setenv("STRATA_GLM_NGRAM_MAX", "8", 1);
    unsetenv("STRATA_GLM_DRAFT_ADAPTIVE");
    std::vector<int32_t> block;
    for (int i = 0; i < 40; ++i) block.push_back(100 + (i * 7) % 61);
    {
        NgramSession s(nullptr);
        check(!s.adaptive_on() && s.depth() == 8, "off: depth() is STRATA_GLM_NGRAM_MAX");
        for (int rep = 0; rep < 3; ++rep)
            for (int32_t t : block) {
                s.pick(-1);
                s.push(t);
            }
        check(s.tracked_drafted() == 0, "off: nothing tracked");
    }
    setenv("STRATA_GLM_DRAFT_ADAPTIVE", "1", 1);
    int climbed = 0;
    {
        NgramSession s(nullptr);
        check(s.adaptive_on() && s.depth() == 1, "on: starts at floor 1");
        size_t longest = 0;
        for (int rep = 0; rep < 8; ++rep)
            for (int32_t t : block) {
                s.pick(-1);
                longest = std::max(longest, s.last_proposal().size());
                s.push(t);
            }
        climbed = s.depth();
        check(climbed == 8 && longest == 8, "a repetitive conversation climbs to the cap; proposals follow depth() (" +
                                                std::to_string(climbed) + ")");
        check(s.tracked_accepted() > 0 && s.tracked_accepted() <= s.tracked_drafted(), "the rounds are tracked");
    }
    {
        NgramSession s(nullptr);
        check(s.depth() == 1, "a new session (generation) starts again at the floor");
    }
    setenv("STRATA_GLM_DRAFT_ADAPTIVE", "2", 1);
    {
        NgramSession s(nullptr);
        check(s.depth() == 2, "STRATA_GLM_DRAFT_ADAPTIVE=2: floor and start 2");
    }
    unsetenv("STRATA_GLM_DRAFT_ADAPTIVE");
    unsetenv("STRATA_GLM_NGRAM_DRAFT");
    unsetenv("STRATA_GLM_NGRAM_MAX");
}

}  // namespace

int main() {
    std::printf("glm_draft_adaptive_test: the adaptive draft depth (B15.9)\n");
    test_controller();
    test_tracker();
    test_rounds();
    test_session();
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
