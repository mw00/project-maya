// tests/glm_dsa/glm_ngram_draft_test.cpp - glm-dsa B14.13 (CPU only): the n-gram draft matcher and its bookkeeping
// (include/strata/core/glm_ngram_draft.hpp).
//   1. the switches: off by default, n_max 4 (capped at 8), min_match 3; a request's GEN ngram= (glm_req_knobs.hpp)
//      over MAYA_GLM_NGRAM_DRAFT, the switch again once a request leaves it out;
//   2. the matcher: the continuation of the earlier occurrence, the longest backward match wins, the most recent wins
//      equal lengths, a match below min_match or with nothing after it proposes nothing, n_max and the history end
//      cap the proposal, hash collisions cannot propose (exact token check), incremental appends equal one batch;
//   3. the bookkeeping: the FIFO of outstanding drafts in the pipeline's order (pick d, pick d2, verdict d), per-source
//      counts, the chooser's warm-up and its fall back to MTP after a run of rejected n-gram drafts;
//   4. the session as Glm5Model::decode_spec drives it (the registered prompt, push / pick / verdict), on a synthetic
//      edit turn (the answer repeats a code block of the prompt with one edit) against a 70%-accurate MTP stand-in:
//      the accepted fraction rises, every emitted token is the truth (drafts never change the output), and with the
//      switch off pick() returns the MTP draft untouched.
#include "strata/core/glm_ngram_draft.hpp"
#include "strata/core/glm_req_knobs.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

using namespace strata::core::glmng;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

std::vector<int32_t> prop(NgramMatcher& m, int max_k = -1) {
    int32_t b[kMaxDraft];
    const int n = m.propose(b, max_k);
    return std::vector<int32_t>(b, b + n);
}

NgramConfig cfg(int n_max = 4, int min_match = 3) {
    NgramConfig c;
    c.on = true;
    c.n_max = n_max;
    c.min_match = min_match;
    return c;
}

void test_switches() {
    std::printf("[1. the switches]\n");
    unsetenv("MAYA_GLM_NGRAM_DRAFT");
    unsetenv("MAYA_GLM_NGRAM_MAX");
    unsetenv("MAYA_GLM_NGRAM_MIN");
    NgramConfig c = ngram_config_env();
    check(!c.on && c.n_max == 4 && c.min_match == 3, "defaults: off, n_max 4, min_match 3");
    setenv("MAYA_GLM_NGRAM_DRAFT", "1", 1);
    setenv("MAYA_GLM_NGRAM_MAX", "64", 1);
    setenv("MAYA_GLM_NGRAM_MIN", "2", 1);
    c = ngram_config_env();
    check(c.on && c.n_max == kMaxDraft && c.min_match == 2, "MAYA_GLM_NGRAM_MAX=64 is capped at 8; MIN=2 read");
    setenv("MAYA_GLM_NGRAM_DRAFT", "0", 1);
    check(!ngram_config_env().on, "MAYA_GLM_NGRAM_DRAFT=0: off");
    namespace gk = strata::core::glmknob;
    gk::set(gk::Ngram, 1);
    check(ngram_config_env().on, "a request's ngram=1 over MAYA_GLM_NGRAM_DRAFT=0: on");
    setenv("MAYA_GLM_NGRAM_DRAFT", "1", 1);
    gk::set(gk::Ngram, 0);
    check(!ngram_config_env().on, "a request's ngram=0 over MAYA_GLM_NGRAM_DRAFT=1: off");
    gk::set(gk::Ngram, gk::kStart);
    check(ngram_config_env().on, "the next request leaves it out: the switch again (on)");
    unsetenv("MAYA_GLM_NGRAM_DRAFT");
    unsetenv("MAYA_GLM_NGRAM_MAX");
    unsetenv("MAYA_GLM_NGRAM_MIN");
}

void test_matcher() {
    std::printf("[2. the matcher]\n");
    {
        NgramMatcher m(cfg());
        const std::vector<int32_t> h = {1, 2, 3, 4, 5, 6, 7, 1, 2, 3};
        m.append(h.data(), h.size());
        check(prop(m) == std::vector<int32_t>({4, 5, 6, 7}) && m.last_match() == 3 && m.last_source() == 3,
              "1 2 3 4 5 6 7 1 2 3 -> 4 5 6 7 (match 3, copied from position 3)");
        check(prop(m, 2) == std::vector<int32_t>({4, 5}), "max_k 2 -> 4 5");
    }
    {   // the longest backward match beats a more recent shorter one
        NgramMatcher m(cfg());
        const std::vector<int32_t> h = {9, 1, 2, 3, 50, 8, 1, 2, 3, 60, 0, 9, 1, 2, 3};
        m.append(h.data(), h.size());
        check(prop(m, 1) == std::vector<int32_t>({50}) && m.last_match() == 4, "longest match wins (9 1 2 3 -> 50)");
    }
    {   // equal lengths: the most recent
        NgramMatcher m(cfg());
        const std::vector<int32_t> h = {7, 1, 2, 3, 50, 8, 1, 2, 3, 60, 0, 9, 1, 2, 3};
        m.append(h.data(), h.size());
        check(prop(m, 1) == std::vector<int32_t>({60}) && m.last_match() == 3, "equal lengths: the most recent (-> 60)");
    }
    {   // below min_match: nothing; min_match 2 then matches
        const std::vector<int32_t> h = {1, 2, 40, 5, 2, 3, 1, 2};
        NgramMatcher m3(cfg(4, 3)), m2(cfg(4, 2));
        m3.append(h.data(), h.size());
        m2.append(h.data(), h.size());
        check(prop(m3).empty() && m3.last_match() == 0, "a 2-token match with min_match 3 proposes nothing");
        check(prop(m2) == std::vector<int32_t>({40, 5, 2, 3}), "min_match 2: 1 2 -> 40 5 2 3");
    }
    {   // the proposal stops at the history's end; nothing after the only occurrence
        NgramMatcher m(cfg(8));
        const std::vector<int32_t> h = {5, 6, 7, 5, 6, 7};
        m.append(h.data(), h.size());
        check(prop(m) == std::vector<int32_t>({5, 6, 7}), "5 6 7 5 6 7 -> 5 6 7 (capped by the history end)");
        NgramMatcher u(cfg());
        const std::vector<int32_t> h2 = {1, 2, 3, 4, 5};
        u.append(h2.data(), h2.size());
        check(prop(u).empty(), "no earlier occurrence: nothing");
    }
    {   // n_max caps
        NgramMatcher m(cfg(2));
        const std::vector<int32_t> h = {1, 2, 3, 4, 5, 6, 1, 2, 3};
        m.append(h.data(), h.size());
        check(prop(m) == std::vector<int32_t>({4, 5}), "n_max 2 -> 4 5");
    }
    {   // incremental appends == one batch; reset clears
        std::vector<int32_t> h;
        for (int i = 0; i < 3000; ++i) h.push_back((int32_t) ((i * 7919) % 97));
        NgramMatcher a(cfg()), b(cfg());
        a.append(h.data(), h.size());
        for (int32_t t : h) b.append(t);
        bool same = true;
        for (int k = 0; k < 50; ++k) {
            same = same && prop(a) == prop(b);
            const int32_t t = (int32_t) ((k * 31) % 97);
            a.append(t);
            b.append(t);
        }
        check(same && a.size() == b.size(), "token-by-token appends propose what one batch append does");
        a.reset();
        check(a.size() == 0 && prop(a).empty(), "reset clears the history");
    }
    {   // large token ids (the full int32 range): exact
        NgramMatcher m(cfg());
        const std::vector<int32_t> h = {154819, 2000000000, -5, 7, 154819, 2000000000, -5};
        m.append(h.data(), h.size());
        check(prop(m) == std::vector<int32_t>({7, 154819, 2000000000, -5}), "ids above 2^16 and negative ids");
    }
}

void test_book() {
    std::printf("[3. the bookkeeping]\n");
    DraftBook b;
    check(!b.verdict(true), "a verdict with nothing outstanding is refused");
    b.issued(DraftSrc::Ngram);   // d
    b.issued(DraftSrc::Mtp);     // d2
    check(b.verdict(true) && b.drafts[1] == 1 && b.hits[1] == 1 && b.drafts[0] == 0, "FIFO: the first verdict is d's");
    check(b.verdict(false) && b.drafts[0] == 1 && b.hits[0] == 0, "then d2's");
    DraftBook c;
    for (int i = 0; i < DraftBook::kWarmup - 1; ++i) {
        c.issued(DraftSrc::Ngram);
        c.verdict(false);
    }
    check(c.prefer_ngram(), "warm-up: the n-gram is tried for its first 8 drafts whatever their outcome");
    c.issued(DraftSrc::Ngram);
    c.verdict(false);
    for (int i = 0; i < 20; ++i) {
        c.issued(DraftSrc::Mtp);
        c.verdict(true);
    }
    check(!c.prefer_ngram(), "after the warm-up a rejected n-gram run against an accepted MTP falls back to MTP");
    DraftBook d;
    for (int i = 0; i < 40; ++i) {
        d.issued(i % 2 ? DraftSrc::Mtp : DraftSrc::Ngram);
        d.verdict(i % 2 ? (i % 4 == 1) : true);   // n-gram always right, MTP half the time
    }
    check(d.prefer_ngram() && d.rate(DraftSrc::Ngram) == 1.0 && d.rate(DraftSrc::Mtp) == 0.5,
          "an accurate n-gram stays preferred; rates 100% / 50%");
    check(d.line().find("n-gram drafts 20") != std::string::npos, "the summary line: " + d.line());
}

// a synthetic edit turn: the prompt holds a code block, the answer repeats it with one token changed
struct Turn {
    std::vector<int32_t> prompt, answer;
};
Turn edit_turn() {
    Turn t;
    for (int i = 0; i < 200; ++i) t.prompt.push_back(1000 + (i * 37) % 500);   // prose
    std::vector<int32_t> code;
    for (int i = 0; i < 300; ++i) code.push_back(5000 + (i * 13) % 211 + (i / 211) * 300);
    t.prompt.insert(t.prompt.end(), code.begin(), code.end());
    for (int i = 0; i < 20; ++i) t.prompt.push_back(1000 + i);   // the request
    t.answer = code;
    t.answer[150] = 9999;   // the edit
    return t;
}

// the decode_spec call sequence on a known answer: push(y), d = pick(mtp), then per step push(y2), d2 = pick(mtp2),
// verdict(d, y2), d = d2.  mtp(i) is right 70% of the time.
struct Sim {
    int drafts = 0, accepted = 0;
    int mtp_noted = 0, mtp_hits = 0;   // what B11.3's counter records: the MTP head's own draft at every position
    bool mtp_ok = true;                // ... always the draft pick() was given
    bool output_ok = true;
};
Sim simulate(const Turn& t, bool on) {
    if (on)
        setenv("MAYA_GLM_NGRAM_DRAFT", "1", 1);
    else
        unsetenv("MAYA_GLM_NGRAM_DRAFT");
    static int model_tag;
    set_history(&model_tag, t.prompt);
    NgramSession ng(&model_tag);
    const auto mtp = [&](size_t i) { return (i * 2654435761u) % 10 < 7 ? t.answer[i] : (int32_t) 77; };
    Sim s;
    std::vector<int32_t> out;
    size_t i = 0;
    int32_t y = t.answer[i];
    out.push_back(y);
    ng.push(y);
    int32_t d = ng.pick(mtp(i + 1));
    for (i = 1; i < t.answer.size(); ++i) {
        const int32_t y2 = t.answer[i];   // the tail's token: the truth, whatever was drafted
        out.push_back(y2);
        ng.push(y2);
        const int32_t d2 = i + 1 < t.answer.size() ? ng.pick(mtp(i + 1)) : -1;
        ++s.drafts;
        s.accepted += d == y2 ? 1 : 0;
        const int32_t dm = ng.mtp_draft(d);   // (decode_spec: glmspec::note(this, 1, q + 2, ng.mtp_draft(d), y2))
        s.mtp_ok = s.mtp_ok && dm == mtp(i);
        ++s.mtp_noted;
        s.mtp_hits += dm == y2 ? 1 : 0;
        ng.verdict(d, y2);
        d = d2;
    }
    s.output_ok = out == t.answer;
    if (!on) {   // off: the MTP draft is returned untouched and nothing is recorded
        s.output_ok = s.output_ok && ng.pick(1234) == 1234 && ng.book().drafts[0] + ng.book().drafts[1] == 0;
    } else {
        std::printf("     %s\n", ng.book().line().c_str());
    }
    unsetenv("MAYA_GLM_NGRAM_DRAFT");
    return s;
}

void test_session() {
    std::printf("[4. the session on a synthetic edit turn]\n");
    const Turn t = edit_turn();
    const Sim off = simulate(t, false), on = simulate(t, true);
    char buf[200];
    std::snprintf(buf, sizeof buf, "accepted drafts: MTP only %d / %d, n-gram + MTP %d / %d", off.accepted, off.drafts,
                  on.accepted, on.drafts);
    check(off.output_ok && on.output_ok && on.drafts == off.drafts && on.accepted > off.accepted + off.drafts / 5, buf);
    std::snprintf(buf, sizeof buf, "the MTP acceptance counter sees the MTP drafts only: %d / %d off, %d / %d on",
                  off.mtp_hits, off.mtp_noted, on.mtp_hits, on.mtp_noted);
    check(off.mtp_ok && on.mtp_ok && on.mtp_noted == off.mtp_noted && on.mtp_hits == off.mtp_hits, buf);
    setenv("MAYA_GLM_NGRAM_DRAFT", "1", 1);   // (set_history keeps nothing while the drafts are off)
    static int other;
    set_history(&other, {1, 2, 3});
    check(take_history(&other) == std::vector<int32_t>({1, 2, 3}) && take_history(&other).empty(),
          "the history registry: taken once per model");
    // the several-conversation scheduler: the prompt and the tokens produced so far (a resumed stint keeps both)
    set_history(&other, std::vector<int32_t>({1, 2}), std::vector<int32_t>({7, 8, 9}));
    check(take_history(&other) == std::vector<int32_t>({1, 2, 7, 8, 9}),
          "set_history(prompt, emitted): the history is the prompt then the emitted tokens");
}

}  // namespace

int main() {
    test_switches();
    test_matcher();
    test_book();
    test_session();
    if (g_fail) {
        std::printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
