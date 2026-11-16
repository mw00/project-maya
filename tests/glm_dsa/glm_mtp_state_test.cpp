// tests/glm_dsa/glm_mtp_state_test.cpp - glm-dsa B15.3 (CPU): the NextN carrier's sequence state, the MTP1 slot
// trailer, the block identity and the draft budget (include/strata/core/glm_mtp_state.hpp).
//
// A toy twin of the engine's MTP data flow runs the engine's own state machine (glmmtp::HostCarrier, the trailer
// functions, spec_budget) next to "legacy", Maya's behaviour (one shared scratch row per card, written by every
// trunk head, never reset or saved):
//   1. slot round trip: the first draft after slot_load equals the first draft of a fresh process; legacy differs
//      (the row belongs to whatever ran last);
//   2. reset at pos 0 (#28333): a prompt whose last row reaches the draft without a token-path note (the batched
//      prompt path) drafts from zeros exactly like a fresh process, even when the previous request had the same
//      length (the position check alone would accept its row); legacy differs;
//   3. two interleaved sequences (MAYA_GLM_SEQS): each drafts from its own carrier; legacy differs;
//   4. snapshot round trip: the first draft after snapshot_restore equals the one at the snapshot;
//   5. the MTP1 trailer: round trip; truncated, wrong magic / width / block / position / checksum are refused and
//      leave the state untouched; a slot written before B15.3 (no trailer) reads as Absent;
//   6. the draft budget (#28473): draft_budget's table, and decode_spec's loop (simulated) issues no speculative
//      verify row past n_predict or the context, with the same tokens (counted for every max_new and context);
//   7. the block identity: order-independent, changes with a tensor type or the source, never 0; fingerprint words.
#include "strata/core/glm_mtp_state.hpp"
#include "strata/core/glm_state_layout.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

using namespace strata::core;

constexpr int E = 16, V = 23;

float emb(int tok, int i) { return std::sin(0.37f * (float) (tok + 1) * (float) (i + 1)) * 0.8f; }

// The toy model: the trunk keeps a running state (its "KV"); the NextN block keeps one row per position (its own
// latent cache) and reads the carrier h_p and the next token's embedding.
struct Toy {
    struct Seq {
        std::vector<float> trunk = std::vector<float>(E, 0.0f);   // the trunk's state after the last position
        std::vector<std::vector<float>> nextn;                     // the NextN cache rows, one a position
        int64_t pos = 0;
        glmmtp::HostCarrier carry{E};
        // snapshot
        std::vector<float> s_trunk;
        std::vector<std::vector<float>> s_nextn;
        int64_t s_pos = -1;
    };
    bool legacy = false;            // Maya: one shared scratch row (head_x), no reset, no save
    bool reset_hook = true;         // B15.3 (a): Glm5Model::reset zeroes the carrier
    std::vector<float> head_x = std::vector<float>(E, 0.0f);
    std::map<int, Seq> seqs;
    int cur = 0;
    Seq& S() { return seqs[cur]; }

    std::vector<float> trunk_step(Seq& q, int tok) {
        std::vector<float> h(E);
        float ss = 0;
        for (int i = 0; i < E; ++i) {
            q.trunk[(size_t) i] = 0.7f * q.trunk[(size_t) i] + emb(tok, i);
            h[(size_t) i] = std::tanh(q.trunk[(size_t) i] + 0.1f * (float) i);
            ss += h[(size_t) i] * h[(size_t) i];
        }
        for (float& v : h) v /= std::sqrt(ss / E + 1e-6f);
        ++q.pos;
        return h;
    }
    // the prompt path: positions without a token-path head (no note); the NextN rows from the batch's own hidden
    void prefill(const std::vector<int>& toks, int last_next) {
        Seq& q = S();
        for (size_t i = 0; i < toks.size(); ++i) {
            const std::vector<float> h = trunk_step(q, toks[i]);
            const int nx = i + 1 < toks.size() ? toks[i + 1] : last_next;
            q.nextn.push_back(row(h, nx));
        }
    }
    // the token path: the trunk head notes the carrier (engine) or writes the shared scratch (legacy)
    int token(int tok) {
        Seq& q = S();
        const int64_t p = q.pos;
        const std::vector<float> h = trunk_step(q, tok);
        head_x = h;
        if (!legacy) q.carry.note(p, h.data());
        int best = 0;
        float bv = -1e30f;
        for (int v = 0; v < V; ++v) {
            float a = 0;
            for (int i = 0; i < E; ++i) a += h[(size_t) i] * emb(v + 3, i);
            if (a > bv) bv = a, best = v;
        }
        return best;
    }
    static std::vector<float> row(const std::vector<float>& h, int next) {
        std::vector<float> r(E);
        for (int i = 0; i < E; ++i) r[(size_t) i] = 0.5f * h[(size_t) i] + emb(next, (i * 7) % E);
        return r;
    }
    // fast_mtp at p (= pos - 1): writes NextN row p, attends over rows 0..p, the draft's argmax
    int draft(int next) {
        Seq& q = S();
        const int64_t p = q.pos - 1;
        std::vector<float> h(E);
        if (legacy) h = head_x;
        else std::memcpy(h.data(), q.carry.read(p), sizeof(float) * E);
        const std::vector<float> r = row(h, next);
        if ((int64_t) q.nextn.size() <= p) q.nextn.resize((size_t) p + 1, std::vector<float>(E, 0.0f));
        q.nextn[(size_t) p] = r;
        std::vector<float> ctx(E, 0.0f);
        for (int64_t t = 0; t <= p; ++t)
            for (int i = 0; i < E; ++i) ctx[(size_t) i] += q.nextn[(size_t) t][(size_t) i] / (float) (p + 1);
        int best = 0;
        float bv = -1e30f;
        for (int v = 0; v < V; ++v) {
            float a = 0;
            for (int i = 0; i < E; ++i) a += (ctx[(size_t) i] + 2.0f * h[(size_t) i]) * emb(v + 11, i);
            if (a > bv) bv = a, best = v;
        }
        return best;
    }
    void reset() {
        Seq& q = S();
        q.trunk.assign(E, 0.0f);
        q.nextn.clear();
        q.pos = 0;
        if (!legacy && reset_hook) q.carry.begin();
    }
    void snapshot_save() {
        Seq& q = S();
        q.s_trunk = q.trunk;
        q.s_nextn = q.nextn;
        q.s_nextn.resize((size_t) q.pos, std::vector<float>(E, 0.0f));
        q.s_pos = q.pos;
        if (!legacy) q.carry.snapshot_save(q.pos);
    }
    void snapshot_restore() {
        Seq& q = S();
        q.trunk = q.s_trunk;
        q.nextn = q.s_nextn;
        q.nextn.resize((size_t) q.s_pos);
        q.pos = q.s_pos;
        if (!legacy) q.carry.snapshot_restore();
    }
    // a slot: the snapshot's trunk state and NextN rows, then (engine) the MTP1 trailer of the snapshot's copy
    std::string slot_save(uint32_t block) {
        Seq& q = S();
        std::ostringstream f;
        f.write((const char*) &q.s_pos, 8);
        f.write((const char*) q.s_trunk.data(), E * 4);
        for (int64_t t = 0; t < q.s_pos; ++t) f.write((const char*) q.s_nextn[(size_t) t].data(), E * 4);
        if (!legacy) glmmtp::write_trailer(f, E, q.carry.st.snap_pos, block, q.carry.snap.data());
        return f.str();
    }
    bool slot_load(const std::string& bytes, uint32_t block) {
        Seq& q = S();
        std::istringstream f(bytes);
        int64_t n = 0;
        f.read((char*) &n, 8);
        q.s_trunk.assign(E, 0.0f);
        f.read((char*) q.s_trunk.data(), E * 4);
        q.s_nextn.assign((size_t) n, std::vector<float>(E, 0.0f));
        for (int64_t t = 0; t < n; ++t) f.read((char*) q.s_nextn[(size_t) t].data(), E * 4);
        q.s_pos = n;
        if (!legacy) {
            int64_t pos = -1;
            std::vector<float> r;
            std::string why;
            if (glmmtp::read_trailer(f, E, n, block, pos, r, why) == glmmtp::Trailer::Ok && pos >= 0) {
                q.carry.snap = r;
                q.carry.st.snap_pos = pos;
            } else {
                q.carry.st.snap_pos = -1;
                std::fill(q.carry.snap.begin(), q.carry.snap.end(), 0.0f);
            }
        }
        snapshot_restore();
        return (bool) f || f.eof();
    }
};

const std::vector<int> kP = {3, 17, 5, 9, 12, 1, 20, 7};
const std::vector<int> kQ = {8, 2, 14, 6, 19, 11, 4, 15};   // same length as kP: the coincident-position case

// a fresh process: the whole prompt through the token path, the snapshot before nothing else, the first draft
int fresh_token_path(bool legacy, int& y) {
    Toy t;
    t.legacy = legacy;
    for (int tok : kP) y = t.token(tok);
    return t.draft(y);
}

void test_slot_round_trip() {
    std::printf("[1. slot round trip: the first draft after slot_load]\n");
    for (int legacy = 0; legacy < 2; ++legacy) {
        int y = 0;
        const int want = fresh_token_path(legacy, y);
        Toy t;
        t.legacy = legacy;
        for (int tok : kP) t.token(tok);
        t.snapshot_save();
        const std::string slot = t.slot_save(77);
        // another conversation runs (its trunk heads overwrite any shared row), then the slot comes back
        t.reset();
        for (int tok : kQ) t.token(tok);
        t.reset();
        t.slot_load(slot, 77);
        const int got = t.draft(y);
        if (!legacy)
            check(got == want, "engine: the first draft after a slot restore == a fresh process's (" + std::to_string(got) +
                                   " vs " + std::to_string(want) + ")");
        else
            check(got != want, "legacy (Maya's shared row) drafts differently after the restore (" + std::to_string(got) +
                                   " vs " + std::to_string(want) + "): the check bites");
    }
}

void test_reset_pos0() {
    std::printf("[2. a sequence that starts at pos 0 drafts from a zero carrier (#28333)]\n");
    // fresh: the whole prompt by the batched path (no trunk head note), then the first draft
    int want = -1;
    {
        Toy t;
        t.prefill(std::vector<int>(kP.begin(), kP.end() - 1), kP.back());
        t.prefill({kP.back()}, 4);
        want = t.draft(4);
    }
    struct Case {
        const char* name;
        bool legacy, reset_hook, expect_equal;
    };
    for (const Case c : {Case{"engine (reset zeroes the carrier)", false, true, true},
                         Case{"position check alone (no reset hook): Q's row at the same position", false, false, false},
                         Case{"legacy (Maya's shared row)", true, false, false}}) {
        Toy t;
        t.legacy = c.legacy;
        t.reset_hook = c.reset_hook;
        for (int tok : kQ) t.token(tok);   // the previous request, same length: its carrier is at position 7
        t.reset();
        t.prefill(std::vector<int>(kP.begin(), kP.end() - 1), kP.back());
        t.prefill({kP.back()}, 4);
        const int got = t.draft(4);
        check((got == want) == c.expect_equal, std::string(c.name) + ": draft " + std::to_string(got) + " vs fresh " +
                                                   std::to_string(want) + (c.expect_equal ? " (equal)" : " (differs)"));
    }
}

void test_two_sequences() {
    std::printf("[3. two interleaved sequences (MAYA_GLM_SEQS)]\n");
    for (int legacy = 0; legacy < 2; ++legacy) {
        int ya = 0, yb = 0;
        int want_a = 0, want_b = 0;
        {
            Toy t;
            for (int tok : kP) ya = t.token(tok);
            want_a = t.draft(ya);
        }
        {
            Toy t;
            for (int tok : kQ) yb = t.token(tok);
            want_b = t.draft(yb);
        }
        Toy t;
        t.legacy = legacy;
        for (size_t i = 0; i < kP.size(); ++i) {
            t.cur = 0;
            t.token(kP[i]);
            t.cur = 1;
            t.token(kQ[i]);
        }
        t.cur = 0;   // sequence 0 drafts after sequence 1 ran last on the card
        const int a = t.draft(ya);
        t.cur = 1;
        const int b = t.draft(yb);
        if (!legacy)
            check(a == want_a && b == want_b, "engine: each sequence drafts from its own carrier");
        else
            check(a != want_a, "legacy: sequence 0 drafts from sequence 1's row (" + std::to_string(a) + " vs " +
                                   std::to_string(want_a) + ")");
    }
}

void test_snapshot() {
    std::printf("[4. snapshot round trip]\n");
    Toy t;
    int y = 0;
    for (int tok : kP) y = t.token(tok);
    t.snapshot_save();
    const int d0 = t.draft(y);
    for (int tok : {2, 4, 6, 8}) t.token(tok);
    t.snapshot_restore();
    const int d1 = t.draft(y);
    check(d0 == d1, "the first draft after snapshot_restore == the draft at the snapshot");
    // a snapshot whose last position came from the batched path keeps no row: the restore drafts from zeros
    Toy u;
    u.prefill(kP, 4);
    u.snapshot_save();
    check(u.S().carry.st.snap_pos == -1, "a snapshot without h_{n-1} keeps no carrier copy");
}

void test_trailer() {
    std::printf("[5. the MTP1 trailer]\n");
    std::vector<float> row(E);
    for (int i = 0; i < E; ++i) row[(size_t) i] = 0.25f * (float) i - 1.0f;
    std::ostringstream o;
    glmmtp::write_trailer(o, E, 9, 1234, row.data());
    const std::string good = o.str();
    check(good.size() == glmmtp::trailer_bytes(E), "trailer_bytes(" + std::to_string(E) + ") = " +
                                                       std::to_string(good.size()));
    {
        std::istringstream f(good);
        int64_t pos = -5;
        std::vector<float> r;
        std::string why;
        const auto t = glmmtp::read_trailer(f, E, 10, 1234, pos, r, why);
        check(t == glmmtp::Trailer::Ok && pos == 9 && r == row, "round trip");
    }
    {
        std::istringstream f("");
        int64_t pos = -5;
        std::vector<float> r;
        std::string why;
        check(glmmtp::read_trailer(f, E, 10, 1234, pos, r, why) == glmmtp::Trailer::Absent && pos == -5,
              "no trailer (a slot written before B15.3): Absent, nothing set");
    }
    struct Bad {
        const char* what;
        std::string bytes;
        int n_embd;
        int64_t n;
        uint32_t block;
    };
    std::string bad_magic = good, bad_sum = good, bad_pos = good, trunc = good.substr(0, good.size() - 3);
    bad_magic[0] ^= 1;
    bad_sum[24 + 5] ^= 0x40;
    int64_t p7 = 7;
    std::memcpy(&bad_pos[8], &p7, 8);
    for (const Bad& b : {Bad{"truncated", trunc, E, 10, 1234}, Bad{"wrong magic", bad_magic, E, 10, 1234},
                         Bad{"another width", good, E + 32, 10, 1234}, Bad{"another NextN block", good, E, 10, 99},
                         Bad{"a position other than n - 1", bad_pos, E, 10, 1234},
                         Bad{"a slot of another length", good, E, 12, 1234}, Bad{"a damaged row", bad_sum, E, 10, 1234}}) {
        std::istringstream f(b.bytes);
        int64_t pos = -5;
        std::vector<float> r = {42.0f};
        std::string why;
        const auto t = glmmtp::read_trailer(f, b.n_embd, b.n, b.block, pos, r, why);
        check(t == glmmtp::Trailer::Bad && pos == -5 && r.size() == 1 && r[0] == 42.0f,
              std::string(b.what) + ": refused, state untouched (" + why + ")");
    }
}

// decode_spec's loop shape (src/core/glm_fast_path.cu) over its counters: how many speculative head positions it
// issues, how many of them are ever verified, and the tokens it emits.  budget: B15.3's spec_budget gate.
struct LoopRun {
    int64_t produced = 0, spec = 0, verified = 0;
};
LoopRun spec_loop(int64_t prompt, int64_t max_new, int64_t max_ctx, bool budget) {
    LoopRun r;
    int64_t q = prompt - 1;
    ++r.produced;
    if (r.produced >= max_new || q + 2 >= max_ctx) return r;
    bool pending = false;
    if (!budget || glmmtp::spec_budget(1, r.produced, max_new, q + 2, max_ctx) > 0) ++r.spec, pending = true;
    for (;;) {
        ++r.produced;   // take y2
        const bool more = r.produced < max_new && q + 3 < max_ctx;
        if (!more) break;
        if (pending) ++r.verified;   // the head's q+2 is checked against y2
        else return LoopRun{-1, -1, -1};   // the loop needs the head's position: a budget that skipped it wrongly
        pending = false;
        if (!budget || glmmtp::spec_budget(1, r.produced, max_new, q + 3, max_ctx) > 0) ++r.spec, pending = true;
        ++q;
    }
    return r;
}

void test_budget() {
    std::printf("[6. the draft budget (#28473)]\n");
    check(glmmtp::draft_budget(4, 10, 10) == 4 && glmmtp::draft_budget(4, 2, 10) == 2 &&
              glmmtp::draft_budget(4, 10, 1) == 1 && glmmtp::draft_budget(4, 0, 10) == 0 &&
              glmmtp::draft_budget(4, -3, 10) == 0 && glmmtp::draft_budget(1, 5, 0) == 0,
          "draft_budget = max(0, min(n_max, remaining n_predict, remaining ctx))");
    int64_t wasted_old = 0, wasted_new = 0, cases = 0;
    bool same = true, none = true;
    for (int64_t max_new = 1; max_new <= 12; ++max_new)
        for (int64_t ctx = 9; ctx <= 24; ++ctx) {
            const LoopRun a = spec_loop(8, max_new, ctx, false), b = spec_loop(8, max_new, ctx, true);
            same &= a.produced == b.produced && b.produced >= 0;
            wasted_old += a.spec - a.verified;
            wasted_new += b.spec - b.verified;
            none &= b.spec == b.verified;
            ++cases;
        }
    check(same, "the budget gate emits the same number of tokens in all " + std::to_string(cases) + " cases");
    check(none && wasted_new == 0, "no speculative head position past n_predict or the context (was " +
                                       std::to_string(wasted_old) + " wasted verify rows over the cases, now " +
                                       std::to_string(wasted_new) + ")");
    check(wasted_old > 0, "the ungated loop does waste rows (the check bites)");
}

void test_block_id() {
    std::printf("[7. the NextN block identity and the fingerprint words]\n");
    std::vector<glmmtp::BlockTensor> a = {{"blk.78.nextn.eh_proj.weight", 16, 100}, {"blk.78.attn_q_a.weight", 16, 50},
                                          {"blk.78.#experts", 16 * 256 + 18, 4096}};
    std::vector<glmmtp::BlockTensor> b = {a[2], a[0], a[1]};
    std::vector<glmmtp::BlockTensor> c = a;
    c[0].type = 12;   // a Q4_K eh_proj instead of IQ2_XXS
    const uint32_t ia = glmmtp::block_id(a, "own"), ib = glmmtp::block_id(b, "own"), ic = glmmtp::block_id(c, "own"),
                   id = glmmtp::block_id(a, "gguf:mtp.gguf:123");
    check(ia == ib, "order-independent");
    check(ia != ic, "another tensor type, another id");
    check(ia != id, "another source (MAYA_GLM_MTP_GGUF), another id");
    check(ia != 0 && ia <= 0x7fffffffu && glmmtp::block_id({}, "") != 0, "never 0, 31 bits");
    check(glmmtp::fp_block_word(ia) == (int32_t) ia && glmmtp::fp_carrier_word(true, ia, 6144) == 24576 &&
              glmmtp::fp_carrier_word(true, 0, 6144) == 0 && glmmtp::fp_carrier_word(false, ia, 6144) == 0,
          "fingerprint words: block id; carrier bytes only with a block on glm-dsa");
    static_assert(std::tuple_size<GlmStateFingerprint>::value == 14, "B15.3: 14 fingerprint words");
    check(true, "GlmStateFingerprint has 14 words (12 + the NextN block and carrier words)");
}

}  // namespace

int main() {
    std::printf("glm_mtp_state_test: the NextN carrier's sequence state (B15.3)\n");
    test_slot_round_trip();
    test_reset_pos0();
    test_two_sequences();
    test_snapshot();
    test_trailer();
    test_budget();
    test_block_id();
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
