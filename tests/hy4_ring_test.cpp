// maya-hy4's rings after rewinds (src/hy4/hy4_ring.hpp's ring_lo_at / ring_holds, #1091; no GPU): a ring of `rows`
// slots, each slot the position it holds and the prefix its row was made from (a hash of every token up to it), driven
// as the serve loop drives the engine - a request reuses the state's shared start when Model::resumable says the rings
// hold the window before it (else it reads from 0), its prompt in chunks, its last token alone, then its answer a token
// (or a draft window, some drafts rejected) at a time.  Every read of a window row is checked against the slot: a row
// of another position or prefix is a wrong answer.
//  - the issue's order at small sizes: B, C, A, B, C - the old check (end - rows) lets the second C read rows A wrote
//    over; ring_lo refuses it (C read from 0) and no read is wrong
//  - random requests over prefixes of one text, rewinds of every depth, draft windows: no wrong read, the rings hold
//    every position in [max(ring_lo, end - rows), end), and ring_holds says yes exactly where every row it would read
//    is there
//  - continuations alone (each request the last one's tokens and more, drafts rejected): no request reads from 0, and
//    every resume the old check took is taken still - but where rejected drafts had written over the window's oldest
//    rows (the old check counted from the end after them, not the end they reached)
#include "../src/hy4/hy4_ring.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

using Toks = std::vector<uint64_t>;

struct Sim {
    int64_t rows, n_swa, chunk;
    bool old = false;                // (the check before #1091: the rings taken as [end - rows, end))
    std::vector<int64_t> slot_pos;
    Toks slot_tok, seq, hs, state;   // (the slots' prefixes; the engine's tokens [0, end), their prefixes; the state)
    int64_t end = 0;
    int64_t ring_lo = 0;
    long long wrong = 0, refused = 0, reads = 0;

    Sim(int64_t r, int64_t w, int64_t c)
        : rows(r), n_swa(w), chunk(c), slot_pos((size_t) r, -1), slot_tok((size_t) r) {}

    int64_t back() const { return n_swa - 1; }
    bool resumable(int64_t pos) const {
        if (!old) return hy4::ring_holds(pos, end, ring_lo, rows, back());
        if (pos == 0) return true;
        if (pos > end) return false;
        return pos - n_swa + 1 >= end - rows;
    }
    bool slot_ok(int64_t q) const {
        const size_t s = (size_t) (q % rows);
        return slot_pos[s] == q && slot_tok[s] == hs[(size_t) q];
    }
    // whether every row a forward from pos reads before it is in its slot
    bool really_holds(int64_t pos) const {
        if (pos == 0) return true;
        if (pos > end) return false;
        for (int64_t q = std::max<int64_t>(0, pos - back()); q < pos; ++q)
            if (!slot_ok(q)) return false;
        return true;
    }
    // Model::forward: n ids from pos0 (chunks of `chunk`), each chunk's rows stored, then its windows read
    bool forward(const uint64_t* ids, int64_t n, int64_t pos0) {
        if (!resumable(pos0)) return false;
        ring_lo = hy4::ring_lo_at(pos0, ring_lo, end, rows);   // (as Model::forward: before end moves)
        seq.resize((size_t) pos0);
        seq.insert(seq.end(), ids, ids + n);
        hs.resize((size_t) pos0);
        for (int64_t i = 0; i < n; ++i)
            hs.push_back(((pos0 + i ? hs[(size_t) (pos0 + i - 1)] : 0) ^ ids[i]) * 0x100000001b3ull + 0x9e37ull);
        end = pos0;
        for (int64_t done = 0; done < n;) {
            const int64_t T = std::min(chunk, n - done), p0 = pos0 + done;
            for (int64_t t = 0; t < T; ++t) {
                slot_pos[(size_t) ((p0 + t) % rows)] = p0 + t;
                slot_tok[(size_t) ((p0 + t) % rows)] = hs[(size_t) (p0 + t)];
            }
            for (int64_t t = 0; t < T; ++t)
                for (int64_t q = std::max<int64_t>(0, p0 + t - back()); q <= p0 + t; ++q) {
                    ++reads;
                    if (!slot_ok(q)) ++wrong;
                }
            done += T;
            end = pos0 + done;
        }
        return true;
    }
    // the serve loop: the shared start (the last token read again), the prompt, its last token alone, the answer
    // (`gen` tokens; drafts: windows of 1 + k, `acc` of the drafts taken).  Returns where the prompt was read from.
    int64_t request(const Toks& ids, int64_t gen, uint64_t salt, std::mt19937* rng = nullptr, int k = 0) {
        int64_t from = 0;
        while (from < (int64_t) state.size() && from + 1 < (int64_t) ids.size() &&
               state[(size_t) from] == ids[(size_t) from])
            ++from;
        if (from > 0 && !resumable(from)) {
            from = 0;
            ++refused;
        }
        state.assign(ids.begin(), ids.begin() + from);
        const int64_t body = (int64_t) ids.size() - 1;
        for (int64_t a = from; a < body; a += chunk) {
            const int64_t T = std::min(chunk, body - a);
            if (!forward(ids.data() + a, T, a)) return -1;
        }
        if (!forward(ids.data() + body, 1, body)) return -1;
        state = ids;
        uint64_t tok = salt * 1000003u + 7;
        for (int64_t i = 0; i < gen;) {
            const int64_t pos = (int64_t) state.size();
            Toks w{tok};
            int acc = 0;
            if (rng && k > 0) {   // (a draft window: k drafts after the token, acc of them right)
                for (int j = 0; j < k; ++j) w.push_back(tok * 31 + (uint64_t) j + 1);
                acc = (int) ((*rng)() % (uint64_t) (k + 1));
            }
            if (!forward(w.data(), (int64_t) w.size(), pos)) return -1;
            for (int j = 0; j <= acc && i < gen; ++j, ++i) state.push_back(w[(size_t) j]);
            tok = tok * 6364136223846793005ull + 1442695040888963407ull;
        }
        return from;
    }
};

// a text's first n tokens and a question of its own
Toks prompt(int64_t n, uint64_t q) {
    Toks v;
    for (int64_t i = 0; i < n; ++i) v.push_back(0x9e3779b97f4a7c15ull * (uint64_t) (i + 1));
    v.push_back(1000 + q);
    v.push_back(2000 + q);
    return v;
}

// the issue's order (#1091) at small sizes: window 8, rows 8 + 16, chunks of 8
int issue_order(bool old, long long& wrong, int64_t& c2_from) {
    Sim s(24, 8, 8);
    s.old = old;
    const Toks A = prompt(54, 1), B = prompt(42, 2), C = prompt(30, 3);
    const int64_t f1 = s.request(B, 2, 1), f2 = s.request(C, 3, 2), f3 = s.request(A, 2, 3), f4 = s.request(B, 2, 4);
    CHECK(f1 == 0 && f2 == 30 && f3 == 30 && f4 == 42);   // (C, A and B reuse as in the issue's table)
    CHECK(s.wrong == 0);
    c2_from = s.request(C, 3, 5);
    wrong = s.wrong;
    return 0;
}

}  // namespace

int main() {
    long long wrong = 0;
    int64_t from = -1;
    CHECK(issue_order(true, wrong, from) == 0);
    CHECK(from == 30 && wrong > 0);   // (the old check: the second C resumed over rows A wrote over - the bug)
    CHECK(issue_order(false, wrong, from) == 0);
    CHECK(from == 0 && wrong == 0);   // (ring_lo: refused, read from 0, every row right)

    // random requests over prefixes of one text, rewinds of every depth, sometimes drafts
    std::mt19937 rng(1091);
    long long refusals = 0, resumes = 0;
    for (int run = 0; run < 400; ++run) {
        const int64_t w = 2 + (int64_t) (rng() % 12), c = 1 + (int64_t) (rng() % 12);
        const int64_t rows = w + std::max<int64_t>((int64_t) (rng() % 24), std::max<int64_t>(c, 4));   // (as the
        // engine's: n_swa + max(slack, its chunk), the chunk at least a draft window)
        Sim s(rows, w, c);
        for (int r = 0; r < 30; ++r) {
            const Toks p = prompt(1 + (int64_t) (rng() % (uint64_t) (5 * rows)), rng() % 3);
            const int k = rng() % 2 ? (int) (rng() % 4) : 0;
            const int64_t f = s.request(p, (int64_t) (rng() % 8), (uint64_t) r, &rng, k);
            CHECK(f >= 0);
            resumes += f > 0;
            CHECK(s.wrong == 0);
            for (int64_t q = std::max(s.ring_lo, s.end - rows); q < s.end; ++q) CHECK(s.slot_ok(q));   // (all held)
            for (int64_t pos = 0; pos <= s.end; ++pos) CHECK(s.resumable(pos) == s.really_holds(pos));
        }
        refusals += s.refused;
    }
    CHECK(refusals > 0 && resumes > 0);

    // continuations alone: each request the state and more; drafts rejected rewind a few positions
    long long old_wrong = 0;
    for (int run = 0; run < 200; ++run) {
        const int64_t w = 2 + (int64_t) (rng() % 12), c = 1 + (int64_t) (rng() % 12);
        const int64_t rows = w + std::max<int64_t>((int64_t) (rng() % 24), std::max<int64_t>(c, 4));   // (as the
        // engine's: n_swa + max(slack, its chunk), the chunk at least a draft window)
        Sim s(rows, w, c);
        Toks p = prompt(1 + (int64_t) (rng() % 20), 0);
        for (int r = 0; r < 20; ++r) {
            const int k = (int) (rng() % 4);
            CHECK(s.request(p, (int64_t) (rng() % 12), (uint64_t) r, &rng, k) >= 0);
            CHECK(s.wrong == 0 && s.refused == 0);
            for (int64_t pos = 0; pos <= s.end; ++pos) {
                const bool was = pos == 0 || pos - w + 1 >= s.end - rows;
                CHECK(!was || s.resumable(pos) || !s.really_holds(pos));
                old_wrong += was && !s.really_holds(pos);
            }
            p = s.state;
            for (int64_t i = 0, n = 1 + (int64_t) (rng() % (uint64_t) (2 * rows)); i < n; ++i) p.push_back(rng());
        }
    }
    std::printf("hy4_ring_test: ok (random: %lld resumes refused, %lld taken; continuations: the old check would take "
                "%lld positions whose rows rejected drafts wrote over)\n", refusals, resumes, old_wrong);
    return 0;
}
