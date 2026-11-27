// maya-hy4's restore points (src/hy4/hy4_snap.hpp; no GPU): the rings of tests/hy4_ring_test.cpp, slot by slot, with
// the points kept as Model::forward and the serve loop keep them - taken before each window from snap_first on (the
// rows [s - n_swa, s) copied out), dropped past a forward's start, and put back by snap_pick where Model::resumable
// says the rings no longer hold the window before a prompt's split.  Every read of a window row is checked against
// its slot: a row of another position or prefix is a wrong answer.
//  - a long text, then a question about its middle: read from the latest point before the split, not from 0
//  - snap_pick takes no point whose sequence is not the prompt's start, nor one past the split or the sequence's end
//  - random requests over prefixes of one text, rewinds of every depth, draft windows: no wrong read, every point
//    copied from rows its sequence holds, resumable never claims a row that is not there, and fewer tokens read
//    than the same requests without points
#include "../src/hy4/hy4_ring.hpp"
#include "../src/hy4/hy4_snap.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

using Toks = std::vector<int>;

struct Sim {
    int64_t rows, n_swa, chunk, step;
    bool points = true;
    std::vector<int64_t> slot_pos;
    std::vector<uint64_t> slot_tok, hs;   // (the slots' prefixes; the engine's tokens' prefixes)
    Toks seq, state;                      // (the engine's tokens [0, end); the serve loop's state)
    int64_t end = 0;
    int64_t ring_lo = 0;
    std::vector<hy4::SnapPoint> pts;
    std::vector<std::vector<std::pair<int64_t, uint64_t>>> kept;   // (each point's rows: the host copy)
    long long wrong = 0, refused = 0, restored = 0, bad_take = 0, read = 0;

    Sim(int64_t r, int64_t w, int64_t c, int64_t st, int64_t max_ctx)
        : rows(r), n_swa(w), chunk(c), step(st), slot_pos((size_t) r, -1), slot_tok((size_t) r),
          pts((size_t) (max_ctx / st)), kept((size_t) (max_ctx / st)) {}

    int64_t back() const { return n_swa - 1; }   // (the engine's max(n_swa, ix_kpool) - 1: the pool within the window)
    bool resumable(int64_t pos) const { return hy4::ring_holds(pos, end, ring_lo, rows, back()); }
    bool slot_ok(int64_t q) const {
        const size_t s = (size_t) (q % rows);
        return slot_pos[s] == q && slot_tok[s] == hs[(size_t) q];
    }
    bool really_holds(int64_t pos) const {
        if (pos == 0) return true;
        if (pos > end) return false;
        for (int64_t q = std::max<int64_t>(0, pos - back()); q < pos; ++q)
            if (!slot_ok(q)) return false;
        return true;
    }
    // Model::Impl::snap_take: before a window at q
    void take(int64_t q) {
        if (!points) return;
        for (int64_t s = hy4::snap_first(q, ring_lo, rows, n_swa, step); s <= q; s += step) {
            const size_t i = (size_t) (s / step - 1);
            if (i >= pts.size()) break;
            if (pts[i].at == s) continue;
            kept[i].clear();
            for (int64_t p = s - n_swa; p < s; ++p) {
                bad_take += !slot_ok(p);
                kept[i].emplace_back(slot_pos[(size_t) (p % rows)], slot_tok[(size_t) (p % rows)]);
            }
            pts[i] = hy4::SnapPoint{s, hy4::snap_hash(seq.data(), s)};
        }
    }
    // Model::Impl::snap_restore
    int64_t restore(const Toks& ids, int64_t upto) {
        const int64_t i = hy4::snap_pick(pts, ids.data(), upto, end);
        if (i < 0) return 0;
        const int64_t s = pts[(size_t) i].at;
        for (int64_t p = s - n_swa, k = 0; p < s; ++p, ++k) {
            slot_pos[(size_t) (p % rows)] = kept[(size_t) i][(size_t) k].first;
            slot_tok[(size_t) (p % rows)] = kept[(size_t) i][(size_t) k].second;
        }
        hy4::snap_drop(pts, s);
        seq.resize((size_t) s);
        hs.resize((size_t) s);
        end = s;
        ring_lo = hy4::snap_ring_lo(s, n_swa);
        return s;
    }
    // Model::forward: n ids from pos0 (chunks of `chunk`), each chunk's points taken, its rows stored, its windows read
    bool forward(const int* ids, int64_t n, int64_t pos0) {
        if (!resumable(pos0)) return false;
        ring_lo = hy4::ring_lo_at(pos0, ring_lo, end, rows);
        hy4::snap_drop(pts, pos0);
        seq.resize((size_t) pos0);
        seq.insert(seq.end(), ids, ids + n);
        hs.resize((size_t) pos0);
        for (int64_t i = 0; i < n; ++i)
            hs.push_back(((pos0 + i ? hs[(size_t) (pos0 + i - 1)] : 0) ^ (uint64_t) (uint32_t) ids[i]) *
                         0x100000001b3ull + 0x9e37ull);
        end = pos0;
        for (int64_t done = 0; done < n;) {
            const int64_t T = std::min(chunk, n - done), p0 = pos0 + done;
            take(p0);
            for (int64_t t = 0; t < T; ++t) {
                slot_pos[(size_t) ((p0 + t) % rows)] = p0 + t;
                slot_tok[(size_t) ((p0 + t) % rows)] = hs[(size_t) (p0 + t)];
            }
            for (int64_t t = 0; t < T; ++t)
                for (int64_t q = std::max<int64_t>(0, p0 + t - back()); q <= p0 + t; ++q)
                    if (!slot_ok(q)) ++wrong;
            done += T;
            end = pos0 + done;
        }
        return true;
    }
    // the serve loop: the shared start, a restore point where the rings no longer hold it (else from 0), the prompt,
    // its last token alone, the answer (`gen` tokens; drafts: windows of 1 + k, some taken).  Returns where the prompt
    // was read from.
    int64_t request(const Toks& ids, int64_t gen, int salt, std::mt19937* rng = nullptr, int k = 0) {
        int64_t from = 0;
        while (from < (int64_t) state.size() && from + 1 < (int64_t) ids.size() &&
               state[(size_t) from] == ids[(size_t) from])
            ++from;
        if (from > 0 && !resumable(from)) {
            const int64_t at = points ? restore(ids, from) : 0;
            if (at > 0) {
                ++restored;
                if (!resumable(at) || !really_holds(at) || at > from) return -1;
            } else {
                ++refused;
            }
            from = at;
        }
        state.assign(ids.begin(), ids.begin() + from);
        const int64_t body = (int64_t) ids.size() - 1;
        read += (int64_t) ids.size() - from;
        for (int64_t a = from; a < body; a += chunk) {
            const int64_t T = std::min(chunk, body - a);
            if (!forward(ids.data() + a, T, a)) return -1;
        }
        if (!forward(ids.data() + body, 1, body)) return -1;
        state = ids;
        int tok = 3000000 + salt * 7919;
        for (int64_t i = 0; i < gen;) {
            const int64_t pos = (int64_t) state.size();
            Toks w{tok};
            int acc = 0;
            if (rng && k > 0) {   // (a draft window: k drafts after the token, acc of them right)
                for (int j = 0; j < k; ++j) w.push_back(tok + 100 * (j + 1));
                acc = (int) ((*rng)() % (uint64_t) (k + 1));
            }
            if (!forward(w.data(), (int64_t) w.size(), pos)) return -1;
            for (int j = 0; j <= acc && i < gen; ++j, ++i) state.push_back(w[(size_t) j]);
            tok += 13;
        }
        return from;
    }
};

// a text's first n tokens and a question of its own
Toks prompt(int64_t n, int q) {
    Toks v;
    for (int64_t i = 0; i < n; ++i) v.push_back((int) ((uint64_t) (i + 1) * 2654435761u % 1000003u));
    v.push_back(2000000 + q);
    v.push_back(2100000 + q);
    return v;
}

int pick_rules() {
    const Toks a = prompt(64, 1);
    Toks b = a;
    b[20] = 7;   // (a start that leaves a's at 20)
    std::vector<hy4::SnapPoint> pts(4);   // (points 16, 32, 48, 64)
    pts[0] = {16, hy4::snap_hash(a.data(), 16)};
    pts[1] = {32, hy4::snap_hash(a.data(), 32)};
    pts[3] = {64, hy4::snap_hash(a.data(), 64)};   // (48 not held)
    CHECK(hy4::snap_pick(pts, a.data(), 66, 66) == 3);
    CHECK(hy4::snap_pick(pts, a.data(), 63, 66) == 1);   // (none past the split)
    CHECK(hy4::snap_pick(pts, a.data(), 66, 40) == 1);   // (none past the sequence's end)
    CHECK(hy4::snap_pick(pts, a.data(), 15, 66) == -1);
    CHECK(hy4::snap_pick(pts, b.data(), 66, 66) == 0);   // (32 and 64 are a's start, not b's)
    hy4::snap_drop(pts, 40);
    CHECK(pts[0].at == 16 && pts[1].at == 32 && pts[3].at == -1);
    // the first point a window may take: its rows [s - 4, s) held
    CHECK(hy4::snap_first(40, 0, 12, 4, 8) == 32);    // (held [28, 40): 32's rows [28, 32))
    CHECK(hy4::snap_first(40, 30, 12, 4, 8) == 40);   // (ring_lo 30: 32's rows not all held)
    CHECK(hy4::snap_first(41, 0, 12, 4, 8) == 40);    // (held [29, 41): 32's rows start at 28, written over)
    CHECK(hy4::snap_first(3, 0, 12, 4, 8) == 8);      // (none before the first step)
    return 0;
}

// a long text, then a question about its middle (window 8, rows 8 + 16, chunks of 8, a point every 16)
int middle(bool points, int64_t& from, long long& wrong) {
    Sim s(24, 8, 8, 16, 512);
    s.points = points;
    CHECK(s.request(prompt(200, 1), 5, 1) == 0);
    for (int64_t p = 16; p <= 192; p += 16) CHECK(!points || s.pts[(size_t) (p / 16 - 1)].at == p);   // (every one)
    from = s.request(prompt(100, 2), 5, 2);   // (shares 100 tokens: the rings hold the last 24 only)
    wrong = s.wrong + s.bad_take;
    CHECK(from >= 0);
    CHECK(s.request(prompt(200, 1), 5, 3) >= 0);   // (back to the long text: from the point the question left)
    wrong += s.wrong + s.bad_take;
    return 0;
}

}  // namespace

int main() {
    CHECK(pick_rules() == 0);
    int64_t from = -1;
    long long wrong = 0;
    CHECK(middle(false, from, wrong) == 0);
    CHECK(from == 0 && wrong == 0);   // (without points: read again from 0)
    CHECK(middle(true, from, wrong) == 0);
    CHECK(from == 96 && wrong == 0);   // (the latest point at or before the split, 100)

    // random requests over prefixes of one text, rewinds of every depth, sometimes drafts - and the same requests
    // without points
    std::mt19937 rng(2048);
    long long restores = 0, refusals = 0, read = 0, read_without = 0;
    for (int run = 0; run < 400; ++run) {
        const int64_t w = 2 + (int64_t) (rng() % 12), c = 1 + (int64_t) (rng() % 12);
        const int64_t rows = w + std::max<int64_t>((int64_t) (rng() % 24), std::max<int64_t>(c, 4));   // (as the
        // engine's: n_swa + max(slack, its chunk), the chunk at least a draft window)
        const int64_t kpool = 1 + (int64_t) (rng() % (uint64_t) w);   // (the indexer's pool, within the window)
        const int64_t step = (w + kpool - 1) / kpool * kpool * (1 + (int64_t) (rng() % 3));   // (a multiple of the
        // pool, at least the window)
        Sim s(rows, w, c, step, 8 * rows + 64), t(rows, w, c, step, 8 * rows + 64);
        t.points = false;
        const uint32_t seed = rng();
        std::mt19937 r1(seed), r2(seed);
        for (int r = 0; r < 30; ++r) {
            const int64_t n = 1 + (int64_t) (rng() % (uint64_t) (6 * rows));   // (one draw a statement: the same
            const int q = (int) (rng() % 3);                                   //  requests with any compiler)
            const int k = rng() % 2 ? (int) (rng() % 4) : 0;
            const int64_t gen = (int64_t) (rng() % 8);
            const Toks p = prompt(n, q);
            CHECK(s.request(p, gen, r, &r1, k) >= 0);
            CHECK(t.request(p, gen, r, &r2, k) >= 0);
            CHECK(s.wrong == 0 && s.bad_take == 0 && t.wrong == 0);
            CHECK(s.state == t.state);
            for (int64_t q = std::max(s.ring_lo, s.end - rows); q < s.end; ++q) CHECK(s.slot_ok(q));   // (all held)
            for (int64_t pos = 0; pos <= s.end; ++pos) CHECK(!s.resumable(pos) || s.really_holds(pos));
        }
        restores += s.restored;
        refusals += t.refused;
        read += s.read;
        read_without += t.read;
    }
    CHECK(restores > 0 && read < read_without);
    std::printf("hy4_snap_test: ok (random: %lld prompts read from a restore point where %lld were read from 0 without "
                "points; %lld tokens read against %lld)\n", restores, refusals, read, read_without);
    return 0;
}
