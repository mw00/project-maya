// maya-hy4's conversation slots (src/hy4/hy4_slots.hpp, SLOTS, MAYA_HY4_SLOT_KEEP=1; no GPU, no model): the
// table of parked conversations and its switch rule, driven with made-up token lists.  Written from the header's
// interface comment and the slots design (its 3.2 matching / park / take rule and 5's test plan), not from the code:
//  - shared_start: the longest common prefix, at most ids.size() - 1 (a prompt's last token is read again), cut at a
//    picture before it that is not in both at the same place with the same hash - have's and want's, either order
//  - usable: c where hy4::ring_holds(c, n, lo, rows, back), else 0 - exactly at the ring's lowest held position, one
//    before it, the end - rows bound, n <= rows (a window clamped at 0), c past n; and a sweep against ring_holds
//  - Table::best: the longest usable resume (> 0); ties go to the most recently used (touch / put); none: npos, 0
//  - Table::put: an older slot whose tokens are a prefix of the new one's (pictures agreeing before its end) is dropped
//    (supersede), then the least recently used past the cap or past ram_bytes (with more than one) - never the new
//    slot, never the protected one, so those two may stay over the caps
//  - decide: take only when best.len > from_cur + gain; park when state_n >= min, not held, and a take or an in-place
//    rewind dropping >= min tokens; a plain continuation neither parks nor (short of the gain) takes - and with a slot
//    past the gain it TAKES WITHOUT PARKING (P2: the state is a prefix of that slot)
//  - v2, the folder (MAYA_HY4_SLOT_DIR): put's caps count RAM slots alone - past them the LRU in RAM with a file
//    spills (keeps its file, Put::spilled), one without is dropped; Put::gone lists every slot put drops; in_ram /
//    bytes; best skips a slot whose bytes are only in a running write; add() appends the folder's slots as the most
//    recently used, no caps; trim_folder never removes keep_id / keep2 / one being written, a RAM slot loses only its
//    file; find_id / live; checksum (a flipped bit, alignment, lengths); meta_encode / meta_decode round trip, every
//    truncation and both magics refused, bounds; file_kind; the v2 knobs (DIR set / empty, GB, DAYS, IDLE_S, QUIT_S)
//  - the .meta's HY4META2 layout read field by field as documented, its CRC-32 (zlib's, checked against "123456789"),
//    HY4META1 refused, a flipped byte anywhere refused, randomly damaged buffers (every one changed) never decoded, a
//    trailing byte refused, the crc's high half zero, and a rewritten identity (or token) with the crc recomputed
//    decoded as well formed with what was written
//  - knobs_from: keep off by default (on() false), the documented defaults, valid values read, invalid ones keep the
//    default and warn
//  - random tables (seeded): shared_start, best and put against the contract's rules written out here (a slot's
//    pictures lie inside its tokens, as a real state's do)
// Failures are counted, not fatal: every broken clause is printed (with the contract line it encodes).
#include "../src/hy4/hy4_ring.hpp"
#include "../src/hy4/hy4_slots.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

int g_fail = 0, g_checks = 0;
long long g_stats[4] = {};   // (random tables: requests with a usable slot, supersedes, evictions, protected over cap)
#define CHECK(expr, clause)                                                                                          \
    do {                                                                                                             \
        ++g_checks;                                                                                                  \
        if (!(expr)) {                                                                                               \
            ++g_fail;                                                                                                \
            std::fprintf(stderr, "FAIL line %d: %s\n      contract: %s\n", __LINE__, #expr, clause);                \
        }                                                                                                            \
    } while (0)

using namespace hy4::slots;
using Toks = std::vector<int>;
using Pics = std::vector<Pic>;

Toks seq(int n, int base = 1) {
    Toks v;
    for (int i = 0; i < n; ++i) v.push_back(base + i);
    return v;
}
Toks cat(Toks a, const Toks& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
Pic pic(int64_t start, uint64_t hash) {
    Pic p{};
    p.start = start;
    p.hash = hash;
    return p;
}

// a slot of these tokens; its bytes a vector of `bytes` (null data when bytes < 0: `size` alone)
Slot slot(const Toks& t, int64_t lo, int64_t bytes, const Pics& pics = {}) {
    Slot s{};
    s.tokens = t;
    s.pics = pics;
    s.lo = lo;
    if (bytes >= 0) s.data = std::make_shared<std::vector<uint8_t>>((size_t) bytes);
    s.size = (size_t) (bytes < 0 ? 0 : bytes);
    return s;
}

Knobs knobs_on(int64_t cap, uint64_t ram_bytes) {
    Knobs k{};
    k.keep = true;
    k.cap = cap;
    k.ram_bytes = ram_bytes;
    k.min = 512;
    k.gain = 256;
    return k;
}

// which of `names` (each a slot's first token) the table holds
std::string held_names(const Table& t) {
    std::string s;
    for (size_t i = 0; i < t.size(); ++i) s += (char) ('A' + (t[i].tokens.empty() ? 0 : t[i].tokens[0] / 1000));
    std::sort(s.begin(), s.end());
    return s;
}
// tokens of conversation `c` ('A' ...), n long: c * 1000 + i (held_names reads the letter back)
Toks conv(char c, int n) { return seq(n, (c - 'A') * 1000 + 1); }

// ---- shared_start ------------------------------------------------------------------------------------------------
void test_shared_start() {
    const char* lcp = "shared_start: the longest common prefix, at most ids.size() - 1";
    CHECK(shared_start(seq(10), {}, seq(10), {}) == 9, lcp);                 // identical: the last token read again
    CHECK(shared_start(seq(3), {}, seq(5), {}) == 3, lcp);                   // have a prefix of ids
    CHECK(shared_start(seq(20), {}, seq(5), {}) == 4, lcp);                  // ids a prefix of have: capped
    CHECK(shared_start(Toks{1, 2, 3, 9}, {}, seq(5), {}) == 3, lcp);         // diverge at 3
    CHECK(shared_start(Toks{7}, {}, seq(5), {}) == 0, lcp);                  // diverge at 0
    CHECK(shared_start({}, {}, seq(5), {}) == 0, lcp);                       // nothing held
    CHECK(shared_start(seq(5), {}, seq(1), {}) == 0, lcp);                   // a one-token prompt: 0
    CHECK(shared_start(seq(5), {}, seq(6), {}) == 5, lcp);                   // have whole, ids one longer

    const char* pc = "shared_start: cut at any picture before it not in both at the same place with the same hash";
    const Toks t = seq(100);
    CHECK(shared_start(t, {pic(10, 7)}, t, {pic(10, 7)}) == 99, pc);         // same place, same hash: kept
    CHECK(shared_start(t, {pic(10, 7)}, t, {pic(10, 8)}) == 10, pc);         // same place, other hash: cut
    CHECK(shared_start(t, {pic(10, 7)}, t, {}) == 10, pc);                   // only in have
    CHECK(shared_start(t, {}, t, {pic(10, 7)}) == 10, pc);                   // only in want
    CHECK(shared_start(t, {pic(10, 7)}, t, {pic(12, 7)}) == 10, pc);         // same hash, other place: the first
    // both orders: have's stray picture first, then want's - and want's first, then have's: the earlier start wins
    CHECK(shared_start(t, {pic(20, 1)}, t, {pic(40, 2)}) == 20, pc);
    CHECK(shared_start(t, {pic(40, 1)}, t, {pic(20, 2)}) == 20, pc);
    CHECK(shared_start(t, {pic(5, 3), pic(40, 1)}, t, {pic(5, 3), pic(20, 2)}) == 20, pc);   // a shared one kept
    CHECK(shared_start(t, {pic(5, 3), pic(60, 1)}, t, {pic(5, 4), pic(60, 1)}) == 5, pc);
    // a stray picture at or after the common prefix does not cut it ("before it")
    const Toks d = cat(seq(30), Toks{-1, -2});
    CHECK(shared_start(d, {pic(30, 9)}, seq(50), {}) == 30, pc);             // at the divergence: the prefix stands
    CHECK(shared_start(t, {pic(99, 9)}, t, {}) == 99, pc);                   // at ids.size() - 1: not before it
    CHECK(shared_start(t, {pic(98, 9)}, t, {}) == 98, pc);                   // one before: cut there
    CHECK(shared_start(t, {pic(0, 9)}, t, {pic(0, 8)}) == 0, pc);            // a picture first: nothing shared
}

// ---- usable ------------------------------------------------------------------------------------------------------
void test_usable() {
    const char* u = "usable(c, n, lo, geo): c when ring_holds(c, n, lo, rows, back), else 0";
    Geometry g{};
    g.rows = 300;
    g.back = 50;
    // a state of 1000 tokens whose rings hold [700, 1000): a resume at c reads [c - 50, c)
    CHECK(usable(0, 1000, 700, g) == 0, u);
    CHECK(usable(750, 1000, 700, g) == 750, u);              // exactly at lo: c - back == lo
    CHECK(usable(749, 1000, 700, g) == 0, u);                // one before: refused (0, a read from 0)
    CHECK(usable(1000, 1000, 700, g) == 1000, u);            // at the end
    CHECK(usable(1001, 1000, 700, g) == 0, u);               // past the end
    CHECK(usable(900, 1000, 700, g) == 900, u);
    // lo below end - rows: the end - rows bound decides ("the rings' rows of [lo, n)" are at most rows of them)
    CHECK(usable(750, 1000, 0, g) == 750, u);
    CHECK(usable(749, 1000, 0, g) == 0, u);
    // lo above end - rows (a trimmed ring, MAYA_HY4_SLOT_RING, or rejected drafts): lo decides
    CHECK(usable(850, 1000, 800, g) == 850, u);
    CHECK(usable(849, 1000, 800, g) == 0, u);
    // n <= rows: everything held; the window clamped at 0 (a resume before `back` tokens)
    CHECK(usable(30, 100, 0, g) == 30, u);
    CHECK(usable(1, 100, 0, g) == 1, u);
    CHECK(usable(100, 100, 0, g) == 100, u);
    CHECK(usable(300, 300, 0, g) == 300, u);                 // n == rows exactly
    CHECK(usable(50, 301, 0, g) == 0, u);                    // n = rows + 1: position 0 gone, c - back == 0 < 1
    CHECK(usable(51, 301, 1, g) == 51, u);
    // a sweep: usable is ring_holds' answer, c or 0
    std::mt19937 rng(4040);
    long long bad = 0;
    for (int i = 0; i < 200000; ++i) {
        Geometry h{};
        h.rows = 1 + (int64_t) (rng() % 64);
        h.back = (int64_t) (rng() % (uint64_t) h.rows);
        const int64_t n = (int64_t) (rng() % 200), lo = n ? (int64_t) (rng() % (uint64_t) (n + 1)) : 0;
        const int64_t c = (int64_t) (rng() % 210);
        const int64_t want = hy4::ring_holds(c, n, lo, h.rows, h.back) ? c : 0;
        bad += (int64_t) usable(c, n, lo, h) != want;
    }
    CHECK(bad == 0, u);
}

// ---- Table::best -------------------------------------------------------------------------------------------------
void test_best() {
    Geometry g{};
    g.rows = 1 << 20;
    g.back = 8;
    const char* lg = "best: the slot whose usable() resume is the longest (len > 0)";
    const char* ti = "best: ties go to the most recently used";
    const char* no = "best: index npos / len 0 when none";
    {
        Table t(knobs_on(8, 0));
        const Match m0 = t.best(seq(100), {}, g);
        CHECK(m0.index == Match::npos && m0.len == 0, no);                    // an empty table
        const Toks base = seq(200);
        t.put(slot(cat(seq(100), seq(50, 5000)), 0, 10), nullptr);           // shares 100 with the request
        t.put(slot(cat(seq(150), seq(50, 6000)), 0, 10), nullptr);           // shares 150
        t.put(slot(seq(80, 9000), 0, 10), nullptr);                          // shares nothing
        const Match m = t.best(base, {}, g);
        CHECK(m.index != Match::npos && m.len == 150, lg);
        CHECK(m.index != Match::npos && m.index < t.size() && t[m.index].tokens.size() > 150 &&
                  t[m.index].tokens[149] == 150 && t[m.index].tokens[150] == 6000, lg);
        const Match none = t.best(seq(50, 7000), {}, g);
        CHECK(none.index == Match::npos && none.len == 0, no);               // shares nothing with any
    }
    {   // the longest common prefix is not the longest usable resume: its ring no longer holds the window
        Table t(knobs_on(8, 0));
        Geometry h{};
        h.rows = 100;
        h.back = 8;
        t.put(slot(cat(seq(400), seq(10, 5000)), 410 - 100, 10), nullptr);   // shares 400; rings hold [310, 410)
        t.put(slot(cat(seq(200), seq(10, 6000)), 210 - 100, 10), nullptr);   // shares 200; rings hold [110, 210)
        const Match m = t.best(seq(500), {}, h);                              // 400: 392 < 310? no: 392 >= 310 - fine
        CHECK(m.len == 400, lg);
        Table t2(knobs_on(8, 0));
        t2.put(slot(cat(seq(400), seq(200, 5000)), 600 - 100, 10), nullptr);  // shares 400; rings [500, 600): refused
        t2.put(slot(cat(seq(200), seq(10, 6000)), 210 - 100, 10), nullptr);   // shares 200; usable
        const Match m2 = t2.best(seq(500), {}, h);
        CHECK(m2.len == 200 && m2.index != Match::npos && t2[m2.index].tokens[200] == 6000, lg);
        Table t3(knobs_on(8, 0));
        t3.put(slot(cat(seq(400), seq(200, 5000)), 600 - 100, 10), nullptr);  // the only one, refused: none
        const Match m3 = t3.best(seq(500), {}, h);
        CHECK(m3.index == Match::npos && m3.len == 0, no);
    }
    {   // pictures: a slot whose picture differs is cut at it
        Table t(knobs_on(8, 0));
        t.put(slot(seq(300), 0, 10, {pic(100, 1)}), nullptr);
        t.put(slot(cat(seq(200), seq(100, 7000)), 0, 10, {pic(100, 2)}), nullptr);
        const Match m = t.best(seq(400), {pic(100, 2)}, g);
        CHECK(m.len == 200 && m.index != Match::npos && t[m.index].tokens[200] == 7000, lg);
    }
    {   // ties: the most recently used; touch makes a slot the most recent
        Table t(knobs_on(8, 0));
        t.put(slot(cat(seq(100), Toks{-1}), 0, 10), nullptr);   // index 0 (parked first)
        t.put(slot(cat(seq(100), Toks{-2}), 0, 10), nullptr);   // index 1: the newer
        size_t i_old = Match::npos, i_new = Match::npos;
        for (size_t i = 0; i < t.size(); ++i) (t[i].tokens[100] == -1 ? i_old : i_new) = i;
        Match m = t.best(seq(150), {}, g);
        CHECK(m.len == 100 && m.index == i_new, ti);
        t.touch(i_old);
        m = t.best(seq(150), {}, g);
        CHECK(m.len == 100 && m.index == i_old, ti);
        t.touch(i_new);
        m = t.best(seq(150), {}, g);
        CHECK(m.len == 100 && m.index == i_new, ti);
    }
}

// ---- Table::put --------------------------------------------------------------------------------------------------
void test_put() {
    const char* su = "put: every older slot whose tokens are a prefix of the new one's (pictures agreeing) is dropped";
    const char* lru = "put: while more than cap slots, the least recently used goes - never the new one";
    const char* ram = "put: while more than ram_bytes over them all with more than one, the least recently used goes";
    const char* pr = "put: never `protect`; the new slot and the protected one may stay over the caps";
    const char* us = "put: the parked slot's `used` set to the newest";
    {   // supersede: an earlier state of this conversation
        Table t(knobs_on(8, 0));
        t.put(slot(conv('A', 100), 0, 10), nullptr);
        t.put(slot(conv('B', 100), 0, 10), nullptr);
        const Put p = t.put(slot(cat(conv('A', 100), Toks{-5, -6}), 0, 10), nullptr);
        CHECK(t.size() == 2, su);
        CHECK(p.superseded == 1 && p.evicted == 0, su);
        bool longer = false;
        for (size_t i = 0; i < t.size(); ++i) longer |= t[i].tokens.size() == 102;
        CHECK(longer, su);
        // identical tokens: a prefix too
        const Put q = t.put(slot(conv('B', 100), 0, 10), nullptr);
        CHECK(q.superseded == 1 && t.size() == 2, su);
        // a shorter state of a kept conversation does not drop the longer (the older is not its prefix)
        const Put r = t.put(slot(conv('A', 50), 0, 10), nullptr);
        CHECK(r.superseded == 0 && t.size() == 3, su);
        // two earlier states at once
        Table t2(knobs_on(8, 0));
        t2.put(slot(conv('C', 10), 0, 10), nullptr);
        t2.put(slot(conv('C', 20), 0, 10), nullptr);   // supersedes the 10
        t2.put(slot(conv('D', 20), 0, 10), nullptr);
        const Put s = t2.put(slot(conv('C', 40), 0, 10), nullptr);
        CHECK(s.superseded == 1 && t2.size() == 2 && held_names(t2) == "CD", su);
    }
    {   // supersede and pictures: an older slot whose picture before its end is not the new one's stays
        Table t(knobs_on(8, 0));
        t.put(slot(conv('A', 100), 0, 10, {pic(10, 1)}), nullptr);
        const Put p = t.put(slot(conv('A', 200), 0, 10, {pic(10, 2)}), nullptr);
        CHECK(p.superseded == 0 && t.size() == 2, su);
        const Put q = t.put(slot(conv('A', 300), 0, 10, {pic(10, 1)}), nullptr);   // agrees with the first only
        CHECK(q.superseded == 1 && t.size() == 2, su);
        // a picture of the new one past the older's end does not matter
        Table t2(knobs_on(8, 0));
        t2.put(slot(conv('A', 100), 0, 10, {pic(10, 1)}), nullptr);
        const Put r = t2.put(slot(conv('A', 200), 0, 10, {pic(10, 1), pic(150, 3)}), nullptr);
        CHECK(r.superseded == 1 && t2.size() == 1, su);
    }
    {   // LRU past the cap
        Table t(knobs_on(2, 0));
        t.put(slot(conv('A', 10), 0, 10), nullptr);
        t.put(slot(conv('B', 10), 0, 10), nullptr);
        const Put p = t.put(slot(conv('C', 10), 0, 10), nullptr);
        CHECK(p.evicted == 1 && t.size() == 2 && held_names(t) == "BC", lru);
        // a take (touch) makes B the newest: D's park drops C
        for (size_t i = 0; i < t.size(); ++i)
            if (t[i].tokens[0] == conv('B', 1)[0]) t.touch(i);
        t.put(slot(conv('D', 10), 0, 10), nullptr);
        CHECK(t.size() == 2 && held_names(t) == "BD", lru);
        // cap 1: the new one stays, every other goes
        Table t1(knobs_on(1, 0));
        t1.put(slot(conv('A', 10), 0, 10), nullptr);
        const Put q = t1.put(slot(conv('B', 10), 0, 10), nullptr);
        CHECK(q.evicted == 1 && t1.size() == 1 && held_names(t1) == "B", lru);
    }
    {   // RAM: past ram_bytes over them all, with more than one
        Table t(knobs_on(8, 100));
        t.put(slot(conv('A', 10), 0, 60), nullptr);
        CHECK(t.ram_bytes() == 60, ram);
        const Put p = t.put(slot(conv('B', 10), 0, 60), nullptr);
        CHECK(p.evicted == 1 && held_names(t) == "B" && t.ram_bytes() == 60, ram);
        const Put q = t.put(slot(conv('C', 10), 0, 40), nullptr);   // 100: not more than ram_bytes
        CHECK(q.evicted == 0 && held_names(t) == "BC" && t.ram_bytes() == 100, ram);
        // one alone over the cap is kept (the new one never goes)
        Table t1(knobs_on(8, 100));
        t1.put(slot(conv('A', 10), 0, 30), nullptr);
        const Put r = t1.put(slot(conv('B', 10), 0, 500), nullptr);
        CHECK(r.evicted == 1 && held_names(t1) == "B" && t1.ram_bytes() == 500, ram);
        // `size` alone (data null) counts
        Table t2(knobs_on(8, 100));
        Slot a = slot(conv('A', 10), 0, -1);
        a.size = 70;
        Slot b = slot(conv('B', 10), 0, -1);
        b.size = 70;
        t2.put(a, nullptr);
        t2.put(b, nullptr);
        CHECK(held_names(t2) == "B" && t2.ram_bytes() == 70, ram);
        // ram 0: the count alone
        Table t3(knobs_on(3, 0));
        for (char c : {'A', 'B', 'C'}) t3.put(slot(conv(c, 10), 0, 1000), nullptr);
        CHECK(t3.size() == 3 && t3.ram_bytes() == 3000, ram);
    }
    {   // protect: the slot about to be taken back
        Table t(knobs_on(2, 0));
        t.put(slot(conv('A', 10), 0, 10), nullptr);
        t.put(slot(conv('B', 10), 0, 10), nullptr);
        size_t ia = t.find(t[0].tokens[0] == conv('A', 1)[0] ? t[0].data.get() : t[1].data.get());
        CHECK(ia != Match::npos && t[ia].tokens[0] == conv('A', 1)[0], "find(data): the slot holding those bytes");
        const auto* pa = t[ia].data.get();
        const Put p = t.put(slot(conv('C', 10), 0, 10), pa);   // A is the LRU but protected: B goes
        CHECK(p.evicted == 1 && held_names(t) == "AC", pr);
        CHECK(t.find(pa) != Match::npos, pr);
        // cap 1: the new one and the protected one both stay, over the cap
        Table t1(knobs_on(1, 0));
        t1.put(slot(conv('A', 10), 0, 10), nullptr);
        const auto* pa1 = t1[0].data.get();
        t1.put(slot(conv('B', 10), 0, 10), pa1);
        CHECK(t1.size() == 2 && held_names(t1) == "AB", pr);
        // RAM: the two alone over it - both stay
        Table t2(knobs_on(8, 100));
        t2.put(slot(conv('A', 10), 0, 80), nullptr);
        const auto* pa2 = t2[0].data.get();
        t2.put(slot(conv('B', 10), 0, 80), pa2);
        CHECK(t2.size() == 2 && held_names(t2) == "AB" && t2.ram_bytes() == 160, pr);
        // with a third: the unprotected older one goes, the two stay
        Table t3(knobs_on(2, 0));
        t3.put(slot(conv('A', 10), 0, 10), nullptr);
        t3.put(slot(conv('B', 10), 0, 10), nullptr);
        const auto* pb3 = t3[0].tokens[0] == conv('B', 1)[0] ? t3[0].data.get() : t3[1].data.get();
        t3.put(slot(conv('C', 10), 0, 10), pb3);
        CHECK(held_names(t3) == "BC", pr);
        // the serve loop's order: park with the take's slot protected, then best() again still finds it
        Geometry g{};
        g.rows = 1 << 20;
        g.back = 8;
        Table t4(knobs_on(1, 0));
        t4.put(slot(conv('A', 600), 0, 10), nullptr);
        const Match m = t4.best(cat(conv('A', 600), Toks{-9}), {}, g);
        CHECK(m.len == 600, pr);
        const auto* want = t4[m.index].data.get();
        t4.put(slot(conv('B', 600), 0, 10), want);
        const Match m2 = t4.best(cat(conv('A', 600), Toks{-9}), {}, g);
        CHECK(m2.len == 600 && m2.index != Match::npos && t4[m2.index].data.get() == want, pr);
        // unprotected, the same park drops it
        Table t5(knobs_on(1, 0));
        t5.put(slot(conv('A', 600), 0, 10), nullptr);
        t5.put(slot(conv('B', 600), 0, 10), nullptr);
        const Match m3 = t5.best(cat(conv('A', 600), Toks{-9}), {}, g);
        CHECK(m3.index == Match::npos && m3.len == 0, lru);
    }
    {   // used: the parked slot is the newest; drop
        Table t(knobs_on(8, 0));
        t.put(slot(conv('A', 10), 0, 10), nullptr);
        t.put(slot(conv('B', 10), 0, 10), nullptr);
        t.touch(0);
        t.put(slot(conv('C', 10), 0, 10), nullptr);
        size_t ic = Match::npos;
        for (size_t i = 0; i < t.size(); ++i)
            if (t[i].tokens[0] == conv('C', 1)[0]) ic = i;
        bool newest = ic != Match::npos;
        for (size_t i = 0; newest && i < t.size(); ++i)
            if (i != ic && !(t[i].used < t[ic].used)) newest = false;
        CHECK(newest, us);
        const size_t before = t.size();
        const uint64_t bytes = t.ram_bytes();
        t.drop(ic);
        CHECK(t.size() == before - 1 && t.ram_bytes() == bytes - 10, "drop(i): the slot gone, its bytes too");
        bool c_gone = true;
        for (size_t i = 0; i < t.size(); ++i) c_gone &= t[i].tokens[0] != conv('C', 1)[0];
        CHECK(c_gone, "drop(i): the slot gone, its bytes too");
    }
}

// ---- random tables against the contract's rules, written out here -----------------------------------------------
bool pic_eq(const Pic& a, const Pic& b) { return a.start == b.start && a.hash == b.hash; }

// the contract's shared_start: the LCP capped at ids.size() - 1, cut at a picture before it not in both
int64_t ref_shared(const Toks& have, const Pics& hp, const Toks& ids, const Pics& wp) {
    int64_t c = 0;
    while (c < (int64_t) have.size() && c + 1 < (int64_t) ids.size() && have[(size_t) c] == ids[(size_t) c]) ++c;
    for (const Pics* side : {&hp, &wp}) {
        const Pics& other = side == &hp ? wp : hp;
        for (const Pic& p : *side) {
            bool both = false;
            for (const Pic& q : other) both |= pic_eq(p, q);
            if (!both && p.start < c) c = p.start;
        }
    }
    return c;
}

void test_random() {
    const char* ss = "shared_start: LCP capped at ids.size() - 1, cut at the first picture not in both (random)";
    const char* bs = "best: the longest usable() resume, ties to the most recently used (random tables)";
    const char* ps = "put: supersede, then LRU past cap / ram_bytes, never the new or the protected one (random)";
    std::mt19937 rng(777);
    auto rtoks = [&](const Toks& base) {   // a prefix of base, then its own tail
        Toks t(base.begin(), base.begin() + (long) (rng() % (base.size() + 1)));
        for (int i = 0, n = (int) (rng() % 40); i < n; ++i) t.push_back((int) (rng() % 4) + 100000);
        return t;
    };
    auto rpics = [&](size_t n) {
        Pics p;
        for (int i = 0, k = (int) (rng() % 3); i < k && n; ++i) p.push_back(pic((int64_t) (rng() % n), rng() % 2));
        std::sort(p.begin(), p.end(), [](const Pic& a, const Pic& b) { return a.start < b.start; });
        return p;
    };
    const Toks text = seq(200);
    long long bad_ss = 0, bad_best = 0, bad_put = 0;
    for (int run = 0; run < 3000; ++run) {
        const Toks a = rtoks(text), b = rtoks(text);
        const Pics pa = rpics(a.size() + 1), pb = rpics(b.size() + 1);
        bad_ss += (int64_t) shared_start(a, pa, b, pb) != ref_shared(a, pa, b, pb);
    }
    for (int run = 0; run < 1500; ++run) {
        Geometry g{};
        g.rows = 20 + (int64_t) (rng() % 200);
        g.back = (int64_t) (rng() % 20);
        const int64_t cap = 1 + (int64_t) (rng() % 5);
        const uint64_t ram = rng() % 2 ? 0 : 50 + rng() % 200;
        Table t(knobs_on(cap, ram));
        for (int step = 0; step < 12; ++step) {
            // a request: best against a brute force over the table
            const Toks ids = rtoks(text);
            const Pics want = rpics(ids.size() + 1);
            const Match m = t.best(ids, want, g);
            int64_t top = 0;
            uint64_t top_used = 0;
            size_t top_i = Match::npos;
            for (size_t i = 0; i < t.size(); ++i) {
                const int64_t c = ref_shared(t[i].tokens, t[i].pics, ids, want);
                const int64_t u = hy4::ring_holds(c, (int64_t) t[i].tokens.size(), t[i].lo, g.rows, g.back) ? c : 0;
                if (u > 0 && (u > top || (u == top && t[i].used > top_used))) {
                    top = u;
                    top_used = (uint64_t) t[i].used;
                    top_i = i;
                }
            }
            bad_best += (int64_t) m.len != top || (top > 0 && m.index != top_i) || (top == 0 && m.index != Match::npos);
            if (m.index != Match::npos && rng() % 2) t.touch(m.index);
            // a park, sometimes protecting a slot
            const Toks st = rtoks(text);
            const int64_t n = (int64_t) st.size();
            Slot s = slot(st, std::max<int64_t>(0, n - (int64_t) (rng() % (uint64_t) (g.rows + 1))),
                          (int64_t) (10 + rng() % 120), rpics(st.size()));   // (a slot's pictures are in its tokens)
            const auto* newd = s.data.get();
            std::vector<const std::vector<uint8_t>*> before_d;
            std::vector<uint64_t> before_used, before_size;
            std::vector<bool> prefix_of_new;
            std::vector<std::pair<Toks, Pics>> before_tp;
            for (size_t i = 0; i < t.size(); ++i) {
                before_tp.emplace_back(t[i].tokens, t[i].pics);
                before_d.push_back(t[i].data.get());
                before_used.push_back((uint64_t) t[i].used);
                before_size.push_back((uint64_t) t[i].size);
                const Toks& o = t[i].tokens;
                bool pre = o.size() <= st.size() && std::equal(o.begin(), o.end(), st.begin());
                if (pre) {   // the older's pictures before its end are the new one's
                    Pics x, y;
                    for (const Pic& q : t[i].pics) if (q.start < (int64_t) o.size()) x.push_back(q);
                    for (const Pic& q : s.pics) if (q.start < (int64_t) o.size()) y.push_back(q);
                    pre = x.size() == y.size() && std::equal(x.begin(), x.end(), y.begin(), pic_eq);
                }
                prefix_of_new.push_back(pre);
            }
            // protect one that is not an earlier state of the new one (the serve loop's: the slot it takes back)
            const std::vector<uint8_t>* prot = nullptr;
            if (t.size() && rng() % 2) {
                const size_t i = rng() % t.size();
                if (!prefix_of_new[i]) prot = before_d[i];
            }
            // the contract's put, written out: supersede, then the least recently used while over a cap
            std::vector<size_t> keep;   // indices into before_*, in order of use
            for (size_t i = 0; i < before_d.size(); ++i) if (!prefix_of_new[i]) keep.push_back(i);
            std::sort(keep.begin(), keep.end(), [&](size_t x, size_t y) { return before_used[x] < before_used[y]; });
            uint64_t bytes = s.size;
            for (size_t i : keep) bytes += before_size[i];
            size_t count = keep.size() + 1, evicted = 0;
            std::vector<const std::vector<uint8_t>*> expect_gone;
            for (size_t j = 0; j < keep.size(); ++j) {
                const bool over = (int64_t) count > cap || (ram && bytes > ram && count > 1);
                if (!over) break;
                if (before_d[keep[j]] == prot) continue;
                expect_gone.push_back(before_d[keep[j]]);
                bytes -= before_size[keep[j]];
                --count;
                ++evicted;
            }
            size_t superseded = 0;
            for (bool x : prefix_of_new) superseded += x;
            const Put p = t.put(s, prot);
            bool ok = t.find(newd) != Match::npos && (!prot || t.find(prot) != Match::npos);
            ok &= (size_t) p.superseded == superseded && (size_t) p.evicted == evicted;
            ok &= t.size() == count && t.ram_bytes() == bytes;
            for (size_t i = 0; i < before_d.size(); ++i) {
                const bool gone = prefix_of_new[i] ||
                                  std::find(expect_gone.begin(), expect_gone.end(), before_d[i]) != expect_gone.end();
                ok &= (t.find(before_d[i]) == Match::npos) == gone;
            }
            for (size_t i = 0; i < t.size(); ++i)   // the new one the most recently used
                if (t[i].data.get() != newd) ok &= t[i].used < t[t.find(newd)].used;
            if (!ok && bad_put == 0)
                std::fprintf(stderr, "      (put: cap %lld ram %llu, %zu before, protect %d: got superseded %zu evicted "
                             "%zu size %zu; want %zu %zu %zu)\n", (long long) cap, (unsigned long long) ram,
                             before_d.size(), prot != nullptr, (size_t) p.superseded, (size_t) p.evicted, t.size(),
                             superseded, evicted, count);
            if (!ok && bad_put == 0 && std::getenv("HY4_SLOT_TEST_DEBUG")) {
                std::fprintf(stderr, "      new: %zu tokens, size %llu, pics", st.size(), (unsigned long long) s.size);
                for (const Pic& q : s.pics) std::fprintf(stderr, " (%lld,%llu)", (long long) q.start, (unsigned long long) q.hash);
                std::fprintf(stderr, "\n");
                for (size_t i = 0; i < before_d.size(); ++i) {
                    std::fprintf(stderr, "      old %zu: %zu tokens, prefix %d, used %llu, size %llu, pics", i,
                                 before_tp[i].first.size(), (int) prefix_of_new[i],
                                 (unsigned long long) before_used[i], (unsigned long long) before_size[i]);
                    for (const Pic& q : before_tp[i].second)
                        std::fprintf(stderr, " (%lld,%llu)", (long long) q.start, (unsigned long long) q.hash);
                    std::fprintf(stderr, "\n");
                }
            }
            bad_put += !ok;
            g_stats[0] += top > 0;
            g_stats[1] += (long long) superseded;
            g_stats[2] += (long long) evicted;
            g_stats[3] += prot && (int64_t) count > cap;
        }
    }
    CHECK(bad_ss == 0, ss);
    CHECK(bad_best == 0, bs);
    CHECK(bad_put == 0, ps);
}

// ---- decide ------------------------------------------------------------------------------------------------------
Match match(int64_t len) {
    Match m{};
    m.index = len > 0 ? 0 : Match::npos;
    m.len = len;
    return m;
}

void test_decide() {
    Knobs k = knobs_on(4, 0);   // min 512, gain 256
    const char* tk = "decide: take iff best.len > from_cur + knobs.gain";
    const char* pk = "decide: park iff state_n >= min, not held, and (a take or state_n - from_cur >= min)";
    const char* pc = "decide: a plain continuation (from_cur == state_n) neither parks nor, short of the gain, takes";
    // plain continuation
    Plan p = decide(2000, 2000, false, match(0), k);
    CHECK(!p.park && !p.take, pc);
    p = decide(2000, 2000, false, match(2000 + 256), k);
    CHECK(!p.park && !p.take, pc);
    p = decide(2000, 2000, false, match(2000 + 257), k);
    // (P2, the contract as settled: the state is a prefix of the longer slot - taken back, not set aside first)
    CHECK(p.take && !p.park, "decide: a plain continuation with a slot past the gain TAKES WITHOUT PARKING (P2)");
    p = decide(2000, 2000, false, match(100000), k);
    CHECK(p.take && !p.park, "decide: a plain continuation with a slot past the gain TAKES WITHOUT PARKING (P2)");
    p = decide(0, 0, false, match(0), k);
    CHECK(!p.park && !p.take, pc);
    // the take threshold, strictly greater
    p = decide(100, 10, false, match(10 + 256), k);
    CHECK(!p.take, tk);
    p = decide(100, 10, false, match(10 + 257), k);
    CHECK(p.take, tk);
    p = decide(5000, 0, false, match(256), k);
    CHECK(!p.take, tk);
    p = decide(5000, 0, false, match(257), k);
    CHECK(p.take, tk);
    p = decide(5000, 4000, false, match(4200), k);    // the slot covers more, not by the gain
    CHECK(!p.take, tk);
    p = decide(5000, 0, false, match(0), k);
    CHECK(!p.take, tk);
    // take and park together: a switch away from a long conversation
    p = decide(600, 10, false, match(300), k);
    CHECK(p.take && p.park, pk);                       // drop 590 too, but the take alone parks
    p = decide(600, 590, false, match(10000), k);      // drop 10 (< min), but a take: park
    CHECK(p.take && p.park, pk);
    // held: already in a slot - never parked again
    p = decide(600, 10, true, match(300), k);
    CHECK(p.take && !p.park, pk);
    p = decide(5000, 0, true, match(0), k);
    CHECK(!p.take && !p.park, pk);
    // short state: never parked
    p = decide(511, 0, false, match(10000), k);
    CHECK(p.take && !p.park, pk);
    p = decide(512, 0, false, match(10000), k);
    CHECK(p.take && p.park, pk);
    p = decide(511, 0, false, match(0), k);
    CHECK(!p.park && !p.take, pk);
    // in-place rewinds: park when they drop at least min
    p = decide(2000, 2000 - 511, false, match(0), k);
    CHECK(!p.park && !p.take, pk);
    p = decide(2000, 2000 - 512, false, match(0), k);
    CHECK(p.park && !p.take, pk);
    p = decide(2000, 0, false, match(0), k);
    CHECK(p.park && !p.take, pk);
    p = decide(512, 0, false, match(0), k);
    CHECK(p.park && !p.take, pk);
    p = decide(2000, 1000, false, match(1100), k);     // a slot not past the gain: the rewind alone parks
    CHECK(p.park && !p.take, pk);
    // the knobs move the thresholds
    Knobs k2 = k;
    k2.min = 100;
    k2.gain = 0;
    p = decide(150, 50, false, match(0), k2);
    CHECK(p.park && !p.take, pk);
    p = decide(150, 51, false, match(0), k2);
    CHECK(!p.park && !p.take, pk);
    p = decide(99, 0, false, match(0), k2);
    CHECK(!p.park, pk);
    p = decide(150, 140, false, match(141), k2);
    CHECK(p.take && p.park, tk);
    p = decide(150, 140, false, match(140), k2);
    CHECK(!p.take, tk);
}

// ---- knobs_from --------------------------------------------------------------------------------------------------
std::map<std::string, std::string> g_env;
const char* fake_env(const char* name) {
    const auto it = g_env.find(name);
    return it == g_env.end() ? nullptr : it->second.c_str();
}

void test_knobs() {
    const char* df = "knobs: MAYA_HY4_SLOT_KEEP off by default; CAP 4, RAM_GB 4, MIN 512, RING the ring, VERIFY off";
    const char* rd = "knobs: valid MAYA_HY4_SLOT_* values are read";
    const char* iv = "knobs: invalid numbers keep the default (warn gets a line)";
    const char* on = "knobs: on() is keep && cap > 0";
    std::string warn_text;
    std::string* warn = &warn_text;
    auto count_lines = [&] { return (int) std::count(warn_text.begin(), warn_text.end(), '\n') + (warn_text.empty() ||
                                                                                         warn_text.back() == '\n' ? 0 : 1); };
    g_env.clear();
    Knobs d = knobs_from(fake_env, warn);
    CHECK(!d.keep && !d.on(), df);
    CHECK(d.cap == 4, df);
    CHECK(d.ram_bytes == 4ull << 30 || d.ram_bytes == 4000000000ull, df);
    CHECK(d.min == 512, df);
    CHECK(d.gain == 256, df);
    CHECK(d.ring == 0, df);   // (the whole ring: conv_save's keep_ring 0)
    CHECK(!d.verify, df);
    CHECK(warn_text.empty(), df);
    // the process environment: none set here -> off
    unsetenv("MAYA_HY4_SLOT_KEEP");
    const Knobs e = knobs_from_env(warn);
    CHECK(!e.keep && !e.on(), df);

    g_env = {{"MAYA_HY4_SLOT_KEEP", "1"}};
    Knobs k = knobs_from(fake_env, warn);
    CHECK(k.keep && k.on() && k.cap == 4, on);
    g_env = {{"MAYA_HY4_SLOT_KEEP", "0"}};
    k = knobs_from(fake_env, warn);
    CHECK(!k.keep && !k.on(), on);
    g_env = {{"MAYA_HY4_SLOT_KEEP", "1"}, {"MAYA_HY4_SLOT_CAP", "0"}};
    k = knobs_from(fake_env, warn);
    CHECK(k.keep && k.cap == 0 && !k.on(), on);
    CHECK(warn_text.empty(), rd);

    g_env = {{"MAYA_HY4_SLOT_KEEP", "1"}, {"MAYA_HY4_SLOT_CAP", "7"}, {"MAYA_HY4_SLOT_RAM_GB", "2"},
             {"MAYA_HY4_SLOT_MIN", "1000"}, {"MAYA_HY4_SLOT_RING", "20000"}, {"MAYA_HY4_SLOT_VERIFY", "1"}};
    k = knobs_from(fake_env, warn);
    CHECK(k.keep && k.on(), rd);
    CHECK(k.cap == 7, rd);
    CHECK(k.ram_bytes == 2ull << 30 || k.ram_bytes == 2000000000ull, rd);
    CHECK(k.min == 1000, rd);
    CHECK(k.ring == 20000, rd);
    CHECK(k.verify, rd);
    CHECK(k.gain == 256, rd);
    CHECK(warn_text.empty(), rd);
    g_env = {{"MAYA_HY4_SLOT_KEEP", "1"}, {"MAYA_HY4_SLOT_RAM_GB", "0"}};
    k = knobs_from(fake_env, warn);
    CHECK(k.ram_bytes == 0 && k.on(), rd);   // (0: the count alone)
    g_env = {{"MAYA_HY4_SLOT_KEEP", "1"}, {"MAYA_HY4_SLOT_RAM_GB", "0.5"}};
    k = knobs_from(fake_env, warn);
    CHECK(k.ram_bytes == 1ull << 29 || k.ram_bytes == 500000000ull, rd);

    const char* bad[][2] = {{"MAYA_HY4_SLOT_CAP", "abc"}, {"MAYA_HY4_SLOT_CAP", "3x"}, {"MAYA_HY4_SLOT_CAP", "-2"},
                            {"MAYA_HY4_SLOT_RAM_GB", "lots"}, {"MAYA_HY4_SLOT_RAM_GB", "-1"},
                            {"MAYA_HY4_SLOT_MIN", "many"}, {"MAYA_HY4_SLOT_MIN", "-5"},
                            {"MAYA_HY4_SLOT_RING", "big"}, {"MAYA_HY4_SLOT_RING", "-100"}};
    for (const auto& b : bad) {
        g_env = {{"MAYA_HY4_SLOT_KEEP", "1"}, {b[0], b[1]}};
        warn_text.clear();   // ("warn gets a line": appended or assigned, either is the contract's)
        k = knobs_from(fake_env, warn);
        const bool kept = k.cap == d.cap && k.ram_bytes == d.ram_bytes && k.min == d.min && k.ring == d.ring;
        const bool said = count_lines() >= 1;
        if (!kept || !said) std::fprintf(stderr, "      (%s=%s)\n", b[0], b[1]);
        CHECK(kept, iv);
        CHECK(said, iv);
        CHECK(k.keep, iv);   // (the others still read)
    }
}

// ---- v2: the folder (MAYA_HY4_SLOT_DIR) - the table's side --------------------------------------------------------
// a slot with files: in RAM (bytes > 0) or the folder's alone (bytes < 0: no data); disk 1 being written, 2 written
Slot fslot(const Toks& t, int64_t bytes, int disk, uint64_t id, uint64_t file_bytes = 0) {
    Slot s = slot(t, 0, bytes);
    if (bytes < 0) s.size = 0;
    s.file = "/folder/slot-" + std::to_string(id);
    s.disk = disk;
    s.id = id;
    s.file_bytes = file_bytes ? file_bytes : (uint64_t) (bytes > 0 ? bytes : 100);
    s.sum = 0x1234 + id;
    return s;
}
size_t by_letter(const Table& t, char c) {
    for (size_t i = 0; i < t.size(); ++i)
        if (!t[i].tokens.empty() && t[i].tokens[0] == conv(c, 1)[0]) return i;
    return Match::npos;
}
bool gone_has(const Put& p, char c) {
    for (const Slot& g : p.gone)
        if (!g.tokens.empty() && g.tokens[0] == conv(c, 1)[0]) return true;
    return false;
}

void test_folder_table() {
    const char* sp = "put (v2): past the caps (RAM slots only) the LRU in RAM with a file gives up its bytes, stays "
                     "as the folder's (Put::spilled), keeps its file";
    const char* ev = "put (v2): past the caps the LRU in RAM without a file is dropped (Put::evicted)";
    const char* gn = "put (v2): every slot put drops (supersede, evict) is in Put::gone; a spilled one is not";
    const char* rc = "put (v2): CAP and RAM_GB count the slots in RAM alone (ram_count, ram_bytes)";
    const char* ir = "Slot (v2): in_ram when it has bytes or has neither bytes nor file; bytes() 0 for the folder's";
    {   // a spill: A (written) is the LRU when C comes over cap 2
        Table t(knobs_on(2, 0));
        t.put(fslot(conv('A', 10), 10, 2, 1), nullptr);
        t.put(fslot(conv('B', 10), 10, 2, 2), nullptr);
        const Put p = t.put(slot(conv('C', 10), 0, 10), nullptr);
        const size_t a = by_letter(t, 'A');
        CHECK(p.spilled == 1 && p.evicted == 0 && p.gone.empty(), sp);
        CHECK(t.size() == 3 && a != Match::npos, sp);
        CHECK(a != Match::npos && !t[a].in_ram() && t[a].bytes() == 0 && t[a].file == "/folder/slot-1" &&
                  t[a].disk == 2 && t[a].id == 1, sp);
        CHECK(t.ram_count() == 2 && t.ram_bytes() == 20, rc);
        CHECK(t.disk_bytes() == 20, "disk_bytes: the folder's slots' (disk 1 and 2) file bytes");
        // the next park: the folder's A does not count, B (written) spills next
        const Put q = t.put(slot(conv('D', 10), 0, 10), nullptr);
        CHECK(q.spilled == 1 && q.evicted == 0 && t.size() == 4 && t.ram_count() == 2, rc);
        const size_t b = by_letter(t, 'B');
        CHECK(b != Match::npos && !t[b].in_ram(), sp);
        // C has no file: the next one drops it
        const Put r = t.put(slot(conv('E', 10), 0, 10), nullptr);
        CHECK(r.evicted == 1 && r.spilled == 0 && gone_has(r, 'C') && by_letter(t, 'C') == Match::npos, ev);
        CHECK(t.size() == 4 && t.ram_count() == 2, rc);
    }
    {   // a slot being written (disk 1) has a file: it spills too, and stays disk 1
        Table t(knobs_on(1, 0));
        t.put(fslot(conv('A', 10), 10, 1, 7), nullptr);
        const Put p = t.put(slot(conv('B', 10), 0, 10), nullptr);
        const size_t a = by_letter(t, 'A');
        CHECK(p.spilled == 1 && a != Match::npos && !t[a].in_ram() && t[a].disk == 1, sp);
    }
    {   // RAM bytes: only RAM slots count
        Table t(knobs_on(8, 100));
        t.put(fslot(conv('A', 10), 60, 2, 1), nullptr);
        const Put p = t.put(slot(conv('B', 10), 0, 60), nullptr);
        CHECK(p.spilled == 1 && t.ram_bytes() == 60 && t.size() == 2 && t.ram_count() == 1, rc);
        t.add(fslot(conv('C', 10), -1, 2, 3, 5000));   // the folder's alone: no RAM
        CHECK(t.ram_bytes() == 60 && t.ram_count() == 1 && t.size() == 3, rc);
        const Put q = t.put(slot(conv('D', 10), 0, 30), nullptr);   // 90 <= 100
        CHECK(q.spilled == 0 && q.evicted == 0 && t.size() == 4, rc);
    }
    {   // the new one and the protected one keep their bytes over the caps
        Table t(knobs_on(1, 0));
        t.put(fslot(conv('A', 10), 10, 2, 1), nullptr);
        const auto* pa = t[0].data.get();
        const Put p = t.put(fslot(conv('B', 10), 10, 2, 2), pa);
        CHECK(p.spilled == 0 && p.evicted == 0 && t.ram_count() == 2, "put (v2): never the new one, never protect");
        CHECK(t[by_letter(t, 'A')].in_ram() && t[by_letter(t, 'B')].in_ram(), "put (v2): never the new one, never protect");
    }
    {   // Put::gone: superseded and evicted slots, with their files
        Table t(knobs_on(1, 0));
        t.put(fslot(conv('A', 10), 10, 2, 1), nullptr);
        const Put p = t.put(fslot(conv('A', 20), 10, 1, 2), nullptr);   // supersedes A(10), its file in gone
        CHECK(p.superseded == 1 && p.gone.size() == 1 && p.gone[0].file == "/folder/slot-1" && p.gone[0].id == 1, gn);
        const Put q = t.put(slot(conv('B', 10), 0, 10), nullptr);   // A(20) has a file: spilled, not gone
        CHECK(q.spilled == 1 && q.gone.empty(), gn);
        const Put r = t.put(slot(conv('C', 10), 0, 10), nullptr);   // B has none: evicted, in gone
        CHECK(r.evicted == 1 && r.gone.size() == 1 && gone_has(r, 'B'), gn);
        // a folder's slot superseded by a later state of its conversation: gone too
        const Put u = t.put(slot(conv('A', 30), 0, 10), nullptr);
        CHECK(u.superseded == 1 && gone_has(u, 'A') && u.gone.size() >= 1 && u.gone[0].file == "/folder/slot-2", gn);
    }
    {   // in_ram / bytes
        Slot a = slot(conv('A', 10), 0, 10);
        CHECK(a.in_ram() && a.bytes() == 10, ir);
        Slot b = slot(conv('B', 10), 0, -1);
        b.size = 70;
        CHECK(b.in_ram() && b.bytes() == 70, ir);   // (a test's slot of `size` alone)
        Slot c = fslot(conv('C', 10), -1, 2, 3);
        CHECK(!c.in_ram() && c.bytes() == 0, ir);
        Slot d = fslot(conv('D', 10), 10, 2, 4);
        CHECK(d.in_ram() && d.bytes() == 10, ir);
    }
    {   // the mixed case: the LRU has no file, a newer RAM slot has one - which goes? (the contract's sentence reads
        // both ways: not asserted, said)
        Table t(knobs_on(2, 0));
        t.put(slot(conv('A', 10), 0, 10), nullptr);
        t.put(fslot(conv('B', 10), 10, 2, 2), nullptr);
        const Put p = t.put(slot(conv('C', 10), 0, 10), nullptr);
        std::printf("  (note) over the cap, the LRU without a file (A) and a newer one with a file (B): evicted %zu, "
                    "spilled %zu, A %s, B %s\n", (size_t) p.evicted, (size_t) p.spilled,
                    by_letter(t, 'A') == Match::npos ? "dropped" : "kept",
                    by_letter(t, 'B') == Match::npos ? "dropped" : (t[by_letter(t, 'B')].in_ram() ? "in RAM" : "spilled"));
    }

    // best: a slot whose bytes are only in a write still running is skipped; the folder's written one is found
    {
        const char* bw = "best (v2): skips a slot whose bytes are only in a write still running (no bytes, disk 1)";
        Geometry g{};
        g.rows = 1 << 20;
        g.back = 8;
        Table t(knobs_on(1, 0));
        t.put(fslot(conv('A', 600), 10, 1, 1), nullptr);   // being written
        t.put(slot(conv('B', 600), 0, 10), nullptr);       // A spills: no bytes, disk 1
        const size_t a = by_letter(t, 'A');
        CHECK(a != Match::npos && !t[a].in_ram() && t[a].disk == 1, bw);
        const Match m = t.best(cat(conv('A', 600), Toks{-1}), {}, g);
        CHECK(m.index == Match::npos && m.len == 0, bw);
        t.slot(a).disk = 2;   // the writer reported it: the folder's now
        const Match m2 = t.best(cat(conv('A', 600), Toks{-1}), {}, g);
        CHECK(m2.index == a && m2.len == 600, "best (v2): a written slot of the folder's alone is found (taken from disk)");
        // in RAM and being written: its bytes are there - found
        Table t2(knobs_on(4, 0));
        t2.put(fslot(conv('C', 600), 10, 1, 3), nullptr);
        const Match m3 = t2.best(cat(conv('C', 600), Toks{-1}), {}, g);
        CHECK(m3.len == 600, "best (v2): a slot in RAM being written is found");
    }

    // add: the folder's slots at the start - appended, the most recently used, no caps
    {
        const char* ad = "add (v2): a folder's slot appended as the most recently used, no caps";
        Geometry g{};
        g.rows = 1 << 20;
        g.back = 8;
        Table t(knobs_on(1, 1));
        t.put(slot(conv('A', 100), 0, 10), nullptr);
        t.add(fslot(cat(conv('A', 50), Toks{-7}), -1, 2, 11));
        t.add(fslot(cat(conv('A', 50), Toks{-8}), -1, 2, 12));
        CHECK(t.size() == 3 && t.ram_count() == 1, ad);
        const size_t x = t.find_id(11), y = t.find_id(12), a = by_letter(t, 'A');
        CHECK(x != Match::npos && y != Match::npos && t[y].used > t[x].used && t[x].used > t[a].used, ad);
        // ties: the later add is the most recently used
        const Match m = t.best(cat(conv('A', 50), Toks{-9, -9}), {}, g);
        CHECK(m.len == 50 && m.index == y, ad);   // (a tie of three: the last add is the most recently used)
        CHECK(t.find_id(999) == Match::npos, "find_id: npos for an id not held");
        CHECK(t.live(conv('A', 100), {}) == a, "live: the slot of exactly this state");
        CHECK(t.live(conv('A', 99), {}) == Match::npos, "live: npos for a prefix of a slot's state");
        CHECK(t.live(cat(conv('A', 100), Toks{1}), {}) == Match::npos, "live: npos for a longer state");
        Table t2(knobs_on(4, 0));
        t2.put(slot(conv('P', 100), 0, 10, {pic(5, 1)}), nullptr);
        CHECK(t2.live(conv('P', 100), {pic(5, 1)}) == 0, "live: the same pictures");
        CHECK(t2.live(conv('P', 100), {pic(5, 2)}) == Match::npos, "live: other pictures - not this state");
        CHECK(t2.live(conv('P', 100), {}) == Match::npos, "live: other pictures - not this state");
    }

    // trim_folder(cap, keep_id, keep2)
    {
        const char* tf = "trim_folder: while the folder's slots (disk 1, 2) hold more than cap, the LRU written one "
                         "goes - never keep_id / keep2, never one being written";
        Table t(knobs_on(8, 0));
        t.add(fslot(conv('A', 10), -1, 2, 1, 100));   // the folder's alone, oldest
        t.add(fslot(conv('B', 10), -1, 2, 2, 100));
        t.put(fslot(conv('C', 10), 10, 2, 3, 100), nullptr);   // in RAM with its file
        t.put(fslot(conv('D', 10), 10, 1, 4, 100), nullptr);   // being written
        CHECK(t.disk_bytes() == 400, "disk_bytes: the folder's slots' (disk 1 and 2) file bytes");
        std::vector<std::string> r = t.trim_folder(250, 0, 0);
        std::sort(r.begin(), r.end());
        CHECK((r == std::vector<std::string>{"/folder/slot-1", "/folder/slot-2"}), tf);
        CHECK(t.size() == 2 && t.find_id(1) == Match::npos && t.find_id(2) == Match::npos, tf);   // the folder's: dropped
        CHECK(t.disk_bytes() == 200, tf);
        // past it again: C (written, in RAM) loses its file and keeps its bytes; D (being written) never goes
        r = t.trim_folder(50, 0, 0);
        CHECK((r == std::vector<std::string>{"/folder/slot-3"}), tf);
        const size_t c = t.find_id(3) != Match::npos ? t.find_id(3) : by_letter(t, 'C');
        CHECK(c != Match::npos && t[c].in_ram() && t[c].bytes() == 10 && t[c].disk == 0 && t[c].file.empty(), tf);
        CHECK(t.find_id(4) != Match::npos && t[t.find_id(4)].disk == 1, tf);
        CHECK(t.size() == 2 && t.disk_bytes() == 100, tf);
    }
    {
        const char* tk = "trim_folder: never keep_id (the one just written) or keep2 (one about to be taken)";
        Table t(knobs_on(8, 0));
        t.add(fslot(conv('A', 10), -1, 2, 1, 100));
        t.add(fslot(conv('B', 10), -1, 2, 2, 100));
        t.add(fslot(conv('C', 10), -1, 2, 3, 100));
        std::vector<std::string> r = t.trim_folder(0, 1, 3);
        CHECK((r == std::vector<std::string>{"/folder/slot-2"}), tk);
        CHECK(t.find_id(1) != Match::npos && t.find_id(3) != Match::npos && t.size() == 2, tk);
        // a single larger conversation, just written, stays
        Table t2(knobs_on(8, 0));
        t2.add(fslot(conv('A', 10), -1, 2, 9, 5000));
        r = t2.trim_folder(100, 9, 0);
        CHECK(r.empty() && t2.size() == 1, tk);
        // under the cap: nothing
        Table t3(knobs_on(8, 0));
        t3.add(fslot(conv('A', 10), -1, 2, 1, 100));
        CHECK(t3.trim_folder(100, 0, 0).empty() && t3.size() == 1, "trim_folder: at or under the cap, nothing");
        // LRU order follows use, not the adds' order: a touched old one stays
        Table t4(knobs_on(8, 0));
        t4.add(fslot(conv('A', 10), -1, 2, 1, 100));
        t4.add(fslot(conv('B', 10), -1, 2, 2, 100));
        t4.touch(t4.find_id(1));
        r = t4.trim_folder(150, 0, 0);
        CHECK((r == std::vector<std::string>{"/folder/slot-2"}), "trim_folder: the least recently used goes");
    }
}

// ---- v2: checksum, the .meta, file_kind ---------------------------------------------------------------------------

// ---- the .meta's bytes as the contract lays them out (HY4META2), written out here --------------------------------
uint32_t zcrc32(const uint8_t* p, size_t n) {   // zlib's CRC-32 (reflected 0xEDB88320), bit by bit
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}
uint64_t rd64(const std::vector<uint8_t>& b, size_t at) {
    uint64_t v = 0;
    for (int k = 7; k >= 0; --k) v = (v << 8) | b[at + (size_t) k];
    return v;
}
void wr64(std::vector<uint8_t>& b, size_t at, uint64_t v) {
    for (int k = 0; k < 8; ++k) b[at + (size_t) k] = (uint8_t) (v >> (8 * k));
}
// the crc field (the 8 bytes before the closing magic) recomputed over every byte before it
void reseal(std::vector<uint8_t>& b) { wr64(b, b.size() - 16, zcrc32(b.data(), b.size() - 16)); }
// a .meta with another identity: L and its bytes replaced, the crc recomputed (what a test of a foreign file does)
std::vector<uint8_t> with_identity(const std::vector<uint8_t>& e, const std::string& id) {
    const uint64_t L = rd64(e, 8);
    std::vector<uint8_t> b(e.begin(), e.begin() + 8);
    b.resize(16);
    wr64(b, 8, id.size());
    b.insert(b.end(), id.begin(), id.end());
    b.insert(b.end(), e.begin() + 16 + (long) L, e.end());
    reseal(b);
    return b;
}

void test_meta_layout(const Meta& m, const std::vector<uint8_t>& e, bool every_byte) {
    const char* ly = "meta: the documented HY4META2 layout (little-endian fields in order, the crc, the magic)";
    // the bytes read field by field, as documented
    size_t at = 8;
    const uint64_t L = rd64(e, at);
    at += 8;
    CHECK(L == m.identity.size() && std::string(e.begin() + 16, e.begin() + 16 + (long) L) == m.identity, ly);
    at += L;
    CHECK(rd64(e, at) == m.bin_bytes && rd64(e, at + 8) == m.checksum && rd64(e, at + 16) == (uint64_t) m.n &&
              rd64(e, at + 24) == (uint64_t) m.lo && rd64(e, at + 32) == m.tokens.size(), ly);
    at += 40;
    bool toks = true;
    for (size_t i = 0; i < m.tokens.size(); ++i) {
        const uint32_t v = (uint32_t) e[at] | (uint32_t) e[at + 1] << 8 | (uint32_t) e[at + 2] << 16 |
                           (uint32_t) e[at + 3] << 24;
        toks &= (int32_t) v == m.tokens[i];
        at += 4;
    }
    CHECK(toks, ly);
    CHECK(rd64(e, at) == m.pics.size(), ly);
    at += 8;
    bool pics = true;
    for (const Pic& p : m.pics) {
        pics &= (int64_t) rd64(e, at) == p.start && rd64(e, at + 8) == p.hash;
        at += 16;
    }
    CHECK(pics, ly);
    CHECK(at + 16 == e.size(), ly);   // then the crc, then the magic, nothing more
    const uint64_t crc = rd64(e, at);
    CHECK(crc >> 32 == 0 && crc == zcrc32(e.data(), at),
          "meta: crc = zlib CRC-32 of every byte before it, low 32 bits, high 32 zero");
    // the library's crc32 is zlib's (the check value of "123456789")
    const char* nine = "123456789";
    CHECK(crc32((const uint8_t*) nine, 9, 0) == 0xCBF43926u && zcrc32((const uint8_t*) nine, 9) == 0xCBF43926u,
          "crc32(p, n, crc): zlib's CRC-32 (Python's zlib.crc32)");
    CHECK(crc32(e.data(), at, 0) == crc, "crc32(p, n, crc): zlib's CRC-32 (Python's zlib.crc32)");

    const char* v1 = "meta_decode: \"HY4META1\" (v2's first format) is refused";
    {   // HY4META1 at both ends, the crc recomputed: still refused
        std::vector<uint8_t> b = e;
        std::copy_n("HY4META1", 8, b.begin());
        std::copy_n("HY4META1", 8, b.end() - 8);
        reseal(b);
        Meta x{};
        std::string w;
        CHECK(!meta_decode(b.data(), b.size(), x, w) && !w.empty(), v1);
        std::vector<uint8_t> c = e;   // the old magic first alone
        std::copy_n("HY4META1", 8, c.begin());
        reseal(c);
        w.clear();
        CHECK(!meta_decode(c.data(), c.size(), x, w) && !w.empty(), v1);
    }
    {   // a flipped byte anywhere - every byte (a small .meta; a big one: its head, its tail and every 997th), two
        // kinds of flip
        long long took = 0;
        for (size_t i = 0; i < e.size(); i += every_byte || i < 512 || i + 512 > e.size() ? 1 : 997)
            for (uint8_t f : {(uint8_t) 0x01, (uint8_t) 0x80}) {
                std::vector<uint8_t> b = e;
                b[i] ^= f;
                Meta x{};
                std::string w;
                if (meta_decode(b.data(), b.size(), x, w) || w.empty()) ++took;
            }
        CHECK(took == 0, "meta_decode: a flipped byte anywhere is refused (bounds, crc or magic)");
    }
    {   // a flipped byte with the crc recomputed (a well-formed .meta that says something else): the crc alone is not
        // what refuses it - here a token changed: decodes, with the changed token (the .bin's own checks are the take's)
        std::vector<uint8_t> b = e;
        const size_t tok0 = 16 + L + 40;
        b[tok0] ^= 0x01;
        reseal(b);
        Meta x{};
        std::string w;
        CHECK(meta_decode(b.data(), b.size(), x, w) && !x.tokens.empty() && x.tokens[0] == (m.tokens[0] ^ 1),
              "meta: a rewritten field with its crc recomputed decodes as written (the layout is the contract's)");
    }
    {   // a rewritten identity (one character, and another length) with the crc recomputed: well formed, the new one
        const char* ri = "meta: a rewritten identity with its crc recomputed decodes as well formed, the new identity";
        std::string id1 = m.identity;
        id1[id1.size() / 2] = id1[id1.size() / 2] == 'x' ? 'y' : 'x';
        for (const std::string& id : {id1, m.identity + " | foreign", std::string("o")}) {
            const std::vector<uint8_t> b = with_identity(e, id);
            Meta x{};
            std::string w;
            CHECK(meta_decode(b.data(), b.size(), x, w), ri);
            CHECK(x.identity == id && x.identity != m.identity && x.tokens == m.tokens && x.n == m.n &&
                      x.lo == m.lo && x.bin_bytes == m.bin_bytes && x.checksum == m.checksum, ri);
            // and without the crc recomputed: refused
            std::vector<uint8_t> c = b;
            wr64(c, c.size() - 16, rd64(e, e.size() - 16));
            w.clear();
            if (zcrc32(c.data(), c.size() - 16) != (uint32_t) rd64(c, c.size() - 16))
                CHECK(!meta_decode(c.data(), c.size(), x, w) && !w.empty(), "meta_decode: a stale crc is refused");
        }
    }
    {   // the crc field's high 32 bits not zero: refused (documented zero)
        std::vector<uint8_t> b = e;
        b[b.size() - 16 + 4] = 1;
        Meta x{};
        std::string w;
        CHECK(!meta_decode(b.data(), b.size(), x, w), "meta_decode: the crc's high 32 bits must be zero");
    }
}

void test_folder_bytes() {
    const char* ck = "checksum: a flipped byte or bit changes it; the same bytes anywhere give the same sum";
    std::vector<uint8_t> buf(1031);
    std::mt19937 rng(99);
    for (auto& b : buf) b = (uint8_t) rng();
    const uint64_t base = checksum(buf.data(), buf.size());
    CHECK(base == checksum(buf.data(), buf.size()), ck);
    long long same = 0;
    for (size_t i = 0; i < buf.size(); ++i)
        for (int bit = 0; bit < 8; ++bit) {
            buf[i] ^= (uint8_t) (1u << bit);
            same += checksum(buf.data(), buf.size()) == base;
            buf[i] ^= (uint8_t) (1u << bit);
        }
    CHECK(same == 0, ck);
    std::vector<uint8_t> shifted(buf.size() + 3);
    std::copy(buf.begin(), buf.end(), shifted.begin() + 3);
    CHECK(checksum(shifted.data() + 3, buf.size()) == base, ck);   // (an unaligned copy)
    CHECK(checksum(buf.data(), buf.size() - 1) != base, "checksum: a shorter .bin sums otherwise");
    long long short_same = 0;   // every length 0..64 of zeros: distinct (the tail and the lanes)
    std::vector<uint8_t> z(65, 0);
    std::vector<uint64_t> seen;
    for (size_t n = 0; n <= 64; ++n) seen.push_back(checksum(z.data(), n));
    std::sort(seen.begin(), seen.end());
    short_same = std::unique(seen.begin(), seen.end()) - seen.begin();
    CHECK(short_same == 65, "checksum: lengths 0..64 of zeros all differ");

    const char* mr = "meta_encode / meta_decode: a round trip keeps every field";
    const char* mt = "meta_decode: refuses a truncated or damaged .meta (why says what)";
    Meta m{};
    m.identity = "hy4-slot v1 | engine 123 | pack /p | q8 kv_rb=672 | ring 16896";
    m.bin_bytes = 1234567890123ull;
    m.checksum = 0xfedcba9876543210ull;
    m.n = 70001;
    m.lo = 53105;
    for (int i = 0; i < 70001; ++i) m.tokens.push_back(i * 7 - 3);
    m.pics = {pic(10, 0xabcdefull), pic(5000, ~0ull)};
    const std::vector<uint8_t> e = meta_encode(m);
    Meta d{};
    std::string why;
    CHECK(meta_decode(e.data(), e.size(), d, why), mr);
    CHECK(d.identity == m.identity && d.bin_bytes == m.bin_bytes && d.checksum == m.checksum && d.n == m.n &&
              d.lo == m.lo && d.tokens == m.tokens && d.pics.size() == 2 && pic_eq(d.pics[0], m.pics[0]) &&
              pic_eq(d.pics[1], m.pics[1]), mr);
    CHECK(e.size() >= 16 && std::string(e.begin(), e.begin() + 8) == "HY4META2" &&
              std::string(e.end() - 8, e.end()) == "HY4META2", "meta: \"HY4META2\" first and last");
    test_meta_layout(m, e, false);
    {   // a small one: every byte flipped
        Meta sm{};
        sm.identity = "hy4-slot layout 3 | engine 9 | pack sample 0123abcd | q8 | ring 16896";
        sm.bin_bytes = 777;
        sm.checksum = 0x0102030405060708ull;
        sm.n = 40;
        sm.lo = 7;
        for (int i = 0; i < 40; ++i) sm.tokens.push_back(1000 - 37 * i);
        sm.pics = {pic(3, 0x55aa55aa55aa55aaull), pic(20, 12345)};
        test_meta_layout(sm, meta_encode(sm), true);
    }
    Meta small{};   // one token, no pictures, an empty identity
    small.n = 1;
    small.tokens = {42};
    small.bin_bytes = 1;
    const std::vector<uint8_t> e0 = meta_encode(small);
    Meta d0{};
    CHECK(meta_decode(e0.data(), e0.size(), d0, why) && d0.tokens == small.tokens && d0.pics.empty() &&
              d0.identity.empty() && d0.n == 1, mr);
    {   // bounds: n not the tokens' count, lo past n - refused
        const char* mb = "meta_decode: checks every length and bound (n is the tokens' count, lo <= n)";
        Meta x = small;
        x.n = 2;
        std::vector<uint8_t> b = meta_encode(x);
        Meta y{};
        std::string w;
        CHECK(!meta_decode(b.data(), b.size(), y, w) && !w.empty(), mb);
        x = small;
        x.lo = 5;
        b = meta_encode(x);
        w.clear();
        CHECK(!meta_decode(b.data(), b.size(), y, w) && !w.empty(), mb);
        Meta z{};   // n 0 (no conversation): said, not asserted
        b = meta_encode(z);
        w.clear();
        std::printf("  (note) a .meta of 0 tokens: %s\n", meta_decode(b.data(), b.size(), y, w) ? "decoded" :
                    ("refused: " + w).c_str());
    }
    long long took = 0;
    for (size_t n = 0; n < e.size(); n += (n < 64 || n + 64 > e.size()) ? 1 : 997) {   // every head and tail length
        Meta x{};
        std::string w;
        if (meta_decode(e.data(), n, x, w) || w.empty()) ++took;
    }
    CHECK(took == 0, mt);
    for (size_t i : {(size_t) 0, (size_t) 7, e.size() - 1, e.size() - 8}) {   // the magic, either end
        std::vector<uint8_t> b = e;
        b[i] ^= 0x20;
        Meta x{};
        std::string w;
        CHECK(!meta_decode(b.data(), b.size(), x, w) && !w.empty(), mt);
    }
    {   // a length field past the bytes: every 8-byte word after the magic set huge - refused or not, never a crash
        //  and never a decode whose fields run past the bytes (the decoded tokens fit the buffer)
        long long bad = 0;
        for (size_t off = 8; off + 8 <= std::min<size_t>(e.size(), 8 + 8 * 40); off += 4) {
            std::vector<uint8_t> b = e;
            for (int k = 0; k < 8; ++k) b[off + (size_t) k] = 0xff;
            Meta x{};
            std::string w;
            if (meta_decode(b.data(), b.size(), x, w) && x.tokens.size() * 4 > b.size()) ++bad;
        }
        CHECK(bad == 0, mt);
    }
    {   // random damage, every buffer CHANGED (a byte xor-ed with 1..255, so never its own value): none decodes -
        // the CRC-32 over every byte before it (or the bounds / the magics) refuses it (run under
        // -fsanitize=address,undefined for the no-crash proof)
        std::mt19937 r2(5);
        int ok_after = 0;
        for (int i = 0; i < 3000; ++i) {
            std::vector<uint8_t> b = i % 2 ? e0 : e;
            const std::vector<uint8_t> orig = b;
            for (int k = 0; k < 1 + (int) (r2() % 4); ++k) b[r2() % b.size()] ^= (uint8_t) (1 + r2() % 255);
            if (b == orig) b[0] ^= 0x40;   // (two hits on one byte that cancelled: still a change)
            Meta x{};
            std::string w;
            ok_after += meta_decode(b.data(), b.size(), x, w);
        }
        CHECK(ok_after == 0, "meta_decode: a damaged .meta (any byte changed) is refused - HY4META2's CRC-32");
        if (ok_after) std::fprintf(stderr, "      (%d of 3000 damaged buffers decoded)\n", ok_after);
    }
    {   // nothing after the closing magic
        std::vector<uint8_t> b = e;
        b.push_back(0);
        Meta x{};
        std::string w;
        CHECK(!meta_decode(b.data(), b.size(), x, w) && !w.empty(),
              "meta_decode: \"HY4META2\" last, nothing after it - a trailing byte refused");
    }

    const char* fk = "file_kind: slot-*.bin / .meta, slot-*.tmp an interrupted write, anything else not ours";
    CHECK(file_kind("slot-20261125-1.bin") == kBin, fk);
    CHECK(file_kind("slot-20261125-1.meta") == kMeta, fk);
    CHECK(file_kind("slot-20261125-1.bin.tmp") == kTmp, fk);
    CHECK(file_kind("slot-20261125-1.meta.tmp") == kTmp, fk);
    CHECK(file_kind("slot-x.tmp") == kTmp, fk);
    CHECK(file_kind(".slots.lock") == kOther, fk);
    CHECK(file_kind("notes.txt") == kOther, fk);
    CHECK(file_kind("slot.bin") == kOther, fk);
    CHECK(file_kind("myslot-1.bin") == kOther, fk);
    CHECK(file_kind("slot-1.binx") == kOther, fk);
    CHECK(file_kind("slot-1.bin.bak") == kOther, fk);
    CHECK(file_kind("") == kOther, fk);
}

void test_folder_knobs() {
    const char* df = "knobs (v2): DIR unset (dir_set false), GB 16, DAYS 7, IDLE_S 5, QUIT_S 10";
    const char* rd = "knobs (v2): MAYA_HY4_SLOT_DIR / _GB / _DAYS / _IDLE_S / _QUIT_S read";
    const char* iv = "knobs (v2): invalid numbers keep the default (warn gets a line)";
    std::string warn;
    g_env.clear();
    Knobs d = knobs_from(fake_env, &warn);
    CHECK(!d.dir_set, df);
    CHECK(d.folder_bytes == 16ull << 30 || d.folder_bytes == 16000000000ull, df);
    CHECK(d.days == 7 && d.idle_s == 5 && d.quit_s == 10, df);
    g_env = {{"MAYA_HY4_SLOT_KEEP", "1"}, {"MAYA_HY4_SLOT_DIR", ""}};
    Knobs k = knobs_from(fake_env, &warn);
    CHECK(k.dir_set && k.dir.empty(), "knobs (v2): DIR set and empty - dir_set, no folder");
    g_env = {{"MAYA_HY4_SLOT_KEEP", "1"}, {"MAYA_HY4_SLOT_DIR", "/mnt/ssd/slots"}, {"MAYA_HY4_SLOT_GB", "2"},
             {"MAYA_HY4_SLOT_DAYS", "1.5"}, {"MAYA_HY4_SLOT_IDLE_S", "0"}, {"MAYA_HY4_SLOT_QUIT_S", "3"}};
    warn.clear();
    k = knobs_from(fake_env, &warn);
    CHECK(k.dir_set && k.dir == "/mnt/ssd/slots", rd);
    CHECK(k.folder_bytes == 2ull << 30 || k.folder_bytes == 2000000000ull, rd);
    CHECK(k.days == 1.5 && k.idle_s == 0 && k.quit_s == 3, rd);
    CHECK(warn.empty(), rd);
    g_env = {{"MAYA_HY4_SLOT_KEEP", "1"}, {"MAYA_HY4_SLOT_GB", "0"}};
    k = knobs_from(fake_env, &warn);
    CHECK(k.folder_bytes == 0, "knobs (v2): MAYA_HY4_SLOT_GB=0 (no folder) read as 0");
    const char* bad[][2] = {{"MAYA_HY4_SLOT_GB", "lots"}, {"MAYA_HY4_SLOT_GB", "-1"}, {"MAYA_HY4_SLOT_DAYS", "week"},
                            {"MAYA_HY4_SLOT_DAYS", "-2"}, {"MAYA_HY4_SLOT_IDLE_S", "soon"},
                            {"MAYA_HY4_SLOT_IDLE_S", "-5"}, {"MAYA_HY4_SLOT_QUIT_S", "x"}, {"MAYA_HY4_SLOT_QUIT_S", "-1"}};
    for (const auto& b : bad) {
        g_env = {{"MAYA_HY4_SLOT_KEEP", "1"}, {b[0], b[1]}};
        warn.clear();
        k = knobs_from(fake_env, &warn);
        const bool kept = k.folder_bytes == d.folder_bytes && k.days == d.days && k.idle_s == d.idle_s &&
                          k.quit_s == d.quit_s;
        if (!kept || warn.empty()) std::fprintf(stderr, "      (%s=%s)\n", b[0], b[1]);
        CHECK(kept, iv);
        CHECK(!warn.empty(), iv);
    }
}

}  // namespace

int main() {
    test_shared_start();
    test_usable();
    test_best();
    test_put();
    test_random();
    test_decide();
    test_knobs();
    test_folder_table();
    test_folder_bytes();
    test_folder_knobs();
    if (g_fail) {
        std::fprintf(stderr, "hy4_slot_test: %d of %d checks FAILED\n", g_fail, g_checks);
        return 1;
    }
    std::printf("hy4_slot_test: ok (%d checks: shared_start, usable, Table best / put / touch / drop / find, decide, "
                "knobs_from; random tables: %lld requests with a usable slot, %lld superseded, %lld evicted, %lld parks "
                "over the cap with a protected slot)\n", g_checks, g_stats[0], g_stats[1], g_stats[2], g_stats[3]);
    return 0;
}
