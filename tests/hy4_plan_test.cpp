// maya-hy4's --kv auto and its pictures' room where the layers are split over the GPUs (one GPU too) -
// src/hy4/hy4_plan.hpp's kv_pick: a start that fits with F16 rows (and the largest picture's room where pictures are
// on) stays as it was; then 8-bit rows with that room, F16 and 8-bit rows with a smaller picture's room, F16 rows
// text only, 8-bit rows text only; nothing that fits no way.
#include "../src/hy4/hy4_plan.hpp"

#include <cstdio>
#include <string>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

// a card's free VRAM against the model's own parts: `dense` + the cache (F16: `kv`, 8-bit rows: 58% of it) + a
// picture's room; `tried`: what kv_pick asked, in order ("f" / "q" the rows; "V" the largest picture's room, "v" a
// smaller one's, "-" none)
const long long kMax = 16384;   // (the tower's largest picture, in patches)
struct Card {
    double vram, dense, kv, vis;   // (vis: the largest picture's room; a smaller one's: 0.25 * vis + 0.75 * vis * n / kMax)
    std::string tried;
    double room(long long n) const { return n >= kMax ? vis : 0.25 * vis + 0.75 * vis * (double) n / (double) kMax; }
    bool operator()(bool q8, double r) {
        tried += q8 ? 'q' : 'f';
        tried += r == 0 ? '-' : r >= vis ? 'V' : 'v';
        return dense + (q8 ? 0.58 * kv : kv) + r <= vram;
    }
};

// (n_min == kMax: no smaller picture - the order of the six steps alone)
hy4::KvPick pick(Card& c, int kv, bool vision, bool vision_auto, long long n_min = kMax) {
    return hy4::kv_pick(c, kv == 1, kv < 0, vision, vision_auto, kMax, n_min, [&](long long n) { return c.room(n); });
}

}  // namespace

int main() {
    // fits with F16 rows: F16, nothing else asked (the start is the one before --kv auto reached this layout)
    for (int kv : {-1, 0}) {
        Card c{8.0, 6.0, 1.0, 0.8, ""};
        const hy4::KvPick p = pick(c, kv, true, true);
        CHECK(p.fits && !p.q8 && p.vision && p.patches == kMax);
        CHECK(c.tried == "fV");
    }
    // pictures go before the rows' precision: where F16 rows + a picture's room do not fit but 8-bit rows + the room
    // do, 8-bit rows with pictures - even where F16 rows alone would fit
    {
        Card c{7.5, 6.0, 1.0, 0.8, ""};
        const hy4::KvPick p = pick(c, -1, true, true);
        CHECK(p.fits && p.q8 && p.vision && p.patches == kMax);
        CHECK(c.tried == "fVqV");
    }
    // ... then F16 rows text only, then 8-bit rows text only (pictures not asked for by name)
    {
        Card c{7.3, 6.0, 1.0, 0.8, ""};
        const hy4::KvPick p = pick(c, -1, true, true);
        CHECK(p.fits && !p.q8 && !p.vision);
        CHECK(c.tried == "fVqVf-");
        Card d{7.3, 6.0, 2.0, 0.8, ""};
        const hy4::KvPick q = pick(d, -1, true, true);
        CHECK(q.fits && q.q8 && !q.vision);
        CHECK(d.tried == "fVqVf-q-");
        Card e{7.3, 6.0, 2.0, 0.8, ""};   // (--vision on: the room is never given up)
        const hy4::KvPick r = pick(e, -1, true, false);
        CHECK(!r.fits && !r.q8 && r.vision);
        CHECK(e.tried == "fVqV");
        Card f{7.3, 6.0, 1.0, 0.8, ""};   // (--kv f16: pictures given up, never the rows)
        const hy4::KvPick t = pick(f, 0, true, true);
        CHECK(t.fits && !t.q8 && !t.vision);
        CHECK(f.tried == "fVf-");
    }
    // a smaller picture before no picture: the most patches that fit, F16 rows before 8-bit rows
    {
        Card c{7.3, 6.0, 1.0, 0.8, ""};   // (0.3 left with F16 rows: a room of 0.2 + 0.6 * n / kMax)
        const hy4::KvPick p = pick(c, -1, true, true, 1024);
        CHECK(p.fits && !p.q8 && p.vision);
        CHECK(p.patches >= 1024 && p.patches < kMax && c.room(p.patches) <= 0.3 + 1e-9 && c.room(p.patches + 64) > 0.3);
        CHECK(c.tried.rfind("fVqVfv", 0) == 0 && c.tried.find('-') == std::string::npos);
        Card d{7.4, 6.0, 2.0, 0.8, ""};   // (F16 rows: none; 8-bit rows: 0.24 left)
        const hy4::KvPick q = pick(d, -1, true, true, 1024);
        CHECK(q.fits && q.q8 && q.vision && q.patches >= 1024 && q.patches < kMax);
        CHECK(d.room(q.patches) <= 0.24 + 1e-9);
        Card e{7.3, 6.0, 2.0, 0.8, ""};   // (not the smallest picture's room either: text only, 8-bit rows)
        const hy4::KvPick r = pick(e, -1, true, true, 1024);
        CHECK(r.fits && r.q8 && !r.vision && r.patches == 0);
        CHECK(e.tried == "fVqVfvqvf-q-");
        Card f{7.3, 6.0, 1.0, 0.8, ""};   // (--vision on takes the smaller picture too)
        const hy4::KvPick t = pick(f, -1, true, false, 1024);
        CHECK(t.fits && !t.q8 && t.vision && t.patches < kMax);
        Card h{7.4, 6.0, 2.0, 0.8, ""};   // (--kv f16: no 8-bit rows for a picture - text only does not fit either)
        const hy4::KvPick u = pick(h, 0, true, true, 1024);
        CHECK(!u.fits && !u.q8 && !u.vision);
        CHECK(h.tried == "fVfvf-");
    }
    // pictures off: 8-bit rows only where F16 rows do not fit (--kv auto; the plan then says so in an INFO line)
    {
        Card c{7.3, 6.0, 2.0, 0.0, ""};
        const hy4::KvPick p = pick(c, -1, false, true);
        CHECK(p.fits && p.q8 && !p.vision);
        CHECK(c.tried == "f-q-");
        Card d{7.3, 6.0, 1.0, 0.0, ""};
        const hy4::KvPick q = pick(d, -1, false, true);
        CHECK(q.fits && !q.q8 && !q.vision);
        CHECK(d.tried == "f-");
    }
    // --kv f16 never takes 8-bit rows; --kv q8 never F16
    {
        Card c{7.3, 6.0, 2.0, 0.0, ""};
        const hy4::KvPick p = pick(c, 0, false, true);
        CHECK(!p.fits && !p.q8);
        CHECK(c.tried == "f-");
        Card d{9.0, 6.0, 2.0, 0.0, ""};
        const hy4::KvPick q = pick(d, 1, false, true);
        CHECK(q.fits && q.q8);
        CHECK(d.tried == "q-");
        Card e{7.9, 6.0, 2.0, 0.8, ""};
        const hy4::KvPick r = pick(e, 1, true, true);
        CHECK(r.fits && r.q8 && !r.vision);
        CHECK(e.tried == "qVq-");
    }
    // fits no way: no start (the plan's error), with pictures not asked for by name their room given up first
    {
        Card c{7.0, 6.0, 2.0, 0.8, ""};
        const hy4::KvPick p = pick(c, -1, true, true);
        CHECK(!p.fits && !p.q8 && !p.vision);
        CHECK(c.tried == "fVqVf-q-");
    }
    std::puts("hy4_plan_test: ok");
    return 0;
}
