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
    // --chunk auto (chunk_pick): the largest chunk whose plan keeps the layout, the cache's rows and the pictures of
    // the plan at the base chunk.  A rig in GB: every card `vram` free beyond the reserve; a tensor-parallel member
    // needs `member`, the head's GPU the head and the MTP block (`head`), a picture's room and a chunk's scratch
    // (`per_k` GB a 1024 tokens) - twice where the block's pass has a scratch of its own, once where it runs in line
    // (the plan's two tries); where a member does not fit: the layer split (it holds everything here), with 8-bit
    // rows above `q8_above` tokens a chunk and a smaller picture above `small_above`
    struct Rig {
        double vram, member, head, vis, per_k;
        int cards, q8_above, small_above;
        std::string tried;
        bool own = false;
        hy4::ChunkTry operator()(int chunk) {
            tried += std::to_string(chunk) + " ";
            const double s = per_k * chunk / 1024.0, rows = hy4::mtp_rows_bytes(chunk, 3072) / 1e9;
            hy4::ChunkTry t;
            t.ok = vram >= head + rows + vis + s;                 // (the layer split's main GPU)
            t.q8 = chunk > q8_above;
            t.vision = vis > 0;
            t.pixels = !t.vision ? 0 : chunk > small_above ? 1 << 20 : 4 << 20;
            own = false;
            if (cards >= 2) {
                own = member + head + rows + vis + 2 * s <= vram;
                if (own || member + head + rows + vis + s <= vram) t.tp = cards, t.ok = true;
            }
            return t;
        }
    };
    const int kCands[2] = {4096, 2048};
    CHECK(hy4::mtp_rows_bytes(1024, 3072) == 1024.0 * 3072 * 20);
    {   // three roomy cards: 4096, tensor parallel, the pass on its own scratch - asked for the base, then the largest
        Rig r{32.0, 17.0, 4.6, 0.83, 1.15, 3, 1 << 30, 1 << 30, ""};
        CHECK(hy4::chunk_pick(r, kCands, 1024) == 4096);
        CHECK(r.tried == "1024 4096 " && r.own);
    }
    {   // two cards with the block (Jupiter's failing start): 4096 would cost tensor parallel - 2048, the pass in line
        Rig r{31.0, 22.5, 4.6, 0.83, 1.15, 2, 1 << 30, 1 << 30, ""};
        CHECK(r(1024).tp == 2 && r.own);                          // (the own scratch is in the plan at 1024 alone)
        CHECK(r(2048).tp == 2 && !r.own);
        CHECK(r(4096).tp == 0 && r(4096).ok);                     // (the layer split holds it: not taken for a chunk)
        r.tried.clear();
        CHECK(hy4::chunk_pick(r, kCands, 1024) == 2048);
        CHECK(r.tried == "1024 4096 2048 ");
    }
    {   // one card: the largest chunk the model's own parts leave room for
        Rig r{9.0, 0, 4.6, 0.83, 1.15, 1, 1 << 30, 1 << 30, ""};
        CHECK(hy4::chunk_pick(r, kCands, 1024) == 2048);
        Rig big{31.0, 0, 4.6, 0.83, 1.15, 1, 1 << 30, 1 << 30, ""};
        CHECK(hy4::chunk_pick(big, kCands, 1024) == 4096);
    }
    {   // pictures and the cache's rows go first: a chunk that would make the pictures smaller, or the rows 8-bit, is
        //  not taken; one that keeps them is
        Rig r{31.0, 0, 4.6, 0.83, 1.15, 1, 1 << 30, 2048, ""};
        CHECK(hy4::chunk_pick(r, kCands, 1024) == 2048);
        Rig q{31.0, 0, 4.6, 0.83, 1.15, 1, 1024, 1 << 30, ""};
        CHECK(hy4::chunk_pick(q, kCands, 1024) == 1024);
        Rig both{31.0, 0, 4.6, 0.83, 1.15, 1, 512, 512, ""};     // (already so at the base: nothing lost by 4096)
        CHECK(hy4::chunk_pick(both, kCands, 1024) == 4096);
        Rig off{31.0, 0, 4.6, 0.0, 1.15, 1, 1 << 30, 1 << 30, ""};   // (pictures off: the chunk alone)
        CHECK(hy4::chunk_pick(off, kCands, 1024) == 4096);
    }
    {   // nothing above the base fits: the base; the base does not fit either: the base, asked once (its own error)
        Rig r{7.0, 0, 4.6, 0.83, 1.15, 1, 1 << 30, 1 << 30, ""};
        CHECK(hy4::chunk_pick(r, kCands, 1024) == 1024);
        CHECK(r.tried == "1024 4096 2048 ");
        Rig none{5.0, 0, 4.6, 0.83, 1.15, 1, 1 << 30, 1 << 30, ""};
        CHECK(hy4::chunk_pick(none, kCands, 1024) == 1024);
        CHECK(none.tried == "1024 ");
        Rig low{31.0, 0, 4.6, 0.83, 1.15, 1, 1 << 30, 1 << 30, ""};   // (a base above a candidate: never below it)
        CHECK(hy4::chunk_pick(low, kCands, 2048) == 4096);
        CHECK(low.tried == "2048 4096 ");
    }
    std::puts("hy4_plan_test: ok");
    return 0;
}
