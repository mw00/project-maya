// maya-hy4's --kv auto where the layers are split over the GPUs (one GPU too) - src/hy4/hy4_plan.hpp's kv_pick: a
// start that fits with F16 rows stays as it was, 8-bit rows only where F16 does not fit, and nothing that fits neither.
#include "../src/hy4/hy4_plan.hpp"

#include <cstdio>
#include <string>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

// a card's free VRAM against the model's own parts: `dense` + the cache (F16: `kv`, 8-bit rows: 58% of it) + a
// picture's room `vis`; `tried`: what kv_pick asked, in order ("f" / "q" the rows, "V" / "-" the picture's room)
struct Card {
    double vram, dense, kv, vis;
    std::string tried;
    bool operator()(bool q8, bool vision) {
        tried += q8 ? 'q' : 'f';
        tried += vision ? 'V' : '-';
        return dense + (q8 ? 0.58 * kv : kv) + (vision ? vis : 0.0) <= vram;
    }
};

hy4::KvPick pick(Card& c, int kv, bool vision, bool vision_auto) {
    return hy4::kv_pick(c, kv == 1, kv < 0, vision, vision_auto);
}

}  // namespace

int main() {
    // fits with F16 rows: F16, nothing else asked (the start is the one before --kv auto reached this layout)
    for (int kv : {-1, 0}) {
        Card c{8.0, 6.0, 1.0, 0.8, ""};
        const hy4::KvPick p = pick(c, kv, true, true);
        CHECK(p.fits && !p.q8 && p.vision);
        CHECK(c.tried == "fV");
    }
    // ... and F16 without a picture's room goes before 8-bit rows with it: the text-only start is kept as it was
    {
        Card c{7.3, 6.0, 1.0, 0.8, ""};
        const hy4::KvPick p = pick(c, -1, true, true);
        CHECK(p.fits && !p.q8 && !p.vision);
        CHECK(c.tried == "fVf-");
    }
    // fits only with 8-bit rows: --kv auto takes them (the plan then says so in an INFO line)
    {
        Card c{7.3, 6.0, 2.0, 0.0, ""};
        const hy4::KvPick p = pick(c, -1, false, true);
        CHECK(p.fits && p.q8 && !p.vision);
        CHECK(c.tried == "f-q-");
    }
    // ... with a picture's room where that fits too, without it where pictures were not asked for by name
    {
        Card c{7.99, 6.0, 2.0, 0.8, ""};
        const hy4::KvPick p = pick(c, -1, true, true);
        CHECK(p.fits && p.q8 && p.vision);
        CHECK(c.tried == "fVf-qV");
        Card d{7.3, 6.0, 2.0, 0.8, ""};
        const hy4::KvPick q = pick(d, -1, true, true);
        CHECK(q.fits && q.q8 && !q.vision);
        CHECK(d.tried == "fVf-qVq-");
        Card e{7.3, 6.0, 2.0, 0.8, ""};   // (--vision on: the room is never given up)
        const hy4::KvPick r = pick(e, -1, true, false);
        CHECK(!r.fits && !r.q8 && r.vision);
        CHECK(e.tried == "fVqV");
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
    // fits neither way: no start (the plan's error), with pictures not asked for by name their room given up first
    {
        Card c{7.0, 6.0, 2.0, 0.8, ""};
        const hy4::KvPick p = pick(c, -1, true, true);
        CHECK(!p.fits && !p.q8 && !p.vision);
        CHECK(c.tried == "fVf-qVq-");
    }
    std::puts("hy4_plan_test: ok");
    return 0;
}
