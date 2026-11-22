// #1026: the engines' shared "reserve held for the whole run" logic (src/core/vram_late.hpp) - no GPU.
#include "../src/core/vram_late.hpp"

#include <cstdio>
#include <string>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace vl = maya::vramlate;
static constexpr size_t MiB = (size_t) 1 << 20;

int main() {
    const size_t tot = (size_t) 16384 * MiB;
    // the default reserve keeps the last release's plan while it holds the figure with 128 MiB to spare
    CHECK(vl::kept_for(320 * MiB, 700 * MiB, false) == 0);
    CHECK(vl::kept_for(572 * MiB, 700 * MiB, false) == 0);
    CHECK(vl::kept_for(576 * MiB, 700 * MiB, false) == 576 * MiB);
    CHECK(vl::kept_for(320 * MiB, 700 * MiB, true) == 320 * MiB);   // asked for: all of it

    // a 4096 MiB reserve asked for, 3587 MiB free at a request: 509 short -> 512 kept, the start said short
    {
        vl::Gpu g;
        vl::Step s = vl::check(g, 3744 * MiB, tot, 4096 * MiB, true);
        CHECK(s.moved && s.short_ && s.want == (int64_t) 352 * MiB);
        s = vl::check(g, 3800 * MiB, tot, 4096 * MiB, true);   // more free: not a new low
        CHECK(!s.moved);
        s = vl::check(g, 3587 * MiB, tot, 4096 * MiB, true);
        CHECK(s.moved && s.want == (int64_t) 512 * MiB && g.kept == 512 * MiB);
        s = vl::check(g, 3586 * MiB, tot, 4096 * MiB, true);   // within a step of the kept figure: not written
        CHECK(!s.moved);
    }
    // the next start keeps 512 beside it and comes out at the reserve: no warning, the figure stays
    {
        vl::Gpu g{512 * MiB, 512 * MiB};
        const vl::Step s = vl::check(g, 4100 * MiB, tot, 4096 * MiB, true);
        CHECK(!s.moved);
    }
    // a plan at its floor (no room): nothing kept, not said short - a restart would not help; a little room: capped
    {
        vl::Gpu g;
        CHECK(!vl::check(g, 3744 * MiB, tot, 4096 * MiB, true, 0).moved);
        vl::Gpu h;
        const vl::Step s = vl::check(h, 3744 * MiB, tot, 4096 * MiB, true, 100 * MiB);
        CHECK(s.moved && s.short_ && s.want == (int64_t) 96 * MiB);
    }
    // the default reserve: 379 of 700 free (320 short) is written but not said short (it holds the figure)
    {
        vl::Gpu g;
        const vl::Step s = vl::check(g, 379 * MiB, tot, 700 * MiB, false);
        CHECK(s.moved && !s.short_ && s.want == (int64_t) 352 * MiB);
    }
    // ... but a default reserve used up is (and the figure is capped at a quarter of the card)
    {
        vl::Gpu g;
        const vl::Step s = vl::check(g, 40 * MiB, tot, 700 * MiB, false);
        CHECK(s.moved && s.short_ && s.want == (int64_t) 672 * MiB);
        vl::Gpu h;
        CHECK(vl::check(h, 0, 1024 * MiB, 4096 * MiB, true).want == (int64_t) 256 * MiB);
    }
    // the file: a line a key, replaced in place; the tracker reads it back (default reserve: none kept)
    {
        const std::string f = "vram_late_test.txt";
        std::remove(f.c_str());
        CHECK(vl::put(f, "a|gpu 0", "352"));
        CHECK(vl::put(f, "a|gpu 1", "64"));
        CHECK(vl::put(f, "a|gpu 0", "512"));
        CHECK(vl::get(f, "a|gpu 0") == "512" && vl::get(f, "a|gpu 1") == "64" && vl::get(f, "a|gpu").empty());
        vl::Tracker t;
        t.init("test", "", nullptr, nullptr, 2);
        t.file = f;
        t.fixed = -1;
        CHECK(t.read(0, "a|gpu 0", 4096 * MiB, true, 0) == 512 * MiB);
        CHECK(t.read(1, "a|gpu 1", 700 * MiB, false, 1) == 0);
        t.check(0, 3000 * MiB, tot, true, 0);   // 1096 short beside the 512 kept -> 1632 written, the start short
        CHECK(t.short_mib == 1632 && vl::get(f, "a|gpu 0") == "1632");
        std::remove(f.c_str());
    }
    std::printf("vram_late_test: ok\n");
    return 0;
}
