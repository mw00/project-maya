// maya-ds4's stopped prompt read (src/ds4/ds4_keep_cancelled.hpp, MAYA_DS4_KEEP_CANCELLED): which prompt reads on from
// the part-way state a STOP left, the restore points such a read leaves out, how long a STOP waits for a run to end at a
// chunk, and the pictures the part-way state holds.  No GPU, no model.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ds4/ds4_keep_cancelled.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

using maya::ds4::part_way_cuts;
using maya::ds4::part_way_resumes;
using maya::ds4::pics_at;
using maya::ds4::PartWay;
using maya::ds4::RunPace;
using maya::ds4::stop_cuts_now;

namespace {

struct Cut {   // (ds4_marks.hpp's MarkAt)
    int64_t pos = 0;
    bool shared = false;
};

struct Pic {   // (ds4_main.cpp's Pic)
    int64_t start = 0;
    int width = 0, height = 0;
    uint64_t hash = 0;
    size_t len = 0;
    std::vector<uint8_t> rgb;
};

Pic pic(int64_t start, uint64_t hash, size_t len) {
    Pic p;
    p.start = start;
    p.hash = hash;
    p.len = len;
    return p;
}

}  // namespace

int main() {
    {   // which prompt reads on: the same prompt sent again, or a longer one; not one that ends before the stretch
        assert(!part_way_resumes(PartWay{}, 100));
        const PartWay p{90000, 100000};
        assert(part_way_resumes(p, 100000));
        assert(part_way_resumes(p, 250000));
        assert(!part_way_resumes(p, 99999));
        assert(!part_way_resumes(p, 90000));
        assert(!part_way_resumes(PartWay{100, 100}, 100));   // (nothing past `at` to read)
        assert(part_way_resumes(PartWay{100, 100}, 101));
    }
    {   // its restore points: none in [at, need); the shared start before `at` and the points from `need` on stay
        const PartWay p{90000, 100000};
        std::vector<Cut> cuts{{50, true}, {90000, false}, {95000, false}, {100000, false}, {120000, false}};
        part_way_cuts(cuts, p);
        assert(cuts.size() == 3);
        assert(cuts[0].pos == 50 && cuts[0].shared && cuts[1].pos == 100000 && cuts[2].pos == 120000);
        std::vector<Cut> none;
        part_way_cuts(none, p);
        assert(none.empty());
    }
    {   // how long a STOP waits for the run to end at a chunk
        RunPace pace;
        assert(pace.ms(1000) == 0);
        assert(!stop_cuts_now(19999, pace.ms(1000)) && stop_cuts_now(20001, pace.ms(1000)));   // (no progress: 20 s)
        pace.see(0, 0);                                    // a run starts
        assert(!stop_cuts_now(30000, pace.ms(40000)));     // its first chunk still running: wait for it
        assert(!stop_cuts_now(39000, pace.ms(40000)));
        pace.see(0, 20000);                                // (the same progress again: not a move)
        assert(pace.ms(40000) == 40000);
        pace.see(8192, 40000);                             // the first chunk took 40 s (a cold Engram read)
        assert(pace.ms(41000) == 40000);
        assert(!stop_cuts_now(100000, pace.ms(41000)));    // 2.5 chunks: 100 s
        assert(stop_cuts_now(100001, pace.ms(41000)));
        pace.see(0, 50000);                                // a new run: its pace from its own start
        assert(pace.ms(52000) == 2000);
        pace.see(8192, 53000);                             // warm: 3 s a chunk -> the 20 s floor
        assert(pace.ms(54000) == 3000);
        assert(!stop_cuts_now(19000, pace.ms(54000)) && stop_cuts_now(20001, pace.ms(54000)));
        RunPace layers;                                    // a run that goes layer by layer moves every layer
        layers.see(0, 0);
        layers.see(10, 1000);
        layers.see(20, 2000);
        assert(layers.ms(2500) == 1000);
        assert(stop_cuts_now(20001, layers.ms(2500)));
    }
    {   // the pictures the part-way state holds
        const std::vector<Pic> held{pic(100, 7, 50), pic(300, 8, 40)};   // the state's pictures before the prompt
        const std::vector<Pic> pics{pic(300, 8, 40), pic(90000, 9, 1024), pic(99000, 10, 1024)};
        const std::vector<Pic> h = pics_at(held, pics, 90500);              // stopped inside the picture at 90000
        assert(h.size() == 3);
        assert(h[0].start == 100 && h[1].start == 300 && h[2].start == 90000 && h[2].len == 1024);
        const std::vector<Pic> h2 = pics_at(held, pics, 200);               // back to a whole state at 200
        assert(h2.size() == 1 && h2[0].start == 100);
        assert(pics_at(held, pics, 100000).size() == 4);                    // past every picture
        assert(pics_at(held, std::vector<Pic>{}, 1000).size() == 2);        // (a prompt without pictures)
    }
    std::printf("ds4_keep_cancelled_test: ok\n");
    return 0;
}
