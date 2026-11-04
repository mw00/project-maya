// maya-ds4 --serve's watchdog rule (src/ds4/ds4_watchdog.hpp, #539): it fires on a request that stopped moving, never
// on a long one that keeps moving, never while no request runs.  No GPU, no model.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ds4/ds4_watchdog.hpp"

#include <cassert>
#include <cstdio>

using strata::ds4::StallRule;

int main() {
    // a slow PC's 1M-token prompt: a chunk every 500 s for ~14 hours - moving all the time, never a stall
    {
        StallRule r;
        uint64_t beats = 1;
        for (double t = 0; t < 50000; t += 1) {
            if (t > 0 && (int) t % 500 == 0) ++beats;
            assert(!r.stalled(true, beats, t));
        }
    }
    // a decode at a token every 0.25 s for an hour
    {
        StallRule r;
        uint64_t beats = 0;
        for (double t = 0; t < 3600; t += 0.25) assert(!r.stalled(true, ++beats, t));
    }
    // a request that moved, then stopped (#539: a chunk that never ends): it fires 600 s after the last beat, not before
    {
        StallRule r;
        uint64_t beats = 0;
        double t = 0;
        for (; t < 120; t += 1) assert(!r.stalled(true, ++beats, t));
        const double last = t - 1;
        bool fired = false;
        for (; t < 2000 && !fired; t += 1) {
            fired = r.stalled(true, beats, t);
            assert(fired == (t - last >= 600));
        }
        assert(fired);
    }
    // a stall from the request's first beat (#494: the first prompt after a start never reads a chunk)
    {
        StallRule r;
        assert(!r.stalled(true, 7, 0));
        assert(!r.stalled(true, 7, 599));
        assert(r.stalled(true, 7, 600));
    }
    // idle between requests for a day: never; the next request's clock starts at its own first look
    {
        StallRule r;
        for (double t = 0; t < 86400; t += 10) assert(!r.stalled(false, 3, t));
        assert(!r.stalled(true, 4, 86400));
        assert(!r.stalled(true, 4, 86400 + 599));
        assert(r.stalled(true, 4, 86400 + 600));
    }
    // off
    {
        StallRule r;
        r.limit_s = 0;
        for (double t = 0; t < 100000; t += 100) assert(!r.stalled(true, 1, t));
    }
    // a shorter limit (MAYA_DS4_WATCHDOG_S=60)
    {
        StallRule r;
        r.limit_s = 60;
        assert(!r.stalled(true, 1, 0));
        assert(!r.stalled(true, 1, 59.5));
        assert(r.stalled(true, 1, 60));
    }
    std::printf("ds4_watchdog_test: OK\n");
    return 0;
}
