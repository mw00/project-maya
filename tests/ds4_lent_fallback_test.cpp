// ds4_lent_fallback_test (#884): DS4_LENT_RAM_GB's failure paths (ds4_borrow.hpp), CPU only - a fake runtime stands in
// for the GPU's.  A failed copy into the lent experts' RAM copies drops the copies and the lend goes on when every
// device drains; a device that does not drain keeps them and fails (the engine ends); a lent room's clear that failed
// once is made once more after the error is cleared, and a second failure fails.
// Build: g++ -std=c++17 -O2 -I src/ds4 tests/ds4_lent_fallback_test.cpp
#include "ds4_borrow.hpp"

#include <cstdio>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                    \
        }                                                                \
    } while (0)

struct FakeRt {
    std::vector<bool> healthy;   // a device's stream drains
    bool error_pending = true;   // the failed copy's error, until cleared
    bool cleared_first = false;  // the error was cleared before any drain
    std::vector<int> drained;
    bool dropped = false;
    void clear_error() {
        if (drained.empty()) cleared_first = true;
        error_pending = false;
    }
    bool drain(int i) {
        drained.push_back(i);
        return !error_pending && healthy[(size_t) i];
    }
    void drop_copies() { dropped = true; }
};

struct FakeClear {
    int fail_first = 0;   // how many clears fail before one takes
    bool sticky = false;  // the error stays: every clear fails
    int tries = 0, clears_of_error = 0;
    void clear_error() { ++clears_of_error; }
    bool clear() {
        ++tries;
        if (sticky) return false;
        return tries > fail_first;
    }
};

}  // namespace

int main() {
    using strata::ds4::clear_retry;
    using strata::ds4::lent_recover;
    {   // one GPU, healthy: the copies dropped, the request goes on
        FakeRt rt;
        rt.healthy = {true};
        CHECK(lent_recover(rt, 1));
        CHECK(rt.cleared_first);
        CHECK(rt.dropped);
        CHECK(rt.drained.size() == 1);
    }
    {   // three GPUs, all healthy: every stream drained before the drop
        FakeRt rt;
        rt.healthy = {true, true, true};
        CHECK(lent_recover(rt, 3));
        CHECK(rt.dropped);
        CHECK((rt.drained == std::vector<int>{0, 1, 2}));
    }
    {   // the second GPU's state unknown: nothing dropped, false (lend_broken: the engine ends, exit 3)
        FakeRt rt;
        rt.healthy = {true, false, true};
        CHECK(!lent_recover(rt, 3));
        CHECK(!rt.dropped);
        CHECK(rt.drained.size() == 3);   // (every one drained, even after one failed)
    }
    {   // the first GPU's: the same
        FakeRt rt;
        rt.healthy = {false};
        CHECK(!lent_recover(rt, 1));
        CHECK(!rt.dropped);
    }
    {   // a clear that takes: one try, no error cleared
        FakeClear c;
        CHECK(clear_retry(c));
        CHECK(c.tries == 1 && c.clears_of_error == 0);
    }
    {   // a clear that failed once (the out of memory of #884's 4/11 report): the second try takes
        FakeClear c;
        c.fail_first = 1;
        CHECK(clear_retry(c));
        CHECK(c.tries == 2 && c.clears_of_error == 1);
    }
    {   // a sticky error: two tries, then false (lend_broken)
        FakeClear c;
        c.sticky = true;
        CHECK(!clear_retry(c));
        CHECK(c.tries == 2);
    }
    {   // failing twice: false - never a third try
        FakeClear c;
        c.fail_first = 2;
        CHECK(!clear_retry(c));
        CHECK(c.tries == 2);
    }
    if (g_fail == 0) std::printf("ds4_lent_fallback_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}
