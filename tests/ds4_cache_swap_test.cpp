#ifdef NDEBUG
#undef NDEBUG
#endif
#include "ds4_cache_swap.hpp"
#include <array>
#include <cassert>
#include <cstdio>

// Inject failure at the production prepare_cache_swap boundary: source read, first or later member's copy/wait.
// The outgoing active slot must remain usable even when the spare was already partially written.
int main() {
    for (int failure = -1; failure < 3; ++failure) {
        std::array<int, 2> active{41, 41}, spare{-7, -7};
        int read_calls = 0, copy_calls = 0, inflight = 1, incoming = 9;
        bool pending = true;
        const bool ready = strata::ds4::prepare_cache_swap(2, [&] {
            ++read_calls;
            return failure != 0;
        }, [&](int member) {
            ++copy_calls;
            spare[(size_t) member] = 99;  // enqueue may already have modified part of the spare before wait fails
            return failure != member + 1;
        });
        assert(read_calls == 1);
        assert(ready == (failure == -1));
        if (!strata::ds4::discard_failed_cache_swap(ready, pending, incoming, inflight)) {
            active = spare;  // the production cache_switch's publication branch
            pending = false;
            incoming = -1;
            --inflight;
        }
        assert(!pending && incoming == -1 && inflight == 0);
        assert(active == (failure == -1 ? std::array<int, 2>{99, 99} : std::array<int, 2>{41, 41}));
        assert(copy_calls == (failure == 0 ? 0 : failure == 1 ? 1 : 2));
        // Failed reservations can be retried; readiness must not leak from the previous attempted move.
        pending = true; incoming = 10; inflight = 1;
        const bool retry = strata::ds4::prepare_cache_swap(2, [] { return true; }, [](int) { return true; });
        assert(retry && !strata::ds4::discard_failed_cache_swap(retry, pending, incoming, inflight));
        assert(pending && incoming == 10 && inflight == 1);
    }
    std::puts("ds4_cache_swap_test: ok");
}
