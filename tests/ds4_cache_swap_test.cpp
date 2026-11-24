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
        const bool ready = maya::ds4::prepare_cache_swap(2, [&] {
            ++read_calls;
            return failure != 0;
        }, [&](int member) {
            ++copy_calls;
            spare[(size_t) member] = 99;  // enqueue may already have modified part of the spare before wait fails
            return failure != member + 1;
        });
        assert(read_calls == 1);
        assert(ready == (failure == -1));
        if (!maya::ds4::discard_failed_cache_swap(ready, pending, incoming, inflight)) {
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
        const bool retry = maya::ds4::prepare_cache_swap(2, [] { return true; }, [](int) { return true; });
        assert(retry && !maya::ds4::discard_failed_cache_swap(retry, pending, incoming, inflight));
        assert(pending && incoming == 10 && inflight == 1);
    }
    // The table inputs (tick_flush): in turn; an input whose reader is done is taken without a wait; one whose reader
    // still runs is waited for, its own event only; an error at either step gives -1 (the caller aborts).
    {
        constexpr int kSlots = 4;
        std::array<int, kSlots> state{0, 0, 0, 0};   // the event's query: 0 done, 1 not yet, -1 an error
        int next = 0, waits = 0, last_wait = -1;
        bool wait_ok = true;
        auto take = [&] {
            return maya::ds4::take_table_input(next, kSlots, [&](int k) { return state[(size_t) k]; },
                                                 [&](int k) {
                                                     ++waits;
                                                     last_wait = k;
                                                     if (wait_ok) state[(size_t) k] = 0;
                                                     return wait_ok;
                                                 });
        };
        for (int t = 0; t < 2 * kSlots; ++t) {   // ordinary ticks: every reader done before its input comes round
            assert(take() == t % kSlots);
            state[(size_t) (t % kSlots)] = 0;
        }
        assert(waits == 0);
        for (int t = 0; t < kSlots; ++t) {   // back-to-back flushes: each input's reader still queued
            assert(take() == t);
            state[(size_t) t] = 1;
        }
        assert(waits == 0);
        assert(take() == 0 && waits == 1 && last_wait == 0);   // (the oldest input: its own reader waited for)
        assert(take() == 1 && waits == 2 && last_wait == 1);
        state[2] = -1;
        assert(take() == -1 && waits == 2);   // (a query error: no wait, an error)
        wait_ok = false;
        assert(take() == -1 && waits == 3 && last_wait == 3);   // (a wait error)
        assert(next == 0);
    }
    std::puts("ds4_cache_swap_test: ok");
}
