// The idle gate (include/strata/platform/idle_gate.hpp, #685): an idle engine's service threads sleep on it after a long
// quiet, and a line read from the engine's input wakes them at once.  Host code only: threads and a clock, no GPU.
#include "strata/platform/idle_gate.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using strata::platform::IdleGate;
using Clock = std::chrono::steady_clock;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
}

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main() {
    // no poke: the sleep lasts its time, and says it was not poked
    {
        IdleGate g;
        const auto t0 = Clock::now();
        const bool poked = g.sleep(60, g.epoch());
        check(!poked, "a sleep nobody poked reports no poke");
        check(ms_since(t0) >= 55.0, "a sleep nobody poked lasts its time");
    }
    // a poke that came after the epoch was read ends the sleep at once (the one between the last look and the sleep)
    {
        IdleGate g;
        const uint64_t seen = g.epoch();
        g.poke();
        const auto t0 = Clock::now();
        check(g.sleep(5000, seen), "a poke before the sleep is not lost");
        check(ms_since(t0) < 500.0, "... and the sleep returns at once");
    }
    // a poke during the sleep wakes the sleeper long before its time is up
    {
        IdleGate g;
        double waited = 0.0;
        bool poked = false;
        std::thread sleeper([&] {
            const uint64_t seen = g.epoch();
            const auto t0 = Clock::now();
            poked = g.sleep(10000, seen);
            waited = ms_since(t0);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        const auto t_poke = Clock::now();
        g.poke();
        sleeper.join();
        check(poked, "a poke during the sleep is reported");
        check(waited < 2000.0, "the sleeper woke long before its 10 s were up");
        check(ms_since(t_poke) < 2000.0, "... and the poke did not wait for it");
    }
    // many sleepers (a service thread a GPU), one poke: all of them wake
    {
        IdleGate g;
        std::vector<std::thread> th;
        std::vector<int> woke(6, 0);
        for (int i = 0; i < 6; ++i)
            th.emplace_back([&, i] { woke[(size_t) i] = g.sleep(10000, g.epoch()) ? 1 : 0; });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        g.poke();
        for (auto& t : th) t.join();
        int n = 0;
        for (int w : woke) n += w;
        check(n == 6, "one poke wakes every sleeper");
    }
    // pokes and sleeps racing: nothing hangs, every sleep returns (poked or timed out)
    {
        IdleGate g;
        std::atomic<bool> stop{false};
        std::thread poker([&] {
            while (!stop.load()) {
                g.poke();
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        });
        const auto t0 = Clock::now();
        for (int i = 0; i < 300; ++i) g.sleep(50, g.epoch());   // (Windows' timer: a poke every ~15 ms)
        stop.store(true);
        poker.join();
        check(ms_since(t0) < 20000.0, "racing pokes and sleeps finish");
    }
    // the deep idle's settings: one that is not set is the fallback
    {
        check(strata::platform::idle_env("STRATA_IDLE_GATE_TEST_UNSET_VARIABLE", 60) == 60, "an unset setting is the fallback");
    }
    if (failures == 0) std::printf("idle_gate_test: ok\n");
    return failures == 0 ? 0 : 1;
}
