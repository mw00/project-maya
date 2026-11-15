// The stall report's watcher (include/strata/platform/stall_watch.hpp, #860): a step past the limit is reported once
// (then again only after again_s), with the phase it is in; steps that keep moving and an idle loop are never
// reported; a limit of 0 starts no thread.  Host code only, no GPU.
#include "strata/platform/stall_watch.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

using strata::platform::StallWatch;

static int failures = 0;

static void expect(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
}

static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

int main() {
    {   // off: no thread, no report
        std::atomic<int> n{0};
        StallWatch w(0, [&](const char*, double) { ++n; });
        expect(!w.on(), "a limit of 0 starts no watcher");
        w.step("x");
        sleep_ms(50);
        expect(n.load() == 0, "a limit of 0 never reports");
    }
    {   // a stuck step: one report (at the limit, with its phase), not one a tick
        std::atomic<int> n{0};
        std::mutex mu;
        std::string seen;
        double secs = 0.0;
        StallWatch w(1, [&](const char* ph, double s) {
            std::lock_guard<std::mutex> lk(mu);
            seen = ph;
            secs = s;
            ++n;
        }, 60, 20);
        expect(w.on(), "a limit of 1 s starts the watcher");
        w.step("the tail's token");
        w.phase("the draft");
        sleep_ms(1600);
        expect(n.load() == 1, "a step stuck past the limit is reported exactly once");
        std::lock_guard<std::mutex> lk(mu);
        expect(seen == "the draft", "the report names the step's current phase");
        expect(secs >= 1.0 && secs < 1.5, "the report gives the step's run time");
    }
    {   // the same stuck step again after again_s
        std::atomic<int> n{0};
        StallWatch w(1, [&](const char*, double) { ++n; }, 1, 20);
        w.step("a");
        sleep_ms(2300);
        expect(n.load() == 2, "a step still stuck is reported again after again_s");
    }
    {   // steps that keep moving (each under the limit) and an idle loop: nothing
        std::atomic<int> n{0};
        StallWatch w(1, [&](const char*, double) { ++n; }, 60, 20);
        for (int i = 0; i < 12; ++i) {
            w.step("a");
            sleep_ms(150);
        }
        w.idle();
        sleep_ms(1300);
        expect(n.load() == 0, "moving steps and an idle loop are never reported");
        expect(w.reports() == 0, "reports() counts none");
    }
    {   // a new stuck step after a reported one is reported for itself
        std::atomic<int> n{0};
        StallWatch w(1, [&](const char*, double) { ++n; }, 60, 20);
        w.step("a");
        sleep_ms(1300);
        w.step("b");
        sleep_ms(1300);
        expect(n.load() == 2, "each stuck step is reported");
    }
    if (failures == 0) std::printf("stall_watch_test: all passed\n");
    return failures == 0 ? 0 : 1;
}
