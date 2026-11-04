// include/strata/platform/idle_gate.hpp - how an idle engine sleeps without making its first request wait.
//
// The service threads (the CPU lane's, one a GPU) poll a ring in pinned memory that the GPU writes its routes to; no
// event reaches the CPU when one lands.  While requests come, they spin or poll every few microseconds.  When nothing
// has come for a minute (a server nobody is talking to) a poll every few microseconds is a core woken thousands of
// times a second for nothing - a busy-looking process on a quiet machine, and its power bill.  So after a long quiet
// they sleep on this gate for up to `deep_poll_ms` at a time instead, and the thread that reads the engine's input
// (the next request's line) pokes it: the sleepers wake at once, long before the request's first route can arrive,
// and the request waits for nothing.  A request that reaches the engine some other way is at most one deep poll late.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace strata::platform {

class IdleGate {
public:
    // A request (or any input) reached the engine: sleepers wake now.  Cheap: one atomic add, and the lock only when
    // someone sleeps.
    void poke() {
        epoch_.fetch_add(1, std::memory_order_release);
        if (sleepers_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
    }
    // The epoch a sleeper compares: read it before the last look at the ring, sleep() returns at once if a poke came
    // since (a poke between the look and the sleep is not lost).
    uint64_t epoch() const { return epoch_.load(std::memory_order_acquire); }
    // Sleeps up to `ms`, or until a poke after `seen`.  True if poked.
    bool sleep(int ms, uint64_t seen) {
        sleepers_.fetch_add(1, std::memory_order_acq_rel);
        std::unique_lock<std::mutex> lk(mu_);
        const bool poked = cv_.wait_for(lk, std::chrono::milliseconds(ms),
                                        [&] { return epoch_.load(std::memory_order_acquire) != seen; });
        lk.unlock();
        sleepers_.fetch_sub(1, std::memory_order_acq_rel);
        return poked;
    }

private:
    std::atomic<uint64_t> epoch_{0};
    std::atomic<int> sleepers_{0};
    std::mutex mu_;
    std::condition_variable cv_;
};

inline IdleGate& idle_gate() {
    static IdleGate g;
    return g;
}

inline void idle_poke() { idle_gate().poke(); }

// The deep idle's settings (seconds of quiet before it starts: 0 = never, the poll's milliseconds), from the
// environment under either name: STRATA_<prefix>_SERVICE_DEEP_S / _DEEP_MS (prefix GLM, DS4).
inline int idle_env(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' ? std::max(0, std::atoi(v)) : fallback;
}

}  // namespace strata::platform
