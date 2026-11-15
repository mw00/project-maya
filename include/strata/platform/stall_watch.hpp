// include/strata/platform/stall_watch.hpp - a watcher for a loop of short steps (a decode's tokens): when one step runs
// longer than a limit, a callback says what the engine is doing - once at the limit, then every `again_s` while the
// same step is still stuck.  The loop only stores two atomics a step (step() / phase()); the watcher is its own thread
// that wakes once a second, so the steps' timing and their results do not change.  Header only, host code (#860: a
// two-GPU drafting decode that crawled at its first step on one PC, with nothing in the log to say where it waited).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace strata::platform {

// MAYA_GLM_STALL_S: the limit in seconds (unset: 20; 0 or less: no watcher)
inline int stall_limit_s(const char* name = "MAYA_GLM_STALL_S", int dflt = 20) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return dflt;
    return std::atoi(v);
}

class StallWatch {
public:
    // report(phase, seconds the step has run): called on the watcher's thread
    using Report = std::function<void(const char* phase, double seconds)>;

    StallWatch(int limit_s, Report report, int again_s = 60, int tick_ms = 1000)
        : limit_ms_((int64_t) limit_s * 1000), again_ms_((int64_t) again_s * 1000), tick_ms_(tick_ms),
          report_(std::move(report)) {
        if (limit_s > 0 && report_) th_ = std::thread([this] { run(); });
    }
    ~StallWatch() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    StallWatch(const StallWatch&) = delete;
    StallWatch& operator=(const StallWatch&) = delete;

    bool on() const { return th_.joinable(); }
    // a new step starts now (its clock restarts), in this phase
    void step(const char* phase) {
        phase_.store(phase, std::memory_order_relaxed);
        since_.store(now_ms(), std::memory_order_release);
    }
    // the step moves on to another phase (its clock keeps running)
    void phase(const char* phase) { phase_.store(phase, std::memory_order_relaxed); }
    // no step is running (between requests): nothing to report
    void idle() { since_.store(-1, std::memory_order_release); }
    uint64_t reports() const { return reports_.load(std::memory_order_acquire); }

private:
    static int64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
    void run() {
        int64_t said_for = -1, next_ms = 0;   // the step reported last (its start) and when to say it again
        std::unique_lock<std::mutex> lk(mu_);
        while (!quit_) {
            cv_.wait_for(lk, std::chrono::milliseconds(tick_ms_));
            if (quit_) break;
            const int64_t s = since_.load(std::memory_order_acquire);
            if (s < 0) continue;
            const int64_t el = now_ms() - s;
            if (el < limit_ms_) continue;
            if (said_for == s && el < next_ms) continue;
            said_for = s;
            next_ms = el + again_ms_;
            report_(phase_.load(std::memory_order_relaxed), (double) el / 1000.0);
            reports_.fetch_add(1, std::memory_order_release);
        }
    }

    const int64_t limit_ms_, again_ms_;
    const int tick_ms_;
    Report report_;
    std::atomic<int64_t> since_{-1};
    std::atomic<const char*> phase_{""};
    std::atomic<uint64_t> reports_{0};
    std::mutex mu_;
    std::condition_variable cv_;
    bool quit_ = false;
    std::thread th_;
};

}  // namespace strata::platform
