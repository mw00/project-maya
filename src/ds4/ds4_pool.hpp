// src/ds4/ds4_pool.hpp - the DeepSeek engine's CPU workers: run(n, fn) runs fn(0..n-1) on the workers and the caller and
// returns when every job is done.  The design is glmfast::Workers' (src/core/glm_fast_state.hpp): one 64-bit ticket
// (generation << 32 | next index) claimed by CAS, so a worker that wakes late for an older batch can never take a job
// of the current one; idle workers spin for a while, then sleep on a condition variable.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "ds4_os.hpp"
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace strata::ds4 {

inline void cpu_relax() {
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#endif
}

class Pool {
public:
    explicit Pool(int n, const std::vector<int>& cpus = {}, int spin_us = 200) : spin_us_(spin_us) {
        for (int i = 0; i < n; ++i) {
            th_.emplace_back([this] { loop(); });
            if (!cpus.empty()) os::pin_thread(th_.back(), cpus[(size_t) i % cpus.size()]);
        }
    }
    ~Pool() {
        quit_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return (int) th_.size() + 1; }

    void run(int n, const std::function<void(int)>& fn) {
        start(n, fn);
        help();
        wait();
    }
    /// The batch goes to the workers; the caller may help() and must wait() before fn goes out of scope.  (Two pools
    /// - one a NUMA node - run at once this way: start both, help one, wait for both.)
    void start(int n, const std::function<void(int)>& fn) {
        n_ = n;
        if (n <= 0) return;
        const uint64_t g = (gen_.load(std::memory_order_relaxed) + 1) & 0xffffffffu;
        ticket_.store((g << 32) | 0x7fffffffu, std::memory_order_seq_cst);
        fn_ = &fn;
        total_.store(n, std::memory_order_release);
        done_.store(0, std::memory_order_relaxed);
        ticket_.store(g << 32, std::memory_order_release);
        gen_.store(g, std::memory_order_release);
        if (sleeping_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
    }
    void help() {
        if (n_ > 0) work(gen_.load(std::memory_order_acquire));
    }
    void wait() {
        if (n_ <= 0) return;
        while (done_.load(std::memory_order_acquire) < n_) cpu_relax();
        n_ = 0;
    }

private:
    void work(uint64_t g) {
        for (;;) {
            uint64_t t = ticket_.load(std::memory_order_acquire);
            if ((t >> 32) != g) return;
            const int n = total_.load(std::memory_order_acquire);
            if ((int) (t & 0xffffffffu) >= n) return;
            if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            (*fn_)((int) (t & 0xffffffffu));
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    void loop() {
        uint64_t seen = gen_.load();
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            uint64_t g;
            for (;;) {
                g = gen_.load(std::memory_order_acquire);
                if (g != seen || quit_.load(std::memory_order_relaxed)) break;
                cpu_relax();
                if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_)) {
                    std::unique_lock<std::mutex> lk(mu_);
                    sleeping_.fetch_add(1);
                    cv_.wait(lk, [&] { return gen_.load() != seen || quit_.load(); });
                    sleeping_.fetch_sub(1);
                }
            }
            if (quit_.load()) return;
            seen = g;
            work(g);
        }
    }

    std::vector<std::thread> th_;
    int spin_us_;
    int n_ = 0;   // the current batch's size (the caller's side)
    const std::function<void(int)>* fn_ = nullptr;
    alignas(64) std::atomic<uint64_t> ticket_{0};
    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int> total_{0};
    alignas(64) std::atomic<int> done_{0};
    std::atomic<int> sleeping_{0};
    std::atomic<bool> quit_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

/// (DS4_FLOW_DOWN_PART) rows [0, rows) of a matrix `pitch` bytes a row: each row's bytes [off, off + len) moved to the
/// front of the same buffer, side by side (row r to r len).  In place: row r's target ends at (r + 1) len, never past
/// row r + 1's bytes ((r + 1) pitch + off, len <= pitch), and memmove takes a row's overlap with itself
inline void compact_rows(uint8_t* base, size_t rows, size_t pitch, size_t off, size_t len) {
    for (size_t r = 0; r < rows; ++r) std::memmove(base + r * len, base + r * pitch + off, len);
}

/// File reads on threads of their own, in the order they were pushed, the pusher never waiting: the prompt path's
/// streamed experts (each part of an expert a job), unbuffered where the job has a direct descriptor and its range is
/// not all in the page cache already (the disk straight into dst, the cache neither used nor disturbed), buffered
/// otherwise or when the unbuffered read fails - the bytes land at dst + off % os::kDirectAlign either way.  A job's `left` is
/// counted down when it is done (`bad` set if it failed); wait_any() sleeps until some job finished or the timeout.
class IoQueue {
public:
    struct Job {
        os::fd_t fd = os::kBadFd, dfd = os::kBadFd;
        uint64_t off = 0, len = 0;
        uint8_t* dst = nullptr;
        std::atomic<int>* left = nullptr;
        std::atomic<bool>* bad = nullptr;
        const uint8_t* mem = nullptr;   // non-null: a copy of len bytes from memory to dst instead of a read
        const void* map = nullptr;      // the range in the file's mapping: all in the page cache, it is read buffered
        uint8_t* keep = nullptr;        // non-null (a read): the len bytes read copied here as well
        // non-null: a copy of len bytes out of device ddev's memory to dst by dcopy (the caller's, so this header
        // needs no GPU runtime) instead of a read - DS4_FLOW_HELPER's experts out of the helper's pool
        const void* dmem = nullptr;
        int ddev = -1;
        bool (*dcopy)(int dev, void* dst, const void* src, size_t n) = nullptr;
        // non-null (a buffered read several readers share - DS4_FLOW_DOWN_SHARE): counted once read (or failed); the
        // share_n-th count drops the range from the page cache (fd, off, len) and starts the count again
        std::atomic<uint8_t>* share = nullptr;
        int share_n = 0;
        // nonzero (DS4_FLOW_DOWN_PART): once read (or copied), the bytes are crows rows cpitch apart, of which each
        // row's [coff, coff + clen) is kept - compacted to the front of where they landed (compact_rows), before `left`
        // counts
        size_t crows = 0, cpitch = 0, coff = 0, clen = 0;
    };
    explicit IoQueue(int n) {
        for (int i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~IoQueue() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    void push(const Job* jobs, int n) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < n; ++i) q_.push_back(jobs[i]);
        }
        if (n == 1) cv_.notify_one();
        else cv_.notify_all();
    }
    void wait_any(int us) {
        std::unique_lock<std::mutex> lk(done_mu_);
        const uint64_t seen = done_;
        done_cv_.wait_for(lk, std::chrono::microseconds(us), [&] { return done_ != seen; });
    }
    double busy_ms() const { return busy_us_.load(std::memory_order_relaxed) / 1000.0; }
    uint64_t direct_bytes() const { return direct_.load(std::memory_order_relaxed); }
    uint64_t buffered_bytes() const { return buffered_.load(std::memory_order_relaxed); }
    uint64_t device_bytes() const { return device_.load(std::memory_order_relaxed); }

private:
    void loop() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || !q_.empty(); });
                if (quit_ && q_.empty()) return;
                j = q_.front();
                q_.pop_front();
            }
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t lead = j.off % os::kDirectAlign;
            bool ok = false;
            if (j.dmem != nullptr) {
                ok = j.dcopy != nullptr && j.dcopy(j.ddev, j.dst, j.dmem, (size_t) j.len);
                if (ok) device_.fetch_add(j.len, std::memory_order_relaxed);
            } else if (j.mem != nullptr) {
                std::memcpy(j.dst, j.mem, (size_t) j.len);
                ok = true;
            } else if (j.dfd != os::kBadFd && !(j.map != nullptr && os::resident(j.map, (size_t) j.len)) &&
                       os::pread_direct(j.dfd, j.dst, j.len, j.off)) {   // (the page cache's ones: buffered, below)
                ok = true;
                direct_.fetch_add(j.len, std::memory_order_relaxed);
            } else if (os::pread_par(j.fd, j.dst + lead, j.len, j.off)) {
                ok = true;
                buffered_.fetch_add(j.len, std::memory_order_relaxed);
            }
            if (ok && j.keep != nullptr && j.mem == nullptr && j.dmem == nullptr)
                std::memcpy(j.keep, j.dst + lead, (size_t) j.len);
            if (ok && j.crows > 0)   // (a copy's bytes land at dst, a read's at dst + lead)
                compact_rows(j.dst + (j.mem != nullptr || j.dmem != nullptr ? 0 : lead), j.crows, j.cpitch, j.coff,
                             j.clen);
            if (j.share != nullptr && (int) j.share->fetch_add(1, std::memory_order_acq_rel) + 1 >= j.share_n) {
                j.share->store(0, std::memory_order_release);
                if (j.fd != os::kBadFd) os::advise(j.fd, j.off, j.len, false);
            }
            busy_us_.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - t0).count(),
                               std::memory_order_relaxed);
            if (!ok && j.bad != nullptr) j.bad->store(true, std::memory_order_release);
            if (j.left != nullptr) j.left->fetch_sub(1, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lk(done_mu_);
                ++done_;
            }
            done_cv_.notify_all();
        }
    }
    std::vector<std::thread> th_;
    std::mutex mu_, done_mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Job> q_;
    uint64_t done_ = 0;
    bool quit_ = false;
    std::atomic<uint64_t> busy_us_{0}, direct_{0}, buffered_{0}, device_{0};
};

}  // namespace strata::ds4
