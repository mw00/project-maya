// src/k31/k31_prefetch.hpp - maya-k31: reads paged experts' file ranges into the page cache ahead of their use.
//
// With --load page an expert without a VRAM slot is read through the file's mapping, and a page fault reads ~128 KB
// at a time, one fault per touching thread: the NVMe sits mostly idle.  Prefetcher keeps many large reads in flight
// instead: worker threads pread() the ranges (not O_DIRECT, so the pages land in the page cache the mapping shares)
// in pieces of up to 4 MB.  Two queues: `high` for the experts a layer's router has just picked, `low` for predicted
// ones (pre-gating); a worker always takes high first.  A range already queued or in flight is not added again.
// The reads only warm the cache: what the engine computes is unchanged.
#pragma once

#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace k31 {

class Prefetcher {
public:
    explicit Prefetcher(int threads) {
        for (int i = 0; i < threads; ++i) th_.emplace_back([this] { work(); });
    }
    ~Prefetcher() {
        {
            std::lock_guard<std::mutex> l(mu_);
            stop_ = true;
            hi_.clear();
            lo_.clear();
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    Prefetcher(const Prefetcher&) = delete;
    Prefetcher& operator=(const Prefetcher&) = delete;

    void add(int fd, uint64_t off, size_t len, bool high) {
        {
            std::lock_guard<std::mutex> l(mu_);
            if (!known_.insert({fd, off}).second) { ++dup_; return; }
            (high ? hi_ : lo_).push_back({fd, off, len});
            ++queued_;
            queued_bytes_ += (long long) len;
        }
        cv_.notify_one();
    }
    // the low queue's waiting ranges dropped (stale predictions); returns how many
    long long drop_low() {
        std::lock_guard<std::mutex> l(mu_);
        const long long n = (long long) lo_.size();
        for (const Req& r : lo_) known_.erase({r.fd, r.off});
        lo_.clear();
        dropped_ += n;
        return n;
    }
    struct Stats { long long queued = 0, queued_bytes = 0, read_bytes = 0, dup = 0, dropped = 0; };
    Stats stats() const {
        Stats s;
        s.queued = queued_; s.queued_bytes = queued_bytes_; s.read_bytes = read_bytes_; s.dup = dup_; s.dropped = dropped_;
        return s;
    }

private:
    struct Req { int fd; uint64_t off; size_t len; };
    void work() {
        std::vector<uint8_t> buf((size_t) 4 << 20);
        for (;;) {
            Req r;
            {
                std::unique_lock<std::mutex> l(mu_);
                cv_.wait(l, [&] { return stop_ || !hi_.empty() || !lo_.empty(); });
                if (stop_) return;
                std::deque<Req>& q = hi_.empty() ? lo_ : hi_;
                r = q.front();
                q.pop_front();
            }
            for (size_t got = 0; got < r.len;) {
                const ssize_t n = ::pread(r.fd, buf.data(), std::min(buf.size(), r.len - got), (off_t) (r.off + got));
                if (n <= 0) break;   // (a failed read only means the fault will read it later)
                got += (size_t) n;
                read_bytes_ += n;
            }
            std::lock_guard<std::mutex> l(mu_);
            known_.erase({r.fd, r.off});
        }
    }
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::deque<Req> hi_, lo_;
    std::set<std::pair<int, uint64_t>> known_;   // queued or in flight
    std::vector<std::thread> th_;
    std::atomic<long long> queued_{0}, queued_bytes_{0}, read_bytes_{0}, dup_{0}, dropped_{0};
};

}  // namespace k31
