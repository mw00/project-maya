// src/k31/k31_ramtier.hpp - maya-k31's RAM expert tier: paged experts read with O_DIRECT into the engine's own pinned,
// device-mapped slots, bypassing the page cache.
//
// With --load page and no tier, a paged expert is read through the file's mapping: page faults and the page cache's
// reclaim with 16 readers into a full cache cost most of the CPU in the kernel (measured on KLX-IQ1_S: stime 648 s
// against utime 225 s) and held the drive to ~1 GB/s.  The tier instead keeps a pool of slots in pinned host memory
// (cudaHostAlloc, mapped: the device reads a slot in place, the CPU lane reads the same bytes), one slot an expert
// (its gate, up and down matrices), an LRU keyed by (layer, expert), and reader threads that pread() each matrix's
// file range with O_DIRECT straight into its slot (start and length rounded out to 4 KB; the matrix sits at an inner
// offset).  What a slot holds is the file's bytes, so what the engine computes is unchanged.
//
// The pool is allocated in chunks (cudaHostAlloc each) up to the size asked for; a refused chunk ends the allocation
// and the pool is what was pinned.  Reads: a high queue (a layer's picks it is about to run) and a low queue (pregate's
// predictions); a slot's three reads complete it (ready).  A slot is never evicted while it is loading or while the
// layer running now uses it.
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace k31 {

class RamTier {
public:
    static constexpr size_t kAlign = 4096;
    enum : int { kEmpty = 0, kLoading = 1, kReady = 2, kFailed = 3 };
    struct Range { int fd = -1; uint64_t off = 0; size_t len = 0; };   // one matrix in the file
    struct Slot {
        int layer = -1, expert = -1;
        std::atomic<int> state{kEmpty};
        std::atomic<int> pending{0};     // reads still to complete
        std::atomic<bool> failed{false}; // a read came short (the slot completes as kFailed once all three are done)
        uint64_t stamp = 0;              // LRU: last use (0: never)
        uint64_t busy = 0;               // the layer run that uses it (not evicted during that run)
        bool hold = false;               // the source of a VRAM upload in flight (not evicted until it is done)
        bool adopted = false;            // filled by the caller (a VRAM victim copied back), not by the readers
        uint8_t* host = nullptr;         // the slot's bytes (pinned, device-mapped)
        uint8_t* dev = nullptr;          // the same bytes' device address
        size_t inner[3] = {0, 0, 0};     // each matrix's offset inside the slot
    };
    struct Stats {
        std::atomic<long long> reads{0}, bytes{0}, fails{0};
        std::atomic<long long> read_ns{0};   // summed over the readers (busy time)
    };

    RamTier() = default;
    RamTier(const RamTier&) = delete;
    RamTier& operator=(const RamTier&) = delete;
    ~RamTier() { stop(); }

    /// The slot layout: each of the three matrices gets max_len[i] rounded out (start and end to kAlign) plus kAlign.
    void layout(const size_t max_len[3]) {
        size_t o = 0;
        for (int i = 0; i < 3; ++i) {
            region_[i] = o;
            o += round_up(max_len[i], kAlign) + 2 * kAlign;
        }
        slot_bytes_ = o;
    }
    size_t slot_bytes() const { return slot_bytes_; }

    /// Takes the pinned chunks (allocated by the caller) and carves them into slots.
    void add_chunk(uint8_t* p, uint8_t* dp, size_t bytes) {
        for (size_t o = 0; o + slot_bytes_ <= bytes; o += slot_bytes_) {
            slots_.emplace_back(new Slot());
            slots_.back()->host = p + o;
            slots_.back()->dev = dp + o;
            free_.push_back((int) slots_.size() - 1);
        }
    }
    size_t n_slots() const { return slots_.size(); }
    Slot& slot(int s) { return *slots_[(size_t) s]; }

    void start(int threads) {
        for (int i = 0; i < threads; ++i) th_.emplace_back([this] { work(); });
    }
    void stop() {
        {
            std::lock_guard<std::mutex> l(mu_);
            stop_ = true;
            hi_.clear();
            lo_.clear();
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
        th_.clear();
    }

    /// A free or evictable slot for (layer, expert), or -1 (every slot loading or busy in run `now_run`).  The victim
    /// is the least recently used one; the caller unmaps the expert it held (returned in old_layer / old_expert).
    int take(int layer, int expert, uint64_t now_run, int& old_layer, int& old_expert) {
        int best = -1;
        uint64_t bs = UINT64_MAX;
        // a clock-free scan is O(slots); a call takes at most a few hundred slots, so keep a free list first
        while (!free_.empty()) {
            const int s = free_.back();
            free_.pop_back();
            if (slots_[(size_t) s]->layer < 0) { best = s; break; }
        }
        if (best < 0) {
            for (size_t i = 0; i < slots_.size(); ++i) {
                Slot& x = *slots_[i];
                if (x.busy == now_run || x.hold || x.state.load(std::memory_order_acquire) == kLoading) continue;
                if (x.stamp < bs) { bs = x.stamp; best = (int) i; }
            }
        }
        if (best < 0) return -1;
        Slot& x = *slots_[(size_t) best];
        old_layer = x.layer;
        old_expert = x.expert;
        x.layer = layer;
        x.expert = expert;
        x.state.store(kEmpty, std::memory_order_release);
        return best;
    }
    /// The slot gives up its expert (a failed read): back on the free list.
    void release(int s) {
        Slot& x = *slots_[(size_t) s];
        x.layer = x.expert = -1;
        x.stamp = 0;
        x.hold = x.adopted = false;
        x.state.store(kEmpty, std::memory_order_release);
        free_.push_back(s);
    }
    /// Slot s (just taken) will be filled by the caller with the three matrices at file offsets off[i] (the same inner
    /// offsets a read would use, so the slot is laid out as if read): loading until adopted_done(s).
    void adopt(int s, const uint64_t off[3]) {
        Slot& x = *slots_[(size_t) s];
        for (int i = 0; i < 3; ++i) x.inner[i] = region_[i] + (size_t) (off[i] & (uint64_t) (kAlign - 1));
        x.adopted = true;
        x.failed.store(false, std::memory_order_relaxed);
        x.state.store(kLoading, std::memory_order_release);
    }
    void adopted_done(int s) {
        Slot& x = *slots_[(size_t) s];
        x.adopted = false;
        x.state.store(kReady, std::memory_order_release);
        std::lock_guard<std::mutex> l(done_mu_);
        done_cv_.notify_all();
    }

    /// Queues the three reads of slot s (high: a layer about to run; low: a prediction).
    void load(int s, const Range r[3], bool high) {
        Slot& x = *slots_[(size_t) s];
        x.pending.store(3, std::memory_order_relaxed);
        x.failed.store(false, std::memory_order_relaxed);
        x.state.store(kLoading, std::memory_order_release);
        {
            std::lock_guard<std::mutex> l(mu_);
            for (int i = 0; i < 3; ++i) {
                const uint64_t a = r[i].off & ~(uint64_t) (kAlign - 1);
                x.inner[i] = region_[i] + (size_t) (r[i].off - a);
                Job j{s, r[i].fd, a, round_up((size_t) (r[i].off + r[i].len - a), kAlign),
                      (size_t) (r[i].off + r[i].len - a), x.host + region_[i]};
                (high ? hi_ : lo_).push_back(j);
            }
        }
        cv_.notify_all();
    }
    /// Moves slot s's reads still waiting in the low queue to the front of the high one (its layer runs now).
    void promote(int s) {
        std::lock_guard<std::mutex> l(mu_);
        for (auto it = lo_.begin(); it != lo_.end();) {
            if (it->slot == s) { hi_.push_front(*it); it = lo_.erase(it); }
            else ++it;
        }
    }
    /// Blocks until slot s is no longer loading; returns its state (kReady or kFailed).
    int wait(int s) {
        Slot& x = *slots_[(size_t) s];
        int st = x.state.load(std::memory_order_acquire);
        if (st != kLoading) return st;
        std::unique_lock<std::mutex> l(done_mu_);
        done_cv_.wait(l, [&] { return (st = x.state.load(std::memory_order_acquire)) != kLoading; });
        return st;
    }
    const Stats& stats() const { return stats_; }

private:
    struct Job {
        int slot;
        int fd;
        uint64_t off;     // aligned start
        size_t len;       // aligned length
        size_t need;      // the bytes that must arrive (a short read past the end of the file is fine after them)
        uint8_t* dst;     // aligned
    };
    static size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
    void work() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> l(mu_);
                cv_.wait(l, [&] { return stop_ || !hi_.empty() || !lo_.empty(); });
                if (stop_) return;
                std::deque<Job>& q = hi_.empty() ? lo_ : hi_;
                j = q.front();
                q.pop_front();
            }
            const auto t0 = std::chrono::steady_clock::now();
            size_t got = 0;
            while (got < j.len) {
                const ssize_t n = ::pread(j.fd, j.dst + got, j.len - got, (off_t) (j.off + got));
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) break;
                got += (size_t) n;
                if (got >= j.need && (size_t) n % kAlign) break;   // the end of the file
            }
            stats_.read_ns += (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            ++stats_.reads;
            stats_.bytes += (long long) got;
            Slot& x = *slots_[(size_t) j.slot];
            if (got < j.need) {
                ++stats_.fails;
                x.failed.store(true, std::memory_order_release);
            }
            // the slot completes only when all three reads are done: none may still write into it after a waiter
            // sees it ready or failed (and maybe gives it to another expert)
            if (x.pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                x.state.store(x.failed.load(std::memory_order_acquire) ? kFailed : kReady, std::memory_order_release);
                std::lock_guard<std::mutex> l(done_mu_);
                done_cv_.notify_all();
            }
        }
    }

    size_t region_[3] = {0, 0, 0};
    size_t slot_bytes_ = 0;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<int> free_;
    std::mutex mu_, done_mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Job> hi_, lo_;
    std::vector<std::thread> th_;
    bool stop_ = false;
    Stats stats_;
};

}  // namespace k31
