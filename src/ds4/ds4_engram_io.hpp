// src/ds4/ds4_engram_io.hpp - DeepSeek's Engram (n-gram memory) rows read off the decode's critical path.
//
// A token's Engram rows depend only on the token ids, so they are asked for as soon as the window's tokens are known
// and collected only right before the layer that adds them (layer 1's hide behind the embedding and layer 0, layer
// 14's behind layers 0-13).  The reads run on the reader's own threads, each with its own handles to the model's files
// (on Windows the ReadFile calls on one synchronous handle run one at a time), FIFO: the first Engram layer's rows
// first.  A bounded row cache (clock eviction) keeps rows already fetched: a row that recurs (the same n-gram again,
// in the prompt or later) is a copy, not a read.  The rows are the file's bytes either way, so the output is the same.
#pragma once
#if !defined(_WIN32)
#include <fcntl.h>
#endif

#include "ds4_os.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <limits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::ds4 {

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Rows already fetched: key -> `bytes` of payload, at most `cap` rows (grown on demand in blocks), clock eviction.
/// Not thread-safe (the caller locks).
class EgRowCache {
public:
    void init(size_t cap, size_t bytes) {
        cap_ = cap;
        bytes_ = bytes;
        map_.reserve(std::min<size_t>(cap, 1u << 20));
    }
    size_t capacity() const { return cap_; }
    size_t size() const { return keys_.size(); }
    size_t row_bytes() const { return bytes_; }
    /// The row's payload, or nullptr (a hit marks it recently used).
    const uint8_t* find(uint64_t key) {
        const auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        ref_[it->second] = 1;
        return at(it->second);
    }
    /// Room for key's payload (the caller fills it): a new slot while below capacity, else the clock's victim.
    uint8_t* put(uint64_t key) {
        if (cap_ == 0 || bytes_ == 0) return nullptr;
        const auto it = map_.find(key);
        if (it != map_.end()) return at(it->second);
        uint32_t s;
        if (keys_.size() < cap_) {
            s = (uint32_t) keys_.size();
            if (s % kBlock == 0) blocks_.emplace_back(new uint8_t[(size_t) kBlock * bytes_]);
            keys_.push_back(key);
            ref_.push_back(0);
        } else {
            while (ref_[hand_]) {   // (a second chance for rows used since the hand last passed)
                ref_[hand_] = 0;
                hand_ = (hand_ + 1) % (uint32_t) cap_;
            }
            s = hand_;
            hand_ = (hand_ + 1) % (uint32_t) cap_;
            map_.erase(keys_[s]);
            keys_[s] = key;
            ref_[s] = 0;
        }
        map_.emplace(key, s);
        return at(s);
    }

private:
    static constexpr uint32_t kBlock = 4096;
    uint8_t* at(uint32_t s) { return blocks_[s / kBlock].get() + (size_t) (s % kBlock) * bytes_; }
    size_t cap_ = 0, bytes_ = 0;
    uint32_t hand_ = 0;
    std::unordered_map<uint64_t, uint32_t> map_;
    std::vector<uint64_t> keys_;
    std::vector<uint8_t> ref_;
    std::vector<std::unique_ptr<uint8_t[]>> blocks_;
};

/// One row to fetch: its codes to dst, its scales to scales.  mx8: n bytes at off of a file, [scale, 32 codes] blocks
/// (a GGUF's MXFP8 row); else n code bytes at off and n2 scale bytes at off2 of file2 (two reads; n2 0: none - a
/// GGUF's Q6_K row, its scales inside its n bytes).
struct EgRead {
    int file = -1;
    uint64_t off = 0;
    uint32_t n = 0;
    uint8_t* dst = nullptr;
    uint8_t* scales = nullptr;
    bool mx8 = false;
    int file2 = -1;
    uint64_t off2 = 0;
    uint32_t n2 = 0;
    uint64_t key = 0;
};

/// A batch of rows (one Engram layer of a window): reads, the other places a row read once goes, and its state.
struct EgBatch {
    std::vector<EgRead> reads;
    std::vector<std::pair<int, std::pair<uint8_t*, uint8_t*>>> dups;   // read i -> another (codes, scales)
    std::vector<uint8_t> ok;                                            // 0 pending/failed, 1 read, 2 sorted duplicate
    std::atomic<int> left{0};                                           // reads not yet done
    std::atomic<int> post{0};                                           // 0 pending, 1 finishing, 2 finished
    int64_t t0 = 0;                                                     // submitted (now_ns())
    std::atomic<int64_t> t1{0};                                         // the last read done
    bool live = false;                                                  // submitted and not yet finished
    int64_t lookups = 0;                                                // logical lookups before sorted dedup
    int io_reads = 0;                                                   // canonical read attempts submitted (after dedup)
    void clear() {
        reads.clear();
        dups.clear();
        ok.clear();
        left.store(0);
        post.store(0);
        live = false;
        lookups = 0;
        io_reads = 0;
    }
};

/// The reader's threads: each read on whichever is free, in submission order, with that thread's own file handles.
class EgReader {
public:
    ~EgReader() {
        stop();
#if !defined(_WIN32)
        for (auto& r : ram_) { ::munlock(r.base, r.mapped); os::unmap(r.base, r.mapped); }
#endif
    }
    // A successful preload is fully resident and locked. In RAM mode a missing range is an error.
    bool pin_range(const std::string& path, int file, uint64_t off, uint64_t n, std::string& err) {
#if defined(_WIN32)
        err = "Engram RAM mode requires Linux"; return false;
#else
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (file < 0 || n == 0 || page_size <= 0) { err = "Engram RAM: invalid file or range"; return false; }
        const uint64_t page = (uint64_t) page_size, begin = off - off % page;
        if (n > std::numeric_limits<size_t>::max() - (off - begin) ||
            begin > (uint64_t) std::numeric_limits<off_t>::max()) {
            err = "Engram RAM: range is too large"; return false;
        }
        const size_t length = (size_t) (off - begin + n);
        const os::fd_t fd = os::open_ro(path);
        if (fd == os::kBadFd) { err = "Engram RAM: cannot open " + path; return false; }
        const uint64_t bytes = os::file_size(fd);
        if (off > bytes || n > bytes - off) {
            os::close_fd(fd); err = "Engram RAM: range exceeds the file"; return false;
        }
        void* raw = ::mmap(nullptr, length, PROT_READ, MAP_SHARED, (int) fd, (off_t) begin);
        os::close_fd(fd);
        if (raw == MAP_FAILED) { err = "Engram RAM: mmap failed"; return false; }
        uint8_t* base = (uint8_t*) raw;
        const size_t chunk = 256ull << 20;
        for (size_t pos = 0; pos < length; pos += chunk) {
            if (::mlock(base + pos, std::min(chunk, length - pos)) != 0) {
                err = "Engram RAM: mlock failed: " + std::string(std::strerror(errno)) +
                      "; check available RAM and RLIMIT_MEMLOCK (ulimit -l)";
                ::munlock(base, length); ::munmap(base, length); return false;
            }
            if ((pos / chunk) % 32 == 0 || pos + chunk >= length)
                std::fprintf(stderr, "ds4: Engram RAM file %d locked %.2f / %.2f GB\n", file,
                             (double) std::min(pos + chunk, length) / 1e9, (double) length / 1e9);
        }
        ram_.push_back({file, off, n, base, length, (size_t) (off - begin)});
        return true;
#endif
    }
    void require_ram() { ram_only_ = true; }
    /// direct[i]: file i read unbuffered (Windows, a file nothing maps: os::open_unbuffered), else buffered
    void start(const std::vector<std::string>& paths, int threads, std::vector<char> direct = {}) {
        paths_ = paths;
        direct_ = std::move(direct);
        for (int i = 0; i < threads; ++i) th_.emplace_back([this] { loop(); });
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
        th_.clear();
        if (check_n_.load() > 0)
            std::fprintf(stderr, "ds4 Engram check (DS4_EG_CHECK): %lld rows read unbuffered, %lld differed from a buffered read\n",
                         (long long) check_n_.load(), (long long) check_bad_.load());
    }
    int threads() const { return (int) th_.size(); }
    bool ram_only() const { return ram_only_; }
    void submit(EgBatch& b) {
        b.io_reads = (int) b.reads.size();
        b.ok.assign(b.reads.size(), 0);
        b.post.store(0);
        b.t0 = now_ns();
        b.t1.store(b.t0);
        b.live = true;
        b.left.store((int) b.reads.size(), std::memory_order_release);
        const int n = (int) b.reads.size();
        if (n == 0) return;
        // a token's rows one job each; a prompt chunk's in runs (fewer trips through the queue's lock)
        const int step = std::max(1, std::min(32, n / (int) (th_.size() * 4)));
        int jobs = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < n; i += step, ++jobs) q_.push_back({&b, i, std::min(n, i + step)});
        }
        if (jobs >= (int) th_.size()) cv_.notify_all();
        else
            for (int i = 0; i < jobs; ++i) cv_.notify_one();
    }
    void submit_zones(EgBatch& b, size_t zone) {
        constexpr uint64_t align = 512, max_zone = 64ull << 20;
        zone = std::min<size_t>(zone, (size_t) max_zone);
        if (ram_only_ || zone == 0 || b.reads.size() < 4096) {
            submit(b);
            return;
        }
        for (const EgRead& r : b.reads) {
            if (r.mx8 || r.n2 != 0 || r.off > UINT64_MAX - r.n || r.off + r.n > UINT64_MAX - (align - 1)) {
                submit(b);
                return;
            }
        }
        b.ok.assign(b.reads.size(), 0);
        b.post.store(0);
        b.t0 = now_ns();
        b.t1.store(b.t0);
        b.live = true;
        b.left.store((int) b.reads.size(), std::memory_order_release);
        const int64_t plan0 = now_ns();
        const size_t n = b.reads.size();
        std::vector<int> ids(n);
        for (size_t i = 0; i < n; ++i) ids[i] = (int) i;
        static const bool force_std = [] {
            const char* s = std::getenv("DS4_EG_SORT_MODE");
            return s && std::strcmp(s, "std") == 0;
        }();
        static const bool force64 = [] {
            const char* s = std::getenv("DS4_EG_SORT_MODE");
            return s && std::strcmp(s, "radix64") == 0;
        }();
        static const bool profile = std::getenv("DS4_EG_PROFILE") != nullptr;
        static const bool dedup = [] {
            const char* s = std::getenv("DS4_EG_DEDUP");
            return !s || std::strcmp(s, "0") != 0;
        }();
        const char* sorted_by = "std";
        bool one_file = true;
        for (const EgRead& r : b.reads) one_file = one_file && r.file == b.reads.front().file;
        auto radix = [](auto& a, auto& tmp, int passes, auto key) {
            std::vector<uint32_t> count(65536);
            for (int pass = 0; pass < passes; ++pass) {
                std::fill(count.begin(), count.end(), 0);
                const int shift = pass * 16;
                for (const auto& x : a) ++count[(key(x) >> shift) & 65535];
                uint32_t sum = 0;
                for (uint32_t& c : count) {
                    const uint32_t v = c;
                    c = sum;
                    sum += v;
                }
                for (const auto& x : a) tmp[count[(key(x) >> shift) & 65535]++] = x;
                a.swap(tmp);
            }
        };
        if (one_file && !force_std) {
            const EgRead& first = b.reads.front();
            const uint64_t id0 = first.key >> 3, row_bytes = first.n;
            bool id32 = !force64 && row_bytes != 0 && id0 <= UINT32_MAX && first.off >= id0 * row_bytes;
            const uint64_t origin = id32 ? first.off - id0 * row_bytes : 0;
            if (id32) {
                for (const EgRead& r : b.reads) {
                    const uint64_t id = r.key >> 3;
                    if (id > UINT32_MAX || r.n != row_bytes || id > (UINT64_MAX - origin) / row_bytes ||
                        r.off != origin + id * row_bytes) {
                        id32 = false;
                        break;
                    }
                }
            }
            if (id32) {
                struct Item { uint32_t key, id; };
                std::vector<Item> a(n), tmp(n);
                for (size_t i = 0; i < n; ++i) a[i] = {(uint32_t) (b.reads[i].key >> 3), (uint32_t) i};
                radix(a, tmp, 2, [](const Item& x) { return x.key; });
                for (size_t i = 0; i < n; ++i) ids[i] = (int) a[i].id;
                sorted_by = "radix32";
            } else {
                struct Item { uint64_t key; uint32_t id, pad; };
                std::vector<Item> a(n), tmp(n);
                for (size_t i = 0; i < n; ++i) a[i] = {b.reads[i].off, (uint32_t) i, 0};
                int passes = 3;   // the model files are below 2^48 bytes; use the fourth pass if that ever changes
                for (const EgRead& r : b.reads)
                    if ((r.off >> 48) != 0) {
                        passes = 4;
                        break;
                    }
                radix(a, tmp, passes, [](const Item& x) { return x.key; });
                for (size_t i = 0; i < n; ++i) ids[i] = (int) a[i].id;
                sorted_by = "radix64";
            }
        } else {
            std::sort(ids.begin(), ids.end(), [&](int a, int c) {
                const EgRead& x = b.reads[(size_t) a];
                const EgRead& y = b.reads[(size_t) c];
                return x.file != y.file ? x.file < y.file : x.off < y.off;
            });
        }
        const int64_t plan1 = now_ns();
        if (dedup) {
            // Equal rows are adjacent after the offset/row-id sort. Keep the first original read as the canonical
            // destination, turn later destinations into the ordinary post-read copies, and submit only canonical rows.
            // State 2 keeps duplicates out of the cache-insertion loop; their destinations are filled from the canonical.
            std::vector<int> unique;
            unique.reserve(ids.size());
            for (int ri : ids) {
                const EgRead& r = b.reads[(size_t) ri];
                if (!unique.empty()) {
                    const int first = unique.back();
                    const EgRead& canonical = b.reads[(size_t) first];
                    if (canonical.key == r.key && canonical.file == r.file && canonical.off == r.off &&
                        canonical.n == r.n) {
                        b.dups.push_back({first, {r.dst, r.scales}});
                        b.ok[(size_t) ri] = 2;
                        continue;
                    }
                }
                unique.push_back(ri);
            }
            ids.swap(unique);
        }
        b.io_reads = (int) ids.size();
        b.left.store((int) ids.size(), std::memory_order_release);
        const int64_t plan2 = now_ns();
        int jobs = 0;
        uint64_t planned_bytes = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (size_t i = 0; i < ids.size();) {
                const EgRead& first = b.reads[(size_t) ids[i]];
                const uint64_t base = first.off - first.off % align;
                uint64_t end = (first.off + first.n + align - 1) / align * align;
                uint64_t payload_end = first.off + first.n;
                size_t j = i + 1;
                while (j < ids.size()) {
                    const EgRead& next = b.reads[(size_t) ids[j]];
                    if (next.file != first.file) break;
                    const uint64_t next_end = (next.off + next.n + align - 1) / align * align;
                    if (std::max(end, next_end) - base > zone) break;
                    end = std::max(end, next_end);
                    payload_end = std::max(payload_end, next.off + next.n);
                    ++j;
                }
                Job job;
                job.b = &b;
                job.zone = true;
                job.off = base;
                job.n = (uint32_t) (end - base);
                job.need = (uint32_t) (payload_end - base);
                job.rows.assign(ids.begin() + i, ids.begin() + j);
                planned_bytes += job.n;
                q_.push_back(std::move(job));
                ++jobs;
                i = j;
            }
        }
        if (profile)
            std::fprintf(stderr,
                         "ds4: Engram plan: %zu lookups, %zu rows, %s %.1f ms, dedup %.1f ms, zones %.1f ms: "
                         "%d reads, %.3f GB\n",
                         n, ids.size(), sorted_by, (plan1 - plan0) / 1e6, (plan2 - plan1) / 1e6,
                         (now_ns() - plan2) / 1e6, jobs, planned_bytes / 1e9);
        if (jobs >= (int) th_.size()) cv_.notify_all();
        else
            for (int i = 0; i < jobs; ++i) cv_.notify_one();
    }
    /// Until every read of b is done.
    void wait(EgBatch& b) {
        if (b.left.load(std::memory_order_acquire) == 0) return;
        std::unique_lock<std::mutex> lk(done_mu_);
        done_cv_.wait(lk, [&] { return b.left.load(std::memory_order_acquire) == 0; });
    }

private:
    struct RamRange { int file; uint64_t off, n; uint8_t* base; size_t mapped, delta; };
    std::vector<RamRange> ram_;
    bool ram_only_ = false;
    bool copy_ram(int file, uint64_t off, uint64_t n, void* dst) const {
        for (const auto& r : ram_) if (r.file == file && off >= r.off && n <= r.n && off - r.off <= r.n - n) {
            std::memcpy(dst, r.base + r.delta + (size_t) (off - r.off), (size_t) n); return true;
        }
        return false;
    }
    void loop() {
        std::vector<os::fd_t> fds(paths_.size(), os::kBadFd);   // this thread's own handles, opened on first use
        std::vector<os::fd_t> ufds(paths_.size(), os::kBadFd);  // (and its unbuffered ones)
        std::vector<uint8_t> buf, abuf(3 * os::kDirectAlign);
        uint8_t* al = abuf.data() + (os::kDirectAlign - (uintptr_t) abuf.data() % os::kDirectAlign) % os::kDirectAlign;
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || !q_.empty(); });
                if (quit_ && q_.empty()) break;
                job = q_.front();
                q_.pop_front();
            }
            EgBatch& b = *job.b;
            int done = 0;
            if (job.zone) {
                read_zone(fds, ufds, buf, b, job);
                done = (int) job.rows.size();
            } else {
                for (int i = job.i0; i < job.i1; ++i) b.ok[(size_t) i] = read(fds, ufds, al, b.reads[(size_t) i], buf) ? 1 : 0;
                done = job.i1 - job.i0;
            }
            const int64_t t = now_ns();
            for (int64_t prev = b.t1.load(); t > prev && !b.t1.compare_exchange_weak(prev, t);) {
            }
            if (b.left.fetch_sub(done, std::memory_order_acq_rel) == done) {   // (b is the waiter's again)
                std::lock_guard<std::mutex> lk(done_mu_);
                done_cv_.notify_all();
            }
        }
        for (os::fd_t f : fds) os::close_fd(f);
        for (os::fd_t f : ufds) os::close_fd(f);
    }
    bool read(std::vector<os::fd_t>& fds, std::vector<os::fd_t>& ufds, uint8_t* al, const EgRead& r,
              std::vector<uint8_t>& buf) {
        if (ram_only_) {
            if (!r.mx8) {
                return copy_ram(r.file, r.off, r.n, r.dst) &&
                       (r.n2 == 0 || copy_ram(r.file2, r.off2, r.n2, r.scales));
            }
            buf.resize(r.n);
            if (!copy_ram(r.file, r.off, r.n, buf.data())) return false;
            for (uint32_t k = 0; k < r.n / 33; ++k) {
                r.scales[k] = buf[(size_t) k * 33];
                std::memcpy(r.dst + (size_t) k * 32, buf.data() + (size_t) k * 33 + 1, 32);
            }
            return true;
        }
        if (r.mx8 && r.file >= 0 && (size_t) r.file < direct_.size() && direct_[(size_t) r.file] &&
            os::direct_span(r.off, r.n) <= 2 * os::kDirectAlign) {
            os::fd_t& u = ufds[(size_t) r.file];
            if (u == os::kBadFd) u = os::open_unbuffered(paths_[(size_t) r.file]);
            if (u != os::kBadFd && os::pread_unbuffered(u, al, r.n, r.off)) {
                const uint8_t* src = al + r.off % os::kDirectAlign;
                static const bool check = std::getenv("DS4_EG_CHECK") != nullptr;   // (each row read buffered too)
                if (check) {
                    std::vector<uint8_t> ref(r.n);
                    os::fd_t& f2 = fds[(size_t) r.file];
                    if (f2 == os::kBadFd) f2 = os::open_ro(paths_[(size_t) r.file]);
                    const bool ok = f2 != os::kBadFd && os::pread_full(f2, ref.data(), r.n, r.off);
                    if (!ok || std::memcmp(ref.data(), src, r.n) != 0) ++check_bad_;
                    ++check_n_;
                }
                for (uint32_t k = 0; k < r.n / 33; ++k) {   // [scale, 32 codes] -> codes | scales
                    r.scales[k] = src[(size_t) k * 33];
                    std::memcpy(r.dst + (size_t) k * 32, src + (size_t) k * 33 + 1, 32);
                }
                return true;
            }   // (else buffered, below)
        }
        auto fd = [&](int file) {
            if (file < 0 || file >= (int) fds.size()) return os::kBadFd;
            if (fds[(size_t) file] == os::kBadFd) {
                fds[(size_t) file] = os::open_ro(paths_[(size_t) file]);
#if !defined(_WIN32)
                // MAYA_DS4_EG_RANDOM=1 (opt-in; read as STRATA_DS4_EG_RANDOM, DS4_EG_RANDOM too): no readahead on this
                // reader's descriptors.  A row is ~1-2 KB at a hashed offset, so a buffered read's readahead window is
                // mostly waste that can push the CPU lane's experts out of the page cache (a 2x MI50 box with 62 GiB,
                // #505); on a V100 with 61 GB it made 2K-8K prompts 3-4% slower, so it stays off unless asked for
                static const bool eg_random = [] {
                    const char* v = std::getenv("STRATA_DS4_EG_RANDOM");
                    if (v == nullptr) v = std::getenv("DS4_EG_RANDOM");
                    return v != nullptr && std::atoi(v) != 0;
                }();
                if (eg_random && fds[(size_t) file] != os::kBadFd) (void) posix_fadvise(fds[(size_t) file], 0, 0, POSIX_FADV_RANDOM);
#endif
            }
            return fds[(size_t) file];
        };
        const os::fd_t f = fd(r.file);
        if (f == os::kBadFd) return false;
        if (!r.mx8) {
            if (!os::pread_full(f, r.dst, r.n, r.off)) return false;
            if (r.n2 == 0) return true;
            const os::fd_t f2 = fd(r.file2);
            return f2 != os::kBadFd && os::pread_full(f2, r.scales, r.n2, r.off2);
        }
        buf.resize(r.n);
        if (!os::pread_full(f, buf.data(), r.n, r.off)) return false;
        for (uint32_t k = 0; k < r.n / 33; ++k) {   // [scale, 32 codes] -> codes | scales
            r.scales[k] = buf[(size_t) k * 33];
            std::memcpy(r.dst + (size_t) k * 32, buf.data() + (size_t) k * 33 + 1, 32);
        }
        return true;
    }

    std::vector<std::string> paths_;
    std::vector<char> direct_;

public:
    std::atomic<int64_t> check_n_{0}, check_bad_{0};   // (DS4_EG_CHECK: rows compared, rows that differed)
    std::vector<std::thread> th_;
    std::mutex mu_;
    std::condition_variable cv_;
    struct Job {
        EgBatch* b = nullptr;
        int i0 = 0, i1 = 0;   // reads [i0, i1)
        bool zone = false;
        uint64_t off = 0;
        uint32_t n = 0;       // aligned read span
        uint32_t need = 0;    // actual payload extent (may be shorter at the file's tail)
        std::vector<int> rows;
    };
    void read_zone(std::vector<os::fd_t>& fds, std::vector<os::fd_t>& ufds, std::vector<uint8_t>& buf,
                   EgBatch& b, const Job& job) {
        const int file = b.reads[(size_t) job.rows[0]].file;
        const bool direct = file >= 0 && (size_t) file < direct_.size() && direct_[(size_t) file];
        bool ok = false;
        if (direct) {
            os::fd_t& u = ufds[(size_t) file];
            if (u == os::kBadFd) u = os::open_unbuffered(paths_[(size_t) file]);
            if (u != os::kBadFd) {
                buf.resize((size_t) job.n + os::kDirectAlign);
                uint8_t* dst = buf.data() + (os::kDirectAlign - (uintptr_t) buf.data() % os::kDirectAlign) % os::kDirectAlign;
                ok = os::pread_unbuffered_aligned(u, dst, job.need, job.off, 512);
                if (ok) {
                    for (int i : job.rows) {
                        const EgRead& r = b.reads[(size_t) i];
                        std::memcpy(r.dst, dst + (r.off - job.off), r.n);
                        b.ok[(size_t) i] = 1;
                    }
                }
            }
        }
        if (!ok) {
            os::fd_t& f = fds[(size_t) file];
            if (f == os::kBadFd) f = os::open_ro(paths_[(size_t) file]);
            if (f != os::kBadFd) {
                buf.resize(job.need);
                ok = os::pread_full(f, buf.data(), job.need, job.off);
                if (ok) {
                    for (int i : job.rows) {
                        const EgRead& r = b.reads[(size_t) i];
                        std::memcpy(r.dst, buf.data() + (r.off - job.off), r.n);
                        b.ok[(size_t) i] = 1;
                    }
                }
            }
        }
    }
    std::deque<Job> q_;
    bool quit_ = false;
    std::mutex done_mu_;
    std::condition_variable done_cv_;
};

}  // namespace strata::ds4
