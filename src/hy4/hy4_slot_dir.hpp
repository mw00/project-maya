// src/hy4/hy4_slot_dir.hpp - maya-hy4's conversation slots on a drive (SLOTS v2, MAYA_HY4_SLOT_KEEP=1 with
// MAYA_HY4_SLOT_DIR): the files' I/O and the background writer.  No GPU, no model; what is decided about the files
// (which to keep, their .meta's bytes, the checksum) is hy4_slots.hpp's, pure.
//
// A slot on the drive is two files beside each other: <stem>.bin (Model::conv_save's bytes, as they are) and
// <stem>.meta (meta_encode: the identity, the .bin's size and checksum, the tokens, the pictures).  Each is written as
// <name>.tmp, flushed to the drive (fsync), its pages dropped from the page cache (posix_fadvise DONTNEED: the RAM is
// the CPU lane's experts'), then renamed - the .bin first, the .meta last, then the folder itself flushed (a flush
// that fails fails the write: the pair removed, no "wrote").  So a .meta names a complete .bin; a .tmp, or a .bin
// without its .meta, is what an interrupted write left (removed at the start).
//
//   write_atomic(path, p, n, go, err)   p[0, n) to path through path.tmp, 64 MB at a time; go() false between two
//                                       pieces (an abort, a deadline) stops it and removes the .tmp
//   read_file(path, out, expect, err)   the file into out (resized to it; its size must be expect), the page cache
//                                       told it is read once
//   remove_files(stem)                  <stem>.bin, .meta and their .tmp - whatever of them is there
//   lock_dir(dir, wait_s)               the folder this engine's alone: an flock on <dir>/.slots.lock, waited for up to
//                                       wait_s (another engine on it - a swap's last one still writing at its end);
//                                       -1: not got (the caller keeps its conversations in RAM only).  It ends with
//                                       the process
//   Writer                              one thread (started by start(on_thread): on_thread runs on it first - the
//                                       serve loop lowers its priority) writing the jobs in turn: submit(job, front)
//                                       queues one (front: next); abort(id) drops a queued job or stops the one in
//                                       flight, abort_idle() every job marked idle (the serve loop's idle-time
//                                       writes, given up when a request comes), set_deadline(t) stops whatever runs
//                                       past t (steady seconds; 0 none); poll() hands over the jobs done (ok, aborted
//                                       or failed, with the .bin's checksum and the time); wait_idle(t) waits until
//                                       nothing is queued or in flight, or t.  A job never leaves a file of its own
//                                       behind unless it ended ok.  The destructor stops every job and joins.
#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "hy4_slots.hpp"

namespace hy4::slots {

inline double steady_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void remove_files(const std::string& stem) {
    if (stem.empty()) return;
    std::error_code ec;
    for (const char* x : {".bin", ".meta", ".bin.tmp", ".meta.tmp"}) std::filesystem::remove(stem + x, ec);
}

#if !defined(_WIN32)
inline bool fsync_dir(const std::string& dir) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}
#endif

inline int lock_dir(const std::string& dir, double wait_s) {
#if !defined(_WIN32)
    const int fd = ::open((dir + "/.slots.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return -1;
    const double end = steady_s() + wait_s;
    while (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if ((errno != EWOULDBLOCK && errno != EINTR) || steady_s() > end) {
            ::close(fd);
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return fd;
#else
    (void) dir;
    (void) wait_s;
    return 0;
#endif
}

inline bool write_atomic(const std::string& path, const uint8_t* p, size_t n, const std::function<bool()>& go,
                         std::string& err) {
    const std::string tmp = path + ".tmp";
    constexpr size_t kPiece = (size_t) 64 << 20;
#if !defined(_WIN32)
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        err = "cannot create " + tmp + ": " + std::strerror(errno);
        return false;
    }
    bool ok = true;
    size_t off = 0;
    while (ok && off < n) {
        if (go && !go()) {
            err = "stopped";
            ok = false;
            break;
        }
        const size_t want = std::min(kPiece, n - off);
        size_t done = 0;
        while (done < want) {
            const ssize_t w = ::write(fd, p + off + done, want - done);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) {
                err = "writing " + tmp + ": " + std::strerror(errno);
                ok = false;
                break;
            }
            done += (size_t) w;
        }
#if defined(__linux__) && defined(SYNC_FILE_RANGE_WRITE)
        // (the piece's write-back started now, the one two before it waited for and its pages dropped: three pieces
        //  dirty at most - the final fsync is short, so a deadline holds, and the page cache keeps the experts)
        if (ok) {
            (void) ::sync_file_range(fd, (off64_t) off, (off64_t) done, SYNC_FILE_RANGE_WRITE);
            if (off >= 2 * kPiece) {
                const off64_t prev = (off64_t) (off - 2 * kPiece);
                (void) ::sync_file_range(fd, prev, (off64_t) kPiece,
                                         SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
                (void) ::posix_fadvise(fd, prev, (off_t) kPiece, POSIX_FADV_DONTNEED);
            }
        }
#endif
        off += done;
    }
    if (ok && ::fsync(fd) != 0) {
        err = "flushing " + tmp + ": " + std::strerror(errno);
        ok = false;
    }
#if defined(POSIX_FADV_DONTNEED)
    (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // (its pages written: the RAM back to the page cache's
#endif                                                       //  other files - the CPU lane's experts)
    if (::close(fd) != 0 && ok) {
        err = "closing " + tmp + ": " + std::strerror(errno);
        ok = false;
    }
    if (ok && ::rename(tmp.c_str(), path.c_str()) != 0) {
        err = "renaming " + tmp + ": " + std::strerror(errno);
        ok = false;
    }
    if (!ok) ::unlink(tmp.c_str());
    return ok;
#else
    bool ok = true;
    {
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (f == nullptr) {
            err = "cannot create " + tmp;
            return false;
        }
        for (size_t off = 0; ok && off < n; off += kPiece) {
            if (go && !go()) {
                err = "stopped";
                ok = false;
                break;
            }
            const size_t want = std::min(kPiece, n - off);
            ok = std::fwrite(p + off, 1, want, f) == want;
            if (!ok) err = "writing " + tmp;
        }
        if (std::fclose(f) != 0 && ok) {
            err = "closing " + tmp;
            ok = false;
        }
    }
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);
    if (ok && ec) {
        err = "renaming " + tmp + ": " + ec.message();
        ok = false;
    }
    if (!ok) std::filesystem::remove(tmp, ec);
    return ok;
#endif
}

inline bool read_file(const std::string& path, std::vector<uint8_t>& out, uint64_t expect, std::string& err) {
    std::error_code ec;
    const uint64_t size = (uint64_t) std::filesystem::file_size(path, ec);
    if (ec) {
        err = "cannot read " + path + ": " + ec.message();
        return false;
    }
    if (size != expect) {
        err = path + " holds " + std::to_string(size) + " bytes, not the " + std::to_string(expect) + " its .meta says";
        return false;
    }
    try {
        out.resize((size_t) size);
    } catch (const std::bad_alloc&) {
        err = "no " + std::to_string(size >> 20) + " MB of RAM to read " + path + " into";
        return false;
    }
#if !defined(_WIN32)
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        err = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }
#if defined(POSIX_FADV_SEQUENTIAL)
    (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    size_t off = 0;
    bool ok = true;
    while (off < size) {
        const ssize_t r = ::read(fd, out.data() + off, (size_t) std::min<uint64_t>(size - off, (uint64_t) 64 << 20));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) {
            err = "reading " + path + ": " + (r == 0 ? std::string("it ended early") : std::string(std::strerror(errno)));
            ok = false;
            break;
        }
        off += (size_t) r;
    }
#if defined(POSIX_FADV_DONTNEED)
    (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // (read once: its pages back to the page cache)
#endif
    ::close(fd);
    return ok;
#else
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    const bool ok = std::fread(out.data(), 1, (size_t) size, f) == (size_t) size;
    std::fclose(f);
    if (!ok) err = "reading " + path + " failed";
    return ok;
#endif
}

class Writer {
public:
    struct Job {
        uint64_t id = 0;
        std::string stem;
        std::shared_ptr<const std::vector<uint8_t>> data;
        Meta meta;   // (bin_bytes and checksum: the writer's)
        bool idle = false;
    };
    struct Done {
        uint64_t id = 0;
        std::string stem;
        bool ok = false, aborted = false, idle = false;
        std::string err;
        uint64_t bytes = 0, sum = 0;
        int64_t n = 0;
        double ms = 0;
    };
    Writer() = default;
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    ~Writer() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
            for (Job& j : q_) remove_files(j.stem);
            q_.clear();
            cur_abort_.store(true);
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    void start(std::function<void()> on_thread) {
        th_ = std::thread([this, on_thread] {
            if (on_thread) on_thread();
            run();
        });
    }
    void submit(Job j, bool front = false) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (front) q_.insert(q_.begin(), std::move(j));
            else q_.push_back(std::move(j));
        }
        cv_.notify_all();
    }
    void abort(uint64_t id) {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < q_.size(); ++i)
            if (q_[i].id == id) {
                finish_unstarted(q_[i]);
                q_.erase(q_.begin() + (long) i);
                return;
            }
        if (cur_id_ == id) cur_abort_.store(true);
    }
    void abort_idle() {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = q_.size(); i-- > 0;)
            if (q_[i].idle) {
                finish_unstarted(q_[i]);
                q_.erase(q_.begin() + (long) i);
            }
        if (cur_id_ != 0 && cur_idle_) cur_abort_.store(true);
    }
    void abort_all() {
        std::lock_guard<std::mutex> lk(mu_);
        for (Job& j : q_) finish_unstarted(j);
        q_.clear();
        if (cur_id_ != 0) cur_abort_.store(true);
    }
    void set_deadline(double t) { deadline_.store(t); }
    std::vector<Done> poll() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<Done> d;
        d.swap(done_);
        return d;
    }
    bool busy() {
        std::lock_guard<std::mutex> lk(mu_);
        return !q_.empty() || cur_id_ != 0;
    }
    bool wait_idle(double t) {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            if (q_.empty() && cur_id_ == 0) return true;
            const double left = t - steady_s();
            if (left <= 0) return false;
            cv_.wait_for(lk, std::chrono::duration<double>(std::min(left, 0.1)));
        }
    }

private:
    void finish_unstarted(const Job& j) {   // (under mu_)
        Done d;
        d.id = j.id;
        d.stem = j.stem;
        d.aborted = true;
        d.idle = j.idle;
        d.n = j.meta.n;
        d.err = "stopped before it began";
        done_.push_back(std::move(d));
    }
    void run() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || !q_.empty(); });
                if (quit_) return;
                j = std::move(q_.front());
                q_.erase(q_.begin());
                cur_id_ = j.id;
                cur_idle_ = j.idle;
                cur_abort_.store(false);
            }
            Done d = write(j);
            {
                std::lock_guard<std::mutex> lk(mu_);
                cur_id_ = 0;
                cur_idle_ = false;
                done_.push_back(std::move(d));
            }
            cv_.notify_all();
        }
    }
    Done write(Job& j) {
        Done d;
        d.id = j.id;
        d.stem = j.stem;
        d.idle = j.idle;
        d.n = j.meta.n;
        const double t0 = steady_s();
        const auto go = [&] {
            const double dl = deadline_.load();
            return !cur_abort_.load() && !(dl > 0 && steady_s() > dl);
        };
        const std::vector<uint8_t>& b = *j.data;
        {
            std::error_code ec;
            const std::filesystem::path dir = std::filesystem::path(j.stem).parent_path();
            const auto sp = std::filesystem::space(dir, ec);
            if (!ec && sp.available < (uint64_t) b.size() + ((uint64_t) 1 << 30)) {
                char m[200];
                std::snprintf(m, sizeof m, "%.2f GB free on the folder's drive, under its %.2f GB and 1 GB left",
                              (double) sp.available / 1073741824.0, (double) b.size() / 1073741824.0);
                d.err = m;
                return d;
            }
        }
        j.meta.bin_bytes = b.size();
        j.meta.checksum = checksum(b.data(), b.size());
        d.sum = j.meta.checksum;
        d.bytes = b.size();
        std::string err;
        bool ok = go() && write_atomic(j.stem + ".bin", b.data(), b.size(), go, err);
        if (ok) {
            const std::vector<uint8_t> mb = meta_encode(j.meta);
            ok = go() && write_atomic(j.stem + ".meta", mb.data(), mb.size(), go, err);
        }
#if !defined(_WIN32)
        // (the renames on the drive too: a folder that cannot be flushed may lose them at a crash - a failed write,
        //  its pair removed, no "wrote")
        if (ok && !fsync_dir(std::filesystem::path(j.stem).parent_path().string())) {
            err = "flushing the folder failed: " + std::string(std::strerror(errno));
            ok = false;
        }
#endif
        if (!ok) {
            d.aborted = !go() && err.rfind("flushing the folder", 0) != 0;
            d.err = d.aborted ? (cur_abort_.load() ? "stopped" : "past the engine's end's time") : err;
            remove_files(j.stem);
        }
        d.ok = ok;
        d.ms = (steady_s() - t0) * 1e3;
        return d;
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Job> q_;
    std::vector<Done> done_;
    std::thread th_;
    bool quit_ = false;
    uint64_t cur_id_ = 0;
    bool cur_idle_ = false;
    std::atomic<bool> cur_abort_{false};
    std::atomic<double> deadline_{0.0};
};

}  // namespace hy4::slots
