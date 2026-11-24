// src/hy4/hy4_spill.hpp - maya-hy4's KV spill file (Options::kv_spill): the indexed layers' cache rows past what the
// RAM holds, on a drive.  One file a start, deleted when it closes (or as soon as it is open, on Linux); reads and
// writes at offsets from any thread (pread / pwrite, ReadFile / WriteFile with an offset).  Rows go through the
// system's file cache: what the RAM has free keeps the rows read most.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hy4 {

class SpillFile {
public:
    SpillFile() = default;
    SpillFile(const SpillFile&) = delete;
    SpillFile& operator=(const SpillFile&) = delete;
    ~SpillFile() { close(); }

    /// a new file in dir (deleted at close); err says why not
    bool open(const std::string& dir, std::string& err) {
        char name[96];
        std::snprintf(name, sizeof name, "maya-hy4-kv-%lld.tmp",
                      (long long) std::chrono::steady_clock::now().time_since_epoch().count());
        path_ = dir.empty() ? std::string(name) : dir + "/" + name;
#ifdef _WIN32
        h_ = CreateFileA(path_.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) {
            h_ = nullptr;
            err = "KV spill: no file at " + path_ + " (error " + std::to_string((long long) GetLastError()) + ")";
            return false;
        }
#else
        fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd_ < 0) {
            err = "KV spill: no file at " + path_;
            return false;
        }
        ::unlink(path_.c_str());   // (gone with the process, however it ends)
#endif
        return true;
    }
    bool is_open() const {
#ifdef _WIN32
        return h_ != nullptr;
#else
        return fd_ >= 0;
#endif
    }
    const std::string& path() const { return path_; }
    void close() {
#ifdef _WIN32
        if (h_) CloseHandle(h_);
        h_ = nullptr;
#else
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
    }
    /// n bytes at offset off: true when all of them moved
    bool read(void* dst, size_t n, uint64_t off) const {
        uint8_t* d = (uint8_t*) dst;
        while (n > 0) {
#ifdef _WIN32
            OVERLAPPED ov{};
            ov.Offset = (DWORD) (off & 0xffffffffu);
            ov.OffsetHigh = (DWORD) (off >> 32);
            DWORD got = 0;
            const DWORD want = (DWORD) std::min<size_t>(n, 1u << 30);
            if (!ReadFile(h_, d, want, &got, &ov) || got == 0) return false;
#else
            const ssize_t got = ::pread(fd_, d, n, (off_t) off);
            if (got <= 0) return false;
#endif
            d += got;
            off += (uint64_t) got;
            n -= (size_t) got;
        }
        return true;
    }
    bool write(const void* src, size_t n, uint64_t off) const {
        const uint8_t* s = (const uint8_t*) src;
        while (n > 0) {
#ifdef _WIN32
            OVERLAPPED ov{};
            ov.Offset = (DWORD) (off & 0xffffffffu);
            ov.OffsetHigh = (DWORD) (off >> 32);
            DWORD put = 0;
            const DWORD want = (DWORD) std::min<size_t>(n, 1u << 30);
            if (!WriteFile(h_, s, want, &put, &ov) || put == 0) return false;
#else
            const ssize_t put = ::pwrite(fd_, s, n, (off_t) off);
            if (put <= 0) return false;
#endif
            s += put;
            off += (uint64_t) put;
            n -= (size_t) put;
        }
        return true;
    }
    /// the drive's reads in pieces of `piece` bytes at random offsets of the first `span` bytes (written first, then
    /// out of the system's file cache where it can say so - Linux): MB/s, 0 when it could not tell
    double measure(size_t span, size_t piece, int n) {
        std::vector<uint8_t> buf(std::max(piece, (size_t) 1 << 20), 0x5a);
        for (size_t o = 0; o < span; o += buf.size())
            if (!write(buf.data(), std::min(buf.size(), span - o), o)) return 0.0;
#ifndef _WIN32
        ::fdatasync(fd_);
#ifdef POSIX_FADV_DONTNEED
        ::posix_fadvise(fd_, 0, (off_t) span, POSIX_FADV_DONTNEED);
#endif
#else
        FlushFileBuffers(h_);
#endif
        std::mt19937_64 rng(12345);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) {
            const uint64_t off = (rng() % (span / piece)) * piece;
            if (!read(buf.data(), piece, off)) return 0.0;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return s > 0 ? (double) piece * n / s / 1e6 : 0.0;
    }

private:
    std::string path_;
#ifdef _WIN32
    HANDLE h_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace hy4
