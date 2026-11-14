#include "peer_cache.hpp"

#include "peer_budget.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace strata::peer {

void (*PeerCache::alloc_log_)(uint64_t bytes) = nullptr;

bool PeerCache::open(Device* dev, uint64_t budget, uint64_t arena_bytes, std::string& err) {
    dev_ = dev;
    budget_ = budget ? budget : budget_probe(32ULL << 30);
    arena_bytes_ = arena_bytes ? arena_bytes : 2ULL << 30;
    if (budget_ == 0) {
        err = "no RAM budget: the cgroup-aware probe found nothing free";
        return false;
    }
    if (arena_bytes_ > budget_) arena_bytes_ = budget_;
    say("slab budget " + std::to_string(budget_ >> 20) + " MiB in " +
        std::to_string(arena_bytes_ >> 20) + " MiB arenas");
    return true;
}

void PeerCache::close() {
    close_shards();
    arenas_.clear();
    where_.clear();
    dropped_.clear();
    cached_bytes_ = 0;
    next_arena_ = 0;
}

bool PeerCache::open_shards(const std::vector<std::string>& names, const std::vector<uint64_t>& sizes,
                            const std::string& pack_dir, std::string& err) {
    close_shards();
    shard_fd_.reserve(names.size());
    shard_size_.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        // the manifest's names are basenames; the daemon resolves them inside its own pack dir (the
        // two machines' mount points differ; the fingerprint is what actually binds the copies)
        std::string clean = names[i];
        const size_t slash = clean.find_last_of("/\\");
        if (slash != std::string::npos) clean = clean.substr(slash + 1);
        const std::string path = pack_dir + (pack_dir.back() == '/' ? "" : "/") + clean;
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "shard " + clean + " not found under " + pack_dir + " (" + std::strerror(errno) + ")";
            close_shards();
            return false;
        }
        const off_t real = lseek(fd, 0, SEEK_END);
        if (real < 0 || (uint64_t) real != sizes[i]) {
            err = "shard " + clean + " is " + std::to_string((uint64_t) std::max<off_t>(real, 0)) +
                  " bytes, the manifest said " + std::to_string(sizes[i]);
            close_shards();
            return false;
        }
        shard_fd_.push_back(fd);
        shard_size_.push_back(sizes[i]);
    }
    return true;
}

void PeerCache::close_shards() {
    for (int fd : shard_fd_) ::close(fd);
    shard_fd_.clear();
    shard_size_.clear();
}

std::string pack_fingerprint(const std::string& dir, const std::vector<std::string>& names,
                             const std::vector<uint64_t>& sizes, std::string& err) {
    if (names.size() != sizes.size() || names.empty()) {
        err = "a pack of " + std::to_string(names.size()) + " shards and " +
              std::to_string(sizes.size()) + " sizes is not a pack";
        return "";
    }
    char piece[80];
    const size_t edge = 1ULL << 20;                  // first and last MiB of each shard
    std::vector<char> buf(edge);
    uint64_t total = 0;
    std::string out;
    for (size_t i = 0; i < names.size(); ++i) {
        std::string clean = names[i];
        const size_t slash = clean.find_last_of("/\\");
        if (slash != std::string::npos) clean = clean.substr(slash + 1);
        const std::string path = dir + (dir.back() == '/' ? "" : "/") + clean;
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "shard " + clean + " not found under " + dir + " (" + strerror(errno) + ")";
            return "";
        }
        const uint64_t size = sizes[i];
        total += size;
        uint64_t h = 1469598103934665603ULL;         // the 64-bit FNV-1a offset basis
        const size_t head = (size_t) std::min<uint64_t>(edge, size);
        if (pread(fd, buf.data(), head, 0) != (ssize_t) head) {
            err = "shard " + clean + ": head pread failed";
            close(fd);
            return "";
        }
        for (size_t b = 0; b < head; ++b) h = (h ^ (unsigned char) buf[b]) * 1099511628211ULL;
        if (size > edge) {
            if (pread(fd, buf.data(), edge, (off_t) (size - edge)) != (ssize_t) edge) {
                err = "shard " + clean + ": tail pread failed";
                close(fd);
                return "";
            }
            for (size_t b = 0; b < edge; ++b) h = (h ^ (unsigned char) buf[b]) * 1099511628211ULL;
        }
        close(fd);
        std::snprintf(piece, sizeof piece, "%s%llu:%016llx", out.empty() ? "" : ";",
                      (unsigned long long) size, (unsigned long long) h);
        out += piece;
    }
    std::snprintf(piece, sizeof piece, "|%llu", (unsigned long long) total);
    return out + piece;
}

void PeerCache::drop_arena(uint32_t id) {
    const auto arena = arenas_.find(id);
    if (arena == arenas_.end()) return;
    for (auto it = where_.begin(); it != where_.end();) {
        if (it->second.arena == id) {
            dropped_.push_back(it->first);
            cached_bytes_ -= it->second.len;
            it = where_.erase(it);
        } else {
            ++it;
        }
    }
    cached_bytes_ -= arena->second.used;
    arenas_.erase(arena);
}

PeerCache::ArenaRec* PeerCache::room_for(size_t bytes, uint64_t stamp, uint32_t* id_out, std::string& err) {
    for (auto& kv : arenas_) {
        if (kv.second.used + bytes <= kv.second.mr->bytes()) {
            kv.second.stamp = stamp;
            *id_out = kv.first;
            return &kv.second;
        }
    }
    // nothing fits: a fresh arena under the budget, else LRU arenas out whole until it does
    const size_t cap = std::max<size_t>(bytes, arena_bytes_);
    while (cached_bytes_ + cap > budget_ && !arenas_.empty()) {
        uint32_t oldest = 0;
        uint64_t best = UINT64_MAX;
        for (const auto& kv : arenas_) {
            if (kv.second.stamp < best) {
                best = kv.second.stamp;
                oldest = kv.first;
            }
        }
        drop_arena(oldest);
    }
    if (cached_bytes_ + cap > budget_) {
        err = "the slab's budget (" + std::to_string(budget_ >> 20) + " MiB) is exhausted and its " +
              "rows are all in use; raise the daemon's --ram-gb";
        return nullptr;
    }
    ArenaRec rec;
    rec.mr = std::make_unique<Mr>();
    log_alloc(cap);                      // GiBs of pinning mid-request: the caller may stall here
    if (!rec.mr->alloc(dev_->pd(), cap, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, err)) {
        return nullptr;
    }
    rec.stamp = stamp;
    const uint32_t id = next_arena_++;
    auto* out = &arenas_[id];
    *out = std::move(rec);
    *id_out = id;
    return out;
}

bool PeerCache::fill(const std::vector<long long>& rows, const std::vector<Row>& table, uint64_t stamp,
                     std::vector<Cached>& out, std::string& err) {
    out.clear();
    dropped_.clear();
    for (long long ri : rows) {
        if (ri < 0 || (size_t) ri >= table.size()) {
            err = "row " + std::to_string(ri) + " is outside the row table (" +
                  std::to_string(table.size()) + " rows)";
            return false;
        }
        const Row& r = table[(size_t) ri];
        const auto have = where_.find(ri);
        if (have != where_.end()) {
            arenas_[have->second.arena].stamp = stamp;
            out.push_back(have->second);
            continue;
        }
        uint32_t arena_id = 0;
        ArenaRec* a = room_for(r.len, stamp, &arena_id, err);
        if (a == nullptr) return false;
        const ssize_t got = pread(shard_fd_[r.shard], (char*) a->mr->buf() + a->used, r.len,
                                  (off_t) r.off);
        if (got < 0 || (size_t) got != r.len) {
            err = "shard " + std::to_string(r.shard) + " pread at " + std::to_string(r.off) +
                  " failed (" + std::strerror(errno) + ")";
            return false;
        }
        Cached c;                     // (the map's arena id: the daemon's own accounting only)
        c.addr = (uint64_t) (uintptr_t) ((char*) a->mr->buf() + a->used);
        c.len = r.len;
        c.rkey = a->mr->rkey();
        c.arena = arena_id;
        a->used += r.len;
        cached_bytes_ += r.len;
        where_[ri] = c;
        out.push_back(c);
    }
    ++clock_;
    return true;
}

}   // namespace strata::peer
