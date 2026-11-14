// The daemon's slab: registered pinned arenas holding cached pack byte ranges, keyed by the row
// index the initiator assigned (its pack's row table travels the control channel once).  The cache
// is read-through on HINT only: the initiator names the rows its routing will touch next, the
// daemon faults them in from its own NVMe while the initiator still computes, then advertises them
// (addr + len + rkey per row); a row the daemon does not advertise is simply not read.
//
// Eviction is by arena, LRU, and arena identity is a stable id (never a container index): a dropped
// arena's rows leave the advertisement map by id, remaining arenas and their map entries keep
// their ids, and the initiator falls back to its disk tier for dropped rows with no protocol
// change.  Fragmentation-free beats fine-grained - rows are several MB, an arena of a few GiB
// holds hundreds of them, and an evicted arena is dropped whole.
#pragma once

#include "peer_rdma.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::peer {

/// A row as the initiator's pack defines it: a shard file, an offset, a length.
struct Row {
    uint16_t shard = 0;
    uint64_t off = 0;
    uint32_t len = 0;
    static void (*alloc_log_)(uint64_t bytes);
};

/// A cached row: the arena that holds it and where (its rkey is the arena's MR's).
struct Cached {
    uint32_t arena = 0;          // the arena's stable id
    uint64_t addr = 0;           // the remote address the initiator reads at
    uint32_t len = 0;
    uint32_t rkey = 0;
};

/// The pack fingerprint over a directory copy: per shard "size:fnv1a64(first and last MiB)",
/// joined with ';' and ended with "|<total bytes>" - the initiator and the daemon each compute it
/// over their own copy and the handshake refuses a mismatch (a re-quantized or half-copied pack
/// would otherwise serve the wrong bytes silently).
std::string pack_fingerprint(const std::string& dir, const std::vector<std::string>& names,
                             const std::vector<uint64_t>& sizes, std::string& err);

class PeerCache {
public:
    /// A hook for the daemon's log: the first arena alloc (GiBs pinned) happens mid-request
    static void set_alloc_log(void (*fn)(uint64_t bytes)) { alloc_log_ = fn; }
    static void log_alloc(uint64_t bytes) { if (alloc_log_) alloc_log_(bytes); }

public:
    /// budget: the bytes the slab may pin (0 = probe: a quarter of what is cgroup-free now, at
    /// most `cap` bytes).  arena_bytes: the MR granularity (default 2 GiB; a row larger than an
    /// arena gets an arena of its own size).  dev: borrowed, the PD the arenas register against.
    bool open(Device* dev, uint64_t budget, uint64_t arena_bytes, std::string& err);
    void close();

    /// The shard files the pack's rows point into, opened read-only and kept (pread for the
    /// fingerprint and the fills).  names/sizes exactly as the manifest carried them in.
    bool open_shards(const std::vector<std::string>& names, const std::vector<uint64_t>& sizes,
                     const std::string& pack_dir, std::string& err);
    void close_shards();

    /// Faults rows in (pread from the shards into the arenas) and returns one map entry per row
    /// (already-cached rows re-advertise their entry; the initiator's map is a cache of these).
    /// `stamp` bumps the touched arenas' LRU clocks.  Rows evicted on the way are in dropped().
    bool fill(const std::vector<long long>& rows, const std::vector<Row>& table, uint64_t stamp,
              std::vector<Cached>& out, std::string& err);

    /// The rows the last fill evicted (the initiator clears its map for them).
    const std::vector<long long>& dropped() const { return dropped_; }

    uint64_t cached_bytes() const { return cached_bytes_; }
    uint64_t budget() const { return budget_; }
    size_t cached_rows() const { return where_.size(); }
    size_t arenas() const { return arenas_.size(); }

private:
    struct ArenaRec {
        std::unique_ptr<Mr> mr;
        size_t used = 0;                    // bump pointer
        uint64_t stamp = 0;                 // LRU: bumped whenever one of its rows is touched
    };

    /// An arena with room for `bytes` (bumped to `stamp`, its stable id in *id_out), a fresh one
    /// under the budget, or null after evicting LRU arenas whole (their rows into dropped_) -
    /// false when nothing fits.
    ArenaRec* room_for(size_t bytes, uint64_t stamp, uint32_t* id_out, std::string& err);
    void drop_arena(uint32_t id);

    uint64_t budget_ = 0, arena_bytes_ = 0;
    Device* dev_ = nullptr;
    std::vector<int> shard_fd_;
    std::vector<uint64_t> shard_size_;
    std::unordered_map<uint32_t, ArenaRec> arenas_;
    uint32_t next_arena_ = 0;
    std::unordered_map<long long, Cached> where_;
    std::vector<long long> dropped_;
    uint64_t cached_bytes_ = 0;
    uint64_t clock_ = 0;
    static void (*alloc_log_)(uint64_t bytes);
};

}   // namespace strata::peer
