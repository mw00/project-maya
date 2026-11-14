#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::statecache {

using Bytes = std::vector<uint8_t>;
std::string content_digest(const void* data, size_t bytes);
// Detects accidental change of a byte range (not a cryptographic hash): the sum, mod 2^64, of a mixed value per
// 8-byte word and its index. state_io computes the same value on the GPU.
uint64_t fingerprint(const void* data, size_t bytes);

struct ExtraInput {
    uint64_t position = 0, length = 0;
    std::string identity;
    bool operator==(const ExtraInput& other) const {
        return position == other.position && length == other.length && identity == other.identity;
    }
};

struct Prefix {
    std::string identity;
    std::vector<int32_t> tokens;
    std::vector<ExtraInput> extra;
};

bool extends(const Prefix& longer, const Prefix& shorter);

struct Region {
    std::string name;
    Bytes bytes;
};

struct Capture {
    Prefix prefix;
    std::vector<Region> regions;
    bool sampling_ready = false;
    bool anchor = false;
    bool frontier = false;  // completed/paused answer, eligible for delayed supersession
    uint64_t min_prompt_tokens = 0;  // e.g. a drafter retaining only its prompt window
};

// A synchronous, coherent native view. The caller keeps every source allocation
// unchanged until publish returns. Reads may span multiple physical segments.
struct RegionSource {
    std::string name;
    uint64_t bytes = 0;
    std::function<bool(uint64_t, uint8_t*, size_t, std::string&)> read;
    // Append-only rows: the bytes a shorter prefix holds never change as the sequence grows. A new state reuses
    // its parent's chunks for them once they are checked to match the live bytes.
    bool rows = false;
    // Optional: the fingerprint of each `chunk`-byte piece of the whole region, computed where its bytes live.
    // Without it, a reuse check reads the live bytes and compares them.
    std::function<bool(size_t chunk, std::vector<uint64_t>&, std::string&)> fingerprints;
    // Optional: the region's bytes when they are one host-memory range, read and compared in place.
    const uint8_t* host = nullptr;
};
struct RegionTarget {
    std::string name;
    uint64_t bytes = 0;
    std::function<bool(uint64_t, const uint8_t*, size_t, std::string&)> write;
};
RegionSource bytes_source(std::string name, Bytes bytes);
RegionTarget check_region(std::string name, Bytes expected);

struct CostModel {
    double per_token = 1, per_token_per_100k = 0;
    double seconds(uint64_t from, uint64_t to) const;
};

struct Config {
    uint64_t ram_bytes = 2ull << 30, disk_bytes = 16ull << 30;
    size_t max_units = 128, chunk_bytes = 64 << 10;
    double idle_seconds = 60;
    double max_age_seconds = 7 * 86400.;
    bool leaf_last = true, anchor_tier = true, write_through = true;
    bool calibrate = true;
    size_t supersede_lag = 2;
    std::string directory;
    CostModel cost;
    static Config environment(const std::string& default_directory, const std::string& family);   // "GLM": MAYA_GLM_STATE_*
};

struct Stats {
    uint64_t hits = 0, misses = 0, restored_tokens = 0;
    uint64_t admitted = 0, rejected = 0, evicted = 0, demoted = 0, promoted = 0;
    uint64_t read_failures = 0, write_failures = 0;
    uint64_t ram_bytes = 0, disk_bytes = 0, logical_bytes = 0, units = 0;
    double watermark = 0;
    double cost_per_token = 1, cost_per_token_per_100k = 0;
    uint64_t calibration_samples = 0;
    uint64_t pending_writes = 0;
    uint64_t pending_updates = 0;
    uint64_t unique_bytes = 0, shared_bytes = 0, pinned_units = 0;
    uint64_t ram_hits = 0, disk_hits = 0;
    uint64_t publishes = 0, read_bytes = 0, reused_bytes = 0, hashed_bytes = 0, restores = 0;
    double publish_seconds = 0, restore_seconds = 0;
};

class Storage {
public:
    virtual ~Storage() = default;
    virtual bool put(const std::string& key, const Bytes& data, std::string& error) = 0;
    virtual bool get(const std::string& key, Bytes& data, std::string& error) = 0;
    virtual bool erase(const std::string& key, std::string& error) = 0;
    virtual std::vector<std::string> manifests(std::string& error) = 0;
    // Removes objects left behind by an interrupted publication.
    virtual void prune(const std::vector<std::string>&, std::string&) {}
    virtual void touch(const std::string&) {}
    virtual uint64_t local_bytes(const std::string&, uint64_t expected) const { return expected; }
    virtual double age_seconds(const std::string&) const { return 0; }
};

std::shared_ptr<Storage> file_storage(const std::string& directory);

class StateCache {
    struct Unit;
    struct Impl;
public:
    class Lease {
        friend class StateCache;
        std::shared_ptr<Unit> unit_;
        bool restored_ = false;
    public:
        explicit operator bool() const { return bool(unit_); }
        uint64_t tokens() const;
        bool sampling_ready() const;
        const Prefix& prefix() const;
    };

    explicit StateCache(Config config, std::shared_ptr<Storage> storage = {},
                        std::function<double()> clock = {});
    ~StateCache();
    StateCache(const StateCache&) = delete;
    StateCache& operator=(const StateCache&) = delete;

    bool publish(Capture capture, std::string& error);
    bool publish(Capture metadata, const std::vector<RegionSource>& sources, std::string& error);
    Lease find_resume(const Prefix& prompt, bool need_sample = true);
    bool restore(const Lease& lease, std::vector<Region>& regions, std::string& error);
    // Layouts are validated before any native write. A failed write/load requires
    // the adapter to reset its private arena before recomputing.
    bool restore(const Lease& lease, const std::vector<RegionTarget>& targets, std::string& error);
    bool read_region(const Lease& lease, const std::string& name, Bytes& bytes, std::string& error,
                     uint64_t limit = 1 << 20);
    void restored(Lease& lease);  // adapter confirms every native device completed
    void release(Lease& lease);
    void supersede(const Prefix& newest, size_t keep_ancestors = 1);
    // Invalidate all recovery units on either side of a failed conversation's path.
    // Readers already holding leases retain allocations until they release them.
    void invalidate(const Prefix& failed);
    void observe_prefill(uint64_t from, uint64_t to, double seconds);
    void flush();
    Stats stats() const;
    std::string report() const;

private:
    std::unique_ptr<Impl> p_;
};

struct ExportPlan {
    uint64_t end = 0, checkpoint = 0;
};
ExportPlan plan_export(uint64_t start, uint64_t candidate_end, const std::vector<uint64_t>& marks,
                       uint64_t alignment, bool supported);

struct Commit {
    uint64_t start = 0, processed = 0, accepted = 0, kept = 0;
    bool completed = false;
    bool exact(uint64_t& endpoint) const;
};

}  // namespace strata::core::statecache
