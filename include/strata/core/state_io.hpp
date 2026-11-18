#pragma once

#include "strata/core/state_cache.hpp"

namespace strata::core::statecache {

// A logical region may span physical pages, heads, or an overlaid KV view.
// Only written bytes belong here. No padding or capacity beyond the endpoint is copied.
struct Segment { void* address = nullptr; size_t bytes = 0; };
// rows: append-only rows (RegionSource::rows); a one-segment device region then also offers GPU fingerprints.
struct MemoryRegion { std::string name; std::vector<Segment> segments; bool rows = false; };

// The fingerprint (statecache::fingerprint) of each `chunk`-byte piece of a device range, on the current device.
bool device_fingerprints(const void* data, uint64_t bytes, size_t chunk, std::vector<uint64_t>& out, std::string& error);

// Call after the producer stream completes. Transfers use bounded double-buffered
// pinned staging; returned captures own their bytes independently of the live sequence.
bool capture_memory(const std::vector<MemoryRegion>& memory, std::vector<Region>& regions,
                    std::string& error, size_t budget = size_t(-1));
// Validates the complete layout before the first write. Caller resets on copy failure.
bool restore_memory(const std::vector<MemoryRegion>& memory, const std::vector<Region>& regions,
                    std::string& error);

class MemoryTransfer;
// One transfer session shares its bounded pinned buffers across all devices.
std::shared_ptr<MemoryTransfer> memory_transfer();
bool memory_sources(const std::vector<MemoryRegion>& memory, int device,
                    const std::shared_ptr<MemoryTransfer>& transfer, std::vector<RegionSource>& out, std::string& error);
bool memory_targets(const std::vector<MemoryRegion>& memory, int device,
                    const std::shared_ptr<MemoryTransfer>& transfer, std::vector<RegionTarget>& out, std::string& error,
                    bool replicate = false);

}  // namespace strata::core::statecache
