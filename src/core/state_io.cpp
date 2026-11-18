#include "strata/core/state_io.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <map>
#include <mutex>
#include <new>

namespace strata::core::statecache {
namespace {
std::mutex staging_mutex;
struct Transfer {
    // One process-wide pool admission: concurrent stage exporters share the
    // configured scarce pinned budget instead of multiplying it per stage.
    std::unique_lock<std::mutex> admission{staging_mutex};
    size_t capacity = 4 << 20;
    std::array<uint8_t*, 2> staging{};
    std::array<cudaEvent_t, 2> ready{};
    cudaStream_t stream = nullptr;
    struct Device { cudaStream_t stream = nullptr; std::array<cudaEvent_t, 2> ready{}; };
    std::map<int, Device> devices;
    ~Transfer() {
        int before = 0; cudaGetDevice(&before);
        for (const auto& [device, context] : devices) {
            cudaSetDevice(device);
            if (context.stream) cudaStreamSynchronize(context.stream);
            for (auto event : context.ready) if (event) cudaEventDestroy(event);
            if (context.stream) cudaStreamDestroy(context.stream);
        }
        cudaSetDevice(before);
        for (auto buffer : staging) if (buffer) cudaFreeHost(buffer);
    }
    bool activate(int device) {
        if (cudaSetDevice(device) != cudaSuccess) return false;
        auto [it, fresh] = devices.try_emplace(device);
        auto& context = it->second;
        if (fresh) {
            if (cudaStreamCreateWithFlags(&context.stream, cudaStreamNonBlocking) != cudaSuccess) return false;
            for (int i = 0; i < 2; ++i)
                if (cudaEventCreateWithFlags(&context.ready[size_t(i)], cudaEventDisableTiming) != cudaSuccess) return false;
        }
        stream = context.stream; ready = context.ready;
        return stream && ready[0] && ready[1];
    }
    bool init() {
        if (const char* limit = std::getenv("MAYA_GLM_STATE_STAGING_MB")) {
            char* end = nullptr;
            const double mb = std::strtod(limit, &end);
            if (end == limit || *end || !std::isfinite(mb) || mb < 1.0/1024) return false;
            capacity = size_t(std::min<double>(double(capacity), mb * 1048576. / 2));
        }
        for (int i = 0; i < 2; ++i)
            if (cudaHostAlloc(&staging[i], capacity, cudaHostAllocPortable) != cudaSuccess) return false;
        int device = 0;
        return cudaGetDevice(&device) == cudaSuccess && activate(device);
    }
    bool copy(void* device, uint8_t* host, size_t bytes, bool load) {
        size_t pending[2]{}, offsets[2]{};
        const auto finish = [&](size_t slot) {
            if (!pending[slot]) return true;
            if (cudaEventSynchronize(ready[slot]) != cudaSuccess) return false;
            if (!load) std::memcpy(host + offsets[slot], staging[slot], pending[slot]);
            pending[slot] = 0;
            return true;
        };
        size_t slot = 0;
        for (size_t at = 0; at < bytes; at += capacity, slot ^= 1) {
            if (!finish(slot)) return false;
            const size_t n = std::min(capacity, bytes - at);
            if (load) std::memcpy(staging[slot], host + at, n);
            void* source = load ? staging[slot] : static_cast<uint8_t*>(device) + at;
            void* target = load ? static_cast<uint8_t*>(device) + at : staging[slot];
            if (cudaMemcpyAsync(target, source, n, cudaMemcpyDefault, stream) != cudaSuccess ||
                cudaEventRecord(ready[slot], stream) != cudaSuccess) return false;
            pending[slot] = n; offsets[slot] = at;
        }
        return finish(0) && finish(1);
    }
};
bool size_of(const MemoryRegion& region, size_t& bytes) {
    bytes = 0;
    for (const auto& s : region.segments) {
        if ((!s.address && s.bytes) || s.bytes > size_t(-1) - bytes) return false;
        bytes += s.bytes;
    }
    return true;
}
}

class MemoryTransfer {
    Transfer transfer_;
    bool initialized_ = false;
public:
    bool copy(const MemoryRegion& region, int device, uint64_t offset, uint8_t* bytes, size_t count,
              bool load, std::string& error) {
        int before = 0;
        if (cudaGetDevice(&before) != cudaSuccess) { error = "state transfer device unavailable"; return false; }
        struct RestoreDevice { int device; ~RestoreDevice() { cudaSetDevice(device); } } restore{before};
        if (!initialized_) {
            if (!transfer_.init()) { error = "state transfer pinned staging unavailable"; return false; }
            initialized_ = true;
        }
        if (!transfer_.activate(device)) { error = "state transfer stream unavailable"; return false; }
        size_t copied = 0;
        for (const auto& segment : region.segments) {
            if (offset >= segment.bytes) { offset -= segment.bytes; continue; }
            const size_t n = std::min<size_t>(segment.bytes-size_t(offset), count-copied);
            if (n && !transfer_.copy(static_cast<uint8_t*>(segment.address)+offset, bytes+copied, n, load)) {
                error = "state view transfer failed"; return false;
            }
            copied += n; offset = 0;
            if (copied == count) return true;
        }
        if (copied != count) { error = "state view range exceeds written bytes"; return false; }
        return true;
    }
};
std::shared_ptr<MemoryTransfer> memory_transfer() { return std::make_shared<MemoryTransfer>(); }
bool memory_sources(const std::vector<MemoryRegion>& memory, int device,
                    const std::shared_ptr<MemoryTransfer>& transfer, std::vector<RegionSource>& out, std::string& error) {
    for (const auto& region : memory) {
        size_t bytes;
        if (!size_of(region, bytes)) { error = "invalid native capture view"; return false; }
        RegionSource source;
        source.name = region.name; source.bytes = bytes; source.rows = region.rows;
        source.read = [region, device, transfer](uint64_t at, uint8_t* dst, size_t n, std::string& e) {
            return transfer->copy(region, device, at, dst, n, false, e);
        };
        cudaPointerAttributes where{};
        bool known = false;
        if (region.segments.size() == 1 && bytes) {
            known = cudaPointerGetAttributes(&where, region.segments[0].address) == cudaSuccess;
            if (!known) cudaGetLastError();   // (only the probe's own error is cleared)
        }
        if (known && region.rows && where.type == cudaMemoryTypeDevice) {
            const void* address = region.segments[0].address;
            source.fingerprints = [address, bytes, device](size_t chunk, std::vector<uint64_t>& prints, std::string& e) {
                int before = 0;
                if (cudaGetDevice(&before) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
                    e = "state fingerprint device unavailable"; return false;
                }
                const bool ok = device_fingerprints(address, bytes, chunk, prints, e);
                cudaSetDevice(before);
                return ok;
            };
        } else if (known && (where.type == cudaMemoryTypeHost || where.type == cudaMemoryTypeUnregistered)) {
            source.host = static_cast<const uint8_t*>(region.segments[0].address);
        }
        out.push_back(std::move(source));
    }
    return true;
}
bool memory_targets(const std::vector<MemoryRegion>& memory, int device,
                    const std::shared_ptr<MemoryTransfer>& transfer, std::vector<RegionTarget>& out, std::string& error,
                    bool replicate) {
    for (const auto& region : memory) {
        size_t bytes;
        if (!size_of(region, bytes)) { error = "invalid native restore view"; return false; }
        RegionTarget target{region.name, bytes, [region, device, transfer](uint64_t at, const uint8_t* src, size_t n, std::string& e) {
            return transfer->copy(region, device, at, const_cast<uint8_t*>(src), n, true, e);
        }};
        const auto same = std::find_if(out.begin(), out.end(), [&](const auto& item) { return item.name == target.name; });
        if (same == out.end()) out.push_back(std::move(target));
        else {
            if (!replicate || same->bytes != target.bytes) { error = "duplicate native restore view"; return false; }
            auto before = same->write;
            same->write = [before, next = std::move(target.write)](uint64_t at, const uint8_t* src, size_t n, std::string& e) {
                return before(at, src, n, e) && next(at, src, n, e);
            };
        }
    }
    return true;
}

bool capture_memory(const std::vector<MemoryRegion>& memory, std::vector<Region>& regions,
                    std::string& error, size_t budget) {
    regions.clear();
    try {
        std::vector<Region> result;
        std::map<std::string, bool> names;
        for (const auto& region : memory) {
            size_t bytes;
            if (region.name.empty() || !names.emplace(region.name, true).second || !size_of(region, bytes) || bytes > budget) {
                error = "state capture layout or host budget exceeded"; return false;
            }
            budget -= bytes;
            result.push_back({region.name, Bytes(bytes)});
        }
        Transfer transfer;
        if (!transfer.init()) { error = "state capture pinned staging unavailable"; return false; }
        for (size_t i = 0; i < memory.size(); ++i) {
            size_t at = 0;
            for (const auto& s : memory[i].segments) {
                if (s.bytes && !transfer.copy(s.address, result[i].bytes.data() + at, s.bytes, false)) {
                    error = "state capture transfer failed"; return false;
                }
                at += s.bytes;
            }
        }
        regions = std::move(result);
        return true;
    } catch (const std::bad_alloc&) { error = "state capture host allocation failed"; return false; }
}

bool restore_memory(const std::vector<MemoryRegion>& memory, const std::vector<Region>& regions,
                    std::string& error) {
    std::map<std::string, const Bytes*> payload;
    for (const auto& r : regions) if (!payload.emplace(r.name, &r.bytes).second) {
        error = "duplicate state region"; return false;
    }
    if (payload.size() != memory.size()) { error = "incomplete state payload"; return false; }
    for (const auto& r : memory) {
        size_t bytes;
        const auto it = payload.find(r.name);
        if (!size_of(r, bytes) || it == payload.end() || it->second->size() != bytes) {
            error = "state region layout mismatch: " + r.name; return false;
        }
    }
    Transfer transfer;
    if (!transfer.init()) { error = "state restore pinned staging unavailable"; return false; }
    for (const auto& r : memory) {
        size_t at = 0;
        for (const auto& s : r.segments) {
            if (s.bytes && !transfer.copy(s.address, const_cast<uint8_t*>(payload.at(r.name)->data()) + at, s.bytes, true)) {
                error = "state restore transfer failed"; return false;
            }
            at += s.bytes;
        }
    }
    return true;
}
}  // namespace strata::core::statecache
