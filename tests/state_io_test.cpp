#include "strata/core/state_io.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <filesystem>
#include <chrono>
#include <tuple>

namespace sc = strata::core::statecache;

void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        std::string error;
        // Larger than both staging buffers: exercise reuse after completion, an
        // overlaid view with an offset, and validation before any destination write.
        const size_t n = 19 << 20;
        uint8_t* device = nullptr;
        require(cudaMalloc(&device, n + 31) == cudaSuccess, "allocate transfer fixture");
        sc::Bytes bytes(n);
        for (size_t i = 0; i < n; ++i) bytes[i] = uint8_t(i * 37 + i / 65536);
        require(cudaMemcpy(device + 17, bytes.data(), n, cudaMemcpyHostToDevice) == cudaSuccess, "seed fixture");
        std::vector<sc::MemoryRegion> memory{{"overlay", {{device + 17, n}}}};
        std::vector<sc::Region> saved, out;
        require(!sc::capture_memory(memory, out, error, n - 1) && out.empty(), "reject capture before over-budget allocation");
        require(sc::capture_memory(memory, saved, error, n) && saved[0].bytes == bytes, "double-buffered capture preserves view");
        cudaMemset(device, 99, n + 31);
        auto malformed = saved; malformed[0].bytes.pop_back();
        require(!sc::restore_memory(memory, malformed, error), "refuse incomplete payload");
        uint8_t before = 0; cudaMemcpy(&before, device + 17, 1, cudaMemcpyDeviceToHost);
        require(before == 99, "layout rejection did not modify live state");
        require(sc::restore_memory(memory, saved, error) && sc::capture_memory(memory, out, error), "restore overlaid view");
        require(out[0].bytes == bytes, "overlaid restore exact");
        {
            // The GPU fingerprints equal the host function, for a misaligned start, odd lengths and chunk sizes.
            for (const size_t length : {n, n - 5, size_t(1), size_t(13)})
                for (const size_t chunk : {size_t(64) << 10, size_t(4093)}) {
                    std::vector<uint64_t> gpu;
                    require(sc::device_fingerprints(device + 17, length, chunk, gpu, error), "GPU fingerprints");
                    require(gpu.size() == (length + chunk - 1) / chunk, "one fingerprint per chunk");
                    for (size_t c = 0; c < gpu.size(); ++c)
                        require(gpu[c] == sc::fingerprint(bytes.data() + c * chunk, std::min(chunk, length - c * chunk)),
                                "GPU fingerprint equals the host fingerprint");
                }
            // A state reuses its parent's device rows by fingerprint, and copies a rewritten row.
            const size_t step = 64 << 10, parent = 40 * step, child = 72 * step + 9;
            sc::Config config; config.ram_bytes = 64 << 20; config.disk_bytes = 0;
            sc::StateCache cache(config);
            auto transfer = sc::memory_transfer();
            const auto publish = [&](std::vector<int32_t> tokens, size_t length) {
                std::vector<sc::RegionSource> sources;
                const std::vector<sc::MemoryRegion> rows{{"kv", {{device + 17, length}}, true}};
                require(sc::memory_sources(rows, 0, transfer, sources, error) &&
                        sources[0].rows && sources[0].fingerprints, "a device rows region offers GPU fingerprints");
                sc::Capture metadata; metadata.prefix = {"gpu", std::move(tokens), {}};
                return cache.publish(metadata, sources, error);
            };
            require(publish({1, 2}, parent), "publish device parent");
            auto before = cache.stats();
            require(publish({1, 2, 3}, child), "publish device child");
            auto after = cache.stats();
            require(after.reused_bytes - before.reused_bytes == parent && after.read_bytes - before.read_bytes == child - parent,
                    "the child copies only the rows its parent lacks");
            const uint8_t changed = 0x5a;
            require(cudaMemcpy(device + 17 + 3 * step + 1, &changed, 1, cudaMemcpyHostToDevice) == cudaSuccess, "rewrite a row");
            before = cache.stats();
            require(publish({1, 2, 3, 4}, child), "publish over a rewritten row");
            after = cache.stats();
            require(after.read_bytes - before.read_bytes == step + 9, "only the rewritten chunk and the partial last chunk are copied");
            std::vector<sc::Region> restored;
            for (const auto& [tokens, length, edited] : {std::tuple{std::vector<int32_t>{1, 2, 9}, parent, false},
                                                         std::tuple{std::vector<int32_t>{1, 2, 3, 9}, child, false},
                                                         std::tuple{std::vector<int32_t>{1, 2, 3, 4, 9}, child, true}}) {
                auto lease = cache.find_resume({"gpu", tokens, {}});
                require(lease && cache.restore(lease, restored, error), "restore a device state");
                sc::Bytes expected(bytes.begin(), bytes.begin() + long(length));
                if (edited) expected[3 * step + 1] = changed;
                require(restored[0].bytes == expected, "each state restores its own rows exactly");
                cache.release(lease);
            }
            require(cudaMemcpy(device + 17, bytes.data(), n, cudaMemcpyHostToDevice) == cudaSuccess, "reseed fixture");
            std::vector<sc::RegionSource> host;
            const std::vector<sc::MemoryRegion> host_rows{{"kv", {{bytes.data(), 64}}, true}};
            require(sc::memory_sources(host_rows, 0, transfer, host, error) &&
                    host[0].rows && !host[0].fingerprints, "a host rows region falls back to comparison");
        }
        const auto directory = std::filesystem::temp_directory_path() /
            ("maya-state-gpu-stream-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        {
            sc::Config config; config.directory = directory.string(); config.ram_bytes = 1 << 20; config.disk_bytes = 32 << 20;
            sc::StateCache cache(config);
            memory[0].segments = {{device+17, (5 << 20)+13}, {device+17+(5 << 20)+13, n-(5 << 20)-13}};
            {
                auto transfer = sc::memory_transfer();
                std::vector<sc::RegionSource> sources;
                require(sc::memory_sources(memory, 0, transfer, sources, error), error);
                sc::Capture metadata; metadata.prefix = {"gpu-view", {1,2}, {}};
                require(cache.publish(std::move(metadata), sources, error), error);
            }
            cache.flush();
            require(cache.stats().ram_bytes == 0 && cache.stats().disk_bytes > config.ram_bytes,
                    "GPU state exceeding host budget streams to SSD");
            require(cudaMemset(device, 99, n+31) == cudaSuccess, "overwrite streamed state source");
            auto lease = cache.find_resume({"gpu-view", {1,2,3}, {}});
            {
                auto transfer = sc::memory_transfer();
                std::vector<sc::RegionTarget> targets;
                require(sc::memory_targets(memory, 0, transfer, targets, error), error);
                require(lease && cache.restore(lease, targets, error), error);
            }
            cache.restored(lease); cache.release(lease); cache.flush();
            require(sc::capture_memory(memory, out, error) && out[0].bytes == bytes,
                    "bounded GPU restore preserves every overlaid byte across segment boundaries");
        }
        std::filesystem::remove_all(directory);
        cudaFree(device);
        std::puts("state IO: bounded staging, overlaid views, invalid payloads, fingerprints and row reuse passed");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
