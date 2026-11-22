// Windows/CUDA: the production staging component, both sources and buffers,
// repeated mixed-geometry DMA, and releasing the original RAM-store arena.
#include "../src/ds4/ds4_recent_stage.hpp"
#include "../src/ds4/ds4_store.hpp"
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using Stage = strata::ds4::RecentStage;
using Store = strata::ds4::ExpertStore;
namespace os = strata::ds4::os;
static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static void CUDART_CB gate(void* p) {
    auto& open = *static_cast<std::atomic<bool>*>(p);
    while (!open.load(std::memory_order_acquire)) std::this_thread::yield();
}

int main() {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess || free < (size_t{2} << 30)) {
        std::puts("SKIP: no CUDA GPU with 2 GiB free"); return 77;
    }
    const char* path = "ds4-recent-stage-test.bin";
    os::fd_t fd = os::kBadFd;
    std::unique_ptr<Store> store;
    Stage stage;
    cudaStream_t stream = nullptr;
    std::array<cudaEvent_t, 2> events{nullptr, nullptr};
    std::array<uint8_t*, 2> device{nullptr, nullptr};
    std::atomic<bool> open{false};
    const auto drain = [&] { return !stream || cudaStreamSynchronize(stream) == cudaSuccess; };
    const auto cleanup = [&] {
        open.store(true, std::memory_order_release);
        stage.close(drain, [](uint8_t* p) { cudaFreeHost(p); });
        for (auto e : events) if (e) cudaEventDestroy(e);
        for (auto p : device) if (p) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
        store.reset();
        if (fd != os::kBadFd) os::close_fd(fd);
        std::remove(path);
    };
    try {
        const std::array<std::array<size_t, 3>, 2> sizes{{{13001, 9001, 5001}, {6001, 8001, 2001}}};
        const std::array<std::array<size_t, 3>, 2> offsets{{{0, 13312, 22528}, {0, 6144, 14336}}};
        const std::array<size_t, 2> blobs{28672, 16384};
        std::vector<Store::Geo> geo(2);
        std::vector<uint8_t> file(512);
        std::array<std::array<std::vector<uint8_t>, 2>, 2> expected;
        for (int l = 0; l < 2; ++l) {
            geo[(size_t) l].n_exp = 2;
            for (int e = 0; e < 2; ++e) expected[(size_t) l][(size_t) e].resize(blobs[(size_t) l], 0);
            for (int r = 0; r < 3; ++r) {
                auto& g = geo[(size_t) l];
                g.file[r] = 0; g.off[r] = file.size();
                g.stride[r] = g.bytes[r] = sizes[(size_t) l][(size_t) r];
                for (int e = 0; e < 2; ++e)
                    for (size_t i = 0; i < sizes[(size_t) l][(size_t) r]; ++i) {
                        const auto value = uint8_t((l * 31 + e * 61 + r * 73 + i * 37) % 251);
                        file.push_back(value);
                        expected[(size_t) l][(size_t) e][offsets[(size_t) l][(size_t) r] + i] = value;
                    }
            }
        }
        { FILE* f = std::fopen(path, "wb"); check(f != nullptr, "create fixture");
          const size_t n = std::fwrite(file.data(), 1, file.size(), f); std::fclose(f);
          check(n == file.size(), "write fixture"); }
        fd = os::open_par(path); check(fd != os::kBadFd, "open parallel fixture handle");
        Store::Files files; files.buf = {fd}; files.direct = {os::kBadFd}; files.paths = {path};
        std::atomic<uint64_t> available{100ull << 30};
        Store::Cfg cfg; cfg.budget = 2 * Store::slot_layout(geo); cfg.chunk_slots = 1; cfg.readers = 1;
        cfg.ram_avail = [&] { return available.load(); };
        store = std::make_unique<Store>(); std::string error;
        check(store->init(cfg, geo, files, error), "store init");
        check(store->preload({{0, 0}}, 0) == 1, "preload first expert only");
        check(stage.init(blobs[0], [](size_t n) -> uint8_t* {
            uint8_t* p = nullptr;
            return cudaHostAlloc((void**) &p, n, cudaHostAllocPortable) == cudaSuccess ? p : nullptr;
        }, [](uint8_t* p) { cudaFreeHost(p); }, [] { return true; }), "production staging pair");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "copy stream");
        for (int i = 0; i < 2; ++i) {
            check(cudaMalloc((void**) &device[(size_t) i], blobs[0]) == cudaSuccess, "small device destination");
            check(cudaEventCreateWithFlags(&events[(size_t) i], cudaEventDisableTiming) == cudaSuccess, "copy event");
        }
        int hits = 0, reads = 0;
        const auto prepare = [&](int l, int e, int& index) {
            const auto& g = geo[(size_t) l];
            return stage.prepare(blobs[(size_t) l], index, [&](uint8_t* p) {
                const int s = store->acquire(l, e);
                if (s < 0) return false;
                // The existing model's store_blob adapter uses this same lease contract.
                for (int r = 0; r < 3; ++r)
                    std::memcpy(p + offsets[(size_t) l][(size_t) r], store->mat_of(s, r), (size_t) g.bytes[r]);
                store->unpin(s); ++hits; return true;
            }, [&](uint8_t* p) {
                ++reads;
                for (int r = 0; r < 3; ++r)
                    if (!os::pread_par(fd, p + offsets[(size_t) l][(size_t) r], g.bytes[r],
                                       g.off[r] + uint64_t(e) * g.stride[r])) return false;
                return true;
            });
        };
        const auto submit = [&](int index, int l) {
            check(cudaMemcpyAsync(device[(size_t) index], stage.data(index), blobs[(size_t) l],
                                  cudaMemcpyHostToDevice, stream) == cudaSuccess, "staged DMA");
            check(cudaEventRecord(events[(size_t) index], stream) == cudaSuccess, "record completion");
        };
        const auto finish = [&](int index, int l, int e) {
            check(cudaEventSynchronize(events[(size_t) index]) == cudaSuccess, "copy completes");
            std::vector<uint8_t> actual(blobs[(size_t) l]);
            check(cudaMemcpy(actual.data(), device[(size_t) index], actual.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
                  "read back copied expert");
            for (int r = 0; r < 3; ++r)
                for (size_t i = 0; i < sizes[(size_t) l][(size_t) r]; ++i) {
                    const size_t at = offsets[(size_t) l][(size_t) r] + i;
                    check(actual[at] == expected[(size_t) l][(size_t) e][at], "all matrix payload bytes match file");
                }
            stage.poll(index, Stage::Completion::Complete, drain);
        };
        check(cudaLaunchHostFunc(stream, gate, &open) == cudaSuccess, "hold both queued DMAs");
        int a = -1, b = -1, waiting = -1;
        check(prepare(0, 0, a) == Stage::Prepare::Ready && a == 0, "ready-store source"); submit(a, 0);
        check(prepare(1, 1, b) == Stage::Prepare::Ready && b == 1, "file fallback source"); submit(b, 1);
        check(prepare(0, 1, waiting) == Stage::Prepare::Waiting && hits == 1 && reads == 1, "no overwrite while both busy");
        available = 0;
        check(store->watch() > 0 && store->slots_now() == 0, "source arena released with both DMAs queued");
        open.store(true, std::memory_order_release);
        finish(a, 0, 0); finish(b, 1, 1);
        for (int cycle = 0; cycle < 12; ++cycle) {
            check(prepare(0, cycle % 2, a) == Stage::Prepare::Ready && a == 0, "reuse first source"); submit(a, 0);
            check(prepare(1, (cycle + 1) % 2, b) == Stage::Prepare::Ready && b == 1, "reuse second source"); submit(b, 1);
            finish(a, 0, cycle % 2); finish(b, 1, (cycle + 1) % 2);
        }
        check(hits == 1 && reads == 25, "real store miss/file fallback exercised after release");
        cleanup();
        std::puts("PASS: production staging pair, store/file sources, queued DMA after arena release, repeated mixed layouts");
        return 0;
    } catch (const std::exception& e) {
        cleanup();
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
