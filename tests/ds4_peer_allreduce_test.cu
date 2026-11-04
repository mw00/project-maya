// Two-rank peer-VRAM transport against the existing host transport, including failure/draining.
// Run with exactly two visible GPUs; needs about 800 MiB of spare VRAM on each, including calibration scratch.
#include "../src/ds4/ds4_kernels.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
using namespace strata::ds4;
#define CK(x) do { const auto status_ = (x); if (status_ != hipSuccess) { \
    std::fprintf(stderr, "line %d %s: %s\n", __LINE__, #x, hipGetErrorString(status_)); std::exit(2); } } while (0)
#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d failed: %s\n", __LINE__, #x); std::exit(3); } } while (0)
constexpr int kCapacity = 8192 * 5120;
constexpr unsigned kGuard = 0xdeadbeefu;

__device__ uint32_t input_bits(int i, int rank, unsigned seed) {
    const unsigned special[2][10] = {
        {0, 0x80000000u, 0x7f7fffffu, 0x7f7fffffu, 0x00800000u, 1, 0x007fffffu, 0x3f800000u, 0x3f800001u, 0xff7fffffu},
        {0x80000000u, 0x80000000u, 0xff7fffffu, 0x3f800000u, 0x00800000u, 1, 0x80000001u, 0xbf800000u, 0xb3800000u, 0x3f800000u}
    };
    if (i % 32 < 10) return special[rank][i % 32];
    const unsigned h = unsigned(i) * 2654435761u ^ (seed * 2246822519u + unsigned(rank) * 3266489917u);
    return (h & 0x807fffffu) | (((h >> 23) % 240 + 1) << 23);
}
__global__ void init_input(float* x, int n, int rank, unsigned seed) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = __uint_as_float(input_bits(i, rank, seed));
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (int j = 0; j < 4; ++j) { ((unsigned*) x)[j - 4] = kGuard; ((unsigned*) x)[n + j] = kGuard; }
}
__global__ void init_flags(unsigned* x, int n, unsigned value) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] = value;
}
__global__ void check_values(const float* x, const float* ref, int n, unsigned* error) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const unsigned actual = __float_as_uint(x[i]), expected = __float_as_uint(ref[i]);
        if (actual != expected) {
            atomicOr(error, 2u);
            if (atomicCAS(error + 1, ~0u, unsigned(i)) == ~0u) { error[2] = actual; error[3] = expected; }
        }
        if (!isfinite(x[i])) atomicOr(error, 8u);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (int j = 0; j < 4; ++j)
            if (((const unsigned*) x)[j - 4] != kGuard || ((const unsigned*) x)[n + j] != kGuard) atomicOr(error, 4u);
}
struct Rank {
    float *storage = nullptr, *x = nullptr, *reference = nullptr;
    PeerAr peer;
    unsigned* counter = nullptr;
    hipStream_t stream{};
    int wall_clock_khz = 0;
};
int main() {
    int count = 0;
    CK(hipGetDeviceCount(&count));
    if (count != 2) { std::puts("SKIP: select exactly two GPUs with HIP_VISIBLE_DEVICES"); return 77; }
    std::array<Rank, 2> ranks;
    // Probe both devices before any large allocation, so a busy serving machine can skip cleanly.
    for (int i = 0; i < 2; ++i) {
        CK(hipSetDevice(i));
        int capable = 0;
        CK(hipDeviceCanAccessPeer(&capable, i, 1 - i));
        auto& r = ranks[i];
        const auto clock_status = hipDeviceGetAttribute(&r.wall_clock_khz, hipDeviceAttributeWallClockRate, i);
        size_t available = 0, total = 0;
        CK(hipMemGetInfo(&available, &total));
        if (!capable || clock_status != hipSuccess || r.wall_clock_khz <= 0 ||
            available < 5ull * kCapacity * sizeof(float) + (64ull << 20)) {
            std::puts("SKIP: peer access, steady clock or spare VRAM unavailable"); return 77;
        }
        const auto status = hipDeviceEnablePeerAccess(1 - i, 0);
        if (status != hipErrorPeerAccessAlreadyEnabled) CK(status);
        (void) hipGetLastError();
    }
    constexpr int flag_count = 2 * kPeerArBlocks * kPeerArFlagStride;
    ArGroup host;
    host.nm = 2; host.slots = 2; host.stride = kCapacity;
    CK(hipHostMalloc(&host.data, 4ull * kCapacity * sizeof(float), hipHostMallocMapped | hipHostMallocPortable));
    CK(hipHostMalloc(&host.flags, 128 * sizeof(unsigned), hipHostMallocMapped | hipHostMallocPortable));
    for (int i = 0; i < 2; ++i) {
        CK(hipSetDevice(i));
        auto& r = ranks[i];
        CK(hipStreamCreateWithFlags(&r.stream, hipStreamNonBlocking));
        CK(hipMalloc(&r.storage, (kCapacity + 8ull) * sizeof(float)));
        r.x = r.storage + 4;
        CK(hipMalloc(&r.reference, kCapacity * sizeof(float)));
        CK(hipMalloc(&r.counter, sizeof(unsigned)));
        CK(hipMalloc(&r.peer.error, 4 * sizeof(unsigned)));
        CK(hipMalloc(&r.peer.failure, sizeof(PeerArFailure)));
        CK(hipExtMallocWithFlags((void**) &r.peer.recv, 2ull * kCapacity * sizeof(float), hipDeviceMallocFinegrained));
        CK(hipExtMallocWithFlags((void**) &r.peer.flags, flag_count * sizeof(unsigned), hipDeviceMallocFinegrained));
        r.peer.stride = kCapacity;
        r.peer.timeout_ticks = uint64_t(r.wall_clock_khz) * 2000;
        hipPointerAttribute_t attr{};
        CK(hipPointerGetAttributes(&attr, r.peer.recv));
        REQUIRE(attr.type == hipMemoryTypeDevice && attr.device == i && !attr.isManaged &&
                (attr.allocationFlags & hipDeviceMallocFinegrained));
    }
    for (int i = 0; i < 2; ++i) {
        ranks[i].peer.peer_recv = ranks[1 - i].peer.recv;
        ranks[i].peer.peer_flags = ranks[1 - i].peer.flags;
    }
    auto sync = [&] {
        for (int i = 0; i < 2; ++i) { CK(hipSetDevice(i)); CK(hipStreamSynchronize(ranks[i].stream)); }
    };
    auto reset = [&](unsigned first) {
        sync();
        std::fill(host.flags, host.flags + 128, first - 1);
        for (int i = 0; i < 2; ++i) {
            CK(hipSetDevice(i));
            auto& r = ranks[i];
            init_flags<<<32, 256, 0, r.stream>>>(r.peer.flags, flag_count, first - 1);
            CK(hipMemsetAsync(r.counter, 0, sizeof(unsigned), r.stream));
            CK(hipMemsetAsync(r.peer.failure, 0, sizeof(PeerArFailure), r.stream));
            CK(hipMemsetAsync(r.peer.error, 0, 4 * sizeof(unsigned), r.stream));
            CK(hipMemsetAsync(r.peer.error + 1, 0xff, sizeof(unsigned), r.stream));
            CK(hipGetLastError());
        }
        sync();
    };
    reset(1);
    CK(hipSetDevice(0));
    REQUIRE(!allreduce_peer(ranks[0].x, 3, ranks[0].peer, 0, 1, ranks[0].stream));
    REQUIRE(!allreduce_peer(ranks[0].x + 1, 4, ranks[0].peer, 0, 1, ranks[0].stream));
    REQUIRE(!allreduce_peer(ranks[0].x, kCapacity + 4, ranks[0].peer, 0, 1, ranks[0].stream));
    REQUIRE(!allreduce_peer(ranks[0].x, 4, ranks[0].peer, 2, 1, ranks[0].stream));
    const int sizes[] = {4, 12, 256, 5120, 5462 * 5120, 8196, kCapacity, kCapacity - 4};
    for (unsigned first : {1u, 0xffffffe0u}) {
        reset(first);
        // No host barrier between calls: exercise slot reuse, unequal rank progress and changing block counts.
        for (int trial = 0; trial < 128; ++trial) {
            const unsigned seq = first + unsigned(trial), seed = unsigned(trial) + first;
            const int n = sizes[trial % 8];
            for (int i = 0; i < 2; ++i) {
                CK(hipSetDevice(i));
                auto& r = ranks[i];
                init_input<<<256, 256, 0, r.stream>>>(r.x, n, i, seed);
                allreduce(r.x, n, host, i, seq, r.counter, r.stream);
                CK(hipMemcpyAsync(r.reference, r.x, size_t(n) * sizeof(float), hipMemcpyDeviceToDevice, r.stream));
                init_input<<<256, 256, 0, r.stream>>>(r.x, n, i, seed);
                REQUIRE(allreduce_peer(r.x, n, r.peer, i, seq, r.stream));
                check_values<<<256, 256, 0, r.stream>>>(r.x, r.reference, n, r.peer.error);
                CK(hipGetLastError());
                if (i == 0 && trial % 17 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        sync();
        for (int i = 0; i < 2; ++i) {
            CK(hipSetDevice(i));
            unsigned e[4];
            CK(hipMemcpy(e, ranks[i].peer.error, sizeof(e), hipMemcpyDeviceToHost));
            if (e[0]) std::fprintf(stderr, "rank %d flags=%u index=%u actual=%08x expected=%08x\n", i, e[0], e[1], e[2], e[3]);
            REQUIRE(e[0] == 0);
            PeerArFailure detail{};
            CK(hipMemcpy(&detail, ranks[i].peer.failure, sizeof(detail), hipMemcpyDeviceToHost));
            REQUIRE(detail.captured == 0);
        }
        std::printf("PEER_CHAIN first=%u calls=128 variable_sizes=1 signed_zero_extreme_subnormal=1 guards=1 exact=1\n", first);
        std::fflush(stdout);
    }
    // Exercise the actual startup selector, including exact AR2 and post-probe sequence reuse.
    ArGroup pipelined = host;
    pipelined.ar2 = 1;
    CK(hipHostMalloc(&pipelined.pflags, size_t(2) * host.slots * kAr2Blocks * 16 * sizeof(unsigned),
                     hipHostMallocMapped | hipHostMallocPortable));
    const int ids[2] = {0, 1};
    const PeerAr peers[2] = {ranks[0].peer, ranks[1].peer};
    const auto no_budget = calibrate_peer(ids, host, &pipelined, peers, 4096 * 5120, 0);
    REQUIRE(!no_budget.complete && no_budget.quiescent && no_budget.valid);
    const auto tuned = calibrate_peer(ids, host, &pipelined, peers, 4096 * 5120, 3000);
    REQUIRE(tuned.complete && tuned.quiescent && tuned.valid && tuned.samples >= 3);
    const int selected = int(tuned.selected);
    REQUIRE(tuned.ms[selected] <= tuned.ms[0] && tuned.ms[selected] <= tuned.ms[1] && tuned.ms[selected] <= tuned.ms[2]);
    std::printf("PEER_AUTO complete=1 selected=%d samples=%d host_ms=%.6f ar2_ms=%.6f peer_ms=%.6f elapsed_ms=%.1f\n",
                 selected, tuned.samples, tuned.ms[0], tuned.ms[1], tuned.ms[2], tuned.elapsed_ms);
    for (int mode = 0; mode < 3; ++mode) {
        for (int m = 0; m < 2; ++m) {
            CK(hipSetDevice(m));
            auto& r = ranks[m];
            init_input<<<32, 256, 0, r.stream>>>(r.x, 5120, m, 19);
            if (mode == 2) REQUIRE(allreduce_peer(r.x, 5120, r.peer, m, 1, r.stream));
            else allreduce(r.x, 5120, mode == 1 ? pipelined : host, m, 1, r.counter, r.stream);
            if (mode == 0) CK(hipMemcpyAsync(r.reference, r.x, 5120 * sizeof(float), hipMemcpyDeviceToDevice, r.stream));
            else check_values<<<32, 256, 0, r.stream>>>(r.x, r.reference, 5120, r.peer.error);
        }
        sync();
    }
    for (int m = 0; m < 2; ++m) {
        CK(hipSetDevice(m)); unsigned error = 0;
        CK(hipMemcpy(&error, ranks[m].peer.error, sizeof(error), hipMemcpyDeviceToHost));
        REQUIRE(error == 0);
    }
    std::puts("PEER_AUTO_REUSE first_sequence=1 all_transports_exact=1");
    const auto without_ar2 = calibrate_peer(ids, host, nullptr, peers, 4096 * 5120, 3000);
    REQUIRE(without_ar2.complete && without_ar2.valid && without_ar2.quiescent &&
            without_ar2.selected != PeerTransport::ar2 && without_ar2.ms[1] == 0);
    CK(hipHostFree(pipelined.pflags));
    // Deliberately omit rank 1. Check physical elapsed time independently of the device's clock.
    reset(1);
    CK(hipSetDevice(0));
    auto timeout = ranks[0].peer;
    timeout.timeout_ticks = uint64_t(ranks[0].wall_clock_khz) * 50;
    const auto timeout_start = std::chrono::steady_clock::now();
    REQUIRE(allreduce_peer(ranks[0].x, 4, timeout, 0, 1, ranks[0].stream));
    sync();
    const double timeout_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - timeout_start).count();
    REQUIRE(timeout_ms >= 45 && timeout_ms < 1000);
    CK(hipSetDevice(0));
    unsigned failure = 0;
    CK(hipMemcpy(&failure, ranks[0].peer.error, sizeof(failure), hipMemcpyDeviceToHost));
    REQUIRE(failure == 1);
    PeerArFailure first_failure{};
    CK(hipMemcpy(&first_failure, ranks[0].peer.failure, sizeof(first_failure), hipMemcpyDeviceToHost));
    REQUIRE(first_failure.captured == 1 && first_failure.sequence == 1 && first_failure.observed == 0 &&
            first_failure.block == 0 && first_failure.n4 == 1 && first_failure.rank == 0);
    REQUIRE(first_failure.elapsed_ticks >= first_failure.timeout_ticks && first_failure.timeout_ticks == timeout.timeout_ticks);
    std::printf("PEER_TIMEOUT detected=1 wall_ms=%.3f detail_checked=1\n", timeout_ms);
    // Production never resets a poisoned transport: later queued calls must drain and preserve the first record.
    CK(hipSetDevice(0));
    const auto poisoned_start = std::chrono::steady_clock::now();
    for (unsigned seq = 2; seq < 34; ++seq)
        REQUIRE(allreduce_peer(ranks[0].x, 4, ranks[0].peer, 0, seq, ranks[0].stream));
    sync();
    const double poisoned_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - poisoned_start).count();
    REQUIRE(poisoned_ms < 1000);
    CK(hipSetDevice(0));
    CK(hipMemcpy(&failure, ranks[0].peer.error, sizeof(failure), hipMemcpyDeviceToHost));
    REQUIRE(failure == 1);
    PeerArFailure after_poison{};
    CK(hipMemcpy(&after_poison, ranks[0].peer.failure, sizeof(after_poison), hipMemcpyDeviceToHost));
    REQUIRE(std::memcmp(&first_failure, &after_poison, sizeof(first_failure)) == 0);
    std::printf("PEER_POISON subsequent_calls=32 drained_ms=%.3f sticky=1\n", poisoned_ms);
    for (int i = 0; i < 2; ++i) {
        CK(hipSetDevice(i));
        auto& r = ranks[i];
        CK(hipFree(r.peer.recv)); CK(hipFree(r.peer.flags)); CK(hipFree(r.peer.error)); CK(hipFree(r.peer.failure));
        CK(hipFree(r.counter)); CK(hipFree(r.reference)); CK(hipFree(r.storage)); CK(hipStreamDestroy(r.stream));
    }
    CK(hipHostFree(host.data)); CK(hipHostFree(host.flags));
    std::puts("PEER_COMPLETE exact=1");
}
