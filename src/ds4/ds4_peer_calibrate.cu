// Small, startup-only exact-F32 transport comparison. Never uses model activations or weights.
#include "ds4_kernels.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace strata::ds4 {
namespace {
using Clock = std::chrono::steady_clock;
__global__ void check_probe(const float* x, int n, unsigned expected, unsigned* error) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (__float_as_uint(x[i]) != expected) atomicOr(error, 1u);
}
struct ProbeRank {
    float* x = nullptr;
    unsigned *counter = nullptr, *error = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t begin = nullptr, end = nullptr, done = nullptr;
};
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n & 1 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
}  // namespace

PeerCalibration calibrate_peer(const int ids[2], const ArGroup& original, const ArGroup* ar2,
                               const PeerAr peer[2], int n, double budget_ms) {
    PeerCalibration result;
    const auto start = Clock::now();
    const auto deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(budget_ms));
    std::array<ProbeRank, 2> rank{};
    ArGroup host = original;
    host.ar2 = 0;
    host.pflags = nullptr;
    int previous = 0;
    hipGetDevice(&previous);
    bool launched = false;
    auto expired = [&] { return Clock::now() >= deadline; };
    auto wait = [&] {
        for (;;) {
            bool ready = true;
            for (int m = 0; m < 2; ++m) {
                if (!rank[m].stream) continue;
                if (hipSetDevice(ids[m]) != hipSuccess) return false;
                const auto status = hipStreamQuery(rank[m].stream);
                if (status != hipSuccess && status != hipErrorNotReady) return false;
                if (status == hipErrorNotReady) (void) hipGetLastError();
                ready = ready && status == hipSuccess;
            }
            if (ready) return true;
            if (expired()) return false;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    };
    auto reset_flags = [&] {
        std::memset(host.flags, 0, size_t(host.nm) * host.slots * 32 * sizeof(unsigned));
        if (ar2) std::memset(ar2->pflags, 0, size_t(ar2->nm) * ar2->slots * kAr2Blocks * 16 * sizeof(unsigned));
        bool ok = true;
        for (int m = 0; m < 2; ++m) {
            ok = hipSetDevice(ids[m]) == hipSuccess && ok;
            ok = hipMemset(peer[m].flags, 0, 2 * kPeerArBlocks * kPeerArFlagStride * sizeof(unsigned)) == hipSuccess && ok;
            ok = hipMemset(peer[m].error, 0, sizeof(unsigned)) == hipSuccess && ok;
            if (peer[m].failure) ok = hipMemset(peer[m].failure, 0, sizeof(PeerArFailure)) == hipSuccess && ok;
        }
        return ok;
    };
    bool ok = n > 0 && n % 4 == 0 && size_t(n) <= host.stride && host.nm == 2 && budget_ms > 0;
    for (int m = 0; m < 2 && ok; ++m) {
        if (expired()) { ok = false; break; }
        auto& r = rank[m];
        ok = hipSetDevice(ids[m]) == hipSuccess &&
             hipStreamCreateWithFlags(&r.stream, hipStreamNonBlocking) == hipSuccess &&
             hipMalloc((void**) &r.x, size_t(n) * sizeof(float)) == hipSuccess &&
             hipMalloc((void**) &r.counter, sizeof(unsigned)) == hipSuccess &&
             hipMalloc((void**) &r.error, sizeof(unsigned)) == hipSuccess &&
             hipEventCreate(&r.begin) == hipSuccess && hipEventCreate(&r.end) == hipSuccess &&
             hipEventCreateWithFlags(&r.done, hipEventDisableTiming) == hipSuccess;
    }
    if (ok) ok = reset_flags();
    const unsigned input[2] = {0x3f3f3f3fu, 0x40404040u};
    float left = 0, right = 0;
    std::memcpy(&left, input, sizeof(float));
    std::memcpy(&right, input + 1, sizeof(float));
    const float sum = left + right;
    unsigned expected = 0;
    std::memcpy(&expected, &sum, sizeof(expected));
    std::vector<double> times[3];
    unsigned sequence[3] = {};
    const int count = ar2 ? 3 : 2;
    const int modes[3] = {0, 2, 1};  // host, peer, optional exact AR2; rotate measured order below
    for (int round = -1; round < 6 && ok; ++round) {
        // One warmup per candidate, then balanced round-robin samples. Never enqueue a long chain.
        if (expired()) break;
        for (int j = 0; j < count && ok; ++j) {
            if (expired()) break;
            const int mode = modes[(j + std::max(round, 0)) % count];
            const unsigned seq = ++sequence[mode];
            for (int m = 0; m < 2 && ok; ++m) {
                auto& r = rank[m];
                ok = hipSetDevice(ids[m]) == hipSuccess &&
                     hipMemsetAsync(r.x, m ? 0x40 : 0x3f, size_t(n) * sizeof(float), r.stream) == hipSuccess &&
                     hipMemsetAsync(r.counter, 0, sizeof(unsigned), r.stream) == hipSuccess &&
                     hipMemsetAsync(r.error, 0, sizeof(unsigned), r.stream) == hipSuccess &&
                     hipEventRecord(r.begin, r.stream) == hipSuccess;
                if (!ok) break;
                launched = true;
                if (mode == 2) ok = allreduce_peer(r.x, n, peer[m], m, seq, r.stream);
                else allreduce(r.x, n, mode == 1 ? *ar2 : host, m, seq, r.counter, r.stream);
                ok = hipGetLastError() == hipSuccess && ok;
                ok = hipEventRecord(r.end, r.stream) == hipSuccess && ok;
                check_probe<<<128, 256, 0, r.stream>>>(r.x, n, expected, r.error);
                ok = hipGetLastError() == hipSuccess && ok;
                ok = hipEventRecord(r.done, r.stream) == hipSuccess && ok;
            }
            result.quiescent = wait();
            if (!result.quiescent) { ok = false; break; }
            double elapsed = 0;
            for (int m = 0; m < 2 && ok; ++m) {
                auto& r = rank[m];
                unsigned error = 0, transport_error = 0;
                float ms = 0;
                ok = hipSetDevice(ids[m]) == hipSuccess &&
                     hipMemcpy(&error, r.error, sizeof(error), hipMemcpyDeviceToHost) == hipSuccess &&
                     hipMemcpy(&transport_error, peer[m].error, sizeof(transport_error), hipMemcpyDeviceToHost) == hipSuccess &&
                     hipEventElapsedTime(&ms, r.begin, r.end) == hipSuccess;
                if (error || transport_error || !std::isfinite(ms) || ms <= 0) { result.valid = false; ok = false; }
                elapsed = std::max(elapsed, double(ms));
            }
            if (ok && round >= 0) times[mode].push_back(elapsed);
        }
    }
    if (ok) {
        result.samples = int(std::min(times[0].size(), times[2].size()));
        if (ar2) result.samples = std::min(result.samples, int(times[1].size()));
        if (result.samples >= 3) {
            result.complete = true;
            for (int mode : modes) {
                if (mode == 1 && !ar2) continue;
                times[mode].resize(size_t(result.samples)); // equal sample counts, including budget-limited runs
                result.ms[mode] = median(times[mode]);
            }
            int best = ar2 && result.ms[1] < result.ms[0] ? 1 : 0;
            if (result.ms[2] < result.ms[best]) best = 2;
            result.selected = static_cast<PeerTransport>(best);
        }
    }
    if (!ok && launched) result.valid = false;
    if (result.quiescent) result.quiescent = wait();
    if (result.quiescent) {
        if (rank[0].stream && rank[1].stream) result.valid = reset_flags() && result.valid;
        for (int m = 0; m < 2; ++m) {
            hipSetDevice(ids[m]);
            auto& r = rank[m];
            if (r.begin) hipEventDestroy(r.begin);
            if (r.end) hipEventDestroy(r.end);
            if (r.done) hipEventDestroy(r.done);
            if (r.x) hipFree(r.x);
            if (r.counter) hipFree(r.counter);
            if (r.error) hipFree(r.error);
            if (r.stream) hipStreamDestroy(r.stream);
        }
    }
    // If a driver/launch failure leaves in-flight work, the caller must fail loading, never reuse/free
    // buffers still read by a kernel. The driver can outlast a userspace budget; do not claim recovery.
    result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    hipSetDevice(previous);
    return result;
}
}  // namespace strata::ds4
