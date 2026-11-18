// The fingerprint of each chunk of a device byte range, the same value as statecache::fingerprint on the host:
// the sum, mod 2^64, of mix(word ^ index * golden) over the chunk's 8-byte words (a zero-padded last word) plus
// mix(chunk length). Addition commutes, so the block-wide sum is the same in any order.
#include "strata/core/state_io.hpp"

#include <cuda_runtime.h>

namespace strata::core::statecache {
namespace {
__device__ __host__ inline uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

__global__ void chunk_fingerprints(const uint8_t* data, uint64_t bytes, uint64_t chunk, unsigned long long* out) {
    const uint64_t c = blockIdx.x, begin = c * chunk, n = bytes - begin < chunk ? bytes - begin : chunk;
    const uint64_t words = (n + 7) / 8;
    unsigned long long sum = 0;
    for (uint64_t i = threadIdx.x; i < words; i += blockDim.x) {
        uint64_t w = 0;
        const uint8_t* p = data + begin + 8 * i;
        if (8 * i + 8 <= n && (reinterpret_cast<uintptr_t>(p) & 7) == 0) w = *reinterpret_cast<const uint64_t*>(p);
        else for (uint64_t b = 0; b < 8 && 8 * i + b < n; ++b) w |= uint64_t(p[b]) << (8 * b);
        sum += mix(w ^ (i * 0x9e3779b97f4a7c15ull));
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
    __shared__ unsigned long long warps[32];
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long total = mix(n);
        for (unsigned w = 0; w < (blockDim.x + 31) / 32; ++w) total += warps[w];
        out[c] = total;
    }
}
}  // namespace

bool device_fingerprints(const void* data, uint64_t bytes, size_t chunk, std::vector<uint64_t>& out, std::string& error) {
    out.clear();
    if (!bytes) return true;
    const uint64_t count = (bytes + chunk - 1) / chunk;
    unsigned long long* sums = nullptr;
    if (count > 0x7fffffffu || cudaMalloc(&sums, count * sizeof(unsigned long long)) != cudaSuccess) {
        error = "state fingerprint buffer unavailable"; return false;
    }
    chunk_fingerprints<<<unsigned(count), 256>>>(static_cast<const uint8_t*>(data), bytes, chunk, sums);
    out.resize(size_t(count));
    const bool ok = cudaGetLastError() == cudaSuccess &&
                    cudaMemcpy(out.data(), sums, count * sizeof(uint64_t), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(sums);
    if (!ok) { out.clear(); error = "state fingerprint kernel failed"; }
    return ok;
}
}  // namespace strata::core::statecache
