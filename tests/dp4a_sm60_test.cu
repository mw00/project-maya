// tests/dp4a_sm60_test.cu - __dp4a as the kernels see it (include/strata/kernels/dp4a.hpp: four vmad on sm_60, the
// hardware instruction on sm_61 and newer) against its byte-wise definition, both integer forms, bit for bit (needs a
// GPU).  Per form ~1.07e9 cases: every edge byte (0x00, 0x7f, 0x80, 0xff) against every edge byte in all four lanes,
// every (lane, a byte, b byte) combination once with random bytes in the other lanes, the rest random.  It also times
// the signed form against the byte loop (informational).
#include <cuda_runtime.h>
#include "../include/strata/kernels/dp4a.hpp"

#include <cstdint>
#include <cstdio>

namespace {

constexpr int BLOCKS = 1024, THREADS = 256, ITERS = 4096;   // 1024 * 256 * 4096 = ~1.07e9 cases a form

__device__ __forceinline__ int byte_dp4a(const int a, const int b, const int c) {
    const int8_t* a8 = (const int8_t*) &a;
    const int8_t* b8 = (const int8_t*) &b;
    // the four products' sum fits an int; adding c wraps as the instruction does (unsigned, no signed overflow)
    return (int) ((unsigned) c + (unsigned) (a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3]));
}
__device__ __forceinline__ unsigned byte_dp4a(const unsigned a, const unsigned b, const unsigned c) {
    const uint8_t* a8 = (const uint8_t*) &a;
    const uint8_t* b8 = (const uint8_t*) &b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}

// splitmix32: every output byte depends on the whole state, so every byte lane sees every byte pair
__device__ uint32_t rnd(uint32_t& s) {
    uint32_t z = (s += 0x9e3779b9u);
    z = (z ^ (z >> 16)) * 0x85ebca6bu;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    return z ^ (z >> 16);
}

// iterations 0-15: the edge bytes against the edge bytes; 16: exhaustive across the grid - its 262,144 threads are
// exactly 4 lanes x 256 x 256, one (lane, a byte, b byte) each; the rest random
template <typename T>
__global__ void check_kernel(unsigned long long* bad_out) {
    const uint32_t edge[4] = {0x7f7f7f7fu, 0x80808080u, 0xffffffffu, 0u};
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t s = t * 0x2545f491u;
    unsigned long long bad = 0;
    for (int i = 0; i < ITERS; ++i) {
        uint32_t a = rnd(s), b = rnd(s);
        const uint32_t c = rnd(s);
        if (i < 16) { a = edge[i & 3]; b = edge[i >> 2]; }
        if (i == 16) {
            const int sh = 8 * (int) (t >> 16);
            a = (a & ~(0xffu << sh)) | (((t >> 8) & 0xffu) << sh);
            b = (b & ~(0xffu << sh)) | ((t & 0xffu) << sh);
        }
        bad += __dp4a((T) a, (T) b, (T) c) != byte_dp4a((T) a, (T) b, (T) c);
    }
    atomicAdd(bad_out, bad);
}

template <bool DP4A>
__global__ void time_kernel(int* out, int iters) {
    unsigned a = threadIdx.x * 0x01020304u, b = blockIdx.x * 0x05060708u + 1u;
    int acc = 0;
    for (int i = 0; i < iters; ++i) {
        acc = DP4A ? __dp4a((int) a, (int) b, acc) : byte_dp4a((int) a, (int) b, acc);
        a ^= (unsigned) acc;
        b += 0x01010101u;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <bool DP4A>
float time_ms(int* out) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    time_kernel<DP4A><<<BLOCKS, THREADS>>>(out, 1 << 12);   // warm-up
    cudaEventRecord(t0);
    time_kernel<DP4A><<<BLOCKS, THREADS>>>(out, 1 << 16);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

}  // namespace

int main() {
    static_assert(BLOCKS * THREADS == 4 * 256 * 256, "iteration 16 needs one thread per (lane, a byte, b byte)");
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::printf("dp4a_sm60_test: no CUDA device - skipped\n");
        return 0;
    }
    unsigned long long* d_bad = nullptr;
    int* d_out = nullptr;
    cudaMalloc(&d_bad, 2 * sizeof(unsigned long long));
    cudaMemset(d_bad, 0, 2 * sizeof(unsigned long long));
    cudaMalloc(&d_out, BLOCKS * THREADS * sizeof(int));
    check_kernel<int><<<BLOCKS, THREADS>>>(d_bad);
    check_kernel<unsigned><<<BLOCKS, THREADS>>>(d_bad + 1);
    unsigned long long bad[2] = {0, 0};
    cudaError_t e = cudaMemcpy(bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost);
    const float dp4a_ms = time_ms<true>(d_out), byte_ms = time_ms<false>(d_out);
    if (e == cudaSuccess) e = cudaGetLastError();
    const unsigned long long cases = (unsigned long long) BLOCKS * THREADS * ITERS;
    std::printf("%s (sm_%d%d, __dp4a %s): signed %llu cases, %llu differ; unsigned %llu cases, %llu differ\n",
                prop.name, prop.major, prop.minor, prop.major * 10 + prop.minor < 61 ? "four vmad" : "the hardware's",
                cases, bad[0], cases, bad[1]);
    std::printf("signed __dp4a %.1f ms, byte loop %.1f ms (%.2fx); cuda: %s\n", dp4a_ms, byte_ms, byte_ms / dp4a_ms,
                cudaGetErrorString(e));
    cudaFree(d_bad);
    cudaFree(d_out);
    return (bad[0] == 0 && bad[1] == 0 && e == cudaSuccess) ? 0 : 1;
}
