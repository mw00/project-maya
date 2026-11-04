// p2p_fetch_probe.cu - can card 1's PCIe link + NVLink feed card 0's expert fetches? (Andy's PC, 2x 3090 NVLink)
//
//   nvcc -O2 -arch=sm_86 -o p2p_fetch_probe p2p_fetch_probe.cu && ./p2p_fetch_probe
//
// Copies a pinned host buffer (8.6 MB "experts", 16 of them) into card 0's VRAM with Maya's fetch-kernel pattern
// (grid-stride 16-byte loads from host-mapped memory):
//   A. card 0 alone (its own x8 link)          - today's fetch
//   B. a kernel on card 1 writing card 0's VRAM through peer access (card 1's link, then NVLink)
//   C. A and B at once, each half of every expert  - the dual-link fetch
//   D. C while card 1 also fetches its own experts (the busy case)
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

// work stealing: chunks of CH bytes claimed from one counter (in card 0's memory; card 1 reaches it over NVLink)
__global__ void steal_k(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t bytes, size_t ch,
                        unsigned int* counter) {
    __shared__ unsigned int c;
    const size_t n = (bytes + ch - 1) / ch;
    for (;;) {
        if (threadIdx.x == 0) c = atomicAdd(counter, 1u);
        __syncthreads();
        const unsigned int my = c;
        __syncthreads();
        if (my >= n) return;
        const size_t o = (size_t) my * ch, len = min(ch, bytes - o);
        const uint4* s4 = (const uint4*) (src + o);
        uint4* d4 = (uint4*) (dst + o);
        for (size_t j = threadIdx.x; j < len / 16; j += blockDim.x) d4[j] = s4[j];
    }
}

__global__ void copy_k(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16) {
    for (size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x; j < n16; j += (size_t) gridDim.x * blockDim.x)
        dst[j] = src[j];
}

int main() {
    const size_t EB = 8978432, NE = 16, BYTES = EB * NE;
    int can01 = 0, can10 = 0;
    cudaDeviceCanAccessPeer(&can01, 0, 1);
    cudaDeviceCanAccessPeer(&can10, 1, 0);
    printf("peer access 0->1 %d, 1->0 %d\n", can01, can10);
    uint8_t *h = nullptr, *h1 = nullptr;
    CK(cudaHostAlloc(&h, BYTES, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc(&h1, BYTES, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(h, 1, BYTES); memset(h1, 2, BYTES);
    uint8_t *d0 = nullptr, *d1 = nullptr;
    cudaStream_t s0, s1, s1b;
    CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0)); CK(cudaMalloc(&d1, BYTES));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s1b, cudaStreamNonBlocking));
    CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0)); CK(cudaMalloc(&d0, BYTES));
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    uint4 *hp = nullptr, *hp1 = nullptr;
    CK(cudaHostGetDevicePointer((void**) &hp, h, 0));
    CK(cudaHostGetDevicePointer((void**) &hp1, h1, 0));
    const int B0 = 82, B1 = 40;
    auto sync = [&] { cudaSetDevice(0); cudaStreamSynchronize(s0); cudaSetDevice(1); cudaStreamSynchronize(s1); cudaStreamSynchronize(s1b); };
    auto run = [&](const char* name, bool a, bool b, bool own1) {
        double best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            sync();
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t e = 0; e < NE; ++e) {
                const size_t off = e * EB, half = (EB / 2) & ~(size_t) 15;
                if (a) { cudaSetDevice(0); copy_k<<<B0, 256, 0, s0>>>((const uint4*) ((uint8_t*) hp + off), (uint4*) (d0 + off), (b ? half : EB) / 16); }
                if (b) { cudaSetDevice(1); copy_k<<<B1, 256, 0, s1>>>((const uint4*) ((uint8_t*) hp + off + (a ? half : 0)), (uint4*) (d0 + off + (a ? half : 0)), (EB - (a ? half : 0)) / 16); }
                if (own1) { cudaSetDevice(1); copy_k<<<B0, 256, 0, s1b>>>((const uint4*) ((uint8_t*) hp1 + off), (uint4*) (d1 + off), EB / 16); }
            }
            sync();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (rep > 0 && s < best) best = s;
        }
        printf("%-58s %7.2f ms  %6.2f GB/s into card 0  (%.3f ms an expert)\n", name, best * 1e3, BYTES / best / 1e9, best * 1e3 / NE);
    };
    run("A. card 0's own link", true, false, false);
    run("B. card 1's link + NVLink (peer write)", false, true, false);
    run("C. both links, half each", true, true, false);
    run("D. both links, while card 1 fetches its own 16", true, true, true);
    run("E. card 0 alone while card 1 fetches its own 16", true, false, true);
    int at = 0; cudaDeviceGetP2PAttribute(&at, cudaDevP2PAttrNativeAtomicSupported, 1, 0);
    printf("native peer atomics 1->0: %d\n", at);
    unsigned int* ctr = nullptr; cudaSetDevice(0); CK(cudaMalloc(&ctr, 4 * NE));
    for (int own : {0, 1}) for (size_t ch : {(size_t) 65536, (size_t) 262144}) for (int b1 : {0, 20, 40}) {
        double best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            cudaSetDevice(0); cudaMemsetAsync(ctr, 0, 4 * NE, s0); sync();
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t e = 0; e < NE; ++e) {
                const size_t off = e * EB;
                cudaSetDevice(0); steal_k<<<B0, 256, 0, s0>>>((const uint8_t*) hp + off, d0 + off, EB, ch, ctr + e);
                if (b1) { cudaSetDevice(1); steal_k<<<b1, 256, 0, s1>>>((const uint8_t*) hp + off, d0 + off, EB, ch, ctr + e); }
                if (own) { cudaSetDevice(1); copy_k<<<B0, 256, 0, s1b>>>((const uint4*) ((uint8_t*) hp1 + off), (uint4*) (d1 + off), EB / 16); }
            }
            sync();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (rep > 0 && s < best) best = s;
        }
        printf("F. steal chunk %4zu KB, card1 blocks %2d, card1 own load %d: %7.2f ms %6.2f GB/s (%.3f ms an expert)\n",
               ch >> 10, b1, own, best * 1e3, BYTES / best / 1e9, best * 1e3 / NE);
    }
    // DMA variants (the prompt path stages experts with cudaMemcpyAsync)
    auto dma = [&](const char* name, int mode) {
        double best = 1e9;
        uint8_t* stg = nullptr; cudaSetDevice(1); cudaMalloc(&stg, BYTES);
        for (int rep = 0; rep < 5; ++rep) {
            sync();
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t e = 0; e < NE; ++e) {
                const size_t off = e * EB, half = EB / 2;
                if (mode == 0) { cudaSetDevice(0); cudaMemcpyAsync(d0 + off, h + off, EB, cudaMemcpyHostToDevice, s0); }
                if (mode == 1) { cudaSetDevice(1); cudaMemcpyAsync(d0 + off, h + off, EB, cudaMemcpyHostToDevice, s1); }
                if (mode == 2) { cudaSetDevice(0); cudaMemcpyAsync(d0 + off, h + off, half, cudaMemcpyHostToDevice, s0);
                                 cudaSetDevice(1); cudaMemcpyAsync(d0 + off + half, h + off + half, EB - half, cudaMemcpyHostToDevice, s1); }
                if (mode == 3) { cudaSetDevice(0); cudaMemcpyAsync(d0 + off, h + off, half, cudaMemcpyHostToDevice, s0);
                                 cudaSetDevice(1); cudaMemcpyAsync(stg + off + half, h + off + half, EB - half, cudaMemcpyHostToDevice, s1);
                                 cudaMemcpyPeerAsync(d0 + off + half, 0, stg + off + half, 1, EB - half, s1); }
            }
            sync();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (rep > 0 && s < best) best = s;
        }
        cudaSetDevice(1); cudaFree(stg);
        printf("%-58s %7.2f ms  %6.2f GB/s into card 0\n", name, best * 1e3, BYTES / best / 1e9);
    };
    dma("G. DMA host->card0 on card 0's stream", 0);
    dma("H. DMA host->card0 issued on card 1's stream", 1);
    dma("I. DMA both, half each (direct)", 2);
    dma("J. DMA half via card 1 staging + peer copy", 3);
    // verify
    std::vector<uint8_t> chk(4096);
    cudaSetDevice(0);
    cudaMemcpy(chk.data(), d0 + BYTES - 4096, 4096, cudaMemcpyDeviceToHost);
    printf("data check: %s\n", chk[0] == 1 && chk[4095] == 1 ? "OK" : "BAD");
    return 0;
}
