// tests/mm3_link_test.cu - src/mm3/mm3_link.cuh on two GPUs (no model): the pair's setup (peer only where tested,
// else the driver's copy, else host memory) and copies both ways in each mode - host memory forced, the PC's own
// choice, peer forced - of sizes from a few bytes to several pinned pieces, many in a row without a host wait between
// (the halves reused), each against the bytes written.  One GPU: skipped (exit 0).
#include "mm3_link.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));  \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

static int run(const char* mode, bool force_host, bool force_peer, int& fails) {
    std::vector<int> devs = {0, 1};
    std::vector<char> used = {0, 1, 1, 0};
    const size_t big = (size_t) 25 << 20;   // (with 1 MiB halves: 25 pieces)
    std::vector<void*> buf(2), tb(2);
    std::vector<cudaStream_t> st(2);
    std::vector<cudaEvent_t> ev(2);
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaMalloc(&buf[d], 2 * big));
        CK(cudaMalloc(&tb[d], (size_t) 4 << 20));
        CK(cudaStreamCreate(&st[d]));
        CK(cudaEventCreateWithFlags(&ev[d], cudaEventDisableTiming));
    }
    gpulink::Links L;
    std::string err;
    if (!L.setup(devs, used, tb, (size_t) 4 << 20, st, force_host, force_peer, big, (size_t) 1 << 20, "TEST_HOST", err)) {
        std::printf("FAIL %s: setup: %s\n", mode, err.c_str());
        ++fails;
        return 0;
    }
    std::printf("%s: %s%s\n", mode, L.say.c_str(), L.all_peer ? " (all peer)" : "");
    if (force_host && (L.of(0, 1) != gpulink::kHost || L.of(1, 0) != gpulink::kHost)) {
        std::printf("FAIL %s: host memory not taken\n", mode);
        ++fails;
    }
    const size_t sizes[] = {4, 1000, 9216, 73728, (size_t) 1 << 20, ((size_t) 1 << 20) + 4, (size_t) 9 << 20, big};
    std::vector<unsigned char> src(big), back(big);
    void* pin = nullptr;
    CK(cudaHostAlloc(&pin, 2 * big, cudaHostAllocPortable));
    unsigned seed = 12345;
    for (int a = 0; a < 2; ++a) {
        const int b = 1 - a;
        // a run of copies a -> b without a host wait: each written on a's stream first, then across, its bytes into its
        // own half of b's buffer (alternating), read back on b's stream at the end of the run
        for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; ++si) {
            const size_t n = sizes[si];
            for (int rep = 0; rep < 3; ++rep) {
                unsigned char* p = (unsigned char*) pin + (size_t) (rep & 1) * big;
                CK(cudaStreamSynchronize(st[a]));   // (the pinned source half free again)
                for (size_t i = 0; i < n; ++i) seed = seed * 1103515245u + 12345u, p[i] = (unsigned char) (seed >> 16);
                CK(cudaSetDevice(devs[a]));
                char* s8 = (char*) buf[a] + (size_t) (rep & 1) * big;
                char* d8 = (char*) buf[b] + (size_t) (rep & 1) * big;
                CK(cudaMemcpyAsync(s8, p, n, cudaMemcpyHostToDevice, st[a]));
                CK(cudaEventRecord(ev[a], st[a]));
                CK(cudaSetDevice(devs[b]));
                CK(cudaStreamWaitEvent(st[b], ev[a], 0));   // (a peer / driver copy on b's stream: after the writer)
                if (!L.copy(a, st[a], s8, b, st[b], d8, n, st[b], err)) {
                    std::printf("FAIL %s: copy %d -> %d of %zu: %s\n", mode, a, b, n, err.c_str());
                    ++fails;
                    continue;
                }
                CK(cudaSetDevice(devs[b]));
                CK(cudaMemcpyAsync(back.data(), d8, n, cudaMemcpyDeviceToHost, st[b]));
                CK(cudaStreamSynchronize(st[b]));
                if (std::memcmp(back.data(), p, n) != 0) {
                    size_t w = 0;
                    for (size_t i = 0; i < n; ++i) w += back[i] != p[i];
                    std::printf("FAIL %s: copy %d -> %d of %zu bytes (rep %d): %zu wrong\n", mode, a, b, n, rep, w);
                    ++fails;
                }
            }
        }
        // the same without even a wait for the read-back: 16 copies of 3 MiB into distinct places, checked at the end
        const size_t n = (size_t) 3 << 20;
        std::vector<std::vector<unsigned char>> want(16, std::vector<unsigned char>(n));
        CK(cudaSetDevice(devs[a]));
        for (int k = 0; k < 16; ++k) {
            for (size_t i = 0; i < n; ++i) seed = seed * 1103515245u + 12345u, want[k][i] = (unsigned char) (seed >> 16);
            CK(cudaMemcpy((char*) buf[a] + (size_t) k * n, want[k].data(), n, cudaMemcpyHostToDevice));
        }
        for (int k = 0; k < 16; ++k)
            if (!L.copy(a, st[a], (char*) buf[a] + (size_t) k * n, b, st[b], (char*) buf[b] + (size_t) k * n, n, st[b], err)) {
                std::printf("FAIL %s: burst copy: %s\n", mode, err.c_str());
                ++fails;
            }
        CK(cudaSetDevice(devs[b]));
        CK(cudaStreamSynchronize(st[b]));
        for (int k = 0; k < 16; ++k) {
            CK(cudaMemcpy(back.data(), (char*) buf[b] + (size_t) k * n, n, cudaMemcpyDeviceToHost));
            if (std::memcmp(back.data(), want[k].data(), n) != 0) {
                std::printf("FAIL %s: burst copy %d -> %d, piece %d wrong\n", mode, a, b, k);
                ++fails;
            }
        }
    }
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaDeviceSynchronize());
        CK(cudaFree(buf[d]));
        CK(cudaFree(tb[d]));
    }
    CK(cudaFreeHost(pin));
    return 0;
}

int main() {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 2) {
        std::printf("mm3_link_test: skipped (%d GPU visible, two needed)\n", nd);
        return 0;
    }
    int fails = 0;
    if (run("host memory forced", true, false, fails) || run("this PC's choice", false, false, fails) ||
        run("peer forced", false, true, fails))
        return 1;
    if (fails) std::printf("mm3_link_test: %d FAIL\n", fails);
    else std::printf("mm3_link_test: all ok\n");
    return fails ? 1 : 0;
}
