// tests/ik2_p2p_stream.cu - peer copies between two GPUs, issued on the SOURCE card's stream and on the DESTINATION card's.
//
// Found on 2x RTX 2080 Ti 22 GB with NVLink (driver 615.71, CUDA 13.4): cudaMemcpyPeerAsync issued on the destination
// card's stream arrives partly wrong for about 256 KB to 3 MB (28-50% of the words at 589,824 bytes; a stream
// synchronise does not help); issued on the source card's stream every size is right.  src/mm3/mm3_link.cuh's load-time test copies on
// the destination's stream, so on such a pair it refuses peer copies; maya-ik2's IK2_PF_PEER=1 issues its copies on the
// source's stream (docs/INKLING2.md).  This program shows which way a PC's pair behaves:
//
//     nvcc -O2 -arch=sm_75 -o ik2_p2p_stream tests/ik2_p2p_stream.cu && ./ik2_p2p_stream [GPU_A GPU_B]
//
// Exit 0: the source-stream copies are right at every size, both ways (the destination-stream lines are information).
// Exit 1: a source-stream copy was wrong (IK2_PF_PEER=1 would refuse the pair at load, and the engine stays on host
// staging).  Exit 2: no peer access between the two (nothing to test; the engine copies through pinned host memory).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CK(x)                                                                                     \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            std::printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));              \
            return 3;                                                                             \
        }                                                                                         \
    } while (0)

int main(int argc, char** argv) {
    const int g0 = argc > 2 ? std::atoi(argv[1]) : 0, g1 = argc > 2 ? std::atoi(argv[2]) : 1;
    int ab = 0, ba = 0;
    if (cudaDeviceCanAccessPeer(&ab, g0, g1) != cudaSuccess || cudaDeviceCanAccessPeer(&ba, g1, g0) != cudaSuccess || !ab || !ba) {
        std::printf("no peer access between GPU %d and GPU %d: nothing to test\n", g0, g1);
        return 2;
    }
    CK(cudaSetDevice(g0));
    CK(cudaDeviceEnablePeerAccess(g1, 0));
    CK(cudaSetDevice(g1));
    CK(cudaDeviceEnablePeerAccess(g0, 0));
    const size_t sizes[] = {512, 9216, 73728, 196608, 262144, 589824, 1u << 20, 2u << 20, 3u << 20, 4u << 20, 16u << 20, 48u << 20};
    const int reps = 40;
    int src_wrong = 0;
    std::printf("%10s  %-12s %s\n", "bytes", "issued on", "copies wrong (of 2 ways x reps)");
    for (size_t nb : sizes) {
        void* buf[2];
        cudaStream_t st[2];
        const int dev[2] = {g0, g1};
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(dev[d]));
            CK(cudaMalloc(&buf[d], nb));
            CK(cudaStreamCreate(&st[d]));
        }
        for (int mode = 0; mode < 2; ++mode) {   // 0: the destination's stream, 1: the source's
            int bad = 0, runs = 0;
            for (int dir = 0; dir < 2; ++dir)
                for (int rep = 0; rep < reps; ++rep, ++runs) {
                    const int a = dir, b = 1 - dir;   // a -> b
                    std::vector<uint32_t> pat(nb / 4), back(nb / 4);
                    for (size_t i = 0; i < pat.size(); ++i) pat[i] = (uint32_t) (i * 2654435761u) ^ (uint32_t) (rep * 977 + dir * 64 + 1);
                    CK(cudaSetDevice(dev[a]));
                    CK(cudaMemcpy(buf[a], pat.data(), nb, cudaMemcpyHostToDevice));
                    CK(cudaSetDevice(dev[b]));
                    CK(cudaMemsetAsync(buf[b], 0, nb, st[b]));
                    CK(cudaStreamSynchronize(st[b]));
                    const int on = mode ? a : b;   // the card whose stream (and current device) issues the copy
                    CK(cudaSetDevice(dev[on]));
                    CK(cudaMemcpyPeerAsync(buf[b], dev[b], buf[a], dev[a], nb, st[on]));
                    CK(cudaStreamSynchronize(st[on]));
                    CK(cudaSetDevice(dev[b]));
                    CK(cudaMemcpy(back.data(), buf[b], nb, cudaMemcpyDeviceToHost));
                    size_t wrong = 0;
                    for (size_t i = 0; i < pat.size(); ++i) wrong += back[i] != pat[i];
                    bad += wrong != 0;
                }
            std::printf("%10zu  %-12s %d of %d\n", nb, mode ? "source" : "destination", bad, runs);
            if (mode) src_wrong += bad;
        }
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(dev[d]));
            cudaFree(buf[d]);
            cudaStreamDestroy(st[d]);
        }
    }
    std::printf(src_wrong ? "FAIL: copies issued on the source's stream were wrong\n"
                          : "ok: copies issued on the source's stream are right at every size\n");
    return src_wrong ? 1 : 0;
}
