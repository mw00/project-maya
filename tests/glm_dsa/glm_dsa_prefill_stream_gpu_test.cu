// tests/glm_dsa/glm_dsa_prefill_stream_gpu_test.cu - glm-dsa B9.9 on a device (label gpu; exits 77 without one): the
// prompt's disk landing ring from glmpfs::host_alloc_node.
//
//   * glm-dsa: the ring is page-locked host memory the device reads and writes by DMA (cudaPointerGetAttributes says
//     host memory; an H2D copy of a pattern from it and a D2H copy back into it on a stream round-trip), whether or
//     not the host has a second NUMA node to bind it to (one node: a plain cudaHostAlloc);
//   * Flash (dsa = false): exactly cudaHostAlloc (cudaHostGetFlags succeeds, as for Maya's ring);
//   * host_free_node frees either kind (a second allocation of the same size succeeds afterwards).
#include "glm_dsa_prefill_stream.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gp = strata::core::glmpfs;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

bool round_trip(uint8_t* ring, size_t bytes, cudaStream_t s) {
    for (size_t i = 0; i < bytes; ++i) ring[i] = (uint8_t) (i * 131u + 7u);
    void* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) return false;
    bool ok = cudaMemcpyAsync(d, ring, bytes, cudaMemcpyHostToDevice, s) == cudaSuccess &&
              cudaStreamSynchronize(s) == cudaSuccess;
    std::memset(ring, 0, bytes);
    ok = ok && cudaMemcpyAsync(ring, d, bytes, cudaMemcpyDeviceToHost, s) == cudaSuccess &&
         cudaStreamSynchronize(s) == cudaSuccess;
    for (size_t i = 0; ok && i < bytes; ++i) ok = ring[i] == (uint8_t) (i * 131u + 7u);
    cudaFree(d);
    return ok;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glm_dsa_prefill_stream_gpu_test: no CUDA device - skipped\n");
        return 77;
    }
    std::printf("glm_dsa_prefill_stream_gpu_test: the B9.9 landing ring\n");
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    const size_t bytes = (size_t) 24 << 20;   // two GLM-5.3 IQ2 experts' worth
    for (int dsa = 1; dsa >= 0; --dsa) {
        const std::string tag = dsa ? "glm-dsa" : "Flash";
        void* p = nullptr;
        const int e = gp::host_alloc_node(&p, bytes, 0, dsa != 0);
        check(e == 0 && p != nullptr, tag + ": allocated");
        if (p == nullptr) continue;
#ifndef STRATA_USE_HIP   // the attribute checks are CUDA-only (no hip_compat mapping); the round trip below runs on both
        cudaPointerAttributes at{};
        const bool host = cudaPointerGetAttributes(&at, p) == cudaSuccess && at.type == cudaMemoryTypeHost;
        cudaGetLastError();
        check(host, tag + ": page-locked host memory the device can DMA");
        if (!dsa) {
            unsigned flags = 0;
            check(cudaHostGetFlags(&flags, p) == cudaSuccess, tag + ": a cudaHostAlloc block (Maya's ring)");
            cudaGetLastError();
        }
#endif
        check(round_trip((uint8_t*) p, bytes, s), tag + ": H2D and D2H on a stream round-trip");
        gp::host_free_node(p);
        void* q = nullptr;
        check(gp::host_alloc_node(&q, bytes, 0, dsa != 0) == 0 && q != nullptr, tag + ": freed and allocated again");
        gp::host_free_node(q);
    }
    cudaStreamDestroy(s);
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
