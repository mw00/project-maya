// src/k31/k31_situ.cu - SiTU-GLU on the device.  See k31_situ.hpp.  Do not build with --use_fast_math.
#include "k31_situ.hpp"

#include <cuda_runtime.h>

namespace strata::k31 {
namespace {

template <bool PORTABLE>
__global__ void situ_kernel(const float* gate, const float* up, float* out, int64_t n,   // (out may alias gate or up)
                            float beta, float lbeta) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float g = gate[i], u = up[i];
        out[i] = PORTABLE ? situ_portable(g, u, beta, lbeta) : situ(g, u, beta, lbeta);
    }
}

}  // namespace

bool situ_launch(const float* gate, const float* up, float* out, int64_t n, float beta, float lbeta, bool portable,
                 void* stream) {
    if (n < 0 || (n > 0 && (!gate || !up || !out))) return false;
    if (n == 0) return true;
    constexpr int kThreads = 256;
    const int64_t want = (n + kThreads - 1) / kThreads;
    const int blocks = (int) (want < 4096 ? want : 4096);
    auto s = (cudaStream_t) stream;
    if (portable) situ_kernel<true><<<blocks, kThreads, 0, s>>>(gate, up, out, n, beta, lbeta);
    else situ_kernel<false><<<blocks, kThreads, 0, s>>>(gate, up, out, n, beta, lbeta);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::k31
