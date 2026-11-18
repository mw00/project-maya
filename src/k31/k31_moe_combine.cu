// src/k31/k31_moe_combine.cu - the K3.1 latent combine on the device.  See k31_moe_combine.hpp.
//
// One block a token (256 threads): each thread sums its latent columns over the k picks in float64 and rounds once
// (moe), keeps them in shared memory, and adds moe^2 in float64; a block reduction gives the mean; then each column
// is divided in float64, rounded, and multiplied by the norm weight in float32 - ref/kimi_k31.py rms_norm's order.
#include "k31_moe_combine.hpp"

#include <cuda_runtime.h>

namespace strata::k31 {
namespace {

constexpr int kThreads = 256;

__global__ void combine_kernel(const float* __restrict__ parts, const float* __restrict__ w,
                               const float* __restrict__ norm_w, int k, int latent, float eps,
                               float* __restrict__ moe_out, float* __restrict__ normed) {
    extern __shared__ float s_moe[];   // [latent]
    __shared__ double s_red[kThreads / 32];
    __shared__ double s_w[kK31MaxPicks];
    const int t = blockIdx.x;
    const float* pt = parts + (size_t) t * k * latent;
    if (threadIdx.x < k) s_w[threadIdx.x] = (double) w[(size_t) t * k + threadIdx.x];
    __syncthreads();
    double ss = 0.0;
    for (int j = threadIdx.x; j < latent; j += kThreads) {
        double acc = 0.0;
        for (int i = 0; i < k; ++i) acc += s_w[i] * (double) pt[(size_t) i * latent + j];
        const float m = (float) acc;
        s_moe[j] = m;
        if (moe_out) moe_out[(size_t) t * latent + j] = m;
        ss += (double) m * (double) m;
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        double tot = 0.0;
        for (int i = 0; i < kThreads / 32; ++i) tot += s_red[i];
        s_red[0] = sqrt(tot / (double) latent + (double) eps);
    }
    __syncthreads();
    const double den = s_red[0];
    for (int j = threadIdx.x; j < latent; j += kThreads) {
        const float y = (float) ((double) s_moe[j] / den);
        normed[(size_t) t * latent + j] = y * norm_w[j];
    }
}

__global__ void add_kernel(const float* a, const float* b, float* out, int64_t n) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) out[i] = a[i] + b[i];
}

}  // namespace

bool moe_combine_launch(const CombineArgs& a, void* stream) {
    if (a.n_tokens < 0 || a.k < 1 || a.k > kK31MaxPicks || a.latent < 1 || a.latent > kK31MaxLatent) return false;
    if (a.n_tokens == 0) return true;
    if (!a.parts || !a.w || !a.norm_w || !a.normed) return false;
    const size_t smem = (size_t) a.latent * sizeof(float);
    if (smem > 48 * 1024 &&
        cudaFuncSetAttribute(combine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) != cudaSuccess)
        return false;
    combine_kernel<<<a.n_tokens, kThreads, smem, (cudaStream_t) stream>>>(a.parts, a.w, a.norm_w, a.k, a.latent, a.eps,
                                                                         a.moe_out, a.normed);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool add_launch(const float* a, const float* b, float* out, int64_t n, void* stream) {
    if (n < 0 || (n > 0 && (!a || !b || !out))) return false;
    if (n == 0) return true;
    const int64_t want = (n + 255) / 256;
    add_kernel<<<(int) (want < 4096 ? want : 4096), 256, 0, (cudaStream_t) stream>>>(a, b, out, n);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::k31
