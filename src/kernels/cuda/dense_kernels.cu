// src/kernels/cuda/dense_kernels.cu - see include/strata/kernels/dense_kernels.hpp.
#include "strata/kernels/dense_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kWarps = 8;            // warps per attention block: 8 cells in flight
constexpr int kChunk = 256;          // cells per attention block (one split)
constexpr int kMaxHeadDim = 256;     // the merge maps one thread to one output dim

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------------------------------------ RMSNorm
// One block per row; the row is read twice (sum of squares, then scale), which for 5120 floats stays in L1.
__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                                int cols, float eps) {
    const float* xr = x + (size_t) blockIdx.x * cols;
    float* orow = out + (size_t) blockIdx.x * cols;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += xr[i] * xr[i];
    acc = warp_sum(acc);
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    __shared__ float scale;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) s += part[i];
        scale = rsqrtf(s / (float) cols + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = xr[i] * scale * w[i];
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ o,
                              int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[i];
    o[i] = (a / (1.0f + __expf(-a))) * u[i];
}

// One warp per output row.
__global__ void gemv_f32_kernel(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                                int n_in, int n_out) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_out) return;
    const int lane = threadIdx.x & 31;
    const float* wr = W + (size_t) row * n_in;
    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32) acc += wr[i] * x[i];
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void fill_i32_kernel(int32_t* dst, int n, int32_t v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = v;
}

// ------------------------------------------------------------------------------------------------ KV cache
__global__ void kv_append_kernel(__half* __restrict__ kc, __half* __restrict__ vc, const float* __restrict__ k,
                                 const float* __restrict__ v, int pos, int head_dim, int max_ctx) {
    const int h = blockIdx.x;
    const size_t cell = ((size_t) h * max_ctx + pos) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        kc[cell + i] = __float2half(k[(size_t) h * head_dim + i]);
        vc[cell + i] = __float2half(v[(size_t) h * head_dim + i]);
    }
}

// ------------------------------------------------------------------------------------------------ attention
// Grid (n_head, n_splits), 256 threads = 8 warps.  A warp owns a cell at a time: its 32 lanes hold head_dim/32
// dims each (strided by 32 so the half loads coalesce), the dot product is a warp reduction, and the warp's
// running (m, l, acc) is the online softmax over the cells it visited (cells t = begin + warp, +8, ...).
__global__ void attn_partial_kernel(const float* __restrict__ q, const __half* __restrict__ K,
                                    const __half* __restrict__ V, float* __restrict__ part_acc,
                                    float* __restrict__ part_ml, int n_head, int n_kv, int head_dim, int n_ctx,
                                    int max_ctx, float scale, int n_splits) {
    const int h = blockIdx.x, split = blockIdx.y;
    const int kvh = h / (n_head / n_kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per = head_dim >> 5;                       // dims per lane, <= 8

    float qv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) qv[i] = i < per ? q[(size_t) h * head_dim + lane + 32 * i] * scale : 0.0f;

    const __half* Kb = K + (size_t) kvh * max_ctx * head_dim;
    const __half* Vb = V + (size_t) kvh * max_ctx * head_dim;
    const int begin = split * kChunk;
    const int end = min(n_ctx, begin + kChunk);

    float m = -FLT_MAX, l = 0.0f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

    for (int t = begin + warp; t < end; t += kWarps) {
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) d += qv[i] * __half2float(Kb[(size_t) t * head_dim + lane + 32 * i]);
        d = warp_sum(d);
        const float mn = fmaxf(m, d);
        const float corr = __expf(m - mn);               // m == -FLT_MAX on the first cell: exp(-huge) = 0
        const float p = __expf(d - mn);
        l = l * corr + p;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) acc[i] = acc[i] * corr + p * __half2float(Vb[(size_t) t * head_dim + lane + 32 * i]);
        m = mn;
    }

    __shared__ float sm[kWarps], sl[kWarps];
    __shared__ float sacc[kWarps][kMaxHeadDim];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (i < per) sacc[warp][lane + 32 * i] = acc[i];
    __syncthreads();

    if (threadIdx.x < head_dim) {
        float gm = -FLT_MAX;
        for (int w = 0; w < kWarps; ++w) gm = fmaxf(gm, sm[w]);
        float a = 0.0f, L = 0.0f;
        for (int w = 0; w < kWarps; ++w) {
            if (sl[w] == 0.0f) continue;                 // this warp saw no cell of the chunk
            const float f = __expf(sm[w] - gm);
            a += sacc[w][threadIdx.x] * f;
            L += sl[w] * f;
        }
        part_acc[((size_t) h * n_splits + split) * head_dim + threadIdx.x] = a;
        if (threadIdx.x == 0) {
            part_ml[((size_t) h * n_splits + split) * 2] = L > 0.0f ? gm : -FLT_MAX;
            part_ml[((size_t) h * n_splits + split) * 2 + 1] = L;
        }
    }
}

// Grid n_head, head_dim threads: merges the chunks of one head.
__global__ void attn_merge_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_ml,
                                  float* __restrict__ out, int head_dim, int n_splits) {
    const int h = blockIdx.x;
    float gm = -FLT_MAX;
    for (int s = 0; s < n_splits; ++s) gm = fmaxf(gm, part_ml[((size_t) h * n_splits + s) * 2]);
    float a = 0.0f, L = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float l = part_ml[((size_t) h * n_splits + s) * 2 + 1];
        if (l == 0.0f) continue;
        const float f = __expf(part_ml[((size_t) h * n_splits + s) * 2] - gm);
        a += part_acc[((size_t) h * n_splits + s) * head_dim + threadIdx.x] * f;
        L += l * f;
    }
    out[(size_t) h * head_dim + threadIdx.x] = L > 0.0f ? a / L : 0.0f;
}


// y[h, :] = rms(o[h, :]) * w * silu(z[h, :])   (Qwen3.5 gated delta net closing norm, 128-wide heads, w plain)
__global__ void gdn_out_norm_silu_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                         const float* __restrict__ w, float* __restrict__ y, int head_dim, float eps) {
    __shared__ float part[4];
    const int h = blockIdx.x, i = threadIdx.x;           // blockDim.x == head_dim == 128
    const float v = o[(size_t) h * head_dim + i];
    float acc = v * v;
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if ((i & 31) == 0) part[i >> 5] = acc;
    __syncthreads();
    const float sum = part[0] + part[1] + part[2] + part[3];
    const float inv = rsqrtf(sum / (float) head_dim + eps);
    const float zz = z[(size_t) h * head_dim + i];
    y[(size_t) h * head_dim + i] = v * inv * w[i] * (zz / (1.0f + expf(-zz)));
}

int splits_for(int cells) { return (cells + kChunk - 1) / kChunk; }

}  // namespace

void dense_rms_norm(const float* x, const float* w, float* out, int rows, int cols, float eps, void* stream) {
    rms_norm_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(x, w, out, cols, eps);
    check("dense_rms_norm");
}

void dense_gdn_out_norm_silu(const float* o, const float* z, const float* w, float* y, int heads, int head_dim, float eps,
                             void* stream) {
    if (head_dim != 128) throw std::invalid_argument("dense_gdn_out_norm_silu: head_dim must be 128");
    gdn_out_norm_silu_kernel<<<heads, 128, 0, (cudaStream_t) stream>>>(o, z, w, y, head_dim, eps);
    check("dense_gdn_out_norm_silu");
}

void dense_swiglu(const float* gate, const float* up, float* out, int64_t n, void* stream) {
    swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gate, up, out, n);
    check("dense_swiglu");
}

void dense_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream) {
    const int warps = 4;
    gemv_f32_kernel<<<(n_out + warps - 1) / warps, warps * 32, 0, (cudaStream_t) stream>>>(W, x, y, n_in, n_out);
    check("dense_gemv_f32");
}

void dense_fill_i32(int32_t* dst, int n, int32_t value, void* stream) {
    fill_i32_kernel<<<(n + 127) / 128, 128, 0, (cudaStream_t) stream>>>(dst, n, value);
    check("dense_fill_i32");
}

void dense_kv_append(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int pos, int n_kv,
                     int head_dim, int max_ctx, void* stream) {
    if (pos < 0 || pos >= max_ctx) throw std::runtime_error("dense_kv_append: position outside the cache");
    kv_append_kernel<<<n_kv, 128, 0, (cudaStream_t) stream>>>((__half*) k_cache, (__half*) v_cache, k, v, pos,
                                                              head_dim, max_ctx);
    check("dense_kv_append");
}

uint64_t dense_attn_scratch_bytes(int n_head, int head_dim, int max_ctx) {
    const uint64_t splits = (uint64_t) splits_for(max_ctx);
    return (uint64_t) n_head * splits * ((uint64_t) head_dim + 2) * sizeof(float);
}

void dense_attn_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, float* scratch,
                       int n_head, int n_kv, int head_dim, int n_ctx, int max_ctx, float scale, void* stream) {
    if (n_ctx < 1 || n_ctx > max_ctx) throw std::runtime_error("dense_attn_decode: n_ctx outside the cache");
    if (head_dim % 32 != 0 || head_dim > kMaxHeadDim || n_head % n_kv != 0)
        throw std::runtime_error("dense_attn_decode: head_dim must be a multiple of 32 up to 256 and n_head a multiple of n_kv");
    const int n_splits = splits_for(n_ctx);
    const int max_splits = splits_for(max_ctx);
    // the partials sit at a stride of n_splits of THIS call; the scratch was sized for max_splits
    float* part_acc = scratch;
    float* part_ml = scratch + (size_t) n_head * max_splits * head_dim;
    dim3 grid(n_head, n_splits);
    attn_partial_kernel<<<grid, kWarps * 32, 0, (cudaStream_t) stream>>>(
        q, (const __half*) k_cache, (const __half*) v_cache, part_acc, part_ml, n_head, n_kv, head_dim, n_ctx,
        max_ctx, scale, n_splits);
    check("dense_attn_partial");
    attn_merge_kernel<<<n_head, head_dim, 0, (cudaStream_t) stream>>>(part_acc, part_ml, out, head_dim, n_splits);
    check("dense_attn_merge");
}

}  // namespace strata::kernels
