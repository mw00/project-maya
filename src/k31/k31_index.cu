// src/k31/k31_index.cu - the K3.1 lightning indexer: the key LayerNorm, the scores, the top-k.  See k31_index.hpp.
//
// index_score_kernel and topk_kernel are copied from src/ds4/ds4_kernels.cu (DeepSeek's indexer, Maya), with:
// the causal row limits in place of ds4's Rows (no compressor ratio, no candidate blocks), a float32 key format,
// the prescale folded into the weights in float64, no programmatic dependent launch.  The selection logic - four
// 8-bit radix passes for the k-th order key, then a block-wide scan that places the picks in position order - is
// unchanged.
#include "k31_index.hpp"

#include <cuda_runtime.h>

namespace strata::k31 {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float bf16_to_f32(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {   // round to nearest even; NaN stays a quiet NaN
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// one warp a token: layer_norm in ref/kimi_k31.py's arithmetic - float64 mean and variance, y = float32((x - mu) /
// sqrt(var + eps)), then float32(y * w), then float32(+ b): two roundings, so the _rn intrinsics (no FMA contraction)
__global__ void index_key_kernel(const float* __restrict__ kpre, const float* __restrict__ ln_w,
                                 const float* __restrict__ ln_b, double eps, int n_tokens, int dim, long long p0,
                                 int fmt, void* __restrict__ cache, float* __restrict__ k_out) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= n_tokens) return;
    const float* x = kpre + (size_t) t * dim;
    double s = 0.0;
    for (int i = lane; i < dim; i += 32) s += (double) x[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const double mu = s / dim;
    double v = 0.0;
    for (int i = lane; i < dim; i += 32) {
        const double d = (double) x[i] - mu;
        v += d * d;
    }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const double den = sqrt(v / dim + eps);
    const size_t row = (size_t) (p0 + t) * dim;
    for (int i = lane; i < dim; i += 32) {
        const float y = (float) (((double) x[i] - mu) / den);
        const float o = __fadd_rn(__fmul_rn(y, ln_w[i]), ln_b[i]);
        if (fmt == (int) IdxKeyFmt::kF32) ((float*) cache)[row + i] = o;
        else ((uint16_t*) cache)[row + i] = f32_to_bf16(o);
        if (k_out) k_out[(size_t) t * dim + i] = o;
    }
}

// a thread a key position: its key row against every head's query (shared memory), relu, the weighted sum
// (ds4 index_score_kernel)
template <int HD, int KF>
__global__ void __launch_bounds__(kThreads) index_score_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                               const uint8_t* __restrict__ kc, int heads, int n_kv,
                                                               long long p0, double prescale, float* __restrict__ out,
                                                               long long out_stride) {
    extern __shared__ float qs[];   // heads * HD, then heads weights
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < heads * HD; i += blockDim.x) qs[i] = q[(size_t) t * heads * HD + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) qs[heads * HD + i] = (float) ((double) w[(size_t) t * heads + i] * prescale);
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_kv) return;
    float* o = out + (long long) t * out_stride;
    if ((long long) i > p0 + t) {   // a future key
        o[i] = -INFINITY;
        return;
    }
    constexpr int kRowBytes = KF == (int) IdxKeyFmt::kF32 ? HD * 4 : HD * 2;
    const uint8_t* kr = kc + (size_t) i * kRowBytes;
    float acc[kIdxMaxHeads];
#pragma unroll
    for (int h = 0; h < kIdxMaxHeads; ++h) acc[h] = 0.0f;
#pragma unroll 2
    for (int c = 0; c < HD; c += 8) {
        float kv[8];
        if (KF == (int) IdxKeyFmt::kF32) {
            const float4 a = *(const float4*) (kr + c * 4), b = *(const float4*) (kr + c * 4 + 16);
            kv[0] = a.x; kv[1] = a.y; kv[2] = a.z; kv[3] = a.w; kv[4] = b.x; kv[5] = b.y; kv[6] = b.z; kv[7] = b.w;
        } else {
            const uint4 raw = *(const uint4*) (kr + c * 2);
            const uint16_t* b = (const uint16_t*) &raw;
#pragma unroll
            for (int x = 0; x < 8; ++x) kv[x] = bf16_to_f32(b[x]);
        }
#pragma unroll
        for (int h = 0; h < kIdxMaxHeads; ++h) {
            if (h >= heads) break;
            const float* qh = qs + h * HD + c;
#pragma unroll
            for (int x = 0; x < 8; ++x) acc[h] += qh[x] * kv[x];
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < kIdxMaxHeads; ++h) {
        if (h >= heads) break;
        s += fmaxf(acc[h], 0.0f) * qs[heads * HD + h];
    }
    o[i] = s;
}

__device__ __forceinline__ uint32_t ord_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// a block a row: the k-th largest key by four 8-bit radix passes, then everything above it and enough of its ties,
// placed in position order by a block-wide scan (ds4 topk_kernel)
__global__ void __launch_bounds__(1024) index_topk_kernel(const float* __restrict__ vals, long long stride, int n_kv,
                                                          int k_all, long long p0, int* __restrict__ out,
                                                          int out_stride) {
    __shared__ unsigned hist[256];
    __shared__ uint32_t prefix_s, mask_s;
    __shared__ int remaining_s;
    __shared__ int cnt_gt, cnt_eq;
    const int t = blockIdx.x;
    const long long vis = p0 + t + 1;
    const int n = (int) (vis < n_kv ? vis : n_kv), k = k_all < n ? k_all : n;
    vals += (long long) t * stride;
    out += (size_t) t * out_stride;
    for (int i = k + threadIdx.x; i < k_all; i += blockDim.x) out[i] = -1;
    if (n <= 0) return;
    if (k >= n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = i;
        return;
    }
    if (threadIdx.x == 0) {
        prefix_s = 0u;
        mask_s = 0u;
        remaining_s = k;
        cnt_gt = 0;
        cnt_eq = 0;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const uint32_t prefix = prefix_s, mask = mask_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t u = ord_key(vals[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int rem = remaining_s;
            int cum = 0;
            for (int d = 255; d >= 0; --d) {
                const int h = (int) hist[d];
                if (cum + h >= rem) {
                    prefix_s = prefix | ((uint32_t) d << shift);
                    mask_s = mask | (0xffu << shift);
                    remaining_s = rem - cum;
                    break;
                }
                cum += h;
            }
        }
        __syncthreads();
    }
    const uint32_t thr = prefix_s;
    const int remaining = remaining_s;
    const int n_gt = k - remaining;
    // the picks in index order (a scan a pass of 1024 values): the ties kept are the lowest positions
    __shared__ int wgt[32], weq[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const unsigned below = (1u << lane) - 1u;
    for (int i0 = 0; i0 < n; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x;
        const uint32_t u = i < n ? ord_key(vals[i]) : 0u;
        const bool gt = i < n && u > thr, eq = i < n && u == thr;
        const unsigned bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            wgt[wid] = __popc(bg);
            weq[wid] = __popc(be);
        }
        __syncthreads();
        if (wid == 0) {
            const int g0 = lane < nw ? wgt[lane] : 0, e0 = lane < nw ? weq[lane] : 0;
            int gs = g0, es = e0;
            for (int o = 1; o < 32; o <<= 1) {
                const int a = __shfl_up_sync(0xffffffffu, gs, o), b = __shfl_up_sync(0xffffffffu, es, o);
                if (lane >= o) {
                    gs += a;
                    es += b;
                }
            }
            if (lane < nw) {
                wgt[lane] = cnt_gt + gs - g0;
                weq[lane] = cnt_eq + es - e0;
            }
            __syncwarp();
            if (lane == 31) {
                cnt_gt += gs;
                cnt_eq += es;
            }
        }
        __syncthreads();
        int slot = -1;
        if (gt) slot = wgt[wid] + __popc(bg & below);
        else if (eq) {
            const int e = weq[wid] + __popc(be & below);
            if (e < remaining) slot = n_gt + e;
        }
        if (slot >= 0) out[slot] = i;
        __syncthreads();
    }
}

template <int HD>
void score_launch(const IndexScoreArgs& a, cudaStream_t s) {
    const dim3 grid((unsigned) ((a.n_kv + kThreads - 1) / kThreads), (unsigned) a.n_tokens);
    const size_t smem = (size_t) (a.n_head * HD + a.n_head) * sizeof(float);
    if (a.fmt == IdxKeyFmt::kF32)
        index_score_kernel<HD, (int) IdxKeyFmt::kF32><<<grid, kThreads, smem, s>>>(
            a.q, a.w, (const uint8_t*) a.keys, a.n_head, a.n_kv, a.p0, a.prescale, a.score, a.score_stride);
    else
        index_score_kernel<HD, (int) IdxKeyFmt::kBf16><<<grid, kThreads, smem, s>>>(
            a.q, a.w, (const uint8_t*) a.keys, a.n_head, a.n_kv, a.p0, a.prescale, a.score, a.score_stride);
}

bool fmt_ok(IdxKeyFmt f) { return f == IdxKeyFmt::kF32 || f == IdxKeyFmt::kBf16; }

}  // namespace

bool index_key_launch(const IndexKeyArgs& a, void* stream) {
    if (a.n_tokens < 0 || a.dim < 1 || a.dim > 1024 || a.p0 < 0 || !(a.eps > 0.0) || !fmt_ok(a.fmt)) return false;
    if (a.n_tokens == 0) return true;
    if (!a.kpre || !a.ln_w || !a.ln_b || !a.cache) return false;
    if (cudaPeekAtLastError() != cudaSuccess) return false;   // someone else's error: refuse rather than blame it
    constexpr int kWarps = 8;
    index_key_kernel<<<(a.n_tokens + kWarps - 1) / kWarps, kWarps * 32, 0, (cudaStream_t) stream>>>(
        a.kpre, a.ln_w, a.ln_b, a.eps, a.n_tokens, a.dim, a.p0, (int) a.fmt, a.cache, a.k_out);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool index_score_launch(const IndexScoreArgs& a, void* stream) {
    if (a.n_head < 1 || a.n_head > kIdxMaxHeads || (a.dim != 32 && a.dim != 64 && a.dim != 128) || a.n_tokens < 0 ||
        a.n_kv < 1 || a.p0 < 0 || a.p0 + a.n_tokens > a.n_kv || a.score_stride < a.n_kv || !(a.prescale > 0.0) ||
        !fmt_ok(a.fmt))
        return false;
    if (a.n_tokens == 0) return true;
    if (a.n_tokens > 65535 || !a.q || !a.w || !a.keys || !a.score) return false;   // (grid.y)
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (a.dim) {
        case 32: score_launch<32>(a, s); break;
        case 64: score_launch<64>(a, s); break;
        default: score_launch<128>(a, s); break;
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

bool index_topk_launch(const IndexTopkArgs& a, void* stream) {
    if (a.k < 1 || a.n_tokens < 0 || a.n_kv < 1 || a.p0 < 0 || a.p0 + a.n_tokens > a.n_kv || a.score_stride < a.n_kv ||
        a.sel_stride < a.k)
        return false;
    if (a.n_tokens == 0) return true;
    if (!a.score || !a.sel) return false;
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    index_topk_kernel<<<a.n_tokens, 1024, 0, (cudaStream_t) stream>>>(a.score, a.score_stride, a.n_kv, a.k, a.p0, a.sel,
                                                                      a.sel_stride);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::k31
