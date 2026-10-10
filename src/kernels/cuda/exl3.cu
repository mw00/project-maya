// src/kernels/cuda/exl3.cu - the EXL3 device kernels (the format: include/strata/kernels/exl3.hpp).
//
// The codebooks and the trellis window arithmetic follow exllamav3 (MIT, turboderp - exllamav3_ext/quant/
// codebook.cuh, exl3_dq.cuh, hadamard_inner.cuh); the kernels around them are shaped for this engine's decode:
//
//   * a block owns 128 outputs (8 tiles - exactly one output rotation block) over a slice of at most 4096 inputs;
//     its 8 warps take the slice's 16-row tile rows in turn, each staging the row's 8 tiles in shared memory with
//     coalesced loads and decoding 8 weights per lane (one fragment: 2 columns x 4 rows), F32 accumulation;
//   * the input rotation runs per block on the slice it reads (F32 or q8_1 in, FP16 in shared memory), so the
//     engine's activation buffers are used as they are;
//   * a matrix with more than 4096 inputs, or too few output blocks to fill the device, splits its inputs over
//     blocks: the partial sums go to a per-stream scratch and the LAST block of an output block adds them up in
//     split order (deterministic) and runs the output rotation.
#include "strata/kernels/exl3.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <utility>

namespace strata::kernels::exl3 {
namespace {

constexpr int kThreads = 256, kWarps = 8;
constexpr int kCols = 128;                // outputs per block: 8 tiles, one rotation block
constexpr int kMaxKR = 4096;              // inputs per block (the rotated slice lives in shared memory)
constexpr float kHad = 0.088388347648f;   // 1 / sqrt(128)

struct Q81 {   // the engine's q8_1 block (ggml block_q8_1): d, sum, 32 int8
    __half2 ds;
    int8_t qs[32];
};

std::atomic<int> g_launch_errors{0};
void launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "exl3 %s: %s\n", what, cudaGetErrorString(e));
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- the codebooks
__device__ __forceinline__ uint32_t bytesum_plus(uint32_t x, uint32_t acc) {
#if defined(STRATA_USE_HIP)
    return acc + (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24);
#else
    return __dp4a(x, 0x01010101u, acc);
#endif
}

// two states -> two values, FP16 arithmetic as exllamav3's decode_3inst_2 (the tensor-core kernels take the half2)
template <int cb>
__device__ __forceinline__ __half2 decode2h(uint32_t s0, uint32_t s1) {
    if constexpr (cb == 2) {
        const uint32_t a = bytesum_plus(s0 * 0x83DCD12Du, 0x6400u);
        const uint32_t b = bytesum_plus(s1 * 0x83DCD12Du, 0x6400u);
        const __half2 h = __halves2half2(__ushort_as_half((unsigned short) a), __ushort_as_half((unsigned short) b));
        return __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
    } else {
        uint32_t x0 = cb == 1 ? s0 * 0xCBAC1FEDu : s0 * 89226354u + 64248484u;
        uint32_t x1 = cb == 1 ? s1 * 0xCBAC1FEDu : s1 * 89226354u + 64248484u;
        x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
        x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
        const __half2 lo = __halves2half2(__ushort_as_half((unsigned short) (x0 & 0xffffu)),
                                          __ushort_as_half((unsigned short) (x1 & 0xffffu)));
        const __half2 hi = __halves2half2(__ushort_as_half((unsigned short) (x0 >> 16)),
                                          __ushort_as_half((unsigned short) (x1 >> 16)));
        return __hadd2(lo, hi);
    }
}
template <int cb>
__device__ __forceinline__ float2 decode2(uint32_t s0, uint32_t s1) {
    return __half22float2(decode2h<cb>(s0, s1));
}

// step p's state: the 16-bit window ending at bit (p + 1) K of the tile's MSB-first bit stream (w: its 8 K words),
// wrapping around the tile (the + 256 K keeps the start non-negative)
template <int K>
__device__ __forceinline__ uint32_t state_at(const uint32_t* w, int p) {
    constexpr int W = 8 * K;
    const int b1 = (p + 1 + 256) * K;
    const int i1 = (b1 - 1) >> 5, i0 = (b1 - 16) >> 5;
    const int sh = ((i1 + 1) << 5) - b1;
    const uint64_t m = ((uint64_t) w[i0 % W] << 32) | (uint64_t) w[i1 % W];
    return (uint32_t) (m >> sh) & 0xffffu;
}

// lane's 8 steps (lane * 8 + j): v[j] sits at tile row (lane & 3) * 2 + (j & 1) + 8 (j >> 1 & 1), column
// (lane >> 2) + 8 (j >> 2)
template <int K, int cb>
__device__ __forceinline__ void decode8(const uint32_t* w, int lane, float* v) {
    const int p0 = lane * 8;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const float2 d = decode2<cb>(state_at<K>(w, p0 + j), state_at<K>(w, p0 + j + 1));
        v[j] = d.x;
        v[j + 1] = d.y;
    }
}

// ---------------------------------------------------------------- the rotation
// the 128-point transform of the 128 values a warp holds 4 per lane (value lane * 4 + i), over sqrt(128)
__device__ __forceinline__ void had128(float* v, int lane) {
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1;
    v[1] = d0 + d1;
    v[2] = s0 - s1;
    v[3] = d0 - d1;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= kHad;
}

__device__ __forceinline__ float h2f(uint16_t b) { return __half2float(__ushort_as_half(b)); }

// xs[i] = H(x * suh)[k0 + i] for i < kr (kr % 128 == 0); x from xf (F32) or xq (q8_1); every warp of the block
__device__ void rotate_in(const float* __restrict__ xf, const Q81* __restrict__ xq, const uint16_t* __restrict__ suh,
                          int k0, int kr, __half* xs) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int c = warp; c < kr / 128; c += kWarps) {
        const int base = k0 + c * 128 + lane * 4;
        float v[4];
        if (xf != nullptr) {
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = xf[base + i];
        } else {
            const Q81* b = xq + base / 32;
            const float d = __low2float(b->ds);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = d * (float) b->qs[base % 32 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= h2f(suh[base + i]);
        had128(v, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) xs[c * 128 + lane * 4 + i] = __float2half_rn(v[i]);
    }
}

// ---------------------------------------------------------------- the block GEMV
// red[c] (c < 128) = sum over the 16-row slices [ks0, ks1) of x' . W'[:, tc0 * 16 + c], x' = xs from slice ks0 on.
// tr: the matrix's first tile, ld tiles per slice row.  stage: kWarps * 64 K words, red: kWarps * 128 floats.
// Ends with a __syncthreads (red[0..127] complete).
template <int K, int cb>
__device__ void block_gemv(const uint32_t* __restrict__ tr, int ld, int tc0, int ks0, int ks1, const __half* xs,
                           uint32_t* stage, float* red) {
    constexpr int W = 8 * K, TW = 8 * W;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t* st = stage + warp * TW;
    float acc[8][2];
#pragma unroll
    for (int t = 0; t < 8; ++t) acc[t][0] = acc[t][1] = 0.0f;
    const int kk = (lane & 3) * 2;
    for (int ks = ks0 + warp; ks < ks1; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) ks * ld + tc0) * W;
        __syncwarp();
#pragma unroll
        for (int i = lane; i < TW; i += 32) st[i] = src[i];
        __syncwarp();
        const __half* xk = xs + (ks - ks0) * 16;
        const float x0 = __half2float(xk[kk]), x1 = __half2float(xk[kk + 1]);
        const float x2 = __half2float(xk[kk + 8]), x3 = __half2float(xk[kk + 9]);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            float v[8];
            decode8<K, cb>(st + t * W, lane, v);
            acc[t][0] += v[0] * x0 + v[1] * x1 + v[2] * x2 + v[3] * x3;
            acc[t][1] += v[4] * x0 + v[5] * x1 + v[6] * x2 + v[7] * x3;
        }
    }
    // the 4 lanes of one lane / 4 hold the same two columns of every tile
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            acc[t][c] += __shfl_xor_sync(0xffffffffu, acc[t][c], 1);
            acc[t][c] += __shfl_xor_sync(0xffffffffu, acc[t][c], 2);
        }
    if ((lane & 3) == 0)
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            red[warp * kCols + t * 16 + (lane >> 2)] = acc[t][0];
            red[warp * kCols + t * 16 + 8 + (lane >> 2)] = acc[t][1];
        }
    __syncthreads();
    if (threadIdx.x < kCols) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) s += red[w * kCols + threadIdx.x];
        red[threadIdx.x] = s;
    }
    __syncthreads();
}

// one warp: y[c] = alpha * (H(red) * svh)[c] + bias[c] for the block's 128 outputs
__device__ __forceinline__ void epilogue(const float* red, const uint16_t* svh, float alpha, const float* bias,
                                         float* y) {
    const int lane = threadIdx.x & 31;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = lane * 4 + i;
        y[c] = alpha * v[i] * h2f(svh[c]) + (bias != nullptr ? bias[c] : 0.0f);
    }
}

// ---------------------------------------------------------------- mv
struct KJob {
    const uint32_t* tr;
    const uint16_t* suh;
    const uint16_t* svh;
    const float* xf;
    const Q81* xq;
    float* y;
    const float* bias;
    float* part;       // S x n partial sums (S > 1)
    unsigned* cnt;     // n / 128 counters, zero between launches
    float alpha;
    int k, n, ld, S, kr;
};
struct KBatch {
    KJob j[kMaxMvJobs];
    int blk_end[kMaxMvJobs];
    int n;
};

template <int K, int cb>
__global__ void __launch_bounds__(kThreads) mv_kernel(const KBatch b) {
    __shared__ __half xs[kMaxKR];
    __shared__ uint32_t stage[kWarps * 64 * K];
    __shared__ float red[kWarps * kCols];
    __shared__ int last;
    int ji = 0;
    while ((int) blockIdx.x >= b.blk_end[ji]) ++ji;
    const KJob& J = b.j[ji];
    const int local = (int) blockIdx.x - (ji > 0 ? b.blk_end[ji - 1] : 0);
    const int g = local / J.S, sp = local % J.S;
    const int k0 = sp * J.kr, k1 = min(J.k, k0 + J.kr);
    rotate_in(J.xf, J.xq, J.suh, k0, k1 - k0, xs);
    __syncthreads();
    block_gemv<K, cb>(J.tr, J.ld, g * 8, k0 / 16, k1 / 16, xs, stage, red);
    const int warp = threadIdx.x >> 5;
    if (J.S == 1) {
        if (warp == 0)
            epilogue(red, J.svh + g * kCols, J.alpha, J.bias != nullptr ? J.bias + g * kCols : nullptr,
                     J.y + g * kCols);
        return;
    }
    if (threadIdx.x < kCols) J.part[(size_t) sp * J.n + g * kCols + threadIdx.x] = red[threadIdx.x];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&J.cnt[g], 1u) == (unsigned) (J.S - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < kCols) {
        float s = 0.0f;
        for (int i = 0; i < J.S; ++i) s += *(volatile const float*) &J.part[(size_t) i * J.n + g * kCols + threadIdx.x];
        red[threadIdx.x] = s;
    }
    __syncthreads();
    if (warp == 0)
        epilogue(red, J.svh + g * kCols, J.alpha, J.bias != nullptr ? J.bias + g * kCols : nullptr, J.y + g * kCols);
    if (threadIdx.x == 0) J.cnt[g] = 0u;
}

// ---------------------------------------------------------------- the routed experts
struct KMoe {
    const unsigned long long* plan_ptr;
    const float* plan_w;
    const float* cpu_part;
    const int* cpu_flag;
    const float* x;
    float* h;            // [k][n_ff]
    const float* sh_out;
    float* out;
    float* part;         // [k][n_embd]
    unsigned* cnt;       // n_embd / 128
    size_t suh[3], svh[3], tr[3];
    int k, n_embd, n_ff;
    float limit;
};

// grid (n_ff / 128, k): expert blockIdx.y's gate and up for 128 h values, then the clamped swiglu
template <int K, int cb>
__global__ void __launch_bounds__(kThreads) moe_gate_up_kernel(const KMoe a) {
    __shared__ __half xs[2][kMaxKR];
    __shared__ uint32_t stage[kWarps * 64 * K];
    __shared__ float red[kWarps * kCols];
    __shared__ float gu[2][kCols];
    const int ei = blockIdx.y, g = blockIdx.x;
    const unsigned long long ptr = a.plan_ptr[ei];
    if (ptr == 0ull) return;
    const uint8_t* blob = (const uint8_t*) ptr;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int r = 0; r < 2; ++r) rotate_in(a.x, nullptr, (const uint16_t*) (blob + a.suh[r]), 0, a.n_embd, xs[r]);
    __syncthreads();
    for (int r = 0; r < 2; ++r) {
        block_gemv<K, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd / 16, xs[r], stage, red);
        if (warp == 0) {
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
            had128(v, lane);
            const uint16_t* svh = (const uint16_t*) (blob + a.svh[r]) + g * kCols;
#pragma unroll
            for (int i = 0; i < 4; ++i) gu[r][lane * 4 + i] = v[i] * h2f(svh[lane * 4 + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x < kCols) {
        float gv = gu[0][threadIdx.x], uv = gu[1][threadIdx.x];
        if (a.limit > 0.0f) {
            gv = fminf(gv, a.limit);
            uv = fminf(fmaxf(uv, -a.limit), a.limit);
        }
        a.h[(size_t) ei * a.n_ff + g * kCols + threadIdx.x] = (gv / (1.0f + expf(-gv))) * uv;
    }
}

// grid (n_embd / 128, k): expert blockIdx.y's down rows for 128 outputs; the last block of each output block adds
// the experts up in plan order (as the engine's moe_down), then the CPU lane's part and the shared expert
template <int K, int cb>
__global__ void __launch_bounds__(kThreads) moe_down_kernel(const KMoe a) {
    __shared__ __half xs[kMaxKR];
    __shared__ uint32_t stage[kWarps * 64 * K];
    __shared__ float red[kWarps * kCols];
    __shared__ int last;
    const int ei = blockIdx.y, g = blockIdx.x;
    const unsigned long long ptr = a.plan_ptr[ei];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* part = a.part + (size_t) ei * a.n_embd + g * kCols;
    if (ptr != 0ull) {
        const uint8_t* blob = (const uint8_t*) ptr;
        rotate_in(a.h + (size_t) ei * a.n_ff, nullptr, (const uint16_t*) (blob + a.suh[2]), 0, a.n_ff, xs);
        __syncthreads();
        block_gemv<K, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff / 16, xs, stage, red);
        if (warp == 0) {
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
            had128(v, lane);
            const uint16_t* svh = (const uint16_t*) (blob + a.svh[2]) + g * kCols;
#pragma unroll
            for (int i = 0; i < 4; ++i) part[lane * 4 + i] = v[i] * h2f(svh[lane * 4 + i]);
        }
    } else if (threadIdx.x < kCols) {
        part[threadIdx.x] = 0.0f;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&a.cnt[g], 1u) == (unsigned) (a.k - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < kCols) {
        const int r = g * kCols + threadIdx.x;
        float m = 0.0f;
        for (int i = 0; i < a.k; ++i) m += a.plan_w[i] * *(volatile const float*) &a.part[(size_t) i * a.n_embd + r];
        if (a.cpu_flag != nullptr && *a.cpu_flag) m += a.cpu_part[r];
        a.out[r] = m + (a.sh_out != nullptr ? a.sh_out[r] : 0.0f);
    }
    if (threadIdx.x == 0) a.cnt[g] = 0u;
}

// ---------------------------------------------------------------- reconstruction
// grid (n / 128, k / 128): the 128 x 128 block of W = diag(suh) H W' H diag(svh), written transposed (dst[c][r])
template <int K, int cb>
__global__ void __launch_bounds__(kThreads) recon_rows_kernel(const uint32_t* __restrict__ tr, int ld,
                                                              const uint16_t* __restrict__ suh,
                                                              const uint16_t* __restrict__ svh, int k, uint16_t* dst,
                                                              bool bf16) {
    constexpr int W = 8 * K;
    constexpr int P = kCols + 2;   // padded row (halves)
    __shared__ __half tile[kCols * P];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nb = blockIdx.x, kb = blockIdx.y;
    // decode: warp w takes the 16-row slices w (8 tiles each)
    for (int ks = warp; ks < 8; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) (kb * 8 + ks) * ld + nb * 8) * W;
        for (int t = 0; t < 8; ++t) {
            float v[8];
            decode8<K, cb>(src + t * W, lane, v);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = ks * 16 + (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
                const int c = t * 16 + (lane >> 2) + 8 * (j >> 2);
                tile[r * P + c] = __float2half_rn(v[j]);
            }
        }
    }
    __syncthreads();
    // H along the outputs (each row of 128 is one block)
    for (int r = warp; r < kCols; r += kWarps) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = __half2float(tile[r * P + lane * 4 + i]);
        had128(v, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) tile[r * P + lane * 4 + i] = __float2half_rn(v[i]);
    }
    __syncthreads();
    // H along the inputs, the scales, the transposed store
    for (int c = warp; c < kCols; c += kWarps) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = __half2float(tile[(lane * 4 + i) * P + c]);
        had128(v, lane);
        const float sv = h2f(svh[nb * kCols + c]);
        uint16_t* d = dst + (size_t) (nb * kCols + c) * k + kb * kCols + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float w = v[i] * sv * h2f(suh[kb * kCols + lane * 4 + i]);
            if (bf16) {
                const uint32_t u = __float_as_uint(w);
                d[i] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);   // round to nearest even
            } else {
                d[i] = __half_as_ushort(__float2half_rn(w));
            }
        }
    }
}

// grid (n / 128, k / 16): W' row-major, one tile per warp
template <int K, int cb>
__global__ void __launch_bounds__(kThreads) recon_inner_kernel(const uint32_t* __restrict__ tr, int ld, int n,
                                                               uint16_t* dst) {
    constexpr int W = 8 * K;
    const int lane = threadIdx.x & 31, t = threadIdx.x >> 5;
    const int tc = blockIdx.x * 8 + t, ks = blockIdx.y;
    float v[8];
    decode8<K, cb>(tr + ((size_t) ks * ld + tc) * W, lane, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = ks * 16 + (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
        const int c = tc * 16 + (lane >> 2) + 8 * (j >> 2);
        dst[(size_t) r * n + c] = __half_as_ushort(__float2half_rn(v[j]));
    }
}

__global__ void rows_f16_kernel(const float* __restrict__ x, int ldx, const int* __restrict__ idx, int ncols,
                                uint16_t* __restrict__ out) {
    const int r = blockIdx.y;
    const float* src = x + (size_t) (idx != nullptr ? idx[r] : r) * ldx;
    uint16_t* dst = out + (size_t) r * ncols;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < ncols; c += gridDim.x * blockDim.x)
        dst[c] = __half_as_ushort(__float2half_rn(src[c]));
}

// ---------------------------------------------------------------- the prompt path's tensor-core GEMMs (sm_80+)
// A block: BM rows x 128 outputs (one output rotation block).  Per 128 inputs the block's rows are scaled by suh,
// rotated and stored FP16 in shared memory (the A operands); per 16-row slice each warp decodes its 16 x 16 tile
// straight into mma.m16n8k16's B fragments - a lane's 8 steps ARE the fragment order (steps 0,1 / 2,3 rows 2t+{0,1} /
// 2t+8+{0,1} of column g, steps 4..7 the same of column g + 8) - and multiplies every m-tile of the block with them.
// The output rotation and svh run on the F32 tile in shared memory.
#if !defined(STRATA_USE_HIP)
namespace tc {

constexpr int BM = 32, MT = BM / 16, AP = 136;   // rows a block, its m-tiles, the A rows' pitch (halves)

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }

// the lane's 4 B registers of tile t (words staged in st): b[0..1] columns 0..7, b[2..3] columns 8..15
template <int K, int cb>
__device__ __forceinline__ void tile_frags(const uint32_t* st, int lane, uint32_t* b) {
    const int p0 = lane * 8;
#pragma unroll
    for (int j = 0; j < 8; j += 2) b[j / 2] = h2u(decode2h<cb>(state_at<K>(st, p0 + j), state_at<K>(st, p0 + j + 1)));
}

// A[r][j] = FP16(H(x_r * suh)[j]), j < 128, for the block's rows (zero past nrows).  x_r: F32 row idx[r] of xf (ldx)
// when xf is given, else FP16 row r of xh (ldx).  Every warp, BM / 8 rows each.
__device__ __forceinline__ void load_a(const float* __restrict__ xf, const int* __restrict__ idx,
                                       const __half* __restrict__ xh, int ldx, int r0, int nrows, int k0,
                                       const uint16_t* __restrict__ suh, __half* A) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int rr = warp; rr < BM; rr += kWarps) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int r = r0 + rr;
        if (rr < nrows) {
            const int c = k0 + lane * 4;
            if (xf != nullptr) {
                const float* src = xf + (size_t) idx[r] * ldx + c;
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = src[i];
            } else {
                const __half* src = xh + (size_t) r * ldx + c;
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = __half2float(src[i]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= h2f(suh[c + i]);
            had128(v, lane);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) A[rr * AP + lane * 4 + i] = __float2half_rn(v[i]);
    }
}

// the m-tile's A registers for the 16-column slice s of A
__device__ __forceinline__ void a_frags(const __half* A, int mt, int s, int lane, uint32_t* a) {
    const int g = lane >> 2, t = lane & 3;
    const __half* p = A + (mt * 16 + g) * AP + s * 16 + 2 * t;
    a[0] = *reinterpret_cast<const uint32_t*>(p);
    a[1] = *reinterpret_cast<const uint32_t*>(p + 8 * AP);
    a[2] = *reinterpret_cast<const uint32_t*>(p + 8);
    a[3] = *reinterpret_cast<const uint32_t*>(p + 8 * AP + 8);
}

// acc (this warp's 16 columns of the block's BM rows) -> T[BM][128] F32
__device__ __forceinline__ void store_acc(const float (&acc)[MT][2][4], float* T) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int ns = 0; ns < 2; ++ns) {
            const int c = warp * 16 + ns * 8 + 2 * t, r = mt * 16 + g;
            T[r * 128 + c] = acc[mt][ns][0];
            T[r * 128 + c + 1] = acc[mt][ns][1];
            T[(r + 8) * 128 + c] = acc[mt][ns][2];
            T[(r + 8) * 128 + c + 1] = acc[mt][ns][3];
        }
}

struct GArgs {
    const uint8_t* wbase;     // expert e's blob at wbase + e * stride (dense: 0, the offsets are pointers)
    size_t stride;
    size_t tr[3], suh[3], svh[3];
    int ld[3];                // tiles per 16-row slice of each matrix
    const int* bounds;        // device: expert e's rows [bounds[e], bounds[e + 1]) (null: one matrix, rows [0, m))
    int row0, m;              // local row of a sorted row: bounds[e] - row0 + r
    int k, n;                 // of the matrix (gate/up: n_embd, n_ff)
    const float* xf;          // F32 input rows by index (idx[local row]) ...
    const int* idx;
    const __half* xh;         // ... or FP16 rows by local row
    int ldx;
    float* y;                 // F32 output rows (ldy), y = beta y + out ...
    int ldy;
    float beta;
    __half* h;                // ... or (gate/up) the swiglu as FP16 rows (ldh)
    int ldh;
    float limit;
};

// one matrix: y = beta y + H(H(x * suh) W') * svh.  grid (row blocks, n / 128, experts)
template <int K, int cb>
__global__ void __launch_bounds__(kThreads) gemm_kernel(const GArgs a) {
    constexpr int W = 8 * K;
    __shared__ __align__(16) unsigned char sm[BM * 128 * 4];   // A (BM x AP halves), then the F32 tile
    __shared__ uint32_t stage[kWarps][64];
    __half* A = reinterpret_cast<__half*>(sm);
    float* T = reinterpret_cast<float*>(sm);
    const int e = blockIdx.z, rb = blockIdx.x, g = blockIdx.y;
    const int e0 = a.bounds != nullptr ? a.bounds[e] - a.row0 : 0;
    const int ne = a.bounds != nullptr ? a.bounds[e + 1] - a.bounds[e] : a.m;
    const int r0 = e0 + rb * BM, nrows = min(BM, ne - rb * BM);
    if (nrows <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const uint32_t* tr = (const uint32_t*) (blob + a.tr[0]);
    const uint16_t* suh = (const uint16_t*) (blob + a.suh[0]);
    const uint16_t* svh = (const uint16_t*) (blob + a.svh[0]);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float acc[MT][2][4] = {};
    for (int k0 = 0; k0 < a.k; k0 += 128) {
        __syncthreads();
        load_a(a.xf, a.idx, a.xh, a.ldx, r0, nrows, k0, suh, A);
        __syncthreads();
#pragma unroll 1
        for (int s = 0; s < 8; ++s) {
            const int ks = k0 / 16 + s;
            const uint32_t* src = tr + ((size_t) ks * a.ld[0] + g * 8 + warp) * W;
            __syncwarp();
            for (int i = lane; i < W; i += 32) stage[warp][i] = src[i];
            __syncwarp();
            uint32_t b[4];
            tile_frags<K, cb>(stage[warp], lane, b);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                uint32_t af[4];
                a_frags(A, mt, s, lane, af);
                mma16816(acc[mt][0], af, b[0], b[1]);
                mma16816(acc[mt][1], af, b[2], b[3]);
            }
        }
    }
    __syncthreads();
    store_acc(acc, T);
    __syncthreads();
    for (int rr = warp; rr < nrows; rr += kWarps) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = T[rr * 128 + lane * 4 + i];
        had128(v, lane);
        float* yr = a.y + (size_t) (r0 + rr) * a.ldy + g * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = v[i] * h2f(svh[g * 128 + lane * 4 + i]);
            yr[i] = a.beta != 0.0f ? a.beta * yr[i] + o : o;
        }
    }
}

// gate and up (matrices 0 and 1) over the same rows, the clamped swiglu, FP16 h.  grid (row blocks, n / 128, experts)
template <int K, int cb>
__global__ void __launch_bounds__(kThreads) gate_up_kernel(const GArgs a) {
    constexpr int W = 8 * K;
    __shared__ __align__(16) unsigned char sm[2 * BM * 128 * 4];   // A_g, A_u, then the two F32 tiles
    __shared__ uint32_t stage[kWarps][64];
    __half* A[2] = {reinterpret_cast<__half*>(sm), reinterpret_cast<__half*>(sm) + BM * AP};
    float* T[2] = {reinterpret_cast<float*>(sm), reinterpret_cast<float*>(sm) + BM * 128};
    const int e = blockIdx.z, rb = blockIdx.x, g = blockIdx.y;
    const int e0 = a.bounds[e] - a.row0, ne = a.bounds[e + 1] - a.bounds[e];
    const int r0 = e0 + rb * BM, nrows = min(BM, ne - rb * BM);
    if (nrows <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float acc[2][MT][2][4] = {};
    for (int k0 = 0; k0 < a.k; k0 += 128) {
        __syncthreads();
        for (int m = 0; m < 2; ++m)
            load_a(a.xf, a.idx, a.xh, a.ldx, r0, nrows, k0, (const uint16_t*) (blob + a.suh[m]), A[m]);
        __syncthreads();
#pragma unroll 1
        for (int s = 0; s < 8; ++s) {
            const int ks = k0 / 16 + s;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const uint32_t* src = (const uint32_t*) (blob + a.tr[m]) + ((size_t) ks * a.ld[m] + g * 8 + warp) * W;
                __syncwarp();
                for (int i = lane; i < W; i += 32) stage[warp][i] = src[i];
                __syncwarp();
                uint32_t b[4];
                tile_frags<K, cb>(stage[warp], lane, b);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    uint32_t af[4];
                    a_frags(A[m], mt, s, lane, af);
                    mma16816(acc[m][mt][0], af, b[0], b[1]);
                    mma16816(acc[m][mt][1], af, b[2], b[3]);
                }
            }
        }
    }
    __syncthreads();
    store_acc(acc[0], T[0]);
    store_acc(acc[1], T[1]);
    __syncthreads();
    const uint16_t* svg = (const uint16_t*) (a.wbase + (size_t) e * a.stride + a.svh[0]) + g * 128;
    const uint16_t* svu = (const uint16_t*) (a.wbase + (size_t) e * a.stride + a.svh[1]) + g * 128;
    for (int rr = warp; rr < nrows; rr += kWarps) {
        float vg[4], vu[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            vg[i] = T[0][rr * 128 + lane * 4 + i];
            vu[i] = T[1][rr * 128 + lane * 4 + i];
        }
        had128(vg, lane);
        had128(vu, lane);
        __half* hr = a.h + (size_t) (r0 + rr) * a.ldh + g * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float gv = vg[i] * h2f(svg[lane * 4 + i]), uv = vu[i] * h2f(svu[lane * 4 + i]);
            if (a.limit > 0.0f) {
                gv = fminf(gv, a.limit);
                uv = fminf(fmaxf(uv, -a.limit), a.limit);
            }
            hr[i] = __float2half_rn((gv / (1.0f + expf(-gv))) * uv);
        }
    }
}

}  // namespace tc
#endif

// ---------------------------------------------------------------- dispatch
// F(K, cb) for K 1..8 and the three codebooks
template <template <int, int> class F, typename... A>
bool dispatch(int K, int cb, A&&... args) {
#define EXL3_CASE(KK)                                                    \
    case KK:                                                             \
        if (cb == 2) F<KK, 2>::run(std::forward<A>(args)...);            \
        else if (cb == 1) F<KK, 1>::run(std::forward<A>(args)...);       \
        else F<KK, 0>::run(std::forward<A>(args)...);                    \
        return true;
    switch (K) {
        EXL3_CASE(1) EXL3_CASE(2) EXL3_CASE(3) EXL3_CASE(4) EXL3_CASE(5) EXL3_CASE(6) EXL3_CASE(7) EXL3_CASE(8)
        default: return false;
    }
#undef EXL3_CASE
}

template <int K, int cb> struct RunMv {
    static void run(const KBatch& b, int blocks, cudaStream_t s) { mv_kernel<K, cb><<<blocks, kThreads, 0, s>>>(b); }
};
template <int K, int cb> struct RunGU {
    static void run(const KMoe& a, cudaStream_t s) {
        moe_gate_up_kernel<K, cb><<<dim3((unsigned) (a.n_ff / kCols), (unsigned) a.k), kThreads, 0, s>>>(a);
    }
};
template <int K, int cb> struct RunDown {
    static void run(const KMoe& a, cudaStream_t s) {
        moe_down_kernel<K, cb><<<dim3((unsigned) (a.n_embd / kCols), (unsigned) a.k), kThreads, 0, s>>>(a);
    }
};
template <int K, int cb> struct RunRows {
    static void run(const Mat& m, uint16_t* dst, bool bf16, cudaStream_t s) {
        recon_rows_kernel<K, cb><<<dim3((unsigned) (m.n / kCols), (unsigned) (m.k / kCols)), kThreads, 0, s>>>(
            m.trellis, m.ld > 0 ? m.ld : m.n / 16, m.suh, m.svh, m.k, dst, bf16);
    }
};
#if !defined(STRATA_USE_HIP)
template <int K, int cb> struct RunGemm {
    static void run(const tc::GArgs& a, dim3 grid, cudaStream_t s) { tc::gemm_kernel<K, cb><<<grid, kThreads, 0, s>>>(a); }
};
template <int K, int cb> struct RunGateUp {
    static void run(const tc::GArgs& a, dim3 grid, cudaStream_t s) {
        tc::gate_up_kernel<K, cb><<<grid, kThreads, 0, s>>>(a);
    }
};
#endif
template <int K, int cb> struct RunInner {
    static void run(const Mat& m, uint16_t* dst, cudaStream_t s) {
        recon_inner_kernel<K, cb><<<dim3((unsigned) (m.n / kCols), (unsigned) (m.k / 16)), kThreads, 0, s>>>(
            m.trellis, m.ld > 0 ? m.ld : m.n / 16, m.n, dst);
    }
};

// ---------------------------------------------------------------- per-stream scratch
constexpr size_t kPartFloats = (size_t) 2 << 20;   // split partials: 8 MB
constexpr size_t kCounters = 16384;
constexpr size_t kHFloats = (size_t) 8 * 8192;     // the experts' h
struct Scratch {
    float* part = nullptr;
    unsigned* cnt = nullptr;
    float* h = nullptr;
};
std::mutex g_mu;
std::map<std::pair<int, cudaStream_t>, Scratch> g_scratch;

Scratch* scratch(cudaStream_t s) {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_scratch.find({dev, s});
    if (it != g_scratch.end()) return &it->second;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "exl3: scratch for a stream first seen inside a graph capture (call exl3::prepare)\n");
        return nullptr;
    }
    Scratch sc;
    if (cudaMalloc(&sc.part, kPartFloats * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&sc.cnt, kCounters * sizeof(unsigned)) != cudaSuccess ||
        cudaMalloc(&sc.h, kHFloats * sizeof(float)) != cudaSuccess ||
        cudaMemsetAsync(sc.cnt, 0, kCounters * sizeof(unsigned), s) != cudaSuccess) {
        std::fprintf(stderr, "exl3: scratch did not allocate\n");
        return nullptr;
    }
    cudaStreamSynchronize(s);
    return &(g_scratch[{dev, s}] = sc);
}

int sm_count() {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return std::max(1, sms);
}

}  // namespace

Mat view(const Mat& m, int c0, int nc) {
    Mat v = m;
    v.ld = m.ld > 0 ? m.ld : m.n / 16;
    v.trellis = m.trellis + (size_t) (c0 / 16) * 8 * m.K;
    v.svh = m.svh + c0;
    v.n = nc;
    return v;
}

bool supported(const Mat& m) {
    return m.trellis != nullptr && m.suh != nullptr && m.svh != nullptr && m.k > 0 && m.n > 0 && m.k % 128 == 0 &&
           m.n % 128 == 0 && m.K >= 1 && m.K <= 8 && m.cb >= 0 && m.cb <= 2;
}

int launch_errors() { return g_launch_errors.load(std::memory_order_relaxed); }

bool prepare(cudaStream_t s) { return scratch(s) != nullptr; }

bool mv(const MvJob* jobs, int n, cudaStream_t s) {
    if (n <= 0 || n > kMaxMvJobs) return false;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    const int sms = sm_count();
    bool done[kMaxMvJobs] = {};
    size_t part_at = 0, cnt_at = 0;
    for (int i = 0; i < n; ++i) {
        if (done[i]) continue;
        const Mat& m0 = *jobs[i].m;
        KBatch b{};
        int blocks = 0, nb = 0;
        for (int j = i; j < n; ++j) {
            const Mat& m = *jobs[j].m;
            if (done[j] || m.K != m0.K || m.cb != m0.cb) continue;
            if (!supported(m) || jobs[j].y == nullptr || (jobs[j].xf == nullptr && jobs[j].xq == nullptr)) {
                std::fprintf(stderr, "exl3 mv: job %d unsupported (k %d n %d K %d cb %d)\n", j, m.k, m.n, m.K, m.cb);
                return false;
            }
            done[j] = true;
            KJob& J = b.j[nb];
            J.tr = m.trellis;
            J.suh = m.suh;
            J.svh = m.svh;
            J.xf = jobs[j].xf;
            J.xq = (const Q81*) jobs[j].xq;
            J.y = jobs[j].y;
            J.bias = jobs[j].bias;
            J.alpha = jobs[j].alpha;
            J.k = m.k;
            J.n = m.n;
            J.ld = m.ld > 0 ? m.ld : m.n / 16;
            // the input split: at most kMaxKR inputs a block, and about two blocks per SM over the batch
            const int groups = m.n / kCols;
            int S = (m.k + kMaxKR - 1) / kMaxKR;
            while (groups * S < 2 * sms && m.k / (S * 2) >= 256) S *= 2;
            int kr = ((m.k + S - 1) / S + 127) / 128 * 128;
            S = (m.k + kr - 1) / kr;
            if (S > 1 && (part_at + (size_t) S * m.n > kPartFloats || cnt_at + (size_t) groups > kCounters)) {
                std::fprintf(stderr, "exl3 mv: scratch too small for k %d n %d\n", m.k, m.n);
                return false;
            }
            J.S = S;
            J.kr = kr;
            J.part = sc->part + part_at;
            J.cnt = sc->cnt + cnt_at;
            if (S > 1) {
                part_at += (size_t) S * m.n;
                cnt_at += (size_t) groups;
            }
            blocks += groups * S;
            b.blk_end[nb++] = blocks;
        }
        b.n = nb;
        if (!dispatch<RunMv>(m0.K, m0.cb, b, blocks, s)) return false;
    }
    launch_check("mv");
    return true;
}

static bool moe_args(const MoeArgs& a, KMoe& m, cudaStream_t s) {
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    if (a.k < 1 || a.k > 8 || a.n_embd % kCols != 0 || a.n_ff % kCols != 0 || a.n_embd > kMaxKR ||
        a.n_ff > kMaxKR || (size_t) a.k * a.n_ff > kHFloats || (size_t) a.k * a.n_embd > kPartFloats ||
        (size_t) (a.n_embd / kCols) > kCounters) {
        std::fprintf(stderr, "exl3 moe: geometry not covered (k %d n_embd %d n_ff %d)\n", a.k, a.n_embd, a.n_ff);
        return false;
    }
    m = KMoe{};
    m.plan_ptr = a.plan_ptr;
    m.plan_w = a.plan_w;
    m.cpu_part = a.cpu_part;
    m.cpu_flag = a.cpu_flag;
    m.h = sc->h;
    m.part = sc->part;
    m.cnt = sc->cnt;
    for (int r = 0; r < 3; ++r) {
        m.suh[r] = a.lay.suh[r];
        m.svh[r] = a.lay.svh[r];
        m.tr[r] = a.lay.trellis[r];
    }
    m.k = a.k;
    m.n_embd = a.n_embd;
    m.n_ff = a.n_ff;
    m.limit = a.limit;
    return true;
}

void moe_gate_up(const MoeArgs& a, const float* x, float* h, cudaStream_t s) {
    KMoe m;
    if (!moe_args(a, m, s)) return;
    if (a.lay.K[0] != a.lay.K[1]) {
        std::fprintf(stderr, "exl3 moe_gate_up: gate K %d != up K %d\n", a.lay.K[0], a.lay.K[1]);
        return;
    }
    m.x = x;
    if (h != nullptr) m.h = h;
    if (!dispatch<RunGU>(a.lay.K[0], a.lay.cb, m, s)) std::fprintf(stderr, "exl3 moe_gate_up: K %d\n", a.lay.K[0]);
    launch_check("moe_gate_up");
}

void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    KMoe m;
    if (!moe_args(a, m, s)) return;
    if (h != nullptr) m.h = const_cast<float*>(h);
    m.sh_out = sh_out;
    m.out = out;
    if (!dispatch<RunDown>(a.lay.K[2], a.lay.cb, m, s)) std::fprintf(stderr, "exl3 moe_down: K %d\n", a.lay.K[2]);
    launch_check("moe_down");
}

bool mma_supported() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major >= 8;
#endif
}

void set_gate_up(const SetArgs& sa, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s) {
#if !defined(STRATA_USE_HIP)
    if (sa.lay.K[0] != sa.lay.K[1]) {
        std::fprintf(stderr, "exl3 set_gate_up: gate K %d != up K %d\n", sa.lay.K[0], sa.lay.K[1]);
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    tc::GArgs a{};
    a.wbase = sa.wbase;
    a.stride = sa.stride;
    for (int r = 0; r < 2; ++r) {
        a.tr[r] = sa.lay.trellis[r];
        a.suh[r] = sa.lay.suh[r];
        a.svh[r] = sa.lay.svh[r];
        a.ld[r] = sa.n_ff / 16;
    }
    a.bounds = sa.bounds;
    a.row0 = sa.row0;
    a.k = sa.n_embd;
    a.n = sa.n_ff;
    a.xf = x;
    a.idx = row_idx;
    a.ldx = ldx;
    a.h = (__half*) h16;
    a.ldh = ldh;
    a.limit = sa.limit;
    const dim3 grid((unsigned) ((sa.max_rows + tc::BM - 1) / tc::BM), (unsigned) (sa.n_ff / kCols), (unsigned) sa.n_exp);
    if (!dispatch<RunGateUp>(sa.lay.K[0], sa.lay.cb, a, grid, s)) std::fprintf(stderr, "exl3 set_gate_up: K\n");
    launch_check("set_gate_up");
#else
    (void) sa; (void) x; (void) ldx; (void) row_idx; (void) h16; (void) ldh; (void) s;
#endif
}

void set_down(const SetArgs& sa, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s) {
#if !defined(STRATA_USE_HIP)
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    tc::GArgs a{};
    a.wbase = sa.wbase;
    a.stride = sa.stride;
    a.tr[0] = sa.lay.trellis[2];
    a.suh[0] = sa.lay.suh[2];
    a.svh[0] = sa.lay.svh[2];
    a.ld[0] = sa.n_embd / 16;
    a.bounds = sa.bounds;
    a.row0 = sa.row0;
    a.k = sa.n_ff;
    a.n = sa.n_embd;
    a.xh = (const __half*) h16;
    a.ldx = ldh;
    a.y = y;
    a.ldy = ldy;
    const dim3 grid((unsigned) ((sa.max_rows + tc::BM - 1) / tc::BM), (unsigned) (sa.n_embd / kCols), (unsigned) sa.n_exp);
    if (!dispatch<RunGemm>(sa.lay.K[2], sa.lay.cb, a, grid, s)) std::fprintf(stderr, "exl3 set_down: K\n");
    launch_check("set_down");
#else
    (void) sa; (void) h16; (void) ldh; (void) y; (void) ldy; (void) s;
#endif
}

void gemm_rows(const Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s) {
#if !defined(STRATA_USE_HIP)
    if (rows <= 0) return;
    tc::GArgs a{};
    a.wbase = nullptr;
    a.tr[0] = (size_t) m.trellis;
    a.suh[0] = (size_t) m.suh;
    a.svh[0] = (size_t) m.svh;
    a.ld[0] = m.ld > 0 ? m.ld : m.n / 16;
    a.m = rows;
    a.k = m.k;
    a.n = m.n;
    a.xh = (const __half*) x16;
    a.ldx = ldx;
    a.y = y;
    a.ldy = ldy;
    a.beta = beta;
    const dim3 grid((unsigned) ((rows + tc::BM - 1) / tc::BM), (unsigned) (m.n / kCols), 1);
    if (!supported(m) || !dispatch<RunGemm>(m.K, m.cb, a, grid, s)) {
        std::fprintf(stderr, "exl3 gemm_rows: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("gemm_rows");
#else
    (void) m; (void) x16; (void) ldx; (void) rows; (void) y; (void) ldy; (void) beta; (void) s;
#endif
}

void reconstruct_rows(const Mat& m, uint16_t* dst, bool bf16, cudaStream_t s) {
    if (!supported(m) || !dispatch<RunRows>(m.K, m.cb, m, dst, bf16, s)) {
        std::fprintf(stderr, "exl3 reconstruct_rows: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("reconstruct_rows");
}

void rows_f16(const float* x, int ldx, const int* idx, int nrows, int ncols, uint16_t* out, cudaStream_t s) {
    if (nrows <= 0) return;
    rows_f16_kernel<<<dim3((unsigned) std::min(8, (ncols + 255) / 256), (unsigned) nrows), 256, 0, s>>>(x, ldx, idx,
                                                                                                         ncols, out);
    launch_check("rows_f16");
}

void reconstruct_inner(const Mat& m, uint16_t* dst, cudaStream_t s) {
    if (!supported(m) || !dispatch<RunInner>(m.K, m.cb, m, dst, s)) {
        std::fprintf(stderr, "exl3 reconstruct_inner: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("reconstruct_inner");
}

}  // namespace strata::kernels::exl3
