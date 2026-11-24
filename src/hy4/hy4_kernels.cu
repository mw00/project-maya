// src/hy4/hy4_kernels.cu - maya-hy4's own kernels (hy4_kernels.cuh).  Correctness first: plain F32 arithmetic on the
// CUDA cores, the latent cache in F16 as the reference keeps it; the reference is AngelSlim's hy4-flash graph
// (src/models/hy-v4-flash.cpp of their hy4-flash patch).
#include "hy4_kernels.cuh"
#include "maya/kernels/fresh_load.hpp"
#include "maya/kernels/lane_xor.hpp"   // warp sums in registers on RDNA
#include "maya/kernels/rdna_wmma.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <map>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace hy4::k {

namespace {

std::atomic<int> g_err{0};
float* g_mix_out = nullptr;   // (tests: hc_pre's mixes, set_mix_out)
// the first failed launch's kernel and error (first_fault), and how many were printed: a fault in a kernel is sticky -
// every launch after it fails the same way, so past the first few they are counted, not printed (hundreds of lines
// hid the first one)
std::mutex g_fault_mu;
std::string g_fault;
int g_printed = 0;
constexpr int kPrintMax = 8;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        bool print = true;
        {
            std::lock_guard<std::mutex> lk(g_fault_mu);
            if (g_fault.empty()) g_fault = what;
            print = g_printed < kPrintMax;
            ++g_printed;
            if (g_printed == kPrintMax + 1)
                std::fprintf(stderr, "maya-hy4: more kernels failed the same way - counted, not printed\n");
        }
        if (print) std::fprintf(stderr, "maya-hy4 kernel %s: %s\n", what, cudaGetErrorString(e));
        g_err.fetch_add(1, std::memory_order_relaxed);
    }
}

// the shuffle's xor order 16, 8, 4, 2, 1; maya_lane_xor moves the same values in registers on RDNA (the same bits)
__device__ __forceinline__ float warp_sum(float v) {
    v += maya_lane_xor<16>(v);
    v += maya_lane_xor<8>(v);
    v += maya_lane_xor<4>(v);
    v += maya_lane_xor<2>(v);
    v += maya_lane_xor<1>(v);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    v = fmaxf(v, maya_lane_xor<16>(v));
    v = fmaxf(v, maya_lane_xor<8>(v));
    v = fmaxf(v, maya_lane_xor<4>(v));
    v = fmaxf(v, maya_lane_xor<2>(v));
    v = fmaxf(v, maya_lane_xor<1>(v));
    return v;
}
// a block's sum (blockDim.x a multiple of 32, at most 1024); every thread gets it
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    __syncthreads();
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = l < nw ? sh[l] : 0.0f;
    r = warp_sum(r);
    return r;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

// ---------------------------------------------------------------- int8 rows
__global__ void q8_kernel(const float* __restrict__ x, size_t ldx, int k, int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = blockIdx.y, b = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const float v = x[(size_t) r * ldx + (size_t) b * 32 + lane];
    const float am = warp_max(fabsf(v));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[(size_t) r * (k / 32) + b] = sc;
}

// A kernel's own values as the next GEMV's int8 rows too (Q8Out): a warp's 32 lanes hold 32 consecutive values of a row
// - block at / 32 of q8_kernel's layout - and make its code and scale, q8_kernel's arithmetic on the values written.
// Every lane of the warp calls it (the row a multiple of 32 values).
__device__ __forceinline__ void q8_put(float v, size_t at, const Q8Out& o) {
    const float am = warp_max(fabsf(v));
    const float sc = am / 127.0f;
    const int8_t c = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    o.q[at] = c;
    if (o.q2 != nullptr) o.q2[at] = c;
    if ((threadIdx.x & 31) == 0) {
        o.d[at / 32] = sc;
        if (o.d2 != nullptr) o.d2[at / 32] = sc;
    }
}
// q8_kernel's rows, each written twice - rows 2 r and 2 r + 1 of q2 / d2 (q8_rows then dup_rows_q8, in one launch)
__global__ void q8_dup_kernel(const float* __restrict__ x, size_t ldx, int k, int8_t* __restrict__ q2,
                              float* __restrict__ d2) {
    const int r = blockIdx.y, b = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const float v = x[(size_t) r * ldx + (size_t) b * 32 + lane];
    const float am = warp_max(fabsf(v));
    const float sc = am / 127.0f;
    const int8_t c = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    q2[(size_t) (2 * r) * k + (size_t) b * 32 + lane] = c;
    q2[(size_t) (2 * r + 1) * k + (size_t) b * 32 + lane] = c;
    if (lane == 0) {
        d2[(size_t) (2 * r) * (k / 32) + b] = sc;
        d2[(size_t) (2 * r + 1) * (k / 32) + b] = sc;
    }
}

__global__ void q8_heads_kernel(const float* __restrict__ x, size_t ldt, size_t ldh, int T, int k,
                                int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = blockIdx.y, h = r / T, t = r % T, b = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const float v = x[(size_t) t * ldt + (size_t) h * ldh + (size_t) b * 32 + lane];
    const float am = warp_max(fabsf(v));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[(size_t) r * (k / 32) + b] = sc;
}
__global__ void copy_cols_kernel(const float* __restrict__ src, size_t lds, float* __restrict__ dst, size_t ldd, int n,
                                 size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / n, c = i % n;
    dst[r * ldd + c] = src[r * lds + c];
}
template <typename D>
__global__ void store_rows_kernel(const float* __restrict__ src, int n, D* __restrict__ ring, int R,
                                  const int* __restrict__ pos) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        ring[(size_t) (pos[t] % R) * n + i] = (D) src[(size_t) t * n + i];
}

// ---------------------------------------------------------------- F32 / BF16 GEMV: a warp a row, up to 8 columns a pass
template <typename W, int NC>
__global__ void gemv_kernel(const W* __restrict__ w, size_t ld, int k, int n, const float* __restrict__ x, size_t ldx,
                            int ncol, float* __restrict__ y, size_t ldy, float alpha) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int c0 = blockIdx.y * NC;
    if (row >= n) return;
    const W* wr = w + (size_t) row * ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int i = lane * 2; i < k; i += 64) {
        float w0, w1;
        if constexpr (sizeof(W) == 2) {
            const uint32_t p = *(const uint32_t*) (wr + i);
            w0 = __uint_as_float(p << 16);
            w1 = __uint_as_float(p & 0xffff0000u);
        } else {
            const float2 p = *(const float2*) (wr + i);
            w0 = p.x;
            w1 = p.y;
        }
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c0 + c < ncol) {
                const float2 xv = *(const float2*) (x + (size_t) (c0 + c) * ldx + i);
                acc[c] = fmaf(w0, xv.x, fmaf(w1, xv.y, acc[c]));
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0 && c0 + c < ncol) y[(size_t) (c0 + c) * ldy + row] = alpha * v;
    }
}
template <typename W>
__global__ void __launch_bounds__(256) gemv_rows_kernel(const W* __restrict__ w, size_t ld, int k,
                                                        const float* __restrict__ x, size_t ldx, int ncol,
                                                        float* __restrict__ y, size_t ldy, float alpha) {
    const int row = blockIdx.x, c = blockIdx.y;
    const W* wr = w + (size_t) row * ld;
    const float* xc = x + (size_t) c * ldx;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float wv;
        if constexpr (sizeof(W) == 2) wv = __uint_as_float((uint32_t) wr[i] << 16);
        else wv = (float) wr[i];
        acc = fmaf(wv, xc[i], acc);
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0) y[(size_t) c * ldy + row] = alpha * acc;
}
// (a row's 4 weights at k index i: W F32 or BF16)
template <typename W>
__device__ __forceinline__ float4 gemv4_w(const W* __restrict__ wr, int i) {
    float4 wv;
    if constexpr (sizeof(W) == 2) {
        const uint2 p = *(const uint2*) (wr + i);
        wv.x = __uint_as_float(p.x << 16);
        wv.y = __uint_as_float(p.x & 0xffff0000u);
        wv.z = __uint_as_float(p.y << 16);
        wv.w = __uint_as_float(p.y & 0xffff0000u);
    } else {
        wv = *(const float4*) (wr + i);
    }
    return wv;
}
// AMD, a window's columns: gemv4_kernel's sums with x's columns staged in shared memory 1024 values at a time - the
// block's 8 rows read x there, where every row re-read it from the caches (a verify window's 4 columns put the router's
// F32 rows at a quarter of the memory's speed on a Radeon; one column gains nothing from it).  Each lane's steps, their
// order and the warp's sum are gemv4_kernel's: the same bits.
template <typename W, int NC>
__global__ void __launch_bounds__(256) gemv4s_kernel(const W* __restrict__ w, size_t ld, int k, int n,
                                                     const float* __restrict__ x, size_t ldx, int ncol,
                                                     float* __restrict__ y, size_t ldy, float alpha) {
    constexpr int KC = 1024;
    __shared__ float4 xs[NC][KC / 4];
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int c0 = blockIdx.y * NC;
    const bool live = row < n;
    const W* wr = w + (size_t) (live ? row : 0) * ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int k0 = 0; k0 < k; k0 += KC) {
        const int kq = min(KC, k - k0) / 4;   // (k a multiple of 128)
        __syncthreads();
        for (int idx = (int) threadIdx.x; idx < NC * kq; idx += (int) blockDim.x) {
            const int c = idx / kq, q = idx - c * kq;
            xs[c][q] = c0 + c < ncol ? *(const float4*) (x + (size_t) (c0 + c) * ldx + k0 + 4 * q)
                                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();
        if (!live) continue;
#pragma unroll 4
        for (int i = lane * 4; i < 4 * kq; i += 128) {
            const float4 wv = gemv4_w(wr, k0 + i);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c0 + c < ncol) {
                    const float4 xv = xs[c][i / 4];
                    acc[c] = fmaf(wv.w, xv.w, fmaf(wv.z, xv.z, fmaf(wv.y, xv.y, fmaf(wv.x, xv.x, acc[c]))));
                }
        }
    }
    if (!live) return;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0 && c0 + c < ncol) y[(size_t) (c0 + c) * ldy + row] = alpha * v;
    }
}
// ... 16-byte loads (4 values a lane a step, rows and columns 16-byte aligned, k a multiple of 128): a column's sums in
// the same order whatever NC, so a window's rows are a token's bits
template <typename W, int NC>
__global__ void gemv4_kernel(const W* __restrict__ w, size_t ld, int k, int n, const float* __restrict__ x, size_t ldx,
                             int ncol, float* __restrict__ y, size_t ldy, float alpha) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int c0 = blockIdx.y * NC;
    if (row >= n) return;
    const W* wr = w + (size_t) row * ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int i = lane * 4; i < k; i += 128) {
        float4 wv;
        if constexpr (sizeof(W) == 2) {
            const uint2 p = *(const uint2*) (wr + i);
            wv.x = __uint_as_float(p.x << 16);
            wv.y = __uint_as_float(p.x & 0xffff0000u);
            wv.z = __uint_as_float(p.y << 16);
            wv.w = __uint_as_float(p.y & 0xffff0000u);
        } else {
            wv = *(const float4*) (wr + i);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c0 + c < ncol) {
                const float4 xv = *(const float4*) (x + (size_t) (c0 + c) * ldx + i);
                acc[c] = fmaf(wv.w, xv.w, fmaf(wv.z, xv.z, fmaf(wv.y, xv.y, fmaf(wv.x, xv.x, acc[c]))));
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0 && c0 + c < ncol) y[(size_t) (c0 + c) * ldy + row] = alpha * v;
    }
}
// gemv4_kernel with a lane's loads asked for AH steps at a time: a lane's 24 steps of a row of 3072 each waited for
// its own weight load and its columns' x loads before the next step's were asked for - a row's time was those round
// trips, one after another.  Here the AH steps' weights and x are asked for together, then their products made in
// gemv4_kernel's order (a column's chain x, y, z, w of step after step), and the columns' sums over the lanes go level
// by level together - each column's own additions.  The same bits.
template <typename W, int NC, int AH>
__global__ void gemv4a_kernel(const W* __restrict__ w, size_t ld, int k, int n, const float* __restrict__ x, size_t ldx,
                              int ncol, float* __restrict__ y, size_t ldy, float alpha) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int c0 = blockIdx.y * NC;
    if (row >= n) return;
    const W* wr = w + (size_t) row * ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int i0 = lane * 4; i0 < k; i0 += 128 * AH) {
        float4 wv[AH], xs[AH][NC];
#pragma unroll
        for (int a = 0; a < AH; ++a) {
            const int i = i0 + 128 * a;
            if constexpr (sizeof(W) == 2) {
                const uint2 p = i < k ? *(const uint2*) (wr + i) : make_uint2(0u, 0u);
                wv[a].x = __uint_as_float(p.x << 16);
                wv[a].y = __uint_as_float(p.x & 0xffff0000u);
                wv[a].z = __uint_as_float(p.y << 16);
                wv[a].w = __uint_as_float(p.y & 0xffff0000u);
            } else {
                wv[a] = i < k ? *(const float4*) (wr + i) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
#pragma unroll
        for (int a = 0; a < AH; ++a)
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int i = i0 + 128 * a;
                const bool in = i < k && c0 + c < ncol;
                xs[a][c] = in ? *(const float4*) (x + (size_t) (c0 + c) * ldx + i) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
#pragma unroll
        for (int a = 0; a < AH; ++a)
            if (i0 + 128 * a < k) {
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    if (c0 + c < ncol) acc[c] = fmaf(wv[a].x, xs[a][c].x, acc[c]);
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    if (c0 + c < ncol) acc[c] = fmaf(wv[a].y, xs[a][c].y, acc[c]);
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    if (c0 + c < ncol) acc[c] = fmaf(wv[a].z, xs[a][c].z, acc[c]);
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    if (c0 + c < ncol) acc[c] = fmaf(wv[a].w, xs[a][c].w, acc[c]);
            }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float t[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) t[c] = __shfl_xor_sync(0xffffffffu, acc[c], o);
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[c] += t[c];
    }
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c0 + c < ncol) y[(size_t) (c0 + c) * ldy + row] = alpha * acc[c];
    }
}
template <typename W>
void gemv(const W* w, size_t ld, int k, int n, const float* x, size_t ldx, int ncol, float* y, size_t ldy, float alpha,
          cudaStream_t s) {
    if (ncol <= 0 || n <= 0) return;
    static const bool old = std::getenv("HY4_GEMV_OLD") != nullptr;
    const bool v4 = !old && n >= 128 && k % 128 == 0 && ld % 8 == 0 && ldx % 4 == 0 && ((uintptr_t) w & 15) == 0 &&
                    ((uintptr_t) x & 15) == 0;
#if defined(MAYA_USE_HIP)
    if (v4 && ncol > 1) {   // (AMD, a window's columns: x staged in shared memory - the same sums; one column: below)
        const dim3 b(256), g1((unsigned) ((n + 7) / 8), 1);
        if (ncol <= 4) gemv4s_kernel<W, 4><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
        else
            gemv4s_kernel<W, 8><<<dim3((unsigned) ((n + 7) / 8), (unsigned) ((ncol + 7) / 8)), b, 0, s>>>(
                w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
        check("gemv4s");
        return;
    }
#endif
    if (v4) {
        const dim3 b(256), g1((unsigned) ((n + 7) / 8), 1);
        // (5 and 6 columns: their own instances - a window of 5 tokens paid for the 8-column one's code; a column's
        //  sums are the same lanes' in the same order whatever the instance.  HY4_GEMV56=0: as before)
        static const bool nc56 = !(std::getenv("HY4_GEMV56") && std::getenv("HY4_GEMV56")[0] == '0');
#if !defined(MAYA_USE_HIP)
        // (gemv4a_kernel: 4 steps' loads together up to 4 columns, 3 with more - the registers; HY4_GEMV_AHEAD=0: as before)
        static const bool ahead = !(std::getenv("HY4_GEMV_AHEAD") && std::getenv("HY4_GEMV_AHEAD")[0] == '0');
        if (ahead && nc56) {
            if (ncol == 1) gemv4a_kernel<W, 1, 4><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
            else if (ncol == 2) gemv4a_kernel<W, 2, 4><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
            else if (ncol <= 4) gemv4a_kernel<W, 4, 4><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
            else if (ncol == 5) gemv4a_kernel<W, 5, 3><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
            else if (ncol == 6) gemv4a_kernel<W, 6, 3><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
            else
                gemv4a_kernel<W, 8, 3><<<dim3((unsigned) ((n + 7) / 8), (unsigned) ((ncol + 7) / 8)), b, 0, s>>>(
                    w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
            check("gemv4a");
            return;
        }
#endif
        if (ncol == 1) gemv4_kernel<W, 1><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
        else if (ncol <= 4) gemv4_kernel<W, 4><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
        else if (ncol == 5 && nc56) gemv4_kernel<W, 5><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
        else if (ncol == 6 && nc56) gemv4_kernel<W, 6><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
        else
            gemv4_kernel<W, 8><<<dim3((unsigned) ((n + 7) / 8), (unsigned) ((ncol + 7) / 8)), b, 0, s>>>(
                w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
        check("gemv4");
        return;
    }
    if (n < 128) {   // (few rows: a block each, every one of its warps on a part of k)
        gemv_rows_kernel<W><<<dim3((unsigned) n, (unsigned) ncol), 256, 0, s>>>(w, ld, k, x, ldx, ncol, y, ldy, alpha);
        check("gemv_rows");
        return;
    }
    const dim3 b(256);
    if (ncol == 1) {
        gemv_kernel<W, 1><<<dim3((unsigned) ((n + 7) / 8), 1), b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
    } else {
        gemv_kernel<W, 8><<<dim3((unsigned) ((n + 7) / 8), (unsigned) ((ncol + 7) / 8)), b, 0, s>>>(w, ld, k, n, x, ldx,
                                                                                                   ncol, y, ldy, alpha);
    }
    check("gemv");
}

// ---------------------------------------------------------------- norms
// (a thread's values and their weights read first - their loads in flight together - then summed and scaled in the
//  order they were: a thread's 12 values of 3072 were 12 memory trips one after another, and its 12 weights - cold, a
//  layer's are read once a window - 12 more.  Up to kRmsPer values a thread; wider rows as before)
constexpr int kRmsPer = 16;
__global__ void rms_kernel(const float* __restrict__ x, size_t ldx, int n, const float* __restrict__ w, float eps,
                           float* __restrict__ y, size_t ldy, const Q8Out q8, const int ahead) {
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float ss = 0.0f;
    if (ahead && n <= kRmsPer * (int) blockDim.x) {
        float* yo = y + (size_t) blockIdx.x * ldy;
        float v[kRmsPer], wv[kRmsPer];
#pragma unroll
        for (int j = 0; j < kRmsPer; ++j) {
            const int i = (int) threadIdx.x + j * (int) blockDim.x;
            v[j] = i < n ? xr[i] : 0.0f;
            wv[j] = i < n && w ? w[i] : 1.0f;
        }
#pragma unroll
        for (int j = 0; j < kRmsPer; ++j)
            if ((int) threadIdx.x + j * (int) blockDim.x < n) ss += v[j] * v[j];
        ss = block_sum(ss);
        const float r = 1.0f / sqrtf(ss / (float) n + eps);
#pragma unroll
        for (int j = 0; j < kRmsPer; ++j) {
            const int i = (int) threadIdx.x + j * (int) blockDim.x;
            if (i < n) {
                const float o = v[j] * r * wv[j];
                yo[i] = o;
                if (q8.q != nullptr) q8_put(o, (size_t) blockIdx.x * n + i, q8);
            }
        }
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float r = 1.0f / sqrtf(ss / (float) n + eps);
    float* yr = y + (size_t) blockIdx.x * ldy;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {   // (n a multiple of 32 with q8: a warp's lanes together)
        const float o = xr[i] * r * (w ? w[i] : 1.0f);
        yr[i] = o;
        if (q8.q != nullptr) q8_put(o, (size_t) blockIdx.x * n + i, q8);
    }
}

// ---------------------------------------------------------------- hyper-connections
// a block a token: rmsnorm of the hc E values, the 2 hc (head: hc) mixes, pre / post, cur
// Two passes so a token's 12288 values are read by many blocks: hc_part (grid (kHcBlocks, T)) - each block a slice's
// sum of squares and its 2 hc (head: hc) partial dots with fn's rows; hc_fin (a block a token) - the sums, the mixes,
// pre / post, cur
constexpr int kHcBlocks = 24, kMaxM = 16;
// (BYEL: fn by element - [HE][nm], nm a multiple of 4 - so an element's nm weights come in nm / 4 loads of 16 bytes,
//  not nm loads a memory trip each: the same products summed in the same order)
template <bool BYEL>
__device__ __forceinline__ void hc_part_body(int t, int b, const float* __restrict__ x, int HE, int nm,
                                             const float* __restrict__ fn, float* __restrict__ part) {
    __shared__ float sh[8][kMaxM + 1];
    const int per = (HE + kHcBlocks - 1) / kHcBlocks, i0 = b * per, i1 = min(HE, i0 + per);
    const float* xt = x + (size_t) t * HE;
    float acc[kMaxM + 1];
#pragma unroll
    for (int j = 0; j <= kMaxM; ++j) acc[j] = 0.0f;
    // (BYEL, a thread's values two at most - a token's 12288 over 24 blocks of 256: both values and their weights
    //  read first, the loads in flight together - the weights are cold, a layer's read once a window - then the same
    //  products summed in the same order)
    constexpr int kIt = 2;
    bool ahead = false;
    if constexpr (BYEL) ahead = i1 - i0 <= kIt * (int) blockDim.x;
    if (ahead) {
        float v[kIt];
        float4 wv[kIt][kMaxM / 4];
#pragma unroll
        for (int q = 0; q < kIt; ++q) {
            const int i = i0 + (int) threadIdx.x + q * (int) blockDim.x;
            const bool in = i < i1;
            v[q] = in ? xt[i] : 0.0f;
            const float4* f4 = (const float4*) (fn + (size_t) (in ? i : i0) * nm);
#pragma unroll
            for (int j = 0; j < kMaxM; j += 4)
                wv[q][j / 4] = in && j < nm ? f4[j / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int q = 0; q < kIt; ++q)
            if (i0 + (int) threadIdx.x + q * (int) blockDim.x < i1) {
                acc[kMaxM] = fmaf(v[q], v[q], acc[kMaxM]);
#pragma unroll
                for (int j = 0; j < kMaxM; j += 4)
                    if (j < nm) {
                        acc[j] = fmaf(wv[q][j / 4].x, v[q], acc[j]);
                        acc[j + 1] = fmaf(wv[q][j / 4].y, v[q], acc[j + 1]);
                        acc[j + 2] = fmaf(wv[q][j / 4].z, v[q], acc[j + 2]);
                        acc[j + 3] = fmaf(wv[q][j / 4].w, v[q], acc[j + 3]);
                    }
            }
    }
    for (int i = i0 + (int) threadIdx.x; i < i1 && !ahead; i += blockDim.x) {
        const float v = xt[i];
        acc[kMaxM] = fmaf(v, v, acc[kMaxM]);
        if constexpr (BYEL) {
            const float4* f4 = (const float4*) (fn + (size_t) i * nm);
#pragma unroll
            for (int j = 0; j < kMaxM; j += 4)
                if (j < nm) {
                    const float4 wv = f4[j / 4];
                    acc[j] = fmaf(wv.x, v, acc[j]);
                    acc[j + 1] = fmaf(wv.y, v, acc[j + 1]);
                    acc[j + 2] = fmaf(wv.z, v, acc[j + 2]);
                    acc[j + 3] = fmaf(wv.w, v, acc[j + 3]);
                }
        } else {
#pragma unroll
            for (int j = 0; j < kMaxM; ++j)
                if (j < nm) acc[j] = fmaf(fn[(size_t) j * HE + i], v, acc[j]);
        }
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int j = 0; j <= kMaxM; ++j) {
        // (the sums past nm are zeros: written as such without their shuffles - the same bits)
        const float v = j < nm || j == kMaxM ? warp_sum(acc[j]) : 0.0f;
        if (l == 0) sh[w][j] = v;
    }
    __syncthreads();
    if ((int) threadIdx.x <= kMaxM) {
        float v = 0.0f;
        for (int q = 0; q < (int) (blockDim.x >> 5); ++q) v += sh[q][threadIdx.x];
        part[((size_t) t * kHcBlocks + b) * (kMaxM + 1) + threadIdx.x] = v;
    }
}
template <bool BYEL>
__global__ void __launch_bounds__(256) hc_part_kernel(const float* __restrict__ x, int HE, int nm,
                                                      const float* __restrict__ fn, float* __restrict__ part) {
    hc_part_body<BYEL>(blockIdx.y, blockIdx.x, x, HE, nm, fn, part);
}
// hc_fin's work for token t, KPER values a thread (E <= blockDim KPER); part read past L1 (another block's writes)
template <bool HEAD, int KPER>
__device__ __forceinline__ void hc_fin_body(int t, const float* __restrict__ x, int hc, int E,
                                            const float* __restrict__ part, const float* __restrict__ scale,
                                            const float* __restrict__ base, float eps, float hc_eps, float mag,
                                            float* __restrict__ cur, float* __restrict__ post,
                                            float* __restrict__ mix_out, const float* __restrict__ nw, float neps,
                                            const Q8Out& q8) {
    __shared__ float s_mix[kMaxM + 1], s_pre[8];
    __shared__ float s_part[kHcBlocks * (kMaxM + 1)];
    const int nm = HEAD ? hc : 2 * hc, HE = hc * E;
    // (the streams' values and the norm's weights - cold, a layer's read once a window - asked for at the start: a
    //  thread's loads in flight together while the mixes are made, not each behind the sum before it, the weights' six
    //  one after another at the end; the same sums in the same order)
    const float* xt = x + (size_t) t * HE;
    constexpr int kPer = KPER;   // (E <= blockDim kPer: a thread's values kept for the norm)
    float xa[kPer][8], nwv[kPer];
#pragma unroll
    for (int q = 0; q < kPer; ++q) {
        const int d = threadIdx.x + q * blockDim.x;
        nwv[q] = nw && d < E ? nw[d] : 1.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) xa[q][j] = d < E && j < hc ? xt[(size_t) j * E + d] : 0.0f;
    }
    // (the slices' partial sums: a thread a value into shared memory - 17 threads read their 24 one after another,
    //  each a trip past L1 - then summed in the slices' order as before: the same bits)
    constexpr int kNp = kHcBlocks * (kMaxM + 1);
    for (int i = threadIdx.x; i < kNp; i += blockDim.x)
        s_part[i] = maya::kernels::ld_l2(part + (size_t) t * kNp + i);
    __syncthreads();
    if ((int) threadIdx.x <= kMaxM) {
        float v = 0.0f;
        for (int b = 0; b < kHcBlocks; ++b) v += s_part[b * (kMaxM + 1) + threadIdx.x];
        s_mix[threadIdx.x] = v;
    }
    __syncthreads();
    // (a thread a stream, not thread 0 looping over them: CUDA 13.0's ptxas (V13.0.88) built that loop for sm_120
    //  with base + hc's address in 32 bits - its upper half dropped, an illegal read at the warm-up; the same products.
    //  12.9's ptxas builds it the same way, 12.8's and 13.3's do not; hc_pre1 (HY4_HC_ONE=1) shares this body)
    if ((int) threadIdx.x < hc) {
        const int j = threadIdx.x;
        const float r = 1.0f / sqrtf(s_mix[kMaxM] / (float) HE + eps);
        const float m0 = s_mix[j] * r;
        if (mix_out) mix_out[(size_t) t * nm + j] = m0;
        s_pre[j] = sigmoidf_(m0 * scale[0] + base[j]) + hc_eps;
        if (!HEAD) {
            const float m1 = s_mix[hc + j] * r;
            if (mix_out) mix_out[(size_t) t * nm + hc + j] = m1;
            post[(size_t) t * hc + j] = sigmoidf_(m1 * scale[1] + base[hc + j]) * mag + hc_eps;
        }
    }
    __syncthreads();
    float vals[kPer], ss = 0.0f, pre[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) pre[j] = j < hc ? s_pre[j] : 0.0f;
#pragma unroll
    for (int q = 0; q < kPer; ++q) {
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) v = fmaf(pre[j], xa[q][j], v);
        vals[q] = v;
        ss += v * v;
    }
    float r = 1.0f;
    if (nw) {
        ss = block_sum(ss);
        r = 1.0f / sqrtf(ss / (float) E + neps);
    }
#pragma unroll
    for (int q = 0; q < kPer; ++q) {
        const int d = threadIdx.x + q * blockDim.x;
        if (d < E) {   // (E a multiple of 32 with q8: a warp's lanes together, 32 values in a row)
            const float o = nw ? vals[q] * r * nwv[q] : vals[q];
            cur[(size_t) t * E + d] = o;
            if (q8.q != nullptr) q8_put(o, (size_t) t * E + d, q8);
        }
    }
}
template <bool HEAD>
__global__ void __launch_bounds__(512) hc_fin_kernel(const float* __restrict__ x, int hc, int E,
                                                     const float* __restrict__ part, const float* __restrict__ scale,
                                                     const float* __restrict__ base, float eps, float hc_eps, float mag,
                                                     float* __restrict__ cur, float* __restrict__ post,
                                                     float* __restrict__ mix_out, const float* __restrict__ nw,
                                                     float neps, const Q8Out q8) {
    hc_fin_body<HEAD, 8>(blockIdx.x, x, hc, E, part, scale, base, eps, hc_eps, mag, cur, post, mix_out, nw, neps, q8);
}
// ... in one launch: the slices' blocks as hc_part, the last of a token's to finish its hc_fin (a counter a token)
__global__ void __launch_bounds__(256) hc_pre1_kernel(const float* __restrict__ x, int hc, int E,
                                                      const float* __restrict__ fn, float* __restrict__ part,
                                                      unsigned* __restrict__ cnt, const float* __restrict__ scale,
                                                      const float* __restrict__ base, float eps, float hc_eps, float mag,
                                                      float* __restrict__ cur, float* __restrict__ post,
                                                      float* __restrict__ mix_out, const float* __restrict__ nw,
                                                      float neps) {
    const int t = blockIdx.y;
    hc_part_body<false>(t, blockIdx.x, x, hc * E, 2 * hc, fn, part);
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&cnt[t], 1u) == (unsigned) kHcBlocks - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    hc_fin_body<false, 12>(t, x, hc, E, part, scale, base, eps, hc_eps, mag, cur, post, mix_out, nw, neps, Q8Out{});
    if (threadIdx.x == 0) cnt[t] = 0u;
}
// hc_pre1's counters (a token's, back to 0 after each launch), a zeroed buffer a device, grown as needed
unsigned* hc_cnt_buf(int T) {
    int dev = 0;
    cudaGetDevice(&dev);
    static std::mutex mu;
    static std::map<int, std::pair<unsigned*, size_t>> bufs;
    std::lock_guard<std::mutex> lk(mu);
    auto& b = bufs[dev];
    if (b.second < (size_t) T) {
        if (b.first) cudaFree(b.first);
        b.first = nullptr;
        b.second = 0;
        const size_t n = std::max<size_t>((size_t) T, 1024);
        if (cudaMalloc(&b.first, n * sizeof(unsigned)) != cudaSuccess || cudaMemset(b.first, 0, n * sizeof(unsigned)) != cudaSuccess)
            return nullptr;
        b.second = n;
    }
    return b.first;
}
// hc_part's sums (kHcBlocks x (kMaxM + 1) floats a token), a buffer a device, grown as needed
float* hc_part_buf(int T) {
    int dev = 0;
    cudaGetDevice(&dev);
    static std::mutex mu;
    static std::map<int, std::pair<float*, size_t>> bufs;
    std::lock_guard<std::mutex> lk(mu);
    auto& b = bufs[dev];
    const size_t need = (size_t) T * kHcBlocks * (kMaxM + 1);
    if (b.second < need) {
        if (b.first) cudaFree(b.first);
        b.first = nullptr;
        b.second = 0;
        if (cudaMalloc(&b.first, need * sizeof(float)) != cudaSuccess) return nullptr;
        b.second = need;
    }
    return b.first;
}
// grid (kPostBlocks, T): every block the RMS of y's row (12 KB from L2), then its slice: x[s] += y r w post[s]
// (yo: the normed y written there too - the debug read-outs)
constexpr int kPostBlocks = 12;
__global__ void __launch_bounds__(256) hc_post_norm_kernel(float* __restrict__ x, int hc, int E,
                                                           const float* __restrict__ y, const float* __restrict__ post,
                                                           const float* __restrict__ w, float eps,
                                                           float* __restrict__ yo) {
    const int t = blockIdx.y, b = blockIdx.x;
    const float* yt = y + (size_t) t * E;
    // (a slice of one value a thread - 3072 over 12 blocks of 256: that value, its weight - cold - and the streams'
    //  asked for now, in flight while the row's sum is made; the same products and sums)
    const int per_one = (E + kPostBlocks - 1) / kPostBlocks, d_one = b * per_one + (int) threadIdx.x;
    const bool one = per_one <= (int) blockDim.x;
    const bool in_one = one && d_one < min(E, (b + 1) * per_one);
    const float y_one = in_one ? yt[d_one] : 0.0f, w_one = in_one ? w[d_one] : 0.0f;
    float xs_one[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) xs_one[j] = in_one && j < hc ? x[((size_t) t * hc + j) * E + d_one] : 0.0f;
    float ss = 0.0f;
    if (E <= 4 * 4 * (int) blockDim.x) {   // (a thread's up to 4 loads in flight together, summed in the same order)
        float4 yv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = ((int) threadIdx.x + j * (int) blockDim.x) * 4;
            yv[j] = d < E ? *(const float4*) (yt + d) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (((int) threadIdx.x + j * (int) blockDim.x) * 4 < E) {
                const float4 v = yv[j];
                ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
    } else {
        for (int d = threadIdx.x * 4; d < E; d += blockDim.x * 4) {
            const float4 v = *(const float4*) (yt + d);
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    ss = block_sum(ss);
    const float r = 1.0f / sqrtf(ss / (float) E + eps);
    const int per = (E + kPostBlocks - 1) / kPostBlocks, d0 = b * per, d1 = min(E, d0 + per);
    float* xt = x + (size_t) t * hc * E;
    float pw[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) pw[j] = j < hc ? post[(size_t) t * hc + j] : 0.0f;
    if (one) {   // (the slice's one value a thread: its loads were asked for before the sum)
        if (d_one < d1) {
            const float v = y_one * r * w_one;
            if (yo) yo[(size_t) t * E + d_one] = v;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (j < hc) xt[(size_t) j * E + d_one] = xs_one[j] + v * pw[j];
        }
        return;
    }
    for (int d = d0 + (int) threadIdx.x; d < d1; d += blockDim.x) {
        const float v = yt[d] * r * w[d];
        if (yo) yo[(size_t) t * E + d] = v;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (j < hc) xt[(size_t) j * E + d] += v * pw[j];
    }
}
__global__ void hc_post_kernel(float* __restrict__ x, int hc, int E, const float* __restrict__ y,
                               const float* __restrict__ post, size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t t = i / ((size_t) hc * E), s = (i / E) % hc, d = i % E;
    x[i] += y[t * E + d] * post[t * hc + s];
}

// ---------------------------------------------------------------- RoPE (NORM: consecutive pairs)
__device__ __forceinline__ void rope_pair(float& a, float& b, float pos, int i, int n_rot, float base) {
    const float theta_scale = powf(base, -2.0f / (float) n_rot);
    const float theta = pos * powf(theta_scale, (float) i);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = a, x1 = b;
    a = x0 * c - x1 * s;
    b = x0 * s + x1 * c;
}
__global__ void rope_kernel(float* __restrict__ x, int nh, size_t ld, size_t ldt, int off, int n_rot,
                            const int* __restrict__ pos, float base) {
    const int t = blockIdx.y, half = n_rot / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // (head, pair)
    if (i >= nh * half) return;
    const int h = i / half, p = i % half;
    float* v = x + (size_t) t * ldt + (size_t) h * ld + off + 2 * p;
    rope_pair(v[0], v[1], (float) pos[t], p, n_rot, base);
}

// A window's queries in one launch (a block a (head, token), a thread a value of the head's k = nope + n_rot): the
// rotated part rotated in place as rope_kernel does and copied to qcat's rotated columns (copy_cols), the first nope
// values' int8 rows for k_b (q8_heads' layout: row h T + t) - three launches' work, each value's arithmetic the same
__global__ void __launch_bounds__(256) rope_q8_copy_kernel(float* __restrict__ q, int k, int nope, int n_rot,
                                                           const int* __restrict__ pos, float base,
                                                           float* __restrict__ qcat, size_t ldq, int kl, const Q8Out q8) {
    const int h = blockIdx.x, t = blockIdx.y, c = threadIdx.x;
    const size_t r = (size_t) t * gridDim.x + h;
    float* v = q + r * k;
    if (c < nope) {   // (nope a multiple of 32: a warp's lanes together)
        q8_put(v[c], ((size_t) h * gridDim.y + t) * nope + c, q8);
    } else if (c < nope + n_rot && ((c - nope) & 1) == 0) {
        rope_pair(v[c], v[c + 1], (float) pos[t], (c - nope) >> 1, n_rot, base);
        float* d = qcat + r * ldq + kl + (c - nope);
        d[0] = v[c];
        d[1] = v[c + 1];
    }
}

// ---------------------------------------------------------------- the latent cache
// KV streaming: the row just written (words 32-bit words) into the host's row of its position as well, and the slot's
// tag
// (KV spill: the newest tier's slot p % hres, its last position's hidx dropped where it still names the slot)
__device__ __forceinline__ void kv_mirror(const void* row, uint32_t* host, int words, int* tag, int slot, int p,
                                          int* htag, int* hidx, int hres) {
    __syncthreads();
    const size_t hs = hidx ? (size_t) (p % hres) : (size_t) p;
    uint32_t* host_row = host + hs * (size_t) words;
    const uint32_t* src = (const uint32_t*) row;
    for (int i = threadIdx.x; i < words; i += blockDim.x) host_row[i] = src[i];
    if (threadIdx.x == 0) {
        tag[slot] = p;
        if (hidx) {
            const int old = htag[hs];
            if (old >= 0 && old != p && hidx[old] == (int) hs) hidx[old] = -1;
            htag[hs] = p;
            hidx[p] = (int) hs;
        }
    }
}
// a reader's row of position p: the VRAM ring's where its tag is p, else the host's
// (KV spill: the host's row of p in slot hidx[p] - kv_need put it there before the read)
__device__ __forceinline__ size_t kv_src(int p, int ring, const int* tag, const int* hidx, bool& host) {
    const int r = ring > 0 ? p % ring : p;
    host = tag != nullptr && tag[r] != p;
    return host ? (size_t) (hidx ? max(hidx[p], 0) : p) : (size_t) r;
}
__global__ void kv_store_kernel(const float* __restrict__ kv, int kv_lora, int n_rot, const float* __restrict__ w,
                                float eps, const int* __restrict__ pos, float base, __half* __restrict__ cache,
                                int ring, int* __restrict__ tag, uint32_t* __restrict__ host, int* htag, int* hidx,
                                int hres) {
    const int t = blockIdx.x, row = kv_lora + n_rot;
    const float* x = kv + (size_t) t * row;
    const int p = pos[t];
    const int slot = ring > 0 ? p % ring : p;
    __half* dst = cache + (size_t) slot * row;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float r = 1.0f / sqrtf(ss / (float) kv_lora + eps);
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) dst[i] = __float2half_rn(x[i] * r * w[i]);
    for (int i = threadIdx.x; i < n_rot / 2; i += blockDim.x) {
        float a = x[kv_lora + 2 * i], b = x[kv_lora + 2 * i + 1];
        rope_pair(a, b, (float) p, i, n_rot, base);
        dst[kv_lora + 2 * i] = __float2half_rn(a);
        dst[kv_lora + 2 * i + 1] = __float2half_rn(b);
    }
    if (tag) kv_mirror(dst, host, row / 2, tag, slot, p, htag, hidx, hres);
}

// ... in Q8_0 rows (kv_q8_row_bytes): the row's values staged in shared memory, then each block of 32 of the latent by
// a warp - its largest magnitude / 127 the F16 scale, every value rounded to an int8 of it (ggml's Q8_0 rounding; the
// values the F16 kernel stores, before their F16 rounding); the rotated part as F16 after the scales
__global__ void kv_store_q8_kernel(const float* __restrict__ kv, int kv_lora, int n_rot, const float* __restrict__ w,
                                   float eps, const int* __restrict__ pos, float base, uint8_t* __restrict__ cache,
                                   int ring, int rb, int* __restrict__ tag, uint32_t* __restrict__ host,
                                   int* htag, int* hidx, int hres) {
    __shared__ float v[1024];
    const int t = blockIdx.x, row = kv_lora + n_rot;
    const float* x = kv + (size_t) t * row;
    const int p = pos[t];
    const int slot = ring > 0 ? p % ring : p;
    uint8_t* dst = cache + (size_t) slot * rb;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float r = 1.0f / sqrtf(ss / (float) kv_lora + eps);
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) v[i] = x[i] * r * w[i];
    for (int i = threadIdx.x; i < n_rot / 2; i += blockDim.x) {
        float a = x[kv_lora + 2 * i], b = x[kv_lora + 2 * i + 1];
        rope_pair(a, b, (float) p, i, n_rot, base);
        v[kv_lora + 2 * i] = a;
        v[kv_lora + 2 * i + 1] = b;
    }
    __syncthreads();
    const int wp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __half* sc = (__half*) (dst + kv_lora);
    __half* rp = sc + kv_lora / 32;
    for (int i = threadIdx.x; i < n_rot; i += blockDim.x) rp[i] = __float2half_rn(v[kv_lora + i]);
    for (int b = wp; b < kv_lora / 32; b += nw) {
        const float x0 = v[32 * b + lane];
        float am = fabsf(x0);
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        const __half dh = __float2half_rn(am / 127.0f);
        const float d = __half2float(dh);
        const int q = d > 0.0f ? max(-127, min(127, __float2int_rn(x0 / d))) : 0;
        dst[32 * b + lane] = (uint8_t) (int8_t) q;
        if (lane == 0) sc[b] = dh;
    }
    for (int i = kv_lora + kv_lora / 16 + 2 * n_rot + threadIdx.x; i < rb; i += blockDim.x) dst[i] = 0;   // (its pad)
    if (tag) kv_mirror(dst, host, rb / 4, tag, slot, p, htag, hidx, hres);
}

// ---------------------------------------------------------------- MLA attention over the latent cache
// grid (chunks, ceil(H / 8), T); a warp a head, the chunk's keys in tiles of 32 rows through shared memory.  A row is
// 576 F16 = 288 words; lane l reads the words l + 32 i (bank l: no conflicts): q's dims 2 (l + 32 i) and the next, the
// value's (i < 8) the same.  Partial (m, l, acc[512]) a (t, head, chunk) -> attn_combine.
constexpr int kKC = 128, kKCmin = 32, kTile = 32;   // (a chunk's keys: kKC, or kKCmin for a few tokens - decode)
constexpr int kRowW = 288;   // words a row (kv_lora 512 + rope 64, F16)
// (Q8: the cache's Q8_0 rows of rb bytes - the latent's int8 values, its 16 blocks' F16 scales, the rotated part's F16
// values -, dequantized to F16 as the tile is loaded: the same arithmetic after it)
// HB: the heads a block (a warp each; grid y ceil(H / HB)) - every head's sums the same whichever HB, only the tiles a
// key is read into: 8 (NVIDIA), 32 on AMD (a gathered key read once for 32 heads rather than 4 times: the indexed
// layers' prompt attention past the dense span and the decode steps' attention are bound by those reads on a Radeon)
template <bool Q8, int HB = 8>
__global__ void __launch_bounds__(32 * HB) attn_kernel(const float* __restrict__ q, int H, const void* __restrict__ cache_v,
                                                       int ring, const int* __restrict__ keys, int ldk,
                                                       const int* __restrict__ lo, const int* __restrict__ nk, float scale,
                                                       int nchunk, int kc, float* __restrict__ ws, int rb) {
    __shared__ __half2 tile[kTile][kRowW];
    __shared__ int rows[kTile];
    const int c = blockIdx.x, t = blockIdx.z, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.y * HB + w;
    const int n = nk[t], k0 = c * kc, k1 = min(n, k0 + kc);
    float qr[18];
    if (h < H) {
        const float* qh = q + ((size_t) t * H + h) * (2 * kRowW);
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            qr[2 * i] = qh[2 * (lane + 32 * i)];
            qr[2 * i + 1] = qh[2 * (lane + 32 * i) + 1];
        }
    }
    float m = -INFINITY, l = 0.0f, acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int kt = k0; kt < k1; kt += kTile) {
        const int nt = min(kTile, k1 - kt);
        __syncthreads();
        if ((int) threadIdx.x < nt) {
            const int p = keys ? keys[(size_t) t * ldk + kt + threadIdx.x] : lo[t] + kt + (int) threadIdx.x;
            rows[threadIdx.x] = ring > 0 ? p % ring : p;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < nt * kRowW; i += blockDim.x) {
            const int r = i / kRowW, col = i % kRowW;
            if (Q8) {
                const uint8_t* rp = (const uint8_t*) cache_v + (size_t) rows[r] * rb;
                if (col < 256) {
                    const char2 qv = *(const char2*) (rp + 2 * col);
                    const float d = __half2float(*(const __half*) (rp + 512 + 2 * (col >> 4)));
                    tile[r][col] = __floats2half2_rn(d * (float) qv.x, d * (float) qv.y);
                } else {
                    tile[r][col] = *(const __half2*) (rp + 544 + 4 * (col - 256));
                }
            } else {
                tile[r][col] = ((const __half2*) cache_v)[(size_t) rows[r] * kRowW + col];
            }
        }
        __syncthreads();
        if (h >= H) continue;
        for (int j = 0; j < nt; ++j) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                const float2 kv = __half22float2(tile[j][lane + 32 * i]);
                s = fmaf(qr[2 * i], kv.x, fmaf(qr[2 * i + 1], kv.y, s));
            }
            s = warp_sum(s) * scale;
            const float mn = fmaxf(m, s);
            const float corr = expf(m - mn), pj = expf(s - mn);
            l = l * corr + pj;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float2 v = __half22float2(tile[j][lane + 32 * i]);
                acc[2 * i] = fmaf(pj, v.x, acc[2 * i] * corr);
                acc[2 * i + 1] = fmaf(pj, v.y, acc[2 * i + 1] * corr);
            }
            m = mn;
        }
    }
    if (h >= H) return;
    float* o = ws + (((size_t) t * H + h) * nchunk + c) * (512 + 2);
    if (lane == 0) {
        o[0] = m;
        o[1] = l;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        o[2 + 2 * (lane + 32 * i)] = acc[2 * i];
        o[2 + 2 * (lane + 32 * i) + 1] = acc[2 * i + 1];
    }
}
// ... the same sums, sooner.  A thread's loads from the cache come back one after another, each a trip to memory - so
// the tile is read 16 bytes a load (a row's 42 such words spread over the block's threads: two or three loads a thread
// where attn_kernel made dozens), its scales first.  The tile holds F32 - the F16 values, widened as they are loaded -
// in half the rows; a key that is not a new maximum (most) leaves the sums unscaled (their factor is expf(0) = 1:
// x * 1 = x).  Every product, sum and rounding as attn_kernel's: the same bits.
constexpr int kTile2 = 16;
template <bool Q8>
__global__ void __launch_bounds__(256) attn2_kernel(const float* __restrict__ q, int H, const void* __restrict__ cache_v,
                                                    int ring, const int* __restrict__ keys, int ldk,
                                                    const int* __restrict__ lo, const int* __restrict__ nk, float scale,
                                                    int nchunk, int kc, float* __restrict__ ws, int rb) {
    __shared__ float2 tile[kTile2][kRowW];
    __shared__ int rows[kTile2];
    __shared__ float sc[kTile2][16];   // (Q8: a row's 16 scales)
    const int c = blockIdx.x, t = blockIdx.z, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.y * 8 + w;
    const int n = nk[t], k0 = c * kc, k1 = min(n, k0 + kc);
    float qr[18];
    if (h < H) {   // (the head's query: this lane's 18 of its 576 values, in 16-byte loads where they are aligned)
        const float* qh = q + ((size_t) t * H + h) * (2 * kRowW);
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            const float2 v = *(const float2*) (qh + 2 * (lane + 32 * i));
            qr[2 * i] = v.x;
            qr[2 * i + 1] = v.y;
        }
    }
    float m = -INFINITY, l = 0.0f, acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int kt = k0; kt < k1; kt += kTile2) {
        const int nt = min(kTile2, k1 - kt);
        __syncthreads();
        if ((int) threadIdx.x < nt) {
            const int p = keys ? keys[(size_t) t * ldk + kt + threadIdx.x] : lo[t] + kt + (int) threadIdx.x;
            rows[threadIdx.x] = ring > 0 ? p % ring : p;
        }
        __syncthreads();
        if (Q8) {
            // a row: 512 int8 (32 words of 16 bytes), its 16 F16 scales (2 words), 64 F16 rotated values (8 words)
            if ((int) threadIdx.x < 2 * nt) {
                const int r = threadIdx.x >> 1, hf = threadIdx.x & 1;
                const uint4 u = *(const uint4*) ((const uint8_t*) cache_v + (size_t) rows[r] * rb + 512 + 16 * hf);
                const unsigned wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int qd = 0; qd < 4; ++qd) {
                    sc[r][hf * 8 + 2 * qd] = __half2float(__ushort_as_half((unsigned short) (wd[qd] & 0xffffu)));
                    sc[r][hf * 8 + 2 * qd + 1] = __half2float(__ushort_as_half((unsigned short) (wd[qd] >> 16)));
                }
            }
            __syncthreads();
            for (int it = threadIdx.x; it < nt * 40; it += blockDim.x) {
                const int r = it / 40, ci = it % 40;
                const uint8_t* rp = (const uint8_t*) cache_v + (size_t) rows[r] * rb;
                if (ci < 32) {   // (16 int8: 8 pairs, one scale - the block of 32 values they are in)
                    const uint4 u = *(const uint4*) (rp + 16 * ci);
                    const unsigned wd[4] = {u.x, u.y, u.z, u.w};
                    const float d = sc[r][ci >> 1];
#pragma unroll
                    for (int qd = 0; qd < 4; ++qd) {
                        const float a0 = (float) (signed char) (wd[qd] & 0xffu), a1 = (float) (signed char) ((wd[qd] >> 8) & 0xffu);
                        const float a2 = (float) (signed char) ((wd[qd] >> 16) & 0xffu), a3 = (float) (signed char) (wd[qd] >> 24);
                        tile[r][8 * ci + 2 * qd] = __half22float2(__floats2half2_rn(d * a0, d * a1));
                        tile[r][8 * ci + 2 * qd + 1] = __half22float2(__floats2half2_rn(d * a2, d * a3));
                    }
                } else {         // (the rotated part: 4 pairs of F16)
                    const uint4 u = *(const uint4*) (rp + 544 + 16 * (ci - 32));
                    const unsigned wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                    for (int qd = 0; qd < 4; ++qd)
                        tile[r][256 + 4 * (ci - 32) + qd] =
                            make_float2(__half2float(__ushort_as_half((unsigned short) (wd[qd] & 0xffffu))),
                                        __half2float(__ushort_as_half((unsigned short) (wd[qd] >> 16))));
                }
            }
        } else {   // (F16 rows: 72 words of 4 pairs)
            for (int it = threadIdx.x; it < nt * 72; it += blockDim.x) {
                const int r = it / 72, ci = it % 72;
                const uint4 u = *(const uint4*) ((const uint8_t*) cache_v + ((size_t) rows[r] * kRowW + 4 * ci) * 4);
                const unsigned wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int qd = 0; qd < 4; ++qd)
                    tile[r][4 * ci + qd] = make_float2(__half2float(__ushort_as_half((unsigned short) (wd[qd] & 0xffffu))),
                                                       __half2float(__ushort_as_half((unsigned short) (wd[qd] >> 16))));
            }
        }
        __syncthreads();
        if (h >= H) continue;
        for (int j = 0; j < nt; ++j) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                const float2 kv = tile[j][lane + 32 * i];
                s = fmaf(qr[2 * i], kv.x, fmaf(qr[2 * i + 1], kv.y, s));
            }
            s = warp_sum(s) * scale;
            const float mn = fmaxf(m, s);
            if (mn == m) {   // (not a new maximum: corr = expf(m - m) = 1)
                const float pj = expf(s - mn);
                l = l * 1.0f + pj;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float2 v = tile[j][lane + 32 * i];
                    acc[2 * i] = fmaf(pj, v.x, acc[2 * i]);
                    acc[2 * i + 1] = fmaf(pj, v.y, acc[2 * i + 1]);
                }
            } else {         // (a new maximum, s: pj = expf(s - s) = 1 - or a NaN, as attn_kernel takes it)
                const float corr = expf(m - mn), pj = expf(s - mn);
                l = l * corr + pj;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float2 v = tile[j][lane + 32 * i];
                    acc[2 * i] = fmaf(pj, v.x, acc[2 * i] * corr);
                    acc[2 * i + 1] = fmaf(pj, v.y, acc[2 * i + 1] * corr);
                }
            }
            m = mn;
        }
    }
    if (h >= H) return;
    float* o = ws + (((size_t) t * H + h) * nchunk + c) * (512 + 2);
    if (lane == 0) {
        o[0] = m;
        o[1] = l;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        o[2 + 2 * (lane + 32 * i)] = acc[2 * i];
        o[2 + 2 * (lane + 32 * i) + 1] = acc[2 * i + 1];
    }
}
// ---------------------------------------------------------------- the same attention on an RDNA GPU's matrix cores
// A prompt sub-batch's tokens over their picked keys (the indexed layers past the dense span), maya-ds4's WMMA pattern
// (sparse_attn_wmma_kernel) on this cache's rows: grid (H / (16 G), T), 256 threads (8 warps) a group of 16 heads, G
// groups a block over one token's keys - each tile of 16 rows staged once into shared memory as F16 for all G (a
// thread's 16-byte loads of the next tile in flight under this tile's sums; Q8 rows dequantized as they are stored).
// The dots: A = 16 heads' queries, B = the tile's 16 rows; warp w over the latent's dimensions w * 64 .. + 64 and, for
// w < 4, the rotated part's 512 + 16 w .. + 16; the warps' parts summed in shared memory in warp order.  The query and
// the probabilities split into an F16 high and a low part (two products each: ~22 bits of mantissa), the rows exact
// in F16 - within rounding of attn_kernel's F32 sums, in another order.  Then a thread a (head, row): the online
// softmax; warp w's output dimensions w * 64 .. + 64; the sink and the division at the end (attn_combine's sums, in
// one pass: no partial results).  Only past the dense span on the prompt path: every token window - decode, a draft's
// verify - keeps attn_kernel's own sums.
constexpr int kWR = 16, kWDS = 576 + 8;   // (the tile's rows; a row's stride in F16 values: 16-byte reads on other banks)
// (AMD: registers for 8 waves a SIMD - two blocks of 512 threads a WGP rather than one at the 200 the compiler takes
//  on its own: the kernel waits on its loads and barriers, so the second block hides them)
#if defined(MAYA_USE_HIP)
#define HY4_WMMA_OCC __attribute__((amdgpu_waves_per_eu(8)))
#else
#define HY4_WMMA_OCC
#endif
template <bool Q8, int G>
__global__ void __launch_bounds__(256 * G) HY4_WMMA_OCC attn_wmma_kernel(const float* __restrict__ q, int H,
                                                           const void* __restrict__ cache_v, int ring,
                                                           const int* __restrict__ keys, int ldk,
                                                           const int* __restrict__ nk, float scale,
                                                           const float* __restrict__ sinks, float* __restrict__ out,
                                                           int rb) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    constexpr int NT = 256 * G, NQ = (kWR * 72 + NT - 1) / NT;   // (a tile: 16 rows x 72 16-byte words)
    constexpr int PS = kWR + 4;
    const int g = G == 1 ? 0 : (int) (threadIdx.x >> 8), lt = (int) (threadIdx.x & 255);
    const int hb = (blockIdx.x * G + g) * 16, t = blockIdx.y;
    const int n = nk[t];
    const int* kt = keys + (size_t) t * ldk;
    __shared__ __align__(16) _Float16 tile[kWR * kWDS];
    __shared__ float sp_[G][8][kWR * kWR];
    __shared__ float pr_[G][16 * PS];
    __shared__ float m_run_[G][16], den_[G][16], tsc_[G][16];
    float(*sp)[kWR * kWR] = sp_[g];
    float* pr = pr_[g];
    float *m_run = m_run_[g], *den = den_[g], *tsc = tsc_[g];
    const int lane = lt & 31, wp = lt >> 5, m16 = lane & 15, hi = lane >> 4;
    const int nkk = wp < 4 ? 5 : 4;   // (k-steps: the latent's 4, the rotated part's 1 for warps 0 .. 3)
    wmma_h16 qa[5], qb[5];
    {
        const float* qh = q + ((size_t) t * H + hb + m16) * 576;
#pragma unroll
        for (int kk = 0; kk < 5; ++kk) {
            const int d0 = kk < 4 ? wp * 64 + kk * 16 : 512 + wp * 16;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const float v = kk < nkk ? qh[d0 + e] : 0.0f;
                const _Float16 h = (_Float16) v;
                qa[kk][e] = h;
                qb[kk][e] = (_Float16) (v - (float) h);
            }
        }
    }
    if (lt < 16) {
        m_run[lt] = -1e30f;
        den[lt] = 0.0f;
    }
    lds_sync();
    wmma_f8 acc[4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) acc[nt] = wmma_f8{};
    // the staging: word i of the tile (row i / 72, 16-byte word i % 72 of it) - thread x's words x, x + NT, ..
    uint4 nx[NQ];
    auto fetch = [&](int f0) {
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int i = (int) threadIdx.x + k * NT, r = i / 72, w = i % 72;
            if (i >= kWR * 72 || f0 + r >= n) continue;
            const int p = kt[f0 + r], row = ring > 0 ? p % ring : p;
            if (!Q8) {
                nx[k] = ((const uint4*) cache_v)[(size_t) row * 72 + w];
            } else {
                const uint8_t* rp = (const uint8_t*) cache_v + (size_t) row * rb;
                if (w < 64) {   // (8 int8 values and their block's scale)
                    const uint2 b = *(const uint2*) (rp + 8 * w);
                    nx[k].x = b.x;
                    nx[k].y = b.y;
                    nx[k].z = *(const uint16_t*) (rp + 512 + 2 * (w >> 2));
                } else {
                    nx[k] = *(const uint4*) (rp + 544 + 16 * (w - 64));
                }
            }
        }
    };
    auto stage = [&](int f0) {
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int i = (int) threadIdx.x + k * NT, r = i / 72, w = i % 72;
            if (i >= kWR * 72) continue;
            uint4* dst = (uint4*) (tile + r * kWDS) + w;
            if (f0 + r >= n) {
                *dst = make_uint4(0u, 0u, 0u, 0u);
            } else if (!Q8 || w >= 64) {
                *dst = nx[k];
            } else {   // (attn_kernel's dequantization: d * q rounded to F16)
                const float d = (float) __builtin_bit_cast(_Float16, (unsigned short) nx[k].z);
                const uint32_t b[2] = {nx[k].x, nx[k].y};
                _Float16 o[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = (_Float16) (d * (float) (int8_t) ((b[j >> 2] >> (8 * (j & 3))) & 0xff));
                *dst = *(const uint4*) o;
            }
        }
    };
    fetch(0);
    for (int t0 = 0; t0 < n; t0 += kWR) {
        const int nr = min(kWR, n - t0);
        stage(t0);   // (this tile's readers before: done at the loop's last barrier)
        lds_sync();
        if (t0 + kWR < n) fetch(t0 + kWR);
        wmma_f8 s = {};
#pragma unroll
        for (int kk = 0; kk < 5; ++kk) {   // B: row m16, the k-step's 16 dimensions
            if (kk >= nkk) break;
            const int d0 = kk < 4 ? wp * 64 + kk * 16 : 512 + wp * 16;
            const uint4* kr = (const uint4*) (tile + m16 * kWDS + d0);
            const uint4 x0 = kr[0], x1 = kr[1];
            wmma_h16 kb;
            const uint32_t w8[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                kb[2 * e] = __builtin_bit_cast(_Float16, (unsigned short) (w8[e] & 0xffffu));
                kb[2 * e + 1] = __builtin_bit_cast(_Float16, (unsigned short) (w8[e] >> 16));
            }
            s = wmma_f16(qa[kk], kb, s);
            s = wmma_f16(qb[kk], kb, s);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) sp[wp][wmma_row(i, hi) * kWR + m16] = s[i];
        lds_sync();
        {   // thread (head lt / 16, row lt % 16): the tile's score, the head's running max and denominator
            const int h = lt >> 4, r = lt & 15;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) v += sp[w][lt];
            v = r < nr ? v * scale : -INFINITY;
            float mx = v;
#pragma unroll
            for (int k = 8; k >= 1; k >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, k));
            const float m_old = m_run[h], m_new = fmaxf(m_old, mx);
            const float p = r < nr ? expf(v - m_new) : 0.0f;
            float sum = p;
#pragma unroll
            for (int k = 8; k >= 1; k >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, k);
            pr[h * PS + r] = p;
            if (r == 0) {   // (the head's 16 lanes read m_run above, in this same instruction stream)
                const float ts = expf(m_old - m_new);
                tsc[h] = ts;
                den[h] = den[h] * ts + sum;
                m_run[h] = m_new;
            }
        }
        lds_sync();
        wmma_h16 pa, pb;   // A of the weighted sums: head m16's probabilities of the tile's rows
#pragma unroll
        for (int r = 0; r < kWR; ++r) {
            const float p = pr[m16 * PS + r];
            const _Float16 h = (_Float16) p;
            pa[r] = h;
            pb[r] = (_Float16) (p - (float) h);
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {   // B: output dimension wp * 64 + nt * 16 + m16 of the tile's rows
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[nt][i] *= tsc[wmma_row(i, hi)];
            const _Float16* vc = tile + wp * 64 + nt * 16 + m16;
            wmma_h16 vb;
#pragma unroll
            for (int r = 0; r < kWR; ++r) vb[r] = vc[r * kWDS];
            acc[nt] = wmma_f16(pa, vb, acc[nt]);
            acc[nt] = wmma_f16(pb, vb, acc[nt]);
        }
        lds_sync();
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int h = wmma_row(i, hi);   // (attn_combine's: the largest of the sink and the scores the base)
            const float mr = m_run[h], M = sinks ? fmaxf(mr, sinks[hb + h]) : mr, f = expf(mr - M);
            const float dn = den[h] * f + (sinks ? expf(sinks[hb + h] - M) : 0.0f);
            out[((size_t) t * H + hb + h) * 512 + wp * 64 + nt * 16 + m16] = dn > 0.0f ? acc[nt][i] * f / dn : 0.0f;
        }
#else
    (void) q, (void) H, (void) cache_v, (void) ring, (void) keys, (void) ldk, (void) nk, (void) scale, (void) sinks;
    (void) out, (void) rb;
#endif
}
// a block a (t, head): the chunks merged with the sink, 512 threads (a value dim each)
__global__ void attn_combine_kernel(int H, int nchunk, int kc, const int* __restrict__ nk,
                                    const float* __restrict__ sinks, const float* __restrict__ ws,
                                    float* __restrict__ out, const Q8Out q8) {
    const int t = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
    const int used = (nk[t] + kc - 1) / kc;
    const float* base = ws + ((size_t) t * H + h) * nchunk * (512 + 2);
    float M = sinks ? sinks[h] : -INFINITY;
    for (int c = 0; c < used; ++c) M = fmaxf(M, base[(size_t) c * 514]);
    float den = sinks ? expf(sinks[h] - M) : 0.0f, num = 0.0f;
    for (int c = 0; c < used; ++c) {
        const float* o = base + (size_t) c * 514;
        if (o[0] == -INFINITY) continue;
        const float f = expf(o[0] - M);
        den += o[1] * f;
        num += o[2 + d] * f;
    }
    const float o = den > 0.0f ? num / den : 0.0f;
    out[((size_t) t * H + h) * 512 + d] = o;
    // (q8_heads' rows for the 512 values a head: row h T + t)
    if (q8.q != nullptr) q8_put(o, ((size_t) h * gridDim.y + t) * 512 + d, q8);
}

// ---------------------------------------------------------------- elementwise
__global__ void sigmoid_mul_kernel(float* x, const float* g, size_t n, const Q8Out q8) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {   // (n a multiple of 32 with q8: a warp's lanes together)
        const float o = x[i] * sigmoidf_(g[i]);
        x[i] = o;
        if (q8.q != nullptr) q8_put(o, i, q8);
    }
}
__global__ void swiglu_kernel(const float* g, const float* u, size_t ldg, int n, float* h, size_t ldh, size_t total,
                              const Q8Out q8) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / n, c = i % n;
    const float gv = g[r * ldg + c];
    const float o = gv / (1.0f + expf(-gv)) * u[r * ldg + c];
    h[r * ldh + c] = o;
    if (q8.q != nullptr) q8_put(o, i, q8);   // (n a multiple of 32: row r's block c / 32)
}
__global__ void add_kernel(float* y, const float* x, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
__global__ void addmul_kernel(float* y, const float* x, const float* w, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i] * w[i];
}

// ---------------------------------------------------------------- the router and the experts' bookkeeping
// a block (1024 threads) a token: n_exp <= 1024 candidates (sigmoid + bias, the index) bitonic-sorted in shared memory
// (larger first, ties: the lower index), the first n_used the picks
__global__ void __launch_bounds__(1024) router_kernel(const float* __restrict__ logits, int n_exp,
                                                      const float* __restrict__ bias, int n_used, int norm, float scale,
                                                      int* __restrict__ ids, float* __restrict__ wout) {
    __shared__ float sv[1024], sp[1024];
    __shared__ int si[1024];
    const int t = blockIdx.x, i = threadIdx.x;
    const float* lr = logits + (size_t) t * n_exp;
    if (i < n_exp) {
        const float p = sigmoidf_(lr[i]);
        sp[i] = p;
        sv[i] = p + (bias ? bias[i] : 0.0f);
        si[i] = i;
    } else {
        sv[i] = -INFINITY;
        si[i] = 0x7fffffff;
    }
    __syncthreads();
    for (int k = 2; k <= 1024; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = i ^ j;
            if (ixj > i) {
                const float a = sv[i], b = sv[ixj];
                const int ia = si[i], ib = si[ixj];
                // "first" in the descending order: larger value, then lower index
                const bool b_first = b > a || (b == a && ib < ia);
                const bool desc = (i & k) == 0;
                if (desc ? b_first : !b_first) {
                    sv[i] = b;
                    sv[ixj] = a;
                    si[i] = ib;
                    si[ixj] = ia;
                }
            }
            __syncthreads();
        }
    if (i < 32) {
        const float p = i < n_used ? sp[si[i]] : 0.0f;
        const float sum = warp_sum(p);
        const float den = norm ? fmaxf(sum, 6.103515625e-5f) : 1.0f;
        if (i < n_used) {
            ids[(size_t) t * n_used + i] = si[i];
            wout[(size_t) t * n_used + i] = p / den * scale;
        }
    }
}
// ... a warp a token: each lane holds every 32nd candidate in registers, n_used rounds of a warp-wide argmax (larger value,
// then the lower index - the sort's order) pick them; the weights summed as above (pick r in lane r)
template <int PER>
__global__ void __launch_bounds__(32) router_warp_kernel(const float* __restrict__ logits, int n_exp,
                                                        const float* __restrict__ bias, int n_used, int norm,
                                                        float scale, int* __restrict__ ids, float* __restrict__ wout) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* lr = logits + (size_t) t * n_exp;
    float v[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = lane + 32 * k;
        v[k] = i < n_exp ? sigmoidf_(lr[i]) + (bias ? bias[i] : 0.0f) : -INFINITY;
    }
    int mine = -1;   // (the pick this lane reports: round `lane`'s)
    for (int r = 0; r < n_used; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int i = lane + 32 * k;
            if (v[k] > bv || (v[k] == bv && i < bi)) {
                bv = v[k];
                bi = i;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if ((bi & 31) == lane) {
#pragma unroll
            for (int k = 0; k < PER; ++k)
                if (lane + 32 * k == bi) v[k] = -INFINITY;
        }
        if (lane == r) mine = bi;
    }
    const float p = lane < n_used ? sigmoidf_(lr[mine]) : 0.0f;
    const float sum = warp_sum(p);
    const float den = norm ? fmaxf(sum, 6.103515625e-5f) : 1.0f;
    if (lane < n_used) {
        ids[(size_t) t * n_used + lane] = mine;
        wout[(size_t) t * n_used + lane] = p / den * scale;
    }
}
// ... 512 candidates at most, 16 picks at most: a thread a candidate, each warp sorting its 32 (a register bitonic
// sort, larger first, then the lower index), the warps' top 16s merged pairwise (16 lists, 8, 4, 2, 1) - the same picks
// in the same order, the weights summed as above
__device__ __forceinline__ bool rk_before(float av, int ai, float bv, int bi) { return av > bv || (av == bv && ai < bi); }
__device__ __forceinline__ void rk_step(float& v, int& i, int j, bool up, int lane) {
    const float pv = __shfl_xor_sync(0xffffffffu, v, j);
    const int pi = __shfl_xor_sync(0xffffffffu, i, j);
    const bool lower = (lane & j) == 0, pb = rk_before(pv, pi, v, i);
    if (lower == up ? pb : !pb) {
        v = pv;
        i = pi;
    }
}
__global__ void __launch_bounds__(512) router_bitonic_kernel(const float* __restrict__ logits, int n_exp,
                                                             const float* __restrict__ bias, int n_used, int norm,
                                                             float scale, int* __restrict__ ids, float* __restrict__ wout) {
    __shared__ float lv[16][16];
    __shared__ int li[16][16];
    const int t = blockIdx.x, e = threadIdx.x, w = e >> 5, lane = e & 31;
    const float* lr = logits + (size_t) t * n_exp;
    float v = e < n_exp ? sigmoidf_(lr[e]) + (bias ? bias[e] : 0.0f) : -INFINITY;
    int i = e < n_exp ? e : 0x7fffffff;
    for (int k = 2; k <= 32; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) rk_step(v, i, j, (lane & k) == 0, lane);
    if (lane < 16) {
        lv[w][lane] = v;
        li[w][lane] = i;
    }
    __syncthreads();
    for (int lists = 16; lists > 1; lists >>= 1) {
        // warp q < lists / 2 merges lists 2q and 2q + 1: the first ascending, the second reversed - a bitonic 32
        float mv = -INFINITY;
        int mi = 0x7fffffff;
        if (w < lists / 2) {
            mv = lane < 16 ? lv[2 * w][lane] : lv[2 * w + 1][31 - lane];
            mi = lane < 16 ? li[2 * w][lane] : li[2 * w + 1][31 - lane];
            for (int j = 16; j > 0; j >>= 1) rk_step(mv, mi, j, true, lane);
        }
        __syncthreads();
        if (w < lists / 2 && lane < 16) {
            lv[w][lane] = mv;
            li[w][lane] = mi;
        }
        __syncthreads();
    }
    if (w == 0) {
        const int pick = lane < n_used ? li[0][lane] : 0;
        const float p = lane < n_used ? sigmoidf_(lr[pick]) : 0.0f;
        const float sum = warp_sum(p);
        const float den = norm ? fmaxf(sum, 6.103515625e-5f) : 1.0f;
        if (lane < n_used) {
            ids[(size_t) t * n_used + lane] = pick;
            wout[(size_t) t * n_used + lane] = p / den * scale;
        }
    }
}
// one thread (a window has at most 16 x 16 entries): counting sort of the entries by expert
__global__ void group_kernel(const int* __restrict__ ids, int T, int n_used, int n_exp, const uint8_t* base,
                             size_t stride, unsigned long long* grp_ptr, int* grp_start, int* grp_ent, int* n_groups,
                             const int* __restrict__ slot, float* __restrict__ w,
                             const unsigned long long* __restrict__ eptr, const int* __restrict__ who) {
    __shared__ int cnt[1024], at[1024];
    // (an entry left out: the CPU lane's - who's flag, else no blob in eptr, else no slot)
    auto out_of = [&](int i, int e) { return who ? who[i] != 0 : eptr ? eptr[e] == 0ull : slot && slot[e] < 0; };
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < n_used; ++j) {
            const int e = ids[t * n_used + j];
            if (out_of(t * n_used + j, e)) w[t * n_used + j] = 0.0f;   // (the CPU lane's)
            else ++cnt[e];
        }
    int g = 0, off = 0;
    for (int e = 0; e < n_exp; ++e) {
        at[e] = off;
        if (cnt[e]) {
            grp_ptr[g] = eptr ? eptr[e] : (unsigned long long) (base + (size_t) (slot ? slot[e] : e) * stride);
            grp_start[g] = off;
            ++g;
            off += cnt[e];
        }
    }
    grp_start[g] = off;
    *n_groups = g;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < n_used; ++j) {
            const int e = ids[t * n_used + j];
            if (!out_of(t * n_used + j, e)) grp_ent[at[e]++] = t * 16 + j;
        }
}
// ... the same groups and the same entry order (by expert, then t 16 + j), a thread an expert: the window's entries
// staged in shared memory, each thread counting and then listing its expert's, a block scan for the offsets and the
// group numbers
__global__ void __launch_bounds__(1024) group_par_kernel(const int* __restrict__ ids, int T, int n_used, int n_exp,
                                                         const uint8_t* base, size_t stride,
                                                         unsigned long long* grp_ptr, int* grp_start, int* grp_ent,
                                                         int* n_groups, const int* __restrict__ slot,
                                                         float* __restrict__ w,
                                                         const unsigned long long* __restrict__ eptr,
                                                         const int* __restrict__ who) {
    __shared__ int ent[16 * 16];
    __shared__ int sc[1024], sg[1024];
    const int e = threadIdx.x, n_ent = T * n_used;
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x) {
        const int x = ids[i];
        const bool out = who ? who[i] != 0 : eptr ? eptr[x] == 0ull : slot && slot[x] < 0;
        if (out) w[i] = 0.0f;   // (not this GPU's)
        ent[i] = out ? -1 : x;
    }
    __syncthreads();
    int c = 0;
    if (e < n_exp)
        for (int i = 0; i < n_ent; ++i) c += ent[i] == e;
    sc[e] = c;
    sg[e] = c > 0;
    __syncthreads();
    // inclusive scans of the counts and of the non-empty flags (Hillis-Steele over the block)
    for (int o = 1; o < 1024; o <<= 1) {
        const int a = e >= o ? sc[e - o] : 0, b = e >= o ? sg[e - o] : 0;
        __syncthreads();
        sc[e] += a;
        sg[e] += b;
        __syncthreads();
    }
    const int off = sc[e] - c, gi = sg[e] - (c > 0);
    if (e < n_exp && c > 0) {
        grp_ptr[gi] = eptr ? eptr[e] : (unsigned long long) (base + (size_t) (slot ? slot[e] : e) * stride);
        grp_start[gi] = off;
        int k = off;
        for (int i = 0; i < n_ent; ++i)
            if (ent[i] == e) grp_ent[k++] = (i / n_used) * 16 + i % n_used;
    }
    if (e == 1023 || e == n_exp - 1) {
        if (e == n_exp - 1) {
            grp_start[sg[e]] = sc[e];
            *n_groups = sg[e];
        }
    }
}
// ... group_par_kernel's groups with few barriers: the entries count themselves into their experts' cells (an atomic
// add an entry, where every expert's thread compared itself with every entry), and the two scans go warp by warp over
// the lanes (shuffles), the warps' totals by one warp, where Hillis-Steele over the block took two barriers a step, ten
// steps.  The same counts, offsets, group numbers and entry order.  A block of 1024 threads
__global__ void __launch_bounds__(1024) group_par2_kernel(const int* __restrict__ ids, int T, int n_used, int n_exp,
                                                          const uint8_t* base, size_t stride,
                                                          unsigned long long* grp_ptr, int* grp_start, int* grp_ent,
                                                          int* n_groups, const int* __restrict__ slot,
                                                          float* __restrict__ w,
                                                          const unsigned long long* __restrict__ eptr,
                                                          const int* __restrict__ who) {
    __shared__ int ent[16 * 16];
    __shared__ int sc[1024];
    __shared__ int wt[32], wf[32], wo[32], wg[32];
    const int e = threadIdx.x, n_ent = T * n_used, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    sc[e] = 0;
    int mine = -1;
    if (e < n_ent) {
        const int x = ids[e];
        const bool out = who ? who[e] != 0 : eptr ? eptr[x] == 0ull : slot && slot[x] < 0;
        if (out) w[e] = 0.0f;   // (not this GPU's)
        mine = out ? -1 : x;
        ent[e] = mine;
    }
    __syncthreads();
    if (mine >= 0) atomicAdd(&sc[mine], 1);
    __syncthreads();
    const int c = e < n_exp ? sc[e] : 0;
    int v = c, f = c > 0 ? 1 : 0;   // (inclusive scans over the warp's lanes)
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int a = __shfl_up_sync(0xffffffffu, v, o), b = __shfl_up_sync(0xffffffffu, f, o);
        if (lane >= o) {
            v += a;
            f += b;
        }
    }
    if (lane == 31) {
        wt[warp] = v;
        wf[warp] = f;
    }
    __syncthreads();
    if (warp == 0) {   // (the warps before each warp)
        const int t0 = wt[lane], f0 = wf[lane];
        int tv = t0, fv = f0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int a = __shfl_up_sync(0xffffffffu, tv, o), b = __shfl_up_sync(0xffffffffu, fv, o);
            if (lane >= o) {
                tv += a;
                fv += b;
            }
        }
        wo[lane] = tv - t0;
        wg[lane] = fv - f0;
    }
    __syncthreads();
    const int Sc = v + wo[warp], Sg = f + wg[warp];
    const int off = Sc - c, gi = Sg - (c > 0 ? 1 : 0);
    if (e < n_exp && c > 0) {
        grp_ptr[gi] = eptr ? eptr[e] : (unsigned long long) (base + (size_t) (slot ? slot[e] : e) * stride);
        grp_start[gi] = off;
        int k = off;
        for (int i = 0; i < n_ent; ++i)
            if (ent[i] == e) grp_ent[k++] = (i / n_used) * 16 + i % n_used;
    }
    if (e == n_exp - 1) {
        grp_start[Sg] = Sc;
        *n_groups = Sg;
    }
}
__global__ void plan_kernel(const int* ids, int n_used, const uint8_t* base, size_t stride, unsigned long long* plan,
                            const int* __restrict__ slot, float* __restrict__ w,
                            const unsigned long long* __restrict__ eptr, const int* __restrict__ who) {
    const int j = threadIdx.x;
    if (j >= n_used) return;
    if (eptr) {   // (the GPU fetch: the blob where it lies, or the CPU lane's)
        const unsigned long long p = who && who[j] != 0 ? 0ull : eptr[ids[j]];
        plan[j] = p;
        if (p == 0ull) w[j] = 0.0f;
        return;
    }
    const int s = slot ? slot[ids[j]] : ids[j];
    plan[j] = s >= 0 ? (unsigned long long) (base + (size_t) s * stride) : 0ull;   // (0: the CPU lane's)
    if (s < 0) w[j] = 0.0f;
}
__global__ void moe_sum_kernel(const float* __restrict__ part, int n_used, int n, const float* __restrict__ w,
                               float* __restrict__ out, int stride16, const Q8Out q8) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    for (int j = 0; j < n_used; ++j) {
        const float wj = w[(size_t) t * n_used + j];
        if (wj != 0.0f) v = fmaf(wj, part[((size_t) t * stride16 + j) * n + i], v);   // (0: the CPU lane's entry)
    }
    out[(size_t) t * n + i] = v;
    if (q8.q != nullptr) q8_put(v, (size_t) t * n + i, q8);   // (n a multiple of 32: a warp's lanes together)
}
__global__ void dup_q8_kernel(const int8_t* q, const float* d, int k, int8_t* q2, float* d2) {
    const int r = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        const int8_t v = q[(size_t) r * k + i];
        q2[(size_t) (2 * r) * k + i] = v;
        q2[(size_t) (2 * r + 1) * k + i] = v;
        if (i < k / 32) {
            const float dv = d[(size_t) r * (k / 32) + i];
            d2[(size_t) (2 * r) * (k / 32) + i] = dv;
            d2[(size_t) (2 * r + 1) * (k / 32) + i] = dv;
        }
    }
}

// ---------------------------------------------------------------- the prompt path's attention (around two GEMMs)
// a block (256 threads) a (token, head) row
__global__ void __launch_bounds__(256) attn_softmax_kernel(const float* __restrict__ S, size_t ldS, int H, int nkr,
                                                           int k0, const int* __restrict__ pos, int window,
                                                           const float* __restrict__ mask, size_t ldm, float scale,
                                                           const float* __restrict__ sinks, __half* __restrict__ P,
                                                           size_t ldP) {
    __shared__ float sh[32];
    const int r = blockIdx.x, t = r / H, h = r % H;
    const int p = pos[t];
    const float* sr = S + (size_t) r * ldS;
    __half* pr = P + (size_t) r * ldP;
    const int lo = window > 0 ? max(0, p - window + 1) : 0;
    auto seen = [&](int j) {
        if (window < 0) return mask[(size_t) t * ldm + j] == 0.0f;   // (the columns a list of keys: the mask alone)
        const int q = k0 + j;
        if (q > p || q < lo) return false;
        return window > 0 || mask[(size_t) t * ldm + j] == 0.0f;
    };
    float m = sinks ? sinks[h] : -INFINITY;
    for (int j = threadIdx.x; j < nkr; j += blockDim.x)
        if (seen(j)) m = fmaxf(m, sr[j] * scale);
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : -INFINITY;
        v = warp_max(v);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    m = sh[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < nkr; j += blockDim.x)
        if (seen(j)) sum += expf(sr[j] * scale - m);
    sum = block_sum(sum) + (sinks ? expf(sinks[h] - m) : 0.0f);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int j = threadIdx.x; j < nkr; j += blockDim.x)
        pr[j] = __float2half_rn(seen(j) ? expf(sr[j] * scale - m) * inv : 0.0f);
}
__global__ void dsa_mask_kernel(const int* __restrict__ keys, int ldk, const int* __restrict__ nk, int k0, int nkr,
                                float* __restrict__ mask, size_t ldm) {
    const int t = blockIdx.x;
    float* mr = mask + (size_t) t * ldm;
    for (int j = threadIdx.x; j < nkr; j += blockDim.x) mr[j] = -INFINITY;
    __syncthreads();
    for (int i = threadIdx.x; i < nk[t]; i += blockDim.x) {
        const int j = keys[(size_t) t * ldk + i] - k0;
        if (j >= 0 && j < nkr) mr[j] = 0.0f;
    }
}
// ... over the columns of a list of pools (col[g]: pool g's rank in it; a pool kpool columns): ncol columns
__global__ void dsa_mask_list_kernel(const int* __restrict__ keys, int ldk, const int* __restrict__ nk,
                                     const int* __restrict__ col, int kpool, int ncol, float* __restrict__ mask,
                                     size_t ldm) {
    const int t = blockIdx.x;
    float* mr = mask + (size_t) t * ldm;
    for (int j = threadIdx.x; j < ncol; j += blockDim.x) mr[j] = -INFINITY;
    __syncthreads();
    for (int i = threadIdx.x; i < nk[t]; i += blockDim.x) {
        const int p = keys[(size_t) t * ldk + i];
        mr[col[p / kpool] * kpool + p % kpool] = 0.0f;
    }
}
// one block (1024 threads): the pools the keys of T tokens name (a token's keys whole pools, then its own pool's first
// positions: every kpool-th key names one) -> col[g] = pool g's rank among them (-1: not named), list[rank] = g, *n
__global__ void __launch_bounds__(1024) ix_union_kernel(const int* __restrict__ keys, int ldk,
                                                        const int* __restrict__ nk, int T, int kpool, int n_pool,
                                                        int* __restrict__ col, int* __restrict__ list, int* n) {
    __shared__ int cnt[1024];
    const int nth = blockDim.x, tid = threadIdx.x;
    for (int g = tid; g < n_pool; g += nth) col[g] = 0;
    __syncthreads();
    const int per_t = (ldk + kpool - 1) / kpool;
    for (int e = tid; e < T * per_t; e += nth) {
        const int t = e / per_t, i = (e % per_t) * kpool;
        if (i < nk[t]) col[keys[(size_t) t * ldk + i] / kpool] = 1;
    }
    __syncthreads();
    const int per = (n_pool + nth - 1) / nth, b0 = min(n_pool, tid * per), b1 = min(n_pool, b0 + per);
    int c = 0;
    for (int g = b0; g < b1; ++g) c += col[g];
    cnt[tid] = c;
    __syncthreads();
    if (tid == 0) {
        int a = 0;
        for (int i = 0; i < nth; ++i) {
            const int x = cnt[i];
            cnt[i] = a;
            a += x;
        }
        *n = a;
    }
    __syncthreads();
    int r = cnt[tid];
    for (int g = b0; g < b1; ++g) {
        if (col[g]) {
            col[g] = r;
            list[r++] = g;
        } else {
            col[g] = -1;
        }
    }
}
// Q8_0 rows (rb bytes) -> F16 rows of kv_lora + n_rot values, a thread a value pair
// (list: row j the position list[j / kpool] kpool + j % kpool - a list of pools' rows; zeros from position kend on)
__global__ void kv_deq_kernel(const uint8_t* __restrict__ cache, int ring, int kv_lora, int n_rot, int rb, int k0, int n,
                              __half2* __restrict__ dst, const int* __restrict__ list, int kpool,
                              int kend, const int* __restrict__ tag, const uint8_t* __restrict__ host,
                              const int* __restrict__ hidx) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int half_row = (kv_lora + n_rot) / 2;
    if (i >= (size_t) n * half_row) return;
    const int j = (int) (i / half_row), c = (int) (i % half_row);
    const int q = list ? list[j / kpool] * kpool + j % kpool : k0 + j;
    if (list && q >= kend) {
        dst[i] = __floats2half2_rn(0.0f, 0.0f);
        return;
    }
    bool fh = false;
    const size_t r = kv_src(q, ring, tag, hidx, fh);
    const uint8_t* rp = (fh ? host : cache) + r * rb;
    if (2 * c < kv_lora) {
        const char2 qv = *(const char2*) (rp + 2 * c);
        const float d = __half2float(*(const __half*) (rp + kv_lora + 2 * (c >> 4)));
        dst[i] = __floats2half2_rn(d * (float) qv.x, d * (float) qv.y);
    } else {
        dst[i] = *(const __half2*) (rp + kv_lora + kv_lora / 16 + 2 * (2 * c - kv_lora));
    }
}
__global__ void kv_gather_kernel(const uint4* __restrict__ cache, int ring, int row16, int k0, int n,
                                 uint4* __restrict__ dst, const int* __restrict__ list, int kpool,
                                 int kend, const int* __restrict__ tag, const uint4* __restrict__ host,
                                 const int* __restrict__ hidx) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * row16) return;
    const int j = (int) (i / row16), c = (int) (i % row16);
    const int q = list ? list[j / kpool] * kpool + j % kpool : k0 + j;
    if (list && q >= kend) {
        dst[i] = make_uint4(0, 0, 0, 0);
        return;
    }
    bool fh = false;
    const size_t r = kv_src(q, ring, tag, hidx, fh);
    dst[i] = (fh ? host : cache)[r * row16 + c];
}
// a block row a token: its picked rows (row16 uint4s each) side by side
__global__ void kv_pick_kernel(const uint4* __restrict__ cache, int ring, int row16, const int* __restrict__ keys,
                               int ldk, const int* __restrict__ nk, uint4* __restrict__ dst,
                               const int* __restrict__ tag, const uint4* __restrict__ host,
                               const int* __restrict__ hidx) {
    const int t = blockIdx.y;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int j = (int) (i / row16), c = (int) (i % row16);
    if (j >= nk[t]) return;
    bool fh = false;
    const size_t r = kv_src(keys[(size_t) t * ldk + j], ring, tag, hidx, fh);
    dst[((size_t) t * ldk + j) * row16 + c] = (fh ? host : cache)[r * row16 + c];
}

// ---------------------------------------------------------------- the prompt path
__global__ void to_f16_kernel(const float* __restrict__ src, size_t lds, int n, __half* __restrict__ dst, size_t ldd,
                              size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / n, c = i % n;
    dst[r * ldd + c] = __float2half_rn(src[r * lds + c]);
}
// The same conversions a row at a time (grid y: the rows, a stride gridDim.y), 4 values a thread - for AMD, where the
// one-value kernel's 64-bit division a value made it slower than the memory (the prompt's queries a sub-batch, the
// GEMMs' inputs).  Rows 16-byte aligned in src, 8 in dst; the same values written.
__global__ void to_f16_rows_kernel(const float* __restrict__ src, size_t lds, int n, __half* __restrict__ dst, size_t ldd,
                                   int rows) {
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (c >= n) return;
    for (int r = (int) blockIdx.y; r < rows; r += (int) gridDim.y) {
        const float* sp = src + (size_t) r * lds + c;
        __half* dp = dst + (size_t) r * ldd + c;
        if (c + 4 <= n) {
            const float4 v = *(const float4*) sp;
            *(__half2*) dp = __floats2half2_rn(v.x, v.y);
            *(__half2*) (dp + 2) = __floats2half2_rn(v.z, v.w);
        } else {
            for (int j = 0; c + j < n; ++j) dp[j] = __float2half_rn(sp[j]);
        }
    }
}
// (the row kernel's threads a block: a row's 4-value groups rounded up to a warp, 256 at most)
inline int rows_threads(int n) { return std::min(256, ((n + 3) / 4 + 31) / 32 * 32); }
// (whether the row kernel takes these operands: AMD only - NVIDIA keeps the one-value kernel)
inline bool f16_rows_form(const void* src, size_t lds, const void* dst, size_t ldd) {
#if defined(MAYA_USE_HIP)
    return ((uintptr_t) src & 15) == 0 && lds % 4 == 0 && ((uintptr_t) dst & 7) == 0 && ldd % 4 == 0;
#else
    (void) src, (void) lds, (void) dst, (void) ldd;
    return false;
#endif
}
// one block: the counts, their scan, the scatter (the order within an expert does not change a row's sums)
__global__ void __launch_bounds__(1024) moe_sort_kernel(const int* __restrict__ ids, int n_ent, int n_exp,
                                                        int* __restrict__ bounds, int* __restrict__ tok,
                                                        int* __restrict__ row_of, int n_used,
                                                        const int* __restrict__ slot, float* __restrict__ w) {
    __shared__ int cnt[1024];
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x) {
        const int s = slot ? slot[ids[i]] : ids[i];
        if (s >= 0) atomicAdd(&cnt[s], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int off = 0;
        for (int e = 0; e < n_exp; ++e) {
            const int c = cnt[e];
            bounds[e] = off;
            cnt[e] = off;
            off += c;
        }
        bounds[n_exp] = off;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x) {
        const int s = slot ? slot[ids[i]] : ids[i];
        if (s < 0) {   // (not in this set: no row here)
            row_of[i] = -1;
            if (w) w[i] = 0.0f;
            continue;
        }
        const int r = atomicAdd(&cnt[s], 1);
        tok[r] = i / n_used;
        row_of[i] = r;
    }
#if defined(MAYA_USE_HIP)
    // AMD: the scatter's order within an expert is the order its threads' atomics land in, which differs from one
    // run to the next there - and the expert kernels' sums of a row depend a little on the rows beside it (their
    // tiles), so two starts gave two answers.  Each expert's rows put in token order (a token routes to an expert
    // once: its rank among the expert's tokens is its row) - the same rows every run
    __syncthreads();
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x) {
        const int s = slot ? slot[ids[i]] : ids[i];
        if (s < 0) continue;
        const int t = i / n_used, b0 = bounds[s], b1 = cnt[s];   // (cnt: each expert's end after the scatter)
        int rank = 0;
        for (int r = b0; r < b1; ++r) rank += tok[r] < t ? 1 : 0;
        row_of[i] = b0 + rank;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x)
        if (row_of[i] >= 0) tok[row_of[i]] = i / n_used;
#endif
}
__global__ void moe_combine_kernel(const float* __restrict__ y, const int* __restrict__ row_of,
                                   const float* __restrict__ w, int n_used, int n, float* __restrict__ out, int acc) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    for (int j = 0; j < n_used; ++j) {
        const int r = row_of[(size_t) t * n_used + j];
        if (r >= 0) v = fmaf(w[(size_t) t * n_used + j], y[(size_t) r * n + i], v);
    }
    out[(size_t) t * n + i] = acc ? out[(size_t) t * n + i] + v : v;
}

// ---------------------------------------------------------------- a draft's top token
__global__ void __launch_bounds__(1024) top1_kernel(const float* __restrict__ row, int n, float* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (row[i] > best) best = row[i], bi = i;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ob > best || (ob == best && oi < bi)) best = ob, bi = oi;
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) sv[w] = best, si[w] = bi;
    __syncthreads();
    if (w == 0) {
        best = l < (int) (blockDim.x >> 5) ? sv[l] : -INFINITY;
        bi = l < (int) (blockDim.x >> 5) ? si[l] : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) best = ob, bi = oi;
        }
        if (l == 0) sv[0] = best, si[0] = bi;
    }
    __syncthreads();
    const float m = sv[0];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += expf(row[i] - m);
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        out[0] = __int_as_float(si[0]);
        out[1] = 1.0f / sum;
    }
}

// ... over several blocks: each its slice's top value (ties: the lower id) and sum of exp(v - its top), the last block to
// finish combining them in block order (one launch)
constexpr int kTop1Blocks = 64;
__device__ unsigned g_top1_done = 0;
__device__ float g_top1_part[kTop1Blocks * 3];
// (PART: a part of a row - its top value, that one's index and its sum of exp(v - top), for the whole row's to be made
//  of its parts': out[0..2])
template <bool PART>
__global__ void __launch_bounds__(256) top1_multi_kernel(const float* __restrict__ row, int n, float* __restrict__ out) {
    __shared__ float sv[8];
    __shared__ int si[8];
    __shared__ bool last;
    const int per = (n + gridDim.x - 1) / gridDim.x, a0 = blockIdx.x * per, a1 = min(n, a0 + per);
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = a0 + threadIdx.x; i < a1; i += blockDim.x)
        if (row[i] > best) best = row[i], bi = i;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ob > best || (ob == best && oi < bi)) best = ob, bi = oi;
    }
    if (l == 0) sv[w] = best, si[w] = bi;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < 8; ++k)
            if (sv[k] > sv[0] || (sv[k] == sv[0] && si[k] < si[0])) sv[0] = sv[k], si[0] = si[k];
    }
    __syncthreads();
    const float m = sv[0];
    float sum = 0.0f;
    for (int i = a0 + threadIdx.x; i < a1; i += blockDim.x) sum += expf(row[i] - m);
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        g_top1_part[blockIdx.x * 3 + 0] = m;
        g_top1_part[blockIdx.x * 3 + 1] = __int_as_float(si[0]);
        g_top1_part[blockIdx.x * 3 + 2] = sum;
        __threadfence();
        last = atomicAdd(&g_top1_done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x != 0) return;
    __threadfence();
    float M = -INFINITY;
    int I = 0x7fffffff;
    for (int b = 0; b < (int) gridDim.x; ++b) {
        const float v = ((volatile float*) g_top1_part)[b * 3];
        const int ix = __float_as_int(((volatile float*) g_top1_part)[b * 3 + 1]);
        if (v > M || (v == M && ix < I)) M = v, I = ix;
    }
    float S = 0.0f;
    for (int b = 0; b < (int) gridDim.x; ++b) {
        const float v = ((volatile float*) g_top1_part)[b * 3];
        if (v > -INFINITY) S += ((volatile float*) g_top1_part)[b * 3 + 2] * expf(v - M);
    }
    if constexpr (PART) {
        out[0] = M;
        out[1] = __int_as_float(I);
        out[2] = S;
        out[3] = 0.0f;
    } else {
        out[0] = __int_as_float(I);
        out[1] = 1.0f / S;
    }
    g_top1_done = 0;
}

// ---------------------------------------------------------------- Engram
__device__ __forceinline__ float f8e4m3(uint8_t x) {
    const int ax = x & 0x7f;
    if (ax == 0x7f) return __int_as_float(0x7fc00000);
    const int e = (ax >> 3) & 0xf, m = ax & 7;
    const float v = e == 0 ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
    return (x & 0x80) ? -v : v;
}
// a block (dim threads, dim <= 1024) a (column, token)
__global__ void engram_rows_kernel(const uint8_t* __restrict__ rows, const float* __restrict__ valid, int cols, int dim,
                                   const float* __restrict__ ln, float eps, float* __restrict__ r,
                                   const float* __restrict__ scale) {
    const int col = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    float v = i < dim ? f8e4m3(rows[((size_t) t * cols + col) * dim + i]) : 0.0f;
    if (scale != nullptr) v *= scale[(size_t) t * cols + col];
    const float ss = block_sum(v * v);
    const float rr = 1.0f / sqrtf(ss / (float) dim + eps);
    if (i < dim) r[((size_t) t * cols + col) * dim + i] = v * rr * ln[(size_t) col * dim + i] * valid[t];
}
__global__ void engram_mix_kernel(float* __restrict__ x, const float* __restrict__ kv, const float* __restrict__ kv_b,
                                  int hc, int E, const float* __restrict__ egate, const float* __restrict__ gamma,
                                  const float* __restrict__ hcs, const float* __restrict__ valid, float eps) {
    __shared__ float s_gate[8];
    __shared__ float s_rv;
    const int t = blockIdx.x;
    if (valid[t] == 0.0f) return;   // (uniform: the whole block)
    const float* kt = kv + (size_t) t * (hc + 1) * E;
    float* xt = x + (size_t) t * hc * E;
    for (int s = 0; s < hc; ++s) {
        float sx = 0.0f, sk = 0.0f, sd = 0.0f;
        for (int d = threadIdx.x; d < E; d += blockDim.x) {
            const float xv = xt[(size_t) s * E + d], kvv = kt[(size_t) s * E + d] + kv_b[(size_t) s * E + d];
            sx += xv * xv;
            sk += kvv * kvv;
            sd += xv * kvv * egate[(size_t) s * E + d];
        }
        sx = block_sum(sx);
        sk = block_sum(sk);
        sd = block_sum(sd);
        if (threadIdx.x == 0) {
            const float rx = 1.0f / sqrtf(sx / (float) E + eps), rk = 1.0f / sqrtf(sk / (float) E + eps);
            const float dot = sd * rx * rk / sqrtf((float) E);
            const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
            s_gate[s] = sigmoidf_(dot < 0.0f ? -mag : mag);
        }
    }
    float sv = 0.0f;
    for (int d = threadIdx.x; d < E; d += blockDim.x) {
        const float v = kt[(size_t) hc * E + d] + kv_b[(size_t) hc * E + d];
        sv += v * v;
    }
    sv = block_sum(sv);
    if (threadIdx.x == 0) s_rv = 1.0f / sqrtf(sv / (float) E + eps);
    __syncthreads();
    for (int d = threadIdx.x; d < E; d += blockDim.x) {
        const float sig = (kt[(size_t) hc * E + d] + kv_b[(size_t) hc * E + d]) * s_rv * gamma[d];
        for (int s = 0; s < hc; ++s) xt[(size_t) s * E + d] += sig * (s_gate[s] + hcs[(size_t) s * E + d]);
    }
}

// ---------------------------------------------------------------- the indexer
// a block (dim threads) a new group
template <typename P, typename V>   // (the pooled keys' and the value | gate ring's types: float, or __half with --kv q8)
__global__ void ix_pool_kernel(const V* __restrict__ ivg, int ring, int g0, int kpool, int dim,
                               const float* __restrict__ ape, const float* __restrict__ w, float eps, int n_rot,
                               float base, P* __restrict__ pooled) {
    __shared__ float sv[1024];
    const int g = g0 + blockIdx.x, c = threadIdx.x;
    float lg[8], mx = -INFINITY;
    for (int k = 0; k < kpool; ++k) {
        const int p = g * kpool + k;
        lg[k] = (float) ivg[(size_t) (p % ring) * 2 * dim + dim + c] + ape[(size_t) k * dim + c];
        mx = fmaxf(mx, lg[k]);
    }
    float den = 0.0f, num = 0.0f;
    for (int k = 0; k < kpool; ++k) {
        const int p = g * kpool + k;
        const float e = expf(lg[k] - mx);
        den += e;
        num += e * (float) ivg[(size_t) (p % ring) * 2 * dim + c];
    }
    const float v = num / den;
    const float ss = block_sum(v * v);
    sv[c] = v / sqrtf(ss / (float) dim + eps) * w[c];
    __syncthreads();
    const int nope = dim - n_rot;
    if (c >= nope && ((c - nope) & 1) == 0) {
        float a = sv[c], b = sv[c + 1];
        rope_pair(a, b, (float) (g * kpool), (c - nope) / 2, n_rot, base);
        sv[c] = a;
        sv[c + 1] = b;
    }
    __syncthreads();
    pooled[(size_t) g * dim + c] = (P) sv[c];
}
// grid (ceil(n_pool / 256), T): a thread a pool
__device__ __forceinline__ float4 pooled4(const float* p, int i) { return ((const float4*) p)[i]; }
__device__ __forceinline__ float4 pooled4(const __half* p, int i) {
    const uint2 u = ((const uint2*) p)[i];
    const float2 a = __half22float2(*(const __half2*) &u.x), b = __half22float2(*(const __half2*) &u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
// (groups i and i + 1 - 8 values - at once)
__device__ __forceinline__ void pooled8(const float* p, int i, float4& a, float4& b) {
    a = ((const float4*) p)[i];
    b = ((const float4*) p)[i + 1];
}
__device__ __forceinline__ void pooled8(const __half* p, int i, float4& a, float4& b) {
    const uint4 u = ((const uint4*) p)[i / 2];
    const float2 a0 = __half22float2(*(const __half2*) &u.x), a1 = __half22float2(*(const __half2*) &u.y);
    const float2 b0 = __half22float2(*(const __half2*) &u.z), b1 = __half22float2(*(const __half2*) &u.w);
    a = make_float4(a0.x, a0.y, a1.x, a1.y);
    b = make_float4(b0.x, b0.y, b1.x, b1.y);
}
template <typename P>
__global__ void __launch_bounds__(256) ix_score_kernel(const float* __restrict__ q, const float* __restrict__ wt,
                                                       int heads, int dim, const P* __restrict__ pooled,
                                                       const int* __restrict__ pos, int kpool, int n_pool,
                                                       float* __restrict__ score, size_t ld) {
    extern __shared__ float sq[];   // [heads][dim], then wt[heads]
    const int t = blockIdx.y, g = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadIdx.x; i < heads * dim; i += blockDim.x) sq[i] = q[(size_t) t * heads * dim + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) sq[heads * dim + i] = wt[(size_t) t * heads + i];
    __syncthreads();
    if (g >= n_pool) return;
    const int nv = pos[t] / kpool;
    float out = -INFINITY;
    if (g < nv) {
        float acc[16];
#pragma unroll
        for (int h = 0; h < 16; ++h) acc[h] = 0.0f;
        const P* pk = pooled + (size_t) g * dim;
        for (int i2 = 0; i2 < dim / 4; i2 += 2) {   // (two groups of 4 values a pass: F16 keys in one 16-byte load)
            float4 k2[2];
            pooled8(pk, i2, k2[0], k2[1]);
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const float4 kk = k2[u];
                const int i = i2 + u;
#pragma unroll
                for (int h = 0; h < 16; ++h)
                    if (h < heads) {
                        const float* qh = sq + h * dim + 4 * i;
                        acc[h] = fmaf(qh[0], kk.x, fmaf(qh[1], kk.y, fmaf(qh[2], kk.z, fmaf(qh[3], kk.w, acc[h]))));
                    }
            }
        }
        out = 0.0f;
#pragma unroll
        for (int h = 0; h < 16; ++h)
            if (h < heads) out = fmaf(fmaxf(acc[h], 0.0f), sq[heads * dim + h], out);
    }
    score[(size_t) t * ld + g] = out;
}
// The same scores on an RDNA GPU's matrix cores (16 heads of 128 dimensions): maya-ds4's WMMA indexer pattern on these
// pools - a warp a token, its 16 heads' query split into an F16 high and a low part held as A (8 k-steps); the pools
// in tiles of 64 staged into shared memory as F16 for the block's 8 tokens (the next tile's loads in flight under this
// one's products), B = 16 of them; a lane's 8 heads' products of its pool, relu, the head weights, the other half of
// the wave's 8 (lane ^ 16).  Grid (ceil(n_pool / kIxPB), ceil(T / 8)).  The pools rounded to F16 (an 8-bit cache's are
// F16 already), the sums in another order: past ix_gemm_from on the prompt path alone (as ix_gemm's GEMMs).
constexpr int kIxTile = 64, kIxPB = 512, kIxLD = 128 + 8;
template <typename P>
__global__ void __launch_bounds__(256) ix_score_wmma_kernel(const float* __restrict__ q, const float* __restrict__ wt,
                                                            int T, const P* __restrict__ pooled,
                                                            const int* __restrict__ pos, int kpool, int n_pool,
                                                            float* __restrict__ score, size_t ld) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    __shared__ __align__(16) _Float16 ks[kIxTile * kIxLD];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, m16 = lane & 15, hi = lane >> 4;
    const int t = blockIdx.y * 8 + wp, g0 = blockIdx.x * kIxPB, g1 = min(n_pool, g0 + kIxPB);
    const bool live = t < T;
    const int nv = live ? pos[t] / kpool : 0;
    wmma_h16 qa[8], qb[8];
    float w8[8];
    if (live) {
        const float* qh = q + ((size_t) t * 16 + m16) * 128;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const float v = qh[kk * 16 + e];
                const _Float16 h = (_Float16) v;
                qa[kk][e] = h;
                qb[kk][e] = (_Float16) (v - (float) h);
            }
#pragma unroll
        for (int i = 0; i < 8; ++i) w8[i] = wt[(size_t) t * 16 + wmma_row(i, hi)];
    }
    // the staging: a tile's 64 pools x 16 groups of 8 values - thread x's groups x, x + 256, x + 512, x + 768
    float4 nx[4][2];
    auto fetch = [&](int f0) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int i = (int) threadIdx.x + 256 * k, r = i >> 4, c = i & 15;
            if (f0 + r < g1) pooled8(pooled + (size_t) (f0 + r) * 128, 2 * c, nx[k][0], nx[k][1]);
            else nx[k][0] = nx[k][1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto stage = [&]() {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int i = (int) threadIdx.x + 256 * k, r = i >> 4, c = i & 15;
            const float v[8] = {nx[k][0].x, nx[k][0].y, nx[k][0].z, nx[k][0].w,
                                nx[k][1].x, nx[k][1].y, nx[k][1].z, nx[k][1].w};
            _Float16 o[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = (_Float16) v[e];
            *(uint4*) (ks + r * kIxLD + 8 * c) = *(const uint4*) o;
        }
    };
    fetch(g0);
    for (int f0 = g0; f0 < g1; f0 += kIxTile) {
        stage();
        lds_sync();
        if (f0 + kIxTile < g1) fetch(f0 + kIxTile);
        if (live) {
#pragma unroll
            for (int sub = 0; sub < kIxTile / 16; ++sub) {
                const int g = f0 + sub * 16 + m16;
                if (f0 + sub * 16 >= g1) break;
                wmma_f8 c = {};
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {   // B: pool g's dimensions kk * 16 .. + 15
                    const uint4* kr = (const uint4*) (ks + (sub * 16 + m16) * kIxLD + kk * 16);
                    const uint4 x0 = kr[0], x1 = kr[1];
                    const uint32_t u[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                    wmma_h16 kb;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        kb[2 * e] = __builtin_bit_cast(_Float16, (unsigned short) (u[e] & 0xffffu));
                        kb[2 * e + 1] = __builtin_bit_cast(_Float16, (unsigned short) (u[e] >> 16));
                    }
                    c = wmma_f16(qa[kk], kb, c);
                    c = wmma_f16(qb[kk], kb, c);
                }
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) s = fmaf(fmaxf(c[i], 0.0f), w8[i], s);
                s += __shfl_xor_sync(0xffffffffu, s, 16);
                if (hi == 0 && g < g1) score[(size_t) t * ld + g] = g < nv ? s : -INFINITY;
            }
        }
        lds_sync();
    }
#else
    (void) q, (void) wt, (void) T, (void) pooled, (void) pos, (void) kpool, (void) n_pool, (void) score, (void) ld;
#endif
}
// The scores from the heads' dot products a GEMM left in S (column (t, h), row g - g0; ldS apart): grid
// (ceil(n / 256), T), a thread a pool - the heads summed in ix_score_kernel's order
__global__ void ix_reduce_kernel(const float* __restrict__ S, size_t ldS, const float* __restrict__ wt, int heads,
                                 const int* __restrict__ pos, int kpool, int g0, int n, float* __restrict__ score,
                                 size_t ld) {
    const int t = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int g = g0 + j, nv = pos[t] / kpool;
    float out = -INFINITY;
    if (g < nv) {
        out = 0.0f;
        for (int h = 0; h < heads; ++h)
            out = fmaf(fmaxf(S[(size_t) (t * heads + h) * ldS + j], 0.0f), wt[(size_t) t * heads + h], out);
    }
    score[(size_t) t * ld + g] = out;
}
__device__ __forceinline__ uint32_t ord_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// a block (1024 threads) a token: the top `top` of its nv scores (radix select on the order-preserving keys, then the
// picks in index order - the ties at the threshold the lowest indices), their kpool positions each, then the tail
__global__ void __launch_bounds__(1024) ix_select_kernel(const float* __restrict__ score, size_t ld,
                                                         const int* __restrict__ pos, int kpool, int top,
                                                         int* __restrict__ keys, int ldk, int* __restrict__ nk) {
    __shared__ int hist[256];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_need;
    __shared__ int s_gt[32], s_eq[32];   // (the warps' totals, then their offsets)
    const int t = blockIdx.x, nth = blockDim.x, tid = threadIdx.x;
    const int p = pos[t], nv = p / kpool;
    const float* sc = score + (size_t) t * ld;
    int* kt = keys + (size_t) t * ldk;
    const int nsel = min(nv, top);
    uint32_t thr = 0;   // the threshold key: every key > thr picked, then ties at thr
    int take_eq = 0;    // ... how many of the ties
    if (nv > top) {
        if (tid == 0) {
            s_prefix = 0;
            s_mask = 0;
            s_need = top;
        }
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += nth) hist[i] = 0;
            __syncthreads();
            const uint32_t pre = s_prefix, msk = s_mask;
            for (int g = tid; g < nv; g += nth) {
                const uint32_t kk = ord_key(sc[g]);
                if ((kk & msk) == pre) atomicAdd(&hist[(kk >> shift) & 255], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int need = s_need, b = 255;
                for (; b > 0; --b) {
                    if (hist[b] >= need) break;
                    need -= hist[b];
                }
                s_need = need;
                s_prefix = pre | ((uint32_t) b << shift);
                s_mask = msk | (255u << shift);
            }
            __syncthreads();
        }
        thr = s_prefix;
        take_eq = s_need;   // (the picks among the keys equal to thr)
    }
    // the picks in index order: each thread a contiguous range, its counts, a block scan, then the writes
    const int per = (nv + nth - 1) / nth, b0 = min(nv, tid * per), b1 = min(nv, b0 + per);
    int cg = 0, ce = 0;
    if (nv > top)
        for (int g = b0; g < b1; ++g) {
            const uint32_t kk = ord_key(sc[g]);
            cg += kk > thr;
            ce += kk == thr;
        }
    else
        cg = b1 - b0;
    // the counts' exclusive scans (integers: the same numbers however they are added): each warp its own, the warps'
    // totals by the first warp - one thread walking 1024 entries was most of this kernel's time
    const int lane = tid & 31, wp = tid >> 5;
    int ia = cg, ie = ce;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int pa = __shfl_up_sync(0xffffffffu, ia, o), pe = __shfl_up_sync(0xffffffffu, ie, o);
        if (lane >= o) {
            ia += pa;
            ie += pe;
        }
    }
    if (lane == 31) {
        s_gt[wp] = ia;
        s_eq[wp] = ie;
    }
    __syncthreads();
    if (tid < 32) {
        const bool in = tid < (nth >> 5);
        const int ta = in ? s_gt[tid] : 0, te = in ? s_eq[tid] : 0;
        int wa = ta, we = te;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int pa = __shfl_up_sync(0xffffffffu, wa, o), pe = __shfl_up_sync(0xffffffffu, we, o);
            if (lane >= o) {
                wa += pa;
                we += pe;
            }
        }
        if (in) {
            s_gt[tid] = wa - ta;
            s_eq[tid] = we - te;
        }
    }
    __syncthreads();
    // a pick's slot: the picks before it in index order = the greater keys before it + the ties taken before it (the
    // first take_eq ties in index order)
    int ng = s_gt[wp] + ia - cg, ne = s_eq[wp] + ie - ce;
    for (int g = b0; g < b1; ++g) {
        int slot = -1;
        if (nv <= top) {
            slot = ng++;
        } else {
            const uint32_t kk = ord_key(sc[g]);
            if (kk > thr) {
                slot = ng + min(ne, take_eq);
                ++ng;
            } else if (kk == thr) {
                if (ne < take_eq) slot = ng + ne;
                ++ne;
            }
        }
        if (slot < 0) continue;
        for (int k = 0; k < kpool; ++k) kt[slot * kpool + k] = g * kpool + k;
    }
    __syncthreads();
    const int ntail = p % kpool + 1;
    for (int k = tid; k < ntail; k += nth) kt[nsel * kpool + k] = nv * kpool + k;
    if (tid == 0) nk[t] = nsel * kpool + ntail;
}
// a block a token: its picked pools (ix_select's keys, kpool positions a pool) as 16-bit numbers into the host's row
__global__ void ix_pack_kernel(const int* __restrict__ keys, int ldk, const int* __restrict__ pos, int kpool, int top,
                               uint16_t* wire, int ldw) {
    const int t = blockIdx.x, nsel = min(pos[t] / kpool, top);
    const int* kt = keys + (size_t) t * ldk;
    uint16_t* w = wire + (size_t) t * ldw;
    for (int j = threadIdx.x; j < nsel; j += blockDim.x) w[j] = (uint16_t) (kt[(size_t) j * kpool] / kpool);
    __threadfence_system();
}
// ... and back: the row's numbers in shared memory (16 bytes a load, past every cache), then ix_select's keys
__global__ void ix_unpack_kernel(const uint16_t* wire, int ldw, const int* __restrict__ pos, int kpool, int top,
                                 int* __restrict__ keys, int ldk, int* __restrict__ nk) {
    extern __shared__ float4 ix_row[];
    const int t = blockIdx.x, p = pos[t], nv = p / kpool, nsel = min(nv, top);
    const float4* src = (const float4*) (wire + (size_t) t * ldw);
    for (int i = threadIdx.x; i < (nsel + 7) / 8; i += blockDim.x) ix_row[i] = maya::kernels::ld_fresh_v(src + i);
    __syncthreads();
    const uint16_t* sp = (const uint16_t*) ix_row;
    int* kt = keys + (size_t) t * ldk;
    for (int i = threadIdx.x; i < nsel * kpool; i += blockDim.x) kt[i] = (int) sp[i / kpool] * kpool + i % kpool;
    const int ntail = p % kpool + 1;
    for (int k = threadIdx.x; k < ntail; k += blockDim.x) kt[nsel * kpool + k] = nv * kpool + k;
    if (threadIdx.x == 0) nk[t] = nsel * kpool + ntail;
}

}  // namespace

// ---------------------------------------------------------------- launches
void q8_rows(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    if (rows <= 0) return;
    q8_kernel<<<dim3((unsigned) ((k / 32 + 7) / 8), (unsigned) rows), 256, 0, s>>>(x, ldx, k, q, d);
    check("q8_rows");
}
void q8_heads(const float* x, size_t ldt, size_t ldh, int T, int H, int k, int8_t* q, float* d, cudaStream_t s) {
    if (T <= 0 || H <= 0) return;
    q8_heads_kernel<<<dim3((unsigned) ((k / 32 + 7) / 8), (unsigned) (T * H)), 256, 0, s>>>(x, ldt, ldh, T, k, q, d);
    check("q8_heads");
}
void copy_cols(const float* src, size_t lds, float* dst, size_t ldd, int rows, int n, cudaStream_t s) {
    const size_t total = (size_t) rows * n;
    if (total == 0) return;
    copy_cols_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(src, lds, dst, ldd, n, total);
    check("copy_cols");
}
void store_rows(const float* src, int T, int n, float* ring, int R, const int* pos, cudaStream_t s, bool f16) {
    if (T <= 0) return;
    if (f16)
        store_rows_kernel<__half><<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(src, n, (__half*) ring,
                                                                                                R, pos);
    else
        store_rows_kernel<float><<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(src, n, ring, R, pos);
    check("store_rows");
}
void gemv_f32w(const float* W, size_t ld, int k, int n, const float* x, size_t ldx, int ncol, float* y, size_t ldy,
               float alpha, cudaStream_t s) {
    gemv<float>(W, ld, k, n, x, ldx, ncol, y, ldy, alpha, s);
}
void gemv_bf16w(const uint16_t* W, size_t ld, int k, int n, const float* x, size_t ldx, int ncol, float* y, size_t ldy,
                float alpha, cudaStream_t s) {
    gemv<uint16_t>(W, ld, k, n, x, ldx, ncol, y, ldy, alpha, s);
}
bool rmsnorm(const float* x, size_t ldx, int rows, int n, const float* w, float eps, float* y, size_t ldy, cudaStream_t s,
             Q8Out q8) {
    if (rows <= 0) return false;
    if (n % 32 != 0) q8 = Q8Out{};
    static const int ahead = !(std::getenv("HY4_RMS_AHEAD") && std::atoi(std::getenv("HY4_RMS_AHEAD")) == 0);   // (=0: as before)
    rms_kernel<<<rows, 256, 0, s>>>(x, ldx, n, w, eps, y, ldy, q8, ahead);
    check("rmsnorm");
    return q8.q != nullptr;
}
// hc_pre's and hc_head's buffers for launches of up to T rows on the current device, made now.  Grown inside a forward,
// the old buffer is freed under the buffers' lock, and the free waits for the GPU's queued work: in tensor parallel that
// work can wait for another member's sum, and that member for the lock.
void hc_reserve(int T) {
    (void) hc_part_buf(T);
    (void) hc_cnt_buf(T);
}
bool hc_pre(const float* x, int T, int hc, int E, const float* fn, const float* scale, const float* base, float eps,
            float hc_eps, float mag, float* cur, float* post, cudaStream_t s, const float* norm_w, float norm_eps,
            bool fn_by_element, Q8Out q8) {
    float* part = hc_part_buf(T);
    if (part == nullptr || 2 * hc > kMaxM || hc > 8 || E > 512 * 8) {
        std::fprintf(stderr, "maya-hy4 hc_pre: no room for its sums (%d streams)\n", hc);
        g_err.fetch_add(1);
        return false;
    }
    if (E % 32 != 0) q8 = Q8Out{};
    // (HY4_HC_ONE=1: one launch, the last block of a token finishing it - measured no faster: the launches queue ahead)
    static const bool two = std::getenv("HY4_HC_ONE") == nullptr;
    unsigned* cnt = two || fn_by_element || E > 256 * 12 ? nullptr : hc_cnt_buf(T);
    if (cnt) {
        hc_pre1_kernel<<<dim3(kHcBlocks, (unsigned) T), 256, 0, s>>>(x, hc, E, fn, part, cnt, scale, base, eps, hc_eps,
                                                                     mag, cur, post, g_mix_out, norm_w, norm_eps);
    } else {
        if (fn_by_element) hc_part_kernel<true><<<dim3(kHcBlocks, (unsigned) T), 256, 0, s>>>(x, hc * E, 2 * hc, fn, part);
        else hc_part_kernel<false><<<dim3(kHcBlocks, (unsigned) T), 256, 0, s>>>(x, hc * E, 2 * hc, fn, part);
        hc_fin_kernel<false><<<T, 512, 0, s>>>(x, hc, E, part, scale, base, eps, hc_eps, mag, cur, post, g_mix_out,
                                               norm_w, norm_eps, q8);
        check("hc_pre");
        return q8.q != nullptr;
    }
    check("hc_pre");
    return false;   // (the one-launch form: no int8 rows)
}
void hc_head(const float* x, int T, int hc, int E, const float* fn, const float* scale, const float* base, float eps,
             float hc_eps, float* cur, cudaStream_t s, bool fn_by_element) {
    float* part = hc_part_buf(T);
    if (part == nullptr || hc > kMaxM) {
        std::fprintf(stderr, "maya-hy4 hc_head: no room for its sums\n");
        g_err.fetch_add(1);
        return;
    }
    if (fn_by_element) hc_part_kernel<true><<<dim3(kHcBlocks, (unsigned) T), 256, 0, s>>>(x, hc * E, hc, fn, part);
    else hc_part_kernel<false><<<dim3(kHcBlocks, (unsigned) T), 256, 0, s>>>(x, hc * E, hc, fn, part);
    hc_fin_kernel<true><<<T, 512, 0, s>>>(x, hc, E, part, scale, base, eps, hc_eps, 1.0f, cur, nullptr, nullptr, nullptr,
                                          0.0f, Q8Out{});
    check("hc_head");
}
void hc_post(float* x, int T, int hc, int E, const float* y, const float* post, cudaStream_t s) {
    const size_t n = (size_t) T * hc * E;
    hc_post_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, hc, E, y, post, n);
    check("hc_post");
}
void hc_post_norm(float* x, int T, int hc, int E, const float* y, const float* post, const float* w, float eps,
                  cudaStream_t s, float* yo) {
    if (hc > 8 || E % 4) {
        std::fprintf(stderr, "maya-hy4 hc_post_norm: %d streams of %d (8 / a multiple of 4)\n", hc, E);
        g_err.fetch_add(1);
        return;
    }
    hc_post_norm_kernel<<<dim3(kPostBlocks, (unsigned) T), 256, 0, s>>>(x, hc, E, y, post, w, eps, yo);
    check("hc_post_norm");
}
bool rope_q8_copy(float* q, int T, int H, int k, int nope, int n_rot, const int* pos, float base, float* qcat, size_t ldq,
                  int kl, Q8Out q8, cudaStream_t s) {
    if (T <= 0 || H <= 0 || q8.q == nullptr || k != 256 || nope + n_rot != k || nope % 32 != 0 || n_rot % 2 != 0)
        return false;   // (the caller's three launches instead)
    rope_q8_copy_kernel<<<dim3((unsigned) H, (unsigned) T), 256, 0, s>>>(q, k, nope, n_rot, pos, base, qcat, ldq, kl, q8);
    check("rope_q8_copy");
    return true;
}
void rope(float* x, int T, int nh, size_t ld, size_t ldt, int off, int n_rot, const int* pos, float base, cudaStream_t s) {
    const int n = nh * n_rot / 2;
    rope_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(x, nh, ld, ldt, off, n_rot, pos, base);
    check("rope");
}
size_t kv_q8_row_bytes(int kv_lora, int n_rot) {
    return ((size_t) kv_lora + (size_t) kv_lora / 16 + 2 * (size_t) n_rot + 15) / 16 * 16;
}
void kv_store(const float* kv, int T, int kv_lora, int n_rot, const float* w, float eps, const int* pos, float base,
              uint16_t* cache, int ring, cudaStream_t s, bool q8, const KvHost& hs) {
    const int row = kv_lora + n_rot;
    if (q8 && (kv_lora % 32 || n_rot % 2 || row > 1024)) {
        std::fprintf(stderr, "maya-hy4 kv_store: a Q8_0 row of %d + %d values (whole blocks of 32, at most 1024)\n",
                     kv_lora, n_rot);
        g_err.fetch_add(1);
        return;
    }
    if (q8)
        kv_store_q8_kernel<<<T, 256, 0, s>>>(kv, kv_lora, n_rot, w, eps, pos, base, (uint8_t*) cache, ring,
                                             (int) kv_q8_row_bytes(kv_lora, n_rot), hs.tag, (uint32_t*) hs.rows,
                                             hs.htag, hs.hidx, hs.hres);
    else
        kv_store_kernel<<<T, 256, 0, s>>>(kv, kv_lora, n_rot, w, eps, pos, base, (__half*) cache, ring, hs.tag,
                                          (uint32_t*) hs.rows, hs.htag, hs.hidx, hs.hres);
    check("kv_store");
}
size_t attn_ws_floats(int T, int H, int kv_lora, int max_keys) {
    // (the larger of the two chunkings: T tokens at kKC, or the few-token calls' kKCmin)
    return std::max((size_t) T * H * ((max_keys + kKC - 1) / kKC),
                    (size_t) kMaxWin * H * ((max_keys + kKCmin - 1) / kKCmin)) * (kv_lora + 2);
}
namespace {
// attn_kernel's heads a block: 8 on NVIDIA, 32 on AMD (HY4_ATTN_HB=8 / 16 / 32 sets it there - the same sums)
int attn_heads_a_block() {
#if defined(MAYA_USE_HIP)
    static const int hb = [] {
        const char* e = std::getenv("HY4_ATTN_HB");
        const int v = e ? std::atoi(e) : 32;
        return v == 8 || v == 16 ? v : 32;
    }();
    return hb;
#else
    return 8;
#endif
}
}  // namespace
void attention(const float* q, int T, int H, int kv_lora, int n_rot, const uint16_t* cache, int ring, const int* keys,
               int ldk, const int* lo, const int* nk, int max_keys, float scale, const float* sinks, float* ws,
               float* out, cudaStream_t s, bool q8, Q8Out q8o) {
    if (kv_lora != 512 || n_rot != 64) {
        std::fprintf(stderr, "maya-hy4 attention: kv_lora %d + rope %d (512 + 64 only)\n", kv_lora, n_rot);
        g_err.fetch_add(1);
        return;
    }
    // (every window of the token path - a decode step, a draft's verify window - in chunks of kKCmin: a token's sums are
    // the same whichever window it is in, so drafting does not change a greedy answer)
    const int kc = T <= kMaxWin ? kKCmin : kKC, nchunk = (max_keys + kc - 1) / kc;
    const int rb = q8 ? (int) kv_q8_row_bytes(kv_lora, n_rot) : 0;
    const int hb = attn_heads_a_block();
    // (8 heads a block - NVIDIA, or HY4_ATTN_HB=8 on AMD: attn2_kernel, attn_kernel's sums in fewer loads; HY4_ATTN_OLD:
    //  attn_kernel, as before.  16 / 32 - AMD: attn_kernel over that many heads.  The same bits every way)
    static const bool old = std::getenv("HY4_ATTN_OLD") != nullptr;
    const dim3 grid((unsigned) nchunk, (unsigned) ((H + hb - 1) / hb), (unsigned) T);
    if (hb == 32) {
        if (q8) attn_kernel<true, 32><<<grid, 1024, 0, s>>>(q, H, cache, ring, keys, ldk, lo, nk, scale, nchunk, kc, ws, rb);
        else attn_kernel<false, 32><<<grid, 1024, 0, s>>>(q, H, cache, ring, keys, ldk, lo, nk, scale, nchunk, kc, ws, rb);
    } else if (hb == 16) {
        if (q8) attn_kernel<true, 16><<<grid, 512, 0, s>>>(q, H, cache, ring, keys, ldk, lo, nk, scale, nchunk, kc, ws, rb);
        else attn_kernel<false, 16><<<grid, 512, 0, s>>>(q, H, cache, ring, keys, ldk, lo, nk, scale, nchunk, kc, ws, rb);
    } else if (old) {
        if (q8) attn_kernel<true><<<grid, 256, 0, s>>>(q, H, cache, ring, keys, ldk, lo, nk, scale, nchunk, kc, ws, rb);
        else attn_kernel<false><<<grid, 256, 0, s>>>(q, H, cache, ring, keys, ldk, lo, nk, scale, nchunk, kc, ws, 0);
    } else {
        if (q8) attn2_kernel<true><<<grid, 256, 0, s>>>(q, H, cache, ring, keys, ldk, lo, nk, scale, nchunk, kc, ws, rb);
        else attn2_kernel<false><<<grid, 256, 0, s>>>(q, H, cache, ring, keys, ldk, lo, nk, scale, nchunk, kc, ws, 0);
    }
    check("attention");
    attn_combine_kernel<<<dim3((unsigned) H, (unsigned) T), 512, 0, s>>>(H, nchunk, kc, nk, sinks, ws, out, q8o);
    check("attn_combine");
}
bool attn_wmma_on() {
#if defined(MAYA_USE_HIP)
    // (the device's arch once a device: gfx11 - RDNA 3 / 3.5 - by default; gfx12 only with HY4_ATTN_WMMA=1, not
    //  measured there; HY4_ATTN_WMMA=0 off everywhere)
    static const int env = std::getenv("HY4_ATTN_WMMA") ? std::atoi(std::getenv("HY4_ATTN_WMMA")) : -1;
    static std::atomic<int> known[64];   // 0: not asked yet, 1: no, 2: yes
    int dev = 0;
    if (env == 0 || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    int v = known[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        cudaDeviceProp p{};
        const bool ok = cudaGetDeviceProperties(&p, dev) == cudaSuccess;
        v = ok && (std::strncmp(p.gcnArchName, "gfx11", 5) == 0 ||
                   (env == 1 && std::strncmp(p.gcnArchName, "gfx12", 5) == 0)) ? 2 : 1;
        known[dev].store(v, std::memory_order_relaxed);
    }
    return v == 2;
#else
    return false;
#endif
}
bool attention_wmma(const float* q, int T, int H, int kv_lora, int n_rot, const uint16_t* cache, int ring,
                    const int* keys, int ldk, const int* nk, float scale, const float* sinks, float* out,
                    cudaStream_t s, bool q8) {
    if (T <= 0 || kv_lora != 512 || n_rot != 64 || keys == nullptr || H % 16 != 0 || !attn_wmma_on()) return false;
    // G: the groups of 16 heads a block sharing a staged tile - 2 where the heads come in 32s (HY4_ATTN_WMMA_G=1: one)
    static const int g1 = std::getenv("HY4_ATTN_WMMA_G") ? std::atoi(std::getenv("HY4_ATTN_WMMA_G")) : 0;
    const int G = H % 32 == 0 && g1 != 1 ? 2 : 1;
    const int rb = q8 ? (int) kv_q8_row_bytes(kv_lora, n_rot) : 0;
    const dim3 grid((unsigned) (H / (16 * G)), (unsigned) T);
    if (G == 2) {
        if (q8) attn_wmma_kernel<true, 2><<<grid, 512, 0, s>>>(q, H, cache, ring, keys, ldk, nk, scale, sinks, out, rb);
        else attn_wmma_kernel<false, 2><<<grid, 512, 0, s>>>(q, H, cache, ring, keys, ldk, nk, scale, sinks, out, rb);
    } else {
        if (q8) attn_wmma_kernel<true, 1><<<grid, 256, 0, s>>>(q, H, cache, ring, keys, ldk, nk, scale, sinks, out, rb);
        else attn_wmma_kernel<false, 1><<<grid, 256, 0, s>>>(q, H, cache, ring, keys, ldk, nk, scale, sinks, out, rb);
    }
    check("attention_wmma");
    return true;
}
bool sigmoid_mul(float* x, const float* g, size_t n, cudaStream_t s, Q8Out q8) {
    if (n % 32 != 0) q8 = Q8Out{};
    sigmoid_mul_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, g, n, q8);
    check("sigmoid_mul");
    return q8.q != nullptr;
}
bool swiglu(const float* g, const float* u, size_t ldg, int rows, int n, float* h, size_t ldh, cudaStream_t s, Q8Out q8) {
    const size_t total = (size_t) rows * n;
    if (n % 32 != 0) q8 = Q8Out{};
    swiglu_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(g, u, ldg, n, h, ldh, total, q8);
    check("swiglu");
    return q8.q != nullptr;
}
void add(float* y, const float* x, size_t n, cudaStream_t s) {
    add_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(y, x, n);
    check("add");
}
void addmul(float* y, const float* x, const float* w, size_t n, cudaStream_t s) {
    addmul_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(y, x, w, n);
    check("addmul");
}
void router(const float* logits, int T, int n_exp, const float* bias, int n_used, bool norm, float scale, int* ids,
            float* w, cudaStream_t s) {
    if (n_exp > 1024 || n_used > 16) {
        std::fprintf(stderr, "maya-hy4 router: %d experts, %d a token (1024 / 16 at most)\n", n_exp, n_used);
        g_err.fetch_add(1);
        return;
    }
    static const bool sort = std::getenv("HY4_ROUTER_SORT") != nullptr;   // (the block-wide sort: the earlier kernel)
    static const bool warp = std::getenv("HY4_ROUTER_WARP") != nullptr;   // (the one-warp picks)
    if (!sort && !warp && n_exp <= 512 && n_used <= 16)
        router_bitonic_kernel<<<(unsigned) T, 512, 0, s>>>(logits, n_exp, bias, n_used, norm ? 1 : 0, scale, ids, w);
    else if (!sort && n_exp <= 512 && n_used <= 32)
        router_warp_kernel<16><<<(unsigned) T, 32, 0, s>>>(logits, n_exp, bias, n_used, norm ? 1 : 0, scale, ids, w);
    else if (!sort && n_used <= 32)
        router_warp_kernel<32><<<(unsigned) T, 32, 0, s>>>(logits, n_exp, bias, n_used, norm ? 1 : 0, scale, ids, w);
    else
        router_kernel<<<(unsigned) T, 1024, 0, s>>>(logits, n_exp, bias, n_used, norm ? 1 : 0, scale, ids, w);
    check("router");
}
void group_experts(const int* ids, int T, int n_used, int n_exp, const uint8_t* base, size_t stride,
                   unsigned long long* grp_ptr, int* grp_start, int* grp_ent, int* n_groups, cudaStream_t s,
                   const int* slot, float* w, const unsigned long long* eptr, const int* who) {
    static const bool serial = std::getenv("HY4_GROUP_SERIAL") != nullptr;   // (the earlier one-thread sort)
    static const bool par2 = !(std::getenv("HY4_GROUP_PAR2") && std::getenv("HY4_GROUP_PAR2")[0] == '0');   // (=0: group_par_kernel)
    if (!serial && par2 && n_exp <= 1024 && T * n_used <= 16 * 16)
        group_par2_kernel<<<1, 1024, 0, s>>>(ids, T, n_used, n_exp, base, stride, grp_ptr, grp_start, grp_ent, n_groups,
                                             slot, w, eptr, who);
    else if (!serial && n_exp <= 1024)
        group_par_kernel<<<1, 1024, 0, s>>>(ids, T, n_used, n_exp, base, stride, grp_ptr, grp_start, grp_ent, n_groups,
                                            slot, w, eptr, who);
    else
        group_kernel<<<1, 256, 0, s>>>(ids, T, n_used, n_exp, base, stride, grp_ptr, grp_start, grp_ent, n_groups, slot, w,
                                       eptr, who);
    check("group_experts");
}
void plan_experts(const int* ids, int n_used, const uint8_t* base, size_t stride, unsigned long long* plan,
                  cudaStream_t s, const int* slot, float* w, const unsigned long long* eptr, const int* who) {
    plan_kernel<<<1, 32, 0, s>>>(ids, n_used, base, stride, plan, slot, w, eptr, who);
    check("plan_experts");
}
// One block of 1024 threads (n_exp <= 1024, T n_used <= 256): see fetch_resolve.  Every choice is a function of the
// window's picks and the layer's state alone (ranks by score with the expert's number breaking ties; the staging slot a
// miss lands in may vary from run to run, where it lies does not change what is computed)
__global__ void __launch_bounds__(1024) fetch_resolve_kernel(const int* __restrict__ ids, int T, int n_used,
                                                             FetchLayer fl, FetchScratch fs, int k_gpu, int cache,
                                                             float decay, int ahead) {
    __shared__ int ent[kMaxWin * 16];
    __shared__ int cnt[1024], miss_e[kMaxWin * 16], vict_e[kMaxWin * 16];
    __shared__ float sc[1024];
    __shared__ unsigned char gpu_e[1024], kind[1024];   // kind: 1 a miss, 2 a victim candidate
    __shared__ int n_miss, n_vict, n_fetch, n_stage, n_hit, n_lane, n_zc, n_adm;
    const int tid = threadIdx.x, n_ent = T * n_used, n_exp = fl.n_exp;
    if (tid == 0) n_miss = n_vict = n_fetch = n_stage = n_hit = n_lane = n_zc = n_adm = 0;
    for (int i = tid; i < n_ent; i += blockDim.x) ent[i] = ids[i];
    if (tid < n_exp) {
        cnt[tid] = 0;
        gpu_e[tid] = 0;
    }
    __syncthreads();
    if (tid < T) {   // (a token's entries in pick order: its VRAM ones, then its first k_gpu others, this GPU's)
        int g = 0, hit = 0, lane = 0;
        for (int j = 0; j < n_used; ++j) {
            const int i = tid * n_used + j, e = ent[i];
            const bool res = fl.slot[e] >= 0;
            const bool mine = res || k_gpu < 0 || g < k_gpu;
            if (!res && mine) ++g;
            if (!ahead) fs.who[i] = mine ? 0 : 1;
            if (mine) gpu_e[e] = 1;
            hit += res;
            lane += !mine;
            atomicAdd(&cnt[e], 1);
        }
        atomicAdd(&n_hit, hit);
        atomicAdd(&n_lane, lane);
    }
    __syncthreads();
    if (tid < n_exp) {
        const int s = fl.slot[tid];
        float v = 0.0f;
        if (cache) {
            v = fl.score[tid] * powf(decay, (float) T) + (float) cnt[tid];
            if (!ahead) fl.score[tid] = v;   // (a guess weighs as the picks would, and is not counted)
        }
        sc[tid] = v;
        kind[tid] = gpu_e[tid] && s < 0 ? 1 : cache && s >= 0 && cnt[tid] == 0 ? 2 : 0;
    }
    __syncthreads();
    if (tid < n_exp && kind[tid] != 0) {   // ranks: the misses by score down, the victims by score up
        const float v = sc[tid];
        const bool m = kind[tid] == 1;
        int r = 0;
        for (int e = 0; e < n_exp; ++e)
            if (kind[e] == kind[tid] && (m ? (sc[e] > v || (sc[e] == v && e < tid)) : (sc[e] < v || (sc[e] == v && e < tid))))
                ++r;
        if (m) {
            miss_e[r] = tid;   // (r < n_ent: at most a miss an entry)
            atomicAdd(&n_miss, 1);
        } else {
            if (r < kMaxWin * 16) vict_e[r] = tid;
            atomicAdd(&n_vict, 1);
        }
    }
    __syncthreads();
    if (tid < n_miss) {   // miss r against victim r: the cache takes it when it scores higher (no admission past one
                          //  that does not - the misses' scores fall as the victims' rise)
        const int m = miss_e[tid];
        unsigned long long dst = 0ull;
        if (cache && tid < n_vict && tid < kMaxWin * 16 && sc[m] > sc[vict_e[tid]]) {
            const int v = vict_e[tid], sl = fl.slot[v];
            fl.slot[v] = -1;
            fl.slot[m] = sl;
            fl.owner[sl] = m;
            dst = (unsigned long long) (fl.vram + (size_t) sl * fl.blob);
            atomicAdd(&n_adm, 1);
        } else if (!ahead) {   // (a guess the cache does not take: not copied)
            const int k = atomicAdd(&n_stage, 1);
            if (k < fs.n_stage) dst = (unsigned long long) (fs.stage + (size_t) k * fl.blob);
        }
        if (dst != 0ull) {
            const int f = atomicAdd(&n_fetch, 1);
            fs.src[f] = fl.hptr[m];
            fs.dst[f] = dst;
            if (!ahead) fs.eptr[m] = dst;
        } else if (!ahead) {
            fs.eptr[m] = fl.hptr[m];   // (no slot left: the kernels read it where it lies)
            atomicAdd(&n_zc, 1);
        }
    }
    __syncthreads();
    if (!ahead && tid < n_exp && kind[tid] != 1) {
        const int s = fl.slot[tid];
        fs.eptr[tid] = gpu_e[tid] && s >= 0 ? (unsigned long long) (fl.vram + (size_t) s * fl.blob) : 0ull;
    }
    if (tid == 0) {
        *fs.nf = (unsigned) n_fetch;
        if (fs.stats) {
            atomicAdd(&fs.stats[0], (unsigned long long) n_ent);
            atomicAdd(&fs.stats[1], (unsigned long long) n_hit);
            atomicAdd(&fs.stats[2], (unsigned long long) n_fetch);
            atomicAdd(&fs.stats[3], (unsigned long long) n_zc);
            atomicAdd(&fs.stats[4], (unsigned long long) n_lane);
            atomicAdd(&fs.stats[5], (unsigned long long) n_adm);
        }
    }
}
// the listed copies one after another, each by the whole grid in 16-byte words (pinned RAM read over the bus)
__global__ void __launch_bounds__(256) fetch_copy_kernel(const unsigned long long* __restrict__ src,
                                                         const unsigned long long* __restrict__ dst,
                                                         const unsigned* __restrict__ nf, size_t n16) {
    const unsigned n = *nf;
    const size_t step = (size_t) gridDim.x * blockDim.x;
    for (unsigned f = 0; f < n; ++f) {
        const uint4* a = (const uint4*) src[f];
        uint4* b = (uint4*) dst[f];
        for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += step) b[i] = a[i];
    }
}
void fetch_resolve(const int* ids, int T, int n_used, const FetchLayer& fl, const FetchScratch& fs, int k_gpu,
                   bool cache, float decay, cudaStream_t s, bool ahead) {
    fetch_resolve_kernel<<<1, 1024, 0, s>>>(ids, T, n_used, fl, fs, k_gpu, cache ? 1 : 0, decay, ahead ? 1 : 0);
    check("fetch_resolve");
}
void fetch_copy(const FetchScratch& fs, size_t blob, int blocks, cudaStream_t s) {
    fetch_copy_kernel<<<(unsigned) std::max(1, blocks), 256, 0, s>>>(fs.src, fs.dst, fs.nf, (blob + 15) / 16);
    check("fetch_copy");
}
bool moe_sum(const float* part, int T, int n_used, int n, const float* w, float* out, cudaStream_t s, Q8Out q8) {
    if (n % 32 != 0) q8 = Q8Out{};
    moe_sum_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(part, n_used, n, w, out, 16, q8);
    check("moe_sum");
    return q8.q != nullptr;
}
void q8_rows_dup(const float* x, size_t ldx, int rows, int k, int8_t* q2, float* d2, cudaStream_t s) {
    if (rows <= 0) return;
    q8_dup_kernel<<<dim3((unsigned) ((k / 32 + 7) / 8), (unsigned) rows), 256, 0, s>>>(x, ldx, k, q2, d2);
    check("q8_rows_dup");
}
void dup_rows_q8(const int8_t* q, const float* d, int rows, int k, int8_t* q2, float* d2, cudaStream_t s) {
    if (rows <= 0) return;
    dup_q8_kernel<<<dim3((unsigned) ((k + 255) / 256), (unsigned) rows), 256, 0, s>>>(q, d, k, q2, d2);
    check("dup_rows_q8");
}
void engram_rows(const uint8_t* rows, const float* valid, int T, int cols, int dim, const float* lookup_norm, float eps,
                 float* r, cudaStream_t s, const float* scale) {
    const int th = (dim + 31) / 32 * 32;
    engram_rows_kernel<<<dim3((unsigned) cols, (unsigned) T), th, 0, s>>>(rows, valid, cols, dim, lookup_norm, eps, r,
                                                                         scale);
    check("engram_rows");
}
void engram_mix(float* x, const float* kv, const float* kv_b, int T, int hc, int E, const float* egate,
                const float* gamma, const float* hc_scale, const float* valid, float eps, cudaStream_t s) {
    engram_mix_kernel<<<T, 512, 0, s>>>(x, kv, kv_b, hc, E, egate, gamma, hc_scale, valid, eps);
    check("engram_mix");
}
void ix_pool(const float* ivg, int ring, int g0, int n, int kpool, int dim, const float* ape, const float* w, float eps,
             int n_rot, float base, float* pooled, cudaStream_t s, bool f16) {
    if (n <= 0) return;
    if (f16)
        ix_pool_kernel<__half, __half><<<n, dim, 0, s>>>((const __half*) ivg, ring, g0, kpool, dim, ape, w, eps, n_rot,
                                                         base, (__half*) pooled);
    else
        ix_pool_kernel<float, float><<<n, dim, 0, s>>>(ivg, ring, g0, kpool, dim, ape, w, eps, n_rot, base, pooled);
    check("ix_pool");
}
void ix_score(const float* q, const float* wt, int T, int heads, int dim, const float* pooled, const int* pos, int kpool,
              int n_pool, float* score, size_t ld, cudaStream_t s, bool f16) {
    if (n_pool <= 0) return;
    if (dim % 8 || heads > 16) {
        std::fprintf(stderr, "maya-hy4 ix_score: %d heads of %d (16 at most, a multiple of 8)\n", heads, dim);
        g_err.fetch_add(1);
        return;
    }
    const size_t sh = (size_t) (heads * dim + heads) * sizeof(float);
    const dim3 grid((unsigned) ((n_pool + 255) / 256), (unsigned) T);
    if (f16)
        ix_score_kernel<__half><<<grid, 256, sh, s>>>(q, wt, heads, dim, (const __half*) pooled, pos, kpool, n_pool,
                                                      score, ld);
    else
        ix_score_kernel<float><<<grid, 256, sh, s>>>(q, wt, heads, dim, pooled, pos, kpool, n_pool, score, ld);
    check("ix_score");
}
bool ix_score_wmma(const float* q, const float* wt, int T, int heads, int dim, const float* pooled, const int* pos,
                   int kpool, int n_pool, float* score, size_t ld, cudaStream_t s, bool f16) {
    // (HY4_IX_WMMA=0: ix_score's own kernel; the device as attention_wmma's)
    static const bool off = std::getenv("HY4_IX_WMMA") != nullptr && std::atoi(std::getenv("HY4_IX_WMMA")) == 0;
    if (off || T <= 0 || heads != 16 || dim != 128 || !attn_wmma_on()) return false;
    if (n_pool <= 0) return true;
    const dim3 grid((unsigned) ((n_pool + kIxPB - 1) / kIxPB), (unsigned) ((T + 7) / 8));
    if (f16)
        ix_score_wmma_kernel<__half><<<grid, 256, 0, s>>>(q, wt, T, (const __half*) pooled, pos, kpool, n_pool, score, ld);
    else
        ix_score_wmma_kernel<float><<<grid, 256, 0, s>>>(q, wt, T, pooled, pos, kpool, n_pool, score, ld);
    check("ix_score_wmma");
    return true;
}
void ix_reduce(const float* S, size_t ldS, const float* wt, int T, int heads, const int* pos, int kpool, int g0, int n,
               float* score, size_t ld, cudaStream_t s) {
    if (n <= 0 || T <= 0) return;
    ix_reduce_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(S, ldS, wt, heads, pos, kpool, g0,
                                                                                    n, score, ld);
    check("ix_reduce");
}
void ix_select(const float* score, size_t ld, int T, const int* pos, int kpool, int top, int* keys, int ldk, int* nk,
               cudaStream_t s) {
    ix_select_kernel<<<T, 1024, 0, s>>>(score, ld, pos, kpool, top, keys, ldk, nk);
    check("ix_select");
}
void ix_pack(const int* keys, int ldk, int T, const int* pos, int kpool, int top, uint16_t* wire, int ldw,
             cudaStream_t s) {
    if (T <= 0) return;
    ix_pack_kernel<<<T, 256, 0, s>>>(keys, ldk, pos, kpool, top, wire, ldw);
    check("ix_pack");
}
void ix_unpack(const uint16_t* wire, int ldw, int T, const int* pos, int kpool, int top, int* keys, int ldk, int* nk,
               cudaStream_t s) {
    if (T <= 0) return;
    ix_unpack_kernel<<<T, 256, (size_t) (top + 7) / 8 * 16, s>>>(wire, ldw, pos, kpool, top, keys, ldk, nk);
    check("ix_unpack");
}
void attn_softmax(const float* S, size_t ldS, int T, int H, int nkr, int k0, const int* pos, int window,
                  const float* mask, size_t ldm, float scale, const float* sinks, uint16_t* P, size_t ldP,
                  cudaStream_t s) {
    if (T <= 0) return;
    attn_softmax_kernel<<<T * H, 256, 0, s>>>(S, ldS, H, nkr, k0, pos, window, mask, ldm, scale, sinks, (__half*) P,
                                             ldP);
    check("attn_softmax");
}
void dsa_mask(const int* keys, int ldk, const int* nk, int T, int k0, int nkr, float* mask, size_t ldm, cudaStream_t s) {
    if (T <= 0) return;
    dsa_mask_kernel<<<T, 256, 0, s>>>(keys, ldk, nk, k0, nkr, mask, ldm);
    check("dsa_mask");
}
void kv_deq(const uint16_t* cache, int ring, int kv_lora, int n_rot, int k0, int n, uint16_t* dst, cudaStream_t s,
            const int* list, int kpool, int kend, const KvHost& hs) {
    const size_t total = (size_t) n * ((kv_lora + n_rot) / 2);
    if (total == 0) return;
    kv_deq_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>((const uint8_t*) cache, ring, kv_lora, n_rot,
                                                                 (int) kv_q8_row_bytes(kv_lora, n_rot), k0, n,
                                                                 (__half2*) dst, list, kpool, kend, hs.tag,
                                                                 (const uint8_t*) hs.rows, hs.hidx);
    check("kv_deq");
}
void kv_gather(const uint16_t* cache, int ring, int row, int k0, int n, uint16_t* dst, cudaStream_t s, const int* list,
               int kpool, int kend, const KvHost& hs) {
    const int row16 = row * 2 / 16;   // (uint4s a row: 576 F16 = 72)
    const size_t total = (size_t) n * row16;
    if (total == 0) return;
    kv_gather_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>((const uint4*) cache, ring, row16, k0, n,
                                                                    (uint4*) dst, list, kpool, kend, hs.tag,
                                                                    (const uint4*) hs.rows, hs.hidx);
    check("kv_gather");
}
void ix_union(const int* keys, int ldk, const int* nk, int T, int kpool, int n_pool, int* col, int* list, int* n,
              cudaStream_t s) {
    ix_union_kernel<<<1, 1024, 0, s>>>(keys, ldk, nk, T, kpool, n_pool, col, list, n);
    check("ix_union");
}
void dsa_mask_list(const int* keys, int ldk, const int* nk, int T, const int* col, int kpool, int ncol, float* mask,
                   size_t ldm, cudaStream_t s) {
    if (T <= 0) return;
    dsa_mask_list_kernel<<<T, 256, 0, s>>>(keys, ldk, nk, col, kpool, ncol, mask, ldm);
    check("dsa_mask_list");
}
namespace {
// a thread a key: the first to claim a position gives it the next row
__global__ void kv_union_claim_kernel(const int* __restrict__ keys, int ldk, const int* __restrict__ nk, int T, int* slot,
                                      int* count, int* __restrict__ list) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * ldk) return;
    if ((int) (i % ldk) >= nk[i / ldk]) return;
    const int p = keys[i];
    if (atomicCAS(slot + p, -1, -2) == -1) {
        const int idx = atomicAdd(count, 1);
        list[idx] = p;
        slot[p] = idx;
    }
}
__global__ void kv_union_fetch_kernel(const uint4* __restrict__ cache, int ring, int row16, const int* __restrict__ list,
                                      const int* __restrict__ count, uint4* __restrict__ dst,
                                      const int* __restrict__ tag, const uint4* __restrict__ host,
                                      const int* __restrict__ hidx) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = (int) (i / row16), c = (int) (i % row16);
    if (idx >= *count) return;
    bool fh = false;
    const size_t r = kv_src(list[idx], ring, tag, hidx, fh);
    dst[i] = (fh ? host : cache)[r * row16 + c];
}
__global__ void kv_union_remap_kernel(const int* __restrict__ keys, int ldk, const int* __restrict__ nk, int T,
                                      const int* __restrict__ slot, int* __restrict__ keys2) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * ldk) return;
    if ((int) (i % ldk) >= nk[i / ldk]) return;
    keys2[i] = slot[keys[i]];
}
__global__ void kv_union_reset_kernel(const int* __restrict__ list, const int* __restrict__ count, int* __restrict__ slot) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *count) slot[list[idx]] = -1;
}
}
void kv_pick_union(const uint16_t* cache, int ring, size_t row_bytes, const int* keys, int ldk, const int* nk, int T,
                   int* slot, int* count, int* list, int* keys2, uint16_t* dst, cudaStream_t s, const KvHost& hs) {
    const int row16 = (int) (row_bytes / 16);
    if (T <= 0 || ldk <= 0) return;
    if (row_bytes % 16) {
        std::fprintf(stderr, "maya-hy4 kv_pick_union: a row of %zu bytes (whole 16s only)\n", row_bytes);
        g_err.fetch_add(1);
        return;
    }
    const size_t n = (size_t) T * ldk;
    const unsigned nb = (unsigned) ((n + 255) / 256);
    cudaMemsetAsync(count, 0, sizeof(int), s);
    kv_union_claim_kernel<<<nb, 256, 0, s>>>(keys, ldk, nk, T, slot, count, list);
    kv_union_fetch_kernel<<<(unsigned) ((n * row16 + 255) / 256), 256, 0, s>>>((const uint4*) cache, ring, row16, list,
                                                                              count, (uint4*) dst, hs.tag,
                                                                              (const uint4*) hs.rows, hs.hidx);
    kv_union_remap_kernel<<<nb, 256, 0, s>>>(keys, ldk, nk, T, slot, keys2);
    kv_union_reset_kernel<<<nb, 256, 0, s>>>(list, count, slot);
    check("kv_pick_union");
}
void kv_pick(const uint16_t* cache, int ring, size_t row_bytes, const int* keys, int ldk, const int* nk, int T,
             uint16_t* dst, cudaStream_t s, const KvHost& hs) {
    const int row16 = (int) (row_bytes / 16);
    if (T <= 0 || ldk <= 0) return;
    if (row_bytes % 16) {
        std::fprintf(stderr, "maya-hy4 kv_pick: a row of %zu bytes (whole 16s only)\n", row_bytes);
        g_err.fetch_add(1);
        return;
    }
    const dim3 grid((unsigned) (((size_t) ldk * row16 + 255) / 256), (unsigned) T);
    kv_pick_kernel<<<grid, 256, 0, s>>>((const uint4*) cache, ring, row16, keys, ldk, nk, (uint4*) dst, hs.tag,
                                        (const uint4*) hs.rows, hs.hidx);
    check("kv_pick");
}
namespace {
__global__ void kv_need_kernel(const int* __restrict__ keys, int ldk, const int* __restrict__ nk, int T,
                               const int* __restrict__ list, int kpool, int kend, int k0, int n, int ring,
                               const int* __restrict__ tag, int* hidx, int hres, int* fgen, int gen, int* miss,
                               int* mcount, int cap) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    int p;
    if (keys) {
        if (i >= (size_t) T * ldk || (int) (i % ldk) >= nk[i / ldk]) return;
        p = keys[i];
    } else if (list) {
        if (i >= (size_t) n * kpool) return;
        p = list[i / kpool] * kpool + (int) (i % kpool);
        if (p >= kend) return;
    } else {
        if (i >= (size_t) n) return;
        p = k0 + (int) i;
    }
    if (p < 0) return;
    if (tag && tag[ring > 0 ? p % ring : p] == p) return;
    const int h = hidx[p];
    if (h >= hres) {
        fgen[h - hres] = gen;
        return;
    }
    if (h != -1) return;
    if (atomicCAS(hidx + p, -1, -2) == -1) {
        const int k = atomicAdd(mcount, 1);
        if (k < cap) miss[k] = p;
    }
}
__global__ void kv_hidx_kernel(const int* __restrict__ set, int n_set, const int* __restrict__ drop, int n_drop,
                               int* hidx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_drop) atomicCAS(hidx + drop[2 * i], drop[2 * i + 1], -1);
    if (i < n_set) hidx[set[2 * i]] = set[2 * i + 1];
}
}
void kv_need(const int* keys, int ldk, const int* nk, int T, const int* list, int kpool, int kend, int k0, int n,
             int ring, const KvHost& hs, int* fgen, int gen, int* miss, int* mcount, int cap, cudaStream_t s) {
    const size_t total = keys ? (size_t) T * ldk : list ? (size_t) n * kpool : (size_t) n;
    if (total == 0 || !hs.hidx) return;
    kv_need_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(keys, ldk, nk, T, list, kpool, kend, k0, n, ring,
                                                                   hs.tag, hs.hidx, hs.hres, fgen, gen, miss, mcount,
                                                                   cap);
    check("kv_need");
}
void kv_hidx(const int* set, int n_set, const int* drop, int n_drop, int* hidx, cudaStream_t s) {
    const int n = std::max(n_set, n_drop);
    if (n <= 0) return;
    kv_hidx_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(set, n_set, drop, n_drop, hidx);
    check("kv_hidx");
}
namespace {
__global__ void bf16_f32_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = bf16f(src[i]);
}
}  // namespace
void bf16_to_f32(const uint16_t* src, float* dst, size_t n, cudaStream_t s) {
    if (n == 0) return;
    bf16_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(src, dst, n);
    check("bf16_to_f32");
}
void to_f16(const float* src, size_t lds, int rows, int n, uint16_t* dst, size_t ldd, cudaStream_t s) {
    const size_t total = (size_t) rows * n;
    if (total == 0) return;
    if (f16_rows_form(src, lds, dst, ldd)) {
        const int b = rows_threads(n);
        const dim3 g((unsigned) (((n + 3) / 4 + b - 1) / b), (unsigned) std::min(rows, 65535));
        to_f16_rows_kernel<<<g, b, 0, s>>>(src, lds, n, (__half*) dst, ldd, rows);
        check("to_f16");
        return;
    }
    to_f16_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(src, lds, n, (__half*) dst, ldd, total);
    check("to_f16");
}
void moe_sort(const int* ids, int T, int n_used, int n_exp, int* bounds, int* tok, int* row_of, cudaStream_t s,
              const int* slot, float* w, int n_slots) {
    if (n_exp > 1024) {
        std::fprintf(stderr, "maya-hy4 moe_sort: %d experts (1024 at most)\n", n_exp);
        g_err.fetch_add(1);
        return;
    }
    moe_sort_kernel<<<1, 1024, 0, s>>>(ids, T * n_used, slot ? n_slots : n_exp, bounds, tok, row_of, n_used, slot, w);
    check("moe_sort");
}
void moe_combine(const float* y, const int* row_of, const float* w, int T, int n_used, int n, float* out,
                 cudaStream_t s, bool accumulate) {
    if (T <= 0) return;
    moe_combine_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(y, row_of, w, n_used, n, out,
                                                                                       accumulate ? 1 : 0);
    check("moe_combine");
}
namespace {
constexpr unsigned long long kWaitLimitNs = 60ull * 1000000000ull;
__device__ unsigned g_wait_timeouts = 0;
// thread 0 of the block waits for *flag >= v (wrapping); every thread after it
__device__ __forceinline__ void wait_for(const unsigned* flag, unsigned v) {
    if (threadIdx.x == 0) {
        const unsigned long long t0 = maya::kernels::clock_ns();
        while ((int) (*(volatile const unsigned*) flag - v) < 0) {
            __nanosleep(64);
            if (maya::kernels::clock_ns() - t0 > kWaitLimitNs) {
                atomicAdd(&g_wait_timeouts, 1u);
                break;
            }
        }
    }
    __syncthreads();
    __threadfence_system();
}
}  // namespace
__global__ void publish_kernel(const float4* __restrict__ src, float4* dst, size_t n4, unsigned* counter, unsigned* flag,
                               unsigned v, const float4* __restrict__ plus) {
    // (plus: src + plus goes out - add_kernel's sums, value by value, without their launch)
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x) {
        float4 a = src[i];
        if (plus != nullptr) {
            const float4 b = plus[i];
            a.x += b.x;
            a.y += b.y;
            a.z += b.z;
            a.w += b.w;
        }
        dst[i] = a;
    }
    __threadfence_system();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (last && threadIdx.x == 0) {
        *counter = 0u;
        __threadfence_system();
        *(volatile unsigned*) flag = v;
        __threadfence_system();
    }
}
__global__ void wait_pull_kernel(const unsigned* flag, unsigned v, float4* __restrict__ dst, const float4* src,
                                 size_t n4) {
    wait_for(flag, v);
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = maya::kernels::ld_fresh_v(src + i);
}
__global__ void wait_flag_kernel(const unsigned* flag, unsigned v) { wait_for(flag, v); }
// (the ranges' lines in turn, every thread a line a step: the first ranges' come first)
__global__ void __launch_bounds__(256) prefetch_kernel(const Prefetch j, unsigned* sink) {
    const uint32_t nt = gridDim.x * blockDim.x, total = j.pre[j.n];
    uint32_t acc = 0;
    int r = 0;
    for (uint32_t l = blockIdx.x * blockDim.x + threadIdx.x; l < total; l += nt) {
        while (l >= j.pre[r + 1]) ++r;
        acc ^= *(const uint32_t*) (j.p[r] + (size_t) (l - j.pre[r]) * 32);
    }
    if (acc == 0x9e3779b9u) *sink = acc;   // (the loads kept: a value no data is expected to make)
}
__global__ void set_flag_kernel(unsigned* flag, unsigned v) {
    __threadfence_system();
    *(volatile unsigned*) flag = v;
    __threadfence_system();
}
void submit(cudaStream_t s) {
#if defined(_WIN32) && defined(MAYA_USE_HIP)
    static const bool on = [] {
        const char* v = std::getenv("HY4_WIN_SUBMIT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (on) {
        (void) cudaStreamQuery(s);
        (void) cudaGetLastError();   // (cudaErrorNotReady: the work is queued, not an error)
    }
#else
    (void) s;
#endif
}
// (a few tokens' rows go through the kernels; a prompt chunk's through the copy engines, the flag after them.  Both
//  end with submit: the host's summing threads and the other GPUs spin on the flag - #303's hang on maya-ds4 was a
//  flag whose writer had never been submitted)
constexpr size_t kViaKernel = 64 * 1024;
// (the read-back kernel's blocks at most: 3 - 768 threads, a token's 768 loads of 16 bytes one each, a window of four's
//  four each.  More blocks read a window's sum more slowly: 8 blocks about 1% slower over four prompts on eight RTX
//  5060 Ti; fewer than a token's 3 read one token's sum 5% slower.  HY4_PULL_BLOCKS: another count)
static size_t pull_blocks() {
    static const size_t v = std::getenv("HY4_PULL_BLOCKS") ? (size_t) std::atoll(std::getenv("HY4_PULL_BLOCKS")) : 3;
    return std::max<size_t>(1, v);
}
void publish(const float* src, float* dst, size_t n, unsigned* counter, unsigned* flag, unsigned v, cudaStream_t s,
             const float* plus) {
    if (n % 4) {
        std::fprintf(stderr, "maya-hy4 publish: %zu values (a multiple of 4 expected)\n", n);
        g_err.fetch_add(1);
        return;
    }
    if (n > kViaKernel) {
        if (plus != nullptr)   // (the copy engine takes the buffer as it is: the addend added first)
            add_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(const_cast<float*>(src), plus, n);
        if (cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost, s) != cudaSuccess) {
            std::fprintf(stderr, "maya-hy4 publish: the copy failed\n");
            g_err.fetch_add(1);
            return;
        }
        set_flag_kernel<<<1, 1, 0, s>>>(flag, v);
        check("publish");
        submit(s);
        return;
    }
    const size_t n4 = n / 4;
    // (3 blocks at most, as the read-back's: a window's partial went out on up to 16 - measured equal or about 1%
    //  slower on eight RTX 5060 Ti.  HY4_PUB_BLOCKS: another count)
    static const size_t pub_blocks = std::getenv("HY4_PUB_BLOCKS") ? (size_t) std::atoll(std::getenv("HY4_PUB_BLOCKS")) : 3;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(std::max<size_t>(1, pub_blocks), (n4 + 255) / 256));
    publish_kernel<<<blocks, 256, 0, s>>>((const float4*) src, (float4*) dst, n4, counter, flag, v, (const float4*) plus);
    check("publish");
    submit(s);
}
void prefetch(const Prefetch& j, unsigned* sink, cudaStream_t s, int max_blocks, int lines) {
    if (j.n <= 0 || j.pre[j.n] == 0) return;
    // (a thread a few lines: 16 thousand threads and more read 5 MB at the memory's speed, 4 thousand at half of it;
    //  lines 1: a small job's - a thread a line where the blocks allow, each line a memory trip the thread waits for)
    const unsigned blocks = (unsigned) std::max<uint32_t>(
        lines > 1 ? 8 : 1, std::min<uint32_t>(max_blocks > 0 ? (uint32_t) max_blocks : 128u,
                                             (j.pre[j.n] + 256 * (uint32_t) lines - 1) / (256 * (uint32_t) lines)));
    prefetch_kernel<<<blocks, 256, 0, s>>>(j, sink);
    check("prefetch");
}
void wait_pull(const unsigned* flag, unsigned v, float* dst, const float* src, size_t n, cudaStream_t s) {
    if (n % 4) {
        std::fprintf(stderr, "maya-hy4 wait_pull: %zu values (a multiple of 4 expected)\n", n);
        g_err.fetch_add(1);
        return;
    }
    if (n > kViaKernel) {
        wait_flag_kernel<<<1, 32, 0, s>>>(flag, v);
        if (cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyHostToDevice, s) != cudaSuccess) {
            std::fprintf(stderr, "maya-hy4 wait_pull: the copy failed\n");
            g_err.fetch_add(1);
        }
        check("wait_pull");
        return;
    }
    const size_t n4 = n / 4;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(pull_blocks(), (n4 + 255) / 256));
    wait_pull_kernel<<<blocks, 256, 0, s>>>(flag, v, (float4*) dst, (const float4*) src, n4);
    check("wait_pull");
}
void wait_flag(const unsigned* flag, unsigned v, cudaStream_t s) {
    wait_flag_kernel<<<1, 32, 0, s>>>(flag, v);
    check("wait_flag");
}
void set_flag(unsigned* flag, unsigned v, cudaStream_t s) {
    set_flag_kernel<<<1, 1, 0, s>>>(flag, v);
    check("set_flag");
    submit(s);   // (the host's summing threads spin on it)
}
__global__ void pack3_kernel(const unsigned* a, size_t na, const unsigned* b, size_t nb, const unsigned* c, size_t nc,
                             unsigned* dst, int split) {
    const size_t n = na + nb + nc;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const unsigned* src = i < na ? a + i : i < na + nb ? b + (i - na) : c + (i - na - nb);
        if (!split) dst[i] = *src;
        else *(unsigned*) src = dst[i];
    }
}
void pack3(const void* a, size_t na, const void* b, size_t nb, const void* c, size_t nc, void* dst, cudaStream_t s) {
    const size_t n = na + nb + nc;
    pack3_kernel<<<(unsigned) std::max<size_t>(1, std::min<size_t>(1024, (n + 255) / 256)), 256, 0, s>>>(
        (const unsigned*) a, na, (const unsigned*) b, nb, (const unsigned*) c, nc, (unsigned*) dst, 0);
    check("pack3");
}
void split3(const void* src, void* a, size_t na, void* b, size_t nb, void* c, size_t nc, cudaStream_t s) {
    const size_t n = na + nb + nc;
    pack3_kernel<<<(unsigned) std::max<size_t>(1, std::min<size_t>(1024, (n + 255) / 256)), 256, 0, s>>>(
        (const unsigned*) a, na, (const unsigned*) b, nb, (const unsigned*) c, nc, (unsigned*) src, 1);
    check("split3");
}
__global__ void __launch_bounds__(256) pack16_kernel(const float* __restrict__ x, int E, uint16_t* __restrict__ h,
                                                     float* __restrict__ sc) {
    const int r = blockIdx.x;
    const float* xr = x + (size_t) r * E;
    float am = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) am = fmaxf(am, fabsf(xr[i]));
    __shared__ float s_am[8];
    am = warp_max(am);
    if ((threadIdx.x & 31) == 0) s_am[threadIdx.x >> 5] = am;
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = s_am[0];
        for (int w = 1; w < (int) (blockDim.x >> 5); ++w) a = fmaxf(a, s_am[w]);
        float f = 1.0f;
        while (a / f > 32768.0f && f < 1e30f) f *= 2.0f;
        s_am[0] = f;
        sc[r] = f;
    }
    __syncthreads();
    const float inv = 1.0f / s_am[0];
    for (int i = threadIdx.x; i < E; i += blockDim.x) h[(size_t) r * E + i] = __half_as_ushort(__float2half_rn(xr[i] * inv));
}
__global__ void unpack16_kernel(const uint16_t* __restrict__ h, const float* __restrict__ sc, int E, size_t n,
                                float* __restrict__ y) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        y[i] = __half2float(__ushort_as_half(h[i])) * sc[i / (size_t) E];
}
void pack16_rows(const float* x, int rows, int E, uint16_t* h, float* sc, cudaStream_t s) {
    if (rows <= 0) return;
    pack16_kernel<<<(unsigned) rows, 256, 0, s>>>(x, E, h, sc);
    check("pack16_rows");
}
void unpack16_rows(const uint16_t* h, const float* sc, int rows, int E, float* y, cudaStream_t s) {
    const size_t n = (size_t) rows * E;
    if (n == 0) return;
    unpack16_kernel<<<(unsigned) std::min<size_t>(4096, (n + 255) / 256), 256, 0, s>>>(h, sc, E, n, y);
    check("unpack16_rows");
}
__global__ void expand_kernel(const float* __restrict__ e, int hc, int E, size_t n, float* __restrict__ x) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const size_t t = i / ((size_t) hc * E), d = i % (size_t) E;
        x[i] = e[t * (size_t) E + d];
    }
}
void expand_streams(const float* e, int T, int hc, int E, float* x, cudaStream_t s) {
    const size_t n = (size_t) T * hc * E;
    if (n == 0) return;
    expand_kernel<<<(unsigned) std::min<size_t>(4096, (n + 255) / 256), 256, 0, s>>>(e, hc, E, n, x);
    check("expand_streams");
}
unsigned wait_timeouts() {
    unsigned n = 0;
    if (cudaMemcpyFromSymbol(&n, g_wait_timeouts, sizeof n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
}

void top1(const float* row, int n, float* out, cudaStream_t s) {
    static const bool one = std::getenv("HY4_TOP1_ONE") != nullptr;   // (the earlier one-block kernel)
    if (one) {
        top1_kernel<<<1, 1024, 0, s>>>(row, n, out);
    } else {
        const int blocks = std::max(1, std::min(kTop1Blocks, (n + 4095) / 4096));
        top1_multi_kernel<false><<<blocks, 256, 0, s>>>(row, n, out);
    }
    check("top1");
}
void top1_part(const float* row, int n, float* out, cudaStream_t s) {
    const int blocks = std::max(1, std::min(kTop1Blocks, (n + 4095) / 4096));
    top1_multi_kernel<true><<<blocks, 256, 0, s>>>(row, n, out);
    check("top1_part");
}
// a block a row (256 threads): each thread's first largest of its stride, then the warps' and the block's - of equal
// values the lower index
__global__ void __launch_bounds__(256) row_top_kernel(const float* __restrict__ x, int n, size_t ld, int i0,
                                                      float* __restrict__ out) {
    __shared__ float sv[8];
    __shared__ int si[8];
    const float* row = x + (size_t) blockIdx.x * ld;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    int nan = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        nan |= v != v;
        if (v > best) best = v, bi = i;
    }
    nan = __syncthreads_or(nan);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ob > best || (ob == best && oi < bi)) best = ob, bi = oi;
    }
    if (l == 0) sv[w] = best, si[w] = bi;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < (int) (blockDim.x >> 5); ++k)
            if (sv[k] > sv[0] || (sv[k] == sv[0] && si[k] < si[0])) sv[0] = sv[k], si[0] = si[k];
        const bool none = nan || si[0] == 0x7fffffff;
        out[2 * blockIdx.x] = __int_as_float(none ? -1 : i0 + si[0]);
        out[2 * blockIdx.x + 1] = sv[0];
    }
}
void row_top(const float* x, int rows, int n, size_t ld, int i0, float* out, cudaStream_t s) {
    if (rows <= 0) return;
    row_top_kernel<<<rows, 256, 0, s>>>(x, n, ld, i0, out);
    check("row_top");
}
int errors() { return g_err.exchange(0); }
std::string first_fault() {
    std::lock_guard<std::mutex> lk(g_fault_mu);
    return g_fault;
}
void set_mix_out(float* p) { g_mix_out = p; }

}  // namespace hy4::k
