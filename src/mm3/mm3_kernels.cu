// src/mm3/mm3_kernels.cu - maya-mm3's kernels around the EXL3 matrices (the interface: mm3_kernels.cuh).
#include "mm3_kernels.cuh"

#include <cuda_fp16.h>
#if !defined(STRATA_USE_HIP)
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#endif
#include "../ds4/ds4_numerics.cuh"   // (BF16 / FP8 on HIP: the same bits in plain arithmetic)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace mm3::k {
namespace {

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
#if !defined(STRATA_USE_HIP)
    return __bfloat16_as_ushort(__float2bfloat16_rn(f));
#else
    return strata::ds4::bf16_rn(f);
#endif
}
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// the sum of v over the block (blockDim.x a multiple of 32, at most 1024), every thread gets it
__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = l < nw ? red[l] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) red[32] = t;
    __syncthreads();
    return red[32];
}

// the block's max of |v| (every thread gets it)
__device__ float block_absmax(float v, float* red) {
    v = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float m = 0.0f;
    for (int i = 0; i < nw; ++i) m = fmaxf(m, red[i]);
    return m;
}

// the device's waits on the host give up after kWaitLimitNs (a host that never answers - a bug, a crash - must not leave
// the GPU spinning and every later host wait hung): counted in g_wait_timeouts, which wait_timeouts() reads
__device__ unsigned g_wait_timeouts = 0;
constexpr unsigned long long kWaitLimitNs = 120ull * 1000000000ull;
__device__ __forceinline__ unsigned long long gtimer() {
#if defined(STRATA_USE_HIP)
    return wall_clock64() * 10ull;   // (RDNA's constant 100 MHz clock, in ns: a safety timeout, not a measurement)
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "mm3 %s: %s\n", what, cudaGetErrorString(e));
}

// ------------------------------------------------------------------ RMSNorm (Gemma: the weight is 1 + w, baked)
__global__ void rms_kernel(const float* x, const float* w1, int D, float eps, float* y, uint16_t* y16) {
    __shared__ float red[33];
    const float* xr = x + (size_t) blockIdx.x * D;
    float ss = 0.0f;
    int i0 = threadIdx.x;
    // (8 loads in flight, then their sums in the loop's order: the same bits, without a load's latency a value)
    for (const int st = (int) blockDim.x; i0 + 7 * st < D; i0 += 8 * st) {
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = xr[i0 + u * st];
#pragma unroll
        for (int u = 0; u < 8; ++u) ss = fmaf(v[u], v[u], ss);
    }
    for (int i = i0; i < D; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
    const float r = rsqrtf(block_sum(ss, red) / (float) D + eps);
    float* yr = y + (size_t) blockIdx.x * D;
    uint16_t* hr = y16 ? y16 + (size_t) blockIdx.x * D : nullptr;
    int j0 = threadIdx.x;
    for (const int st = (int) blockDim.x; j0 + 7 * st < D; j0 += 8 * st) {   // (the same, 8 values in flight)
        float a[8], b[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            a[u] = xr[j0 + u * st];
            b[u] = w1[j0 + u * st];
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float v = a[u] * r * b[u];
            yr[j0 + u * st] = v;
            if (hr) hr[j0 + u * st] = f2h(v);
        }
    }
    for (int i = j0; i < D; i += blockDim.x) {
        const float v = xr[i] * r * w1[i];
        yr[i] = v;
        if (hr) hr[i] = f2h(v);
    }
}

// ------------------------------------------------------------------ one head: RMSNorm over hd then NeoX RoPE
// (blockDim.x = hd = 128; the rotary on dims < n_rot: out[d] = x[d] c - x[d + n_rot/2] s, out[d + n_rot/2] =
//  x[d + n_rot/2] c + x[d] s for d < n_rot / 2, as transformers' rotate_half)
__device__ float norm_rope(float x, const float* w1, int pos, int n_rot, const float* inv, float eps, float* sh,
                           float* red) {
    const int d = threadIdx.x, hd = blockDim.x;
    const float r = rsqrtf(block_sum(x * x, red) / (float) hd + eps);
    const float v = x * r * w1[d];
    sh[d] = v;
    __syncthreads();
    float o = v;
    const int half = n_rot / 2;
    if (d < n_rot) {
        const int i = d < half ? d : d - half;
        const float a = (float) pos * inv[i];
        const float c = cosf(a), s = sinf(a);
        o = d < half ? v * c - sh[d + half] * s : v * c + sh[d - half] * s;
    }
    __syncthreads();
    return o;
}

// ---------------------------------------------------------------- the K / V caches' rows (mm3_kernels.cuh: KvQ)
// the 128-point Sylvester transform over sqrt(128) of the 128 values a warp holds 4 per lane (value lane * 4 + i) -
// mm3_exl3.cu's had128
__device__ __forceinline__ void had128w(float* v, int lane) {
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
    for (int i = 0; i < 4; ++i) v[i] *= 0.088388347648f;
}
// the same transform of the 128 values a block of 128 threads holds one each (value threadIdx.x); sh: 128 floats
__device__ float had_block(float v, float* sh) {
    const int d = threadIdx.x, lane = d & 31;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, m);
        v = (lane & m) ? o - v : v + o;
    }
    __syncthreads();
    sh[d] = v;
    __syncthreads();
    {
        const float o = sh[d ^ 32];
        v = (d & 32) ? o - v : v + o;
    }
    __syncthreads();
    sh[d] = v;
    __syncthreads();
    {
        const float o = sh[d ^ 64];
        v = (d & 64) ? o - v : v + o;
    }
    __syncthreads();
    return v * 0.088388347648f;
}
// E2M1 (FP4): magnitudes 0 .5 1 1.5 2 3 4 6, the nearest (ties to even), its code (bit 3 the sign) and a code's value
__device__ __forceinline__ float e2m1_rt(float v) {
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysignf(r, v);
}
__device__ __forceinline__ uint32_t e2m1_code(float q) {
    const float a = fabsf(q);
    const uint32_t m = a <= 1.5f ? (uint32_t) (a * 2.0f) : a <= 2.0f ? 4u : a <= 3.0f ? 5u : a <= 4.0f ? 6u : 7u;
    return m | ((__float_as_uint(q) >> 31) << 3);
}
__device__ __forceinline__ float e2m1_val(uint32_t c) {
    const uint32_t m = c & 7u;
    const float a = m < 4u ? 0.5f * (float) m : m == 4u ? 2.0f : m == 5u ? 3.0f : m == 6u ? 4.0f : 6.0f;
    return (c & 8u) ? -a : a;
}
// E4M3 (FP8): the nearest of v (saturated at 448), and a code's value (exact; no NaN codes are written)
__device__ __forceinline__ uint32_t e4m3_enc(float v) {
#if !defined(STRATA_USE_HIP)
    return (uint32_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#else   // (the same code: ds4_numerics' rounding - ties to even, saturated - then that value's bits)
    if (!(v == v)) return 0x7fu;
    const float a = fabsf(strata::ds4::e4m3_round(v));
    const uint32_t s = (__float_as_uint(v) >> 31) << 7, f = __float_as_uint(a);
    const int E = (int) ((f >> 23) & 0xffu) - 127;
    if (a == 0.0f) return s;
    if (E < -6) return s | (uint32_t) (a * 512.0f);                     // (subnormal: m 2^-9)
    return s | ((uint32_t) (E + 7) << 3) | ((f >> 20) & 7u);
#endif
}
__device__ __forceinline__ float e4m3_dec(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    const float a = e == 0u ? (float) m * 0.001953125f : __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -a : a;
}

// one cache row (a position's KV head, 128 values, value threadIdx.x of the block's 128 threads) in format fmt into
// code / sc (and hcode / hsc: KV streaming's host copy), rotated first when rot:
//   kKvI8   128 int8 round(v / s), s = max |v| / 127 (an F32 scale)
//   kKvFp8  128 E4M3 codes of v / 2^e, e the smallest with every |v| / 2^e of a group of 32 at most 448 (a UE8M0 byte a
//           group: e + 127)
//   kKvFp4  64 bytes of E2M1 codes (value 2i in the low nibble) of v / s, s the E4M3 of a group of 16's max |v| / 6
//           (an E4M3 byte a group) - NVFP4's
// red: 33 floats, sh: 128 floats of shared memory
__device__ void put_row(float v, int fmt, bool rot, uint8_t* code, uint8_t* sc, uint8_t* hcode, uint8_t* hsc,
                        float* red, float* sh) {
    const int d = threadIdx.x, lane = d & 31;
    if (rot) v = had_block(v, sh);
    if (fmt == kKvI8) {
        const float am = block_absmax(v, red);
        const float s = am / 127.0f;
        const int8_t q = s > 0.0f ? (int8_t) fminf(fmaxf(rintf(v / s), -127.0f), 127.0f) : (int8_t) 0;
        code[d] = (uint8_t) q;
        if (hcode) hcode[d] = (uint8_t) q;
        if (d == 0) {
            *(float*) sc = s;
            if (hsc) *(float*) hsc = s;
        }
    } else if (fmt == kKvFp8) {
        float am = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        int e = -127;
        if (am > 0.0f) {
            const uint32_t b = __float_as_uint(am / 448.0f);
            e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
            e = max(-127, min(127, e));
        }
        const float s = ldexpf(1.0f, e);
        const uint32_t q = e4m3_enc(v / s);
        code[d] = (uint8_t) q;
        if (hcode) hcode[d] = (uint8_t) q;
        if (lane == 0) {
            sc[d >> 5] = (uint8_t) (e + 127);
            if (hsc) hsc[d >> 5] = (uint8_t) (e + 127);
        }
    } else {   // kKvFp4
        float am = fabsf(v);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        const uint32_t sb = e4m3_enc(fmaxf(am / 6.0f, 0.001953125f));
        const float s = e4m3_dec(sb);
        const uint32_t c = e2m1_code(e2m1_rt(fminf(fmaxf(v / s, -6.0f), 6.0f)));
        const uint32_t hi = __shfl_down_sync(0xffffffffu, c, 1);   // (value d + 1: the next lane's)
        if ((d & 1) == 0) {
            code[d >> 1] = (uint8_t) (c | (hi << 4));
            if (hcode) hcode[d >> 1] = (uint8_t) (c | (hi << 4));
        }
        if ((d & 15) == 0) {
            sc[d >> 4] = (uint8_t) sb;
            if (hsc) hsc[d >> 4] = (uint8_t) sb;
        }
    }
}

// (q.kf / q.vf: K's / V's caches in that format instead - codes and scales, kc / vc unused for it.  hs.ring: KV
//  streaming - the VRAM rows a ring, row p % hs.ring tagged p, and position p's row in host memory as well)
__global__ void qk_kernel(float* q, const float* k, const float* v, const float* qn1, const float* kn1,
                          const int* pos, int Hq, int Hkv, int n_rot, const float* inv, float eps, uint16_t* kc,
                          uint16_t* vc, int ring, KvQ kq, KvHost hs) {
    __shared__ float red[33], sh[128];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, hd = blockDim.x;
    const int p = pos[t];
    const int row = ring > 0 ? p % ring : hs.ring > 0 ? p % hs.ring : p;
    const bool host = hs.ring > 0;
    if (h < Hq) {
        float* x = q + ((size_t) t * Hq + h) * hd;
        x[d] = norm_rope(x[d], qn1, p, n_rot, inv, eps, sh, red);
        return;
    }
    const bool is_k = h < Hq + Hkv;
    const int kh = is_k ? h - Hq : h - Hq - Hkv;
    const float o = is_k ? norm_rope(k[((size_t) t * Hkv + kh) * hd + d], kn1, p, n_rot, inv, eps, sh, red)
                         : v[((size_t) t * Hkv + kh) * hd + d];
    const size_t at = (size_t) row * Hkv + kh, ah = (size_t) p * Hkv + kh;
    const int fmt = is_k ? kq.kf : kq.vf;
    if (fmt != kKvBf16) {
        const size_t cb = kv_code_bytes(fmt), sb = kv_scale_bytes(fmt);
        uint8_t* code = (is_k ? kq.kc : kq.vc) + at * cb;
        uint8_t* sc = (is_k ? kq.ks : kq.vs) + at * sb;
        uint8_t* hcode = host ? (is_k ? hs.q.kc : hs.q.vc) + ah * cb : nullptr;
        uint8_t* hsc = host ? (is_k ? hs.q.ks : hs.q.vs) + ah * sb : nullptr;
        put_row(o, fmt, (kq.rot >> (is_k ? 0 : 1)) & 1, code, sc, hcode, hsc, red, sh);
    } else {
        const uint16_t b = f2bf(o);
        (is_k ? kc : vc)[at * hd + d] = b;
        if (host) (is_k ? hs.kc : hs.vc)[ah * hd + d] = b;
    }
    if (host && is_k && kh == 0 && d == 0) hs.tag[row] = p;
}

__global__ void index_kernel(float* iq, const float* ik, const float* qn1, const float* kn1, const int* pos, int Hi,
                             int n_rot, const float* inv, float eps, uint16_t* ic) {
    __shared__ float red[33], sh[128];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, D = blockDim.x;
    const int p = pos[t];
    if (h < Hi) {
        float* x = iq + ((size_t) t * Hi + h) * D;
        x[d] = bf2f(f2bf(norm_rope(x[d], qn1, p, n_rot, inv, eps, sh, red)));
    } else {
        ic[(size_t) p * D + d] = f2bf(norm_rope(ik[(size_t) t * D + d], kn1, p, n_rot, inv, eps, sh, red));
    }
}

// ------------------------------------------------------------------ MSA: block scores
// A block of 256 threads takes 16 vectors - 16 / Hi queries x Hi heads (Hi 1, 2 or 4) - and a span of the key blocks
// (blockIdx.y: blocks [y span, (y + 1) span), those past the tile's last query's block idle): thread (v = tid / 16,
// s = tid % 16) the keys s + 16 i (i < 8) of each key block, their max, then the max over s.  A block's score is one
// thread group's dot products and max whatever the tiles and spans, so the scores are the same bits for any T or span
// (the spans: a decode window's few queries over many blocks - one block walking them all left the GPU idle).
constexpr int kBsV = 16, kKP = 136;   // (key rows padded to 136 halves in shared memory)
__global__ void __launch_bounds__(256) block_scores_kernel(const float* iq, const uint16_t* ic, const int* pos, int T,
                                                           int Hi, int nblk, int blk, int span, float* bs) {
    __shared__ float qv[kBsV][129];
    __shared__ __align__(16) uint16_t kt[128][kKP];
    const int nqt = kBsV / Hi;   // the tile's queries
    const int t0 = blockIdx.x * nqt, tid = threadIdx.x;
    const int nq = min(nqt, T - t0);
    int pmax = 0;
    for (int qi = 0; qi < nq; ++qi) pmax = max(pmax, pos[t0 + qi]);
    const int last = min(pmax / blk, nblk - 1);
    const int b0 = blockIdx.y * span, b1 = min(last + 1, b0 + span);
    if (b0 > last) return;   // (the whole block: no barrier yet)
    for (int i = tid; i < kBsV * 128; i += 256) {
        const int v = i / 128, d = i % 128, qi = v / Hi, h = v % Hi;
        qv[v][d] = qi < nq ? iq[((size_t) (t0 + qi) * Hi + h) * 128 + d] : 0.0f;
    }
    const int v = tid >> 4, s = tid & 15, qi = v / Hi, h = v % Hi;
    const int myp = qi < nq ? pos[t0 + qi] : -1;
    for (int b = b0; b < b1; ++b) {
        __syncthreads();
        for (int i = tid; i < 128 * 16; i += 256) {   // 128 keys x 16 chunks of 8 halves
            const int j = i >> 4, c = i & 15;
            const int kp = b * blk + j;
            uint4 w = make_uint4(0, 0, 0, 0);
            if (kp <= pmax) w = *(const uint4*) (ic + (size_t) kp * 128 + c * 8);
            *(uint4*) &kt[j][c * 8] = w;
        }
        __syncthreads();
        float best = -INFINITY;
        if (qi < nq) {
#pragma unroll 1
            for (int i = 0; i < 8; ++i) {
                const int j = s + 16 * i, kp = b * blk + j;
                if (kp > myp) continue;
                float dot = 0.0f;
#pragma unroll 8
                for (int d = 0; d < 128; ++d) dot = fmaf(qv[v][d], bf2f(kt[j][d]), dot);
                best = fmaxf(best, dot);
            }
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
        if (s == 0 && qi < nq) bs[((size_t) h * T + t0 + qi) * nblk + b] = best;
    }
    // (blocks past the tile's last query's stay unwritten: select_blocks reads only blocks <= its query's)
}
// A prompt's: a block takes 64 vectors (64 / Hi queries x Hi heads) and a key block's keys 64 at a time, thread
// (tv = tid / 16, tk = tid % 16) the dot products of vectors 4 tv .. 4 tv + 3 with keys tk + 16 i (i < 4) of each 64 -
// every dot the same fmaf chain over d = 0 .. 127 as block_scores_kernel's, so the same bits (the max over a block's
// keys is exact in any order), at a quarter of its shared memory reads a product
// (KW keys a step: 64 on AMD - 50 KB of its 64 KB LDS; 32 under CUDA's 48 KB of static shared memory - the same bits)
constexpr int kBwV = 64, kBwP = 132;   // (rows padded to 132 floats / halves: 16 rows of keys, 16 banks)
#if defined(STRATA_USE_HIP)
constexpr int kBwK = 64;
#else
constexpr int kBwK = 32;
#endif
__global__ void __launch_bounds__(256) block_scores_wide_kernel(const float* iq, const uint16_t* ic, const int* pos,
                                                                int T, int Hi, int nblk, int blk, int span, float* bs) {
    constexpr int KI = kBwK / 16;   // (keys a thread a step)
    __shared__ float qv[kBwV][kBwP];
    __shared__ __align__(16) uint16_t kt[kBwK][kBwP];
    const int nqt = kBwV / Hi;   // the tile's queries
    const int t0 = blockIdx.x * nqt, tid = threadIdx.x;
    const int nq = min(nqt, T - t0);
    int pmax = 0;
    for (int qi = 0; qi < nq; ++qi) pmax = max(pmax, pos[t0 + qi]);
    const int last = min(pmax / blk, nblk - 1);
    const int b0 = blockIdx.y * span, b1 = min(last + 1, b0 + span);
    if (b0 > last) return;   // (the whole block: no barrier yet)
    for (int i = tid; i < kBwV * 128; i += 256) {
        const int v = i / 128, d = i % 128, qi = v / Hi, h = v % Hi;
        qv[v][d] = qi < nq ? iq[((size_t) (t0 + qi) * Hi + h) * 128 + d] : 0.0f;
    }
    const int tv = tid >> 4, tk = tid & 15;
    int myp[4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int qi = (tv * 4 + a) / Hi;
        myp[a] = qi < nq ? pos[t0 + qi] : -1;
    }
    for (int b = b0; b < b1; ++b) {
        float best[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (int k0 = 0; k0 < blk; k0 += kBwK) {
            __syncthreads();
            for (int i = tid; i < kBwK * 32; i += 256) {   // KW keys x 32 pieces of 4 halves
                const int j = i >> 5, c = i & 31;
                const int kp = b * blk + k0 + j;
                uint2 w = make_uint2(0u, 0u);
                if (kp <= pmax) w = *(const uint2*) (ic + (size_t) kp * 128 + c * 4);
                *(uint2*) &kt[j][c * 4] = w;
            }
            __syncthreads();
            float acc[4][KI];
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int i = 0; i < KI; ++i) acc[a][i] = 0.0f;
#pragma unroll 4
            for (int d = 0; d < 128; d += 2) {
                float q0[4], q1[4];
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    q0[a] = qv[tv * 4 + a][d];
                    q1[a] = qv[tv * 4 + a][d + 1];
                }
#pragma unroll
                for (int i = 0; i < KI; ++i) {
                    const uint32_t kw = *(const uint32_t*) &kt[tk + 16 * i][d];
                    const float x0 = __uint_as_float(kw << 16), x1 = __uint_as_float(kw & 0xffff0000u);
#pragma unroll
                    for (int a = 0; a < 4; ++a) {
                        acc[a][i] = fmaf(q0[a], x0, acc[a][i]);
                        acc[a][i] = fmaf(q1[a], x1, acc[a][i]);
                    }
                }
            }
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int i = 0; i < KI; ++i)
                    if (b * blk + k0 + tk + 16 * i <= myp[a]) best[a] = fmaxf(best[a], acc[a][i]);
        }
#pragma unroll
        for (int a = 0; a < 4; ++a) {
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) best[a] = fmaxf(best[a], __shfl_xor_sync(0xffffffffu, best[a], o));
            const int v = tv * 4 + a, qi = v / Hi, h = v % Hi;
            if (tk == 0 && qi < nq) bs[((size_t) h * T + t0 + qi) * nblk + b] = best[a];
        }
    }
}

// ------------------------------------------------------------------ MSA: the kept blocks
// one warp per (head, query): its own block, then topk - 1 rounds of the best block after the previous pick in the
// order (score desc, block asc) - the set torch.topk keeps, ties to the lower block.  stage: the warp's score row
// read once into shared memory (kSelWarps rows of nblk a block) for the rounds, else each round reads it again.  A
// round that finds nothing (scores all NaN) writes -1, as past the valid blocks (sparse_attn skips it).
__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }
constexpr int kSelWarps = 4;

__global__ void select_kernel(const float* bs, const int* pos, int T, int Hi, int nblk, int blk, int topk, int* kept,
                              bool stage) {
    extern __shared__ float srow[];
    const int wl = threadIdx.x >> 5, wid = blockIdx.x * kSelWarps + wl, lane = threadIdx.x & 31;
    if (wid >= T * Hi) return;
    const int h = wid / T, t = wid % T;
    const int qb = pos[t] / blk;
    int* out = kept + ((size_t) h * T + t) * topk;
    const float* row = bs + ((size_t) h * T + t) * nblk;
    if (qb + 1 <= topk) {
        for (int i = lane; i < topk; i += 32) out[i] = i <= qb ? i : -1;
        return;
    }
    if (stage) {
        float* sr = srow + (size_t) wl * nblk;
        for (int b = lane; b < qb; b += 32) sr[b] = row[b];
        __syncwarp();
        row = sr;
    }
    if (lane == 0) out[0] = qb;
    float ps = INFINITY;
    int pi = -1;
    for (int r = 1; r < topk; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int b = lane; b < qb; b += 32) {
            const float sv = row[b];
            if (better(ps, pi, sv, b) && better(sv, b, bv, bi)) {
                bv = sv;
                bi = b;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) out[r] = bi == 0x7fffffff ? -1 : bi;
        ps = bv;
        pi = bi;
    }
}

// ------------------------------------------------------------------ MSA: attention over the kept blocks
// one block of 128 threads per (query, KV group): its 16 query heads over 64-key tiles of the kept blocks, online
// softmax in F32
constexpr int kG = 16, kTile = 64, kRow = 130;   // (K / V rows padded to 130 halves)
// key kp's row: its VRAM row (with KV streaming its ring row, when the ring holds kp), else (host) its host row
__device__ __forceinline__ size_t kv_row(const KvHost& hs, int kp, bool& host) {
    if (hs.ring <= 0) return (size_t) kp;
    const int r = kp % hs.ring;
    host = hs.tag[r] != kp;
    return host ? (size_t) kp : (size_t) r;
}

__device__ __forceinline__ uint32_t bf2pack(float a, float b) {
    return (uint32_t) f2bf(a) | ((uint32_t) f2bf(b) << 16);
}

// one operand's tile (64 keys of KV group g from k0 on, keys past p zero) into shared memory: BF16 rows in t16 - FP8 /
// FP4 rows decoded on the way (exact: their values are BF16 values) - or (F kKvI8) int8 rows in t8 and their scales in
// tsc.  c16: the BF16 cache, cq / sq: the format's codes and scales (VRAM; host: hc16, hcq, hsq)
template <int F, int NT = 128>
__device__ __forceinline__ void load_tile(int k0, int p, int g, int Hkv, const KvHost& hs, bool is_k,
                                          const uint16_t* c16, const uint8_t* cq, const uint8_t* sq,
                                          uint16_t (*t16)[kRow], int8_t (*t8)[2 * kRow], float* tsc) {
    const int tid = threadIdx.x;
    const uint16_t* h16 = is_k ? hs.kc : hs.vc;
    const uint8_t* hcq = is_k ? hs.q.kc : hs.q.vc;
    const uint8_t* hsq = is_k ? hs.q.ks : hs.q.vs;
    if constexpr (F == kKvBf16) {
        for (int i = tid; i < kTile * 64; i += NT) {   // 64 keys x 64 pairs of halves
            const int j = i >> 6, c = i & 63;
            const int kp = k0 + j;
            uint32_t w = 0;
            if (kp <= p) {
                bool fh = false;
                const size_t r = kv_row(hs, kp, fh);
                w = *(const uint32_t*) ((fh ? h16 : c16) + (r * Hkv + g) * 128 + c * 2);
            }
            *(uint32_t*) &t16[j][c * 2] = w;
        }
    } else if constexpr (F == kKvI8) {
        for (int i = tid; i < kTile * 32; i += NT) {   // 64 keys x 32 words of 4 int8
            const int j = i >> 5, c = i & 31;
            const int kp = k0 + j;
            uint32_t w = 0;
            if (kp <= p) {
                bool fh = false;
                const size_t r = kv_row(hs, kp, fh);
                w = *(const uint32_t*) ((fh ? hcq : cq) + (r * Hkv + g) * 128 + c * 4);
            }
            *(uint32_t*) &t8[j][c * 4] = w;
        }
        if (tid < kTile) {
            const int kp = k0 + tid;
            float a = 0.0f;
            if (kp <= p) {
                bool fh = false;
                const size_t r = kv_row(hs, kp, fh);
                a = *(const float*) ((fh ? hsq : sq) + (r * Hkv + g) * 4);
            }
            tsc[tid] = a;
        }
    } else if constexpr (F == kKvFp8) {
        for (int i = tid; i < kTile * 32; i += NT) {   // 64 keys x 32 words of 4 E4M3 codes (a scale byte a group of 32)
            const int j = i >> 5, c = i & 31;
            const int kp = k0 + j;
            uint32_t w = 0, sb = 0;
            if (kp <= p) {
                bool fh = false;
                const size_t r = kv_row(hs, kp, fh);
                w = *(const uint32_t*) ((fh ? hcq : cq) + (r * Hkv + g) * 128 + c * 4);
                sb = (fh ? hsq : sq)[(r * Hkv + g) * 4 + (c >> 3)];
            }
            const float s = __uint_as_float(sb << 23);   // (2^(sb - 127); 0 for an unwritten row)
            uint32_t* dst = (uint32_t*) &t16[j][c * 4];
            dst[0] = bf2pack(e4m3_dec(w & 0xffu) * s, e4m3_dec((w >> 8) & 0xffu) * s);
            dst[1] = bf2pack(e4m3_dec((w >> 16) & 0xffu) * s, e4m3_dec(w >> 24) * s);
        }
    } else {   // kKvFp4
        for (int i = tid; i < kTile * 16; i += NT) {   // 64 keys x 16 words of 8 E2M1 codes (an E4M3 scale a 16)
            const int j = i >> 4, c = i & 15;
            const int kp = k0 + j;
            uint32_t w = 0, sb = 0;
            if (kp <= p) {
                bool fh = false;
                const size_t r = kv_row(hs, kp, fh);
                w = *(const uint32_t*) ((fh ? hcq : cq) + (r * Hkv + g) * 64 + c * 4);
                sb = (fh ? hsq : sq)[(r * Hkv + g) * 8 + (c >> 1)];
            }
            const float s = e4m3_dec(sb);
            uint32_t* dst = (uint32_t*) &t16[j][c * 8];
#pragma unroll
            for (int x = 0; x < 4; ++x)
                dst[x] = bf2pack(e2m1_val((w >> (8 * x)) & 15u) * s, e2m1_val((w >> (8 * x + 4)) & 15u) * s);
        }
    }
}

// KF / VF: K's and V's cache formats (kKvI8: a key's dot product over its int8 values times its scale, a value's scale
// folded into its probability; FP8 / FP4: decoded to BF16 rows in the tile).  rot (bit 0 K, bit 1 V): the rows are
// stored rotated (H k, H v - the 128-point transform, its own inverse): the queries are rotated alike, the output
// rotated back.  S > 1: block z of S takes the kept blocks [z topk / S, (z + 1) topk / S) and leaves its unnormalised
// sums, max and total in part [T][Hkv][S][kG][kPart] for attn_combine - a query's split is S's alone, whatever T (a
// window's rows and the token path's agree)
constexpr int kPart = 130;   // (a head's partial: 128 sums, the max, the total)

template <int KF, int VF>
__global__ void __launch_bounds__(128) sparse_attn_kernel(const float* q, const uint16_t* kc, const uint16_t* vc,
                                                          const int* kept, const int* pos, int T, int Hq, int Hkv,
                                                          int topk, int blk, float scale, float* out, KvQ kq,
                                                          KvHost hs, int S, float* part) {
    __shared__ float qs[kG][128];
    __shared__ __align__(4) uint16_t ks[kTile][kRow], vs[kTile][kRow];
    __shared__ float ksc[kTile], vsc[kTile];
    int8_t (*k8)[2 * kRow] = (int8_t (*)[2 * kRow]) ks;   // (int8: the tiles as int8 rows, 260 bytes apart)
    int8_t (*v8)[2 * kRow] = (int8_t (*)[2 * kRow]) vs;
    __shared__ float sc[kG][kTile + 1];
    __shared__ float mrow[kG], lrow[kG], arow[kG];
    const int t = blockIdx.x, g = blockIdx.y, z = blockIdx.z, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const int p = pos[t];
    const int bi0 = z * topk / S, bi1 = (z + 1) * topk / S;
    const bool krot = KF != kKvBf16 && (kq.rot & 1), vrot = VF != kKvBf16 && (kq.rot & 2);
    for (int i = tid; i < kG * 128; i += 128) {
        const int h = i / 128, d = i % 128;
        qs[h][d] = q[((size_t) t * Hq + g * kG + h) * 128 + d] * scale;
    }
    if (krot) {   // (the queries rotated as the keys: (H q) . (H k) = q . k)
        __syncthreads();
        for (int h = wp; h < kG; h += 4) {
            float x[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) x[i] = qs[h][lane * 4 + i];
            had128w(x, lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) qs[h][lane * 4 + i] = x[i];
        }
    }
    if (tid < kG) {
        mrow[tid] = -INFINITY;
        lrow[tid] = 0.0f;
    }
    float acc[kG];
#pragma unroll
    for (int h = 0; h < kG; ++h) acc[h] = 0.0f;
    const int* kb = kept + ((size_t) g * T + t) * topk;
    for (int bi = bi0; bi < bi1; ++bi) {
        const int b = kb[bi];
        if (b < 0) continue;
        for (int sub = 0; sub < blk / kTile; ++sub) {
            const int k0 = b * blk + sub * kTile;
            if (k0 > p) break;
            __syncthreads();
            // (KV streaming: a key from its VRAM ring row when that row holds it, else from host memory)
            load_tile<KF>(k0, p, g, Hkv, hs, true, kc, kq.kc, kq.ks, ks, k8, ksc);
            load_tile<VF>(k0, p, g, Hkv, hs, false, vc, kq.vc, kq.vs, vs, v8, vsc);
            __syncthreads();
            {   // scores: thread (j = tid % 64) for the heads (tid / 64) * 8 .. + 8
                const int j = tid & 63, h0 = (tid >> 6) * 8;
                const bool ok = k0 + j <= p;
                float s8[8];
#pragma unroll
                for (int h = 0; h < 8; ++h) s8[h] = 0.0f;
                if (ok) {
#pragma unroll 4
                    for (int d = 0; d < 128; ++d) {
                        const float kv = KF == kKvI8 ? (float) k8[j][d] : bf2f(ks[j][d]);
#pragma unroll
                        for (int h = 0; h < 8; ++h) s8[h] = fmaf(qs[h0 + h][d], kv, s8[h]);
                    }
                    if constexpr (KF == kKvI8) {
#pragma unroll
                        for (int h = 0; h < 8; ++h) s8[h] *= ksc[j];
                    }
                }
#pragma unroll
                for (int h = 0; h < 8; ++h) sc[h0 + h][j] = ok ? s8[h] : -INFINITY;
            }
            __syncthreads();
            if (tid < kG) {   // the row's new max, the old sum's factor, the tile's probabilities
                const int h = tid;
                float m = mrow[h];
                for (int j = 0; j < kTile; ++j) m = fmaxf(m, sc[h][j]);
                const float a = mrow[h] == -INFINITY ? 0.0f : expf(mrow[h] - m);
                float l = lrow[h] * a;
                for (int j = 0; j < kTile; ++j) {
                    const float e = sc[h][j] == -INFINITY ? 0.0f : expf(sc[h][j] - m);
                    sc[h][j] = e;
                    l += e;
                }
                mrow[h] = m;
                lrow[h] = l;
                arow[h] = a;
            }
            __syncthreads();
            {   // the values: thread d over the 16 heads
                const int d = tid;
#pragma unroll
                for (int h = 0; h < kG; ++h) {
                    float s = acc[h] * arow[h];
                    if constexpr (VF == kKvI8) {
                        for (int j = 0; j < kTile; ++j) s = fmaf(sc[h][j] * vsc[j], (float) v8[j][d], s);
                    } else {
                        for (int j = 0; j < kTile; ++j) s = fmaf(sc[h][j], bf2f(vs[j][d]), s);
                    }
                    acc[h] = s;
                }
            }
        }
    }
    __syncthreads();
    if (S > 1) {   // (the split's partials: attn_combine finishes)
        float* pz = part + (((size_t) t * Hkv + g) * S + z) * kG * kPart;
#pragma unroll
        for (int h = 0; h < kG; ++h) pz[h * kPart + tid] = acc[h];
        if (tid < kG) {
            pz[tid * kPart + 128] = mrow[tid];
            pz[tid * kPart + 129] = lrow[tid];
        }
        return;
    }
    if (!vrot) {
#pragma unroll
        for (int h = 0; h < kG; ++h)
            out[((size_t) t * Hq + g * kG + h) * 128 + tid] = lrow[h] > 0.0f ? acc[h] / lrow[h] : 0.0f;
        return;
    }
    // (the values were rotated: the output rotated back, H (sum p H v) = sum p v)
#pragma unroll
    for (int h = 0; h < kG; ++h) qs[h][tid] = lrow[h] > 0.0f ? acc[h] / lrow[h] : 0.0f;
    __syncthreads();
    for (int h = wp; h < kG; h += 4) {
        float x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = qs[h][lane * 4 + i];
        had128w(x, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) out[((size_t) t * Hq + g * kG + h) * 128 + lane * 4 + i] = x[i];
    }
}

// A decode token's or a verify window's few queries (T <= kWideT): sparse_attn_kernel with 512 threads a block -
// a token gives the GPU T x Hkv blocks only, so each one's tiles are its latency: four times the threads take a tile's
// scores (2 heads a thread), its probabilities and its values (4 heads a thread) in a quarter of the steps.  Every
// score, probability, sum and output is computed as sparse_attn_kernel computes it - the same operations in the same
// order (a head's sum over a tile's keys still one thread's, in key order) - so the same bits.
constexpr int kWideT = 16, kWideNT = 512;
template <int KF, int VF>
__global__ void __launch_bounds__(kWideNT) sparse_attn_w_kernel(const float* q, const uint16_t* kc, const uint16_t* vc,
                                                                const int* kept, const int* pos, int T, int Hq, int Hkv,
                                                                int topk, int blk, float scale, float* out, KvQ kq,
                                                                KvHost hs, int S, float* part) {
    constexpr int NT = kWideNT, HV = kG / (NT / 128), HS = 8 / (NT / 128);   // (heads a thread: values, scores)
    __shared__ float qs[kG][128];
    __shared__ __align__(4) uint16_t ks[kTile][kRow], vs[kTile][kRow];
    __shared__ float ksc[kTile], vsc[kTile];
    int8_t (*k8)[2 * kRow] = (int8_t (*)[2 * kRow]) ks;
    int8_t (*v8)[2 * kRow] = (int8_t (*)[2 * kRow]) vs;
    __shared__ float sc[kG][kTile + 1];
    __shared__ float mrow[kG], lrow[kG], arow[kG], mnew[kG];
    const int t = blockIdx.x, g = blockIdx.y, z = blockIdx.z, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const int d = tid & 127, hq = tid >> 7;   // (values: dimension d of heads hq HV .. hq HV + HV - 1)
    const int p = pos[t];
    const int bi0 = z * topk / S, bi1 = (z + 1) * topk / S;
    const bool krot = KF != kKvBf16 && (kq.rot & 1), vrot = VF != kKvBf16 && (kq.rot & 2);
    for (int i = tid; i < kG * 128; i += NT) {
        const int h = i / 128, dd = i % 128;
        qs[h][dd] = q[((size_t) t * Hq + g * kG + h) * 128 + dd] * scale;
    }
    if (krot) {
        __syncthreads();
        for (int h = wp; h < kG; h += NT / 32) {
            float x[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) x[i] = qs[h][lane * 4 + i];
            had128w(x, lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) qs[h][lane * 4 + i] = x[i];
        }
    }
    if (tid < kG) {
        mrow[tid] = -INFINITY;
        lrow[tid] = 0.0f;
    }
    float acc[HV];
#pragma unroll
    for (int h = 0; h < HV; ++h) acc[h] = 0.0f;
    const int* kb = kept + ((size_t) g * T + t) * topk;
    for (int bi = bi0; bi < bi1; ++bi) {
        const int b = kb[bi];
        if (b < 0) continue;
        for (int sub = 0; sub < blk / kTile; ++sub) {
            const int k0 = b * blk + sub * kTile;
            if (k0 > p) break;
            __syncthreads();
            load_tile<KF, NT>(k0, p, g, Hkv, hs, true, kc, kq.kc, kq.ks, ks, k8, ksc);
            load_tile<VF, NT>(k0, p, g, Hkv, hs, false, vc, kq.vc, kq.vs, vs, v8, vsc);
            __syncthreads();
            {   // scores: thread (j = tid % 64) for the heads (tid / 64) * HS .. + HS
                const int j = tid & 63, h0 = (tid >> 6) * HS;
                const bool ok = k0 + j <= p;
                float s8[HS];
#pragma unroll
                for (int h = 0; h < HS; ++h) s8[h] = 0.0f;
                if (ok) {
#pragma unroll 4
                    for (int dd = 0; dd < 128; ++dd) {
                        const float kv = KF == kKvI8 ? (float) k8[j][dd] : bf2f(ks[j][dd]);
#pragma unroll
                        for (int h = 0; h < HS; ++h) s8[h] = fmaf(qs[h0 + h][dd], kv, s8[h]);
                    }
                    if constexpr (KF == kKvI8) {
#pragma unroll
                        for (int h = 0; h < HS; ++h) s8[h] *= ksc[j];
                    }
                }
#pragma unroll
                for (int h = 0; h < HS; ++h) sc[h0 + h][j] = ok ? s8[h] : -INFINITY;
            }
            __syncthreads();
            if (tid < kG) {   // each head's new max
                float m = mrow[tid];
                for (int j = 0; j < kTile; ++j) m = fmaxf(m, sc[tid][j]);
                mnew[tid] = m;
            }
            __syncthreads();
            for (int i = tid; i < kG * kTile; i += NT) {   // the tile's probabilities, every thread
                const int h = i / kTile, j = i % kTile;
                const float sv = sc[h][j];
                sc[h][j] = sv == -INFINITY ? 0.0f : expf(sv - mnew[h]);
            }
            __syncthreads();
            if (tid < kG) {   // the old sum's factor, the new sum (in key order)
                const int h = tid;
                const float m = mnew[h];
                const float a = mrow[h] == -INFINITY ? 0.0f : expf(mrow[h] - m);
                float l = lrow[h] * a;
                for (int j = 0; j < kTile; ++j) l += sc[h][j];
                mrow[h] = m;
                lrow[h] = l;
                arow[h] = a;
            }
            __syncthreads();
            {   // the values: thread (d, hq) over its HV heads
#pragma unroll
                for (int i = 0; i < HV; ++i) {
                    const int h = hq * HV + i;
                    float s = acc[i] * arow[h];
                    if constexpr (VF == kKvI8) {
                        for (int j = 0; j < kTile; ++j) s = fmaf(sc[h][j] * vsc[j], (float) v8[j][d], s);
                    } else {
                        for (int j = 0; j < kTile; ++j) s = fmaf(sc[h][j], bf2f(vs[j][d]), s);
                    }
                    acc[i] = s;
                }
            }
        }
    }
    __syncthreads();
    if (S > 1) {   // (the split's partials: attn_combine finishes)
        float* pz = part + (((size_t) t * Hkv + g) * S + z) * kG * kPart;
#pragma unroll
        for (int i = 0; i < HV; ++i) pz[(hq * HV + i) * kPart + d] = acc[i];
        if (tid < kG) {
            pz[tid * kPart + 128] = mrow[tid];
            pz[tid * kPart + 129] = lrow[tid];
        }
        return;
    }
    if (!vrot) {
#pragma unroll
        for (int i = 0; i < HV; ++i) {
            const int h = hq * HV + i;
            out[((size_t) t * Hq + g * kG + h) * 128 + d] = lrow[h] > 0.0f ? acc[i] / lrow[h] : 0.0f;
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < HV; ++i) {
        const int h = hq * HV + i;
        qs[h][d] = lrow[h] > 0.0f ? acc[i] / lrow[h] : 0.0f;
    }
    __syncthreads();
    for (int h = wp; h < kG; h += NT / 32) {
        float x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = qs[h][lane * 4 + i];
        had128w(x, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) out[((size_t) t * Hq + g * kG + h) * 128 + lane * 4 + i] = x[i];
    }
}

// sparse_attn's S partials of query t, KV group g -> out: the heads' max over the splits, each split's sums and total
// scaled to it, added in split order (rot: the rows through H)
// (a block a head - grid (T, Hkv, 16): each head's values are its own, so the same bits as one block walking the 16
//  heads in turn, which left the GPU idle; each split's max and total read before the sums)
constexpr int kCombMaxS = 32;
__global__ void __launch_bounds__(128) attn_combine_kernel(const float* part, int Hq, int Hkv, int S, bool rot,
                                                           float* out) {
    __shared__ float o[128];
    __shared__ float ms[kCombMaxS], ls[kCombMaxS];
    const int t = blockIdx.x, g = blockIdx.y, h = blockIdx.z, d = threadIdx.x;
    const float* pt = part + ((size_t) t * Hkv + g) * S * kG * kPart;
    if (d < S) {
        ms[d] = pt[((size_t) d * kG + h) * kPart + 128];
        ls[d] = pt[((size_t) d * kG + h) * kPart + 129];
    }
    __syncthreads();
    float M = -INFINITY;
    for (int z = 0; z < S; ++z) M = fmaxf(M, ms[z]);
    float L = 0.0f, a = 0.0f;
    if (M != -INFINITY) {
        float pv[kCombMaxS];
#pragma unroll
        for (int z = 0; z < kCombMaxS; ++z)
            if (z < S) pv[z] = pt[((size_t) z * kG + h) * kPart + d];
#pragma unroll
        for (int z = 0; z < kCombMaxS; ++z)
            if (z < S) {
                const float m = ms[z];
                const float f = m == -INFINITY ? 0.0f : expf(m - M);
                L = fmaf(ls[z], f, L);
                a = fmaf(pv[z], f, a);
            }
    }
    const float v = L > 0.0f ? a / L : 0.0f;
    float* orow = out + ((size_t) t * Hq + g * kG + h) * 128;
    if (!rot) {
        orow[d] = v;
        return;
    }
    o[d] = v;
    __syncthreads();
    if (d < 32) {   // (the values were rotated: the row back through H)
        float x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = o[d * 4 + i];
        had128w(x, d);
#pragma unroll
        for (int i = 0; i < 4; ++i) orow[d * 4 + i] = x[i];
    }
}

// ------------------------------------------------------------------ the router (one warp a token, E <= 256)
__global__ void route_kernel(const float* logits, const float* bias, int T, int E, int K, int* top, float* w) {
    const int t = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (t >= T) return;
    float sg[8], ch[8];
    const int per = (E + 31) / 32;
    for (int i = 0; i < per; ++i) {
        const int e = lane + 32 * i;
        sg[i] = e < E ? 1.0f / (1.0f + expf(-logits[(size_t) t * E + e])) : 0.0f;
        ch[i] = e < E ? sg[i] + bias[e] : -INFINITY;
    }
    float ps = INFINITY;
    int pi = -1;
    float sum = 0.0f;
    for (int r = 0; r < K; ++r) {
        float bv = -INFINITY, bsg = 0.0f;
        int bi = 0x7fffffff;
        for (int i = 0; i < per; ++i) {
            const int e = lane + 32 * i;
            if (e < E && better(ps, pi, ch[i], e) && better(ch[i], e, bv, bi)) {
                bv = ch[i];
                bi = e;
                bsg = sg[i];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const float os = __shfl_xor_sync(0xffffffffu, bsg, o);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
                bsg = os;
            }
        }
        if (lane == 0) {
            top[(size_t) t * K + r] = bi;
            w[(size_t) t * K + r] = bsg;
        }
        sum += bsg;
        ps = bv;
        pi = bi;
    }
    __syncwarp();
    if (lane < K) w[(size_t) t * K + lane] /= sum;
}

// ------------------------------------------------------------------ the routed experts' plan on the device
// (a layer whose experts are all in VRAM: mm3_model.cu's host plan value for value, no round trip through the host)
// one block, a thread an entry e = t * 8 + i, then thread 0 groups the entries by expert pointer in order of first
// appearance, each group's entries in entry order - the host's grouping (layer_window's by_ptr) exactly
constexpr int kPlanEnt = 16 * 8;
__global__ void __launch_bounds__(kPlanEnt) plan_dev_kernel(const int* __restrict__ top, const float* __restrict__ w,
                                                            int T, int K, const unsigned long long* __restrict__ tab,
                                                            float scale, unsigned long long* plan_ptr, float* plan_w,
                                                            unsigned* counts, unsigned long long* grp_ptr,
                                                            int* grp_start, int* grp_ent, int* n_groups) {
    __shared__ unsigned long long pp[kPlanEnt], gp[kPlanEnt];
    __shared__ int cnt[kPlanEnt], gof[kPlanEnt], fill[kPlanEnt + 1];
    const int n = T * 8;
    for (int e = threadIdx.x; e < n; e += blockDim.x) {
        const int t = e >> 3, i = e & 7;
        unsigned long long p = 0ull;
        float wt = 0.0f;
        if (i < K) {
            const int x = top[t * K + i];
            p = tab[x];
            wt = __fmul_rn(w[t * K + i], scale);
            if (counts != nullptr) atomicAdd(counts + x, 1u);
        }
        plan_ptr[e] = p;
        plan_w[e] = wt;
        pp[e] = p;
    }
    if (grp_ptr == nullptr) return;
    __syncthreads();
    if (threadIdx.x != 0) return;
    int ng = 0;
    for (int e = 0; e < n; ++e) {
        gof[e] = -1;
        const unsigned long long p = pp[e];
        if (p == 0ull) continue;
        int g = 0;
        while (g < ng && gp[g] != p) ++g;
        if (g == ng) {
            gp[ng] = p;
            cnt[ng++] = 0;
        }
        ++cnt[g];
        gof[e] = g;
    }
    fill[0] = 0;
    for (int g = 0; g < ng; ++g) {
        grp_ptr[g] = gp[g];
        grp_start[g] = fill[g];
        fill[g + 1] = fill[g] + cnt[g];
    }
    grp_start[ng] = fill[ng];
    for (int e = 0; e < n; ++e)
        if (gof[e] >= 0) grp_ent[fill[gof[e]]++] = e;
    *n_groups = ng;
}

__device__ __forceinline__ float swoai(float g, float u, float alpha, float limit) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    return g / (1.0f + expf(-alpha * g)) * (u + 1.0f);
}

__global__ void swigluoai_kernel(const float* g, const float* u, size_t n, float alpha, float limit, float* hf,
                                 uint16_t* h16) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float v = swoai(g[i], u[i], alpha, limit);
        if (hf) hf[i] = v;
        if (h16) h16[i] = f2h(v);
    }
}

__global__ void swigluoai_pairs_kernel(const float* gu, int ff, float alpha, float limit, uint16_t* h16) {
    const float* r = gu + (size_t) blockIdx.x * 2 * ff;
    for (int i = threadIdx.x; i < ff; i += blockDim.x)
        h16[(size_t) blockIdx.x * ff + i] = f2h(swoai(r[i], r[ff + i], alpha, limit));
}

__global__ void to_f16_kernel(const float* x, int cols, int ld, uint16_t* y, int ldo) {
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[(size_t) blockIdx.x * ldo + i] = f2h(x[(size_t) blockIdx.x * ld + i]);
}

__global__ void combine_kernel(const float* res, const float* y, const int* inv, const float* w, const float* sh, int K,
                               int D, float scale, float* out) {
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float a = 0.0f;
        for (int i = 0; i < K; ++i) a = fmaf(w[(size_t) t * K + i], y[(size_t) inv[t * K + i] * D + d], a);
        float m = a * scale;
        if (sh) m += sh[(size_t) t * D + d];
        out[(size_t) t * D + d] = (res ? res[(size_t) t * D + d] : 0.0f) + m;
    }
}

__global__ void add_kernel(float* x, const float* a, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        x[i] += a[i];
}

unsigned grid_for(size_t n) { return (unsigned) std::min<size_t>(4096, (n + 255) / 256); }

// one warp a weight row: lane l takes the row's values l, l + 32, l + 64 ... into one F32 sum in that order, then
// warp_sum - a token's row is the same bits whatever T and NC.  The row's weights read once for NC of the T rows (each
// its own sum), kRdU of them in flight (a load's latency a value left the GPU idle).  Q8: the row in ggml's Q8_0 blocks
// (34 bytes: an F16 scale, 32 int8 - lane l always byte l of a block), value = scale * q: exactly the F32 an upload
// dequantizes, read at a quarter of the bytes.  Rows n >= Na: the second matrix's (Wb, Nb, Yb - one launch for two).
constexpr int kRdU = 16;
template <bool Q8>
__device__ __forceinline__ float rd_w(const void* W, int r, int K, int i) {
    if constexpr (Q8) {
        const uint8_t* b = (const uint8_t*) W + ((size_t) r * (K / 32) + (size_t) (i >> 5)) * 34;
        const float d = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))));
        return d * (float) (int8_t) b[2 + (i & 31)];
    } else {
        return ((const float*) W)[(size_t) r * K + i];
    }
}
template <int NC, bool Q8>
__global__ void __launch_bounds__(256) rows_dot_kernel(const void* Wa, int Na, const void* Wb, int Nb, int K,
                                                       const float* X, int T, float* Ya, float* Yb) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (n >= Na + Nb) return;
    const bool two = n >= Na;
    const int r = two ? n - Na : n, N = two ? Nb : Na;
    const void* W = two ? Wb : Wa;
    float* Y = two ? Yb : Ya;
    const int t0 = blockIdx.y * NC, nc = min(NC, T - t0);
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    int i0 = lane;
    for (; i0 + (kRdU - 1) * 32 < K; i0 += kRdU * 32) {
        float a[kRdU];
#pragma unroll
        for (int u = 0; u < kRdU; ++u) a[u] = rd_w<Q8>(W, r, K, i0 + u * 32);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < nc) {
                const float* x = X + (size_t) (t0 + c) * K + i0;
#pragma unroll
                for (int u = 0; u < kRdU; ++u) acc[c] = fmaf(a[u], x[u * 32], acc[c]);
            }
    }
    for (int i = i0; i < K; i += 32) {
        const float a = rd_w<Q8>(W, r, K, i);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < nc) acc[c] = fmaf(a, X[(size_t) (t0 + c) * K + i], acc[c]);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < nc) {
            const float v = warp_sum(acc[c]);
            if (lane == 0) Y[(size_t) (t0 + c) * N + r] = v;
        }
}

__global__ void q8_0_f32_kernel(const uint8_t* W, size_t n, float* out) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint8_t* b = W + i / 32 * 34;
        out[i] = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8)))) * (float) (int8_t) b[2 + i % 32];
    }
}

// ------------------------------------------------------------------ the DSpark draft's
__global__ void mxfp8_kernel(const uint8_t* w, const uint8_t* sc, int K, uint16_t* out) {
    const size_t n = blockIdx.y;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < K; k += gridDim.x * blockDim.x) {
        const uint8_t e = sc[n * (size_t) (K / 32) + (size_t) (k / 32)];
#if !defined(STRATA_USE_HIP)
        __nv_fp8_e4m3 f;
        f.__x = w[n * (size_t) K + k];
        const float v = ldexpf((float) f, (int) e - 127);   // (exact: E4M3's 3 mantissa bits fit BF16's 7)
#else
        const float v = ldexpf(strata::ds4::e4m3_value(w[n * (size_t) K + k]), (int) e - 127);
#endif
        out[n * (size_t) K + k] = f2bf(v);
    }
}

// a block 32 outputs (8 warps x 4) over a range of the inputs (split-K: blockIdx.y), the activations 512 inputs at a
// time in shared memory; a lane 4 weights of each output at a 128-byte stride (a warp's loads one line, its float4
// reads of the activations free of bank conflicts) - a tile's 16 weight words and their scales loaded before any
// use; then each output's sums over the lanes (a fixed tree) into y (one range) or the range's partials.  R: the rows.
template <int R>
__global__ void __launch_bounds__(256) mx_rows_kernel(const uint8_t* w, const uint8_t* sc, int N, int K, int kc,
                                                      const float* x, int ldx, float* y, int ldy, float beta,
                                                      float* part) {
    __shared__ float tab[256];
    __shared__ __align__(16) float xs[R][512];
    {
#if !defined(STRATA_USE_HIP)
        __nv_fp8_e4m3 f;
        f.__x = (uint8_t) threadIdx.x;
        tab[threadIdx.x] = (float) f;
#else
        tab[threadIdx.x] = strata::ds4::e4m3_value((uint8_t) threadIdx.x);
#endif
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n0 = blockIdx.x * 32 + warp * 4;
    const int ka = blockIdx.y * kc, kb = min(K, ka + kc);
    const size_t ks = (size_t) (K / 32);
    float acc[4][R];
#pragma unroll
    for (int o = 0; o < 4; ++o)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[o][r] = 0.0f;
    for (int k0 = ka; k0 < kb; k0 += 512) {
        uint32_t wb[4][4];
        uint32_t se[4][4];
#pragma unroll
        for (int o = 0; o < 4; ++o)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const size_t n = (size_t) (n0 + o);
                const int kk = k0 + j * 128 + lane * 4;
                wb[o][j] = __ldg((const unsigned int*) (w + n * K + kk));
                se[o][j] = __ldg(sc + n * ks + (size_t) (kk / 32));
            }
        __syncthreads();
        for (int i = threadIdx.x; i < R * 128; i += 256) {
            const int r = i >> 7, c = (i & 127) * 4;
            *(float4*) &xs[r][c] = *(const float4*) (x + (size_t) r * ldx + k0 + c);
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float4 xv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) xv[r] = *(const float4*) &xs[r][j * 128 + lane * 4];
#pragma unroll
            for (int o = 0; o < 4; ++o) {
                const uint32_t b = wb[o][j];
                // the UE8M0 scale 2^(e - 127) as a float's exponent (e 1 .. 254; 0 the subnormal 2^-127)
                const uint32_t e = se[o][j];
                const float scale = e ? __uint_as_float(e << 23) : __uint_as_float(0x00400000u);
                const float w0 = tab[b & 0xffu] * scale, w1 = tab[(b >> 8) & 0xffu] * scale,
                            w2 = tab[(b >> 16) & 0xffu] * scale, w3 = tab[b >> 24] * scale;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float a = acc[o][r];
                    a = fmaf(w0, xv[r].x, a);
                    a = fmaf(w1, xv[r].y, a);
                    a = fmaf(w2, xv[r].z, a);
                    a = fmaf(w3, xv[r].w, a);
                    acc[o][r] = a;
                }
            }
        }
    }
#pragma unroll
    for (int o = 0; o < 4; ++o)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[o][r];
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
            if (lane == 0) {
                if (part != nullptr) {
                    part[((size_t) blockIdx.y * R + r) * N + n0 + o] = v;
                } else {
                    float* yp = y + (size_t) r * ldy + n0 + o;
                    *yp = beta != 0.0f ? beta * *yp + v : v;
                }
            }
        }
}

// y = beta y + the ranges' partials, in range order
__global__ void mx_reduce_kernel(const float* part, int ks, int rows, int N, float* y, int ldy, float beta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * N) return;
    const int r = i / N, n = i % N;
    float v = 0.0f;
    for (int z = 0; z < ks; ++z) v += part[((size_t) z * rows + r) * N + n];
    float* yp = y + (size_t) r * ldy + n;
    *yp = beta != 0.0f ? beta * *yp + v : v;
}

__global__ void to_bf16_kernel(const float* x, int cols, int ld, uint16_t* y, int ldo) {
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[(size_t) blockIdx.x * ldo + i] = f2bf(x[(size_t) blockIdx.x * ld + i]);
}

// window attention: sparse_attn_kernel's tiles over a range of key positions instead of kept blocks, not causal
__global__ void __launch_bounds__(128) window_attn_kernel(const float* q, const uint16_t* kc, const uint16_t* vc,
                                                          int ring, const int* lo, int hi, int Hq, int Hkv,
                                                          float scale, float* out) {
    __shared__ float qs[kG][128];
    __shared__ __align__(4) uint16_t ks[kTile][kRow], vs[kTile][kRow];
    __shared__ float sc[kG][kTile + 1];
    __shared__ float mrow[kG], lrow[kG], arow[kG];
    const int t = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const int k_lo = lo[t];
    for (int i = tid; i < kG * 128; i += 128) {
        const int h = i / 128, d = i % 128;
        qs[h][d] = q[((size_t) t * Hq + g * kG + h) * 128 + d] * scale;
    }
    if (tid < kG) {
        mrow[tid] = -INFINITY;
        lrow[tid] = 0.0f;
    }
    float acc[kG];
#pragma unroll
    for (int h = 0; h < kG; ++h) acc[h] = 0.0f;
    for (int k0 = k_lo; k0 <= hi; k0 += kTile) {
        __syncthreads();
        for (int i = tid; i < kTile * 64; i += 128) {
            const int j = i >> 6, c = i & 63;
            const int kp = k0 + j;
            uint32_t kw = 0, vw = 0;
            if (kp <= hi) {
                const size_t row = (size_t) (kp % ring);
                kw = *(const uint32_t*) (kc + (row * Hkv + g) * 128 + c * 2);
                vw = *(const uint32_t*) (vc + (row * Hkv + g) * 128 + c * 2);
            }
            *(uint32_t*) &ks[j][c * 2] = kw;
            *(uint32_t*) &vs[j][c * 2] = vw;
        }
        __syncthreads();
        {
            const int j = tid & 63, h0 = (tid >> 6) * 8;
            const bool ok = k0 + j <= hi;
            float s8[8];
#pragma unroll
            for (int h = 0; h < 8; ++h) s8[h] = 0.0f;
            if (ok) {
#pragma unroll 4
                for (int d = 0; d < 128; ++d) {
                    const float kv = bf2f(ks[j][d]);
#pragma unroll
                    for (int h = 0; h < 8; ++h) s8[h] = fmaf(qs[h0 + h][d], kv, s8[h]);
                }
            }
#pragma unroll
            for (int h = 0; h < 8; ++h) sc[h0 + h][j] = ok ? s8[h] : -INFINITY;
        }
        __syncthreads();
        if (tid < kG) {
            const int h = tid;
            float m = mrow[h];
            for (int j = 0; j < kTile; ++j) m = fmaxf(m, sc[h][j]);
            const float a = mrow[h] == -INFINITY ? 0.0f : expf(mrow[h] - m);
            float l = lrow[h] * a;
            for (int j = 0; j < kTile; ++j) {
                const float e = sc[h][j] == -INFINITY ? 0.0f : expf(sc[h][j] - m);
                sc[h][j] = e;
                l += e;
            }
            mrow[h] = m;
            lrow[h] = l;
            arow[h] = a;
        }
        __syncthreads();
        {
            const int d = tid;
#pragma unroll
            for (int h = 0; h < kG; ++h) {
                float s = acc[h] * arow[h];
                for (int j = 0; j < kTile; ++j) s = fmaf(sc[h][j], bf2f(vs[j][d]), s);
                acc[h] = s;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < kG; ++h)
        out[((size_t) t * Hq + g * kG + h) * 128 + tid] = lrow[h] > 0.0f ? acc[h] / lrow[h] : 0.0f;
}

__global__ void gather_row_kernel(const uint16_t* table, int R, const int* prev, uint16_t* emb, float* embf) {
    const uint16_t* r = table + (size_t) *prev * R;
    for (int i = threadIdx.x; i < R; i += blockDim.x) {
        emb[i] = r[i];
        embf[i] = bf2f(r[i]);
    }
}

// one block of 1024: the biased logits' argmax (ties to the lower id), then the confidence
__global__ void __launch_bounds__(1024) markov_pick_kernel(const float* logits, const float* bias, int V, int* prev,
                                                           int* out, float* biased, const float* w, const float* b,
                                                           const float* hidden, int D, const float* embf, int R,
                                                           float* conf) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ float red[33];
    float best = -INFINITY;
    int bid = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = logits[i] + bias[i];
        if (biased) biased[i] = v;
        if (v > best || (v == best && i < bid)) {
            best = v;
            bid = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bid, o);
        if (ov > best || (ov == best && oi < bid)) {
            best = ov;
            bid = oi;
        }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) {
        bv[wid] = best;
        bi[wid] = bid;
    }
    __syncthreads();
    if (wid == 0) {
        best = lane < (int) (blockDim.x >> 5) ? bv[lane] : -INFINITY;
        bid = lane < (int) (blockDim.x >> 5) ? bi[lane] : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bid, o);
            if (ov > best || (ov == best && oi < bid)) {
                best = ov;
                bid = oi;
            }
        }
    }
    // (every thread has read *prev's row before: gather_row ran earlier on the stream)
    if (conf != nullptr) {
        float acc = 0.0f;
        for (int i = threadIdx.x; i < D + R; i += blockDim.x) acc = fmaf(w[i], i < D ? hidden[i] : embf[i - D], acc);
        const float s = block_sum(acc, red);
        if (threadIdx.x == 0) *conf = 1.0f / (1.0f + expf(-(s + *b)));
    }
    if (threadIdx.x == 0) {
        *prev = bid;
        *out = bid;
    }
}

// ------------------------------------------------------------------ candidates: a row's largest values
// a float's orderable key (a larger float, a larger key; -0 as +0)
__device__ __forceinline__ uint32_t fkey(float f) {
    uint32_t u = __float_as_uint(f);
    if (u == 0x80000000u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct SelScratch {           // (shared memory: kth_key's histograms, the gathered keys)
    unsigned wh[32][256];     // a histogram a warp
    unsigned hist[256];
    uint32_t sh[2];
    uint32_t skey[kCandCap];
    int sid[kCandCap];
    int cnt;
};

// the k-th largest of val(0 .. V - 1)'s keys (blockDim.x 1024; every thread calls it, all get it): four rounds of an
// 8-bit histogram over the keys whose digits so far match, each warp's counts aggregated by bin into its own
// histogram, then summed
template <class F>
__device__ uint32_t kth_key(const F& val, int V, int k, SelScratch& S) {
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nw = blockDim.x >> 5;
    uint32_t prefix = 0, mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) S.wh[w][i] = 0;
        __syncwarp();
        for (int i0 = 0; i0 < V; i0 += blockDim.x) {
            const int i = i0 + tid;
            bool in = false;
            uint32_t bin = 0;
            if (i < V) {
                const uint32_t key = fkey(val(i));
                in = (key & mask) == prefix;
                bin = (key >> shift) & 255u;
            }
            const unsigned act = __ballot_sync(0xffffffffu, in);
#if defined(STRATA_USE_HIP)
            // (HIP has no 32-lane-mask __match_any_sync: the same peer masks from one ballot per distinct bin, the
            //  whole warp in the loop - act and rest are the warp's own, so every lane takes the same turns)
            unsigned peers = 0;
            for (unsigned rest = act; rest != 0u;) {
                const uint32_t sv = __shfl_sync(0xffffffffu, bin, __ffs(rest) - 1);
                const bool same = in && bin == sv;
                const unsigned m = __ballot_sync(0xffffffffu, same);
                if (same) peers = m;
                rest &= ~m;
            }
            if (in && __ffs(peers) - 1 == lane) S.wh[w][bin] += __popc(peers);
#else
            if (in) {
                const unsigned peers = __match_any_sync(act, bin);
                if (__ffs(peers) - 1 == lane) S.wh[w][bin] += __popc(peers);
            }
#endif
            __syncwarp();
        }
        __syncthreads();
        for (int b = tid; b < 256; b += blockDim.x) {
            unsigned c = 0;
            for (int j = 0; j < nw; ++j) c += S.wh[j][b];
            S.hist[b] = c;
        }
        __syncthreads();
        if (tid == 0) {
            unsigned cum = 0;
            int b = 255;
            for (; b > 0; --b) {
                if (cum + S.hist[b] >= (unsigned) k) break;
                cum += S.hist[b];
            }
            S.sh[0] = prefix | ((uint32_t) b << shift);
            S.sh[1] = (uint32_t) k - cum;
        }
        __syncthreads();
        prefix = S.sh[0];
        k = (int) S.sh[1];
        mask |= 255u << shift;
        __syncthreads();
    }
    return prefix;
}

// every i with a key >= thr into S.skey / S.sid (at most kCandCap; S.cnt the count - more: more ties than room, the
// first kCandCap stored), then those sorted by key desc, id asc (bitonic over kCandCap, the rest padded last); returns
// the count
template <class F>
__device__ int gather_sorted(const F& val, int V, uint32_t thr, SelScratch& S) {
    if (threadIdx.x == 0) S.cnt = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const uint32_t key = fkey(val(i));
        if (key >= thr) {
            const int at = atomicAdd(&S.cnt, 1);
            if (at < kCandCap) {
                S.skey[at] = key;
                S.sid[at] = i;
            }
        }
    }
    __syncthreads();
    const int n = S.cnt;
    for (int i = min(n, kCandCap) + threadIdx.x; i < kCandCap; i += blockDim.x) {
        S.skey[i] = 0u;
        S.sid[i] = 0x7fffffff;
    }
    for (int size = 2; size <= kCandCap; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < kCandCap; i += blockDim.x) {
                const int j = i ^ stride;
                if (j <= i) continue;
                const uint32_t ka = S.skey[i], kb = S.skey[j];
                const int ia = S.sid[i], ib = S.sid[j];
                const bool a_first = ka > kb || (ka == kb && ia < ib);
                if (((i & size) == 0) != a_first) {
                    S.skey[i] = kb;
                    S.skey[j] = ka;
                    S.sid[i] = ib;
                    S.sid[j] = ia;
                }
            }
        }
    __syncthreads();
    return n;
}

// a row a block: its kc largest logits and every one equal to the kc-th, sorted, and the extra ids' logits
__global__ void __launch_bounds__(1024) topk_rows_kernel(const float* lg, int V, int kc, const int* extra, int n_extra,
                                                         CandRow* out) {
    __shared__ SelScratch S;
    const float* x = lg + (size_t) blockIdx.x * V;
    CandRow* o = out + blockIdx.x;
    const auto val = [x](int i) { return x[i]; };
    const uint32_t thr = kth_key(val, V, max(1, min(kc, V)), S);
    const int n = gather_sorted(val, V, thr, S);
    for (int j = threadIdx.x; j < n_extra; j += blockDim.x) o->ex[j] = x[extra[j]];
    if (n > kCandCap) {
        if (threadIdx.x == 0) o->n = -1;
        return;
    }
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        o->id[j] = S.sid[j];
        o->v[j] = x[S.sid[j]];
    }
    if (threadIdx.x == 0) o->n = n;
}

// a sampled Markov step: the biased logits' kq largest (sorted as topk_rows sorts them) as main.cpp's dist() takes
// a row - / temperature, softmax, then min_p and top_p cut - the draft's distribution q; the token drawn from it with
// u, q into *qo; then the confidence as markov_pick's
__global__ void __launch_bounds__(1024) markov_sample_kernel(const float* logits, const float* bias, int V, int* prev,
                                                             int* out, const float* w, const float* b,
                                                             const float* hidden, int D, const float* embf, int R,
                                                             float* conf, SampleArgs sa, QRow* qo) {
    __shared__ SelScratch S;
    __shared__ float red[33];
    __shared__ int pick;
    const auto val = [logits, bias](int i) { return logits[i] + bias[i]; };
    const int kq = max(1, min(sa.top_k, kQMax));
    const uint32_t thr = kth_key(val, V, kq, S);
    const int n = min(gather_sorted(val, V, thr, S), kq);   // (more ties than room: the kq largest still lead)
    if (threadIdx.x == 0) {
        double p[kQMax], sum = 0;
        const float v0 = val(S.sid[0]) / sa.temperature;
        for (int i = 0; i < n; ++i) sum += (p[i] = exp((double) (val(S.sid[i]) / sa.temperature - v0)));
        int keep = n;
        double cum = 0;
        for (int i = 0; i < n; ++i) {
            cum += p[i] / sum;
            if (sa.min_p > 0.0f && p[i] < sa.min_p * p[0]) {
                keep = max(1, i);
                break;
            }
            if (sa.top_p < 1.0f && cum >= sa.top_p) {
                keep = i + 1;
                break;
            }
        }
        double s2 = 0;
        for (int i = 0; i < keep; ++i) s2 += p[i];
        int tok = S.sid[keep - 1];
        double c2 = 0;
        bool got = false;
        for (int i = 0; i < keep; ++i) {
            const float q = (float) (p[i] / s2);
            qo->id[i] = S.sid[i];
            qo->q[i] = q;
            c2 += q;
            if (!got && (double) sa.u < c2) {
                tok = S.sid[i];
                got = true;
            }
        }
        qo->n = keep;
        pick = tok;
    }
    __syncthreads();
    if (conf != nullptr) {   // (every thread has read *prev's row before: gather_row ran earlier on the stream)
        float acc = 0.0f;
        for (int i = threadIdx.x; i < D + R; i += blockDim.x) acc = fmaf(w[i], i < D ? hidden[i] : embf[i - D], acc);
        const float s = block_sum(acc, red);
        if (threadIdx.x == 0) *conf = 1.0f / (1.0f + expf(-(s + *b)));
    }
    if (threadIdx.x == 0) {
        *prev = pick;
        *out = pick;
    }
}

}  // namespace

void topk_rows(const float* lg, int rows, int V, int kc, const int* extra, int n_extra, CandRow* out, cudaStream_t s) {
    if (rows <= 0) return;
    topk_rows_kernel<<<rows, 1024, 0, s>>>(lg, V, kc, extra, std::min(n_extra, kCandExtra), out);
    check("topk_rows");
}

void markov_sample(const float* logits, const float* bias, int V, int* prev, int* out, const float* w, const float* b,
                   const float* hidden, int D, const float* embf, int R, float* conf, const SampleArgs& sa, QRow* qo,
                   cudaStream_t s) {
    markov_sample_kernel<<<1, 1024, 0, s>>>(logits, bias, V, prev, out, w, b, hidden, D, embf, R, conf, sa, qo);
    check("markov_sample");
}

void rms(const float* x, const float* w1, int rows, int D, float eps, float* y, uint16_t* y16, cudaStream_t s) {
    if (rows <= 0) return;
    rms_kernel<<<rows, 256, 0, s>>>(x, w1, D, eps, y, y16);
    check("rms");
}

void rope_inv_freq(float theta, int n_rot, float* inv) {
    for (int i = 0; i < n_rot / 2; ++i)
        inv[i] = (float) (1.0 / std::pow((double) theta, (double) (float) (2 * i) / (double) n_rot));
}

void qk_norm_rope(float* q, const float* k, const float* v, const float* qn1, const float* kn1, const int* pos, int T,
                  int Hq, int Hkv, int hd, int n_rot, const float* inv, float eps, uint16_t* kc, uint16_t* vc,
                  cudaStream_t s, int ring, const KvQ& kq, const KvHost& hs) {
    if (T <= 0) return;
    qk_kernel<<<dim3(T, Hq + 2 * Hkv), hd, 0, s>>>(q, k, v, qn1, kn1, pos, Hq, Hkv, n_rot, inv, eps, kc, vc, ring, kq,
                                                   hs);
    check("qk_norm_rope");
}
void index_norm_rope(float* iq, const float* ik, const float* qn1, const float* kn1, const int* pos, int T, int Hi,
                     int D, int n_rot, const float* inv, float eps, uint16_t* ic, cudaStream_t s) {
    if (T <= 0) return;
    index_kernel<<<dim3(T, Hi + 1), D, 0, s>>>(iq, ik, qn1, kn1, pos, Hi, n_rot, inv, eps, ic);
    check("index_norm_rope");
}

// (MM3_MSA_OLD=1: the spans and the staging off - the kernels as before, for timing them; the same bits)
const bool msa_old = std::getenv("MM3_MSA_OLD") != nullptr;

void block_scores(const float* iq, const uint16_t* ic, const int* pos, int T, int Hi, int nblk, int blk, float* bs,
                  cudaStream_t s) {
    if (T <= 0 || nblk <= 0 || kBsV % Hi) return;
    // a prompt's (MAYA_MM3_MSA_WIDE=0: the token kernel for them too - the same bits either way)
    static const bool wide_on = [] {
        const char* v = std::getenv("MAYA_MM3_MSA_WIDE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (wide_on && !msa_old && T >= 2 * (kBwV / Hi) && blk % kBwK == 0) {
        const int tiles = (T + kBwV / Hi - 1) / (kBwV / Hi);
        int span = std::max(1, (int) (((long) nblk * tiles + 511) / 512));
        span = std::min(span, nblk);
        block_scores_wide_kernel<<<dim3((unsigned) tiles, (unsigned) ((nblk + span - 1) / span)), 256, 0, s>>>(
            iq, ic, pos, T, Hi, nblk, blk, span, bs);
        check("block_scores (wide)");
        return;
    }
    // (the spans: ~512 blocks over the tiles and spans, 2 key blocks a span at least - the bits are the same for any)
    const int tiles = (T + kBsV / Hi - 1) / (kBsV / Hi);
    int span = msa_old ? nblk : std::max(2, (int) (((long) nblk * tiles + 511) / 512));
    span = std::min(span, nblk);
    block_scores_kernel<<<dim3((unsigned) tiles, (unsigned) ((nblk + span - 1) / span)), 256, 0, s>>>(
        iq, ic, pos, T, Hi, nblk, blk, span, bs);
    check("block_scores");
}

void select_blocks(const float* bs, const int* pos, int T, int Hi, int nblk, int blk, int topk, int* kept,
                   cudaStream_t s) {
    if (T <= 0) return;
    const int warps = T * Hi;
    const size_t smem = (size_t) kSelWarps * nblk * sizeof(float);
    const bool stage = !msa_old && smem <= 48 * 1024;
    select_kernel<<<(warps + kSelWarps - 1) / kSelWarps, 32 * kSelWarps, stage ? smem : 0, s>>>(
        bs, pos, T, Hi, nblk, blk, topk, kept, stage);
    check("select_blocks");
}

void sparse_attn(const float* q, const uint16_t* kc, const uint16_t* vc, const int* kept, const int* pos, int T,
                 int Hq, int Hkv, int topk, int blk, float scale, float* out, cudaStream_t s, const KvQ& kq,
                 const KvHost& hs, int splits, float* part) {
    if (T <= 0) return;
    const int S = part != nullptr && splits > 1 ? std::min({splits, topk, kCombMaxS}) : 1;
    const dim3 grid((unsigned) T, (unsigned) Hkv, (unsigned) S);
    // (a decode token's or window's few queries: the wide kernel - the same bits; MAYA_MM3_ATTN_WIDE=0: the other)
    static const bool wide_on = [] {
        const char* v = std::getenv("MAYA_MM3_ATTN_WIDE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool wide = wide_on && T <= kWideT;
#define MM3_ATTN(KK, VV)                                                                                             \
    case KK * 4 + VV:                                                                                                \
        if (wide)                                                                                                    \
            sparse_attn_w_kernel<KK, VV><<<grid, kWideNT, 0, s>>>(q, kc, vc, kept, pos, T, Hq, Hkv, topk, blk, scale, \
                                                                  out, kq, hs, S, part);                             \
        else                                                                                                         \
            sparse_attn_kernel<KK, VV><<<grid, 128, 0, s>>>(q, kc, vc, kept, pos, T, Hq, Hkv, topk, blk, scale, out,  \
                                                            kq, hs, S, part);                                         \
        break;
    switch ((kq.kf & 3) * 4 + (kq.vf & 3)) {
        MM3_ATTN(0, 0) MM3_ATTN(0, 1) MM3_ATTN(0, 2) MM3_ATTN(0, 3)
        MM3_ATTN(1, 0) MM3_ATTN(1, 1) MM3_ATTN(1, 2) MM3_ATTN(1, 3)
        MM3_ATTN(2, 0) MM3_ATTN(2, 1) MM3_ATTN(2, 2) MM3_ATTN(2, 3)
        MM3_ATTN(3, 0) MM3_ATTN(3, 1) MM3_ATTN(3, 2) MM3_ATTN(3, 3)
    }
#undef MM3_ATTN
    if (S > 1)
        attn_combine_kernel<<<dim3((unsigned) T, (unsigned) Hkv, (unsigned) kG), 128, 0, s>>>(part, Hq, Hkv, S,
                                                                             kq.vf != kKvBf16 && (kq.rot & 2), out);
    check("sparse_attn");
}

void route(const float* logits, const float* bias, int T, int E, int K, int* top, float* w, cudaStream_t s) {
    if (T <= 0) return;
    route_kernel<<<(T + 7) / 8, 256, 0, s>>>(logits, bias, T, E, K, top, w);
    check("route");
}

bool plan_dev(const int* top, const float* w, int T, int K, const unsigned long long* tab, float scale,
              unsigned long long* plan_ptr, float* plan_w, unsigned* counts, unsigned long long* grp_ptr, int* grp_start,
              int* grp_ent, int* n_groups, cudaStream_t s) {
    if (T < 1 || T * 8 > kPlanEnt || K < 1 || K > 8) return false;
    plan_dev_kernel<<<1, kPlanEnt, 0, s>>>(top, w, T, K, tab, scale, plan_ptr, plan_w, counts, grp_ptr, grp_start,
                                           grp_ent, n_groups);
    check("plan_dev");
    return true;
}

void submit(cudaStream_t s) {
#if defined(_WIN32)
    static const bool on = [] {
        const char* v = std::getenv("MAYA_MM3_WIN_SUBMIT");
        if (v == nullptr) v = std::getenv("MM3_WIN_SUBMIT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (on) {
        (void) cudaStreamQuery(s);
        (void) cudaGetLastError();
    }
#else
    (void) s;
#endif
}

void swigluoai(const float* g, const float* u, size_t n, float alpha, float limit, float* hf, uint16_t* h16,
               cudaStream_t s) {
    if (n == 0) return;
    swigluoai_kernel<<<grid_for(n), 256, 0, s>>>(g, u, n, alpha, limit, hf, h16);
    check("swigluoai");
}

void swigluoai_pairs(const float* gu, int rows, int ff, float alpha, float limit, uint16_t* h16, cudaStream_t s) {
    if (rows <= 0) return;
    swigluoai_pairs_kernel<<<rows, 256, 0, s>>>(gu, ff, alpha, limit, h16);
    check("swigluoai_pairs");
}

void to_f16(const float* x, int rows, int cols, int ld, uint16_t* y, int ldo, cudaStream_t s) {
    if (rows <= 0) return;
    to_f16_kernel<<<rows, 256, 0, s>>>(x, cols, ld, y, ldo);
    check("to_f16");
}

void moe_combine(const float* res, const float* y, const int* inv, const float* w, const float* sh, int T, int K,
                 int D, float scale, float* out, cudaStream_t s) {
    if (T <= 0) return;
    combine_kernel<<<T, 256, 0, s>>>(res, y, inv, w, sh, K, D, scale, out);
    check("moe_combine");
}

__global__ void wait_flag_kernel(const unsigned* flag, unsigned v) {
    if (threadIdx.x != 0) return;
    const unsigned long long t0 = gtimer();
    while ((int) (*(volatile const unsigned*) flag - v) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
        if (gtimer() - t0 > kWaitLimitNs) {   // (never a hung device: the host's answer did not come)
            atomicAdd(&g_wait_timeouts, 1u);
            break;
        }
    }
    __threadfence_system();
}

unsigned wait_timeouts() {
    unsigned n = 0;
    if (cudaMemcpyFromSymbol(&n, g_wait_timeouts, sizeof n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
}

void wait_flag(const unsigned* flag, unsigned v, cudaStream_t s) {
    wait_flag_kernel<<<1, 32, 0, s>>>(flag, v);
    check("wait_flag");
}

__global__ void pull_kernel(float4* dst, const float4* src, size_t n4) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// one block: thread 0 spins on the host's counter, then the block reads the rows (four 16-byte loads in flight a thread)
__global__ void __launch_bounds__(1024) wait_pull_kernel(const unsigned* flag, unsigned v, float4* dst,
                                                         const float4* src, size_t n4) {
    if (threadIdx.x == 0) {
        while ((int) (*(volatile const unsigned*) flag - v) < 0) __nanosleep(100);
        __threadfence_system();
    }
    __syncthreads();
    const size_t st = blockDim.x;
    size_t i = threadIdx.x;
    for (; i + 3 * st < n4; i += 4 * st) {
        const float4 a = src[i], b = src[i + st], c = src[i + 2 * st], d = src[i + 3 * st];
        dst[i] = a;
        dst[i + st] = b;
        dst[i + 2 * st] = c;
        dst[i + 3 * st] = d;
    }
    for (; i < n4; i += st) dst[i] = src[i];
}

__global__ void iota_kernel(int* p, int n, int base) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = base + i;
}

__global__ void set_int_kernel(int* p, int v) { *p = v; }

void set_int(int* p, int v, cudaStream_t s) {
    set_int_kernel<<<1, 1, 0, s>>>(p, v);
    check("set_int");
}

// the draft's block: row 0 the anchor's BF16 row (mapped host memory), rows 1 .. B - 1 the mask token's (device);
// pos[t] = P + t, lo[t] = max(0, P + t - window + 1)
__global__ void draft_block_kernel(const uint16_t* anchor, const float* mask, int D, int B, int P, int window, float* h,
                                   int* pos, int* lo) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) h[(size_t) t * D + i] = t == 0 ? bf2f(anchor[i]) : mask[i];
    if (threadIdx.x == 0) {
        pos[t] = P + t;
        lo[t] = max(0, P + t - window + 1);
    }
}

void draft_block(const uint16_t* anchor, const float* mask, int D, int B, int P, int window, float* h, int* pos, int* lo,
                 cudaStream_t s) {
    draft_block_kernel<<<B, 256, 0, s>>>(anchor, mask, D, B, P, window, h, pos, lo);
    check("draft_block");
}

void iota(int* p, int n, int base, cudaStream_t s) {
    if (n <= 0) return;
    iota_kernel<<<(n + 255) / 256, 256, 0, s>>>(p, n, base);
    check("iota");
}

void wait_pull(const unsigned* flag, unsigned v, float* dst, const float* src, size_t n, cudaStream_t s) {
    wait_pull_kernel<<<1, 1024, 0, s>>>(flag, v, (float4*) dst, (const float4*) src, n / 4);
    check("wait_pull");
}

void pull(float* dst, const float* src, size_t n, cudaStream_t s) {
    const size_t n4 = n / 4;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(48, (n4 + 255) / 256));
    pull_kernel<<<blocks, 256, 0, s>>>((float4*) dst, (const float4*) src, n4);
    check("pull");
}

__global__ void publish_kernel(const float4* src, float4* dst, size_t n4, unsigned* counter, unsigned* flag,
                               unsigned v) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
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

void publish(const float* src, float* dst, size_t n, unsigned* counter, unsigned* flag, unsigned v, cudaStream_t s) {
    const size_t n4 = n / 4;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(48, (n4 + 255) / 256));
    publish_kernel<<<blocks, 256, 0, s>>>((const float4*) src, (float4*) dst, n4, counter, flag, v);
    check("publish");
}

void add(float* x, const float* a, size_t n, cudaStream_t s) {
    if (n == 0) return;
    add_kernel<<<grid_for(n), 256, 0, s>>>(x, a, n);
    check("add");
}

namespace {
template <bool Q8>
void rows_dot_launch(const void* Wa, int Na, const void* Wb, int Nb, int K, const float* X, int T, float* Ya, float* Yb,
                     cudaStream_t s) {
    if (T <= 0 || Na + Nb <= 0) return;
    const unsigned bx = (unsigned) ((Na + Nb + 7) / 8);
    if (T == 1) rows_dot_kernel<1, Q8><<<dim3(bx, 1), 256, 0, s>>>(Wa, Na, Wb, Nb, K, X, T, Ya, Yb);
    else if (T <= 4) rows_dot_kernel<4, Q8><<<dim3(bx, 1), 256, 0, s>>>(Wa, Na, Wb, Nb, K, X, T, Ya, Yb);
    else if (T <= 8) rows_dot_kernel<8, Q8><<<dim3(bx, 1), 256, 0, s>>>(Wa, Na, Wb, Nb, K, X, T, Ya, Yb);
    else rows_dot_kernel<8, Q8><<<dim3(bx, (unsigned) ((T + 7) / 8)), 256, 0, s>>>(Wa, Na, Wb, Nb, K, X, T, Ya, Yb);
}
}  // namespace

void rows_dot(const float* W, int N, int K, const float* X, int T, float* Y, cudaStream_t s) {
    rows_dot_launch<false>(W, N, nullptr, 0, K, X, T, Y, nullptr, s);
    check("rows_dot");
}
void rows_dot_q8(const uint8_t* Wa, int Na, const uint8_t* Wb, int Nb, int K, const float* X, int T, float* Ya,
                 float* Yb, cudaStream_t s) {
    if (K % 32 != 0) return;
    rows_dot_launch<true>(Wa, Na, Wb, Nb, K, X, T, Ya, Yb, s);
    check("rows_dot_q8");
}
void q8_0_f32(const uint8_t* W, size_t n, float* out, cudaStream_t s) {
    if (n == 0) return;
    q8_0_f32_kernel<<<grid_for(n), 256, 0, s>>>(W, n, out);
    check("q8_0_f32");
}

void mx_rows(const uint8_t* w, const uint8_t* scale, int N, int K, const float* x, int rows, int ldx, float* y, int ldy,
             float beta, float* part, cudaStream_t s) {
    if (rows <= 0) return;
    // split K while the blocks are fewer than ~2 waves of an 82-SM GPU (each range a whole number of 512s)
    const int tiles = K / 512, nb = N / 32;
    int ks = 1;
    while (part != nullptr && ks * 2 <= tiles && nb * ks < 160 && tiles % (ks * 2) == 0) ks *= 2;
    const int kc = K / ks;
    float* pp = ks > 1 ? part : nullptr;
    const dim3 grid((unsigned) nb, (unsigned) ks);
    switch (rows) {
#define MX_CASE(R) case R: mx_rows_kernel<R><<<grid, 256, 0, s>>>(w, scale, N, K, kc, x, ldx, y, ldy, beta, pp); break;
        MX_CASE(1) MX_CASE(2) MX_CASE(3) MX_CASE(4) MX_CASE(5) MX_CASE(6) MX_CASE(7) MX_CASE(8)
        MX_CASE(9) MX_CASE(10) MX_CASE(11) MX_CASE(12) MX_CASE(13) MX_CASE(14) MX_CASE(15) MX_CASE(16)
#undef MX_CASE
        default: return;
    }
    if (ks > 1) mx_reduce_kernel<<<(rows * N + 255) / 256, 256, 0, s>>>(part, ks, rows, N, y, ldy, beta);
    check("mx_rows");
}

void mxfp8_to_bf16(const uint8_t* w, const uint8_t* scale, int N, int K, uint16_t* out, cudaStream_t s) {
    if (N <= 0 || K <= 0) return;
    mxfp8_kernel<<<dim3((unsigned) std::min(64, (K + 255) / 256), (unsigned) N), 256, 0, s>>>(w, scale, K, out);
    check("mxfp8_to_bf16");
}

void to_bf16(const float* x, int rows, int cols, int ld, uint16_t* y, int ldo, cudaStream_t s) {
    if (rows <= 0) return;
    to_bf16_kernel<<<rows, 256, 0, s>>>(x, cols, ld, y, ldo);
    check("to_bf16");
}

void window_attn(const float* q, const uint16_t* kc, const uint16_t* vc, int ring, const int* lo, int hi, int T,
                 int Hq, int Hkv, float scale, float* out, cudaStream_t s) {
    if (T <= 0) return;
    window_attn_kernel<<<dim3(T, Hkv), 128, 0, s>>>(q, kc, vc, ring, lo, hi, Hq, Hkv, scale, out);
    check("window_attn");
}

void gather_row(const uint16_t* table, int R, const int* prev, uint16_t* emb, float* embf, cudaStream_t s) {
    gather_row_kernel<<<1, 256, 0, s>>>(table, R, prev, emb, embf);
    check("gather_row");
}

void markov_pick(const float* logits, const float* bias, int V, int* prev, int* out, float* biased, const float* w,
                 const float* b, const float* hidden, int D, const float* embf, int R, float* conf, cudaStream_t s) {
    markov_pick_kernel<<<1, 1024, 0, s>>>(logits, bias, V, prev, out, biased, w, b, hidden, D, embf, R, conf);
    check("markov_pick");
}

}  // namespace mm3::k
