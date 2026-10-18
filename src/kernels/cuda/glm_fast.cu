// src/kernels/cuda/glm_fast.cu - see include/strata/kernels/glm_fast.hpp.
//
// The dot products are llama.cpp's vecdotq (ggml/src/ggml-cuda/vecdotq.cuh, MIT, third_party/ggml/LICENSE),
// the same expressions iq_kernels.cu and native_mmvq.cu transcribe; the block structs and codebooks come
// from ggml-common.h.  Every reduction below is parallel, so the summation ORDER differs from the
// serial reference kernels - the values agree to float rounding, which the fast-vs-reference check in
// glm_pack_test pins on the real model.
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/exl3.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/kernels/glm_expert_bench.hpp"
#include <cstring>
#endif

#if !defined(STRATA_USE_HIP)
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"   // sm_60: __dp4a -> the vmad fallback

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "dsa_topk.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>

#if defined(STRATA_USE_HIP)
// The HC grid exchanges FP32 partials between workgroups. An agent-scope
// acquire load preserves visibility after their publication fences; HIP only
// supplies CUDA's __ldcg spelling for half types.
__device__ __forceinline__ float __ldcg(const float* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif

// __grid_constant__ (a kernel's by-value struct argument read in place, never copied) is a compute_70 feature: CUDA
// 12.9's nvcc refuses it in a Pascal pass ("__grid_constant__ annotation is only allowed for architecture compute_70
// or later", 22 kernels here), so below sm_70 it is left out - the arguments are const either way.  HIP keeps
// hip_compat's empty definition (__CUDA_ARCH__ is undefined there).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define STRATA_GRID_CONSTANT __grid_constant__
#else
#define STRATA_GRID_CONSTANT
#endif
// the device pass for cc 6.0 (Tesla P100 / PH402, GP100: no dp4a, the SIMD-in-word intrinsics emulated)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610 && !defined(STRATA_USE_HIP)
#define GLMF_SM60 1
#else
#define GLMF_SM60 0
#endif

namespace strata::kernels::glmf {
namespace {

// ---------------------------------------------------------------- small helpers
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum; every thread gets the total.  sh: >= 32 floats.  blockDim % 32 == 0.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    return t;
}
__device__ __forceinline__ float block_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int i = 0; i < nw; ++i) t = fmaxf(t, sh[i]);
    return t;
}
// one q8_1 block from 32 values held one per lane (quantize.cu's arithmetic)
__device__ __forceinline__ void q8_1_store_warp(float v, block_q8_1* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    // the scale and the sum stay finite in FP16 (Strata #1448: an activation past ~8.3M overflowed the scale to inf,
    // and the dot products to NaN); below that nothing changes
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}
__device__ __forceinline__ float dsigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t) v << 16); }

// ---------------------------------------------------------------- llama.cpp vecdotq helpers
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// --- K quants (bq8_1 = the q8_1 blocks of this weight block's 256 values)
__device__ __forceinline__ float dot_q4_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q4_K* bq4 = (const block_q4_K*) vbq + kbx;
    int v[2], u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = (const int*) (bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];
    const uint16_t* scales = (const uint16_t*) bq4->scales;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(bq4->dm);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float dot_q5_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q5_K* bq5 = (const block_q5_K*) vbq + kbx;
    int vl[2], vh[2], u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = (const int*) (bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    const uint16_t* scales = (const uint16_t*) bq5->scales;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = __dp4a(v0i, u[2 * i], __dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i], __dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(bq5->dm);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

// sm_60: Q6_K's integer steps as exact rewrites (the float expressions and their order untouched, so every sum is the
// generic form's bit for bit) - worth ~9% of the Q6_K GEMV once its loads stopped being the limit (mv_q6_p100_kernel):
//   q - 32 per byte, q in [0, 63]: (q + 0x60) ^ 0x80 never carries out of its byte - __vsubss4(q, 0x20202020) (which
//     never saturates there) in 2 instructions instead of its SIMD-in-word emulation;
//   dp4a * scale: |dp4a| <= 4 * 32 * 128 and |scale| <= 128 fit 16 bits - one mul.wide.s16 for the 32-bit multiply;
//   the product to float: |p| <= 2^21 < 2^22, so 0x4B400000 + p is the float 1.5 * 2^23 + p exactly and subtracting
//     1.5 * 2^23 is exact - an add and an FADD for the conversion (the same float as (float) p).
#if GLMF_SM60
__device__ __forceinline__ int q6_m32(int q) { return (q + 0x60606060) ^ (int) 0x80808080; }
__device__ __forceinline__ int mul_s16(int a, int b) {
    int r;
    asm("mul.wide.s16 %0, %1, %2;" : "=r"(r) : "h"((short) a), "h"((short) b));
    return r;
}
__device__ __forceinline__ float i2f_s22(int p) { return __int_as_float(p + 0x4B400000) - 12582912.0f; }
#endif

__device__ __forceinline__ float dot_q6_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q6_K* w = (const block_q6_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = get_int_b2(w->ql, iqs);
    const int vh = get_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    const int8_t* scales = w->scales + scale_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int u = ((const int*) bq8_1[bq8_offset + 2 * i].qs)[iqs % 8];
        const float d8 = __low2float(bq8_1[bq8_offset + 2 * i].ds);
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
#if GLMF_SM60
        sumf += d8 * i2f_s22(mul_s16(__dp4a(q6_m32(vil | vih), u, 0), sc));
#else
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8 * (__dp4a(vi, u, 0) * sc);
#endif
    }
    return __half2float(w->d) * sumf;
}

__device__ __forceinline__ float dot_q8_0(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q8_0* w = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = get_int_b2(w->qs, iqs + i);
        const int u = ((const int*) bq8_1->qs)[iqs + i];
        sumi = __dp4a(v, u, sumi);
    }
    return __half2float(w->d) * __low2float(bq8_1->ds) * float(sumi);
}

// --- the same dots split in two for several activation columns: load() decodes this call's weights once, load_x()
// takes one column's values for the call once, dot() combines them - the arithmetic of dot_q8_0 / dot_q4_K / dot_q6_K,
// bit for bit
template<int T> struct DecW;
template<> struct DecW<8> {
    int v[2];
    float d;
    struct X {
        int u[2];
        float d8;
    };
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_q8_0* w = (const block_q8_0*) vbq + kbx;
        v[0] = get_int_b2(w->qs, iqs);
        v[1] = get_int_b2(w->qs, iqs + 1);
        d = __half2float(w->d);
    }
    __device__ __forceinline__ static X load_x(const block_q8_1* bq8_1, int iqs) {
        X x;
        x.u[0] = ((const int*) bq8_1->qs)[iqs];
        x.u[1] = ((const int*) bq8_1->qs)[iqs + 1];
        x.d8 = __low2float(bq8_1->ds);
        return x;
    }
    __device__ __forceinline__ float dot(const X& x) const {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) sumi = __dp4a(v[i], x.u[i], sumi);
        return d * x.d8 * float(sumi);
    }
};
template<> struct DecW<12> {
    int v0i[2], v1i[2];
    int sc[2], m[2];
    float2 dm;
    struct X {
        int u[4];
        int su[2];   // the sum of each half's 8 values (the dmin term's dot: the same for every row)
        float d8[2];
    };
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_q4_K* bq4 = (const block_q4_K*) vbq + kbx;
        const int off = 2 * ((iqs / 2) / 4), q = (iqs / 2) % 4;
        const int* ql = (const int*) (bq4->qs + 16 * off + 4 * q);
        const int v0 = ql[0], v1 = ql[4];
        const uint16_t* scales = (const uint16_t*) bq4->scales;
        const int j = off / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        uint16_t aux[2];
        aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        const uint8_t* scb = (const uint8_t*) aux;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            v0i[i] = (v0 >> (4 * i)) & 0x0f0f0f0f;
            v1i[i] = (v1 >> (4 * i)) & 0x0f0f0f0f;
            sc[i] = scb[i];
            m[i] = scb[2 + i];
        }
        dm = __half22float2(bq4->dm);
    }
    // a column's values for the call: the same q8_1 ints for every row of the k step (they depend on iqs only)
    __device__ __forceinline__ static X load_x(const block_q8_1* bq8_1, int iqs) {
        X x;
        const int o = 2 * ((iqs / 2) / 4), qq = (iqs / 2) % 4;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const block_q8_1* bq8i = bq8_1 + o + i;
            x.d8[i] = __low2float(bq8i->ds);
            const int* q8 = (const int*) bq8i->qs + qq;
            x.u[2 * i] = q8[0];
            x.u[2 * i + 1] = q8[4];
            x.su[i] = __dp4a(0x01010101, x.u[2 * i + 1], __dp4a(0x01010101, x.u[2 * i], 0));
        }
        return x;
    }
    __device__ __forceinline__ float dot(const X& x) const {
        float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u0 = x.u[2 * i], u1 = x.u[2 * i + 1];
            const int dot1 = __dp4a(v1i[i], u1, __dp4a(v0i[i], u0, 0));
            sumf_d += x.d8[i] * (dot1 * sc[i]);
            sumf_m += x.d8[i] * (x.su[i] * m[i]);
        }
        return dm.x * sumf_d - dm.y * sumf_m;
    }
};
template<> struct DecW<14> {
    int vi[2], sc[2];
    float d;
    struct X {
        int u[2];
        float d8[2];
    };
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_q6_K* w = (const block_q6_K*) vbq + kbx;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vl = get_int_b2(w->ql, iqs);
        const int vh = get_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        const int8_t* scales = w->scales + scale_offset;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sc[i] = scales[4 * i];
            const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
#if GLMF_SM60
            vi[i] = q6_m32(vil | vih);   // (dot_q6_K's sm_60 rewrites)
#else
            vi[i] = __vsubss4(vil | vih, 0x20202020);
#endif
        }
        d = __half2float(w->d);
    }
    __device__ __forceinline__ static X load_x(const block_q8_1* bq8_1, int iqs) {
        X x;
        const int o = 4 * (iqs / 16) + (iqs % 16) / 8, qq = iqs % 8;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            x.u[i] = ((const int*) bq8_1[o + 2 * i].qs)[qq];
            x.d8[i] = __low2float(bq8_1[o + 2 * i].ds);
        }
        return x;
    }
    __device__ __forceinline__ float dot(const X& x) const {
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i)
#if GLMF_SM60
            sumf += x.d8[i] * i2f_s22(mul_s16(__dp4a(vi[i], x.u[i], 0), sc[i]));
#else
            sumf += x.d8[i] * (__dp4a(vi[i], x.u[i], 0) * sc[i]);
#endif
        return d * sumf;
    }
};

// --- i-quants (iq_kernels.cu's transcriptions)
// --- Q2_K / Q3_K (the NextN block's experts): llama.cpp's vec_dot_q2_K_q8_1 / vec_dot_q3_K_q8_1, VDR 1 - 16 calls per
// 256-block, call iqs reading int (iqs % 8) of the 4 q8_1 blocks 4 * (iqs / 8) .. + 3
__device__ __forceinline__ float dot_q2_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q2_K* b = (const block_q2_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const uint8_t* scales = b->scales + (iqs - iqs % 8 + (iqs % 8) / 4);
    const int v = get_int_b4(b->qs, iqs);
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const block_q8_1* q8 = bq8_1 + bq8_offset + i;
        const int u = get_int_b4(q8->qs, iqs % 8);
        const float d8 = __low2float(q8->ds);
        const int sc = scales[2 * i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        sumf_d += d8 * (__dp4a(vi, u, 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += d8 * __dp4a(m, u, 0);
    }
    const float2 dm = __half22float2(b->dm);
    return dm.x * sumf_d - dm.y * sumf_m;
}
__device__ __forceinline__ float dot_q3_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q3_K* b = (const block_q3_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = __half2float(b->d);
    const int vl = get_int_b2(b->qs, iqs);
    const int vh = ~get_int_b2(b->hmask, iqs % 8) >> bq8_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const block_q8_1* q8 = bq8_1 + bq8_offset + i;
        const int u = get_int_b4(q8->qs, iqs % 8);
        const float d8 = __low2float(q8->ds);
        const int isc = scale_offset + 2 * i;
        const int sc_low = (b->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((b->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        sumf += d8 * (__dp4a((int) __vsubss4(vil, vih), u, 0) * sc);
    }
    return d * sumf;
}
__device__ __forceinline__ float dot_iq2_xxs(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = __dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
__device__ __forceinline__ float dot_iq3_xxs(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = __dp4a(grid_l, u0, sumi);
        sumi = __dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
__device__ __forceinline__ float dot_iq1_s(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = __dp4a(grid0, u0, sumi);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x * sumi + ds.y * delta);
}
__device__ __forceinline__ float dot_iq4_xs(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = __dp4a(v.x, u0, sumi);
        sumi = __dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

// qk = values per block, ipb = dot calls per block, step = iqs stride, bytes = block bytes
template<int T> struct F;
template<> struct F<12> { static constexpr int qk = 256, ipb = 16, step = 2, bytes = sizeof(block_q4_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q4_K(v, y, kbx, iqs); } };
template<> struct F<13> { static constexpr int qk = 256, ipb = 16, step = 2, bytes = sizeof(block_q5_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q5_K(v, y, kbx, iqs); } };
template<> struct F<14> { static constexpr int qk = 256, ipb = 32, step = 1, bytes = sizeof(block_q6_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q6_K(v, y, kbx, iqs); } };
template<> struct F<8> { static constexpr int qk = 32, ipb = 4, step = 2, bytes = sizeof(block_q8_0);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q8_0(v, y, kbx, iqs); } };
template<> struct F<16> { static constexpr int qk = 256, ipb = 8, step = 2, bytes = sizeof(block_iq2_xxs);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq2_xxs(v, y, kbx, iqs); } };
template<> struct F<18> { static constexpr int qk = 256, ipb = 8, step = 2, bytes = sizeof(block_iq3_xxs);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq3_xxs(v, y, kbx, iqs); } };
template<> struct F<19> { static constexpr int qk = 256, ipb = 8, step = 1, bytes = sizeof(block_iq1_s);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq1_s(v, y, kbx, iqs); } };
template<> struct F<10> { static constexpr int qk = 256, ipb = 16, step = 1, bytes = sizeof(block_q2_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q2_K(v, y, kbx, iqs); } };
template<> struct F<11> { static constexpr int qk = 256, ipb = 16, step = 1, bytes = sizeof(block_q3_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q3_K(v, y, kbx, iqs); } };
template<> struct F<23> { static constexpr int qk = 256, ipb = 8, step = 4, bytes = sizeof(block_iq4_xs);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq4_xs(v, y, kbx, iqs); } };

// one row against one q8_1 activation, the whole warp (lane-strided calls)
template<int T>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int n_in, int lane) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    float s = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
        s += Fm::dot(row, x + kbx * (Fm::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}
__device__ __forceinline__ float row_dot_f32(const float* row, const float* x, int n_in, int lane) {
    float s = 0.0f;
    const float4* r4 = (const float4*) row;
    const float4* x4 = (const float4*) x;
    for (int k = lane; k < n_in / 4; k += 32) {
        const float4 a = r4[k], b = x4[k];
        s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    return warp_sum(s);
}
__device__ __forceinline__ float row_dot_bf16(const uint16_t* row, const float* x, int n_in, int lane) {
    float s = 0.0f;
    const uint4* r8 = (const uint4*) row;
    const float4* x4 = (const float4*) x;
    for (int k = lane; k < n_in / 8; k += 32) {
        const uint4 w = r8[k];
        const float4 a = x4[2 * k], b = x4[2 * k + 1];
        s += __uint_as_float(w.x << 16) * a.x + __uint_as_float(w.x & 0xffff0000u) * a.y +
             __uint_as_float(w.y << 16) * a.z + __uint_as_float(w.y & 0xffff0000u) * a.w +
             __uint_as_float(w.z << 16) * b.x + __uint_as_float(w.z & 0xffff0000u) * b.y +
             __uint_as_float(w.w << 16) * b.z + __uint_as_float(w.w & 0xffff0000u) * b.w;
    }
    return warp_sum(s);
}

template<int T>
__device__ __forceinline__ size_t rbytes(int n_in) { return (size_t) (n_in / F<T>::qk) * F<T>::bytes; }

// ---- the same i-quant dots with the codebook in SHARED memory (the expert kernels): the grid lookups are
// scattered across the 32 lanes, and from global memory every one of them replays through L1
__device__ __forceinline__ float dot_iq1_s_t(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs,
                                             const uint32_t* tab) {
    const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = __dp4a(grid0, u0, sumi);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x * sumi + ds.y * delta);
}
__device__ __forceinline__ float dot_iq2_xxs_t(const void* vbq, const block_q8_1* bq8_1, const int& kbx,
                                               const int& iqs, const uint32_t* tab) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = __dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
__device__ __forceinline__ float dot_iq3_xxs_t(const void* vbq, const block_q8_1* bq8_1, const int& kbx,
                                               const int& iqs, const uint32_t* tab) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = __dp4a(grid_l, u0, sumi);
        sumi = __dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
template<int T> __device__ __forceinline__ float dot_t(const void* v, const block_q8_1* y, int kbx, int iqs,
                                                       const uint32_t* tab);
template<> __device__ __forceinline__ float dot_t<19>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t* tab) { return dot_iq1_s_t(v, y, kbx, iqs, tab); }
template<> __device__ __forceinline__ float dot_t<16>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t* tab) { return dot_iq2_xxs_t(v, y, kbx, iqs, tab); }
template<> __device__ __forceinline__ float dot_t<18>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t* tab) { return dot_iq3_xxs_t(v, y, kbx, iqs, tab); }
template<> __device__ __forceinline__ float dot_t<23>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t*) { return dot_iq4_xs(v, y, kbx, iqs); }
// u32 words of the type's codebook (0 = none needed)
template<int T> struct TabWords { static constexpr int n = 0; };
template<> struct TabWords<19> { static constexpr int n = 2048; };
template<> struct TabWords<16> { static constexpr int n = 512; };
template<> struct TabWords<18> { static constexpr int n = 256; };
template<> struct TabWords<17> { static constexpr int n = 1024; };
template<> struct TabWords<22> { static constexpr int n = 2048; };
template<> struct TabWords<21> { static constexpr int n = 512; };
template<> struct TabWords<29> { static constexpr int n = 2048; };
template<int T>
__device__ __forceinline__ void load_tab(uint32_t* s_tab) {
    const uint32_t* src = T == 19 || T == 29 ? (const uint32_t*) iq1s_grid_gpu
                          : T == 16 ? (const uint32_t*) iq2xxs_grid
                          : T == 18 ? (const uint32_t*) iq3xxs_grid
                          : T == 17 ? (const uint32_t*) iq2xs_grid
                          : T == 22 ? (const uint32_t*) iq2s_grid
                          : T == 21 ? (const uint32_t*) iq3s_grid : nullptr;
    for (int i = threadIdx.x; i < TabWords<T>::n; i += blockDim.x) s_tab[i] = src[i];
}
// ---- the activation half of a dot, loaded ONCE and reused across rows: every i-quant call k reads exactly one
// q8_1 block (index kbx * qk/32 + k % ipb) and all 8 of its ints
struct XV {
    int u[8];
    float d, s;
};
__device__ __forceinline__ XV load_xv(const block_q8_1* xb) {
    XV v;
    const int* q = (const int*) xb->qs;
#pragma unroll
    for (int i = 0; i < 8; ++i) v.u[i] = q[i];
    const float2 ds = __half22float2(xb->ds);
    v.d = ds.x;
    v.s = ds.y;
    return v;
}
template<int T> __device__ __forceinline__ float dw(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab);
template<> __device__ __forceinline__ float dw<19>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq1_s* bq1 = (const block_iq1_s*) row + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        sumi = __dp4a((grid >> 0) & 0x0F0F0F0F, x.u[l0 + 0], sumi);
        sumi = __dp4a((grid >> 4) & 0x0F0F0F0F, x.u[l0 + 1], sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    return d1q * (x.d * sumi + x.s * delta);
}
template<> __device__ __forceinline__ float dw<16>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        sumi = __dp4a((int) __vsub4(grid_pos.x ^ signs0, signs0), x.u[k0 + 0], sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        sumi = __dp4a((int) __vsub4(grid_pos.y ^ signs1, signs1), x.u[k0 + 1], sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<18>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        sumi = __dp4a((int) __vsub4(grid_pos.x ^ signs0, signs0), x.u[l0 + 0], sumi);
        sumi = __dp4a((int) __vsub4(grid_pos.y ^ signs1, signs1), x.u[l0 + 1], sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    return __half2float(bq3->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<23>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t*) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) row + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = __dp4a(v.x, x.u[j + 0], sumi);
        sumi = __dp4a(v.y, x.u[j + 4], sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    return __half2float(bq4->d) * x.d * sumi;
}
// ---- the other i-quants of the Maya quants (docs/QUANT-PLAN.md): vecdotq's IQ2_XS, IQ2_S, IQ3_S and IQ1_M (the
// expressions iq_kernels.cu transcribes) in the same activation-preloaded, table-in-shared-memory form
template<> __device__ __forceinline__ float dw<17>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) row + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<22>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq2_s* bq2 = (const block_iq2_s*) row + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const int signs0 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<21>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq3_s* bq3 = (const block_iq3_s*) row + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)], tab[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        sumi = __dp4a(grid_l, x.u[l0 + 0], sumi);
        sumi = __dp4a(grid_h, x.u[l0 + 1], sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    return __half2float(bq3->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<29>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq1_m* bq1 = (const block_iq1_m*) row + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = tab[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        sumi[l0 / 4] = __dp4a((grid >> 0) & 0x0F0F0F0F, x.u[l0 + 0], sumi[l0 / 4]);
        sumi[l0 / 4] = __dp4a((grid >> 4) & 0x0F0F0F0F, x.u[l0 + 1], sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = __dp4a(x.u[l0 + 0], 0x01010101, sumy);
        sumy = __dp4a(x.u[l0 + 1], 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * x.d;
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}
// sm_60: the routed experts' i-quant sign decode through a table in shared memory (moe_gate_up_kernel /
// moe_down_kernel).  dw<T> spreads a sign byte's bits over the four bytes with __vcmpne4 and negates with __vsub4 -
// two SIMD-in-word ops that sm_60 emulates in ~10 instructions each.  The table holds each sign byte's byte masks
// for its low and high four bits (0xff where the value is negated); (g ^ m) + (m & 0x01010101) is __vsub4(g ^ m, m)
// because every IQ2_S / IQ3_XXS / IQ3_S grid byte is nonzero (negating a nonzero byte never carries into the next) -
// the same bytes, so the same sums.  IQ3_XXS: 7 sign bits + their parity (unpack_ksigns), 128 entries; IQ2_S /
// IQ3_S: the sign byte, 256.  On a PH402 die (8 experts, the decode shapes): IQ2_S gate/up 312 -> 273 us, IQ3_XXS
// down 167 -> 156, IQ3_S down 195 -> 185; IQ2_S keeps the two masks (the +1s by an AND: less shared-memory traffic,
// faster for it), IQ3_* the masks with their +1s.
template<int T> constexpr bool sm60_sign_on = GLMF_SM60 && (T == 18 || T == 21 || T == 22);
template<int T> constexpr int sm60_sign_n = T == 18 ? 128 : 256;
template<int T> constexpr int sm60_sign_w = T == 22 ? 2 : 4;   // u32 words an entry
#if GLMF_SM60
template<int T>
__device__ __forceinline__ void sm60_load_signs(uint32_t* t) {
    for (int i = threadIdx.x; i < sm60_sign_n<T>; i += blockDim.x) {
        unsigned b = i;
        if constexpr (T == 18) b |= (__popc(b) & 1) << 7;
        unsigned lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lo |= (0u - ((b >> j) & 1u)) & (255u << (8 * j));
            hi |= (0u - ((b >> (j + 4)) & 1u)) & (255u << (8 * j));
        }
        if constexpr (sm60_sign_w<T> == 4) ((uint4*) t)[i] = make_uint4(lo, lo & 0x01010101u, hi, hi & 0x01010101u);
        else ((uint2*) t)[i] = make_uint2(lo, hi);
    }
}
struct Sm60Sign {
    uint32_t lm, lp, hm, hp;   // the low / high four values' masks and +1s
};
template<int T>
__device__ __forceinline__ Sm60Sign sm60_sign(const uint32_t* t, uint32_t idx) {
    Sm60Sign r;
    if constexpr (sm60_sign_w<T> == 4) {
        const uint4 v = ((const uint4*) t)[idx];
        r.lm = v.x, r.lp = v.y, r.hm = v.z, r.hp = v.w;
    } else {
        const uint2 v = ((const uint2*) t)[idx];
        r.lm = v.x, r.lp = v.x & 0x01010101u, r.hm = v.y, r.hp = v.y & 0x01010101u;
    }
    return r;
}
// dw<T> with the table's signs (st: sm60_load_signs<T>'s table)
template<int T> __device__ __forceinline__ float dws(const uint8_t* row, int kbx, int iqs, const XV& x,
                                                     const uint32_t* tab, const uint32_t* st);
template<> __device__ __forceinline__ float dws<18>(const uint8_t* row, int kbx, int iqs, const XV& x,
                                                    const uint32_t* tab, const uint32_t* st) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
        const Sm60Sign sg = sm60_sign<18>(st, (aux32 >> (7 * l0 / 2)) & 127);
        sumi = __dp4a((int) ((grid_pos.x ^ sg.lm) + sg.lp), x.u[l0 + 0], sumi);
        sumi = __dp4a((int) ((grid_pos.y ^ sg.hm) + sg.hp), x.u[l0 + 1], sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    return __half2float(bq3->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dws<22>(const uint8_t* row, int kbx, int iqs, const XV& x,
                                                    const uint32_t* tab, const uint32_t* st) {
    const block_iq2_s* bq2 = (const block_iq2_s*) row + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const Sm60Sign sg = sm60_sign<22>(st, signs_packed_8[l0 / 2]);
        const int grid_l = (int) ((grid_pos.x ^ sg.lm) + sg.lp);
        const int grid_h = (int) ((grid_pos.y ^ sg.hm) + sg.hp);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dws<21>(const uint8_t* row, int kbx, int iqs, const XV& x,
                                                    const uint32_t* tab, const uint32_t* st) {
    const block_iq3_s* bq3 = (const block_iq3_s*) row + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)], tab[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const Sm60Sign sg = sm60_sign<21>(st, signs_packed_8[l0 / 2]);
        sumi = __dp4a((int) ((grid_pos.x ^ sg.lm) + sg.lp), x.u[l0 + 0], sumi);
        sumi = __dp4a((int) ((grid_pos.y ^ sg.hm) + sg.hp), x.u[l0 + 1], sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    return __half2float(bq3->d) * x.d * sumi;
}
#endif
// HIP sign decode adapted from Strata src/prefill/moe_fused_iq.cu (MIT).
// Every IQ2/IQ3 grid byte is nonzero and <= 62, so two's-complement negation
// can use a whole-word add without carries across byte boundaries. Multiplying
// a four-bit sign nibble by 0x10204080 puts its bits at each byte's high bit;
// after masking/shifting, these are the +1 corrections and *255 gives the masks.
// The integer scale divisions and floating-point accumulation order stay intact.
#if defined(STRATA_USE_HIP)
__device__ __forceinline__ uint4 direct_signs(unsigned b, bool parity) {
    b &= parity ? 127u : 255u;
    if (parity) b |= (__popc(b) & 1) << 7;
    const unsigned lo = (((b & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    const unsigned hi = ((((b >> 4) & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    return make_uint4(lo * 255u, lo, hi * 255u, hi);
}
__device__ __forceinline__ uint2 quant_load8(const void* p) {
    uint2 words;
    __builtin_memcpy(&words, p, sizeof(words)); // GGUF super-blocks are only 2-byte aligned.
    return words;
}
template<int T> constexpr int sign_count = T == 22 ? 256 : 128;
template<int T>
__device__ __forceinline__ void load_signs(uint4* signs) {
    for (int i = threadIdx.x; i < sign_count<T>; i += blockDim.x) {
        unsigned b = i;
        if constexpr (T != 22) b |= (__popc(b) & 1) << 7;
        unsigned lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lo |= (0u - ((b >> j) & 1u)) & (255u << (8 * j));
            hi |= (0u - ((b >> (j + 4)) & 1u)) & (255u << (8 * j));
        }
        signs[i] = make_uint4(lo, lo & 0x01010101u, hi, hi & 0x01010101u);
    }
}
template<int T>
__device__ __forceinline__ float dw_rdna(const uint8_t* row, int kbx, int iqs, const XV& x,
                                        const uint32_t* tab, const uint4* signs, bool direct, bool wide);
template<> __device__ __forceinline__ float dw_rdna<16>(const uint8_t* row, int kbx, int iqs,
    const XV& x, const uint32_t* tab, const uint4* sign_tab, bool direct, bool wide) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
    const int q2 = wide ? quant_load8(bq2->qs + 2 * iqs).x : get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = wide ? quant_load8(bq2->qs + 2 * iqs).y : get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
        const uint4 signs = direct ? direct_signs(aux32 >> (7 * k0 / 2), true) : sign_tab[(aux32 >> (7 * k0 / 2)) & 127];
        sumi = __dp4a((int) ((grid_pos.x ^ signs.x) + signs.y), x.u[k0 + 0], sumi);
        sumi = __dp4a((int) ((grid_pos.y ^ signs.z) + signs.w), x.u[k0 + 1], sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw_rdna<18>(const uint8_t* row, int kbx, int iqs,
    const XV& x, const uint32_t* tab, const uint4* sign_tab, bool direct, bool wide) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
    const uint2 q3_wide = wide ? quant_load8(bq3->qs + 4 * iqs) : make_uint2(0, 0);
    const int2 q3_packed = wide ? make_int2(q3_wide.x, q3_wide.y) :
        make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
        const uint4 signs = direct ? direct_signs(aux32 >> (7 * l0 / 2), true) : sign_tab[(aux32 >> (7 * l0 / 2)) & 127];
        sumi = __dp4a((int) ((grid_pos.x ^ signs.x) + signs.y), x.u[l0 + 0], sumi);
        sumi = __dp4a((int) ((grid_pos.y ^ signs.z) + signs.w), x.u[l0 + 1], sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    return __half2float(bq3->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw_rdna<22>(const uint8_t* row, int kbx, int iqs,
    const XV& x, const uint32_t* tab, const uint4* sign_tab, bool direct, bool wide) {
    (void) wide;
    const block_iq2_s* bq2 = (const block_iq2_s*) row + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const uint4 signs = direct ? direct_signs(signs_packed_8[l0 / 2], false) : sign_tab[signs_packed_8[l0 / 2]];
        const int grid_l = ((grid_pos.x ^ signs.x) + signs.y);
        const int grid_h = ((grid_pos.y ^ signs.z) + signs.w);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * x.d * sumi;
}
template<int T, int NR>
__device__ __forceinline__ void rows_dot_rdna(const uint8_t* const* rows, const block_q8_1* x,
                                             int n_in, int lane, const uint32_t* tab,
                                             const uint4* signs, float* s, bool direct = false, bool wide = false) {
    float acc[NR] = {};
    for (int k = lane; k < n_in / 32; k += 32) {
        const XV xv = load_xv(x + k);
#pragma unroll
        for (int r = 0; r < NR; ++r)
            acc[r] += dw_rdna<T>(rows[r], k / 8, 2 * (k % 8), xv, tab, signs, direct, wide);
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) s[r] = warp_sum(acc[r]);
}
#endif

// their F entries (the dense GEMV and row_dot read the codebook from global memory; the call's q8_1 block is
// iqs / step of the super-block's)
#define GLMF_IQ_F(T, BLOCK, STEP, TABLE)                                                                          \
    template<> struct F<T> { static constexpr int qk = 256, ipb = 8, step = STEP, bytes = sizeof(BLOCK);         \
        __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) {                      \
            return dw<T>((const uint8_t*) v, kbx, iqs, load_xv(y + iqs / STEP), (const uint32_t*) TABLE); } };
GLMF_IQ_F(17, block_iq2_xs, 2, iq2xs_grid)
GLMF_IQ_F(22, block_iq2_s, 2, iq2s_grid)
GLMF_IQ_F(21, block_iq3_s, 2, iq3s_grid)
GLMF_IQ_F(29, block_iq1_m, 1, iq1s_grid_gpu)
#undef GLMF_IQ_F
// NR rows against ONE activation, the whole warp; s[r] gets each row's full sum
// the K-quant calls read 4 q8_1 blocks' int (iqs % 8) each: loaded ONCE per call position for all rows
struct XK {
    int u[4];
    float d8[4];
};
__device__ __forceinline__ XK load_xk(const block_q8_1* xb, int iqs) {
    XK v;
    const int off = 4 * (iqs / 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v.u[i] = get_int_b4(xb[off + i].qs, iqs % 8);
        v.d8[i] = __low2float(xb[off + i].ds);
    }
    return v;
}
template<int T> __device__ __forceinline__ float dwk(const uint8_t* row, int kbx, int iqs, const XK& x);
template<> __device__ __forceinline__ float dwk<10>(const uint8_t* row, int kbx, int iqs, const XK& x) {
    const block_q2_K* b = (const block_q2_K*) row + kbx;
    const uint8_t* scales = b->scales + (iqs - iqs % 8 + (iqs % 8) / 4);
    const int v = get_int_b4(b->qs, iqs);
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int sc = scales[2 * i];
        sumf_d += x.d8[i] * (__dp4a((v >> (2 * i)) & 0x03030303, x.u[i], 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += x.d8[i] * __dp4a(m, x.u[i], 0);
    }
    const float2 dm = __half22float2(b->dm);
    return dm.x * sumf_d - dm.y * sumf_m;
}
template<> __device__ __forceinline__ float dwk<11>(const uint8_t* row, int kbx, int iqs, const XK& x) {
    const block_q3_K* b = (const block_q3_K*) row + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const int vl = get_int_b2(b->qs, iqs);
    const int vh = ~get_int_b2(b->hmask, iqs % 8) >> bq8_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int sc_low = (b->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((b->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        sumf += x.d8[i] * (__dp4a((int) __vsubss4(vil, vih), x.u[i], 0) * ((sc_low | sc_high) - 32));
    }
    return __half2float(b->d) * sumf;
}
// (SG: sm_60's sign table st - sm60_sign_on<T> types only)
template<int T, int NR, bool SG = false>
__device__ __forceinline__ void rows_dot(const uint8_t* const* rows, const block_q8_1* x, int n_in, int lane,
                                         const uint32_t* tab, float* s, const uint32_t* st = nullptr) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    float acc[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) acc[r] = 0.0f;
    if constexpr (T == 10 || T == 11) {
        for (int k = lane; k < nb * Fm::ipb; k += 32) {
            const int kbx = k / Fm::ipb, iqs = k % Fm::ipb;
            const XK xk = load_xk(x + kbx * (Fm::qk / 32), iqs);
#pragma unroll
            for (int r = 0; r < NR; ++r) acc[r] += dwk<T>(rows[r], kbx, iqs, xk);
        }
    } else if constexpr (T == 8 || T == 12 || T == 13 || T == 14) {
        // Q4_K / Q5_K / Q6_K experts (K-quant GGUFs other than Maya's): no shared-activation form yet - each row
        // takes the dense GEMV's dot (row_dot), the activation re-read per row
        for (int k = lane; k < nb * Fm::ipb; k += 32) {
            const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
#pragma unroll
            for (int r = 0; r < NR; ++r) acc[r] += Fm::dot(rows[r], x + kbx * (Fm::qk / 32), kbx, iqs);
        }
    } else {
        for (int k = lane; k < nb * Fm::ipb; k += 32) {
            const int kbx = k / Fm::ipb, ki = k % Fm::ipb, iqs = Fm::step * ki;
            const XV xv = load_xv(x + kbx * (Fm::qk / 32) + ki);
#if GLMF_SM60
            if constexpr (SG && sm60_sign_on<T>) {
#pragma unroll
                for (int r = 0; r < NR; ++r) acc[r] += dws<T>(rows[r], kbx, iqs, xv, tab, st);
                continue;
            }
#endif
            (void) st;
#pragma unroll
            for (int r = 0; r < NR; ++r) acc[r] += dw<T>(rows[r], kbx, iqs, xv, tab);
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) s[r] = warp_sum(acc[r]);
}

template<int T>
__device__ __forceinline__ float row_dot_t(const uint8_t* row, const block_q8_1* x, int n_in, int lane,
                                           const uint32_t* tab) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    float s = 0.0f;
#pragma unroll 4
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
        s += dot_t<T>(row, x + kbx * (Fm::qk / 32), kbx, iqs, tab);
    }
    return warp_sum(s);
}

__device__ __forceinline__ float any_row_dot(int type, const void* w, int row, const void* xq, const float* xf,
                                             int n_in, int lane) {
    switch (type) {
        case kTypeF32: return row_dot_f32((const float*) w + (size_t) row * n_in, xf, n_in, lane);
        case kTypeBF16: return row_dot_bf16((const uint16_t*) w + (size_t) row * n_in, xf, n_in, lane);
#define GF_CASE(T) case T: return row_dot<T>((const uint8_t*) w + (size_t) row * rbytes<T>(n_in), (const block_q8_1*) xq, n_in, lane);
        GF_CASE(12) GF_CASE(13) GF_CASE(14) GF_CASE(8) GF_CASE(16) GF_CASE(18) GF_CASE(19) GF_CASE(23) GF_CASE(10) GF_CASE(11)
        GF_CASE(17) GF_CASE(22) GF_CASE(21) GF_CASE(29)
#undef GF_CASE
        default: return 0.0f;
    }
}

// One row against ncol q8_1 activations (xs bytes apart) - a verify window's or a draft block's tokens - the whole warp:
// each weight block is read once for all of them (back to back, from L1), where a column loop around row_dot read the
// row again from L2 / DRAM for every column.  Column c's sum is row_dot's for it alone, bit for bit (the same calls in
// the same order).
template<int T>
__device__ __forceinline__ void row_dot_cols(const uint8_t* row, const uint8_t* x, int64_t xs, int ncol, int n_in,
                                             int lane, float* s) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
#pragma unroll
    for (int c = 0; c < kMaxMvCols; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
#pragma unroll
        for (int c = 0; c < kMaxMvCols; ++c)
            if (c < ncol) s[c] += Fm::dot(row, (const block_q8_1*) (x + c * xs) + kbx * (Fm::qk / 32), kbx, iqs);
    }
#pragma unroll
    for (int c = 0; c < kMaxMvCols; ++c)
        if (c < ncol) s[c] = warp_sum(s[c]);
}
__device__ __forceinline__ void row_dot_bf16_cols(const uint16_t* row, const float* x, int64_t xs, int ncol, int n_in,
                                                  int lane, float* s) {
#pragma unroll
    for (int c = 0; c < kMaxMvCols; ++c) s[c] = 0.0f;
    const uint4* r8 = (const uint4*) row;
    for (int k = lane; k < n_in / 8; k += 32) {
        const uint4 w = r8[k];
        const float w0 = __uint_as_float(w.x << 16), w1 = __uint_as_float(w.x & 0xffff0000u);
        const float w2 = __uint_as_float(w.y << 16), w3 = __uint_as_float(w.y & 0xffff0000u);
        const float w4 = __uint_as_float(w.z << 16), w5 = __uint_as_float(w.z & 0xffff0000u);
        const float w6 = __uint_as_float(w.w << 16), w7 = __uint_as_float(w.w & 0xffff0000u);
#pragma unroll
        for (int c = 0; c < kMaxMvCols; ++c)
            if (c < ncol) {
                const float4* x4 = (const float4*) (x + c * xs);
                const float4 a = x4[2 * k], b = x4[2 * k + 1];
                s[c] += w0 * a.x + w1 * a.y + w2 * a.z + w3 * a.w + w4 * b.x + w5 * b.y + w6 * b.z + w7 * b.w;
            }
    }
#pragma unroll
    for (int c = 0; c < kMaxMvCols; ++c)
        if (c < ncol) s[c] = warp_sum(s[c]);
}

// ---------------------------------------------------------------- multi-job GEMV
struct MvBatch {
    MvJob j[kMaxMvJobs];
    int blk_end[kMaxMvJobs];
    int n;
    const int* skip;   // set_skip's flag (nullptr: none)
};

// set_skip: the calling thread's flag for mv / moe_gate_up / moe_down (a pipelined window's enqueue threads)
thread_local const int* g_skip = nullptr;

__global__ void skip_gate_kernel(const volatile int* host_from, int t, int* flag) {
    *flag = *host_from <= t ? 1 : 0;
}
constexpr int MV_ROWS = 8;   // rows (warps) per block

// a GEMV's output: alpha * s + bias, the product rounded and then the sum, in every GEMV kernel alike - left to the
// compiler, the two were fused in some kernels and not in others (CUDA: none of the one-activation ones; ROCm: it
// depended on the instantiation), so a batched output could differ in its last bit from the same one alone.  Without a
// bias (every call of the decode) both ways give the same bits.
__device__ __forceinline__ float mv_out(float alpha, float s, float bias) {
#if defined(STRATA_USE_HIP)
    float m = alpha * s;
    asm volatile("" : "+v"(m));   // the product materialised: never fused with the sum
    return m + bias;
#else
    return __fadd_rn(__fmul_rn(alpha, s), bias);
#endif
}

__global__ void __launch_bounds__(256) mv_kernel(const STRATA_GRID_CONSTANT MvBatch b) {
    if (b.skip != nullptr && *b.skip) return;
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    const float s = any_row_dot(J.type, J.w, row, J.xq, J.xf, J.n_in, lane);
    if (lane == 0) J.y[row] = mv_out(J.alpha, s, J.bias ? J.bias[row] : 0.0f);
}

// ... several columns (a verify window's tokens) of a type the tiled kernels below do not take, or every type with
// STRATA_GLM_MV_TILED=0: kernels of their own, so the one-column kernels keep their registers and their speed
__global__ void __launch_bounds__(256) mv_kernel_cols(const STRATA_GRID_CONSTANT MvBatch b) {
    if (b.skip != nullptr && *b.skip) return;
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    if (J.type == kTypeBF16) {
        float s[kMaxMvCols];
        row_dot_bf16_cols((const uint16_t*) J.w + (size_t) row * J.n_in, J.xf, J.xf_cstride, J.ncol, J.n_in, lane, s);
        if (lane == 0)
#pragma unroll
            for (int c = 0; c < kMaxMvCols; ++c)
                if (c < J.ncol) J.y[c * J.y_cstride + row] = mv_out(J.alpha, s[c], J.bias ? J.bias[row] : 0.0f);
        return;
    }
    for (int c = 0; c < J.ncol; ++c) {   // (the other types: the row's later columns read its weights from cache)
        const float s = any_row_dot(J.type, J.w, row, (const uint8_t*) J.xq + c * J.xq_cstride,
                                    J.xf + c * J.xf_cstride, J.n_in, lane);
        if (lane == 0) J.y[c * J.y_cstride + row] = mv_out(J.alpha, s, J.bias ? J.bias[row] : 0.0f);
    }
}

// ... the same for jobs of ONE quantized type: compiled for that type alone (the switch over every type costs the
// generic kernel registers and occupancy - a Q6_K 8192 x 4096 GEMV on a V100: 53.4 -> 47.8 us, the same arithmetic)
template<int T>
__global__ void __launch_bounds__(256) mv_kernel_t(const STRATA_GRID_CONSTANT MvBatch b) {
    if (b.skip != nullptr && *b.skip) return;
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    const float s = row_dot<T>((const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in), (const block_q8_1*) J.xq,
                               J.n_in, lane);
    if (lane == 0) J.y[row] = mv_out(J.alpha, s, J.bias ? J.bias[row] : 0.0f);
}
template<int T>
__global__ void __launch_bounds__(256) mv_kernel_t_cols(const STRATA_GRID_CONSTANT MvBatch b) {
    if (b.skip != nullptr && *b.skip) return;
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    float s[kMaxMvCols];
    row_dot_cols<T>((const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in), (const uint8_t*) J.xq, J.xq_cstride, J.ncol,
                    J.n_in, lane, s);
    if (lane == 0)
#pragma unroll
        for (int c = 0; c < kMaxMvCols; ++c)
            if (c < J.ncol) J.y[c * J.y_cstride + row] = mv_out(J.alpha, s[c], J.bias ? J.bias[row] : 0.0f);
}

// ---------------------------------------------------------------- the same GEMVs for NT activations at once (mv_nt)
// Several conversations' tokens through one layer: activation t is xs q8_1 blocks (or n_in floats) after activation
// t - 1, output t n_out floats after output t - 1.  Each lane makes row_dot's calls in row_dot's order for every
// activation, and each sum ends in the same warp_sum, so column t is bit for bit the one-activation kernel's; the
// weight row is fetched from memory once for all NT.  Separate kernels: the one-activation ones keep their registers.
template<int T, int NT>
__device__ __forceinline__ void row_dot_nt(const uint8_t* row, const block_q8_1* x, size_t xs, int n_in, int lane,
                                           float (&s)[NT]) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] += Fm::dot(row, x + (size_t) t * xs + kbx * (Fm::qk / 32), kbx, iqs);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) s[t] = warp_sum(acc[t]);
}
template<int NT>
__device__ __forceinline__ void row_dot_f32_nt(const float* row, const float* x, int n_in, int lane, float (&s)[NT]) {
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    const float4* r4 = (const float4*) row;
    for (int k = lane; k < n_in / 4; k += 32) {
        const float4 a = r4[k];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 b = ((const float4*) (x + (size_t) t * n_in))[k];
            acc[t] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) s[t] = warp_sum(acc[t]);
}
template<int NT>
__device__ __forceinline__ void row_dot_bf16_nt(const uint16_t* row, const float* x, int n_in, int lane,
                                                float (&s)[NT]) {
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    const uint4* r8 = (const uint4*) row;
    for (int k = lane; k < n_in / 8; k += 32) {
        const uint4 w = r8[k];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4* x4 = (const float4*) (x + (size_t) t * n_in);
            const float4 a = x4[2 * k], b = x4[2 * k + 1];
            acc[t] += __uint_as_float(w.x << 16) * a.x + __uint_as_float(w.x & 0xffff0000u) * a.y +
                      __uint_as_float(w.y << 16) * a.z + __uint_as_float(w.y & 0xffff0000u) * a.w +
                      __uint_as_float(w.z << 16) * b.x + __uint_as_float(w.z & 0xffff0000u) * b.y +
                      __uint_as_float(w.w << 16) * b.z + __uint_as_float(w.w & 0xffff0000u) * b.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) s[t] = warp_sum(acc[t]);
}
template<int NT>
__device__ __forceinline__ void any_row_dot_nt(int type, const void* w, int row, const void* xq, const float* xf,
                                               int n_in, int lane, float (&s)[NT]) {
    switch (type) {
        case kTypeF32: row_dot_f32_nt<NT>((const float*) w + (size_t) row * n_in, xf, n_in, lane, s); return;
        case kTypeBF16: row_dot_bf16_nt<NT>((const uint16_t*) w + (size_t) row * n_in, xf, n_in, lane, s); return;
#define GF_CASE_NT(T)                                                                                               \
    case T:                                                                                                         \
        row_dot_nt<T, NT>((const uint8_t*) w + (size_t) row * rbytes<T>(n_in), (const block_q8_1*) xq,             \
                          (size_t) (n_in / 32), n_in, lane, s);                                                     \
        return;
        GF_CASE_NT(12) GF_CASE_NT(13) GF_CASE_NT(14) GF_CASE_NT(8) GF_CASE_NT(16) GF_CASE_NT(18) GF_CASE_NT(19)
        GF_CASE_NT(23) GF_CASE_NT(10) GF_CASE_NT(11) GF_CASE_NT(17) GF_CASE_NT(22) GF_CASE_NT(21) GF_CASE_NT(29)
#undef GF_CASE_NT
        default:
#pragma unroll
            for (int t = 0; t < NT; ++t) s[t] = 0.0f;
    }
}
template<int NT>
__global__ void __launch_bounds__(256) mv_kernel_nt(const STRATA_GRID_CONSTANT MvBatch b) {
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    float s[NT];
    any_row_dot_nt<NT>(J.type, J.w, row, J.xq, J.xf, J.n_in, lane, s);
    if (lane == 0) {
        const float bias = J.bias ? J.bias[row] : 0.0f;
#pragma unroll
        for (int t = 0; t < NT; ++t) J.y[(size_t) t * J.n_out + row] = mv_out(J.alpha, s[t], bias);
    }
}
template<int T, int NT>
__global__ void __launch_bounds__(256) mv_kernel_t_nt(const STRATA_GRID_CONSTANT MvBatch b) {
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    float s[NT];
    row_dot_nt<T, NT>((const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in), (const block_q8_1*) J.xq,
                      (size_t) (J.n_in / 32), J.n_in, lane, s);
    if (lane == 0) {
        const float bias = J.bias ? J.bias[row] : 0.0f;
#pragma unroll
        for (int t = 0; t < NT; ++t) J.y[(size_t) t * J.n_out + row] = mv_out(J.alpha, s[t], bias);
    }
}

// ---- several columns (a verify window's, a draft block's tokens) of a Q8_0 matrix on the tensor cores (sm_80+:
// mma.sync m16n8k32 s8): a block takes 16 rows, its 8 warps split the row's 32-value blocks, each block's 16 x 32 weights
// against 8 columns' q8_1 values in one int8 MMA, scaled by the two blocks' d and summed in float; the warps' partials
// meet in shared memory.  The dp4a GEMV's per-column work grew linearly with the columns (8 columns ~3x one, however
// the loads were arranged); this one reads the weights once at the memory's speed.  The sums are not the single
// column's bit for bit (the order differs) - the same value to float rounding.
constexpr int MMA_W = 8;   // warps a block

__global__ void __launch_bounds__(MMA_W * 32) mvc_mma_q8_0(const STRATA_GRID_CONSTANT MvBatch b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(STRATA_USE_HIP)
    if (b.skip != nullptr && *b.skip) return;
    __shared__ float red[MMA_W][16][kMaxMvCols + 1];
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row0 = (bid - blk0) * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const int nb = J.n_in / 32, ncol = J.ncol, ntiles = (ncol + 7) / 8;
    const size_t rb = (size_t) nb * sizeof(block_q8_0);
    const block_q8_0* wa = (const block_q8_0*) ((const uint8_t*) J.w + (size_t) min(row0 + g, J.n_out - 1) * rb);
    const block_q8_0* wb = (const block_q8_0*) ((const uint8_t*) J.w + (size_t) min(row0 + g + 8, J.n_out - 1) * rb);
    const uint8_t* xq = (const uint8_t*) J.xq;
    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    // MMA_U consecutive blocks a warp at a time, their loads first (one 34-byte block a row a step left too little in
    // flight: 154880 x 4096, 8 columns, RTX 3090 1254 -> 1133 us, RTX 5060 Ti 2038 -> 1725; eight blocks: 1183 / 1714)
    constexpr int MMA_U = 4;
    for (int kb0 = warp * MMA_U; kb0 < nb; kb0 += MMA_W * MMA_U) {
        int qa[MMA_U][4];
        float dd[MMA_U][2];
#pragma unroll
        for (int u = 0; u < MMA_U; ++u) {
            const int kb = min(kb0 + u, nb - 1);
            qa[u][0] = get_int_b2(wa[kb].qs, tig);
            qa[u][2] = get_int_b2(wa[kb].qs, 4 + tig);
            qa[u][1] = get_int_b2(wb[kb].qs, tig);
            qa[u][3] = get_int_b2(wb[kb].qs, 4 + tig);
            dd[u][0] = __half2float(wa[kb].d);
            dd[u][1] = __half2float(wb[kb].d);
        }
#pragma unroll
        for (int u = 0; u < MMA_U; ++u) {
        const int kb = kb0 + u;
        if (kb >= nb) break;
        const int a0 = qa[u][0], a1 = qa[u][1], a2 = qa[u][2], a3 = qa[u][3];
        const float da = dd[u][0], db = dd[u][1];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            if (t >= ntiles) break;
            const block_q8_1* xb = (const block_q8_1*) (xq + (int64_t) min(8 * t + g, ncol - 1) * J.xq_cstride) + kb;
            const int b0 = ((const int*) xb->qs)[tig], b1 = ((const int*) xb->qs)[4 + tig];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                         "{%0,%1,%2,%3};\n"
                         : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            const float dx0 = __low2float(((const block_q8_1*) (xq + (int64_t) min(8 * t + 2 * tig, ncol - 1) *
                                                                        J.xq_cstride) + kb)->ds);
            const float dx1 = __low2float(((const block_q8_1*) (xq + (int64_t) min(8 * t + 2 * tig + 1, ncol - 1) *
                                                                        J.xq_cstride) + kb)->ds);
            acc[t][0] += da * dx0 * (float) c0;
            acc[t][1] += da * dx1 * (float) c1;
            acc[t][2] += db * dx0 * (float) c2;
            acc[t][3] += db * dx1 * (float) c3;
        }
        }
    }
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        red[warp][g][8 * t + 2 * tig] = acc[t][0];
        red[warp][g][8 * t + 2 * tig + 1] = acc[t][1];
        red[warp][g + 8][8 * t + 2 * tig] = acc[t][2];
        red[warp][g + 8][8 * t + 2 * tig + 1] = acc[t][3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * kMaxMvCols; i += blockDim.x) {
        const int rr = i / kMaxMvCols, cc = i % kMaxMvCols;
        if (cc >= ncol || row0 + rr >= J.n_out) continue;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < MMA_W; ++w) v += red[w][rr][cc];
        J.y[cc * J.y_cstride + row0 + rr] = J.alpha * v + (J.bias ? J.bias[row0 + rr] : 0.0f);
    }
#endif
}

// ---- several columns of a Q4_K matrix on the tensor cores, for jobs that allow float-rounding differences
// (MvJob::approx: a drafter's GEMVs).  A block takes 16 rows, its warps split the rows' 256-value super-blocks; each
// 32-value sub-block's 16 x 32 weights (4-bit, widened to int8) against 8 columns' q8_1 values in one mma.sync
// m16n8k32, then its scale and min in float (the min against the block's int8 sum, as the dp4a kernel).  A lane's eight k slots are eight
// consecutive values - one 8-byte load of the packed pair of sub-blocks, the activations in the same order.  The
// dp4a kernel re-reads every column's activations for each pair of rows (8 columns: ~half the memory's speed)
__global__ void __launch_bounds__(MMA_W * 32) mvc_mma_q4_K(const STRATA_GRID_CONSTANT MvBatch b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(STRATA_USE_HIP)
    if (b.skip != nullptr && *b.skip) return;
    __shared__ float red[MMA_W][16][kMaxMvCols + 1];
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row0 = (bid - blk0) * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const int nb = J.n_in / 256, ncol = J.ncol, ntiles = (ncol + 7) / 8;
    const size_t rb = (size_t) nb * sizeof(block_q4_K);
    const block_q4_K* wa = (const block_q4_K*) ((const uint8_t*) J.w + (size_t) min(row0 + g, J.n_out - 1) * rb);
    const block_q4_K* wb = (const block_q4_K*) ((const uint8_t*) J.w + (size_t) min(row0 + g + 8, J.n_out - 1) * rb);
    const uint8_t* xq = (const uint8_t*) J.xq;
    // this lane's columns: g for the B fragment, 2 tig and 2 tig + 1 for its sums (a padding column re-reads the last)
    const block_q8_1* xg[2];
    const block_q8_1* xs0[2];
    const block_q8_1* xs1[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        xg[t] = (const block_q8_1*) (xq + (int64_t) min(8 * t + g, ncol - 1) * J.xq_cstride);
        xs0[t] = (const block_q8_1*) (xq + (int64_t) min(8 * t + 2 * tig, ncol - 1) * J.xq_cstride);
        xs1[t] = (const block_q8_1*) (xq + (int64_t) min(8 * t + 2 * tig + 1, ncol - 1) * J.xq_cstride);
    }
    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    for (int kb = warp; kb < nb; kb += MMA_W) {
        int2 qa[4], qb[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            qa[p] = *(const int2*) (wa[kb].qs + 32 * p + 8 * tig);
            qb[p] = *(const int2*) (wb[kb].qs + 32 * p + 8 * tig);
        }
        const float2 dma = __half22float2(wa[kb].dm), dmb = __half22float2(wb[kb].dm);
        // the 6-bit scales and mins of the 8 sub-blocks, a byte each (ggml's get_scale_min_k4, four at once)
        const int* sa = (const int*) wa[kb].scales;
        const int* sb = (const int*) wb[kb].scales;
        const int a0s = sa[0], a1s = sa[1], a2s = sa[2], b0s = sb[0], b1s = sb[1], b2s = sb[2];
        const int sc_a[2] = {a0s & 0x3f3f3f3f, (a2s & 0x0f0f0f0f) | ((a0s >> 2) & 0x30303030)};
        const int mn_a[2] = {a1s & 0x3f3f3f3f, ((a2s >> 4) & 0x0f0f0f0f) | ((a1s >> 2) & 0x30303030)};
        const int sc_b[2] = {b0s & 0x3f3f3f3f, (b2s & 0x0f0f0f0f) | ((b0s >> 2) & 0x30303030)};
        const int mn_b[2] = {b1s & 0x3f3f3f3f, ((b2s >> 4) & 0x0f0f0f0f) | ((b1s >> 2) & 0x30303030)};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int p = j >> 1, sh = 4 * (j & 1), by = 8 * (j & 3);
            const int a0 = (qa[p].x >> sh) & 0x0f0f0f0f, a2 = (qa[p].y >> sh) & 0x0f0f0f0f;
            const int a1 = (qb[p].x >> sh) & 0x0f0f0f0f, a3 = (qb[p].y >> sh) & 0x0f0f0f0f;
            const float fa = dma.x * (float) ((sc_a[j >> 2] >> by) & 0xff), ma = dma.y * (float) ((mn_a[j >> 2] >> by) & 0xff);
            const float fb = dmb.x * (float) ((sc_b[j >> 2] >> by) & 0xff), mb = dmb.y * (float) ((mn_b[j >> 2] >> by) & 0xff);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                if (t >= ntiles) break;
                const block_q8_1* xb = xg[t] + kb * 8 + j;
                const int b0 = ((const int*) xb->qs)[2 * tig], b1 = ((const int*) xb->qs)[2 * tig + 1];
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                             "{%0,%1,%2,%3};\n"
                             : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                // the min term against the block's int8 sum (not the q8_1 block's FP16 sum: it clamps at 65504 and
                // rounds - a drafter's hidden-state cosine fell from 0.995 to 0.954 with it): column 8 t + g's sum
                // over the group's four lanes, then each lane takes its two columns' from their groups
                int sq = __dp4a(b1, 0x01010101, __dp4a(b0, 0x01010101, 0));
                sq += __shfl_xor_sync(0xffffffffu, sq, 1);
                sq += __shfl_xor_sync(0xffffffffu, sq, 2);
                const float s0 = (float) __shfl_sync(0xffffffffu, sq, 8 * tig), s1 = (float) __shfl_sync(0xffffffffu, sq, 8 * tig + 4);
                const float d0 = __low2float(xs0[t][kb * 8 + j].ds), d1 = __low2float(xs1[t][kb * 8 + j].ds);
                acc[t][0] += d0 * (fa * (float) c0 - ma * s0);
                acc[t][1] += d1 * (fa * (float) c1 - ma * s1);
                acc[t][2] += d0 * (fb * (float) c2 - mb * s0);
                acc[t][3] += d1 * (fb * (float) c3 - mb * s1);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        red[warp][g][8 * t + 2 * tig] = acc[t][0];
        red[warp][g][8 * t + 2 * tig + 1] = acc[t][1];
        red[warp][g + 8][8 * t + 2 * tig] = acc[t][2];
        red[warp][g + 8][8 * t + 2 * tig + 1] = acc[t][3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * kMaxMvCols; i += blockDim.x) {
        const int rr = i / kMaxMvCols, cc = i % kMaxMvCols;
        if (cc >= ncol || row0 + rr >= J.n_out) continue;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < MMA_W; ++w) v += red[w][rr][cc];
        J.y[cc * J.y_cstride + row0 + rr] = J.alpha * v + (J.bias ? J.bias[row0 + rr] : 0.0f);
    }
#endif
}

// ---- several columns, register-blocked dp4a (any device; bit for bit the single-column kernel's sums): a warp takes
// MVC_R rows at once, each lane decodes its weight call of every row ONCE (DecW<T>) and loads each column's values for
// the call ONCE (DecW<T>::load_x) for all of them - NC columns, a compile-time 2 / 4 / 8 / 16 (the job's ncol rounded
// up: a padding column re-reads the last one and is not stored).  Each lane visits its k steps in row_dot's order with
// the same arithmetic, so a (row, column) sum is the single-column kernel's.
constexpr int MVC_R = 4;   // rows a warp - 2 for Q8_0 / Q4_K from 8 columns (four rows of those ran out of registers:
                           // 8 columns took 2.7x the time of 4; with two, 1.6-1.8x), Q6_K keeps four (two: slower)
constexpr int mvc_rows(int type, int nc) { return nc >= 8 && type != 14 ? 2 : MVC_R; }

template<int T, int NC, int R = mvc_rows(T, NC)>
__global__ void __launch_bounds__(256) mvc_kernel_t(const STRATA_GRID_CONSTANT MvBatch b) {
    if (b.skip != nullptr && *b.skip) return;
    using Fm = F<T>;
    using D = DecW<T>;
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row0 = ((bid - blk0) * MV_ROWS + (threadIdx.x >> 5)) * R;
    const int lane = threadIdx.x & 31;
    if (row0 >= J.n_out) return;
    const int ncol = J.ncol;
    const size_t rb = rbytes<T>(J.n_in);
    const uint8_t* w[R];
#pragma unroll
    for (int r = 0; r < R; ++r) w[r] = (const uint8_t*) J.w + (size_t) min(row0 + r, J.n_out - 1) * rb;
    const uint8_t* xc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) xc[c] = (const uint8_t*) J.xq + (int64_t) min(c, ncol - 1) * J.xq_cstride;
    float s[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) s[r][c] = 0.0f;
    const int nk = (J.n_in / Fm::qk) * Fm::ipb;
    for (int k = lane; k < nk; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
        D dw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) dw[r].load(w[r], kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const typename D::X xv = D::load_x((const block_q8_1*) xc[c] + kbx * (Fm::qk / 32), iqs);
#pragma unroll
            for (int r = 0; r < R; ++r) s[r][c] += dw[r].dot(xv);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float v = warp_sum(s[r][c]);
            if (lane == 0 && c < ncol && row0 + r < J.n_out)
                J.y[c * J.y_cstride + row0 + r] = mv_out(J.alpha, v, J.bias ? J.bias[row0 + r] : 0.0f);
        }
}

template<int T>
void mvc_launch(const MvBatch& b, int blocks, int nc, cudaStream_t s) {
    if (nc <= 2) mvc_kernel_t<T, 2><<<blocks, 256, 0, s>>>(b);
    else if (nc <= 4) mvc_kernel_t<T, 4><<<blocks, 256, 0, s>>>(b);
    else if (nc <= 8) mvc_kernel_t<T, 8><<<blocks, 256, 0, s>>>(b);
    else mvc_kernel_t<T, 16><<<blocks, 256, 0, s>>>(b);
}

// whether this device has the int8 mma.sync of the tensor-core GEMVs (NVIDIA sm_80+), asked once per device
bool mma_arch() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static int cache[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    if (cache[dev] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        static const bool off = getenv("STRATA_GLM_MV_APPROX") != nullptr && std::atoi(getenv("STRATA_GLM_MV_APPROX")) == 0;
        cache[dev] = major >= 8 && !off ? 1 : 2;
    }
    return cache[dev] == 1;
#endif
}

// whether this device runs mvc_mma_q8_0 (NVIDIA sm_80+), asked once per device
bool mma_ok() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static int cache[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    if (cache[dev] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cache[dev] = 2;
        if (const char* v = getenv("STRATA_GLM_MV_MMA")) cache[dev] = std::atoi(v) != 0 && major >= 8 ? 1 : 2;
    }
    return cache[dev] == 1;
#endif
}

// cc 6.0 (Tesla P100 / PH402, GP100 - no dp4a, the VMAD emulation): the one-column Q6_K GEMV (the KDA / DSA
// projections, the shared experts' gate/up, the dense FFN, the head - ~2/3 of a decode token's bytes) runs at
// 135-150 GB/s of the 732 in mv_kernel_t, bound by its loads: a lane's call reads its weights in seven 8- and 16-bit
// pieces (the 210-byte blocks are only 2-aligned), the instructions' arithmetic does not matter (an exact rewrite of
// the integer work, 66 -> 52 SASS a row and call, changed nothing).  mv_q6_p100_kernel: a warp takes R rows, reads
// each row's block as 53 aligned words (two coalesced loads a lane) and hands every lane its call's fields with six
// shuffles; the activation is staged in shared memory once per block.  Each lane then makes dot_q6_K's calls in
// row_dot's order with the same ints and floats (DecW<14>), so every row's sum is mv_kernel_t's bit for bit.  On a
// PH402 die, us (mv_kernel_t -> this): the KDA q/k/v 24576 x 4096 561 -> 389, attn_output 4096 x 16384 406 -> 279,
// q_b 16384 x 1536 142 -> 105, the shared expert's gate/up 4096 x 4096 106 -> 79, q_a + kv_a 2048 x 4096 59 -> 48
// (R = 2), the FFN 12288 x 4096 287 -> 204, the head 154880 x 4096 3466 -> 2362.
// p100_mv_mode: 0 off (mv_kernel_t; STRATA_GLM_MV_P100=0), else R by the launch's rows (STRATA_GLM_MV_P100_R=2 / 4
// forces one), asked once per device
#if !defined(STRATA_USE_HIP)
// a Q6_K block's 53 words, two a lane: w0 = word lane, w1 = word 32 + lane (lanes 0-20) of the words starting sh = the
// block's address mod 4 (0 or 2) bytes before it; its first / last half word read as a half, so nothing outside the
// block is read
__device__ __forceinline__ void q6_words(const uint8_t* s, int lane, uint32_t& w0, uint32_t& w1, int& sh) {
    sh = (int) ((uintptr_t) s & 3);
    const uint32_t* a = (const uint32_t*) (s - sh);
    w0 = (lane == 0 && sh) ? (uint32_t) (*(const uint16_t*) s) << 16 : a[lane];
    w1 = lane > 20 ? 0u : (lane == 20 && !sh) ? (uint32_t) * (const uint16_t*) (s + 208) : a[32 + lane];
}
// DecW<14>::load's fields for iqs = lane from the block's words (q6_words; the whole warp, the same block)
__device__ __forceinline__ void q6_fields(DecW<14>& D, uint32_t w0, uint32_t w1, int sh, int lane) {
    const int f = 8 * sh;
    const uint32_t nx = __shfl_sync(0xffffffffu, lane == 0 ? w1 : w0, (lane + 1) & 31);
    const int vl = (int) __funnelshift_r(w0, nx, f);   // ql int lane
    const int qi = 8 * (lane / 16) + lane % 8;          // qh int qi
    const uint32_t h0 = __shfl_sync(0xffffffffu, w1, qi), h1 = __shfl_sync(0xffffffffu, w1, qi + 1);
    const int vh = (int) __funnelshift_r(h0, h1, f) >> (2 * ((lane % 16) / 8));
    const int sb = 8 * (lane / 16) + (lane % 16) / 4 + sh;   // scales[so] and [so + 4] (bytes 192 +)
    const uint32_t s0 = __shfl_sync(0xffffffffu, w1, 16 + (sb >> 2)), s1 = __shfl_sync(0xffffffffu, w1, 17 + (sb >> 2));
    D.sc[0] = (int) (int8_t) (s0 >> (8 * (sb & 3)));
    D.sc[1] = (int) (int8_t) (s1 >> (8 * (sb & 3)));
    const uint32_t dw = __shfl_sync(0xffffffffu, w1, 20);   // d (bytes 208-209)
    D.d = __half2float(__ushort_as_half((unsigned short) (dw >> f)));
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
#if GLMF_SM60
        D.vi[i] = q6_m32(vil | vih);   // (as DecW<14>::load)
#else
        D.vi[i] = __vsubss4(vil | vih, 0x20202020);
#endif
    }
}
// R rows a warp (MV_ROWS warps a block); dynamic shared memory: the largest job's n_in / 32 q8_1 blocks
template<int R>
__global__ void __launch_bounds__(256) mv_q6_p100_kernel(const STRATA_GRID_CONSTANT MvBatch b) {
    if (b.skip != nullptr && *b.skip) return;
    extern __shared__ int mvq6_xs[];
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    {
        const int nw = (J.n_in / 32) * (int) (sizeof(block_q8_1) / 4);
        for (int i = threadIdx.x; i < nw; i += blockDim.x) mvq6_xs[i] = ((const int*) J.xq)[i];
        __syncthreads();
    }
    const block_q8_1* xq = (const block_q8_1*) mvq6_xs;
    const int row0 = ((bid - blk0) * MV_ROWS + (threadIdx.x >> 5)) * R;
    const int lane = threadIdx.x & 31;
    if (row0 >= J.n_out) return;
    const size_t rb = rbytes<14>(J.n_in);
    const uint8_t* w[R];
#pragma unroll
    for (int r = 0; r < R; ++r) w[r] = (const uint8_t*) J.w + (size_t) min(row0 + r, J.n_out - 1) * rb;
    float s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) s[r] = 0.0f;
    const int nk = (J.n_in / 256) * 32;   // (a multiple of 32: every lane makes the same trips, the shuffles' warp whole)
    for (int k = lane; k < nk; k += 32) {
        const int kbx = k / 32, iqs = k % 32;
        uint32_t w0[R], w1[R];
        int sh[R];
#pragma unroll
        for (int r = 0; r < R; ++r) q6_words(w[r] + (size_t) kbx * sizeof(block_q6_K), lane, w0[r], w1[r], sh[r]);
        DecW<14> dw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) q6_fields(dw[r], w0[r], w1[r], sh[r], lane);
        const DecW<14>::X xv = DecW<14>::load_x(xq + kbx * 8, iqs);
#pragma unroll
        for (int r = 0; r < R; ++r) s[r] += dw[r].dot(xv);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(s[r]);
        if (lane == 0 && row0 + r < J.n_out) J.y[row0 + r] = mv_out(J.alpha, v, J.bias ? J.bias[row0 + r] : 0.0f);
    }
}
#endif
// moe_gate_up on cc 6.0 stages the layer input in shared memory (STRATA_GLM_GU_STAGE=0: it reads it from global)
bool sm60_gu_stage() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static int cache[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    if (cache[dev] == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        const char* v = getenv("STRATA_GLM_GU_STAGE");
        cache[dev] = major == 6 && minor == 0 && (v == nullptr || std::atoi(v) != 0) ? 1 : 2;
    }
    return cache[dev] == 1;
#endif
}
int p100_mv_mode() {
#if defined(STRATA_USE_HIP)
    return 0;
#else
    static int cache[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return 0;
    if (cache[dev] == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        const char* v = getenv("STRATA_GLM_MV_P100");
        const char* rv = getenv("STRATA_GLM_MV_P100_R");
        const int r = rv != nullptr ? std::atoi(rv) : 0;
        const bool on = major == 6 && minor == 0 && (v == nullptr || std::atoi(v) != 0);
        cache[dev] = on ? (r == 2 || r == 4 ? r : 1) : -1;   // 1: by the rows
    }
    return cache[dev] > 0 ? cache[dev] : 0;
#endif
}

// ... BF16 weights against float activations (the routers, the indexer's projections): the same staging, in floats
constexpr int MVC_FTILE = 512;   // floats of every column per tile

__global__ void __launch_bounds__(256) mvc_kernel_bf16(const STRATA_GRID_CONSTANT MvBatch b) {
    if (b.skip != nullptr && *b.skip) return;
    extern __shared__ __align__(16) uint8_t mvc_smem[];
    float* sx = (float*) mvc_smem;   // [ncol][MVC_FTILE]
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int ncol = J.ncol;
    const bool live = row < J.n_out;
    const uint4* r8 = (const uint4*) ((const uint16_t*) J.w + (size_t) (live ? row : 0) * J.n_in);
    float s[kMaxMvCols];
#pragma unroll
    for (int c = 0; c < kMaxMvCols; ++c) s[c] = 0.0f;
    int k = lane;   // this lane's next group of 8 values (row_dot_bf16's order)
    for (int t0 = 0; t0 < J.n_in; t0 += MVC_FTILE) {
        const int tn = min(MVC_FTILE, J.n_in - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < ncol * (tn / 4); i += blockDim.x) {
            const int c = i / (tn / 4), r = i % (tn / 4);
            ((float4*) (sx + c * MVC_FTILE))[r] = ((const float4*) (J.xf + c * J.xf_cstride + t0))[r];
        }
        __syncthreads();
        if (!live) continue;
        const int k_end = (t0 + tn) / 8;
        for (; k < k_end; k += 32) {
            const uint4 wv = r8[k];
            const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
            const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
            const float w4 = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
            const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
            const int o = 8 * k - t0;
#pragma unroll
            for (int c = 0; c < kMaxMvCols; ++c)
                if (c < ncol) {
                    const float4 a = *(const float4*) (sx + c * MVC_FTILE + o);
                    const float4 bb = *(const float4*) (sx + c * MVC_FTILE + o + 4);
                    s[c] += w0 * a.x + w1 * a.y + w2 * a.z + w3 * a.w + w4 * bb.x + w5 * bb.y + w6 * bb.z + w7 * bb.w;
                }
        }
    }
    if (!live) return;
#pragma unroll
    for (int c = 0; c < kMaxMvCols; ++c)
        if (c < ncol) {
            const float v = warp_sum(s[c]);
            if (lane == 0) J.y[c * J.y_cstride + row] = mv_out(J.alpha, v, J.bias ? J.bias[row] : 0.0f);
        }
}

#if defined(STRATA_USE_HIP)
// ---------------------------------------------------------------- the dense GEMV on RDNA3 (gfx11)
// Q6_K is ~64% of a decode token's bytes (Maya-S: the KDA/DSA projections, the shared experts, the dense FFN and the
// head), and on a Radeon 8065S row_dot<14> ran them at 165-220 GB/s of the 273 the LPDDR5X gives: its loop is not
// unrolled, so each wave keeps ONE 210-byte block (9 loads) in flight and then waits ~1-2 us for it - the APU's
// memory latency, not its bandwidth, set the pace.  These kernels issue the loads of U blocks before the arithmetic of
// the first, give a wave R output rows that share one activation load, and do Q6_K's "-32" with a second dot instead
// of __vsubss4's ten-instruction SWAR emulation (q - 32 never saturates: sum (q - 32) u = sum q u + sum (-32) u, both
// exact in v_dot4_i32_iu8).  Every row's value is still dot_q6_K's arithmetic in row_dot's order - lane L takes iqs L
// of every super-block, the blocks in order, then warp_sum - so the outputs are bit for bit mv_kernel_t's.
struct Q6W {   // one lane's share of one Q6_K super-block (dot_q6_K's loads)
    int vl, vh, sc0, sc1;
    float d;
};
struct Q6X {   // ... and of the q8_1 activation blocks it meets
    int u0, u1;
    float d0, d1;
};
__device__ __forceinline__ Q6W q6_wload(const block_q6_K* w, int lane) {
    Q6W r;
    r.vl = get_int_b2(w->ql, lane);
    r.vh = get_int_b2(w->qh, 8 * (lane / 16) + lane % 8) >> (2 * ((lane % 16) / 8));
    const int8_t* sc = w->scales + 8 * (lane / 16) + (lane % 16) / 4;
    r.sc0 = sc[0];
    r.sc1 = sc[4];
    r.d = __half2float(w->d);
    return r;
}
__device__ __forceinline__ Q6X q6_xload(const block_q8_1* x, int lane) {   // x: the super-block's 8 q8_1 blocks
    const block_q8_1* b = x + 4 * (lane / 16) + (lane % 16) / 8;
    Q6X r;
    r.u0 = ((const int*) b[0].qs)[lane % 8];
    r.d0 = __low2float(b[0].ds);
    r.u1 = ((const int*) b[2].qs)[lane % 8];
    r.d1 = __low2float(b[2].ds);
    return r;
}
// sum (q_i - 32) u_i for four 6-bit q: __dp4a(__vsubss4(q, 0x20202020), u, 0) without the SWAR subtract
__device__ __forceinline__ int q6_idot(int q, int u) {
#if __has_builtin(__builtin_amdgcn_sudot4) && (defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__))
    return __builtin_amdgcn_sudot4(true, (int) 0xe0e0e0e0u, true, u, __builtin_amdgcn_sudot4(false, q, true, u, 0, false),
                                   false);
#else
    return __dp4a(__vsubss4(q, 0x20202020), u, 0);
#endif
}
// dot_q6_K's float arithmetic, term for term (the same products, sums and order)
__device__ __forceinline__ float q6_val(const Q6W& w, const Q6X& x) {
    float sumf = 0.0f;
    sumf += x.d0 * (q6_idot((w.vl & 0x0f0f0f0f) | ((w.vh << 4) & 0x30303030), x.u0) * w.sc0);
    sumf += x.d1 * (q6_idot(((w.vl >> 4) & 0x0f0f0f0f) | (w.vh & 0x30303030), x.u1) * w.sc1);
    return w.d * sumf;
}
// R weight rows (rb bytes apart) x NT activations (xs q8_1 blocks apart), nb super-blocks, U of them per step
template<int R, int NT, int U>
__device__ __forceinline__ void q6_rows_rdna(const uint8_t* w0, size_t rb, const block_q8_1* x, size_t xs, int nb,
                                             int lane, float (&s)[R][NT]) {
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
    int b = 0;
    for (; b + U <= nb; b += U) {
        Q6W w[U][R];
        Q6X xv[U][NT];
#pragma unroll
        for (int k = 0; k < U; ++k) {
#pragma unroll
            for (int r = 0; r < R; ++r) w[k][r] = q6_wload((const block_q6_K*) (w0 + (size_t) r * rb) + b + k, lane);
#pragma unroll
            for (int t = 0; t < NT; ++t) xv[k][t] = q6_xload(x + (size_t) t * xs + (size_t) (b + k) * 8, lane);
        }
#pragma unroll
        for (int k = 0; k < U; ++k)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int t = 0; t < NT; ++t) acc[r][t] += q6_val(w[k][r], xv[k][t]);
    }
    for (; b < nb; ++b) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const Q6W w = q6_wload((const block_q6_K*) (w0 + (size_t) r * rb) + b, lane);
#pragma unroll
            for (int t = 0; t < NT; ++t) acc[r][t] += q6_val(w, q6_xload(x + (size_t) t * xs + (size_t) b * 8, lane));
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) s[r][t] = warp_sum(acc[r][t]);
}
// rows_dot_bf16 / row_dot_bf16 with U loads in flight per lane (the same expression per 8 values, the same order)
template<int NT, int U>
__device__ __forceinline__ void bf16_rows_rdna(const uint16_t* row, const float* x, int n_in, int lane, float (&s)[NT]) {
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    const uint4* r8 = (const uint4*) row;
#pragma unroll U
    for (int k = lane; k < n_in / 8; k += 32) {
        const uint4 w = r8[k];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4* x4 = (const float4*) (x + (size_t) t * n_in);
            const float4 a = x4[2 * k], b = x4[2 * k + 1];
            acc[t] += __uint_as_float(w.x << 16) * a.x + __uint_as_float(w.x & 0xffff0000u) * a.y +
                      __uint_as_float(w.y << 16) * a.z + __uint_as_float(w.y & 0xffff0000u) * a.w +
                      __uint_as_float(w.z << 16) * b.x + __uint_as_float(w.z & 0xffff0000u) * b.y +
                      __uint_as_float(w.w << 16) * b.z + __uint_as_float(w.w & 0xffff0000u) * b.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) s[t] = warp_sum(acc[t]);
}
// One launch for a batch's Q6_K AND BF16 jobs (the router, the KDA gates, the indexer projections ride along with the
// big Q6_K GEMVs instead of a second, 36-block launch behind them).  A Q6_K job's block holds WPB waves of R rows, a
// BF16 job's WPB waves of one row; the dispatcher puts the BF16 jobs first so their long-K rows start early.
template<int R, int NT, int U, int WPB, bool XL>
__global__ void __launch_bounds__(WPB * 32) mv_rdna_kernel(const STRATA_GRID_CONSTANT MvBatch b) {
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    if (J.type == 14) {
        const block_q8_1* xq = (const block_q8_1*) J.xq;
        if constexpr (XL) {   // the block's activation rows staged in LDS once (dynamic: NT * n_in / 32 q8_1 blocks)
            extern __shared__ int mv_rdna_xs[];
            const int nw = NT * (J.n_in / 32) * (int) (sizeof(block_q8_1) / 4);
            for (int i = threadIdx.x; i < nw; i += WPB * 32) mv_rdna_xs[i] = ((const int*) J.xq)[i];
            __syncthreads();
            xq = (const block_q8_1*) mv_rdna_xs;
        }
        const int row = ((bid - blk0) * WPB + wave) * R;
        if (row >= J.n_out) return;
        const size_t rb = (size_t) (J.n_in / 256) * sizeof(block_q6_K);
        float s[R][NT];
        if (row + R <= J.n_out) {
            q6_rows_rdna<R, NT, U>((const uint8_t*) J.w + (size_t) row * rb, rb, xq, (size_t) (J.n_in / 32),
                                   J.n_in / 256, lane, s);
        } else {   // the job's last rows: one at a time
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row + r >= J.n_out) break;
                float s1[1][NT];
                q6_rows_rdna<1, NT, U>((const uint8_t*) J.w + (size_t) (row + r) * rb, rb, xq, (size_t) (J.n_in / 32),
                                       J.n_in / 256, lane, s1);
#pragma unroll
                for (int t = 0; t < NT; ++t) s[r][t] = s1[0][t];
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row + r >= J.n_out) break;
                const float bias = J.bias ? J.bias[row + r] : 0.0f;
#pragma unroll
                for (int t = 0; t < NT; ++t) J.y[(size_t) t * J.n_out + row + r] = mv_out(J.alpha, s[r][t], bias);
            }
        }
    } else {   // kTypeBF16
        const int row = (bid - blk0) * WPB + wave;
        if (row >= J.n_out) return;
        float s[NT];
        bf16_rows_rdna<NT, 4>((const uint16_t*) J.w + (size_t) row * J.n_in, J.xf, J.n_in, lane, s);
        if (lane == 0) {
            const float bias = J.bias ? J.bias[row] : 0.0f;
#pragma unroll
            for (int t = 0; t < NT; ++t) J.y[(size_t) t * J.n_out + row] = mv_out(J.alpha, s[t], bias);
        }
    }
}
#endif

__global__ void quantize_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // n % 32 == 0: whole warps only
    q8_1_store_warp(x[i], y + i / 32, threadIdx.x & 31);
}

// ---------------------------------------------------------------- mHC
constexpr int HC_CHUNK = 512;
constexpr int HC_THREADS = 256;
constexpr int HC_MIX = 24;      // (2 + hc) * hc for hc = 4

__global__ void __launch_bounds__(HC_THREADS) hc_kernel(const STRATA_GRID_CONSTANT HcArgs a) {
    const int hc = 4;
    const int n_embd = a.n_embd;
    const int hc_dim = hc * n_embd;
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float sR[HC_CHUNK];
    __shared__ float sred[32];
    __shared__ float spart[HC_MIX];
    __shared__ bool s_last;

    // ---- phase 1: this block's chunk of R_new, its sum of squares and the 24 partial dots
    float ss = 0.0f;
    for (int i = tid; i < HC_CHUNK; i += HC_THREADS) {
        const int idx = b * HC_CHUNK + i;
        float r;
        if (a.block_out != nullptr) {
            const int d = idx / n_embd, e = idx - d * n_embd;
            r = a.block_out[e] * a.post_in[d];
#pragma unroll
            for (int s = 0; s < hc; ++s) r += a.comb_in[d * hc + s] * a.R_old[s * n_embd + e];
            a.R_new[idx] = r;
        } else {
            r = a.R_old[idx];
        }
        sR[i] = r;
        ss += r * r;
    }
    __syncthreads();
    for (int m = warp; m < HC_MIX; m += HC_THREADS / 32) {
        const uint16_t* wr = a.w_fn + (size_t) m * hc_dim + (size_t) b * HC_CHUNK;
        float acc = 0.0f;
        for (int i = lane; i < HC_CHUNK; i += 32) acc += sR[i] * bf(wr[i]);
        acc = warp_sum(acc);
        if (lane == 0) spart[m] = acc;
    }
    ss = block_sum(ss, sred);
    if (tid < HC_MIX) a.part[(size_t) b * 25 + tid] = spart[tid];
    if (tid == 0) a.part[(size_t) b * 25 + 24] = ss;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(a.counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last) return;

    // ---- phase 2 (the last block): the gates, the Sinkhorn, mixed, the norm, q8_1
    __shared__ float smix[HC_MIX + 1];
    __shared__ float s_pre[4];
    __shared__ float s_mixed[4096];
    if (tid == 0) *a.counter = 0;
    if (tid < 25) {
        float acc = 0.0f;
        for (int bb = 0; bb < (int) gridDim.x; ++bb) acc += __ldcg(&a.part[(size_t) bb * 25 + tid]);
        smix[tid] = acc;
    }
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) hc_dim + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            const float post_v = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
            a.pre[lane] = pre_v;
            s_pre[lane] = pre_v;
            a.post[lane] = post_v;
        }
        // comb[d][s] on lane d*4 + s (lanes 0..15); lanes 16..31 shadow harmlessly
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        // softmax over dst (d) per column s: lanes sharing s differ in bits 2..3
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        // dst-normalise: sum over s (bits 0..1); src-normalise: sum over d (bits 2..3)
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (lane < 16) a.comb[d * hc + s] = c;
    }
    __syncthreads();
    const float* Rn = a.block_out != nullptr ? a.R_new : a.R_old;
    float ss2 = 0.0f;
    for (int e = tid; e < n_embd; e += HC_THREADS) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * __ldcg(&Rn[s * n_embd + e]);
        s_mixed[e] = acc;
        ss2 += acc * acc;
    }
    ss2 = block_sum(ss2, sred);
    const float inv2 = rsqrtf(ss2 / (float) n_embd + a.norm_eps);
    for (int e = tid; e < n_embd; e += HC_THREADS) {   // warp w holds 32 consecutive e at every step
        const float xv = s_mixed[e] * inv2 * a.norm_w[e];
        a.x[e] = xv;
        q8_1_store_warp(xv, (block_q8_1*) a.xq + e / 32, lane);
    }
}

// ---- hc, take two: the grid splits by EMBEDDING range (all 4 streams of 128 values per block), so after the partial
// dots every block can finish its own range - two grid barriers (the blocks are co-resident: 32 blocks, checked
// at launch) instead of one block doing the whole tail.  Same arithmetic as hc_kernel.
constexpr int HC2_EB = 128;
__device__ __forceinline__ void grid_barrier(unsigned long long* ctr, unsigned int nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned long long v = atomicAdd(ctr, 1ull);
        const unsigned long long target = (v / nb + 1) * nb;
        while (*(volatile unsigned long long*) ctr < target) {
        }
        __threadfence();
    }
    __syncthreads();
}

__global__ void __launch_bounds__(HC_THREADS) hc2_kernel(const STRATA_GRID_CONSTANT HcArgs a) {
    const int hc = 4;
    const int n_embd = a.n_embd;
    const int hc_dim = hc * n_embd;
    const unsigned int nb = gridDim.x;
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int e0 = b * HC2_EB;
    unsigned long long* ctr = (unsigned long long*) a.counter;
    __shared__ float sR[4][HC2_EB];
    __shared__ float sred[32];
    __shared__ float spart[HC_MIX];
    __shared__ float smix[HC_MIX + 1];
    __shared__ float s_pre[4];
    __shared__ float s_mixed[HC2_EB];

    // ---- phase 1: this block's 4 x 128 values of R_new, their sum of squares and the 24 partial dots
    float ss = 0.0f;
    for (int i = tid; i < 4 * HC2_EB; i += HC_THREADS) {
        const int d = i / HC2_EB, j = i - d * HC2_EB, e = e0 + j;
        float r;
        if (a.block_out != nullptr) {
            r = a.block_out[e] * a.post_in[d];
#pragma unroll
            for (int s = 0; s < hc; ++s) r += a.comb_in[d * hc + s] * a.R_old[s * n_embd + e];
            a.R_new[d * n_embd + e] = r;
        } else {
            r = a.R_old[d * n_embd + e];
        }
        sR[d][j] = r;
        ss += r * r;
    }
    __syncthreads();
    for (int m = warp; m < HC_MIX; m += HC_THREADS / 32) {
        const uint16_t* wr = a.w_fn + (size_t) m * hc_dim + e0;
        float acc = 0.0f;
        for (int i = lane; i < 4 * HC2_EB; i += 32) {
            const int d = i / HC2_EB, j = i - d * HC2_EB;
            acc += sR[d][j] * bf(wr[(size_t) d * n_embd + j]);
        }
        acc = warp_sum(acc);
        if (lane == 0) spart[m] = acc;
    }
    ss = block_sum(ss, sred);
    if (tid < HC_MIX) a.part[(size_t) b * 25 + tid] = spart[tid];
    if (tid == 0) a.part[(size_t) b * 25 + 24] = ss;
    grid_barrier(ctr, nb);

    // ---- phase 2 (every block, redundantly): the gates and the Sinkhorn from the partial sums
    if (tid < 25) {
        float acc = 0.0f;
        for (int bb = 0; bb < (int) nb; ++bb) acc += __ldcg(&a.part[(size_t) bb * 25 + tid]);
        smix[tid] = acc;
    }
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) hc_dim + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            s_pre[lane] = pre_v;
            if (b == 0) {
                a.pre[lane] = pre_v;
                a.post[lane] = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
            }
        }
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (b == 0 && lane < 16) a.comb[d * hc + s] = c;
    }
    __syncthreads();
    // ---- phase 3: mixed for this block's 128 values, the norm's sum of squares across the grid, x and q8_1
    float ss2 = 0.0f;
    if (tid < HC2_EB) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * sR[s][tid];
        s_mixed[tid] = acc;
        ss2 = acc * acc;
    }
    ss2 = block_sum(ss2, sred);
    float* part2 = a.part + (size_t) nb * 25;
    if (tid == 0) part2[b] = ss2;
    grid_barrier(ctr, nb);
    float tot = 0.0f;
    for (int bb = 0; bb < (int) nb; ++bb) tot += __ldcg(&part2[bb]);
    const float inv2 = rsqrtf(tot / (float) n_embd + a.norm_eps);
    if (tid < HC2_EB) {   // warps 0..3: 32 consecutive values each
        const int e = e0 + tid;
        const float xv = s_mixed[tid] * inv2 * a.norm_w[e];
        a.x[e] = xv;
        q8_1_store_warp(xv, (block_q8_1*) a.xq + e / 32, lane);
    }
}

// ---- hc, take three: hc2's arithmetic with the latency taken out - every block issues its 24 weight rows' loads
// (8 bf16 per lane per stream segment, uint4) BEFORE it computes its R values, the partial sums are reduced by whole
// warps (lane = block) instead of one thread walking 32 of them, and the second barrier's total likewise.
// 32 blocks of 128 embedding values; n_embd == 4096 only.
__global__ void __launch_bounds__(HC_THREADS) hc3_kernel(const STRATA_GRID_CONSTANT HcArgs a) {
    const int hc = 4;
    const int n_embd = a.n_embd;
    const unsigned int nb = gridDim.x;   // 32
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int e0 = b * HC2_EB;
    unsigned long long* ctr = (unsigned long long*) a.counter;
    __shared__ float sR[4][HC2_EB];
    __shared__ float sred[32];
    __shared__ float smix[HC_MIX + 1];
    __shared__ float s_pre[4];
    __shared__ float s_mixed[HC2_EB];
    // the weights first: warp w owns mix rows w, w + 8, w + 16; per row and stream segment d a lane loads the 4 bf16
    // values j = 4 * lane .. 4 * lane + 3 of [e0, e0 + 128)
    uint2 wv[3][4];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int m = warp + 8 * r;
        const uint16_t* wr = a.w_fn + (size_t) m * (hc * n_embd) + e0 + 4 * lane;
#pragma unroll
        for (int d = 0; d < 4; ++d) wv[r][d] = *(const uint2*) (wr + (size_t) d * n_embd);
    }
    // R_new for this block's 4 x 128 values (thread t: stream t / 64... two values per thread)
    float ss = 0.0f;
    for (int i = tid; i < 4 * HC2_EB; i += HC_THREADS) {
        const int d = i / HC2_EB, j = i - d * HC2_EB, e = e0 + j;
        float r;
        if (a.block_out != nullptr) {
            r = a.block_out[e] * a.post_in[d];
#pragma unroll
            for (int s = 0; s < hc; ++s) r += a.comb_in[d * hc + s] * a.R_old[s * n_embd + e];
            a.R_new[d * n_embd + e] = r;
        } else {
            r = a.R_old[d * n_embd + e];
        }
        sR[d][j] = r;
        ss += r * r;
    }
    __syncthreads();
    float* part = a.part;   // [m][block]: 25 rows of nb partials (the 25th: the sum of squares)
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int m = warp + 8 * r;
        float acc = 0.0f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const uint2 w = wv[r][d];
            const float* x = &sR[d][4 * lane];
            acc += __uint_as_float(w.x << 16) * x[0] + __uint_as_float(w.x & 0xffff0000u) * x[1] +
                   __uint_as_float(w.y << 16) * x[2] + __uint_as_float(w.y & 0xffff0000u) * x[3];
        }
        acc = warp_sum(acc);
        if (lane == 0) part[(size_t) m * nb + b] = acc;
    }
    ss = block_sum(ss, sred);
    if (tid == 0) part[(size_t) HC_MIX * nb + b] = ss;
    grid_barrier(ctr, nb);
    // the 25 totals: warp w reduces rows w, w + 8, w + 16 (and warp 0 row 24), one partial per lane
    for (int m = warp; m < HC_MIX + 1; m += 8) {
        const float v = lane < (int) nb ? __ldcg(&part[(size_t) m * nb + lane]) : 0.0f;
        const float t = warp_sum(v);
        if (lane == 0) smix[m] = t;
    }
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) (hc * n_embd) + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            s_pre[lane] = pre_v;
            if (b == 0) {
                a.pre[lane] = pre_v;
                a.post[lane] = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
            }
        }
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (b == 0 && lane < 16) a.comb[d * hc + s] = c;
    }
    __syncthreads();
    float ss2 = 0.0f;
    if (tid < HC2_EB) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * sR[s][tid];
        s_mixed[tid] = acc;
        ss2 = acc * acc;
    }
    ss2 = block_sum(ss2, sred);
    float* part2 = a.part + (size_t) (HC_MIX + 1) * nb;
    if (tid == 0) part2[b] = ss2;
    grid_barrier(ctr, nb);
    const float tot = warp_sum(lane < (int) nb ? __ldcg(&part2[lane]) : 0.0f);
    const float inv2 = rsqrtf(tot / (float) n_embd + a.norm_eps);
    if (tid < HC2_EB) {
        const int e = e0 + tid;
        const float xv = s_mixed[tid] * inv2 * a.norm_w[e];
        a.x[e] = xv;
        q8_1_store_warp(xv, (block_q8_1*) a.xq + e / 32, lane);
    }
}

__global__ void hc_post_kernel(const float* __restrict__ block_out, const float* __restrict__ R_old,
                               const float* __restrict__ post, const float* __restrict__ comb, int n_embd,
                               float* __restrict__ R_new) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 4 * n_embd) return;
    const int d = idx / n_embd, e = idx - d * n_embd;
    float acc = block_out[e] * post[d];
#pragma unroll
    for (int s = 0; s < 4; ++s) acc += comb[d * 4 + s] * R_old[s * n_embd + e];
    R_new[idx] = acc;
}

__global__ void head_prep_kernel(const float* __restrict__ R, const float* __restrict__ norm_w, float eps,
                                 int n_embd, float* __restrict__ x, block_q8_1* __restrict__ xq) {
    __shared__ float sred[32];
    __shared__ float sm[4096];
    const int tid = threadIdx.x, lane = tid & 31;
    float ss = 0.0f;
    for (int e = tid; e < n_embd; e += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < 4; ++s) acc += R[e + (size_t) n_embd * s];
        acc = acc / 4.0f;
        sm[e] = acc;
        ss += acc * acc;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) n_embd + eps);
    for (int e = tid; e < n_embd; e += blockDim.x) {
        const float v = sm[e] * inv * norm_w[e];
        x[e] = v;
        q8_1_store_warp(v, xq + e / 32, lane);
    }
}

__global__ void embed_streams_kernel(const float* __restrict__ emb, float* __restrict__ R, int n_embd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 4 * n_embd) return;
    R[idx] = emb[idx % n_embd];
}

// ---------------------------------------------------------------- KDA
// one block of kda_prep (head h, role) - the one-layer kernel's and the layers' (kda_layers) alike
__device__ __forceinline__ void kda_prep_block(const KdaPrepArgs& a, int h, int role) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int hd = a.head_dim;
    const int d_inner = a.n_head * hd;
    __shared__ float sred[32];
    __shared__ float sv[128];
    if (role < 3) {
        const int c = h * hd + tid;
        const int hist = a.d_conv - 1;
        float* st = a.conv_state + (size_t) role * d_inner * hist + (size_t) hist * c;
        const float* w = a.conv_w[role] + (size_t) a.d_conv * c;
        const float xp = a.proj[role][c];
        float acc = 0.0f;
        for (int tap = 0; tap < hist; ++tap) acc += w[tap] * st[tap];
        acc += w[hist] * xp;
        float y = acc / (1.0f + __expf(-acc));
        for (int j = 0; j + 1 < hist; ++j) st[j] = st[j + 1];
        st[hist - 1] = xp;
        if (role < 2) {
            const float ss = block_sum(y * y, sred);
            y = y * rsqrtf(ss + 1e-6f);
        }
        a.out[role][c] = y;
        return;
    }
    // roles 3..6: g1 = lb * sigmoid(-(f_b . fa + dt_bias) * ssm_a[h]) for a quarter of the head's rows each;
    // roles 7..10: g2 = g_b . ga, likewise (a quarter = 32 rows: 8 per warp)
    const bool is_g1 = role < 7;
    const int quarter = (role - 3) & 3;
    const float* in = is_g1 ? a.fa : a.ga;
    const uint16_t* W = is_g1 ? a.f_b : a.g_b;
    sv[tid] = in[tid];
    __syncthreads();
    const int rbase = quarter * (hd / 4) + warp * (hd / 16);
    for (int r = rbase; r < rbase + hd / 16; ++r) {
        const int c = h * hd + r;
        const uint16_t* wr = W + (size_t) c * hd;
        float acc = 0.0f;
        for (int j = lane; j < hd; j += 32) acc += bf(wr[j]) * sv[j];
        acc = warp_sum(acc);
        if (lane == 0) {
            if (is_g1) {
                const float v = acc + a.dt_bias[c];
                a.g1[c] = a.lower_bound * dsigmoid(-v * a.ssm_a[h]);
            } else {
                a.g2[c] = acc;
            }
        }
    }
}

__global__ void __launch_bounds__(128) kda_prep_kernel(const STRATA_GRID_CONSTANT KdaPrepArgs a) {
    kda_prep_block(a, blockIdx.x, blockIdx.y);
}
struct KdaPrepBatch {
    KdaPrepArgs a[kKdaLayers];
};
__global__ void __launch_bounds__(128) kda_prep_layers_kernel(const STRATA_GRID_CONSTANT KdaPrepBatch b) {
    kda_prep_block(b.a[blockIdx.z], blockIdx.x, blockIdx.y);
}

// one block (head h) of kda_rec; out == nullptr: the state only (a commit's replay)
__device__ __forceinline__ void kda_rec_block(const float* __restrict__ q, const float* __restrict__ k,
                                              const float* __restrict__ v, const float* __restrict__ g1,
                                              const float* __restrict__ beta_raw, float* __restrict__ state,
                                              const float* __restrict__ g2, const float* __restrict__ norm_w,
                                              float eps, block_q8_1* __restrict__ out, int h) {
    constexpr int HD = 128;
    const int t = threadIdx.x, j = t & (HD - 1), half = t >> 7, lane = t & 31;
    __shared__ float sq[HD], sk[HD], sdec[HD], sdel[HD];
    __shared__ float sacc[2][HD];
    __shared__ float sred[32];
    if (t < HD) {
        sq[t] = q[h * HD + t];
        sk[t] = k[h * HD + t];
        sdec[t] = __expf(g1[h * HD + t]);
    }
    const float b = 1.0f / (1.0f + expf(-beta_raw[h]));
    __syncthreads();
    float* Sh = state + (size_t) h * HD * HD;
    float reg[64];
    float A = 0.0f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s = Sh[i * HD + j] * sdec[i];
        reg[ii] = s;
        A += s * sk[i];
    }
    sacc[half][j] = A;
    __syncthreads();
    if (half == 0) sdel[j] = b * (v[h * HD + j] - (sacc[0][j] + sacc[1][j]));
    __syncthreads();
    const float dl = sdel[j];
    float O = 0.0f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s2 = reg[ii] + sk[i] * dl;
        Sh[i * HD + j] = s2;
        O += s2 * sq[i];
    }
    __syncthreads();
    if (out == nullptr) return;   // (uniform: the whole block)
    sacc[half][j] = O;
    __syncthreads();
    const float o = (sacc[0][j] + sacc[1][j]) * rsqrtf((float) HD);
    // out gate over the head (threads 0..127 own j; the upper half mirrors and does not write)
    float ss = half == 0 ? o * o : 0.0f;
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) HD + eps);
    if (half == 0) {
        const float gv = o * inv * norm_w[j] * dsigmoid(g2[h * HD + j]);
        q8_1_store_warp(gv, out + h * (HD / 32) + (j >> 5), lane);
    }
}

__global__ void __launch_bounds__(256) kda_rec_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                      const float* __restrict__ v, const float* __restrict__ g1,
                                                      const float* __restrict__ beta_raw, float* __restrict__ state,
                                                      const float* __restrict__ g2, const float* __restrict__ norm_w,
                                                      float eps, int n_head, block_q8_1* __restrict__ out) {
    kda_rec_block(q, k, v, g1, beta_raw, state, g2, norm_w, eps, out, blockIdx.x);
}
struct KdaRecBatch {
    KdaRecArgs a[kKdaLayers];
    float eps;
};
__global__ void __launch_bounds__(256) kda_rec_layers_kernel(const STRATA_GRID_CONSTANT KdaRecBatch b) {
    const KdaRecArgs& r = b.a[blockIdx.y];
    kda_rec_block(r.q, r.k, r.v, r.g1, r.beta_raw, r.state, r.g2, r.norm_w, b.eps, nullptr, blockIdx.x);
}

// the DSA latent cache is FP16 (uint16_t bits): read / written in F32
__device__ __forceinline__ float lat_f(uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ uint16_t lat_h(float v) { return __half_as_ushort(__float2half(v)); }
// the INT8 latent records (lat8_rec_bytes): value c of the record at rec
__device__ __forceinline__ float lat8_f(const uint8_t* rec, int kv_lora, int c) {
    return (float) ((const int8_t*) rec)[c] * __half2float(((const __half*) (rec + kv_lora))[c >> 5]);
}
// latent value c of cell: FP16 rows or INT8 records
__device__ __forceinline__ float lat_any(const uint16_t* lat, bool q8, int kv_lora, int cell, int c) {
    return q8 ? lat8_f((const uint8_t*) lat + (size_t) lat8_rec_bytes(kv_lora) * cell, kv_lora, c)
              : lat_f(lat[(size_t) kv_lora * cell + c]);
}
// one latent row into its INT8 record: lane l of the warp holding values [32 w, 32 w + 32) stores value c
__device__ __forceinline__ void lat8_store_warp(float y, uint8_t* rec, int kv_lora, int c, int lane) {
    float amax = fabsf(y);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const __half d16 = __float2half(amax / 127.0f);
    const float d = __half2float(d16);
    const int q = d > 0.0f ? max(-127, min(127, __float2int_rn(y / d))) : 0;
    ((int8_t*) rec)[c] = (int8_t) q;
    if (lane == 0) ((__half*) (rec + kv_lora))[c >> 5] = d16;
}
// value c of a latent row y into row `row` of a cache (FP16 rows, or INT8 records written by whole warps)
__device__ __forceinline__ void lat_put(uint16_t* lat, int64_t row, float y, bool q8, int kv_lora, int c) {
    if (q8)
        lat8_store_warp(y, (uint8_t*) lat + (size_t) lat8_rec_bytes(kv_lora) * row, kv_lora, c, c & 31);
    else
        lat[(size_t) kv_lora * row + c] = lat_h(y);
}
// ---------------------------------------------------------------- DSA
__global__ void __launch_bounds__(512) dsa_prep_kernel(const STRATA_GRID_CONSTANT DsaPrepArgs a) {
    __shared__ float sred[32];
    const int tid = threadIdx.x, lane = tid & 31;
    if (blockIdx.x == 0) {
        if (a.qr_raw == nullptr) return;   // the caches only (the one-GPU MTP decode's NextN cache fill)
        // q_a norm -> qr (f32) and qr_q (q8_1); q_lora <= 2048 (stride 512 keeps warps on 32-runs)
        float vals[4];
        float ss = 0.0f;
        int n = 0;
        for (int e = tid; e < a.q_lora; e += 512, ++n) {
            vals[n] = a.qr_raw[e];
            ss += vals[n] * vals[n];
        }
        ss = block_sum(ss, sred);
        const float inv = rsqrtf(ss / (float) a.q_lora + a.eps);
        n = 0;
        for (int e = tid; e < a.q_lora; e += 512, ++n) {
            const float y = vals[n] * inv * a.q_a_norm[e];
            a.qr[e] = y;
            q8_1_store_warp(y, (block_q8_1*) a.qr_q + e / 32, lane);
        }
        return;
    }
    if (blockIdx.x == 1) {
        float v = tid < a.kv_lora ? a.kv_raw[tid] : 0.0f;
        const float ss = block_sum(v * v, sred);
        const float inv = rsqrtf(ss / (float) a.kv_lora + a.eps);
        if (tid < a.kv_lora) {   // (kv_lora % 32 == 0: whole warps)
            const float y = v * inv * a.kv_norm[tid];
            if (a.lat != nullptr) lat_put(a.lat, a.p, y, a.lat_q8, a.kv_lora, tid);
            if (a.lat_host != nullptr) {   // KV streaming: the host copy, and the block's VRAM slot when it is resident
                lat_put(a.lat_host, a.p, y, a.lat_q8, a.kv_lora, tid);
                const int sl = a.lat_table[a.p / a.lat_page];
                if (sl >= 0) lat_put(a.lat_slots, (int64_t) sl * a.lat_page + a.p % a.lat_page, y, a.lat_q8, a.kv_lora, tid);
            }
        }
        return;
    }
    // block 2: the indexer key (layer norm) and the compressor gate into their caches; the pool
    const int K = a.idx_key;
    const float v = tid < K ? a.ik_raw[tid] : 0.0f;
    const float s1 = block_sum(v, sred);
    const float s2 = block_sum(v * v, sred);
    const float mu = s1 / (float) K;
    const float inv = rsqrtf(s2 / (float) K - mu * mu + a.eps);
    if (tid < K) {
        a.ik_cache[(size_t) K * (a.p % a.ring) + tid] = (v - mu) * inv * a.k_norm_w[tid] + a.k_norm_b[tid];
        a.ig_cache[(size_t) K * (a.p % a.ring) + tid] = a.ig_raw[tid];
    }
    if ((a.p + 1) % a.kpool != 0) return;
    __syncthreads();
    __threadfence_block();
    const int pi = (a.p + 1) / a.kpool - 1;
    if (tid < K) {
        float lg[16];
        float mx = -INFINITY;
        for (int m = 0; m < a.kpool; ++m) {
            lg[m] = a.ig_cache[(size_t) tid + (size_t) K * ((pi * a.kpool + m) % a.ring)] + a.ape[tid + K * m];
            mx = fmaxf(mx, lg[m]);
        }
        float den = 0.0f;
        for (int m = 0; m < a.kpool; ++m) {
            lg[m] = expf(lg[m] - mx);
            den += lg[m];
        }
        float acc = 0.0f;
        for (int m = 0; m < a.kpool; ++m)
            acc += (lg[m] / den) * a.ik_cache[(size_t) tid + (size_t) K * ((pi * a.kpool + m) % a.ring)];
        a.pooled[(size_t) tid + (size_t) K * pi] = acc;
    }
}

__global__ void __launch_bounds__(256) dsa_score_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                        const float* __restrict__ iw, int key_dim, int idx_heads,
                                                        int n_pools, float* __restrict__ score) {
    extern __shared__ float s_iq[];   // idx_heads * key_dim, then idx_heads weights
    float* s_iw = s_iq + idx_heads * key_dim;
    for (int i = threadIdx.x; i < idx_heads * key_dim; i += blockDim.x) s_iq[i] = iq[i];
    for (int i = threadIdx.x; i < idx_heads; i += blockDim.x) s_iw[i] = iw[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int p = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (p >= n_pools) return;
    const float* pk = pooled + (size_t) key_dim * p;
    float acc = 0.0f;
    for (int h = 0; h < idx_heads; ++h) {
        float dot = 0.0f;
        for (int e = lane; e < key_dim; e += 32) dot += s_iq[h * key_dim + e] * pk[e];
        dot = warp_sum(dot);
        acc += fmaxf(dot, 0.0f) * s_iw[h];
    }
    if (lane == 0) score[p] = acc;
}

__global__ void __launch_bounds__(1024) dsa_select_kernel(const float* __restrict__ score, int n_vis, int kpool,
                                                          int top_pools, int n_sel, int pos, int* __restrict__ cells) {
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) cells[i] = -1;
    __syncthreads();
    // the top pools in rank order (score descending, ties by the lower index), O(n_vis) - src/kernels/cuda/dsa_topk.cuh
    if (n_vis > 0) strata::kernels::dsa::select_top_pools(score, n_vis, min(top_pools, n_vis), kpool, cells);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int m = 0; m < kpool - 1; ++m) {
            const int cell = n_vis * kpool + m;
            if (cell <= pos) cells[top_pools * kpool + m] = cell;
        }
    }
}

// One block per head: q_abs = wk_b_h . q_h (kv_lora rows of qk_nope), scores over the selected latents,
// softmax, ctx, out = wv_b_h . ctx (v_head rows of kv_lora), q8_1 of out.
__global__ void __launch_bounds__(256) mla_kernel(const float* __restrict__ q, const uint16_t* __restrict__ wk_b,
                                                  const uint16_t* __restrict__ wv_b, const uint16_t* __restrict__ lat,
                                                  const int* __restrict__ cells, int n_sel, int qk_nope, int kv_lora,
                                                  int v_head, block_q8_1* __restrict__ out, bool q8) {
    extern __shared__ float smem[];
    float* s_q = smem;                    // qk_nope
    float* s_qa = s_q + qk_nope;          // kv_lora
    float* s_ctx = s_qa + kv_lora;        // kv_lora
    float* s_out = s_ctx + kv_lora;       // v_head
    float* s_p = s_out + v_head;          // n_sel
    __shared__ float sred[32];
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int i = tid; i < qk_nope; i += blockDim.x) s_q[i] = q[(size_t) qk_nope * h + i];
    __syncthreads();
    const uint16_t* wk = wk_b + (size_t) qk_nope * kv_lora * h;
    for (int c = warp; c < kv_lora; c += nw) {
        const uint16_t* wr = wk + (size_t) qk_nope * c;
        float acc = 0.0f;
        for (int e = lane; e < qk_nope; e += 32) acc += bf(wr[e]) * s_q[e];
        acc = warp_sum(acc);
        if (lane == 0) s_qa[c] = acc;
    }
    __syncthreads();
    const float scale = rsqrtf((float) qk_nope);
    for (int s = warp; s < n_sel; s += nw) {
        const int cell = cells[s];
        float dot = -INFINITY;
        if (cell >= 0) {
            float acc = 0.0f;
            for (int e = lane; e < kv_lora; e += 32) acc += s_qa[e] * lat_any(lat, q8, kv_lora, cell, e);
            dot = warp_sum(acc) * scale;
        }
        if (lane == 0) s_p[s] = dot;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int s = tid; s < n_sel; s += blockDim.x) mx = fmaxf(mx, s_p[s]);
    mx = block_max(mx, sred);
    float den = 0.0f;
    for (int s = tid; s < n_sel; s += blockDim.x) {
        const float e = s_p[s] == -INFINITY ? 0.0f : expf(s_p[s] - mx);
        s_p[s] = e;
        den += e;
    }
    den = block_sum(den, sred);
    const float invd = 1.0f / den;
    for (int c = tid; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const int cell = cells[s];
            if (cell >= 0) acc += (s_p[s] * invd) * lat_any(lat, q8, kv_lora, cell, c);
        }
        s_ctx[c] = acc;
    }
    __syncthreads();
    const uint16_t* wv = wv_b + (size_t) kv_lora * v_head * h;
    for (int vv = warp; vv < v_head; vv += nw) {
        const uint16_t* wr = wv + (size_t) kv_lora * vv;
        float acc = 0.0f;
        for (int c = lane; c < kv_lora; c += 32) acc += bf(wr[c]) * s_ctx[c];
        acc = warp_sum(acc);
        if (lane == 0) s_out[vv] = acc;
    }
    __syncthreads();
    for (int vv = tid; vv < v_head; vv += blockDim.x)   // warp-aligned runs of 32
        q8_1_store_warp(s_out[vv], out + ((size_t) v_head * h + vv) / 32, lane);
}

// ---- MLA, take two: the absorbed projections as head-wise GEMVs over the whole GPU (warp per row, 32 rows per
// block - 4 per warp) and the attention proper in between
// out[h * R + r] = W[h][r][:] . in[h * K : h * K + K] (W BF16, rows of K); q8 != nullptr also writes the q8_1
__global__ void __launch_bounds__(256) headwise_gemv_kernel(const uint16_t* __restrict__ W, const float* __restrict__ in,
                                                           int R, int K, float* __restrict__ out,
                                                           block_q8_1* __restrict__ q8) {
    __shared__ float s_o[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = blockIdx.x * 32;   // R % 32 == 0: a block never straddles heads
    const int h = row0 / R;
    const float* x = in + (size_t) h * K;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int row = row0 + warp * 4 + j;
        const float s = row_dot_bf16(W + (size_t) row * K, x, K, lane);
        if (lane == 0) s_o[warp * 4 + j] = s;
    }
    __syncthreads();
    if (warp == 0) {
        const float v = s_o[lane];
        if (out) out[row0 + lane] = v;
        if (q8) q8_1_store_warp(v, q8 + row0 / 32, lane);
    }
}

// ... the same for ncol columns at once (a verify window's tokens): column c reads in + c * in_s and writes out + c *
// out_s / q8 + c * q8_s blocks; each row's BF16 weights are loaded once for every column, and each (row, column) sum is
// row_dot_bf16's arithmetic, bit for bit
__global__ void __launch_bounds__(256) headwise_gemv_cols_kernel(const uint16_t* __restrict__ W, const float* __restrict__ in,
                                                                int64_t in_s, int R, int K, int ncol,
                                                                float* __restrict__ out, int64_t out_s,
                                                                block_q8_1* __restrict__ q8, int64_t q8_s) {
    __shared__ float s_o[kMaxMvCols][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = blockIdx.x * 32;   // R % 32 == 0: a block never straddles heads
    const int h = row0 / R;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int row = row0 + warp * 4 + j;
        const uint4* r8 = (const uint4*) (W + (size_t) row * K);
        float s[kMaxMvCols];
#pragma unroll
        for (int c = 0; c < kMaxMvCols; ++c) s[c] = 0.0f;
        for (int k = lane; k < K / 8; k += 32) {
            const uint4 w = r8[k];
#pragma unroll
            for (int c = 0; c < kMaxMvCols; ++c)
                if (c < ncol) {
                    const float4* x4 = (const float4*) (in + c * in_s + (size_t) h * K);
                    const float4 a = x4[2 * k], b = x4[2 * k + 1];
                    s[c] += __uint_as_float(w.x << 16) * a.x + __uint_as_float(w.x & 0xffff0000u) * a.y +
                            __uint_as_float(w.y << 16) * a.z + __uint_as_float(w.y & 0xffff0000u) * a.w +
                            __uint_as_float(w.z << 16) * b.x + __uint_as_float(w.z & 0xffff0000u) * b.y +
                            __uint_as_float(w.w << 16) * b.z + __uint_as_float(w.w & 0xffff0000u) * b.w;
                }
        }
#pragma unroll
        for (int c = 0; c < kMaxMvCols; ++c)
            if (c < ncol) {
                const float v = warp_sum(s[c]);
                if (lane == 0) s_o[c][warp * 4 + j] = v;
            }
    }
    __syncthreads();
    if (warp == 0)
        for (int c = 0; c < ncol; ++c) {
            const float v = s_o[c][lane];
            if (out) out[c * out_s + row0 + lane] = v;
            if (q8) q8_1_store_warp(v, q8 + c * q8_s + row0 / 32, lane);
        }
}

__global__ void __launch_bounds__(256) mla_attn_kernel(const float* __restrict__ q_abs, const uint16_t* __restrict__ lat,
                                                       const int* __restrict__ cells, int n_sel, int qk_nope,
                                                       int kv_lora, float* __restrict__ ctx, bool q8) {
    extern __shared__ float smem[];
    float* s_qa = smem;               // kv_lora
    float* s_p = s_qa + kv_lora;      // n_sel
    __shared__ float sred[32];
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int i = tid; i < kv_lora; i += blockDim.x) s_qa[i] = q_abs[(size_t) kv_lora * h + i];
    __syncthreads();
    const float scale = rsqrtf((float) qk_nope);
    for (int s = warp; s < n_sel; s += nw) {
        const int cell = cells[s];
        float dot = -INFINITY;
        if (cell >= 0) {
            float acc = 0.0f;
            for (int e = lane; e < kv_lora; e += 32) acc += s_qa[e] * lat_any(lat, q8, kv_lora, cell, e);
            dot = warp_sum(acc) * scale;
        }
        if (lane == 0) s_p[s] = dot;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int s = tid; s < n_sel; s += blockDim.x) mx = fmaxf(mx, s_p[s]);
    mx = block_max(mx, sred);
    float den = 0.0f;
    for (int s = tid; s < n_sel; s += blockDim.x) {
        const float e = s_p[s] == -INFINITY ? 0.0f : expf(s_p[s] - mx);
        s_p[s] = e;
        den += e;
    }
    den = block_sum(den, sred);
    const float invd = 1.0f / den;
    for (int c = tid; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const int cell = cells[s];
            if (cell >= 0) acc += (s_p[s] * invd) * lat_any(lat, q8, kv_lora, cell, c);
        }
        ctx[(size_t) kv_lora * h + c] = acc;
    }
}

// ---- MLA attention, split over the selected cells (flash-decoding): every head reads the SAME latents, so a block
// loads a chunk of MLA_CHUNK latent rows into shared memory once and scores it against ALL heads; per head it keeps
// its chunk's max, sum of exponentials and exp-weighted context, and the combine kernel merges the chunks.  At 2k
// selected cells this reads the latents once per layer instead of once per head (64x less).
// The grid is (chunks, head groups of MLA_HPB): ~20 chunks at 600 cells would leave most SMs idle with every head in
// one block (208 us a call on a V100); a group re-reads its chunk's latents, which is cheap next to the arithmetic.
// The latents stay FP16 in shared memory, as in the cache (34 KB at kv_lora 512; F32 rows were over Turing's 64 KB).
constexpr int MLA_CHUNK = 32;
constexpr int MLA_HPB = 16;
__global__ void __launch_bounds__(256) mla_split_kernel(const float* __restrict__ q_abs, const uint16_t* __restrict__ lat,
                                                        const int* __restrict__ cells, int n_sel, int n_head, int qk_nope,
                                                        int kv_lora, float* __restrict__ part, bool q8) {
    extern __shared__ float smem[];
    uint16_t* sL = (uint16_t*) smem;                   // MLA_CHUNK x kv_lora, FP16
    float* sS = smem + MLA_CHUNK * kv_lora / 2;        // MLA_HPB x MLA_CHUNK scores -> weights
    __shared__ int s_cell[MLA_CHUNK];
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const int h0 = blockIdx.y * MLA_HPB, nh = min(MLA_HPB, n_head - h0);
    const int s0 = b * MLA_CHUNK;
    const int ns = min(MLA_CHUNK, n_sel - s0);
    if (tid < MLA_CHUNK) s_cell[tid] = tid < ns ? cells[s0 + tid] : -1;
    __syncthreads();
    for (int i = tid; i < MLA_CHUNK * kv_lora; i += blockDim.x) {
        const int s = i / kv_lora, c = i - s * kv_lora;
        const int cell = s_cell[s];
        sL[i] = cell < 0 ? (uint16_t) 0 : q8 ? lat_h(lat_any(lat, true, kv_lora, cell, c)) : lat[(size_t) kv_lora * cell + c];
    }
    __syncthreads();
    const float scale = rsqrtf((float) qk_nope);
    // scores: a warp per head, the head's q in registers (kv_lora / 32 per lane), one warp reduction per cell
    for (int hh = warp; hh < nh; hh += nw) {
        float q[16];
        const float* qh = q_abs + (size_t) kv_lora * (h0 + hh);
#pragma unroll
        for (int j = 0; j < 16; ++j) q[j] = (lane + 32 * j) < kv_lora ? qh[lane + 32 * j] : 0.0f;
        for (int s = 0; s < MLA_CHUNK; ++s) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + 32 * j < kv_lora) acc += q[j] * lat_f(sL[s * kv_lora + lane + 32 * j]);
            acc = warp_sum(acc);
            if (lane == 0) sS[hh * MLA_CHUNK + s] = s_cell[s] >= 0 ? acc * scale : -INFINITY;
        }
    }
    __syncthreads();
    // per head: the chunk's max and sum; the scores become exp-weights
    float* pm = part + ((size_t) b * n_head + h0) * (kv_lora + 2);
    for (int hh = warp; hh < nh; hh += nw) {
        const float v = lane < MLA_CHUNK ? sS[hh * MLA_CHUNK + lane] : -INFINITY;
        const float m = warp_max(v);
        const float e = (v == -INFINITY) ? 0.0f : expf(v - m);
        const float l = warp_sum(e);
        if (lane < MLA_CHUNK) sS[hh * MLA_CHUNK + lane] = e;
        if (lane == 0) {
            pm[(size_t) hh * (kv_lora + 2) + kv_lora] = m;
            pm[(size_t) hh * (kv_lora + 2) + kv_lora + 1] = l;
        }
    }
    __syncthreads();
    // exp-weighted context per head over this chunk
    for (int i = tid; i < nh * kv_lora; i += blockDim.x) {
        const int hh = i / kv_lora, c = i - hh * kv_lora;
        float acc = 0.0f;
#pragma unroll 8
        for (int s = 0; s < MLA_CHUNK; ++s) acc += sS[hh * MLA_CHUNK + s] * lat_f(sL[s * kv_lora + c]);
        pm[(size_t) hh * (kv_lora + 2) + c] = acc;
    }
}

__global__ void mla_combine_kernel(const float* __restrict__ part, int n_chunks, int n_head, int kv_lora,
                                   float* __restrict__ ctx) {
    const int h = blockIdx.x;
    float M = -INFINITY;
    for (int b = 0; b < n_chunks; ++b) M = fmaxf(M, part[((size_t) b * n_head + h) * (kv_lora + 2) + kv_lora]);
    float L = 0.0f;
    for (int b = 0; b < n_chunks; ++b) {
        const float* pb = part + ((size_t) b * n_head + h) * (kv_lora + 2);
        if (pb[kv_lora] != -INFINITY) L += expf(pb[kv_lora] - M) * pb[kv_lora + 1];
    }
    const float invL = L > 0.0f ? 1.0f / L : 0.0f;
    for (int c = threadIdx.x; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (int b = 0; b < n_chunks; ++b) {
            const float* pb = part + ((size_t) b * n_head + h) * (kv_lora + 2);
            if (pb[kv_lora] != -INFINITY) acc += expf(pb[kv_lora] - M) * pb[c];
        }
        ctx[(size_t) kv_lora * h + c] = acc * invL;
    }
}

// mla_split_kernel's chunk on tensor cores (NVIDIA sm_80+, kv_lora 512): the scores are a 16-head x 32-cell product
// over 512 latent values and the context a 16 x 512 product over the 32 cells, so FP16 mma.sync (F32 sums) takes both
// instead of one shared-memory read per multiply-add.  q (with the score scale folded in) and the exp-weights enter as
// FP16 high + low parts, two mma each, so the products keep F32's precision; the latents are FP16 already.
// The 4 warps split the scores' 512 values (their partial scores summed in a fixed order) and the context's columns.
constexpr int MLA_KV = 512;
constexpr int MLA_ROW = MLA_KV + 8;   // halves per shared latent row: 16 bytes of pad put ldmatrix's 8 rows on distinct banks
constexpr int MLA_PROW = MLA_CHUNK + 8;
constexpr size_t MLA_MMA_SMEM = (size_t) MLA_CHUNK * MLA_ROW * 2 + (size_t) 4 * 16 * MLA_CHUNK * 4 + (size_t) 2 * 16 * MLA_PROW * 2;
__device__ __forceinline__ void mla_h2(float a, float b, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(a, b);
    const float2 r = __half22float2(h);
    const __half2 l = __floats2half2_rn(a - r.x, b - r.y);
    hi = *(const uint32_t*) &h;
    lo = *(const uint32_t*) &l;
}
// a token's cells per block: at most MLA_NC blocks per head group, whole MLA_CHUNKs - a function of its cell count
// alone, so a token's sums are the same whichever tokens share its launch
constexpr int MLA_NC = 16;
__host__ __device__ __forceinline__ int mla_cblk(int n_sel) {
    return max(MLA_CHUNK, ((n_sel + MLA_NC - 1) / MLA_NC + MLA_CHUNK - 1) / MLA_CHUNK * MLA_CHUNK);
}
// the tokens of one launch (blockIdx.z): each one's selected cell count
struct MlaToks {
    int n_sel[kMlaTokMax];
};
// a block: 16 heads of one token over its cells [blockIdx.x * cblk, + cblk), MLA_CHUNK at a time with a running max
// and sum per head (the context rows rescaled when the max grows), so a block's partial covers cblk cells
__global__ void __launch_bounds__(128) mla_split_mma_kernel(const float* __restrict__ q_abs, int64_t q_stride,
                                                            const uint16_t* __restrict__ lat,
                                                            const int* __restrict__ cells, int64_t c_stride, MlaToks nt,
                                                            int n_head, int qk_nope, float* __restrict__ part, bool q8) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(STRATA_USE_HIP)
    extern __shared__ uint4 smem4[];
    uint16_t* sL = (uint16_t*) smem4;                       // MLA_CHUNK x MLA_ROW
    float* sS = (float*) (sL + MLA_CHUNK * MLA_ROW);        // per warp: 16 heads x MLA_CHUNK partial scores
    uint16_t* sP = (uint16_t*) (sS + 4 * 16 * MLA_CHUNK);   // exp-weights' high / low parts: 2 x 16 x MLA_PROW
    __shared__ int s_cell[MLA_CHUNK];
    __shared__ float s_m[16], s_l[16], s_a[16];
    const int tk = blockIdx.z, n_sel = nt.n_sel[tk], cblk = mla_cblk(n_sel);
    const int c_begin = blockIdx.x * cblk;
    if (c_begin >= n_sel) return;
    const int c_end = min(n_sel, c_begin + cblk);
    q_abs += tk * q_stride;
    cells += tk * c_stride;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int h0 = blockIdx.y * 16, nh = min(16, n_head - h0);
    float* pm = part + (((size_t) tk * MLA_NC + blockIdx.x) * n_head + h0) * (MLA_KV + 2);
    if (tid < 16) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    const float scale = rsqrtf((float) qk_nope);
    const uint32_t sL_a = (uint32_t) __cvta_generic_to_shared(sL);
    const int m = lane >> 3, r = lane & 7;
    float ctx[8][2][4] = {};   // this warp's 128 context columns of the 16 heads
    for (int sc = c_begin; sc < c_end; sc += MLA_CHUNK) {
        if (tid < MLA_CHUNK) s_cell[tid] = sc + tid < c_end ? cells[sc + tid] : -1;
        __syncthreads();
        if (!q8) {
            uint4 v[MLA_CHUNK * MLA_KV / 8 / 128];
#pragma unroll
            for (int j = 0; j < MLA_CHUNK * MLA_KV / 8 / 128; ++j) {
                const int i = tid + 128 * j, s = i / (MLA_KV / 8), c8 = i % (MLA_KV / 8);
                const int cell = s_cell[s];
                v[j] = cell < 0 ? make_uint4(0, 0, 0, 0) : ((const uint4*) (lat + (size_t) MLA_KV * cell))[c8];
            }
#pragma unroll
            for (int j = 0; j < MLA_CHUNK * MLA_KV / 8 / 128; ++j) {
                const int i = tid + 128 * j, s = i / (MLA_KV / 8), c8 = i % (MLA_KV / 8);
                *(uint4*) (sL + s * MLA_ROW + 8 * c8) = v[j];
            }
        } else {
            for (int i = tid; i < MLA_CHUNK * MLA_KV; i += 128) {
                const int s = i / MLA_KV, c = i - s * MLA_KV, cell = s_cell[s];
                sL[s * MLA_ROW + c] = cell < 0 ? (uint16_t) 0 : lat_h(lat_any(lat, true, MLA_KV, cell, c));
            }
        }
        __syncthreads();
        {   // scores over this warp's 128 latent values
            float acc[4][4] = {};
            const float* q0 = q_abs + (size_t) MLA_KV * (h0 + g) + 128 * warp + 2 * t;
            const float* q1 = q0 + (size_t) MLA_KV * 8;
            const bool r0 = g < nh, r1 = g + 8 < nh;
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int k0 = 128 * warp + 16 * ks;
                const float2 x[4] = {r0 ? *(const float2*) (q0 + 16 * ks) : make_float2(0.f, 0.f),
                                     r1 ? *(const float2*) (q1 + 16 * ks) : make_float2(0.f, 0.f),
                                     r0 ? *(const float2*) (q0 + 16 * ks + 8) : make_float2(0.f, 0.f),
                                     r1 ? *(const float2*) (q1 + 16 * ks + 8) : make_float2(0.f, 0.f)};
                uint32_t ah[4], al[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) mla_h2(x[i].x * scale, x[i].y * scale, ah[i], al[i]);
#pragma unroll
                for (int np = 0; np < 2; ++np) {   // two cell tiles of 8 per ldmatrix.x4
                    uint32_t b[4];
                    const uint32_t addr =
                        sL_a + 2u * (uint32_t) ((16 * np + (m >> 1) * 8 + r) * MLA_ROW + k0 + (m & 1) * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float* c = acc[2 * np + h];
#pragma unroll
                        for (int lo = 0; lo < 2; ++lo) {
                            const uint32_t* a = lo ? al : ah;
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                                         "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2 * h]), "r"(b[2 * h + 1]));
                        }
                    }
                }
            }
            float* sw = sS + warp * 16 * MLA_CHUNK;
#pragma unroll
            for (int n8 = 0; n8 < 4; ++n8) {
                *(float2*) (sw + g * MLA_CHUNK + 8 * n8 + 2 * t) = make_float2(acc[n8][0], acc[n8][1]);
                *(float2*) (sw + (g + 8) * MLA_CHUNK + 8 * n8 + 2 * t) = make_float2(acc[n8][2], acc[n8][3]);
            }
        }
        __syncthreads();
        // per head: the running max and sum, the rescale of the context so far, the exp-weights' FP16 parts
        for (int hh = warp; hh < 16; hh += 4) {
            const float sc4 = sS[hh * MLA_CHUNK + lane] + sS[(16 + hh) * MLA_CHUNK + lane] +
                              sS[(32 + hh) * MLA_CHUNK + lane] + sS[(48 + hh) * MLA_CHUNK + lane];
            const float v = s_cell[lane] >= 0 && hh < nh ? sc4 : -INFINITY;
            const float m_old = s_m[hh], m_new = fmaxf(m_old, warp_max(v));
            const float alpha = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
            const float e = (v == -INFINITY) ? 0.0f : expf(v - m_new);
            const float l = warp_sum(e);
            const __half eh = __float2half_rn(e);
            sP[hh * MLA_PROW + lane] = __half_as_ushort(eh);
            sP[(16 + hh) * MLA_PROW + lane] = __half_as_ushort(__float2half_rn(e - __half2float(eh)));
            __syncwarp();
            if (lane == 0) {
                s_m[hh] = m_new;
                s_l[hh] = s_l[hh] * alpha + l;
                s_a[hh] = alpha;
            }
        }
        __syncthreads();
        {   // the exp-weighted context over this warp's 128 columns
            uint32_t ah[2][4], al[2][4];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const uint16_t* P = sP + 16 * q * MLA_PROW;
                    uint32_t* a = q ? al[ks] : ah[ks];
                    a[0] = *(const uint32_t*) (P + g * MLA_PROW + 16 * ks + 2 * t);
                    a[1] = *(const uint32_t*) (P + (g + 8) * MLA_PROW + 16 * ks + 2 * t);
                    a[2] = *(const uint32_t*) (P + g * MLA_PROW + 16 * ks + 8 + 2 * t);
                    a[3] = *(const uint32_t*) (P + (g + 8) * MLA_PROW + 16 * ks + 8 + 2 * t);
                }
            }
            const float a0 = s_a[g], a1 = s_a[g + 8];
#pragma unroll
            for (int np = 0; np < 8; ++np) {   // two column tiles of 8 per ldmatrix.x4.trans
                const int c0 = 128 * warp + 16 * np;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    ctx[np][h][0] *= a0;
                    ctx[np][h][1] *= a0;
                    ctx[np][h][2] *= a1;
                    ctx[np][h][3] *= a1;
                }
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    uint32_t b[4];
                    const uint32_t addr =
                        sL_a + 2u * (uint32_t) ((16 * ks + (m & 1) * 8 + r) * MLA_ROW + c0 + (m >> 1) * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            const uint32_t* a = q ? al[ks] : ah[ks];
                            float* c = ctx[np][h];
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                                         "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2 * h]), "r"(b[2 * h + 1]));
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int np = 0; np < 8; ++np) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = 128 * warp + 16 * np + 8 * h + 2 * t;
            if (g < nh) *(float2*) (pm + (size_t) g * (MLA_KV + 2) + c) = make_float2(ctx[np][h][0], ctx[np][h][1]);
            if (g + 8 < nh)
                *(float2*) (pm + (size_t) (g + 8) * (MLA_KV + 2) + c) = make_float2(ctx[np][h][2], ctx[np][h][3]);
        }
    }
    if (tid < nh) {
        pm[(size_t) tid * (MLA_KV + 2) + MLA_KV] = s_m[tid];
        pm[(size_t) tid * (MLA_KV + 2) + MLA_KV + 1] = s_l[tid];
    }
#endif
}

// the merge of mla_split_mma_kernel's partials (MLA_NC slots a token): a block per (head, 128 columns, token), the
// chunks' weights computed once into shared memory, then one coalesced pass over the chunks' partial rows
__global__ void __launch_bounds__(128) mla_combine_wide_kernel(const float* __restrict__ part, MlaToks nt, int n_head,
                                                               int kv_lora, float* __restrict__ ctx, int64_t ctx_stride) {
    __shared__ float s_w[MLA_NC];
    __shared__ float s_red[4];
    const int h = blockIdx.x, tk = blockIdx.z, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n_chunks = (nt.n_sel[tk] + mla_cblk(nt.n_sel[tk]) - 1) / mla_cblk(nt.n_sel[tk]);
    const size_t st = (size_t) n_head * (kv_lora + 2);
    const float* ph = part + (size_t) tk * MLA_NC * st + (size_t) h * (kv_lora + 2);
    float M = -INFINITY;
    for (int b = tid; b < n_chunks; b += 128) M = fmaxf(M, ph[b * st + kv_lora]);
    M = warp_max(M);
    if (lane == 0) s_red[warp] = M;
    __syncthreads();
    M = fmaxf(fmaxf(s_red[0], s_red[1]), fmaxf(s_red[2], s_red[3]));
    __syncthreads();
    float L = 0.0f;
    for (int b = tid; b < n_chunks; b += 128) {
        const float mb = ph[b * st + kv_lora];
        const float w = mb == -INFINITY ? 0.0f : expf(mb - M);
        s_w[b] = w;
        L += w * ph[b * st + kv_lora + 1];
    }
    L = warp_sum(L);
    if (lane == 0) s_red[warp] = L;
    __syncthreads();
    L = (s_red[0] + s_red[1]) + (s_red[2] + s_red[3]);
    const float invL = L > 0.0f ? 1.0f / L : 0.0f;
    const int c = blockIdx.y * 128 + tid;
    if (c >= kv_lora) return;
    float acc = 0.0f;
#pragma unroll 8
    for (int b = 0; b < n_chunks; ++b) acc += s_w[b] * ph[b * st + c];
    ctx[tk * ctx_stride + (size_t) kv_lora * h + c] = acc * invL;
}

__global__ void swiglu_q8_kernel(const float* __restrict__ gate, const float* __restrict__ up, float limit, int n,
                                 block_q8_1* __restrict__ hq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // n % 32 == 0
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    q8_1_store_warp((g / (1.0f + expf(-g))) * u, hq + i / 32, threadIdx.x & 31);
}

// ---------------------------------------------------------------- routed experts
constexpr int ROUTE_THREADS = 320;

// the k largest of sel[0..E) in descending order, ties by the lower index - exactly the stable-descending rank rule
// (rank(e) = #{f: sel f > sel e} + #{f < e: sel f == sel e}) - as k rounds of a block argmax; sel is consumed
// (the picked entries become -inf).  All threads of the block call it.
__device__ void topk_argmax(float* sel, int E, int k, int* out, float* s_bv, int* s_bi) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int e = tid; e < E; e += blockDim.x) {
            const float v = sel[e];
            if (v > bv || (v == bv && e < bi)) {
                bv = v;
                bi = e;
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
        if (lane == 0) {
            s_bv[warp] = bv;
            s_bi[warp] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float v = s_bv[0];
            int i = s_bi[0];
            for (int w = 1; w < nw; ++w)
                if (s_bv[w] > v || (s_bv[w] == v && s_bi[w] < i)) {
                    v = s_bv[w];
                    i = s_bi[w];
                }
            out[r] = i >= 0 && i < E && isfinite(v) ? i : -1;
            if (out[r] >= 0) sel[i] = -INFINITY;
        }
        __syncthreads();
    }
}

// topk_argmax's picks (the largest first, ties to the lower id, a non-finite pick -1; the picks left at -inf in sel)
// for E <= 512, by warp 0 alone: its 16 values a lane in registers and k rounds of a warp arg-max, no block barrier -
// topk_argmax's k rounds of two barriers each were half of a 20 us route on an RTX 5060 Ti.  The block waits for it
__device__ void topk_warp(float* sel, int E, int k, int* out) {
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int e = lane + 32 * j;
            v[j] = e < E ? sel[e] : -INFINITY;
        }
        for (int r = 0; r < k; ++r) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int e = lane + 32 * j;
                if (e < E && (v[j] > bv || (v[j] == bv && e < bi))) {
                    bv = v[j];
                    bi = e;
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
            const bool ok = bi >= 0 && bi < E && isfinite(bv);
            if (ok) {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    if (lane + 32 * j == bi) v[j] = -INFINITY;
            }
            if (lane == 0) out[r] = ok ? bi : -1;
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int e = lane + 32 * j;
            if (e < E) sel[e] = v[j];
        }
    }
    __syncthreads();
}

struct RouteArgs {
    const float* logits;
    const float* bias;
    int n_expert, k;
    float w_scale;
    int norm_w;
    int layer;
    const float* x;
    int n_embd;
    MoeDev d;
    const float* sh_gate;
    const float* sh_up;
    float sh_limit;
    int n_ff_sh;
    block_q8_1* sh_hq;
    const float* pred_logits;   // the next layer's router on this layer's input (nullptr: no prediction)
    const float* pred_bias;
    int max_pf;                 // prefetch at most this many predicted non-resident experts of the next layer
    const float* ahead_logits;  // n_ahead x n_expert: the next layers' routers on this layer's input (LOOKAHEAD)
    const float* ahead_bias[kAhead];
    int n_ahead;
    int skip_from;              // experts not in VRAM at route rank >= skip_from are left out (no fetch, no wait):
                                // the draft block, STRATA_GLM_TRUNK_KEEP (0: all of them; >= k: none)
    unsigned long long cpu_plan;   // the CPU LANE: of f RAM-tier experts, (plan >> 4 f) & 15 go to the host (0: off)
    int promote_min;            // STRATA_GLM_PROMOTE_MIN: keep a fetched expert only when its aged route count
                                // clears this; 0 keeps the old rule (a spare, if one is free)
    int pf_rank;                // STRATA_GLM_PREFETCH_RANK: prefetch only the prediction's first pf_rank ranks
    int near_n;                 // STRATA_GLM_ROUTE_LOG: the route's next near_n ranks after the top k (0: none)
    int win;                    // a verify window's token: the CPU lane's answer slot (0: the lone one)
    unsigned int* scr_used;     // a verify window's layer: the scratch slots its tokens took (bit 31: one was taken
                                // twice), so the batched experts read each token's own (null: entry i's slot)
    const int* skip;            // set_skip's flag: non-zero = a token past a wrong draft (see moe_route_kernel)
    int win_n;                  // a batched verify window's tokens (0: not one) - the host spreads their CPU lanes
    int disk_skip_from;         // STRATA_GLM_DISK_SKIP_FROM: experts in neither VRAM nor the RAM tier at route rank
                                // >= this are left out (no disk read); >= k: none
    float disk_skip_w;          // STRATA_GLM_DISK_SKIP_W: ... and, at any rank, those whose share of the route's
                                // weights is below this (0: none)
    float miss_skip_w;          // STRATA_GLM_MISS_SKIP_W (prototype): experts not in VRAM whose share of the route's
                                // weights is below this are left out like skip_from's, at any rank (0: none)
    float affinity;             // STRATA_GLM_ROUTE_AFFINITY: added to a VRAM-resident expert's selection score (its
                                // weight stays its own probability) - LOSSY, it changes which experts run (0: off)
};

__global__ void __launch_bounds__(ROUTE_THREADS) moe_route_kernel(const STRATA_GRID_CONSTANT RouteArgs a) {
    const int tid = threadIdx.x, lane = tid & 31;
    // a token past a wrong draft (set_skip): every expert not in VRAM is left out - no fetch, no CPU work, no counts
    const bool skipped = a.skip != nullptr && *a.skip;
    const int skip_from = skipped ? 0 : a.skip_from;
    if (blockIdx.x > 0) {
        if (skipped) return;
        // the shared expert's swiglu + q8_1 (warp-aligned 32-runs)
        const int i = (blockIdx.x - 1) * ROUTE_THREADS + tid;
        if (i >= a.n_ff_sh) return;
        const float g = fminf(a.sh_gate[i], a.sh_limit);
        const float u = fminf(fmaxf(a.sh_up[i], -a.sh_limit), a.sh_limit);
        q8_1_store_warp((g / (1.0f + expf(-g))) * u, a.sh_hq + i / 32, lane);
        return;
    }
    const int E = a.n_expert;
    __shared__ float s_p[512], s_sel[512];
    __shared__ int s_ids[8];
    __shared__ float s_w[8];
    __shared__ unsigned int s_mask, s_seq;
    __shared__ int bad;
    if (tid == 0) bad = a.d.route_error != nullptr && *a.d.route_error != 0 ? 1 : 0;
    __syncthreads();
    // the CPU lane's frequency counts age: every 4096 routes all of them halve (the whole block, before any use)
    if (a.cpu_plan != 0ull && a.d.dcnt != nullptr && ((*a.d.seq + 1) & 4095u) == 0u)
        for (int i = tid; i < a.d.n_keys; i += ROUTE_THREADS) a.d.dcnt[i] >>= 1;
    for (int e = tid; e < E; e += ROUTE_THREADS) {
        const float p = 1.0f / (1.0f + expf(-a.logits[e]));
        s_p[e] = p;
        s_sel[e] = a.bias ? p + a.bias[e] : p;
        if (a.affinity > 0.0f && a.d.tab != nullptr && a.d.tab[(size_t) a.layer * E + e] != 0ull) s_sel[e] += a.affinity;
        if (!isfinite(a.logits[e]) || !isfinite(s_sel[e])) atomicMax(&bad, 1);
    }
    __shared__ float s_bv[32];
    __shared__ int s_bi[32];
    __syncthreads();
    if (E <= 512) topk_warp(s_sel, E, a.k, s_ids);
    else topk_argmax(s_sel, E, a.k, s_ids, s_bv, s_bi);
    if (tid == 0)
        for (int i = 0; i < a.k; ++i)
            if (s_ids[i] < 0 || s_ids[i] >= E) atomicMax(&bad, 1);
    // the NEAR MISSES (STRATA_GLM_ROUTE_LOG only, cache studies): ranks k+1 .. k+near_n by the same selection score -
    // the top k are -inf in s_sel now, so the selection simply goes on (s_sel is rewritten for the prediction below)
    __shared__ int s_near[16];
    if (a.near_n > 0) {
        if (E <= 512) topk_warp(s_sel, E, a.near_n, s_near);
        else topk_argmax(s_sel, E, a.near_n, s_near, s_bv, s_bi);
    }
    // the next layer's predicted top-k (same selection rule, its own bias)
    __shared__ int s_pred[8];
    if (tid < 8) s_pred[tid] = -1;
    if (a.pred_logits != nullptr) {
        __syncthreads();
        for (int e = tid; e < E; e += ROUTE_THREADS) {
            const float p = 1.0f / (1.0f + expf(-a.pred_logits[e]));
            s_sel[e] = a.pred_bias ? p + a.pred_bias[e] : p;   // s_sel is free again: the selection is done
            if (!isfinite(a.pred_logits[e]) || !isfinite(s_sel[e])) atomicMax(&bad, 2);
        }
        __syncthreads();
        if (E <= 512) topk_warp(s_sel, E, a.k, s_pred);
        else topk_argmax(s_sel, E, a.k, s_pred, s_bv, s_bi);
        if (tid == 0)
            for (int i = 0; i < a.k; ++i)
                if (s_pred[i] < 0 || s_pred[i] >= E) atomicMax(&bad, 2);
    }
    // LOOKAHEAD: one warp per later layer, its top-k in registers (E <= 512; the same selection rule and ties)
    __shared__ short s_ah[kAhead][8];
    if ((tid >> 5) < a.n_ahead) {
        const int w = tid >> 5;
        const float* lg = a.ahead_logits + (size_t) w * E;
        const float* bs = a.ahead_bias[w];
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int e = lane + 32 * j;
            float x = -INFINITY;
            if (e < E) {
                const float p = 1.0f / (1.0f + expf(-lg[e]));
                x = bs ? p + bs[e] : p;
                if (!isfinite(lg[e]) || !isfinite(x)) atomicMax(&bad, 3);
            }
            v[j] = x;
        }
        for (int r = 0; r < a.k; ++r) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + 32 * j < E && (v[j] > bv || (v[j] == bv && lane + 32 * j < bi))) {
                    bv = v[j];
                    bi = lane + 32 * j;
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
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + 32 * j == bi) v[j] = -INFINITY;
            if (lane == 0) {
                const bool valid = bi >= 0 && bi < E && isfinite(bv);
                s_ah[w][r] = (short) (valid ? bi : -1);
                if (!valid) atomicMax(&bad, 3);
            }
        }
    }
    // the routed experts' VRAM and RAM-tier table entries, loaded at once (thread 0's plan below read them one after
    // another, each a dependent global load)
    __shared__ unsigned long long s_tab[8], s_rtab[8];
    if (tid < a.k && tid < 8) {
        const int e = s_ids[tid];
        const size_t key = (size_t) a.layer * E + (size_t) (e >= 0 && e < E ? e : 0);
        s_tab[tid] = a.d.tab[key];
        s_rtab[tid] = a.d.tab[(size_t) a.d.n_keys + key];
    }
    __syncthreads();
    if (bad != 0) {
        if (tid == 0) {
            // No probability/table lookup or DMA for an invalid route. Still publish and retire its sequence.
            if (a.d.route_error != nullptr && *a.d.route_error == 0) *a.d.route_error = 4 * a.layer + bad;
            *a.d.wait_seq = 0u;
            if (a.d.cpu_seq != nullptr) *a.d.cpu_seq = 0u;
            *a.d.cpu_flag = 0;
            *a.d.pf_n = 0;
            for (int i = 0; i < a.k; ++i) {
                a.d.plan_ptr[i] = a.d.fetch_src[i] = 0ull;
                a.d.plan_w[i] = 0.0f;
                a.d.plan_id[i] = -1;
            }
            const unsigned int seq = *a.d.seq + 1;
            *a.d.seq = seq;
            MoeRequest* rq = (MoeRequest*) a.d.ring + (seq % kRingSize);
            rq->layer = a.layer;
            rq->error = bad;
            rq->win = a.win;
            rq->win_n = a.win_n;
            __threadfence_system();
            rq->seq = seq;
            __threadfence_system();
        }
        return;
    }
    __shared__ unsigned int s_fetch, s_promo, s_cpu, s_skip;
    __shared__ unsigned long long s_pp[8], s_cs[8];
    __shared__ int s_npf, s_pfid[8];
    __shared__ unsigned long long s_pfptr[8];
    // the route counts the CPU lane's plan reads, one thread an expert (a route's experts are distinct keys): thread
    // 0's loop below took them one dependent read-modify-write after another
    __shared__ unsigned int s_cnt[8];
    __shared__ unsigned long long s_plan[8], s_fsrc[8];
    const bool lane_plan = a.cpu_plan != 0ull && skip_from > 0;
    if (tid < a.k && tid < 8)
        s_cnt[tid] = lane_plan && a.d.dcnt != nullptr ? ++a.d.dcnt[(size_t) a.layer * E + s_ids[tid]] : 0u;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < a.k; ++i) sum += (double) s_p[s_ids[i]];
        const float inv = (float) (a.norm_w ? 1.0 / fmax(sum, 6.103515625e-5) : 1.0);
        unsigned int mask = 0, fetch = 0, promo = 0, cpu = 0, skip = 0;
        unsigned long long* rtab = a.d.tab + a.d.n_keys;
        unsigned long long* spares = a.d.tab + 2 * (size_t) a.d.n_keys + (size_t) a.layer * kSpares;
        int next_spare = 0;
        // STRATA_GLM_MISS_SKIP_W: a light expert (its share of the route's weights below the bar) that is not in VRAM
        // is left out instead of fetched or computed on the CPU - the weight it carries is what the token loses
        const float msw = skipped ? 0.0f : a.miss_skip_w;
        const auto light_miss = [&](int i) { return msw > 0.0f && (float) ((double) s_p[s_ids[i]] * inv) < msw; };
        // the CPU LANE: of this route's f RAM-tier experts, cpu_take(f) (the plan's split table) are computed by the
        // host from their RAM-tier blobs while the device pulls the others over PCIe - the COLDEST by the route counts
        // (ties: the later in route order), so the hot ones are still promoted into VRAM
        unsigned int host_set = 0;
        if (lane_plan) {
            int f_ram = 0, ri[8];
            unsigned int rcnt[8];
            for (int i = 0; i < a.k; ++i) {
                const size_t key = (size_t) a.layer * E + s_ids[i];
                const unsigned int c = s_cnt[i];
                if (i < skip_from && !light_miss(i) && s_tab[i] == 0ull && s_rtab[i] != 0ull &&
                    !(a.d.helper_owned && a.d.helper_owned[key])) {
                    ri[f_ram] = i;
                    rcnt[f_ram++] = c;
                }
            }
            const int to_cpu = (int) ((a.cpu_plan >> (4 * f_ram)) & 15ull);
            for (int t = 0; t < to_cpu && t < f_ram; ++t) {
                int best = -1;
                for (int j = 0; j < f_ram; ++j)
                    if (((host_set >> ri[j]) & 1u) == 0u && (best < 0 || rcnt[j] <= rcnt[best])) best = j;
                host_set |= 1u << ri[best];
            }
        }
        for (int i = 0; i < a.k; ++i) {
            const float w = (float) ((double) s_p[s_ids[i]] * inv * a.w_scale);
            s_w[i] = w;
            const size_t key = (size_t) a.layer * E + s_ids[i];
            unsigned long long ptr = s_tab[i];
            unsigned long long src = 0ull;
            s_pp[i] = 0ull;
            s_cs[i] = 0ull;
            if (ptr == 0ull && (i >= skip_from || light_miss(i))) {
                // left out: the expert kernels skip a null plan entry (a draft only has to be a good guess)
            } else if (ptr == 0ull && a.d.helper_owned && a.d.helper_owned[key]) {
                // A helper resident is claimed BEFORE PCIe quotas, promotion or disk skipping.
                // cpu_src stays zero: the host resolves this key in the helper's own cache.
                cpu |= 1u << i;
            } else if (ptr == 0ull) {
                src = s_rtab[i];
                if (src != 0ull && ((host_set >> i) & 1u)) {
                    // to the host: a null plan entry (the expert kernels skip it), no spare, no fetch
                    cpu |= 1u << i;
                    s_cs[i] = src;
                    src = 0ull;
                } else if (src == 0ull && (i >= a.disk_skip_from || (float) ((double) s_p[s_ids[i]] * inv) < a.disk_skip_w)) {
                    // on disk only and ranked low: left out like a draft's miss (the expert kernels skip a null
                    // plan entry) - this token goes without it instead of waiting on a disk read
                    skip |= 1u << i;
                } else {
                    // not in VRAM: land it in a spare slot of this layer (it becomes resident: the table says so
                    // from here on) or, with none left, in this entry's scratch slot.  promote_min > 1: keep it
                    // only when its aged route count clears the bar - one-offs take scratch and do not churn the tier
                    const unsigned int rc = a.d.dcnt != nullptr ? a.d.dcnt[key] : 0u;
                    while (next_spare < kSpares && spares[next_spare] == 0ull) ++next_spare;
                    if (next_spare < kSpares && (a.promote_min <= 1 || rc >= (unsigned int) a.promote_min)) {
                        ptr = spares[next_spare];
                        spares[next_spare] = 0ull;
                        a.d.tab[key] = ptr;
                        promo |= 1u << i;
                        s_pp[i] = ptr;
                    } else if (a.scr_used != nullptr) {
                        // a verify window: a scratch slot no earlier token of the layer took
                        const unsigned int u = *a.scr_used;
                        int j = 0;
                        while (j < 8 && ((u >> j) & 1u)) ++j;
                        if (j < 8) {
                            *a.scr_used = u | (1u << j);
                            ptr = a.d.scratch[j];
                        } else {
                            a.scr_used[1] |= 0x80000000u;   // sticky for the window (word 0 resets each layer)
                            // a staggered group (bits 8-15 of the word: its own slots) reuses one of ITS slots: the
                            // other group's still hold the experts of a layer it has yet to compute - maybe of another
                            // type, whose garbage can be non-finite, and after a non-finite route every later one is
                            // refused (the redo too)
                            const unsigned int own = (u >> 8) & 0xFFu;
                            int f = i;
                            if (own != 0u) {
                                const int n = i % __popc(own);
                                f = 0;
                                for (int c = 0; f < 8; ++f)
                                    if (((own >> f) & 1u) && c++ == n) break;
                            }
                            ptr = a.d.scratch[f];
                        }
                    } else {
                        ptr = a.d.scratch[i];
                    }
                    if (src != 0ull) fetch |= 1u << i;
                    else mask |= 1u << i;   // only on disk: the host provides the source
                }
            }
            s_plan[i] = ptr;   // (out to the device's plan below, a thread an entry)
            s_fsrc[i] = src;
        }
        s_mask = mask;
        s_fetch = fetch;
        s_promo = promo;
        s_cpu = cpu;
        s_skip = skip;
        // PREFETCH: the predicted experts of the next layer that are not in VRAM but are in the RAM tier claim
        // that layer's spares now (one spare stays for its real misses); the side stream copies them while this
        // layer and the next one's attention compute, and the next route sees them as resident.  Only the
        // prediction's first pf_rank ranks (STRATA_GLM_PREFETCH_RANK): its top guesses are right almost always (Maya-M
        // on a 3090 + 3060, the next layer used rank 0 / 1 / 2 / 3 96 / 88 / 75 / 62 % of the time, ranks 4..7 52 .. 24 %)
        // - the first NON-resident guess is often a low rank, and a wrong copy costs the link what a right one saves
        int npf = 0;
        if (a.pred_logits != nullptr && a.max_pf > 0 && !skipped) {
            const int L1 = a.layer + 1;
            unsigned long long* sp1 = a.d.tab + 2 * (size_t) a.d.n_keys + (size_t) L1 * kSpares;
            for (int i = 0; i < a.k && i < a.pf_rank && npf < a.max_pf; ++i) {
                const int e = s_pred[i];
                if (e < 0) continue;
                const size_t key = (size_t) L1 * E + e;
                if (a.d.tab[key] != 0ull || (a.d.helper_owned && a.d.helper_owned[key])) continue;
                const unsigned long long src = rtab[key];
                if (src == 0ull) continue;
                int navail = 0, first = -1;
                for (int j = 0; j < kSpares; ++j)
                    if (sp1[j] != 0ull) {
                        ++navail;
                        if (first < 0) first = j;
                    }
                if (navail < 2) break;
                const unsigned long long dst = sp1[first];
                sp1[first] = 0ull;
                a.d.tab[key] = dst;
                a.d.pf_src[npf] = src;
                a.d.pf_dst[npf] = dst;
                s_pfid[npf] = e;
                s_pfptr[npf] = dst;
                ++npf;
            }
        }
        *a.d.pf_n = npf;
        s_npf = npf;
        const unsigned int seq = *a.d.seq + 1;
        *a.d.seq = seq;
        s_seq = seq;
        *a.d.wait_seq = mask ? seq : 0u;
        if (a.d.cpu_seq != nullptr) *a.d.cpu_seq = cpu ? seq : 0u;
        *a.d.cpu_flag = 0;
    }
    __syncthreads();
    if (tid < 8) {
        const bool on = tid < a.k;
        if (on) {
            a.d.plan_ptr[tid] = s_plan[tid];
            a.d.plan_w[tid] = s_w[tid];
            a.d.plan_id[tid] = s_ids[tid];
        }
        a.d.fetch_src[tid] = on ? s_fsrc[tid] : 0ull;
    }
    MoeRequest* rq = (MoeRequest*) a.d.ring + (s_seq % kRingSize);
    if (s_cpu != 0u)
        for (int e = tid; e < a.n_embd; e += ROUTE_THREADS) rq->x[e] = a.x[e];
    // the request's fields before x: put together in shared memory by thread 0, then out over PCIe in 8-byte words by
    // as many threads at once (thread 0's ~70 scalar stores to host memory one after another were most of the
    // kernel's time); the first word holds seq (written last, below) and layer - layer goes alone
    constexpr int kHeadWords = (int) (offsetof(MoeRequest, x) / 8);
    static_assert(offsetof(MoeRequest, x) % 8 == 0 && offsetof(MoeRequest, layer) == 4, "MoeRequest's head");
    __shared__ unsigned long long s_rq[kHeadWords];
    if (tid == 0) {
        MoeRequest* q = (MoeRequest*) s_rq;
        q->layer = a.layer;
        q->error = 0;
        q->miss_mask = s_mask;
        q->fetch_mask = s_fetch;
        q->promo_mask = s_promo;
        q->cpu_mask = s_cpu;
        q->win = a.win;
        q->win_n = a.win_n;
        q->skip_mask = s_skip;
        for (int i = 0; i < 8; ++i) {
            const bool on = i < a.k;   // (the entries past k as the host never reads them: zero)
            q->ids[i] = on ? s_ids[i] : 0;
            q->w[i] = on ? s_w[i] : 0.0f;
            q->promo_ptr[i] = on ? s_pp[i] : 0ull;
            q->pred[i] = on ? s_pred[i] : 0;
            q->cpu_src[i] = on ? s_cs[i] : 0ull;
            q->pf_ids[i] = i < s_npf ? s_pfid[i] : 0;
            q->pf_ptr[i] = i < s_npf ? s_pfptr[i] : 0ull;
        }
        q->pf_n = s_npf;
        for (int d2 = 0; d2 < kAhead; ++d2)
            for (int i = 0; i < 8; ++i) q->ahead[d2][i] = d2 < a.n_ahead && i < a.k ? s_ah[d2][i] : (short) -1;
        for (int i = 0; i < 16; ++i) q->near_ids[i] = i < a.near_n ? (short) s_near[i] : (short) -1;
        rq->layer = a.layer;
    }
    __syncthreads();
    for (int w8 = 1 + tid; w8 < kHeadWords; w8 += ROUTE_THREADS) ((unsigned long long*) rq)[w8] = s_rq[w8];
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        rq->seq = s_seq;
#if defined(__HIPCC__)
        // The CPU polls this signal without a stream query. As in Strata #697,
        // publish the signal itself after publishing the request payload.
        __threadfence_system();
#endif
    }
}

__global__ void moe_wait_kernel(MoeDev d, int n_embd) {
    __shared__ unsigned int s_need;
    const int tid = threadIdx.x;
    if (tid == 0) s_need = *d.wait_seq;
    __syncthreads();
    if (s_need == 0) return;
    const MoeResponse* r = (const MoeResponse*) d.resp;
    if (tid == 0) {
        // spin on the host's answer (bounded: a dead host must not hang the GPU forever - ~60 s)
        const long long t0 = clock64();
        while (r->seq != s_need) {
            if (clock64() - t0 > (1ll << 36)) break;
        }
        __threadfence_system();
    }
    __syncthreads();
    // the disk-only experts: the host read each into host-mapped memory and answers with that SOURCE; the
    // fetch kernel copies it into the slot the route already planned
    const unsigned int pm = ((volatile const MoeResponse*) r)->ptr_mask;
    if (tid < 8 && (pm >> tid) & 1u) d.fetch_src[tid] = ((volatile const MoeResponse*) r)->ptr[tid];
    const int nu = ((volatile const MoeResponse*) r)->n_upd;
    for (int u = tid; u < nu && u < 64; u += blockDim.x)
        d.tab[((volatile const MoeResponse*) r)->upd_key[u]] = ((volatile const MoeResponse*) r)->upd_val[u];
    if (((volatile const MoeResponse*) r)->cpu) {
        for (int e = tid; e < n_embd; e += blockDim.x) d.cpu_part[e] = ((volatile const MoeResponse*) r)->cpu_part[e];
        if (tid == 0) *d.cpu_flag = 1;
    }
}

// the CPU LANE's answer for the last route (when it sent experts to the host): spin, then out += its weighted sum
__global__ void moe_cpu_wait_kernel(MoeDev d, int n_embd, float* out) {
    __shared__ unsigned int s_need;
    const int tid = threadIdx.x;
    if (tid == 0) s_need = *d.cpu_seq;
    __syncthreads();
    if (s_need == 0) return;
    const CpuAnswer* r = (const CpuAnswer*) d.cpu_ans;
    if (tid == 0) {
        const long long t0 = clock64();   // bounded like moe_wait (~60 s)
        while (r->seq != s_need) {
            if (clock64() - t0 > (1ll << 36)) break;
        }
        __threadfence_system();
    }
    __syncthreads();
    for (int e = tid; e < n_embd; e += blockDim.x) out[e] += ((volatile const CpuAnswer*) r)->part[e];
}

// the deferred form (STRATA_GLM_DEFER_CPU): the answer of a layer whose FFN write already went out without it, added to
// the streams it would have reached - R[j] += post[j] * part (the layer's FFN-side post weights)
__global__ void moe_cpu_wait_streams_kernel(MoeDev d, int n_embd, const float* post, float* R, int hc) {
    __shared__ unsigned int s_need;
    __shared__ float s_post[8];
    const int tid = threadIdx.x;
    if (tid == 0) s_need = *d.cpu_seq;
    if (tid < hc) s_post[tid] = post[tid];
    __syncthreads();
    if (s_need == 0) return;
    const CpuAnswer* r = (const CpuAnswer*) d.cpu_ans;
    if (tid == 0) {
        const long long t0 = clock64();
        while (r->seq != s_need) {
            if (clock64() - t0 > (1ll << 36)) break;
        }
        __threadfence_system();
    }
    __syncthreads();
    for (int e = tid; e < n_embd; e += blockDim.x) {
        const float v = ((volatile const CpuAnswer*) r)->part[e];
        for (int j = 0; j < hc; ++j) R[(size_t) j * n_embd + e] += s_post[j] * v;
    }
}

__global__ void __launch_bounds__(256) moe_prefetch_kernel(MoeDev d, size_t n16) {
    const int n = *d.pf_n;
    for (int i = 0; i < n; ++i) {
        const uint4* s = (const uint4*) d.pf_src[i];
        uint4* t = (uint4*) d.pf_dst[i];
        for (size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x; j < n16; j += (size_t) gridDim.x * blockDim.x)
            t[j] = s[j];
    }
}

// one block per SM, 16-byte loads: the PCIe pull measured 11.5 GB/s (a little over cudaMemcpy's 10.7)
__global__ void __launch_bounds__(256) moe_fetch_kernel(MoeDev d, int k, size_t n16) {
    for (int i = 0; i < k; ++i) {
        const unsigned long long src = d.fetch_src[i];
        if (src == 0ull) continue;
        const uint4* s = (const uint4*) src;
        uint4* t = (uint4*) d.plan_ptr[i];
        for (size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x; j < n16; j += (size_t) gridDim.x * blockDim.x)
            t[j] = s[j];
    }
}

constexpr int GU_ROWS = 32;   // h values (gate rows = up rows) per block: one q8_1 block of h
constexpr int GU_WARPS = 16;  // ... 2 gate + 2 up rows per warp (8 per warp measured 90 vs 65 us a layer: occupancy)

// grid (n_ff / GU_ROWS, k [+ 1]): the extra row of blocks (blockIdx.y == k) computes the SHARED expert's down rows
// into sh_out while the routed experts' blocks run - in moe_down it was a serial tail (~25 us of 72)
#if defined(STRATA_USE_HIP)
template<int TG, int NW = GU_WARPS, bool LUT = false, bool GLOBAL = false, bool SPLIT = false,
         bool DIRECT = false, bool WIDE = false, int MINWAVES = 0, bool MULTI = false>
__attribute__((amdgpu_waves_per_eu(MINWAVES > 0 ? MINWAVES : 1)))
__global__ void __launch_bounds__(NW * 32)
#else
template<int TG>
__global__ void __launch_bounds__(GU_WARPS * 32)
#endif
moe_gate_up_kernel(MoeDev d, int k, int n_embd, int n_ff, float limit,
                                                                    const block_q8_1* __restrict__ xq,
                                                                    block_q8_1* __restrict__ hq,
                                                                    const uint8_t* __restrict__ sh_down, int sh_type,
                                                                    const block_q8_1* __restrict__ sh_hq, int n_ff_sh,
                                                                    float* __restrict__ sh_out) {
    if (d.skip != nullptr && *d.skip) return;
#if !defined(STRATA_USE_HIP)
    constexpr int NW = GU_WARPS;
    const int ei = blockIdx.y, chunk = blockIdx.x, n_chunks = (int) gridDim.x;
#else
    int ei = blockIdx.y, chunk = blockIdx.x, n_chunks = (int) gridDim.x;
    if constexpr (MULTI) {
        // moe_experts_multi: x = (expert, token) with the token fastest, y = the chunk - every token's blocks for a chunk
        // run together; token t's plan, input, h and shared expert at d's mt_ strides
        const int t = (int) blockIdx.x % d.mt_T;
        ei = (int) blockIdx.x / d.mt_T;
        chunk = blockIdx.y;
        n_chunks = (int) gridDim.y;
        d.plan_ptr += 8 * t;
        xq += (size_t) t * d.mt_xq_s;
        hq += (size_t) t * d.mt_hq_s;
        if (sh_hq != nullptr) sh_hq += (size_t) t * d.mt_sh_hq_s;
        if (sh_out != nullptr) sh_out += (size_t) t * d.mt_out_s;
    }
#endif
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#if defined(STRATA_USE_HIP)
    if constexpr (!SPLIT)
#endif
    if (ei == k) {
        const int per = (n_embd + n_chunks - 1) / n_chunks;
        const int r1 = min(n_embd, (chunk + 1) * per);
        for (int r = chunk * per + warp; r < r1; r += NW) {
            const float v = any_row_dot(sh_type, sh_down, r, sh_hq, nullptr, n_ff_sh, lane);
            if (lane == 0) sh_out[r] = v;
        }
        return;
    }
    const unsigned long long ptr = d.plan_ptr[ei];
    if (ptr == 0ull) return;
    __shared__ float sg[GU_ROWS], su[GU_ROWS];
    __shared__ uint32_t s_tab[TabWords<TG>::n > 0 ? TabWords<TG>::n : 1];
    const block_q8_1* xg = xq;
#if GLMF_SM60
    __shared__ __align__(16) uint32_t s_sg[sm60_sign_on<TG> ? sm60_sign_n<TG> * sm60_sign_w<TG> : 1];
    if constexpr (sm60_sign_on<TG>) sm60_load_signs<TG>(s_sg);
    // the layer input staged in shared memory when the launch gives it the room (moe_gate_up on cc 6.0: every block
    // reads the same 4.6 KB, the calls' activation loads then hit shared memory - IQ2_S 274 -> 256 us on a PH402 die)
    {
        extern __shared__ int s_gx[];
        uint32_t dyn;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
        const int nxw = (n_embd / 32) * (int) (sizeof(block_q8_1) / 4);
        if (dyn >= (uint32_t) nxw * 4u) {
            for (int i = threadIdx.x; i < nxw; i += blockDim.x) s_gx[i] = ((const int*) xq)[i];
            xg = (const block_q8_1*) s_gx;
        }
    }
#endif
#if defined(STRATA_USE_HIP)
    __shared__ uint4 s_sign[LUT && !DIRECT ? sign_count<TG> : 1];
    if constexpr (LUT && !DIRECT) load_signs<TG>(s_sign);
    if constexpr (!GLOBAL)
#endif
    load_tab<TG>(s_tab);
    __syncthreads();
    const size_t rb = rbytes<TG>(n_embd);
    const uint8_t* blob = (const uint8_t*) ptr;
    // each warp: 2 gate + 2 up rows in ONE pass (one activation load per call for all four)
    constexpr int RPW = GU_ROWS / NW;
    const uint8_t* rows[2 * RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        const int r = chunk * GU_ROWS + warp * RPW + j;
        rows[j] = blob + (size_t) r * rb;
        rows[RPW + j] = blob + (size_t) n_ff * rb + (size_t) r * rb;
    }
    float s[2 * RPW];
    const uint32_t* tab = s_tab;
#if defined(STRATA_USE_HIP)
    if constexpr (GLOBAL) tab = TG == 16 ? (const uint32_t*) iq2xxs_grid :
                                TG == 18 ? (const uint32_t*) iq3xxs_grid : (const uint32_t*) iq2s_grid;
    if constexpr (LUT) {
        rows_dot_rdna<TG, 2 * RPW>(rows, xq, n_embd, lane, tab, s_sign, s, DIRECT, WIDE);
    } else
#endif
#if GLMF_SM60
    if constexpr (sm60_sign_on<TG>) rows_dot<TG, 2 * RPW, true>(rows, xg, n_embd, lane, tab, s, s_sg); else
#endif
    rows_dot<TG, 2 * RPW>(rows, xg, n_embd, lane, tab, s);
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < RPW; ++j) {
            sg[warp * RPW + j] = s[j];
            su[warp * RPW + j] = s[RPW + j];
        }
    }
    __syncthreads();
    if (warp == 0) {
        const float g = fminf(sg[lane], limit);
        const float u = fminf(fmaxf(su[lane], -limit), limit);
        q8_1_store_warp((g / (1.0f + expf(-g))) * u, hq + (size_t) ei * (n_ff / 32) + chunk, lane);
    }
}

constexpr int DOWN_ROWS = 4;

#if defined(STRATA_USE_HIP)
template<int TD, int NR = DOWN_ROWS, bool LUT = false, bool GLOBAL = false, bool DIRECT = false, bool WIDE = false,
         bool MULTI = false>
#else
template<int TD>
#endif
__global__ void __launch_bounds__(256) moe_down_kernel(MoeDev d, int k, int n_embd, int n_ff, size_t down_off,
                                                       const block_q8_1* __restrict__ hq,
                                                       const float* __restrict__ sh_out, float* __restrict__ out) {
    if (d.skip != nullptr && *d.skip) return;
#if !defined(STRATA_USE_HIP)
    constexpr int NR = DOWN_ROWS;
    const int rb0 = blockIdx.x;
#else
    int rb0 = blockIdx.x;
    if constexpr (MULTI) {
        // moe_experts_multi: x = (row block, token) with the token fastest; token t's plan, h, shared expert output,
        // output and CPU part at d's mt_ strides
        const int t = (int) blockIdx.x % d.mt_T;
        rb0 = (int) blockIdx.x / d.mt_T;
        d.plan_ptr += 8 * t;
        d.plan_w += 8 * t;
        hq += (size_t) t * d.mt_hq_s;
        if (sh_out != nullptr) sh_out += (size_t) t * d.mt_out_s;
        out += (size_t) t * d.mt_out_s;
        d.cpu_flag += t;
        d.cpu_part += (size_t) t * d.mt_cpu_s;
    }
#endif
    // NR output rows per block; warp w computes those rows for expert w (one activation load per call
    // for all of them), the combine runs in plan order like the reference's axpy chain
    __shared__ unsigned long long s_ptr[8];
    __shared__ float s_w[8];
    __shared__ float s_part[8][NR];
    __shared__ uint32_t s_tab[TabWords<TD>::n > 0 ? TabWords<TD>::n : 1];
#if GLMF_SM60
    __shared__ __align__(16) uint32_t s_sg[sm60_sign_on<TD> ? sm60_sign_n<TD> * sm60_sign_w<TD> : 1];
    if constexpr (sm60_sign_on<TD>) sm60_load_signs<TD>(s_sg);
#endif
#if defined(STRATA_USE_HIP)
    __shared__ uint4 s_sign[LUT && !DIRECT ? sign_count<TD> : 1];
    if constexpr (LUT && !DIRECT) load_signs<TD>(s_sign);
    if constexpr (!GLOBAL)
#endif
    load_tab<TD>(s_tab);
    if (threadIdx.x < 8) {
        s_ptr[threadIdx.x] = (int) threadIdx.x < k ? d.plan_ptr[threadIdx.x] : 0ull;
        s_w[threadIdx.x] = (int) threadIdx.x < k ? d.plan_w[threadIdx.x] : 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = rb0 * NR;
    const size_t drb = rbytes<TD>(n_ff);
    {
        float s[NR];
        if (warp < k && s_ptr[warp] != 0ull) {
            const uint8_t* rows[NR];
#pragma unroll
            for (int j = 0; j < NR; ++j) rows[j] = (const uint8_t*) s_ptr[warp] + down_off + (size_t) (r0 + j) * drb;
            const uint32_t* tab = s_tab;
#if defined(STRATA_USE_HIP)
            if constexpr (GLOBAL) tab = TD == 16 ? (const uint32_t*) iq2xxs_grid :
                                        TD == 18 ? (const uint32_t*) iq3xxs_grid : (const uint32_t*) iq2s_grid;
            if constexpr (LUT) {
                rows_dot_rdna<TD, NR>(rows, hq + (size_t) warp * (n_ff / 32), n_ff, lane, tab, s_sign, s, DIRECT, WIDE);
            } else
#endif
#if GLMF_SM60
            if constexpr (sm60_sign_on<TD>) rows_dot<TD, NR, true>(rows, hq + (size_t) warp * (n_ff / 32), n_ff, lane, tab, s, s_sg); else
#endif
            rows_dot<TD, NR>(rows, hq + (size_t) warp * (n_ff / 32), n_ff, lane, tab, s);
        } else {
#pragma unroll
            for (int j = 0; j < NR; ++j) s[j] = 0.0f;
        }
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < NR; ++j) s_part[warp][j] = s[j];
    }
    __syncthreads();
    if (threadIdx.x < NR) {
        const int j = threadIdx.x, r = r0 + j;
        float m = 0.0f;
        for (int i = 0; i < k; ++i) m += s_w[i] * s_part[i][j];
        if (*d.cpu_flag) m += d.cpu_part[r];
        out[r] = m + (sh_out != nullptr ? sh_out[r] : 0.0f);
    }
}

// ---------------------------------------------------------------- the prompt path's lightly routed experts
// An expert with a few rows (tokens) of a prompt chunk: its weights read ONCE, each 32-value sub-block decoded once
// and dotted with up to LR_MT tokens at a time (MMQ computes tiles of 64-128 tokens - mostly empty at ~10 rows an
// expert, as at a 300-token prompt).  The sub-block decode is the decode path's dw<T>, split from its dot.
constexpr int LR_MT = 4;
#ifndef GLMF_LR_MT2
#define GLMF_LR_MT2 8
#endif
constexpr int LR_MT2 = GLMF_LR_MT2;   // tokens a pass of the light kernels (rows_multi2)
template<int T>
__device__ __forceinline__ void dwm(const uint8_t* row, int kbx, int iqs, const XV* x, int nt, const uint32_t* tab,
                                    float* acc) {
    int g[8];
    if constexpr (T == 19) {
        const block_iq1_s* bq1 = (const block_iq1_s*) row + kbx;
        const int qs_packed = get_int_b2(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq1->qh[iqs];
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
            g[l0] = (grid >> 0) & 0x0F0F0F0F;
            g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
        }
        const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
        const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) {
            if (t >= nt) break;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(g[i], x[t].u[i], sumi);
            acc[t] += d1q * (x[t].d * sumi + x[t].s * delta);
        }
    } else if constexpr (T == 18) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            g[l0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            g[l0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        const int ls = aux32 >> 28;
        const float d = __half2float(bq3->d);
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) {
            if (t >= nt) break;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(g[i], x[t].u[i], sumi);
            sumi = (ls * sumi + sumi / 2) / 2;
            acc[t] += d * x[t].d * sumi;
        }
    } else {   // 16: IQ2_XXS
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            g[k0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            g[k0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        const int ls = aux32 >> 27 | 1;
        const float d = __half2float(bq2->d);
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) {
            if (t >= nt) break;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(g[i], x[t].u[i], sumi);
            sumi = sumi * ls / 8;
            acc[t] += d * x[t].d * sumi;
        }
    }
}

// one 32-value sub-block decoded (dwm's first half), then dotted with any number of activations
struct Dec {
    int g[8];
    float a, b;   // IQ1_S: d1q, delta; IQ3_XXS / IQ2_XXS: d, -
    int ls;
};
template<int T>
__device__ __forceinline__ void dec_sub(const uint8_t* row, int kbx, int iqs, const uint32_t* tab, Dec& o) {
    if constexpr (T == 19) {
        const block_iq1_s* bq1 = (const block_iq1_s*) row + kbx;
        const int qs_packed = get_int_b2(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq1->qh[iqs];
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
            o.g[l0] = (grid >> 0) & 0x0F0F0F0F;
            o.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
        }
        o.a = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
        o.b = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    } else if constexpr (T == 18) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            o.g[l0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            o.g[l0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        o.ls = aux32 >> 28;
        o.a = __half2float(bq3->d);
    } else {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            o.g[k0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            o.g[k0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        o.ls = aux32 >> 27 | 1;
        o.a = __half2float(bq2->d);
    }
}
template<int T>
__device__ __forceinline__ float dec_dot(const Dec& o, const XV& x) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sumi = __dp4a(o.g[i], x.u[i], sumi);
    if constexpr (T == 19) return o.a * (x.d * sumi + x.s * o.b);
    else if constexpr (T == 18) return o.a * x.d * ((o.ls * sumi + sumi / 2) / 2);
    else return o.a * x.d * (sumi * o.ls / 8);
}

// NR weight rows x MT tokens a pass: each k step decodes the NR rows' sub-blocks once, then every token's activation
// is loaded once and dotted with all NR (dwm decoded per row and per token group)
template<int T, int NR, int MT, typename XF>
__device__ __forceinline__ void rows_multi2(const uint8_t* const* rows, int n_in, int ntok, int lane, const uint32_t* tab,
                                            XF x_of, float (&out)[NR][MT], int t0) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    const int nt = min(MT, ntok - t0);
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) out[r][t] = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, ki = k % Fm::ipb, iqs = Fm::step * ki;
        Dec dr[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) dec_sub<T>(rows[r], kbx, iqs, tab, dr[r]);
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t >= nt) break;
            const XV xv = load_xv(x_of(t0 + t) + kbx * (Fm::qk / 32) + ki);
#pragma unroll
            for (int r = 0; r < NR; ++r) out[r][t] += dec_dot<T>(dr[r], xv);
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) out[r][t] = warp_sum(out[r][t]);
}

// NR weight rows (pointers) x the rows' tokens, LR_MT tokens a pass; x(t) = the q8_1 activation of token t
template<int T, int NR, typename XF>
__device__ __forceinline__ void rows_multi(const uint8_t* const* rows, int n_in, int ntok, int lane, const uint32_t* tab,
                                           XF x_of, float (&out)[NR][LR_MT], int t0) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    const int nt = min(LR_MT, ntok - t0);
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) out[r][t] = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, ki = k % Fm::ipb, iqs = Fm::step * ki;
        XV xv[LR_MT];
#pragma unroll
        for (int t = 0; t < LR_MT; ++t)
            if (t < nt) xv[t] = load_xv(x_of(t0 + t) + kbx * (Fm::qk / 32) + ki);
#pragma unroll
        for (int r = 0; r < NR; ++r) dwm<T>(rows[r], kbx, iqs, xv, nt, tab, out[r]);
    }
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) out[r][t] = warp_sum(out[r][t]);
}

// light[i] = {slot, r0, nr}: grid (n_ff / 32, n); 16 warps x (1 gate + 1 up row)... as the decode kernel: 2 + 2
template<int TG>
__global__ void __launch_bounds__(GU_WARPS * 32) rows_gate_up_kernel(const uint8_t* __restrict__ base, size_t stride,
                                                                     const int* __restrict__ light,
                                                                     const int* __restrict__ row_tok,
                                                                     const block_q8_1* __restrict__ xq, int n_embd,
                                                                     int n_ff, float* __restrict__ out, int ld) {
    const int i = blockIdx.y, chunk = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ uint32_t s_tab[TabWords<TG>::n > 0 ? TabWords<TG>::n : 1];
    load_tab<TG>(s_tab);
    __syncthreads();
    const int slot = light[3 * i], R0 = light[3 * i + 1], NR = light[3 * i + 2];
    const uint8_t* blob = base + (size_t) slot * stride;
    const size_t rb = rbytes<TG>(n_embd);
    constexpr int RPW = GU_ROWS / GU_WARPS;
    const uint8_t* rows[2 * RPW];
    int col[2 * RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        const int r = chunk * GU_ROWS + warp * RPW + j;
        rows[j] = blob + (size_t) r * rb;
        rows[RPW + j] = blob + (size_t) (n_ff + r) * rb;
        col[j] = r;
        col[RPW + j] = n_ff + r;
    }
    const int nblk = n_embd / 32;
    const auto x_of = [&](int t) { return xq + (size_t) row_tok[R0 + t] * nblk; };
    for (int t0 = 0; t0 < NR; t0 += LR_MT2) {
        float s[2 * RPW][LR_MT2];
        rows_multi2<TG, 2 * RPW, LR_MT2>(rows, n_embd, NR, lane, s_tab, x_of, s, t0);
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < 2 * RPW; ++j)
#pragma unroll
                for (int t = 0; t < LR_MT2; ++t)
                    if (t0 + t < NR) out[(size_t) (R0 + t0 + t) * ld + col[j]] = s[j][t];
    }
}

// h = swiglu(gate, up) of the light rows [rlo, rhi) (GU at out rows, ld floats apart) -> q8_1 rows (n_ff / 32 each)
__global__ void rows_swiglu_q8_kernel(const float* __restrict__ gu, int ld, int rlo, int rhi, int n_ff, float limit,
                                      block_q8_1* __restrict__ hq) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, per = n_ff / 32;
    const int64_t r = rlo + w / per, b = w % per;
    if (r >= rhi) return;
    const float* row = gu + (size_t) r * ld;
    const float gv = fminf(row[b * 32 + lane], limit);
    const float uv = fminf(fmaxf(row[n_ff + b * 32 + lane], -limit), limit);
    q8_1_store_warp((gv / (1.0f + expf(-gv))) * uv, hq + (size_t) (r - rlo) * per + b, lane);
}

// the down rows: grid (n_embd / 32, n); 16 warps x 2 rows; out row r (ld floats) cols [0, n_embd)
template<int TD>
__global__ void __launch_bounds__(GU_WARPS * 32) rows_down_kernel(const uint8_t* __restrict__ base, size_t stride,
                                                                  size_t down_off, const int* __restrict__ light,
                                                                  const block_q8_1* __restrict__ hq, int rlo, int n_ff,
                                                                  int n_embd, float* __restrict__ out, int ld) {
    const int i = blockIdx.y, chunk = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ uint32_t s_tab[TabWords<TD>::n > 0 ? TabWords<TD>::n : 1];
    load_tab<TD>(s_tab);
    __syncthreads();
    const int slot = light[3 * i], R0 = light[3 * i + 1], NR = light[3 * i + 2];
    const uint8_t* blob = base + (size_t) slot * stride + down_off;
    const size_t rb = rbytes<TD>(n_ff);
    constexpr int RPW = 2;
    const uint8_t* rows[RPW];
    int col[RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        col[j] = chunk * (GU_WARPS * RPW) + warp * RPW + j;
        rows[j] = blob + (size_t) col[j] * rb;
    }
    const int per = n_ff / 32;
    const auto x_of = [&](int t) { return hq + (size_t) (R0 + t - rlo) * per; };
    for (int t0 = 0; t0 < NR; t0 += LR_MT2) {
        float s[RPW][LR_MT2];
        rows_multi2<TD, RPW, LR_MT2>(rows, n_ff, NR, lane, s_tab, x_of, s, t0);
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < RPW; ++j)
#pragma unroll
                for (int t = 0; t < LR_MT2; ++t)
                    if (t0 + t < NR) out[(size_t) (R0 + t0 + t) * ld + col[j]] = s[j][t];
    }
}

// x rows (ld floats apart) -> q8_1 rows (n / 32 blocks each): a warp a block
__global__ void rows_q8_kernel(const float* __restrict__ x, int ld, int rows, int n, block_q8_1* __restrict__ q) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, per = n / 32;
    if (w >= (int64_t) rows * per) return;
    const int64_t r = w / per, b = w % per;
    q8_1_store_warp(x[(size_t) r * ld + b * 32 + lane], q + (size_t) r * per + b, lane);
}

// ---------------------------------------------------------------- the NextN block's glue
// A chained draft's embedding row: row *tok of token_embd kept on the device, the token the previous draft step's argmax
// left there.  Dequantized as ggml's to_float does it, so the floats are the host's bit for bit: Q6_K d * scale * q (the
// two products in that order, dequantize_row_q6_K), Q8_0 q * d.  One thread a value.
__global__ void embed_tok_kernel(const void* __restrict__ table, int type, int n_rows, const int* __restrict__ tok,
                                 int n_embd, float* __restrict__ out) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_embd) return;
    const int t = *tok;
    if (t < 0 || t >= n_rows) {   // (never: a draft is a row of the head)
        out[e] = 0.0f;
        return;
    }
    if (type == 8) {
        const block_q8_0* b = (const block_q8_0*) table + (size_t) t * (n_embd / QK8_0) + e / QK8_0;
        out[e] = (float) b->qs[e % QK8_0] * __half2float(b->d);
        return;
    }
    // Q6_K: 256 values a block in two halves of 128, each four runs of 32 (ggml-quants.c's loop order)
    const block_q6_K* b = (const block_q6_K*) table + (size_t) t * (n_embd / QK_K) + e / QK_K;
    const int j = e % QK_K, h = j / 128, r = j % 128, quarter = r / 32, l = r % 32, is = l / 16;
    const uint8_t* ql = b->ql + 64 * h;
    const uint8_t* qh = b->qh + 32 * h;
    const int8_t* sc = b->scales + 8 * h;
    const int lo = quarter & 1 ? ql[l + 32] : ql[l];
    const int q = (quarter < 2 ? (lo & 0xF) : (lo >> 4)) | (((qh[l] >> (2 * quarter)) & 3) << 4);
    out[e] = __half2float(b->d) * (float) sc[is + 2 * quarter] * (float) (int8_t) (q - 32);
}

// cat = [rms(emb) * enorm, rms(h) * hnorm] as q8_1 (eh_proj's input); n_embd <= 4096, one block of 1024
__global__ void __launch_bounds__(1024) mtp_in_kernel(const float* __restrict__ emb, const float* __restrict__ h,
                                                      const float* __restrict__ enorm, const float* __restrict__ hnorm,
                                                      float eps, int n, block_q8_1* __restrict__ catq) {
    __shared__ float sred[32];
    const int tid = threadIdx.x, lane = tid & 31;
    float a[4], b[4];
    float sa = 0.0f, sb = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        a[i] = e < n ? emb[e] : 0.0f;
        b[i] = e < n ? h[e] : 0.0f;
        sa += a[i] * a[i];
        sb += b[i] * b[i];
    }
    sa = block_sum(sa, sred);
    sb = block_sum(sb, sred);
    const float ia = rsqrtf(sa / (float) n + eps), ib = rsqrtf(sb / (float) n + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        if (e >= n) break;
        q8_1_store_warp(a[i] * ia * enorm[e], catq + e / 32, lane);
        q8_1_store_warp(b[i] * ib * hnorm[e], catq + (n + e) / 32, lane);
    }
}
// h += add (when given), then x = rms(h) * w and its q8_1; n_embd <= 4096, one block of 1024
__global__ void __launch_bounds__(1024) rms_q8_kernel(float* __restrict__ h, const float* __restrict__ add,
                                                      const float* __restrict__ w, float eps, int n,
                                                      float* __restrict__ x, block_q8_1* __restrict__ xq) {
    __shared__ float sred[32];
    const int tid = threadIdx.x, lane = tid & 31;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        float t = e < n ? h[e] : 0.0f;
        if (add != nullptr && e < n) {
            t += add[e];
            h[e] = t;
        }
        v[i] = t;
        ss += t * t;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) n + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        if (e >= n) break;
        const float xv = v[i] * inv * w[e];
        x[e] = xv;
        q8_1_store_warp(xv, xq + e / 32, lane);
    }
}

// one block: float4 loads, four in flight per thread (the vocabulary is ~0.6 MB; scalar loads ran at ~10 GB/s)
__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ x, int n, int* __restrict__ out,
                                                      const int* __restrict__ map, int map_from) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    const auto take = [&](float v, int i) {
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    };
    const int B = (int) blockDim.x;
    int tail = 0;
    if ((((uintptr_t) x) & 15u) == 0) {
        const float4* x4 = (const float4*) x;
        const int n4 = n >> 2;
        int i = threadIdx.x;
        for (; i + 3 * B < n4; i += 4 * B) {
            float4 a[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) a[u] = x4[i + u * B];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int b = 4 * (i + u * B);
                take(a[u].x, b);
                take(a[u].y, b + 1);
                take(a[u].z, b + 2);
                take(a[u].w, b + 3);
            }
        }
        for (; i < n4; i += B) {
            const float4 a = x4[i];
            take(a.x, 4 * i);
            take(a.y, 4 * i + 1);
            take(a.z, 4 * i + 2);
            take(a.w, 4 * i + 3);
        }
        tail = 4 * n4;
    }
    for (int i = tail + threadIdx.x; i < n; i += B) take(x[i], i);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (int) (blockDim.x >> 5); ++i)
            if (sv[i] > best || (sv[i] == best && si[i] < bi)) { best = sv[i]; bi = si[i]; }
        *out = map != nullptr && bi >= map_from ? map[bi - map_from] : bi;
    }
}

int mv_type_ok(int t) {
    switch (t) {
        case kTypeF32: case kTypeBF16: case 12: case 13: case 14: case 8: case 16: case 18: case 19: case 23: case 10: case 11:
        case 17: case 22: case 21: case 29: return 1;
        case exl3::kTypeEXL3: return 1;   // forwarded to the EXL3 kernels (mv below)
        default: return 0;
    }
}

// a failed launch is STICKY: the forward that issued it reports an error (launch_errors) instead of computing on
std::atomic<int> g_launch_errors{0};
void launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_fast %s: %s\n", what, cudaGetErrorString(e));
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace

bool mv_supported(int type) { return mv_type_ok(type) != 0; }

bool moe_supported(int type) {   // moe_gate_up's and moe_down's instantiations
    return type == 8 || type == 10 || type == 11 || type == 12 || type == 13 || type == 14 || type == 16 || type == 17 ||
           type == 18 || type == 19 || type == 21 || type == 22 || type == 23 || type == 29;
}

int launch_errors() { return g_launch_errors.load(std::memory_order_relaxed) + exl3::launch_errors(); }

size_t row_bytes(int type, int64_t n_in) {
    switch (type) {
        case kTypeF32: return (size_t) n_in * 4;
        case kTypeBF16: return (size_t) n_in * 2;
        case 12: return (size_t) (n_in / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n_in / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n_in / 256) * sizeof(block_q6_K);
        case 8: return (size_t) (n_in / 32) * sizeof(block_q8_0);
        case 16: return (size_t) (n_in / 256) * sizeof(block_iq2_xxs);
        case 18: return (size_t) (n_in / 256) * sizeof(block_iq3_xxs);
        case 19: return (size_t) (n_in / 256) * sizeof(block_iq1_s);
        case 23: return (size_t) (n_in / 256) * sizeof(block_iq4_xs);
        case 10: return (size_t) (n_in / 256) * sizeof(block_q2_K);
        case 11: return (size_t) (n_in / 256) * sizeof(block_q3_K);
        case 17: return (size_t) (n_in / 256) * sizeof(block_iq2_xs);
        case 22: return (size_t) (n_in / 256) * sizeof(block_iq2_s);
        case 21: return (size_t) (n_in / 256) * sizeof(block_iq3_s);
        case 29: return (size_t) (n_in / 256) * sizeof(block_iq1_m);
        default: return 0;
    }
}

#if defined(STRATA_USE_HIP)
namespace {
// The RDNA3 GEMV's shape: R rows per wave, U super-blocks in flight, WPB waves per block, the Q6_K activations in LDS
// or not.  The bench sets it (hip_expert_bench::mv_config); r == 0 is the original kernels, r < 0 the default: one row
// per wave, two super-blocks in flight, 8 waves - the one shape never slower than the originals at any decode shape
// on a Radeon 8065S (glm_mv_bench --sweep: -4..-17% per batch, the head even).  More rows per wave (the activation
// shared) or U = 4 ran slower: the extra VGPRs cost occupancy, and the latency is hidden by waves, not by registers.
// LDS (lds < 0: auto) only where it measured a win: one row (mv) with K <= 4096 - the small batches (the DSA
// projections 52.5 -> 46.1 us, the router + shared expert 82.2 -> 77.7); K = 12288 / 16384 ran slower with it (the
// 24-48 KB of LDS per block cost occupancy).
std::atomic<int> g_mv_r{-1}, g_mv_u{2}, g_mv_wpb{8}, g_mv_lds{-1};
// The device the launch runs on: by default the RDNA 3.5 APUs (gfx115x - measured on a Radeon 8065S), whose LPDDR5X
// latency the in-flight loads hide; STRATA_GLM_MV_RDNA=1 on any gfx11 (an RX 7900's GDDR6: not measured yet),
// STRATA_GLM_MV_RDNA=0 keeps mv_kernel_t and the generic kernel (A/B)
bool mv_rdna_device() {
    static const int mode = [] {   // -1: default, 0: off, 1: every gfx11
        const char* v = std::getenv("STRATA_GLM_MV_RDNA");
        return v == nullptr ? -1 : v[0] == '0' ? 0 : 1;
    }();
    if (mode == 0 || g_mv_r.load(std::memory_order_relaxed) == 0) return false;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return false;
    static thread_local int cached_device = -1;
    static thread_local bool on = false;
    if (cached_device != device) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
        on = std::strncmp(prop.gcnArchName, mode > 0 ? "gfx11" : "gfx115", mode > 0 ? 5 : 6) == 0;
        cached_device = device;
    }
    return on;
}
template<int NT, int R, int U>
void mv_rdna_launch_w(const MvBatch& b, int blocks, int wpb, int lds, cudaStream_t s) {
    if (lds > 0) {
        if (wpb == 4) mv_rdna_kernel<R, NT, U, 4, true><<<blocks, 128, lds, s>>>(b);
        else mv_rdna_kernel<R, NT, U, 8, true><<<blocks, 256, lds, s>>>(b);
    } else {
        if (wpb == 4) mv_rdna_kernel<R, NT, U, 4, false><<<blocks, 128, 0, s>>>(b);
        else mv_rdna_kernel<R, NT, U, 8, false><<<blocks, 256, 0, s>>>(b);
    }
}
template<int NT>
void mv_rdna_launch(const MvBatch& b, int blocks, int r, int u, int wpb, int lds, cudaStream_t s) {
    if constexpr (NT == 1) {
        switch (r * 10 + u) {
            case 11: mv_rdna_launch_w<NT, 1, 1>(b, blocks, wpb, lds, s); break;
            case 12: mv_rdna_launch_w<NT, 1, 2>(b, blocks, wpb, lds, s); break;
            case 14: mv_rdna_launch_w<NT, 1, 4>(b, blocks, wpb, lds, s); break;
            case 21: mv_rdna_launch_w<NT, 2, 1>(b, blocks, wpb, lds, s); break;
            case 22: mv_rdna_launch_w<NT, 2, 2>(b, blocks, wpb, lds, s); break;
            case 24: mv_rdna_launch_w<NT, 2, 4>(b, blocks, wpb, lds, s); break;
            case 41: mv_rdna_launch_w<NT, 4, 1>(b, blocks, wpb, lds, s); break;
            case 42: mv_rdna_launch_w<NT, 4, 2>(b, blocks, wpb, lds, s); break;
            default: mv_rdna_launch_w<NT, 4, 4>(b, blocks, wpb, lds, s); break;
        }
    } else {
        switch (r * 10 + u) {
            case 11: mv_rdna_launch_w<NT, 1, 1>(b, blocks, wpb, lds, s); break;
            case 12: mv_rdna_launch_w<NT, 1, 2>(b, blocks, wpb, lds, s); break;
            case 21: mv_rdna_launch_w<NT, 2, 1>(b, blocks, wpb, lds, s); break;
            default: mv_rdna_launch_w<NT, 1, 4>(b, blocks, wpb, lds, s); break;
        }
    }
}
// The batch's Q6_K and BF16 jobs in one launch (BF16 first); marks them done.  False: not this device.
bool mv_rdna(const MvJob* jobs, int n, int nt, bool* done, cudaStream_t s) {
    if (!mv_rdna_device()) return false;
    int r = g_mv_r.load(std::memory_order_relaxed), u = g_mv_u.load(std::memory_order_relaxed);
    const int wpb = g_mv_wpb.load(std::memory_order_relaxed) == 4 ? 4 : 8;
    if (r < 0) { r = 1; u = 2; }
    if (nt > 1 && r > 2) r = 2;
    MvBatch b{};
    int acc = 0, m = 0, max_in = 0;
    for (const int T : {kTypeBF16, 14})
        for (int i = 0; i < n; ++i)
            if (jobs[i].type == T) {
                b.j[m] = jobs[i];
                acc += (jobs[i].n_out + (T == 14 ? wpb * r : wpb) - 1) / (T == 14 ? wpb * r : wpb);
                b.blk_end[m++] = acc;
                done[i] = true;
                if (T == 14) max_in = std::max(max_in, jobs[i].n_in);
            }
    if (m == 0) return true;
    b.n = m;
    const int lds_mode = g_mv_lds.load(std::memory_order_relaxed);
    const bool use_lds = lds_mode > 0 || (lds_mode < 0 && nt == 1 && max_in <= 4096);
    int lds = use_lds ? nt * (max_in / 32) * (int) sizeof(block_q8_1) : 0;
    if (lds > 48 * 1024) lds = 0;
    switch (nt) {
        case 1: mv_rdna_launch<1>(b, acc, r, u, wpb, lds, s); break;
        case 2: mv_rdna_launch<2>(b, acc, r, u, wpb, lds, s); break;
        case 3: mv_rdna_launch<3>(b, acc, r, u, wpb, lds, s); break;
        case 4: mv_rdna_launch<4>(b, acc, r, u, wpb, lds, s); break;
        case 5: mv_rdna_launch<5>(b, acc, r, u, wpb, lds, s); break;
        case 6: mv_rdna_launch<6>(b, acc, r, u, wpb, lds, s); break;
        case 7: mv_rdna_launch<7>(b, acc, r, u, wpb, lds, s); break;
        default: mv_rdna_launch<8>(b, acc, r, u, wpb, lds, s); break;
    }
    return true;
}
}  // namespace
namespace hip_expert_bench {
void mv_config(int r, int u, int wpb, int lds) {
    g_mv_r.store(r);
    g_mv_u.store(u);
    g_mv_wpb.store(wpb);
    g_mv_lds.store(lds);
}
}  // namespace hip_expert_bench
#endif

// A batch's EXL3 jobs (w: an exl3::Mat) through exl3::mv, marked done, in launches of up to exl3::kMaxMvJobs - nt > 0:
// nt activations one after another (mv_nt's layout), else each job's ncol at its strides; a job's activations (a
// verify window's tokens) stay one EXL3 job of up to exl3::kMaxMvCols, its weights decoded once for several of them
static bool exl3_jobs(const MvJob* jobs, int n, int nt, bool* done, cudaStream_t s) {
    exl3::MvJob xj[exl3::kMaxMvJobs];
    int nx = 0;
    for (int i = 0; i < n; ++i) {
        const MvJob& j = jobs[i];
        if (j.type != exl3::kTypeEXL3) continue;
        const int ncol = nt > 0 ? nt : j.ncol;
        const int64_t xq_cs = nt > 0 ? (int64_t) (j.n_in / 32) * (int64_t) sizeof(block_q8_1) : j.xq_cstride;
        const int64_t xf_cs = nt > 0 ? j.n_in : j.xf_cstride;
        const int64_t y_cs = nt > 0 ? j.n_out : j.y_cstride;
        for (int c = 0; c < ncol; c += exl3::kMaxMvCols) {
            exl3::MvJob& x = xj[nx++];
            x.m = (const exl3::Mat*) j.w;
            x.xf = j.xf != nullptr ? j.xf + c * xf_cs : nullptr;
            x.xq = j.xq != nullptr ? (const void*) ((const uint8_t*) j.xq + c * xq_cs) : nullptr;
            x.y = j.y + c * y_cs;
            x.bias = j.bias;
            x.alpha = j.alpha;
            x.ncol = std::min(exl3::kMaxMvCols, ncol - c);
            x.xf_cs = (size_t) xf_cs;
            x.xq_cs = (size_t) xq_cs;
            x.y_cs = (size_t) y_cs;
            if (nx == exl3::kMaxMvJobs) {
                if (!exl3::mv(xj, nx, s)) return false;
                nx = 0;
            }
        }
        done[i] = true;
    }
    return nx == 0 || exl3::mv(xj, nx, s);
}

// A job without the activations its weight type reads - F32 / BF16 the floats (xf), the quantized types the q8_1
// (xq), EXL3 either - is refused: its kernel would read from address 0 (issue #173: a BF16 DFlash2 drafter's GEMVs
// carried no xf, and its first draft faulted the device for good)
static bool mv_inputs_ok(const MvJob* jobs, int n, const char* who) {
    for (int i = 0; i < n; ++i) {
        const MvJob& j = jobs[i];
        const bool fl = j.type == kTypeF32 || j.type == kTypeBF16;
        if (j.type == exl3::kTypeEXL3 ? j.xf != nullptr || j.xq != nullptr : fl ? j.xf != nullptr : j.xq != nullptr)
            continue;
        std::fprintf(stderr, "glm_fast %s: a type %d job without its %s activations\n", who, j.type,
                     fl ? "float" : "q8_1");
        return false;
    }
    return true;
}

bool mv(const MvJob* jobs, int n, cudaStream_t s) {
    if (n <= 0 || n > kMaxMvJobs) return false;
    for (int i = 0; i < n; ++i)
        if (jobs[i].ncol < 1 || jobs[i].ncol > kMaxMvCols) {
            std::fprintf(stderr, "glm_fast mv: %d columns (1..%d)\n", jobs[i].ncol, kMaxMvCols);
            return false;
        }
    for (int i = 0; i < n; ++i)
        if (!mv_type_ok(jobs[i].type)) {
            std::fprintf(stderr, "glm_fast mv: type %d unsupported\n", jobs[i].type);
            return false;
        }
    if (!mv_inputs_ok(jobs, n, "mv")) return false;
    // the jobs of the common dense types go to a kernel compiled for their type (one launch per type present), the
    // rest to the generic one; the jobs are independent, so their order does not matter
    static const bool typed = getenv("STRATA_GLM_MV_GENERIC") == nullptr;
    constexpr int kTyped[] = {14, 8, 12, 13};
    bool done[kMaxMvJobs] = {};
    // EXL3 weights (w: an exl3::Mat): their own kernels, F32 input when the job has it, else the q8_1 - a job of
    // several columns (a verify window's tokens) as one EXL3 job a column
    if (!exl3_jobs(jobs, n, 0, done, s)) return false;
    {
        bool all = true;
        for (int i = 0; i < n; ++i) all = all && done[i];
        if (all) return true;
    }
#if defined(STRATA_USE_HIP)
    // (one-column batches only: a verify window's several-column jobs go to the tiled kernels below)
    bool one_col = true;
    for (int i = 0; i < n; ++i) one_col = one_col && jobs[i].ncol == 1;
    if (typed && one_col) {
        mv_rdna(jobs, n, 1, done, s);
    } else if (typed && mv_rdna_device()) {
        // a verify window's several-column jobs whose columns sit one after another (mv_nt's layout): the RDNA3 kernel
        // takes them as NT activations, each column bit for bit its one-column result - the token path's on the APUs
        // (the MTP decode's window on the Strix / Gorgon Halo).  The rest go to the tiled kernels below.
        MvJob sub[kMaxMvJobs];
        int idx[kMaxMvJobs], m = 0, nc = 0;
        for (int i = 0; i < n; ++i) {
            const MvJob& J = jobs[i];
            const bool packed = J.ncol >= 2 && J.ncol <= kMaxMvNt && (J.type == kTypeBF16 || J.type == 14) &&
                                (J.xq == nullptr || J.xq_cstride == (int64_t) (J.n_in / 32) * (int64_t) sizeof(block_q8_1)) &&
                                (J.xf == nullptr || J.xf_cstride == J.n_in) && J.y_cstride == J.n_out;
            if (!done[i] && packed && (nc == 0 || J.ncol == nc)) {
                nc = J.ncol;
                sub[m] = J;
                sub[m].ncol = 1;
                idx[m++] = i;
            }
        }
        if (m > 0) {
            bool sub_done[kMaxMvJobs] = {};
            mv_rdna(sub, m, nc, sub_done, s);
            for (int k = 0; k < m; ++k)
                if (sub_done[k]) done[idx[k]] = true;
        }
    }
#endif
    // several columns: the tiled kernels (STRATA_GLM_MV_TILED=0: the column loops)
    static const bool tiled = [] {
        const char* v = getenv("STRATA_GLM_MV_TILED");
        return v == nullptr || std::atoi(v) != 0;
    }();
    // five or more columns of Q4_K / Q8_0 that may differ in float rounding (a drafter's): the tensor cores.  Below five
    // the dp4a kernel is as fast or faster (Q4_K, 216 MB from DRAM, us at 2 / 4 / 8 columns - RTX 3090, 4096 in: dp4a
    // 321 / 527 / 674, mma 295 / 352 / 366; 12288 in: 335 / 524 / 1045 against 471 / 498 / 516; RTX 5060 Ti: 546 / 552 /
    // 926 against 609 / 634 / 574).  Q8_0 at 8 columns: 3090 2312 against 1133, but 5060 Ti 1625 against 1725 (dp4a at
    // the memory's speed there) - a caller that may run on either times both (GlmDflash's head)
    if (tiled && mma_arch()) {
        for (const int T : {12, 8}) {
            MvBatch b{};
            b.skip = g_skip;
            int acc = 0, m = 0;
            for (int i = 0; i < n; ++i)
                if (!done[i] && jobs[i].approx && jobs[i].type == T && jobs[i].ncol >= 5 &&
                    jobs[i].n_in % (T == 12 ? 256 : 32) == 0) {
                    b.j[m] = jobs[i];
                    acc += (jobs[i].n_out + 15) / 16;
                    b.blk_end[m++] = acc;
                    done[i] = true;
                }
            if (m == 0) continue;
            b.n = m;
            if (T == 12)
                mvc_mma_q4_K<<<acc, MMA_W * 32, 0, s>>>(b);
            else
                mvc_mma_q8_0<<<acc, MMA_W * 32, 0, s>>>(b);
        }
    }
    if (tiled) {
        // Q8_0 on the tensor cores only when asked (STRATA_GLM_MV_MMA=1; it is not bit for bit the single column's)
        const bool mma = mma_ok();
        for (const int T : {8, 12, 14, (int) kTypeBF16}) {
            MvBatch b{};
            b.skip = g_skip;
            int acc = 0, m = 0, nc = 1;
            const bool use_mma = T == 8 && mma;
            for (int i = 0; i < n; ++i)
                if (!done[i] && jobs[i].type == T && jobs[i].ncol > 1) nc = std::max(nc, jobs[i].ncol);
            const int nc_t = nc <= 2 ? 2 : nc <= 4 ? 4 : nc <= 8 ? 8 : 16;   // mvc_launch's instantiation
            const int per = T == kTypeBF16 ? MV_ROWS : use_mma ? 16 : MV_ROWS * mvc_rows(T, nc_t);   // rows a block
            for (int i = 0; i < n; ++i)
                if (!done[i] && jobs[i].type == T && jobs[i].ncol > 1) {
                    b.j[m] = jobs[i];
                    acc += (jobs[i].n_out + per - 1) / per;
                    b.blk_end[m++] = acc;
                    done[i] = true;
                }
            if (m == 0) continue;
            b.n = m;
            if (use_mma)
                mvc_mma_q8_0<<<acc, MMA_W * 32, 0, s>>>(b);
            else if (T == 8)
                mvc_launch<8>(b, acc, nc, s);
            else if (T == 12)
                mvc_launch<12>(b, acc, nc, s);
            else if (T == 14)
                mvc_launch<14>(b, acc, nc, s);
            else
                mvc_kernel_bf16<<<acc, 256, (size_t) nc * MVC_FTILE * sizeof(float), s>>>(b);
        }
    }
    // several columns the tiled kernels left (another type, or STRATA_GLM_MV_TILED=0): the column kernels
    for (const int T : kTyped) {
        if (!typed) break;
        MvBatch b{};
        b.skip = g_skip;
        int acc = 0, m = 0;
        for (int i = 0; i < n; ++i)
            if (!done[i] && jobs[i].ncol > 1 && jobs[i].type == T) {
                b.j[m] = jobs[i];
                acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
                b.blk_end[m++] = acc;
                done[i] = true;
            }
        if (m == 0) continue;
        b.n = m;
        switch (T) {
            case 14: mv_kernel_t_cols<14><<<acc, 256, 0, s>>>(b); break;
            case 8: mv_kernel_t_cols<8><<<acc, 256, 0, s>>>(b); break;
            case 12: mv_kernel_t_cols<12><<<acc, 256, 0, s>>>(b); break;
            case 13: mv_kernel_t_cols<13><<<acc, 256, 0, s>>>(b); break;
        }
    }
    {
        MvBatch b{};
        b.skip = g_skip;
        int acc = 0, m = 0;
        for (int i = 0; i < n; ++i)
            if (!done[i] && jobs[i].ncol > 1) {
                b.j[m] = jobs[i];
                acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
                b.blk_end[m++] = acc;
                done[i] = true;
            }
        if (m > 0) {
            b.n = m;
            mv_kernel_cols<<<acc, 256, 0, s>>>(b);
        }
    }
    // one column
    if (typed) {
        for (const int T : kTyped) {
            MvBatch b{};
            b.skip = g_skip;
            int acc = 0, m = 0;
            // cc 6.0: Q6_K through mv_q6_p100_kernel - R = 2 rows a warp below 4096 rows in the launch, else 4 (a
            // launch of fewer than 1024 rows, and an activation over 48 KB, keep mv_kernel_t)
            int pr = 0;
            size_t xs_bytes = 0;
#if !defined(STRATA_USE_HIP)
            if (T == 14) {
                if (const int mode = p100_mv_mode(); mode > 0) {
                    int rows = 0, max_in = 0;
                    for (int i = 0; i < n; ++i)
                        if (!done[i] && jobs[i].type == T) {
                            rows += jobs[i].n_out;
                            max_in = std::max(max_in, jobs[i].n_in);
                        }
                    xs_bytes = (size_t) (max_in / 32) * sizeof(block_q8_1);
                    if (rows >= 1024 && xs_bytes <= 48 * 1024) pr = mode > 1 ? mode : rows < 4096 ? 2 : 4;
                }
            }
#endif
            const int per = MV_ROWS * (pr > 0 ? pr : 1);
            for (int i = 0; i < n; ++i)
                if (!done[i] && jobs[i].type == T) {
                    b.j[m] = jobs[i];
                    acc += (jobs[i].n_out + per - 1) / per;
                    b.blk_end[m++] = acc;
                    done[i] = true;
                }
            if (m == 0) continue;
            b.n = m;
#if !defined(STRATA_USE_HIP)
            if (pr > 0) {
                if (pr == 2) mv_q6_p100_kernel<2><<<acc, 256, xs_bytes, s>>>(b);
                else mv_q6_p100_kernel<4><<<acc, 256, xs_bytes, s>>>(b);
                continue;
            }
#endif
            switch (T) {
                case 14: mv_kernel_t<14><<<acc, 256, 0, s>>>(b); break;
                case 8: mv_kernel_t<8><<<acc, 256, 0, s>>>(b); break;
                case 12: mv_kernel_t<12><<<acc, 256, 0, s>>>(b); break;
                case 13: mv_kernel_t<13><<<acc, 256, 0, s>>>(b); break;
            }
        }
    }
    MvBatch b{};
    b.skip = g_skip;
    int acc = 0, m = 0;
    for (int i = 0; i < n; ++i) {
        if (done[i]) continue;
        b.j[m] = jobs[i];
        acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
        b.blk_end[m++] = acc;
    }
    if (m > 0) {
        b.n = m;
        mv_kernel<<<acc, 256, 0, s>>>(b);
    }
    launch_check("mv");
    return true;
}

namespace {
template<int NT>
void mv_nt_launch(const MvJob* jobs, int n, bool typed, bool* done, cudaStream_t s) {
    constexpr int kTyped[] = {14, 8, 12, 13};
    if (typed) {
        for (const int T : kTyped) {
            MvBatch b{};
            int acc = 0, m = 0;
            for (int i = 0; i < n; ++i)
                if (!done[i] && jobs[i].type == T) {
                    b.j[m] = jobs[i];
                    acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
                    b.blk_end[m++] = acc;
                    done[i] = true;
                }
            if (m == 0) continue;
            b.n = m;
            switch (T) {
                case 14: mv_kernel_t_nt<14, NT><<<acc, 256, 0, s>>>(b); break;
                case 8: mv_kernel_t_nt<8, NT><<<acc, 256, 0, s>>>(b); break;
                case 12: mv_kernel_t_nt<12, NT><<<acc, 256, 0, s>>>(b); break;
                case 13: mv_kernel_t_nt<13, NT><<<acc, 256, 0, s>>>(b); break;
            }
        }
    }
    MvBatch b{};
    int acc = 0, m = 0;
    for (int i = 0; i < n; ++i) {
        if (done[i]) continue;
        b.j[m] = jobs[i];
        acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
        b.blk_end[m++] = acc;
    }
    if (m > 0) {
        b.n = m;
        mv_kernel_nt<NT><<<acc, 256, 0, s>>>(b);
    }
}
}  // namespace

bool mv_nt(const MvJob* jobs, int n, int nt, cudaStream_t s) {
    if (nt <= 1) return mv(jobs, n, s);
    if (n <= 0 || n > kMaxMvJobs || nt > kMaxMvNt) return false;
    for (int i = 0; i < n; ++i)
        if (!mv_type_ok(jobs[i].type) || jobs[i].ncol != 1) {
            std::fprintf(stderr, "glm_fast mv_nt: type %d / %d columns unsupported\n", jobs[i].type, jobs[i].ncol);
            return false;
        }
    if (!mv_inputs_ok(jobs, n, "mv_nt")) return false;
    static const bool typed = getenv("STRATA_GLM_MV_GENERIC") == nullptr;
    bool done[kMaxMvJobs] = {};
    if (!exl3_jobs(jobs, n, nt, done, s)) return false;   // (EXL3: one EXL3 job an activation)
#if defined(STRATA_USE_HIP)
    if (typed) mv_rdna(jobs, n, nt, done, s);   // (RDNA3: its Q6_K and BF16 kernels take NT activations already)
#endif
    switch (nt) {
        case 2: mv_nt_launch<2>(jobs, n, typed, done, s); break;
        case 3: mv_nt_launch<3>(jobs, n, typed, done, s); break;
        case 4: mv_nt_launch<4>(jobs, n, typed, done, s); break;
        case 5: mv_nt_launch<5>(jobs, n, typed, done, s); break;
        case 6: mv_nt_launch<6>(jobs, n, typed, done, s); break;
        case 7: mv_nt_launch<7>(jobs, n, typed, done, s); break;
        default: mv_nt_launch<8>(jobs, n, typed, done, s); break;
    }
    launch_check("mv_nt");
    return true;
}

void quantize_q8_1(const float* x, void* xq, int n, cudaStream_t s) {
    quantize_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, (block_q8_1*) xq, n);
    launch_check("quantize");
}

void hc(const HcArgs& a, cudaStream_t s) {
    // the grid-barrier kernel needs every block resident at once; checked once per process (and per device count
    // of SMs), the single-tail kernel otherwise.  STRATA_GLM_HC1=1 forces the old one (A/B).
    static int mode = -1;
    if (mode < 0) {
        int dev = 0, sms = 0, per = 0, per3 = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, hc2_kernel, HC_THREADS, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per3, hc3_kernel, HC_THREADS, 0);
        mode = (getenv("STRATA_GLM_HC1") == nullptr && per * sms >= a.n_embd / HC2_EB) ? 2 : 1;
        if (mode == 2 && getenv("STRATA_GLM_HC2") == nullptr && a.n_embd == 4096 && per3 * sms >= a.n_embd / HC2_EB)
            mode = 3;
    }
    if (mode == 3) {
        hc3_kernel<<<a.n_embd / HC2_EB, HC_THREADS, 0, s>>>(a);
        launch_check("hc3");
        return;
    }
    if (mode == 2 && a.n_embd % HC2_EB == 0) {
        hc2_kernel<<<a.n_embd / HC2_EB, HC_THREADS, 0, s>>>(a);
        launch_check("hc2");
        return;
    }
    const int blocks = 4 * a.n_embd / HC_CHUNK;
    hc_kernel<<<blocks, HC_THREADS, 0, s>>>(a);
    launch_check("hc");
}

void hc_post(const float* block_out, const float* R_old, const float* post, const float* comb, int n_embd,
             float* R_new, cudaStream_t s) {
    hc_post_kernel<<<(4 * n_embd + 255) / 256, 256, 0, s>>>(block_out, R_old, post, comb, n_embd, R_new);
    launch_check("hc_post");
}

void head_prep(const float* R, const float* norm_w, float eps, int n_embd, float* x, void* xq, cudaStream_t s) {
    head_prep_kernel<<<1, 1024, 0, s>>>(R, norm_w, eps, n_embd, x, (block_q8_1*) xq);
    launch_check("head_prep");
}

void embed_streams(const float* emb, float* R, int n_embd, cudaStream_t s) {
    embed_streams_kernel<<<(4 * n_embd + 255) / 256, 256, 0, s>>>(emb, R, n_embd);
    launch_check("embed_streams");
}

void kda_prep(const KdaPrepArgs& a, cudaStream_t s) {
    kda_prep_kernel<<<dim3(a.n_head, 11), 128, 0, s>>>(a);   // 3 conv roles + 2 x 4 gate quarters
    launch_check("kda_prep");
}

void kda_rec(const float* q, const float* k, const float* v, const float* g1, const float* beta_raw, float* state,
             const float* g2, const float* norm_w, float eps, int n_head, int head_dim, void* out_q8_1,
             cudaStream_t s) {
    if (head_dim != 128) {
        std::fprintf(stderr, "glm_fast kda_rec: head_dim %d (the kernel is built for 128)\n", head_dim);
        return;
    }
    kda_rec_kernel<<<n_head, 256, 0, s>>>(q, k, v, g1, beta_raw, state, g2, norm_w, eps, n_head,
                                          (block_q8_1*) out_q8_1);
    launch_check("kda_rec");
}

void kda_layers(const KdaPrepArgs* prep, const KdaRecArgs* rec, int n, float eps, cudaStream_t s) {
    if (n <= 0) return;
    if (n > kKdaLayers || prep[0].head_dim != 128) {
        std::fprintf(stderr, "glm_fast kda_layers: %d layers, head_dim %d (at most %d, 128)\n", n, prep[0].head_dim,
                     kKdaLayers);
        return;
    }
    KdaPrepBatch pb;
    KdaRecBatch rb;
    for (int i = 0; i < n; ++i) {
        pb.a[i] = prep[i];
        rb.a[i] = rec[i];
    }
    rb.eps = eps;
    kda_prep_layers_kernel<<<dim3(prep[0].n_head, 11, n), 128, 0, s>>>(pb);
    kda_rec_layers_kernel<<<dim3(prep[0].n_head, n), 256, 0, s>>>(rb);
    launch_check("kda_layers");
}

void dsa_prep(const DsaPrepArgs& a, cudaStream_t s) {
    dsa_prep_kernel<<<3, 512, 0, s>>>(a);
    launch_check("dsa_prep");
}

void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int n_pools,
               float* score, cudaStream_t s) {
    if (n_pools <= 0) return;
    const size_t smem = ((size_t) idx_heads * key_dim + idx_heads) * sizeof(float);
    dsa_score_kernel<<<(n_pools + 7) / 8, 256, smem, s>>>(iq, pooled, iw, key_dim, idx_heads, n_pools, score);
    launch_check("dsa_score");
}

void dsa_select(const float* score, int n_vis, int kpool, int top_pools, int n_sel, int pos, int* cells,
                cudaStream_t s) {
    dsa_select_kernel<<<1, 1024, 0, s>>>(score, n_vis, kpool, top_pools, n_sel, pos, cells);
    launch_check("dsa_select");
}

// every pool's cells in pool order, then the tail (dsa_select's when it takes all n_vis pools - the same cells, in
// position order instead of the scores')
__global__ void dsa_select_all_kernel(int n_vis, int kpool, int n_sel, int pos, int* __restrict__ cells) {
    const int full = n_vis * kpool;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_sel; i += gridDim.x * blockDim.x)
        cells[i] = i < full || (i - full < kpool - 1 && i <= pos) ? i : -1;
}

bool dsa_select_all_on() {
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_DSA_ALL");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

void dsa_select_all(int n_vis, int kpool, int n_sel, int pos, int* cells, cudaStream_t s) {
    dsa_select_all_kernel<<<(n_sel + 255) / 256, 256, 0, s>>>(n_vis, kpool, n_sel, pos, cells);
    launch_check("dsa_select_all");
}

// the split attention of one token: q_abs (n_head x kv_lora) over its selected cells -> ctx; false when the split
// kernels are not available on this device (mla() then runs its one-kernel form)
static bool mla_split_attend(const float* q_abs, const uint16_t* lat, const int* cells, int n_sel, int n_head,
                             int qk_nope, int kv_lora, float* ctx, cudaStream_t s, bool lat_q8);

void mla_q_abs(const float* q, int64_t q_stride, const uint16_t* wk_b, int n_head, int qk_nope, int kv_lora, int ncol,
               float* q_abs, int64_t qa_stride, cudaStream_t s) {
    headwise_gemv_cols_kernel<<<n_head * kv_lora / 32, 256, 0, s>>>(wk_b, q, q_stride, kv_lora, qk_nope, ncol, q_abs,
                                                                    qa_stride, nullptr, 0);
    launch_check("mla_q_abs");
}

bool mla_attend(const float* q_abs, const uint16_t* lat, const int* cells, int n_sel, int n_head, int qk_nope,
                int kv_lora, float* ctx, cudaStream_t s, bool lat_q8) {
    if (kv_lora % 32 != 0 || getenv("STRATA_GLM_MLA1") != nullptr) return false;
    return mla_split_attend(q_abs, lat, cells, n_sel, n_head, qk_nope, kv_lora, ctx, s, lat_q8);
}

void mla_out(const float* ctx, int64_t ctx_stride, const uint16_t* wv_b, int n_head, int kv_lora, int v_head, int ncol,
             void* out_q8_1, int64_t out_stride_bytes, cudaStream_t s) {
    headwise_gemv_cols_kernel<<<n_head * v_head / 32, 256, 0, s>>>(wv_b, ctx, ctx_stride, v_head, kv_lora, ncol, nullptr,
                                                                   0, (block_q8_1*) out_q8_1,
                                                                   out_stride_bytes / (int64_t) sizeof(block_q8_1));
    launch_check("mla_out");
}

// the split attention's partials buffer of this device, at least `chunks` (chunk, n_head) slots; nullptr: not allocated
static float* mla_part(int dev, int n_head, int kv_lora, int chunks, int* cap) {
    static float* part[16] = {};
    static int part_chunks[16] = {};
    if (part_chunks[dev] < chunks) {
        if (part[dev]) cudaFree(part[dev]);
        const int want = std::max(chunks, 80);
        if (cudaMalloc(&part[dev], (size_t) want * n_head * (kv_lora + 2) * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            part[dev] = nullptr;
            part_chunks[dev] = 0;
        } else {
            part_chunks[dev] = want;
        }
    }
    *cap = part_chunks[dev];
    return part[dev];
}

// whether this device takes the tensor-core attention (NVIDIA sm_80+); STRATA_GLM_MLA_MMA=0: the F32 chunks (A/B)
static bool mla_mma_on(int dev) {
    static int on_[16] = {};
    if (on_[dev] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        const char* v = getenv("STRATA_GLM_MLA_MMA");
        bool on = major >= 8 && (v == nullptr || std::atoi(v) != 0) && getenv("STRATA_GLM_MLA_ATTN1") == nullptr;
#if defined(STRATA_USE_HIP)
        on = false;
#endif
        if (on && cudaFuncSetAttribute(mla_split_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int) MLA_MMA_SMEM) != cudaSuccess) {
            cudaGetLastError();
            on = false;
        }
        on_[dev] = on ? 1 : 2;
    }
    return on_[dev] == 1;
}

bool mla_attend_batch(const float* q_abs, int64_t q_stride, const uint16_t* lat, const int* cells, int64_t c_stride,
                      const int* n_sel, int T, int n_head, int qk_nope, int kv_lora, float* ctx, int64_t ctx_stride,
                      cudaStream_t s, bool lat_q8) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 16 || kv_lora != MLA_KV || T < 1 || getenv("STRATA_GLM_MLA1") != nullptr ||
        !mla_mma_on(dev))
        return false;
    int cap = 0;
    float* part = mla_part(dev, n_head, kv_lora, MLA_NC, &cap);
    if (part == nullptr) return false;
    // tokens a launch: their partials in the buffer at once
    const int G = std::max(1, std::min({T, cap / MLA_NC, kMlaTokMax}));
    for (int t0 = 0; t0 < T; t0 += G) {
        const int nt = std::min(G, T - t0);
        MlaToks tk{};
        for (int i = 0; i < nt; ++i) tk.n_sel[i] = n_sel[t0 + i];
        mla_split_mma_kernel<<<dim3(MLA_NC, (n_head + 15) / 16, nt), 128, MLA_MMA_SMEM, s>>>(
            q_abs + t0 * q_stride, q_stride, lat, cells + t0 * c_stride, c_stride, tk, n_head, qk_nope, part, lat_q8);
        mla_combine_wide_kernel<<<dim3(n_head, (kv_lora + 127) / 128, nt), 128, 0, s>>>(part, tk, n_head, kv_lora,
                                                                                         ctx + t0 * ctx_stride, ctx_stride);
    }
    launch_check("mla_attend_batch");
    return true;
}

static bool mla_split_attend(const float* q_abs, const uint16_t* lat, const int* cells, int n_sel, int n_head,
                             int qk_nope, int kv_lora, float* ctx, cudaStream_t s, bool lat_q8) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 16) return false;
    if (mla_attend_batch(q_abs, 0, lat, cells, 0, &n_sel, 1, n_head, qk_nope, kv_lora, ctx, 0, s, lat_q8)) return true;
    const int n_chunks = (n_sel + MLA_CHUNK - 1) / MLA_CHUNK;
    const size_t smem = (size_t) MLA_CHUNK * kv_lora * sizeof(uint16_t) + (size_t) MLA_HPB * MLA_CHUNK * sizeof(float);
    static int attr_smem[16] = {};
    static bool split_ok[16] = {true, true, true, true, true, true, true, true,
                                true, true, true, true, true, true, true, true};
    if (split_ok[dev] && attr_smem[dev] < (int) smem) {
        if (cudaFuncSetAttribute(mla_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) == cudaSuccess) {
            attr_smem[dev] = (int) smem;
        } else {
            cudaGetLastError();
            split_ok[dev] = false;
            std::fprintf(stderr, "glm_fast mla: the split kernel cannot get %zu bytes of shared memory - the one-pass "
                                 "attention instead\n", smem);
        }
    }
    int cap = 0;
    float* part = split_ok[dev] && kv_lora <= 512 && n_head <= 128 && getenv("STRATA_GLM_MLA_ATTN1") == nullptr
                      ? mla_part(dev, n_head, kv_lora, n_chunks, &cap)
                      : nullptr;
    if (part != nullptr && cap >= n_chunks) {
        mla_split_kernel<<<dim3(n_chunks, (n_head + MLA_HPB - 1) / MLA_HPB), 256, smem, s>>>(
            q_abs, lat, cells, n_sel, n_head, qk_nope, kv_lora, part, lat_q8);
        mla_combine_kernel<<<n_head, 256, 0, s>>>(part, n_chunks, n_head, kv_lora, ctx);
    } else {
        mla_attn_kernel<<<n_head, 256, ((size_t) kv_lora + n_sel) * sizeof(float), s>>>(q_abs, lat, cells, n_sel,
                                                                                        qk_nope, kv_lora, ctx, lat_q8);
    }
    launch_check("mla_attend");
    return true;
}

void mla(const float* q, const uint16_t* wk_b, const uint16_t* wv_b, const uint16_t* lat, const int* cells, int n_sel,
         int n_head, int qk_nope, int kv_lora, int v_head, void* out_q8_1, cudaStream_t s, bool lat_q8) {
    // scratch for q_abs and ctx (n_head * kv_lora each), per device
    static float* scratch[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (getenv("STRATA_GLM_MLA1") != nullptr || dev >= 16 || kv_lora % 32 || v_head % 32) {
        const size_t smem = ((size_t) qk_nope + 2 * kv_lora + v_head + n_sel) * sizeof(float);
        mla_kernel<<<n_head, 256, smem, s>>>(q, wk_b, wv_b, lat, cells, n_sel, qk_nope, kv_lora, v_head,
                                             (block_q8_1*) out_q8_1, lat_q8);
        launch_check("mla");
        return;
    }
    if (scratch[dev] == nullptr && cudaMalloc(&scratch[dev], (size_t) 2 * n_head * kv_lora * sizeof(float)) != cudaSuccess) {
        scratch[dev] = nullptr;
        cudaGetLastError();
        const size_t smem = ((size_t) qk_nope + 2 * kv_lora + v_head + n_sel) * sizeof(float);
        mla_kernel<<<n_head, 256, smem, s>>>(q, wk_b, wv_b, lat, cells, n_sel, qk_nope, kv_lora, v_head,
                                             (block_q8_1*) out_q8_1, lat_q8);
        launch_check("mla");
        return;
    }
    float* q_abs = scratch[dev];
    float* ctx = q_abs + (size_t) n_head * kv_lora;
    headwise_gemv_kernel<<<n_head * kv_lora / 32, 256, 0, s>>>(wk_b, q, kv_lora, qk_nope, q_abs, nullptr);
    // the split attention: chunks of MLA_CHUNK cells, partials per (chunk, head), then the merge (mla_split_attend)
    mla_split_attend(q_abs, lat, cells, n_sel, n_head, qk_nope, kv_lora, ctx, s, lat_q8);
    headwise_gemv_kernel<<<n_head * v_head / 32, 256, 0, s>>>(wv_b, ctx, v_head, kv_lora, nullptr,
                                                              (block_q8_1*) out_q8_1);
    launch_check("mla2");
}

void swiglu_q8(const float* gate, const float* up, float limit, int n, void* hq, cudaStream_t s) {
    swiglu_q8_kernel<<<(n + 255) / 256, 256, 0, s>>>(gate, up, limit, n, (block_q8_1*) hq);
    launch_check("swiglu_q8");
}

void moe_route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w, int layer,
               const float* x, int n_embd, const MoeDev& d, const float* sh_gate, const float* sh_up, float sh_limit,
               int n_ff_sh, void* sh_hq, cudaStream_t s, const float* pred_logits, const float* pred_bias,
               int max_prefetch, const float* ahead_logits, const float* const* ahead_bias, int n_ahead,
               int skip_from, unsigned long long cpu_plan, int promote_min, int pf_rank, int near_n, int win,
               unsigned int* scr_used, int win_n, int disk_skip_from, float disk_skip_w, float miss_skip_w) {
    RouteArgs a{logits, bias, n_expert, k, w_scale, norm_w ? 1 : 0, layer, x, n_embd, d,
                sh_gate, sh_up, sh_limit, n_ff_sh, (block_q8_1*) sh_hq, pred_logits, pred_bias, max_prefetch,
                ahead_logits, {}, 0, skip_from, d.cpu_seq != nullptr ? cpu_plan : 0ull, promote_min,
                pf_rank > 0 ? pf_rank : k, std::max(0, std::min(near_n, std::min(16, n_expert - k))), win, scr_used,
                g_skip, win_n, disk_skip_from, disk_skip_w, miss_skip_w};
    static const float affinity = [] {
        const char* v = getenv("STRATA_GLM_ROUTE_AFFINITY");
        return v ? std::max(0.0f, (float) std::atof(v)) : 0.0f;
    }();
    a.affinity = affinity;
    if (ahead_logits != nullptr && ahead_bias != nullptr && n_expert <= 512) {
        a.n_ahead = std::min(n_ahead, kAhead);
        for (int i = 0; i < a.n_ahead; ++i) a.ahead_bias[i] = ahead_bias[i];
    }
    const int blocks = 1 + (n_ff_sh + ROUTE_THREADS - 1) / ROUTE_THREADS;
    moe_route_kernel<<<blocks, ROUTE_THREADS, 0, s>>>(a);
    launch_check("moe_route");
}

void moe_wait(const MoeDev& d, int n_embd, cudaStream_t s) {
    moe_wait_kernel<<<1, 256, 0, s>>>(d, n_embd);
    launch_check("moe_wait");
}

void moe_cpu_wait_streams(const MoeDev& d, int n_embd, const float* post, float* R, int hc, cudaStream_t s) {
    if (d.cpu_seq == nullptr) return;
    moe_cpu_wait_streams_kernel<<<1, 1024, 0, s>>>(d, n_embd, post, R, hc);
    launch_check("moe_cpu_wait_streams");
}

void moe_cpu_wait(const MoeDev& d, int n_embd, float* out, cudaStream_t s) {
    if (d.cpu_seq == nullptr) return;
    moe_cpu_wait_kernel<<<1, 256, 0, s>>>(d, n_embd, out);
    launch_check("moe_cpu_wait");
}

void moe_prefetch(const MoeDev& d, size_t blob_bytes, cudaStream_t s) {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 80;
    }
    // half the SMs: it runs beside the main stream's kernels.  STRATA_GLM_PREFETCH_BLOCKS=<n>: n blocks instead - a
    // PCIe pull needs few SMs to keep the link busy, and every SM it holds is one the main stream's kernels wait for
    static const int blocks_env = [] {
        const char* v = getenv("STRATA_GLM_PREFETCH_BLOCKS");
        return v != nullptr ? std::max(1, std::atoi(v)) : 0;
    }();
    moe_prefetch_kernel<<<blocks_env > 0 ? blocks_env : std::max(1, sms / 2), 256, 0, s>>>(d, (blob_bytes + 15) / 16);
    launch_check("moe_prefetch");
}

void moe_fetch(const MoeDev& d, int k, size_t blob_bytes, cudaStream_t s) {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 80;
    }
    moe_fetch_kernel<<<sms, 256, 0, s>>>(d, k, (blob_bytes + 15) / 16);
    launch_check("moe_fetch");
}

#if defined(STRATA_USE_HIP)
namespace {
// Limit automatic dispatch to the measured decode geometry and RDNA3/3.5
// targets. RDNA4 and all other shapes retain the original expert kernels.
bool rdna3_expert_device(bool allow_gfx1151 = true) {
    static const bool legacy = [] {
        const char* v = std::getenv("STRATA_HIP_EXPERTS_LEGACY");
        return v && v[0] == '1';
    }();
    if (legacy) return false;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return false;
    static thread_local int cached_device = -1;
    static thread_local bool gfx1100 = false;
    static thread_local bool gfx1151 = false;
    if (cached_device != device) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
        gfx1100 = std::strncmp(prop.gcnArchName, "gfx1100", 7) == 0;
        gfx1151 = std::strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        cached_device = device;
    }
    return gfx1100 || (allow_gfx1151 && gfx1151);
}
// Splitting shared down removes its dynamic type switch from routed gate/up's
// register budget. The routed grid keeps exactly the original dot/reduction order.
__global__ void __launch_bounds__(256) rdna_shared_down_kernel(const uint8_t* weights, int type,
                                                         const block_q8_1* h, int n_in,
                                                         int n_out, float* out) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + wave;
    if (row < n_out) {
        const float v = any_row_dot(type, weights, row, h, nullptr, n_in, lane);
        if (lane == 0) out[row] = v;
    }
}
template<int T, int NW, bool GLOBAL = false, bool LUT = true, bool SPLIT = false,
         bool DIRECT = false, bool WIDE = false, int MINWAVES = 0>
void launch_expert_gu(const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq,
               const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s) {
    const bool shared = sh_down && sh_out;
    const dim3 grid(n_ff / GU_ROWS, k + (shared && !SPLIT ? 1 : 0));
    moe_gate_up_kernel<T, NW, LUT, GLOBAL, SPLIT, DIRECT, WIDE, MINWAVES><<<grid, NW * 32, 0, s>>>(d, shared ? k : k + 1, n_embd, n_ff,
        limit, (const block_q8_1*) xq, (block_q8_1*) hq, (const uint8_t*) sh_down, sh_type,
        (const block_q8_1*) sh_hq, n_ff_sh, sh_out);
    if constexpr (SPLIT) {
        if (shared) rdna_shared_down_kernel<<<(n_embd + 7) / 8, 256, 0, s>>>(
            (const uint8_t*) sh_down, sh_type, (const block_q8_1*) sh_hq, n_ff_sh, n_embd, sh_out);
    }
}
template<int T, int NR, bool GLOBAL = false, bool LUT = true, bool DIRECT = false, bool WIDE = false>
void launch_expert_down(const MoeDev& d, int k, int n_embd, int n_ff, size_t off, const void* hq,
               const float* sh, float* out, cudaStream_t s) {
    moe_down_kernel<T, NR, LUT, GLOBAL, DIRECT, WIDE><<<n_embd / NR, 256, 0, s>>>(d, k, n_embd, n_ff, off,
        (const block_q8_1*) hq, sh, out);
}
} // namespace
#endif

void moe_gate_up(int gu_type, const MoeDev& d_in, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq,
                 const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s) {
    MoeDev d = d_in;
    d.skip = g_skip;
#if defined(STRATA_USE_HIP)
    if (gu_type == 16 && n_embd == 4096 && n_ff == 2048 && rdna3_expert_device()) {
        // Packed sign expansion, original 16-wave/four-row geometry, grid in LDS.
        launch_expert_gu<16, 16, false, true, false, true>(d, k, n_embd, n_ff, limit,
            xq, hq, sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s);
        launch_check("moe_gate_up RDNA3");
        return;
    }
#endif
    const bool sh = sh_down != nullptr && sh_out != nullptr;
    const dim3 grid((unsigned) (n_ff / GU_ROWS), (unsigned) (k + (sh ? 1 : 0)));
    const auto* X = (const block_q8_1*) xq;
    auto* H = (block_q8_1*) hq;
    const auto* SD = (const uint8_t*) sh_down;
    const auto* SH = (const block_q8_1*) sh_hq;
    const int kk = sh ? k : k + 1;   // no shared expert: no block has blockIdx.y == kk
    // cc 6.0: the room to stage the layer input in shared memory (moe_gate_up_kernel takes it when given)
    const size_t gx = sm60_gu_stage() && n_embd <= 4 * 4096 ? (size_t) (n_embd / 32) * sizeof(block_q8_1) : 0;
    switch (gu_type) {
#define GLMF_GU(T) \
        case T: moe_gate_up_kernel<T><<<grid, GU_WARPS * 32, gx, s>>>(d, kk, n_embd, n_ff, limit, X, H, SD, sh_type, SH, \
                                                                      n_ff_sh, sh_out); break;
        GLMF_GU(16) GLMF_GU(18) GLMF_GU(19) GLMF_GU(23) GLMF_GU(10) GLMF_GU(11) GLMF_GU(17) GLMF_GU(22) GLMF_GU(21) GLMF_GU(29)
        GLMF_GU(12) GLMF_GU(13) GLMF_GU(14) GLMF_GU(8)
#undef GLMF_GU
        default: std::fprintf(stderr, "glm_fast moe_gate_up: type %d unsupported\n", gu_type); return;
    }
    launch_check("moe_gate_up");
}

#if defined(STRATA_USE_HIP)
namespace hip_expert_bench {

void gate_up(Variant v, int type, const MoeDev& d, int k, int n_embd, int n_ff, float limit,
             const void* xq, void* hq, const void* sh_down, int sh_type, const void* sh_hq,
             int n_ff_sh, float* sh_out, cudaStream_t s) {
    if (v == Variant::production) return moe_gate_up(type, d, k, n_embd, n_ff, limit, xq, hq,
                                                   sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s);
#define BENCH_GU(T) case T: \
    if (v == Variant::baseline) launch_expert_gu<T,16,false,false>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::direct_lds) launch_expert_gu<T,16,false,true,false,true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::direct_signs) launch_expert_gu<T,16,true,true,false,true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::vector_load) launch_expert_gu<T,16,true,true,false,false,true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::occupancy8) launch_expert_gu<T,16,true,true,true,false,false,8>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::split_shared) launch_expert_gu<T, 16, true, true, true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::global_plain) launch_expert_gu<T, 16, true, false>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::global_waves8) launch_expert_gu<T, 8, true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::waves8) launch_expert_gu<T, 8>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::global_grid) launch_expert_gu<T, 16, true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else launch_expert_gu<T, 16>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); break;
    switch (type) { BENCH_GU(16) BENCH_GU(18) BENCH_GU(22)
        default: std::fprintf(stderr, "expert bench: unsupported type %d\n", type); std::abort(); }
#undef BENCH_GU
    launch_check("expert bench gate/up");
}

void down(Variant v, int type, const MoeDev& d, int k, int n_embd, int n_ff, size_t off,
          const void* hq, const float* sh, float* out, cudaStream_t s) {
    if (v == Variant::production) return moe_down(type,d,k,n_embd,n_ff,off,hq,sh,out,s);
#define BENCH_DN(T) case T: \
    if (v == Variant::baseline) launch_expert_down<T,4,false,false>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_lds) launch_expert_down<T,2,false,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_rows4) launch_expert_down<T,4,true,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_lds_rows4) launch_expert_down<T,4,false,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_signs) launch_expert_down<T,2,true,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::vector_load) launch_expert_down<T,2,true,true,false,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_plain) launch_expert_down<T, 4, true, false>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_rows2) launch_expert_down<T, 2, true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_rows8) launch_expert_down<T, 8, true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::rows2) launch_expert_down<T, 2>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::rows8) launch_expert_down<T, 8>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_grid) launch_expert_down<T, 4, true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else launch_expert_down<T, 4>(d,k,n_embd,n_ff,off,hq,sh,out,s); break;
    switch (type) { BENCH_DN(16) BENCH_DN(18) BENCH_DN(22)
        default: std::fprintf(stderr, "expert bench: unsupported type %d\n", type); std::abort(); }
#undef BENCH_DN
    launch_check("expert bench down");
}
} // namespace hip_expert_bench
#endif

bool rows_experts(int gu_type, int d_type, const uint8_t* base, size_t stride, size_t down_off, const int* light, int n,
                  const int* row_tok, const float* x, int n_tok, int n_embd, int n_ff, float limit, int rlo, int rhi,
                  void* xq_scratch, void* hq_scratch, float* out, int ld, cudaStream_t s) {
    const auto ok = [](int t) { return t == 16 || t == 18 || t == 19; };
    if (!ok(gu_type) || !ok(d_type) || n_embd % 256 || n_ff % (GU_ROWS) || n_embd % (GU_WARPS * 2)) return false;
    if (n <= 0 || rhi <= rlo) return true;
    auto* xq = (block_q8_1*) xq_scratch;
    auto* hq = (block_q8_1*) hq_scratch;
    rows_q8_kernel<<<(unsigned) (((int64_t) n_tok * (n_embd / 32) * 32 + 255) / 256), 256, 0, s>>>(x, n_embd, n_tok,
                                                                                                 n_embd, xq);
    const dim3 ggu((unsigned) (n_ff / GU_ROWS), (unsigned) n), gdn((unsigned) (n_embd / (GU_WARPS * 2)), (unsigned) n);
    switch (gu_type) {
        case 16: rows_gate_up_kernel<16><<<ggu, GU_WARPS * 32, 0, s>>>(base, stride, light, row_tok, xq, n_embd, n_ff, out, ld); break;
        case 18: rows_gate_up_kernel<18><<<ggu, GU_WARPS * 32, 0, s>>>(base, stride, light, row_tok, xq, n_embd, n_ff, out, ld); break;
        default: rows_gate_up_kernel<19><<<ggu, GU_WARPS * 32, 0, s>>>(base, stride, light, row_tok, xq, n_embd, n_ff, out, ld); break;
    }
    rows_swiglu_q8_kernel<<<(unsigned) (((int64_t) (rhi - rlo) * (n_ff / 32) * 32 + 255) / 256), 256, 0, s>>>(
        out, ld, rlo, rhi, n_ff, limit, hq);
    switch (d_type) {
        case 16: rows_down_kernel<16><<<gdn, GU_WARPS * 32, 0, s>>>(base, stride, down_off, light, hq, rlo, n_ff, n_embd, out, ld); break;
        case 19: rows_down_kernel<19><<<gdn, GU_WARPS * 32, 0, s>>>(base, stride, down_off, light, hq, rlo, n_ff, n_embd, out, ld); break;
        default: rows_down_kernel<18><<<gdn, GU_WARPS * 32, 0, s>>>(base, stride, down_off, light, hq, rlo, n_ff, n_embd, out, ld); break;
    }
    launch_check("rows_experts");
    return true;
}

void moe_down(int d_type, const MoeDev& d_in, int k, int n_embd, int n_ff, size_t down_off, const void* hq,
              const float* sh_out, float* out, cudaStream_t s) {
    MoeDev d = d_in;
    d.skip = g_skip;
#if defined(STRATA_USE_HIP)
    // IQ3_XXS down regresses on gfx1151; let it use the original dispatch below.
    if ((d_type == 22 || d_type == 18) && n_embd == 4096 && n_ff == 2048 && rdna3_expert_device(d_type == 22)) {
        // Two rows lower the register budget; IQ2_S prefers global grid reads,
        // while the smaller IQ3_XXS grid is faster in LDS on gfx1100.
        if (d_type == 22) launch_expert_down<22, 2, true, true, true>(d, k, n_embd, n_ff, down_off, hq, sh_out, out, s);
        else launch_expert_down<18, 2, false, true, true>(d, k, n_embd, n_ff, down_off, hq, sh_out, out, s);
        launch_check("moe_down RDNA3");
        return;
    }
#endif
    const int blocks = n_embd / DOWN_ROWS;   // n_embd % DOWN_ROWS == 0 (4096)
    const auto* H = (const block_q8_1*) hq;
    switch (d_type) {
#define GLMF_DN(T) \
        case T: moe_down_kernel<T><<<blocks, 256, 0, s>>>(d, k, n_embd, n_ff, down_off, H, sh_out, out); break;
        GLMF_DN(16) GLMF_DN(18) GLMF_DN(19) GLMF_DN(23) GLMF_DN(10) GLMF_DN(11) GLMF_DN(17) GLMF_DN(22) GLMF_DN(21) GLMF_DN(29)
        GLMF_DN(12) GLMF_DN(13) GLMF_DN(14) GLMF_DN(8)
#undef GLMF_DN
        default: std::fprintf(stderr, "glm_fast moe_down: type %d unsupported\n", d_type); return;
    }
    launch_check("moe_down");
}

bool moe_experts_multi(int gu_type, int d_type, const MoeDev& d_in, int k, int n_embd, int n_ff, float limit,
                       size_t down_off, const void* xq, void* hq, const void* sh_down, int sh_type, const void* sh_hq,
                       int n_ff_sh, float* sh_out, float* out, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    MoeDev d = d_in;
    d.skip = g_skip;
    const int T = d.mt_T;
    if (T < 1 || k > 8 || n_ff % GU_ROWS != 0 || n_embd % DOWN_ROWS != 0) return false;
    const bool sh = sh_down != nullptr && sh_out != nullptr;
    const int kk = sh ? k : k + 1;   // (as moe_gate_up: no shared expert, no block has ei == kk)
    const auto* X = (const block_q8_1*) xq;
    auto* H = (block_q8_1*) hq;
    const auto* SD = (const uint8_t*) sh_down;
    const auto* SH = (const block_q8_1*) sh_hq;
    // gate/up: moe_gate_up's kernel for the type, the (expert, token) blocks of a chunk side by side
    const dim3 ggu((unsigned) ((k + (sh ? 1 : 0)) * T), (unsigned) (n_ff / GU_ROWS));
    if (gu_type == 16 && n_embd == 4096 && n_ff == 2048 && rdna3_expert_device()) {
        moe_gate_up_kernel<16, 16, true, false, false, true, false, 0, true><<<ggu, 16 * 32, 0, s>>>(
            d, kk, n_embd, n_ff, limit, X, H, SD, sh_type, SH, n_ff_sh, sh_out);
    } else {
        switch (gu_type) {
#define GLMF_GUM(TT) \
            case TT: moe_gate_up_kernel<TT, GU_WARPS, false, false, false, false, false, 0, true><<<ggu, GU_WARPS * 32, 0, s>>>( \
                d, kk, n_embd, n_ff, limit, X, H, SD, sh_type, SH, n_ff_sh, sh_out); break;
            GLMF_GUM(16) GLMF_GUM(18) GLMF_GUM(19) GLMF_GUM(23) GLMF_GUM(10) GLMF_GUM(11) GLMF_GUM(17) GLMF_GUM(22)
            GLMF_GUM(21) GLMF_GUM(29) GLMF_GUM(12) GLMF_GUM(13) GLMF_GUM(14)
#undef GLMF_GUM
            default: return false;
        }
    }
    // down: moe_down's kernel for the type, the (row block, token) blocks side by side
    const auto* HC = (const block_q8_1*) hq;
    if ((d_type == 22 || d_type == 18) && n_embd == 4096 && n_ff == 2048 && rdna3_expert_device(d_type == 22)) {
        if (d_type == 22)
            moe_down_kernel<22, 2, true, true, true, false, true><<<(unsigned) (n_embd / 2 * T), 256, 0, s>>>(
                d, k, n_embd, n_ff, down_off, HC, sh_out, out);
        else
            moe_down_kernel<18, 2, true, false, true, false, true><<<(unsigned) (n_embd / 2 * T), 256, 0, s>>>(
                d, k, n_embd, n_ff, down_off, HC, sh_out, out);
    } else {
        const unsigned blocks = (unsigned) (n_embd / DOWN_ROWS * T);
        switch (d_type) {
#define GLMF_DNM(TT) \
            case TT: moe_down_kernel<TT, DOWN_ROWS, false, false, false, false, true><<<blocks, 256, 0, s>>>( \
                d, k, n_embd, n_ff, down_off, HC, sh_out, out); break;
            GLMF_DNM(16) GLMF_DNM(18) GLMF_DNM(19) GLMF_DNM(23) GLMF_DNM(10) GLMF_DNM(11) GLMF_DNM(17) GLMF_DNM(22)
            GLMF_DNM(21) GLMF_DNM(29) GLMF_DNM(12) GLMF_DNM(13) GLMF_DNM(14)
#undef GLMF_DNM
            // (the gate/up launch already went: an unknown down type cannot happen with a known gate/up type -
            // moe_gate_up_kernel's and moe_down_kernel's lists are the same)
            default: return false;
        }
    }
    launch_check("moe_experts_multi");
    return true;
#else
    (void) gu_type, (void) d_type, (void) d_in, (void) k, (void) n_embd, (void) n_ff, (void) limit, (void) down_off;
    (void) xq, (void) hq, (void) sh_down, (void) sh_type, (void) sh_hq, (void) n_ff_sh, (void) sh_out, (void) out;
    (void) s;
    return false;
#endif
}

// ---------------------------------------------------------------- a verify window's experts, each once
namespace {
constexpr int kWinEnt = kMaxWinTokens * 8;

// the entries grouped by expert slot (in order of first appearance), each group's members in entry order (t * 8 + i)
// (the tables in shared memory: as locals they were a 2.6 KB stack a thread, which the driver reserves for every thread
// the device can hold - ~420 MB on a V100 - and a launch failed "out of memory" once the expert pool had the rest)
__global__ void moe_win_group_kernel(const STRATA_GRID_CONSTANT MoeWin w) {
    __shared__ unsigned long long gp[kWinEnt];
    __shared__ int cnt[kWinEnt], gof[kWinEnt], fill[kWinEnt + 1];
    if (threadIdx.x != 0) return;
    int ng = 0;
    for (int t = 0; t < w.T; ++t)
        for (int i = 0; i < 8; ++i) {
            const int e = t * 8 + i;
            gof[e] = -1;
            const unsigned long long p = i < w.k ? w.plan_ptr[e] : 0ull;
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
        w.grp_ptr[g] = gp[g];
        w.grp_start[g] = fill[g];
        fill[g + 1] = fill[g] + cnt[g];
    }
    w.grp_start[ng] = fill[ng];
    for (int e = 0; e < w.T * 8; ++e)
        if (gof[e] >= 0) w.grp_ent[fill[gof[e]]++] = e;
    *w.n_groups = ng;
}

// grid (n_ff / GU_ROWS, groups): a group's expert's gate / up rows of this chunk against each member's FFN input, the
// clamped swiglu into the member's h (moe_gate_up_kernel's rows, arithmetic and layout per member)
template<int TG>
__global__ void __launch_bounds__(GU_WARPS * 32) moe_win_gu_kernel(const STRATA_GRID_CONSTANT MoeWin w) {
    const int g = blockIdx.y, chunk = blockIdx.x;
    if (g >= *w.n_groups) return;
    constexpr int NW = GU_WARPS;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float sg[GU_ROWS], su[GU_ROWS];
    __shared__ uint32_t s_tab[TabWords<TG>::n > 0 ? TabWords<TG>::n : 1];
    load_tab<TG>(s_tab);
    __syncthreads();
    const size_t rb = rbytes<TG>(w.n_embd);
    const uint8_t* blob = (const uint8_t*) w.grp_ptr[g];
    constexpr int RPW = GU_ROWS / NW;
    const uint8_t* rows[2 * RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        const int r = chunk * GU_ROWS + warp * RPW + j;
        rows[j] = blob + (size_t) r * rb;
        rows[RPW + j] = blob + (size_t) w.n_ff * rb + (size_t) r * rb;
    }
    for (int m = w.grp_start[g]; m < w.grp_start[g + 1]; ++m) {
        const int e = w.grp_ent[m], t = e >> 3;
        const block_q8_1* xq = (const block_q8_1*) ((const uint8_t*) w.xq + (size_t) t * w.xq_stride);
        float s[2 * RPW];
        rows_dot<TG, 2 * RPW>(rows, xq, w.n_embd, lane, s_tab, s);
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < RPW; ++j) {
                sg[warp * RPW + j] = s[j];
                su[warp * RPW + j] = s[RPW + j];
            }
        }
        __syncthreads();
        if (warp == 0) {
            const float gv = fminf(sg[lane], w.limit);
            const float u = fminf(fmaxf(su[lane], -w.limit), w.limit);
            q8_1_store_warp((gv / (1.0f + expf(-gv))) * u, (block_q8_1*) w.hq + (size_t) e * (w.n_ff / 32) + chunk, lane);
        }
        __syncthreads();
    }
}

// grid (n_embd / DOWN_ROWS, groups): a group's expert's down rows of this chunk against each member's h (a warp a
// member, moe_down_kernel's rows and arithmetic) into the member's partial rows
template<int TD>
__global__ void __launch_bounds__(256) moe_win_down_kernel(const STRATA_GRID_CONSTANT MoeWin w) {
    const int g = blockIdx.y;
    if (g >= *w.n_groups) return;
    constexpr int NR = DOWN_ROWS;
    __shared__ uint32_t s_tab[TabWords<TD>::n > 0 ? TabWords<TD>::n : 1];
    load_tab<TD>(s_tab);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = blockIdx.x * NR;
    const size_t drb = rbytes<TD>(w.n_ff);
    const uint8_t* blob = (const uint8_t*) w.grp_ptr[g];
    const uint8_t* rows[NR];
#pragma unroll
    for (int j = 0; j < NR; ++j) rows[j] = blob + w.down_off + (size_t) (r0 + j) * drb;
    for (int m = w.grp_start[g] + warp; m < w.grp_start[g + 1]; m += (int) (blockDim.x >> 5)) {
        const int e = w.grp_ent[m];
        float s[NR];
        rows_dot<TD, NR>(rows, (const block_q8_1*) w.hq + (size_t) e * (w.n_ff / 32), w.n_ff, lane, s_tab, s);
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < NR; ++j) w.part[(size_t) e * w.n_embd + r0 + j] = s[j];
    }
}

// each token's output rows: its entries' partials in plan order, the host's disk answer, the shared expert (moe_down's
// combine)
__global__ void moe_win_combine_kernel(const STRATA_GRID_CONSTANT MoeWin w) {
    const int t = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= w.n_embd) return;
    float m = 0.0f;
    for (int i = 0; i < w.k; ++i) {
        const int e = t * 8 + i;
        const float p = w.plan_ptr[e] != 0ull ? w.part[(size_t) e * w.n_embd + r] : 0.0f;
        m += w.plan_w[e] * p;
    }
    if (w.cpu_flag[t]) m += w.cpu_part[(size_t) t * w.n_embd + r];
    w.out[(size_t) t * w.out_stride + r] = m + (w.sh_out != nullptr ? w.sh_out[(size_t) t * w.n_embd + r] : 0.0f);
}

}  // namespace

bool moe_win_experts(int gu_type, int d_type, const MoeWin& w, cudaStream_t s) {
    if (w.T < 1 || w.T > kMaxWinTokens || w.k > 8 || w.n_ff % GU_ROWS != 0 || w.n_embd % DOWN_ROWS != 0) return false;
    moe_win_group_kernel<<<1, 32, 0, s>>>(w);
    const int G = w.T * w.k;   // groups at most (the blocks past the real count return at once)
    const dim3 ggu((unsigned) (w.n_ff / GU_ROWS), (unsigned) G), gdn((unsigned) (w.n_embd / DOWN_ROWS), (unsigned) G);
    switch (gu_type) {
#define GLMF_WGU(T)         case T: moe_win_gu_kernel<T><<<ggu, GU_WARPS * 32, 0, s>>>(w); break;
        GLMF_WGU(16) GLMF_WGU(18) GLMF_WGU(19) GLMF_WGU(23) GLMF_WGU(10) GLMF_WGU(11) GLMF_WGU(17) GLMF_WGU(22)
        GLMF_WGU(21) GLMF_WGU(29) GLMF_WGU(12) GLMF_WGU(13) GLMF_WGU(14) GLMF_WGU(8)
#undef GLMF_WGU
        default: return false;
    }
    switch (d_type) {
#define GLMF_WDN(T)         case T: moe_win_down_kernel<T><<<gdn, 256, 0, s>>>(w); break;
        GLMF_WDN(16) GLMF_WDN(18) GLMF_WDN(19) GLMF_WDN(23) GLMF_WDN(10) GLMF_WDN(11) GLMF_WDN(17) GLMF_WDN(22)
        GLMF_WDN(21) GLMF_WDN(29) GLMF_WDN(12) GLMF_WDN(13) GLMF_WDN(14) GLMF_WDN(8)
#undef GLMF_WDN
        default: return false;
    }
    moe_win_combine_kernel<<<dim3((unsigned) ((w.n_embd + 255) / 256), (unsigned) w.T), 256, 0, s>>>(w);
    launch_check("moe_win_experts");
    return true;
}

void moe_win_group(const MoeWin& w, cudaStream_t s) {
    moe_win_group_kernel<<<1, 32, 0, s>>>(w);
    launch_check("moe_win_group");
}

namespace {
__global__ void tab_update_kernel(unsigned long long* tab, const int* keys, const unsigned long long* vals, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tab[keys[i]] = vals[i];
}
}  // namespace

void tab_update(unsigned long long* tab, const int* keys, const unsigned long long* vals, int n, cudaStream_t s) {
    if (n <= 0) return;
    tab_update_kernel<<<(n + 255) / 256, 256, 0, s>>>(tab, keys, vals, n);
    launch_check("tab_update");
}

void mtp_in(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int n, void* cat_q8_1,
            cudaStream_t s) {
    mtp_in_kernel<<<1, 1024, 0, s>>>(emb, h, enorm, hnorm, eps, n, (block_q8_1*) cat_q8_1);
    launch_check("mtp_in");
}

void rms_q8(float* h, const float* add, const float* w, float eps, int n, float* x, void* xq, cudaStream_t s) {
    rms_q8_kernel<<<1, 1024, 0, s>>>(h, add, w, eps, n, x, (block_q8_1*) xq);
    launch_check("rms_q8");
}

void set_skip(const int* flag) {
    g_skip = flag;
    exl3::set_skip(flag);   // (the EXL3 weights' kernels: the same flag)
}

void skip_gate(const volatile int* host_from, int t, int* flag, cudaStream_t s) {
    skip_gate_kernel<<<1, 1, 0, s>>>(host_from, t, flag);
    launch_check("skip_gate");
}

void embed_tok(const void* table, int type, int n_rows, const int* tok, int n_embd, float* out, cudaStream_t s) {
    embed_tok_kernel<<<(n_embd + 255) / 256, 256, 0, s>>>(table, type, n_rows, tok, n_embd, out);
    launch_check("embed_tok");
}

void argmax(const float* x, int n, int* out, cudaStream_t s, const int* map, int map_from) {
    argmax_kernel<<<1, 1024, 0, s>>>(x, n, out, map, map_from);
    launch_check("argmax");
}

// argmax_kernel's pick with its value: out[0] the index (0x7fffffff: none finite), out[1] the value's bits
__global__ void __launch_bounds__(1024) argmax_pair_kernel(const float* __restrict__ x, int n, int* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    const auto take = [&](float v, int i) {
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    };
    // (argmax_kernel's loads: float4, four in flight a thread)
    const int B = (int) blockDim.x;
    int tail = 0;
    if ((((uintptr_t) x) & 15u) == 0) {
        const float4* x4 = (const float4*) x;
        const int n4 = n >> 2;
        int i = threadIdx.x;
        for (; i + 3 * B < n4; i += 4 * B) {
            float4 a[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) a[u] = x4[i + u * B];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int b = 4 * (i + u * B);
                take(a[u].x, b);
                take(a[u].y, b + 1);
                take(a[u].z, b + 2);
                take(a[u].w, b + 3);
            }
        }
        for (; i < n4; i += B) {
            const float4 a = x4[i];
            take(a.x, 4 * i);
            take(a.y, 4 * i + 1);
            take(a.z, 4 * i + 2);
            take(a.w, 4 * i + 3);
        }
        tail = 4 * n4;
    }
    for (int i = tail + threadIdx.x; i < n; i += B) take(x[i], i);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (int) (blockDim.x >> 5); ++i)
            if (sv[i] > best || (sv[i] == best && si[i] < bi)) { best = sv[i]; bi = si[i]; }
        out[0] = bi;
        out[1] = __float_as_int(best);
    }
}

// the pick over two halves' argmax_pair results (b's indices from c0 on): b only when strictly greater - the whole
// row's argmax, ties to the lower index
__global__ void argmax_join_kernel(const int* __restrict__ a, const int* __restrict__ b, int c0, int* __restrict__ out) {
    const bool bv = b[0] != 0x7fffffff, av = a[0] != 0x7fffffff;
    *out = bv && (!av || __int_as_float(b[1]) > __int_as_float(a[1])) ? c0 + b[0] : a[0];
}

void argmax_pair(const float* x, int n, int* out, cudaStream_t s) {
    argmax_pair_kernel<<<1, 1024, 0, s>>>(x, n, out);
    launch_check("argmax_pair");
}

void argmax_join(const int* a, const int* b, int c0, int* out, cudaStream_t s) {
    argmax_join_kernel<<<1, 1, 0, s>>>(a, b, c0, out);
    launch_check("argmax_join");
}

}  // namespace strata::kernels::glmf
