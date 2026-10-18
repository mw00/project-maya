// src/kernels/cuda/iq_kernels.cu - see include/strata/kernels/iq_kernels.hpp.
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at the commit in third_party/ggml/VERSION.txt;
// MIT license, third_party/ggml/LICENSE).  The block structs and codebook grids come from its ggml-common.h,
// included unchanged.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"   // sm_60: __dp4a -> the vmad fallback

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
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
#define ggml_cuda_dp4a(a, b, c) __dp4a((a), (b), (c))

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
__device__ __forceinline__ float vec_dot_q2_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
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
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
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
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
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
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq1_m_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

// IQ1_S: one call covers one 32-elem q8_1 chunk (iqs 0..7 within a 256 block); the delta term rides
// the q8_1 block's fp16 sum (ds.y) exactly as vecdotq.cuh does it.
__device__ __forceinline__ float vec_dot_iq1_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
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
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x * sumi + ds.y * delta);
}

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].  Strata's Qwen IQ3_S
// file keeps one layer's routed gate/up experts in this format.
__device__ __forceinline__ float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<19> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template<int TY>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

template<int TY>
__global__ void __launch_bounds__(128) mmvq_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                   const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in,
                                                   int n_out, int ncols) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s = row_dot<TY>(wr, x + (size_t) c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// ---------------------------------------------------------------- grouped native experts
constexpr int GU_ROWS = 8;     // rows per block (one warp each)

template<int TG>
__global__ void __launch_bounds__(256) native_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_tok,
                                                        const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                        float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = row_dot<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
        if (lane == 0) (is_up ? up : gate)[(size_t) e * L.n_ff + r] = s;
    }
}

__global__ void swiglu_entries_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                      float limit, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // limit = +inf is the plain unclamped path (qwen4exp, __expf); a finite limit is GLM's clamped
    // swiglu, which needs the precise expf - glm_ffn's parity tolerance is tighter than the
    // intrinsic's argument-scaled error at the |g| ~ 30 the model's gates reach
    if (limit == INFINITY) {
        const float g = gate[i];
        h[i] = (g / (1.0f + __expf(-g))) * up[i];
        return;
    }
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    h[i] = (g / (1.0f + expf(-g))) * u;
}

template<int TD>
__global__ void __launch_bounds__(256) native_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_dst,
                                                          const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                          float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    if (r >= L.n_embd) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = row_dot<TD>(wr, hq + (size_t) e * hb, nb, lane);
        if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s;
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
__global__ void quantize_q8_1_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = fminf(amax / 127.0f, 65504.0f);   // finite in FP16 (Strata #1448); below that nothing changes
    const int8_t q = amax == 0.0f ? 0 : fmaxf(-127.0f, fminf(127.0f, roundf(xi / d)));
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template<typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template<> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template<> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half(v); }

template<typename dst_t>
__device__ void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid, const uint64_t* codebook = iq2s_grid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint64_t grid = codebook[x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)];
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j)
        y[j] = cvt<dst_t>(d * (uint8_t) (grid >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid, const uint32_t* codebook = iq3xxs_grid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint32_t grid1 = codebook[q3[2 * il + 0]];
    const uint32_t grid2 = codebook[q3[2 * il + 1]];
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid, const uint32_t* codebook = iq3s_grid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint32_t grid1 = codebook[qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)];
    const uint32_t grid2 = codebook[qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)];
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
// One 128-value half of an IQ3_S superblock: `h` (0 or 1) selects the sub-blocks ib = 4*h .. 4*h + 3, whose
// values land at the chunk-local positions 0 .. 127 (a WK-128 chunk's own column space, not the superblock's
// 256).  `sub` in 0..15 picks the (il, ib&3) pair that dq_iq3_s reads from a lane's tid, so a warp's lanes
// 0..15 and 16..31 decode two rows at once and a lane still decodes the eight values per row that dq_iq3_s
// gives each of its 32.  d, qs, qh, scales and signs stay indexed by the superblock's sub-block ib, and the
// float d and the sign application are dq_iq3_s's own expressions.
template<typename dst_t>
__device__ void dq_iq3_s_half(const void* vx, int64_t ibs, int h, int sub, dst_t* yy, const uint32_t* codebook = iq3s_grid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int il = sub >> 2, ib4 = sub & 3, ib = 4 * h + ib4;
    dst_t* y = yy + 32 * ib4 + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint32_t grid1 = codebook[qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)];
    const uint32_t grid2 = codebook[qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)];
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    if constexpr (std::is_same_v<dst_t, __half>) {
        // the eight values are contiguous and eight-aligned: four aligned half2 stores, not eight
#pragma unroll
        for (int j = 0; j < 4; j += 2) {
            const float l0 = d * (float) (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            const float l1 = d * (float) (uint8_t) (grid1 >> (8 * j + 8)) * (signs & kmask_iq2xs[j + 1] ? -1.f : 1.f);
            const float h0 = d * (float) (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            const float h1 = d * (float) (uint8_t) (grid2 >> (8 * j + 8)) * (signs & kmask_iq2xs[j + 5] ? -1.f : 1.f);
            *reinterpret_cast<__half2*>(y + j) = __floats2half2_rn(l0, l1);
            *reinterpret_cast<__half2*>(y + j + 4) = __floats2half2_rn(h0, h1);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y[j + 0] = cvt<dst_t>(d * (float) (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
            y[j + 4] = cvt<dst_t>(d * (float) (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
        }
    }
}
// The same 128-value half for IQ3_XXS: `h` and `sub` give the sub-block ib = 4*h + (sub & 3) and the value group
// il = sub >> 2 exactly as dq_iq3_s_half, and the codebook, the four-bit gas scale (aux32 bits 28..31) and the
// seven-bit sign group (bits 7*il .. 7*il + 6) are dq_iq3_xxs's own expressions.  Its grid1 holds y[0..3] and
// grid2 y[4..7], so the half path splits the same way dq_iq3_s_half's does.
template<typename dst_t>
__device__ void dq_iq3_xxs_half(const void* vx, int64_t ibs, int h, int sub, dst_t* yy, const uint32_t* codebook = iq3xxs_grid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int il = sub >> 2, ib4 = sub & 3, ib = 4 * h + ib4;
    dst_t* y = yy + 32 * ib4 + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint32_t grid1 = codebook[q3[2 * il + 0]];
    const uint32_t grid2 = codebook[q3[2 * il + 1]];
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    if constexpr (std::is_same_v<dst_t, __half>) {
        // the eight values are contiguous and eight-aligned: four aligned half2 stores, not eight
#pragma unroll
        for (int j = 0; j < 4; j += 2) {
            const float l0 = d * (float) (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            const float l1 = d * (float) (uint8_t) (grid1 >> (8 * j + 8)) * (signs & kmask_iq2xs[j + 1] ? -1.f : 1.f);
            const float h0 = d * (float) (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            const float h1 = d * (float) (uint8_t) (grid2 >> (8 * j + 8)) * (signs & kmask_iq2xs[j + 5] ? -1.f : 1.f);
            *reinterpret_cast<__half2*>(y + j) = __floats2half2_rn(l0, l1);
            *reinterpret_cast<__half2*>(y + j + 4) = __floats2half2_rn(h0, h1);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y[j + 0] = cvt<dst_t>(d * (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
            y[j + 4] = cvt<dst_t>(d * (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
        }
    }
}
// And the same 128-value half for IQ2_S: `h`, `sub`, `il`, `ib4` and the output mapping are dq_iq3_s_half's, while
// the 1024-entry 64-bit grid, the nibble scale (bits 0..3 for il 0/1, 4..7 for il 2/3) and the sign byte at
// qs[QK_K/8 + 4*ib + il] are dq_iq2_s's own expressions.  Its eight values are one contiguous group, so the half
// path is four half2 stores with nothing split between two grids.
template<typename dst_t>
__device__ void dq_iq2_s_half(const void* vx, int64_t ibs, int h, int sub, dst_t* yy, const uint64_t* codebook = iq2s_grid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int il = sub >> 2, ib4 = sub & 3, ib = 4 * h + ib4;
    dst_t* y = yy + 32 * ib4 + 8 * il;
    const uint64_t grid = codebook[x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)];
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    if constexpr (std::is_same_v<dst_t, __half>) {
        // the eight values are contiguous and eight-aligned: four aligned half2 stores, not eight
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const float l0 = d * (float) (uint8_t) (grid >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            const float l1 = d * (float) (uint8_t) (grid >> (8 * j + 8)) * (signs & kmask_iq2xs[j + 1] ? -1.f : 1.f);
            *reinterpret_cast<__half2*>(y + j) = __floats2half2_rn(l0, l1);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
            y[j] = cvt<dst_t>(d * (uint8_t) (grid >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq1_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_s* x = (const block_iq1_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float) x[ibs].d * (2 * ((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[ib] >> 3 * il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template<typename dst_t>
__device__ void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template<typename dst_t>
__device__ void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t>
__device__ void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid, const int8_t* codebook = kvalues_iq4nl) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * codebook[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * codebook[q4[j] >> 4]);
    }
}
template<typename dst_t>
__device__ void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}

template<typename dst_t>
__device__ __forceinline__ void dq_dispatch(int ty, const void* vx, int64_t ibs, dst_t* y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid); break;
        case 17: dq_iq2_xs(vx, ibs, y, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid); break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 19: dq_iq1_s(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template<typename dst_t>
__global__ void dequant_flat_kernel(int ty, const void* __restrict__ vx, dst_t* __restrict__ y) {
    const int64_t i = blockIdx.x;
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, threadIdx.x);
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
__global__ void dequant_gu_kernel(int ty, const void* __restrict__ gate, const void* __restrict__ up, int64_t per_row,
                                  __half* __restrict__ y) {
    const int64_t i = blockIdx.x;
    const int parity = blockIdx.y;
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<__half>(ty, parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * QK_K, threadIdx.x);
}

bool is_iq(int t) { return t == 16 || t == 17 || t == 18 || t == 19 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11; }

}  // namespace

bool iq_supported(int t) noexcept { return is_iq(t); }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 2: return (size_t) (n / 32) * sizeof(block_q4_0);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 19: return (size_t) (n / 256) * sizeof(block_iq1_s);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    quantize_q8_1_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, (block_q8_1*) y, n);
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const dim3 grid((unsigned) ((n_out + 3) / 4)), block(32, 4);
    const size_t rb = iq_row_bytes(t, n_in);
    cudaStream_t s = (cudaStream_t) stream;
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
        case 16: mmvq_kernel<16><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        case 17: mmvq_kernel<17><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        case 18: mmvq_kernel<18><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        case 20: mmvq_kernel<20><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        case 21: mmvq_kernel<21><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        case 22: mmvq_kernel<22><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        case 23: mmvq_kernel<23><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        case 29: mmvq_kernel<29><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        case 19: mmvq_kernel<19><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        case 42: mmvq_kernel<42><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<__half><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, (__half*) dst);
    check("iq_dequant_f16");
}

namespace {
__global__ void embed_rows_kernel(int ty, const uint8_t* __restrict__ table, size_t row_bytes,
                                  const int32_t* __restrict__ tokens, int64_t n_embd, float* __restrict__ y) {
    const int t = blockIdx.y;
    const int64_t b = blockIdx.x;
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t) t * n_embd + b * QK_K, threadIdx.x);
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    embed_rows_kernel<<<dim3((unsigned) (n_embd / 256), (unsigned) n_tok), 32, 0, (cudaStream_t) stream>>>(
        t, (const uint8_t*) table, row_bytes, tokens, n_embd, out);
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<float><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, dst);
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    const int64_t per_row = n_embd / 256;
    dequant_gu_kernel<<<dim3((unsigned) (n_ff * per_row), 2), 32, 0, (cudaStream_t) stream>>>(t, gate, up, per_row,
                                                                                           (__half*) dst);
    check("iq_dequant_gu_f16");
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           float swiglu_limit) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const dim3 ggu((unsigned) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS), (unsigned) cap_groups);
    switch (L.gu_type) {
        case 16: native_gu_kernel<16><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 17: native_gu_kernel<17><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 18: native_gu_kernel<18><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 21: native_gu_kernel<21><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 22: native_gu_kernel<22><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 23: native_gu_kernel<23><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 29: native_gu_kernel<29><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 19: native_gu_kernel<19><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 42: native_gu_kernel<42><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
    check("native_expert_grouped/gu");
    const long long nh = (long long) cap_entries * L.n_ff;
    swiglu_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, swiglu_limit, nh);
    quantize_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(h, hq, nh);
    const dim3 gd((unsigned) ((L.n_embd + 7) / 8), (unsigned) cap_groups);
    switch (L.d_type) {
        case 18: native_down_kernel<18><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 20: native_down_kernel<20><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 23: native_down_kernel<23><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 42: native_down_kernel<42><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
    check("native_expert_grouped/down");
}
}  // namespace strata::kernels

// ---- Volta prompt experts (STRATA_PF_WMMA): a group's experts as one launch of FP16 tensor-core tiles, the weights
// dequantized into shared memory by dq_dispatch (the FP16 path's own formulas), the activations FP16, FP32 sums.
// Replaces MMQ's dp4a products on sm_70, and the FP16 path's dequantize-to-global + one cuBLAS call per expert.
#if !defined(__HIPCC__)
#include <mma.h>
namespace strata::kernels {
namespace {
constexpr int WG_BM = 64, WG_BN = 128, WG_THREADS = 256, WG_XK = 128, WG_MT = 4;
// The static codebook a TY carries (the kernel's `codebook`), which counts toward the block's shared memory.
__host__ __device__ constexpr size_t wg_codebook_words(int ty) {
    return (size_t) (ty == 22 ? 1024 : ty == 21 ? 256 : ty == 18 ? 128 : 2);
}
__host__ __device__ constexpr size_t gemm_iq_f16_grouped_smem(int wk, int xk = WG_XK) {
    return (size_t) WG_BN * (wk + 8) * 2 + (size_t) WG_BM * (xk + 8) * 2;
}
// A block covers 128 weight rows of one expert and up to MT * WG_BM token rows.
// Each weight chunk is decoded once into shared memory; activations use a BM x XK tile.
// Eight warps use 32 x 32 m-tiles and FP32 accumulators for MT m-tiles.
// DEC selects the decoder: 0 dq_dispatch (WK 256) or, for TY 18/21/22 at WK 128, dq_iq3_xxs_half / dq_iq3_s_half /
// dq_iq2_s_half, 1 Q2_0 (64 values), 2 IQ4_NL (32 values).  Two tile shapes: the wide one (MT = WG_MT = 4, 256 token
// rows, one block per SM) and the small one (MT = 2, 128 token rows, 43 KiB of weight tile at XK 64 or 39 KiB at
// XK 32, plus the static codebook under 48 KiB, so two blocks fit one V100 SM).  The launcher picks the shape from
// the launch's largest expert; both are exact and differ only in how many token rows share one weight decode.
// Optional ids gather input rows.  With a weight table (wsrc 1 = gate/up, 2 = down) expert e's matrix is the
// table's device pointer; otherwise experts are contiguous with expert_bytes stride.
template<int WK, int DEC, int TY = 0, int XK = WG_XK, int MT = WG_MT>
__global__ void __launch_bounds__(WG_THREADS, MT == 2 ? 2 : 1) gemm_iq_f16_grouped_kernel(
        int ty, const uint8_t* __restrict__ W, size_t expert_bytes, size_t row_bytes, int K,
        const __half* __restrict__ X, int ldx, const int32_t* __restrict__ bounds, const int32_t* __restrict__ ids,
        int ch_per_e, float* __restrict__ Y, int ldy, int wsrc, GroupedWeightTable tbl) {
    static_assert(gemm_iq_f16_grouped_smem(WK, XK) >= (size_t) WG_BM * (WG_BN + 4) * sizeof(float));
    static_assert(DEC != 2 || WK == 128, "IQ4_NL decodes one 128-value chunk per k0 step");
    static_assert(MT != 2 || (WK == 128 && (XK == 64 || (TY == 22 && XK == 32))),
                  "the two-m-tile tiles are the 128-value chunks; only IQ2_S's 8 KiB codebook needs XK 32");
    static_assert(MT != 2 || wg_codebook_words(TY) * sizeof(uint64_t) + gemm_iq_f16_grouped_smem(WK, XK) <= 48 * 1024,
                  "a small tile's static codebook and dynamic tiles must fit 48 KiB so two blocks fit one SM");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int LDW = WK + 8, LDX = XK + 8;
    extern __shared__ __align__(16) unsigned char wg_smem[];
    __half* sw = reinterpret_cast<__half*>(wg_smem);            // [BN][LDW]
    __half* sx = sw + WG_BN * LDW;                              // [BM][LDX]
    const int e = blockIdx.y / ch_per_e, ch = blockIdx.y % ch_per_e;
    const int c0 = bounds[e] + ch * MT * WG_BM, r1 = bounds[e + 1];
    if (c0 >= r1) return;
    const int crow = min(MT * WG_BM, r1 - c0);
    const int nmt = (crow + WG_BM - 1) / WG_BM;
    const int n0 = blockIdx.x * WG_BN;
    const uint8_t* we = wsrc == 0 ? W + (size_t) e * expert_bytes
                                  : (const uint8_t*) (wsrc == 1 ? tbl.gu[e] : tbl.dn[e]);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;                     // rows wm*32, cols wn*32 of an m-tile
    __shared__ uint64_t codebook[wg_codebook_words(TY)];
    auto* grid32 = reinterpret_cast<uint32_t*>(codebook);
    auto* grid8 = reinterpret_cast<int8_t*>(codebook);
    if constexpr (DEC == 0 && TY != 0) {
        if constexpr (TY == 22) {
            for (int i = tid; i < 1024; i += WG_THREADS) codebook[i] = iq2s_grid[i];
        } else if constexpr (TY == 21) {
            for (int i = tid; i < 512; i += WG_THREADS) grid32[i] = iq3s_grid[i];
        } else if constexpr (TY == 18) {
            for (int i = tid; i < 256; i += WG_THREADS) grid32[i] = iq3xxs_grid[i];
        } else if (tid < 16) grid8[tid] = kvalues_iq4nl[tid];
        __syncthreads();
    } else if constexpr (DEC == 2) {
        if (tid < 16) grid8[tid] = kvalues_iq4nl[tid];
        __syncthreads();
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT][2][2];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int b = 0; b < 2; ++b) wmma::fill_fragment(acc[t][a][b], 0.0f);
    // the activation tiles in order (k0, m-tile, half): the next one's XQ uint4 per thread are loaded into registers
    // while the current one is multiplied (the first before the weights' dequantization)
    constexpr int XQ = WG_BM * (XK / 8) / WG_THREADS;
    const int nx = WK / XK, per_k = nmt * nx, total = (K / WK) * per_k;
    uint4 xr[XQ];
    auto load_x = [&](int it) {
        const int kb = it / per_k, rem = it % per_k, t = rem / nx, xh = rem % nx;
        const int rows = min(WG_BM, crow - t * WG_BM);
#pragma unroll
        for (int q = 0; q < XQ; ++q) {
            const int i = tid + q * WG_THREADS, r = i / (XK / 8), c = (i % (XK / 8)) * 8;
            const int row = c0 + t * WG_BM + r;                  // the output row; ids gathers its input row
            xr[q] = r < rows ? *reinterpret_cast<const uint4*>(X + (size_t) (ids ? ids[row] : row) * ldx + kb * WK +
                                                              xh * XK + c)
                             : make_uint4(0, 0, 0, 0);
        }
    };
    load_x(0);
    int it = 0;
    for (int k0 = 0; k0 < K; k0 += WK) {
        if constexpr (DEC == 0) {
            if constexpr (WK == 128 && (TY == 18 || TY == 21 || TY == 22)) {
                // The small tile: a WK-128 chunk is one 128-value half of an IQ3_XXS / IQ3_S / IQ2_S superblock,
                // its sub-blocks ib = 4*kh .. 4*kh + 3 landing at columns 0 .. 127, sixteen lanes to a row and two
                // rows per warp, so a lane decodes the same eight values per half as the 256-value decoders below
                // give it per superblock.  The wide tile keeps those decoders.
                const int kh = (k0 % 256) / 128, sub = lane & 15, rr = lane >> 4;
                for (int r = 2 * warp + rr; r < WG_BN; r += 2 * (WG_THREADS / 32)) {
                    const uint8_t* wr = we + (size_t) (n0 + r) * row_bytes;
                    auto* out = sw + r * LDW;
                    if constexpr (TY == 18) dq_iq3_xxs_half(wr, k0 / 256, kh, sub, out, grid32);
                    else if constexpr (TY == 21) dq_iq3_s_half(wr, k0 / 256, kh, sub, out, grid32);
                    else dq_iq2_s_half(wr, k0 / 256, kh, sub, out, codebook);
                }
            } else {
                for (int r = warp; r < WG_BN; r += WG_THREADS / 32) {
                    const auto* wr = we + (size_t) (n0 + r) * row_bytes;
                    auto* out = sw + r * LDW;
                    if constexpr (TY == 22) dq_iq2_s(wr, k0 / 256, out, lane, codebook);
                    else if constexpr (TY == 21) dq_iq3_s(wr, k0 / 256, out, lane, grid32);
                    else if constexpr (TY == 18) dq_iq3_xxs(wr, k0 / 256, out, lane, grid32);
                    else if constexpr (TY == 23) dq_iq4_xs(wr, k0 / 256, out, lane, grid8);
                    else dq_dispatch<__half>(ty, wr, k0 / 256, out, lane);
                }
            }
        } else if constexpr (DEC == 1) {
            const int r = tid >> 1, blk = tid & 1;               // one thread per (row, 64-value block)
            const block_q2_0* b = reinterpret_cast<const block_q2_0*>(we + (size_t) (n0 + r) * row_bytes) + k0 / 64 + blk;
            // Codes select {-d, signed zero, d, 2*d}. Pack two exact FP16 weights with
            // a byte permutation instead of repeated integer-to-float conversions.
            const uint32_t d = __half_as_ushort(b->d);
            const uint32_t z = __half_as_ushort(__hmul(b->d, __float2half(0.0f)));
            const uint32_t twice = __half_as_ushort(__hadd(b->d, b->d));
            const uint32_t table01 = (d ^ 0x8000u) | (z << 16);
            const uint32_t table23 = d | (twice << 16);
            uint4 q4;
            memcpy(&q4, b->qs, 16);
            const uint8_t* qb = reinterpret_cast<const uint8_t*>(&q4);
            __half* o = sw + r * LDW + blk * 64;
#pragma unroll
            for (int c = 0; c < 16; ++c) {
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const unsigned q = qb[c] >> (4 * pair);
                    const unsigned a = (q & 3u) << 1, b = ((q >> 2) & 3u) << 1;
                    const unsigned selector = 0x1010u | a | (a << 4) | (b << 8) | (b << 12);
                    *reinterpret_cast<uint32_t*>(o + 4 * c + 2 * pair) = __byte_perm(table01, table23, selector);
                }
            }
        } else {
            const int r = tid >> 1, h = tid & 1;                 // one thread per (row, 64-value half of the chunk)
            const block_iq4_nl* b = reinterpret_cast<const block_iq4_nl*>(we + (size_t) (n0 + r) * row_bytes) + k0 / 32 + 2 * h;
            __half* o = sw + r * LDW + h * 64;
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const __half2 d = __halves2half2(b[blk].d, b[blk].d);
                const uint8_t* qs = b[blk].qs;
#pragma unroll
                for (int j = 0; j < 16; j += 2) {
                    uint16_t q;
                    memcpy(&q, qs + j, sizeof(q));
                    const __half2 lo = __floats2half2_rn((float) grid8[q & 0xf], (float) grid8[(q >> 8) & 0xf]);
                    const __half2 hi = __floats2half2_rn((float) grid8[(q >> 4) & 0xf], (float) grid8[q >> 12]);
                    *reinterpret_cast<__half2*>(o + 32 * blk + j) = __hmul2(d, lo);
                    *reinterpret_cast<__half2*>(o + 32 * blk + 16 + j) = __hmul2(d, hi);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t >= nmt) break;
#pragma unroll 1
            for (int xh = 0; xh < WK / XK; ++xh, ++it) {
                __syncthreads();                                 // the previous X tile is consumed (and W is written)
#pragma unroll
                for (int q = 0; q < XQ; ++q) {
                    const int i = tid + q * WG_THREADS, r = i / (XK / 8), c = (i % (XK / 8)) * 8;
                    *reinterpret_cast<uint4*>(sx + r * LDX + c) = xr[q];
                }
                __syncthreads();
                if (it + 1 < total) load_x(it + 1);
#pragma unroll
                for (int kk = 0; kk < XK; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        wmma::load_matrix_sync(a[f], sx + (wm * 32 + f * 16) * LDX + kk, LDX);
                        wmma::load_matrix_sync(b[f], sw + (wn * 32 + f * 16) * LDW + xh * XK + kk, LDW);
                    }
#pragma unroll
                    for (int fa = 0; fa < 2; ++fa)
#pragma unroll
                        for (int fb = 0; fb < 2; ++fb) wmma::mma_sync(acc[t][fa][fb], a[fa], b[fb], acc[t][fa][fb]);
                }
            }
        }
        __syncthreads();                                         // W is consumed before the next chunk
    }
    float* so = reinterpret_cast<float*>(wg_smem);               // [BM][BN + 4] per m-tile
    constexpr int LDO = WG_BN + 4;
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (t >= nmt) break;
        const int rows = min(WG_BM, crow - t * WG_BM);
#pragma unroll
        for (int fa = 0; fa < 2; ++fa)
#pragma unroll
            for (int fb = 0; fb < 2; ++fb)
                wmma::store_matrix_sync(so + (wm * 32 + fa * 16) * LDO + wn * 32 + fb * 16, acc[t][fa][fb], LDO, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < rows * (WG_BN / 4); i += WG_THREADS) {
            const int r = i / (WG_BN / 4), c = (i % (WG_BN / 4)) * 4;
            *reinterpret_cast<float4*>(Y + (size_t) (c0 + t * WG_BM + r) * ldy + n0 + c) = *reinterpret_cast<const float4*>(so + r * LDO + c);
        }
        __syncthreads();
    }
#endif
}

// The FP16 tensor cores need sm_70+, and the dynamic-shared-memory attribute is per device (a two-GPU process must
// raise it on each card): cache, per device, which instantiations the attribute was accepted for.  A 128 x 264-half
// weight tile is 83 KiB, past the 64 KiB a Turing card allows and within Volta's / Ampere's opt-in limit.  The bits:
// 1 queried, 2 the WK-256 gate/up kernels, 4 the WK-128 Q2_0 kernel, 8 the wide WK-128 IQ4_NL kernel,
// 16 the small IQ3_S tile, 32 the small IQ4_NL tile, 64 the small IQ3_XXS tile, 128 the small IQ2_S tile.  Each
// tile shape has its own bit: a device that refuses one attribute must keep that shape on the MMQ path rather than
// launch a kernel whose tile was never attributed.
unsigned gemm_iq_f16_grouped_caps(int dev) {
    static thread_local unsigned per_dev[64];   // per-thread: two prefill objects may first touch one device at once
    if (dev < 0 || dev >= 64) return 0;
    if (!(per_dev[dev] & 1u)) {
        unsigned caps = 1u;                      // queried
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 7) {
            const bool gu = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess;
            const bool g18 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, 0, 18>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess;
            const bool g21 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, 0, 21>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess;
            const bool g22 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, 0, 22>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess;
            const bool g23 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, 0, 23>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess;
            const bool d1 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 1, 0, 64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128, 64)) == cudaSuccess;
            const bool d2 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128)) == cudaSuccess;
            const bool d21 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 0, 21, 64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128, 64)) == cudaSuccess;
            const bool d2s = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 2, 0, 64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128, 64)) == cudaSuccess;
            const bool d18 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 0, 18, 64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128, 64)) == cudaSuccess;
            const bool d22 = cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 0, 22, 32, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gemm_iq_f16_grouped_smem(128, 32)) == cudaSuccess;
            if (!gu || !g18 || !g21 || !g22 || !g23 || !d1 || !d2 || !d21 || !d2s || !d18 || !d22) (void) cudaGetLastError();
            caps |= ((gu && g18 && g21 && g22 && g23) ? 2u : 0u) | (d1 ? 4u : 0u) | (d2 ? 8u : 0u) |
                    (d21 ? 16u : 0u) | (d2s ? 32u : 0u) | (d18 ? 64u : 0u) | (d22 ? 128u : 0u);
        }
        per_dev[dev] = caps;
    }
    return per_dev[dev];
}
}  // namespace

bool gemm_iq_f16_grouped_supported(int dev) { return (gemm_iq_f16_grouped_caps(dev) & 2u) != 0; }

bool gemm_iq_f16_grouped(int ty, const void* W, size_t expert_bytes, int n_out, int K, const void* X, int ldx,
                         const int32_t* bounds, int n_experts, int max_rows, float* Y, int ldy, void* stream,
                         const int32_t* ids, const GroupedWeightTable* indirect, bool down) {
    if (n_out <= 0 || K <= 0 || n_out % WG_BN != 0 || n_experts <= 0 || max_rows <= 0) return false;
    if (indirect != nullptr && n_experts > indirect->n) return false;   // the table must cover the whole launch
    if (ldx < K || ldx % 8 != 0 || ldy < n_out || ldy % 4 != 0) return false;   // uint4 loads / float4 stores
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    // The IQ gate/up formats and IQ4_NL use 128-row tiles for short groups and 256-row tiles for wide groups.
    // The weight chunk is 128 values for IQ4_NL and Q2_0, and 128 (short) or 256 (wide) for IQ3_XXS / IQ3_S / IQ2_S.
    const int dec = ty == 42 ? 1 : ty == 20 ? 2 : 0;
    // A gate/up superblock is 256 values, so its K must split into whole 256-value chunks; the Q2_0 and IQ4_NL
    // kernels work on 64- and 32-value blocks, so a 640-column down product splits into whole 128-value chunks.
    if (dec == 0 ? (K % 256 != 0 || !is_iq(ty)) : K % 128 != 0) return false;
    const unsigned caps = gemm_iq_f16_grouped_caps(dev);
    // The small tile decodes a weight chunk once per 128 token rows but hides the decode behind two blocks per
    // SM, which wins while the group's largest expert holds at most one chunk; past that the wide tile's half as
    // frequent weight decode pays for its single block.  Both tiles are exact: only the chunking differs.
    const bool small = max_rows <= 2 * WG_BM && (dec == 2 || (dec == 0 && (ty == 18 || ty == 21 || ty == 22)));
    const unsigned need = small ? (dec == 2 ? 32u : ty == 22 ? 128u : ty == 18 ? 64u : 16u)
                                : dec == 0 ? 2u : dec == 1 ? 4u : 8u;
    if (!(caps & need)) return false;
    const size_t row_bytes = iq_row_bytes(ty, K);
    const int mt = small || dec == 1 ? 2 : WG_MT;
    const int chunks = (max_rows + mt * WG_BM - 1) / (mt * WG_BM);
    const dim3 grid((unsigned) (n_out / WG_BN), (unsigned) (n_experts * chunks));
    cudaStream_t s = (cudaStream_t) stream;
    const uint8_t* w = (const uint8_t*) W;
    const __half* x = (const __half*) X;
    const int wsrc = indirect == nullptr ? 0 : (down ? 2 : 1);
    const GroupedWeightTable tbl = indirect != nullptr ? *indirect : GroupedWeightTable{};
    if (dec == 0 && small && ty == 18)
        gemm_iq_f16_grouped_kernel<128, 0, 18, 64, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 64), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && small && ty == 21)
        gemm_iq_f16_grouped_kernel<128, 0, 21, 64, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 64), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && small && ty == 22)
        gemm_iq_f16_grouped_kernel<128, 0, 22, 32, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 32), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && ty == 18)
        gemm_iq_f16_grouped_kernel<256, 0, 18><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && ty == 21)
        gemm_iq_f16_grouped_kernel<256, 0, 21><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && ty == 22)
        gemm_iq_f16_grouped_kernel<256, 0, 22><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0 && ty == 23)
        gemm_iq_f16_grouped_kernel<256, 0, 23><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 0)
        gemm_iq_f16_grouped_kernel<256, 0><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (dec == 1)
        gemm_iq_f16_grouped_kernel<128, 1, 0, 64, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 64), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else if (small)
        gemm_iq_f16_grouped_kernel<128, 2, 0, 64, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 64), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    else
        gemm_iq_f16_grouped_kernel<128, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, wsrc, tbl);
    return cudaGetLastError() == cudaSuccess;
}
}  // namespace strata::kernels
#else
namespace strata::kernels {
bool gemm_iq_f16_grouped(int, const void*, size_t, int, int, const void*, int, const int32_t*, int, int, float*, int, void*,
                         const int32_t*, const GroupedWeightTable*, bool) {
    return false;
}
bool gemm_iq_f16_grouped_supported(int) { return false; }
}  // namespace strata::kernels
#endif
