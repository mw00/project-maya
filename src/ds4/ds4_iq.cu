// src/ds4/ds4_iq.cu - a GGUF's low-bit routed experts on the device (see ds4_iq.cuh).  The block layouts and grids are
// ggml's (ggml-common.h, MIT, third_party/ggml/LICENSE); the sign and grid tricks are llama.cpp's vecdotq.cuh.
#include "ds4_iq.cuh"
#include "ds4_iq1_codes.hpp"
#include "ds4_pdl.cuh"
#if defined(STRATA_USE_HIP) && !defined(MAYA_DS4_IQ_ONLY)   // (#354; maya-mm3 builds ds4_iq.cu without it)
#include "strata/prefill/moe_mmq.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/prefill/moe_fused_rdna4.hpp"
#endif
#include <cstring>
#include <map>
#include <memory>
#endif

#include <cuda_fp16.h>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#endif
// rocWMMA (the prompt sets on RDNA 3 / 4's matrix cores) as glm_batch.cu reads it: a device pass for gfx11 / gfx12
// only, the host pass when the build targets such a card (STRATA_HAS_WMMA)
#if defined(STRATA_USE_HIP) && \
    ((!(defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__) && STRATA_HAS_WMMA) || \
     ((defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__) && (defined(__gfx1100__) || defined(__gfx1101__) || \
      defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__))))
#define DS4_ROCWMMA 1
#pragma push_macro("__shfl_xor_sync")
#undef __shfl_xor_sync
#pragma push_macro("__shfl_down_sync")
#undef __shfl_down_sync
#pragma push_macro("__shfl_up_sync")
#undef __shfl_up_sync
#pragma push_macro("__shfl_sync")
#undef __shfl_sync
#include <rocwmma/rocwmma.hpp>
#pragma pop_macro("__shfl_xor_sync")
#pragma pop_macro("__shfl_down_sync")
#pragma pop_macro("__shfl_up_sync")
#pragma pop_macro("__shfl_sync")
#else
#define DS4_ROCWMMA 0
#endif

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>

namespace strata::ds4::gguf::iq {
namespace {

constexpr int kWarps = 8, kThreads = kWarps * 32;

// ---------------------------------------------------------------- the types
template <int T>
struct Ty;
template <> struct Ty<kIQ2_XXS> { static constexpr int B = 66, TAB = 512, MIN = 0; };
template <> struct Ty<kIQ2_XS> { static constexpr int B = 74, TAB = 1024, MIN = 0; };   // (#357, @OscarSheng)
template <> struct Ty<kIQ2_S> { static constexpr int B = 82, TAB = 2048, MIN = 0; };
template <> struct Ty<kIQ3_XXS> { static constexpr int B = 98, TAB = 256, MIN = 0; };
template <> struct Ty<kIQ3_S> { static constexpr int B = 110, TAB = 512, MIN = 0; };
template <> struct Ty<kIQ4_XS> { static constexpr int B = 136, TAB = 4, MIN = 0; };
template <> struct Ty<kIQ4_NL> { static constexpr int B = 18, TAB = 4, MIN = 0; };   // (32 values a block: d, 16 bytes)
template <> struct Ty<kQ4_K> { static constexpr int B = 144, TAB = 0, MIN = 1; };
template <> struct Ty<kQ5_K> { static constexpr int B = 176, TAB = 0, MIN = 1; };   // (a GGUF's dense weights in Q5_K)
template <> struct Ty<kQ2_K> { static constexpr int B = 84, TAB = 0, MIN = 1; };
template <> struct Ty<kQ3_K> { static constexpr int B = 110, TAB = 0, MIN = 0; };
template <> struct Ty<kIQ1_S> { static constexpr int B = 50, TAB = 2048, MIN = 0; };   // (iq1s_grid_gpu)
template <> struct Ty<kIQ1_M> { static constexpr int B = 56, TAB = 2048, MIN = 0; };
// the dense matrices' Q6_K (ql[128] the low nibbles, qh[64] the high 2 bits, scales[16] int8 a 16 values, d) and Q8_0
// (d, then 32 int8: a block of 32 values) - the GEMVs, the FP16 rows and decode_values (maya-mm3's MiniMax quants)
template <> struct Ty<kQ6_K> { static constexpr int B = 210, TAB = 0, MIN = 0; };
template <> struct Ty<kQ8_0> { static constexpr int B = 34, TAB = 0, MIN = 0; };
// MXFP4 (ds4_gguf.cu's own kernels everywhere else): a block of 32 values, an E8M0 scale then 16 code bytes; the light
// sets take it too
constexpr int kMXFP4 = 39;
template <> struct Ty<kMXFP4> { static constexpr int B = 17, TAB = 0, MIN = 0; };
template <int T> constexpr int kQK = T == kMXFP4 || T == kQ8_0 || T == kIQ4_NL ? 32 : 256;   // values a block
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_s) == 82 && sizeof(block_iq3_xxs) == 98 &&
                  sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 && sizeof(block_q4_K) == 144 &&
                  sizeof(block_q2_K) == 84 && sizeof(block_q3_K) == 110 && sizeof(block_iq1_s) == 50 &&
                  sizeof(block_iq1_m) == 56 && sizeof(block_iq2_xs) == 74 && sizeof(block_q5_K) == 176 &&
                  sizeof(block_q6_K) == 210 && sizeof(block_q8_0) == 34 && sizeof(block_iq4_nl) == 18,
              "ggml's block sizes");
static_assert(IQ1S_DELTA == 0.125f && IQ1M_DELTA == 0.125f, "IQ1's delta: the codes 8 g +- 1 below");

template <int T>
__host__ __device__ __forceinline__ size_t row_bytes_t(int k) { return (size_t) (k / kQK<T>) * Ty<T>::B; }

// the type's codebook into shared memory (u32 words; IQ4_XS, IQ4_NL: their 16 values, from bytes - an int8 table's
// alignment)
template <int T>
__device__ __forceinline__ void load_tab(uint32_t* tab) {
    if constexpr (T == kIQ4_XS || T == kIQ4_NL) {
        if (threadIdx.x < 4) {
            const uint8_t* kv = (const uint8_t*) kvalues_iq4nl + 4 * threadIdx.x;
            tab[threadIdx.x] = (uint32_t) kv[0] | (uint32_t) kv[1] << 8 | (uint32_t) kv[2] << 16 | (uint32_t) kv[3] << 24;
        }
    } else if constexpr (Ty<T>::TAB > 0) {
        const uint32_t* src = T == kIQ2_XXS                  ? (const uint32_t*) iq2xxs_grid
                              : T == kIQ2_XS                  ? (const uint32_t*) iq2xs_grid
                              : T == kIQ2_S                   ? (const uint32_t*) iq2s_grid
                              : T == kIQ3_XXS                 ? (const uint32_t*) iq3xxs_grid
                              : T == kIQ1_S || T == kIQ1_M    ? (const uint32_t*) iq1s_grid_gpu
                                                              : (const uint32_t*) iq3s_grid;
        for (int i = threadIdx.x; i < Ty<T>::TAB; i += blockDim.x) tab[i] = src[i];
    }
}

// (#529) DS4_IQ_STRICT_FP: on RDNA 4 (gfx1200 / gfx1201) ROCm's clang contracted the token kernels' (NE 1) and the
// window kernels' (NE 4) multiply-adds differently, so their sums differed in the last bits (ds4_gguf_test there):
// there every rounding is explicit (no contraction, barriers the compiler cannot fold through) and IQ4_XS's rows load
// in 16-bit halves.  Elsewhere - gfx11 and the other AMD GPUs, CUDA - the kernels compile as before (the same bits and
// speed as v1.1.30: the barriers cost a Strix Halo ~7% of its decode).
#if defined(STRATA_USE_HIP) && defined(__clang__) && (defined(__gfx1200__) || defined(__gfx1201__))
#define DS4_IQ_STRICT_FP 1
#else
#define DS4_IQ_STRICT_FP 0
#endif
__device__ __forceinline__ uint32_t ld2(const uint8_t* p) {   // 4 bytes from a 2-byte aligned address
    const uint16_t* q = (const uint16_t*) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}
__device__ __forceinline__ uint32_t ld4(const uint8_t* p) { return *(const uint32_t*) p; }
__device__ __forceinline__ float h2f(const uint8_t* p) { return __half2float(__ushort_as_half(*(const uint16_t*) p)); }
// 7 sign bits (bits 0..6 of v) -> the 8 signs, bit 7 their parity (ggml's ksigns_iq2xs)
__device__ __forceinline__ uint32_t ksigns8(uint32_t v) {
    v &= 127u;
    return v | ((__popc(v) & 1u) << 7);
}
// 4 grid bytes with the signs of bits 0..3 of b applied (bit i: byte i negative) - llama.cpp's HIP form, cheap where the
// SIMD byte instructions (__vcmpne4 / __vsub4) are emulated (Volta, RDNA): every grid byte is nonzero and <= 62, so
// (g ^ 0xff) + 1 = -g in each byte with no carry into the next
__device__ __forceinline__ int sgn4(uint32_t g, uint32_t b) {
    const uint32_t lo = (((b & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    return (int) ((g ^ (lo * 255u)) + lo);
}
__device__ __forceinline__ int2 tab16(uint32_t q4, const uint32_t* t) {   // vecdotq's get_int_from_table_16
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(t[0], t[1], q4 >> sh), hi = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2((int) __byte_perm(tmp[0], tmp[1], 0x6420), (int) __byte_perm(tmp[0], tmp[1], 0x7531));
}
// IQ1_S / IQ1_M: 4 grid values as their int8 codes 8 g +- 1 (g in {-1, 0, 1}): iq1c, iq1_bias (ds4_iq1_codes.hpp)
// IQ1_M's F16 block scale: the top nibbles of its 4 scale words (s01: words 0, 1; s23: words 2, 3)
__device__ __forceinline__ float iq1m_d(uint32_t s01, uint32_t s23) {
    return __half2float(__ushort_as_half((unsigned short) iq1m_d16(s01, s23)));
}
__device__ __forceinline__ int q3k_scale(const uint8_t* sc, int is) {
    const int lo = is < 8 ? sc[is] & 15 : sc[is - 8] >> 4;
    const int hi = (sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3;
    return (lo | (hi << 4)) - 32;
}
__device__ __forceinline__ void q4k_scale(const uint8_t* q, int j, int& d, int& m) {   // ggml's get_scale_min_k4
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// a 32-value sub-block (kb: its index in the row) as 32 int8 codes and its scales: weight i = d s q_i - m, s and m of
// its half (i < 16: s0, m0; else s1, m1)
struct Sub {
    int q[8];
    float d, m0, m1;
    int s0, s1;
};
// IQ1_S / IQ1_M: a sub-block's bytes as 4 words, apart from their decode (the token kernels load the next sub-block's
// while they decode this one's: a step of these small blocks is mostly the loads' latency) - IQ1_S: its qs word, its
// qh, d's F16 bits; IQ1_M: its qs word, its two qh bytes, the 4 scale words (d's nibbles in their tops)
struct Raw1 {
    uint32_t q4, h, s01, s23;
};
template <int T>
__device__ __forceinline__ void iq1_load(const uint8_t* __restrict__ row, int kb, Raw1& r) {
    const int ib = kb & 7;
    const uint8_t* b = row + (size_t) (kb >> 3) * Ty<T>::B;
    if constexpr (T == kIQ1_S) {   // d, qs[32], qh[8] (u16: 4 x 3 index bits, the scale, the delta's sign)
        r.q4 = ld2(b + 2 + 4 * ib);
        r.h = *(const uint16_t*) (b + 34 + 2 * ib);
        r.s01 = *(const uint16_t*) b;
        r.s23 = 0;
    } else {   // qs[32], qh[16] (a byte for two groups of 8), scales[8] (+ d's nibbles)
        r.q4 = ld4(b + 4 * ib);
        r.h = *(const uint16_t*) (b + 32 + 2 * ib);
        r.s01 = ld4(b + 48);
        r.s23 = ld4(b + 52);
    }
}
// CT: IQ1_M's codes from its code table (tab: kIq1mCodes entries of two words, the persistent kernels') - the grid
// lookup and the delta's bias in one load, the same codes
template <int T, bool CT = false>
__device__ __forceinline__ void iq1_decode(const Raw1& r, int kb, const uint32_t* tab, Sub& s) {
    const int ib = kb & 7;
    if constexpr (T == kIQ1_S) {
        const uint32_t bias = iq1_bias(r.h & 0x8000u);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g = tab[((r.q4 >> (8 * l)) & 255u) | (((r.h >> (3 * l)) & 7u) << 8)];
            s.q[2 * l] = iq1c(g << 3, bias);       // values 8 l .. 8 l + 3: the low nibbles
            s.q[2 * l + 1] = iq1c(g >> 1, bias);   // 8 l + 4 .. 8 l + 7: the high ones
        }
        s.s0 = s.s1 = 2 * (int) ((r.h >> 12) & 7u) + 1;
        s.d = __half2float(__ushort_as_half((unsigned short) r.s01)) * 0.125f;
    } else {
#pragma unroll
        for (int l = 0; l < 4; ++l) {   // group l: index bits 4 l .. 4 l + 2 of h, the delta's sign bit 4 l + 3
            if constexpr (CT) {
                const uint2 c = ((const uint2*) tab)[iq1m_entry(r.q4, r.h, l)];
                s.q[2 * l] = (int) c.x;
                s.q[2 * l + 1] = (int) c.y;
            } else {
                const uint32_t g = tab[((r.q4 >> (8 * l)) & 255u) | (((r.h >> (4 * l)) & 7u) << 8)];
                const uint32_t bias = iq1_bias((r.h >> (4 * l + 3)) & 1u);
                s.q[2 * l] = iq1c(g << 3, bias);
                s.q[2 * l + 1] = iq1c(g >> 1, bias);
            }
        }
        iq1m_scales(r.s01, r.s23, ib, s.s0, s.s1);   // (values 0..15, 16..31)
        s.d = iq1m_d(r.s01, r.s23) * 0.125f;
    }
    (void) ib;
}
template <int T, bool CT = false>
__device__ __forceinline__ void decode(const uint8_t* __restrict__ row, int kb, const uint32_t* tab, Sub& s) {
    const int ib = kb & 7;
    const uint8_t* b = row + (size_t) (kQK<T> == 32 ? kb : kb >> 3) * Ty<T>::B;
    if constexpr (T == kMXFP4) {   // codes: value j in byte j's low nibble, j + 16 in its high one (any alignment)
        const uint32_t t[4] = {0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu};   // ggml's kvalues_mxfp4
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t* q = b + 1 + 4 * i;
            const int2 v = tab16((uint32_t) q[0] | (uint32_t) q[1] << 8 | (uint32_t) q[2] << 16 | (uint32_t) q[3] << 24, t);
            s.q[i] = v.x;
            s.q[4 + i] = v.y;
        }
        const uint32_t e = b[0];   // 2^(e - 128): ggml's GGML_E8M0_TO_FP32_HALF (the codes are the E2M1 values doubled)
        s.d = __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
        s.s0 = s.s1 = 1;
        (void) ib;
        (void) tab;
    } else if constexpr (T == kIQ2_XXS) {
        const uint8_t* q = b + 2 + 8 * ib;
        const uint32_t a0 = ld2(q), a1 = ld2(q + 4);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[(a0 >> (8 * l)) & 255u];
            const uint32_t sg = ksigns8(a1 >> (7 * l));
            s.q[2 * l] = sgn4(g.x, sg);
            s.q[2 * l + 1] = sgn4(g.y, sg >> 4);
        }
        s.s0 = s.s1 = 2 * (int) (a1 >> 28) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_XS) {   // 4 x (9-bit grid index | 7 sign bits), a 4-bit scale a half (#357)
        const uint8_t* q = b + 2 + 8 * ib;
        const uint32_t sc = b[66 + ib];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t v = *(const uint16_t*) (q + 2 * l);
            const uint2 g = ((const uint2*) tab)[v & 511u];
            const uint32_t sg = ksigns8(v >> 9);
            s.q[2 * l] = sgn4(g.x, sg);
            s.q[2 * l + 1] = sgn4(g.y, sg >> 4);
        }
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_S) {
        const uint8_t* qs = b + 2;
        const uint32_t i4 = ld2(qs + 4 * ib), sg4 = ld2(qs + 32 + 4 * ib), h = b[66 + ib], sc = b[74 + ib];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[((i4 >> (8 * l)) & 255u) | ((h << (8 - 2 * l)) & 0x300u)];
            const uint32_t sg = sg4 >> (8 * l);
            s.q[2 * l] = sgn4(g.x, sg);
            s.q[2 * l + 1] = sgn4(g.y, sg >> 4);
        }
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ3_XXS) {
        const uint8_t* qs = b + 2;
        const uint32_t w0 = ld2(qs + 8 * ib), w1 = ld2(qs + 8 * ib + 4), aux = ld2(qs + 64 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            const uint32_t sg = ksigns8(aux >> (7 * l));
            s.q[2 * l] = sgn4(tab[(w >> sh) & 255u], sg);
            s.q[2 * l + 1] = sgn4(tab[(w >> (sh + 8)) & 255u], sg >> 4);
        }
        s.s0 = s.s1 = 2 * (int) (aux >> 28) + 1;
        s.d = h2f(b) * 0.25f;
    } else if constexpr (T == kIQ3_S) {
        const uint8_t* qs = b + 2;
        const uint32_t w0 = ld2(qs + 8 * ib), w1 = ld2(qs + 8 * ib + 4), h = b[66 + ib], s4 = ld2(b + 74 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            const uint32_t i1 = ((w >> sh) & 255u) | ((h << (8 - 2 * l)) & 256u);
            const uint32_t i2 = ((w >> (sh + 8)) & 255u) | ((h << (7 - 2 * l)) & 256u);
            const uint32_t sg = s4 >> (8 * l);
            s.q[2 * l] = sgn4(tab[i1], sg);
            s.q[2 * l + 1] = sgn4(tab[i2], sg >> 4);
        }
        s.s0 = s.s1 = 1 + 2 * (int) ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15);
        s.d = h2f(b);
    } else if constexpr (T == kIQ4_XS) {
        const uint32_t sh = *(const uint16_t*) (b + 2);
        const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
        const uint8_t* qs = b + 8 + 16 * ib;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int2 v = tab16(DS4_IQ_STRICT_FP ? ld2(qs + 4 * t) : ld4(qs + 4 * t), tab);
            s.q[t] = v.x;
            s.q[4 + t] = v.y;
        }
        s.s0 = s.s1 = ls - 32;
        s.d = h2f(b);
    } else if constexpr (T == kIQ4_NL) {   // a block of 32 (b: block kb): d, then value j's code in byte j's low nibble
        const uint8_t* qs = b + 2;           // (j < 16) or byte j - 16's high one; 18-byte blocks: 2-byte aligned
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int2 v = tab16(ld2(qs + 4 * t), tab);
            s.q[t] = v.x;
            s.q[4 + t] = v.y;
        }
        s.s0 = s.s1 = 1;
        s.d = h2f(b);
        (void) ib;
    } else if constexpr (T == kQ4_K) {
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const uint8_t* qs = b + 16 + 32 * (ib >> 1);
        const int sh = 4 * (ib & 1);
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) ((ld4(qs + 4 * t) >> sh) & 0x0f0f0f0fu);
        s.s0 = s.s1 = sc;
        s.d = h2f(b);
        float m0 = h2f(b + 2) * (float) m;
#if DS4_IQ_STRICT_FP
        asm volatile("" : "+v"(m0));
#endif
        s.m0 = s.m1 = m0;
    } else if constexpr (T == kQ5_K) {   // Q4_K's nibbles and scales, the 5th bit from qh (bit ib of each value's byte)
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const uint8_t* qs = b + 48 + 32 * (ib >> 1);
        const int sh = 4 * (ib & 1);
#pragma unroll
        for (int t = 0; t < 8; ++t)
            s.q[t] = (int) (((ld4(qs + 4 * t) >> sh) & 0x0f0f0f0fu) | (((ld4(b + 16 + 4 * t) >> ib) & 0x01010101u) << 4));
        s.s0 = s.s1 = sc;
        s.d = h2f(b);
        float m0 = h2f(b + 2) * (float) m;
#if DS4_IQ_STRICT_FP
        asm volatile("" : "+v"(m0));
#endif
        s.m0 = s.m1 = m0;
    } else if constexpr (T == kQ2_K) {
        const int n = ib >> 2, j = ib & 3;
        const uint8_t* qs = b + 16 + 32 * n;
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) ((ld4(qs + 4 * t) >> (2 * j)) & 0x03030303u);
        const int c0 = b[8 * n + 2 * j], c1 = b[8 * n + 2 * j + 1];
        const float dmin = h2f(b + 82);
        s.s0 = c0 & 15;
        s.s1 = c1 & 15;
        s.d = h2f(b + 80);
        float m0 = dmin * (float) (c0 >> 4), m1 = dmin * (float) (c1 >> 4);
#if DS4_IQ_STRICT_FP
        asm volatile("" : "+v"(m0), "+v"(m1));
#endif
        s.m0 = m0;
        s.m1 = m1;
    } else if constexpr (T == kIQ1_S || T == kIQ1_M) {
        Raw1 r;
        iq1_load<T>(row, kb, r);
        iq1_decode<T, CT>(r, kb, tab, s);
        (void) b;
    } else if constexpr (T == kQ6_K) {   // quarter ib & 3 of half ib >> 2: ggml's dequantize_row_q6_K's loop
        const int hf = ib >> 2, qt = ib & 3;
        const uint8_t* ql = b + 64 * hf + 32 * (qt & 1);
        const uint8_t* qh = b + 128 + 32 * hf;
        const int8_t* sc = (const int8_t*) (b + 192 + 8 * hf + 2 * qt);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t v = ((ld2(ql + 4 * t) >> (4 * (qt >> 1))) & 0x0f0f0f0fu) |
                               (((ld2(qh + 4 * t) >> (2 * qt)) & 0x03030303u) << 4);
            // (each byte v - 32 for v in 0..63, no carry between them: v ^ 32, the top two bits set where v < 32)
            const uint32_t lo = (~v) & 0x20202020u;
            s.q[t] = (int) ((v ^ 0x20202020u) | (lo << 1) | (lo << 2));
        }
        s.s0 = sc[0];
        s.s1 = sc[1];
        s.d = h2f(b + 208);
    } else if constexpr (T == kQ8_0) {   // a block of 32 (b: block kb), 2-byte aligned
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) ld2(b + 2 + 4 * t);
        s.s0 = s.s1 = 1;
        s.d = h2f(b);
        (void) ib;
        (void) tab;
    } else {   // Q3_K
        const int n = ib >> 2, j = ib & 3;
        const uint8_t* qs = b + 32 + 32 * n;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t ql = (ld2(qs + 4 * t) >> (2 * j)) & 0x03030303u;
            const uint32_t neg = ((~ld2(b + 4 * t)) >> (4 * n + j)) & 0x01010101u;
            s.q[t] = (int) (ql | (neg * 0xFCu));   // (q - 4 in a byte: q + 0xFC, no carry for q <= 3)
        }
        s.s0 = q3k_scale(b + 96, 8 * n + 2 * j);
        s.s1 = q3k_scale(b + 96, 8 * n + 2 * j + 1);
        s.d = h2f(b + 108);
    }
}
__device__ __forceinline__ float fadd_no_fma(float a, float b) {
#if DS4_IQ_STRICT_FP
#pragma clang fp contract(off)
    float r = a + b;
    asm volatile("" : "+v"(r));
    return r;
#else
    return __fadd_rn(a, b);
#endif
}
__device__ __forceinline__ float fsub_no_fma(float a, float b) {
#if DS4_IQ_STRICT_FP
#pragma clang fp contract(off)
    float r = a - b;
    asm volatile("" : "+v"(r));
    return r;
#else
    return __fsub_rn(a, b);
#endif
}
__device__ __forceinline__ float fmul_no_fma(float a, float b) {
#if DS4_IQ_STRICT_FP
#pragma clang fp contract(off)
    float r = a * b;
    asm volatile("" : "+v"(r));
    return r;
#else
    return __fmul_rn(a, b);
#endif
}
__device__ __forceinline__ float fma_no_contract(float a, float b, float c) {
#if DS4_IQ_STRICT_FP
#pragma clang fp contract(off)
    asm volatile("" : "+v"(a), "+v"(b), "+v"(c));
    float r = __builtin_fmaf(a, b, c);
    asm volatile("" : "+v"(r));
    return r;
#else
    return fmaf(a, b, c);
#endif
}

// the sub-block against 32 int8 activations (xa: values 0..15, xb: 16..31), before their scale: an exact integer dot,
// its scales in F32 (explicit roundings: every instantiation the same bits)
template <int T>
__device__ __forceinline__ float sub_dot(const Sub& s, const int4& xa, const int4& xb) {
    int lo = __dp4a(s.q[0], xa.x, 0);
    lo = __dp4a(s.q[1], xa.y, lo);
    lo = __dp4a(s.q[2], xa.z, lo);
    lo = __dp4a(s.q[3], xa.w, lo);
    int hi = __dp4a(s.q[4], xb.x, 0);
    hi = __dp4a(s.q[5], xb.y, hi);
    hi = __dp4a(s.q[6], xb.z, hi);
    hi = __dp4a(s.q[7], xb.w, hi);
    float v = fmul_no_fma(s.d, (float) (s.s0 * lo + s.s1 * hi));
    if constexpr (Ty<T>::MIN) {
        int sl = __dp4a(0x01010101, xa.x, 0);
        sl = __dp4a(0x01010101, xa.y, sl);
        sl = __dp4a(0x01010101, xa.z, sl);
        sl = __dp4a(0x01010101, xa.w, sl);
        int sh = __dp4a(0x01010101, xb.x, 0);
        sh = __dp4a(0x01010101, xb.y, sh);
        sh = __dp4a(0x01010101, xb.z, sh);
        sh = __dp4a(0x01010101, xb.w, sh);
        v = fsub_no_fma(v, fadd_no_fma(fmul_no_fma(s.m0, (float) sl), fmul_no_fma(s.m1, (float) sh)));
    }
    return v;
}

// values [8 p, 8 p + 8) of sub-block kb, as F32 (the prompt sets' tiles): ggml's dequantize_row_* values
template <int T>
__device__ __forceinline__ void deq8(const uint8_t* __restrict__ row, int kb, int p, const uint32_t* tab, float* w) {
    const int ib = kb & 7;
    const uint8_t* b = row + (size_t) (kQK<T> == 32 ? kb : kb >> 3) * Ty<T>::B;
    if constexpr (T == kIQ2_XXS) {
        const uint8_t* q = b + 2 + 8 * ib;
        const uint32_t a1 = ld2(q + 4);
        const uint2 g = ((const uint2*) tab)[q[p]];
        const uint32_t v = (a1 >> (7 * p)) & 127u, sg = v | ((__popc(v) & 1u) << 7);
        const float dl = h2f(b) * (float) (2 * (int) (a1 >> 28) + 1) * 0.125f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float gv = (float) ((j < 4 ? g.x >> (8 * j) : g.y >> (8 * (j - 4))) & 255u);
            w[j] = (sg >> j) & 1u ? -(dl * gv) : dl * gv;
        }
    } else if constexpr (T == kIQ2_XS) {   // (#357)
        const uint32_t v = *(const uint16_t*) (b + 2 + 8 * ib + 2 * p), sc = b[66 + ib];
        const uint2 g = ((const uint2*) tab)[v & 511u];
        const uint32_t sg = ksigns8(v >> 9);
        const float dl = h2f(b) * (float) (2 * (int) (p < 2 ? sc & 15 : sc >> 4) + 1) * 0.125f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float gv = (float) ((j < 4 ? g.x >> (8 * j) : g.y >> (8 * (j - 4))) & 255u);
            w[j] = (sg >> j) & 1u ? -(dl * gv) : dl * gv;
        }
    } else if constexpr (T == kIQ2_S) {
        const uint8_t* qs = b + 2;
        const uint32_t idx = qs[4 * ib + p] | ((((uint32_t) b[66 + ib]) << (8 - 2 * p)) & 0x300u);
        const uint32_t sg = qs[32 + 4 * ib + p], sc = b[74 + ib];
        const uint2 g = ((const uint2*) tab)[idx];
        const float dl = h2f(b) * (float) (2 * (int) (p < 2 ? sc & 15 : sc >> 4) + 1) * 0.125f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float gv = (float) ((j < 4 ? g.x >> (8 * j) : g.y >> (8 * (j - 4))) & 255u);
            w[j] = (sg >> j) & 1u ? -(dl * gv) : dl * gv;
        }
    } else if constexpr (T == kIQ3_XXS || T == kIQ3_S) {
        const uint8_t* qs = b + 2;
        uint32_t g1, g2, sg;
        float dl;
        if constexpr (T == kIQ3_XXS) {
            const uint32_t aux = ld2(qs + 64 + 4 * ib), v = (aux >> (7 * p)) & 127u;
            g1 = tab[qs[8 * ib + 2 * p]];
            g2 = tab[qs[8 * ib + 2 * p + 1]];
            sg = v | ((__popc(v) & 1u) << 7);
            dl = h2f(b) * (float) (2 * (int) (aux >> 28) + 1) * 0.25f;
        } else {
            const uint32_t h = b[66 + ib];
            g1 = tab[qs[8 * ib + 2 * p] | ((h << (8 - 2 * p)) & 256u)];
            g2 = tab[qs[8 * ib + 2 * p + 1] | ((h << (7 - 2 * p)) & 256u)];
            sg = b[74 + 4 * ib + p];
            dl = h2f(b) * (float) (1 + 2 * (int) ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15));
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float gv = (float) ((j < 4 ? g1 >> (8 * j) : g2 >> (8 * (j - 4))) & 255u);
            w[j] = (sg >> j) & 1u ? -(dl * gv) : dl * gv;
        }
    } else if constexpr (T == kIQ4_XS) {
        const uint32_t sh = *(const uint16_t*) (b + 2);
        const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = h2f(b) * (float) (ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib + 8 * (p & 1);
        const int8_t* kv = (const int8_t*) tab;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = dl * (float) kv[p < 2 ? qs[j] & 15 : qs[j] >> 4];
    } else if constexpr (T == kIQ4_NL) {   // (block kb: 32 values, as IQ4_XS's sub-block)
        const float dl = h2f(b);
        const uint8_t* qs = b + 2 + 8 * (p & 1);
        const int8_t* kv = (const int8_t*) tab;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = dl * (float) kv[p < 2 ? qs[j] & 15 : qs[j] >> 4];
        (void) ib;
    } else if constexpr (T == kQ4_K) {
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const float d1 = h2f(b) * (float) sc, m1 = h2f(b + 2) * (float) m;
        const uint8_t* qs = b + 16 + 32 * (ib >> 1) + 8 * p;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = d1 * (float) ((ib & 1) ? qs[j] >> 4 : qs[j] & 15) - m1;
    } else if constexpr (T == kQ5_K) {
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const float d1 = h2f(b) * (float) sc, m1 = h2f(b + 2) * (float) m;
        const uint8_t* qs = b + 48 + 32 * (ib >> 1) + 8 * p;
        const uint8_t* qh = b + 16 + 8 * p;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            w[j] = d1 * (float) (((ib & 1) ? qs[j] >> 4 : qs[j] & 15) | (((qh[j] >> ib) & 1) << 4)) - m1;
    } else if constexpr (T == kQ2_K) {
        const int n = ib >> 2, jj = ib & 3;
        const int c = b[8 * n + 2 * jj + (p >> 1)];
        const float dl = h2f(b + 80) * (float) (c & 15), ml = h2f(b + 82) * (float) (c >> 4);
        const uint8_t* qs = b + 16 + 32 * n + 8 * p;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = dl * (float) ((qs[j] >> (2 * jj)) & 3) - ml;
    } else if constexpr (T == kIQ1_S || T == kIQ1_M) {   // ggml's dl (grid + delta), group p of the sub-block
        uint32_t g;
        float dl, dt;
        if constexpr (T == kIQ1_S) {
            const uint32_t h = *(const uint16_t*) (b + 34 + 2 * ib);
            g = tab[b[2 + 4 * ib + p] | (((h >> (3 * p)) & 7u) << 8)];
            dl = h2f(b) * (float) (2 * (int) ((h >> 12) & 7u) + 1);
            dt = h & 0x8000u ? -IQ1S_DELTA : IQ1S_DELTA;
        } else {
            const uint32_t h = *(const uint16_t*) (b + 32 + 2 * ib);
            const uint32_t s01 = ld4(b + 48), s23 = ld4(b + 52);
            const uint32_t sw = ((ib >> 1) < 2 ? s01 : s23) >> (16 * ((ib >> 1) & 1) + 6 * (ib & 1) + 3 * (p >> 1));
            g = tab[b[4 * ib + p] | (((h >> (4 * p)) & 7u) << 8)];
            dl = iq1m_d(s01, s23) * (float) (2 * (int) (sw & 7u) + 1);
            dt = (h >> (4 * p + 3)) & 1u ? -IQ1S_DELTA : IQ1S_DELTA;   // (ggml's dequantize_row_iq1_m: IQ1S_DELTA)
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = dl * ((float) ((int) ((g >> (8 * (j & 3) + 4 * (j >> 2))) & 15u) - 1) + dt);
    } else if constexpr (T == kQ6_K) {   // d * scale * q, ggml's order (ds4_q6k.cuh's q6k_value)
        const int hf = ib >> 2, qt = ib & 3;
        const uint8_t* ql = b + 64 * hf + 32 * (qt & 1) + 8 * p;
        const uint8_t* qh = b + 128 + 32 * hf + 8 * p;
        const float dl = __fmul_rn(h2f(b + 208), (float) ((const int8_t*) (b + 192 + 8 * hf + 2 * qt))[p >> 1]);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            w[j] = __fmul_rn(dl, (float) ((int) (((ql[j] >> (4 * (qt >> 1))) & 15) | (((qh[j] >> (2 * qt)) & 3) << 4)) - 32));
    } else if constexpr (T == kQ8_0) {   // (blocks of 32: block kb, not kb >> 3)
        const uint8_t* q8 = row + (size_t) kb * Ty<T>::B;
        const float d = h2f(q8);
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = d * (float) (int8_t) q8[2 + 8 * p + j];
        (void) b;
    } else {   // Q3_K
        const int n = ib >> 2, jj = ib & 3;
        const float dl = h2f(b + 108) * (float) q3k_scale(b + 96, 8 * n + 2 * jj + (p >> 1));
        const uint8_t* qs = b + 32 + 32 * n + 8 * p;
        const uint8_t* hm = b + 8 * p;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            w[j] = dl * (float) ((int) ((qs[j] >> (2 * jj)) & 3) - (((hm[j] >> (4 * n + jj)) & 1) ? 0 : 4));
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
#if DS4_IQ_STRICT_FP
    for (int o = 16; o > 0; o >>= 1) v = fadd_no_fma(v, __shfl_xor_sync(0xffffffffu, v, o));
#else
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
#endif
    return v;
}
__device__ __forceinline__ float swiglu(float gv, float uv, float limit) {   // (ds4_gguf.cu's)
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return fmul_no_fma(__fdiv_rn(gv, fadd_no_fma(1.0f, expf(-gv))), uv);
}
// MiniMax-M3.1's swigluoai (mm3_exl3.cu's: the same operations)
__device__ __forceinline__ float swigluoai(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(__fmul_rn(-kOaiAlpha, gv)))), __fadd_rn(uv, 1.0f));
}
// The gate / up kernels' epilogue, a template parameter EPI (DeepSeek's kernels: EPI 0, the code they always had):
//   0  gate and up in one pass, h = DeepSeek's clamped SwiGLU      1  ... h = swigluoai (Args::act kActOai)
//   2  the gate alone, its F32 sums out (h, or SetArgs::graw)       3  the up alone, h = swigluoai(those sums, up)
// 2 then 3: a layer whose gate and up are of different types (Args::tu) - each value the one pass's
template <int EPI>
constexpr bool kBoth = EPI < 2;   // (gate and up in this pass)
template <int EPI>
__device__ __forceinline__ float act_out(float gv, float uv, float limit) {
    if constexpr (EPI == 0) return swiglu(gv, uv, limit);
    else return swigluoai(gv, uv, limit);
}
template <int EPI>
constexpr int kRole0 = EPI == 3 ? 1 : 0;   // the matrix the pass reads first (3: the up)
__device__ __forceinline__ bool group(const Args& a, int grp, const uint8_t*& blob, int& m0, int& m1) {
    if (a.grp_ptr == nullptr) {
        const unsigned long long p = a.plan_ptr[grp];
        if (p == 0ull) return false;
        blob = (const uint8_t*) p;
        m0 = grp;
        m1 = grp + 1;
        return true;
    }
    if (grp >= *a.n_groups) return false;
    blob = (const uint8_t*) a.grp_ptr[grp];
    m0 = a.grp_start[grp];
    m1 = a.grp_start[grp + 1];
    return true;
}

// ---------------------------------------------------------------- token / window experts
// RW rows a warp, 32 / RW lanes a row (a lane a sub-block a step), their sums over those lanes
template <int RW>
__device__ __forceinline__ float lanes_sum(float v) {
#pragma unroll
#if DS4_IQ_STRICT_FP
    for (int o = 16 / RW; o > 0; o >>= 1) v = fadd_no_fma(v, __shfl_xor_sync(0xffffffffu, v, o));
#else
    for (int o = 16 / RW; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
#endif
    return v;
}
constexpr int kGuRW = 1, kDnRW = 2;   // rows a warp: gate / up (n_embd 5120: 10 sub-blocks a lane), down (2304: 9)

// AH: a lane's next sub-block's raw words loaded a step ahead (IQ1 only); else each decoded from the row as before
// (CT: IQ1_M through its code table)
template <int T, bool AH, bool CT = false>
struct Ahead {
    static constexpr bool on = AH && (T == kIQ1_S || T == kIQ1_M);
    Raw1 r{};
    __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int kb) {
        if constexpr (on) iq1_load<T>(row, kb, r);
    }
    __device__ __forceinline__ void get(const uint8_t* __restrict__ row, int kb, const uint32_t* tab, Sub& s) const {
        if constexpr (on) iq1_decode<T, CT>(r, kb, tab, s);
        else decode<T, CT>(row, kb, tab, s);
    }
};

// A block's rows in RP passes of kWarps RW rows (RP > 1: IQ1_S / IQ1_M, whose 8 KB grid table a block loads would
// otherwise rival the weights it reads - 50 or 56 bytes a block).  Every row's sums are the same whatever RP and AH (the
// same lanes, the same order), so the window's entries keep the token path's bits.

// ---- persistent grids (PS; ds4_iq.cuh's persist_on, DS4_IQ_PERSIST)
// The kernels below load their type's codebook into shared memory at a block's start; for a large one (8 KiB and more:
// IQ1_S, IQ1_M, IQ2_S) that load rivals the block's weights.  PS: a grid of the blocks the device holds at once, each
// block loading its codebook once (when its first live work item comes) and then taking work items round the grid - an
// item the full grid's block (x, group), group by group.  An item's rows are the same warps' and lanes' as that block's,
// summed in the same order: every output keeps its bits, whatever the grid.  The smaller codebooks keep the full grid (a
// block there loads little, and one of an empty group leaves at once).  CUDA's maya-ds4 only: HIP and maya-mm3's build
// of this file (MAYA_DS4_IQ_ONLY) keep the full grids, and so do MiniMax's epilogues (EPI 1..3).
#if defined(STRATA_USE_HIP) || defined(MAYA_DS4_IQ_ONLY)
#define DS4_IQ_PERSIST_BUILD 0
#else
#define DS4_IQ_PERSIST_BUILD 1
#endif
template <int T, int EPI>
constexpr bool kPersist = DS4_IQ_PERSIST_BUILD && Ty<T>::TAB * 4 >= 8192 && EPI == 0;
// IQ1_M in a persistent kernel: its codes from a table of kIq1mCodes entries (ds4_iq1_codes.hpp), built once a block
template <int T, bool PS>
constexpr bool kCodes = PS && T == kIQ1_M;
template <int T, bool CT>   // (the words of shared memory the codebook takes)
constexpr int kTabW = CT ? 2 * kIq1mCodes : Ty<T>::TAB > 0 ? Ty<T>::TAB : 1;
template <int T, bool CT>
__device__ __forceinline__ void load_tab_k(uint32_t* tab) {
    if constexpr (CT) {   // (entry j: grid entry j mod 2048's codes, the delta's sign j >> 11)
        uint2* c = (uint2*) tab;
        for (int j = threadIdx.x; j < kIq1mCodes; j += blockDim.x) {
            uint32_t lo, hi;
            iq1m_codes(iq1s_grid_gpu[j & 2047], (uint32_t) j >> 11, lo, hi);
            c[j] = make_uint2(lo, hi);
        }
    } else {
        load_tab<T>(tab);
    }
}
// the groups a token / window kernel's items cover (a window's: those of its count)
__device__ __forceinline__ int live_groups(const Args& a, int groups) {
    return a.grp_ptr != nullptr ? min(groups, *a.n_groups) : groups;
}

// grid (n_ff / (8 RW RP), groups): RW rows of the gate and of the up a warp a pass, NE of the group's entries a pass
// (each sub-block decoded once for them) -> h = the clamped swiglu (EPI: act_out's).  PS (RP 1): the persistent grid
// above, an item the full grid's block (bx, group)
template <int T, int NE, int RP = 1, bool AH = false, int RW = kGuRW, int EPI = 0, bool PS = false>
__global__ void __launch_bounds__(kThreads) gate_up_kernel(const Args a, int groups) {
#if DS4_IQ_STRICT_FP
#pragma clang fp contract(off)
#endif
    pdl_wait();
    pdl_trigger();
    constexpr int LPR = 32 / RW;
    constexpr bool TWO = kBoth<EPI>;
    constexpr bool CT = kCodes<T, PS>;
    static_assert(!PS || RP == 1, "a persistent item: one pass of rows");
    __shared__ __align__(16) uint32_t tab[kTabW<T, CT>];
    __shared__ int es[NE], xr[NE];
    const int lane32 = threadIdx.x & 31, lane = lane32 % LPR;
    const size_t rb = row_bytes_t<T>(a.n_embd);
    const int nsb = a.n_embd / 32;
    const int nbx = PS ? a.n_ff / (kWarps * RW) : 1;   // (the full grid's blocks a group)
    const int items = PS ? live_groups(a, groups) * nbx : 1;
    bool loaded = false;
    for (int it = PS ? (int) blockIdx.x : 0; it < items; it += PS ? (int) gridDim.x : 1) {
        const int grp = PS ? it / nbx : (int) blockIdx.y, bx = PS ? it % nbx : (int) blockIdx.x;
        const uint8_t* blob;
        int m0, m1;
        if (!group(a, grp, blob, m0, m1)) continue;   // (uniform; the full grid's block: its one item)
        if (!loaded) {   // (the codebook once a block; the barrier before the first use is the loop's below)
            load_tab_k<T, CT>(tab);
            loaded = true;
        }
        for (int mm = m0; mm < m1; mm += NE) {
            const int ne = min(NE, m1 - mm);
            __syncthreads();
            if ((int) threadIdx.x < ne) {
                const int e = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
                es[threadIdx.x] = e;
                xr[threadIdx.x] = a.grp_ptr != nullptr ? e >> 3 : 0;
            }
            __syncthreads();
            for (int p = 0; p < RP; ++p) {
                const int row = ((bx * RP + p) * kWarps + (threadIdx.x >> 5)) * RW + lane32 / LPR;
                if (row >= a.n_ff) continue;   // (a whole warp: n_ff % (8 RW) == 0)
                const uint8_t* g = blob + a.tr[kRole0<EPI>] + (size_t) row * rb;
                const uint8_t* u = blob + a.tr[1] + (size_t) row * rb;
                float ag[NE], au[NE];
#pragma unroll
                for (int i = 0; i < NE; ++i) ag[i] = au[i] = 0.0f;
                Ahead<T, AH, CT> pg, pu;
                if (lane < nsb) {
                    pg.load(g, lane);
                    if constexpr (TWO) pu.load(u, lane);
                }
                for (int kb = lane; kb < nsb; kb += LPR) {
                    Ahead<T, AH, CT> ng = pg, nu = pu;
                    if (kb + LPR < nsb) {
                        ng.load(g, kb + LPR);
                        if constexpr (TWO) nu.load(u, kb + LPR);
                    }
                    Sub sg, su;
                    pg.get(g, kb, tab, sg);
                    if constexpr (TWO) pu.get(u, kb, tab, su);
#pragma unroll
                    for (int i = 0; i < NE; ++i)
                        if (i < ne) {
                            const int4* xp = (const int4*) (a.xq + (size_t) xr[i] * a.n_embd + (size_t) kb * 32);
                            const int4 xa = xp[0], xb = xp[1];
                            const float dx = a.xd[(size_t) xr[i] * nsb + kb];
                            ag[i] = fma_no_contract(dx, sub_dot<T>(sg, xa, xb), ag[i]);
                            if constexpr (TWO) au[i] = fma_no_contract(dx, sub_dot<T>(su, xa, xb), au[i]);
                        }
                    pg = ng;
                    if constexpr (TWO) pu = nu;
                }
#pragma unroll
                for (int i = 0; i < NE; ++i)
                    if (i < ne) {
                        float* hp = &a.h[(size_t) es[i] * a.n_ff + row];
                        if constexpr (TWO) {
                            const float gv = lanes_sum<RW>(ag[i]), uv = lanes_sum<RW>(au[i]);
                            if (lane == 0) *hp = act_out<EPI>(gv, uv, a.limit);
                        } else {
                            const float v = lanes_sum<RW>(ag[i]);   // (EPI 2: the gate's sums; 3: the up's)
                            if (lane == 0) *hp = EPI == 2 ? v : swigluoai(*hp, v, a.limit);
                        }
                    }
            }
        }
    }
}

// grid (n_embd / (8 RW RP), groups): RW rows of the down a warp a pass for NE entries a pass, against their int8 h ->
// part.  PS (RP 1): as gate_up_kernel's
template <int T, int NE, int RP = 1, bool AH = false, int RW = kDnRW, bool PS = false>
__global__ void __launch_bounds__(kThreads) down_kernel(const Args a, int groups) {
#if DS4_IQ_STRICT_FP
#pragma clang fp contract(off)
#endif
    pdl_wait();
    pdl_trigger();
    constexpr int LPR = 32 / RW;
    constexpr bool CT = kCodes<T, PS>;
    static_assert(!PS || RP == 1, "a persistent item: one pass of rows");
    __shared__ __align__(16) uint32_t tab[kTabW<T, CT>];
    __shared__ int es[NE];
    const int lane32 = threadIdx.x & 31, lane = lane32 % LPR;
    const size_t rb = row_bytes_t<T>(a.n_ff);
    const int nsb = a.n_ff / 32;
    const int nbx = PS ? a.n_embd / (kWarps * RW) : 1;
    const int items = PS ? live_groups(a, groups) * nbx : 1;
    bool loaded = false;
    for (int it = PS ? (int) blockIdx.x : 0; it < items; it += PS ? (int) gridDim.x : 1) {
        const int grp = PS ? it / nbx : (int) blockIdx.y, bx = PS ? it % nbx : (int) blockIdx.x;
        const uint8_t* blob;
        int m0, m1;
        if (!group(a, grp, blob, m0, m1)) continue;
        if (!loaded) {
            load_tab_k<T, CT>(tab);
            loaded = true;
        }
        for (int mm = m0; mm < m1; mm += NE) {
            const int ne = min(NE, m1 - mm);
            __syncthreads();
            if ((int) threadIdx.x < ne) es[threadIdx.x] = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
            __syncthreads();
            for (int p = 0; p < RP; ++p) {
                const int row = ((bx * RP + p) * kWarps + (threadIdx.x >> 5)) * RW + lane32 / LPR;
                if (row >= a.n_embd) continue;
                const uint8_t* d = blob + a.tr[2] + (size_t) row * rb;
                float ad[NE];
#pragma unroll
                for (int i = 0; i < NE; ++i) ad[i] = 0.0f;
                Ahead<T, AH, CT> pd;
                if (lane < nsb) pd.load(d, lane);
                for (int kb = lane; kb < nsb; kb += LPR) {
                    Ahead<T, AH, CT> nd = pd;
                    if (kb + LPR < nsb) nd.load(d, kb + LPR);
                    Sub sd;
                    pd.get(d, kb, tab, sd);
                    pd = nd;
#pragma unroll
                    for (int i = 0; i < NE; ++i)
                        if (i < ne) {
                            const int4* hp = (const int4*) (a.hq + (size_t) es[i] * a.n_ff + (size_t) kb * 32);
                            const int4 ha = hp[0], hb = hp[1];
                            ad[i] = fma_no_contract(a.hd[(size_t) es[i] * nsb + kb], sub_dot<T>(sd, ha, hb), ad[i]);
                        }
                }
#pragma unroll
                for (int i = 0; i < NE; ++i)
                    if (i < ne) {
                        const float v = lanes_sum<RW>(ad[i]);
                        if (lane == 0) a.part[(size_t) es[i] * a.n_embd + row] = v;
                    }
            }
        }
    }
}

// ---------------------------------------------------------------- the prompt path's sets
constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
static_assert((BM / TM) * (BN / TN) == kThreads, "one output tile a thread");

// F32 FMAs (any device): a block RM rows (64, or 16 for experts of a few rows: a short prompt's) x BN outputs, a 32-value
// sub-block of K a step, the weights dequantised into shared memory (each thread 8 values of a row), each thread RT x TN
// outputs.  (deq8 is the tile loader a tensor-core form takes too.)
template <int T, bool GATE_UP, int RM = BM, int EPI = 0>
__global__ void __launch_bounds__(kThreads) set_kernel(const SetArgs a) {
    pdl_wait();
    pdl_trigger();
    constexpr int RT = RM / (kThreads / (BN / TN));
    static_assert(RT >= 1 && RT * (kThreads / (BN / TN)) == RM, "the block's rows");
    constexpr bool TWO = GATE_UP && kBoth<EPI>;   // (gate and up in this pass; EPI 2 / 3: one of them, as wg)
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ float wg[32][BN + 1];
    __shared__ float wu[TWO ? 32 : 1][BN + 1];
    __shared__ float xs[RM][33];
    __shared__ int src[RM];
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * RM;
    if (r0 >= hi || hi - lo <= a.skip_le) return;   // (uniform; skip_le: the light kernels' experts)
    if (a.skip_ge > 0 && hi - lo >= a.skip_ge) return;   // (the caller's FP16 GEMMs')
    load_tab<T>(tab);
    const int nr = min(RM, hi - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const size_t rb = row_bytes_t<T>(K);
    const int tid = (int) threadIdx.x;
    if (tid < RM) src[tid] = tid < nr ? (GATE_UP ? a.idx[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const int ty = tid / (BN / TN), tx = tid % (BN / TN);
    const int tr = tid >> 2, tq = tid & 3;
    float ag[RT][TN], au[RT][TN];
#pragma unroll
    for (int i = 0; i < RT; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) ag[i][j] = au[i][j] = 0.0f;
    for (int kb = 0; kb < K / 32; ++kb) {
        float w8[8];
        deq8<T>(blob + a.tr[GATE_UP ? kRole0<EPI> : 2] + (size_t) (c0 + tr) * rb, kb, tq, tab, w8);
#pragma unroll
        for (int j = 0; j < 8; ++j) wg[tq * 8 + j][tr] = w8[j];
        if (TWO) {
            deq8<T>(blob + a.tr[1] + (size_t) (c0 + tr) * rb, kb, tq, tab, w8);
#pragma unroll
            for (int j = 0; j < 8; ++j) wu[tq * 8 + j][tr] = w8[j];
        }
        for (int i = tid; i < RM * 32; i += kThreads) {
            const int r = i / 32, c = i % 32, sr = src[r];
            float v = 0.0f;
            if (sr >= 0) {
                const size_t at = (size_t) sr * a.ldx + (size_t) kb * 32 + c;
                v = GATE_UP ? a.xf[at] : __half2float(__ushort_as_half(a.xh[at]));
            }
            xs[r][c] = v;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < 32; ++kk) {
            float xv[RT], gw[TN], uw[TN];
#pragma unroll
            for (int i = 0; i < RT; ++i) xv[i] = xs[ty * RT + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                gw[j] = wg[kk][tx * TN + j];
                if (TWO) uw[j] = wu[kk][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < RT; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    ag[i][j] = fmaf(xv[i], gw[j], ag[i][j]);
                    if (TWO) au[i][j] = fmaf(xv[i], uw[j], au[i][j]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < RT; ++i) {
        const int r = ty * RT + i;
        if (r >= nr) continue;
        const size_t lr = (size_t) (r0 + r);
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = c0 + tx * TN + j;
            if constexpr (!GATE_UP) a.y[lr * a.ldo + c] = ag[i][j];
            else if constexpr (TWO)
                a.h[lr * a.ldo + c] = __half_as_ushort(__float2half_rn(act_out<EPI>(ag[i][j], au[i][j], a.limit)));
            else if constexpr (EPI == 2) a.graw[lr * a.n_ff + c] = ag[i][j];
            else a.h[lr * a.ldo + c] = __half_as_ushort(__float2half_rn(swigluoai(a.graw[lr * a.n_ff + c], ag[i][j], a.limit)));
        }
    }
}

#if !defined(STRATA_USE_HIP) || STRATA_HAS_WMMA
// the same on the tensor cores (sm_70+; AMD: RDNA 3 / 4's, rocWMMA): 8 warps each 16 rows x 32 outputs, the tiles
// F16 in shared memory
constexpr int kLdT = 40;
template <int T, bool GATE_UP, int EPI = 0>
__global__ void __launch_bounds__(kThreads) set_tc_kernel(const SetArgs a) {
    pdl_wait();
    pdl_trigger();
#if defined(STRATA_USE_HIP) && !DS4_ROCWMMA
    __builtin_trap();   // (a device pass for a card without WMMA: never launched there - tc_ok)
#else
#if defined(STRATA_USE_HIP)
    namespace wm = rocwmma;
    using wh = _Float16;
#else
    namespace wm = nvcuda::wmma;
    using wh = __half;
#endif
    constexpr bool TWO = GATE_UP && kBoth<EPI>;   // (gate and up in this pass; EPI 2 / 3: one of them, as wg)
    struct Tiles {
        __half x[BM][kLdT], wg[BN][kLdT], wu[TWO ? BN : 1][kLdT];
    };
    // the results leave through a 16 x 16 tile a warp in the tiles' room (the gate's and the up's accumulators hold the
    // same elements, so the SwiGLU runs on them in registers): a block's shared memory is its tiles' alone - 15 KB, not
    // a 64 x 64 staging of both (35 KB), so more blocks fit a CU to hide the loads (a Radeon 8065S, 32 experts x 128
    // rows: 7.7-11.4 -> 13.6-14.6 TFLOP/s; a 4K prompt +8%)
    static_assert(sizeof(Tiles) >= (size_t) (kThreads / 32) * 16 * 16 * sizeof(float), "the warps' output tiles");
    __shared__ __align__(32) unsigned char smem[sizeof(Tiles)];
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ int src[BM];
    Tiles& Tl = *reinterpret_cast<Tiles*>(smem);
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi || hi - lo <= a.skip_le) return;   // (uniform; skip_le: the light kernels' experts)
    if (a.skip_ge > 0 && hi - lo >= a.skip_ge) return;   // (the caller's FP16 GEMMs')
    load_tab<T>(tab);
    const int nr = min(BM, hi - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const size_t rb = row_bytes_t<T>(K);
    const int tid = (int) threadIdx.x, warp = tid >> 5;
    if (tid < BM) src[tid] = tid < nr ? (GATE_UP ? a.idx[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const int wm_ = warp >> 1, wn = (warp & 1) * 2;
    wm::fragment<wm::accumulator, 16, 16, 16, float> fg[2], fu[2];
    for (int j = 0; j < 2; ++j) {
        wm::fill_fragment(fg[j], 0.0f);
        wm::fill_fragment(fu[j], 0.0f);
    }
    const int tr = tid >> 2, tq = tid & 3;
    for (int kb = 0; kb < K / 32; ++kb) {
        for (int m = 0; m < (TWO ? 2 : 1); ++m) {
            float w8[8];
            deq8<T>(blob + a.tr[GATE_UP ? (TWO ? m : kRole0<EPI>) : 2] + (size_t) (c0 + tr) * rb, kb, tq, tab, w8);
            __half* dst = m == 0 ? &Tl.wg[tr][tq * 8] : &Tl.wu[tr][tq * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) dst[j] = __float2half_rn(w8[j]);
        }
        const int sr = src[tr];
        if (a.xh_vec) {   // FP16 rows, 16-byte aligned: 8 values in one load (gate / up: the caller's rows_f16 of the same F32
            // rows - __float2half_rn as below, so the same bits - not 36 conversions of them, a block column each)
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (sr >= 0) v = *reinterpret_cast<const uint4*>(a.xh + (size_t) sr * a.ldx + (size_t) kb * 32 + tq * 8);
            *reinterpret_cast<uint4*>(&Tl.x[tr][tq * 8]) = v;
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                __half v = __float2half_rn(0.0f);
                if (sr >= 0) {
                    const size_t at = (size_t) sr * a.ldx + (size_t) kb * 32 + tq * 8 + j;
                    v = GATE_UP ? __float2half_rn(a.xf[at]) : __ushort_as_half(a.xh[at]);
                }
                Tl.x[tr][tq * 8 + j] = v;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, wh, wm::row_major> fa;
            wm::load_matrix_sync(fa, (const wh*) &Tl.x[wm_ * 16][kk], kLdT);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wm::fragment<wm::matrix_b, 16, 16, 16, wh, wm::col_major> fb;
                wm::load_matrix_sync(fb, (const wh*) &Tl.wg[(wn + j) * 16][kk], kLdT);
                wm::mma_sync(fg[j], fa, fb, fg[j]);
                if (TWO) {
                    wm::load_matrix_sync(fb, (const wh*) &Tl.wu[(wn + j) * 16][kk], kLdT);
                    wm::mma_sync(fu[j], fa, fb, fu[j]);
                }
            }
        }
        __syncthreads();
    }
    float* ow = reinterpret_cast<float*>(smem) + (size_t) warp * 16 * 16;   // (the K loop's last barrier: tiles free)
    const int lane = tid & 31;
    for (int j = 0; j < 2; ++j) {
        if (TWO)
            for (int i = 0; i < (int) fg[j].num_elements; ++i)
                fg[j].x[i] = act_out<EPI>(fg[j].x[i], fu[j].x[i], a.limit);
        wm::store_matrix_sync(ow, fg[j], 16, wm::mem_row_major);
        __syncthreads();   // (uniform: every warp stores, then reads its own tile)
        for (int i = lane; i < 16 * 16; i += 32) {
            const int r = wm_ * 16 + i / 16, c = c0 + (wn + j) * 16 + i % 16;
            if (r >= nr) continue;
            const size_t lr = (size_t) (r0 + r);
            if constexpr (!GATE_UP) a.y[lr * a.ldo + c] = ow[i];
            else if constexpr (TWO) a.h[lr * a.ldo + c] = __half_as_ushort(__float2half_rn(ow[i]));
            else if constexpr (EPI == 2) a.graw[lr * a.n_ff + c] = ow[i];
            else a.h[lr * a.ldo + c] = __half_as_ushort(__float2half_rn(swigluoai(a.graw[lr * a.n_ff + c], ow[i], a.limit)));
        }
        __syncthreads();
    }
#endif
}
#endif

// ---------------------------------------------------------------- GEMVs
constexpr int kMaxJobs = 8;
struct Jobs {
    MvJob j[kMaxJobs];
};
template <int T, int NC>
__global__ void __launch_bounds__(kThreads) mv_kernel(const Jobs J) {
    pdl_wait();
    pdl_trigger();
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    load_tab<T>(tab);
    __syncthreads();
    const MvJob& b = J.j[blockIdx.y];
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= b.n) return;
    const uint8_t* w = b.w + (size_t) row * b.ld;
    const int nsb = b.k / 32;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int kb = lane; kb < nsb; kb += 32) {
        Sub s;
        decode<T>(w, kb, tab, s);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < b.ncol) {
                const int4* xp = (const int4*) (b.xq + (size_t) c * b.k + (size_t) kb * 32);
                acc[c] = fmaf(b.xd[(size_t) c * nsb + kb], sub_dot<T>(s, xp[0], xp[1]), acc[c]);
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < b.ncol) {
            const float v = warp_sum(acc[c]);
            if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
        }
}

// ---------------------------------------------------------------- the prompt sets' light experts
// A set's experts of a few rows (<= kLightRows: a prompt of a few hundred tokens spreads ~10 over most of 384 experts)
// in the token kernels' form: each weight sub-block decoded once and dotted with every row's int8 input (rounded per 32
// values as the token path rounds them) - the weights read once at the memory's speed, where a 64-row GEMM tile did 64
// rows' work for them.  Each expert by its own rows: the light kernels take the experts of <= kLightRows rows, the tiles
// the others (SetArgs::skip_le: they skip the light ones) - every expert's sums in one fixed order whatever the set.
// Expert e's row i is input row e kLightRows + i of the scratch, output (local) row bounds[e] - row0 + i.
constexpr int kLightRows = 16;
constexpr int kLightMaxExp = 512;   // a set's experts at most (the scratch: kLightMaxExp kLightRows rows)

// grid (k / 32 / 8, n_exp kLightRows): the light experts' rows of k values (gate / up: F32 x[idx[lr]]; down: F16 h[lr],
// ldx apart) -> int8 q[e kLightRows + i][k], scales d[..][k / 32]
__global__ void __launch_bounds__(kThreads) light_q8_kernel(const float* __restrict__ xf, const uint16_t* __restrict__ xh,
                                                           const int* __restrict__ idx, int ldx, int k,
                                                           const int* __restrict__ bounds, int row0,
                                                           int8_t* __restrict__ q, float* __restrict__ d) {
    pdl_wait();
    pdl_trigger();
    const int e = blockIdx.y / kLightRows, i = blockIdx.y % kLightRows;
    const int rows = bounds[e + 1] - bounds[e];
    const int b = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (i >= rows || rows > kLightRows || b * 32 >= k) return;
    const int lr = bounds[e] - row0 + i;
    const size_t at = (size_t) b * 32 + lane, qr = (size_t) blockIdx.y;
    const float v = xf != nullptr ? xf[(size_t) idx[lr] * ldx + at] : __half2float(__ushort_as_half(xh[(size_t) lr * ldx + at]));
    float am = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;   // (ds4_gguf.cu's q8_quant: the same operations)
    q[qr * k + at] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[qr * (k / 32) + b] = sc;
}

// grid (n_ff / 8 or n_embd / 8, n_exp): a light expert's rows, a warp a row of the weights (gate and up, or down), every
// row of the expert a pass -> h16 (the F16 swiglu) or y.  PS: the persistent grid (as the token kernels'), an item the
// full grid's block (bx, expert)
template <int T, int NE, bool GATE_UP, int EPI = 0, bool PS = false>
__global__ void __launch_bounds__(kThreads) light_kernel(const SetArgs a, const int8_t* __restrict__ xq,
                                                          const float* __restrict__ xd) {
    pdl_wait();
    pdl_trigger();
    constexpr bool TWO = GATE_UP && kBoth<EPI>;   // (gate and up in this pass; EPI 2 / 3: one of them, as w0)
    constexpr bool CT = kCodes<T, PS>;
    __shared__ __align__(16) uint32_t tab[kTabW<T, CT>];
    const int K = GATE_UP ? a.n_embd : a.n_ff, N = GATE_UP ? a.n_ff : a.n_embd, nsb = K / 32;
    const int lane = threadIdx.x & 31;
    const size_t rb = row_bytes_t<T>(K);
    const int nbx = PS ? (N + kWarps - 1) / kWarps : 1;
    const int items = PS ? a.n_exp * nbx : 1;
    bool loaded = false;
    for (int it = PS ? (int) blockIdx.x : 0; it < items; it += PS ? (int) gridDim.x : 1) {
        const int e = PS ? it / nbx : (int) blockIdx.y, bx = PS ? it % nbx : (int) blockIdx.x;
        const int lo = a.bounds[e] - a.row0, ne = a.bounds[e + 1] - a.bounds[e];
        if (ne <= 0 || ne > NE) continue;   // (uniform: the tiles take an expert of more rows)
        if (!loaded) {   // (the codebook once a block: the only barrier, at the first live item - every warp is there)
            load_tab_k<T, CT>(tab);
            __syncthreads();
            loaded = true;
        }
        const int row = bx * kWarps + (threadIdx.x >> 5);
        if (row >= N) continue;   // (a warp's)
        const uint8_t* blob = a.wbase + (size_t) e * a.stride;
        const uint8_t* w0 = blob + a.tr[GATE_UP ? kRole0<EPI> : 2] + (size_t) row * rb;
        const uint8_t* w1 = blob + a.tr[1] + (size_t) row * rb;
        const size_t q0 = (size_t) e * kLightRows;
        float a0[NE], a1[NE];
#pragma unroll
        for (int i = 0; i < NE; ++i) a0[i] = a1[i] = 0.0f;
        for (int kb = lane; kb < nsb; kb += 32) {
            Sub s0, s1;
            decode<T, CT>(w0, kb, tab, s0);
            if (TWO) decode<T, CT>(w1, kb, tab, s1);
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const int4* xp = (const int4*) (xq + (q0 + i) * K + (size_t) kb * 32);
                    const int4 xa = xp[0], xb = xp[1];
                    const float dx = xd[(q0 + i) * nsb + kb];
                    a0[i] = fmaf(dx, sub_dot<T>(s0, xa, xb), a0[i]);
                    if (TWO) a1[i] = fmaf(dx, sub_dot<T>(s1, xa, xb), a1[i]);
                }
        }
#pragma unroll
        for (int i = 0; i < NE; ++i)
            if (i < ne) {
                const float v0 = warp_sum(a0[i]);
                const float v1 = TWO ? warp_sum(a1[i]) : 0.0f;
                if (lane == 0) {
                    const size_t lr = (size_t) (lo + i);
                    if constexpr (!GATE_UP) a.y[lr * a.ldo + row] = v0;
                    else if constexpr (TWO) a.h[lr * a.ldo + row] = __half_as_ushort(__float2half_rn(act_out<EPI>(v0, v1, a.limit)));
                    else if constexpr (EPI == 2) a.graw[lr * a.n_ff + row] = v0;
                    else a.h[lr * a.ldo + row] = __half_as_ushort(__float2half_rn(swigluoai(a.graw[lr * a.n_ff + row], v0, a.limit)));
                }
            }
    }
}

// (tests: decode_values) each sub-block of n rows decoded as the token kernels (decode: the codes times their scales,
// sub_dot's roundings) and as the tiles' loader (deq8) read it
// (CT: IQ1_M's codes through the persistent kernels' code table, the tiles' values through the grid as theirs)
template <int T, bool CT = false>
__global__ void __launch_bounds__(kThreads) values_kernel(const uint8_t* __restrict__ w, int k, int n, float* sub,
                                                           float* tile) {
    pdl_wait();
    pdl_trigger();
    __shared__ __align__(16) uint32_t tab[kTabW<T, false>];
    __shared__ __align__(16) uint32_t ctab[CT ? kTabW<T, true> : 1];
    load_tab<T>(tab);
    if constexpr (CT) load_tab_k<T, true>(ctab);
    __syncthreads();
    const int nsb = k / 32;
    const size_t rb = row_bytes_t<T>(k);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * nsb; i += gridDim.x * blockDim.x) {
        const int r = i / nsb, kb = i % nsb;
        const uint8_t* row = w + (size_t) r * rb;
        float* o = sub + (size_t) r * k + (size_t) kb * 32;
        Sub s;
        decode<T, CT>(row, kb, CT ? ctab : tab, s);
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int at = 4 * t + j, q = (int) (int8_t) (uint8_t) ((uint32_t) s.q[t] >> (8 * j));
                float v = __fmul_rn(s.d, (float) ((at < 16 ? s.s0 : s.s1) * q));
                if constexpr (Ty<T>::MIN) v = __fsub_rn(v, at < 16 ? s.m0 : s.m1);
                o[at] = v;
            }
        float w8[8];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            deq8<T>(row, kb, p, tab, w8);
#pragma unroll
            for (int j = 0; j < 8; ++j) tile[(size_t) r * k + (size_t) kb * 32 + 8 * p + j] = w8[j];
        }
    }
}

// as ds4_gguf.cu's: sm_70 and later (AMD: RDNA 3 / 4), STRATA_DS4_NO_TC=1 keeps the F32 kernel - per device (the GPUs
// of a run need not be of one generation: one before sm_70 beside newer ones keeps its F32 kernel)
bool tc_ok() {
#if defined(STRATA_USE_HIP) && !STRATA_HAS_WMMA
    return false;
#else
    static const bool off = [] {
        const char* v = std::getenv("STRATA_DS4_NO_TC");
        return v != nullptr && v[0] == '1';
    }();
    static std::atomic<int> known[64];   // 0: not asked yet, 1: no, 2: yes
    int dev = 0;
    if (off || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    int k = known[dev].load(std::memory_order_relaxed);
    if (k == 0) {
#if defined(STRATA_USE_HIP)
        cudaDeviceProp p{};
        const bool ok = cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                        (std::strncmp(p.gcnArchName, "gfx11", 5) == 0 || std::strncmp(p.gcnArchName, "gfx12", 5) == 0);
#else
        int major = 0;
        const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 7;
#endif
        k = ok ? 2 : 1;
        known[dev].store(k, std::memory_order_relaxed);
    }
    return k == 2;
#endif
}

// ---------------------------------------------------------------- dispatch
// F(T) for the type t at run time; false when it is not one of these
#define DS4_IQ_SWITCH(t, F)          \
    switch (t) {                     \
        case kIQ2_XXS: F(kIQ2_XXS); break; \
        case kIQ2_XS: F(kIQ2_XS); break;   \
        case kQ5_K: F(kQ5_K); break;       \
        case kIQ2_S: F(kIQ2_S); break;     \
        case kIQ3_XXS: F(kIQ3_XXS); break; \
        case kIQ3_S: F(kIQ3_S); break;     \
        case kIQ4_XS: F(kIQ4_XS); break;   \
        case kQ4_K: F(kQ4_K); break;       \
        case kQ2_K: F(kQ2_K); break;       \
        case kQ3_K: F(kQ3_K); break;       \
        case kIQ1_S: F(kIQ1_S); break;     \
        case kIQ1_M: F(kIQ1_M); break;     \
        case kIQ4_NL: F(kIQ4_NL); break;   \
        default: return false;       \
    }
// ... and the dense matrices' types beside them (the GEMVs, the FP16 rows, decode_values)
#define DS4_IQ_SWITCH_MV(t, F)             \
    switch (t) {                           \
        case kQ6_K: F(kQ6_K); break;       \
        case kQ8_0: F(kQ8_0); break;       \
        default: DS4_IQ_SWITCH(t, F)       \
    }

// IQ1's token / window kernels: RP rows passes a block (1 or 4), AH the loads a step ahead - measured per platform
// (ds4_gguf_test --bench-q), the defaults below; DS4_IQ_RP=1|4 and DS4_IQ_AHEAD=0|1 choose (the same sums either way)
struct Iq1Var {
    int rp;
    int ah;   // -1: the type's default (kAhead)
};
// the early loads by default: IQ1_M on AMD (a Radeon 8065S, 6 experts' gate / up: 200 -> 150 us); IQ1_S there and both
// on the V100 gain nothing from them
#if defined(STRATA_USE_HIP)
template <int T>
constexpr bool kAhead = T == kIQ1_M;
#else
template <int T>
constexpr bool kAhead = false;
#endif
const Iq1Var& iq1_var() {
    static const Iq1Var v = [] {
        Iq1Var r{1, -1};
        if (const char* e = std::getenv("DS4_IQ_RP")) r.rp = std::atoi(e) == 4 ? 4 : 1;
        if (const char* e = std::getenv("DS4_IQ_AHEAD")) r.ah = std::atoi(e) != 0 ? 1 : 0;
        return r;
    }();
    return v;
}

// the persistent grids (kPersist): DS4_IQ_PERSIST (MAYA_DS4_IQ_PERSIST, read as STRATA_DS4_IQ_PERSIST): 0 off, 1 on,
// unset: on for a device of compute capability 8.0 and up (measured about 2% slower on a V100); set_persist over it (tests)
std::atomic<int> g_persist_set{-1};
#if DS4_IQ_PERSIST_BUILD
int persist_env() {   // -1: not set
    static const int v = [] {
        const char* e = std::getenv("STRATA_DS4_IQ_PERSIST");
        if (e == nullptr) e = std::getenv("DS4_IQ_PERSIST");
        return e == nullptr ? -1 : (std::atoi(e) != 0 ? 1 : 0);
    }();
    return v;
}
// the current device's default: compute capability 8.0 and up
bool persist_dev_default() {
    static std::atomic<signed char> known[64];   // 0: not asked yet, 1: off, 2: on
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { (void) cudaGetLastError(); return false; }
    int v = known[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        v = major >= 8 ? 2 : 1;
        known[dev].store((signed char) v, std::memory_order_relaxed);
    }
    return v == 2;
}
#endif
// the token kernels' persistent form: their plain variant (RP 1, no early loads - DS4_IQ_RP=4 / DS4_IQ_AHEAD=1 keep
// the full grid's variants they ask for)
template <int T>
bool persist_plain() {
    if (!persist_on()) return false;   // (kPersist's types only)
    if constexpr (T == kIQ1_S || T == kIQ1_M) {
        const Iq1Var& v = iq1_var();
        return v.rp != 4 && !(v.ah >= 0 ? v.ah != 0 : kAhead<T>);
    }
    return true;
}
#if DS4_IQ_PERSIST_BUILD
// a persistent grid of kernel K for `items` work items: the blocks the current device holds at once (its SMs times
// K's occupancy, asked once a device), at most one an item
template <auto K>
unsigned persist_blocks(long long items) {
    static std::atomic<int> known[64];   // 0: not asked yet, -1: not known (a block an item), else the blocks
    int dev = 0, v = -1;
    if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64) {
        v = known[dev].load(std::memory_order_relaxed);
        if (v == 0) {
            int sms = 0, per = 0;
            if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, K, kThreads, 0) != cudaSuccess) {
                (void) cudaGetLastError();
                sms = per = 0;
            }
            v = sms > 0 && per > 0 ? sms * per : -1;
            known[dev].store(v, std::memory_order_relaxed);
        }
    }
    return (unsigned) std::max<long long>(1, v > 0 ? std::min<long long>(items, v) : items);
}
#endif

// grid.x: the blocks of one pass (n / (kWarps RW))
template <int T, int NE, int EPI = 0>
void launch_gu(const Args& a, dim3 grid, cudaStream_t s) {
#if DS4_IQ_PERSIST_BUILD
    if constexpr (kPersist<T, EPI>)
        if (persist_plain<T>()) {   // (an item a block (x, group) of the grid below)
            constexpr auto K = gate_up_kernel<T, NE, 1, false, kGuRW, EPI, true>;
            pdl_launch(K, persist_blocks<K>((long long) grid.x * grid.y), kThreads, 0, s, a, (int) grid.y);
            return;
        }
#endif
    if constexpr (T == kIQ1_S || T == kIQ1_M) {
        const Iq1Var& v = iq1_var();
        const bool rp = v.rp == 4 && grid.x % 4 == 0, ah = v.ah >= 0 ? v.ah != 0 : kAhead<T>;
        const dim3 g4(grid.x / 4, grid.y, grid.z);
        const int ng = (int) grid.y;
        if (rp && ah) { pdl_launch(gate_up_kernel<T, NE, 4, true, kGuRW, EPI>, g4, kThreads, 0, s, a, ng); return; }
        if (rp) { pdl_launch(gate_up_kernel<T, NE, 4, false, kGuRW, EPI>, g4, kThreads, 0, s, a, ng); return; }
        if (ah) { pdl_launch(gate_up_kernel<T, NE, 1, true, kGuRW, EPI>, grid, kThreads, 0, s, a, ng); return; }
    }
    pdl_launch(gate_up_kernel<T, NE, 1, false, kGuRW, EPI>, grid, kThreads, 0, s, a, (int) grid.y);
}
template <int T, int NE>
void launch_dn(const Args& a, dim3 grid, cudaStream_t s) {
#if DS4_IQ_PERSIST_BUILD
    if constexpr (kPersist<T, 0>)
        if (persist_plain<T>()) {
            constexpr auto K = down_kernel<T, NE, 1, false, kDnRW, true>;
            pdl_launch(K, persist_blocks<K>((long long) grid.x * grid.y), kThreads, 0, s, a, (int) grid.y);
            return;
        }
#endif
    if constexpr (T == kIQ1_S || T == kIQ1_M) {
        const Iq1Var& v = iq1_var();
        const bool rp = v.rp == 4 && grid.x % 4 == 0, ah = v.ah >= 0 ? v.ah != 0 : kAhead<T>;
        const dim3 g4(grid.x / 4, grid.y, grid.z);
        const int ng = (int) grid.y;
        if (rp && ah) { pdl_launch(down_kernel<T, NE, 4, true>, g4, kThreads, 0, s, a, ng); return; }
        if (rp) { pdl_launch(down_kernel<T, NE, 4, false>, g4, kThreads, 0, s, a, ng); return; }
        if (ah) { pdl_launch(down_kernel<T, NE, 1, true>, grid, kThreads, 0, s, a, ng); return; }
    }
    pdl_launch(down_kernel<T, NE>, grid, kThreads, 0, s, a, (int) grid.y);
}
template <int T, int EPI = 0>
void gu_ne(const Args& a, dim3 grid, int ne, cudaStream_t s) {
    if (ne <= 1) launch_gu<T, 1, EPI>(a, grid, s);
    else if (ne <= 4) launch_gu<T, 4, EPI>(a, grid, s);
    else if (ne <= 8) launch_gu<T, 8, EPI>(a, grid, s);
    else launch_gu<T, 16, EPI>(a, grid, s);
}
template <int T>
void dn_ne(const Args& a, dim3 grid, int ne, cudaStream_t s) {
    if (ne <= 1) launch_dn<T, 1>(a, grid, s);
    else if (ne <= 4) launch_dn<T, 4>(a, grid, s);
    else if (ne <= 8) launch_dn<T, 8>(a, grid, s);
    else launch_dn<T, 16>(a, grid, s);
}
template <int T, bool GU, int EPI = 0>
void launch_set(const SetArgs& a, dim3 grid, cudaStream_t s) {
#if !defined(STRATA_USE_HIP) || STRATA_HAS_WMMA
    if (tc_ok()) {
        SetArgs b = a;   // (gate / up: FP16 rows when the caller gave them - set_gate_up's x16)
        b.xh_vec = b.xh != nullptr && ((uintptr_t) b.xh & 15) == 0 && (b.ldx & 7) == 0 ? 1 : 0;
        pdl_launch(set_tc_kernel<T, GU, EPI>, grid, kThreads, 0, s, b);
        return;
    }
#endif
    if (a.max_rows <= 16) {   // (a block of 64 rows would do 64 rows' FMAs for the few there are)
        grid.x = (unsigned) ((a.max_rows + 15) / 16);
        pdl_launch(set_kernel<T, GU, 16, EPI>, grid, kThreads, 0, s, a);
        return;
    }
    pdl_launch(set_kernel<T, GU, BM, EPI>, grid, kThreads, 0, s, a);
}
// the light sets' int8 rows, per device and stream (kLightMaxExp kLightRows rows of k): the default stream is one handle
// on every device (and a destroyed stream's handle may come back on another)
struct LightBuf {
    int8_t* q = nullptr;
    float* d = nullptr;
    size_t cap = 0;
};
std::mutex g_lmu;
std::map<std::pair<int, cudaStream_t>, LightBuf> g_light;
LightBuf* light_buf(cudaStream_t s, int k) {
    std::lock_guard<std::mutex> lk(g_lmu);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;
    LightBuf& b = g_light[{dev, s}];
    const size_t need = (size_t) kLightMaxExp * kLightRows * (size_t) std::max(k, 8192);
    if (b.cap < need) {
        if (b.q != nullptr) cudaFree(b.q);
        if (b.d != nullptr) cudaFree(b.d);
        b = LightBuf{};
        if (cudaMalloc(&b.q, need) != cudaSuccess || cudaMalloc(&b.d, need / 32 * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "ds4 iq: the light sets' scratch (%zu MB) failed\n", need >> 20);
            return nullptr;
        }
        b.cap = need;
    }
    return &b;
}
bool light_on() {   // STRATA_DS4_NO_LIGHT=1: every set through the GEMM tiles
    static const bool on = [] {
        const char* v = std::getenv("STRATA_DS4_NO_LIGHT");
        return !(v != nullptr && v[0] == '1');
    }();
    return on;
}
template <int T, bool GU, int EPI = 0>
void launch_light(const SetArgs& a, const LightBuf& b, cudaStream_t s) {
    const dim3 grid((unsigned) (((GU ? a.n_ff : a.n_embd) + kWarps - 1) / kWarps), (unsigned) a.n_exp);
#if DS4_IQ_PERSIST_BUILD
    if constexpr (kPersist<T, EPI>)
        if (a.max_rows <= 4 && persist_on()) {   // (experts of a few rows: the codebook a block is then the cost)
            constexpr auto K = light_kernel<T, 4, GU, EPI, true>;
            pdl_launch(K, persist_blocks<K>((long long) grid.x * grid.y), kThreads, 0, s, a, b.q, b.d);
            return;
        }
#endif
    if (a.max_rows <= 4) pdl_launch(light_kernel<T, 4, GU, EPI>, grid, kThreads, 0, s, a, b.q, b.d);
    else if (a.max_rows <= 8) pdl_launch(light_kernel<T, 8, GU, EPI>, grid, kThreads, 0, s, a, b.q, b.d);
    else pdl_launch(light_kernel<T, kLightRows, GU, EPI>, grid, kThreads, 0, s, a, b.q, b.d);
}
// a set's gate and up of different types (SetArgs::tu, kActOai only): two passes, the gate's sums in graw between them
bool split_gu(const SetArgs& a) { return a.act == kActOai && a.tu != 0 && a.tu != a.tg; }
// the gate / up's inputs (the up's own type too) and the down's in whole blocks of their types (256 values; IQ4_NL's 32)
bool k_whole(int n_embd, int n_ff, int tg, int tu, int td) {
    return n_embd % block_values(tg) == 0 && (tu == 0 || n_embd % block_values(tu) == 0) && n_ff % block_values(td) == 0;
}
bool light_set(const SetArgs& a, bool gu, cudaStream_t s) {
    const int K = gu ? a.n_embd : a.n_ff;
    if (gu && a.act != kActSwiglu && (a.tg == kMXFP4 || (split_gu(a) && a.graw == nullptr))) return false;
    LightBuf* b = light_buf(s, K);
    if (b == nullptr) return false;
    const dim3 gq((unsigned) ((K / 32 + kWarps - 1) / kWarps), (unsigned) (a.n_exp * kLightRows));
    pdl_launch(light_q8_kernel, gq, kThreads, 0, s, gu ? a.xf : nullptr, gu ? nullptr : a.xh, a.idx, a.ldx, K, a.bounds,
               a.row0, b->q, b->d);
    if (gu && a.act != kActSwiglu) {   // (MiniMax's: one pass, or the gate's then the up's on the same int8 rows)
        if (!split_gu(a)) {
#define LO_(T) launch_light<T, true, 1>(a, *b, s)
            DS4_IQ_SWITCH(a.tg, LO_)
#undef LO_
            return true;
        }
#define LO2_(T) launch_light<T, true, 2>(a, *b, s)
        DS4_IQ_SWITCH(a.tg, LO2_)
#undef LO2_
#define LO3_(T) launch_light<T, true, 3>(a, *b, s)
        DS4_IQ_SWITCH(a.tu, LO3_)
#undef LO3_
        return true;
    }
    if (gu) {
        if (a.tg == kMXFP4) {
            launch_light<kMXFP4, true>(a, *b, s);
            return true;
        }
#define LG_(T) launch_light<T, true>(a, *b, s)
        DS4_IQ_SWITCH(a.tg, LG_)
#undef LG_
    } else {
        if (a.td == kMXFP4) {
            launch_light<kMXFP4, false>(a, *b, s);
            return true;
        }
#define LD_(T) launch_light<T, false>(a, *b, s)
        DS4_IQ_SWITCH(a.td, LD_)
#undef LD_
    }
    return true;
}

template <int T>
void mv_nc(const Jobs& J, int nj, int max_n, int max_c, cudaStream_t s) {
    const dim3 grid((unsigned) ((max_n + kWarps - 1) / kWarps), (unsigned) nj);
    if (max_c <= 1) pdl_launch(mv_kernel<T, 1>, grid, kThreads, 0, s, J);
    else if (max_c <= 4) pdl_launch(mv_kernel<T, 4>, grid, kThreads, 0, s, J);
    else pdl_launch(mv_kernel<T, 16>, grid, kThreads, 0, s, J);
}

}  // namespace

bool supported(int t) { return t != kQ6_K && t != kQ8_0 && block_bytes(t) > 0; }
bool mv_supported(int t) { return block_bytes(t) > 0; }

int block_bytes(int t) {
    switch (t) {
        case kQ6_K: return Ty<kQ6_K>::B;
        case kQ8_0: return Ty<kQ8_0>::B;
        case kIQ2_XXS: return Ty<kIQ2_XXS>::B;
        case kIQ2_XS: return Ty<kIQ2_XS>::B;
        case kQ5_K: return Ty<kQ5_K>::B;
        case kIQ2_S: return Ty<kIQ2_S>::B;
        case kIQ3_XXS: return Ty<kIQ3_XXS>::B;
        case kIQ3_S: return Ty<kIQ3_S>::B;
        case kIQ4_XS: return Ty<kIQ4_XS>::B;
        case kQ4_K: return Ty<kQ4_K>::B;
        case kQ2_K: return Ty<kQ2_K>::B;
        case kQ3_K: return Ty<kQ3_K>::B;
        case kIQ1_S: return Ty<kIQ1_S>::B;
        case kIQ1_M: return Ty<kIQ1_M>::B;
        case kIQ4_NL: return Ty<kIQ4_NL>::B;
        default: return 0;
    }
}

const char* name(int t) {
    switch (t) {
        case kQ6_K: return "Q6_K";
        case kQ8_0: return "Q8_0";
        case kIQ2_XXS: return "IQ2_XXS";
        case kIQ2_XS: return "IQ2_XS";
        case kQ5_K: return "Q5_K";
        case kIQ2_S: return "IQ2_S";
        case kIQ3_XXS: return "IQ3_XXS";
        case kIQ3_S: return "IQ3_S";
        case kIQ4_XS: return "IQ4_XS";
        case kQ4_K: return "Q4_K";
        case kQ2_K: return "Q2_K";
        case kQ3_K: return "Q3_K";
        case kIQ1_S: return "IQ1_S";
        case kIQ1_M: return "IQ1_M";
        case kIQ4_NL: return "IQ4_NL";
        default: return "?";
    }
}

bool gate_up(const Args& a, int groups, int ne_max, cudaStream_t s) {
    if (!k_whole(a.n_embd, 0, a.tg, a.tu, 0) || a.n_ff % (kWarps * kGuRW) != 0) return false;
    const dim3 grid((unsigned) (a.n_ff / (kWarps * kGuRW)), (unsigned) groups);
    if (a.act != kActSwiglu) {   // MiniMax's swigluoai: one pass, or (gate and up of different types) two
        if (a.tu == 0 || a.tu == a.tg) {
#define GO_(T) gu_ne<T, 1>(a, grid, ne_max, s)
            DS4_IQ_SWITCH(a.tg, GO_)
#undef GO_
            return true;
        }
#define GO2_(T) gu_ne<T, 2>(a, grid, ne_max, s)
        DS4_IQ_SWITCH(a.tg, GO2_)
#undef GO2_
#define GO3_(T) gu_ne<T, 3>(a, grid, ne_max, s)
        DS4_IQ_SWITCH(a.tu, GO3_)
#undef GO3_
        return true;
    }
#define GU_(T) gu_ne<T>(a, grid, ne_max, s)
    DS4_IQ_SWITCH(a.tg, GU_)
#undef GU_
    return true;
}

bool down(const Args& a, int groups, int ne_max, cudaStream_t s) {
    if (a.n_ff % block_values(a.td) != 0 || a.n_embd % (kWarps * kDnRW) != 0) return false;
    const dim3 grid((unsigned) (a.n_embd / (kWarps * kDnRW)), (unsigned) groups);
#define DN_(T) dn_ne<T>(a, grid, ne_max, s)
    DS4_IQ_SWITCH(a.td, DN_)
#undef DN_
    return true;
}

bool light_gate_up(const SetArgs& a, cudaStream_t s) {   // (both roles' blocks whole, as for the down)
    if (a.n_exp > kLightMaxExp || !light_on() || !k_whole(a.n_embd, a.n_ff, a.tg, a.tu, a.td)) return false;
    return light_set(a, true, s);
}
bool light_down(const SetArgs& a, cudaStream_t s) {
    if (a.n_exp > kLightMaxExp || !light_on() || !k_whole(a.n_embd, a.n_ff, a.tg, a.tu, a.td)) return false;
    return light_set(a, false, s);
}

bool persist_on() {
#if !DS4_IQ_PERSIST_BUILD
    return false;
#else
    const int t = g_persist_set.load(std::memory_order_relaxed);
    if (t >= 0) return t != 0;
    const int e = persist_env();
    return e >= 0 ? e != 0 : persist_dev_default();
#endif
}
void set_persist(int on) { g_persist_set.store(on < 0 ? -1 : on != 0 ? 1 : 0, std::memory_order_relaxed); }

#if defined(STRATA_USE_HIP) && !defined(MAYA_DS4_IQ_ONLY)
// Prompt expert sets through the existing GLM MMQ path (HIP, opt-in).
// The same q8_1/MMA path the GLM prompt uses (strata::prefill::mmq): the packed expert is read once, activations are
// rounded to q8_1, and the products run on int8 tensor cores.  Opt-in (MAYA_DS4_EXPERT_MMQ=1), sets of at least 32 rows;
// smaller sets, unsupported shapes/types and allocation failure keep the FP32 tiled kernel above.
namespace {
thread_local int mmq_rows = 0;
thread_local const int* mmq_host_bounds = nullptr;
}
MmqRowsScope::MmqRowsScope(int rows, const int* bounds)
    : previous_rows_(mmq_rows), previous_bounds_(mmq_host_bounds) {
    mmq_rows = rows;
    mmq_host_bounds = bounds;
}
MmqRowsScope::~MmqRowsScope() {
    mmq_rows = previous_rows_;
    mmq_host_bounds = previous_bounds_;
}
namespace {

constexpr int kMmqCap = 32768;  // maximum compact rows in one prompt set
bool mmq_on() {
    static const bool enabled = [] {
        const char* e = std::getenv("STRATA_DS4_EXPERT_MMQ");
        return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    return enabled;
}

bool mmq_wanted(const SetArgs& a, bool gate_up) {
    if (!mmq_on() || a.max_rows < 32 || a.n_exp <= 0 || a.n_exp > 512 || !mmq_host_bounds || mmq_rows <= 0) return false;
    if (mmq_host_bounds[0] != a.row0 || mmq_host_bounds[a.n_exp] - a.row0 != mmq_rows) return false;
    for (int i = 0; i < a.n_exp; ++i) {
        const int rows = mmq_host_bounds[i + 1] - mmq_host_bounds[i];
        if (rows < 0 || rows > a.max_rows) return false;
    }
    if (a.n_embd != 5120 || a.n_ff != 2304) return false;   // the DS4 pack's geometry
    if (a.ldx < (gate_up ? a.n_embd : a.n_ff) || a.ldo < (gate_up ? a.n_ff : a.n_embd)) return false;
    if (gate_up && a.ldx % 4 != 0) return false;   // the MMQ quantizer loads float4s
    if (gate_up ? (a.xf == nullptr || a.h == nullptr) : (a.xh == nullptr || a.y == nullptr)) return false;
    const int type = gate_up ? a.tg : a.td;
    if (type != kIQ2_S && type != kIQ2_XXS && type != kIQ3_S && type != kIQ3_XXS) return false;
    if (!strata::prefill::mmq::built() || !strata::prefill::mmq::supported(gate_up ? a.tg : a.td)) return false;
    return true;
}

void mmq_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "DeepSeek expert MMQ %s failed: %s\n", operation, cudaGetErrorString(error));
        std::abort();
    }
}

bool mmq_resource_ok(cudaError_t error) {
    if (error == cudaSuccess) return true;
    if (error != cudaErrorMemoryAllocation) mmq_check(error, "scratch setup");
    // Only an allocation shortage is a supported fallback. Do not hide an asynchronous execution error.
    const cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess && last != cudaErrorMemoryAllocation) mmq_check(last, "scratch allocation");
    return false;
}

bool mmq_alloc(void** ptr, size_t bytes) {
    return mmq_resource_ok(cudaMalloc(ptr, bytes));
}

struct MmqScratch {
    int device;
    cudaEvent_t ready = nullptr;
    bool recorded = false;
    strata::prefill::mmq::Context* ctx = nullptr;
    explicit MmqScratch(int dev) : device(dev) {}
    ~MmqScratch() {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return;
        release();
        delete ctx;
        if (ready) (void) cudaEventDestroy(ready);
        if (previous != device) (void) cudaSetDevice(previous);
    }

    void* xq = nullptr;
    float* gu = nullptr;
    float* hf = nullptr;
    int* b = nullptr;
    int* ids = nullptr;
    int rows = 0;

    bool ensure(int nrows, int ne) {
        if (nrows <= 0 || nrows > kMmqCap || ne <= 0 || ne > 512) return false;
        if (!ready) {
            cudaEvent_t created = nullptr;
            if (!mmq_resource_ok(cudaEventCreateWithFlags(&created, cudaEventDisableTiming))) return false;
            ready = created;
        }
        if (rows >= nrows && b != nullptr) return true;   // grow only
        release();
        const size_t q8 = strata::prefill::mmq::q8_bytes(nrows, 5120);
        if (!mmq_alloc(&xq, q8) ||
            !mmq_alloc((void**) &gu, (size_t) nrows * 2 * 2304 * sizeof(float)) ||
            !mmq_alloc((void**) &hf, (size_t) nrows * 2304 * sizeof(float)) ||
            !mmq_alloc((void**) &b, (size_t) 513 * sizeof(int)) ||
            !mmq_alloc((void**) &ids, (size_t) (nrows + 256) * sizeof(int))) {
            release();
            return false;
        }
        if (!ctx) ctx = new strata::prefill::mmq::Context();
        rows = nrows;
        return true;
    }
    void release() {
        if (xq) (void) cudaFree(xq);
        if (gu) (void) cudaFree(gu);
        if (hf) (void) cudaFree(hf);
        if (b) (void) cudaFree(b);
        if (ids) (void) cudaFree(ids);
        xq = nullptr; gu = nullptr; hf = nullptr; b = nullptr; ids = nullptr; rows = 0;
    }
};

// A thread owns its contexts; each device/stream owns separate buffers and a separate GGML fixup pool.
// A completion event also orders reuse if the runtime recycles a destroyed stream's handle.
MmqScratch& mmq_scratch(int device, cudaStream_t stream) {
    using Key = std::pair<int, uintptr_t>;
    thread_local std::map<Key, std::unique_ptr<MmqScratch>> states;
    auto& state = states[{device, reinterpret_cast<uintptr_t>(stream)}];
    if (!state) state = std::make_unique<MmqScratch>(device);
    return *state;
}

__global__ void mmq_shift_bounds(const int* src, int* dst, int n, int row0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) dst[i] = src[i] - row0;
}

__global__ void mmq_iota(int* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = i;
}

__global__ void mmq_half_float(const uint16_t* __restrict__ h, float* __restrict__ f, int rows, int cols, int ld) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int r = i / cols, c = i - r * cols;
    f[i] = __half2float(__ushort_as_half(h[(size_t) r * ld + c]));
}

// gate/up rows [r][2 nff] -> FP16 SwiGLU at h[r*ldo + c], the same clamp and rounding order as set_kernel.
__global__ void mmq_epilogue(const float* __restrict__ g, uint16_t* __restrict__ h, int rows, int nff, int ldo,
                             float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nff) return;
    const int r = i / nff, c = i - r * nff;
    const float gv = fminf(g[(size_t) r * 2 * nff + c], limit);
    const float uv = fminf(fmaxf(g[(size_t) r * 2 * nff + nff + c], -limit), limit);
    const float v = gv / (1.0f + __expf(-gv)) * uv;
    h[(size_t) r * ldo + c] = __half_as_ushort(__float2half_rn(v));
}

bool mmq_gate_up(const SetArgs& a, cudaStream_t s) {
    const int total = mmq_rows;
    if (total <= 0 || total > kMmqCap) return false;
    int device = 0;
    cudaStreamCaptureStatus capture;
    mmq_check(cudaGetDevice(&device), "device query");
    mmq_check(cudaStreamIsCapturing(s, &capture), "capture query");
    if (capture != cudaStreamCaptureStatusNone) return false;
    MmqScratch& sc = mmq_scratch(device, s);
    if (!sc.ensure(total, a.n_exp)) return false;
    if (sc.recorded) mmq_check(cudaStreamWaitEvent(s, sc.ready, 0), "scratch wait");
    mmq_shift_bounds<<<(a.n_exp + 256) / 256, 256, 0, s>>>(a.bounds, sc.b, a.n_exp, a.row0);
    mmq_check(cudaGetLastError(), "launch");
    mmq_iota<<<(sc.rows + 256 + 255) / 256, 256, 0, s>>>(sc.ids, sc.rows + 256);
    mmq_check(cudaGetLastError(), "launch");
    strata::prefill::mmq::quantize(a.xf, a.idx, sc.xq, a.tg, a.n_embd, a.ldx, total, (void*) s);
    strata::prefill::mmq::Product p{};
    p.w = (const uint8_t*) a.wbase + a.tr[0];
    p.type = a.tg;
    p.w_rows = a.n_ff;
    p.w_cols = a.n_embd;
    p.expert_bytes = a.stride;
    p.n = a.n_exp;
    p.xq = sc.xq;
    p.bounds = sc.b;
    p.ids = sc.ids;
    p.total_rows = total;
    p.max_rows = a.max_rows;
    p.dst = sc.gu;
    p.ld_dst = 2 * a.n_ff;
    auto* ctx = sc.ctx;
    const int* hb = mmq_host_bounds;
    for (int e = 0; e < a.n_exp; ++e) {
        const int cnt = hb[e + 1] - hb[e];
        if (cnt <= 0) continue;
        p.n = 1;
        p.bounds = sc.b + e;
        p.max_rows = a.tg == kIQ2_S ? std::max(cnt, 32) : cnt;
        p.expert_bytes = 0;
        p.w = (const uint8_t*) a.wbase + a.tr[0] + (size_t) e * a.stride;
        p.dst = sc.gu;
        ctx->run(p, (void*) s);
        p.w = (const uint8_t*) a.wbase + a.tr[1] + (size_t) e * a.stride;
        p.dst = sc.gu + a.n_ff;
        ctx->run(p, (void*) s);
    }
    mmq_epilogue<<<(total * a.n_ff + 255) / 256, 256, 0, s>>>(sc.gu, a.h, total, a.n_ff, a.ldo, a.limit);
    mmq_check(cudaGetLastError(), "launch");
    mmq_check(cudaEventRecord(sc.ready, s), "scratch completion");
    sc.recorded = true;
    return true;
}

bool mmq_down(const SetArgs& a, cudaStream_t s) {
    const int total = mmq_rows;
    if (total <= 0 || total > kMmqCap) return false;
    int device = 0;
    cudaStreamCaptureStatus capture;
    mmq_check(cudaGetDevice(&device), "device query");
    mmq_check(cudaStreamIsCapturing(s, &capture), "capture query");
    if (capture != cudaStreamCaptureStatusNone) return false;
    MmqScratch& sc = mmq_scratch(device, s);
    if (!sc.ensure(total, a.n_exp)) return false;
    if (sc.recorded) mmq_check(cudaStreamWaitEvent(s, sc.ready, 0), "scratch wait");
    mmq_shift_bounds<<<(a.n_exp + 256) / 256, 256, 0, s>>>(a.bounds, sc.b, a.n_exp, a.row0);
    mmq_check(cudaGetLastError(), "launch");
    mmq_iota<<<(sc.rows + 256 + 255) / 256, 256, 0, s>>>(sc.ids, sc.rows + 256);
    mmq_check(cudaGetLastError(), "launch");
    mmq_half_float<<<(total * a.n_ff + 255) / 256, 256, 0, s>>>(a.xh, sc.hf, total, a.n_ff, a.ldx);
    mmq_check(cudaGetLastError(), "launch");
    strata::prefill::mmq::quantize(sc.hf, nullptr, sc.xq, a.td, a.n_ff, a.n_ff, total, (void*) s);
    strata::prefill::mmq::Product p{};
    p.w = (const uint8_t*) a.wbase + a.tr[2];
    p.type = a.td;
    p.w_rows = a.n_embd;
    p.w_cols = a.n_ff;
    p.expert_bytes = a.stride;
    p.n = a.n_exp;
    p.xq = sc.xq;
    p.bounds = sc.b;
    p.ids = sc.ids;
    p.total_rows = total;
    p.max_rows = a.max_rows;
    p.dst = a.y;
    p.ld_dst = a.ldo;
    auto* ctx = sc.ctx;
    const int* hb = mmq_host_bounds;
    for (int e = 0; e < a.n_exp; ++e) {
        const int cnt = hb[e + 1] - hb[e];
        if (cnt <= 0) continue;
        p.n = 1;
        p.bounds = sc.b + e;
        p.max_rows = cnt;
        p.expert_bytes = 0;
        p.w = (const uint8_t*) a.wbase + a.tr[2] + (size_t) e * a.stride;
        p.dst = a.y;
        ctx->run(p, (void*) s);
    }
    mmq_check(cudaEventRecord(sc.ready, s), "scratch completion");
    sc.recorded = true;
    return true;
}

}  // namespace

#endif

#if defined(STRATA_USE_HIP) && !defined(MAYA_DS4_IQ_ONLY)
// Prompt expert sets through the fused int8 experts the GLM prompt uses (strata::prefill::rdna4, gfx1151 / gfx1100 /
// RDNA4): the weights decoded to int8 in the WMMA tiles' shared memory, the activations and the SwiGLU's H rounded to
// int8 per 32 values (as GLM's), the products on the int8 matrix cores.  On by default where it covers the set's formats
// (STRATA_DS4_PREFILL_FUSED=0: the FP16 tiles above).  The set's rows are gathered into the scratch of the caller's
// FusedScratchScope (its down output: unused until the down writes it), H goes into the caller's FP16 rows at their
// stride (2 n_ff bytes: an int8 row is act_bytes(1, n_ff)), and the down reads it there - both passes decide alike.
namespace {
thread_local void* fused_scratch = nullptr;
thread_local size_t fused_scratch_bytes = 0;
}
FusedScratchScope::FusedScratchScope(void* scratch, size_t bytes)
    : previous_(fused_scratch), previous_bytes_(fused_scratch_bytes) {
    fused_scratch = scratch;
    fused_scratch_bytes = bytes;
}
FusedScratchScope::~FusedScratchScope() {
    fused_scratch = previous_;
    fused_scratch_bytes = previous_bytes_;
}
namespace {
// STRATA_DS4_PREFILL_FUSED: unset = on for gfx1151 (Strix / Gorgon Halo, where it is measured), 1 = on wherever the
// fused experts run (gfx1100, RDNA4: not measured), 0 = off
bool fused_on() {
    static const int mode = [] {
        const char* v = std::getenv("STRATA_DS4_PREFILL_FUSED");
        return v == nullptr ? -1 : std::atoi(v) != 0 ? 1 : 0;
    }();
    if (mode >= 0) return mode == 1;
    static std::mutex mu;
    static std::map<int, bool> halo;   // (a device's answer, asked once)
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    std::lock_guard<std::mutex> lk(mu);
    const auto it = halo.find(dev);
    if (it != halo.end()) return it->second;
    cudaDeviceProp prop{};
    const bool on = cudaGetDeviceProperties(&prop, dev) == cudaSuccess && std::strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
    halo[dev] = on;
    return on;
}
// (the same decision for the set's two passes: none of what it reads differs between them)
bool fused_wanted(const SetArgs& a) {
    if (!fused_on() || fused_scratch == nullptr || a.n_exp <= 0 || a.n_exp > 512 || !mmq_host_bounds || mmq_rows <= 0)
        return false;
    if (mmq_host_bounds[0] != a.row0 || mmq_host_bounds[a.n_exp] - a.row0 != mmq_rows) return false;
    if (a.n_embd != 5120 || a.n_ff != 2304 || a.act != kActSwiglu || (a.tu != 0 && a.tu != a.tg) || a.skip_ge > 0)
        return false;
    if (fused_scratch_bytes < strata::prefill::rdna4::act_bytes(mmq_rows, a.n_embd)) return false;
    return strata::prefill::rdna4::supported(a.tg, a.td);
}
strata::prefill::rdna4::Geometry fused_geo(const SetArgs& a, size_t h_ld) {
    strata::prefill::rdna4::Geometry g;
    g.gu_type = a.tg;
    g.d_type = a.td;
    g.gu_row = (size_t) (a.n_embd / 256) * (size_t) block_bytes(a.tg);
    g.d_row = (size_t) (a.n_ff / 256) * (size_t) block_bytes(a.td);
    g.gate_off = a.tr[0];
    g.up_off = a.tr[1];
    g.down_off = a.tr[2];
    g.limit = a.limit;
    g.n_embd = a.n_embd;
    g.n_ff = a.n_ff;
    g.h_ld = h_ld;
    return g;
}
// the tiles' list (two ints a tile), a device's own: at most the rows' tiles and one partial tile an expert
void* fused_tiles(int rows, int n_exp) {
    static std::mutex mu;
    static std::map<int, std::pair<void*, size_t>> per_dev;
    const size_t need = ((size_t) rows / 64 + (size_t) n_exp + 8) * 2 * sizeof(int);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;
    std::lock_guard<std::mutex> lk(mu);
    auto& t = per_dev[dev];
    if (t.second < need) {
        if (t.first != nullptr) {
            cudaDeviceSynchronize();
            cudaFree(t.first);
        }
        t = {nullptr, 0};
        if (cudaMalloc(&t.first, need) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        t.second = need;
    }
    return t.first;
}
bool fused_gate_up(const SetArgs& a, cudaStream_t s) {
    if (a.xf == nullptr || a.idx == nullptr || a.h == nullptr) return false;
    void* tiles = fused_tiles(mmq_rows, a.n_exp);
    if (tiles == nullptr) return false;
    strata::prefill::rdna4::quantize_rows(a.xf, a.ldx, a.idx, mmq_rows, a.n_embd, fused_scratch, s);
    strata::prefill::rdna4::experts_gate_up(a.wbase, a.stride, fused_geo(a, (size_t) a.ldo * 2), a.n_exp, a.bounds,
                                            mmq_host_bounds, tiles, fused_scratch, nullptr, a.h, s);
    return true;
}
bool fused_down(const SetArgs& a, cudaStream_t s) {
    if (a.xh == nullptr || a.y == nullptr || a.ldo != a.n_embd) return false;
    void* tiles = fused_tiles(mmq_rows, a.n_exp);
    if (tiles == nullptr) return false;
    strata::prefill::rdna4::experts_down(a.wbase, a.stride, fused_geo(a, (size_t) a.ldx * 2), a.n_exp, a.bounds,
                                         mmq_host_bounds, tiles, a.xh, a.y, s);
    return true;
}
}  // namespace
#endif

bool set_gate_up(const SetArgs& a, cudaStream_t s) {
#if defined(STRATA_USE_HIP) && !defined(MAYA_DS4_IQ_ONLY)
    if (fused_wanted(a)) {
        if (fused_gate_up(a, s)) return true;
        std::fprintf(stderr, "ds4: the fused prompt experts could not run a set\n");
        std::abort();   // (its down would read H the FP16 tiles never wrote)
    }
    if (mmq_wanted(a, true) && mmq_gate_up(a, s)) return true;
#endif
    if (!k_whole(a.n_embd, 0, a.tg, a.tu, 0) || a.n_ff % BN != 0) return false;
    if (split_gu(a) && a.graw == nullptr) return false;
    SetArgs t = a;
    if (light_gate_up(a, s)) {   // the experts of a few rows; the tiles the others
        if (a.max_rows <= kLightRows) return true;
        t.skip_le = kLightRows;
    }
    const dim3 grid((unsigned) ((a.max_rows + BM - 1) / BM), (unsigned) (a.n_ff / BN), (unsigned) a.n_exp);
    if (a.act != kActSwiglu) {   // MiniMax's swigluoai: one pass, or the gate's then the up's (graw between them)
        if (!split_gu(a)) {
#define SO_(T) launch_set<T, true, 1>(t, grid, s)
            DS4_IQ_SWITCH(a.tg, SO_)
#undef SO_
            return true;
        }
#define SO2_(T) launch_set<T, true, 2>(t, grid, s)
        DS4_IQ_SWITCH(a.tg, SO2_)
#undef SO2_
#define SO3_(T) launch_set<T, true, 3>(t, grid, s)
        DS4_IQ_SWITCH(a.tu, SO3_)
#undef SO3_
        return true;
    }
#define SG_(T) launch_set<T, true>(t, grid, s)
    DS4_IQ_SWITCH(a.tg, SG_)
#undef SG_
    return true;
}

bool set_down(const SetArgs& a, cudaStream_t s) {
#if defined(STRATA_USE_HIP) && !defined(MAYA_DS4_IQ_ONLY)
    if (fused_wanted(a)) {
        if (fused_down(a, s)) return true;
        std::fprintf(stderr, "ds4: the fused prompt experts could not run a set's down\n");
        std::abort();   // (H is int8 here: the FP16 tiles cannot read it)
    }
    if (mmq_wanted(a, false) && mmq_down(a, s)) return true;
#endif
    if (a.n_ff % block_values(a.td) != 0 || a.n_embd % BN != 0) return false;
    SetArgs t = a;
    if (light_down(a, s)) {
        if (a.max_rows <= kLightRows) return true;
        t.skip_le = kLightRows;
    }
    const dim3 grid((unsigned) ((a.max_rows + BM - 1) / BM), (unsigned) (a.n_embd / BN), (unsigned) a.n_exp);
#define SD_(T) launch_set<T, false>(t, grid, s)
    DS4_IQ_SWITCH(a.td, SD_)
#undef SD_
    return true;
}

bool mv(const MvJob* jobs, int n, cudaStream_t s) {
    // the jobs by type, up to kMaxJobs a launch
    bool done[64] = {};
    if (n > 64) return false;
    for (int i = 0; i < n; ++i)
        while (!done[i]) {
            const int t = jobs[i].type;
            Jobs J{};
            int nj = 0, max_n = 0, max_c = 1;
            for (int k = i; k < n && nj < kMaxJobs; ++k) {
                if (done[k] || jobs[k].type != t) continue;
                if (jobs[k].k % block_values(t) != 0 || jobs[k].ncol < 1 || jobs[k].ncol > 16) return false;
                J.j[nj++] = jobs[k];
                max_n = std::max(max_n, jobs[k].n);
                max_c = std::max(max_c, jobs[k].ncol);
                done[k] = true;
            }
#define MV_(T) mv_nc<T>(J, nj, max_n, max_c, s)
            DS4_IQ_SWITCH_MV(t, MV_)
#undef MV_
        }
    return true;
}

// ---------------------------------------------------------------- the busy experts' FP16 weights (f16_rows)
// 8 values a thread: row r's sub-block kb, piece p - deq8's values rounded as the tensor-core tiles round them
template <int T>
__global__ void __launch_bounds__(kThreads) f16_rows_kernel(const uint8_t* __restrict__ w, int k, int n,
                                                             uint16_t* __restrict__ dst, int ldd) {
    pdl_wait();
    pdl_trigger();
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    load_tab<T>(tab);
    __syncthreads();
    const int nq = k / 8;
    const size_t rb = row_bytes_t<T>(k);
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < (long long) n * nq;
         i += (long long) gridDim.x * blockDim.x) {
        const int r = (int) (i / nq), q = (int) (i % nq);
        float w8[8];
        deq8<T>(w + (size_t) r * rb, q >> 2, q & 3, tab, w8);
        union {
            uint4 v;
            uint16_t h[8];
        } u;
#pragma unroll
        for (int j = 0; j < 8; ++j) u.h[j] = __half_as_ushort(__float2half_rn(w8[j]));
        *reinterpret_cast<uint4*>(dst + (size_t) r * ldd + (size_t) q * 8) = u.v;
    }
}
__global__ void __launch_bounds__(kThreads) gather_f16_kernel(const uint16_t* __restrict__ x, int ldx,
                                                               const int* __restrict__ idx, int n, int k,
                                                               uint16_t* __restrict__ dst) {
    pdl_wait();
    pdl_trigger();
    const int nq = k / 8;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < (long long) n * nq;
         i += (long long) gridDim.x * blockDim.x) {
        const int r = (int) (i / nq), q = (int) (i % nq);
        *reinterpret_cast<uint4*>(dst + (size_t) r * k + (size_t) q * 8) =
            *reinterpret_cast<const uint4*>(x + (size_t) idx[r] * ldx + (size_t) q * 8);
    }
}
__global__ void __launch_bounds__(kThreads) swiglu_f16_kernel(const float* __restrict__ gu, int ld, int n, int ff,
                                                               float limit, uint16_t* __restrict__ h, int ldh) {
    pdl_wait();
    pdl_trigger();
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < (long long) n * ff;
         i += (long long) gridDim.x * blockDim.x) {
        const int r = (int) (i / ff), c = (int) (i % ff);
        const float* g = gu + (size_t) r * ld;
        h[(size_t) r * ldh + c] = __half_as_ushort(__float2half_rn(swiglu(g[c], g[ff + c], limit)));
    }
}
static unsigned grid_for(long long items) {
    return (unsigned) std::max<long long>(1, std::min<long long>(4096, (items + kThreads - 1) / kThreads));
}
bool f16_rows(int type, const uint8_t* w, int k, int n, uint16_t* dst, int ldd, cudaStream_t s) {
    if (k % block_values(type) != 0 || k % 32 != 0 || n < 1 || ldd % 8 != 0 || ((uintptr_t) dst & 15) != 0)
        return false;
    const unsigned blocks = grid_for((long long) n * (k / 8));
#define FR_(T) pdl_launch(f16_rows_kernel<T>, blocks, kThreads, 0, s, w, k, n, dst, ldd)
    DS4_IQ_SWITCH_MV(type, FR_)
#undef FR_
    return cudaGetLastError() == cudaSuccess;
}
void gather_f16(const uint16_t* x, int ldx, const int* idx, int n, int k, uint16_t* dst, cudaStream_t s) {
    pdl_launch(gather_f16_kernel, grid_for((long long) n * (k / 8)), kThreads, 0, s, x, ldx, idx, n, k, dst);
}
void swiglu_f16(const float* gu, int ld, int n, int ff, float limit, uint16_t* h, int ldh, cudaStream_t s) {
    pdl_launch(swiglu_f16_kernel, grid_for((long long) n * ff), kThreads, 0, s, gu, ld, n, ff, limit, h, ldh);
}

bool decode_values(int type, const uint8_t* w, int k, int n, float* sub, float* tile, cudaStream_t s) {
    if (k % block_values(type) != 0 || k % 32 != 0 || n < 1) return false;
    const unsigned blocks = (unsigned) std::min<int>(1024, (n * (k / 32) + kThreads - 1) / kThreads);
#if DS4_IQ_PERSIST_BUILD
    if (type == kIQ1_M && persist_on()) {   // (the persistent kernels' decode: the code table)
        pdl_launch(values_kernel<kIQ1_M, true>, blocks, kThreads, 0, s, w, k, n, sub, tile);
        return true;
    }
#endif
#define DV_(T) pdl_launch(values_kernel<T>, blocks, kThreads, 0, s, w, k, n, sub, tile)
    DS4_IQ_SWITCH_MV(type, DV_)
#undef DV_
    return true;
}

}  // namespace strata::ds4::gguf::iq
