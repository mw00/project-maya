// src/ds4/ds4_iq.cu - a GGUF's low-bit routed experts on the device (see ds4_iq.cuh).  The block layouts and grids are
// ggml's (ggml-common.h, MIT, third_party/ggml/LICENSE); the sign and grid tricks are llama.cpp's vecdotq.cuh.
#include "ds4_iq.cuh"
#if defined(STRATA_USE_HIP)
#include "strata/prefill/moe_mmq.hpp"
#include <cstring>
#include <map>
#include <memory>
#endif

#include <cuda_fp16.h>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#endif

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace strata::ds4::gguf::iq {
namespace {

constexpr int kWarps = 8, kThreads = kWarps * 32;

// ---------------------------------------------------------------- the types
template <int T>
struct Ty;
template <> struct Ty<kIQ2_XXS> { static constexpr int B = 66, TAB = 512, MIN = 0; };
template <> struct Ty<kIQ2_S> { static constexpr int B = 82, TAB = 2048, MIN = 0; };
template <> struct Ty<kIQ3_XXS> { static constexpr int B = 98, TAB = 256, MIN = 0; };
template <> struct Ty<kIQ3_S> { static constexpr int B = 110, TAB = 512, MIN = 0; };
template <> struct Ty<kIQ4_XS> { static constexpr int B = 136, TAB = 4, MIN = 0; };
template <> struct Ty<kQ4_K> { static constexpr int B = 144, TAB = 0, MIN = 1; };
template <> struct Ty<kQ2_K> { static constexpr int B = 84, TAB = 0, MIN = 1; };
template <> struct Ty<kQ3_K> { static constexpr int B = 110, TAB = 0, MIN = 0; };
// MXFP4 (ds4_gguf.cu's own kernels everywhere else): a block of 32 values, an E8M0 scale then 16 code bytes; the light
// sets take it too
constexpr int kMXFP4 = 39;
template <> struct Ty<kMXFP4> { static constexpr int B = 17, TAB = 0, MIN = 0; };
template <int T> constexpr int kQK = T == kMXFP4 ? 32 : 256;   // values a block
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_s) == 82 && sizeof(block_iq3_xxs) == 98 &&
                  sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 && sizeof(block_q4_K) == 144 &&
                  sizeof(block_q2_K) == 84 && sizeof(block_q3_K) == 110,
              "ggml's block sizes");

template <int T>
__host__ __device__ __forceinline__ size_t row_bytes_t(int k) { return (size_t) (k / kQK<T>) * Ty<T>::B; }

// the type's codebook into shared memory (u32 words; IQ4_XS: its 16 values, from bytes - an int8 table's alignment)
template <int T>
__device__ __forceinline__ void load_tab(uint32_t* tab) {
    if constexpr (T == kIQ4_XS) {
        if (threadIdx.x < 4) {
            const uint8_t* kv = (const uint8_t*) kvalues_iq4nl + 4 * threadIdx.x;
            tab[threadIdx.x] = (uint32_t) kv[0] | (uint32_t) kv[1] << 8 | (uint32_t) kv[2] << 16 | (uint32_t) kv[3] << 24;
        }
    } else if constexpr (Ty<T>::TAB > 0) {
        const uint32_t* src = T == kIQ2_XXS  ? (const uint32_t*) iq2xxs_grid
                              : T == kIQ2_S   ? (const uint32_t*) iq2s_grid
                              : T == kIQ3_XXS ? (const uint32_t*) iq3xxs_grid
                                              : (const uint32_t*) iq3s_grid;
        for (int i = threadIdx.x; i < Ty<T>::TAB; i += blockDim.x) tab[i] = src[i];
    }
}

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
template <int T>
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
            const int2 v = tab16(ld4(qs + 4 * t), tab);
            s.q[t] = v.x;
            s.q[4 + t] = v.y;
        }
        s.s0 = s.s1 = ls - 32;
        s.d = h2f(b);
    } else if constexpr (T == kQ4_K) {
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const uint8_t* qs = b + 16 + 32 * (ib >> 1);
        const int sh = 4 * (ib & 1);
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) ((ld4(qs + 4 * t) >> sh) & 0x0f0f0f0fu);
        s.s0 = s.s1 = sc;
        s.d = h2f(b);
        s.m0 = s.m1 = h2f(b + 2) * (float) m;
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
        s.m0 = dmin * (float) (c0 >> 4);
        s.m1 = dmin * (float) (c1 >> 4);
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
    float v = __fmul_rn(s.d, (float) (s.s0 * lo + s.s1 * hi));
    if constexpr (Ty<T>::MIN) {
        int sl = __dp4a(0x01010101, xa.x, 0);
        sl = __dp4a(0x01010101, xa.y, sl);
        sl = __dp4a(0x01010101, xa.z, sl);
        sl = __dp4a(0x01010101, xa.w, sl);
        int sh = __dp4a(0x01010101, xb.x, 0);
        sh = __dp4a(0x01010101, xb.y, sh);
        sh = __dp4a(0x01010101, xb.z, sh);
        sh = __dp4a(0x01010101, xb.w, sh);
        v = __fsub_rn(v, __fadd_rn(__fmul_rn(s.m0, (float) sl), __fmul_rn(s.m1, (float) sh)));
    }
    return v;
}

// values [8 p, 8 p + 8) of sub-block kb, as F32 (the prompt sets' tiles): ggml's dequantize_row_* values
template <int T>
__device__ __forceinline__ void deq8(const uint8_t* __restrict__ row, int kb, int p, const uint32_t* tab, float* w) {
    const int ib = kb & 7;
    const uint8_t* b = row + (size_t) (kb >> 3) * Ty<T>::B;
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
    } else if constexpr (T == kQ4_K) {
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const float d1 = h2f(b) * (float) sc, m1 = h2f(b + 2) * (float) m;
        const uint8_t* qs = b + 16 + 32 * (ib >> 1) + 8 * p;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = d1 * (float) ((ib & 1) ? qs[j] >> 4 : qs[j] & 15) - m1;
    } else if constexpr (T == kQ2_K) {
        const int n = ib >> 2, jj = ib & 3;
        const int c = b[8 * n + 2 * jj + (p >> 1)];
        const float dl = h2f(b + 80) * (float) (c & 15), ml = h2f(b + 82) * (float) (c >> 4);
        const uint8_t* qs = b + 16 + 32 * n + 8 * p;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = dl * (float) ((qs[j] >> (2 * jj)) & 3) - ml;
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
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float swiglu(float gv, float uv, float limit) {   // (ds4_gguf.cu's)
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);
}
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
    for (int o = 16 / RW; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
constexpr int kGuRW = 1, kDnRW = 2;   // rows a warp: gate / up (n_embd 5120: 10 sub-blocks a lane), down (2304: 9)

// grid (n_ff / (8 RW), groups): RW rows of the gate and of the up a warp, NE of the group's entries a pass (each
// sub-block decoded once for them) -> h = the clamped swiglu
template <int T, int NE, int RW = kGuRW>
__global__ void __launch_bounds__(kThreads) gate_up_kernel(const Args a) {
    constexpr int LPR = 32 / RW;
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ int es[NE], xr[NE];
    const uint8_t* blob;
    int m0, m1;
    if (!group(a, blockIdx.y, blob, m0, m1)) return;   // (uniform)
    load_tab<T>(tab);
    const int lane32 = threadIdx.x & 31;
    const int row = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * RW + lane32 / LPR, lane = lane32 % LPR;
    const size_t rb = row_bytes_t<T>(a.n_embd);
    const uint8_t* g = blob + a.tr[0] + (size_t) row * rb;
    const uint8_t* u = blob + a.tr[1] + (size_t) row * rb;
    const int nsb = a.n_embd / 32;
    for (int mm = m0; mm < m1; mm += NE) {
        const int ne = min(NE, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) {
            const int e = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
            es[threadIdx.x] = e;
            xr[threadIdx.x] = a.grp_ptr != nullptr ? e >> 3 : 0;
        }
        __syncthreads();
        if (row >= a.n_ff) continue;   // (a whole warp: n_ff % (8 RW) == 0)
        float ag[NE], au[NE];
#pragma unroll
        for (int i = 0; i < NE; ++i) ag[i] = au[i] = 0.0f;
        for (int kb = lane; kb < nsb; kb += LPR) {
            Sub sg, su;
            decode<T>(g, kb, tab, sg);
            decode<T>(u, kb, tab, su);
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const int4* xp = (const int4*) (a.xq + (size_t) xr[i] * a.n_embd + (size_t) kb * 32);
                    const int4 xa = xp[0], xb = xp[1];
                    const float dx = a.xd[(size_t) xr[i] * nsb + kb];
                    ag[i] = fmaf(dx, sub_dot<T>(sg, xa, xb), ag[i]);
                    au[i] = fmaf(dx, sub_dot<T>(su, xa, xb), au[i]);
                }
        }
#pragma unroll
        for (int i = 0; i < NE; ++i)
            if (i < ne) {
                const float gv = lanes_sum<RW>(ag[i]), uv = lanes_sum<RW>(au[i]);
                if (lane == 0) a.h[(size_t) es[i] * a.n_ff + row] = swiglu(gv, uv, a.limit);
            }
    }
}

// grid (n_embd / (8 RW), groups): RW rows of the down a warp for NE entries a pass, against their int8 h -> part
template <int T, int NE, int RW = kDnRW>
__global__ void __launch_bounds__(kThreads) down_kernel(const Args a) {
    constexpr int LPR = 32 / RW;
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ int es[NE];
    const uint8_t* blob;
    int m0, m1;
    if (!group(a, blockIdx.y, blob, m0, m1)) return;
    load_tab<T>(tab);
    const int lane32 = threadIdx.x & 31;
    const int row = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * RW + lane32 / LPR, lane = lane32 % LPR;
    const uint8_t* d = blob + a.tr[2] + (size_t) row * row_bytes_t<T>(a.n_ff);
    const int nsb = a.n_ff / 32;
    for (int mm = m0; mm < m1; mm += NE) {
        const int ne = min(NE, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) es[threadIdx.x] = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
        __syncthreads();
        if (row >= a.n_embd) continue;
        float ad[NE];
#pragma unroll
        for (int i = 0; i < NE; ++i) ad[i] = 0.0f;
        for (int kb = lane; kb < nsb; kb += LPR) {
            Sub sd;
            decode<T>(d, kb, tab, sd);
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const int4* hp = (const int4*) (a.hq + (size_t) es[i] * a.n_ff + (size_t) kb * 32);
                    const int4 ha = hp[0], hb = hp[1];
                    ad[i] = fmaf(a.hd[(size_t) es[i] * nsb + kb], sub_dot<T>(sd, ha, hb), ad[i]);
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

// ---------------------------------------------------------------- the prompt path's sets
constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
static_assert((BM / TM) * (BN / TN) == kThreads, "one output tile a thread");

// F32 FMAs (any device): a block RM rows (64, or 16 for experts of a few rows: a short prompt's) x BN outputs, a 32-value
// sub-block of K a step, the weights dequantised into shared memory (each thread 8 values of a row), each thread RT x TN
// outputs.  (deq8 is the tile loader a tensor-core form takes too.)
template <int T, bool GATE_UP, int RM = BM>
__global__ void __launch_bounds__(kThreads) set_kernel(const SetArgs a) {
    constexpr int RT = RM / (kThreads / (BN / TN));
    static_assert(RT >= 1 && RT * (kThreads / (BN / TN)) == RM, "the block's rows");
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ float wg[32][BN + 1];
    __shared__ float wu[GATE_UP ? 32 : 1][BN + 1];
    __shared__ float xs[RM][33];
    __shared__ int src[RM];
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * RM;
    if (r0 >= hi || hi - lo <= a.skip_le) return;   // (uniform; skip_le: the light kernels' experts)
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
        deq8<T>(blob + a.tr[GATE_UP ? 0 : 2] + (size_t) (c0 + tr) * rb, kb, tq, tab, w8);
#pragma unroll
        for (int j = 0; j < 8; ++j) wg[tq * 8 + j][tr] = w8[j];
        if (GATE_UP) {
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
                if (GATE_UP) uw[j] = wu[kk][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < RT; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    ag[i][j] = fmaf(xv[i], gw[j], ag[i][j]);
                    if (GATE_UP) au[i][j] = fmaf(xv[i], uw[j], au[i][j]);
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
            if (GATE_UP) a.h[lr * a.ldo + c] = __half_as_ushort(__float2half_rn(swiglu(ag[i][j], au[i][j], a.limit)));
            else a.y[lr * a.ldo + c] = ag[i][j];
        }
    }
}

#if !defined(STRATA_USE_HIP)
// the same on the tensor cores (sm_70+): 8 warps each 16 rows x 32 outputs, the tiles F16 in shared memory
namespace wm = nvcuda::wmma;
constexpr int kLdT = 40;
template <int T, bool GATE_UP>
__global__ void __launch_bounds__(kThreads) set_tc_kernel(const SetArgs a) {
    struct Tiles {
        __half x[BM][kLdT], wg[BN][kLdT], wu[GATE_UP ? BN : 1][kLdT];
    };
    struct Out {
        float g[BM][BN + 4], u[GATE_UP ? BM : 1][BN + 4];
    };
    __shared__ __align__(32) unsigned char smem[sizeof(Tiles) > sizeof(Out) ? sizeof(Tiles) : sizeof(Out)];
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ int src[BM];
    Tiles& Tl = *reinterpret_cast<Tiles*>(smem);
    Out& O = *reinterpret_cast<Out*>(smem);
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi || hi - lo <= a.skip_le) return;   // (uniform; skip_le: the light kernels' experts)
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
        for (int m = 0; m < (GATE_UP ? 2 : 1); ++m) {
            float w8[8];
            deq8<T>(blob + a.tr[GATE_UP ? m : 2] + (size_t) (c0 + tr) * rb, kb, tq, tab, w8);
            __half* dst = m == 0 ? &Tl.wg[tr][tq * 8] : &Tl.wu[tr][tq * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) dst[j] = __float2half_rn(w8[j]);
        }
        const int sr = src[tr];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            __half v = __float2half_rn(0.0f);
            if (sr >= 0) {
                const size_t at = (size_t) sr * a.ldx + (size_t) kb * 32 + tq * 8 + j;
                v = GATE_UP ? __float2half_rn(a.xf[at]) : __ushort_as_half(a.xh[at]);
            }
            Tl.x[tr][tq * 8 + j] = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> fa;
            wm::load_matrix_sync(fa, &Tl.x[wm_ * 16][kk], kLdT);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> fb;
                wm::load_matrix_sync(fb, &Tl.wg[(wn + j) * 16][kk], kLdT);
                wm::mma_sync(fg[j], fa, fb, fg[j]);
                if (GATE_UP) {
                    wm::load_matrix_sync(fb, &Tl.wu[(wn + j) * 16][kk], kLdT);
                    wm::mma_sync(fu[j], fa, fb, fu[j]);
                }
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < 2; ++j) {
        wm::store_matrix_sync(&O.g[wm_ * 16][(wn + j) * 16], fg[j], BN + 4, wm::mem_row_major);
        if (GATE_UP) wm::store_matrix_sync(&O.u[wm_ * 16][(wn + j) * 16], fu[j], BN + 4, wm::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < BM * BN; i += kThreads) {
        const int r = i / BN, c = i % BN;
        if (r >= nr) continue;
        const size_t lr = (size_t) (r0 + r);
        if (GATE_UP) a.h[lr * a.ldo + c0 + c] = __half_as_ushort(__float2half_rn(swiglu(O.g[r][c], O.u[r][c], a.limit)));
        else a.y[lr * a.ldo + c0 + c] = O.g[r][c];
    }
}
#endif

// ---------------------------------------------------------------- GEMVs
constexpr int kMaxJobs = 8;
struct Jobs {
    MvJob j[kMaxJobs];
};
template <int T, int NC>
__global__ void __launch_bounds__(kThreads) mv_kernel(const Jobs J) {
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
// row of the expert a pass -> h16 (the F16 swiglu) or y
template <int T, int NE, bool GATE_UP>
__global__ void __launch_bounds__(kThreads) light_kernel(const SetArgs a, const int8_t* __restrict__ xq,
                                                          const float* __restrict__ xd) {
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    const int e = blockIdx.y;
    const int lo = a.bounds[e] - a.row0, ne = a.bounds[e + 1] - a.bounds[e];
    if (ne <= 0 || ne > NE) return;   // (uniform: the tiles take an expert of more rows)
    load_tab<T>(tab);
    __syncthreads();
    const int K = GATE_UP ? a.n_embd : a.n_ff, N = GATE_UP ? a.n_ff : a.n_embd, nsb = K / 32;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const size_t rb = row_bytes_t<T>(K);
    const uint8_t* w0 = blob + a.tr[GATE_UP ? 0 : 2] + (size_t) row * rb;
    const uint8_t* w1 = blob + a.tr[1] + (size_t) row * rb;
    const size_t q0 = (size_t) e * kLightRows;
    float a0[NE], a1[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) a0[i] = a1[i] = 0.0f;
    for (int kb = lane; kb < nsb; kb += 32) {
        Sub s0, s1;
        decode<T>(w0, kb, tab, s0);
        if (GATE_UP) decode<T>(w1, kb, tab, s1);
#pragma unroll
        for (int i = 0; i < NE; ++i)
            if (i < ne) {
                const int4* xp = (const int4*) (xq + (q0 + i) * K + (size_t) kb * 32);
                const int4 xa = xp[0], xb = xp[1];
                const float dx = xd[(q0 + i) * nsb + kb];
                a0[i] = fmaf(dx, sub_dot<T>(s0, xa, xb), a0[i]);
                if (GATE_UP) a1[i] = fmaf(dx, sub_dot<T>(s1, xa, xb), a1[i]);
            }
    }
#pragma unroll
    for (int i = 0; i < NE; ++i)
        if (i < ne) {
            const float v0 = warp_sum(a0[i]);
            const float v1 = GATE_UP ? warp_sum(a1[i]) : 0.0f;
            if (lane == 0) {
                const size_t lr = (size_t) (lo + i);
                if (GATE_UP) a.h[lr * a.ldo + row] = __half_as_ushort(__float2half_rn(swiglu(v0, v1, a.limit)));
                else a.y[lr * a.ldo + row] = v0;
            }
        }
}

bool tc_ok() {   // as ds4_gguf.cu's: sm_70 and later, STRATA_DS4_NO_TC=1 keeps the F32 kernel
#if defined(STRATA_USE_HIP)
    return false;
#else
    static const bool ok = [] {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        const char* v = std::getenv("STRATA_DS4_NO_TC");
        return major >= 7 && !(v != nullptr && v[0] == '1');
    }();
    return ok;
#endif
}

// ---------------------------------------------------------------- dispatch
// F(T) for the type t at run time; false when it is not one of these
#define DS4_IQ_SWITCH(t, F)          \
    switch (t) {                     \
        case kIQ2_XXS: F(kIQ2_XXS); break; \
        case kIQ2_S: F(kIQ2_S); break;     \
        case kIQ3_XXS: F(kIQ3_XXS); break; \
        case kIQ3_S: F(kIQ3_S); break;     \
        case kIQ4_XS: F(kIQ4_XS); break;   \
        case kQ4_K: F(kQ4_K); break;       \
        case kQ2_K: F(kQ2_K); break;       \
        case kQ3_K: F(kQ3_K); break;       \
        default: return false;       \
    }

template <int T, int NE>
void launch_gu(const Args& a, dim3 grid, cudaStream_t s) { gate_up_kernel<T, NE><<<grid, kThreads, 0, s>>>(a); }
template <int T, int NE>
void launch_dn(const Args& a, dim3 grid, cudaStream_t s) { down_kernel<T, NE><<<grid, kThreads, 0, s>>>(a); }
template <int T>
void gu_ne(const Args& a, dim3 grid, int ne, cudaStream_t s) {
    if (ne <= 1) launch_gu<T, 1>(a, grid, s);
    else if (ne <= 4) launch_gu<T, 4>(a, grid, s);
    else if (ne <= 8) launch_gu<T, 8>(a, grid, s);
    else launch_gu<T, 16>(a, grid, s);
}
template <int T>
void dn_ne(const Args& a, dim3 grid, int ne, cudaStream_t s) {
    if (ne <= 1) launch_dn<T, 1>(a, grid, s);
    else if (ne <= 4) launch_dn<T, 4>(a, grid, s);
    else if (ne <= 8) launch_dn<T, 8>(a, grid, s);
    else launch_dn<T, 16>(a, grid, s);
}
template <int T, bool GU>
void launch_set(const SetArgs& a, dim3 grid, cudaStream_t s) {
#if !defined(STRATA_USE_HIP)
    if (tc_ok()) {
        set_tc_kernel<T, GU><<<grid, kThreads, 0, s>>>(a);
        return;
    }
#endif
    if (a.max_rows <= 16) {   // (a block of 64 rows would do 64 rows' FMAs for the few there are)
        grid.x = (unsigned) ((a.max_rows + 15) / 16);
        set_kernel<T, GU, 16><<<grid, kThreads, 0, s>>>(a);
        return;
    }
    set_kernel<T, GU><<<grid, kThreads, 0, s>>>(a);
}
// the light sets' int8 rows, per stream (kLightMaxExp kLightRows rows of k)
struct LightBuf {
    int8_t* q = nullptr;
    float* d = nullptr;
    size_t cap = 0;
};
std::mutex g_lmu;
std::unordered_map<cudaStream_t, LightBuf> g_light;
LightBuf* light_buf(cudaStream_t s, int k) {
    std::lock_guard<std::mutex> lk(g_lmu);
    LightBuf& b = g_light[s];
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
template <int T, bool GU>
void launch_light(const SetArgs& a, const LightBuf& b, cudaStream_t s) {
    const dim3 grid((unsigned) (((GU ? a.n_ff : a.n_embd) + kWarps - 1) / kWarps), (unsigned) a.n_exp);
    if (a.max_rows <= 4) light_kernel<T, 4, GU><<<grid, kThreads, 0, s>>>(a, b.q, b.d);
    else if (a.max_rows <= 8) light_kernel<T, 8, GU><<<grid, kThreads, 0, s>>>(a, b.q, b.d);
    else light_kernel<T, kLightRows, GU><<<grid, kThreads, 0, s>>>(a, b.q, b.d);
}
bool light_set(const SetArgs& a, bool gu, cudaStream_t s) {
    const int K = gu ? a.n_embd : a.n_ff;
    LightBuf* b = light_buf(s, K);
    if (b == nullptr) return false;
    const dim3 gq((unsigned) ((K / 32 + kWarps - 1) / kWarps), (unsigned) (a.n_exp * kLightRows));
    light_q8_kernel<<<gq, kThreads, 0, s>>>(gu ? a.xf : nullptr, gu ? nullptr : a.xh, a.idx, a.ldx, K, a.bounds,
                                            a.row0, b->q, b->d);
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
    if (max_c <= 1) mv_kernel<T, 1><<<grid, kThreads, 0, s>>>(J);
    else if (max_c <= 4) mv_kernel<T, 4><<<grid, kThreads, 0, s>>>(J);
    else mv_kernel<T, 16><<<grid, kThreads, 0, s>>>(J);
}

}  // namespace

bool supported(int t) { return block_bytes(t) > 0; }

int block_bytes(int t) {
    switch (t) {
        case kIQ2_XXS: return Ty<kIQ2_XXS>::B;
        case kIQ2_S: return Ty<kIQ2_S>::B;
        case kIQ3_XXS: return Ty<kIQ3_XXS>::B;
        case kIQ3_S: return Ty<kIQ3_S>::B;
        case kIQ4_XS: return Ty<kIQ4_XS>::B;
        case kQ4_K: return Ty<kQ4_K>::B;
        case kQ2_K: return Ty<kQ2_K>::B;
        case kQ3_K: return Ty<kQ3_K>::B;
        default: return 0;
    }
}

const char* name(int t) {
    switch (t) {
        case kIQ2_XXS: return "IQ2_XXS";
        case kIQ2_S: return "IQ2_S";
        case kIQ3_XXS: return "IQ3_XXS";
        case kIQ3_S: return "IQ3_S";
        case kIQ4_XS: return "IQ4_XS";
        case kQ4_K: return "Q4_K";
        case kQ2_K: return "Q2_K";
        case kQ3_K: return "Q3_K";
        default: return "?";
    }
}

bool gate_up(const Args& a, int groups, int ne_max, cudaStream_t s) {
    if (a.n_embd % 256 != 0 || a.n_ff % (kWarps * kGuRW) != 0) return false;
    const dim3 grid((unsigned) (a.n_ff / (kWarps * kGuRW)), (unsigned) groups);
#define GU_(T) gu_ne<T>(a, grid, ne_max, s)
    DS4_IQ_SWITCH(a.tg, GU_)
#undef GU_
    return true;
}

bool down(const Args& a, int groups, int ne_max, cudaStream_t s) {
    if (a.n_ff % 256 != 0 || a.n_embd % (kWarps * kDnRW) != 0) return false;
    const dim3 grid((unsigned) (a.n_embd / (kWarps * kDnRW)), (unsigned) groups);
#define DN_(T) dn_ne<T>(a, grid, ne_max, s)
    DS4_IQ_SWITCH(a.td, DN_)
#undef DN_
    return true;
}

bool light_gate_up(const SetArgs& a, cudaStream_t s) {
    if (a.n_exp > kLightMaxExp || !light_on() || a.n_embd % 256 != 0 || a.n_ff % 256 != 0) return false;
    return light_set(a, true, s);
}
bool light_down(const SetArgs& a, cudaStream_t s) {
    if (a.n_exp > kLightMaxExp || !light_on() || a.n_embd % 256 != 0 || a.n_ff % 256 != 0) return false;
    return light_set(a, false, s);
}

#if defined(STRATA_USE_HIP)
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

bool set_gate_up(const SetArgs& a, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    if (mmq_wanted(a, true) && mmq_gate_up(a, s)) return true;
#endif
    if (a.n_embd % 256 != 0 || a.n_ff % BN != 0) return false;
    SetArgs t = a;
    if (light_gate_up(a, s)) {   // the experts of a few rows; the tiles the others
        if (a.max_rows <= kLightRows) return true;
        t.skip_le = kLightRows;
    }
    const dim3 grid((unsigned) ((a.max_rows + BM - 1) / BM), (unsigned) (a.n_ff / BN), (unsigned) a.n_exp);
#define SG_(T) launch_set<T, true>(t, grid, s)
    DS4_IQ_SWITCH(a.tg, SG_)
#undef SG_
    return true;
}

bool set_down(const SetArgs& a, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    if (mmq_wanted(a, false) && mmq_down(a, s)) return true;
#endif
    if (a.n_ff % 256 != 0 || a.n_embd % BN != 0) return false;
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
                if (jobs[k].k % 256 != 0 || jobs[k].ncol < 1 || jobs[k].ncol > 16) return false;
                J.j[nj++] = jobs[k];
                max_n = std::max(max_n, jobs[k].n);
                max_c = std::max(max_c, jobs[k].ncol);
                done[k] = true;
            }
#define MV_(T) mv_nc<T>(J, nj, max_n, max_c, s)
            DS4_IQ_SWITCH(t, MV_)
#undef MV_
        }
    return true;
}

}  // namespace strata::ds4::gguf::iq
