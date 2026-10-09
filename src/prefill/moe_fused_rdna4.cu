// HIP-only GLM prompt experts, adapted from Niko1221/Strata (MIT), fb58e0d
// and PR #1570 e667a774. Copyright (c) 2026 Niko1221 and the Strata contributors;
// see LICENSE. Maya keeps its sorted rows, resident/prestaged/ring slots and output windows.
//
// Activations have signed int8 codes and FP32 scales per 32 values (80 bytes per
// 64 values, natural order). IQ weight blocks are decoded with GGML's codebooks
// into signed int8 and their original scales. WMMA accumulates exact int32 dots;
// the MAGIC bias converts each dot to FP32 exactly before scaling and accumulation.
// GU also clips the gate/up, applies SwiGLU and quantizes H; down writes FP32 rows.
//
// Unlike the reference's 2560/640 experts, GLM uses E=4096, FF=2048. IQ2_S and
// IQ3_XXS down matrices also use the 256-value block addressing of the GU decoder.
// Only populated 64-row tiles are enumerated from Maya's existing expert bounds.
#include "strata/prefill/moe_fused_rdna4.hpp"

#include "common.cuh"

#include "ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strata::prefill::rdna4 {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fused experts (native): %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int kTileRows = 64;

// Enumerate only populated expert tiles. Each set already has sorted bounds;
// no routing, row ordering or staging policy changes are needed.
__global__ void make_tiles_kernel(const int* bounds, int ne, int2* tiles) {
    __shared__ int start[513];
    if (threadIdx.x == 0) {
        start[0] = 0;
        for (int e = 0; e < ne; ++e)
            start[e + 1] = start[e] + (bounds[e + 1] - bounds[e] + 63) / 64;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < ne; e += blockDim.x)
        for (int t = start[e]; t < start[e + 1]; ++t)
            tiles[t] = make_int2(e, bounds[e] - bounds[0] + 64 * (t - start[e]));
}
constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, {d0, d1}, 8 unused
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
constexpr int WLD = 80; // 64 decoded int8 weights plus padding to avoid LDS bank conflicts
constexpr int GU_ROWS_K = 4096, D_ROWS_K = 2048;   // K of gate/up (n_embd) and of down (n_ff)

// the formats (ggml type ids)
constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S, T_IQ4_XS = GGML_TYPE_IQ4_XS,
              T_IQ4_NL = GGML_TYPE_IQ4_NL, T_Q2_0 = GGML_TYPE_Q2_0;
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 &&
              sizeof(block_iq4_nl) == 18 && sizeof(block_q2_0) == 18,
              "the block layouts this file decodes");

// block bytes, scale per 16 values, codebook bytes in shared memory
__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : t == T_IQ3_S ? 110
         : t == T_IQ4_XS ? 136 : 18;
}
// These covered formats have no additive minimum and need at most five raw words.
__host__ __device__ constexpr bool has_min(int) { return false; }
__host__ __device__ constexpr int raw_words(int) { return 5; }
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4
         : t == T_IQ3_S ? 512 * 4 : 0;
}
// ---- activations in natural order: one warp per 64 values
__global__ void quant_act_nat_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o, 32));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o, 32));
    }
    uint8_t* out = xa + w * AB;
    const int c0 = a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0, c1 = a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0;
    out[lane] = (uint8_t) (int8_t) c0;
    out[32 + lane] = (uint8_t) (int8_t) c1;
    int s0 = c0, s1 = c1;   // the codes' sums per 32 values (the formats with a minimum read them)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o, 32);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o, 32);
    }
    if (lane == 0) *(float4*) (out + 64) = make_float4(a0 / 127.0f, a1 / 127.0f, (float) s0, (float) s1);
}

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || defined(__HIPCC__)
#if defined(__HIPCC__)
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
#endif
// ---- the load stage: a 32-value sub-block's bytes into registers, then int8 and scales
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// llama.cpp's sign unpacking: 7 bits of signs, the 8th their parity (bit 7 of v may be anything)
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 bytes of a codebook entry with the sign byte `s` (broadcast) applied: bits 0-3 to .x, 4-7 to .y
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}
#if defined(__HIPCC__)
// gfx11 (Aurora): the signs from a shared-memory table instead of the compare / subtract emulation.  An entry (one per
// sign byte b; for the 7-bit sign formats per 7 bits, the parity bit folded in) is {m0, c0, m1, c1}: m = 0xFF in the
// bytes whose sign bit is set (bits 0-3 for .x, 4-7 for .y), c = m & 0x01010101.  q = (g ^ m) + c is -g where the sign
// is set - the two's complement, byte-wise - and g elsewhere; no byte carries into the next, because no codebook entry
// has a zero byte (256 - g with g in 1..127; the grids' bytes are 1..62), so it is the bytes signed8 gives.
__device__ __forceinline__ void signed8t(uint32_t gx, uint32_t gy, const uint4 sg, uint32_t& qx, uint32_t& qy) {
    qx = (gx ^ sg.x) + sg.y;
    qy = (gy ^ sg.z) + sg.w;
}
__host__ __device__ constexpr int sign_entries(int t) { return t == T_IQ2_S || t == T_IQ3_S ? 256 : 128; }
__device__ __forceinline__ uint4 sign_entry(int i, bool parity) {
    uint32_t b = (uint32_t) i;
    if (parity) b ^= (__popc(b) & 1) << 7;
    const uint32_t m0 = __vcmpne4(((b & 15) * 0x01010101u) & 0x08040201u, 0),
                   m1 = __vcmpne4(((b >> 4) * 0x01010101u) & 0x08040201u, 0);
    return make_uint4(m0, m0 & 0x01010101u, m1, m1 & 0x01010101u);
}
#endif
// 8 nibbles of q4 through a 16-entry int8 table (4 words): the low nibbles' values in .x, the high ones' in .y
// W (X5): direct-selector form on gfx: p = low 3 bits of each nibble picks within the table half, bit 3 picks the half;
// exhaustively bit-exact over all 2^32 inputs (X5). Disable with -DSTRATA_W_NO_T16.
#if defined(__HIPCC__) && !defined(STRATA_W_NO_T16)
__device__ __forceinline__ uint32_t table16_one(uint32_t x, const uint32_t (&t)[4]) {
    const uint32_t p = x & 0x07070707u;
    const uint32_t a = __builtin_amdgcn_perm(t[1], t[0], p), b = __builtin_amdgcn_perm(t[3], t[2], p);
    return __builtin_amdgcn_perm(b, a, ((x >> 1) & 0x04040404u) | 0x03020100u);
}
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    lo = table16_one(q4, t);
    hi = table16_one(q4 >> 4, t);
}
#else
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}
#endif

// Raw bytes of sub-block `ib` of the block at `bp` (IQ: the 256-value super-block, ib 0..7; Q2_0: the 64-value block,
// ib 0..1; IQ4_NL: the 32-value block).
// 32 bits at a 2-byte aligned address (one load when it is 4-byte aligned)
__device__ __forceinline__ uint32_t ld32a(const uint8_t* p) {
    return ((uintptr_t) p & 3) == 0 ? *(const uint32_t*) p : ld32(p);
}
template <int T, int NW> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[NW]) {
    static_assert(NW >= raw_words(T), "raw words");
    if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 8 + 16 * ib + 4 * k);
        const uint32_t ls = ((bp[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((ld16(bp + 2) >> (2 * ib)) & 3) << 4);
        w[4] = ld16(bp) | (ls << 16);
    } else if constexpr (T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    } else {   // Q2_0
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    }
}

// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31).
template <int T, int NW>
__device__ __forceinline__ void convert(const uint32_t (&w)[NW], const uint8_t* grid, const uint32_t (&kv)[4],
                                        uint32_t (&q)[8], float& s0, float& s1, const uint4* sgn = nullptr) {
    if constexpr (T == T_IQ2_XXS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(w[1] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, unpack_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(c >> 9) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, unpack_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
#endif
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(w[1] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
#endif
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
#if defined(__HIPCC__)
            signed8t(g[i0], g[i1], sgn[(w[2] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else if constexpr (T == T_IQ3_S) {
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
#if defined(__HIPCC__)
            signed8t(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                     sgn[(w[2] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
#else
            signed8(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                    ((w[2] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    } else if constexpr (T == T_IQ4_XS || T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s0 = s1 = T == T_IQ4_XS ? half_at(w[4]) * (float) ((int) (w[4] >> 16) - 32) : half_at(w[4]);
    } else {   // Q2_0: code - 1 via a byte table {-1, 0, 1, 2}
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint32_t c = (w[h >> 1] >> (16 * (h & 1))) & 0xFFFF;
            const uint32_t qe = __byte_perm(0x020100FFu, 0x020100FFu, c & 0x7777);
            const uint32_t qo = __byte_perm(0x020100FFu, 0x020100FFu, (c >> 2) & 0x7777);
            q[2 * h] = __byte_perm(qe, qo, 0x5140);
            q[2 * h + 1] = __byte_perm(qe, qo, 0x7362);
        }
        s0 = s1 = half_at(w[2]);
    }
}

template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == T_IQ3_S) return iq3s_grid;
    else return nullptr;
}
#endif

#if defined(__HIPCC__)
// ---- Aurora (S23): the native packs' fused experts on gfx11 (RDNA3 / RDNA3.5) matrix cores, v_wmma_i32_16x16x16_iu8
// (wave32).  The CUDA kernel's arithmetic: a 32-value sub-block decoded to int8 (load_unit / convert above) and its
// scales; per 32 values the int32 dot starts from MAGIC (the C operand), so as_float(d) - 1.5 * 2^23 is the dot; the
// formats with a scale per 16 values take each 16-value k-step's dot alone.  Fragments (gfx11): A lane l holds the 16
// k of row l % 16 (lanes 16..31 repeat lanes 0..15), B lane l the 16 k of column l % 16, C/D lane l holds
// D[2i + l / 16][l % 16].  Both operands are in natural order: the decoded weights (LDS, double-buffered) and the
// activations (quant_act_nat_kernel; read by each lane from global/L2, a stage ahead).
// Weights are A and activations are B: lane l owns token column l % 16,
// and element i owns weight row 2*i + l/16. Swapping operands requires changing
// scale selection and both epilogues, not just the intrinsic arguments.
// A work item = NW_ROWS weight rows x a 64-row tile; 8 waves: 4 along the weight rows (32 each) x 2 along the tile
// (32 each), each 2 x 2 WMMA tiles.  Gate/up: local row r is feature r / 2, its gate (r even) or up (r odd) row, so a
// lane pair l, l + 16 holds gate and up of one feature.
// gfx12 (RDNA4: gfx1200 / gfx1201, wave32) has the same instruction with another layout (checked on gfx1201, R9700):
// A lane l holds A[l % 16][8 (l / 16) + j], B lane l B[8 (l / 16) + j][l % 16] (j = 0..7: 8 int8, two VGPRs, no
// replication), C/D lane l holds D[8 (l / 16) + i][l % 16].  A WMMA still contracts 16 k; lane half hi supplies k
// 8 hi .. 8 hi + 7 of them, so a 16-value k-step is still one WMMA (the K16 scales stay per WMMA).  NW_DROW is the
// D row of a lane's element i; on gfx12 a lane's 8 rows are consecutive, so a feature's gate (even local row) and up
// (odd) sit in one lane (elements 2j, 2j + 1) and the SwiGLU needs no cross-lane exchange.
// gfx11 device code follows Strata's moe_fused_iq.cu (the same v_wmma_i32_16x16x16_iu8 layout;
// gfx1151 is RDNA3.5); gfx12 keeps the RDNA4 layout below. Maya builds one arch at a time.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_NAT_W11 1
#else
#define STRATA_NAT_W11 0
#endif
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_NAT_W12 1
#else
#define STRATA_NAT_W12 0
#endif
#define STRATA_NAT_WMMA (STRATA_NAT_W11 || STRATA_NAT_W12)
#if STRATA_NAT_W12
#define NW_DROW(i, hi) (8 * (hi) + (i))
#else
#define NW_DROW(i, hi) (2 * (i) + (hi))
#endif
// The VGPR cap of the occupancy variant (amdgpu_waves_per_eu).  8 was measured against ROCm 7's clang; clang 22 (ROCm 7.10)
// computes wrong results with it (#1180: gfx1100, every run), so there the cap is off.  -DSTRATA_W_LB=N sets it.
#if defined(STRATA_W_LB)
constexpr int NW_LB = STRATA_W_LB;
#elif defined(__clang_major__) && __clang_major__ >= 22
constexpr int NW_LB = 1;
#else
constexpr int NW_LB = 1;
#endif
typedef int nw_i4 __attribute__((ext_vector_type(4)));
typedef int nw_i2 __attribute__((ext_vector_type(2)));
typedef int nw_i8 __attribute__((ext_vector_type(8)));
constexpr int NW_ROWS = 128;
constexpr int NW_THREADS = 256;

// one A / B fragment: gfx11 16 int8 per lane (uint4), gfx12 8 (uint2)
#if STRATA_NAT_W12
typedef uint2 nw_frag;
typedef nw_i2 nw_ab;
__device__ __forceinline__ nw_ab nw_v(uint2 v) { return nw_ab{(int) v.x, (int) v.y}; }
#else
typedef uint4 nw_frag;
typedef nw_i4 nw_ab;
__device__ __forceinline__ nw_ab nw_v(uint4 v) { return nw_ab{(int) v.x, (int) v.y, (int) v.z, (int) v.w}; }
#endif
__device__ __forceinline__ nw_i8 nw_wmma(nw_ab a, nw_ab b, nw_i8 c) {
#if STRATA_NAT_W11
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#elif STRATA_NAT_W12
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    __builtin_trap();
    return c;
#endif
}

// W (X1): ALIAS = the H tile lives on the weight buffers (one extra barrier); LB > 0 = amdgpu_waves_per_eu(LB) (VGPR cap).
// Disable both with -DSTRATA_W_NO_OCC (the launch sites then use <WT, GU, false, 0> and grid factor wgp_blocks).
template <int WT, bool GU, bool ALIAS = false, int LB = 0>
__global__ void __launch_bounds__(NW_THREADS) __attribute__((amdgpu_waves_per_eu(LB > 0 ? LB : 1)))
native_kernel(const uint8_t* wbase, size_t stride, int ntiles, const int2* tiles, const int* bounds,
                  const Geometry geo, const uint8_t* __restrict__ act,
                  const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if STRATA_NAT_WMMA
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;  // 64-value stages along K
    constexpr int NFB = 4096 / NW_ROWS;     // work items per tile
    constexpr int ACT_LD = NS * AB;
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    constexpr int GB = grid_bytes(WT) > 0 ? grid_bytes(WT) : 16;
    __shared__ __align__(16) uint8_t wtbuf[2 * NW_ROWS * WLD];
    uint8_t(*wt)[NW_ROWS][WLD] = reinterpret_cast<uint8_t(*)[NW_ROWS][WLD]>(wtbuf);
    __shared__ __align__(16) float ws[2][NW_ROWS][4];           // their scales per 16 values
    __shared__ __align__(16) uint8_t sgrid[GB];
    constexpr int SGN = (WT == T_IQ2_XXS || WT == T_IQ2_XS || WT == T_IQ2_S || WT == T_IQ3_XXS || WT == T_IQ3_S)
                            ? sign_entries(WT) : 1;
    __shared__ __align__(16) uint4 ssign[SGN];                   // the signs of a sign byte, as byte masks
    __shared__ int srow[kTileRows];
    __shared__ float hs_sep[(GU && !ALIAS) ? kTileRows : 1][65];
    static_assert(!ALIAS || (size_t) kTileRows * 65 * 4 <= (size_t) 2 * NW_ROWS * WLD, "H tile must fit the weight buffers");
    float(*hs)[65] = ALIAS ? reinterpret_cast<float(*)[65]>(wtbuf) : hs_sep;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 3, wn = wave >> 2, l16 = lane & 15, hi = lane >> 4;
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += NW_THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    if constexpr (SGN > 1) {
        for (int i = tid; i < SGN; i += NW_THREADS) ssign[i] = sign_entry(i, SGN == 128);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    const int ur = tid >> 1, uj = tid & 1;                       // this thread's decode unit: row ur, sub-block uj
    const int nwork = ntiles * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tile = tiles[w / NFB];
        const int fb = w % NFB, e = tile.x, row0 = tile.y;
        const int nrows = min(kTileRows, bounds[e + 1] - bounds[0] - row0);
        if (nrows <= 0) continue; // uniform for the block; no barriers are skipped by individual lanes
        const uint8_t* blob = wbase + (size_t) e * stride;
        const int rbase = fb * NW_ROWS;
        const uint8_t* wrow = GU ? blob + ((ur & 1) ? geo.up_off : 0) + (size_t) (fb * (NW_ROWS / 2) + (ur >> 1)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        auto unit = [&](int s) -> const uint8_t* {
            if (GU || WT == T_IQ2_S || WT == T_IQ3_XXS) return wrow + (s >> 2) * BS;
            return (WT == T_IQ4_NL) ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return (GU || WT == T_IQ2_S || WT == T_IQ3_XXS) ? 2 * (s & 3) + uj : uj; };
        auto put = [&](const uint32_t (&raw)[raw_words(WT)], int buf) {
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1, ssign);
            uint4* d = (uint4*) &wt[buf][ur][32 * uj];
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) &ws[buf][ur][2 * uj] = make_float2(s0, s1);
        };
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < kTileRows) {
            const int r = row0 + min(tid, nrows - 1);             // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        uint32_t raw[raw_words(WT)] = {};
        load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (NS > 1) load_unit<WT>(unit(1), sub(1), raw);
        __syncthreads();                                          // srow
        const bool on = 32 * wn < nrows;
        const uint8_t* brow[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) brow[nt] = act + (size_t) srow[32 * wn + 16 * nt + l16] * ACT_LD;
        float acc[2][2][8];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][nt][i] = 0.0f;
        nw_frag bq[2][4];        // B per 16-value k-step: bq[nt][2 h + kk] = k 32 h + 16 kk .. (gfx12: this lane's 8 of them)
        float2 bx[2], bsum[2];   // the activation scales, and (the formats with a minimum) the codes' sums per 32 values
        auto fetch_b = [&](int s, nw_frag (&q)[2][4], float2 (&x)[2], float2 (&sm)[2]) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#if STRATA_NAT_W12
                const uint2* p = reinterpret_cast<const uint2*>(brow[nt] + s * AB + 8 * hi);   // k 16 j + 8 hi .. + 7
                q[nt][0] = p[0]; q[nt][1] = p[2]; q[nt][2] = p[4]; q[nt][3] = p[6];
#else
                const uint4* p = reinterpret_cast<const uint4*>(brow[nt] + s * AB);
                q[nt][0] = p[0]; q[nt][1] = p[1]; q[nt][2] = p[2]; q[nt][3] = p[3];
#endif
                x[nt] = *reinterpret_cast<const float2*>(brow[nt] + s * AB + 64);
                if constexpr (has_min(WT)) sm[nt] = *reinterpret_cast<const float2*>(brow[nt] + s * AB + 72);
                else sm[nt] = make_float2(0.0f, 0.0f);
            }
        };
        if (on) fetch_b(0, bq, bx, bsum);
        for (int s = 0; s < NS; ++s) {
            __syncthreads();                                      // stage s's weights are in buffer s & 1
            nw_frag nbq[2][4];
            float2 nbx[2], nbsum[2];
            if (on && s + 1 < NS) fetch_b(s + 1, nbq, nbx, nbsum);
            if (on) {
                const int bf = s & 1;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const int rb = 32 * wm + 16 * mt;
#if STRATA_NAT_W12
                        // k 32 h + 8 hi .. + 7 (A0) and 32 h + 16 + 8 hi .. + 7 (A1) of row rb + l16
                        const uint2* ap = reinterpret_cast<const uint2*>(&wt[bf][rb + l16][32 * h + 8 * hi]);
                        const nw_ab A0 = nw_v(ap[0]), A1 = nw_v(ap[2]);
#else
                        const uint4* ap = reinterpret_cast<const uint4*>(&wt[bf][rb + l16][32 * h]);
                        const nw_ab A0 = nw_v(ap[0]), A1 = nw_v(ap[1]);
#endif
                        float w0[8], w1[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float2 sw = *reinterpret_cast<const float2*>(&ws[bf][rb + NW_DROW(i, hi)][2 * h]);
                            w0[i] = sw.x; w1[i] = sw.y;
                        }
#pragma unroll
                        for (int nt = 0; nt < 2; ++nt) {
                            const float dx = h ? bx[nt].y : bx[nt].x;
                            const nw_ab B0 = nw_v(bq[nt][2 * h]), B1 = nw_v(bq[nt][2 * h + 1]);
                            const nw_i8 m = nw_i8{MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC};
                            if constexpr (K16) {
                                const nw_i8 d0 = nw_wmma(A0, B0, m), d1 = nw_wmma(A1, B1, m);
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    const float v = fmaf(w1[i], dotf(d1[i]), w0[i] * dotf(d0[i]));
                                    acc[mt][nt][i] = fmaf(dx, v, acc[mt][nt][i]);
                                }
                            } else {
                                const nw_i8 d = nw_wmma(A1, B1, nw_wmma(A0, B0, m));
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    acc[mt][nt][i] = fmaf(w0[i] * dx, dotf(d[i]), acc[mt][nt][i]);
                                    if constexpr (has_min(WT)) acc[mt][nt][i] = fmaf(w1[i] * dx, h ? bsum[nt].y : bsum[nt].x, acc[mt][nt][i]);
                                }
                            }
                        }
                    }
                }
            }
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);                            // the other buffer: its readers passed this barrier
                __syncthreads();                                      // the writes of every thread are in LDS before the next stage reads them (#1180)
                if (s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
                if (on) {
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        bx[nt] = nbx[nt];
                        bsum[nt] = nbsum[nt];
#pragma unroll
                        for (int j = 0; j < 4; ++j) bq[nt][j] = nbq[nt][j];
                    }
                }
            }
        }
        if constexpr (GU) {
            if constexpr (ALIAS) __syncthreads();                 // every wave is done reading the weight buffers
            if (on) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
#if STRATA_NAT_W12
                        // lane l: local rows rb + 8 hi + i = features 16 wm + 8 mt + 4 hi + j, gate (i = 2j) and up (2j + 1)
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float gt = fminf(acc[mt][nt][2 * j], geo.limit);
                            const float up = fminf(fmaxf(acc[mt][nt][2 * j + 1], -geo.limit), geo.limit);
                            hs[32 * wn + 16 * nt + l16][16 * wm + 8 * mt + 4 * hi + j] = gt / (1.0f + __expf(-gt)) * up;
                        }
#else
                        // Even/odd weight rows put gate/up in paired lanes; match swiglu_rows' clamp.
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float raw_up = __shfl_xor_sync(0xffffffffu, acc[mt][nt][i], 16, 32);
                            const float up = fminf(fmaxf(raw_up, -geo.limit), geo.limit);
                            const float gt = fminf(acc[mt][nt][i], geo.limit);
                            if (hi == 0) hs[32 * wn + 16 * nt + l16][16 * wm + 8 * mt + i] = gt / (1.0f + __expf(-gt)) * up;
                        }
#endif
                    }
            }
            __syncthreads();
            // H block fb (64 features) to int8 per 32: one thread per tile row and half
            if (tid < 2 * kTileRows) {
                const int r = tid >> 1, hh = tid & 1;
                if (r < nrows) {
                    float am = 0.0f;
#pragma unroll 8
                    for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(hs[r][32 * hh + j]));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    uint8_t* o = out + (size_t) (row0 + r) * (32 * AB) + (size_t) fb * AB;
                    uint32_t wd[8] = {};
                    int csum = 0;
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        const int c = __float2int_rn(hs[r][32 * hh + j] * inv);
                        csum += c;
                        wd[j >> 2] |= (uint32_t) (uint8_t) (int8_t) c << (8 * (j & 3));
                    }
                    uint4* o4 = reinterpret_cast<uint4*>(o + 32 * hh);
                    o4[0] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
                    o4[1] = make_uint4(wd[4], wd[5], wd[6], wd[7]);
                    *reinterpret_cast<float*>(o + 64 + 4 * hh) = am / 127.0f;
                    *reinterpret_cast<float*>(o + 72 + 4 * hh) = (float) csum;   // (the formats with a minimum)
                }
            }
        } else {
            if (on) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const int r = 32 * wn + 16 * nt + l16;
                        if (r < nrows) {
#if STRATA_NAT_W12
                            // columns rbase + 32 wm + 16 mt + 8 hi .. + 7: contiguous, 32-byte aligned (dm: 16, checked in experts_native)
                            float4* d = reinterpret_cast<float4*>(dm + (size_t) (row0 + r) * 4096 + rbase + 32 * wm + 16 * mt + 8 * hi);
                            d[0] = make_float4(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3]);
                            d[1] = make_float4(acc[mt][nt][4], acc[mt][nt][5], acc[mt][nt][6], acc[mt][nt][7]);
#else
                            float* d = dm + (size_t) (row0 + r) * 4096 + rbase + 32 * wm + 16 * mt + hi;
#pragma unroll
                            for (int i = 0; i < 8; ++i) d[2 * i] = acc[mt][nt][i];
#endif
                        }
                    }
            }
        }
    }
#else
    __builtin_trap();
#endif
}
#endif  // __HIPCC__


} // namespace

namespace {
struct DeviceInfo { bool ready = false, ok = false; int sms = 0; };
DeviceInfo device_info() {
    static DeviceInfo devices[32];
    static std::mutex lock;
    int dev = 0; ck(cudaGetDevice(&dev), "device");
    std::lock_guard<std::mutex> guard(lock);
    if (dev < 0 || dev >= 32) return {};
    auto& d = devices[dev];
    if (d.ready) return d;
    d.ready = true;
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, dev), "properties");
    // Other compiled architectures retain MMQ until measured and validated separately.
    if (std::strncmp(prop.gcnArchName, "gfx1151", 7)) return d;
    hipFuncAttributes fa{};
    if (hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(native_kernel<T_IQ3_XXS, true, true, NW_LB>)) != hipSuccess) {
        (void) cudaGetLastError(); return d;
    }
    d.ok = fa.sharedSizeBytes >= (size_t) 2 * NW_ROWS * WLD;
    ck(cudaDeviceGetAttribute(&d.sms, hipDeviceAttributeMultiprocessorCount, dev), "sms");
    return d;
}
} // namespace

bool available() {
    static const bool on = [] {
        const char* off = std::getenv("STRATA_GLM_PREFILL_FUSED");
        return !(off && off[0] == '0');
    }();
    if (!on) return false;
    return device_info().ok;
}

bool supported(int gu, int down) {
    if (!(gu == T_IQ2_XXS || gu == T_IQ2_XS || gu == T_IQ2_S || gu == T_IQ3_XXS || gu == T_IQ3_S || gu == T_IQ4_XS) ||
        !(down == T_Q2_0 || down == T_IQ4_NL || down == T_IQ2_S || down == T_IQ3_XXS)) return false;
    return available();
}

size_t act_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) (cols / 64) * AB; }

void quantize(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int64_t nblk = rows * (cols / 64);
    quant_act_nat_kernel<<<(unsigned)((nblk + 7) / 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize");
}

void experts(const uint8_t* wbase, size_t stride, const Geometry& geo, int ne,
             const int* bounds, const int* host_bounds, void* tile_scratch,
             const void* xa, const int* src, void* ha, float* out, void* stream) {
    if (ne <= 0) return;
    if (ne > 512 || ((uintptr_t) out & 15)) std::abort();
    const auto d = device_info();
    if (!d.ok) std::abort();
    const int sms = d.sms;
    int ntiles = 0;
    for (int e = 0; e < ne; ++e) ntiles += (host_bounds[e+1] - host_bounds[e] + 63) / 64;
    if (!ntiles) return;
    const int gu_blocks = geo.gu_type == T_IQ2_S || geo.gu_type == T_IQ2_XS ? 3 : 4;
    const unsigned grid_gu = (unsigned) std::min(ntiles * 32, sms * gu_blocks);
    const unsigned grid_dn = (unsigned) std::min(ntiles * 32, sms * (geo.d_type == T_IQ2_S ? 3 : 4));
    const auto s = (cudaStream_t) stream;
    auto* tiles = (int2*) tile_scratch;
    make_tiles_kernel<<<1, 256, 0, s>>>(bounds, ne, tiles);
#define GU(T) native_kernel<T, true, true, NW_LB><<<grid_gu, 256, 0, s>>>(wbase, stride, ntiles, tiles, bounds, geo, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr)
    switch (geo.gu_type) {
        case T_IQ2_XXS: GU(T_IQ2_XXS); break;
        case T_IQ2_XS: GU(T_IQ2_XS); break;
        case T_IQ2_S: GU(T_IQ2_S); break;
        case T_IQ3_XXS: GU(T_IQ3_XXS); break;
        case T_IQ3_S: GU(T_IQ3_S); break;
        case T_IQ4_XS: GU(T_IQ4_XS); break;
        default: std::abort();
    }
#undef GU
#define DN(T) native_kernel<T, false, true, NW_LB><<<grid_dn, 256, 0, s>>>(wbase, stride, ntiles, tiles, bounds, geo, (const uint8_t*) ha, src, nullptr, out)
    if (geo.d_type == T_Q2_0) { DN(T_Q2_0); }
    else if (geo.d_type == T_IQ4_NL) { DN(T_IQ4_NL); }
    else if (geo.d_type == T_IQ2_S) { DN(T_IQ2_S); }
    else if (geo.d_type == T_IQ3_XXS) { DN(T_IQ3_XXS); }
    else std::abort();
#undef DN
    ck(cudaGetLastError(), "experts");
}
} // namespace strata::prefill::rdna4
