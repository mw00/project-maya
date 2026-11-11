// src/prefill/moe_pascal.cu - see include/strata/prefill/moe_pascal.hpp.
// A block takes a 128 (weight rows) x 64 (tokens) tile of one expert's product.  Per 32 values of K every thread
// decodes 16 weights of one row into a shared FP32 tile, the block stages the 64 x 32 activations next to it, and
// each thread then runs 8 x 4 FFMAs per k.  The block-format decoders follow ggml's dequantize.cuh (MIT) and the
// codebooks are ggml-common.h's.
#include "strata/prefill/moe_pascal.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill::pascal {
namespace {

constexpr int BM = 128, BN = 64, BK = 16, NTHR = 256;

__device__ __forceinline__ float h2f(uint16_t v) { return __half2float(__ushort_as_half(v)); }

// byte J of w as an exact float via 2^23 + b: one PRMT; the caller subtracts 2^23 inside its FFMA
template <int J>
__device__ __forceinline__ float ubyte_biased(uint32_t w) {
    return __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440 | J));
}

template <int J>
__device__ __forceinline__ float wval(uint32_t grid, uint32_t sg, int sj, float dl, float c) {
    const float v = fmaf(ubyte_biased<J>(grid), dl, -c);   // dl * byte, exactly rounded once
    const uint32_t flip = (sg << (31 - (sj + J))) & 0x80000000u;
    return __uint_as_float(__float_as_uint(v) ^ flip);
}

// 8 values: grid bytes (lo: values 0-3, hi: 4-7), sign bits 0-7 of sg (bit set = negative)
__device__ __forceinline__ void put8(float* o, uint32_t lo, uint32_t hi, uint32_t sg, float dl, float c) {
    o[0] = wval<0>(lo, sg, 0, dl, c);
    o[1] = wval<1>(lo, sg, 0, dl, c);
    o[2] = wval<2>(lo, sg, 0, dl, c);
    o[3] = wval<3>(lo, sg, 0, dl, c);
    o[4] = wval<0>(hi, sg, 4, dl, c);
    o[5] = wval<1>(hi, sg, 4, dl, c);
    o[6] = wval<2>(hi, sg, 4, dl, c);
    o[7] = wval<3>(hi, sg, 4, dl, c);
}

// the FP16 form: 4 half2 of consecutive value pairs.  A grid byte b is (1024 + b) - 1024 in half (0x6400 | b), and
// one HFMA2 with (dl, dl) and -1024 dl gives dl * b rounded once; the sign bits are xor-ed into the halves' bit 15.
__device__ __forceinline__ __half2 pair_h(uint32_t w, int j, uint32_t sg, int sj, __half2 d2, __half2 c2) {
    const uint32_t sel = 0x4040u | ((uint32_t) (j + 1) << 8) | (uint32_t) j;
    uint32_t hv = __byte_perm(w, 0x64646464u, sel);
    const int b = sj + j;
    const uint32_t flip = ((sg << (15 - b)) & 0x00008000u) | ((sg << (30 - b)) & 0x80000000u);
    const __half2 v = __hfma2(*reinterpret_cast<__half2*>(&hv), d2, __hneg2(c2));
    uint32_t vu = *reinterpret_cast<const uint32_t*>(&v) ^ flip;
    return *reinterpret_cast<__half2*>(&vu);
}
__device__ __forceinline__ void put8h(__half2* o, uint32_t lo, uint32_t hi, uint32_t sg, float dl) {
    const __half2 d2 = __float2half2_rn(dl);
    const __half2 c2 = __hmul2(d2, __float2half2_rn(1024.f));
    o[0] = pair_h(lo, 0, sg, 0, d2, c2);
    o[1] = pair_h(lo, 2, sg, 0, d2, c2);
    o[2] = pair_h(hi, 0, sg, 4, d2, c2);
    o[3] = pair_h(hi, 2, sg, 4, d2, c2);
}

template <int T> struct Fmt;

// Each format loads one 32-value record (a "sub-block") per thread and decodes 8-value groups from it: group
// l = 2 * B + hh, B the half of the record (a 16-value step of K), hh the thread's half of the block.

// {half d; uint16 qs[32]}: per 32 values 4 grid indices (bytes) then 4 x 7 sign bits and a 4-bit scale
template <> struct Fmt<GGML_TYPE_IQ2_XXS> {
    static constexpr int BYTES = sizeof(block_iq2_xxs);
    struct Raw { uint32_t idx4, aux; float d; };
    __device__ static Raw load(const uint8_t* blk, int sub) {
        const uint16_t* q = (const uint16_t*) (blk + 2) + 4 * sub;
        Raw r;
        r.idx4 = (uint32_t) __ldg(q) | ((uint32_t) __ldg(q + 1) << 16);
        r.aux = (uint32_t) __ldg(q + 2) | ((uint32_t) __ldg(q + 3) << 16);
        r.d = h2f(__ldg((const uint16_t*) blk));
        return r;
    }
    template <int B> __device__ static void decode(const Raw& r, int hh, float* o) {
        const int l = 2 * B + hh;
        const float dl = r.d * (0.5f + (float) (r.aux >> 28)) * 0.25f, c = dl * 8388608.0f;
        const uint2 grid = ((const uint2*) iq2xxs_grid)[(r.idx4 >> (8 * l)) & 0xff];
        put8(o, grid.x, grid.y, ksigns_iq2xs[(r.aux >> (7 * l)) & 127], dl, c);
    }
    template <int B> __device__ static void decode_h(const Raw& r, int hh, __half2* o) {
        const int l = 2 * B + hh;
        const float dl = r.d * (0.5f + (float) (r.aux >> 28)) * 0.25f;
        const uint2 grid = ((const uint2*) iq2xxs_grid)[(r.idx4 >> (8 * l)) & 0xff];
        put8h(o, grid.x, grid.y, ksigns_iq2xs[(r.aux >> (7 * l)) & 127], dl);
    }
};

// {half d; uint8 qs[64] (32 grid bytes, 32 sign bytes); uint8 qh[8]; uint8 scales[8]}
template <> struct Fmt<GGML_TYPE_IQ2_S> {
    static constexpr int BYTES = sizeof(block_iq2_s);
    struct Raw { uint32_t idx4, sgn4, qh, sc; float d; };
    __device__ static Raw load(const uint8_t* blk, int sub) {
        const uint16_t* a = (const uint16_t*) (blk + 2 + 4 * sub);
        const uint16_t* b = (const uint16_t*) (blk + 2 + 32 + 4 * sub);
        Raw r;
        r.idx4 = (uint32_t) __ldg(a) | ((uint32_t) __ldg(a + 1) << 16);
        r.sgn4 = (uint32_t) __ldg(b) | ((uint32_t) __ldg(b + 1) << 16);
        r.qh = __ldg(blk + 66 + sub);
        r.sc = __ldg(blk + 74 + sub);
        r.d = h2f(__ldg((const uint16_t*) blk));
        return r;
    }
    template <int B> __device__ static void decode(const Raw& r, int hh, float* o) {
        const int l = 2 * B + hh;
        const float dl = r.d * (0.5f + (float) ((r.sc >> (4 * B)) & 0xf)) * 0.25f, c = dl * 8388608.0f;
        const uint2 grid = ((const uint2*) iq2s_grid)[((r.idx4 >> (8 * l)) & 0xff) | ((r.qh << (8 - 2 * l)) & 0x300)];
        put8(o, grid.x, grid.y, (r.sgn4 >> (8 * l)) & 0xff, dl, c);
    }
    template <int B> __device__ static void decode_h(const Raw& r, int hh, __half2* o) {
        const int l = 2 * B + hh;
        const float dl = r.d * (0.5f + (float) ((r.sc >> (4 * B)) & 0xf)) * 0.25f;
        const uint2 grid = ((const uint2*) iq2s_grid)[((r.idx4 >> (8 * l)) & 0xff) | ((r.qh << (8 - 2 * l)) & 0x300)];
        put8h(o, grid.x, grid.y, (r.sgn4 >> (8 * l)) & 0xff, dl);
    }
};

// {half d; uint8 qs[96]}: 64 grid bytes (4 values each), then per 32 values a uint32 of 4 x 7 sign bits and a scale
template <> struct Fmt<GGML_TYPE_IQ3_XXS> {
    static constexpr int BYTES = sizeof(block_iq3_xxs);
    struct Raw { uint32_t lo, hi, aux; float d; };
    __device__ static Raw load(const uint8_t* blk, int sub) {
        const uint16_t* q = (const uint16_t*) (blk + 2 + 8 * sub);
        const uint16_t* a = (const uint16_t*) (blk + 2 + 64 + 4 * sub);
        Raw r;
        r.lo = (uint32_t) __ldg(q) | ((uint32_t) __ldg(q + 1) << 16);
        r.hi = (uint32_t) __ldg(q + 2) | ((uint32_t) __ldg(q + 3) << 16);
        r.aux = (uint32_t) __ldg(a) | ((uint32_t) __ldg(a + 1) << 16);
        r.d = h2f(__ldg((const uint16_t*) blk));
        return r;
    }
    template <int B> __device__ static void decode(const Raw& r, int hh, float* o) {
        const int l = 2 * B + hh;
        const float dl = r.d * (0.5f + (float) (r.aux >> 28)) * 0.5f, c = dl * 8388608.0f;
        const uint32_t w = (B == 0 ? r.lo : r.hi) >> (16 * hh);
        put8(o, iq3xxs_grid[w & 0xff], iq3xxs_grid[(w >> 8) & 0xff], ksigns_iq2xs[(r.aux >> (7 * l)) & 127], dl, c);
    }
    template <int B> __device__ static void decode_h(const Raw& r, int hh, __half2* o) {
        const int l = 2 * B + hh;
        const float dl = r.d * (0.5f + (float) (r.aux >> 28)) * 0.5f;
        const uint32_t w = (B == 0 ? r.lo : r.hi) >> (16 * hh);
        put8h(o, iq3xxs_grid[w & 0xff], iq3xxs_grid[(w >> 8) & 0xff], ksigns_iq2xs[(r.aux >> (7 * l)) & 127], dl);
    }
};

__device__ __forceinline__ float4 ldg4(const float* p) { return __ldg((const float4*) p); }

// 16 values of K: each thread's 8 x 4 outputs from the stage's weight tile (rows tx*4.. and 64 + tx*4..) and
// activation tile (tokens ty*4..; the token index is swizzled by the k chunk, see the store)
template <int BUF>
__device__ __forceinline__ void fma_stage(const float (*Ws)[BK][BM], const float (*Xs)[BK][BN], int tx, int ty,
                                          float (&acc)[8][4]) {
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
        const float4 wa = *(const float4*) &Ws[BUF][kk][tx * 4];
        const float4 wb = *(const float4*) &Ws[BUF][kk][BM / 2 + tx * 4];
        const float4 xv = *(const float4*) &Xs[BUF][kk][(ty * 4) ^ ((kk >> 2) << 3)];
        const float a[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
        const float b[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
}

template <int T>
__global__ void __launch_bounds__(NTHR, 2)
    expert_gemm(const uint8_t* __restrict__ w, size_t expert_bytes, int K, const float* __restrict__ x, int64_t ld_x,
                const int32_t* __restrict__ row_map, const int32_t* __restrict__ bounds, float* __restrict__ dst,
                int64_t ld_dst) {
    using F = Fmt<T>;
    const int e = blockIdx.z, mt = blockIdx.x;
    const int r0 = bounds[e], r1 = bounds[e + 1];
    const int t0 = r0 + (int) blockIdx.y * BN;
    if (t0 >= r1) return;
    const int ntok = min(BN, r1 - t0);

    // two stages of 16 values of K: the weights (FP32, row-minor) and the activations (token-minor)
    __shared__ __align__(16) float Ws[2][BK][BM];
    __shared__ __align__(16) float Xs[2][BK][BN];

    const int tid = threadIdx.x;
    const int wrow = tid & (BM - 1), hh = tid >> 7;   // the weight row, and which 8 of the stage's 16 values
    const uint8_t* wp = w + (size_t) e * expert_bytes + (size_t) (mt * BM + wrow) * (size_t) (K / 256) * F::BYTES;
    const int xr = tid >> 2, xc = tid & 3;            // activation row and 4-value chunk of the stage
    const int tok = t0 + min(xr, ntok - 1);
    const float* xp = x + (int64_t) (row_map ? row_map[tok] : tok) * ld_x + 4 * xc;
    const int xs = xr ^ (xc << 3);                    // (the 4 chunks of a row land in 4 different bank groups)
    const int tx = tid & 15, ty = tid >> 4;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    const auto put_x = [&](int buf, const float4& v) {
        Xs[buf][4 * xc + 0][xs] = v.x;
        Xs[buf][4 * xc + 1][xs] = v.y;
        Xs[buf][4 * xc + 2][xs] = v.z;
        Xs[buf][4 * xc + 3][xs] = v.w;
    };

    const int nsb = K / 32;   // 32-value records
    typename F::Raw rawC = F::load(wp, 0);
    {
        float wv[8];
        F::template decode<0>(rawC, hh, wv);
#pragma unroll
        for (int j = 0; j < 8; ++j) Ws[0][8 * hh + j][wrow] = wv[j];
        put_x(0, ldg4(xp));
    }
    float4 xa = ldg4(xp + 16);   // the stage after
    __syncthreads();
    for (int sb = 0; sb < nsb; ++sb) {
        const bool more = sb + 1 < nsb;
        typename F::Raw rawN;
        float4 xb;
        {   // stage 2 sb (buffer 0) is computed while stage 2 sb + 1 (buffer 1) is filled
            float wv[8];
            F::template decode<1>(rawC, hh, wv);
#pragma unroll
            for (int j = 0; j < 8; ++j) Ws[1][8 * hh + j][wrow] = wv[j];
            put_x(1, xa);
            if (more) {
                const int k1 = sb + 1;
                rawN = F::load(wp + (size_t) (k1 >> 3) * F::BYTES, k1 & 7);
                xb = ldg4(xp + (2 * sb + 2) * 16);
            }
            fma_stage<0>(Ws, Xs, tx, ty, acc);
            __syncthreads();
        }
        {   // stage 2 sb + 1 (buffer 1) is computed while the next record's first stage (buffer 0) is filled
            if (more) {
                float wv[8];
                F::template decode<0>(rawN, hh, wv);
#pragma unroll
                for (int j = 0; j < 8; ++j) Ws[0][8 * hh + j][wrow] = wv[j];
                put_x(0, xb);
                xa = ldg4(xp + (2 * sb + 3) * 16);
                rawC = rawN;
            }
            fma_stage<1>(Ws, Xs, tx, ty, acc);
            __syncthreads();
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = ty * 4 + j;
        if (t >= ntok) continue;
        float* d = dst + (int64_t) (t0 + t) * ld_dst + mt * BM;
        *(float4*) (d + tx * 4) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
        *(float4*) (d + BM / 2 + tx * 4) = make_float4(acc[4][j], acc[5][j], acc[6][j], acc[7][j]);
    }
}


// ---- FP16 form: HFMA2 runs at twice the FFMA rate.  The k axis is packed in pairs: a thread's weight row holds
// half2 (w[k], w[k+1]), a token's activations half2 (x[k], x[k+1]), and one HFMA2 adds both products to a half2 whose
// lanes are summed when it is flushed to the FP32 accumulators (every 256 values of K - a block - so a half chain is
// 128 products).  The weights are the dequantized values rounded to half (a scale error of 2^-12 a 32-value group),
// the activations are rounded to half (clamped to its range).  A tile with few rows (most experts of a chunk have
// under 32) lets the warps whose 8 tokens lie past its last row skip the FMAs, and the threads of a missing token skip
// its activation loads: the weights are decoded and staged by every warp either way.
__device__ __forceinline__ __half2 h2_from(const uint32_t u) { return *reinterpret_cast<const __half2*>(&u); }

constexpr int WPAD = BM + 4;   // a stage row of the FP16 weight tile, padded so a row's two decode threads store apart

template <int BUF>
__device__ __forceinline__ void fma_stage_h(const uint4 (*Ws)[BK / 2][WPAD / 4], const uint4 (*Xs)[BK / 2][BN / 4],
                                            int tx, int ty, __half2 (&acc)[8][4]) {
#pragma unroll
    for (int kp = 0; kp < BK / 2; ++kp) {
        const uint4 wa = Ws[BUF][kp][tx];
        const uint4 wb = Ws[BUF][kp][BM / 8 + tx];
        const uint4 xv = Xs[BUF][kp][ty ^ ((kp >> 1) << 1)];
        const uint32_t wu[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
        const uint32_t xu[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = __hfma2(h2_from(wu[i]), h2_from(xu[j]), acc[i][j]);
    }
}

__device__ __forceinline__ uint32_t pack_h(float a, float b) {
    const __half2 v = __floats2half2_rn(fminf(fmaxf(a, -65504.f), 65504.f), fminf(fmaxf(b, -65504.f), 65504.f));
    return *reinterpret_cast<const uint32_t*>(&v);
}

template <int T>
__global__ void __launch_bounds__(NTHR, 2)
    expert_gemm_h(const uint8_t* __restrict__ w, size_t expert_bytes, int K, const float* __restrict__ x, int64_t ld_x,
                  const int32_t* __restrict__ row_map, const int32_t* __restrict__ bounds, float* __restrict__ dst,
                  int64_t ld_dst) {
    using F = Fmt<T>;
    const int e = blockIdx.z, mt = blockIdx.x;
    const int r0 = bounds[e], r1 = bounds[e + 1];
    const int t0 = r0 + (int) blockIdx.y * BN;
    if (t0 >= r1) return;
    const int ntok = min(BN, r1 - t0);

    // Ws[buf][kp][row]: half2 (k = 2 kp, 2 kp + 1) of a weight row; Xs[buf][kp][token]: the same of a token.  Both
    // are read as uint4 (4 rows / 4 tokens); the token index is swizzled by the k chunk as in the FP32 kernel.
    __shared__ __align__(16) uint32_t Ws[2][BK / 2][WPAD];
    __shared__ __align__(16) uint32_t Xs[2][BK / 2][BN];

    const int tid = threadIdx.x;
    const int wrow = tid >> 1, hh = tid & 1;   // a row's two threads side by side: one load transaction serves both
    const uint8_t* wp = w + (size_t) e * expert_bytes + (size_t) (mt * BM + wrow) * (size_t) (K / 256) * F::BYTES;
    const int xr = tid >> 2, xc = tid & 3;
    const int tok = t0 + min(xr, ntok - 1);
    const float* xp = x + (int64_t) (row_map ? row_map[tok] : tok) * ld_x + 4 * xc;
    const int xs = xr ^ (xc << 3);
    const int tx = tid & 15, ty = tid >> 4;
    const bool xok = xr < ntok;                    // this thread's token exists
    const bool active = (tid >> 5) * 8 < ntok;     // a warp's 2 x 16 threads hold 8 tokens

    float acc32[8][4];
    __half2 acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc32[i][j] = 0.f;
            acc[i][j] = __float2half2_rn(0.f);
        }

    const float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);
    const auto put_x = [&](int buf, const float4& v) {
        Xs[buf][2 * xc][xs] = pack_h(v.x, v.y);
        Xs[buf][2 * xc + 1][xs] = pack_h(v.z, v.w);
    };
    const auto put_w = [&](int buf, const __half2 (&wv)[4]) {
#pragma unroll
        for (int j = 0; j < 4; ++j) Ws[buf][4 * hh + j][wrow] = *reinterpret_cast<const uint32_t*>(&wv[j]);
    };
    const auto flush = [&]() {
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc32[i][j] += __low2float(acc[i][j]) + __high2float(acc[i][j]);
                acc[i][j] = __float2half2_rn(0.f);
            }
    };

    const auto Wv = reinterpret_cast<const uint4 (*)[BK / 2][WPAD / 4]>(&Ws[0][0][0]);
    const auto Xv = reinterpret_cast<const uint4 (*)[BK / 2][BN / 4]>(&Xs[0][0][0]);

    const int nsb = K / 32;
    typename F::Raw rawC = F::load(wp, 0);
    {
        __half2 wv[4];
        F::template decode_h<0>(rawC, hh, wv);
        put_w(0, wv);
        put_x(0, xok ? ldg4(xp) : zero4);
    }
    float4 xa = xok ? ldg4(xp + 16) : zero4;
    __syncthreads();
    for (int sb = 0; sb < nsb; ++sb) {
        const bool more = sb + 1 < nsb;
        typename F::Raw rawN;
        float4 xb;
        {
            __half2 wv[4];
            F::template decode_h<1>(rawC, hh, wv);
            put_w(1, wv);
            put_x(1, xa);
            if (more) {
                const int k1 = sb + 1;
                rawN = F::load(wp + (size_t) (k1 >> 3) * F::BYTES, k1 & 7);
                xb = xok ? ldg4(xp + (2 * sb + 2) * 16) : zero4;
            }
            if (active) fma_stage_h<0>(Wv, Xv, tx, ty, acc);
            __syncthreads();
        }
        {
            if (more) {
                __half2 wv[4];
                F::template decode_h<0>(rawN, hh, wv);
                put_w(0, wv);
                put_x(0, xb);
                xa = xok ? ldg4(xp + (2 * sb + 3) * 16) : zero4;
                rawC = rawN;
            }
            if (active) fma_stage_h<1>(Wv, Xv, tx, ty, acc);
            __syncthreads();
        }
        if (active && (sb & 7) == 7) flush();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = ty * 4 + j;
        if (t >= ntok) continue;
        float* d = dst + (int64_t) (t0 + t) * ld_dst + mt * BM;
        *(float4*) (d + tx * 4) = make_float4(acc32[0][j], acc32[1][j], acc32[2][j], acc32[3][j]);
        *(float4*) (d + BM / 2 + tx * 4) = make_float4(acc32[4][j], acc32[5][j], acc32[6][j], acc32[7][j]);
    }
}

// ---- the dense FP16 product: the FP16 expert kernel's tile and FMA stage with the weights read as they are (a row's
// two threads load 32 contiguous bytes of its next 32 values; the activations are half already)
// SPLIT: block z takes sub-blocks [z * piece_sb, (z + 1) * piece_sb) of K (piece_sb % 8 == 0: the FP32 flushes fall as in
// the whole product) and adds its sums into a Y that is zero
template <bool SPLIT>
__global__ void __launch_bounds__(NTHR, 2)
    dense_gemm_h(const __half* __restrict__ W, int N, int K, const __half* __restrict__ X, int64_t ldx,
                 float* __restrict__ Y, int64_t ldy, int T, int piece_sb) {
    const int sb0 = SPLIT ? blockIdx.z * piece_sb : 0;
    const int nsb = SPLIT ? min(K / 32, sb0 + piece_sb) - sb0 : K / 32;
    if (SPLIT && nsb <= 0) return;
    const int mt = blockIdx.x;
    const int t0 = blockIdx.y * BN;
    const int ntok = min(BN, T - t0);
    __shared__ __align__(16) uint32_t Ws[2][BK / 2][WPAD];
    __shared__ __align__(16) uint32_t Xs[2][BK / 2][BN];
    const int tid = threadIdx.x;
    const int wrow = tid >> 1, hh = tid & 1;
    const int nrow = min(mt * BM + wrow, N - 1);
    const __half* wp = W + (size_t) nrow * K + sb0 * 32 + 8 * hh;
    const int xr = tid >> 2, xc = tid & 3;
    const bool xok = xr < ntok;
    const __half* xp = X + (int64_t) (t0 + min(xr, ntok - 1)) * ldx + sb0 * 32 + 4 * xc;
    const int xs = xr ^ (xc << 3);
    const int tx = tid & 15, ty = tid >> 4;
    const bool active = (tid >> 5) * 8 < ntok;
    float acc32[8][4];
    __half2 acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc32[i][j] = 0.f;
            acc[i][j] = __float2half2_rn(0.f);
        }
    const uint2 zero2 = make_uint2(0u, 0u);
    const auto put_x = [&](int buf, const uint2& v) {
        Xs[buf][2 * xc][xs] = v.x;
        Xs[buf][2 * xc + 1][xs] = v.y;
    };
    const auto put_w = [&](int buf, const uint4& v) {
        Ws[buf][4 * hh][wrow] = v.x;
        Ws[buf][4 * hh + 1][wrow] = v.y;
        Ws[buf][4 * hh + 2][wrow] = v.z;
        Ws[buf][4 * hh + 3][wrow] = v.w;
    };
    const auto ldw = [&](int k) { return __ldg((const uint4*) (wp + k)); };
    const auto ldx2 = [&](int k) { return xok ? __ldg((const uint2*) (xp + k)) : zero2; };
    const auto flush = [&]() {
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc32[i][j] += __low2float(acc[i][j]) + __high2float(acc[i][j]);
                acc[i][j] = __float2half2_rn(0.f);
            }
    };
    const auto Wv = reinterpret_cast<const uint4 (*)[BK / 2][WPAD / 4]>(&Ws[0][0][0]);
    const auto Xv = reinterpret_cast<const uint4 (*)[BK / 2][BN / 4]>(&Xs[0][0][0]);
    uint4 wc1;
    put_w(0, ldw(0));
    wc1 = ldw(16);
    put_x(0, ldx2(0));
    uint2 xa = ldx2(16);
    __syncthreads();
    for (int sb = 0; sb < nsb; ++sb) {
        const bool more = sb + 1 < nsb;
        put_w(1, wc1);
        put_x(1, xa);
        uint4 wn0 = make_uint4(0, 0, 0, 0), wn1 = wn0;
        uint2 xb = zero2;
        if (more) {
            wn0 = ldw((sb + 1) * 32);
            wn1 = ldw((sb + 1) * 32 + 16);
            xb = ldx2((sb + 1) * 32);
        }
        if (active) fma_stage_h<0>(Wv, Xv, tx, ty, acc);
        __syncthreads();
        if (more) {
            put_w(0, wn0);
            put_x(0, xb);
            xa = ldx2((sb + 1) * 32 + 16);
            wc1 = wn1;
        }
        if (active) fma_stage_h<1>(Wv, Xv, tx, ty, acc);
        __syncthreads();
        if (active && (sb & 7) == 7) flush();
    }
    if (active && (nsb & 7) != 0) flush();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = ty * 4 + j;
        if (t >= ntok) continue;
        float* d = Y + (int64_t) (t0 + t) * ldy + mt * BM;
        const int c0 = mt * BM + tx * 4, c1 = mt * BM + BM / 2 + tx * 4;
        if (SPLIT) {
            // two pieces: 0 + a is exact and a + b == b + a, so the sum does not depend on the order they land in
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (c0 + 3 < N) atomicAdd(d + tx * 4 + i, acc32[i][j]);
                if (c1 + 3 < N) atomicAdd(d + BM / 2 + tx * 4 + i, acc32[4 + i][j]);
            }
        } else {
            if (c0 + 3 < N) *(float4*) (d + tx * 4) = make_float4(acc32[0][j], acc32[1][j], acc32[2][j], acc32[3][j]);
            if (c1 + 3 < N) *(float4*) (d + BM / 2 + tx * 4) = make_float4(acc32[4][j], acc32[5][j], acc32[6][j], acc32[7][j]);
        }
    }
}

// K pieces for a dense product by its tile grid (0: cuBLAS keeps it).  A P100 holds 112 blocks (56 SMs x 2).  Measured
// for K 512..8192, cuBLAS 3.0-3.5 TMAC/s: two pieces win under 81 tiles (about one block an SM) and from 113 to 139 (a
// short second wave); one piece wins from 81 to 112 (one wave) and from 140.  Under 20 tiles cuBLAS stays ahead.
int grid_pieces(const Dense& d) {
    const int64_t tiles = ((d.n + BM - 1) / BM) * ((d.t + BN - 1) / BN);
    if (tiles < 20) return 0;
    if (d.k >= 1024 && (tiles <= 80 || (tiles >= 113 && tiles <= 139))) return 2;   // (a shorter K gains nothing)
    return tiles >= 30 ? 1 : 0;
}

}  // namespace

bool supported(int t) { return t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ3_XXS; }

bool usable(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    if (!supported(gu_type) || !supported(d_type) || n_embd % 256 != 0 || n_ff % 256 != 0 || (2 * n_ff) % BM != 0 ||
        n_embd % BM != 0)
        return false;
    const char* v = std::getenv("STRATA_GLM_PREFILL_PASCAL");
    if (v != nullptr) return std::atoi(v) != 0;
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major * 10 + minor < 61;
}

bool f16_default() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_GLM_PREFILL_PASCAL_FP16");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return v;
}

void run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    if (!supported(p.type) || p.w_rows % BM != 0 || p.w_cols % 256 != 0) {
        std::fprintf(stderr, "prefill pascal: type %d, %ld x %ld is not covered\n", p.type, (long) p.w_rows,
                     (long) p.w_cols);
        std::exit(1);
    }
    const dim3 grid((unsigned) (p.w_rows / BM), (unsigned) ((p.max_rows + BN - 1) / BN), (unsigned) p.n);
    const cudaStream_t s = (cudaStream_t) stream;
    const uint8_t* w = (const uint8_t*) p.w;
    if (p.f16) {
        switch (p.type) {
            case GGML_TYPE_IQ2_XXS:
                expert_gemm_h<GGML_TYPE_IQ2_XXS><<<grid, NTHR, 0, s>>>(w, p.expert_bytes, (int) p.w_cols, p.x, p.ld_x,
                                                                      p.row_map, p.bounds, p.dst, p.ld_dst);
                break;
            case GGML_TYPE_IQ2_S:
                expert_gemm_h<GGML_TYPE_IQ2_S><<<grid, NTHR, 0, s>>>(w, p.expert_bytes, (int) p.w_cols, p.x, p.ld_x,
                                                                    p.row_map, p.bounds, p.dst, p.ld_dst);
                break;
            default:
                expert_gemm_h<GGML_TYPE_IQ3_XXS><<<grid, NTHR, 0, s>>>(w, p.expert_bytes, (int) p.w_cols, p.x, p.ld_x,
                                                                      p.row_map, p.bounds, p.dst, p.ld_dst);
                break;
        }
        const cudaError_t eh = cudaGetLastError();
        if (eh != cudaSuccess) {
            std::fprintf(stderr, "prefill pascal: %s\n", cudaGetErrorString(eh));
            std::exit(1);
        }
        return;
    }
    switch (p.type) {
        case GGML_TYPE_IQ2_XXS:
            expert_gemm<GGML_TYPE_IQ2_XXS><<<grid, NTHR, 0, s>>>(w, p.expert_bytes, (int) p.w_cols, p.x, p.ld_x,
                                                                 p.row_map, p.bounds, p.dst, p.ld_dst);
            break;
        case GGML_TYPE_IQ2_S:
            expert_gemm<GGML_TYPE_IQ2_S><<<grid, NTHR, 0, s>>>(w, p.expert_bytes, (int) p.w_cols, p.x, p.ld_x,
                                                               p.row_map, p.bounds, p.dst, p.ld_dst);
            break;
        default:
            expert_gemm<GGML_TYPE_IQ3_XXS><<<grid, NTHR, 0, s>>>(w, p.expert_bytes, (int) p.w_cols, p.x, p.ld_x,
                                                                 p.row_map, p.bounds, p.dst, p.ld_dst);
            break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill pascal: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool dense_usable(const Dense& d) {
    if (d.n <= 0 || d.t <= 0 || d.k % 32 != 0 || d.n % 4 != 0 || d.ld_x % 4 != 0 || d.ld_y % 4 != 0 || d.k > (1 << 20))
        return false;
    if (((uintptr_t) d.w & 15) != 0 || ((uintptr_t) d.x & 7) != 0 || ((uintptr_t) d.y & 15) != 0) return false;
    static const int env = std::getenv("STRATA_GLM_PREFILL_PASCAL_DENSE") ? std::atoi(std::getenv("STRATA_GLM_PREFILL_PASCAL_DENSE")) : -1;
    if (env == 0) return false;
    if (env < 0) {
        int dev = 0, major = 0, minor = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        if (major * 10 + minor >= 61) return false;
    }
    return grid_pieces(d) > 0;
}

void dense_run(const Dense& d, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    const int groups = ((int) (d.k / 32) + 7) / 8;   // (flush groups of 256 values of K)
    const int pieces = std::min(d.pieces > 0 ? std::min(d.pieces, 2) : std::max(grid_pieces(d), 1), groups);
    const dim3 grid((unsigned) ((d.n + BM - 1) / BM), (unsigned) ((d.t + BN - 1) / BN), (unsigned) pieces);
    if (pieces > 1) {
        // the two pieces add into y: zero its [t][n] slice first
        const cudaError_t em =
            cudaMemset2DAsync(d.y, (size_t) d.ld_y * sizeof(float), 0, (size_t) d.n * sizeof(float), (size_t) d.t, s);
        if (em != cudaSuccess) {
            std::fprintf(stderr, "prefill pascal dense: %s\n", cudaGetErrorString(em));
            std::exit(1);
        }
        dense_gemm_h<true><<<grid, NTHR, 0, s>>>((const __half*) d.w, (int) d.n, (int) d.k, (const __half*) d.x, d.ld_x, d.y,
                                                 d.ld_y, (int) d.t, 8 * ((groups + pieces - 1) / pieces));
    } else {
        dense_gemm_h<false><<<grid, NTHR, 0, s>>>((const __half*) d.w, (int) d.n, (int) d.k, (const __half*) d.x, d.ld_x, d.y,
                                                  d.ld_y, (int) d.t, 0);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill pascal dense: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::prefill::pascal
