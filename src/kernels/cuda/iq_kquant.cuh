// src/kernels/cuda/iq_kquant.cuh - glm-dsa B3.5 / B5.3: the K-quant formats (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) for
// iq_kernels.cu's native_expert_grouped, the slow path's (MAYA_GLM_SLOW) routed experts.  GLM-5.3 GGUFs keep
// experts in K-quants, and so do the PACK / PACK9 presets and tools/iq_fixture.py's 12-type fixture; the grouped
// kernel knew only Maya's i-quant set and stopped the process on Q2_K (the test PC's first GPU run, 2026-10-17).
//
// The dots are glm_fast.cu's dot_q2_K .. dot_q6_K, copied unchanged: llama.cpp's vec_dot_q{2,3,4,5,6}_K_q8_1 at the
// MMVQ VDR (1, 1, 2, 2, 1; ggml/src/ggml-cuda/vecdotq.cuh at third_party/ggml/VERSION.txt, MIT), the arithmetic the
// fast path's gf::mv runs on these types (glmd_mv_parity holds it to a double reference on the V100s).  `bq8_1` is
// the super-block's first q8_1 block (row_dot passes x + kbx * 8), and iqs = step * part, as in llama.cpp's mmvq.
//
// NO #include here: iq_kernels.cu includes this file INSIDE its anonymous namespace, after its llama.cpp helpers
// (get_int_b2, get_int_b4) and the Fmt formats; the block structs come from ggml-common.h, included there.
#pragma once

// Q2_K / Q3_K: VDR 1 - 16 calls per 256-block, call iqs reading int (iqs % 8) of the 4 q8_1 blocks 4 * (iqs / 8) .. + 3
__device__ __forceinline__ float vec_dot_q2_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
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

__device__ __forceinline__ float vec_dot_q3_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
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

// Q4_K / Q5_K: VDR 2 - 16 calls per 256-block (iqs 0, 2, .. 30), each reading 2 q8_1 blocks
__device__ __forceinline__ float vec_dot_q4_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
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

// Q6_K: VDR 1 - 32 calls per 256-block, each reading 2 q8_1 blocks
__device__ __forceinline__ float vec_dot_q6_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
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
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8 * (__dp4a(vi, u, 0) * sc);
    }
    return __half2float(w->d) * sumf;
}

// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls (glm_fast.cu's F<>)
template<> struct Fmt<10> { static constexpr int qk = 256, ipb = 16, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<11> { static constexpr int qk = 256, ipb = 16, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q3_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<12> { static constexpr int qk = 256, ipb = 16, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<14> { static constexpr int qk = 256, ipb = 32, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q6_K_q8_1(v, y, kbx, iqs); } };
