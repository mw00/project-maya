// src/mm3/mm3_gv.cu - maya-mm3's dense GEMVs (mm3_gv.cuh).  The block layouts are ggml's (ggml-common.h, MIT,
// third_party/ggml/LICENSE); each sub-block's codes and scales are exactly ds4_iq.cu's decode (the same integers and
// floats), its sum exactly ds4_iq.cu's sub_dot, so a row's sums are ds4_iq's mv's to the bit.
#include "mm3_gv.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "mm3_env.hpp"

namespace mm3::gv {
namespace {

constexpr int kQ4_K = 12, kQ5_K = 13, kQ6_K = 14, kQ8_0 = 8;   // (ggml's type ids)
constexpr int kWarps = 8, kThreads = kWarps * 32;
constexpr int kMaxJobs = 8;

struct Jobs {
    Job j[kMaxJobs];
};

__device__ __forceinline__ float h2f(uint32_t bits16) { return __half2float(__ushort_as_half((unsigned short) bits16)); }
// the 32 bits at byte offset sh / 8 of the pair {hi:lo} (sh 0, 8, 16, 24)
__device__ __forceinline__ uint32_t funnel(uint32_t lo, uint32_t hi, uint32_t sh) {
    return (uint32_t) ((((unsigned long long) hi << 32) | lo) >> sh);
}

// a sub-block's codes and scales (ds4_iq.cu's Sub): weight i = d s q_i - m, s / m of its half
struct Sub {
    int q[8];
    float d, m;    // (the K-quants with a minimum: m0 = m1 = m)
    int s0, s1;
};

// ggml's get_scale_min_k4 for sub-block ib from the 12 scale bytes (y: bytes 0-3, z: 4-7, w: 8-11)
__device__ __forceinline__ void k4_scale(uint32_t y, uint32_t z, uint32_t w, int ib, int& sc, int& m) {
    const int sh = 8 * (ib & 3);
    const int by = (int) ((y >> sh) & 255u), bz = (int) ((z >> sh) & 255u), bw = (int) ((w >> sh) & 255u);
    if (ib < 4) {
        sc = by & 63;
        m = bz & 63;
    } else {
        sc = (bw & 0xF) | ((by >> 6) << 4);
        m = (bw >> 4) | ((bz >> 6) << 4);
    }
}

// The raw words of row `row`'s sub-block kb a lane reads, then its decode.  ib = kb & 7 is the lane's own (lane & 7):
// every offset inside a block is fixed for the lane, and a step of 32 sub-blocks moves 4 blocks (Q8_0: 32), which keeps
// a Q6_K / Q8_0 block's alignment modulo 8 (resp. 4) the same for the lane - the re-alignment's shifts are the lane's.
template <int T>
struct Raw;

template <>
struct Raw<kQ5_K> {   // 176 bytes, 16-aligned: d, dmin, scales[12] | qh[32] | qs[128]
    uint4 hdr, h0, h1, s0, s1;
    __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int kb) {
        const uint8_t* b = row + (size_t) (kb >> 3) * 176;
        const int ib = kb & 7;
        hdr = *(const uint4*) b;
        h0 = *(const uint4*) (b + 16);
        h1 = *(const uint4*) (b + 32);
        s0 = *(const uint4*) (b + 48 + 32 * (ib >> 1));
        s1 = *(const uint4*) (b + 64 + 32 * (ib >> 1));
    }
    __device__ __forceinline__ void decode(int kb, Sub& s) const {
        const int ib = kb & 7, sh = 4 * (ib & 1);
        int sc, m;
        k4_scale(hdr.y, hdr.z, hdr.w, ib, sc, m);
        const uint32_t qs[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        const uint32_t qh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) (((qs[t] >> sh) & 0x0f0f0f0fu) | (((qh[t] >> ib) & 0x01010101u) << 4));
        s.s0 = s.s1 = sc;
        s.d = h2f(hdr.x & 0xffffu);
        s.m = h2f(hdr.x >> 16) * (float) m;
    }
};

template <>
struct Raw<kQ4_K> {   // 144 bytes, 16-aligned: d, dmin, scales[12] | qs[128]
    uint4 hdr, s0, s1;
    __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int kb) {
        const uint8_t* b = row + (size_t) (kb >> 3) * 144;
        const int ib = kb & 7;
        hdr = *(const uint4*) b;
        s0 = *(const uint4*) (b + 16 + 32 * (ib >> 1));
        s1 = *(const uint4*) (b + 32 + 32 * (ib >> 1));
    }
    __device__ __forceinline__ void decode(int kb, Sub& s) const {
        const int ib = kb & 7, sh = 4 * (ib & 1);
        int sc, m;
        k4_scale(hdr.y, hdr.z, hdr.w, ib, sc, m);
        const uint32_t qs[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) ((qs[t] >> sh) & 0x0f0f0f0fu);
        s.s0 = s.s1 = sc;
        s.d = h2f(hdr.x & 0xffffu);
        s.m = h2f(hdr.x >> 16) * (float) m;
    }
};

// 32 bytes at a (a lane-fixed alignment modulo 8) as 8 words: five 8-byte loads from a rounded down to 8 (they stay
// inside the block's 210 bytes: the chunks below end at byte 160 at most)
struct Chunk8 {
    uint2 v[5];
    __device__ __forceinline__ void load(const uint8_t* a) {
        const uint2* p = (const uint2*) ((uintptr_t) a & ~(uintptr_t) 7);
#pragma unroll
        for (int i = 0; i < 5; ++i) v[i] = p[i];
    }
    __device__ __forceinline__ void words(uintptr_t a, uint32_t* w) const {
        const uint32_t h[10] = {v[0].x, v[0].y, v[1].x, v[1].y, v[2].x, v[2].y, v[3].x, v[3].y, v[4].x, v[4].y};
        const bool o = (a & 4u) != 0;
        const uint32_t sh = 8u * (uint32_t) (a & 3u);
#pragma unroll
        for (int t = 0; t < 8; ++t) w[t] = funnel(o ? h[t + 1] : h[t], o ? h[t + 2] : h[t + 1], sh);
    }
};

template <>
struct Raw<kQ6_K> {   // 210 bytes (2-aligned): ql[128] | qh[64] | scales[16] int8 | d
    Chunk8 l, h;
    uint32_t sc, d;
    uintptr_t al, ah;
    __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int kb) {
        const uint8_t* b = row + (size_t) (kb >> 3) * 210;
        const int ib = kb & 7, hf = ib >> 2, qt = ib & 3;
        const uint8_t* pl = b + 64 * hf + 32 * (qt & 1);
        const uint8_t* ph = b + 128 + 32 * hf;
        al = (uintptr_t) pl;
        ah = (uintptr_t) ph;
        l.load(pl);
        h.load(ph);
        sc = *(const uint16_t*) (b + 192 + 8 * hf + 2 * qt);
        d = *(const uint16_t*) (b + 208);
    }
    __device__ __forceinline__ void decode(int kb, Sub& s) const {
        const int qt = (kb & 7) & 3;
        uint32_t ql[8], qh[8];
        l.words(al, ql);
        h.words(ah, qh);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t v = ((ql[t] >> (4 * (qt >> 1))) & 0x0f0f0f0fu) | (((qh[t] >> (2 * qt)) & 0x03030303u) << 4);
            const uint32_t lo = (~v) & 0x20202020u;
            s.q[t] = (int) ((v ^ 0x20202020u) | (lo << 1) | (lo << 2));
        }
        s.s0 = (int) (int8_t) (sc & 255u);
        s.s1 = (int) (int8_t) (sc >> 8);
        s.d = h2f(d);
        s.m = 0.0f;
    }
};

template <>
struct Raw<kQ8_0> {   // 34 bytes (2-aligned): d | qs[32]; a sub-block is a block
    uint32_t w[9], d;
    uintptr_t a;
    __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int kb) {
        const uint8_t* b = row + (size_t) kb * 34;
        a = (uintptr_t) (b + 2);
        const uint32_t* p = (const uint32_t*) (a & ~(uintptr_t) 3);
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = p[i];
        // (the ninth word only when the codes straddle it: a block ending a row - and so the matrix - never does)
        w[8] = (a & 3u) ? p[8] : 0u;
        d = *(const uint16_t*) b;
    }
    __device__ __forceinline__ void decode(int, Sub& s) const {
        const uint32_t sh = 8u * (uint32_t) (a & 3u);
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) funnel(w[t], w[t + 1], sh);
        s.s0 = s.s1 = 1;
        s.d = h2f(d);
        s.m = 0.0f;
    }
};

template <int T>
constexpr bool kMin = T == kQ4_K || T == kQ5_K;

// ds4_iq.cu's sub_dot to the bit: an exact integer dot, its scales in F32 with explicit roundings; the minimum's sums
// (sl, sh: the activations' 16-value sums, the same integers its dp4a with 0x01010101 make)
template <int T>
__device__ __forceinline__ float dot(const Sub& s, const int4& xa, const int4& xb, int sums) {
    int lo = __dp4a(s.q[0], xa.x, 0);
    lo = __dp4a(s.q[1], xa.y, lo);
    lo = __dp4a(s.q[2], xa.z, lo);
    lo = __dp4a(s.q[3], xa.w, lo);
    int hi = __dp4a(s.q[4], xb.x, 0);
    hi = __dp4a(s.q[5], xb.y, hi);
    hi = __dp4a(s.q[6], xb.z, hi);
    hi = __dp4a(s.q[7], xb.w, hi);
    float v = __fmul_rn(s.d, (float) (s.s0 * lo + s.s1 * hi));
    if constexpr (kMin<T>) {
        const int sl = (int) (int16_t) (sums & 0xffff), sh = sums >> 16;
        v = __fsub_rn(v, __fadd_rn(__fmul_rn(s.m, (float) sl), __fmul_rn(s.m, (float) sh)));
    }
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {   // (ds4_iq.cu's butterfly)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// grid (rows / (kWarps R), jobs): a warp R rows, a lane sub-blocks lane, lane + 32, ... of each - its R rows' raw words
// loaded first, then each column's activations once for the R rows
template <int T, int NC, int R, bool PF>
__global__ void __launch_bounds__(kThreads) gv_kernel(const Jobs J) {
    const Job& b = J.j[blockIdx.y];
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * R;
    if (row0 >= b.n) return;
    const int nsb = b.k / 32;
    const uint8_t* rows[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rows[r] = b.w + (size_t) min(row0 + r, b.n - 1) * b.ld;   // (past the end: row n - 1 again)
    float acc[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
    // (PF: the next step's raw words loaded while this one's are decoded and summed)
    Raw<T> nxt[R];
    if (PF && lane < nsb)
#pragma unroll
        for (int r = 0; r < R; ++r) nxt[r].load(rows[r], lane);
    for (int kb = lane; kb < nsb; kb += 32) {
        Raw<T> raw[R];
        if constexpr (PF) {
#pragma unroll
            for (int r = 0; r < R; ++r) raw[r] = nxt[r];
            if (kb + 32 < nsb)
#pragma unroll
                for (int r = 0; r < R; ++r) nxt[r].load(rows[r], kb + 32);
        } else {
#pragma unroll
            for (int r = 0; r < R; ++r) raw[r].load(rows[r], kb);
        }
        Sub s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) raw[r].decode(kb, s[r]);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < b.ncol) {
                const int4* xp = (const int4*) (b.xq + (size_t) c * b.k + (size_t) kb * 32);
                const int4 xa = xp[0], xb = xp[1];
                const float dx = b.xd[(size_t) c * nsb + kb];
                const int sums = kMin<T> ? b.xs[(size_t) c * nsb + kb] : 0;
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r][c] = fmaf(dx, dot<T>(s[r], xa, xb, sums), acc[r][c]);
            }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < b.ncol) {
                const float v = warp_sum(acc[r][c]);
                if (lane == 0 && row < b.n) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
            }
    }
}

// Q4_K / Q5_K, one activation column (the token's GEMVs): a step's 4 blocks of each of the warp's R rows loaded
// coalesced (16 bytes a lane, each byte once) into shared memory, the next step's in registers meanwhile; each lane's
// sub-block then read from shared memory.  The same lanes, sub-blocks, order and roundings as gv_kernel.
template <int T, int R>
__global__ void __launch_bounds__(kThreads) gvs_kernel(const Jobs J) {
    constexpr int SB = 4 * (T == kQ5_K ? 176 : 144);   // a step's bytes a row
    constexpr int NV = SB / 16, NV2 = NV - 32;          // uint4 a step a row: lanes < NV2 take a second
    static_assert(NV2 > 0 && NV2 <= 32, "a step: 32 to 64 uint4");
    __shared__ __align__(16) uint8_t sm[kWarps][R][SB];
    const Job& b = J.j[blockIdx.y];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * kWarps + wp) * R;
    if (row0 >= b.n) return;
    const int nsb = b.k / 32, steps = nsb / 32;
    const uint8_t* rows[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rows[r] = b.w + (size_t) min(row0 + r, b.n - 1) * b.ld;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    uint4 g0[R], g1[R];
    const bool two = lane < NV2;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        g0[r] = ((const uint4*) rows[r])[lane];
        g1[r] = two ? ((const uint4*) rows[r])[32 + lane] : make_uint4(0u, 0u, 0u, 0u);
    }
    for (int st = 0; st < steps; ++st) {
        __syncwarp();
#pragma unroll
        for (int r = 0; r < R; ++r) {
            ((uint4*) sm[wp][r])[lane] = g0[r];
            if (two) ((uint4*) sm[wp][r])[32 + lane] = g1[r];
        }
        __syncwarp();
        if (st + 1 < steps)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint4* p = (const uint4*) (rows[r] + (size_t) (st + 1) * SB);
                g0[r] = p[lane];
                if (two) g1[r] = p[32 + lane];
            }
        const int kb = st * 32 + lane;
        const int4* xp = (const int4*) (b.xq + (size_t) kb * 32);
        const int4 xa = xp[0], xb = xp[1];
        const float dx = b.xd[kb];
        const int sums = b.xs[kb];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            Raw<T> raw;
            raw.load(sm[wp][r], lane);   // (the step's blocks from 0: sub-block lane of them is kb's)
            Sub s;
            raw.decode(kb, s);
            acc[r] = fmaf(dx, dot<T>(s, xa, xb, sums), acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0 && row0 + r < b.n) b.y[row0 + r] = b.alpha * v + (b.bias != nullptr ? b.bias[row0 + r] : 0.0f);
    }
}

int sm_count() {
    static const int n = [] {
        int dev = 0, v = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) !=
                                                       cudaSuccess || v <= 0) {
            cudaGetLastError();
            v = 80;
        }
        return v;
    }();
    return n;
}

template <int T, int NC, int R, bool PF = false>
void launch(const Jobs& J, int nj, int max_n, cudaStream_t s) {
    const dim3 grid((unsigned) ((max_n + kWarps * R - 1) / (kWarps * R)), (unsigned) nj);
    gv_kernel<T, NC, R, PF><<<grid, kThreads, 0, s>>>(J);
}

// The form a launch takes (measured on a V100 over MiniMax's shapes - the sums are the same in every form):
//   one column: Q4_K / Q5_K the staged kernel, 2 rows a warp (each row's steps whole: k % 1024 == 0); Q6_K / Q8_0 (and
//   the K-quants otherwise) a row a warp with the next step's words loaded ahead;
//   more columns (<= 8): 4 rows a warp while the launch keeps 3 blocks an SM, else one; more than 8: one
template <int T, int NC>
void launch_r(const Jobs& J, int nj, int max_n, cudaStream_t s) {
    if constexpr (NC == 1) {
        if constexpr (T == kQ4_K || T == kQ5_K) {
            bool whole = true;
            for (int i = 0; i < nj; ++i) whole = whole && J.j[i].k % 1024 == 0;
            if (whole) {
                const dim3 grid((unsigned) ((max_n + kWarps * 2 - 1) / (kWarps * 2)), (unsigned) nj);
                gvs_kernel<T, 2><<<grid, kThreads, 0, s>>>(J);
                return;
            }
        }
        return launch<T, 1, 1, true>(J, nj, max_n, s);
    }
    const long b4 = (long) nj * ((max_n + kWarps * 4 - 1) / (kWarps * 4));
    if constexpr (NC <= 8) {
        if (b4 >= 3L * sm_count()) return launch<T, NC, 4>(J, nj, max_n, s);
    }
    launch<T, NC, 1>(J, nj, max_n, s);
}

template <int T>
void launch_t(const Jobs& J, int nj, int max_n, int max_c, cudaStream_t s) {
    // (a verify window's columns exactly up to 8: the accumulators of columns that are not there cost registers -
    //  occupancy - in a kernel that waits on its loads)
    switch (max_c) {
        case 1: return launch_r<T, 1>(J, nj, max_n, s);
        case 2: return launch_r<T, 2>(J, nj, max_n, s);
        case 3: return launch_r<T, 3>(J, nj, max_n, s);
        case 4: return launch_r<T, 4>(J, nj, max_n, s);
        case 5: return launch_r<T, 5>(J, nj, max_n, s);
        case 6: return launch_r<T, 6>(J, nj, max_n, s);
        case 7: return launch_r<T, 7>(J, nj, max_n, s);
        case 8: return launch_r<T, 8>(J, nj, max_n, s);
        default: return launch_r<T, 16>(J, nj, max_n, s);
    }
}

size_t block_bytes(int t) {
    switch (t) {
        case kQ4_K: return 144;
        case kQ5_K: return 176;
        case kQ6_K: return 210;
        case kQ8_0: return 34;
        default: return 0;
    }
}

}  // namespace

bool enabled() {
#if defined(STRATA_USE_HIP)
    return false;   // (CUDA only: HIP's compiler fuses these kernels' multiplies and adds unlike ds4_iq's mv - other bits)
#else
    static const bool on = [] {
        const char* v = ::mm3::env("MM3_GV");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
#endif
}

bool needs_sums(int type) { return type == kQ4_K || type == kQ5_K; }

bool covered(const Job& j) {
#if defined(STRATA_USE_HIP)
    (void) j;
    return false;
#endif
    const size_t bb = block_bytes(j.type);
    if (bb == 0 || j.w == nullptr || j.ncol < 1 || j.ncol > 16 || j.n < 1 || j.k < 256) return false;
    const int qk = j.type == kQ8_0 ? 32 : 256;
    if (j.k % qk != 0 || j.ld != (size_t) (j.k / qk) * bb) return false;
    // (the wide loads' alignment: Q4_K / Q5_K blocks 16-byte aligned; Q6_K's and Q8_0's alignment modulo 8 / 4 the
    //  same from one step of a lane to the next - rows a multiple of 1024 / 128 values, the matrix 16-byte aligned)
    if (((uintptr_t) j.w & 15u) != 0) return false;
    if (j.type == kQ6_K && j.k % 1024 != 0) return false;
    if (j.type == kQ8_0 && j.k % 128 != 0) return false;
    if (needs_sums(j.type) && j.xs == nullptr) return false;
    // (the activations' int4 loads)
    if (((uintptr_t) j.xq & 15u) != 0 || j.k % 16 != 0) return false;
    return true;
}

bool run(const Job* jobs, int n, cudaStream_t s) {
    bool done[64] = {};
    if (n > 64) return false;
    for (int i = 0; i < n; ++i)
        if (!covered(jobs[i])) return false;
    for (int i = 0; i < n; ++i)
        while (!done[i]) {
            const int t = jobs[i].type;
            Jobs J{};
            int nj = 0, max_n = 0, max_c = 1;
            for (int k = i; k < n && nj < kMaxJobs; ++k) {
                if (done[k] || jobs[k].type != t) continue;
                J.j[nj++] = jobs[k];
                max_n = std::max(max_n, jobs[k].n);
                max_c = std::max(max_c, jobs[k].ncol);
                done[k] = true;
            }
            switch (t) {
                case kQ4_K: launch_t<kQ4_K>(J, nj, max_n, max_c, s); break;
                case kQ5_K: launch_t<kQ5_K>(J, nj, max_n, max_c, s); break;
                case kQ6_K: launch_t<kQ6_K>(J, nj, max_n, max_c, s); break;
                case kQ8_0: launch_t<kQ8_0>(J, nj, max_n, max_c, s); break;
                default: return false;
            }
        }
    return true;
}

}  // namespace mm3::gv
