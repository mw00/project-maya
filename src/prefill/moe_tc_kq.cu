// src/prefill/moe_tc_kq.cu - see include/strata/prefill/moe_tc_kq.hpp: K-quant routed experts on Volta's FP16 tensor
// cores, register fragments.
//
// mma.m8n8k4 splits a warp into four quad pairs (lanes 4q..4q+3 and 16+4q..16+4q+3), each an 8 x 8 x 4 product: a lane
// holds one row of A (4 halves), one column of B (4 halves) and 8 FP32 sums.  A block is 64 routed rows by two sets of 64
// weight rows (rows r and r + 64 of a 128-row block), eight warps of 8 rows a set.  A k step is 128 values (one q8_1
// block); quad pair q takes the step's 32-value chunk q - its lane decodes the row's 32 values (d * scale * q - dmin * min,
// as ggml's dequantize_row_q*_K) straight into eight A fragments.  The activations are staged once a step as FP16
// (double-buffered); the four chunks' partial sums are added by shuffles at the end.
#include "strata/prefill/moe_tc_kq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::prefill::tckq {
namespace {

constexpr int T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14;
constexpr int TS = 256, RT = 64, RK = 128, RAS = RK + 8, ACTB = 144;

template <int TYPE> __host__ __device__ constexpr int sb_bytes() { return TYPE == T_Q4_K ? 144 : TYPE == T_Q5_K ? 176 : 210; }

__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<const uint32_t*>(&h); }

// the 6-bit scale and min of chunk j (0..7) of a Q4_K / Q5_K super-block (ggml's get_scale_min_k4)
__device__ __forceinline__ void scale_min(int j, const uint8_t* q, float& sc, float& m) {
    if (j < 4) {
        sc = (float) (q[j] & 63);
        m = (float) (q[j + 4] & 63);
    } else {
        sc = (float) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (float) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// chunk j (32 values) of the super-block B as 16 half2 (values 2t, 2t + 1)
template <int TYPE>
__device__ __forceinline__ void decode32(const uint8_t* B, int j, uint32_t* a) {
    if constexpr (TYPE == T_Q4_K || TYPE == T_Q5_K) {
        const float d = __half2float(*(const __half*) B), dmin = __half2float(*(const __half*) (B + 2));
        float sc, m;
        scale_min(j, B + 4, sc, m);
        const float dl = d * sc, ml = dmin * m;
        const uint8_t* qs = B + (TYPE == T_Q4_K ? 16 : 48) + (j >> 1) * 32;
        const uint8_t* qh = B + 16;
        const int sh = (j & 1) * 4;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            float v[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int l = 2 * t + e;
                int q = (qs[l] >> sh) & 0xF;
                if constexpr (TYPE == T_Q5_K) q += ((qh[l] >> j) & 1) << 4;
                v[e] = dl * (float) q - ml;
            }
            a[t] = h2u(__floats2half2_rn(v[0], v[1]));
        }
    } else {   // Q6_K: ql[128], qh[64], scales[16] (int8), d
        const uint8_t* ql = B + 64 * (j >> 2);
        const uint8_t* qh = B + 128 + 32 * (j >> 2);
        const int8_t* scs = (const int8_t*) (B + 192) + 8 * (j >> 2);
        const float d = __half2float(*(const __half*) (B + 208));
        const int sub = j & 3;
        const uint8_t* qlp = ql + ((sub & 1) ? 32 : 0);
        const int nsh = sub >= 2 ? 4 : 0, hsh = 2 * sub;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            float v[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int l = 2 * t + e;
                const int q = (((qlp[l] >> nsh) & 0xF) | (((qh[l] >> hsh) & 3) << 4)) - 32;
                v[e] = d * (float) scs[l / 16 + 2 * sub] * (float) q;
            }
            a[t] = h2u(__floats2half2_rn(v[0], v[1]));
        }
    }
}

__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// one thread's share of a step's activations: row tid / 4 of the tile, values [32 (tid % 4), +32) of the q8_1 block;
// the scale from DS4 (half2 d, sum) or D4 (float d)
struct RawA { uint4 q0, q1; float d; };
template <bool DS4>
__device__ __forceinline__ RawA load_a(const uint8_t* xq, int64_t xq_rows, int64_t row, bool live, int kb, int part) {
    RawA r;
    if (!live) { r.q0 = make_uint4(0, 0, 0, 0); r.q1 = r.q0; r.d = 0.0f; return r; }
    const uint8_t* base = xq + ((size_t) kb * (size_t) xq_rows + (size_t) row) * ACTB;
    if (DS4) r.d = __low2float(__ldg((const __half2*) base + part));
    else r.d = __ldg((const float*) base + part);
    const uint4* q = reinterpret_cast<const uint4*>(base + 16 + 32 * part);
    r.q0 = __ldg(q);
    r.q1 = __ldg(q + 1);
    return r;
}
__device__ __forceinline__ void store_a(const RawA& r, __half* o) {
    const __half2 d2 = __float2half2_rn(r.d);
    const uint32_t c1152u = 0x64806480u;
    const __half2 c1152 = *reinterpret_cast<const __half2*>(&c1152u);
    const uint32_t w[8] = {r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w};
    uint32_t h[16];
#pragma unroll
    for (int k = 0; k < 8; ++k) {   // the exact signed code from a mantissa bias: {0x64, byte ^ 0x80} = 1152 + code
        const uint32_t v = w[k] ^ 0x80808080u;
        const uint32_t lo = __byte_perm(v, 0x64646464u, 0x4140), hi = __byte_perm(v, 0x64646464u, 0x4342);
        h[2 * k] = h2u(__hmul2(__hsub2(*reinterpret_cast<const __half2*>(&lo), c1152), d2));
        h[2 * k + 1] = h2u(__hmul2(__hsub2(*reinterpret_cast<const __half2*>(&hi), c1152), d2));
    }
    uint4* o4 = reinterpret_cast<uint4*>(o);
#pragma unroll
    for (int i = 0; i < 4; ++i) o4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
}

template <int TYPE>
__global__ void __launch_bounds__(TS, 1) tc_kq_kernel(const uint8_t* __restrict__ W, size_t expert_bytes, size_t row_bytes,
                                                     int n_in, const int32_t* __restrict__ bounds,
                                                     const uint8_t* __restrict__ xq, int64_t xq_rows,
                                                     float* __restrict__ dst, int64_t ld) {
    constexpr bool DS4 = TYPE != T_Q6_K;
    const int z = blockIdx.z;
    const int lo = bounds[z], hi = bounds[z + 1];
    const int row0 = lo + (int) blockIdx.y * RT;
    if (row0 >= hi) return;
    const int local = min(RT, hi - row0), ngrp = (local + 7) >> 3;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int qp = (lane >> 2) & 3, j = (lane & 3) | ((lane >> 4) << 2);
    const int cb0 = (int) blockIdx.x * 128 + 8 * warp, cb1 = cb0 + 64;   // the warp's first row of each set
    const uint8_t* We = W + (size_t) z * expert_bytes;
    const uint8_t* w0 = We + (size_t) (cb0 + j) * row_bytes;
    const uint8_t* w1 = We + (size_t) (cb1 + j) * row_bytes;
    __shared__ __align__(16) __half sa[2][RT * RAS];
    float c[2][8][8];
#pragma unroll
    for (int s2 = 0; s2 < 2; ++s2)
#pragma unroll
        for (int gi = 0; gi < 8; ++gi)
#pragma unroll
            for (int i = 0; i < 8; ++i) c[s2][gi][i] = 0.0f;
    const int am = tid >> 2, apart = tid & 3;
    const bool alive = am < local;
    const int64_t arow = (int64_t) row0 + am;
    const int nsteps = n_in / RK;
    RawA ra = load_a<DS4>(xq, xq_rows, arow, alive, 0, apart);
    store_a(ra, &sa[0][am * RAS + 32 * apart]);
    __syncthreads();
    for (int st = 0; st < nsteps; ++st) {
        const int buf = st & 1;
        if (st + 1 < nsteps) ra = load_a<DS4>(xq, xq_rows, arow, alive, st + 1, apart);   // (in flight under the products)
        const int k = st * RK + 32 * qp;
        uint32_t a0[16], a1[16];
        decode32<TYPE>(w0 + (size_t) (k >> 8) * sb_bytes<TYPE>(), (k & 255) >> 5, a0);
        decode32<TYPE>(w1 + (size_t) (k >> 8) * sb_bytes<TYPE>(), (k & 255) >> 5, a1);
        const __half* sb = sa[buf];
#pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
            if (gi < ngrp) {
                const uint4* bp = reinterpret_cast<const uint4*>(sb + (8 * gi + j) * RAS + 32 * qp);
                uint4 bv[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) bv[i] = bp[i];
                const uint32_t* bw = reinterpret_cast<const uint32_t*>(bv);
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    mma884(c[0][gi], a0[2 * kk], a0[2 * kk + 1], bw[2 * kk], bw[2 * kk + 1]);
                    mma884(c[1][gi], a1[2 * kk], a1[2 * kk + 1], bw[2 * kk], bw[2 * kk + 1]);
                }
            }
        }
        if (st + 1 < nsteps) store_a(ra, &sa[buf ^ 1][am * RAS + 32 * apart]);
        __syncthreads();
    }
#pragma unroll
    for (int s2 = 0; s2 < 2; ++s2)
#pragma unroll
        for (int gi = 0; gi < 8; ++gi)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = c[s2][gi][i];
                v += __shfl_xor_sync(0xffffffffu, v, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 8);
                c[s2][gi][i] = v;
            }
    if (qp != 0) return;
#pragma unroll
    for (int gi = 0; gi < 8; ++gi) {
        if (gi >= ngrp) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (lane & 1) + (i & 2) + ((lane >> 4) << 2);   // the weight row in the 8
            const int tt = 8 * gi + (i & 4) + (lane & 2) + (i & 1);      // the routed row in the tile
            if (tt < local) {
                float* o = dst + (size_t) (row0 + tt) * ld;
                o[cb0 + rr] = c[0][gi][i];
                o[cb1 + rr] = c[1][gi][i];
            }
        }
    }
}

}  // namespace

bool supported(int type, int64_t n_in, int64_t n_out) {
    return (type == T_Q4_K || type == T_Q5_K || type == T_Q6_K) && n_in % 256 == 0 && n_out % 128 == 0;
}

bool available() {
    static const int ok = [] {
        int dev = 0, major = 0, minor = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        return major == 7 && minor == 0 ? 1 : 0;
    }();
    return ok != 0;
}

void product(int type, const void* W, size_t expert_bytes, int64_t n_in, int64_t n_out, int n_experts, const int32_t* bounds,
             int max_rows, const void* xq, int64_t xq_rows, float* dst, int64_t ld, void* stream) {
    if (!supported(type, n_in, n_out)) throw std::invalid_argument("moe_tc_kq: type / shape");
    if (max_rows <= 0 || n_experts <= 0) return;
    const int sbb = type == T_Q4_K ? 144 : type == T_Q5_K ? 176 : 210;
    const size_t rb = (size_t) (n_in / 256) * sbb;
    const dim3 grid((unsigned) (n_out / 128), (unsigned) ((max_rows + RT - 1) / RT), (unsigned) n_experts);
    cudaStream_t s = (cudaStream_t) stream;
    const uint8_t* w = (const uint8_t*) W;
    const uint8_t* x = (const uint8_t*) xq;
    if (type == T_Q4_K) tc_kq_kernel<T_Q4_K><<<grid, TS, 0, s>>>(w, expert_bytes, rb, (int) n_in, bounds, x, xq_rows, dst, ld);
    else if (type == T_Q5_K) tc_kq_kernel<T_Q5_K><<<grid, TS, 0, s>>>(w, expert_bytes, rb, (int) n_in, bounds, x, xq_rows, dst, ld);
    else tc_kq_kernel<T_Q6_K><<<grid, TS, 0, s>>>(w, expert_bytes, rb, (int) n_in, bounds, x, xq_rows, dst, ld);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("moe_tc_kq: ") + cudaGetErrorString(e));
}

}  // namespace strata::prefill::tckq
