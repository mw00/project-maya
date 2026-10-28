// src/ds4/ds4_gguf.cu - a GGUF's BF16 matrices and MXFP4 experts on the device (see ds4_gguf.cuh).
#include "ds4_gguf.cuh"
#include "ds4_pdl.cuh"

#include "ds4_iq.cuh"
#include "ds4_numerics.cuh"

#include <cuda_fp16.h>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#include <thread>
#include <string>
#include <chrono>

namespace strata::ds4::gguf {
namespace {

std::atomic<int> g_errors{0};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds4 gguf %s: %s\n", what, cudaGetErrorString(e));
        g_errors.fetch_add(1, std::memory_order_relaxed);
    }
}
void fail(const char* what) {
    std::fprintf(stderr, "ds4 gguf %s\n", what);
    g_errors.fetch_add(1, std::memory_order_relaxed);
}

constexpr int kWarps = 8, kThreads = kWarps * 32;
constexpr int kMaxEnt = 16;   // a group's entries in one pass (a window's tokens at most)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------- BF16 matrices
struct BfJob {
    const void* w;
    size_t ld;          // values a row (BF16), bytes a row (MXFP4)
    const float* x;
    size_t xcs;
    float* y;
    size_t ycs;
    const float* bias;
    float alpha;
    int k, n, ncol;
    const int8_t* xq;   // (Q8_0: the activations rounded to int8, q8_quant's)
    const float* xd;
};
struct BfJobs {
    BfJob j[x3::kMaxMvJobs];
};

// grid (max rows / 8, jobs): a warp a row, each of its activations' dot in F32 - a lane 8 BF16 weights (16 bytes) and
// 8 activations a step (VEC), or one of each (rows or activations not 16-byte aligned)
template <int NC, bool VEC>
__global__ void __launch_bounds__(kThreads) bf16_mv_kernel(const BfJobs J) {
    pdl_wait();
    pdl_trigger();
    const BfJob& b = J.j[blockIdx.y];
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= b.n) return;
    const uint16_t* w = (const uint16_t*) b.w + (size_t) row * b.ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    if (VEC) {
        for (int i = lane * 8; i < b.k; i += 256) {
            const uint4 q = *(const uint4*) (w + i);
            const float wf[8] = {__uint_as_float(q.x << 16), __uint_as_float(q.x & 0xffff0000u),
                                 __uint_as_float(q.y << 16), __uint_as_float(q.y & 0xffff0000u),
                                 __uint_as_float(q.z << 16), __uint_as_float(q.z & 0xffff0000u),
                                 __uint_as_float(q.w << 16), __uint_as_float(q.w & 0xffff0000u)};
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < b.ncol) {
                    const float4* xp = (const float4*) (b.x + c * b.xcs + i);
                    const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
                    float a = acc[c];
                    a = fmaf(wf[0], x0.x, a);
                    a = fmaf(wf[1], x0.y, a);
                    a = fmaf(wf[2], x0.z, a);
                    a = fmaf(wf[3], x0.w, a);
                    a = fmaf(wf[4], x1.x, a);
                    a = fmaf(wf[5], x1.y, a);
                    a = fmaf(wf[6], x1.z, a);
                    a = fmaf(wf[7], x1.w, a);
                    acc[c] = a;
                }
        }
    } else {
        for (int i = lane; i < b.k; i += 32) {
            const float wf = __uint_as_float((uint32_t) w[i] << 16);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < b.ncol) acc[c] = fmaf(wf, b.x[c * b.xcs + i], acc[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < b.ncol) {
            const float v = warp_sum(acc[c]);
            if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
        }
}


__global__ void q8_rows_kernel(const uint8_t* __restrict__ w, size_t ld, int k, int n, uint16_t* __restrict__ dst,
                               bool bf16) {
    pdl_wait();
    pdl_trigger();
    const size_t total = (size_t) n * (size_t) k;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
        const size_t r = i / (size_t) k, c = i - r * (size_t) k;
        const uint8_t* p = w + r * ld + (c / 32) * kQ8Bytes;
        const float v = __half2float(__ushort_as_half(*(const uint16_t*) p)) * (float) (int8_t) p[2 + c % 32];
        dst[i] = bf16 ? strata::ds4::bf16_rn(v) : __half_as_ushort(__float2half_rn(v));
    }
}

__global__ void bf16_rows_kernel(const uint16_t* __restrict__ w, size_t ld, int k, int n, uint16_t* __restrict__ dst,
                                 bool bf16) {
    pdl_wait();
    pdl_trigger();
    const size_t total = (size_t) n * (size_t) k;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
        const size_t r = i / (size_t) k, c = i - r * (size_t) k;
        const uint16_t b = w[r * ld + c];
        dst[i] = bf16 ? b : __half_as_ushort(__float2half_rn(__uint_as_float((uint32_t) b << 16)));
    }
}

// ---------------------------------------------------------------- MXFP4 experts
__device__ __forceinline__ float e8m0_half(uint32_t e) {   // 2^(e - 128), ggml's GGML_E8M0_TO_FP32_HALF
    return __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
}
__device__ __forceinline__ void lut_init(float* lut) {
    if (threadIdx.x < 16) {
        const float v[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f};
        lut[threadIdx.x] = threadIdx.x < 8 ? v[threadIdx.x] : -v[threadIdx.x - 8];
    }
}
// block p's 32 values, exactly (a code times a power of two)
__device__ __forceinline__ void mx_block(const uint8_t* __restrict__ p, const float* lut, float* w) {
    const float d = e8m0_half(p[0]);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint32_t q = p[1 + j];
        w[j] = lut[q & 15u] * d;
        w[j + 16] = lut[q >> 4] * d;
    }
}
__device__ __forceinline__ float swiglu(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);
}
__device__ __forceinline__ float combine(const float* plan_w, const float* p, int k, const float* cpu_part, float sh) {
    float m = 0.0f;
    for (int i = 0; i < k; ++i) m = __fmaf_rn(plan_w[i], p[i], m);
    if (cpu_part != nullptr) m = __fadd_rn(m, *cpu_part);
    return __fadd_rn(m, sh);
}

// a Q8_0 block's 32 values, exactly (an FP16 scale times an int8; the block is only 2-byte aligned)
__device__ __forceinline__ void q8_block(const uint8_t* __restrict__ p, float* w) {
    const float d = __half2float(__ushort_as_half(*(const uint16_t*) p));
    const uint16_t* q = (const uint16_t*) (p + 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint16_t v = q[j];
        w[2 * j] = d * (float) (int8_t) (v & 0xff);
        w[2 * j + 1] = d * (float) (int8_t) (v >> 8);
    }
}

// ---- int8 activations (llama.cpp's MMVQ: its q8_1 blocks' rounding)
// rows of k F32 values (row r at x + r ldx; k % 32 == 0) -> q[r][k] int8 and d[r][k / 32] = max|x| / 127 per 32
// (a warp a 32-block: row r's block b)
__device__ __forceinline__ void q8_quant_block(const float* __restrict__ x, size_t ldx, int k, int8_t* __restrict__ q,
                                               float* __restrict__ d, int r, int b, int lane) {
    if (b * 32 >= k) return;
    const float v = x[(size_t) r * ldx + (size_t) b * 32 + lane];
    float am = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[(size_t) r * (k / 32) + b] = sc;
}
__global__ void __launch_bounds__(kThreads) q8_quant_kernel(const float* __restrict__ x, size_t ldx, int k,
                                                            int8_t* __restrict__ q, float* __restrict__ d) {
    pdl_wait();
    pdl_trigger();
    q8_quant_block(x, ldx, k, q, d, blockIdx.y, blockIdx.x * kWarps + (threadIdx.x >> 5), threadIdx.x & 31);
}
void q8_quant(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    const dim3 grid((unsigned) ((k / 32 + kWarps - 1) / kWarps), (unsigned) rows);
    pdl_launch(q8_quant_kernel, grid, kThreads, 0, s, x, ldx, k, q, d);
}
// (DS4_Q8_BATCH) a mv call's int8 inputs in one launch: job z's rows as q8_quant_kernel rounds them, bit for bit -
// the same warp a 32-block; a job's blocks past its k and rows past its rows end at once
struct Q8Job {
    const float* x;
    size_t ldx;
    int8_t* q;
    float* d;
    int k, rows;
};
struct Q8Jobs {
    Q8Job j[x3::kMaxMvJobs];
};
__global__ void __launch_bounds__(kThreads) q8_quant_jobs_kernel(const Q8Jobs J) {
    pdl_wait();
    pdl_trigger();
    const Q8Job& a = J.j[blockIdx.z];
    if ((int) blockIdx.y >= a.rows) return;
    q8_quant_block(a.x, a.ldx, a.k, a.q, a.d, blockIdx.y, blockIdx.x * kWarps + (threadIdx.x >> 5), threadIdx.x & 31);
}
void q8_quant_jobs(const Q8Jobs& J, int n, cudaStream_t s) {
    int bx = 1, by = 1;
    for (int i = 0; i < n; ++i) {
        bx = std::max(bx, (J.j[i].k / 32 + kWarps - 1) / kWarps);
        by = std::max(by, J.j[i].rows);
    }
    pdl_launch(q8_quant_jobs_kernel, dim3((unsigned) bx, (unsigned) by, (unsigned) n), kThreads, 0, s, J);
}
// a mv call's int8 inputs: each job's rows rounded into its own scratch, in one launch before the GEMVs (batch) or a
// launch at each job; share: a job whose input an earlier one's is (the same x, stride, rows and k - wq_a, wkv and the
// compressor's two on one x, the shared expert's w1 and w3) reads that job's rows instead of rounding them again
struct Q8Round {
    Q8Jobs J;
    int n = 0;   // (<= kMaxMvJobs: a job adds one at most)
    bool batch = true, share = true;
    // (rq, rd): where the job reads its rows - (q, d), or the earlier job's
    void add(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, const int8_t*& rq, const float*& rd,
             cudaStream_t s) {
        for (int i = 0; share && i < n; ++i) {
            const Q8Job& a = J.j[i];
            if (a.x == x && a.ldx == ldx && a.rows == rows && a.k == k) {
                rq = a.q;
                rd = a.d;
                return;
            }
        }
        J.j[n++] = {x, ldx, q, d, k, rows};
        rq = q;
        rd = d;
        if (!batch) q8_quant(x, ldx, rows, k, q, d, s);
    }
    void flush(cudaStream_t s) const {
        if (batch && n > 0) q8_quant_jobs(J, n, s);
    }
};

// ---- the low-bit experts' activations as the CPU lane rounds them (ggml's Q8_K, quantize_row_q8_K_ref, ggml-cpu's
// on x86): per 256 values max = the signed value of the largest magnitude (the first one), iscale = -127 / max, q =
// min(127, nearest_int(iscale x)) - the product rounded to F32 before the magic-number rounding, as the source does
// (src/kernels/cuda/quantize_act.cu's note) - and d = 1 / iscale, written to each of the 256's eight 32-blocks of d:
// the kernels' sums then take the CPU lane's int8 values and scale, so an expert gives the same sums on either lane
// (up to their order, as MXFP4's do) wherever the placement, the expert cache or the PCIe lane puts it: Maya-S's
// tokens no longer depend on the placement (Strix Halo, --vram-experts 1.0 vs 0.85: 96 of 96 tokens the same, 50 with
// the per-32 rounding), the KL against the FP8 reference the same (0.162 against 0.171, two sequences), the kernels
// as fast (6 IQ2_XXS / IQ3_XXS experts 275 against 280 us).  DS4_IQ_Q8K=0: the per-32 rounding (q8_1's), as before
__device__ __forceinline__ int q8k_nearest(float fval) {   // ggml's nearest_int
    const float val = __fadd_rn(fval, 12582912.0f);
    int i;
    memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}
__global__ void __launch_bounds__(256) q8k_quant_kernel(const float* __restrict__ x, size_t ldx, int k,
                                                        int8_t* __restrict__ q, float* __restrict__ d) {
    pdl_wait();
    pdl_trigger();
    __shared__ float s_am[8];
    __shared__ int s_at[8];
    const int r = blockIdx.y, j = (int) threadIdx.x, lane = j & 31, w = j >> 5;
    const size_t at = (size_t) blockIdx.x * 256 + (size_t) j;
    const float v = x[(size_t) r * ldx + at];
    float am = fabsf(v);
    int ix = j;   // (the first index of the largest magnitude: ties keep the lower one)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
        if (oa > am || (oa == am && oi < ix)) am = oa, ix = oi;
    }
    if (lane == 0) s_am[w] = am, s_at[w] = ix;
    __syncthreads();
    if (j == 0) {
        for (int i = 1; i < 8; ++i)
            if (s_am[i] > s_am[0] || (s_am[i] == s_am[0] && s_at[i] < s_at[0])) s_am[0] = s_am[i], s_at[0] = s_at[i];
    }
    __syncthreads();
    const float amax = s_am[0];
    int8_t qv = 0;
    float dv = 0.0f;
    if (amax > 0.0f) {
        const float mx = x[(size_t) r * ldx + (size_t) blockIdx.x * 256 + (size_t) s_at[0]];
        const float iscale = __fdiv_rn(-127.0f, mx);
        qv = (int8_t) min(127, q8k_nearest(__fmul_rn(iscale, v)));
        dv = __fdiv_rn(1.0f, iscale);
    }
    q[(size_t) r * k + at] = qv;
    if (lane == 0) d[(size_t) r * (k / 32) + at / 32] = dv;
}
void q8k_quant(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    pdl_launch(q8k_quant_kernel, dim3((unsigned) (k / 256), (unsigned) rows), 256, 0, s, x, ldx, k, q, d);
}
// the low-bit experts' activations: Q8_K's rounding (the CPU lane's; DS4_IQ_Q8K=0: q8_1's, per 32)
bool iq_q8k() {
    static const bool on = [] {
        const char* v = std::getenv("DS4_IQ_Q8K");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}
void iq_act_quant(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    if (iq_q8k() && k % 256 == 0) q8k_quant(x, ldx, rows, k, q, d, s);
    else q8_quant(x, ldx, rows, k, q, d, s);
}

// 8 E2M1 codes (a word of 4 code bytes) -> their values as int8: .x the low nibbles' (values 0..3 of the word's bytes),
// .y the high nibbles' - llama.cpp's get_int_from_table_16 (MIT), the table kvalues_mxfp4
__device__ __forceinline__ int2 e2m1_x8(uint32_t q4) {
    const uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t idx = (q4 >> (16 * i)) & 0x7777u;
        const uint32_t lo = __byte_perm(t0, t1, idx), hi = __byte_perm(t2, t3, idx);
        tmp[i] = __byte_perm(lo, hi, sel >> (16 * i));
    }
    return make_int2((int) __byte_perm(tmp[0], tmp[1], 0x6420), (int) __byte_perm(tmp[0], tmp[1], 0x7531));
}
// an MXFP4 block's codes (4 words) against 32 int8 activations (xa: values 0..15, xb: 16..31)
__device__ __forceinline__ int mx_dot(const uint4& q, const int4& xa, const int4& xb) {
    int sum = 0;
    int2 v = e2m1_x8(q.x);
    sum = __dp4a(v.x, xa.x, sum);
    sum = __dp4a(v.y, xb.x, sum);
    v = e2m1_x8(q.y);
    sum = __dp4a(v.x, xa.y, sum);
    sum = __dp4a(v.y, xb.y, sum);
    v = e2m1_x8(q.z);
    sum = __dp4a(v.x, xa.z, sum);
    sum = __dp4a(v.y, xb.z, sum);
    v = e2m1_x8(q.w);
    sum = __dp4a(v.x, xa.w, sum);
    sum = __dp4a(v.y, xb.w, sum);
    return sum;
}
// the warp copies the blocks [b0, b0 + nbk) of a row (BB bytes each; the row 4-byte aligned, nbk BB % 4 == 0) into its
// stage in 4-byte words; every lane calls it
template <int BB>
__device__ __forceinline__ void stage_chunk(const uint8_t* row, int b0, int nbk, uint32_t* st) {
    const uint32_t* src = (const uint32_t*) (row + (size_t) b0 * BB);
    __syncwarp();   // (the lanes are done with the chunk before)
    for (int i = threadIdx.x & 31; i < nbk * BB / 4; i += 32) st[i] = src[i];
    __syncwarp();
}
// a row's next chunk of 32 blocks (BB bytes each) in the lanes' registers, then into the warp's stage
template <int BB>
struct Fetch {
    static constexpr int W = (32 * BB / 4 + 31) / 32;   // words a lane
    uint32_t r[W];
    int n = 0;
    __device__ __forceinline__ void load(const uint8_t* row, int b0, int nbk) {
        const uint32_t* src = (const uint32_t*) (row + (size_t) b0 * BB);
        n = nbk * BB / 4;
#pragma unroll
        for (int j = 0; j < W; ++j) {
            const int i = (int) (threadIdx.x & 31) + 32 * j;
            r[j] = i < n ? src[i] : 0u;
        }
    }
    __device__ __forceinline__ void store(uint32_t* st) const {
        __syncwarp();   // (the lanes are done with the chunk before)
#pragma unroll
        for (int j = 0; j < W; ++j) {
            const int i = (int) (threadIdx.x & 31) + 32 * j;
            if (i < n) st[i] = r[j];
        }
        __syncwarp();
    }
};

// MXFP4 block i of a stage (17-byte blocks from a 4-byte aligned start): its scale byte, its 16 code bytes as 4 words
__device__ __forceinline__ uint32_t mx_words_s(const uint32_t* st, int i, uint4& q) {
    const int o = 17 * i, w = o >> 2, sh = (o & 3) * 8;
    const uint32_t a0 = st[w], a1 = st[w + 1], a2 = st[w + 2], a3 = st[w + 3], a4 = st[w + 4];
    const uint32_t s0 = __funnelshift_r(a0, a1, sh), s1 = __funnelshift_r(a1, a2, sh), s2 = __funnelshift_r(a2, a3, sh),
                   s3 = __funnelshift_r(a3, a4, sh), s4 = a4 >> sh;
    q.x = __funnelshift_r(s0, s1, 8);
    q.y = __funnelshift_r(s1, s2, 8);
    q.z = __funnelshift_r(s2, s3, 8);
    q.w = __funnelshift_r(s3, s4, 8);
    return s0 & 0xffu;
}
// ... the same from global memory, byte by byte (rows not 4-byte aligned)
__device__ __forceinline__ uint32_t mx_words_g(const uint8_t* __restrict__ p, uint4& q) {
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        w[j] = (uint32_t) p[1 + 4 * j] | (uint32_t) p[2 + 4 * j] << 8 | (uint32_t) p[3 + 4 * j] << 16 |
               (uint32_t) p[4 + 4 * j] << 24;
    q = make_uint4(w[0], w[1], w[2], w[3]);
    return p[0];
}
// Q8_0 block i of a stage (34-byte blocks): its scale, its 32 int8 as 8 words
__device__ __forceinline__ float q8_words_s(const uint32_t* st, int i, int* v) {
    const int o = 34 * i, w = o >> 2, sh = (o & 3) * 8;
    uint32_t a[9];
#pragma unroll
    for (int j = 0; j < 9; ++j) a[j] = st[w + j];
    uint32_t sw[9];
#pragma unroll
    for (int j = 0; j < 8; ++j) sw[j] = __funnelshift_r(a[j], a[j + 1], sh);
    sw[8] = a[8] >> sh;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = (int) __funnelshift_r(sw[j], sw[j + 1], 16);
    return __half2float(__ushort_as_half((uint16_t) (sw[0] & 0xffffu)));
}
__device__ __forceinline__ float q8_words_g(const uint8_t* __restrict__ p, int* v) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
        v[j] = (int) ((uint32_t) p[2 + 4 * j] | (uint32_t) p[3 + 4 * j] << 8 | (uint32_t) p[4 + 4 * j] << 16 |
                      (uint32_t) p[5 + 4 * j] << 24);
    return __half2float(__ushort_as_half((uint16_t) (p[0] | p[1] << 8)));
}

// grid (max rows / 8, jobs): a Q8_0 matrix's rows against the jobs' int8 activations (q8_quant's: activation c of job
// j at xq + c k, its scales at xd + c k / 32) - a warp a row, a lane a block a step, the blocks staged (ST) or read
template <int NC, bool ST>
__global__ void __launch_bounds__(kThreads) q8_mv_kernel(const BfJobs J) {
    pdl_wait();
    pdl_trigger();
    __shared__ __align__(16) uint32_t stage[kWarps][32 * kQ8Bytes / 4];
    const BfJob& b = J.j[blockIdx.y];
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= b.n) return;
    const uint8_t* w = (const uint8_t*) b.w + (size_t) row * b.ld;
    const int nb = b.k / 32;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    Fetch<kQ8Bytes> F;
    if (ST) F.load(w, 0, min(32, nb));
    for (int b0 = 0; b0 < nb; b0 += 32) {
        const int nbk = min(32, nb - b0), kb = b0 + lane;
        if (ST) {
            F.store(stage[threadIdx.x >> 5]);
            if (b0 + 32 < nb) F.load(w, b0 + 32, min(32, nb - b0 - 32));
        }
        if (lane >= nbk) continue;
        int v[8];
        const float dw = ST ? q8_words_s(stage[threadIdx.x >> 5], lane, v) : q8_words_g(w + (size_t) kb * kQ8Bytes, v);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < b.ncol) {
                const int4* xp = (const int4*) (b.xq + (size_t) c * b.k + (size_t) kb * 32);
                const int4 xa = xp[0], xb = xp[1];
                int sum = __dp4a(v[0], xa.x, 0);
                sum = __dp4a(v[1], xa.y, sum);
                sum = __dp4a(v[2], xa.z, sum);
                sum = __dp4a(v[3], xa.w, sum);
                sum = __dp4a(v[4], xb.x, sum);
                sum = __dp4a(v[5], xb.y, sum);
                sum = __dp4a(v[6], xb.z, sum);
                sum = __dp4a(v[7], xb.w, sum);
                acc[c] = fmaf(dw * b.xd[(size_t) c * nb + kb], (float) sum, acc[c]);
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < b.ncol) {
            const float v = warp_sum(acc[c]);
            if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
        }
}

// grid (max rows / 8, jobs): bf16_mv_kernel for MXFP4 (Q8 false) or Q8_0 (Q8 true) matrices - a lane a 32-value block
// a step, dequantised once for the job's activations
template <int NC, bool VEC, bool Q8>
__global__ void __launch_bounds__(kThreads) mx_mv_kernel(const BfJobs J) {
    pdl_wait();
    pdl_trigger();
    __shared__ float lut[16];
    lut_init(lut);
    __syncthreads();
    const BfJob& b = J.j[blockIdx.y];
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= b.n) return;
    const uint8_t* w = (const uint8_t*) b.w + (size_t) row * b.ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int kb = lane; kb < b.k / kMxBlock; kb += 32) {
        float wv[32];
        if (Q8) q8_block(w + (size_t) kb * kQ8Bytes, wv);
        else mx_block(w + (size_t) kb * kMxBytes, lut, wv);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < b.ncol) {
                const float* x = b.x + c * b.xcs + kb * kMxBlock;
                float a = acc[c];
                if (VEC) {
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        const float4 xv = __ldg((const float4*) x + q);
                        a = fmaf(wv[4 * q], xv.x, a);
                        a = fmaf(wv[4 * q + 1], xv.y, a);
                        a = fmaf(wv[4 * q + 2], xv.z, a);
                        a = fmaf(wv[4 * q + 3], xv.w, a);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 32; ++j) a = fmaf(wv[j], x[j], a);
                }
                acc[c] = a;
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < b.ncol) {
            const float v = warp_sum(acc[c]);
            if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
        }
}

// a token's experts (grp_ptr null: group i = expert i of the plan, plan_ptr[i] its blob or 0 off the device, its one
// entry i, every entry's input x) or a window's groups (group g: one expert, blob grp_ptr[g], entries e = t * 8 + i at
// grp_ent[grp_start[g] ..), entry e's input x + (e >> 3) x_stride)
struct MxArgs {
    const unsigned long long* plan_ptr;
    const unsigned long long* grp_ptr;
    const int *grp_start, *grp_ent, *n_groups;
    const float* x;
    size_t x_stride;
    float* h;      // [entries][n_ff]
    float* part;   // [entries][n_embd]
    size_t tr[3];
    int n_embd, n_ff;
    float limit;
    const int8_t* xq;   // x rounded to int8 (a token's: row 0; a window's: row t), q8_quant's rows of n_embd
    const float* xd;
    const int8_t* hq;   // h rounded to int8, row e = the entry, rows of n_ff
    const float* hd;
};

__device__ __forceinline__ bool mx_group(const MxArgs& a, int grp, const uint8_t*& blob, int& m0, int& m1) {
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

// grid (n_ff / 8, groups): a warp a row of the gate and of the up, for NE of the group's entries a pass (each weight
// block decoded once for them) against the entries' int8 x -> h = the clamped swiglu
template <int NE, bool ST>
__global__ void __launch_bounds__(kThreads) mx_gate_up_kernel(const MxArgs a) {
    pdl_wait();
    pdl_trigger();
    __shared__ __align__(16) uint32_t stage[kWarps][2][32 * kMxBytes / 4];
    __shared__ int es[NE], xr[NE];
    const uint8_t* blob;
    int m0, m1;
    if (!mx_group(a, blockIdx.y, blob, m0, m1)) return;   // (uniform)
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const size_t rb = mx_row_bytes(a.n_embd);
    const uint8_t* g = blob + a.tr[0] + (size_t) row * rb;
    const uint8_t* u = blob + a.tr[1] + (size_t) row * rb;
    const int nb = a.n_embd / kMxBlock;
    for (int mm = m0; mm < m1; mm += NE) {
        const int ne = min(NE, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) {
            const int e = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
            es[threadIdx.x] = e;
            xr[threadIdx.x] = a.grp_ptr != nullptr ? e >> 3 : 0;
        }
        __syncthreads();
        if (row >= a.n_ff) continue;
        float ag[NE], au[NE];
#pragma unroll
        for (int i = 0; i < NE; ++i) ag[i] = au[i] = 0.0f;
        Fetch<kMxBytes> FG, FU;
        if (ST) {
            FG.load(g, 0, min(32, nb));
            FU.load(u, 0, min(32, nb));
        }
        for (int b0 = 0; b0 < nb; b0 += 32) {
            const int nbk = min(32, nb - b0), b = b0 + lane;
            if (ST) {
                FG.store(stage[wp][0]);
                FU.store(stage[wp][1]);
                if (b0 + 32 < nb) {
                    FG.load(g, b0 + 32, min(32, nb - b0 - 32));
                    FU.load(u, b0 + 32, min(32, nb - b0 - 32));
                }
            }
            if (lane >= nbk) continue;
            uint4 qg, qu;
            const uint32_t eg = ST ? mx_words_s(stage[wp][0], lane, qg) : mx_words_g(g + (size_t) b * kMxBytes, qg);
            const uint32_t eu = ST ? mx_words_s(stage[wp][1], lane, qu) : mx_words_g(u + (size_t) b * kMxBytes, qu);
            const float dg = e8m0_half(eg), du = e8m0_half(eu);
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const int4* xp = (const int4*) (a.xq + (size_t) xr[i] * a.n_embd + (size_t) b * kMxBlock);
                    const int4 xa = xp[0], xb = xp[1];
                    const float dx = a.xd[(size_t) xr[i] * nb + b];
                    ag[i] = fmaf(dg * dx, (float) mx_dot(qg, xa, xb), ag[i]);
                    au[i] = fmaf(du * dx, (float) mx_dot(qu, xa, xb), au[i]);
                }
        }
#pragma unroll
        for (int i = 0; i < NE; ++i)
            if (i < ne) {
                const float gv = warp_sum(ag[i]), uv = warp_sum(au[i]);
                if (lane == 0) a.h[(size_t) es[i] * a.n_ff + row] = swiglu(gv, uv, a.limit);
            }
    }
}

// grid (n_embd / 8, groups): a warp a row of the down for NE entries a pass, against their int8 h -> part
template <int NE, bool ST>
__global__ void __launch_bounds__(kThreads) mx_down_kernel(const MxArgs a) {
    pdl_wait();
    pdl_trigger();
    __shared__ __align__(16) uint32_t stage[kWarps][32 * kMxBytes / 4];
    __shared__ int es[NE];
    const uint8_t* blob;
    int m0, m1;
    if (!mx_group(a, blockIdx.y, blob, m0, m1)) return;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const uint8_t* d = blob + a.tr[2] + (size_t) row * mx_row_bytes(a.n_ff);
    const int nb = a.n_ff / kMxBlock;
    for (int mm = m0; mm < m1; mm += NE) {
        const int ne = min(NE, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) es[threadIdx.x] = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
        __syncthreads();
        if (row >= a.n_embd) continue;
        float ad[NE];
#pragma unroll
        for (int i = 0; i < NE; ++i) ad[i] = 0.0f;
        Fetch<kMxBytes> FD;
        if (ST) FD.load(d, 0, min(32, nb));
        for (int b0 = 0; b0 < nb; b0 += 32) {
            const int nbk = min(32, nb - b0), b = b0 + lane;
            if (ST) {
                FD.store(stage[wp]);
                if (b0 + 32 < nb) FD.load(d, b0 + 32, min(32, nb - b0 - 32));
            }
            if (lane >= nbk) continue;
            uint4 q;
            const float dd = e8m0_half(ST ? mx_words_s(stage[wp], lane, q) : mx_words_g(d + (size_t) b * kMxBytes, q));
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const int4* hp = (const int4*) (a.hq + (size_t) es[i] * a.n_ff + (size_t) b * kMxBlock);
                    const int4 ha = hp[0], hb = hp[1];
                    ad[i] = fmaf(dd * a.hd[(size_t) es[i] * nb + b], (float) mx_dot(q, ha, hb), ad[i]);
                }
        }
#pragma unroll
        for (int i = 0; i < NE; ++i)
            if (i < ne) {
                const float v = warp_sum(ad[i]);
                if (lane == 0) a.part[(size_t) es[i] * a.n_embd + row] = v;
            }
    }
}

struct CombArgs {
    const unsigned long long* plan_ptr;   // [T][8]
    const float* plan_w;                  // [T][8]
    const float* part;                    // [T * 8][n_embd]
    const float* cpu_part;
    size_t cpu_stride;
    const int* cpu_flag;
    const float* sh_out;                  // [T][n_embd]
    float* out;
    size_t out_stride;
    int k, n_embd;
};
// grid (n_embd / 128, T): each token's outputs - the experts in plan order (an expert off the device adds 0), then
// the CPU lane's part and the shared expert's (x3's combine)
__global__ void __launch_bounds__(128) mx_combine_kernel(const CombArgs a) {
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.y, r = blockIdx.x * 128 + threadIdx.x;
    if (r >= a.n_embd) return;
    float p[8];
    for (int i = 0; i < a.k; ++i) {
        const int e = t * 8 + i;
        p[i] = a.plan_ptr[e] != 0ull ? a.part[(size_t) e * a.n_embd + r] : 0.0f;
    }
    a.out[(size_t) t * a.out_stride + r] =
        combine(a.plan_w + t * 8, p, a.k,
                a.cpu_flag != nullptr && a.cpu_flag[t] ? a.cpu_part + (size_t) t * a.cpu_stride + r : nullptr,
                a.sh_out != nullptr ? a.sh_out[(size_t) t * a.n_embd + r] : 0.0f);
}

__global__ void mx_wait_kernel(const unsigned* p, unsigned v) {
    pdl_wait();
    if (threadIdx.x != 0) return;
    while (*(volatile const unsigned*) p != v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
    }
    __threadfence_system();
    pdl_trigger();   // (only after the wait: ds4_pdl.cuh)
}

// ---------------------------------------------------------------- the prompt path's expert sets
// a set's expert i (blob wbase + i stride) owns the sorted rows [bounds[i], bounds[i + 1]); local row = sorted row -
// row0.  A block: BM of its rows x BN outputs, the weights dequantised a 32-value block at a time into shared memory,
// F32 FMAs (each thread TM x TN outputs)
constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
static_assert((BM / TM) * (BN / TN) == kThreads, "one output tile a thread");

struct SetK {
    const uint8_t* wbase;
    size_t stride;
    size_t tr[3];
    const int* bounds;
    int row0;
    int n_embd, n_ff;
    float limit;
    const float* xf;      // gate / up: F32 input rows x[idx[local]] (ldx)
    const int* idx;
    const __half* xh;     // down: F16 rows (ldx)
    int ldx;
    __half* h;            // gate / up: F16 out
    float* y;             // down: F32 out
    int ldo;
    int skip_le;          // the experts of at most this many rows left out (the light kernels took them; ds4_iq.cuh)
    int xh_vec;           // (AMD's tiles) xh 16-byte aligned F16 rows - gate / up too when the caller gave them (x16, the
                          // rows_f16 of xf: the same bits) - read 8 values a load
};

// one 32-value block of BN weight rows (rows c0 .., row bytes rb, block kb) -> ws[32][BN + 1]
__device__ __forceinline__ void mx_tile(const uint8_t* w, size_t rb, int c0, int kb, const float* lut,
                                        float (*ws)[BN + 1]) {
    const int r = threadIdx.x >> 2, q = threadIdx.x & 3;   // row r of the tile, values [8 q, 8 q + 8)
    const uint8_t* p = w + (size_t) (c0 + r) * rb + (size_t) kb * kMxBytes;
    const float d = e8m0_half(p[0]);
    const int b0 = (q & 1) * 8, hi = q >> 1;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t c = p[1 + b0 + j];
        ws[q * 8 + j][r] = lut[hi ? (c >> 4) : (c & 15u)] * d;
    }
}

template <bool GATE_UP>
__global__ void __launch_bounds__(kThreads) mx_set_kernel(const SetK a) {
    pdl_wait();
    pdl_trigger();
    __shared__ float lut[16];
    __shared__ float wg[32][BN + 1];
    __shared__ float wu[GATE_UP ? 32 : 1][BN + 1];
    __shared__ float xs[BM][33];
    __shared__ int src[BM];
    lut_init(lut);
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi || hi - lo <= a.skip_le) return;   // (uniform: before any barrier but lut_init's writes, which nobody
                                                    // reads then; skip_le: the light kernels' experts)
    const int nr = min(BM, hi - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const size_t rb = mx_row_bytes(K);
    const int tid = (int) threadIdx.x;
    if (tid < BM) src[tid] = tid < nr ? (GATE_UP ? a.idx[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const int ty = threadIdx.x / (BN / TN), tx = threadIdx.x % (BN / TN);
    float ag[TM][TN], au[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) ag[i][j] = au[i][j] = 0.0f;
    for (int kb = 0; kb < K / kMxBlock; ++kb) {
        mx_tile(blob + a.tr[GATE_UP ? 0 : 2], rb, c0, kb, lut, wg);
        if (GATE_UP) mx_tile(blob + a.tr[1], rb, c0, kb, lut, wu);
        for (int i = threadIdx.x; i < BM * kMxBlock; i += kThreads) {   // the rows' 32 inputs
            const int r = i / kMxBlock, c = i % kMxBlock, sr = src[r];
            float v = 0.0f;
            if (sr >= 0) {
                const size_t at = (size_t) sr * a.ldx + (size_t) kb * kMxBlock + c;
                v = GATE_UP ? a.xf[at] : __half2float(a.xh[at]);
            }
            xs[r][c] = v;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < kMxBlock; ++kk) {
            float xv[TM], gw[TN], uw[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) xv[i] = xs[ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                gw[j] = wg[kk][tx * TN + j];
                if (GATE_UP) uw[j] = wu[kk][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    ag[i][j] = fmaf(xv[i], gw[j], ag[i][j]);
                    if (GATE_UP) au[i][j] = fmaf(xv[i], uw[j], au[i][j]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = ty * TM + i;
        if (r >= nr) continue;
        const size_t lr = (size_t) (r0 + r);
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = c0 + tx * TN + j;
            if (GATE_UP) a.h[lr * a.ldo + c] = __float2half_rn(swiglu(ag[i][j], au[i][j], a.limit));
            else a.y[lr * a.ldo + c] = ag[i][j];
        }
    }
}

// the same on the tensor cores: a block BM rows x BN outputs, 8 warps each 16 rows x 32 outputs (2 fragments, gate and
// up: 4), a 32-value MXFP4 block of K a step (2 WMMA k-steps), the tiles in shared memory as F16
#if !defined(STRATA_USE_HIP)
namespace wm = nvcuda::wmma;
constexpr int kLdT = 40;   // halves a tile row (32 + 8: WMMA wants a multiple of 8; the pad spreads the banks)
template <bool GATE_UP>
__global__ void __launch_bounds__(kThreads) mx_set_tc_kernel(const SetK a) {
    pdl_wait();
    pdl_trigger();
    struct Tiles {
        __half x[BM][kLdT], wg[BN][kLdT], wu[GATE_UP ? BN : 1][kLdT];
    };
    struct Out {
        float g[BM][BN + 4], u[GATE_UP ? BM : 1][BN + 4];
    };
    __shared__ __align__(32) unsigned char smem[sizeof(Tiles) > sizeof(Out) ? sizeof(Tiles) : sizeof(Out)];
    __shared__ float lut[16];
    __shared__ int src[BM];
    Tiles& T = *reinterpret_cast<Tiles*>(smem);
    Out& O = *reinterpret_cast<Out*>(smem);
    lut_init(lut);
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi || hi - lo <= a.skip_le) return;   // (uniform; skip_le: the light kernels' experts)
    const int nr = min(BM, hi - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const size_t rb = mx_row_bytes(K);
    const int tid = (int) threadIdx.x, warp = tid >> 5;
    if (tid < BM) src[tid] = tid < nr ? (GATE_UP ? a.idx[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const int wm_ = warp >> 1, wn = (warp & 1) * 2;   // the warp's 16 rows, its 2 x 16 outputs
    wm::fragment<wm::accumulator, 16, 16, 16, float> fg[2], fu[2];
    for (int j = 0; j < 2; ++j) {
        wm::fill_fragment(fg[j], 0.0f);
        wm::fill_fragment(fu[j], 0.0f);
    }
    const int tr = tid >> 2, tq = tid & 3;   // the thread's tile row, its 8 values [8 tq, 8 tq + 8)
    for (int kb = 0; kb < K / kMxBlock; ++kb) {
        {   // the weights: row tr of the outputs, values 8 tq ..
            for (int m = 0; m < (GATE_UP ? 2 : 1); ++m) {
                const uint8_t* q = blob + a.tr[GATE_UP ? m : 2] + (size_t) (c0 + tr) * rb + (size_t) kb * kMxBytes;
                const float d = e8m0_half(q[0]);
                const int b0 = (tq & 1) * 8, hn = tq >> 1;
                __half* dst = m == 0 ? &T.wg[tr][tq * 8] : &T.wu[tr][tq * 8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t c = q[1 + b0 + j];
                    dst[j] = __float2half_rn(lut[hn ? (c >> 4) : (c & 15u)] * d);
                }
            }
            // the inputs: row tr, values 8 tq ..
            const int sr = src[tr];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                __half v = __float2half_rn(0.0f);
                if (sr >= 0) {
                    const size_t at = (size_t) sr * a.ldx + (size_t) kb * kMxBlock + tq * 8 + j;
                    v = GATE_UP ? __float2half_rn(a.xf[at]) : a.xh[at];
                }
                T.x[tr][tq * 8 + j] = v;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> fa;
            wm::load_matrix_sync(fa, &T.x[wm_ * 16][kk], kLdT);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> fb;
                wm::load_matrix_sync(fb, &T.wg[(wn + j) * 16][kk], kLdT);
                wm::mma_sync(fg[j], fa, fb, fg[j]);
                if (GATE_UP) {
                    wm::load_matrix_sync(fb, &T.wu[(wn + j) * 16][kk], kLdT);
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
        if (GATE_UP) a.h[lr * a.ldo + c0 + c] = __float2half_rn(swiglu(O.g[r][c], O.u[r][c], a.limit));
        else a.y[lr * a.ldo + c0 + c] = O.g[r][c];
    }
}
#endif
#if defined(STRATA_USE_HIP)
// the same on an AMD wave32 GPU's WMMA (RDNA3 / RDNA3.5 gfx11, RDNA4 gfx12): 8 waves each 16 rows x 32 outputs, the
// tiles in shared memory as F16, F32 accumulation - each lane gives its row / column's 16 values of a k-step (gfx12:
// its half of them), and holds 8 of the 16 x 16 results (rows 2 i + hi on gfx11, 8 hi + i on gfx12; column lane % 16)
using h16v = _Float16 __attribute__((ext_vector_type(16)));
using f8v = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ f8v set_wmma(h16v a, h16v b, f8v c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif defined(__gfx1200__) || defined(__gfx1201__)
    using h8 = _Float16 __attribute__((ext_vector_type(8)));
    const int k0 = (threadIdx.x & 31) / 16 * 8;
    h8 aa, bb;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        aa[i] = a[k0 + i];
        bb[i] = b[k0 + i];
    }
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aa, bb, c);
#else
    __builtin_trap();
    return c;
#endif
}
__device__ __forceinline__ int set_wmma_row(int i, int hi) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return 8 * hi + i;
#else
    return 2 * i + hi;
#endif
}
constexpr int kLdH = 40;   // halves a tile row (32 + 8: the pad spreads the banks)
template <bool GATE_UP>
__global__ void __launch_bounds__(kThreads) mx_set_wmma_kernel(const SetK a) {
    pdl_wait();
    pdl_trigger();
    struct Tiles {
        _Float16 x[BM][kLdH], wg[BN][kLdH], wu[GATE_UP ? BN : 1][kLdH];
    };
    // (the results go out from the accumulators, whose rows set_wmma_row knows: no 64 x 64 staging of them - 15 KB of
    // shared memory a block, not 35, so more blocks a CU)
    __shared__ __align__(16) unsigned char smem[sizeof(Tiles)];
    __shared__ float lut[16];
    __shared__ int src[BM];
    Tiles& T = *reinterpret_cast<Tiles*>(smem);
    lut_init(lut);
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi_r = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi_r || hi_r - lo <= a.skip_le) return;   // (uniform; skip_le: the light kernels' experts)
    const int nr = min(BM, hi_r - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const size_t rb = mx_row_bytes(K);
    const int tid = (int) threadIdx.x, warp = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    if (tid < BM) src[tid] = tid < nr ? (GATE_UP ? a.idx[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const int wm_ = warp >> 1, wn = (warp & 1) * 2;   // the wave's 16 rows, its 2 x 16 outputs
    f8v fg[2], fu[2];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) fg[j][i] = fu[j][i] = 0.0f;
    const int tr = tid >> 2, tq = tid & 3;   // the thread's tile row, its 8 values [8 tq, 8 tq + 8)
    for (int kb = 0; kb < K / kMxBlock; ++kb) {
        for (int m = 0; m < (GATE_UP ? 2 : 1); ++m) {
            const uint8_t* q = blob + a.tr[GATE_UP ? m : 2] + (size_t) (c0 + tr) * rb + (size_t) kb * kMxBytes;
            const float d = e8m0_half(q[0]);
            const int b0 = (tq & 1) * 8, hn = tq >> 1;
            _Float16* dst = m == 0 ? &T.wg[tr][tq * 8] : &T.wu[tr][tq * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t cc = q[1 + b0 + j];
                dst[j] = (_Float16) (lut[hn ? (cc >> 4) : (cc & 15u)] * d);
            }
        }
        const int sr = src[tr];
        if (a.xh_vec) {   // (F16 rows: 8 values in one 16-byte load)
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (sr >= 0) v = *reinterpret_cast<const uint4*>(a.xh + (size_t) sr * a.ldx + (size_t) kb * kMxBlock + tq * 8);
            *reinterpret_cast<uint4*>(&T.x[tr][tq * 8]) = v;
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float v = 0.0f;
                if (sr >= 0) {
                    const size_t at = (size_t) sr * a.ldx + (size_t) kb * kMxBlock + tq * 8 + j;
                    v = GATE_UP ? a.xf[at] : __half2float(a.xh[at]);
                }
                T.x[tr][tq * 8 + j] = (_Float16) v;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            h16v av;
#pragma unroll
            for (int i = 0; i < 16; ++i) av[i] = T.x[wm_ * 16 + l16][kk + i];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                h16v bv;
#pragma unroll
                for (int i = 0; i < 16; ++i) bv[i] = T.wg[(wn + j) * 16 + l16][kk + i];
                fg[j] = set_wmma(av, bv, fg[j]);
                if (GATE_UP) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) bv[i] = T.wu[(wn + j) * 16 + l16][kk + i];
                    fu[j] = set_wmma(av, bv, fu[j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = wm_ * 16 + set_wmma_row(i, hi), cc = (wn + j) * 16 + l16;
            if (r >= nr) continue;
            const size_t lr = (size_t) (r0 + r);
            if (GATE_UP) a.h[lr * a.ldo + c0 + cc] = __float2half_rn(swiglu(fg[j][i], fu[j][i], a.limit));
            else a.y[lr * a.ldo + c0 + cc] = fg[j][i];
        }
}
#endif
// sm_70 and later (WMMA F16) / AMD gfx11, gfx12 (wave32 WMMA); STRATA_DS4_NO_TC=1 keeps the F32 kernel - per device
// (the GPUs of a run need not be of one generation: a card before sm_70 beside newer ones keeps its F32 kernel)
bool tc_ok() {
    static const bool off = [] {
        const char* v = std::getenv("STRATA_DS4_NO_TC");
        return v != nullptr && v[0] == '1';
    }();
    static std::atomic<int> known[64];   // 0: not asked yet, 1: no, 2: yes
    int dev = 0;
    if (off || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    int k = known[dev].load(std::memory_order_relaxed);
    if (k == 0) {
        bool ok = false;
#if defined(STRATA_USE_HIP)
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
            for (const char* p : {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201"})
                ok = ok || std::strncmp(prop.gcnArchName, p, 7) == 0;
#else
        int major = 0;
        ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 7;
#endif
        k = ok ? 2 : 1;
        known[dev].store(k, std::memory_order_relaxed);
    }
    return k == 2;
}

// ---------------------------------------------------------------- scratch
constexpr int kMaxEmbd = 8192;
constexpr int kMaxQK = 32768;   // a Q8_0 GEMV's inputs at most
struct Scratch {
    float* part = nullptr;   // [8][kMaxEmbd]: a token's experts' down outputs
    int8_t *xq = nullptr, *hq = nullptr, *mq = nullptr;   // int8 rows: x [kMaxEnt], h [kMaxEnt * 8], the GEMVs'
    float *xd = nullptr, *hd = nullptr, *md = nullptr;     // [kMaxMvJobs * kMaxMvCols] inputs; their scales
};
std::mutex g_mu;
// (by device and stream: the default stream is one handle on every device, and a destroyed stream's handle may come
// back on another)
std::map<std::pair<int, cudaStream_t>, Scratch> g_scratch;

Scratch* scratch(cudaStream_t s) {
    std::lock_guard<std::mutex> lk(g_mu);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;
    auto it = g_scratch.find({dev, s});
    if (it != g_scratch.end()) return &it->second;
    Scratch sc;
    const size_t nx = (size_t) kMaxEnt * kMaxEmbd, nh = (size_t) kMaxEnt * 8 * kMaxEmbd,
                 nm = (size_t) x3::kMaxMvJobs * x3::kMaxMvCols * kMaxQK;
    if (cudaMalloc(&sc.part, (size_t) 8 * kMaxEmbd * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&sc.xq, nx) != cudaSuccess || cudaMalloc(&sc.xd, nx / 32 * 4) != cudaSuccess ||
        cudaMalloc(&sc.hq, nh) != cudaSuccess || cudaMalloc(&sc.hd, nh / 32 * 4) != cudaSuccess ||
        cudaMalloc(&sc.mq, nm) != cudaSuccess || cudaMalloc(&sc.md, nm / 32 * 4) != cudaSuccess) {
        fail("scratch: cudaMalloc failed");
        return nullptr;
    }
    return &g_scratch.emplace(std::make_pair(dev, s), sc).first->second;
}

bool mx_geometry(const char* who, int k, int n_embd, int n_ff) {
    if (k < 1 || k > 8 || n_embd % BN != 0 || n_ff % BN != 0 || n_embd > kMaxEmbd || n_ff % kMxBlock != 0) {
        std::fprintf(stderr, "ds4 gguf %s: geometry not covered (k %d n_embd %d n_ff %d)\n", who, k, n_embd, n_ff);
        g_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------- entry points
bool prepare(cudaStream_t s) { return scratch(s) != nullptr; }

namespace {
std::atomic<int> g_q8_mode{-1};   // (tests: q8_mode)
bool q8_env(const char* name) {   // (unset: on)
    const char* v = std::getenv((std::string("STRATA_") + name).c_str());
    if (v == nullptr) v = std::getenv(name);
    return v == nullptr || std::atoi(v) != 0;
}
}  // namespace
bool q8_batch() {   // a mv call's int8 inputs in one launch (DS4_Q8_BATCH=0: a launch a job, as before)
    const int m = g_q8_mode.load(std::memory_order_relaxed);
    if (m >= 0) return (m & 1) != 0;
    static const bool on = [] {
        const bool b = q8_env("DS4_Q8_BATCH");
        if (!b) std::fprintf(stderr, "ds4: a GEMV call's int8 inputs rounded a launch a job (DS4_Q8_BATCH=0)\n");
        return b;
    }();
    return on;
}
bool q8_share() {   // jobs on one input share its rounding (DS4_Q8_SHARE=0: each job its own, as before)
    const int m = g_q8_mode.load(std::memory_order_relaxed);
    if (m >= 0) return (m & 2) != 0;
    static const bool on = [] {
        const bool b = q8_env("DS4_Q8_SHARE");
        if (!b) std::fprintf(stderr, "ds4: each GEMV job rounds its own int8 input (DS4_Q8_SHARE=0)\n");
        return b;
    }();
    return on;
}
void q8_mode(int m) { g_q8_mode.store(m, std::memory_order_relaxed); }

// DS4_Q8_CHECK=1 (at load, before any request): eight jobs - k 5120 / 2304 / 256 / 32, rows 1 / 4 / 16, strided rows,
// two jobs reading one input with other k, two of the same input, stride, rows and k, values with zero blocks,
// negative zeros, halves at the rounding, tiny and large magnitudes - rounded by the per-job launches, each job its own
// (no sharing), into a set of outputs filled 0xA5, then by mv's batched launch with sharing (Q8Round) into a second
// set filled 0xA5 and a third filled 0x5A, each job's int8 rows and scales between 256 guard bytes.  Only the last job
// may share (the second's twin): every other one reads its own rows.  What each job reads must be the same as the
// first set's rows of that job byte for byte (every int8, every scale's bits); the guards stay 0xA5 / 0x5A (none
// touched past a job's rows or k), and the sharing job's own rows and scales are never written.  The uploads, the
// fills and the launches on one stream of its own, their end waited for, bounded; a CUDA error, a mismatch or no clean
// end in time fails the load (without a clean end everything is kept).  No tolerance: the batched kernel runs the
// per-job one's code
bool q8_batch_check(int dev, std::string& err) {
    struct Spec {
        int k, rows, ldx, input;
    };
    static const Spec spec[x3::kMaxMvJobs] = {{5120, 1, 5120, 0}, {2304, 4, 2368, 1}, {256, 16, 288, 2},
                                              {32, 1, 32, 3},     {5120, 4, 5248, 4}, {2304, 1, 5120, 0},
                                              {32, 16, 64, 5},    {2304, 4, 2368, 1}};
    constexpr int kInputs = 6;
    constexpr int kTwin = 7, kTwinOf = 1;   // (the job that shares, the one it shares)
    constexpr size_t kGuard = 256;
    const char* where = "";
    auto fail_at = [&](const std::string& why) {
        err = "Q8 batch check on CUDA" + std::to_string(dev) + ": " + why;
        std::fprintf(stderr, "ds4: CUDA%d Q8 batch check FAILED: %s\n", dev, why.c_str());
        return false;
    };
    // the inputs (host): a fixed generator; block kinds by the block's index
    std::vector<std::vector<float>> in((size_t) kInputs);
    for (int i = 0; i < x3::kMaxMvJobs; ++i) {
        const Spec& p = spec[i];
        std::vector<float>& v = in[(size_t) p.input];
        v.resize(std::max(v.size(), (size_t) p.rows * (size_t) p.ldx), 0.0f);
    }
    uint32_t st = 0x9e3779b9u;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return st; };
    for (auto& v : in)
        for (size_t c = 0; c < v.size(); ++c) {
            const size_t blk = c / 32;
            const int lane = (int) (c % 32);
            const float u = (float) (int) (rnd() % 2001u - 1000u) / 250.0f;   // -4 .. 4
            switch (blk % 8) {
                case 0: v[c] = 0.0f; break;                                     // a block of zeros (scale 0)
                case 1: v[c] = lane == 0 ? -127.0f : (float) (lane - 16) + 0.5f; break;   // halves at scale 1
                case 2: v[c] = lane % 3 == 0 ? -0.0f : u * 1e-30f; break;       // tiny, negative zeros
                case 3: v[c] = u * 1e30f; break;                                // large
                case 4: v[c] = lane == 5 ? 127.0f * 3.0f : u; break;            // a lone maximum
                default: v[c] = u * (float) (1 + blk % 5); break;
            }
        }
    int cur = 0;
    cudaGetDevice(&cur);
    if (cudaSetDevice(dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return fail_at("cudaSetDevice");
    }
    // the outputs: each job's int8 rows then its scales, each between guards; three sets in one allocation
    size_t qat[x3::kMaxMvJobs], dat[x3::kMaxMvJobs], set_bytes = 0, payload_q = 0, payload_d = 0;
    for (int i = 0; i < x3::kMaxMvJobs; ++i) {
        const size_t qn = (size_t) spec[i].rows * (size_t) spec[i].k, dn = qn / 32 * sizeof(float);
        qat[i] = set_bytes + kGuard;
        set_bytes = qat[i] + qn + kGuard;
        dat[i] = set_bytes + kGuard;
        set_bytes = dat[i] + dn + kGuard;
        payload_q += qn;
        payload_d += dn / sizeof(float);
    }
    // the inputs held until their uploads are known to have ended: uploaded on the check's own stream, before its
    // kernels on that stream (no default-stream ordering relied on)
    auto* held = new std::vector<std::vector<float>>(std::move(in));
    std::vector<float*> dx((size_t) kInputs, nullptr);
    uint8_t *out = nullptr, *back = nullptr;   // (back: the three sets read back, pinned, on the same stream)
    cudaStream_t s = nullptr;
    cudaEvent_t ev = nullptr;
    bool ok = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess &&
              cudaMalloc((void**) &out, 3 * set_bytes) == cudaSuccess &&
              cudaMallocHost((void**) &back, 3 * set_bytes) == cudaSuccess;
    bool submitted = false;   // (anything enqueued on s: waited for before any free)
    for (int j = 0; ok && j < kInputs; ++j) {
        const std::vector<float>& v = (*held)[(size_t) j];
        ok = cudaMalloc((void**) &dx[(size_t) j], v.size() * sizeof(float)) == cudaSuccess;
        if (ok) {
            submitted = true;
            ok = cudaMemcpyAsync(dx[(size_t) j], v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice, s) ==
                 cudaSuccess;
        }
    }
    where = "setup";
    int reads[3][x3::kMaxMvJobs];   // (sets 1 and 2: the job whose rows job i reads)
    for (int b = 0; b < 3; ++b)
        for (int i = 0; i < x3::kMaxMvJobs; ++i) reads[b][i] = i;
    if (ok) {
        uint8_t* set[3] = {out, out + set_bytes, out + 2 * set_bytes};
        ok = cudaMemsetAsync(set[0], 0xA5, set_bytes, s) == cudaSuccess &&
             cudaMemsetAsync(set[1], 0xA5, set_bytes, s) == cudaSuccess &&
             cudaMemsetAsync(set[2], 0x5A, set_bytes, s) == cudaSuccess;
        for (int i = 0; ok && i < x3::kMaxMvJobs; ++i)
            q8_quant(dx[(size_t) spec[i].input], (size_t) spec[i].ldx, spec[i].rows, spec[i].k,
                     (int8_t*) (set[0] + qat[i]), (float*) (set[0] + dat[i]), s);
        for (int b = 1; ok && b < 3; ++b) {
            Q8Round R;
            R.batch = true;
            R.share = true;
            for (int i = 0; i < x3::kMaxMvJobs; ++i) {
                const int8_t* rq = nullptr;
                const float* rd = nullptr;
                R.add(dx[(size_t) spec[i].input], (size_t) spec[i].ldx, spec[i].rows, spec[i].k,
                      (int8_t*) (set[b] + qat[i]), (float*) (set[b] + dat[i]), rq, rd, s);
                int j = 0;   // (the job whose rows these are; kMaxMvJobs: none of them)
                while (j < x3::kMaxMvJobs &&
                       !((const uint8_t*) rq == set[b] + qat[j] && (const uint8_t*) rd == set[b] + dat[j]))
                    ++j;
                reads[b][i] = j;
            }
            R.flush(s);
        }
        const bool launched = cudaGetLastError() == cudaSuccess;
        ok = ok && launched &&
             cudaMemcpyAsync(back, out, 3 * set_bytes, cudaMemcpyDeviceToHost, s) == cudaSuccess;   // (before the event)
        where = launched ? "completion" : "launch";
    }
    // the end of whatever went in (the uploads, the fills, the launches, the readback): its event, else the stream
    // itself, bounded; only a clean end frees anything (an error or no end in time: everything kept - the inputs, the
    // device outputs, the pinned readback, the stream - and the load fails)
    bool done = !submitted;
    if (submitted) {
        const bool recorded = cudaEventRecord(ev, s) == cudaSuccess;
        ok = ok && recorded;
        const auto t0 = std::chrono::steady_clock::now();
        cudaError_t q;
        while ((q = recorded ? cudaEventQuery(ev) : cudaStreamQuery(s)) == cudaErrorNotReady &&
               std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10))
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        done = q == cudaSuccess;
        if (q == cudaErrorNotReady) where = "no end in 10 s";
        else if (!done) where = "an error at the end";
        ok = ok && done;
    }
    std::vector<uint8_t> h;
    if (ok) h.assign(back, back + 3 * set_bytes);
    else (void) cudaGetLastError();
    if (done) {
        for (float* p : dx)
            if (p != nullptr) cudaFree(p);
        if (out != nullptr) cudaFree(out);
        if (back != nullptr) cudaFreeHost(back);
        if (ev != nullptr) cudaEventDestroy(ev);
        if (s != nullptr) cudaStreamDestroy(s);
        delete held;
    }
    cudaSetDevice(cur);
    if (!ok) return fail_at(std::string("a CUDA error or no end (") + where + ")");
    for (int b = 1; b < 3; ++b)   // (the twin reads its original's rows, every other job its own)
        for (int i = 0; i < x3::kMaxMvJobs; ++i)
            if (reads[b][i] != (i == kTwin ? kTwinOf : i)) {
                char m[160];
                std::snprintf(m, sizeof m, "job %d reads job %d's rows (expected %d's)", i, reads[b][i],
                              i == kTwin ? kTwinOf : i);
                return fail_at(m);
            }
    const uint8_t *a = h.data(), *b = a + set_bytes, *c = b + set_bytes;
    for (int i = 0; i < x3::kMaxMvJobs; ++i) {
        const size_t qn = (size_t) spec[i].rows * (size_t) spec[i].k, dn = qn / 32 * sizeof(float);
        const size_t lo[4] = {qat[i] - kGuard, qat[i] + qn, dat[i] - kGuard, dat[i] + dn};   // the guards
        for (size_t g : lo)
            for (size_t o = g; o < g + kGuard; ++o)
                if (a[o] != 0xA5 || b[o] != 0xA5 || c[o] != 0x5A) {
                    char m[160];
                    std::snprintf(m, sizeof m, "job %d: a guard byte at %zu written (per-job 0x%02x, batched 0x%02x / 0x%02x)",
                                  i, o, a[o], b[o], c[o]);
                    return fail_at(m);
                }
        const int j = reads[1][i];   // (what job i reads in the batched sets: job j's rows, at its offsets)
        const size_t body[2][2] = {{qat[i], qat[i] + qn}, {dat[i], dat[i] + dn}};
        const size_t at[2] = {qat[j], dat[j]};
        for (int r = 0; r < 2; ++r)
            for (size_t o = body[r][0]; o < body[r][1]; ++o) {
                const size_t ob = at[r] + (o - body[r][0]);
                if (a[o] != b[ob] || a[o] != c[ob]) {
                    char m[160];
                    std::snprintf(m, sizeof m, "job %d: %s byte %zu (per-job 0x%02x, batched 0x%02x / 0x%02x)", i,
                                  r == 0 ? "an int8" : "a scale", o - body[r][0], a[o], b[ob], c[ob]);
                    return fail_at(m);
                }
                if (j != i && (b[o] != 0xA5 || c[o] != 0x5A)) {   // (a sharing job's own rows: never written)
                    char m[160];
                    std::snprintf(m, sizeof m, "job %d (sharing job %d's rows): its own %s byte %zu written", i, j,
                                  r == 0 ? "int8" : "scale", o - body[r][0]);
                    return fail_at(m);
                }
            }
    }
    std::fprintf(stderr,
                 "ds4: CUDA%d Q8 batch check: 8 jobs (k 32 .. 5120, rows 1 .. 16, strided, one input read with two k, "
                 "two jobs sharing a rounding): %zu int8 and %zu scales the per-job launches' bit for bit, every guard "
                 "intact (DS4_Q8_CHECK)\n",
                 dev, payload_q, payload_d);
    return true;
}

int launch_errors() { return g_errors.load(std::memory_order_relaxed); }

void q8_round(const float* x, int k, int8_t* q, float* d, cudaStream_t s) {
    if (q8_batch()) {
        Q8Jobs J;
        J.j[0] = {x, (size_t) k, q, d, k, 1};
        q8_quant_jobs(J, 1, s);
    } else
        q8_quant(x, (size_t) k, 1, k, q, d, s);
}

// (the low-bit types ds4_iq's GEMVs take: the routed experts' and the dense matrices' beside them - Q6_K, as
//  orcarouter's mixes and maya-mm3's GGUFs hold some; a routed expert's own: expert_type_ok)
size_t type_row_bytes(int type, int64_t k) {
    if (type == kTypeMXFP4) return k % kMxBlock == 0 ? mx_row_bytes(k) : 0;
    return iq::mv_supported(type) && k % iq::block_values(type) == 0 ? iq::row_bytes(type, k) : 0;
}
int type_block(int type) { return type == kTypeMXFP4 ? kMxBlock : iq::mv_supported(type) ? iq::block_values(type) : 0; }
bool expert_type_ok(int type) { return type == kTypeMXFP4 || iq::supported(type); }
const char* type_name(int type) { return type == kTypeMXFP4 ? "MXFP4" : iq::name(type); }

namespace {
size_t row_stride(const x3::Mat& m) {   // values (BF16) or bytes (MXFP4, Q8_0, a low-bit type) a row
    if (m.ld > 0) return (size_t) m.ld;
    return m.fmt == kFmtMXFP4 ? mx_row_bytes(m.k)
           : m.fmt == kFmtQ8_0 ? q8_row_bytes(m.k)
           : m.fmt == kFmtGG   ? type_row_bytes(m.gg_type, m.k)
                               : (size_t) m.k;
}
// whether a layer's gate / up (gu) or its down is MXFP4 - these kernels - or a low-bit type (ds4_iq's)
bool mx_role(const x3::ExpertLayout& lay, bool gu) { return expert_type(lay, gu ? 0 : 2) == kTypeMXFP4; }
// a layer's low-bit gate / up (gu) or down as ds4_iq's arguments - the other role may be MXFP4, which takes these
// kernels (false: the role's types not covered - printed)
bool iq_args(const x3::ExpertLayout& lay, int n_embd, int n_ff, float limit, iq::Args& ia, const char* who, bool gu) {
    ia = iq::Args{};
    for (int r = 0; r < 3; ++r) ia.tr[r] = lay.trellis[r];
    ia.n_embd = n_embd;
    ia.n_ff = n_ff;
    ia.limit = limit;
    ia.tg = expert_type(lay, 0);
    ia.td = expert_type(lay, 2);
    // (a row of the role's inputs in whole blocks: 256 values, IQ4_NL's 32)
    if (gu ? ia.tg != expert_type(lay, 1) || !iq::supported(ia.tg) || n_embd % iq::block_values(ia.tg) != 0
           : !iq::supported(ia.td) || n_ff % iq::block_values(ia.td) != 0) {
        std::fprintf(stderr, "ds4 gguf %s: expert types %s / %s / %s (n_embd %d, n_ff %d) not covered\n", who,
                     type_name(expert_type(lay, 0)), type_name(expert_type(lay, 1)), type_name(expert_type(lay, 2)),
                     n_embd, n_ff);
        g_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}
template <int NC>
void mv_launch(int f, const BfJobs& J, int nj, int max_n, bool vec, cudaStream_t s) {
    const dim3 grid((unsigned) ((max_n + kWarps - 1) / kWarps), (unsigned) nj);
    if (f == 1) {
        if (vec) pdl_launch(mx_mv_kernel<NC, true, false>, grid, kThreads, 0, s, J);
        else pdl_launch(mx_mv_kernel<NC, false, false>, grid, kThreads, 0, s, J);
    } else if (f == 2) {   // (vec here: every row 4-byte aligned, so staged)
        if (vec) pdl_launch(q8_mv_kernel<NC, true>, grid, kThreads, 0, s, J);
        else pdl_launch(q8_mv_kernel<NC, false>, grid, kThreads, 0, s, J);
    } else {
        if (vec) pdl_launch(bf16_mv_kernel<NC, true>, grid, kThreads, 0, s, J);
        else pdl_launch(bf16_mv_kernel<NC, false>, grid, kThreads, 0, s, J);
    }
}
void mv_set(int f, const BfJobs& J, int nj, int max_n, int max_c, bool vec, cudaStream_t s) {
    if (nj == 0) return;
    if (max_c <= 1) mv_launch<1>(f, J, nj, max_n, vec, s);
    else if (max_c <= 2) mv_launch<2>(f, J, nj, max_n, vec, s);
    else if (max_c <= 4) mv_launch<4>(f, J, nj, max_n, vec, s);
    else if (max_c <= 8) mv_launch<8>(f, J, nj, max_n, vec, s);
    else mv_launch<16>(f, J, nj, max_n, vec, s);
}
}  // namespace

x3::Mat view(const x3::Mat& m, int c0, int nc) {
    x3::Mat v = m;
    const size_t ld = row_stride(m);
    v.ld = (int) ld;
    v.raw = m.fmt != kFmtBF16 ? (const void*) ((const uint8_t*) m.raw + (size_t) c0 * ld)
                              : (const void*) ((const uint16_t*) m.raw + (size_t) c0 * ld);
    v.n = nc;
    return v;
}

bool mv(const x3::MvJob* jobs, int n, cudaStream_t s) { return mv(jobs, n, s, nullptr); }

bool mv(const x3::MvJob* jobs, int n, cudaStream_t s, const PreQ8* pre) {
    if (n <= 0 || n > x3::kMaxMvJobs) return false;
    BfJobs J[3] = {};   // the BF16 jobs, the MXFP4 ones, the Q8_0 ones: a launch each
    int nj[3] = {0, 0, 0}, max_n[3] = {0, 0, 0}, max_c[3] = {1, 1, 1};
    bool vec[3] = {true, true, true};
    iq::MvJob IJ[x3::kMaxMvJobs];   // the low-bit ones (ds4_iq: a launch a type)
    int nij = 0;
    Q8Round QR;   // (the jobs' int8 inputs: one launch before the GEMVs, an input read by several jobs rounded once)
    QR.batch = q8_batch();
    QR.share = q8_share();
    for (int i = 0; i < n; ++i) {
        const x3::MvJob& q = jobs[i];
        const x3::Mat& m = *q.m;
        if (m.fmt == kFmtGG) {   // the activations rounded to int8 (job i's rows of the scratch), then ds4_iq's GEMV
            Scratch* sc = scratch(s);
            if (sc == nullptr || m.raw == nullptr || q.xf == nullptr || q.y == nullptr || q.ncol < 1 ||
                q.ncol > x3::kMaxMvCols || m.k > kMaxQK || type_row_bytes(m.gg_type, m.k) == 0) {
                std::fprintf(stderr, "ds4 gguf mv: job %d unsupported (type %s k %d n %d)\n", i, type_name(m.gg_type), m.k,
                             m.n);
                g_errors.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            const size_t at = (size_t) i * x3::kMaxMvCols * kMaxQK;
            // (per 32: a dense matrix - the CPU lane computes only experts, so Q8_K's rounding, a scale per 256, would only
            //  lose precision here)
            iq::MvJob& b = IJ[nij++];
            QR.add(q.xf, q.ncol > 1 ? q.xf_cs : (size_t) m.k, q.ncol, m.k, sc->mq + at, sc->md + at / 32, b.xq, b.xd,
                   s);
            b.w = (const uint8_t*) m.raw;
            b.ld = row_stride(m);
            b.type = m.gg_type;
            b.k = m.k;
            b.n = m.n;
            b.ncol = q.ncol;
            b.y = q.y;
            b.ycs = q.y_cs;
            b.bias = q.bias;
            b.alpha = q.alpha;
            continue;
        }
        const int f = m.fmt == kFmtMXFP4 ? 1 : m.fmt == kFmtQ8_0 ? 2 : 0;
        if ((m.fmt != kFmtBF16 && m.fmt != kFmtMXFP4 && m.fmt != kFmtQ8_0) || m.raw == nullptr || q.xf == nullptr ||
            q.y == nullptr || q.ncol < 1 || q.ncol > x3::kMaxMvCols || m.k <= 0 || m.n <= 0 ||
            (f != 0 && m.k % kMxBlock != 0)) {
            std::fprintf(stderr, "ds4 gguf mv: job %d unsupported (fmt %d k %d n %d, %d activations)\n", i, m.fmt, m.k,
                         m.n, q.ncol);
            g_errors.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        BfJob& b = J[f].j[nj[f]++];
        b.w = m.raw;
        b.ld = row_stride(m);
        b.x = q.xf;
        b.xcs = q.xf_cs;
        b.y = q.y;
        b.ycs = q.y_cs;
        b.bias = q.bias;
        b.alpha = q.alpha;
        b.k = m.k;
        b.n = m.n;
        b.ncol = q.ncol;
        max_n[f] = std::max(max_n[f], m.n);
        max_c[f] = std::max(max_c[f], q.ncol);
        const bool xa = ((uintptr_t) b.x & 15u) == 0 && (q.ncol == 1 || b.xcs % 4 == 0);
        if (f == 2 && pre != nullptr && q.xf == pre->x && m.k == pre->k && q.ncol == pre->rows &&
            (q.ncol == 1 || q.xf_cs == (size_t) m.k)) {
            b.xq = pre->q;   // (DS4_HC_FUSE: x rounded where it was made, q8_quant_block's bits)
            b.xd = pre->d;
            vec[f] = vec[f] && b.ld % 4 == 0 && ((uintptr_t) b.w & 3u) == 0 && (m.k / 32) % 2 == 0;
        } else if (f == 2) {   // the activations rounded to int8 first (each job its own rows of the scratch)
            Scratch* sc = scratch(s);
            if (sc == nullptr || m.k > kMaxQK) {
                fail("mv: Q8_0 inputs past the scratch");
                return false;
            }
            const size_t at = (size_t) i * x3::kMaxMvCols * kMaxQK;
            QR.add(q.xf, q.ncol > 1 ? q.xf_cs : (size_t) m.k, q.ncol, m.k, sc->mq + at, sc->md + at / 32, b.xq, b.xd,
                   s);
            vec[f] = vec[f] && b.ld % 4 == 0 && ((uintptr_t) b.w & 3u) == 0 && (m.k / 32) % 2 == 0;
        } else {
            vec[f] = vec[f] && xa && (f != 0 || (m.k % 8 == 0 && b.ld % 8 == 0 && ((uintptr_t) b.w & 15u) == 0));
        }
    }
    QR.flush(s);
    for (int f = 0; f < 3; ++f) mv_set(f, J[f], nj[f], max_n[f], max_c[f], vec[f], s);
    if (nij > 0 && !iq::mv(IJ, nij, s)) {
        fail("mv: a low-bit job not covered");
        return false;
    }
    check("mv");
    return true;
}

void rows(const x3::Mat& m, uint16_t* dst, bool bf16, cudaStream_t s) {
    if (m.fmt == kFmtGG && m.raw != nullptr && !bf16) {   // (a low-bit dense matrix: ds4_iq's FP16 values of it)
        if (!iq::f16_rows(m.gg_type, (const uint8_t*) m.raw, m.k, m.n, dst, m.k, s)) fail("rows: a low-bit type not covered");
        check("rows");
        return;
    }
    if ((m.fmt != kFmtBF16 && m.fmt != kFmtQ8_0) || m.raw == nullptr) {
        fail("rows: not a BF16 or Q8_0 matrix");
        return;
    }
    const size_t ld = row_stride(m);
    const size_t total = (size_t) m.n * (size_t) m.k;
    const unsigned blocks = (unsigned) std::min<size_t>((total + 255) / 256, 65535);
    if (m.fmt == kFmtQ8_0) pdl_launch(q8_rows_kernel, blocks, 256, 0, s, (const uint8_t*) m.raw, ld, m.k, m.n, dst, bf16);
    else pdl_launch(bf16_rows_kernel, blocks, 256, 0, s, (const uint16_t*) m.raw, ld, m.k, m.n, dst, bf16);
    check("rows");
}

namespace {
// (staged when the matrix's rows are 4-byte aligned: 17 nb % 4 == 0; the blobs' offsets are)
template <int NE>
struct GateUpK {
    static void run(const MxArgs& m, dim3 grid, cudaStream_t s) {
        if (mx_row_bytes(m.n_embd) % 4 == 0) pdl_launch(mx_gate_up_kernel<NE, true>, grid, kThreads, 0, s, m);
        else pdl_launch(mx_gate_up_kernel<NE, false>, grid, kThreads, 0, s, m);
    }
};
template <int NE>
struct DownK {
    static void run(const MxArgs& m, dim3 grid, cudaStream_t s) {
        if (mx_row_bytes(m.n_ff) % 4 == 0) pdl_launch(mx_down_kernel<NE, true>, grid, kThreads, 0, s, m);
        else pdl_launch(mx_down_kernel<NE, false>, grid, kThreads, 0, s, m);
    }
};
template <template <int> class Kern>
void launch_groups(const MxArgs& m, int rows, int groups, int ne_max, cudaStream_t s) {
    const dim3 grid((unsigned) ((rows + kWarps - 1) / kWarps), (unsigned) groups);
    if (ne_max <= 1) Kern<1>::run(m, grid, s);
    else if (ne_max <= 4) Kern<4>::run(m, grid, s);
    else if (ne_max <= 8) Kern<8>::run(m, grid, s);
    else Kern<16>::run(m, grid, s);
}
MxArgs mx_args(const x3::ExpertLayout& lay, int n_embd, int n_ff, float limit) {
    MxArgs m{};
    for (int r = 0; r < 3; ++r) m.tr[r] = lay.trellis[r];
    m.n_embd = n_embd;
    m.n_ff = n_ff;
    m.limit = limit;
    return m;
}
}  // namespace

void act_q8k(const float* x, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    q8k_quant(x, (size_t) k, rows, k, q, d, s);
    check("act_q8k");
}

void moe_gate_up(const x3::MoeArgs& a, const float* x, float* h, cudaStream_t s) {
    if (!mx_geometry("moe_gate_up", a.k, a.n_embd, a.n_ff) || h == nullptr) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return;
    if (!mx_role(a.lay, true)) {   // a low-bit gate / up: ds4_iq's kernels
        iq::Args ia;
        if (!iq_args(a.lay, a.n_embd, a.n_ff, a.limit, ia, "moe_gate_up", true)) return;
        ia.plan_ptr = a.plan_ptr;
        ia.h = h;
        ia.xq = sc->xq;
        ia.xd = sc->xd;
        iq_act_quant(x, (size_t) a.n_embd, 1, a.n_embd, sc->xq, sc->xd, s);
        iq::gate_up(ia, a.k, 1, s);
        check("moe_gate_up");
        return;
    }
    MxArgs m = mx_args(a.lay, a.n_embd, a.n_ff, a.limit);
    m.plan_ptr = a.plan_ptr;
    m.x = x;
    m.h = h;
    m.xq = sc->xq;
    m.xd = sc->xd;
    q8_quant(x, (size_t) a.n_embd, 1, a.n_embd, sc->xq, sc->xd, s);
    launch_groups<GateUpK>(m, a.n_ff, a.k, 1, s);
    check("moe_gate_up");
}

void moe_down(const x3::MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    if (!mx_geometry("moe_down", a.k, a.n_embd, a.n_ff) || h == nullptr) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return;
    MxArgs m = mx_args(a.lay, a.n_embd, a.n_ff, a.limit);
    m.plan_ptr = a.plan_ptr;
    m.h = const_cast<float*>(h);
    m.part = sc->part;
    m.hq = sc->hq;
    m.hd = sc->hd;
    const bool mx = mx_role(a.lay, false);
    (mx ? q8_quant : iq_act_quant)(h, (size_t) a.n_ff, a.k, a.n_ff, sc->hq, sc->hd, s);
    if (!mx) {   // a low-bit down: ds4_iq's, then the same combine
        iq::Args ia;
        if (!iq_args(a.lay, a.n_embd, a.n_ff, a.limit, ia, "moe_down", false)) return;
        ia.plan_ptr = a.plan_ptr;
        ia.part = sc->part;
        ia.hq = sc->hq;
        ia.hd = sc->hd;
        iq::down(ia, a.k, 1, s);
    } else {
        launch_groups<DownK>(m, a.n_embd, a.k, 1, s);
    }
    CombArgs c{};
    c.plan_ptr = a.plan_ptr;
    c.plan_w = a.plan_w;
    c.part = sc->part;
    c.cpu_part = a.cpu_part;
    c.cpu_stride = (size_t) a.n_embd;
    c.cpu_flag = a.cpu_flag;
    c.sh_out = sh_out;
    c.out = out;
    c.out_stride = (size_t) a.n_embd;
    c.k = a.k;
    c.n_embd = a.n_embd;
    pdl_launch(mx_combine_kernel, dim3((unsigned) ((a.n_embd + 127) / 128), 1), 128, 0, s, c);
    check("moe_down");
}

bool win_experts(const x3::WinArgs& a, cudaStream_t s) {
    if (a.T < 1 || a.T > kMaxEnt || !mx_geometry("win_experts", a.k, a.n_embd, a.n_ff) || a.h == nullptr ||
        a.part == nullptr || a.out == nullptr) {
        std::fprintf(stderr, "ds4 gguf win_experts: not covered (T %d)\n", a.T);
        return false;
    }
    MxArgs m = mx_args(a.lay, a.n_embd, a.n_ff, a.limit);
    m.grp_ptr = a.grp_ptr;
    m.grp_start = a.grp_start;
    m.grp_ent = a.grp_ent;
    m.n_groups = a.n_groups;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    m.x = a.x;
    m.x_stride = a.x_stride;
    m.h = a.h;
    m.part = a.part;
    m.xq = sc->xq;
    m.xd = sc->xd;
    m.hq = sc->hq;
    m.hd = sc->hd;
    const int groups = a.T * a.k;   // at most (the blocks past the real count return at once)
    const bool mx_gu = mx_role(a.lay, true), mx_d = mx_role(a.lay, false);
    iq::Args ia;
    if (!mx_gu || !mx_d) {   // a low-bit role: ds4_iq's kernels on the same groups and scratch
        if ((!mx_gu && !iq_args(a.lay, a.n_embd, a.n_ff, a.limit, ia, "win_experts", true)) ||
            (!mx_d && !iq_args(a.lay, a.n_embd, a.n_ff, a.limit, ia, "win_experts", false)))
            return false;
        ia.grp_ptr = a.grp_ptr;
        ia.grp_start = a.grp_start;
        ia.grp_ent = a.grp_ent;
        ia.n_groups = a.n_groups;
        ia.h = a.h;
        ia.part = a.part;
        ia.xq = sc->xq;
        ia.xd = sc->xd;
        ia.hq = sc->hq;
        ia.hd = sc->hd;
    }
    (mx_gu ? q8_quant : iq_act_quant)(a.x, a.x_stride, a.T, a.n_embd, sc->xq, sc->xd, s);
    if (mx_gu) launch_groups<GateUpK>(m, a.n_ff, groups, a.T, s);
    else iq::gate_up(ia, groups, a.T, s);
    // (rows of picks not on the device: unused)
    (mx_d ? q8_quant : iq_act_quant)(a.h, (size_t) a.n_ff, a.T * 8, a.n_ff, sc->hq, sc->hd, s);
    if (mx_d) launch_groups<DownK>(m, a.n_embd, groups, a.T, s);
    else iq::down(ia, groups, a.T, s);
    if (a.wait_ptr != nullptr) pdl_launch(mx_wait_kernel, 1, 32, 0, s, a.wait_ptr, a.wait_val);
    CombArgs c{};
    c.plan_ptr = a.plan_ptr;
    c.plan_w = a.plan_w;
    c.part = a.part;
    c.cpu_part = a.cpu_part;
    c.cpu_stride = a.cpu_stride != 0 ? a.cpu_stride : (size_t) a.n_embd;
    c.cpu_flag = a.cpu_flag;
    c.sh_out = a.sh_out;
    c.out = a.out;
    c.out_stride = a.out_stride;
    c.k = a.k;
    c.n_embd = a.n_embd;
    pdl_launch(mx_combine_kernel, dim3((unsigned) ((a.n_embd + 127) / 128), (unsigned) a.T), 128, 0, s, c);
    check("win_experts");
    return true;
}

namespace {
iq::SetArgs iq_set(const x3::SetArgs& sa) {   // a set as ds4_iq's arguments (the inputs / outputs and types to fill in)
    iq::SetArgs q;
    q.wbase = sa.wbase;
    q.stride = sa.stride;
    for (int r = 0; r < 3; ++r) q.tr[r] = sa.lay.trellis[r];
    q.bounds = sa.bounds;
    q.row0 = sa.row0;
    q.max_rows = sa.max_rows;
    q.n_exp = sa.n_exp;
    q.n_embd = sa.n_embd;
    q.n_ff = sa.n_ff;
    q.limit = sa.limit;
    q.skip_ge = sa.skip_ge;
    return q;
}
}  // namespace

void set_gate_up(const x3::SetArgs& sa, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s, const uint16_t* x16) {
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    if (!mx_geometry("set_gate_up", 1, sa.n_embd, sa.n_ff)) return;
    if (!mx_role(sa.lay, true)) {   // a low-bit gate / up: ds4_iq's sets
        iq::Args ia;
        if (!iq_args(sa.lay, sa.n_embd, sa.n_ff, sa.limit, ia, "set_gate_up", true)) return;
        iq::SetArgs q = iq_set(sa);
        q.xf = x;
        q.xh = x16;   // (the tensor-core tiles' FP16 rows; the light kernels keep xf)
        q.idx = row_idx;
        q.ldx = ldx;
        q.h = h16;
        q.ldo = ldh;
        q.tg = ia.tg;
        q.td = ia.td;
        if (!iq::set_gate_up(q, s)) fail("set_gate_up: not covered");
        check("set_gate_up");
        return;
    }
    int skip = 0;
    {   // the experts of at most 16 rows: the light kernels (each weight read once; ds4_iq.cuh), the others the tiles
        iq::SetArgs q = iq_set(sa);
        q.xf = x;
        q.idx = row_idx;
        q.ldx = ldx;
        q.h = h16;
        q.ldo = ldh;
        q.tg = q.td = kTypeMXFP4;
        if (iq::light_gate_up(q, s)) {
            if (sa.max_rows <= 16) {
                check("set_gate_up");
                return;
            }
            skip = 16;
        }
    }
    SetK k{};
    k.skip_le = skip;
    k.wbase = sa.wbase;
    k.stride = sa.stride;
    for (int r = 0; r < 3; ++r) k.tr[r] = sa.lay.trellis[r];
    k.bounds = sa.bounds;
    k.row0 = sa.row0;
    k.n_embd = sa.n_embd;
    k.n_ff = sa.n_ff;
    k.limit = sa.limit;
    k.xf = x;
    k.idx = row_idx;
    k.xh = (const __half*) x16;
    k.ldx = ldx;
    k.h = (__half*) h16;
    k.ldo = ldh;
    k.xh_vec = x16 != nullptr && ((uintptr_t) x16 & 15) == 0 && (ldx & 7) == 0;
    const dim3 grid((unsigned) ((sa.max_rows + BM - 1) / BM), (unsigned) (sa.n_ff / BN), (unsigned) sa.n_exp);
#if !defined(STRATA_USE_HIP)
    if (tc_ok()) pdl_launch(mx_set_tc_kernel<true>, grid, kThreads, 0, s, k);
#else
    if (tc_ok()) pdl_launch(mx_set_wmma_kernel<true>, grid, kThreads, 0, s, k);
#endif
    else
        pdl_launch(mx_set_kernel<true>, grid, kThreads, 0, s, k);
    check("set_gate_up");
}

void set_down(const x3::SetArgs& sa, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s) {
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    if (!mx_geometry("set_down", 1, sa.n_embd, sa.n_ff)) return;
    if (!mx_role(sa.lay, false)) {   // a low-bit down: ds4_iq's sets
        iq::Args ia;
        if (!iq_args(sa.lay, sa.n_embd, sa.n_ff, sa.limit, ia, "set_down", false)) return;
        iq::SetArgs q = iq_set(sa);
        q.xh = h16;
        q.ldx = ldh;
        q.y = y;
        q.ldo = ldy;
        q.tg = ia.tg;
        q.td = ia.td;
        if (!iq::set_down(q, s)) fail("set_down: not covered");
        check("set_down");
        return;
    }
    int skip = 0;
    {   // (as set_gate_up's)
        iq::SetArgs q = iq_set(sa);
        q.xh = h16;
        q.ldx = ldh;
        q.y = y;
        q.ldo = ldy;
        q.tg = q.td = kTypeMXFP4;
        if (iq::light_down(q, s)) {
            if (sa.max_rows <= 16) {
                check("set_down");
                return;
            }
            skip = 16;
        }
    }
    SetK k{};
    k.skip_le = skip;
    k.wbase = sa.wbase;
    k.stride = sa.stride;
    for (int r = 0; r < 3; ++r) k.tr[r] = sa.lay.trellis[r];
    k.bounds = sa.bounds;
    k.row0 = sa.row0;
    k.n_embd = sa.n_embd;
    k.n_ff = sa.n_ff;
    k.xh = (const __half*) h16;
    k.ldx = ldh;
    k.y = y;
    k.ldo = ldy;
    k.xh_vec = ((uintptr_t) h16 & 15) == 0 && (ldh & 7) == 0;
    const dim3 grid((unsigned) ((sa.max_rows + BM - 1) / BM), (unsigned) (sa.n_embd / BN), (unsigned) sa.n_exp);
#if !defined(STRATA_USE_HIP)
    if (tc_ok()) pdl_launch(mx_set_tc_kernel<false>, grid, kThreads, 0, s, k);
#else
    if (tc_ok()) pdl_launch(mx_set_wmma_kernel<false>, grid, kThreads, 0, s, k);
#endif
    else
        pdl_launch(mx_set_kernel<false>, grid, kThreads, 0, s, k);
    check("set_down");
}

}  // namespace strata::ds4::gguf
