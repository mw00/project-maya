// src/kernels/cuda/q35_kernels.cu - see include/strata/kernels/q35_kernels.hpp.
#include "strata/kernels/q35_kernels.hpp"
#include "strata/kernels/dense_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// sum over the block (blockDim.x a multiple of 32); every thread gets it
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float s = lane < nw ? sh[lane] : 0.0f;
    return warp_sum(s);
}

// the GDN recurrent state's storage: FP16, the arithmetic F32 in registers (kQ35StateBytes; -DQ35_STATE_F32 for F32).
// Chain 292: the same 512 greedy tokens at 2K / 35K / 68K, DFlash2 35K 84.35 -> 85.65, MTP 68K 39.0 -> 39.7 tok/s
#ifndef Q35_STATE_F32
#define GST_LD(p, i) __half2float(((const __half*) (p))[i])
#define GST_ST(p, i, v) (((__half*) (p))[i] = __float2half(v))
#else
#define GST_LD(p, i) ((p)[i])
#define GST_ST(p, i, v) ((p)[i] = (v))
#endif

// ------------------------------------------------------------------------------------------------ norms
__global__ void rms_rows_kernel(const float* __restrict__ x, int ldx, const float* __restrict__ w,
                                float* __restrict__ of, __half* __restrict__ oh, int cols, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float acc = 0.0f;
    // (4 strided loads in flight a thread, summed in the same order: the same bits, without a load latency an element)
    const int bd = blockDim.x;
    for (int i0 = threadIdx.x; i0 < cols; i0 += 4 * bd) {
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) v[k] = i0 + k * bd < cols ? xr[i0 + k * bd] : 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (i0 + k * bd < cols) acc += v[k] * v[k];
    }
    const float scale = rsqrtf(block_sum(acc, sh) / (float) cols + eps);
#pragma unroll 4
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = xr[i] * scale * (w ? w[i] : 1.0f);   // (no weight: a plain rms norm)
        if (of) of[(size_t) blockIdx.x * cols + i] = v;
        if (oh) oh[(size_t) blockIdx.x * cols + i] = __float2half(v);
    }
}

// x += a first (the residual add), then rms_rows_kernel's arithmetic on the new x (the same bits as the two kernels)
__global__ void add_rms_rows_kernel(float* __restrict__ x, int ldx, const float* __restrict__ a,
                                    const float* __restrict__ w, float* __restrict__ of, __half* __restrict__ oh,
                                    int cols, float eps) {
    __shared__ float sh[32];
    float* xr = x + (size_t) blockIdx.x * ldx;
    const float* ar = a + (size_t) blockIdx.x * cols;
    float acc = 0.0f;
    const int bd = blockDim.x;
    for (int i0 = threadIdx.x; i0 < cols; i0 += 4 * bd) {   // (4 elements in flight a thread, the same order)
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) v[k] = i0 + k * bd < cols ? xr[i0 + k * bd] + ar[i0 + k * bd] : 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (i0 + k * bd < cols) { xr[i0 + k * bd] = v[k]; acc += v[k] * v[k]; }
    }
    const float scale = rsqrtf(block_sum(acc, sh) / (float) cols + eps);
#pragma unroll 4
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = xr[i] * scale * (w ? w[i] : 1.0f);
        if (of) of[(size_t) blockIdx.x * cols + i] = v;
        if (oh) oh[(size_t) blockIdx.x * cols + i] = __float2half(v);
    }
}

// SwiGLU (dense_kernels.cu's arithmetic) writing F32 and FP16 rows
__global__ void swiglu_h_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ o,
                                __half* __restrict__ oh, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[i];
    const float v = (a / (1.0f + __expf(-a))) * u[i];
    o[i] = v;
    oh[i] = __float2half(v);
}

__global__ void to_half_kernel(const float* __restrict__ s, __half* __restrict__ d, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __float2half(s[i]);
}

// ------------------------------------------------------------------------------------------------ GDN
// a thread per channel: the window [s0 s1 s2 | x_t] slides over the T tokens (native_gdn_preprocess.cu's conv_silu)
__global__ void conv_seq_kernel(float* __restrict__ st, const float* __restrict__ raw, int ld,
                                const float* __restrict__ w, float* __restrict__ out, int T, int C, int commit) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h0 = st[c * 3], h1 = st[c * 3 + 1], h2 = st[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float x = raw[(size_t) t * ld + c];
        float sum = 0.0f;
        sum += h0 * w0;
        sum += h1 * w1;
        sum += h2 * w2;
        sum += x * w3;
        sum = __fadd_rn(sum, 0.0f);
        out[(size_t) t * ld + c] = sum / (1.0f + expf(-sum));
        h0 = h1; h1 = h2; h2 = x;
    }
    if (commit) { st[c * 3] = h0; st[c * 3 + 1] = h1; st[c * 3 + 2] = h2; }
}

// the same convolution a thread a (channel, token) - a prompt chunk's tokens in parallel (conv_seq_kernel walked them
// one after another a channel); each output's arithmetic as conv_seq_kernel's, so the same bits.  The state is
// read here and moved on by conv_commit_kernel afterwards
__global__ void conv_par_kernel(const float* __restrict__ st, const float* __restrict__ raw, int ld,
                                const float* __restrict__ w, float* __restrict__ out, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    auto in = [&](int u) { return u >= 0 ? raw[(size_t) u * ld + c] : st[c * 3 + 3 + u]; };   // (u -3..-1: the state)
    const float h0 = in(t - 3), h1 = in(t - 2), h2 = in(t - 1), x = raw[(size_t) t * ld + c];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float sum = 0.0f;
    sum += h0 * w0;
    sum += h1 * w1;
    sum += h2 * w2;
    sum += x * w3;
    sum = __fadd_rn(sum, 0.0f);
    out[(size_t) t * ld + c] = sum / (1.0f + expf(-sum));
}
__global__ void conv_commit_kernel(float* __restrict__ st, const float* __restrict__ raw, int ld, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int u = T - 3 + i;
        h[i] = u >= 0 ? raw[(size_t) u * ld + c] : st[c * 3 + 3 + u];
    }
    st[c * 3] = h[0]; st[c * 3 + 1] = h[1]; st[c * 3 + 2] = h[2];
}

// block (row, token), 128 threads: x / sqrt(sum x^2 + eps), as native_gdn_preprocess.cu's l2_norm computes it
__global__ void qk_l2_kernel(float* __restrict__ x, int ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (size_t) blockIdx.y * ld + (size_t) blockIdx.x * 128;
    const float v = r[threadIdx.x];
    const float sum = block_sum(v * v, sh);
    const float scale = rsqrtf(sum / 128.0f + eps / 128.0f);
    r[threadIdx.x] = __fmaf_rn(__fmul_rn(scale, v), 1.0f / sqrtf(128.0f), 0.0f);
}

// conv_seq + qk_l2 + gate_beta in one launch (q35_gdn_prep): block b < C / 128 the conv of channels 128 b .. (one q / k
// head row when b < n_qk: its L2 norm per token in the block, the same arithmetic as qk_l2_kernel), the rest gate_beta's
// elements, 128 a block
__global__ void __launch_bounds__(128) gdn_prep_kernel(float* __restrict__ st, const float* __restrict__ raw, int ld,
                                                       const float* __restrict__ w, float* __restrict__ out, int T, int C,
                                                       int commit, int n_qk, float eps, float* __restrict__ a,
                                                       float* __restrict__ bt, const float* __restrict__ dt,
                                                       const float* __restrict__ ssm_a, int h_v, int ldg) {
    __shared__ float sh[32];
    const int nb = C / 128;
    if ((int) blockIdx.x >= nb) {
        const int k = ((int) blockIdx.x - nb) * 128 + threadIdx.x;
        if (k >= T * h_v) return;
        const int h = k % h_v, i = (k / h_v) * ldg + h;
        const float v = __fadd_rn(a[i], dt[h]);
        const float sp = v > 20.0f ? v : logf(1.0f + expf(v));
        a[i] = sp * ssm_a[h];
        bt[i] = 1.0f / (1.0f + expf(-bt[i]));
        return;
    }
    const int c = blockIdx.x * 128 + threadIdx.x;
    const bool qk = (int) blockIdx.x < n_qk;
    float h0 = st[c * 3], h1 = st[c * 3 + 1], h2 = st[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float x = raw[(size_t) t * ld + c];
        float sum = 0.0f;
        sum += h0 * w0;
        sum += h1 * w1;
        sum += h2 * w2;
        sum += x * w3;
        sum = __fadd_rn(sum, 0.0f);
        float o = sum / (1.0f + expf(-sum));
        if (qk) {
            const float s2 = block_sum(o * o, sh);
            const float scale = rsqrtf(s2 / 128.0f + eps / 128.0f);
            o = __fmaf_rn(__fmul_rn(scale, o), 1.0f / sqrtf(128.0f), 0.0f);
            __syncthreads();   // (sh reused by the next token's sum)
        }
        out[(size_t) t * ld + c] = o;
        h0 = h1; h1 = h2; h2 = x;
    }
    if (commit) { st[c * 3] = h0; st[c * 3 + 1] = h1; st[c * 3 + 2] = h2; }
}

__global__ void gate_beta_kernel(float* __restrict__ a, float* __restrict__ b, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, int n, int h_v, int ld) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int h = k % h_v, i = (k / h_v) * ld + h;
    const float v = __fadd_rn(a[i], dt[h]);
    const float sp = v > 20.0f ? v : logf(1.0f + expf(v));
    a[i] = sp * ssm_a[h];
    b[i] = 1.0f / (1.0f + expf(-b[i]));
}

// a value head's key head: GGUF files tile them (h % h_k, llama.cpp's converter reorders the value heads), checkpoints
// with the model's own order group them (h / (h_v / h_k): q35_set_gdn_grouped)
__constant__ int c_gdn_group;
__device__ __forceinline__ int gdn_kh(int h, int h_k) { return c_gdn_group > 0 ? h / c_gdn_group : h % h_k; }

// block = value head h, thread = state column j; the column's 128 state values in registers
__global__ void __launch_bounds__(128) gdn_seq_kernel(float* __restrict__ state, const float* __restrict__ x, int ld,
                                                      const float* __restrict__ g, const float* __restrict__ b, int ldg,
                                                      float* __restrict__ o, int T, int h_k, int h_v, int commit) {
    constexpr int S = 128;
    const int h = blockIdx.x, j = threadIdx.x, kh = gdn_kh(h, h_k);
    __shared__ float sq[S], sk[S];
    float st[S];
#pragma unroll
    for (int i = 0; i < S; ++i) st[i] = GST_LD(state, ((size_t) i * h_v + h) * S + j);
    const float qscale = 1.0f / sqrtf((float) S);
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t) t * ld;
        __syncthreads();
        sq[j] = xr[kh * S + j];
        sk[j] = xr[h_k * S + kh * S + j];
        __syncthreads();
        const float v = xr[2 * h_k * S + h * S + j];
        const float dec = expf(g[(size_t) t * ldg + h]);
        const float beta = b[(size_t) t * ldg + h];
        float skd = 0.0f;
#pragma unroll
        for (int i = 0; i < S; ++i) { st[i] *= dec; skd += st[i] * sk[i]; }
        const float d = (v - skd) * beta;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < S; ++i) { st[i] += sk[i] * d; acc += st[i] * sq[i]; }
        o[(size_t) t * h_v * S + h * S + j] = acc * qscale;
    }
    if (commit) {
#pragma unroll
        for (int i = 0; i < S; ++i) GST_ST(state, ((size_t) i * h_v + h) * S + j, st[i]);
    }
}

// block = (value head h, 32 state columns), 4 warps: warp w holds rows [32w, 32w + 32) of its lane's column (32
// registers, no spill; the one-block-a-head form above needs all 128 and spills), the dot products are summed over
// the warps through shared memory in a fixed order.  4x the blocks of gdn_seq_kernel (192 on the 27B), coalesced
// state loads (a warp reads 32 consecutive columns of a row)
__global__ void __launch_bounds__(128) gdn_seq4_kernel(float* __restrict__ state, const float* __restrict__ x, int ld,
                                                       const float* __restrict__ g, const float* __restrict__ b, int ldg,
                                                       float* __restrict__ o, int T, int h_k, int h_v, int commit) {
    constexpr int S = 128, R = 32;
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, kh = gdn_kh(h, h_k);
    const int j = blockIdx.y * 32 + lane, i0 = w * R;
    __shared__ float sq[S], sk[S], rs[4][32], ra[4][32];
    float st[R];
#pragma unroll
    for (int i = 0; i < R; ++i) st[i] = GST_LD(state, ((size_t) (i0 + i) * h_v + h) * S + j);
    const float qscale = 1.0f / sqrtf((float) S);
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t) t * ld;
        __syncthreads();
        sq[threadIdx.x] = xr[kh * S + threadIdx.x];
        sk[threadIdx.x] = xr[h_k * S + kh * S + threadIdx.x];
        __syncthreads();
        const float v = xr[2 * h_k * S + h * S + j];
        const float dec = expf(g[(size_t) t * ldg + h]);
        const float beta = b[(size_t) t * ldg + h];
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < R; ++i) { st[i] *= dec; p += st[i] * sk[i0 + i]; }
        rs[w][lane] = p;
        __syncthreads();
        const float skd = ((rs[0][lane] + rs[1][lane]) + rs[2][lane]) + rs[3][lane];
        const float d = (v - skd) * beta;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < R; ++i) { st[i] += sk[i0 + i] * d; acc += st[i] * sq[i0 + i]; }
        ra[w][lane] = acc;
        __syncthreads();
        if (w == 0) o[(size_t) t * h_v * S + h * S + j] = (((ra[0][lane] + ra[1][lane]) + ra[2][lane]) + ra[3][lane]) * qscale;
    }
    if (commit) {
#pragma unroll
        for (int i = 0; i < R; ++i) GST_ST(state, ((size_t) (i0 + i) * h_v + h) * S + j, st[i]);
    }
}

// gdn_seq4_kernel with the next token's inputs (q / k / v / gate / beta) loaded into registers during this token's
// sums and q / k double-buffered in shared memory: two barriers a token instead of three, the loads off the critical
// path.  The same arithmetic in the same order (the same bits)
__global__ void __launch_bounds__(128) gdn_seq4p_kernel(float* __restrict__ state, const float* __restrict__ x, int ld,
                                                        const float* __restrict__ g, const float* __restrict__ b, int ldg,
                                                        float* __restrict__ o, int T, int h_k, int h_v, int commit,
                                                        float* __restrict__ d_rec) {
    constexpr int S = 128, R = 32;
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, kh = gdn_kh(h, h_k);
    const int j = blockIdx.y * 32 + lane, i0 = w * R;
    __shared__ float sq[2][S], sk[2][S], rs[4][32], ra[4][32];
    float st[R];
#pragma unroll
    for (int i = 0; i < R; ++i) st[i] = GST_LD(state, ((size_t) (i0 + i) * h_v + h) * S + j);
    const float qscale = 1.0f / sqrtf((float) S);
    // token 0's inputs
    float nv = x[2 * h_k * S + h * S + j], ng = g[h], nb = b[h];
    sq[0][tid] = x[kh * S + tid];
    sk[0][tid] = x[h_k * S + kh * S + tid];
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        const int cb = t & 1;
        const float v = nv, dec = expf(ng), beta = nb;
        float nq = 0.0f, nk = 0.0f;
        if (t + 1 < T) {   // (the next token's, in flight while this one sums)
            const float* xn = x + (size_t) (t + 1) * ld;
            nq = xn[kh * S + tid];
            nk = xn[h_k * S + kh * S + tid];
            nv = xn[2 * h_k * S + h * S + j];
            ng = g[(size_t) (t + 1) * ldg + h];
            nb = b[(size_t) (t + 1) * ldg + h];
        }
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < R; ++i) { st[i] *= dec; p += st[i] * sk[cb][i0 + i]; }
        rs[w][lane] = p;
        __syncthreads();
        sq[cb ^ 1][tid] = nq;   // (the other buffer: last read before the barrier above)
        sk[cb ^ 1][tid] = nk;
        const float skd = ((rs[0][lane] + rs[1][lane]) + rs[2][lane]) + rs[3][lane];
        const float d = (v - skd) * beta;
        if (d_rec && w == 0) d_rec[((size_t) t * h_v + h) * S + j] = d;   // (a verify pass: the update, for commit)
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < R; ++i) { st[i] += sk[cb][i0 + i] * d; acc += st[i] * sq[cb][i0 + i]; }
        ra[w][lane] = acc;
        __syncthreads();
        if (w == 0) o[(size_t) t * h_v * S + h * S + j] = (((ra[0][lane] + ra[1][lane]) + ra[2][lane]) + ra[3][lane]) * qscale;
    }
    if (commit) {
#pragma unroll
        for (int i = 0; i < R; ++i) GST_ST(state, ((size_t) (i0 + i) * h_v + h) * S + j, st[i]);
    }
}

// ---- chunked GDN prefill (STRATA_DENSE_GDN_CHUNK=1, prompt chunks): the gated delta rule over chunks of GC tokens
// in its WY form, all F32 (only the summation order differs from gdn_seq4p_kernel).  With gamma_t the gate's running
// sum inside a chunk, A_ts = beta_t e^(gamma_t - gamma_s) k_t.k_s (s < t), T = (I + A)^-1, U = T (beta v),
// W = T (beta e^gamma k), M_ts = e^(gamma_t - gamma_s) q_t.k_s (s <= t):  D = U - W S0,
// O = qscale (e^gamma (Q S0) + M D),  S = e^gamma_last S0 + K^T (e^(gamma_last - gamma) D).
// Every decay factor is <= 1.  gdn_chunk_prep_kernel: U, W, M and the factors of every chunk at once;
// gdn_chunk_scan_kernel: the chunks in order, a block a (value head, 32 state columns)
// Measured (chains 284 / 285, opt-in only): the same 512 greedy tokens at 35K and 68K, but slower - a 4096-token
// chunk's layer 6.2 ms scan + 2.0 ms prep against the sequential walk's 3.9 ms (PP 1046 -> 1000 at 35K, 781 -> 748
// at 68K).  In F32 the chunked form does about as many FMAs as the walk (~3.5 vs 4 dk dv a token); it can only win
// with the products on tensor cores, which needs FP16 state operands (rounding the state) or hi / lo splits (~3x).
constexpr int GC = 32;
__global__ void __launch_bounds__(256) gdn_chunk_prep_kernel(const float* __restrict__ x, int ld,
                                                             const float* __restrict__ g, const float* __restrict__ b,
                                                             int ldg, int R, int h_k, int h_v, float* __restrict__ U,
                                                             float* __restrict__ W, float* __restrict__ M,
                                                             float* __restrict__ F) {
    constexpr int S = 128, KP = S + 1;
    __shared__ float sK[GC][KP], sQ[GC][KP], sA[GC][GC + 1], sT[GC][GC + 1], sgam[GC], sbet[GC];
    const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, kh = gdn_kh(h, h_k);
    const int tok0 = c * GC;
    for (int e = tid; e < GC * S; e += 256) {
        const int r = e / S, i = e % S, tok = tok0 + r;
        const bool ok = tok < R;
        sQ[r][i] = ok ? x[(size_t) tok * ld + kh * S + i] : 0.0f;
        sK[r][i] = ok ? x[(size_t) tok * ld + h_k * S + kh * S + i] : 0.0f;
    }
    if (tid < 32) {   // the gate's running sum (log decays; padding rows 0) and beta
        const int tok = tok0 + tid;
        float gv = tok < R ? g[(size_t) tok * ldg + h] : 0.0f;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, gv, o);
            if (tid >= o) gv += n;
        }
        sgam[tid] = gv;
        sbet[tid] = tok < R ? b[(size_t) tok * ldg + h] : 0.0f;
    }
    __syncthreads();
    float* Mc = M + ((size_t) c * h_v + h) * GC * GC;
    for (int e = tid; e < GC * GC; e += 256) {
        const int t = e / GC, sidx = e % GC;
        float kk = 0.0f, qk = 0.0f;
        if (sidx <= t) {
            for (int i = 0; i < S; ++i) { kk += sK[t][i] * sK[sidx][i]; qk += sQ[t][i] * sK[sidx][i]; }
        }
        const float dec = sidx <= t ? __expf(sgam[t] - sgam[sidx]) : 0.0f;
        sA[t][sidx] = sidx < t ? sbet[t] * dec * kk : 0.0f;
        Mc[e] = dec * qk;
    }
    __syncthreads();
    // T = (I + A)^-1, unit lower triangular: lane c its column - T_tc = -sum_{s=c..t-1} A_ts T_sc (T_cc = 1); a lane
    // reads A and its own column of T only, so no barrier between rows
    if (tid < GC) {
        const int col = tid;
        for (int t = col + 1; t < GC; ++t) {
            float v = -sA[t][col];   // (s = c: T_cc = 1)
            for (int s2 = col + 1; s2 < t; ++s2) v -= sA[t][s2] * sT[s2][col];
            sT[t][col] = v;
        }
    }
    __syncthreads();
    // U_tj = sum_{s<=t} T_ts beta_s v_sj, W_tj = sum_{s<=t} T_ts beta_s e^gamma_s k_sj: thread (j, half of the rows)
    {
        const int j = tid & (S - 1), half = tid >> 7;
        float bv[GC], bk[GC];
#pragma unroll
        for (int s2 = 0; s2 < GC; ++s2) {
            const int tok = tok0 + s2;
            bv[s2] = tok < R ? sbet[s2] * x[(size_t) tok * ld + 2 * h_k * S + h * S + j] : 0.0f;
            bk[s2] = sbet[s2] * __expf(sgam[s2]) * sK[s2][j];
        }
#pragma unroll
        for (int tt = 0; tt < GC / 2; ++tt) {
            const int t = half * (GC / 2) + tt;
            float u = 0.0f, w = 0.0f;
#pragma unroll
            for (int s2 = 0; s2 < GC; ++s2) {   // (T_tt = 1; the arrays indexed by constants only)
                const float tv = s2 < t ? sT[t][s2] : s2 == t ? 1.0f : 0.0f;
                u += tv * bv[s2];
                w += tv * bk[s2];
            }
            const size_t o = ((size_t) (tok0 + t) * h_v + h) * S + j;
            U[o] = u;
            W[o] = w;
        }
    }
    if (tid < GC) {
        float* Fc = F + ((size_t) c * h_v + h) * 2 * GC;
        Fc[tid] = __expf(sgam[tid]);
        Fc[GC + tid] = __expf(sgam[GC - 1] - sgam[tid]);
    }
}

__global__ void __launch_bounds__(256) gdn_chunk_scan_kernel(float* __restrict__ state, const float* __restrict__ x,
                                                             int ld, int R, int h_k, int h_v,
                                                             const float* __restrict__ U, const float* __restrict__ W,
                                                             const float* __restrict__ M, const float* __restrict__ F,
                                                             float* __restrict__ o) {
    constexpr int S = 128, CW = 32, KP = S + 4;
    __shared__ __align__(16) float sS[S][CW];
    __shared__ __align__(16) float sX[GC][KP];
    __shared__ __align__(16) float sD[GC][CW];
    __shared__ float sM[GC][GC + 1], sF[2 * GC];
    const int h = blockIdx.x, j0 = blockIdx.y * CW, tid = threadIdx.x, kh = gdn_kh(h, h_k);
    const float qscale = 1.0f / sqrtf((float) S);
    for (int e = tid; e < S * CW; e += 256) sS[e / CW][e % CW] = GST_LD(state, ((size_t) (e / CW) * h_v + h) * S + j0 + e % CW);
    const int nch = (R + GC - 1) / GC;
    const int dt = tid >> 3, dj = (tid & 7) * 4;          // D / O: row dt, columns dj .. dj + 3
    const int si = tid >> 1, sj = (tid & 1) * 16;          // S: row si, columns sj .. sj + 15
    for (int c = 0; c < nch; ++c) {
        const int tok0 = c * GC;
        __syncthreads();
        for (int e = tid; e < GC * S / 4; e += 256) {     // W's rows (head h)
            const int r = e / (S / 4), i4 = (e % (S / 4)) * 4;
            *(float4*) &sX[r][i4] = *(const float4*) (W + ((size_t) (tok0 + r) * h_v + h) * S + i4);
        }
        for (int e = tid; e < GC * GC; e += 256) sM[e / GC][e % GC] = M[((size_t) c * h_v + h) * GC * GC + e];
        if (tid < 2 * GC) sF[tid] = F[((size_t) c * h_v + h) * 2 * GC + tid];
        __syncthreads();
        {   // D = U - W S0
            float4 d = *(const float4*) (U + ((size_t) (tok0 + dt) * h_v + h) * S + j0 + dj);
            for (int i = 0; i < S; ++i) {
                const float wv = sX[dt][i];
                const float4 sv = *(const float4*) &sS[i][dj];
                d.x -= wv * sv.x; d.y -= wv * sv.y; d.z -= wv * sv.z; d.w -= wv * sv.w;
            }
            *(float4*) &sD[dt][dj] = d;
        }
        __syncthreads();
        for (int e = tid; e < GC * S / 4; e += 256) {     // Q's rows (key head kh)
            const int r = e / (S / 4), i4 = (e % (S / 4)) * 4, tok = tok0 + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (tok < R) v = *(const float4*) (x + (size_t) tok * ld + kh * S + i4);
            *(float4*) &sX[r][i4] = v;
        }
        __syncthreads();
        {   // O = qscale (e^gamma Q S0 + M D)
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f), m = a;
            for (int i = 0; i < S; ++i) {
                const float qv = sX[dt][i];
                const float4 sv = *(const float4*) &sS[i][dj];
                a.x += qv * sv.x; a.y += qv * sv.y; a.z += qv * sv.z; a.w += qv * sv.w;
            }
            for (int s2 = 0; s2 <= dt; ++s2) {
                const float mv = sM[dt][s2];
                const float4 dv = *(const float4*) &sD[s2][dj];
                m.x += mv * dv.x; m.y += mv * dv.y; m.z += mv * dv.z; m.w += mv * dv.w;
            }
            const float eg = sF[dt];
            if (tok0 + dt < R) {
                float* op = o + (size_t) (tok0 + dt) * h_v * S + h * S + j0 + dj;
                *(float4*) op = make_float4((eg * a.x + m.x) * qscale, (eg * a.y + m.y) * qscale,
                                            (eg * a.z + m.z) * qscale, (eg * a.w + m.w) * qscale);
            }
        }
        __syncthreads();
        for (int e = tid; e < GC * S / 4; e += 256) {     // K's rows
            const int r = e / (S / 4), i4 = (e % (S / 4)) * 4, tok = tok0 + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (tok < R) v = *(const float4*) (x + (size_t) tok * ld + h_k * S + kh * S + i4);
            *(float4*) &sX[r][i4] = v;
        }
        __syncthreads();
        {   // S = e^gamma_last S0 + K^T (e^(gamma_last - gamma) D)
            const float el = sF[GC - 1];
            float acc[16];
#pragma unroll
            for (int q = 0; q < 16; ++q) acc[q] = el * sS[si][sj + q];
            for (int s2 = 0; s2 < GC; ++s2) {
                const float kv = sX[s2][si] * sF[GC + s2];
#pragma unroll
                for (int q = 0; q < 16; ++q) acc[q] += kv * sD[s2][sj + q];
            }
#pragma unroll
            for (int q = 0; q < 16; ++q) sS[si][sj + q] = acc[q];
        }
    }
    __syncthreads();
    for (int e = tid; e < S * CW; e += 256) GST_ST(state, ((size_t) (e / CW) * h_v + h) * S + j0 + e % CW, sS[e / CW][e % CW]);
}

// ---- commit of every GDN layer in two launches (a verify window's kept n tokens): the conv states take the last 3
// of [state | raw rows], the recurrence runs gdn_seq4's arithmetic over the stashed conv rows (no output)
__global__ void commit_conv_kernel(const Q35CommitLayer* __restrict__ tab, int n, int C) {
    const Q35CommitLayer& L = tab[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h[3] = {L.conv_state[c * 3], L.conv_state[c * 3 + 1], L.conv_state[c * 3 + 2]};
    for (int t = 0; t < n; ++t) { h[0] = h[1]; h[1] = h[2]; h[2] = L.raw[(size_t) t * C + c]; }
    L.conv_state[c * 3] = h[0]; L.conv_state[c * 3 + 1] = h[1]; L.conv_state[c * 3 + 2] = h[2];
}
__global__ void __launch_bounds__(128) commit_gdn_kernel(const Q35CommitLayer* __restrict__ tab, int n, int C, int h_k,
                                                         int h_v) {
    constexpr int S = 128, R = 32;
    const Q35CommitLayer& L = tab[blockIdx.z];
    float* __restrict__ state = L.state;
    const float* __restrict__ x = L.conv;
    const float* __restrict__ g = L.ab;
    const float* __restrict__ b = L.ab + h_v;
    const int ldg = 2 * h_v;
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, kh = gdn_kh(h, h_k);
    const int j = blockIdx.y * 32 + lane, i0 = w * R;
    __shared__ float sk[S], rs[4][32];
    float st[R];
#pragma unroll
    for (int i = 0; i < R; ++i) st[i] = GST_LD(state, ((size_t) (i0 + i) * h_v + h) * S + j);
    for (int t = 0; t < n; ++t) {
        const float* xr = x + (size_t) t * C;
        __syncthreads();
        sk[threadIdx.x] = xr[h_k * S + kh * S + threadIdx.x];
        __syncthreads();
        const float v = xr[2 * h_k * S + h * S + j];
        const float dec = expf(g[(size_t) t * ldg + h]);
        const float beta = b[(size_t) t * ldg + h];
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < R; ++i) { st[i] *= dec; p += st[i] * sk[i0 + i]; }
        rs[w][lane] = p;
        __syncthreads();
        const float skd = ((rs[0][lane] + rs[1][lane]) + rs[2][lane]) + rs[3][lane];
        const float d = (v - skd) * beta;
#pragma unroll
        for (int i = 0; i < R; ++i) st[i] += sk[i0 + i] * d;
    }
#pragma unroll
    for (int i = 0; i < R; ++i) GST_ST(state, ((size_t) (i0 + i) * h_v + h) * S + j, st[i]);
}

// ------------------------------------------------------------------------------------------------ attention prep
// the orthonormal 256-point Walsh-Hadamard transform of s (shared) by 256 threads, in place
__device__ __forceinline__ void fwht256(float* s, int t) {
#pragma unroll
    for (int h = 1; h < 256; h <<= 1) {
        __syncthreads();
        const float a = s[t], b = s[t ^ h];
        __syncthreads();
        s[t] = (t & h) ? b - a : a + b;
    }
    __syncthreads();
    s[t] *= 0.0625f;   // 1 / sqrt(256)
    __syncthreads();
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// block (token, head), D threads: the head's rms * weight, then NeoX rope on the first n_rot dims (native_rope.cu);
// K and V into the cache (FP16, or K8V4: INT8 K, rotated 4-bit V).  No q_full: the K / V heads only (a cache fill)
__global__ void attn_prep_kernel(const float* __restrict__ q_full, int ldq, const float* __restrict__ k,
                                 const float* __restrict__ v, int ldk, const float* __restrict__ q_norm,
                                 const float* __restrict__ k_norm, float eps, float theta_scale, int n_rot, int p0,
                                 int n_head, int D, float* __restrict__ q_out, Q35Kv kv) {
    __shared__ float sh[32];
    __shared__ float row[256];
    const int t = blockIdx.x, hh = blockIdx.y + (q_full ? 0 : n_head), d = threadIdx.x, pos = p0 + t;
    const bool is_q = hh < n_head;
    const float* src = is_q ? q_full + (size_t) t * ldq + (size_t) hh * 2 * D : k + (size_t) t * ldk + (size_t) (hh - n_head) * D;
    const float val = src[d];
    const float inv = rsqrtf(block_sum(val * val, sh) / (float) D + eps);
    row[d] = val * inv * (is_q ? q_norm[d] : k_norm[d]);
    __syncthreads();
    float r = row[d];
    const int half_rot = n_rot / 2;
    if (d < n_rot) {
        const int pair = d < half_rot ? d : d - half_rot;
        const float theta = (float) pos * powf(theta_scale, (float) pair);
        const float c = cosf(theta), s = sinf(theta);
        const float a = row[pair], bb = row[pair + half_rot];
        r = d < half_rot ? a * c - bb * s : a * s + bb * c;
    }
    if (is_q) {
        q_out[(size_t) t * n_head * D + (size_t) hh * D + d] = r;
        return;
    }
    const int kvh = hh - n_head;
    const size_t cell = (size_t) kvh * kv.max_ctx + pos;
    float vv = v[(size_t) t * ldk + (size_t) kvh * D + d];
    if (kv.mode == 0) {
        ((__half*) kv.k)[cell * D + d] = __float2half(r);
        ((__half*) kv.v)[cell * D + d] = __float2half(vv);
        return;
    }
    // K8V4: K codes with a scale per 32 (a warp), V rotated then 4-bit codes with a scale per 32 (K8V8: V as K)
    {
        const float amax = warp_max(fabsf(r));
        const float sc = amax / 127.0f;
        const int q = sc > 0.0f ? max(-127, min(127, __float2int_rn(r / sc))) : 0;
        ((int8_t*) kv.k)[cell * D + d] = (int8_t) q;
        if ((d & 31) == 0) ((__half*) kv.ks)[cell * (D / 32) + d / 32] = __float2half(sc);
    }
    if (kv.mode == 2) {   // K8V8: V codes as K's, not rotated
        const float amax = warp_max(fabsf(vv));
        const float sc = amax / 127.0f;
        const int q = sc > 0.0f ? max(-127, min(127, __float2int_rn(vv / sc))) : 0;
        ((int8_t*) kv.v)[cell * D + d] = (int8_t) q;
        if ((d & 31) == 0) ((__half*) kv.vs)[cell * (D / 32) + d / 32] = __float2half(sc);
        return;
    }
    __syncthreads();   // (row is reused)
    row[d] = vv;
    fwht256(row, d);
    vv = row[d];
    {
        const float amax = warp_max(fabsf(vv));
        const float sc = amax / 7.0f;
        const int q = sc > 0.0f ? max(-7, min(7, __float2int_rn(vv / sc))) : 0;
        const int qn = __shfl_xor_sync(0xffffffffu, q, 1);
        if ((d & 1) == 0) ((uint8_t*) kv.v)[cell * (D / 2) + d / 2] = (uint8_t) ((q + 8) | ((qn + 8) << 4));
        if ((d & 31) == 0) ((__half*) kv.vs)[cell * (D / 32) + d / 32] = __float2half(sc);
    }
}

// K8V4's output: per (token, head) rotated back by H, then the output gate (block (token, head), 256 threads)
__global__ void unrotate_gate_kernel(float* __restrict__ out, int n_head, const float* __restrict__ q_full, int ldq) {
    __shared__ float row[256];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    float* o = out + (size_t) t * n_head * 256 + (size_t) h * 256;
    row[d] = o[d];
    fwht256(row, d);
    const float g = q_full[(size_t) t * ldq + (size_t) h * 512 + 256 + d];
    o[d] = row[d] / (1.0f + __expf(-g));
}

// ------------------------------------------------------------------------------------------------ attention
constexpr int AD = 256, AR = 64, AK = 32, AW = 4;
constexpr int LDQ = AD + 8, LDS = AK + 4, LDP = AK + 8;   // (wmma: ldm % 8 halves / 4 floats)
constexpr size_t kAttnSmem = (size_t) AR * LDQ * 2 + 2 * (size_t) AK * LDQ * 2 + (size_t) AW * 16 * LDS * 4 +
                             (size_t) AW * 16 * LDP * 2;

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 halves of a K8V4 K row (8 INT8 codes, the group's scale) / V row (8 4-bit codes): the codes become FP16
// 1024 + c with a byte permute (0x64 high bytes), then (h - offset) * scale in half2 arithmetic
__device__ __forceinline__ uint4 deq_k8(uint2 u, __half s1) {
    const __half2 sc = __half2half2(s1), off = __float2half2_rn(1152.0f);   // 1024 + 128 (the sign flip)
    const uint32_t a = u.x ^ 0x80808080u, b = u.y ^ 0x80808080u;
    uint32_t r[4];
    const uint32_t w[4] = {__byte_perm(a, 0x64u, 0x4140), __byte_perm(a, 0x64u, 0x4342),
                           __byte_perm(b, 0x64u, 0x4140), __byte_perm(b, 0x64u, 0x4342)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 h = __hmul2(__hsub2(*(const __half2*) &w[i], off), sc);
        r[i] = *(const uint32_t*) &h;
    }
    return make_uint4(r[0], r[1], r[2], r[3]);
}
__device__ __forceinline__ uint4 deq_v4(uint32_t u, __half s1) {
    const __half2 sc = __half2half2(s1), off = __float2half2_rn(1032.0f);    // 1024 + 8 (the code bias)
    // nibbles in order: byte i holds values 2i (low) and 2i + 1 (high)
    const uint32_t lo = u & 0x0F0F0F0Fu, hi = (u >> 4) & 0x0F0F0F0Fu;
    const uint32_t e = __byte_perm(lo, hi, 0x5140), f = __byte_perm(lo, hi, 0x7362);   // lo0 hi0 lo1 hi1 | lo2..
    uint32_t r[4];
    const uint32_t w[4] = {__byte_perm(e, 0x64u, 0x4140), __byte_perm(e, 0x64u, 0x4342),
                           __byte_perm(f, 0x64u, 0x4140), __byte_perm(f, 0x64u, 0x4342)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 h = __hmul2(__hsub2(*(const __half2*) &w[i], off), sc);
        r[i] = *(const uint32_t*) &h;
    }
    return make_uint4(r[0], r[1], r[2], r[3]);
}

// grid (row tiles, n_kv, splits), AW warps of 16 rows; see the header.  gated: the output gate applied here (FP16
// cache); K8V4 leaves it to unrotate_gate_kernel (the output is still rotated)
template <int AR_, int AK_, int AW_, bool DIRECT_ = false>
constexpr size_t attn_smem() {
    return (size_t) AR_ * LDQ * 2 + (DIRECT_ ? 0 : 2 * (size_t) AK_ * LDQ * 2) + (size_t) AW_ * 16 * (AK_ + 4) * 4 +
           (size_t) AW_ * 16 * (AK_ + 8) * 2;
}
// (the tile shape is a parameter: the 64-row default for prompts, 32-row tiles of 2 warps for verify windows)
// DIRECT_ (an FP16 cache, opt-in): the K / V fragments read in place from the cache (L1) instead of staged
// in shared memory - 48 instead of 82 KB a block, two blocks an SM; the cache is zeroed with 32 spare cells, so a
// tile's keys past the last one are finite (masked to P = 0)
template <int MODE, int AR_ = 64, int AK_ = 32, int AW_ = 4, bool DIRECT_ = false>
__global__ void __launch_bounds__(AW_ * 32) attn_tc_kernel(const float* __restrict__ q, Q35Kv kv, int p0, int T, int n_head,
                                                          int G, int max_ctx, int split_len, float scale,
                                                          const float* __restrict__ q_full, int ldq,
                                                          float* __restrict__ out, float* __restrict__ part_o,
                                                          float* __restrict__ part_ml) {
    using namespace nvcuda;
    constexpr int AR = AR_, AK = AK_, AW = AW_, LDS = AK + 4, LDP = AK + 8;
    static_assert(AR == AW * 16, "a warp's 16 rows");
    extern __shared__ __align__(32) unsigned char smem[];
    __half* sQ = (__half*) smem;
    constexpr bool DIRECT = DIRECT_ && MODE == 0;
    __half* sK = sQ + AR * LDQ;
    __half* sV = sK + AK * LDQ;
    float* sS = (float*) (DIRECT ? sK : sV + AK * LDQ);
    __half* sP = (__half*) (sS + AW * 16 * LDS);
    const int kvh = blockIdx.y, split = blockIdx.z, n_split = gridDim.z;
    const int rows = T * G, r0 = blockIdx.x * AR;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int qld = n_head * AD;
    // Q (scaled) -> sQ
    for (int i = tid; i < AR * (AD / 4); i += AW * 32) {
        const int r = i / (AD / 4), d = (i % (AD / 4)) * 4, row = r0 + r;
        float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row < rows) {
            const int tok = row / G, head = kvh * G + row % G;
            f = *(const float4*) (q + (size_t) tok * qld + (size_t) head * AD + d);
        }
        __half2* dst = (__half2*) (sQ + r * LDQ + d);
        dst[0] = __floats2half2_rn(f.x * scale, f.y * scale);
        dst[1] = __floats2half2_rn(f.z * scale, f.w * scale);
    }
    const int last_row = min(r0 + AR, rows) - 1;
    const int kend = p0 + last_row / G + 1;                        // the block's keys: [0, kend)
    const int ks = split * split_len, ke = min(kend, ks + split_len);
    const int wr0 = r0 + w * 16;
    const bool w_on = wr0 < rows;
    const int wlast = w_on ? p0 + (min(wr0 + 16, rows) - 1) / G : -1;
    const int ri = lane >> 1, hi = lane & 1;
    const int row = wr0 + ri;
    const bool row_ok = row < rows;
    const int pos = row_ok ? p0 + row / G : -1;
    float* Sw = sS + w * 16 * LDS;
    __half* Pw = sP + w * 16 * LDP;
    constexpr float kLazy = 8.0f;
    float m = -INFINITY, l = 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[AD / 16];
#pragma unroll
    for (int n = 0; n < AD / 16; ++n) wmma::fill_fragment(fo[n], 0.0f);
    // the next tile's K / V (raw: FP16, or K8V4 codes and scales) prefetched into registers while this one computes
    constexpr int PF = AK * (AD / 8) / (AW * 32);
    uint4 pk[PF], pv[PF];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int j = 0; j < PF; ++j) {
            const int i = tid + j * AW * 32, r = i / (AD / 8), d = (i % (AD / 8)) * 8, key = k0 + r;
            pk[j] = pv[j] = make_uint4(0, 0, 0, 0);
            if (key < ke) {
                const size_t cell = (size_t) kvh * max_ctx + key;
                if (MODE == 0) {
                    pk[j] = __ldg((const uint4*) ((const __half*) kv.k + cell * AD + d));
                    pv[j] = __ldg((const uint4*) ((const __half*) kv.v + cell * AD + d));
                } else {   // K: 8 codes + its scale's bits; V: 8 nibbles (K8V8: 8 codes) + its scale's bits
                    const uint2 kc = __ldg((const uint2*) ((const int8_t*) kv.k + cell * AD + d));
                    pk[j] = make_uint4(kc.x, kc.y, __ldg((const unsigned short*) kv.ks + cell * (AD / 32) + d / 32), 0);
                    if (MODE == 2) {
                        const uint2 vc = __ldg((const uint2*) ((const int8_t*) kv.v + cell * AD + d));
                        pv[j] = make_uint4(vc.x, vc.y, __ldg((const unsigned short*) kv.vs + cell * (AD / 32) + d / 32), 0);
                    } else {
                        pv[j] = make_uint4(__ldg((const uint32_t*) ((const uint8_t*) kv.v + cell * (AD / 2) + d / 2)),
                                           __ldg((const unsigned short*) kv.vs + cell * (AD / 32) + d / 32), 0, 0);
                    }
                }
            }
        }
    };
    if (!DIRECT && ks < ke) fetch(ks);
    for (int k0 = ks; k0 < ke; k0 += AK) {
        __syncthreads();
        // (DIRECT: the tile's K / V rows in the cache, ld = AD)
        const __half* gK = (const __half*) kv.k + ((size_t) kvh * max_ctx + k0) * AD;
        const __half* gV = (const __half*) kv.v + ((size_t) kvh * max_ctx + k0) * AD;
        if (!DIRECT)
#pragma unroll
        for (int j = 0; j < PF; ++j) {
            const int i = tid + j * AW * 32, r = i / (AD / 8), d = (i % (AD / 8)) * 8;
            uint4 kv4 = pk[j], vv4 = pv[j];
            if (MODE != 0) {
                kv4 = deq_k8(make_uint2(pk[j].x, pk[j].y), __ushort_as_half((unsigned short) pk[j].z));
                vv4 = MODE == 2 ? deq_k8(make_uint2(pv[j].x, pv[j].y), __ushort_as_half((unsigned short) pv[j].z))
                                : deq_v4(pv[j].x, __ushort_as_half((unsigned short) pv[j].y));
            }
            *(uint4*) (sK + r * LDQ + d) = kv4;
            *(uint4*) (sV + r * LDQ + d) = vv4;
        }
        __syncthreads();
        if (!DIRECT && k0 + AK < ke) fetch(k0 + AK);
        if (!w_on || k0 > wlast) continue;
        // S (16 x AK) = Q_w . K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[AK / 16];
#pragma unroll
        for (int n = 0; n < AK / 16; ++n) wmma::fill_fragment(acc[n], 0.0f);
#pragma unroll
        for (int kk = 0; kk < AD / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fq;
            wmma::load_matrix_sync(fq, sQ + (w * 16) * LDQ + kk * 16, LDQ);
#pragma unroll
            for (int n = 0; n < AK / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                if (DIRECT) wmma::load_matrix_sync(fb, gK + (n * 16) * AD + kk * 16, AD);
                else wmma::load_matrix_sync(fb, sK + (n * 16) * LDQ + kk * 16, LDQ);
                wmma::mma_sync(acc[n], fq, fb, acc[n]);
            }
        }
#pragma unroll
        for (int n = 0; n < AK / 16; ++n) wmma::store_matrix_sync(Sw + n * 16, acc[n], LDS, wmma::mem_row_major);
        __syncwarp();
        constexpr int KH = AK / 2;
        float sv[KH], mt = -INFINITY;
#pragma unroll
        for (int j = 0; j < KH; ++j) {
            const int c = hi * KH + j, key = k0 + c;
            sv[j] = row_ok && key < ke && key <= pos ? Sw[ri * LDS + c] : -INFINITY;
            mt = fmaxf(mt, sv[j]);
        }
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
        const bool need = mt > m + kLazy;
        float corr = 1.0f;
        if (need) {
            corr = m == -INFINITY ? 0.0f : __expf(m - mt);
            l *= corr;
            m = mt;
        }
        float ps = 0.0f;
#pragma unroll
        for (int j = 0; j < KH; ++j) {
            const float pj = sv[j] == -INFINITY ? 0.0f : __expf(sv[j] - m);
            ps += pj;
            Pw[ri * LDP + hi * KH + j] = __float2half(pj);
        }
        l += ps + __shfl_xor_sync(0xffffffffu, ps, 1);
        __syncwarp();
        if (__any_sync(0xffffffffu, need)) {
#pragma unroll
            for (int n = 0; n < AD / 16; ++n) {
                wmma::store_matrix_sync(Sw, fo[n], LDS, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 8; ++j) Sw[ri * LDS + hi * 8 + j] *= corr;
                __syncwarp();
                wmma::load_matrix_sync(fo[n], Sw, LDS, wmma::mem_row_major);
                __syncwarp();
            }
        }
#pragma unroll
        for (int kk = 0; kk < AK / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
            wmma::load_matrix_sync(fp, Pw + kk * 16, LDP);
#pragma unroll
            for (int n = 0; n < AD / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                if (DIRECT) wmma::load_matrix_sync(fv, gV + (kk * 16) * AD + n * 16, AD);
                else wmma::load_matrix_sync(fv, sV + (kk * 16) * LDQ + n * 16, LDQ);
                wmma::mma_sync(fo[n], fp, fv, fo[n]);
            }
        }
        __syncwarp();
    }
    if (!w_on) return;
    const int tok = row_ok ? row / G : 0, head = kvh * G + (row_ok ? row % G : 0);
    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
    for (int n = 0; n < AD / 16; ++n) {
        wmma::store_matrix_sync(Sw, fo[n], LDS, wmma::mem_row_major);
        __syncwarp();
        if (row_ok) {
            const int d0 = n * 16 + hi * 8;
            if (n_split == 1) {
                float* o = out + (size_t) tok * qld + (size_t) head * AD + d0;
                const float* gt = q_full + (size_t) tok * ldq + (size_t) head * 2 * AD + AD + d0;
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = Sw[ri * LDS + hi * 8 + j] * inv_l * (MODE != 1 ? sigm(gt[j]) : 1.0f);
            } else {
                float* o = part_o + (((size_t) split * gridDim.y + kvh) * rows + row) * AD + d0;
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = Sw[ri * LDS + hi * 8 + j];
            }
        }
        __syncwarp();
    }
    if (n_split > 1 && row_ok && hi == 0) {
        part_ml[(((size_t) split * gridDim.y + kvh) * rows + row) * 2] = m;
        part_ml[(((size_t) split * gridDim.y + kvh) * rows + row) * 2 + 1] = l;
    }
}

// ---- the prompt path (FP16 cache): attn_tc_kernel's 64-row x 32-key tiles with 8 warps instead of 4 - warp (rt, h)
// the rows 16 rt .. +15 and the head dims 128 h .. +127.  S = Q K^T: each warp of a pair its half of the head
// dimension (the two partials summed in the softmax); the softmax: warp (rt, h) the rows 16 rt + 8 h .. +7, 4 lanes a
// row (8 keys each), m and l in its registers, the rows' correction factors through shared memory; O += P V: each
// warp its 8 of the 16 output fragments.  Twice the warps an SM (the old kernel: 4 a block, one block an SM - its
// shared memory), the same tiles, splits and partial layout.
constexpr int PP_AR = 64, PP_AK = 32, PP_W = 8, PP_LDS = PP_AK + 4, PP_LDP = PP_AK + 8;
// (Q is staged over the K / V buffers: its fragments go to registers before the first tile's stores - 90 -> 57 KiB,
// the same bits, ~0.3% faster; a 64 KiB carveout preference on top measured slower)
constexpr size_t kPpSmem = 2 * (size_t) PP_AK * LDQ * 2 + 2 * (size_t) PP_AR * PP_LDS * 4 +
                           (size_t) PP_AR * PP_LDP * 2 + 4 * (size_t) PP_AR * 4 + 16;
// HALVES: the rows 0-31 (warps 0-3) and 32-63 (warps 4-7) synchronise their S / softmax / P V phases with their own
// named barrier, so one half's softmax overlaps the other half's products (only the K / V tile swap is block-wide)
template <bool HALVES>
__global__ void __launch_bounds__(PP_W * 32) attn_pp_kernel(const float* __restrict__ q, Q35Kv kv, int p0, int T,
                                                           int n_head, int G, int max_ctx, int split_len, float scale,
                                                           const float* __restrict__ q_full, int ldq,
                                                           float* __restrict__ out, float* __restrict__ part_o,
                                                           float* __restrict__ part_ml) {
    using namespace nvcuda;
    constexpr int AR = PP_AR, AK = PP_AK, LDS = PP_LDS, LDP = PP_LDP;
    extern __shared__ __align__(32) unsigned char smem[];
    static_assert(PP_AR * LDQ <= 2 * PP_AK * LDQ + 2 * PP_AR * PP_LDS * 2, "Q staging fits the K / V / S area");
    __half* sQ = (__half*) smem;   // (only until the warps hold their Q fragments)
    __half* sK = sQ;
    __half* sV = sK + AK * LDQ;
    float* sS = (float*) (sV + AK * LDQ);            // [2 halves][AR][LDS] partial scores; the rescale's scratch
    __half* sP = (__half*) (sS + 2 * AR * LDS);      // [AR][LDP]
    float* sCorr = (float*) (sP + AR * LDP);         // [AR] this tile's correction factors
    float* sL = sCorr + AR;                          // [AR] the final totals / maxima (epilogue)
    float* sM = sL + AR;
    int* sNeed = (int*) (sM + AR);                   // [4] a row tile's O needs rescaling this tile
    const int kvh = blockIdx.y, split = blockIdx.z, n_split = gridDim.z;
    const int rows = T * G, r0 = blockIdx.x * AR;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, rt = w >> 1, hf = w & 1;
    const int qld = n_head * AD;
    auto phase_sync = [&] {
        if (HALVES) asm volatile("bar.sync %0, 128;" ::"r"(1 + (rt >> 1)) : "memory");
        else __syncthreads();
    };
    for (int i = tid; i < AR * (AD / 4); i += PP_W * 32) {   // Q (scaled) -> sQ
        const int r = i / (AD / 4), d = (i % (AD / 4)) * 4, row = r0 + r;
        float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row < rows) {
            const int tok = row / G, head = kvh * G + row % G;
            f = *(const float4*) (q + (size_t) tok * qld + (size_t) head * AD + d);
        }
        __half2* dst = (__half2*) (sQ + r * LDQ + d);
        dst[0] = __floats2half2_rn(f.x * scale, f.y * scale);
        dst[1] = __floats2half2_rn(f.z * scale, f.w * scale);
    }
    const int last_row = min(r0 + AR, rows) - 1;
    const int kend = p0 + last_row / G + 1;
    const int ks = split * split_len, ke = min(kend, ks + split_len);
    const int tr0 = r0 + rt * 16;                    // the warp's row tile
    const bool t_on = tr0 < rows;
    const int tlast = t_on ? p0 + (min(tr0 + 16, rows) - 1) / G : -1;
    // the softmax rows: 16 rt + 8 hf + (lane >> 2), lane & 3 its 8 keys
    const int sr = rt * 16 + hf * 8 + (lane >> 2), sq = lane & 3;
    const int srow = r0 + sr;
    const bool srow_ok = srow < rows;
    const int spos = srow_ok ? p0 + srow / G : -1;
    constexpr float kLazy = 8.0f;
    float m = -INFINITY, l = 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[8];   // O rows of the tile, dims 128 hf + 16 n
    // the warp's Q fragments (its rows, its half of the head dims) held in registers for every key tile: 13% faster
    // than loading them from sQ each tile (235 registers, one block an SM either way)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fqr[8];
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) wmma::load_matrix_sync(fqr[kk], sQ + (rt * 16) * LDQ + (8 * hf + kk) * 16, LDQ);
#pragma unroll
    for (int n = 0; n < 8; ++n) wmma::fill_fragment(fo[n], 0.0f);
    constexpr int PF = AK * (AD / 8) / (PP_W * 32);   // 16-byte chunks a thread a tile, K and V each
    uint4 pk[PF], pv[PF];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int j = 0; j < PF; ++j) {
            const int i = tid + j * PP_W * 32, r = i / (AD / 8), d = (i % (AD / 8)) * 8, key = k0 + r;
            pk[j] = pv[j] = make_uint4(0, 0, 0, 0);
            if (key < ke) {
                const size_t cell = (size_t) kvh * max_ctx + key;
                pk[j] = __ldg((const uint4*) ((const __half*) kv.k + cell * AD + d));
                pv[j] = __ldg((const uint4*) ((const __half*) kv.v + cell * AD + d));
            }
        }
    };
    if (ks < ke) fetch(ks);
    for (int k0 = ks; k0 < ke; k0 += AK) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PF; ++j) {
            const int i = tid + j * PP_W * 32, r = i / (AD / 8), d = (i % (AD / 8)) * 8;
            *(uint4*) (sK + r * LDQ + d) = pk[j];
            *(uint4*) (sV + r * LDQ + d) = pv[j];
        }
        if (tid < 4) sNeed[tid] = 0;
        __syncthreads();
        if (k0 + AK < ke) fetch(k0 + AK);
        const bool tile_on = t_on && k0 <= tlast;   // (the same for both warps of a pair)
        if (tile_on) {   // S partial (16 x AK) over the head dims 128 hf .. +127
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[AK / 16];
#pragma unroll
            for (int n = 0; n < AK / 16; ++n) wmma::fill_fragment(acc[n], 0.0f);
#pragma unroll
            for (int kk = 8 * hf; kk < 8 * hf + 8; ++kk) {
                const auto& fq = fqr[kk - 8 * hf];
#pragma unroll
                for (int n = 0; n < AK / 16; ++n) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                    wmma::load_matrix_sync(fb, sK + (n * 16) * LDQ + kk * 16, LDQ);
                    wmma::mma_sync(acc[n], fq, fb, acc[n]);
                }
            }
#pragma unroll
            for (int n = 0; n < AK / 16; ++n)
                wmma::store_matrix_sync(sS + (size_t) hf * AR * LDS + (rt * 16) * LDS + n * 16, acc[n], LDS,
                                        wmma::mem_row_major);
        }
        phase_sync();
        if (tile_on) {   // the softmax of the warp's 8 rows
            float sv[8], mt = -INFINITY;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int c = sq * 8 + j, key = k0 + c;
                const float s2 = sS[sr * LDS + c] + sS[(size_t) AR * LDS + sr * LDS + c];
                sv[j] = srow_ok && key < ke && key <= spos ? s2 : -INFINITY;
                mt = fmaxf(mt, sv[j]);
            }
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
            const bool need = mt > m + kLazy;
            float corr = 1.0f;
            if (need) {
                corr = m == -INFINITY ? 0.0f : __expf(m - mt);
                l *= corr;
                m = mt;
            }
            float ps = 0.0f;
            __half pr[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float pj = sv[j] == -INFINITY ? 0.0f : __expf(sv[j] - m);
                ps += pj;
                pr[j] = __float2half(pj);
            }
            *(uint4*) (sP + sr * LDP + sq * 8) = *(const uint4*) pr;
            ps += __shfl_xor_sync(0xffffffffu, ps, 1);
            ps += __shfl_xor_sync(0xffffffffu, ps, 2);
            l += ps;
            if (sq == 0) sCorr[sr] = corr;
            if (__any_sync(0xffffffffu, need) && lane == 0) sNeed[rt] = 1;
        }
        phase_sync();
        if (tile_on) {
            if (sNeed[rt]) {   // (rare after the first tile) the pair's O rows scaled by their factors
                // (a warp's 16 x 16 scratch inside its row tile's first partial: read, and within its half)
                float* sc = sS + (size_t) (rt * 16) * LDS + hf * 256;
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    wmma::store_matrix_sync(sc, fo[n], 16, wmma::mem_row_major);
                    __syncwarp();
                    for (int e = lane; e < 256; e += 32) sc[e] *= sCorr[rt * 16 + e / 16];
                    __syncwarp();
                    wmma::load_matrix_sync(fo[n], sc, 16, wmma::mem_row_major);
                    __syncwarp();
                }
            }
#pragma unroll
            for (int kk = 0; kk < AK / 16; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
                wmma::load_matrix_sync(fp, sP + (rt * 16) * LDP + kk * 16, LDP);
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                    wmma::load_matrix_sync(fv, sV + (kk * 16) * LDQ + hf * 128 + n * 16, LDQ);
                    wmma::mma_sync(fo[n], fp, fv, fo[n]);
                }
            }
        }
    }
    // the rows' totals for both warps of a pair, then the output (or the partials)
    if (sq == 0) { sL[sr] = l; sM[sr] = m; }
    __syncthreads();
    if (!t_on) return;
    float* sc = sS + (size_t) (rt * 16) * LDS + hf * 256;
    const int ri = lane >> 1, hi = lane & 1, row = tr0 + ri;
    const bool row_ok = row < rows;
    const int tok = row_ok ? row / G : 0, head = kvh * G + (row_ok ? row % G : 0);
    const float lr = sL[rt * 16 + ri], inv_l = lr > 0.0f ? 1.0f / lr : 0.0f;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        wmma::store_matrix_sync(sc, fo[n], 16, wmma::mem_row_major);
        __syncwarp();
        if (row_ok) {
            const int d0 = hf * 128 + n * 16 + hi * 8;
            if (n_split == 1) {
                float* o = out + (size_t) tok * qld + (size_t) head * AD + d0;
                const float* gt = q_full + (size_t) tok * ldq + (size_t) head * 2 * AD + AD + d0;
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = sc[ri * 16 + hi * 8 + j] * inv_l * sigm(gt[j]);
            } else {
                float* o = part_o + (((size_t) split * gridDim.y + kvh) * rows + row) * AD + d0;
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = sc[ri * 16 + hi * 8 + j];
            }
        }
        __syncwarp();
    }
    if (n_split > 1 && row_ok && hi == 0 && hf == 0) {
        part_ml[(((size_t) split * gridDim.y + kvh) * rows + row) * 2] = sM[rt * 16 + ri];
        part_ml[(((size_t) split * gridDim.y + kvh) * rows + row) * 2 + 1] = lr;
    }
}

// ---- decode / short verify (the tile kernel above fills 1 of its 4 warps with 6-48 rows and runs 1 block a SM):
// block = (kv head, split, token t), 4 warps, warp w takes the split's keys w, w + 4, ..; lane l the dims 8 l .. 8 l + 7
// of the G rows of token t (CUDA cores: q.k by a warp sum, the online softmax and O in registers), two keys in flight
// a warp.  The warps fold through shared memory; partials in attn_tc_kernel's layout (attn_merge_kernel folds them).
template <int MODE, int G, int PF = 2, int MINB = 1>
__global__ void __launch_bounds__(128, MINB) attn_dec_kernel(const float* __restrict__ q, Q35Kv kv, int p0, int n_head,
                                                       int max_ctx, int split_len, float scale,
                                                       const float* __restrict__ q_full, int ldq,
                                                       float* __restrict__ out, float* __restrict__ part_o,
                                                       float* __restrict__ part_ml) {
    __shared__ float sm_m[4][G], sm_l[4][G];
    __shared__ float sm_o[4][G][AD];
    const int kvh = blockIdx.x, split = blockIdx.y, n_split = gridDim.y, t = blockIdx.z, T = gridDim.z;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, n_kv = gridDim.x;
    const int ks = split * split_len, ke = min(p0 + t + 1, ks + split_len);
    const int qld = n_head * AD;
    float qv[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float4* qp = (const float4*) (q + (size_t) t * qld + (size_t) (kvh * G + g) * AD + lane * 8);
        const float4 a = qp[0], b = qp[1];
        qv[g][0] = a.x * scale; qv[g][1] = a.y * scale; qv[g][2] = a.z * scale; qv[g][3] = a.w * scale;
        qv[g][4] = b.x * scale; qv[g][5] = b.y * scale; qv[g][6] = b.z * scale; qv[g][7] = b.w * scale;
    }
    float m[G], l[G], acc[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY; l[g] = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[g][j] = 0.0f;
    }
    auto load = [&](int key, uint4& kr, uint4& vr) {
        const size_t cell = (size_t) kvh * max_ctx + key;
        if (MODE == 0) {
            kr = __ldg((const uint4*) ((const __half*) kv.k + cell * AD + lane * 8));
            vr = __ldg((const uint4*) ((const __half*) kv.v + cell * AD + lane * 8));
        } else {
            const uint2 kc = __ldg((const uint2*) ((const int8_t*) kv.k + cell * AD + lane * 8));
            kr = make_uint4(kc.x, kc.y, __ldg((const unsigned short*) kv.ks + cell * (AD / 32) + lane / 4), 0);
            if (MODE == 2) {
                const uint2 vc = __ldg((const uint2*) ((const int8_t*) kv.v + cell * AD + lane * 8));
                vr = make_uint4(vc.x, vc.y, __ldg((const unsigned short*) kv.vs + cell * (AD / 32) + lane / 4), 0);
            } else {
                vr = make_uint4(__ldg((const uint32_t*) ((const uint8_t*) kv.v + cell * (AD / 2) + lane * 4)),
                                __ldg((const unsigned short*) kv.vs + cell * (AD / 32) + lane / 4), 0, 0);
            }
        }
    };
    auto step = [&](uint4 kr, uint4 vr) {
        if (MODE != 0) {
            kr = deq_k8(make_uint2(kr.x, kr.y), __ushort_as_half((unsigned short) kr.z));
            vr = MODE == 2 ? deq_k8(make_uint2(vr.x, vr.y), __ushort_as_half((unsigned short) vr.z))
                           : deq_v4(vr.x, __ushort_as_half((unsigned short) vr.y));
        }
        float kf[8], vf[8];
        const __half2* k2 = (const __half2*) &kr;
        const __half2* v2 = (const __half2*) &vr;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = __half22float2(k2[j]), b = __half22float2(v2[j]);
            kf[2 * j] = a.x; kf[2 * j + 1] = a.y; vf[2 * j] = b.x; vf[2 * j + 1] = b.y;
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float sv = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) sv += qv[g][j] * kf[j];
            sv = warp_sum(sv);
            const float mn = fmaxf(m[g], sv), corr = __expf(m[g] - mn), p = __expf(sv - mn);
            l[g] = l[g] * corr + p;
            m[g] = mn;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[g][j] = acc[g][j] * corr + p * vf[j];
        }
    };
    // PF keys a warp in flight: their K / V loads issued together, then the steps
    int key = ks + w;
    for (; key + 4 * (PF - 1) < ke; key += 4 * PF) {
        uint4 kr[PF], vr[PF];
#pragma unroll
        for (int i = 0; i < PF; ++i) load(key + 4 * i, kr[i], vr[i]);
#pragma unroll
        for (int i = 0; i < PF; ++i) step(kr[i], vr[i]);
    }
    for (; key < ke; key += 4) {
        uint4 k0, v0;
        load(key, k0, v0);
        step(k0, v0);
    }
    if (lane == 0)
#pragma unroll
        for (int g = 0; g < G; ++g) { sm_m[w][g] = m[g]; sm_l[w][g] = l[g]; }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < 8; ++j) sm_o[w][g][lane * 8 + j] = acc[g][j];
    __syncthreads();
    const int rows = T * G;
    for (int i = threadIdx.x; i < G * AD; i += 128) {
        const int g = i / AD, d = i % AD;
        float M = -INFINITY;
#pragma unroll
        for (int ww = 0; ww < 4; ++ww) if (sm_l[ww][g] > 0.0f) M = fmaxf(M, sm_m[ww][g]);
        float a = 0.0f, L = 0.0f;
#pragma unroll
        for (int ww = 0; ww < 4; ++ww) {
            if (!(sm_l[ww][g] > 0.0f)) continue;
            const float f = __expf(sm_m[ww][g] - M);
            a += sm_o[ww][g][d] * f;
            L += sm_l[ww][g] * f;
        }
        const int head = kvh * G + g, row = t * G + g;
        if (n_split == 1) {
            const float gt = q_full[(size_t) t * ldq + (size_t) head * 2 * AD + AD + d];
            out[(size_t) t * qld + (size_t) head * AD + d] = (L > 0.0f ? a / L : 0.0f) * (MODE != 1 ? sigm(gt) : 1.0f);
        } else {
            part_o[(((size_t) split * n_kv + kvh) * rows + row) * AD + d] = a;
            if (d == 0) {
                part_ml[(((size_t) split * n_kv + kvh) * rows + row) * 2] = M;
                part_ml[(((size_t) split * n_kv + kvh) * rows + row) * 2 + 1] = L;
            }
        }
    }
}

// ---- the streaming kernel (STRATA_DENSE_ATTN_STREAM): 1-8 tokens (rows R = T G <= 16 RT) over a split of the keys,
// one block of 8 warps a (kv head, split).  K / V tiles of SK keys read with 16-byte loads - the next tile's issued
// into registers before this tile's math - and stored to shared memory; S = Q K^T and O += P V on the tensor cores
// (wmma 16x16x16, FP16 in, F32 sums), O in each warp's accumulator fragments.  The softmax a warp a row, a lane a key
// (or two): each row keeps a reference max m; P = exp(s - m) in FP16 (<= e^8 while no score passes m + 8), so O and l
// stay unscaled - a tile whose max passes m + 8 (rare after the first) moves m up and rescales O through a warp's
// shared scratch first.  Partials in attn_dec_kernel's layout (the merge folds them, the gate there).  FP16 cache, or
// K8V4 (MODE 1) unpacked to FP16 as a tile is stored - its output still rotated, unrotate_gate_kernel after the merge.
constexpr int SNT = 256, SLD = AD + 8, SRES = 8;   // threads, K / V / Q row pitch (halves), the rescale threshold
// one tile buffer: K stored for S, then V over it for O (the next tile's K and V loads issued separately, each after its
// own store, so both stay in flight).  P is written over its own row of S (a warp owns its rows); the rare rescale and
// the final write go through the tile buffer (between K's use and V's store / after the last tile) - so a 16-row block
// of 64-key tiles fits 2 a SM.
template <int RT, int SK_> struct StreamGeo {
    static constexpr int SK = SK_;                              // keys a tile
#ifndef Q35_STREAM_SPAD
#define Q35_STREAM_SPAD 12   // (S / P row pitch padding: 4 put rows 8 apart on the same banks - K8V8 1 token 0.277 -> 0.249 ms at 68K)
#endif
    static constexpr int SSLD = SK + Q35_STREAM_SPAD;           // S row pitch (floats); P (halves) over it
    static constexpr int NF = RT * 16 / 8;                      // O fragments a warp (RT x 16 over 8 warps)
    static constexpr int NFR = RT * SK / 16;                    // S fragments a tile
    // S's tasks (REUSE): a warp a (key column of 16, head-dimension part) - each K fragment loaded once for the RT row
    // tiles; the head dimension split in KS parts so there are 8 (summed in the softmax).  32 rows x 32 keys keeps
    // the per-fragment tasks instead (a warp a (row tile, key column, part), KS 2), whose S fits 2 blocks a SM
    static constexpr bool REUSE = !(RT == 2 && SK == 32);
    static constexpr int KS = REUSE ? 8 / (SK / 16) : (NFR >= 8 ? 1 : 8 / NFR);
    static constexpr size_t smem = (size_t) RT * 16 * SLD * 2 + (size_t) SK * SLD * 2 + (size_t) KS * RT * 16 * SSLD * 4 +
                                   3 * (size_t) RT * 16 * 4 + 16;
    // (2 a SM only for 32-key tiles: 16 rows x 64 keys also fits, but its register cap made it slower - 706 vs 762 GB/s)
    static constexpr int per_sm = SK == 32 && smem <= 48 * 1024 ? 2 : 1;
};
#ifdef Q35_STREAM_PROBE
// (attn_bench's -DQ35_STREAM_PROBE build only) 1: no math (the tiles' loads and stores alone), 2: no global loads (the
// math over the last tile's registers again) - the memory and the compute floors of the stream kernel
__constant__ int c_stream_probe;
#define Q35_PROBE(bit) ((c_stream_probe & (bit)) != 0)
// cycles a block spends in each phase of the tile loop (thread 0, between the barriers): K store, S = Q K^T, the
// softmax, the rescale, V store, O += P V - summed over the blocks
__device__ unsigned long long g_stream_phase[6];
#define Q35_MARK(i)                                                                                              \
    do {                                                                                                         \
        if (tid == 0) { const long long c_ = clock64(); ph[i] += c_ - ph_last; ph_last = c_; }                   \
    } while (0)
#else
#define Q35_MARK(i) do { } while (0)
#define Q35_PROBE(bit) false
#endif
template <int RT, int SK_, int MODE = 0>
__global__ void __launch_bounds__(SNT, (StreamGeo<RT, SK_>::per_sm)) attn_stream_kernel(const float* __restrict__ q,
                                                              Q35Kv kv, int p0, int T, int G,
                                                              int n_head, int max_ctx, int split_len, float scale,
                                                              float* __restrict__ part_o, float* __restrict__ part_ml) {
    using namespace nvcuda;
    using GE = StreamGeo<RT, SK_>;
    constexpr int RR = RT * 16, SK = GE::SK, SSLD = GE::SSLD, NF = GE::NF, NFR = GE::NFR, KS = GE::KS;
    constexpr int PLD = SSLD * 2;   // P's row pitch in halves (over its S row)
    extern __shared__ __align__(16) unsigned char smem[];
    __half* Qs = (__half*) smem;
    __half* KV = Qs + RR * SLD;               // the tile: K, then V
    float* Ss = (float*) (KV + SK * SLD);
    __half* Ps = (__half*) Ss;
    float* ms = Ss + KS * RR * SSLD;          // (after S's KS parts)
    float* ls = ms + RR;
    float* cs = ls + RR;
    int* any = (int*) (cs + RR);
    float* scr = (float*) KV;                  // [8 warps][16 x 16] when the tile holds neither K in use nor V
    const int kvh = blockIdx.x, split = blockIdx.y, n_kv = gridDim.x, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int R = T * G, qld = n_head * AD;
    const int ks = split * split_len, ke = min(ks + split_len, p0 + T);
    for (int i = tid; i < RR * AD; i += SNT) {
        const int r = i / AD, d = i % AD;
        float v = 0.0f;
        if (r < R) v = q[(size_t) (r / G) * qld + (size_t) (kvh * G + r % G) * AD + d] * scale;
        Qs[r * SLD + d] = __float2half_rn(v);
    }
    if (tid < RR) {
        ms[tid] = -INFINITY;
        ls[tid] = 0.0f;
    }
    // the tile's loads: FP16 - 16-byte chunks of K / V rows (32 a row); K8V4 - K: 16 INT8 codes a chunk (16 a row) with
    // their group's scale, V: 32 4-bit codes a chunk (8 a row) with theirs - unpacked to FP16 when stored
    constexpr int NUK = MODE == 0 ? SK * 32 / SNT : SK * 16 / SNT, NUV = MODE == 0 ? SK * 32 / SNT : MODE == 2 ? SK * 16 / SNT : SK * 8 / SNT;
    const size_t cell0 = (size_t) kvh * max_ctx;
    uint4 rk[NUK], rv[NUV];
    unsigned short sk_[MODE == 0 ? 1 : NUK], sv_[MODE == 0 ? 1 : NUV];
    auto issue_k = [&](int k0) {
#pragma unroll
        for (int i = 0; i < NUK; ++i) {
            const int u = tid + SNT * i;
            if (MODE == 0) {
                const int key = k0 + u / 32, c = u % 32;
                rk[i] = key < ke ? __ldg((const uint4*) ((const __half*) kv.k + (cell0 + key) * AD) + c) : make_uint4(0, 0, 0, 0);
            } else {
                const int key = k0 + u / 16, c = u % 16;
                if (key < ke) {
                    rk[i] = __ldg((const uint4*) ((const int8_t*) kv.k + (cell0 + key) * AD) + c);
                    sk_[i] = __ldg((const unsigned short*) kv.ks + (cell0 + key) * (AD / 32) + c / 2);
                } else {
                    rk[i] = make_uint4(0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u);   // (codes 0)
                    sk_[i] = 0;
                }
            }
        }
    };
    auto issue_v = [&](int k0) {
#pragma unroll
        for (int i = 0; i < NUV; ++i) {
            const int u = tid + SNT * i;
            if (MODE == 0) {
                const int key = k0 + u / 32, c = u % 32;
                rv[i] = key < ke ? __ldg((const uint4*) ((const __half*) kv.v + (cell0 + key) * AD) + c) : make_uint4(0, 0, 0, 0);
            } else if (MODE == 2) {   // (K8V8: V's codes as K's)
                const int key = k0 + u / 16, c = u % 16;
                if (key < ke) {
                    rv[i] = __ldg((const uint4*) ((const int8_t*) kv.v + (cell0 + key) * AD) + c);
                    sv_[i] = __ldg((const unsigned short*) kv.vs + (cell0 + key) * (AD / 32) + c / 2);
                } else {
                    rv[i] = make_uint4(0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u);
                    sv_[i] = 0;
                }
            } else {
                const int key = k0 + u / 8, c = u % 8;
                if (key < ke) {
                    rv[i] = __ldg((const uint4*) ((const uint8_t*) kv.v + (cell0 + key) * (AD / 2)) + c);
                    sv_[i] = __ldg((const unsigned short*) kv.vs + (cell0 + key) * (AD / 32) + c);
                } else {
                    rv[i] = make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);   // (codes 0)
                    sv_[i] = 0;
                }
            }
        }
    };
    auto store_k = [&]() {
#pragma unroll
        for (int i = 0; i < NUK; ++i) {
            const int u = tid + SNT * i;
            if (MODE == 0) {
                *(uint4*) (KV + (u / 32) * SLD + (u % 32) * 8) = rk[i];
            } else {
                const __half sc = __ushort_as_half(sk_[i]);
                __half* dst = KV + (u / 16) * SLD + (u % 16) * 16;
                *(uint4*) dst = deq_k8(make_uint2(rk[i].x, rk[i].y), sc);
                *(uint4*) (dst + 8) = deq_k8(make_uint2(rk[i].z, rk[i].w), sc);
            }
        }
    };
    auto store_v = [&]() {
#pragma unroll
        for (int i = 0; i < NUV; ++i) {
            const int u = tid + SNT * i;
            if (MODE == 0) {
                *(uint4*) (KV + (u / 32) * SLD + (u % 32) * 8) = rv[i];
            } else if (MODE == 2) {
                const __half sc = __ushort_as_half(sv_[i]);
                __half* dst = KV + (u / 16) * SLD + (u % 16) * 16;
                *(uint4*) dst = deq_k8(make_uint2(rv[i].x, rv[i].y), sc);
                *(uint4*) (dst + 8) = deq_k8(make_uint2(rv[i].z, rv[i].w), sc);
            } else {
                const __half sc = __ushort_as_half(sv_[i]);
                __half* dst = KV + (u / 8) * SLD + (u % 8) * 32;
                *(uint4*) dst = deq_v4(rv[i].x, sc);
                *(uint4*) (dst + 8) = deq_v4(rv[i].y, sc);
                *(uint4*) (dst + 16) = deq_v4(rv[i].z, sc);
                *(uint4*) (dst + 24) = deq_v4(rv[i].w, sc);
            }
        }
    };
    // O fragments: warp w the row tile mi = i / 2 ... as (mi, ni) = (j, w) and (j, w + 8) for j < RT
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[NF];
#pragma unroll
    for (int f = 0; f < NF; ++f) wmma::fill_fragment(fo[f], 0.0f);
    if (ks < ke) {
        issue_k(ks);
        issue_v(ks);
    }
    __syncthreads();
    constexpr int RPW = RR / (SNT / 32), KPL = SK / 32;   // rows a warp, keys a lane a tile
    float mrow[RPW], lsum[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) { mrow[i] = -INFINITY; lsum[i] = 0.0f; }
#ifdef Q35_STREAM_PROBE
    long long ph[6] = {}, ph_last = clock64();
#endif
    // one row tile (REUSE): the warp's head-dimension part of Q held in registers for every key tile, loaded on the
    // first (K8V8 1 / 2 tokens at 68K 0.26 -> 0.23 ms; FP16 ~1%)
    constexpr int QRN0 = GE::REUSE ? RT * (AD / 16 / KS) : 0;
    // (at most 8: 48 rows x 32 keys with its 12 fragments held, 8 tokens 0.615 -> 0.558 ms, still slower than 64 keys' 0.511)
    constexpr bool QR = QRN0 > 0 && QRN0 <= 8;
    constexpr int QRN = QR ? QRN0 : 1;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fqr[QRN];
    for (int k0 = ks; k0 < ke; k0 += SK) {
        store_k();
        if (tid == 0) *any = 0;
        __syncthreads();
        Q35_MARK(0);
        if (k0 + SK < ke && !Q35_PROBE(2)) issue_k(k0 + SK);
        if (!Q35_PROBE(1)) {
        // S = Q K^T: warp w the key column ni of 16 over the head-dimension part `part` (KS parts, summed in the
        // softmax), all RT row tiles - each K fragment loaded once for them; with one row tile two accumulators so the
        // mma chain is half as deep
        if (GE::REUSE) {
            const int part = w / (SK / 16), ni = w % (SK / 16);
            constexpr int NKK = AD / 16 / KS, NA = RT == 1 ? 2 : RT;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs[NA];
#pragma unroll
            for (int i = 0; i < NA; ++i) wmma::fill_fragment(fs[i], 0.0f);
#pragma unroll
            for (int k = 0; k < NKK; ++k) {
                const int kk = part * NKK + k;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, KV + ni * 16 * SLD + kk * 16, SLD);
#pragma unroll
                for (int mi = 0; mi < RT; ++mi) {
                    const int a = RT == 1 ? (k & 1) : mi;
                    if (QR) {
                        auto& fa = fqr[(k * RT + mi) % QRN];
                        if (k0 == ks) wmma::load_matrix_sync(fa, Qs + mi * 16 * SLD + kk * 16, SLD);
                        wmma::mma_sync(fs[a], fa, fb, fs[a]);
                    } else {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                        wmma::load_matrix_sync(fa, Qs + mi * 16 * SLD + kk * 16, SLD);
                        wmma::mma_sync(fs[a], fa, fb, fs[a]);
                    }
                }
            }
            if (RT == 1)
#pragma unroll
                for (int e = 0; e < fs[0].num_elements; ++e) fs[0].x[e] += fs[NA - 1].x[e];
#pragma unroll
            for (int mi = 0; mi < RT; ++mi)
                wmma::store_matrix_sync(Ss + (size_t) part * RR * SSLD + mi * 16 * SSLD + ni * 16, fs[mi], SSLD,
                                        wmma::mem_row_major);
        } else {
            for (int f = w; f < NFR * KS; f += SNT / 32) {
                const int part = f / NFR, mi = (f % NFR) / (SK / 16), ni = f % (SK / 16);
                constexpr int NKK = AD / 16 / KS;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs[2];
                wmma::fill_fragment(fs[0], 0.0f);
                wmma::fill_fragment(fs[1], 0.0f);
#pragma unroll
                for (int k = 0; k < NKK; ++k) {
                    const int kk = part * NKK + k;
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                    wmma::load_matrix_sync(fa, Qs + mi * 16 * SLD + kk * 16, SLD);
                    wmma::load_matrix_sync(fb, KV + ni * 16 * SLD + kk * 16, SLD);
                    wmma::mma_sync(fs[k & 1], fa, fb, fs[k & 1]);
                }
#pragma unroll
                for (int e = 0; e < fs[0].num_elements; ++e) fs[0].x[e] += fs[1].x[e];
                wmma::store_matrix_sync(Ss + (size_t) part * RR * SSLD + mi * 16 * SSLD + ni * 16, fs[0], SSLD,
                                        wmma::mem_row_major);
            }
        }
        __syncthreads();
        Q35_MARK(1);
        // the softmax: warp w the rows w, w + 8, ..; lane the keys lane, lane + 32; P over the row's own S.  Each lane
        // keeps its part of a row's total (lsum) and the row's reference max (mrow) in registers across the tiles; one
        // warp vote asks whether a score passes mrow + SRES - only then (the first tile, rarely after) the row's max
        // is reduced and mrow moves.  The totals are reduced once, after the last tile
        {
            float sv[RPW][KPL];
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const int r = w + 8 * i, last = r < R ? p0 + r / G : -1;
                float lmx = -INFINITY;
#pragma unroll
                for (int j = 0; j < KPL; ++j) {
                    const int key = k0 + lane + 32 * j;
                    float sc = Ss[r * SSLD + lane + 32 * j];
#pragma unroll
                    for (int pt = 1; pt < KS; ++pt) sc += Ss[pt * RR * SSLD + r * SSLD + lane + 32 * j];
                    sv[i][j] = (key <= last && key < ke) ? sc : -INFINITY;
                    lmx = fmaxf(lmx, sv[i][j]);
                }
                float corr = 1.0f;
                if (__any_sync(0xffffffffu, lmx > mrow[i] + SRES)) {   // (mrow -inf: any finite score)
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) lmx = fmaxf(lmx, __shfl_xor_sync(0xffffffffu, lmx, o));
                    corr = mrow[i] == -INFINITY ? 0.0f : __expf(mrow[i] - lmx);
                    mrow[i] = lmx;
                    lsum[i] *= corr;
                    if (lane == 0) *any = 1;
                }
                if (lane == 0) cs[r] = corr;
            }
            __syncwarp();   // (the rows' S read by every lane before P goes over them)
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const int r = w + 8 * i;
#pragma unroll
                for (int j = 0; j < KPL; ++j) {
                    const float pr = sv[i][j] == -INFINITY ? 0.0f : __expf(sv[i][j] - mrow[i]);
                    Ps[r * PLD + lane + 32 * j] = __float2half_rn(pr);
                    lsum[i] += pr;
                }
            }
        }
        }   // (!Q35_PROBE(1))
        __syncthreads();
        Q35_MARK(2);
        if (*any != 0) {   // (rare: a row's reference max moved - its O rows scaled through the tile buffer, K done)
#pragma unroll
            for (int i = 0; i < NF; ++i) {
                const int mi = i / 2, ni = w + 8 * (i % 2);
                float* sc = scr + w * 256;
                wmma::store_matrix_sync(sc, fo[i], 16, wmma::mem_row_major);
                __syncwarp();
                for (int e = lane; e < 256; e += 32) sc[e] *= cs[mi * 16 + e / 16];
                __syncwarp();
                wmma::load_matrix_sync(fo[i], sc, 16, wmma::mem_row_major);
                __syncwarp();
            }
            __syncthreads();
        }
        Q35_MARK(3);
        store_v();   // V over K
        __syncthreads();
        Q35_MARK(4);
        if (k0 + SK < ke && !Q35_PROBE(2)) issue_v(k0 + SK);
        if (!Q35_PROBE(1)) {
        // O += P V: each V fragment loaded once for the warp's RT row tiles
#pragma unroll
        for (int kk = 0; kk < SK / 16; ++kk) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) wmma::load_matrix_sync(fv[h], KV + kk * 16 * SLD + (w + 8 * h) * 16, SLD);
#pragma unroll
            for (int mi = 0; mi < RT; ++mi) {   // (each P fragment loaded once for both V fragments)
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
                wmma::load_matrix_sync(fp, Ps + mi * 16 * PLD + kk * 16, PLD);
#pragma unroll
                for (int h = 0; h < 2; ++h) wmma::mma_sync(fo[mi * 2 + h], fp, fv[h], fo[mi * 2 + h]);
            }
        }
        }   // (!Q35_PROBE(1))
        __syncthreads();
        Q35_MARK(5);
    }
#ifdef Q35_STREAM_PROBE
    if (tid == 0)
        for (int i = 0; i < 6; ++i) atomicAdd(&g_stream_phase[i], (unsigned long long) ph[i]);
#endif
    // the rows' totals and reference maxima
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        float t = lsum[i];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (lane == 0) { ms[w + 8 * i] = mrow[i]; ls[w + 8 * i] = t; }
    }
    __syncthreads();
    // the partials (unnormalised O, its reference max and total) through the tile buffer
#pragma unroll
    for (int i = 0; i < NF; ++i) {
        const int mi = i / 2, ni = w + 8 * (i % 2);
        float* sc = scr + w * 256;
        wmma::store_matrix_sync(sc, fo[i], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = mi * 16 + e / 16;
            if (r < R) part_o[(((size_t) split * n_kv + kvh) * R + r) * AD + ni * 16 + e % 16] = sc[e];
        }
        __syncwarp();
    }
    if (tid < R) {
        part_ml[(((size_t) split * n_kv + kvh) * R + tid) * 2] = ms[tid];
        part_ml[(((size_t) split * n_kv + kvh) * R + tid) * 2 + 1] = ls[tid];
    }
}

// K8V8's keys [0, n_keys) of every KV head, unpacked to FP16 at the same cells (16 threads a cell, each 16 values of
// K and of V): the prompt path's attention then reads the FP16 copy instead of unpacking every tile once a query tile
__global__ void kv_expand_kernel(Q35Kv kv, int n_keys, __half* __restrict__ k16, __half* __restrict__ v16) {
    const int key = blockIdx.x * 16 + threadIdx.x / 16, c = threadIdx.x % 16, kvh = blockIdx.y;
    if (key >= n_keys) return;
    const size_t cell = (size_t) kvh * kv.max_ctx + key;
    const uint4 kc = __ldg((const uint4*) ((const int8_t*) kv.k + cell * AD) + c);
    const uint4 vc = __ldg((const uint4*) ((const int8_t*) kv.v + cell * AD) + c);
    const __half ks = __ushort_as_half(__ldg((const unsigned short*) kv.ks + cell * (AD / 32) + c / 2));
    const __half vs = __ushort_as_half(__ldg((const unsigned short*) kv.vs + cell * (AD / 32) + c / 2));
    uint4* kd = (uint4*) (k16 + cell * AD + c * 16);
    uint4* vd = (uint4*) (v16 + cell * AD + c * 16);
    kd[0] = deq_k8(make_uint2(kc.x, kc.y), ks);
    kd[1] = deq_k8(make_uint2(kc.z, kc.w), ks);
    vd[0] = deq_k8(make_uint2(vc.x, vc.y), vs);
    vd[1] = deq_k8(make_uint2(vc.z, vc.w), vs);
}

// block = row, AD threads: fold the splits' partials, then the output gate
__global__ void attn_merge_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml, int rows,
                                  int n_split, int G, int n_head, const float* __restrict__ q_full, int ldq,
                                  float* __restrict__ out, int p0, int split_len, int gated) {
    const int row = blockIdx.x, d = threadIdx.x;
    const int kvh = blockIdx.y, tok = row / G, head = kvh * G + row % G;
    part_o += (size_t) kvh * rows * AD;
    part_ml += (size_t) kvh * rows * 2;
    const int rows_all = rows * gridDim.y;   // a split's partials: [n_kv][rows]
    const int n_use = min(n_split, (p0 + tok) / split_len + 1);   // the splits that hold keys of this row
    float M = -INFINITY;
    for (int s = 0; s < n_use; ++s) {
        const float l = part_ml[((size_t) s * rows_all + row) * 2 + 1];
        if (l > 0.0f) M = fmaxf(M, part_ml[((size_t) s * rows_all + row) * 2]);
    }
    float a = 0.0f, L = 0.0f;
    for (int s = 0; s < n_use; ++s) {
        const float l = part_ml[((size_t) s * rows_all + row) * 2 + 1];
        if (!(l > 0.0f)) continue;
        const float f = __expf(part_ml[((size_t) s * rows_all + row) * 2] - M);
        a += part_o[((size_t) s * rows_all + row) * AD + d] * f;
        L += l * f;
    }
    const float gt = q_full[(size_t) tok * ldq + (size_t) head * 2 * AD + AD + d];
    out[(size_t) tok * n_head * AD + (size_t) head * AD + d] = (L > 0.0f ? a / L : 0.0f) * (gated ? sigm(gt) : 1.0f);
}

// ------------------------------------------------------------------------------------------------ small matmuls
// block = (output row, 8 tokens), 256 threads along n_in; the 8 sums reduced over the block
__global__ void __launch_bounds__(256) gemv_f32_cols_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                            int ldx, float* __restrict__ y, int n_in, int n_out, int T) {
    constexpr int NC = 8;
    const int r = blockIdx.x, t0 = blockIdx.y * NC, nc = min(NC, T - t0);
    const float* wr = W + (size_t) r * n_in;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) {
        const float w = wr[i];
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < nc) acc[c] += w * x[(size_t) (t0 + c) * ldx + i];
    }
    __shared__ float sh[NC][8];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0) sh[c][wp] = v;
    }
    __syncthreads();
    if (threadIdx.x < nc) {
        float v = 0.0f;
        for (int k = 0; k < (int) (blockDim.x >> 5); ++k) v += sh[threadIdx.x][k];
        y[(size_t) (t0 + threadIdx.x) * n_out + r] = v;
    }
}

__global__ void argmax_kernel(const float* __restrict__ x, int64_t ld, int n, int32_t* __restrict__ out) {
    const float* xr = x + (size_t) blockIdx.x * ld;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        best = lane < (int) (blockDim.x >> 5) ? sv[lane] : -INFINITY;
        bi = lane < (int) (blockDim.x >> 5) ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) out[blockIdx.x] = bi;
    }
}

// the greedy pick over a long row in two steps (STRATA_DENSE_ARGMAX_MB, the default): blocks of 256 threads each take
// kAmChunk values of a row (16-byte loads when the row allows) -> (best, index) partials; then a warp a row reduces them.
// Ties keep the lowest index, as argmax_kernel: the same token.
constexpr int kAmChunk = 8192, kAmMaxParts = 64, kAmMaxRows = 16;
__device__ __forceinline__ void am_take(float v, int i, float& best, int& bi) {
    if (v > best || (v == best && i < bi)) { best = v; bi = i; }
}
__global__ void __launch_bounds__(256) argmax_part_kernel(const float* __restrict__ x, int64_t ld, int n,
                                                          float* __restrict__ pv, int* __restrict__ pi, int parts) {
    const float* xr = x + (size_t) blockIdx.y * ld;
    const int lo = blockIdx.x * kAmChunk, hi = min(n, lo + kAmChunk);
    float best = -INFINITY;
    int bi = 0x7fffffff;
    if ((((uintptr_t) (xr + lo)) & 15u) == 0 && ((hi - lo) & 3) == 0) {
        const float4* x4 = (const float4*) (xr + lo);
        for (int i = threadIdx.x; i < (hi - lo) / 4; i += 256) {
            const float4 v = x4[i];
            const int b = lo + 4 * i;
            am_take(v.x, b, best, bi);
            am_take(v.y, b + 1, best, bi);
            am_take(v.z, b + 2, best, bi);
            am_take(v.w, b + 3, best, bi);
        }
    } else {
        for (int i = lo + threadIdx.x; i < hi; i += 256) am_take(xr[i], i, best, bi);
    }
    for (int o = 16; o > 0; o >>= 1) am_take(__shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, bi, o), best, bi);
    __shared__ float sv[8];
    __shared__ int si[8];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        best = lane < 8 ? sv[lane] : -INFINITY;
        bi = lane < 8 ? si[lane] : 0x7fffffff;
        for (int o = 4; o > 0; o >>= 1) am_take(__shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, bi, o), best, bi);
        if (lane == 0) { pv[blockIdx.y * parts + blockIdx.x] = best; pi[blockIdx.y * parts + blockIdx.x] = bi; }
    }
}
__global__ void argmax_join_kernel(const float* __restrict__ pv, const int* __restrict__ pi, int parts, int32_t* __restrict__ out) {
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < parts; i += 32) am_take(pv[blockIdx.x * parts + i], pi[blockIdx.x * parts + i], best, bi);
    for (int o = 16; o > 0; o >>= 1) am_take(__shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, bi, o), best, bi);
    if (threadIdx.x == 0) out[blockIdx.x] = bi;
}

__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void swiglu_rows_kernel(const float* __restrict__ g, const float* __restrict__ u, int ldg,
                                   float* __restrict__ o, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[(size_t) blockIdx.y * ldg + i];
    o[(size_t) blockIdx.y * n + i] = (a / (1.0f + __expf(-a))) * u[(size_t) blockIdx.y * ldg + i];
}

unsigned blocks(int64_t n, int b) { return (unsigned) ((n + b - 1) / b); }

}  // namespace

void q35_rms_rows(const float* x, int ldx, const float* w, float* out_f32, uint16_t* out_f16, int rows, int cols,
                  float eps, void* stream) {
    rms_rows_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(x, ldx, w, out_f32, (__half*) out_f16, cols, eps);
    check("q35_rms_rows");
}

void q35_add_rms_rows(float* x, int ldx, const float* a, const float* w, float* out_f32, uint16_t* out_f16, int rows,
                      int cols, float eps, void* stream) {
    add_rms_rows_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(x, ldx, a, w, out_f32, (__half*) out_f16, cols, eps);
    check("q35_add_rms_rows");
}

void q35_swiglu_h(const float* gate, const float* up, float* out, uint16_t* out_f16, int64_t n, void* stream) {
    swiglu_h_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(gate, up, out, (__half*) out_f16, n);
    check("q35_swiglu_h");
}

void q35_to_half(const float* src, uint16_t* dst, int64_t n, void* stream) {
    to_half_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(src, (__half*) dst, n);
    check("q35_to_half");
}

void q35_conv_seq(float* conv_state, const float* raw, int ld, const float* w, float* out, int T, int C, bool commit,
                  void* stream) {
    if (T > 8 && T <= 65535) {   // (a prompt chunk: the tokens in parallel, then the state moved on)
        conv_par_kernel<<<dim3(blocks(C, 128), T), 128, 0, (cudaStream_t) stream>>>(conv_state, raw, ld, w, out, T, C);
        if (commit) conv_commit_kernel<<<blocks(C, 128), 128, 0, (cudaStream_t) stream>>>(conv_state, raw, ld, T, C);
    } else {
        conv_seq_kernel<<<blocks(C, 128), 128, 0, (cudaStream_t) stream>>>(conv_state, raw, ld, w, out, T, C, commit);
    }
    check("q35_conv_seq");
}

void q35_qk_l2(float* x, int ld, int T, int n_rows, float eps, void* stream) {
    qk_l2_kernel<<<dim3(n_rows, T), 128, 0, (cudaStream_t) stream>>>(x, ld, eps);
    check("q35_qk_l2");
}

bool q35_gdn_prep(float* conv_state, const float* raw, int ld, const float* w, float* out, int T, int C, bool commit,
                  int n_qk_rows, float eps, float* alpha, float* beta, int ldg, const float* dt, const float* ssm_a, int h_v,
                  void* stream) {
    if (C % 128 != 0 || n_qk_rows * 128 > C) return false;
    const int nb = C / 128 + (T * h_v + 127) / 128;
    gdn_prep_kernel<<<nb, 128, 0, (cudaStream_t) stream>>>(conv_state, raw, ld, w, out, T, C, commit, n_qk_rows, eps, alpha,
                                                           beta, dt, ssm_a, h_v, ldg);
    check("q35_gdn_prep");
    return true;
}

void q35_gate_beta(float* alpha, float* beta, int ld, const float* dt, const float* ssm_a, int T, int h_v, void* stream) {
    gate_beta_kernel<<<blocks((int64_t) T * h_v, 256), 256, 0, (cudaStream_t) stream>>>(alpha, beta, dt, ssm_a,
                                                                                       T * h_v, h_v, ld);
    check("q35_gate_beta");
}

void q35_set_gdn_grouped(int group) { cudaMemcpyToSymbol(c_gdn_group, &group, sizeof group); }

static float* g_gdn_scratch = nullptr;
static int64_t g_gdn_scratch_floats = 0;
void q35_set_gdn_scratch(float* p, int64_t floats) { g_gdn_scratch = p; g_gdn_scratch_floats = floats; }

void q35_gdn_seq(float* state, const float* x, int ld, const float* g, const float* b, int ldg, float* o, int T,
                 int h_k, int h_v, bool commit, void* stream, float* d_rec) {
    if (T <= 0) return;
    // (STRATA_DENSE_GDN_CHUNK=1: prompt chunks of 64+ tokens through the chunked F32 form, in ranges that fit the
    // scratch q35_set_gdn_scratch gave)
    static const bool chunked = std::getenv("STRATA_DENSE_GDN_CHUNK") && std::atoi(std::getenv("STRATA_DENSE_GDN_CHUNK")) != 0;
    if (chunked && commit && !d_rec && T >= 64 && g_gdn_scratch && ld % 4 == 0) {
        constexpr int S = 128;
        const int64_t per_tok = (int64_t) 2 * h_v * S + (int64_t) h_v * GC + 2 * h_v + 8;
        int R = (int) std::min<int64_t>(T, g_gdn_scratch_floats / per_tok / GC * GC);
        if (R >= GC) {
            cudaStream_t st = (cudaStream_t) stream;
            for (int r0 = 0; r0 < T; r0 += R) {
                const int n = std::min(R, T - r0), nch = (n + GC - 1) / GC;
                float* U = g_gdn_scratch;
                float* Wm = U + (size_t) nch * GC * h_v * S;
                float* Mm = Wm + (size_t) nch * GC * h_v * S;
                float* F = Mm + (size_t) nch * h_v * GC * GC;
                gdn_chunk_prep_kernel<<<dim3(nch, h_v), 256, 0, st>>>(x + (size_t) r0 * ld, ld, g + (size_t) r0 * ldg,
                                                                      b + (size_t) r0 * ldg, ldg, n, h_k, h_v, U, Wm,
                                                                      Mm, F);
                gdn_chunk_scan_kernel<<<dim3(h_v, S / 32), 256, 0, st>>>(state, x + (size_t) r0 * ld, ld, n, h_k, h_v,
                                                                         U, Wm, Mm, F, o + (size_t) r0 * h_v * S);
            }
            check("q35_gdn_chunked");
            return;
        }
    }
    static const bool old = std::getenv("STRATA_DENSE_GDN_OLD") && std::atoi(std::getenv("STRATA_DENSE_GDN_OLD")) != 0;
    // (STRATA_DENSE_GDN_PIPE=0: gdn_seq4_kernel, the next token's loads not overlapped)
    static const bool pipe = !std::getenv("STRATA_DENSE_GDN_PIPE") || std::atoi(std::getenv("STRATA_DENSE_GDN_PIPE")) != 0;
    if (old)
        gdn_seq_kernel<<<h_v, 128, 0, (cudaStream_t) stream>>>(state, x, ld, g, b, ldg, o, T, h_k, h_v, commit);
    else if (pipe)
        gdn_seq4p_kernel<<<dim3(h_v, 4), 128, 0, (cudaStream_t) stream>>>(state, x, ld, g, b, ldg, o, T, h_k, h_v, commit,
                                                                          d_rec);
    else
        gdn_seq4_kernel<<<dim3(h_v, 4), 128, 0, (cudaStream_t) stream>>>(state, x, ld, g, b, ldg, o, T, h_k, h_v, commit);
    check("q35_gdn_seq");
}

// commit_gdn_kernel with the verify pass's recorded updates d (Q35CommitLayer::d): no dot product, cross-warp sum or
// its barrier a token - st = st dec + k d in the same order as commit_gdn_kernel's (the same bits)
__global__ void __launch_bounds__(128) commit_gdn_rec_kernel(const Q35CommitLayer* __restrict__ tab, int n, int C,
                                                             int h_k, int h_v) {
    constexpr int S = 128, R = 32;
    const Q35CommitLayer& L = tab[blockIdx.z];
    float* __restrict__ state = L.state;
    const float* __restrict__ x = L.conv;
    const float* __restrict__ g = L.ab;
    const int ldg = 2 * h_v;
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, kh = gdn_kh(h, h_k);
    const int j = blockIdx.y * 32 + lane, i0 = w * R;
    __shared__ float sk[2][S];
    float st[R];
#pragma unroll
    for (int i = 0; i < R; ++i) st[i] = GST_LD(state, ((size_t) (i0 + i) * h_v + h) * S + j);
    for (int t = 0; t < n; ++t) {
        float* skb = sk[t & 1];
        skb[threadIdx.x] = x[(size_t) t * C + h_k * S + kh * S + threadIdx.x];
        __syncthreads();
        const float dec = expf(g[(size_t) t * ldg + h]);
        const float d = L.d[((size_t) t * h_v + h) * S + j];
#pragma unroll
        for (int i = 0; i < R; ++i) st[i] *= dec;
#pragma unroll
        for (int i = 0; i < R; ++i) st[i] += skb[i0 + i] * d;
    }
#pragma unroll
    for (int i = 0; i < R; ++i) GST_ST(state, ((size_t) (i0 + i) * h_v + h) * S + j, st[i]);
}

void q35_commit_all(const Q35CommitLayer* tab_dev, int n_layers, int n, int C, int h_k, int h_v, void* stream, bool rec) {
    if (n <= 0 || n_layers <= 0) return;
    commit_conv_kernel<<<dim3(blocks(C, 128), n_layers), 128, 0, (cudaStream_t) stream>>>(tab_dev, n, C);
    check("q35_commit_conv");
    if (rec) commit_gdn_rec_kernel<<<dim3(h_v, 4, n_layers), 128, 0, (cudaStream_t) stream>>>(tab_dev, n, C, h_k, h_v);
    else commit_gdn_kernel<<<dim3(h_v, 4, n_layers), 128, 0, (cudaStream_t) stream>>>(tab_dev, n, C, h_k, h_v);
    check("q35_commit_gdn");
}

void q35_gdn_out(const float* o, const float* z, const float* w, float* y, int rows, float eps, void* stream) {
    dense_gdn_out_norm_silu(o, z, w, y, rows, 128, eps, stream);
}

uint64_t q35_kv_bytes(int mode, int n_kv, int D, int max_ctx) {
    const uint64_t cells = (uint64_t) n_kv * max_ctx;
    return mode == 0 ? cells * D * 2 * 2 : mode == 2 ? cells * (D + D / 32 * 2) * 2 : cells * (D + D / 32 * 2 + D / 2 + D / 32 * 2);
}

void q35_attn_prep(const float* q_full, int ldq, const float* k, const float* v, int ldk, const float* q_norm,
                   const float* k_norm, float eps, float rope_base, int n_rot, int p0, int T, int n_head, int n_kv,
                   int D, float* q_out, const Q35Kv& kv, void* stream) {
    if (D != 256 || n_rot > D) throw std::invalid_argument("q35_attn_prep: head size 256");
    const float theta_scale = powf(rope_base, -2.0f / n_rot);
    attn_prep_kernel<<<dim3(T, (q_full ? n_head : 0) + n_kv), D, 0, (cudaStream_t) stream>>>(
        q_full, ldq, k, v, ldk, q_norm, k_norm, eps, theta_scale, n_rot, p0, n_head, D, q_out, kv);
    check("q35_attn_prep");
}

namespace {
constexpr int kSplitLen = 128;   // the shortest split (a decode step at a short context still spreads)
constexpr int kMaxCols8 = 8;     // (the engine sizes the attention scratch for 8 tokens)
int g_dec_max = -1;
int g_attn_var = -1;
}

// the CUDA-core decode kernel serves windows of up to this many tokens (STRATA_DENSE_ATTN_DEC, default below)
int q35_attn_dec_max_tokens() {
    if (g_dec_max < 0) {
        const char* e = std::getenv("STRATA_DENSE_ATTN_DEC");
        g_dec_max = e ? std::atoi(e) : 1;   // (one token: 28K plain decode 29.8 -> 30.8 tok/s; 2+ rows: the tile kernel)
    }
    return g_dec_max;
}
void q35_set_attn_dec_max_tokens(int n) { g_dec_max = n; }
void q35_set_attn_var(int v) { g_attn_var = v; }

int64_t q35_attn_scratch_floats(int T, int n_head, int D, int max_ctx) {
    const int64_t splits = (max_ctx + kSplitLen - 1) / kSplitLen;
    return splits * (int64_t) T * n_head * (D + 2);
}

namespace {
// the decode kernel's keys in flight a warp (2, 4, 8) and its target block count (320: ~4 blocks a SM on a V100);
// STRATA_DENSE_ATTN_DEC_PF / STRATA_DENSE_ATTN_DEC_BLOCKS, q35_set_attn_dec_tune for benchmarks
int g_dec_pf = -1, g_dec_blocks = -1, g_dec_minb = -1;
int g_attn_stream = -1, g_stream_blocks = -1, g_stream_sk = -1;
}
#ifdef Q35_STREAM_PROBE
void q35_set_attn_stream_probe(int probe) { cudaMemcpyToSymbol(c_stream_probe, &probe, sizeof probe); }
void q35_stream_phases(unsigned long long* out, bool reset) {
    cudaMemcpyFromSymbol(out, g_stream_phase, 6 * sizeof(unsigned long long));
    if (reset) {
        const unsigned long long z[6] = {};
        cudaMemcpyToSymbol(g_stream_phase, z, sizeof z);
    }
}
#endif
int g_attn_pp = -1;
void q35_set_attn_pp(int on) { g_attn_pp = on; }
void q35_set_attn_stream(int on, int blocks, int sk) {
    g_attn_stream = on;
    g_stream_blocks = blocks;
    g_stream_sk = sk;
}
void q35_set_attn_dec_tune(int pf, int blocks, int minb) {
    g_dec_pf = pf;
    g_dec_blocks = blocks;
    g_dec_minb = minb;
}

void q35_attn(const float* q, const Q35Kv& kv, int p0, int T, int n_head, int n_kv, int D, const float* q_full,
              int ldq, float* out, float* scratch, int splits, void* stream) {
    if (D != AD || n_head % n_kv != 0 || n_head / n_kv > 8) throw std::invalid_argument("q35_attn: geometry");
    static const bool attr_ok =
        cudaFuncSetAttribute(attn_tc_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kAttnSmem) == cudaSuccess &&
        cudaFuncSetAttribute(attn_tc_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kAttnSmem) == cudaSuccess &&
        cudaFuncSetAttribute(attn_tc_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kAttnSmem) == cudaSuccess;
    if (!attr_ok) throw std::runtime_error("q35_attn: shared memory attribute");
    const int max_ctx = kv.max_ctx;
    const int G = n_head / n_kv, rows = T * G;
    // verify windows (2-8 tokens): STRATA_DENSE_ATTN_VAR 1 = 32-row tiles of 2 warps (32 keys a step), 2 = the same with
    // 16 keys a step (2 blocks a SM); 0 = the 64-row tile
    static int var = -1;
    // (attn_bench 64K FP16 a layer, variant 2: 2 rows 0.674 -> 0.450 ms, 4 rows 0.698 -> 0.542, 8 rows 0.753 -> 0.876;
    // the engine, 27B MTP3 at 28K, two cooled rounds: verify 40.1 / 40.3 -> 39.3 / 39.3 ms, 58.4 / 58.2 -> 59.5 / 59.5
    // tok/s - the default for 2-4 tokens (-2); 0: the 64-row tile)
    if (var < 0) var = std::getenv("STRATA_DENSE_ATTN_VAR") ? std::atoi(std::getenv("STRATA_DENSE_ATTN_VAR")) : -2;
    const int vsel = g_attn_var >= 0 ? g_attn_var : var;
    // STRATA_DENSE_ATTN_DIRECT=1: the 64-row tile on an FP16 cache reads its K / V fragments in place (2 blocks a SM) -
    // slower (27B 28K prompt, two cooled rounds: 807 / 788 -> 755 / 748 tok/s; strided fragment loads from L1 lose to the
    // staged tile), so off
    static const bool direct = std::getenv("STRATA_DENSE_ATTN_DIRECT") && std::atoi(std::getenv("STRATA_DENSE_ATTN_DIRECT")) != 0;
    // (STRATA_DENSE_ATTN_PP=0: prompt chunks on attn_tc_kernel's 4 warps)
    // (STRATA_DENSE_ATTN_PP: 0 attn_tc_kernel, 1 attn_pp_kernel, 2 attn_pp_kernel with its halves' own barriers)
    static const int pp_env = std::getenv("STRATA_DENSE_ATTN_PP") ? std::atoi(std::getenv("STRATA_DENSE_ATTN_PP")) : 2;
    const int pp_mode = g_attn_pp >= 0 ? g_attn_pp : pp_env;
    const bool pp_on = pp_mode != 0;
    const int v = vsel == -2 ? (T >= 2 && T <= 4 ? 2 : 0) : (T >= 2 && T <= 8) ? vsel : 0;
    const int AR_v = v ? 32 : AR, AK_v = v == 2 ? 16 : AK, AW_v = v ? 2 : AW;
    const int tiles = (rows + AR_v - 1) / AR_v;
    const int n_keys = p0 + T;
    // on by default for the FP16 and K8V8 caches (FP16, chain205: MTP 57.3 -> 61.2 tok/s at 35K, the same tokens;
    // K8V8 the fastest of its kernels at every width); K8V4 only when asked for
    static const int stream_env = std::getenv("STRATA_DENSE_ATTN_STREAM") ? std::atoi(std::getenv("STRATA_DENSE_ATTN_STREAM")) : -1;
    const int stream_sel = g_attn_stream >= 0 ? g_attn_stream : stream_env;
    const bool stream_on = stream_sel > 0 || (stream_sel < 0 && kv.mode != 1);
    if (stream_on && splits <= 0 && rows <= 48) {
        const int rt = rows <= 16 ? 1 : rows <= 32 ? 2 : 3;
        // keys a tile by rows (attn_bench, V100, FP16, 68K / 131K): 16 rows 64 (779 / 814 GB/s), 32 rows 32 at 2 blocks a
        // SM (602 / 720), 48 rows 64 (468 / 505)
        static const int sk_env = std::getenv("STRATA_DENSE_ATTN_STREAM_SK") ? std::atoi(std::getenv("STRATA_DENSE_ATTN_STREAM_SK")) : 0;
        // (K8V4, 16 rows: 32 keys at 2 a SM, 313 -> 327 / 366 -> 387 GB/s)
        const int skv = g_stream_sk > 0 ? g_stream_sk : sk_env > 0 ? sk_env : rt == 2 || (rt == 1 && kv.mode != 0) ? 32 : 64;
        static int sms = 0;
        static bool attr = false;
        if (!attr) {
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
#define Q35_SATTR(RTV, SKV) cudaFuncSetAttribute(attn_stream_kernel<RTV, SKV, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                                 (int) StreamGeo<RTV, SKV>::smem);                                       \
                            cudaFuncSetAttribute(attn_stream_kernel<RTV, SKV, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                                 (int) StreamGeo<RTV, SKV>::smem);                                       \
                            cudaFuncSetAttribute(attn_stream_kernel<RTV, SKV, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                                 (int) StreamGeo<RTV, SKV>::smem)
            Q35_SATTR(1, 32); Q35_SATTR(1, 64); Q35_SATTR(2, 32); Q35_SATTR(2, 64); Q35_SATTR(3, 32); Q35_SATTR(3, 64);
#undef Q35_SATTR
            attr = true;
        }
        const int per_sm = rt == 1 ? (skv == 32 ? StreamGeo<1, 32>::per_sm : StreamGeo<1, 64>::per_sm)
                         : rt == 2 ? (skv == 32 ? StreamGeo<2, 32>::per_sm : StreamGeo<2, 64>::per_sm)
                                   : (skv == 32 ? StreamGeo<3, 32>::per_sm : StreamGeo<3, 64>::per_sm);
        const int target = g_stream_blocks > 0 ? g_stream_blocks : sms * per_sm;   // (one wave: the blocks that fit)
        const int cap = (int) (q35_attn_scratch_floats(T, n_head, D, max_ctx) / ((int64_t) rows * n_kv * (D + 2)));
        const int SKr = skv == 32 ? 32 : 64;
        int n_split = std::max(1, std::min({target / n_kv, (n_keys + SKr - 1) / SKr, cap}));
        int split_len = (n_keys + n_split - 1) / n_split;
        split_len = (split_len + SKr - 1) / SKr * SKr;   // (whole tiles a split)
        n_split = (n_keys + split_len - 1) / split_len;
        float* part_o = scratch;
        float* part_ml = scratch + (size_t) n_split * rows * n_kv * D;
        const dim3 grid(n_kv, n_split);
        const float sc = 1.0f / sqrtf((float) D);
#define Q35_STREAM(RTV, SKV)                                                                                     \
        do {                                                                                                         \
            if (kv.mode == 0)                                                                                        \
                attn_stream_kernel<RTV, SKV, 0><<<grid, SNT, StreamGeo<RTV, SKV>::smem, (cudaStream_t) stream>>>(     \
                    q, kv, p0, T, G, n_head, max_ctx, split_len, sc, part_o, part_ml);                               \
            else if (kv.mode == 2)                                                                                   \
                attn_stream_kernel<RTV, SKV, 2><<<grid, SNT, StreamGeo<RTV, SKV>::smem, (cudaStream_t) stream>>>(     \
                    q, kv, p0, T, G, n_head, max_ctx, split_len, sc, part_o, part_ml);                               \
            else                                                                                                     \
                attn_stream_kernel<RTV, SKV, 1><<<grid, SNT, StreamGeo<RTV, SKV>::smem, (cudaStream_t) stream>>>(     \
                    q, kv, p0, T, G, n_head, max_ctx, split_len, sc, part_o, part_ml);                               \
        } while (0)
        if (rt == 1) { if (skv == 32) Q35_STREAM(1, 32); else Q35_STREAM(1, 64); }
        else if (rt == 2) { if (skv == 32) Q35_STREAM(2, 32); else Q35_STREAM(2, 64); }
        else { if (skv == 32) Q35_STREAM(3, 32); else Q35_STREAM(3, 64); }
#undef Q35_STREAM
        check("q35_attn_stream");
        attn_merge_kernel<<<dim3(rows, n_kv), D, 0, (cudaStream_t) stream>>>(part_o, part_ml, rows, n_split, G, n_head,
                                                                             q_full, ldq, out, p0, split_len, kv.mode != 1);
        check("q35_attn_merge");
        if (kv.mode == 1) {   // (K8V4: the output still rotated - back by H, then the gate)
            unrotate_gate_kernel<<<dim3(T, n_head), 256, 0, (cudaStream_t) stream>>>(out, n_head, q_full, ldq);
            check("q35_attn_unrotate");
        }
        return;
    }
    if (G == 6 && splits <= 0 && T <= q35_attn_dec_max_tokens()) {
        // the CUDA-core decode kernel: ~4 blocks a SM over (kv head, token), at least 64 keys a split
        const int cap = (int) (q35_attn_scratch_floats(T, n_head, D, max_ctx) / ((int64_t) rows * n_kv * (D + 2)));
        static const int pf_env = std::getenv("STRATA_DENSE_ATTN_DEC_PF") ? std::atoi(std::getenv("STRATA_DENSE_ATTN_DEC_PF")) : 2;
        static const int bl_env = std::getenv("STRATA_DENSE_ATTN_DEC_BLOCKS") ? std::atoi(std::getenv("STRATA_DENSE_ATTN_DEC_BLOCKS")) : 320;
        static const int mb_env = std::getenv("STRATA_DENSE_ATTN_DEC_MINB") ? std::atoi(std::getenv("STRATA_DENSE_ATTN_DEC_MINB")) : 1;
        const int pf = g_dec_pf > 0 ? g_dec_pf : pf_env, blocks = g_dec_blocks > 0 ? g_dec_blocks : bl_env;
        const int minb = g_dec_minb > 0 ? g_dec_minb : mb_env;
        int n_split = std::max(1, std::min({blocks / (n_kv * T), (n_keys + 63) / 64, cap}));
        const int split_len = (n_keys + n_split - 1) / n_split;
        n_split = (n_keys + split_len - 1) / split_len;
        if ((int64_t) n_split * rows * n_kv * (D + 2) > q35_attn_scratch_floats(T, n_head, D, max_ctx))
            throw std::runtime_error("q35_attn: scratch too small");
        float* part_o = scratch;
        float* part_ml = scratch + (size_t) n_split * rows * n_kv * D;
        const dim3 grid(n_kv, n_split, T);
        const float sc = 1.0f / sqrtf((float) D);
#define Q35_DEC(M, PFV, MB)                                                                                     \
        attn_dec_kernel<M, 6, PFV, MB><<<grid, 128, 0, (cudaStream_t) stream>>>(q, kv, p0, n_head, max_ctx, split_len,  \
                                                                               sc, q_full, ldq, out, part_o, part_ml)
        if (kv.mode == 0) {
            if (minb >= 4) { if (pf >= 4) Q35_DEC(0, 4, 4); else Q35_DEC(0, 2, 4); }
            else if (pf >= 8) Q35_DEC(0, 8, 1); else if (pf >= 4) Q35_DEC(0, 4, 1); else Q35_DEC(0, 2, 1);
        } else if (kv.mode == 2) {
            if (minb >= 4) { if (pf >= 4) Q35_DEC(2, 4, 4); else Q35_DEC(2, 2, 4); }
            else if (pf >= 8) Q35_DEC(2, 8, 1); else if (pf >= 4) Q35_DEC(2, 4, 1); else Q35_DEC(2, 2, 1);
        } else {
            if (minb >= 4) { if (pf >= 4) Q35_DEC(1, 4, 4); else Q35_DEC(1, 2, 4); }
            else if (pf >= 8) Q35_DEC(1, 8, 1); else if (pf >= 4) Q35_DEC(1, 4, 1); else Q35_DEC(1, 2, 1);
        }
#undef Q35_DEC
        check("q35_attn_dec");
        if (n_split > 1) {
            attn_merge_kernel<<<dim3(rows, n_kv), D, 0, (cudaStream_t) stream>>>(part_o, part_ml, rows, n_split, G, n_head,
                                                                                 q_full, ldq, out, p0, split_len,
                                                                                 kv.mode != 1);
            check("q35_attn_merge");
        }
        if (kv.mode == 1) {
            unrotate_gate_kernel<<<dim3(T, n_head), 256, 0, (cudaStream_t) stream>>>(out, n_head, q_full, ldq);
            check("q35_attn_unrotate");
        }
        return;
    }
    // splits: enough blocks to fill the GPU when the rows are few
    int n_split = splits;
    if (n_split <= 0) {
        // (rounded up: a short prompt chunk over a long cache - 256 rows of 6 heads, 96 blocks - takes 2 splits, not 1;
        // capped by the partials' scratch.  ideas.md 38: a 16-128-token continuation at 68K 10.6 -> 1.2-7.2 ms)
        const int blk = std::max(1, tiles * n_kv), target = v == 2 ? 320 : 160;
        const int want = (target + blk - 1) / blk;
        const int64_t room = q35_attn_scratch_floats(kMaxCols8, n_head, D, max_ctx) / ((int64_t) rows * n_kv * (D + 2));
        n_split = std::max(1, std::min({want, (n_keys + kSplitLen - 1) / kSplitLen, (int) std::max<int64_t>(1, room)}));
    }
    const int split_len = n_split == 1 ? n_keys : ((n_keys + n_split - 1) / n_split + AK - 1) / AK * AK;   // (AK: 32)
    n_split = (n_keys + split_len - 1) / split_len;
    if ((int64_t) n_split * rows * n_kv * (D + 2) > q35_attn_scratch_floats(T, n_head, D, max_ctx))
        throw std::runtime_error("q35_attn: scratch too small");
    float* part_o = scratch;
    float* part_ml = scratch + (size_t) n_split * rows * n_kv * D;
    const dim3 grid(tiles, n_kv, n_split);
    const float sc = 1.0f / sqrtf((float) D);
    auto go = [&](auto kern, size_t smem, int threads) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        kern<<<grid, threads, smem, (cudaStream_t) stream>>>(q, kv, p0, T, n_head, G, max_ctx, split_len, sc, q_full, ldq,
                                                             out, part_o, part_ml);
    };
    if (v == 1) {
        if (kv.mode == 0) go(attn_tc_kernel<0, 32, 32, 2>, attn_smem<32, 32, 2>(), 64);
        else if (kv.mode == 2) go(attn_tc_kernel<2, 32, 32, 2>, attn_smem<32, 32, 2>(), 64);
        else go(attn_tc_kernel<1, 32, 32, 2>, attn_smem<32, 32, 2>(), 64);
    } else if (v == 2) {
        if (kv.mode == 0) go(attn_tc_kernel<0, 32, 16, 2>, attn_smem<32, 16, 2>(), 64);
        else if (kv.mode == 2) go(attn_tc_kernel<2, 32, 16, 2>, attn_smem<32, 16, 2>(), 64);
        else go(attn_tc_kernel<1, 32, 16, 2>, attn_smem<32, 16, 2>(), 64);
    } else if (kv.mode == 0 && direct) {
        attn_tc_kernel<0, AR, AK, AW, true><<<grid, AW * 32, attn_smem<AR, AK, AW, true>(), (cudaStream_t) stream>>>(
            q, kv, p0, T, n_head, G, max_ctx, split_len, sc, q_full, ldq, out, part_o, part_ml);
    } else if (kv.mode == 0 && T > 8 && pp_on) {   // (a prompt chunk: the 8-warp kernel)
        static const bool pp_attr =
            cudaFuncSetAttribute(attn_pp_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kPpSmem) == cudaSuccess &&
            cudaFuncSetAttribute(attn_pp_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kPpSmem) == cudaSuccess;
        if (!pp_attr) throw std::runtime_error("q35_attn: attn_pp_kernel shared memory attribute");
        if (pp_mode == 2)
            attn_pp_kernel<true><<<grid, PP_W * 32, kPpSmem, (cudaStream_t) stream>>>(
                q, kv, p0, T, n_head, G, max_ctx, split_len, sc, q_full, ldq, out, part_o, part_ml);
        else
            attn_pp_kernel<false><<<grid, PP_W * 32, kPpSmem, (cudaStream_t) stream>>>(
                q, kv, p0, T, n_head, G, max_ctx, split_len, sc, q_full, ldq, out, part_o, part_ml);
    } else if (kv.mode == 0) {
        attn_tc_kernel<0><<<grid, AW * 32, kAttnSmem, (cudaStream_t) stream>>>(
            q, kv, p0, T, n_head, G, max_ctx, split_len, sc, q_full, ldq, out, part_o, part_ml);
    } else if (kv.mode == 2) {
        attn_tc_kernel<2><<<grid, AW * 32, kAttnSmem, (cudaStream_t) stream>>>(
            q, kv, p0, T, n_head, G, max_ctx, split_len, sc, q_full, ldq, out, part_o, part_ml);
    } else {
        attn_tc_kernel<1><<<grid, AW * 32, kAttnSmem, (cudaStream_t) stream>>>(
            q, kv, p0, T, n_head, G, max_ctx, split_len, sc, q_full, ldq, out, part_o, part_ml);
    }
    check("q35_attn");
    if (n_split > 1) {
        attn_merge_kernel<<<dim3(rows, n_kv), D, 0, (cudaStream_t) stream>>>(part_o, part_ml, rows, n_split, G, n_head,
                                                                             q_full, ldq, out, p0, split_len,
                                                                             kv.mode != 1);
        check("q35_attn_merge");
    }
    if (kv.mode == 1) {
        unrotate_gate_kernel<<<dim3(T, n_head), 256, 0, (cudaStream_t) stream>>>(out, n_head, q_full, ldq);
        check("q35_attn_unrotate");
    }
}

void q35_kv_expand_f16(const Q35Kv& kv, int n_keys, int n_kv, int D, uint16_t* k16, uint16_t* v16, void* stream) {
    if (kv.mode != 2 || D != AD) throw std::invalid_argument("q35_kv_expand_f16: a K8V8 cache, head size 256");
    if (n_keys <= 0) return;
    kv_expand_kernel<<<dim3((n_keys + 15) / 16, n_kv), 256, 0, (cudaStream_t) stream>>>(kv, n_keys, (__half*) k16,
                                                                                      (__half*) v16);
    check("q35_kv_expand_f16");
}

__global__ void map_ids_kernel(int32_t* __restrict__ x, const int32_t* __restrict__ map, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = map[x[i]];
}
void q35_map_ids(int32_t* x, const int32_t* map, int n, void* stream) {
    if (n <= 0) return;
    map_ids_kernel<<<(n + 127) / 128, 128, 0, (cudaStream_t) stream>>>(x, map, n);
    check("q35_map_ids");
}

void q35_gemv_f32_cols(const float* W, const float* x, int ldx, float* y, int n_in, int n_out, int T, void* stream) {
    gemv_f32_cols_kernel<<<dim3(n_out, (T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(W, x, ldx, y, n_in, n_out, T);
    check("q35_gemv_f32_cols");
}

namespace {
// softmax probability of the row's argmax: 1 / sum_j exp(x_j - x_best) (one block of 1024)
__global__ void top1_prob_kernel(const float* x, int n, const int32_t* best, float* out) {
    __shared__ float red[32];
    const float xb = x[*best];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += __expf(x[i] - xb);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) *out = 1.0f / v;
    }
}
}  // namespace

void q35_top1_prob(const float* x, int n, const int32_t* best, float* out, void* stream) {
    top1_prob_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(x, n, best, out);
    check("q35_top1_prob");
}

void q35_argmax_rows(const float* x, int64_t ld, int n, int T, int32_t* out, void* stream) {
    // (two steps over a long row - STRATA_DENSE_ARGMAX_MB=0: one block of 1024 a row)
    static const bool mb = !(std::getenv("STRATA_DENSE_ARGMAX_MB") && std::atoi(std::getenv("STRATA_DENSE_ARGMAX_MB")) == 0);
    static float* pv = nullptr;
    static int* pi = nullptr;
    const int parts = (n + kAmChunk - 1) / kAmChunk;
    if (mb && parts >= 2 && parts <= kAmMaxParts && T <= kAmMaxRows) {
        if (pv == nullptr && (cudaMalloc(&pv, (size_t) kAmMaxRows * kAmMaxParts * sizeof(float)) != cudaSuccess ||
                              cudaMalloc(&pi, (size_t) kAmMaxRows * kAmMaxParts * sizeof(int)) != cudaSuccess)) {
            cudaGetLastError();
            pv = nullptr;
        }
        if (pv != nullptr) {
            argmax_part_kernel<<<dim3(parts, T), 256, 0, (cudaStream_t) stream>>>(x, ld, n, pv, pi, parts);
            argmax_join_kernel<<<T, 32, 0, (cudaStream_t) stream>>>(pv, pi, parts, out);
            check("q35_argmax_rows");
            return;
        }
    }
    argmax_kernel<<<T, 1024, 0, (cudaStream_t) stream>>>(x, ld, n, out);
    check("q35_argmax_rows");
}

void q35_add(float* y, const float* x, int64_t n, void* stream) {
    add_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(y, x, n);
    check("q35_add");
}

void q35_swiglu_rows(const float* g, const float* u, int ldg, float* out, int n, int T, void* stream) {
    swiglu_rows_kernel<<<dim3(blocks(n, 256), T), 256, 0, (cudaStream_t) stream>>>(g, u, ldg, out, n);
    check("q35_swiglu_rows");
}

}  // namespace strata::kernels
