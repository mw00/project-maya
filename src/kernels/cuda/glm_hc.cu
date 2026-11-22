// src/kernels/cuda/glm_hc.cu - the glm5-next mHC read and write halves.  See glm_hc.hpp for the math.
//
// Three kernels:
//
//   inv_rms   ONE block, a standard parallel reduction over hc_dim values - it produces the single
//             scalar the GEMV consumes (the JOINT-stack norm; NOT per-stream, see glm_hc.hpp).
//   pre       one thread per mixes row (hc_mix_dim rows, each a strided walk over its contiguous
//             row of w_fn), then lane 0 turns the 24 mixes into the three gates, runs the Sinkhorn
//             SERIALLY on the 4x4 comb, and forms `mixed` - 16 floats x <= 20 iterations and one
//             n_embd pass, once per layer per token.  Serial-in-one-lane is the easiest thing to
//             keep bitwise-comparable with ref/glm.py; if profiling ever asks, the mixes GEMV
//             becomes gr.cu L344's warp-per-row and the rest stays.
//   post      one thread per (stream, embd) pair, a flat elementwise pass.
//
// All arithmetic is f32 (the contract in glm_hc.hpp); the parity harness holds the kernel to
// ref/glm.py's f64 reference at f32 tolerances, and the negative tests require the wrong readings
// (per-stream norm, dst-major comb, mean-not-sum, no eps floor) to differ materially.
#include "maya/kernels/glm_hc.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace maya::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAX_HC = 8;   // the serial lane's fixed scratch (col/denom) is sized for this

__device__ __forceinline__ float dsigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void glm_hc_inv_rms_kernel(const float* __restrict__ R, int hc_dim, float norm_eps,
                                      float* __restrict__ inv_rms) {
    __shared__ float acc[THREADS];
    const int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < hc_dim; i += THREADS) local += R[i] * R[i];
    acc[tid] = local;
    __syncthreads();
    for (int span = THREADS / 2; span > 0; span >>= 1) {
        if (tid < span) acc[tid] += acc[tid + span];
        __syncthreads();
    }
    if (tid == 0) inv_rms[0] = rsqrtf(acc[0] / (float) hc_dim + norm_eps);
}

__global__ void glm_hc_pre_kernel(const float* __restrict__ R, const float* __restrict__ w_fn,
                                  const float* __restrict__ w_scale, const float* __restrict__ w_base,
                                  int n_embd, int hc, int iters, float hc_eps,
                                  const float* __restrict__ inv_rms,
                                  float* __restrict__ mixed, float* __restrict__ pre,
                                  float* __restrict__ post, float* __restrict__ comb) {
    // One WARP per mix output, strided over k.  The first version had each of the 24 outputs on ONE
    // thread doing a 16k serial dot over an uncoalesced weight row: 1.78 ms per call, 55% of decode
    // time (measured, nsys).  A warp reads its row coalesced (one 128B line per 32 elements), so the
    // same 1.5 MB streams at bandwidth instead of latency.
    const int hc_dim = n_embd * hc;
    const int hc_mix_dim = (2 + hc) * hc;
    __shared__ float s_mix[2 * MAX_HC * MAX_HC];   // hc_mix_dim for hc <= MAX_HC
    __shared__ float s_pre[MAX_HC];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int m = warp; m < hc_mix_dim; m += THREADS / 32) {
        const float inv = inv_rms[0];
        const float* wrow = w_fn + (size_t) m * hc_dim;
        float acc = 0.0f;
        if ((hc_dim & 3) == 0) {   // float4 stream (n_embd*hc is a multiple of 4 in every real config)
            const float4* R4 = (const float4*) R;
            const float4* w4 = (const float4*) wrow;
            const int n4 = hc_dim / 4;
            // four independent chains: one load in flight per warp was latency-bound (the loop
            // measured 461 us/call, 41 ms/token); 4-way ILP keeps the memory pipeline full
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            int k = lane;
            for (; k + 96 < n4; k += 128) {
                const float4 r0 = R4[k], v0 = w4[k];
                const float4 r1 = R4[k + 32], v1 = w4[k + 32];
                const float4 r2 = R4[k + 64], v2 = w4[k + 64];
                const float4 r3 = R4[k + 96], v3 = w4[k + 96];
                a0 += (r0.x * inv) * v0.x + (r0.y * inv) * v0.y + (r0.z * inv) * v0.z + (r0.w * inv) * v0.w;
                a1 += (r1.x * inv) * v1.x + (r1.y * inv) * v1.y + (r1.z * inv) * v1.z + (r1.w * inv) * v1.w;
                a2 += (r2.x * inv) * v2.x + (r2.y * inv) * v2.y + (r2.z * inv) * v2.z + (r2.w * inv) * v2.w;
                a3 += (r3.x * inv) * v3.x + (r3.y * inv) * v3.y + (r3.z * inv) * v3.z + (r3.w * inv) * v3.w;
            }
            for (; k < n4; k += 32) {
                const float4 r = R4[k];
                const float4 v = w4[k];
                a0 += (r.x * inv) * v.x + (r.y * inv) * v.y + (r.z * inv) * v.z + (r.w * inv) * v.w;
            }
            acc = (a0 + a1) + (a2 + a3);
        } else {
            for (int k = lane; k < hc_dim; k += 32) acc += R[k] * inv * wrow[k];
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) s_mix[m] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // the three gates.  pre carries the +hc_eps FLOOR (ggml_scale_bias(pre, 1, eps)); post does not.
        for (int s = 0; s < hc; ++s) {
            pre[s] = dsigmoid(s_mix[s] * w_scale[0] + w_base[s]) + hc_eps;
            s_pre[s] = pre[s];
        }
        for (int s = 0; s < hc; ++s) post[s] = 2.0f * dsigmoid(s_mix[hc + s] * w_scale[1] + w_base[hc + s]);
    for (int d = 0; d < hc; ++d)
        for (int s = 0; s < hc; ++s)
            comb[d * hc + s] = s_mix[2 * hc + d + hc * s] * w_scale[2] + w_base[2 * hc + d + hc * s];

    // ---- the Sinkhorn, per ref/glm.py::hc_sinkhorn (the unfused graph path is the authority):
    // softmax over dst, +eps ONTO THE VALUES, then dst-normalize, then (iters-1) x [src, dst].
    // Every divisor carries its own +eps guard; the +eps onto the values after the softmax is a
    // separate step and skipping it is exactly the kind of plausible-wrong-number this file exists
    // to be tested against.
    float col_max[MAX_HC];
    for (int s = 0; s < hc; ++s) {
        float mx = -INFINITY;
        for (int d = 0; d < hc; ++d) mx = fmaxf(mx, comb[d * hc + s]);
        col_max[s] = mx;
    }
    float denom[MAX_HC];
    for (int s = 0; s < hc; ++s) {
        float e = 0.0f;
        for (int d = 0; d < hc; ++d) e += __expf(comb[d * hc + s] - col_max[s]);
        denom[s] = e;
    }
    for (int d = 0; d < hc; ++d)
        for (int s = 0; s < hc; ++s) comb[d * hc + s] = __expf(comb[d * hc + s] - col_max[s]) / denom[s];
    for (int d = 0; d < hc; ++d)
        for (int s = 0; s < hc; ++s) comb[d * hc + s] += hc_eps;

    auto norm_dst = [&]() {
        for (int d = 0; d < hc; ++d) {
            float sum = hc_eps;
            for (int s = 0; s < hc; ++s) sum += comb[d * hc + s];
            for (int s = 0; s < hc; ++s) comb[d * hc + s] /= sum;
        }
    };
    auto norm_src = [&]() {
        for (int s = 0; s < hc; ++s) {
            float sum = hc_eps;
            for (int d = 0; d < hc; ++d) sum += comb[d * hc + s];
            for (int d = 0; d < hc; ++d) comb[d * hc + s] /= sum;
        }
    };
    norm_dst();
    for (int it = 1; it < iters; ++it) {
        norm_src();
        norm_dst();
    }
    }   // thread 0: the gates and the sinkhorn
    __syncthreads();
    // mixed[e] = sum_s pre[s] * R_s - the whole block (was a 4096-iteration single-thread loop)
    for (int e = threadIdx.x; e < n_embd; e += THREADS) {
        float acc = 0.0f;
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * R[s * n_embd + e];
        mixed[e] = acc;
    }
}

__global__ void glm_hc_post_kernel(const float* __restrict__ block_out, const float* __restrict__ residual,
                                   const float* __restrict__ post, const float* __restrict__ comb,
                                   int n_embd, int hc, float* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_embd * hc;
    if (idx >= total) return;
    const int e = idx % n_embd;
    const int d = idx / n_embd;
    float acc = block_out[e] * post[d];
    for (int s = 0; s < hc; ++s) acc += comb[d * hc + s] * residual[s * n_embd + e];
    out[d * n_embd + e] = acc;
}

}  // namespace

size_t glm_hc_workspace_bytes(const GlmHcShapes& s) {
    (void) s;
    return sizeof(float);
}

void glm_hc_workspace_init(const GlmHcShapes& s, void* base, GlmHcWorkspace& ws) {
    ws.inv_rms = (float*) base;
    ws.bytes = glm_hc_workspace_bytes(s);
}

void glm_hc_pre(const float* R, const float* w_fn, const float* w_scale, const float* w_base,
                float norm_eps, float hc_eps, const GlmHcShapes& s, GlmHcWorkspace& ws,
                float* mixed, float* pre, float* post, float* comb, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.sinkhorn_iters <= 0) return;
    if (s.hc > MAX_HC || s.hc_mix_dim() > 2 * MAX_HC * MAX_HC) {
        std::fprintf(stderr, "glm_hc_pre: hc %lld exceeds the kernel's fixed scratch (max %d)\n",
                     (long long) s.hc, MAX_HC);
        std::exit(1);
    }
    if (ws.inv_rms == nullptr || ws.bytes < glm_hc_workspace_bytes(s)) {
        std::fprintf(stderr, "glm_hc_pre: GlmHcWorkspace is not initialised (see glm_hc_workspace_init)\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    glm_hc_inv_rms_kernel<<<1, THREADS, 0, st>>>(R, (int) s.hc_dim(), norm_eps, ws.inv_rms);
    glm_hc_pre_kernel<<<1, THREADS, 0, st>>>(R, w_fn, w_scale, w_base, (int) s.n_embd, (int) s.hc,
                                             (int) s.sinkhorn_iters, hc_eps, ws.inv_rms, mixed, pre, post, comb);
}

void glm_hc_post(const float* block_out, const float* residual, const float* post, const float* comb,
                 const GlmHcShapes& s, float* out, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0) return;
    const int total = (int) (s.n_embd * s.hc);
    cudaStream_t st = (cudaStream_t) stream;
    glm_hc_post_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0, st>>>(block_out, residual, post, comb,
                                                                            (int) s.n_embd, (int) s.hc, out);
}

}  // namespace maya::kernels
