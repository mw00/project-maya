// src/kernels/cuda/glmd_fast.cu - glm-dsa decode attention kernels (B4.3, B4.4, B4.5); see
// include/strata/kernels/glmd_fast.hpp.
//
// The small helpers (warp/block reductions, the q8_1 store, the BF16 row dot, Maya's INT8 latent record) repeat
// glm_fast.cu's expressions, so a latent written here reads back through glmf's lat_any unchanged, and the q8_1
// blocks are the ones quantize.cu writes.  The rope is glmd_rope.cuh: this file must not be built with
// --use_fast_math (the header refuses).
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_dec.hpp"
#include "strata/kernels/glmd_volta.hpp"
#include "strata/kernels/glm_fast.hpp"

#include "glmd_arch.cuh"   // STRATA_GRID_CONSTANT (Pascal pass)
#include "glmd_had.cuh"
#include "glmd_rope.cuh"
#include "glmd_topk.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>

namespace glmd {
namespace {

static_assert(glmd::lat8_rec_bytes(512) == strata::kernels::glmf::lat8_rec_bytes(512) &&
                  glmd::lat8_rec_bytes(256) == strata::kernels::glmf::lat8_rec_bytes(256),
              "the INT8 latent record must stay Maya's");

std::atomic<int> g_launch_errors{0};

void refuse(const char* what) {
    g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "glmd_fast %s: shape refused\n", what);
}

bool launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "glmd_fast %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// ---------------------------------------------------------------- helpers (glm_fast.cu's arithmetic)
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum; every thread gets the total.  sh: >= 32 floats.  blockDim % 32 == 0.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    return t;
}
// one q8_1 block from 32 values held one per lane (glm_fast.cu q8_1_store_warp)
__device__ __forceinline__ void q8_1_store_warp(float v, Q81* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) {
        dst->d = __half_as_ushort(__float2half(d));
        dst->s = __half_as_ushort(__float2half(fmaxf(-65504.0f, fminf(65504.0f, sum))));
    }
}
// the INT8 latent record size on the device (glmd::lat8_rec_bytes is host constexpr)
__device__ __forceinline__ int lat8_rec(int kv_lora) { return kv_lora + kv_lora / 16; }
__device__ __forceinline__ float h2f(uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ uint16_t f2h(float v) { return __half_as_ushort(__float2half(v)); }
// value c of an INT8 latent record
__device__ __forceinline__ float lat8_f(const uint8_t* rec, int kv_lora, int c) {
    return (float) ((const int8_t*) rec)[c] * __half2float(((const __half*) (rec + kv_lora))[c >> 5]);
}
// one latent row into its INT8 record: lane l of the warp holding values [32 w, 32 w + 32) stores value c
__device__ __forceinline__ void lat8_store_warp(float y, uint8_t* rec, int kv_lora, int c, int lane) {
    float amax = fabsf(y);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const __half d16 = __float2half(amax / 127.0f);
    const float d = __half2float(d16);
    const int q = d > 0.0f ? max(-127, min(127, __float2int_rn(y / d))) : 0;
    ((int8_t*) rec)[c] = (int8_t) q;
    if (lane == 0) ((__half*) (rec + kv_lora))[c >> 5] = d16;
}
__device__ __forceinline__ float row_dot_bf16(const uint16_t* row, const float* x, int n_in, int lane) {
    float s = 0.0f;
    const uint4* r8 = (const uint4*) row;
    const float4* x4 = (const float4*) x;
    for (int k = lane; k < n_in / 8; k += 32) {
        const uint4 w = r8[k];
        const float4 a = x4[2 * k], b = x4[2 * k + 1];
        s += __uint_as_float(w.x << 16) * a.x + __uint_as_float(w.x & 0xffff0000u) * a.y +
             __uint_as_float(w.y << 16) * a.z + __uint_as_float(w.y & 0xffff0000u) * a.w +
             __uint_as_float(w.z << 16) * b.x + __uint_as_float(w.z & 0xffff0000u) * b.y +
             __uint_as_float(w.w << 16) * b.z + __uint_as_float(w.w & 0xffff0000u) * b.w;
    }
    return warp_sum(s);
}

// ---------------------------------------------------------------- B4.3 prep
// block 0: q_a RMS -> qr / qr_q;  block 1: c_kv RMS -> latent record, k_pe rope -> FP16;  block 2: the index key
__global__ void __launch_bounds__(512) dsa_prep_kernel(const STRATA_GRID_CONSTANT DsaPrep a) {
    __shared__ float sred[32];
    __shared__ float s_y[512];
    const int tid = threadIdx.x, lane = tid & 31;
    const int p = a.p_dev ? *a.p_dev : a.p;
    if (blockIdx.x == 0) {
        float vals[4];
        float ss = 0.0f;
        int n = 0;
        for (int e = tid; e < a.q_lora; e += 512, ++n) {
            vals[n] = a.qr_raw[e];
            ss += vals[n] * vals[n];
        }
        ss = block_sum(ss, sred);
        const float inv = rsqrtf(ss / (float) a.q_lora + a.eps);
        n = 0;
        for (int e = tid; e < a.q_lora; e += 512, ++n) {   // q_lora % 32 == 0: whole warps per 32-run
            const float y = vals[n] * inv * a.q_a_norm[e];
            if (a.qr) a.qr[e] = y;
            if (a.qr_q) q8_1_store_warp(y, (Q81*) a.qr_q + e / 32, lane);
        }
        return;
    }
    if (blockIdx.x == 1) {
        const float v = tid < a.kv_lora ? a.kv_raw[tid] : 0.0f;
        const float ss = block_sum(v * v, sred);
        const float inv = rsqrtf(ss / (float) a.kv_lora + a.eps);
        // L1-6 KV streaming: the host copy's row p always, the slot row only while its block is resident
        size_t rs = (size_t) p;
        bool to_slot = true;
        if (a.lat_host != nullptr) {
            const int sl = a.kv_table[p / a.kv_page];
            to_slot = sl >= 0;
            rs = to_slot ? (size_t) sl * a.kv_page + p % a.kv_page : 0;
        }
        if (tid < a.kv_lora) {   // kv_lora % 32 == 0: whole warps
            const float y = v * inv * a.kv_norm[tid];
            if (a.lat_q8) {
                if (to_slot) lat8_store_warp(y, (uint8_t*) a.lat + (size_t) lat8_rec(a.kv_lora) * rs, a.kv_lora, tid, lane);
                if (a.lat_host != nullptr)
                    lat8_store_warp(y, (uint8_t*) a.lat_host + (size_t) lat8_rec(a.kv_lora) * p, a.kv_lora, tid, lane);
            } else {
                if (to_slot) ((uint16_t*) a.lat)[(size_t) a.kv_lora * rs + tid] = f2h(y);
                if (a.lat_host != nullptr) ((uint16_t*) a.lat_host)[(size_t) a.kv_lora * p + tid] = f2h(y);
            }
        }
        if (tid < a.qk_rope / 2) {   // the k_pe tail, NORM pairs, not normed
            float x0 = a.kv_raw[a.kv_lora + 2 * tid], x1 = a.kv_raw[a.kv_lora + 2 * tid + 1];
            float c, s;
            rope_cs(p, a.inv_freq[tid], c, s);
            rope_pair(x0, x1, c, s);
            if (to_slot) {
                uint16_t* kp = a.k_pe + (size_t) a.qk_rope * rs;
                kp[2 * tid] = f2h(x0);
                kp[2 * tid + 1] = f2h(x1);
            }
            if (a.kpe_host != nullptr) {
                uint16_t* kh = a.kpe_host + (size_t) a.qk_rope * p;
                kh[2 * tid] = f2h(x0);
                kh[2 * tid + 1] = f2h(x1);
            }
        }
        return;
    }
    // block 2: LayerNorm(ik_raw) (two-pass mean / variance, biased) then rope dims [0, idx_rope)
    const int K = a.idx_dim;
    const float v = tid < K ? a.ik_raw[tid] : 0.0f;
    const float mu = block_sum(v, sred) / (float) K;
    const float d = tid < K ? v - mu : 0.0f;
    const float var = block_sum(d * d, sred) / (float) K;
    const float inv = rsqrtf(var + a.idx_ln_eps);
    if (tid < K) s_y[tid] = d * inv * a.ik_ln_w[tid] + a.ik_ln_b[tid];
    __syncthreads();
    if (tid < a.idx_rope / 2) {
        float c, s;
        rope_cs(p, a.idx_inv_freq[tid], c, s);
        rope_pair(s_y[2 * tid], s_y[2 * tid + 1], c, s);
    }
    __syncthreads();
    if (a.ik_had) had_block(s_y, K);   // B14.11: the rotated key (uniform branch: every thread synchronises)
    if (tid < K) {
        if (a.ik_f16)   // B10.8: FP16 index keys
            ((uint16_t*) a.ik_cache)[(size_t) K * p + tid] = f2h(s_y[tid]);
        else
            a.ik_cache[(size_t) K * p + tid] = s_y[tid];
    }
}

// one NORM pair per thread: the MLA q_pe pairs of every head, then the indexer query pairs
__global__ void __launch_bounds__(256) dsa_rope_q_kernel(const STRATA_GRID_CONSTANT DsaRopeQ a) {
    const int p = a.p_dev ? *a.p_dev : a.p;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nq = a.n_head * (a.n_rot / 2);
    float* x;
    float f;
    if (i < nq) {
        const int h = i / (a.n_rot / 2), j = i - h * (a.n_rot / 2);
        x = a.q + (size_t) a.q_stride * h + a.rot_off + 2 * j;
        f = a.inv_freq[j];
    } else {
        const int k = i - nq;
        if (a.iq == nullptr || k >= a.idx_heads * (a.idx_rope / 2)) return;
        const int h = k / (a.idx_rope / 2), j = k - h * (a.idx_rope / 2);
        x = a.iq + (size_t) a.idx_dim * h + 2 * j;
        f = a.idx_inv_freq[j];
    }
    float c, s;
    rope_cs(p, f, c, s);
    rope_pair(x[0], x[1], c, s);
}

// ---------------------------------------------------------------- B4.4 mla2
// out[h * R + r] = W[h * R + r][0, K) . in[h * in_stride, + K) (W BF16, rows of K); 32 rows per block (4 per warp);
// q8 != nullptr also writes the q8_1 of the 32 rows
__global__ void __launch_bounds__(256) headwise_gemv_kernel(const uint16_t* __restrict__ W, const float* __restrict__ in,
                                                           int R, int K, int in_stride, float* __restrict__ out,
                                                           Q81* __restrict__ q8) {
    __shared__ float s_o[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = blockIdx.x * 32;   // R % 32 == 0: a block never straddles heads
    const int h = row0 / R;
    const float* x = in + (size_t) h * in_stride;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int row = row0 + warp * 4 + j;
        const float s = row_dot_bf16(W + (size_t) row * K, x, K, lane);
        if (lane == 0) s_o[warp * 4 + j] = s;
    }
    __syncthreads();
    if (warp == 0) {
        const float v = s_o[lane];
        if (out) out[row0 + lane] = v;
        if (q8) q8_1_store_warp(v, q8 + row0 / 32, lane);
    }
}

// The split attention (flash-decoding, glm_fast.cu mla_split_kernel's structure) with an R-wide rope tail: a
// block stages kMla2Chunk selected cells (FP16 latents and FP16 k_pe) in shared memory once and scores them
// against kMla2HeadsPerBlock heads; per (chunk, head) it writes the exp-weighted context, the chunk max and the
// chunk sum, which mla2_combine_kernel merges.  R = 0 is the NoPE form (no k_pe staged, no q_pe term).
template <int R>
__global__ void __launch_bounds__(256) mla2_split_kernel(const float* __restrict__ q_abs, const float* __restrict__ q,
                                                         int q_stride, int q_pe_off, const void* __restrict__ lat,
                                                         int lat_q8, const uint16_t* __restrict__ k_pe,
                                                         const int* __restrict__ cells, int n_sel,
                                                         const int* __restrict__ n_sel_dev, int n_head,
                                                         int kv_lora, float scale, float* __restrict__ part) {
    constexpr int CH = kMla2Chunk, HPB = kMla2HeadsPerBlock;
    extern __shared__ float smem[];
    uint16_t* sL = (uint16_t*) smem;                    // CH x kv_lora, FP16
    uint16_t* sP = sL + CH * kv_lora;                   // CH x R, FP16
    float* sS = (float*) (sP + CH * R);                 // HPB x CH scores -> weights (4-B aligned: CH even)
    __shared__ int s_cell[CH];
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const int h0 = blockIdx.y * HPB, nh = min(HPB, n_head - h0);
    const int s0 = b * CH;
    const int n_act = n_sel_dev ? min(n_sel, *n_sel_dev) : n_sel;   // (B14.4: the device count, when given)
    const int ns = min(CH, n_act - s0);
    if (ns <= 0) {   // a chunk past the device count: an empty partial (the combine skips m = -inf)
        float* pe = part + ((size_t) b * n_head + h0) * (kv_lora + 2);
        for (int hh = tid; hh < nh; hh += blockDim.x) {
            pe[(size_t) hh * (kv_lora + 2) + kv_lora] = -INFINITY;
            pe[(size_t) hh * (kv_lora + 2) + kv_lora + 1] = 0.0f;
        }
        return;
    }
    if (tid < CH) s_cell[tid] = tid < ns ? cells[s0 + tid] : -1;
    __syncthreads();
    const int rec = lat8_rec(kv_lora);
    for (int i = tid; i < CH * kv_lora; i += blockDim.x) {
        const int s = i / kv_lora, c = i - s * kv_lora;
        const int cell = s_cell[s];
        sL[i] = cell < 0 ? (uint16_t) 0
                : lat_q8 ? f2h(lat8_f((const uint8_t*) lat + (size_t) rec * cell, kv_lora, c))
                         : ((const uint16_t*) lat)[(size_t) kv_lora * cell + c];
    }
    if constexpr (R > 0) {
        for (int i = tid; i < CH * R; i += blockDim.x) {
            const int s = i / R, j = i - s * R;
            const int cell = s_cell[s];
            sP[i] = cell < 0 ? (uint16_t) 0 : k_pe[(size_t) R * cell + j];
        }
    }
    __syncthreads();
    // scores: a warp per head, the head's q_abs (kv_lora / 32 per lane) and q_pe (R / 32 per lane) in registers
    for (int hh = warp; hh < nh; hh += nw) {
        float qa[16];
        const float* qh = q_abs + (size_t) kv_lora * (h0 + hh);
#pragma unroll
        for (int j = 0; j < 16; ++j) qa[j] = (lane + 32 * j) < kv_lora ? qh[lane + 32 * j] : 0.0f;
        float qp[R > 0 ? R / 32 : 1];
        if constexpr (R > 0) {
            const float* qph = q + (size_t) q_stride * (h0 + hh) + q_pe_off;
#pragma unroll
            for (int j = 0; j < R / 32; ++j) qp[j] = qph[lane + 32 * j];
        }
        for (int s = 0; s < CH; ++s) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + 32 * j < kv_lora) acc += qa[j] * h2f(sL[s * kv_lora + lane + 32 * j]);
#pragma unroll
            for (int j = 0; j < R / 32; ++j) acc += qp[j] * h2f(sP[s * R + lane + 32 * j]);
            acc = warp_sum(acc);
            if (lane == 0) sS[hh * CH + s] = s_cell[s] >= 0 ? acc * scale : -INFINITY;
        }
    }
    __syncthreads();
    // per head: the chunk's max and sum; the scores become exp-weights
    float* pm = part + ((size_t) b * n_head + h0) * (kv_lora + 2);
    for (int hh = warp; hh < nh; hh += nw) {
        const float v = lane < CH ? sS[hh * CH + lane] : -INFINITY;
        const float m = warp_max(v);
        const float e = (v == -INFINITY) ? 0.0f : expf(v - m);
        const float l = warp_sum(e);
        if (lane < CH) sS[hh * CH + lane] = e;
        if (lane == 0) {
            pm[(size_t) hh * (kv_lora + 2) + kv_lora] = m;
            pm[(size_t) hh * (kv_lora + 2) + kv_lora + 1] = l;
        }
    }
    __syncthreads();
    // exp-weighted context per head over this chunk (c_kv only: the value side has no rope tail)
    for (int i = tid; i < nh * kv_lora; i += blockDim.x) {
        const int hh = i / kv_lora, c = i - hh * kv_lora;
        float acc = 0.0f;
#pragma unroll 8
        for (int s = 0; s < CH; ++s) acc += sS[hh * CH + s] * h2f(sL[s * kv_lora + c]);
        pm[(size_t) hh * (kv_lora + 2) + c] = acc;
    }
}

__global__ void mla2_combine_kernel(const float* __restrict__ part, int n_chunks, int n_head, int kv_lora,
                                    float* __restrict__ ctx) {
    const int h = blockIdx.x;
    float M = -INFINITY;
    for (int b = 0; b < n_chunks; ++b) M = fmaxf(M, part[((size_t) b * n_head + h) * (kv_lora + 2) + kv_lora]);
    float L = 0.0f;
    for (int b = 0; b < n_chunks; ++b) {
        const float* pb = part + ((size_t) b * n_head + h) * (kv_lora + 2);
        if (pb[kv_lora] != -INFINITY) L += expf(pb[kv_lora] - M) * pb[kv_lora + 1];
    }
    const float invL = L > 0.0f ? 1.0f / L : 0.0f;
    for (int c = threadIdx.x; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (int b = 0; b < n_chunks; ++b) {
            const float* pb = part + ((size_t) b * n_head + h) * (kv_lora + 2);
            if (pb[kv_lora] != -INFINITY) acc += expf(pb[kv_lora] - M) * pb[c];
        }
        ctx[(size_t) kv_lora * h + c] = acc * invL;
    }
}

}  // namespace

int launch_errors() { return g_launch_errors.load(std::memory_order_relaxed); }

bool dsa_prep(const DsaPrep& a, void* stream) {
    const bool full = a.ik_raw != nullptr;
    if (a.q_lora <= 0 || a.q_lora > 2048 || a.q_lora % 32 || a.kv_lora <= 0 || a.kv_lora > 512 || a.kv_lora % 32 ||
        a.qk_rope < 0 || a.qk_rope > 128 || a.qk_rope % 2 || !a.qr_raw || !a.q_a_norm || !a.kv_raw || !a.kv_norm ||
        !a.lat || (a.qk_rope > 0 && (!a.k_pe || !a.inv_freq)) ||
        (full && (a.idx_dim <= 0 || a.idx_dim > 512 || a.idx_rope < 0 || a.idx_rope > a.idx_dim || a.idx_rope % 2 ||
                  !a.ik_ln_w || !a.ik_ln_b || !a.ik_cache || (a.idx_rope > 0 && !a.idx_inv_freq) ||
                  (a.ik_had && !had_len_ok(a.idx_dim))))) {
        refuse("dsa_prep");
        return false;
    }
    dsa_prep_kernel<<<full ? 3 : 2, 512, 0, (cudaStream_t) stream>>>(a);
    return launch_check("dsa_prep");
}

bool dsa_rope_q(const DsaRopeQ& a, void* stream) {
    if (!a.q || !a.inv_freq || a.n_rot % 2 || a.n_rot < 0 || a.rot_off + a.n_rot > a.q_stride ||
        (a.iq && (!a.idx_inv_freq || a.idx_rope % 2 || a.idx_rope < 0 || a.idx_rope > a.idx_dim ||
                  (a.iq_had && (!had_len_ok(a.idx_dim) || a.idx_heads <= 0))))) {
        refuse("dsa_rope_q");
        return false;
    }
    const int n = a.n_head * (a.n_rot / 2) + (a.iq ? a.idx_heads * (a.idx_rope / 2) : 0);
    if (n == 0 && !(a.iq && a.iq_had)) return true;
    if (n > 0) dsa_rope_q_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(a);
    if (a.iq && a.iq_had)   // B14.11: each indexer query head rotated after its rope (same stream: ordered)
        had_rows_kernel<<<dim3(a.idx_heads, 1), 128, (size_t) a.idx_dim * sizeof(float), (cudaStream_t) stream>>>(
            a.iq, a.idx_dim, 0);
    return launch_check("dsa_rope_q");
}

bool mla2(const Mla2& a, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    if ((a.qk_rope != 0 && a.qk_rope != 64) || a.kv_lora <= 0 || a.kv_lora > 512 || a.kv_lora % 32 ||
        a.v_head <= 0 || a.v_head % 32 || a.qk_nope <= 0 || a.qk_nope % 8 || a.q_stride % 4 ||
        (a.w_type != kWTypeBF16 && a.w_type != kWTypeQ8_0) || (a.w_type == kWTypeQ8_0 && a.qk_nope % 32) ||
        a.q_stride < a.qk_nope + a.qk_rope || a.n_head <= 0 || a.n_sel < 0 || !a.q || !a.wk_b || !a.wv_b ||
        !a.lat || (a.qk_rope > 0 && !a.k_pe) || (a.n_sel > 0 && !a.cells) || !a.work ||
        a.work_bytes < mla2_work_bytes(a.n_head, a.kv_lora, a.n_sel) || (!a.out && !a.out_q8_1)) {
        refuse("mla2");
        return false;
    }
    float* q_abs = (float*) a.work;
    float* ctx = q_abs + (size_t) a.n_head * a.kv_lora;
    float* part = ctx + (size_t) a.n_head * a.kv_lora;
    const int n_chunks = (a.n_sel + kMla2Chunk - 1) / kMla2Chunk;
    if (a.w_type == kWTypeQ8_0) {   // B10.7: native Q8_0 W_kb
        if (!headwise(a.wk_b, a.w_type, a.q, a.n_head, a.kv_lora, a.qk_nope, a.q_stride, q_abs, nullptr, stream))
            return false;
    } else {
        headwise_gemv_kernel<<<a.n_head * a.kv_lora / 32, 256, 0, s>>>(a.wk_b, a.q, a.kv_lora, a.qk_nope, a.q_stride,
                                                                       q_abs, nullptr);
    }
    // L1-4: the attention through mla_dec (glmd_dec.cu) where it serves the shape and the device takes it (a HIP
    // build: the archs it was measured faster on, glmd_dec.cu's device policy - mla2's own kernels below elsewhere);
    // its splits never exceed the ceil(n_sel / 32) chunks this scratch holds partials for
    const bool dec = !a.legacy_attn && mla_dec_ok(a.n_head, a.kv_lora, a.qk_rope) &&
                     mla_dec(q_abs, a.q, a.q_stride, a.qk_nope, a.lat, a.lat_q8, a.k_pe, a.cells, a.n_sel, a.n_sel_dev,
                             a.n_head, a.kv_lora, a.qk_rope, a.scale, part, ctx, stream);
    if (!dec) {
        if (n_chunks > 0) {
            const dim3 grid(n_chunks, (a.n_head + kMla2HeadsPerBlock - 1) / kMla2HeadsPerBlock);
            const size_t smem = mla2_smem_bytes(a.kv_lora, a.qk_rope);
            if (a.qk_rope == 64)
                mla2_split_kernel<64><<<grid, 256, smem, s>>>(q_abs, a.q, a.q_stride, a.qk_nope, a.lat, a.lat_q8,
                                                              a.k_pe, a.cells, a.n_sel, a.n_sel_dev, a.n_head,
                                                              a.kv_lora, a.scale, part);
            else
                mla2_split_kernel<0><<<grid, 256, smem, s>>>(q_abs, a.q, a.q_stride, a.qk_nope, a.lat, a.lat_q8,
                                                             a.k_pe, a.cells, a.n_sel, a.n_sel_dev, a.n_head,
                                                             a.kv_lora, a.scale, part);
        }
        mla2_combine_kernel<<<a.n_head, 256, 0, s>>>(part, n_chunks, a.n_head, a.kv_lora, ctx);
    }
    if (a.w_type == kWTypeQ8_0) {   // B10.7: native Q8_0 W_vb
        if (!launch_check("mla2")) return false;
        return headwise(a.wv_b, a.w_type, ctx, a.n_head, a.v_head, a.kv_lora, a.kv_lora, a.out, a.out_q8_1, stream);
    }
    headwise_gemv_kernel<<<a.n_head * a.v_head / 32, 256, 0, s>>>(a.wv_b, ctx, a.v_head, a.kv_lora, a.kv_lora, a.out,
                                                                  (Q81*) a.out_q8_1);
    return launch_check("mla2");
}

bool topk(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel, void* stream) {
    if (T <= 0) return true;
    if (top_k <= 0 || n_kv < 0 || ld < n_kv || !sel || (n_kv > top_k && !score)) {
        refuse("topk");
        return false;
    }
    topk_detail::topk_kernel<<<T, topk_detail::kThreads, 0, (cudaStream_t) stream>>>(score, ld, n_kv, pos, top_k, sel);
    return launch_check("topk");
}

bool topk_hist(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel, void* stream) {
    if (T <= 0) return true;
    if (top_k <= 0 || n_kv < 0 || ld < n_kv || !sel || (n_kv > top_k && !score)) {
        refuse("topk_hist");
        return false;
    }
    topk_detail::topk_hist_kernel<<<T, topk_detail::kThreads, 0, (cudaStream_t) stream>>>(score, ld, n_kv, pos, top_k,
                                                                                           sel);
    return launch_check("topk_hist");
}

}  // namespace glmd
