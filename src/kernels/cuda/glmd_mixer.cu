// src/kernels/cuda/glmd_mixer.cu - the glm-dsa decode mixer's own kernels (B4.6, B14.3, B14.4); see
// include/strata/kernels/glmd_mixer.hpp.  Also the decode dispatch policy's GPU side (glm-dsa wave 5,
// include/strata/kernels/glmd_decode_policy.hpp): decode_score / decode_topk / decode_mla.
#include "strata/kernels/glmd_mixer.hpp"
#include "strata/kernels/glmd_decode_policy.hpp"

#include "glmd_arch.cuh"   // STRATA_GRID_CONSTANT (Pascal pass)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <type_traits>
#include <cmath>
#include <cstdio>

namespace glmd {
namespace {

std::atomic<int> g_errors{0};

void refuse(const char* what) {
    g_errors.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "glmd_mixer %s: shape refused\n", what);
}

bool launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    g_errors.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "glmd_mixer %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

constexpr int kScoreThreads = 256;               // 8 warps, a warp per cell
constexpr int kScoreWarps = kScoreThreads / 32;
constexpr int kScoreMaxBlocks = 320;             // 4 x 80 SMs: fixed, so the grid never depends on the position

__device__ __forceinline__ float key_at(const float* k, size_t i, std::false_type) { return k[i]; }
__device__ __forceinline__ float key_at(const float* k, size_t i, std::true_type) {
    return __half2float(__ushort_as_half(((const unsigned short*) k)[i]));
}

// Maya's dsa_score_kernel arithmetic per cell, a grid-stride loop over the cells, n read on the device; KF16: the
// keys are FP16 rows (B10.8), each element widened to f32 as it is read (the products and sums stay f32)
template <bool KF16>
__global__ void __launch_bounds__(kScoreThreads) idx_score_kernel(const STRATA_GRID_CONSTANT IdxScore a) {
    extern __shared__ float s_iq[];   // idx_heads * idx_dim, then idx_heads weights
    const int n = a.p_dev ? min(a.n_cap, *a.p_dev + 1) : min(a.n_cap, a.p + 1);
    if (blockIdx.x == 0 && threadIdx.x == 0 && a.n_sel_out) *a.n_sel_out = min(n, a.sel_cap);
    if (n <= a.skip_le) return;   // the selection is the identity: nobody reads the scores
    const int key_dim = a.idx_dim, heads = a.idx_heads;
    float* s_iw = s_iq + heads * key_dim;
    for (int i = threadIdx.x; i < heads * key_dim; i += blockDim.x) s_iq[i] = a.iq[i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) s_iw[i] = a.w[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    for (int c = blockIdx.x * kScoreWarps + (threadIdx.x >> 5); c < n; c += gridDim.x * kScoreWarps) {
        const size_t row = (size_t) key_dim * c;
        float acc = 0.0f;
        for (int h = 0; h < heads; ++h) {
            float dot = 0.0f;
            for (int e = lane; e < key_dim; e += 32)
                dot += s_iq[h * key_dim + e] * key_at(a.ik_cache, row + e, std::integral_constant<bool, KF16>());
            dot = warp_sum(dot);
            acc += fmaxf(dot, 0.0f) * s_iw[h];
        }
        if (lane == 0) a.score[c] = acc;
    }
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t u = __float_as_uint(x[i]);
    if ((u & 0x7fffffffu) > 0x7f800000u) {   // NaN: keep it a quiet NaN
        y[i] = (uint16_t) ((u >> 16) | 0x40u);
        return;
    }
    y[i] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// the device cell count of a WMMA-scored token (decode_score with n_sel_out set)
__global__ void sel_count_kernel(const int* p_dev, int p, int n_cap, int sel_cap, int* out) {
    const int n = min(n_cap, (p_dev ? *p_dev : p) + 1);
    *out = min(n, sel_cap);
}

__global__ void iota_kernel(int* cells, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cells[i] = i;
}

}  // namespace

int mixer_launch_errors() { return g_errors.load(std::memory_order_relaxed); }

bool idx_score(const IdxScore& a, void* stream) {
    if (a.idx_heads <= 0 || a.idx_dim <= 0 || a.idx_dim > 512 || a.idx_heads * a.idx_dim > 12288 || a.n_cap <= 0 ||
        !a.iq || !a.ik_cache || !a.w || !a.score || a.sel_cap <= 0) {
        refuse("idx_score");
        return false;
    }
    const size_t smem = ((size_t) a.idx_heads * a.idx_dim + a.idx_heads) * sizeof(float);
    const int blocks = (int) std::min<long long>(kScoreMaxBlocks, ((long long) a.n_cap + kScoreWarps - 1) / kScoreWarps);
    if (a.key_f16 && a.idx_dim % 8) {   // 16-byte FP16 rows (the state layout's rule)
        refuse("idx_score (FP16 keys)");
        return false;
    }
    const auto kern = a.key_f16 ? idx_score_kernel<true> : idx_score_kernel<false>;
    if (smem > 48 * 1024 &&
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) != cudaSuccess) {
        cudaGetLastError();
        refuse("idx_score (shared memory)");
        return false;
    }
    kern<<<blocks, kScoreThreads, smem, (cudaStream_t) stream>>>(a);
    return launch_check("idx_score");
}

bool f32_to_bf16(const float* x, uint16_t* y, size_t n, void* stream) {
    if (n == 0) return true;
    if (!x || !y) {
        refuse("f32_to_bf16");
        return false;
    }
    f32_to_bf16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    return launch_check("f32_to_bf16");
}

bool iota(int* cells, int n, void* stream) {
    if (n <= 0) return true;
    if (!cells) {
        refuse("iota");
        return false;
    }
    iota_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(cells, n);
    return launch_check("iota");
}

// ---------------------------------------------------------------- the decode dispatch policy (GPU side)
// (HIP builds have no WMMA scorer and no fused MLA - both are nvcuda::wmma kernels: the mixer's policy,
// decode_policy_built, keeps them off there, and a policy that asks for them anyway gets the portable kernel,
// idx_score / mla2)
bool decode_score(const DecodePolicy& pol, const IdxScore& a, int64_t p, void* stream) {
#if defined(STRATA_USE_HIP)
    (void) pol;
    (void) p;
    return idx_score(a, stream);
#else
    if (!use_idx_wmma(pol, p + 1, a.skip_le, a.idx_heads, a.idx_dim)) return idx_score(a, stream);
    if (!idx_score_wmma(wmma_args(a), stream)) return false;
    if (a.n_sel_out == nullptr) return true;
    sel_count_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(a.p_dev, a.p, a.n_cap, a.sel_cap, a.n_sel_out);
    return launch_check("decode_score count");
#endif
}

bool decode_topk(const DecodePolicy& pol, const float* score, int ld, int n_kv, const int* pos, int64_t p, int top_k,
                 int* sel, void* work, void* stream) {
    if (use_topk_mb(pol, p + 1, top_k)) {
        if (work == nullptr) {
            refuse("decode_topk (no topk_mb work)");
            return false;
        }
        return topk_mb(score, ld, n_kv, 1, pos, top_k, sel, work, stream);
    }
    if (use_topk_hist(pol, p + 1, top_k)) return topk_hist(score, ld, n_kv, 1, pos, top_k, sel, stream);   // B14.15
    return topk(score, ld, n_kv, 1, pos, top_k, sel, stream);
}

bool decode_mla(const DecodePolicy& pol, const Mla2& a, void* stream) {
#if defined(STRATA_USE_HIP)
    (void) pol;
#else
    if (use_fused_mla(pol, a.n_head, a.qk_nope, a.qk_rope, a.kv_lora, a.v_head, a.w_type))
        return mla_fused(fused_args(a, pol.fused_split), stream);
#endif
    if (pol.mla_dec) return mla2(a, stream);
    Mla2 b = a;   // STRATA_GLMD_MLA_DEC=0: mla2's own attention kernels (L1-4)
    b.legacy_attn = 1;
    return mla2(b, stream);
}

}  // namespace glmd
