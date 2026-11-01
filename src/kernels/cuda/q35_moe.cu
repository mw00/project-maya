// src/kernels/cuda/q35_moe.cu - see include/strata/kernels/q35_moe.hpp.
#include "strata/kernels/q35_moe.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// a block (256 threads) a row: the softmax over n_exp (<= 1024), then k rounds of an arg-max
__global__ void route_kernel(const float* __restrict__ logits, int ld, int n_exp, int k, float scale,
                             int32_t* __restrict__ ids, float* __restrict__ w) {
    __shared__ float p[1024];
    __shared__ float rv[8];
    __shared__ int ri[8];
    __shared__ float s_max, s_sum;
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
    const float* lr = logits + (size_t) blockIdx.x * ld;
    float m = -FLT_MAX;
    for (int i = t; i < n_exp; i += blockDim.x) { p[i] = lr[i]; m = fmaxf(m, p[i]); }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) rv[wp] = m;
    __syncthreads();
    if (t == 0) { float x = rv[0]; for (int i = 1; i < 8; ++i) x = fmaxf(x, rv[i]); s_max = x; }
    __syncthreads();
    float s = 0.0f;
    for (int i = t; i < n_exp; i += blockDim.x) { p[i] = expf(p[i] - s_max); s += p[i]; }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    __syncthreads();
    if (lane == 0) rv[wp] = s;
    __syncthreads();
    if (t == 0) { float x = 0; for (int i = 0; i < 8; ++i) x += rv[i]; s_sum = x; }
    __syncthreads();
    __shared__ float sel_w[32];
    __shared__ int sel_i[32];
    for (int r = 0; r < k; ++r) {
        float bv = -1.0f;
        int bi = 0x7fffffff;
        for (int i = t; i < n_exp; i += blockDim.x)
            if (p[i] > bv || (p[i] == bv && i < bi)) { bv = p[i]; bi = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { rv[wp] = bv; ri[wp] = bi; }
        __syncthreads();
        if (t == 0) {
            float b = rv[0];
            int j = ri[0];
            for (int q = 1; q < 8; ++q)
                if (rv[q] > b || (rv[q] == b && ri[q] < j)) { b = rv[q]; j = ri[q]; }
            sel_w[r] = b / s_sum;
            sel_i[r] = j;
            p[j] = -2.0f;   // (taken)
        }
        __syncthreads();
    }
    if (t == 0) {
        float tot = 0.0f;
        for (int r = 0; r < k; ++r) tot += sel_w[r];
        for (int r = 0; r < k; ++r) {
            ids[(size_t) blockIdx.x * k + r] = sel_i[r];
            w[(size_t) blockIdx.x * k + r] = sel_w[r] / tot * scale;
        }
    }
}

// one block: count the pairs a expert, an exclusive scan, then each pair its position (pairs of an expert in pair
// order: a stable counting sort, so the grouping is deterministic)
__global__ void group_kernel(const int32_t* __restrict__ ids, int P, int n_exp, int32_t* __restrict__ bounds,
                             int32_t* __restrict__ tok_of_pos, int32_t* __restrict__ pos_of_pair, int k,
                             int32_t* __restrict__ max_rows, int32_t* __restrict__ cnt) {
    const int t = threadIdx.x;
    for (int e = t; e <= n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = t; i < P; i += blockDim.x) atomicAdd(&cnt[ids[i]], 1);
    __syncthreads();
    if (t == 0) {
        int acc = 0, mx = 0;
        for (int e = 0; e < n_exp; ++e) {
            bounds[e] = acc;
            mx = max(mx, cnt[e]);
            acc += cnt[e];
        }
        bounds[n_exp] = acc;
        max_rows[0] = mx;
    }
    __syncthreads();
    // stable: a thread an expert walks the pairs in order and gives its own their positions (the ids are read by
    // every thread at once: broadcast loads)
    for (int e = t; e < n_exp; e += blockDim.x) {
        int pos = bounds[e];
        if (cnt[e] == 0) continue;
        for (int i = 0; i < P; ++i)
            if (__ldg(ids + i) == e) {
                tok_of_pos[pos] = i / k;
                pos_of_pair[i] = pos;
                ++pos;
            }
    }
}

// a block a token row: out[t][e] (+)= sum_j w y[pos] + sig(g) sh
__global__ void combine_kernel(const float* __restrict__ y, const int32_t* __restrict__ pos_of_pair,
                               const float* __restrict__ w, int k, int E, const float* __restrict__ sh_logit, int sh_ld,
                               const float* __restrict__ sh, float* __restrict__ out, int add) {
    const int t = blockIdx.y;
    const float g = sh_logit ? 1.0f / (1.0f + expf(-sh_logit[(size_t) t * sh_ld])) : (sh ? 1.0f : 0.0f);
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < E; e += gridDim.x * blockDim.x) {
        float a = 0.0f;
        for (int j = 0; j < k; ++j) a += w[(size_t) t * k + j] * y[(size_t) pos_of_pair[(size_t) t * k + j] * E + e];
        if (sh) a += g * sh[(size_t) t * E + e];
        float* o = out + (size_t) t * E + e;
        *o = add ? *o + a : a;
    }
}

// the same routing a warp a row (n_exp <= 32 * kPerLane): the logits in registers, softmax and k rounds of a warp
// arg-max with shuffles only (the block form's barriers cost more than its work at one token); same picks, same ties
constexpr int kPerLane = 8;
__global__ void route_warp_kernel(const float* __restrict__ logits, int ld, int n_exp, int k, float scale,
                                  int32_t* __restrict__ ids, float* __restrict__ w) {
    const int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const float* lr = logits + (size_t) row * ld;
    float p[kPerLane];
    float m = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        p[i] = e < n_exp ? lr[e] : -FLT_MAX;
        m = fmaxf(m, p[i]);
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        p[i] = lane + 32 * i < n_exp ? expf(p[i] - m) : -1.0f;
        if (p[i] > 0.0f || lane + 32 * i < n_exp) sum += fmaxf(p[i], 0.0f);
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    float sel = 0.0f, tot = 0.0f;
    int mine = -1;   // lane r keeps pick r
    for (int r = 0; r < k; ++r) {
        float bv = -1.0f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int e = lane + 32 * i;
            if (p[i] > bv || (p[i] == bv && e < bi)) { bv = p[i]; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const float wv = bv / sum;
        tot += wv;
        if (lane == r) { sel = wv; mine = bi; }
#pragma unroll
        for (int i = 0; i < kPerLane; ++i)
            if (lane + 32 * i == bi) p[i] = -2.0f;   // (taken)
    }
    if (lane < k) {
        ids[(size_t) row * k + lane] = mine;
        w[(size_t) row * k + lane] = sel / tot * scale;
    }
}

}  // namespace

void q35_moe_route(const float* logits, int T, int n_exp, int k, float scale, int32_t* ids, float* w, void* stream) {
    if (n_exp > 1024 || k > 32) throw std::invalid_argument("q35_moe_route: up to 1024 experts, 32 picks");
    route_kernel<<<T, 256, 0, (cudaStream_t) stream>>>(logits, n_exp, n_exp, k, scale, ids, w);
    check("q35_moe_route");
}

void q35_moe_group(const int32_t* ids, int T, int k, int n_exp, int32_t* bounds, int32_t* tok_of_pos,
                   int32_t* pos_of_pair, int32_t* max_rows, int32_t* scratch, void* stream) {
    group_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(ids, T * k, n_exp, bounds, tok_of_pos, pos_of_pair, k, max_rows,
                                                         scratch);
    check("q35_moe_group");
}

void q35_moe_combine(const float* y, const int32_t* pos_of_pair, const float* w, int T, int k, int E,
                     const float* sh_logit, const float* sh, float* out, bool add, void* stream) {
    combine_kernel<<<dim3((E + 255) / 256, T), 256, 0, (cudaStream_t) stream>>>(y, pos_of_pair, w, k, E, sh_logit, 1, sh, out, add);
    check("q35_moe_combine");
}

void q35_moe_route_ld(const float* logits, int ld, int T, int n_exp, int k, float scale, int32_t* ids, float* w,
                      void* stream) {
    if (n_exp > 1024 || k > 32) throw std::invalid_argument("q35_moe_route: up to 1024 experts, 32 picks");
    if (n_exp <= 32 * kPerLane) {
        route_warp_kernel<<<T, 32, 0, (cudaStream_t) stream>>>(logits, ld, n_exp, k, scale, ids, w);
        check("q35_moe_route_warp");
        return;
    }
    route_kernel<<<T, 256, 0, (cudaStream_t) stream>>>(logits, ld, n_exp, k, scale, ids, w);
    check("q35_moe_route");
}

void q35_moe_combine_ld(const float* y, const int32_t* pos_of_pair, const float* w, int T, int k, int E,
                        const float* sh_logit, int sh_ld, const float* sh, float* out, bool add, void* stream) {
    combine_kernel<<<dim3((E + 255) / 256, T), 256, 0, (cudaStream_t) stream>>>(y, pos_of_pair, w, k, E, sh_logit, sh_ld, sh, out, add);
    check("q35_moe_combine");
}

}  // namespace strata::kernels
