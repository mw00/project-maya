// src/k31/k31_mla.cu - the K3.1 MLA attention core and its output gate.  See k31_mla.hpp.
//
// One block a (token, head), 256 threads:
//   1. the query row (576) to shared memory in float64; the count of valid picks (a prefix of sel)
//   2. scores: warp w takes picks w, w+8, ...; its lanes split the 576 dims; float64 sums; times scale
//   3. a float64 softmax over the picks (block max and sum)
//   4. ctx: thread d sums p_j * latent[j][d] over the picks in float64 (rows read coalesced across threads), rounds once
#include "k31_mla.hpp"

#include <cuda_runtime.h>

namespace strata::k31 {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ double block_reduce(double v, bool is_max, double* scratch) {
    for (int o = 16; o > 0; o >>= 1) {
        const double u = __shfl_down_sync(0xffffffffu, v, o);
        v = is_max ? (u > v ? u : v) : v + u;
    }
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        double r = scratch[0];
        for (int i = 1; i < kWarps; ++i) r = is_max ? (scratch[i] > r ? scratch[i] : r) : r + scratch[i];
        scratch[0] = r;
    }
    __syncthreads();
    const double r = scratch[0];
    __syncthreads();
    return r;
}

__global__ void mla_attn_kernel(MlaAttnArgs a) {
    __shared__ double s_q[kMlaRow];
    __shared__ double s_s[kMlaMaxSel];
    __shared__ int s_key[kMlaMaxSel];
    __shared__ double s_red[kWarps];
    __shared__ int s_n;
    const int t = blockIdx.x / a.n_head, h = blockIdx.x % a.n_head;
    const float* q = a.q + ((size_t) t * a.n_head + h) * kMlaRow;
    const int32_t* sel = a.sel + (size_t) t * a.k_sel;
    if (threadIdx.x == 0) s_n = a.k_sel;
    for (int d = threadIdx.x; d < kMlaRow; d += kThreads) s_q[d] = (double) q[d];
    __syncthreads();
    for (int i = threadIdx.x; i < a.k_sel; i += kThreads) {
        const int key = sel[i];
        s_key[i] = key;
        if (key < 0) atomicMin(&s_n, i);   // the valid picks are a prefix: the first -1 ends it
    }
    __syncthreads();
    const int n = s_n;
    float* ctx = a.ctx + ((size_t) t * a.n_head + h) * kMlaLatent;
    if (n == 0) {
        for (int d = threadIdx.x; d < kMlaLatent; d += kThreads) ctx[d] = 0.0f;
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int j = warp; j < n; j += kWarps) {
        const float* row = a.kv + (size_t) s_key[j] * kMlaRow;
        double acc = 0.0;
        for (int d = lane; d < kMlaRow; d += 32) acc += s_q[d] * (double) row[d];
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (lane == 0) s_s[j] = acc * a.scale;
    }
    __syncthreads();
    double mx = -1e300;
    for (int j = threadIdx.x; j < n; j += kThreads) mx = s_s[j] > mx ? s_s[j] : mx;
    mx = block_reduce(mx, true, s_red);
    double sum = 0.0;
    for (int j = threadIdx.x; j < n; j += kThreads) {
        const double e = exp(s_s[j] - mx);
        s_s[j] = e;
        sum += e;
    }
    sum = block_reduce(sum, false, s_red);
    for (int j = threadIdx.x; j < n; j += kThreads) s_s[j] /= sum;
    __syncthreads();
    for (int d = threadIdx.x; d < kMlaLatent; d += kThreads) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j) acc += s_s[j] * (double) a.kv[(size_t) s_key[j] * kMlaRow + d];
        ctx[d] = (float) acc;
    }
}

__global__ void mla_gate_kernel(const float* kqv, const float* g, float* out, int64_t n) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float sg = (float) (1.0 / (1.0 + exp(-(double) g[i])));
        out[i] = kqv[i] * sg;
    }
}

}  // namespace

bool mla_attn_launch(const MlaAttnArgs& a, void* stream) {
    if (a.n_tokens < 0 || a.n_head < 1 || a.n_kv < 1 || a.k_sel < 1 || a.k_sel > kMlaMaxSel) return false;
    if (a.n_tokens == 0) return true;
    if (!a.q || !a.kv || !a.sel || !a.ctx) return false;
    const long long blocks = (long long) a.n_tokens * a.n_head;
    if (blocks > 0x7fffffffLL) return false;
    mla_attn_kernel<<<(unsigned) blocks, kThreads, 0, (cudaStream_t) stream>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool mla_gate_launch(const float* kqv, const float* gate_pre, float* out, int64_t n, void* stream) {
    if (n < 0 || (n > 0 && (!kqv || !gate_pre || !out))) return false;
    if (n == 0) return true;
    const int64_t want = (n + 255) / 256;
    mla_gate_kernel<<<(int) (want < 4096 ? want : 4096), 256, 0, (cudaStream_t) stream>>>(kqv, gate_pre, out, n);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::k31
