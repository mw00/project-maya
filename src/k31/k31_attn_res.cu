// src/k31/k31_attn_res.cu - the Block Attention Residuals mix on the device.  See k31_attn_res.hpp.
//
// One block a token (256 threads): each thread accumulates, per candidate, sum(w*v) and sum(v^2) over its columns
// in float64; warp and block reductions; thread 0 forms the scores and the float64 softmax; then every column is
// mixed from the raw candidates in float64 and rounded once.  The output is written only after every read of the
// token's columns (the second loop reads column j before writing it), so `out` may alias the partial.
#include "k31_attn_res.hpp"

#include <cuda_runtime.h>

namespace strata::k31 {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__global__ void attn_res_kernel(AttnResArgs a) {
    __shared__ double s_dot[kWarps][kK31MaxCand];
    __shared__ double s_sq[kWarps][kK31MaxCand];
    __shared__ double s_p[kK31MaxCand];
    const int t = blockIdx.x, n = a.n_embd, nc = a.n_cand;
    const size_t row = (size_t) t * n;
    double dot[kK31MaxCand], sq[kK31MaxCand];
#pragma unroll
    for (int i = 0; i < kK31MaxCand; ++i) { dot[i] = 0.0; sq[i] = 0.0; }
    for (int j = threadIdx.x; j < n; j += kThreads) {
        const double w = (double) a.score_w[j];
#pragma unroll
        for (int i = 0; i < kK31MaxCand; ++i) {
            if (i < nc) {
                const double v = (double) a.cand[i][row + j];
                dot[i] += w * v;
                sq[i] += v * v;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kK31MaxCand; ++i) {
        if (i < nc) {
            for (int o = 16; o > 0; o >>= 1) {
                dot[i] += __shfl_down_sync(0xffffffffu, dot[i], o);
                sq[i] += __shfl_down_sync(0xffffffffu, sq[i], o);
            }
            if ((threadIdx.x & 31) == 0) { s_dot[threadIdx.x >> 5][i] = dot[i]; s_sq[threadIdx.x >> 5][i] = sq[i]; }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double s[kK31MaxCand], mx = -1e300;
        for (int i = 0; i < nc; ++i) {
            double d = 0.0, q = 0.0;
            for (int wv = 0; wv < kWarps; ++wv) { d += s_dot[wv][i]; q += s_sq[wv][i]; }
            s[i] = d / sqrt(q / (double) n + (double) a.eps);
            mx = s[i] > mx ? s[i] : mx;
        }
        double tot = 0.0;
        for (int i = 0; i < nc; ++i) { s[i] = exp(s[i] - mx); tot += s[i]; }
        for (int i = 0; i < nc; ++i) {
            s_p[i] = s[i] / tot;
            if (a.probs) a.probs[(size_t) t * nc + i] = (float) s_p[i];
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += kThreads) {
        double acc = 0.0;
        for (int i = 0; i < nc; ++i) acc += s_p[i] * (double) a.cand[i][row + j];
        a.out[row + j] = (float) acc;
    }
}

}  // namespace

bool attn_res_launch(const AttnResArgs& a, void* stream) {
    if (a.n_cand < 1 || a.n_cand > kK31MaxCand || a.n_embd < 1 || a.n_tokens < 0) return false;
    if (a.n_tokens == 0) return true;
    if (!a.score_w || !a.out) return false;
    for (int i = 0; i < a.n_cand; ++i)
        if (!a.cand[i]) return false;
    attn_res_kernel<<<a.n_tokens, kThreads, 0, (cudaStream_t) stream>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::k31
