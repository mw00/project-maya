// src/kernels/cuda/router_sigmoid.cu - the glm5-next MoE router.  See router_sigmoid.hpp for the math.
//
// The same one-block-per-token shape as router_top10.cu, and simpler: sigmoid needs no max
// subtraction (it is bounded in (0, 1)), so there is no tree reduction at all - the kernel is
//
//     parallel : p[e] = sigmoid(l[e]);  sel[e] = p[e] + bias[e]
//     parallel : rank(e) over sel, the SAME stable-descending rank formula router_top10 uses
//     one thread: gather w in rank order, sum it ascending in double, clamp at 2**-14, scale
//
// The rank formula is reproduced exactly because it IS the tie rule:
//
//     rank(e) = #{ f : sel[f] > sel[e] }  +  #{ f < e : sel[f] == sel[e] }
//
// which is "stable argsort of -sel, ties by ascending index" - what ref/glm.py's top_k_desc
// produces with numpy's stable sort.  The bias is applied to sel ONLY; the gathered weights are
// the unbiased p (router_sigmoid.hpp's trap).
#include "maya/kernels/router_sigmoid.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace maya::kernels {
namespace {

constexpr int RS_MAX_THREADS = 512;

__global__ void router_sigmoid_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                      int n_tokens, int n_expert, int k, float w_scale, int norm_w,
                                      int* __restrict__ ids, float* __restrict__ weights) {
    extern __shared__ float s_mem[];      // p (n_expert) then sel (n_expert)
    float* s_p = s_mem;
    float* s_sel = s_mem + n_expert;

    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const float* l = logits + (size_t) t * n_expert;
    // the bias is a PER-LAYER parameter (ffn_exp_probs_b, n_expert values) broadcast across tokens
    const float* b = bias;

    for (int e = tid; e < n_expert; e += nt) {
        const float x = l[e];
        const float p = 1.0f / (1.0f + expf(-x));
        s_p[e] = p;
        s_sel[e] = b ? p + b[e] : p;
    }
    __syncthreads();

    // the selection, by rank over sel
    for (int e = tid; e < n_expert; e += nt) {
        const float v = s_sel[e];
        int rank = 0;
        for (int f = 0; f < n_expert; ++f) {
            const float u = s_sel[f];
            rank += u > v ? 1 : 0;
            rank += (f < e && u == v) ? 1 : 0;
        }
        if (rank < k) {
            ids[(size_t) t * k + rank] = e;
        }
    }
    __syncthreads();

    // the weights, from the UNBIASED p, in rank order, summed ascending on one thread
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += (double) s_p[ids[(size_t) t * k + i]];
        const float inv = (float) (norm_w ? 1.0 / fmax(sum, 6.103515625e-5) : 1.0);
        for (int i = 0; i < k; ++i) weights[(size_t) t * k + i] = (float) ((double) s_p[ids[(size_t) t * k + i]] * inv * w_scale);
    }
}

}  // namespace

void router_sigmoid(const float* logits, const float* bias, int n_tokens, int n_expert, int k,
                    float w_scale, bool norm_w, int* ids, float* weights, void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_sigmoid: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > RS_MAX_THREADS) {
        std::fprintf(stderr, "router_sigmoid: n_expert %d is past the kernel's %d\n", n_expert, RS_MAX_THREADS);
        std::exit(1);
    }
    int threads = (n_expert + 31) & ~31;   // at least one full warp, like router_top10
    const size_t smem = (size_t) 2 * n_expert * sizeof(float);
    router_sigmoid_kernel<<<n_tokens, threads, smem, (cudaStream_t) stream>>>(
        logits, bias, n_tokens, n_expert, k, w_scale, norm_w ? 1 : 0, ids, weights);
}

}  // namespace maya::kernels
