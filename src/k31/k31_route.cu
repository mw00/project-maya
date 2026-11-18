// src/k31/k31_route.cu - the K3.1 router on the device.  See k31_route.hpp for the math and the contract.
//
// One block a token:
//     parallel   : p[e] = float(1 / (1 + exp(-(double) l[e])));  sel[e] = p[e] + bias[e]
//     parallel   : key[e] = an order-preserving uint32 of sel[e] (-0 folded to +0, NaN below -inf)
//                  rank(e) = #{f : key[f] > key[e]} + #{f < e : key[f] == key[e]}   (stable descending, lower id first)
//                  rank < k writes ids[rank]
//     one thread : the k weights, summed in double in rank order, one rounding each
// The rank formula is the tie rule; it is router_sigmoid.cu's, over up to 1024 experts.
#include "k31_route.hpp"

#include <cuda_runtime.h>

namespace strata::k31 {
namespace {

// the float order as an unsigned order: equal floats (+0 and -0 included) give equal keys, every NaN gives 0,
// below -inf's 0x007fffff
__device__ __forceinline__ uint32_t order_key(float x) {
    if (x != x) return 0u;
    const uint32_t b = x == 0.0f ? 0u : __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(kK31MaxExperts)
route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert, int k, float scale,
             int norm, double eps, int32_t* __restrict__ ids, float* __restrict__ weights, float* __restrict__ probs) {
    __shared__ float s_p[kK31MaxExperts];
    __shared__ uint32_t s_key[kK31MaxExperts];
    __shared__ int32_t s_ids[kK31MaxPicks];
    const int t = blockIdx.x;
    const int e = threadIdx.x;
    const float* l = logits + (size_t) t * n_expert;
    if (e < n_expert) {
        const float p = (float) (1.0 / (1.0 + exp(-(double) l[e])));
        s_p[e] = p;
        s_key[e] = order_key(bias ? p + bias[e] : p);   // a total order, so the ranks are a permutation (every
                                                         // slot written, every id in range)
        if (probs) probs[(size_t) t * n_expert + e] = p;
    }
    __syncthreads();
    if (e < n_expert) {
        const uint32_t v = s_key[e];
        int rank = 0;
        for (int f = 0; f < n_expert; ++f) {
            const uint32_t u = s_key[f];
            rank += (u > v) | (f < e && u == v);
        }
        if (rank < k) s_ids[rank] = e;
    }
    __syncthreads();
    if (e == 0) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += (double) s_p[s_ids[i]];
        const double den = norm ? sum + eps : 1.0;
        for (int i = 0; i < k; ++i) {
            ids[(size_t) t * k + i] = s_ids[i];
            weights[(size_t) t * k + i] = (float) ((double) s_p[s_ids[i]] / den * (double) scale);
        }
    }
}

}  // namespace

bool route_launch(const RouteArgs& a, void* stream) {
    if (a.n_tokens < 0 || a.n_expert < 1 || a.n_expert > kK31MaxExperts) return false;
    if (a.k < 1 || a.k > a.n_expert || a.k > kK31MaxPicks) return false;
    if (a.n_tokens == 0) return true;
    if (!a.logits || !a.ids || !a.weights) return false;
    if (cudaPeekAtLastError() != cudaSuccess) return false;   // someone else's error: refuse rather than blame it
    const int threads = (a.n_expert + 31) & ~31;
    route_kernel<<<a.n_tokens, threads, 0, (cudaStream_t) stream>>>(a.logits, a.bias, a.n_expert, a.k, a.scale,
                                                                     a.norm ? 1 : 0, a.eps, a.ids, a.weights, a.probs);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::k31
