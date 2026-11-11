// src/k31/k31_kda.cu - the K3.1 KDA layer adapter.  See k31_kda.hpp.
//
// Seven launches in order: three convs (each with its history slide), the two l2 norms, the k31 prep kernel (the
// dt_bias add into `gated`'s buffer and the float64 beta sigmoid), the gate, the recurrence, the output gate.  No
// buffer is read and written by the same launch.
#include "k31_kda.hpp"

#include "strata/kernels/gdn.hpp"
#include "strata/kernels/glm_kda.hpp"

#include <cuda_runtime.h>

namespace strata::k31 {
namespace {

// gpre = f_pre + dt_bias (one float32 add, the oracle's); beta = float32(sigmoid(beta_pre)) in float64
__global__ void kda_prep_kernel(const float* __restrict__ f_pre, const float* __restrict__ dt_bias,
                                float* __restrict__ gpre, int64_t d_inner, int64_t n_gate,
                                const float* __restrict__ beta_pre, float* __restrict__ beta, int64_t n_beta) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n_gate; i += stride)
        gpre[i] = f_pre[i] + dt_bias[i % d_inner];
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n_beta; i += stride)
        beta[i] = (float) (1.0 / (1.0 + exp(-(double) beta_pre[i])));
}

}  // namespace

bool kda_layer_launch(const KdaLayerArgs& a, void* stream) {
    if (a.n_tokens < 0 || a.n_head < 1 || a.head_dim < 1 || a.head_dim > kKdaMaxHeadDim || a.d_conv < 2 ||
        a.d_conv > kKdaMaxConv)
        return false;
    if (!(a.lower_bound < 0.0f) || !(a.eps_l2 > 0.0f) || !(a.eps_rms > 0.0f)) return false;
    const int64_t d_inner = (int64_t) a.n_head * a.head_dim;
    const int64_t n_gate = d_inner * a.n_tokens, n_beta = (int64_t) a.n_head * a.n_tokens;
    if (n_gate > 0x7fffffffLL) return false;   // the GLM kernels index with int
    if (a.n_tokens == 0) return true;
    if (!a.xq || !a.xk || !a.xv || !a.f_pre || !a.beta_pre || !a.g2 || !a.dt_bias || !a.ssm_a || !a.norm_w ||
        !a.ssm_state || !a.q || !a.k || !a.v || !a.g1 || !a.beta || !a.o || !a.gated)
        return false;
    for (int s = 0; s < 3; ++s)
        if (!a.conv_w[s] || !a.conv_state[s]) return false;
    if (cudaPeekAtLastError() != cudaSuccess) return false;   // someone else's error: refuse rather than blame it

    namespace K = strata::kernels;
    const int d = (int) d_inner, T = a.n_tokens, H = a.n_head, dk = a.head_dim;
    const float* x[3] = {a.xq, a.xk, a.xv};
    float* out[3] = {a.q, a.k, a.v};
    for (int s = 0; s < 3; ++s) K::glm_kda_conv(x[s], a.conv_w[s], a.conv_state[s], d, a.d_conv, T, out[s], stream);
    // gdn_l2_norm synchronises (and exits on an error) when stream is null; the k31 contract allows a null stream
    K::gdn_l2_norm(a.q, (int64_t) T * H, dk, a.eps_l2, stream);
    K::gdn_l2_norm(a.k, (int64_t) T * H, dk, a.eps_l2, stream);
    const int64_t want = (n_gate + 255) / 256;
    kda_prep_kernel<<<(int) (want < 4096 ? want : 4096), 256, 0, (cudaStream_t) stream>>>(
        a.f_pre, a.dt_bias, a.gated, d_inner, n_gate, a.beta_pre, a.beta, n_beta);
    K::glm_kda_gate(a.gated, a.ssm_a, a.lower_bound, dk, H, T, a.g1, stream);
    K::glm_kda_recurrence(a.q, a.k, a.v, a.g1, a.beta, a.ssm_state, H, dk, T, a.o, stream);
    K::glm_kda_out_gate(a.o, a.norm_w, a.g2, dk, H, T, a.eps_rms, a.gated, stream);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::k31
