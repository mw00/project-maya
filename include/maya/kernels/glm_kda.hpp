// include/maya/kernels/glm_kda.hpp - the glm5-next KDA (Kimi Delta Attention) layer pieces.
//
// Phase 2 of docs/GLM5-FLASH.md.  glm5-next's 34 linear-attention layers are KDA: the qwen4exp GDN
// (`gdn.hpp`) with the per-head scalar decay replaced by a PER-CHANNEL decay on the KEY axis, produced
// by a low-rank gate and squashed through the architecture's lower-bound form.  The reference is
// `ref/glm.py::kda_layer` / `conv1d_causal`, transcribed from `ref/upstream-llama.cpp/glm5-next.cpp`
// L687-763 and ggml's `ggml_gated_delta_net` kernel (cpu-ops.cpp L10898-11050).  Per token, per head,
// with S the (dk, dv) state (dk = dv = head_dim here):
//
//     S     = diag(exp(g1)) @ S      the gate scales the KEY axis (kernel: "S[i][:] *= exp(g[i])")
//     delta = beta * (v - S^T k)     the delta correction - THIS is what makes it delta net, not
//     S    += k (delta)^T            linear attention; dropping it is the parity test's negative
//     o     = (1/sqrt(dv)) * S^T q   the 1/sqrt(S_v) scale is INSIDE the ggml kernel, so it is here
//
// THE GATE, end to end (the caller does the two GEMVs; this family does the shapes and the squash):
//
//     g  = f_b^T (f_a^T x) + dt_b          LOW-RANK: rank = head_dim, NOT a direct d_inner projection
//     g1 = lower_bound * sigmoid(-g * A)   with A = ssm_a holding -exp(A_log) (negative), so
//                                          g1 in (lower_bound, 0) and exp(g1) in (e^lb, 1)
//
// The qwen4exp form is `softplus(g) * A` - no lower bound, unbounded above - and swapping the two
// forms is one of the parity test's negatives.  The OUTPUT gate g2 = g_b^T (g_a^T x) is the same
// low-rank trick on the output side, applied as rms_norm(o) * sigmoid(g2) per head.
//
// LAYOUT CONTRACTS (everything f32, ggml order - features fastest, so Phase 3 wiring needs no
// transposes anywhere between the pack and these kernels):
//   x_proj / conv out : (d_inner, T)                     element (c, t) at c + d_inner*t
//   conv_w            : (d_conv, d_inner)                element (tap, c) at tap + d_conv*c
//   conv_state        : (d_conv-1, d_inner)              history rows, OLDEST first; element (j, c) at
//                                                          j + (d_conv-1)*c; updated in place
//   q, k, v, g1, out  : (head_dim, n_head, T)            element (e, h) at e + head_dim*h (+ T stride)
//   beta              : (n_head, T)                      element (h, t) at h + n_head*t
//   state             : (n_head, head_dim, head_dim)     S[h] row-major, row = KEY axis; updated in place
//
// DECODE-FIRST: the recurrence kernel carries the state across T tokens inside one block per head,
// which is the decode shape and the oracle's teacher-forced shape at once.  Chunked parallel-scan
// prefill is a later sibling; the state contract above is what both must produce.
#pragma once

#include <cstdint>
#include <cstddef>

namespace maya::kernels {

struct GlmKdaShapes {
    int64_t n_head = 0;
    int64_t head_dim = 0;    // dk = dv (the state is square in this architecture)
    int64_t d_conv = 0;
    int64_t T = 0;

    int64_t d_inner() const { return n_head * head_dim; }
};

/// silu(conv1d(x_proj)) for ONE stream (call three times for q/k/v), with the (d_inner, d_conv-1)
/// history updated in place.  Output position t's window is x_proj[t-(d_conv-1) .. t]; taps older
/// than the sequence come from the state, oldest first.
void glm_kda_conv(const float* x_proj, const float* conv_w, float* conv_state, int d_inner, int d_conv,
                  int T, float* out, void* stream);

/// The decay gate's squash: g1[e, h, t] = lower_bound * sigmoid(-g[e + head_dim*h, t] * A[h]).
/// `g` is (d_inner, T) - the f_b GEMV's output plus dt_b, in ggml head order.
void glm_kda_gate(const float* g, const float* ssm_a, float lower_bound, int head_dim, int n_head, int T,
                  float* g1, void* stream);

/// The recurrent delta rule over T tokens, per head, state and conv carried in place.  `out`
/// includes the 1/sqrt(head_dim) scale.  state is (n_head, head_dim, head_dim), zeroed by the
/// caller for a fresh sequence.
void glm_kda_recurrence(const float* q, const float* k, const float* v, const float* g1, const float* beta,
                        float* state, int n_head, int head_dim, int T, float* out, void* stream);

/// rms_norm(o, ssm_norm) * sigmoid(g2), per head: `o` and `g2` are (head_dim, n_head, T),
/// `norm_w` is (head_dim,), `eps` is the model's rms epsilon.
void glm_kda_out_gate(const float* o, const float* norm_w, const float* g2, int head_dim, int n_head, int T,
                      float eps, float* gated, void* stream);

}  // namespace maya::kernels
