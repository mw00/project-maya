// Gemma 4 (llama.cpp src/models/gemma4.cpp) kernels for strata-dense: the attention's preparation and core for head
// sizes 256 (sliding-window layers) and 512 (full layers), GELU, the per-layer embeddings, the logit soft cap.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// A layer's K / V cache: FP16 rows [n_kv][max_ctx][D].
struct G4Kv {
    uint16_t* k = nullptr;
    uint16_t* v = nullptr;
    int max_ctx = 0;   // cells a KV head
    int ring = 0;      // > 0: a ring of `ring` (= max_ctx) cells, position p in cell p % ring (a sliding-window layer
                       // that keeps the window and a chunk; the caller never reads a position the ring has dropped)
};

/// T tokens at p0 + t: q heads (rows of n_head * D) rms * q_norm then rope over all D dims (theta = pos *
/// base^(-2i/D) / ff[i], ff null: 1; rope_mode 1 NeoX, 2 the plain (2i, 2i+1) pairs, 0 none) -> q_out; with kv.k:
/// k rows rms * k_norm + rope, v rows (rms, no weight, when v_norm), both into the cache at their positions.
/// D 128, 256 or 512.
void g4_attn_prep(const float* q, const float* k, const float* v, int T, int n_head, int n_kv, int D,
                  const float* q_norm, const float* k_norm, float eps, float rope_base, const float* ff, int p0,
                  float* q_out, const G4Kv& kv, void* stream, int rope_mode = 1, bool v_norm = true);

/// out rows (n_head * D) = softmax(scale q k^T) V over the cells [max(0, pos - window + 1), pos] (window 0: all);
/// tensor cores, split over the keys when the rows are few.  n_head / n_kv <= 16.
int64_t g4_attn_scratch_floats(int T, int n_head, int D, int max_ctx);
void g4_attn(const float* q, const G4Kv& kv, int p0, int T, int n_head, int n_kv, int D, int window, float scale,
             float* out, float* scratch, void* stream);

/// out = gelu_tanh(g) * u, n values
void g4_gelu_mul(const float* g, const float* u, float* out, int64_t n, void* stream);
/// the per-layer inputs of T tokens: out[t][l][j] = (rms_256(proj[t][l] / sqrt(E)) * w + tok[t][l] * 16) / sqrt 2
/// (rows of n_layer * 256; tok: the token's per-layer embedding row)
void g4_ple(const float* proj, const float* tok, const float* w, float eps, int n_embd, int n_layer, int T, float* out,
            void* stream);
/// g[t][j] = gelu_tanh(g[t][j]) * ple[t][l][j], rows of 256
void g4_gelu_ple(float* g, const float* ple, int l, int n_layer, int T, void* stream);
/// x = (x + y) * s, n values
void g4_add_scale(float* x, const float* y, float s, int64_t n, void* stream);
/// x *= sigmoid(g), n values
void g4_sigmoid_mul(float* x, const float* g, int64_t n, void* stream);
/// x *= s, n values
void g4_scale(float* x, float s, int64_t n, void* stream);
/// x = cap * tanh(x / cap), rows of n (stride ld)
void g4_softcap(float* x, int64_t ld, int n, int rows, float cap, void* stream);

}  // namespace strata::kernels
