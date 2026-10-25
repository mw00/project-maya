// src/mm3/mm3_kernels.cuh - maya-mm3's own kernels around the EXL3 matrices (mm3_exl3.hpp): MiniMax-M3.1-Flash's
// norms, rotary, MiniMax Sparse Attention (the indexer's block selection and the attention over the kept blocks), the
// sigmoid router and swigluoai.  The arithmetic follows transformers' modeling_minimax_m3_vl.py, in F32; the KV cache
// and the indexer's keys are BF16 (vLLM's and transformers' model dtype).
//
// Layouts: rows of tokens; a token's heads back to back ([T][H][128]); the caches by position, [ctx][Hkv][128] for K
// and V and [ctx][128] for the index keys, of one layer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace mm3::k {

/// y = x * rsqrt(mean(x^2) + eps) * w1 per row of D (w1 = 1 + the stored weight: Gemma's); y16 (if given) the same
/// rounded to FP16.  x and y may alias.
void rms(const float* x, const float* w1, int rows, int D, float eps, float* y, uint16_t* y16, cudaStream_t s);

/// The rotary's inverse frequencies theta^(-2i / n_rot), i < n_rot / 2 (host, F32 as transformers computes them).
void rope_inv_freq(float theta, int n_rot, float* inv /*[n_rot / 2]*/);

/// q [T][Hq][hd] in place: the per-head RMSNorm (one [hd] weight w1 = 1 + w for every head) then NeoX RoPE on the
/// first n_rot dims at pos[t]; k [T][Hkv][hd] the same and v as it is, both written to the caches at row pos[t] as
/// BF16.  inv [n_rot / 2] on the device.
void qk_norm_rope(float* q, const float* k, const float* v, const float* qn1, const float* kn1, const int* pos, int T,
                  int Hq, int Hkv, int hd, int n_rot, const float* inv, float eps, uint16_t* kc, uint16_t* vc,
                  cudaStream_t s, int ring = 0);
/// (ring > 0: the caches are rings of that many rows, position p at row p % ring - the DSpark draft's window.  Hq 0:
///  K and V alone)

/// The indexer's queries iq [T][Hi][D] in place (norm, RoPE, rounded through BF16 as the model dtype keeps them) and
/// its key ik [T][D] (norm, RoPE) into the index cache ic at row pos[t] (BF16).
void index_norm_rope(float* iq, const float* ik, const float* qn1, const float* kn1, const int* pos, int T, int Hi,
                     int D, int n_rot, const float* inv, float eps, uint16_t* ic, cudaStream_t s);

/// bs [Hi][T][nblk]: for query t (at pos[t]) and index head h, block b's score = max over the keys j of block b with
/// j <= pos[t] of iq[t][h] . ic[j] (F32); -inf for a block past the query.  D = 128.
void block_scores(const float* iq, const uint16_t* ic, const int* pos, int T, int Hi, int nblk, int blk, float* bs,
                  cudaStream_t s);

/// kept [Hi][T][topk]: the query's own block (pos / blk) and the topk - 1 best of the blocks before it (ties: the lower
/// block), -1 past the valid ones - MiniMaxM3VLIndexer's selection with sparse_local_block 1, as a set.
void select_blocks(const float* bs, const int* pos, int T, int Hi, int nblk, int blk, int topk, int* kept,
                   cudaStream_t s);

/// out [T][Hq][hd] (F32): each query head over its KV group's kept blocks (kept [Hkv][T][topk], -1 unused), the keys
/// j <= pos[t] of them, softmax(q . k * scale) . v - exact (online softmax in F32).  hd = 128, Hq / Hkv = 16.
void sparse_attn(const float* q, const uint16_t* kc, const uint16_t* vc, const int* kept, const int* pos, int T,
                 int Hq, int Hkv, int topk, int blk, float scale, float* out, cudaStream_t s);

/// The router over logits [T][E] (F32): s = sigmoid, chosen by s + bias (the K largest, ties to the lower expert),
/// weights s / sum over the chosen -> top [T][K], w [T][K].
void route(const float* logits, const float* bias, int T, int E, int K, int* top, float* w, cudaStream_t s);

/// h = swigluoai(g, u) = min(g, L) * sigmoid(alpha min(g, L)) * (clamp(u, -L, L) + 1), elementwise over n values; F32
/// into hf and / or FP16 into h16 (either may be null).
void swigluoai(const float* g, const float* u, size_t n, float alpha, float limit, float* hf, uint16_t* h16,
               cudaStream_t s);
/// The same over gate / up halves of rows: gu [rows][2 ff] (gate first) -> h16 [rows][ff].
void swigluoai_pairs(const float* gu, int rows, int ff, float alpha, float limit, uint16_t* h16, cudaStream_t s);

/// FP16 copies of rows (ld floats a row in, ldo halves out).
void to_f16(const float* x, int rows, int cols, int ld, uint16_t* y, int ldo, cudaStream_t s);

/// out[t] = res[t] + scale * sum over slots i of w[t][i] * y[inv[t * K + i]] + (sh ? sh[t] : 0), rows of D.
void moe_combine(const float* res, const float* y, const int* inv, const float* w, const float* sh, int T, int K,
                 int D, float scale, float* out, cudaStream_t s);

/// x += a (n floats).
void add(float* x, const float* a, size_t n, cudaStream_t s);
/// (the tensor-parallel path) one thread spins until *flag (mapped host memory, a counter) reaches v, wrap-safe.
void wait_flag(const unsigned* flag, unsigned v, cudaStream_t s);
/// (the tensor-parallel path) n floats (a multiple of 4) from src (mapped host memory) into dst (device) by a kernel - no
/// copy engine: a copy waiting behind a wait_flag on one stream holds up the device's host -> device copies of the others
void pull(float* dst, const float* src, size_t n, cudaStream_t s);
/// (the tensor-parallel path) n floats (a multiple of 4) from src (device) to dst (mapped host memory), then *flag = v
/// once every block's writes are the system's: each block fences, the last to count (counter: a device word, 0
/// between calls - it is left 0) raises the flag.
void publish(const float* src, float* dst, size_t n, unsigned* counter, unsigned* flag, unsigned v, cudaStream_t s);
/// Y [T][N] = X [T][K] . W [N][K]^T in F32, one warp an output in a fixed order: a token's row is the same bits
/// whatever T (the token path's and a window's router and indexer agree).
void rows_dot(const float* W, int N, int K, const float* X, int T, float* Y, cudaStream_t s);

// ---- the DSpark draft's

/// out [N][K] BF16 = an MXFP8 matrix: w (E4M3 bytes [N][K]) times 2^(scale - 127), scale (E8M0 bytes [N][K / 32]).
void mxfp8_to_bf16(const uint8_t* w, const uint8_t* scale, int N, int K, uint16_t* out, cudaStream_t s);
/// BF16 copies of rows (ld floats a row in, ldo out).
void to_bf16(const float* x, int rows, int cols, int ld, uint16_t* y, int ldo, cudaStream_t s);
/// out [T][Hq][hd]: each query head over its KV group's keys at the positions lo[t] .. hi (rings of `ring` rows,
/// position p at row p % ring), not causal - softmax(q . k * scale) . v in F32.  hd = 128, Hq / Hkv = 16.
void window_attn(const float* q, const uint16_t* kc, const uint16_t* vc, int ring, const int* lo, int hi, int T,
                 int Hq, int Hkv, float scale, float* out, cudaStream_t s);
/// One Markov step: emb (BF16 [R], for the GEMV) and embf (F32 [R]) = table row *prev.
void gather_row(const uint16_t* table, int R, const int* prev, uint16_t* emb, float* embf, cudaStream_t s);
/// ... then: the next token = argmax(logits + bias) (ties: the lower id) -> *prev and *out; biased (if given) = logits +
/// bias; conf (if given) = sigmoid(w . [hidden ; embf] + b) with w [D + R] F32, b F32.
void markov_pick(const float* logits, const float* bias, int V, int* prev, int* out, float* biased, const float* w,
                 const float* b, const float* hidden, int D, const float* embf, int R, float* conf, cudaStream_t s);

}  // namespace mm3::k
