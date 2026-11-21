// src/ik2/hip/ik2_hip_kernels.cuh - maya-ik2's AMD build's own kernels (ik2_hip_kernels.cu): what Inkling 2 adds to
// the GGUF matrices that src/ds4/ds4_iq runs (docs/INKLING2_AMD.md, llama.cpp's src/models/inkling2.cpp).  One token a
// call (the decode path): every activation an F32 vector on the device.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ik2hip::k {

// the depth mix's states at most (the saved residual streams and the current one): the model needs ceil(48 / 6) + 1 = 9
// (n_layers_per_attn_res_block 6)
constexpr int kMaxStates = 16;

/// The attention over the depth: the states are the nb rows of blocks [nb][E] and cur [E], each RMS-normed (no gain,
/// eps); for each of the nq queries (query [nq][E]) w = softmax_s(S_s . q / sqrt(E)), out[q] = sum_s w_s S_s.
/// stat: 4 kMaxStates floats of scratch.
void mix(const float* blocks, int nb, const float* cur, int E, const float* query, int nq, float eps, float* stat,
         float* out, cudaStream_t s);

/// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) (* w, a gain of n values, or none), rows of n (ldx / ldy apart; in place ok)
void rms(const float* x, size_t ldx, int rows, int n, const float* w, float eps, float* y, size_t ldy, cudaStream_t s);

/// The causal short convolution with its residual, in place over y [C]: hist [3][C] the stream's last three inputs
/// (oldest first), w [C][4] (ggml's (4, C): tap k of channel c at w[4 c + k], k = 3 the current input);
/// dyn (or null) [heads][4] input-dependent taps, lag 0 the current input: channel c of head c / (C / heads).
/// y = (y + sum_k w[k] in[p - 3 + k] + sum_lag dyn[lag] in[p - lag]) / sqrt(4 (dyn ? 2 : 1) + 1); hist then holds
/// this input as its newest row.
void conv(float* y, float* hist, const float* w, int C, const float* dyn, int heads, cudaStream_t s);

/// Each head's row (x [H][n], n <= 256) times scale [H][n] (or none), scaled down to an L2 norm of at most thr
/// (v min(1, thr / (|v| + 1e-6))), times mul
void head_clip(float* x, int H, int n, const float* scale, float thr, float mul, cudaStream_t s);

/// One token's attention over its layer's cache.  q [H][D] (clipped), rel [H][n_rel], sink [H] (the query's sink
/// logits); the keys at positions lo .. p (nk = p - lo + 1 of them), position j's row at slot j % ring (ring 0: j) of
/// K / V [slots][Hkv][D]; rel_bias [extent][n_rel] (ggml's (n_rel, extent)).  Head h reads kv head h / (H / Hkv):
///   s_j = scale q_h . K_j + (p - j < extent ? rel_h . rel_bias[p - j] : 0) - sink_h
///   out_h = sum_j exp(s_j) V_j / (1 + sum_j exp(s_j))                        (a zero sink logit)
/// S: scratch of H (2 nk + 256) floats (D <= 256).  out [H][D].
void attention(const float* q, const float* rel, const float* sink, int H, int Hkv, int D, int n_rel,
               const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p, int64_t lo,
               float scale, float* S, float* out, cudaStream_t s);

/// The same attention, split over the keys (flash decoding): a launch of key tiles - each block a tile of one kv head:
/// its query heads' scores, their softmax within the tile, the tile's part of the output - sized to the key count
/// (64 keys or 32k both fill the device), then a small combine, which also applies the output gate when `gate`
/// [H][D] is given (out *= sigmoid(gate): the step's next launch folded in).  S: scratch of attention_split_floats.
size_t attention_split_floats(int H, int D, int64_t max_keys);
void attention_split(const float* q, const float* rel, const float* sink, int H, int Hkv, int D, int n_rel,
                     const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p,
                     int64_t lo, float scale, float* S, const float* gate, float* out, cudaStream_t s);

/// attention_split for a verify window's T rows in one pair of launches: the K / V tiles read once for all T x (H /
/// Hkv) queries of a kv head.  Row t (position p0 + t) sees the keys up to its own position and, with window > 0, the
/// last `window` of them; lo0: the first key position row 0 sees.  q [T][H][D], rel [T][H][n_rel], sink [T][H], gate
/// and out [T][H][D].  false (nothing launched): the scratch (S_floats) does not hold the tiles' parts - the caller
/// runs the rows one at a time.
bool attention_split_rows(const float* q, const float* rel, const float* sink, int T, int H, int Hkv, int D, int n_rel,
                          const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p0,
                          int64_t lo0, int window, float scale, float* S, size_t S_floats, const float* gate, float* out,
                          cudaStream_t s);

/// x *= sigmoid(g), n values
void sigmoid_mul(float* x, const float* g, size_t n, cudaStream_t s);
/// y += x
void add(float* y, const float* x, size_t n, cudaStream_t s);
/// The routed experts' SITU activation: gu [rows][2 ff] (gate | up) -> h [rows][ff] =
/// beta tanh(g / beta) sigmoid(g) lb tanh(u / lb) (lb <= 0: u as it is)
void situ(const float* gu, int rows, int ff, float beta, float lb, float* h, cudaStream_t s);
/// out[i] = sum_j w[j] part[j][i] (m rows of n; m 0: zeros; a row of weight 0 is not read)
void wsum(const float* part, int m, int n, const float* w, float* out, cudaStream_t s);
/// x = cap tanh(x / cap), n values
void softcap(float* x, size_t n, float cap, cudaStream_t s);
/// out[0] = the argmax of x[0 .. n) as a float (ties: the lower index), out[1] = its value
void argmax(const float* x, int n, float* out, cudaStream_t s);

// ---- the decode step's fused forms (a launch each where the reference graph has several: a step's time is mostly
// its launches) ---------------------------------------------------------------------------------------------------
/// mix, then each query's result RMS-normed with its gain (g0, g1, g2: nq of them; eps): out [nq][E] - k::mix and
/// k::rms in a launch
void mix_rms(const float* blocks, int nb, const float* cur, int E, const float* query, int nq, const float* g0,
             const float* g1, const float* g2, float eps, float* out, cudaStream_t s);
/// The CPU lane's sum into y, when the token has a pick outside VRAM (an id whose slot[id] < 0): the device waits for
/// *flag (host memory it maps) to reach v, then y += src (n floats of that memory).  No such pick: nothing.
void lane_add(const unsigned* flag, unsigned v, const int* ids, const int* slot, int nu, const float* src, float* y, int n,
              cudaStream_t s);
/// K and V of one token, from their projections to the cache: conv (as k::conv, in place over ky / vy, the histories
/// advanced; dk / dv the global layers' taps or null), then K clipped per head (thr) and V RMS-normed per head (eps) -
/// left in ky / vy and stored as F16 in the caches' row p % ring (ring 0: p).  D <= 256.
void kv_finish(float* ky, float* vy, float* hist_k, float* hist_v, const float* k_conv, const float* v_conv,
               const float* dk, const float* dv, int Hkv, int D, float thr, float eps, uint16_t* K, uint16_t* V, int ring,
               int64_t p, cudaStream_t s);
/// head_clip of q (H rows of D, scale qs, threshold qthr) and of rel (H rows of n_rel, scale rs, threshold rthr), both
/// times mul
void clip2(float* q, int H, int D, const float* qs, float qthr, float* rel, int n_rel, const float* rs, float rthr,
           float mul, cudaStream_t s);
/// k::conv over y (no taps), then cur += y
void conv_add(float* y, float* hist, const float* w, int C, float* cur, cudaStream_t s);

// ---- the prompt path: a chunk of T consecutive tokens (positions p0 .. p0 + T - 1) a call ---------------------------

/// mix for T tokens: state st of token t at blocks + st ldb + t E (ldb floats between two states), cur [T][E];
/// out [nq][T][E].  stat: 4 kMaxStates T floats of scratch.
void mix_rows(const float* blocks, size_t ldb, int nb, const float* cur, int T, int E, const float* query, int nq,
              float eps, float* stat, float* out, cudaStream_t s);
/// conv for T tokens: y [T][C] -> out [T][C] (another buffer), token t's earlier inputs the chunk's rows before it,
/// then hist's; dyn [T][heads][4] or null.  hist then holds the chunk's last three inputs (fewer tokens: shifted).
void conv_rows(const float* y, int T, float* hist, const float* w, int C, const float* dyn, int heads, float* out,
               cudaStream_t s);
/// head_clip for T tokens (x [T][H][n], scale [H][n]); token t's factor 1 + alpha log(max(1, (p0 + t + 1) / floor))
/// (alpha 0: 1)
void head_clip_rows(float* x, int T, int H, int n, const float* scale, float thr, float alpha, double floor, int64_t p0,
                    cudaStream_t s);
/// cache row (p0 + t) % ring (ring 0: p0 + t) = src[t] rounded to F16, rows of n values
void store_rows(const float* src, int T, int n, uint16_t* cache, int ring, int64_t p0, cudaStream_t s);
/// The heads of a chunk regrouped for the batched GEMMs: x [T][Hkv per][D] <-> xp [Hkv][T per][D] (to_p: x -> xp)
void perm_heads(const float* src, float* dst, int T, int Hkv, int per, int D, bool to_p, cudaStream_t s);
/// The chunk attention's middle step.  S [Hkv][T per][nk] holds each query's raw dots with the keys at positions
/// klo .. klo + nk - 1 (query (g, t, r) = head g per + r of the token at position p0 + t); in place it becomes the
/// probabilities: s = scale S + (d < extent ? tbl[t H + h][d] : 0) - sink[t H + h] over the keys the token sees (d =
/// its position - the key's: 0 <= d, and d < window when window > 0), a zero sink logit in the sum; 0 elsewhere.
/// tbl [T H][extent] (rel . rel_bias per distance).
void attn_softmax_rows(float* S, int T, int Hkv, int per, int nk, int64_t klo, int64_t p0, int window, int extent,
                       const float* tbl, const float* sink, float scale, cudaStream_t s);

// ---- the verify window: T (a few) tokens a decode step ----------------------------------------------------------
/// mix_rms for T tokens (state st of token t at blocks + st ldb + t E; out [nq][T][E]): a block a token, each the
/// one-token launch's arithmetic
void mix_rms_rows(const float* blocks, size_t ldb, int nb, const float* cur, int T, int E, const float* query, int nq,
                  const float* g0, const float* g1, const float* g2, float eps, float* out, cudaStream_t s);
/// conv_rows, the histories as they were before the window copied to save [3][C] first (a window is cut back to its
/// first rows from them and its inputs: hist_set4)
void conv_rows_save(const float* y, int T, float* hist, const float* w, int C, const float* dyn, int heads, float* out,
                    float* save, cudaStream_t s);
/// lane_add for T tokens: ids [T][nu], src and y [T][n]; the wait once, when any of the tokens has a pick outside VRAM
void lane_add_rows(const unsigned* flag, unsigned v, const int* ids, const int* slot, int nu, const float* src, float* y,
                   int n, int T, cudaStream_t s);
/// A layer's four histories as after `keep` rows of a window: stream q's hist[q] [3][C[q]] = the last three rows of
/// (h0[q] [3][C[q]], then in[q]'s first `keep` rows)
void hist_set4(float* const hist[4], const float* const h0[4], const float* const in[4], const int C[4], int keep,
               cudaStream_t s);

/// The rows a window's attention takes in one launch pair over `keys` keys with S_floats of scratch (0: not one)
int attention_rows_fit(int H, int D, int64_t keys, size_t S_floats);
/// kv_finish for T rows (T 1: a token): in_k / in_v [T][Hkv D] the projections (left as they are: a window is cut back
/// from them), hk0 / hv0 the histories before the rows, hk1 / hv1 (other buffers) the histories after them; dk / dv
/// [T][Hkv][4] or null; the rows' K / V into the caches at p0 .. (nothing else is written)
void kv_finish_rows(const float* in_k, const float* in_v, int T, const float* hk0, const float* hv0, float* hk1, float* hv1,
                    const float* k_conv, const float* v_conv, const float* dk, const float* dv, int Hkv, int D, float thr,
                    float eps, uint16_t* K, uint16_t* V, int ring, int64_t p0, cudaStream_t s);
/// clip2 for T rows (q [T][H][D], rel [T][H][n_rel]), row t times mul[t]
void clip2_rows(float* q, int T, int H, int D, const float* qs, float qthr, float* rel, int n_rel, const float* rs,
                float rthr, const float* mul, cudaStream_t s);
/// conv_add for T rows: y [T][C] the inputs (left as they are), h0 the history before the rows, h1 (another buffer)
/// the history after them; cur [T][C] += the rows' conv - or, with blk [T][C]: blk = cur, cur = the conv
void conv_add_rows(const float* y, int T, const float* h0, float* h1, const float* w, int C, float* cur, float* blk,
                   cudaStream_t s);
/// The drafter's taps of a window: dst [T][n][E] (rows ld floats apart) from n sources of [T][E] (n <= 8); bit k of
/// vol: source k is host memory the device maps, written by another device (read fresh)
void taps_gather(const float* const* src, unsigned vol, int n, int T, int E, float* dst, size_t ld, cudaStream_t s);
/// A request's logit bias: x[r ld + ids[i]] += vals[i] for each of the n entries (distinct ids) and each of `rows` rows
void bias_rows(float* x, size_t ld, int rows, const int* ids, const float* vals, int n, cudaStream_t s);
/// argmax of each of `rows` rows of x (ld apart, n values): out[2 r] the index as a float, out[2 r + 1] the value
void argmax_rows(const float* x, size_t ld, int rows, int n, float* out, cudaStream_t s);

/// A chunk's expert outputs by entry, whichever set computed them: ent[e] = y[row_of[e]] for the n entries with a row
/// (row_of >= 0; rows of E) - then out[t] = sum_j w[t][j] ent[t nu + j], the picks in their order: a token's sum does
/// not depend on which experts were in VRAM and which pass streamed the others
void scatter_rows(const float* y, const int* row_of, int n, int E, float* ent, cudaStream_t s);
void combine_rows(const float* ent, const float* w, int T, int nu, int E, float* out, cudaStream_t s);

/// Launch failures since the last call (each printed); 0 none
int errors();

}  // namespace ik2hip::k
