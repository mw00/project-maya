// include/strata/kernels/q35_kernels.hpp - the BATCHED kernels of the dense qwen35 model (Qwen3.8-27B): a forward
// pass over T tokens at once (a prompt chunk, or a speculative verify window of the anchor + its drafts).
//
// Row layouts are token-major: activation row t of width n is x + t * ld (ld >= n).
//
// GDN.  The recurrence is sequential in the tokens, so q35_gdn_seq walks them inside one kernel (block = value head,
// thread = state column j, the 128 state values of the column in registers).  `commit` = false leaves the state as it
// was: a verify window runs with commit off, and after the target has picked how many tokens it keeps,
// q35_gdn_seq / q35_conv_commit run again over those tokens only (their inputs stashed by the window) with commit on.
// State layout: Maya's (S, h_v, S) = state[(i * h_v + h) * S + j] (see gdn.hpp).
//
// Attention.  Query rows are (token, head of the KV group) pairs: row r = t * G + g reads KV head kvh, query head
// kvh * G + g; row t sees cache cells [0, p0 + t].  One block holds 64 rows of one KV head, so a K/V tile is read once
// for all G heads of the group; S = Q K^T and O += P V on the tensor cores (WMMA, FP16 in, FP32 accumulate), the
// online softmax rescaled lazily (the prompt kernel of glm_gqa.cu).  With splits > 1 the cells are cut into pieces
// whose partial (m, l, O) a merge kernel folds: the decode / verify case, where 1-16 tokens leave too few blocks.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// out_f32 / out_f16 (either may be null) = rms_norm(x row) * w, rows of `cols`, x row stride ldx, outputs dense.
void q35_rms_rows(const float* x, int ldx, const float* w, float* out_f32, uint16_t* out_f16, int rows, int cols,
                  float eps, void* stream);
/// x += a (rows of cols, a contiguous), then q35_rms_rows of the new x - the same bits as q35_add + q35_rms_rows.
void q35_add_rms_rows(float* x, int ldx, const float* a, const float* w, float* out_f32, uint16_t* out_f16, int rows,
                      int cols, float eps, void* stream);
/// dense_swiglu's out, and its FP16 copy (the next GEMV's input rows).
void q35_swiglu_h(const float* gate, const float* up, float* out, uint16_t* out_f16, int64_t n, void* stream);

/// dst (FP16 bits) = src, n values.
void q35_to_half(const float* src, uint16_t* dst, int64_t n, void* stream);

/// The causal conv1d (4 taps) + SiLU over T tokens: raw row t (stride ld) -> out row t (stride ld).  conv_state
/// (3 x C, oldest first, Maya's [c * 3 + tap] layout) is read; with commit it ends holding the last three inputs.
void q35_conv_seq(float* conv_state, const float* raw, int ld, const float* w, float* out, int T, int C, bool commit,
                  void* stream);

/// x / sqrt(sum(x^2) + eps) over each 128-wide row of the first n_rows rows of every token (q and k heads).
void q35_qk_l2(float* x, int ld, int T, int n_rows, float eps, void* stream);

/// g = softplus(alpha + dt) * ssm_a, b = sigmoid(beta), in place, rows of h_v for T tokens (row stride ld).
void q35_gate_beta(float* alpha, float* beta, int ld, const float* dt, const float* ssm_a, int T, int h_v, void* stream);
/// q35_conv_seq + q35_qk_l2 (the first n_qk_rows 128-wide rows) + q35_gate_beta in one launch, the same arithmetic;
/// false (nothing launched) when C is not a multiple of 128
bool q35_gdn_prep(float* conv_state, const float* raw, int ld, const float* w, float* out, int T, int C, bool commit,
                  int n_qk_rows, float eps, float* alpha, float* beta, int ldg, const float* dt, const float* ssm_a, int h_v,
                  void* stream);

/// The delta rule over T tokens (see the header).  x rows: [q (h_k*128) | k (h_k*128) | v (h_v*128)], stride ld;
/// g, b rows of h_v (row stride ldg); o rows of h_v*128 (dense).  q scaled by 1/sqrt(128) inside.
/// The GDN value heads' order: 0 (default) GGUF's tiled one (value head h reads key head h % h_k), g > 0 the model's
/// grouped one (h / g, g = h_v / h_k) - the EXL3 checkpoints keep it.
void q35_set_gdn_grouped(int group);
/// bytes an element of the GDN recurrent state (FP16 storage; -DQ35_STATE_F32 for F32)
#ifdef Q35_STATE_F32
constexpr int kQ35StateBytes = 4;
#else
constexpr int kQ35StateBytes = 2;
#endif
/// d_rec (a verify pass; with the pipelined kernel): each token's update d [T][h_v][128] recorded for commit
void q35_gdn_seq(float* state, const float* x, int ld, const float* g, const float* b, int ldg, float* o, int T,
                 int h_k, int h_v, bool commit, void* stream, float* d_rec = nullptr);
/// scratch for the chunked prompt form (STRATA_DENSE_GDN_CHUNK=1): ~(2 h_v 128 + 32 h_v) floats a token of a range
void q35_set_gdn_scratch(float* p, int64_t floats);

/// One GDN layer's commit inputs: its conv state (3 x C) and recurrent state, the verify window's stashed raw conv
/// input and conv output rows (stride C), and its [g | b] rows (stride 2 * h_v).
struct Q35CommitLayer {
    float* conv_state;
    const float* raw;
    float* state;
    const float* conv;
    const float* ab;
    const float* d = nullptr;   // the verify pass's recorded updates (commit with rec)
};
/// commit of n kept tokens for every layer of tab_dev (a device array): conv states, then the recurrence (the result
/// of q35_conv_seq + q35_gdn_seq with commit on, without their outputs) - two launches for all layers
/// rec: the recurrence from the recorded updates (Q35CommitLayer::d), not recomputed - the same bits
void q35_commit_all(const Q35CommitLayer* tab_dev, int n_layers, int n, int C, int h_k, int h_v, void* stream,
                    bool rec = false);

/// rms(o) * w * silu(z) per 128-wide head (`rows` heads in all, dense)
void q35_gdn_out(const float* o, const float* z, const float* w, float* y, int rows, float eps, void* stream);

/// A layer's K / V cache.  mode 0: FP16 rows ([n_kv][max_ctx][D] halves in k, v).  mode 1 (K8V4): K INT8 codes
/// ([n_kv][max_ctx][D] bytes in k) with an FP16 scale per 32 (ks), V rotated by the orthonormal D-point Walsh-Hadamard
/// transform then 4-bit codes, two a byte ([n_kv][max_ctx][D / 2] in v), an FP16 scale per 32 (vs); the attention's
/// output is rotated back (H H = I).  416 bytes a head and position instead of 1024.
struct Q35Kv {
    void* k = nullptr;
    void* v = nullptr;
    uint16_t* ks = nullptr;
    uint16_t* vs = nullptr;
    int mode = 0;
    int max_ctx = 0;
};
/// bytes of one layer's cache
uint64_t q35_kv_bytes(int mode, int n_kv, int D, int max_ctx);

/// The attention layer's per-token preparation for T tokens at positions p0 + t:
///   q rows (from q_full: per head [q | gate] of 2 * D, stride ldq): rms * q_norm, NeoX rope on the first n_rot dims
///   -> q_out (rows of n_head * D, dense);  k rows (stride ldk): rms * k_norm, rope -> K cache cell p0 + t;  v rows
///   (stride ldk) -> V cache cell.  Caches [n_kv][max_ctx][D] FP16.
/// A K8V8 cache's keys [0, n_keys) unpacked to FP16 at the same cells of k16 / v16 (an FP16 cache's layout, n_kv x
/// max_ctx rows of D) - the same values the attention kernels unpack (the prompt path then reads them as FP16).
void q35_kv_expand_f16(const Q35Kv& kv, int n_keys, int n_kv, int D, uint16_t* k16, uint16_t* v16, void* stream);
/// images: a device int32 [cells][3] table of (t, h, w) rotary positions read by q35_attn_prep (interleaved M-RoPE,
/// pair i of the rotated dims takes component i % 3); nullptr (the default) rotates by the cell index.  Set it before
/// any graph is captured; its contents may change between requests
void q35_set_mrope_table(const int32_t* device_table);
void q35_attn_prep(const float* q_full, int ldq, const float* k, const float* v, int ldk, const float* q_norm,
                   const float* k_norm, float eps, float rope_base, int n_rot, int p0, int T, int n_head, int n_kv,
                   int D, float* q_out, const Q35Kv& kv, void* stream);

/// Scratch floats q35_attn needs for T tokens with up to max_ctx cells (the split partials).
int64_t q35_attn_scratch_floats(int T, int n_head, int D, int max_ctx);

/// out rows (n_head * D, dense) = softmax(q k^T / sqrt(D)) V over the causal cells; then out *= sigmoid(gate) where
/// gate is the second half of every head's 2 * D block of q_full (stride ldq).  D must be 256, n_head / n_kv <= 8.
/// splits: 0 = choose.
void q35_attn(const float* q, const Q35Kv& kv, int p0, int T, int n_head, int n_kv, int D, const float* q_full,
              int ldq, float* out, float* scratch, int splits, void* stream);

/// Windows of up to this many tokens (and G = 6) run a CUDA-core split-K decode kernel instead of the tile kernel
/// (STRATA_DENSE_ATTN_DEC, default 1: decode steps; the kernel reads the cache once a token, so wider windows lose);
/// the setter overrides the environment (benchmarks).
int q35_attn_dec_max_tokens();
/// the decode kernel's keys in flight a warp (2, 4 or 8), target block count and minimum resident blocks a SM (1, or 4:
/// registers capped so 4 fit) - benchmarks; <= 0: the environment's (STRATA_DENSE_ATTN_DEC_PF / _BLOCKS / _MINB)
void q35_set_attn_dec_tune(int pf, int blocks, int minb);
/// the streaming kernel (1-5 tokens, FP16 cache): on / off, its block count and keys a tile (32 / 64) - benchmarks;
/// < 0: the environment's (STRATA_DENSE_ATTN_STREAM, _STREAM_SK); blocks <= 0: one wave of what fits
/// The prompt path's attention kernel: 1 the 8-warp attn_pp_kernel (default), 0 attn_tc_kernel, -1 the env's.
void q35_set_attn_pp(int on);
void q35_set_attn_stream(int on, int blocks, int sk);
void q35_set_attn_dec_max_tokens(int n);
/// the tile kernel's shape for 2-8 token windows (STRATA_DENSE_ATTN_VAR: 0 the 64-row tile, 1 32 rows / 2 warps, 2 the
/// same with 16 keys a step); the setter overrides the environment (benchmarks)
void q35_set_attn_var(int v);

/// y[t][r] = sum_i W[r][i] x[t][i] for T tokens: an F32 matrix (n_out rows of n_in), x rows stride ldx, y dense.
void q35_gemv_f32_cols(const float* W, const float* x, int ldx, float* y, int n_in, int n_out, int T, void* stream);

/// out[t] = argmax of row t (n wide, stride ld), ties to the lower id.
void q35_argmax_rows(const float* x, int64_t ld, int n, int T, int32_t* out, void* stream);
/// *out = the softmax probability of x[*best] in the row x [n] (a draft's confidence: its top-1, *best its argmax).
void q35_top1_prob(const float* x, int n, const int32_t* best, float* out, void* stream);
/// x[i] = map[x[i]] for i < n (device): a draft head's slots to token ids.
void q35_map_ids(int32_t* x, const int32_t* map, int n, void* stream);

/// y += x, n values
void q35_add(float* y, const float* x, int64_t n, void* stream);

/// out = silu(g) * u for T rows: g and u rows of n (strides ldg), out dense rows of n
void q35_swiglu_rows(const float* g, const float* u, int ldg, float* out, int n, int T, void* stream);

/// The tensor-core GEMV for 1..8 token rows (q35_mmt.cu): weights Q4_K / Q5_K / Q6_K / Q8_0 in GGUF blocks.
bool q35_mmt_supported(int ggml_type);
/// xh8 (FP16 [8][K], rows M..7 zero) = x rows 0..M-1 (dense rows of K)
void q35_mmt_prep(const float* x, int K, int M, uint16_t* xh8, void* stream);
/// floats of the k-split partials for N outputs
int64_t q35_mmt_part_floats(int N);
/// y (M rows of N, dense) = W . X^T, W N rows of K in `ggml_type` blocks
void q35_mmt(int ggml_type, const void* W, const uint16_t* xh8, float* y, int K, int N, int M, float* part,
             void* stream);

/// The packed layout of q35_mmtp (32-row groups, blocks interleaved at 16 bytes; see q35_mmt.cu): its bytes, the
/// repacking of a GGUF matrix (W N rows of K) into it, and the GEMV over it (as q35_mmt).
uint64_t q35_mmt_packed_bytes(int ggml_type, int K, int N);
void q35_mmt_pack(int ggml_type, const void* W, void* packed, int K, int N, void* stream);
/// (mmt_bench) q35_mmtp's kernel with only its weight loads (1) or only its math (2), 0: whole; splits > 0: that
/// many super-block splits instead of the heuristic's.
void q35_mmtp_set_probe(int probe, int splits = 0);
/// A packed matrix of the rows ids[0..n) of another packed matrix (both K inputs; dst row j = src row ids[j]).
void q35_mmt_gather_packed(int ggml_type, const void* packed, int K, const int* ids, int n, void* dst, void* stream);
void q35_mmt_unpack(int ggml_type, const void* packed, void* W, int K, int N, void* stream);
/// rows row0 .. row0 + rows - 1 (row0 a multiple of 32) of a packed matrix to FP16 rows of K (the prompt's GEMMs)
void q35_mmt_dequant_packed(int ggml_type, const void* packed, int K, int N, int row0, int rows, uint16_t* out,
                            void* stream);
/// the packed form's dp4a GEMV for 1..3 rows: x as the rows' Q8_1 blocks (native_quantize_q8_1), K a multiple of 256
bool q35_mmtp_dp4a_supported(int ggml_type);
void q35_mmtp_dp4a(int ggml_type, const void* packed, const void* xq8_1, float* y, int K, int N, int M, float* part,
                   void* stream);
void q35_mmtp(int ggml_type, const void* packed, const uint16_t* xh8, float* y, int K, int N, int M, float* part,
              void* stream);

/// The MoE decode's expert GEMVs: for each pair p, y[p] (N) = W_e . x[row] with e = pair_exp[p], row = pair_row[p]:
/// W_base the stacked experts in GGUF rows (expert e at e * expert_bytes), xh FP16 rows of K.  part: splits x P x N.
void q35_mmt_pairs(int ggml_type, const void* W, size_t expert_bytes, const int32_t* pair_exp, const int32_t* pair_row,
                   const uint16_t* xh, float* y, int K, int N, int P, float* part, void* stream);

}  // namespace strata::kernels
