// include/strata/kernels/glmd_batch.hpp - the glm-dsa batched prompt kernels (glm-dsa B9.2-B9.5) and their host plans
// and CPU twins (B9.1 tiling, B9.3 multi-row selection, B14.6, B14.7, B14.12).
//
// The prompt path runs a chunk of T tokens at positions p0..p0+T-1 through every layer.  Per glm-dsa layer and per
// query sub-batch of at most PromptTiling::q_sub tokens (the GEMMs in between are cuBLAS, src/core/glm_prefill_dsa.cu):
//   dsa_prep_rows       B4.3 for T rows: qr = RMS(qr_raw) * q_a_norm (f32 + FP16 for the q_b GEMM), latent[p] and
//                       k_pe[p] for every row, full layers: ik_cache[p] = rope(LayerNorm(ik_raw)).  The arithmetic is
//                       glmd::dsa_prep's, row by row, so a row's bytes equal the token path's (B9.2 gate: T = 1 exact).
//   dsa_rope_q_rows     B4.3's q-side rope for T rows (q_pe of every head, the indexer query dims [0, idx_rope)).
//   index_score_tile    the token-level indexer score of the sub-batch's queries against ONE block of at most
//                       key_block keys (16K): I(t, s) = sum_h w_h(t) relu(iq_h(t) . ik(s)).  The score scratch is
//                       q_sub x key_block floats whatever the context (B9.1: no T x max_ctx buffer).
//   topk_merge_tile     the multi-row top-2048: each row keeps a running candidate set (<= top_k cells in ascending
//                       cell order with their scores); a tile merges its causal cells into it.  Over all tiles the
//                       result is EXACTLY glmd::topk's selection of the whole row (proof in the .cu), ascending, -1
//                       padded, so shared layers reuse it as is (A-96).
//   mla_prompt_f32      B9.4: absorbed MLA with the rope tail, F32 arithmetic over the FP16/INT8 cache, per (query,
//                       16 heads): gather of the query's selected cells, or the dense causal / masked walk (the
//                       crossover is a parameter, B14.12).
//   mla_prompt_wmma     B9.5: the same on Volta FP16 tensor cores at width 576 (S = Q K^T and O += P V with WMMA,
//                       f16 P, f32 O in shared memory, 1/l at the end; 91,456 B of dynamic shared memory, checked at
//                       compile time and at launch).
//   mla_prompt_h64      B9.5's arithmetic with all 64 heads of a row in one block (M = 64: a cache tile serves every
//                       head), O in WMMA accumulators, the next tile prefetched into registers (97,088 B).
//   route_rows          the HF router per row (sigmoid, bias for selection only, top-k ties to the lower id,
//                       weights / (sum + 1e-20) * scale): glmd::cpu::route_select with kRouteNormHf, batched.
//   res_rows            the plain residual for a chunk: R += block_out (optional), x = RMS(R) * w, x16 (B2.3's
//                       res_read over T rows).
// All 64-bit offsets (B14.6): every T x keys x heads product is formed in size_t.
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/kernels/glmd_fast.hpp"

namespace glmd {

/// Launch errors and refused shapes of glmd_batch.cu's kernels (each one makes the prompt wrong; tests read it).
int batch_launch_errors();

// ---- host plans (plain C++; src/core/glm_dsa_prompt_plan.cpp) -----------------------------------------------------

/// The prompt path's DSA scratch for one chunk (B9.1, B14.6).  Bytes are 64-bit.
struct PromptDims {
    int64_t T = 8192;          // chunk tokens
    int64_t max_ctx = 65536;   // the context the chunk ends in (keys a query may see)
    int n_head = 64, kv_lora = 512, qk_rope = 64, qk_nope = 192, v_head = 256;
    int q_lora = 2048, idx_heads = 32, idx_dim = 128, top_k = 2048, n_embd = 6144;
};
struct PromptTiling {
    int key_block = 16384;     // keys per score tile (MAYA_GLMD_KEY_BLOCK)
    int q_sub = 2048;          // queries per sub-batch: q_abs / ctx / score tile rows (MAYA_GLMD_Q_SUB)
    int n_sub = 0;             // sub-batches of the chunk
    int n_tiles_max = 0;       // key tiles of the last sub-batch
    // per sub-batch scratch (bytes)
    size_t proj = 0;           // qr_raw, qr, qr16, kv_raw 576, ik_raw, iw, q (n_head x 256), iq
    size_t score = 0;          // q_sub x key_block floats
    size_t run = 0;            // two running candidate sets (cells + scores) of q_sub x top_k
    size_t mask = 0;           // the dense-masked walk's bit rows: q_sub x ceil(max_ctx / 32) words
    size_t q16 = 0;            // FP16 q_nope (the B9.7 products' input)
    size_t q_abs = 0, ctx = 0, attn = 0;
    size_t sub_bytes = 0;      // all of the above, 256-aligned
    // across the chunk (kept from a full layer to the shared layers after it)
    size_t sel_bytes = 0;      // T x top_k ints
    size_t keep_bytes = 0;     // sel + the rope tables (256-aligned)
};
/// The plan for dims under a byte budget for the per-sub-batch scratch (0: no budget).  q_sub halves (down to 64)
/// until sub_bytes fits; key_block halves (down to 1024) after that.  Environment overrides first.
PromptTiling prompt_tiling(const PromptDims& d, size_t budget = 0);

/// Attention walk for a sub-batch whose last query sees n_kv keys (B14.12): Dense while n_kv <= top_k (every
/// causal cell is selected), DenseMasked while n_kv < gather_ratio * top_k (walk every cell, skip unselected ones
/// via a bit row), else Gather (walk the query's selected cell list).  All three give the same set of cells.
enum class AttnWalk : int { Dense = 0, DenseMasked = 1, Gather = 2 };
/// The ratio for the kernel that runs: dflt, or MAYA_GLMD_GATHER_RATIO (>= 1; 1 = always gather).  The 16-head
/// kernels take ik_llama.cpp's 4 (they gather at n_kv >= 4 x top_k); mla_prompt_h64 takes kH64GatherRatio.
double gather_ratio(double dflt = 4.0);
/// mla_prompt_h64 gathers from top_k + 1 keys on, on any device: both walks run the same 16-cell tile (the same work
/// whichever cells fill it), and the gather walk's tiles are the row's selection packed while the dense-masked walk's
/// are every 16-cell window holding a selected cell - never fewer.  V100, 2,048 queries: gather 20.3 / 23.4 / 23.7 ms
/// vs dense-masked 23.7 / 45.3 / 77.2 ms at 3K / 5K / 8K keys, the same cost a tile within 5% (glmd_batch_parity
/// GLMD_BATCH_BENCH=1).
constexpr double kH64GatherRatio = 1.0;
AttnWalk attn_walk(int64_t n_kv, int top_k, double ratio);

/// B14.7: whether a RAM-tier expert of a prompt chunk is uploaded over PCIe (true) or computed on the CPU lane.
/// rows = the chunk rows routed to the expert (~T * 8 / 256 on average).  Upload costs bytes / link_bps once; the
/// lane costs rows * bytes / lane_bps (it streams the expert's bytes per row block) plus the per-row compute.  The
/// threshold in rows is upload_rows_min(); MAYA_GLM_UPLOAD_MIN_ROWS overrides it.
struct LaneLink {
    double link_bps = 10.0e9;     // measured host->device bandwidth of the part's link (B0.11)
    double lane_bps = 60.0e9;     // measured CPU-lane weight streaming bandwidth (both sockets' share)
    double lane_row_s = 2.0e-6;   // CPU-lane compute per row of one expert (s)
    double bytes_per_weight = 0.34375;   // IQ2_M ~ 2.75 bits
    int64_t weights = 3LL * 2048 * 6144; // gate + up + down of one expert
};
int upload_rows_min(const LaneLink& l);
bool upload_expert(int rows, const LaneLink& l);
/// The same rule from the prompt path's own measurements (glm_prefill.cu's plan): upload_ms = one expert's PCIe copy,
/// lane_expert_ms = the lane's fixed cost of an expert (its weights read once, about 2 x the one-token cost),
/// lane_row_ms = the lane's cost per routed row.  rows* = ceil((upload_ms - lane_expert_ms) / lane_row_ms), at least 1
/// (0 when the lane's fixed cost alone exceeds the upload: never forced).  MAYA_GLM_UPLOAD_MIN_ROWS overrides.
int upload_rows_min_measured(double upload_ms, double lane_expert_ms, double lane_row_ms);

/// B9.9: the per-part expert upload schedule of one prompt chunk.  Each part streams only its own layers'
/// RAM-tier experts over its own link from staging pinned on its GPU's NUMA node, through `buffers` (2: double
/// buffered) device staging slots of `slot_bytes`; while part i computes, part i + 1 prefetches its first
/// `prefetch_layers` layers.  Pure arithmetic; glm_dsa_prefill_stream.cpp drives the copies.
struct StreamPart {
    int dev = 0, numa_node = 0, l0 = 0, l1 = 0;
    double link_bps = 10.0e9;
    int64_t ram_bytes_per_layer = 0;   // RAM-tier expert bytes this part uploads per MoE layer (chunk-routed share)
};
struct StreamPlan {
    struct Part {
        int dev = 0, numa_node = 0;
        int buffers = 2;
        int64_t slot_bytes = 0;
        int prefetch_layers = 0;
        double upload_s = 0.0;   // this part's whole chunk upload at its link
    };
    Part parts[8];
    int n_parts = 0;
    double serial_s = 0.0;       // every part's upload one after another (the B9.6 baseline)
    double overlap_s = 0.0;      // max over parts (the three links concurrently)
};
StreamPlan stream_plan(const StreamPart* parts, int n, int64_t staging_bytes_per_part);

// ---- B9.2 batched prep -----------------------------------------------------------------------------------------

/// dsa_prep for T rows at positions a.p + t (or p_dev[t] when a.p_dev is set: one device int per row).  Row t
/// reads qr_raw + t * q_lora, kv_raw + t * kv_ld, ik_raw + t * idx_dim and writes qr / qr16 / qr_q (row t) and the
/// caches at its position.  a.qr_q rows are q_lora / 32 blocks.  T = 1 is byte-identical to glmd::dsa_prep.
struct DsaPrepRows {
    DsaPrep a;
    int T = 1;
    int kv_ld = 576;            // floats between kv_raw rows (the kv_a GEMM's output row: kv_lora + qk_rope)
    uint16_t* qr16 = nullptr;   // FP16 qr rows for the q_b / idx_q_b GEMMs (may be null)
};
bool dsa_prep_rows(const DsaPrepRows& r, void* stream);

/// dsa_rope_q for T rows: row t's q at q + t * q_row, iq at iq + t * iq_row, position a.p + t (p_dev[t] if set).
struct DsaRopeQRows {
    DsaRopeQ a;
    int T = 1;
    int64_t q_row = 64 * 256, iq_row = 32 * 128;
};
bool dsa_rope_q_rows(const DsaRopeQRows& r, void* stream);

// ---- B9.3 indexer score tiles and the multi-row selector ---------------------------------------------------------

/// score[t * ld + j] = sum_h w[t * idx_heads + h] * relu(iq[t] head h . ik_cache[(k0 + j) * idx_dim]) for
/// j < nk.  Cells past a row's position are computed too (never read: the selector counts per row).  F32.
bool index_score_tile(const float* iq, const float* w, const float* ik_cache, int idx_heads, int idx_dim, int T,
                      int64_t k0, int nk, float* score, int64_t ld, void* stream);
/// The same over the state layout's key dtype (B10.8, glm-dsa wave 5): keys F32, or FP16 rows (key_f16) widened to f32
/// as they are staged.  index_score_tile is this with key_f16 = 0.
bool index_score_tile_keys(const float* iq, const float* w, const void* keys, int key_f16, int idx_heads, int idx_dim,
                           int T, int64_t k0, int nk, float* score, int64_t ld, void* stream);
/// The same scores on tensor cores: queries and keys rounded to FP16, products summed in F32 (WMMA), each head's dot
/// relu'd and weighted in head order (a (query, head) row past 65504 is scaled by a power of two, its weight back;
/// F32 keys saturate at +-65504).  idx_dim 128, idx_heads <= 64, 16-byte aligned iq / keys rows.  The glm-dsa prompt
/// path's scorer where the device takes it (MAYA_GLMD_IDX_WMMA=0: the F32 one, which stays the reference).
bool index_score_tile_wmma(const float* iq, const float* w, const void* keys, int key_f16, int idx_heads, int idx_dim,
                           int T, int64_t k0, int nk, float* score, int64_t ld, void* stream);
/// Whether index_score_tile_wmma takes this shape on the current device (opt-in shared memory, checked once).
bool index_score_wmma_fits(int key_f16, int idx_heads, int idx_dim);

/// One key tile into each row's running set.  Row t (position p0 + t) sees n_t = clamp(p0 + t + 1 - k0, 0, nk)
/// cells of the tile; with r_t = run_n[t] the merged candidates are the r_t running cells then the tile's n_t cells
/// (ascending cell order throughout).  r_t + n_t <= top_k: all kept; otherwise the top_k by (score descending, cell
/// ascending), the radix rule glmd::topk uses.  out_cell rows are -1 padded; run_in_* may be null on the first tile
/// (r_t = 0).  Every array is (top_k) per row; run_n_in / run_n_out are T ints.
struct TopkMerge {
    const float* score = nullptr;   // tile scores, row stride ld
    int64_t ld = 0;
    int T = 0;
    int64_t p0 = 0;                 // position of row 0
    int64_t k0 = 0;                 // first cell of the tile
    int nk = 0;
    int top_k = kTopK;
    const int* in_cell = nullptr;
    const float* in_score = nullptr;
    const int* in_n = nullptr;
    int* out_cell = nullptr;
    float* out_score = nullptr;
    int* out_n = nullptr;
};
bool topk_merge_tile(const TopkMerge& m, void* stream);

/// mask[t * words + w] bit b = cell 32 w + b is in sel row t (sel rows of top_k, -1 padded); words = ceil(n_kv/32).
bool sel_to_mask(const int* sel, int top_k, int T, int64_t n_kv, uint32_t* mask, int64_t words, void* stream);

// ---- B9.4 / B9.5 prompt MLA ---------------------------------------------------------------------------------

struct MlaPrompt {
    const float* q_abs = nullptr;   // (kv_lora, n_head) per row, rows n_head * kv_lora apart
    const float* q = nullptr;       // (q_stride, n_head) per row, q_pe at q_pe_off (roped); unused when qk_rope 0
    int q_stride = 256, q_pe_off = 192;
    const void* lat = nullptr;      // FP16 rows or INT8 records
    int lat_q8 = 0;
    const uint16_t* k_pe = nullptr; // FP16 [cell][qk_rope]
    AttnWalk walk = AttnWalk::Gather;
    const int* sel = nullptr;       // Gather: rows of top_k ascending cells, -1 padded
    const uint32_t* mask = nullptr; // DenseMasked: bit rows of mask_words
    int64_t mask_words = 0;
    int top_k = kTopK;
    int64_t p0 = 0;                 // position of row 0 (row t attends cells <= p0 + t)
    int T = 0;
    int n_head = 64, kv_lora = 512, qk_rope = 64;
    float scale = 0.0625f;          // glmd::kGlm53KqScale
    float* ctx = nullptr;           // (kv_lora, n_head) per row
};
/// R in {0, 64} (qk_rope), kv_lora % 32 == 0 and <= 512, n_head % 16 == 0.
bool mla_prompt_f32(const MlaPrompt& a, void* stream);
/// kv_lora 512, qk_rope in {0, 64}, n_head % 16 == 0.  False (and a counted launch error) when the device cannot
/// give the kernel its shared memory: the caller then runs mla_prompt_f32.
bool mla_prompt_wmma(const MlaPrompt& a, void* stream);
/// The WMMA kernel's dynamic shared memory at kv_lora 512 + qk_rope 64 (static_assert'ed <= 98,304 B in the .cu).
constexpr size_t kMlaWmmaSmem = 91456;
/// Whether the WMMA kernel can run on the current device (opt-in shared memory and the attribute check); cached.
bool mla_prompt_wmma_ok();
/// The same arithmetic with a block per (query row, 64 heads) instead of (row, 16 heads): each 16-cell tile of the
/// cache is staged (INT8 decoded) once for all 64 heads, O stays in WMMA accumulators, the next tile's cells are read
/// while the current one is multiplied.  kv_lora 512, qk_rope in {0, 64}, n_head % 64 == 0.  False (and a counted
/// launch error) for a call mla_prompt_h64_fits refuses.
bool mla_prompt_h64(const MlaPrompt& a, void* stream);
/// Whether mla_prompt_h64 takes the call: the shape above, 16-byte aligned rows (q_abs, q's rope part, the caches,
/// ctx) and the device's opt-in shared memory (kMlaH64Smem; checked once per device).
bool mla_prompt_h64_fits(const MlaPrompt& a);
/// The H64 kernel's dynamic shared memory at qk_rope 64: Q 64 x 576 + a 16-cell tile 16 x 576 (FP16), S / P 64 x 16,
/// m / l / alpha, the tile's cells.
constexpr size_t kMlaH64Smem = 97088;
/// HIP (S7): B9.5's arithmetic on RDNA 3 / 3.5 / 4's wave32 WMMA (gfx1100-1102, gfx1150-1151, gfx1200-1201) in ~27 KiB
/// of LDS - a block per (16 heads, query row), Q in registers, 16-cell tiles, O in WMMA accumulators.  kv_lora 512,
/// qk_rope in {0, 64}, n_head % 16 == 0, T <= 65535.  False (nothing launched; the caller runs mla_prompt_f32) for any
/// other call, any other device, a CUDA build, or MAYA_GLMD_PF_RDNA=0.
bool mla_prompt_rdna(const MlaPrompt& a, void* stream);
/// Whether the current device takes mla_prompt_rdna (cached per device; false on CUDA builds).
bool mla_prompt_rdna_ok();

// ---- the rest of the prompt layer -------------------------------------------------------------------------------

/// HF router per row: logits (n_expert) per row -> ids / w (k per row).  n_expert <= 512, k <= 8.  A row with a
/// non-finite logit gets ids -1 (the caller's count check rejects it).
bool route_rows(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w, int norm_mode,
                int T, int* ids, float* w, void* stream);
/// R[t] += add[t] (add may be null); x[t] = RMS(R[t]) * w; x16[t] = FP16(x[t]).  n_embd % 32 == 0, <= 8192.
bool res_rows(float* R, const float* add, const float* w, float eps, int T, int n_embd, float* x, uint16_t* x16,
              void* stream);
/// y[i] += x[i] (n floats, 64-bit n).
bool add_rows(float* y, const float* x, int64_t n, void* stream);
/// Head-major FP16 pack: dst[(t * n_head + h) * K + k] = FP16(src[t * src_row + h * src_head + k]).
bool pack_f16(const float* src, int64_t src_row, int src_head, int n_head, int K, int T, uint16_t* dst, void* stream);
/// A BF16 (type 30) or F32 (type 0) weight narrowed to FP16 (for the FP16 strided-batched products, B9.7).
bool weight_f16(const void* w, int type, int64_t n, uint16_t* dst, void* stream);
/// B14.1 for MMQ activation buffers: every +-inf FP16 in the q8_1 MMQ blocks' scale / sum halves becomes +-65504
/// (ik_llama.cpp's clamp, applied after ggml's quantizer, which has none).  ds_halves_per_block: the halves at the
/// head of each block (8 for the DS4 / D2S6 layouts, 0 for D4 which keeps f32 scales); block_bytes 144.
bool q8_mmq_clamp(void* xq, int64_t n_blocks, int block_bytes, int ds_halves_per_block, void* stream);

// ---- CPU twins (host pointers, the same contracts) ---------------------------------------------------------------

namespace twin {

void dsa_prep_rows(const DsaPrepRows& r);
void dsa_rope_q_rows(const DsaRopeQRows& r);
void index_score_tile(const float* iq, const float* w, const float* ik_cache, int idx_heads, int idx_dim, int T,
                      int64_t k0, int nk, float* score, int64_t ld);
void index_score_tile_keys(const float* iq, const float* w, const void* keys, int key_f16, int idx_heads, int idx_dim,
                           int T, int64_t k0, int nk, float* score, int64_t ld);
/// The GPU merge's exact algorithm (radix cut-off over the merged candidates, lower index wins ties).
void topk_merge_tile(const TopkMerge& m);
/// The whole tiled selection of a sub-batch: rows at p0..p0+T-1, keys 0..p0+T-1, scores from score_fn(t, cell)
/// tile by tile (key_block keys), the result in sel (T x top_k).  The B9.3 CPU gate compares it with twin::topk.
using ScoreFn = float (*)(void* user, int64_t pos, int64_t cell);
void topk_tiled(ScoreFn fn, void* user, int T, int64_t p0, int key_block, int top_k, int* sel);
void mla_prompt(const MlaPrompt& a);

}  // namespace twin

}  // namespace glmd
