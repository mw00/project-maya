// include/strata/kernels/glmd_fast.hpp - glm-dsa decode attention kernels (B4.3 prep, B4.4 mla2, B4.5 top-2048).
//
// The fast-path counterparts of the F32 reference kernels in glmd_ref.hpp, for ONE decode token at position p.
// Every GPU function has a CPU twin in glmd::twin (src/core/glm_dsa_cpu_fast_twin.cpp) that follows the same
// algorithm and the same storage formats, so tests/glm_dsa/glmd_fast_cpu_test.cpp can check the arithmetic and the
// selection on a machine with no GPU; tests/glm_dsa/glmd_fast_parity.cu compares the GPU kernels with the twins and
// with glmd::ref_*.  These are new kernels: Maya's Flash decode (glmf::dsa_prep / dsa_select / mla in glm_fast.cu)
// is untouched, so Flash stays bit-identical by construction.
//
// Decode dataflow per DSA layer (GLM-5.3 sizes; the layer loop wires it in B4.6):
//   GEMVs            qr_raw = W_q_a x [2048], kv_raw = W_kv_a x [576], full layers: ik_raw = W_idx_k x [128]
//   dsa_prep         qr = RMS(qr_raw) * q_a_norm (f32 + q8_1 for W_q_b / W_idx_q_b),
//                    latent[p] = RMS(kv_raw[0, 512)) * kv_norm (FP16 row or Maya's INT8 record),
//                    k_pe[p] = rope_p(kv_raw[512, 576)) (FP16), full layers: ik_cache[p] = rope_p(LN(ik_raw))
//   GEMVs            q = W_q_b qr [64 x 256 = (q_nope 192 | q_pe 64)], iq = W_idx_q_b qr [32 x 128]
//   dsa_rope_q       q_pe and iq dims 0..63 rotated at p, in place
//   score            the token-level indexer score over cells 0..p (B4.6, glmf::dsa_score with n = p + 1)
//   topk             the top-2048 cells in position order (identity while p + 1 <= 2048)
//   mla2             q_abs = W_kb_h^T q_nope, l = (q_abs . c_kv + q_pe . k_pe) * scale, softmax, ctx over c_kv,
//                    out = W_vb_h ctx [64 x 256], f32 and/or q8_1 for attn_output
// All rotations are NORM pairs through src/kernels/cuda/glmd_rope.cuh (the same arithmetic as the reference).
//
// STORAGE FORMATS (shared with Maya's DSA cache, B4.1):
//   latent   FP16 rows of kv_lora (uint16_t bits), or INT8 records of lat8_rec_bytes(kv_lora) bytes: kv_lora int8
//            codes, then one FP16 scale per 32 values (absmax / 127, round to nearest even) - glmf's format;
//   k_pe     FP16 [max_ctx][qk_rope];
//   ik_cache F32 [max_ctx][idx_dim], cell s holds position s (no ring);
//   weights  W_kb, W_vb BF16 in glmd_ref.hpp's index order: wk_b[(h * kv_lora + r) * qk_nope + c],
//            wv_b[(h * v_head + v) * kv_lora + c];
//   q8_1     36-byte blocks {fp16 d, fp16 sum, int8 qs[32]} (ggml block_q8_1; Q81 below is its host view).
// Positions: every kernel takes the host p and an optional device p_dev (read instead when non-null: CUDA-graph
// friendly, B10.1).
//
// B4.6: Glm5Model::fast_dsa_glm (src/core/glm_dsa_mixer.cu, one-line dispatch in fast_dsa) wires them into the layer
// loop through the orchestration in src/core/glm_dsa_mixer.hpp; the token-level scorer is glmd::idx_score
// (glmd_mixer.hpp).
#pragma once

#include <cstddef>
#include <cstdint>

namespace glmd {

/// Maya's INT8 latent record: kv_lora codes + kv_lora / 32 FP16 scales (544 B at 512).  Equals glmf::lat8_rec_bytes.
constexpr int lat8_rec_bytes(int kv_lora) { return kv_lora + kv_lora / 16; }

/// The host view of ggml's block_q8_1 (36 bytes).
struct Q81 {
    uint16_t d;     // fp16 scale
    uint16_t s;     // fp16 sum of the 32 values
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "block_q8_1 is 36 bytes");

/// Top-k width of GLM-5.3's lightning indexer.
constexpr int kTopK = 2048;

/// Launch errors and refused shapes of this file's kernels (each one makes the forward wrong; tests read it).
int launch_errors();

// ---- B4.3 decode prep ---------------------------------------------------------------------------------------

struct DsaPrep {
    // query latent: qr = RMS(qr_raw) * q_a_norm; q_lora % 32 == 0, q_lora <= 2048
    const float* qr_raw = nullptr;
    const float* q_a_norm = nullptr;
    float* qr = nullptr;           // f32 (q_lora), may be null
    void* qr_q = nullptr;          // q8_1 (q_lora / 32 blocks), may be null
    int q_lora = 2048;
    // kv_a output [c_kv kv_lora | k_pe qk_rope]; kv_lora % 32 == 0, kv_lora <= 512, qk_rope even, <= 128
    const float* kv_raw = nullptr;
    const float* kv_norm = nullptr;   // RMS weight of c_kv (kv_lora)
    void* lat = nullptr;              // latent cache: FP16 rows or INT8 records (lat_q8)
    int lat_q8 = 0;
    int kv_lora = 512;
    uint16_t* k_pe = nullptr;         // FP16 [max_ctx][qk_rope]
    int qk_rope = 64;
    const float* inv_freq = nullptr;  // device, qk_rope / 2 (glmd::rope_inv_freq)
    // full layers only (ik_raw == nullptr skips): ik_cache[p] = rope_0..idx_rope( LayerNorm(ik_raw) )
    const float* ik_raw = nullptr;
    const float* ik_ln_w = nullptr;
    const float* ik_ln_b = nullptr;
    float idx_ln_eps = 1e-6f;
    float* ik_cache = nullptr;        // F32 [max_ctx][idx_dim], or FP16 rows when ik_f16 (cast to uint16_t*)
    int idx_dim = 128;                // <= 512
    int idx_rope = 64;
    const float* idx_inv_freq = nullptr;  // device, idx_rope / 2 (may be the same array as inv_freq)
    int p = 0;
    const int* p_dev = nullptr;
    float eps = 1e-5f;                // RMS eps of q_a_norm / kv_a_norm
    int ik_f16 = 0;                   // B10.8: ik_cache holds FP16 rows (uint16_t bits, round to nearest even)
    int ik_had = 0;                   // B14.11: the key is Walsh-Hadamard rotated (orthonormal, idx_dim a power of
                                      // two <= 512, src/kernels/cuda/glmd_had.cuh) after its rope, before the store
    // L1-6 KV streaming (glm_kv_stream.hpp): lat / k_pe are VRAM SLOTS, the row of position p goes to the host copies
    // (device-mapped, position order) always and to slot row kv_table[p / kv_page] * kv_page + p % kv_page when that
    // block is resident.  Null lat_host: lat / k_pe are the identity caches (row p).
    void* lat_host = nullptr;
    uint16_t* kpe_host = nullptr;
    const int* kv_table = nullptr;
    int kv_page = 1;
};
/// One launch (2 blocks, 3 on full layers).  Returns false (and counts a launch error) on a refused shape.
bool dsa_prep(const DsaPrep& a, void* stream);

struct DsaRopeQ {
    float* q = nullptr;               // (q_stride, n_head): rotate dims [rot_off, rot_off + n_rot) of each head
    int n_head = 64, q_stride = 256, rot_off = 192, n_rot = 64;
    const float* inv_freq = nullptr;  // device, n_rot / 2
    float* iq = nullptr;              // (idx_dim, idx_heads): dims [0, idx_rope); null skips (shared layers)
    int idx_heads = 32, idx_dim = 128, idx_rope = 64;
    const float* idx_inv_freq = nullptr;
    int p = 0;
    const int* p_dev = nullptr;
    int iq_had = 0;                   // B14.11: each indexer query head is Hadamard rotated after its rope (iq set);
                                      // must equal the keys' DsaPrep::ik_had (GlmStateLayout::ik_had decides both)
};
/// One launch, two with iq_had (the per-head rotation follows the rope on the same stream).
bool dsa_rope_q(const DsaRopeQ& a, void* stream);

// ---- B4.4 decode MLA with a rope tail ---------------------------------------------------------------------

struct Mla2 {
    const float* q = nullptr;         // (q_stride, n_head): [q_nope qk_nope | q_pe qk_rope], q_pe rotated
    int q_stride = 256;               // >= qk_nope + qk_rope, % 4 == 0
    const uint16_t* wk_b = nullptr;   // BF16 (or Q8_0 blocks when w_type == 8)
    const uint16_t* wv_b = nullptr;   // BF16 (or Q8_0 blocks when w_type == 8)
    const void* lat = nullptr;
    int lat_q8 = 0;
    const uint16_t* k_pe = nullptr;   // FP16 [cell][qk_rope]; unused when qk_rope == 0
    const int* cells = nullptr;       // device, n_sel entries, -1 = padding
    int n_sel = 0;
    // B4.6 / B14.4: an optional device count.  When set, the attention reads min(n_sel, *n_sel_dev) cells and n_sel
    // is only the capacity (grid and work size), so one launch shape serves every position (CUDA-graph replay safe);
    // chunks past the count write an empty partial and return.  Null: n_sel cells, as before.
    const int* n_sel_dev = nullptr;
    int n_head = 64, qk_nope = 192, qk_rope = 64, kv_lora = 512, v_head = 256;
    float scale = 0.0625f;            // GLM-5.3: glmd::kGlm53KqScale (1/16); explicit, never derived here
    void* work = nullptr;             // device scratch of mla2_work_bytes(n_head, kv_lora, >= n_sel)
    size_t work_bytes = 0;
    float* out = nullptr;             // f32 (v_head, n_head), may be null
    void* out_q8_1 = nullptr;         // q8_1 of the same n_head * v_head values, may be null
    int w_type = 30;                  // wk_b / wv_b type: 30 BF16 (default), 8 Q8_0 (B10.7: rows of 34-byte blocks,
                                      // qk_nope % 32 == 0; the q_abs and W_vb GEMVs run glmd::headwise, glmd_volta.hpp)
    int legacy_attn = 0;              // L1-4: 1 = the attention through mla2's own split / combine kernels; 0 = mla_dec
                                      // (glmd_dec.hpp) where it serves the shape (GLM-5.3's), else mla2's own
};
/// Split-KV chunk (cells per block) and heads per block of the attention kernel.
constexpr int kMla2Chunk = 32;
constexpr int kMla2HeadsPerBlock = 16;
/// Dynamic shared memory of the split kernel: FP16 latents + FP16 k_pe of one chunk + the chunk's scores
/// (38,912 B at kv_lora 512, qk_rope 64: inside the 48 KB default, no opt-in).
constexpr size_t mla2_smem_bytes(int kv_lora, int qk_rope) {
    return (size_t) kMla2Chunk * (kv_lora + qk_rope) * 2 + (size_t) kMla2HeadsPerBlock * kMla2Chunk * 4;
}
/// q_abs + ctx (n_head * kv_lora each) + one partial (kv_lora + 2) per (chunk, head).  Host-inline (the CPU twins'
/// tests size their scratch with it too).
inline size_t mla2_work_bytes(int n_head, int kv_lora, int max_sel) {
    const size_t n_chunks = (size_t) (max_sel + kMla2Chunk - 1) / kMla2Chunk;
    return ((size_t) 2 * n_head * kv_lora + n_chunks * n_head * (kv_lora + 2)) * sizeof(float);
}
/// q_abs GEMV, split attention + combine, W_vb GEMV: 4 launches.  qk_rope must be 0 or 64; kv_lora <= 512.
bool mla2(const Mla2& a, void* stream);

// ---- B4.5 top-k selector ----------------------------------------------------------------------------------

/// Per row t (T rows, row stride ld floats): n_t = pos ? min(n_kv, pos[t] + 1) : n_kv candidate cells 0..n_t-1.
/// n_t <= top_k: sel = 0, 1, .., n_t - 1, then -1 (identity, the scores are not read).  Otherwise the top_k cells
/// by score (ties to the lower cell), written in ascending cell order.  Scores in [0, n_t) must not be NaN (-inf
/// is an ordinary, smallest value).  One 1024-thread block per row, a 4-pass 8-bit radix select then one
/// position-ordered compaction: O(n) per row, any n (tested to 131072), any top_k.  pos is a device array or null.
bool topk(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel, void* stream);

/// B14.15: the bucket (histogram) selector with topk's exact contract and output (the same cells, the same order, the
/// same ties): per row a min / max pass over the order-preserving keys, a histogram of kTopkHistBuckets buckets spread
/// linearly over [kmin, kmax], the boundary bucket's keys (at most kTopkHistCand, else the radix cut-off) sorted in
/// shared memory, then topk's compaction.  5 passes over the row against topk's 6.  The decode policy runs it under
/// STRATA_GLMD_TOPK_HIST=1 (glmd_decode_policy.hpp); off by default.
constexpr int kTopkHistBuckets = 2048;
constexpr int kTopkHistCand = 4096;
bool topk_hist(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel, void* stream);
/// The bucket of order-preserving key k in a row whose keys span [kmin, kmin + span], span > 0 (monotone in k).
inline int topk_hist_bucket(uint32_t k, uint32_t kmin, uint32_t span) {
    return (int) (((uint64_t) (k - kmin) * (uint64_t) kTopkHistBuckets) / ((uint64_t) span + 1u));
}

// ---- CPU twins (host pointers, same contracts, same formats) ----------------------------------------------

namespace twin {

/// IEEE float -> fp16 bits, round to nearest even (as __float2half); fp16 bits -> float.
uint16_t half_bits(float f);
float half_float(uint16_t h);

/// B14.11: the orthonormal n-point Walsh-Hadamard transform in place (n a power of two, 2..512), the kernels' bits.
void hadamard(float* x, int n);
/// dsa_prep and dsa_rope_q on host arrays (p_dev and the inv_freq arrays are host pointers here).
void dsa_prep(const DsaPrep& a);
void dsa_rope_q(const DsaRopeQ& a);
/// mla2 in float with the GPU's chunked online softmax (a.work is ignored; cells is a host array).
void mla2(const Mla2& a);
/// The exact selection algorithm of glmd::topk (radix cut-off, then position-ordered compaction).
void topk(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel);
/// B14.15: the exact algorithm of glmd::topk_hist (range, histogram, boundary-bucket sort or the radix fallback, the
/// same compaction).  Its output equals twin::topk's on every input (tests/glm_dsa/glmd_topk_hist_cpu_test.cpp).
/// fallbacks (optional) counts the rows that took the radix fallback (a boundary bucket over kTopkHistCand keys).
void topk_hist(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel, int* fallbacks = nullptr);
/// The order-preserving key both selectors use (larger float, larger key; -0 is +0).
uint32_t score_key(float f);

}  // namespace twin

}  // namespace glmd
