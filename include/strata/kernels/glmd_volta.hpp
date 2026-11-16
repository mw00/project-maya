// include/strata/kernels/glmd_volta.hpp - glm-dsa decode kernels tuned for Volta (sm_70), glm-dsa B10.2-B10.4, B10.8.
//
// These are ALTERNATIVES to the simple glmd kernels of glmd_fast.hpp (mla2, topk) and to the CUDA-core indexer
// scorer; the simple kernels stay the default.  Each one is selected by its own environment switch (read once per
// process, see the *_enabled() functions) for the kernel tests' dispatchers below:
//   MAYA_GLMD_FUSED_MLA=1   mla_auto() runs mla_fused() (B10.2) instead of mla2() when the shape is supported
//   MAYA_GLMD_IDX_WMMA=1    idx_score() runs the WMMA scorer (B10.3) at T > 1 or n_kv > kIdxWmmaMinDecode
//   MAYA_GLMD_FUSED_SPLIT=n split count of mla_fused (default kFusedSplitDefault)
// The FORWARD does not use these dispatchers or their *_enabled() readers: it decides through the decode policy
// (glmd_decode_policy.hpp: MAYA_GLMD_IDX_WMMA / MAYA_GLMD_TOPK_MB are auto / 1 / 0 there, auto when unset), and
// the index-key dtype (MAYA_GLM_IDX_FP16, B10.8: dsa_prep writes FP16 keys when DsaPrep::ik_f16 is set) comes from
// the state layout (GlmStateLayout::ik_bytes).
// Every kernel has a CPU twin in glmd::vtwin (src/core/glm_dsa_cpu_volta_twin.cpp) that runs the same algorithm
// (tiling, split, selection, accumulation order where the hardware lets it be reproduced); the CPU test
// tests/glm_dsa/glmd_volta_cpu_test.cpp runs them, tests/glm_dsa/glmd_volta_parity.cu pins the kernels to the
// twins and to glmd::ref_* on a GPU.
#pragma once

#include "strata/kernels/glmd_fast.hpp"

#include <cstddef>
#include <cstdint>

namespace glmd {

/// ggml type ids the Volta kernels take for W_kb / W_vb (the headwise GEMVs): BF16 (the pack's upcast) and native
/// Q8_0 (B10.7: 34-byte blocks {fp16 d, int8 qs[32]}, rows of qk_nope resp. kv_lora values).
constexpr int kWTypeBF16 = 30;
constexpr int kWTypeQ8_0 = 8;
/// Bytes of one W_kb / W_vb row of n values in that type (0 for any other type or n % 32 != 0 with Q8_0).
size_t head_row_bytes(int w_type, int n);
/// Launch errors and refused shapes of the Volta kernels (glmd::launch_errors() counts glmd_fast.cu's own).
int volta_launch_errors();

// ---- B10.2 fused sparse-gather MLA decode ---------------------------------------------------------------------
//
// One block holds all heads (the Q tile, up to 64 x 576 FP16 = 72 KB of shared memory, rows padded to 584 halves)
// and walks a contiguous range of the selected cells in 32-cell gather units, each processed as two 16-cell WMMA
// sub-tiles (a 32 x 576 FP16 tile next to the 72 KB Q tile does not fit sm_70's 96 KB):
//   stage    the 16 cells' FP16 latents [c_kv | k_pe] (an INT8 record is decoded, then rounded to FP16), -1 = zero
//   S        64 x 16 = Q . L^T with m16n16k16 WMMA, f32 accumulate; 16 warps = 4 head tiles x 4 k-quarters, the
//            quarters summed through shared memory in the order p0 + (p1 + (p2 + p3))
//   softmax  online, per head: m' = max(m, max_c S*scale), alpha = exp(m - m'), P = fp16(exp(S*scale - m')),
//            l = l*alpha + sum P (the rounded P, so O and l agree); P stays in [0, 1] and O, l stay f32 (the
//            FA rule of B14: f16 P, f32 O, normalized by 1/l once at the end)
//   PV       O(64 x 512, 8 accumulator tiles per warp) = alpha . O + P . L_ckv; the per-row alpha is applied
//            through each lane's row map of the accumulator fragment, read once from a known matrix
//            (load_matrix_sync), which is Volta-correct whatever the fragment layout is
// Split-KV: the n_sel cells are cut into n_split ranges of whole 32-cell units, one block each; each writes its
// unnormalized O and (m, l) per head, a combine kernel merges them (FP32), then the W_vb headwise GEMV.
// q_abs is NOT folded into the attention block: with split-KV every block would re-read W_kb (12.6 MB per
// layer at GLM-5.3 sizes); it is a separate headwise GEMV that writes the FP16 Q tile [q_abs | q_pe] once.
// Launches: q (1), attention (1), combine (1), W_vb (1).
struct FusedMla {
    const float* q = nullptr;         // (q_stride, n_head): [q_nope qk_nope | q_pe qk_rope], q_pe rotated
    int q_stride = 256;
    const void* wk_b = nullptr;       // W_kb rows (h * kv_lora + r) of qk_nope values, w_type
    const void* wv_b = nullptr;       // W_vb rows (h * v_head + v) of kv_lora values, w_type
    int w_type = kWTypeBF16;          // kWTypeBF16 or kWTypeQ8_0 (both weights)
    const void* lat = nullptr;        // FP16 rows or INT8 records (lat_q8)
    int lat_q8 = 0;
    const uint16_t* k_pe = nullptr;   // FP16 [cell][qk_rope]
    const int* cells = nullptr;       // device, n_sel entries, -1 = padding
    int n_sel = 0;
    int n_head = 64, qk_nope = 192, qk_rope = 64, kv_lora = 512, v_head = 256;
    float scale = 0.0625f;
    int n_split = 0;                  // 0: fused_split_default()
    void* work = nullptr;             // device scratch of mla_fused_work_bytes(...)
    size_t work_bytes = 0;
    float* out = nullptr;             // f32 (v_head, n_head), may be null
    void* out_q8_1 = nullptr;         // q8_1 of the n_head * v_head values, may be null
};
constexpr int kFusedHeads = 64;       // the Q tile's rows (n_head <= 64; missing heads are zero rows)
constexpr int kFusedLora = 512;       // kv_lora must equal it
constexpr int kFusedSub = 16;         // cells per WMMA sub-tile
constexpr int kFusedUnit = 32;        // cells per gather unit (the split granularity)
constexpr int kFusedThreads = 512;
constexpr int kFusedSplitDefault = 32;
/// Dynamic shared memory of the attention kernel (97,536 B at qk_rope 64, inside sm_70's 96 KB opt-in with the
/// kernel's 384 B of static shared memory (ptxas)).
constexpr size_t mla_fused_smem_bytes(int qk_rope) {
    return (size_t) (kFusedHeads + kFusedSub) * (kFusedLora + qk_rope + 8) * 2 + (size_t) kFusedHeads * kFusedSub * 4;
}
/// The split count for n_sel cells: min(units, req) ranges of whole units, req <= 0 -> fused_split_default().
int mla_fused_splits(int n_sel, int req);
/// Cells (a multiple of kFusedUnit) per split for that count.
int mla_fused_split_cells(int n_sel, int req);
size_t mla_fused_work_bytes(int n_head, int kv_lora, int qk_rope, int max_sel, int req = 0);
/// The shapes the fused kernel takes: n_head 1..64, kv_lora 512, qk_rope 0 or 64, qk_nope % 32 == 0 (Q8_0) or
/// % 8 == 0 (BF16), v_head % 32 == 0, w_type BF16 or Q8_0.
bool mla_fused_supported(int n_head, int qk_nope, int qk_rope, int kv_lora, int v_head, int w_type);
bool mla_fused(const FusedMla& a, void* stream);
/// MAYA_GLMD_FUSED_MLA=1 (read once); fused_split_default(): MAYA_GLMD_FUSED_SPLIT or kFusedSplitDefault.
bool fused_mla_enabled();
int fused_split_default();
/// The dispatcher B4.6 calls: mla_fused() when fused_mla_enabled() and the shape is supported (the BF16 Mla2
/// weights; a.work must then hold max(mla2_work_bytes, mla_fused_work_bytes)), else mla2().
size_t mla_auto_work_bytes(int n_head, int kv_lora, int qk_rope, int max_sel);
bool mla_auto(const Mla2& a, void* stream);

// ---- B10.7 native headwise GEMV (W_kb / W_vb in Q8_0 or BF16) -------------------------------------------------
/// out[h * R + r] = W[h * R + r] . in[h * in_stride, + K) for r < R (R % 32 == 0), K values per row, w_type BF16 or
/// Q8_0 (K % 32 == 0); q8 != nullptr also writes q8_1 blocks of the n_head * R outputs.  The Q8_0 dot is
/// sum_b d_b * sum_j q_bj x_bj in f32 (the dequantized row's dot to f32 rounding).
bool headwise(const void* W, int w_type, const float* in, int n_head, int R, int K, int in_stride, float* out,
              void* q8, void* stream);

// ---- B10.3 WMMA lightning-indexer scorer (and the CUDA-core scorer it replaces), B10.8 key dtype -------------
//
// score[t * ld + s] = sum_h w[t * idx_heads + h] * relu(iq_h(t) . key[s]) for s <= pos_t (pos_t = pos ? pos[t] :
// n_kv - 1, s < n_kv), -inf for the other s < n_kv.  Keys are F32 or FP16 (key_f16) rows of idx_dim, cell s =
// position s.  The WMMA kernel stages keys and queries in FP16 (keys read in place: an F32 cache is converted
// per 16-cell tile in registers, never as a whole) and multiplies with f32 accumulation, so no product or sum is
// formed in f16 (llama.cpp 1b43d311: f16 products overflow); the ReLU x head-weight sum is fused in f32.  One
// block = 8 warps x 16 cells against up to kIdxTokPerBlock tokens.  The CUDA-core scorer (idx_score_simple) is
// the exact F32 dot (glmf::dsa_score's arithmetic) on F32 or FP16 keys.
struct IdxScoreWmma {
    const float* iq = nullptr;   // (idx_dim, idx_heads, T), rope applied
    const float* w = nullptr;    // (idx_heads, T): the head weights incl. 1/sqrt(idx_heads * idx_dim)
    const void* keys = nullptr;  // F32 or FP16 [n_kv][idx_dim]
    int key_f16 = 0;
    int idx_heads = 32, idx_dim = 128;
    int n_kv = 0, T = 1;
    const int* pos = nullptr;    // device (T) or null
    float* score = nullptr;      // (ld, T)
    int ld = 0;                  // >= n_kv; 0 = n_kv
};
constexpr int kIdxWmmaMinDecode = 16384;   // the decode crossover of B10.3 (n_kv > it)
constexpr int kIdxTokPerBlock = 4;         // tokens per WMMA block (the B tile 128 x 128 FP16)
/// WMMA shapes: idx_heads % 16 == 0, idx_heads <= 32, idx_dim % 16 == 0, idx_dim <= 256, and the warp's 16 x idx_heads
/// f32 output scratch fits the 16 x (idx_dim + 8) FP16 key tile it reuses: 64 * idx_heads <= 32 * (idx_dim + 8),
/// i.e. 2 * idx_heads <= idx_dim + 8 (GLM-5.3 32 x 128 fits; 32 x 32 or 16 x 16 would overrun the next warp's tile).
/// The one rule: idx_score_wmma_supported (CUDA) and glmd::idx_wmma_shape (the CPU policy) both return it.
constexpr bool idx_wmma_shape_rule(int idx_heads, int idx_dim) {
    return idx_heads > 0 && idx_heads % 16 == 0 && idx_heads <= 32 && idx_dim > 0 && idx_dim % 16 == 0 &&
           idx_dim <= 256 && 2 * idx_heads <= idx_dim + 8;
}
bool idx_score_wmma_supported(int idx_heads, int idx_dim);
bool idx_score_wmma(const IdxScoreWmma& a, void* stream);
bool idx_score_simple(const IdxScoreWmma& a, void* stream);
/// MAYA_GLMD_IDX_WMMA=1 (read once; off when unset).  The kernel tests' dispatcher only: the forward decides through
/// the decode policy, where the same switch defaults to auto (glmd_decode_policy.hpp).
bool idx_wmma_enabled();
/// WMMA when idx_wmma_enabled() and (T > 1 or n_kv > kIdxWmmaMinDecode) and the shape is supported, else simple.
bool idx_score(const IdxScoreWmma& a, void* stream);

// ---- B10.4 multi-block radix top-k ----------------------------------------------------------------------------
//
// The same contract and the same output as glmd::topk (ascending cell order, ties to the lower cell, identity
// with -1 padding when n_t <= top_k; no final sort), for long rows: kTopkMbBlocks blocks per row share each
// radix pass through a global histogram (per-block shared histograms, one atomicAdd per bin, the last block to
// finish picks the digit), 4 passes of 8 bits; then each block counts its range's winners, the last block scans
// the per-block counts (keys above the cut-off, and the equal ones in cell order for the ties), and every block
// writes its winners at its offset.  Fixed grids (n read on the device from pos), so it is graph-safe: 6 launches
// per call, all T rows at once.  work: topk_mb_work_bytes(T), zeroed once (topk_mb_init) and left zeroed.
constexpr int kTopkMbBlocks = 64;
constexpr int kTopkMbThreads = 512;
constexpr int kTopkMbMin = 32768;
size_t topk_mb_work_bytes(int T);
bool topk_mb_init(void* work, int T, void* stream);
bool topk_mb(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel, void* work,
             void* stream);

// ---- CPU twins ---------------------------------------------------------------------------------------------
namespace vtwin {

/// mla_fused on host arrays (cells host; the same Q/L FP16 rounding, sub-tiles, k-quarter order, online softmax
/// with fp16 P, split ranges, combine and headwise GEMVs).
void mla_fused(const FusedMla& a);
/// headwise on host arrays.
void headwise(const void* W, int w_type, const float* in, int n_head, int R, int K, int in_stride, float* out,
              Q81* q8);
/// The WMMA scorer's arithmetic: fp16(iq) . fp16(key) products in f32 (exact), summed in 16-wide k-steps, ReLU x
/// w summed over heads in order, f32.
void idx_score_wmma(const IdxScoreWmma& a);
/// The CUDA-core scorer: f32 dot of iq with the key (FP16 keys widened), lane-strided sums.
void idx_score_simple(const IdxScoreWmma& a);
/// The multi-block selection: the same per-block decomposition (blocks of the row, counts, scans, offsets).
void topk_mb(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel);
/// Q8_0 row helpers for tests: quantize n values (n % 32 == 0) like ggml's quantize_row_q8_0 (d = amax / 127,
/// q = round(x / d)), dequantize.
void q8_0_quantize(const float* x, int n, void* blocks);
void q8_0_dequantize(const void* blocks, int n, float* y);

}  // namespace vtwin

}  // namespace glmd
