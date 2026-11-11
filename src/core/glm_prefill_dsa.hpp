// src/core/glm_prefill_dsa.hpp - the glm-dsa prompt layer (glm-dsa B9.1, B9.6, B9.7, B14.1) as Maya's prompt path
// (src/core/glm_prefill.cu) calls it.  Internal header: include it with src/core on the include path.
//
// glm_prefill.cu owns the chunk (PrefillState: the rows x / x16 / mixer / R, the weight scratch, the cuBLAS handle,
// the per-layer union `uni`) and hands this file what one glm-dsa layer needs in an Io.  Its hooks (STRATA-EDITS.txt):
//   prefill_setup  glmpf::uncovered (the attention weights' coverage) and glmpf::built() (HIP: token by token);
//   carve_dsa      glmpf::scratch_bytes: the per-layer union is sized for the tiled glm-dsa scratch, not T x max_ctx;
//   chunk_bytes /  glmpf::keep_bytes: the selection kept from a full layer to the shared layers after it (T x 2048
//   prefill_carve  ints) and the rope tables, beside the chunk's rows;
//   prefill_half   hc_read -> glmpf::res_read on a plain residual; the mixer loop -> glmpf::dsa_layer; the router ->
//                  glmpf::route (F32 router, HF normalisation); the last write half -> R += ffn; the NextN cache fill
//                  -> glmpf::mtp_fill; after each MMQ quantize glmpf::mmq_clamp (B14.1, glm-dsa only);
//   prefill_run    the embedding lands in R as is (no 4-stream fan-out).
// Flash (glm5-next) never reaches any of these: every hook is gated on g.arch == GlmArch::GlmDsa or !g.mhc().
#pragma once

#include "glm_dsa_fast.hpp"
#include "strata/core/glm_model.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

#include <cublas_v2.h>   // (HIP: the build's compat headers map it, as for glm_prefill.cu)

namespace strata::kernels { struct KvStreamMap; }
namespace strata::prefill {
class Gemm;   // include/strata/prefill/gemm.hpp
}

namespace strata::core::glmpf {

/// What one prompt layer of a chunk works with (device pointers; valid while the prompt path is lent its memory).
/// L1-6 KV streaming: a part's streamed glm-dsa caches for the prompt path.  A streamed layer's chunk runs on the
/// stage (identity layout: the latent's stage_rows rows, then k_pe's): its prefix comes in from the host copies, the
/// chunk's rows go back to them, and the layer's residency map is reset (its slots may hold blocks the chunk rewrote).
struct KvIo {
    uint8_t* const* lat_host = nullptr;   // per layer (null: not streamed); device-mapped, position order
    uint8_t* const* kpe_host = nullptr;   // per layer (null array: no k_pe copy)
    const strata::kernels::KvStreamMap* map = nullptr;   // per layer
    int lat_row = 0, kpe_row = 0;         // bytes a position
    uint8_t* stage = nullptr;
    int64_t stage_rows = 0;
};

struct Io {
    const Glm5Geometry* g = nullptr;
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    uint16_t* w16 = nullptr;          // weight scratch (FP16 elements)
    int64_t w16_elems = 0;
    float* w32 = nullptr;             // weight scratch (F32 elements)
    int64_t w32_elems = 0;
    uint8_t* uni = nullptr;           // the per-layer union (scratch_bytes)
    uint8_t* keep = nullptr;          // keep_bytes, kept across the chunk's layers
    float* x = nullptr;               // T x n_embd: the layer input (normed), F32 and FP16
    uint16_t* x16 = nullptr;
    float* mixer = nullptr;           // T x n_embd: the attention output
    float* R = nullptr;               // T x n_embd: the plain residual
    int T = 0;                        // chunk tokens
    int64_t p0 = 0;                   // position of row 0
    float* state = nullptr;           // the part's state arena (Glm5Model::state_)
    const GlmStateLayout* lay = nullptr;
    bool lat_q8 = false;
    int64_t max_ctx = 0;
    const glmdsa::GlmDsaFast* dsa = nullptr;
    bool skip_output = false;         // the tail-skip layer: caches only, no attn_output
    int sub = 2048;                   // the mixer sub-batch the union was carved for (PrefillState::sub)
    const KvIo* kv = nullptr;         // L1-6 KV streaming (null: every cache in the arena)
    strata::prefill::Gemm* lt = nullptr;   // HIP (S7): the prompt path's hipBLASLt GEMM (the setup's tuning tables) for
                                           // the FP16 projections; null: cuBLAS / plain hipBLAS (every CUDA build)
};

/// The glm-dsa mixer sub-batch (mixer_sub returns min(T, this) for a glm-dsa part; MAYA_GLM_PREFILL_SUB wins).
constexpr int kSubBatch = 2048;

/// False on HIP: the glm-dsa prompt kernels are CUDA-only, the prompt runs token by token there.
bool built();
/// The FP16 coverage of the cuBLAS projections' weights: BF16 and Maya's dequant_bf16 table (Maya v1.1.3's
/// prefill_setup set, all Flash gets); with dsa (a glm-dsa model on a CUDA build) also the i-quants dequant_f16 routes
/// to iq_dequant_f16 and glm-dsa's extra types F16, Q4_1, Q5_1, Q2_K, IQ1_S (glm_prefill_types.hpp). A glm-dsa type
/// outside the set is reported once on stderr (refusal()).
bool f16_covers(int type, bool dsa = false);
/// dequant_f16 plus the extra types: rows [row0, row0 + rows) of a (.. x cols) weight as FP16 into out.
void dequant16(int type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
               cudaStream_t stream);
/// The glm-dsa attention weights of layer il the prompt layer cannot serve ("" when all are covered).
std::string uncovered(const glmdsa::GlmDsaFast* dsa, int il);

/// prefill_setup's geometry gate: Maya's rule on glm5-next (kv_lora 512, n_head % 16, kda_head_dim 128); on glm-dsa
/// the batched kernels' limits (kv_lora <= 512 and % 32, qk_rope 0 / 64, n_head % 16, idx_key <= 128, top-k > 0).
bool geometry_ok(const Glm5Geometry& g);
/// B14.2 at setup: the expert type's MMQ instances are real on this device (cached per type; always true off
/// glm-dsa and on HIP).
bool mmq_ok(const Glm5Geometry& g, int ggml_type);

/// B14.7 in the prompt plan's CPU share: a RAM-tier expert with fewer routed rows than the measured threshold
/// (glmd::upload_rows_min_measured) is computed on the CPU lane even past the lanes' balance - a short turn never
/// waits for its uploads.  Always false off glm-dsa (Flash keeps Maya's balance alone).
bool lane_first(const Glm5Geometry& g, int rows, double upload_ms, double lane_expert_ms, double lane_row_ms);

/// Per-layer scratch of a chunk of T tokens (the union's glm-dsa share) and what is kept across layers (64-bit).
size_t scratch_bytes(const Glm5Geometry& g, size_t T, int64_t max_ctx);
size_t keep_bytes(const Glm5Geometry& g, size_t T, int64_t max_ctx);

/// The plain residual read half for the chunk: R += block_out (null: none), x = RMS(R) * norm_w, x16.
bool res_read(const Io& io, const float* block_out, const float* norm_w, std::string& err);
/// R += ffn (the last layer's write half before the hop or the head).
bool res_close(const Io& io, const float* ffn, std::string& err);

/// One glm-dsa attention layer for the whole chunk: caches at p0..p0+T-1, the mixer rows (unless skip_output).
bool dsa_layer(const Io& io, int il, std::string& err);

/// The MoE router for the chunk: logits = W_router x (F32 or BF16 router_w), HF top-k (sum + 1e-20) -> ids / w.
bool route(const Io& io, const void* router_w, int router_type, const float* bias, float* logits, int* ids, float* w,
           std::string& err);

/// The NextN block's caches for the chunk's first Tv positions (B9.6, A-97): h = RMS(R) * output_norm,
/// hid = eh_proj([enorm(emb), hnorm(h)]), x = RMS(hid) * attn_norm, then kv_a and the index key written at p0 + t.
/// emb_host: Tv rows of the next tokens' embeddings in pinned host memory (the prompt path's emb_h).
struct MtpIo {
    int il = -1;
    WSlot eh;
    const float *enorm = nullptr, *hnorm = nullptr, *attn_norm = nullptr, *output_norm = nullptr;
    const float* emb_host = nullptr;
    int Tv = 0;
};
bool mtp_fill(const Io& io, const MtpIo& m, std::string& err);

/// B14.1 after ggml's MMQ quantizer: +-inf scale / sum halves of `rows` quantized rows of `cols` become +-65504.
/// No-op unless `on` (glm-dsa) and on HIP.
void mmq_clamp(bool on, void* xq, int64_t rows, int64_t cols, int type, cudaStream_t s);

}  // namespace strata::core::glmpf
