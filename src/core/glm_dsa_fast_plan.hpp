// src/core/glm_dsa_fast_plan.hpp - the host-only half of the glm-dsa fast binder (glm-dsa B2.4).
//
// Plain C++ with no CUDA dependency, so the CPU test (tests/glm_dsa/glmd_fast_plan_test.cpp) can check it on any build:
//   * weight resolution: where a glm-dsa weight is served from (a native slot in ws_map_, the BF16 copy in w16_ or an
//     F32 tensor in w_), in the order the fast binder needs it;
//   * the glm-dsa activation arena: every buffer the glm-dsa mixer reads or writes for one decode token, with its
//     offset and size (256-byte aligned), so the device carve is plain arithmetic and an overflow is impossible;
//   * the q8_1 block layout and a host dequantizer (the twin of the device fallback in glm_dsa_fast.cu).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace strata::core::glmdsa {

// The weight types the binder hands to gf::mv (mirrors strata::kernels::glmf::kTypeF32 / kTypeBF16; glm_dsa_fast.cu
// static_asserts they agree).
constexpr int kTypeF32 = 0;
constexpr int kTypeBF16 = 30;
// gf::kMaxMvJobs: the float-input fallback keeps one scratch row per job of a launch.
constexpr int kMaxMvJobs = 8;
// One q8_1 block: half2 {d, sum} then 32 int8 values (36 bytes).
constexpr int kQ8Block = 32;
constexpr int kQ8Bytes = 36;
inline size_t q8_1_bytes(int64_t n) { return (size_t) (n / kQ8Block) * kQ8Bytes; }

/// The gf::mv types moe_gate_up's fused shared-expert down can dot (glm_fast.cu any_row_dot's q8_1 cases: Q8_0 and the
/// K / IQ quants).  Every other mv type runs apart (shexp_down_apart): F32 / BF16 need the float input the fused dot
/// never gets, and B10.7's F16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / IQ4_NL have no case there (the dot would read 0).
/// Mirrors glmd_mv_ext.cuh ext_shared_down_ok, which refuses any other type at the launch.
inline bool shexp_fused_ok(int type) {
    switch (type) {
        case 8: case 10: case 11: case 12: case 13: case 14: case 16: case 17: case 18: case 19: case 21: case 22:
        case 23: case 29: return true;
        default: return false;
    }
}

/// Where a resolved weight comes from.
enum class Src : int {
    Missing = 0,   // in none of the tables (or only as a native type gf::mv has no kernel for)
    Native = 1,    // ws_map_: the GGUF's own quantized rows, a type gf::mv serves (dotted against the q8_1 input)
    Bf16 = 2,      // w16_: the BF16 copy (iq_pack's upcast of a type outside mv, or a tensor the GGUF ships as BF16)
    F32 = 3,       // w_ (or an F32 slot in ws_map_): small tensors under the 65,536-element keep16 cut, and the router
};
const char* src_name(Src s);

/// A resolved weight: its gf::mv type and device pointer (null when Missing).  `why` explains a Missing pick.
struct Pick {
    Src src = Src::Missing;
    int type = -1;
    const void* w = nullptr;
    std::string why;
};

/// The three name -> pointer tables of a loaded pack, as lookups (the binder wraps ws_map_, w16_ and w_; the CPU test
/// wraps std::maps).  native(): true when the name has a ws_map_ entry; type 0 there is an F32 slot (f32 set).
struct Tables {
    std::function<bool(const std::string&, int& type, const void*& q, const void*& f32)> native;
    std::function<const void*(const std::string&)> bf16;
    std::function<const void*(const std::string&)> f32;
    std::function<bool(int)> mv_ok;   // gf::mv_supported
};

/// The q_or_b16 rule for every projection gf::mv consumes, extended to the B1.6 upcast set (attn_q_a, attn_q_b,
/// attn_kv_a_mqa, attn_output, ffn_*_shexp, dense ffn_*, nextn.eh_proj): a native slot whose type mv serves, else the
/// BF16 copy in w16_, else an F32 tensor (an F32 slot in ws_map_, then w_).
Pick pick_mv(const Tables& t, const std::string& name);
/// Tensors kept unquantized by design (attn_k_b, attn_v_b, the indexer projections): the BF16 copy, else F32.  A
/// native slot is not accepted (the consumers read float rows).
Pick pick_float(const Tables& t, const std::string& name);
/// B10.7: W_kb / W_vb for the headwise GEMVs (glmd::headwise, Mla2 / FusedMla w_type): a native Q8_0 slot (ggml type
/// 8, served when the pack keeps them native: tools/strata/glm_dsa_pack.py with STRATA_PACK_NATIVE32=1), else
/// pick_float's BF16 / F32.  The binder passes Pick::type to Mla2::w_type (8 Q8_0, 30 BF16).
constexpr int kTypeQ8_0 = 8;
Pick pick_headwise(const Tables& t, const std::string& name);
/// The router ffn_gate_inp: F32 in w_ first (G23: glm-dsa keeps it F32), else the BF16 copy in w16_.
Pick pick_router(const Tables& t, const std::string& name);

/// The numbers the glm-dsa arena is sized from (Glm5Geometry fields; max_ctx is the model's context cap).
struct ArenaDims {
    int64_t n_embd = 0, n_head = 0, q_lora = 0, kv_lora = 0, qk_nope = 0, qk_rope = 0, v_head = 0;
    int64_t idx_heads = 0, idx_key = 0, idx_top_k = 0, max_ctx = 0, n_ff_dense = 0, n_ff_sh = 0;
};

/// The glm-dsa arena's buffers (one decode token).  Sizes for GLM-5.3 (E 6144, 64 heads, q_lora 2048, kv_lora 512,
/// qk_nope 192, qk_rope 64, v_head 256, indexer 32 x 128, top-k 2048):
enum Buf : int {
    kQ = 0,      // f32  n_head * (qk_nope + qk_rope) = 16384: q_b's output, per head [q_nope 192 | q_pe 64]
    kKvRaw,      // f32  kv_lora + qk_rope = 576: kv_a's output [c_kv 512 | k_pe 64]
    kQAbs,       // f32  n_head * (kv_lora + qk_rope) = 36864: the absorbed query per head [q_abs 512 | q_pe 64]
    kScore,      // f32  max_ctx: the indexer score of every cached position
    kCells,      // i32  idx_top_k = 2048: the selected cells (-1 padded)
    kQrRaw,      // f32  q_lora: q_a's output before attn_q_a_norm
    kQr,         // f32  q_lora: after attn_q_a_norm (q_b's and indexer.attn_q_b's input)
    kQrQ,        // q8_1 q_lora: kQr quantized (a native q_b's input)
    kIkRaw,      // f32  idx_key: indexer.attn_k's output (before the LayerNorm and rope)
    kIq,         // f32  idx_heads * idx_key: indexer.attn_q_b's output
    kIw,         // f32  idx_heads: indexer.proj's output (the per-head weights)
    kAttnOut,    // f32  n_head * v_head = 16384: the attention's output (attn_output's float input)
    kAttnOutQ,   // q8_1 n_head * v_head: kAttnOut quantized (a native attn_output's input)
    kAct,        // f32  kMaxMvJobs x act_floats: the float-input fallback (glmdsa::mv dequantizes a q8_1 input here)
    kNumBufs
};
const char* buf_name(int b);

struct ArenaPlan {
    size_t off[kNumBufs] = {};
    size_t bytes[kNumBufs] = {};   // unpadded
    size_t total = 0;              // every buffer, each rounded up to 256 bytes
    int64_t act_floats = 0;        // one fallback row: the widest q8_1-only GEMV input (n_ff_dense, n_ff_sh, 2 E, ...)
};
/// The arena layout; false + err on a geometry it cannot hold (a width not a multiple of 32, a non-positive size).
bool arena_plan(const ArenaDims& d, ArenaPlan& out, std::string& err);

/// Host twin of the device q8_1 -> f32 dequantizer: y[i] = d_block * qs[i] (n a multiple of 32).
void q8_1_to_f32_host(const void* xq, int64_t n, float* y);
/// Host q8_1 quantizer with the device's rounding (d = min(amax / 127, 65504) stored as FP16, q = round(v / d)).
void q8_1_quantize_host(const float* x, int64_t n, void* xq);
float half_to_float(uint16_t h);
uint16_t float_to_half(float f);

}  // namespace strata::core::glmdsa
