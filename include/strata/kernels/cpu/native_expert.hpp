// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

/// Bytes of the largest quantized activation any native layer uses.  qwen4exp: 2560 values as
/// Q8_K (10 x 292 = 2604); glm5-next: 4096 values as Q8_K (16 x 260 = 4164); glm-dsa: 6144 values as Q8_K
/// (24 x 292 = 7008, fits; 8192 would need 9344 and native_fmt refuses it).
inline constexpr size_t kNativeActBytes = 8192;
/// Bytes of the largest quantized down activation.  qwen4exp: 640 values as Q8_0/Q8_K (1024);
/// glm5-next: 2048 values as Q8_K (8 x 260 = 2084).
inline constexpr size_t kNativeHBytes = 4096;

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
    /// The up rows' type and row bytes: gu_type / gu_row except on a mixed layer (native_fmt_mixed: unsloth's
    /// Qwen3.8 UD-IQ1_S keeps blk.18's gate IQ2_S, its up IQ2_XXS).  The up rows then take down_off - up_off bytes.
    int u_type = -1;
    size_t u_row = 0;
    bool mixed() const { return u_type >= 0 && u_type != gu_type; }
};

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);
/// The same for a layer whose gate and up types differ (up_type == gate_type: exactly native_fmt).  Both must share
/// one activation format (vec_dot_type), so one quantized activation serves the two; Q2_0 cannot be mixed.
bool native_fmt_mixed(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f,
                      std::string& err);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// Split-base variants for files whose gate/up/down tensors are NOT adjacent (the unsloth
/// GLM-5.3-FF shards interleave other tensors between them).  One token: ggml-cpu's dot per row.
/// Several tokens (a verify window's tokens sharing an expert, a prompt's CPU rows): the
/// multi-token kernels (iq_avx512 / iq_avx2: IQ2_*, IQ3_*, IQ4_XS, Q4_K, Q5_K) when they give
/// ggml's results bit for bit - checked once per type (native_multi_rows_exact) - so a window
/// computes exactly what its tokens one at a time would; MAYA_GLM_CPU_MT=0 keeps ggml's
/// dots.  swiglu_limit is GLM's swiglu clamp (gate clamped above, up both sides; +infinity =
/// qwen4exp's plain path) - the same semantics as the device grouped path, so the two agree.
void native_gu_rows_split(const NativeFmt& f, const uint8_t* gate_base, const uint8_t* up_base,
                          const void* const* act, int nt, float* const* ff, int r0, int r1,
                          float swiglu_limit = std::numeric_limits<float>::infinity());
void native_down_rows_split(const NativeFmt& f, const uint8_t* down_base, const void* const* hq,
                            int nt, float* const* out, int r0, int r1);
/// Whether the split variants take the multi-token kernels for this weight type here (the CPU has
/// them, they are not switched off, and they matched ggml-cpu's dot bit for bit on synthetic rows).
bool native_multi_rows_exact(int ggml_type);
/// The same for the one-token rows (iq256_rows1 / iq512_one_rows; false: ggml-cpu's dot).
bool native_rows1_exact(int ggml_type);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

/// out[r - r0] = row r of one tensor of ggml type `type` (rows of `n` values, `row_bytes` each, `base` at the first)
/// against the quantized activation `act` of traits(type)->vec_dot_type (native_quant_act / native_quant_h) - ggml-
/// cpu's own dot, so the caller takes the same sums llama.cpp's CPU backend does for that row, and gets gate and up
/// apart (its own activation between them, not native_gu_rows' silu).
void native_dot_rows(int type, size_t row_bytes, const uint8_t* base, const void* act, int n, int r0, int r1,
                     float* out);
/// The dequantized row (`out` [n] floats) and the dequantized activation, ggml's own wideners - the reference a
/// caller checks its own scalar decode and dots against.
void native_to_float(int type, const void* blocks, float* out, int n);

/// The row dots against a host scalar dequant, a type: rows of the type quantized from known values, the activation
/// too, then vec_dot against the same two widened and summed in double.  One line a type into `out` (rel error);
/// false when one is off by more than 1e-3 - a type whose kernel and scalar disagree cannot be trusted on the lane.
bool native_selftest(const std::vector<int>& types, std::string& out);

// ---- the micro-bench and the hand-tuned rows
/// ggml's name of the type ("q3_K", ...).
const char* native_type_name(int type);
/// Bytes the type's activation takes for 256 values (the vec_dot_type's row).
size_t native_act_bytes_for(int type);
/// 256 values quantized the way the type's dot wants them (from_float of vec_dot_type).
void native_quant_act_test(int type, const float* x, void* dst);
/// A synthetic row of the type (8 x 256 values) and its dequantization - the same bytes for the selftest and the
/// bench; false when the type has no quantizer to make one.
bool native_make_test_row(int type, std::vector<uint8_t>& row, std::vector<float>& ref, size_t* row_bytes = nullptr);
/// The rows through a hand-tuned kernel when this build has one for the type and `hand` is set, else ggml-cpu's.
void native_dot_rows_ex(int type, size_t row_bytes, const uint8_t* base, const void* act, int n, int r0, int r1,
                        float* out, bool hand);
/// Whether this PC runs the hand-tuned AVX2 rows at all (AVX2 + FMA).
bool native_hand_ok();

}  // namespace strata::kernels::cpu
