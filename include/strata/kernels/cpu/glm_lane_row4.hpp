// include/strata/kernels/cpu/glm_lane_row4.hpp - the glm-dsa port B14.9: lockstep 4-row CPU decode GEMV for the
// CPU expert lane over UNCHANGED GGUF rows (ik_llama.cpp's R4 idea without the repack).
//
// One Q8_K activation chunk (32 values) is loaded once and multiplied into four weight rows; the four rows keep
// their own accumulators and are reduced together at the end (one 4x4 transpose).  AVX2 + FMA only: no VNNI, no
// VBMI, so the Xeon Gold 6138 runs it.
//
// Bit-exactness contract: for every supported type the per-row result equals ggml-cpu's AVX2 vec_dot (the one the
// lane's nt = 1 path calls) bit for bit.  Each row reproduces ggml's int32 lane sums exactly (same maddubs / madd
// operands per 16-bit lane, so the same saturation and the same lane assignment; int32 adds are exact modulo 2^32,
// so their order is free), the same per-block float fmadd chain per lane, the same horizontal reduction order and
// the same final scale.  Supported only when ggml-cpu itself was compiled with AVX2 (ggml_cpu_has_avx2()), since
// that is the path being matched; anything else reports unsupported and the caller keeps ggml's vec_dot.
//
// Types: Q2_K (10), Q3_K (11), IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ1_S (19), IQ2_S (22), IQ4_XS (23); all
// against Q8_K.  IQ1_S's scalar delta term is matched in whichever form ggml-cpu's compiler gave it (one fmaf or a
// multiply and an add), found once by a probe against ggml's vec_dot; neither form matching leaves IQ1_S unsupported.
//
// Builds: every native-experts build has these functions (src/kernels/cpu/glm_lane_row4.cpp, compiled for any CPU of
// its architecture: on x86-64 the kernels, glm_lane_row4_avx2.cpp, run only after its CPUID check for AVX2 + FMA +
// F16C).  The kernels are x86-64 only (the CMake option STRATA_GLM_LANE_ROW4_KERNELS, ON by default).  Without them -
// aarch64 (DGX Spark / Grace), ppc64le (IBM AC922), or the option OFF - row4_supported() is false for every type,
// row4_iq1s_form() is -1, lane_row4_on() is false whatever MAYA_GLM_LANE_ROW4 says, lane_row4_types() is 0 and both
// hooks return false (row4_dot / row4_rows, which require row4_supported(), abort).
#pragma once

#include "strata/kernels/cpu/native_expert.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// True when `ggml_type` has a row4 kernel AND this CPU / this ggml-cpu build can run it bit-exactly.
bool row4_supported(int ggml_type);

/// IQ1_S only, for tests and logs: the form of ggml's delta term (accum1 += d * sumi1) the probe found in this
/// ggml-cpu: 2 = one fmaf (ggml's gnu11 C with FMA on, GCC's default contraction), 1 = a multiply and an add,
/// 0 = neither matched (row4_supported(19) is false), -1 = not probed (no AVX2 path here).
int row4_iq1s_form();

/// out[k] = row k . q8 for the four rows w[0..3] (n values each, n % 256 == 0), q8 one Q8_K activation of n values.
/// Bit-identical to ggml-cpu's vec_dot for each row.  Requires row4_supported(type).
void row4_dot(int ggml_type, int n, const uint8_t* const w[4], const void* q8, float out[4]);

/// out[r] = row r . q8 for r in [r0, r1), row r at w + r * row_bytes: groups of four through row4_dot, the
/// (r1 - r0) % 4 rows left over through ggml-cpu's vec_dot.  Requires row4_supported(type).
void row4_rows(int ggml_type, int n, const uint8_t* w, size_t row_bytes, const void* q8, float* out, int r0, int r1);

/// MAYA_GLM_LANE_ROW4=0 turns the lane hooks off (default on).  lane_row4_override(-1) = follow the env,
/// 0 = off, 1 = on (tests and the bench compare both in one process); it leaves lane_row4_set's type mask as it is.
bool lane_row4_on();
void lane_row4_override(int mode) noexcept;

/// lane_row4_set's type mask: bit t lets the hooks take ggml type t (every row4 type is below 32).
inline constexpr uint32_t kRow4AllTypes = 0xFFFFFFFFu;
inline constexpr uint32_t row4_type_bit(int ggml_type) { return 1u << ggml_type; }

/// Run-time switch for the engine's per-request knobs (an A/B without a restart; set it between requests: relaxed
/// atomics, and every setting is bit-identical anyway): mode -1 = back to MAYA_GLM_LANE_ROW4, 0 = off, 1 = on, and
/// `types` = the ggml types the hooks may take (row4_type_bit; a type without a kernel stays on ggml's loop).
/// lane_row4_set(-1) restores the start state; lane_row4_set(1, kRow4AllTypes & ~(row4_type_bit(19) |
/// row4_type_bit(23))) is row4 without the IQ1_S / IQ4_XS kernels.  lane_row4_types() reads the mask back.
void lane_row4_set(int mode, uint32_t types = kRow4AllTypes) noexcept;
uint32_t lane_row4_types() noexcept;

/// Hooks for native_gu_rows_split / native_down_rows_split at nt = 1: when the switch is on, the type is in the
/// mask and both the type and the CPU are supported they compute rows [r0, r1) exactly as the ggml path would and
/// return true; otherwise they return false and the caller runs its own loop.
bool lane_row4_gu(const NativeFmt& f, const uint8_t* gate_base, const uint8_t* up_base, const void* act, float* ff,
                  int r0, int r1, float swiglu_limit);
bool lane_row4_down(const NativeFmt& f, const uint8_t* down_base, const void* hq, float* out, int r0, int r1);

}  // namespace strata::kernels::cpu
