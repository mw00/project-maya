// src/ling/ling_parity.hpp - GPU-identical CPU expert lane (Q8_K per-256 activation rounding and ds4_iq warp-lane
// reduction order) plus --bench-lane ggml verification across all shipped GGUF expert quant formats.
#pragma once

#include "hy4_lane.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ling::par {

/// True when `type` has bit-exact GPU-parity CPU row arithmetic (IQ2_XXS/IQ2_S/IQ3_XXS/IQ3_S/MXFP4 via ds4::par,
/// plus Q4_K/Q5_K/Q6_K/Q2_K/Q3_K/Q8_0/IQ4_NL/IQ4_XS matching ds4_iq.cu lane reduction).
bool supported(int type);

/// Exact Q8_K per-256 activation rounding (same bits as GPU q8k_rows / maya::ds4::par::quant).
inline size_t act_bytes(int k) { return (size_t) k + (size_t) (k / 32) * 4; }
void quant_q8k(const float* x, int k, void* act);

/// Run CPU-lane experts in exact GPU lane-parity arithmetic (Q8_K per-256 + ds4_iq warp-lane reduction + swiglu_portable).
/// When `blob_ptrs` is non-null, blob_ptrs[e] points directly to packed expert e (gate at tr[0], up at tr[1], down at tr[2]).
void cpu_experts_parity(hy4::Lane& lane, int tg, int td, int E, int F, float limit,
                        const std::vector<hy4::CpuEntry>& entries, const uint8_t* const* gate_ptrs,
                        const uint8_t* const* up_ptrs, const uint8_t* const* down_ptrs, const float* x, int T,
                        float* out);

/// Verify native CPU row kernels vs ggml-cpu across all shipped formats (~n_calls random calls per format).
/// Prints SHA-256 / FNV-1a hashes and max diff; returns true if every format matches bit-for-bit or falls back cleanly.
bool bench_lane_verify(int n_calls = 300);

/// Verify a single expert format against ggml-cpu at first use (cached per type).
/// Returns true if native AVX2 rows match ggml bit-for-bit; otherwise disables native override for that type (ggml fallback).
bool verify_format_once(int type, std::string& msg);

}  // namespace ling::par
