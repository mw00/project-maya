// include/strata/prefill/moe_tc_kq.hpp - a prompt's routed experts on Volta's FP16 tensor cores for the K-quant formats
// (Q4_K / Q5_K / Q6_K: the Qwen3.6-35B-A3B experts), the register-fragment design of the Strata V100 tree's
// moe_fp16tc (mma.m8n8k4, the weights decoded straight into A fragments).
//
// dst[r][c] = sum_k W_e[c][k] x[r][k] for the rows r in [bounds[e], bounds[e + 1]) of each expert e: W the experts'
// matrices expert_bytes apart (n_out rows of n_in values, GGUF blocks), x the rows as mmq::quantize writes them for that
// type ([n_in / 128][xq_rows] block_q8_1_mmq: DS4 for Q4_K / Q5_K, D4 for Q6_K), dst rows of ld floats.  FP16 products,
// FP32 sums - not bit-identical to MMQ.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::tckq {

#if defined(STRATA_PREFILL_TCKQ)
bool supported(int type, int64_t n_in, int64_t n_out);
/// the device is sm_70 (on sm_75+ MMQ runs int8 tensor cores already)
bool available();
void product(int type, const void* W, size_t expert_bytes, int64_t n_in, int64_t n_out, int n_experts, const int32_t* bounds,
             int max_rows, const void* xq, int64_t xq_rows, float* dst, int64_t ld, void* stream);
#else   // (not in this build: no-ops, so the caller needs no #ifdef)
inline bool supported(int, int64_t, int64_t) { return false; }
inline bool available() { return false; }
inline void product(int, const void*, size_t, int64_t, int64_t, int, const int32_t*, int, const void*, int64_t, float*, int64_t,
                    void*) {}
#endif

}  // namespace strata::prefill::tckq
