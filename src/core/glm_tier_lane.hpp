// src/core/glm_tier_lane.hpp - the glm-dsa port B8.4: the node-split CPU lane's arithmetic on one socket's half.
//
// A split RAM-tier expert (glm_tier_plan.hpp SplitLayout) is computed by two sockets at once: socket h dots the gate
// and up rows [h*H, (h+1)*H) of every expert in the call, quantizes its h half (n_ff / 2 values; the down activation's
// blocks never straddle the halves, so the bytes equal the matching part of the whole h's quant), dots each down row's
// inner columns [h*H, (h+1)*H) against it, and weights the experts into a partial y_h.  The device adds y_0 + y_1
// (one n_embd add per layer, glmtier::add_partial).  Gate/up rows are bit-identical to the unsplit lane; down rows
// differ by the reassociation of one float sum (two partial dots added).
//
// Plain C++ over ggml-cpu (strata_kernels_cpu); the threads, pools and NUMA placement live in glm_tier_numa.cu.
#pragma once

#include "glm_tier_plan.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <limits>
#include <string>

namespace strata::core::glmtier {

namespace kc = strata::kernels::cpu;

/// The half format: the full layer's types at n_ff / 2 (same gate/up rows; down rows of hd_row bytes; h of H values).
bool half_fmt(const kc::NativeFmt& full, kc::NativeFmt& half, std::string& why);

/// ff[r - 0] for half rows r in [r0, r1) of [0, H): silu(clamp(gate . a)) * clamp(up . a), from the half blob.
void half_gu_rows(const kc::NativeFmt& half, const SplitLayout& L, const uint8_t* half_blob, const void* act,
                  float* ff_half, int r0, int r1, float swiglu_limit = std::numeric_limits<float>::infinity());
/// y[r] for down rows r in [r0, r1) of [0, n_embd): the half's inner columns against hq_half.
void half_down_rows(const kc::NativeFmt& half, const SplitLayout& L, const uint8_t* half_blob, const void* hq_half,
                    float* y, int r0, int r1);

/// One socket's whole share of a call, single-threaded (the reference the pooled lane and the tests use):
/// y_h[r] = sum_i w[i] * down_half_i(quant(h_half_i))[r].  scratch: ff (H floats), hq (half.h_bytes), dn (n_embd).
void half_experts_serial(const kc::NativeFmt& half, const SplitLayout& L, int ne, const uint8_t* const* half_blob,
                         const float* w, const void* act, float swiglu_limit, float* y_h, float* ff, void* hq,
                         float* dn);

}  // namespace strata::core::glmtier
