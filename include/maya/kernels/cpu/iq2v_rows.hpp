// include/maya/kernels/cpu/iq2v_rows.hpp - the CPU lane's second-generation IQ2_XXS / IQ2_S rows (iq2v_rows.cpp):
// ggml-cpu's vec_dot to the bit, each form for the CPUs that run it (the caller checks: src/ds4/ds4_iq_cpu.cpp).
#pragma once

#include <cstddef>
#include <cstdint>

namespace maya::kernels::cpu::iq2v {

/// out[t][r - r0] = (row r of w) . act[t] for rows [r0, r1) and t < nt: w's rows hold n values (n % 256 == 0) in rb
/// bytes, act[t] n values as Q8_K.  v: 1 AVX2, 2 AVX2 + AVX-VNNI, 3 AVX-512 F BW VL DQ VBMI BITALG VNNI - for IQ2_XXS;
/// 11-13 the same for IQ2_S.
void rows(int v, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1);
/// whether this build has the AVX-VNNI forms (2, 12: the compiler's intrinsics)
bool vnni_form();

}  // namespace maya::kernels::cpu::iq2v
