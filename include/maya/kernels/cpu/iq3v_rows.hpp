// include/maya/kernels/cpu/iq3v_rows.hpp - the CPU lane's second-generation IQ3_XXS and IQ4_XS rows and the rows of
// its 4-bit IQ2_XXS copies (Q4X, src/ds4/ds4_iq_cpu.cpp) (iq3v_rows.cpp): ggml-cpu's vec_dot to the bit, each form for
// the CPUs that run it (the caller checks each against ggml-cpu's dot and times it: src/ds4/ds4_iq_cpu.cpp own()).
#pragma once

#include <cstddef>
#include <cstdint>

namespace maya::kernels::cpu::iq3v {

/// out[t][r - r0] = (row r of w) . act[t] for rows [r0, r1) and t < nt: w's rows hold n values (n % 256 == 0) in rb
/// bytes, act[t] n values as Q8_K.
/// IQ3_XXS, v: 1 AVX2 (grid by vpgatherdd), 2 the same + AVX-VNNI, 3 AVX2 (grid by loads + blends), 4 the same +
/// AVX-VNNI, 5 AVX-512 F BW VL DQ VBMI BITALG VNNI (grid codes in registers).
void rows_iq3xxs(int v, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0,
                 int r1);
/// IQ4_XS, v: 1 AVX2, 2 AVX2 + AVX-VNNI, 3 AVX-512 F BW VL DQ VBMI BITALG VNNI.
void rows_iq4xs(int v, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0,
                int r1);
/// the Q4X form's rows (rb: its row bytes, ds4_iq_cpu.cpp q4x_row_bytes), v as IQ4_XS's; 0.125 x ggml's IQ2_XXS sums.
void rows_q4x(int v, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0,
              int r1);
/// whether this build has the AVX-VNNI forms (the compiler's intrinsics)
bool vnni_form();

}  // namespace maya::kernels::cpu::iq3v
