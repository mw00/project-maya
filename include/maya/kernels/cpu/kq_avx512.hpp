// include/maya/kernels/cpu/kq_avx512.hpp - Q3_K / Q2_K weight rows against MANY Q8_K activations (AVX-512BW).
//
// ggml's vec_dot unpacks a weight row's 2-3 bit codes again for every activation it dots; a prompt's CPU experts
// dot each row with all the tokens routed to the expert, so here each row is unpacked ONCE into bytes and per-16
// int16 scales, and the tokens - four at a time - run vpmaddubsw / vpmaddwd over it.  The integer arithmetic is
// ggml's (Q3_K: (q + 4h - 4) * scale, the -4 through the activation's 16-value sums; Q2_K: q * scale - min * sums),
// so the results match ggml_vec_dot_q{2,3}_K_q8_K up to float summation order.
#pragma once

#include <cstddef>
#include <cstdint>

namespace maya::kernels::cpu {

/// Whether this CPU and build run kq_rows (AVX-512F + BW).
bool kq_avx512_ok() noexcept;
/// Whether kq_rows takes this ggml weight type (Q2_K = 10, Q3_K = 11).
bool kq_type_ok(int ggml_type) noexcept;
/// The activations of a call, packed once (64-byte aligned quants, then the blocks' scales and 16-value sums): the
/// bytes for nt rows of n values, and the packing of act[0..nt) (Q8_K rows) into dst (64-byte aligned).
size_t kq_act_bytes(int nt, int n);
void kq_pack_act(const void* const* act, int nt, int n, void* dst);
/// out[t][r] = row r of w (`n` values a row, rows `row_bytes` apart, type Q2_K or Q3_K) . activation t of `packed`,
/// for r in [r0, r1) and t in [0, nt).  n a multiple of 256.  Faster than ggml's per-pair dots from ~8 rows of
/// activations (one 2-socket Xeon 6152, 2048 rows of 4096: Q3_K 2.1-2.4x, Q2_K 1.3-2.1x at 8-128); slower for one.
void kq_rows_packed(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* packed, int nt,
                    float* const* out, int r0, int r1);
/// The same from unpacked Q8_K rows (packs into a per-thread buffer first).
void kq_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
             float* const* out, int r0, int r1);

}  // namespace maya::kernels::cpu
