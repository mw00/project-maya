#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::rdna4 {

// Fixed GLM native expert geometry: E=4096, FF=2048, separate gate/up rows.
struct Geometry {
    int gu_type = -1, d_type = -1;
    size_t gu_row = 0, d_row = 0, up_off = 0, down_off = 0;
    float limit = 0;
    int n_embd = 0, n_ff = 0;   // 0: GLM's 4096 / 2048; DeepSeek-V4.1: 5120 / 2304
    size_t gate_off = 0;        // the gate's rows from the expert's start (GLM: 0)
    size_t h_ld = 0;            // the bytes of an H row (gate / up's output, down's input; 0: packed, act_bytes(1, n_ff))
};

// Default on only on validated gfx1151 and gfx1100 with compiled WMMA bodies and covered types.
// STRATA_GLM_PREFILL_FUSED=0 retains MMQ. Other devices always retain MMQ.
bool available();
bool supported(int gu_type, int down_type);
size_t act_bytes(int64_t rows, int64_t cols);
void quantize(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

// Bounds may start at a nonzero offset. src, ha and out start at this set's row 0.
// xa is the whole layer's quantized token input; src indexes its token rows.
// Weights remain in the existing resident/prestaged/ring slots (stride bytes).
void experts(const uint8_t* wbase, size_t stride, const Geometry& geometry,
             int n_expert, const int* bounds, const int* host_bounds, void* tiles, const void* xa,
             const int* src, void* ha, float* out, void* stream);
// the two passes apart (the caller's own buffers between them; src nullptr: xa holds the rows in their order)
void experts_gate_up(const uint8_t* wbase, size_t stride, const Geometry& geometry, int n_expert, const int* bounds,
                     const int* host_bounds, void* tiles, const void* xa, const int* src, void* ha, void* stream);
void experts_down(const uint8_t* wbase, size_t stride, const Geometry& geometry, int n_expert, const int* bounds,
                  const int* host_bounds, void* tiles, const void* ha, float* out, void* stream);
// quantize's int8 rows of the rows idx[r] of x (ldx floats apart), packed in that order
void quantize_rows(const float* x, int64_t ldx, const int* idx, int64_t rows, int64_t cols, void* xa, void* stream);

} // namespace strata::prefill::rdna4
