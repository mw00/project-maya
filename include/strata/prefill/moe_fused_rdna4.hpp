#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::rdna4 {

// Fixed GLM native expert geometry: E=4096, FF=2048, separate gate/up rows.
struct Geometry {
    int gu_type = -1, d_type = -1;
    size_t gu_row = 0, d_row = 0, up_off = 0, down_off = 0;
    float limit = 0;
};

// Default on only on validated gfx1151 with compiled WMMA bodies and covered types.
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

} // namespace strata::prefill::rdna4
