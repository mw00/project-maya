// Quantized DFlash2 selector entries, shared with the GPU/ggml parity test.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace strata::core::dflash_detail {

struct BlkQ5K {
    __half2 dm; // d, dmin
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
static_assert(sizeof(BlkQ5K) == 176, "ggml Q5_K block layout");

__device__ __forceinline__ float q5_k_at(const uint8_t* tab, int rank, int row, int e) {
    const BlkQ5K* q = (const BlkQ5K*) tab + (size_t) row * (rank / 256) + e / 256;
    const int r = e % 256, j = r / 64, l = r % 32, hi = (r % 64) >= 32, is = 2 * j + hi;
    uint8_t sc, mn;
    if (is < 4) {
        sc = q->scales[is] & 63;
        mn = q->scales[is + 4] & 63;
    } else {
        sc = (q->scales[is + 4] & 0xF) | ((q->scales[is - 4] >> 6) << 4);
        mn = (q->scales[is + 4] >> 4) | ((q->scales[is] >> 6) << 4);
    }
    const uint8_t b = q->qs[32 * j + l];
    const int v = (hi ? b >> 4 : b & 0xF) | (((q->qh[l] >> is) & 1) << 4);
    return __low2float(q->dm) * (float) sc * (float) v - __high2float(q->dm) * (float) mn;
}

} // namespace strata::core::dflash_detail
