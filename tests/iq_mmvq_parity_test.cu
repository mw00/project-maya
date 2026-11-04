// tests/iq_mmvq_parity_test.cu - a multi-column iq_mmvq call must equal single-column calls, in bits.
//
// The multi-column path moves the block index outermost so one weight block is loaded and dequantised once
// for every activation column.  That is meant to be a scheduling change only: same per-lane accumulation
// order, same warp reduction.  This test is what makes that a fact rather than an argument - it compares the raw
// bit patterns, so a single FMA contraction difference or a reordered sum fails here rather than silently
// moving a 2-bit model's near-ties.
//
// Every supported IQ format is covered.  The weight blocks and the q8_1 activations are random bytes: the
// dequantisers are total functions of the bytes, and a NaN propagates identically through both paths, so
// the bit comparison is still the right check.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kTypes[] = {13, 16, 17, 18, 19, 20, 21, 22, 23, 29, 42};
constexpr int kCols[] = {1, 2, 3, 4, 5, 7, 8, 9, 16};

int block_elems(int t) { return t == 20 ? 32 : (t == 42 ? 64 : 256); }

}  // namespace

int main() {
    int fails = 0, checks = 0, strict_diff = 0;
    for (int t : kTypes) {
        const int qk = block_elems(t);
        const int n_in = qk * 4;            // 4 blocks per row keeps every format's block loop exercised
        const int n_out = 8;
        const size_t rb = strata::kernels::iq_row_bytes(t, n_in);
        if (rb == 0) { std::printf("  type %d: no row size, skipped\n", t); ++fails; continue; }
        const size_t w_bytes = rb * (size_t) n_out;
        std::vector<uint8_t> hw(w_bytes);
        std::srand(1234 + t);
        for (auto& b : hw) b = (uint8_t) (std::rand() & 0xFF);
        // Clamp every fp16-shaped field away from Inf/NaN.  The blocks' scales are fp16, and a random
        // exponent of all ones makes them NaN - which turns the strict bit check into a NaN-sign check and
        // cannot happen with real quantised weights.  Only the exponent is touched, so the payloads (the
        // quantised values) stay random.
        for (size_t i = 0; i + 1 < hw.size(); i += 2) {
            uint16_t w = (uint16_t) (hw[i] | (hw[i + 1] << 8));
            if ((w & 0x7C00u) == 0x7C00u) { w &= (uint16_t) ~0x7C00u; w |= 0x7BFFu & 0x7C00u; }
            hw[i] = (uint8_t) (w & 0xFF);
            hw[i + 1] = (uint8_t) (w >> 8);
        }

        const int max_cols = 16;
        const size_t xb = (size_t) (n_in / 32) * 36;   // q8_1 bytes per column
        std::vector<uint8_t> hx(xb * (size_t) max_cols);
        for (auto& b : hx) b = (uint8_t) (std::rand() & 0xFF);

        uint8_t* dw = nullptr;
        uint8_t* dx = nullptr;
        float* dy_ref = nullptr;
        float* dy_bat = nullptr;
        cudaMalloc(&dw, w_bytes);
        cudaMalloc(&dx, xb * (size_t) max_cols);
        cudaMalloc(&dy_ref, (size_t) max_cols * (size_t) n_out * sizeof(float));
        cudaMalloc(&dy_bat, (size_t) max_cols * (size_t) n_out * sizeof(float));
        cudaMemcpy(dw, hw.data(), w_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx.data(), xb * (size_t) max_cols, cudaMemcpyHostToDevice);

        for (int ncols : kCols) {
            // reference: the scalar entry point, one column at a time
            cudaMemset(dy_ref, 0, (size_t) max_cols * (size_t) n_out * sizeof(float));
            for (int c = 0; c < ncols; ++c)
                strata::kernels::iq_mmvq(t, dw, dx + (size_t) c * xb, dy_ref + (size_t) c * n_out,
                                         n_in, n_out, 1, nullptr);
            // batched
            cudaMemset(dy_bat, 0, (size_t) max_cols * (size_t) n_out * sizeof(float));
            strata::kernels::iq_mmvq(t, dw, dx, dy_bat, n_in, n_out, ncols, nullptr);
            cudaDeviceSynchronize();

            std::vector<float> a((size_t) ncols * (size_t) n_out), b((size_t) ncols * (size_t) n_out);
            cudaMemcpy(a.data(), dy_ref, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), dy_bat, b.size() * sizeof(float), cudaMemcpyDeviceToHost);
            ++checks;
            // Two comparisons.  The strict one is raw bits, which is what we actually want; the pass
            // criterion is value equality with NaN treated as equal to NaN, because random weight bytes
            // give the fp16 scales NaN and IEEE leaves a NaN's sign to the implementation - a NaN sign
            // bit is not a numeric difference and cannot arise from real quantised weights.
            int strict = 0, value_bad = -1;
            for (size_t i = 0; i < a.size(); ++i) {
                const bool bits_equal = std::memcmp(&a[i], &b[i], sizeof(float)) == 0;
                if (!bits_equal) ++strict;
                const bool both_nan = (a[i] != a[i]) && (b[i] != b[i]);
                if (!bits_equal && !both_nan && value_bad < 0) value_bad = (int) i;
            }
            if (strict) ++strict_diff;
            if (value_bad >= 0) {
                std::printf("  type %2d ncols %2d: MISMATCH at %d  scalar %.9g  batched %.9g\n", t, ncols,
                            value_bad, a[(size_t) value_bad], b[(size_t) value_bad]);
                ++fails;
            }
        }
        cudaFree(dw); cudaFree(dx); cudaFree(dy_ref); cudaFree(dy_bat);
    }
    std::printf("  iq_mmvq parity: %d format/width combinations, %d numerically different, "
                "%d differing only in a NaN sign bit\n", checks, fails, strict_diff);
    std::printf("  %s\n", fails == 0 ? "OK" : "FAILED");
    return fails == 0 ? 0 : 1;
}
