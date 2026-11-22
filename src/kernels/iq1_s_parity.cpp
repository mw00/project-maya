// src/kernels/iq1_s_parity.cpp - the IQ1_S device kernels against ggml's own interpretation.
//
//   build/iq1_s_parity
//
// Synthesized-but-valid blocks (random f16 scale, random grid indices and qh bits - every bit
// pattern the tables accept) dequantized by BOTH our iq_dequant_f32 and ggml's trait to_float must
// agree to fp32 rounding; the device MMVQ dot through the production dispatcher (native_mmvq,
// type 19, q8_1 activations) must track an f32 matvec of the dequanted weights within the
// activation rounding.  Byte accounting (native_mmvq_weight_bytes) is asserted too.
#include "maya/kernels/iq_kernels.hpp"
#include "maya/kernels/native_mmvq.hpp"
#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    constexpr int kRows = 8, kCols = 4096, kBlocks = kCols / 256, kBlockBytes = 50;
    int failures = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);

    // 1. the byte accounting
    const std::size_t expect_bytes = (std::size_t) kRows * kBlocks * kBlockBytes;
    if (maya::kernels::native_mmvq_weight_bytes(19, kCols, kRows) != expect_bytes) {
        std::printf("FAIL weight_bytes(19, %d, %d) = %zu, expected %zu\n", kCols, kRows,
                    maya::kernels::native_mmvq_weight_bytes(19, kCols, kRows), expect_bytes);
        ++failures;
    }

    // 2. synthesized-but-valid blocks: f16 scale in +/-[1/256, 1/4], qs/qh fully random (every
    //    bit pattern the tables accept is a valid IQ1_S block; the grid index is 11 bits)
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> raw(expect_bytes);
    for (int r = 0; r < kRows; ++r)
        for (int b = 0; b < kBlocks; ++b) {
            uint8_t* blk = raw.data() + ((size_t) r * kBlocks + b) * kBlockBytes;
            float d = 1.f / 256.f + ud(rng) * 0.25f;
            if ((r ^ b) & 1) d = -d;
            const uint16_t dh = ggml_fp32_to_fp16(d);
            blk[0] = (uint8_t) dh;
            blk[1] = (uint8_t) (dh >> 8);
            for (int i = 2; i < kBlockBytes; ++i) blk[i] = (uint8_t) byte(rng);
        }

    // 3. dequant parity vs ggml's trait table
    std::vector<float> ref((size_t) kRows * kCols);
    const ggml_type_traits* tt = ggml_get_type_traits(GGML_TYPE_IQ1_S);
    if (!tt || !tt->to_float) {
        std::printf("FAIL ggml has no IQ1_S to_float\n");
        return 1;
    }
    tt->to_float(raw.data(), ref.data(), (int64_t) kRows * kCols);

    uint8_t* dw = nullptr;
    float* dq = nullptr;
    if (cudaMalloc(&dw, raw.size()) || cudaMemcpy(dw, raw.data(), raw.size(), cudaMemcpyHostToDevice) ||
        cudaMalloc(&dq, ref.size() * 4)) {
        std::printf("FAIL cuda alloc\n");
        return 1;
    }
    maya::kernels::iq_dequant_f32(19, dw, (int64_t) kRows * kCols, dq, s);
    std::vector<float> got(ref.size());
    cudaMemcpy(got.data(), dq, got.size() * 4, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) { num += std::fabs(got[i] - ref[i]); den += std::fabs(ref[i]); }
    const double dq_rel = num / (den + 1e-30);
    std::printf("dequant rel err = %.3e\n", dq_rel);
    if (dq_rel > 1e-6) { std::printf("FAIL dequant parity\n"); ++failures; }

    // 4. the dot through the production dispatcher, two columns, vs an f32 matvec of the dequanted
    //    weights (the tolerance is the q8_1 activation rounding)
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) 2 * kCols);
    for (auto& v : x) v = nd(rng);
    float* dx = nullptr;
    void* xq = nullptr;
    float* dy = nullptr;
    cudaMalloc(&dx, x.size() * 4);
    cudaMalloc(&xq, (size_t) 2 * (kCols / 32) * 36);
    cudaMalloc(&dy, (size_t) 2 * kRows * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    maya::kernels::quantize_q8_1_rows(dx, 2, kCols, xq, s);
    maya::kernels::native_mmvq(19, dw, xq, dy, kCols, kRows, 2, s);
    cudaStreamSynchronize(s);
    std::vector<float> y((size_t) 2 * kRows);
    cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);

    // per-row values are printed for inspection; the ASSERTION is the aggregate relative error the
    // other i-quant harnesses use - a near-cancelling row's dot has no absolute headroom for the
    // q8_1 activation rounding (about 0.5 here: sqrt(4096) * (max|x|/127)/sqrt(12) at |w| ~ 1)
    double num2 = 0, den2 = 0;
    for (int c = 0; c < 2; ++c)
        for (int r = 0; r < kRows; ++r) {
            float ref_dot = 0.f;
            for (int k = 0; k < kCols; ++k) ref_dot += ref[(size_t) r * kCols + k] * x[(size_t) c * kCols + k];
            std::printf("dot col %d row %d: got %+.6f ref %+.6f\n", c, r, y[(size_t) c * kRows + r], ref_dot);
            num2 += std::fabs(y[(size_t) c * kRows + r] - ref_dot);
            den2 += std::fabs(ref_dot);
        }
    const double mm_rel = num2 / (den2 + 1e-30);
    std::printf("mmvq aggregate rel err = %.3e\n", mm_rel);
    if (mm_rel > 2e-2) ++failures;

    std::printf(failures ? "iq1_s_parity: FAIL (%d)\n" : "iq1_s_parity: PASS\n", failures);
    return failures ? 1 : 0;
}
