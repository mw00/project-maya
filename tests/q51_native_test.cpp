// tests/q51_native_test.cpp - GGML Q5_1 (type 7) on the native MMVQ path.
//
// WHY: some community GGUF files store Q5_1 tensors.  Before this item
// native_mmvq_supported(7) and dequant_bf16_supported(7) were false, so every Q5_1 tensor fell
// off the native MMVQ path onto the packed-BF16 fallback .  This test pins three things:
//
//   A  the capability queries answer true for type 7 and the byte math says 24 B per 32 elements
//      (this part compiles BEFORE the change, so it is the RED leg);
//   B  the new Q5_1 dequant branch reproduces the q51 reader semantics that PATCH 3 proved
//      bit-identical against the real shard: unsigned 5-bit codes, additive min, x = q*d + m;
//   C  the new Q5_1 MMVQ dot equals the closed form of that same dequant against a Q8_1
//      activation, with the min term split across the two threads that share one block.
//
// CPU only - there is no GPU on this box.  The two CUDA translation units are compiled through
// tests/cpu_cuda_shim with their launch configs stripped (see tests/q51_native_cpu_test.sh) and the
// device helpers are called directly instead of launched.  Parts B and C name symbols that do not
// exist before the change, so the RED leg runs with --gate-only and the GREEN leg runs everything.
//
// Run: q51_native_test [--gate-only]      exit 0 = all checks pass, 1 = at least one failed.

#include "cuda_runtime.h"   // the CPU shim first: dp4a.hpp wants the runtime header ahead of it
#include "cuda_fp16.h"

#include "native_mmvq_cpu.cu"    // generated: src/kernels/cuda/native_mmvq.cu, launch configs stripped
#include "dequant_bf16_cpu.cu"   // generated: src/kernels/cuda/dequant_bf16.cu, launch configs stripped

#include "strata/artifact/dequant.hpp"   // the CPU reference the GPU dequant test compares against

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void report(bool passed, const std::string& label, const std::string& detail) {
    ++g_checks;
    if (!passed) ++g_failures;
    std::printf("%s  %s  %s\n", passed ? "ok  " : "FAIL", label.c_str(), detail.c_str());
}

// xorshift64* keeps the fixtures reproducible across runs and compilers.
struct Rng {
    uint64_t state = 0x9E3779B97F4A7C15ull;
    uint64_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1Dull;
    }
    uint8_t byte() { return (uint8_t) (next() >> 33); }
    float signed_unit() {
        const double scaled = (double) (int64_t) next() / (double) INT64_MAX;   // [-1, 1]
        return (float) scaled;
    }
};

float f16_bits_to_float(uint16_t bits) { return __half2float(__ushort_as_half(bits)); }

// Parts B and C name the Q5_1 symbols this item adds (Q51Block, the group32<7> branch, the
// Q5_1 small_q8_dot overload), so they cannot compile against the pre-change tree.  The RED leg
// builds with -DSTRATA_Q51_GATE_ONLY=1 and runs part A only; the GREEN leg drops the guard.
#ifndef STRATA_Q51_GATE_ONLY

// The Q5_1 block contract PATCH 3 verified against the real shard bytes: 24 bytes per 32 elements,
// [f16 d][f16 m][4 x uint8 qh][16 x uint8 qs], codes unsigned in [0,31], min ADDITIVE,
// split-half nibble order (qs[j] low nibble = element j, high nibble = element j+16; the 5th bit
// of those elements is qh bit j and qh bit j+16).  This is ggml's dequantize_row_q5_1.
void reference_q51_block(const uint8_t* block, float* out32) {
    const float d = f16_bits_to_float((uint16_t) (block[0] | (block[1] << 8)));
    const float m = f16_bits_to_float((uint16_t) (block[2] | (block[3] << 8)));
    const uint32_t qh = (uint32_t) block[4] | ((uint32_t) block[5] << 8) |
                        ((uint32_t) block[6] << 16) | ((uint32_t) block[7] << 24);
    const uint8_t* qs = block + 8;
    for (int j = 0; j < 16; ++j) {
        const int low = (qs[j] & 0x0F) | (int) (((qh >> j) & 1u) << 4);
        const int high = (qs[j] >> 4) | (int) (((qh >> (j + 16)) & 1u) << 4);
        out32[j] = (float) low * d + m;
        out32[j + 16] = (float) high * d + m;
    }
}

// Same shape for Q5_0 (22 bytes, no min, offset 16) so the harness proves it can also see the
// difference between the two formats and the new branch did not disturb its neighbour.
void reference_q50_block(const uint8_t* block, float* out32) {
    const float d = f16_bits_to_float((uint16_t) (block[0] | (block[1] << 8)));
    const uint32_t qh = (uint32_t) block[2] | ((uint32_t) block[3] << 8) |
                        ((uint32_t) block[4] << 16) | ((uint32_t) block[5] << 24);
    const uint8_t* qs = block + 6;
    for (int j = 0; j < 16; ++j) {
        const int low = (qs[j] & 0x0F) | (int) (((qh >> j) & 1u) << 4);
        const int high = (qs[j] >> 4) | (int) (((qh >> (j + 16)) & 1u) << 4);
        out32[j] = (float) (low - 16) * d;
        out32[j + 16] = (float) (high - 16) * d;
    }
}

// A Q8_1 activation block built exactly the way native_quantize_q8_1_kernel builds one:
// symmetric scale amax/127 clamped into fp16, codes round(x/d) clamped to [-127,127], and
// ds = (scale, warp sum of the ORIGINAL floats).  The sum of the originals is what the affine
// min/offset correction consumes, so the test has to carry it the same way.
struct Q81Fixture {
    strata::kernels::Q81Block block;
    double d8 = 0.0;
    double original_sum = 0.0;
    int32_t codes[32] = {};
};

Q81Fixture make_q81_fixture(Rng& rng) {
    Q81Fixture fixture;
    float values[32];
    float abs_max = 0.0f;
    double sum = 0.0;
    for (int i = 0; i < 32; ++i) {
        values[i] = rng.signed_unit() * 4.0f;
        abs_max = std::fabs(values[i]) > abs_max ? std::fabs(values[i]) : abs_max;
        sum += (double) values[i];
    }
    const float d8 = (std::fmin)(abs_max / 127.0f, 65504.0f);
    const float clamped_sum = (std::fmax)(-65504.0f, (std::fmin)(65504.0f, sum));
    fixture.block.ds = make_half2(__float2half_rn(d8), __float2half_rn(clamped_sum));
    fixture.d8 = (double) __half2float(__float2half_rn(d8));
    fixture.original_sum = (double) __half2float(__float2half_rn(clamped_sum));
    for (int i = 0; i < 32; ++i) {
        const float raw = d8 == 0.0f ? 0.0f : (std::fmax)(-127.0f, (std::fmin)(127.0f, std::nearbyintf(values[i] / d8)));
        fixture.codes[i] = (int32_t) raw;
        fixture.block.qs[i] = (int8_t) raw;
    }
    return fixture;
}

// A Q5_1 block with random 5-bit codes and non-degenerate scale/min, plus the codes kept for the
// closed-form check.
struct Q51Fixture {
    uint8_t bytes[24] = {};
    double d = 0.0;
    double m = 0.0;
    int32_t codes[32] = {};
};

Q51Fixture make_q51_fixture(Rng& rng) {
    Q51Fixture fixture;
    const uint16_t d_bits = (uint16_t) (0x2C00u | (uint16_t) (rng.next() & 0x03FFu));
    fixture.d = (double) f16_bits_to_float(d_bits);
    // The min is the block minimum of quantize_row_q5_1, so on zero-centred weights it is
    // NEGATIVE and of the same order as the block's own span: m/d = 31*min/(max-min) = -15.5
    // for a block symmetric about zero.  A tiny positive m (the first draft used 0x1400|rand,
    // ~3e-2 against d ~ 1e2) dilutes a sign or scale error in the min term until C1 catches it
    // only narrowly, so draw the realistic magnitude here instead.
    //
    // The band stops below 16*d on purpose: m == -16*d makes a Q5_1 block numerically identical
    // to the Q5_0 block with the same d and codes, which is exactly the copy-paste C2 guards
    // against, so a fixture that lands there would agree by construction, not by bug.
    const double min_over_d = 14.0 + (double) (rng.next() % 200ull) / 100.0;   // 14.00 .. 15.99
    const uint16_t m_bits = __half_as_ushort(__float2half_rn((float) (-min_over_d * fixture.d)));
    fixture.m = (double) f16_bits_to_float(m_bits);
    fixture.bytes[0] = (uint8_t) (d_bits & 0xFF);
    fixture.bytes[1] = (uint8_t) (d_bits >> 8);
    fixture.bytes[2] = (uint8_t) (m_bits & 0xFF);
    fixture.bytes[3] = (uint8_t) (m_bits >> 8);
    uint32_t qh = 0;
    for (int i = 0; i < 32; ++i) {
        const int code = (int) (rng.next() % 32ull);
        fixture.codes[i] = code;
        if (code & 16) qh |= (1u << i);
        const int low_nibble = code & 0x0F;
        if (i < 16) {
            fixture.bytes[8 + i] = (uint8_t) ((fixture.bytes[8 + i] & 0xF0) | low_nibble);
        } else {
            fixture.bytes[8 + (i - 16)] = (uint8_t) ((fixture.bytes[8 + (i - 16)] & 0x0F) | (low_nibble << 4));
        }
    }
    fixture.bytes[4] = (uint8_t) (qh & 0xFF);
    fixture.bytes[5] = (uint8_t) ((qh >> 8) & 0xFF);
    fixture.bytes[6] = (uint8_t) ((qh >> 16) & 0xFF);
    fixture.bytes[7] = (uint8_t) ((qh >> 24) & 0xFF);
    return fixture;
}

#endif  // STRATA_Q51_GATE_ONLY (part A compiles against the pre-change tree too)

// ---- Part A: the capability queries (the leg that also compiles before the change) ----

void check_capability_queries() {
    report(strata::kernels::native_mmvq_supported(7), "A1 native_mmvq_supported(Q5_1=7) is true",
           std::string("got ") + (strata::kernels::native_mmvq_supported(7) ? "true" : "false"));
    report(strata::kernels::dequant_bf16_supported(7), "A2 dequant_bf16_supported(Q5_1=7) is true",
           std::string("got ") + (strata::kernels::dequant_bf16_supported(7) ? "true" : "false"));

    std::string byte_detail;
    std::size_t bytes_for_one_block = 0;
    bool byte_math_ok = true;
    try {
        bytes_for_one_block = strata::kernels::native_mmvq_weight_bytes(7, 32, 1);
    } catch (const std::exception& error) {
        byte_math_ok = false;
        byte_detail = error.what();
    }
    report(byte_math_ok && bytes_for_one_block == 24, "A3 native_mmvq_weight_bytes(7, 32, 1) == 24",
           byte_math_ok ? ("got " + std::to_string(bytes_for_one_block)) : ("threw: " + byte_detail));

    // A wider row proves the per-block byte count is applied per block, not hardcoded for one.
    std::size_t bytes_for_wide_row = 0;
    bool wide_ok = true;
    try {
        bytes_for_wide_row = strata::kernels::native_mmvq_weight_bytes(7, 4096, 8);
    } catch (const std::exception& error) {
        wide_ok = false;
        byte_detail = error.what();
    }
    report(wide_ok && bytes_for_wide_row == (std::size_t) 8 * (4096 / 32) * 24,
           "A4 native_mmvq_weight_bytes(7, 4096, 8) == 24576",
           wide_ok ? ("got " + std::to_string(bytes_for_wide_row)) : ("threw: " + byte_detail));
}

void check_no_regression_on_other_types() {
    const int previously_supported[] = {2, 6, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 42};
    for (int type : previously_supported) {
        report(strata::kernels::native_mmvq_supported(type),
               "A5 type " + std::to_string(type) + " still supported by native_mmvq_supported", "");
    }
    // Q4_1 (3) and Q8_1 (9) were never on the native MMVQ path and must stay off it.
    report(!strata::kernels::native_mmvq_supported(3), "A6 Q4_1 (3) still NOT natively supported", "");
    report(!strata::kernels::native_mmvq_supported(9), "A7 Q8_1 (9) still NOT natively supported", "");
    report(!strata::kernels::dequant_bf16_supported(3), "A8 Q4_1 (3) still NOT dequant_bf16 supported", "");
}

#ifndef STRATA_Q51_GATE_ONLY

// ---- Part B: the Q5_1 dequant branch vs the PATCH 3 reader semantics ----

void check_q51_dequant() {
    Rng rng;
    int worst_mismatch_index = -1;
    double worst_abs_diff = 0.0;
    int blocks_checked = 0;
    for (int block_index = 0; block_index < 64; ++block_index) {
        const Q51Fixture fixture = make_q51_fixture(rng);
        float kernel_out[32] = {};
        float reference_out[32] = {};
        // group32 is the real device dequant helper; gi_in_row indexes the 32-element group, and
        // for a 32-element format the group index is the block index.
        strata::kernels::group32<7, float>(fixture.bytes, 0, kernel_out);
        reference_q51_block(fixture.bytes, reference_out);
        for (int i = 0; i < 32; ++i) {
            const double diff = std::fabs((double) kernel_out[i] - (double) reference_out[i]);
            if (diff > worst_abs_diff) {
                worst_abs_diff = diff;
                worst_mismatch_index = block_index * 32 + i;
            }
        }
        ++blocks_checked;
    }
    report(worst_abs_diff == 0.0, "B1 dequant_bf16 group32<7> bit-identical to the q51 reader decode",
           blocks_checked > 0
               ? ("blocks=" + std::to_string(blocks_checked) + " elements=" + std::to_string(blocks_checked * 32) +
                  " max_abs_diff=" + std::to_string(worst_abs_diff) + " worst_element=" + std::to_string(worst_mismatch_index))
               : "no blocks checked");

    // The neighbour must not have moved: Q5_0 keeps its offset form and its 22-byte stride.
    Rng rng50;
    double worst_q50_diff = 0.0;
    for (int block_index = 0; block_index < 64; ++block_index) {
        uint8_t block[22] = {};
        const uint16_t d_bits = (uint16_t) (0x2C00u | (uint16_t) (rng50.next() & 0x03FFu));
        block[0] = (uint8_t) (d_bits & 0xFF);
        block[1] = (uint8_t) (d_bits >> 8);
        uint32_t qh = 0;
        int codes[32];
        for (int i = 0; i < 32; ++i) {
            codes[i] = (int) (rng50.next() % 32ull);
            if (codes[i] & 16) qh |= (1u << i);
            const int low_nibble = codes[i] & 0x0F;
            if (i < 16) block[6 + i] = (uint8_t) ((block[6 + i] & 0xF0) | low_nibble);
            else block[6 + (i - 16)] = (uint8_t) ((block[6 + (i - 16)] & 0x0F) | (low_nibble << 4));
        }
        block[2] = (uint8_t) (qh & 0xFF);
        block[3] = (uint8_t) ((qh >> 8) & 0xFF);
        block[4] = (uint8_t) ((qh >> 16) & 0xFF);
        block[5] = (uint8_t) ((qh >> 24) & 0xFF);
        float kernel_out[32] = {};
        float reference_out[32] = {};
        strata::kernels::group32<6, float>(block, 0, kernel_out);
        reference_q50_block(block, reference_out);
        for (int i = 0; i < 32; ++i) {
            worst_q50_diff = std::fmax(worst_q50_diff, std::fabs((double) kernel_out[i] - (double) reference_out[i]));
        }
    }
    report(worst_q50_diff == 0.0, "B2 Q5_0 dequant neighbour unchanged (bit-identical to its own reference)",
           "max_abs_diff=" + std::to_string(worst_q50_diff));

    // B3: the artifact's CPU reference dequantizer (include/strata/artifact/dequant.hpp) and the
    // device branch must agree bit for bit.  dequant_bf16_test compares exactly these two on real
    // shard rows in the GPU window; checking it here means a mismatch is found on this box instead of
    // costing a GPU leg to discover.
    Rng rng_ref;
    double worst_reference_diff = 0.0;
    for (int block_index = 0; block_index < 64; ++block_index) {
        const Q51Fixture fixture = make_q51_fixture(rng_ref);
        float kernel_out[32] = {};
        float artifact_out[32] = {};
        strata::dequantize_q5_1(fixture.bytes, artifact_out);
        strata::kernels::group32<7, float>(fixture.bytes, 0, kernel_out);
        for (int i = 0; i < 32; ++i) {
            worst_reference_diff =
                std::fmax(worst_reference_diff, std::fabs((double) kernel_out[i] - (double) artifact_out[i]));
        }
    }
    report(worst_reference_diff == 0.0, "B3 device group32<7> bit-identical to artifact dequantize_q5_1",
           "max_abs_diff=" + std::to_string(worst_reference_diff));
}

// ---- Part C: the Q5_1 MMVQ dot vs the closed form of the dequant ----

void check_q51_mmvq_dot() {
    Rng rng;
    double worst_relative_error = 0.0;
    int iterations = 0;
    for (int trial = 0; trial < 256; ++trial) {
        const Q51Fixture weight = make_q51_fixture(rng);
        const Q81Fixture activation = make_q81_fixture(rng);

        // Lay the fixture bytes out as the kernel's Q51Block and call the real dot helper for the
        // two threads that share one 32-element block (iqs = 0 and iqs = 2 at Qi = 4).
        strata::kernels::Q51Block block;
        std::memcpy(&block, weight.bytes, sizeof(block));
        const float partial_low = strata::kernels::small_q8_dot(&block, &activation.block, 0);
        const float partial_high = strata::kernels::small_q8_dot(&block, &activation.block, 2);
        const double actual = (double) partial_low + (double) partial_high;

        // Closed form of the Q8_1 contract: w = q*d + m, x_hat = q8*d8, and the min term is
        // applied against the ORIGINAL activation sum that Q8_1 carries in ds.y.
        double code_dot = 0.0;
        for (int i = 0; i < 32; ++i) code_dot += (double) weight.codes[i] * (double) activation.codes[i];
        const double expected = weight.d * activation.d8 * code_dot + weight.m * activation.original_sum;

        const double scale = std::fabs(expected) > 1.0 ? std::fabs(expected) : 1.0;
        const double relative_error = std::fabs(actual - expected) / scale;
        if (relative_error > worst_relative_error) worst_relative_error = relative_error;
        ++iterations;
    }
    // The tolerance is relative, not bit-exactness, on purpose: small_q8_dot<Q51Block> keeps the
    // d*d8 and m*s8 products in fp32 where the pinned oracle rounds them with __hmul2
    // (vecdotq.cuh:228-240), so the two agree to the last bits and not in them.  A 1-ulp diff
    // against 3cf03257 is expected here, not a regression.
    report(worst_relative_error < 1e-5, "C1 Q5_1 MMVQ dot equals dequant-then-dot (additive min)",
           "trials=" + std::to_string(iterations) + " worst_relative_error=" + std::to_string(worst_relative_error));

    // Guard against a copy-paste of Q5_0's -16 offset: with a non-zero min the two formats must
    // not agree.  Q5_0's dot is computed on the same activation with the same codes.
    Rng rng2;
    int disagreements = 0;
    for (int trial = 0; trial < 64; ++trial) {
        const Q51Fixture weight = make_q51_fixture(rng2);
        const Q81Fixture activation = make_q81_fixture(rng2);
        strata::kernels::Q51Block q51;
        std::memcpy(&q51, weight.bytes, sizeof(q51));
        uint8_t q50_bytes[22] = {};
        q50_bytes[0] = weight.bytes[0];
        q50_bytes[1] = weight.bytes[1];
        q50_bytes[2] = weight.bytes[4];   // qh moves from offset 4 (Q5_1) to offset 2 (Q5_0)
        q50_bytes[3] = weight.bytes[5];
        q50_bytes[4] = weight.bytes[6];
        q50_bytes[5] = weight.bytes[7];
        for (int i = 0; i < 16; ++i) q50_bytes[6 + i] = weight.bytes[8 + i];
        strata::kernels::Q50Block q50;
        std::memcpy(&q50, q50_bytes, sizeof(q50));
        const double q51_dot = (double) strata::kernels::small_q8_dot(&q51, &activation.block, 0) +
                              (double) strata::kernels::small_q8_dot(&q51, &activation.block, 2);
        const double q50_dot = (double) strata::kernels::small_q8_dot(&q50, &activation.block, 0) +
                              (double) strata::kernels::small_q8_dot(&q50, &activation.block, 2);
        if (std::fabs(q51_dot - q50_dot) > 1e-6) ++disagreements;
    }
    report(disagreements == 64, "C2 Q5_1 dot differs from the Q5_0 offset form on every trial",
           "disagreements=" + std::to_string(disagreements) + "/64");
}

#endif  // STRATA_Q51_GATE_ONLY

}  // namespace

int main(int argc, char** argv) {
    const bool gate_only = argc > 1 && std::strcmp(argv[1], "--gate-only") == 0;
    std::printf("q51_native_test: Q5_1 (ggml type 7) on the native MMVQ path, CPU shim build\n");
    check_capability_queries();
    check_no_regression_on_other_types();
#ifndef STRATA_Q51_GATE_ONLY
    if (!gate_only) {
        check_q51_dequant();
        check_q51_mmvq_dot();
    } else {
        std::printf("note  --gate-only: parts B and C skipped (they name the new symbols)\n");
    }
#endif
    std::printf("q51_native_test: %d checks, %d failed\n", g_checks, g_failures);
    if (g_failures == 0) {
        std::printf("Q51_TEST=PASS\n");
        return 0;
    }
    std::printf("Q51_TEST=FAIL\n");
    return 1;
}
