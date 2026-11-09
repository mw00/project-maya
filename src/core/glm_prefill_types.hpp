// src/core/glm_prefill_types.hpp - the batched glm-dsa prompt's FP16 weight types (glm-dsa B9.1 / W7 build review).
//
// The prompt's cuBLAS projections read every non-routed weight as FP16 slices. Maya's dequant_f16 (dequant_bf16.cu)
// covers BF16 (narrowed), Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K..Q6_K, IQ4_XS, the i-quants IQ2_XXS / IQ2_XS / IQ3_XXS /
// IQ3_S / IQ2_S / IQ1_M and Maya's type 42; Maya's prefill_setup admits only BF16 and the dequant_bf16 table, so the
// i-quants are taken on glm-dsa only (glmpf::f16_covers). The glm-dsa pack (tools/strata/glm_dsa_pack.py
// plan_tensor) can also leave F16, Q4_1, Q5_1 (STRATA_PACK_NATIVE32), Q2_K and IQ1_S native; those five are glm-dsa's:
// decoded here (F16 / Q4_1 / Q5_1 / Q2_K, ggml's dequantize_row_* order and float math) or by iq_dequant_f16 (IQ1_S),
// on glm-dsa only (Flash keeps Maya's set: bit-identical). Any other type is refused with refusal() and the prompt
// runs token by token. tools/strata/test_prefill_types.py checks these lists against tools/strata/type_coverage.py's tables and the
// pack's native types; glm_prefill_types_test checks the decoders against ggml's to_float on the CPU.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GLMPF_HD __host__ __device__ inline
#else
#define GLMPF_HD inline
#endif

namespace strata::core::glmpf {

/// Maya's dequant_f16 types beyond its dequant_bf16 geometry table: the i-quants it routes to iq_dequant_f16 (the
/// batched prompt takes them on glm-dsa only: Flash keeps Maya v1.1.3's prefill_setup set).
constexpr int kMayaIqF16Types[] = {16, 17, 18, 21, 22, 29};
/// glm-dsa's extra FP16 types for the glm-dsa prompt: F16, Q4_1, Q5_1, Q2_K (decoded below), IQ1_S (iq_dequant_f16).
constexpr int kExtF16Types[] = {1, 3, 7, 10, 19};

GLMPF_HD bool ext_f16_type(int t) { return t == 1 || t == 3 || t == 7 || t == 10 || t == 19; }

/// ggml block geometry of the extra types (elements, bytes); false for any other type.
GLMPF_HD bool ext_geometry(int t, int& elems, int& bytes) {
    switch (t) {
    case 1: elems = 1; bytes = 2; return true;       // F16
    case 3: elems = 32; bytes = 20; return true;     // Q4_1: d, m, qs[16]
    case 7: elems = 32; bytes = 24; return true;     // Q5_1: d, m, qh[4], qs[16]
    case 10: elems = 256; bytes = 84; return true;   // Q2_K: scales[16], qs[64], d, dmin
    case 19: elems = 256; bytes = 50; return true;   // IQ1_S: d, qs[32], qh[8 x u16]
    default: return false;
    }
}

/// IEEE half bits -> float (exact; host and device).
GLMPF_HD float half_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {   // subnormal: normalise
            e = 113;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            bits = s | (e << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        bits = s | 0x7f800000u | (m << 13);
    } else {
        bits = s | ((e + 112) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

GLMPF_HD uint16_t rd16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }

/// The g-th group of 32 values of one row of type t (F16 / Q4_1 / Q5_1 / Q2_K), in ggml's dequantize_row_* order.
GLMPF_HD void ext_group(int t, const uint8_t* row, int64_t g, float* y) {
    if (t == 1) {
        const uint8_t* p = row + g * 64;
        for (int i = 0; i < 32; ++i) y[i] = half_to_f32(rd16(p + 2 * i));
    } else if (t == 3) {
        const uint8_t* b = row + g * 20;
        const float d = half_to_f32(rd16(b)), m = half_to_f32(rd16(b + 2));
        for (int j = 0; j < 16; ++j) {
            y[j] = (b[4 + j] & 0x0F) * d + m;
            y[j + 16] = (b[4 + j] >> 4) * d + m;
        }
    } else if (t == 7) {
        const uint8_t* b = row + g * 24;
        const float d = half_to_f32(rd16(b)), m = half_to_f32(rd16(b + 2));
        const uint32_t qh = (uint32_t) b[4] | ((uint32_t) b[5] << 8) | ((uint32_t) b[6] << 16) | ((uint32_t) b[7] << 24);
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh0 = (uint8_t) (((qh >> (j + 0)) << 4) & 0x10);
            const uint8_t xh1 = (uint8_t) ((qh >> (j + 12)) & 0x10);
            const int x0 = (b[8 + j] & 0x0F) | xh0, x1 = (b[8 + j] >> 4) | xh1;
            y[j] = x0 * d + m;
            y[j + 16] = x1 * d + m;
        }
    } else if (t == 10) {
        const uint8_t* b = row + (g / 8) * 84;
        const int k = (int) (g % 8), n = k / 4, j = k % 4, shift = 2 * j;
        const uint8_t* sc = b;
        const uint8_t* q = b + 16 + 32 * n;
        const float d = half_to_f32(rd16(b + 80)), mn = half_to_f32(rd16(b + 82));
        for (int h = 0; h < 2; ++h) {
            const uint8_t s = sc[2 * k + h];
            const float dl = d * (s & 0xF), ml = mn * (s >> 4);
            for (int l = 0; l < 16; ++l) y[16 * h + l] = dl * (int8_t) ((q[16 * h + l] >> shift) & 3) - ml;
        }
    }
}

/// ggml type names for the refusal message.
inline const char* type_name(int t) {
    static const char* const n[] = {"F32", "F16", "Q4_0", "Q4_1", "type4", "type5", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
                                    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS",
                                    "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64",
                                    "IQ1_M", "BF16"};
    return t >= 0 && t < (int) (sizeof(n) / sizeof(n[0])) ? n[t] : "unknown";
}

/// Why the batched glm-dsa prompt refuses a weight type (the prompt then runs token by token).
inline std::string refusal(int t) {
    return std::string("ggml type ") + std::to_string(t) + " (" + type_name(t) +
           ") has no FP16 dequantizer for the batched glm-dsa prompt (covered: F32, BF16, F16, Q4_0, Q4_1, Q5_0, "
           "Q5_1, Q8_0, IQ4_NL, Q2_K..Q6_K, IQ4_XS, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S); the prompt "
           "runs token by token";
}

}  // namespace strata::core::glmpf
