// tests/glm_dsa/glm_prefill_types_test.cpp - CPU: glm-dsa's extra FP16 prompt types (src/core/glm_prefill_types.hpp)
// against ggml's own to_float (ggml-base), bit for bit up to float rounding of the d * q + m products.
//
// F16, Q4_1, Q5_1 and Q2_K rows of random blocks (finite half scales), decoded group by group through ext_group (the
// same function the device kernel runs) and compared with ggml_get_type_traits(t)->to_float on the whole row. Also
// checks ext_geometry against ggml's block size / type size, and the refusal text names the type.
#include "../../src/core/glm_prefill_types.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace pf = strata::core::glmpf;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// a finite half with a modest exponent (no inf / NaN in the reference either)
uint16_t rand_half(std::mt19937& rng) {
    const uint16_t sign = (uint16_t) ((rng() & 1u) << 15);
    const uint16_t exp = (uint16_t) (8 + rng() % 12);   // 2^-7 .. 2^4
    return (uint16_t) (sign | (exp << 10) | (rng() & 0x3ffu));
}

}  // namespace

int main() {
    std::printf("glm_prefill_types_test: the glm-dsa prompt's extra FP16 types vs ggml to_float\n");
    std::mt19937 rng(1234);
    for (int t : {1, 3, 7, 10}) {
        int be = 0, bb = 0;
        const bool geo = pf::ext_geometry(t, be, bb);
        const auto* tr = ggml_get_type_traits((ggml_type) t);
        check(geo && be == (int) tr->blck_size && bb == (int) tr->type_size,
              std::string(pf::type_name(t)) + ": block geometry matches ggml (" + std::to_string(be) + " / " +
                  std::to_string(bb) + ")");
        const int64_t cols = 512, rows = 3;
        const int64_t row_bytes = cols / be * bb;
        std::vector<uint8_t> w((size_t) (rows * row_bytes));
        for (auto& b : w) b = (uint8_t) rng();
        // finite half fields: every half of an F16 row, d/m of Q4_1 / Q5_1, d/dmin of Q2_K
        const auto put = [&](size_t off) { const uint16_t h = rand_half(rng); w[off] = (uint8_t) h; w[off + 1] = (uint8_t) (h >> 8); };
        for (int64_t r = 0; r < rows; ++r) {
            uint8_t* row = w.data() + r * row_bytes;
            const size_t base = (size_t) (row - w.data());
            for (int64_t b = 0; b < cols / be; ++b) {
                const size_t o = base + (size_t) b * bb;
                if (t == 1) put(o);
                else if (t == 3 || t == 7) { put(o); put(o + 2); }
                else { put(o + 80); put(o + 82); }
            }
        }
        double worst = 0.0;
        int64_t bad = 0;
        for (int64_t r = 0; r < rows; ++r) {
            const uint8_t* row = w.data() + r * row_bytes;
            std::vector<float> ref((size_t) cols), got((size_t) cols);
            tr->to_float(row, ref.data(), cols);
            for (int64_t g = 0; g < cols / 32; ++g) pf::ext_group(t, row, g, got.data() + g * 32);
            for (int64_t i = 0; i < cols; ++i) {
                const double d = std::fabs((double) got[(size_t) i] - ref[(size_t) i]);
                const double tol = 1e-6 * std::fabs((double) ref[(size_t) i]) + 1e-30;
                worst = std::max(worst, d / (std::fabs((double) ref[(size_t) i]) + 1e-30));
                bad += !(d <= tol) || !std::isfinite(got[(size_t) i]);
            }
        }
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s: %lld x %lld values equal ggml's (worst rel %.2e, %lld off)", pf::type_name(t),
                      (long long) rows, (long long) cols, worst, (long long) bad);
        check(bad == 0, buf);
    }
    // the half decoder on the edge cases
    check(pf::half_to_f32(0x3c00) == 1.0f && pf::half_to_f32(0xc000) == -2.0f && pf::half_to_f32(0x0001) == ggml_fp16_to_fp32(0x0001) &&
              std::isinf(pf::half_to_f32(0x7c00)) && pf::half_to_f32(0x8000) == 0.0f,
          "half_to_f32: 1, -2, the smallest subnormal, inf, -0");
    int be = 0, bb = 0;
    check(pf::ext_geometry(19, be, bb) && be == (int) ggml_get_type_traits(GGML_TYPE_IQ1_S)->blck_size &&
              bb == (int) ggml_get_type_traits(GGML_TYPE_IQ1_S)->type_size,
          "IQ1_S geometry matches ggml (decoded by iq_dequant_f16)");
    for (int t : pf::kExtF16Types) check(pf::ext_f16_type(t), std::string(pf::type_name(t)) + " is an extra type");
    check(!pf::ext_f16_type(8) && !pf::ext_f16_type(30) && !pf::ext_f16_type(0), "Q8_0 / BF16 / F32 stay Maya's");
    const std::string why = pf::refusal(34);
    check(why.find("ggml type 34") != std::string::npos && why.find("token by token") != std::string::npos,
          "refusal names the type and the fallback: " + why.substr(0, 60) + "...");
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
