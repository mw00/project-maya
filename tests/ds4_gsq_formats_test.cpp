// CPU-only format checks against ggml. Device kernels are exercised by ds4_gguf_test.
#include "../src/ds4/ds4_engram_format.hpp"
#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "ggml.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace ds = strata::ds4;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", what);
        std::exit(1);
    }
}

static void test_engram_formats() {
    struct Case { const char* dtype; bool scales; int width; ds::EngramFormat format; size_t bytes; };
    for (const Case& c : std::array<Case, 5>{{
             {"MXFP8", false, 1280, ds::EngramFormat::Fp8, 1320},
             {"Q6_K", false, 1280, ds::EngramFormat::Q6K, 1050},
             {"IQ4_NL", false, 1280, ds::EngramFormat::Iq4Nl, 720},
             {"IQ4_NL", false, 32, ds::EngramFormat::Iq4Nl, 18},
             {"F8_E4M3", true, 1280, ds::EngramFormat::SplitFp8, 1280}}}) {
        ds::EngramFormat got;
        check(ds::engram_row_format(c.dtype, c.scales, c.width, got), "valid Engram format rejected");
        check(got == c.format && ds::engram_source_row_bytes(got, c.width) == c.bytes,
              "wrong Engram format or source stride");
    }
    // Exact width domain, including partial blocks, empty and negative widths.
    for (int width = -1; width <= 1024; ++width) {
        for (const char* dtype : {"IQ4_NL", "Q6_K", "MXFP8", "F8_E4M3", "unknown"}) {
            for (bool scales : {false, true}) {
                ds::EngramFormat got;
                const bool whole = width > 0 && width % (std::strcmp(dtype, "Q6_K") == 0 ? 256 : 32) == 0;
                const bool want = whole && std::strcmp(dtype, "unknown") != 0 &&
                                  (std::strcmp(dtype, "F8_E4M3") != 0 || scales);
                check(ds::engram_row_format(dtype, scales, width, got) == want,
                      "Engram format accepted invalid width/type/scales or rejected a valid row");
            }
        }
    }
}

static void test_iq4nl_values() {
    const auto* traits = ggml_get_type_traits(GGML_TYPE_IQ4_NL);
    // Enumerate every packed byte at every position, with distinct neighbors.
    // Zero, signed zero, subnormal, positive and negative FP16 scales are all included.
    for (uint16_t bits : {0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x2800, 0x3c00, 0xbc00, 0x7bff}) {
        for (int code = 0; code < 256; ++code) {
            std::array<uint8_t, 18> block{};
            std::memcpy(block.data(), &bits, sizeof bits);
            for (int i = 0; i < 16; ++i) block[(size_t) i + 2] = (uint8_t) (code + 17 * i);
            std::array<float, 32> want{};
            traits->to_float(block.data(), want.data(), 32);
            const float scale = ggml_fp16_to_fp32(bits);
            for (int i = 0; i < 32; ++i) {
                const float got = ds::engram_iq4nl_value(block.data(), i, scale);
                check(std::memcmp(&got, &want[(size_t) i], sizeof got) == 0,
                      "IQ4_NL Engram value differs from ggml, including zero sign");
            }
        }
    }
}

static void test_embedding_rows() {
    std::mt19937 rng(1234);
    constexpr int width = 512;
    for (ggml_type type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ2_XS, GGML_TYPE_Q8_0}) {
        const auto* traits = ggml_get_type_traits(type);
        const size_t bytes = ggml_row_size(type, width);
        for (int sample = 0; sample < 64; ++sample) {
            std::vector<uint8_t> row(bytes);
            for (auto& v : row) v = (uint8_t) rng();
            for (size_t off = 0; off < bytes; off += traits->type_size) {
                const uint16_t scale = sample % 2 ? 0x2800 : 0xa800;
                std::memcpy(row.data() + off, &scale, 2);
                if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)
                    std::memcpy(row.data() + off + 2, &scale, 2);
            }
            std::array<float, width> want{};
            traits->to_float(row.data(), want.data(), width);
            std::array<uint16_t, width + 2> out{};
            out.front() = out.back() = 0xdead;
            check(ds::gguf::gguf_row_to_bf16(type, row.data(), width, out.data() + 1), "valid embedding row rejected");
            for (int i = 0; i < width; ++i)
                check(out[(size_t) i + 1] == ggml_fp32_to_bf16(want[(size_t) i]).bits, "wrong BF16 embedding value");
            check(out.front() == 0xdead && out.back() == 0xdead, "embedding conversion overwrote a guard");
            check(!ds::gguf::gguf_row_to_bf16(type, row.data(), width - 1, out.data()), "partial embedding block accepted");
            check(!ds::gguf::gguf_row_to_bf16(type, nullptr, width, out.data()), "null embedding source accepted");
            check(!ds::gguf::gguf_row_to_bf16(type, row.data(), width, nullptr), "null embedding destination accepted");
        }
    }
    std::array<uint8_t, 256> row{};
    std::array<uint16_t, width> out{};
    check(!ds::gguf::gguf_row_to_bf16(-1, row.data(), width, out.data()), "negative ggml type accepted");
    check(!ds::gguf::gguf_row_to_bf16(GGML_TYPE_COUNT, row.data(), width, out.data()), "out-of-range ggml type accepted");
    check(!ds::gguf::gguf_row_to_bf16(GGML_TYPE_Q5_K, row.data(), 0, out.data()), "empty embedding row accepted");
}

int main() {
    test_engram_formats();
    test_iq4nl_values();
    test_embedding_rows();
    std::puts("Engram formats, 73728 IQ4_NL values, and 256 embedding rows passed");
}
