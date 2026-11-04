// tests/ds4_markov_q8_test.cpp - the DSpark drafter's markov head as Q8_0 blocks (DS4_MARKOV_Q8:
// src/ds4/ds4_markov_q8.hpp, rounded on the host as the head loads) against ggml's own quantize_row_q8_0_ref (its type
// traits' from_float_ref), byte for byte: random rows from F16, BF16 and F32 bits, and blocks at the edges - zeros and
// negative zeros, a lone value, halves at the rounding, the largest magnitude negative, scales that are F16
// subnormals, that underflow to 0 or overflow past 65504.  Then a head group member's rows rounded alone against the
// same rows of the whole head (markov_share_q8 against markov_add_q8), and every code within half a step of its
// source.  No GPU, no model.
#include "../src/ds4/ds4_markov_q8.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace dk = strata::ds4;

namespace {

int g_fail = 0;

uint16_t bf16_bits(float f) {   // (truncating: any BF16 bits are a source)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
float bf16_value(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// the source rows (n x R) as F32 values and as the pack's bits of type fmt
struct Src {
    std::vector<float> f;
    std::vector<uint8_t> bits;
};
Src make(int fmt, const std::vector<float>& want) {
    Src s;
    s.f.resize(want.size());
    s.bits.resize(want.size() * (fmt == dk::kMarkovF32 ? 4 : 2));
    for (size_t i = 0; i < want.size(); ++i) {
        if (fmt == dk::kMarkovF32) {
            s.f[i] = want[i];
            std::memcpy(&s.bits[i * 4], &want[i], 4);
        } else {
            const uint16_t h = fmt == dk::kMarkovBF16 ? bf16_bits(want[i]) : strata::kernels::f16_from_f32(want[i]);
            s.f[i] = fmt == dk::kMarkovBF16 ? bf16_value(h) : strata::kernels::f32_from_f16(h);
            std::memcpy(&s.bits[i * 2], &h, 2);
        }
    }
    return s;
}

// values for n rows of R: block kinds by the block's index (random ones in between)
std::vector<float> values(size_t n, int R, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(n * (size_t) R);
    for (size_t b = 0; b < v.size() / 32; ++b) {
        float* x = v.data() + b * 32;
        const float sc = std::ldexp(1.0f, (int) (rng() % 13) - 8);   // (2^-8 .. 2^4)
        for (int j = 0; j < 32; ++j) x[j] = nd(rng) * sc;
        switch (b % 16) {
            case 1: std::fill(x, x + 32, 0.0f); break;                                     // zeros (scale 0)
            case 2: for (int j = 0; j < 32; ++j) x[j] = j % 2 ? -0.0f : 0.0f; break;        // negative zeros
            case 3: std::fill(x, x + 32, 0.0f); x[7] = -3.25f; break;                       // a lone value
            case 4: for (int j = 0; j < 32; ++j) x[j] = (float) (j - 16) + 0.5f; x[0] = 127.0f; break;   // halves
            case 5: x[19] = -40.0f * sc; break;                                             // the largest negative
            case 6: for (int j = 0; j < 32; ++j) x[j] *= 1e-4f; break;                      // scale an F16 subnormal
            case 7: for (int j = 0; j < 32; ++j) x[j] *= 1e-9f; break;                      // scale 0 in F16, x not 0
            case 8: for (int j = 0; j < 32; ++j) x[j] *= 1e7f; break;                       // scale past 65504 (inf)
            case 9: for (int j = 0; j < 32; ++j) x[j] = (j % 3 - 1) * 254.0f / 2.0f; break; // exact steps
            default: break;
        }
    }
    return v;
}

void check(const char* what, bool ok, const char* detail = "") {
    if (!ok) ++g_fail;
    std::printf("%-64s %s%s\n", what, ok ? "ok" : "FAIL", detail);
}

void run(int fmt, int R, size_t n, std::mt19937& rng) {
    const char* fn = fmt == dk::kMarkovF32 ? "F32" : fmt == dk::kMarkovBF16 ? "BF16" : "F16";
    const Src s = make(fmt, values(n, R, rng));
    const size_t rb = dk::markov_q8_row_bytes(R);
    std::vector<uint8_t> got(n * rb), want(n * rb);
    dk::markov_q8_rows(s.bits.data(), fmt, n, R, got.data());
    const ggml_type_traits* tr = ggml_get_type_traits(GGML_TYPE_Q8_0);
    if (tr == nullptr || tr->from_float_ref == nullptr || ggml_row_size(GGML_TYPE_Q8_0, R) != rb) {
        check("ggml's Q8_0 quantizer", false);
        return;
    }
    for (size_t r = 0; r < n; ++r) tr->from_float_ref(s.f.data() + r * R, want.data() + r * rb, R);
    size_t bad = 0, at = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i] && bad++ == 0) at = i;
    char what[96], detail[96] = "";
    std::snprintf(what, sizeof what, "%s, rank %d, %zu rows: the bytes ggml's quantize_row_q8_0_ref", fn, R, n);
    if (bad) std::snprintf(detail, sizeof detail, " (%zu bytes differ, the first at %zu)", bad, at);
    check(what, bad == 0, detail);

    // a member's rows alone: the whole head's rows, byte for byte (uneven shares, one row, the last row)
    const size_t cuts[][2] = {{0, n / 3}, {n / 3, n / 3 + 1}, {n / 2, n - n / 2}, {n - 1, 1}, {7, 13}};
    bool same = true;
    for (const auto& c : cuts) {
        const size_t v0 = std::min(c[0], n - 1), vn = std::min(c[1], n - v0);
        std::vector<uint8_t> part(vn * rb);
        dk::markov_q8_rows(s.bits.data() + v0 * (size_t) R * (fmt == dk::kMarkovF32 ? 4 : 2), fmt, vn, R, part.data());
        same = same && std::memcmp(part.data(), got.data() + v0 * rb, part.size()) == 0;
    }
    std::snprintf(what, sizeof what, "%s, rank %d: a member's rows alone the whole head's", fn, R);
    check(what, same);

    // each code within half a step of its source (the step: max|x| / 127 in F32; a block of zeros all 0)
    double worst = 0.0;
    bool zeros = true;
    for (size_t b = 0; b < n * (size_t) (R / 32); ++b) {
        const float* x = s.f.data() + b * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
        const double step = amax / 127.0f;
        const int8_t* q = (const int8_t*) (got.data() + b * dk::kMarkovQ8Bytes + 2);
        for (int j = 0; j < 32; ++j) {
            if (step == 0.0) zeros = zeros && q[j] == 0;
            else worst = std::max(worst, std::fabs((double) q[j] * step - (double) x[j]) / step);
        }
    }
    std::snprintf(what, sizeof what, "%s, rank %d: every code within half a step", fn, R);
    std::snprintf(detail, sizeof detail, " (at most %.4f steps)", worst);
    check(what, zeros && worst <= 0.5 + 1e-3, detail);
}

}  // namespace

int main() {
    std::mt19937 rng(20261028);
    for (int fmt : {dk::kMarkovF16, dk::kMarkovBF16, dk::kMarkovF32})
        for (int R : {256, 32, 96}) run(fmt, R, R == 256 ? 1000 : 300, rng);
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
