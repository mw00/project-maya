// tests/glm_dsa/glmd_mv_cpu_test.cpp - glm-dsa B10.6 / B10.7 (CPU): the dots of the new mv types run on the host from
// the very header the GPU GEMV compiles (src/kernels/cuda/glmd_mv_dots.cuh), on rows quantized by ggml.
//   1. row bytes / type ids (Q4_0 18 B, Q4_1 20, Q5_0 22, Q5_1 24, IQ4_NL 18 per 32, F16 2 per value) match ggml;
//   2. per type, 64 rows x 6144: the warp-per-row algorithm (lane parts, butterfly) vs the same q8_1 semantics in
//      double (sum_b d (d8 sum q u) + the type's offset term on the q8_1 block sum): rel err < 2e-6 per row;
//   3. vs ggml's own dequantized rows dotted with the dequantized q8_1 activation in double: rel err < 2e-3 (the
//      q8_1 activation's fp16 block sum is the only difference: llama.cpp's MMVQ arithmetic);
//   4. the two-warp split (B10.6 NW2): every call of the row taken exactly once, values equal to one warp within
//      f32 rounding (rel < 1e-6);
//   5. F16 rows with the f32 input and with the q8_1 input vs double.
#include <cstdint>
#include <cstring>

#include "../../src/kernels/cuda/glmd_mv_dots.cuh"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
void expect(bool ok, const char* what) {
    std::printf("%-72s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}
void gate(const char* what, double err, double lim) {
    const bool ok = err < lim;
    std::printf("%-64s %.3e (gate %.1e) %s\n", what, err, lim, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

uint16_t half_bits(float f) {   // RN-even (finite inputs in range)
    return ggml_fp32_to_fp16(f);
}

// q8_1 of n values (glm_fast.cu q8_1_store_warp's arithmetic)
std::vector<glmd_mvq::BQ8_1> q8_1(const std::vector<float>& x) {
    std::vector<glmd_mvq::BQ8_1> q(x.size() / 32);
    for (size_t b = 0; b < q.size(); ++b) {
        float amax = 0, sum = 0;
        for (int l = 0; l < 32; ++l) {
            amax = std::max(amax, std::fabs(x[32 * b + l]));
            sum += x[32 * b + l];
        }
        const float d = std::min(amax / 127.0f, 65504.0f);
        for (int l = 0; l < 32; ++l)
            q[b].qs[l] = amax == 0.0f ? 0 : (int8_t) std::max(-127.0f, std::min(127.0f, std::round(x[32 * b + l] / d)));
        q[b].d = half_bits(d);
        q[b].s = half_bits(sum);
    }
    return q;
}

int iq4nl_value(int q) {   // ggml kvalues_iq4nl, independent of the header's packed words
    static const int k[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return k[q & 15];
}

float tree(float* s) {   // warp_sum's butterfly as lane 0 sees it
    for (int o = 16; o > 0; o >>= 1)
        for (int l = 0; l < 32; ++l)
            if ((l & o) == 0) s[l] = s[l] + s[l + o];
    return s[0];
}

// the q8_1 semantics in double: per 32-block, d * d8 * sum(q u) + offset term (as each dot's formula)
double sem_dot(int type, const uint8_t* row, const std::vector<glmd_mvq::BQ8_1>& x, int n_in) {
    double acc = 0;
    for (int b = 0; b < n_in / 32; ++b) {
        const glmd_mvq::BQ8_1& xb = x[(size_t) b];
        const double d8 = glmd_mvq::h2f(xb.d), s8 = glmd_mvq::h2f(xb.s);
        int sumi = 0;
        double d = 0, off = 0;
        for (int j = 0; j < 32; ++j) {
            int q = 0;
            switch (type) {
                case glmd_mvq::kQ4_0: {
                    const auto* w = (const glmd_mvq::BQ4_0*) row + b;
                    q = j < 16 ? (w->qs[j] & 15) : (w->qs[j - 16] >> 4);
                    d = glmd_mvq::h2f(w->d);
                    off = -8.0 * d * s8;
                    break;
                }
                case glmd_mvq::kQ4_1: {
                    const auto* w = (const glmd_mvq::BQ4_1*) row + b;
                    q = j < 16 ? (w->qs[j] & 15) : (w->qs[j - 16] >> 4);
                    d = glmd_mvq::h2f(w->d);
                    off = (double) glmd_mvq::h2f(w->m) * s8;
                    break;
                }
                case glmd_mvq::kQ5_0:
                case glmd_mvq::kQ5_1: {
                    const bool one = type == glmd_mvq::kQ5_1;
                    const uint8_t* qs = one ? ((const glmd_mvq::BQ5_1*) row + b)->qs : ((const glmd_mvq::BQ5_0*) row + b)->qs;
                    const uint8_t* qh = one ? ((const glmd_mvq::BQ5_1*) row + b)->qh : ((const glmd_mvq::BQ5_0*) row + b)->qh;
                    uint32_t h;
                    std::memcpy(&h, qh, 4);
                    q = (j < 16 ? (qs[j] & 15) : (qs[j - 16] >> 4)) | (((h >> j) & 1) << 4);
                    if (one) {
                        d = glmd_mvq::h2f(((const glmd_mvq::BQ5_1*) row + b)->d);
                        off = (double) glmd_mvq::h2f(((const glmd_mvq::BQ5_1*) row + b)->m) * s8;
                    } else {
                        d = glmd_mvq::h2f(((const glmd_mvq::BQ5_0*) row + b)->d);
                        off = -16.0 * d * s8;
                    }
                    break;
                }
                case glmd_mvq::kIQ4_NL: {
                    const auto* w = (const glmd_mvq::BIQ4_NL*) row + b;
                    q = iq4nl_value(j < 16 ? (w->qs[j] & 15) : (w->qs[j - 16] >> 4));
                    d = glmd_mvq::h2f(w->d);
                    off = 0;
                    break;
                }
            }
            sumi += q * xb.qs[j];
        }
        acc += d * d8 * sumi + off;
    }
    return acc;
}

void test_types() {
    expect(glmd_mvq::row_bytes(glmd_mvq::kQ4_0, 6144) == ggml_row_size(GGML_TYPE_Q4_0, 6144) &&
               glmd_mvq::row_bytes(glmd_mvq::kQ4_1, 6144) == ggml_row_size(GGML_TYPE_Q4_1, 6144) &&
               glmd_mvq::row_bytes(glmd_mvq::kQ5_0, 6144) == ggml_row_size(GGML_TYPE_Q5_0, 6144) &&
               glmd_mvq::row_bytes(glmd_mvq::kQ5_1, 6144) == ggml_row_size(GGML_TYPE_Q5_1, 6144) &&
               glmd_mvq::row_bytes(glmd_mvq::kIQ4_NL, 6144) == ggml_row_size(GGML_TYPE_IQ4_NL, 6144) &&
               glmd_mvq::row_bytes(glmd_mvq::kF16, 6144) == ggml_row_size(GGML_TYPE_F16, 6144) &&
               glmd_mvq::row_bytes(glmd_mvq::kQ4_0, 100) == 0 && glmd_mvq::row_bytes(8, 6144) == 0,
           "row bytes == ggml_row_size for the six types");
    expect(GGML_TYPE_Q4_0 == glmd_mvq::kQ4_0 && GGML_TYPE_Q4_1 == glmd_mvq::kQ4_1 && GGML_TYPE_Q5_0 == glmd_mvq::kQ5_0 &&
               GGML_TYPE_Q5_1 == glmd_mvq::kQ5_1 && GGML_TYPE_IQ4_NL == glmd_mvq::kIQ4_NL && GGML_TYPE_F16 == glmd_mvq::kF16,
           "type ids == ggml's");
}

void test_dots(std::mt19937& rng) {
    const int n_in = 6144, rows = 64;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(n_in);
    for (auto& v : x) v = nd(rng);
    const auto xq = q8_1(x);
    std::vector<float> xd(n_in);   // the dequantized activation
    for (int e = 0; e < n_in; ++e) xd[e] = xq[e / 32].qs[e % 32] * glmd_mvq::h2f(xq[e / 32].d);
    for (int type : {glmd_mvq::kQ4_0, glmd_mvq::kQ4_1, glmd_mvq::kQ5_0, glmd_mvq::kQ5_1, glmd_mvq::kIQ4_NL}) {
        const char* name = ggml_type_name((ggml_type) type);
        std::vector<float> w((size_t) rows * n_in);
        for (auto& v : w) v = nd(rng) * 0.05f + 0.01f;
        const size_t rb = glmd_mvq::row_bytes(type, n_in);
        std::vector<uint8_t> q(rb * rows);
        ggml_quantize_chunk((ggml_type) type, w.data(), q.data(), 0, rows, n_in, nullptr);
        std::vector<float> deq(n_in);
        double e_sem = 0, e_deq = 0, e_nw2 = 0;
        bool once = true;
        for (int r = 0; r < rows; ++r) {
            const uint8_t* row = q.data() + rb * r;
            float lanes[32];
            for (int l = 0; l < 32; ++l) lanes[l] = glmd_mvq::lane_part(type, row, xq.data(), n_in, l, 0, 1);
            const float y = tree(lanes);
            float p0[32], p1[32];
            for (int l = 0; l < 32; ++l) {
                p0[l] = glmd_mvq::lane_part(type, row, xq.data(), n_in, l, 0, 2);
                p1[l] = glmd_mvq::lane_part(type, row, xq.data(), n_in, l, 1, 2);
            }
            const float y2 = tree(p0) + tree(p1);
            const double ys = sem_dot(type, row, xq, n_in);
            ggml_get_type_traits((ggml_type) type)->to_float(row, deq.data(), n_in);
            double yd = 0, mag = 0;
            for (int e = 0; e < n_in; ++e) {
                yd += (double) deq[e] * xd[e];
                mag += std::fabs((double) deq[e] * xd[e]);
            }
            e_sem = std::max(e_sem, std::fabs(y - ys) / mag);
            e_deq = std::max(e_deq, std::fabs(y - yd) / mag);
            e_nw2 = std::max(e_nw2, std::fabs(y2 - y) / mag);
            // every call exactly once over the two parts
            std::vector<int> seen((size_t) 2 * n_in / 32, 0);
            for (int part = 0; part < 2; ++part)
                for (int l = 0; l < 32; ++l)
                    for (int k = 32 * part + l; k < 2 * n_in / 32; k += 64) ++seen[(size_t) k];
            once &= std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; });
        }
        char what[96];
        std::snprintf(what, sizeof what, "%s: warp dot vs q8_1 semantics in double (err / sum|.|)", name);
        gate(what, e_sem, 2e-6);
        std::snprintf(what, sizeof what, "%s: vs ggml dequant x dequant q8_1 (err / sum|.|)", name);
        gate(what, e_deq, 2e-3);
        std::snprintf(what, sizeof what, "%s: two-warp split vs one warp (err / sum|.|)", name);
        gate(what, e_nw2, 1e-6);
        expect(once, "two-warp split takes every call once");
    }
    // F16
    {
        std::vector<uint16_t> w16((size_t) n_in);
        std::vector<float> wf(n_in);
        for (int e = 0; e < n_in; ++e) wf[e] = glmd_mvq::h2f(w16[e] = half_bits(nd(rng) * 0.05f));
        float lf[32], lq[32];
        for (int l = 0; l < 32; ++l) {
            lf[l] = glmd_mvq::f16_lane(w16.data(), x.data(), nullptr, n_in, l);
            lq[l] = glmd_mvq::f16_lane(w16.data(), nullptr, xq.data(), n_in, l);
        }
        double rf = 0, rq = 0, mag = 0;
        for (int e = 0; e < n_in; ++e) {
            rf += (double) wf[e] * x[e];
            rq += (double) wf[e] * xd[e];
            mag += std::fabs((double) wf[e] * x[e]);
        }
        gate("F16: f32 input vs double (err / sum|.|)", std::fabs(tree(lf) - rf) / mag, 1e-6);
        gate("F16: q8_1 input vs double on the dequantized input", std::fabs(tree(lq) - rq) / mag, 1e-6);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(77);
    test_types();
    test_dots(rng);
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
