// Real batched MLA attention against a CPU softmax reference, without weights.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "strata/kernels/glm_batch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

// Host fixture in #41's record layout: KV signed codes followed by KV/32 FP16 scales.
// Round each scale to FP16 before quantizing, as dsa_prep does; include zero-scale groups.
static std::vector<uint8_t> int8_latents(const std::vector<uint16_t>& lat, int kv) {
    const size_t rows = lat.size() / kv, stride = kv + kv / 16;
    std::vector<uint8_t> packed(rows * stride);
    for (size_t r = 0; r < rows; ++r) {
        uint8_t* rec = packed.data() + r * stride;
        for (int g = 0; g < kv / 32; ++g) {
            float vals[32], amax = 0;
            for (int k = 0; k < 32; ++k) {
                __half h;
                std::memcpy(&h, &lat[r * kv + g * 32 + k], 2);
                vals[k] = (r + g) % 17 == 0 ? 0.0f : __half2float(h);
                amax = std::max(amax, std::fabs(vals[k]));
            }
            const __half hscale = __float2half(amax / 127.0f);
            const float scale = __half2float(hscale);
            std::memcpy(rec + kv + g * 2, &hscale, 2);
            for (int k = 0; k < 32; ++k) {
                const int code = scale > 0 ? (int) std::nearbyint(vals[k] / scale) : 0;
                const int8_t q = (int8_t) std::max(-127, std::min(127, code));
                std::memcpy(rec + g * 32 + k, &q, 1);
            }
        }
    }
    return packed;
}

int main() {
    constexpr int T = 5, H = 32, KV = 512, rows = 67, stride = 37;
    const float scale = 1.0f / std::sqrt(128.0f);
    std::vector<float> q(T * H * KV), lat(rows * KV), got(q.size());
    std::vector<uint16_t> half(lat.size());
    std::vector<int> cells(T * stride, -1), counts = {0, 1, 17, 37, 35};
    for (size_t i = 0; i < q.size(); ++i) q[i] = 0.4f * std::sin(0.17f * (float) i);
    for (size_t i = 0; i < lat.size(); ++i) {
        const __half h = __float2half(0.7f * std::cos(0.11f * (float) i));
        std::memcpy(&half[i], &h, sizeof(h));
        lat[i] = __half2float(h);
    }
    for (int t = 0; t < T - 1; ++t)
        for (int s = 0; s < counts[t]; ++s)
            cells[t * stride + s] = s % 7 == 5 ? -1 : (11 * s + 3 * t) % rows;
    // The last token has only masked cells; the first has no selected cells.
    float* d_q = nullptr;
    float* d_out = nullptr;
    uint16_t* d_lat = nullptr;
    int* d_cells = nullptr;
    int* d_counts = nullptr;
    CHECK(hipMalloc((void**) &d_q, q.size() * sizeof(float)));
    CHECK(hipMalloc((void**) &d_out, q.size() * sizeof(float)));
    CHECK(hipMalloc((void**) &d_lat, half.size() * sizeof(uint16_t)));
    CHECK(hipMalloc((void**) &d_cells, cells.size() * sizeof(int)));
    CHECK(hipMalloc((void**) &d_counts, counts.size() * sizeof(int)));
    CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_lat, half.data(), half.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_cells, cells.data(), cells.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_counts, counts.data(), counts.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemset(d_out, 0xff, q.size() * sizeof(float))); // catch unwritten outputs
    strata::kernels::glmb::mla_attn(d_q, d_lat, d_cells, d_counts, stride, H, KV, scale, T, d_out, nullptr);
    CHECK(hipDeviceSynchronize());
    if (strata::kernels::glmb::launch_errors()) return 3;
    CHECK(hipMemcpy(got.data(), d_out, got.size() * sizeof(float), hipMemcpyDeviceToHost));

    double worst = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            const size_t base = ((size_t) t * H + h) * KV;
            std::vector<double> scores(counts[t], -INFINITY);
            double max_score = -INFINITY, denom = 0.0;
            for (int s = 0; s < counts[t]; ++s) {
                const int cell = cells[t * stride + s];
                if (cell < 0) continue;
                double dot = 0.0;
                for (int k = 0; k < KV; ++k) dot += (double) q[base + k] * lat[cell * KV + k];
                scores[s] = dot * scale;
                max_score = std::max(max_score, scores[s]);
            }
            for (double& score : scores) {
                score = std::isfinite(score) ? std::exp(score - max_score) : 0.0;
                denom += score;
            }
            for (int k = 0; k < KV; ++k) {
                double ref = 0.0;
                if (denom > 0.0)
                    for (int s = 0; s < counts[t]; ++s) {
                        const int cell = cells[t * stride + s];
                        if (cell >= 0) ref += scores[s] / denom * lat[cell * KV + k];
                    }
                if (!std::isfinite(got[base + k])) {
                    std::fprintf(stderr, "nonfinite attention output: token=%d head=%d column=%d\n", t, h, k);
                    return 4;
                }
                worst = std::max(worst, std::fabs(got[base + k] - ref));
            }
        }
    }
    CHECK(hipFree(d_counts)); CHECK(hipFree(d_cells)); CHECK(hipFree(d_lat)); CHECK(hipFree(d_out)); CHECK(hipFree(d_q));
    std::printf("GLM prefill attention: max absolute error %.3e (tolerance 5e-5)\n", worst);
    if (worst > 5e-5) return 5;

    // The rocWMMA kernel (FP16 Q) against the F32 kernel on the same FP16-rounded Q: random data, many cells.
    // Same guard as glm_prefill.cu's mla_attn_path(): on a card without WMMA (gfx908, gfx1030) the f16q device
    // pass compiles to a trap - the engine never takes that path, so there is nothing to compare here.
    if (!strata::kernels::glmb::mla_wmma2_supported()) {
        std::puts("GLM prefill WMMA attention: skipped (unsupported architecture)");
    } else {
        constexpr int T2 = 48, H2 = 32, rows2 = 2048, stride2 = 700;
        std::vector<float> q2((size_t) T2 * H2 * KV), lat2((size_t) rows2 * KV), a(q2.size()), b(q2.size());
        std::vector<uint16_t> lat16(lat2.size()), q16(q2.size());
        std::vector<int> cells2((size_t) T2 * stride2, -1), cnt2(T2);
        uint32_t rng = 12345u;
        const auto rnd = [&rng]() { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) & 0xffff) / 32768.0f - 1.0f; };
        const auto to_h = [](float v) { const __half h = __float2half(v); uint16_t u; std::memcpy(&u, &h, 2); return u; };
        for (size_t i = 0; i < q2.size(); ++i) {
            q16[i] = to_h(1.5f * rnd());
            const __half h = *reinterpret_cast<const __half*>(&q16[i]);
            q2[i] = __half2float(h);
        }
        for (size_t i = 0; i < lat2.size(); ++i) lat16[i] = to_h(rnd());
        const auto lat8 = int8_latents(lat16, KV);
        for (int t = 0; t < T2; ++t) {
            cnt2[t] = t % 5 == 0 ? 3 : std::min(stride2, 40 + 15 * t);   // includes partial 16-cell chunks
            for (int s = 0; s < cnt2[t]; ++s) cells2[(size_t) t * stride2 + s] = (7 * s + 13 * t) % rows2;
        }
        uint16_t *dq16 = nullptr, *dl = nullptr;
        float *dq = nullptr, *da = nullptr, *db = nullptr;
        int *dc = nullptr, *dn = nullptr;
        CHECK(hipMalloc((void**) &dq16, q16.size() * 2)); CHECK(hipMalloc((void**) &dq, q2.size() * 4));
        CHECK(hipMalloc((void**) &dl, lat16.size() * 2)); CHECK(hipMalloc((void**) &da, q2.size() * 4));
        CHECK(hipMalloc((void**) &db, q2.size() * 4)); CHECK(hipMalloc((void**) &dc, cells2.size() * 4));
        CHECK(hipMalloc((void**) &dn, cnt2.size() * 4));
        CHECK(hipMemcpy(dq16, q16.data(), q16.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dq, q2.data(), q2.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dc, cells2.data(), cells2.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dn, cnt2.data(), cnt2.size() * 4, hipMemcpyHostToDevice));
        for (int q8 = 0; q8 < 2; ++q8) {
            CHECK(hipMemcpy(dl, q8 ? (const void*) lat8.data() : (const void*) lat16.data(),
                            q8 ? lat8.size() : lat16.size() * 2, hipMemcpyHostToDevice));
            CHECK(hipMemset(da, 0xff, q2.size() * 4));
            strata::kernels::glmb::mla_attn_f16q(dq16, dl, dc, dn, stride2, H2, KV, scale, T2, da, nullptr, q8 != 0);
            strata::kernels::glmb::mla_attn(dq, dl, dc, dn, stride2, H2, KV, scale, T2, db, nullptr, q8 != 0);
            CHECK(hipDeviceSynchronize());
            if (strata::kernels::glmb::launch_errors()) return 3;
            CHECK(hipMemcpy(a.data(), da, a.size() * 4, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(b.data(), db, b.size() * 4, hipMemcpyDeviceToHost));
            double num = 0, den = 0, worst_head = 0;
            for (size_t r = 0; r < (size_t) T2 * H2; ++r) {
                double rn = 0, rd = 0;
                for (int j = 0; j < KV; ++j) {
                    const double d = (double) a[r * KV + j] - b[r * KV + j];
                    if (!std::isfinite(a[r * KV + j]) || !std::isfinite(b[r * KV + j])) { std::fprintf(stderr, "nonfinite WMMA output row %zu\n", r); return 4; }
                    rn += d * d; rd += (double) b[r * KV + j] * b[r * KV + j];
                }
                num += rn; den += rd;
                worst_head = std::max(worst_head, std::sqrt(rn / std::max(1e-30, rd)));
            }
            const double rel = std::sqrt(num / std::max(1e-30, den));
            std::printf("GLM prefill WMMA attention (%s latents) vs F32 kernel: rel L2 %.3e, worst head %.3e (tolerance 1e-3)\n",
                        q8 ? "INT8" : "FP16", rel, worst_head);
            if (rel > 1e-3 || worst_head > 5e-3) return 6;
        }
        hipFree(dq16); hipFree(dq); hipFree(dl); hipFree(da); hipFree(db); hipFree(dc); hipFree(dn);
    }

    // The wave32 WMMA kernel (F32 Q, FP16 high/residual) against the F32 kernel: random data, partial 16/32-cell
    // chunks, masked cells, multiple head groups, and zero/tiny/large queries
    if (!strata::kernels::glmb::mla_wmma2_supported()) {
        std::puts("GLM prefill WMMA2 attention: skipped (unsupported architecture)");
    } else {
        constexpr int T3 = 12, H3 = 32, rows3 = 256, stride3 = 70;
        std::vector<float> q3((size_t) T3 * H3 * KV), c(q3.size()), d(q3.size());
        std::vector<uint16_t> lat3((size_t) rows3 * KV);
        std::vector<int> cells3((size_t) T3 * stride3, -1);
        std::vector<int> cnt3 = {0, 1, 15, 16, 17, 31, 32, 33, 48, 64, 70, 7};
        uint32_t rng3 = 987654321u;
        const auto rnd3 = [&rng3]() { rng3 = rng3 * 1664525u + 1013904223u; return ((rng3 >> 8) & 0xffff) / 32768.0f - 1.0f; };
        for (int t = 0; t < T3; ++t)
            for (int h = 0; h < H3; ++h) {
                const float mag = h == 0 ? 0.0f : h == 1 ? 1e-3f : h == 2 ? 1e3f : 1.5f;
                for (int k = 0; k < KV; ++k) q3[((size_t) t * H3 + h) * KV + k] = mag * rnd3();
            }
        for (size_t i = 0; i < lat3.size(); ++i) {
            const __half h = __float2half(rnd3());
            std::memcpy(&lat3[i], &h, 2);
        }
        const auto lat8 = int8_latents(lat3, KV);
        // Leave the final token fully masked, even though it has selected cells.
        for (int t = 0; t < T3 - 1; ++t)
            for (int s = 0; s < cnt3[t]; ++s)
                cells3[(size_t) t * stride3 + s] = s % 9 == 4 ? -1 : (7 * s + 13 * t) % rows3;
        float *dq3 = nullptr, *dc3 = nullptr, *dd3 = nullptr;
        uint16_t* dl3 = nullptr;
        int *dcl3 = nullptr, *dn3 = nullptr;
        CHECK(hipMalloc((void**) &dq3, q3.size() * 4)); CHECK(hipMalloc((void**) &dc3, q3.size() * 4));
        CHECK(hipMalloc((void**) &dd3, q3.size() * 4)); CHECK(hipMalloc((void**) &dl3, lat3.size() * 2));
        CHECK(hipMalloc((void**) &dcl3, cells3.size() * 4)); CHECK(hipMalloc((void**) &dn3, cnt3.size() * 4));
        CHECK(hipMemcpy(dq3, q3.data(), q3.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dcl3, cells3.data(), cells3.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dn3, cnt3.data(), cnt3.size() * 4, hipMemcpyHostToDevice));
        for (int q8 = 0; q8 < 2; ++q8) {
            CHECK(hipMemcpy(dl3, q8 ? (const void*) lat8.data() : (const void*) lat3.data(),
                            q8 ? lat8.size() : lat3.size() * 2, hipMemcpyHostToDevice));
            CHECK(hipMemset(dc3, 0xff, q3.size() * 4));
            strata::kernels::glmb::mla_attn_wmma2(dq3, dl3, dcl3, dn3, stride3, H3, KV, scale, T3, dc3, nullptr, q8 != 0);
            strata::kernels::glmb::mla_attn(dq3, dl3, dcl3, dn3, stride3, H3, KV, scale, T3, dd3, nullptr, q8 != 0);
            CHECK(hipDeviceSynchronize());
            if (strata::kernels::glmb::launch_errors()) return 3;
            CHECK(hipMemcpy(c.data(), dc3, c.size() * 4, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(d.data(), dd3, d.size() * 4, hipMemcpyDeviceToHost));
            double num3 = 0, den3 = 0, worst3 = 0;
            for (size_t r = 0; r < (size_t) T3 * H3; ++r) {
                double rn = 0, rd = 0;
                for (int j = 0; j < KV; ++j) {
                    const double diff = (double) c[r * KV + j] - d[r * KV + j];
                    if (!std::isfinite(c[r * KV + j]) || !std::isfinite(d[r * KV + j])) { std::fprintf(stderr, "nonfinite WMMA2 output row %zu\n", r); return 4; }
                    rn += diff * diff; rd += (double) d[r * KV + j] * d[r * KV + j];
                }
                num3 += rn; den3 += rd;
                worst3 = std::max(worst3, std::sqrt(rn / std::max(1e-30, rd)));
            }
            const double rel3 = std::sqrt(num3 / std::max(1e-30, den3));
            std::printf("GLM prefill WMMA2 attention (%s latents) vs F32 kernel: rel L2 %.3e, worst head %.3e (tolerance 1e-3)\n",
                        q8 ? "INT8" : "FP16", rel3, worst3);
            if (rel3 > 1e-3 || worst3 > 5e-3) return 7;
        }
        hipFree(dq3); hipFree(dc3); hipFree(dd3); hipFree(dl3); hipFree(dcl3); hipFree(dn3);
    }
    std::puts("PASS: empty/masked cells, partial tiles, multiple tiles and head groups");
    return 0;
}
