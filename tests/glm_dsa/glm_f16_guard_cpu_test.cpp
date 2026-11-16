// tests/glm_dsa/glm_f16_guard_cpu_test.cpp - glm-dsa B15.1 on the CPU: the FP16 activation range guard's twins.
//
// Rows with amax 1e5 (llama.cpp #30260's MUL_MAT_ID case) go through the prompt path's three FFN shapes as FP16
// tensor-core GEMMs with F32 accumulation (glmd::guard_twin, the kernels' exact element results):
//   dense FFN      x16 -> gate / up -> FP16 SwiGLU h -> down      (glm_prefill.cu hgemm_q: B.dh16)
//   shared expert  the same at the shared expert's width            (M.sh16)
//   routed down    h -> q8_1 MMQ activations -> down               (the B14.1 clamp; MMQ never converts to FP16)
// Checks: without the guard the outlier rows are not finite; with it every row is finite and within FP16 precision
// of an F64 reference; every other row is bit-identical to the unguarded path; a batch without an outlier (Flash, a
// normal prompt) is bit-identical in every output with every scale 1.  Plus the scale rule itself.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/glmd_f16_guard.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace sk = strata::kernels;
namespace gt = glmd::guard_twin;

int g_fail = 0;
#define CHECK(c, ...)                                     \
    do {                                                  \
        if (!(c)) {                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
            ++g_fail;                                     \
        }                                                 \
    } while (0)

bool row_finite(const float* y, int n) {
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(y[i])) return false;
    return true;
}
double row_rel(const float* y, const double* ref, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        num += (y[i] - ref[i]) * (y[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

std::vector<uint16_t> to16(const std::vector<float>& w) {
    std::vector<uint16_t> o(w.size());
    for (size_t i = 0; i < w.size(); ++i) o[i] = sk::f16_from_f32(w[i]);
    return o;
}

void scale_rule() {
    CHECK(glmd::f16_guard_scale(0.0f) == 1.0f, "0");
    CHECK(glmd::f16_guard_scale(65504.0f) == 1.0f, "65504 is finite in FP16: scale 1");
    CHECK(glmd::f16_guard_scale(65505.0f) == 2.0f, "just above 65504: 2");
    CHECK(glmd::f16_guard_scale(65536.0f) == 2.0f, "65536 / 2 = 32768");
    CHECK(glmd::f16_guard_scale(65537.0f) == 4.0f, "65537 / 2 > 32768: 4");
    CHECK(glmd::f16_guard_scale(1e5f) == 4.0f, "1e5: 4 (1e5 / 4 = 25000)");
    CHECK(glmd::f16_guard_scale(1e10f) == 524288.0f, "1e10: 2^19 (got %g)", (double) glmd::f16_guard_scale(1e10f));
    CHECK(glmd::f16_guard_scale(INFINITY) == 1.0f, "inf: nothing to repair");
    CHECK(glmd::f16_guard_scale(NAN) == 1.0f, "NaN: nothing to repair");
    CHECK(glmd::f16_guard_scale(3.0e38f) > 1.0f && std::isfinite(3.0e38f / glmd::f16_guard_scale(3.0e38f)), "3e38");
    for (float a = 65505.0f; a < 1e30f; a *= 1.37f) {
        const float s = glmd::f16_guard_scale(a);
        CHECK(a / s <= glmd::kF16Target && a / s > glmd::kF16Target / 2.0f, "amax %g scale %g", (double) a, (double) s);
        CHECK(std::isfinite(sk::f32_from_f16(sk::f16_from_f32(a / s))), "amax %g scaled is finite in FP16", (double) a);
    }
}

// One FFN through FP16 GEMMs: Y = Wd . swiglu(Wg . x, Wu . x), guarded or not.  Rows T, widths E / FF.
struct Ffn {
    int E, FF, T;
    float limit;
    std::vector<float> wg, wu, wd;
    std::vector<uint16_t> wg16, wu16, wd16;
};

Ffn make_ffn(int E, int FF, int T, float limit, uint32_t seed) {
    Ffn f{E, FF, T, limit, {}, {}, {}, {}, {}, {}};
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    f.wg.resize((size_t) FF * E);
    f.wu.resize((size_t) FF * E);
    f.wd.resize((size_t) E * FF);
    for (auto& v : f.wg) v = sk::f32_from_f16(sk::f16_from_f32(nd(rng)));   // FP16-exact weights: only x rounds
    for (auto& v : f.wu) v = sk::f32_from_f16(sk::f16_from_f32(nd(rng)));
    for (auto& v : f.wd) v = sk::f32_from_f16(sk::f16_from_f32(nd(rng)));
    f.wg16 = to16(f.wg);
    f.wu16 = to16(f.wu);
    f.wd16 = to16(f.wd);
    return f;
}

std::vector<float> run_ffn(const Ffn& f, const std::vector<float>& x, bool guard, std::vector<float>* sx_out = nullptr,
                           std::vector<float>* sh_out = nullptr) {
    const int E = f.E, FF = f.FF, T = f.T;
    std::vector<uint16_t> x16((size_t) T * E), h16((size_t) T * FF);
    std::vector<float> g((size_t) T * FF), u((size_t) T * FF), y((size_t) T * E), sx(T, 1.0f), sh(T, 1.0f);
    gt::pack_f16(x.data(), E, 0, 1, E, T, x16.data());                 // the producer (res_rows / hc_finish)
    if (guard) gt::guard_rows(x.data(), E, 0, 1, E, T, x16.data(), sx.data());
    gt::gemm_f16(x16.data(), E, f.wg16.data(), g.data(), FF, T, FF, E);
    if (guard) gt::unscale_rows(g.data(), FF, FF, T, sx.data());
    gt::gemm_f16(x16.data(), E, f.wu16.data(), u.data(), FF, T, FF, E);
    if (guard) gt::unscale_rows(u.data(), FF, FF, T, sx.data());
    gt::swiglu_f16(g.data(), u.data(), (int64_t) T * FF, f.limit, h16.data());   // glm_batch.cu swiglu_f16
    if (guard) gt::guard_swiglu(g.data(), u.data(), FF, T, f.limit, h16.data(), sh.data());
    gt::gemm_f16(h16.data(), FF, f.wd16.data(), y.data(), E, T, E, FF);
    if (guard) gt::unscale_rows(y.data(), E, E, T, sh.data());
    if (sx_out) *sx_out = sx;
    if (sh_out) *sh_out = sh;
    return y;
}

std::vector<double> ref_ffn(const Ffn& f, const std::vector<float>& x) {
    const int E = f.E, FF = f.FF, T = f.T;
    std::vector<double> y((size_t) T * E);
    std::vector<double> h(FF);
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < FF; ++o) {
            // the FP16 rounding of x is part of the GEMM's input either way: the reference sees x as F32
            double a = 0.0, b = 0.0;
            for (int k = 0; k < E; ++k) {
                a += (double) f.wg[(size_t) o * E + k] * x[(size_t) t * E + k];
                b += (double) f.wu[(size_t) o * E + k] * x[(size_t) t * E + k];
            }
            const double gg = std::fmin(a, (double) f.limit), uu = std::fmin(std::fmax(b, -(double) f.limit), (double) f.limit);
            h[o] = gg / (1.0 + std::exp(-gg)) * uu;
        }
        for (int o = 0; o < E; ++o) {
            double a = 0.0;
            for (int k = 0; k < FF; ++k) a += (double) f.wd[(size_t) o * FF + k] * h[k];
            y[(size_t) t * E + o] = a;
        }
    }
    return y;
}

void ffn_case(const char* name, int E, int FF, float limit, uint32_t seed) {
    const int T = 6, hot = 2, hot2 = 4;
    const Ffn f = make_ffn(E, FF, T, limit, seed);
    std::mt19937 rng(seed + 7);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> x((size_t) T * E);
    for (auto& v : x) v = nd(rng);
    x[(size_t) hot * E + 17] = 1e5f;      // the #30260 row: amax 1e5 at one channel
    x[(size_t) hot * E + 300 % E] = -7.5e4f;
    for (int k = 0; k < E; ++k) x[(size_t) hot2 * E + k] *= 2e4f;   // a row overflowing everywhere (amax ~ 1.6e5)
    std::vector<float> sx, sh;
    const std::vector<float> y0 = run_ffn(f, x, false);
    const std::vector<float> y1 = run_ffn(f, x, true, &sx, &sh);
    const std::vector<double> yr = ref_ffn(f, x);
    if (std::isinf(limit)) {   // (a finite limit's fminf maps a NaN gate to the limit: wrong, but maybe finite)
        CHECK(!row_finite(&y0[(size_t) hot * E], E), "%s: the unguarded outlier row should not be finite", name);
        CHECK(!row_finite(&y0[(size_t) hot2 * E], E), "%s: the unguarded wide row should not be finite", name);
    }
    for (int t = 0; t < T; ++t) {
        const float* yg = &y1[(size_t) t * E];
        const double rel = row_rel(yg, &yr[(size_t) t * E], E);
        CHECK(row_finite(yg, E), "%s: guarded row %d not finite", name, t);
        CHECK(rel < 4e-3, "%s: guarded row %d rel %.3e vs the F64 reference", name, t, rel);
        if (t != hot && t != hot2) {
            CHECK(std::memcmp(yg, &y0[(size_t) t * E], (size_t) E * 4) == 0, "%s: row %d is not bit-identical", name, t);
            CHECK(sx[t] == 1.0f && sh[t] == 1.0f, "%s: row %d scales %g %g", name, t, (double) sx[t], (double) sh[t]);
        }
        std::printf("%s row %d: x scale %-8g h scale %-12g rel %.2e (unguarded: %s, rel %.2e)\n", name, t, (double) sx[t],
                    (double) sh[t], rel, row_finite(&y0[(size_t) t * E], E) ? "finite" : "inf/NaN",
                    row_rel(&y0[(size_t) t * E], &yr[(size_t) t * E], E));
    }
    CHECK(sx[hot] == 4.0f, "%s: the 1e5 row's x scale %g", name, (double) sx[hot]);
    if (std::isinf(limit)) CHECK(sh[hot] > 1.0f, "%s: the 1e5 row's h scale", name);
    else CHECK(sh[hot] == 1.0f, "%s: a finite limit bounds h (|h| <= limit^2): scale 1", name);
    // no outlier: every bit the same, every scale 1
    std::vector<float> xs = x;
    for (int k = 0; k < E; ++k) {
        xs[(size_t) hot * E + k] = nd(rng);
        xs[(size_t) hot2 * E + k] = nd(rng);
    }
    const std::vector<float> z0 = run_ffn(f, xs, false);
    const std::vector<float> z1 = run_ffn(f, xs, true, &sx, &sh);
    CHECK(std::memcmp(z0.data(), z1.data(), z0.size() * 4) == 0, "%s: a batch without outliers is not bit-identical", name);
    for (int t = 0; t < T; ++t) CHECK(sx[t] == 1.0f && sh[t] == 1.0f, "%s: scale != 1 without an outlier", name);
}

// ---- the routed down: q8_1 MMQ activations (32-value blocks), the B14.1 clamp, a weight with and without mins ----

struct Q81 {
    float d;      // the scale (FP16 bits in DS4 blocks; F32 in D4 blocks)
    float s;      // DS4 only: the block's sum of x, FP16
    int8_t q[32];
};
Q81 q8_1(const float* x, bool ds4, bool clamp) {
    Q81 b{};
    float amax = 0.0f, sum = 0.0f;
    for (int j = 0; j < 32; ++j) {
        amax = std::fmax(amax, std::fabs(x[j]));
        sum += x[j];
    }
    const float d = amax / 127.0f;
    for (int j = 0; j < 32; ++j) b.q[j] = (int8_t) (d > 0.0f ? std::lround(x[j] / d) : 0);
    if (ds4) {
        b.d = sk::f32_from_f16(sk::f16_from_f32(d));
        b.s = sk::f32_from_f16(sk::f16_from_f32(sum));
        if (clamp && std::isinf(b.s)) b.s = b.s > 0 ? 65504.0f : -65504.0f;   // glmd::q8_mmq_clamp (B14.1)
        if (clamp && std::isinf(b.d)) b.d = 65504.0f;
    } else {
        b.d = d;
        b.s = 0.0f;
    }
    return b;
}

void routed_down() {
    const int FF = 256, E = 64;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> q4(0, 15);
    // a weight row per output: w = dw * q - mw (Q4_K-like, with mins) per 32-block; the IQ types have no min (D4)
    std::vector<float> h(FF);
    for (auto& v : h) v = nd(rng) * 50.0f;
    h[3] = 1e5f;   // the outlier
    for (int j = 32; j < 64; ++j) h[j] = 4e3f;   // a block whose float sum (128000) overflows the FP16 sum field
    for (int mode = 0; mode < 3; ++mode) {   // 0: D4 (IQ types), 1: DS4 with the clamp, 2: DS4 without
        const bool ds4 = mode > 0, clamp = mode == 1;
        bool finite = true;
        double num = 0.0, den = 0.0;
        for (int o = 0; o < E; ++o) {
            double y = 0.0, yr = 0.0;
            for (int b = 0; b < FF / 32; ++b) {
                const float dw = 0.01f * (1 + (o + b) % 5), mw = ds4 ? 0.05f : 0.0f;
                int qw[32];
                for (int j = 0; j < 32; ++j) qw[j] = ds4 ? q4(rng) : q4(rng) - 8;
                const Q81 xb = q8_1(&h[b * 32], ds4, clamp);
                int sumi = 0;
                for (int j = 0; j < 32; ++j) sumi += qw[j] * xb.q[j];
                y += (double) dw * xb.d * sumi - (ds4 ? (double) mw * xb.s : 0.0);
                for (int j = 0; j < 32; ++j) yr += ((double) dw * qw[j] - mw) * h[b * 32 + j];
            }
            finite = finite && std::isfinite(y);
            num += (y - yr) * (y - yr);
            den += yr * yr;
        }
        const double rel = std::sqrt(num / den);
        std::printf("routed down %s: %s, rel %.3e\n", mode == 0 ? "D4 (IQ2/IQ3 types)" : mode == 1 ? "DS4 + B14.1 clamp" : "DS4 unclamped",
                    finite ? "finite" : "NOT finite", rel);
        if (mode == 0) CHECK(finite && rel < 1e-2, "routed down D4: finite and rel %.3e < 1e-2", rel);
        if (mode == 1) CHECK(finite, "routed down DS4 with the B14.1 clamp is finite");
        if (mode == 2) CHECK(!finite, "routed down DS4 without the clamp should overflow (the case B14.1 guards)");
    }
}

// the head-major pack (q_abs / ctx / attn16 in the glm-dsa prompt mixer): the guard covers every head of a row
void heads() {
    const int T = 3, H = 4, K = 16, row = H * 24, head = 24;
    std::vector<float> src((size_t) T * row);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (float) ((int) (i % 13) - 6);
    src[(size_t) 1 * row + 2 * head + 5] = 262144.0f;   // row 1, head 2, k 5 (2^18)
    src[(size_t) 2 * row + 3 * head + 20] = 9e5f;   // row 2, head 3, OUTSIDE the K = 16 packed values: no scale
    std::vector<uint16_t> a((size_t) T * H * K), b((size_t) T * H * K);
    std::vector<float> sc(T);
    gt::pack_f16(src.data(), row, head, H, K, T, a.data());
    b = a;
    gt::guard_rows(src.data(), row, head, H, K, T, b.data(), sc.data());
    CHECK(sc[0] == 1.0f && sc[1] == 8.0f && sc[2] == 1.0f, "head scales %g %g %g", (double) sc[0], (double) sc[1], (double) sc[2]);
    CHECK(std::memcmp(a.data(), b.data(), (size_t) H * K * 2) == 0, "row 0 untouched");
    CHECK(std::memcmp(a.data() + 2 * H * K, b.data() + 2 * H * K, (size_t) H * K * 2) == 0, "row 2 untouched");
    CHECK(sk::f32_from_f16(b[(size_t) (1 * H + 2) * K + 5]) == 32768.0f, "row 1's outlier / 8");
    for (int h = 0; h < H; ++h)
        for (int k = 0; k < K; ++k)
            CHECK(b[(size_t) (1 * H + h) * K + k] == sk::f16_from_f32(src[(size_t) row + h * head + k] / 8.0f),
                  "row 1 head %d k %d: FP16(x / 8)", h, k);
}

}  // namespace

int main() {
    scale_rule();
    heads();
    ffn_case("dense FFN", 512, 384, INFINITY, 1);        // glm-dsa: swiglu_exp = +INF (no clamp)
    ffn_case("shared expert", 512, 256, INFINITY, 2);    // glm-dsa: swiglu_shexp = +INF
    ffn_case("dense FFN, limit 10", 512, 384, 10.0f, 3); // Flash's finite limit: |h| <= 100, only x16 can overflow
    routed_down();
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
