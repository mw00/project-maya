// tests/glm_dsa/glm_dsa_router_cpu_test.cpp - glm-dsa B5.1 on the CPU (no GPU, runs in CI): the decode router's CPU
// twin (include/strata/kernels/glmd_route.hpp, moe_route_kernel's arithmetic) against the F64 HF oracle.
//
//   1. 10^5 random router rows at 256 experts, top-8, x2.5, HF normalisation: ids equal the oracle's outside near-ties
//      (biased-score gap < 1e-6), weights' relative error < 1e-6;
//   2. the bias selects but never weights: a huge bias on one expert puts it in the top-8 with its own sigmoid weight;
//   3. the HF vs llama.cpp normalisation: identical weights for a normal row, and for an all-tiny row (sum < 2^-14) HF
//      still sums to 2.5 where the clamp (Maya / Flash) scales the weights down - Flash keeps the clamp;
//   4. no SwiGLU clamp: with limit +INF the clamped SwiGLU equals the plain silu(g) * u bit for bit (also at +-INF and
//      +-FLT_MAX), where Flash's limit 10 cuts large activations.
#include "strata/kernels/glmd_route.hpp"

#include "glm_dsa_router_oracle.hpp"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-96s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

}  // namespace

int main() {
    using namespace router_oracle;
    std::printf("glm_dsa_router_cpu_test\n");

    // 1. the twin against the oracle on 10^5 rows
    {
        const Rows R = make_rows(100000, 75, 53u);
        Tally t;
        long invalid = 0;
        for (int r = 0; r < R.n; ++r) {
            const float* lg = R.logits.data() + (size_t) r * kE;
            const float* bs = R.bias.data() + (size_t) (r % R.n_bias) * kE;
            int ids[kK];
            float w[kK];
            if (!glmd::cpu::route_select(lg, bs, kE, kK, kScale, true, glmd::kRouteNormHf, ids, w)) {
                ++invalid;
                continue;
            }
            compare(oracle(lg, bs), ids, w, t);
        }
        std::printf("  rows %ld, near-ties skipped %ld, id mismatches %ld, weight failures %ld, worst rel err %.3g\n",
                    t.rows, t.near_ties, t.id_fail, t.w_fail, t.worst_rel);
        check(invalid == 0, "every random row routes");
        check(t.id_fail == 0, "twin top-8 ids == oracle outside near-ties (10^5 rows, E 256)");
        check(t.w_fail == 0 && t.worst_rel < kRelTol, "twin weights rel err < 1e-6");
        check(t.near_ties < t.rows / 100, "near-ties are rare (< 1%)");
    }

    // 2. the bias selects, it never weights
    {
        std::vector<float> lg(kE, 0.0f), bs(kE, 0.0f);
        for (int e = 0; e < kE; ++e) lg[(size_t) e] = 0.01f * (float) (e % 50);
        lg[200] = -6.0f;     // a weak expert ...
        bs[200] = 10.0f;     // ... selected first by its bias
        int ids[kK];
        float w[kK];
        const bool ok = glmd::cpu::route_select(lg.data(), bs.data(), kE, kK, kScale, true, glmd::kRouteNormHf, ids, w);
        const Pick o = oracle(lg.data(), bs.data());
        double sum = 0;
        for (int i = 0; i < kK; ++i) sum += w[i];
        check(ok && ids[0] == 200 && o.ids[0] == 200, "a +10 bias puts expert 200 first");
        check(std::fabs(w[0] - o.w[0]) <= 1e-6 * o.w[0] && w[0] < 0.01f, "its weight is its own sigmoid's (tiny), not the bias's");
        check(std::fabs(sum - 2.5) < 1e-5, "the 8 weights sum to the scale 2.5");
    }

    // 3. HF vs the clamp
    {
        std::vector<float> lg(kE), tiny(kE);
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        for (int e = 0; e < kE; ++e) {
            lg[(size_t) e] = nd(rng);
            tiny[(size_t) e] = -30.0f + 0.01f * nd(rng);   // every sigmoid ~ 1e-13: sum ~ 7.5e-13 < 2^-14
        }
        int a[kK], b[kK];
        float wa[kK], wb[kK];
        glmd::cpu::route_select(lg.data(), nullptr, kE, kK, kScale, true, glmd::kRouteNormHf, a, wa);
        glmd::cpu::route_select(lg.data(), nullptr, kE, kK, kScale, true, glmd::kRouteNormClamp, b, wb);
        bool same = true;
        for (int i = 0; i < kK; ++i) same = same && a[i] == b[i] && same_bits(wa[i], wb[i]);
        check(same, "normal row: HF and clamp normalisation give identical ids and weights");
        glmd::cpu::route_select(tiny.data(), nullptr, kE, kK, kScale, true, glmd::kRouteNormHf, a, wa);
        glmd::cpu::route_select(tiny.data(), nullptr, kE, kK, kScale, true, glmd::kRouteNormClamp, b, wb);
        double sa = 0, sb = 0;
        for (int i = 0; i < kK; ++i) {
            sa += wa[i];
            sb += wb[i];
        }
        const Pick o = oracle(tiny.data(), nullptr);
        double rel = 0;
        for (int i = 0; i < kK; ++i) rel = std::max(rel, std::fabs(wa[i] - o.w[i]) / o.w[i]);
        check(std::fabs(sa - 2.5) < 1e-5 && rel < kRelTol, "all-tiny row: HF weights still sum to 2.5 (== oracle)");
        check(sb < 1e-6, "all-tiny row: llama.cpp's 2^-14 clamp scales them down (Flash's rule, kept for Flash)");
        // norm_w off: w = p * scale
        glmd::cpu::route_select(lg.data(), nullptr, kE, kK, kScale, false, glmd::kRouteNormHf, a, wa);
        const float p0 = 1.0f / (1.0f + std::exp(-lg[(size_t) a[0]]));
        check(same_bits(wa[0], (float) ((double) p0 * 1.0f * kScale)), "norm_w off: w = sigmoid x scale");
        // a non-finite logit: refused like the kernel's invalid route
        lg[3] = std::numeric_limits<float>::quiet_NaN();
        check(!glmd::cpu::route_select(lg.data(), nullptr, kE, kK, kScale, true, glmd::kRouteNormHf, a, wa),
              "a NaN logit makes the route invalid");
    }

    // 4. no SwiGLU clamp: +INF limit is a no-op
    {
        const float inf = std::numeric_limits<float>::infinity();
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.0f, 40.0f);
        bool noop = true;
        for (int i = 0; i < 200000; ++i) {
            const float g = nd(rng), u = nd(rng);
            noop = noop && same_bits(std::fmin(g, inf), g) && same_bits(std::fmin(std::fmax(u, -inf), inf), u);
            noop = noop && same_bits(glmd::cpu::swiglu_clamped(g, u, inf), (g / (1.0f + std::exp(-g))) * u);
        }
        for (float v : {inf, -inf, FLT_MAX, -FLT_MAX, 0.0f, -0.0f, FLT_MIN})
            noop = noop && same_bits(std::fmin(v, inf), v) && same_bits(std::fmin(std::fmax(v, -inf), inf), v);
        check(noop, "limit +INF: fminf/fmaxf return the value unchanged (2x10^5 values, +-INF, +-FLT_MAX, -0)");
        const double dinf = (double) inf;   // the CPU lane's double clamp (glm_fast_path.cu, native_expert.cpp)
        check(std::min(1e300, dinf) == 1e300 && std::max(-1e300, -dinf) == -1e300, "the CPU lane's double clamp too");
        check(glmd::cpu::swiglu_clamped(30.0f, 20.0f, 10.0f) != glmd::cpu::swiglu_clamped(30.0f, 20.0f, inf),
              "Flash's limit 10 does cut a large activation (the clamp is real where it applies)");
    }

    if (g_fail) {
        std::printf("glm_dsa_router_cpu_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_dsa_router_cpu_test: ok\n");
    return 0;
}
