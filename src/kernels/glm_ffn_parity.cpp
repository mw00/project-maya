// src/kernels/glm_ffn_parity.cpp - the clamped SwiGLU vs ref/glm.py, plus the wrong readings.
//
// The misreadings (glm_ffn.hpp, cpu-ops.cpp L3450):
//   1. BOTH-SIDED GATE CLAMP.  clamp(gate, -limit, +limit) instead of min(gate, limit) - the gate's
//      lower tail is NOT clamped, and with gate values below -limit the two readings differ a lot.
//   2. CLAMPING silu(gate) instead of the raw gate.  The order matters when they disagree.
//   3. UNCLAMPED UP.  The up clamp is what keeps the product finite when up explodes.
//   4. NO CLAMP AT ALL - the plain SwiGLU, which agrees wherever |values| < limit and drifts outside.
//
// The fixture's gate values therefore reach well beyond +/-limit on purpose.
#include "maya/kernels/glm_ffn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// `gate_both_sided`, `clamp_after_silu`, `no_up_clamp` and `no_clamp` are the WRONG readings.
void ref_swiglu(const std::vector<float>& gate, const std::vector<float>& up, float limit,
                std::vector<float>& out, bool gate_both_sided = false, bool clamp_after_silu = false,
                bool no_up_clamp = false, bool no_clamp = false) {
    out.resize(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        double g = gate[i];
        if (no_clamp) {
            // nothing
        } else if (clamp_after_silu) {
            g = g / (1.0 + std::exp((double) -g));
            g = std::min(g, (double) limit);
        } else {
            g = gate_both_sided ? std::min(std::max(g, (double) -limit), (double) limit) : std::min(g, (double) limit);
        }
        const double u = no_up_clamp ? (double) up[i] : std::min(std::max((double) up[i], (double) -limit), (double) limit);
        out[i] = (float) ((g / (1.0 + std::exp(-g))) * u);
    }
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "glm_ffn_parity: %s\n", what.c_str());
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    if (!selftest) {
        std::fprintf(stderr, "glm_ffn_parity: run with --selftest (the CTest form)\n");
        return 2;
    }
    constexpr float LIMIT = 10.0f;   // the released model's swiglu_limit
    const int n = 4096;
    std::mt19937 rng(31);
    std::normal_distribution<float> nd(0.0f, 8.0f);   // gate/up reach well past +/-limit on purpose
    std::vector<float> gate(n), up(n);
    for (auto& v : gate) v = nd(rng);
    for (auto& v : up) v = nd(rng);

    float *d_g, *d_u, *d_o;
    check(cudaMalloc(&d_g, n * sizeof(float)), "gate");
    check(cudaMalloc(&d_u, n * sizeof(float)), "up");
    check(cudaMalloc(&d_o, n * sizeof(float)), "out");
    check(cudaMemcpy(d_g, gate.data(), n * sizeof(float), cudaMemcpyHostToDevice), "gate H2D");
    check(cudaMemcpy(d_u, up.data(), n * sizeof(float), cudaMemcpyHostToDevice), "up H2D");
    maya::kernels::glm_swiglu_clamp(d_g, d_u, LIMIT, n, d_o, nullptr);
    check(cudaGetLastError(), "glm_swiglu_clamp launch");
    std::vector<float> kernel(n);
    check(cudaMemcpy(kernel.data(), d_o, n * sizeof(float), cudaMemcpyDeviceToHost), "out D2H");
    check(cudaDeviceSynchronize(), "sync");

    std::vector<float> ref, wrong;
    ref_swiglu(gate, up, LIMIT, ref);
    // the fixture's products reach ~100 in magnitude, where one f32 ulp is ~7.6e-6 - the tolerance
    // is a few of those, not an absolute-quality claim
    if (getenv("GF_DEBUG")) {
        size_t worst = 0;
        double worst_d = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs((double) kernel[i] - ref[i]);
            if (d > worst_d) {
                worst_d = d;
                worst = i;
            }
        }
        std::fprintf(stderr, "debug worst i=%zu delta %.3e: gate %.4f up %.4f kernel %.7f ref %.7f\n", worst,
                     worst_d, gate[worst], up[worst], kernel[worst], ref[worst]);
    }
    require(max_abs(kernel, ref) < 5e-5, "glm_swiglu_clamp diverges from ref/glm.py");

    ref_swiglu(gate, up, LIMIT, wrong, true);
    // the one-sided-vs-both-sided difference lives entirely in silu's damped lower tail: it is
    // bounded by ~silu(-limit)*max|u| = 4.5e-4 * 10 with this fixture, so its threshold is 2e-3 -
    // still 40x the positive tolerance and far above f32 noise
    require(max_abs(ref, wrong) > 2e-3, "the one-sided gate clamp is invisible");
    ref_swiglu(gate, up, LIMIT, wrong, false, true);
    // bounded by (limit - silu(limit))*max|u| = 4.5e-4 * 10 for the same reason
    require(max_abs(ref, wrong) > 1e-3, "clamping after silu is indistinguishable");
    ref_swiglu(gate, up, LIMIT, wrong, false, false, true);
    require(max_abs(ref, wrong) > 1e-2, "the up clamp is invisible - enlarge the spread");
    ref_swiglu(gate, up, LIMIT, wrong, false, false, false, true);
    require(max_abs(ref, wrong) > 1e-2, "the clamp is invisible - enlarge the spread");

    cudaFree(d_g);
    cudaFree(d_u);
    cudaFree(d_o);
    std::printf("glm_ffn_parity: PASS\n");
    return 0;
}
