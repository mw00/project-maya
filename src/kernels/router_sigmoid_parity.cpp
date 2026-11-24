// src/kernels/router_sigmoid_parity.cpp - the glm5-next router vs ref/glm.py, plus the wrong readings.
//
// The misreadings this architecture invites (router_sigmoid.hpp, glm5-next.cpp's own comment
// "leave probs unbiased as it's later used to get expert weights"):
//
//   1. BIASED WEIGHTS.  The selection bias (ffn_exp_probs_b) enters the top-k ONLY; the weights
//      are the unbiased sigmoid values.  Using the biased scores for the weights is the natural
//      DeepSeek noaux_tc misreading and it changes the answer silently - the whole reason this
//      test exists.
//   2. SOFTMAX INSTEAD OF SIGMOID.  The qwen4exp reading, same shapes, plausible numbers.
//   3. NO RENORMALISATION, and 4. NO SCALE.  norm_topk_prob and routed_scaling_factor are config
//      flags; dropping either looks like a calibration issue, not a bug.
//   5. TIE ORDER.  Equal selection scores must pick the LOWER index first (stable descending
//      argsort); the rank formula in the kernel reproduces it, and a quantised-logits fixture
//      makes the ties happen on purpose.
#include "maya/kernels/router_sigmoid.hpp"

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

// ---- the reference, straight from ref/glm.py::moe_ffn + top_k_desc
//
// `biased_weights`, `softmax`, `no_renorm` and `no_scale` are the WRONG readings; the negative
// tests ask for them by name and require the kernel to differ from them materially.
void ref_router(const std::vector<float>& logits, const std::vector<float>& bias, int n_tokens, int n_expert,
                int k, float w_scale, bool norm_w, bool softmax, bool biased_weights, bool no_renorm,
                bool no_scale, std::vector<int>& ids, std::vector<float>& weights) {
    ids.assign((size_t) n_tokens * k, 0);
    weights.assign((size_t) n_tokens * k, 0.0f);
    std::vector<double> score(n_expert);
    for (int t = 0; t < n_tokens; ++t) {
        std::vector<double> p(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            const double x = logits[(size_t) t * n_expert + e];
            p[e] = softmax ? 0.0 : 1.0 / (1.0 + std::exp(-x));
        }
        if (softmax) {   // softmax over all experts, the qwen4exp reading
            double mx = -1e300, sum = 0.0;
            for (int e = 0; e < n_expert; ++e) mx = std::max(mx, (double) logits[(size_t) t * n_expert + e]);
            for (int e = 0; e < n_expert; ++e) {
                p[e] = std::exp((double) logits[(size_t) t * n_expert + e] - mx);
                sum += p[e];
            }
            for (int e = 0; e < n_expert; ++e) p[e] /= sum;
        }
        for (int e = 0; e < n_expert; ++e) {
            score[e] = p[e];
            if (!bias.empty()) score[e] += bias[e];   // the bias is a per-layer (n_expert,) parameter
        }
        // stable argsort of -score, ties by ascending index
        std::vector<int> order(n_expert);
        for (int e = 0; e < n_expert; ++e) order[e] = e;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return score[a] > score[b]; });
        double sum = 0.0;
        for (int i = 0; i < k; ++i) {
            ids[(size_t) t * k + i] = order[i];
            weights[(size_t) t * k + i] = (float) p[order[i]];
            sum += p[order[i]];
        }
        for (int i = 0; i < k; ++i) {
            double w = biased_weights ? weights[(size_t) t * k + i] + (bias.empty() ? 0.0 : bias[ids[(size_t) t * k + i]])
                                      : weights[(size_t) t * k + i];
            if (norm_w && !no_renorm) w /= std::max(sum, 6.103515625e-5);   // the oracle gates the renorm on norm_topk_prob
            if (!no_scale) w *= w_scale;
            weights[(size_t) t * k + i] = (float) w;
        }
    }
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "router_sigmoid_parity: %s\n", what.c_str());
        std::exit(1);
    }
}

// One fixture: build inputs, run the kernel, and hold it to the right reading while requiring the
// wrong ones to differ.  `quantize` creates exact selection-score ties on purpose.
void test_case(int n_tokens, int n_expert, int k, float w_scale, bool norm_w, bool with_bias, bool quantize,
               uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> logits((size_t) n_tokens * n_expert), bias;
    for (auto& v : logits) v = quantize ? std::round(nd(rng) * 4.0f) / 4.0f : nd(rng);
    if (with_bias) {
        std::normal_distribution<float> bd(0.0f, 0.1f);
        bias.resize(n_expert);   // a per-layer (n_expert,) parameter, broadcast across tokens
        for (auto& v : bias) v = quantize ? std::round(bd(rng) * 4.0f) / 4.0f : bd(rng);
    }

    float *d_logits, *d_bias = nullptr, *d_weights;
    int* d_ids;
    check(cudaMalloc(&d_logits, logits.size() * sizeof(float)), "logits");
    check(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "logits H2D");
    if (!bias.empty()) {
        check(cudaMalloc(&d_bias, bias.size() * sizeof(float)), "bias");
        check(cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice), "bias H2D");
    }    check(cudaMalloc(&d_ids, (size_t) n_tokens * k * sizeof(int)), "ids");
    check(cudaMalloc(&d_weights, (size_t) n_tokens * k * sizeof(float)), "weights");

    maya::kernels::router_sigmoid(d_logits, with_bias ? d_bias : nullptr, n_tokens, n_expert, k, w_scale,
                                    norm_w, d_ids, d_weights, nullptr);
    check(cudaGetLastError(), "router_sigmoid launch");
    std::vector<int> ids;
    std::vector<float> weights;
    ids.resize((size_t) n_tokens * k);
    weights.resize((size_t) n_tokens * k);
    check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(int), cudaMemcpyDeviceToHost), "ids D2H");
    check(cudaMemcpy(weights.data(), d_weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost), "w D2H");
    check(cudaDeviceSynchronize(), "sync");

    // ---- positive: the right reading, at f32-sigmoid tolerance
    std::vector<int> r_ids;
    std::vector<float> r_weights;
    ref_router(logits, bias, n_tokens, n_expert, k, w_scale, norm_w, false, false, false, false, r_ids, r_weights);
    require(ids == r_ids, "ids diverge from ref/glm.py (order or ties)");
    if (getenv("RS_DEBUG")) {
        std::fprintf(stderr, "debug ids: kernel");
        for (int i = 0; i < k; ++i) std::fprintf(stderr, " %d", ids[i]);
        std::fprintf(stderr, " | ref");
        for (int i = 0; i < k; ++i) std::fprintf(stderr, " %d", r_ids[i]);
        std::fprintf(stderr, "\ndebug w: kernel");
        for (int i = 0; i < k; ++i) std::fprintf(stderr, " %.7f", weights[i]);
        std::fprintf(stderr, " | ref");
        for (int i = 0; i < k; ++i) std::fprintf(stderr, " %.7f", r_weights[i]);
        std::fprintf(stderr, "\ndebug delta %.3e\n", max_abs(weights, r_weights));
    }
    require(max_abs(weights, r_weights) < 1e-5, "weights diverge from ref/glm.py");

    // ---- negative: the wrong readings must differ materially
    std::vector<int> w_ids;
    std::vector<float> w_weights;
    if (with_bias) {
        ref_router(logits, bias, n_tokens, n_expert, k, w_scale, norm_w, false, true, false, false, w_ids, w_weights);
        require(max_abs(weights, w_weights) > 1e-3, "biased weights are indistinguishable - enlarge the bias");
    }
    ref_router(logits, bias, n_tokens, n_expert, k, w_scale, norm_w, true, false, false, false, w_ids, w_weights);
    require(max_abs(weights, w_weights) > 1e-2, "sigmoid and softmax are indistinguishable");
    if (norm_w) {
        ref_router(logits, bias, n_tokens, n_expert, k, w_scale, norm_w, false, false, true, false, w_ids, w_weights);
        require(max_abs(weights, w_weights) > 1e-2, "renormalisation is invisible - enlarge k or spread");
    }
    if (w_scale != 1.0f) {
        ref_router(logits, bias, n_tokens, n_expert, k, w_scale, norm_w, false, false, false, true, w_ids, w_weights);
        require(max_abs(weights, w_weights) > 1e-2, "the scale is invisible - enlarge w_scale");
    }

    cudaFree(d_logits);
    cudaFree(d_bias);
    cudaFree(d_ids);
    cudaFree(d_weights);
}

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    if (!selftest) {
        std::fprintf(stderr, "router_sigmoid_parity: run with --selftest (the CTest form)\n");
        return 2;
    }
    test_case(4, 32, 8, 2.5f, true, true, false, 11u);    // the small shape, with bias
    test_case(1, 288, 8, 2.5f, true, true, false, 12u);   // the real routed shape
    test_case(4, 32, 8, 2.5f, true, true, true, 13u);     // quantised logits: exact ties, tie order matters
    test_case(4, 32, 8, 1.0f, false, false, false, 14u);  // no bias, no norm, scale 1: the bare contract
    std::printf("router_sigmoid_parity: PASS\n");
    return 0;
}
