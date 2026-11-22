// src/kernels/glm_kda_parity.cpp - the glm5-next KDA pieces vs ref/glm.py, plus the wrong readings.
//
// Same discipline as gr_parity and glm_hc_parity: every wrong reading is computed first and required
// to differ materially.  The readings a GDN-native engine gets wrong on KDA (glm_kda.hpp):
//
//   1. SCALAR GATE.  The qwen4exp GDN decays the state by ONE exp(gate) per head; KDA's decay is
//      PER-CHANNEL on the key axis (ggml kernel: "S[i][:] *= exp(g[i])").  Averaging the channel
//      gate into a per-head scalar is the natural port and is wrong.
//   2. VALUE-AXIS GATE.  Scaling the COLUMNS (j) instead of the rows (i) - the two state axes are
//      both head_dim wide, so the shapes agree and the answer is plausible.
//   3. NO DELTA CORRECTION.  S += k (beta*v)^T without the -S^T k term is plain linear attention:
//      same shapes, same scales, a different model.  This is THE delta-net negative.
//   4. SOFTPLUS FORM.  The qwen4exp gate is softplus(g) * A (unbounded above); KDA's released form
//      is lower_bound * sigmoid(-g * A), bounded in (lb, 0).  Swapping them changes every decay.
//   5. MISSING OUTPUT SCALE.  ggml's fused kernel bakes 1/sqrt(dv) into o; leaving it out is a
//      uniform 11.3x on the real head_dim and looks like a temperature change.
//   6. TAP ORDER in the conv: the taps apply OLDEST FIRST (ggml_ssm_conv's contract); reversing
//      them passes every shape check and produces a different filter.
//
// The positive comparison holds the f32 kernels to ref/glm.py's f64 loop at f32 tolerance, over two
// geometries (the synthetic model's, and the real model's head_dim with fewer heads).
#include "maya/kernels/glm_kda.hpp"

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

double dsigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Fixture {
    int n_head, head_dim, d_conv, T;
    float norm_eps, lower_bound;
    // layouts per glm_kda.hpp
    std::vector<float> x_proj;      // (d_inner, T): c + d_inner*t
    std::vector<float> conv_w;      // (d_conv, d_inner): tap + d_conv*c
    std::vector<float> state_in;    // (hist, d_inner): j + hist*c
    std::vector<float> q, k, v, g;  // g is (d_inner, T); q/k/v (head_dim, n_head, T)
    std::vector<float> ssm_a;       // (n_head,), negative
    std::vector<float> beta;        // (n_head, T)
    std::vector<float> norm_w, g2;  // (head_dim,), (head_dim, n_head, T)
    size_t d_inner() const { return (size_t) n_head * head_dim; }
    size_t hist() const { return (size_t) d_conv - 1; }
};

Fixture make_fixture(int n_head, int head_dim, int T, float lower_bound, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    Fixture f;
    f.n_head = n_head;
    f.head_dim = head_dim;
    f.d_conv = 4;
    f.T = T;
    f.norm_eps = 1e-5f;
    f.lower_bound = lower_bound;
    const size_t di = f.d_inner();
    for (size_t i = 0; i < di * T; ++i) f.x_proj.push_back(nd(rng));
    for (size_t i = 0; i < (size_t) f.d_conv * di; ++i) f.conv_w.push_back(nd(rng) * 0.5f);
    for (size_t i = 0; i < f.hist() * di; ++i) f.state_in.push_back(nd(rng) * 0.5f);
    for (size_t i = 0; i < (size_t) head_dim * n_head * T; ++i) {
        f.q.push_back(nd(rng));
        f.k.push_back(nd(rng));
        f.v.push_back(nd(rng));
        f.g2.push_back(nd(rng));
    }
    for (size_t i = 0; i < di * T; ++i) f.g.push_back(nd(rng));
    for (int h = 0; h < n_head; ++h) f.ssm_a.push_back(-std::exp(0.5f + nd(rng) * 0.3f));
    for (size_t i = 0; i < (size_t) n_head * T; ++i) f.beta.push_back(ud(rng));
    for (int e = 0; e < head_dim; ++e) f.norm_w.push_back(1.0f + nd(rng) * 0.02f);
    return f;
}

// ---- the reference, straight from ref/glm.py (f64 internally)

void ref_conv(const Fixture& f, std::vector<float>& out, std::vector<float>& state_out, bool taps_reversed) {
    const int hist = f.d_conv - 1;
    out.assign(f.d_inner() * f.T, 0.0f);
    state_out.assign(f.hist() * f.d_inner(), 0.0f);
    for (size_t c = 0; c < f.d_inner(); ++c) {
        for (int t = 0; t < f.T; ++t) {
            double acc = 0.0;
            for (int tap = 0; tap < f.d_conv; ++tap) {
                const int tap_eff = taps_reversed ? f.d_conv - 1 - tap : tap;
                const int src = t - hist + tap_eff;
                const double x = src >= 0 ? (double) f.x_proj[(size_t) src * f.d_inner() + c]
                                          : (double) f.state_in[(size_t)(src + hist) + f.hist() * c];
                acc += (double) f.conv_w[(size_t) tap + (size_t) f.d_conv * c] * x;
            }
            out[(size_t) t * f.d_inner() + c] = (float) (acc / (1.0 + std::exp(-acc)));
        }
        for (int j = 0; j < hist; ++j) {
            const int src = f.T - hist + j;
            state_out[(size_t) j + f.hist() * c] =
                src >= 0 ? f.x_proj[(size_t) src * f.d_inner() + c] : f.state_in[(size_t)(src + hist) + f.hist() * c];
        }
    }
}

void ref_gate(const Fixture& f, std::vector<float>& g1, bool softplus_form) {
    g1.assign((size_t) f.head_dim * f.n_head * f.T, 0.0f);
    for (int h = 0; h < f.n_head; ++h)
        for (int e = 0; e < f.head_dim; ++e)
            for (int t = 0; t < f.T; ++t) {
                const double v = f.g[(size_t)(e + f.head_dim * h) + (size_t) t * f.d_inner()];
                double out;
                if (softplus_form) {
                    out = std::log1p(std::exp((double) v)) * (double) f.ssm_a[h];   // the qwen4exp form
                } else {
                    out = (double) f.lower_bound * dsigmoid(-(double) v * (double) f.ssm_a[h]);
                }
                g1[((size_t) t * f.n_head + h) * f.head_dim + e] = (float) out;
            }
}

// `scalar_gate`, `value_axis`, `no_delta` and `no_scale` are the WRONG readings - the negative tests
// require the kernel to differ from them materially.
void ref_recurrence(const Fixture& f, const std::vector<float>& g1, std::vector<float>& out,
                    bool scalar_gate = false, bool value_axis = false, bool no_delta = false,
                    bool no_scale = false) {
    const int dk = f.head_dim;
    out.assign((size_t) dk * f.n_head * f.T, 0.0f);
    std::vector<double> S((size_t) dk * dk, 0.0);
    const double scale = no_scale ? 1.0 : 1.0 / std::sqrt((double) dk);
    for (int h = 0; h < f.n_head; ++h) {
        std::fill(S.begin(), S.end(), 0.0);
        double scalar_decay = 0.0;
        for (int t = 0; t < f.T; ++t) {
            const double b = f.beta[(size_t) t * f.n_head + h];
            if (scalar_gate) {   // the GDN reading: ONE decay per head, the channel gate averaged
                double m = 0.0;
                for (int e = 0; e < dk; ++e) m += g1[((size_t) t * f.n_head + h) * dk + e];
                scalar_decay = std::exp(m / dk);
            }
            for (int i = 0; i < dk; ++i)
                for (int j = 0; j < dk; ++j) {
                    const double decay = scalar_gate ? scalar_decay
                                                     : std::exp((double) g1[((size_t) t * f.n_head + h) * dk +
                                                                            (value_axis ? j : i)]);
                    S[(size_t) i * dk + j] *= decay;
                }
            std::vector<double> u(dk, 0.0), delta(dk, 0.0);
            for (int j = 0; j < dk; ++j) {
                for (int i = 0; i < dk; ++i)
                    u[j] += S[(size_t) i * dk + j] * (double) f.k[((size_t) t * f.n_head + h) * dk + i];
                delta[j] = b * ((double) f.v[((size_t) t * f.n_head + h) * dk + j] - u[j]);
            }
            for (int i = 0; i < dk; ++i)
                for (int j = 0; j < dk; ++j)
                    S[(size_t) i * dk + j] += (double) f.k[((size_t) t * f.n_head + h) * dk + i] *
                                              (no_delta ? b * (double) f.v[((size_t) t * f.n_head + h) * dk + j]
                                                        : delta[j]);
            for (int j = 0; j < dk; ++j) {
                double acc = 0.0;
                for (int i = 0; i < dk; ++i)
                    acc += S[(size_t) i * dk + j] * (double) f.q[((size_t) t * f.n_head + h) * dk + i];
                out[((size_t) t * f.n_head + h) * dk + j] = (float) (acc * scale);
            }
        }
    }
}

void ref_out_gate(const Fixture& f, const std::vector<float>& o, std::vector<float>& gated, bool no_gate) {
    gated.assign((size_t) f.head_dim * f.n_head * f.T, 0.0f);
    for (int h = 0; h < f.n_head; ++h)
        for (int t = 0; t < f.T; ++t) {
            const float* ot = &o[((size_t) t * f.n_head + h) * f.head_dim];
            double ms = 0.0;
            for (int e = 0; e < f.head_dim; ++e) ms += (double) ot[e] * ot[e];
            const double inv = 1.0 / std::sqrt(ms / f.head_dim + f.norm_eps);
            for (int e = 0; e < f.head_dim; ++e) {
                const double gate = no_gate ? 1.0 : dsigmoid((double) f.g2[((size_t) t * f.n_head + h) * f.head_dim + e]);
                gated[((size_t) t * f.n_head + h) * f.head_dim + e] =
                    (float) ((double) ot[e] * inv * (double) f.norm_w[e] * gate);
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
        std::fprintf(stderr, "glm_kda_parity: %s\n", what.c_str());
        std::exit(1);
    }
}

struct Arena {
    float* d = nullptr;
    size_t floats = 0;
    explicit Arena(size_t n) : floats(n) { check(cudaMalloc(&d, n * sizeof(float)), "cudaMalloc"); }
    ~Arena() { cudaFree(d); }
};

void test_case(int n_head, int head_dim, int T, float lower_bound, uint32_t seed) {
    const Fixture f = make_fixture(n_head, head_dim, T, lower_bound, seed);
    const size_t di = f.d_inner(), dk = head_dim, seq = (size_t) dk * n_head * T;

    // device arena: x_proj | conv_w | conv_state | conv_out | q | k | v | g | ssm_a | beta |
    //               rec_state | g1 | rec | norm_w | g2 | gated
    // The conv state and the recurrence's S state are SEPARATE states in the real model - the conv
    // slides its window, the recurrence starts a fresh sequence from S = 0 (ref/glm.py's zeros()).
    Arena a(di * T + (size_t) f.d_conv * di + f.hist() * di + di * T + 3 * seq + di * T + n_head +
            (size_t) n_head * T + dk * dk * n_head + seq + seq + head_dim + seq + seq);
    size_t off = 0;
    auto put = [&](const std::vector<float>& v) {
        check(cudaMemcpy(a.d + off, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D");
        float* p = a.d + off;
        off += v.size();
        return p;
    };
    const float* d_x = put(f.x_proj);
    const float* d_w = put(f.conv_w);
    float* d_conv_state = put(f.state_in);
    float* d_conv_out = a.d + off; off += di * T;
    const float* d_q = put(f.q);
    const float* d_k = put(f.k);
    const float* d_v = put(f.v);
    const float* d_g = put(f.g);
    const float* d_a = put(f.ssm_a);
    const float* d_beta = put(f.beta);
    float* d_rec_state = a.d + off; off += (size_t) dk * dk * n_head;   // zeros: a fresh sequence
    check(cudaMemset(d_rec_state, 0, dk * dk * n_head * sizeof(float)), "zero recurrence state");
    float* d_g1 = a.d + off; off += seq;
    float* d_rec = a.d + off; off += seq;
    const float* d_nw = put(f.norm_w);
    const float* d_g2 = put(f.g2);
    float* d_gated = a.d + off; off += seq;

    // ---- conv (reads the history; the slide moves it forward for the next call)
    maya::kernels::glm_kda_conv(d_x, d_w, d_conv_state, (int) di, f.d_conv, T, d_conv_out, nullptr);
    check(cudaGetLastError(), "conv launch");
    std::vector<float> r_conv, r_state;
    ref_conv(f, r_conv, r_state, false);
    std::vector<float> k_conv(di * T), k_state(f.hist() * di);
    check(cudaMemcpy(k_conv.data(), d_conv_out, k_conv.size() * sizeof(float), cudaMemcpyDeviceToHost), "conv D2H");
    check(cudaMemcpy(k_state.data(), d_conv_state, k_state.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "conv state D2H");
    require(max_abs(k_conv, r_conv) < 5e-5, "conv diverges from ref/glm.py");
    require(max_abs(k_state, r_state) < 1e-6, "conv state slide diverges from ref/glm.py");
    std::vector<float> w_conv, w_state;
    ref_conv(f, w_conv, w_state, true);
    require(max_abs(r_conv, w_conv) > 1e-2, "the conv tap order is indistinguishable - enlarge conv_w");

    // ---- gate
    maya::kernels::glm_kda_gate(d_g, d_a, f.lower_bound, head_dim, n_head, T, d_g1, nullptr);
    check(cudaGetLastError(), "gate launch");
    std::vector<float> k_g1(seq);
    check(cudaMemcpy(k_g1.data(), d_g1, k_g1.size() * sizeof(float), cudaMemcpyDeviceToHost), "g1 D2H");
    std::vector<float> r_g1;
    ref_gate(f, r_g1, false);
    require(max_abs(k_g1, r_g1) < 1e-6, "gate diverges from ref/glm.py");
    std::vector<float> w_g1;
    ref_gate(f, w_g1, true);
    require(max_abs(r_g1, w_g1) > 1e-2, "the lower-bound and softplus forms are indistinguishable");

    // ---- recurrence (the gate kernel's own output feeds it, like the engine's graph)
    maya::kernels::glm_kda_recurrence(d_q, d_k, d_v, d_g1, d_beta, d_rec_state, n_head, head_dim, T, d_rec,
                                        nullptr);
    check(cudaGetLastError(), "recurrence launch");
    std::vector<float> k_rec(seq);
    check(cudaMemcpy(k_rec.data(), d_rec, k_rec.size() * sizeof(float), cudaMemcpyDeviceToHost), "rec D2H");
    std::vector<float> r_rec;
    ref_recurrence(f, r_g1, r_rec);
    require(max_abs(k_rec, r_rec) < 2e-4,
            "recurrence diverges from ref/glm.py (max error " + std::to_string(max_abs(k_rec, r_rec)) + ")");

    // the state must have advanced: a fresh sequence's zero state is gone after T tokens
    std::vector<float> k_rec_state((size_t) dk * dk * n_head);
    check(cudaMemcpy(k_rec_state.data(), d_rec_state, k_rec_state.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "rec state D2H");
    double state_abs = 0.0;
    for (float v : k_rec_state) state_abs = std::max(state_abs, (double) std::fabs(v));
    require(state_abs > 1e-3, "the recurrence left the state untouched");

    std::vector<float> w_rec;
    ref_recurrence(f, r_g1, w_rec, true);
    require(max_abs(r_rec, w_rec) > 1e-3, "per-channel and scalar gates are indistinguishable");
    ref_recurrence(f, r_g1, w_rec, false, true);
    require(max_abs(r_rec, w_rec) > 1e-3, "key-axis and value-axis gates are indistinguishable");
    ref_recurrence(f, r_g1, w_rec, false, false, true);
    require(max_abs(r_rec, w_rec) > 1e-3, "the delta correction is invisible - enlarge k or beta");
    ref_recurrence(f, r_g1, w_rec, false, false, false, true);
    require(max_abs(r_rec, w_rec) > 1e-2, "the 1/sqrt(dv) output scale is invisible");

    // ---- out gate
    maya::kernels::glm_kda_out_gate(d_rec, d_nw, d_g2, head_dim, n_head, T, f.norm_eps, d_gated, nullptr);
    check(cudaGetLastError(), "out_gate launch");
    std::vector<float> k_gated(seq);
    check(cudaMemcpy(k_gated.data(), d_gated, k_gated.size() * sizeof(float), cudaMemcpyDeviceToHost), "gated D2H");
    std::vector<float> r_gated;
    ref_out_gate(f, r_rec, r_gated, false);
    require(max_abs(k_gated, r_gated) < 2e-4, "out_gate diverges from ref/glm.py");
    std::vector<float> w_gated;
    ref_out_gate(f, r_rec, w_gated, true);
    require(max_abs(r_gated, w_gated) > 1e-2, "the output gate is invisible - enlarge g2");
}

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    if (!selftest) {
        std::fprintf(stderr, "glm_kda_parity: run with --selftest (the CTest form)\n");
        return 2;
    }
    test_case(4, 32, 8, -5.0f, 21u);    // the synthetic model's shape
    test_case(2, 128, 4, -5.0f, 22u);   // the real model's head_dim
    test_case(4, 32, 8, -2.0f, 23u);    // a milder lower bound
    std::printf("glm_kda_parity: PASS\n");
    return 0;
}
