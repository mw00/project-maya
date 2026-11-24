// src/kernels/glm_dsa_parity.cpp - the DSA indexer + absorbed-MLA attention vs ref/glm.py.
//
// The wrong readings (glm_dsa.hpp):
//   1. JOINT POOL SOFTMAX.  One softmax over the kpool members for the whole key-dim vector, instead
//      of PER KEY-DIM ELEMENT - the natural reading of "softmax the pool", and not what llama.cpp does.
//   2. UNWEIGHTED POOL.  A plain member average - plausible pooling, a different indexer.
//   3. NO RELU in the score.  The lightning indexer's relu is load-bearing; without it negative dots
//      subtract and the selection changes.
//   4. NO TAIL.  Dropping the incomplete tail (select_tail) hides the newest tokens from every query.
//   5. WRONG ATTENTION SCALE.  1/sqrt(kv_lora) instead of 1/sqrt(qk_nope) - the two are both "head
//      dims" of this layer and only one is the ggml kq_scale.
#include "maya/kernels/glm_dsa.hpp"

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

struct Fixture {
    int T, kpool, idx_heads, key_dim, kv_lora, n_head, v_head, qk_nope, top_pools, select_tail;
    std::vector<float> ik, ig, ape, iq, weights, q_abs, latents, wv_b;
    int n_pools() const { return T / kpool; }
    int n_sel() const { return kpool * top_pools + (select_tail ? kpool - 1 : 0); }
};

Fixture make_fixture(int T, int kpool, int idx_heads, int key_dim, int kv_lora, int n_head, int v_head,
                     int qk_nope, int top_pools, int select_tail, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Fixture f;
    f.T = T;
    f.kpool = kpool;
    f.idx_heads = idx_heads;
    f.key_dim = key_dim;
    f.kv_lora = kv_lora;
    f.n_head = n_head;
    f.v_head = v_head;
    f.qk_nope = qk_nope;
    f.top_pools = top_pools;
    f.select_tail = select_tail;
    for (size_t i = 0; i < (size_t) key_dim * T; ++i) {
        f.ik.push_back(nd(rng));
        f.ig.push_back(nd(rng) * 0.5f);
    }
    for (size_t i = 0; i < (size_t) key_dim * kpool; ++i) f.ape.push_back(nd(rng) * 0.3f);
    for (size_t i = 0; i < (size_t) key_dim * idx_heads * T; ++i) f.iq.push_back(nd(rng));
    for (size_t i = 0; i < (size_t) idx_heads * T; ++i) f.weights.push_back(nd(rng) * 0.2f);
    for (size_t i = 0; i < (size_t) kv_lora * n_head * T; ++i) f.q_abs.push_back(nd(rng));
    for (size_t i = 0; i < (size_t) kv_lora * T; ++i) f.latents.push_back(nd(rng));
    for (size_t i = 0; i < (size_t) kv_lora * v_head * n_head; ++i) f.wv_b.push_back(nd(rng) * 0.3f);
    return f;
}

// ---- the reference, straight from ref/glm.py::dsa_layer (f64 internally)

void ref_pool(const Fixture& f, std::vector<float>& pooled, bool joint_softmax = false, bool unweighted = false) {
    pooled.assign((size_t) f.key_dim * f.n_pools(), 0.0f);
    for (int p = 0; p < f.n_pools(); ++p)
        for (int e = 0; e < f.key_dim; ++e) {
            if (unweighted) {
                double acc = 0.0;
                for (int m = 0; m < f.kpool; ++m)
                    acc += f.ik[(size_t) e + f.key_dim * (p * f.kpool + m)];
                pooled[(size_t) e + f.key_dim * p] = (float) (acc / f.kpool);
                continue;
            }
            if (joint_softmax) {
                // ONE softmax over the pool's whole kpool*key_dim logit block: the weights then sum
                // to 1 per POOL, not per key-dim element - the natural "softmax the pool" reading
                double mx = -1e300;
                for (int m = 0; m < f.kpool; ++m)
                    for (int e2 = 0; e2 < f.key_dim; ++e2)
                        mx = std::max(mx, (double) f.ig[(size_t) e2 + f.key_dim * (p * f.kpool + m)] +
                                              (double) f.ape[(size_t) e2 + f.key_dim * m]);
                double denom = 0.0;
                for (int m = 0; m < f.kpool; ++m)
                    for (int e2 = 0; e2 < f.key_dim; ++e2)
                        denom += std::exp((double) f.ig[(size_t) e2 + f.key_dim * (p * f.kpool + m)] +
                                          (double) f.ape[(size_t) e2 + f.key_dim * m] - mx);
                double acc = 0.0;
                for (int m = 0; m < f.kpool; ++m)
                    acc += std::exp((double) f.ig[(size_t) e + f.key_dim * (p * f.kpool + m)] +
                                    (double) f.ape[(size_t) e + f.key_dim * m] - mx) /
                           denom * f.ik[(size_t) e + f.key_dim * (p * f.kpool + m)];
                pooled[(size_t) e + f.key_dim * p] = (float) acc;
                continue;
            }
            double mx = -1e300, denom = 0.0, acc = 0.0;
            std::vector<double> w(f.kpool);
            for (int m = 0; m < f.kpool; ++m) {
                w[m] = f.ig[(size_t) e + f.key_dim * (p * f.kpool + m)] + f.ape[(size_t) e + f.key_dim * m];
                mx = std::max(mx, w[m]);
            }
            for (int m = 0; m < f.kpool; ++m) {
                w[m] = std::exp(w[m] - mx);
                denom += w[m];
            }
            for (int m = 0; m < f.kpool; ++m)
                acc += w[m] / denom * f.ik[(size_t) e + f.key_dim * (p * f.kpool + m)];
            pooled[(size_t) e + f.key_dim * p] = (float) acc;
        }
}

void ref_score(const Fixture& f, const std::vector<float>& pooled, std::vector<float>& score, bool no_relu) {
    score.assign((size_t) f.n_pools() * f.T, 0.0f);
    for (int t = 0; t < f.T; ++t)
        for (int p = 0; p < f.n_pools(); ++p) {
            double acc = 0.0;
            for (int h = 0; h < f.idx_heads; ++h) {
                double dot = 0.0;
                for (int e = 0; e < f.key_dim; ++e)
                    dot += (double) f.iq[(size_t) e + f.key_dim * h + (size_t) f.key_dim * f.idx_heads * t] *
                           pooled[(size_t) e + f.key_dim * p];
                acc += (no_relu ? dot : std::max(dot, 0.0)) * (double) f.weights[(size_t) h + f.idx_heads * t];
            }
            score[(size_t) p + (size_t) f.n_pools() * t] = (float) acc;
        }
}

void ref_select(const Fixture& f, const std::vector<float>& score, std::vector<int>& cells,
                bool see_all_pools = false) {
    cells.assign((size_t) f.n_sel() * f.T, -1);
    for (int t = 0; t < f.T; ++t) {
        const int n_vis = see_all_pools ? f.n_pools() : (t + 1) / f.kpool;
        int* row = &cells[(size_t) f.n_sel() * t];
        for (int p = 0; p < n_vis; ++p) {
            const double v = score[(size_t) p + (size_t) f.n_pools() * t];
            int rank = 0;
            for (int q = 0; q < n_vis; ++q) {
                const double u = score[(size_t) q + (size_t) f.n_pools() * t];
                rank += u > v ? 1 : 0;
                rank += (q < p && u == v) ? 1 : 0;
            }
            if (rank < f.top_pools)
                for (int m = 0; m < f.kpool; ++m) row[rank * f.kpool + m] = p * f.kpool + m;
        }
        if (f.select_tail)
            for (int m = 0; m < f.kpool - 1; ++m) {
                const int cell = n_vis * f.kpool + m;
                if (cell <= t) row[f.top_pools * f.kpool + m] = cell;
            }
    }
}

void ref_attention(const Fixture& f, const std::vector<int>& cells, std::vector<float>& out,
                   bool wrong_scale = false) {
    out.assign((size_t) f.v_head * f.n_head * f.T, 0.0f);
    const double scale = wrong_scale ? 1.0 / std::sqrt((double) f.kv_lora) : 1.0 / std::sqrt((double) f.qk_nope);
    for (int t = 0; t < f.T; ++t)
        for (int h = 0; h < f.n_head; ++h) {
            const int* row = &cells[(size_t) f.n_sel() * t];
            std::vector<double> s(f.n_sel(), 0.0);
            std::vector<int> valid;
            for (int s_i = 0; s_i < f.n_sel(); ++s_i) {
                if (row[s_i] < 0) continue;
                double dot = 0.0;
                for (int e = 0; e < f.kv_lora; ++e)
                    dot += (double) f.q_abs[(size_t) e + f.kv_lora * h + (size_t) f.kv_lora * f.n_head * t] *
                           f.latents[(size_t) e + f.kv_lora * row[s_i]];
                s[s_i] = dot * scale;
                valid.push_back(s_i);
            }
            double mx = -1e300;
            for (int s_i : valid) mx = std::max(mx, s[s_i]);
            double denom = 0.0;
            for (int s_i : valid) {
                s[s_i] = std::exp(s[s_i] - mx);
                denom += s[s_i];
            }
            std::vector<double> ctx(f.kv_lora, 0.0);
            for (int s_i : valid)
                for (int e = 0; e < f.kv_lora; ++e)
                    ctx[e] += s[s_i] / denom * f.latents[(size_t) e + f.kv_lora * row[s_i]];
            for (int v = 0; v < f.v_head; ++v) {
                double acc = 0.0;
                for (int c = 0; c < f.kv_lora; ++c)
                    acc += (double) f.wv_b[(size_t) c + f.kv_lora * v + (size_t) f.kv_lora * f.v_head * h] * ctx[c];
                out[(size_t) v + f.v_head * h + (size_t) f.v_head * f.n_head * t] = (float) acc;
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
        std::fprintf(stderr, "glm_dsa_parity: %s\n", what.c_str());
        std::exit(1);
    }
}

struct Arena {
    float* d = nullptr;
    explicit Arena(size_t floats) { check(cudaMalloc(&d, floats * sizeof(float)), "cudaMalloc"); }
    ~Arena() { cudaFree(d); }
};

void test_case(int T, int kpool, int idx_heads, int key_dim, int kv_lora, int n_head, int v_head, int qk_nope,
               int top_pools, uint32_t seed) {
    constexpr int SELECT_TAIL = 1;   // the released config's index_kpool_always_select_tail
    const Fixture f = make_fixture(T, kpool, idx_heads, key_dim, kv_lora, n_head, v_head, qk_nope, top_pools,
                                   SELECT_TAIL, seed);
    const int n_pools = f.n_pools(), n_sel = f.n_sel();

    // device arena: ik | ig | ape | iq | weights | pooled | score | q_abs | latents | wv_b | cells | out
    const size_t cf = (size_t) f.key_dim * f.T, qf = (size_t) f.key_dim * f.idx_heads * f.T;
    Arena a(cf + cf + (size_t) f.key_dim * kpool + qf + (size_t) idx_heads * T + cf + (size_t) n_pools * T +
            (size_t) kv_lora * n_head * T + (size_t) kv_lora * T + (size_t) kv_lora * v_head * n_head +
            (size_t) n_sel * T + (size_t) v_head * n_head * T);
    size_t off = 0;
    auto put = [&](const std::vector<float>& v) {
        check(cudaMemcpy(a.d + off, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D");
        float* p = a.d + off;
        off += v.size();
        return p;
    };
    const float* d_ik = put(f.ik);
    const float* d_ig = put(f.ig);
    const float* d_ape = put(f.ape);
    const float* d_iq = put(f.iq);
    const float* d_weights = put(f.weights);
    float* d_pooled = a.d + off; off += cf;
    float* d_score = a.d + off; off += (size_t) n_pools * T;
    const float* d_qabs = put(f.q_abs);
    const float* d_lat = put(f.latents);
    const float* d_wvb = put(f.wv_b);
    int* d_cells = (int*) (a.d + off); off += (size_t) n_sel * T;
    float* d_out = a.d + off; off += (size_t) v_head * n_head * T;

    // ---- pool
    maya::kernels::glm_dsa_pool(d_ik, d_ig, d_ape, key_dim, kpool, n_pools, d_pooled, nullptr);
    check(cudaGetLastError(), "pool launch");
    std::vector<float> k_pooled((size_t) key_dim * n_pools);
    check(cudaMemcpy(k_pooled.data(), d_pooled, k_pooled.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "pooled D2H");
    std::vector<float> r_pooled;
    ref_pool(f, r_pooled);
    require(max_abs(k_pooled, r_pooled) < 1e-5, "pool diverges from ref/glm.py");
    std::vector<float> w_pooled;
    ref_pool(f, w_pooled, true);
    require(max_abs(r_pooled, w_pooled) > 1e-3, "the per-key-dim softmax is invisible - enlarge ig/ape");
    ref_pool(f, w_pooled, false, true);
    require(max_abs(r_pooled, w_pooled) > 1e-3, "the pool weights are invisible");

    // ---- score
    maya::kernels::glm_dsa_score(d_iq, d_pooled, d_weights, key_dim, idx_heads, T, n_pools, d_score, nullptr);
    check(cudaGetLastError(), "score launch");
    std::vector<float> k_score((size_t) n_pools * T);
    check(cudaMemcpy(k_score.data(), d_score, k_score.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "score D2H");
    std::vector<float> r_score;
    ref_score(f, r_pooled, r_score, false);
    require(max_abs(k_score, r_score) < 1e-4, "score diverges from ref/glm.py");
    std::vector<float> w_score;
    ref_score(f, r_pooled, w_score, true);
    require(max_abs(r_score, w_score) > 1e-3, "the score's relu is invisible");

    // ---- select
    std::vector<int> ident(T);
    for (int t = 0; t < T; ++t) ident[t] = t;
    int* d_ident = nullptr;   // its own allocation: the cells region ends the useful arena
    check(cudaMalloc(&d_ident, T * sizeof(int)), "ident");
    check(cudaMemcpy(d_ident, ident.data(), T * sizeof(int), cudaMemcpyHostToDevice), "ident H2D");
    maya::kernels::glm_dsa_select(d_score, n_pools, kpool, top_pools, SELECT_TAIL, T, n_sel, d_ident,
                                    d_cells, nullptr);
    check(cudaGetLastError(), "select launch");
    std::vector<int> k_cells((size_t) n_sel * T);
    check(cudaMemcpy(k_cells.data(), d_cells, k_cells.size() * sizeof(int), cudaMemcpyDeviceToHost), "cells D2H");
    std::vector<int> r_cells;
    ref_select(f, r_score, r_cells);
    require(k_cells == r_cells, "selection diverges from ref/glm.py (order, ties or tail)");
    {
        Fixture no_tail = f;
        no_tail.select_tail = 0;
        std::vector<int> w_cells;
        ref_select(no_tail, r_score, w_cells);
        require(k_cells != w_cells, "the incomplete tail is invisible - make T a non-multiple of kpool");
    }

    // ---- attention
    cudaFree(d_ident);
    maya::kernels::glm_mla_attention(d_qabs, d_lat, d_cells, d_wvb, kv_lora, n_head, v_head, qk_nope, T, n_sel,
                                       d_out, nullptr);
    check(cudaGetLastError(), "attention launch");
    std::vector<float> k_out((size_t) v_head * n_head * T);
    check(cudaMemcpy(k_out.data(), d_out, k_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "out D2H");
    std::vector<float> r_out;
    ref_attention(f, r_cells, r_out);
    require(max_abs(k_out, r_out) < 1e-4, "attention diverges from ref/glm.py");
    std::vector<float> w_out;
    ref_attention(f, r_cells, w_out, true);
    require(max_abs(r_out, w_out) > 1e-2, "the attention scale is indistinguishable - enlarge q_abs");
}

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    if (!selftest) {
        std::fprintf(stderr, "glm_dsa_parity: run with --selftest (the CTest form)\n");
        return 2;
    }
    test_case(7, 2, 4, 16, 16, 4, 8, 12, 2, 31u);    // the synthetic model's shape, T not a kpool multiple
    test_case(11, 4, 8, 32, 32, 8, 16, 24, 3, 32u);  // closer to the real ratios, tail exercised
    std::printf("glm_dsa_parity: PASS\n");
    return 0;
}
