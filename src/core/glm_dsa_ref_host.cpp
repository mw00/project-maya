// src/core/glm_dsa_ref_host.cpp - host-side pieces of the glm-dsa F32 reference forward (B3.5).  Contracts:
// include/strata/kernels/glmd_forward.hpp.  Plain C++, shared by src/core/glm_dsa_ref.cu and the CPU test.
#include "strata/kernels/glmd_forward.hpp"

#include "strata/kernels/glmd_ref.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace glmd {

void route_sigmoid(const float* logits, const float* bias, int n_expert, int k, bool norm, float scale, int* ids,
                   float* w) {
    std::vector<double> p((size_t) n_expert), sel((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        p[(size_t) e] = 1.0 / (1.0 + std::exp(-(double) logits[e]));
        sel[(size_t) e] = p[(size_t) e] + (bias ? (double) bias[e] : 0.0);
    }
    std::vector<int> order((size_t) n_expert);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[(size_t) a] > sel[(size_t) b]; });
    double sum = 0.0;
    for (int i = 0; i < k; ++i) sum += p[(size_t) order[(size_t) i]];
    const double div = norm ? sum + 1e-20 : 1.0;
    for (int i = 0; i < k; ++i) {
        ids[i] = order[(size_t) i];
        w[i] = (float) (p[(size_t) order[(size_t) i]] / div * (double) scale);
    }
}

float attn_scale(int qk_head) {
    if (qk_head == 256) return kGlm53KqScale;
    return (float) (1.0 / std::sqrt((double) qk_head));
}

bool kv_carve(int64_t lat, int64_t ik, int64_t ig, int64_t pool, int64_t lat_floats, int64_t ring, int64_t n_pools,
              int idx_key, int kv_row, int64_t max_ctx, bool full, KvCarve& out) {
    out = KvCarve{};
    if (lat < 0 || ik != lat + lat_floats || ig != ik + (int64_t) idx_key * ring ||
        pool != ig + (int64_t) idx_key * ring)
        return false;   // not one contiguous span
    const int64_t span = pool + (int64_t) idx_key * n_pools - lat;
    const int64_t need = (int64_t) kv_row * max_ctx + (full ? (int64_t) idx_key * max_ctx : 0);
    if (need > span) return false;
    out.kv = lat;
    out.ik = full ? lat + (int64_t) kv_row * max_ctx : -1;
    return true;
}

}  // namespace glmd
