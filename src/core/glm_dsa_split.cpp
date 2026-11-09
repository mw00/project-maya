// src/core/glm_dsa_split.cpp - the glm-dsa port B6.1-B6.3: the glm-dsa layer split's pure host-side rules
// (include/strata/core/glm_dsa_split.hpp).  No CUDA, no I/O: tests/glm_dsa/glm_dsa_split_test.cpp runs them on the CPU.
#include "strata/core/glm_dsa_split.hpp"
#include "strata/core/glm_state_layout.hpp"
#include "glm_tier_plan.hpp"   // lane_pool_mode: the runtime's STRATA_GLM_CPU_SHARED rule

#include <algorithm>
#include <cstdlib>
#include <limits>

namespace strata::core {

namespace {

bool dsa(const Glm5Geometry& g) { return g.arch == GlmArch::GlmDsa; }

// the layers a part may start on, rising
std::vector<int> valid_boundaries(const Glm5Geometry& g) {
    std::vector<int> v;
    for (int il = 1; il < g.n_layers; ++il)
        if (g.boundary_ok(il)) v.push_back(il);
    return v;
}

}  // namespace

int64_t glm_dsa_state_row_bytes(const Glm5Geometry& g, int layer, bool int8, bool f32_latent) {
    if (!dsa(g) || layer < 0 || layer >= g.n_layers + g.nextn) return 0;
    int64_t row = 0;
    if (f32_latent)
        row = 4 * (int64_t) (g.kv_lora + g.qk_rope);
    else if (int8)   // glm_fast.hpp lat8_rec_bytes: kv_lora codes + an FP16 scale per 32 (544 B), then FP16 k_pe
        row = (int64_t) g.kv_lora + g.kv_lora / 16 + 2 * (int64_t) g.qk_rope;
    else
        row = 2 * (int64_t) (g.kv_lora + g.qk_rope);
    // index keys on full-indexer layers only: F32, or FP16 on the fast path with STRATA_GLM_IDX_FP16 (B10.8)
    if (g.idx_full(layer)) row += (!f32_latent && glm_state_idx_fp16() ? 2 : 4) * (int64_t) g.idx_key;
    return row;
}

double glm_dsa_state_bytes(const Glm5Geometry& g, int layer, int64_t ctx, bool int8, bool f32_latent) {
    return (double) glm_dsa_state_row_bytes(g, layer, int8, f32_latent) * (double) std::max<int64_t>(0, ctx);
}

bool glm_split_bounds_full(const Glm5Geometry& g, const std::vector<int>& bounds) {
    if (!dsa(g)) return true;
    for (int b : bounds)
        if (!g.boundary_ok(b)) return false;
    return true;
}

void glm_nearest_boundaries(const Glm5Geometry& g, int il, int& below, int& above) {
    below = above = -1;
    for (int v : valid_boundaries(g)) {
        if (v <= il) below = v;
        if (v >= il && above < 0) above = v;
    }
}

std::string glm_nearest_boundaries_text(const Glm5Geometry& g, int il) {
    int lo = -1, hi = -1;
    glm_nearest_boundaries(g, il, lo, hi);
    if (lo >= 0 && hi >= 0 && lo != hi) return std::to_string(lo) + " or " + std::to_string(hi);
    if (lo >= 0) return std::to_string(lo);
    if (hi >= 0) return std::to_string(hi);
    return std::string();
}

std::vector<int> glm_snap_bounds(const Glm5Geometry& g, const std::vector<int>& bounds) {
    if (!dsa(g) || bounds.empty() || glm_split_bounds_full(g, bounds)) return bounds;
    const std::vector<int> v = valid_boundaries(g);
    const int m = (int) bounds.size(), V = (int) v.size();
    if (V < m) return bounds;
    // cost[i][j]: the least total distance placing bounds[0..i] with bounds[i] on v[j] (rising); a strict < on the
    // lower j first keeps ties on the lower layers
    constexpr double kInf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> cost((size_t) m, std::vector<double>((size_t) V, kInf));
    std::vector<std::vector<int>> from((size_t) m, std::vector<int>((size_t) V, -1));
    for (int j = 0; j < V; ++j) cost[0][(size_t) j] = std::abs(v[(size_t) j] - bounds[0]);
    for (int i = 1; i < m; ++i) {
        double best = kInf;
        int arg = -1;
        for (int j = 0; j < V; ++j) {
            if (j > 0 && cost[(size_t) i - 1][(size_t) j - 1] < best) {
                best = cost[(size_t) i - 1][(size_t) j - 1];
                arg = j - 1;
            }
            if (arg < 0) continue;
            cost[(size_t) i][(size_t) j] = best + std::abs(v[(size_t) j] - bounds[(size_t) i]);
            from[(size_t) i][(size_t) j] = arg;
        }
    }
    int j = -1;
    for (int k = 0; k < V; ++k)
        if (cost[(size_t) m - 1][(size_t) k] < kInf && (j < 0 || cost[(size_t) m - 1][(size_t) k] < cost[(size_t) m - 1][(size_t) j]))
            j = k;
    if (j < 0) return bounds;
    std::vector<int> out((size_t) m);
    for (int i = m - 1; i >= 0; --i) {
        out[(size_t) i] = v[(size_t) j];
        j = from[(size_t) i][(size_t) j];
    }
    return out;
}

bool glm_split_prices_mtp(const Glm5Geometry& g, int n_parts) { return n_parts == 2 || (dsa(g) && n_parts >= 2); }

std::vector<int> glm_cpu_pool_ids(const Glm5Geometry& g, int n_parts, const char* shared_env,
                                  const std::vector<int>& nodes, const std::vector<int>& part_node) {
    const int n = std::max(0, n_parts);
    std::vector<int> id((size_t) n, 0);
    const glmtier::LanePool m = glmtier::lane_pool_mode(n, false, shared_env, false, (int) nodes.size(), dsa(g));
    for (int i = 0; i < n; ++i) {
        if (m == glmtier::LanePool::PerPart) {
            id[(size_t) i] = i;
        } else if (m == glmtier::LanePool::Socket) {   // glm_tier_numa.cu lane_pools: the GPU's node, else round robin
            const int gn = (size_t) i < part_node.size() ? part_node[(size_t) i] : -1;
            const bool online = std::find(nodes.begin(), nodes.end(), gn) != nodes.end();
            id[(size_t) i] = online ? gn : nodes[(size_t) i % nodes.size()];
        }
    }
    return id;
}

int glm_cpu_pool_count(const std::vector<int>& pool_ids) {
    std::vector<int> v = pool_ids;
    std::sort(v.begin(), v.end());
    return (int) (std::unique(v.begin(), v.end()) - v.begin());
}

double glm_split_miss_bps(const Glm5Geometry& g, double host_bps, int n_parts, int n_pools, bool pipelined) {
    const double h = host_bps > 0 ? host_bps : 2e10;
    const int n = std::max(1, n_parts);
    if (dsa(g) && !pipelined && n_pools >= 1 && n_pools < n) return std::max(1e9, h / n_pools);
    return std::max(1e9, h / n);
}

std::vector<int> glm_part_groups(const std::vector<int>& group_node, const std::vector<int>& gpu_node, int n,
                                 bool by_node_over) {
    const int G = (int) group_node.size();
    if (G == 0 || n < 1) return {};
    std::vector<int> grp((size_t) n, -1);
    const bool known = (int) gpu_node.size() == n;
    const auto group_of = [&](int node) {
        int at = -1;
        for (int j = 0; j < G; ++j)
            if (group_node[(size_t) j] >= 0 && group_node[(size_t) j] == node) at = j;
        return at;
    };
    if (n <= G) {   // Maya's rule: distinct known nodes, else i % G
        bool by_gpu = known;
        std::vector<bool> taken((size_t) G, false);
        for (int i = 0; by_gpu && i < n; ++i) {
            const int at = group_of(gpu_node[(size_t) i]);
            by_gpu = at >= 0 && !taken[(size_t) at];
            if (by_gpu) {
                taken[(size_t) at] = true;
                grp[(size_t) i] = at;
            }
        }
        if (!by_gpu)
            for (int i = 0; i < n; ++i) grp[(size_t) i] = i % G;
        return grp;
    }
    // more parts than groups (glm-dsa B6.3): by the GPU's node when every node is known and every group is used
    // (by_node_over false - every model but glm-dsa - keeps Maya's i % G)
    bool by_gpu = known && by_node_over;
    std::vector<bool> used((size_t) G, false);
    for (int i = 0; by_gpu && i < n; ++i) {
        const int at = group_of(gpu_node[(size_t) i]);
        by_gpu = at >= 0;
        if (by_gpu) {
            grp[(size_t) i] = at;
            used[(size_t) at] = true;
        }
    }
    for (int j = 0; by_gpu && j < G; ++j) by_gpu = used[(size_t) j];
    if (!by_gpu)
        for (int i = 0; i < n; ++i) grp[(size_t) i] = i % G;
    return grp;
}

}  // namespace strata::core
