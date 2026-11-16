// src/core/glm_pool_classes.cpp - glm-dsa B3.7: the slow-path expert pool's size classes (glm_pool_classes.hpp).
#include "strata/core/glm_pool_classes.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace strata::core {

namespace {
std::atomic<int> g_last{-1};
}

int glm_pool_class_cap(const char* env_value) {
    if (env_value == nullptr || env_value[0] == '\0') return 0;
    char* end = nullptr;
    const long v = std::strtol(env_value, &end, 10);
    if (end == env_value || *end != '\0' || v <= 0 || v > 1 << 20) return 0;
    return (int) v;
}

int glm_pool_merge_classes(std::vector<int>& layer_class, std::vector<size_t>& stride, std::vector<size_t>& weight,
                           std::vector<int>& layers, int cap) {
    const int n = (int) stride.size();
    if (weight.size() != stride.size() || layers.size() != stride.size()) {
        std::fprintf(stderr, "glm pool classes: inconsistent class vectors (%zu / %zu / %zu) - not merged\n",
                     stride.size(), weight.size(), layers.size());
        g_last.store(n);
        return n;
    }
    if (cap > 0 && n > cap) {
        const int keep = cap - 1;   // classes [0, keep) stay; [keep, n) fold into class `keep`
        size_t s = 0, w = 0;
        int l = 0;
        for (int c = keep; c < n; ++c) {
            s = std::max(s, stride[(size_t) c]);
            w += weight[(size_t) c];
            l += layers[(size_t) c];
        }
        stride.resize((size_t) cap);
        weight.resize((size_t) cap);
        layers.resize((size_t) cap);
        stride[(size_t) keep] = s;
        weight[(size_t) keep] = w;
        layers[(size_t) keep] = l;
        for (int& c : layer_class)
            if (c > keep) c = keep;
        std::fprintf(stderr, "glm pool classes: %d expert size classes folded to %d (MAYA_GLM_POOL_CLASSES)\n", n,
                     cap);
    }
    const int out = (int) stride.size();
    g_last.store(out);
    return out;
}

int glm_pool_classes_last() { return g_last.load(); }

}  // namespace strata::core
