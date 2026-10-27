// src/mm3/mm3_cands.hpp - a row of logits as the sampler's candidates on the host: what k::topk_rows (mm3_kernels.cuh)
// leaves on the device, for the engines without it and a row with more ties than its room.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace mm3 {

/// The kc largest of row [V] and every one equal to the kc-th, by logit desc then id asc ((id, logit) pairs); ex the
/// extra ids' logits.
inline void host_cands(const float* row, int V, int kc, const std::vector<int>& extra,
                       std::vector<std::pair<int, float>>& top, std::vector<float>& ex) {
    kc = std::max(1, std::min(kc, V));
    std::vector<float> v(row, row + V);
    for (float& x : v)
        if (x == 0.0f) x = 0.0f;   // (-0 as +0, as the device's keys)
    std::nth_element(v.begin(), v.begin() + (kc - 1), v.end(), [](float a, float b) { return a > b; });
    const float thr = v[(size_t) kc - 1];
    top.clear();
    for (int i = 0; i < V; ++i)
        if (row[i] >= thr) top.push_back({i, row[i]});
    std::sort(top.begin(), top.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    });
    ex.clear();
    for (int id : extra) ex.push_back(id >= 0 && id < V ? row[id] : 0.0f);
}

}  // namespace mm3
