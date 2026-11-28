#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hy4 {

struct TpHotSlot {
    int rank;
    int expert;
    float score;
};

struct TpHotSwap {
    float gain;
    int il;
    int gpu_rank;
    int out_e;
    int cpu_rank;
    int in_e;
};

inline std::vector<TpHotSwap> plan_tp_hot_layer(
    int il, const float* scores, int n_expert,
    const std::vector<std::vector<int>>& slot_of,
    const std::vector<char>& lane_e,
    const std::vector<std::vector<char>>& stream_e,
    double ratio, double margin) {
    std::vector<TpHotSlot> res, cpu;
    const int tp = static_cast<int>(slot_of.size());
    for (int e = 0; e < n_expert; ++e) {
        int owner_gpu = -1;
        for (int m = 0; m < tp; ++m) {
            if (slot_of[static_cast<size_t>(m)][static_cast<size_t>(e)] >= 0) {
                owner_gpu = m;
                break;
            }
        }
        if (owner_gpu >= 0) {
            res.push_back(TpHotSlot{owner_gpu, e, scores[e]});
            continue;
        }
        if (e < static_cast<int>(lane_e.size()) && lane_e[static_cast<size_t>(e)] != 0) {
            int owner_cpu = -1;
            for (int k = 0; k < tp; ++k) {
                if (e < static_cast<int>(stream_e[static_cast<size_t>(k)].size()) &&
                    stream_e[static_cast<size_t>(k)][static_cast<size_t>(e)] != 0) {
                    owner_cpu = k;
                    break;
                }
            }
            if (owner_cpu >= 0) cpu.push_back(TpHotSlot{owner_cpu, e, scores[e]});
        }
    }
    std::stable_sort(res.begin(), res.end(), [](const TpHotSlot& a, const TpHotSlot& b) { return a.score < b.score; });
    std::stable_sort(cpu.begin(), cpu.end(), [](const TpHotSlot& a, const TpHotSlot& b) { return a.score > b.score; });
    std::vector<TpHotSwap> out;
    for (size_t i = 0; i < res.size() && i < cpu.size(); ++i) {
        if (static_cast<double>(cpu[i].score) <= ratio * static_cast<double>(res[i].score) + margin) break;
        out.push_back(TpHotSwap{cpu[i].score - res[i].score, il, res[i].rank, res[i].expert, cpu[i].rank, cpu[i].expert});
    }
    return out;
}

inline int apply_tp_hot_layer_swap(
    int out_e, int in_e,
    std::vector<int>& gpu_slot_of,
    std::vector<char>& lane_e,
    std::vector<char>& cpu_stream_e) {
    const int slot = gpu_slot_of[static_cast<size_t>(out_e)];
    if (slot < 0) return -1;
    gpu_slot_of[static_cast<size_t>(out_e)] = -1;
    gpu_slot_of[static_cast<size_t>(in_e)] = slot;
    lane_e[static_cast<size_t>(in_e)] = 0;
    lane_e[static_cast<size_t>(out_e)] = 1;
    cpu_stream_e[static_cast<size_t>(in_e)] = 0;
    cpu_stream_e[static_cast<size_t>(out_e)] = 1;
    return slot;
}

}  // namespace hy4
