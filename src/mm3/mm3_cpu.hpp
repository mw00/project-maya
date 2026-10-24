// src/mm3/mm3_cpu.hpp - maya-mm3's CPU lane: a token's routed experts that are not in VRAM, computed on the CPU from
// host memory while the GPU computes the others (the shared CPU EXL3 kernels, include/strata/kernels/cpu/exl3_cpu.hpp,
// with MiniMax-M3.1's swigluoai between gate / up and down).
#pragma once

#include <cstdint>
#include <vector>

#include "strata/kernels/cpu/exl3_cpu.hpp"

namespace mm3 {

struct CpuExpert {
    strata::kernels::cpu::Exl3Part p[3];   // gate (w1), up (w3), down (w2)
    float w = 0.0f;                         // its weight in the sum (routed_scaling_factor included)
};

class CpuLane {
public:
    /// threads: workers besides the caller (0: none, the caller alone).
    CpuLane(int threads, int dim, int ff, float alpha, float limit);
    ~CpuLane();
    CpuLane(const CpuLane&) = delete;
    CpuLane& operator=(const CpuLane&) = delete;
    /// out [dim] = sum over ex of w * down(swigluoai(gate(x), up(x))), x [dim] (F32).  At most 8 experts.
    void run(const std::vector<CpuExpert>& ex, const float* x, float* out);
    /// A window's: expert i of ex for the tokens users[i] ((token, weight) pairs), x rows [T][dim] -> out rows
    /// [T][dim] = each token's weighted sum of its experts here (0 for a token with none).  Each expert's weights are
    /// decoded once for up to 8 of its tokens.  At most kMaxEntries (expert, token) pairs.
    static constexpr int kMaxEntries = 64;
    /// order (if given): the (expert, user) pairs in the order each token's sum takes them (the token path's: its
    /// slots'), so a token's sum is run()'s bits; else expert by expert.
    void run_multi(const std::vector<CpuExpert>& ex, const std::vector<std::vector<std::pair<int, float>>>& users,
                   const float* x, int T, float* out, const std::vector<std::pair<int, int>>* order = nullptr);
    int threads() const;

private:
    struct Impl;
    Impl* p;
};

}  // namespace mm3
