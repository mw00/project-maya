// src/mm3/mm3_cpu.hpp - maya-mm3's CPU lane: a token's routed experts that are not in VRAM, computed on the CPU from
// host memory while the GPU computes the others (the shared CPU EXL3 kernels, include/strata/kernels/cpu/exl3_cpu.hpp,
// with MiniMax-M3.1's swigluoai between gate / up and down).
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "strata/kernels/cpu/exl3_cpu.hpp"

namespace mm3 {

struct CpuExpert {
    strata::kernels::cpu::Exl3Part p[3];   // gate (w1), up (w3), down (w2)
    float w = 0.0f;                         // its weight in the sum (routed_scaling_factor included)
};

class CpuLane {
public:
    /// threads: workers besides the caller (0: none, the caller alone); max_entries: run_multi's (expert, token)
    /// pairs at most.
    CpuLane(int threads, int dim, int ff, float alpha, float limit, int max_entries = 64);
    ~CpuLane();
    CpuLane(const CpuLane&) = delete;
    CpuLane& operator=(const CpuLane&) = delete;
    /// out [dim] = sum over ex of w * down(swigluoai(gate(x), up(x))), x [dim] (F32).  At most 8 experts.
    void run(const std::vector<CpuExpert>& ex, const float* x, float* out);
    /// A window's: expert i of ex for the tokens users[i] ((token, weight) pairs), x rows [T][dim] -> out rows
    /// [T][dim] = each token's weighted sum of its experts here (0 for a token with none).  Each expert's weights are
    /// decoded once for up to 8 of its tokens.  At most max_entries (expert, token) pairs.
    /// order (if given): the (expert, user) pairs in the order each token's sum takes them (the token path's: its
    /// slots'), so a token's sum is run()'s bits; else expert by expert.  accumulate: out += the sums (else out =).
    void run_multi(const std::vector<CpuExpert>& ex, const std::vector<std::vector<std::pair<int, float>>>& users,
                   const float* x, int T, float* out, const std::vector<std::pair<int, int>>* order = nullptr,
                   bool accumulate = false);
    /// The most (expert, token) pairs a run_multi takes.
    int max_entries() const;
    int threads() const;
    /// f(0 .. n - 1) on the lane's threads (the caller one of them), back when all are done.
    void parallel_for(int n, const std::function<void(int)>& f);
    /// The experts' FFN width from here on (a multiple of 128, at most the constructor's): sub-experts' slices.
    bool set_width(int ff);

private:
    struct Impl;
    Impl* p;
};

}  // namespace mm3
