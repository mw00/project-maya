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
    int threads() const;

private:
    struct Impl;
    Impl* p;
};

}  // namespace mm3
