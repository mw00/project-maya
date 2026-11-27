// src/ik2/ik2_cpu.hpp - maya-ik2's CPU lane for the cache path: a routed expert that is not in VRAM computed where it
// is, in host memory, while the GPU runs the experts it holds - instead of waiting for the expert's 18 MB over PCIe.
// NVFP4 experts only (AVX-512 F / BW / VL + FMA, AVX2 + FMA, or scalar), run across hy4::Lane's physical-core
// workers with adaptive spins and optional core reserve.  The arithmetic is ref/inkling2.py's in F32 (the activation
// not rounded to FP16 as the device's is).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace hy4 {
class Lane;
}

namespace ik2::cpu {

/// Whether this CPU (and OS) runs an SIMD NVFP4 kernel (AVX-512 or AVX2 + FMA).
bool available();

/// Worker count (excluding caller) for --threads: leaves one physical core free for CUDA/OS when saturated.
int recommended_workers(int threads);

/// SIMD form nvfp4_rows() uses here ("avx512", "avx2", or "scalar"); rows_force() forces a form (for tests).
const char* rows_form();
bool rows_force(const char* form);

/// out[r - r0] = W[r] . x for rows r0 .. r1 - 1 of an NVFP4 matrix (rows of K values, K % 64 == 0).
void nvfp4_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out);

/// One NVFP4 expert: gate, up [F][C] and down [C][F] blocks, x [C]; y [C] = down . h, h = gcap tanh(g / gcap)
/// sigmoid(g) ucap tanh(u / ucap), g = (gate . x) sg, u = (up . x) su - the raw sums of down (its scale is the
/// caller's).
struct ExpertJob {
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    const float* x = nullptr;
    float sg = 1.0f, su = 1.0f;
    float* y = nullptr;
};

/// Synchronous batch runner backed by hy4::Lane (physical-core pinned workers + caller).
class Lane {
public:
    explicit Lane(int workers);
    ~Lane();
    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;

    int threads() const;
    void run(const ExpertJob* jobs, int n, int C, int F, float gcap, float ucap);

private:
    std::unique_ptr<hy4::Lane> lane_;
    std::vector<float> h_;
};

}  // namespace ik2::cpu
