// src/ik2/ik2_cpu.hpp - maya-ik2's CPU lane: a routed expert that is not in VRAM computed where it is, in host
// memory, while the GPU runs the experts it holds - instead of waiting for the expert's 18 MB over PCIe.
// NVFP4 experts only (AVX-512 F / BW; the weights read as they are stored), a few worker threads sharing an expert's
// rows.  The arithmetic is ref/inkling2.py's in F32 (the activation not rounded to FP16 as the device's is).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ik2::cpu {

/// Whether this CPU (and OS) runs the AVX-512 kernels.
bool available();

/// Worker threads taking tasks in order; wait(): until every task posted so far (and the ones they posted) is done.
class Pool {
public:
    explicit Pool(int n);
    ~Pool();
    void post(std::function<void()> f);
    void wait();
    int size() const { return (int) th_.size(); }

private:
    std::vector<std::thread> th_;
    std::deque<std::function<void()>> q_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    int pending_ = 0;
    bool stop_ = false;
};

/// out[r - r0] = W[r] . x for rows r0 .. r1 - 1 of an NVFP4 matrix (rows of K values, K % 64 == 0).
void nvfp4_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out);

/// One NVFP4 expert: gate, up [F][C] and down [C][F] blocks, x [C]; y [C] = down . h, h = gcap tanh(g / gcap)
/// sigmoid(g) ucap tanh(u / ucap), g = (gate . x) sg, u = (up . x) su - the raw sums of down (its scale is the
/// caller's).  The rows shared by the pool's threads; done when pool.wait() returns.  job must live until then.
struct ExpertJob {
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    const float* x = nullptr;
    float sg = 1.0f, su = 1.0f;
    float* y = nullptr;
    std::vector<float> gu, h;
    std::atomic<int> left{0};
};
void post_expert(Pool& pool, ExpertJob* job, int C, int F, float gcap, float ucap);

}  // namespace ik2::cpu
