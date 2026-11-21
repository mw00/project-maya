// src/ik2/ik2_lane.hpp - maya-ik2's CPU lane for the offload path's lane-parity mode (IK2_LANE): the routed experts a
// visit gives it (ik2_off.cuh's LaneJob, in host memory the GPU writes), computed in ik2_exact.h's arithmetic - the
// same bits as the GPU's off_gate_up_x / off_down_x - on Hy4's lane threads (hy4::Lane: a physical core each, NUMA
// nodes, the busy-PC reserve), the rows answered in LaneDone for the GPU to pull.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace hy4 {
class Lane;
}

namespace ik2::lane {

/// out[i] = ik2_exact.h's row_dot of rows r0 + i of W (rows of K values, type ty: NVFP4 or Q8_0) with x [K]: the
/// fastest form this CPU runs (AVX-512, AVX2 + FMA, or the reference), all of them the same bits.
void rows(int ty, const uint8_t* W, int K, int r0, int r1, const float* x, float* out);
/// the form rows() uses here ("avx512", "avx2", "scalar"); force (tests): that form from now on (false: not here)
const char* rows_form();
bool rows_force(const char* form);

/// A layer's experts in host memory: expert e's gate at base + e * stride, up at + off_up, down at + off_down
struct LayerW {
    const uint8_t* base = nullptr;
    size_t stride = 0, off_up = 0, off_down = 0;
    int ty_gu = 0, ty_d = 0;
};

/// The lane: a thread polls the job's seq (spins while jobs come, naps when they stop), runs each job's experts on
/// the lane threads (gate / up rows and the SwiGLU, then down rows, in pieces) and writes the rows and its seq.
class Server {
public:
    /// threads: the lane's workers besides the polling thread (< 0: one a physical core but two); job / done: the
    /// mailboxes' host addresses (ik2_off.cuh's layouts)
    Server(int threads, int C, int F, float gcap, float ucap, std::vector<LayerW> layers, void* job, void* done);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    int threads() const;
    /// the lane's workers made again (n besides the polling thread; < 0: Hy4's default size, a physical core each
    /// but one, with its measured reserve on a busy PC) - at the start only, before any job
    void set_threads(int n);
    /// microseconds an expert as measured while it runs (fmt 0 NVFP4, 1 Q8_0; 0: none yet), and the experts done
    double us_expert(int fmt) const;
    unsigned long long experts() const { return n_done_.load(); }
    /// one job run here and now (the polling thread not involved): n experts of layer l - the calibration's timing
    double time_experts(int layer, const int* e, int n, const float* x, int reps);

private:
    void loop();
    void run(uint32_t seq);
    void compute(int layer, const int* e, const float* sg, const float* su, int n, const float* x, float* y);
    int C_, F_;
    float gcap_, ucap_;
    std::vector<LayerW> L_;
    uint8_t *job_, *done_;
    std::unique_ptr<hy4::Lane> lane_;
    std::vector<float> h_;   // [16][F]
    std::atomic<bool> quit_{false};
    std::atomic<unsigned long long> n_done_{0};
    std::atomic<double> us_[2];
    struct Hist {
        unsigned long long n = 0;
        double sum = 0, max = 0;
    };
    std::vector<Hist> hist_ = std::vector<Hist>(16);
    std::thread th_;
};

}  // namespace ik2::lane
