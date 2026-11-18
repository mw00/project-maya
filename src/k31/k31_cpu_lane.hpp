// src/k31/k31_cpu_lane.hpp - maya-k31's CPU lane: a quantized expert's rows on the host in iq::mv's arithmetic, so an
// expert gives the same bits on either side and the answers never depend on where the placement put it.
//
// iq::mv (src/ds4/ds4_iq.cu mv_kernel): one warp a row; lane l of 32 takes the row's 32-value sub-blocks l, l + 32, ...
// and sums acc = fmaf(xd[kb], d * (float) (s0 lo + s1 hi), acc) in that order (lo / hi: the sub-block's exact int8 dots
// of values 0..15 and 16..31; d the block's F16 scale / 8, s0 = s1 = 2 s + 1 for IQ1_S and IQ2_XXS; IQ1_S's codes 8 g +- 1:
// ds4_iq1_codes.hpp); then the lanes added by the xor tree 16, 8, 4, 2, 1; y = 1 * sum + 0.  The input is the runner's
// int8 rounding (q8: per 32 values sc = max|x| / 127, q = rint(x / sc)), the same bytes on both sides.
// Built with -ffp-contract=off and no fast math (CMake), every float operation the one written.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace k31::cpu {

/// The ggml types the CPU lane takes (IQ1_S, IQ2_XXS).
bool rows_supported(int type);
/// y[c * ycs + (r - r0)] = row r . x_c for rows [r0, r1) of a ggml matrix of `type` with k inputs (ld bytes a row) and
/// ncol int8 inputs: xq[c * k + j], xd[c * (k / 32) + b] - iq::mv's bits.
void rows(int type, const uint8_t* w, size_t ld, int k, int r0, int r1, const int8_t* xq, const float* xd, int ncol,
          float* y, size_t ycs);
/// x[rows][k] -> the runner's int8 rounding (q8_quant_kernel's bytes): q[rows][k], d[rows][k / 32].
void q8(const float* x, int k, int rows, int8_t* q, float* d);
/// Which form rows() runs ("avx512" or "scalar"; the same bits).
const char* form();

/// A fixed set of worker threads: run(n, fn) calls fn(0) .. fn(n - 1) over them (and the caller) and returns when all
/// are done.  One caller at a time.
class Pool {
public:
    explicit Pool(int threads);
    ~Pool();
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    void run(int n, const std::function<void(int)>& fn);
    int threads() const { return (int) workers_.size() + 1; }

private:
    void loop();
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable go_, done_;
    const std::function<void(int)>* fn_ = nullptr;
    std::atomic<int> next_{0}, n_{0};
    int busy_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
    void drain();
};

}  // namespace k31::cpu
