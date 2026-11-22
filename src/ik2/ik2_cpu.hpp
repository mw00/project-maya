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
#include <string>
#include <thread>
#include <vector>

#include "strata/kernels/cpu/native_expert.hpp"

namespace ik2::cpu {

/// Whether this CPU (and OS) runs the AVX-512 kernels.
bool available();
/// The AVX-512 gate of the NVFP4 row kernels alone (the GGUF forms take whatever ggml-cpu picked).
bool nvfp4_ok();
/// Whether this build takes GGUF-form experts at all (ggml-cpu's row dots: AVX2 or AVX-512, chosen at run time).
bool native_ok();

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

// ---- IK2_CPU_FRAC: the experts in their GGUF forms (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, the IQ ones), ggml-cpu's own
// quantized dots - AVX2 on a plain PC, AVX-512 where there is one - and the layer's geometry worked out once.

/// One layer's expert geometry: the three tensors' types, their row bytes, the offsets inside [gate|up|down].
struct NativeLayer {
    strata::kernels::cpu::NativeFmt f;
    bool ok = false;   ///< ggml-cpu has a dot for the three types and the packed size is the engine's slot
};
/// Fills l for the layer's gate / up / down types; slot_bytes: the engine's bytes an expert (the packed form must be
/// exactly that, else the CPU cannot read the same bytes the GPU would).
bool native_layer(int ty0, int ty1, int ty2, int64_t C, int64_t F, size_t slot_bytes, NativeLayer& l);

/// One expert of a GGUF-form layer: y [C] = down . h, h the engine's capped activation of (gate . x, up . x).
struct NativeJob {
    const NativeLayer* lay = nullptr;
    const uint8_t* blob = nullptr;   ///< the expert's [gate rows | up rows | down rows] (pinned host memory)
    const float* x = nullptr;        ///< [C] host
    float sg = 1.0f, su = 1.0f;
    float* y = nullptr;              ///< [C] host: the down sums' rows
    std::vector<float> gu, h;
    std::vector<uint8_t> act, hac;   ///< the quantized activations (ggml-cpu's vec_dot_type)
    std::atomic<int> left{0};
};
/// gate/up rows and down rows shared by the pool's threads; done when pool.wait() returns.  job lives until then.
void post_expert_native(Pool& pool, NativeJob* job, int C, int F, float gcap, float ucap);

/// The dots against a host scalar dequant: one line a type into `out`, false when one is off by more than 1e-3.
bool selftest(const std::vector<int>& types, std::string& out);
/// ggml's name of a type ("q3_K", "nvfp4", ...).
const char* type_name(int type);

/// A synthetic expert row of `type` in `row` (8 x 256 values from known floats): the micro-bench's and the selftest's
/// input, so a hand-tuned kernel can be checked against the same bytes.
bool make_test_row(int type, std::vector<uint8_t>& row, std::vector<float>& ref, size_t* row_bytes = nullptr);
/// GB/s of the row dots over `rows` rows split across the pool (all cores), `reps` times: hand-tuned where this build
/// has one, else ggml-cpu's.  One line into `out`.  `hand`: only the hand kernel (false: only ggml's).
void bench(Pool& pool, const std::vector<int>& types, int rows, int reps, const char* which, std::string& out);

}  // namespace ik2::cpu
