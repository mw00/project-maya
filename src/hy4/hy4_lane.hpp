// src/hy4/hy4_lane.hpp - maya-hy4's CPU lane: the routed experts that are not in VRAM, computed on the CPU from the
// model's file (read in place - the page cache keeps what it can) while the GPU computes the others.  The arithmetic is
// maya-ds4's CPU rows for a GGUF's experts (src/ds4/ds4_iq_cpu.cpp: ggml-cpu's dots, its native multi-token rows where
// they are ggml's bits) with the clamped SwiGLU between gate / up and down.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../ds4/ds4_gguf_cpu.hpp"

namespace hy4 {

/// A thread pool: parallel_for(n, f) runs f(0 .. n - 1) on the workers and the caller, back when all are done.  The
/// workers are pinned a physical core each, NUMA node by node; with node_of (an item's node, n values) each worker takes
/// its own node's items first, then helps the others - the experts read where their pages are.
class Lane {
public:
    /// threads: workers besides the caller (0: the caller alone); < 0: one a physical core but the caller's
    explicit Lane(int threads);
    ~Lane();
    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;
    void parallel_for(int n, const std::function<void(int)>& f, const std::vector<int>* node_of = nullptr);
    int threads() const;
    int nodes() const;
    /// The NUMA node of the page at p (-1: not known - not resident, or not Linux)
    static int node_of_page(const void* p);
    /// Move the pages of [p, p + bytes) to NUMA node `node` (their page cache's pages too, when only this process maps
    /// them); false: not moved (one node, not Linux, or the kernel refused)
    static bool move_to_node(const void* p, size_t bytes, int node);

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

/// How the lane and the thread that feeds it wait through a decode's gaps.  The measured waits by default on every
/// system: a thread that sleeps comes back late (milliseconds on Windows, less on Linux), and every lane job after
/// it pays that - the waits keep the lane's cores busy through a whole answer instead, which is faster and makes the
/// CPU hotter while the model answers.  HY4_LANE_WAITS=0 picks the plain ones (a 3 ms spin, no polling, no pinned
/// caller), =1 the measured set; each
/// variable alone (HY4_LANE_SPIN_US, HY4_EV_POLL_US, HY4_LANE_PIN_CALLER) then decides its own.
struct LaneWaits {
    int64_t spin_us = 3000;       // the workers spin this long for the next job before they sleep ...
    bool spin_measured = false;   // ... or twice the longest gap seen lately between two jobs, spin_us at least
    int64_t ev_poll_us = 0;       // a layer's routes: the GPU's event polled this long before the blocking wait
    bool pin_caller = false;      // the thread that brings the jobs stays on the core the workers leave free
};
/// windows: the system's kind; env: getenv (both parameters for the tests)
LaneWaits lane_waits(bool windows, const std::function<const char*(const char*)>& env);
/// This system, the process's environment
LaneWaits lane_waits();

/// A layer's routed experts in host memory: each role's tensor (the GGUF's stacked [n_exp][rows][row bytes]) and the
/// layer's format (ds4's IqCpu: the types, the row bytes, the activations' types).
struct CpuExperts {
    strata::ds4::gguf::IqCpu f;
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    size_t g_stride = 0, d_stride = 0;   // bytes an expert in the gate / up tensors, in the down tensor
    float limit = 0.0f;
    // (on a host of several NUMA nodes expert e's work goes to node e % nodes' threads, where place_experts moves its
    //  pages: a token's experts read by every socket, each from its own memory)
    bool init(int tg, int td, int n_embd, int n_ff, std::string& err);
};

/// One routed expert of one token: out[t] += w down(swiglu(gate x[t], up x[t]))
struct CpuEntry {
    int t = 0, e = 0;
    float w = 0.0f;
};

/// (tests, HY4_PROF) cpu_experts' time by part, summed while `on`: the grouping, the gate / up pass, the down pass, the
/// tokens' sums
struct LaneProf {
    bool on = false;
    double prep_s = 0, gate_up_s = 0, down_s = 0, sum_s = 0;
    long long calls = 0, entries = 0;
};
LaneProf& lane_prof();

/// The pages of these experts moved to node e % nodes (their work's node in cpu_experts), the ones there already passed
/// over; returns early when stop is set.  For a background thread: the model's CPU-lane experts after the load.
void place_experts(const CpuExperts& ex, const std::vector<int>& experts, int nodes, const std::atomic<bool>& stop);
/// These experts' pages read into this process (a byte a page; returns early when stop is set), for a background
/// thread after the load: a decode then finds them mapped - its first touch of an expert otherwise pays a page fault
/// a page, or the disk.  ram_available: the RAM the system can still give, file cache included (0: not known).
void warm_experts(const CpuExperts& ex, const std::vector<int>& experts, const std::atomic<bool>& stop);
uint64_t ram_available();
/// the system's RAM (0: not known)
uint64_t ram_total();
/// Linux, an AMD GPU at this PCI address: the free bytes of the memory that is the GPU's own (on an APU the BIOS's
/// share, which the system does not count as RAM) - 0 where the driver does not say
uint64_t gpu_own_free(int domain, int bus, int dev);

/// out [T][n_embd] = each token's weighted sum of its entries' experts (rows with no entry: 0), x [T][n_embd]; the
/// experts' weights read once for up to 8 of their tokens.  Scratch is the lane's own (grown as needed).
void cpu_experts(Lane& lane, const CpuExperts& ex, const std::vector<CpuEntry>& entries, const float* x, int T,
                 float* out);

}  // namespace hy4
