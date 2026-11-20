// src/hy4/hy4_tp.hpp - maya-hy4's tensor parallel (Options::tp): the host's side.
//
// The members (GPUs of one kind) each hold every layer's slice - their attention heads, a share of the dense FFN's and
// the shared expert's width, a slice of each routed expert's width - and a copy of the small matrices and the caches.
// Each queues a whole forward up front (a Crew thread a member); a layer's two partial sums (its attention's output,
// its FFN's) go to pinned host memory with a counter each, the host adds them in member order (a Summer) and publishes
// the sum and its counter, which the members' next step waits for.  Every member reads the same sum, so their copies
// of the residual stay identical bit for bit.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace hy4 {

/// n threads, each running f(i) when started: start() returns at once, wait() when every one is done.  spin_s: how long
/// a thread watches for the next start before it sleeps (0: it sleeps at once) - a token window's threads are started
/// again a few milliseconds after their last (HY4_CREW_SPIN_MS overrides it).
class Crew {
public:
    explicit Crew(int n, double spin_s = 0.0);
    ~Crew();
    Crew(const Crew&) = delete;
    Crew& operator=(const Crew&) = delete;
    void start(const std::function<void(int)>& f);
    void wait();
    int size() const { return (int) th_.size(); }

private:
    std::vector<std::thread> th_;
    std::function<void(int)> f_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::atomic<unsigned> gen_{0};
    std::atomic<int> left_{0};
    std::atomic<bool> quit_{false};
};

/// The host's sums: dst[i] = parts[0][i] + parts[1][i] + ... (that order, whatever the threads), on the caller and
/// `helpers` threads that spin while sums come (a few milliseconds) and sleep after.  cpus: the CPUs they may run on
/// (empty: any) - the NUMA node of most of the GPUs whose sums they add.
class Summer {
public:
    Summer(int helpers, const std::vector<int>& cpus);
    ~Summer();
    Summer(const Summer&) = delete;
    Summer& operator=(const Summer&) = delete;
    /// (and after them n_extra parts more, `stride` apart too, added over their first n_short values only: the helper
    ///  GPUs' expert sums, the latent part of an FFN's)
    void sum(float* dst, const float* parts, int n_parts, size_t stride, size_t n, int n_extra = 0, size_t n_short = 0);
    /// dst = src (first) or dst += src, n values - one member's part as it comes (the same sums as sum()'s, part by part)
    void add(float* dst, const float* src, size_t n, bool first);
    /// F16 rows (pack16_rows' form): out row r = the sum of the parts' rows (part j's at parts + j * stride rows of E,
    /// its factor at scales + j * sstride), in part order, in F32, packed back to F16 with its own factor (out, osc)
    void sum16(uint16_t* out, float* osc, const uint16_t* parts, const float* scales, int n_parts, size_t stride,
               size_t sstride, size_t rows, int E);

private:
    void run(size_t a0, size_t a1) const;
    std::vector<std::thread> th_;
    std::atomic<unsigned> gen_{0}, done_{0};
    std::atomic<bool> quit_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    // the job
    float* dst_ = nullptr;
    const float* parts_ = nullptr;
    int n_parts_ = 0, n_extra_ = 0;
    size_t n_short_ = 0;
    size_t stride_ = 0, n_ = 0, per_ = 0;
    int mode_ = 0;   // (0: sum()'s, 1: add()'s copy, 2: add()'s add, 3: sum16()'s - by rows)
    uint16_t* out16_ = nullptr;
    float* osc_ = nullptr;
    const uint16_t* parts16_ = nullptr;
    const float* scales_ = nullptr;
    size_t sstride_ = 0;
    int E_ = 0;
    void run16(size_t r0, size_t r1) const;
    void go(size_t n);
};

/// The NUMA node of a CUDA device's PCI slot (-1: not known), and a node's CPUs.
int gpu_numa_node(int dev);
std::vector<int> node_cpus(int node);
/// Pin the calling thread to these CPUs (none: unchanged).
void pin_to(const std::vector<int>& cpus);
/// The calling thread's new pages on this NUMA node only (node < 0, or not Linux: nothing), and back to the default.
bool bind_memory_to(int node);
void unbind_memory();

}  // namespace hy4
