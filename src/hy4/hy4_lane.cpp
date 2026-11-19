// src/hy4/hy4_lane.cpp - maya-hy4's CPU lane (hy4_lane.hpp).
#include "hy4_lane.hpp"

#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <sstream>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace hy4 {

namespace iqc = strata::ds4::gguf;

namespace {
std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::vector<int> cpu_list(const std::string& s) {   // "0-3,8,10-11"
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string part; std::getline(ss, part, ',');) {
        if (part.empty() || part == "\n") continue;
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str());
        const int b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) v.push_back(c);
    }
    return v;
}
// each NUMA node's physical cores (their first logical CPU), only those this process may run on; one node of every
// core when the host has no NUMA information
std::vector<std::vector<int>> node_cores() {
    std::vector<std::vector<int>> nodes;
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    const bool mask = sched_getaffinity(0, sizeof allowed, &allowed) == 0;
    auto ok = [&](int c) { return c >= 0 && (!mask || (c < CPU_SETSIZE && CPU_ISSET(c, &allowed))); };
    for (int n = 0; n < 64; ++n) {
        const std::string cl = read_file("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (cl.empty()) break;
        std::vector<int> cores, seen;
        for (int c : cpu_list(cl)) {
            if (!ok(c)) continue;
            std::vector<int> sib = cpu_list(
                read_file("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
            const int first = sib.empty() ? c : sib[0];
            if (std::find(seen.begin(), seen.end(), first) != seen.end()) continue;
            seen.push_back(first);
            cores.push_back(c);
        }
        if (!cores.empty()) nodes.push_back(cores);
    }
#endif
    if (nodes.empty()) {
        std::vector<int> all = strata::kernels::cpu::physical_cores(false);
        nodes.push_back(all);
    }
    return nodes;
}
inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}
}  // namespace

// ---------------------------------------------------------------- the pool
// A job is published by bumping `gen`; the workers spin on it for kSpin after their last job (a decode step's layers
// come every few hundred microseconds), then sleep on the condition variable until the next one.  Its items are split
// by node: a worker drains its node's list, then the others'.
struct Lane::Impl {
    // (the spin: 3 ms at least, and twice the longest gap seen lately between two jobs - a decode's are the GPU's
    //  share of a layer and, once a token, its head and first layers; a worker that sleeps through one comes back
    //  late (Windows: milliseconds), every job after it paying that.  Gaps over kIdle are a pause between requests,
    //  not counted; HY4_LANE_SPIN_US: a fixed spin instead, 0 = asleep at once)
    static constexpr int64_t kSpinMin = 3000, kSpinMax = 100000, kIdle = 250000;   // us
    std::atomic<int64_t> spin_us{kSpinMin};
    bool spin_fixed = false;
    double gap_us = 0.0;   // (the caller's: the recent gaps' decaying maximum)
    std::chrono::steady_clock::time_point last_end{};
    static constexpr int kMaxNodes = 8;
    std::vector<std::thread> th;
    std::vector<int> wnode;   // each worker's node
    int n_nodes = 1, caller_node = 0, caller_core = -1;
    std::mutex mu;
    std::condition_variable cv;
    const std::function<void(int)>* f = nullptr;
    std::vector<int> items[kMaxNodes];   // the job's items, by node
    std::atomic<int> next[kMaxNodes];
    std::atomic<int> busy{0};
    std::atomic<uint64_t> gen{0};
    std::atomic<int> sleepers{0};
    std::atomic<bool> quit{false};

    void work(int node) {
        for (int k = 0; k < n_nodes; ++k) {
            const int q = (node + k) % n_nodes;
            const std::vector<int>& v = items[q];
            const int nn = (int) v.size();
            for (int i; (i = next[q].fetch_add(1, std::memory_order_relaxed)) < nn;) (*f)(v[(size_t) i]);
        }
    }
    void loop(int core, int node) {
        if (core >= 0) strata::kernels::cpu::pin_current_thread(core);
        uint64_t seen = 0;
        for (;;) {
            auto t0 = std::chrono::steady_clock::now();
            uint64_t g;
            for (int k = 0;; ++k) {   // (spin, then sleep)
                g = gen.load(std::memory_order_acquire);
                if (g != seen || quit.load(std::memory_order_relaxed)) break;
                cpu_relax();
                if ((k & 1023) == 1023 && std::chrono::steady_clock::now() - t0 >
                                              std::chrono::microseconds(spin_us.load(std::memory_order_relaxed))) {
                    std::unique_lock<std::mutex> lk(mu);
                    sleepers.fetch_add(1);
                    cv.wait(lk, [&] { return gen.load() != seen || quit.load(); });
                    sleepers.fetch_sub(1);
                    t0 = std::chrono::steady_clock::now();
                }
            }
            if (quit.load()) return;
            seen = g;
            work(node);
            busy.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
};

LaneWaits lane_waits(bool windows, const std::function<const char*(const char*)>& env) {
    LaneWaits w;
    const char* all = env("HY4_LANE_WAITS");
    (void)windows;   // (the same default on every system; a sleeping thread comes back latest on Windows)
    const bool on = all ? std::atoi(all) != 0 : true;
    w.spin_measured = on;
    w.ev_poll_us = on ? 2000 : 0;
    w.pin_caller = on;
    if (const char* e = env("HY4_LANE_SPIN_US")) {
        w.spin_us = std::max(0, std::atoi(e));
        w.spin_measured = false;
    }
    if (const char* e = env("HY4_EV_POLL_US")) w.ev_poll_us = std::max(0, std::atoi(e));
    if (const char* e = env("HY4_LANE_PIN_CALLER")) w.pin_caller = std::atoi(e) != 0;
    return w;
}

LaneWaits lane_waits() {
#if defined(_WIN32)
    const bool windows = true;
#else
    const bool windows = false;
#endif
    return lane_waits(windows, [](const char* k) -> const char* { return std::getenv(k); });
}

Lane::Lane(int threads) : p(new Impl) {
    const std::vector<std::vector<int>> nodes = node_cores();
    p->n_nodes = std::max(1, std::min((int) nodes.size(), Impl::kMaxNodes));
    const LaneWaits lw = lane_waits();
    p->spin_us.store(lw.spin_us);
    p->spin_fixed = !lw.spin_measured;
    for (int q = 0; q < Impl::kMaxNodes; ++q) p->next[q].store(0);
    // every node's cores but the first core of node 0 (the caller's), round-robin over the nodes so a smaller lane
    // still spans them all
    std::vector<std::pair<int, int>> order;   // (core, node)
    size_t most = 0;
    for (const auto& c : nodes) most = std::max(most, c.size());
    for (size_t i = 0; i < most; ++i)
        for (int q = 0; q < p->n_nodes; ++q)
            if (i < nodes[(size_t) q].size() && !(q == 0 && i == 0)) order.push_back({nodes[(size_t) q][i], q});
    if (threads < 0) threads = (int) order.size();
    // the caller's core: node 0's first, the one no worker takes (not pinned: the caller left where the system runs
    // it - beside a worker, on its core's second thread, when it comes back from a wait)
    if (threads > 0 && !nodes[0].empty() && lw.pin_caller) p->caller_core = nodes[0][0];
    for (int i = 0; i < threads; ++i) {
        const int core = i < (int) order.size() ? order[(size_t) i].first : -1;
        const int node = i < (int) order.size() ? order[(size_t) i].second : 0;
        p->wnode.push_back(node);
        p->th.emplace_back([this, core, node] { p->loop(core, node); });
    }
}
Lane::~Lane() {
    {
        std::lock_guard<std::mutex> lk(p->mu);
        p->quit.store(true);
    }
    p->cv.notify_all();
    for (auto& t : p->th) t.join();
}
int Lane::threads() const { return (int) p->th.size(); }
int Lane::nodes() const { return p->n_nodes; }

int Lane::node_of_page(const void* ptr) {
#if defined(__linux__) && defined(SYS_move_pages)
    void* page = (void*) ((uintptr_t) ptr & ~(uintptr_t) 4095);
    int status = -1;
    if (syscall(SYS_move_pages, 0, 1UL, &page, nullptr, &status, 0) == 0 && status >= 0) return status;
#else
    (void) ptr;
#endif
    return -1;
}

bool Lane::move_to_node(const void* ptr, size_t bytes, int node) {
#if defined(__linux__) && defined(SYS_move_pages)
    const uintptr_t a = (uintptr_t) ptr & ~(uintptr_t) 4095, b = ((uintptr_t) ptr + bytes + 4095) & ~(uintptr_t) 4095;
    const size_t n = (b - a) / 4096;
    std::vector<void*> pages(n);
    std::vector<int> nodes(n, node), status(n, -1);
    for (size_t i = 0; i < n; ++i) pages[i] = (void*) (a + i * 4096);
    constexpr int kMoveMapped = 2;   // MPOL_MF_MOVE: the pages only this process maps
    return syscall(SYS_move_pages, 0, (unsigned long) n, pages.data(), nodes.data(), status.data(), kMoveMapped) == 0;
#else
    (void) ptr;
    (void) bytes;
    (void) node;
    return false;
#endif
}

void Lane::parallel_for(int n, const std::function<void(int)>& f, const std::vector<int>* node_of) {
    if (n <= 0) return;
    if (p->th.empty() || n == 1) {
        for (int i = 0; i < n; ++i) f(i);
        return;
    }
    for (int q = 0; q < p->n_nodes; ++q) {
        p->items[q].clear();
        p->next[q].store(0, std::memory_order_relaxed);
    }
    for (int i = 0; i < n; ++i) {
        const int q = node_of && (*node_of)[(size_t) i] >= 0 ? (*node_of)[(size_t) i] % p->n_nodes : i % p->n_nodes;
        p->items[q].push_back(i);
    }
    if (p->caller_core >= 0) {   // (the thread that brings the jobs: pinned once)
        thread_local const Impl* pinned_for = nullptr;
        if (pinned_for != p.get()) {
            strata::kernels::cpu::pin_current_thread(p->caller_core);
            pinned_for = p.get();
        }
    }
    if (!p->spin_fixed) {
        const auto t = std::chrono::steady_clock::now();
        const double gap = std::chrono::duration<double, std::micro>(t - p->last_end).count();
        if (p->last_end != std::chrono::steady_clock::time_point{} && gap < (double) Impl::kIdle) {
            p->gap_us = std::max(gap, p->gap_us * 0.995);
            p->spin_us.store(std::max(Impl::kSpinMin, std::min(Impl::kSpinMax, (int64_t) (2.0 * p->gap_us))),
                             std::memory_order_relaxed);
        }
    }
    p->f = &f;
    p->busy.store((int) p->th.size(), std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(p->mu);   // (a sleeper checks gen under the lock: no lost wake-up)
        p->gen.fetch_add(1, std::memory_order_release);
    }
    if (p->sleepers.load() > 0) p->cv.notify_all();
    p->work(p->caller_node);
    while (p->busy.load(std::memory_order_acquire) > 0) cpu_relax();
    p->f = nullptr;
    if (!p->spin_fixed) p->last_end = std::chrono::steady_clock::now();
}

// ---------------------------------------------------------------- the experts
bool CpuExperts::init(int tg, int td, int n_embd, int n_ff, std::string& err) {
    if (!iqc::iq_cpu_fmt(tg, td, n_embd, n_ff, f, err)) return false;
    g_stride = f.gu_row * (size_t) n_ff;
    d_stride = f.d_row * (size_t) n_embd;
    return true;
}

namespace {

constexpr int kMaxNt = 64;       // tokens one call's rows take at most (a group's: its weights read once for them)
int group_nt() {                 // (HY4_LANE_NT: a group's tokens, 1 .. kMaxNt; default 32)
    static const int v = [] {
        const char* e = std::getenv("HY4_LANE_NT");
        const int n = e ? std::atoi(e) : 32;
        return std::max(1, std::min(kMaxNt, n));
    }();
    return v;
}
constexpr int kRowChunk = 128;   // rows a work item

struct Scratch {
    std::vector<float> h, y;   // the entries' h and down outputs
};

}  // namespace

LaneProf& lane_prof() {
    static LaneProf p;
    return p;
}

// Two passes (each a parallel_for): gate / up (each item its tokens' activations quantized itself - a few
// microseconds, saved a barrier), then down (each item its entries' h quantized itself).
void place_experts(const CpuExperts& ex, const std::vector<int>& experts, int nodes, const std::atomic<bool>& stop) {
    if (nodes < 2) return;
    for (int e : experts) {
        if (stop.load(std::memory_order_relaxed)) return;
        const int want = e % nodes;
        const uint8_t* g = ex.gate + (size_t) e * ex.g_stride;
        const uint8_t* u = ex.up + (size_t) e * ex.g_stride;
        const uint8_t* d = ex.down + (size_t) e * ex.d_stride;
        if (Lane::node_of_page(g) == want && Lane::node_of_page(d) == want) continue;   // (there already)
        volatile uint8_t sink = 0;   // (mapped into this process first: move_pages takes mapped pages)
        for (size_t b = 0; b < ex.g_stride; b += 4096) sink ^= g[b] ^ u[b];
        for (size_t b = 0; b < ex.d_stride; b += 4096) sink ^= d[b];
        Lane::move_to_node(g, ex.g_stride, want);
        Lane::move_to_node(u, ex.g_stride, want);
        Lane::move_to_node(d, ex.d_stride, want);
    }
}

void warm_experts(const CpuExperts& ex, const std::vector<int>& experts, const std::atomic<bool>& stop) {
    volatile uint8_t sink = 0;
    for (int e : experts) {
        if (stop.load(std::memory_order_relaxed)) return;
        const uint8_t* g = ex.gate + (size_t) e * ex.g_stride;
        const uint8_t* u = ex.up + (size_t) e * ex.g_stride;
        const uint8_t* d = ex.down + (size_t) e * ex.d_stride;
        for (size_t b = 0; b < ex.g_stride; b += 4096) sink = sink ^ g[b] ^ u[b];
        for (size_t b = 0; b < ex.d_stride; b += 4096) sink = sink ^ d[b];
    }
}

uint64_t ram_available() {
#if defined(_WIN32)
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    return GlobalMemoryStatusEx(&s) ? (uint64_t) s.ullAvailPhys : 0;
#else
    const std::string m = read_file("/proc/meminfo");
    const size_t at = m.find("MemAvailable:");
    return at == std::string::npos ? 0 : (uint64_t) std::strtoull(m.c_str() + at + 13, nullptr, 10) * 1024;
#endif
}

uint64_t gpu_own_free(int domain, int bus, int dev) {
#if defined(__linux__)
    char p[96];
    std::snprintf(p, sizeof p, "/sys/bus/pci/devices/%04x:%02x:%02x.0/mem_info_vram_", domain, bus, dev);
    const std::string tot = read_file(std::string(p) + "total"), used = read_file(std::string(p) + "used");
    if (tot.empty() || used.empty()) return 0;
    const uint64_t t = std::strtoull(tot.c_str(), nullptr, 10), u = std::strtoull(used.c_str(), nullptr, 10);
    return t > u ? t - u : 0;
#else
    (void) domain;
    (void) bus;
    (void) dev;
    return 0;
#endif
}

uint64_t ram_total() {
#if defined(_WIN32)
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    return GlobalMemoryStatusEx(&s) ? (uint64_t) s.ullTotalPhys : 0;
#else
    const std::string m = read_file("/proc/meminfo");
    const size_t at = m.find("MemTotal:");
    return at == std::string::npos ? 0 : (uint64_t) std::strtoull(m.c_str() + at + 9, nullptr, 10) * 1024;
#endif
}

void cpu_experts(Lane& lane, const CpuExperts& ex, const std::vector<CpuEntry>& entries, const float* x, int T,
                 float* out) {
    const iqc::IqCpu& f = ex.f;
    const int E = f.n_embd, F = f.n_ff, n = (int) entries.size();
    std::fill(out, out + (size_t) T * E, 0.0f);
    if (n == 0) return;
    LaneProf& lp = lane_prof();
    auto clk = [&] {
        return lp.on ? std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() : 0.0;
    };
    const double tp0 = clk();
    thread_local Scratch tls;
    Scratch& sc = tls;   // (the caller's: the workers use it through this reference, never their own thread_local)
    sc.h.resize((size_t) n * F);
    sc.y.resize((size_t) n * E);
    // the entries grouped by expert, a group up to kMaxNt of them (an expert's rows read once for its tokens)
    std::map<int, std::vector<int>> by;
    for (int i = 0; i < n; ++i) by[entries[(size_t) i].e].push_back(i);
    std::vector<std::pair<int, std::vector<int>>> groups;
    const size_t gn = (size_t) group_nt();
    for (auto& [e, v] : by)
        for (size_t a = 0; a < v.size(); a += gn)
            groups.push_back({e, std::vector<int>(v.begin() + (long) a, v.begin() + (long) std::min(v.size(), a + gn))});
    const int gu_chunks = (F + kRowChunk - 1) / kRowChunk, dn_chunks = (E + kRowChunk - 1) / kRowChunk;
    // each group's expert's NUMA node - e % nodes, where place_experts puts its pages - the work items routed by it
    std::vector<int> gu_node, dn_node;
    if (lane.nodes() > 1)
        for (const auto& [e, ids] : groups) {
            const int q = e % lane.nodes();
            for (int c = 0; c < gu_chunks; ++c) gu_node.push_back(q);
            for (int c = 0; c < dn_chunks; ++c) dn_node.push_back(q);
        }
    const double tp1 = clk();
    lane.parallel_for((int) groups.size() * gu_chunks, [&](int it) {
        const auto& [e, ids] = groups[(size_t) (it / gu_chunks)];
        const int r0 = (it % gu_chunks) * kRowChunk, r1 = std::min(F, r0 + kRowChunk);
        thread_local std::vector<uint8_t> abuf;
        abuf.resize((size_t) kMaxNt * f.act_bytes);
        const void* act[kMaxNt];
        float* ff[kMaxNt];
        for (size_t k = 0; k < ids.size(); ++k) {
            uint8_t* a = abuf.data() + k * f.act_bytes;
            iqc::iq_cpu_quant(f.gu_act, x + (size_t) entries[(size_t) ids[k]].t * E, E, a);
            act[k] = a;
            ff[k] = sc.h.data() + (size_t) ids[k] * F;
        }
        iqc::iq_cpu_gate_up(f, ex.gate + (size_t) e * ex.g_stride, ex.up + (size_t) e * ex.g_stride, act,
                            (int) ids.size(), ff, r0, r1, ex.limit);
    }, gu_node.empty() ? nullptr : &gu_node);
    const double tp2 = clk();
    lane.parallel_for((int) groups.size() * dn_chunks, [&](int it) {
        const auto& [e, ids] = groups[(size_t) (it / dn_chunks)];
        const int r0 = (it % dn_chunks) * kRowChunk, r1 = std::min(E, r0 + kRowChunk);
        thread_local std::vector<uint8_t> hbuf;
        hbuf.resize((size_t) kMaxNt * f.h_bytes);
        const void* hq[kMaxNt];
        float* y[kMaxNt];
        for (size_t k = 0; k < ids.size(); ++k) {
            uint8_t* h = hbuf.data() + k * f.h_bytes;
            iqc::iq_cpu_quant(f.d_act, sc.h.data() + (size_t) ids[k] * F, F, h);
            hq[k] = h;
            y[k] = sc.y.data() + (size_t) ids[k] * E;
        }
        iqc::iq_cpu_down(f, ex.down + (size_t) e * ex.d_stride, hq, (int) ids.size(), y, r0, r1);
    }, dn_node.empty() ? nullptr : &dn_node);
    const double tp3 = clk();
    // each token's weighted sum, in its entries' order
    for (int i = 0; i < n; ++i) {
        const CpuEntry& e = entries[(size_t) i];
        float* o = out + (size_t) e.t * E;
        const float* y = sc.y.data() + (size_t) i * E;
        for (int d = 0; d < E; ++d) o[d] += e.w * y[d];
    }
    if (lp.on) {
        lp.prep_s += tp1 - tp0;
        lp.gate_up_s += tp2 - tp1;
        lp.down_s += tp3 - tp2;
        lp.sum_s += clk() - tp3;
        ++lp.calls;
        lp.entries += n;
    }
}

}  // namespace hy4
