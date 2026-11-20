// src/hy4/hy4_lane.cpp - maya-hy4's CPU lane (hy4_lane.hpp).
#include "hy4_lane.hpp"

#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>   // (__cpuid: the CPU's name in the 4-bit copies' verdict key)
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

#include "strata/platform/commit.hpp"
#include "strata/platform/dedicated.hpp"
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
#if defined(__linux__)
// the CPUs this process may run on, read once - at the first lane, before its caller pins itself: on Linux
// sched_getaffinity(0) reads the calling thread's own mask, and the caller's is one core after its first job (a lane
// made after that saw that one core: every worker unpinned on it, beside the caller - and a thread the caller starts
// inherits the mask, so a worker left unpinned takes this one instead)
bool process_mask(cpu_set_t& out) {
    static cpu_set_t m;
    static const bool ok = [] {
        CPU_ZERO(&m);
        return sched_getaffinity(0, sizeof m, &m) == 0;
    }();
    out = m;
    return ok;
}
#endif
// each NUMA node's physical cores (their first logical CPU), only those this process may run on; one node of every
// core when the host has no NUMA information
std::vector<std::vector<int>> node_cores() {
    std::vector<std::vector<int>> nodes;
#if defined(__linux__)
    cpu_set_t allowed;
    const bool mask = process_mask(allowed);
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
#if defined(__linux__)
        else {   // (a worker beyond the cores: the process's CPUs, not the mask of the thread that started it)
            cpu_set_t all;
            if (process_mask(all)) pthread_setaffinity_np(pthread_self(), sizeof all, &all);
        }
#endif
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
bool Lane::idle() const { return !p->th.empty() && p->sleepers.load() == (int) p->th.size(); }
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
    iqc::iq_cpu_row_hint(n_embd);   // (the row forms timed on this length)
    g_stride = f.gu_row * (size_t) n_ff;
    d_stride = f.d_row * (size_t) n_embd;
    return true;
}

bool CpuExperts::q4x_can() {
    if (!iqc::iq_cpu_q4x_ok(f.tg) || !iqc::iq_cpu_q4x_ok(f.td) || f.n_embd % 256 || f.n_ff % 256) return false;
    q4x_gu = iqc::iq_cpu_q4x_row_bytes(f.n_embd) * (size_t) f.n_ff;
    q4x_d = iqc::iq_cpu_q4x_row_bytes(f.n_ff) * (size_t) f.n_embd;
    return true;
}

bool CpuExperts::q4x_make(int e) {
    if (!q4x || q4x_gu == 0 || q4x->p[(size_t) e].load(std::memory_order_acquire) != nullptr) return false;
    std::unique_ptr<uint8_t[]> m(new (std::nothrow) uint8_t[2 * q4x_gu + q4x_d + 32]);
    if (!m) return false;
    uint8_t* b = (uint8_t*) (((uintptr_t) m.get() + 31) & ~(uintptr_t) 31);   // (32-byte rows)
    iqc::iq_cpu_q4x_convert(gate + (size_t) e * g_stride, f.gu_row, f.n_ff, f.n_embd, b);
    iqc::iq_cpu_q4x_convert(up + (size_t) e * g_stride, f.gu_row, f.n_ff, f.n_embd, b + q4x_gu);
    iqc::iq_cpu_q4x_convert(down + (size_t) e * d_stride, f.d_row, f.n_embd, f.n_ff, b + 2 * q4x_gu);
    q4x->owned.push_back(std::move(m));
    q4x->p[(size_t) e].store(b, std::memory_order_release);
    return true;
}

namespace {

constexpr int kMaxNt = 64;      // tokens one call's rows take at most (a group's: its weights read once for them)
int group_nt() {                 // (HY4_LANE_NT: a group's tokens, 1 .. kMaxNt; default 32)
    static const int v = [] {
        const char* e = std::getenv("HY4_LANE_NT");
        const int n = e ? std::atoi(e) : 32;
        return std::max(1, std::min(kMaxNt, n));
    }();
    return v;
}
constexpr int kRowChunk = 128;   // rows a work item (64 and 32: slower on Sun's Zen 5)

struct Grp {
    int e, a, c;                    // the expert, its first entry in Scratch::order, how many
    const uint8_t* q4 = nullptr;    // (the expert's 4-bit copy when it has one: the same bits, faster)
};
struct Scratch {
    std::vector<Grp> groups;          // the entries by expert, a group up to kMaxNt of them
    std::vector<float> h, y;          // the entries' h and down outputs
    std::vector<uint8_t> xq, hq;      // the tokens' quantized activations, the entries' quantized h
    std::vector<int> toks, order;     // the tokens that have entries; the entries by expert
};

}  // namespace

LaneCount& lane_count() {
    static LaneCount c;
    return c;
}

LaneProf& lane_prof() {
    static LaneProf p;
    return p;
}

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

void unpin_this_thread() {
#if defined(__linux__)
    cpu_set_t all;
    if (process_mask(all)) pthread_setaffinity_np(pthread_self(), sizeof all, &all);
#endif
}

void lower_this_thread() {
#if defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif defined(__linux__)
    (void) nice(10);   // (Linux: a thread's own nice value)
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

uint64_t gpu_dedicated(const void* luid) { return strata::platform::dedicated_bytes(luid); }

uint64_t commit_available() {
#if defined(_WIN32)
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    return GlobalMemoryStatusEx(&s)
               ? (uint64_t) s.ullAvailPageFile + strata::platform::commit_growth((uint64_t) s.ullTotalPhys)
               : 0;
#else
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
    LaneCount& lcm = lane_count();
    const int qmode = lcm.q4x_mode.load(std::memory_order_relaxed);
    const bool trial = qmode == 2 && T == 1;
    const bool use_q4 = qmode == 1 || (trial && (++lcm.trial_tick & 1));
    const auto tt0 = trial ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    thread_local Scratch tls;
    Scratch& sc = tls;   // (the caller's: the workers use it through this reference, never their own thread_local)
    sc.h.resize((size_t) n * F);
    sc.y.resize((size_t) n * E);
    // each token's activations quantized once - the same bytes for all of its experts (each work item quantized its
    // own tokens' before: a decode's 16 experts did a token 12 times over, a row chunk each) - few tokens on the
    // caller, more over the lane
    sc.xq.resize((size_t) T * f.act_bytes);
    sc.toks.clear();
    {
        thread_local std::vector<char> seen;
        seen.assign((size_t) T, 0);
        for (const CpuEntry& c : entries)
            if (!seen[(size_t) c.t]) {
                seen[(size_t) c.t] = 1;
                sc.toks.push_back(c.t);
            }
    }
    const auto qx = [&](int i) {
        const int t = sc.toks[(size_t) i];
        iqc::iq_cpu_quant(f.gu_act, x + (size_t) t * E, E, sc.xq.data() + (size_t) t * f.act_bytes);
    };
    const int nq = (int) sc.toks.size();
    if (nq <= 2)
        for (int i = 0; i < nq; ++i) qx(i);
    else
        lane.parallel_for(nq, qx);
    // the entries grouped by expert (ascending; in each, the entries' order), a group up to kMaxNt of them: an
    // expert's rows read once for its tokens
    std::vector<Grp>& groups = sc.groups;   // (the caller's, as sc: a worker's own thread_local would be another)
    groups.clear();
    sc.order.resize((size_t) n);
    for (int i = 0; i < n; ++i) sc.order[(size_t) i] = i;
    std::stable_sort(sc.order.begin(), sc.order.end(),
                     [&](int i, int j) { return entries[(size_t) i].e < entries[(size_t) j].e; });
    const int gn = group_nt();
    for (int i = 0; i < n;) {
        const int e = entries[(size_t) sc.order[(size_t) i]].e;
        int j = i;
        while (j < n && entries[(size_t) sc.order[(size_t) j]].e == e) ++j;
        const uint8_t* q4 = ex.q4x && use_q4 ? ex.q4x->p[(size_t) e].load(std::memory_order_acquire) : nullptr;
        for (int a = i; a < j; a += gn) groups.push_back({e, a, std::min(gn, j - a), q4});
        i = j;
    }
    const int gu_chunks = (F + kRowChunk - 1) / kRowChunk, dn_chunks = (E + kRowChunk - 1) / kRowChunk;
    // each group's expert's NUMA node - e % nodes, where place_experts puts its pages - the work items routed by it
    std::vector<int> gu_node, dn_node;
    if (lane.nodes() > 1)
        for (const Grp& gr : groups) {
            const int q = gr.e % lane.nodes();
            for (int c = 0; c < gu_chunks; ++c) gu_node.push_back(q);
            for (int c = 0; c < dn_chunks; ++c) dn_node.push_back(q);
        }
    const double tp1 = clk();
    lane.parallel_for((int) groups.size() * gu_chunks, [&](int it) {
        const Grp& gr = groups[(size_t) (it / gu_chunks)];
        const int r0 = (it % gu_chunks) * kRowChunk, r1 = std::min(F, r0 + kRowChunk);
        const void* act[kMaxNt];
        float* ff[kMaxNt];
        for (int k = 0; k < gr.c; ++k) {
            const int id = sc.order[(size_t) (gr.a + k)];
            act[k] = sc.xq.data() + (size_t) entries[(size_t) id].t * f.act_bytes;
            ff[k] = sc.h.data() + (size_t) id * F;
        }
        if (gr.q4)
            iqc::iq_cpu_q4x_gate_up(f, gr.q4, gr.q4 + ex.q4x_gu, act, gr.c, ff, r0, r1, ex.limit);
        else
            iqc::iq_cpu_gate_up(f, ex.gate + (size_t) gr.e * ex.g_stride, ex.up + (size_t) gr.e * ex.g_stride, act,
                                gr.c, ff, r0, r1, ex.limit);
    }, gu_node.empty() ? nullptr : &gu_node);
    // each entry's h quantized once (each down item quantized its entries' h before: 12 times over in a decode)
    sc.hq.resize((size_t) n * f.h_bytes);
    const auto qh = [&](int i) {
        iqc::iq_cpu_quant(f.d_act, sc.h.data() + (size_t) i * F, F, sc.hq.data() + (size_t) i * f.h_bytes);
    };
    if (n <= 2)
        for (int i = 0; i < n; ++i) qh(i);
    else
        lane.parallel_for(n, qh);
    const double tp2 = clk();
    lane.parallel_for((int) groups.size() * dn_chunks, [&](int it) {
        const Grp& gr = groups[(size_t) (it / dn_chunks)];
        const int r0 = (it % dn_chunks) * kRowChunk, r1 = std::min(E, r0 + kRowChunk);
        const void* hq[kMaxNt];
        float* y[kMaxNt];
        for (int k = 0; k < gr.c; ++k) {
            const int id = sc.order[(size_t) (gr.a + k)];
            hq[k] = sc.hq.data() + (size_t) id * f.h_bytes;
            y[k] = sc.y.data() + (size_t) id * E;
        }
        if (gr.q4)
            iqc::iq_cpu_q4x_down(f, gr.q4 + 2 * ex.q4x_gu, hq, gr.c, y, r0, r1);
        else
            iqc::iq_cpu_down(f, ex.down + (size_t) gr.e * ex.d_stride, hq, gr.c, y, r0, r1);
    }, dn_node.empty() ? nullptr : &dn_node);
    const double tp3 = clk();
    // each token's weighted sum, in its entries' order
    for (int i = 0; i < n; ++i) {
        const CpuEntry& e = entries[(size_t) i];
        float* o = out + (size_t) e.t * E;
        const float* y = sc.y.data() + (size_t) i * E;
        for (int d = 0; d < E; ++d) o[d] += e.w * y[d];
    }
    {
        long long nq = 0;
        for (const Grp& gr : groups)
            if (gr.q4) nq += gr.c;
        LaneCount& lc = lane_count();
        lc.entries.fetch_add(n, std::memory_order_relaxed);
        if (nq) lc.q4x.fetch_add(nq, std::memory_order_relaxed);
        if (trial) {
            const int k = use_q4 ? 1 : 0;
            lc.trial_s[k] += std::chrono::duration<double>(std::chrono::steady_clock::now() - tt0).count();
            lc.trial_entries[k] += n;
            ++lc.trial_calls[k];
        }
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

bool q4x_probe(Lane& lane, const CpuExperts& ex0, int n_expert, double& ms_file, double& ms_q4x) {
    ms_file = ms_q4x = 0.0;
    CpuExperts ex = ex0;   // (the same file rows; copies of its own)
    if (!ex.q4x_can() || n_expert < 16) return false;
    // up to 64 experts spread over the layer (the file's rows read once first: the page cache timed, not the disk)
    const int n_exp = std::min(64, n_expert);
    std::vector<int> pick((size_t) n_exp);
    for (int i = 0; i < n_exp; ++i) pick[(size_t) i] = (int) ((int64_t) i * n_expert / n_exp);
    ex.q4x = std::make_shared<Q4xSet>(n_expert);
    volatile uint8_t sink = 0;
    for (int e : pick) {
        for (size_t b = 0; b < ex.g_stride; b += 4096) sink = sink ^ ex.gate[(size_t) e * ex.g_stride + b] ^ ex.up[(size_t) e * ex.g_stride + b];
        for (size_t b = 0; b < ex.d_stride; b += 4096) sink = sink ^ ex.down[(size_t) e * ex.d_stride + b];
        if (!ex.q4x_make(e)) return false;
    }
    const int E = ex.f.n_embd;
    std::vector<float> x((size_t) E), out((size_t) E);
    uint32_t s = 2463534242u;
    auto rnd = [&s] {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    };
    for (float& v : x) v = ((float) (rnd() % 2001) - 1000.0f) * 1e-3f;
    std::vector<CpuEntry> ent;
    double best[2] = {1e30, 1e30};
    const std::shared_ptr<Q4xSet> set = ex.q4x;
    for (int r = 0; r < 24; ++r) {
        if (r % 2 == 0) {   // (the same 16 experts for both forms)
            std::vector<int> p = pick;
            for (int i = 0; i < 16 && i < n_exp; ++i) std::swap(p[(size_t) i], p[(size_t) (i + rnd() % (uint32_t) (n_exp - i))]);
            ent.clear();
            for (int i = 0; i < 16 && i < n_exp; ++i) ent.push_back({0, p[(size_t) i], 0.25f});
        }
        const int form = r % 2;
        ex.q4x = form ? set : nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        cpu_experts(lane, ex, ent, x.data(), 1, out.data());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (r >= 4) best[form] = std::min(best[form], ms);
    }
    ms_file = best[0];
    ms_q4x = best[1];
    return true;
}

// ------------------------------------------------------------------------------------- the 4-bit copies' verdict
#ifndef HY4_LANE_KERNELS
#define HY4_LANE_KERNELS unknown
#endif
#define HY4_STR2(x) #x
#define HY4_STR(x) HY4_STR2(x)
const char* lane_kernels_id() { return HY4_STR(HY4_LANE_KERNELS); }

namespace {

// the CPU's own name (x86: its brand string; else Linux's cpuinfo), tabs and line ends out
std::string cpu_brand() {
    std::string s;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int r[4] = {0, 0, 0, 0};
    __cpuid(r, (int) 0x80000000);
    if ((unsigned) r[0] >= 0x80000004u)
        for (int leaf = 0; leaf < 3; ++leaf) {
            __cpuid(r, (int) (0x80000002u + leaf));
            s.append((const char*) r, 16);
        }
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    unsigned w[4] = {0, 0, 0, 0};
    if (__get_cpuid(0x80000000u, &w[0], &w[1], &w[2], &w[3]) && w[0] >= 0x80000004u)
        for (unsigned leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf)
            if (__get_cpuid(leaf, &w[0], &w[1], &w[2], &w[3])) s.append((const char*) w, 16);
#endif
#if defined(__linux__)
    if (s.find_first_not_of(std::string(" \0", 2)) == std::string::npos) {
        const std::string info = read_file("/proc/cpuinfo");
        for (const char* k : {"model name", "Hardware", "CPU part"}) {
            const size_t at = info.find(k);
            if (at == std::string::npos) continue;
            const size_t colon = info.find(':', at), end = info.find('\n', at);
            if (colon != std::string::npos && colon < end) s = info.substr(colon + 1, end - colon - 1);
            break;
        }
    }
#endif
    std::string out;
    for (char ch : s) {
        if (ch == '\0') break;
        out += (ch == '\t' || ch == '\n' || ch == '\r') ? ' ' : ch;
    }
    const size_t b0 = out.find_first_not_of(' '), b1 = out.find_last_not_of(' ');
    return b0 == std::string::npos ? std::string("cpu") : out.substr(b0, b1 - b0 + 1);
}

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> f;
    size_t a = 0;
    for (size_t t; (t = line.find('\t', a)) != std::string::npos; a = t + 1) f.push_back(line.substr(a, t - a));
    f.push_back(line.substr(a));
    return f;
}

}  // namespace

std::string q4x_memo_key(const std::string& model, int lane_threads) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "|threads %u|RAM %.0f GB|lane %d threads|", std::thread::hardware_concurrency(),
                  (double) ram_total() / 1e9, lane_threads);
    std::string k = cpu_brand() + buf + model + "|lane kernels " + lane_kernels_id();
    for (char& ch : k)
        if (ch == '\t' || ch == '\n' || ch == '\r') ch = ' ';
    return k;
}

Q4xMemo q4x_memo_get(const std::string& path, const std::string& key) {
    Q4xMemo m;
    std::ifstream f(path);
    for (std::string line; f && std::getline(f, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> v = split_tabs(line);   // key, kept|dropped, us with, us without, date
        if (v.size() < 5 || v[0] != key || (v[1] != "kept" && v[1] != "dropped")) continue;
        m.found = true;
        m.kept = v[1] == "kept";
        m.us_with = std::atof(v[2].c_str());
        m.us_without = std::atof(v[3].c_str());
        m.date = v[4];
    }
    return m;
}

bool q4x_memo_put(const std::string& path, const std::string& key, Q4xMemo m) {
    if (path.empty()) return false;
    std::vector<std::string> keep;
    {
        std::ifstream f(path);
        for (std::string line; f && std::getline(f, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#' || split_tabs(line)[0] == key) continue;
            keep.push_back(line);
        }
    }
    const std::time_t now = std::time(nullptr);
    char day[16] = "?";
    std::tm tmv{};
#if defined(_WIN32)
    if (localtime_s(&tmv, &now) == 0) std::strftime(day, sizeof day, "%Y-%m-%d", &tmv);
#else
    if (localtime_r(&now, &tmv)) std::strftime(day, sizeof day, "%Y-%m-%d", &tmv);
#endif
    m.date = day;
    char nums[64];
    std::snprintf(nums, sizeof nums, "%.1f\t%.1f\t", m.us_with, m.us_without);
    keep.push_back(key + "\t" + (m.kept ? "kept" : "dropped") + "\t" + nums + m.date);
    const std::string tmp = path + ".new";
    {
        std::ofstream o(tmp, std::ios::trunc);
        if (!o) return false;
        o << "# maya-hy4: the CPU lane's 4-bit expert copies, measured on this PC - a line a PC, model and lane kernels\n"
             "# (key, kept / dropped, an expert run's us with them and without, date); delete a line, or run\n"
             "# maya --speed-calibration, to measure again\n";
        for (const std::string& l : keep) o << l << "\n";
        if (!o.flush()) return false;
    }
    std::remove(path.c_str());   // (Windows: rename does not replace)
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

}  // namespace hy4
