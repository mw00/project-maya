// src/core/glm_tier_numa.cu - the glm-dsa port B8.2 / B8.3 / B8.4 / B8.7: the memory tiers at run time (see
// glm_tier_numa.hpp).  Host code plus two small copy kernels; nothing here runs unless its feature is on.
#include "strata/core/glm_env.hpp"
#include "strata/core/glm_req_knobs.hpp"   // speed plan: per-request prefetch_blocks / cpu_split
#include "glm_tier_numa.hpp"

#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_dsa_split.hpp"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <utility>
#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::core::glmtier {

using glmfast::cpu_relax;

namespace {

// ---------------------------------------------------------------- topology (sysfs; Linux only)

std::vector<int> parse_list(const std::string& list, int limit) {
    std::vector<int> out;
    std::stringstream ss(list);
    std::string r;
    while (std::getline(ss, r, ',')) {
        if (r.empty()) continue;
        const size_t d = r.find('-');
        const int a = std::atoi(r.c_str()), b = d == std::string::npos ? a : std::atoi(r.c_str() + d + 1);
        for (int n = a; n <= b && n < limit; ++n) out.push_back(n);
    }
    return out;
}

std::vector<int> online_nodes() {
#ifdef __linux__
    std::string list;
    std::ifstream("/sys/devices/system/node/online") >> list;
    return parse_list(list, 256);
#else
    return {};
#endif
}

// this process's CPUs on NUMA node `node` ({} when unknown)
std::vector<int> node_cpus(int node) {
    std::vector<int> out;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return out;
    std::string list;
    std::ifstream("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist") >> list;
    for (int c : parse_list(list, CPU_SETSIZE))
        if (CPU_ISSET(c, &set)) out.push_back(c);
#else
    (void) node;
#endif
    return out;
}

// the physical cores among CPUs (distinct package / core id: SMT siblings count once); 0 when unknown
int physical_cores(const std::vector<int>& cpus) {
    std::vector<std::pair<int, int>> cores;
#ifdef __linux__
    for (int c : cpus) {
        int pkg = -1, core = -1;
        const std::string t = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
        std::ifstream(t + "physical_package_id") >> pkg;
        std::ifstream(t + "core_id") >> core;
        if (pkg < 0 || core < 0) return 0;
        cores.push_back({pkg, core});
    }
#else
    (void) cpus;
#endif
    std::sort(cores.begin(), cores.end());
    return (int) (std::unique(cores.begin(), cores.end()) - cores.begin());
}

std::string read_file(const char* path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void pin_self(const std::vector<int>& cpus) {
#ifdef __linux__
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
        if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
    sched_setaffinity(0, sizeof set, &set);
#else
    (void) cpus;
#endif
}

int env_int(const char* name, int dflt) {
    const char* v = getenv(name);
    return v != nullptr && v[0] != '\0' ? std::atoi(v) : dflt;
}

// ---------------------------------------------------------------- device copies (B8.4)

template<int W> struct Word;
template<> struct Word<4> { using T = unsigned int; };
template<> struct Word<8> { using T = unsigned long long; };
template<> struct Word<16> { using T = uint4; };

// a split expert (half 0 at h0, half 1 at h0 + D) into its unified blob at dst: the threads walk the halves' words in
// order (contiguous reads over the link) and write each to its unified place
template<int W>
__device__ void unsplit_words(const uint8_t* h0, size_t D, uint8_t* dst, const SplitLayout& L, size_t tid, size_t nth) {
    using T = typename Word<W>::T;
    const size_t nw = L.half_bytes / W;
    for (size_t j = tid; j < 2 * nw; j += nth) {
        const int h = j >= nw ? 1 : 0;
        const size_t off = (j - (size_t) h * nw) * W;
        const T v = *(const T*) (h0 + (size_t) h * D + off);
        *(T*) (dst + unsplit_off(L, h, off)) = v;
    }
}

// Maya's contiguous pull (moe_fetch_kernel's loop): a source outside the split ranges (the disk staging)
__device__ void plain_words(const uint8_t* src, uint8_t* dst, size_t n16, size_t tid, size_t nth) {
    const uint4* s = (const uint4*) src;
    uint4* t = (uint4*) dst;
    for (size_t j = tid; j < n16; j += nth) t[j] = s[j];
}

template<int W>
__global__ void __launch_bounds__(256) split_fetch_kernel(gf::MoeDev d, int k, SplitLayout L, unsigned long long lo,
                                                          unsigned long long hi, size_t D, size_t n16) {
    const size_t tid = blockIdx.x * (size_t) blockDim.x + threadIdx.x, nth = (size_t) gridDim.x * blockDim.x;
    for (int i = 0; i < k; ++i) {
        const unsigned long long src = d.fetch_src[i];
        if (src == 0ull) continue;
        uint8_t* dst = (uint8_t*) d.plan_ptr[i];
        if (src >= lo && src < hi) unsplit_words<W>((const uint8_t*) src, D, dst, L, tid, nth);
        else plain_words((const uint8_t*) src, dst, n16, tid, nth);
    }
}

template<int W>
__global__ void __launch_bounds__(256) split_prefetch_kernel(gf::MoeDev d, SplitLayout L, unsigned long long lo,
                                                             unsigned long long hi, size_t D, size_t n16) {
    const size_t tid = blockIdx.x * (size_t) blockDim.x + threadIdx.x, nth = (size_t) gridDim.x * blockDim.x;
    const int n = *d.pf_n;
    for (int i = 0; i < n; ++i) {
        const unsigned long long src = d.pf_src[i];
        uint8_t* dst = (uint8_t*) d.pf_dst[i];
        if (src >= lo && src < hi) unsplit_words<W>((const uint8_t*) src, D, dst, L, tid, nth);
        else plain_words((const uint8_t*) src, dst, n16, tid, nth);
    }
}

int sm_count() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 80;
    }
    return sms;
}

std::atomic<uint64_t> g_launch_errors{0};

void launch_ok(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        g_launch_errors.fetch_add(1);
        std::fprintf(stderr, "glm tier: %s launch failed: %s\n", what, cudaGetErrorString(e));
    }
}

const TierClass* layer_class(const TierState* T, int il) {
    if (T == nullptr || il < 0 || il >= (int) T->layer_cls.size() || il >= (int) T->L.size()) return nullptr;
    const int c = T->layer_cls[(size_t) il];
    if (c < 0 || c >= (int) T->cls.size()) return nullptr;
    const TierClass& tc = T->cls[(size_t) c];
    return tc.ours && tc.split && T->L[(size_t) il].blob > 0 ? &tc : nullptr;
}

bool in_class(const TierClass& tc, const void* p) {
    const uint8_t* q = (const uint8_t*) p;
    return q >= tc.map && q < tc.map + (size_t) tc.n * tc.stride;
}

// ---------------------------------------------------------------- node lanes (B8.3 / B8.4)

std::mutex g_lane_mu;
std::map<std::pair<int, int>, std::weak_ptr<NodeLane>> g_lanes;

std::shared_ptr<NodeLane> node_lane(int node, int threads) {
    std::lock_guard<std::mutex> lk(g_lane_mu);
    auto& w = g_lanes[{node, threads}];
    std::shared_ptr<NodeLane> p = w.lock();
    if (p == nullptr) {
        p = std::make_shared<NodeLane>(node, node_cpus(node), threads);
        w = p;
    }
    return p;
}

// F's tier state, made (and filled) on first use: by pin_class, or by lane_pools when no class was pinned here
TierState& tier_of(Glm5Model::FastState& F, const Glm5Geometry& g, int dev, const std::vector<int>& nodes) {
    if (F.tier == nullptr) F.tier = std::make_shared<TierState>();
    TierState& T = *F.tier;
    if (!T.ready) {
        const bool dsa = g.arch == GlmArch::GlmDsa;
        T.ready = true;
        T.dsa = dsa;
        // the port keeps Maya v1.1.55's RAM tier for glm-dsa too: one class per size, pinned by numa_pinned (with
        // MAYA_GLM_ROW_HALVES for the per-node halves) - pin_class's node-split segments are not wired in, so the
        // node-split lanes stay off whatever MAYA_GLM_NUMA_SPLIT says
        T.split_req = false;
        if (dsa && numa_split_requested(dsa, getenv("MAYA_GLM_NUMA_SPLIT")))
            std::fprintf(stderr, "glm tier: MAYA_GLM_NUMA_SPLIT is not served by this build (MAYA_GLM_ROW_HALVES=1 "
                                 "gives each node its half): ignored\n");
        T.seg_cap = seg_cap_bytes(dsa, getenv("MAYA_GLM_RAM_SEG_GB"));
        T.dev = dev;
        T.nodes = nodes;
        T.L.assign(F.L.size(), SplitLayout{});
        T.half.assign(F.L.size(), kc::NativeFmt{});
        if (!dsa && numa_split_requested(true, getenv("MAYA_GLM_NUMA_SPLIT")))
            std::fprintf(stderr, "glm tier: MAYA_GLM_NUMA_SPLIT is glm-dsa only: ignored\n");
    }
    return T;
}

}  // namespace

// ---------------------------------------------------------------- NodeLane

NodeLane::NodeLane(int node_, std::vector<int> cpus_, int threads) : node(node_), cpus(std::move(cpus_)) {
    pool = std::make_shared<glmfast::Workers>(std::max(0, threads - 1), 20000, cpus);
    th_ = std::thread([this] { loop(); });
    glmfast::pin_thread(th_.native_handle(), cpus);
}

NodeLane::~NodeLane() {
    quit_.store(true);
    gen_.fetch_add(1);
    if (th_.joinable()) th_.join();
}

void NodeLane::start(const std::function<void()>* task) {
    task_ = task;
    gen_.fetch_add(1, std::memory_order_acq_rel);
}

void NodeLane::wait() {
    const uint64_t g = gen_.load(std::memory_order_acquire);
    while (done_.load(std::memory_order_acquire) != g) cpu_relax();
}

void NodeLane::loop() {
    uint64_t seen = 0;
    for (;;) {
        // spin 20 ms for the next task (a decode's routes come ~1 ms apart), then poll every 200 us
        auto t0 = std::chrono::steady_clock::now();
        int spins = 0;
        uint64_t g;
        while ((g = gen_.load(std::memory_order_acquire)) == seen) {
            if (quit_.load(std::memory_order_relaxed)) return;
            cpu_relax();
            if ((++spins & 1023) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(20))
                std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        if (quit_.load()) return;
        seen = g;
        if (task_ != nullptr) (*task_)();
        done_.store(g, std::memory_order_release);
    }
}

bool TierState::split_any() const {
    for (const auto& c : cls)
        if (c.ours && c.split) return true;
    return false;
}

// ---------------------------------------------------------------- B8.2 pinning

void* pin_class(Glm5Model::FastState& F, const Glm5Geometry& g, int c, int64_t& n, int64_t least, int dev) {
#ifdef __linux__
    auto& R = F.rc[(size_t) c];
    const bool dsa = g.arch == GlmArch::GlmDsa;
    const size_t cap = seg_cap_bytes(dsa, getenv("MAYA_GLM_RAM_SEG_GB"));
    if (cap == 0 || n <= 0 || R.stride == 0) return nullptr;
    if (const char* v = getenv("MAYA_GLM_NUMA"); v != nullptr && std::atoi(v) == 0) return nullptr;
    const std::vector<int> nodes = online_nodes();
    if (nodes.size() < 2) return nullptr;   // (Maya's numa_pinned rule: one node pins with cudaHostAlloc)
    int ok_attr = 0;
    if (cudaDeviceGetAttribute(&ok_attr, cudaDevAttrCanUseHostPointerForRegisteredMem, dev) != cudaSuccess || !ok_attr) {
        cudaGetLastError();
        return nullptr;
    }
    TierState& T = tier_of(F, g, dev, nodes);
    T.layer_cls = F.layer_rc;
    if (T.cls.size() < F.rc.size()) T.cls.resize(F.rc.size());

    // ---- B8.4: the class's layers' split layouts (every one must split, and its lane must have a half format)
    bool split = T.split_req;
    std::vector<int> layers;
    for (int il = 0; il < (int) F.L.size() && il < (int) F.layer_rc.size(); ++il)
        if (F.layer_rc[(size_t) il] == c && F.L[(size_t) il].moe) layers.push_back(il);
    std::vector<SplitLayout> lays;
    std::string why;
    for (int il : layers) {
        if (!split) break;
        const auto& Ly = F.L[(size_t) il];
        const int64_t n_ff = g.n_ff_exp, n_embd = g.n_embd;
        SplitLayout L;
        if (n_ff <= 0 || Ly.gu_bytes % (size_t) n_ff != 0 || Ly.dn_bytes % (size_t) n_embd != 0 ||
            !split_layout(Ly.gu_bytes / (size_t) n_ff, Ly.dn_bytes / (size_t) n_embd, n_ff, n_embd,
                          ggml_blck_size((ggml_type) Ly.d_type), ggml_type_size((ggml_type) Ly.d_type), L, why) ||
            L.blob != Ly.blob) {
            if (why.empty()) why = "the blob is not [gate | up | down] rows";
            split = false;
            break;
        }
        kc::NativeFmt full, half;
        if (kc::native_experts_available()) {
            if (!kc::native_fmt(Ly.gu_type, Ly.d_type, n_embd, n_ff, full, why) || !half_fmt(full, half, why)) {
                split = false;
                break;
            }
        }
        T.L[(size_t) il] = L;
        T.half[(size_t) il] = half;
        lays.push_back(L);
    }
    if (T.split_req && !split) {
        std::fprintf(stderr, "glm tier: CUDA%d RAM class %d stays interleaved (no node split): %s\n", dev, c, why.c_str());
        for (int il : layers) T.L[(size_t) il] = SplitLayout{};
    }
    const char* segpol = getenv("MAYA_GLM_NUMA_SEG");
    bool bind_seg = segpol != nullptr && std::strcmp(segpol, "bind") == 0;
    // every segment must start on a page (mbind, and cudaHostRegister works in whole pages: two segments sharing one
    // would register it twice), so a slot is a whole number of pages here (+ < 4 KB on a ~11 MB expert)
    size_t stride = split ? half_stride(lays) : (R.stride + 4095) & ~(size_t) 4095;
    // ---- W6 runtime fix: MPOL_BIND never falls back to another node, and the RAM budget is one machine-wide figure,
    // so a bound class (node split, NUMA_SEG=bind) is capped by each node's own free memory less a reserve before any
    // page is faulted in: shrunk when the nodes still hold >= 3/4 of it, else interleaved (which spills over)
    for (int pass = 0; pass < 2 && (split || bind_seg); ++pass) {
        const int64_t reserve = node_reserve_bytes(getenv("MAYA_GLM_NODE_RESERVE_GB"));
        std::vector<int64_t> budget(nodes.size(), -1);
        for (size_t j = 0; j < nodes.size(); ++j) {
            const std::string mi = read_file(("/sys/devices/system/node/node" + std::to_string(nodes[j]) + "/meminfo").c_str());
            const int64_t a = node_mem_avail(mi);
            if (a >= 0) budget[j] = std::max<int64_t>(0, a - reserve);
        }
        const int64_t fit = node_fit_slots(n, stride, cap, split, nodes, budget);
        const int act = node_fit_action(n, fit, least);
        if (act == 0) break;
        if (act == 1) {
            std::fprintf(stderr, "glm tier: CUDA%d RAM class %d: %lld of %lld slots fit the NUMA nodes' free memory "
                         "(bound: no fallback node), shrunk\n", dev, c, (long long) fit, (long long) n);
            n = fit;
            break;
        }
        std::fprintf(stderr, "glm tier: CUDA%d RAM class %d: only %lld of %lld slots fit the NUMA nodes' free memory: "
                     "%s\n", dev, c, (long long) fit, (long long) n,
                     split ? "no node split (interleaved)" : "interleaved instead of bound");
        if (split) {
            split = false;
            for (int il : layers) T.L[(size_t) il] = SplitLayout{};
            stride = (R.stride + 4095) & ~(size_t) 4095;
        } else {
            bind_seg = false;
        }
    }
    const SegPlan plan = seg_plan(n, stride, cap);
    const size_t D = (size_t) n * stride;
    const size_t map_bytes = split ? 2 * D : D;
    void* mp = mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (mp == MAP_FAILED) return nullptr;
    uint8_t* map = (uint8_t*) mp;
    const auto t0 = std::chrono::steady_clock::now();

    // ---- huge pages per (segment, half), in order (B8.7: MAYA_GLM_THP=0/1, auto = Maya's buddyinfo rule)
    const int halves = split ? 2 : 1;
    const int ns = (int) plan.slots.size();
    std::vector<size_t> unit_bytes;
    for (int k = 0; k < ns; ++k)
        for (int h = 0; h < halves; ++h) unit_bytes.push_back((size_t) plan.slots[(size_t) k] * stride);
    const int thp = thp_mode(getenv("MAYA_GLM_THP"));
    const auto node_index = [&](int node) {
        for (size_t j = 0; j < nodes.size(); ++j)
            if (nodes[j] == node) return (int) j;
        return 0;
    };
    // W6 runtime fix (B8.7): a bound unit's huge pages come from its own node only, so it draws on that node's free
    // 2 MB blocks; an interleaved unit on the machine-wide sum (Maya's rule)
    const std::string buddy = read_file("/proc/buddyinfo");
    std::vector<int> unit_node;
    for (int k = 0; k < ns; ++k)
        for (int h = 0; h < halves; ++h) unit_node.push_back(split || bind_seg ? node_index(seg_node(k, h, split, nodes)) : -1);
    const std::vector<char> huge = thp_plan_nodes(unit_bytes, unit_node, buddy_huge_free_nodes(buddy, nodes),
                                                  buddy_huge_free(buddy), thp);
    const int fault_threads = std::max(1, std::min(32, env_int("MAYA_GLM_PIN_THREADS", 4)));
    std::vector<std::vector<int>> cpus_of(nodes.size());
    for (size_t j = 0; j < nodes.size(); ++j) cpus_of[j] = node_cpus(nodes[j]);

    // one (segment, half) unit: policy, huge pages, fault-in, registration
    const auto unit_ptr = [&](int k, int h) { return map + (size_t) h * D + (size_t) k * (size_t) plan.seg_slots * stride; };
    const auto pin_unit = [&](int k, int h, int slots) -> bool {
        uint8_t* p = unit_ptr(k, h);
        const size_t bytes = (size_t) slots * stride;
        if (bytes == 0) return true;
        unsigned long mask[4] = {0, 0, 0, 0};
        int mode = 3;   // MPOL_INTERLEAVE over every node (Maya's default)
        if (split || bind_seg) {
            const int nd = seg_node(k, h, split, nodes);
            mask[nd / 64] |= 1ul << (nd % 64);
            mode = 2;   // MPOL_BIND
        } else {
            for (int nd : nodes) mask[nd / 64] |= 1ul << (nd % 64);
        }
        if (syscall(SYS_mbind, p, bytes, mode, mask, (unsigned long) (sizeof mask * 8), 0u) != 0) return false;
        const bool hp = huge[(size_t) (k * halves + h)] != 0;
        if (hp) madvise(p, bytes, MADV_HUGEPAGE);
        else if (thp == 0) madvise(p, bytes, MADV_NOHUGEPAGE);
        // fault in before registering (cudaHostRegister holds the driver lock): huge pages in parallel 2 MB strides by
        // threads on the unit's node, else this (node-pinned) thread page by page
        const int nd = seg_node(k, h, split, nodes);
        const int ft = hp ? fault_threads : 1;
        const auto touch = [p, bytes, hp, ft](int t) {
            for (const auto& r : fault_plan(bytes, hp, ft, t))
                for (size_t o = r.first; o < r.second; o += 4096) ((volatile char*) p)[o] = 0;
        };
        if (ft > 1) {
            std::vector<std::thread> th;
            for (int t = 1; t < ft; ++t) {
                th.emplace_back(touch, t);
                glmfast::pin_thread(th.back().native_handle(), cpus_of[(size_t) node_index(nd)]);
            }
            touch(0);
            for (auto& x : th) x.join();
        } else {
            touch(0);
        }
        cudaSetDevice(dev);
        if (cudaHostRegister(p, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
            cudaGetLastError();
            madvise(p, bytes, MADV_DONTNEED);
            return false;
        }
        return true;
    };
    const auto release_unit = [&](int k, int h, int slots) {
        uint8_t* p = unit_ptr(k, h);
        const size_t bytes = (size_t) slots * stride;
        if (bytes == 0) return;
        cudaSetDevice(dev);
        cudaHostUnregister(p);
        cudaGetLastError();
        madvise(p, bytes, MADV_DONTNEED);
    };
    // the first round: one thread per NUMA node, each pinned to its node, taking the units placed there in order
    const auto first_round = [&](std::vector<char>& ok) {
        std::vector<char> uok((size_t) ns * halves, 0);
        std::vector<std::thread> th;
        for (size_t j = 0; j < nodes.size(); ++j)
            th.emplace_back([&, j] {
                pin_self(cpus_of[j]);
                for (int k = 0; k < ns; ++k)
                    for (int h = 0; h < halves; ++h)
                        if (seg_node(k, h, split, nodes) == nodes[j])
                            uok[(size_t) (k * halves + h)] = pin_unit(k, h, plan.slots[(size_t) k]) ? 1 : 0;
            });
        for (auto& x : th) x.join();
        for (int k = 0; k < ns; ++k) {
            bool all = true;
            for (int h = 0; h < halves; ++h) all = all && uok[(size_t) (k * halves + h)];
            ok[(size_t) k] = all ? 1 : 0;
            if (!all)   // a half that pinned while the other did not: let it go (the segment is retried whole)
                for (int h = 0; h < halves; ++h)
                    if (uok[(size_t) (k * halves + h)]) release_unit(k, h, plan.slots[(size_t) k]);
        }
    };
    const auto pin_seg = [&](int k, int slots) -> bool {
        std::vector<char> uok((size_t) halves, 0);
        std::vector<std::thread> th;
        for (int h = 0; h < halves; ++h)
            th.emplace_back([&, h] {
                pin_self(cpus_of[(size_t) node_index(seg_node(k, h, split, nodes))]);
                uok[(size_t) h] = pin_unit(k, h, slots) ? 1 : 0;
            });
        for (auto& x : th) x.join();
        bool all = true;
        for (int h = 0; h < halves; ++h) all = all && uok[(size_t) h];
        if (!all)
            for (int h = 0; h < halves; ++h)
                if (uok[(size_t) h]) release_unit(k, h, slots);
        return all;
    };
    const auto release_seg = [&](int k, int slots) {
        for (int h = 0; h < halves; ++h) release_unit(k, h, slots);
    };
    std::vector<int> pinned;
    const int64_t total = seg_pin(plan, std::max<int64_t>(1, least), first_round, pin_seg, release_seg, pinned);
    if (total <= 0) {
        munmap(map, map_bytes);
        std::fprintf(stderr, "glm tier: CUDA%d RAM class %d: segmented pinning refused (Maya's path takes over)\n", dev, c);
        if (split)
            for (int il : layers) T.L[(size_t) il] = SplitLayout{};
        return nullptr;
    }
    TierClass& tc = T.cls[(size_t) c];
    tc = TierClass{};
    tc.ours = true;
    tc.split = split;
    tc.map = map;
    tc.D = split ? D : 0;
    tc.stride = stride;
    tc.n = (int) total;
    for (int k = 0; k < (int) pinned.size(); ++k)
        for (int h = 0; h < halves; ++h)
            tc.regs.push_back({(size_t) h * D + (size_t) k * (size_t) plan.seg_slots * stride, (size_t) pinned[(size_t) k] * stride});
    // the reservation past the pinned slots goes back (each region keeps [0, total * stride))
    const size_t keep = (size_t) total * stride;
    for (int h = 0; h < halves; ++h) {
        if (keep < D) munmap(map + (size_t) h * D + keep, D - keep);
        tc.mapped.push_back({(size_t) h * D, keep});
    }
    n = total;
    R.stride = stride;
    R.split_d = tc.D;
    R.registered = true;
    // ---- B8.7: what the tier got (AnonHugePages of its mappings)
    std::vector<std::pair<uintptr_t, size_t>> ranges;
    for (const auto& m : tc.mapped) ranges.push_back({(uintptr_t) (map + m.first), m.second});
    const SmapsSum sm = smaps_sum(read_file("/proc/self/smaps"), ranges);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double G = 1073741824.0, bytes = (double) keep * halves;
    int n_huge = 0;
    for (char hh : huge) n_huge += hh != 0;
    std::fprintf(stderr,
                 "glm tier: CUDA%d RAM class %d: %.2f GiB in %d segment(s) of <= %.1f GiB%s, %s, pinned in %.1f s; "
                 "THP %s (%d of %zu units madvised): AnonHugePages %.2f GiB = %.1f%% of the class\n",
                 dev, c, bytes / G, (int) pinned.size(), (double) plan.seg_slots * (double) stride / G,
                 (int64_t) total < plan.total() ? " (last one shrunk)" : "",
                 split ? "node-split (half 0 / half 1 bound to nodes)" : (bind_seg ? "each bound to a node" : "interleaved"),
                 sec, thp == 0 ? "off" : thp == 1 ? "on" : "auto", n_huge, huge.size(), (double) sm.anon_huge / G,
                 bytes > 0 ? 100.0 * (double) sm.anon_huge / bytes : 0.0);
    return map;
#else
    (void) F;
    (void) g;
    (void) c;
    (void) n;
    (void) least;
    (void) dev;
    return nullptr;
#endif
}

bool unpin_class(TierState* T, int c) {
    if (T == nullptr || c < 0 || c >= (int) T->cls.size() || !T->cls[(size_t) c].ours) return false;
#ifdef __linux__
    TierClass& tc = T->cls[(size_t) c];
    cudaSetDevice(T->dev);
    for (const auto& r : tc.regs)
        if (r.second > 0) cudaHostUnregister(tc.map + r.first);
    cudaGetLastError();
    for (const auto& m : tc.mapped)
        if (m.second > 0) munmap(tc.map + m.first, m.second);
    tc = TierClass{};
#endif
    return true;
}

// ---------------------------------------------------------------- B8.4 transfers

bool split_layer(const TierState* T, int il) { return layer_class(T, il) != nullptr; }

size_t split_offset(const TierState* T, int il) {
    const TierClass* tc = layer_class(T, il);
    return tc != nullptr ? tc->D : 0;
}

std::mutex& lane_mutex() {
    static std::mutex mu;
    return mu;
}

const uint8_t* unified(const TierState* T, int il, const uint8_t* slot, std::vector<uint8_t>& buf) {
    const TierClass* tc = layer_class(T, il);
    if (tc == nullptr || slot == nullptr || !in_class(*tc, slot)) return slot;
    const SplitLayout& L = T->L[(size_t) il];
    buf.resize(L.blob);
    gather(L, slot, slot + tc->D, buf.data());
    return buf.data();
}

bool fetch(TierState* T, const gf::MoeDev& d, int il, int k, cudaStream_t s) {
    const TierClass* tc = layer_class(T, il);
    if (tc == nullptr) return false;
    const SplitLayout& L = T->L[(size_t) il];
    const unsigned long long lo = (unsigned long long) tc->map, hi = lo + (unsigned long long) tc->n * tc->stride;
    const size_t n16 = (L.blob + 15) / 16;
    const int w = split_word(L);
    if (w == 16) split_fetch_kernel<16><<<sm_count(), 256, 0, s>>>(d, k, L, lo, hi, tc->D, n16);
    else if (w == 8) split_fetch_kernel<8><<<sm_count(), 256, 0, s>>>(d, k, L, lo, hi, tc->D, n16);
    else split_fetch_kernel<4><<<sm_count(), 256, 0, s>>>(d, k, L, lo, hi, tc->D, n16);
    launch_ok("split fetch");
    return true;
}

bool prefetch(TierState* T, const gf::MoeDev& d, int il1, cudaStream_t s) {
    const TierClass* tc = layer_class(T, il1);
    if (tc == nullptr) return false;
    const SplitLayout& L = T->L[(size_t) il1];
    const unsigned long long lo = (unsigned long long) tc->map, hi = lo + (unsigned long long) tc->n * tc->stride;
    const size_t n16 = (L.blob + 15) / 16;
    static const int blocks_start = env_int("MAYA_GLM_PREFETCH_BLOCKS", 0);
    const int blocks_env = glmknob::get(glmknob::PrefetchBlocks, blocks_start);   // GEN prefetch_blocks= (knobs)
    const int blocks = blocks_env > 0 ? blocks_env : std::max(1, sm_count() / 2);
    const int w = split_word(L);
    if (w == 16) split_prefetch_kernel<16><<<blocks, 256, 0, s>>>(d, L, lo, hi, tc->D, n16);
    else if (w == 8) split_prefetch_kernel<8><<<blocks, 256, 0, s>>>(d, L, lo, hi, tc->D, n16);
    else split_prefetch_kernel<4><<<blocks, 256, 0, s>>>(d, L, lo, hi, tc->D, n16);
    launch_ok("split prefetch");
    return true;
}

bool h2d_plain(const TierState* T, int il, const void* slot) {
    const TierClass* tc = layer_class(T, il);
    return tc == nullptr || !in_class(*tc, slot);
}

cudaError_t h2d(TierState* T, int il, void* dst, const void* slot, size_t blob, cudaStream_t s) {
    const TierClass* tc = layer_class(T, il);
    if (tc == nullptr || !in_class(*tc, slot)) return cudaMemcpyAsync(dst, slot, blob, cudaMemcpyHostToDevice, s);
    const SplitLayout& L = T->L[(size_t) il];
    cudaError_t e = cudaSuccess;
    for (const CopyOp& op : split_ops(L)) {
        const uint8_t* h = (const uint8_t*) slot + (size_t) op.half * tc->D;
        const cudaError_t e2 = cudaMemcpy2DAsync((uint8_t*) dst + op.u_off, op.u_pitch, h + op.h_off, op.h_pitch, op.width,
                                                 op.rows, cudaMemcpyHostToDevice, s);
        if (e == cudaSuccess) e = e2;
    }
    return e;
}

cudaError_t d2h(TierState* T, int il, void* slot, const void* src, size_t blob, cudaStream_t s) {
    const TierClass* tc = layer_class(T, il);
    if (tc == nullptr || !in_class(*tc, slot)) return cudaMemcpyAsync(slot, src, blob, cudaMemcpyDeviceToHost, s);
    const SplitLayout& L = T->L[(size_t) il];
    cudaError_t e = cudaSuccess;
    for (const CopyOp& op : split_ops(L)) {
        uint8_t* h = (uint8_t*) slot + (size_t) op.half * tc->D;
        const cudaError_t e2 = cudaMemcpy2DAsync(h + op.h_off, op.h_pitch, (const uint8_t*) src + op.u_off, op.u_pitch,
                                                 op.width, op.rows, cudaMemcpyDeviceToHost, s);
        if (e == cudaSuccess) e = e2;
    }
    return e;
}

bool read_split(TierState* T, int il, uint8_t* slot, size_t u0, size_t len, const std::function<void(uint8_t*)>& read) {
    const TierClass* tc = layer_class(T, il);
    if (tc == nullptr || slot == nullptr || !in_class(*tc, slot)) return false;
    static thread_local std::vector<uint8_t> buf;
    if (buf.size() < len) buf.resize(len);
    read(buf.data());
    scatter_range(T->L[(size_t) il], u0, buf.data(), len, slot, slot + tc->D);
    return true;
}

// ---------------------------------------------------------------- B8.3 / B8.4 lane

bool lane_pools(Glm5Model::FastState& F, const Glm5Geometry& g, int n_parts, int part, bool own_count, bool lane_env,
                int threads, int gpu_node, int dev) {
    TierState* T = F.tier.get();
    const std::vector<int> nodes = online_nodes();
    const bool split = T != nullptr && T->split_any();
    const char* shv = getenv("MAYA_GLM_CPU_SHARED");
    const LanePool mode = lane_pool_mode(n_parts, own_count, shv, split, (int) nodes.size(),
                                         g.arch == GlmArch::GlmDsa);
    if (T != nullptr) T->mode = mode;
    if (n_parts >= 2 || split)
        std::fprintf(stderr, "glm tier: CUDA%d CPU lane pool mode: %s (MAYA_GLM_CPU_SHARED=%s, %d part(s), %zu NUMA node(s))\n",
                     dev, lane_pool_name(mode), shv != nullptr ? shv : "unset", n_parts, nodes.size());
    if (mode != LanePool::Socket && mode != LanePool::NodeSplit) return false;
    const auto per_node = [&](int node) {
        if (const int t = env_int("MAYA_GLM_NUMA_LANE_THREADS", 0); t > 0) return std::min(64, t);
        if (lane_env) return std::max(2, threads);
        const std::vector<int> cpus = node_cpus(node);
        return numa_lane_threads((int) cpus.size(), physical_cores(cpus));
    };
    if (mode == LanePool::Socket) {
        int node = -1;
        for (int nd : nodes)
            if (nd == gpu_node) node = nd;
        if (node < 0) node = nodes[(size_t) part % nodes.size()];
        auto ln = socket_lane(F, g, node, per_node(node), dev);   // W6 runtime fix: held for the load (shared)
        F.cpu_pool = ln->pool;
        F.cpu_node = node;
        F.cpu_pin = ln->cpus;
        std::fprintf(stderr, "glm tier: CUDA%d CPU lane: the pool of NUMA node %d (%d threads on its %zu CPUs)\n", dev,
                     node, ln->pool->size(), ln->cpus.size());
        return true;
    }
    // node-split: both sockets' pools; the service thread sits on node 0 with its pool
    for (int h = 0; h < 2; ++h) T->lane[h] = node_lane(nodes[(size_t) h], per_node(nodes[(size_t) h]));
    F.cpu_pool = T->lane[0]->pool;
    F.cpu_node = nodes[0];
    F.cpu_pin = T->lane[0]->cpus;
    const int64_t H = std::max<int64_t>(1, g.n_ff_exp / 2);
    T->act.assign(kc::kNativeActBytes, 0);
    // each half's scratch is first touched by a thread pinned to its node (its pool writes and reads it every MoE
    // layer; touched here, node 1's would sit on node 0 behind UPI).  act stays with the service thread (node 0).
    for (int h = 0; h < 2; ++h) {
        std::thread touch([&, h] {
#ifdef __linux__
            glmfast::pin_thread(pthread_self(), T->lane[h]->cpus);   // pinned before it allocates and touches
#endif                                                               // (pin_thread is a no-op off Linux)
            T->ff[h].assign((size_t) 8 * H, 0.0f);
            T->hq[h].assign((size_t) 8 * kc::kNativeHBytes, 0);
            T->dn[h].assign((size_t) 8 * g.n_embd, 0.0f);
            T->part[h].assign((size_t) g.n_embd, 0.0f);
        });
        touch.join();
    }
    std::fprintf(stderr, "glm tier: CUDA%d CPU lane node-split: node %d %d threads, node %d %d threads\n", dev, nodes[0],
                 T->lane[0]->pool->size(), nodes[1], T->lane[1]->pool->size());
    return true;
}

std::shared_ptr<NodeLane> socket_lane(Glm5Model::FastState& F, const Glm5Geometry& g, int node, int threads, int dev) {
    // the registry holds weak references: the part's tier state keeps the lane (and its pool) alive for the load, so
    // every later part on the node finds this one; the last part's teardown lets it go
    TierState& T = tier_of(F, g, dev, online_nodes());
    T.socket_lane = node_lane(node, threads);
    return T.socket_lane;
}

bool set_lane_threads(Glm5Model::FastState& F, int n) {
    TierState* T = F.tier.get();
    if (T == nullptr || T->mode != LanePool::NodeSplit || T->lane[0] == nullptr || T->lane[1] == nullptr) return false;
    // n counts node 0's pool (the one cpu_lane_threads reports); node 1's scales with its size, so both halves of
    // every expert slow down alike and cpu_experts' wait on the two lanes follows the knob
    const int ref = T->lane[0]->pool->size();
    for (int h = 0; h < 2; ++h) {
        glmfast::Workers& pool = *T->lane[h]->pool;
        pool.set_active(n > 0 ? lane_threads_scaled(n, ref, pool.size()) : pool.size());
    }
    return true;
}

bool cpu_experts(Glm5Model::FastState& F, const Glm5Geometry& g, int il, int ne, const uint8_t* const* blob,
                 const float* w, const float* x, float* out) {
    TierState* T = F.tier.get();
    const TierClass* tc = layer_class(T, il);
    if (tc == nullptr) return false;
    if (T->lane[0] == nullptr || T->lane[1] == nullptr || T->half[(size_t) il].n_ff == 0) {
        // (a split tier with the lane in another mode cannot happen: lane_pools picks NodeSplit whenever a class split)
        std::fprintf(stderr, "glm tier: the CPU lane has no node-split pools for layer %d\n", il);
        std::abort();
    }
    const SplitLayout& L = T->L[(size_t) il];
    const kc::NativeFmt& hf = T->half[(size_t) il];
    const kc::NativeFmt& nf = F.cpu_fmt[(size_t) il];
    const int H = (int) L.H, n_embd = (int) L.n_embd;
    const float limit = g.swiglu_exp;
    static const int kSplitStart = [] {
        const char* v = getenv("MAYA_GLM_CPU_SPLIT");
        return v ? std::max(1, std::min(64, std::atoi(v))) : 0;
    }();
    const int kSplit = glmknob::get(glmknob::CpuSplit, kSplitStart);   // GEN cpu_split= (knobs)
    // the node lanes are shared by a load's parts, and two parts can run at once (the pipelined MTP decode)
    std::lock_guard<std::mutex> one(lane_mutex());   // (B9.9: the prompt's node-split share takes it too)
    kc::native_quant_act(nf, x, T->act.data());
    std::function<void()> task[2];
    for (int h = 0; h < 2; ++h)
        task[h] = [&, h] {
            glmfast::Workers& pool = *T->lane[h]->pool;
            const int threads = pool.active();
            const int J = threads * (kSplit > 0 ? kSplit : std::max(1, std::min(8, 48 / std::max(1, threads))));
            const size_t off = (size_t) h * tc->D;
            float* ff = T->ff[h].data();
            uint8_t* hq = T->hq[h].data();
            float* dn = T->dn[h].data();
            const int gu_tot = ne * H;
            pool.run(J, [&](int job) {
                const int q1 = (int) ((int64_t) gu_tot * (job + 1) / J);
                for (int q = (int) ((int64_t) gu_tot * job / J); q < q1;) {
                    const int i = q / H, r0 = q % H, r1 = std::min(H, r0 + (q1 - q));
                    half_gu_rows(hf, L, blob[i] + off, T->act.data(), ff + (size_t) i * H, r0, r1, limit);
                    q += r1 - r0;
                }
            });
            for (int i = 0; i < ne; ++i) kc::native_quant_h(hf, ff + (size_t) i * H, hq + (size_t) i * kc::kNativeHBytes);
            const int dn_tot = ne * n_embd;
            pool.run(J, [&](int job) {
                const int q1 = (int) ((int64_t) dn_tot * (job + 1) / J);
                for (int q = (int) ((int64_t) dn_tot * job / J); q < q1;) {
                    const int i = q / n_embd, r0 = q % n_embd, r1 = std::min(n_embd, r0 + (q1 - q));
                    half_down_rows(hf, L, blob[i] + off, hq + (size_t) i * kc::kNativeHBytes, dn + (size_t) i * n_embd,
                                   r0, r1);
                    q += r1 - r0;
                }
            });
            float* part = T->part[h].data();
            for (int r = 0; r < n_embd; ++r) {
                float m = 0.0f;
                for (int i = 0; i < ne; ++i) m += w[i] * dn[(size_t) i * n_embd + r];
                part[r] = m;
            }
        };
    T->lane[1]->start(&task[1]);
    T->lane[0]->start(&task[0]);
    T->lane[0]->wait();
    T->lane[1]->wait();
    add_partial(T->part[0].data(), T->part[1].data(), out, n_embd);
    T->split_calls.fetch_add(1, std::memory_order_relaxed);
    return true;
}

}  // namespace strata::core::glmtier

// ---------------------------------------------------------------- the split planner's view of the lane pools
// (include/strata/core/glm_dsa_split.hpp: glm_split_pool_ids; glm_model.cu glm_search_bounds prices misses and the
// B11.2 head with them)
namespace strata::core {

namespace {
// the NUMA node a GPU hangs off (sysfs, by its PCI address), -1 when unknown
int dev_numa_node(int dev) {
    int node = -1;
#ifdef __linux__
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, (int) sizeof bus, dev) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    std::string b(bus);
    for (char& ch : b) ch = (char) std::tolower((unsigned char) ch);
    std::ifstream("/sys/bus/pci/devices/" + b + "/numa_node") >> node;
#else
    (void) dev;
#endif
    return node;
}
}  // namespace

std::vector<int> glm_split_pool_ids(const Glm5Geometry& g, const std::vector<int>& devs) {
    std::vector<int> gpu_node;
    for (int d : devs) gpu_node.push_back(dev_numa_node(d));
    return glm_cpu_pool_ids(g, (int) devs.size(), std::getenv("MAYA_GLM_CPU_SHARED"), glmtier::online_nodes(),
                            gpu_node);
}

}  // namespace strata::core
