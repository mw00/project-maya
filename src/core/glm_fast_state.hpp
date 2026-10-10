// src/core/glm_fast_state.hpp - the fast path's private state (Glm5Model::FastState) and host helpers, shared by
// src/core/glm_fast_path.cu (the one-token decode path) and src/core/glm_prefill.cu (the batched prompt path).
// Not a public header: glm_model.cu also uses it to reset request-local routing errors.
#pragma once

#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "glm_memory.hpp"
#include "glm_layer_graphs.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::core {
namespace glmfast {
inline bool cuda_ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}
inline bool wait_event(cudaEvent_t ev, const char* what, std::string& err) {
    cudaError_t e;
    while ((e = cudaEventQuery(ev)) == cudaErrorNotReady) std::this_thread::yield();
    return cuda_ok(e, what, err);
}

namespace gf = strata::kernels::glmf;

inline void cpu_relax() {
#if defined(__x86_64__) && !defined(_WIN32)
    asm volatile("pause" ::: "memory");
#endif
}

// a thread's CPUs: `cpus` (empty: left to the OS).  Linux only; elsewhere a no-op.
inline void pin_thread(std::thread::native_handle_type h, const std::vector<int>& cpus) {
#ifdef __linux__
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
        if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
    pthread_setaffinity_np(h, sizeof set, &set);
#else
    (void) h;
    (void) cpus;
#endif
}

// a fixed pool of worker threads: run(n, fn) calls fn(0..n-1) across the workers AND the caller and returns
// when all finished.  Jobs are claimed lock-free; a worker spins ~300 us for the next batch before it sleeps,
// because one miss request runs its batches back to back (reads, gate/up, down) and a futex wake per batch
// was most of the CPU path's latency.
class Workers {
public:
    // spin_us: how long an idle worker spins for the next batch before it sleeps; cpus: the CPUs the workers run on
    // (empty: any)
    explicit Workers(int n, int spin_us = 300, const std::vector<int>& cpus = {}) : spin_us_(spin_us), active_(n + 1) {
        for (int i = 0; i < n; ++i) {
            th_.emplace_back([this, i] { loop(i); });
            pin_thread(th_.back().native_handle(), cpus);
        }
    }
    ~Workers() {
        quit_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    void run(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        std::lock_guard<std::mutex> one(run_mu_);   // a pool a split's parts share: one batch at a time
        const uint64_t g = (gen_.load(std::memory_order_relaxed) + 1) & 0xffffffffu;
        fn_ = &fn;
        total_.store(n, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        // the ticket packs (generation, next index): a straggler of an older batch can never claim (or burn)
        // an index of this one - its CAS fails on the generation
        ticket_.store(g << 32, std::memory_order_release);
        gen_.store(g, std::memory_order_release);
        if (sleeping_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
        work(g);
        while (done_.load(std::memory_order_acquire) < n) cpu_relax();
    }
    int size() const { return (int) th_.size() + 1; }
    // only the caller and the first n - 1 workers take jobs (fewer threads without a restart: setup's calibration);
    // a batch's job count should be active() for one job a thread
    void set_active(int n) { active_.store(std::max(1, std::min(n, size())), std::memory_order_release); }
    int active() const { return active_.load(std::memory_order_acquire); }

private:
    void work(uint64_t g) {
        for (;;) {
            uint64_t t = ticket_.load(std::memory_order_acquire);
            if ((t >> 32) != g) return;
            const int n = total_.load(std::memory_order_acquire);
            if ((int) (t & 0xffffffffu) >= n) return;
            if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            (*fn_)((int) (t & 0xffffffffu));
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    void loop(int id) {
        uint64_t seen = gen_.load();
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            int spins = 0;
            // a worker set_active left out sleeps at once: spinning, it held a CPU the active ones and the disk
            // readers wanted, so the setup's calibration timed fewer threads on a CPU still full of spinners
            // (24-core Ultra 9 275HX, no SMT, Windows: a decode's disk wait 16.8 ms with 24 threads, 8.1 ms with 16)
            const bool idle = id + 1 >= active_.load(std::memory_order_acquire);
            while (gen_.load(std::memory_order_acquire) == seen) {
                if (quit_.load(std::memory_order_relaxed)) return;
                cpu_relax();
                if (idle || ((++spins & 255) == 0 &&
                             std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_))) {
                    std::unique_lock<std::mutex> lk(mu_);
                    sleeping_.fetch_add(1);
                    cv_.wait(lk, [&] { return quit_.load() || gen_.load() != seen; });
                    sleeping_.fetch_sub(1);
                    break;
                }
            }
            if (quit_.load()) return;
            seen = gen_.load(std::memory_order_acquire);
            if (id + 1 < active_.load(std::memory_order_acquire)) work(seen);
        }
    }
    int spin_us_;
    std::vector<std::thread> th_;
    std::mutex mu_, run_mu_;
    std::condition_variable cv_;
    const std::function<void(int)>* volatile fn_ = nullptr;
    std::atomic<int> total_{0}, done_{0}, sleeping_{0}, active_;
    std::atomic<uint64_t> ticket_{0}, gen_{0};
    std::atomic<bool> quit_{false};
};

// [off, off + len) of a shard into dst (O_DIRECT when the shard has a direct fd; see glm_fast_path.cu)
void read_slice(const Glm5Model::Shard& sh, uint64_t off, size_t len, uint8_t* dst);

// A layer's VRAM slot stride: the expert blob ([gate | up | down]) rounded up so that both matrices' block sizes and
// 16 bytes divide it - llama.cpp's MMQ addresses expert e of a pool partition as base + e * stride in WHOLE blocks,
// so the prompt path multiplies the resident experts where they lie.
inline size_t expert_stride(size_t blob, int gu_type, int d_type) {
    size_t a = 16;
    const size_t b1 = ggml_type_size((ggml_type) gu_type), b2 = ggml_type_size((ggml_type) d_type);
    if (b1 > 0) a = std::lcm(a, b1);
    if (b2 > 0) a = std::lcm(a, b2);
    return (blob + a - 1) / a * a;
}

}  // namespace glmfast

namespace gf = strata::kernels::glmf;

struct Glm5Model::FastState {
    bool unified_memory = false;   // set only for integrated HIP devices on Linux; CUDA and Windows keep the discrete policy
    cudaStream_t cs = nullptr, copy = nullptr, ps = nullptr;   // ps: the prefetch side stream
    cudaEvent_t ev_hop = nullptr, ev_done = nullptr, ev_pred = nullptr, ev_pf = nullptr, ev_pf_prev = nullptr;
    int max_pf = 0;   // STRATA_GLM_PREFETCH_N: measured a net LOSS on Mercury (the window between routes is shorter than one fetch)
    std::atomic<uint64_t> prefetches{0};
    unsigned long long* pf_buf[2] = {nullptr, nullptr};   // [src 8 | dst 8] per layer parity
    int* pf_n_buf[2] = {nullptr, nullptr};
    void* arena = nullptr;
    size_t arena_bytes = 0;
    // activations (device)
    float *x = nullptr, *mixer = nullptr, *ffn = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    float* part = nullptr;
    unsigned int* counter = nullptr;
    void* xq = nullptr;
    float* proj[3] = {nullptr, nullptr, nullptr};
    float* conv[3] = {nullptr, nullptr, nullptr};
    float *fa = nullptr, *ga = nullptr, *beta = nullptr, *g1 = nullptr, *g2 = nullptr;
    void* gated_q = nullptr;
    float *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *ik_raw = nullptr, *ig_raw = nullptr, *iw = nullptr;
    float *q = nullptr, *iq = nullptr, *score = nullptr;
    int* cells = nullptr;
    void *qr_q = nullptr, *attn_q = nullptr;
    float *rlog = nullptr, *plog = nullptr, *sh_g = nullptr, *sh_u = nullptr;
    float* sh_out = nullptr;   // the shared expert's down (n_embd), computed alongside the routed gate/up
    // prediction quality (the next layer's router on this layer's FFN input)
    std::vector<std::array<int, 8>> pred_of;   // per layer: the prediction made one layer earlier
    std::atomic<uint64_t> pred_n{0}, pred_overlap{0}, pred_miss{0}, pred_miss_hit{0};
    // ---- LOOKAHEAD (STRATA_GLM_AHEAD=<n layers>, 0 off): every route also names the top-k of the next n layers' routers
    //      on its FFN input; the disk-only experts among them are read into the RAM tier by the reader threads while
    //      the layers in between compute, so their own route finds them on the host (a short wait or none)
    int n_ahead = 0;
    bool ahead_read = false;                   // STRATA_GLM_AHEAD_READ=0: predict and count only
    float* alog = nullptr;                     // kAhead x n_expert logits (device)
    std::vector<std::array<std::array<short, 8>, gf::kAhead>> ah_pred;   // per layer: the predictions d+1 layers early
    struct AheadJob {
        int rclass, rslot, il, e;
        uint64_t tag;
    };
    std::mutex ah_mu;                          // the job queue (never held with mu)
    std::condition_variable ah_cv;
    std::vector<AheadJob> ah_q;
    std::vector<std::thread> ah_th;
    bool ah_quit = false;
    std::atomic<int> ah_inflight{0};
    // per RAM class and slot, for kRLoad slots: tag << 2 | phase (0 queued, 1 reading, 2 landed) - a job reads only while
    // the tag it carries is current (a slot freed and reused under it gets a new tag)
    std::vector<std::unique_ptr<std::atomic<uint64_t>[]>> rload;
    uint64_t ah_tag = 0;
    // stats (the service thread)
    uint64_t ah_routes = 0, ah_ov[gf::kAhead] = {0, 0, 0, 0}, ah_npred[gf::kAhead] = {0, 0, 0, 0};
    uint64_t ah_disk = 0, ah_disk_cov[gf::kAhead] = {0, 0, 0, 0}, ah_disk_any = 0;
    uint64_t ah_issued = 0, ah_used = 0, ah_waited = 0, ah_stolen = 0, ah_full = 0, ah_nofree = 0;
    std::atomic<uint64_t> ah_wait_us{0};
    void *sh_hq = nullptr, *hq = nullptr;
    float *dg = nullptr, *du = nullptr;
    void* dhq = nullptr;
    float* head_x = nullptr;
    void* head_xq = nullptr;
    float* emb = nullptr;
    int* tok = nullptr;
    // the NextN block (the half that carries it): its residual, eh_proj's q8_1 input, its logits and token
    float* mtp_h = nullptr;
    void* mtp_catq = nullptr;
    float* mtp_logits = nullptr;
    int* mtp_tok = nullptr;
    int* mtp_tok_h = nullptr;    // pinned
    cudaEvent_t ev_mtp = nullptr;
    // routing
    gf::MoeDev md;
    unsigned long long* tab = nullptr;
    gf::MoeRequest* ring_h = nullptr;
    gf::MoeResponse* resp_h = nullptr;
    float* emb_h = nullptr;     // pinned
    float* hop_h = nullptr;     // pinned, hc * n_embd
    int* tok_h = nullptr;       // pinned
    // per layer: the resolved weights
    struct Layer {
        bool recr = false, moe = false, mtp = false;
        WSlot eh;                                                   // the NextN block: eh_proj and its norms
        const float *enorm = nullptr, *hnorm = nullptr, *shnorm = nullptr;
        const uint16_t *hc_attn_fn = nullptr, *hc_ffn_fn = nullptr;
        const float *hc_attn_scale = nullptr, *hc_attn_base = nullptr, *hc_ffn_scale = nullptr, *hc_ffn_base = nullptr;
        const float *attn_norm = nullptr, *ffn_norm = nullptr;
        WSlot q, k, v, out;
        const uint16_t *f_a = nullptr, *g_a = nullptr, *f_b = nullptr, *g_b = nullptr, *beta = nullptr;
        const float *conv[3] = {nullptr, nullptr, nullptr};
        const float *dt_bias = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
        WSlot q_a, q_b, kv_a;
        const float *q_a_norm = nullptr, *kv_a_norm = nullptr, *k_norm_w = nullptr, *k_norm_b = nullptr, *ape = nullptr;
        const uint16_t *idx_k = nullptr, *idx_gate = nullptr, *idx_q_b = nullptr, *idx_proj = nullptr;
        const uint16_t *k_b = nullptr, *v_b = nullptr;
        const uint16_t* router = nullptr;
        const float* router_bias = nullptr;
        WSlot sh_gate, sh_up, sh_down;
        WSlot ffn_gate, ffn_up, ffn_down;
        int gu_type = 0, d_type = 0;
        size_t blob = 0, down_off = 0, gu_bytes = 0, dn_bytes = 0;
    };
    std::vector<Layer> L;
    // ---- the VRAM tier: per-layer partitions.  A slot is FREE, RESIDENT (in tab), a SPARE (handed to the
    //      device, which may fill it during a token and mark it resident itself) or DRAINING (its old expert
    //      is being demoted to the RAM tier by a D2H; it becomes free when that copy lands).
    // kLanding: a background promotion's copy is on its way into the slot (fast_boundary step 5)
    enum : char { kFree = 0, kResident = 1, kSpare = 2, kDraining = 3, kLent = 4, kLanding = 5 };
    // A layer's slots [0, n_main) are at base, [n_main, n) at xbase: every layer's tail lies in ONE region (xpool),
    // which the prompt path borrows for its buffers while a prompt runs (those slots are kLent then) - and an
    // on-demand vision encoder while it encodes (the region is freed; xbase is null until it comes back).
    struct LayerPool {
        uint8_t* base = nullptr;
        uint8_t* xbase = nullptr;
        size_t stride = 0;
        int n = 0, n_main = 0;
        uint8_t* slot_ptr(int s) const {
            return s < n_main ? base + (size_t) s * stride : xbase + (size_t) (s - n_main) * stride;
        }
        int slot_index(unsigned long long p) const {
            const unsigned long long b0 = (unsigned long long) base, x0 = (unsigned long long) xbase;
            if (p >= b0 && p < b0 + (unsigned long long) n_main * stride) return (int) ((p - b0) / stride);
            if (xbase != nullptr && p >= x0 && p < x0 + (unsigned long long) (n - n_main) * stride)
                return n_main + (int) ((p - x0) / stride);
            return -1;
        }
        std::vector<int> key;        // slot -> expert, -1 none
        std::vector<char> st;
        std::vector<uint64_t> tick;  // recency (LFU tie-break)
        int spare[gf::kSpares] = {-1, -1, -1};   // the device's spare table, mirrored (slot or -1)
        uint64_t evictions = 0;
    };
    std::vector<LayerPool> lp;   // indexed by absolute layer
    std::vector<int> slot_of;    // il * n_expert + e -> slot in lp[il], -1 absent
    std::vector<uint32_t> cnt;   // il * n_expert + e -> LFU count (aged), over both tiers
    // ... and the long memory: routes per key across sessions (expert_usage.txt), halved when one passes 2^30 -
    // the warm-up fills the tiers in its order (this user's experts first)
    std::vector<uint32_t> usage;
    uint64_t cnt_events = 0;
    uint8_t* pool = nullptr;
    // the main slots' further allocations, where one did not allocate (each holds whole layers: lp[il].base)
    std::vector<uint8_t*> pool_more;
    size_t pool_bytes = 0;
    int64_t pool_slots = 0;
    // the lendable tail (every layer's slots [n_main, n)) is its own allocation: an on-demand vision encoder can
    // have that memory while it runs (vision_lend frees it, vision_reclaim allocates it again)
    uint8_t* xpool = nullptr;
    size_t xpool_bytes = 0;
    // ---- the RAM tier: pinned host slots per blob size class, holding experts that are NOT in VRAM
    //      (exclusive tiers).  rtab (on the device) points at HOLDING slots only.
    // kRPin: still the expert's live copy, being promoted (never a victim until the copy landed)
    enum : char { kRFree = 0, kRHold = 1, kRRelease = 2, kRDemote = 3, kRNew = 4, kRLoad = 5, kRPin = 6 };
    struct RamClass {
        size_t stride = 0;
        uint8_t* base = nullptr;
        int n = 0;
        bool registered = false;   // numa_pinned's (mmap + cudaHostRegister), not cudaHostAlloc's
        std::vector<int> key;   // slot -> il * n_expert + e, -1 none
        std::vector<uint64_t> tick;   // the route clock when a route last used it (or it arrived): eviction's recency
        std::vector<char> st;   // kRFree / kRHold / kRRelease (promoted: free at the boundary) /
                                // kRDemote (a D2H landing) / kRNew (a disk read this token: rtab at the boundary) /
                                // kRLoad (a LOOKAHEAD read, maybe still in flight: rload says; rtab once it landed)
    };
    std::vector<RamClass> rc;
    std::vector<int> layer_rc;   // layer -> RAM class
    std::vector<int> ram_of;     // key -> slot in its class, -1
    // STRATA_GLM_TIER_DIAG=1 (debug): how each expert last left the tiers (0 never held, 1 dropped from VRAM, 2 evicted
    // from RAM, 3 dropped by the prompt path's lending), and the disk reads by that cause
    std::vector<uint8_t> left;
    uint64_t diag_disk[4] = {0, 0, 0, 0}, diag_drop = 0, diag_ram_evict = 0, diag_lend = 0, diag_lend_park = 0, diag_b = 0;
    uint64_t diag_resident_skip = 0;   // RAM-resident mode: spare refills skipped (no free RAM slot, no drop)
    uint64_t diag_dup = 0, diag_adopt = 0;   // the boundary's clean-up: duplicate RAM copies freed, lost ones adopted
    size_t ram_bytes = 0;
    // ---- demotions VRAM -> RAM (copy stream, D2H) of the victims that make room for spares
    struct Drain {
        int il, vslot, key, rclass, rslot;
        cudaEvent_t ev;
    };
    std::vector<Drain> draining;
    // ---- background moves between the tiers (fast_boundary step 5): a promotion (up, RAM -> VRAM) or a demotion
    //      (VRAM -> RAM), both on the copy stream; the expert stays live where it was until its copy landed
    struct BgMove {
        int il, vslot, key, rclass, rslot;
        bool up;
        cudaEvent_t ev;
    };
    std::vector<BgMove> bg;
    bool bg_hold = false;   // lend_tail: no new moves while the pool's tail is lent
    volatile int* route_error_h = nullptr;   // in the mapped ring allocation, visible after compute completion
    uint64_t bg_up = 0, bg_down = 0;
    std::vector<cudaEvent_t> ev_free;
    int* upd_key_h = nullptr;
    unsigned long long* upd_val_h = nullptr;
    int* upd_key_d = nullptr;
    unsigned long long* upd_val_d = nullptr;
    static constexpr int kMaxUpd = 8192;
    // a batch holds each table key ONCE (its last value): the update kernel writes the batch in parallel, so a key
    // edited twice in one boundary (made live, then freed as the coldest) had no defined winner
    std::vector<int> upd_at;          // table key -> its index in the current batch, valid when upd_gen[key] == upd_g
    std::vector<uint32_t> upd_gen;
    uint32_t upd_g = 0;
    int n_keys = 0;
    uint8_t* scratch = nullptr;
    std::vector<uint8_t*> stage;   // pinned: disk reads when no RAM slot is free
    std::mutex mu;   // the service thread vs the between-token boundary
    std::unique_ptr<glmfast::Workers> workers;   // parallel disk reads
    // ---- the CPU LANE (STRATA_GLM_CPU_LANE=<threads>, 0 off): a route's RAM-tier experts split between the device's
    //      PCIe pulls and this pool, which computes its share from the pinned blobs meanwhile; the device adds the
    //      weighted sum (cpu_ans_h) before its down combine.  cpu_plan: how many of f RAM-tier experts go to the host.
    //      A split of 3+ GPUs shares one pool across its parts (fast_cpu_lane_setup).
    std::shared_ptr<glmfast::Workers> cpu_pool;
    int cpu_node = -1;                         // the NUMA node the pool is pinned to (-1: unpinned) ...
    std::vector<int> cpu_pin;                  // ... and its CPUs (the service thread runs there too)
    std::string cal_file, cal_key;             // STRATA_GLM_CPU_CAL
    double cal_ref = 0.0;                      // decode's CPU ms an expert after calibrating (0: not measured yet)
    uint64_t drift_e0 = 0, drift_us0 = 0;
    int drift_n = 0, drift_dir = 0;            // windows in a row >15% off, and which way
    double cpu_c_ms = 0.0, cpu_p_ms = 0.0;   // the lane's calibration: an expert on the CPU, one over PCIe
    double cpu_ps_ms = 0.0;                  // ... one in a stream of copies (the prompt's staging)
    unsigned long long cpu_plan = 0;
    unsigned long long cpu_plan_start = 0;   // ... the plan the engine started with (set_pcie_share(-1) restores it)
    gf::CpuAnswer* cpu_ans_h = nullptr;                      // host-mapped
    unsigned int* cpu_seq_d = nullptr;
    unsigned int* dcnt_d = nullptr;                          // the device's route counts (the coldest go to the host)
    std::vector<strata::kernels::cpu::NativeFmt> cpu_fmt;   // per layer; n_ff == 0: the lane skips that layer
    std::vector<uint8_t> cpu_act, cpu_hq;                    // x's activation quant; 8 h quants
    std::vector<float> cpu_ff, cpu_dn;                       // 8 x n_ff; 8 x n_embd
    std::atomic<uint64_t> cpu_experts{0}, cpu_routes{0}, cpu_us{0};
    std::thread svc;
    std::atomic<bool> quit{false};
    uint64_t clock = 0;
    uint64_t expected = 0;               // requests this half's device has issued (one per MoE layer per token)
    std::atomic<uint64_t> processed{0};  // ... and the service thread has consumed
    // stats (written by the service thread, read by anyone)
    std::atomic<uint64_t> hits{0}, misses{0}, miss_layers{0}, ram_hits{0}, disk_reads{0}, promotions{0};
    std::atomic<uint64_t> scratch_uses{0}, demotions{0}, drops{0};
    std::atomic<uint64_t> miss_us{0}, disk_us{0};
    uint64_t tokens = 0;
    double ms = 0;
    bool timing = false;
    bool ram_resident = false;   // STRATA_GLM_RAM_RESIDENT=1 / --glm-ram-resident: the RAM tier holds EVERY expert
                                 // VRAM does not, nothing ever drops to disk; the start fails if the tier is small
    int ram_slack = 0;           // extra RAM-tier slots per MoE layer (STRATA_GLM_RAM_SLACK; default 16 resident)
    // ---- STRATA_GLM_PROF=1: a cudaEvent after every launch, the gaps summed per kernel name (debug only)
    bool prof_on = false;
    bool kda_graph_on = false;   // experimental, single-device ordinary decode only
    glmfast::LayerGraphs kda_graphs;
    std::vector<cudaEvent_t> pev;
    std::vector<const char*> pname;
    size_t pn = 0;
    std::map<std::string, double> pacc;
    uint64_t ptokens = 0;
    void mark(const char* nm) {
        if (pn >= pev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            pev.push_back(e);
            pname.push_back(nm);
        }
        cudaEventRecord(pev[pn], cs);
        pname[pn] = nm;
        ++pn;
    }
    uint64_t pseen = 0, pskip = 0;   // STRATA_GLM_PROF=<n>: the first n tokens (the cold tiers) are not counted
    void collect() {
        if (pseen++ < pskip) {
            pn = 0;
            return;
        }
        for (size_t i = 1; i < pn; ++i) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, pev[i - 1], pev[i]) == cudaSuccess) pacc[pname[i]] += ms;
        }
        if (pn > 0) ++ptokens;
        pn = 0;
    }
    size_t tab_key(int key) const { return (size_t) key; }
    size_t rtab_key(int key) const { return (size_t) n_keys + (size_t) key; }
    size_t spare_key(int il, int j) const { return 2 * (size_t) n_keys + (size_t) il * gf::kSpares + (size_t) j; }
    cudaEvent_t get_event() {
        if (!ev_free.empty()) {
            cudaEvent_t e = ev_free.back();
            ev_free.pop_back();
            return e;
        }
        cudaEvent_t e = nullptr;
        cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        return e;
    }
    bool route_ok(std::string& err) const {
        const int e = route_error_h ? *route_error_h : 0;
        if (e == 0) return true;
        const char* kind = (e % 4 == 1) ? "decode" : (e % 4 == 2) ? "predicted" : "lookahead";
        err = "glm router: layer " + std::to_string(e / 4) + " " + kind + " invalid expert ID or non-finite score";
        return false;
    }
};


}  // namespace strata::core
