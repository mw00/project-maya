// src/core/glm_fast_state.hpp - the fast path's private state (Glm5Model::FastState) and host helpers, shared by
// src/core/glm_fast_path.cu (the one-token decode path) and src/core/glm_prefill.cu (the batched prompt path).
// Not a public header: glm_model.cu also uses it to reset request-local routing errors.
#pragma once

#include "strata/core/glm_model.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/glm_kv_stream.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "glm_memory.hpp"
#include "glm_layer_graphs.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
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

// Windows' HIP runtime holds the launches a stream queues until the host waits on the GPU (Linux and CUDA submit each),
// so a token's first layers sat queued while the main thread went on enqueuing the rest, until fast_service's 1 ms kick
// (or the token's own wait) submitted them.  This submits what `s` has queued without waiting: cudaStreamQuery, whose
// "not ready" is no error - it is taken back off the thread's last error, which launch_check reads after every launch.
// A launch error already pending stays for the caller's check (no submit then).  Elsewhere a no-op.
// STRATA_GLM_FLUSH=<n>: the decode submits after its first layer and then every n layers (submit_every), 0: never -
// the default: on the 8065S (Maya-S, drafting with the MTP rework) a submission every 4 layers lost ~4% of the decode
// (26.5 against 27.7 tok/s, same answers); the runtime's own batching plus the service thread's kick keep the GPU fed.
inline void submit_queued(cudaStream_t s) {
#if defined(_WIN32) && defined(STRATA_USE_HIP)
    if (cudaPeekAtLastError() != cudaSuccess) return;
    const cudaError_t e = cudaStreamQuery(s);
    if (e == cudaErrorNotReady || e == cudaSuccess) (void) cudaGetLastError();
#else
    (void) s;
#endif
}
inline int submit_every() {
#if defined(_WIN32) && defined(STRATA_USE_HIP)
    static const int v = [] {
        const char* e = getenv("STRATA_GLM_FLUSH");
        return e != nullptr ? std::max(0, std::atoi(e)) : 0;
    }();
    return v;
#else
    return 0;
#endif
}
// ... and the prompt path after each mixer sub-batch and each layer's experts: STRATA_GLM_PREFILL_FLUSH=1 (off by
// default: Maya-S on the 8065S, 2K / 8K / 16K prompts 271 / 323 / 325 tok/s with it against 273-276 / 323-325 /
// 329-339 without - the sub-batches are long enough that the runtime's batching costs nothing)
inline bool prefill_submit() {
#if defined(_WIN32) && defined(STRATA_USE_HIP)
    static const bool v = [] {
        const char* e = getenv("STRATA_GLM_PREFILL_FLUSH");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v;
#else
    return false;
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
// `concurrent`: several callers' batches at once (a pool a split's parts share - the parts' CPU-lane experts):
// each run() takes one of kSlots batch slots, works its own batch and waits for it, and the workers take jobs from
// the batch of the highest priority (set_thread_prio: the caller thread's; ties to the lower slot) a job at a time.
// One batch at a time (the default) held a part's route behind another's whole batch - a lone expert's down
// pieces leave most threads idle, and those threads now take the other part's jobs.
class Workers {
public:
    // spin_us: how long an idle worker spins for the next batch before it sleeps; cpus: the CPUs the workers run on
    // (empty: any)
    explicit Workers(int n, int spin_us = 300, const std::vector<int>& cpus = {}, bool concurrent = false)
        : spin_us_(spin_us), active_(n + 1), concurrent_(concurrent) {
        for (int i = 0; i < n; ++i) {
            th_.emplace_back([this, i] { concurrent_ ? loop_c(i) : loop(i); });
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
        if (concurrent_) {
            run_c(n, fn);
            return;
        }
        std::lock_guard<std::mutex> one(run_mu_);   // a pool a split's parts share: one batch at a time
        const uint64_t g = (gen_.load(std::memory_order_relaxed) + 1) & 0xffffffffu;
        // the ticket packs (generation, next index): a straggler of an older batch can never claim (or burn)
        // an index of this one - its CAS fails on the generation.  The old batch's ticket is CLOSED before total_
        // changes: a straggler that read the spent ticket (old generation, index n_old) and then this batch's larger
        // total_ would otherwise claim index n_old of the old batch - and run this batch's job with it, done_ counting
        // it too, so run() could return while a job still ran (a crash once the caller's frame was gone)
        ticket_.store((g << 32) | 0x7fffffffu, std::memory_order_seq_cst);   // (an index no batch reaches)
        fn_ = &fn;
        total_.store(n, std::memory_order_release);
        done_.store(0, std::memory_order_relaxed);
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
    // the priority of this thread's batches on a concurrent pool (higher first; default 0)
    static void set_thread_prio(int p) { tl_prio() = p; }

private:
    static int& tl_prio() {
        static thread_local int p = 0;
        return p;
    }
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
    // ---- the concurrent pool: a slot's ticket packs (generation, next index) as above, its own generation
    static constexpr int kSlots = 16;
    static constexpr uint64_t kClosed = 0x7fffffffu;
    struct Slot {
        std::atomic<uint64_t> ticket{kClosed};
        std::atomic<int> total{0}, done{0}, prio{0};
        std::atomic<const std::function<void(int)>*> fn{nullptr};
        std::atomic<bool> busy{false};
        uint64_t gen = 0;   // (its owner's alone)
    };
    // one job of slot s's batch g (false: none left to claim)
    bool claim(Slot& S, uint64_t g) {
        for (;;) {
            uint64_t t = S.ticket.load(std::memory_order_acquire);
            if ((t >> 32) != g || (t & 0xffffffffu) == kClosed) return false;
            if ((int) (t & 0xffffffffu) >= S.total.load(std::memory_order_acquire)) return false;
            if (!S.ticket.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            (*S.fn.load(std::memory_order_acquire))((int) (t & 0xffffffffu));
            S.done.fetch_add(1, std::memory_order_acq_rel);
            return true;
        }
    }
    void run_c(int n, const std::function<void(int)>& fn) {
        int s = 0;
        for (bool got = false; !got;)
            for (s = 0; s < kSlots && !got; ++s) {
                bool f = false;
                got = slots_[s].busy.compare_exchange_strong(f, true, std::memory_order_acq_rel);
                if (got) break;
            }
        Slot& S = slots_[s];
        const uint64_t g = ++S.gen & 0xffffffffu;
        S.ticket.store((g << 32) | kClosed, std::memory_order_seq_cst);
        S.fn.store(&fn, std::memory_order_relaxed);
        S.prio.store(tl_prio(), std::memory_order_relaxed);
        S.total.store(n, std::memory_order_release);
        S.done.store(0, std::memory_order_relaxed);
        S.ticket.store(g << 32, std::memory_order_seq_cst);
        epoch_.fetch_add(1, std::memory_order_seq_cst);
        if (sleeping_.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
        while (claim(S, g)) {
        }
        while (S.done.load(std::memory_order_acquire) < n) cpu_relax();
        S.ticket.store((g << 32) | kClosed, std::memory_order_seq_cst);
        S.busy.store(false, std::memory_order_release);
    }
    // the open slot of the highest priority with a job left (-1: none), its generation in g
    int pick(uint64_t& g) {
        int best = -1, bp = 0;
        for (int s = 0; s < kSlots; ++s) {
            Slot& S = slots_[s];
            const uint64_t t = S.ticket.load(std::memory_order_acquire);
            if ((t & 0xffffffffu) == kClosed || (int) (t & 0xffffffffu) >= S.total.load(std::memory_order_acquire))
                continue;
            const int p = S.prio.load(std::memory_order_relaxed);
            if (best < 0 || p > bp) best = s, bp = p, g = t >> 32;
        }
        return best;
    }
    void loop_c(int id) {
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            int spins = 0;
            for (;;) {
                if (quit_.load(std::memory_order_relaxed)) return;
                const uint64_t e = epoch_.load(std::memory_order_acquire);
                if (id + 1 < active_.load(std::memory_order_acquire)) {
                    uint64_t g = 0;
                    const int s = pick(g);
                    if (s >= 0) {
                        claim(slots_[s], g);
                        break;
                    }
                }
                cpu_relax();
                const bool idle = id + 1 >= active_.load(std::memory_order_acquire);
                if (idle || ((++spins & 255) == 0 &&
                             std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_))) {
                    std::unique_lock<std::mutex> lk(mu_);
                    sleeping_.fetch_add(1, std::memory_order_seq_cst);
                    cv_.wait(lk, [&] { return quit_.load() || epoch_.load(std::memory_order_seq_cst) != e; });
                    sleeping_.fetch_sub(1);
                    break;
                }
            }
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
    const bool concurrent_;
    Slot slots_[kSlots];
    std::atomic<uint64_t> epoch_{0};
};

// [off, off + len) of a shard into dst (O_DIRECT when the shard has a direct fd; see glm_fast_path.cu)
void read_slice(const Glm5Model::Shard& sh, uint64_t off, size_t len, uint8_t* dst);

// A layer's VRAM slot stride: the expert blob ([gate | up | down]) rounded up so that both matrices' block sizes and
// 16 bytes divide it - llama.cpp's MMQ addresses expert e of a pool partition as base + e * stride in WHOLE blocks,
// so the prompt path multiplies the resident experts where they lie.
inline size_t expert_stride(size_t blob, int gu_type, int d_type) {
    // an EXL3 blob: every piece is a multiple of 256 bytes, no ggml blocks to keep whole (the prompt path does not
    // multiply these with MMQ)
    if (gu_type == strata::kernels::exl3::kTypeEXL3 || d_type == strata::kernels::exl3::kTypeEXL3)
        return (blob + 255) / 256 * 256;
    size_t a = 16;
    const size_t b1 = ggml_type_size((ggml_type) gu_type), b2 = ggml_type_size((ggml_type) d_type);
    if (b1 > 0) a = std::lcm(a, b1);
    if (b2 > 0) a = std::lcm(a, b2);
    return (blob + a - 1) / a * a;
}

// Linux with ZFS: what its ARC holds above its floor (c_min). The ARC gives that memory back under pressure, as the
// page cache does, but MemAvailable leaves it out (#89: an ARC of 54 GB after a 95 GB download, MemAvailable 5 GB),
// so the RAM-free measurements add it.  0 without ZFS and elsewhere.
int64_t zfs_arc_reclaimable();

// --dump-routing (Glm5Model::set_routing_dump): whether a trace file is open, and one route appended to it as
// Strata's trace record (int32 layer, int32 k, then k int32 expert ids and k float weights), under a lock - the
// decode's CPU-lane threads (one per half) and the prompt path write to the same file
bool route_dumping();
void route_dump(int layer, int k, const int* ids, const float* w);

// The expert usage file (opt-in, as Strata keeps nothing between starts but --expert-profile-save's profile):
// STRATA_GLM_USAGE=<file>; unset or 0, none - the warm-up and the split search then go by the expert profile
std::string usage_file();

// STRATA_GLM_TRUNK_KEEP / _MAP: the route rank from which layer il leaves out its experts that are not in VRAM (8: none)
// - the token path's routes and the verify windows' alike
int trunk_skip_from(int il);

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
    size_t scratch_bytes = 0, win_bytes = 0;   // the scratch slots; the verify window's buffers (the VRAM report)
    // activations (device)
    float *x = nullptr, *mixer = nullptr, *ffn = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    float* defer_post = nullptr;   // STRATA_GLM_DEFER_CPU: the deferred layer's FFN-side post weights (8 floats)
    float* part = nullptr;
    unsigned int* counter = nullptr;
    void* xq = nullptr;
    float* proj[3] = {nullptr, nullptr, nullptr};
    float* conv[3] = {nullptr, nullptr, nullptr};
    float *fa = nullptr, *ga = nullptr, *beta = nullptr, *g1 = nullptr, *g2 = nullptr;
    void* gated_q = nullptr;
    float* rep = nullptr;   // a window commit's replay (win_commit_part): kKdaLayers x 5 x d_inner (q k v g1 g2)
    float *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *ik_raw = nullptr, *ig_raw = nullptr, *iw = nullptr;
    float *q = nullptr, *iq = nullptr, *score = nullptr;
    int* cells = nullptr;
    int* slot_cells = nullptr;   // KV streaming: the selection in VRAM slot rows (glm_kv_resolve), what the attention reads
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
    void* spec_q = nullptr;      // speculative sampling (STRATA_GLM_SPEC_SAMPLE): the draft's kept list, a device SpecList
    cudaEvent_t ev_mtp = nullptr;
    // the one-GPU MTP decode (src/core/glm_mtp.cu) - the draft head over the draft vocabulary (Glm5Model::mtp_vocab_scan;
    // mtp_dv == 0: the whole head): the head's first mtp_dv rows in place, then mtp_dx rows gathered from the rest
    // (mtp_dx_w) whose tokens are mtp_dx_ids (device: the argmax writes a token id either way)
    int mtp_dv = 0, mtp_dx = 0;
    void* mtp_dx_w = nullptr;
    int* mtp_dx_ids = nullptr;
    // ... the chained drafts (STRATA_GLM_MTP_CHAIN): token_embd on the device as stored (Q6_K / Q8_0), the token of a
    // step after the first read there; null: every draft step takes its embedding from the host.  The host's own rows
    // for a draft step go through mtp_emb_h, its copy out ordered by ev_mtp_emb
    void* mtp_emb = nullptr;
    int mtp_emb_type = -1;
    float* mtp_emb_h = nullptr;   // pinned
    cudaEvent_t ev_mtp_emb = nullptr;
    bool mtp_emb_pending = false;
    // ... and its NextN cache fill (Glm5Model::mtp_cache_rows): up to cap rows at once - the next tokens' embeddings
    // (emb_h pinned), eh_proj's input and output, the attention norm and the cache rows' projections
    struct MtpCacheRows {
        int cap = 0;
        void* arena = nullptr;
        float *emb = nullptr, *hid = nullptr, *x = nullptr, *kv_raw = nullptr, *ik_raw = nullptr, *ig_raw = nullptr;
        void *catq = nullptr, *xq = nullptr;
        float* emb_h = nullptr;
    } mtpc;
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
        // EXL3 (an exl3.txt pack): the routed experts' blob layout, and the indexer's q_b when it is an EXL3 matrix
        bool exl3 = false;
        strata::kernels::exl3::ExpertLayout xl;
        WSlot idx_q_b_x;
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
    // the pool is COMPLETE (glmfast::complete_pool: every layer's main slots hold every expert and the spares; fast_setup
    // sized it so) ...
    bool complete = false;
    // ... and, as of the last boundary, every expert of every MoE layer IS in VRAM (the device tables point at all of
    // them): a route can neither miss nor fetch, so the per-route moe_wait / moe_fetch / moe_cpu_wait launches - no-ops
    // then - are left out (fast_moe).  Only the boundary and the prompt path's lending change the tables
    // (a promotion only adds), so it is set there (check_resident) and holds until the next one.
    // STRATA_GLM_RESIDENT_SKIP=0 launches them anyway (A/B); on by default on AMD only (glmfast::on_by_default_on_amd).
    bool all_resident = false;
    bool all_in_vram = false;   // every expert in VRAM as of the last boundary, whatever STRATA_GLM_RESIDENT_SKIP says
    // STRATA_GLM_SKIP_EMPTY=1: the same per layer for the batched rows (glm_rows.cu; the token path keeps the rule
    // above, its layers' launches are in its graphs' key) - layer_res[il]: every expert of layer il in VRAM as of the
    // last boundary, so its routes leave out the wait, the fetch and the CPU lane's wait.  Off: empty.
    std::vector<uint8_t> layer_res;
    bool layer_resident(int il) const { return (size_t) il < layer_res.size() && layer_res[(size_t) il] != 0; }
    void check_resident(int l0, int lt, int n_expert) {
        static const bool on = glmfast::on_by_default_on_amd("STRATA_GLM_RESIDENT_SKIP");
        if (glmfast::skip_empty_on()) {
            layer_res.assign(L.size(), 0);
            for (int il = l0; il < lt; ++il) {
                if (!L[(size_t) il].moe) continue;
                const int* so = slot_of.data() + (size_t) il * n_expert;
                bool all = true;
                for (int e = 0; e < n_expert && all; ++e) all = so[e] >= 0;
                layer_res[(size_t) il] = all ? 1 : 0;
            }
        }
        bool all = true;
        for (int il = l0; il < lt && all; ++il) {
            if (!L[(size_t) il].moe) continue;
            const int* so = slot_of.data() + (size_t) il * n_expert;
            for (int e = 0; e < n_expert; ++e)
                if (so[e] < 0) {
                    all = false;
                    break;
                }
        }
        all_in_vram = all;
        all_resident = on && all;
    }
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
    // how far the prompt's lending (lend_tail) freed xpool: the first tail slot it kept, so [xpool + the prompt's
    // need, lend_end) is a loan the layout did not use - where the prompt path's MMQ scratch lives (mmq::pool_slab)
    size_t lend_end = 0;
    // ---- the RAM tier: pinned host slots per blob size class, holding experts that are NOT in VRAM
    //      (exclusive tiers).  rtab (on the device) points at HOLDING slots only.
    // kRPin: still the expert's live copy, being promoted (never a victim until the copy landed)
    enum : char { kRFree = 0, kRHold = 1, kRRelease = 2, kRDemote = 3, kRNew = 4, kRLoad = 5, kRPin = 6,
                  kRHelper = 7 };   // helper-owned UMA weights: readable by prefill, never a RAM-tier victim
    struct RamClass {
        size_t stride = 0;
        uint8_t* base = nullptr;   // the first piece (the only one unless per_part > 0)
        int n = 0;
        bool registered = false;   // numa_pinned's (mmap + cudaHostRegister), not cudaHostAlloc's
        // pinned in pieces of per_part slots (Windows: a big block takes far more of the per-process pinning budget
        // than its size - a 2-GPU split's second part got 0 GB, #417); 0: one block from base
        int per_part = 0;
        std::vector<uint8_t*> parts;
        uint8_t* slot(size_t s) const {
            return per_part > 0 ? parts[s / (size_t) per_part] + (s % (size_t) per_part) * stride : base + s * stride;
        }
        size_t part_slots(size_t i) const {   // the slots of piece i
            return per_part > 0 ? std::min((size_t) per_part, (size_t) n - i * (size_t) per_part) : (size_t) n;
        }
        size_t n_parts() const { return per_part > 0 ? parts.size() : (base != nullptr ? 1 : 0); }
        uint8_t* part(size_t i) const { return per_part > 0 ? parts[i] : base; }
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
    // from RAM, 3 dropped by the prompt path's lending, 4 evicted from RAM by the lending's park), and the disk reads by
    // that cause
    std::vector<uint8_t> left;
    uint64_t diag_disk[8] = {}, diag_lend_evict = 0, diag_drop = 0, diag_ram_evict = 0, diag_lend = 0, diag_lend_park = 0, diag_b = 0;
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
    // STRATA_GLM_GROUP_EXPERTS: rows 1 .. rows_group - 1's scratch slots, and every row's 8 slot pointers (device)
    uint8_t* rows_scratch = nullptr;
    unsigned long long* rows_scr = nullptr;
    int rows_group = 0;                        // a group's rows at most (0: grouping off)
    // STRATA_GLM_MULTI_MTP: rows_scr's rows (a group of verify windows); rows rows_group .. rows_scr_rows - 1 have no
    // scratch slots of their own (they name earlier rows'), so they group only while every expert is in VRAM
    int rows_scr_rows = 0;
    std::vector<uint8_t*> stage;   // pinned: disk reads when no RAM slot is free
    size_t stage_bytes = 0;        // ... each (an expert's blob); the start's uploads go through stage[0] too
    std::mutex mu;   // the service thread vs the between-token boundary
    std::unique_ptr<glmfast::Workers> workers;   // parallel disk reads
    // ---- the CPU LANE (STRATA_GLM_CPU_LANE=<threads>, 0 off): a route's RAM-tier experts split between the device's
    //      PCIe pulls and this pool, which computes its share from the pinned blobs meanwhile; the device adds the
    //      weighted sum (cpu_ans_h) before its down combine.  cpu_plan: how many of f RAM-tier experts go to the host.
    //      A split of 3+ GPUs shares one pool across its parts (fast_cpu_lane_setup).
    std::shared_ptr<glmfast::Workers> cpu_pool;
    GlmExpertHelper helper;   // opt-in, independent expert ownership on the secondary GPU
    uint8_t* helper_owned_d = nullptr;   // primary router excludes these experts from PCIe admission/prefetch
    uint64_t helper_swaps = 0;
    std::atomic<bool> helper_failed{false};
    std::string helper_error;
    std::atomic<uint64_t> helper_experts{0}, helper_routes{0}, helper_us{0};
    // one GPU with a DFlash drafter, on a host of two NUMA nodes or more (STRATA_GLM_WIN_LANES, on): a batched verify
    // window's tokens' CPU-lane experts go to two lane threads, token t to lane t % 2 - lane 0 drives cpu_pool (then
    // pinned to the GPU's node), lane 1 a pool of its own on another node - so the window's tokens run on both
    // sockets at once (a decode's one token keeps the service thread and cpu_pool).  glm_fast_path.cu
    struct WinLane;
    WinLane* win_lane[2] = {nullptr, nullptr};
    // ---- the verify WINDOW (src/core/glm_window.cu): T tokens at consecutive positions through this part's layers in
    //      one pass; every per-token activation has a row per token (row t at t * its stride), the KDA layers' inputs
    //      are kept per layer for window_commit's replay
    struct Win {
        int tmax = 0;
        void* arena = nullptr;
        float* R[2] = {nullptr, nullptr};   // residual rows (hc x n_embd each), double-buffered across the layers
        int out = 0;                        // which of R holds this part's output rows after the window
        float *x = nullptr, *mixer = nullptr, *ffn = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
        void* xq = nullptr;
        void* gated_q = nullptr;
        float *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *ik_raw = nullptr, *ig_raw = nullptr, *iw = nullptr;
        float *q = nullptr, *iq = nullptr;
        float *q_abs = nullptr, *ctx = nullptr;   // the absorbed MLA's per-token rows (n_head x kv_lora each)
        int* cells = nullptr;                     // each token's selected cells (n_sel_max a row): one batched attention
        void *qr_q = nullptr, *attn_q = nullptr;
        float *rlog = nullptr, *sh_g = nullptr, *sh_u = nullptr, *dg = nullptr, *du = nullptr;
        void* dhq = nullptr;
        float* head_x = nullptr;
        void* head_xq = nullptr;
        float* logits = nullptr;
        int* tok = nullptr;
        float* emb = nullptr;
        size_t xq_s = 0, gq_s = 0, qrq_s = 0, aq_s = 0, dhq_s = 0;   // q8_1 bytes per row
        // per KDA layer: its tmax tokens' projections q | k | v (d_inner each), then f_a, g_a (head_dim), beta (n_head)
        float* kin = nullptr;
        size_t kin_layer = 0;               // floats per layer
        std::vector<int> kin_of;            // layer -> its slot in kin (-1: not a KDA layer here)
        int* tok_h = nullptr;               // pinned
        float* emb_h = nullptr;             // pinned, tmax x n_embd
        float* hop_h = nullptr;             // pinned, tmax x hc x n_embd: the rows to the next part
        cudaEvent_t ev_hop = nullptr;
        cudaEvent_t ev_tok[32] = {};        // window_pipe's hop of each token slot (its threads record and wait out of
        cudaEvent_t ev_pick[32] = {};       // turn) ... and (the last part) each token's pick copied out (kPipeSlots)
        int bank = 0;                       // ... the slots' bank this window (they alternate: kPipeMax a bank)
        bool pick_sampled = false;          // ... the picks drawn with pick_sp (row t at its counter + t), not argmax
        // ... the tokens past the first wrong draft skip their heavy kernels: the head part's pinned word per bank
        // (`from_h`, the host writes the first doomed token there), this part's device flag (`skip_d`, each layer's
        // skip_gate copies the verdict for its token in) and the word this window reads (`from`)
        volatile int* from_h = nullptr;
        int* skip_d = nullptr;
        const volatile int* from = nullptr;
        strata::kernels::SamplerParams pick_sp;
        int32_t pick_tok[32] = {};          // ... the tokens pushed, by slot (a pick's penalty rows count the drafts before it)
        int64_t p0 = 0;
        int T = 0;
        // STRATA_GLM_WIN_PROF=1: an event after each phase of the window's layers, the gaps summed per phase name
        std::vector<cudaEvent_t> ev;
        std::vector<const char*> ev_name;
        size_t ev_n = 0;
        std::map<std::string, double> ev_ms;
        int64_t n_prof = 0;
        double h_setup = 0, h_enqueue = 0, h_wait = 0, h_after = 0;   // window()'s host phases (the first part's)
        int64_t h_n = 0;
        // STRATA_GLM_WIN_UNION=1: each MoE layer's routed experts per window token (device, layers x kWinMax x 8), and
        // the distinct experts a window's layer routes against its picks (how much computing each expert once would save)
        int* uid = nullptr;
        std::vector<int> uid_h;
        int64_t u_layers = 0, u_distinct = 0, u_picks = 0;
        // ---- the batched MoE (fast_moe_window, gf::moe_win_experts): each token's own route state - plan, fetch
        //      sources, the host's disk answer, the CPU lane's answer slot (token 0 the lone one, cpu_ans_h) and seq word
        unsigned long long *mplan_ptr = nullptr, *mfetch = nullptr;
        float *mplan_w = nullptr, *mcpu_part = nullptr;
        int *mplan_id = nullptr, *mcpu_flag = nullptr;
        void* sh_hq = nullptr;              // [tmax] the shared expert's h (q8_1, sh_hq_s bytes a token)
        size_t sh_hq_s = 0;
        float* sh_out = nullptr;            // [tmax][n_embd] its down rows
        void* whq = nullptr;                // [tmax * 8][n_ff / 32] q8_1: each entry's h
        float* wpart = nullptr;             // [tmax * 8][n_embd]: each entry's down rows
        float* wh = nullptr;                // EXL3: [tmax * 8][n_ff] each entry's h (F32)
        unsigned long long* grp_ptr = nullptr;
        int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
        unsigned int* scr = nullptr;        // [0] the layer's scratch claims, [1] sticky: a slot taken twice
        unsigned int* scr_h = nullptr;      // pinned: [1] read after the window
        unsigned int* cpu_seq_w = nullptr;  // [tmax] seq words 64 bytes apart
        gf::CpuAnswer* ans_h[16] = {};      // host-mapped answer slots for tokens 1.. (token 0: cpu_ans_h)
        void* ans_d[16] = {};
        bool per_token = false;             // the batched MoE off for this pass (the redo after a scratch clash)
        bool piped = false;                 // this window is a pipelined one (win_pipe_begin), not window()'s
        int pipe_t = -1;                    // window_pipe: the token whose layers are being enqueued (its KDA inputs
                                            // to kin row pipe_t); -1 otherwise
        void mark(const char* nm, cudaStream_t st) {
            if (ev_n >= ev.size()) {
                cudaEvent_t e = nullptr;
                cudaEventCreate(&e);
                ev.push_back(e);
                ev_name.push_back(nm);
            }
            cudaEventRecord(ev[ev_n], st);
            ev_name[ev_n] = nm;
            ++ev_n;
        }
    } win;
    // ---- the DFlash drafter's target features (src/core/glm_dflash.cu): at each captured layer here, the input's
    //      stream mean of the rows a pass leaves for the drafter, n_embd floats a row (cap_rows rows) - copied out to
    //      the drafter's pinned staging (by position) on this part's stream
    float* cap = nullptr;
    int cap_rows = 0;
    std::atomic<int> bnd_tokens{0};          // pipelined window tokens this part ran since its last tier boundary
    int64_t pf_end = -1;                     // the prompt's end while one runs (the drafter takes its last positions)
    int cpu_node = -1;                         // the NUMA node the pool is pinned to (-1: unpinned) ...
    std::vector<int> cpu_pin;                  // ... and its CPUs (the service thread runs there too)
    std::string cal_file, cal_key;             // STRATA_GLM_CPU_CAL
    double cal_ref = 0.0;                      // decode's CPU ms an expert after calibrating (0: not measured yet)
    uint64_t drift_e0 = 0, drift_us0 = 0;
    int drift_n = 0, drift_dir = 0;            // windows in a row >15% off, and which way
    // lane_retime: the split re-planned from the lane's own cost in decode (only one the start-up timing chose)
    bool cpu_retime = false;
    uint64_t rt_e0 = 0, rt_us0 = 0, rt_r0 = 0; // the counters at the window's start
    int rt_windows = 0, rt_changes = 0;        // windows looked at, splits changed
    double rt_c_ms = 0.0;                      // the CPU's cost an expert the split in use was planned with
    int rt_dir = 0;                            // the last window's way off it (+1 slower, -1 faster, 0 within 15%)
    double rt_prev_ms = 0.0;                   // ... and its cost an expert
    bool lane_pending = false;                 // the lane and the service thread wait for fast_setup_finish
    int64_t ram_check_head = 0;                // a parallel load: the headroom ram_left_check holds the free RAM to
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
    struct CpuWin {                                          // a merged window layer (win_cpu_merged):
        std::vector<uint8_t> act, hq;                        // each token's activation quant; each entry's h quant
        std::vector<float> ff, dn;                           // each entry's n_ff / n_embd rows
        std::vector<float> x3;                               // EXL3: each entry's rows (X3Rows)
    } cpu_win;
    std::vector<float> cpu_x3;                               // EXL3: per expert gate / up inputs, h, down input, out
    std::atomic<uint64_t> cpu_experts{0}, cpu_routes{0}, cpu_us{0};
    // STRATA_GLM_SPEC_CANCEL=1 (the pipelined two-half decode, a head half of one GPU): the head's speculative pass reads
    // the pinned, host-mapped word spec_live_h at every layer (skip_gate) into spec_skip_d - 1: its draft still stands,
    // 0: the draft was wrong, its heavy kernels do nothing from that layer on and the redo can start at once
    int* spec_live_h = nullptr;
    int* spec_skip_d = nullptr;
    bool spec_cancellable = false;   // spec_head: the pass it enqueues next is a speculative one (may be cancelled)
    bool spec_gate = false;          // fast_layers: a skip_gate before every layer (set while that pass is enqueued)
    std::thread svc;
    std::atomic<bool> quit{false};
    uint64_t clock = 0;
    uint64_t expected = 0;               // requests this half's device has issued (one per MoE layer per token)
    std::atomic<uint64_t> processed{0};  // ... and the service thread has consumed
    // stats (written by the service thread, read by anyone)
    std::atomic<uint64_t> hits{0}, misses{0}, miss_layers{0}, ram_hits{0}, disk_reads{0}, promotions{0};
    std::atomic<uint64_t> disk_skips{0};   // STRATA_GLM_DISK_SKIP_W / _FROM: disk-only experts left out
    int disk_skip_from = 8;                // ... at route rank >= this (8: none)
    float disk_skip_w = 0.0f;              // ... or of a smaller weight share than this (0: none)
    // STRATA_GLM_MISS_SKIP_W / _FROM: a routed expert not in VRAM whose weight share is below miss_w is left out, on layers
    // from miss_from on (0: none); a request's strata_tune miss_skip_w / miss_skip_from override them (< 0: these)
    float miss_w = 0.0f;
    int miss_from = 0;
    float miss_w_req = -1.0f;
    int miss_from_req = -1;
    std::atomic<uint64_t> scratch_uses{0}, demotions{0}, drops{0};
    std::atomic<uint64_t> miss_us{0}, disk_us{0};
    uint64_t tokens = 0;
    double ms = 0;
    bool timing = false;
    bool ram_resident = false;   // STRATA_GLM_RAM_RESIDENT=1 / --glm-ram-resident: the RAM tier holds EVERY expert
                                 // VRAM does not, nothing ever drops to disk; the start fails if the tier is small
    int ram_slack = 0;           // extra RAM-tier slots per MoE layer (STRATA_GLM_RAM_SLACK; default 16 resident)
    int read_chunks = 0;         // set_read_chunks: a disk read's pieces (0: STRATA_GLM_READ_CHUNKS, else 8)
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
        if (helper_failed.load(std::memory_order_acquire)) { err = helper_error; return false; }
        const int e = route_error_h ? *route_error_h : 0;
        if (e == 0) return true;
        const char* kind = (e % 4 == 1) ? "decode" : (e % 4 == 2) ? "predicted" : "lookahead";
        err = "glm router: layer " + std::to_string(e / 4) + " " + kind + " invalid expert ID or non-finite score";
        return false;
    }
};

// KV streaming (Glm5Model::kv_stream_setup): per DSA layer of this part (the NextN block's too), its latent cache in
// pinned host memory - the authoritative copy, position order - and the residency map of its VRAM slots, which are
// state_ + dsa_lat_[il] (slots * page rows).
struct Glm5Model::KvStream {
    int64_t slots = 0;     // VRAM pages a layer
    int page = 4;          // positions a page: one indexer pool
    int row_bytes = 0;     // one position's latent row: FP16 kv_lora, or an INT8 record (lat8_rec_bytes)
    int64_t n_blocks = 0;  // pages of the context
    std::vector<uint8_t*> host;                          // per layer: the host copy (device-mapped), null: not a DSA layer
    std::vector<strata::kernels::KvStreamMap> map;       // per layer
    uint8_t* host_arena = nullptr;                       // pinned: every layer's host copy
    void* map_arena = nullptr;                           // device: every layer's map
    uint64_t host_bytes = 0;
};

// KV streaming's two hooks for a DSA attention at decode (the token path, the verify window, the rows of several
// sequences), for layer il of a sequence whose stream is K (null: not streamed) and whose latent run is `lat` (a
// streamed layer's VRAM slots): where dsa_prep writes a position's latent row - the identity cache, or the host copy and
// its block's VRAM slot - and which cells the attention reads - the selection resolved into slot rows, or itself
inline bool kv_streamed(const Glm5Model::KvStream* K, int il) { return K != nullptr && K->host[(size_t) il] != nullptr; }
inline void kv_prep_targets(const Glm5Model::KvStream* K, int il, uint16_t* lat, gf::DsaPrepArgs& d) {
    if (!kv_streamed(K, il)) {
        d.lat = lat;
        return;
    }
    d.lat = nullptr;
    d.lat_host = (uint16_t*) K->host[(size_t) il];
    d.lat_slots = lat;
    d.lat_table = K->map[(size_t) il].page_table;
    d.lat_page = K->page;
}
inline const int* kv_attend_cells(const Glm5Model::KvStream* K, int il, uint16_t* lat, const int* cells, int n_sel,
                                  int* slot_cells, cudaStream_t s) {
    if (!kv_streamed(K, il)) return cells;
    gf::glm_kv_resolve(K->map[(size_t) il], K->host[(size_t) il], (uint8_t*) lat, K->row_bytes, K->page, cells, n_sel,
                       slot_cells, s);
    return slot_cells;
}


}  // namespace strata::core
