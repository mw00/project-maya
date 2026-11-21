// src/ik2/hip/ik2_hip_model.cu - maya-ik2's model (ik2_hip_model.hpp): loading, placement over the GPUs, and
// the forward.
//
// A layer, for the token at position p (llama.cpp's src/models/inkling2.cpp; docs/INKLING2_AMD.md):
//   m = mix(blocks, cur, attn_res_query) (layer 0: cur); the sliding-window layers: xq = xk = xv = rms(m) attn_norm;
//   the global ones: xq, xk, xv = rms(m[0..2]) with their own gains
//   Q = Wq xq, gate = Wgate xq, R = Wrel xq, sink = Wsink xq;  K = conv(Wk xk), V = conv(Wv xv) (the global layers'
//   convolutions with input-dependent taps Wk_conv_proj xk / Wv_conv_proj xv)
//   Q clipped per head (after its per-channel scale), K clipped, V rms per head, R clipped (after its scale); the
//   global layers' Q and R times tau(p);  K, V -> the cache
//   o = softmax(Q.K / sqrt(128) + R.rel_bias[p - j] - sink, a zero sink logit) . V over the window (or every position)
//   o = conv(Wo (o sigmoid(gate)));  L % 6 == 0: blocks += cur, cur = o;  else cur += o
//   xf = rms(mix(blocks, cur, ffn_res_query)) ffn_norm;  the router (sigmoid, + bias, top 8, renormalised, * 8), the
//   picks' experts: down(situ(gate xf, up xf));  cur += conv(their weighted sum)
// and logits = softcap(Wout rms(mix(blocks, cur, output_res_query)) output_norm).
#include "ik2_hip_model.hpp"

#include "ik2_hip_dspark.hpp"

#include "ik2_hip_kernels.cuh"
#include "ik2_hip_lane.hpp"
#include "ik2_hip_pack.hpp"
#include "hy4_kernels.cuh"
#include "../ds4/ds4_iq.cuh"

#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <thread>

namespace ik2hip {

namespace iq = strata::ds4::gguf::iq;
namespace hk = hy4::k;

extern "C" void dequantize_row_q5_K(const void* x, float* y, int64_t k);   // (ggml-quants: a token's embedding row)
extern "C" void dequantize_row_q6_K(const void* x, float* y, int64_t k);

namespace {

constexpr int kF32 = 0, kBF16 = 30;

int type_id(const std::string& s) {
    static const std::pair<const char*, int> names[] = {
        {"F32", 0},     {"F16", 1},     {"Q8_0", 8},     {"Q2_K", 10},   {"Q3_K", 11},    {"Q4_K", 12},  {"Q5_K", 13},
        {"Q6_K", 14},   {"IQ2_XXS", 16}, {"IQ2_XS", 17}, {"IQ3_XXS", 18}, {"IQ1_S", 19},  {"IQ4_NL", 20}, {"IQ3_S", 21},
        {"IQ2_S", 22},  {"IQ4_XS", 23}, {"IQ1_M", 29},   {"BF16", 30}};
    for (const auto& [n, i] : names)
        if (s == n) return i;
    return -1;
}
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
size_t al256(size_t b) { return (b + 255) / 256 * 256; }

/// A matrix on the device: n rows of k inputs (ld bytes apart), in ggml type `type` (F32: plain rows)
struct Mat {
    const uint8_t* w = nullptr;
    int type = -1, k = 0, n = 0;
    size_t ld = 0;
};

struct Layer {
    int il = 0, gpu = -1;
    bool swa = true;
    float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr;
    float *attn_res = nullptr, *ffn_res = nullptr, *ffn_norm = nullptr;
    float *q_clip = nullptr, *rel_clip = nullptr, *rel_bias = nullptr;
    float *k_conv = nullptr, *v_conv = nullptr, *a_conv = nullptr, *f_conv = nullptr, *router_b = nullptr;
    Mat q, gate, wk, wv, rel, sink, wo, k_proj, v_proj, router;
    // the state: the convolutions' histories [3][C] and the K / V caches [slots][Hkv][D]
    float *hist_k = nullptr, *hist_v = nullptr, *hist_a = nullptr, *hist_f = nullptr;
    uint16_t *K = nullptr, *V = nullptr;   // (F16 rows)
    int ring = 0;                 // the caches' rows when they are a ring (0: a row a position)
    int* slot_d = nullptr;        // (slot_of on the device: the prompt path's sets, the host-free step's plan)
    float *win_in = nullptr, *win_h0 = nullptr;   // (a step's conv inputs [K | V | attention | FFN][kWin]; the other
                                                  //  buffers of the histories: a step writes hist_* -> alt_*, swapped)
    float *alt_k = nullptr, *alt_v = nullptr, *alt_a = nullptr, *alt_f = nullptr;
    cudaEvent_t pe[10] = {};      // (IK2_HIP_PHASES: the host-free step's phase marks on its GPU's stream)
    float *m_pub = nullptr, *m_cpu = nullptr;   // (mapped host memory: its routing out to the lane, the lane's sum back)
    float* snap = nullptr;        // the histories' saved copies [kSnaps][k | v | attention | FFN]
    // the routed experts: a blob each (the fused gate | up rows, then down), in VRAM or in RAM
    const Tensor *t_gu = nullptr, *t_dn = nullptr;
    int tg = 0, td = 0;
    size_t gub = 0, db = 0, blob = 0, tr[3] = {0, 0, 0};
    uint8_t* experts = nullptr;   // the VRAM ones, a blob a slot
    std::vector<int> slot_of;     // each expert's VRAM slot (-1: in RAM), and its RAM slot (-1: in VRAM)
    std::vector<int> ram_of;
    int n_res = 0;
    int n_copy = 0;               // RAM slots below it: copied into `ram`; from it on: in place in the model's files
    uint8_t* ram = nullptr;
    bool ram_reg = false;
    size_t ram_bytes = 0;
    CpuExperts cex;
    std::vector<float> h_bias;    // (the router's selection bias on the host: a tap's scores)
    std::vector<float> use;       // each expert's decayed routings (the cache's: who is in VRAM)
    int spare = -1;               // the VRAM slot no expert is in (the cache's next arrival lands there)
    int picks[16 * 16] = {};      // (the last host-free step's picks, [row][16]: counted once its rows are kept)
    size_t dense = 0;             // (the plan) its VRAM without the experts
};

struct Gpu {
    int dev = 0;
    cudaStream_t s = nullptr;
    std::vector<void*> allocs;
    size_t used = 0, free0 = 0, total = 0;
    size_t gran = (size_t) 2 << 20;   // (the runtime's rounding of an allocation over 1 MiB: measured at the load)
    size_t free_plan = 0;             // (free when the experts were placed)
    size_t expert_alloc = 0;          // (the experts' one allocation, every layer's spare slot in it)
    uint8_t* h_dummy = nullptr;       // (the warm-up's one zero expert, pinned: every layer's experts before place)
    int l0 = 0, l1 = 0;           // its layers [l0, l1)
    size_t expert_bytes = 0;
    int n_res = 0;
    // a token's activations
    float *cur = nullptr, *blocks = nullptr, *mixo = nullptr, *xn = nullptr, *stat = nullptr;
    float *q = nullptr, *gate = nullptr, *rel = nullptr, *sink = nullptr, *ky = nullptr, *vy = nullptr;
    float *dk = nullptr, *dv = nullptr, *S = nullptr, *ao = nullptr, *o = nullptr, *xf = nullptr;
    float *rl = nullptr, *rw = nullptr, *wres = nullptr, *gu = nullptr, *eh = nullptr, *part = nullptr;
    float *y = nullptr, *cpo = nullptr, *xd = nullptr, *ehd = nullptr, *logits = nullptr, *top = nullptr;
    int* ids = nullptr;
    int8_t *xq = nullptr, *ehq = nullptr;
    unsigned long long* plan = nullptr;
    // the prompt path: a chunk's activations ([T][...] each), the GEMMs' F16 operands, the experts' sorted rows, the
    // attention's scores, the streamed experts' two staging buffers
    cublasHandle_t blas = nullptr;
    cudaStream_t s2 = nullptr;
    cudaEvent_t ev_copy[2] = {nullptr, nullptr}, ev_done[2] = {nullptr, nullptr};
    float *c_cur = nullptr, *c_blocks = nullptr, *c_mixo = nullptr, *c_xn = nullptr, *c_stat = nullptr;
    float *c_q = nullptr, *c_gate = nullptr, *c_ao = nullptr, *c_qp = nullptr, *c_op = nullptr, *c_rel = nullptr;
    float *c_sink = nullptr, *c_ky = nullptr, *c_vy = nullptr, *c_kc = nullptr, *c_vc = nullptr, *c_dk = nullptr;
    float *c_dv = nullptr, *c_o = nullptr, *c_oc = nullptr, *c_xf = nullptr, *c_rl = nullptr, *c_rw = nullptr;
    float *c_y = nullptr, *c_yc = nullptr, *c_sy = nullptr, *c_S = nullptr, *c_tbl = nullptr, *c_ent = nullptr;
    int *c_ids = nullptr, *c_bounds = nullptr, *c_stok = nullptr, *c_srow = nullptr, *pmap[2] = {nullptr, nullptr};
    uint16_t *xh16 = nullptr, *wband = nullptr, *c_sh16 = nullptr, *c_P16 = nullptr;
    uint8_t* xstage[2] = {nullptr, nullptr};
    size_t wband_n = 0, S_n = 0, blob_max = 0, S_floats = 0;
    int ts_max = 0;
    // (pinned host memory of its own: the stages run on a thread a GPU)
    float *hc_rw = nullptr, *hc_rl = nullptr, *hc_in = nullptr;
    int *hc_ids = nullptr, *hc_pmap = nullptr;
    int64_t n_streamed = 0, n_experts = 0;
    // the expert cache's move in hand (one a GPU, its own thread and stream): expert `hot` from its RAM slot rs into
    // the layer's spare VRAM slot, expert `cold`'s blob from its VRAM slot out to RAM slot rs.  The decode thread
    // changes the maps between two steps: 1 the copies run, 2 they are done (hot's map flips), 3 cold's bytes go to
    // the RAM slot hot left, 4 done (cold's map flips, its VRAM slot the new spare), 0 idle
    std::thread sw_thread;
    std::mutex sw_mu;
    std::condition_variable sw_cv;
    int sw_state = 0;
    struct Move {
        int il, hot, cold, rs;
    };
    std::vector<Move> sw_jobs;   // (a batch: a layer at most once - it has one spare slot)
    bool sw_quit = false;
    cudaStream_t s3 = nullptr;
    uint8_t* h_swap = nullptr;
    // (--mmap-experts: a prompt's experts in place in the files are gathered here - pinned, kStage blobs - and copied
    //  to the staging slots from it; ev_fst: its last copy out, waited on before the next gather)
    uint8_t* h_fst = nullptr;
    cudaEvent_t ev_fst = nullptr;
    bool fst_busy = false;
    // (a verify window's rows: kWin of each)
    float *w_cb = nullptr, *w_mixo = nullptr, *w_xn = nullptr, *w_stat = nullptr, *w_q = nullptr, *w_gate = nullptr;
    float *w_ao = nullptr, *w_rel = nullptr, *w_sink = nullptr, *w_kc = nullptr, *w_vc = nullptr, *w_dk = nullptr;
    float *w_dv = nullptr, *w_oc = nullptr, *w_xf = nullptr, *w_rl = nullptr, *w_rw = nullptr, *w_yc = nullptr;
    float *w_xd = nullptr, *w_x2d = nullptr, *w_eh = nullptr, *w_ehd = nullptr, *w_part = nullptr, *w_pub = nullptr;
    float *w_top = nullptr, *w_logits = nullptr;
    // (the drafter's taps, the last GPU: a step's rows [kWin][taps][E] in two buffers by turns - the drafter reads the
    //  last step's while the next writes the other - this GPU's own tap rows, the other GPUs' as copied from the host;
    //  a prompt chunk's the same, [Tc] rows)
    float *d_taps[2] = {nullptr, nullptr}, *d_tapl = nullptr, *d_tapr = nullptr;
    // (recorded on the drafter's stream after its reads of d_taps[b]: the step that writes d_taps[b] again waits on it)
    cudaEvent_t ev_tap[2] = {nullptr, nullptr};
    float *c_taps = nullptr, *c_tapl = nullptr, *c_tapr = nullptr;
    int8_t *w_xq = nullptr, *w_x2q = nullptr, *w_ehq = nullptr;
    int *w_ids = nullptr, *w_gstart = nullptr, *w_gent = nullptr, *w_ng = nullptr;
    unsigned long long* w_gptr = nullptr;
    float *pub = nullptr, *m_hand = nullptr;   // (the routing packed; mapped host memory: the hand-off into this GPU)
    unsigned* cnt = nullptr;                   // (a zero word: publish's tally)
    uint8_t* stage = nullptr;     // (the load's: pinned host memory every copy to the device goes through)
    size_t stage_bytes = 0;

    // src -> dst on the device through the pinned stage.  (Not straight from the file's mapping or a vector: under
    // the load's memory pressure - the page cache turning over, huge pages forming - a copy from pageable memory
    // was seen to land a stale page in VRAM, one expert in a few thousand.)
    bool h2d(void* dst, const void* src, size_t bytes, std::string& err) {
        for (size_t o = 0; o < bytes; o += stage_bytes) {
            const size_t n = std::min(stage_bytes, bytes - o);
            std::memcpy(stage, (const uint8_t*) src + o, n);
            const cudaError_t e = cudaMemcpy((uint8_t*) dst + o, stage, n, cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                err = "GPU " + std::to_string(dev) + ": a copy to the device: " + cudaGetErrorString(e);
                return false;
            }
        }
        return true;
    }

    // IK2_HIP_VRAM_LOG=1 (tests): this GPU's VRAM at a stage - free, this engine's own allocations (`extra`: the
    // drafter's), and what left the GPU besides them since the start (free0)
    void note(const char* what, size_t extra = 0) const {
        static const bool on = std::getenv("IK2_HIP_VRAM_LOG") != nullptr;
        if (!on) return;
        size_t fr = 0, tot = 0;
        cudaSetDevice(dev);
        cudaMemGetInfo(&fr, &tot);
        std::fprintf(stderr, "VRAM GPU %d %-34s free %9.1f MiB, ours %9.1f MiB (%zu allocations), other %8.1f MiB\n", dev,
                     what, fr / 1048576.0, (used + extra) / 1048576.0, allocs.size(),
                     ((double) free0 - (double) fr - (double) (used + extra)) / 1048576.0);
    }

    template <typename T>
    bool alloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        const size_t b = std::max<size_t>(n * sizeof(T), 256);
        const cudaError_t e = cudaMalloc(&v, b);
        if (e != cudaSuccess) {
            err = "GPU " + std::to_string(dev) + ": cudaMalloc " + std::to_string(b >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
        allocs.push_back(v);
        used += b;
        p = (T*) v;
        return true;
    }
};

}  // namespace

#define CK(x)                                                                              \
    do {                                                                                   \
        const cudaError_t e_ = (x);                                                        \
        if (e_ != cudaSuccess) {                                                           \
            err = std::string(#x) + ": " + cudaGetErrorString(e_);                         \
            return false;                                                                  \
        }                                                                                  \
    } while (0)

struct Model::Impl {
    Options o;
    Pack pk;
    Geo g;
    Stats st;
    int NL = 0;                    // the layers run
    std::vector<Layer> L;
    std::deque<Gpu> G;   // (a deque: a Gpu holds its mover's mutex and does not move)
    std::unique_ptr<hy4::Lane> lane;
    int64_t n_end = 0;
    int top_id = -1;
    Tap tapf;
    PickHook pick_hook;
    int64_t tap_p = 0;               // (the taps' position and rows: a token's, or a chunk's first and its tokens)
    int tap_T = 1;
    std::vector<float> tap_vec;
    std::vector<float> h_rl;         // (a tap's / a hook's: the router's logits)
    std::vector<uint8_t> tok_embd;   // the embedding table's bytes, in RAM
    int tok_type = 0;
    size_t tok_row = 0;
    float* tok_norm = nullptr;       // (GPU 0)
    Mat head;                        // (the last GPU)
    float *out_norm = nullptr, *out_res = nullptr;
    // pinned host buffers: a token's embedding, the hand-off between two GPUs, the routes, the lane's input and sum
    float *h_x = nullptr, *h_state = nullptr, *h_w = nullptr, *h_wres = nullptr, *h_xf = nullptr, *h_cpu = nullptr;
    float* h_top = nullptr;
    int* h_ids = nullptr;
    unsigned long long* h_plan = nullptr;
    std::vector<CpuEntry> lane_entries;
    std::vector<uint32_t> prof_in, prof_cnt;
    float* h_big = nullptr;          // (pinned: a tap's tensor, the logits on their way out)
    size_t max_keys = 0;
    int Tc = 0;                      // the prompt path's chunk (0: none)
    // ---- the values the measurement phase swept (docs/INKLING2_AMD.md, "Values and their sources"), each with an
    // IK2_HIP_<NAME> override as maya-ds4's DS4_CACHE_*: read at the load; Model::retune reads the run-time ones again
    // (--sweep).  The load's: IK2_HIP_STAGE, IK2_HIP_TS_MAX, IK2_HIP_S_FLOOR_MI, IK2_HIP_WARM_TOKENS
    static double env_d(const char* k, double d, double lo, double hi) {
        const char* v = std::getenv(k);
        if (v == nullptr || *v == 0) return d;
        char* end = nullptr;
        const double x = std::strtod(v, &end);
        if (end == v || !(x >= lo && x <= hi)) {
            std::fprintf(stderr, "maya-ik2: %s=%s is not a number from %g to %g: keeping %g\n", k, v, lo, hi, d);
            return d;
        }
        return x;
    }
    struct Tune {
        // the expert cache's rule, maya-ds4's (DS4_CACHE_DECAY / _MARGIN / _BONUS): the experts' heat x 0.9 every 256
        // tokens (here a token's share of it, 0.9^(1/256), as each token decays), and a RAM expert moves in when its heat
        // passes the coldest VRAM one's x 1.25 + 1.0 (IK2_HIP_CACHE_DECAY / _MARGIN / _BONUS; measured against this
        // engine's earlier x0.995 a token, 1.5 and 0.5: a six-turn chat 51.3 / 51.0 tok/s against 48.7 / 51.2 - docs)
        double cache_decay = 0.9;
        float tok_decay = 0.9995885f;    // (retune sets it from cache_decay)
        float cache_margin = 1.25f, cache_bonus = 1.0f;
        float prompt_rows = 64.0f;     // a prompt chunk of T rows: each routing counts min(1, this / T) (_CACHE_PROMPT_ROWS)
        int win_rows = 4;              // a window's moves: its rows count as this many tokens' at most (IK2_HIP_SWAP_WIN_ROWS)
        int lane_spin_us = 20000;      // the lane's dispatcher spins this long between two tokens, then sleeps (_LANE_SPIN_US:
                                       // maya-ds4's DS4_LANE_SPIN_US; 200 / 1800 / 20000 us decoded alike - docs)
        int snap_every = 64;           // the histories saved every this many decoded tokens (IK2_HIP_SNAP_EVERY)
        int burst_rounds = 48, burst_moves = 64, burst_min = 64;   // the swap burst (IK2_HIP_BURST_ROUNDS / _MOVES / _MIN)
        bool tap_event = true;         // (tests, IK2_HIP_TAP_EVENT=0: the taps' hand-off without its event - timing alone)
        // (tests, --sweep: at most this much of the load's allocation - IK2_HIP_STAGE_USE experts a staging pass,
        //  IK2_HIP_TS_USE tokens an attention pass, IK2_HIP_S_USE_MI Mi score floats; 0: all of it)
        int stage_use = 0, ts_use = 0;
        size_t s_use = 0;
    } tn;
    void retune() {
        tn.cache_decay = env_d("IK2_HIP_CACHE_DECAY", 0.9, 0.0, 1.0);
        tn.tok_decay = (float) std::pow(tn.cache_decay, 1.0 / 256.0);
        tn.cache_margin = (float) env_d("IK2_HIP_CACHE_MARGIN", 1.25, 0.0, 100.0);
        tn.cache_bonus = (float) env_d("IK2_HIP_CACHE_BONUS", 1.0, 0.0, 1e6);
        tn.prompt_rows = (float) env_d("IK2_HIP_CACHE_PROMPT_ROWS", 64, 1, 1 << 20);
        tn.win_rows = (int) env_d("IK2_HIP_SWAP_WIN_ROWS", 4, 1, 16);
        tn.lane_spin_us = (int) env_d("IK2_HIP_LANE_SPIN_US", 20000, 0, 1e6);
        tn.snap_every = (int) env_d("IK2_HIP_SNAP_EVERY", 64, 1, 1 << 20);
        tn.burst_rounds = (int) env_d("IK2_HIP_BURST_ROUNDS", 48, 1, 1 << 20);
        tn.burst_moves = (int) env_d("IK2_HIP_BURST_MOVES", 64, 1, 16);
        tn.burst_min = (int) env_d("IK2_HIP_BURST_MIN", 64, 1, 1 << 30);
        tn.tap_event = env_d("IK2_HIP_TAP_EVENT", 1, 0, 1) != 0;
        tn.stage_use = (int) env_d("IK2_HIP_STAGE_USE", 0, 0, 384);
        tn.ts_use = (int) env_d("IK2_HIP_TS_USE", 0, 0, 1 << 16);
        tn.s_use = (size_t) env_d("IK2_HIP_S_USE_MI", 0, 0, 1 << 16) << 20;
    }
    // streamed experts a pass, in two buffers: one is copied while the other's experts run - fewer passes against
    // 2 x kStage blobs of VRAM (IK2_HIP_STAGE).  Measured, 2048-token chunks, tok/s at 24 / 48 / 64 / 96: a 20K prompt
    // 1579 / 1713 / 1812 / 1831, 80K 1358 / 1436 / 1499 / 1509 - 64's 0.8 GiB a GPU more, with E9's and E42's, cost the
    // shorter prompts and the RAM (Options::prefill_chunk), so 24
    int kStage = 24;
    size_t blob_max = 0;
    // the hand-off between two GPUs' stages of a chunk: slot gi is GPU gi's input ([1 + states][T][E], pinned)
    struct Slot {
        float* h = nullptr;
        std::mutex mu;
        std::condition_variable cv;
        bool full = false;
        int nb = 0;
    };
    std::vector<std::unique_ptr<Slot>> slots;
    std::atomic<bool> abort_run{false};

    // (tests) the old decode step: the time its host thread waits in the stream's synchronisations (the GPU still at
    // work on what is queued) - the rest of a step is the host queueing, the GPU idle whenever its queue runs dry
    double sync_wait_s = 0, step_s = 0;
    int64_t step_n = 0;
    cudaError_t sync_timed(cudaStream_t s) {
        const double t = now_s();
        const cudaError_t e = cudaStreamSynchronize(s);
        sync_wait_s += now_s() - t;
        return e;
    }
    // IK2_HIP_PROF=1 (tests): the decode step's time by part (a synchronisation each: slower in all), printed at the end
    bool prof = std::getenv("IK2_HIP_PROF") != nullptr;
    double prof_t[8] = {0, 0, 0, 0, 0, 0, 0, 0}, prof_last = 0;
    void mark(Gpu& gp, int part) {
        if (!prof) return;
        cudaStreamSynchronize(gp.s);
        const double t = now_s();
        if (part >= 0) prof_t[part] += t - prof_last;
        prof_last = t;
    }
    ~Impl() {
        fast_stop();
        if (step_n > 0 && std::getenv("IK2_HIP_TIMELINE"))
            std::fprintf(stderr, "TIMELINE (the step with a host round trip a layer): %.2f ms a token, %.2f ms of it waiting on the GPU, "
                                 "%.2f ms queueing\n", 1e3 * step_s / step_n, 1e3 * sync_wait_s / step_n,
                         1e3 * (step_s - sync_wait_s) / step_n);
        if (fast_n > 0 && std::getenv("IK2_HIP_TIMELINE"))
            std::fprintf(stderr, "TIMELINE (the host-free step): %.2f ms a token; queueing a GPU's layers %.2f ms (each GPU's own thread), "
                                 "the lane's work %.2f ms, the lane waiting for routes %.2f ms\n", 1e3 * fast_s / fast_n,
                         1e3 * fast_enq_s / fast_n / std::max<size_t>(1, G.size()), 1e3 * lane_work_s / fast_n,
                         1e3 * lane_wait_s / fast_n);
        if (phases && fast_n > 0) {
            static const char* nm[10] = {"", "mix + rms", "input projections", "convs + clips + cache rows", "attention",
                                         "gate + output + residual", "mix + rms + router + publish + plan", "GPU experts + sum",
                                         "lane wait", "lane sum + conv + residual"};
            for (int T = 1; T <= kWin; ++T) {   // (a line a window size: T 1 is the token step)
                if (ph_n[T] == 0) continue;
                const double n = (double) ph_n[T];
                double sum = 0;
                for (int k2 = 1; k2 < 10; ++k2) sum += ph_ms[T][k2];
                std::fprintf(stderr, "PHASES T=%d (device time between marks, ms a pass over %lld; step %.2f):", T,
                             (long long) ph_n[T], 1e3 * ph_s[T] / n);
                for (int k2 = 1; k2 < 10; ++k2) std::fprintf(stderr, " %s %.2f;", nm[k2], ph_ms[T][k2] / n);
                std::fprintf(stderr, " between layers / hand-offs / head / host %.2f\n", 1e3 * ph_s[T] / n - sum / n);
            }
        }
        if (prof)
            std::fprintf(stderr, "PROF decode, s: inputs + projections %.3f, convs + clips + store %.3f, attention %.3f, gate + "
                                 "output %.3f, mix + router %.3f, experts %.3f, rest %.3f\n",
                         prof_t[0], prof_t[1], prof_t[2], prof_t[3], prof_t[4], prof_t[5], prof_t[6]);
        for (Gpu& gp : G)
            if (gp.sw_thread.joinable()) {
                {
                    std::lock_guard<std::mutex> lk(gp.sw_mu);
                    gp.sw_quit = true;
                }
                gp.sw_cv.notify_all();
                gp.sw_thread.join();
            }
        for (Gpu& gp : G) {
            cudaSetDevice(gp.dev);
            for (void* p : gp.allocs) cudaFree(p);
            if (gp.s) cudaStreamDestroy(gp.s);
        }
        for (void* p : {(void*) h_x, (void*) h_state, (void*) h_w, (void*) h_wres, (void*) h_xf, (void*) h_cpu,
                        (void*) h_top, (void*) h_ids, (void*) h_plan, (void*) h_big})
            if (p) cudaFreeHost(p);
#if defined(__linux__)
        for (Layer& l : L)
            if (l.ram) {
                if (l.ram_reg) cudaHostUnregister(l.ram);
                munmap(l.ram, l.ram_bytes);
            }
#endif
    }

    // ---- loading -------------------------------------------------------------------------------------------------
    const Tensor* need(const std::string& n, std::string& err) {
        const Tensor* t = pk.get(n);
        if (!t) err = "the pack has no " + n;
        return t;
    }
    // an F32 tensor of `want` values on the device
    bool f32(Gpu& gp, const std::string& n, float*& p, std::string& err, int64_t want) {
        const Tensor* t = need(n, err);
        if (!t) return false;
        if (t->type != "F32" || (int64_t) (t->bytes / 4) != want) {
            err = n + ": " + t->type + " of " + std::to_string(t->bytes / 4) + " values, F32 of " + std::to_string(want) +
                  " expected";
            return false;
        }
        return gp.alloc(p, (size_t) want, err) && gp.h2d(p, pk.data(*t), t->bytes, err);
    }
    // a matrix [k, n] in a type the GEMVs take (BF16: as F32 in VRAM)
    bool mat(Gpu& gp, const std::string& n, Mat& m, std::string& err, int k, int nout) {
        const Tensor* t = need(n, err);
        if (!t) return false;
        m.type = type_id(t->type);
        m.k = (int) t->n(0);
        m.n = (int) t->n(1);
        if (m.k != k || m.n != nout || t->ne.size() > 2) {
            err = n + ": [" + std::to_string(m.k) + ", " + std::to_string(m.n) + "], expected [" + std::to_string(k) +
                  ", " + std::to_string(nout) + "]";
            return false;
        }
        if (m.type == kBF16) {
            const size_t nv = (size_t) m.k * m.n;
            std::vector<float> f(nv);
            const uint16_t* src = (const uint16_t*) pk.data(*t);
            for (size_t i = 0; i < nv; ++i) {
                const uint32_t u = (uint32_t) src[i] << 16;
                std::memcpy(&f[i], &u, 4);
            }
            float* d = nullptr;
            if (!gp.alloc(d, nv, err) || !gp.h2d(d, f.data(), nv * 4, err)) return false;
            m.w = (const uint8_t*) d;
            m.type = kF32;
            m.ld = (size_t) m.k * 4;
            return true;
        }
        m.ld = m.type == kF32 ? (size_t) m.k * 4 : (iq::mv_supported(m.type) && m.k % iq::block_values(m.type) == 0
                                                         ? iq::row_bytes(m.type, m.k) : 0);
        if (m.ld == 0 || m.ld * (size_t) m.n != t->bytes) {
            err = n + ": " + t->type + ", not a type maya-ik2's GEMVs read";
            return false;
        }
        uint8_t* d = nullptr;
        if (!gp.alloc(d, t->bytes, err) || !gp.h2d(d, pk.data(*t), t->bytes, err)) return false;
        m.w = d;
        return true;
    }
    template <typename T>
    bool zeros(Gpu& gp, T*& p, size_t n, std::string& err) {
        if (!gp.alloc(p, n, err)) return false;
        CK(cudaMemset(p, 0, n * sizeof(T)));
        return true;
    }

    size_t tensor_bytes(const std::string& n) const {
        const Tensor* t = pk.get(n);
        return t ? (size_t) t->bytes * (t->type == "BF16" ? 2 : 1) : 0;
    }
    size_t cache_rows(int il) const { return g.swa[(size_t) il] ? (size_t) (g.n_swa + o.rewind + Tc) : (size_t) o.max_ctx; }
    int n_states() const { return (g.n_layer + g.res_block - 1) / g.res_block; }
    // the attention's scores (F32, and their F16 copy beside): floats.  A pass of ts tokens over nk keys takes
    // n_head x ts x nk of them (attention_rows halves ts until it fits), so they must hold one token over every key
    // the context has, n_head x max_keys (48 x 1M = 50.3M at the model's 1M).  kS_n, the floor (IK2_HIP_S_FLOOR_MI):
    // the passes a long prompt's chunk takes - measured, an 80K-token prompt in 2048-token chunks, tok/s at 11 / 22 /
    // 44 / 88 / 176 / 352 Mi: 853 / 1082 / 1247 / 1354 / 1409 / 1400 (20K: within 1%, 352 3% slower); 44 Mi is 264 MiB
    // a GPU (F32 + F16) - 176's 0.8 GiB more, with E9's and E44's, cost the shorter prompts and the RAM (prefill_chunk)
    size_t kS_n = (size_t) 44 << 20;
    size_t S_n() const { return std::max(kS_n, (size_t) g.n_head * max_keys); }
    // ... and its tokens a pass at most (IK2_HIP_TS_MAX); its rel-bias table is kTsMax x n_head x rel_extent floats,
    // 25 MB.  Measured, 2048-token chunks, tok/s at 32 / 64 / 128 / 256 / 512: a 20K prompt 1806 / 1861 / 1875 / 1844 /
    // 1841, 80K 1483 / 1524 / 1532 / 1513 / -
    int kTsMax = 128;
    bool chunk_scratch(Gpu& gp, std::string& err) {
        if (Tc <= 0) return true;
        const size_t blob_max = gp.blob_max;
        const size_t T = (size_t) Tc, E = (size_t) g.n_embd, HD = (size_t) g.n_head * g.head_dim, nu = (size_t) g.n_used;
        const size_t kvd = (size_t) g.n_head_kv * g.head_dim, H = (size_t) g.n_head;
        gp.wband_n = HD * E;
        gp.S_n = S_n();
        gp.ts_max = kTsMax;
        if (cublasCreate(&gp.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(gp.blas, gp.s) != CUBLAS_STATUS_SUCCESS) {
            err = "GPU " + std::to_string(gp.dev) + ": the BLAS did not start";
            return false;
        }
        CK(cudaStreamCreateWithFlags(&gp.s2, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            CK(cudaEventCreateWithFlags(&gp.ev_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&gp.ev_done[b], cudaEventDisableTiming));
            if (!gp.alloc(gp.xstage[b], (size_t) kStage * blob_max, err) || !gp.alloc(gp.pmap[b], (size_t) g.n_expert, err))
                return false;
        }
        CK(cudaMallocHost((void**) &gp.hc_ids, T * nu * 4));
        CK(cudaMallocHost((void**) &gp.hc_rw, T * nu * 4));
        CK(cudaMallocHost((void**) &gp.hc_rl, T * (size_t) g.n_expert * 4));
        CK(cudaMallocHost((void**) &gp.hc_pmap, (size_t) g.n_expert * g.n_expert * 4));   // (a pass's map each: one expert a pass at most)
        return gp.alloc(gp.c_cur, T * E, err) && gp.alloc(gp.c_blocks, (size_t) n_states() * T * E, err) &&
               gp.alloc(gp.c_mixo, 3 * T * E, err) && gp.alloc(gp.c_xn, 3 * T * E, err) &&
               gp.alloc(gp.c_stat, 4 * k::kMaxStates * T, err) && gp.alloc(gp.c_q, T * HD, err) &&
               gp.alloc(gp.c_gate, T * HD, err) && gp.alloc(gp.c_ao, T * HD, err) && gp.alloc(gp.c_qp, T * HD, err) &&
               gp.alloc(gp.c_op, T * HD, err) && gp.alloc(gp.c_rel, T * H * g.d_rel, err) &&
               gp.alloc(gp.c_sink, T * H, err) && gp.alloc(gp.c_ky, T * kvd, err) && gp.alloc(gp.c_vy, T * kvd, err) &&
               gp.alloc(gp.c_kc, T * kvd, err) && gp.alloc(gp.c_vc, T * kvd, err) && gp.alloc(gp.c_dk, T * 32, err) &&
               gp.alloc(gp.c_dv, T * 32, err) && gp.alloc(gp.c_o, T * E, err) && gp.alloc(gp.c_oc, T * E, err) &&
               gp.alloc(gp.c_xf, T * E, err) && gp.alloc(gp.c_rl, T * (size_t) g.n_expert, err) &&
               gp.alloc(gp.c_rw, T * nu, err) && gp.alloc(gp.c_ids, T * nu, err) && gp.alloc(gp.c_y, T * E, err) &&
               gp.alloc(gp.c_yc, T * E, err) && gp.alloc(gp.c_sy, T * nu * E, err) && gp.alloc(gp.c_ent, T * nu * E, err) &&
               gp.alloc(gp.c_sh16, T * nu * (size_t) g.n_ff_exp, err) && gp.alloc(gp.c_bounds, (size_t) g.n_expert + 4, err) &&
               gp.alloc(gp.c_stok, T * nu, err) && gp.alloc(gp.c_srow, T * nu, err) && gp.alloc(gp.xh16, T * HD, err) &&
               gp.alloc(gp.wband, gp.wband_n, err) && gp.alloc(gp.c_S, gp.S_n, err) && gp.alloc(gp.c_P16, gp.S_n, err) &&
               gp.alloc(gp.c_tbl, (size_t) kTsMax * H * (size_t) g.rel_extent, err);
    }
    // a layer's VRAM without its experts: every other tensor of the block, the caches and the histories
    size_t dense_bytes(int il) const {
        const std::string p = "blk." + std::to_string(il) + ".";
        size_t b = 0;
        for (auto it = pk.t.lower_bound(p); it != pk.t.end() && it->first.compare(0, p.size(), p) == 0; ++it)
            if (it->first.find("_exps.") == std::string::npos) b += tensor_bytes(it->first);
        const size_t kvd = (size_t) g.n_head_kv * g.head_dim;
        b += cache_rows(il) * kvd * 2 * 2 + (size_t) (1 + kSnaps) * hist_n() * 4;   // (the split's weights only)
        return b;
    }
    // the geometry of a layer's expert blobs (no data)
    bool expert_geo(Layer& l, std::string& err) {
        const std::string p = "blk." + std::to_string(l.il) + ".";
        const int E = g.n_embd, ff = g.n_ff_exp;
        l.t_gu = need(p + "ffn_gate_up_exps.weight", err);
        l.t_dn = l.t_gu ? need(p + "ffn_down_exps.weight", err) : nullptr;
        if (!l.t_dn) return false;
        l.tg = type_id(l.t_gu->type);
        l.td = type_id(l.t_dn->type);
        if (!iq::expert_supported(l.tg) || !iq::expert_supported(l.td) || !iq::mv_supported(l.tg) ||
            !iq::mv_supported(l.td) || l.t_gu->n(0) != E || l.t_gu->n(1) != 2 * ff || l.t_gu->n(2) != g.n_expert ||
            l.t_dn->n(0) != ff || l.t_dn->n(1) != E || l.t_dn->n(2) != g.n_expert) {
            err = p + ": routed experts " + l.t_gu->type + " / " + l.t_dn->type + " (maya-ik2 runs ggml's expert "
                  "types, a fused gate | up tensor)";
            return false;
        }
        const size_t rg = iq::row_bytes(l.tg, E);
        l.gub = rg * (size_t) (2 * ff);
        l.db = iq::row_bytes(l.td, ff) * (size_t) E;
        if (l.gub * (size_t) g.n_expert != l.t_gu->bytes || l.db * (size_t) g.n_expert != l.t_dn->bytes) {
            err = p + ": the experts' tensors are not whole rows";
            return false;
        }
        l.tr[0] = 0;
        l.tr[1] = rg * (size_t) ff;   // (the fused tensor's up rows follow its gate rows)
        l.tr[2] = al256(l.gub);
        l.blob = al256(l.tr[2] + l.db);
        return true;
    }
    void blob_copy(const Layer& l, int e, uint8_t* dst) const {
        std::memcpy(dst, pk.data(*l.t_gu) + (size_t) e * l.gub, l.gub);
        std::memcpy(dst + l.tr[2], pk.data(*l.t_dn) + (size_t) e * l.db, l.db);
    }

    // the attention's scratch, floats: a token's tiles over every key, or a window's 9 rows' (kWin) over up to 4096 keys
    // - every SWA layer's window and a global layer's first 4096 positions in one launch pair, 58 MB a GPU (a window
    // whose keys are more runs its rows one at a time)
    size_t S_need() const {
        const size_t one = std::max((size_t) g.n_head * (2 * max_keys + 256),
                                    k::attention_split_floats(g.n_head, g.head_dim, (int64_t) max_keys));
        return std::max(one, (size_t) 9 * g.n_head * (4096 / 16 + 2) * (size_t) (g.head_dim + 2));
    }
    bool scratch(Gpu& gp, bool last, std::string& err) {
        const size_t E = (size_t) g.n_embd, HD = (size_t) g.n_head * g.head_dim, nu = (size_t) g.n_used;
        const size_t kvd = (size_t) g.n_head_kv * g.head_dim, ff = (size_t) g.n_ff_exp;
        const size_t qmax = std::max(nu * ff, std::max(E, HD));
        return gp.alloc(gp.cur, (1 + k::kMaxStates) * E, err) && (gp.blocks = gp.cur + E) != nullptr && gp.alloc(gp.mixo, 3 * E, err) &&
               gp.alloc(gp.xn, 3 * E, err) && gp.alloc(gp.stat, 4 * k::kMaxStates, err) && gp.alloc(gp.q, HD, err) &&
               gp.alloc(gp.gate, HD, err) && gp.alloc(gp.rel, (size_t) g.n_head * g.d_rel, err) &&
               gp.alloc(gp.sink, (size_t) g.n_head, err) && gp.alloc(gp.ky, kvd, err) && gp.alloc(gp.vy, kvd, err) &&
               gp.alloc(gp.dk, (size_t) 4 * g.n_head_kv, err) && gp.alloc(gp.dv, (size_t) 4 * g.n_head_kv, err) &&
               gp.alloc(gp.S, gp.S_floats = S_need(), err) && gp.alloc(gp.ao, HD, err) &&
               gp.alloc(gp.o, E, err) && gp.alloc(gp.xf, E, err) && gp.alloc(gp.rl, (size_t) g.n_expert, err) &&
               gp.alloc(gp.rw, 16, err) && gp.alloc(gp.wres, 16, err) && gp.alloc(gp.ids, 16, err) &&
               gp.alloc(gp.gu, nu * 2 * ff, err) && gp.alloc(gp.eh, nu * ff, err) && gp.alloc(gp.part, nu * E, err) &&
               gp.alloc(gp.y, E, err) && gp.alloc(gp.cpo, E, err) && gp.alloc(gp.xq, qmax, err) &&
               gp.alloc(gp.xd, qmax / 32, err) && gp.alloc(gp.ehq, nu * ff, err) && gp.alloc(gp.ehd, nu * ff / 32, err) &&
               gp.alloc(gp.plan, 16, err) && gp.alloc(gp.top, 2, err) &&
               (!last || gp.alloc(gp.logits, (size_t) g.n_vocab, err));
    }

    // RAM left to the rest (the system, the page cache's churn): 8 GiB, or IK2_HIP_RAM_MARGIN_GB=<GiB> (0 to 4096; read
    // once, logged once; an unreadable value keeps the 8) - maya-ds4's ram_margin and its DS4_RAM_MARGIN_GB
    static uint64_t ram_margin() {
        static const uint64_t m = [] {
            double gib = 8.0;
            if (const char* v = std::getenv("IK2_HIP_RAM_MARGIN_GB")) {
                char* end = nullptr;
                const double x = std::strtod(v, &end);
                if (end != v && x >= 0.0 && x <= 4096.0) gib = x;
                else std::fprintf(stderr, "ik2: IK2_HIP_RAM_MARGIN_GB=%s is not a number from 0 to 4096: keeping 8 GiB\n", v);
                if (gib != 8.0) std::fprintf(stderr, "ik2: RAM margin %.1f GiB (IK2_HIP_RAM_MARGIN_GB)\n", gib);
            }
            return (uint64_t) (gib * 1073741824.0);
        }();
        return m;
    }
    // The split: the layers over the GPUs by their bytes (--layer-split sets it by hand)
    bool split(std::string& err) {
        const int ng = (int) G.size();
        std::vector<size_t> all((size_t) NL);
        size_t total = 0;
        for (int il = 0; il < NL; ++il) {
            Layer& l = L[(size_t) il];
            l.il = il;
            l.swa = g.swa[(size_t) il] != 0;
            l.ring = l.swa ? g.n_swa + o.rewind + Tc : 0;
            if (!expert_geo(l, err)) return false;
            l.dense = dense_bytes(il);
            all[(size_t) il] = l.dense + l.blob * (size_t) g.n_expert;
            total += all[(size_t) il];
            l.slot_of.assign((size_t) g.n_expert, -1);
            l.ram_of.assign((size_t) g.n_expert, -1);
        }
        for (const Layer& l : L) blob_max = std::max(blob_max, l.blob);
        std::vector<int> first((size_t) ng + 1, NL);
        first[0] = 0;
        if (!o.layer_split.empty()) {
            if ((int) o.layer_split.size() != ng - 1) {
                err = "--layer-split: " + std::to_string(ng - 1) + " values for " + std::to_string(ng) + " GPUs";
                return false;
            }
            for (int i = 1; i < ng; ++i) first[(size_t) i] = std::min(NL, std::max(first[(size_t) i - 1], o.layer_split[(size_t) i - 1]));
        } else {
            size_t acc = 0;
            int gi = 1;
            for (int il = 0; il < NL && gi < ng; ++il) {
                acc += all[(size_t) il];
                while (gi < ng && acc * (size_t) ng >= total * (size_t) gi) first[(size_t) gi++] = il + 1;
            }
        }
        st.experts_all = NL * g.n_expert;
        for (int gi = 0; gi < ng; ++gi) {
            Gpu& gp = G[(size_t) gi];
            gp.l0 = first[(size_t) gi];
            gp.l1 = first[(size_t) gi + 1];
            gp.blob_max = 1;
            for (int il = gp.l0; il < gp.l1; ++il) {
                gp.blob_max = std::max(gp.blob_max, L[(size_t) il].blob);
                L[(size_t) il].gpu = gi;
            }
        }
        return true;
    }
    // The experts' plan, each GPU on its own (maya-ds4's place_experts: the experts take what is left): what the GPU
    // has free now - every other allocation of the engine made (its layers' dense tensors and caches, the scratch, the
    // drafter) and the runtime's own taken (warm_up: the BLAS's workspace and kernels, the kernels' code, their
    // scratch) - less reserve_mib, filled with its layers' experts, the most routed first by the profile (none: each
    // layer's first ones, the layers in turn), each layer a spare slot more.  A GPU's experts are one allocation, which
    // the runtime rounds up to its granularity (gran, measured at the load)
    bool place(std::string& err) {
        const uint64_t keep = (uint64_t) std::max<int64_t>(0, o.reserve_mib) << 20;
        for (size_t gi = 0; gi < G.size(); ++gi) {
            Gpu& gp = G[gi];
            CK(cudaSetDevice(gp.dev));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            gp.free_plan = fr;
            auto rounded = [&](size_t b) { return (b + gp.gran - 1) / gp.gran * gp.gran; };
            size_t need = 0;   // (the experts' allocation: every layer's spare slot, then the experts placed)
            for (int il = gp.l0; il < gp.l1; ++il) need += L[(size_t) il].blob;
            if ((uint64_t) rounded(need) + keep > fr) {
                err = "GPU " + std::to_string(gp.dev) + ": " + std::to_string(fr >> 20) + " MiB free with layers " +
                      std::to_string(gp.l0) + " .. " + std::to_string(gp.l1 - 1) + " in place without their experts, " +
                      std::to_string(o.reserve_mib) + " MiB to keep free: no room for the experts (fewer layers on it: "
                      "--layer-split, or a smaller --max-context)";
                return false;
            }
            const size_t room = fr - (size_t) keep;
            struct Cand {
                uint32_t cnt;
                int e, il;
            };
            std::vector<Cand> c;
            const int cap = std::max(0, std::min(g.n_expert, (int) std::lround(o.vram_experts * g.n_expert)));
            for (int il = gp.l0; il < gp.l1; ++il) {
                const bool pr = (size_t) (il + 1) * g.n_expert <= prof_in.size();
                for (int e = 0; e < g.n_expert; ++e)
                    c.push_back({pr ? prof_in[(size_t) il * g.n_expert + e] : 0u, e, il});
            }
            // (routings a byte: the last layers' experts are larger)
            std::stable_sort(c.begin(), c.end(), [&](const Cand& a, const Cand& b) {
                const double ka = (double) a.cnt / (double) L[(size_t) a.il].blob, kb = (double) b.cnt / (double) L[(size_t) b.il].blob;
                return ka != kb ? ka > kb : a.e < b.e;
            });
            std::vector<int> n_in((size_t) NL, 0);
            for (const Cand& x : c) {
                Layer& l = L[(size_t) x.il];
                if (n_in[(size_t) x.il] >= cap || rounded(need + l.blob) > room) continue;
                need += l.blob;
                l.slot_of[(size_t) x.e] = 0;   // (numbered below, in the experts' order)
                ++n_in[(size_t) x.il];
            }
            gp.expert_alloc = need;
            for (int il = gp.l0; il < gp.l1; ++il) {
                Layer& l = L[(size_t) il];
                int ns = 0;
                for (int e = 0; e < g.n_expert; ++e)
                    if (l.slot_of[(size_t) e] == 0) l.slot_of[(size_t) e] = ns++;
                l.n_res = ns;
                gp.n_res += ns;
                gp.expert_bytes += l.blob * (size_t) ns;
                st.experts_vram += ns;
            }
        }
        return place_ram();
    }
    // The experts outside VRAM: copied into RAM or read in place from the model's files - maya-ds4's copy_experts
    // rule (--mmap-experts): off: the most routed (routings a byte, as the VRAM ones) that fit the RAM this process
    // can get beside ram_margin(), the others in place; on: none.  auto, the default: maya-ds4's copies all of them
    // when they fit with 8 GiB more to spare, else none (its #339: a part copied cost decode speed on 32 / 64 GB) -
    // here it is off's: measured at a 1M context (131 GB outside VRAM, 109 GB to be had), the 2.7k-token prompt
    // continued with drafts at 47.2 tok/s with 100 GiB copied against 32.2 with none, its prompt read alike (49 / 51
    // tok/s, both bound by the disk: docs/INKLING2_AMD.md).  A layer's RAM slots: the copied ones first (0 .. n_copy -
    // 1, a blob each in `ram`), then the ones in the files
    bool place_ram() {
        struct Out {
            double key;
            int il, e;
        };
        std::vector<Out> out;
        double all = 0;
        for (int il = 0; il < NL; ++il) {
            const Layer& l = L[(size_t) il];
            const bool pr = (size_t) (il + 1) * g.n_expert <= prof_in.size();
            for (int e = 0; e < g.n_expert; ++e)
                if (l.slot_of[(size_t) e] < 0) {
                    out.push_back({(pr ? (double) prof_in[(size_t) il * g.n_expert + e] : 0.0) / (double) l.blob, il, e});
                    all += (double) l.blob;
                }
        }
        std::stable_sort(out.begin(), out.end(), [](const Out& a, const Out& b) { return a.key > b.key; });
        const uint64_t avail = hy4::ram_available();
        const double can = avail == 0 ? 1e300 : std::max(0.0, (double) avail - (double) ram_margin());
        double budget = o.mmap_experts > 0 ? 0.0 : std::min(all, can);
        if (const char* v = std::getenv("IK2_HIP_COPY_GB")) budget = std::atof(v) * 1e9;   // (tests: the hottest n GB)
        if (o.mmap_experts <= 0 && all > can)
            std::fprintf(stderr, "maya-ik2: --mmap-experts %s: the CPU lane's experts (%.1f GB) do not all fit the RAM "
                                 "this process can get (%.1f GB beside %.0f GB for the rest) - the most routed copied, the "
                                 "others read in place from the model's files\n", o.mmap_experts < 0 ? "auto" : "off",
                         all / 1e9, can / 1e9, ram_margin() / 1e9);
        std::vector<char> cp((size_t) NL * g.n_expert, 0);
        double used = 0;
        for (const Out& x : out) {   // (the hottest that fit the budget)
            const double b = (double) L[(size_t) x.il].blob;
            if (used + b > budget) break;
            used += b;
            cp[(size_t) x.il * g.n_expert + x.e] = 1;
        }
        for (int il = 0; il < NL; ++il) {
            Layer& l = L[(size_t) il];
            int nr = 0;
            for (int e = 0; e < g.n_expert; ++e)
                if (l.slot_of[(size_t) e] < 0 && cp[(size_t) il * g.n_expert + e]) l.ram_of[(size_t) e] = nr++;
            l.n_copy = nr;
            for (int e = 0; e < g.n_expert; ++e)
                if (l.slot_of[(size_t) e] < 0 && !cp[(size_t) il * g.n_expert + e]) l.ram_of[(size_t) e] = nr++;
        }
        st.ram_gb = used / 1e9;
        st.file_gb = (all - used) / 1e9;
        return true;
    }

    // a layer's every tensor but its routed experts, its caches and histories (the experts: load_experts, after place)
    bool load_dense(Layer& l, Gpu& gp, std::string& err) {
        const std::string p = "blk." + std::to_string(l.il) + ".";
        const int E = g.n_embd, HD = g.n_head * g.head_dim, kvd = g.n_head_kv * g.head_dim;
        bool ok = true;
        if (l.swa) {
            ok = f32(gp, p + "attn_norm.weight", l.attn_norm, err, E);
        } else {
            ok = f32(gp, p + "attn_q_inp_norm.weight", l.q_norm, err, E) &&
                 f32(gp, p + "attn_k_inp_norm.weight", l.k_norm, err, E) &&
                 f32(gp, p + "attn_v_inp_norm.weight", l.v_norm, err, E) &&
                 mat(gp, p + "attn_k_conv_proj.weight", l.k_proj, err, E, 4 * g.n_head_kv) &&
                 mat(gp, p + "attn_v_conv_proj.weight", l.v_proj, err, E, 4 * g.n_head_kv);
        }
        if (l.il > 0) ok = ok && f32(gp, p + "attn_res_query.weight", l.attn_res, err, (int64_t) E * (l.swa ? 1 : 3));
        ok = ok && mat(gp, p + "attn_q.weight", l.q, err, E, HD) && mat(gp, p + "attn_gate.weight", l.gate, err, E, HD) &&
             mat(gp, p + "attn_k.weight", l.wk, err, E, kvd) && mat(gp, p + "attn_v.weight", l.wv, err, E, kvd) &&
             mat(gp, p + "attn_rel.weight", l.rel, err, E, g.n_head * g.d_rel) &&
             mat(gp, p + "attn_sink_proj.weight", l.sink, err, E, g.n_head) &&
             mat(gp, p + "attn_output.weight", l.wo, err, HD, E) &&
             f32(gp, p + "attn_q_clip.weight", l.q_clip, err, HD) &&
             f32(gp, p + "attn_rel_clip.weight", l.rel_clip, err, (int64_t) g.n_head * g.d_rel) &&
             f32(gp, p + "attn_rel_bias.weight", l.rel_bias, err, (int64_t) g.d_rel * (l.swa ? g.n_swa : g.rel_extent)) &&
             f32(gp, p + "attn_k_conv.weight", l.k_conv, err, 4 * kvd) &&
             f32(gp, p + "attn_v_conv.weight", l.v_conv, err, 4 * kvd) &&
             f32(gp, p + "attn_post_conv.weight", l.a_conv, err, 4 * E) &&
             f32(gp, p + "ffn_post_conv.weight", l.f_conv, err, 4 * E) &&
             f32(gp, p + "ffn_res_query.weight", l.ffn_res, err, E) && f32(gp, p + "ffn_norm.weight", l.ffn_norm, err, E) &&
             mat(gp, p + "ffn_gate_inp.weight", l.router, err, E, g.n_expert);
        if (!ok) return false;
        if (l.sink.type != kF32 || l.router.type != kF32 || (!l.swa && (l.k_proj.type != kF32 || l.v_proj.type != kF32))) {
            err = p + ": the sink / router / tap projections are not F32 or BF16";
            return false;
        }
        if (const Tensor* tb = pk.get(p + "exp_probs_b.bias")) {
            if (!f32(gp, p + "exp_probs_b.bias", l.router_b, err, g.n_expert)) return false;
            l.h_bias.resize((size_t) g.n_expert);
            std::memcpy(l.h_bias.data(), pk.data(*tb), (size_t) g.n_expert * 4);
        }
        const size_t rows = cache_rows(l.il);
        if (!zeros(gp, l.hist_k, (size_t) 3 * kvd, err) || !zeros(gp, l.hist_v, (size_t) 3 * kvd, err) ||
            !zeros(gp, l.hist_a, (size_t) 3 * E, err) || !zeros(gp, l.hist_f, (size_t) 3 * E, err) ||
            !gp.alloc(l.K, rows * kvd, err) || !gp.alloc(l.V, rows * kvd, err) || !gp.alloc(l.snap, (size_t) kSnaps * hist_n(), err))
            return false;
        l.use.assign((size_t) g.n_expert, 0.0f);   // (the warm-up's routings land here: put back after it)
        return gp.alloc(l.slot_d, (size_t) g.n_expert, err) &&
               gp.h2d(l.slot_d, l.slot_of.data(), sizeof(int) * (size_t) g.n_expert, err);
    }
    // its routed experts: the VRAM ones at `at` (its share of the GPU's experts' allocation, n_res + 1 blobs: the
    // cache's spare slot), the others copied into RAM
    bool load_experts(Layer& l, Gpu& gp, uint8_t* at, std::string& err) {
        const std::string p = "blk." + std::to_string(l.il) + ".";
        const int E = g.n_embd;
        if (!gp.h2d(l.slot_d, l.slot_of.data(), sizeof(int) * (size_t) g.n_expert, err)) return false;
        l.experts = at;
        l.spare = l.n_res;
        l.use.assign((size_t) g.n_expert, 0.0f);
        if ((size_t) (l.il + 1) * g.n_expert <= prof_in.size()) {   // (the profile as the counts' start, at most 1)
            const uint32_t* c = prof_in.data() + (size_t) l.il * g.n_expert;
            const uint32_t mx = std::max<uint32_t>(1, *std::max_element(c, c + g.n_expert));
            for (int e = 0; e < g.n_expert; ++e) l.use[(size_t) e] = (float) c[e] / (float) mx;
        }
        std::vector<int> res, ram, fil;   // (ram / fil: by their RAM slots - the copied ones, the ones in the files)
        for (int e = 0; e < g.n_expert; ++e)
            if (l.slot_of[(size_t) e] >= 0) res.push_back(e);
        ram.resize((size_t) l.n_copy);
        for (int e = 0; e < g.n_expert; ++e) {
            const int r = l.ram_of[(size_t) e];
            if (l.slot_of[(size_t) e] >= 0 || r < 0) continue;
            if (r < l.n_copy) ram[(size_t) r] = e;
            else {
                if ((size_t) (r - l.n_copy) >= fil.size()) fil.resize((size_t) (r - l.n_copy) + 1);
                fil[(size_t) (r - l.n_copy)] = e;
            }
        }
        const size_t per = std::max<size_t>(1, gp.stage_bytes / l.blob);   // (straight into the pinned stage)
        for (size_t s0 = 0; s0 < res.size(); s0 += per) {
            const size_t ne = std::min(per, res.size() - s0);
            for (size_t i = 0; i < ne; ++i) blob_copy(l, res[s0 + i], gp.stage + i * l.blob);
            CK(cudaMemcpy(l.experts + s0 * l.blob, gp.stage, l.blob * ne, cudaMemcpyHostToDevice));
        }
        // the others into RAM (anonymous memory, huge pages where the system gives them, page-locked when it lets us),
        // or read in place from the files (--mmap-experts)
        if (!ram.empty() || !fil.empty()) {
            l.cex.beta = g.situ_beta;
            l.cex.lb = g.situ_lb;
            if (!l.cex.init(l.tg, l.td, E, g.n_ff_exp, err)) {
                err = p + ": the CPU lane does not take these experts: " + err;
                return false;
            }
            l.cex.n_copy = l.n_copy;
            l.cex.fgu.resize(fil.size());
            l.cex.fdn.resize(fil.size());
            for (size_t i = 0; i < fil.size(); ++i) {
                l.cex.fgu[i] = pk.data(*l.t_gu) + (size_t) fil[i] * l.gub;
                l.cex.fdn[i] = pk.data(*l.t_dn) + (size_t) fil[i] * l.db;
            }
        }
        if (!ram.empty()) {
#if defined(__linux__)
            l.ram_bytes = l.blob * ram.size();
            void* m = mmap(nullptr, l.ram_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (m == MAP_FAILED) {
                err = p + ": no RAM for " + std::to_string(l.ram_bytes >> 20) + " MiB of experts";
                return false;
            }
            l.ram = (uint8_t*) m;
            for (size_t i = 0; i < ram.size(); ++i) blob_copy(l, ram[i], l.ram + i * l.blob);
            // (page-locked through the GPU's driver, which the prompt path's copies to the staging slots need - DMA
            //  straight from these pages; mlock alone stops at the user's limit)
            // (after the load, register_ram: pinned pages while the file is still being read slow the reads to a crawl)
            // (the file's pages of this layer's experts are not read again: the mapping lets go of them - unless some
            //  are read in place)
            if (fil.empty()) {
                madvise((void*) ((uintptr_t) pk.data(*l.t_gu) & ~(uintptr_t) 4095), (size_t) l.t_gu->bytes, MADV_DONTNEED);
                madvise((void*) ((uintptr_t) pk.data(*l.t_dn) & ~(uintptr_t) 4095), (size_t) l.t_dn->bytes, MADV_DONTNEED);
            }
#else
            err = "the CPU lane's experts in RAM: Linux only for now";
            return false;
#endif
        }
#if defined(__linux__)
        if (!fil.empty())   // (the pages of its experts copied into VRAM or RAM: let go of, the others' kept)
            for (int e = 0; e < g.n_expert; ++e) {
                if (l.slot_of[(size_t) e] < 0 && l.ram_of[(size_t) e] >= l.n_copy) continue;
                for (const auto& [base, n] : {std::pair<uintptr_t, size_t>{(uintptr_t) (pk.data(*l.t_gu) + (size_t) e * l.gub), l.gub},
                                              std::pair<uintptr_t, size_t>{(uintptr_t) (pk.data(*l.t_dn) + (size_t) e * l.db), l.db}}) {
                    const uintptr_t a0 = (base + 4095) & ~(uintptr_t) 4095, a1 = (base + n) & ~(uintptr_t) 4095;
                    if (a1 > a0) madvise((void*) a0, a1 - a0, MADV_DONTNEED);
                }
            }
#endif
        l.cex.base = l.ram;
        l.cex.blob = l.blob;
        l.cex.tr_down = l.tr[2];
        return true;
    }
    // expert e of layer l from the files into dst in the blob layout (the gate | up rows at 0, down at tr[2])
    void file_blob(const Layer& l, int e, uint8_t* dst) const { blob_copy(l, e, dst); }
    // a gather of experts from the files into pinned memory, on up to 8 threads (the page cache's copies, or the
    // disk's reads, side by side): (expert, destination) pairs
    void file_gather(const Layer& l, const std::vector<std::pair<int, uint8_t*>>& jobs) const {
        if (jobs.empty()) return;
#if defined(__linux__)
        for (const auto& j : jobs) {   // (the disk reads ahead what is not in the page cache)
            const uintptr_t a = (uintptr_t) (pk.data(*l.t_gu) + (size_t) j.first * l.gub), b = (uintptr_t) (pk.data(*l.t_dn) + (size_t) j.first * l.db);
            madvise((void*) (a & ~(uintptr_t) 4095), l.gub + (a & 4095), MADV_WILLNEED);
            madvise((void*) (b & ~(uintptr_t) 4095), l.db + (b & 4095), MADV_WILLNEED);
        }
#endif
        const int nt = (int) std::min<size_t>(8, jobs.size());
        if (nt <= 1) {
            for (const auto& j : jobs) file_blob(l, j.first, j.second);
            return;
        }
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        for (int t = 0; t < nt; ++t)
            th.emplace_back([&] {
                for (size_t i; (i = next.fetch_add(1)) < jobs.size();) file_blob(l, jobs[i].first, jobs[i].second);
            });
        for (auto& x : th) x.join();
    }
    std::atomic<bool> ram_unlocked{false}, ram_unreg{false};

    // IK2_HIP_VERIFY_LOAD=1 (tests): every expert's bytes in VRAM and in RAM against the file's, a line a difference
    bool verify(std::string& err) {
        long bad = 0;
        for (Layer& l : L) {
            CK(cudaSetDevice(G[(size_t) l.gpu].dev));
            uint8_t* back = nullptr;
            CK(cudaMallocHost((void**) &back, l.blob * 32));
            std::vector<int> res((size_t) l.n_res);
            for (int e = 0; e < g.n_expert; ++e)
                if (l.slot_of[(size_t) e] >= 0) res[(size_t) l.slot_of[(size_t) e]] = e;
            auto cmp = [&](const uint8_t* b, int e, const char* where) {
                const uint8_t *fg = pk.data(*l.t_gu) + (size_t) e * l.gub, *fd = pk.data(*l.t_dn) + (size_t) e * l.db;
                size_t n = 0, first = (size_t) -1;
                for (size_t i = 0; i < l.gub; ++i)
                    if (b[i] != fg[i]) {
                        ++n;
                        first = std::min(first, i);
                    }
                for (size_t i = 0; i < l.db; ++i)
                    if (b[l.tr[2] + i] != fd[i]) {
                        ++n;
                        first = std::min(first, l.tr[2] + i);
                    }
                if (n) {
                    ++bad;
                    std::fprintf(stderr, "VERIFY layer %d expert %d (%s): %zu bytes differ, the first at %zu of %zu\n", l.il, e,
                                 where, n, first, l.blob);
                }
            };
            for (size_t s0 = 0; s0 < res.size(); s0 += 32) {
                const size_t ne = std::min<size_t>(32, res.size() - s0);
                CK(cudaMemcpy(back, l.experts + s0 * l.blob, l.blob * ne, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < ne; ++i) cmp(back + i * l.blob, res[s0 + i], "VRAM");
            }
            CK(cudaFreeHost(back));
            for (int e = 0; e < g.n_expert; ++e)
                if (l.ram_of[(size_t) e] >= 0 && l.ram_of[(size_t) e] < l.n_copy) cmp(l.ram + (size_t) l.ram_of[(size_t) e] * l.blob, e, "RAM");
        }
        std::fprintf(stderr, "VERIFY: %ld experts differ from the file\n", bad);
        return true;
    }

    // the runtime's rounding of an allocation over 1 MiB (here, ROCm 7 on an R9700: whole 2 MiB pages), measured -
    // what two probe allocations take of the free VRAM: the smallest power of two from 4 KiB that rounds both so (none:
    // the larger of the two, which no rounding of the experts' one allocation can pass)
    bool granule(Gpu& gp, std::string& err) {
        const size_t probe[2] = {((size_t) 1 << 20) + 1, ((size_t) 2 << 20) + 1};   // (2 MiB + 1: 3 MiB in 1 MiB pages)
        size_t took[2] = {0, 0};
        for (int i = 0; i < 2; ++i) {
            size_t a = 0, b = 0, t = 0;
            void* v = nullptr;
            CK(cudaMemGetInfo(&a, &t));
            CK(cudaMalloc(&v, probe[i]));
            CK(cudaMemGetInfo(&b, &t));
            CK(cudaFree(v));
            took[i] = a > b ? a - b : 0;
        }
        auto up = [](size_t x, size_t q) { return (x + q - 1) / q * q; };
        for (size_t q = 4096; q <= ((size_t) 64 << 20); q *= 2)
            if (up(probe[0], q) == took[0] && up(probe[1], q) == took[1]) {
                gp.gran = q;
                return true;
            }
        gp.gran = std::max({took[0], took[1], (size_t) 4096});
        return true;
    }
    // The runtime's own VRAM taken before the experts are placed, so that place counts it: the prompt path's first
    // run makes the BLAS load its kernels and the runtime the kernels' code and their scratch (and pays the seconds a
    // first run takes) - a dummy chunk through every layer, each layer's experts one zero blob in pinned memory (every
    // pick streamed from it), the drafter fed and asked once; the state wiped after it, the counts as they were
    // dummy false: the same chunk once the experts are in place - the first run of their own layout (the VRAM experts'
    // sets) pays seconds more, here and not in the first prompt
    bool warm_up(std::string& err, bool dummy = true) {
        if (Tc <= 1 || std::getenv("IK2_HIP_NO_WARMUP")) return true;
        const double t0 = now_s();
        if (!dummy) {
            const std::vector<uint32_t> keep = prof_cnt;
            std::vector<std::vector<float>> use;
            for (const Layer& l : L) use.push_back(l.use);
            const Stats st0 = st;
            const bool sw = o.expert_cache;
            o.expert_cache = false;
            // (what a request touches, again after the experts' load: that load's memory pressure sends the host memory
            //  the first warm-up touched - the BLAS's tables, the runtime's, the engine's own - to swap, and the first
            //  prompt then paged it back in: one of its layers 5.6 s, 2.7k tokens in 8.5 s instead of 2.8.  So a full
            //  chunk and a part of one, tokens spread over the vocabulary - their picks stream many of the RAM experts
            //  as a real prompt's do - then a few decode windows of a round's sizes)
            std::vector<int> ids((size_t) std::min<int64_t>(Tc + std::max(2, Tc / 4), o.max_ctx - 16), 0);   // (16: the windows after it)
            for (size_t t = 0; t < ids.size(); ++t) ids[t] = (int) ((t * 7919 + 13) % (size_t) g.n_vocab_unpadded);
            bool ok = prompt(ids.data(), (int) ids.size(), 0, nullptr, err);
            n_end = (int64_t) ids.size();
            if (ok && fast_on()) {
                int tops[kWin];
                for (int T : {1, 4, kWin}) {
                    ok = ok && step_win(ids.data(), T, n_end, tops, nullptr, err);
                    n_end += T;
                }
            }
            ok = ok && reset(err);
            o.expert_cache = sw;
            prof_cnt = keep;
            st = st0;
            for (size_t i = 0; i < use.size(); ++i) L[i].use = use[i];
            for (Gpu& gp : G) gp.n_streamed = gp.n_experts = 0;
            if (o.verbose) std::fprintf(stderr, "the warm-up with the experts in place: %.1f s\n", now_s() - t0);
            return ok;
        }
        for (Gpu& gp : G) {
            CK(cudaSetDevice(gp.dev));
            CK(cudaMallocHost((void**) &gp.h_dummy, gp.blob_max));
            std::memset(gp.h_dummy, 0, gp.blob_max);
        }
        for (Layer& l : L) {
            l.n_res = 0;
            l.ram = G[(size_t) l.gpu].h_dummy;
            l.experts = G[(size_t) l.gpu].xstage[0];   // (never read: no expert is in a VRAM slot)
            std::fill(l.ram_of.begin(), l.ram_of.end(), 0);
            l.n_copy = 1;   // (RAM slot 0, the zero blob, copied)
        }
        const std::vector<uint32_t> keep = prof_cnt;
        const Stats st0 = st;
        const bool sw = o.expert_cache;
        o.expert_cache = false;
        std::vector<int> ids((size_t) std::min(Tc, warm_n), 0);
        bool ok = prompt(ids.data(), (int) ids.size(), 0, nullptr, err);
        if (ok && ds) {
            n_end = (int64_t) ids.size();
            std::vector<int> t;
            std::vector<float> pr;
            ok = ds->draft(0, Dspark::kBlock - 1, t, pr, err);
        }
        ok = ok && reset(err);
        o.expert_cache = sw;
        prof_cnt = keep;
        st = st0;
        for (Layer& l : L) {
            l.ram = nullptr;
            l.experts = nullptr;
            std::fill(l.ram_of.begin(), l.ram_of.end(), -1);
            l.n_copy = 0;
            l.use.assign((size_t) g.n_expert, 0.0f);
        }
        for (Gpu& gp : G) {
            CK(cudaFreeHost(gp.h_dummy));
            gp.h_dummy = nullptr;
            gp.n_streamed = gp.n_experts = 0;
        }
        if (o.verbose) std::fprintf(stderr, "the warm-up: %zu tokens in %.1f s\n", ids.size(), now_s() - t0);
        return ok;
    }
    int warm_n = 256;   // (the warm-up's tokens, at most a chunk)

    bool load(const Options& opt, std::string& err) {
        const double t0 = now_s();
        o = opt;
        if (!pk.load(o.pack, err)) return false;
        g = pk.g;
        if (o.log_floor >= 1) g.log_floor = o.log_floor;
        NL = o.layers > 0 ? std::min(o.layers, g.n_layer) : g.n_layer;
        max_keys = (size_t) std::max<int64_t>(g.n_swa, o.max_ctx);
        Tc = o.prefill_chunk > 1 ? (int) std::min<int64_t>(o.prefill_chunk, o.max_ctx) : 0;
        retune();
        kStage = (int) env_d("IK2_HIP_STAGE", 24, 1, 384);
        kTsMax = (int) env_d("IK2_HIP_TS_MAX", 128, 16, 1 << 16);
        kS_n = (size_t) env_d("IK2_HIP_S_FLOOR_MI", 44, 1, 4096) << 20;
        warm_n = (int) env_d("IK2_HIP_WARM_TOKENS", 256, 2, 1 << 20);
        o.rewind = (int) std::max<int64_t>(0, std::min<int64_t>(o.rewind, o.max_ctx));
        for (int64_t& x : snap_pos) x = -1;
        int ndev = 0;
        CK(cudaGetDeviceCount(&ndev));
        std::vector<int> devs = o.gpus;
        if (devs.empty())
            for (int i = 0; i < ndev; ++i) devs.push_back(i);
        if (devs.empty()) {
            err = "no GPU";
            return false;
        }
        G.resize(devs.size());
        for (size_t i = 0; i < devs.size(); ++i) {
            if (devs[i] < 0 || devs[i] >= ndev) {
                err = "GPU " + std::to_string(devs[i]) + ": not there (" + std::to_string(ndev) + " visible)";
                return false;
            }
            G[i].dev = devs[i];
            CK(cudaSetDevice(devs[i]));
            size_t f_pre = 0;
            CK(cudaMemGetInfo(&f_pre, &G[i].total));
            CK(cudaStreamCreateWithFlags(&G[i].s, cudaStreamNonBlocking));
            CK(cudaMemGetInfo(&G[i].free0, &G[i].total));
            if (!granule(G[i], err)) return false;
            if (std::getenv("IK2_HIP_VRAM_LOG"))
                std::fprintf(stderr, "VRAM GPU %d at the start: free %.1f MiB, %.1f MiB after its first stream; allocations "
                             "over 1 MiB rounded to %zu KiB\n", G[i].dev, f_pre / 1048576.0, G[i].free0 / 1048576.0,
                             G[i].gran >> 10);
        }
        // the expert profile (maya-ds4's: <pack>/expert-profile.bin unless --expert-profile names one, "none" none):
        // read when it is there and whole, else the counts start from nothing - written back either way
        // (save_profile).  A pack from before the name (its default.prof) is read in its place, once
        if (o.profile.empty()) o.profile = o.pack + "/expert-profile.bin";
        if (o.profile == "none") o.profile.clear();
        prof_in.assign((size_t) g.n_layer * g.n_expert, 0u);
        if (!o.profile.empty()) {
            std::ifstream f(o.profile, std::ios::binary);
            if (!f && o.profile == o.pack + "/expert-profile.bin") f.open(o.pack + "/default.prof", std::ios::binary);
            std::vector<uint32_t> p(prof_in.size());
            if (f && f.read((char*) p.data(), (std::streamsize) (p.size() * 4))) prof_in = p;
        }
        prof_cnt.assign((size_t) g.n_layer * g.n_expert, 0);
        h_rl.assign((size_t) g.n_expert, 0.0f);
        L.resize((size_t) NL);
        if (!split(err)) return false;
        // the embedding table in RAM (a row a token decoded on the host)
        {
            const Tensor* t = need("token_embd.weight", err);
            if (!t) return false;
            tok_type = type_id(t->type);
            if ((tok_type != 13 && tok_type != 14 && tok_type != kF32) || t->n(0) != g.n_embd || t->n(1) != g.n_vocab) {
                err = "token_embd.weight: " + t->type + ", Q5_K / Q6_K / F32 rows of n_embd expected";
                return false;
            }
            tok_row = (size_t) t->bytes / (size_t) g.n_vocab;
            tok_embd.assign(pk.data(*t), pk.data(*t) + t->bytes);
        }
        // each GPU's layers on a thread of its own
        std::vector<std::string> errs(G.size());
        std::vector<char> oks(G.size(), 1);
        std::vector<std::thread> th;
        for (size_t gi = 0; gi < G.size(); ++gi)
            th.emplace_back([&, gi] {
                Gpu& gp = G[gi];
                std::string& e = errs[gi];
                auto run = [&]() -> bool {
                    std::string& err = e;
                    CK(cudaSetDevice(gp.dev));
                    // (the load's pinned staging at least: copies of 64 MiB already run at 99% of 1 GiB's, 56.3 against
                    //  56.9 GB/s, and the load reads its files at about 1 GB/s - measured, docs/INKLING2_AMD.md)
                    gp.stage_bytes = (size_t) 256 << 20;
                    for (int il = gp.l0; il < gp.l1; ++il) gp.stage_bytes = std::max(gp.stage_bytes, L[(size_t) il].blob);
                    CK(cudaMallocHost((void**) &gp.stage, gp.stage_bytes));
                    gp.note("after the load's pinned stage");
                    if (!scratch(gp, gi + 1 == G.size(), e)) return false;
                    gp.note("after the decode scratch");
                    if (!chunk_scratch(gp, e)) return false;
                    gp.note("after the prompt path's scratch");
                    if (gi == 0 && !f32(gp, "token_embd_norm.weight", tok_norm, e, g.n_embd)) return false;
                    if (gi + 1 == G.size() &&
                        (!mat(gp, "output.weight", head, e, g.n_embd, g.n_vocab) ||
                         !f32(gp, "output_norm.weight", out_norm, e, g.n_embd) ||
                         !f32(gp, "output_res_query.weight", out_res, e, g.n_embd)))
                        return false;
                    gp.note("after the head");
                    for (int il = gp.l0; il < gp.l1; ++il)
                        if (!load_dense(L[(size_t) il], gp, e)) return false;
                    gp.note("after its layers without experts");
                    if (o.expert_cache) CK(cudaStreamCreateWithFlags(&gp.s3, cudaStreamNonBlocking));   // (the mover's)
                    return true;
                };
                oks[gi] = run() ? 1 : 0;
            });
        for (auto& x : th) x.join();
        for (size_t gi = 0; gi < G.size(); ++gi)
            if (!oks[gi]) {
                err = errs[gi];
                return false;
            }
        CK(cudaSetDevice(G[0].dev));
        const size_t E = (size_t) g.n_embd;
        CK(cudaMallocHost((void**) &h_x, E * 4));
        CK(cudaMallocHost((void**) &h_state, (1 + k::kMaxStates) * E * 4));
        CK(cudaMallocHost((void**) &h_w, 64));
        CK(cudaMallocHost((void**) &h_wres, 64));
        CK(cudaMallocHost((void**) &h_ids, 64));
        CK(cudaMallocHost((void**) &h_xf, E * 4));
        CK(cudaMallocHost((void**) &h_cpu, E * 4));
        CK(cudaMallocHost((void**) &h_top, 8));
        CK(cudaMallocHost((void**) &h_big, (size_t) g.n_vocab * 4));
        CK(cudaMallocHost((void**) &h_plan, 128));
        lane.reset(new hy4::Lane(o.cpu_threads > 0 ? o.cpu_threads - 1 : -1));   // (workers besides the caller)
        if (!fast_init(err)) return false;
        notes("after the host-free step's buffers");
        seqs.assign((size_t) std::max(1, o.seqs), Seq{});
        for (Seq& q : seqs) {
            q.L.resize(L.size());
            for (int64_t& x : q.snap_pos) x = -1;
        }
        if (seqs.size() > 1) {   // (--parallel: the other conversations' state, before the experts take the room)
            if (!fast_ready) {
                err = "--parallel needs the host-free step (IK2_HIP_OLD_STEP is set)";
                return false;
            }
            for (Gpu& gp : G) {
                CK(cudaSetDevice(gp.dev));
                if (!seq_alloc(gp, err)) return false;
            }
            seq_save(seqs[0]);
            notes("after the other conversations' state");
        }
        if (Tc > 0) {
            for (size_t gi = 0; gi < G.size(); ++gi) {
                slots.emplace_back(new Slot);
                CK(cudaMallocHost((void**) &slots.back()->h, (size_t) (1 + n_states()) * Tc * E * 4));
            }
        }
        {   // the drafter (its load's own temporaries fit here, before the experts take the room)
            std::string e;
            if (!dspark_init(e)) std::fprintf(stderr, "maya-ik2: no drafter (%s): decoding without drafts\n", e.c_str());
        }
        if (ds && seqs.size() > 1 && !ds->add_states((int) seqs.size(), err)) return false;
        notes("after the drafter");
        if (!warm_up(err)) return false;
        notes("after the warm-up");
        if (!place(err)) return false;
        if (o.verbose)
            for (const std::string& s : info_lines()) std::fprintf(stderr, "%s\n", s.c_str());
        {   // each GPU's experts on a thread of its own
            std::vector<std::thread> te;
            for (size_t gi = 0; gi < G.size(); ++gi)
                te.emplace_back([&, gi] {
                    Gpu& gp = G[gi];
                    std::string& e = errs[gi];
                    auto run = [&]() -> bool {
                        std::string& err = e;
                        CK(cudaSetDevice(gp.dev));
                        uint8_t* at = nullptr;
                        if (!gp.alloc(at, gp.expert_alloc, err)) return false;
                        for (int il = gp.l0; il < gp.l1; ++il) {
                            Layer& l = L[(size_t) il];
                            if (!load_experts(l, gp, at, err)) return false;
                            at += l.blob * (size_t) (l.n_res + 1);
                            if (o.verbose) std::fprintf(stderr, "layer %d on GPU %d (%.0f s)\n", il, gp.dev, now_s() - t0);
                        }
                        gp.note("after its experts");
                        CK(cudaFreeHost(gp.stage));
                        gp.stage = nullptr;
                        bool files = false;   // (--mmap-experts: the prompt path's gather buffer, kStage blobs pinned)
                        for (int il = gp.l0; il < gp.l1; ++il) files = files || !L[(size_t) il].cex.fgu.empty();
                        if (files && Tc > 0) {
                            CK(cudaMallocHost((void**) &gp.h_fst, (size_t) kStage * gp.blob_max));
                            CK(cudaEventCreateWithFlags(&gp.ev_fst, cudaEventDisableTiming));
                        }
                        return true;
                    };
                    oks[gi] = run() ? 1 : 0;
                });
            for (auto& x : te) x.join();
            for (size_t gi = 0; gi < G.size(); ++gi)
                if (!oks[gi]) {
                    err = errs[gi];
                    return false;
                }
        }
        if (std::getenv("IK2_HIP_VERIFY_LOAD") && !verify(err)) return false;
        {   // the RAM experts registered with the driver, a thread a GPU's layers
            const double tr = now_s();
            std::vector<std::thread> rt;
            for (size_t gi = 0; gi < G.size(); ++gi)
                rt.emplace_back([&, gi] {
                    cudaSetDevice(G[gi].dev);
                    for (int il = G[gi].l0; il < G[gi].l1; ++il) {
                        Layer& l = L[(size_t) il];
                        if (!l.ram) continue;
                        if (cudaHostRegister(l.ram, l.ram_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess)
                            l.ram_reg = true;
                        else if (mlock(l.ram, l.ram_bytes) != 0)
                            ram_unlocked = true;
                        if (!l.ram_reg) ram_unreg = true;
                        // (each registered page's first copy to the device is several times slower than the later
                        //  ones: paid here, not by the first prompt)
                        // (on the side stream, as the prompt path copies them)
                        if (l.ram_reg && Tc > 0) {
                            for (size_t o2 = 0; o2 < l.ram_bytes; o2 += (size_t) kStage * l.blob)
                                cudaMemcpyAsync(G[gi].xstage[0], l.ram + o2, std::min((size_t) kStage * l.blob, l.ram_bytes - o2),
                                                cudaMemcpyHostToDevice, G[gi].s2);
                            cudaStreamSynchronize(G[gi].s2);
                        }
                    }
                });
            for (auto& x : rt) x.join();
            if (o.verbose) std::fprintf(stderr, "RAM experts registered and streamed once in %.1f s\n", now_s() - tr);
        }
        notes("after the RAM experts' registration");
        if (o.expert_cache && st.ram_gb + st.file_gb > 0 && !ram_unreg.load())
            for (size_t gi = 0; gi < G.size(); ++gi) {
                CK(cudaSetDevice(G[gi].dev));
                G[gi].blob_max = std::max<size_t>(G[gi].blob_max, 1);
                CK(cudaMallocHost((void**) &G[gi].h_swap, G[gi].blob_max * 16));   // (16: swap_poll's most moves a round)
                G[gi].sw_thread = std::thread([this, gi] { swap_run(gi); });
            }
        // (at least one move; above 16 swap_poll's bounce cap takes it)
        if (const char* e = std::getenv("IK2_HIP_SWAP_BATCH")) swap_batch = std::max(1, std::min(64, std::atoi(e)));
        if (const char* e = std::getenv("IK2_HIP_SWAP_BURST_MS")) burst_s = std::atof(e) * 1e-3;
        notes("after the cache's movers");
        st.ram_locked = st.ram_gb > 0 && !ram_unlocked.load();
        st.ram_pinned = st.ram_gb > 0 && !ram_unreg.load();
        if (Tc > 0 && st.ram_gb > 0 && !st.ram_pinned) {
            std::fprintf(stderr, "maya-ik2: the RAM experts could not be registered with the GPUs' driver: prompts "
                                 "run token by token\n");
            Tc = 0;
        }
        if (!warm_up(err, false)) return false;
        notes("after the second warm-up");
        for (const Gpu& gp : G) st.vram_gb += (double) gp.used / 1e9;
        st.load_s = now_s() - t0;
        return true;
    }

    // ---- the forward -----------------------------------------------------------------------------------------------
    bool tap(const char* name, int il, Gpu& gp, const float* p, size_t n, std::string& err) {
        if (!tapf) return true;
        if (tap_T > 1) {   // (a chunk's rows)
            tap_vec.resize(n);
            CK(cudaMemcpyAsync(tap_vec.data(), p, n * 4, cudaMemcpyDeviceToHost, gp.s));
            CK(cudaStreamSynchronize(gp.s));
            tapf(name, il, tap_vec.data(), n, tap_p, tap_T);
            return true;
        }
        if (n > (size_t) g.n_vocab) return true;
        CK(cudaMemcpyAsync(h_big, p, n * 4, cudaMemcpyDeviceToHost, gp.s));
        CK(cudaStreamSynchronize(gp.s));
        tapf(name, il, h_big, n, tap_p, 1);
        return true;
    }
    // y = M x (one activation): the F32 matrices' own GEMV, ds4_iq's over the int8 input for the block types
    bool mm(Gpu& gp, const Mat& M, const float* x, float* y) {
        if (M.type == kF32) {
            hk::gemv_f32w((const float*) M.w, (size_t) M.k, M.k, M.n, x, (size_t) M.k, 1, y, (size_t) M.n, 1.0f, gp.s);
            return true;
        }
        return mm_multi(gp, {{&M, y}}, x);
    }
    struct Out {
        const Mat* m;
        float* y;
    };
    // several block-type matrices over one input: its int8 rows made once, the GEMVs in one launch
    bool mm_multi(Gpu& gp, std::initializer_list<Out> outs, const float* x) {
        const int kk = outs.begin()->m->k;
        hk::q8_rows(x, (size_t) kk, 1, kk, gp.xq, gp.xd, gp.s);
        iq::MvJob jobs[8];
        int n = 0;
        for (const Out& o2 : outs) {
            if (o2.m->type == kF32 || o2.m->k != kk || n >= 8) return false;
            iq::MvJob& j = jobs[n++];
            j.w = o2.m->w;
            j.ld = o2.m->ld;
            j.type = o2.m->type;
            j.k = kk;
            j.n = o2.m->n;
            j.ncol = 1;
            j.xq = gp.xq;
            j.xd = gp.xd;
            j.y = o2.y;
            j.ycs = (size_t) o2.m->n;
        }
        return iq::mv(jobs, n, gp.s);
    }

    // The routed experts of the token in gp.xf (the picks in gp.ids, their weights in gp.rw) -> gp.y
    bool experts(Layer& l, Gpu& gp, std::string& err) {
        const int E = g.n_embd, ff = g.n_ff_exp, nu = g.n_used;
        cudaStream_t s = gp.s;
        const bool lane_on = l.n_res < g.n_expert;
        CK(cudaMemcpyAsync(h_ids, gp.ids, sizeof(int) * nu, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_w, gp.rw, sizeof(float) * nu, cudaMemcpyDeviceToHost, s));
        static const bool check = std::getenv("IK2_HIP_EXPERT_CHECK") != nullptr;   // (tests: see below)
        if (lane_on || check) CK(cudaMemcpyAsync(h_xf, gp.xf, sizeof(float) * E, cudaMemcpyDeviceToHost, s));
        if (tapf || pick_hook) CK(cudaMemcpyAsync(h_rl.data(), gp.rl, sizeof(float) * g.n_expert, cudaMemcpyDeviceToHost, s));
        CK(sync_timed(s));
        if (tapf) {
            std::vector<float> sc((size_t) g.n_expert);
            for (int e = 0; e < g.n_expert; ++e)
                sc[(size_t) e] = 1.0f / (1.0f + std::exp(-h_rl[(size_t) e])) + (l.h_bias.empty() ? 0.0f : l.h_bias[(size_t) e]);
            tapf("ffn_moe_scores", l.il, sc.data(), sc.size(), tap_p, 1);
            std::vector<int32_t> pw((size_t) 2 * nu);
            std::memcpy(pw.data(), h_ids, sizeof(int) * nu);
            std::memcpy(pw.data() + nu, h_w, sizeof(float) * nu);
            tapf("ffn_moe_topk", l.il, pw.data(), (size_t) nu, tap_p, 1);
        }
        if (pick_hook && pick_hook(l.il, h_ids, nu, tap_p, 1)) {   // (the hook's picks: the router's weights for them)
            float sum = 0.0f;
            for (int j = 0; j < nu; ++j) {
                if (h_ids[j] < 0 || h_ids[j] >= g.n_expert) {
                    err = "layer " + std::to_string(l.il) + ": the pick hook named expert " + std::to_string(h_ids[j]);
                    return false;
                }
                h_w[j] = 1.0f / (1.0f + std::exp(-h_rl[(size_t) h_ids[j]]));
                sum += h_w[j];
            }
            for (int j = 0; j < nu; ++j) h_w[j] = h_w[j] / (sum + 1e-20f) * g.w_scale;
        }
        int m = 0, slots[16];
        lane_entries.clear();
        uint32_t* cnt = prof_cnt.data() + (size_t) l.il * g.n_expert;
        for (int j = 0; j < nu; ++j) {
            const int e = h_ids[j];
            if (e < 0 || e >= g.n_expert) {
                err = "layer " + std::to_string(l.il) + ": the router picked expert " + std::to_string(e);
                return false;
            }
            ++cnt[e];
            l.use[(size_t) e] += 1.0f;
            if (l.slot_of[(size_t) e] >= 0) {
                slots[m] = l.slot_of[(size_t) e];
                h_wres[m++] = h_w[j];
            } else {
                lane_entries.push_back({l.ram_of[(size_t) e], h_w[j]});
            }
        }
        bool ok = true;
#if defined(IK2_HIP_IQ_SITU)
        // ds4_iq's token kernels with the SITU activation: the plan a blob a pick (0: the CPU lane's, its weight 0)
        const int rows = nu;
        if (m > 0) {
            for (int j = 0; j < nu; ++j) {
                const int sl = l.slot_of[(size_t) h_ids[j]];
                h_plan[j] = sl >= 0 ? (unsigned long long) (uintptr_t) (l.experts + (size_t) sl * l.blob) : 0ull;
                h_wres[j] = sl >= 0 ? h_w[j] : 0.0f;
            }
            CK(cudaMemcpyAsync(gp.plan, h_plan, sizeof(unsigned long long) * nu, cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(gp.wres, h_wres, sizeof(float) * nu, cudaMemcpyHostToDevice, s));
            hk::q8_rows(gp.xf, (size_t) E, 1, E, gp.xq, gp.xd, s);
            iq::Args a;
            a.plan_ptr = gp.plan;
            a.tr[0] = l.tr[0];
            a.tr[1] = l.tr[1];
            a.tr[2] = l.tr[2];
            a.n_embd = E;
            a.n_ff = ff;
            a.xq = gp.xq;
            a.xd = gp.xd;
            a.h = gp.eh;
            a.part = gp.part;
            a.tg = l.tg;
            a.td = l.td;
            a.act = iq::kActSitu;
            a.situ_beta = g.situ_beta;
            a.situ_lb = g.situ_lb;
            ok = iq::gate_up(a, nu, 1, s);
            hk::q8_rows(gp.eh, (size_t) ff, nu, ff, gp.ehq, gp.ehd, s);
            a.hq = gp.ehq;
            a.hd = gp.ehd;
            ok = ok && iq::down(a, nu, 1, s);
        }
        k::wsum(gp.part, m > 0 ? rows : 0, E, gp.wres, gp.y, s);
#else
        if (m > 0) {
            // (until ds4_iq's token kernels take the SITU activation: an expert as two GEMVs over int8 inputs - its
            //  fused gate | up rows, then down - with the activation between them)
            hk::q8_rows(gp.xf, (size_t) E, 1, E, gp.xq, gp.xd, s);
            iq::MvJob jobs[16];
            for (int r = 0; r < m; ++r) {
                iq::MvJob& j = jobs[r];
                j = iq::MvJob();
                j.w = l.experts + (size_t) slots[r] * l.blob + l.tr[0];
                j.ld = iq::row_bytes(l.tg, E);
                j.type = l.tg;
                j.k = E;
                j.n = 2 * ff;
                j.xq = gp.xq;
                j.xd = gp.xd;
                j.y = gp.gu + (size_t) r * 2 * ff;
                j.ycs = (size_t) 2 * ff;
            }
            ok = iq::mv(jobs, m, s);
            k::situ(gp.gu, m, ff, g.situ_beta, g.situ_lb, gp.eh, s);
            hk::q8_rows(gp.eh, (size_t) ff, m, ff, gp.ehq, gp.ehd, s);
            for (int r = 0; r < m; ++r) {
                iq::MvJob& j = jobs[r];
                j = iq::MvJob();
                j.w = l.experts + (size_t) slots[r] * l.blob + l.tr[2];
                j.ld = iq::row_bytes(l.td, ff);
                j.type = l.td;
                j.k = ff;
                j.n = E;
                j.xq = gp.ehq + (size_t) r * ff;
                j.xd = gp.ehd + (size_t) r * ff / 32;
                j.y = gp.part + (size_t) r * E;
                j.ycs = (size_t) E;
            }
            ok = ok && iq::mv(jobs, m, s);
            CK(cudaMemcpyAsync(gp.wres, h_wres, sizeof(float) * m, cudaMemcpyHostToDevice, s));
        }
        k::wsum(gp.part, m, E, gp.wres, gp.y, s);
#endif
        // IK2_HIP_EXPERT_CHECK=1 (tests): every pick the GPU ran against the CPU lane's arithmetic on the file's bytes
        if (check && m > 0) {
            std::vector<float> hp((size_t) nu * E), ref((size_t) E);
            CK(cudaMemcpyAsync(hp.data(), gp.part, sizeof(float) * nu * E, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            std::vector<uint8_t> tmp(l.blob);
            CpuExperts cx;
            cx.beta = g.situ_beta;
            cx.lb = g.situ_lb;
            if (!cx.init(l.tg, l.td, E, ff, err)) return false;
            cx.base = tmp.data();
            cx.blob = l.blob;
            cx.tr_down = l.tr[2];
            int r = 0;
            for (int j = 0; j < nu; ++j) {
                if (l.slot_of[(size_t) h_ids[j]] < 0) continue;
#if defined(IK2_HIP_IQ_SITU)
                const float* mine = hp.data() + (size_t) j * E;
#else
                const float* mine = hp.data() + (size_t) r * E;
#endif
                ++r;
                blob_copy(l, h_ids[j], tmp.data());
                cpu_experts(*lane, cx, {{0, 1.0f}}, h_xf, ref.data());
                double dot = 0, a2 = 0, b2 = 0;
                for (int i = 0; i < E; ++i) {
                    dot += (double) mine[i] * ref[(size_t) i];
                    a2 += (double) mine[i] * mine[i];
                    b2 += (double) ref[(size_t) i] * ref[(size_t) i];
                }
                const double c = dot / (std::sqrt(a2 * b2) + 1e-30);
                if (!(c > 0.99))
                    std::fprintf(stderr, "EXPERT_CHECK layer %d expert %d (slot %d): cos %.4f, |gpu| %.3g, |cpu| %.3g\n", l.il,
                                 h_ids[j], l.slot_of[(size_t) h_ids[j]], c, std::sqrt(a2), std::sqrt(b2));
            }
        }
        if (!lane_entries.empty()) {   // (the GPU's experts run meanwhile)
            const double tc = now_s();
            cpu_experts(*lane, l.cex, lane_entries, h_xf, h_cpu);
            st.cpu_ms += (now_s() - tc) * 1e3;
            st.cpu_experts += (int64_t) lane_entries.size();
            CK(cudaMemcpyAsync(gp.cpo, h_cpu, sizeof(float) * E, cudaMemcpyHostToDevice, s));
            k::add(gp.y, gp.cpo, (size_t) E, s);
            CK(sync_timed(s));   // (h_cpu and h_wres are the next layer's too)
        } else if (m > 0) {
            CK(sync_timed(s));
        }
        st.experts += nu;
        if (!ok) err = "layer " + std::to_string(l.il) + ": an expert kernel did not take its shapes";
        return ok;
    }

    bool layer(Layer& l, Gpu& gp, int64_t p, int& nb, std::string& err) {
        const int E = g.n_embd, H = g.n_head, Hkv = g.n_head_kv, D = g.head_dim, HD = H * D, kvd = Hkv * D;
        cudaStream_t s = gp.s;
        bool ok = true;
        mark(gp, -1);
        // the attention's inputs
        const float* src = gp.cur;
        const bool fused = l.il > 0;   // (the mix and the norms in a launch)
        if (fused) k::mix_rms(gp.blocks, nb, gp.cur, E, l.attn_res, l.swa ? 1 : 3, l.swa ? l.attn_norm : l.q_norm, l.k_norm,
                              l.v_norm, g.eps, gp.xn, s);
        const float *xq = gp.xn, *xk = gp.xn, *xv = gp.xn;
        if (l.swa) {
            if (!fused) k::rms(src, (size_t) E, 1, E, l.attn_norm, g.eps, gp.xn, (size_t) E, s);
            ok = mm_multi(gp, {{&l.q, gp.q}, {&l.gate, gp.gate}, {&l.rel, gp.rel}, {&l.wk, gp.ky}, {&l.wv, gp.vy}}, xq);
        } else {
            const size_t st1 = l.il > 0 ? (size_t) E : 0;
            if (!fused) {
                k::rms(src, (size_t) E, 1, E, l.q_norm, g.eps, gp.xn, (size_t) E, s);
                k::rms(src + st1, (size_t) E, 1, E, l.k_norm, g.eps, gp.xn + E, (size_t) E, s);
                k::rms(src + 2 * st1, (size_t) E, 1, E, l.v_norm, g.eps, gp.xn + 2 * E, (size_t) E, s);
            }
            xk = gp.xn + E;
            xv = gp.xn + 2 * E;
            ok = mm_multi(gp, {{&l.q, gp.q}, {&l.gate, gp.gate}, {&l.rel, gp.rel}}, xq) && mm(gp, l.wk, xk, gp.ky) &&
                 mm(gp, l.wv, xv, gp.vy) && mm(gp, l.k_proj, xk, gp.dk) && mm(gp, l.v_proj, xv, gp.dv);
        }
        ok = ok && mm(gp, l.sink, xq, gp.sink);
        if (!ok) {
            err = "layer " + std::to_string(l.il) + ": an attention GEMV did not take its shapes";
            return false;
        }
        mark(gp, 0);
        const float tau = l.swa ? 1.0f : 1.0f + g.log_alpha * std::log(std::max(1.0f, (float) ((double) (p + 1) / g.log_floor)));
        // (the K / V convs, K's clip, V's norm and the cache rows in a launch; the Q and R clips in another)
        k::kv_finish(gp.ky, gp.vy, l.hist_k, l.hist_v, l.k_conv, l.v_conv, l.swa ? nullptr : gp.dk, l.swa ? nullptr : gp.dv,
                     Hkv, D, g.qk_clip, g.eps, l.K, l.V, l.ring, p, s);
        k::clip2(gp.q, H, D, l.q_clip, g.qk_clip, gp.rel, g.d_rel, l.rel_clip, g.rel_clip, tau, s);
        if (!tap("Qcur", l.il, gp, gp.q, (size_t) HD, err) || !tap("Kcur", l.il, gp, gp.ky, (size_t) kvd, err) ||
            !tap("Vcur", l.il, gp, gp.vy, (size_t) kvd, err) || !tap("Rcur", l.il, gp, gp.rel, (size_t) H * g.d_rel, err))
            return false;
        const int64_t lo = l.swa ? std::max<int64_t>(0, p - (g.n_swa - 1)) : 0;
        mark(gp, 1);
        // (the keys split over tiles, the output gate in the combine: IK2_HIP_ATTN_OLD=1 for the earlier launches)
        if (!attn_old) {
            k::attention_split(gp.q, gp.rel, gp.sink, H, Hkv, D, g.d_rel, l.rel_bias, l.swa ? g.n_swa : g.rel_extent, l.K, l.V,
                               l.ring, p, lo, 1.0f / std::sqrt((float) D), gp.S, tapf ? nullptr : gp.gate, gp.ao, s);
            if (!tap("kqv_out", l.il, gp, gp.ao, (size_t) HD, err)) return false;
            mark(gp, 2);
            if (tapf) k::sigmoid_mul(gp.ao, gp.gate, (size_t) HD, s);
        } else {
            k::attention(gp.q, gp.rel, gp.sink, H, Hkv, D, g.d_rel, l.rel_bias, l.swa ? g.n_swa : g.rel_extent, l.K, l.V,
                         l.ring, p, lo, 1.0f / std::sqrt((float) D), gp.S, gp.ao, s);
            if (!tap("kqv_out", l.il, gp, gp.ao, (size_t) HD, err)) return false;
            mark(gp, 2);
            k::sigmoid_mul(gp.ao, gp.gate, (size_t) HD, s);
        }
        if (!mm(gp, l.wo, gp.ao, gp.o)) {
            err = "layer " + std::to_string(l.il) + ": the attention's output GEMV did not take its shapes";
            return false;
        }
        const bool newstream = l.il % g.res_block == 0;
        if (newstream) k::conv(gp.o, l.hist_a, l.a_conv, E, nullptr, 1, s);
        else k::conv_add(gp.o, l.hist_a, l.a_conv, E, gp.cur, s);   // (the conv and the residual in a launch)
        if (!tap("attn_out", l.il, gp, gp.o, (size_t) E, err)) return false;
        // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
        if (newstream) {
            CK(cudaMemcpyAsync(gp.blocks + (size_t) nb * E, gp.cur, sizeof(float) * E, cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(gp.cur, gp.o, sizeof(float) * E, cudaMemcpyDeviceToDevice, s));
            ++nb;
        }
        if (!tap("ffn_inp", l.il, gp, gp.cur, (size_t) E, err)) return false;
        mark(gp, 3);
        // the routed experts
        k::mix_rms(gp.blocks, nb, gp.cur, E, l.ffn_res, 1, l.ffn_norm, nullptr, nullptr, g.eps, gp.xf, s);
        mm(gp, l.router, gp.xf, gp.rl);
        hk::router(gp.rl, 1, g.n_expert, l.router_b, g.n_used, true, g.w_scale, gp.ids, gp.rw, s);
        mark(gp, 4);
        if (!experts(l, gp, err)) return false;
        mark(gp, 5);
        k::conv_add(gp.y, l.hist_f, l.f_conv, E, gp.cur, s);
        if (!tap("ffn_out", l.il, gp, gp.y, (size_t) E, err)) return false;
        mark(gp, 6);
        return tap("l_out", l.il, gp, gp.cur, (size_t) E, err);
    }

    bool step(int id, int64_t p, float* logits_out, std::string& err) {
        const int E = g.n_embd;
        tap_p = p;
        tap_T = 1;
        if (fast_on()) return step_fast(id, p, logits_out, err);
        if (!taps_flush(err)) return false;
        const double ts0 = now_s();
        struct Tm {
            Impl* m;
            double t0;
            ~Tm() {
                m->step_s += now_s() - t0;
                ++m->step_n;
            }
        } tm{this, ts0};
        if (id < 0 || id >= g.n_vocab) {
            err = "token " + std::to_string(id) + " is outside the vocabulary";
            return false;
        }
        const uint8_t* row = tok_embd.data() + (size_t) id * tok_row;
        if (tok_type == 13) dequantize_row_q5_K(row, h_x, E);
        else if (tok_type == 14) dequantize_row_q6_K(row, h_x, E);
        else std::memcpy(h_x, row, (size_t) E * 4);
        int nb = 0;
        use_decay(tn.tok_decay);
        if (!swap_poll(err)) return false;
        hw = std::max(hw, p + 1);
        const int slot = p % tn.snap_every == 0 || snap_next ? snap_alloc(p) : -1;
        snap_next = false;
        for (size_t gi = 0; gi < G.size(); ++gi) {
            Gpu& gp = G[gi];
            CK(cudaSetDevice(gp.dev));
            if (slot >= 0 && !snap_copy(gp, slot, true, err)) return false;
            if (gi == 0) {
                CK(cudaMemcpyAsync(gp.o, h_x, sizeof(float) * E, cudaMemcpyHostToDevice, gp.s));
                k::rms(gp.o, (size_t) E, 1, E, tok_norm, g.eps, gp.cur, (size_t) E, gp.s);
            } else {   // (the hand-off: the stream and the saved ones)
                CK(cudaMemcpyAsync(gp.cur, h_state, sizeof(float) * E, cudaMemcpyHostToDevice, gp.s));
                if (nb > 0)
                    CK(cudaMemcpyAsync(gp.blocks, h_state + E, sizeof(float) * E * nb, cudaMemcpyHostToDevice, gp.s));
            }
            for (int il = gp.l0; il < gp.l1; ++il)
                if (!layer(L[(size_t) il], gp, p, nb, err)) return false;
            if (gi + 1 < G.size()) {
                CK(cudaMemcpyAsync(h_state, gp.cur, sizeof(float) * E, cudaMemcpyDeviceToHost, gp.s));
                if (nb > 0)
                    CK(cudaMemcpyAsync(h_state + E, gp.blocks, sizeof(float) * E * nb, cudaMemcpyDeviceToHost, gp.s));
                CK(sync_timed(gp.s));
            }
        }
        if (!head_out(nb, logits_out, err)) return false;
        ++st.tokens;
        return true;
    }
    // ---- the host-free decode step ------------------------------------------------------------------------------
    // A token's layers are queued on every GPU at once, each by its own thread, with nothing in them that waits for
    // the host: the picks, the plan and the VRAM / RAM map stay on the device; a layer publishes its routing (the FFN
    // input, the picks, their weights) into host memory the device writes itself and waits - on the device - for the
    // CPU lane's sum of the picks that are in RAM, which the lane's thread writes back with a flag; a GPU hands the
    // stream to the next the same way.  The host's one wait a token is the last GPU's.
    struct Worker {
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        int job = 0, done = 0;   // (counts)
        bool quit = false, ok = true;
        std::string err;
    };
    std::deque<Worker> workers;
    std::thread lane_th;
    std::atomic<unsigned> lane_seq{0}, lane_done{0};
    std::atomic<bool> lane_quit{false};
    unsigned seq = 0;
    int64_t job_p = 0;
    int job_slot = -1;
    int job_T = 1;                  // (1: a token's step; more: a window's rows)
    // (--parallel) a multi step's rows: each its own conversation's next token, at that conversation's end - its
    // state in seqs[seq] (the bound one parked there too), its snapshot slot (-1: none)
    struct MRow {
        int seq = 0;
        int64_t p = 0;
        int slot = -1;
    };
    bool job_multi = false;
    MRow mrows[16];
    const int* job_ids = nullptr;
    float* job_rows = nullptr;      // (a window's logits out, [T][vocab], or null)
    int win_tops[16];
    int64_t win_p0 = -1;            // the last window's first position and rows (keep cuts back to them); -1: none
    int win_T = 0;
    // a window's rows at most: a round takes 8 (the anchor and 7 drafts, mtp_config.block_size 8); the 9th only
    // --window-test's 9-row window uses
    static constexpr int kWin = 9;
    // the drafter (Options::dspark_index): its taps out of every step and prompt chunk, fed once the rows are kept
    std::unique_ptr<Dspark> ds;
    bool taps_on = false;
    std::vector<int> tap_layers, tap_of;   // the layers whose output stream is tap k (the last: the head's input); a layer's tap or -1
    float* m_taps = nullptr;               // (mapped host memory: the other GPUs' taps of a step, [taps - 1][T][E])
    std::vector<float*> p_taps;            // (pinned: a prompt chunk's, [taps - 1][Tc][E], a buffer a chunk in flight)
    int tap_buf = 0, tap_pending = 0;      // the last step's buffer, its rows not fed yet (keep() cuts them)
    // the last step's rows whose routings are not counted yet (the profile, the cache's counts): the rows keep()
    // leaves are counted, the drafts the model turned down are not
    int cnt_pending = 0;
    void counts_flush() {
        const int n = cnt_pending, nu = g.n_used;
        cnt_pending = 0;
        if (n <= 0) return;
        for (Layer& l : L) {
            uint32_t* cnt = prof_cnt.data() + (size_t) l.il * g.n_expert;
            for (int t = 0; t < n; ++t)
                for (int j = 0; j < nu; ++j) {
                    const int e = l.picks[(size_t) t * 16 + j];
                    if (e < 0 || e >= g.n_expert) continue;
                    ++cnt[e];
                    l.use[(size_t) e] += 1.0f;
                }
        }
    }
    int64_t tap_p0 = 0;
    double ds_feed_s = 0;
    bool taps_flush(std::string& err) {
        counts_flush();
        if (tap_pending <= 0) return true;
        const int n = tap_pending;
        tap_pending = 0;
        if (!taps_on || ds->end() != tap_p0) return true;   // (the drafter is not at these rows: it rests until a rewind)
        const double t0 = now_s();
        const bool ok = ds->feed_dev(G.back().d_taps[tap_buf], tap_layers.size() * (size_t) g.n_embd, n, tap_p0, err,
                                     tn.tap_event ? (void*) G.back().ev_tap[tap_buf] : nullptr);
        ds_feed_s += now_s() - t0;
        return ok;
    }
    // the drafter back at the context's end after a rewind (it holds every position's rows up to its own end)
    void taps_rollback() {
        tap_pending = 0;
        cnt_pending = 0;
        if (ds) ds->rollback(std::min<int64_t>(ds->end(), n_end));
    }
    bool dspark_init(std::string& err) {
        if (o.dspark_index.empty() || !fast_ready) return true;
        Gpu& gp = G.back();
        CK(cudaSetDevice(gp.dev));
        DsparkOptions d;
        d.index = o.dspark_index;
        d.target_pack = o.pack;
        d.gpu = gp.dev;
        d.max_ctx = o.max_ctx;
        d.rewind = o.rewind + Tc;   // (its rings as long as the engine's: a rewind the engine can make, the drafter follows)
        d.cum_min = o.dspark_cum;
        d.output_dev = head.w;
        d.output_type = head.type;
        d.output_ld = head.ld;
        d.verbose = o.verbose;
        ds.reset(new Dspark);
        if (!ds->load(d, err)) {
            ds.reset();
            (void) cudaGetLastError();   // (a failed allocation's error is not the next launch's)
            return false;
        }
        const std::vector<int>& tl = ds->target_layers();
        const size_t E = (size_t) g.n_embd, nt = tl.size();
        if (ds->n_embd() != g.n_embd || nt < 2 || nt > 8 || tl.back() != g.n_layer) {
            err = "the drafter's taps are not this model's";
            ds.reset();
            return false;
        }
        tap_of.assign((size_t) NL, -1);
        tap_layers.clear();
        for (size_t kt = 0; kt + 1 < nt; ++kt) {   // (tap k: the stream entering layer tl[k] = the output of tl[k] - 1)
            if (tl[kt] < 1 || tl[kt] > NL) {
                err = "the drafter taps layer " + std::to_string(tl[kt]);
                ds.reset();
                return false;
            }
            tap_layers.push_back(tl[kt] - 1);
            tap_of[(size_t) tl[kt] - 1] = (int) kt;
        }
        tap_layers.push_back(NL);
        CK(cudaHostAlloc((void**) &m_taps, sizeof(float) * (nt - 1) * kWin * E, cudaHostAllocMapped | cudaHostAllocPortable));
        for (cudaEvent_t& e : gp.ev_tap) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        if (!gp.alloc(gp.d_taps[0], (size_t) kWin * nt * E, err) || !gp.alloc(gp.d_taps[1], (size_t) kWin * nt * E, err) ||
            !gp.alloc(gp.d_tapl, (nt - 1) * kWin * E, err) || !gp.alloc(gp.d_tapr, (nt - 1) * kWin * E, err))
            return false;
        if (Tc > 1) {
            const size_t T = (size_t) Tc;
            if (!gp.alloc(gp.c_taps, T * nt * E, err) || !gp.alloc(gp.c_tapl, (nt - 1) * T * E, err) ||
                !gp.alloc(gp.c_tapr, (nt - 1) * T * E, err))
                return false;
            p_taps.resize(2 * G.size() + 1, nullptr);
            for (float*& b : p_taps) CK(cudaMallocHost((void**) &b, sizeof(float) * (nt - 1) * T * E));
        }
        taps_on = true;
        return true;
    }
    float *h_xw = nullptr, *h_wtop = nullptr, *h_wlog = nullptr;
    float* job_logits = nullptr;
    unsigned *f_pub = nullptr, *f_cpu = nullptr, *f_hand = nullptr;   // (mapped host memory: a line a flag)
    static constexpr int kLine = 16;   // (16 unsigned = 64 bytes, a cache line: each flag on its own line)
    double fast_s = 0, fast_enq_s = 0, lane_work_s = 0, lane_wait_s = 0;
    int64_t fast_n = 0;
    bool fast_ready = false;
    // IK2_HIP_PHASES=1 (tests): device-side marks between a layer's phases in the host-free step (events on the
    // streams, read after the token: no extra waits), summed over the layers and tokens
    bool phases = std::getenv("IK2_HIP_PHASES") != nullptr;
    bool attn_old = std::getenv("IK2_HIP_ATTN_OLD") != nullptr;
    double ph_ms[kWin + 1][10] = {}, ph_s[kWin + 1] = {};
    int64_t ph_n[kWin + 1] = {};
    void ph(Layer& l, Gpu& gp, int k) {
        if (phases) cudaEventRecord(l.pe[k], gp.s);
    }
    bool fast_on() const { return fast_ready && !tapf && !pick_hook && !prof; }

    bool fast_init(std::string& err) {
        if (std::getenv("IK2_HIP_OLD_STEP")) return true;
        const size_t E = (size_t) g.n_embd;
        const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
        CK(cudaSetDevice(G[0].dev));
        CK(cudaHostAlloc((void**) &f_pub, sizeof(unsigned) * kLine * (size_t) NL, fl));
        CK(cudaHostAlloc((void**) &f_cpu, sizeof(unsigned) * kLine * (size_t) NL, fl));
        CK(cudaHostAlloc((void**) &f_hand, sizeof(unsigned) * kLine * G.size(), fl));
        std::memset(f_pub, 0, sizeof(unsigned) * kLine * (size_t) NL);
        std::memset(f_cpu, 0, sizeof(unsigned) * kLine * (size_t) NL);
        std::memset(f_hand, 0, sizeof(unsigned) * kLine * G.size());
        for (Layer& l : L) {
            CK(cudaSetDevice(G[(size_t) l.gpu].dev));
            CK(cudaHostAlloc((void**) &l.m_pub, sizeof(float) * (E + 16) * kWin, fl));
            CK(cudaHostAlloc((void**) &l.m_cpu, sizeof(float) * E * kWin, fl));
        }
        for (Gpu& gp : G) {
            CK(cudaSetDevice(gp.dev));
            CK(cudaHostAlloc((void**) &gp.m_hand, sizeof(float) * (1 + k::kMaxStates) * E * kWin, fl));
            if (!win_scratch(gp, err)) return false;
            if (!gp.alloc(gp.pub, E + 16, err) || !zeros(gp, gp.cnt, 4, err)) return false;
        }
        if (phases)
            for (Layer& l : L) {
                CK(cudaSetDevice(G[(size_t) l.gpu].dev));
                for (cudaEvent_t& e : l.pe) CK(cudaEventCreate(&e));
            }
        CK(cudaMallocHost((void**) &h_xw, sizeof(float) * kWin * E));
        CK(cudaMallocHost((void**) &h_wtop, sizeof(float) * 2 * kWin));
        CK(cudaMallocHost((void**) &h_wlog, sizeof(float) * kWin * (size_t) g.n_vocab));
        workers.resize(G.size());
        for (size_t gi = 0; gi < G.size(); ++gi) workers[gi].th = std::thread([this, gi] { worker_run(gi); });
        lane_th = std::thread([this] { lane_run(); });
        fast_ready = true;
        return true;
    }
    void fast_stop() {
        if (!fast_ready) return;
        lane_quit = true;
        for (Worker& w : workers) {
            {
                std::lock_guard<std::mutex> lk(w.mu);
                w.quit = true;
            }
            w.cv.notify_all();
            if (w.th.joinable()) w.th.join();
        }
        if (lane_th.joinable()) lane_th.join();
        fast_ready = false;
    }
    // the lane's thread: a token's layers in order - each layer's routing as the device publishes it, its RAM picks'
    // sum back (zeros when it has none) and the flag the device waits on
    void lane_run() {
        const int E = g.n_embd, nu = g.n_used;
        unsigned at = 0;
        std::vector<CpuEntry> ent;
        std::vector<CpuRowEntry> rent;
        for (;;) {
            double idle_t0 = 0;
            for (unsigned idle = 0; lane_seq.load(std::memory_order_acquire) == at; ++idle) {
                if (lane_quit.load()) return;
                // (between two tokens: awake; no request: asleep - tn.lane_spin_us, maya-ds4's 20 ms)
                if (idle == 0) idle_t0 = now_s();
                if (now_s() - idle_t0 < tn.lane_spin_us * 1e-6) std::this_thread::yield();
                else std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            const unsigned v = lane_seq.load();
            for (int il = 0; il < NL; ++il) {
                Layer& l = L[(size_t) il];
                const double tw = now_s();
                volatile unsigned* fp = f_pub + (size_t) il * kLine;
                for (unsigned spin = 0; *fp != v; ++spin) {
                    if ((spin & 0xfff) == 0xfff && (lane_quit.load() || lane_abort.load())) goto next;
#if defined(__x86_64__)
                    __builtin_ia32_pause();
#endif
                }
                {
                    const double t0 = now_s();
                    lane_wait_s += t0 - tw;
                    const float* x = l.m_pub;
                    const int Tn = job_T;   // (a token, or a window's rows: [T][E] inputs, [T][nu] picks, [T][nu] weights)
                    size_t n_ent = 0;
                    rent.clear();
                    for (int t = 0; t < Tn; ++t) {
                        int32_t ids[16];
                        float w[16];
                        std::memcpy(ids, x + (size_t) Tn * E + (size_t) t * nu, sizeof(int32_t) * (size_t) nu);
                        std::memcpy(w, x + (size_t) Tn * E + (size_t) (Tn + t) * nu, sizeof(float) * (size_t) nu);
                        for (int j = 0; j < nu; ++j) {
                            const int e = ids[j];
                            l.picks[(size_t) t * 16 + j] = e;   // (counted when the row is kept: counts_flush)
                            if (e < 0 || e >= g.n_expert) continue;
                            if (l.slot_of[(size_t) e] < 0) rent.push_back({t, l.ram_of[(size_t) e], w[j]});
                        }
                    }
                    // (the rows' RAM picks together: an expert several rows picked is read once for them)
                    n_ent = rent.size();
                    if (rent.empty()) std::memset(l.m_cpu, 0, sizeof(float) * (size_t) E * (size_t) Tn);
                    else if (dbg & 4) {   // (tests: a row at a time)
                        std::vector<CpuRowEntry> one;
                        for (int t = 0; t < Tn; ++t) {
                            one.clear();
                            for (const CpuRowEntry& e : rent)
                                if (e.row == t) one.push_back({0, e.slot, e.w});
                            cpu_experts_rows(*lane, l.cex, one, x + (size_t) t * E, 1, l.m_cpu + (size_t) t * E);
                        }
                    } else cpu_experts_rows(*lane, l.cex, rent, x, Tn, l.m_cpu);
                    ent.resize(n_ent);   // (below: only whether there were any)
                    st.cpu_experts += (int64_t) n_ent;
                    st.experts += nu * Tn;
                    __atomic_store_n(f_cpu + (size_t) il * kLine, v, __ATOMIC_SEQ_CST);
                    const double t1 = now_s();
                    lane_work_s += t1 - t0;
                    if (!ent.empty()) st.cpu_ms += (t1 - t0) * 1e3;
                }
            }
        next:
            at = v;
            lane_done.store(v, std::memory_order_release);
        }
    }
    std::atomic<bool> lane_abort{false};

    void worker_run(size_t gi) {
        Worker& w = workers[gi];
        cudaSetDevice(G[gi].dev);
        for (int seen = 0;;) {
            {
                std::unique_lock<std::mutex> lk(w.mu);
                w.cv.wait(lk, [&] { return w.quit || w.job != seen; });
                if (w.quit) return;
                seen = w.job;
            }
            const double t0 = now_s();
            std::string e;
            const bool ok = enqueue_win(gi, e);
            {
                std::lock_guard<std::mutex> lk(w.mu);
                w.ok = ok;
                w.err = e;
                w.done = seen;
                if (gi + 1 < G.size()) fast_enq_s += now_s() - t0;
            }
            w.cv.notify_all();
        }
    }
    // a token's step: a window of one row
    bool step_fast(int id, int64_t p, float* logits_out, std::string& err) {
        int top = 0;
        if (!step_win(&id, 1, p, &top, logits_out, err)) return false;
        win_p0 = -1;
        return true;
    }
    // a token's (or a window's) layers queued on every GPU by its worker, the lane's thread told; back when the last
    // GPU is done
    bool fast_run(double t0, std::string& err) {
        ++seq;
        lane_abort = false;
        lane_seq.store(seq, std::memory_order_release);
        for (Worker& w : workers) {
            {
                std::lock_guard<std::mutex> lk(w.mu);
                ++w.job;
            }
            w.cv.notify_all();
        }
        bool ok = true;
        for (Worker& w : workers) {
            std::unique_lock<std::mutex> lk(w.mu);
            w.cv.wait(lk, [&] { return w.done == w.job; });
            if (!w.ok) {
                ok = false;
                err = w.err;
            }
        }
        if (!ok || hk::wait_timeouts()) {   // (the lane's thread let go of the token, the streams left to drain)
            lane_abort = true;
            if (ok) err = "a device wait for the CPU lane or the hand-off gave up";
            ok = false;
        }
        while (lane_done.load(std::memory_order_acquire) != seq) std::this_thread::yield();
        if (!ok) {
            for (Gpu& gp : G) {
                cudaSetDevice(gp.dev);
                cudaStreamSynchronize(gp.s);
            }
            return false;
        }
        fast_s += now_s() - t0;
        ++fast_n;
        if (phases) {
            ph_s[job_T] += now_s() - t0;
            ++ph_n[job_T];
            for (Layer& l : L) {
                CK(cudaSetDevice(G[(size_t) l.gpu].dev));
                for (int k2 = 1; k2 < 10; ++k2) {
                    float ms = 0.0f;
                    if (cudaEventElapsedTime(&ms, l.pe[k2 - 1], l.pe[k2]) == cudaSuccess) ph_ms[job_T][k2] += ms;
                }
            }
        }
        CK(cudaSetDevice(G.back().dev));
        st.tokens += job_T;
        return true;
    }

    // ---- the host-free step: T (1 .. kWin) tokens through every layer, a token's step and a verify window alike -------
    // One path for every T: the GEMVs take T columns (the weights read once), the fused kernels a block a row (each row
    // the arithmetic of a token's step: a window's row t and the token step at its position run the same code on the
    // same values), the attention's tiles lie at fixed positions (attention_split_rows), the routed experts are
    // grouped over the rows (ds4_iq's grouped form; T 1: its plan form, the same sums), the lane gets T rows.
    // Each stream's conv inputs stay in the layer (win_in) and the histories go from one buffer to the other (hist_* ->
    // alt_*, then swapped): the histories before the step stay whole, and keep() rebuilds those of any shorter prefix.
    bool win_scratch(Gpu& gp, std::string& err) {
        const size_t W = kWin, E = (size_t) g.n_embd, HD = (size_t) g.n_head * g.head_dim, nu = (size_t) g.n_used;
        const size_t kvd = (size_t) g.n_head_kv * g.head_dim, ff = (size_t) g.n_ff_exp, H = (size_t) g.n_head;
        const size_t qmax = std::max(E, HD);
        for (int il = gp.l0; il < gp.l1; ++il) {
            Layer& l = L[(size_t) il];
            if (!gp.alloc(l.win_in, W * (2 * kvd + 2 * E), err) || !zeros(gp, l.win_h0, 3 * (2 * kvd + 2 * E), err)) return false;
            l.alt_k = l.win_h0;
            l.alt_v = l.alt_k + 3 * kvd;
            l.alt_a = l.alt_v + 3 * kvd;
            l.alt_f = l.alt_a + 3 * E;
        }
        const bool last = &gp == &G.back();
        (void) nu;
        return gp.alloc(gp.w_cb, (1 + k::kMaxStates) * W * E, err) && gp.alloc(gp.w_mixo, 3 * W * E, err) &&
               gp.alloc(gp.w_xn, 3 * W * E, err) && gp.alloc(gp.w_stat, 4 * k::kMaxStates * W, err) &&
               gp.alloc(gp.w_q, W * HD, err) && gp.alloc(gp.w_gate, W * HD, err) && gp.alloc(gp.w_ao, W * HD, err) &&
               gp.alloc(gp.w_rel, W * H * g.d_rel, err) && gp.alloc(gp.w_sink, W * H, err) &&
               gp.alloc(gp.w_dk, W * 32, err) && gp.alloc(gp.w_dv, W * 32, err) && gp.alloc(gp.w_xf, W * E, err) &&
               gp.alloc(gp.w_rl, W * (size_t) g.n_expert, err) && gp.alloc(gp.w_rw, W * 16, err) &&
               gp.alloc(gp.w_ids, W * 16, err) && gp.alloc(gp.w_xq, W * qmax, err) &&
               gp.alloc(gp.w_xd, W * qmax / 32, err) && gp.alloc(gp.w_x2q, 2 * W * E, err) &&
               gp.alloc(gp.w_x2d, 2 * W * E / 32, err) && gp.alloc(gp.w_eh, 16 * W * ff, err) &&
               gp.alloc(gp.w_ehq, 16 * W * ff, err) && gp.alloc(gp.w_ehd, 16 * W * ff / 32, err) &&
               gp.alloc(gp.w_part, 16 * W * E, err) && gp.alloc(gp.w_gptr, (size_t) g.n_expert + 16, err) &&
               gp.alloc(gp.w_gstart, (size_t) g.n_expert + 16, err) && gp.alloc(gp.w_gent, 16 * W + 16, err) &&
               gp.alloc(gp.w_ng, 16, err) && gp.alloc(gp.w_pub, W * (E + 16), err) && gp.alloc(gp.w_top, 2 * W, err) &&
               (!last || gp.alloc(gp.w_logits, W * (size_t) g.n_vocab, err));
    }
    // (tests, Model::set_debug: the window's rows through a token's launches, a part at a time - 1: the block types'
    //  GEMVs a column a launch, 2: the experts' plan form a row at a time, 4: the lane a row at a time, 8: the F32
    //  GEMVs a column a launch)
    int dbg = 0;
    void gemv32(Gpu& gp, const float* W, int kk, int n, const float* x, int T, float* y) {
        if ((dbg & 8) && T > 1) {
            for (int t = 0; t < T; ++t)
                hk::gemv_f32w(W, (size_t) kk, kk, n, x + (size_t) t * kk, (size_t) kk, 1, y + (size_t) t * n, (size_t) n, 1.0f, gp.s);
            return;
        }
        hk::gemv_f32w(W, (size_t) kk, kk, n, x, (size_t) kk, T, y, (size_t) n, 1.0f, gp.s);
    }
    bool mmw(Gpu& gp, const Mat& M, const float* x, int T, float* y) {
        if (M.type == kF32) {
            gemv32(gp, (const float*) M.w, M.k, M.n, x, T, y);
            return true;
        }
        return mmw_multi(gp, {{&M, y}}, x, T);
    }
    bool mmw_multi(Gpu& gp, std::initializer_list<Out> outs, const float* x, int T) {
        const int kk = outs.begin()->m->k;
        if ((dbg & 1) && T > 1) {
            for (int t = 0; t < T; ++t) {
                hk::q8_rows(x + (size_t) t * kk, (size_t) kk, 1, kk, gp.w_xq, gp.w_xd, gp.s);
                iq::MvJob jobs[8];
                int n = 0;
                for (const Out& o2 : outs) {
                    iq::MvJob& j = jobs[n++];
                    j.w = o2.m->w;
                    j.ld = o2.m->ld;
                    j.type = o2.m->type;
                    j.k = kk;
                    j.n = o2.m->n;
                    j.ncol = 1;
                    j.xq = gp.w_xq;
                    j.xd = gp.w_xd;
                    j.y = o2.y + (size_t) t * o2.m->n;
                    j.ycs = (size_t) o2.m->n;
                }
                if (!iq::mv(jobs, n, gp.s)) return false;
            }
            return true;
        }
        hk::q8_rows(x, (size_t) kk, T, kk, gp.w_xq, gp.w_xd, gp.s);
        iq::MvJob jobs[8];
        int n = 0;
        for (const Out& o2 : outs) {
            if (o2.m->type == kF32 || o2.m->k != kk || n >= 8) return false;
            iq::MvJob& j = jobs[n++];
            j.w = o2.m->w;
            j.ld = o2.m->ld;
            j.type = o2.m->type;
            j.k = kk;
            j.n = o2.m->n;
            j.ncol = T;
            j.xq = gp.w_xq;
            j.xd = gp.w_xd;
            j.y = o2.y;
            j.ycs = (size_t) o2.m->n;
        }
        return iq::mv(jobs, n, gp.s);
    }
    bool layer_win(Layer& l, Gpu& gp, int T, int64_t p0, int& nb, unsigned v, std::string& err) {
        const int E = g.n_embd, H = g.n_head, Hkv = g.n_head_kv, D = g.head_dim, HD = H * D, kvd = Hkv * D, ff = g.n_ff_exp;
        const int nu = g.n_used;
        const size_t TE = (size_t) T * E, ldb = TE;   // (the states a window's rows apart: the hand-off carries no gaps)
        cudaStream_t s = gp.s;
        float *cur = gp.w_cb, *blocks = gp.w_cb + ldb;
        float *in_k = l.win_in, *in_v = in_k + (size_t) kWin * kvd, *in_a = in_v + (size_t) kWin * kvd, *in_f = in_a + (size_t) kWin * E;
        const float* src = cur;
        ph(l, gp, 0);
        const bool fused = l.il > 0;   // (the mix and the norms in a launch, a block a row)
        if (fused) k::mix_rms_rows(blocks, ldb, nb, cur, T, E, l.attn_res, l.swa ? 1 : 3, l.swa ? l.attn_norm : l.q_norm,
                                   l.k_norm, l.v_norm, g.eps, gp.w_xn, s);
        const float *xq = gp.w_xn, *xk = gp.w_xn, *xv = gp.w_xn;
        bool ok = true;
        if (l.swa) {
            if (!fused) k::rms(src, (size_t) E, T, E, l.attn_norm, g.eps, gp.w_xn, (size_t) E, s);
            ph(l, gp, 1);
            ok = mmw_multi(gp, {{&l.q, gp.w_q}, {&l.gate, gp.w_gate}, {&l.rel, gp.w_rel}, {&l.wk, in_k}, {&l.wv, in_v}}, xq, T);
        } else {
            const size_t st1 = l.il > 0 ? TE : 0;
            if (!fused) {
                k::rms(src, (size_t) E, T, E, l.q_norm, g.eps, gp.w_xn, (size_t) E, s);
                k::rms(src + st1, (size_t) E, T, E, l.k_norm, g.eps, gp.w_xn + TE, (size_t) E, s);
                k::rms(src + 2 * st1, (size_t) E, T, E, l.v_norm, g.eps, gp.w_xn + 2 * TE, (size_t) E, s);
            }
            xk = gp.w_xn + TE;
            xv = gp.w_xn + 2 * TE;
            ph(l, gp, 1);
            ok = mmw_multi(gp, {{&l.q, gp.w_q}, {&l.gate, gp.w_gate}, {&l.rel, gp.w_rel}}, xq, T) && mmw(gp, l.wk, xk, T, in_k) &&
                 mmw(gp, l.wv, xv, T, in_v) && mmw(gp, l.k_proj, xk, T, gp.w_dk) && mmw(gp, l.v_proj, xv, T, gp.w_dv);
        }
        ok = ok && mmw(gp, l.sink, xq, T, gp.w_sink);
        ph(l, gp, 2);
        float tau[16];
        for (int t = 0; t < T; ++t) {
            const int64_t pt = job_multi ? mrows[t].p : p0 + t;
            tau[t] = l.swa ? 1.0f : 1.0f + g.log_alpha * std::log(std::max(1.0f, (float) ((double) (pt + 1) / g.log_floor)));
        }
        if (job_multi) {   // (each row its own conversation's: its histories, its caches, its position)
            for (int t = 0; t < T; ++t) {
                SeqLayer& q = seqs[(size_t) mrows[t].seq].L[(size_t) l.il];
                k::kv_finish_rows(in_k + (size_t) t * kvd, in_v + (size_t) t * kvd, 1, q.hist_k, q.hist_v, q.alt_k, q.alt_v,
                                  l.k_conv, l.v_conv, l.swa ? nullptr : gp.w_dk + (size_t) t * 4 * Hkv,
                                  l.swa ? nullptr : gp.w_dv + (size_t) t * 4 * Hkv, Hkv, D, g.qk_clip, g.eps, q.K, q.V,
                                  l.ring, mrows[t].p, s);
            }
        } else {
            k::kv_finish_rows(in_k, in_v, T, l.hist_k, l.hist_v, l.alt_k, l.alt_v, l.k_conv, l.v_conv, l.swa ? nullptr : gp.w_dk,
                              l.swa ? nullptr : gp.w_dv, Hkv, D, g.qk_clip, g.eps, l.K, l.V, l.ring, p0, s);
        }
        k::clip2_rows(gp.w_q, T, H, D, l.q_clip, g.qk_clip, gp.w_rel, g.d_rel, l.rel_clip, g.rel_clip, tau, s);
        ph(l, gp, 3);
        if (job_multi)
            for (int t = 0; t < T; ++t) {
                const SeqLayer& q = seqs[(size_t) mrows[t].seq].L[(size_t) l.il];
                const int64_t pf = mrows[t].p, lo = l.swa ? std::max<int64_t>(0, pf - (g.n_swa - 1)) : 0;
                if (!k::attention_split_rows(gp.w_q + (size_t) t * HD, gp.w_rel + (size_t) t * H * g.d_rel,
                                             gp.w_sink + (size_t) t * H, 1, H, Hkv, D, g.d_rel, l.rel_bias,
                                             l.swa ? g.n_swa : g.rel_extent, q.K, q.V, l.ring, pf, lo, l.swa ? g.n_swa : 0,
                                             1.0f / std::sqrt((float) D), gp.S, gp.S_floats, gp.w_gate + (size_t) t * HD,
                                             gp.w_ao + (size_t) t * HD, s)) {
                    err = "layer " + std::to_string(l.il) + ": the attention's scratch does not hold a row's tiles";
                    return false;
                }
            }
        // the rows in one pair of launches (the K / V tiles read once) - or, when the scratch does not hold that many
        // rows' parts over this many keys, a few rows a pair
        if (!job_multi) {
            const int64_t keys = l.swa ? std::min<int64_t>(p0 + T, (int64_t) g.n_swa + T) : p0 + T;
            const int fit = std::max(1, std::min(T, k::attention_rows_fit(H, D, keys, gp.S_floats)));
            for (int t0 = 0; t0 < T; t0 += fit) {
                const int tn = std::min(fit, T - t0);
                const int64_t pf = p0 + t0, lo = l.swa ? std::max<int64_t>(0, pf - (g.n_swa - 1)) : 0;
                if (!k::attention_split_rows(gp.w_q + (size_t) t0 * HD, gp.w_rel + (size_t) t0 * H * g.d_rel,
                                             gp.w_sink + (size_t) t0 * H, tn, H, Hkv, D, g.d_rel, l.rel_bias,
                                             l.swa ? g.n_swa : g.rel_extent, l.K, l.V, l.ring, pf, lo, l.swa ? g.n_swa : 0,
                                             1.0f / std::sqrt((float) D), gp.S, gp.S_floats, gp.w_gate + (size_t) t0 * HD,
                                             gp.w_ao + (size_t) t0 * HD, s)) {
                    err = "layer " + std::to_string(l.il) + ": the attention's scratch does not hold a row's tiles";
                    return false;
                }
            }
        }
        ph(l, gp, 4);
        ok = ok && mmw(gp, l.wo, gp.w_ao, T, in_a);
        if (job_multi) {
            for (int t = 0; t < T; ++t) {
                SeqLayer& q = seqs[(size_t) mrows[t].seq].L[(size_t) l.il];
                k::conv_add_rows(in_a + (size_t) t * E, 1, q.hist_a, q.alt_a, l.a_conv, E, cur + (size_t) t * E,
                                 l.il % g.res_block == 0 ? blocks + (size_t) nb * ldb + (size_t) t * E : nullptr, s);
            }
            if (l.il % g.res_block == 0) ++nb;
        } else if (l.il % g.res_block == 0) {   // (a block's first layer: the stream saved, the attention's output a new one)
            k::conv_add_rows(in_a, T, l.hist_a, l.alt_a, l.a_conv, E, cur, blocks + (size_t) nb * ldb, s);
            ++nb;
        } else {
            k::conv_add_rows(in_a, T, l.hist_a, l.alt_a, l.a_conv, E, cur, nullptr, s);
        }
        ph(l, gp, 5);
        k::mix_rms_rows(blocks, ldb, nb, cur, T, E, l.ffn_res, 1, l.ffn_norm, nullptr, nullptr, g.eps, gp.w_xf, s);
        gemv32(gp, (const float*) l.router.w, E, g.n_expert, gp.w_xf, T, gp.w_rl);
        hk::router(gp.w_rl, T, g.n_expert, l.router_b, nu, true, g.w_scale, gp.w_ids, gp.w_rw, s);
        // the routing out to the lane (its picks are the ones the device's map has no slot for), then this GPU's
        hk::pack3(gp.w_xf, TE, gp.w_ids, (size_t) T * nu, gp.w_rw, (size_t) T * nu, gp.w_pub, s);
        hk::publish(gp.w_pub, l.m_pub, TE + 2 * (size_t) T * nu, gp.cnt, f_pub + (size_t) l.il * kLine, v, s);
        iq::Args a;
        a.tr[0] = l.tr[0];
        a.tr[1] = l.tr[1];
        a.tr[2] = l.tr[2];
        a.n_embd = E;
        a.n_ff = ff;
        a.h = gp.w_eh;
        a.part = gp.w_part;
        a.tg = l.tg;
        a.td = l.td;
        a.act = iq::kActSitu;
        a.situ_beta = g.situ_beta;
        a.situ_lb = g.situ_lb;
        a.hq = gp.w_ehq;
        a.hd = gp.w_ehd;
        if (T == 1 || (dbg & 2)) {   // (a token: the plan form - an entry a pick, no grouping)
            for (int t = 0; t < T; ++t) {
                hk::plan_experts(gp.w_ids + (size_t) t * nu, nu, l.experts, l.blob, gp.w_gptr, s, l.slot_d, gp.w_rw + (size_t) t * nu);
                if (t == 0) ph(l, gp, 6);
                hk::q8_rows(gp.w_xf + (size_t) t * E, (size_t) E, 1, E, gp.w_xq, gp.w_xd, s);
                a.plan_ptr = gp.w_gptr;
                a.xq = gp.w_xq;
                a.xd = gp.w_xd;
                ok = ok && iq::gate_up(a, nu, 1, s);
                hk::q8_rows(gp.w_eh, (size_t) ff, nu, ff, gp.w_ehq, gp.w_ehd, s);
                ok = ok && iq::down(a, nu, 1, s);
                k::wsum(gp.w_part, nu, E, gp.w_rw + (size_t) t * nu, in_f + (size_t) t * E, s);
            }
        } else {   // (the window's picks grouped by expert; the device's map: the RAM ones' weights zeroed, the lane's)
            hk::group_experts(gp.w_ids, T, nu, g.n_expert, l.experts, l.blob, gp.w_gptr, gp.w_gstart, gp.w_gent, gp.w_ng, s,
                              l.slot_d, gp.w_rw);
            ph(l, gp, 6);
            hk::q8_rows(gp.w_xf, (size_t) E, T, E, gp.w_xq, gp.w_xd, s);
            hk::dup_rows_q8(gp.w_xq, gp.w_xd, T, E, gp.w_x2q, gp.w_x2d, s);
            a.grp_ptr = gp.w_gptr;
            a.grp_start = gp.w_gstart;
            a.grp_ent = gp.w_gent;
            a.n_groups = gp.w_ng;
            a.xq = gp.w_x2q;
            a.xd = gp.w_x2d;
            const int groups = std::min(nu * T, g.n_expert);
            ok = ok && iq::gate_up(a, groups, T, s);
            hk::q8_rows(gp.w_eh, (size_t) ff, 16 * T, ff, gp.w_ehq, gp.w_ehd, s);
            ok = ok && iq::down(a, groups, T, s);
            hk::moe_sum(gp.w_part, T, nu, E, gp.w_rw, in_f, s);
        }
        ph(l, gp, 7);
        k::lane_add_rows(f_cpu + (size_t) l.il * kLine, v, gp.w_ids, l.slot_d, nu, l.m_cpu, in_f, E, T, s);
        ph(l, gp, 8);
        if (job_multi) {
            for (int t = 0; t < T; ++t) {
                SeqLayer& q = seqs[(size_t) mrows[t].seq].L[(size_t) l.il];
                k::conv_add_rows(in_f + (size_t) t * E, 1, q.hist_f, q.alt_f, l.f_conv, E, cur + (size_t) t * E, nullptr, s);
                std::swap(q.hist_k, q.alt_k);
                std::swap(q.hist_v, q.alt_v);
                std::swap(q.hist_a, q.alt_a);
                std::swap(q.hist_f, q.alt_f);
            }
        } else {
            k::conv_add_rows(in_f, T, l.hist_f, l.alt_f, l.f_conv, E, cur, nullptr, s);
        }
        ph(l, gp, 9);
        if (!job_multi) {
            std::swap(l.hist_k, l.alt_k);
            std::swap(l.hist_v, l.alt_v);
            std::swap(l.hist_a, l.alt_a);
            std::swap(l.hist_f, l.alt_f);
        }
        // the drafter's tap: the stream after this layer, for the last GPU (its own: a copy; another's: through host
        // memory both map, there before this GPU's hand-off is)
        if (taps_on && tap_of[(size_t) l.il] >= 0) {
            const int kt = tap_of[(size_t) l.il];
            if (&gp == &G.back()) CK(cudaMemcpyAsync(gp.d_tapl + (size_t) kt * TE, cur, sizeof(float) * TE, cudaMemcpyDeviceToDevice, s));
            else hk::publish(cur, m_taps + (size_t) kt * TE, TE, gp.cnt, (unsigned*) (gp.cnt + 2), v, s);
        }
        if (!ok) err = "layer " + std::to_string(l.il) + ": a kernel did not take its shapes";
        return ok;
    }
    // GPU gi's part of a step, queued (the last GPU: with the head, and waited for)
    bool enqueue_win(size_t gi, std::string& err) {
        const size_t E = (size_t) g.n_embd;
        const int T = job_T;
        const size_t ldb = (size_t) T * E;
        Gpu& gp = G[gi];
        const unsigned v = seq;
        int nb = 0;
        for (int il = 0; il < gp.l0; ++il) nb += il % g.res_block == 0;
        if (job_slot >= 0 && !snap_copy(gp, job_slot, true, err)) return false;
        if (job_multi)
            for (int t = 0; t < T; ++t)
                if (mrows[t].slot >= 0 && !snap_copy_seq(gp, seqs[(size_t) mrows[t].seq], mrows[t].slot, err)) return false;
        if (gi == 0) {
            CK(cudaMemcpyAsync(gp.w_xn, h_xw, sizeof(float) * T * E, cudaMemcpyHostToDevice, gp.s));
            k::rms(gp.w_xn, E, T, (int) E, tok_norm, g.eps, gp.w_cb, E, gp.s);
        } else {
            hk::wait_pull(f_hand + gi * kLine, v, gp.w_cb, gp.m_hand, (size_t) (1 + nb) * ldb, gp.s);
        }
        for (int il = gp.l0; il < gp.l1; ++il)
            if (!layer_win(L[(size_t) il], gp, T, job_p, nb, v, err)) return false;
        if (gi + 1 < G.size()) {
            hk::publish(gp.w_cb, G[gi + 1].m_hand, (size_t) (1 + nb) * ldb, gp.cnt, f_hand + (gi + 1) * kLine, v, gp.s);
            return true;
        }
        // the head over the rows: each row's logits, its argmax
        k::mix_rms_rows(gp.w_cb + ldb, ldb, nb, gp.w_cb, T, (int) E, out_res, 1, out_norm, nullptr, nullptr, g.eps, gp.w_xf,
                        gp.s);
        if (taps_on) {   // (the drafter's rows: the other GPUs' taps from the host memory, this GPU's, the head's input)
            const int nt = (int) tap_layers.size(), last = (int) G.size() - 1;
            const float* srcs[8];
            bool remote = false;
            for (int kt = 0; kt + 1 < nt; ++kt) {
                const bool mine = L[(size_t) tap_layers[(size_t) kt]].gpu == last;
                srcs[kt] = (mine ? gp.d_tapl : gp.d_tapr) + (size_t) kt * ldb;
                remote = remote || !mine;
            }
            srcs[nt - 1] = gp.w_xf;
            if (remote)
                CK(cudaMemcpyAsync(gp.d_tapr, m_taps, sizeof(float) * (size_t) (nt - 1) * ldb, cudaMemcpyHostToDevice, gp.s));
            // (the drafter's last feed from this buffer read on its own stream: done before it is written again)
            if (tn.tap_event) CK(cudaStreamWaitEvent(gp.s, gp.ev_tap[tap_buf], 0));
            k::taps_gather(srcs, 0u, nt, T, (int) E, gp.d_taps[tap_buf], (size_t) nt * E, gp.s);
        }
        if (!mmw(gp, head, gp.w_xf, T, gp.w_logits)) {
            err = "the output head's GEMV did not take its shapes";
            return false;
        }
        if (g.softcap > 0.0f) k::softcap(gp.w_logits, (size_t) T * g.n_vocab, g.softcap, gp.s);
        if (bias_n > 0 && !job_multi) k::bias_rows(gp.w_logits, (size_t) g.n_vocab, T, d_bias_ids, d_bias_vals, bias_n, gp.s);
        k::argmax_rows(gp.w_logits, (size_t) g.n_vocab, T, g.n_vocab_unpadded, gp.w_top, gp.s);
        CK(cudaMemcpyAsync(h_wtop, gp.w_top, sizeof(float) * 2 * T, cudaMemcpyDeviceToHost, gp.s));
        if (job_rows) CK(cudaMemcpyAsync(h_wlog, gp.w_logits, sizeof(float) * T * g.n_vocab, cudaMemcpyDeviceToHost, gp.s));
        CK(cudaStreamSynchronize(gp.s));
        for (int t = 0; t < T; ++t) {
            win_tops[t] = (int) h_wtop[2 * t];
            if (job_rows)
                std::memcpy(job_rows + (size_t) t * g.n_vocab_unpadded, h_wlog + (size_t) t * g.n_vocab,
                            sizeof(float) * (size_t) g.n_vocab_unpadded);
        }
        top_id = win_tops[T - 1];
        if (k::errors() || hk::errors()) {
            err = "a kernel launch failed";
            return false;
        }
        return true;
    }
    bool step_win(const int* ids, int T, int64_t p0, int* tops, float* rows, std::string& err) {
        const int E = g.n_embd;
        const double t0 = now_s();
        for (int t = 0; t < T; ++t) {
            if (ids[t] < 0 || ids[t] >= g.n_vocab) {
                err = "token " + std::to_string(ids[t]) + " is outside the vocabulary";
                return false;
            }
            const uint8_t* row = tok_embd.data() + (size_t) ids[t] * tok_row;
            float* dst = h_xw + (size_t) t * E;
            if (tok_type == 13) dequantize_row_q5_K(row, dst, E);
            else if (tok_type == 14) dequantize_row_q6_K(row, dst, E);
            else std::memcpy(dst, row, (size_t) E * 4);
        }
        if (!taps_flush(err)) return false;   // (the rows the last step left for the drafter: its buffer is the other one)
        use_decay(T == 1 ? tn.tok_decay : std::pow(tn.tok_decay, (float) T));
        // (a window: its moves a batch as large as its rows' steps would have had, 16 at most)
        // (a window's rows route up to T tokens' worth: up to 4 of them count, at most the bounce's 16)
        if (!swap_poll(err, std::min(16, swap_batch * std::min(T, tn.win_rows)))) return false;
        hw = std::max(hw, p0 + T);
        job_slot = snap_next || p0 / tn.snap_every != (p0 + T - 1) / tn.snap_every || p0 % tn.snap_every == 0 ? snap_alloc(p0) : -1;
        snap_next = false;
        job_p = p0;
        job_T = T;
        job_ids = ids;
        job_rows = rows;
        tap_buf ^= 1;
        const bool ok = fast_run(t0, err);
        if (!ok) return false;
        for (int t = 0; t < T; ++t) tops[t] = win_tops[t];
        win_p0 = p0;
        win_T = T;
        cnt_pending = T;
        if (taps_on) {
            tap_pending = T;
            tap_p0 = p0;
        }
        return true;
    }
    // the context cut back to the last window's first `rows` rows: each layer's histories from the window's inputs and
    // the histories before it (the buffers the step left: alt_*)
    bool win_keep(int rows, std::string& err) {
        if (win_p0 < 0 || rows < 1 || rows > win_T || n_end != win_p0 + win_T) {
            err = "keep: no window to cut back, or not " + std::to_string(rows) + " of its rows";
            return false;
        }
        if (rows == win_T) return true;
        const int kvd = g.n_head_kv * g.head_dim, E = g.n_embd;
        for (Gpu& gp : G) {
            CK(cudaSetDevice(gp.dev));
            for (int il = gp.l0; il < gp.l1; ++il) {
                Layer& l = L[(size_t) il];
                float* hist[4] = {l.hist_k, l.hist_v, l.hist_a, l.hist_f};
                const float* in[4] = {l.win_in, l.win_in + (size_t) kWin * kvd, l.win_in + 2 * (size_t) kWin * kvd,
                                      l.win_in + 2 * (size_t) kWin * kvd + (size_t) kWin * E};
                const float* h0[4] = {l.alt_k, l.alt_v, l.alt_a, l.alt_f};
                const int C[4] = {kvd, kvd, E, E};
                k::hist_set4(hist, h0, in, C, rows, gp.s);
            }
        }
        for (Gpu& gp : G) {
            CK(cudaSetDevice(gp.dev));
            CK(cudaStreamSynchronize(gp.s));
        }
        n_end = win_p0 + rows;
        win_T = rows;
        if (tap_pending > rows) tap_pending = rows;
        if (cnt_pending > rows) cnt_pending = rows;
        for (int i = 0; i < kSnaps; ++i)
            if (snap_pos[i] > n_end) snap_pos[i] = -1;
        return true;
    }

    // The request's logit bias (Model::set_logit_bias): its entries (distinct ids of the unpadded vocabulary, none of
    // them zero) on the last GPU, added to every row of logits after the soft cap - before the argmax and the copy the
    // sampler reads - by the token step, every window row and the prompt's last row.  bias_n 0: no launch, the logits
    // exactly as without.  The buffers (a vocabulary of entries at most) are allocated with the first request that has
    // a bias; the caller sets it between two steps, when the last GPU's stream is idle
    int *d_bias_ids = nullptr;
    float* d_bias_vals = nullptr;
    int bias_n = 0;
    std::vector<std::pair<int, float>> bias_cur;
    bool set_logit_bias(const std::vector<std::pair<int, float>>& entries, std::string& err) {
        std::map<int, float> u;
        for (const auto& [id, b] : entries)
            if (id >= 0 && id < g.n_vocab_unpadded) u[id] = b;   // (an id twice: its last bias)
        std::vector<std::pair<int, float>> e;
        for (const auto& [id, b] : u)
            if (b != 0.0f) e.push_back({id, b});
        if (e == bias_cur) return true;
        bias_n = 0;
        bias_cur.clear();
        if (e.empty()) return true;
        Gpu& gp = G.back();
        CK(cudaSetDevice(gp.dev));
        CK(cudaStreamSynchronize(gp.s));
        if (!d_bias_ids && (!gp.alloc(d_bias_ids, (size_t) g.n_vocab_unpadded, err) ||
                            !gp.alloc(d_bias_vals, (size_t) g.n_vocab_unpadded, err)))
            return false;
        std::vector<int> ids(e.size());
        std::vector<float> vals(e.size());
        for (size_t i = 0; i < e.size(); ++i) {
            ids[i] = e[i].first;
            vals[i] = e[i].second;
        }
        CK(cudaMemcpy(d_bias_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_bias_vals, vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice));
        bias_cur = std::move(e);
        bias_n = (int) bias_cur.size();
        return true;
    }

    // the head, on the last GPU: the token whose stream and saved ones are in its gp.cur / gp.blocks
    bool head_out(int nb, float* logits_out, std::string& err) {
        const int E = g.n_embd;
        Gpu& gp = G.back();
        k::mix(gp.blocks, nb, gp.cur, E, out_res, 1, 1e-6f, gp.stat, gp.mixo, gp.s);
        k::rms(gp.mixo, (size_t) E, 1, E, out_norm, g.eps, gp.xf, (size_t) E, gp.s);
        if (!tap("result_norm", -1, gp, gp.xf, (size_t) E, err)) return false;
        if (!mm(gp, head, gp.xf, gp.logits)) {
            err = "the output head's GEMV did not take its shapes";
            return false;
        }
        if (g.softcap > 0.0f) k::softcap(gp.logits, (size_t) g.n_vocab_unpadded, g.softcap, gp.s);
        if (bias_n > 0) k::bias_rows(gp.logits, (size_t) g.n_vocab, 1, d_bias_ids, d_bias_vals, bias_n, gp.s);
        k::argmax(gp.logits, g.n_vocab_unpadded, gp.top, gp.s);
        CK(cudaMemcpyAsync(h_top, gp.top, 8, cudaMemcpyDeviceToHost, gp.s));
        if (logits_out)
            CK(cudaMemcpyAsync(h_big, gp.logits, sizeof(float) * g.n_vocab_unpadded, cudaMemcpyDeviceToHost, gp.s));
        CK(sync_timed(gp.s));
        if (logits_out) std::memcpy(logits_out, h_big, sizeof(float) * g.n_vocab_unpadded);
        top_id = (int) h_top[0];
        if (tapf && logits_out) tapf("result_output", -1, logits_out, (size_t) g.n_vocab_unpadded, tap_p, 1);
        if (k::errors() || hk::errors()) {
            err = "a kernel launch failed";
            return false;
        }
        return true;
    }

    // ---- the prompt path: a chunk of T tokens through a GPU's layers at once -----------------------------------------
    // y [T][n] = x [T][k] W^T on the BLAS: an F32 matrix as it is, a block type's rows decoded to F16 a band (ds4_iq's
    // f16_rows) against x in F16 (fresh: x is not the previous call's - its F16 rows are made)
    bool gemm(Gpu& gp, const Mat& M, const float* x, int T, float* y, bool fresh = true) {
        const float one = 1.0f, zero = 0.0f;
        if (M.type == kF32)
            return cublasSgemm(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, M.n, T, M.k, &one, (const float*) M.w, M.k, x, M.k, &zero,
                               y, M.n) == CUBLAS_STATUS_SUCCESS;
        if (fresh) hk::to_f16(x, (size_t) M.k, T, M.k, gp.xh16, (size_t) M.k, gp.s);
        const int band = (int) std::max<size_t>(1, gp.wband_n / (size_t) M.k);
        for (int r0 = 0; r0 < M.n; r0 += band) {
            const int nb = std::min(band, M.n - r0);
            if (!iq::f16_rows(M.type, M.w + (size_t) r0 * M.ld, M.k, nb, gp.wband, M.k, gp.s)) return false;
            if (cublasGemmEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, nb, T, M.k, &one, gp.wband, CUDA_R_16F, M.k, gp.xh16,
                             CUDA_R_16F, M.k, &zero, y + r0, CUDA_R_32F, M.n, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                return false;
        }
        return true;
    }

    // The chunk's attention as GEMMs over each kv head's keys (the cache's rows, the chunk's own among them), a pass of
    // tokens at a time: S = K^T Q, the probabilities (attn_softmax_rows), O = V P
    bool attention_rows(Layer& l, Gpu& gp, int T, int64_t p0, std::string& err) {
        const int H = g.n_head, Hkv = g.n_head_kv, D = g.head_dim, per = H / Hkv, kvd = Hkv * D, HD = H * D;
        const int extent = l.swa ? g.n_swa : g.rel_extent, window = l.swa ? g.n_swa : 0;
        cudaStream_t s = gp.s;
        const float one = 1.0f, zero = 0.0f;
        k::store_rows(gp.c_kc, T, kvd, l.K, l.ring, p0, s);
        k::store_rows(gp.c_vc, T, kvd, l.V, l.ring, p0, s);
        for (int t0 = 0; t0 < T;) {
            const size_t s_n = tn.s_use > 0 ? std::min(tn.s_use, gp.S_n) : gp.S_n;
            int ts = std::min(tn.ts_use > 0 ? std::min(tn.ts_use, gp.ts_max) : gp.ts_max, T - t0);
            int64_t pf = p0 + t0, klo = l.swa ? std::max<int64_t>(0, pf - (g.n_swa - 1)) : 0, nkr = pf + ts - klo;
            while (ts > 1 && (size_t) H * ts * (size_t) nkr > s_n) {
                ts /= 2;
                nkr = pf + ts - klo;
            }
            if ((size_t) H * ts * (size_t) nkr > s_n) {
                err = "layer " + std::to_string(l.il) + ": the attention's scores do not fit (" + std::to_string(nkr) + " keys)";
                return false;
            }
            const int Q = per * ts, nk = (int) nkr;
            k::perm_heads(gp.c_q + (size_t) t0 * HD, gp.c_qp, ts, Hkv, per, D, true, s);
            bool ok = cublasSgemm(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, extent, H * ts, g.d_rel, &one, l.rel_bias, g.d_rel,
                                  gp.c_rel + (size_t) t0 * H * g.d_rel, g.d_rel, &zero, gp.c_tbl, extent) == CUBLAS_STATUS_SUCCESS;
            const size_t slot0 = (size_t) (l.ring > 0 ? klo % l.ring : klo);
            const int n1 = l.ring > 0 ? (int) std::min<int64_t>(nk, (int64_t) l.ring - (int64_t) slot0) : nk, n2 = nk - n1;
            // (the caches hold F16: the queries and the probabilities in F16 beside them, F32 sums)
            auto bg = [&](decltype(CUBLAS_OP_T) ta, int m, int kk, const uint16_t* A, const uint16_t* B, int ldb, long long sb,
                          const float* beta, float* C, int ldc, long long sc) {
                return cublasGemmStridedBatchedEx(gp.blas, ta, CUBLAS_OP_N, m, Q, kk, &one, A, CUDA_R_16F, kvd, D, B,
                                                  CUDA_R_16F, ldb, sb, beta, C, CUDA_R_32F, ldc, sc, Hkv, CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
            };
            hk::to_f16(gp.c_qp, (size_t) D, Hkv * Q, D, gp.xh16, (size_t) D, s);
            ok = ok && bg(CUBLAS_OP_T, n1, D, l.K + slot0 * kvd, gp.xh16, D, (long long) Q * D, &zero, gp.c_S, nk,
                          (long long) Q * nk);
            if (n2 > 0)
                ok = ok && bg(CUBLAS_OP_T, n2, D, l.K, gp.xh16, D, (long long) Q * D, &zero, gp.c_S + n1, nk, (long long) Q * nk);
            k::attn_softmax_rows(gp.c_S, ts, Hkv, per, nk, klo, pf, window, extent, gp.c_tbl, gp.c_sink + (size_t) t0 * H,
                                 1.0f / std::sqrt((float) D), s);
            hk::to_f16(gp.c_S, (size_t) nk, Hkv * Q, nk, gp.c_P16, (size_t) nk, s);
            ok = ok && bg(CUBLAS_OP_N, D, n1, l.V + slot0 * kvd, gp.c_P16, nk, (long long) Q * nk, &zero, gp.c_op, D,
                          (long long) Q * D);
            if (n2 > 0)
                ok = ok && bg(CUBLAS_OP_N, D, n2, l.V, gp.c_P16 + n1, nk, (long long) Q * nk, &one, gp.c_op, D, (long long) Q * D);
            if (!ok) {
                err = "layer " + std::to_string(l.il) + ": an attention GEMM failed";
                return false;
            }
            k::perm_heads(gp.c_op, gp.c_ao + (size_t) t0 * HD, ts, Hkv, per, D, false, s);
            t0 += ts;
        }
        return true;
    }

    // ds4_iq's expert sets over the chunk's entries whose expert has a slot in `slot` (a device map, n_slots blobs at
    // wbase): gp.c_y = (accumulate: +=) each token's weighted sum of them.  max_rows: the most entries an expert has.
    bool sets(Layer& l, Gpu& gp, int T, const int* slot, const uint8_t* wbase, int n_slots, int max_rows, bool accumulate,
              std::string& err) {
        const int E = g.n_embd, ff = g.n_ff_exp, nu = g.n_used;
        cudaStream_t s = gp.s;
        hk::moe_sort(gp.c_ids, T, nu, g.n_expert, gp.c_bounds, gp.c_stok, gp.c_srow, s, slot, nullptr, n_slots);
        (void) accumulate;
        if (n_slots == 0 || max_rows == 0) return true;
        iq::SetArgs a;
        a.wbase = wbase;
        a.stride = l.blob;
        a.tr[0] = l.tr[0];
        a.tr[1] = l.tr[1];
        a.tr[2] = l.tr[2];
        a.bounds = gp.c_bounds;
        a.row0 = 0;
        a.max_rows = max_rows;
        a.n_exp = n_slots;
        a.n_embd = E;
        a.n_ff = ff;
        a.xf = gp.c_xf;
        a.idx = gp.c_stok;
        a.ldx = E;
        a.h = gp.c_sh16;
        a.ldo = ff;
        a.tg = l.tg;
        a.td = l.td;
        a.act = iq::kActSitu;
        a.situ_beta = g.situ_beta;
        a.situ_lb = g.situ_lb;
        if (!iq::set_gate_up(a, s)) {
            err = "layer " + std::to_string(l.il) + ": the experts' gate / up sets did not run";
            return false;
        }
        iq::SetArgs d = a;
        d.xf = nullptr;
        d.idx = nullptr;
        d.xh = gp.c_sh16;
        d.ldx = ff;
        d.h = nullptr;
        d.y = gp.c_sy;
        d.ldo = E;
        if (!iq::set_down(d, s)) {
            err = "layer " + std::to_string(l.il) + ": the experts' down sets did not run";
            return false;
        }
        k::scatter_rows(gp.c_sy, gp.c_srow, T * nu, E, gp.c_ent, s);   // (each entry's output: experts_rows sums them)
        return true;
    }

    // The chunk's routed experts (gp.c_xf -> gp.c_y): the router, the VRAM experts' sets, then the RAM experts the
    // chunk routes to streamed through the two staging buffers - a pass copied (the side stream, straight from the
    // registered RAM) while the pass before it computes
    bool experts_rows(Layer& l, Gpu& gp, int T, int64_t p0, std::string& err) {
        const int nu = g.n_used, ne = g.n_expert;
        cudaStream_t s = gp.s;
        if (!gemm(gp, l.router, gp.c_xf, T, gp.c_rl)) {
            err = "layer " + std::to_string(l.il) + ": the router's GEMM failed";
            return false;
        }
        hk::router(gp.c_rl, T, ne, l.router_b, nu, true, g.w_scale, gp.c_ids, gp.c_rw, s);
        CK(cudaMemcpyAsync(gp.hc_ids, gp.c_ids, sizeof(int) * T * nu, cudaMemcpyDeviceToHost, s));
        if (tapf || pick_hook) {
            CK(cudaMemcpyAsync(gp.hc_rw, gp.c_rw, sizeof(float) * T * nu, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(gp.hc_rl, gp.c_rl, sizeof(float) * T * ne, cudaMemcpyDeviceToHost, s));
        }
        CK(cudaStreamSynchronize(s));
        if (tapf) {
            std::vector<float> sc((size_t) T * ne);
            for (int t = 0; t < T; ++t)
                for (int e = 0; e < ne; ++e)
                    sc[(size_t) t * ne + e] = 1.0f / (1.0f + std::exp(-gp.hc_rl[(size_t) t * ne + e])) +
                                              (l.h_bias.empty() ? 0.0f : l.h_bias[(size_t) e]);
            tapf("ffn_moe_scores", l.il, sc.data(), sc.size(), p0, T);
            std::vector<int32_t> pw((size_t) 2 * T * nu);
            std::memcpy(pw.data(), gp.hc_ids, sizeof(int) * T * nu);
            std::memcpy(pw.data() + (size_t) T * nu, gp.hc_rw, sizeof(float) * T * nu);
            tapf("ffn_moe_topk", l.il, pw.data(), (size_t) T * nu, p0, T);
        }
        if (pick_hook && pick_hook(l.il, gp.hc_ids, nu, p0, T)) {   // (the hook's picks: the router's weights for them)
            for (int t = 0; t < T; ++t) {
                float sum = 0.0f;
                int* id = gp.hc_ids + (size_t) t * nu;
                float* w = gp.hc_rw + (size_t) t * nu;
                for (int j = 0; j < nu; ++j) {
                    if (id[j] < 0 || id[j] >= ne) {
                        err = "layer " + std::to_string(l.il) + ": the pick hook named expert " + std::to_string(id[j]);
                        return false;
                    }
                    w[j] = 1.0f / (1.0f + std::exp(-gp.hc_rl[(size_t) t * ne + id[j]]));
                    sum += w[j];
                }
                for (int j = 0; j < nu; ++j) w[j] = w[j] / (sum + 1e-20f) * g.w_scale;
            }
            CK(cudaMemcpyAsync(gp.c_ids, gp.hc_ids, sizeof(int) * T * nu, cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(gp.c_rw, gp.hc_rw, sizeof(float) * T * nu, cudaMemcpyHostToDevice, s));
        }
        // each expert's entries; the most among the VRAM ones
        std::vector<int> cnt((size_t) ne, 0);
        uint32_t* prof = prof_cnt.data() + (size_t) l.il * ne;
        for (int i = 0; i < T * nu; ++i) {
            const int e = gp.hc_ids[i];
            if (e < 0 || e >= ne) {
                err = "layer " + std::to_string(l.il) + ": the router picked expert " + std::to_string(e);
                return false;
            }
            ++cnt[(size_t) e];
            ++prof[e];
        }
        int mr = 0;
        std::vector<int> u;   // (the RAM experts routed to, in their RAM slots' order)
        for (int e = 0; e < ne; ++e) l.use[(size_t) e] += (float) cnt[(size_t) e] * std::min(1.0f, tn.prompt_rows / (float) T);
        for (int e = 0; e < ne; ++e) {
            if (l.slot_of[(size_t) e] >= 0) mr = std::max(mr, cnt[(size_t) e]);
            else if (cnt[(size_t) e] > 0) u.push_back(e);
        }
        if (!sets(l, gp, T, l.slot_d, l.experts, l.n_res + 1, mr, false, err)) return false;
        const size_t stg = (size_t) (tn.stage_use > 0 ? std::min(tn.stage_use, kStage) : kStage);
        for (size_t q0 = 0, pass = 0; q0 < u.size(); q0 += stg, ++pass) {
            const int np = (int) std::min(u.size() - q0, stg), b = (int) (pass & 1);
            int* pm = gp.hc_pmap + pass * (size_t) ne;
            for (int e = 0; e < ne; ++e) pm[e] = -1;
            int pmr = 0;
            if (pass >= 2) CK(cudaStreamWaitEvent(gp.s2, gp.ev_done[b], 0));   // (the buffer's earlier pass has run)
            std::vector<std::pair<int, uint8_t*>> fj;   // (the pass's experts in the files: gathered, then copied)
            for (int i = 0; i < np;) {   // (runs of neighbours in RAM: a copy each)
                const int r0 = l.ram_of[(size_t) u[q0 + i]];
                int run = 1;
                if (r0 < l.n_copy) {
                    while (i + run < np && l.ram_of[(size_t) u[q0 + i + run]] == r0 + run && r0 + run < l.n_copy) ++run;
                    CK(cudaMemcpyAsync(gp.xstage[b] + (size_t) i * l.blob, l.ram + (size_t) r0 * l.blob,
                                       (size_t) run * l.blob, cudaMemcpyHostToDevice, gp.s2));
                } else {
                    fj.push_back({u[q0 + i], gp.h_fst + (size_t) i * l.blob});
                }
                for (int r = 0; r < run; ++r) {
                    pm[u[q0 + i + r]] = i + r;
                    pmr = std::max(pmr, cnt[(size_t) u[q0 + i + r]]);
                }
                i += run;
            }
            if (!fj.empty()) {
                if (!gp.h_fst) {
                    err = "layer " + std::to_string(l.il) + ": experts in the files without the gather buffer";
                    return false;
                }
                if (gp.fst_busy) CK(cudaEventSynchronize(gp.ev_fst));   // (its last copy out has run)
                file_gather(l, fj);
                for (size_t j = 0; j < fj.size();) {   // (runs of neighbouring slots: a copy each)
                    size_t run = 1;
                    while (j + run < fj.size() && fj[j + run].second == fj[j].second + run * l.blob) ++run;
                    const size_t i = (size_t) (fj[j].second - gp.h_fst) / l.blob;
                    CK(cudaMemcpyAsync(gp.xstage[b] + i * l.blob, fj[j].second, run * l.blob, cudaMemcpyHostToDevice, gp.s2));
                    j += run;
                }
                CK(cudaEventRecord(gp.ev_fst, gp.s2));
                gp.fst_busy = true;
            }
            CK(cudaMemcpyAsync(gp.pmap[b], pm, sizeof(int) * ne, cudaMemcpyHostToDevice, gp.s2));
            CK(cudaEventRecord(gp.ev_copy[b], gp.s2));
            CK(cudaStreamWaitEvent(s, gp.ev_copy[b], 0));
            if (!sets(l, gp, T, gp.pmap[b], gp.xstage[b], np, pmr, true, err)) return false;
            CK(cudaEventRecord(gp.ev_done[b], s));
        }
        if (u.size() > (size_t) g.n_expert) {
            err = "layer " + std::to_string(l.il) + ": more staging passes than the maps hold";
            return false;
        }
        k::combine_rows(gp.c_ent, gp.c_rw, T, nu, g.n_embd, gp.c_y, s);
        gp.n_streamed += (int64_t) u.size();
        gp.n_experts += (int64_t) T * nu;
        return true;
    }

    bool layer_rows(Layer& l, Gpu& gp, int T, int64_t p0, int& nb, std::string& err) {
        const int E = g.n_embd, H = g.n_head, Hkv = g.n_head_kv, D = g.head_dim, HD = H * D, kvd = Hkv * D;
        const size_t TE = (size_t) T * E, ldb = (size_t) Tc * E;
        cudaStream_t s = gp.s;
        const float* src = gp.c_cur;
        if (l.il > 0) {
            k::mix_rows(gp.c_blocks, ldb, nb, gp.c_cur, T, E, l.attn_res, l.swa ? 1 : 3, 1e-6f, gp.c_stat, gp.c_mixo, s);
            src = gp.c_mixo;
        }
        const float *xq = gp.c_xn, *xk = gp.c_xn, *xv = gp.c_xn;
        bool ok = true;
        if (l.swa) {
            k::rms(src, (size_t) E, T, E, l.attn_norm, g.eps, gp.c_xn, (size_t) E, s);
            ok = gemm(gp, l.q, xq, T, gp.c_q) && gemm(gp, l.gate, xq, T, gp.c_gate, false) &&
                 gemm(gp, l.rel, xq, T, gp.c_rel, false) && gemm(gp, l.wk, xq, T, gp.c_ky, false) &&
                 gemm(gp, l.wv, xq, T, gp.c_vy, false);
        } else {
            const size_t st1 = l.il > 0 ? TE : 0;
            k::rms(src, (size_t) E, T, E, l.q_norm, g.eps, gp.c_xn, (size_t) E, s);
            k::rms(src + st1, (size_t) E, T, E, l.k_norm, g.eps, gp.c_xn + TE, (size_t) E, s);
            k::rms(src + 2 * st1, (size_t) E, T, E, l.v_norm, g.eps, gp.c_xn + 2 * TE, (size_t) E, s);
            xk = gp.c_xn + TE;
            xv = gp.c_xn + 2 * TE;
            ok = gemm(gp, l.q, xq, T, gp.c_q) && gemm(gp, l.gate, xq, T, gp.c_gate, false) &&
                 gemm(gp, l.rel, xq, T, gp.c_rel, false) && gemm(gp, l.wk, xk, T, gp.c_ky) && gemm(gp, l.wv, xv, T, gp.c_vy) &&
                 gemm(gp, l.k_proj, xk, T, gp.c_dk) && gemm(gp, l.v_proj, xv, T, gp.c_dv);
        }
        ok = ok && gemm(gp, l.sink, xq, T, gp.c_sink);
        if (!ok) {
            err = "layer " + std::to_string(l.il) + ": an attention GEMM failed";
            return false;
        }
        k::conv_rows(gp.c_ky, T, l.hist_k, l.k_conv, kvd, l.swa ? nullptr : gp.c_dk, Hkv, gp.c_kc, s);
        k::conv_rows(gp.c_vy, T, l.hist_v, l.v_conv, kvd, l.swa ? nullptr : gp.c_dv, Hkv, gp.c_vc, s);
        const float alpha = l.swa ? 0.0f : g.log_alpha;
        k::head_clip_rows(gp.c_q, T, H, D, l.q_clip, g.qk_clip, alpha, g.log_floor, p0, s);
        k::head_clip_rows(gp.c_kc, T, Hkv, D, nullptr, g.qk_clip, 0.0f, 1.0, p0, s);
        k::rms(gp.c_vc, (size_t) D, T * Hkv, D, nullptr, g.eps, gp.c_vc, (size_t) D, s);
        k::head_clip_rows(gp.c_rel, T, H, g.d_rel, l.rel_clip, g.rel_clip, alpha, g.log_floor, p0, s);
        if (!tap("Qcur", l.il, gp, gp.c_q, (size_t) T * HD, err) || !tap("Kcur", l.il, gp, gp.c_kc, (size_t) T * kvd, err) ||
            !tap("Vcur", l.il, gp, gp.c_vc, (size_t) T * kvd, err) ||
            !tap("Rcur", l.il, gp, gp.c_rel, (size_t) T * H * g.d_rel, err))
            return false;
        if (!attention_rows(l, gp, T, p0, err)) return false;
        if (!tap("kqv_out", l.il, gp, gp.c_ao, (size_t) T * HD, err)) return false;
        k::sigmoid_mul(gp.c_ao, gp.c_gate, (size_t) T * HD, s);
        if (!gemm(gp, l.wo, gp.c_ao, T, gp.c_o)) {
            err = "layer " + std::to_string(l.il) + ": the attention's output GEMM failed";
            return false;
        }
        k::conv_rows(gp.c_o, T, l.hist_a, l.a_conv, E, nullptr, 1, gp.c_oc, s);
        if (!tap("attn_out", l.il, gp, gp.c_oc, TE, err)) return false;
        if (l.il % g.res_block == 0) {
            CK(cudaMemcpyAsync(gp.c_blocks + (size_t) nb * ldb, gp.c_cur, sizeof(float) * TE, cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(gp.c_cur, gp.c_oc, sizeof(float) * TE, cudaMemcpyDeviceToDevice, s));
            ++nb;
        } else {
            k::add(gp.c_cur, gp.c_oc, TE, s);
        }
        if (!tap("ffn_inp", l.il, gp, gp.c_cur, TE, err)) return false;
        k::mix_rows(gp.c_blocks, ldb, nb, gp.c_cur, T, E, l.ffn_res, 1, 1e-6f, gp.c_stat, gp.c_mixo, s);
        k::rms(gp.c_mixo, (size_t) E, T, E, l.ffn_norm, g.eps, gp.c_xf, (size_t) E, s);
        if (!experts_rows(l, gp, T, p0, err)) return false;
        k::conv_rows(gp.c_y, T, l.hist_f, l.f_conv, E, nullptr, 1, gp.c_yc, s);
        if (!tap("ffn_out", l.il, gp, gp.c_yc, TE, err)) return false;
        k::add(gp.c_cur, gp.c_yc, TE, s);
        return tap("l_out", l.il, gp, gp.c_cur, TE, err);
    }

    // GPU gi's stage of a chunk: its input (the embeddings, or the stage before it's hand-off in its slot), its layers,
    // the hand-off into the next GPU's slot - or, the last GPU, the chunk's last token into the decode buffers for the
    // head.  A stage waits for its slot to fill and for the next one to empty: the GPUs run the chunks as a pipeline.
    bool stage(size_t gi, const int* ids, int T, int64_t p0, int slot, int64_t done, int ci, std::string& err) {
        const size_t E = (size_t) g.n_embd, TE = (size_t) T * E, ldb = (size_t) Tc * E;
        Gpu& gp = G[gi];
        CK(cudaSetDevice(gp.dev));
        if (abort_run.load()) return true;
        if (slot >= 0 && !snap_copy(gp, slot, true, err)) return false;   // (its layers' histories before the chunk)
        int nb = 0;
        Slot& in = *slots[gi];
        if (gi == 0) {
            for (int t = 0; t < T; ++t) {
                if (ids[t] < 0 || ids[t] >= g.n_vocab) {
                    err = "token " + std::to_string(ids[t]) + " is outside the vocabulary";
                    return false;
                }
                const uint8_t* row = tok_embd.data() + (size_t) ids[t] * tok_row;
                float* dst = in.h + (size_t) t * E;
                if (tok_type == 13) dequantize_row_q5_K(row, dst, (int64_t) E);
                else if (tok_type == 14) dequantize_row_q6_K(row, dst, (int64_t) E);
                else std::memcpy(dst, row, E * 4);
            }
            CK(cudaMemcpyAsync(gp.c_o, in.h, sizeof(float) * TE, cudaMemcpyHostToDevice, gp.s));
            k::rms(gp.c_o, E, T, (int) E, tok_norm, g.eps, gp.c_cur, E, gp.s);
            CK(cudaStreamSynchronize(gp.s));
        } else {
            {
                std::unique_lock<std::mutex> lk(in.mu);
                in.cv.wait(lk, [&] { return in.full || abort_run.load(); });
                if (abort_run.load()) return true;
                nb = in.nb;
            }
            CK(cudaMemcpyAsync(gp.c_cur, in.h, sizeof(float) * TE, cudaMemcpyHostToDevice, gp.s));
            for (int b = 0; b < nb; ++b)
                CK(cudaMemcpyAsync(gp.c_blocks + (size_t) b * ldb, in.h + (size_t) (1 + b) * TE, sizeof(float) * TE,
                                   cudaMemcpyHostToDevice, gp.s));
            CK(cudaStreamSynchronize(gp.s));
            {
                std::lock_guard<std::mutex> lk(in.mu);
                in.full = false;
            }
            in.cv.notify_all();
        }
        const bool taps = taps_on && !tapf && !pick_hook;
        float* ptap = taps && !p_taps.empty() ? p_taps[(size_t) ci % p_taps.size()] : nullptr;
        const size_t TcE = (size_t) Tc * E;
        for (int il = gp.l0; il < gp.l1; ++il) {
            if (!layer_rows(L[(size_t) il], gp, T, p0, nb, err)) return false;
            if (taps && tap_of[(size_t) il] >= 0) {   // (the drafter's tap: to the host - the last GPU's stay on it)
                const size_t kt = (size_t) tap_of[(size_t) il];
                if (gi + 1 == G.size())
                    CK(cudaMemcpyAsync(gp.c_tapl + kt * TcE, gp.c_cur, sizeof(float) * TE, cudaMemcpyDeviceToDevice, gp.s));
                else CK(cudaMemcpyAsync(ptap + kt * TcE, gp.c_cur, sizeof(float) * TE, cudaMemcpyDeviceToHost, gp.s));
            }
        }
        if (gi + 1 < G.size()) {
            Slot& out = *slots[gi + 1];
            {
                std::unique_lock<std::mutex> lk(out.mu);
                out.cv.wait(lk, [&] { return !out.full || abort_run.load(); });
                if (abort_run.load()) return true;
            }
            CK(cudaMemcpyAsync(out.h, gp.c_cur, sizeof(float) * TE, cudaMemcpyDeviceToHost, gp.s));
            for (int b = 0; b < nb; ++b)
                CK(cudaMemcpyAsync(out.h + (size_t) (1 + b) * TE, gp.c_blocks + (size_t) b * ldb, sizeof(float) * TE,
                                   cudaMemcpyDeviceToHost, gp.s));
            CK(cudaStreamSynchronize(gp.s));
            {
                std::lock_guard<std::mutex> lk(out.mu);
                out.full = true;
                out.nb = nb;
            }
            out.cv.notify_all();
        } else {   // (the last token's rows where the head reads them)
            CK(cudaMemcpyAsync(gp.cur, gp.c_cur + (size_t) (T - 1) * E, sizeof(float) * E, cudaMemcpyDeviceToDevice, gp.s));
            for (int b = 0; b < nb; ++b)
                CK(cudaMemcpyAsync(gp.blocks + (size_t) b * E, gp.c_blocks + (size_t) b * ldb + (size_t) (T - 1) * E,
                                   sizeof(float) * E, cudaMemcpyDeviceToDevice, gp.s));
            if (taps && ds->end() == p0) {   // (every row's taps to the drafter: the head's input for each of them too)
                const int nt = (int) tap_layers.size();
                k::mix_rms_rows(gp.c_blocks, ldb, nb, gp.c_cur, T, (int) E, out_res, 1, out_norm, nullptr, nullptr, g.eps,
                                gp.c_xf, gp.s);
                const float* srcs[8];
                for (int kt = 0; kt + 1 < nt; ++kt) {
                    const bool mine = L[(size_t) tap_layers[(size_t) kt]].gpu == (int) gi;
                    if (!mine)
                        CK(cudaMemcpyAsync(gp.c_tapr + (size_t) kt * TcE, ptap + (size_t) kt * TcE, sizeof(float) * TE,
                                           cudaMemcpyHostToDevice, gp.s));
                    srcs[kt] = (mine ? gp.c_tapl : gp.c_tapr) + (size_t) kt * TcE;
                }
                srcs[nt - 1] = gp.c_xf;
                k::taps_gather(srcs, 0u, nt, T, (int) E, gp.c_taps, (size_t) nt * E, gp.s);
                CK(cudaStreamSynchronize(gp.s));
                const double tf = now_s();
                if (!ds->feed_dev(gp.c_taps, (size_t) nt * E, T, p0, err)) return false;
                CK(cudaSetDevice(gp.dev));
                CK(cudaDeviceSynchronize());   // (the drafter has read the chunk's rows: the next chunk may write them)
                ds_feed_s += now_s() - tf;
            }
            CK(cudaStreamSynchronize(gp.s));
            last_nb = nb;
            if (progress && !progress(done)) {   // (the caller cancels)
                abort_run = true;
                for (auto& sl : slots) {
                    std::lock_guard<std::mutex> lk(sl->mu);
                    sl->cv.notify_all();
                }
            }
        }
        return true;
    }
    int last_nb = 0;

    // ---- the expert cache: decayed routings a (layer, expert); between two steps each GPU's idle mover is given the
    // layer whose most-routed RAM expert beats its least-routed VRAM one by the most ------------------------------
    void swap_run(size_t gi) {
        Gpu& gp = G[gi];
        cudaSetDevice(gp.dev);
        for (;;) {
            int stt;
            {
                std::unique_lock<std::mutex> lk(gp.sw_mu);
                gp.sw_cv.wait(lk, [&] { return gp.sw_quit || gp.sw_state == 1 || gp.sw_state == 3; });
                if (gp.sw_quit) return;
                stt = gp.sw_state;
            }
            if (stt == 1) {
                bool ok = true;
                size_t q = 0;
                for (const Gpu::Move& mv : gp.sw_jobs) {
                    Layer& l = L[(size_t) mv.il];
                    const int vs = l.slot_of[(size_t) mv.cold];
                    uint8_t* bounce = gp.h_swap + q++ * gp.blob_max;
                    if (mv.rs >= l.n_copy) {   // (in the files: in through the bounce; cold's bytes stay in the files)
                        file_blob(l, mv.hot, bounce);
                        ok = ok && cudaMemcpyAsync(l.experts + (size_t) l.spare * l.blob, bounce, l.blob,
                                                   cudaMemcpyHostToDevice, gp.s3) == cudaSuccess;
                        continue;
                    }
                    ok = ok && cudaMemcpyAsync(l.experts + (size_t) l.spare * l.blob, l.ram + (size_t) mv.rs * l.blob, l.blob,
                                               cudaMemcpyHostToDevice, gp.s3) == cudaSuccess &&
                         cudaMemcpyAsync(bounce, l.experts + (size_t) vs * l.blob, l.blob,
                                         cudaMemcpyDeviceToHost, gp.s3) == cudaSuccess;
                }
                ok = ok && cudaStreamSynchronize(gp.s3) == cudaSuccess;
                std::lock_guard<std::mutex> lk(gp.sw_mu);
                gp.sw_state = ok ? 2 : 0;   // (a failed copy: nothing changed, the moves are dropped)
            } else {
                size_t q = 0;
                for (const Gpu::Move& mv : gp.sw_jobs) {
                    Layer& l = L[(size_t) mv.il];
                    const uint8_t* bounce = gp.h_swap + q++ * gp.blob_max;
                    if (mv.rs < l.n_copy) std::memcpy(l.ram + (size_t) mv.rs * l.blob, bounce, l.blob);
                    else {   // (the RAM slot now names cold's place in the files)
                        l.cex.fgu[(size_t) (mv.rs - l.n_copy)] = pk.data(*l.t_gu) + (size_t) mv.cold * l.gub;
                        l.cex.fdn[(size_t) (mv.rs - l.n_copy)] = pk.data(*l.t_dn) + (size_t) mv.cold * l.db;
                    }
                }
                std::lock_guard<std::mutex> lk(gp.sw_mu);
                gp.sw_state = 4;
            }
            gp.sw_cv.notify_all();
        }
    }
    bool swap_on() const { return o.expert_cache && !tapf && !pick_hook && st.ram_gb + st.file_gb > 0; }
    // the decode thread, between two steps (no layer runs): the finished copies' maps, the next move
    bool swap_poll(std::string& err, int batch = 0) {
        if (!swap_on()) return true;
        if (batch <= 0) batch = swap_batch;
        batch = std::min(batch, 16);   // (the movers' bounce holds 16 blobs)
        for (Gpu& gp : G) {
            if (!gp.sw_thread.joinable()) continue;
            std::unique_lock<std::mutex> lk(gp.sw_mu);
            const int was = gp.sw_state;
            if (was == 2 || was == 4) {
                CK(cudaSetDevice(gp.dev));
                for (const Gpu::Move& mv : gp.sw_jobs) {
                    Layer& l = L[(size_t) mv.il];
                    if (was == 2) {
                        l.slot_of[(size_t) mv.hot] = l.spare;
                        l.ram_of[(size_t) mv.hot] = -1;
                    } else {
                        l.spare = l.slot_of[(size_t) mv.cold];
                        l.slot_of[(size_t) mv.cold] = -1;
                        l.ram_of[(size_t) mv.cold] = mv.rs;
                        ++st.swaps;
                    }
                    // (the device's map: the prompt path's sets and the host-free step's plan)
                    CK(cudaMemcpy(l.slot_d, l.slot_of.data(), sizeof(int) * (size_t) g.n_expert, cudaMemcpyHostToDevice));
                }
                gp.sw_state = was == 2 ? 3 : 0;
            }
            if (gp.sw_state == 0) {
                std::vector<std::pair<float, Gpu::Move>> c;
                for (int il = gp.l0; il < gp.l1; ++il) {
                    Layer& l = L[(size_t) il];
                    if (l.n_res == 0 || l.n_res >= g.n_expert) continue;   // (no VRAM slot, or no expert outside)
                    int hot = -1, cold = -1;
                    for (int e = 0; e < g.n_expert; ++e) {
                        const float u = l.use[(size_t) e];
                        if (l.slot_of[(size_t) e] < 0) {
                            if (hot < 0 || u > l.use[(size_t) hot]) hot = e;
                        } else if (cold < 0 || u < l.use[(size_t) cold]) {
                            cold = e;
                        }
                    }
                    if (hot < 0 || cold < 0) continue;
                    const float gain = l.use[(size_t) hot] - tn.cache_margin * l.use[(size_t) cold] - tn.cache_bonus;
                    if (gain > 0.0f) c.push_back({gain, {il, hot, cold, l.ram_of[(size_t) hot]}});
                }
                std::sort(c.begin(), c.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
                gp.sw_jobs.clear();
                for (size_t i = 0; i < c.size() && (int) i < batch; ++i) gp.sw_jobs.push_back(c[i].second);
                if (!gp.sw_jobs.empty()) gp.sw_state = 1;
            }
            lk.unlock();
            gp.sw_cv.notify_all();
        }
        return true;
    }
    // after a prompt: the movers run whole batches until the prompt's hottest RAM experts are in VRAM (a few rounds:
    // the answer's first tokens find them there)
    bool swap_burst(std::string& err) {
        if (!swap_on()) return true;
        const double t0 = now_s();
        for (int round = 0; round < tn.burst_rounds && now_s() - t0 < burst_s; ++round) {
            if (!swap_poll(err, tn.burst_moves)) return false;
            bool any = false;
            for (Gpu& gp : G) {
                if (!gp.sw_thread.joinable()) continue;
                std::unique_lock<std::mutex> lk(gp.sw_mu);
                any = any || gp.sw_state != 0;
                gp.sw_cv.wait(lk, [&] { return gp.sw_state == 0 || gp.sw_state == 2 || gp.sw_state == 4; });
            }
            if (!any) break;
        }
        return true;
    }
    int swap_batch = 4;        // moves a GPU a round between two decode steps (IK2_HIP_SWAP_BATCH; measured: 4 and 8
                               // decode alike, the CPU lane at 2 - 3 % of the picks)
    double burst_s = 0.0;      // a prompt's burst at most, seconds (IK2_HIP_SWAP_BURST_MS; off: the first prompt after
                               // a load was seen to take 5 s longer with it - not understood yet)
    void use_decay(float f) {
        for (Layer& l : L)
            for (float& c : l.use) c *= f;
    }

    // ---- rewinds: the histories saved at positions (a ring of kSnaps slots, the same slot in every layer) -------------
    // (96 x 64 = 6144 decoded positions, more than the 4096 a rewind reaches; the cadence, tn.snap_every: every 16 /
    //  64 / 256 decoded tokens alike, measured)
    static constexpr int kSnaps = 96;
    int64_t snap_pos[kSnaps];
    int snap_head = 0;
    int64_t hw = 0;          // the furthest position ever read since the reset: the rings' rows before hw - ring are gone
    bool snap_next = true;   // the next decoded token's position is saved whatever it is (a prompt's end)
    std::function<bool(int64_t)> progress;
    bool was_cancelled = false;
    size_t hist_n() const { return (size_t) 3 * (2 * (size_t) g.n_head_kv * g.head_dim + 2 * (size_t) g.n_embd); }
    int snap_alloc(int64_t pos) {
        for (int i = 0; i < kSnaps; ++i)
            if (snap_pos[i] == pos) return i;
        const int i = snap_head;
        snap_head = (snap_head + 1) % kSnaps;
        snap_pos[i] = pos;
        return i;
    }
    // GPU gp's layers' histories <-> slot (queued on its stream)
    bool snap_copy(Gpu& gp, int slot, bool save, std::string& err) {
        const size_t kvd = (size_t) g.n_head_kv * g.head_dim, E = (size_t) g.n_embd;
        for (int il = gp.l0; il < gp.l1; ++il) {
            Layer& l = L[(size_t) il];
            float* b = l.snap + (size_t) slot * hist_n();
            float* h[4] = {l.hist_k, l.hist_v, l.hist_a, l.hist_f};
            const size_t n[4] = {3 * kvd, 3 * kvd, 3 * E, 3 * E};
            for (int q = 0; q < 4; ++q) {
                CK(cudaMemcpyAsync(save ? b : h[q], save ? h[q] : b, n[q] * 4, cudaMemcpyDeviceToDevice, gp.s));
                b += n[q];
            }
        }
        return true;
    }
    // ---- (--parallel, Options::seqs) several conversations: each its own caches, histories, saved histories and
    // drafter state.  The bound one's are in the layers and the fields above (every single-conversation path - a
    // prompt, a step, a window, keep, rewind, reset - works on it); the others' wait in seqs.  A multi step (step_multi)
    // parks the bound one too and takes each row's from its own -------------------------------------------------------
    struct SeqLayer {
        uint16_t *K = nullptr, *V = nullptr;
        float *hist_k = nullptr, *hist_v = nullptr, *hist_a = nullptr, *hist_f = nullptr;
        float *alt_k = nullptr, *alt_v = nullptr, *alt_a = nullptr, *alt_f = nullptr;
        float* snap = nullptr;
    };
    struct Seq {
        std::vector<SeqLayer> L;
        int64_t n_end = 0, hw = 0;
        int64_t snap_pos[kSnaps];
        int snap_head = 0;
        bool snap_next = true;
    };
    std::vector<Seq> seqs;
    int bound_seq = 0;
    void seq_save(Seq& q) {
        q.L.resize(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            const Layer& l = L[i];
            q.L[i] = {l.K, l.V, l.hist_k, l.hist_v, l.hist_a, l.hist_f, l.alt_k, l.alt_v, l.alt_a, l.alt_f, l.snap};
        }
        q.n_end = n_end;
        q.hw = hw;
        std::memcpy(q.snap_pos, snap_pos, sizeof snap_pos);
        q.snap_head = snap_head;
        q.snap_next = snap_next;
    }
    void seq_load(const Seq& q) {
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            const SeqLayer& x = q.L[i];
            l.K = x.K;
            l.V = x.V;
            l.hist_k = x.hist_k;
            l.hist_v = x.hist_v;
            l.hist_a = x.hist_a;
            l.hist_f = x.hist_f;
            l.alt_k = x.alt_k;
            l.alt_v = x.alt_v;
            l.alt_a = x.alt_a;
            l.alt_f = x.alt_f;
            l.snap = x.snap;
        }
        n_end = q.n_end;
        hw = q.hw;
        std::memcpy(snap_pos, q.snap_pos, sizeof snap_pos);
        snap_head = q.snap_head;
        snap_next = q.snap_next;
    }
    // the other conversations' state on GPU gp (before the experts are placed: they take what is left), zeros - as
    // load_dense and win_scratch allocate the first one's
    bool seq_alloc(Gpu& gp, std::string& err) {
        const size_t E = (size_t) g.n_embd, kvd = (size_t) g.n_head_kv * g.head_dim;
        for (size_t si = 1; si < seqs.size(); ++si)
            for (int il = gp.l0; il < gp.l1; ++il) {
                SeqLayer& q = seqs[si].L[(size_t) il];
                const size_t rows = cache_rows(il);
                float* h = nullptr;
                if (!zeros(gp, h, 2 * 3 * (2 * kvd + 2 * E), err) || !zeros(gp, q.K, rows * kvd, err) ||
                    !zeros(gp, q.V, rows * kvd, err) || !gp.alloc(q.snap, (size_t) kSnaps * hist_n(), err))
                    return false;
                q.hist_k = h;
                q.hist_v = q.hist_k + 3 * kvd;
                q.hist_a = q.hist_v + 3 * kvd;
                q.hist_f = q.hist_a + 3 * E;
                q.alt_k = q.hist_f + 3 * E;
                q.alt_v = q.alt_k + 3 * kvd;
                q.alt_a = q.alt_v + 3 * kvd;
                q.alt_f = q.alt_a + 3 * E;
            }
        return true;
    }
    // conversation s's state from here on (the bound one's pending rows fed to its drafter, its counts kept first)
    bool bind(int s, std::string& err) {
        if (s < 0 || s >= (int) seqs.size()) {
            err = "no conversation " + std::to_string(s) + " (--parallel " + std::to_string(seqs.size()) + ")";
            return false;
        }
        if (s == bound_seq) return true;
        if (!taps_flush(err)) return false;
        seq_save(seqs[(size_t) bound_seq]);
        seq_load(seqs[(size_t) s]);
        bound_seq = s;
        win_p0 = -1;
        if (ds) ds->bind(s);
        return true;
    }
    // GPU gp's layers' histories of conversation q -> its snapshot slot (queued on its stream)
    bool snap_copy_seq(Gpu& gp, const Seq& q, int slot, std::string& err) {
        const size_t kvd = (size_t) g.n_head_kv * g.head_dim, E = (size_t) g.n_embd;
        for (int il = gp.l0; il < gp.l1; ++il) {
            const SeqLayer& x = q.L[(size_t) il];
            float* b = x.snap + (size_t) slot * hist_n();
            const float* h[4] = {x.hist_k, x.hist_v, x.hist_a, x.hist_f};
            const size_t n[4] = {3 * kvd, 3 * kvd, 3 * E, 3 * E};
            for (int k2 = 0; k2 < 4; ++k2) {
                CK(cudaMemcpyAsync(b, h[k2], n[k2] * 4, cudaMemcpyDeviceToDevice, gp.s));
                b += n[k2];
            }
        }
        return true;
    }
    static int seq_snap_alloc(Seq& q, int64_t pos) {
        for (int i = 0; i < kSnaps; ++i)
            if (q.snap_pos[i] == pos) return i;
        const int i = q.snap_head;
        q.snap_head = (q.snap_head + 1) % kSnaps;
        q.snap_pos[i] = pos;
        return i;
    }
    // A token of each of n conversations (distinct, each read to its end) through every layer at once: the weights
    // read once for all the rows, each row's attention, short convolutions and histories its own conversation's.
    // rows [n][vocab]: each row's logits (soft-capped, no logit bias: the caller's per request); each conversation's
    // context then ends a position later and its drafter is fed the row.  The bound conversation stays bound
    bool step_multi(const int* sq, const int* ids, int n, float* rows, std::string& err) {
        if (!fast_ready || n < 1 || n > std::min(kWin, 16)) {
            err = "a multi step takes 1 .. " + std::to_string(kWin) + " rows (and the host-free step)";
            return false;
        }
        for (int i = 0; i < n; ++i) {
            bool bad = sq[i] < 0 || sq[i] >= (int) seqs.size();
            for (int j = 0; j < i; ++j) bad = bad || sq[i] == sq[j];
            if (bad) {
                err = "a multi step's rows: distinct conversations of the " + std::to_string(seqs.size());
                return false;
            }
        }
        const int E = g.n_embd;
        const double t0 = now_s();
        if (!taps_flush(err)) return false;
        seq_save(seqs[(size_t) bound_seq]);
        for (int i = 0; i < n; ++i) {
            Seq& q = seqs[(size_t) sq[i]];
            if (q.n_end + 1 > o.max_ctx) {
                err = "conversation " + std::to_string(sq[i]) + " is at the context's end";
                return false;
            }
            if (ids[i] < 0 || ids[i] >= g.n_vocab) {
                err = "token " + std::to_string(ids[i]) + " is outside the vocabulary";
                return false;
            }
            const uint8_t* row = tok_embd.data() + (size_t) ids[i] * tok_row;
            float* dst = h_xw + (size_t) i * E;
            if (tok_type == 13) dequantize_row_q5_K(row, dst, E);
            else if (tok_type == 14) dequantize_row_q6_K(row, dst, E);
            else std::memcpy(dst, row, (size_t) E * 4);
            mrows[i].seq = sq[i];
            mrows[i].p = q.n_end;
            mrows[i].slot = q.snap_next || q.n_end % tn.snap_every == 0 ? seq_snap_alloc(q, q.n_end) : -1;
            q.snap_next = false;
            q.hw = std::max(q.hw, q.n_end + 1);
        }
        use_decay(std::pow(tn.tok_decay, (float) n));
        if (!swap_poll(err, std::min(16, swap_batch * std::min(n, tn.win_rows)))) return false;
        job_multi = true;
        job_slot = -1;
        job_p = mrows[0].p;
        job_T = n;
        job_ids = ids;
        job_rows = rows;
        tap_buf ^= 1;
        const bool ok = fast_run(t0, err);
        job_multi = false;
        if (!ok) {
            seq_load(seqs[(size_t) bound_seq]);
            return false;
        }
        for (int i = 0; i < n; ++i) ++seqs[(size_t) sq[i]].n_end;
        seq_load(seqs[(size_t) bound_seq]);
        win_p0 = -1;
        cnt_pending = n;
        if (taps_on) {   // (each row to its own conversation's drafter, the last feed's read marked for the next step)
            const size_t ld = tap_layers.size() * (size_t) E;
            for (int i = 0; i < n; ++i) {
                ds->bind(sq[i]);
                if (ds->end() != mrows[i].p) continue;   // (it rests until a rewind)
                const double tf = now_s();
                if (!ds->feed_dev(G.back().d_taps[tap_buf] + (size_t) i * ld, ld, 1, mrows[i].p, err,
                                  tn.tap_event ? (void*) G.back().ev_tap[tap_buf] : nullptr))
                    return false;
                ds_feed_s += now_s() - tf;
            }
            ds->bind(bound_seq);
        }
        return true;
    }

    // the context back at the latest saved position <= pos the rings still cover (force: even when it is the end)
    bool rewind(int64_t pos, bool force, int64_t& at, std::string& err) {
        at = n_end;
        if (pos >= n_end && !force) return true;
        if (!taps_flush(err)) return false;
        const int64_t ring = (int64_t) g.n_swa + o.rewind + Tc;
        int best = -1;
        for (int i = 0; i < kSnaps; ++i) {
            const int64_t sp = snap_pos[i];
            if (sp < 0 || sp > pos || std::max<int64_t>(0, sp - (g.n_swa - 1)) < hw - ring) continue;
            if (best < 0 || sp > snap_pos[best]) best = i;
        }
        if (best < 0) {
            at = 0;
            return reset(err);
        }
        for (Gpu& gp : G) {
            CK(cudaSetDevice(gp.dev));
            if (!snap_copy(gp, best, false, err)) return false;
            CK(cudaStreamSynchronize(gp.s));
        }
        at = n_end = snap_pos[best];
        for (int i = 0; i < kSnaps; ++i)
            if (snap_pos[i] > at) snap_pos[i] = -1;
        snap_next = false;
        win_p0 = -1;
        taps_rollback();
        return true;
    }
    int run_chunk = 0;   // (tests: Model::set_chunk)

    // n tokens at positions p0 .. as chunks through the GPUs' stages: a thread a GPU, the chunks one behind the other
    // (with a tap or a pick hook: a chunk at a time, a stage at a time - their calls in the layers' order)
    bool prompt(const int* ids, int n, int64_t p0, float* logits_out, std::string& err) {
        const int tc = run_chunk > 0 ? std::min(run_chunk, Tc) : Tc;
        // the chunks end at the multiples of tc from position 0, wherever the prompt starts: a prompt read on from a
        // saved chunk start takes the chunks - and so the arithmetic, to the bit - of the same prompt read from 0
        std::vector<std::pair<int, int>> ch;
        for (int a = 0; a < n;) {
            const int e = (int) std::min<int64_t>(n, ((p0 + a) / tc + 1) * tc - p0);
            ch.push_back({a, e - a});
            a = e;
        }
        if (!taps_flush(err) || !swap_poll(err)) return false;
        abort_run = false;
        was_cancelled = false;
        for (auto& sl : slots) sl->full = false;
        // the histories at every chunk's start: the first saved here on every GPU (what a cancelled prompt goes back
        // to), the others by each stage as it starts its chunk
        hw = std::max(hw, p0 + n);
        std::vector<int> cslot;
        for (const auto& [a, T] : ch) cslot.push_back(snap_alloc(p0 + a));
        for (Gpu& gp : G) {
            CK(cudaSetDevice(gp.dev));
            if (!snap_copy(gp, cslot[0], true, err)) return false;
            CK(cudaStreamSynchronize(gp.s));
        }
        const bool serial = (bool) tapf || (bool) pick_hook || G.size() == 1;
        if (serial) {
            for (size_t c = 0; c < ch.size() && !abort_run.load(); ++c) {
                tap_p = p0 + ch[c].first;
                tap_T = ch[c].second;
                for (size_t gi = 0; gi < G.size(); ++gi)
                    if (!stage(gi, ids + ch[c].first, ch[c].second, p0 + ch[c].first, c ? cslot[c] : -1,
                               ch[c].first + ch[c].second, (int) c, err))
                        return false;
            }
        } else {
            std::vector<std::string> errs(G.size());
            std::vector<std::thread> th;
            for (size_t gi = 0; gi < G.size(); ++gi)
                th.emplace_back([&, gi] {
                    for (size_t c = 0; c < ch.size(); ++c) {
                        if (abort_run.load()) break;
                        if (!stage(gi, ids + ch[c].first, ch[c].second, p0 + ch[c].first, c ? cslot[c] : -1,
                                   ch[c].first + ch[c].second, (int) c, errs[gi])) {
                            abort_run = true;
                            for (auto& sl : slots) {
                                std::lock_guard<std::mutex> lk(sl->mu);
                                sl->cv.notify_all();
                            }
                            break;
                        }
                    }
                });
            for (auto& t : th) t.join();
            for (const std::string& e : errs)
                if (!e.empty()) {
                    err = e;
                    return false;
                }
        }
        if (abort_run.load()) {   // (cancelled: the context as it was before this prompt)
            int64_t at = 0;
            was_cancelled = true;
            n_end = p0;
            return rewind(p0, true, at, err);
        }
        snap_next = true;
        if (n >= tn.burst_min && !swap_burst(err)) return false;
        for (Gpu& gp : G) {
            st.streamed += gp.n_streamed;
            st.experts += gp.n_experts;
            gp.n_streamed = gp.n_experts = 0;
        }
        tap_p = p0 + n - 1;
        tap_T = 1;
        CK(cudaSetDevice(G.back().dev));
        if (!head_out(last_nb, logits_out, err)) return false;
        st.tokens += n;
        return true;
    }

    bool reset(std::string& err) {
        const size_t E = (size_t) g.n_embd, kvd = (size_t) g.n_head_kv * g.head_dim;
        for (Layer& l : L) {
            CK(cudaSetDevice(G[(size_t) l.gpu].dev));
            CK(cudaMemset(l.hist_k, 0, 3 * kvd * 4));
            CK(cudaMemset(l.hist_v, 0, 3 * kvd * 4));
            CK(cudaMemset(l.hist_a, 0, 3 * E * 4));
            CK(cudaMemset(l.hist_f, 0, 3 * E * 4));
            CK(cudaMemset(l.K, 0, cache_rows(l.il) * kvd * 2));
            CK(cudaMemset(l.V, 0, cache_rows(l.il) * kvd * 2));
        }
        for (Gpu& gp : G) {
            CK(cudaSetDevice(gp.dev));
            CK(cudaStreamSynchronize(gp.s));
        }
        n_end = 0;
        hw = 0;
        win_p0 = -1;
        tap_pending = 0;
        cnt_pending = 0;
        if (ds) ds->reset();
        snap_head = 0;
        snap_next = true;
        for (int64_t& x : snap_pos) x = -1;
        return true;
    }

    void notes(const char* what) const {
        for (const Gpu& gp : G) gp.note(what, &gp == &G.back() && ds ? (size_t) (ds->stats().vram_gb * 1e9 + 0.5) : 0);
    }

    std::vector<std::string> info_lines() const {
        std::vector<std::string> v;
        char b[512];
        std::snprintf(b, sizeof b, "maya-ik2: %d layers, %d experts a layer (top %d), context %lld, floor %.0f",
                      NL, g.n_expert, g.n_used, (long long) o.max_ctx, g.log_floor);
        v.push_back(b);
        for (const Gpu& gp : G) {
            size_t dense = 0;
            for (int il = gp.l0; il < gp.l1; ++il) dense += L[(size_t) il].dense;
            std::snprintf(b, sizeof b,
                          "GPU %d: layers %d .. %d, %.2f GiB dense + caches, %d of %d experts in VRAM (%.2f GiB), "
                          "%.2f of %.2f GiB free before",
                          gp.dev, gp.l0, gp.l1 - 1, dense / 1073741824.0, gp.n_res, (gp.l1 - gp.l0) * g.n_expert,
                          gp.expert_bytes / 1073741824.0, gp.free0 / 1073741824.0, gp.total / 1073741824.0);
            v.push_back(b);
        }
        std::snprintf(b, sizeof b, "experts: %d of %d in VRAM (%.1f%%), the others in RAM (%.1f GiB%s) and in place in the "
                      "files (%.1f GiB), CPU lane %d threads",
                      st.experts_vram, st.experts_all, 100.0 * st.experts_vram / std::max(1, st.experts_all),
                      st.ram_gb * 1e9 / 1073741824.0, st.ram_gb > 0 ? (st.ram_locked ? ", page-locked" : ", not page-locked") : "",
                      st.file_gb * 1e9 / 1073741824.0, lane ? lane->threads() + 1 : 0);
        v.push_back(b);
        return v;
    }
};

Model::Model() : m(new Impl) {}
Model::~Model() = default;

bool Model::load(const Options& o, std::string& err) { return m->load(o, err); }
bool Model::forward(const int* ids, int n, int64_t pos0, float* logits_out, std::string& err) {
    if (pos0 != m->n_end) {
        err = "forward at position " + std::to_string(pos0) + ": the context ends at " + std::to_string(m->n_end) +
              " (this version runs forward only: reset() first)";
        return false;
    }
    if (pos0 + n > m->o.max_ctx) {
        err = "position " + std::to_string(pos0 + n) + " is beyond the context (" + std::to_string(m->o.max_ctx) + ")";
        return false;
    }
    m->was_cancelled = false;
    m->win_p0 = -1;
    if (m->Tc > 1 && n > 1) {   // (the prompt path)
        if (!m->prompt(ids, n, pos0, logits_out, err)) return false;
        if (!m->was_cancelled) m->n_end += n;
        return true;
    }
    for (int i = 0; i < n; ++i) {
        if (!m->step(ids[i], pos0 + i, i == n - 1 ? logits_out : nullptr, err)) return false;
        ++m->n_end;
    }
    return true;
}
int Model::seq_count() const { return (int) std::max<size_t>(1, m->seqs.size()); }
int Model::bound() const { return m->bound_seq; }
bool Model::bind(int s, std::string& err) {
    if (m->seqs.empty()) {
        if (s == 0) return true;
        err = "one conversation (no --parallel)";
        return false;
    }
    return m->bind(s, err);
}
int64_t Model::end_of(int s) const {
    if (s == m->bound_seq || m->seqs.empty()) return m->n_end;
    return s >= 0 && s < (int) m->seqs.size() ? m->seqs[(size_t) s].n_end : -1;
}
bool Model::step_multi(const int* seqs, const int* ids, int n, float* rows, std::string& err) {
    std::vector<float> lg((size_t) n * m->g.n_vocab_unpadded);
    if (!m->step_multi(seqs, ids, n, rows ? rows : lg.data(), err)) return false;
    return true;
}
void Model::reset() {
    std::string err;
    if (!m->reset(err)) std::fprintf(stderr, "maya-ik2 reset: %s\n", err.c_str());
}
int64_t Model::end() const { return m->n_end; }
int64_t Model::rewind(int64_t pos) {
    std::string err;
    int64_t at = 0;
    if (!m->rewind(pos, false, at, err)) std::fprintf(stderr, "maya-ik2 rewind: %s\n", err.c_str());
    return at;
}
void Model::set_progress(std::function<bool(int64_t)> f) { m->progress = std::move(f); }
bool Model::cancelled() const { return m->was_cancelled; }
int Model::top() const { return m->top_id; }
bool Model::save_profile(std::string& err) const {
    // (maya-ds4's: --profile-out gets this run's routing alone, the profile the counts read at the load and this run's
    //  - unless --profile-readonly)
    const Options& o = m->o;
    if (o.profile_out.empty() && (o.profile.empty() || o.profile_readonly)) return true;
    m->counts_flush();
    const auto write = [&](const std::string& path, const std::vector<uint32_t>& v) {
        const std::string tmp = path + ".tmp";
        std::ofstream f(tmp, std::ios::binary);
        if (!f || !f.write((const char*) v.data(), (std::streamsize) (v.size() * 4)) || !f.flush()) {
            err = tmp + ": cannot write";
            return false;
        }
        f.close();
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            std::remove(tmp.c_str());
            err = path + ": cannot rename";
            return false;
        }
        return true;
    };
    if (!o.profile_out.empty() && !write(o.profile_out, m->prof_cnt)) return false;
    if (o.profile.empty() || o.profile_readonly) return true;
    std::vector<uint32_t> all = m->prof_cnt;
    for (size_t i = 0; i < all.size() && i < m->prof_in.size(); ++i) all[i] += m->prof_in[i];
    return write(o.profile, all);
}
int Model::vocab() const { return m->g.n_vocab_unpadded; }
int64_t Model::max_ctx() const { return m->o.max_ctx; }
int Model::eos() const { return m->g.eos; }
bool Model::window(const int* ids, int T, int64_t pos0, int* tops, float* logits_rows, std::string& err) {
    if (T < 1 || T > window_max() || pos0 != m->n_end || pos0 + T > m->o.max_ctx) {
        err = "window of " + std::to_string(T) + " at " + std::to_string(pos0) + ": the context ends at " +
              std::to_string(m->n_end) + ", a window is 1 .. " + std::to_string(window_max()) + " rows";
        return false;
    }
    if (!m->fast_on()) {   // (T 1 here: a token's step with its taps)
        if (!m->step(ids[0], pos0, logits_rows, err)) return false;
        tops[0] = m->top_id;
        m->win_p0 = -1;
        ++m->n_end;
        return true;
    }
    m->tap_p = pos0;
    if (!m->step_win(ids, T, pos0, tops, logits_rows, err)) return false;
    m->n_end += T;
    return true;
}
bool Model::keep(int rows, std::string& err) { return m->win_keep(rows, err); }
int Model::window_max() const { return m->fast_on() ? Impl::kWin : 1; }
bool Model::set_logit_bias(const std::vector<std::pair<int, float>>& entries, std::string& err) {
    return m->set_logit_bias(entries, err);
}
void Model::set_debug(int mask) { m->dbg = mask; }
void Model::set_swap(bool on) { m->o.expert_cache = on; }
bool Model::dspark() const { return m->taps_on; }
bool Model::draft(int anchor, int max_n, std::vector<int>& toks, std::vector<float>& probs, std::string& err) {
    toks.clear();
    probs.clear();
    if (!m->taps_on || !m->fast_on()) return true;
    if (!m->taps_flush(err)) return false;
    if (m->ds->end() != m->n_end) return true;   // (the drafter missed rows - a step with a tap or a hook: none until a rewind)
    return m->ds->draft(anchor, max_n, toks, probs, err);
}
void Model::set_draft_cum(float c) {
    if (m->ds) m->ds->set_cum_min(c);
}
double Model::dspark_vram_gb() const { return m->ds ? m->ds->stats().vram_gb : 0.0; }
int Model::layers() const { return m->NL; }
int Model::chunk() const { return m->Tc; }
void Model::set_chunk(int n) { m->run_chunk = n; }
const Stats& Model::stats() const { return m->st; }
std::vector<std::string> Model::info_lines() const { return m->info_lines(); }
void Model::set_tap(Tap t) { m->tapf = std::move(t); }
void Model::set_pick_hook(PickHook h) { m->pick_hook = std::move(h); }
void Model::vram_notes(const char* what) const { m->notes(what); }
void Model::retune() { m->retune(); }

}  // namespace ik2hip
