// src/core/glm_prefill.cu - the glm5-next batched PROMPT path (Glm5Model members).
//
// The token path (glm_fast_path.cu) reads a prompt one token at a time: every weight of the model crosses the
// memory bus once per token, ~31-55 ms each.  Here a chunk of T tokens goes through one layer at a time:
//
//   * projections are GEMMs - quantized weights dequantized to FP16 into a scratch and multiplied on the tensor
//     cores (cuBLAS), BF16 weights widened to F32 (SGEMM, the router and the hyper-connections stay F32-exact);
//   * the recurrences (KDA's delta rule, the causal convs) walk the chunk inside one kernel, the DSA caches are
//     written for every position before any token of the chunk attends;
//   * the routed experts are grouped by expert: the ones RESIDENT in the VRAM tier are multiplied where they lie
//     (llama.cpp's MMQ over the layer's whole pool partition in one launch - the slot stride is MMQ-addressable,
//     glmfast::expert_stride), the others are copied from the pinned RAM tier (or read from disk) into a small ring
//     of staging groups on the copy stream while the resident ones compute.
//
// The kernels are the token path's arithmetic (strata/kernels/glm_batch.hpp); what differs is the projections'
// rounding (FP16 activations instead of q8_1), as between llama.cpp's batched and one-token paths.  The prompt's
// routing also feeds the expert tiers' LFU counts, so the decode that follows starts from this conversation's
// experts.  STRATA_GLM_NO_PREFILL=1 keeps the token-at-a-time prompt (A/B).
#include "glm_fast_state.hpp"

#include "strata/core/progress.hpp"
#include "strata/kernels/cpu/kq_avx512.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/iq_kernels.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/prefill/gemm.hpp"
#endif
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {

namespace gb = strata::kernels::glmb;
namespace mmq = strata::prefill::mmq;

namespace {

// the RAM free now (MemAvailable; Windows: the smaller of free RAM and free commit), 0 when unknown
int64_t avail_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (int64_t) std::min(ms.ullAvailPhys, ms.ullAvailPageFile) : 0;
#else
    long long kb = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
        std::fclose(f);
    }
    return (int64_t) kb * 1024;
#endif
}

// carves 256-byte-aligned buffers off a base; with base == nullptr it only measures
struct Carve {
    uint8_t* base = nullptr;
    size_t off = 0;
    template <class Tp>
    Tp* take(size_t n) {
        Tp* p = base ? (Tp*) (base + off) : nullptr;
        off += (n * sizeof(Tp) + 255u) & ~(size_t) 255u;
        return p;
    }
};

constexpr int kSub = 256;   // the sub-batch the mixers, the dense FFN and the shared expert start from
// prestaging (PrefillState::pbuf): the buffer's slots on one GPU, the chunk from which it is carved, and the fewest
// predicted rows an expert is prestaged for (below that it may not be routed at all)
constexpr int kPreSlots = 160, kPreMinT = 1024, kPreMinRows = 4;
// resident experts routed by at most kLightRows rows of a chunk take the light kernels (gf::rows_experts) instead of
// MMQ (STRATA_GLM_PREFILL_LIGHT overrides, 0 = MMQ only); their {slot, r0, nr} ride the bounds array from kLightOff
constexpr int kLightRows = 32, kLightOff = 6144, kLightMax = (8192 - kLightOff) / 3;

struct KdaBufs {
    float* proj[3];
    float* conv[3];
    float *fa, *ga, *beta, *g1, *g2;
    uint16_t* out16;
};
KdaBufs carve_kda(Carve& c, size_t T, const Glm5Geometry& g) {
    KdaBufs b{};
    const size_t DI = (size_t) g.d_inner();
    for (int i = 0; i < 3; ++i) b.proj[i] = c.take<float>(T * DI);
    for (int i = 0; i < 3; ++i) b.conv[i] = c.take<float>(T * DI);
    b.fa = c.take<float>(T * (size_t) g.kda_head_dim);
    b.ga = c.take<float>(T * (size_t) g.kda_head_dim);
    b.beta = c.take<float>(T * (size_t) g.n_head);
    b.g1 = c.take<float>(T * DI);
    b.g2 = c.take<float>(T * DI);
    b.out16 = c.take<uint16_t>(T * DI);
    return b;
}

#if defined(STRATA_USE_HIP)
// The MLA batched products in FP16 with every head's tokens contiguous (hipBLAS picks far faster kernels than for the
// F32 ones).  The prompt attention has three paths: the rocWMMA kernel on FP16 q_abs (f16q), the wave32 WMMA kernel on
// F32 q_abs (wmma2), and the F32 kernel.  Both FP16 products and the WMMA attention are on by default;
// STRATA_GLM_MLA_F16=0 and STRATA_GLM_MLA_WMMA=0 turn them off (WMMA off keeps the F32 attention, F16 off also the F32
// products).  STRATA_GLM_PREFILL_ATTN=wmma2|f16q|f32 forces an attention path; unset, gfx12 (RDNA4) defaults to wmma2
// and gfx11 to f16q.
__global__ void pack_heads_f16(const float* __restrict__ src, uint16_t* __restrict__ dst, int T, int H, int K) {
    const int64_t n = (int64_t) T * H * K;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int k = (int) (i % K);
        const int64_t r = i / K;
        const int t = (int) (r % T), h = (int) (r / T);
        dst[i] = __half_as_ushort(__float2half(src[((int64_t) t * H + h) * K + k]));
    }
}
__global__ void bf16_to_f16(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = __half_as_ushort(__float2half(__uint_as_float((uint32_t) src[i] << 16)));
}
bool env_on(const char* name) {
    const char* v = getenv(name);
    return !(v && v[0] == '0');
}
bool mla_wmma_on() {
    static const bool on = env_on("STRATA_GLM_MLA_WMMA");
    return on;
}
bool mla_f16_on() {
    static const bool on = env_on("STRATA_GLM_MLA_F16") || mla_wmma_on();
    return on;
}
enum class MlaAttnPath { F32, F16Q, WMMA2 };
MlaAttnPath mla_attn_path() {
    if (!mla_wmma_on()) return MlaAttnPath::F32;   // STRATA_GLM_MLA_WMMA=0 keeps the F32 kernel as before
    static const char* mode = getenv("STRATA_GLM_PREFILL_ATTN");
    static const bool want_wmma2 = mode != nullptr && std::strcmp(mode, "wmma2") == 0;
    static const bool want_f16q = mode != nullptr && std::strcmp(mode, "f16q") == 0;
    static const bool want_f32 = mode != nullptr && std::strcmp(mode, "f32") == 0;
    if (want_f16q) return MlaAttnPath::F16Q;
    if (want_f32) return MlaAttnPath::F32;
    if (want_wmma2) return gb::mla_wmma2_supported() ? MlaAttnPath::WMMA2 : MlaAttnPath::F16Q;
    return gb::mla_wmma2_default() ? MlaAttnPath::WMMA2 : MlaAttnPath::F16Q;
}
#endif
struct DsaBufs {
    float *qr_raw, *qr, *kv_raw, *ik_raw, *ig_raw, *iw, *q, *iq, *score, *q_abs, *ctx, *attn;
    uint16_t *qr16, *attn16;
    int *cells, *n_sel;
    int score_ld;
};
DsaBufs carve_dsa(Carve& c, size_t T, const Glm5Geometry& g, int max_pools) {
    DsaBufs b{};
    b.qr_raw = c.take<float>(T * g.q_lora);
    b.qr = c.take<float>(T * g.q_lora);
    b.qr16 = c.take<uint16_t>(T * g.q_lora);
    b.kv_raw = c.take<float>(T * g.kv_lora);
    b.ik_raw = c.take<float>(T * g.idx_key);
    b.ig_raw = c.take<float>(T * g.idx_key);
    b.iw = c.take<float>(T * g.idx_heads);
    b.q = c.take<float>(T * (size_t) g.n_head * g.qk_nope);
    b.iq = c.take<float>(T * (size_t) g.idx_heads * g.idx_key);
    b.cells = c.take<int>(T * (size_t) g.n_sel_max());
    b.n_sel = c.take<int>(T);
    b.score_ld = std::max(1, max_pools);
    b.score = c.take<float>(T * (size_t) b.score_ld);
    b.q_abs = c.take<float>(T * (size_t) g.n_head * g.kv_lora);
    b.ctx = c.take<float>(T * (size_t) g.n_head * g.kv_lora);
    b.attn = c.take<float>(T * (size_t) g.n_head * g.v_head);
    b.attn16 = c.take<uint16_t>(T * (size_t) g.n_head * g.v_head);
    return b;
}

struct DenseBufs {
    float *dg, *du;
    uint16_t* dh16;
};
DenseBufs carve_dense(Carve& c, size_t T, const Glm5Geometry& g) {
    DenseBufs b{};
    b.dg = c.take<float>(T * g.n_ff_dense);
    b.du = c.take<float>(T * g.n_ff_dense);
    b.dh16 = c.take<uint16_t>(T * g.n_ff_dense);
    return b;
}

// the MoE output rows a layer keeps on the device at once: a WINDOW (each expert set's rows are added into ffn as the
// window fills - moe_combine_add - instead of every routed row living until one combine at the end: 128 KB a token),
// and as many again for the CPU experts' uploaded rows.  At least T rows (one expert never has more than the chunk's
// tokens) and the light kernels' most; never more than the chunk routes.
// (STRATA_GLM_PREFILL_WINDOW=<rows> sets it, at least the floor; 0 = every routed row, the layout before the window)
int moe_window_rows(size_t T, const Glm5Geometry& g) {
    const size_t all = T * (size_t) g.n_exp_used, floor = std::max(T, (size_t) kLightMax * kLightRows);
    static const long long env = [] {
        const char* v = getenv("STRATA_GLM_PREFILL_WINDOW");
        return v ? std::atoll(v) : -1LL;
    }();
    if (env == 0) return (int) all;
    return (int) std::min(all, env > 0 ? std::max(floor, (size_t) env) : floor);
}

// the MoE runs on the WHOLE chunk (every expert's weights are read once per chunk); the shared expert in sub-batches
struct MoeBufs {
    float *logits, *rw, *OUTP, *OUTPc;
    float *sh_g, *sh_u;
    uint16_t* sh16;
    int *ids, *rank, *counts, *base, *row_tok, *pos, *bounds;
    uint8_t *Xq, *Hq;
};
MoeBufs carve_moe(Carve& c, size_t T, size_t sub, const Glm5Geometry& g) {
    MoeBufs b{};
    const size_t rows = T * (size_t) g.n_exp_used;
    const size_t FF = (size_t) g.n_ff_exp * g.n_shared;
    b.logits = c.take<float>(T * g.n_expert);
    b.ids = c.take<int>(rows);
    b.rw = c.take<float>(rows);
    b.rank = c.take<int>(rows);
    b.pos = c.take<int>(rows);
    b.row_tok = c.take<int>(rows);
    b.counts = c.take<int>((size_t) g.n_expert);
    b.base = c.take<int>((size_t) g.n_expert);
    b.bounds = c.take<int>(8192);
    const size_t W = (size_t) moe_window_rows(T, g);
    b.Xq = c.take<uint8_t>(mmq::q8_bytes((int64_t) W, g.n_embd));
    b.Hq = c.take<uint8_t>(mmq::q8_bytes((int64_t) W, g.n_ff_exp));
    // each set's gate/up rows live in its own OUTP rows until the down product, the swiglu over their gate halves
    b.OUTP = c.take<float>(W * g.n_embd);
    b.OUTPc = c.take<float>(W * g.n_embd);
    const size_t ts = std::min(T, sub);   // the shared expert's sub-batch
    b.sh_g = c.take<float>(ts * FF);
    b.sh_u = c.take<float>(ts * FF);
    b.sh16 = c.take<uint16_t>(ts * FF);
    return b;
}

// the mixers' and the dense FFN's sub-batch for chunks of T tokens: kSub, doubled (up to 4096) while their buffers
// stay within what the chunk's MoE buffers take of the union anyway - no extra memory, fewer and larger GEMMs, and
// each projection dequantized to FP16 once a sub-batch instead of once every 256 tokens (a 14.8K chunk: 58 -> 4).
// sub_env (STRATA_GLM_PREFILL_SUB, 16-8192) sets every sub-batch instead; HIP keeps kSub without it.
int mixer_sub(size_t T, const Glm5Geometry& g, bool kda, bool dsa, bool dense, bool moe, int max_pools, int sub_env) {
    if (sub_env > 0) return sub_env;
#if defined(STRATA_USE_HIP)
    (void) T; (void) g; (void) kda; (void) dsa; (void) dense; (void) moe; (void) max_pools;
    return kSub;
#else
    size_t room = 0;
    if (moe) { Carve m; carve_moe(m, T, kSub, g); room = m.off; }
    int ts = kSub;
    for (int t2 = 2 * kSub; t2 <= 4096 && (size_t) t2 <= T; t2 *= 2) {
        size_t need = 0;
        if (kda) { Carve k; carve_kda(k, (size_t) t2, g); need = std::max(need, k.off); }
        if (dsa) { Carve d; carve_dsa(d, (size_t) t2, g, max_pools); need = std::max(need, d.off); }
        if (dense) { Carve d; carve_dense(d, (size_t) t2, g); need = std::max(need, d.off); }
        if (need > room) break;
        ts = t2;
    }
    return ts;
#endif
}

void blas_ck(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "glm prefill: %s: cuBLAS status %d\n", what, (int) st);
}

}  // namespace

struct Glm5Model::PrefillState {
    int T = 0;                       // tokens per chunk
    // --prefill: the chunk and its prestage slots for a lend cap (% of the pool's slots), the cap it was sized with,
    // whether the measured pinned share may still lower it (prefill_settle), and how it reads in the log
    std::function<std::pair<int64_t, int>(int64_t)> choose;
    int64_t lend_pct = 90;
    bool lend_pct_fixed = false;
    std::string mode;
    int sub_env = 0;                 // STRATA_GLM_PREFILL_SUB: every sub-batch (0: mixer_sub, the shared expert kSub)
    int sh_sub() const { return sub_env > 0 ? sub_env : kSub; }   // the shared expert's sub-batch
    int max_pools = 0;
    cublasHandle_t blas = nullptr;
    void* ws = nullptr;
#if defined(STRATA_USE_HIP)
    std::unique_ptr<strata::prefill::Gemm> lt;   // hipBLASLt with a tuning table (STRATA_HIPBLASLT_TUNING)
    bool lt_tried = false;
#endif
    std::unique_ptr<mmq::Context> mq;
    uint8_t* arena = nullptr;
    size_t arena_bytes = 0;
    // every device buffer lives in the pool's lendable tail (prefill_bind): valid only while lent
    size_t borrow_bytes = 0, ws_bytes = 0, gbuf_bytes = 0;
    bool lent = false;
    bool has_kda = false, has_dsa = false, has_dense = false, has_moe = false;
    uint8_t* region = nullptr;       // the pool's lendable tail
    size_t region_bytes = 0;
    int T_bound = 0;                 // the chunk the pointers are carved for (<= T)
    int sub = kSub;                  // ... and its mixer sub-batch (mixer_sub)
    size_t need = 0;                 // ... and the bytes of the region that layout uses
    uint64_t dropped = 0;            // experts the lending evicted from VRAM (cumulative)
    uint64_t moved = 0;              // ... of which a lent slot's expert took a colder one's slot instead
    // per chunk, across layers
    float *R = nullptr, *x = nullptr, *mixer = nullptr, *ffn = nullptr;
    float *pre = nullptr, *post = nullptr, *comb = nullptr, *ss = nullptr, *mix = nullptr;
    uint16_t* x16 = nullptr;
    int* iota = nullptr;
    // KV streaming: one layer's latent cache in position order, staged from its host copy - what the prompt attention
    // reads (a chunk's selections can name any position, so the slots' window does not hold them).  The copy runs on
    // kv_ss once the attention before has read the stage (kv_free), while the layers in between compute; the layer it
    // is for waits on kv_ready (kv_next: that layer, -1 none).  STRATA_GLM_KV_PREFETCH=0: on the compute stream.
    uint8_t* kv_stage = nullptr;
    cudaStream_t kv_ss = nullptr;
    cudaEvent_t kv_ready = nullptr, kv_free = nullptr;
    int kv_next = -1;
    uint8_t* uni = nullptr;          // the per-layer region (KDA | DSA | dense | MoE)
    // weight scratch
    uint16_t* w16 = nullptr;
    int64_t w16_elems = 0;
    float* w32 = nullptr;
    int64_t w32_elems = 0;
    // non-resident experts: a ring of NG staging groups of GE slots each (the layer's slot stride apart)
    static constexpr int NG = 3, GE = 4;
    uint8_t* gbuf = nullptr;
    size_t gstride = 0;
    // pinned: the disk reads' landing ring, nland slots of gstride - deep enough that the reader runs a layer's disk
    // experts ahead of the groups that use them (12 slots kept the reader and the GPU waiting on each other: a 16k
    // prompt on Mercury read the NVMe at ~0.7 of its ~2.9 GB/s); STRATA_GLM_PREFILL_LAND=<slots>, at least NG * GE
    uint8_t* gpin = nullptr;
    int nland = NG * GE;
    cudaEvent_t ev_ready[NG] = {}, ev_free[NG] = {};
    std::vector<cudaEvent_t> ev_land;    // a landing slot's copy to the device ran: the reader may refill it
    // pinned host staging
    float* emb_h = nullptr;          // T x n_embd
    float* hop_h = nullptr;          // 2 x T x hc x n_embd: the split's hand-over (first half), double-buffered
    int* h_counts = nullptr;
    int* h_base = nullptr;
    int* h_bounds = nullptr;
    cudaEvent_t ev_hop = nullptr;
    // stats
    double ms = 0, ms_plan = 0, ms_disk = 0;
    int64_t tokens = 0, chunks = 0;
    // STRATA_GLM_PREFILL_PROF=1: events between the phases of every layer, summed per phase (debug)
    bool prof = false;
    std::vector<cudaEvent_t> pev;
    std::vector<const char*> pname;
    size_t pn = 0;
    std::map<std::string, double> pacc;
    void mark(const char* nm, cudaStream_t st) {
        if (!prof) return;
        if (pn >= pev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            pev.push_back(e);
            pname.push_back(nm);
        }
        cudaEventRecord(pev[pn], st);
        pname[pn] = nm;
        ++pn;
    }
    void collect() {
        if (!prof || pn == 0) return;
        cudaEventSynchronize(pev[pn - 1]);
        for (size_t i = 1; i < pn; ++i) {
            float t = 0.0f;
            if (cudaEventElapsedTime(&t, pev[i - 1], pev[i]) == cudaSuccess) pacc[pname[i]] += t;
        }
        pn = 0;
    }
    uint64_t staged_ram = 0, staged_vram = 0, staged_disk = 0, rows_resident = 0, rows_staged = 0, disk_issued = 0;
    // the prompt's CPU experts: x and the rows' tokens down, the outputs up, on their own stream
    cudaStream_t xs = nullptr;
    cudaEvent_t ev_x = nullptr, ev_out = nullptr;
    float* x_h = nullptr;
    size_t x_h_n = 0;
    int* rt_h = nullptr;
    float* out_h = nullptr;
    size_t rows_h_n = 0;
    std::vector<uint8_t> c_act, c_hq, c_pack;
    std::vector<float> c_ff;
    uint64_t cpu_experts = 0, cpu_rows = 0;
    double ms_cpu = 0, row_ms = 0.05;   // row_ms: the host's cost of a row, learned (the split's estimate)
    // the split's balance, learned: the host's estimates times kappa, nudged each layer by the host's time over the
    // copy stream's (ev_c0 .. ev_c1: the staged copies) - the two lanes should end together
    double kappa = 1.0;
    cudaEvent_t ev_c0 = nullptr, ev_c1 = nullptr;
    // PRESTAGING (one GPU): the copy stream idles while a layer's mixer runs (~30% of a long prompt on a 3090), so
    // the next MoE layer's likely experts - RAM-tier ones, by predicted rows - are copied then, into a buffer of NP
    // slots that every layer reuses (one slot serves each layer: worth ~a slot per layer of the pool it is borrowed
    // from, for the cost of one).  The plan computes the ones its routing hit from there and splits the rest between
    // the CPU and PCIe as before; copies still in flight at the plan count against the PCIe side.
    int NP = 0;                           // slots (of the largest MoE layer stride)
    size_t pbuf_bytes = 0;
    uint8_t* pbuf = nullptr;              // carved for chunks of kPreMinT tokens and more
    int pre_layer = -1;                   // the layer whose experts pbuf holds (or is receiving)
    std::vector<int> pre_e;               // ... in slot order
    static constexpr int kPreEv = 8;      // an event every kPreEv copies: how far they are at the plan
    std::vector<cudaEvent_t> ev_pdone;
    cudaEvent_t ev_pready = nullptr, ev_pfree = nullptr, ev_pstart = nullptr;
    // STRATA_GLM_PRESTAGE_ADAPT=1 (experimental): how many a layer prestages, in experts per chunk token by mixer
    // (KDA, DSA), learned from its lanes - the copy stream idle before the plan: more; the CPU done before copies the
    // plan could not move to it: fewer (0: unset).  Off: every layer fills the buffer
    double pre_rate[2] = {0.0, 0.0};
    double copy_ms = 0.0;                 // a prestage copy's time, measured (the split's PCIe estimate)
    std::vector<std::vector<int>> last_cnt;   // per layer: the routing of its last chunk (half the prediction)
    uint64_t pre_issued = 0, pre_hit = 0, pre_rows = 0;
    // STRATA_GLM_PREFILL_TRACE=1 (debug): per MoE layer of a chunk, when its lanes ended - printed after the chunk
    struct Tr {
        cudaEvent_t start = nullptr, plan = nullptr, pre = nullptr, c1 = nullptr, end = nullptr;
        int pre_n = 0, pre_left = 0, grp = 0, ncpu = 0, crows = 0;
        double dl = 0, dcpu = 0;
        bool on = false;
    };
    bool trace = false;
    std::vector<Tr> tr;
};

int Glm5Model::prefill_chunk() const { return pf_ ? pf_->T : 0; }

// --prefill as the engine reads it: STRATA_GLM_PREFILL (generate sets it from the config's args), unset = auto;
// STRATA_GLM_PREFILL_CHUNK=N (the older knob) is --prefill N
static std::string prefill_mode() {
    if (const char* c = getenv("STRATA_GLM_PREFILL_CHUNK"); c != nullptr && c[0]) return c;
    if (const char* v = getenv("STRATA_GLM_PREFILL"); v != nullptr && v[0]) return v;
    return "auto";
}

// a split's chunks are its smallest part's (prefill()): a part set up for longer ones keeps its pinned staging to that
// (a 24 GB card sized for 32K tokens beside a 16 GB card's 22K would pin ~1.5 GB of hop and embedding rows no chunk
// uses)
void Glm5Model::prefill_cap(int T, const char* why) {
    PrefillState* S = pf_;
    if (S == nullptr || S->T <= T) return;
    const size_t E = (size_t) g_.n_embd;
    const bool hop = S->hop_h != nullptr;
    cudaSetDevice(dev_);
    cudaFreeHost(S->emb_h);
    S->emb_h = nullptr;
    if (hop) cudaFreeHost(S->hop_h);
    S->hop_h = nullptr;
    if (cudaHostAlloc((void**) &S->emb_h, (size_t) T * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        (hop && cudaHostAlloc((void**) &S->hop_h, (size_t) 2 * T * 4 * E * sizeof(float), cudaHostAllocDefault) !=
                    cudaSuccess)) {
        cudaGetLastError();
        std::fprintf(stderr, "glm prefill: CUDA%d pinned staging did not allocate - token by token\n", dev_);
        prefill_destroy();
        return;
    }
    std::fprintf(stderr, "glm prefill: CUDA%d chunks of %d tokens (%s)\n", dev_, T, why);
    S->T = T;
}

// Strata's lend cap, settled once the RAM tier exists (prefill_setup sized the chunk with its 90): 90% of the pool's
// slots when at least 90% of this card's expert bytes are held pinned - in the VRAM pool or the RAM tier, whose copies
// are DMA - else 85% (the rest comes from the disk through host copies, the limit there: lending more only streams
// more through them)
void Glm5Model::prefill_settle(double pinned_share) {
    PrefillState* S = pf_;
    if (S == nullptr || S->lend_pct_fixed || pinned_share >= 0.9) return;
    S->lend_pct = 85;
    const auto pick = S->choose(S->lend_pct);
    std::fprintf(stderr, "glm prefill: CUDA%d %.0f%% of the experts' bytes are held pinned: the pool lends at most 85%% "
                         "of its slots\n", dev_, 100.0 * pinned_share);
    if (pick.first == 0) {
        std::fprintf(stderr, "glm prefill: CUDA%d the expert pool can lend no prompt chunk - token by token\n", dev_);
        prefill_destroy();
        return;
    }
    if (pick.first >= S->T) return;
    S->NP = std::min(S->NP, pick.second);
    S->pbuf_bytes = (size_t) S->NP * S->gstride;
    prefill_cap((int) pick.first, "--prefill auto at 85%");
}

// the weight scratch the prompt path keeps beside a chunk's rows: {BF16 elements, F32 elements}
static std::pair<int64_t, int64_t> weight_scratch(const Glm5Geometry& g, bool has_dsa) {
    const int64_t E = g.n_embd;
    int64_t w32 = 2 << 20;
    if (has_dsa)
        w32 = std::max<int64_t>({w32, (int64_t) g.n_head * g.kv_lora * g.qk_nope, (int64_t) g.n_head * g.v_head * g.kv_lora});
    int64_t w16 = std::max<int64_t>((int64_t) g.d_inner() * E, (int64_t) 2048 * E);
#if defined(STRATA_USE_HIP)
    if (has_dsa)   // the MLA products' FP16 copy of wk_b / wv_b
        w16 = std::max<int64_t>({w16, (int64_t) g.n_head * g.kv_lora * g.qk_nope, (int64_t) g.n_head * g.v_head * g.kv_lora});
#endif
    return {w16, w32};
}
static constexpr size_t kPfWorkspace = (size_t) 16 << 20;   // cuBLAS

// ---------------------------------------------------------------- setup
bool Glm5Model::prefill_setup(std::string& err) {
    (void) err;
    if (getenv("STRATA_GLM_NO_PREFILL") != nullptr) return true;
    if (prefill_mode() != "auto" && std::atoll(prefill_mode().c_str()) <= 0) {
        std::fprintf(stderr, "glm prefill: --prefill %s: the prompt runs token by token\n", prefill_mode().c_str());
        return true;
    }
    if (!mmq::built()) {
        std::fprintf(stderr, "glm prefill: this build has no MMQ kernels - the prompt runs token by token\n");
        return true;
    }
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    bool has_kda = false, has_dsa = false, has_moe = false, has_dense = false;
    size_t gstride = 0;
    std::string why;
    const auto deq_ok = [&](const WSlot& w, const char* nm, int il) {
        if (w.q == nullptr || (w.type != gf::kTypeBF16 && !strata::kernels::dequant_bf16_supported(w.type)))
            why += "blk." + std::to_string(il) + "." + nm + " (type " + std::to_string(w.type) + ") ";
    };
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (Ly.recr) {
            has_kda = true;
            deq_ok(Ly.q, "attn_q", il);
            deq_ok(Ly.k, "attn_k", il);
            deq_ok(Ly.v, "attn_v", il);
        } else {
            has_dsa = true;
            deq_ok(Ly.q_a, "attn_q_a", il);
            deq_ok(Ly.q_b, "attn_q_b", il);
            deq_ok(Ly.kv_a, "attn_kv_a", il);
        }
        deq_ok(Ly.out, "attn_output", il);
        if (Ly.moe) {
            has_moe = true;
            deq_ok(Ly.sh_gate, "ffn_gate_shexp", il);
            deq_ok(Ly.sh_up, "ffn_up_shexp", il);
            deq_ok(Ly.sh_down, "ffn_down_shexp", il);
            if (!mmq::supported(Ly.gu_type) || !mmq::supported(Ly.d_type))
                why += "blk." + std::to_string(il) + " experts (types " + std::to_string(Ly.gu_type) + "/" +
                       std::to_string(Ly.d_type) + ") ";
            gstride = std::max(gstride, glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type));
        } else {
            has_dense = true;
            deq_ok(Ly.ffn_gate, "ffn_gate", il);
            deq_ok(Ly.ffn_up, "ffn_up", il);
            deq_ok(Ly.ffn_down, "ffn_down", il);
        }
    }
    if (!why.empty()) {
        std::fprintf(stderr, "glm prefill: CUDA%d has no prompt path (not covered: %s) - token by token\n", dev_,
                     why.substr(0, 400).c_str());
        return true;
    }
    if (has_dsa && (g.kv_lora != 512 || g.n_head % 16 != 0 || g.kda_head_dim != 128)) {
        std::fprintf(stderr, "glm prefill: geometry not covered by the batched kernels - token by token\n");
        return true;
    }
    auto* S = new PrefillState();
    pf_ = S;
    S->has_kda = has_kda;
    S->has_dsa = has_dsa;
    S->has_dense = has_dense;
    S->has_moe = has_moe;
    S->max_pools = (int) std::max<int64_t>(1, max_ctx_ / g.idx_kpool);
    if (const char* v = getenv("STRATA_GLM_PREFILL_SUB"))
        S->sub_env = std::clamp(std::atoi(v), 16, 8192);
    S->prof = getenv("STRATA_GLM_PREFILL_PROF") != nullptr;
    S->gstride = gstride;

    // ---- sizes: the chunk T from the budget (per device)
    const int64_t E = g.n_embd;
    const auto wsc = weight_scratch(g, has_dsa);
    const int64_t w16_elems = wsc.first, w32_elems = wsc.second;
    const size_t ws_bytes = kPfWorkspace;
    // ---- the chunk T, as Strata's --prefill (the config's args; STRATA_GLM_PREFILL, unset = auto).  `auto`: the
    // largest chunk on the 256-token grid, up to 32768 on one GPU and 8192 on a split (each the faster there: one V100,
    // 26K-token prompts 559 tok/s at 32768 against 367 at 8192; two V100s 709 at 8192 against 551 at 32768), whose
    // buffers the expert pool fast_setup carves next can lend.
    // The pool lends at most STRATA_PREFILL_LEND_PCT of its slots - Strata's 90 when at least 90% of the expert bytes
    // are held pinned (the copies are DMA), 85 when host copies are the limit; that share is known once the RAM tier
    // exists, so the chunk is sized with 90 here and prefill_settle() takes it down to 85's when the share falls short
    // - and keeps every layer a route's experts and the spares (fast_setup's floor).  The prestage buffer is Strata's
    // ring and comes out of the same loan: the chunk is the largest that leaves it whole, else the largest that leaves
    // it 16 slots, else none.  A number N: chunks of N tokens, halved until the pool can lend them (only the floor
    // applies - the operator's number).  A split reads in its smallest part's chunk (load_pack_split).
    // STRATA_GLM_PREFILL_MB=<MB> fixes what may be lent.
    const std::string mode = prefill_mode();
    const bool is_auto = mode == "auto";
    const int64_t auto_max = n_parts_ == 1 ? 32768 : 8192;
    const int64_t ceiling = is_auto ? auto_max : (int64_t) std::atoll(mode.c_str());
    // the pool to come: `per` slots a layer over the tier layers, as fast_setup sizes it from the same free VRAM
    size_t dev_free = 0, dev_total = 0;
    cudaMemGetInfo(&dev_free, &dev_total);
    const size_t avail = pool_avail(dev_free, dev_total);
    size_t blob_sum = 0, stride_sum = 0;
    for (int il = l0_; il < lt_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.moe) continue;
        blob_sum += Ly.blob;
        stride_sum += glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type);
    }
    int64_t per = std::min<int64_t>(g.n_expert, (int64_t) (avail / std::max<size_t>(1, blob_sum)));
    while (per > 0 && (size_t) per * stride_sum > avail) --per;
    const int64_t keep = g.n_exp_used + gf::kSpares + 2;
    const char* pct_env = getenv("STRATA_PREFILL_LEND_PCT");
    const char* mb_env = getenv("STRATA_GLM_PREFILL_MB");
    const size_t gbuf = has_moe ? (size_t) PrefillState::NG * PrefillState::GE * gstride : 0;
    // the prestage buffer (the ring): one GPU (a split's parts hold most of their experts); STRATA_GLM_PRESTAGE=<slots>
    int ring_max = has_moe && n_parts_ == 1 ? kPreSlots : 0;
    if (const char* v = getenv("STRATA_GLM_PRESTAGE")) ring_max = has_moe ? std::max(0, std::atoi(v)) : 0;
    // the chunk and its prestage slots for a lend cap of `pct`% of the pool's slots ({0, 0}: no chunk fits)
    S->choose = [=, this](int64_t pct) -> std::pair<int64_t, int> {
        size_t lendable = 0;   // the bytes the pool can lend the prompt path
        if (mb_env != nullptr) {
            lendable = (size_t) (std::atof(mb_env) * 1048576.0);
        } else if (stride_sum == 0) {
            lendable = avail;
        } else {
            int64_t k = std::max<int64_t>(0, per - keep);
            if (is_auto) k = std::min<int64_t>(k, pct * per / 100);
            lendable = (size_t) k * stride_sum;
        }
        const auto nonring = [&](int64_t T) -> size_t {
            const auto pu = prefill_bytes_for((size_t) T);
            Carve c;
            c.take<uint8_t>(pu.first + pu.second);
            c.take<uint16_t>((size_t) w16_elems);
            c.take<float>((size_t) w32_elems);
            c.take<uint8_t>(ws_bytes);
            c.take<uint8_t>(gbuf);
            return c.off;
        };
        constexpr int kRingMin = 16;   // a prestage buffer below this is none (prefill_trim_prestage)
        // the prestage slots chunk T leaves beside its buffers (at most ring_max), -1 when T does not fit at all
        const auto room = [&](int64_t T) -> int64_t {
            const size_t nr = nonring(T);
            if (nr + 256 > lendable) return -1;
            if (T < kPreMinT || gstride == 0) return ring_max;   // (a chunk this short carves no prestage buffer)
            return std::min<int64_t>(ring_max, (int64_t) ((lendable - nr - 256) / gstride));
        };
        // Strata's biggest_chunk: every test rises with T, so the largest chunk that passes is a bisection on the grid
        const auto biggest = [&](int64_t floor) -> int64_t {
            int64_t lo = 1, hi = ceiling / 256, best = 0;
            while (lo <= hi) {
                const int64_t mid = lo + (hi - lo) / 2;
                if (room(mid * 256) >= floor) {
                    best = mid * 256;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            return best;
        };
        int64_t T = 0;
        if (is_auto) {
            T = biggest(ring_max);
            if (T == 0 && ring_max > kRingMin) T = biggest(kRingMin);
            if (T == 0) T = biggest(0);
        } else {
            for (int64_t c = ceiling; c >= 256; c /= 2)
                if (room(c) >= 0) {
                    T = c;
                    break;
                }
            if (T == 0 && ceiling > 0 && ceiling < 256 && room(ceiling) >= 0) T = ceiling;
        }
        const int64_t r = T > 0 ? room(T) : 0;
        return {T, T >= kPreMinT && r >= kRingMin ? (int) r : 0};
    };
    S->lend_pct = pct_env != nullptr ? (int64_t) std::atoi(pct_env) : 90;
    S->lend_pct_fixed = pct_env != nullptr || mb_env != nullptr || !is_auto;
    S->mode = is_auto ? "auto, up to " + std::to_string(auto_max) : mode;
    const auto pick = S->choose(S->lend_pct);
    const int64_t T = pick.first;
    if (T == 0) {
        std::fprintf(stderr, "glm prefill: CUDA%d the expert pool (%lld slots a layer) can lend no prompt chunk "
                             "(--prefill %s) - token by token\n", dev_, (long long) per, mode.c_str());
        prefill_destroy();
        return true;
    }
    if (!is_auto && T != ceiling)
        std::fprintf(stderr, "glm prefill: CUDA%d prompt chunk %lld -> %lld tokens so its buffers fit in the expert "
                             "pool\n", dev_, (long long) ceiling, (long long) T);
    S->T = (int) T;
    const auto pu = prefill_bytes_for((size_t) T);
    S->arena_bytes = pu.first + pu.second;
    S->w16_elems = w16_elems;
    S->w32_elems = w32_elems;
    S->ws_bytes = ws_bytes;
    S->gbuf_bytes = gbuf;
    S->NP = pick.second;
    S->pbuf_bytes = (size_t) pick.second * gstride;
    {
        Carve c;
        c.take<uint8_t>(S->arena_bytes);
        c.take<uint16_t>((size_t) w16_elems);
        c.take<float>((size_t) w32_elems);
        c.take<uint8_t>(ws_bytes);
        c.take<uint8_t>(S->gbuf_bytes);
        c.take<uint8_t>(S->pbuf_bytes);
        S->borrow_bytes = c.off;
    }
    if (cudaHostAlloc((void**) &S->emb_h, (size_t) T * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &S->h_counts, 4096 * sizeof(int), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc((void**) &S->h_base, 4096 * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &S->h_bounds, 8192 * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        (l1_ < g.n_layers && cudaHostAlloc((void**) &S->hop_h, (size_t) 2 * T * 4 * E * sizeof(float),
                                           cudaHostAllocDefault) != cudaSuccess)) {
        cudaGetLastError();
        std::fprintf(stderr, "glm prefill: CUDA%d pinned staging did not allocate - token by token\n", dev_);
        prefill_destroy();
        return true;
    }
    if (has_moe) {
        // the landing ring: ~3% of the free RAM per device (it is pinned before the RAM tier measures what is left,
        // so it comes out of that tier), 12 to 96 slots; halved while the pinning fails.  96 slots let the second
        // GPU of a split read ahead through a whole layer (2x V100, 30 GB RAM: its disk waits 31 s -> 3-6 s, +1%)
        constexpr int kMinLand = PrefillState::NG * PrefillState::GE;
        int want = (int) std::min<int64_t>(96, std::max<int64_t>(kMinLand, (int64_t) (0.03 * (double) avail_ram_bytes() /
                                                                                      (double) gstride)));
        // On an APU these pinned pages compete with the expert pool. A small
        // landing ring suffices for disk reads while the prompt borrows its tail.
        if (F->unified_memory) want = kMinLand;
        if (const char* v = getenv("STRATA_GLM_PREFILL_LAND")) want = std::max(kMinLand, std::atoi(v));
        int n = want;
        while (cudaHostAlloc((void**) &S->gpin, (size_t) n * gstride, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            S->gpin = nullptr;
            if (n == kMinLand) break;
            n = std::max(kMinLand, n / 2);
        }
        if (S->gpin == nullptr) {
            std::fprintf(stderr, "glm prefill: CUDA%d the disk landing ring did not allocate - token by token\n", dev_);
            prefill_destroy();
            return true;
        }
        S->nland = n;
        S->ev_land.assign((size_t) n, nullptr);
    }
    if (cublasCreate(&S->blas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "glm prefill: CUDA%d cublasCreate failed - token by token\n", dev_);
        prefill_destroy();
        return true;
    }
    cublasSetStream(S->blas, F->cs);
    cublasSetMathMode(S->blas, CUBLAS_DEFAULT_MATH);
    S->mq = std::make_unique<mmq::Context>();
    // an event that did not create (a GPU the expert pool filled) stays null and the prompt runs token by token:
    // the staging below records and waits on every one of them, and a null event there is no event at all
    bool events_ok = true;
    const auto create_event = [&](cudaEvent_t& e, unsigned flags) {
        if (events_ok && cudaEventCreateWithFlags(&e, flags) != cudaSuccess) {
            e = nullptr;
            events_ok = false;
        }
    };
    for (int b = 0; b < PrefillState::NG; ++b) {
        create_event(S->ev_ready[b], cudaEventDisableTiming);
        create_event(S->ev_free[b], cudaEventDisableTiming);
    }
    for (auto& e : S->ev_land) create_event(e, cudaEventDisableTiming);
    create_event(S->ev_hop, cudaEventDisableTiming);
    static const bool kv_prefetch = [] {
        const char* v = getenv("STRATA_GLM_KV_PREFETCH");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (kv_ != nullptr && has_dsa && kv_prefetch) {   // (no side stream: the stage copies on the compute stream)
        if (cudaStreamCreateWithFlags(&S->kv_ss, cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError();
            S->kv_ss = nullptr;
        }
        create_event(S->kv_ready, cudaEventDisableTiming);
        create_event(S->kv_free, cudaEventDisableTiming);
    }
    if (S->NP > 0) {
        S->ev_pdone.assign((size_t) (S->NP + PrefillState::kPreEv - 1) / PrefillState::kPreEv, nullptr);
        for (auto& e : S->ev_pdone) create_event(e, cudaEventDisableTiming);
        create_event(S->ev_pready, 0);   // (timed, flags 0: the controller reads when the copies landed)
        create_event(S->ev_pstart, 0);
        create_event(S->ev_pfree, cudaEventDisableTiming);
    }
    if (!events_ok) {
        cudaGetLastError();
        std::fprintf(stderr, "glm prefill: CUDA%d staging events did not create - token by token\n", dev_);
        prefill_destroy();
        return true;
    }
    S->last_cnt.assign((size_t) g.n_layers, {});
    S->trace = getenv("STRATA_GLM_PREFILL_TRACE") != nullptr;
    if (S->trace) {
        S->tr.resize((size_t) g.n_layers);
        for (auto& t : S->tr)
            for (cudaEvent_t* e : {&t.start, &t.plan, &t.pre, &t.c1, &t.end}) cudaEventCreate(e);
    }
    std::fprintf(stderr, "glm prefill: CUDA%d chunks of %d tokens (--prefill %s; mixer sub-batches of %d), borrowing "
                         "%.0f MB of the expert pool while a prompt runs (activations %.0f, weight scratch %.0f, expert "
                         "staging %.0f, prestage %.0f: %d experts); disk landing ring %d experts (%.0f MB pinned)\n",
                 dev_, S->T, S->mode.c_str(),
                 mixer_sub((size_t) S->T, g, has_kda, has_dsa, has_dense, has_moe, S->max_pools, S->sub_env),
                 (double) S->borrow_bytes / 1048576.0, (double) S->arena_bytes / 1048576.0,
                 (double) (w16_elems * 2 + w32_elems * 4) / 1048576.0, (double) S->gbuf_bytes / 1048576.0,
                 (double) S->pbuf_bytes / 1048576.0, S->NP, S->nland, (double) S->nland * (double) gstride / 1048576.0);
    return true;
}

size_t Glm5Model::prefill_borrow_bytes() const { return pf_ ? pf_->borrow_bytes : 0; }

// a pool that cannot lend the whole borrow: the prestage buffer shrinks to what it can (none below 16 slots)
size_t Glm5Model::prefill_trim_prestage(size_t max_borrow) {
    PrefillState* S = pf_;
    if (S == nullptr) return 0;
    if (S->borrow_bytes <= max_borrow || S->pbuf_bytes == 0) return S->borrow_bytes;
    const size_t base = S->borrow_bytes - ((S->pbuf_bytes + 255u) & ~(size_t) 255u);
    int np = base < max_borrow ? (int) ((max_borrow - base) / std::max<size_t>(1, S->gstride)) : 0;
    if (np < 16) np = 0;
    std::fprintf(stderr, "glm prefill: CUDA%d prestage buffer %d -> %d experts (what the pool can lend)\n", dev_, S->NP, np);
    S->NP = np;
    S->pbuf_bytes = (size_t) np * S->gstride;
    S->borrow_bytes = base + ((S->pbuf_bytes + 255u) & ~(size_t) 255u);
    return S->borrow_bytes;
}

bool Glm5Model::prefill_bind(uint8_t* region, size_t bytes, std::string& err) {
    PrefillState* S = pf_;
    if (S == nullptr) return true;
    if (bytes < S->borrow_bytes) {
        err = "glm prefill: the lendable tail (" + std::to_string(bytes >> 20) + " MB) is smaller than the prompt path (" +
              std::to_string(S->borrow_bytes >> 20) + " MB)";
        return false;
    }
    S->region = region;
    S->region_bytes = bytes;
    prefill_carve(S->T);
    return true;
}

// a chunk of T tokens' device buffers: {the rows kept across a layer (and KV streaming's staging copy, kv_stage bytes),
// the largest of the mixers' / FFN's / NextN cache fill's scratch} - the layout prefill_carve lays out
static std::pair<size_t, size_t> chunk_bytes(const Glm5Geometry& g, size_t T, bool has_kda, bool has_dsa,
                                             bool has_dense, bool has_moe, int max_pools, int sub_env, bool mtp,
                                             size_t kv_stage) {
    const size_t E = (size_t) g.n_embd;
    Carve a;
    a.take<float>(T * 4 * E);                    // R
    a.take<float>(T * E);                        // x
    a.take<uint16_t>(T * E);                     // x16
    a.take<float>(T * E);                        // mixer
    a.take<float>(T * E);                        // ffn
    a.take<float>(T * 4);
    a.take<float>(T * 4);
    a.take<float>(T * 16);
    a.take<float>(T);
    a.take<float>(T * 24);
    a.take<int>(T * (size_t) g.n_exp_used);      // iota
    a.take<uint8_t>(kv_stage);
    size_t uni = 0;
    const size_t ts = std::min<size_t>(T, mixer_sub(T, g, has_kda, has_dsa, has_dense, has_moe, max_pools, sub_env));
    if (has_kda) { Carve k; carve_kda(k, ts, g); uni = std::max(uni, k.off); }
    if (has_dsa) { Carve d; carve_dsa(d, ts, g, max_pools); uni = std::max(uni, d.off); }
    if (has_dense) { Carve d; carve_dense(d, ts, g); uni = std::max(uni, d.off); }
    if (has_moe) { Carve m; carve_moe(m, T, (size_t) (sub_env > 0 ? sub_env : kSub), g); uni = std::max(uni, m.off); }
    // the NextN block's cache fill (the last half) carves 4 n_embd + 1.5 n_embd rows of its own
    if (mtp) uni = std::max(uni, T * (size_t) (6 * g.n_embd + g.kv_lora + 2 * g.idx_key) * 4 + 8 * 256);
    return {a.off, uni};
}

Glm5Model::PinnedPrompt Glm5Model::prefill_pinned() const {
    const PrefillState* S = pf_;
    if (S == nullptr) return {};
    const size_t T = (size_t) S->T, E = (size_t) g_.n_embd;
    PinnedPrompt p;
    p.staging = T * E * sizeof(float) + (S->gpin != nullptr ? (size_t) S->nland * S->gstride : 0);
    p.hop = (size_t) 2 * T * 4 * E * sizeof(float);
    return p;
}

std::pair<size_t, size_t> Glm5Model::prefill_bytes_for(size_t T) const {
    const PrefillState* S = pf_;
    const size_t kv_stage = kv_ != nullptr ? (size_t) kv_->n_blocks * kv_->page * kv_->row_bytes : 0;
    return chunk_bytes(g_, T, S->has_kda, S->has_dsa, S->has_dense, S->has_moe, S->max_pools, S->sub_env, mtp_il_ >= 0,
                       kv_stage);
}


// what a part with these layers lends the prompt path for a chunk of T tokens, its prestage buffer aside (the split
// search's startability gate: prefill_setup's layout without a PrefillState)
size_t glm_prefill_lend_bytes(const Glm5Geometry& g, size_t T, bool has_kda, bool has_dsa, bool has_dense,
                              bool has_moe, bool mtp, int64_t max_ctx, size_t gstride, size_t kv_stage) {
    const int max_pools = (int) std::max<int64_t>(1, max_ctx / g.idx_kpool);
    int sub_env = 0;
    if (const char* v = getenv("STRATA_GLM_PREFILL_SUB")) sub_env = std::clamp(std::atoi(v), 16, 8192);
    const auto pu = chunk_bytes(g, T, has_kda, has_dsa, has_dense, has_moe, max_pools, sub_env, mtp, kv_stage);
    const auto ws = weight_scratch(g, has_dsa);
    Carve c;
    c.take<uint8_t>(pu.first + pu.second);
    c.take<uint16_t>((size_t) ws.first);
    c.take<float>((size_t) ws.second);
    c.take<uint8_t>(kPfWorkspace);
    c.take<uint8_t>(has_moe ? (size_t) Glm5Model::PrefillState::NG * Glm5Model::PrefillState::GE * gstride : 0);
    return c.off;
}

// the prompt path's device buffers for chunks of T tokens, carved from the start of the lendable region; returns the
// bytes that layout uses (a short prompt borrows - and evicts - only that much of the pool's tail)
size_t Glm5Model::prefill_carve(int Tn) {
    PrefillState* S = pf_;
    const Glm5Geometry& g = g_;
    const size_t T = (size_t) Tn, E = (size_t) g.n_embd;
    S->sub = mixer_sub(T, g, S->has_kda, S->has_dsa, S->has_dense, S->has_moe, S->max_pools, S->sub_env);
    const auto [pers, uni] = prefill_bytes_for(T);
    Carve c{S->region};
    S->arena = c.take<uint8_t>(pers + uni);
    S->w16 = c.take<uint16_t>((size_t) S->w16_elems);
    S->w32 = c.take<float>((size_t) S->w32_elems);
    S->ws = c.take<uint8_t>(S->ws_bytes);
    S->gbuf = S->gbuf_bytes ? c.take<uint8_t>(S->gbuf_bytes) : nullptr;
    S->pbuf = S->pbuf_bytes && Tn >= kPreMinT ? c.take<uint8_t>(S->pbuf_bytes) : nullptr;   // (a short prompt: none)
    Carve a{S->arena};
    S->R = a.take<float>(T * 4 * E);
    S->x = a.take<float>(T * E);
    S->x16 = a.take<uint16_t>(T * E);
    S->mixer = a.take<float>(T * E);
    S->ffn = a.take<float>(T * E);
    S->pre = a.take<float>(T * 4);
    S->post = a.take<float>(T * 4);
    S->comb = a.take<float>(T * 16);
    S->ss = a.take<float>(T);
    S->mix = a.take<float>(T * 24);
    S->iota = a.take<int>(T * (size_t) g.n_exp_used);
    S->kv_stage = kv_ != nullptr ? a.take<uint8_t>((size_t) kv_->n_blocks * kv_->page * kv_->row_bytes) : nullptr;
    S->uni = S->arena + a.off;
    cublasSetWorkspace(S->blas, S->ws, S->ws_bytes);
    S->T_bound = Tn;
    S->need = c.off;
    return c.off;
}

// The pool's tail -> the prompt path: every lendable slot's expert leaves VRAM (it is only on disk until the decode
// fetches it again), a spare there leaves the spare table, and the device tables learn it before any prompt work.
bool Glm5Model::prefill_lend(std::string& err) {
    FastState* F = fast_;
    PrefillState* S = pf_;
    if (F == nullptr || S == nullptr || S->lent) return true;
    cudaSetDevice(dev_);
    if (!lend_tail(S->need, S->moved, S->dropped, err)) return false;
    // the borrowed memory holds expert bytes: what the prompt path reads before writing is set up again
    mmq::iota(S->iota, (int64_t) S->T_bound * g_.n_exp_used, F->cs);
    for (int b = 0; b < PrefillState::NG; ++b) cudaEventRecord(S->ev_free[b], F->cs);
    if (S->ev_pfree) cudaEventRecord(S->ev_pfree, F->cs);
    S->pre_layer = -1;
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm prefill lend", err)) return false;
    S->lent = true;
    return true;
}

// The on-demand vision encoder (serve VLEND): the whole tail is emptied the same way and its memory freed, so the
// encoder's process can allocate it; the prompt path waits (token by token) until vision_reclaim gives it back.
bool Glm5Model::vision_lend(size_t& bytes, std::string& err) {
    FastState* F = fast_;
    bytes = 0;
    if (F == nullptr || !vis_lend_ok_) {
        err = "this engine has no pool tail to lend to the vision encoder";
        return false;
    }
    if (vis_lent_) {
        bytes = F->xpool_bytes;
        return true;
    }
    cudaSetDevice(dev_);
    uint64_t moved = 0, dropped = 0;
    if (!lend_tail(F->xpool_bytes, moved, dropped, err)) return false;
    cudaFree(F->xpool);
    F->xpool = nullptr;
    for (auto& P : F->lp) P.xbase = nullptr;
    vis_lent_ = true;
    bytes = F->xpool_bytes;
    std::fprintf(stderr, "glm fast: CUDA%d %.2f GB lent to the vision encoder (%llu experts moved to colder slots, "
                         "%llu dropped)\n", dev_, (double) bytes / 1073741824.0, (unsigned long long) moved,
                 (unsigned long long) (dropped - moved));
    return true;
}

size_t Glm5Model::vision_lend_bytes() const { return fast_ != nullptr && vis_lend_ok_ ? fast_->xpool_bytes : 0; }

// ... and back once the encoder has ended: the tail is allocated again (false while the memory is still taken) and
// its slots are free - the decode's misses refill them
bool Glm5Model::vision_reclaim(std::string& err) {
    FastState* F = fast_;
    if (F == nullptr || !vis_lent_) return true;
    cudaSetDevice(dev_);
    if (cudaMalloc(&F->xpool, F->xpool_bytes) != cudaSuccess) {
        cudaGetLastError();
        F->xpool = nullptr;
        err = "the pool's tail is not free yet";
        return false;
    }
    uint8_t* b = F->xpool;
    for (int il = l0_; il < lt_; ++il) {
        if (!F->L[(size_t) il].moe) continue;
        auto& P = F->lp[(size_t) il];
        P.xbase = b;
        b += (size_t) (P.n - P.n_main) * P.stride;
    }
    if (pf_ != nullptr && !prefill_bind(F->xpool, F->xpool_bytes, err)) return false;
    {
        std::lock_guard<std::mutex> lk(F->mu);
        for (int il = l0_; il < lt_; ++il) {
            auto& P = F->lp[(size_t) il];
            for (int s = P.n_main; s < P.n; ++s)
                if (P.st[(size_t) s] == FastState::kLent) P.st[(size_t) s] = FastState::kFree;
        }
    }
    vis_lent_ = false;
    std::fprintf(stderr, "glm fast: CUDA%d the vision encoder's %.2f GB are back in the expert pool\n", dev_,
                 (double) F->xpool_bytes / 1073741824.0);
    return true;
}

// The tail's slots below xpool + limit leave the decode: a resident expert takes the slot of a colder main-slot
// resident (a device copy) or goes; a spare there leaves the spare table; the slots are kLent after this.
bool Glm5Model::lend_tail(size_t limit, uint64_t& moved, uint64_t& dropped, std::string& err) {
    FastState* F = fast_;
    // background moves in flight land first and no new ones start (fast_boundary step 5): a slot with a copy on its
    // way would otherwise be lent under it
    F->bg_hold = true;
    // Retire metadata as well as DMA. With the hold set, the boundary cannot start replacement moves.
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->copy), "glm lending copies", err) || !fast_boundary(err))
        return false;
    if (!F->bg.empty() || !F->draining.empty()) {
        err = "glm lending: tier moves were not retired";
        return false;
    }
    const int NE = g_.n_expert;
    {
        std::lock_guard<std::mutex> lk(F->mu);
        int nu = 0;
        bool ok = true;
        const auto flush = [&]() {
            if (nu == 0 || !ok) return;
            ok = glmfast::cuda_ok(cudaMemcpyAsync(F->upd_key_d, F->upd_key_h, (size_t) nu * sizeof(int),
                                                  cudaMemcpyHostToDevice, F->cs), "glm lending keys", err) &&
                 glmfast::cuda_ok(cudaMemcpyAsync(F->upd_val_d, F->upd_val_h,
                                                  (size_t) nu * sizeof(unsigned long long), cudaMemcpyHostToDevice,
                                                  F->cs), "glm lending values", err);
            if (ok) {
                gf::tab_update(F->tab, F->upd_key_d, F->upd_val_d, nu, F->cs);
                ok = glmfast::cuda_ok(cudaGetLastError(), "glm lending table update", err);
            }
            const cudaError_t e = cudaStreamSynchronize(F->cs);   // pinned sources may now be reused
            if (ok) ok = glmfast::cuda_ok(e, "glm lending table sync", err);
            nu = 0;
        };
        // (every key is edited at most once here, so a batch never carries two values for one key)
        const auto upd = [&](size_t k, unsigned long long v) {
            if (!ok) return;
            if (nu == FastState::kMaxUpd) flush();
            if (!ok) return;
            F->upd_key_h[nu] = (int) k;
            F->upd_val_h[nu] = v;
            ++nu;
        };
        // STRATA_GLM_LEND_DROP=1: drop the lent slots' experts as before (instead of moving them, below)
        static const bool lend_drop = getenv("STRATA_GLM_LEND_DROP") != nullptr;
        // an expert that leaves VRAM here goes to the RAM tier when it has room (a free slot, else a colder entry):
        // the decode after the prompt then finds it in RAM, not on disk (with the experts all in VRAM - a split
        // across many GPUs - the RAM tier holds little else, and every lent expert was a ~20 ms disk read back)
        std::vector<char> parked;   // (keys parked by this lending: never its victims - one table edit per key)
        const auto park = [&](int il, int s, int key) -> bool {
            if (lend_drop || F->layer_rc[(size_t) il] < 0) return false;
            if (parked.empty()) parked.assign(F->cnt.size(), 0);
            auto& R = F->rc[(size_t) F->layer_rc[(size_t) il]];
            int rs = -1;
            for (int s2 = 0; s2 < R.n && rs < 0; ++s2)
                if (R.st[(size_t) s2] == FastState::kRFree) rs = s2;
            if (rs < 0) {
                uint32_t bc = F->cnt[(size_t) key];
                for (int s2 = 0; s2 < R.n; ++s2)
                    if (R.st[(size_t) s2] == FastState::kRHold && !parked[(size_t) R.key[(size_t) s2]] &&
                        F->cnt[(size_t) R.key[(size_t) s2]] < bc) {
                        bc = F->cnt[(size_t) R.key[(size_t) s2]];
                        rs = s2;
                    }
                if (rs < 0) return false;
                const int old = R.key[(size_t) rs];
                upd(F->rtab_key((size_t) old), 0ull);
                F->ram_of[(size_t) old] = -1;
                F->left[(size_t) old] = 2;
            }
            // on the compute stream: the prompt's work behind it cannot overwrite the slot before the copy read it
            if (!glmfast::cuda_ok(cudaMemcpyAsync(R.base + (size_t) rs * R.stride,
                                                   F->lp[(size_t) il].slot_ptr(s), F->L[(size_t) il].blob,
                                                   cudaMemcpyDeviceToHost, F->cs), "glm lending park", err)) {
                ok = false;
                return false;
            }
            R.key[(size_t) rs] = key;
            R.tick[(size_t) rs] = F->clock;
            R.st[(size_t) rs] = FastState::kRHold;
            F->ram_of[(size_t) key] = rs;
            parked[(size_t) key] = 1;
            upd(F->rtab_key((size_t) key), (unsigned long long) (R.base + (size_t) rs * R.stride));
            ++F->diag_lend_park;
            return true;
        };
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            auto& P = F->lp[(size_t) il];
            // the layer's main-slot residents, coldest first: a lent slot's expert moves over the coldest one that is
            // colder than it (a device copy) - the prompt costs the decode its coldest experts, not whichever sat in
            // the borrowed tail (a chat turn used to drop up to ~145 warm experts to disk)
            std::vector<std::pair<uint64_t, int>> cold;   // (count << 32 | recency rank, slot)
            if (!lend_drop)
                for (int s = 0; s < P.n_main; ++s)
                    if (P.st[(size_t) s] == FastState::kResident && P.key[(size_t) s] >= 0)
                        cold.push_back({(uint64_t) F->cnt[(size_t) il * NE + P.key[(size_t) s]], s});
            std::sort(cold.begin(), cold.end(), [&](const auto& a, const auto& b) {
                return a.first != b.first ? a.first < b.first : P.tick[(size_t) a.second] < P.tick[(size_t) b.second];
            });
            size_t ci = 0;
            for (int s = P.n_main; s < P.n; ++s) {
                // a slot past the bytes this lending uses (a prompt's layout) stays with the decode
                if (P.slot_ptr(s) >= F->xpool + limit) continue;
                char& st = P.st[(size_t) s];
                if (st == FastState::kResident && P.key[(size_t) s] >= 0) {
                    const int key = il * NE + P.key[(size_t) s];
                    if (ci < cold.size() && cold[ci].first < (uint64_t) F->cnt[(size_t) key]) {
                        const int v = cold[ci++].second;   // the coldest: it goes, this expert takes its slot
                        const int vkey = il * NE + P.key[(size_t) v];
                        const bool parked = park(il, v, vkey);   // (read before the copy below overwrites the slot)
                        if (!ok || !glmfast::cuda_ok(cudaMemcpyAsync(P.slot_ptr(v), P.slot_ptr(s),
                                                                     F->L[(size_t) il].blob, cudaMemcpyDeviceToDevice,
                                                                     F->cs), "glm lending relocation", err))
                            return false;
                        upd(F->tab_key((size_t) vkey), 0ull);
                        F->slot_of[(size_t) vkey] = -1;
                        if (!parked) F->left[(size_t) vkey] = 3;
                        upd(F->tab_key((size_t) key), (unsigned long long) P.slot_ptr(v));
                        F->slot_of[(size_t) key] = v;
                        P.key[(size_t) v] = P.key[(size_t) s];
                        P.tick[(size_t) v] = P.tick[(size_t) s];
                        ++moved;
                    } else {
                        upd(F->tab_key((size_t) key), 0ull);
                        F->slot_of[(size_t) key] = -1;
                        if (!park(il, s, key)) F->left[(size_t) key] = 3;
                        if (!ok) return false;
                    }
                    ++dropped;
                    ++F->diag_lend;
                } else if (st == FastState::kSpare) {
                    for (int j = 0; j < gf::kSpares; ++j)
                        if (P.spare[j] == s) {
                            P.spare[j] = -1;
                            upd(F->spare_key(il, j), 0ull);
                        }
                }
                P.key[(size_t) s] = -1;
                st = FastState::kLent;
            }
        }
        flush();
        if (!ok) return false;
    }
    F->bg_hold = false;
    return true;
}

// ... and back: the slots are free again; the decode's boundaries hand them out as spares (misses refill them)
bool Glm5Model::prefill_return(std::string& err) {
    FastState* F = fast_;
    PrefillState* S = pf_;
    if (F == nullptr || S == nullptr || !S->lent) return true;
    cudaSetDevice(dev_);
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm return compute", err) ||
        !glmfast::cuda_ok(cudaStreamSynchronize(F->copy), "glm return copies", err) ||
        (S->kv_ss && !glmfast::cuda_ok(cudaStreamSynchronize(S->kv_ss), "glm return KV stage", err))) return false;
    S->kv_next = -1;   // (a prompt that ended early leaves a staged layer nobody waits for)
    std::lock_guard<std::mutex> lk(F->mu);
    for (int il = l0_; il < lt_; ++il) {
        auto& P = F->lp[(size_t) il];
        for (int s = P.n_main; s < P.n; ++s)
            if (P.st[(size_t) s] == FastState::kLent) P.st[(size_t) s] = FastState::kFree;
    }
    S->lent = false;
    return true;
}

void Glm5Model::prefill_destroy() {
    PrefillState* S = pf_;
    if (S == nullptr) return;
    cudaSetDevice(dev_);
    if (fast_ && fast_->cs) cudaStreamSynchronize(fast_->cs);
    if (fast_ && fast_->copy) cudaStreamSynchronize(fast_->copy);
    if (S->prof && !S->pacc.empty()) {
        double tot = 0;
        for (auto& kv : S->pacc) tot += kv.second;
        std::fprintf(stderr, "glm prefill prof CUDA%d (%lld tokens, %.0f ms of stream time; waits on disk %.0f ms, plan %.0f "
                             "ms host; disk experts used %llu, read %llu):\n", dev_, (long long) S->tokens, tot, S->ms_disk,
                     S->ms_plan, (unsigned long long) S->staged_disk, (unsigned long long) S->disk_issued);
        std::vector<std::pair<double, std::string>> v;
        for (auto& kv : S->pacc) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        for (auto& e : v)
            std::fprintf(stderr, "  %-14s %9.1f ms  %5.1f%%  (%.3f ms/token)\n", e.second.c_str(), e.first,
                         100.0 * e.first / std::max(1e-9, tot), e.first / (double) std::max<int64_t>(1, S->tokens));
    }
    for (auto e : S->pev) cudaEventDestroy(e);
    for (auto& t : S->tr)
        for (cudaEvent_t e : {t.start, t.plan, t.pre, t.c1, t.end})
            if (e) cudaEventDestroy(e);
    if (S->xs) cudaStreamSynchronize(S->xs);
    if (S->x_h) cudaFreeHost(S->x_h);
    if (S->rt_h) cudaFreeHost(S->rt_h);
    if (S->out_h) cudaFreeHost(S->out_h);
    if (S->ev_x) cudaEventDestroy(S->ev_x);
    if (S->ev_out) cudaEventDestroy(S->ev_out);
    if (S->ev_c0) cudaEventDestroy(S->ev_c0);
    if (S->ev_c1) cudaEventDestroy(S->ev_c1);
    for (auto e : S->ev_pdone)
        if (e) cudaEventDestroy(e);
    if (S->ev_pready) cudaEventDestroy(S->ev_pready);
    if (S->ev_pstart) cudaEventDestroy(S->ev_pstart);
    if (S->ev_pfree) cudaEventDestroy(S->ev_pfree);
    if (S->xs) cudaStreamDestroy(S->xs);
    if (S->kv_ss) {
        cudaStreamSynchronize(S->kv_ss);
        cudaStreamDestroy(S->kv_ss);
    }
    if (S->kv_ready) cudaEventDestroy(S->kv_ready);
    if (S->kv_free) cudaEventDestroy(S->kv_free);
    S->mq.reset();
    if (S->blas) cublasDestroy(S->blas);
    for (int b = 0; b < PrefillState::NG; ++b) {
        if (S->ev_ready[b]) cudaEventDestroy(S->ev_ready[b]);
        if (S->ev_free[b]) cudaEventDestroy(S->ev_free[b]);
    }
    for (auto e : S->ev_land)
        if (e) cudaEventDestroy(e);
    if (S->ev_hop) cudaEventDestroy(S->ev_hop);
    if (S->gpin) cudaFreeHost(S->gpin);
    if (S->emb_h) cudaFreeHost(S->emb_h);
    if (S->hop_h) cudaFreeHost(S->hop_h);
    if (S->h_counts) cudaFreeHost(S->h_counts);
    if (S->h_base) cudaFreeHost(S->h_base);
    if (S->h_bounds) cudaFreeHost(S->h_bounds);
    delete S;
    pf_ = nullptr;
}

// ---------------------------------------------------------------- one half's layers over a chunk
bool Glm5Model::prefill_half(int64_t p0, int T, std::string& err, const int32_t* next_ids) {
    FastState* F = fast_;
    PrefillState* S = pf_;
    const Glm5Geometry& g = g_;
    const cudaStream_t s = F->cs;
    const int E = g.n_embd, DI = g.d_inner(), HD = g.kda_head_dim, K = g.n_exp_used;
    const float one = 1.0f, zero = 0.0f;

    // Y[t][N] (row stride ldy) = alpha * X[t][K] (F32, row stride ldx) . W[N][K]^T, W BF16 (widened in slices)
    const auto sgemm_bf16 = [&](const uint16_t* W, int N, int Kd, const float* X, int ldx, float* Y, int ldy, int Tn,
                                float alpha) {
        const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, S->w32_elems / Kd));
        for (int r0 = 0; r0 < N; r0 += rows) {
            const int n = std::min(rows, N - r0);
            gb::bf16_to_f32(W + (size_t) r0 * Kd, S->w32, (int64_t) n * Kd, s);
            blas_ck(cublasSgemm(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, Tn, Kd, &alpha, S->w32, Kd, X, ldx, &zero, Y + r0,
                                ldy),
                    "sgemm");
        }
    };
    // Y[t][N] = X16[t][K] . deq(W)[N][K]^T (tensor cores, F32 accumulate); beta = 1 adds to Y.  A BF16 weight (a
    // projection the GGUF keeps unquantized) is narrowed to FP16 the same way
    const auto hgemm_q = [&](const WSlot& w, int N, int Kd, const uint16_t* X16, int ldx, float* Y, int ldy, int Tn,
                             float beta) {
#if defined(STRATA_USE_HIP)
        if (!S->lt_tried) {
            S->lt_tried = true;
            if (getenv("STRATA_HIPBLASLT_TUNING")) {
                auto lt = std::make_unique<strata::prefill::Gemm>();
                std::string lt_err;
                if (lt->init_external(F->cs, S->w16, S->w16_elems, S->ws, S->ws_bytes, lt_err))
                    S->lt = std::move(lt);
                else
                    std::fprintf(stderr, "glm prefill: hipBLASLt GEMM unavailable (%s)\n", lt_err.c_str());
            }
        }
        if (S->lt) S->lt->rebind(S->w16, S->w16_elems, S->ws, S->ws_bytes);
#endif
        const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, S->w16_elems / Kd));
        for (int r0 = 0; r0 < N; r0 += rows) {
            const int n = std::min(rows, N - r0);
            if (w.type == gf::kTypeBF16)
                gb::bf16_to_f16((const uint16_t*) w.q + (size_t) r0 * Kd, S->w16, (int64_t) n * Kd, s);
            else
                strata::kernels::dequant_f16(w.type, w.q, r0, n, Kd, S->w16, s);
#if defined(STRATA_USE_HIP)
            if (S->lt && ldx == Kd) {
                S->lt->f16(X16, S->w16, Y + r0, Tn, n, Kd, ldy, beta);
                continue;
            }
#endif
            blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, Tn, Kd, &one, S->w16, CUDA_R_16F, Kd, X16,
                                 CUDA_R_16F, ldx, &beta, Y + r0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "gemm f16");
        }
    };
    // the mHC read half for the whole chunk: (write half of the previous block,) mix dots, gates, x / x16
    const auto hc_read = [&](const float* block_out, const uint16_t* fn, const float* scale, const float* base,
                             const float* norm) {
        gb::hc_update(block_out, S->R, S->post, S->comb, S->ss, T, E, s);
        sgemm_bf16(fn, 24, 4 * E, S->R, 4 * E, S->mix, 24, T, 1.0f);
        gb::HcFinishArgs h;
        h.mix = S->mix;
        h.ss = S->ss;
        h.R = S->R;
        h.w_scale = scale;
        h.w_base = base;
        h.norm_w = norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = E;
        h.T = T;
        h.pre = S->pre;
        h.post = S->post;
        h.comb = S->comb;
        h.x = S->x;
        h.x16 = S->x16;
        gb::hc_finish(h, s);
    };

    // GLM_CB_DIR seam dumps of the chunk's LAST row, under the token path's names (debug only, synchronous)
    static const char* cb_dir = getenv("GLM_CB_DIR");
    // A single-device prompt chunk only conditions later tokens; decode computes the last prompt token and its
    // logits separately. The terminal layer's outputs have no reader unless NextN is loaded. Keep its mixer/cache
    // updates, but omit the output projection and FFN (including both mHC write halves). Split boundaries and seam
    // dumps consume these rows, so keep their full path. STRATA_GLM_PREFILL_TAIL_SKIP=0 is the A/B baseline.
    const char* tail_env = getenv("STRATA_GLM_PREFILL_TAIL_SKIP");
    const bool tail_skip = l0_ == 0 && l1_ == g.n_layers && split_next_ == nullptr && mtp_il_ < 0 &&
                           (cb_dir == nullptr || !cb_dir[0]) && (tail_env == nullptr || std::atoi(tail_env) != 0);
    const auto dump_row = [&](const std::string& name, const float* rows, int width, int row = -1) {
        if (cb_dir == nullptr || !cb_dir[0]) return;
        cudaStreamSynchronize(s);
        std::vector<float> buf((size_t) width);
        cudaMemcpy(buf.data(), rows + (size_t) (row < 0 ? T - 1 : row) * width, (size_t) width * sizeof(float),
                   cudaMemcpyDeviceToHost);
        if (FILE* f = std::fopen((std::string(cb_dir) + "/" + name + ".f32").c_str(), "wb")) {
            std::fwrite(buf.data(), 4, buf.size(), f);
            std::fclose(f);
        }
    };

    // ---- the disk reader: one thread for the chunk, reading the experts the MoE layers append - (layer, expert) in
    //      order - into the pinned landing ring, while the resident product and the RAM-tier groups run (Mercury,
    //      2.6k-token prompt: 5.1 -> 4.5 ms/token).  A layer appends its disk-only experts once its plan is known.
    //      From kPredT tokens on (STRATA_GLM_PREFILL_PRED_T=<tokens>, 0 = never), a layer also appends the NEXT MoE
    //      layer's whole disk-only set (a superset of what it will route), so those reads overlap its attention too.
    //      With the 12-slot ring it was slower (the reader could not get ahead); with the deep ring it pays from 2k
    //      chunks (Mercury 16k prompt at 2048-token chunks: 394 -> 417 tok/s).  A slot is refilled once the copy that
    //      read it ran (ev_land) and the layers moved past it.
    const int KL = S->nland;
    constexpr int kPredT = 1024;
    struct DiskQ {
        std::mutex mu;
        std::vector<std::pair<int, int>> q;
        std::atomic<int> n{0}, landed{0}, copied{0};
        std::atomic<bool> stop{false};
        int push(int il, int e) {
            std::lock_guard<std::mutex> lk(mu);
            q.emplace_back(il, e);
            n.store((int) q.size(), std::memory_order_release);
            return (int) q.size() - 1;
        }
    } dq;
    // STRATA_GLM_DISK_QD=<n>: experts read at once (one batch across the workers) when they are queued and their
    // slots free - an NVMe gives more with a few in flight (Uranus 1.87 -> 2.2 GB/s at 2, Mercury 2.65 -> 2.96).
    // Up to 32: on Windows each read has a fixed cost, so more in flight matter more (RX 7900 XTX, PCIe 4 NVMe,
    // with STRATA_GLM_READ_CHUNKS=1: 1.66 GB/s at 2 -> 4.27 GB/s at 16)
    static constexpr int kMaxDiskQd = 32;
    static const int disk_qd = [] {
        const char* v = getenv("STRATA_GLM_DISK_QD");
        return std::max(1, std::min(kMaxDiskQd, v ? std::atoi(v) : 2));
    }();
    std::thread reader([&] {
        cudaSetDevice(dev_);
        // each part in pieces across the workers, as the decode's disk path (STRATA_GLM_READ_CHUNKS)
        static const int kChunks = [] {
            const char* v = getenv("STRATA_GLM_READ_CHUNKS");
            return std::max(1, std::min(16, v ? std::atoi(v) : 8));
        }();
        for (int k = 0;;) {
            while (dq.n.load(std::memory_order_acquire) <= k ||
                   (k >= KL && dq.copied.load(std::memory_order_acquire) < k - KL + 1)) {
                if (dq.stop.load()) return;
                std::this_thread::yield();
            }
            // this one, and the ones after it that are queued and whose slots are free already
            int nb = 1;
            while (nb < disk_qd && dq.n.load(std::memory_order_acquire) > k + nb &&
                   (k + nb < KL || dq.copied.load(std::memory_order_acquire) >= k + nb - KL + 1))
                ++nb;
            std::pair<int, int> it[kMaxDiskQd];
            uint8_t* land[kMaxDiskQd];
            for (int b = 0; b < nb; ++b) {
                cudaEventSynchronize(S->ev_land[(k + b) % KL]);   // the slot's last copy (this chunk's or earlier)
                std::lock_guard<std::mutex> lk(dq.mu);
                it[b] = dq.q[(size_t) (k + b)];
                land[b] = S->gpin + (size_t) ((k + b) % KL) * S->gstride;
            }
            F->workers->run(nb * 3 * kChunks, [&](int job) {
                const int b = job / (3 * kChunks), part = job % (3 * kChunks);
                fast_read_part(it[b].first, it[b].second, part / kChunks, land[b], part % kChunks, kChunks);
            });
            k += nb;
            dq.landed.store(k, std::memory_order_release);
        }
    });
    struct Joiner {
        std::thread& t;
        DiskQ& q;
        uint64_t& issued;
        ~Joiner() {
            q.stop.store(true);
            if (t.joinable()) t.join();
            issued += (uint64_t) q.n.load();
        }
    } joiner{reader, dq, S->disk_issued};
    // STRATA_GLM_PREFILL_PRED_T=<tokens>: the chunk size from which the next layer's disk set is read ahead (0 = never)
    static const int pred_t = [] {
        const char* v = getenv("STRATA_GLM_PREFILL_PRED_T");
        return v ? std::atoi(v) : kPredT;
    }();
    int pred_layer = -1, pred0 = 0;   // the predicted segment: its layer, first index and experts (ascending)
    std::vector<int> pred_e;
    // the experts of layer il that are neither resident in its main slots nor held by the RAM tier (ascending)
    const auto disk_only = [&](int il) {
        std::vector<int> v;
        const auto& P = F->lp[(size_t) il];
        const auto& RC = F->rc[(size_t) F->layer_rc[(size_t) il]];
        std::vector<char> res((size_t) g.n_expert, 0);
        std::lock_guard<std::mutex> lk(F->mu);
        for (int sl = 0; sl < P.n_main; ++sl)
            if (P.st[(size_t) sl] == FastState::kResident && P.key[(size_t) sl] >= 0) res[(size_t) P.key[(size_t) sl]] = 1;
        for (int e = 0; e < g.n_expert; ++e) {
            if (res[(size_t) e]) continue;
            const int rs = F->ram_of[(size_t) il * g.n_expert + e];
            if (rs >= 0 && RC.st[(size_t) rs] == FastState::kRHold) continue;
            const int vs = F->slot_of[(size_t) il * g.n_expert + e];   // a tail slot this prompt did not borrow
            if (vs >= 0 && P.st[(size_t) vs] == FastState::kResident) continue;
            v.push_back(e);
        }
        return v;
    };

    // ---- PRESTAGING: layer il's predicted experts -> pbuf on the copy stream, behind what it carries (the layer
    //      before's staged groups) and once the layer before has read pbuf (ev_pfree)
    const auto next_moe = [&](int il) {
        while (il < l1_ && !F->L[(size_t) il].moe) ++il;
        return il;
    };
    const auto prestage = [&](int il) {
        S->pre_layer = -1;
        S->pre_e.clear();
        if (S->pbuf == nullptr || T < kPreMinT || il >= l1_ || !F->L[(size_t) il].moe || F->layer_rc[(size_t) il] < 0 ||
            (tail_skip && il == g.n_layers - 1))   // the skipped terminal FFN reads no experts
            return;
        const auto& Ly = F->L[(size_t) il];
        const auto& P = F->lp[(size_t) il];
        const auto& RC = F->rc[(size_t) F->layer_rc[(size_t) il]];
        const int NE = g.n_expert;
        // an expert's predicted rows: its share of this machine's routing (the usage profile, which the prompts
        // feed) and of the layer's last chunk, half each
        double us = 0.0, ls = 0.0;
        for (int e = 0; e < NE; ++e) us += F->usage[(size_t) il * NE + e];
        const auto& lc = S->last_cnt[(size_t) il];
        for (int c : lc) ls += c;
        struct Cand {
            double pred;
            int e, rs;
        };
        std::vector<Cand> cand;
        {
            std::lock_guard<std::mutex> lk(F->mu);
            for (int e = 0; e < NE; ++e) {
                const size_t key = (size_t) il * NE + e;
                const int vs = F->slot_of[key];
                if (vs >= 0 && vs < P.n && P.st[(size_t) vs] == FastState::kResident && P.key[(size_t) vs] == e)
                    continue;   // in VRAM (a main slot, or a tail slot this prompt did not borrow)
                const int rs = F->ram_of[key];
                if (rs < 0 || RC.st[(size_t) rs] != FastState::kRHold) continue;   // (disk: the reader's)
                double sh = 0.0, w = 0.0;
                if (us > 0.0) sh += F->usage[key] / us, w += 1.0;
                if (ls > 0.0 && !lc.empty()) sh += lc[(size_t) e] / ls, w += 1.0;
                const double pred = (w > 0.0 ? sh / w : 1.0 / NE) * (double) T * K;
                if (pred >= kPreMinRows) cand.push_back(Cand{pred, e, rs});
            }
        }
        // the layer's count: the buffer's, or (STRATA_GLM_PRESTAGE_ADAPT=1) what its mixer's window took before
        static const bool adapt = getenv("STRATA_GLM_PRESTAGE_ADAPT") != nullptr && std::atoi(getenv("STRATA_GLM_PRESTAGE_ADAPT")) != 0;
        const int cap = std::min(S->NP, (int) std::min<size_t>(S->pbuf_bytes / P.stride,
                                                                S->ev_pdone.size() * PrefillState::kPreEv));
        const double rate = S->pre_rate[Ly.recr ? 0 : 1];
        const int want = !adapt || rate <= 0.0 ? cap : std::max(std::min(cap, 16), std::min(cap, (int) std::lround(rate * T)));
        const int n = (int) std::min<size_t>((size_t) want, cand.size());
        if (n == 0) return;
        std::partial_sort(cand.begin(), cand.begin() + n, cand.end(),
                          [](const Cand& a, const Cand& b) { return a.pred > b.pred; });
        cudaStreamWaitEvent(F->copy, S->ev_pfree, 0);
        cudaEventRecord(S->ev_pstart, F->copy);
        for (int j = 0; j < n; ++j) {
            cudaMemcpyAsync(S->pbuf + (size_t) j * P.stride, RC.base + (size_t) cand[(size_t) j].rs * RC.stride, Ly.blob,
                            cudaMemcpyHostToDevice, F->copy);
            if ((j + 1) % PrefillState::kPreEv == 0 || j + 1 == n)
                cudaEventRecord(S->ev_pdone[(size_t) j / PrefillState::kPreEv], F->copy);
            S->pre_e.push_back(cand[(size_t) j].e);
        }
        cudaEventRecord(S->ev_pready, F->copy);
        if (S->trace) {
            cudaEventRecord(S->tr[(size_t) il].pre, F->copy);
            S->tr[(size_t) il].pre_n = n;
        }
        S->pre_layer = il;
        S->pre_issued += (uint64_t) n;
    };

    // whether layer il's plan may give experts to the CPU (STRATA_GLM_PREFILL_CPU=0: never)
    static const bool pf_cpu = [] {
        const char* v = getenv("STRATA_GLM_PREFILL_CPU");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const auto cpu_lane = [&](int il) {
        return pf_cpu && F->cpu_pool && F->cpu_c_ms > 0.0 && F->cpu_p_ms > 2.0 * F->cpu_c_ms &&
               (size_t) il < F->cpu_fmt.size() && F->cpu_fmt[(size_t) il].n_ff > 0;
    };
    // the CPU lane's stream (x down, the rows' tokens down) and its pinned buffers
    const auto cpu_bufs = [&](int cpu_nrows) {
        bool ok = true;
        if (S->xs == nullptr)
            ok = cudaStreamCreateWithFlags(&S->xs, cudaStreamNonBlocking) == cudaSuccess &&
                 cudaEventCreateWithFlags(&S->ev_x, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&S->ev_out, cudaEventDisableTiming) == cudaSuccess;
        if (ok && S->x_h_n < (size_t) T * E) {
            if (S->x_h) cudaFreeHost(S->x_h);
            S->x_h_n = (size_t) T * E;
            ok = cudaHostAlloc((void**) &S->x_h, S->x_h_n * sizeof(float), cudaHostAllocDefault) == cudaSuccess;
        }
        if (ok && S->rows_h_n < (size_t) cpu_nrows) {
            if (S->rt_h) cudaFreeHost(S->rt_h);
            if (S->out_h) cudaFreeHost(S->out_h);
            // (what this layer needs and a quarter more, at most a chunk's routes: T x K rows of n_embd floats
            // would be 2 GB of pinned memory at 16K tokens)
            S->rows_h_n = std::min((size_t) T * K, (size_t) cpu_nrows + (size_t) cpu_nrows / 4 + 64);
            ok = cudaHostAlloc((void**) &S->rt_h, S->rows_h_n * sizeof(int), cudaHostAllocDefault) == cudaSuccess &&
                 cudaHostAlloc((void**) &S->out_h, S->rows_h_n * E * sizeof(float), cudaHostAllocDefault) == cudaSuccess;
        }
        return ok;
    };

    // KV streaming: a DSA layer's cache before the chunk (whole pages: a chunk starting inside one finds it complete)
    // goes into the stage on kv_ss once the work queued so far has read the stage, so the layers before that DSA
    // layer overlap the copy (glm DSA layers have recurrent ones between them)
    const auto kv_stage_bytes = [&]() { return (size_t) ((p0 + kv_->page - 1) / kv_->page) * kv_->page * kv_->row_bytes; };
    const auto kv_dsa_from = [&](int il) {
        for (; il < l1_; ++il)
            if (!g.is_recr(il)) return il;
        return -1;
    };
    const auto kv_prefetch = [&](int il) -> bool {
        if (S->kv_ss == nullptr || kv_ == nullptr || p0 <= 0 || il < 0 || kv_->host[(size_t) il] == nullptr) return true;
        if (!glmfast::cuda_ok(cudaEventRecord(S->kv_free, s), "glm prefill: the KV stage", err) ||
            !glmfast::cuda_ok(cudaStreamWaitEvent(S->kv_ss, S->kv_free, 0), "glm prefill: the KV stage", err) ||
            !glmfast::cuda_ok(cudaMemcpyAsync(S->kv_stage, kv_->host[(size_t) il], kv_stage_bytes(), cudaMemcpyDefault,
                                              S->kv_ss), "glm prefill: the KV stage", err) ||
            !glmfast::cuda_ok(cudaEventRecord(S->kv_ready, S->kv_ss), "glm prefill: the KV stage", err))
            return false;
        S->kv_next = il;
        return true;
    };
    S->mark("start", s);
    if (!kv_prefetch(kv_dsa_from(l0_))) return false;   // (the layers before the first DSA layer run meanwhile)
    prestage(next_moe(l0_));   // (the dense layers before the first MoE one run meanwhile)
    for (auto& t : S->tr) t.on = false;
    for (int il = l0_; il < l1_; ++il) {
        progress_at("the prompt: layer", il);   // a layer is the serve watchdog's heartbeat (issue #40)
        progress_beat();
        const auto& Ly = F->L[(size_t) il];
        const bool skip_output = tail_skip && il == g.n_layers - 1;
        if (S->trace) cudaEventRecord(S->tr[(size_t) il].start, s);
        hc_read(il == l0_ ? nullptr : S->ffn, Ly.hc_attn_fn, Ly.hc_attn_scale, Ly.hc_attn_base, Ly.attn_norm);
        S->mark("hc", s);
        dump_row("attn_norm-" + std::to_string(il), S->x, E);
        // KV streaming: the layer's cache before the chunk, from its host copy into the staging copy (whole pages: a
        // chunk starting inside one finds it complete), which the chunk's rows are written into too and the attention
        // reads; the host copy and the resident slots get the chunk's rows as well (dsa_prep)
        uint8_t* const kv_host = !Ly.recr && kv_ != nullptr ? kv_->host[(size_t) il] : nullptr;
        if (kv_host != nullptr && p0 > 0) {
            if (S->kv_next == il) {   // staged on kv_ss meanwhile
                if (!glmfast::cuda_ok(cudaStreamWaitEvent(s, S->kv_ready, 0), "glm prefill: the KV stage wait", err))
                    return false;
                S->kv_next = -1;
                S->mark("kv_wait", s);
            } else {
                if (!glmfast::cuda_ok(cudaMemcpyAsync(S->kv_stage, kv_host, kv_stage_bytes(), cudaMemcpyDefault, s),
                                      "glm prefill: the KV stage", err))
                    return false;
                S->mark("kv_stage", s);
            }
        }
        uint16_t* const lat = kv_host != nullptr ? (uint16_t*) S->kv_stage : (uint16_t*) (state_ + dsa_lat_[(size_t) il]);

        // ---- the mixer, in sub-batches of S->sub tokens (the recurrences carry their state across)
        for (int t0 = 0; t0 < T; t0 += S->sub) {
            const int tn = std::min(S->sub, T - t0);
            const float* xs = S->x + (size_t) t0 * E;
            const uint16_t* x16 = S->x16 + (size_t) t0 * E;
            float* mixer = S->mixer + (size_t) t0 * E;
            if (Ly.recr) {
                Carve c{S->uni};
                KdaBufs B = carve_kda(c, (size_t) tn, g);
                hgemm_q(Ly.q, DI, E, x16, E, B.proj[0], DI, tn, 0.0f);
                hgemm_q(Ly.k, DI, E, x16, E, B.proj[1], DI, tn, 0.0f);
                hgemm_q(Ly.v, DI, E, x16, E, B.proj[2], DI, tn, 0.0f);
                sgemm_bf16(Ly.f_a, HD, E, xs, E, B.fa, HD, tn, 1.0f);
                sgemm_bf16(Ly.g_a, HD, E, xs, E, B.ga, HD, tn, 1.0f);
                sgemm_bf16(Ly.beta, g.n_head, E, xs, E, B.beta, g.n_head, tn, 1.0f);
                const float* pr[3] = {B.proj[0], B.proj[1], B.proj[2]};
                float* cv[3] = {B.conv[0], B.conv[1], B.conv[2]};
                float* cst = state_ + kda_conv_[(size_t) il];
                gb::kda_conv(pr, Ly.conv, cst, cv, tn, DI, g.d_conv, s);
                gb::kda_conv_state(pr, cst, tn, DI, g.d_conv, s);
                sgemm_bf16(Ly.f_b, DI, HD, B.fa, HD, B.g1, DI, tn, 1.0f);
                sgemm_bf16(Ly.g_b, DI, HD, B.ga, HD, B.g2, DI, tn, 1.0f);
                S->mark("kda_proj", s);
                gb::kda_rec(B.conv[0], B.conv[1], B.conv[2], B.g1, Ly.dt_bias, Ly.ssm_a, g.kda_lb, B.beta,
                            state_ + kda_S_[(size_t) il], B.g2, Ly.ssm_norm, g.norm_eps, g.n_head, tn, B.out16, s);
                S->mark("kda_rec", s);
                if (!skip_output) hgemm_q(Ly.out, E, DI, B.out16, DI, mixer, E, tn, 0.0f);
            } else {
                Carve c{S->uni};
                DsaBufs B = carve_dsa(c, (size_t) tn, g, S->max_pools);
                const int pt = (int) (p0 + t0);
                const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
                hgemm_q(Ly.q_a, g.q_lora, E, x16, E, B.qr_raw, g.q_lora, tn, 0.0f);
                hgemm_q(Ly.kv_a, g.kv_lora, E, x16, E, B.kv_raw, g.kv_lora, tn, 0.0f);
                sgemm_bf16(Ly.idx_k, g.idx_key, E, xs, E, B.ik_raw, g.idx_key, tn, 1.0f);
                sgemm_bf16(Ly.idx_gate, g.idx_key, E, xs, E, B.ig_raw, g.idx_key, tn, 1.0f);
                sgemm_bf16(Ly.idx_proj, g.idx_heads, E, xs, E, B.iw, g.idx_heads, tn, prescale);
                gb::DsaPrepArgs d;
                d.qr_raw = B.qr_raw;
                d.q_a_norm = Ly.q_a_norm;
                d.qr = B.qr;
                d.qr16 = B.qr16;
                d.q_lora = g.q_lora;
                d.kv_raw = B.kv_raw;
                d.kv_norm = Ly.kv_a_norm;
                d.lat = lat;
                if (kv_host != nullptr) {
                    d.lat_host = (uint16_t*) kv_host;
                    d.lat_slots = (uint16_t*) (state_ + dsa_lat_[(size_t) il]);
                    d.lat_table = kv_->map[(size_t) il].page_table;
                    d.lat_page = kv_->page;
                }
                d.lat_q8 = lat_q8_;
                d.kv_lora = g.kv_lora;
                d.ik_raw = B.ik_raw;
                d.k_norm_w = Ly.k_norm_w;
                d.k_norm_b = Ly.k_norm_b;
                d.ik_cache = state_ + dsa_ik_[(size_t) il];
                d.ig_raw = B.ig_raw;
                d.ig_cache = state_ + dsa_ig_[(size_t) il];
                d.idx_key = g.idx_key;
                d.ring = ik_ring_;
                d.p0 = pt;
                d.T = tn;
                d.eps = g.norm_eps;
                gb::dsa_prep(d, s);
                // the pools completed inside this sub-batch: pool pi ends at position (pi + 1) * kpool - 1
                const int kp = g.idx_kpool;
                const int pool_lo = (pt + kp) / kp - 1, pool_hi = (pt + tn) / kp - 1;
                if (pool_hi >= pool_lo)
                    gb::dsa_pool(state_ + dsa_ik_[(size_t) il], state_ + dsa_ig_[(size_t) il], Ly.ape,
                                 state_ + dsa_pool_[(size_t) il], g.idx_key, kp, pool_lo, pool_hi - pool_lo + 1, s,
                                 ik_ring_);
                hgemm_q(Ly.q_b, g.n_head * g.qk_nope, g.q_lora, B.qr16, g.q_lora, B.q, g.n_head * g.qk_nope, tn, 0.0f);
                sgemm_bf16(Ly.idx_q_b, g.idx_heads * g.idx_key, g.q_lora, B.qr, g.q_lora, B.iq, g.idx_heads * g.idx_key,
                           tn, 1.0f);
                S->mark("dsa_proj", s);
                const int max_vis = (pt + tn) / kp;
                if (max_vis > g.top_pools_max())
                    gb::dsa_score(B.iq, state_ + dsa_pool_[(size_t) il], B.iw, g.idx_key, g.idx_heads, pt, kp, tn,
                                  max_vis, B.score, B.score_ld, s);
                gb::dsa_select(B.score, B.score_ld, pt, kp, g.top_pools_max(), g.idx_select_tail, tn, g.n_sel_max(),
                               B.cells, B.n_sel, s);
                S->mark("dsa_index", s);
                // q_abs[t][h] = wk_b[h] (kv_lora x qk_nope) . q[t][h]
#if defined(STRATA_USE_HIP)
                // FP16 products; q is packed per head into B.attn's space and the context into B.q_abs's (both idle there)
                const bool f16 = mla_f16_on() && (int64_t) g.qk_nope <= 2 * (int64_t) g.v_head;
                const MlaAttnPath attn_path = mla_attn_path();
                const bool use_f16q = f16 && attn_path == MlaAttnPath::F16Q;
                const bool use_wmma2 = attn_path == MlaAttnPath::WMMA2;
                uint16_t* const hp_q = (uint16_t*) B.attn;
                uint16_t* const hp_c = (uint16_t*) B.q_abs;
                if (f16) {
                    bf16_to_f16<<<1024, 256, 0, s>>>(Ly.k_b, S->w16, (int64_t) g.n_head * g.kv_lora * g.qk_nope);
                    pack_heads_f16<<<2048, 256, 0, s>>>(B.q, hp_q, tn, g.n_head, g.qk_nope);
                    blas_ck(hipblasGemmStridedBatchedEx(S->blas, HIPBLAS_OP_T, HIPBLAS_OP_N, g.kv_lora, tn, g.qk_nope, &one,
                                                        S->w16, HIP_R_16F, g.qk_nope, (long long) g.kv_lora * g.qk_nope,
                                                        hp_q, HIP_R_16F, g.qk_nope, (long long) tn * g.qk_nope, &zero,
                                                        B.q_abs, use_f16q ? HIP_R_16F : HIP_R_32F, g.n_head * g.kv_lora,
                                                        g.kv_lora, g.n_head, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                            "q_abs f16");
                } else
#endif
                {
                gb::bf16_to_f32(Ly.k_b, S->w32, (int64_t) g.n_head * g.kv_lora * g.qk_nope, s);
                blas_ck(cublasSgemmStridedBatched(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, g.kv_lora, tn, g.qk_nope, &one,
                                                  S->w32, g.qk_nope, (long long) g.kv_lora * g.qk_nope, B.q,
                                                  g.n_head * g.qk_nope, g.qk_nope, &zero, B.q_abs, g.n_head * g.kv_lora,
                                                  g.kv_lora, g.n_head),
                        "q_abs");
                }
                S->mark("dsa_qabs", s);
#if defined(STRATA_USE_HIP)
                if (use_f16q)
                    gb::mla_attn_f16q((const uint16_t*) B.q_abs, lat, B.cells,
                                      B.n_sel, g.n_sel_max(), g.n_head, g.kv_lora, 1.0f / std::sqrt((float) g.qk_nope), tn,
                                      B.ctx, s, lat_q8_);
                else if (use_wmma2)
                    gb::mla_attn_wmma2(B.q_abs, lat, B.cells,
                                       B.n_sel, g.n_sel_max(), g.n_head, g.kv_lora, 1.0f / std::sqrt((float) g.qk_nope), tn,
                                       B.ctx, s, lat_q8_);
                else
#endif
                gb::mla_attn(B.q_abs, lat, B.cells, B.n_sel, g.n_sel_max(), g.n_head,
                             g.kv_lora, 1.0f / std::sqrt((float) g.qk_nope), tn, B.ctx, s, lat_q8_);
                S->mark("dsa_attn", s);
                // out[t][h] = wv_b[h] (v_head x kv_lora) . ctx[t][h]
#if defined(STRATA_USE_HIP)
                if (f16) {
                    bf16_to_f16<<<1024, 256, 0, s>>>(Ly.v_b, S->w16, (int64_t) g.n_head * g.v_head * g.kv_lora);
                    pack_heads_f16<<<2048, 256, 0, s>>>(B.ctx, hp_c, tn, g.n_head, g.kv_lora);
                    blas_ck(hipblasGemmStridedBatchedEx(S->blas, HIPBLAS_OP_T, HIPBLAS_OP_N, g.v_head, tn, g.kv_lora, &one,
                                                        S->w16, HIP_R_16F, g.kv_lora, (long long) g.v_head * g.kv_lora,
                                                        hp_c, HIP_R_16F, g.kv_lora, (long long) tn * g.kv_lora, &zero,
                                                        B.attn, HIP_R_32F, g.n_head * g.v_head, g.v_head, g.n_head,
                                                        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                            "mla out f16");
                } else
#endif
                {
                gb::bf16_to_f32(Ly.v_b, S->w32, (int64_t) g.n_head * g.v_head * g.kv_lora, s);
                blas_ck(cublasSgemmStridedBatched(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, g.v_head, tn, g.kv_lora, &one,
                                                  S->w32, g.kv_lora, (long long) g.v_head * g.kv_lora, B.ctx,
                                                  g.n_head * g.kv_lora, g.kv_lora, &zero, B.attn, g.n_head * g.v_head,
                                                  g.v_head, g.n_head),
                        "mla out");
                }
                gb::f32_to_f16(B.attn, B.attn16, (int64_t) tn * g.n_head * g.v_head, s);
                if (!skip_output)
                    hgemm_q(Ly.out, E, g.n_head * g.v_head, B.attn16, g.n_head * g.v_head, mixer, E, tn, 0.0f);
                if (t0 + tn == T) {
                    const std::string Ls = std::to_string(il);
                    dump_row("dsa_qr-" + Ls, B.qr, g.q_lora, tn - 1);
                    dump_row("dsa_q-" + Ls, B.q, g.n_head * g.qk_nope, tn - 1);
                    dump_row("dsa_iq-" + Ls, B.iq, g.idx_heads * g.idx_key, tn - 1);
#if defined(STRATA_USE_HIP)
                    if (!f16)   // with the FP16 products q_abs holds FP16 / the packed context
#endif
                    dump_row("pf_qabs-" + Ls, B.q_abs, g.n_head * g.kv_lora, tn - 1);
                    dump_row("pf_ctx-" + Ls, B.ctx, g.n_head * g.kv_lora, tn - 1);
                    dump_row("pf_attn-" + Ls, B.attn, g.n_head * g.v_head, tn - 1);
                }
            }
        }
        if (kv_host != nullptr && !kv_prefetch(kv_dsa_from(il + 1))) return false;   // (the stage is read: the next one's)

        S->mark(Ly.recr ? "kda" : "dsa", s);
        if (skip_output) continue;   // every KDA/DSA cache update above still ran; these output rows feed nothing
        dump_row("mixer-" + std::to_string(il), S->mixer, E);
        hc_read(S->mixer, Ly.hc_ffn_fn, Ly.hc_ffn_scale, Ly.hc_ffn_base, Ly.ffn_norm);
        S->mark("hc", s);
        dump_row("ffn_norm-" + std::to_string(il), S->x, E);
        // the CPU lane's x (this layer's FFN input) goes down now, under the shared expert and the routing - not
        // after the plan, where the host's experts waited for it (~40 ms of a 16K chunk's 260 MB)
        bool x_down = false;
        if (Ly.moe && cpu_lane(il)) {
            if (!cpu_bufs(0)) {
                err = "glm prefill: the CPU experts' host buffers did not allocate";
                return false;
            }
            cudaEventRecord(S->ev_x, s);
            cudaStreamWaitEvent(S->xs, S->ev_x, 0);
            cudaMemcpyAsync(S->x_h, S->x, (size_t) T * E * sizeof(float), cudaMemcpyDeviceToHost, S->xs);
            x_down = true;
        }

        if (!Ly.moe) {
            for (int t0 = 0; t0 < T; t0 += S->sub) {
                const int tn = std::min(S->sub, T - t0);
                Carve c{S->uni};
                DenseBufs B = carve_dense(c, (size_t) tn, g);
                const uint16_t* x16 = S->x16 + (size_t) t0 * E;
                hgemm_q(Ly.ffn_gate, g.n_ff_dense, E, x16, E, B.dg, g.n_ff_dense, tn, 0.0f);
                hgemm_q(Ly.ffn_up, g.n_ff_dense, E, x16, E, B.du, g.n_ff_dense, tn, 0.0f);
                gb::swiglu_f16(B.dg, B.du, B.dh16, (int64_t) tn * g.n_ff_dense, g.swiglu_shexp, s);
                hgemm_q(Ly.ffn_down, E, g.n_ff_dense, B.dh16, g.n_ff_dense, S->ffn + (size_t) t0 * E, E, tn, 0.0f);
            }
            S->mark("dense_ffn", s);
            dump_row("ffn_out-" + std::to_string(il), S->ffn, E);
            continue;
        }

        // ---- MoE: the shared expert (sub-batches, straight into ffn), the routes, the expert groups
        Carve c{S->uni};
        MoeBufs M = carve_moe(c, (size_t) T, (size_t) S->sh_sub(), g);
        const int FF = g.n_ff_exp * g.n_shared, NE = g.n_expert, nff = g.n_ff_exp;
        for (int t0 = 0, shs = S->sh_sub(); t0 < T; t0 += shs) {
            const int tn = std::min(shs, T - t0);
            const uint16_t* x16 = S->x16 + (size_t) t0 * E;
            hgemm_q(Ly.sh_gate, FF, E, x16, E, M.sh_g, FF, tn, 0.0f);
            hgemm_q(Ly.sh_up, FF, E, x16, E, M.sh_u, FF, tn, 0.0f);
            gb::swiglu_f16(M.sh_g, M.sh_u, M.sh16, (int64_t) tn * FF, g.swiglu_shexp, s);
            hgemm_q(Ly.sh_down, E, FF, M.sh16, FF, S->ffn + (size_t) t0 * E, E, tn, 0.0f);
        }
        S->mark("shexp", s);
        sgemm_bf16(Ly.router, NE, E, S->x, E, M.logits, NE, T, 1.0f);
        gb::route(M.logits, Ly.router_bias, NE, K, g.w_scale, g.norm_w != 0, T, M.ids, M.rw, s);
        gb::expert_count(M.ids, T * K, NE, M.counts, M.rank, s);
        gb::copy_i32(M.counts, S->h_counts, NE, s);   // (not a copy: the engine may be busy with x, x_down)
        S->mark("route", s);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "glm prompt route/count sync", err)) return false;
        int counted = 0;
        for (int e = 0; e < NE; ++e) counted += S->h_counts[e];
        if (counted != T * K) {
            err = "glm router: prompt layer " + std::to_string(il) + " invalid expert ID or non-finite score (" +
                  std::to_string(counted) + " of " + std::to_string(T * K) + " routes)";
            return false;
        }
        const auto tp = std::chrono::steady_clock::now();
        // (the split's balance measures both lanes from here: the compute stream is idle, the event is the plan's start)
        if (S->ev_c0 == nullptr) {
            cudaEventCreate(&S->ev_c0);
            cudaEventCreate(&S->ev_c1);
        }
        cudaEventRecord(S->ev_c0, s);
        if (S->trace) {
            auto& t = S->tr[(size_t) il];
            cudaEventRecord(t.plan, s);
            t.on = true;
            t.pre_left = 0;
            if (S->pre_layer == il) {
                const int np = (int) S->pre_e.size();
                t.pre_left = np;
                for (size_t k = 0; k * PrefillState::kPreEv < (size_t) np; ++k) {
                    if (cudaEventQuery(S->ev_pdone[k]) != cudaSuccess) break;
                    t.pre_left = std::max(0, np - (int) (k + 1) * PrefillState::kPreEv);
                }
            } else {
                t.pre_n = 0;
            }
        }

        // ---- the plan (host): resident experts in slot order, then the staged ones in groups
        auto& P = F->lp[(size_t) il];
        auto& RC = F->rc[(size_t) F->layer_rc[(size_t) il]];
        struct Staged {
            int e;
            const uint8_t* ram;   // the RAM tier slot (or, dev, its VRAM slot), or null: disk
            bool dev;
        };
        std::vector<Staged> staged;
        int nb = 0;   // h_bounds fill
        int rows_res = 0, max_res = 0;
        const int b_res = nb;
        // resident experts with at most light_rows rows go to the light kernels (gf::rows_experts): their rows come
        // after the MMQ product's, which sees them as empty
        static const int light_env = [] {
            const char* v = getenv("STRATA_GLM_PREFILL_LIGHT");
            return v ? std::atoi(v) : kLightRows;
        }();
        const auto lt_ok = [](int t) { return t == 16 || t == 18 || t == 19; };
        static const int light_layer = getenv("STRATA_GLM_PREFILL_LIGHT_LAYER") ? std::atoi(getenv("STRATA_GLM_PREFILL_LIGHT_LAYER")) : -1;
        const int light_rows = lt_ok(Ly.gu_type) && lt_ok(Ly.d_type) && (light_layer < 0 || il == light_layer) ? light_env : 0;
        int n_light = 0, rows_mmq = 0;
        int b_pre = -1, rows_pre = 0, max_pre = 0;   // the prestaged set: its bounds, rows and largest expert
        const int pre_n = S->pre_layer == il ? (int) S->pre_e.size() : 0;
        int* h_light = S->h_bounds + kLightOff;
        {
            std::lock_guard<std::mutex> lk(F->mu);
            S->h_bounds[nb++] = 0;
            for (int sl = 0; sl < P.n_main; ++sl) {
                int cnt = 0;
                if (P.st[(size_t) sl] == FastState::kResident && P.key[(size_t) sl] >= 0) {
                    const int e = P.key[(size_t) sl];
                    cnt = S->h_counts[e];
                    if (cnt > 0 && cnt <= light_rows && n_light < kLightMax) {
                        S->h_counts[e] = -cnt;   // claimed (rows assigned below)
                        h_light[3 * n_light] = sl;
                        h_light[3 * n_light + 1] = e;   // (the expert until its rows are known)
                        h_light[3 * n_light + 2] = cnt;
                        ++n_light;
                        cnt = 0;
                    } else if (cnt > 0) {
                        S->h_base[e] = rows_res;
                        S->h_counts[e] = -cnt;   // claimed
                    }
                }
                rows_res += std::max(0, cnt);
                max_res = std::max(max_res, cnt);
                S->h_bounds[nb++] = rows_res;
            }
            rows_mmq = rows_res;
            for (int i = 0; i < n_light; ++i) {
                const int e = h_light[3 * i + 1];
                S->h_base[e] = rows_res;
                h_light[3 * i + 1] = rows_res;
                rows_res += h_light[3 * i + 2];
            }
            // the prestaged experts this routing hit: their rows after the resident ones, computed from pbuf
            if (S->pre_layer == il) {
                b_pre = nb;
                S->h_bounds[nb++] = 0;
                for (int e : S->pre_e) {
                    const int cnt = std::max(0, S->h_counts[e]);
                    if (cnt > 0) {
                        S->h_base[e] = rows_res + rows_pre;
                        S->h_counts[e] = -cnt;   // claimed
                        rows_pre += cnt;
                        max_pre = std::max(max_pre, cnt);
                        ++S->pre_hit;
                    }
                    S->h_bounds[nb++] = rows_pre;
                }
            }
            for (int e = 0; e < NE; ++e) {
                if (S->h_counts[e] <= 0) continue;
                const int key = il * NE + e;
                const int rs = F->ram_of[(size_t) key];
                const int vs = F->slot_of[(size_t) key];
                const uint8_t* src = nullptr;
                // still in VRAM: a tail slot past what this prompt borrows (a short prompt borrows a little of
                // the tail) - a device copy, not the disk read it used to be
                const bool dev = vs >= P.n_main && vs < P.n && P.st[(size_t) vs] == FastState::kResident &&
                                 P.key[(size_t) vs] == e;
                if (dev) src = P.slot_ptr(vs);
                else if (rs >= 0 && RC.st[(size_t) rs] == FastState::kRHold) src = RC.base + (size_t) rs * RC.stride;
                staged.push_back(Staged{e, src, dev});
            }
            // the prompt's routing feeds the LFU counts the decode's tiers evict by
            uint64_t ev = 0;
            for (int e = 0; e < NE; ++e) {
                const int cnt = std::abs(S->h_counts[e]);
                if (cnt == 0) continue;
                uint32_t& cc = F->cnt[(size_t) il * NE + e];
                cc = (uint32_t) std::min<uint64_t>((uint64_t) cc + (uint64_t) cnt, 1u << 24);
                ev += (uint64_t) cnt;
                uint32_t& uu = F->usage[(size_t) il * NE + e];
                uu = (uint32_t) std::min<uint64_t>((uint64_t) uu + (uint64_t) cnt, 1u << 30);
                if (uu >= (1u << 30))
                    for (auto& u : F->usage) u >>= 1;
            }
            F->cnt_events += ev;
            while (F->cnt_events >= 32768) {
                F->cnt_events -= 32768;
                for (auto& cc : F->cnt) cc >>= 1;
            }
            auto& lc = S->last_cnt[(size_t) il];
            lc.resize((size_t) NE);
            for (int e = 0; e < NE; ++e) lc[(size_t) e] = std::abs(S->h_counts[e]);
        }
        // RAM-tier experts first (DMA), disk ones last (their reads overlap the earlier groups)
        std::stable_partition(staged.begin(), staged.end(), [](const Staged& x) { return x.ram != nullptr; });
        // THE CPU'S SHARE (a host whose CPU lane computes an expert faster than PCIe moves it - one GPU beside two
        // sockets: 0.15 vs 2.1 ms): the least routed RAM-tier experts are computed on the host, from the RAM tier, while
        // the copy stream stages the rest - taken in ascending row count while the host's estimated time (an expert's
        // weights + STRATA_GLM_PREFILL_CPU_ROW_MS a row, times the learned balance kappa) stays under the PCIe time of
        // what is left.  Their rows go last in the sorted order; their outputs are uploaded into OUTP before the
        // combine.  STRATA_GLM_PREFILL_CPU=0: off.
        std::vector<Staged> cpu_set;
        {
            static const char* row_env = getenv("STRATA_GLM_PREFILL_CPU_ROW_MS");
            // the planning cost of a row: the env's, else learned from the layers before (their host time less an
            // expert's weights each, over their rows; one 3090 beside 2x Xeon 6152: ~0.056 ms)
            const double row_ms = row_env != nullptr ? std::atof(row_env) : S->row_ms;
            if (cpu_lane(il)) {
                std::vector<size_t> ram_i;
                int n_pcie = 0;
                for (size_t i = 0; i < staged.size(); ++i) {
                    if (staged[i].dev) continue;
                    ++n_pcie;
                    if (staged[i].ram != nullptr) ram_i.push_back(i);
                }
                std::sort(ram_i.begin(), ram_i.end(), [&](size_t a, size_t b) {
                    return S->h_counts[staged[a].e] < S->h_counts[staged[b].e];
                });
                // (the streaming rate: the prestage copies', else the calibration's)
                const double p_ms = S->copy_ms > 0.0 ? S->copy_ms : F->cpu_ps_ms > 0.0 ? F->cpu_ps_ms : F->cpu_p_ms;
                // the prestage copies the copy stream still owes (a mixer shorter than the buffer's copies): the
                // staged groups queue behind them
                int pre_left = 0;
                if (S->pre_layer == il) {
                    const int np = (int) S->pre_e.size();
                    pre_left = np;
                    for (size_t k = 0; k * PrefillState::kPreEv < (size_t) np; ++k) {
                        if (cudaEventQuery(S->ev_pdone[k]) != cudaSuccess) break;
                        pre_left = std::max(0, np - (int) (k + 1) * PrefillState::kPreEv);
                    }
                }
                double t_cpu = 0.0, t_pcie = (double) (n_pcie + pre_left) * p_ms;
                std::vector<char> take(staged.size(), 0);
                const int cpu_cap = moe_window_rows((size_t) T, g);   // (the device rows their outputs go up into)
                int cpu_rows_taken = 0;
                for (size_t i : ram_i) {
                    // (an expert's fixed cost is about twice the lane's one-token expert: its weights, and the
                    // multi-row kernel's unpacking of every row)
                    const double c = S->kappa * (2.0 * F->cpu_c_ms + row_ms * S->h_counts[staged[i].e]);
                    if (t_cpu + c > t_pcie - p_ms) break;
                    if (cpu_rows_taken + S->h_counts[staged[i].e] > cpu_cap) break;
                    cpu_rows_taken += S->h_counts[staged[i].e];
                    take[i] = 1;
                    t_cpu += c;
                    t_pcie -= p_ms;
                }
                std::vector<Staged> rest;
                for (size_t i = 0; i < staged.size(); ++i) (take[i] ? cpu_set : rest).push_back(staged[i]);
                staged.swap(rest);
            }
        }
        int rows = rows_res + rows_pre;
        struct Group {
            int first, n, r0, nrows, max_rows, b_off;
        };
        std::vector<Group> groups;
        for (size_t i = 0; i < staged.size(); i += PrefillState::GE) {
            Group gr{(int) i, (int) std::min<size_t>(PrefillState::GE, staged.size() - i), rows, 0, 0, nb};
            S->h_bounds[nb++] = 0;
            for (int j = 0; j < gr.n; ++j) {
                const int e = staged[(size_t) gr.first + j].e;
                const int cnt = S->h_counts[e];
                S->h_base[e] = rows;
                rows += cnt;
                gr.nrows += cnt;
                gr.max_rows = std::max(gr.max_rows, cnt);
                S->h_bounds[nb++] = gr.nrows;
            }
            groups.push_back(gr);
        }
        const int cpu_r0 = rows;
        for (const auto& cx : cpu_set) {
            S->h_base[cx.e] = rows;
            rows += S->h_counts[cx.e];
        }
        const int cpu_nrows = rows - cpu_r0;
        if (rows != T * K) {
            err = "glm prefill: layer " + std::to_string(il) + " planned " + std::to_string(rows) + " of " +
                  std::to_string(T * K) + " routed rows";
            return false;
        }
        if (nb > kLightOff) {
            err = "glm prefill: the expert bounds overflowed";
            return false;
        }
        cudaMemcpyAsync(M.base, S->h_base, (size_t) NE * sizeof(int), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(M.bounds, S->h_bounds, (size_t) nb * sizeof(int), cudaMemcpyHostToDevice, s);
        if (n_light > 0)
            cudaMemcpyAsync(M.bounds + kLightOff, h_light, (size_t) 3 * n_light * sizeof(int), cudaMemcpyHostToDevice, s);
        gb::expert_scatter(M.ids, M.rank, M.base, T * K, K, M.row_tok, M.pos, s);
        if (cpu_nrows > 0) {
            if (!cpu_bufs(cpu_nrows)) {
                err = "glm prefill: the CPU experts' host buffers did not allocate";
                return false;
            }
            cudaEventRecord(S->ev_x, s);   // the scatter (and x) are done up to here
            cudaStreamWaitEvent(S->xs, S->ev_x, 0);
            if (!x_down) cudaMemcpyAsync(S->x_h, S->x, (size_t) T * E * sizeof(float), cudaMemcpyDeviceToHost, S->xs);
            cudaMemcpyAsync(S->rt_h, M.row_tok + cpu_r0, (size_t) cpu_nrows * sizeof(int), cudaMemcpyDeviceToHost, S->xs);
        }
        S->mark("plan", s);
        S->ms_plan += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();

        // ---- the OUTPUT WINDOW: OUTP row 0 is sorted row wb; rows [w_lo, w_hi) are computed and not yet added into ffn.
        //      A set's rows go in contiguous with what it holds, else the window is added into ffn first (moe_combine_add)
        //      and starts at the set; a set larger than the window runs in parts at expert boundaries
        const int W = moe_window_rows((size_t) T, g);
        int wb = 0, w_lo = 0, w_hi = 0;
        const auto flush = [&]() {
            if (w_hi > w_lo) gb::moe_combine_add(M.OUTP, wb, w_lo, w_hi, M.pos, M.rw, T, K, E, S->ffn, s);
            w_lo = w_hi = 0;
        };
        const auto place = [&](int a, int n) {   // -> the OUTP row of sorted row a
            if (!(w_hi > w_lo && a == w_hi && a + n <= wb + W)) {
                flush();
                wb = a;
                w_lo = a;
            }
            w_hi = a + n;
            return a - wb;
        };
        const size_t qblk = mmq::q8_block_bytes();
        // one expert set through gate/up, swiglu and down: its rows [r0 + hb[0], r0 + hb[n_exp]) of the sorted order
        // (hb: the set's bounds on the host, d_bounds the same on the device), in parts of at most W rows
        const auto run_set = [&](const uint8_t* wbase, int n_exp, size_t stride, const int* d_bounds, const int* hb,
                                 int r0) {
            for (int i0 = 0; i0 < n_exp;) {
                int i1 = i0 + 1;   // (one expert never has more rows than T <= W)
                while (i1 < n_exp && hb[i1 + 1] - hb[i0] <= W) ++i1;
                const int a = hb[i0], nrows = hb[i1] - a;
                int max_rows = 0;
                for (int i = i0; i < i1; ++i) max_rows = std::max(max_rows, hb[i + 1] - hb[i]);
                if (nrows > 0) {
                    const int o = place(r0 + a, nrows);
                    mmq::quantize(S->x, M.row_tok + r0 + a, M.Xq, Ly.gu_type, E, E, nrows, s);
                    // the part's bounds are the set's (from a): the activations and the destination are addressed a
                    // rows back so that bound a is the part's first row
                    float* GU = M.OUTP + (size_t) o * E;   // 2 * n_ff == n_embd: the set's own OUTP rows hold its gate/up
                    mmq::Product gu;
                    gu.w = wbase + (size_t) i0 * stride;
                    gu.type = Ly.gu_type;
                    gu.w_rows = 2 * nff;
                    gu.w_cols = E;
                    gu.expert_bytes = stride;
                    gu.n = i1 - i0;
                    gu.xq = (const uint8_t*) M.Xq - (size_t) a * qblk;
                    gu.bounds = d_bounds + i0;
                    gu.ids = S->iota;
                    gu.total_rows = nrows;
                    gu.max_rows = max_rows;
                    gu.dst = GU - (size_t) a * (2 * nff);
                    gu.ld_dst = 2 * nff;
                    S->mq->run(gu, s);
                    gb::swiglu_rows(GU, GU, nrows, nff, g.swiglu_exp, s, 2 * nff);   // (in place: no T x K x n_ff buffer)
                    mmq::quantize(GU, nullptr, M.Hq, Ly.d_type, nff, 2 * nff, nrows, s);
                    mmq::Product dn;
                    dn.w = wbase + (size_t) i0 * stride + Ly.down_off;
                    dn.type = Ly.d_type;
                    dn.w_rows = E;
                    dn.w_cols = nff;
                    dn.expert_bytes = stride;
                    dn.n = i1 - i0;
                    dn.xq = (const uint8_t*) M.Hq - (size_t) a * qblk;
                    dn.bounds = d_bounds + i0;
                    dn.ids = S->iota;
                    dn.total_rows = nrows;
                    dn.max_rows = max_rows;
                    dn.dst = M.OUTP + (size_t) (o - a) * E;
                    dn.ld_dst = E;
                    S->mq->run(dn, s);
                }
                i0 = i1;
            }
        };
        if (2 * nff != E) {
            err = "glm prefill: 2 * n_ff_exp != n_embd (the gate/up rows would not fit the set's output rows)";
            return false;
        }
        // the disk-only experts (staged last, ascending): their indices in the reader's order - inside the segment
        // predicted one layer ago, or appended now
        std::vector<int> dlist, gidx;   // staged indices of the disk experts; their reader indices
        for (size_t i = 0; i < staged.size(); ++i)
            if (staged[i].ram == nullptr) dlist.push_back((int) i);
        int seg_end = dq.n.load();
        if (pred_layer == il) {
            size_t p = 0;
            for (int k : dlist) {
                const int e = staged[(size_t) k].e;
                while (p < pred_e.size() && pred_e[p] != e) ++p;
                if (p == pred_e.size()) break;
                gidx.push_back(pred0 + (int) p);
            }
            seg_end = pred0 + (int) pred_e.size();
            if (gidx.size() != dlist.size()) {   // (cannot happen: the plan's disk set is within the prediction)
                dq.copied.store(seg_end, std::memory_order_release);   // the predicted slots are free again
                gidx.clear();
            }
        }
        if (gidx.size() != dlist.size()) {
            for (int k : dlist) gidx.push_back(dq.push(il, staged[(size_t) k].e));
            seg_end = dq.n.load();
        }
        pred_layer = -1;
        if (pred_t > 0 && T >= pred_t && il + 1 < l1_ && F->L[(size_t) il + 1].moe) {
            pred_e = disk_only(il + 1);
            pred0 = dq.n.load();
            for (int e : pred_e) dq.push(il + 1, e);
            pred_layer = il + 1;
        }
        // the resident experts: the lightly routed ones through the light kernels, the rest in one MMQ product over
        // the layer's whole pool partition (the light ones empty there), while the copy stream stages
        if (n_light > 0) {
            const int o = place(rows_mmq, rows_res - rows_mmq);   // (light rows: at most kLightMax * kLightRows <= W)
            if (!gf::rows_experts(Ly.gu_type, Ly.d_type, P.base, P.stride, Ly.down_off, M.bounds + kLightOff, n_light,
                                  M.row_tok, S->x, T, E, nff, g.swiglu_exp, rows_mmq, rows_res, M.Xq, M.Hq,
                                  M.OUTP + (size_t) (o - rows_mmq) * E, E, s)) {
                err = "glm prefill: the light expert kernels refused the layer's types";
                return false;
            }
        }
        S->mark("moe_light", s);
        // STRATA_GLM_PREFILL_LIGHT_CHECK=1 (debug): the light rows again through MMQ, compared row by row
        // (it reads the light rows where they were before the output window: only before any other set ran)
        static const bool light_check = getenv("STRATA_GLM_PREFILL_LIGHT_CHECK") != nullptr;
        if (light_check && n_light > 0 && wb == rows_mmq) {
            const int nl_rows = rows_res - rows_mmq;
            std::vector<float> A((size_t) nl_rows * E), B((size_t) nl_rows * E);
            cudaStreamSynchronize(s);
            cudaMemcpy(A.data(), M.OUTP, A.size() * sizeof(float), cudaMemcpyDeviceToHost);
            int* hb = S->h_bounds + 4096;
            int nbc = 0, max_l = 0, li = 0;
            hb[nbc++] = 0;
            for (int sl = 0; sl < P.n_main; ++sl) {
                int add = 0;
                if (li < n_light && h_light[3 * li] == sl) {
                    add = h_light[3 * li + 2];
                    max_l = std::max(max_l, add);
                    ++li;
                }
                hb[nbc] = hb[nbc - 1] + add;
                ++nbc;
            }
            cudaMemcpy(M.bounds + 4096, hb, (size_t) nbc * sizeof(int), cudaMemcpyHostToDevice);
            w_lo = w_hi = 0;   // (the light rows are re-run over the same window rows: not added twice)
            run_set(P.base, P.n_main, P.stride, M.bounds + 4096, hb, rows_mmq);
            cudaStreamSynchronize(s);
            cudaMemcpy(B.data(), M.OUTP, B.size() * sizeof(float), cudaMemcpyDeviceToHost);
            double num = 0, den = 0, worst = 0;
            for (int r = 0; r < nl_rows; ++r) {
                double rn = 0, rd = 0;
                for (int j = 0; j < E; ++j) {
                    const double dlt = (double) A[(size_t) r * E + j] - B[(size_t) r * E + j];
                    rn += dlt * dlt;
                    rd += (double) B[(size_t) r * E + j] * B[(size_t) r * E + j];
                }
                num += rn;
                den += rd;
                worst = std::max(worst, std::sqrt(rn / std::max(1e-30, rd)));
            }
            std::fprintf(stderr, "glm prefill light check layer %d: %d experts, %d rows, rel L2 %.3e, worst row %.3e\n", il,
                         n_light, nl_rows, std::sqrt(num / std::max(1e-30, den)), worst);
        }
        run_set(P.base, P.n_main, P.stride, M.bounds + b_res, S->h_bounds + b_res, 0);
        (void) max_res;
        S->rows_resident += (uint64_t) rows_res;
        S->mark("moe_resident", s);
        if (S->pre_layer == il) {
            if (rows_pre > 0) {
                cudaStreamWaitEvent(s, S->ev_pready, 0);
                run_set(S->pbuf, (int) S->pre_e.size(), P.stride, M.bounds + b_pre, S->h_bounds + b_pre, rows_res);
                S->pre_rows += (uint64_t) rows_pre;
            }
            cudaEventRecord(S->ev_pfree, s);   // (the next layer's prestage overwrites pbuf after this)
            S->pre_layer = -1;
            S->mark("moe_prestaged", s);
        }
        int dk = 0;   // the next disk expert (index into dlist / gidx)
        int n_staged_pcie = 0;   // (the split's balance: the copy stream's time when the host computes experts too)
        for (const auto& st : staged) n_staged_pcie += st.dev ? 0 : 1;
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            const Group& gr = groups[gi];
            const int b = (int) (gi % PrefillState::NG);
            uint8_t* gdst = S->gbuf + (size_t) b * PrefillState::GE * P.stride;
            cudaStreamWaitEvent(F->copy, S->ev_free[b], 0);
            const int dk0 = dk;
            for (int j = 0; j < gr.n; ++j) {
                const Staged& st = staged[(size_t) gr.first + j];
                const uint8_t* src = st.ram;
                if (src == nullptr) {
                    const int gi2 = gidx[(size_t) dk];
                    // every index below this one was copied (its event recorded) or predicted and never routed:
                    // released, so the reader can get to this one however far the prediction's gaps put it
                    if (dq.copied.load() < gi2) dq.copied.store(gi2, std::memory_order_release);
                    if (dq.landed.load(std::memory_order_acquire) <= gi2) {
                        const auto td = std::chrono::steady_clock::now();
                        while (dq.landed.load(std::memory_order_acquire) <= gi2) glmfast::cpu_relax();
                        S->ms_disk += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
                    }
                    src = S->gpin + (size_t) (gi2 % KL) * S->gstride;
                    ++dk;
                }
                cudaMemcpyAsync(gdst + (size_t) j * P.stride, src, Ly.blob,
                                st.dev ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, F->copy);
                if (st.ram == nullptr) {   // its landing slot is free once this copy ran
                    cudaEventRecord(S->ev_land[gidx[(size_t) dk - 1] % KL], F->copy);
                    dq.copied.store(gidx[(size_t) dk - 1] + 1, std::memory_order_release);
                }
                S->staged_ram += st.ram != nullptr && !st.dev;
                S->staged_vram += st.dev;
            }
            (void) dk0;
            cudaEventRecord(S->ev_ready[b], F->copy);
            cudaStreamWaitEvent(s, S->ev_ready[b], 0);
            run_set(gdst, gr.n, P.stride, M.bounds + gr.b_off, S->h_bounds + gr.b_off, gr.r0);
            cudaEventRecord(S->ev_free[b], s);
            S->rows_staged += (uint64_t) gr.nrows;
        }
        dq.copied.store(std::max(dq.copied.load(), seg_end), std::memory_order_release);   // predicted, never routed
        S->staged_disk += dlist.size();
        if (cpu_nrows > 0) cudaEventRecord(S->ev_c1, F->copy);
        if (S->trace) {
            auto& t = S->tr[(size_t) il];
            cudaEventRecord(t.c1, F->copy);
            t.grp = n_staged_pcie;
            t.ncpu = (int) cpu_set.size();
            t.crows = cpu_nrows;
            t.dl = t.dcpu = 0.0;
        }
        // the next MoE layer's prestage: its copies run once this layer's are done - during the next mixer.  With
        // CPU experts it is queued after their outputs' upload, on the same stream: an upload on a stream of its own
        // waited out the whole prestage (one copy engine a direction), and the combine with it
        if (cpu_nrows == 0) prestage(next_moe(il + 1));
        if (cpu_nrows > 0) {
            namespace kc = strata::kernels::cpu;
            const auto tc = std::chrono::steady_clock::now();
            const auto& nf = F->cpu_fmt[(size_t) il];
            constexpr size_t kA = kc::kNativeActBytes, kH = kc::kNativeHBytes;
            constexpr int kGu = 32, kDn = 64;
            if (S->c_act.size() < (size_t) cpu_nrows * kA) S->c_act.resize((size_t) cpu_nrows * kA);
            if (S->c_hq.size() < (size_t) cpu_nrows * kH) S->c_hq.resize((size_t) cpu_nrows * kH);
            if (S->c_ff.size() < (size_t) cpu_nrows * nff) S->c_ff.resize((size_t) cpu_nrows * nff);
            if (!glmfast::cuda_ok(cudaStreamSynchronize(S->xs), "glm prompt CPU rows", err)) return false;
            const int ncx = (int) cpu_set.size();
            std::vector<int> r_of((size_t) ncx);   // each expert's first row in the CPU segment
            for (int j = 0; j < ncx; ++j) r_of[(size_t) j] = S->h_base[cpu_set[(size_t) j].e] - cpu_r0;
            F->cpu_pool->run(cpu_nrows, [&](int r) {
                kc::native_quant_act(nf, S->x_h + (size_t) S->rt_h[r] * E, S->c_act.data() + (size_t) r * kA);
            });
            // experts with enough rows take the multi-row AVX-512 kernel (a weight row unpacked once for all of them,
            // ~2x ggml's per-pair dots from 8 rows on - kq_avx512.hpp); their activations packed once each
            static const bool kq_on = [] {
                const char* v = getenv("STRATA_GLM_KQ");
                return kc::kq_avx512_ok() && (v == nullptr || std::atoi(v) != 0);
            }();
            const auto kq_use = [&](int type, int act_type, int cnt) {
                return kq_on && kc::kq_type_ok(type) && act_type == 15 /* Q8_K */ && cnt >= (type == 11 ? 8 : 12);
            };
            std::vector<size_t> off_a((size_t) ncx, SIZE_MAX), off_h((size_t) ncx, SIZE_MAX);
            size_t pack_a = 0, pack_h = 0;
            for (int j = 0; j < ncx; ++j) {
                const int cnt = S->h_counts[cpu_set[(size_t) j].e];
                if (kq_use(nf.gu_type, nf.gu_act, cnt)) {
                    off_a[(size_t) j] = pack_a;
                    pack_a += (kc::kq_act_bytes(cnt, E) + 63) & ~(size_t) 63;
                }
                if (kq_use(nf.d_type, nf.d_act, cnt)) {
                    off_h[(size_t) j] = pack_h;
                    pack_h += (kc::kq_act_bytes(cnt, nff) + 63) & ~(size_t) 63;
                }
            }
            if (S->c_pack.size() < pack_a + pack_h + 64) S->c_pack.resize(pack_a + pack_h + 64);
            uint8_t* pk = (uint8_t*) (((uintptr_t) S->c_pack.data() + 63) & ~(uintptr_t) 63);
            uint8_t* pk_h = pk + pack_a;
            const auto pack = [&](bool gu) {
                F->cpu_pool->run(ncx, [&](int j) {
                    const size_t off = gu ? off_a[(size_t) j] : off_h[(size_t) j];
                    if (off == SIZE_MAX) return;
                    const int cnt = S->h_counts[cpu_set[(size_t) j].e], b = r_of[(size_t) j];
                    thread_local std::vector<const void*> a;
                    a.resize((size_t) cnt);
                    for (int t = 0; t < cnt; ++t)
                        a[(size_t) t] = gu ? (const void*) (S->c_act.data() + (size_t) (b + t) * kA)
                                           : (const void*) (S->c_hq.data() + (size_t) (b + t) * kH);
                    kc::kq_pack_act(a.data(), cnt, gu ? E : nff, (gu ? pk : pk_h) + off);
                });
            };
            if (pack_a > 0) pack(true);
            F->cpu_pool->run(ncx * (nff / kGu), [&](int job) {
                const int j = job % ncx, r0 = (job / ncx) * kGu, b = r_of[(size_t) j];
                const int cnt = S->h_counts[cpu_set[(size_t) j].e];
                const uint8_t* blob = cpu_set[(size_t) j].ram;
                if (off_a[(size_t) j] != SIZE_MAX) {
                    // gate and up rows [r0, r0 + kGu) for every row of the expert, then the clamped swiglu
                    thread_local std::vector<float> gu_t;
                    thread_local std::vector<float*> go, uo;
                    gu_t.resize((size_t) 2 * cnt * kGu);
                    go.resize((size_t) cnt);
                    uo.resize((size_t) cnt);
                    for (int t = 0; t < cnt; ++t) {
                        go[(size_t) t] = gu_t.data() + (size_t) t * kGu;
                        uo[(size_t) t] = gu_t.data() + (size_t) (cnt + t) * kGu;
                    }
                    const void* pa = pk + off_a[(size_t) j];
                    kc::kq_rows_packed(nf.gu_type, blob + (size_t) r0 * nf.gu_row, nf.gu_row, E, pa, cnt, go.data(), 0,
                                       kGu);
                    kc::kq_rows_packed(nf.gu_type, blob + Ly.gu_bytes + (size_t) r0 * nf.gu_row, nf.gu_row, E, pa,
                                       cnt, uo.data(), 0, kGu);
                    const float lim = g.swiglu_exp;
                    for (int t = 0; t < cnt; ++t) {
                        float* ff = S->c_ff.data() + (size_t) (b + t) * nff + r0;
                        for (int r = 0; r < kGu; ++r) {
                            const float gg = std::fmin(go[(size_t) t][r], lim);
                            const float uu = std::fmin(std::fmax(uo[(size_t) t][r], -lim), lim);
                            ff[r] = (gg / (1.f + std::exp(-gg))) * uu;
                        }
                    }
                    return;
                }
                thread_local std::vector<const void*> a;
                thread_local std::vector<float*> o;
                a.resize((size_t) cnt);
                o.resize((size_t) cnt);
                for (int t = 0; t < cnt; ++t) {
                    a[(size_t) t] = S->c_act.data() + (size_t) (b + t) * kA;
                    o[(size_t) t] = S->c_ff.data() + (size_t) (b + t) * nff;
                }
                kc::native_gu_rows_split(nf, blob, blob + Ly.gu_bytes, a.data(), cnt, o.data(), r0, r0 + kGu,
                                         g.swiglu_exp);
            });
            F->cpu_pool->run(cpu_nrows, [&](int r) {
                kc::native_quant_h(nf, S->c_ff.data() + (size_t) r * nff, S->c_hq.data() + (size_t) r * kH);
            });
            if (pack_h > 0) pack(false);
            // the down rows in two parts: the first part's outputs (about two thirds of the rows) go up on the copy
            // stream while the second part computes
            int j_split = ncx;
            for (int j = 0; j < ncx; ++j)
                if (r_of[(size_t) j] >= cpu_nrows * 2 / 3) {
                    j_split = j;
                    break;
                }
            const auto down = [&](int j0, int j1) {
              const int nj = j1 - j0;
              if (nj <= 0) return;
              F->cpu_pool->run(nj * (E / kDn), [&](int job) {
                const int j = j0 + job % nj, r0 = (job / nj) * kDn, b = r_of[(size_t) j];
                const int cnt = S->h_counts[cpu_set[(size_t) j].e];
                thread_local std::vector<const void*> h;
                thread_local std::vector<float*> o;
                h.resize((size_t) cnt);
                o.resize((size_t) cnt);
                const uint8_t* down = cpu_set[(size_t) j].ram + Ly.down_off;
                if (off_h[(size_t) j] != SIZE_MAX) {
                    for (int t = 0; t < cnt; ++t) o[(size_t) t] = S->out_h + (size_t) (b + t) * E + r0;
                    kc::kq_rows_packed(nf.d_type, down + (size_t) r0 * nf.d_row, nf.d_row, nff, pk_h + off_h[(size_t) j],
                                       cnt, o.data(), 0, kDn);
                    return;
                }
                for (int t = 0; t < cnt; ++t) {
                    h[(size_t) t] = S->c_hq.data() + (size_t) (b + t) * kH;
                    o[(size_t) t] = S->out_h + (size_t) (b + t) * E;
                }
                kc::native_down_rows_split(nf, down, h.data(), cnt, o.data(), r0, r0 + kDn);
              });
            };
            const auto up = [&](int ra, int rb) {
                if (rb > ra)
                    cudaMemcpyAsync(M.OUTPc + (size_t) ra * E, S->out_h + (size_t) ra * E,
                                    (size_t) (rb - ra) * E * sizeof(float), cudaMemcpyHostToDevice, F->copy);
            };
            const int r_split = j_split < ncx ? r_of[(size_t) j_split] : cpu_nrows;
            down(0, j_split);
            up(0, r_split);
            down(j_split, ncx);
            up(r_split, cpu_nrows);
            cudaEventRecord(S->ev_out, F->copy);
            cudaStreamWaitEvent(s, S->ev_out, 0);   // (the combine reads them; the next layer's x overwrites x after)
            prestage(next_moe(il + 1));
            const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
            S->cpu_experts += (uint64_t) ncx;
            S->cpu_rows += (uint64_t) cpu_nrows;
            S->ms_cpu += dt;
            if (S->trace) {
                S->tr[(size_t) il].dcpu = dt;
                S->tr[(size_t) il].dl =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();
            }
            if (cpu_nrows >= 64)   // (a layer with a few rows says little about the per-row cost)
                S->row_ms = 0.5 * S->row_ms +
                            0.5 * std::max(0.005, (dt - 2.0 * (double) ncx * F->cpu_c_ms) / (double) cpu_nrows);
            // the balance: when the copy stream's staged copies ended against when the host's experts did, both from
            // the plan's start (prestage copies it still owed and x's way down included)
            if (n_staged_pcie > 0) {
                const double dl = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();
                cudaEventSynchronize(S->ev_c1);
                float pc = 0.0f;
                if (cudaEventElapsedTime(&pc, S->ev_c0, S->ev_c1) == cudaSuccess && pc > 1.0f && dl > 1.0)
                    S->kappa = std::min(4.0, std::max(0.25, S->kappa * std::min(1.25, std::max(0.8, std::sqrt(dl / pc)))));
            }
            // the prestage count: the copy stream idle before the plan (the copies landed early) -> as many more as
            // that time moves; every other expert on the CPU and it still done before the copies -> fewer by the
            // difference; else (the plan balanced the two) the same
            // (not the chunk's first MoE layer: its copies had the dense layers before it too)
            if (pre_n > 0 && il != next_moe(l0_)) {
                const double dl = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();
                cudaEventSynchronize(S->ev_c1);
                float landed = 0.0f, pc = 0.0f, pt = 0.0f;
                if (cudaEventElapsedTime(&landed, S->ev_c0, S->ev_pready) == cudaSuccess &&
                    cudaEventElapsedTime(&pc, S->ev_c0, S->ev_c1) == cudaSuccess &&
                    cudaEventElapsedTime(&pt, S->ev_pstart, S->ev_pready) == cudaSuccess) {
                    const double cm = pt / pre_n;   // (the copies ran back to back from ev_pstart)
                    if (cm > 0.2 && cm < 20.0) S->copy_ms = S->copy_ms > 0.0 ? 0.8 * S->copy_ms + 0.2 * cm : cm;
                    const double pm = S->copy_ms > 0.0 ? S->copy_ms : 1.6;
                    double nn = pre_n;
                    if (landed < -3.0f) nn = pre_n + 0.8 * (-landed) / pm;
                    else if (n_staged_pcie == 0 && dl + 5.0 < pc) nn = pre_n - 0.8 * (pc - dl) / pm;
                    double& r = S->pre_rate[Ly.recr ? 0 : 1];
                    r = r > 0.0 ? 0.5 * r + 0.5 * nn / T : nn / T;
                }
            }
        }
        S->mark("moe_staged", s);
        dump_row("pf_shexp-" + std::to_string(il), S->ffn, E);
        flush();
        if (cpu_nrows > 0) gb::moe_combine_add(M.OUTPc, cpu_r0, cpu_r0, rows, M.pos, M.rw, T, K, E, S->ffn, s);
        S->mark("combine", s);
        if (S->trace) cudaEventRecord(S->tr[(size_t) il].end, s);
        dump_row("ffn_out-" + std::to_string(il), S->ffn, E);
    }
    // the last layer's write half: R = post x ffn + comb . R
    if (!tail_skip) {
        gb::hc_update(S->ffn, S->R, S->post, S->comb, nullptr, T, E, s);
        S->mark("hc", s);
    }
    // ---- the NextN block's caches for these positions (the half that carries it): position p reads h_p and the
    //      embedding of the token at p + 1, so only the cache-writing half of the block runs - eh_proj, the attention
    //      norm, kv_a and the indexer's key/gate - no queries, no attention, no FFN
    if (mtp_il_ >= 0 && next_ids != nullptr) {
        int Tv = 0;   // rows whose next token is known (all but possibly the prompt's last)
        while (Tv < T && next_ids[Tv] >= 0) ++Tv;
        if (Tv > 0) {
            const auto& Ly = F->L[(size_t) mtp_il_];
            const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
            if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
                err = "glm prefill: no embedding dequantizer for the draft block";
                return false;
            }
            const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, E);
            if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "glm prompt mtp embedding", err)) return false;
            for (int t = 0; t < Tv; ++t)
                tt->to_float(pack_emb_src_ + (size_t) next_ids[t] * row_b, S->emb_h + (size_t) t * E, E);
            Carve c{S->uni};
            float* h = c.take<float>((size_t) Tv * E);
            float* emb = c.take<float>((size_t) Tv * E);
            uint16_t* cat16 = c.take<uint16_t>((size_t) Tv * 2 * E);
            float* hid = c.take<float>((size_t) Tv * E);
            float* kv = c.take<float>((size_t) Tv * g.kv_lora);
            float* ik = c.take<float>((size_t) Tv * g.idx_key);
            float* ig = c.take<float>((size_t) Tv * g.idx_key);
            cudaMemcpyAsync(emb, S->emb_h, (size_t) Tv * E * sizeof(float), cudaMemcpyHostToDevice, s);
            gb::head_rows(S->R, w_.at("output_norm.weight"), g.norm_eps, Tv, E, h, s);
            gb::mtp_in_rows(emb, h, Ly.enorm, Ly.hnorm, g.norm_eps, Tv, E, cat16, s);
            hgemm_q(Ly.eh, E, 2 * E, cat16, 2 * E, hid, E, Tv, 0.0f);
            gb::rms_rows(hid, Ly.attn_norm, g.norm_eps, Tv, E, S->x, S->x16, s);
            hgemm_q(Ly.kv_a, g.kv_lora, E, S->x16, E, kv, g.kv_lora, Tv, 0.0f);
            sgemm_bf16(Ly.idx_k, g.idx_key, E, S->x, E, ik, g.idx_key, Tv, 1.0f);
            sgemm_bf16(Ly.idx_gate, g.idx_key, E, S->x, E, ig, g.idx_key, Tv, 1.0f);
            gb::DsaPrepArgs d;
            d.kv_raw = kv;
            d.kv_norm = Ly.kv_a_norm;
            d.lat = (uint16_t*) (state_ + dsa_lat_[(size_t) mtp_il_]);
            if (kv_ != nullptr && kv_->host[(size_t) mtp_il_] != nullptr) {   // KV streaming: no attention here, no stage
                d.lat = nullptr;
                d.lat_host = (uint16_t*) kv_->host[(size_t) mtp_il_];
                d.lat_slots = (uint16_t*) (state_ + dsa_lat_[(size_t) mtp_il_]);
                d.lat_table = kv_->map[(size_t) mtp_il_].page_table;
                d.lat_page = kv_->page;
            }
            d.lat_q8 = lat_q8_;
            d.kv_lora = g.kv_lora;
            d.ik_raw = ik;
            d.k_norm_w = Ly.k_norm_w;
            d.k_norm_b = Ly.k_norm_b;
            d.ik_cache = state_ + dsa_ik_[(size_t) mtp_il_];
            d.ig_raw = ig;
            d.ig_cache = state_ + dsa_ig_[(size_t) mtp_il_];
            d.idx_key = g.idx_key;
            d.ring = ik_ring_;
            d.eps = g.norm_eps;
            // in pieces the indexer's key / gate ring holds with the open pool before them (a one-GPU chunk is up to
            // 32K positions, the ring ~8K): each piece's rows are pooled before the next one overwrites the ring
            const int kp = g.idx_kpool;
            const int piece = std::max(kp, ik_ring_ - 64);
            for (int t0 = 0; t0 < Tv; t0 += piece) {
                const int tn = std::min(piece, Tv - t0);
                d.kv_raw = kv + (size_t) t0 * g.kv_lora;
                d.ik_raw = ik + (size_t) t0 * g.idx_key;
                d.ig_raw = ig + (size_t) t0 * g.idx_key;
                d.p0 = (int) p0 + t0;
                d.T = tn;
                gb::dsa_prep(d, s);
                const int pool_lo = (int) ((p0 + t0 + kp) / kp - 1), pool_hi = (int) ((p0 + t0 + tn) / kp - 1);
                if (pool_hi >= pool_lo)
                    gb::dsa_pool(state_ + dsa_ik_[(size_t) mtp_il_], state_ + dsa_ig_[(size_t) mtp_il_], Ly.ape,
                                 state_ + dsa_pool_[(size_t) mtp_il_], g.idx_key, kp, pool_lo, pool_hi - pool_lo + 1, s,
                                 ik_ring_);
            }
            S->mark("mtp_cache", s);
        }
    }
    S->collect();
    if (S->trace) {
        cudaStreamSynchronize(s);
        cudaStreamSynchronize(F->copy);
        const auto el = [](cudaEvent_t a, cudaEvent_t b) {
            float v = 0.0f;
            return cudaEventElapsedTime(&v, a, b) == cudaSuccess ? (double) v : -1.0;
        };
        std::fprintf(stderr, "glm prefill trace CUDA%d, a chunk of %d tokens (ms; plan = the routing known, the lanes "
                             "from there):\n", dev_, T);
        for (int il = l0_; il < l1_; ++il) {
            const auto& t = S->tr[(size_t) il];
            if (!t.on) continue;
            std::fprintf(stderr, "  layer %2d: start->plan %6.1f | prestage %3d, %3d left at plan, landed plan%+7.1f | copies "
                                 "%3d end plan+%6.1f | cpu %3d experts %5d rows end plan+%6.1f (compute %6.1f) | layer "
                                 "end plan+%6.1f\n",
                         il, el(t.start, t.plan), t.pre_n, t.pre_left, t.pre_n > 0 ? el(t.plan, t.pre) : 0.0, t.grp,
                         el(t.plan, t.c1), t.ncpu, t.crows, t.dl, t.dcpu, el(t.plan, t.end));
        }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm prefill: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Glm5Model::dump_state(const std::string& path) {
    int half = 0;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get(), ++half) {
        cudaSetDevice(m->dev_);
        cudaDeviceSynchronize();
        std::vector<float> buf(m->state_bytes_ / sizeof(float));
        if (cudaMemcpy(buf.data(), m->state_, buf.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        const std::string base = path + "." + std::to_string(half);
        FILE* f = std::fopen((base + ".bin").c_str(), "wb");
        if (f == nullptr) return false;
        std::fwrite(buf.data(), sizeof(float), buf.size(), f);
        std::fclose(f);
        FILE* t = std::fopen((base + ".txt").c_str(), "w");
        if (t == nullptr) return false;
        std::fprintf(t, "max_ctx %lld pos %lld\n", (long long) m->max_ctx_, (long long) m->pos_);
        for (int il = m->l0_; il < m->l1_; ++il)
            std::fprintf(t, "%d %lld %lld %lld %lld %lld %lld\n", il, (long long) m->kda_S_[(size_t) il],
                         (long long) m->kda_conv_[(size_t) il], (long long) m->dsa_lat_[(size_t) il],
                         (long long) m->dsa_ik_[(size_t) il], (long long) m->dsa_ig_[(size_t) il],
                         (long long) m->dsa_pool_[(size_t) il]);
        std::fclose(t);
    }
    cudaSetDevice(dev_);
    return true;
}

// ---------------------------------------------------------------- the prompt
bool Glm5Model::prefill(const std::vector<int32_t>& tokens, std::string& err, int32_t next_token) {
    err.clear();
    prefill_next_ = next_token;
    if (fast_ == nullptr || tokens.empty()) return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())   // (a tail lent to the vision encoder: the
        if (m->pf_ == nullptr || m->fast_ == nullptr || m->vis_lent_) return false;   // token path, until it is back)
    // a handful of tokens costs less through the token path (the batched path streams every expert they touch)
    int64_t min_n = 32;
    if (const char* mn = getenv("STRATA_GLM_PREFILL_MIN")) min_n = std::max<int64_t>(1, std::atoll(mn));
    if ((int64_t) tokens.size() < min_n) return false;
    // the chunk for this prompt: no bigger than it needs (rounded up to 64), so it borrows only that much; a prompt
    // longer than the largest chunk is cut into EQUAL chunks (the parts of a split pipeline chunks)
    int Tmax = pf_->T;
    for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) Tmax = std::min(Tmax, m->pf_->T);
    const int64_t nn = (int64_t) tokens.size(), nchunks = (nn + Tmax - 1) / Tmax;
    // one GPU has no pipeline to balance: full chunks and the remainder last - every chunk stages nearly every expert,
    // so 8449 tokens as 8192 + 257 cost about one chunk's experts, as two equal 4225s about two (240 -> ~300 tok/s)
    const int Trun = split_next_ == nullptr ? (int) std::min<int64_t>(Tmax, (nn + 63) / 64 * 64)
                                            : (int) std::min<int64_t>(Tmax, ((nn + nchunks - 1) / nchunks + 63) / 64 * 64);
    bool ok = true;
    progress_at("the prompt: lending the expert pool's tail");   // (the serve watchdog's stages, issue #40)
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        m->prefill_carve(Trun);
        if (!m->prefill_lend(err)) { ok = false; break; }
    }
    // STRATA_GLM_STALL_TEST=<s> (a test of the serve watchdog): every batched prompt stops here for s seconds
    static const int stall_test = [] {
        const char* v = getenv("STRATA_GLM_STALL_TEST");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    if (ok && stall_test > 0) {
        progress_at("the prompt: STRATA_GLM_STALL_TEST's pause");
        std::this_thread::sleep_for(std::chrono::seconds(stall_test));
    }
    if (ok) ok = prefill_run(tokens, err);
    progress_at("the prompt: returning the expert pool's tail");
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        std::string return_err;
        if (!m->prefill_return(return_err)) {
            if (ok) err = return_err;
            ok = false;
        }
    }
    cudaSetDevice(dev_);
    return ok;
}

bool Glm5Model::prefill_run(const std::vector<int32_t>& tokens, std::string& err) {
    const Glm5Geometry& g = g_;
    const int64_t n = (int64_t) tokens.size();
    if (pos_ + n > max_ctx_) {
        err = "glm prefill: the prompt (" + std::to_string(pos_ + n) + " positions) exceeds the context (" +
              std::to_string(max_ctx_) + ")";
        return false;
    }
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm prefill: no embedding dequantizer";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int E = g.n_embd;
    const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, E);
    int Tc = pf_->T_bound;
    for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) Tc = std::min(Tc, m->pf_->T_bound);
    const int64_t pos_start = pos_;
    const int nc = (int) ((n + Tc - 1) / Tc);
    const auto chunk_len = [&](int c) { return (int) std::min<int64_t>(Tc, n - (int64_t) c * Tc); };
    // the token after every position of chunk c (the draft block's caches read it): the prompt's own next token,
    // the caller's for the last position (-1: unknown - that position is filled by the first draft of the decode)
    const auto next_of = [&](int c, std::vector<int32_t>& out) {
        const int T = chunk_len(c);
        out.resize((size_t) T);
        for (int t = 0; t < T; ++t) {
            const int64_t j = (int64_t) c * Tc + t + 1;
            out[(size_t) t] = j < n ? tokens[(size_t) j] : prefill_next_;
        }
    };
    // chunk c's embedding rows into this half's R (host-dequantized from the shard mapping)
    const auto embed = [&](int c) -> bool {
        PrefillState* S = pf_;
        const int T = chunk_len(c);
        const int64_t i = (int64_t) c * Tc;
        if (!glmfast::cuda_ok(cudaStreamSynchronize(fast_->cs), "glm prompt embedding", err)) return false;
        for (int t = 0; t < T; ++t) {
            tt->to_float(pack_emb_src_ + (size_t) tokens[(size_t) (i + t)] * row_b, S->emb_h + (size_t) t * E, E);
            if (const float* img = image_row(pos_start + i + t))   // an image's row in place of its <|image|> token
                std::memcpy(S->emb_h + (size_t) t * E, img, (size_t) E * sizeof(float));
        }
        // the embedding lands in x (free until the first layer's gates write it) and fans out to the 4 streams
        cudaMemcpyAsync(S->x, S->emb_h, (size_t) T * E * sizeof(float), cudaMemcpyHostToDevice, fast_->cs);
        gb::embed_rows(S->x, S->R, T, E, fast_->cs);
        return true;
    };
    std::vector<Glm5Model*> parts;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) parts.push_back(m);
    const int np = (int) parts.size();
    const bool pipelined = np > 1 && nc > 1 && getenv("STRATA_GLM_PREFILL_SERIAL") == nullptr;
    bool cancelled = false;
    if (!pipelined) {
        // one half after the other, chunk by chunk
        for (int c = 0; c < nc; ++c) {
            const int T = chunk_len(c);
            const int64_t p0 = pos_start + (int64_t) c * Tc;
            cudaSetDevice(dev_);
            if (!embed(c)) return false;
            std::vector<int32_t> nx0;
            if (mtp_il_ >= 0) next_of(c, nx0);   // a single device carries the draft block itself
            if (!prefill_half(p0, T, err, nx0.empty() ? nullptr : nx0.data())) return false;
            Glm5Model* prev = this;
            for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) {
                PrefillState* SP = prev->pf_;
                const size_t hop = (size_t) T * 4 * E * sizeof(float);
                cudaSetDevice(prev->dev_);
                cudaMemcpyAsync(SP->hop_h, SP->R, hop, cudaMemcpyDeviceToHost, prev->fast_->cs);
                cudaEventRecord(SP->ev_hop, prev->fast_->cs);
                cudaSetDevice(m->dev_);
                cudaStreamWaitEvent(m->fast_->cs, SP->ev_hop, 0);
                cudaMemcpyAsync(m->pf_->R, SP->hop_h, hop, cudaMemcpyHostToDevice, m->fast_->cs);
                m->pos_ = p0;
                std::vector<int32_t> nx;
                if (m->mtp_il_ >= 0) next_of(c, nx);
                if (!m->prefill_half(p0, T, err, nx.empty() ? nullptr : nx.data())) return false;
                prev = m;
            }
            pos_ = p0 + T;
            for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) m->pos_ = pos_;
            ++pf_->chunks;
            if (prefill_progress && c + 1 < nc && !prefill_progress(pos_ - pos_start, n)) {
                cancelled = true;
                break;
            }
        }
    } else {
        // THE PIPELINE: part k reads chunk c while part k - 1 reads chunk c + 1 - the residual rows cross each
        // boundary through the earlier part's two pinned host slots (slot c & 1 is free again once the next part has
        // copied chunk c - 2 in).  This thread runs the first part, one thread each the later ones.
        std::mutex mu;
        std::condition_variable cv;
        // produced[k]: chunks part k has put in its slots; consumed[k]: chunks part k + 1 has copied out of them
        std::vector<int> produced((size_t) np, 0), consumed((size_t) np, 0);
        bool stop = false;
        std::string stage_err;
        const size_t slot = (size_t) Tc * 4 * E;
        const auto fail = [&](std::string e) {
            std::lock_guard<std::mutex> lk(mu);
            if (stage_err.empty()) stage_err = std::move(e);
            stop = true;
            cv.notify_all();
        };
        // part k has read chunk c: its rows into its slot for part k + 1
        const auto hand_on = [&](int k, int c, int T) -> bool {
            Glm5Model* m = parts[(size_t) k];
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return consumed[(size_t) k] >= c - 1 || stop; });
                if (stop) return false;
            }
            cudaMemcpyAsync(m->pf_->hop_h + (size_t) (c & 1) * slot, m->pf_->R, (size_t) T * 4 * E * sizeof(float),
                            cudaMemcpyDeviceToHost, m->fast_->cs);
            std::string e;
            if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm prompt hop out", e)) {
                fail(e);
                return false;
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                produced[(size_t) k] = c + 1;
            }
            cv.notify_all();
            return true;
        };
        std::vector<std::thread> later;
        for (int k = 1; k < np; ++k)
            later.emplace_back([&, k] {
                Glm5Model* m = parts[(size_t) k];
                const Glm5Model* up = parts[(size_t) k - 1];
                cudaSetDevice(m->dev_);
                for (int c = 0; c < nc; ++c) {
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return produced[(size_t) k - 1] > c || stop; });
                        if (stop) return;
                    }
                    const int T = chunk_len(c);
                    const int64_t p0 = pos_start + (int64_t) c * Tc;
                    cudaMemcpyAsync(m->pf_->R, up->pf_->hop_h + (size_t) (c & 1) * slot,
                                    (size_t) T * 4 * E * sizeof(float), cudaMemcpyHostToDevice, m->fast_->cs);
                    std::string e2;
                    if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm prompt hop in", e2)) {
                        fail(e2);
                        return;
                    }
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        consumed[(size_t) k - 1] = c + 1;
                    }
                    cv.notify_all();
                    m->pos_ = p0;
                    std::vector<int32_t> nx;
                    if (m->mtp_il_ >= 0) next_of(c, nx);
                    if (!m->prefill_half(p0, T, e2, nx.empty() ? nullptr : nx.data())) {
                        fail(e2.empty() ? "glm prefill: part " + std::to_string(k) + " failed" : e2);
                        return;
                    }
                    if (k + 1 < np) {
                        if (!hand_on(k, c, T)) return;
                        continue;
                    }
                    if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm prompt pipeline", e2)) {
                        fail(e2);
                        return;
                    }
                    if (prefill_progress && c + 1 < nc &&
                        !prefill_progress(std::min<int64_t>(n, (int64_t) (c + 1) * Tc), n)) {
                        fail("cancelled");
                        return;
                    }
                }
            });
        cudaSetDevice(dev_);
        for (int c = 0; c < nc; ++c) {
            {
                std::lock_guard<std::mutex> lk(mu);
                if (stop) break;
            }
            const int T = chunk_len(c);
            const int64_t p0 = pos_start + (int64_t) c * Tc;
            if (!embed(c)) { fail(err); break; }
            std::string e1;
            if (!prefill_half(p0, T, e1)) {
                fail(e1.empty() ? "glm prefill: the head half failed" : e1);
                break;
            }
            if (!hand_on(0, c, T)) break;
            ++pf_->chunks;
        }
        for (auto& t : later) t.join();
        if (stop) {
            if (stage_err == "cancelled") {
                cancelled = true;
            } else {
                err = stage_err;
                return false;
            }
        }
        pos_ = pos_start + n;
        for (Glm5Model* m : parts) m->pos_ = pos_;
    }
    if (cancelled) {
        err = "cancelled";
        return false;
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm prefill compute", err) ||
            !glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->copy), "glm prefill copies", err)) {
            cudaSetDevice(dev_);
            return false;
        }
    }
    if (gb::launch_errors() > 0 || strata::kernels::glmf::launch_errors() > 0) {
        err = "glm prefill: " + std::to_string(gb::launch_errors() + strata::kernels::glmf::launch_errors()) +
              " kernel launch(es) failed (see stderr)";
        return false;
    }
    cudaSetDevice(dev_);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    pf_->ms += ms;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->pf_->tokens += n;
    static const bool verbose = getenv("STRATA_GLM_PREFILL_VERBOSE") != nullptr;
    if (verbose) {
        uint64_t sr = 0, sv = 0, sd = 0, rr = 0, rs = 0, ce = 0, cr = 0, pi = 0, ph = 0, pr = 0;
        double plan = 0, mc = 0;
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            sr += m->pf_->staged_ram;
            sv += m->pf_->staged_vram;
            ce += m->pf_->cpu_experts;
            cr += m->pf_->cpu_rows;
            mc += m->pf_->ms_cpu;
            sd += m->pf_->staged_disk;
            rr += m->pf_->rows_resident;
            rs += m->pf_->rows_staged;
            plan += m->pf_->ms_plan;
            pi += m->pf_->pre_issued;
            ph += m->pf_->pre_hit;
            pr += m->pf_->pre_rows;
        }
        std::fprintf(stderr, "glm prefill: %lld tokens in %.1f ms (%.2f ms/token, %.0f tok/s) | staged experts: %llu vram, %llu "
                             "ram, %llu disk, %llu on the CPU (%llu rows, %.0f ms; split kappa %.2f, %.3f ms a row), %llu prestaged "
                             "(%llu routed, %llu rows) | rows resident %.1f%% | plan %.1f ms (cumulative)\n",
                     (long long) n, ms, ms / (double) n, 1000.0 * (double) n / ms, (unsigned long long) sv,
                     (unsigned long long) sr, (unsigned long long) sd, (unsigned long long) ce, (unsigned long long) cr, mc, pf_->kappa,
                     pf_->row_ms, (unsigned long long) pi, (unsigned long long) ph, (unsigned long long) pr,
                     100.0 * (double) rr / (double) std::max<uint64_t>(1, rr + rs), plan);
    }
    return true;
}

}  // namespace strata::core
