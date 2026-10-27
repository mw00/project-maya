// src/mm3/mm3_model.cu - maya-mm3's model (mm3_model.hpp): loading, placement over the GPUs, and the forward.
//
// A layer's forward, as transformers' MiniMaxM3VLDecoderLayer:
//   x = rms(h) (1 + w);  q, k, v = x Wq, x Wk, x Wv (EXL3);  per-head RMSNorm, RoPE on 64 of 128 dims;  K, V -> cache
//   the indexer: iq = x Wiq, ik = x Wik (F32 GEMMs over the FP16 weights), normed and rotated, ik -> its cache; each
//   query's block scores (the max over a block's keys), its own block and the 15 best kept, per KV group
//   attention over the kept blocks -> x Wo;  h += it
//   x = rms(h) (1 + w');  layers 0-2: the dense swigluoai MLP;  else the router (F32 logits, sigmoid, + bias, top 4,
//   renormalised), the routed experts (EXL3 sets: the resident ones in one pass, the staged ones in passes of the
//   staging slots), the shared expert, h += 2.0 * routed + shared
#include "mm3_model.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <tuple>

#include "mm3_cands.hpp"
#include "mm3_cpu.hpp"
#include "mm3_env.hpp"
#include "mm3_dspark.hpp"
#include "mm3_exl3.hpp"
#include "mm3_geo.hpp"
#include "mm3_gguf.cuh"
#include "mm3_gguf_host.hpp"
#include "mm3_kernels.cuh"
#include "mm3_os.hpp"
#include "mm3_tp.hpp"
#include "mm3_vision.hpp"

namespace mm3 {
namespace x3 = strata::mm3::exl3;

#define CK(call)                                                                       \
    do {                                                                               \
        const cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                       \
            err = std::string(#call) + ": " + cudaGetErrorString(e_);                  \
            return false;                                                              \
        }                                                                              \
    } while (0)
#define CB(call)                                                                       \
    do {                                                                               \
        const cublasStatus_t b_ = (call);                                              \
        if (b_ != CUBLAS_STATUS_SUCCESS) {                                             \
            err = std::string(#call) + ": cublas status " + std::to_string((int) b_);  \
            return false;                                                              \
        }                                                                              \
    } while (0)

namespace {

constexpr size_t kAlign = 256;
constexpr int kHeadRows = 32;   // the output head's rows a GEMM (its F32 logits buffer)
constexpr int kWin = 16;        // a window's tokens at most (the token path's kernels: mv's columns, win_experts)
size_t up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const std::string P = "language_model.model.";

struct Src {   // an expert's nine pieces in the mapped files: (w1, w3, w2) x (trellis, suh, svh) - a GGUF's: the three
               // matrices' rows alone (pieces 0, 3, 6; the others empty)
    const uint8_t* p[9];
    size_t n[9];
};

struct Layer {
    int dev = 0;
    bool moe = false;
    float *in1 = nullptr, *post1 = nullptr, *qn1 = nullptr, *kn1 = nullptr, *iqn1 = nullptr, *ikn1 = nullptr;
    x3::Mat q, k, v, o, mg, mu, md, sg, su, sd;
    float *wiq = nullptr, *wik = nullptr, *gate = nullptr, *bias = nullptr;
    // (a GGUF's Q8_0 indexer projections: kept as stored instead of wiq / wik - the token kernels read the blocks, the
    //  prompt path decodes them into Dev::wtmp for its F32 GEMM: the F32 values either way, a quarter of the bytes)
    const uint8_t *wiq8 = nullptr, *wik8 = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr, *ic = nullptr;
    k::KvQ kvq;   // (--kv-cache: K's / V's quantized caches - a format other than BF16 instead of kc / vc)
    k::KvHost kvh;   // (--kv-streaming: the caches above a ring, every position in host memory too)
    x3::ExpertLayout lay;
    GgFmt gf;   // (a GGUF's experts: their format on the CPU lane)
    size_t blob = 0;
    std::vector<Src> src;
    std::vector<int> res_slot;     // expert -> its slot in the resident region (-1: none)
    std::vector<int> res_ids;      // slot -> expert
    uint8_t* res_base = nullptr;   // the resident region: res_ids.size() blobs, blob bytes apart
    std::vector<const uint8_t*> host_blob;   // expert -> its assembled blob in the pinned pool (null: none)
    // host-free (every routed expert of the layer in VRAM): the decode's plan made on the device from exp_tab (expert ->
    // its blob's device address) - no wait for the routing, no round trip through the host
    bool hf = false;
    unsigned long long* exp_tab = nullptr;
};

struct Dev {
    int id = 0;
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    std::vector<int> layers;
    size_t free0 = 0;
    // staging
    uint8_t* stg = nullptr;
    size_t slot_bytes = 0;
    int n_slots = 0;
    std::vector<long long> slot_owner;   // layer * 1024 + expert, -1
    std::vector<long long> slot_used;    // the forward counter at its last use
    // the promotions (Options::promote): a slot being filled on the copy stream (pending: its key, its event) is no
    // one's until the copy is done
    cudaStream_t cs = nullptr;
    std::vector<cudaEvent_t> slot_ev;
    std::vector<long long> slot_pend;    // the key a pending slot will hold (-1: none)
    std::vector<long long> slot_iss;     // the forward at whose end its copy began
    int n_pending = 0;
    struct Promo {
        int slot;
        const uint8_t* src;
        size_t bytes;
    };
    std::vector<Promo> promo_q;          // the promotions a forward chose, copied at its end (its own copies done)
    uint8_t* pin = nullptr;              // a pinned blob for the copies
    unsigned long long* plan_h = nullptr;   // pinned: the token's plan (8 pointers, then 8 weights)
    unsigned long long* plan_d = nullptr;
    float* xh_h = nullptr;               // pinned: the token's normed input (the CPU lane's x), then its part + flag
    // pinned staging of a forward's inputs and the head's logits: the copies are the stream's own DMA commands, in its
    // order (a copy from or to pageable memory is not: on Windows + AMD, whose queued work starts only at a host wait,
    // such a copy may run before the kernels queued ahead of it - stale logits)
    float* emb_h = nullptr;              // [chunk][dim] the embeddings (the first layer's GPU)
    int* pos_h = nullptr;                // [chunk] the positions
    int* top_h = nullptr;                // [chunk][k] a prompt chunk's routing, read back
    float* lg_h = nullptr;               // [kHeadRows][vocab] the head's logits (the head's GPU)
    float* cpu_d = nullptr;              // the CPU lane's part [dim] and its flag (an int after it)
    // a window (2 .. kWin tokens: a DSpark verify): the readback (routes, weights, normed inputs), the plan for
    // win_experts (pinned, then on the device), the CPU lane's parts and flags in mapped memory with the flag the
    // device waits on, and the experts' scratch
    uint8_t* win_rb = nullptr;           // pinned: top [kWin][8] int, w [kWin][8] float, x [kWin][dim] float
    uint8_t* win_plan_h = nullptr;       // pinned: plan_ptr [kWin][8], plan_w [kWin][8], grp_ptr [kWin 8], grp_start
    uint8_t* win_plan_d = nullptr;       //   [kWin 8 + 1], grp_ent [kWin 8], n_groups
    float* win_cpu = nullptr;            // mapped: the CPU parts [kWin][dim], then the flags [kWin] int, the wait word
    unsigned win_wait = 0;
    float *win_h = nullptr, *win_part = nullptr;
    float* attp = nullptr;               // a token's / a window's attention splits' partials (k::attn_part_floats)
    float* wtmp = nullptr;               // (Q8_0 indexer projections) the prompt path's F32 copy of one of them
    // scratch (chunk rows)
    float *h = nullptr, *xn = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *iq = nullptr, *ik = nullptr;
    float *bs = nullptr, *att = nullptr, *ao = nullptr, *g = nullptr, *u = nullptr, *logit = nullptr, *w = nullptr;
    float *y = nullptr, *sh = nullptr, *inv = nullptr, *out = nullptr, *lg = nullptr;   // (lg: the head's rows)
    uint16_t *xn16 = nullptr, *att16 = nullptr, *h16 = nullptr;
    int *pos = nullptr, *kept = nullptr, *top = nullptr, *rowidx = nullptr, *invrow = nullptr, *bounds = nullptr;
    size_t nblk = 0;
    unsigned* prof_d = nullptr;   // (host-free layers) their routings counted on the device, [layer][expert]
};

size_t g_up = 0;   // (the load's upload_mat / upload_f32 bytes so far: the Monitor's dense weights)

// a GGUF's matrix (pack_gguf.py: <pre>.weight, [n][k] in a ggml block type) onto the device as it is stored
x3::Mat upload_gg(const Tensor& t, const Pack& pk, const std::string& pre, std::vector<void*>& owned, std::string& err) {
    x3::Mat m;
    const int ty = gg::type_of(t.dtype);
    if (t.shape.size() != 2 || !gg::dense_type(ty)) {
        err = pre + ": a " + t.dtype + " tensor the GGUF kernels do not take as a matrix";
        return m;
    }
    m.fmt = gg::kFmtGG;
    m.gg_type = ty;
    m.n = (int) t.shape[0];
    m.k = (int) t.shape[1];
    const size_t rb = gg::row_bytes(ty, m.k);
    if (rb == 0 || t.bytes != rb * (size_t) m.n || m.k % 256 != 0) {
        err = pre + ": " + std::to_string(t.bytes) + " bytes for [" + std::to_string(m.n) + "][" + std::to_string(m.k) +
              "] " + t.dtype;
        return m;
    }
    m.ld = (int) rb;
    void* d = nullptr;
    if (cudaMalloc(&d, t.bytes) != cudaSuccess) {
        err = pre + ": out of device memory";
        return m;
    }
    owned.push_back(d);
    g_up += t.bytes;
    cudaMemcpy(d, pk.data(t), t.bytes, cudaMemcpyHostToDevice);
    m.raw = d;
    return m;
}

x3::Mat upload_mat(const Pack& pk, const std::string& pre, std::vector<void*>& owned, std::string& err) {
    x3::Mat m;
    if (const Tensor* w = pk.get(pre + ".weight"); w != nullptr && gg::type_of(w->dtype) > 0)
        return upload_gg(*w, pk, pre, owned, err);
    const Tensor* tr = pk.get(pre + ".trellis");
    const Tensor* su = pk.get(pre + ".suh");
    const Tensor* sv = pk.get(pre + ".svh");
    if (!tr || !su || !sv || !pk.get(pre + ".mul1") || tr->shape.size() != 3) {
        err = pre + ": not an EXL3 (mul1) matrix";
        return m;
    }
    void *dt = nullptr, *ds = nullptr, *dv = nullptr;
    if (cudaMalloc(&dt, tr->bytes) != cudaSuccess || cudaMalloc(&ds, su->bytes) != cudaSuccess ||
        cudaMalloc(&dv, sv->bytes) != cudaSuccess) {
        err = pre + ": out of device memory";
        return m;
    }
    owned.insert(owned.end(), {dt, ds, dv});
    g_up += tr->bytes + su->bytes + sv->bytes;
    cudaMemcpy(dt, pk.data(*tr), tr->bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, pk.data(*su), su->bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, pk.data(*sv), sv->bytes, cudaMemcpyHostToDevice);
    m.trellis = (const uint32_t*) dt;
    m.suh = (const uint16_t*) ds;
    m.svh = (const uint16_t*) dv;
    m.k = (int) (su->bytes / 2);
    m.n = (int) (sv->bytes / 2);
    m.k2 = (int) tr->shape[2] / 8;
    m.cb = 2;
    if (!x3::supported(m)) err = pre + ": shape / bits not covered by the EXL3 kernels";
    return m;
}

// a tensor's stored bytes on the device as they are (a GGUF's Q8_0 indexer projections: k::rows_dot_q8's blocks)
const uint8_t* upload_raw(const Pack& pk, const std::string& name, std::vector<void*>& owned, std::string& err) {
    const Tensor* t = pk.get(name);
    if (!t) {
        err = name + ": missing";
        return nullptr;
    }
    void* dp = nullptr;
    if (cudaMalloc(&dp, t->bytes) != cudaSuccess) {
        err = name + ": out of device memory";
        return nullptr;
    }
    owned.push_back(dp);
    g_up += t->bytes;
    cudaMemcpy(dp, pk.data(*t), t->bytes, cudaMemcpyHostToDevice);
    return (const uint8_t*) dp;
}

// a float vector on the device: the stored F32 / F16 / BF16 tensor (+ add to every value); a GGUF's Q8_0 (the
// indexer's projections) dequantized exactly - ggml's value, an F16 scale times an int8
float* upload_f32(const Pack& pk, const std::string& name, float add, std::vector<void*>& owned, size_t& n,
                  std::string& err) {
    const Tensor* t = pk.get(name);
    if (!t) {
        err = name + ": missing";
        return nullptr;
    }
    const bool q8 = t->dtype == "Q8_0";
    const size_t es = t->dtype == "F32" ? 4 : 2;
    n = q8 ? t->bytes / 34 * 32 : t->bytes / es;
    std::vector<float> h(n);
    const uint8_t* d = pk.data(*t);
    for (size_t i = 0; i < n; ++i) {
        float v;
        if (q8) {
            const uint8_t* b = d + i / 32 * 34;
            v = gg::host_f16((uint16_t) (b[0] | b[1] << 8)) * (float) (int8_t) b[2 + i % 32];
        } else if (t->dtype == "F32") {
            std::memcpy(&v, d + 4 * i, 4);
        } else if (t->dtype == "BF16") {
            uint16_t b;
            std::memcpy(&b, d + 2 * i, 2);
            const uint32_t u = (uint32_t) b << 16;
            std::memcpy(&v, &u, 4);
        } else if (t->dtype == "F16") {
            uint16_t b;
            std::memcpy(&b, d + 2 * i, 2);
            // FP16 -> F32
            const uint32_t s = (uint32_t) (b & 0x8000) << 16, e = (b >> 10) & 0x1f, m = b & 0x3ff;
            uint32_t u;
            if (e == 0) {
                if (m == 0) {
                    u = s;
                } else {   // subnormal
                    float f = (float) m / 1024.0f / 16384.0f;
                    std::memcpy(&u, &f, 4);
                    u |= s;
                }
            } else if (e == 31) {
                u = s | 0x7f800000u | (m << 13);
            } else {
                u = s | ((e + 112) << 23) | (m << 13);
            }
            std::memcpy(&v, &u, 4);
        } else {
            err = name + ": dtype " + t->dtype;
            return nullptr;
        }
        h[i] = v + add;
    }
    void* dp = nullptr;
    if (cudaMalloc(&dp, n * 4) != cudaSuccess) {
        err = name + ": out of device memory";
        return nullptr;
    }
    owned.push_back(dp);
    g_up += n * 4;
    cudaMemcpy(dp, h.data(), n * 4, cudaMemcpyHostToDevice);
    return (float*) dp;
}

// the prefill calibration's key of this PC: the GPUs (name, architecture, memory), the GPU runtime and driver - a
// calibration measured elsewhere (another GPU, a new driver) is not taken
std::string tune_key_of(const std::vector<int>& gpus) {
    std::string k = "mm3-prefill-1";
    for (int id : gpus) {
        cudaDeviceProp pr{};
        if (cudaGetDeviceProperties(&pr, id) != cudaSuccess) continue;
#if defined(STRATA_USE_HIP)
        std::string arch = pr.gcnArchName;
        if (arch.find(':') != std::string::npos) arch.resize(arch.find(':'));
#else
        const std::string arch = "sm_" + std::to_string(pr.major) + std::to_string(pr.minor);
#endif
        std::string name = pr.name;
        for (char& c : name)
            if (c == '\t' || c == '\n' || c == '|') c = ' ';
        k += "|" + name + " " + arch + " " + std::to_string((pr.totalGlobalMem + (1u << 29)) >> 30) + "GB";
    }
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    return k + "|rt" + std::to_string(rt) + "|drv" + std::to_string(drv);
}

}  // namespace

struct Model::Impl {
    Options o;
    Pack pk;
    Geo g;
    Stats st;
    bool gguf = false;   // (the pack's format: a GGUF's - tools/mm3/pack_gguf.py - else EXL3)
    // the RMSNorm weights' Gemma offset: the checkpoint stores w (the engine multiplies by 1 + w); llama.cpp's
    // converter writes w + 1 into a GGUF already - added once either way
    float norm_add() const { return gguf ? 0.0f : 1.0f; }
    std::vector<Layer> L;
    std::vector<Dev> D;
    std::vector<void*> owned;   // (cudaFree at exit, by device)
    std::vector<int> owned_dev;
    x3::Mat head;
    float* norm1 = nullptr;
    int head_dev = 0;
    // a token's and a verify window's attention: each query's kept blocks in this many runs on blocks of their own,
    // combined in run order (k::sparse_attn's splits - one block a KV group walking all of them left the GPU idle: a
    // query's sums depend on the splits alone, so a window's rows keep the token path's bits).  MAYA_MM3_ATTN_SPLIT=1:
    // one run (the decode as before)
    int attn_splits = 16;
    std::vector<float> inv_freq;
    std::vector<float*> inv_dev;
    long long fwd = 0;
    int cur_pos = 0;                 // the last position of the chunk running
    std::vector<std::string> info;
    std::vector<std::string> vram;   // the Monitor's VRAM lines (serve/gpu_map.py), measured at the load's end
    // dalloc's bytes by what they hold, a device id's (the dense weights: upload_mat's / upload_f32's tally)
    enum { kAW, kAKV, kAAct, kAPool, kADraft, kAN };
    int acat = kAAct;
    std::map<int, std::array<size_t, kAN>> acct;
    std::map<int, size_t> acct_kvh;
    uint8_t* kv_host = nullptr;      // KV streaming: every layer's whole K / V cache (pinned, mapped)
    std::vector<uint32_t> prof;      // routings [layer][expert]: the loaded profile's plus this run's
    std::vector<uint32_t> cpu_seen;  // (MM3_MISS_STATS: the CPU lane's computations [layer][expert] this run)
    // the promotions' scores: an expert not in VRAM, each use (on the CPU lane, or from a slot) adds 1, decayed by
    // kDecay a forward; at kPromote it is copied into the slot whose expert scores least
    std::vector<float> use_score;
    std::vector<long long> use_last;
    static constexpr float kDecay = 0.95f, kPromote = 1.8f;
    static constexpr int kMaxPending = 4;
    uint8_t* pool = nullptr;         // the pinned pool
    std::unique_ptr<CpuLane> lane;
    std::unique_ptr<Draft> draftm;   // the DSpark draft (on the head's GPU)
    int last_done = 0;               // the last forward's tokens in the cache
    int chunk_run = 0;               // (the prefill calibration: a forward's chunk, at most o.chunk - 0: o.chunk)
    int run_chunk() const { return chunk_run > 0 ? std::min(chunk_run, o.chunk) : o.chunk; }
    std::string tkey, tfile;         // (the prefill calibration's key of this PC and its file)
    std::unique_ptr<Vision> vis;     // the vision tower (Options::vision)
    std::vector<Model::Image> imgs;  // the pictures the image tokens of the next forwards take their rows from
    float* taps = nullptr;           // its context's input: [taps][chunk][dim] F32 (the head's GPU)
    std::vector<int> tap_of;         // layer -> its tap (-1: none)

    // ---- MAYA_MM3_PROF=1 (one GPU): each forward's GPU time by part, from events on the stream - the time between two
    // marks goes to the second's part (a host wait before it included); printed at the end by kind of forward
    enum { kPAttnIn, kPIxProj, kPIndex, kPIxSel, kPAttn, kPOGemm, kPOut, kPDense, kPRouter, kPRoute, kPSync, kPExperts,
           kPHead, kPDraft, kPN };
    struct Prof {
        bool on = false;
        bool sync = false;        // (MAYA_MM3_PROF=2: the host waits at each mark and times it - AMD's events may be
                                  //  stamped as their packet is dispatched, ahead of the kernels before them)
        std::vector<double> ht;
        std::vector<cudaEvent_t> ev;
        std::vector<int> part;
        size_t used = 0;
        double ms[3][kPN] = {};   // [token, window, prompt]
        long fwd[3] = {}, tok[3] = {};
        double wall[3] = {};
    } pf;
    void pmark(Dev& d, int part) {
        if (!pf.on) return;
        if (pf.sync) {
            cudaStreamSynchronize(d.s);
            if (pf.used == pf.ht.size()) {
                pf.ht.push_back(0.0);
                pf.part.push_back(0);
            }
            pf.ht[pf.used] = now();
            pf.part[pf.used++] = part;
            return;
        }
        if (pf.used == pf.ev.size()) {
            cudaEvent_t e = nullptr;
            if (cudaEventCreate(&e) != cudaSuccess) return;
            pf.ev.push_back(e);
            pf.part.push_back(0);
        }
        cudaEventRecord(pf.ev[pf.used], d.s);
        pf.part[pf.used++] = part;
    }
    void pcollect(int kind, int T, double wall_s) {
        if (!pf.on) return;
        if (pf.sync)
            for (size_t i = 1; i < pf.used; ++i) pf.ms[kind][pf.part[i]] += 1e3 * (pf.ht[i] - pf.ht[i - 1]);
        else if (pf.used > 1 && cudaEventSynchronize(pf.ev[pf.used - 1]) == cudaSuccess)
            for (size_t i = 1; i < pf.used; ++i) {
                float ms = 0.0f;
                if (cudaEventElapsedTime(&ms, pf.ev[i - 1], pf.ev[i]) == cudaSuccess) pf.ms[kind][pf.part[i]] += ms;
            }
        pf.used = 0;
        ++pf.fwd[kind];
        pf.tok[kind] += T;
        pf.wall[kind] += wall_s;
    }
    void preport() const {
        if (!pf.on) return;
        static const char* names[kPN] = {"attn-in", "ix-proj", "indexer", "ix-select", "attention", "o-gemm",
                                         "attn-out", "dense-ffn", "router", "router+shared", "route-sync", "experts",
                                         "head", "draft-ctx"};
        static const char* kinds[3] = {"token", "window", "prompt"};
        for (int k = 0; k < 3; ++k) {
            if (pf.fwd[k] == 0) continue;
            double tot = 0;
            for (int i = 0; i < kPN; ++i) tot += pf.ms[k][i];
            std::fprintf(stderr, "MM3 PROF %s: %ld forwards, %ld tokens, %.2f ms a forward (wall %.2f ms):", kinds[k],
                         pf.fwd[k], pf.tok[k], tot / pf.fwd[k], 1e3 * pf.wall[k] / pf.fwd[k]);
            for (int i = 0; i < kPN; ++i)
                if (pf.ms[k][i] > 0) std::fprintf(stderr, " %s %.2f", names[i], pf.ms[k][i] / pf.fwd[k]);
            std::fprintf(stderr, "\n");
        }
    }

    ~Impl() {
        preport();
        graphs_clear();
        for (cudaEvent_t e : pf.ev) cudaEventDestroy(e);
        if (::mm3::env("MM3_MISS_STATS") && !cpu_seen.empty()) {
            long tot = 0, distinct = 0, twice = 0;
            for (uint32_t c : cpu_seen) {
                tot += c;
                distinct += c > 0;
                twice += c > 1 ? c - 1 : 0;
            }
            std::fprintf(stderr, "CPU lane: %ld expert computations, %ld distinct experts, %ld repeats (%.0f%%)\n", tot,
                         distinct, twice, tot ? 100.0 * twice / tot : 0.0);
        }
        for (size_t i = 0; i < owned.size(); ++i) {
            cudaSetDevice(owned_dev[i]);
            cudaFree(owned[i]);
        }
        if (pool) cudaFreeHost(pool);
        if (kv_host) cudaFreeHost(kv_host);
        for (auto& d : D) {
            cudaSetDevice(d.id);
            if (d.cs) cudaStreamSynchronize(d.cs);
            for (cudaEvent_t e : d.slot_ev) cudaEventDestroy(e);
            if (d.cs) cudaStreamDestroy(d.cs);
            if (d.plan_h) cudaFreeHost(d.plan_h);
            if (d.emb_h) cudaFreeHost(d.emb_h);
            if (d.pos_h) cudaFreeHost(d.pos_h);
            if (d.top_h) cudaFreeHost(d.top_h);
            if (d.lg_h) cudaFreeHost(d.lg_h);
            if (d.xh_h) cudaFreeHost(d.xh_h);
            if (d.win_rb) cudaFreeHost(d.win_rb);
            if (d.win_plan_h) cudaFreeHost(d.win_plan_h);
            if (d.win_cpu) cudaFreeHost(d.win_cpu);
            if (d.pin) cudaFreeHost(d.pin);
            if (d.blas) cublasDestroy(d.blas);
            if (d.s) cudaStreamDestroy(d.s);
        }
    }

    template <class T>
    bool dalloc(int dev, T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaMalloc(&v, std::max<size_t>(n * sizeof(T), 16)) != cudaSuccess) {
            err = "CUDA" + std::to_string(dev) + ": out of device memory (" + std::to_string(n * sizeof(T) >> 20) +
                  " MB)";
            return false;
        }
        owned.push_back(v);
        owned_dev.push_back(dev);
        acct[dev][(size_t) acat] += std::max<size_t>(n * sizeof(T), 16);
        p = (T*) v;
        return true;
    }
    void own_since(size_t n0, int dev) {
        while (owned_dev.size() < owned.size()) owned_dev.push_back(dev);
        (void) n0;
    }

    // a GGUF's experts (pack_gguf.py: experts.E.w1 / w3 / w2.weight, each expert's slice of the GGUF's stacked tensor):
    // the blob gate | up | down, each [rows][blocks] as stored, 256-byte aligned; gate, up and down of any types the
    // kernels take (one each for the layer's 128 experts)
    bool expert_layout_gg(Layer& l, int li, std::string& err) {
        const std::string E = P + "layers." + std::to_string(li) + ".block_sparse_moe.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};
        l.src.assign((size_t) g.n_exp, {});
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const int n = r < 2 ? g.ff : g.dim, k = r < 2 ? g.dim : g.ff;
            const Tensor* t0 = pk.get(E + "0." + parts[r] + ".weight");
            const int ty = t0 ? gg::type_of(t0->dtype) : -1;
            const size_t bytes = (size_t) n * gg::row_bytes(ty, k);
            if (t0 == nullptr || !gg::expert_type(ty) || bytes == 0 || t0->bytes != bytes) {
                err = E + "0." + parts[r] + ": not a [" + std::to_string(n) + "][" + std::to_string(k) +
                      "] expert matrix the GGUF kernels take (" + (t0 ? t0->dtype : std::string("missing")) + ")";
                return false;
            }
            for (int e = 0; e < g.n_exp; ++e) {
                const Tensor* t = pk.get(E + std::to_string(e) + "." + parts[r] + ".weight");
                if (t == nullptr || t->dtype != t0->dtype || t->bytes != bytes) {
                    err = E + std::to_string(e) + "." + parts[r] + ": missing, or not expert 0's type and size";
                    return false;
                }
                Src& s = l.src[(size_t) e];
                s.p[r * 3] = pk.data(*t);
                s.n[r * 3] = bytes;
                for (int q = 1; q < 3; ++q) {
                    s.p[r * 3 + q] = nullptr;
                    s.n[r * 3 + q] = 0;
                }
            }
            l.lay.trellis[r] = l.lay.suh[r] = l.lay.svh[r] = off;
            l.lay.gg_type[r] = ty;
            l.lay.k2[r] = 0;
            off += up(bytes);
        }
        l.lay.fmt = gg::kFmtGG;
        l.lay.cb = 0;
        l.blob = off;
        if (o.cpu_threads >= 0 && !gg_fmt(l.lay.gg_type[0], l.lay.gg_type[1], l.lay.gg_type[2], g.dim, g.ff, l.gf, err)) {
            // (a build without ggml-cpu, or a type it has no dot for: the experts outside VRAM are staged instead)
            info.push_back("NOTE: no CPU lane - layer " + std::to_string(li) + "'s experts: " + err);
            std::fprintf(stderr, "maya-mm3: %s\n", info.back().c_str());
            err.clear();
            o.cpu_threads = -1;
        }
        return true;
    }

    bool expert_layout(Layer& l, int li, std::string& err) {
        if (gguf) return expert_layout_gg(l, li, err);
        const std::string E = P + "layers." + std::to_string(li) + ".block_sparse_moe.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        const char* pieces[3] = {".trellis", ".suh", ".svh"};
        l.src.assign((size_t) g.n_exp, {});
        size_t off = 0;
        for (int e = 0; e < g.n_exp; ++e) {
            size_t o2 = 0;
            for (int r = 0; r < 3; ++r) {
                for (int q = 0; q < 3; ++q) {
                    const Tensor* t = pk.get(E + std::to_string(e) + "." + parts[r] + pieces[q]);
                    if (!t) {
                        err = E + std::to_string(e) + "." + parts[r] + pieces[q] + ": missing";
                        return false;
                    }
                    l.src[(size_t) e].p[r * 3 + q] = pk.data(*t);
                    l.src[(size_t) e].n[r * 3 + q] = t->bytes;
                    if (e == 0) {
                        if (q == 0) {
                            l.lay.trellis[r] = o2;
                            l.lay.k2[r] = (int) t->shape[2] / 8;
                        } else if (q == 1) {
                            l.lay.suh[r] = o2;
                        } else {
                            l.lay.svh[r] = o2;
                        }
                    } else if (q == 0 && (int) t->shape[2] / 8 != l.lay.k2[r]) {
                        err = E + std::to_string(e) + "." + parts[r] + ": bits differ from expert 0's";
                        return false;
                    }
                    o2 += up(t->bytes);
                }
            }
            if (e == 0) off = o2;
            else if (o2 != off) {
                err = E + std::to_string(e) + ": pieces differ in size from expert 0's";
                return false;
            }
        }
        l.lay.cb = 2;
        l.blob = off;
        if (l.lay.k2[0] != l.lay.k2[1]) {
            err = E + ": gate and up at different bits";
            return false;
        }
        return true;
    }

    // expert e's blob assembled in host memory dst (l.lay's offsets)
    void assemble(const Layer& l, int e, uint8_t* dst) const {
        const Src& s = l.src[(size_t) e];
        for (int r = 0; r < 3; ++r) {
            std::memcpy(dst + l.lay.trellis[r], s.p[r * 3 + 0], s.n[r * 3 + 0]);
            if (s.n[r * 3 + 1] > 0) std::memcpy(dst + l.lay.suh[r], s.p[r * 3 + 1], s.n[r * 3 + 1]);   // (a GGUF's: none)
            if (s.n[r * 3 + 2] > 0) std::memcpy(dst + l.lay.svh[r], s.p[r * 3 + 2], s.n[r * 3 + 2]);
        }
    }

    // ---- the CPU lane or a staging copy, by their measured costs (as maya-ds4's PCIe lane): a token's experts that are
    // neither in VRAM nor in a staging slot are computed on the CPU lane, or - those in the pinned pool - copied into a
    // staging slot and computed by the GPU (the copies run on the GPU's copy engine while the CPU lane computes the
    // others, and the slot keeps the expert for the next tokens).  calibrate_lane times both on this PC at the load:
    // an expert's copy from the pool, and the CPU lane's time for one expert and for four.  MM3_LANE_SPLIT=0: every such
    // expert on the CPU lane (as before)
    double lane_cpu1 = 0, lane_cpu_add = 0, lane_copy = 0;   // ms (lane_copy 0: not measured - no split)

    // of m such experts (each in the pool), how many to stage: the fewest that bring max(CPU lane, copies) down by more
    // than a tenth
    int lane_split(int m) const {
        if (lane_copy <= 0 || m <= 0) return 0;
        const auto cpu = [&](int c) { return c > 0 ? lane_cpu1 + (c - 1) * lane_cpu_add : 0.0; };
        int best = 0;
        double bt = cpu(m);
        for (int s = 1; s <= m; ++s) {
            const double t = std::max(cpu(m - s), s * lane_copy);
            if (t < bt * 0.9) {
                bt = t;
                best = s;
            }
        }
        return best;
    }

    void calibrate_lane() {
        if (!lane || !pool) return;
        if (const char* v = ::mm3::env("MM3_LANE_SPLIT"); v != nullptr && std::atoi(v) == 0) return;
        // eight pooled experts of one GPU's layers (different ones: the CPU's caches must not hold them yet), all of the
        // first one's format - the lane's call for four takes one layer's format for them all (with few experts outside
        // VRAM the eight can span layers whose experts differ in type: a GGUF's IQ1_S and IQ2_XXS - it read past them)
        Dev* dp = nullptr;
        std::vector<std::pair<const Layer*, int>> ex;
        const auto same_fmt = [&](const Layer& a, const Layer& b) {
            if (a.blob != b.blob) return false;
            for (int r = 0; r < 3; ++r)
                if (a.lay.gg_type[r] != b.lay.gg_type[r] || a.lay.k2[r] != b.lay.k2[r]) return false;
            return true;
        };
        for (Dev& d : D) {
            if (d.n_slots <= 0) continue;
            ex.clear();
            for (int li : d.layers) {
                const Layer& l = L[(size_t) li];
                if (!ex.empty() && !same_fmt(l, *ex[0].first)) continue;
                for (int e = 0; e < g.n_exp && ex.size() < 8 && !l.host_blob.empty(); ++e)
                    if (l.host_blob[(size_t) e] != nullptr) ex.push_back({&l, e});
            }
            if (ex.size() >= 8) {
                dp = &d;
                break;
            }
        }
        if (dp == nullptr) return;
        auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        cudaSetDevice(dp->id);
        std::vector<double> tc;
        for (int r = 0; r < 5; ++r) {   // (the first warms the path up)
            const Layer& l = *ex[(size_t) r].first;
            const double t0 = now();
            if (cudaMemcpyAsync(dp->stg, l.host_blob[(size_t) ex[(size_t) r].second], l.blob, cudaMemcpyHostToDevice,
                                dp->s) != cudaSuccess ||
                cudaStreamSynchronize(dp->s) != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            if (r > 0) tc.push_back(1e3 * (now() - t0));
        }
        dp->slot_owner.assign((size_t) dp->n_slots, -1);   // (slot 0 was written)
        std::vector<float> x((size_t) g.dim), out((size_t) g.dim);
        for (int i = 0; i < g.dim; ++i) x[(size_t) i] = 0.01f * (float) ((i * 37) % 101 - 50);
        std::vector<double> t1;
        for (int r = 0; r < 4; ++r) {
            const std::vector<CpuExpert> one{cpu_expert(*ex[(size_t) r].first, ex[(size_t) r].second, 1.0f)};
            const double t0 = now();
            lane_run(one, x.data(), out.data());
            t1.push_back(1e3 * (now() - t0));
        }
        std::vector<CpuExpert> four;
        for (int r = 4; r < 8; ++r) four.push_back(cpu_expert(*ex[(size_t) r].first, ex[(size_t) r].second, 0.25f));
        const double t0 = now();
        lane_run(four, x.data(), out.data());
        const double t4 = 1e3 * (now() - t0);
        lane_copy = med(tc);
        lane_cpu1 = med(t1);
        lane_cpu_add = std::max(0.0, (t4 - lane_cpu1) / 3.0);
        char b[200];
        std::snprintf(b, sizeof b, "CPU lane vs staging (measured): an expert %.2f ms on the CPU lane (+%.2f ms each more), "
                      "%.2f ms to copy - 4 such experts: %d staged", lane_cpu1, lane_cpu_add, lane_copy, lane_split(4));
        info.push_back(b);
    }

    // RAM a load leaves free (the OS, the page cache's working set, the engine's own threads): a tenth of the RAM,
    // 4 to 16 GB (8 when the RAM is not known)
    static double ram_headroom() {
        const double t = (double) os::ram_total();
        return t > 0 ? std::min(16.0, std::max(4.0, t / 10.0 / (1u << 30))) * (1u << 30) : 8.0 * (1u << 30);
    }

    // a layer's dense tensors' bytes in VRAM (the EXL3 matrices as stored, the F32 / BF16 vectors and matrices as F32),
    // its routed experts left out
    size_t dense_bytes(int li) const {
        const std::string pre = P + "layers." + std::to_string(li) + ".";
        size_t b = 0;
        for (auto it = pk.t.lower_bound(pre); it != pk.t.end() && it->first.compare(0, pre.size(), pre) == 0; ++it) {
            if (it->first.find(".experts.") != std::string::npos) continue;
            const Tensor& x = it->second;
            // (a GGUF's Q8_0 indexer projections: kept as stored where q8_index takes them, else F32 on the device; its
            //  other block types as stored)
            const bool q8f = x.dtype == "Q8_0" && it->first.find(".index_") != std::string::npos && !q8_index(it->first);
            b += x.dtype == "BF16" || x.dtype == "F16" ? 2 * x.bytes : q8f ? x.bytes / 34 * 128 : x.bytes;
        }
        return b;
    }
    // whether an indexer projection stays in its Q8_0 blocks on the device (a GGUF's [n][dim] Q8_0 rows)
    bool q8_index(const std::string& name) const {
        const Tensor* t = pk.get(name);
        return gguf && t != nullptr && t->dtype == "Q8_0" && t->shape.size() == 2 && t->shape[1] == g.dim &&
               g.dim % 32 == 0 && t->bytes == (uint64_t) t->shape[0] * (uint64_t) (g.dim / 32) * 34u;
    }
    // the decode's indexer projections (a token's or a window's T rows of xn -> iq, ik): both Q8_0 matrices in one
    // launch, else each as it is held
    void index_proj(const Layer& l, Dev& d, int T) {
        const int nq = g.ix_heads * g.ix_dim;
        if (l.wiq8 != nullptr) {
            k::rows_dot_q8(l.wiq8, nq, l.wik8, g.ix_dim, g.dim, d.xn, T, d.iq, d.ik, d.s);
            return;
        }
        k::rows_dot(l.wiq, nq, g.dim, d.xn, T, d.iq, d.s);
        k::rows_dot(l.wik, g.ix_dim, g.dim, d.xn, T, d.ik, d.s);
    }
    // a layer's expert blob (the plan's estimate before expert_layout reads it): EXL3's three trellises with their
    // scales, a GGUF's three matrices aligned
    size_t blob_estimate(int li) const {
        const std::string E = P + "layers." + std::to_string(li) + ".block_sparse_moe.experts.0.";
        if (gguf) {
            size_t b = 0;
            for (const char* r : {"w1", "w3", "w2"})
                if (const Tensor* t = pk.get(E + r + ".weight")) b += up(t->bytes);
            return b;
        }
        const Tensor* t = pk.get(E + "w1.trellis");
        return t ? (size_t) (3.0 * (double) t->bytes * 1.02) : 0;
    }

    // plan_memory: the context (o.max_ctx), KV streaming's ring and the staging slots that fit each GPU and the RAM.
    // Per GPU: its layers' dense weights, the scratch of a chunk, the staging slots, the output head (the last GPU),
    // the DSpark draft and the vision tower (estimates), then per position its layers' K / V rows (the ring's with KV
    // streaming) and index keys - with reserve_gb and 1 GB for the experts' first slots kept free.  The host cache of
    // KV streaming against the RAM this process can still get, less ram_headroom().
    bool plan_memory(int& kv_ring, std::string& err) {
        const double GB = (double) (1u << 30);
        const size_t kvpos = k::kv_bytes(o.kv_k, o.kv_v, (size_t) g.n_kv);   // a layer's K / V of one position
        const size_t idxpos = (size_t) g.ix_dim * 2;                         // ... and its index key
        const double reserve = (o.reserve_gb < 0 ? 1.0 : o.reserve_gb) * GB + 1.0 * GB;
        size_t blob_max = 0;
        for (int li = 0; li < g.n_layers; ++li)
            if (g.moe[(size_t) li]) blob_max = std::max(blob_max, blob_estimate(li));
        auto fixed_of = [&](size_t di, int slots) {
            double b = 0;
            for (int li : D[di].layers) b += (double) dense_bytes(li);
            const double T = (double) o.chunk;
            b += T * g.dim * 4 * 9 + T * g.n_head * g.hd * 6 + T * std::max({g.ff_dense, g.ff_shared, g.ff}) * 8 +
                 T * g.k_exp * (double) g.dim * 4 + T * g.k_exp * (double) g.ff * 2;   // (the chunk's scratch)
            b += (double) slots * (double) blob_max + 0.2 * GB;                    // (staging; the window's buffers)
            if (di + 1 == D.size()) {                                               // (the output head, its logits)
                const Tensor* h = pk.get(gguf ? "language_model.lm_head.weight" : "language_model.lm_head.trellis");
                b += (h ? (double) h->bytes : 0.0) + 32.0 * g.vocab * 4;
                if (o.dspark) b += 3.0 * GB;
            }
            // (the vision tower: no room - it stays in host memory, a picture's room lent by the staging slots)
            if (gguf) b += (double) gg::scratch_bytes();   // (the GGUF kernels' activations, FP16 bands, cuBLAS)
            if (gguf) b += 4.0 * g.ix_heads * g.ix_dim * g.dim;   // (wtmp: a Q8_0 indexer projection in F32)
            else if (x3::blas_wanted()) b += 0.2 * GB;
            return b;
        };
        auto kv_of = [&](size_t di, int ctx, int ring) {
            const double n = (double) D[di].layers.size();
            return n * ((double) (ring > 0 ? ring : ctx) * (double) kvpos + (double) ctx * (double) idxpos);
        };
        auto fits = [&](int ctx, int ring, int slots, std::string* worst) {
            bool ok = true;
            for (size_t di = 0; di < D.size(); ++di) {
                const double need = fixed_of(di, slots) + kv_of(di, ctx, ring) + reserve;
                if (need > (double) D[di].free0) {
                    ok = false;
                    if (worst)
                        *worst = "CUDA" + std::to_string(D[di].id) + " needs " + std::to_string(need / GB).substr(0, 5) +
                                 " GB, has " + std::to_string((double) D[di].free0 / GB).substr(0, 5) + " GB free";
                }
            }
            return ok;
        };
        const int want_ctx = o.max_ctx;
        int ctx = o.max_ctx, ring = kv_ring, slots = o.staging;
        std::string why;
        if (!fits(ctx, ring, slots, &why) && ring == 0 && ctx > 32768) {
            ring = 32768;   // (KV streaming: the last 32768 positions a layer in VRAM, all of them in host memory)
            info.push_back("KV streaming on: the whole context's K / V do not fit beside the model (" + why + ")");
        }
        while (!fits(ctx, ring, slots, &why) && ctx > 8192) {
            ctx = std::max(8192, ctx / 2 / 4096 * 4096);
            if (ring >= ctx) ring = 0;
        }
        while (!fits(ctx, ring, slots, &why) && slots > 16) slots = std::max(16, slots / 2);
        if (!fits(ctx, ring, slots, &why)) {
            err = "not enough GPU memory for MiniMax-M3.1's dense part at 8K positions (" + why +
                  "): more GPUs, or a smaller --chunk";
            return false;
        }
        // KV streaming's host cache: every position of every layer, pinned
        if (ring > 0) {
            const double per = (double) g.n_layers * (double) kvpos;
            const double avail = (double) os::ram_avail() - ram_headroom();
            if (os::ram_avail() > 0 && (double) ctx * per > avail) {
                int c2 = (int) std::max(0.0, std::floor(avail / per / 4096.0) * 4096.0);
                if (c2 < 8192) {
                    err = "KV streaming: " + std::to_string((int) ((double) ctx * per / GB + 0.5)) + " GB of pinned RAM "
                          "for a context of " + std::to_string(ctx) + " positions, " +
                          std::to_string((int) std::max(0.0, avail / GB)) + " GB to spare: a smaller --max-context";
                    return false;
                }
                info.push_back("KV streaming: the RAM holds " + std::to_string(c2) + " positions' K / V, not " +
                               std::to_string(ctx));
                ctx = c2;
                if (ring >= ctx) ring = 0;
            }
        }
        if (ctx != want_ctx)
            info.push_back("context " + std::to_string(ctx) + " positions (asked " + std::to_string(want_ctx) +
                           "): what this PC's GPU memory and RAM hold");
        if (slots != o.staging)
            info.push_back("staging slots " + std::to_string(slots) + " a GPU (asked " + std::to_string(o.staging) +
                           "): GPU memory");
        o.max_ctx = ctx;
        o.staging = slots;
        if (ring > 0) ring = std::max({4096, o.chunk, (ring + g.blk - 1) / g.blk * g.blk});
        kv_ring = ring >= o.max_ctx ? 0 : ring;
        return true;
    }

    bool load(const Options& opt, std::string& err) {
        const double t0 = now();
        o = opt;
        if (!pk.load(o.pack, err) || !g.read(pk, err)) return false;
        gguf = pk.str("format") == "gguf";
        if (gguf) {   // (the prompt GEMMs' backend; hipBLASLt's measured kernels kept beside the pack)
            const char* ls = ::mm3::env("MM3_LT_STORE");
            gg::set_lt_store(ls != nullptr ? ls : (o.pack + "/lt-picks.txt").c_str());
            gg::set_gemm(o.prefill_gemm > 0 ? 1 : 0);
        }
        if (gguf) {
            const Tensor* emb = pk.get(P + "embed_tokens.weight");
            if (emb == nullptr || gg::host_row_bytes(emb->dtype, g.dim) * (size_t) g.vocab != emb->bytes) {
                err = "the GGUF's token embedding: " + (emb ? emb->dtype + " of " + std::to_string(emb->bytes) + " bytes"
                                                            : std::string("missing")) + ", not [vocab][dim] in a type read here";
                return false;
            }
            info.push_back("weights: GGUF " + pk.str("quant_name", "?") + " (dense " + pk.str("gguf.dense", "?") +
                           ", experts " + pk.str("gguf.experts", "?") + ")");
            if (o.dspark && !pk.has("dspark.num_hidden_layers")) {
                // (the DSpark draft is the EXL3 repo's dspark/: tools/mm3/pack_gguf.py indexes it from --companions)
                info.push_back("NOTE: DSpark drafting: this GGUF pack has no draft (tools/mm3/pack_gguf.py --companions) "
                               "- off");
                o.dspark = false;
            }
        }
        int ndev = 0;
        CK(cudaGetDeviceCount(&ndev));
        if (o.gpus.empty())
            for (int i = 0; i < ndev; ++i) o.gpus.push_back(i);
        for (int id : o.gpus) {
            if (id < 0 || id >= ndev) {
                err = "no CUDA" + std::to_string(id);
                return false;
            }
            Dev d;
            d.id = id;
            CK(cudaSetDevice(id));
            CK(cudaStreamCreateWithFlags(&d.s, cudaStreamNonBlocking));
            CB(cublasCreate(&d.blas));
            CB(cublasSetStream(d.blas, d.s));
            CB(cublasSetMathMode(d.blas, CUBLAS_PEDANTIC_MATH));   // F32 GEMMs in F32 (no TF32)
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            d.free0 = fr;
            D.push_back(std::move(d));
        }
        // the layers over the GPUs by what each needs against each GPU's free memory: a layer's dense weights, its routed
        // experts and its caches, the last GPU's room less the output head (and the DSpark draft) - so every GPU keeps
        // about the same share of its memory over (the dense layers hold no experts: by layer count the first GPU kept
        // GBs free while the others' experts went to the CPU lane).  MAYA_MM3_SPLIT_OLD=1: by layer count and free memory
        {
            const bool old = ::mm3::env("MM3_SPLIT_OLD") != nullptr;
            std::vector<double> w((size_t) g.n_layers, 1.0), cap(D.size());
            const double ctx = (double) o.max_ctx;
            const double kvr = o.kv_resident > 0 ? std::min(ctx, (double) o.kv_resident) : ctx;
            for (int li = 0; li < g.n_layers && !old; ++li)
                w[(size_t) li] = (double) dense_bytes(li) +
                                 (g.moe[(size_t) li] ? (double) g.n_exp * (double) blob_estimate(li) : 0.0) +
                                 kvr * (double) k::kv_bytes(o.kv_k, o.kv_v, (size_t) g.n_kv) + ctx * g.ix_dim * 2.0;
            double total = 0, wsum = 0;
            for (size_t i = 0; i < D.size(); ++i) {
                cap[i] = (double) D[i].free0;
                if (!old && i + 1 == D.size()) {
                    const Tensor* h = pk.get(gguf ? "language_model.lm_head.weight" : "language_model.lm_head.trellis");
                    cap[i] -= (h ? (double) h->bytes : 0.0) + 32.0 * g.vocab * 4 + (o.dspark ? 3.0 * (1u << 30) : 0.0);
                }
                cap[i] = std::max(cap[i], 0.05 * (double) D[i].free0);
                total += cap[i];
            }
            for (double x : w) wsum += x;
            double acc = 0, wacc = 0;
            size_t di = 0;
            L.assign((size_t) g.n_layers, {});
            const auto& sp = o.layer_split;
            if (!sp.empty() && (sp.size() + 1 != D.size() || !std::is_sorted(sp.begin(), sp.end()) || sp.front() < 1 ||
                                sp.back() >= g.n_layers || std::adjacent_find(sp.begin(), sp.end()) != sp.end())) {
                err = "--layer-split: " + std::to_string(D.size() - 1) + " increasing first layers in 1 .. " +
                      std::to_string(g.n_layers - 1) + " (one for each GPU after the first)";
                return false;
            }
            for (int li = 0; li < g.n_layers; ++li) {
                if (sp.empty()) {
                    const double mid = (wacc + 0.5 * w[(size_t) li]) / wsum * total;
                    wacc += w[(size_t) li];
                    while (di + 1 < D.size() && acc + cap[di] < mid) acc += cap[di++];
                } else {
                    while (di < sp.size() && li >= sp[di]) ++di;
                }
                L[(size_t) li].dev = (int) di;
                D[di].layers.push_back(li);
            }
        }
        inv_freq.resize((size_t) g.n_rot / 2);
        k::rope_inv_freq(g.theta, g.n_rot, inv_freq.data());
        inv_dev.resize(D.size());
        if (const char* sv = ::mm3::env("MM3_ATTN_SPLIT")) attn_splits = std::max(1, std::atoi(sv));
        attn_splits = std::min(attn_splits, g.topk);
        const int T = o.chunk;
        // KV streaming (--kv-resident): the VRAM caches a ring of the last kv_ring positions (whole blocks, 4096 and a
        // prompt chunk at least), the rest of the context read from host memory; 0: all in VRAM
        int kv_ring = 0;
        if (o.kv_resident > 0) {
            kv_ring = std::max({4096, o.chunk, (o.kv_resident + g.blk - 1) / g.blk * g.blk});
            if (kv_ring >= o.max_ctx) kv_ring = 0;
        }
        // the context this PC holds: each GPU's dense weights, scratch, staging slots and caches against its free memory,
        // the KV streaming host cache against the RAM - too little turns KV streaming on, then gives a shorter context,
        // then fewer staging slots, each with a NOTE; an error only when even 8K positions do not fit
        if (!plan_memory(kv_ring, err)) return false;
        const size_t nblk = (size_t) (o.max_ctx + g.blk - 1) / g.blk;
        // the dense part, caches and scratch, GPU by GPU
        for (size_t di = 0; di < D.size(); ++di) {
            Dev& d = D[di];
            CK(cudaSetDevice(d.id));
            const size_t n0 = owned.size(), up0 = g_up;
            const int dv = d.id;
            auto A = [&](auto*& p, size_t n) { return dalloc(dv, p, n, err); };
            const int ffmax = std::max({g.ff_dense, g.ff_shared, g.ff});
            const int rows = T * g.k_exp;
            d.nblk = nblk;
            if (!A(d.h, (size_t) T * g.dim) || !A(d.xn, (size_t) T * g.dim) || !A(d.q, (size_t) T * g.n_head * g.hd) ||
                !A(d.k, (size_t) T * g.n_kv * g.hd) || !A(d.v, (size_t) T * g.n_kv * g.hd) ||
                !A(d.iq, (size_t) T * g.ix_heads * g.ix_dim) || !A(d.ik, (size_t) T * g.ix_dim) ||
                !A(d.bs, (size_t) g.ix_heads * T * nblk) || !A(d.att, (size_t) T * g.n_head * g.hd) ||
                !A(d.ao, (size_t) T * g.dim) || !A(d.g, (size_t) T * ffmax) || !A(d.u, (size_t) T * ffmax) ||
                !A(d.logit, (size_t) T * g.n_exp) || !A(d.w, (size_t) T * g.k_exp) ||
                !A(d.y, (size_t) rows * g.dim) || !A(d.sh, (size_t) T * g.dim) || !A(d.out, (size_t) T * g.dim) ||
                !A(d.xn16, (size_t) T * g.dim) || !A(d.att16, (size_t) T * g.n_head * g.hd) ||
                !A(d.h16, (size_t) std::max((size_t) rows * g.ff, (size_t) T * ffmax)) || !A(d.pos, (size_t) T) ||
                !A(d.kept, (size_t) g.ix_heads * T * g.topk) || !A(d.top, (size_t) T * g.k_exp) ||
                !A(d.rowidx, (size_t) rows) || !A(d.invrow, (size_t) rows) || !A(d.bounds, (size_t) g.n_exp + 1) ||
                !A(inv_dev[di], inv_freq.size()))
                return false;
            CK(cudaMemcpy(inv_dev[di], inv_freq.data(), inv_freq.size() * 4, cudaMemcpyHostToDevice));
            // before sm_80 (V100, T4, RTX 20): the prompt path's busy matrices as cuBLAS FP16 GEMMs over their decoded
            // weights - a handle and 160 MB of scratch for this GPU's stream (none: the tensor-core tiles take them)
            if (gguf) {   // (a GGUF's kernels: their activations' rows, FP16 weight bands and cuBLAS, before the weights)
                if (!gg::prepare(d.s)) {
                    err = "CUDA" + std::to_string(dv) + ": the GGUF kernels' scratch did not fit";
                    return false;
                }
                acct[dv][kAAct] += gg::scratch_bytes();
            } else if (x3::blas_wanted()) {
                if (x3::blas_enable(d.s, 160)) acct[dv][kAAct] += x3::blas_bytes(d.s);
                else info.push_back("CUDA" + std::to_string(dv) + ": no VRAM for the prompt path's FP16 GEMMs (tiles instead)");
            }
            for (int li : d.layers) {
                Layer& l = L[(size_t) li];
                const std::string N = P + "layers." + std::to_string(li) + ".";
                const std::string AT = N + "self_attn.";
                size_t n = 0;
                std::vector<void*> ow;
                l.moe = g.moe[(size_t) li] != 0;
                l.in1 = upload_f32(pk, N + "input_layernorm.weight", norm_add(), ow, n, err);
                l.post1 = upload_f32(pk, N + "post_attention_layernorm.weight", norm_add(), ow, n, err);
                l.qn1 = upload_f32(pk, AT + "q_norm.weight", norm_add(), ow, n, err);
                l.kn1 = upload_f32(pk, AT + "k_norm.weight", norm_add(), ow, n, err);
                l.iqn1 = upload_f32(pk, AT + "index_q_norm.weight", norm_add(), ow, n, err);
                l.ikn1 = upload_f32(pk, AT + "index_k_norm.weight", norm_add(), ow, n, err);
                if (q8_index(AT + "index_q_proj.weight") && q8_index(AT + "index_k_proj.weight")) {
                    l.wiq8 = upload_raw(pk, AT + "index_q_proj.weight", ow, err);
                    l.wik8 = upload_raw(pk, AT + "index_k_proj.weight", ow, err);
                    if (!err.empty()) return false;
                    if (d.wtmp == nullptr && !A(d.wtmp, (size_t) std::max(g.ix_heads, 1) * g.ix_dim * g.dim)) return false;
                } else {
                    l.wiq = upload_f32(pk, AT + "index_q_proj.weight", 0.0f, ow, n, err);
                    l.wik = upload_f32(pk, AT + "index_k_proj.weight", 0.0f, ow, n, err);
                }
                if (!err.empty()) return false;
                l.q = upload_mat(pk, AT + "q_proj", ow, err);
                l.k = upload_mat(pk, AT + "k_proj", ow, err);
                l.v = upload_mat(pk, AT + "v_proj", ow, err);
                l.o = upload_mat(pk, AT + "o_proj", ow, err);
                if (!err.empty()) return false;
                if (l.moe) {
                    const std::string M = N + "block_sparse_moe.";
                    l.gate = upload_f32(pk, M + "gate.weight", 0.0f, ow, n, err);
                    l.bias = upload_f32(pk, M + "e_score_correction_bias", 0.0f, ow, n, err);
                    if (!err.empty()) return false;
                    l.sg = upload_mat(pk, M + "shared_experts.gate_proj", ow, err);
                    l.su = upload_mat(pk, M + "shared_experts.up_proj", ow, err);
                    l.sd = upload_mat(pk, M + "shared_experts.down_proj", ow, err);
                    if (!err.empty() || !expert_layout(l, li, err)) return false;
                    d.slot_bytes = std::max(d.slot_bytes, l.blob);
                    ++st.experts;
                } else {
                    l.mg = upload_mat(pk, N + "mlp.gate_proj", ow, err);
                    l.mu = upload_mat(pk, N + "mlp.up_proj", ow, err);
                    l.md = upload_mat(pk, N + "mlp.down_proj", ow, err);
                    if (!err.empty()) return false;
                }
                for (void* x : ow) {
                    owned.push_back(x);
                    owned_dev.push_back(dv);
                }
                acat = kAKV;
                const size_t kvrows = (size_t) (kv_ring > 0 ? kv_ring : o.max_ctx) * g.n_kv;
                uint8_t* kvb = nullptr;
                if (!A(kvb, k::kv_bytes(o.kv_k, o.kv_v, kvrows))) return false;
                k::kv_carve(kvb, kvrows, o.kv_k, o.kv_v, o.kv_rot ? 3 : 0, l.kc, l.vc, l.kvq);
                if (!A(l.ic, (size_t) o.max_ctx * g.ix_dim)) return false;
                if (kv_ring > 0) {
                    l.kvh.ring = kv_ring;
                    if (!A(l.kvh.tag, (size_t) kv_ring)) return false;
                    CK(cudaMemset(l.kvh.tag, 0xff, (size_t) kv_ring * 4));
                }
                acat = kAAct;
            }
            if (di + 1 == D.size()) {
                head_dev = (int) di;
                if (!A(d.lg, (size_t) kHeadRows * g.vocab)) return false;
                CK(cudaMallocHost(&d.lg_h, (size_t) kHeadRows * g.vocab * sizeof(float)));
                std::vector<void*> ow;
                size_t n = 0;
                norm1 = upload_f32(pk, P + "norm.weight", norm_add(), ow, n, err);
                if (!err.empty()) return false;
                head = upload_mat(pk, "language_model.lm_head", ow, err);
                if (!err.empty()) return false;
                for (void* x : ow) {
                    owned.push_back(x);
                    owned_dev.push_back(dv);
                }
            }
            own_since(n0, dv);
            acct[dv][kAW] += g_up - up0;
            if (d.slot_bytes > 0) {
                d.n_slots = o.staging;
                acat = kAPool;
                const bool sok = A(d.stg, (size_t) d.n_slots * d.slot_bytes);
                acat = kAAct;
                if (!sok) return false;
                d.slot_owner.assign((size_t) d.n_slots, -1);
                d.slot_used.assign((size_t) d.n_slots, -1);
                d.slot_pend.assign((size_t) d.n_slots, -1);
                d.slot_iss.assign((size_t) d.n_slots, 0);
                CK(cudaStreamCreateWithFlags(&d.cs, cudaStreamNonBlocking));
                d.slot_ev.resize((size_t) d.n_slots);
                for (auto& e : d.slot_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                CK(cudaMallocHost(&d.pin, d.slot_bytes));
            }
            CK(cudaMallocHost(&d.plan_h, 16 * sizeof(unsigned long long)));
            CK(cudaMallocHost(&d.pos_h, (size_t) T * sizeof(int)));
            CK(cudaMallocHost(&d.top_h, (size_t) T * g.k_exp * sizeof(int)));
            if (L[(size_t) 0].dev == (int) di) CK(cudaMallocHost(&d.emb_h, (size_t) T * g.dim * sizeof(float)));
            if (!A(d.plan_d, 16)) return false;
            CK(cudaMallocHost(&d.xh_h, (size_t) (2 * g.dim + 4) * sizeof(float)));
            if (!A(d.cpu_d, (size_t) g.dim + 4)) return false;
            CK(cudaMallocHost(&d.win_rb, win_rb_bytes()));
            CK(cudaMallocHost(&d.win_plan_h, win_plan_bytes()));
            if (!A(d.win_plan_d, win_plan_bytes())) return false;
            CK(cudaHostAlloc(&d.win_cpu, ((size_t) kWin * g.dim + kWin + 4) * 4, cudaHostAllocMapped | cudaHostAllocPortable));
            if (!A(d.win_h, (size_t) kWin * 8 * g.ff) || !A(d.win_part, (size_t) kWin * 8 * g.dim)) return false;
            if (attn_splits > 1 && !A(d.attp, k::attn_part_floats(kWin, g.n_kv, attn_splits))) return false;
        }
        // KV streaming: every layer's whole cache in host memory (one pinned, mapped allocation, carved), written as the
        // VRAM ring is and read where the ring no longer holds a key
        if (kv_ring > 0) {
            const size_t rows = (size_t) o.max_ctx * g.n_kv;
            const size_t nb = k::kv_bytes(o.kv_k, o.kv_v, rows);
            size_t total = 0;
            for (const Layer& l : L) total += l.kvh.ring > 0 ? nb : 0;
            CK(cudaSetDevice(D[0].id));
            if (cudaHostAlloc((void**) &kv_host, total, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                kv_host = nullptr;
                err = "KV streaming: " + std::to_string(total >> 20) + " MB of pinned host memory not allocated";
                return false;
            }
            size_t off = 0;
            for (Layer& l : L) {
                if (l.kvh.ring <= 0) continue;
                k::kv_carve(kv_host + off, rows, o.kv_k, o.kv_v, o.kv_rot ? 3 : 0, l.kvh.kc, l.kvh.vc, l.kvh.q);
                off += nb;
                acct_kvh[D[(size_t) l.dev].id] += nb;
            }
            st.kv_ring = kv_ring;
            st.kv_host_gb = (double) total / (1u << 30);
        }
        // the DSpark draft on the head's GPU, before the experts take the memory
        tap_of.assign((size_t) g.n_layers, -1);
        if (o.dspark) {
            Dev& hd = D[(size_t) head_dev];
            CK(cudaSetDevice(hd.id));
            draftm = std::make_unique<Draft>();
            const Tensor* et = pk.get(P + "embed_tokens.weight");
            if (gguf)   // (the GGUF's table in its block type: a row dequantized as ggml does, then BF16 - the draft's dtype)
                draftm->set_emb_rows([this, et](int id, uint16_t* dst) {
                    std::vector<float> r((size_t) g.dim);
                    gg::host_row(et->dtype, pk.data(*et), g.dim, id, r.data());
                    for (int i = 0; i < g.dim; ++i) {
                        uint32_t u;
                        std::memcpy(&u, &r[(size_t) i], 4);
                        dst[i] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);   // (round to nearest even)
                    }
                });
            if (!draftm->load(pk, hd.id, hd.s, hd.blas, &head, (const uint16_t*) pk.data(*et), o.chunk, err))
                return false;
            const auto& tl = draftm->geo().taps;
            for (size_t j = 0; j < tl.size(); ++j) {
                if (tl[j] < 0 || tl[j] >= g.n_layers) {
                    err = "dspark: a tap at layer " + std::to_string(tl[j]);
                    return false;
                }
                tap_of[(size_t) tl[j]] = (int) j;
            }
            acat = kADraft;
            const bool tok = dalloc(hd.id, taps, tl.size() * (size_t) o.chunk * g.dim, err);
            acat = kAAct;
            if (!tok) return false;
            info.push_back("DSpark draft: " + std::to_string(draftm->bytes() >> 20) + " MB on CUDA" +
                           std::to_string(hd.id) + ", block " + std::to_string(draftm->geo().block) + ", taps " +
                           std::to_string(tl.size()));
        }
        // the vision tower, also before the experts take the memory (on by default: a pack without one runs text only)
        if (o.vision && o.vision_auto && !Vision::in_pack(pk)) {
            o.vision = false;
            info.push_back("vision: the pack has no vision tower - text only");
        }
        if (o.vision) {
            if (!Vision::in_pack(pk)) {
                err = "--vision: the pack has no vision tower (pack it again with tools/mm3/pack.py)";
                return false;
            }
            if (o.vision_gpu < 0 || o.vision_gpu >= (int) D.size()) {
                err = "--vision-gpu " + std::to_string(o.vision_gpu) + ": not one of the " + std::to_string(D.size()) +
                      " GPUs";
                return false;
            }
            vis = std::make_unique<Vision>();
            if (!vis->load(pk, D[(size_t) o.vision_gpu].id, err)) {
                if (o.vision_must) return false;
                info.push_back("NOTE: vision off: the tower did not load on CUDA" +
                               std::to_string(D[(size_t) o.vision_gpu].id) + " (" + err + ") - text only");
                std::fprintf(stderr, "maya-mm3: %s\n", info.back().c_str());
                err.clear();
                cudaGetLastError();
                vis.reset();
                o.vision = false;
            }
            if (vis) {
                const size_t vi = (size_t) o.vision_gpu;
                vis->set_lender([this, vi](size_t need) { return lend_slots(D[vi], need); },
                                [this, vi]() { drop_slots(D[vi]); });
                info.push_back("vision tower: on demand - its " + std::to_string(vis->host_bytes() >> 20) +
                               " MB in host memory (the pack's mapping), a picture at a time on CUDA" +
                               std::to_string(vis->device()) + " (up to " + std::to_string(vis->room_bytes() >> 20) +
                               " MB, given back after it), up to " + std::to_string(vis->geo().max_pixels) +
                               " pixels a picture");
                info.push_back("NOTE: pictures: the vision tower is not kept in VRAM - each picture copies it in, the "
                               "first one slower (its weights read from the SSD)");
            }
        }
        // the profile: routings [layer][expert] (a wrong size is ignored)
        prof.assign((size_t) g.n_layers * g.n_exp, 0);
        bool have_prof = false;
        if (!o.profile.empty()) {
            if (std::FILE* f = std::fopen(o.profile.c_str(), "rb")) {
                std::fseek(f, 0, SEEK_END);
                const long n = std::ftell(f);
                std::fseek(f, 0, SEEK_SET);
                if (n == (long) (prof.size() * 4) && std::fread(prof.data(), 4, prof.size(), f) == prof.size())
                    have_prof = true;
                std::fclose(f);
            }
            info.push_back(have_prof ? "expert profile " + o.profile
                                     : "expert profile " + o.profile + ": not read (missing or another geometry)");
            if (!have_prof) std::fill(prof.begin(), prof.end(), 0u);
        }
        // the routed experts kept in VRAM while the GPU keeps reserve_gb free: by the profile, the most routed of the
        // GPU's layers first; without one round robin over its layers (expert 0 of every layer, then 1, ...)
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const size_t reserve = (size_t) ((o.reserve_gb < 0 ? 1.0 : o.reserve_gb) * (1u << 30));
            size_t budget = fr > reserve ? fr - reserve : 0;
            std::vector<int> moe_layers;
            for (int li : d.layers)
                if (L[(size_t) li].moe) moe_layers.push_back(li);
            std::vector<int> nres(moe_layers.size(), 0);
            const int cap = o.resident_cap < 0 ? g.n_exp : std::min(o.resident_cap, g.n_exp);
            std::vector<std::vector<int>> chosen(moe_layers.size());
            if (have_prof) {
                struct C {
                    uint32_t n;
                    int i, e;
                };
                std::vector<C> c;
                for (size_t i = 0; i < moe_layers.size(); ++i)
                    for (int e = 0; e < g.n_exp; ++e)
                        c.push_back({prof[(size_t) moe_layers[i] * g.n_exp + e], (int) i, e});
                std::stable_sort(c.begin(), c.end(), [](const C& a, const C& b) { return a.n > b.n; });
                for (const C& x : c) {
                    const Layer& l = L[(size_t) moe_layers[(size_t) x.i]];
                    if (nres[(size_t) x.i] < cap && budget >= l.blob) {
                        budget -= l.blob;
                        ++nres[(size_t) x.i];
                        chosen[(size_t) x.i].push_back(x.e);
                    }
                }
                for (auto& v : chosen) std::sort(v.begin(), v.end());
            }
            for (bool grew = !have_prof; grew;) {
                grew = false;
                for (size_t i = 0; i < moe_layers.size(); ++i) {
                    const Layer& l = L[(size_t) moe_layers[i]];
                    if (nres[i] < cap && budget >= l.blob) {
                        budget -= l.blob;
                        ++nres[i];
                        grew = true;
                    }
                }
            }
            for (size_t i = 0; i < moe_layers.size(); ++i) {
                Layer& l = L[(size_t) moe_layers[i]];
                l.res_slot.assign((size_t) g.n_exp, -1);
                l.res_ids.clear();
                if (nres[i] > 0 && !dalloc(d.id, l.res_base, (size_t) nres[i] * l.blob, err)) return false;
                for (int sl = 0; sl < nres[i]; ++sl) {
                    const int e = have_prof ? chosen[i][(size_t) sl] : sl;
                    l.res_slot[(size_t) e] = sl;
                    l.res_ids.push_back(e);
                }
                st.resident += nres[i];
                st.vram_gb += (double) ((size_t) nres[i] * l.blob) / (1u << 30);
            }
        }
        st.experts = 0;
        for (const Layer& l : L)
            if (l.moe) st.experts += g.n_exp;
        // the resident blobs from the files, one thread a GPU
        {
            std::vector<std::thread> th;
            std::atomic<int> bad{0};
            std::mutex em;
            for (Dev& d : D) {
                th.emplace_back([&, dp = &d] {
                    cudaSetDevice(dp->id);
                    uint8_t* pin[2] = {nullptr, nullptr};
                    cudaEvent_t ev[2];
                    size_t mx = 0;
                    for (int li : dp->layers) mx = std::max(mx, L[(size_t) li].blob);
                    if (mx == 0) return;
                    bool pinned = true;   // (two pinned buffers in turn; the OS refusing them: one pageable, copied in turn)
                    for (int b = 0; b < 2; ++b) {
                        if (cudaMallocHost(&pin[b], mx) != cudaSuccess) {
                            cudaGetLastError();
                            pin[b] = nullptr;
                            pinned = false;
                        }
                        cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming);
                        cudaEventRecord(ev[b], dp->s);
                    }
                    std::vector<uint8_t> pageable;
                    if (!pinned) pageable.resize(mx);
                    int n = 0;
                    bool failed = false;
                    for (int li : dp->layers) {
                        const Layer& l = L[(size_t) li];
                        for (size_t s = 0; s < l.res_ids.size() && !failed; ++s, ++n) {
                            if (!pinned) {
                                assemble(l, l.res_ids[s], pageable.data());
                                failed = cudaMemcpy(l.res_base + s * l.blob, pageable.data(), l.blob,
                                                    cudaMemcpyHostToDevice) != cudaSuccess;
                                continue;
                            }
                            const int b = n & 1;
                            cudaEventSynchronize(ev[b]);
                            assemble(l, l.res_ids[s], pin[b]);
                            cudaMemcpyAsync(l.res_base + s * l.blob, pin[b], l.blob, cudaMemcpyHostToDevice, dp->s);
                            cudaEventRecord(ev[b], dp->s);
                        }
                    }
                    if (failed || cudaStreamSynchronize(dp->s) != cudaSuccess) {
                        std::lock_guard<std::mutex> lk(em);
                        ++bad;
                    }
                    for (int b = 0; b < 2; ++b) {
                        if (pin[b]) cudaFreeHost(pin[b]);
                        cudaEventDestroy(ev[b]);
                    }
                });
            }
            for (auto& t : th) t.join();
            if (bad) {
                err = "copying the resident experts failed";
                return false;
            }
        }
        // the pinned pool: the experts not in VRAM assembled once (a staging copy is then one DMA, the CPU lane reads them
        // from it) - as many as the RAM this process can still get holds beside ram_headroom(), the most routed first
        // (the expert profile); the rest are read from the model's mapped files.  A pool the OS will not pin whole is
        // tried at three quarters of the size, down to an eighth
        if (o.pin_pool) {
            struct Cand {
                uint32_t n;
                int li, e;
            };
            std::vector<Cand> cand;
            size_t total = 0;
            for (size_t li = 0; li < L.size(); ++li) {
                const Layer& l = L[li];
                if (!l.moe) continue;
                for (int e = 0; e < g.n_exp; ++e)
                    if (l.res_slot[(size_t) e] < 0) {
                        cand.push_back({prof[li * g.n_exp + e], (int) li, e});
                        total += l.blob;
                    }
            }
            std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.n > b.n; });
            const double avail = (double) os::ram_avail();   // (the page cache counts: the mapped files can be read again)
            double budget = avail > 0 ? avail - ram_headroom() : (double) total;
            size_t want = 0, nw = 0;
            for (; nw < cand.size(); ++nw) {
                const size_t b = L[(size_t) cand[nw].li].blob;
                if ((double) (want + b) > budget) break;
                want += b;
            }
            CK(cudaSetDevice(D[0].id));
            while (want > 0 && cudaHostAlloc((void**) &pool, want, cudaHostAllocPortable) != cudaSuccess) {
                pool = nullptr;
                cudaGetLastError();
                const size_t next = want / 4 * 3;
                if (next < total / 8) {
                    want = 0;
                    break;
                }
                while (nw > 0 && want > next) want -= L[(size_t) cand[--nw].li].blob;
            }
            if (want == 0 || pool == nullptr) {
                pool = nullptr;
                if (total > 0)
                    info.push_back("pinned pool: none - " + std::to_string(total >> 30) + " GB of experts outside VRAM, " +
                                   std::to_string((long long) std::max(0.0, budget / (1u << 30))) +
                                   " GB of RAM to spare: the CPU lane reads them from the model's files");
            } else {
                std::vector<std::pair<int, int>> todo;
                size_t off = 0;
                for (Layer& l : L)
                    if (l.moe) l.host_blob.assign((size_t) g.n_exp, nullptr);
                for (size_t i = 0; i < nw; ++i) {
                    Layer& l = L[(size_t) cand[i].li];
                    l.host_blob[(size_t) cand[i].e] = pool + off;
                    off += l.blob;
                    todo.push_back({cand[i].li, cand[i].e});
                }
                std::atomic<size_t> next{0};
                std::vector<std::thread> th;
                const unsigned nt = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
                for (unsigned t = 0; t < nt; ++t)
                    th.emplace_back([&] {
                        for (size_t i; (i = next.fetch_add(1)) < todo.size();) {
                            const Layer& l = L[(size_t) todo[i].first];
                            assemble(l, todo[i].second, (uint8_t*) l.host_blob[(size_t) todo[i].second]);
                        }
                    });
                for (auto& t : th) t.join();
                st.pinned_gb = (double) want / (1u << 30);
                st.pooled = (long) todo.size();
                if (nw < cand.size())
                    info.push_back("pinned pool: " + std::to_string(nw) + " of the " + std::to_string(cand.size()) +
                                   " experts outside VRAM (" + std::to_string(want >> 30) + " GB: the RAM to spare), the "
                                   "most routed; the rest from the model's files");
            }
        }
        // host-free layers: a MoE layer with every routed expert in VRAM makes its decode plan on the device (the routing
        // read, the experts' addresses looked up and a window's experts grouped there) - a token or a verify window is
        // enqueued whole, with no wait for the host between its layers.  MAYA_MM3_HOST_FREE=0: every layer through the
        // host as before (the same plan values, so the same tokens either way)
        {
            const char* hv = ::mm3::env("MM3_HOST_FREE");
            const bool on = hv == nullptr || std::atoi(hv) != 0;
            int nhf = 0, nmoe = 0;
            for (Layer& l : L) {
                if (!l.moe) continue;
                ++nmoe;
                if (!on || (int) l.res_ids.size() != g.n_exp) continue;
                Dev& d = D[(size_t) l.dev];
                CK(cudaSetDevice(d.id));
                std::vector<unsigned long long> tab((size_t) g.n_exp);
                for (int e = 0; e < g.n_exp; ++e)
                    tab[(size_t) e] = (unsigned long long) (l.res_base + (size_t) l.res_slot[(size_t) e] * l.blob);
                if (!dalloc(d.id, l.exp_tab, tab.size(), err)) return false;
                CK(cudaMemcpy(l.exp_tab, tab.data(), tab.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice));
                if (d.prof_d == nullptr) {
                    if (!dalloc(d.id, d.prof_d, (size_t) g.n_layers * g.n_exp, err)) return false;
                    CK(cudaMemset(d.prof_d, 0, (size_t) g.n_layers * g.n_exp * sizeof(unsigned)));
                }
                l.hf = true;
                ++nhf;
            }
            if (nhf > 0)
                info.push_back("decode: " + std::to_string(nhf) + " of " + std::to_string(nmoe) +
                               " MoE layers host-free (every expert in VRAM: the routing stays on the GPU)");
            // a token's and a window's whole forward as a graph (graph_ready): one GPU, every MoE layer host-free
            const char* gv = ::mm3::env("MM3_GRAPH");
            graph_on = (gv == nullptr || std::atoi(gv) != 0) && D.size() == 1 && nhf > 0 && nhf == nmoe && !o.verbose &&
                       ::mm3::env("MM3_DUMP_HS") == nullptr;
            if (graph_on) info.push_back("decode: a token's and a verify window's forward replayed as a graph");
        }
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            if (!x3::prepare(d.s)) {
                err = "exl3 prepare failed on CUDA" + std::to_string(d.id);
                return false;
            }
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            int nr = 0;
            for (int li : d.layers) nr += (int) L[(size_t) li].res_ids.size();
            info.push_back("CUDA" + std::to_string(d.id) + ": layers " + std::to_string(d.layers.front()) + "-" +
                           std::to_string(d.layers.back()) + ", " + std::to_string(nr) + " experts resident, " +
                           std::to_string(d.n_slots) + " staging slots, " + std::to_string(fr >> 20) + " MB free");
        }
        if (o.cpu_threads >= 0) {
            lane = std::make_unique<CpuLane>(o.cpu_threads, g.dim, g.ff, g.alpha, g.limit);
            info.push_back("CPU lane: " + std::to_string(lane->threads()) + " threads");
            st.cpu_threads = lane->threads();
            calibrate_lane();
        }
        {
            std::string kv = std::string("KV cache: K ") + k::kv_name(o.kv_k) + ", V " + k::kv_name(o.kv_v) +
                             (o.kv_rot && (o.kv_k || o.kv_v) ? " (rows rotated)" : "") + ", " +
                             std::to_string(k::kv_bytes(o.kv_k, o.kv_v, (size_t) g.n_kv) * g.n_layers / 1024) +
                             " KB a position";
            if (st.kv_ring > 0)
                kv += ", streamed: " + std::to_string(st.kv_ring) + " positions a layer in VRAM, all " +
                      std::to_string(o.max_ctx) + " in host memory (" + std::to_string((int) (st.kv_host_gb + 0.5)) +
                      " GB pinned)";
            info.push_back(kv);
        }
        // the Monitor's map of each GPU: the dense weights, the KV caches, the scratch, the DSpark draft, the routed
        // experts' slots (resident and staging), the vision tower
        for (const Dev& d : D) {
            cudaSetDevice(d.id);
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            std::array<size_t, kAN> a{};
            if (auto it = acct.find(d.id); it != acct.end()) a = it->second;
            const bool head = &d - D.data() == head_dev;
            const size_t drafter = a[kADraft] + (head && draftm ? draftm->bytes() : 0);
            const size_t vision = vis && vis->device() == d.id ? vis->bytes() : 0;
            long long slots = d.n_slots;
            for (int li : d.layers) slots += (long long) L[(size_t) li].res_ids.size();
            char pci[32] = "-";
            if (cudaDeviceGetPCIBusId(pci, sizeof pci, d.id) != cudaSuccess) std::snprintf(pci, sizeof pci, "-");
            char b[512];
            std::snprintf(b, sizeof b,
                          "VRAM dev=%d pci=%s total=%zu free=%zu weights=%zu kv=%zu kv_ram=%zu act=%zu drafter=%zu "
                          "window=0 pool=%zu tail=0 borrow=0 vision=0 vision_own=%zu prompt=0 tp=0 tp_n=1 slots=%lld "
                          "ctx=%d layers=%d-%d",
                          d.id, pci, tot, fr, a[kAW], a[kAKV], acct_kvh.count(d.id) ? acct_kvh.at(d.id) : 0, a[kAAct],
                          drafter, a[kAPool], vision, slots, o.max_ctx, d.layers.front(), d.layers.back());
            vram.push_back(b);
        }
        // the load's work on the legacy stream (cudaMemset of the KV rings' tags, the copies) done before the first
        // forward's on the non-blocking streams: nothing else orders the two (Windows + AMD submit queued work only when
        // the host waits - a pending memset there read as done)
        for (const Dev& d : D) {
            CK(cudaSetDevice(d.id));
            CK(cudaDeviceSynchronize());
        }
        if (const char* pv = ::mm3::env("MM3_PROF")) {
            pf.on = std::atoi(pv) != 0 && D.size() == 1;
            pf.sync = std::atoi(pv) == 2;
        }
        st.load_s = now() - t0;
        return true;
    }

    // expert e of layer l into staging slot s of d (in stream order: the slot's earlier readers are before it)
    bool stage(const Layer& l, Dev& d, int e, int s, std::string& err) {
        const uint8_t* src = l.host_blob.empty() ? nullptr : l.host_blob[(size_t) e];
        if (src == nullptr) {
            CK(cudaStreamSynchronize(d.s));   // (the pinned blob is free)
            assemble(l, e, d.pin);
            src = d.pin;
        }
        CK(cudaMemcpyAsync(d.stg + (size_t) s * d.slot_bytes, src, l.blob, cudaMemcpyHostToDevice, d.s));
        ++st.staged;
        return true;
    }

    // the device address of expert e of layer l for this call, staging it if it is not in VRAM (busy: the slots this
    // call already took)
    bool locate(Layer& l, Dev& d, int e, std::vector<char>& busy, unsigned long long& ptr, std::string& err) {
        const int rs = l.res_slot[(size_t) e];
        if (rs >= 0) {
            ptr = (unsigned long long) (l.res_base + (size_t) rs * l.blob);
            return true;
        }
        const long long key = (long long) (&l - L.data()) * 1024 + e;
        int at = -1;
        for (int s = 0; s < d.n_slots; ++s)
            if (d.slot_owner[(size_t) s] == key) at = s;
        if (at >= 0) score(key, true);   // (a promoted expert's use keeps it)
        if (at < 0) {
            for (int s = 0; s < d.n_slots; ++s)
                if (!busy[(size_t) s] && d.slot_pend[(size_t) s] < 0 &&
                    (at < 0 || d.slot_used[(size_t) s] < d.slot_used[(size_t) at]))
                    at = s;
            if (at < 0) {
                err = "no free staging slot";
                return false;
            }
            if (!stage(l, d, e, at, err)) return false;
            d.slot_owner[(size_t) at] = key;
        }
        busy[(size_t) at] = 1;
        d.slot_used[(size_t) at] = fwd;
        ptr = (unsigned long long) (d.stg + (size_t) at * d.slot_bytes);
        return true;
    }

    size_t win_rb_bytes() const { return (size_t) kWin * 8 * 8 + (size_t) kWin * g.dim * 4; }
    size_t win_plan_bytes() const { return (size_t) kWin * 8 * 8 * 2 + (size_t) kWin * 8 * 4 * 3 + 64; }

    // a window of T tokens' layer on the token path's kernels: the dense matrices on mv (T columns, each weight
    // decoded once for them all), the routed experts on win_experts (each distinct expert read once for every token that
    // picked it), the ones neither in VRAM nor staged on the CPU lane meanwhile (its parts in mapped memory, the device
    // waits for them before the combine)
    bool layer_window(Layer& l, Dev& d, int di, int T, std::string& err) {
        const auto job = [&](const x3::Mat& m, const float* x, int ldx, float* y, int ldy) {
            x3::MvJob j;
            j.m = &m;
            j.xf = x;
            j.y = y;
            j.ncol = T;
            j.xf_cs = (size_t) ldx;
            j.y_cs = (size_t) ldy;
            return j;
        };
        const int Q = g.n_head * g.hd, KV = g.n_kv * g.hd;
        k::rms(d.h, l.in1, T, g.dim, g.eps, d.xn, nullptr, d.s);
        const x3::MvJob qkv[3] = {job(l.q, d.xn, g.dim, d.q, Q), job(l.k, d.xn, g.dim, d.k, KV),
                                  job(l.v, d.xn, g.dim, d.v, KV)};
        if (!x3::mv(qkv, 3, d.s)) {
            err = "mv q/k/v (window)";
            return false;
        }
        k::qk_norm_rope(d.q, d.k, d.v, l.qn1, l.kn1, d.pos, T, g.n_head, g.n_kv, g.hd, g.n_rot, inv_dev[(size_t) di],
                        g.eps, l.kc, l.vc, d.s, 0, l.kvq, l.kvh);
        pmark(d, kPAttnIn);
        index_proj(l, d, T);
        k::index_norm_rope(d.iq, d.ik, l.iqn1, l.ikn1, d.pos, T, g.ix_heads, g.ix_dim, g.n_rot, inv_dev[(size_t) di],
                           g.eps, l.ic, d.s);
        const int nblk = cur_pos / g.blk + 1;
        k::block_scores(d.iq, l.ic, d.pos, T, g.ix_heads, nblk, g.blk, d.bs, d.s);
        k::select_blocks(d.bs, d.pos, T, g.ix_heads, nblk, g.blk, g.topk, d.kept, d.s);
        pmark(d, kPIndex);
        k::sparse_attn(d.q, l.kc, l.vc, d.kept, d.pos, T, g.n_head, g.n_kv, g.topk, g.blk, 1.0f / std::sqrt((float) g.hd),
                       d.att, d.s, l.kvq, l.kvh, attn_splits, d.attp);
        pmark(d, kPAttn);
        const x3::MvJob oj = job(l.o, d.att, Q, d.ao, g.dim);
        if (!x3::mv(&oj, 1, d.s)) {
            err = "mv o (window)";
            return false;
        }
        k::add(d.h, d.ao, (size_t) T * g.dim, d.s);
        k::rms(d.h, l.post1, T, g.dim, g.eps, d.xn, nullptr, d.s);
        pmark(d, kPOut);
        if (!l.moe) {
            const x3::MvJob gu[2] = {job(l.mg, d.xn, g.dim, d.g, g.ff_dense), job(l.mu, d.xn, g.dim, d.u, g.ff_dense)};
            if (!x3::mv(gu, 2, d.s)) {
                err = "mv dense gate/up (window)";
                return false;
            }
            k::swigluoai(d.g, d.u, (size_t) T * g.ff_dense, g.alpha, g.limit, d.g, nullptr, d.s);
            const x3::MvJob dj = job(l.md, d.g, g.ff_dense, d.ao, g.dim);
            if (!x3::mv(&dj, 1, d.s)) {
                err = "mv dense down (window)";
                return false;
            }
            k::add(d.h, d.ao, (size_t) T * g.dim, d.s);
            pmark(d, kPDense);
            return true;
        }
        k::rows_dot(l.gate, g.n_exp, g.dim, d.xn, T, d.logit, d.s);
        k::route(d.logit, l.bias, T, g.n_exp, g.k_exp, d.top, d.w, d.s);
        const x3::MvJob sgu[2] = {job(l.sg, d.xn, g.dim, d.g, g.ff_shared), job(l.su, d.xn, g.dim, d.u, g.ff_shared)};
        if (!x3::mv(sgu, 2, d.s)) {
            err = "mv shared gate/up (window)";
            return false;
        }
        k::swigluoai(d.g, d.u, (size_t) T * g.ff_shared, g.alpha, g.limit, d.g, nullptr, d.s);
        const x3::MvJob sdj = job(l.sd, d.g, g.ff_shared, d.sh, g.dim);
        if (!x3::mv(&sdj, 1, d.s)) {
            err = "mv shared down (window)";
            return false;
        }
        // the device plan's layout (win_plan_d): plan_ptr [kWin][8], plan_w [kWin][8], grp_ptr [kWin 8], grp_start
        // [kWin 8 + 1], grp_ent [kWin 8], n_groups
        const size_t off_w = (size_t) kWin * 8 * 8, off_gp = off_w + (size_t) kWin * 8 * 4,
                     off_gs = off_gp + (size_t) kWin * 8 * 8, off_ge = off_gs + ((size_t) kWin * 8 + 1) * 4,
                     off_ng = off_ge + (size_t) kWin * 8 * 4;
        const auto win_args = [&]() {
            x3::WinArgs a;
            a.grp_ptr = (const unsigned long long*) (d.win_plan_d + off_gp);
            a.grp_start = (const int*) (d.win_plan_d + off_gs);
            a.grp_ent = (const int*) (d.win_plan_d + off_ge);
            a.n_groups = (const int*) (d.win_plan_d + off_ng);
            a.plan_ptr = (const unsigned long long*) d.win_plan_d;
            a.plan_w = (const float*) (d.win_plan_d + off_w);
            a.x = d.xn;
            a.x_stride = (size_t) g.dim;
            a.h = d.win_h;
            a.part = d.win_part;
            a.sh_out = d.sh;
            a.out = d.out;
            a.out_stride = (size_t) g.dim;
            a.lay = l.lay;
            a.T = T;
            a.k = g.k_exp;
            a.n_embd = g.dim;
            a.n_ff = g.ff;
            a.limit = g.limit;
            return a;
        };
        pmark(d, kPRoute);
        if (l.hf) {   // host-free: the plan and its grouping made on the device (the host's values), nothing waited for
            const size_t lh = (size_t) (&l - L.data());
            if (!k::plan_dev(d.top, d.w, T, g.k_exp, l.exp_tab, g.scale, (unsigned long long*) d.win_plan_d,
                             (float*) (d.win_plan_d + off_w), d.prof_d + lh * g.n_exp,
                             (unsigned long long*) (d.win_plan_d + off_gp), (int*) (d.win_plan_d + off_gs),
                             (int*) (d.win_plan_d + off_ge), (int*) (d.win_plan_d + off_ng), d.s)) {
                err = "plan_dev (window)";
                return false;
            }
            st.profiled += (long) T * g.k_exp;
            if (!x3::win_experts(win_args(), d.s)) {
                err = "win_experts";
                return false;
            }
            k::add(d.h, d.out, (size_t) T * g.dim, d.s);
            pmark(d, kPExperts);
            return true;
        }
        int* rtop = (int*) d.win_rb;
        float* rw = (float*) (d.win_rb + (size_t) kWin * 8 * 4);
        float* rx = (float*) (d.win_rb + (size_t) kWin * 8 * 8);
        CK(cudaMemcpyAsync(rtop, d.top, (size_t) T * g.k_exp * 4, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(rw, d.w, (size_t) T * g.k_exp * 4, cudaMemcpyDeviceToHost, d.s));
        if (lane) CK(cudaMemcpyAsync(rx, d.xn, (size_t) T * g.dim * 4, cudaMemcpyDeviceToHost, d.s));
        const double ts = now();
        CK(cudaStreamSynchronize(d.s));
        st.t_sync += now() - ts;
        const size_t li = (size_t) (&l - L.data());
        // the plan: each token's experts (its 8 slots, k used), their pointers (0: the CPU lane's), the groups
        unsigned long long* pp = (unsigned long long*) d.win_plan_h;
        float* pw = (float*) (pp + (size_t) kWin * 8);
        unsigned long long* gp = (unsigned long long*) (pw + (size_t) kWin * 8);
        int* gs = (int*) (gp + (size_t) kWin * 8);
        int* ge = gs + kWin * 8 + 1;
        int* ng = ge + kWin * 8;
        std::vector<char> busy((size_t) d.n_slots, 0);
        std::vector<CpuExpert> cpu;
        std::vector<std::vector<std::pair<int, float>>> users;
        std::vector<std::pair<int, int>> order;   // the CPU entries (expert, its user) token by token, slot by slot
        std::vector<int> cpu_of((size_t) g.n_exp, -1);
        std::vector<std::vector<int>> by_ptr_ent;
        std::vector<unsigned long long> by_ptr;
        std::vector<char> stage_it((size_t) g.n_exp, 0);   // (lane_split's: distinct experts staged instead)
        if (lane && lane_copy > 0) {
            std::vector<int> cand;
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < g.k_exp; ++i) {
                    const int x = rtop[t * g.k_exp + i];
                    if (on_cpu(l, d, x) && pooled(l, x) && std::find(cand.begin(), cand.end(), x) == cand.end())
                        cand.push_back(x);
                }
            const int s = std::min((int) cand.size(), std::min(lane_split((int) cand.size()), d.n_slots / 2));
            for (int i = 0; i < s; ++i) stage_it[(size_t) cand[(size_t) i]] = 1;
        }
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < 8; ++i) {
                const int e = t * 8 + i;
                pp[e] = 0;
                pw[e] = 0.0f;
                if (i >= g.k_exp) continue;
                const int x = rtop[t * g.k_exp + i];
                const float wt = rw[t * g.k_exp + i] * g.scale;
                pw[e] = wt;
                ++prof[li * g.n_exp + x];
                if (lane && on_cpu(l, d, x) && !stage_it[(size_t) x]) {
                    if (cpu_seen.empty()) cpu_seen.assign((size_t) g.n_layers * g.n_exp, 0);
                    if (cpu_of[(size_t) x] < 0) ++cpu_seen[li * g.n_exp + x];
                    if (cpu_of[(size_t) x] < 0) consider_promotion(l, d, x);
                    if (cpu_of[(size_t) x] < 0) {
                        cpu_of[(size_t) x] = (int) cpu.size();
                        cpu.push_back(cpu_expert(l, x, 1.0f));
                        users.emplace_back();
                    }
                    users[(size_t) cpu_of[(size_t) x]].push_back({t, wt});
                    order.push_back({cpu_of[(size_t) x], (int) users[(size_t) cpu_of[(size_t) x]].size() - 1});
                    continue;
                }
                if (!locate(l, d, x, busy, pp[e], err)) return false;
                size_t gi = 0;
                while (gi < by_ptr.size() && by_ptr[gi] != pp[e]) ++gi;
                if (gi == by_ptr.size()) {
                    by_ptr.push_back(pp[e]);
                    by_ptr_ent.emplace_back();
                }
                by_ptr_ent[gi].push_back(e);
            }
        st.profiled += (long) T * g.k_exp;
        int ne = 0;
        for (size_t gi = 0; gi < by_ptr.size(); ++gi) {
            gp[gi] = by_ptr[gi];
            gs[gi] = ne;
            for (int e : by_ptr_ent[gi]) ge[ne++] = e;
        }
        gs[by_ptr.size()] = ne;
        *ng = (int) by_ptr.size();
        CK(cudaMemcpyAsync(d.win_plan_d, d.win_plan_h, win_plan_bytes(), cudaMemcpyHostToDevice, d.s));
        float* cpu_rows = d.win_cpu;
        int* cpu_flags = (int*) (d.win_cpu + (size_t) kWin * g.dim);
        unsigned* wait_word = (unsigned*) (cpu_flags + kWin);
        x3::WinArgs a = win_args();
        const bool with_cpu = !cpu.empty();
        if (with_cpu) {
            a.cpu_part = cpu_rows;
            a.cpu_stride = (size_t) g.dim;
            a.cpu_flag = cpu_flags;
            a.wait_ptr = wait_word;
            a.wait_val = ++d.win_wait;
        }
        if (!x3::win_experts(a, d.s)) {
            if (with_cpu) *(volatile unsigned*) wait_word = d.win_wait;   // (never leave the device waiting)
            err = "win_experts";
            return false;
        }
        if (with_cpu) {   // the CPU lane's parts while the device runs the others; then the word it waits on
            const double tl = now();
            lane_run_multi(cpu, users, rx, T, cpu_rows, &order);
            for (int t = 0; t < T; ++t) cpu_flags[t] = 0;
            for (const auto& u : users)
                for (const auto& tu : u) cpu_flags[tu.first] = 1;
            st.t_lane += now() - tl;
            st.cpu_experts += (long) cpu.size();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile unsigned*) wait_word = d.win_wait;
        }
        k::add(d.h, d.out, (size_t) T * g.dim, d.s);
        pmark(d, kPExperts);
        return true;
    }

    // a non-resident expert's use: its decayed score, now
    float score(long long key, bool add) {
        if (use_score.empty()) {
            use_score.assign((size_t) g.n_layers * 1024, 0.0f);
            use_last.assign(use_score.size(), 0);
        }
        float& sc = use_score[(size_t) key];
        sc *= std::pow(kDecay, (float) (fwd - use_last[(size_t) key]));
        use_last[(size_t) key] = fwd;
        if (add) sc += 1.0f;
        return sc;
    }

    // the promotions' copies on the copy stream - at a forward's end: during it, each 20+ MB copy would hold up the
    // forward's own small copies behind it on the GPU's one host -> device engine (its plans, the CPU lane's parts)
    void issue_promotions(Dev& d) {
        for (const Dev::Promo& q : d.promo_q) {
            cudaMemcpyAsync(d.stg + (size_t) q.slot * d.slot_bytes, q.src, q.bytes, cudaMemcpyHostToDevice, d.cs);
            cudaEventRecord(d.slot_ev[(size_t) q.slot], d.cs);
            d.slot_iss[(size_t) q.slot] = fwd;
        }
        d.promo_q.clear();
    }

    // the pending slots become their expert's - at a forward's start only, their copies waited for: which experts a
    // forward finds in VRAM follows from the tokens before it, not from when a copy happened to end (the GPU's and the
    // CPU lane's sums of an expert differ in their last bits: the answer is the same from run to run this way)
    // a picture's room (Vision::set_lender) when its GPU has none free: the staging slots, between forwards - their
    // experts dropped, staged or promoted again as the forwards after it ask (drop_slots)
    void* lend_slots(Dev& d, size_t need) {
        if (d.stg == nullptr || (size_t) d.n_slots * d.slot_bytes < need) return nullptr;
        cudaSetDevice(d.id);
        if (cudaStreamSynchronize(d.s) != cudaSuccess || (d.cs && cudaStreamSynchronize(d.cs) != cudaSuccess))
            return nullptr;
        return d.stg;
    }
    void drop_slots(Dev& d) {
        for (int s = 0; s < d.n_slots; ++s) {
            d.slot_owner[(size_t) s] = -1;
            d.slot_used[(size_t) s] = -1;
            d.slot_pend[(size_t) s] = -1;
            d.slot_iss[(size_t) s] = 0;
        }
        d.n_pending = 0;
        d.promo_q.clear();
    }

    void refresh_slots(Dev& d) {
        issue_promotions(d);   // (a forward that failed before its end)
        if (d.n_pending == 0) return;
        const double tw = now();
        for (int s = 0; s < d.n_slots; ++s)
            if (d.slot_pend[(size_t) s] >= 0 && fwd - d.slot_iss[(size_t) s] >= o.promote_lag &&
                cudaEventSynchronize(d.slot_ev[(size_t) s]) == cudaSuccess) {
                d.slot_owner[(size_t) s] = d.slot_pend[(size_t) s];
                d.slot_pend[(size_t) s] = -1;
                d.slot_used[(size_t) s] = fwd;
                --d.n_pending;
            }
        st.t_promo += now() - tw;
    }

    // expert e of layer l ran on the CPU lane: when it keeps coming back, copied into a staging slot meanwhile (the
    // slot whose expert scores least, not used in this forward nor pending)
    void consider_promotion(const Layer& l, Dev& d, int e) {
        const long long key = (long long) (&l - L.data()) * 1024 + e;
        const float sc = score(key, true);
        if (!o.promote || l.host_blob.empty() || l.host_blob[(size_t) e] == nullptr || sc < kPromote ||
            d.n_pending >= kMaxPending || d.n_slots == 0)
            return;
        int best = -1;
        float bs = sc;
        for (int s = 0; s < d.n_slots; ++s) {
            if (d.slot_pend[(size_t) s] >= 0 || d.slot_used[(size_t) s] == fwd) continue;
            const long long ok = d.slot_owner[(size_t) s];
            const float os = ok < 0 ? -1.0f : score(ok, false);
            if (os < bs) {
                bs = os;
                best = s;
            }
        }
        if (best < 0) return;
        d.slot_owner[(size_t) best] = -1;
        d.slot_pend[(size_t) best] = key;
        ++d.n_pending;
        d.promo_q.push_back({best, l.host_blob[(size_t) e], l.blob});
        ++st.promoted;
    }

    // the CPU lane's experts: the EXL3 lane's rows, or a GGUF's (mm3_cpu_gg.cpp) - one layer's experts a call
    void lane_run(const std::vector<CpuExpert>& ex, const float* x, float* out) {
        if (gguf) gg_run(*lane, g.dim, g.ff, g.limit, ex, x, out);
        else lane->run(ex, x, out);
    }
    void lane_run_multi(const std::vector<CpuExpert>& ex, const std::vector<std::vector<std::pair<int, float>>>& users,
                        const float* x, int T, float* out, const std::vector<std::pair<int, int>>* order) {
        if (gguf) gg_run_multi(*lane, g.dim, g.ff, g.limit, ex, users, x, T, out, order);
        else lane->run_multi(ex, users, x, T, out, order);
    }

    // whether expert e of layer l has its blob in the pinned pool (a staging copy is then one DMA)
    bool pooled(const Layer& l, int e) const { return !l.host_blob.empty() && l.host_blob[(size_t) e] != nullptr; }

    // whether expert e of layer l is the CPU lane's this token: neither in VRAM nor in one of d's staging slots
    bool on_cpu(const Layer& l, const Dev& d, int e) const {
        if (l.res_slot[(size_t) e] >= 0) return false;
        const long long key = (long long) (&l - L.data()) * 1024 + e;
        for (int s = 0; s < d.n_slots; ++s)
            if (d.slot_owner[(size_t) s] == key) return false;
        return true;
    }

    // expert e of layer l in host memory (the pinned pool's blob, or the mapped files' pieces) for the CPU lane
    CpuExpert cpu_expert(const Layer& l, int e, float w) const {
        CpuExpert c;
        c.w = w;
        const uint8_t* blob = l.host_blob.empty() ? nullptr : l.host_blob[(size_t) e];
        if (gguf) {   // (the three matrices' rows: the pool's blob, or the GGUF's mapping)
            c.gf = &l.gf;
            for (int r = 0; r < 3; ++r) c.gg[r] = blob ? blob + l.lay.trellis[r] : l.src[(size_t) e].p[r * 3];
            return c;
        }
        for (int r = 0; r < 3; ++r) {
            strata::kernels::cpu::Exl3Part& m = c.p[r];
            if (blob) {
                m.trellis = (const uint32_t*) (blob + l.lay.trellis[r]);
                m.suh = (const uint16_t*) (blob + l.lay.suh[r]);
                m.svh = (const uint16_t*) (blob + l.lay.svh[r]);
            } else {
                const Src& sp = l.src[(size_t) e];
                m.trellis = (const uint32_t*) sp.p[r * 3 + 0];
                m.suh = (const uint16_t*) sp.p[r * 3 + 1];
                m.svh = (const uint16_t*) sp.p[r * 3 + 2];
            }
            m.k = r < 2 ? g.dim : g.ff;
            m.n = r < 2 ? g.ff : g.dim;
            m.k2 = l.lay.k2[r];
        }
        return c;
    }

    // one token's layer on the decode kernels
    bool layer_token(Layer& l, Dev& d, int di, std::string& err) {
        const auto job = [](const x3::Mat& m, const float* x, float* y) {
            x3::MvJob j;
            j.m = &m;
            j.xf = x;
            j.y = y;
            return j;
        };
        k::rms(d.h, l.in1, 1, g.dim, g.eps, d.xn, nullptr, d.s);
        const x3::MvJob qkv[3] = {job(l.q, d.xn, d.q), job(l.k, d.xn, d.k), job(l.v, d.xn, d.v)};
        if (!x3::mv(qkv, 3, d.s)) {
            err = "mv q/k/v";
            return false;
        }
        k::qk_norm_rope(d.q, d.k, d.v, l.qn1, l.kn1, d.pos, 1, g.n_head, g.n_kv, g.hd, g.n_rot, inv_dev[(size_t) di],
                        g.eps, l.kc, l.vc, d.s, 0, l.kvq, l.kvh);
        pmark(d, kPAttnIn);
        const float one = 1.0f, zero = 0.0f;
        index_proj(l, d, 1);
        k::index_norm_rope(d.iq, d.ik, l.iqn1, l.ikn1, d.pos, 1, g.ix_heads, g.ix_dim, g.n_rot, inv_dev[(size_t) di],
                           g.eps, l.ic, d.s);
        const int nblk = cur_pos / g.blk + 1;
        k::block_scores(d.iq, l.ic, d.pos, 1, g.ix_heads, nblk, g.blk, d.bs, d.s);
        k::select_blocks(d.bs, d.pos, 1, g.ix_heads, nblk, g.blk, g.topk, d.kept, d.s);
        pmark(d, kPIndex);
        k::sparse_attn(d.q, l.kc, l.vc, d.kept, d.pos, 1, g.n_head, g.n_kv, g.topk, g.blk, 1.0f / std::sqrt((float) g.hd),
                       d.att, d.s, l.kvq, l.kvh, attn_splits, d.attp);
        pmark(d, kPAttn);
        const x3::MvJob oj = job(l.o, d.att, d.ao);
        if (!x3::mv(&oj, 1, d.s)) {
            err = "mv o";
            return false;
        }
        k::add(d.h, d.ao, (size_t) g.dim, d.s);
        k::rms(d.h, l.post1, 1, g.dim, g.eps, d.xn, nullptr, d.s);
        pmark(d, kPOut);
        if (!l.moe) {
            const x3::MvJob gu[2] = {job(l.mg, d.xn, d.g), job(l.mu, d.xn, d.u)};
            if (!x3::mv(gu, 2, d.s)) {
                err = "mv dense gate/up";
                return false;
            }
            k::swigluoai(d.g, d.u, (size_t) g.ff_dense, g.alpha, g.limit, d.g, nullptr, d.s);
            const x3::MvJob dj = job(l.md, d.g, d.ao);
            if (!x3::mv(&dj, 1, d.s)) {
                err = "mv dense down";
                return false;
            }
            k::add(d.h, d.ao, (size_t) g.dim, d.s);
            pmark(d, kPDense);
            return true;
        }
        k::rows_dot(l.gate, g.n_exp, g.dim, d.xn, 1, d.logit, d.s);
        k::route(d.logit, l.bias, 1, g.n_exp, g.k_exp, d.top, d.w, d.s);
        // the shared expert into sh (while the router's answer comes back)
        const x3::MvJob sgu[2] = {job(l.sg, d.xn, d.g), job(l.su, d.xn, d.u)};
        if (!x3::mv(sgu, 2, d.s)) {
            err = "mv shared gate/up";
            return false;
        }
        k::swigluoai(d.g, d.u, (size_t) g.ff_shared, g.alpha, g.limit, d.g, nullptr, d.s);
        const x3::MvJob sdj = job(l.sd, d.g, d.sh);
        if (!x3::mv(&sdj, 1, d.s)) {
            err = "mv shared down";
            return false;
        }
        pmark(d, kPRoute);
        if (l.hf) {   // host-free: the plan made on the device (the host's values), nothing waited for
            const size_t lh = (size_t) (&l - L.data());
            if (!k::plan_dev(d.top, d.w, 1, g.k_exp, l.exp_tab, g.scale, d.plan_d, (float*) (d.plan_d + 8),
                             d.prof_d + lh * g.n_exp, nullptr, nullptr, nullptr, nullptr, d.s)) {
                err = "plan_dev";
                return false;
            }
            st.profiled += g.k_exp;
            x3::MoeArgs a;
            a.plan_ptr = d.plan_d;
            a.plan_w = (const float*) (d.plan_d + 8);
            a.lay = l.lay;
            a.k = g.k_exp;
            a.n_embd = g.dim;
            a.n_ff = g.ff;
            a.limit = g.limit;
            x3::moe_gate_up(a, d.xn, nullptr, d.s);
            x3::moe_down(a, nullptr, d.sh, d.out, d.s);
            k::add(d.h, d.out, (size_t) g.dim, d.s);
            pmark(d, kPExperts);
            return true;
        }
        int top[8];
        float w[8];
        CK(cudaMemcpyAsync(d.plan_h, d.top, (size_t) g.k_exp * 4, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(d.plan_h + 4, d.w, (size_t) g.k_exp * 4, cudaMemcpyDeviceToHost, d.s));
        if (lane) CK(cudaMemcpyAsync(d.xh_h, d.xn, (size_t) g.dim * 4, cudaMemcpyDeviceToHost, d.s));
        const double ts = now();
        CK(cudaStreamSynchronize(d.s));
        st.t_sync += now() - ts;
        std::memcpy(top, d.plan_h, (size_t) g.k_exp * 4);
        std::memcpy(w, d.plan_h + 4, (size_t) g.k_exp * 4);
        const size_t li = (size_t) (&l - L.data());
        for (int i = 0; i < g.k_exp; ++i) ++prof[li * g.n_exp + top[i]];
        st.profiled += g.k_exp;
        std::vector<char> busy((size_t) d.n_slots, 0);
        unsigned long long* ph = d.plan_h;
        float* pw = (float*) (d.plan_h + 8);
        std::vector<CpuExpert> cpu;
        // (of the experts the CPU lane would take, lane_split's - ones in the pinned pool - to staging slots instead)
        int to_stage = 0;
        if (lane && lane_copy > 0) {
            int cand = 0;
            for (int i = 0; i < g.k_exp; ++i)
                cand += on_cpu(l, d, top[i]) && pooled(l, top[i]);
            to_stage = lane_split(cand);
        }
        for (int i = 0; i < g.k_exp; ++i) {
            pw[i] = w[i] * g.scale;
            if (lane && on_cpu(l, d, top[i]) && to_stage > 0 && pooled(l, top[i])) {
                --to_stage;   // (staged: locate below)
            } else if (lane && on_cpu(l, d, top[i])) {   // (not in VRAM nor in a staging slot: the CPU lane's)
                if (cpu_seen.empty()) cpu_seen.assign((size_t) g.n_layers * g.n_exp, 0);
                ++cpu_seen[li * g.n_exp + top[i]];
                consider_promotion(l, d, top[i]);
                ph[i] = 0;
                cpu.push_back(cpu_expert(l, top[i], pw[i]));
                continue;
            }
            if (!locate(l, d, top[i], busy, ph[i], err)) return false;
        }
        CK(cudaMemcpyAsync(d.plan_d, ph, 16 * sizeof(unsigned long long), cudaMemcpyHostToDevice, d.s));
        x3::MoeArgs a;
        a.plan_ptr = d.plan_d;
        a.plan_w = (const float*) (d.plan_d + 8);
        a.lay = l.lay;
        a.k = g.k_exp;
        a.n_embd = g.dim;
        a.n_ff = g.ff;
        a.limit = g.limit;
        x3::moe_gate_up(a, d.xn, nullptr, d.s);
        if (lane) {   // the CPU lane's part while the GPU runs the others' gate / up
            float* part = d.xh_h + g.dim;
            int flag = 0;
            if (!cpu.empty()) {
                const double tl = now();
                lane_run(cpu, d.xh_h, part);
                st.t_lane += now() - tl;
                flag = 1;
                st.cpu_experts += (long) cpu.size();
            }
            std::memcpy(part + g.dim, &flag, 4);
            CK(cudaMemcpyAsync(d.cpu_d, part, (size_t) (g.dim + 1) * 4, cudaMemcpyHostToDevice, d.s));
            a.cpu_part = d.cpu_d;
            a.cpu_flag = (const int*) (d.cpu_d + g.dim);
        }
        x3::moe_down(a, nullptr, d.sh, d.out, d.s);
        k::add(d.h, d.out, (size_t) g.dim, d.s);
        pmark(d, kPExperts);
        // (the plan's pinned copy is rewritten by the next MoE layer of this GPU after its routing sync, which
        //  waits on this one's kernels)
        return true;
    }

    // the routed experts of layer l for this chunk: top [T][K] on the host -> y rows (by the sorted rows' order) and
    // invrow; a pass over the resident experts (by resident slot), then one a batch of the staged experts (by staging
    // slot: an expert already in a slot keeps it, the others take the least recently used slots)
    bool experts(Layer& l, Dev& d, int T, const std::vector<int>& top, std::string& err) {
        const int K = g.k_exp;
        std::vector<std::vector<int>> rows_of((size_t) g.n_exp);   // expert -> its (t * K + slot) entries
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < K; ++i) rows_of[(size_t) top[(size_t) t * K + i]].push_back(t * K + i);
        std::vector<int> staged;
        for (int e = 0; e < g.n_exp; ++e)
            if (!rows_of[(size_t) e].empty() && l.res_slot[(size_t) e] < 0) staged.push_back(e);
        struct Pass {
            const uint8_t* wbase;
            size_t stride;
            std::vector<int> slot_expert;   // the pass's experts by position (-1: none)
            int row0 = 0, rows = 0, max_rows = 0;
            std::vector<int> bounds;
            bool staged = false;
        };
        std::vector<Pass> passes;
        if (!l.res_ids.empty()) passes.push_back({l.res_base, l.blob, l.res_ids});
        const long long lid = (long long) (&l - L.data());
        for (size_t si = 0; si < staged.size();) {
            std::vector<int> se((size_t) d.n_slots, -1), fresh;
            for (int n = 0; si < staged.size() && n < d.n_slots; ++si, ++n) {
                const int e = staged[si];
                int at = -1;
                for (int s = 0; s < d.n_slots; ++s)
                    if (d.slot_owner[(size_t) s] == lid * 1024 + e) at = s;
                if (at >= 0 && se[(size_t) at] < 0) se[(size_t) at] = e;
                else fresh.push_back(e);
            }
            for (size_t fi = 0; fi < fresh.size(); ++fi) {
                int best = -1;
                for (int s = 0; s < d.n_slots; ++s)
                    if (se[(size_t) s] < 0 && d.slot_pend[(size_t) s] < 0 &&
                        (best < 0 || d.slot_used[(size_t) s] < d.slot_used[(size_t) best]))
                        best = s;
                if (best < 0) {   // (every free slot pending: the rest first in the next batch)
                    staged.insert(staged.begin() + (long) si, fresh.begin() + (long) fi, fresh.end());
                    break;
                }
                se[(size_t) best] = fresh[fi];
            }
            Pass p{d.stg, d.slot_bytes, std::move(se)};
            p.staged = true;
            passes.push_back(std::move(p));
        }
        std::vector<int> rowidx, invrow((size_t) T * K, -1);
        rowidx.reserve((size_t) T * K);
        for (Pass& p : passes) {
            p.row0 = (int) rowidx.size();
            for (int e : p.slot_expert) {
                p.bounds.push_back((int) rowidx.size());
                if (e < 0) continue;
                for (int ent : rows_of[(size_t) e]) {
                    invrow[(size_t) ent] = (int) rowidx.size();
                    rowidx.push_back(ent / K);
                }
                p.max_rows = std::max(p.max_rows, (int) rows_of[(size_t) e].size());
            }
            p.bounds.push_back((int) rowidx.size());
            p.rows = (int) rowidx.size() - p.row0;
        }
        if ((int) rowidx.size() != T * K) {
            err = "internal: routed rows " + std::to_string(rowidx.size());
            return false;
        }
        CK(cudaMemcpyAsync(d.rowidx, rowidx.data(), rowidx.size() * 4, cudaMemcpyHostToDevice, d.s));
        CK(cudaMemcpyAsync(d.invrow, invrow.data(), invrow.size() * 4, cudaMemcpyHostToDevice, d.s));
        for (const Pass& p : passes) {
            if (p.rows == 0) continue;
            if (p.staged) {
                for (int s = 0; s < d.n_slots; ++s) {
                    const int e = p.slot_expert[(size_t) s];
                    if (e < 0) continue;
                    if (d.slot_owner[(size_t) s] != lid * 1024 + e) {
                        if (!stage(l, d, e, s, err)) return false;
                        d.slot_owner[(size_t) s] = lid * 1024 + e;
                    }
                    d.slot_used[(size_t) s] = fwd;
                }
            }
            CK(cudaMemcpyAsync(d.bounds, p.bounds.data(), p.bounds.size() * 4, cudaMemcpyHostToDevice, d.s));
            x3::SetArgs sa;
            sa.wbase = p.wbase;
            sa.stride = p.stride;
            sa.lay = l.lay;
            sa.n_exp = (int) p.slot_expert.size();
            sa.bounds = d.bounds;
            sa.row0 = p.row0;
            sa.max_rows = p.max_rows;
            sa.n_embd = g.dim;
            sa.n_ff = g.ff;
            sa.limit = g.limit;
            sa.hbounds = p.bounds.data();   // (before sm_80: the busy experts through the FP16 GEMM)
            if (gguf) {   // (the rows also as FP16 - rms' xn16 - and the gate's sums of a layer whose gate and up differ
                          //  in type in this pass's own y rows, which its down writes only after them)
                sa.x16 = d.xn16;
                sa.graw = d.y + (size_t) p.row0 * g.dim;
            }
            x3::set_gate_up(sa, d.xn, g.dim, d.rowidx + p.row0, d.h16, g.ff, d.s);
            x3::set_down(sa, d.h16, g.ff, d.y + (size_t) p.row0 * g.dim, g.dim, d.s);
            // (the next pass rewrites the bounds and h16: let this one finish with them)
            CK(cudaStreamSynchronize(d.s));
        }
        return true;
    }

    bool layer(int li, int T, int pmax, std::string& err) {
        Layer& l = L[(size_t) li];
        Dev& d = D[(size_t) l.dev];
        const int di = l.dev;
        if (T == 1 && o.fast_decode) return layer_token(l, d, di, err);
        if (T <= kWin && o.fast_decode) return layer_window(l, d, di, T, err);
        // h += x W: the BLAS's beta 1; with hipBLASLt (beta 0 products only) the product into ao, then the add - the
        // same sum of the product and h
        const auto gemm_acc = [&](const x3::Mat& m, const uint16_t* x16, int ldx) {
            if (gguf && gg::gemm_backend() == 1) {
                x3::gemm_rows(m, x16, ldx, T, d.ao, g.dim, 0.0f, d.s);
                k::add(d.h, d.ao, (size_t) T * g.dim, d.s);
            } else {
                x3::gemm_rows(m, x16, ldx, T, d.h, g.dim, 1.0f, d.s);
            }
        };
        // attention
        k::rms(d.h, l.in1, T, g.dim, g.eps, d.xn, d.xn16, d.s);
        x3::gemm_rows(l.q, d.xn16, g.dim, T, d.q, g.n_head * g.hd, 0.0f, d.s);
        x3::gemm_rows(l.k, d.xn16, g.dim, T, d.k, g.n_kv * g.hd, 0.0f, d.s);
        x3::gemm_rows(l.v, d.xn16, g.dim, T, d.v, g.n_kv * g.hd, 0.0f, d.s);
        k::qk_norm_rope(d.q, d.k, d.v, l.qn1, l.kn1, d.pos, T, g.n_head, g.n_kv, g.hd, g.n_rot, inv_dev[(size_t) di],
                        g.eps, l.kc, l.vc, d.s, 0, l.kvq, l.kvh);
        pmark(d, kPAttnIn);
        const float one = 1.0f, zero = 0.0f;
        const int nq = g.ix_heads * g.ix_dim;
        // (Q8_0 projections: each decoded into wtmp first - the F32 values of an F32 upload, the same GEMM)
        if (l.wiq8 != nullptr) k::q8_0_f32(l.wiq8, (size_t) nq * g.dim, d.wtmp, d.s);
        CB(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, nq, T, g.dim, &one, l.wiq8 ? d.wtmp : l.wiq, g.dim, d.xn, g.dim,
                       &zero, d.iq, nq));
        if (l.wik8 != nullptr) k::q8_0_f32(l.wik8, (size_t) g.ix_dim * g.dim, d.wtmp, d.s);
        CB(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, g.ix_dim, T, g.dim, &one, l.wik8 ? d.wtmp : l.wik, g.dim, d.xn, g.dim, &zero,
                       d.ik, g.ix_dim));
        k::index_norm_rope(d.iq, d.ik, l.iqn1, l.ikn1, d.pos, T, g.ix_heads, g.ix_dim, g.n_rot, inv_dev[(size_t) di],
                           g.eps, l.ic, d.s);
        pmark(d, kPIxProj);
        const int nblk = pmax / g.blk + 1;
        k::block_scores(d.iq, l.ic, d.pos, T, g.ix_heads, nblk, g.blk, d.bs, d.s);
        pmark(d, kPIndex);
        k::select_blocks(d.bs, d.pos, T, g.ix_heads, nblk, g.blk, g.topk, d.kept, d.s);
        pmark(d, kPIxSel);
        if (const char* dp = ::mm3::env("MM3_DUMP_KEPT")) {   // (debugging: the kept blocks, int32 records)
            std::vector<int> kh((size_t) g.ix_heads * T * g.topk);
            CK(cudaMemcpyAsync(kh.data(), d.kept, kh.size() * 4, cudaMemcpyDeviceToHost, d.s));
            CK(cudaStreamSynchronize(d.s));
            if (std::FILE* f = std::fopen(dp, li == 0 && cur_pos - T + 1 == 0 ? "wb" : "ab")) {
                const int hdr[5] = {li, cur_pos - T + 1, T, g.ix_heads, g.topk};
                std::fwrite(hdr, 4, 5, f);
                std::fwrite(kh.data(), 4, kh.size(), f);
                std::fclose(f);
            }
        }
        k::sparse_attn(d.q, l.kc, l.vc, d.kept, d.pos, T, g.n_head, g.n_kv, g.topk, g.blk, 1.0f / std::sqrt((float) g.hd),
                       d.att, d.s, l.kvq, l.kvh);
        pmark(d, kPAttn);
        k::to_f16(d.att, T, g.n_head * g.hd, g.n_head * g.hd, d.att16, g.n_head * g.hd, d.s);
        gemm_acc(l.o, d.att16, g.n_head * g.hd);   // h += attn Wo
        pmark(d, kPOGemm);
        // MLP
        k::rms(d.h, l.post1, T, g.dim, g.eps, d.xn, d.xn16, d.s);
        pmark(d, kPOut);
        if (!l.moe) {
            x3::gemm_rows(l.mg, d.xn16, g.dim, T, d.g, g.ff_dense, 0.0f, d.s);
            x3::gemm_rows(l.mu, d.xn16, g.dim, T, d.u, g.ff_dense, 0.0f, d.s);
            k::swigluoai(d.g, d.u, (size_t) T * g.ff_dense, g.alpha, g.limit, nullptr, d.h16, d.s);
            gemm_acc(l.md, d.h16, g.ff_dense);   // h += mlp
            pmark(d, kPDense);
            return true;
        }
        CB(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, g.n_exp, T, g.dim, &one, l.gate, g.dim, d.xn, g.dim, &zero,
                       d.logit, g.n_exp));
        k::route(d.logit, l.bias, T, g.n_exp, g.k_exp, d.top, d.w, d.s);
        pmark(d, kPRouter);
        // the shared expert into sh
        x3::gemm_rows(l.sg, d.xn16, g.dim, T, d.g, g.ff_shared, 0.0f, d.s);
        x3::gemm_rows(l.su, d.xn16, g.dim, T, d.u, g.ff_shared, 0.0f, d.s);
        k::swigluoai(d.g, d.u, (size_t) T * g.ff_shared, g.alpha, g.limit, nullptr, d.h16, d.s);
        x3::gemm_rows(l.sd, d.h16, g.ff_shared, T, d.sh, g.dim, 0.0f, d.s);
        pmark(d, kPRoute);
        CK(cudaMemcpyAsync(d.top_h, d.top, (size_t) T * g.k_exp * 4, cudaMemcpyDeviceToHost, d.s));
        CK(cudaStreamSynchronize(d.s));
        const std::vector<int> top(d.top_h, d.top_h + (size_t) T * g.k_exp);
        for (int e : top) ++prof[(size_t) li * g.n_exp + e];
        st.profiled += (long) top.size();
        pmark(d, kPSync);
        if (!experts(l, d, T, top, err)) return false;
        k::moe_combine(d.h, d.y, d.invrow, d.w, d.sh, T, g.k_exp, g.dim, g.scale, d.out, d.s);
        std::swap(d.h, d.out);
        pmark(d, kPExperts);
        return true;
    }

    // ---- graphs: a token's or a verify window's whole forward (its layers and the head's GEMV) captured once as a
    // CUDA / HIP graph and replayed - the same kernels with the same arguments, so the same bits, without a launch a
    // kernel from the host (Windows + AMD: ~2000 launches a token, each a host-side cost that left the GPU waiting).
    // Only where nothing in the forward needs the host: one GPU, every MoE layer host-free, the token kernels.  A
    // graph is keyed by its tokens, its head rows, the indexer's block count (the one argument that moves with the
    // position: every 128 tokens - a new capture then) and the residual stream's buffer (a prompt's MoE layers swap h
    // and out: a graph captured before a prompt would read the other one after it).  MAYA_MM3_GRAPH=0: no graphs
    struct GraphEntry {
        cudaGraphExec_t exec = nullptr;
        long profiled = 0;   // (st.profiled's count of one replay: the capture's)
    };
    std::map<std::tuple<int, int, int, uintptr_t>, GraphEntry> graphs;
    bool graph_on = false;   // (load: the conditions above, and the setting)
    bool graph_ready(int T, int want) const {
        return graph_on && T <= kWin && o.fast_decode && (want == 0 || want <= kWin) && !pf.on;
    }
    void graphs_clear() {
        for (auto& kv : graphs)
            if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
        graphs.clear();
    }
    template <class F>
    bool run_graph(int T, int want, F& body, std::string& err) {
        Dev& d = D[(size_t) 0];
        const int nblk = cur_pos / g.blk + 1;
        const auto key = std::make_tuple(T, want, nblk, (uintptr_t) d.h);
        auto it = graphs.find(key);
        if (it == graphs.end()) {
            // (the graphs of other block counts are behind the position now: dropped)
            for (auto g2 = graphs.begin(); g2 != graphs.end();) {
                if (std::get<2>(g2->first) != nblk) {
                    cudaGraphExecDestroy(g2->second.exec);
                    g2 = graphs.erase(g2);
                } else {
                    ++g2;
                }
            }
            const long p0 = st.profiled;
            cudaGraph_t gr = nullptr;
            cudaGraphExec_t ex = nullptr;
            bool ok = cudaStreamBeginCapture(d.s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
            const bool ran = ok && body(true);
            const cudaError_t ee = cudaStreamEndCapture(d.s, &gr);   // (always: the stream leaves capture)
            ok = ok && ran && ee == cudaSuccess && gr != nullptr && cudaGraphInstantiate(&ex, gr, 0) == cudaSuccess;
            if (gr) cudaGraphDestroy(gr);
            const long added = st.profiled - p0;
            st.profiled = p0;
            if (!ok) {   // (nothing of the capture ran: the forward the usual way, and no graphs from now on)
                if (ex) cudaGraphExecDestroy(ex);
                cudaGetLastError();
                graph_on = false;
                info.push_back("NOTE: graphs off: a forward did not capture (" + (err.empty() ? std::string("CUDA") : err) +
                               ") - the kernels launched one by one");
                std::fprintf(stderr, "maya-mm3: %s\n", info.back().c_str());
                err.clear();
                return body(false);
            }
            it = graphs.emplace(key, GraphEntry{ex, added}).first;
        }
        CK(cudaGraphLaunch(it->second.exec, d.s));
        st.profiled += it->second.profiled;
        return true;
    }

    bool chunk(const int* ids, int T, int pos0, int want, float* logits, std::string& err) {
        const double tc = now();
        const bool ok = chunk_(ids, T, pos0, want, logits, err);
        if (T == 1) st.t_token += now() - tc;
        if (pf.on) {
            if (ok) pcollect(T == 1 ? 0 : T <= kWin && o.fast_decode ? 1 : 2, T, now() - tc);
            pf.used = 0;
        }
        return ok;
    }

    bool chunk_(const int* ids, int T, int pos0, int want, float* logits, std::string& err) {
        ++fwd;
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            refresh_slots(d);
        }
        cur_pos = pos0 + T - 1;
        if (pos0 + T > o.max_ctx) {
            err = "the context is full (" + std::to_string(o.max_ctx) + ")";
            return false;
        }
        // the embeddings (rows from the file: BF16, or a GGUF's block type dequantized as ggml does) -> F32 on the first
        // layer's GPU
        const Tensor* emb = pk.get(P + "embed_tokens.weight");
        float* h = D[(size_t) L[0].dev].emb_h;   // (pinned: the previous forward's copy of it is done - it waited)
        const uint16_t* e16 = (const uint16_t*) pk.data(*emb);
        for (int t = 0; t < T; ++t) {
            if (ids[t] < 0 || ids[t] >= g.vocab) {
                err = "token id " + std::to_string(ids[t]) + " outside the vocabulary";
                return false;
            }
            if (emb->dtype != "BF16") {
                if (!gg::host_row(emb->dtype, pk.data(*emb), g.dim, ids[t], &h[(size_t) t * g.dim])) {
                    err = "the token embedding's type " + emb->dtype + " is not read here";
                    return false;
                }
                continue;
            }
            const uint16_t* r = e16 + (size_t) ids[t] * g.dim;
            for (int i = 0; i < g.dim; ++i) {
                const uint32_t u = (uint32_t) r[i] << 16;
                std::memcpy(&h[(size_t) t * g.dim + i], &u, 4);
            }
        }
        for (const Model::Image& im : imgs) {   // (a picture's image tokens: its rows)
            const int n = (int) (im.rows.size() / (size_t) g.dim);
            for (int q = std::max(im.start, pos0); q < std::min(im.start + n, pos0 + T); ++q)
                std::memcpy(&h[(size_t) (q - pos0) * g.dim], im.rows.data() + (size_t) (q - im.start) * g.dim,
                            (size_t) g.dim * 4);
        }
        for (Dev& d : D) {
            for (int t = 0; t < T; ++t) d.pos_h[t] = pos0 + t;
            CK(cudaSetDevice(d.id));
            CK(cudaMemcpyAsync(d.pos, d.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, d.s));
        }
        int cur = L[0].dev;
        CK(cudaSetDevice(D[(size_t) cur].id));
        CK(cudaMemcpyAsync(D[(size_t) cur].h, h, (size_t) T * g.dim * 4, cudaMemcpyHostToDevice, D[(size_t) cur].s));
        pmark(D[(size_t) cur], kPAttnIn);   // (MAYA_MM3_PROF: the forward's start)
        // a token's or a verify window's rows through the head's GEMV (the same bits for each row), else the prompt's
        // head below
        const bool fast_head = want > 0 && want <= kWin && o.fast_decode;
        // the layers (and a fast head's GEMV): enqueued as they come, or (cap) captured into a graph
        auto body = [&](bool cap) -> bool {
            for (int li = 0; li < g.n_layers; ++li) {
                const int dv = L[(size_t) li].dev;
                if (dv != cur) {
                    Dev &a = D[(size_t) cur], &b = D[(size_t) dv];
                    CK(cudaSetDevice(a.id));
                    CK(cudaStreamSynchronize(a.s));
                    CK(cudaSetDevice(b.id));
                    CK(cudaMemcpyPeerAsync(b.h, b.id, a.h, a.id, (size_t) T * g.dim * 4, b.s));
                    cur = dv;
                }
                CK(cudaSetDevice(D[(size_t) dv].id));
                if (!layer(li, T, pos0 + T - 1, err)) return false;
                // (a token's or a window's layer: Windows + AMD start its queued work now - a host-free layer has no
                //  wait of its own that would, so the GPU runs this layer while the host enqueues the next)
                if (!cap && T <= kWin && o.fast_decode) k::submit(D[(size_t) dv].s);
                if (draftm && tap_of[(size_t) li] >= 0) {   // (the draft's context: this layer's output rows)
                    Dev& a = D[(size_t) dv];
                    Dev& hd = D[(size_t) head_dev];
                    float* dst = taps + (size_t) tap_of[(size_t) li] * T * g.dim;
                    if (dv == head_dev)
                        CK(cudaMemcpyAsync(dst, a.h, (size_t) T * g.dim * 4, cudaMemcpyDeviceToDevice, a.s));
                    else
                        CK(cudaMemcpyPeerAsync(dst, hd.id, a.h, a.id, (size_t) T * g.dim * 4, a.s));
                }
                if (const char* dp = ::mm3::env("MM3_DUMP_HS")) {   // (debugging: every layer's residual stream, F32 rows)
                    std::vector<float> hh((size_t) T * g.dim);
                    CK(cudaMemcpyAsync(hh.data(), D[(size_t) dv].h, hh.size() * 4, cudaMemcpyDeviceToHost,
                                       D[(size_t) dv].s));
                    CK(cudaStreamSynchronize(D[(size_t) dv].s));
                    if (std::FILE* f = std::fopen(dp, li == 0 && pos0 == 0 ? "wb" : "ab")) {
                        std::fwrite(hh.data(), 4, hh.size(), f);
                        std::fclose(f);
                    }
                }
                if (o.verbose) {
                    CK(cudaStreamSynchronize(D[(size_t) dv].s));
                    const int e = x3::launch_errors();
                    if (e) {
                        err = "EXL3 launch errors: " + std::to_string(e);
                        return false;
                    }
                }
            }
            if (want <= 0) return true;
            Dev& a = D[(size_t) cur];
            Dev& hd = D[(size_t) head_dev];
            if (cur != head_dev) {   // (the last layer on another GPU: its rows across; on the head's GPU stream order holds)
                CK(cudaSetDevice(a.id));
                CK(cudaStreamSynchronize(a.s));
                CK(cudaSetDevice(hd.id));
                CK(cudaMemcpyPeerAsync(hd.h, hd.id, a.h, a.id, (size_t) T * g.dim * 4, hd.s));
            }
            CK(cudaSetDevice(hd.id));
            const float* last = hd.h + (size_t) (T - want) * g.dim;
            k::rms(last, norm1, want, g.dim, g.eps, hd.xn, hd.xn16, hd.s);
            if (fast_head) {
                const int cols = std::max(1, x3::mv_cols(head));   // (the split partials' scratch: ~5 rows a pass)
                for (int c0 = 0; c0 < want; c0 += cols) {
                    x3::MvJob j;
                    j.m = &head;
                    j.xf = hd.xn + (size_t) c0 * g.dim;
                    j.y = hd.lg + (size_t) c0 * g.vocab;
                    j.ncol = std::min(cols, want - c0);
                    j.xf_cs = (size_t) g.dim;
                    j.y_cs = (size_t) g.vocab;
                    if (!x3::mv(&j, 1, hd.s)) {
                        err = "mv head";
                        return false;
                    }
                }
            }
            return true;
        };
        if (!(graph_ready(T, want) ? run_graph(T, want, body, err) : body(false))) return false;
        if (want > 0) {
            Dev& hd = D[(size_t) head_dev];
            CK(cudaSetDevice(hd.id));
            if (fast_head) {
                pmark(hd, kPHead);
                CK(cudaMemcpyAsync(hd.lg_h, hd.lg, (size_t) want * g.vocab * 4, cudaMemcpyDeviceToHost, hd.s));
                CK(cudaStreamSynchronize(hd.s));
                std::memcpy(logits, hd.lg_h, (size_t) want * g.vocab * 4);
                want = 0;
            }
            // the head's output kHeadRows rows at a time
            for (int r0 = 0; r0 < want; r0 += kHeadRows) {
                const int nr = std::min<int>(kHeadRows, want - r0);
                x3::gemm_rows(head, hd.xn16 + (size_t) r0 * g.dim, g.dim, nr, hd.lg, g.vocab, 0.0f, hd.s);
                CK(cudaMemcpyAsync(hd.lg_h, hd.lg, (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, hd.s));
                CK(cudaStreamSynchronize(hd.s));
                std::memcpy(logits + (size_t) r0 * g.vocab, hd.lg_h, (size_t) nr * g.vocab * 4);
            }
            if (want > 0) pmark(hd, kPHead);
        }
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            CK(cudaStreamSynchronize(d.s));
        }
        const int e = x3::launch_errors();
        if (e) {
            err = "EXL3 launch errors: " + std::to_string(e);
            return false;
        }
        for (Dev& d : D)
            if (!d.promo_q.empty()) {
                CK(cudaSetDevice(d.id));
                issue_promotions(d);
            }
        if (draftm) {
            CK(cudaSetDevice(D[(size_t) head_dev].id));
            if (!draftm->context(taps, T, pos0, err)) return false;
            pmark(D[(size_t) head_dev], kPDraft);
        }
        return true;
    }
};

Model::Model() : p(new Impl) {}
Model::~Model() {
    delete tpe;
    delete p;
}
bool Model::load(const Options& o_in, std::string& err) {
    Options o = o_in;
    if (o.gpus.empty()) {   // (no --gpus: the GPUs this engine runs on - compute capability 7.0 or newer; a display card
                            //  beside them, e.g. a Quadro P400, is left out with a NOTE)
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
            err = "no CUDA device";
            return false;
        }
        std::vector<std::string> skipped;
        for (int i = 0; i < n; ++i) {
            cudaDeviceProp pr{};
            if (cudaGetDeviceProperties(&pr, i) != cudaSuccess) continue;
            if (pr.major * 10 + pr.minor >= 70) o.gpus.push_back(i);
            else skipped.push_back("CUDA" + std::to_string(i) + " (" + pr.name + ", sm_" + std::to_string(pr.major) +
                                   std::to_string(pr.minor) + ")");
        }
        if (o.gpus.empty()) {
            err = "no GPU of compute capability 7.0 or newer (MiniMax-M3.1's engine needs a V100 or newer)";
            return false;
        }
        for (const auto& s : skipped)
            std::fprintf(stderr, "maya-mm3: NOTE: %s left out - compute capability below 7.0 (--gpus to choose)\n",
                         s.c_str());
    }
    // the prefill calibration of this PC (maya-mm3 --calibrate-prefill, the setup's calibration step): its chunk and
    // GEMM backend, where --chunk / --prefill-gemm do not choose; none: the engine's defaults, with a NOTE
    p->tkey = tune_key_of(o.gpus);
    p->tfile = o.prefill_tune.empty() ? o.pack + "/prefill-tune.txt" : o.prefill_tune;
    std::string tuned;
    if (o.tune && o.tp <= 0) {
        std::ifstream f(p->tfile);
        for (std::string line; std::getline(f, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t tab = line.find('\t');
            if (tab == std::string::npos || line.compare(0, tab, p->tkey) != 0) continue;
            std::istringstream is(line.substr(tab + 1));
            for (std::string kv; is >> kv;) {
                const size_t eq = kv.find('=');
                if (eq == std::string::npos) continue;
                const std::string k = kv.substr(0, eq);
                const int v = std::atoi(kv.c_str() + eq + 1);
                if (k == "chunk" && !o.chunk_given && v >= 64 && v <= 16384) o.chunk = v;
                else if (k == "gemm" && o.prefill_gemm < 0) o.prefill_gemm = v;
            }
            tuned = line.substr(tab + 1);
        }
    }
    if (o.tp > 0) {   // (a GGUF pack: the layer split - the tensor-parallel path splits EXL3 matrices only)
        bool gg_pack = false;
        std::ifstream f(o.pack + "/mm3.txt");
        for (std::string line; std::getline(f, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            gg_pack = gg_pack || line == "format gguf";
        }
        if (gg_pack) {
            Options o2 = o;
            o2.tp = 0;
            const bool ok = p->load(o2, err);
            p->info.insert(p->info.begin(), "NOTE: --tp " + std::to_string(o.tp) + ": not with a GGUF pack - the layer "
                                            "split instead");
            return ok;
        }
    }
#if defined(_WIN32)
    // Windows: the tensor-parallel path's host waits on the GPUs' counters depend on its launches being submitted as
    // they are enqueued, which the WDDM driver batches - the layer split there (MAYA_MM3_TP_WINDOWS=1: try it anyway)
    if (o.tp > 0 && !(std::getenv("MAYA_MM3_TP_WINDOWS") && std::atoi(std::getenv("MAYA_MM3_TP_WINDOWS")) != 0)) {
        Options o2 = o;
        o2.tp = 0;
        const bool ok = p->load(o2, err);
        p->info.insert(p->info.begin(), "--tp " + std::to_string(o.tp) + ": not on Windows - the layer split instead");
        return ok;
    }
#endif
    if (o.tp > 0) {
        tpe = new TpEngine;
        if (tpe->load(o, err)) return true;
        // (the tensor-parallel path does not fit this PC - its memory, its GPUs: the layer split instead, which sizes
        //  itself to the memory, with a NOTE; its own error when that does not load either)
        const std::string why = err;
        delete tpe;   // (frees its GPU memory and pinned pool first)
        tpe = nullptr;
        Options o2 = o;
        o2.tp = 0;
        err.clear();
        std::fprintf(stderr, "maya-mm3: --tp %d did not load (%s): the layer split instead\n", o.tp, why.c_str());
        if (!p->load(o2, err)) {
            err = "--tp " + std::to_string(o.tp) + ": " + why + "; the layer split: " + err;
            return false;
        }
        p->info.insert(p->info.begin(), "NOTE: --tp " + std::to_string(o.tp) + " did not fit (" + why +
                                            "): the layer split instead");
        return true;
    }
    if (!p->load(o, err)) return false;
    if (!tuned.empty())
        p->info.push_back("prefill: this PC's calibration (" + p->tfile + "): chunk " + std::to_string(p->o.chunk) +
                          (p->gguf ? std::string(", ") + (gg::gemm_backend() == 1 ? "hipBLASLt" : "BLAS") + " GEMMs" : ""));
    else if (o.tune && o.tp <= 0)
        p->info.push_back("NOTE: prefill not calibrated on this PC (./maya.sh --calibrate measures it): chunk " +
                          std::to_string(p->o.chunk));
    return true;
}
int Model::chunk() const { return tpe ? 0 : p->run_chunk(); }
int Model::chunk_max() const { return tpe ? 0 : p->o.chunk; }
void Model::set_chunk(int c) { p->chunk_run = c; }
void Model::set_prefill_gemm(int b) { gg::set_gemm(b); }
int Model::prefill_gemm() const { return gg::gemm_backend(); }
bool Model::lt_available() const { return !tpe && p->gguf && gg::gemm_lt_available(); }
std::string Model::tune_key() const { return p->tkey; }
std::string Model::tune_file() const { return p->tfile; }
bool Model::write_tune(const std::string& settings, std::string& err) const {
    std::vector<std::string> keep;
    {
        std::ifstream f(p->tfile);
        for (std::string line; std::getline(f, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t tab = line.find('\t');
            if (!line.empty() && (tab == std::string::npos || line.compare(0, tab, p->tkey) != 0)) keep.push_back(line);
        }
    }
    keep.push_back(p->tkey + "\t" + settings);
    const std::string tmp = p->tfile + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        for (const auto& l : keep) f << l << "\n";
        if (!f) {
            err = tmp + ": cannot write";
            return false;
        }
    }
    if (!strata::ds4::os::replace_file(tmp, p->tfile)) {
        err = p->tfile + ": cannot replace";
        return false;
    }
    return true;
}
int Model::vocab() const { return tpe ? tpe->vocab() : p->g.vocab; }
int Model::max_ctx() const { return tpe ? tpe->max_ctx() : p->o.max_ctx; }
int Model::bos() const { return tpe ? tpe->bos() : p->g.bos; }
int Model::eos() const { return tpe ? tpe->eos() : p->g.eos; }
const Stats& Model::stats() const { return tpe ? tpe->stats() : p->st; }
std::vector<std::string> Model::info_lines() const { return tpe ? tpe->info_lines() : p->info; }
std::vector<std::string> Model::vram_lines() const { return tpe ? tpe->vram_lines() : p->vram; }

int Model::draft_block() const {
    if (tpe) return tpe->draft_block();
    return p->draftm ? p->draftm->geo().block : 0;
}

const VisionGeo* Model::vision() const {
    if (tpe) return tpe->vision();
    return p->vis ? &p->vis->geo() : nullptr;
}

bool Model::encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err) {
    if (tpe) return tpe->encode_image(rgb, width, height, rows, err);
    if (!p->vis) {
        err = "no vision tower loaded (--vision on)";
        return false;
    }
    return p->vis->encode(rgb, width, height, rows, err);
}

void Model::set_images(std::vector<Image> images) {
    if (tpe) tpe->set_images(std::move(images));
    else p->imgs = std::move(images);
}

bool Model::draft(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, std::string& err) {
    if (tpe) return tpe->draft(anchor, pos, n, drafts, conf, err);
    if (!p->draftm) {
        err = "no DSpark draft loaded (--dspark)";
        return false;
    }
    cudaSetDevice(p->D[(size_t) p->head_dev].id);
    return p->draftm->propose(anchor, pos, n, drafts, conf, nullptr, err);
}

bool Model::draft_sampled(int anchor, int pos, int n, const DraftSampling& ds, const std::vector<float>& u,
                          std::vector<int>& drafts, std::vector<float>& conf,
                          std::vector<std::vector<std::pair<int, float>>>& q, std::string& err) {
    if (tpe) return tpe->draft_sampled(anchor, pos, n, ds, u, drafts, conf, q, err);
    if (!p->draftm) {
        err = "no DSpark draft loaded (--dspark)";
        return false;
    }
    k::SampleArgs sa;
    sa.temperature = ds.temperature;
    sa.top_p = ds.top_p;
    sa.min_p = ds.min_p;
    sa.top_k = ds.top_k;
    cudaSetDevice(p->D[(size_t) p->head_dev].id);
    return p->draftm->propose_sampled(anchor, pos, n, sa, u, drafts, conf, q, err);
}

bool Model::save_profile(const std::string& path, std::string& err) const {
    if (tpe) return tpe->save_profile(path, err);
    // (the host-free layers' routings were counted on their GPUs: added in)
    std::vector<uint32_t> all = p->prof, dc(all.size());
    for (const auto& d : p->D) {
        if (d.prof_d == nullptr) continue;
        cudaSetDevice(d.id);
        if (cudaMemcpy(dc.data(), d.prof_d, dc.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError();
            err = "the expert profile's device counts not read";
            return false;
        }
        for (size_t i = 0; i < all.size(); ++i) all[i] += dc[i];
    }
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(all.data(), 4, all.size(), f) != all.size()) {
        if (f) std::fclose(f);
        err = path + ": cannot write";
        return false;
    }
    std::fclose(f);
    return true;
}

bool Model::forward(const int* ids, int n, int pos0, int want, float* logits_out, std::string& err) {
    if (tpe) return tpe->forward(ids, n, pos0, want, logits_out, err);
    p->last_done = 0;
    want = std::min(want, n);
    const int C = p->run_chunk();
    for (int a = 0; a < n; a += C) {
        const int T = std::min(C, n - a);
        // the logits this chunk owns: the call's last `want` positions that fall in it
        const int first = n - want;
        const int w0 = std::max(first, a), w1 = a + T;
        const int w = logits_out && w1 > w0 ? w1 - w0 : 0;
        if (w > 0 && w0 != a + T - w) {
            err = "internal: logits rows";
            return false;
        }
        if (!p->chunk(ids + a, T, pos0 + a, w, w > 0 ? logits_out + (size_t) (w0 - first) * p->g.vocab : nullptr,
                      err))
            return false;
    }
    p->last_done = n;
    return true;
}

void Model::set_progress(std::function<bool(int)> f) {
    if (tpe) tpe->set_progress(std::move(f));
}
int Model::done() const { return tpe ? tpe->done() : p->last_done; }
bool Model::pipelined() const { return tpe != nullptr; }
bool Model::forward_ppl(const int* ids, int n, std::string& err) {
    if (!tpe) {
        err = "forward_ppl: the tensor-parallel path's (--tp); the layer-split path's --ppl takes its logits";
        return false;
    }
    tpe->set_ppl(true);
    const bool ok = tpe->forward(ids, n, 0, 0, nullptr, err);
    tpe->set_ppl(false);
    return ok;
}
const std::vector<double>& Model::ppl_nll() const {
    static const std::vector<double> none;
    return tpe ? tpe->ppl_nll() : none;
}

// (the layer-split path: the logits on the host, the candidates from them)
bool Model::forward_cands(const int* ids, int n, int pos0, int want, Cands& c, std::string& err) {
    if (tpe) return tpe->forward_cands(ids, n, pos0, want, c, err);
    want = std::max(0, std::min(want, n));
    std::vector<float> lg((size_t) want * vocab());
    if (!forward(ids, n, pos0, want, lg.data(), err)) return false;
    c.top.assign((size_t) want, {});
    c.ex.assign((size_t) want, {});
    for (int r = 0; r < want; ++r)
        host_cands(lg.data() + (size_t) r * vocab(), vocab(), c.kc, c.extra, c.top[(size_t) r], c.ex[(size_t) r]);
    return true;
}

}  // namespace mm3
