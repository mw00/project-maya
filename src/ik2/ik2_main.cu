// src/ik2/ik2_main.cu - maya-ik2: Inkling 2 (thinkingmachines/Inkling-2-Small, 594B / ~17B active) on one or two
// GPUs, from an `inkling2` GGUF in llama.cpp PR #32187's layout (docs/INKLING2.md; unsloth/Inkling-2-Small-GGUF's
// files: NVFP4, IQ3_XXS / IQ4_XS and Q*_K experts).
//
//   * The dense weights (attention, router, head) decoded once to FP16 in VRAM (the router in F32); cuBLAS' FP16 GEMMs
//     with F32 sums.  The attention (relative bias, sinks, sliding window, L2 clips) fused: an mma.sync kernel, and a
//     wgmma one on Hopper (sm_90a); the KV cache in FP16 (IK2_KV=fp8: E4M3).
//   * The routed experts as the GGUF holds them, all in pinned host memory (~323 GiB for UD-Q4_K_XL), and a cache of
//     them in VRAM (--cache-gb, at most what the GPU has room for).  A prompt goes through layer by layer in passes of
//     up to 16384 tokens, so each pass streams every expert it misses once, the next layer's prefetched while this one
//     runs; a scan-resistant eviction keeps a pass from flushing the cache.  IK2_WARM fills the cache at load with the
//     experts tokens pick most (a profile written with IK2_POP_OUT).
//   * Decode a token at a time; with IK2_CPU=1 a missing expert is computed on the CPU (AVX-512) when that is sooner
//     than its copy.  Two GPUs (--gpus 2): IK2_EP=1 deals each layer's experts to both (expert parallel - each with its
//     own cache and PCIe link), else the layers are split between them.
//
//   maya-ik2 --model FIRST-SHARD.gguf --serve [--max-context 262144] [--cache-gb 60] [--gpus 1] [--threads 8]
//            [--eos a,b]
//   maya-ik2 --model FIRST-SHARD.gguf --tokens 1,2,3|@FILE [--gen N] [--chunk 2048] [--dump DIR]
// --serve: Maya's engine protocol on stdin / stdout (serve/server.py's StrataEngine: INFO / READY, GEN, PP, T, DONE,
// STOP); --serve-ik2 the older @-line one.  --dump: each layer's output (blk.L.out) and the log-probabilities of every
// prompt position, F32 raw files ([T][C], [T][n_vocab]) - what tools/ik2_compare.py holds against ref/inkling2.py's
// seams.  Settings (IK2_*): docs/SETTINGS.md.
#include "ik2_cpu.hpp"
#include "ik2_draft.hpp"
#include "ik2_kernels.cuh"
#include "ik2_lane.hpp"
#include "ik2_off.cuh"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/spec/suffix_drafter.hpp"
#ifdef STRATA_USE_HIP
#include "ds4_lt.hpp"   // (AMD: the prompt's FP16 GEMMs on hipBLASLt, the kernel a shape measured and kept)
#endif
#include "strata/kernels/cpu/pool.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>
#define O_RDONLY (_O_RDONLY | _O_BINARY)
#define open _open
#define close _close
typedef SSIZE_T ssize_t;
#define off_t int64_t
static inline ssize_t pread(int fd, void* buf, size_t count, int64_t offset) {
    if (_lseeki64(fd, offset, SEEK_SET) == -1) return -1;
    return _read(fd, buf, (unsigned int)count);
}
#define posix_fadvise(fd, off, len, adv) ((void)0)
#define MADV_WILLNEED 0
static inline int madvise(void*, size_t, int) { return 0; }
static inline int munmap(void*, size_t) { return 0; }
#ifndef SIGUSR1
#define SIGUSR1 30
#endif
#else
#include <sys/mman.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <csignal>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

namespace k = ik2::k;

#define CU(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "maya-ik2: %s: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)
#define CB(x)                                                                                              \
    do {                                                                                                   \
        cublasStatus_t e_ = (x);                                                                           \
        if (e_ != CUBLAS_STATUS_SUCCESS) {                                                                 \
            std::fprintf(stderr, "maya-ik2: %s: cuBLAS status %d (line %d)\n", #x, (int) e_, __LINE__);    \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "maya-ik2: %s\n", m.c_str());
    std::exit(1);
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- the files

struct Shards {
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::vector<std::string> paths;
    std::map<std::string, std::pair<int, const strata::TensorInfo*>> by_name;

    explicit Shards(const std::string& first) {
        std::vector<std::string> ps{first};
        const size_t at = first.find("-00001-of-");
        if (at != std::string::npos) {
            const int n = std::atoi(first.c_str() + at + 10);
            for (int i = 2; i <= n; ++i) {
                char num[16];
                std::snprintf(num, sizeof num, "%05d", i);
                std::string p = first;
                p.replace(at + 1, 5, num);
                ps.push_back(p);
            }
        }
        for (const auto& p : ps) {
            files.emplace_back(new strata::GgufFile(p));
            paths.push_back(p);
            for (const auto& t : files.back()->tensors()) by_name[t.name] = {(int) files.size() - 1, &t};
        }
    }
    const strata::TensorInfo* info(const std::string& n) const {
        auto it = by_name.find(n);
        return it == by_name.end() ? nullptr : it->second.second;
    }
    const uint8_t* data(const std::string& n) const {
        auto it = by_name.find(n);
        if (it == by_name.end()) die("no tensor " + n);
        return files[it->second.first]->tensor_data(*it->second.second);
    }
    uint64_t file_offset(const std::string& n) const {
        auto it = by_name.find(n);
        return files[it->second.first]->data_start() + it->second.second->offset;
    }
    int file_of(const std::string& n) const { return by_name.at(n).first; }
    double num(const std::string& key) const {
        const auto* v = files[0]->get(key);
        if (!v) die("no key " + key);
        return v->num();
    }
    const strata::MetaValue* get(const std::string& key) const { return files[0]->get(key); }
};

size_t tensor_bytes(const strata::TensorInfo& t) {
    // (the blocks decoded here first: strata's block_geometry names NVFP4 but has no geometry for it)
    if (k::block_values((int) t.type)) return (size_t) (t.elements() / k::block_values((int) t.type)) * k::block_bytes((int) t.type);
    int e = 0, b = 0;
    if (!strata::block_geometry(t.type, e, b)) die(std::string("type ") + t.type_name() + " of " + t.name);
    return (size_t) (t.elements() / e) * b;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- the model

struct Config {
    int n_layer, C, H, KV, D, E, K, F, window, d_rel, extent, extent_swa, block, n_vocab, n_rows;
    // A looped model (Inkling-2-Mini): the layers core0 .. core1 - 1 (whole blocks) run n_pass times a token, so a
    // token is n_visit layer visits (n_layer without a loop).  rank: the rank of the loop's message gates.
    int n_visit = 0, n_pass = 1, core0 = 0, core1 = 0, rank = 0;
    float eps, route_scale, qk_clip, rel_clip, alpha, n_floor, gate_cap, up_cap, logit_cap;
    std::vector<bool> sliding;   // by layer
};

// a dense matrix as the GGUF holds it, in VRAM: its blocks [N][K]
struct Mat {
    // (IK2_DENSE=q8 with Q8_0 blocks and K a multiple of 128: the blocks taken apart - the int8 values [N][K] and the
    //  blocks' FP16 scales [N][K / 32] - which a warp reads 128 values an instruction; raw is then null)
    int8_t* q8 = nullptr;
    uint16_t* q8d = nullptr;
    uint8_t* raw = nullptr;     // IK2_DENSE=q8: the blocks, one token's products straight over them
    uint16_t* f16 = nullptr;    // else (the default): an FP16 copy, every product a cuBLAS GEMM - measured faster on an
    int type = 0, N = 0, K = 0; // RTX PRO 6000 (16 against 21 ms a token), for 4 GiB more VRAM
};

struct Layer {
    int dev = 0;   // the GPU it runs on (--gpus: the layers in equal runs)
    bool sliding;
    // FP16 [out][in] (fused projections on !g_dense_q8, or when quant types match on g_dense_q8)
    Mat wq, wk, wv, wr, wgate, wo;
    Mat wqkvrg, wqrg, wk_dyn, wv_dyn;
    uint16_t *sink16 = nullptr;                    // F16 [H][C] (used in g_dense_q8; fused in !g_dense_q8)
    uint16_t *router16 = nullptr;                  // F16 [E][C]
    float *probs_b;                                // F32 [E]
    float *attn_norm, *q_in_norm, *k_in_norm, *v_in_norm, *ffn_norm;
    float *q_clip, *r_clip, *rel_proj;             // [H][D], [H][d_rel], [extent][d_rel]
    float *res_score, *ffn_res_score;              // [nq][C], [C]
    int n_res_q;
    float *sc_k, *sc_v, *sc_attn, *sc_mlp;         // [C'][4]
    uint16_t *sc_k_dyn16 = nullptr, *sc_v_dyn16 = nullptr;   // F16 [KV * 4][C] (used in g_dense_q8; fused in !g_dense_q8)
    // the experts: types, bytes an expert, host blocks, scales (host)
    int ty[3];
    size_t eb[3];            // bytes of a role's part of an expert
    size_t stride[3];        // bytes between two experts' parts in host[r] (eb, or 2 eb in a fused gate_up, or slot_bytes)
    const uint8_t* host[3];
    bool host_contig = false;
    size_t slot_bytes = 0;
    // the offload path's tier, pinned after its plan: nopin[e] 1 - expert e is held in VRAM only, its place in host[]
    // never read (no RAM); copied in from where the files are mapped instead, fh[r] + e * fstride[r]. Empty: all pinned
    std::vector<uint8_t> nopin;
    const uint8_t* fh[3] = {nullptr, nullptr, nullptr};
    size_t fstride[3] = {0, 0, 0};
    std::vector<float> scale[3];                   // empty: 1
    // a looped core's layer: its later passes' message gates by module (0 q's, 1 k's, 2 v's, 3 the MoE's) -
    // down [rank][2 C], forget and gain [rank][C] - and the four messages' queries [4][C] in that order (F32)
    float *lp_down[4] = {}, *lp_forget[4] = {}, *lp_gain[4] = {}, *lp_query = nullptr;
    // the resident token path (every expert in VRAM): where each expert's blocks are, and the experts' scales, on the
    // device ([E] each; a scale's null: 1)
    const uint8_t** tab_blob = nullptr;
    float* tab_sc[3] = {};
    std::vector<const uint8_t*> tab_h;             // (tab_blob on the host: a short prompt's groups)
    // the state: of the visit being run (bind_visit)
    uint16_t *kc = nullptr, *vc = nullptr;         // FP16 [kv_cap][KV * D] (IK2_KV=f16)
    uint8_t *kc8 = nullptr, *vc8 = nullptr;        // FP8 E4M3 [kv_cap][KV * D] (IK2_KV=fp8)
    int kv_cap = 0;
    float *h_k, *h_v, *h_attn, *h_mlp;             // conv histories [3][C']
    float *rec_old[4] = {}, *rec_x[4] = {};        // (speculative) the convs' histories and inputs of a verify window
};

// A layer visit and its own state.  Without a loop a visit is a layer; a looped core's layer is visited once a pass,
// each time with that visit's K / V cache and conv histories (the weights, and the experts in the cache, are the
// layer's).  bind_visit() points the layer's state at a visit's before it runs.
struct Visit {
    int l = 0;        // the layer
    int pass = 0;     // the core's pass (0: its first, or a layer outside the core)
    uint16_t *kc = nullptr, *vc = nullptr;
    uint8_t *kc8 = nullptr, *vc8 = nullptr;
    int kv_cap = 0;
    float *h_k = nullptr, *h_v = nullptr, *h_attn = nullptr, *h_mlp = nullptr;
    float *rec_old[4] = {}, *rec_x[4] = {};
};

struct Model {
    Config c;
    std::vector<Layer> L;
    std::vector<Visit> V;   // the visits of a token, in order
    Mat out_w;            // [n_rows][C]
    float *out_norm, *out_res_score, *embd_norm;
    float* lp_query_all = nullptr;   // a looped core's message queries, layer by layer ([n_core][4][C]: Layer::lp_query's)
    const uint8_t* embd;  // host Q8_0 [n_rows][C]
    int embd_type;
};

inline Layer& bind_visit(Model& M, int v) {
    const Visit& Vi = M.V[v];
    Layer& Ly = M.L[Vi.l];
    Ly.kc = Vi.kc, Ly.vc = Vi.vc, Ly.kc8 = Vi.kc8, Ly.vc8 = Vi.vc8, Ly.kv_cap = Vi.kv_cap;
    Ly.h_k = Vi.h_k, Ly.h_v = Vi.h_v, Ly.h_attn = Vi.h_attn, Ly.h_mlp = Vi.h_mlp;
    for (int i = 0; i < 4; ++i) Ly.rec_old[i] = Vi.rec_old[i], Ly.rec_x[i] = Vi.rec_x[i];
    return Ly;
}

// ---- the device helpers

cublasHandle_t g_blas;
cudaStream_t g_s;
cudaStream_t g_cs;          // the experts' copies of the token path, beside the compute stream
cudaEvent_t g_ev;
bool g_overlap = true;      // IK2_OVERLAP=0: the copies on the compute stream, before the experts
// The CPU lane (IK2_CPU=1: on): a missing NVFP4 expert computed in host memory by the pool while the others' blocks
// are copied; its own blocks follow on the copy stream without being waited for ("lazy"), so it is in VRAM next time.
ik2::cpu::Pool* g_pool = nullptr;
double g_t_cpu = 0.5e-3, g_t_up = 0.36e-3;   // seconds an expert: the pool's (measured as it runs), the link's
uint64_t g_n_cpu = 0, g_n_wait = 0;
bool g_cpu_lazy = true;   // IK2_CPU_LAZY=0: a CPU-computed expert is not copied in afterwards (the link carries less)
constexpr int kRing = 256;
cudaEvent_t g_ring[kRing];                   // a lazy batch's end on the copy stream, by sequence number
uint64_t g_seq_next = 1, g_seq_done = 0;
void lazy_poll() {
    while (g_seq_done + 1 < g_seq_next && cudaEventQuery(g_ring[(g_seq_done + 1) % kRing]) == cudaSuccess) ++g_seq_done;
}

// IK2_PROF=1: the stream waited for at each phase's end, its time added to the phase (slower, but where it goes)
bool g_prof = false;
bool g_mv = true;   // IK2_MV=0: a few tokens' experts through the FP16 GEMMs too
// How often each expert is picked by a token (prefill and decode), [n_layer * E]: IK2_POP_OUT=FILE writes it at
// the end (int32 n_layer, int32 E, then uint64 counts); IK2_WARM=FILE fills the expert cache with the most picked
// experts at load, so the first requests do not start from an empty cache.
std::vector<uint64_t> g_pop;
int g_pop_E = 0;
void pop_count(int l, const int* top, int S) {
    if (g_pop.empty()) return;
    for (int i = 0; i < S; ++i) ++g_pop[(size_t) l * g_pop_E + top[i]];
}
void pop_dump() {
    const char* f = getenv("IK2_POP_OUT");
    if (!f || g_pop.empty()) return;
    FILE* o = std::fopen(f, "wb");
    if (!o) return;
    const int32_t hdr[2] = {(int32_t) (g_pop.size() / g_pop_E), (int32_t) g_pop_E};
    std::fwrite(hdr, 4, 2, o);
    std::fwrite(g_pop.data(), 8, g_pop.size(), o);
    std::fclose(o);
}
int g_pop_E_ep = 0;
// IK2_EP=1 with --gpus 2: expert parallel. Every layer's attention and dense weights (and the KV cache) on GPU 0;
// expert e of every layer belongs to GPU e & 1, each GPU with its own cache over all layers and its own PCIe link.
// In decode a layer's picks go to their GPUs at once, both links copying the misses side by side.
bool g_ep = false;
cudaEvent_t g_ep_x, g_ep_cp[2], g_ep_done;
// IK2_TP=1 with --gpus 2: the experts tensor-parallel instead. GPU d holds half of every expert: rows [d F/2, ..) of
// gate and up and rows [d C/2, ..) of down, so every miss is half on each link. A layer: each GPU its half of h
// (gate/up/SwiGLU), the halves swapped (P2P), each GPU its half of the output rows (down), concatenated on GPU 0.
// Attention, dense weights and the KV cache on GPU 0 as with IK2_EP.
bool g_tp = false;
cudaEvent_t g_tp_h[2];
int g_tp_F = 0, g_tp_C = 0;
inline bool dual() { return g_ep || g_tp; }
// which GPU an expert belongs to, [n_layer * E]: IK2_EP_SPLIT = GPU 1's share of the picks (0.5; GPU 0 also holds the
// attention, the dense weights and the KV cache). With IK2_WARM's profile each layer's experts are dealt by how often
// they are picked, so the two GPUs' shares come out as asked; without it by a hash of the expert.
std::vector<uint8_t> g_ep_own;
inline int ep_owner(int l, int e) { return g_ep_own[(size_t) l * g_pop_E_ep + e]; }
bool g_flash = true;   // IK2_FLASH=0: the prefill's attention (FP16 KV) through the logits in memory and cuBLAS, not fused
int g_n_sm = 0;
bool g_group = true;   // IK2_GROUP=0: 2 .. 8 tokens' experts a slot at a time, their attention a head at a time
bool g_group_exp = true, g_group_attn = true;   // (IK2_GROUP=1: the experts grouped only, 2: the attention only)
FILE* g_trace = nullptr;   // IK2_TRACE=FILE: each layer's picks, records of int32 (layer, T, K, T x K experts)
struct Prof {
    double attn = 0, route = 0, upload = 0, experts = 0, rest = 0;
    double t = 0;
    void start() {
        if (g_prof) cudaStreamSynchronize(g_s), t = now();
    }
    void lap(double& acc) {
        if (!g_prof) return;
        cudaStreamSynchronize(g_s);
        const double n = now();
        acc += n - t;
        t = n;
    }
} g_pf;

template <class T>
T* dalloc(size_t n) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    return p;
}

// Y [T][N] (ldy) = X [T][K] FP16 . W [N][K]^T FP16, F32 sums
void gemm16(const uint16_t* W, int N, int Kd, const uint16_t* X, int T, float* Y, int ldy, float beta = 0.0f) {
#ifdef STRATA_USE_HIP
    // (AMD: hipBLASLt's kernel for the shape measured once on this GPU and kept in the store - hipBLAS's own pick for
    //  these FP16 shapes on gfx11 is its generic kernel, ~5x slower; fewer than kMinRows rows: hipBLAS)
    if (beta == 0.0f && strata::ds4::lt::gemm(g_s, W, N, Kd, X, Kd, T, Y, ldy)) return;
#endif
    const float one = 1.0f;
    CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, Kd, &one, W, CUDA_R_16F, Kd, X, CUDA_R_16F, Kd, &beta, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

uint8_t* g_stage = nullptr;   // a raw tensor on its way to its FP16 decode
size_t g_stage_bytes = 0;

bool g_dense_q8 = false;
// The KV cache in FP16 (IK2_KV=fp8: E4M3, half the bytes).  Against ref/inkling2.py over 48 tokens FP16 kept every
// top-1 token (KL mean 8.5e-4) and FP8 changed 5 (KL 8.3e-3): Inkling's router is full of near-ties, so FP8's 1.5% in
// layer 0's attention output already moves 7 of 48 tokens' experts.  The full layers' cache is 32 KB a token in FP16.
bool g_kv_fp8 = false;
// Q8_0 blocks (2 bytes of FP16 scale, 32 int8 values) taken apart into m.q8 / m.q8d on the device; false if the
// matrix cannot be kept so (another type, or K not a multiple of 128)
bool upload_planar(Mat& m, const std::vector<const uint8_t*>& parts, const std::vector<size_t>& part_values) {
    if (m.type != k::kQ8_0 || m.K % 128) return false;
    size_t n = 0;
    for (size_t v : part_values) n += v;
    std::vector<int8_t> q(n);
    std::vector<uint16_t> d(n / 32);
    size_t b = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        const uint8_t* p = parts[i];
        for (size_t j = 0; j < part_values[i] / 32; ++j, ++b, p += 34) {
            std::memcpy(&d[b], p, 2);
            std::memcpy(&q[b * 32], p + 2, 32);
        }
    }
    m.q8 = dalloc<int8_t>(n);
    m.q8d = dalloc<uint16_t>(n / 32);
    CU(cudaMemcpy(m.q8, q.data(), n, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(m.q8d, d.data(), n / 32 * 2, cudaMemcpyHostToDevice));
    return true;
}

Mat load_mat(const Shards& S, const std::string& n, int N, int Kd) {
    const auto* t = S.info(n);
    if (!t) die("no tensor " + n);
    if (t->elements() != (uint64_t) N * Kd) die(n + ": " + std::to_string(t->elements()) + " values");
    if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
        die(n + ": type " + t->type_name() + " not read here");
    Mat m;
    m.type = (int) t->type, m.N = N, m.K = Kd;
    const size_t nb = tensor_bytes(*t);
    if (g_dense_q8) {
        if (upload_planar(m, {S.data(n)}, {(size_t) N * Kd})) return m;
        m.raw = dalloc<uint8_t>(nb);
        CU(cudaMemcpy(m.raw, S.data(n), nb, cudaMemcpyHostToDevice));
        return m;
    }
    if (nb > g_stage_bytes) {
        if (g_stage) cudaFree(g_stage);
        g_stage = dalloc<uint8_t>(g_stage_bytes = nb);
    }
    CU(cudaMemcpy(g_stage, S.data(n), nb, cudaMemcpyHostToDevice));
    m.f16 = dalloc<uint16_t>((size_t) N * Kd);
    k::decode_f16(m.type, g_stage, (size_t) N * Kd, m.f16, g_s);
    CU(cudaStreamSynchronize(g_s));
    return m;
}

struct SegPack {
    float* dst[6];
    int off[7];
    int n;
};
// (float4 a thread when every width is a multiple of 4 - the segments' lookup unrolled, so the parameters stay in
// the constant bank instead of being copied to local memory for a dynamic index)
__global__ void unpack_segments4_kernel(const float4* __restrict__ src, int T, SegPack p) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    const int stride4 = p.off[p.n] >> 2;
    if (c4 >= stride4 || t >= T) return;
    const float4 v = src[(size_t) t * stride4 + c4];
    const int col = c4 << 2;
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < p.n && col >= p.off[s] && col < p.off[s + 1]) {
            const int w = p.off[s + 1] - p.off[s];
            *(float4*) (p.dst[s] + (size_t) t * w + (col - p.off[s])) = v;
        }
    }
}
__global__ void unpack_segments_kernel(const float* __restrict__ src, int T, SegPack p) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int stride = p.off[p.n];
    if (col >= stride || t >= T) return;
    const float v = src[(size_t) t * stride + col];
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < p.n && col >= p.off[s] && col < p.off[s + 1]) {
            const int w = p.off[s + 1] - p.off[s];
            p.dst[s][(size_t) t * w + (col - p.off[s])] = v;
        }
    }
}
void unpack_segments(const float* src, int T, const std::initializer_list<std::pair<float*, int>>& segs, cudaStream_t s) {
    SegPack p{};
    p.n = (int) segs.size();
    int off = 0, idx = 0;
    bool vec = ((uintptr_t) src & 15) == 0;
    p.off[0] = 0;
    for (const auto& sg : segs) {
        p.dst[idx] = sg.first;
        vec = vec && (sg.second & 3) == 0 && ((uintptr_t) sg.first & 15) == 0;
        off += sg.second;
        p.off[++idx] = off;
    }
    if (vec) {
        const dim3 grid((unsigned) ((off / 4 + 255) / 256), (unsigned) T);
        unpack_segments4_kernel<<<grid, 256, 0, s>>>((const float4*) src, T, p);
    } else {
        const dim3 grid((unsigned) ((off + 255) / 256), (unsigned) T);
        unpack_segments_kernel<<<grid, 256, 0, s>>>(src, T, p);
    }
}

struct MatPart {
    std::string name;
    int N = 0;
    std::vector<float> f32;
};

void ensure_stage(size_t nb) {
    if (nb > g_stage_bytes) {
        if (g_stage) cudaFree(g_stage);
        g_stage = dalloc<uint8_t>(g_stage_bytes = nb);
    }
}

Mat load_fused_f16(const Shards& S, const std::vector<MatPart>& parts, int Kd) {
    int N_tot = 0;
    for (const auto& pt : parts) N_tot += pt.N;
    Mat m;
    m.N = N_tot;
    m.K = Kd;
    m.f16 = dalloc<uint16_t>((size_t) N_tot * Kd);
    int row = 0;
    for (const auto& pt : parts) {
        if (pt.f32.empty()) {
            const auto* t = S.info(pt.name);
            if (!t) die("no tensor " + pt.name);
            if (t->elements() != (uint64_t) pt.N * Kd) die(pt.name + ": " + std::to_string(t->elements()) + " values");
            if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
                die(pt.name + ": type " + t->type_name() + " not read here");
            const size_t nb = tensor_bytes(*t);
            ensure_stage(nb);
            CU(cudaMemcpy(g_stage, S.data(pt.name), nb, cudaMemcpyHostToDevice));
            k::decode_f16((int) t->type, g_stage, (size_t) pt.N * Kd, m.f16 + (size_t) row * Kd, g_s);
        } else {
            const size_t nb = pt.f32.size() * 4;
            ensure_stage(nb);
            CU(cudaMemcpy(g_stage, pt.f32.data(), nb, cudaMemcpyHostToDevice));
            k::to_f16((const float*) g_stage, pt.f32.size(), m.f16 + (size_t) row * Kd, g_s);
        }
        row += pt.N;
    }
    CU(cudaStreamSynchronize(g_s));
    return m;
}

bool try_load_fused_raw(const Shards& S, const std::vector<MatPart>& parts, int Kd, Mat& out) {
    int ty = -1, N_tot = 0;
    size_t nb_tot = 0;
    for (const auto& pt : parts) {
        if (!pt.f32.empty()) return false;
        const auto* t = S.info(pt.name);
        if (!t) die("no tensor " + pt.name);
        if (t->elements() != (uint64_t) pt.N * Kd) die(pt.name + ": " + std::to_string(t->elements()) + " values");
        if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
            die(pt.name + ": type " + t->type_name() + " not read here");
        if (ty < 0) ty = (int) t->type;
        else if (ty != (int) t->type) return false;
        N_tot += pt.N;
        nb_tot += tensor_bytes(*t);
    }
    out = Mat{};
    out.type = ty;
    out.N = N_tot;
    out.K = Kd;
    {
        std::vector<const uint8_t*> ps;
        std::vector<size_t> vs;
        for (const auto& pt : parts) ps.push_back(S.data(pt.name)), vs.push_back((size_t) pt.N * Kd);
        if (upload_planar(out, ps, vs)) return true;
    }
    out.raw = dalloc<uint8_t>(nb_tot);
    size_t off = 0;
    for (const auto& pt : parts) {
        const size_t nb = tensor_bytes(*S.info(pt.name));
        CU(cudaMemcpy(out.raw + off, S.data(pt.name), nb, cudaMemcpyHostToDevice));
        off += nb;
    }
    return true;
}

// one token's product over a dense matrix kept as the GGUF's blocks, or planar
void mv1(const Mat& m, const float* x, float* y) {
    if (m.q8) k::mv_q8p(m.q8, m.q8d, m.N, m.K, x, y, g_s);
    else k::mv_blocks(m.type, m.raw, m.N, m.K, x, y, g_s);
}
// ... with up to two short convs on its rows: the planar Q8 product has them as its epilogue (a launch less each;
// those rows of y are then not written), any other matrix gets the product and then the convs
void mv1c(const Mat& m, const float* x, float* y, const k::MvConv& c0, const k::MvConv& c1 = k::MvConv{}) {
    if (m.q8) {
        k::mv_q8p_conv(m.q8, m.q8d, m.N, m.K, x, y, c0, c1, g_s);
        return;
    }
    k::mv_blocks(m.type, m.raw, m.N, m.K, x, y, g_s);
    for (const k::MvConv* cv : {&c0, &c1})
        if (cv->n) k::conv4_res(y + cv->r0, cv->hist, cv->w, cv->dyn, cv->heads, 1, cv->n, cv->res, cv->y, g_s);
}

uint16_t* g_w16 = nullptr;   // a matrix's FP16 decode for the prompt path's GEMMs (the largest of a layer's)
size_t g_w16_n = 0;          // values it holds
// Y [T][N] = X [T][K] . W^T: one token straight over the blocks (x32), more through an FP16 decode and a GEMM (x16).
// A matrix larger than the decode buffer (the head: its few rows of logits) goes a token at a time over its blocks.
void matmul(const Mat& m, const float* x32, const uint16_t* x16, int T, float* Y, int ldy) {
    if (m.f16) {
        gemm16(m.f16, m.N, m.K, x16, T, Y, ldy);
        return;
    }
    if (m.K <= 8192 && (T == 1 || (size_t) m.N * m.K > g_w16_n)) {
        for (int t = 0; t < T; ++t) mv1(m, x32 + (size_t) t * m.K, Y + (size_t) t * ldy);
        return;
    }
    if (m.q8) k::decode_q8p_f16(m.q8, m.q8d, (size_t) m.N * m.K, g_w16, g_s);
    else k::decode_f16(m.type, m.raw, (size_t) m.N * m.K, g_w16, g_s);
    gemm16(g_w16, m.N, m.K, x16, T, Y, ldy);
}

std::vector<float> host_f32(const Shards& S, const std::string& n, size_t expect) {
    const auto* t = S.info(n);
    if (!t) die("no tensor " + n);
    if (t->elements() != expect) die(n + ": " + std::to_string(t->elements()) + " values, expected " + std::to_string(expect));
    std::vector<float> v(expect);
    const uint8_t* p = S.data(n);
    if (t->type == 0) {
        std::memcpy(v.data(), p, expect * 4);
    } else if (t->type == 30) {
        for (size_t i = 0; i < expect; ++i) v[i] = bf16_to_f32((uint16_t) (p[2 * i] | (p[2 * i + 1] << 8)));
    } else if (k::block_values((int) t->type)) {
        k::decode_host((int) t->type, p, expect, v.data());
    } else {
        die(n + ": type " + t->type_name());
    }
    return v;
}
float* dev_f32(const std::vector<float>& v) {
    float* p = dalloc<float>(v.size());
    CU(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return p;
}
uint16_t* dev_f16(const std::vector<float>& v) {
    const size_t nb = v.size() * 4;
    ensure_stage(nb);
    CU(cudaMemcpy(g_stage, v.data(), nb, cudaMemcpyHostToDevice));
    uint16_t* p = dalloc<uint16_t>(v.size());
    k::to_f16((const float*) g_stage, v.size(), p, g_s);
    CU(cudaStreamSynchronize(g_s));
    return p;
}
float* load_f32(const Shards& S, const std::string& n, size_t expect) { return dev_f32(host_f32(S, n, expect)); }
uint16_t* load_f16(const Shards& S, const std::string& n, size_t expect) { return dev_f16(host_f32(S, n, expect)); }

// ---- loading the experts into pinned host memory: pread from the files, several at once

struct Pinned {
    struct Block {
        void* ptr = nullptr;
        size_t bytes = 0;
        bool mapped = false;
        bool registered = false;
    };
    std::vector<Block> blocks;
    std::vector<std::pair<void*, size_t>> runs;   // (the offload path's pinned runs inside blocks not registered whole)
    ~Pinned() {
        for (const auto& r : runs) cudaHostUnregister(r.first);
        for (const auto& b : blocks) {
            if (b.mapped) {
                if (b.registered) cudaHostUnregister(b.ptr);
                ::munmap(b.ptr, b.bytes);
            } else if (b.ptr) {
                cudaFreeHost(b.ptr);
            }
        }
    }
};

void read_into_strided(const std::string& path, uint64_t off, int n_exp, size_t exp_bytes, size_t dst_stride,
                       uint8_t* dst) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open " + path);
    const size_t chunk = 64ull << 20;
    for (int e = 0; e < n_exp; ++e) {
        size_t done = 0;
        uint8_t* ep = dst + (size_t) e * dst_stride;
        const uint64_t eoff = off + (uint64_t) e * exp_bytes;
        while (done < exp_bytes) {
            const ssize_t r = ::pread(fd, ep + done, std::min(chunk, exp_bytes - done), (off_t) (eoff + done));
            if (r <= 0) die("read failed: " + path);
            done += (size_t) r;
        }
    }
    posix_fadvise(fd, (off_t) off, (off_t) ((uint64_t) n_exp * exp_bytes), POSIX_FADV_DONTNEED);
    ::close(fd);
}

#ifndef _WIN32
// can anonymous mappings be pinned (cudaHostRegister)?  A 2 MiB probe
bool host_register_works() {
    const size_t probe_bytes = 2ULL << 20;
    void* mp = ::mmap(nullptr, probe_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mp == MAP_FAILED) return false;
    *((volatile uint8_t*) mp) = 0;
    bool ok = cudaHostRegister(mp, probe_bytes, cudaHostRegisterPortable) == cudaSuccess;
    if (ok) CU(cudaHostUnregister(mp));
    else cudaGetLastError();
    ::munmap(mp, probe_bytes);
    return ok;
}
#endif

// ---- the expert cache in VRAM: per-size-class pools, O(1) S3-FIFO with ghost history

// GPU d's half of role r (0 gate, 1 up, 2 down) of an expert: offset in the role's part, bytes (IK2_TP)
inline void tp_part(const Layer& Ly, int d, int r, size_t& off, size_t& bytes) {
    const size_t rows = r < 2 ? (size_t) g_tp_F : (size_t) g_tp_C, rb = Ly.eb[r] / rows, half = rows / 2;
    off = (size_t) d * half * rb, bytes = half * rb;
}
inline size_t tp_bytes(const Layer& Ly, int d) {
    size_t o, b, t = 0;
    for (int r = 0; r < 3; ++r) tp_part(Ly, d, r, o, b), t += b;
    return t;
}
struct ExpertCache {
    static constexpr uint8_t kMaxFreq = 7;
    struct Pool {
        size_t slot_bytes = 0;
        int n_slots = 0, prob_cap = 0, n_layers = 0;
        int n_active = 0;               // slots in use as cache; [n_active, n_slots) lent out (lend)
        uint8_t* data = nullptr;
        std::vector<int> owner;         // slot -> expert idx (l * E + e), -1 if free
        std::vector<uint64_t> used;     // slot -> last tick
        std::vector<uint64_t> sseq;     // slot -> lazy batch seq (0: none)
        std::vector<uint8_t> qid;       // 0=free, 1=prob, 2=main
        std::vector<uint8_t> freq;      // saturating access counter [0..kMaxFreq]
        std::vector<int> prev, next;
        int prob_head = -1, prob_tail = -1, prob_sz = 0;
        int main_head = -1, main_tail = -1, main_sz = 0;
        int free_head = -1;
        uint64_t ghost_seq = 1;
    };

    int n_layers = 0, E = 0;
    std::vector<Pool> pools;
    std::vector<int> layer_pool;       // layer l -> pool index (-1 if on another GPU)
    std::vector<int> loc;              // l * E + e -> slot in pools[layer_pool[l]], or -1
    std::vector<uint64_t> ghost_at;    // l * E + e -> ghost_seq when evicted in decode (0: none)
    int last_handle = -1;
    bool scan = false;   // a long prefill's pass (evict_scan; IK2_SCAN=0: off)
    // the offload path's tier (set before init): the pools are only what the profile is worth a byte - no minimum
    // of a layer's experts (prompts take their misses through the staging ring, run_prefill_moe), none merged
    bool tier_only = false;
    int tp = -1;         // IK2_TP: this GPU's half of every expert (its index), not whole experts
    uint64_t tick = 0, hits = 0, misses = 0, bytes_up = 0;
    double t_up = 0;

    static void push_back(Pool& P, int q, int s) {
        P.qid[s] = (uint8_t) q;
        P.prev[s] = -1;
        P.next[s] = -1;
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        if (tl < 0) hd = tl = s;
        else P.next[tl] = s, P.prev[s] = tl, tl = s;
        ++sz;
    }
    static int pop_front(Pool& P, int q) {
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        const int s = hd;
        if (s < 0) return -1;
        hd = P.next[s];
        if (hd >= 0) P.prev[hd] = -1;
        else tl = -1;
        P.prev[s] = P.next[s] = -1;
        P.qid[s] = 0;
        --sz;
        return s;
    }
    static int evict_from_prob(Pool& P, uint64_t t, bool& saw_busy_lazy) {
        const int n = P.prob_sz;
        for (int i = 0; i < n; ++i) {
            const int s = pop_front(P, 1);
            if (P.used[s] >= t) {
                push_back(P, 1, s);
                continue;
            }
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) {
                    saw_busy_lazy = true;
                    push_back(P, 1, s);
                    continue;
                }
            }
            if (P.freq[s] > 0) {
                P.freq[s] = std::min<uint8_t>(P.freq[s] - 1, 1);
                push_back(P, 2, s);
                continue;
            }
            return s;
        }
        return -1;
    }
    static int evict_from_main(Pool& P, uint64_t t, bool& saw_busy_lazy) {
        int pinned_run = 0;
        while (P.main_sz > 0 && pinned_run < P.main_sz) {
            const int s = pop_front(P, 2);
            if (P.used[s] >= t) {
                ++pinned_run;
                push_back(P, 2, s);
                continue;
            }
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) {
                    saw_busy_lazy = true;
                    ++pinned_run;
                    push_back(P, 2, s);
                    continue;
                }
            }
            pinned_run = 0;
            if (P.freq[s] > 0) {
                --P.freq[s];
                push_back(P, 2, s);
                continue;
            }
            return s;
        }
        return -1;
    }
    // A pass that streams nearly every expert of every layer in order (a long prefill), once the cache is full:
    // the victim is the most recently taken probation slot that is free to go (in practice an expert of the layer
    // just computed), not the oldest. Under the usual order such a pass, larger than the cache, evicts the experts of
    // the layers it is about to reach, and the next pass finds none of them (32k after other prompts: 9% hits);
    // like this, the streamed experts take turns in a few slots and the rest of the cache stays for the next pass.
    static int evict_scan(Pool& P, uint64_t t) {
        for (int s = P.prob_tail; s >= 0; s = P.prev[s]) {
            if (P.used[s] >= t) continue;
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) continue;
            }
            unlink(P, s);
            return s;
        }
        return -1;
    }
    static int alloc_slot(Pool& P, uint64_t t, bool scan = false) {
        if (P.free_head >= 0) {
            const int s = P.free_head;
            P.free_head = P.next[s];
            P.next[s] = P.prev[s] = -1;
            return s;
        }
        if (scan) {
            const int s = evict_scan(P, t);
            if (s >= 0) return s;
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool saw_busy_lazy = false;
            int s = -1;
            if (P.prob_sz > P.prob_cap || P.main_sz == 0) {
                s = evict_from_prob(P, t, saw_busy_lazy);
                if (s < 0) s = evict_from_main(P, t, saw_busy_lazy);
            } else {
                s = evict_from_main(P, t, saw_busy_lazy);
                if (s < 0) {
                    const int prev_main = P.main_sz;
                    s = evict_from_prob(P, t, saw_busy_lazy);
                    if (s < 0 && P.main_sz > prev_main) s = evict_from_main(P, t, saw_busy_lazy);
                }
            }
            if (s >= 0) return s;
            if (attempt == 0 && saw_busy_lazy) {
                CU(cudaStreamSynchronize(g_cs));
                g_seq_done = g_seq_next - 1;
                continue;
            }
            break;
        }
        die("expert cache: no slot free");
    }

    void init(const std::vector<Layer>& L, int dev, int n_layers_in, int E_in, size_t budget,
              const std::vector<double>* value = nullptr) {
        n_layers = n_layers_in;
        E = E_in;
        layer_pool.assign(n_layers, -1);
        loc.assign((size_t) n_layers * E, -1);
        ghost_at.assign((size_t) n_layers * E, 0);
        struct Group {
            size_t sb;
            std::vector<int> layers;
        };
        std::map<size_t, std::vector<int>> by_sb;
        for (int l = 0; l < n_layers; ++l) {
            if (dev >= 0 && L[l].dev != dev) continue;   // (dev -1: every layer, an expert-parallel GPU)
            const size_t eb = tp >= 0 ? tp_bytes(L[l], tp) : L[l].eb[0] + L[l].eb[1] + L[l].eb[2];
            const size_t sb = (eb + 255) / 256 * 256;
            by_sb[sb].push_back(l);
        }
        std::vector<Group> grps;
        for (auto& kv : by_sb) grps.push_back({kv.first, std::move(kv.second)});
        auto min_req_bytes = [&]() {
            size_t s = 0;
            for (const auto& g : grps) s += (size_t) std::min(E, (int) g.layers.size() * E) * g.sb;
            return s;
        };
        while (grps.size() > 1 && ((!tier_only && min_req_bytes() > budget) || grps.size() > 4)) {
            size_t best_waste = SIZE_MAX;
            size_t best_i = 0;
            for (size_t i = 0; i + 1 < grps.size(); ++i) {
                const size_t w = grps[i].layers.size() * (grps[i + 1].sb - grps[i].sb);
                if (w < best_waste) best_waste = w, best_i = i;
            }
            grps[best_i + 1].layers.insert(grps[best_i + 1].layers.end(), grps[best_i].layers.begin(),
                                           grps[best_i].layers.end());
            grps.erase(grps.begin() + (ptrdiff_t) best_i);
        }
        const size_t hard_min = tier_only ? 0 : min_req_bytes();
        if (hard_min > budget)
            die("--cache-gb too small: need at least " + std::to_string((hard_min + (1 << 20) - 1) >> 20) + " MiB");
        size_t soft_min = 0;
        for (const auto& g : grps) soft_min += (size_t) std::min(E + 16, (int) g.layers.size() * E) * g.sb;
        const int min_base = tier_only ? 0 : soft_min <= budget ? E + 16 : E;

        pools.resize(grps.size());
        std::vector<int> pmin(grps.size()), pmax(grps.size());
        for (size_t p = 0; p < grps.size(); ++p) {
            pools[p].slot_bytes = grps[p].sb;
            pools[p].n_layers = (int) grps[p].layers.size();
            for (int l : grps[p].layers) layer_pool[l] = (int) p;
            pmax[p] = pools[p].n_layers * E;
            pmin[p] = std::min(min_base, pmax[p]);
        }
        auto used_for = [&](double spl) {
            size_t u = 0;
            for (size_t p = 0; p < pools.size(); ++p) {
                const int ns = std::clamp((int) std::floor(pools[p].n_layers * spl), pmin[p], pmax[p]);
                u += (size_t) ns * pools[p].slot_bytes;
            }
            return u;
        };
        double lo = 0.0, hi = (double) E;
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (used_for(mid) <= budget) lo = mid;
            else hi = mid;
        }
        size_t used = 0;
        for (size_t p = 0; p < pools.size(); ++p) {
            pools[p].n_slots = std::clamp((int) std::floor(pools[p].n_layers * lo), pmin[p], pmax[p]);
            used += (size_t) pools[p].n_slots * pools[p].slot_bytes;
        }
        if (value && value->size() == (size_t) n_layers * E) {
            // (the offload path's tier: the pools sized by what their experts are worth a byte - a profile's picks
            //  over the slot's bytes, the most first - each pool at least its minimum, the prompt path's)
            std::vector<std::pair<double, int>> ord;   // (picks a byte, pool)
            std::vector<std::vector<double>> by_pool(pools.size());
            for (int l = 0; l < n_layers; ++l)
                if (layer_pool[l] >= 0)
                    for (int e = 0; e < E; ++e) by_pool[layer_pool[l]].push_back((*value)[(size_t) l * E + e]);
            size_t left = budget;
            for (size_t p = 0; p < pools.size(); ++p) {
                std::sort(by_pool[p].begin(), by_pool[p].end(), std::greater<double>());
                pools[p].n_slots = pmin[p];
                left -= std::min(left, (size_t) pmin[p] * pools[p].slot_bytes);
                for (size_t i = (size_t) pmin[p]; i < by_pool[p].size(); ++i)
                    ord.push_back({by_pool[p][i] / (double) pools[p].slot_bytes, (int) p});
            }
            std::stable_sort(ord.begin(), ord.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (const auto& o : ord) {
                Pool& P = pools[(size_t) o.second];
                if (P.n_slots < pmax[(size_t) o.second] && P.slot_bytes <= left) ++P.n_slots, left -= P.slot_bytes;
            }
            used = budget - left;
        }
        size_t rem = budget - used;
        bool grew = true;
        while (grew) {
            grew = false;
            for (size_t p = 0; p < pools.size(); ++p) {
                if (pools[p].n_slots < pmax[p] && pools[p].slot_bytes <= rem) {
                    ++pools[p].n_slots;
                    rem -= pools[p].slot_bytes;
                    grew = true;
                }
            }
        }
        for (size_t p = 0; p < pools.size(); ++p) {
            Pool& P = pools[p];
            P.prob_cap = std::max(1, P.n_slots / 5);
            P.data = dalloc<uint8_t>((size_t) P.n_slots * P.slot_bytes);
            P.owner.assign(P.n_slots, -1);
            P.used.assign(P.n_slots, 0);
            P.sseq.assign(P.n_slots, 0);
            P.qid.assign(P.n_slots, 0);
            P.freq.assign(P.n_slots, 0);
            P.prev.assign(P.n_slots, -1);
            P.next.assign(P.n_slots, -1);
            for (int s = 0; s < P.n_slots - 1; ++s) P.next[s] = s + 1;
            P.free_head = P.n_slots > 0 ? 0 : -1;
            P.n_active = P.n_slots;
        }
    }
    int total_slots() const {
        int s = 0;
        for (const auto& P : pools) s += P.n_slots;
        return s;
    }
    size_t total_bytes() const {
        size_t b = 0;
        for (const auto& P : pools) b += (size_t) P.n_slots * P.slot_bytes;
        return b;
    }
    uint64_t& slot_sseq(int handle) { return pools[handle >> 20].sseq[handle & 0xfffff]; }
    // the slot of expert e of layer l, its blocks copied in if missing; pinned for this tick (not evicted in it)
    // st: the stream the copy is queued on; *missed (given): whether it was copied in now
    // copy false: the slot is taken but its blocks are the caller's to copy in (copy_in)
    uint8_t* get(const Layer& Ly, int l, int e, uint64_t t, cudaStream_t st = nullptr, bool* missed = nullptr,
                 bool copy = true, bool is_prefill = false) {
        if (!st) st = g_s;
        if (missed) *missed = false;
        const int idx = l * E + e;
        const int p = layer_pool[l];
        Pool& P = pools[p];
        if (loc[idx] >= 0) {
            const int s = loc[idx];
            ++hits;
            P.used[s] = t;
            if (P.freq[s] < kMaxFreq) ++P.freq[s];
            last_handle = (p << 20) | s;
            return P.data + (size_t) s * P.slot_bytes;
        }
        ++misses;
        if (missed) *missed = true;
        const int best = alloc_slot(P, t, scan);
        if (P.owner[best] >= 0) {
            const int old_idx = P.owner[best];
            loc[old_idx] = -1;
            if (!is_prefill) ghost_at[old_idx] = P.ghost_seq;
        }
        if (!is_prefill) ++P.ghost_seq;
        const bool is_ghost = !is_prefill && ghost_at[idx] != 0 && (P.ghost_seq - ghost_at[idx]) <= (uint64_t) P.n_slots;
        P.owner[best] = idx;
        P.used[best] = t;
        P.sseq[best] = 0;
        P.freq[best] = is_ghost ? 1 : 0;
        push_back(P, is_ghost ? 2 : 1, best);
        loc[idx] = best;
        last_handle = (p << 20) | best;
        uint8_t* dst = P.data + (size_t) best * P.slot_bytes;
        if (copy) copy_in(Ly, e, dst, st);
        return dst;
    }
    static void unlink(Pool& P, int s) {
        const int q = P.qid[s];
        if (!q) return;
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        if (P.prev[s] >= 0) P.next[P.prev[s]] = P.next[s];
        else hd = P.next[s];
        if (P.next[s] >= 0) P.prev[P.next[s]] = P.prev[s];
        else tl = P.prev[s];
        P.prev[s] = P.next[s] = -1;
        P.qid[s] = 0;
        --sz;
    }
    static void rebuild_free(Pool& P) {
        P.free_head = -1;
        for (int s = P.n_active - 1; s >= 0; --s)
            if (P.owner[s] < 0 && P.qid[s] == 0) P.next[s] = P.free_head, P.prev[s] = -1, P.free_head = s;
    }
    // Lends the last k cache slots of pool p as one block of scratch (the experts in them are dropped), for a long
    // prefill whose passes stream the experts anyway; give_back() makes them cache slots again. Nothing may be in
    // flight on those slots (the caller synchronizes first).
    uint8_t* lend(int p, int k) {
        Pool& P = pools[p];
        if (k <= 0 || k > P.n_active) return nullptr;
        const int s0 = P.n_active - k;
        for (int s = s0; s < P.n_active; ++s) {
            unlink(P, s);
            if (P.owner[s] >= 0) loc[P.owner[s]] = -1, P.owner[s] = -1;
            P.used[s] = 0, P.sseq[s] = 0, P.freq[s] = 0;
        }
        P.n_active = s0;
        rebuild_free(P);
        return P.data + (size_t) s0 * P.slot_bytes;
    }
    void give_back(int p) {
        Pool& P = pools[p];
        P.n_active = P.n_slots;
        rebuild_free(P);
    }
    bool has(int l, int e) const { return loc[l * E + e] >= 0; }
    void copy_in(const Layer& Ly, int e, uint8_t* dst, cudaStream_t st) {
        if (!Ly.nopin.empty() && Ly.nopin[e]) {   // (not in host memory: from the files' mapping, in the slot's order)
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                CU(cudaMemcpyAsync(dst + off, Ly.fh[r] + (size_t) e * Ly.fstride[r], Ly.eb[r], cudaMemcpyHostToDevice, st));
                off += Ly.eb[r], bytes_up += Ly.eb[r];
            }
            return;
        }
        if (tp >= 0) {   // (gate's half, up's half, down's half, packed)
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                size_t o, b;
                tp_part(Ly, tp, r, o, b);
                CU(cudaMemcpyAsync(dst + off, Ly.host[r] + (size_t) e * Ly.stride[r] + o, b, cudaMemcpyHostToDevice, st));
                off += b, bytes_up += b;
            }
            return;
        }
        if (Ly.host_contig) {
            CU(cudaMemcpyAsync(dst, Ly.host[0] + (size_t) e * Ly.slot_bytes, Ly.slot_bytes, cudaMemcpyHostToDevice, st));
            bytes_up += Ly.slot_bytes;
            return;
        }
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            CU(cudaMemcpyAsync(dst + off, Ly.host[r] + (size_t) e * Ly.stride[r], Ly.eb[r], cudaMemcpyHostToDevice, st));
            off += Ly.eb[r];
            bytes_up += Ly.eb[r];
        }
    }
};

// ---- the forward

struct Work {
    static constexpr int kPackRing = 4;
    static constexpr int kMissBatch = 24;
    int Tm, ctx, nk_max, T_sup = 0;
    size_t blk_ld = 0, lg_cap = 0;
    float *h, *res, *r, *a, *tmp, *xq, *xk, *xv, *q, *kk, *vv, *rr, *gate, *sink, *dyn, *kconv, *vconv, *o, *lg;
    float *qkv;         // fused projection staging (aliases gu32)
    float *blocks;      // [n_blk][blk_ld][C]
    uint16_t *x16, *xk16, *xv16, *xq16_sup = nullptr;
    uint16_t *kv_k16 = nullptr, *kv_v16 = nullptr;
    int kv_tmp_cap = 0;
    int* pos;
    float* rlog;
    int* route_d;       // contiguous [S] top (int) followed by [S] weights (float)
    // the experts
    uint16_t *ex16, *w16, *h16;
    float *gu32, *y32, *sg, *su, *wt;
    int *gidx, *inv;
    const uint8_t** blob;
    int* tok;
    uint8_t *pack_h[kPackRing] = {}, *pack_d[kPackRing] = {};   // ring-buffered token path tables
    uint16_t* tp_h = nullptr;   // IK2_TP: the whole h, [S][F] FP16 (each GPU fills its half, then the other's)
    uint8_t* tp_pk = nullptr;   // (the decode's tables of this layer on the device: canonical blobs at +1536)
    float *y_h[kPackRing] = {};                                 // ring-buffered CPU lane outputs [64][C]
    int pack_slot = 0;
    float *x_h = nullptr;                                       // CPU lane inputs [8][C], pinned
    float *emb_h = nullptr;                                     // pinned host staging
    int *pos_h = nullptr, *route_h = nullptr;
    int *gidx_h[2] = {}, *inv_h[2] = {};
    float *sg_h[2] = {}, *su_h[2] = {}, *wt_h[2] = {};
    int *argmax_d = nullptr, *argmax_h = nullptr;
    std::vector<cudaEvent_t> miss_ev;
    // prefill: the next layer's experts copied in while this one runs (prefetch_layer)
    // the prefetches in flight, a ring by layer: their batches' events, expert -> its batch
    // (-1: not prefetched), the layer each slot is for
    static constexpr int kPfRing = 4;
    std::vector<cudaEvent_t> pf_evs[kPfRing];
    std::vector<int> pf_bs[kPfRing];
    int pf_layers[kPfRing] = {-1, -1, -1, -1};
    std::vector<int> miss_pos;   // expert -> its index among this layer's misses (-1: not one)
    std::vector<uint32_t> use;   // [layer * E + e] how many tokens each expert ran in prefills so far
    std::vector<uint64_t> use_tot;   // [layer] their sum
    float* wtr = nullptr;        // [S] the routing weight of each expert-sorted row
    float* wtr_h[2] = {nullptr, nullptr};
    // prefill attention: the relative bias of up to kRelbRows queries x heads, [rows * H][extent] (IK2_RELB=0: off)
    static constexpr int kRelbRows = 512;
    float* relb = nullptr;
    int relb_ext = 0;
    uint16_t* lg16 = nullptr;   // the prefill attention's probabilities (FP16, lg_cap of them; aliases gu32)
    size_t gu32_bytes = 0;   // (gu32's size: a lent scratch only replaces it when it needs more)
    int T_host = 0;          // tokens the pinned host staging (pos_h, route_h, gidx_h, ..) holds
    std::vector<int> cnt, start, fill, hit_exps, miss_exps;
    std::vector<uint8_t*> slot_ptrs;
    std::vector<uint32_t> seen_stamp, gi_stamp;
    std::vector<char> seen_late;
    std::vector<int> gi_idx;
    uint32_t stamp = 0;
    float* logits;
    // A looped core's passes ([Tm][C] each): the stream the core was first given; the bank of the pass before (the
    // blocks it added after its first, then its final stream); a layer's four messages (q's, k's, v's, the MoE's);
    // its three attention inputs before the messages.
    float *lp_pre = nullptr, *lp_bank = nullptr, *lp_msg = nullptr, *lp_in = nullptr;
    float* lp_msg_all = nullptr;   // a token's messages of a pass, every core layer's ([n_core][4][C]): one launch a pass
    float* lp_g = nullptr;   // the message gates' scratch [3][Tm][2 rank + 2]
    float *lp_a = nullptr, *lp_f = nullptr;   // ... of a chunk's: [3][Tm][rank] and [2][Tm][C]
    float* logits_h = nullptr;   // the resident token path's logits, pinned [n_vocab]
    int *cand_d = nullptr, *cand_h = nullptr, *cand_part = nullptr;   // ... its candidates (k::cand_topk's out), scratch
    // a short prompt's experts by groups, every expert in VRAM (forward): the tables, pinned and on the device -
    // [rg_groups] pointers | n [rg_groups] | tok, row [rg_groups][kGroupMax] | sg, su, wt, id [rg_slots]
    uint8_t *rg_h = nullptr, *rg_d = nullptr;
    int rg_slots = 0, rg_groups = 0;
    size_t rg_bytes = 0;
};

// --serve: SIGUSR1 asks the request running to stop (its client went away) at its next token or prompt chunk
volatile std::sig_atomic_t g_cancel = 0;
// IK2_STATS=1: per request, how many tokens the prefill's experts each ran ([0] 1, [1] 2, [2] 3-7, [3] 8+)
uint64_t g_cnt_hist[4] = {0, 0, 0, 0};
// IK2_STATS: a request's experts by where they came from: used and already cached / used and prefetched for this
// pass / used and copied after the routing / prefetched and not used
uint64_t g_src_hist[4] = {0, 0, 0, 0};
extern "C" void on_usr1(int) { g_cancel = 1; }

// Speculative decoding: g_taps [8][g_taps_ld][C] gets the drafter's taps of every position a forward call runs (the
// stream at the start of layers 5, 11, .., 41, the final normed hidden); with g_rec each short conv's history and
// inputs are kept, so that a verify window's rejected part can be taken back (rollback()).
float* g_taps = nullptr;
int g_taps_ld = 0, g_final_tap = 7;   // (the final normed hidden: tap 7, a looped model's 3)
bool g_rec = false;
// the drafter's tap a visit's stream (at its start) is, or -1
inline int draft_tap(const Config& c, int l, int vis) {
    if (c.n_pass == 1) return (l % 6 == 5 && l < c.n_layer - 1) ? l / 6 : -1;
    return vis == 5 ? 0 : vis == 23 ? 1 : vis == 41 ? 2 : -1;
}
void conv_rec(Layer& Ly, int which, const float* x, const float* hist, int T, int Cc) {
    if (!g_rec) return;
    CU(cudaMemcpyAsync(Ly.rec_old[which], hist, (size_t) 3 * Cc * 4, cudaMemcpyDeviceToDevice, g_s));
    CU(cudaMemcpyAsync(Ly.rec_x[which], x, (size_t) T * Cc * 4, cudaMemcpyDeviceToDevice, g_s));
}

// The pictures of the request running (GENI): the image encoder's rows (maya-vision's SVE1 records, [n][C] F32), each
// in place of one image token of the prompt - the model reads them as they are, without the embedding's norm.
struct Pictures {
    std::vector<float> rows;
    std::vector<int> row_at;   // a prompt position's row (-1: a token)
    int n = 0;
    void clear() { rows.clear(), row_at.clear(), n = 0; }
    int at(int pos) const { return n && pos < (int) row_at.size() ? row_at[pos] : -1; }
};
Pictures g_img;
// the rows of `path` for the image tokens of ids in order; sig[i]: a hash of position i's row (0: a token), the
// kept state's key beside the ids.  -> an error's text, or empty
std::string load_pictures(const std::string& path, const std::vector<int>& ids, int image_token, int C, Pictures& P,
                          std::vector<uint64_t>& sig) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot read the pictures' file";
    for (;;) {
        int32_t h[5];
        if (std::fread(h, 4, 5, f) != 5) break;
        if (h[0] != 0x31455653 || h[1] <= 0 || h[4] != C) {
            std::fclose(f);
            return "the pictures' file is not the image encoder's for this model";
        }
        const size_t at = P.rows.size(), n = (size_t) h[1] * C;
        P.rows.resize(at + n);
        if (std::fread(P.rows.data() + at, 4, n, f) != n) {
            std::fclose(f);
            return "the pictures' file is cut short";
        }
    }
    std::fclose(f);
    const int n_rows = (int) (P.rows.size() / C);
    P.row_at.assign(ids.size(), -1);
    int k = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != image_token) continue;
        if (k >= n_rows) return "the prompt has more image tokens than its pictures have rows";
        P.row_at[i] = k;
        uint64_t hsh = 1469598103934665603ull;   // FNV-1a over the row's bytes
        const uint8_t* b = (const uint8_t*) (P.rows.data() + (size_t) k * C);
        for (size_t j = 0; j < (size_t) C * 4; ++j) hsh = (hsh ^ b[j]) * 1099511628211ull;
        sig[i] = hsh | 1;
        ++k;
    }
    if (k != n_rows) return "the prompt's image tokens and its pictures' rows do not match";
    P.n = n_rows;
    return "";
}
// the pictures' rows of positions pos0 .. pos0 + T - 1 over their tokens' embeddings in h [T][C]
void place_pictures(float* h, int T, int pos0, int C, cudaStream_t s) {
    if (!g_img.n) return;
    for (int t = 0; t < T;) {
        const int r = g_img.at(pos0 + t);
        if (r < 0) {
            ++t;
            continue;
        }
        int n = 1;
        while (t + n < T && g_img.at(pos0 + t + n) == r + n) ++n;
        CU(cudaMemcpyAsync(h + (size_t) t * C, g_img.rows.data() + (size_t) r * C, (size_t) n * C * 4,
                           cudaMemcpyHostToDevice, s));
        t += n;
    }
}

// a GPU's own: its streams and cuBLAS handle, work buffers and expert cache; use_dev() makes it the current one (the
// globals g_s, g_cs, g_ev, g_blas the kernels and copies go to)
struct DevCtx {
    cudaStream_t s = nullptr, cs = nullptr;
    cudaEvent_t ev = nullptr;
    cublasHandle_t blas = nullptr;
    Work W{};
    ExpertCache EC;
};
std::vector<DevCtx> g_dev;
int g_cur = -1;
Work& dev_work(int d) { return g_dev[d].W; }
ExpertCache& dev_cache(int d) { return g_dev[d].EC; }
void use_dev(int d) {
    if (d == g_cur) return;
    if (g_stage) cudaFree(g_stage), g_stage = nullptr, g_stage_bytes = 0;   // (the previous device's)
    CU(cudaSetDevice(d));
    g_cur = d;
    g_s = g_dev[d].s, g_cs = g_dev[d].cs, g_ev = g_dev[d].ev, g_blas = g_dev[d].blas;
}

// (kCand: sampling's candidates - the resident token path finds the largest logits on the device, g_cand; any other
// path gives the logits as kLogits does and leaves g_cand.n at -1)
enum class OutMode { kNone, kArgmax, kLogits, kCand };
constexpr int kCandK = 1000, kCandCap = 4096;   // the largest logits sampling looks at; the candidates a token keeps
struct Cand {
    int n = -1;                    // candidates of the last kCand token (-1: its logits were given instead)
    const int* idx = nullptr;      // [n] token ids, in no order
    const float* val = nullptr;    // [n] their logits
} g_cand;

__global__ void gather_f16_16_kernel(const uint16_t* __restrict__ x16, const int* __restrict__ idx, int n, int C,
                                     uint16_t* __restrict__ y16) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const uint4* src = (const uint4*) (x16 + (size_t) idx[row] * C);
    uint4* dst = (uint4*) (y16 + (size_t) row * C);
    const int n4 = C >> 3;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dst[i] = src[i];
}

// out[idx[row]] += wr[row] * y[row] over rows (one expert's tokens: each token at most once, so no two rows collide)
__global__ void scatter_add_rows_kernel(const float* __restrict__ y, const int* __restrict__ idx,
                                        const float* __restrict__ wr, int C, float* __restrict__ out, int ld) {
    const int row = blockIdx.x;
    const float w = wr[row];
    const float4* src = (const float4*) (y + (size_t) row * C);
    float4* dst = (float4*) (out + (size_t) idx[row] * ld);
    for (int i = threadIdx.x; i < (C >> 2); i += blockDim.x) {
        const float4 a = src[i];
        float4 b = dst[i];
        b.x += w * a.x, b.y += w * a.y, b.z += w * a.z, b.w += w * a.w;
        dst[i] = b;
    }
}

// Y [T] rows (ldy) = rows [r0, r0 + n) of a dense matrix over X: a fused projection's part that takes its own input
Mat mat_rows(const Mat& m, int r0, int n) {   // the rows [r0, r0 + n) of a matrix
    Mat p = m;
    p.N = n;
    if (m.f16) p.f16 = m.f16 + (size_t) r0 * m.K;
    else if (m.q8) p.q8 = m.q8 + (size_t) r0 * m.K, p.q8d = m.q8d + (size_t) r0 * (m.K >> 5);
    else p.raw = m.raw + (size_t) r0 * (m.K / k::block_values(m.type)) * k::block_bytes(m.type);
    return p;
}
void matmul_rows(const Mat& m, int r0, int n, const float* x32, const uint16_t* x16, int T, float* Y, int ldy) {
    matmul(mat_rows(m, r0, n), x32, x16, T, Y, ldy);
}

// A looped layer's message gates over a chunk of tokens.  A gate is two matrix products - g = D [rms(x); rms(m)],
// then F g and G g - and k::loop_gates' kernels take them a token at a time: over a prefill chunk that was a quarter
// of the prefill.  Here the products go through cuBLAS and the kernels are what is left around them.
constexpr int kLoopGemmMin = 32;   // tokens from which the products are cuBLAS's
void loop_gates_chunk(Work& W, const k::LoopGate* gt, int n, int T, int C, int R, float eps) {
    if (T < kLoopGemmMin) {
        k::loop_gates(gt, n, T, C, R, eps, W.lp_g, g_s);
        return;
    }
    const float one = 1.0f, zero = 0.0f;
    float *a1 = W.lp_a, *a2 = a1 + (size_t) T * R, *g = a2 + (size_t) T * R;
    float *f = W.lp_f, *gn = f + (size_t) T * C;
    for (int j = 0; j < n; ++j) {
        // a1 [T][R] = x D_x^T, a2 = m D_m^T (D [R][2 C]: its rows' halves)
        CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, C, &one, gt[j].D, 2 * C, gt[j].x, C, &zero, a1, R));
        CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, C, &one, gt[j].D + C, 2 * C, gt[j].m, C, &zero, a2, R));
        k::loop_gate_mix(gt[j].x, gt[j].m, a1, a2, T, C, R, eps, g, g_s);
        // f [T][C] = g F, gn = g G (F, G [R][C])
        CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, C, T, R, &one, gt[j].F, C, g, R, &zero, f, C));
        CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, C, T, R, &one, gt[j].G, C, g, R, &zero, gn, C));
        k::loop_gate_blend(gt[j].x, gt[j].m, f, gn, (size_t) T * C, gt[j].y, g_s);
    }
}

// One layer's attention + post-attention conv + residual into r_out ([T][C]).
// h_in ([T][C]) is unmodified and acts as the residual input; when l % c.block == 0, it is saved to blk(nb).
// msg: a looped core's later pass - the layer's messages [4][T][C] (q's, k's, v's, the MoE's), which gate its q, k and
// v inputs before their norms (so a sliding layer too projects each from its own input).
void run_layer_attn(const Model& M, Layer& Ly, int l, int T, int pos0, const float* h_in, const int* pos_p, int nb,
                    size_t blk_off, float* r_out, Work& W, const float* msg = nullptr) {
    const Config& c = M.c;
    const int C = c.C;
    const size_t TC = (size_t) T * C;
    auto blk = [&](int i) { return W.blocks + ((size_t) i * W.blk_ld + blk_off) * C; };

    const bool need_x32 = g_dense_q8 && T == 1;
    const uint16_t* xk16 = W.x16;
    const uint16_t* xv16 = W.x16;
    const float *in_q = h_in, *in_k = h_in, *in_v = h_in;
    if (msg) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = h_in;
        k::attn_res(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, W.lp_in, g_s);
        const float* nw[3] = {Ly.sliding ? Ly.attn_norm : Ly.q_in_norm, Ly.sliding ? Ly.attn_norm : Ly.k_in_norm,
                              Ly.sliding ? Ly.attn_norm : Ly.v_in_norm};
        uint16_t* y16[3] = {W.x16, W.xk16, W.xv16};
        float* y32[3] = {W.xq, W.xk, W.xv};
        k::LoopGate gt[3];
        for (int j = 0; j < 3; ++j)
            gt[j] = {W.lp_in + (size_t) (Ly.n_res_q == 3 ? j : 0) * TC, msg + (size_t) j * TC, Ly.lp_down[j],
                     Ly.lp_forget[j], Ly.lp_gain[j], y32[j]};
        loop_gates_chunk(W, gt, 3, T, C, c.rank, 1e-6f);
        for (int j = 0; j < 3; ++j)
            k::rms_f16(y32[j], nw[j], T, C, c.eps, y16[j], need_x32 ? y32[j] : nullptr, g_s);
        in_q = W.xq, in_k = W.xk, in_v = W.xv;
        xk16 = W.xk16, xv16 = W.xv16;
    } else if (l > 0) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = h_in;
        k::RmsOuts outs{};
        if (Ly.sliding) {
            outs.w[0] = Ly.attn_norm;
            outs.y16[0] = W.x16;
            outs.y32[0] = need_x32 ? W.xq : nullptr;
            k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, c.eps, outs, g_s);
            in_q = in_k = in_v = W.xq;
        } else {
            outs.w[0] = Ly.q_in_norm, outs.w[1] = Ly.k_in_norm, outs.w[2] = Ly.v_in_norm;
            outs.y16[0] = W.x16, outs.y16[1] = W.xk16, outs.y16[2] = W.xv16;
            if (need_x32) outs.y32[0] = W.xq, outs.y32[1] = W.xk, outs.y32[2] = W.xv;
            k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, c.eps, outs, g_s);
            in_q = W.xq, in_k = W.xk, in_v = W.xv;
            xk16 = W.xk16, xv16 = W.xv16;
        }
    } else if (Ly.sliding) {
        k::rms_f16(in_q, Ly.attn_norm, T, C, c.eps, W.x16, need_x32 ? W.xq : nullptr, g_s);
        in_q = in_k = in_v = W.xq;
    } else {
        k::rms_f16(in_q, Ly.q_in_norm, T, C, c.eps, W.x16, need_x32 ? W.xq : nullptr, g_s);
        k::rms_f16(in_k, Ly.k_in_norm, T, C, c.eps, W.xk16, need_x32 ? W.xk : nullptr, g_s);
        k::rms_f16(in_v, Ly.v_in_norm, T, C, c.eps, W.xv16, need_x32 ? W.xv : nullptr, g_s);
        in_q = W.xq, in_k = W.xk, in_v = W.xv;
        xk16 = W.xk16, xv16 = W.xv16;
    }
    const int HD = c.H * c.D, KD = c.KV * c.D, HR = c.H * c.d_rel;
    float *q_p = W.q, *k_p = W.kk, *v_p = W.vv, *r_p = W.rr, *gate_p = W.gate, *sink_p = W.sink;
    float *kd_p = W.dyn, *vd_p = W.dyn;
    if (Ly.sliding) {
        if (Ly.wqkvrg.N > 0) {
            if (msg) {   // the fused rows by the input they take: q | k | v | r, gate (and the sink)
                const int N = Ly.wqkvrg.N;
                matmul_rows(Ly.wqkvrg, 0, HD, in_q, W.x16, T, W.qkv, N);
                matmul_rows(Ly.wqkvrg, HD, KD, in_k, xk16, T, W.qkv + HD, N);
                matmul_rows(Ly.wqkvrg, HD + KD, KD, in_v, xv16, T, W.qkv + HD + KD, N);
                matmul_rows(Ly.wqkvrg, HD + 2 * KD, N - HD - 2 * KD, in_q, W.x16, T, W.qkv + HD + 2 * KD, N);
            } else {
                matmul(Ly.wqkvrg, in_q, W.x16, T, W.qkv, Ly.wqkvrg.N);
            }
            const bool has_sink = !g_dense_q8;
            if (T == 1) {
                q_p = W.qkv;
                k_p = q_p + HD;
                v_p = k_p + KD;
                r_p = v_p + KD;
                gate_p = r_p + HR;
                if (has_sink) sink_p = gate_p + HD;
            } else {
                if (has_sink)
                    unpack_segments(W.qkv, T,
                                    {{W.q, HD}, {W.kk, KD}, {W.vv, KD}, {W.rr, HR}, {W.gate, HD}, {W.sink, c.H}}, g_s);
                else
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.kk, KD}, {W.vv, KD}, {W.rr, HR}, {W.gate, HD}}, g_s);
            }
            if (!has_sink) gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        } else {
            matmul(Ly.wq, in_q, W.x16, T, q_p, HD);
            matmul(Ly.wk, in_k, xk16, T, k_p, KD);
            matmul(Ly.wv, in_v, xv16, T, v_p, KD);
            matmul(Ly.wr, in_q, W.x16, T, r_p, HR);
            matmul(Ly.wgate, in_q, W.x16, T, gate_p, HD);
            gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        }
        conv_rec(Ly, 0, k_p, Ly.h_k, T, KD);
        conv_rec(Ly, 1, v_p, Ly.h_v, T, KD);
        k::conv4(k_p, Ly.h_k, Ly.sc_k, nullptr, 0, T, KD, W.kconv, g_s);
        k::conv4(v_p, Ly.h_v, Ly.sc_v, nullptr, 0, T, KD, W.vconv, g_s);
    } else {
        if (Ly.wqrg.N > 0) {
            matmul(Ly.wqrg, in_q, W.x16, T, W.qkv, Ly.wqrg.N);
            const bool has_sink = !g_dense_q8;
            if (T == 1) {
                q_p = W.qkv;
                r_p = q_p + HD;
                gate_p = r_p + HR;
                if (has_sink) sink_p = gate_p + HD;
            } else {
                if (has_sink)
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.rr, HR}, {W.gate, HD}, {W.sink, c.H}}, g_s);
                else
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.rr, HR}, {W.gate, HD}}, g_s);
            }
            if (!has_sink) gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        } else {
            matmul(Ly.wq, in_q, W.x16, T, q_p, HD);
            matmul(Ly.wr, in_q, W.x16, T, r_p, HR);
            matmul(Ly.wgate, in_q, W.x16, T, gate_p, HD);
            gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        }
        if (!g_dense_q8) {
            const int off_k = T == 1 ? Ly.wqrg.N : 0;
            const int off_v = T == 1 ? Ly.wqrg.N + Ly.wk_dyn.N : 0;
            matmul(Ly.wk_dyn, in_k, xk16, T, W.qkv + off_k, Ly.wk_dyn.N);
            if (T == 1) {
                k_p = W.qkv + off_k;
                kd_p = k_p + KD;
            } else {
                unpack_segments(W.qkv + off_k, T, {{W.kk, KD}, {W.dyn, c.KV * 4}}, g_s);
            }
            conv_rec(Ly, 0, k_p, Ly.h_k, T, KD);
            k::conv4(k_p, Ly.h_k, Ly.sc_k, kd_p, c.KV, T, KD, W.kconv, g_s);

            matmul(Ly.wv_dyn, in_v, xv16, T, W.qkv + off_v, Ly.wv_dyn.N);
            if (T == 1) {
                v_p = W.qkv + off_v;
                vd_p = v_p + KD;
            } else {
                unpack_segments(W.qkv + off_v, T, {{W.vv, KD}, {W.dyn, c.KV * 4}}, g_s);
            }
            conv_rec(Ly, 1, v_p, Ly.h_v, T, KD);
            k::conv4(v_p, Ly.h_v, Ly.sc_v, vd_p, c.KV, T, KD, W.vconv, g_s);
        } else {
            matmul(Ly.wk, in_k, xk16, T, k_p, KD);
            gemm16(Ly.sc_k_dyn16, c.KV * 4, C, xk16, T, kd_p, c.KV * 4);
            conv_rec(Ly, 0, k_p, Ly.h_k, T, KD);
            k::conv4(k_p, Ly.h_k, Ly.sc_k, kd_p, c.KV, T, KD, W.kconv, g_s);

            matmul(Ly.wv, in_v, xv16, T, v_p, KD);
            gemm16(Ly.sc_v_dyn16, c.KV * 4, C, xv16, T, vd_p, c.KV * 4);
            conv_rec(Ly, 1, v_p, Ly.h_v, T, KD);
            k::conv4(v_p, Ly.h_v, Ly.sc_v, vd_p, c.KV, T, KD, W.vconv, g_s);
        }
    }
    const float k_scale = c.qk_clip <= 28.0f ? 16.0f : 1.0f;
    const float v_scale = 32.0f;
    const float k_inv = 1.0f / k_scale, v_inv = 1.0f / v_scale;
    if (g_kv_fp8) {
        k::prep_qkv_r_fp8(q_p, Ly.q_clip, W.kconv, Ly.kc8, W.vconv, Ly.vc8, r_p, Ly.r_clip, T, c.H, c.KV, c.D, c.d_rel,
                          c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha, c.n_floor, pos0,
                          Ly.kv_cap, k_scale, v_scale, g_s);
    } else {
        k::prep_qkv_r_f16(q_p, Ly.q_clip, W.kconv, Ly.kc, W.vconv, Ly.vc, r_p, Ly.r_clip, T, c.H, c.KV, c.D, c.d_rel,
                          c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha, c.n_floor, pos0,
                          Ly.kv_cap, g_s);
    }

    const int extent = Ly.sliding ? c.extent_swa : c.extent;
    const int window = Ly.sliding ? c.window : (1 << 30);
    const int k0 = Ly.sliding ? std::max(0, pos0 - c.window + 1) : 0;
    const int nk = pos0 + T - k0;
    const float one = 1.0f, zero = 0.0f;
    const float qk_scale = 1.0f / std::sqrt((float) c.D);
    const int attn_kv_cap = Ly.sliding ? Ly.kv_cap : 0;
    if (T == 1) {
        if (g_kv_fp8) {
            k::attn_decode_fp8(q_p, r_p, sink_p, Ly.kc8, Ly.vc8, attn_kv_cap, c.H, c.KV, c.D, k0, nk, pos0,
                               qk_scale * k_inv, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else {
            k::attn_decode_f16(q_p, r_p, sink_p, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, c.D, k0, nk, pos0, qk_scale,
                               Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        }
    } else if (T <= 8 && g_group && g_group_attn) {
        if (g_kv_fp8) {
            k::attn_multi_fp8(q_p, r_p, sink_p, Ly.kc8, Ly.vc8, attn_kv_cap, T, c.H, c.KV, k0, nk, pos0,
                              qk_scale * k_inv, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else {
            k::attn_multi_f16(q_p, r_p, sink_p, Ly.kc, Ly.vc, attn_kv_cap, T, c.H, c.KV, k0, nk, pos0, qk_scale,
                              Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        }
    } else {
        k::to_f16(q_p, (size_t) T * HD, W.x16, g_s);
        uint16_t* lg16 = W.lg16;
        const int GQA = c.H / c.KV;
        // softmax rows (with the relative bias and the sink) -> lg16: the bias as one small GEMM (K = d_rel) first
        auto attn_probs = [&](int Tq, int nk_s, int ts, int k0_s) {
            if (W.relb && extent <= W.relb_ext) {
                const float one_f = 1.0f, zero_f = 0.0f;
                CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, extent, Tq * c.H, c.d_rel, &one_f, Ly.rel_proj,
                                CUDA_R_32F, c.d_rel, r_p + (size_t) ts * HR, CUDA_R_32F, c.d_rel, &zero_f, W.relb,
                                CUDA_R_32F, extent, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                k::attn_probs_bias_f16(W.lg, lg16, c.H, Tq, nk_s, pos0 + ts, k0_s, qk_scale, W.relb, extent, window,
                                       sink_p + (size_t) ts * c.H, g_s);
            } else {
                k::attn_probs_f16(W.lg, lg16, c.H, Tq, nk_s, pos0 + ts, k0_s, qk_scale, r_p + (size_t) ts * HR,
                                  c.d_rel, Ly.rel_proj, extent, window, sink_p + (size_t) ts * c.H, g_s);
            }
        };
        if (g_flash && c.D == 128 && W.relb && extent <= W.relb_ext) {
            if (!g_n_sm) CU(cudaDeviceGetAttribute(&g_n_sm, cudaDevAttrMultiProcessorCount, Ly.dev));
            for (int ts = 0; ts < T; ts += Work::kRelbRows) {
                const int Tq = std::min(Work::kRelbRows, T - ts);
                const float one_f = 1.0f, zero_f = 0.0f;
                CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, extent, Tq * c.H, c.d_rel, &one_f, Ly.rel_proj,
                                CUDA_R_32F, c.d_rel, r_p + (size_t) ts * HR, CUDA_R_32F, c.d_rel, &zero_f, W.relb,
                                CUDA_R_32F, extent, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                const int k_lo = Ly.sliding ? std::max(0, pos0 + ts - c.window + 1) : 0;
                const int ns = k::attn_flash_splits(Tq, c.H, pos0 + ts + Tq - k_lo, g_n_sm);
                if (g_kv_fp8)
                    k::attn_flash_fp8(W.x16 + (size_t) ts * HD, HD, Ly.kc8, Ly.vc8, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                      pos0 + ts, qk_scale * k_inv, v_inv, W.relb, extent, window,
                                      sink_p + (size_t) ts * c.H, W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
                else
                    k::attn_flash_f16(W.x16 + (size_t) ts * HD, HD, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                      pos0 + ts, qk_scale, W.relb, extent, window, sink_p + (size_t) ts * c.H,
                                      W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
            }
        } else
        for (int ts = 0; ts < T;) {
            const int k0_s = Ly.sliding ? std::max(0, pos0 + ts - c.window + 1) : 0;
            const int nk_ub = pos0 + T - k0_s;
            int Tq = std::max(1, std::min(T - ts, (int) (W.lg_cap / ((size_t) c.H * nk_ub))));
            if (W.relb) Tq = std::min(Tq, Work::kRelbRows);
            const int nk_s = pos0 + ts + Tq - k0_s;
            const long long qn = (long long) Tq * nk_s;
            if (g_kv_fp8) {
                for (int k_off = 0; k_off < nk_s; k_off += W.kv_tmp_cap) {
                    const int nk_c = std::min(W.kv_tmp_cap, nk_s - k_off);
                    k::unpack_kv_fp8_to_f16(Ly.kc8, Ly.vc8, attn_kv_cap, k0_s + k_off, nk_c, KD, k_inv, v_inv, W.kv_k16,
                                            W.kv_v16, g_s);
                    for (int g = 0; g < c.KV; ++g) {
                        const uint16_t* Kp = W.kv_k16 + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_T, CUBLAS_OP_N, nk_c, Tq, c.D, &one, Kp, CUDA_R_16F, KD, 0,
                            W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                            W.lg + (size_t) g * GQA * qn + k_off, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                }
                attn_probs(Tq, nk_s, ts, k0_s);
                for (int k_off = 0; k_off < nk_s; k_off += W.kv_tmp_cap) {
                    const int nk_c = std::min(W.kv_tmp_cap, nk_s - k_off);
                    if (nk_s > W.kv_tmp_cap) {
                        k::unpack_kv_fp8_to_f16(Ly.kc8, Ly.vc8, attn_kv_cap, k0_s + k_off, nk_c, KD, k_inv, v_inv,
                                                W.kv_k16, W.kv_v16, g_s);
                    }
                    const float beta = k_off == 0 ? zero : one;
                    for (int g = 0; g < c.KV; ++g) {
                        const uint16_t* Vp = W.kv_v16 + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, nk_c, &one, Vp, CUDA_R_16F, KD, 0,
                            lg16 + (size_t) g * GQA * qn + k_off, CUDA_R_16F, nk_s, qn, &beta,
                            W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    }
                }
            } else {
                const int start = attn_kv_cap > 0 ? (k0_s % attn_kv_cap) : k0_s;
                const int n1 = attn_kv_cap > 0 ? std::min(nk_s, attn_kv_cap - start) : nk_s;
                const int n2 = nk_s - n1;
                for (int g = 0; g < c.KV; ++g) {
                    const uint16_t* Kp1 = Ly.kc + (size_t) start * KD + (size_t) g * c.D;
                    CB(cublasGemmStridedBatchedEx(
                        g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n1, Tq, c.D, &one, Kp1, CUDA_R_16F, KD, 0,
                        W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                        W.lg + (size_t) g * GQA * qn, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT));
                    if (n2 > 0) {
                        const uint16_t* Kp2 = Ly.kc + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n2, Tq, c.D, &one, Kp2, CUDA_R_16F, KD, 0,
                            W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                            W.lg + (size_t) g * GQA * qn + n1, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                }
                attn_probs(Tq, nk_s, ts, k0_s);
                for (int g = 0; g < c.KV; ++g) {
                    const uint16_t* Vp1 = Ly.vc + (size_t) start * KD + (size_t) g * c.D;
                    CB(cublasGemmStridedBatchedEx(
                        g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, n1, &one, Vp1, CUDA_R_16F, KD, 0,
                        lg16 + (size_t) g * GQA * qn, CUDA_R_16F, nk_s, qn, &zero,
                        W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    if (n2 > 0) {
                        const uint16_t* Vp2 = Ly.vc + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, n2, &one, Vp2, CUDA_R_16F, KD, 0,
                            lg16 + (size_t) g * GQA * qn + n1, CUDA_R_16F, nk_s, qn, &one,
                            W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    }
                }
            }
            ts += Tq;
        }
    }
    k::gate_mul_f16(W.o, gate_p, (size_t) T * HD, W.x16, g_dense_q8 && T == 1, g_s);
    matmul(Ly.wo, W.o, W.x16, T, W.tmp, C);
    conv_rec(Ly, 2, W.tmp, Ly.h_attn, T, C);
    if (l % c.block == 0) {
        CU(cudaMemcpyAsync(blk(nb), h_in, TC * 4, cudaMemcpyDeviceToDevice, g_s));
        k::conv4(W.tmp, Ly.h_attn, Ly.sc_attn, nullptr, 0, T, C, r_out, g_s);
    } else {
        k::conv4_res(W.tmp, Ly.h_attn, Ly.sc_attn, nullptr, 0, T, C, h_in, r_out, g_s);
    }
}

// ---- the offload token path (ik2_off.cuh): the resident path's launches where the experts do not all fit - a tier
// of them in VRAM (the expert cache's slots: Ly.tab_blob, null where an expert is not there) and the rest copied by
// the GPU from pinned host memory into staging slots after the router, the same expert kernels on the same bytes
bool g_off = false;        // set at load
bool g_off_want = false;   // the layout chosen (the fit, IK2_OFF)
struct OffState {
    std::vector<const uint8_t*> host;   // [n_layer] the layer's pinned experts as the device sees them
    std::vector<int> pool;              // [n_layer] its staging pool (the cache's pool of its slot size)
    std::vector<k::OffStage> stage;     // a pool a slot size
    // (prompts: the staging slots as a ring - an event a slot for its copy, one for the expert that read it)
    std::vector<std::vector<cudaEvent_t>> ring_free, ring_full;
    std::vector<int> ring_hand;
    k::OffFetch* fl = nullptr;
    k::OffCount* cnt = nullptr;         // (device)
    uint32_t* heat = nullptr;           // [n_layer * E] (device): picks since load
    int blocks = 0;                     // off_fetch_copy's grid
    uint64_t tick = ~0ull;              // the cache's tick the tables were made at (a prompt moves experts)
    // the lane-parity mode (IK2_LANE): the CPU lane's mailboxes (mapped host memory) and the split's figures
    std::vector<int> fmt;               // [n_layer] 0 NVFP4, 1 Q8_0
    k::LaneCost* cost = nullptr;        // (device)
    k::LaneCost cost_h{};               // (pinned host: what was last sent)
    k::LaneState* ls = nullptr;         // (device)
    int* cls = nullptr;                 // (device) [K]
    float* hx = nullptr;                // (device) [K][F] the exact lane's h
    k::LaneJob* job_d = nullptr;        // (the GPU's view of the mailboxes)
    k::LaneDone* done_d = nullptr;
    void *job_h = nullptr, *done_h = nullptr;
    std::unique_ptr<ik2::lane::Server> lane;
};
OffState g_offs;
bool g_lane = false;   // IK2_LANE: the offload path's lane-parity mode (set at load)

// A layer-major prefill pass of many tokens uses nearly every expert of a layer, so the ones not in VRAM can be
// copied in before the layer's routing is known: while the previous layer's experts and this layer's attention run.
// The copy stream then has no idle gap between layers. Slots are taken at the current tick, which keeps the previous
// layer's slots (still being computed) from being evicted; only done when the pool holds both layers with room.
void prefetch_layer(Model& M, int l, Work& W, ExpertCache& EC, int T, const Work* Wpop = nullptr, int own = -1, int ahead = 1) {
    const int pr = ((l % Work::kPfRing) + Work::kPfRing) % Work::kPfRing;
    W.pf_layers[pr] = -1;
    if (g_off || (getenv("IK2_PREFETCH") && !std::atoi(getenv("IK2_PREFETCH")))) return;   // (offload: the ring)
    if (!g_overlap || g_prof || l >= M.c.n_layer || (!dual() && M.L[l].dev != g_cur)) return;
    const Work& U = Wpop ? *Wpop : W;   // (whose picks so far: under EP, GPU 0's)
    const int E = M.c.E, pi = EC.layer_pool[l];
    if (pi < 0 || EC.pools[pi].n_active < (ahead + 1) * E + 64) return;
    Layer& Ly = M.L[l];
    const uint64_t t = EC.tick + (uint64_t) ahead;   // (the tick its layer will run in: kept till then)
    W.pf_bs[pr].assign(E, -1);
    // Only the experts this pass will likely use: P(used) = 1 - (1 - p_e)^(T K) from the prefills so far (p_e the
    // expert's share of the layer's tokens); with little history yet, all of them (IK2_PF_P: the threshold)
    const bool known = !U.use_tot.empty() && U.use_tot[l] >= (uint64_t) 16 * E;
    // (0.999, not 0.5: a request's tokens are alike, so the independent-picks estimate is far too high - at 0.5 a
    // fifth of the prefetched experts went unused; 500 / 800 new tokens 4.2 / 4.3 -> 3.5 / 3.6 s, 21.8k unchanged)
    const double thr = getenv("IK2_PF_P") ? std::atof(getenv("IK2_PF_P")) : 0.999;
    const double picks = (double) T * M.c.K;
    int n = 0;
    for (int e = 0; e < E; ++e) {
        if (EC.has(l, e) || (own >= 0 && ep_owner(l, e) != own)) continue;
        if (known) {
            const double pe = (U.use[(size_t) l * E + e] + 0.5) / (U.use_tot[l] + 0.5 * E);
            if (1.0 - std::pow(1.0 - pe, picks) < thr) continue;
        }
        uint8_t* dst = EC.get(Ly, l, e, t, g_cs, nullptr, false, true);
        EC.copy_in(Ly, e, dst, g_cs);
        W.pf_bs[pr][e] = n / Work::kMissBatch;
        if (++n % Work::kMissBatch == 0) CU(cudaEventRecord(W.pf_evs[pr][n / Work::kMissBatch - 1], g_cs));
    }
    if (n % Work::kMissBatch) CU(cudaEventRecord(W.pf_evs[pr][n / Work::kMissBatch], g_cs));
    W.pf_layers[pr] = l;
}

#ifdef STRATA_USE_HIP
// AMD, one GPU: the grouped prompt experts' buffers (k::experts_rdna_*) - the rows' inputs, hidden and outputs, the
// tiles and the experts' pointers (their host copies by layer parity, as Work's gidx_h); grown to the largest chunk
struct RdnaMoe {
    size_t rows = 0;
    int E = 0;
    uint16_t* x16 = nullptr;
    uint16_t* h16 = nullptr;
    float* y32 = nullptr;
    int* tiles = nullptr;
    const uint8_t** eptr = nullptr;
    int* tiles_h[2] = {};
    const uint8_t** eptr_h[2] = {};
};
RdnaMoe& rdna_moe(int S, int E, int C, int F) {
    static RdnaMoe R;
    if ((size_t) S > R.rows || E > R.E) {
        CU(cudaDeviceSynchronize());   // (only when a larger chunk comes: the old buffers may still be read)
        if (R.rows) {
            CU(cudaFree(R.x16));
            CU(cudaFree(R.h16));
            CU(cudaFree(R.y32));
            CU(cudaFree(R.tiles));
            CU(cudaFree((void*) R.eptr));
            for (int i = 0; i < 2; ++i) {
                CU(cudaFreeHost(R.tiles_h[i]));
                CU(cudaFreeHost((void*) R.eptr_h[i]));
            }
        }
        R.rows = std::max<size_t>(R.rows, (size_t) S), R.E = std::max(R.E, E);
        const size_t nt = R.rows / 32 + (size_t) R.E + 1;   // (tiles: up to 32 rows of one expert each)
        R.x16 = dalloc<uint16_t>(R.rows * C), R.h16 = dalloc<uint16_t>(R.rows * F), R.y32 = dalloc<float>(R.rows * C);
        R.tiles = dalloc<int>(nt * 4);
        CU(cudaMalloc((void**) &R.eptr, (size_t) R.E * sizeof(void*)));
        for (int i = 0; i < 2; ++i) {
            CU(cudaHostAlloc((void**) &R.tiles_h[i], nt * 16, cudaHostAllocDefault));
            CU(cudaHostAlloc((void**) &R.eptr_h[i], (size_t) R.E * sizeof(void*), cudaHostAllocDefault));
        }
    }
    return R;
}
#endif

void run_prefill_moe(const Model& M, Layer& Ly, int l, int T, const uint16_t* xq16, const int* top, const float* w,
                     const float* r_in, float* h_out, Work& W, ExpertCache& EC, double& t_exp) {
    const Config& c = M.c;
    const int C = c.C, F = c.F, S = T * c.K;
    const double te0 = now();
    if (g_seq_done + 1 < g_seq_next) {
        CU(cudaStreamSynchronize(g_cs));
        g_seq_done = g_seq_next - 1;
    }
    std::fill(W.cnt.begin(), W.cnt.end(), 0);
    for (int i = 0; i < S; ++i) ++W.cnt[top[i]];
    if (W.use.empty()) W.use.assign((size_t) c.n_layer * c.E, 0), W.use_tot.assign(c.n_layer, 0);
    const int prl = l % Work::kPfRing;
    const bool pf_here = W.pf_layers[prl] == l && (int) W.pf_bs[prl].size() == c.E;
    for (int e = 0; e < c.E; ++e) {
        W.use[(size_t) l * c.E + e] += W.cnt[e];
        const int k = W.cnt[e];
        if (k) ++g_cnt_hist[k == 1 ? 0 : k == 2 ? 1 : k < 8 ? 2 : 3];
        const bool pfd = pf_here && W.pf_bs[prl][e] >= 0;
        if (k) ++g_src_hist[pfd ? 1 : EC.has(l, e) ? 0 : 2];
        else if (pfd) ++g_src_hist[3];
    }
    W.use_tot[l] += S;
    W.start[0] = 0;
    for (int e = 0; e < c.E; ++e) W.start[e + 1] = W.start[e] + W.cnt[e];
    for (int e = 0; e < c.E; ++e) W.fill[e] = W.start[e];

    const int ps = l & 1;
    int* gidx = W.gidx_h[ps];
    int* inv = W.inv_h[ps];
    float* sg = W.sg_h[ps];
    float* su = W.su_h[ps];
    float* wt = W.wt_h[ps];
    float* wtr = W.wtr_h[ps];
    for (int i = 0; i < S; ++i) {
        const int e = top[i], row = W.fill[e]++;
        gidx[row] = i / c.K;
        inv[i] = row;
        sg[row] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
        su[row] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
        wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
        wtr[row] = wt[i];
    }

    // the experts on a GPU: d -1 all of them (one GPU), else the ones it owns (IK2_EP), into X.tmp
    auto exec = [&](int d, Work& X, ExpertCache& XC, const uint16_t* xin) {
        const uint64_t tick = g_off ? XC.tick : ++XC.tick;   // (the offload path: the tier is not touched)
        const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
        X.hit_exps.clear();
        X.miss_exps.clear();
        const double tu = now();
        if (g_off) {   // (the offload path: a hit read in place in the tier, the misses through the staging ring below)
            const auto& P = XC.pools[XC.layer_pool[l]];
            for (int e = 0; e < c.E; ++e) {
                X.slot_ptrs[e] = nullptr;
                if (!W.cnt[e]) continue;
                const int s = XC.loc[(size_t) l * c.E + e];
                if (s >= 0) X.slot_ptrs[e] = P.data + (size_t) s * P.slot_bytes, X.hit_exps.push_back(e);
                else X.miss_exps.push_back(e);
            }
            XC.hits += X.hit_exps.size(), XC.misses += X.miss_exps.size();
        } else {
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e] || (d >= 0 && ep_owner(l, e) != d)) {
                    X.slot_ptrs[e] = nullptr;
                    continue;
                }
                if (XC.has(l, e)) {
                    X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
                    X.hit_exps.push_back(e);
                } else {
                    X.slot_ptrs[e] = nullptr;
                    X.miss_exps.push_back(e);
                }
            }
            for (int e : X.miss_exps) X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
        }

        const int n_miss = (int) X.miss_exps.size();
        const int n_batches = g_off ? 0 : (n_miss + Work::kMissBatch - 1) / Work::kMissBatch;
        for (int b = 0; b < n_batches; ++b) {
            const int b0 = b * Work::kMissBatch, b1 = std::min(n_miss, b0 + Work::kMissBatch);
            for (int m = b0; m < b1; ++m) {
                const int e = X.miss_exps[m];
                XC.copy_in(Ly, e, X.slot_ptrs[e], cst);
            }
            if (cst != g_s) CU(cudaEventRecord(X.miss_ev[b], cst));
        }
        XC.t_up += now() - tu;
        g_pf.lap(g_pf.upload);

        CU(cudaMemcpyAsync(X.gidx, gidx, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.inv, inv, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.sg, sg, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.su, su, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.wtr, wtr, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemsetAsync(X.tmp, 0, (size_t) T * C * 4, g_s));

        // Each expert: its tokens gathered, gate/up, SwiGLU, down, then added into tmp with their weights. The experts run
        // in index order whatever is cached (each waiting for its copies), so the sums' order and the result do not
        // depend on the cache. Only [T][C] of scratch: no [T * K][C] copies of the inputs or the outputs.
        auto run_exp = [&](int e) {
            const int cnt_e = W.cnt[e], st_e = W.start[e];
            const uint8_t* sl = X.slot_ptrs[e];
            gather_f16_16_kernel<<<cnt_e, 256, 0, g_s>>>(xin, X.gidx + st_e, cnt_e, C, X.ex16);
            if (Ly.ty[0] == Ly.ty[1]) {
                k::decode_f16(Ly.ty[0], sl, (size_t) 2 * F * C, X.w16, g_s);
            } else {
                k::decode_f16(Ly.ty[0], sl, (size_t) F * C, X.w16, g_s);
                k::decode_f16(Ly.ty[1], sl + Ly.eb[0], (size_t) F * C, X.w16 + (size_t) F * C, g_s);
            }
            gemm16(X.w16, 2 * F, C, X.ex16, cnt_e, X.gu32, 2 * F);
            k::swiglu_cap_f32(X.gu32, cnt_e, F, X.sg + st_e, X.su + st_e, c.gate_cap, c.up_cap, X.h16, g_s);
            k::decode_f16(Ly.ty[2], sl + Ly.eb[0] + Ly.eb[1], (size_t) C * F, X.w16, g_s);
            gemm16(X.w16, C, F, X.h16, cnt_e, X.y32, C);
            scatter_add_rows_kernel<<<cnt_e, 256, 0, g_s>>>(X.y32, X.gidx + st_e, X.wtr + st_e, C, X.tmp, C);
        };
        if (g_off) {
            // The offload path: the misses copied into the staging ring of their slot size (the decode's staging
            // slots), up to a ring ahead of the experts that read them - a slot copied again only once the expert
            // before it there has run (its event); index order kept, so the sums do not depend on the tier.
            const int pi = g_offs.pool[l];
            const auto& st = g_offs.stage[pi];
            auto& rf = g_offs.ring_free[pi];
            auto& rc = g_offs.ring_full[pi];
            int issued = 0;
            X.miss_pos.assign(c.E, -1);
            auto issue = [&]() {
                const int e = X.miss_exps[issued++];
                const int r = g_offs.ring_hand[pi];
                g_offs.ring_hand[pi] = (r + 1) % st.n;
                uint8_t* dst = st.data + (size_t) r * st.sb;
                if (cst != g_s) CU(cudaStreamWaitEvent(cst, rf[r], 0));
                XC.copy_in(Ly, e, dst, cst);
                if (cst != g_s) CU(cudaEventRecord(rc[r], cst));
                X.slot_ptrs[e] = dst;
                X.miss_pos[e] = r;
            };
            while (issued < std::min(n_miss, st.n)) issue();
            XC.t_up += now() - tu;
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) continue;
                const int r = X.miss_pos[e];
                if (r >= 0 && cst != g_s) CU(cudaStreamWaitEvent(g_s, rc[r], 0));
                run_exp(e);
                if (r >= 0) {
                    if (cst != g_s) CU(cudaEventRecord(rf[r], g_s));
                    if (issued < n_miss) issue();
                }
            }
            if (n_miss) CU(cudaMemsetAsync(st.tag, 0xff, (size_t) st.n * 4, g_s));   // (no decode copy left there)
            return;
        }
        const bool pf = X.pf_layers[l % Work::kPfRing] == l;
#ifdef STRATA_USE_HIP
        if (d < 0 && k::experts_rdna_available(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, F)) {
            // (AMD matrix cores: all the layer's experts in one launch each for gate / up and down, then each token's
            //  picks summed in expert order - so every copy and prefetch they read is waited for first)
            if (pf) {
                int b = -1;
                for (int e = 0; e < c.E; ++e)
                    if (W.cnt[e]) b = std::max(b, X.pf_bs[l % Work::kPfRing][e]);
                if (b >= 0) CU(cudaStreamWaitEvent(g_s, X.pf_evs[l % Work::kPfRing][b], 0));
            }
            if (n_miss > 0 && cst != g_s) CU(cudaStreamWaitEvent(g_s, X.miss_ev[n_batches - 1], 0));
            RdnaMoe& R = rdna_moe(S, c.E, C, F);
            int* th = R.tiles_h[ps];
            int nt = 0;
            for (int e = 0; e < c.E; ++e) {
                R.eptr_h[ps][e] = W.cnt[e] ? X.slot_ptrs[e] : nullptr;
                for (int m0 = 0; m0 < W.cnt[e]; m0 += 32, ++nt)
                    th[4 * nt] = e, th[4 * nt + 1] = W.start[e] + m0, th[4 * nt + 2] = std::min(32, W.cnt[e] - m0),
                    th[4 * nt + 3] = 0;
            }
            CU(cudaMemcpyAsync(R.tiles, th, (size_t) nt * 16, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(R.eptr, R.eptr_h[ps], (size_t) c.E * sizeof(void*), cudaMemcpyHostToDevice, g_s));
            gather_f16_16_kernel<<<S, 256, 0, g_s>>>(xin, X.gidx, S, C, R.x16);
            k::experts_rdna_gu(R.eptr, R.tiles, nt, Ly.ty[0], Ly.eb[0], R.x16, C, F, X.sg, X.su, c.gate_cap,
                               c.up_cap, R.h16, g_s);
            k::experts_rdna_down(R.eptr, R.tiles, nt, Ly.ty[2], Ly.eb[0] + Ly.eb[1], R.h16, C, F, R.y32, g_s);
            k::moe_rows_combine(R.y32, X.inv, X.wtr, T, c.K, C, X.tmp, g_s);
            X.pf_layers[l % Work::kPfRing] = -1;
            return;
        }
#endif
        X.miss_pos.assign(c.E, -1);
        for (int m = 0; m < n_miss; ++m) X.miss_pos[X.miss_exps[m]] = m;
        int pf_waited = -1, miss_waited = -1;
        for (int e = 0; e < c.E; ++e) {
            if (!W.cnt[e] || (d >= 0 && ep_owner(l, e) != d)) continue;
            if (pf && X.pf_bs[l % Work::kPfRing][e] >= 0 && X.pf_bs[l % Work::kPfRing][e] > pf_waited)
                CU(cudaStreamWaitEvent(g_s, X.pf_evs[l % Work::kPfRing][pf_waited = X.pf_bs[l % Work::kPfRing][e]], 0));
            if (X.miss_pos[e] >= 0 && cst != g_s && X.miss_pos[e] / Work::kMissBatch > miss_waited)
                CU(cudaStreamWaitEvent(g_s, X.miss_ev[miss_waited = X.miss_pos[e] / Work::kMissBatch], 0));
            run_exp(e);
        }
        X.pf_layers[l % Work::kPfRing] = -1;
    };
    if (g_ep) {   // GPU 1's experts on GPU 1 (the input over, its sums back), GPU 0's meanwhile
        CU(cudaEventRecord(g_ep_x, g_s));
        use_dev(1);
        Work& X1 = dev_work(1);
        CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
        CU(cudaMemcpyPeerAsync(X1.xq16_sup, 1, xq16, 0, (size_t) T * C * 2, g_s));
        exec(1, X1, dev_cache(1), X1.xq16_sup);
        CU(cudaEventRecord(g_ep_done, g_s));
        use_dev(0);
        exec(0, W, EC, xq16);
        CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
        CU(cudaMemcpyPeerAsync(W.y32, 0, X1.tmp, 1, (size_t) T * C * 4, g_s));
        k::add(W.tmp, W.y32, (size_t) T * C, W.tmp, g_s);
    } else if (g_tp) {
        // tensor-parallel experts: (1) each GPU its half of every expert's h into tp_h (GPU 1 with the input copied
        // over), (2) the halves swapped, (3) each GPU its half of the output rows, GPU 1's columns copied to GPU 0
        const int Fd = F / 2, Cd = C / 2;
        auto tp_gu = [&](int d, Work& X, ExpertCache& XC, const uint16_t* xin) {
            const uint64_t tick = ++XC.tick;
            const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
            X.hit_exps.clear();
            X.miss_exps.clear();
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) {
                    X.slot_ptrs[e] = nullptr;
                    continue;
                }
                if (XC.has(l, e)) {
                    X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
                    X.hit_exps.push_back(e);
                } else {
                    X.slot_ptrs[e] = nullptr;
                    X.miss_exps.push_back(e);
                }
            }
            for (int e : X.miss_exps) X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
            const int n_miss = (int) X.miss_exps.size();
            const int n_batches = (n_miss + Work::kMissBatch - 1) / Work::kMissBatch;
            for (int b = 0; b < n_batches; ++b) {
                const int b0 = b * Work::kMissBatch, b1 = std::min(n_miss, b0 + Work::kMissBatch);
                for (int m = b0; m < b1; ++m) XC.copy_in(Ly, X.miss_exps[m], X.slot_ptrs[X.miss_exps[m]], cst);
                if (cst != g_s) CU(cudaEventRecord(X.miss_ev[b], cst));
            }
            CU(cudaMemcpyAsync(X.gidx, gidx, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.sg, sg, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.su, su, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.wtr, wtr, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            size_t o0, b0g, o1, b1u;
            tp_part(Ly, d, 0, o0, b0g);
            tp_part(Ly, d, 1, o1, b1u);
            const bool pf = X.pf_layers[l % Work::kPfRing] == l;
            X.miss_pos.assign(c.E, -1);
            for (int m = 0; m < n_miss; ++m) X.miss_pos[X.miss_exps[m]] = m;
            int pf_waited = -1, miss_waited = -1;
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) continue;
                if (pf && X.pf_bs[l % Work::kPfRing][e] >= 0 && X.pf_bs[l % Work::kPfRing][e] > pf_waited)
                    CU(cudaStreamWaitEvent(g_s, X.pf_evs[l % Work::kPfRing][pf_waited = X.pf_bs[l % Work::kPfRing][e]], 0));
                if (X.miss_pos[e] >= 0 && cst != g_s && X.miss_pos[e] / Work::kMissBatch > miss_waited)
                    CU(cudaStreamWaitEvent(g_s, X.miss_ev[miss_waited = X.miss_pos[e] / Work::kMissBatch], 0));
                const int cnt_e = W.cnt[e], st_e = W.start[e];
                const uint8_t* sl = X.slot_ptrs[e];
                gather_f16_16_kernel<<<cnt_e, 256, 0, g_s>>>(xin, X.gidx + st_e, cnt_e, C, X.ex16);
                if (Ly.ty[0] == Ly.ty[1]) {
                    k::decode_f16(Ly.ty[0], sl, (size_t) 2 * Fd * C, X.w16, g_s);
                } else {
                    k::decode_f16(Ly.ty[0], sl, (size_t) Fd * C, X.w16, g_s);
                    k::decode_f16(Ly.ty[1], sl + b0g, (size_t) Fd * C, X.w16 + (size_t) Fd * C, g_s);
                }
                gemm16(X.w16, 2 * Fd, C, X.ex16, cnt_e, X.gu32, 2 * Fd);
                k::swiglu_cap_f32(X.gu32, cnt_e, Fd, X.sg + st_e, X.su + st_e, c.gate_cap, c.up_cap, X.h16, g_s);
                CU(cudaMemcpy2DAsync(X.tp_h + (size_t) st_e * F + (size_t) d * Fd, (size_t) F * 2, X.h16, (size_t) Fd * 2,
                                     (size_t) Fd * 2, cnt_e, cudaMemcpyDeviceToDevice, g_s));
            }
            X.pf_layers[l % Work::kPfRing] = -1;
            CU(cudaEventRecord(g_tp_h[d], g_s));
        };
        auto tp_dn = [&](int d, Work& X, const Work& O) {
            // the other GPU's half of h in, then this GPU's rows of down for every expert
            CU(cudaStreamWaitEvent(g_s, g_tp_h[1 - d], 0));
            CU(cudaMemcpy2DAsync(X.tp_h + (size_t) (1 - d) * Fd, (size_t) F * 2, O.tp_h + (size_t) (1 - d) * Fd,
                                 (size_t) F * 2, (size_t) Fd * 2, S, cudaMemcpyDefault, g_s));
            CU(cudaMemsetAsync(X.tmp, 0, (size_t) T * C * 4, g_s));
            size_t o, b0g, b1u;
            tp_part(Ly, d, 0, o, b0g);
            tp_part(Ly, d, 1, o, b1u);
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) continue;
                const int cnt_e = W.cnt[e], st_e = W.start[e];
                k::decode_f16(Ly.ty[2], X.slot_ptrs[e] + b0g + b1u, (size_t) Cd * F, X.w16, g_s);
                gemm16(X.w16, Cd, F, X.tp_h + (size_t) st_e * F, cnt_e, X.y32, Cd);
                scatter_add_rows_kernel<<<cnt_e, 256, 0, g_s>>>(X.y32, X.gidx + st_e, X.wtr + st_e, Cd,
                                                               X.tmp + (size_t) d * Cd, C);
            }
        };
        Work& X1 = dev_work(1);
        CU(cudaEventRecord(g_ep_x, g_s));
        use_dev(1);
        CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
        CU(cudaMemcpyPeerAsync(X1.xq16_sup, 1, xq16, 0, (size_t) T * C * 2, g_s));
        tp_gu(1, X1, dev_cache(1), X1.xq16_sup);
        use_dev(0);
        tp_gu(0, W, EC, xq16);
        use_dev(1);
        tp_dn(1, X1, W);
        CU(cudaEventRecord(g_ep_done, g_s));
        use_dev(0);
        tp_dn(0, W, X1);
        CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
        CU(cudaMemcpy2DAsync(W.tmp + Cd, (size_t) C * 4, X1.tmp + Cd, (size_t) C * 4, (size_t) Cd * 4, T,
                             cudaMemcpyDefault, g_s));
    } else {
        exec(-1, W, EC, xq16);
    }
    g_pf.lap(g_pf.experts);
    t_exp += now() - te0;
    conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
    k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, r_in, h_out, g_s);
}

void run_head(const Model& M, int T, bool all_logits, OutMode out_mode, int nb, const float* h_in, Work& W,
              std::vector<float>& logits_out, std::vector<int>* argmax_out) {
    const Config& c = M.c;
    const int C = c.C;
    const size_t TC = (size_t) T * C;
    g_pf.start();
    if (out_mode != OutMode::kNone || g_taps) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + (size_t) i * W.blk_ld * C;
        st.st[nb] = h_in;
        if (out_mode != OutMode::kNone && (all_logits || T == 1)) {
            k::RmsOuts outs{};
            outs.w[0] = M.out_norm;
            outs.y16[0] = W.x16;
            outs.y32[0] = (g_taps || g_dense_q8) ? W.xq : nullptr;
            k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
            if (g_taps)
                CU(cudaMemcpyAsync(g_taps + (size_t) g_final_tap * g_taps_ld * C, W.xq, TC * 4, cudaMemcpyDeviceToDevice, g_s));
        } else {
            k::attn_res(st, nb + 1, M.out_res_score, 1, T, C, 1e-6f, W.tmp, g_s);
            k::rms(W.tmp, M.out_norm, T, C, c.eps, W.xq, g_s);
            if (g_taps)
                CU(cudaMemcpyAsync(g_taps + (size_t) g_final_tap * g_taps_ld * C, W.xq, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            if (out_mode != OutMode::kNone) k::to_f16(W.xq + (size_t) (T - 1) * C, (size_t) C, W.x16, g_s);
        }
        if (out_mode != OutMode::kNone) {
            const int t0 = all_logits ? 0 : T - 1, nt = T - t0;
            if (out_mode == OutMode::kArgmax) {
                argmax_out->resize(nt);
                for (int i = 0; i < nt; i += 16) {
                    const int n = std::min(16, nt - i);
                    matmul(M.out_w, W.xq + (size_t) (t0 + i) * C, W.x16 + (size_t) i * C, n, W.logits, c.n_rows);
                    if (n == 1) k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d + i, g_s);
                    else k::argmax_rows(W.logits, c.n_rows, n, c.n_vocab, W.argmax_d + i, g_s);
                }
                CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, (size_t) nt * 4, cudaMemcpyDeviceToHost, g_s));
                CU(cudaStreamSynchronize(g_s));
                for (int i = 0; i < nt; ++i) (*argmax_out)[i] = W.argmax_h[i];
            } else {
                for (int i = 0; i < nt; i += 16) {   // (W.logits holds 16 rows)
                    const int n = std::min(16, nt - i);
                    matmul(M.out_w, W.xq + (size_t) (t0 + i) * C, W.x16 + (size_t) i * C, n, W.logits, c.n_rows);
                    k::softcap(W.logits, (size_t) n * c.n_rows, c.logit_cap, g_s);
                    const size_t at = logits_out.size();
                    logits_out.resize(at + (size_t) n * c.n_vocab);
                    CU(cudaMemcpy2DAsync(logits_out.data() + at, (size_t) c.n_vocab * 4, W.logits,
                                         (size_t) c.n_rows * 4, (size_t) c.n_vocab * 4, n, cudaMemcpyDeviceToHost,
                                         g_s));
                }
                CU(cudaStreamSynchronize(g_s));
            }
        }
    }
    g_pf.lap(g_pf.rest);
}

// ---- the resident token path
// When the cache holds every expert of the model (Inkling-2-Mini on a 24 GB card) none is ever missing or evicted, so
// an expert's blocks are found on the device (Layer::tab_blob) and a token needs nothing from the host between its
// embedding and its logits: no routing round trip a layer, no table upload.  The position is read on the device too
// (W.pos), so a token's launches do not change from one token to the next and are replayed as one CUDA graph.  The
// dense weights are the GGUF's blocks (IK2_DENSE=q8, the default when the experts fit), one product a launch.
bool g_fast = false;        // set at load: every expert resident, one GPU, the dense weights as blocks (IK2_FAST=0: no)
bool g_fast_want = false;
bool g_graph = true;        // IK2_GRAPH=0: the resident path's launches one by one
bool g_block = true;        // IK2_BLOCK=0: a verify window through forward() instead of the block path
int g_attn_mma = 1;         // a token's attention on the tensor cores (IK2_ATTN_MMA=0: the scalar kernels; 2: the
                            // layers without a window only)
int g_mma_split = 1;        // its blocks a K / V head (k::attn_decode_mma_blocks: the device filled once)
struct FastGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
};
FastGraph g_fgraph[4];      // by OutMode

// one token's launches: its embedding row in W.emb_h and its position in W.pos_h[0] (pinned host; copied first)
void fast_token_launches(Model& M, Work& W, OutMode out_mode) {
    const Config& c = M.c;
    const int C = c.C, HD = c.H * c.D, KD = c.KV * c.D, HR = c.H * c.d_rel;
    const size_t Cb = (size_t) C * 4;
    const float qk_scale = 1.0f / std::sqrt((float) c.D);
    const float k_scale = c.qk_clip <= 28.0f ? 16.0f : 1.0f, v_scale = 32.0f;
    // the attention's key splits: fixed, so that the launches are the same at every position
    const int P_sl = std::min(128, std::max(1, (c.window + 63) / 64)), P_full = 128;
    CU(cudaMemcpyAsync(W.tmp, W.emb_h, Cb, cudaMemcpyHostToDevice, g_s));
    k::rms(W.tmp, M.embd_norm, 1, C, c.eps, W.h, g_s);
    CU(cudaMemcpyAsync(W.pos, W.pos_h, 4, cudaMemcpyHostToDevice, g_s));
    auto blk = [&](int i) { return W.blocks + (size_t) i * W.blk_ld * C; };
    int nb = 0, nb_pre = 0, n_bank = 0;
    for (int vis = 0; vis < c.n_visit; ++vis) {
        const int l = M.V[vis].l;
        Layer& Ly = bind_visit(M, vis);
        // a looped core's passes (as forward()'s)
        const float* msg = nullptr;
        if (c.n_pass > 1 && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre, W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
            } else {
                n_bank = 0;
                for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                    CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * C, blk(i), Cb, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * C, W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
                ++n_bank;
                CU(cudaMemcpyAsync(W.h, W.lp_pre, Cb, cudaMemcpyDeviceToDevice, g_s));
                nb = nb_pre;
                // the pass's messages, every core layer's at once: the bank stays as it is until the pass ends, and a
                // layer's launch of its own was 7 us on one SM (a job a block: the same sums, 24 launches less a token)
                k::ResStates st{};
                for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank + (size_t) i * C;
                k::attn_res_jobs(st, n_bank, M.lp_query_all, 2, 2 * (c.core1 - c.core0), C, 1e-6f, W.lp_msg_all, g_s);
            }
        }
        if (M.V[vis].pass > 0) msg = W.lp_msg_all + (size_t) (l - c.core0) * 4 * C;   // (q, k | v, MoE)
        if (g_taps) {   // (the drafter's taps of the token: row 0)
            const int tap = draft_tap(c, l, vis);
            if (tap >= 0)
                CU(cudaMemcpyAsync(g_taps + (size_t) tap * g_taps_ld * C, W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
        }
        // the attention's inputs, normed, in F32: W.xq, W.xk, W.xv (sep: q, k and v each have their own)
        float* x3[3] = {W.xq, W.xk, W.xv};
        const float* nw[3] = {Ly.sliding ? Ly.attn_norm : Ly.q_in_norm, Ly.sliding ? Ly.attn_norm : Ly.k_in_norm,
                              Ly.sliding ? Ly.attn_norm : Ly.v_in_norm};
        bool sep = !Ly.sliding;
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.h;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.res_score, Ly.n_res_q, 1, C, 1e-6f, W.lp_in, g_s);
                k::LoopGate gt[3];
                for (int j = 0; j < 3; ++j)
                    gt[j] = {W.lp_in + (size_t) (Ly.n_res_q == 3 ? j : 0) * C, msg + (size_t) j * C, Ly.lp_down[j],
                             Ly.lp_forget[j], Ly.lp_gain[j], x3[j]};
                k::loop_gates(gt, 3, 1, C, c.rank, 1e-6f, W.lp_g, g_s);
                k::rms_multi(k::RmsJobs{{x3[0], x3[1], x3[2]}, {nw[0], nw[1], nw[2]}, {x3[0], x3[1], x3[2]}}, 3, C, c.eps,
                             g_s);
                sep = true;
            } else if (l > 0) {
                k::RmsOuts outs{};
                uint16_t* y16[3] = {W.x16, W.xk16, W.xv16};
                for (int j = 0; j < Ly.n_res_q; ++j) outs.w[j] = nw[j], outs.y16[j] = y16[j], outs.y32[j] = x3[j];
                k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, 1, C, 1e-6f, c.eps, outs, g_s);
            } else {
                for (int j = 0; j < (sep ? 3 : 1); ++j) k::rms(W.h, nw[j], 1, C, c.eps, x3[j], g_s);
            }
        }
        // the projections, straight over the blocks
        float *q_p, *k_p, *v_p, *r_p, *gate_p, *kd_p = nullptr, *vd_p = nullptr;
        // (k's and v's short convs ride their products' launches - mv1c - and the sinks the clips' launch below)
        if (Ly.sliding) {
            const Mat& m = Ly.wqkvrg;   // q | k | v | r | gate
            q_p = W.qkv, k_p = q_p + HD, v_p = k_p + KD, r_p = v_p + KD, gate_p = r_p + HR;
            if (!sep) {
                mv1c(m, W.xq, W.qkv, k::MvConv{HD, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv},
                     k::MvConv{HD + KD, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv});
            } else if (m.q8) {   // (q, k and v each with its input: k's and v's rows take theirs, one launch)
                k::mv_q8p_conv(m.q8, m.q8d, m.N, m.K, W.xq, W.qkv,
                               k::MvConv{HD, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv},
                               k::MvConv{HD + KD, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv}, g_s,
                               k::MvAlt{W.xk, HD, KD}, k::MvAlt{W.xv, HD + KD, KD});
            } else {
                mv1(mat_rows(m, 0, HD), W.xq, W.qkv);
                mv1c(mat_rows(m, HD, KD), W.xk, k_p, k::MvConv{0, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv});
                mv1c(mat_rows(m, HD + KD, KD), W.xv, v_p,
                     k::MvConv{0, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv});
                mv1(mat_rows(m, HD + 2 * KD, m.N - HD - 2 * KD), W.xq, W.qkv + HD + 2 * KD);
            }
        } else {
            const Mat& m = Ly.wqrg;     // q | r | gate
            mv1(m, W.xq, W.qkv);
            q_p = W.qkv, r_p = q_p + HD, gate_p = r_p + HR;
            k_p = W.kk, v_p = W.vv, kd_p = W.dyn, vd_p = W.dyn + (size_t) c.KV * 4;
            k::mv_f16(Ly.sc_k_dyn16, c.KV * 4, C, W.xk, kd_p, g_s);
            k::mv_f16(Ly.sc_v_dyn16, c.KV * 4, C, W.xv, vd_p, g_s);
            mv1c(Ly.wk, W.xk, k_p, k::MvConv{0, KD, Ly.h_k, Ly.sc_k, kd_p, c.KV, nullptr, W.kconv});
            mv1c(Ly.wv, W.xv, v_p, k::MvConv{0, KD, Ly.h_v, Ly.sc_v, vd_p, c.KV, nullptr, W.vconv});
        }
        // the clips, the token's K and V into the cache at its position, the attention
        const int extent = Ly.sliding ? c.extent_swa : c.extent;
        const int window = Ly.sliding ? c.window : (1 << 30);
        const int P = Ly.sliding ? P_sl : P_full;
        // the attention on the tensor cores, from an FP16 copy of q (IK2_ATTN_MMA=0: the scalar kernels - a layer
        // without a window, its keys the whole context, is bound by their arithmetic)
        const bool mma = g_attn_mma == 1 || (g_attn_mma == 2 && !Ly.sliding);
        const int ring = Ly.sliding ? Ly.kv_cap : 0;
        const int ns = Ly.sliding ? std::min(g_mma_split, std::max(1, (c.window + 63) / 64)) : g_mma_split;
        uint16_t* q16 = mma ? W.x16 : nullptr;
        if (g_kv_fp8) {
            k::prep_qkv_r_fp8_at(q_p, Ly.q_clip, W.kconv, Ly.kc8, W.vconv, Ly.vc8, r_p, Ly.r_clip, c.H, c.KV, c.D, c.d_rel,
                                 c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha, c.n_floor, Ly.kv_cap,
                                 k_scale, v_scale, q16, g_s, Ly.sink16, W.xq, C, W.sink);
            if (mma)
                k::attn_decode_mma_fp8(q16, r_p, W.sink, Ly.kc8, Ly.vc8, ring, c.H, c.KV, W.pos, 0, ns,
                                       qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg,
                                       W.o, gate_p, W.x16, g_s);
            else
                k::attn_decode_fp8_at(q_p, r_p, W.sink, Ly.kc8, Ly.vc8, Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P,
                                      qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg,
                                      W.o, g_s);
        } else {
            k::prep_qkv_r_f16_at(q_p, Ly.q_clip, W.kconv, Ly.kc, W.vconv, Ly.vc, r_p, Ly.r_clip, c.H, c.KV, c.D, c.d_rel,
                                 c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha, c.n_floor, Ly.kv_cap, q16,
                                 g_s, Ly.sink16, W.xq, C, W.sink);
            if (mma)
                k::attn_decode_mma_f16(q16, r_p, W.sink, Ly.kc, Ly.vc, ring, c.H, c.KV, W.pos, 0, ns, qk_scale,
                                       Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, gate_p, W.x16, g_s);
            else
                k::attn_decode_f16_at(q_p, r_p, W.sink, Ly.kc, Ly.vc, Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P,
                                      qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        }
        if (!mma) k::gate_mul_f16(W.o, gate_p, (size_t) HD, W.x16, true, g_s);   // (the tensor cores' merge has it)
        // the output projection with its short conv (and the stream added, but where a block starts: there the
        // stream joins the blocks and starts again)
        const bool first = l % c.block == 0;
        if (first) CU(cudaMemcpyAsync(blk(nb), W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
        mv1c(Ly.wo, W.o, W.tmp, k::MvConv{0, C, Ly.h_attn, Ly.sc_attn, nullptr, 0, first ? nullptr : W.h, W.r});
        if (first) ++nb;
        // the MoE: its input, the router and the picks' tables on the device, the experts
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.r;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, 1, C, 1e-6f, W.xq, g_s);
                k::loop_gate(W.xq, msg + (size_t) 3 * C, Ly.lp_down[3], Ly.lp_forget[3], Ly.lp_gain[3], 1, C, c.rank,
                             1e-6f, W.lp_g, W.xq, g_s);
                k::rms(W.xq, Ly.ffn_norm, 1, C, c.eps, W.xq, g_s);
            } else {
                k::RmsOuts outs{};
                outs.w[0] = Ly.ffn_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
                k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
            }
        }
        k::mv_f16(Ly.router16, c.E, C, W.xq, W.rlog, g_s);
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + c.K);
        // (NVFP4 experts go through the tensor cores, their input as two FP16 limbs the router's launch writes)
#ifdef STRATA_USE_HIP   // (AMD: the scalar kernels read the blocks faster than the mma's layouts on the shuffles)
        static const int exp_mma = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 0;
#else
        static const int exp_mma = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 1;
#endif
        const bool exp_tc = exp_mma && !g_lane && k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, c.F);
        k::route_token(W.rlog, Ly.probs_b, c.E, c.K, c.route_scale, Ly.tab_blob, Ly.tab_sc[0], Ly.tab_sc[1],
                       Ly.tab_sc[2], rtop, rw, W.blob, W.tok, W.inv, W.sg, W.su, W.wt, g_s, W.xq, C,
                       exp_tc ? (uint16_t*) W.gu32 : nullptr);
        if (g_off) {   // (the offload path: the picks the tier does not hold copied into staging, their slots filled)
            const auto& st = g_offs.stage[g_offs.pool[l]];
            if (g_lane) {   // (lane parity: every expert in ik2_exact.h's sums, the CPU lane beside the fetch)
                k::off_lane_plan(W.blob, rtop, c.K, l * c.E, g_offs.host[l], Ly.slot_bytes, st, g_offs.fl, g_offs.cnt,
                                 g_offs.heat, g_offs.cost, g_offs.fmt[l], W.xq, C, W.sg, W.su, l, g_offs.job_d,
                                 g_offs.ls, g_offs.cls, g_s);
                k::off_fetch_copy(g_offs.fl, Ly.slot_bytes, g_offs.blocks, g_s);
                k::off_gate_up_x(W.blob, g_offs.cls, c.K, Ly.ty[0], Ly.eb[0], W.xq, C, c.F, W.sg, W.su, c.gate_cap,
                                 c.up_cap, g_offs.hx, g_s);
                k::off_down_x(W.blob, g_offs.cls, c.K, Ly.ty[2], Ly.eb[0] + Ly.eb[1], g_offs.hx, C, c.F, W.y32, g_s);
                k::off_lane_wait(g_offs.ls, g_offs.done_d, C, W.y32, g_s);
                k::moe_combine_conv(W.y32, W.inv, W.wt, c.K, C, Ly.h_mlp, Ly.sc_mlp, W.r, W.h, g_s);
                continue;
            }
            k::off_fetch_plan(W.blob, rtop, c.K, l * c.E, g_offs.host[l], Ly.slot_bytes, st, g_offs.fl, g_offs.cnt,
                              g_offs.heat, g_s);
            k::off_fetch_copy(g_offs.fl, Ly.slot_bytes, g_offs.blocks, g_s);
        }
        if (exp_tc) {
            k::experts_gate_up_swiglu_mma(W.blob, c.K, Ly.eb[0], (const uint16_t*) W.gu32, C, c.F, W.sg, W.su,
                                          c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_down_mma(W.blob, c.K, Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
        } else {
            k::experts_gate_up_swiglu(W.blob, W.tok, c.K, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq, C, c.F, W.sg, W.su,
                                      c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_down(W.blob, c.K, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
        }
        k::moe_combine_conv(W.y32, W.inv, W.wt, c.K, C, Ly.h_mlp, Ly.sc_mlp, W.r, W.h, g_s);
    }
    if (out_mode == OutMode::kNone) return;
    {   // the head
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = W.h;
        k::RmsOuts outs{};
        outs.w[0] = M.out_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
        k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
        if (g_taps)
            CU(cudaMemcpyAsync(g_taps + (size_t) g_final_tap * g_taps_ld * C, W.xq, Cb, cudaMemcpyDeviceToDevice, g_s));
        mv1(M.out_w, W.xq, W.logits);
        if (out_mode == OutMode::kArgmax) {
            k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d, g_s);
            CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, 4, cudaMemcpyDeviceToHost, g_s));
        } else if (out_mode == OutMode::kCand) {   // (the softcap keeps the logits within +- logit_cap)
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            k::cand_topk(W.logits, c.n_vocab, -c.logit_cap, c.logit_cap, kCandK, kCandCap, W.cand_part, W.cand_d, g_s);
            CU(cudaMemcpyAsync(W.cand_h, W.cand_d, (size_t) (2 + 2 * kCandCap) * 4, cudaMemcpyDeviceToHost, g_s));
        } else {
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            CU(cudaMemcpyAsync(W.logits_h, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost, g_s));
        }
    }
}

// The offload path's tier tables (Ly.tab_blob) from the cache as it is: an expert's slot, or null where it is not in
// VRAM.  Made again whenever the cache path ran since (a prompt moves experts in and out of the cache's slots).
void off_tables(Model& M) {
    ExpertCache& EC = g_dev[0].EC;
    if (g_offs.tick == EC.tick) return;
    const Config& c = M.c;
    static std::vector<const uint8_t*> tab;
    tab.resize((size_t) c.n_layer * c.E);
    for (int l = 0; l < c.n_layer; ++l) {
        const auto& P = EC.pools[EC.layer_pool[l]];
        for (int e = 0; e < c.E; ++e) {
            const int s = EC.loc[(size_t) l * c.E + e];
            tab[(size_t) l * c.E + e] = s >= 0 && s < P.n_active ? P.data + (size_t) s * P.slot_bytes : nullptr;
        }
    }
    for (int l = 0; l < c.n_layer; ++l)   // (on the compute stream: after whatever still reads them)
        CU(cudaMemcpyAsync(M.L[l].tab_blob, tab.data() + (size_t) l * c.E, (size_t) c.E * sizeof(void*),
                           cudaMemcpyHostToDevice, g_s));
    CU(cudaStreamSynchronize(g_s));   // (tab is the source until then)
    g_offs.tick = EC.tick;
}

// The lane-parity mode (IK2_LANE=1, =fetch / =cpu: every miss to one side - tests): the CPU lane started and the
// split's figures measured here (an expert on the lane, over the bus, on the GPU).  False: the layers' experts are
// not in formats the exact lane has (NVFP4 / Q8_0, gate and up of one type) - the offload path goes on without it.
bool lane_init(Model& M, const std::string& want, int threads) {
    const Config& c = M.c;
    const int C = c.C, F = c.F, K = c.K;
    auto fmt_ok = [](int t) { return t == k::kNVFP4 || t == k::kQ8_0; };
    if (F % 32 || C % 64 || K > k::kOffMaxK) return false;
    g_offs.fmt.assign(c.n_layer, 0);
    std::vector<ik2::lane::LayerW> lw(c.n_layer);
    for (int l = 0; l < c.n_layer; ++l) {
        const Layer& Ly = M.L[l];
        if (!fmt_ok(Ly.ty[0]) || Ly.ty[1] != Ly.ty[0] || !fmt_ok(Ly.ty[2])) return false;
        g_offs.fmt[l] = Ly.ty[0] == k::kNVFP4 ? 0 : 1;
        lw[l] = {Ly.host[0], Ly.slot_bytes, Ly.eb[0], Ly.eb[0] + Ly.eb[1], Ly.ty[0], Ly.ty[2]};
    }
    g_offs.cost = dalloc<k::LaneCost>(1);
    g_offs.ls = dalloc<k::LaneState>(1);
    g_offs.cls = dalloc<int>(k::kOffMaxK);
    g_offs.hx = dalloc<float>((size_t) k::kOffMaxK * F);
    CU(cudaMemsetAsync(g_offs.ls, 0, sizeof(k::LaneState), g_s));
    const size_t jb = k::kLaneJobX + (size_t) C * 4, db = k::kLaneDoneY + (size_t) k::kOffMaxK * C * 4;
    CU(cudaHostAlloc(&g_offs.job_h, jb, cudaHostAllocMapped | cudaHostAllocPortable));
    CU(cudaHostAlloc(&g_offs.done_h, db, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(g_offs.job_h, 0, jb);
    std::memset(g_offs.done_h, 0, db);
    CU(cudaHostGetDevicePointer((void**) &g_offs.job_d, g_offs.job_h, 0));
    CU(cudaHostGetDevicePointer((void**) &g_offs.done_d, g_offs.done_h, 0));
    g_offs.lane = std::make_unique<ik2::lane::Server>(threads, C, F, c.gate_cap, c.up_cap, lw, g_offs.job_h, g_offs.done_h);
    std::vector<float> xr(C);
    for (int i = 0; i < C; ++i) xr[i] = std::sin(0.37f * i) * 0.8f;
    // (the timings' experts: the i-th of a layer's in host memory - the tier's are not there when only the rest is
    //  pinned; n_host: how many)
    auto n_host = [&](int l) {
        const auto& np = M.L[l].nopin;
        return np.empty() ? c.E : (int) std::count(np.begin(), np.end(), (uint8_t) 0);
    };
    auto host_ex = [&](int l, int i) {
        const auto& np = M.L[l].nopin;
        if (np.empty()) return i % c.E;
        int k = i % std::max(1, n_host(l));
        for (int e = 0; e < c.E; ++e)
            if (!np[e] && k-- == 0) return e;
        return 0;
    };
    // The per-PC calibration (IK2_CALIB: its file, default ~/.maya/ik2-offload-calib.txt; 0: measured every start,
    // not kept): the lane's size and the split's figures measured once for this GPU, CPU and layer formats, kept
    // under that key and read back on the next start (the decode re-times the lane as it goes either way).
    const char* ce = getenv("IK2_CALIB");
    std::string cal_path = ce && *ce ? ce : "";
    if (cal_path.empty()) {
        const char* home = getenv("HOME") ? getenv("HOME") : getenv("USERPROFILE");
        if (home) cal_path = std::string(home) + "/.maya/ik2-offload-calib.txt";
    }
    std::string key;
    {
        cudaDeviceProp p{};
        CU(cudaGetDeviceProperties(&p, 0));
        size_t fr = 0, tot = 0;
        CU(cudaMemGetInfo(&fr, &tot));
        char kb[256];
        std::snprintf(kb, sizeof kb, "%s|%zu MiB|%u logical|%zu physical|%s|lane %d|C %d F %d", p.name, tot >> 20,
                      std::thread::hardware_concurrency(), strata::kernels::cpu::physical_cores(false).size(),
                      ik2::lane::rows_form(), threads, C, F);
        key = kb;
        for (char& ch : key) ch = ch == ' ' ? '_' : ch;
    }
    k::LaneCost cal{};
    int cal_thr = 0;
    bool have = false;
    if (!cal_path.empty() && cal_path != "0")
        if (std::ifstream in(cal_path); in)
            for (std::string ln; std::getline(in, ln);) {
                std::istringstream is(ln);
                std::string kk;
                k::LaneCost t{};
                int th = 0;
                if ((is >> kk >> th >> t.cpu[0] >> t.cpu[1] >> t.fetch[0] >> t.fetch[1] >> t.gpu[0] >> t.gpu[1] >> t.lat) &&
                    kk == key && th > 0)
                    cal = t, cal_thr = th, have = true;   // (the last line of the key wins)
            }
    if (have) g_offs.lane->set_threads(cal_thr - 1);
    if (threads < 0 && !have) {   // the lane's size: counts below the CPU's cores timed on experts read from memory, the
                         // fastest kept (a smaller one within 5%) - a lane as wide as the CPU loses on a busy PC
        const int phys = (int) strata::kernels::cpu::physical_cores(false).size();
        std::vector<int> cand;
        // (and the logical CPUs but two where there are more: on a small desktop CPU the second thread a core still
        //  adds - Uranus' 6 cores: 29 GB/s on 6 threads, 44 on 12)
        const int logi = (int) std::thread::hardware_concurrency();
        for (int n : {phys / 4, phys / 2, (3 * phys) / 4, phys - 2, logi > phys + 2 ? logi - 2 : 0})
            if (n >= 1 && (cand.empty() || n > cand.back())) cand.push_back(n);
        std::vector<int> l4;   // (the timing's layers: one format - the most common one, NVFP4 in Mini's file)
        for (int l = 0; l < c.n_layer; ++l)
            if (g_offs.fmt[l] == g_offs.fmt[c.n_layer / 2] && n_host(l) >= 24) l4.push_back(l);
        if (l4.empty()) l4.push_back(c.n_layer / 2);
        int best_n = -1, call = 0;
        double best_us = 1e30;
        for (int n : cand) {
            g_offs.lane->set_threads(n - 1);
            std::vector<double> t;
            for (int r = 0; r < 3; ++r, ++call) {   // (8 experts a call, none of them read before: cold, as in decode)
                int ex[8];
                const int l = l4[(size_t) (call * 5) % l4.size()];
                for (int i = 0; i < 8; ++i) ex[i] = host_ex(l, call * 8 + i);
                t.push_back(g_offs.lane->time_experts(l, ex, 8, xr.data(), 1));
            }
            std::sort(t.begin(), t.end());
            if (t[1] < best_us * 0.95) best_us = t[1], best_n = n;
        }
        g_offs.lane->set_threads(best_n - 1);
    }
    // the split's figures: the fastest of a few runs each (an idle start: the decode re-times the lane as it goes)
    k::LaneCost cs = have ? cal : k::LaneCost{};
    if (!have) cs.lat = 30.0f;
    int lay[2] = {-1, -1};
    for (int l = 0; l < c.n_layer; ++l)
        if (lay[g_offs.fmt[l]] < 0 || n_host(l) > n_host(lay[g_offs.fmt[l]])) lay[g_offs.fmt[l]] = l;
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    const uint8_t** tb = dalloc<const uint8_t*>(k::kOffMaxK);
    float* sgd = dalloc<float>(k::kOffMaxK);
    float* xd = dalloc<float>(C);
    float* yd = dalloc<float>((size_t) k::kOffMaxK * C);
    {
        std::vector<float> ones(k::kOffMaxK, 1.0f);
        CU(cudaMemcpyAsync(sgd, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(xd, xr.data(), (size_t) C * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemsetAsync(g_offs.cls, 0, k::kOffMaxK * 4, g_s));
        CU(cudaStreamSynchronize(g_s));
    }
    for (int f = 0; f < 2 && !have; ++f) {
        const int l = lay[f];
        if (l < 0) continue;
        const Layer& Ly = M.L[l];
        int ex[8];
        for (int i = 0; i < 8; ++i) ex[i] = host_ex(l, i * 37 + 5);
        cs.cpu[f] = (float) g_offs.lane->time_experts(l, ex, 4, xr.data(), 3);
        // the bus: 8 experts into staging, the copy kernel the decode uses
        const auto& st = g_offs.stage[g_offs.pool[l]];
        k::OffFetch fh{};
        fh.n = std::min(8, st.n);
        std::vector<const uint8_t*> slots(fh.n);
        for (int i = 0; i < fh.n; ++i) {
            fh.src[i] = g_offs.host[l] + (size_t) ex[i] * Ly.slot_bytes;
            fh.dst[i] = st.data + (size_t) i * st.sb;
            slots[i] = fh.dst[i];
        }
        CU(cudaMemcpyAsync(g_offs.fl, &fh, sizeof fh, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(tb, slots.data(), slots.size() * sizeof(void*), cudaMemcpyHostToDevice, g_s));
        float best_f = 1e30f, best_g = 1e30f;
        for (int r = 0; r < 3; ++r) {
            float ms = 0;
            CU(cudaEventRecord(e0, g_s));
            k::off_fetch_copy(g_offs.fl, Ly.slot_bytes, g_offs.blocks, g_s);
            CU(cudaEventRecord(e1, g_s));
            CU(cudaEventSynchronize(e1));
            CU(cudaEventElapsedTime(&ms, e0, e1));
            best_f = std::min(best_f, ms * 1e3f / fh.n);
            CU(cudaEventRecord(e0, g_s));
            k::off_gate_up_x(tb, g_offs.cls, fh.n, Ly.ty[0], Ly.eb[0], xd, C, F, sgd, sgd, c.gate_cap, c.up_cap,
                             g_offs.hx, g_s);
            k::off_down_x(tb, g_offs.cls, fh.n, Ly.ty[2], Ly.eb[0] + Ly.eb[1], g_offs.hx, C, F, yd, g_s);
            CU(cudaEventRecord(e1, g_s));
            CU(cudaEventSynchronize(e1));
            CU(cudaEventElapsedTime(&ms, e0, e1));
            best_g = std::min(best_g, ms * 1e3f / fh.n);
        }
        cs.fetch[f] = best_f, cs.gpu[f] = best_g;
        CU(cudaMemsetAsync(st.tag, 0xff, (size_t) st.n * 4, g_s));   // (the slots used: free again)
    }
    for (int f = 0; f < 2; ++f)   // (a format with no layer: the other's figures)
        if (lay[f] < 0) cs.cpu[f] = cs.cpu[1 - f], cs.fetch[f] = cs.fetch[1 - f], cs.gpu[f] = cs.gpu[1 - f];
    if (!have && !cal_path.empty() && cal_path != "0") {   // (kept for the next start)
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(cal_path).parent_path(), ec);
        if (std::ofstream out(cal_path, std::ios::app); out)
            out << key << ' ' << g_offs.lane->threads() << ' ' << cs.cpu[0] << ' ' << cs.cpu[1] << ' ' << cs.fetch[0]
                << ' ' << cs.fetch[1] << ' ' << cs.gpu[0] << ' ' << cs.gpu[1] << ' ' << cs.lat << '\n';
    }
    cs.force = want == "fetch" ? 1 : want == "cpu" ? 2 : 0;
    g_offs.cost_h = cs;
    CU(cudaMemcpyAsync(g_offs.cost, &g_offs.cost_h, sizeof cs, cudaMemcpyHostToDevice, g_s));
    CU(cudaStreamSynchronize(g_s));
    cudaFree(tb), cudaFree(sgd), cudaFree(xd), cudaFree(yd);
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    std::printf("lane parity: the CPU lane on %d threads (%s rows); an expert %.0f / %.0f us on the CPU, %.0f / %.0f us "
                "over the bus, %.0f / %.0f us on the GPU (NVFP4 / Q8_0)%s\n",
                g_offs.lane->threads(), ik2::lane::rows_form(), cs.cpu[0], cs.cpu[1], cs.fetch[0], cs.fetch[1],
                cs.gpu[0], cs.gpu[1], cs.force == 1 ? "; every miss fetched (IK2_LANE=fetch)"
                                      : cs.force == 2 ? "; every miss on the CPU (IK2_LANE=cpu)" : "");
    return true;
}

// one token through the resident path: its logits appended to logits_out (kLogits) or its argmax (kArgmax)
void fast_token(Model& M, int tok, int pos, OutMode out_mode, std::vector<float>& logits_out,
                std::vector<int>* argmax_out) {
    const Config& c = M.c;
    Work& W = dev_work(0);
    if (g_off) off_tables(M);
    if (g_lane) {   // (the lane re-timed as it runs: its figure sent when it moved by more than a tenth - a busy PC)
        bool moved = false;
        for (int f = 0; f < 2; ++f) {
            const double us = g_offs.lane->us_expert(f);
            if (us > 0 && std::fabs(us - g_offs.cost_h.cpu[f]) > 0.1 * g_offs.cost_h.cpu[f])
                g_offs.cost_h.cpu[f] = (float) us, moved = true;
        }
        if (moved) CU(cudaMemcpyAsync(g_offs.cost, &g_offs.cost_h, sizeof(k::LaneCost), cudaMemcpyHostToDevice, g_s));
    }
    const size_t rb = (size_t) c.C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    k::decode_host(M.embd_type, M.embd + (size_t) tok * rb, c.C, W.emb_h);
    W.pos_h[0] = pos;
    if (!g_graph) {
        fast_token_launches(M, W, out_mode);
    } else {
        FastGraph& G = g_fgraph[(int) out_mode];
        if (!G.exec) {   // the first token of this kind: its launches recorded
            CU(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
            fast_token_launches(M, W, out_mode);
            CU(cudaStreamEndCapture(g_s, &G.graph));
            CU(cudaGraphInstantiate(&G.exec, G.graph, 0));
        }
        CU(cudaGraphLaunch(G.exec, g_s));
    }
    CU(cudaStreamSynchronize(g_s));
    if (out_mode == OutMode::kArgmax) {
        argmax_out->assign(1, W.argmax_h[0]);
    } else if (out_mode == OutMode::kLogits) {
        logits_out.insert(logits_out.end(), W.logits_h, W.logits_h + c.n_vocab);
    } else if (out_mode == OutMode::kCand) {
        if (W.cand_h[0] <= kCandCap) {
            g_cand.n = W.cand_h[0], g_cand.idx = W.cand_h + 2, g_cand.val = (const float*) (W.cand_h + 2 + kCandCap);
        } else {   // (more than the buffer holds - logits flat past belief: the logits themselves, this once)
            CU(cudaMemcpy(W.logits_h, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost));
            logits_out.insert(logits_out.end(), W.logits_h, W.logits_h + c.n_vocab);
            g_cand.n = -1;
        }
    }
}


// ---- the block path: the T tokens (2 .. 8) of a speculative step's verify window through the resident path in one
// go - fast_token_launches for several tokens: every position read from the device, the experts grouped on the
// device (an expert's blocks read once for the tokens that picked it), the attention a key's rows read once for the
// window's queries, the convs' histories and inputs kept for a rollback - and its launches replayed as one CUDA graph
// a window size.  The dense products are the tensor cores' (q8_tile, the prompt path's arithmetic).
constexpr int kBlockMax = 8;
struct BlockBufs {
    bool ready = false;
    float* emb_h = nullptr;      // pinned [kBlockMax][C]: the window's embedding rows
    int* pos_h = nullptr;        // pinned [kBlockMax]: its positions
    float* logits_h = nullptr;   // pinned [kBlockMax][n_vocab] (kLogits)
    const uint8_t** blob = nullptr;   // group_route's arrays, device
    int *n = nullptr, *tok = nullptr, *row = nullptr, *inv = nullptr;
    float *sg = nullptr, *su = nullptr, *wt = nullptr;
    float* lp_msg = nullptr;     // a pass's messages, [n_core][4][kBlockMax][C]
    k::ConvRec* recs = nullptr;  // every visit's four convs (rollback_block)
    int n_recs = 0;
    int *cand_d = nullptr, *cand_h = nullptr;   // kCand: each row's k::cand_topk out, [kBlockMax][2 + 2 kCandCap]
};
BlockBufs g_blk;
FastGraph g_bgraph[kBlockMax + 1][4];   // by window size and OutMode

void block_init(Model& M) {
    if (g_blk.ready) return;
    const Config& c = M.c;
    const int C = c.C, KD = c.KV * c.D, S = kBlockMax * c.K;
    CU(cudaHostAlloc(&g_blk.emb_h, (size_t) kBlockMax * C * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&g_blk.pos_h, (size_t) kBlockMax * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&g_blk.logits_h, (size_t) kBlockMax * c.n_vocab * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&g_blk.cand_h, (size_t) kBlockMax * (2 + 2 * kCandCap) * 4, cudaHostAllocPortable));
    g_blk.cand_d = dalloc<int>((size_t) kBlockMax * (2 + 2 * kCandCap));
    g_blk.blob = dalloc<const uint8_t*>(S);
    g_blk.n = dalloc<int>(S), g_blk.inv = dalloc<int>(S);
    g_blk.tok = dalloc<int>((size_t) S * k::kGroupMax), g_blk.row = dalloc<int>((size_t) S * k::kGroupMax);
    g_blk.sg = dalloc<float>(S), g_blk.su = dalloc<float>(S), g_blk.wt = dalloc<float>(S);
    if (c.n_pass > 1) g_blk.lp_msg = dalloc<float>((size_t) (c.core1 - c.core0) * 4 * kBlockMax * C);
    std::vector<k::ConvRec> recs;
    for (auto& Vi : M.V) {
        float* hist[4] = {Vi.h_k, Vi.h_v, Vi.h_attn, Vi.h_mlp};
        const int cc[4] = {KD, KD, C, C};
        for (int i = 0; i < 4; ++i) {
            if (!Vi.rec_old[i] || !Vi.rec_x[i]) die("the block path: the convs' records are not allocated");
            recs.push_back({Vi.rec_old[i], Vi.rec_x[i], hist[i], cc[i]});
        }
    }
    g_blk.n_recs = (int) recs.size();
    g_blk.recs = dalloc<k::ConvRec>(recs.size());
    CU(cudaMemcpy(g_blk.recs, recs.data(), recs.size() * sizeof(k::ConvRec), cudaMemcpyHostToDevice));
    g_blk.ready = true;
}

// rows [r0, r0 + n) of a planar Q8_0 matrix times x16 [T][K] (FP16): y [T][ldy] (the tensor cores, q8_tile)
void tile_rows(const Mat& m, int r0, int n, const uint16_t* x16, int T, float* y, int ldy) {
    if (!m.q8) die("the block path: a dense matrix that is not planar Q8_0");
    k::q8_tile(m.q8 + (size_t) r0 * m.K, m.q8d + (size_t) r0 * (m.K >> 5), n, m.K, x16, T, y, ldy, g_s);
}
// The exact window (g_block_exact, the default; IK2_BLOCK_EXACT=0: the tensor cores' tiles and the window's own
// attention - a few 1e-5 off the one-token path, so that drafting changes greedy answers): every token's values the
// one-token path's to the bit - its products with mv1's sums (the weights read once), its attention through the
// token's own kernels, its experts' rows with the token kernels' sums (an expert read once for its tokens).
bool g_block_exact = true;
// rows [r0, r0 + n) of a dense matrix times x [T][K] (F32, rows x_ld apart): y [T][ldy], each token's mv1's
void exact_rows(const Mat& m, int r0, int n, const float* x, int x_ld, int T, float* y, int ldy) {
    if (m.q8) {
        k::mv_q8p_cols(m.q8 + (size_t) r0 * m.K, m.q8d + (size_t) r0 * (m.K >> 5), n, m.K, x, x_ld, T, y, ldy, g_s);
        return;
    }
    const Mat part = mat_rows(m, r0, n);   // (not planar Q8_0: mv1 a token at a time)
    for (int t = 0; t < T; ++t) mv1(part, x + (size_t) t * x_ld, y + (size_t) t * ldy);
}

void fast_block_launches(Model& M, Work& W, int T, OutMode out_mode) {
    const Config& c = M.c;
    const int C = c.C, HD = c.H * c.D, KD = c.KV * c.D, HR = c.H * c.d_rel, S = T * c.K;
    const size_t TC = (size_t) T * C, TCb = TC * 4;
    const float qk_scale = 1.0f / std::sqrt((float) c.D);
    const float k_scale = c.qk_clip <= 28.0f ? 16.0f : 1.0f, v_scale = 32.0f;
    CU(cudaMemcpyAsync(W.tmp, g_blk.emb_h, TCb, cudaMemcpyHostToDevice, g_s));
    k::rms(W.tmp, M.embd_norm, T, C, c.eps, W.h, g_s);
    CU(cudaMemcpyAsync(W.pos, g_blk.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, g_s));
    auto blk = [&](int i) { return W.blocks + (size_t) i * W.blk_ld * C; };
    int nb = 0, nb_pre = 0, n_bank = 0;
    for (int vis = 0; vis < c.n_visit; ++vis) {
        const int l = M.V[vis].l;
        Layer& Ly = bind_visit(M, vis);
        const float* msg = nullptr;
        if (c.n_pass > 1 && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre, W.h, TCb, cudaMemcpyDeviceToDevice, g_s));
            } else {
                n_bank = 0;
                for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                    CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, blk(i), TCb, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, W.h, TCb, cudaMemcpyDeviceToDevice, g_s));
                ++n_bank;
                CU(cudaMemcpyAsync(W.h, W.lp_pre, TCb, cudaMemcpyDeviceToDevice, g_s));
                nb = nb_pre;
                k::ResStates st{};
                for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank + (size_t) i * TC;
                k::attn_res_jobs_block(st, n_bank, M.lp_query_all, 2, 2 * (c.core1 - c.core0), T, C, 1e-6f,
                                       g_blk.lp_msg, g_s);
            }
        }
        if (M.V[vis].pass > 0) msg = g_blk.lp_msg + (size_t) (l - c.core0) * 4 * TC;   // [4][T][C]: q, k, v, MoE
        if (g_taps) {
            const int tap = draft_tap(c, l, vis);
            if (tap >= 0)
                CU(cudaMemcpyAsync(g_taps + (size_t) tap * g_taps_ld * C, W.h, TCb, cudaMemcpyDeviceToDevice, g_s));
        }
        float* x3[3] = {W.xq, W.xk, W.xv};
        uint16_t* y16[3] = {W.x16, W.xk16, W.xv16};
        const float* nw[3] = {Ly.sliding ? Ly.attn_norm : Ly.q_in_norm, Ly.sliding ? Ly.attn_norm : Ly.k_in_norm,
                              Ly.sliding ? Ly.attn_norm : Ly.v_in_norm};
        bool sep = !Ly.sliding;
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.h;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, W.lp_in, g_s);
                k::LoopGate gt[3];
                for (int j = 0; j < 3; ++j)
                    gt[j] = {W.lp_in + (size_t) (Ly.n_res_q == 3 ? j : 0) * TC, msg + (size_t) j * TC, Ly.lp_down[j],
                             Ly.lp_forget[j], Ly.lp_gain[j], x3[j]};
                k::loop_gates(gt, 3, T, C, c.rank, 1e-6f, W.lp_g, g_s);
                for (int j = 0; j < 3; ++j) k::rms_f16(x3[j], nw[j], T, C, c.eps, y16[j], x3[j], g_s);
                sep = true;
            } else if (l > 0) {
                k::RmsOuts outs{};
                for (int j = 0; j < Ly.n_res_q; ++j) outs.w[j] = nw[j], outs.y16[j] = y16[j], outs.y32[j] = x3[j];
                k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, c.eps, outs, g_s);
            } else {
                for (int j = 0; j < (sep ? 3 : 1); ++j) k::rms_f16(W.h, nw[j], T, C, c.eps, y16[j], x3[j], g_s);
            }
        }
        // the projections (the tensor cores), the k and v short convs (their histories and inputs recorded)
        float *q_p, *r_p, *gate_p;
        int q_ld, r_ld, gate_ld;
        const bool ex = g_block_exact;
        if (Ly.sliding) {
            const Mat& m = Ly.wqkvrg;   // q | k | v | r | gate
            const int N = m.N;
            if (ex && !sep) {
                exact_rows(m, 0, N, W.xq, C, T, W.qkv, N);
            } else if (ex) {   // (q, k and v each with its input, as mv_q8p_conv's alternative inputs)
                exact_rows(m, 0, HD, W.xq, C, T, W.qkv, N);
                exact_rows(m, HD, KD, W.xk, C, T, W.qkv + HD, N);
                exact_rows(m, HD + KD, KD, W.xv, C, T, W.qkv + HD + KD, N);
                exact_rows(m, HD + 2 * KD, N - HD - 2 * KD, W.xq, C, T, W.qkv + HD + 2 * KD, N);
            } else if (!sep) {
                tile_rows(m, 0, N, W.x16, T, W.qkv, N);
            } else {
                tile_rows(m, 0, HD, W.x16, T, W.qkv, N);
                tile_rows(m, HD, KD, W.xk16, T, W.qkv + HD, N);
                tile_rows(m, HD + KD, KD, W.xv16, T, W.qkv + HD + KD, N);
                tile_rows(m, HD + 2 * KD, N - HD - 2 * KD, W.x16, T, W.qkv + HD + 2 * KD, N);
            }
            q_p = W.qkv, r_p = W.qkv + HD + 2 * KD, gate_p = r_p + HR;
            q_ld = r_ld = gate_ld = N;
            k::conv_block(W.qkv + HD, N, Ly.h_k, Ly.sc_k, nullptr, 0, 0, T, KD, nullptr, 0, W.kconv, KD, Ly.rec_old[0],
                          Ly.rec_x[0], g_s);
            k::conv_block(W.qkv + HD + KD, N, Ly.h_v, Ly.sc_v, nullptr, 0, 0, T, KD, nullptr, 0, W.vconv, KD,
                          Ly.rec_old[1], Ly.rec_x[1], g_s);
        } else {
            const Mat& m = Ly.wqrg;     // q | r | gate
            const int N = m.N;
            if (ex) exact_rows(m, 0, N, W.xq, C, T, W.qkv, N);
            else tile_rows(m, 0, N, W.x16, T, W.qkv, N);
            q_p = W.qkv, r_p = W.qkv + HD, gate_p = r_p + HR;
            q_ld = r_ld = gate_ld = N;
            const int DL = c.KV * 4;
            float *kd = W.dyn, *vd = W.dyn + (size_t) T * DL;
            k::mv_f16_rows(Ly.sc_k_dyn16, DL, C, W.xk, C, T, kd, DL, g_s);
            k::mv_f16_rows(Ly.sc_v_dyn16, DL, C, W.xv, C, T, vd, DL, g_s);
            if (ex) {
                exact_rows(Ly.wk, 0, KD, W.xk, C, T, W.kk, KD);
                exact_rows(Ly.wv, 0, KD, W.xv, C, T, W.vv, KD);
            } else {
                tile_rows(Ly.wk, 0, KD, W.xk16, T, W.kk, KD);
                tile_rows(Ly.wv, 0, KD, W.xv16, T, W.vv, KD);
            }
            k::conv_block(W.kk, KD, Ly.h_k, Ly.sc_k, kd, DL, c.KV, T, KD, nullptr, 0, W.kconv, KD, Ly.rec_old[0],
                          Ly.rec_x[0], g_s);
            k::conv_block(W.vv, KD, Ly.h_v, Ly.sc_v, vd, DL, c.KV, T, KD, nullptr, 0, W.vconv, KD, Ly.rec_old[1],
                          Ly.rec_x[1], g_s);
        }
        // the clips, the window's K and V into the cache at its positions, the attention of its queries
        const int extent = Ly.sliding ? c.extent_swa : c.extent;
        const int window = Ly.sliding ? c.window : (1 << 30);
        const int ring = Ly.sliding ? Ly.kv_cap : 0;
        const int ns = Ly.sliding ? std::min(g_mma_split, std::max(1, (c.window + 63) / 64)) : g_mma_split;
        uint16_t* q16 = W.x16;
        if (ex) {   // (each token through fast_token_launches' own clips and attention, its position from W.pos + t)
            const bool mma = g_attn_mma == 1 || (g_attn_mma == 2 && !Ly.sliding);
            const int P = Ly.sliding ? std::min(128, std::max(1, (c.window + 63) / 64)) : 128;
            for (int t = 0; t < T; ++t) {
                float *qt = q_p + (size_t) t * q_ld, *rt = r_p + (size_t) t * r_ld, *ot = W.o + (size_t) t * HD;
                const float* gt = gate_p + (size_t) t * gate_ld;
                const int* pt = W.pos + t;
                uint16_t* q16t = mma ? W.x16 + (size_t) t * HD : nullptr;
                float* sk = W.sink + (size_t) t * c.H;
                if (g_kv_fp8) {
                    k::prep_qkv_r_fp8_at(qt, Ly.q_clip, W.kconv + (size_t) t * KD, Ly.kc8, W.vconv + (size_t) t * KD,
                                         Ly.vc8, rt, Ly.r_clip, c.H, c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps,
                                         pt, !Ly.sliding, c.alpha, c.n_floor, Ly.kv_cap, k_scale, v_scale, q16t, g_s,
                                         Ly.sink16, W.xq + (size_t) t * C, C, sk);
                    if (mma)
                        k::attn_decode_mma_fp8(q16t, rt, sk, Ly.kc8, Ly.vc8, ring, c.H, c.KV, pt, 0, ns,
                                               qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window,
                                               W.lg, ot, gt, W.x16 + (size_t) t * HD, g_s);
                    else
                        k::attn_decode_fp8_at(qt, rt, sk, Ly.kc8, Ly.vc8, Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, pt, P,
                                              qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window,
                                              W.lg, ot, g_s);
                } else {
                    k::prep_qkv_r_f16_at(qt, Ly.q_clip, W.kconv + (size_t) t * KD, Ly.kc, W.vconv + (size_t) t * KD,
                                         Ly.vc, rt, Ly.r_clip, c.H, c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps, pt,
                                         !Ly.sliding, c.alpha, c.n_floor, Ly.kv_cap, q16t, g_s, Ly.sink16,
                                         W.xq + (size_t) t * C, C, sk);
                    if (mma)
                        k::attn_decode_mma_f16(q16t, rt, sk, Ly.kc, Ly.vc, ring, c.H, c.KV, pt, 0, ns, qk_scale,
                                               Ly.rel_proj, c.d_rel, extent, window, W.lg, ot, gt,
                                               W.x16 + (size_t) t * HD, g_s);
                    else
                        k::attn_decode_f16_at(qt, rt, sk, Ly.kc, Ly.vc, Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, pt, P,
                                              qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, ot, g_s);
                }
                if (!mma) k::gate_mul_f16(ot, gt, (size_t) HD, W.x16 + (size_t) t * HD, true, g_s);
            }
        } else if (g_kv_fp8) {
            k::prep_qkv_r_fp8_block(q_p, q_ld, Ly.q_clip, W.kconv, Ly.kc8, W.vconv, Ly.vc8, r_p, r_ld, Ly.r_clip, T, c.H,
                                    c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha,
                                    c.n_floor, Ly.kv_cap, k_scale, v_scale, q16, g_s, Ly.sink16, W.xq, C, W.sink);
            k::attn_block_fp8(q16, r_p, r_ld, W.sink, Ly.kc8, Ly.vc8, ring, c.H, c.KV, W.pos, T, ns, qk_scale / k_scale,
                              1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, gate_p, gate_ld, W.x16,
                              g_s);
        } else {
            k::prep_qkv_r_f16_block(q_p, q_ld, Ly.q_clip, W.kconv, Ly.kc, W.vconv, Ly.vc, r_p, r_ld, Ly.r_clip, T, c.H,
                                    c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha,
                                    c.n_floor, Ly.kv_cap, q16, g_s, Ly.sink16, W.xq, C, W.sink);
            k::attn_block_f16(q16, r_p, r_ld, W.sink, Ly.kc, Ly.vc, ring, c.H, c.KV, W.pos, T, ns, qk_scale,
                              Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, gate_p, gate_ld, W.x16, g_s);
        }
        // the output projection and its short conv (the stream added, but where a block starts)
        const bool first = l % c.block == 0;
        if (first) CU(cudaMemcpyAsync(blk(nb), W.h, TCb, cudaMemcpyDeviceToDevice, g_s));
        if (ex) exact_rows(Ly.wo, 0, C, W.o, HD, T, W.tmp, C);
        else tile_rows(Ly.wo, 0, C, W.x16, T, W.tmp, C);
        k::conv_block(W.tmp, C, Ly.h_attn, Ly.sc_attn, nullptr, 0, 0, T, C, first ? nullptr : W.h, C, W.r, C,
                      Ly.rec_old[2], Ly.rec_x[2], g_s);
        if (first) ++nb;
        // the MoE: its input, the router, the picks grouped by expert on the device, the experts
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.r;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, W.xq, g_s);
                const k::LoopGate gt{W.xq, msg + (size_t) 3 * TC, Ly.lp_down[3], Ly.lp_forget[3], Ly.lp_gain[3], W.xq};
                k::loop_gates(&gt, 1, T, C, c.rank, 1e-6f, W.lp_g, g_s);
                k::rms(W.xq, Ly.ffn_norm, T, C, c.eps, W.xq, g_s);
            } else {
                k::RmsOuts outs{};
                outs.w[0] = Ly.ffn_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
                k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
            }
        }
        k::mv_f16_rows(Ly.router16, c.E, C, W.xq, C, T, W.rlog, c.E, g_s);
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + S);
        k::route_rows(W.rlog, Ly.probs_b, T, c.E, c.K, c.route_scale, rtop, rw, g_s);
        k::group_route(rtop, rw, S, c.K, Ly.tab_blob, Ly.tab_sc[0], Ly.tab_sc[1], Ly.tab_sc[2], g_blk.blob, g_blk.n,
                       g_blk.tok, g_blk.row, g_blk.sg, g_blk.su, g_blk.wt, g_blk.inv, g_s);
        // (exact: the token path's choice - NVFP4 on the tensor cores unless IK2_EXPERT_MMA=0, else the row dots)
#ifdef STRATA_USE_HIP   // (AMD: the token path's default is the row dots - so is the window's)
        static const int exp_mma = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 0;
#else
        static const int exp_mma = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 1;
#endif
        if (ex && !(exp_mma && k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, c.F))) {
            k::experts_gate_up_swiglu_mem(g_blk.blob, g_blk.n, g_blk.tok, g_blk.row, S, Ly.ty[0], Ly.ty[1], Ly.eb[0],
                                          W.xq, C, c.F, g_blk.sg, g_blk.su, c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_down_mem(g_blk.blob, g_blk.n, g_blk.row, S, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32,
                                g_s);
        } else if (k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, c.F)) {
            uint16_t* limbs = (uint16_t*) W.gu32;
            k::f16_limbs_rows(W.xq, T, C, limbs, g_s);
            k::experts_block_gu(g_blk.blob, g_blk.n, g_blk.tok, g_blk.row, S, T, Ly.eb[0], C, c.F, limbs, g_blk.sg,
                                g_blk.su, c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_block_down(g_blk.blob, g_blk.n, g_blk.row, S, Ly.eb[0] + Ly.eb[1], C, c.F, W.h16, W.y32, g_s);
        } else if (Ly.ty[0] == k::kQ8_0 && Ly.ty[1] == k::kQ8_0 && Ly.ty[2] == k::kQ8_0 && C % 32 == 0 && c.F % 32 == 0) {
            uint16_t* limbs = (uint16_t*) W.gu32;
            k::f16_limbs_rows(W.xq, T, C, limbs, g_s);
            k::experts_block_gu_q8(g_blk.blob, g_blk.n, g_blk.tok, g_blk.row, S, T, Ly.eb[0], C, c.F, limbs, g_blk.sg,
                                   g_blk.su, c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_block_down_q8(g_blk.blob, g_blk.n, g_blk.row, S, Ly.eb[0] + Ly.eb[1], C, c.F, W.h16, W.y32, g_s);
        } else {
            k::experts_gate_up_swiglu_g(g_blk.blob, g_blk.n, g_blk.tok, g_blk.row, S, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq,
                                        C, c.F, g_blk.sg, g_blk.su, c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_down_g(g_blk.blob, g_blk.n, g_blk.row, S, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32,
                              g_s);
        }
        k::moe_combine_f32(W.y32, g_blk.inv, g_blk.wt, T, c.K, C, W.tmp, false, g_s);
        k::conv_block(W.tmp, C, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, 0, T, C, W.r, C, W.h, C, Ly.rec_old[3], Ly.rec_x[3],
                      g_s);
    }
    if (out_mode == OutMode::kNone) return;
    {   // the head: every token's logits
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = W.h;
        k::RmsOuts outs{};
        outs.w[0] = M.out_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
        k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
        if (g_taps)
            CU(cudaMemcpyAsync(g_taps + (size_t) g_final_tap * g_taps_ld * C, W.xq, TCb, cudaMemcpyDeviceToDevice, g_s));
        if (g_block_exact) exact_rows(M.out_w, 0, c.n_rows, W.xq, C, T, W.logits, c.n_rows);
        else tile_rows(M.out_w, 0, c.n_rows, W.x16, T, W.logits, c.n_rows);
        if (out_mode == OutMode::kArgmax) {
            k::argmax_rows(W.logits, c.n_rows, T, c.n_vocab, W.argmax_d, g_s);
            CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, (size_t) T * 4, cudaMemcpyDeviceToHost, g_s));
        } else if (out_mode == OutMode::kCand) {   // (each row's candidates, as the token path's)
            const size_t cs = 2 + 2 * (size_t) kCandCap;
            k::softcap(W.logits, (size_t) T * c.n_rows, c.logit_cap, g_s);
            for (int t = 0; t < T; ++t)
                k::cand_topk(W.logits + (size_t) t * c.n_rows, c.n_vocab, -c.logit_cap, c.logit_cap, kCandK, kCandCap,
                             W.cand_part, g_blk.cand_d + t * cs, g_s);
            CU(cudaMemcpyAsync(g_blk.cand_h, g_blk.cand_d, (size_t) T * cs * 4, cudaMemcpyDeviceToHost, g_s));
        } else {
            k::softcap(W.logits, (size_t) T * c.n_rows, c.logit_cap, g_s);
            for (int t = 0; t < T; ++t)
                CU(cudaMemcpyAsync(g_blk.logits_h + (size_t) t * c.n_vocab, W.logits + (size_t) t * c.n_rows,
                                   (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost, g_s));
        }
    }
}

// The window tok [T] at positions pos0 .. pos0 + T - 1 through the block path: argmax_out [T] (kArgmax) or logits_out
// [T][n_vocab] appended (kLogits).  The convs' records then hold the window (block_rollback takes it back).
void fast_block(Model& M, const int* tok, int T, int pos0, OutMode out_mode, std::vector<float>& logits_out,
                std::vector<int>* argmax_out) {
    const Config& c = M.c;
    if (T < 1 || T > kBlockMax) die("fast_block: a window of 1 .. 8 tokens");
    if (!g_blk.ready) block_init(M);
    Work& W = dev_work(0);
    const size_t rb = (size_t) c.C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    for (int t = 0; t < T; ++t) {
        k::decode_host(M.embd_type, M.embd + (size_t) tok[t] * rb, c.C, g_blk.emb_h + (size_t) t * c.C);
        g_blk.pos_h[t] = pos0 + t;
    }
    if (!g_graph) {
        fast_block_launches(M, W, T, out_mode);
    } else {
        FastGraph& G = g_bgraph[T][(int) out_mode];
        if (!G.exec) {
            CU(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
            fast_block_launches(M, W, T, out_mode);
            CU(cudaStreamEndCapture(g_s, &G.graph));
            CU(cudaGraphInstantiate(&G.exec, G.graph, 0));
        }
        CU(cudaGraphLaunch(G.exec, g_s));
    }
    CU(cudaStreamSynchronize(g_s));
    if (out_mode == OutMode::kArgmax) {
        argmax_out->assign(W.argmax_h, W.argmax_h + T);
    } else if (out_mode == OutMode::kLogits) {
        logits_out.insert(logits_out.end(), g_blk.logits_h, g_blk.logits_h + (size_t) T * c.n_vocab);
    }
}
// row t's candidates of the last kCand window (logit, token); false: more than kept (the row's logits then)
bool block_cands(const Model& M, int t, std::vector<std::pair<float, int>>& cd) {
    const int* r = g_blk.cand_h + (size_t) t * (2 + 2 * kCandCap);
    if (r[0] > kCandCap) return false;
    const int* idx = r + 2;
    const float* val = (const float*) (r + 2 + kCandCap);
    cd.resize(r[0]);
    for (int i = 0; i < r[0]; ++i) cd[i] = {val[i], idx[i]};
    return true;
}
// every conv's history as after the first n tokens of the last window (fast_block's records)
void block_rollback(int n) {
    k::rollback_block(g_blk.recs, g_blk.n_recs, n, g_s);
}


// ---- speculative decoding on the resident path: the drafter's tokens verified as one window (fast_block)

void embed_row(Model& M, int t, float* raw, float* out);

struct SampleCfg {   // temp <= 0: greedy
    double temp = 0, top_p = 1;
    int top_k = 0;
    std::mt19937_64* rng = nullptr;
};
// the serve loop's sampling distribution over candidates cd (logit, token): softmax(logit / temp) over the top_p
// mass of the min(top_k, kCandK) largest (ties: the lower token first) - pick_top's - as (token, probability)
std::vector<std::pair<int, double>> sample_dist(std::vector<std::pair<float, int>>& cd, const SampleCfg& sc) {
    const int kk = std::min((int) cd.size(), sc.top_k > 0 ? std::min(sc.top_k, kCandK) : kCandK);
    std::partial_sort(cd.begin(), cd.begin() + kk, cd.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first || (a.first == b.first && a.second < b.second);
                      });
    std::vector<double> pr(kk);
    double z = 0;
    for (int j = 0; j < kk; ++j) z += (pr[j] = std::exp((cd[j].first - cd[0].first) / sc.temp));
    double acc = 0;
    int keep = kk;
    for (int j = 0; j < kk; ++j) {
        acc += pr[j] / z;
        if (acc >= sc.top_p) {
            keep = j + 1;
            break;
        }
    }
    double zk = 0;
    for (int j = 0; j < keep; ++j) zk += pr[j];
    std::vector<std::pair<int, double>> out(keep);
    for (int j = 0; j < keep; ++j) out[j] = {cd[j].second, pr[j] / zk};
    return out;
}
int sample_from(const std::vector<std::pair<int, double>>& d, std::mt19937_64& rng, int skip = -1) {
    double z = 0;
    for (const auto& e : d)
        if (e.first != skip) z += e.second;
    if (z <= 0) return d[0].first;
    double u = std::uniform_real_distribution<double>(0, z)(rng);
    for (const auto& e : d) {
        if (e.first == skip) continue;
        u -= e.second;
        if (u <= 0) return e.first;
    }
    for (auto it = d.rbegin(); it != d.rend(); ++it)
        if (it->first != skip) return it->first;
    return d[0].first;
}

// When to draft and how wide (IK2_DRAFT_WIDTH the widest; IK2_DRAFT_ADAPT=0: always that wide): a step's expected
// tokens (1 - a^B) / (1 - a) - a the drafts' acceptance, each accepted given the one before it, from the recent
// steps - over its measured time, against a plain token's; a plain run drafts once in a while to look again.
struct SpecCtl {
    int wmax = 5;
    bool adapt = true;
    double c_plain = 7.8e-3, c[kBlockMax + 1] = {};
    double S = 1, F = 1;   // (the drafts accepted, the ones rejected: decayed counts)
    int plain_run = 0;
    int probe = getenv("IK2_DRAFT_PROBE") ? atoi(getenv("IK2_DRAFT_PROBE")) : 8;   // plain tokens before a look
    int probe_w = 4;
    double decay = getenv("IK2_DRAFT_DECAY") ? atof(getenv("IK2_DRAFT_DECAY")) : 0.85;
    uint64_t steps = 0, plain = 0, drafted = 0, accepted = 0;
    uint64_t hist[kBlockMax] = {};
    SpecCtl() {
        const double prior[kBlockMax + 1] = {0, 7.8e-3, 13.3e-3, 15.1e-3, 16.7e-3, 18.2e-3, 19.8e-3, 21.3e-3, 22.8e-3};
        for (int b = 0; b <= kBlockMax; ++b) c[b] = prior[b];
    }
    int choose(int room) {   // room: the widest a step may be (1: plain)
        const int bmax = std::min(wmax, room);
        if (bmax < 2) return 1;
        if (!adapt) return bmax;
        const double a = std::min(0.98, (S + 0.5) / (S + F + 1.0));
        int best = 1;
        double vb = 1.0 / c_plain;
        for (int b = 2; b <= bmax; ++b) {
            const double e = (1.0 - std::pow(a, b)) / (1.0 - a), v = e / c[b];
            if (v > vb) vb = v, best = b;
        }
        if (best == 1 && plain_run >= probe) best = std::min(probe_w, bmax);   // (a look at the acceptance again)
        return best;
    }
    void took_plain(double t) {
        c_plain = 0.9 * c_plain + 0.1 * t;
        ++plain, ++plain_run;
    }
    void took_spec(int B, int acc, double t) {
        c[B] = 0.8 * c[B] + 0.2 * t;
        S = decay * S + acc, F = decay * F + (acc < B - 1 ? 1 : 0);
        ++steps, drafted += B - 1, accepted += acc, ++hist[acc];
        plain_run = 0;
    }
    // The prompt lookup's drafts (the tokens after the latest earlier copy of the text's end, k of them): their own
    // acceptance (copied text is kept long or not at all), the same window costs.  pick: a step's source and width
    // - 0 a plain token, 1 the drafter's B - 1 drafts, 2 the lookup's - the most tokens a second expected.
    double SL = 2, FL = 1;
    uint64_t l_steps = 0, l_drafted = 0, l_accepted = 0;
    bool use_model = true, use_lookup = true;   // (the sources this start runs: --drafting's choice)
    int pick(int room, int k, int& B) {
        const int bmax = std::min(wmax, room);
        B = 1;
        if (bmax < 2) return 0;
        int src = 0;
        double vb = 1.0 / c_plain;
        if (use_lookup && k > 0) {
            const double a = std::min(0.98, (SL + 0.5) / (SL + FL + 1.0));
            for (int b = 2; b <= std::min(bmax, k + 1); ++b) {
                const double e = (1.0 - std::pow(a, b)) / (1.0 - a), v = e / c[b];
                if (v > vb) vb = v, B = b, src = 2;
            }
        }
        if (use_model) {
            const int bm = choose(room);
            if (bm >= 2) {
                const double a = std::min(0.98, (S + 0.5) / (S + F + 1.0));
                const double v = (1.0 - std::pow(a, bm)) / (1.0 - a) / c[bm];
                if (src == 0 || v > vb || plain_run >= probe) vb = v, B = bm, src = 1;
            }
        }
        return src;
    }
    void took_lookup(int B, int acc, double t) {
        c[B] = 0.8 * c[B] + 0.2 * t;
        SL = decay * SL + acc, FL = decay * FL + (acc < B - 1 ? 1 : 0);
        ++l_steps, l_drafted += B - 1, l_accepted += acc;
        plain_run = 0;
    }
    uint64_t offered() const { return drafted + l_drafted; }
    uint64_t kept() const { return accepted + l_accepted; }
};

bool g_lookup = false;   // the prompt lookup (src/spec/suffix_drafter): drafts copied from the text so far

// VRAM drafting takes beside the model (the plan's check that it changes nothing): the window's records (every
// visit's four convs, for rollback_block) and buffers, whichever drafts; with the drafter (path) its weights (the
// file's BF16), its fold through the head (FP16 [256][n_rows] and a bias), its layer without a window's context
// (FP16 K and V, 8 heads of 128: 4 KiB a position) and its attention's scratch (24 heads: 768 B a position), the 4
// windowed layers' rings (F32, 2048 positions) and the target's taps (4 rows of a chunk, F32).  (The head's FP16 copy
// its fold is made from is gone again before the work buffers take the plan's reserve.)  tools/ik2_fit.py's
// draft_gib has the same figures.
size_t draft_room(const std::string& path, int ctx, int chunk, const Config& c, int n_rows) {
    const size_t C = c.C, KD = (size_t) c.KV * c.D;
    size_t b = (size_t) c.n_visit * (2 * KD + 2 * C) * (3 + 8) * 4 + ((size_t) 16 << 20);
    if (path.empty()) return b;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const size_t file = f ? (size_t) f.tellg() : ((size_t) 640 << 20);
    b += file + (size_t) n_rows * 258 * 2 + (size_t) ctx * (4096 + 768) + (size_t) 4 * 2048 * 8 * 128 * 8 +
         (size_t) 4 * std::max(chunk, 8) * C * 4 + ((size_t) 32 << 20);
    return b;
}

// a sampled step's pick at a drafted position: the draft with the target's probability of it, else the target's
// distribution without it - the target's own distribution, whatever drafted (speculative sampling with a point-mass
// draft).  accepted: the draft was kept.
int spec_pick(const std::vector<std::pair<int, double>>& dist, int draft, std::mt19937_64& rng, bool& accepted) {
    double pd = 0;
    for (const auto& e : dist)
        if (e.first == draft) pd = e.second;
    accepted = std::uniform_real_distribution<double>(0, 1)(rng) < pd;
    return accepted ? draft : sample_from(dist, rng, draft);
}

int g_window_check = 0;                       // IK2_WINDOW_CHECK: windows still to check against fast_token
uint64_t g_window_rows = 0, g_window_same = 0;   // (rows checked, rows the same to the bit)

// One speculative step: the anchor (the last chosen token, not run yet) at position P, B - 1 drafts after it, the B
// verified as one window.  next: the tokens chosen after the anchor - the accepted drafts and the target's own next
// token (the last: not run yet, the next step's anchor), cut after a stop token.  The anchor and next but its last
// are then in the cache, the convs' histories taken back to them and the drafter's context written for them.
// given: the B - 1 drafts from elsewhere (the prompt lookup's), else the drafter's.  t_draft: the drafter's seconds.
void spec_step(Model& M, ik2::draft::Drafter* drafter, int anchor, int P, int B, const SampleCfg& sc,
               const std::vector<int>& stops, std::vector<int>& next, const int* given = nullptr,
               double* t_draft = nullptr) {
    const Config& c = M.c;
    int dr[kBlockMax];
    if (given) {
        for (int j = 0; j + 1 < B; ++j) dr[j] = given[j];
    } else {
        const double t0 = now();
        static float *raw = nullptr, *anchor_h = nullptr;
        if (!raw) raw = dalloc<float>(c.C), anchor_h = dalloc<float>(c.C);
        embed_row(M, anchor, raw, anchor_h);
        ik2::draft::propose(drafter, anchor, anchor_h, P, nullptr, c.n_rows, c.n_vocab, dr, B - 1);
        if (t_draft) *t_draft += now() - t0;
    }
    int tk[kBlockMax];
    tk[0] = anchor;
    for (int j = 1; j < B; ++j) tk[j] = dr[j - 1];
    next.clear();
    std::vector<float> lo;
    if (g_window_check > 0) {   // (IK2_WINDOW_CHECK=n: the first n windows against the one-token path, logit for logit)
        --g_window_check;
        std::vector<float> lb, lt;
        fast_block(M, tk, B, P, OutMode::kLogits, lb, nullptr);
        block_rollback(0);
        for (int j = 0; j < B; ++j) fast_token(M, tk[j], P + j, OutMode::kLogits, lt, nullptr);
        int same = 0;
        double mx = 0;
        for (int j = 0; j < B; ++j) {
            const float *a = lb.data() + (size_t) j * c.n_vocab, *b = lt.data() + (size_t) j * c.n_vocab;
            same += !std::memcmp(a, b, (size_t) c.n_vocab * 4);
            for (int v = 0; v < c.n_vocab; ++v) mx = std::max(mx, (double) std::fabs(a[v] - b[v]));
        }
        g_window_rows += B, g_window_same += same;
        std::fprintf(stderr, "window check: %d tokens at %d, %d of %d rows the one-token path's to the bit "
                     "(max |dlogit| %.3g)\n", B, P, same, B, mx);
        block_rollback(0);   // (and the window again: its records and the drafter's taps)
    }
    if (sc.temp <= 0) {
        std::vector<int> g;
        fast_block(M, tk, B, P, OutMode::kArgmax, lo, &g);
        int n = 0;
        while (n < B - 1 && dr[n] == g[n]) ++n;
        for (int j = 0; j < n; ++j) next.push_back(dr[j]);
        next.push_back(g[n]);
    } else {
        // the drafts are the drafter's argmax (a point mass): accept draft d with the target's probability of it,
        // else sample the target's distribution without d - the target's distribution exactly
        fast_block(M, tk, B, P, OutMode::kCand, lo, nullptr);
        for (int j = 0; j < B; ++j) {
            std::vector<std::pair<float, int>> cd;
            if (!block_cands(M, j, cd)) {   // (more candidates than kept: the row's logits)
                std::vector<float> row((size_t) c.n_vocab);
                CU(cudaMemcpy(row.data(), dev_work(0).logits + (size_t) j * c.n_rows, (size_t) c.n_vocab * 4,
                              cudaMemcpyDeviceToHost));
                cd.resize(c.n_vocab);
                for (int v = 0; v < c.n_vocab; ++v) cd[v] = {row[v], v};
            }
            const auto dist = sample_dist(cd, sc);
            if (j == B - 1) {
                next.push_back(sample_from(dist, *sc.rng));
                break;
            }
            bool kept = false;
            next.push_back(spec_pick(dist, dr[j], *sc.rng, kept));
            if (!kept) break;
        }
    }
    for (size_t j = 0; j + 1 < next.size(); ++j)   // (a stop among the accepted: it is the last)
        if (std::find(stops.begin(), stops.end(), next[j]) != stops.end()) {
            next.resize(j + 1);
            break;
        }
    const int m = (int) next.size();   // (the anchor and next[0 .. m - 2] are kept: m positions)
    if (m < B) block_rollback(m);
    if (drafter) {   // (its context for the kept positions, whichever drafted)
        ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, m, P);
        ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, m - 1);
    }
}

void forward(Model& M, const int* tok, int T, int pos0, bool all_logits, OutMode out_mode, const std::string& dump,
             std::vector<float>& logits_out, std::vector<int>* argmax_out, double& t_exp) {
    const Config& c = M.c;
    const int C = c.C;
    if ((g_fast || g_off) && T == 1 && dump.empty() && !g_prof && !g_rec && !g_trace && g_pop.empty() &&
        g_img.at(pos0) < 0) {
        fast_token(M, tok[0], pos0, out_mode, logits_out, argmax_out);
        return;
    }
    if (out_mode == OutMode::kCand) out_mode = OutMode::kLogits, g_cand.n = -1;   // (this path: the logits)
    use_dev(M.L[0].dev);
    Work* Wp = &dev_work(g_cur);
    ExpertCache* ECp = &dev_cache(g_cur);
#define W (*Wp)
#define EC (*ECp)
    // (the offload path: a prompt streams its experts through the cache without pushing the tier out - the slot
    //  it takes is the one it took last, evict_scan; IK2_SCAN=0: the usual order)
    if (g_off) EC.scan = T > 1 && !(getenv("IK2_SCAN") && !std::atoi(getenv("IK2_SCAN")));
    // the embedding rows, normed (pinned host staging, no stream sync needed)
    {
        const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
        for (int t = 0; t < T; ++t)
            k::decode_host(M.embd_type, M.embd + (size_t) tok[t] * rb, C, W.emb_h + (size_t) t * C);
        CU(cudaMemcpyAsync(W.tmp, W.emb_h, (size_t) T * C * 4, cudaMemcpyHostToDevice, g_s));
        k::rms(W.tmp, M.embd_norm, T, C, c.eps, W.h, g_s);
        place_pictures(W.h, T, pos0, C, g_s);
        for (int t = 0; t < T; ++t) W.pos_h[t] = pos0 + t;
        CU(cudaMemcpyAsync(W.pos, W.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, g_s));
    }
    int nb = 0;
    const size_t TC = (size_t) T * C;
    auto blk = [&](int i) { return W.blocks + (size_t) i * W.blk_ld * C; };
    auto dump_f = [&](const std::string& name, const float* d, size_t n) {
        if (dump.empty()) return;
        std::vector<float> h(n);
        CU(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
        FILE* f = std::fopen((dump + "/" + name + ".f32").c_str(), "wb");
        if (!f) die("cannot write in " + dump);
        std::fwrite(h.data(), 4, n, f);
        std::fclose(f);
    };

    int nb_pre = 0, n_bank = 0;   // (a looped core: the blocks before it, the states of the pass before's bank)
    for (int vis = 0; vis < c.n_visit; ++vis) {
        const int l = M.V[vis].l;   // (the layer: its weights, its experts in the cache; the state is the visit's)
        Layer& Ly = bind_visit(M, vis);
        if (Ly.dev != g_cur) {   // the next GPU's layers: the stream and the block states go along
            const int from = g_cur;
            Work& F = dev_work(from);
            Work& Tn = dev_work(Ly.dev);
            CU(cudaStreamSynchronize(g_dev[from].s));
            CU(cudaMemcpyPeer(Tn.h, Ly.dev, F.h, from, TC * 4));
            for (int i = 0; i < nb; ++i)
                CU(cudaMemcpyPeer(Tn.blocks + (size_t) i * Tn.blk_ld * C, Ly.dev, F.blocks + (size_t) i * F.blk_ld * C,
                                  from, TC * 4));
            CU(cudaMemcpyPeer(Tn.pos, Ly.dev, F.pos, from, (size_t) T * 4));
            use_dev(Ly.dev);
            Wp = &dev_work(g_cur);
            ECp = &dev_cache(g_cur);
        }
        g_pf.start();
        // A looped core.  Its first pass: the stream it is given is kept.  A later pass starts again from that stream
        // and the blocks before the core, and reads the pass before through messages: that pass's bank is the blocks
        // it added after its first (the first is the kept stream) and its final stream.
        const float* msg = nullptr;
        if (c.n_pass > 1 && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            } else {
                n_bank = 0;
                for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                    CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, blk(i), TC * 4, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
                ++n_bank;
                CU(cudaMemcpyAsync(W.h, W.lp_pre, TC * 4, cudaMemcpyDeviceToDevice, g_s));
                nb = nb_pre;
            }
        }
        if (M.V[vis].pass > 0) {   // the layer's messages: q's, k's, v's, the MoE's
            k::ResStates st{};
            for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank + (size_t) i * TC;
            // (two queries a launch: the kernel takes at most three)
            k::attn_res(st, n_bank, Ly.lp_query, 2, T, C, 1e-6f, W.lp_msg, g_s);
            k::attn_res(st, n_bank, Ly.lp_query + (size_t) 2 * C, 2, T, C, 1e-6f, W.lp_msg + 2 * TC, g_s);
            msg = W.lp_msg;
        }
        // (the drafter's taps: the stream at the start of layers 5, 11, .., 41 - and of a looped model's visits 5, 23
        // and 41, its mtp_config's target_layer_ids)
        if (g_taps) {
            const int tap = draft_tap(c, l, vis);
            if (tap >= 0)
                CU(cudaMemcpyAsync(g_taps + (size_t) tap * g_taps_ld * C, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
        }
        run_layer_attn(M, Ly, l, T, pos0, W.h, W.pos, nb, 0, W.r, W, msg);
        dump_f("blk." + std::to_string(vis) + ".attn_out", W.r, TC);
        if (l % c.block == 0) ++nb;
        // the MoE
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.r;
            if (msg) {   // (its message gates the input before the norm)
                k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, W.xq, g_s);
                const k::LoopGate gt{W.xq, msg + (size_t) 3 * TC, Ly.lp_down[3], Ly.lp_forget[3], Ly.lp_gain[3], W.xq};
                loop_gates_chunk(W, &gt, 1, T, C, c.rank, 1e-6f);
                k::rms_f16(W.xq, Ly.ffn_norm, T, C, c.eps, W.x16, W.xq, g_s);
            } else {
                k::RmsOuts outs{};
                outs.w[0] = Ly.ffn_norm;
                outs.y16[0] = W.x16;
                outs.y32[0] = W.xq;
                k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
            }
        }
        g_pf.lap(g_pf.attn);
        gemm16(Ly.router16, c.E, C, W.x16, T, W.rlog, c.E);
        const int S = T * c.K;
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + S);
        k::route(W.rlog, Ly.probs_b, T, c.E, c.K, c.route_scale, rtop, rw, g_s);
        CU(cudaMemcpyAsync(W.route_h, W.route_d, (size_t) S * 8, cudaMemcpyDeviceToHost, g_s));
        const bool cpu_lane = g_pool && g_overlap && !g_prof && T <= 8 && Ly.ty[0] == k::kNVFP4 &&
                              Ly.ty[1] == k::kNVFP4 && Ly.ty[2] == k::kNVFP4;
        if (cpu_lane) CU(cudaMemcpyAsync(W.x_h, W.xq, TC * 4, cudaMemcpyDeviceToHost, g_s));
        CU(cudaStreamSynchronize(g_s));
        const int* top = W.route_h;
        const float* w = (const float*) (W.route_h + S);
        pop_count(l, top, S);
        g_pf.lap(g_pf.route);
        if (!dump.empty()) {   // the experts picked, [T][K] int32 (compare with the reference's blk.L.picks)
            FILE* f = std::fopen((dump + "/blk." + std::to_string(vis) + ".picks.i32").c_str(), "wb");
            if (f) std::fwrite(top, 4, (size_t) S, f), std::fclose(f);
        }
        if (g_trace) {
            const int hdr[3] = {l, T, c.K};
            std::fwrite(hdr, 4, 3, g_trace);
            std::fwrite(top, 4, S, g_trace);
        }
        if (g_tp && T == 1 && g_overlap && !g_prof) {   // tensor-parallel experts, one token
            const double te0 = now();
            const int F = c.F, Fd = F / 2, Cd = C / 2;
            CU(cudaEventRecord(g_ep_x, g_dev[0].s));
            const uint8_t* canon[2][16];
            // (1) each GPU: its halves in (hits run first), its half of h for every pick into tp_h (row = the pick)
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                ExpertCache& Ed = dev_cache(d);
                const uint64_t tick = ++Ed.tick;
                int ord[16], nhit = 0, nmis = 0, mis[16];
                for (int i = 0; i < S; ++i) {
                    bool missed = false;
                    canon[d][i] = Ed.get(Ly, l, top[i], tick, g_cs, &missed, true, false);
                    if (missed) mis[nmis++] = i;
                    else ord[nhit++] = i;
                }
                for (int j = 0; j < nmis; ++j) ord[nhit + j] = mis[j];
                CU(cudaEventRecord(g_ep_cp[d], g_cs));
                const int ps = Wd.pack_slot;
                Wd.pack_slot = (Wd.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = Wd.pack_h[ps];
                uint8_t* pk_d = Wd.pack_d[ps];
                const uint8_t* blob[16];
                int zeros[16] = {}, orow[16];
                float sgv[16], suv[16];
                for (int j = 0; j < S; ++j) {
                    const int i = ord[j], e = top[i];
                    blob[j] = canon[d][i], orow[j] = i;
                    sgv[j] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                    suv[j] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                }
                std::memcpy(pk, blob, S * sizeof(void*));
                std::memcpy(pk + 512, zeros, S * 4);
                std::memcpy(pk + 768, orow, S * 4);
                std::memcpy(pk + 1024, sgv, S * 4);
                std::memcpy(pk + 1280, suv, S * 4);
                std::memcpy(pk + 1536, canon[d], S * sizeof(void*));
                CU(cudaMemcpyAsync(pk_d, pk, 1600, cudaMemcpyHostToDevice, g_s));
                const auto* pb = (const uint8_t* const*) pk_d;
                const int* ptk = (const int*) (pk_d + 512);
                const int* por = (const int*) (pk_d + 768);
                const float* psg = (const float*) (pk_d + 1024);
                const float* psu = (const float*) (pk_d + 1280);
                if (d == 1) {
                    CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
                    CU(cudaMemcpyPeerAsync(Wd.xq, 1, dev_work(0).xq, 0, (size_t) C * 4, g_s));
                }
                size_t o, b0g;
                tp_part(Ly, d, 0, o, b0g);
                for (int part = 0; part < 2; ++part) {
                    const int p0 = part ? nhit : 0, m = part ? nmis : nhit;
                    if (part) CU(cudaStreamWaitEvent(g_s, g_ep_cp[d], 0));
                    if (!m) continue;
                    k::experts_gate_up_swiglu_rows(pb + p0, ptk + p0, m, Ly.ty[0], Ly.ty[1], b0g, Wd.xq, C, Fd,
                                                   psg + p0, psu + p0, c.gate_cap, c.up_cap, Wd.tp_h + (size_t) d * Fd,
                                                   por + p0, F, g_s);
                }
                CU(cudaEventRecord(g_tp_h[d], g_s));
                Wd.tp_pk = pk_d;
            }
            // (2) each GPU: the other's half of h in, its rows of down; GPU 1's rows to GPU 0, combined there
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                const Work& O = dev_work(1 - d);
                CU(cudaStreamWaitEvent(g_s, g_tp_h[1 - d], 0));
                CU(cudaMemcpy2DAsync(Wd.tp_h + (size_t) (1 - d) * Fd, (size_t) F * 2, O.tp_h + (size_t) (1 - d) * Fd,
                                     (size_t) F * 2, (size_t) Fd * 2, S, cudaMemcpyDefault, g_s));
                size_t o, b0g, b1u;
                tp_part(Ly, d, 0, o, b0g);
                tp_part(Ly, d, 1, o, b1u);
                k::experts_down((const uint8_t* const*) (Wd.tp_pk + 1536), S, Ly.ty[2], b0g + b1u, Wd.tp_h, Cd, F,
                                Wd.y32, g_s);
                if (d == 1) {
                    CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) 8 * C, 0, Wd.y32, 1, (size_t) S * Cd * 4, g_s));
                    CU(cudaEventRecord(g_ep_done, g_s));
                }
            }
            CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            {
                const int ps = W.pack_slot;
                W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = W.pack_h[ps];
                uint8_t* pk_d = W.pack_d[ps];
                int id[16];
                float wt[16];
                for (int i = 0; i < S; ++i) {
                    id[i] = i;
                    wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][top[i]]);
                }
                std::memcpy(pk + 768, id, S * 4);
                std::memcpy(pk + 1536, wt, S * 4);
                CU(cudaMemcpyAsync(pk_d + 768, pk + 768, 1024, cudaMemcpyHostToDevice, g_s));
                k::moe_combine_f32(W.y32, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), 1, c.K, Cd, W.tmp,
                                   false, g_s);
                k::moe_combine_f32(W.y32 + (size_t) 8 * C, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), 1,
                                   c.K, Cd, W.tmp + Cd, false, g_s);
            }
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        if (g_ep && T == 1 && g_overlap && !g_prof) {
            const double te0 = now();
            const int F = c.F;
            int row_of[16], n_d[2] = {0, 0};
            const uint8_t* blob_d[2][16];
            float sg_d[2][16], su_d[2][16];
            int dev_of[16];
            for (int i = 0; i < S; ++i) dev_of[i] = ep_owner(l, top[i]);
            // GPU 1 first (it has the farther to go): x over, its picks (hits first, then the copied ones), y back
            CU(cudaEventRecord(g_ep_x, g_dev[0].s));
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                ExpertCache& Ed = dev_cache(d);
                const uint64_t tick = ++Ed.tick;
                int hit[16], mis[16], nhit = 0, nmis = 0;
                const uint8_t* where[16];
                for (int i = 0; i < S; ++i) {
                    if (dev_of[i] != d) continue;
                    bool missed = false;
                    where[i] = Ed.get(Ly, l, top[i], tick, g_cs, &missed, true, false);
                    (missed ? mis[nmis++] : hit[nhit++]) = i;
                }
                CU(cudaEventRecord(g_ep_cp[d], g_cs));
                int n = 0;
                for (int pass = 0; pass < 2; ++pass)
                    for (int j = 0; j < (pass ? nmis : nhit); ++j) {
                        const int i = pass ? mis[j] : hit[j], e = top[i];
                        blob_d[d][n] = where[i];
                        sg_d[d][n] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                        su_d[d][n] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                        row_of[i] = n++;
                    }
                n_d[d] = n;
                if (d == 1) CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
                if (!n) continue;
                const int ps = Wd.pack_slot;
                Wd.pack_slot = (Wd.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = Wd.pack_h[ps];
                uint8_t* pk_d = Wd.pack_d[ps];
                const int zeros[16] = {};
                std::memcpy(pk, blob_d[d], n * sizeof(void*));
                std::memcpy(pk + 512, zeros, n * 4);
                std::memcpy(pk + 1024, sg_d[d], n * 4);
                std::memcpy(pk + 1280, su_d[d], n * 4);
                CU(cudaMemcpyAsync(pk_d, pk, 1536, cudaMemcpyHostToDevice, g_s));
                const auto* pb = (const uint8_t* const*) pk_d;
                const int* ptk = (const int*) (pk_d + 512);
                const float* psg = (const float*) (pk_d + 1024);
                const float* psu = (const float*) (pk_d + 1280);
                if (d == 1) CU(cudaMemcpyPeerAsync(Wd.xq, 1, dev_work(0).xq, 0, (size_t) C * 4, g_s));
                for (int part = 0; part < 2; ++part) {
                    const int p0 = part ? nhit : 0, m = part ? nmis : nhit;
                    if (part) CU(cudaStreamWaitEvent(g_s, g_ep_cp[d], 0));
                    if (!m) continue;
                    k::experts_gate_up_swiglu(pb + p0, ptk + p0, m, Ly.ty[0], Ly.ty[1], Ly.eb[0], Wd.xq, C, F, psg + p0,
                                              psu + p0, c.gate_cap, c.up_cap, Wd.h16 + (size_t) p0 * F, g_s);
                    k::experts_down(pb + p0, m, Ly.ty[2], Ly.eb[0] + Ly.eb[1], Wd.h16 + (size_t) p0 * F, C, F,
                                    Wd.y32 + (size_t) p0 * C, g_s);
                }
                if (d == 1) {
                    CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) 8 * C, 0, Wd.y32, 1, (size_t) n * C * 4, g_s));
                    CU(cudaEventRecord(g_ep_done, g_s));
                }
            }
            // on GPU 0 (current): the combine over both GPUs' rows (GPU 1's from row 8)
            if (n_d[1]) CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            {
                const int ps = W.pack_slot;
                W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = W.pack_h[ps];
                uint8_t* pk_d = W.pack_d[ps];
                int id[16];
                float wt[16];
                for (int i = 0; i < S; ++i) {
                    id[i] = (dev_of[i] ? 8 : 0) + row_of[i];
                    wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][top[i]]);
                }
                std::memcpy(pk + 768, id, S * 4);
                std::memcpy(pk + 1536, wt, S * 4);
                CU(cudaMemcpyAsync(pk_d + 768, pk + 768, 1024, cudaMemcpyHostToDevice, g_s));
                k::moe_combine_f32(W.y32, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), T, c.K, C, W.tmp,
                                   false, g_s);
            }
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        if (g_fast && T > 1 && S <= W.rg_slots && !Ly.tab_h.empty()) {
            // Every expert in VRAM and a prompt of a few tokens (a chat turn): the experts straight over their
            // blocks, an expert's blocks read once for up to kGroupMax of the tokens that picked it.  (The GEMM path
            // below decodes a whole expert to FP16 for the two or three tokens it has: a 47-token prompt took 0.34 s.)
            const double te0 = now();
            const int GM = W.rg_groups;
            const size_t o_n = (size_t) GM * 8, o_tok = o_n + (size_t) GM * 4, o_row = o_tok + (size_t) GM * k::kGroupMax * 4,
                         o_sg = o_row + (size_t) GM * k::kGroupMax * 4, o_su = o_sg + (size_t) W.rg_slots * 4,
                         o_wt = o_su + (size_t) W.rg_slots * 4, o_id = o_wt + (size_t) W.rg_slots * 4;
            auto** gb = (const uint8_t**) W.rg_h;
            int *gn = (int*) (W.rg_h + o_n), *gtok = (int*) (W.rg_h + o_tok), *grow = (int*) (W.rg_h + o_row);
            float *sg = (float*) (W.rg_h + o_sg), *su = (float*) (W.rg_h + o_su), *wt = (float*) (W.rg_h + o_wt);
            int* id = (int*) (W.rg_h + o_id);
            if (++W.stamp == 0) {
                std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                W.stamp = 1;
            }
            const uint32_t stp = W.stamp;
            int G = 0;
            for (int i = 0; i < S; ++i) {   // slot i (token i / K, one of its picks): row i of h16 and y32
                const int e = top[i];
                int g = W.gi_stamp[e] == stp ? W.gi_idx[e] : -1;
                if (g < 0 || gn[g] == k::kGroupMax) {   // the expert's first group, or its next
                    g = G++;
                    W.gi_stamp[e] = stp, W.gi_idx[e] = g;
                    gb[g] = Ly.tab_h[e], gn[g] = 0;
                }
                gtok[(size_t) g * k::kGroupMax + gn[g]] = i / c.K;
                grow[(size_t) g * k::kGroupMax + gn[g]] = i;
                ++gn[g];
                sg[i] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                su[i] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
                id[i] = i;
            }
            // (the stream was synchronised after the routing: the copy of the last layer's tables is done)
            CU(cudaMemcpyAsync(W.rg_d, W.rg_h, W.rg_bytes, cudaMemcpyHostToDevice, g_s));
            k::experts_gate_up_swiglu_g((const uint8_t* const*) W.rg_d, (const int*) (W.rg_d + o_n),
                                        (const int*) (W.rg_d + o_tok), (const int*) (W.rg_d + o_row), G, Ly.ty[0],
                                        Ly.ty[1], Ly.eb[0], W.xq, C, c.F, (const float*) (W.rg_d + o_sg),
                                        (const float*) (W.rg_d + o_su), c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_down_g((const uint8_t* const*) W.rg_d, (const int*) (W.rg_d + o_n), (const int*) (W.rg_d + o_row),
                              G, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            k::moe_combine_f32(W.y32, (const int*) (W.rg_d + o_id), (const float*) (W.rg_d + o_wt), T, c.K, C, W.tmp,
                               false, g_s);
            g_pf.lap(g_pf.experts);
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            g_pf.lap(g_pf.rest);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        if (g_mv && S <= 64) {
            const double te0 = now();
            // the token path: each slot (token t's pick i) its expert's blob straight from the cache
            // The slots already in VRAM first, then the ones copied in now: the first group's experts run while the
            // second's blocks are on their way (the copies on a stream of their own).
            const uint8_t* blob[64];
            int tk[64], id[64];
            float sg[64], su[64], wt[64];
            const uint64_t tick = ++EC.tick;
            const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
            const uint8_t* where[64];
            char late[64] = {};
            int slot_of[64];
            if (++W.stamp == 0) {
                std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                W.stamp = 1;
            }
            const uint32_t stp = W.stamp;
            lazy_poll();
            double acc_up = 0, acc_cpu = 0;
            int urgent[64], lazy[64];
            int n_urgent = 0, n_lazy = 0;
            for (int i = 0; i < S; ++i) {
                const int e = top[i];
                if (W.seen_stamp[e] == stp) {   // a second token of this call on an expert missing in it
                    late[i] = W.seen_late[e];
                    where[i] = EC.has(l, e) ? EC.get(Ly, l, e, tick, cst, nullptr, false, false) : nullptr;
                    slot_of[i] = where[i] ? EC.last_handle : -1;
                    continue;
                }
                if (EC.has(l, e)) {
                    where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                    slot_of[i] = EC.last_handle;
                    if (EC.slot_sseq(EC.last_handle) > g_seq_done) late[i] = 1;   // still arriving from a lazy copy
                    continue;
                }
                // a miss: the CPU's or the link's, whichever is free first
                if (cpu_lane && acc_cpu + g_t_cpu <= acc_up + g_t_up) {
                    late[i] = 2, acc_cpu += g_t_cpu;
                    if (g_cpu_lazy) {   // a slot for its blocks, copied in afterwards
                        where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                        slot_of[i] = EC.last_handle;
                        lazy[n_lazy++] = i;
                    } else {            // no slot taken: nothing in VRAM is evicted for it
                        ++EC.misses;
                        where[i] = nullptr, slot_of[i] = -1;
                    }
                } else {
                    late[i] = 1, acc_up += g_t_up;
                    where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                    slot_of[i] = EC.last_handle;
                    urgent[n_urgent++] = i;
                }
                W.seen_stamp[e] = stp;
                W.seen_late[e] = late[i];
            }
            int nh = 0, nw = 0;
            for (int i = 0; i < S; ++i) nh += late[i] == 0, nw += late[i] == 1;
            for (int ui = 0; ui < n_urgent; ++ui) {
                const int i = urgent[ui];
                EC.copy_in(Ly, top[i], (uint8_t*) where[i], cst);
            }
            if (nw && cst != g_s) CU(cudaEventRecord(g_ev, g_cs));
            if (n_lazy > 0) {
                const uint64_t seq = g_seq_next++;
                if (seq > g_seq_done + kRing - 1) {   // the ring's entry is still an unfinished batch's
                    CU(cudaStreamSynchronize(g_cs));
                    g_seq_done = seq - 1;
                }
                for (int li = 0; li < n_lazy; ++li) {
                    const int i = lazy[li];
                    EC.copy_in(Ly, top[i], (uint8_t*) where[i], g_cs);
                    EC.slot_sseq(slot_of[i]) = seq;
                }
                CU(cudaEventRecord(g_ring[seq % kRing], g_cs));
            }
            const int ps = W.pack_slot;
            W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
            uint8_t* pk = W.pack_h[ps];
            uint8_t* pk_d = W.pack_d[ps];
            float* yh = W.y_h[ps];
            int ph = 0, pm = nh, pc = nh + nw;
            ik2::cpu::ExpertJob jobs[64];
            const int n_jobs = S - nh - nw;
            const double tc0 = now();
            for (int i = 0; i < S; ++i) {
                const int e = top[i], p = late[i] == 2 ? pc++ : late[i] ? pm++ : ph++;
                blob[p] = where[i];
                tk[p] = i / c.K;
                id[i] = p;
                sg[p] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                su[p] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
                if (late[i] == 2) {
                    auto& J = jobs[p - nh - nw];
                    J.gate = Ly.host[0] + (size_t) e * Ly.stride[0];
                    J.up = Ly.host[1] + (size_t) e * Ly.stride[1];
                    J.down = Ly.host[2] + (size_t) e * Ly.stride[2];
                    J.x = W.x_h + (size_t) (i / c.K) * C;
                    J.sg = sg[p], J.su = su[p];
                    J.y = yh + (size_t) (p - nh - nw) * C;
                    ik2::cpu::post_expert(*g_pool, &J, C, c.F, c.gate_cap, c.up_cap);
                }
            }
            g_n_cpu += n_jobs, g_n_wait += nw;
            // the groups: a part's (0: in VRAM, 1: waited for) slots by expert, each expert's blocks read once
            int ng[2] = {0, 0};
            const uint8_t* gb[64];
            int gn[64];
            int gtok[64 * k::kGroupMax];
            int grow[64 * k::kGroupMax];
            int G = 0;
            for (int part = 0; part < 2; ++part) {
                if (++W.stamp == 0) {
                    std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                    std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                    W.stamp = 1;
                }
                const uint32_t gstp = W.stamp;
                for (int i = 0; i < S; ++i) {
                    if (late[i] != part) continue;
                    const int e = top[i];
                    int g;
                    if (W.gi_stamp[e] != gstp) {
                        W.gi_stamp[e] = gstp;
                        g = G++;
                        W.gi_idx[e] = g;
                        gb[g] = where[i];
                        gn[g] = 0;
                        ++ng[part];
                    } else {
                        g = W.gi_idx[e];
                    }
                    gtok[(size_t) g * k::kGroupMax + gn[g]] = i / c.K;
                    grow[(size_t) g * k::kGroupMax + gn[g]] = id[i];
                    ++gn[g];
                }
            }
            g_pf.lap(g_pf.upload);
            // one copy of the tables from pinned memory: [64] pointers, then tok, inv, sg, su, wt [64] each
            std::memcpy(pk, blob, S * sizeof(void*));
            std::memcpy(pk + 512, tk, S * 4);
            std::memcpy(pk + 768, id, S * 4);
            std::memcpy(pk + 1024, sg, S * 4);
            std::memcpy(pk + 1280, su, S * 4);
            std::memcpy(pk + 1536, wt, S * 4);
            // ... and the groups: [64] pointers, n [64], tok [64][8], row [64][8]
            std::memcpy(pk + 1792, gb, G * sizeof(void*));
            std::memcpy(pk + 2304, gn, G * 4);
            std::memcpy(pk + 2560, gtok, (size_t) G * k::kGroupMax * 4);
            std::memcpy(pk + 4608, grow, (size_t) G * k::kGroupMax * 4);
            CU(cudaMemcpyAsync(pk_d, pk, 6656, cudaMemcpyHostToDevice, g_s));
            const auto* pgb = (const uint8_t* const*) (pk_d + 1792);
            const int* pgn = (const int*) (pk_d + 2304);
            const int* pgt = (const int*) (pk_d + 2560);
            const int* pgr = (const int*) (pk_d + 4608);
            const auto* pb = (const uint8_t* const*) pk_d;
            const int* ptk = (const int*) (pk_d + 512);
            const int* pid = (const int*) (pk_d + 768);
            const float* psg = (const float*) (pk_d + 1024);
            const float* psu = (const float*) (pk_d + 1280);
            const float* pwt = (const float*) (pk_d + 1536);
            const int F = c.F;
            for (int part = 0; part < 2; ++part) {
                const int p0 = part ? nh : 0, n = part ? nw : nh;
                if (part && cst != g_s) CU(cudaStreamWaitEvent(g_s, g_ev, 0));
                if (!n) continue;
                if (g_group && g_group_exp && T > 1) {
                    const int g0 = part ? ng[0] : 0, gc = ng[part];
                    k::experts_gate_up_swiglu_g(pgb + g0, pgn + g0, pgt + (size_t) g0 * k::kGroupMax,
                                                pgr + (size_t) g0 * k::kGroupMax, gc, Ly.ty[0], Ly.ty[1], Ly.eb[0],
                                                W.xq, C, F, psg, psu, c.gate_cap, c.up_cap, W.h16, g_s);
                    k::experts_down_g(pgb + g0, pgn + g0, pgr + (size_t) g0 * k::kGroupMax, gc, Ly.ty[2],
                                      Ly.eb[0] + Ly.eb[1], W.h16, C, F, W.y32, g_s);
                    continue;
                }
                k::experts_gate_up_swiglu(pb + p0, ptk + p0, n, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq, C, F, psg + p0,
                                          psu + p0, c.gate_cap, c.up_cap, W.h16 + (size_t) p0 * F, g_s);
                k::experts_down(pb + p0, n, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16 + (size_t) p0 * F, C, F,
                                W.y32 + (size_t) p0 * C, g_s);
            }
            if (n_jobs > 0) {
                g_pool->wait();
                g_t_cpu = 0.9 * g_t_cpu + 0.1 * (now() - tc0) / n_jobs;
                CU(cudaMemcpyAsync(W.y32 + (size_t) (nh + nw) * C, yh, (size_t) n_jobs * C * 4, cudaMemcpyHostToDevice,
                                   g_s));
            }
            k::moe_combine_f32(W.y32, pid, pwt, T, c.K, C, W.tmp, false, g_s);
            g_pf.lap(g_pf.experts);
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            g_pf.lap(g_pf.rest);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        run_prefill_moe(M, Ly, l, T, W.x16, top, w, W.r, W.h, W, EC, t_exp);
        g_pf.lap(g_pf.rest);
        dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
    }
    run_head(M, T, all_logits, out_mode, nb, W.h, W, logits_out, argmax_out);
#undef W
#undef EC
}

// Layer-major super-chunk prefill (single GPU, no dump, no drafter):
// A long prompt in fewer passes: the prefill scratch for a pass of T tokens (~256 KB a token) is lent by the expert
// cache (ExpertCache::lend), which a long prefill streams through anyway, and given back afterwards - so the passes
// can be as long as the cache allows without the decode's cache being smaller (IK2_BORROW=0: off).
struct BigScratch {
    int pool = -1, T_sup = 0;
    size_t blk_ld = 0;
    float *h = nullptr, *r = nullptr, *tmp = nullptr, *blocks = nullptr, *gu32 = nullptr, *qkv = nullptr,
          *y32 = nullptr, *sg = nullptr, *su = nullptr, *wt = nullptr, *wtr = nullptr, *rlog = nullptr;
    uint16_t *xq16_sup = nullptr, *ex16 = nullptr, *h16 = nullptr;
    int *pos = nullptr, *route_d = nullptr, *gidx = nullptr, *inv = nullptr;
    float* lg = nullptr;
    uint16_t* lg16 = nullptr;
    size_t lg_cap = 0;
};
// the scratch of a T-token pass, carved from base (or only its size, base null)
size_t big_carve(const Config& c, int T, size_t gu32_have, size_t lg_want, size_t lg_have, Work* W, uint8_t* base) {
    const size_t C = c.C, F = c.F, S = (size_t) T * c.K, nblk = c.n_layer / c.block + 1;
    const size_t Tr = std::max<size_t>(64, T);
    size_t off = 0;
    auto take = [&](size_t bytes) {
        uint8_t* p = base ? base + off : nullptr;
        off += (bytes + 255) / 256 * 256;
        return p;
    };
    float* h = (float*) take(T * C * 4);
    float* r = (float*) take(T * C * 4);
    float* tmp = (float*) take(T * C * 4);
    float* blocks = (float*) take(nblk * T * C * 4);
    uint16_t* xq = (uint16_t*) take(T * C * 2);
    int* pos = (int*) take((size_t) T * 4);
    float* rlog = (float*) take((size_t) T * c.E * 4);
    int* route = (int*) take(S * 8);
    uint16_t* ex16 = (uint16_t*) take(T * C * 2);
    uint16_t* h16 = (uint16_t*) take(Tr * F * 2);
    const bool new_gu = (size_t) T * 2 * F * 4 > gu32_have;
    float* gu32 = new_gu ? (float*) take((size_t) T * 2 * F * 4) : nullptr;
    float* y32 = (float*) take(Tr * C * 4);
    float* sg = (float*) take(S * 4);
    float* su = (float*) take(S * 4);
    float* wt = (float*) take(S * 4);
    float* wtr = (float*) take(S * 4);
    int* gidx = (int*) take(S * 4);
    int* inv = (int*) take(S * 4);
    // the attention's logits and probabilities, room for kAttnRows queries at the prompt's end (lg_want of each)
    const bool new_lg = lg_want > lg_have;
    float* lg = new_lg ? (float*) take(lg_want * 4) : nullptr;
    uint16_t* lg16 = new_lg ? (uint16_t*) take(lg_want * 2) : nullptr;
    if (W && base) {
        W->h = h, W->r = r, W->tmp = tmp, W->blocks = blocks, W->xq16_sup = xq, W->pos = pos, W->rlog = rlog;
        W->route_d = route, W->ex16 = ex16, W->h16 = h16, W->y32 = y32;
        W->sg = sg, W->su = su, W->wt = wt, W->wtr = wtr, W->gidx = gidx, W->inv = inv;
        if (new_gu) W->gu32 = gu32, W->qkv = gu32, W->lg16 = (uint16_t*) gu32;
        if (new_lg) W->lg = lg, W->lg16 = lg16, W->lg_cap = lg_want;
        W->blk_ld = (size_t) T;
        W->T_sup = T;
    }
    return off;
}
// Tokens a pass for a prompt of n new tokens: W.T_sup, or more with a lent scratch (then big_end gives it back).
int big_begin(const Model& M, Work& W, ExpertCache& EC, int n, int ctx_end, BigScratch& bs) {
    bs.pool = -1;
    // (the offload path lends none of the tier: what a prompt took would be gone from it for the decode)
    if (W.T_sup <= 0 || dual() || (g_off && EC.tier_only) || (getenv("IK2_BORROW") && !std::atoi(getenv("IK2_BORROW"))))
        return W.T_sup;
    const Config& c = M.c;
    // the attention's scratch for 256 queries against the whole context: few, large GEMMs (the default ~400 MB left
    // room for only ~20 queries a step at 100k tokens) - also lent for a short prompt at the end of a long context
    const size_t lg_want = (g_flash && !g_kv_fp8 && c.D == 128) ? 0 : (size_t) c.H * 256 * (size_t) ctx_end;
    const bool long_ctx = lg_want > W.lg_cap;
    if (n <= W.T_sup && !long_ctx) return W.T_sup;
    int p = -1;
    size_t most = 0;
    for (size_t i = 0; i < EC.pools.size(); ++i) {
        const size_t b = (size_t) EC.pools[i].n_active * EC.pools[i].slot_bytes;
        if (b > most) most = b, p = (int) i;
    }
    if (p < 0) return W.T_sup;
    const auto& P = EC.pools[p];
    const int k_max = P.n_active - (2 * c.E + 80);   // (prefetch_layer still needs two layers' slots)
    if (k_max <= 0) return W.T_sup;
    const size_t avail = (size_t) k_max * P.slot_bytes;
    const size_t lg_bytes = long_ctx ? (lg_want * 4 + 255) / 256 * 256 + lg_want * 2 + 256 : 0;
    if (lg_bytes >= avail) return W.T_sup;
    int T = W.T_sup;
    if (n > W.T_sup) {
        const size_t per_tok = big_carve(c, 1 << 14, W.gu32_bytes, 0, 0, nullptr, nullptr) >> 14;
        const int T_cap = (int) std::min<size_t>((size_t) W.T_host, (avail - lg_bytes) / (per_tok + 1));
        if (T_cap > W.T_sup) {
            const int passes = (n + T_cap - 1) / T_cap;
            T = (n + passes - 1) / passes;
            while (T > W.T_sup && big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, nullptr, nullptr) > avail) T -= 256;
            if (T < W.T_sup) T = W.T_sup;
        }
    }
    if (T == W.T_sup && !long_ctx) return W.T_sup;
    const size_t bytes = T > W.T_sup ? big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, nullptr, nullptr) : lg_bytes;
    const int k = (int) ((bytes + P.slot_bytes - 1) / P.slot_bytes);
    CU(cudaStreamSynchronize(g_cs));
    CU(cudaStreamSynchronize(g_s));
    g_seq_done = g_seq_next - 1;
    uint8_t* base = EC.lend(p, k);
    if (!base) return W.T_sup;
    bs.h = W.h, bs.r = W.r, bs.tmp = W.tmp, bs.blocks = W.blocks, bs.gu32 = W.gu32, bs.qkv = W.qkv, bs.y32 = W.y32;
    bs.sg = W.sg, bs.su = W.su, bs.wt = W.wt, bs.wtr = W.wtr, bs.rlog = W.rlog, bs.xq16_sup = W.xq16_sup;
    bs.ex16 = W.ex16, bs.h16 = W.h16, bs.pos = W.pos, bs.route_d = W.route_d, bs.gidx = W.gidx, bs.inv = W.inv;
    bs.blk_ld = W.blk_ld, bs.T_sup = W.T_sup;
    bs.lg = W.lg, bs.lg16 = W.lg16, bs.lg_cap = W.lg_cap;
    if (T > W.T_sup) {
        big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, &W, base);
    } else {   // only the attention's scratch
        W.lg = (float*) base;
        W.lg16 = (uint16_t*) (base + (lg_want * 4 + 255) / 256 * 256);
        W.lg_cap = lg_want;
    }
    bs.pool = p;
    return T;
}
void big_end(Work& W, ExpertCache& EC, BigScratch& bs) {
    if (bs.pool < 0) return;
    CU(cudaStreamSynchronize(g_cs));
    CU(cudaStreamSynchronize(g_s));
    W.h = bs.h, W.r = bs.r, W.tmp = bs.tmp, W.blocks = bs.blocks, W.gu32 = bs.gu32, W.qkv = bs.qkv, W.y32 = bs.y32;
    W.sg = bs.sg, W.su = bs.su, W.wt = bs.wt, W.wtr = bs.wtr, W.rlog = bs.rlog, W.xq16_sup = bs.xq16_sup;
    W.ex16 = bs.ex16, W.h16 = bs.h16, W.pos = bs.pos, W.route_d = bs.route_d, W.gidx = bs.gidx, W.inv = bs.inv;
    W.blk_ld = bs.blk_ld, W.T_sup = bs.T_sup;
    W.lg = bs.lg, W.lg16 = bs.lg16, W.lg_cap = bs.lg_cap;
    EC.give_back(bs.pool);
    bs.pool = -1;
}

// Sweeps each layer once across up to W.T_sup tokens (sub-chunked attention for KV/attention scratch, then one combined MoE pass).
void forward_prefill_sup(Model& M, const int* tok, int T, int pos0, int sub_chunk, OutMode out_mode,
                         std::vector<float>& logits_out, std::vector<int>* argmax_out, double& t_exp) {
    const Config& c = M.c;
    const int C = c.C;
    use_dev(M.L[0].dev);
    Work& W = dev_work(g_cur);
    ExpertCache& EC = dev_cache(g_cur);

    // Embed and norm all T tokens in sub-chunks of <= W.Tm
    {
        const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
        for (int off = 0; off < T; off += W.Tm) {
            const int ts = std::min(W.Tm, T - off);
            for (int t = 0; t < ts; ++t)
                k::decode_host(M.embd_type, M.embd + (size_t) tok[off + t] * rb, C, W.emb_h + (size_t) t * C);
            CU(cudaMemcpyAsync(W.tmp, W.emb_h, (size_t) ts * C * 4, cudaMemcpyHostToDevice, g_s));
            k::rms(W.tmp, M.embd_norm, ts, C, c.eps, W.h + (size_t) off * C, g_s);
            place_pictures(W.h + (size_t) off * C, ts, pos0 + off, C, g_s);
            if (off + ts < T) CU(cudaStreamSynchronize(g_s));   // (W.emb_h reused on next iteration)
        }
        for (int t = 0; t < T; ++t) W.pos_h[t] = pos0 + t;
        CU(cudaMemcpyAsync(W.pos, W.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, g_s));
    }

    // (enough tokens that a layer uses most of its experts; below, the misses copied after the routing cost less than
    // prefetching experts that turn out unused - a 119-token turn took 3.4 s with 2E, 1.9 s without)
    const bool pf = (size_t) T * c.K >= (size_t) 8 * c.E;
    EC.scan = pf && !(getenv("IK2_SCAN") && !std::atoi(getenv("IK2_SCAN")));
    if (dual()) dev_cache(1).scan = EC.scan;
    // (under EP each GPU prefetches the experts it owns, by GPU 0's picks so far)
    // One layer ahead: two or three layers in flight cost more than they hide - their slots squeeze the cache (2x RTX
    // PRO 6000, a 21.8k prompt: copies 244 -> 328 GiB, 8.05 -> 9.73 s); the next layer's prefetch issued before this
    // layer's routing round trip instead of after its experts are queued was slower too
    const int depth = 1;
    auto prefetch = [&](int ll, int ahead) {
        if (!dual()) {
            prefetch_layer(M, ll, W, EC, T, nullptr, -1, ahead);
            return;
        }
        use_dev(1);
        prefetch_layer(M, ll, dev_work(1), dev_cache(1), T, &W, g_tp ? -1 : 1, ahead);
        use_dev(0);
        prefetch_layer(M, ll, W, EC, T, nullptr, g_tp ? -1 : 0, ahead);
    };
    if (pf)
        for (int ll = 0; ll < depth; ++ll) prefetch(ll, ll + 1);
    int nb = 0;
    for (int vis = 0; vis < c.n_visit; ++vis) {   // (no loop here: a looped model's prompts go through forward())
        const int l = M.V[vis].l;
        Layer& Ly = bind_visit(M, vis);
        g_pf.start();
        for (int off = 0; off < T; off += sub_chunk) {
            const int ts = std::min(sub_chunk, T - off);
            run_layer_attn(M, Ly, l, ts, pos0 + off, W.h + (size_t) off * C, W.pos + off, nb, off,
                           W.r + (size_t) off * C, W);
        }
        if (l % c.block == 0) ++nb;
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + (size_t) i * W.blk_ld * C;
            st.st[nb] = W.r;
            k::RmsOuts outs{};
            outs.w[0] = Ly.ffn_norm;
            outs.y16[0] = W.xq16_sup;
            k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
        }
        g_pf.lap(g_pf.attn);
        gemm16(Ly.router16, c.E, C, W.xq16_sup, T, W.rlog, c.E);
        const int S = T * c.K;
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + S);
        k::route(W.rlog, Ly.probs_b, T, c.E, c.K, c.route_scale, rtop, rw, g_s);
        CU(cudaMemcpyAsync(W.route_h, W.route_d, (size_t) S * 8, cudaMemcpyDeviceToHost, g_s));
        CU(cudaStreamSynchronize(g_s));
        const int* top_h = W.route_h;
        const float* w_h = (const float*) (W.route_h + S);
        pop_count(l, top_h, S);
        g_pf.lap(g_pf.route);
        if (g_trace) {
            const int hdr[3] = {l, T, c.K};
            std::fwrite(hdr, 4, 3, g_trace);
            std::fwrite(top_h, 4, S, g_trace);
        }
        run_prefill_moe(M, Ly, l, T, W.xq16_sup, top_h, w_h, W.r, W.h, W, EC, t_exp);
        if (pf) prefetch(l + depth, depth);
        g_pf.lap(g_pf.rest);
    }
    EC.scan = false;
    if (dual()) dev_cache(1).scan = false;

    if (out_mode != OutMode::kNone) {
        g_pf.start();
        const int last = T - 1;
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + ((size_t) i * W.blk_ld + last) * C;
        st.st[nb] = W.h + (size_t) last * C;
        k::RmsOuts outs{};
        outs.w[0] = M.out_norm;
        outs.y16[0] = W.x16;
        outs.y32[0] = g_dense_q8 ? W.xq : nullptr;
        k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
        matmul(M.out_w, W.xq, W.x16, 1, W.logits, c.n_rows);
        if (out_mode == OutMode::kArgmax) {
            argmax_out->resize(1);
            k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d, g_s);
            CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
            (*argmax_out)[0] = W.argmax_h[0];
        } else {
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            const size_t at = logits_out.size();
            logits_out.resize(at + (size_t) c.n_vocab);
            CU(cudaMemcpyAsync(logits_out.data() + at, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
        }
        g_pf.lap(g_pf.rest);
    } else {
        CU(cudaStreamSynchronize(g_s));
    }
}

// every short conv's history as after the first n of the verify window's tokens
void rollback(Model& M, int n) {
    const int C = M.c.C, KD = M.c.KV * M.c.D;
    for (int vis = 0; vis < (int) M.V.size(); ++vis) {
        Layer& Ly = bind_visit(M, vis);
        float* hist[4] = {Ly.h_k, Ly.h_v, Ly.h_attn, Ly.h_mlp};
        const int cc[4] = {KD, KD, C, C};
        for (int i = 0; i < 4; ++i) k::conv_hist_after(Ly.rec_old[i], Ly.rec_x[i], n, cc[i], hist[i], g_s);
    }
}

// the embedding row of token t, normed as the model's input, into out (device [C])
void embed_row(Model& M, int t, float* raw, float* out) {
    const int C = M.c.C;
    float* e = dev_work(M.L[0].dev).emb_h;
    const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    k::decode_host(M.embd_type, M.embd + (size_t) t * rb, C, e);
    CU(cudaMemcpyAsync(raw, e, (size_t) C * 4, cudaMemcpyHostToDevice, g_s));
    k::rms(raw, M.embd_norm, 1, C, M.c.eps, out, g_s);
}

}  // namespace

// The Inkling 2 preview: Linux, one NVIDIA GPU of 24 GB, Ampere (sm_80) or newer.  Outside it the engine stops with one
// line saying why; IK2_UNSUPPORTED=1 runs it anyway where it can run at all (two GPUs, smaller cards with the experts
// streamed from RAM, Windows: untested).
// Greedy decode on the resident path from `anchor` (chosen, not run yet) at position P until out holds gen tokens
// (the anchor first): each step a plain token or a window of the drafter's or the prompt lookup's drafts (ctl.pick);
// lk holds the text before the anchor.  The CLI's generation and --drafting auto's trials.
void decode_greedy(Model& M, ik2::draft::Drafter* drafter, int anchor, int P, int gen, int ctx, SpecCtl& ctl,
                   strata::spec::SuffixDrafter& lk, std::vector<int>& out, double* t_draft) {
    double t_exp = 0;
    const SampleCfg sc;
    std::vector<int> next;
    out.push_back(anchor);
    while ((int) out.size() < gen && P < ctx) {
        lk.append(anchor);
        const int room = std::min(gen - (int) out.size(), ctx - P);
        int32_t ld[kBlockMax];
        const int k = g_lookup && ctl.use_lookup && room >= 2 ? lk.propose(std::min(room, kBlockMax) - 1, ld) : 0;
        int B = 1;
        int src = ctl.pick(room, k, B);
        if (src == 1 && !drafter) src = 0, B = 1;
        const double ts = now();
        if (B < 2) {
            std::vector<float> lo;
            std::vector<int> ag;
            forward(M, &anchor, 1, P, false, OutMode::kArgmax, std::string(), lo, &ag, t_exp);
            if (drafter) {
                ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, 1, P);
                ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, 0);
            }
            CU(cudaStreamSynchronize(g_s));
            ctl.took_plain(now() - ts);
            anchor = ag[0], P += 1;
            out.push_back(anchor);
            continue;
        }
        int given[kBlockMax];
        for (int j = 0; j + 1 < B; ++j) given[j] = ld[j];
        spec_step(M, drafter, anchor, P, B, sc, std::vector<int>{}, next, src == 2 ? given : nullptr, t_draft);
        CU(cudaStreamSynchronize(g_s));
        if (src == 2) ctl.took_lookup(B, (int) next.size() - 1, now() - ts);
        else ctl.took_spec(B, (int) next.size() - 1, now() - ts);
        for (int j = 0; j + 1 < (int) next.size(); ++j) lk.append(next[j]);
        for (int j = 0; j < (int) next.size() && (int) out.size() < gen; ++j) out.push_back(next[j]);
        P += (int) next.size();
        anchor = next.back();
    }
}

// --drafting auto: which of no drafting, the prompt lookup, the drafter, and both decodes fastest on this PC -
// measured once at the first start (before READY, on IK2_DRAFT_IDS's requests: the server's chat, code and tool call
// in the model's own template) and remembered in IK2_DRAFT_FILE (the server's data folder) a line a key: the GPU, its
// driver, the model and the drafter files, the context and the cache; IK2_DRAFT_MEASURE=1 measures again.  A line:
// "<key>\t<none> <lookup> <drafter> <both> tok/s <date>" (-1: not measured here).
const char* kDraftNames[4] = {"no drafting", "the prompt lookup", "the drafter", "the drafter and the prompt lookup"};
std::string draft_key(const std::string& model, const std::string& draft_path, int ctx) {
    cudaDeviceProp p{};
    int dev = 0, drv = 0;
    CU(cudaGetDevice(&dev));
    CU(cudaGetDeviceProperties(&p, dev));
    (void) cudaDriverGetVersion(&drv);
    auto fsize = [](const std::string& f) -> long long {
        std::ifstream in(f, std::ios::binary | std::ios::ate);
        return in ? (long long) in.tellg() : -1;
    };
    const std::string base = model.substr(model.find_last_of("/\\") + 1);
    char buf[640];
    std::snprintf(buf, sizeof buf, "ik2 drafting v1|%s %.0f GB sm_%d%d|driver %d|%s %lld|drafter %lld|ctx %d %s",
                  p.name, p.totalGlobalMem / 1e9, p.major, p.minor, drv, base.c_str(), fsize(model),
                  draft_path.empty() ? -1LL : fsize(draft_path), ctx, g_kv_fp8 ? "fp8" : "f16");
    return buf;
}
// the remembered choice (0 .. 3, kDraftNames), or -1; tps: its four figures
int draft_remembered(const std::string& key, double* tps, std::string& date) {
    const char* f = getenv("IK2_DRAFT_FILE");
    const char* again = getenv("IK2_DRAFT_MEASURE");
    if (!f || !*f || (again && std::atoi(again))) return -1;
    std::ifstream in(f);
    std::string line;
    int c = -1;
    while (std::getline(in, line)) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos || line.compare(0, tab, key) != 0) continue;
        char d[64] = "";
        int ch = -1;
        if (std::sscanf(line.c_str() + tab + 1, "%lf %lf %lf %lf %d %63s", &tps[0], &tps[1], &tps[2], &tps[3], &ch,
                        d) == 6 && ch >= 0 && ch < 4)
            c = ch, date = d;   // (the last line of the key wins)
    }
    return c;
}
void draft_remember(const std::string& key, const double* tps, int c) {
    const char* f = getenv("IK2_DRAFT_FILE");
    if (!f || !*f) return;
    std::vector<std::string> keep;
    {
        std::ifstream in(f);
        std::string line;
        while (std::getline(in, line))
            if (line.compare(0, key.size() + 1, key + "\t") != 0) keep.push_back(line);
    }
    char day[32];
    const std::time_t tt = std::time(nullptr);
    std::strftime(day, sizeof day, "%Y-%m-%d", std::localtime(&tt));
    char vals[200];
    std::snprintf(vals, sizeof vals, "%.3f %.3f %.3f %.3f %d %s", tps[0], tps[1], tps[2], tps[3], c, day);
    keep.push_back(key + "\t" + vals);
    const std::string tmp = std::string(f) + ".tmp";
    {
        std::ofstream out(tmp);
        for (const auto& l : keep) out << l << "\n";
        if (!out) return;
    }
    std::remove(f);
    std::rename(tmp.c_str(), f);
}

// --check-sampling (no GPU, no model): sampled drafting keeps the distribution - a drafted position's pick (spec_pick:
// the draft with the model's probability of it, else the distribution without it) against the plain pick
// (sample_from) of the same distribution, a chi-square of each against the probabilities themselves, on candidate
// lists like the engine's (random logits, temperatures, top_p / top_k) and drafts likely, unlikely and impossible.
// Exit 0: every p-value above 1e-4.
int check_sampling() {
    std::mt19937_64 rng(20261121);
    int fails = 0;
    auto chi_p = [](double chi, int df) {   // (upper tail, Wilson-Hilferty)
        if (df <= 0) return 1.0;
        const double z = (std::cbrt(chi / df) - (1.0 - 2.0 / (9.0 * df))) / std::sqrt(2.0 / (9.0 * df));
        return 0.5 * std::erfc(z / std::sqrt(2.0));
    };
    const double temps[3] = {0.7, 1.0, 1.3}, tops[3] = {1.0, 0.9, 0.95};
    const int topks[3] = {0, 20, 0};
    for (int cs = 0; cs < 3; ++cs) {
        std::vector<std::pair<float, int>> cd(64);
        for (int i = 0; i < 64; ++i) cd[i] = {(float) std::normal_distribution<double>(0, 2.0)(rng), 1000 + i};
        SampleCfg sc;
        sc.temp = temps[cs], sc.top_p = tops[cs], sc.top_k = topks[cs];
        const auto dist = sample_dist(cd, sc);
        const int drafts[3] = {dist[0].first, dist[std::min<size_t>(3, dist.size() - 1)].first, 99999};
        for (int dk = 0; dk < 4; ++dk) {   // (dk 3: the plain pick)
            std::map<int, int> n;
            const int N = 400000;
            int kept = 0;
            for (int i = 0; i < N; ++i) {
                bool acc = false;
                const int t = dk == 3 ? sample_from(dist, rng) : spec_pick(dist, drafts[dk], rng, acc);
                ++n[t], kept += acc;
            }
            double chi = 0;
            int df = -1, outside = 0;
            for (const auto& e : dist) {
                const double want = e.second * N, got = n.count(e.first) ? n[e.first] : 0;
                if (want >= 5) chi += (got - want) * (got - want) / want, ++df;
            }
            for (const auto& kv : n) {
                bool in = false;
                for (const auto& e : dist) in |= e.first == kv.first;
                if (!in) outside += kv.second;
            }
            const double p = chi_p(chi, df);
            const bool ok = p > 1e-4 && outside == 0;
            fails += !ok;
            std::printf("check sampling: temperature %.1f top_p %.2f top_k %d, %s: chi-square %.1f (%d df), p %.3g, "
                        "drafts kept %.1f%%%s - %s\n", sc.temp, sc.top_p, sc.top_k,
                        dk == 3 ? "plain pick" : dk == 0 ? "the likeliest draft" : dk == 1 ? "a less likely draft"
                                                                                    : "an impossible draft",
                        chi, df, p, 100.0 * kept / N, outside ? " (tokens outside the distribution)" : "",
                        ok ? "ok" : "FAIL");
        }
    }
    std::printf(fails ? "check sampling: FAILED (%d)\n" : "check sampling: the distribution is kept\n", fails);
    return fails ? 1 : 0;
}

void check_preview(int n_gpu, bool serve) {
    auto refuse = [&](const std::string& m) {   // (--serve: the server prints a NOTE line)
        if (serve) std::printf("NOTE maya-ik2: %s\n", m.c_str()), std::fflush(stdout);
        die(m);
    };
    const bool force = getenv("IK2_UNSUPPORTED") && std::atoi(getenv("IK2_UNSUPPORTED"));
    const std::string run_anyway = " (IK2_UNSUPPORTED=1 runs it anyway, untested)";
#ifdef _WIN32
    if (!force) refuse("the Inkling 2 preview runs on Linux, Windows is outside it" + run_anyway);
#endif
    int nd = 0;
    const cudaError_t e = cudaGetDeviceCount(&nd);
#ifdef STRATA_USE_HIP
    const char* vendor = "AMD";
#else
    const char* vendor = "NVIDIA";
#endif
    if (e != cudaSuccess || nd < 1)
        refuse(std::string("no ") + vendor + " GPU visible (" + (e != cudaSuccess ? cudaGetErrorString(e) : "0 devices") +
            "): the Inkling 2 preview runs on one " + vendor + " GPU");
    if (n_gpu < 1 || n_gpu > 2) refuse("--gpus " + std::to_string(n_gpu) + ": the Inkling 2 preview runs on one GPU");
    if (n_gpu == 2 && !force)
        refuse("--gpus 2: the Inkling 2 preview runs on one GPU, two (IK2_EP=1 or IK2_TP=1) are outside it" + run_anyway);
    for (int d = 0; d < std::min(n_gpu, nd); ++d) {
        cudaDeviceProp p{};
        CU(cudaGetDeviceProperties(&p, d));
#ifdef STRATA_USE_HIP
        // (AMD, Linux: every arch of this build - include/strata/hip_compat; the device check of the build's arch
        //  list stops a card it has no code for with a clear line, as the other engines)
        std::string arch = p.gcnArchName;
        arch = arch.substr(0, arch.find(':'));
        const std::string gpu = "GPU " + std::to_string(d) + " (" + p.name + ", " + arch + ")";
        if ((std::string(",") + STRATA_HIP_ARCHS + ",").find("," + arch + ",") == std::string::npos)
            refuse("this build has no kernels for " + gpu + ": it is built for " + STRATA_HIP_ARCHS +
                   " (add " + arch + " to CMAKE_HIP_ARCHITECTURES and rebuild)");
        if (arch.rfind("gfx11", 0) != 0 && arch.rfind("gfx12", 0) != 0 && !force)   // (the matrix-core prompt paths)
            refuse(gpu + ": maya-ik2 on AMD needs an RDNA 3 / 3.5 / 4 GPU (gfx11 / gfx12)" + run_anyway);
#else
        const std::string gpu = "GPU " + std::to_string(d) + " (" + p.name + ", sm_" + std::to_string(p.major) +
                                std::to_string(p.minor) + ")";
        if (p.major < 7) refuse(gpu + " is too old: maya-ik2 needs an NVIDIA GPU of sm_70 (Volta) or newer");
#endif
        CU(cudaSetDevice(d));
#ifndef STRATA_USE_HIP   // (a kernel of this build for the GPU: none when its arch is not in the list)
        cudaFuncAttributes fa{};
        if (cudaFuncGetAttributes(&fa, unpack_segments_kernel) != cudaSuccess) {
            (void) cudaGetLastError();
            refuse("this build has no kernels for " + gpu + ": add " + std::to_string(p.major) + std::to_string(p.minor) +
                " to CMAKE_CUDA_ARCHITECTURES and rebuild");
        }
#endif
        // (a card that does not hold every expert takes the offload path: a tier of them in VRAM, the rest in RAM;
        //  under 7 GiB - an 8 GB card's - the dense weights, the caches and the smallest tier do not fit beside)
#ifdef STRATA_USE_HIP   // (AMD: no offload path there yet - every expert in the GPU's memory, a 24 GB card's)
        if (p.totalGlobalMem < (22ull << 30) && !force) {
            char m[64];
            std::snprintf(m, sizeof m, "%.1f GB", (double) p.totalGlobalMem / 1e9);
            refuse(gpu + " has " + m + ", the Inkling 2 preview needs a 24 GB card" + run_anyway);
        }
#else
        if (p.totalGlobalMem < (7ull << 30) && !force) {
            char m[64];
            std::snprintf(m, sizeof m, "%.1f GB", (double) p.totalGlobalMem / 1e9);
            refuse(gpu + " has " + m + ", the Inkling 2 preview needs a card of 8 GB or more" + run_anyway);
        }
#endif
    }
    CU(cudaSetDevice(0));
}

int main(int argc, char** argv) {
    std::string model, toks, dump, draft_path, drafting;
    int gen = 16, chunk = 2048, ctx = 262144, threads = 8, n_gpu = 1;
    bool serve = false, serve_maya = true;   // (--serve: Maya's engine protocol; --serve-ik2: the @-line one)
    std::vector<int> eos_ids;                 // --eos a,b: the tokens a request ends at (Maya's protocol)
    double cache_gb = 60;
    int image_token = 200054;   // --image-token: Inkling 2's <|unused_200054|>, the template's placeholder of a picture
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value of " + a);
            return argv[++i];
        };
        if (a == "--check-sampling") return check_sampling();   // (no GPU, no model)
        if (a == "--model") model = next();
        else if (a == "--tokens") {   // (--tokens @FILE: the ids from a file - a long prompt passes a shell's limit)
            toks = next();
            if (!toks.empty() && toks[0] == '@') {
                std::ifstream in(toks.substr(1));
                toks.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
                while (!toks.empty() && std::isspace((unsigned char) toks.back())) toks.pop_back();
            }
        }
        else if (a == "--gen") gen = std::atoi(next().c_str());
        else if (a == "--chunk") chunk = std::atoi(next().c_str());
        else if (a == "--ctx" || a == "--max-context") ctx = std::atoi(next().c_str());   // (--max-context: serve/server.py's name)
        else if (a == "--cache-gb") cache_gb = std::atof(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--gpus") n_gpu = std::atoi(next().c_str());
        else if (a == "--draft") draft_path = next();
        else if (a == "--serve") serve = true;
        else if (a == "--serve-ik2") serve = true, serve_maya = false;
        else if (a == "--eos") {
            const std::string v = next();
            for (const char* p = v.c_str(); *p;) {
                eos_ids.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        }
        else if (a == "--image-token") image_token = std::atoi(next().c_str());
        else if (a == "--dump") dump = next();
        else if (a == "--drafting") {   // (auto: measured once on this PC; on: the drafter and the prompt lookup; off)
            drafting = next();
            if (drafting != "auto" && drafting != "on" && drafting != "off") die("--drafting: auto, on or off");
        }
        else die("unknown argument " + a);
    }
    if (drafting.empty()) drafting = draft_path.empty() ? "off" : "on";   // (--draft alone: as before, drafting on)
    if (drafting == "off") draft_path.clear();
    g_lookup = drafting != "off" && !(getenv("IK2_LOOKUP") && !std::atoi(getenv("IK2_LOOKUP")));
    g_prof = getenv("IK2_PROF") && std::atoi(getenv("IK2_PROF"));
    g_flash = !(getenv("IK2_FLASH") && !std::atoi(getenv("IK2_FLASH")));
#ifdef STRATA_USE_HIP
    if (!k::attn_flash_available()) g_flash = false;   // (AMD: the fused prompt attention where its kernel is built)
#endif
    g_mv = !getenv("IK2_MV") || std::atoi(getenv("IK2_MV"));
    g_group = !getenv("IK2_GROUP") || std::atoi(getenv("IK2_GROUP"));
    if (getenv("IK2_GROUP")) {
        const int v = std::atoi(getenv("IK2_GROUP"));
        g_group_exp = v == 1 || v >= 3, g_group_attn = v >= 2;
    }
    g_dense_q8 = getenv("IK2_DENSE") && std::string(getenv("IK2_DENSE")) == "q8";
    if (getenv("IK2_KV")) {
        const std::string kv = getenv("IK2_KV");
        if (kv == "f16" || kv == "fp16") g_kv_fp8 = false;
        else if (kv == "fp8" || kv == "e4m3") g_kv_fp8 = true;
        else die("unknown IK2_KV=" + kv + " (expected fp8 or f16)");
    }
    if (getenv("IK2_TRACE") && *getenv("IK2_TRACE")) {
        g_trace = std::fopen(getenv("IK2_TRACE"), "wb");
        if (!g_trace) die(std::string("cannot write ") + getenv("IK2_TRACE"));
    }
    if (model.empty() || (toks.empty() && !serve))
        die("usage: maya-ik2 --model FIRST.gguf (--tokens 1,2,3 [--gen N] | --serve) [--chunk 2048] [--ctx N] ...");
    std::vector<int> prompt;
    for (const char* p = toks.c_str(); *p;) {
        prompt.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    check_preview(n_gpu, serve);
#ifdef STRATA_USE_HIP
    {   // (AMD: the GEMMs' measured kernels - lt-picks.txt beside the GGUF, or IK2_LT_STORE's file; empty: none kept)
        const char* ls = std::getenv("IK2_LT_STORE");
        const size_t sl = model.find_last_of('/');
        const std::string dir = sl == std::string::npos ? std::string(".") : model.substr(0, sl);
        strata::ds4::lt::set_store(ls != nullptr ? ls : (dir + "/lt-picks.txt").c_str(), "IK2_LT_STORE");
    }
#endif
    // IK2_VRAM_GB=N (tests): the first GPU as a card of N GiB - the rest of its memory taken now and kept, so every
    // figure the engine reads (cudaMemGetInfo) is a smaller card's
    if (const char* vg = getenv("IK2_VRAM_GB"); vg && *vg) {
        size_t fr = 0, tot = 0;
        CU(cudaMemGetInfo(&fr, &tot));
        const size_t want = (size_t) (std::atof(vg) * 1073741824.0);
        if (want < tot) {
            void* ballast = nullptr;
            CU(cudaMalloc(&ballast, tot - want));
            std::printf("IK2_VRAM_GB=%s: GPU 0 as a card of %.1f GiB (%.1f GiB held back)\n", vg, want / 1073741824.0,
                        (tot - want) / 1073741824.0);
        }
    }
    const double t_start = now();
    Shards S(model);
    Model M;
    Config& c = M.c;
    const std::string a = "inkling2.";
    // a key under PR #32187's name, else under llama.cpp master's; a value master writes a visit (an array of equal
    // values) read as one
    auto key = [&](const char* k1, const char* k2 = nullptr) -> double {
        const auto* v = S.get(a + k1);
        if (!v && k2) v = S.get(a + k2);
        if (!v) die(std::string("no key ") + a + k1);
        return v->items.empty() ? v->num() : v->items[0].num();
    };
    c.n_visit = (int) key("block_count");
    c.C = (int) key("embedding_length");
    c.H = (int) key("attention.head_count");
    c.KV = (int) key("attention.head_count_kv");
    c.D = (int) key("attention.key_length");
    c.E = (int) key("expert_count");
    c.K = (int) key("expert_used_count");
    c.F = (int) key("expert_feed_forward_length");
    c.window = (int) key("attention.sliding_window");
    c.d_rel = (int) key("d_rel", "attention.relative_length");
    c.extent = (int) key("rel_extent", "attention.relative_extent");
    c.extent_swa = S.get(a + "rel_extent_swa") ? (int) key("rel_extent_swa") : c.window;   // (master: the window)
    c.block = (int) key("attn_res.block_size");
    c.n_vocab = (int) key("unpadded_vocab_size", "vocab_size_unpadded");
    c.eps = (float) key("attention.layer_norm_rms_epsilon");
    c.route_scale = (float) key("expert_weights_scale");
    c.qk_clip = (float) key("qk_clip_threshold", "attention.qk_clip");
    c.rel_clip = (float) key("rel_clip_threshold", "attention.relative_clip");
    c.alpha = (float) key("log_scaling_alpha", "attention.log_scale_alpha");
    c.n_floor = (float) key("log_scaling_n_floor", "attention.log_scale_floor");
    c.gate_cap = (float) key("mlp_gate_softcap", "activation.situ_beta");
    c.up_cap = (float) key("mlp_up_softcap", "activation.situ_linear_beta");
    c.logit_cap = (float) key("final_logit_softcapping");
    // the loop (Inkling-2-Mini): block_count counts the layer visits, the core's layers once a pass
    if (S.get(a + "num_loops")) c.n_pass = (int) key("num_loops");
    if (c.n_pass > 1) {
        c.core0 = (int) key("loop.core_start"), c.core1 = (int) key("loop.core_end"), c.rank = (int) key("loop.gate_rank");
        if (c.core0 <= 0 || c.core1 <= c.core0 || c.core0 % c.block || (c.core1 - c.core0) % c.block || c.rank <= 0)
            die("the looped core must start after the first block and span whole blocks");
    }
    const int n_core = c.core1 - c.core0;
    c.n_layer = c.n_visit - (c.n_pass - 1) * n_core;
    {   // the pattern is a visit's; a layer's is its first visit's (the same in every pass)
        const auto* sp = S.get(a + "attention.sliding_window_pattern");
        if (!sp || (int) sp->items.size() != c.n_visit) die("sliding_window_pattern");
        for (int l = 0; l < c.n_layer; ++l)
            c.sliding.push_back(sp->items[l < c.core1 ? l : l + (c.n_pass - 1) * n_core].u != 0);
    }
    // (a row of an expert is whole blocks of its type: checked per tensor below; 64 covers NVFP4 and Q8_0)
    if (c.D != 128 || c.H % c.KV || c.C % 64 || c.F % 64) die("unexpected shapes");
    const int n_sliding = (int) std::count(c.sliding.begin(), c.sliding.end(), true);
    std::printf("maya-ik2: %d layers (%d sliding), %d experts (%d a token), width %d, vocab %d; %zu shards\n",
                c.n_layer, n_sliding, c.E, c.K, c.C, c.n_vocab, S.files.size());
    if (c.n_pass > 1) {
        std::printf("looped: layers %d .. %d run %d times - %d layer visits a token; message gates of rank %d\n", c.core0,
                    c.core1 - 1, c.n_pass, c.n_visit, c.rank);
        // if (n_gpu != 1) die("a looped model: one GPU so far (--gpus 1)");
        if (!draft_path.empty() && (c.C != 2304 || c.n_visit != 48)) die("--draft: a looped model other than Inkling-2-Mini");
    }
    if (getenv("IK2_POP_OUT")) g_pop.assign((size_t) c.n_layer * c.E, 0), g_pop_E = c.E;
    g_tp_F = c.F, g_tp_C = c.C;
    if (n_gpu == 2 && getenv("IK2_EP") && std::atoi(getenv("IK2_EP"))) {
        const double share = getenv("IK2_EP_SPLIT") ? std::atof(getenv("IK2_EP_SPLIT")) : 0.5;
        g_pop_E_ep = c.E;
        g_ep_own.assign((size_t) c.n_layer * c.E, 0);
        std::vector<uint64_t> cnt;
        if (const char* wf = getenv("IK2_WARM"); wf && *wf) {
            FILE* in = std::fopen(wf, "rb");
            int32_t hdr[2] = {0, 0};
            if (in && std::fread(hdr, 4, 2, in) == 2 && hdr[0] == c.n_layer && hdr[1] == c.E) {
                cnt.resize((size_t) c.n_layer * c.E);
                if (std::fread(cnt.data(), 8, cnt.size(), in) != cnt.size()) cnt.clear();
            }
            if (in) std::fclose(in);
        }
        for (int l = 0; l < c.n_layer; ++l) {
            if (cnt.empty()) {
                for (int e = 0; e < c.E; ++e)
                    g_ep_own[(size_t) l * c.E + e] = ((uint32_t) (e * 2654435761u + l * 40503u) >> 8) % 10000 < share * 10000;
                continue;
            }
            std::vector<int> ord(c.E);
            for (int e = 0; e < c.E; ++e) ord[e] = e;
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return cnt[(size_t) l * c.E + a] > cnt[(size_t) l * c.E + b]; });
            double got[2] = {0, 0};
            const double want[2] = {1.0 - share, share};
            for (int e : ord) {   // (the GPU furthest below its share takes the next most picked)
                const double v = (double) cnt[(size_t) l * c.E + e] + 1.0;
                const int d = got[0] / want[0] <= got[1] / want[1] ? 0 : 1;
                g_ep_own[(size_t) l * c.E + e] = (uint8_t) d;
                got[d] += v;
            }
        }
    }

    {
        int nd = 0;
        CU(cudaGetDeviceCount(&nd));
        if (n_gpu < 1 || n_gpu > nd) die("--gpus " + std::to_string(n_gpu) + ": " + std::to_string(nd) + " visible");
    }
    g_dev.resize(n_gpu);
    for (int d = 0; d < n_gpu; ++d) {
        CU(cudaSetDevice(d));
        CU(cudaStreamCreate(&g_dev[d].s));
        CU(cudaStreamCreate(&g_dev[d].cs));
        CU(cudaEventCreateWithFlags(&g_dev[d].ev, cudaEventDisableTiming));
        CB(cublasCreate(&g_dev[d].blas));
        CB(cublasSetStream(g_dev[d].blas, g_dev[d].s));
        CB(cublasSetMathMode(g_dev[d].blas, CUBLAS_DEFAULT_MATH));
    }
    g_ep = n_gpu == 2 && getenv("IK2_EP") && std::atoi(getenv("IK2_EP"));
    g_tp = n_gpu == 2 && !g_ep && getenv("IK2_TP") && std::atoi(getenv("IK2_TP"));
    if (dual()) {
        int p01 = 0, p10 = 0;
        CU(cudaDeviceCanAccessPeer(&p01, 0, 1));
        CU(cudaDeviceCanAccessPeer(&p10, 1, 0));
        if (p01 && p10) {
            CU(cudaSetDevice(0));
            CU(cudaDeviceEnablePeerAccess(1, 0));
            CU(cudaSetDevice(1));
            CU(cudaDeviceEnablePeerAccess(0, 0));
        }
        CU(cudaSetDevice(0));
        CU(cudaEventCreateWithFlags(&g_ep_x, cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&g_ep_cp[0], cudaEventDisableTiming));
        CU(cudaSetDevice(1));
        CU(cudaEventCreateWithFlags(&g_ep_cp[1], cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&g_ep_done, cudaEventDisableTiming));
        CU(cudaSetDevice(0));
        CU(cudaEventCreateWithFlags(&g_tp_h[0], cudaEventDisableTiming));
        CU(cudaSetDevice(1));
        CU(cudaEventCreateWithFlags(&g_tp_h[1], cudaEventDisableTiming));
        std::printf("%s over 2 GPUs (peer access %s)\n", g_tp ? "experts tensor-parallel" : "expert parallel",
                    p01 && p10 ? "on" : "off: copies through the host");
    }
    use_dev(0);
    g_overlap = !getenv("IK2_OVERLAP") || std::atoi(getenv("IK2_OVERLAP"));
    for (auto& e : g_ring) CU(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    // (off by default: on an RTX PRO 6000 node's Xeon 6737P an NVFP4 expert takes 0.73 ms on 7 threads against 0.33 ms
    //  over PCIe, and the lane measured 16.4 against 16.8 tok/s - IK2_CPU=1 to try it)
    if (getenv("IK2_CPU") && std::atoi(getenv("IK2_CPU")) && ik2::cpu::available() && threads > 1 && n_gpu == 1)
        g_pool = new ik2::cpu::Pool(threads - 1);
    if (getenv("IK2_UP_MS")) g_t_up = std::atof(getenv("IK2_UP_MS")) * 1e-3;
    g_cpu_lazy = !getenv("IK2_CPU_LAZY") || std::atoi(getenv("IK2_CPU_LAZY"));
    if (n_gpu > 1 && getenv("IK2_DENSE")) die("IK2_DENSE with --gpus > 1");

    // --drafting auto: this PC's choice when it was measured before (then only that drafter takes room), else every
    // source that fits, measured before READY (serve)
    std::string draft_key_s;
    int draft_choice = -1;
    if (drafting == "auto" && (!draft_path.empty() || g_lookup)) {
        draft_key_s = draft_key(model, draft_path, ctx);
        double tps[4] = {-1, -1, -1, -1};
        std::string date;
        draft_choice = draft_remembered(draft_key_s, tps, date);
        if (draft_choice >= 0) {
            if (draft_choice < 2) draft_path.clear();
            g_lookup = g_lookup && (draft_choice == 1 || draft_choice == 3);
            std::printf("drafting: %s - the fastest on this PC (measured %s: no drafting %.1f, the prompt lookup %.1f, "
                        "the drafter %.1f, both %.1f tokens/s)\n", kDraftNames[draft_choice], date.c_str(), tps[0],
                        tps[1], tps[2], tps[3]);
        }
    }

    // Do the experts all fit in VRAM beside the dense weights and the K / V cache (Inkling-2-Mini's 14.7 GiB on a 24 GB
    // card)?  Then they all stay there: the dense weights are kept as the GGUF's blocks unless IK2_DENSE says otherwise
    // (half an FP16 copy's bytes, a token's products straight over them) and tokens take the resident path
    // (IK2_FAST=0: off; IK2_GRAPH=0: its launches one by one instead of a CUDA graph a token).
    g_graph = !(getenv("IK2_GRAPH") && !std::atoi(getenv("IK2_GRAPH")));
    g_block = !(getenv("IK2_BLOCK") && !std::atoi(getenv("IK2_BLOCK")));
    g_block_exact = !(getenv("IK2_BLOCK_EXACT") && !std::atoi(getenv("IK2_BLOCK_EXACT")));
    g_window_check = getenv("IK2_WINDOW_CHECK") ? std::atoi(getenv("IK2_WINDOW_CHECK")) : 0;
    if (g_block_exact && !k::experts_block_exact()) {   // (IK2_EXPERT_MMA_SPLIT* not the defaults: no exact window)
        if (!draft_path.empty())
            std::printf("INFO maya-ik2: drafting off - IK2_EXPERT_MMA_SPLIT / _DOWN are not 8 / 4, and the verify "
                        "window would then change the answers\n");
        g_block = false;
    }
    g_attn_mma = c.D == 128 && c.H / c.KV <= 8 ? (getenv("IK2_ATTN_MMA") ? std::atoi(getenv("IK2_ATTN_MMA")) : 1) : 0;
    g_mma_split = k::attn_decode_mma_blocks(c.KV);
    // Older NVIDIA (docs/INKLING2.md 8): Volta (sm_70) and Turing (sm_75) take their own attention and expert kernels
    // (k::arch_class); a prompt's attention goes through cuBLAS there (the fused kernel is Ampere's: cp.async, and more
    // shared memory than a Turing SM has), and Volta has no verify window (the block path's kernels are mma.sync's).
    // Drafting needs a verify window with the token path's own arithmetic: proven on Ampere on and AMD only, so on
    // Volta and Turing (their own token kernels, no window twins yet) it stays off - the answers stay plain decoding's.
    {
        int cls = 80;
        for (int d = 0; d < n_gpu; ++d) use_dev(d), cls = std::min(cls, k::arch_class());
        use_dev(0);
        if (cls < 80) g_flash = false;
        if (cls < 75) g_block = false;
        if (cls < 80 && (!draft_path.empty() || g_lookup)) {
            std::printf("NOTE maya-ik2: drafting off - on this GPU (sm_%d class) the verify window does not have the "
                        "one-token path's kernels yet, so it would change the answers; plain decoding\n", cls);
            std::fflush(stdout);
            draft_path.clear(), g_lookup = false;
        }
    }
    if (n_gpu == 1 && !g_pool) {
        size_t exp_b = 0, dense_b = 0, kv_b = 0;
        for (const auto& kv : S.by_name) {
            const auto& t = *kv.second.second;
            if (!k::block_values((int) t.type)) continue;
            if (t.name.find("_exps.weight") != std::string::npos) exp_b += tensor_bytes(t);
            else if (t.name != "token_embd.weight") dense_b += tensor_bytes(t);
        }
        for (int l = 0; l < c.n_layer; ++l) {
            const size_t cap = c.sliding[l] ? (size_t) std::min(ctx, c.window + std::max(chunk, ik2::draft::kBlock)) : (size_t) ctx;
            kv_b += (l >= c.core0 && l < c.core1 ? c.n_pass : 1) * cap * c.KV * c.D * (g_kv_fp8 ? 1 : 2) * 2;
        }
        size_t fr0 = 0, tot0 = 0;
        CU(cudaMemGetInfo(&fr0, &tot0));
#ifdef STRATA_USE_HIP
        {   // (an AMD APU whose memory is the RAM - Strix / Gorgon Halo on Linux: HIP's free figure is the GTT's, which
            //  can exceed what the RAM has left; the engines keep max(8 GiB, 10% of the RAM) free for the system)
            cudaDeviceProp p{};
            CU(cudaGetDeviceProperties(&p, 0));
            FILE* mi = p.integrated ? std::fopen("/proc/meminfo", "r") : nullptr;
            if (mi) {
                char line[256];
                unsigned long long tot_kb = 0, av_kb = 0;
                while (std::fgets(line, sizeof line, mi)) {
                    std::sscanf(line, "MemTotal: %llu kB", &tot_kb);
                    std::sscanf(line, "MemAvailable: %llu kB", &av_kb);
                }
                std::fclose(mi);
                const size_t margin = std::max<size_t>((size_t) 8 << 30, (size_t) (tot_kb * 1024 / 10));
                const size_t room = av_kb * 1024 > margin ? (size_t) (av_kb * 1024 - margin) : 0;
                if (av_kb && room < fr0) fr0 = room;
            }
        }
#endif
        const size_t reserve = (size_t) 3 << 29;   // (the work buffers and a quarter of a GiB of room: 1.5 GiB)
        const bool fits = exp_b + dense_b + kv_b + reserve <= fr0 && exp_b <= (size_t) (cache_gb * 1073741824.0);
        // Where they do not all fit (a card under 24 GB, or a context past what one holds), the offload token path:
        // the resident path's launches with a tier of the experts in VRAM and the others fetched by the GPU from
        // host memory (ik2_off.cuh).  IK2_OFF=0: the cache path instead (forward() a visit at a time); IK2_OFF=1:
        // the offload path even where every expert fits (tests).  Not with two GPUs or the CPU lane (n_gpu == 1,
        // !g_pool here).
        const char* off_env = getenv("IK2_OFF");
        const int off_set = off_env && *off_env ? std::atoi(off_env) : -1;
        if (fits && !getenv("IK2_DENSE")) g_dense_q8 = true;
        g_fast_want = fits && off_set != 1 && g_dense_q8 && !(getenv("IK2_FAST") && !std::atoi(getenv("IK2_FAST")));
        g_off_want = (off_set == 1 || (off_set != 0 && !fits)) && !getenv("IK2_DENSE");
#ifdef STRATA_USE_HIP   // (AMD: the offload path is not checked there yet - the cache path where they do not fit)
        if (off_set != 1) g_off_want = false;
#endif
        if (g_off_want) g_dense_q8 = true;
        std::printf("experts %.1f GiB, dense weights %.1f GiB as blocks, K / V cache %.1f GiB, %.1f GiB free: %s\n",
                    exp_b / 1073741824.0, dense_b / 1073741824.0, kv_b / 1073741824.0, fr0 / 1073741824.0,
                    fits && !g_off_want ? "every expert stays in VRAM"
                    : g_off_want ? "the experts in host memory, a tier of them in VRAM, the rest fetched by the GPU"
                                 : "the experts in host memory, a cache of them in VRAM");
        // Drafting never changes this plan (a loaded drafter takes room only where everything above stays as it is):
        // it runs on the resident path, its room - the drafter's, or the prompt lookup's window records alone - beside
        // the model at this context, and not the room the pictures' encoder would have had on the GPU (IK2_VISION_GIB,
        // the server's: the encoder starts on the GPU when the free VRAM holds it).  Else it is left out, said why.
        const size_t used = exp_b + dense_b + kv_b + reserve;
        const double vis = getenv("IK2_VISION_GIB") ? std::atof(getenv("IK2_VISION_GIB")) * 1073741824.0 : 0.0;
        const double left = fr0 > used ? (double) (fr0 - used) : 0.0;
        auto room_ok = [&](size_t need, std::string& why) {
            if (!g_fast_want) why = "it runs where every expert stays in VRAM (the resident path), not here";
            else if (used + need > fr0) why = "its room does not fit beside the model at this context";
            else if (vis > 0 && left >= vis && left - (double) need < vis)
                why = "its room would move the pictures' encoder from the GPU to the CPU";
            return why.empty();
        };
        for (int pass = 0; pass < 2; ++pass) {   // (the drafter, else the prompt lookup alone)
            const bool model_drafter = pass == 0;
            if (model_drafter ? draft_path.empty() : !g_lookup) continue;
            const auto* head = S.info("output.weight");
            const int head_rows = head ? (int) (head->elements() / c.C) : 201024;
            const size_t need = draft_room(model_drafter ? draft_path : std::string(), ctx, chunk, c, head_rows);
            std::string why;
            if (room_ok(need, why)) {
                std::printf("drafting: %s, %.2f GiB of VRAM beside the model (the plan as without it)\n",
                            model_drafter ? (g_lookup ? "the drafter and the prompt lookup" : "the drafter")
                                          : "the prompt lookup", need / 1073741824.0);
                break;
            }
            std::printf("NOTE maya-ik2: %s left out (%.2f GiB): %s\n", model_drafter ? "the drafter" : "the prompt lookup",
                        need / 1073741824.0, why.c_str());
            if (model_drafter) draft_path.clear();
            else g_lookup = false;
        }
        std::fflush(stdout);
    } else if (!draft_path.empty() || g_lookup) {
        std::printf("NOTE maya-ik2: drafting off - it runs on one GPU with every expert in VRAM, not here\n");
        draft_path.clear(), g_lookup = false;
    }

    // the dense weights
    const double t_dense = now();
    const size_t C = c.C, HD = (size_t) c.H * c.D, KD = (size_t) c.KV * c.D;
    M.L.resize(c.n_layer);
    // two layouts: llama.cpp PR #32187's (ref/inkling2.py's: the short convs folded, gate and up apart) and unsloth's
    // "GGUF_shard_rewrite" one that llama.cpp's master reads (names of its own, the convs' residual and scale outside
    // their weights, gate and up fused - gate's rows first - with one scale): the second's folded here as it loads
    const bool rw = S.info("blk.0.ffn_gate_up_exps.weight") != nullptr;
    auto nm = [&](const std::string& b, const char* old_name) -> std::string {
        static const std::map<std::string, std::string> m = {
            {"attn_r.weight", "attn_rel.weight"}, {"attn_r_clip.weight", "attn_rel_clip.weight"},
            {"attn_rel_proj.weight", "attn_rel_bias.weight"}, {"attn_res_score.weight", "attn_res_query.weight"},
            {"ffn_res_score.weight", "ffn_res_query.weight"}, {"output_res_score.weight", "output_res_query.weight"},
            {"attn_q_in_norm.weight", "attn_q_inp_norm.weight"}, {"attn_k_in_norm.weight", "attn_k_inp_norm.weight"},
            {"attn_v_in_norm.weight", "attn_v_inp_norm.weight"}, {"shortconv_k.weight", "attn_k_conv.weight"},
            {"shortconv_v.weight", "attn_v_conv.weight"}, {"shortconv_attn.weight", "attn_post_conv.weight"},
            {"shortconv_mlp.weight", "ffn_post_conv.weight"}, {"shortconv_k_dyn.weight", "attn_k_conv_proj.weight"},
            {"shortconv_v_dyn.weight", "attn_v_conv_proj.weight"}};
        auto it = m.find(old_name);
        return b + (rw && it != m.end() ? it->second : std::string(old_name));
    };
    // a conv's weights [C'][4] with the residual (tap 3: the current token) and the output scale folded in
    auto conv_w = [&](const std::string& b, const char* n, size_t Cc, bool dynamic) -> float* {
        auto w = host_f32(S, nm(b, n), Cc * 4);
        if (rw) {
            const float sc = 1.0f / std::sqrt((float) ((dynamic ? 2 : 1) * 4 + 1));
            for (size_t i = 0; i < Cc; ++i) {
                for (int j = 0; j < 4; ++j) w[i * 4 + j] *= sc;
                w[i * 4 + 3] += sc;
            }
        }
        return dev_f32(w);
    };
    auto dyn_host = [&](const std::string& b, const char* n) -> std::vector<float> {
        auto w = host_f32(S, nm(b, n), (size_t) c.KV * 4 * C);
        if (rw)
            for (auto& x : w) x *= 1.0f / std::sqrt(9.0f);
        return w;
    };
    std::printf("layout: %s\n", rw ? "unsloth's GGUF_shard_rewrite (llama.cpp master)" : "llama.cpp PR #32187");
    struct ExpJob {
        int l;
        std::string name;   // a tensor read whole into pinned memory
        std::vector<int> roles;
    };
    std::vector<ExpJob> exp_jobs;
    size_t host_total = 0;
    M.embd_norm = load_f32(S, "token_embd_norm.weight", C);   // (on GPU 0, where the embedding rows go)
    std::vector<float> lp_q, lp_kv;   // a looped core's message queries, [2 n_core][C] each
    std::vector<float> lp_all;
    if (c.n_pass > 1) {
        lp_q = host_f32(S, "loop_res_query.weight", (size_t) 2 * n_core * C);
        lp_kv = host_f32(S, "loop_kv_res_query.weight", (size_t) 2 * n_core * C);
    }
    for (int l = 0; l < c.n_layer; ++l) {
        Layer& Ly = M.L[l];
        const std::string b = "blk." + std::to_string(l) + ".";
        Ly.dev = dual() ? 0 : l * n_gpu / c.n_layer;
        use_dev(Ly.dev);
        Ly.sliding = c.sliding[l];
        auto sink_v = host_f32(S, b + "attn_sink_proj.weight", (size_t) c.H * C);
        if (Ly.sliding) {
            std::vector<MatPart> qkv_parts = {
                {b + "attn_q.weight", (int) HD, {}},
                {b + "attn_k.weight", (int) KD, {}},
                {b + "attn_v.weight", (int) KD, {}},
                {nm(b, "attn_r.weight"), c.H * c.d_rel, {}},
                {b + "attn_gate.weight", (int) HD, {}},
            };
            if (!g_dense_q8) {
                qkv_parts.push_back({"", c.H, std::move(sink_v)});
                Ly.wqkvrg = load_fused_f16(S, qkv_parts, (int) C);
            } else {
                if (!try_load_fused_raw(S, qkv_parts, (int) C, Ly.wqkvrg)) {
                    Ly.wq = load_mat(S, b + "attn_q.weight", (int) HD, (int) C);
                    Ly.wk = load_mat(S, b + "attn_k.weight", (int) KD, (int) C);
                    Ly.wv = load_mat(S, b + "attn_v.weight", (int) KD, (int) C);
                    Ly.wr = load_mat(S, nm(b, "attn_r.weight"), c.H * c.d_rel, (int) C);
                    Ly.wgate = load_mat(S, b + "attn_gate.weight", (int) HD, (int) C);
                }
                Ly.sink16 = dev_f16(sink_v);
            }
        } else {
            std::vector<MatPart> qrg_parts = {
                {b + "attn_q.weight", (int) HD, {}},
                {nm(b, "attn_r.weight"), c.H * c.d_rel, {}},
                {b + "attn_gate.weight", (int) HD, {}},
            };
            auto kd_v = dyn_host(b, "shortconv_k_dyn.weight");
            auto vd_v = dyn_host(b, "shortconv_v_dyn.weight");
            if (!g_dense_q8) {
                qrg_parts.push_back({"", c.H, std::move(sink_v)});
                Ly.wqrg = load_fused_f16(S, qrg_parts, (int) C);
                Ly.wk_dyn = load_fused_f16(S, {{b + "attn_k.weight", (int) KD, {}}, {"", c.KV * 4, std::move(kd_v)}},
                                           (int) C);
                Ly.wv_dyn = load_fused_f16(S, {{b + "attn_v.weight", (int) KD, {}}, {"", c.KV * 4, std::move(vd_v)}},
                                           (int) C);
            } else {
                if (!try_load_fused_raw(S, qrg_parts, (int) C, Ly.wqrg)) {
                    Ly.wq = load_mat(S, b + "attn_q.weight", (int) HD, (int) C);
                    Ly.wr = load_mat(S, nm(b, "attn_r.weight"), c.H * c.d_rel, (int) C);
                    Ly.wgate = load_mat(S, b + "attn_gate.weight", (int) HD, (int) C);
                }
                Ly.wk = load_mat(S, b + "attn_k.weight", (int) KD, (int) C);
                Ly.wv = load_mat(S, b + "attn_v.weight", (int) KD, (int) C);
                Ly.sink16 = dev_f16(sink_v);
                Ly.sc_k_dyn16 = dev_f16(kd_v);
                Ly.sc_v_dyn16 = dev_f16(vd_v);
            }
        }
        Ly.wo = load_mat(S, b + "attn_output.weight", (int) C, (int) HD);
        Ly.router16 = load_f16(S, b + "ffn_gate_inp.weight", (size_t) c.E * C);
        Ly.probs_b = load_f32(S, b + "exp_probs_b.bias", c.E);
        Ly.ffn_norm = load_f32(S, b + "ffn_norm.weight", C);
        if (Ly.sliding) {
            Ly.attn_norm = load_f32(S, b + "attn_norm.weight", C);
        } else {
            Ly.q_in_norm = load_f32(S, nm(b, "attn_q_in_norm.weight"), C);
            Ly.k_in_norm = load_f32(S, nm(b, "attn_k_in_norm.weight"), C);
            Ly.v_in_norm = load_f32(S, nm(b, "attn_v_in_norm.weight"), C);
        }
        Ly.q_clip = load_f32(S, b + "attn_q_clip.weight", HD);
        Ly.r_clip = load_f32(S, nm(b, "attn_r_clip.weight"), (size_t) c.H * c.d_rel);
        {   // [extent][d_rel]: PR #32187's attn_rel_proj is [d_rel][extent], the rewrite's attn_rel_bias already this
            const int ext = Ly.sliding ? c.extent_swa : c.extent;
            const auto v = host_f32(S, nm(b, "attn_rel_proj.weight"), (size_t) ext * c.d_rel);
            std::vector<float> t(v.size());
            if (rw) {
                t = v;
            } else {
                for (int r = 0; r < c.d_rel; ++r)
                    for (int e = 0; e < ext; ++e) t[(size_t) e * c.d_rel + r] = v[(size_t) r * ext + e];
            }
            Ly.rel_proj = dev_f32(t);
        }
        if (l > 0) {
            const std::string rn = nm(b, "attn_res_score.weight");
            const auto* ti = S.info(rn);
            if (!ti) die(rn);
            Ly.n_res_q = (int) (ti->elements() / C);
            Ly.res_score = load_f32(S, rn, (size_t) Ly.n_res_q * C);
        }
        Ly.ffn_res_score = load_f32(S, nm(b, "ffn_res_score.weight"), C);
        Ly.sc_k = conv_w(b, "shortconv_k.weight", KD, !Ly.sliding);
        Ly.sc_v = conv_w(b, "shortconv_v.weight", KD, !Ly.sliding);
        Ly.sc_attn = conv_w(b, "shortconv_attn.weight", C, false);
        Ly.sc_mlp = conv_w(b, "shortconv_mlp.weight", C, false);
        // the experts: (role tensor, its roles); a fused gate_up is gate's F rows then up's, an expert's 2 F rows
        std::vector<std::pair<std::string, std::vector<int>>> parts;
        if (rw) parts = {{"gate_up", {0, 1}}, {"down", {2}}};
        else parts = {{"gate", {0}}, {"up", {1}}, {"down", {2}}};
        for (const auto& pr : parts) {
            const std::string n = b + "ffn_" + pr.first + "_exps.weight";
            const auto* ti = S.info(n);
            if (!ti) die("no " + n);
            const int nr = (int) pr.second.size();
            if (ti->elements() != (uint64_t) c.E * c.F * C * nr) die(n + ": shape");
            if (!k::block_values((int) ti->type)) die(n + ": type " + ti->type_name() + " not decoded here");
            if (C % k::block_values((int) ti->type) || c.F % k::block_values((int) ti->type))
                die(n + ": a row is not whole " + ti->type_name() + " blocks");
            const size_t per = tensor_bytes(*ti) / c.E;
            std::vector<float> sc;
            const std::string sn = b + "ffn_" + pr.first + "_exps.scale";
            if (S.info(sn)) sc = host_f32(S, sn, c.E);
            else if (ti->type == k::kNVFP4) die(n + ": NVFP4 without its .scale");
            for (int i = 0; i < nr; ++i) {
                const int r = pr.second[i];
                Ly.ty[r] = (int) ti->type;
                Ly.eb[r] = per / nr;
                Ly.stride[r] = per;
                Ly.scale[r] = sc;
            }
            host_total += tensor_bytes(*ti);
            exp_jobs.push_back({l, n, pr.second});
        }
        if (c.n_pass > 1 && l >= c.core0 && l < c.core1) {   // the core: its later passes' message gates and queries
            static const char* mod[4] = {"attn", "k", "v", "mlp"};
            for (int m = 0; m < 4; ++m) {
                const std::string p = b + "loop_" + mod[m] + "_";
                Ly.lp_down[m] = load_f32(S, p + "down.weight", (size_t) c.rank * 2 * C);
                for (int fg = 0; fg < 2; ++fg) {   // (the file's [C][rank] by rank, [rank][C]: k::loop_gate's layout)
                    const auto w = host_f32(S, p + (fg ? "gain.weight" : "forget.weight"), C * c.rank);
                    std::vector<float> t(w.size());
                    for (size_t ch = 0; ch < C; ++ch)
                        for (int r = 0; r < c.rank; ++r) t[(size_t) r * C + ch] = w[ch * c.rank + r];
                    (fg ? Ly.lp_gain[m] : Ly.lp_forget[m]) = dev_f32(t);
                }
            }
            // loop_res_query: rows 2 i (q's) and 2 i + 1 (the MoE's) of core layer i; loop_kv_res_query: k's and v's
            const size_t i = (size_t) (l - c.core0);
            std::vector<float> q(4 * C);
            std::memcpy(q.data(), lp_q.data() + 2 * i * C, C * 4);
            std::memcpy(q.data() + C, lp_kv.data() + 2 * i * C, C * 4);
            std::memcpy(q.data() + 2 * C, lp_kv.data() + (2 * i + 1) * C, C * 4);
            std::memcpy(q.data() + 3 * C, lp_q.data() + (2 * i + 1) * C, C * 4);
            Ly.lp_query = dev_f32(q);
            lp_all.insert(lp_all.end(), q.begin(), q.end());
            if (l == c.core1 - 1) M.lp_query_all = dev_f32(lp_all);
        }
    }
    // the visits of a token and their state (FP8 E4M3 or FP16 KV; sliding layers ring-buffered in both CLI and --serve)
    M.V.resize(c.n_visit);
    for (int v = 0; v < c.n_visit; ++v) {
        Visit& Vi = M.V[v];
        if (c.n_pass == 1 || v < c.core0) Vi.l = v;
        else if (v < c.core0 + c.n_pass * n_core) Vi.l = c.core0 + (v - c.core0) % n_core, Vi.pass = (v - c.core0) / n_core;
        else Vi.l = v - (c.n_pass - 1) * n_core;
        const Layer& Ly = M.L[Vi.l];
        use_dev(Ly.dev);
        Vi.kv_cap = Ly.sliding ? std::min(ctx, c.window + std::max(chunk, ik2::draft::kBlock)) : ctx;
        if (g_kv_fp8) {
            Vi.kc8 = dalloc<uint8_t>((size_t) Vi.kv_cap * KD);
            Vi.vc8 = dalloc<uint8_t>((size_t) Vi.kv_cap * KD);
        } else {
            Vi.kc = dalloc<uint16_t>((size_t) Vi.kv_cap * KD);
            Vi.vc = dalloc<uint16_t>((size_t) Vi.kv_cap * KD);
        }
        Vi.h_k = dalloc<float>(3 * KD);
        Vi.h_v = dalloc<float>(3 * KD);
        Vi.h_attn = dalloc<float>(3 * C);
        Vi.h_mlp = dalloc<float>(3 * C);
    }
    c.n_rows = (int) (S.info("output.weight")->elements() / C);
    M.out_w = load_mat(S, "output.weight", c.n_rows, (int) C);
    if (g_dense_q8) {   // (a layer's largest; the head's logits go a token at a time over its blocks)
        g_w16_n = (2 * HD + 2 * KD + (size_t) c.H * c.d_rel) * C;
        g_w16 = dalloc<uint16_t>(g_w16_n);
    }
    M.out_norm = load_f32(S, "output_norm.weight", C);
    M.out_res_score = load_f32(S, rw ? "output_res_query.weight" : "output_res_score.weight", C);
    M.embd = S.data("token_embd.weight");
    M.embd_type = (int) S.info("token_embd.weight")->type;
    if (!k::block_values(M.embd_type)) die("token_embd's type");
    if (g_stage) cudaFree(g_stage), g_stage = nullptr, g_stage_bytes = 0;
    {
        size_t kv_bytes = 0;
        for (const auto& Vi : M.V) kv_bytes += (size_t) Vi.kv_cap * KD * (g_kv_fp8 ? 1 : 2) * 2;
        std::printf("kv cache: %s (%.2f GiB across %d layer%s)\n", g_kv_fp8 ? "FP8 E4M3" : "FP16",
                    kv_bytes / 1073741824.0, c.n_visit, c.n_pass > 1 ? " visits" : "s");
        std::fflush(stdout);
    }
    size_t fr = 0, tot = 0;
    for (int d = 0; d < n_gpu; ++d) {
        use_dev(d);
        CU(cudaMemGetInfo(&fr, &tot));
        std::printf("dense weights in VRAM: %.1f s; GPU %d: %.1f GiB of %.1f free\n", now() - t_dense, d,
                    fr / 1073741824.0, tot / 1073741824.0);
        std::fflush(stdout);
    }

    // the experts into pinned host memory: the allocations, then the reads, several files at once.  IK2_HOST=mmap:
    // read where the files are mapped instead (pageable: slower copies, no 323 GiB of host memory - for several
    // GPUs whose caches hold most of the experts)
    const double t_host = now();
    Pinned pin;
    double t_alloc = 0;
    // (the resident path keeps every expert in VRAM, so no pinned host copy is needed: the experts go from where the
    //  files are mapped straight to their slots, and stay in the page cache for the next load - IK2_HOST=pinned: as
    //  for the other models)
    const bool host_mmap = getenv("IK2_HOST") ? std::string(getenv("IK2_HOST")) == "mmap" : g_fast_want;
    // The offload path pins only the experts the VRAM tier does not hold (owner, v1.1.73): the layers' blocks only
    // reserved here (no RAM), the tier planned on the real free VRAM (EC.init), copied in from where the files are
    // mapped, then the rest read and pinned (below the warm) - an 8 GB card's tier is RAM a 16 GB PC does not need.
    // Linux, where the GPU reads a registered range at its own address; IK2_PIN_ALL=1: every expert pinned here.
    bool pin_later = false;
#ifndef _WIN32
    if (!host_mmap && g_off_want && !(getenv("IK2_PIN_ALL") && std::atoi(getenv("IK2_PIN_ALL"))) &&
        !(getenv("IK2_TIER_MIN") && std::atoi(getenv("IK2_TIER_MIN"))) &&
        !(getenv("IK2_ALLOC") && std::string(getenv("IK2_ALLOC")) == "hostalloc")) {
        int dv = 0, same = 0;
        CU(cudaGetDevice(&dv));
        CU(cudaDeviceGetAttribute(&same, cudaDevAttrCanUseHostPointerForRegisteredMem, dv));
        pin_later = same && host_register_works();
    }
#endif
    if (host_mmap) {
        if (g_pool) die("IK2_HOST=mmap with the CPU lane");
        if (g_off_want) die("IK2_HOST=mmap with the offload path (it reads pinned host memory: IK2_OFF=0)");
        for (const auto& J : exp_jobs) {
            Layer& Ly = M.L[J.l];
            const uint8_t* p = S.data(J.name);
            for (size_t i = 0; i < J.roles.size(); ++i) Ly.host[J.roles[i]] = p + i * Ly.eb[J.roles[i]];
            {   // (read ahead: a cold file otherwise comes in a page fault at a time)
                const size_t pg = 4096, a0 = (size_t) p & ~(pg - 1);
                ::madvise((void*) a0, (size_t) p - a0 + tensor_bytes(*S.info(J.name)), MADV_WILLNEED);
            }
        }
        std::printf("experts: read where the files are mapped (IK2_HOST=mmap)\n");
        std::fflush(stdout);
#ifndef _WIN32
    } else if (pin_later) {
        pin.blocks.resize(c.n_layer);
        for (int l = 0; l < c.n_layer; ++l) {
            Layer& Ly = M.L[l];
            const size_t slot_bytes = Ly.eb[0] + Ly.eb[1] + Ly.eb[2], layer_bytes = (size_t) c.E * slot_bytes;
            void* p = ::mmap(nullptr, layer_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) die("the offload path: no address space for layer " + std::to_string(l) + "'s experts");
            pin.blocks[l] = {p, layer_bytes, true, false};
            Ly.host_contig = true;
            Ly.slot_bytes = slot_bytes;
            Ly.stride[0] = Ly.stride[1] = Ly.stride[2] = slot_bytes;
            Ly.host[0] = (const uint8_t*) p;
            Ly.host[1] = Ly.host[0] + Ly.eb[0];
            Ly.host[2] = Ly.host[1] + Ly.eb[1];
            Ly.nopin.assign(c.E, 1);   // (till the rest is pinned, every copy in comes from the files)
        }
        for (const auto& J : exp_jobs) {
            Layer& Ly = M.L[J.l];
            const uint8_t* p = S.data(J.name);
            const size_t per = tensor_bytes(*S.info(J.name)) / c.E;
            for (size_t i = 0; i < J.roles.size(); ++i)
                Ly.fh[J.roles[i]] = p + i * Ly.eb[J.roles[i]], Ly.fstride[J.roles[i]] = per;
        }
        std::printf("experts: %.1f GiB, pinned once the VRAM tier is planned (only the ones it does not hold)\n",
                    host_total / 1073741824.0);
        std::fflush(stdout);
#endif
    } else {
        // Pinned host memory: anonymous mappings read into and then registered with CUDA, a layer as soon as its
        // reads are done (323 GiB in ~60 s from VAST); IK2_ALLOC=hostalloc: cudaHostAlloc instead, which pins at
        // ~3 GiB/s (the driver serializes it; 110-190 s)
#ifndef _WIN32
        bool use_mmap_reg = !(getenv("IK2_ALLOC") && std::string(getenv("IK2_ALLOC")) == "hostalloc") &&
                            host_register_works();
        pin.blocks.resize(c.n_layer);
        if (use_mmap_reg) {
            for (int l = 0; l < c.n_layer; ++l) {
                const Layer& Ly = M.L[l];
                const size_t slot_bytes = Ly.eb[0] + Ly.eb[1] + Ly.eb[2];
                const size_t layer_bytes = (size_t) c.E * slot_bytes;
                void* p = ::mmap(nullptr, layer_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) {
                    for (int k = 0; k < l; ++k) ::munmap(pin.blocks[k].ptr, pin.blocks[k].bytes);
                    pin.blocks.assign(c.n_layer, {});
                    use_mmap_reg = false;
                    break;
                }
                pin.blocks[l] = {p, layer_bytes, true, false};
            }
        }
#else
        bool use_mmap_reg = false;
        pin.blocks.resize(c.n_layer);
#endif
        // (pinned: the allocations and the reads overlap - a layer's reads start as soon as its buffer exists;
        // IK2_ALLOC_THREADS threads allocate, default 1)
        std::vector<std::atomic<int>> ready(c.n_layer);
        for (auto& r : ready) r.store(use_mmap_reg ? 1 : 0);
        for (int l = 0; l < c.n_layer; ++l) {
            Layer& Ly = M.L[l];
            const size_t slot_bytes = Ly.eb[0] + Ly.eb[1] + Ly.eb[2];
            Ly.host_contig = true;
            Ly.slot_bytes = slot_bytes;
            Ly.stride[0] = Ly.stride[1] = Ly.stride[2] = slot_bytes;
        }
        std::vector<std::thread> alloc_th;
        std::atomic<int> next_alloc{0};
        double t_alloc_end = 0;
        std::atomic<int> alloc_left{use_mmap_reg ? 0 : c.n_layer};
        if (!use_mmap_reg) {
            const int na = std::max(1, getenv("IK2_ALLOC_THREADS") ? std::atoi(getenv("IK2_ALLOC_THREADS")) : 1);
            for (int t = 0; t < na; ++t)
                alloc_th.emplace_back([&]() {
#if defined(__linux__)
                    // (the offload path on a PC of several NUMA nodes: the pinned experts spread over all of them,
                    //  page by page, so the CPU lane's threads on every node stream them at the sum of the nodes'
                    //  bandwidth - Jupiter's two Xeons: 85 GB/s against 50 from one node; IK2_NUMA=0: where the
                    //  allocation lands)
                    if (g_off_want && !(getenv("IK2_NUMA") && !std::atoi(getenv("IK2_NUMA")))) {
                        int nn = 0;
                        while (nn < 64 && std::filesystem::exists("/sys/devices/system/node/node" + std::to_string(nn))) ++nn;
                        if (nn > 1) {
                            const unsigned long mask = nn >= 64 ? ~0ul : (1ul << nn) - 1;
                            syscall(SYS_set_mempolicy, 3 /* MPOL_INTERLEAVE */, &mask, (unsigned long) nn + 1);
                        }
                    }
#endif
                    for (;;) {
                        const int l = next_alloc++;
                        if (l >= c.n_layer) return;
                        const size_t layer_bytes = (size_t) c.E * M.L[l].slot_bytes;
                        void* p = nullptr;
                        // (the offload path: mapped too - the GPU reads its misses from here itself)
                        CU(cudaHostAlloc(&p, layer_bytes, cudaHostAllocPortable | (g_off_want ? cudaHostAllocMapped : 0)));
                        pin.blocks[l] = {p, layer_bytes, false, false};
                        ready[l].store(1, std::memory_order_release);
                        if (--alloc_left == 0) t_alloc_end = now();
                    }
                });
        }
        auto wait_layer = [&](int l) {
            while (!ready[l].load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            Layer& Ly = M.L[l];
            uint8_t* buf = (uint8_t*) pin.blocks[l].ptr;
            Ly.host[0] = buf;
            Ly.host[1] = buf + Ly.eb[0];
            Ly.host[2] = buf + Ly.eb[0] + Ly.eb[1];
        };
        std::vector<std::atomic<int>> jobs_left(c.n_layer);
        for (auto& v : jobs_left) v.store(0);
        for (const auto& J : exp_jobs) ++jobs_left[J.l];
        std::mutex reg_m;
        std::condition_variable reg_cv;
        std::deque<int> reg_q;
        int reg_fail = 0;
        std::thread reg_th;
        if (use_mmap_reg)
            reg_th = std::thread([&]() {   // registers each layer once its reads are done, while the others are read
                for (int done = 0; done < c.n_layer; ++done) {
                    int l;
                    {
                        std::unique_lock<std::mutex> lk(reg_m);
                        reg_cv.wait(lk, [&] { return !reg_q.empty(); });
                        l = reg_q.front();
                        reg_q.pop_front();
                    }
                    const unsigned fl = cudaHostRegisterPortable | (g_off_want ? cudaHostRegisterMapped : 0);
                    if (cudaHostRegister(pin.blocks[l].ptr, pin.blocks[l].bytes, fl) == cudaSuccess)
                        pin.blocks[l].registered = true;
                    else
                        cudaGetLastError(), ++reg_fail;
                }
            });
        auto layer_read = [&](int l) {
            if (!use_mmap_reg) return;
            {
                std::lock_guard<std::mutex> lk(reg_m);
                reg_q.push_back(l);
            }
            reg_cv.notify_one();
        };
        for (int l = 0; l < c.n_layer; ++l)
            if (jobs_left[l].load() == 0) layer_read(l);
        std::vector<std::thread> th;
        std::atomic<int> next{0};
        for (int t = 0; t < threads; ++t)
            th.emplace_back([&]() {
                for (;;) {
                    const int j = next++;
                    if (j >= (int) exp_jobs.size()) return;
                    const auto& J = exp_jobs[j];
                    wait_layer(J.l);
                    const Layer& Ly = M.L[J.l];
                    const size_t role_off = J.roles[0] == 0 ? 0 : (J.roles[0] == 1 ? Ly.eb[0] : Ly.eb[0] + Ly.eb[1]);
                    const size_t exp_bytes = tensor_bytes(*S.info(J.name)) / c.E;
                    uint8_t* dst = (uint8_t*) pin.blocks[J.l].ptr + role_off;
                    read_into_strided(S.paths[S.file_of(J.name)], S.file_offset(J.name), c.E, exp_bytes,
                                      Ly.slot_bytes, dst);
                    if (--jobs_left[J.l] == 0) layer_read(J.l);
                }
            });
        for (auto& t : alloc_th) t.join();
        for (auto& t : th) t.join();
        for (int l = 0; l < c.n_layer; ++l) wait_layer(l);   // (host[] of layers no job read)
        t_alloc = (use_mmap_reg ? 0.0 : t_alloc_end - t_host);
        const double t_read = now() - t_host - t_alloc;   // (the reads left after the last allocation)
        double t_pin = 0;
        if (use_mmap_reg) {
            const double tp0 = now();
            reg_th.join();   // (the layers read last are still registering)
            t_pin = now() - tp0;
            if (reg_fail > 0)
                std::fprintf(stderr, "maya-ik2: warning: cudaHostRegister failed on %d layers (running pageable)\n",
                             reg_fail);
        }
        if (use_mmap_reg) {
            std::printf("experts in pinned (mmap+register) memory: %.1f GiB; allocating %.1f s, reading %.1f s "
                        "(%.2f GiB/s), pinning %.1f s\n",
                        host_total / 1073741824.0, t_alloc, t_read, host_total / 1073741824.0 / t_read, t_pin);
        } else {
            std::printf("experts in pinned memory: %.1f GiB; allocating %.1f s, reading %.1f s (%.2f GiB/s)\n",
                        host_total / 1073741824.0, t_alloc, t_read, host_total / 1073741824.0 / t_read);
        }
        std::fflush(stdout);
    }

    // the drafter (before the expert cache takes what VRAM is left)
    ik2::draft::Drafter* drafter = nullptr;
    uint16_t* draft_head = M.out_w.f16;
    if (!draft_path.empty()) {
        if (n_gpu != 1) die("--draft: one GPU only so far");
        if (!draft_head && M.out_w.q8) {   // (a Q8_0 head: an FP16 copy for the drafter's base logits and its fold)
            const size_t n = (size_t) c.n_rows * c.C;
            draft_head = dalloc<uint16_t>(n);
            k::decode_q8p_f16(M.out_w.q8, M.out_w.q8d, n, draft_head, g_s);
        }
        if (!draft_head) die("--draft: the output head is neither FP16 nor planar Q8_0");
        g_final_tap = c.n_pass == 1 ? 7 : 3;
        const double t0 = now();
        try {
            drafter = ik2::draft::load(draft_path, c.C, ctx, g_blas, g_s);
            ik2::draft::fold(drafter, draft_head, c.n_rows);
        } catch (const std::exception& e) {
            die(e.what());
        }
        if (M.out_w.q8 && draft_head != M.out_w.f16) {   // (the FP16 copy was the fold's: the base logits from the blocks)
            ik2::draft::set_head_q8(drafter, M.out_w.q8, M.out_w.q8d);
            CU(cudaStreamSynchronize(g_s));
            CU(cudaFree(draft_head));
            draft_head = nullptr;
        }
        g_taps_ld = std::max(chunk, ik2::draft::kBlock);
        g_taps = dalloc<float>((size_t) (g_final_tap + 1) * g_taps_ld * c.C);
        std::printf("drafter: %s, %.2f GiB of VRAM, %.1f s\n", draft_path.c_str(),
                    ik2::draft::bytes(drafter) / 1073741824.0, now() - t0);
    }
    if (drafter || g_lookup) {   // (the verify window's records of every conv: rollback_block)
        for (auto& Vi : M.V) {
            const size_t cc[4] = {KD, KD, C, C};
            for (int i = 0; i < 4; ++i) {
                Vi.rec_old[i] = dalloc<float>(3 * cc[i]);
                Vi.rec_x[i] = dalloc<float>((size_t) ik2::draft::kBlock * cc[i]);
            }
        }
    }

    // the work buffers and the expert cache, a set a GPU
    for (int d = 0; d < n_gpu; ++d) {
    use_dev(d);
    CU(cudaMemGetInfo(&fr, &tot));
    Work& W = g_dev[d].W;
    W.Tm = std::max(chunk, 1);
    W.ctx = ctx;
    W.nk_max = std::min(ctx, ctx);
    // Tokens a layer-major prefill pass (every expert not in VRAM streams in once a pass): 16384 when there is room
    // (a pass's scratch is ~280 KB a token: 4.6 GB, taken from the expert cache), else 4096.
    // (a looped model's prompts go through forward(): forward_prefill_sup has no loop yet)
    // ... nor the resident path's models: a pass borrows cache slots, which would evict experts
    W.T_sup = ((n_gpu == 1 || dual()) && draft_path.empty() && c.n_pass == 1 && !g_fast_want && fr >= (20ULL << 30)) ? (fr >= (60ULL << 30) ? 16384 : 4096) : 0;
    // IK2_SUP=N: tokens a layer-major prefill pass (each pass streams every expert not in the cache once)
    if (W.T_sup > 0 && getenv("IK2_SUP")) W.T_sup = std::max(W.Tm, std::atoi(getenv("IK2_SUP")));
    const size_t Tm = W.Tm, S8 = Tm * c.K;
    const size_t T_buf = std::max(Tm, (size_t) W.T_sup), S_buf = T_buf * c.K;
    W.blk_ld = T_buf;
    for (float** p : {&W.h, &W.r, &W.tmp}) *p = dalloc<float>(T_buf * C);
    for (float** p : {&W.res, &W.a, &W.xq, &W.xk, &W.xv}) *p = dalloc<float>(Tm * C);
    W.q = dalloc<float>(Tm * HD), W.gate = dalloc<float>(Tm * HD), W.o = dalloc<float>(Tm * HD);
    W.kk = dalloc<float>(Tm * KD), W.vv = dalloc<float>(Tm * KD), W.kconv = dalloc<float>(Tm * KD);
    W.vconv = dalloc<float>(Tm * KD);
    W.rr = dalloc<float>(Tm * c.H * c.d_rel), W.sink = dalloc<float>(Tm * c.H);
    W.dyn = dalloc<float>(2 * Tm * c.KV * 4);   // (a full layer's dynamic taps: k's, and on the resident path v's beside)
    W.lg_cap = std::max((size_t) 8 * c.H * (size_t) ctx + (size_t) 8 * c.H * 128 * 130, (size_t) 64 << 20);
    W.lg = dalloc<float>(W.lg_cap);
    W.blocks = dalloc<float>((size_t) (c.n_layer / c.block + 1) * T_buf * C);
    W.x16 = dalloc<uint16_t>(Tm * std::max(C, HD)), W.xk16 = dalloc<uint16_t>(Tm * C), W.xv16 = dalloc<uint16_t>(Tm * C);
    if (W.T_sup > 0 || dual()) W.xq16_sup = dalloc<uint16_t>(T_buf * C);
    if (!getenv("IK2_RELB") || std::atoi(getenv("IK2_RELB"))) {
        W.relb_ext = std::max(c.extent, c.extent_swa);
        W.relb = dalloc<float>((size_t) Work::kRelbRows * c.H * W.relb_ext);
    }
    if (g_kv_fp8) {
        W.kv_tmp_cap = std::min(ctx, 32768);
        W.kv_k16 = dalloc<uint16_t>((size_t) W.kv_tmp_cap * KD);
        W.kv_v16 = dalloc<uint16_t>((size_t) W.kv_tmp_cap * KD);
    }
    W.pos = dalloc<int>(T_buf);
    W.rlog = dalloc<float>(T_buf * c.E), W.route_d = dalloc<int>(S_buf * 2);
    W.ex16 = dalloc<uint16_t>(T_buf * C), W.w16 = dalloc<uint16_t>((size_t) 2 * c.F * C);   // (one expert's tokens)
    W.h16 = dalloc<uint16_t>(std::max<size_t>(64, T_buf) * c.F);
    W.gu32_bytes = 4 * std::max({S8 * 2 * (size_t) c.F,
                  Tm * std::max<size_t>(2 * HD + 2 * KD + (size_t) c.H * c.d_rel + c.H,
                                        (2 * HD + (size_t) c.H * c.d_rel + c.H) + 2 * (KD + (size_t) c.KV * 4)),
                  T_buf * 2 * (size_t) c.F, (W.lg_cap + 1) / 2});
    W.gu32 = dalloc<float>(W.gu32_bytes / 4);
    W.qkv = W.gu32;
    W.lg16 = (uint16_t*) W.gu32;
    W.y32 = dalloc<float>(std::max<size_t>(64, T_buf) * C);
    if (g_tp) W.tp_h = dalloc<uint16_t>(std::max<size_t>(64, S_buf) * c.F);   // (one expert's tokens; the token path's <= 64 rows)
    W.wtr = dalloc<float>(S_buf);
    W.sg = dalloc<float>(S_buf), W.su = dalloc<float>(S_buf), W.wt = dalloc<float>(S_buf);
    W.gidx = dalloc<int>(S_buf), W.inv = dalloc<int>(S_buf);
    W.blob = dalloc<const uint8_t*>(S8), W.tok = dalloc<int>(S8);
    for (int i = 0; i < Work::kPackRing; ++i) {
        CU(cudaHostAlloc(&W.pack_h[i], 8192, cudaHostAllocPortable));
        W.pack_d[i] = dalloc<uint8_t>(8192);
        if (g_pool) CU(cudaHostAlloc(&W.y_h[i], (size_t) 64 * C * 4, cudaHostAllocPortable));
    }
    W.T_host = (int) std::max<size_t>(T_buf, W.T_sup > 0 ? (size_t) ctx : 0);   // (a lent scratch can take up to ctx)
    const size_t S_host = (size_t) W.T_host * c.K;
    for (int i = 0; i < 2; ++i) {
        CU(cudaHostAlloc(&W.gidx_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.inv_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.sg_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.su_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.wt_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.wtr_h[i], S_host * 4, cudaHostAllocPortable));
    }
    CU(cudaHostAlloc(&W.x_h, (size_t) 8 * C * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.emb_h, Tm * C * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.pos_h, (size_t) W.T_host * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.route_h, S_host * 8, cudaHostAllocPortable));
    W.argmax_d = dalloc<int>(Tm);
    CU(cudaHostAlloc(&W.argmax_h, Tm * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.logits_h, (size_t) c.n_rows * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.cand_h, (size_t) (2 + 2 * kCandCap) * 4, cudaHostAllocPortable));
    {   // (IK2_RG_SLOTS: the slots - tokens x picks - up to which a prompt's experts go by groups; 0: never)
        // (96 tokens by default: past ~128 the FP16 GEMMs are the faster way)
        const int want = getenv("IK2_RG_SLOTS") ? std::atoi(getenv("IK2_RG_SLOTS")) : 96 * c.K;
        W.rg_slots = std::max(0, std::min(want, (int) std::max<size_t>(64, T_buf)));
        W.rg_groups = c.E + W.rg_slots / k::kGroupMax;
        W.rg_bytes = (size_t) W.rg_groups * (8 + 4 + 8 * k::kGroupMax) + (size_t) W.rg_slots * 16;
        CU(cudaHostAlloc(&W.rg_h, W.rg_bytes, cudaHostAllocPortable));
        W.rg_d = dalloc<uint8_t>(W.rg_bytes);
    }
    W.cand_d = dalloc<int>(2 + 2 * kCandCap);
    W.cand_part = dalloc<int>(k::cand_scratch(c.n_vocab));
    const int n_ev = (c.E + Work::kMissBatch - 1) / Work::kMissBatch;
    W.miss_ev.resize(n_ev);
    for (int i = 0; i < n_ev; ++i) CU(cudaEventCreateWithFlags(&W.miss_ev[i], cudaEventDisableTiming));
    for (int r = 0; r < Work::kPfRing; ++r) {
        W.pf_evs[r].resize(n_ev);
        for (int i = 0; i < n_ev; ++i) CU(cudaEventCreateWithFlags(&W.pf_evs[r][i], cudaEventDisableTiming));
    }
    W.cnt.assign(c.E + 1, 0);
    W.start.assign(c.E + 1, 0);
    W.fill.assign(c.E + 1, 0);
    W.hit_exps.reserve(c.E);
    W.miss_exps.reserve(c.E);
    W.slot_ptrs.assign(c.E, nullptr);
    W.seen_stamp.assign(c.E, 0);
    W.gi_stamp.assign(c.E, 0);
    W.seen_late.assign(c.E, 0);
    W.gi_idx.assign(c.E, 0);
    W.logits = dalloc<float>((size_t) 16 * c.n_rows);
    if (c.n_pass > 1) {
        W.lp_pre = dalloc<float>(Tm * C);
        W.lp_bank = dalloc<float>((size_t) (n_core / c.block) * Tm * C);   // (the pass's blocks after its first, its final stream)
        W.lp_msg = dalloc<float>(4 * Tm * C);
        W.lp_msg_all = dalloc<float>((size_t) (c.core1 - c.core0) * 4 * C);
        W.lp_in = dalloc<float>(3 * Tm * C);
        W.lp_g = dalloc<float>(3 * Tm * (2 * (size_t) c.rank + 2));
        W.lp_a = dalloc<float>(3 * Tm * (size_t) c.rank);
        W.lp_f = dalloc<float>(2 * Tm * C);
    }
    for (auto& Vi : M.V) {   // the conv histories start at zero
        if (M.L[Vi.l].dev != d) continue;
        CU(cudaMemset(Vi.h_k, 0, 3 * KD * 4));
        CU(cudaMemset(Vi.h_v, 0, 3 * KD * 4));
        CU(cudaMemset(Vi.h_attn, 0, 3 * C * 4));
        CU(cudaMemset(Vi.h_mlp, 0, 3 * C * 4));
    }
    ExpertCache& EC = g_dev[d].EC;
    if (g_off_want && d == 0) {   // the offload path's staging: 2 K slots a slot size (an expert's bytes)
        g_offs.pool.assign(c.n_layer, -1);
        std::vector<size_t> sizes;
        for (int l = 0; l < c.n_layer; ++l) {
            const size_t sb = M.L[l].slot_bytes;
            if (sb % 16 || !M.L[l].host_contig) die("the offload path: an expert's blocks not as it needs them");
            auto it = std::find(sizes.begin(), sizes.end(), sb);
            g_offs.pool[l] = (int) (it - sizes.begin());
            if (it == sizes.end()) sizes.push_back(sb);
        }
        const int n_st = std::max(c.K, getenv("IK2_STAGE") ? std::atoi(getenv("IK2_STAGE")) : 2 * c.K);
        for (size_t sb : sizes) {
            k::OffStage st;
            st.sb = sb, st.n = n_st;
            st.data = dalloc<uint8_t>((size_t) n_st * sb);
            st.tag = dalloc<int>(n_st);
            st.used = dalloc<uint32_t>(n_st);
            st.hand = dalloc<int>(1);
            CU(cudaMemsetAsync(st.tag, 0xff, (size_t) n_st * 4, g_s));
            CU(cudaMemsetAsync(st.used, 0, (size_t) n_st * 4, g_s));
            CU(cudaMemsetAsync(st.hand, 0, 4, g_s));
            g_offs.stage.push_back(st);
            std::vector<cudaEvent_t> a(n_st), b(n_st);
            for (int i = 0; i < n_st; ++i) {
                CU(cudaEventCreateWithFlags(&a[i], cudaEventDisableTiming));
                CU(cudaEventCreateWithFlags(&b[i], cudaEventDisableTiming));
            }
            g_offs.ring_free.push_back(a), g_offs.ring_full.push_back(b), g_offs.ring_hand.push_back(0);
        }
        g_offs.fl = dalloc<k::OffFetch>(1);
        g_offs.cnt = dalloc<k::OffCount>(1);
        g_offs.heat = dalloc<uint32_t>((size_t) c.n_layer * c.E);
        CU(cudaMemsetAsync(g_offs.cnt, 0, sizeof(k::OffCount), g_s));
        CU(cudaMemsetAsync(g_offs.heat, 0, (size_t) c.n_layer * c.E * 4, g_s));
        CU(cudaStreamSynchronize(g_s));
        int sms = 0;
        CU(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        g_offs.blocks = std::max(8, 2 * sms);
    }
    CU(cudaMemGetInfo(&fr, &tot));
    // (what is left free: 2 GiB beside a cache that streams experts in and out; a quarter of one when every expert
    //  stays in VRAM - nothing more is allocated but a token's graph; the offload path: IK2_ROOM_MB, 1 GiB)
    const size_t room = g_fast_want ? (size_t) 256 << 20
                      : g_off_want ? (size_t) (getenv("IK2_ROOM_MB") ? std::atof(getenv("IK2_ROOM_MB")) : 1024.0) << 20
                                   : (size_t) 2 << 30;
    const size_t budget = std::min((size_t) (cache_gb * 1073741824.0), fr > room ? fr - room : (size_t) 0);
    EC.tp = g_tp ? d : -1;
    EC.tier_only = g_off_want && d == 0 && !(getenv("IK2_TIER_MIN") && std::atoi(getenv("IK2_TIER_MIN")));
    std::vector<double> off_value;   // (the offload path: the profile's picks an expert, the tier's pools sized by them)
    if (g_off_want && d == 0)
        if (const char* wf = getenv("IK2_WARM"); wf && *wf)
            if (FILE* in = std::fopen(wf, "rb")) {
                int32_t hdr[2] = {0, 0};
                std::vector<uint64_t> cnt((size_t) c.n_layer * c.E);
                if (std::fread(hdr, 4, 2, in) == 2 && hdr[0] == c.n_layer && hdr[1] == c.E &&
                    std::fread(cnt.data(), 8, cnt.size(), in) == cnt.size())
                    off_value.assign(cnt.begin(), cnt.end());
                std::fclose(in);
            }
    EC.init(M.L, dual() ? -1 : d, c.n_layer, c.E, budget, off_value.empty() ? nullptr : &off_value);
    std::printf("expert cache, GPU %d: %d slots across %zu pools (%.1f GiB); load %.1f s in all\n", d, EC.total_slots(),
                EC.pools.size(), EC.total_bytes() / 1073741824.0, now() - t_start);
    for (size_t p = 0; p < EC.pools.size(); ++p)
        std::printf("  pool %zu: %d slots of %.1f MiB (%.1f GiB, %d layers)\n", p, EC.pools[p].n_slots,
                    EC.pools[p].slot_bytes / 1048576.0,
                    (double) EC.pools[p].n_slots * EC.pools[p].slot_bytes / 1073741824.0, EC.pools[p].n_layers);
    const char* wf = getenv("IK2_WARM");
    if ((wf && *wf) || g_ep) {   // the cache filled with the most picked experts (IK2_POP_OUT's file or all owned for EP)
        FILE* in = (wf && *wf) ? std::fopen(wf, "rb") : nullptr;
        int32_t hdr[2] = {0, 0};
        std::vector<uint64_t> cnt;
        if (in && std::fread(hdr, 4, 2, in) == 2 && hdr[0] == c.n_layer && hdr[1] == c.E) {
            cnt.resize((size_t) c.n_layer * c.E);
            if (std::fread(cnt.data(), 8, cnt.size(), in) != cnt.size()) cnt.clear();
        }
        if (in) std::fclose(in);
        if (cnt.empty() && g_ep) cnt.assign((size_t) c.n_layer * c.E, 1);
        if (cnt.empty()) {
            std::printf("warm: the cache starts empty\n");
        } else {
            const double tw = now();
            const uint64_t m0 = EC.misses, b0 = EC.bytes_up;
            int n = 0;
            for (size_t p = 0; p < EC.pools.size(); ++p) {
                std::vector<int> idx;
                for (int l = 0; l < c.n_layer; ++l)
                    if (EC.layer_pool[l] == (int) p)
                        for (int e = 0; e < c.E; ++e)
                            if (!g_ep || ep_owner(l, e) == d) idx.push_back(l * c.E + e);
                // the most picked first; the chosen ones then go in least picked first, so that the evictions that
                // start from the oldest take the least picked
                std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
                const int k = std::min((int) idx.size(), EC.pools[p].n_active);
                for (int i = k - 1; i >= 0; --i) {
                    const int l = idx[i] / c.E, e = idx[i] % c.E;
                    EC.get(M.L[l], l, e, EC.tick, g_cs, nullptr, true, true);
                    ++n;
                }
            }
            CU(cudaStreamSynchronize(g_cs));
            ++EC.tick;   // (taken in the tick the first prefetch uses, they would all count as in use: none evictable)
            // the profile also seeds the prefetch's picture of which experts a pass uses (W.use, as 100k picks a
            // layer): without it the first long-enough request prefetches all 18k experts
            W.use.assign((size_t) c.n_layer * c.E, 0), W.use_tot.assign(c.n_layer, 0);
            for (int l = 0; l < c.n_layer; ++l) {
                long double sum = 0;
                for (int e = 0; e < c.E; ++e) sum += cnt[(size_t) l * c.E + e];
                if (sum <= 0) continue;
                for (int e = 0; e < c.E; ++e) {
                    W.use[(size_t) l * c.E + e] = (uint64_t) (100000.0L * cnt[(size_t) l * c.E + e] / sum + 0.5L);
                    W.use_tot[l] += W.use[(size_t) l * c.E + e];
                }
            }
            std::printf("warm: %d experts (%.1f GiB) copied in %.1f s\n", n, (EC.bytes_up - b0) / 1073741824.0,
                        now() - tw);
            EC.misses = m0, EC.bytes_up = b0;
        }
    }
    if (g_off_want && d == 0) {   // the offload path: the tier is what the cache holds (warmed above), tables a layer
#ifndef _WIN32
        if (pin_later) {   // the experts the tier does not hold: read into their blocks, then pinned a run at a time
            const double tp0 = now();
            for (int l = 0; l < c.n_layer; ++l)
                for (int e = 0; e < c.E; ++e) M.L[l].nopin[e] = EC.loc[(size_t) l * c.E + e] >= 0;
            auto runs_of = [&](int l, auto&& f) {   // f(e0, n): a run of the layer's experts not in the tier
                const auto& np = M.L[l].nopin;
                for (int e = 0; e < c.E;) {
                    int e1 = e;
                    while (e1 < c.E && !np[e1]) ++e1;
                    if (e1 > e) f(e, e1 - e);
                    e = e1 + 1;
                }
            };
            std::vector<std::atomic<int>> left(c.n_layer);
            for (auto& v : left) v.store(0);
            for (const auto& J : exp_jobs) ++left[J.l];
            std::atomic<int> next{0}, fails{0};
            std::atomic<size_t> pinned{0};
            std::mutex rm;
            const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
            std::vector<std::thread> th;
            for (int t = 0; t < std::max(1, threads); ++t)
                th.emplace_back([&]() {
                    for (;;) {
                        const int j = next++;
                        if (j >= (int) exp_jobs.size()) return;
                        const auto& J = exp_jobs[j];
                        const Layer& Ly = M.L[J.l];
                        const size_t role_off = J.roles[0] == 0 ? 0 : (J.roles[0] == 1 ? Ly.eb[0] : Ly.eb[0] + Ly.eb[1]);
                        const size_t exp_bytes = tensor_bytes(*S.info(J.name)) / c.E;
                        uint8_t* blk = (uint8_t*) pin.blocks[J.l].ptr;
                        runs_of(J.l, [&](int e0, int n) {
                            read_into_strided(S.paths[S.file_of(J.name)], S.file_offset(J.name) + (uint64_t) e0 * exp_bytes,
                                              n, exp_bytes, Ly.slot_bytes, blk + (size_t) e0 * Ly.slot_bytes + role_off);
                        });
                        if (--left[J.l]) continue;
                        runs_of(J.l, [&](int e0, int n) {   // (the layer's reads done: its runs pinned, mapped)
                            const uintptr_t a = ((uintptr_t) blk + (size_t) e0 * Ly.slot_bytes) & ~(uintptr_t) (pg - 1);
                            const uintptr_t b = ((uintptr_t) blk + (size_t) (e0 + n) * Ly.slot_bytes + pg - 1) & ~(uintptr_t) (pg - 1);
                            void* dp = nullptr;
                            if (cudaHostRegister((void*) a, b - a, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
                                cudaGetLastError(), ++fails;
                                return;
                            }
                            {
                                std::lock_guard<std::mutex> lk(rm);
                                pin.runs.push_back({(void*) a, (size_t) (b - a)});
                            }
                            if (cudaHostGetDevicePointer(&dp, (void*) a, 0) != cudaSuccess || dp != (void*) a)
                                cudaGetLastError(), ++fails;
                            pinned += (size_t) n * Ly.slot_bytes;
                        });
                    }
                });
            for (auto& t : th) t.join();
            if (fails) die("the offload path: the experts the tier does not hold could not be pinned (IK2_PIN_ALL=1: "
                           "every expert pinned at the load)");
            double anon = 0;
            if (FILE* f = std::fopen("/proc/self/status", "r")) {
                char ln[256];
                while (std::fgets(ln, sizeof ln, f))
                    if (!std::strncmp(ln, "RssAnon:", 8)) anon = std::atof(ln + 8) / 1048576.0;
                std::fclose(f);
            }
            std::printf("experts in pinned memory: %.1f GiB (the %.1f GiB the tier holds not in RAM), read and pinned in "
                        "%.1f s; the engine's RAM %.1f GiB in all\n", pinned.load() / 1073741824.0,
                        (host_total - pinned.load()) / 1073741824.0, now() - tp0, anon);
            std::fflush(stdout);
        }
#endif
        g_offs.host.assign(c.n_layer, nullptr);
        for (int l = 0; l < c.n_layer; ++l) {
            void* dp = (void*) M.L[l].host[0];   // (pinned in runs: at their own addresses, checked above)
            if (M.L[l].nopin.empty() && (cudaHostGetDevicePointer(&dp, (void*) M.L[l].host[0], 0) != cudaSuccess || !dp))
                die("the offload path: layer " + std::to_string(l) + "'s experts are not mapped for the GPU (pinning "
                    "failed - IK2_OFF=0 runs the cache path)");
            g_offs.host[l] = (const uint8_t*) dp;
        }
        auto blocks = [](const Mat& m) { return m.raw || m.q8; };
        bool raw = blocks(M.out_w);
        for (const auto& Ly : M.L)
            raw = raw && blocks(Ly.wo) && (Ly.sliding ? blocks(Ly.wqkvrg) : (blocks(Ly.wqrg) && blocks(Ly.wk) && blocks(Ly.wv)));
        if (!raw) die("the offload path: a dense matrix is not kept as blocks");
        for (int l = 0; l < c.n_layer; ++l) {
            Layer& Ly = M.L[l];
            Ly.tab_blob = dalloc<const uint8_t*>(c.E);
            for (int r = 0; r < 3; ++r)
                if (!Ly.scale[r].empty()) Ly.tab_sc[r] = dev_f32(Ly.scale[r]);
        }
        g_off = true;
        off_tables(M);
        // The decode mode (owner, v1.1.73): lane parity wherever it is the faster one on this PC (IK2_LANE unset or
        // auto: decided from the calibration's figures), else mode A (the resident kernels, the bus only);
        // IK2_LANE=0: mode A, =1 / fetch / cpu: lane parity (tests).
        const char* le = getenv("IK2_LANE");
        const std::string lw = le && *le ? le : "auto";
        if (lw != "0") {
            g_lane = lane_init(M, lw, getenv("IK2_LANE_THREADS") ? std::atoi(getenv("IK2_LANE_THREADS")) : -1);
            if (!g_lane) std::printf("lane parity: off - an expert format the exact lane does not have\n");
        }
        if (g_lane && lw == "auto") {
            // a visit's expected misses from the tier's share of the profile's picks (all picks' share of the bytes
            // when there is no profile), then mode A's time (copies, then the experts; the resident kernels taken
            // as twice as fast as the exact ones) against lane parity's best split of them
            double hit = 0, all = 0;
            for (int i = 0; i < c.n_layer * c.E; ++i) {
                const double v = off_value.empty() ? 1.0 : off_value[(size_t) i];
                all += v, hit += EC.loc[(size_t) i] >= 0 ? v : 0.0;
            }
            const double h = all > 0 ? hit / all : 0.0, nm = c.K * (1.0 - h);
            const k::LaneCost& cs = g_offs.cost_h;
            const double t_a = nm * cs.fetch[0] + 0.5 * c.K * cs.gpu[0];
            double t_b = 1e30;
            for (int n = 0; n <= (int) std::ceil(nm); ++n) {
                const double nn = std::min<double>(n, nm);
                t_b = std::min(t_b, std::max(nn > 0 ? cs.lat + nn * cs.cpu[0] : 0.0,
                                             (nm - nn) * cs.fetch[0] + c.K * cs.gpu[0]));
            }
            const bool faster = t_b < 0.95 * t_a;
            std::printf("decode mode: %s (a visit ~%.0f us with the lane, ~%.0f us without; tier %.0f%% of the "
                        "profile's picks)\n", faster ? "lane parity" : "A (the lane is not faster here)", t_b, t_a,
                        100.0 * h);
            if (!faster) g_lane = false, g_offs.lane.reset();
        }
        int n_tier = 0;
        for (int i = 0; i < c.n_layer * c.E; ++i) n_tier += EC.loc[i] >= 0;
        size_t st_b = 0;
        for (const auto& st : g_offs.stage) st_b += (size_t) st.n * st.sb;
        std::printf("offload: %d of %d experts in VRAM (%.1f GiB), the rest fetched by the GPU from pinned host "
                    "memory through %zu staging pools (%.2f GiB); tokens take the resident path's launches%s\n",
                    n_tier, c.n_layer * c.E, EC.total_bytes() / 1073741824.0, g_offs.stage.size(), st_b / 1073741824.0,
                    g_graph ? ", a CUDA graph each" : "");
    }
    if (g_fast_want && d == 0) {   // the resident path: every expert in now, and where each is on the device
        auto blocks = [](const Mat& m) { return m.raw || m.q8; };   // (kept as the GGUF's blocks, whole or planar)
        bool all = !EC.pools.empty(), raw = blocks(M.out_w);
        for (const auto& P : EC.pools) all = all && P.n_active >= P.n_layers * c.E;
        for (const auto& Ly : M.L)
            raw = raw && blocks(Ly.wo) && (Ly.sliding ? blocks(Ly.wqkvrg) : (blocks(Ly.wqrg) && blocks(Ly.wk) && blocks(Ly.wv)));
        if (all && raw) {
            const double tw = now();
            const uint64_t m0 = EC.misses, b0 = EC.bytes_up;
            int n = 0;
            for (int l = 0; l < c.n_layer; ++l)
                for (int e = 0; e < c.E; ++e)
                    if (!EC.has(l, e)) EC.get(M.L[l], l, e, EC.tick, g_cs, nullptr, true, true), ++n;
            CU(cudaStreamSynchronize(g_cs));
            ++EC.tick;
            for (int l = 0; l < c.n_layer; ++l) {
                Layer& Ly = M.L[l];
                const auto& P = EC.pools[EC.layer_pool[l]];
                std::vector<const uint8_t*> tab(c.E);
                for (int e = 0; e < c.E; ++e) tab[e] = P.data + (size_t) EC.loc[(size_t) l * c.E + e] * P.slot_bytes;
                Ly.tab_blob = dalloc<const uint8_t*>(c.E);
                CU(cudaMemcpy(Ly.tab_blob, tab.data(), (size_t) c.E * sizeof(void*), cudaMemcpyHostToDevice));
                Ly.tab_h = tab;
                for (int r = 0; r < 3; ++r)
                    if (!Ly.scale[r].empty()) Ly.tab_sc[r] = dev_f32(Ly.scale[r]);
            }
            EC.misses = m0, EC.bytes_up = b0;
            g_fast = true;
            std::printf("resident: every expert in VRAM (%d copied in %.1f s); tokens take the resident path%s\n", n,
                        now() - tw, g_graph ? ", a CUDA graph each" : "");
        } else {
            std::printf("resident path off: %s\n", all ? "a dense matrix is not kept as blocks" : "the cache does not hold every expert");
        }
    }
    std::fflush(stdout);
    }
    // the caches' counters, summed over the GPUs
    auto ec_sum = [&](auto f) {
        double v = 0;
        for (auto& D : g_dev) v += (double) f(D.EC);
        return v;
    };

    if (serve) {
        // --serve: requests on stdin, a line each - "GEN max_new temperature top_p seed stop,stop ids,ids,..." - and
        // the answer on stdout as "@TOK id" lines then "@END reason prompt_tokens cached_tokens prompt_s gen_s".  The
        // tokens whose state the model holds are kept: a request that starts with them runs only what follows.
        std::vector<int> hist;
        std::vector<uint64_t> hist_sig;   // a picture row's hash at its position (shorter than hist: the rest are tokens)
        std::vector<float> last;
        double t_exp = 0;
        // Snapshots of every short conv's history (plus sliding-layer KV windows in host RAM) at the end of a
        // request's prompt (before its generation): an agent's next turn usually agrees with the last one up to
        // there and no further (the model's own reply comes back re-rendered, its thinking dropped), so the state
        // goes back to that point instead of the start. Full-layer KV needs nothing: positions rewrite cleanly.
        struct Snap {
            int pos = -1;
            std::vector<float*> h;   // [layer * 4 + conv]
            std::vector<std::vector<uint8_t>> kv_k, kv_v;
        };
        const size_t kv_elem_b = g_kv_fp8 ? 1 : 2;
        std::vector<Snap> snaps(8);
        for (auto& sn : snaps) {
            sn.kv_k.resize(M.V.size());
            sn.kv_v.resize(M.V.size());
            for (size_t l = 0; l < M.V.size(); ++l) {   // (l: a visit)
                auto& Ly = bind_visit(M, (int) l);
                use_dev(Ly.dev);
                const size_t cc[4] = {KD, KD, C, C};
                for (int i = 0; i < 4; ++i) sn.h.push_back(dalloc<float>(3 * cc[i]));
                if (Ly.sliding && Ly.kv_cap < ctx) {
                    const size_t bytes = (size_t) c.window * KD * kv_elem_b;
                    sn.kv_k[l].resize(bytes);
                    sn.kv_v[l].resize(bytes);
                }
            }
        }
        int snap_next = 0;
        auto snap_copy = [&](Snap& sn, bool save) {
            const size_t cc[4] = {KD, KD, C, C};
            const cudaMemcpyKind kv_dir = save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            const size_t row_b = (size_t) KD * kv_elem_b;
            for (size_t l = 0; l < M.V.size(); ++l) {   // (l: a visit)
                Layer& Ly = bind_visit(M, (int) l);
                use_dev(Ly.dev);
                float* hs[4] = {Ly.h_k, Ly.h_v, Ly.h_attn, Ly.h_mlp};
                for (int i = 0; i < 4; ++i) {
                    float* sv = sn.h[l * 4 + i];
                    CU(cudaMemcpyAsync(save ? sv : hs[i], save ? hs[i] : sv, 3 * cc[i] * 4, cudaMemcpyDeviceToDevice, g_s));
                }
                if (sn.pos > 0 && Ly.sliding && Ly.kv_cap < ctx) {
                    const int w = std::min(sn.pos, c.window);
                    const int k_start = sn.pos - w;
                    const int s0 = k_start % Ly.kv_cap;
                    const int n1 = std::min(w, Ly.kv_cap - s0);
                    const int n2 = w - n1;
                    uint8_t* dk = g_kv_fp8 ? Ly.kc8 : (uint8_t*) Ly.kc;
                    uint8_t* dv = g_kv_fp8 ? Ly.vc8 : (uint8_t*) Ly.vc;
                    uint8_t* hk = sn.kv_k[l].data();
                    uint8_t* hv = sn.kv_v[l].data();
                    CU(cudaMemcpyAsync(save ? hk : dk + (size_t) s0 * row_b, save ? dk + (size_t) s0 * row_b : hk,
                                       (size_t) n1 * row_b, kv_dir, g_s));
                    CU(cudaMemcpyAsync(save ? hv : dv + (size_t) s0 * row_b, save ? dv + (size_t) s0 * row_b : hv,
                                       (size_t) n1 * row_b, kv_dir, g_s));
                    if (n2 > 0) {
                        CU(cudaMemcpyAsync(save ? hk + (size_t) n1 * row_b : dk,
                                           save ? dk : hk + (size_t) n1 * row_b, (size_t) n2 * row_b, kv_dir, g_s));
                        CU(cudaMemcpyAsync(save ? hv + (size_t) n1 * row_b : dv,
                                           save ? dv : hv + (size_t) n1 * row_b, (size_t) n2 * row_b, kv_dir, g_s));
                    }
                }
            }
            for (auto& D : g_dev) CU(cudaStreamSynchronize(D.s));
        };
        auto reset = [&]() {
            for (auto& sn : snaps) sn.pos = -1;
            for (int vis = 0; vis < (int) M.V.size(); ++vis) {
                Layer& Ly = bind_visit(M, vis);
                use_dev(Ly.dev);
#ifdef STRATA_USE_HIP   // (AMD: on the engine's stream - a plain memset is a device-wide wait there, 192 a request)
                CU(cudaMemsetAsync(Ly.h_k, 0, 3 * KD * 4, g_s));
                CU(cudaMemsetAsync(Ly.h_v, 0, 3 * KD * 4, g_s));
                CU(cudaMemsetAsync(Ly.h_attn, 0, 3 * C * 4, g_s));
                CU(cudaMemsetAsync(Ly.h_mlp, 0, 3 * C * 4, g_s));
#else
                CU(cudaMemset(Ly.h_k, 0, 3 * KD * 4));
                CU(cudaMemset(Ly.h_v, 0, 3 * KD * 4));
                CU(cudaMemset(Ly.h_attn, 0, 3 * C * 4));
                CU(cudaMemset(Ly.h_mlp, 0, 3 * C * 4));
#endif
            }
            hist.clear();
            hist_sig.clear();
            last.clear();
        };
        auto parse_ints = [](const std::string& t) {
            std::vector<int> v;
            for (const char* p = t.c_str(); *p;) {
                if (*p == '-' && p[1] == 0) break;
                v.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
            return v;
        };
        std::signal(SIGUSR1, on_usr1);
        // Maya's protocol (serve/server.py's StrataEngine): "GEN <max_new> [key=value ...] <ids>" in; "PP ..." per
        // prompt pass, "T <id>" per token and "DONE <generated> <prompt> <prompt_ms> <decode_ms> <stop|length|cancel>
        // 0 0 <reused>" out; "STOP" mid-request ends it at the next token (or between the prompt's passes), "QUIT"
        // ends the engine.  The tokens a request ends at: --eos, else the GGUF's eos and <|endoftext|>.
        if (eos_ids.empty()) {
            if (const auto* e = S.get("tokenizer.ggml.eos_token_id")) eos_ids.push_back((int) e->u);
            eos_ids.push_back(199999);   // (Inkling 2's <|endoftext|>)
        }
        // --drafting auto, not measured on this PC yet: each source that loaded - none, the prompt lookup, the
        // drafter, both - times greedy answers to IK2_DRAFT_IDS's requests (IK2_DRAFT_TOKENS each, 128), the
        // fastest kept (within IK2_DRAFT_TIE, 3%, of none: none) and remembered; the answers are the same whichever
        // (drafting on = off), so one load measures them all
        if (drafting == "auto" && draft_choice < 0 && (drafter || g_lookup) && g_fast && g_block) {
            std::vector<std::vector<int>> reqs;
            if (const char* f = getenv("IK2_DRAFT_IDS")) {
                std::ifstream in(f);
                std::string l;
                while (std::getline(in, l))
                    if (!l.empty() && l[0] != '#') {
                        auto v = parse_ints(l);
                        if (!v.empty() && (int) v.size() + 256 < ctx) reqs.push_back(std::move(v));
                    }
            }
            if (reqs.empty()) {
                std::printf("NOTE maya-ik2: drafting on, not measured (no requests to time it on: IK2_DRAFT_IDS)\n");
            } else {
                const int gen_n = getenv("IK2_DRAFT_TOKENS") ? std::max(16, std::atoi(getenv("IK2_DRAFT_TOKENS"))) : 128;
                const double tie = getenv("IK2_DRAFT_TIE") ? std::atof(getenv("IK2_DRAFT_TIE")) : 0.03;
                const bool have[4] = {true, g_lookup, drafter != nullptr, drafter && g_lookup};
                std::printf("NOTE maya-ik2: measuring this PC once, before the first answer: drafting (%zu requests, %d "
                            "tokens each, a source at a time); remembered, the next starts skip it\n", reqs.size(), gen_n);
                std::fflush(stdout);
                const double tm0 = now();
                auto trial = [&](int mode, int gen_t, double& secs) {
                    int made = 0;
                    secs = 0;
                    for (const auto& ids : reqs) {
                        reset();
                        int first = 0;
                        for (size_t p0 = 0; p0 < ids.size(); p0 += chunk) {
                            const int n = (int) std::min<size_t>(chunk, ids.size() - p0);
                            const bool last_c = p0 + n == ids.size();
                            std::vector<float> lo;
                            std::vector<int> ag;
                            forward(M, ids.data() + p0, n, (int) p0, false, last_c ? OutMode::kArgmax : OutMode::kNone,
                                    std::string(), lo, &ag, t_exp);
                            if (drafter) {
                                ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, n, (int) p0);
                                if (last_c) ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n - 1);
                            }
                            if (last_c) first = ag[0];
                        }
                        CU(cudaStreamSynchronize(g_s));
                        SpecCtl ctl;
                        ctl.wmax = getenv("IK2_DRAFT_WIDTH") ? std::clamp(atoi(getenv("IK2_DRAFT_WIDTH")), 2, kBlockMax) : 5;
                        ctl.use_lookup = mode == 1 || mode == 3, ctl.use_model = mode >= 2;
                        if (mode == 0) ctl.wmax = 1;   // (plain tokens only)
                        strata::spec::SuffixDrafter lk1;
                        lk1.append(ids.data(), ids.size());
                        std::vector<int> out;
                        const double t0 = now();
                        decode_greedy(M, mode >= 2 ? drafter : nullptr, first, (int) ids.size(), gen_t, ctx, ctl, lk1, out,
                                      nullptr);
                        secs += now() - t0;
                        made += (int) out.size() - 1;
                    }
                    return made / std::max(1e-9, secs);
                };
                double warm = 0, tps[4] = {-1, -1, -1, -1};
                trial(have[3] ? 3 : have[2] ? 2 : 1, 24, warm);   // (the windows' graphs made once, untimed)
                for (int m = 0; m < 4; ++m)
                    if (have[m]) tps[m] = trial(m, gen_n, warm);
                int best = 0;
                for (int m = 1; m < 4; ++m)
                    if (tps[m] > tps[best]) best = m;
                if (best && tps[best] < tps[0] * (1.0 + tie)) best = 0;   // (no clear gain: no drafting)
                draft_choice = best;
                draft_remember(draft_key_s, tps, best);
                reset();
                std::printf("NOTE maya-ik2: measured this PC once in %.0f s: drafting - %s (no drafting %.1f, the prompt "
                            "lookup %.1f, the drafter %.1f, both %.1f tokens/s)\n", now() - tm0, kDraftNames[best],
                            tps[0], tps[1], tps[2], tps[3]);
            }
            if (draft_choice >= 0) {   // (the sources not chosen: off - the drafter's room stays as it was planned)
                if (draft_choice < 2) drafter = nullptr;
                if (draft_choice != 1 && draft_choice != 3) g_lookup = false;
            }
        }
        if (serve_maya) {
            // (dspark= / lookup=: the drafters this start runs - the Monitor's Speculation; drafting= the choice)
            const bool sp = (drafter || g_lookup) && g_fast && g_block;
            std::printf("INFO engine=ik2 engine_kind=ik2 model=Inkling-2 kv=%s ctx=%d gpus=%d dspark=%d lookup=%d "
                        "drafting=%s\n", g_kv_fp8 ? "fp8" : "f16", ctx, n_gpu, sp && drafter ? 1 : 0, sp && g_lookup ? 1 : 0,
                        drafting.c_str());
            std::printf("READY %d stop\n", ctx);
        } else {
            std::printf("@READY ctx %d\n", ctx);
        }
        std::fflush(stdout);
        // stdin on a thread: a STOP while a request runs is seen at once (the request loop looks at g_cancel)
        // (static: the reader thread is detached and may outlive this scope at QUIT)
        static std::mutex in_m;
        static std::condition_variable in_cv;
        static std::deque<std::string> in_q;
        static bool in_eof = false;
        static std::atomic<bool> busy{false};
        std::thread([]() {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    if (busy) g_cancel = 1;
                    continue;
                }
                std::lock_guard<std::mutex> lk(in_m);
                in_q.push_back(l);
                in_cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(in_m);
            in_eof = true;
            in_cv.notify_one();
        }).detach();
        auto next_line = [&](std::string& l) {
            std::unique_lock<std::mutex> lk(in_m);
            in_cv.wait(lk, [&] { return !in_q.empty() || in_eof; });
            if (in_q.empty()) return false;
            l = std::move(in_q.front());
            in_q.pop_front();
            return true;
        };
        std::string line;
        std::mt19937_64 rng(1234);
        // speculative decoding (--draft, the resident path): the controller decides each step; IK2_DRAFT_WIDTH the
        // widest window, IK2_DRAFT_ADAPT=0 always that wide
        SpecCtl sctl;
        const bool spec_on = (drafter || g_lookup) && g_fast && g_block;
        sctl.wmax = getenv("IK2_DRAFT_WIDTH") ? std::clamp(atoi(getenv("IK2_DRAFT_WIDTH")), 2, kBlockMax) : 5;
        sctl.adapt = !(getenv("IK2_DRAFT_ADAPT") && !atoi(getenv("IK2_DRAFT_ADAPT")));
        sctl.use_model = drafter != nullptr, sctl.use_lookup = g_lookup;
        static strata::spec::SuffixDrafter lk;   // (the prompt lookup: a request's tokens so far)
        while (next_line(line)) {
            std::istringstream in(line);
            std::string cmd, stops_s, ids_s, emb_path;
            int max_new = 0, top_k = 0, spec_key = 0;
            double temp = 0, top_p = 1;
            long long seed = -1;
            in >> cmd;
            if (cmd == "QUIT") {
                pop_dump();
                break;
            }
            // GENI: GEN with the request's pictures - "GENI <max_new> [key=value ...] <pictures file> <ids>", the file
            // the image encoder's rows (SVE1 records back to back), one row an image token of the ids in order
            const bool geni = cmd == "GENI";
            if (cmd != "GEN" && !geni) {   // (Maya's PAUSE / WRAP and the like: not offered in READY, so not sent)
                if (!serve_maya) {
                    std::printf("@END error unknown_command 0 0 0 0\n");
                    std::fflush(stdout);
                }
                continue;
            }
            std::vector<int> stops;
            if (serve_maya) {
                in >> max_new;
                std::string w;
                while (in >> w) {
                    const size_t eq = w.find('=');
                    if (eq == std::string::npos) {
                        if (geni && emb_path.empty() && ids_s.empty() && w.find_first_not_of("0123456789,-") != std::string::npos)
                            emb_path = w;   // (the pictures' file comes before the ids)
                        else
                            ids_s = w;
                        continue;
                    }
                    const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                    if (k == "temperature") temp = std::atof(v.c_str());
                    else if (k == "top_p") top_p = std::atof(v.c_str());
                    else if (k == "top_k") top_k = std::atoi(v.c_str());
                    else if (k == "seed") seed = std::atoll(v.c_str());
                    else if (k == "spec") spec_key = std::atoi(v.c_str());   // (1: no drafting; 2 .. 8: the widest window)
                }
                stops = eos_ids;
            } else {
                in >> max_new >> temp >> top_p >> seed >> stops_s >> ids_s;
                stops = parse_ints(stops_s);
            }
            const std::vector<int> ids = parse_ints(ids_s);
            if (ids.empty() || (int) ids.size() + max_new > ctx) {
                if (serve_maya)
                    std::printf("ERR %s\nDONE 0 %zu 0 0 cancel 0 0 0\n",
                                ids.empty() ? "an empty prompt" :
                                    ("the prompt and max_tokens pass the context (" + std::to_string(ctx) + ")").c_str(),
                                ids.size());
                else
                    std::printf("@END error %s %zu 0 0 0\n", ids.empty() ? "empty_prompt" : "context_length", ids.size());
                std::fflush(stdout);
                continue;
            }
            g_img.clear();
            std::vector<uint64_t> sig(ids.size(), 0);
            if (!emb_path.empty()) {
                const std::string err = load_pictures(emb_path, ids, image_token, c.C, g_img, sig);
                if (!err.empty()) {
                    g_img.clear();
                    std::printf("ERR %s\nDONE 0 %zu 0 0 cancel 0 0 0\n", err.c_str(), ids.size());
                    std::fflush(stdout);
                    continue;
                }
            }
            busy = true;
            if (seed >= 0) rng.seed((uint64_t) seed);
            g_cancel = 0;
            // (drafting for this request: spec=1 asks for none; the prompt lookup's history its tokens; the Monitor's
            //  figures - STAT draft_acc / draft_off, DONE's kept and offered - this request's drafts)
            const bool req_spec = spec_on && spec_key != 1;
            const int req_wmax = spec_key >= 2 ? std::min(spec_key, kBlockMax) : kBlockMax;
            const uint64_t kept0 = sctl.kept(), offered0 = sctl.offered();
            uint64_t req_rounds = 0;
            double req_draft_s = 0, req_verify_s = 0;
            if (req_spec) {
                lk.reset();
                lk.append(ids.data(), ids.size());
            }
            auto stat_line = [&]() {
                std::printf("STAT draft_acc=%llu draft_off=%llu draft_rounds=%llu draft_ms=%.1f verify_ms=%.1f\n",
                            (unsigned long long) (sctl.kept() - kept0), (unsigned long long) (sctl.offered() - offered0),
                            (unsigned long long) req_rounds, req_draft_s * 1e3, req_verify_s * 1e3);
            };
            size_t L = 0;   // (a picture's position is the same only with the same rows: its hash beside its token)
            auto hsig = [&](size_t i) { return i < hist_sig.size() ? hist_sig[i] : (uint64_t) 0; };
            while (L < hist.size() && L < ids.size() && hist[L] == ids[L] && hsig(L) == sig[L]) ++L;
            if (L < hist.size() || L == ids.size()) {
                // back to the latest snapshot at or before L that leaves a token to run (one past it is dropped:
                // its position's state follows from a prefix the request does not share)
                int best = -1;
                for (int i = 0; i < (int) snaps.size(); ++i)
                    if (snaps[i].pos >= 0 && snaps[i].pos <= (int) L && snaps[i].pos < (int) ids.size() &&
                        (best < 0 || snaps[i].pos > snaps[best].pos))
                        best = i;
                if (L == hist.size() && L == ids.size() && !last.empty()) {
                    // (the same tokens again: the last logits answer)
                } else if (best >= 0) {
                    const int sp = snaps[best].pos;
                    snap_copy(snaps[best], false);
                    for (auto& sn : snaps)
                        if (sn.pos > sp) sn.pos = -1;
                    hist.resize(sp);
                    if (hist_sig.size() > (size_t) sp) hist_sig.resize(sp);
                    last.clear();
                    L = sp;
                } else {
                    reset(), L = 0;
                }
            }
            const int cached = (int) L;
            const double t0 = now();
            const uint64_t miss0 = dev_cache(0).misses, bytes0 = dev_cache(0).bytes_up;
            for (auto& v : g_cnt_hist) v = 0;
            for (auto& v : g_src_hist) v = 0;
            bool cancelled = false;
            if (L < ids.size()) {
                const int T_sup = g_dev[0].W.T_sup;
                const int sub_chunk = std::min(g_dev[0].W.Tm, 512);
                if (T_sup > 0 && (n_gpu == 1 || dual()) && (int) (ids.size() - L) > 64) {
                    BigScratch bs;
                    const int T_pass = big_begin(M, g_dev[0].W, dev_cache(0), (int) (ids.size() - L), (int) ids.size(), bs);
                    for (size_t p0 = L; p0 < ids.size(); p0 += T_pass) {
                        if (g_cancel) {   // (a prompt half run: nothing of it is kept)
                            cancelled = true;
                            break;
                        }
                        const int n = (int) std::min<size_t>(T_pass, ids.size() - p0);
                        const bool last_c = p0 + n == ids.size();
                        std::vector<float> lo;
                        forward_prefill_sup(M, ids.data() + p0, n, (int) p0, sub_chunk,
                                            last_c ? OutMode::kLogits : OutMode::kNone, lo, nullptr, t_exp);
                        if (last_c) last = std::move(lo);
                        if (serve_maya) {   // (read, total, ms, tok/s, reused)
                            const double ms = (now() - t0) * 1e3;
                            std::printf("PP %zu %zu %.0f %.1f %d\n", p0 + n, ids.size(), ms,
                                        (p0 + n - L) / std::max(1e-3, ms / 1e3), (int) L);
                            std::fflush(stdout);
                        }
                    }
                    big_end(g_dev[0].W, dev_cache(0), bs);
                } else {
                    for (size_t p0 = L; p0 < ids.size(); p0 += chunk) {
                        if (g_cancel) {   // (a prompt half run: nothing of it is kept)
                            cancelled = true;
                            break;
                        }
                        const int n = (int) std::min<size_t>(chunk, ids.size() - p0);
                        const bool last_c = p0 + n == ids.size();
                        std::vector<float> lo;
                        forward(M, ids.data() + p0, n, (int) p0, false, last_c ? OutMode::kLogits : OutMode::kNone,
                                std::string(), lo, nullptr, t_exp);
                        if (last_c) last = std::move(lo);
                        if (drafter) {   // (the drafter's context for these positions, from their taps)
                            ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, n, (int) p0);
                            if (last_c) ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n - 1);
                        }
                    }
                }
            }
            if (cancelled) {
                reset();
                if (serve_maya) std::printf("DONE 0 %zu %.1f 0 cancel 0 0 %d\n", ids.size(), (now() - t0) * 1e3, cached);
                else std::printf("@END cancelled %zu %d %.3f %.3f\n", ids.size(), cached, now() - t0, 0.0);
                std::fflush(stdout);
                busy = false;
                continue;
            }
            if (L < ids.size()) {
                hist = ids;
                hist_sig = sig;
                Snap& sn = snaps[snap_next];
                snap_next = (snap_next + 1) % (int) snaps.size();
                sn.pos = (int) hist.size();
                snap_copy(sn, true);
            }
            const double t1 = now();
            if (getenv("IK2_STATS"))
                std::printf("prefill stats: %zu tokens run, %llu expert copies (%.1f GiB) in %.2f s; experts that ran 1 / 2 / 3-7 / 8+ "
                            "tokens: %llu / %llu / %llu / %llu\n",
                            ids.size() - (size_t) cached, (unsigned long long) (dev_cache(0).misses - miss0),
                            (dev_cache(0).bytes_up - bytes0) / 1073741824.0, t1 - t0,
                            (unsigned long long) g_cnt_hist[0], (unsigned long long) g_cnt_hist[1],
                            (unsigned long long) g_cnt_hist[2], (unsigned long long) g_cnt_hist[3]);
            if (getenv("IK2_STATS"))
                std::printf("prefill sources: used and cached %llu, used and prefetched %llu, used and copied after the "
                            "routing %llu, prefetched and unused %llu\n",
                            (unsigned long long) g_src_hist[0], (unsigned long long) g_src_hist[1],
                            (unsigned long long) g_src_hist[2], (unsigned long long) g_src_hist[3]);
            // softmax(logits / T) over the top_p mass of the 1000 largest of the (logit, token) pairs given (the
            // whole vocabulary, or the candidates the resident token path found on the device: every logit of the
            // highest bins that hold 1000 - the same 1000 either way; ties: the lower token first)
            auto pick_top = [&](std::vector<std::pair<float, int>>& cd) -> int {
                const int kk = std::min((int) cd.size(), top_k > 0 ? std::min(top_k, kCandK) : kCandK);
                std::partial_sort(cd.begin(), cd.begin() + kk, cd.end(),
                                  [](const std::pair<float, int>& a2, const std::pair<float, int>& b2) {
                                      return a2.first > b2.first || (a2.first == b2.first && a2.second < b2.second);
                                  });
                std::vector<double> pr(kk);
                double z = 0;
                for (int j = 0; j < kk; ++j) z += (pr[j] = std::exp((cd[j].first - cd[0].first) / temp));
                double acc = 0;
                int keep = kk;
                for (int j = 0; j < kk; ++j) {
                    acc += pr[j] / z;
                    if (acc >= top_p) {
                        keep = j + 1;
                        break;
                    }
                }
                double zk = 0;
                for (int j = 0; j < keep; ++j) zk += pr[j];
                double u = std::uniform_real_distribution<double>(0, zk)(rng);
                for (int j = 0; j < keep; ++j) {
                    u -= pr[j];
                    if (u <= 0) return cd[j].second;
                }
                return cd[keep - 1].second;
            };
            auto pick = [&](const float* r) -> int {
                if (temp <= 0) return (int) (std::max_element(r, r + c.n_vocab) - r);
                std::vector<std::pair<float, int>> cd(c.n_vocab);
                for (int v = 0; v < c.n_vocab; ++v) cd[v] = {r[v], v};
                return pick_top(cd);
            };
            auto pick_cand = [&]() -> int {
                std::vector<std::pair<float, int>> cd(g_cand.n);
                for (int i = 0; i < g_cand.n; ++i) cd[i] = {g_cand.val[i], g_cand.idx[i]};
                return pick_top(cd);
            };
            // IK2_CAND=0: sampling takes the whole vocabulary's logits from the device every token
            const bool use_cand = !(getenv("IK2_CAND") && !std::atoi(getenv("IK2_CAND")));
            bool cand = false;   // the last token's logits came as candidates (g_cand), not in `last`
            int pending_sample = -1;   // (a speculative window's last token, sampled already)
            const char* reason = "length";
            int n_out = 0;
            int greedy_tok = temp <= 0 && !last.empty() ? pick(last.data()) : -1;
            for (int i = 0; i < max_new; ++i) {
                if (g_cancel) {
                    reason = "cancelled";
                    break;
                }
                const int tok = temp <= 0             ? greedy_tok
                                : pending_sample >= 0 ? pending_sample
                                : cand                ? pick_cand()
                                                      : pick(last.data());
                pending_sample = -1;
                ++n_out;
                std::printf(serve_maya ? "T %d\n" : "@TOK %d\n", tok);
                std::fflush(stdout);
                if (std::find(stops.begin(), stops.end(), tok) != stops.end()) {
                    reason = "stop";
                    break;
                }
                if (i + 1 == max_new) break;
                if (req_spec) lk.append(tok);
                // (the step: a plain token, or a window of the drafter's or the prompt lookup's drafts)
                int B = 1, src = 0;
                int32_t ld[kBlockMax];
                if (req_spec) {
                    const int room = std::min({max_new - i - 1, ctx - (int) hist.size(), req_wmax});
                    const int k = g_lookup && room >= 2 ? lk.propose(std::min(room, kBlockMax) - 1, ld) : 0;
                    src = sctl.pick(room, k, B);
                    if (src == 1 && !drafter) src = 0, B = 1;
                }
                if (B >= 2) {
                    SampleCfg sc;
                    sc.temp = temp, sc.top_p = top_p, sc.top_k = top_k, sc.rng = &rng;
                    std::vector<int> next;
                    const double ts = now();
                    int given[kBlockMax];
                    for (int j = 0; j + 1 < B; ++j) given[j] = ld[j];
                    spec_step(M, drafter, tok, (int) hist.size(), B, sc, stops, next, src == 2 ? given : nullptr,
                              &req_draft_s);
                    CU(cudaStreamSynchronize(g_s));
                    const double tw = now() - ts;
                    req_verify_s += tw, ++req_rounds;
                    if (src == 2) sctl.took_lookup(B, (int) next.size() - 1, tw);
                    else sctl.took_spec(B, (int) next.size() - 1, tw);
                    hist.push_back(tok);
                    for (size_t j = 0; j + 1 < next.size(); ++j) {
                        std::printf(serve_maya ? "T %d\n" : "@TOK %d\n", next[j]);
                        hist.push_back(next[j]);
                        lk.append(next[j]);
                    }
                    if (serve_maya && req_rounds % 8 == 0) stat_line();
                    std::fflush(stdout);
                    n_out += (int) next.size() - 1;
                    i += (int) next.size() - 1;
                    if (temp <= 0) greedy_tok = next.back();
                    else pending_sample = next.back();
                    last.clear();
                    cand = false;
                    if (g_cancel) {
                        reason = "cancelled";
                        break;
                    }
                    continue;
                }
                const double tp0 = now();
                if (temp <= 0) {
                    std::vector<float> lo;
                    std::vector<int> ag;
                    forward(M, &tok, 1, (int) hist.size(), false, OutMode::kArgmax, std::string(), lo, &ag, t_exp);
                    greedy_tok = ag[0];
                    last.clear();
                } else {
                    std::vector<float> lo;
                    forward(M, &tok, 1, (int) hist.size(), false, use_cand ? OutMode::kCand : OutMode::kLogits,
                            std::string(), lo, nullptr, t_exp);
                    cand = use_cand && g_cand.n >= 0;
                    last = std::move(lo);
                }
                if (drafter && spec_on) {   // (the drafter's context for the token: its taps, row 0)
                    ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, 1, (int) hist.size());
                    ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, 0);
                }
                if (req_spec) {
                    CU(cudaStreamSynchronize(g_s));
                    sctl.took_plain(now() - tp0);
                }
                hist.push_back(tok);
            }
            if (spec_on && getenv("IK2_STATS"))
                std::fprintf(stderr, "speculative: %llu windows, %llu of %llu drafts accepted (the lookup's %llu of "
                             "%llu), %llu plain tokens\n", (unsigned long long) (sctl.steps + sctl.l_steps),
                             (unsigned long long) sctl.kept(), (unsigned long long) sctl.offered(),
                             (unsigned long long) sctl.l_accepted, (unsigned long long) sctl.l_drafted,
                             (unsigned long long) sctl.plain);
            if (serve_maya && req_rounds) stat_line();
            if (serve_maya)   // (DONE's 6th and 7th fields: this answer's drafts kept and offered)
                std::printf("DONE %d %zu %.1f %.1f %s %llu %llu %d\n", n_out, ids.size(), (t1 - t0) * 1e3,
                            (now() - t1) * 1e3, std::strcmp(reason, "cancelled") == 0 ? "cancel" : reason,
                            (unsigned long long) (sctl.kept() - kept0), (unsigned long long) (sctl.offered() - offered0),
                            cached);
            else
                std::printf("@END %s %zu %d %.3f %.3f\n", reason, ids.size(), cached, t1 - t0, now() - t1);
            std::fflush(stdout);
            busy = false;
        }
        g_offs.lane.reset();   // (the CPU lane's thread polls mapped memory: stopped before the driver goes)
        return 0;
    }
    if ((int) prompt.size() + gen > ctx) die("prompt + gen past --ctx");
    // the prompt
    std::vector<float> logits;
    double t_exp = 0;
    const double tp = now();
    const bool all = !dump.empty();
    int first_best = -1;
    const int T_sup = g_dev[0].W.T_sup;
    const int sub_chunk = std::min(g_dev[0].W.Tm, 512);
    if (T_sup > 0 && (n_gpu == 1 || dual()) && !drafter && !all && (int) prompt.size() > 64) {
        BigScratch bs;
        const int T_pass = big_begin(M, g_dev[0].W, dev_cache(0), (int) prompt.size(), (int) prompt.size(), bs);
        if (T_pass != T_sup) std::printf("prefill passes of %d tokens (scratch lent by the expert cache)\n", T_pass);
        for (int p0 = 0; p0 < (int) prompt.size(); p0 += T_pass) {
            const int n = std::min(T_pass, (int) prompt.size() - p0);
            const bool last = p0 + n == (int) prompt.size();
            const OutMode om = (last && gen > 0) ? OutMode::kArgmax : OutMode::kNone;
            std::vector<float> lo;
            std::vector<int> ag;
            forward_prefill_sup(M, prompt.data() + p0, n, p0, sub_chunk, om, lo, &ag, t_exp);
            if (last && !ag.empty()) first_best = ag[0];
        }
        big_end(g_dev[0].W, dev_cache(0), bs);
    } else {
        for (int p0 = 0; p0 < (int) prompt.size(); p0 += chunk) {
            const int n = std::min(chunk, (int) prompt.size() - p0);
            const bool last = p0 + n == (int) prompt.size();
            const OutMode om = all ? OutMode::kLogits : (!last ? OutMode::kNone : ((gen > 0 || drafter) ? OutMode::kArgmax : OutMode::kNone));
            std::vector<float> lo;
            std::vector<int> ag;
            forward(M, prompt.data() + p0, n, p0, all, om, all ? dump : std::string(), lo, &ag, t_exp);
            if (all) logits.insert(logits.end(), lo.begin(), lo.end());
            else if (last && !ag.empty()) first_best = ag[0];
            if (drafter) {
                ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, n, p0);
                if (last) ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n - 1);
            }
        }
    }
    const double prompt_s = now() - tp;
    std::printf("prompt: %zu tokens in %.2f s (%.1f tok/s); experts %.2f s of it, uploads %.2f s; cache %llu hits, "
                "%llu misses, %.1f GiB copied\n",
                prompt.size(), prompt_s, prompt.size() / prompt_s, t_exp, ec_sum([](ExpertCache& e) { return e.t_up; }),
                (unsigned long long) ec_sum([](ExpertCache& e) { return e.hits; }),
                (unsigned long long) ec_sum([](ExpertCache& e) { return e.misses; }),
                ec_sum([](ExpertCache& e) { return e.bytes_up; }) / 1073741824.0);
    std::fflush(stdout);
    if (all) {
        // log-probabilities of every prompt position
        std::vector<float> lp(logits.size());
        for (size_t t = 0; t < prompt.size(); ++t) {
            const float* r = logits.data() + t * c.n_vocab;
            double m = -1e30, s = 0;
            for (int v = 0; v < c.n_vocab; ++v) m = std::max(m, (double) r[v]);
            for (int v = 0; v < c.n_vocab; ++v) s += std::exp(r[v] - m);
            const double ls = m + std::log(s);
            for (int v = 0; v < c.n_vocab; ++v) lp[t * c.n_vocab + v] = (float) (r[v] - ls);
        }
        FILE* f = std::fopen((dump + "/logprobs.f32").c_str(), "wb");
        std::fwrite(lp.data(), 4, lp.size(), f);
        std::fclose(f);
        const float* r_last = logits.data() + (prompt.size() - 1) * c.n_vocab;
        first_best = (int) (std::max_element(r_last, r_last + c.n_vocab) - r_last);
        logits.clear();
    }
    // greedy generation
    g_pf = Prof{};
    std::vector<int> out;
    const double tg = now();
    const uint64_t h0 = (uint64_t) ec_sum([](ExpertCache& e) { return e.hits; }),
                   m0 = (uint64_t) ec_sum([](ExpertCache& e) { return e.misses; });
    int n_steps = 0;
    const int verify_check = getenv("IK2_VERIFY_CHECK") ? std::atoi(getenv("IK2_VERIFY_CHECK")) : 0;
    const int block_check = getenv("IK2_BLOCK_CHECK") ? std::atoi(getenv("IK2_BLOCK_CHECK")) : 0;
    double t_draft = 0, t_verify = 0;
    std::vector<int> acc_hist(ik2::draft::kBlock, 0);
    // IK2_FORCE=FILE (comma-separated ids): the tokens fed are these instead of the model's own picks, and how often
    // its pick agrees with the next one is counted - two builds then run the same tokens (a fair speed comparison)
    // and the agreement measures their numerics on real text
    std::vector<int> force;
    int agree = 0;
    if (getenv("IK2_FORCE")) {
        FILE* ff = std::fopen(getenv("IK2_FORCE"), "r");
        if (!ff) die("IK2_FORCE: cannot read the file");
        int v;
        while (std::fscanf(ff, "%d,", &v) == 1) force.push_back(v);
        std::fclose(ff);
    }
    // IK2_LOGITS_OUT=FILE: every generated token's logits [n_vocab] F32 appended there (with IK2_FORCE: two builds'
    // logits over the same tokens, to compare)
    FILE* lo = getenv("IK2_LOGITS_OUT") ? std::fopen(getenv("IK2_LOGITS_OUT"), "wb") : nullptr;
    const bool lookup_only = !drafter && g_lookup && g_fast && g_block;   // (then the drafting loop below)
    for (int i = 0, best = first_best; i < gen && !drafter && !lookup_only; ++i) {
        const int tok = i < (int) force.size() ? force[i] : best;
        if (i < (int) force.size()) agree += best == force[i];
        out.push_back(tok);
        std::printf("%d%s", tok, i + 1 < gen ? "," : "\n");
        std::fflush(stdout);
        if (i + 1 == gen) break;
        std::vector<int> ag;
        if (lo) logits.clear();
        forward(M, &tok, 1, (int) prompt.size() + i, false, lo ? OutMode::kLogits : OutMode::kArgmax, std::string(),
                logits, &ag, t_exp);
        if (lo) {
            std::fwrite(logits.data(), sizeof(float), (size_t) c.n_vocab, lo);
            best = (int) (std::max_element(logits.begin(), logits.begin() + c.n_vocab) - logits.begin());
        } else {
            best = ag[0];
        }
    }
    if (lo) std::fclose(lo);
    if (!force.empty())
        std::printf("forced: %d tokens fed, the model's pick agreed with the next one %d times (%.2f%%)\n",
                    (int) std::min<size_t>(force.size(), out.size()), agree,
                    100.0 * agree / std::max<size_t>(1, std::min<size_t>(force.size(), out.size())));

    SpecCtl ctl;
    if ((drafter || g_lookup) && g_fast && g_block) {
        // (the resident path: the controller picks plain tokens or a window of the drafter's or the prompt lookup's
        //  drafts; greedy, as the CLI's generation is)
        ctl.wmax = getenv("IK2_DRAFT_WIDTH") ? std::clamp(atoi(getenv("IK2_DRAFT_WIDTH")), 2, kBlockMax) : 5;
        ctl.adapt = !(getenv("IK2_DRAFT_ADAPT") && !atoi(getenv("IK2_DRAFT_ADAPT")));
        ctl.use_model = drafter != nullptr, ctl.use_lookup = g_lookup;
        strata::spec::SuffixDrafter lk;
        lk.append(prompt.data(), prompt.size());
        int anchor = first_best, P = (int) prompt.size();
        out.push_back(anchor);
        const SampleCfg sc;
        std::vector<int> next;
        while ((int) out.size() < gen && P < ctx) {
            lk.append(anchor);
            const int room = std::min(gen - (int) out.size(), ctx - P);
            int32_t ld[kBlockMax];
            const int k = g_lookup && room >= 2 ? lk.propose(std::min(room, kBlockMax) - 1, ld) : 0;
            int B = 1;
            int src = ctl.pick(room, k, B);
            if (src == 1 && !drafter) src = 0, B = 1;
            const double ts = now();
            if (B < 2) {
                std::vector<float> lo;
                std::vector<int> ag;
                forward(M, &anchor, 1, P, false, OutMode::kArgmax, std::string(), lo, &ag, t_exp);
                if (drafter) {
                    ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, 1, P);
                    ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, 0);
                }
                CU(cudaStreamSynchronize(g_s));
                ctl.took_plain(now() - ts);
                anchor = ag[0], P += 1;
                out.push_back(anchor);
                continue;
            }
            int given[kBlockMax];
            for (int j = 0; j + 1 < B; ++j) given[j] = ld[j];
            spec_step(M, drafter, anchor, P, B, sc, std::vector<int>{}, next, src == 2 ? given : nullptr, &t_draft);
            CU(cudaStreamSynchronize(g_s));
            if (src == 2) ctl.took_lookup(B, (int) next.size() - 1, now() - ts);
            else ctl.took_spec(B, (int) next.size() - 1, now() - ts);
            for (int j = 0; j + 1 < (int) next.size(); ++j) lk.append(next[j]);
            for (int j = 0; j < (int) next.size() && (int) out.size() < gen; ++j) out.push_back(next[j]);
            P += (int) next.size();
            anchor = next.back();
        }
        for (size_t j = 0; j < out.size(); ++j) std::printf("%d%s", out[j], j + 1 < out.size() ? "," : "\n");
    } else if (drafter) {
        // draft 7 after the anchor, verify the 8 in one call, keep the agreeing prefix and the target's next token
        float *raw = dalloc<float>(c.C), *anchor_h = dalloc<float>(c.C);
        auto argmax_row = [&](const float* r) { return (int) (std::max_element(r, r + c.n_vocab) - r); };
        int anchor = first_best, P = (int) prompt.size();
        out.push_back(anchor);
        // IK2_DRAFT_WIDTH: the tokens a step verifies, the anchor and width - 1 drafts (2 .. 8)
        const int width = getenv("IK2_DRAFT_WIDTH") ? std::clamp(atoi(getenv("IK2_DRAFT_WIDTH")), 2, ik2::draft::kBlock)
                                                    : ik2::draft::kBlock;
        const bool margin_check = getenv("IK2_MARGIN") != nullptr;
        while ((int) out.size() < gen && P < ctx) {
            const int B = std::min(width, ctx - P);
            const double t0 = now();
            embed_row(M, anchor, raw, anchor_h);
            int dr[ik2::draft::kBlock - 1];
            if (B > 1) ik2::draft::propose(drafter, anchor, anchor_h, P, draft_head, c.n_rows, c.n_vocab, dr, B - 1);
            const double t1 = now();
            int tk[ik2::draft::kBlock];
            tk[0] = anchor;
            for (int j = 1; j < B; ++j) tk[j] = dr[j - 1];
            g_rec = true;
            logits.clear();
            std::vector<int> ag;
            const bool need_logits = n_steps < verify_check || margin_check;
            const bool use_block = g_fast && g_block && !need_logits;
            if (use_block && n_steps < block_check) {
                // (IK2_BLOCK_CHECK=n: the first n windows through both paths, their logits compared; forward()'s
                //  then taken)
                std::vector<float> lb;
                fast_block(M, tk, B, P, OutMode::kLogits, lb, nullptr);
                block_rollback(0);
                forward(M, tk, B, P, true, OutMode::kLogits, std::string(), logits, nullptr, t_exp);
                double mx = 0, kl = 0;
                int dis = 0;
                for (int j = 0; j < B; ++j) {
                    const float* a = logits.data() + (size_t) j * c.n_vocab;
                    const float* b = lb.data() + (size_t) j * c.n_vocab;
                    double ma = -1e30, mb = -1e30, sa = 0, sb = 0;
                    for (int v = 0; v < c.n_vocab; ++v) ma = std::max(ma, (double) a[v]), mb = std::max(mb, (double) b[v]);
                    for (int v = 0; v < c.n_vocab; ++v) sa += std::exp(a[v] - ma), sb += std::exp(b[v] - mb);
                    for (int v = 0; v < c.n_vocab; ++v) {
                        mx = std::max(mx, (double) std::fabs(a[v] - b[v]));
                        const double pa = std::exp(a[v] - ma) / sa;
                        if (pa > 0) kl += pa * ((a[v] - ma - std::log(sa)) - (b[v] - mb - std::log(sb)));
                    }
                    dis += argmax_row(a) != argmax_row(b);
                }
                std::printf("block check, step %d (%d tokens at %d): max |dlogit| %.4f, KL %.2e a row, argmax differs in %d\n",
                            n_steps, B, P, mx, kl / B, dis);
                ag.clear();
                for (int j = 0; j < B; ++j) ag.push_back(argmax_row(logits.data() + (size_t) j * c.n_vocab));
            } else if (use_block)
                fast_block(M, tk, B, P, OutMode::kArgmax, logits, &ag);
            else
                forward(M, tk, B, P, true, need_logits ? OutMode::kLogits : OutMode::kArgmax, std::string(), logits,
                        &ag, t_exp);
            if (n_steps < verify_check) {
                // the same window again through each other path (the convs' history taken back first): the
                // logits' largest difference and the argmax disagreements against the grouped run
                const bool ge = g_group_exp, ga = g_group_attn;
                const int modes[3][2] = {{0, 0}, {1, 0}, {0, 1}};
                const char* names[3] = {"neither grouped", "experts grouped only", "attention grouped only"};
                for (int m = 0; m < 3; ++m) {
                    rollback(M, 0);
                    g_group_exp = modes[m][0], g_group_attn = modes[m][1];
                    std::vector<float> lb;
                    forward(M, tk, B, P, true, OutMode::kLogits, std::string(), lb, nullptr, t_exp);
                    double mx = 0;
                    int dis = 0;
                    for (int j = 0; j < B; ++j) {
                        const float* a = logits.data() + (size_t) j * c.n_vocab;
                        const float* b = lb.data() + (size_t) j * c.n_vocab;
                        for (int v = 0; v < c.n_vocab; ++v) mx = std::max(mx, (double) std::fabs(a[v] - b[v]));
                        dis += argmax_row(a) != argmax_row(b);
                    }
                    std::printf("verify check, step %d, grouped vs %s: max |dlogit| %.4f, argmax differs in %d of %d rows\n",
                                n_steps, names[m], mx, dis, B);
                }
                g_group_exp = ge, g_group_attn = ga;
                rollback(M, 0);
                logits.clear();
                forward(M, tk, B, P, true, OutMode::kLogits, std::string(), logits, nullptr, t_exp);
            }
            g_rec = false;
            int g[ik2::draft::kBlock];
            if (need_logits) {
                for (int j = 0; j < B; ++j) g[j] = argmax_row(logits.data() + (size_t) j * c.n_vocab);
            } else {
                for (int j = 0; j < B; ++j) g[j] = ag[j];
            }
            if (margin_check) {   // the target's top-2 margin at each verified position (near ties flip)
                for (int j = 0; j < B; ++j) {
                    const float* r = logits.data() + (size_t) j * c.n_vocab;
                    float b1 = -1e30f, b2 = -1e30f;
                    int i1 = -1, i2 = -1;
                    for (int v = 0; v < c.n_vocab; ++v) {
                        if (r[v] > b1) b2 = b1, i2 = i1, b1 = r[v], i1 = v;
                        else if (r[v] > b2) b2 = r[v], i2 = v;
                    }
                    if (b1 - b2 < 0.05f)
                        std::printf("near tie at position %d: %d %.4f, %d %.4f\n", P + j, i1, b1, i2, b2);
                }
            }
            int n = 0;
            while (n < B - 1 && dr[n] == g[n]) ++n;
            if (n + 1 < B) {
                if (use_block && n_steps >= block_check) block_rollback(n + 1);
                else rollback(M, n + 1);
            }
            ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, n + 1, P);
            ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n);
            for (int j = 0; j < n && (int) out.size() < gen; ++j) out.push_back(dr[j]);
            if ((int) out.size() < gen) out.push_back(g[n]);
            anchor = g[n];
            P += n + 1;
            ++n_steps;
            ++acc_hist[n];
            t_draft += t1 - t0;
            t_verify += now() - t1;
        }
        for (size_t j = 0; j < out.size(); ++j) std::printf("%d%s", out[j], j + 1 < out.size() ? "," : "\n");
    }
    const double gen_s = now() - tg;
    if ((drafter || g_lookup) && (ctl.steps || ctl.plain || ctl.l_steps)) {
        if (ctl.l_steps)
            std::printf("speculative: the prompt lookup %llu windows (%llu drafts, %llu accepted: %.2f tokens a window)\n",
                        (unsigned long long) ctl.l_steps, (unsigned long long) ctl.l_drafted,
                        (unsigned long long) ctl.l_accepted, 1.0 + (double) ctl.l_accepted / ctl.l_steps);
        std::printf("speculative: %llu windows (%llu drafts, %llu accepted: %.2f tokens a window; accepted drafts:",
                    (unsigned long long) ctl.steps, (unsigned long long) ctl.drafted, (unsigned long long) ctl.accepted,
                    ctl.steps ? 1.0 + (double) ctl.accepted / ctl.steps : 0.0);
        for (int j = 0; j < kBlockMax; ++j) std::printf(" %llu", (unsigned long long) ctl.hist[j]);
        std::printf("), %llu plain tokens; a window %.1f ms (widths 2 .. 6:", (unsigned long long) ctl.plain,
                    ctl.c[std::min(ctl.wmax, kBlockMax)] * 1e3);
        for (int b = 2; b <= 6; ++b) std::printf(" %.1f", ctl.c[b] * 1e3);
        std::printf("), a plain token %.1f ms\n", ctl.c_plain * 1e3);
    }
    if (g_window_rows)
        std::printf("window check: %llu of %llu rows the one-token path's to the bit\n",
                    (unsigned long long) g_window_same, (unsigned long long) g_window_rows);
    if (drafter && n_steps) {
        std::printf("speculative: %d steps, %.2f tokens a step (accepted drafts: ", n_steps,
                    (double) (out.size() - 1) / n_steps);
        for (int j = 0; j < ik2::draft::kBlock; ++j) std::printf("%d%s", acc_hist[j], j + 1 < ik2::draft::kBlock ? " " : "");
        std::printf("); drafting %.1f ms a step, verifying %.1f ms\n", t_draft * 1e3 / n_steps, t_verify * 1e3 / n_steps);
    }
    if (g_pool)
        std::printf("CPU lane: %d threads, %llu experts computed on the CPU (%.2f ms each at the end), %llu waited for\n",
                    g_pool->size(), (unsigned long long) g_n_cpu, g_t_cpu * 1e3, (unsigned long long) g_n_wait);
    if (g_prof)
        std::printf("profile (all calls): attention %.2f s, router %.2f s, uploads %.2f s, experts %.2f s, rest %.2f s\n",
                    g_pf.attn, g_pf.route, g_pf.upload, g_pf.experts, g_pf.rest);
    if (g_off) {   // (the offload path's counters since load: decoded tokens only)
        k::OffCount oc{};
        CU(cudaMemcpyAsync(&oc, g_offs.cnt, sizeof oc, cudaMemcpyDeviceToHost, g_s));
        CU(cudaStreamSynchronize(g_s));
        const double p = oc.picks ? 100.0 / (double) oc.picks : 0.0;
        std::printf("offload: %llu picks, %.1f%% in the VRAM tier, %.1f%% in staging, %.1f%% fetched (%.2f GiB, %.1f MiB "
                    "a token), %.1f%% on the CPU lane\n", oc.picks, oc.tier * p, oc.stage * p, oc.fetched * p,
                    oc.bytes / 1073741824.0,
                    oc.picks ? oc.bytes / 1048576.0 / ((double) oc.picks / (M.c.n_visit * M.c.K)) : 0.0, oc.cpu * p);
        if (g_lane)
            std::printf("lane: %.0f / %.0f us an expert as measured (NVFP4 / Q8_0)\n", g_offs.lane->us_expert(0),
                        g_offs.lane->us_expert(1));
    }
    if (out.size() > 1)
        std::printf("generation: %d tokens in %.2f s (%.2f tok/s); cache %llu hits, %llu misses in it\n",
                    (int) out.size() - 1, gen_s, (out.size() - 1) / gen_s, (unsigned long long) (ec_sum([](ExpertCache& e) { return e.hits; }) - h0),
                    (unsigned long long) (ec_sum([](ExpertCache& e) { return e.misses; }) - m0));
    std::printf("generated:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    std::fflush(stdout);
    if (g_trace) std::fclose(g_trace);
    pop_dump();
    g_offs.lane.reset();
    return 0;
}
