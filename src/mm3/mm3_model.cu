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
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include <unistd.h>

#include "mm3_cpu.hpp"
#include "mm3_dspark.hpp"
#include "mm3_exl3.hpp"
#include "mm3_geo.hpp"
#include "mm3_kernels.cuh"
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

struct Src {   // an expert's nine pieces in the mapped files: (w1, w3, w2) x (trellis, suh, svh)
    const uint8_t* p[9];
    size_t n[9];
};

struct Layer {
    int dev = 0;
    bool moe = false;
    float *in1 = nullptr, *post1 = nullptr, *qn1 = nullptr, *kn1 = nullptr, *iqn1 = nullptr, *ikn1 = nullptr;
    x3::Mat q, k, v, o, mg, mu, md, sg, su, sd;
    float *wiq = nullptr, *wik = nullptr, *gate = nullptr, *bias = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr, *ic = nullptr;
    k::KvI8 i8;   // (--kv-cache int8: the K / V caches instead of kc / vc)
    x3::ExpertLayout lay;
    size_t blob = 0;
    std::vector<Src> src;
    std::vector<int> res_slot;     // expert -> its slot in the resident region (-1: none)
    std::vector<int> res_ids;      // slot -> expert
    uint8_t* res_base = nullptr;   // the resident region: res_ids.size() blobs, blob bytes apart
    std::vector<const uint8_t*> host_blob;   // expert -> its assembled blob in the pinned pool (null: none)
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
    // scratch (chunk rows)
    float *h = nullptr, *xn = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *iq = nullptr, *ik = nullptr;
    float *bs = nullptr, *att = nullptr, *ao = nullptr, *g = nullptr, *u = nullptr, *logit = nullptr, *w = nullptr;
    float *y = nullptr, *sh = nullptr, *inv = nullptr, *out = nullptr, *lg = nullptr;   // (lg: the head's rows)
    uint16_t *xn16 = nullptr, *att16 = nullptr, *h16 = nullptr;
    int *pos = nullptr, *kept = nullptr, *top = nullptr, *rowidx = nullptr, *invrow = nullptr, *bounds = nullptr;
    size_t nblk = 0;
};

x3::Mat upload_mat(const Pack& pk, const std::string& pre, std::vector<void*>& owned, std::string& err) {
    x3::Mat m;
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

// a float vector on the device: the stored F32 / F16 / BF16 tensor (+ add to every value)
float* upload_f32(const Pack& pk, const std::string& name, float add, std::vector<void*>& owned, size_t& n,
                  std::string& err) {
    const Tensor* t = pk.get(name);
    if (!t) {
        err = name + ": missing";
        return nullptr;
    }
    const size_t es = t->dtype == "F32" ? 4 : 2;
    n = t->bytes / es;
    std::vector<float> h(n);
    const uint8_t* d = pk.data(*t);
    for (size_t i = 0; i < n; ++i) {
        float v;
        if (t->dtype == "F32") {
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
    cudaMemcpy(dp, h.data(), n * 4, cudaMemcpyHostToDevice);
    return (float*) dp;
}

}  // namespace

struct Model::Impl {
    Options o;
    Pack pk;
    Geo g;
    Stats st;
    std::vector<Layer> L;
    std::vector<Dev> D;
    std::vector<void*> owned;   // (cudaFree at exit, by device)
    std::vector<int> owned_dev;
    x3::Mat head;
    float* norm1 = nullptr;
    int head_dev = 0;
    std::vector<float> inv_freq;
    std::vector<float*> inv_dev;
    long long fwd = 0;
    int cur_pos = 0;                 // the last position of the chunk running
    std::vector<std::string> info;
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
    std::unique_ptr<Vision> vis;     // the vision tower (Options::vision)
    std::vector<Model::Image> imgs;  // the pictures the image tokens of the next forwards take their rows from
    float* taps = nullptr;           // its context's input: [taps][chunk][dim] F32 (the head's GPU)
    std::vector<int> tap_of;         // layer -> its tap (-1: none)

    ~Impl() {
        if (std::getenv("MM3_MISS_STATS") && !cpu_seen.empty()) {
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
        for (auto& d : D) {
            cudaSetDevice(d.id);
            if (d.cs) cudaStreamSynchronize(d.cs);
            for (cudaEvent_t e : d.slot_ev) cudaEventDestroy(e);
            if (d.cs) cudaStreamDestroy(d.cs);
            if (d.plan_h) cudaFreeHost(d.plan_h);
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
        p = (T*) v;
        return true;
    }
    void own_since(size_t n0, int dev) {
        while (owned_dev.size() < owned.size()) owned_dev.push_back(dev);
        (void) n0;
    }

    bool expert_layout(Layer& l, int li, std::string& err) {
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
            std::memcpy(dst + l.lay.suh[r], s.p[r * 3 + 1], s.n[r * 3 + 1]);
            std::memcpy(dst + l.lay.svh[r], s.p[r * 3 + 2], s.n[r * 3 + 2]);
        }
    }

    bool load(const Options& opt, std::string& err) {
        const double t0 = now();
        o = opt;
        if (!pk.load(o.pack, err) || !g.read(pk, err)) return false;
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
        // the layers over the GPUs by their free memory (the last GPU also takes the output head: 1 GB)
        {
            double total = 0;
            for (auto& d : D) total += (double) d.free0;
            double acc = 0;
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
                    const double mid = (li + 0.5) / g.n_layers * total;
                    while (di + 1 < D.size() && acc + (double) D[di].free0 < mid) acc += (double) D[di++].free0;
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
        const int T = o.chunk;
        const size_t nblk = (size_t) (o.max_ctx + g.blk - 1) / g.blk;
        // the dense part, caches and scratch, GPU by GPU
        for (size_t di = 0; di < D.size(); ++di) {
            Dev& d = D[di];
            CK(cudaSetDevice(d.id));
            const size_t n0 = owned.size();
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
            for (int li : d.layers) {
                Layer& l = L[(size_t) li];
                const std::string N = P + "layers." + std::to_string(li) + ".";
                const std::string AT = N + "self_attn.";
                size_t n = 0;
                std::vector<void*> ow;
                l.moe = g.moe[(size_t) li] != 0;
                l.in1 = upload_f32(pk, N + "input_layernorm.weight", 1.0f, ow, n, err);
                l.post1 = upload_f32(pk, N + "post_attention_layernorm.weight", 1.0f, ow, n, err);
                l.qn1 = upload_f32(pk, AT + "q_norm.weight", 1.0f, ow, n, err);
                l.kn1 = upload_f32(pk, AT + "k_norm.weight", 1.0f, ow, n, err);
                l.iqn1 = upload_f32(pk, AT + "index_q_norm.weight", 1.0f, ow, n, err);
                l.ikn1 = upload_f32(pk, AT + "index_k_norm.weight", 1.0f, ow, n, err);
                l.wiq = upload_f32(pk, AT + "index_q_proj.weight", 0.0f, ow, n, err);
                l.wik = upload_f32(pk, AT + "index_k_proj.weight", 0.0f, ow, n, err);
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
                const size_t cache = (size_t) o.max_ctx * g.n_kv * g.hd;
                if (o.kv_int8 ? !A(l.i8.kc, cache) || !A(l.i8.vc, cache) || !A(l.i8.ks, cache / g.hd) ||
                                    !A(l.i8.vs, cache / g.hd)
                              : !A(l.kc, cache) || !A(l.vc, cache))
                    return false;
                if (!A(l.ic, (size_t) o.max_ctx * g.ix_dim)) return false;
            }
            if (di + 1 == D.size()) {
                head_dev = (int) di;
                if (!A(d.lg, (size_t) kHeadRows * g.vocab)) return false;
                std::vector<void*> ow;
                size_t n = 0;
                norm1 = upload_f32(pk, P + "norm.weight", 1.0f, ow, n, err);
                if (!err.empty()) return false;
                head = upload_mat(pk, "language_model.lm_head", ow, err);
                if (!err.empty()) return false;
                for (void* x : ow) {
                    owned.push_back(x);
                    owned_dev.push_back(dv);
                }
            }
            own_since(n0, dv);
            if (d.slot_bytes > 0) {
                d.n_slots = o.staging;
                if (!A(d.stg, (size_t) d.n_slots * d.slot_bytes)) return false;
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
            if (!A(d.plan_d, 16)) return false;
            CK(cudaMallocHost(&d.xh_h, (size_t) (2 * g.dim + 4) * sizeof(float)));
            if (!A(d.cpu_d, (size_t) g.dim + 4)) return false;
            CK(cudaMallocHost(&d.win_rb, win_rb_bytes()));
            CK(cudaMallocHost(&d.win_plan_h, win_plan_bytes()));
            if (!A(d.win_plan_d, win_plan_bytes())) return false;
            CK(cudaHostAlloc(&d.win_cpu, ((size_t) kWin * g.dim + kWin + 4) * 4, cudaHostAllocMapped | cudaHostAllocPortable));
            if (!A(d.win_h, (size_t) kWin * 8 * g.ff) || !A(d.win_part, (size_t) kWin * 8 * g.dim)) return false;
        }
        // the DSpark draft on the head's GPU, before the experts take the memory
        tap_of.assign((size_t) g.n_layers, -1);
        if (o.dspark) {
            Dev& hd = D[(size_t) head_dev];
            CK(cudaSetDevice(hd.id));
            draftm = std::make_unique<Draft>();
            const Tensor* et = pk.get(P + "embed_tokens.weight");
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
            if (!dalloc(hd.id, taps, tl.size() * (size_t) o.chunk * g.dim, err)) return false;
            info.push_back("DSpark draft: " + std::to_string(draftm->bytes() >> 20) + " MB on CUDA" +
                           std::to_string(hd.id) + ", block " + std::to_string(draftm->geo().block) + ", taps " +
                           std::to_string(tl.size()));
        }
        // the vision tower, also before the experts take the memory
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
            if (!vis->load(pk, D[(size_t) o.vision_gpu].id, err)) return false;
            info.push_back("vision tower: " + std::to_string(vis->bytes() >> 20) + " MB on CUDA" +
                           std::to_string(vis->device()) + ", up to " + std::to_string(vis->geo().max_pixels) +
                           " pixels a picture");
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
                    for (int b = 0; b < 2; ++b) {
                        cudaMallocHost(&pin[b], mx);
                        cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming);
                        cudaEventRecord(ev[b], dp->s);
                    }
                    int n = 0;
                    for (int li : dp->layers) {
                        const Layer& l = L[(size_t) li];
                        for (size_t s = 0; s < l.res_ids.size(); ++s, ++n) {
                            const int b = n & 1;
                            cudaEventSynchronize(ev[b]);
                            assemble(l, l.res_ids[s], pin[b]);
                            cudaMemcpyAsync(l.res_base + s * l.blob, pin[b], l.blob, cudaMemcpyHostToDevice, dp->s);
                            cudaEventRecord(ev[b], dp->s);
                        }
                    }
                    if (cudaStreamSynchronize(dp->s) != cudaSuccess) {
                        std::lock_guard<std::mutex> lk(em);
                        ++bad;
                    }
                    for (int b = 0; b < 2; ++b) {
                        cudaFreeHost(pin[b]);
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
        // the pinned pool: every expert not in VRAM assembled once (a staging copy is then one DMA)
        if (o.pin_pool) {
            size_t total = 0;
            for (const Layer& l : L)
                if (l.moe) total += (size_t) (g.n_exp - (int) l.res_ids.size()) * l.blob;
            double avail = 0;   // MemAvailable (the page cache counts: the mapped files can be read again)
            if (std::FILE* mi = std::fopen("/proc/meminfo", "r")) {
                char line[256];
                while (std::fgets(line, sizeof line, mi))
                    if (std::strncmp(line, "MemAvailable:", 13) == 0) avail = std::atof(line + 13) * 1024.0;
                std::fclose(mi);
            }
            if (total > 0 && (double) total < avail - 24.0 * (1u << 30)) {
                CK(cudaSetDevice(D[0].id));
                if (cudaHostAlloc((void**) &pool, total, cudaHostAllocPortable) != cudaSuccess) {
                    pool = nullptr;
                    cudaGetLastError();
                    info.push_back("pinned pool: " + std::to_string(total >> 30) + " GB not allocated");
                } else {
                    std::vector<std::pair<int, int>> todo;
                    size_t off = 0;
                    for (size_t li = 0; li < L.size(); ++li) {
                        Layer& l = L[li];
                        if (!l.moe) continue;
                        l.host_blob.assign((size_t) g.n_exp, nullptr);
                        for (int e = 0; e < g.n_exp; ++e)
                            if (l.res_slot[(size_t) e] < 0) {
                                l.host_blob[(size_t) e] = pool + off;
                                off += l.blob;
                                todo.push_back({(int) li, e});
                            }
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
                    st.pinned_gb = (double) total / (1u << 30);
                    st.pooled = (long) todo.size();
                }
            } else if (total > 0) {
                info.push_back("pinned pool: " + std::to_string(total >> 30) + " GB does not fit the free RAM");
            }
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
        }
        if (o.kv_int8) info.push_back("KV cache: INT8 (a scale a row of 128)");
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
                        g.eps, l.kc, l.vc, d.s, 0, l.i8);
        const int nq = g.ix_heads * g.ix_dim;
        k::rows_dot(l.wiq, nq, g.dim, d.xn, T, d.iq, d.s);
        k::rows_dot(l.wik, g.ix_dim, g.dim, d.xn, T, d.ik, d.s);
        k::index_norm_rope(d.iq, d.ik, l.iqn1, l.ikn1, d.pos, T, g.ix_heads, g.ix_dim, g.n_rot, inv_dev[(size_t) di],
                           g.eps, l.ic, d.s);
        const int nblk = cur_pos / g.blk + 1;
        k::block_scores(d.iq, l.ic, d.pos, T, g.ix_heads, nblk, g.blk, d.bs, d.s);
        k::select_blocks(d.bs, d.pos, T, g.ix_heads, nblk, g.blk, g.topk, d.kept, d.s);
        k::sparse_attn(d.q, l.kc, l.vc, d.kept, d.pos, T, g.n_head, g.n_kv, g.topk, g.blk, 1.0f / std::sqrt((float) g.hd),
                       d.att, d.s, l.i8);
        const x3::MvJob oj = job(l.o, d.att, Q, d.ao, g.dim);
        if (!x3::mv(&oj, 1, d.s)) {
            err = "mv o (window)";
            return false;
        }
        k::add(d.h, d.ao, (size_t) T * g.dim, d.s);
        k::rms(d.h, l.post1, T, g.dim, g.eps, d.xn, nullptr, d.s);
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
                if (lane && on_cpu(l, d, x)) {
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
        const size_t off_w = (size_t) kWin * 8 * 8, off_gp = off_w + (size_t) kWin * 8 * 4,
                     off_gs = off_gp + (size_t) kWin * 8 * 8, off_ge = off_gs + ((size_t) kWin * 8 + 1) * 4,
                     off_ng = off_ge + (size_t) kWin * 8 * 4;
        float* cpu_rows = d.win_cpu;
        int* cpu_flags = (int*) (d.win_cpu + (size_t) kWin * g.dim);
        unsigned* wait_word = (unsigned*) (cpu_flags + kWin);
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
            lane->run_multi(cpu, users, rx, T, cpu_rows, &order);
            for (int t = 0; t < T; ++t) cpu_flags[t] = 0;
            for (const auto& u : users)
                for (const auto& tu : u) cpu_flags[tu.first] = 1;
            st.t_lane += now() - tl;
            st.cpu_experts += (long) cpu.size();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile unsigned*) wait_word = d.win_wait;
        }
        k::add(d.h, d.out, (size_t) T * g.dim, d.s);
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
                        g.eps, l.kc, l.vc, d.s, 0, l.i8);
        const float one = 1.0f, zero = 0.0f;
        const int nq = g.ix_heads * g.ix_dim;
        k::rows_dot(l.wiq, nq, g.dim, d.xn, 1, d.iq, d.s);
        k::rows_dot(l.wik, g.ix_dim, g.dim, d.xn, 1, d.ik, d.s);
        k::index_norm_rope(d.iq, d.ik, l.iqn1, l.ikn1, d.pos, 1, g.ix_heads, g.ix_dim, g.n_rot, inv_dev[(size_t) di],
                           g.eps, l.ic, d.s);
        const int nblk = cur_pos / g.blk + 1;
        k::block_scores(d.iq, l.ic, d.pos, 1, g.ix_heads, nblk, g.blk, d.bs, d.s);
        k::select_blocks(d.bs, d.pos, 1, g.ix_heads, nblk, g.blk, g.topk, d.kept, d.s);
        k::sparse_attn(d.q, l.kc, l.vc, d.kept, d.pos, 1, g.n_head, g.n_kv, g.topk, g.blk, 1.0f / std::sqrt((float) g.hd),
                       d.att, d.s, l.i8);
        const x3::MvJob oj = job(l.o, d.att, d.ao);
        if (!x3::mv(&oj, 1, d.s)) {
            err = "mv o";
            return false;
        }
        k::add(d.h, d.ao, (size_t) g.dim, d.s);
        k::rms(d.h, l.post1, 1, g.dim, g.eps, d.xn, nullptr, d.s);
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
        for (int i = 0; i < g.k_exp; ++i) {
            pw[i] = w[i] * g.scale;
            if (lane && on_cpu(l, d, top[i])) {   // (not in VRAM nor in a staging slot: the CPU lane's)
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
                lane->run(cpu, d.xh_h, part);
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
        // attention
        k::rms(d.h, l.in1, T, g.dim, g.eps, d.xn, d.xn16, d.s);
        x3::gemm_rows(l.q, d.xn16, g.dim, T, d.q, g.n_head * g.hd, 0.0f, d.s);
        x3::gemm_rows(l.k, d.xn16, g.dim, T, d.k, g.n_kv * g.hd, 0.0f, d.s);
        x3::gemm_rows(l.v, d.xn16, g.dim, T, d.v, g.n_kv * g.hd, 0.0f, d.s);
        k::qk_norm_rope(d.q, d.k, d.v, l.qn1, l.kn1, d.pos, T, g.n_head, g.n_kv, g.hd, g.n_rot, inv_dev[(size_t) di],
                        g.eps, l.kc, l.vc, d.s, 0, l.i8);
        const float one = 1.0f, zero = 0.0f;
        const int nq = g.ix_heads * g.ix_dim;
        CB(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, nq, T, g.dim, &one, l.wiq, g.dim, d.xn, g.dim, &zero, d.iq, nq));
        CB(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, g.ix_dim, T, g.dim, &one, l.wik, g.dim, d.xn, g.dim, &zero,
                       d.ik, g.ix_dim));
        k::index_norm_rope(d.iq, d.ik, l.iqn1, l.ikn1, d.pos, T, g.ix_heads, g.ix_dim, g.n_rot, inv_dev[(size_t) di],
                           g.eps, l.ic, d.s);
        const int nblk = pmax / g.blk + 1;
        k::block_scores(d.iq, l.ic, d.pos, T, g.ix_heads, nblk, g.blk, d.bs, d.s);
        k::select_blocks(d.bs, d.pos, T, g.ix_heads, nblk, g.blk, g.topk, d.kept, d.s);
        if (const char* dp = std::getenv("MM3_DUMP_KEPT")) {   // (debugging: the kept blocks, int32 records)
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
                       d.att, d.s, l.i8);
        k::to_f16(d.att, T, g.n_head * g.hd, g.n_head * g.hd, d.att16, g.n_head * g.hd, d.s);
        x3::gemm_rows(l.o, d.att16, g.n_head * g.hd, T, d.h, g.dim, 1.0f, d.s);   // h += attn Wo
        // MLP
        k::rms(d.h, l.post1, T, g.dim, g.eps, d.xn, d.xn16, d.s);
        if (!l.moe) {
            x3::gemm_rows(l.mg, d.xn16, g.dim, T, d.g, g.ff_dense, 0.0f, d.s);
            x3::gemm_rows(l.mu, d.xn16, g.dim, T, d.u, g.ff_dense, 0.0f, d.s);
            k::swigluoai(d.g, d.u, (size_t) T * g.ff_dense, g.alpha, g.limit, nullptr, d.h16, d.s);
            x3::gemm_rows(l.md, d.h16, g.ff_dense, T, d.h, g.dim, 1.0f, d.s);   // h += mlp
            return true;
        }
        CB(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, g.n_exp, T, g.dim, &one, l.gate, g.dim, d.xn, g.dim, &zero,
                       d.logit, g.n_exp));
        k::route(d.logit, l.bias, T, g.n_exp, g.k_exp, d.top, d.w, d.s);
        // the shared expert into sh
        x3::gemm_rows(l.sg, d.xn16, g.dim, T, d.g, g.ff_shared, 0.0f, d.s);
        x3::gemm_rows(l.su, d.xn16, g.dim, T, d.u, g.ff_shared, 0.0f, d.s);
        k::swigluoai(d.g, d.u, (size_t) T * g.ff_shared, g.alpha, g.limit, nullptr, d.h16, d.s);
        x3::gemm_rows(l.sd, d.h16, g.ff_shared, T, d.sh, g.dim, 0.0f, d.s);
        std::vector<int> top((size_t) T * g.k_exp);
        CK(cudaMemcpyAsync(top.data(), d.top, top.size() * 4, cudaMemcpyDeviceToHost, d.s));
        CK(cudaStreamSynchronize(d.s));
        for (int e : top) ++prof[(size_t) li * g.n_exp + e];
        st.profiled += (long) top.size();
        if (!experts(l, d, T, top, err)) return false;
        k::moe_combine(d.h, d.y, d.invrow, d.w, d.sh, T, g.k_exp, g.dim, g.scale, d.out, d.s);
        std::swap(d.h, d.out);
        return true;
    }

    bool chunk(const int* ids, int T, int pos0, int want, float* logits, std::string& err) {
        const double tc = now();
        const bool ok = chunk_(ids, T, pos0, want, logits, err);
        if (T == 1) st.t_token += now() - tc;
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
        // the embeddings (BF16 rows from the file) -> F32 on the first layer's GPU
        const Tensor* emb = pk.get(P + "embed_tokens.weight");
        std::vector<float> h((size_t) T * g.dim);
        const uint16_t* e16 = (const uint16_t*) pk.data(*emb);
        for (int t = 0; t < T; ++t) {
            if (ids[t] < 0 || ids[t] >= g.vocab) {
                err = "token id " + std::to_string(ids[t]) + " outside the vocabulary";
                return false;
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
        std::vector<int> pos((size_t) T);
        for (int t = 0; t < T; ++t) pos[(size_t) t] = pos0 + t;
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            CK(cudaMemcpyAsync(d.pos, pos.data(), (size_t) T * 4, cudaMemcpyHostToDevice, d.s));
        }
        int cur = L[0].dev;
        CK(cudaSetDevice(D[(size_t) cur].id));
        CK(cudaMemcpyAsync(D[(size_t) cur].h, h.data(), h.size() * 4, cudaMemcpyHostToDevice, D[(size_t) cur].s));
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
            if (draftm && tap_of[(size_t) li] >= 0) {   // (the draft's context: this layer's output rows)
                Dev& a = D[(size_t) dv];
                Dev& hd = D[(size_t) head_dev];
                float* dst = taps + (size_t) tap_of[(size_t) li] * T * g.dim;
                if (dv == head_dev)
                    CK(cudaMemcpyAsync(dst, a.h, (size_t) T * g.dim * 4, cudaMemcpyDeviceToDevice, a.s));
                else
                    CK(cudaMemcpyPeerAsync(dst, hd.id, a.h, a.id, (size_t) T * g.dim * 4, a.s));
            }
            if (const char* dp = std::getenv("MM3_DUMP_HS")) {   // (debugging: every layer's residual stream, F32 rows)
                std::vector<float> hh((size_t) T * g.dim);
                CK(cudaMemcpyAsync(hh.data(), D[(size_t) dv].h, hh.size() * 4, cudaMemcpyDeviceToHost, D[(size_t) dv].s));
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
        if (want > 0) {
            Dev& a = D[(size_t) cur];
            Dev& hd = D[(size_t) head_dev];
            CK(cudaSetDevice(a.id));
            CK(cudaStreamSynchronize(a.s));
            CK(cudaSetDevice(hd.id));
            if (cur != head_dev)
                CK(cudaMemcpyPeerAsync(hd.h, hd.id, a.h, a.id, (size_t) T * g.dim * 4, hd.s));
            const float* last = hd.h + (size_t) (T - want) * g.dim;
            k::rms(last, norm1, want, g.dim, g.eps, hd.xn, hd.xn16, hd.s);
            if (want <= kWin && o.fast_decode) {   // (a token's and a window's rows: mv, the same bits for each row)
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
                CK(cudaMemcpyAsync(logits, hd.lg, (size_t) want * g.vocab * 4, cudaMemcpyDeviceToHost, hd.s));
                CK(cudaStreamSynchronize(hd.s));
                want = 0;
            }
            // the head's output kHeadRows rows at a time
            for (int r0 = 0; r0 < want; r0 += kHeadRows) {
                const int nr = std::min<int>(kHeadRows, want - r0);
                x3::gemm_rows(head, hd.xn16 + (size_t) r0 * g.dim, g.dim, nr, hd.lg, g.vocab, 0.0f, hd.s);
                CK(cudaMemcpyAsync(logits + (size_t) r0 * g.vocab, hd.lg, (size_t) nr * g.vocab * 4,
                                   cudaMemcpyDeviceToHost, hd.s));
                CK(cudaStreamSynchronize(hd.s));
            }
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
        }
        return true;
    }
};

Model::Model() : p(new Impl) {}
Model::~Model() {
    delete tpe;
    delete p;
}
bool Model::load(const Options& o, std::string& err) {
    if (o.tp > 0) {
        tpe = new TpEngine;
        return tpe->load(o, err);
    }
    return p->load(o, err);
}
int Model::vocab() const { return tpe ? tpe->vocab() : p->g.vocab; }
int Model::max_ctx() const { return tpe ? tpe->max_ctx() : p->o.max_ctx; }
int Model::bos() const { return tpe ? tpe->bos() : p->g.bos; }
int Model::eos() const { return tpe ? tpe->eos() : p->g.eos; }
const Stats& Model::stats() const { return tpe ? tpe->stats() : p->st; }
std::vector<std::string> Model::info_lines() const { return tpe ? tpe->info_lines() : p->info; }

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

bool Model::save_profile(const std::string& path, std::string& err) const {
    if (tpe) return tpe->save_profile(path, err);
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(p->prof.data(), 4, p->prof.size(), f) != p->prof.size()) {
        if (f) std::fclose(f);
        err = path + ": cannot write";
        return false;
    }
    std::fclose(f);
    return true;
}

bool Model::forward(const int* ids, int n, int pos0, int want, float* logits_out, std::string& err) {
    if (tpe) return tpe->forward(ids, n, pos0, want, logits_out, err);
    want = std::min(want, n);
    for (int a = 0; a < n; a += p->o.chunk) {
        const int T = std::min(p->o.chunk, n - a);
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
    return true;
}

}  // namespace mm3
