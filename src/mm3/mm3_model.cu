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
#include <mutex>
#include <thread>

#include "mm3_exl3.hpp"
#include "mm3_kernels.cuh"

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
size_t up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const std::string P = "language_model.model.";

struct Geo {
    int dim, n_layers, n_head, n_kv, hd, n_rot, vocab, n_exp, k_exp, ff, ff_dense, ff_shared;
    float theta, eps, alpha, limit, scale;
    std::vector<int> moe, sparse;
    int ix_heads, ix_dim, blk, topk, local;
    int bos, eos;

    bool read(const Pack& p, std::string& err) {
        dim = (int) p.num("hidden_size");
        n_layers = (int) p.num("num_hidden_layers");
        n_head = (int) p.num("num_attention_heads");
        n_kv = (int) p.num("num_key_value_heads");
        hd = (int) p.num("head_dim");
        n_rot = (int) p.num("rotary_dim", hd * p.num("partial_rotary_factor", 1.0));
        vocab = (int) p.num("vocab_size");
        n_exp = (int) p.num("num_local_experts");
        k_exp = (int) p.num("num_experts_per_tok");
        ff = (int) p.num("intermediate_size");
        ff_dense = (int) p.num("dense_intermediate_size");
        ff_shared = (int) p.num("shared_intermediate_size");
        theta = (float) p.num("rope_theta");
        eps = (float) p.num("rms_norm_eps");
        alpha = (float) p.num("swiglu_alpha");
        limit = (float) p.num("swiglu_limit");
        scale = (float) p.num("routed_scaling_factor");
        moe = p.ints("moe_layer_freq");
        sparse = p.ints("sparse_attention_config.sparse_attention_freq");
        ix_heads = (int) p.num("sparse_attention_config.sparse_num_index_heads");
        ix_dim = (int) p.num("sparse_attention_config.sparse_index_dim");
        blk = (int) p.num("sparse_attention_config.sparse_block_size");
        topk = (int) p.num("sparse_attention_config.sparse_topk_blocks");
        local = (int) p.num("sparse_attention_config.sparse_local_block");
        bos = (int) p.num("bos_token_id");
        eos = (int) p.num("eos_token_id");
        const auto no = [&](const std::string& what) {
            err = "geometry not covered: " + what;
            return false;
        };
        if (hd != 128 || n_rot % 2 || n_rot > hd) return no("head_dim " + std::to_string(hd));
        if (n_head != 16 * n_kv) return no("query heads per KV head " + std::to_string(n_head / std::max(1, n_kv)));
        if (ix_heads != n_kv || ix_dim != 128 || ix_heads != 4) return no("indexer heads / dim");
        if (blk != 128 || local != 1 || topk < 1 || topk > 64) return no("sparse blocks (block 128, local 1)");
        if (p.num("sparse_attention_config.sparse_init_block", 0) != 0) return no("init blocks");
        if ((int) moe.size() != n_layers || (int) sparse.size() != n_layers) return no("per-layer lists");
        for (int s : sparse)
            if (!s) return no("a full-attention layer (every layer of M3.1 is sparse)");
        if (std::fabs(alpha - x3::kSwigluAlpha) > 1e-6f) return no("swiglu_alpha " + std::to_string(alpha));
        if (k_exp > 8 || n_exp > 256) return no("experts");
        if (dim % 128 || ff % 128 || ff_dense % 128 || ff_shared % 128) return no("matrix sides");
        return true;
    }
};

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
    x3::ExpertLayout lay;
    size_t blob = 0;
    std::vector<Src> src;
    std::vector<int> res_slot;     // expert -> its slot in the resident region (-1: none)
    std::vector<int> res_ids;      // slot -> expert
    uint8_t* res_base = nullptr;   // the resident region: res_ids.size() blobs, blob bytes apart
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
    uint8_t* pin = nullptr;              // a pinned blob for the copies
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
    std::vector<std::string> info;

    ~Impl() {
        for (size_t i = 0; i < owned.size(); ++i) {
            cudaSetDevice(owned_dev[i]);
            cudaFree(owned[i]);
        }
        for (auto& d : D) {
            cudaSetDevice(d.id);
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
            for (int li = 0; li < g.n_layers; ++li) {
                const double mid = (li + 0.5) / g.n_layers * total;
                while (di + 1 < D.size() && acc + (double) D[di].free0 < mid) acc += (double) D[di++].free0;
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
                if (!A(l.kc, cache) || !A(l.vc, cache) || !A(l.ic, (size_t) o.max_ctx * g.ix_dim)) return false;
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
                CK(cudaMallocHost(&d.pin, d.slot_bytes));
            }
        }
        // the routed experts kept in VRAM: round robin over each GPU's layers (expert 0 of every layer, then 1, ...)
        // while the GPU keeps reserve_gb free
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const size_t reserve = (size_t) (o.reserve_gb * (1u << 30));
            size_t budget = fr > reserve ? fr - reserve : 0;
            std::vector<int> moe_layers;
            for (int li : d.layers)
                if (L[(size_t) li].moe) moe_layers.push_back(li);
            std::vector<int> nres(moe_layers.size(), 0);
            const int cap = o.resident_cap < 0 ? g.n_exp : std::min(o.resident_cap, g.n_exp);
            for (bool grew = true; grew;) {
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
                for (int e = 0; e < nres[i]; ++e) {
                    l.res_slot[(size_t) e] = e;
                    l.res_ids.push_back(e);
                }
                st.resident += nres[i];
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
        st.load_s = now() - t0;
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
            for (int e : fresh) {
                int best = -1;
                for (int s = 0; s < d.n_slots; ++s)
                    if (se[(size_t) s] < 0 && (best < 0 || d.slot_used[(size_t) s] < d.slot_used[(size_t) best]))
                        best = s;
                se[(size_t) best] = e;
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
                        CK(cudaStreamSynchronize(d.s));   // (the pinned blob and the slot are free)
                        assemble(l, e, d.pin);
                        CK(cudaMemcpyAsync(d.stg + (size_t) s * d.slot_bytes, d.pin, l.blob, cudaMemcpyHostToDevice,
                                           d.s));
                        d.slot_owner[(size_t) s] = lid * 1024 + e;
                        ++st.staged;
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
        // attention
        k::rms(d.h, l.in1, T, g.dim, g.eps, d.xn, d.xn16, d.s);
        x3::gemm_rows(l.q, d.xn16, g.dim, T, d.q, g.n_head * g.hd, 0.0f, d.s);
        x3::gemm_rows(l.k, d.xn16, g.dim, T, d.k, g.n_kv * g.hd, 0.0f, d.s);
        x3::gemm_rows(l.v, d.xn16, g.dim, T, d.v, g.n_kv * g.hd, 0.0f, d.s);
        k::qk_norm_rope(d.q, d.k, d.v, l.qn1, l.kn1, d.pos, T, g.n_head, g.n_kv, g.hd, g.n_rot, inv_dev[(size_t) di],
                        g.eps, l.kc, l.vc, d.s);
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
        k::sparse_attn(d.q, l.kc, l.vc, d.kept, d.pos, T, g.n_head, g.n_kv, g.topk, g.blk, 1.0f / std::sqrt((float) g.hd),
                       d.att, d.s);
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
        if (!experts(l, d, T, top, err)) return false;
        k::moe_combine(d.h, d.y, d.invrow, d.w, d.sh, T, g.k_exp, g.dim, g.scale, d.out, d.s);
        std::swap(d.h, d.out);
        return true;
    }

    bool chunk(const int* ids, int T, int pos0, int want, float* logits, std::string& err) {
        ++fwd;
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
        return true;
    }
};

Model::Model() : p(new Impl) {}
Model::~Model() { delete p; }
bool Model::load(const Options& o, std::string& err) { return p->load(o, err); }
int Model::vocab() const { return p->g.vocab; }
int Model::max_ctx() const { return p->o.max_ctx; }
int Model::bos() const { return p->g.bos; }
int Model::eos() const { return p->g.eos; }
const Stats& Model::stats() const { return p->st; }
std::vector<std::string> Model::info_lines() const { return p->info; }

bool Model::forward(const int* ids, int n, int pos0, int want, float* logits_out, std::string& err) {
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
