// src/mm3/mm3_dspark.cu - maya-mm3's DSpark draft (mm3_dspark.hpp).
#include "mm3_dspark.hpp"

#include <functional>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mm3_kernels.cuh"

namespace mm3 {
namespace x3 = strata::mm3::exl3;

#define DK(call)                                                                       \
    do {                                                                               \
        const cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                       \
            err = std::string("dspark: ") + #call + ": " + cudaGetErrorString(e_);    \
            return false;                                                              \
        }                                                                              \
    } while (0)
#define DB(call)                                                                       \
    do {                                                                               \
        const cublasStatus_t b_ = (call);                                              \
        if (b_ != CUBLAS_STATUS_SUCCESS) {                                             \
            err = std::string("dspark: ") + #call + ": cublas " + std::to_string((int) b_);  \
            return false;                                                              \
        }                                                                              \
    } while (0)

namespace {
const std::string DP = "dspark.language_model.model.dspark.";
// MAYA_MM3_PROF=2: a proposal's parts timed (the host waits at each) and printed at the end
struct DraftProf {
    bool on = [] {
        const char* v = std::getenv("MAYA_MM3_PROF");
        return v != nullptr && std::atoi(v) == 2;
    }();
    double ms[4] = {};
    long n = 0;
    double last = 0;
    static double now() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void start(cudaStream_t s) {
        if (!on) return;
        cudaStreamSynchronize(s);
        last = now();
    }
    void mark(cudaStream_t s, int part) {
        if (!on) return;
        cudaStreamSynchronize(s);
        const double t = now();
        ms[part] += 1e3 * (t - last);
        last = t;
    }
    ~DraftProf() {
        if (on && n > 0)
            std::fprintf(stderr, "MM3 PROF draft: %ld proposals, layers %.2f head %.2f markov %.2f ms a proposal\n", n,
                         ms[0] / n, ms[1] / n, ms[2] / n);
    }
};
}

struct Draft::Impl {
    DraftGeo g;
    int dev = 0, ring = 0, max_rows = 0;
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    const x3::Mat* head = nullptr;
    const uint16_t* emb = nullptr;
    std::function<void(int, uint16_t*)> emb_row;   // (set_emb_rows: a GGUF's table, dequantized a row at a time)
    std::vector<void*> owned;
    // row id of the target's token table as BF16 into dst (D values)
    void row(int id, uint16_t* dst) const {
        if (emb_row) emb_row(id, dst);
        else std::memcpy(dst, emb + (size_t) id * g.dim, (size_t) g.dim * 2);
    }
    size_t bytes = 0;
    struct MX {   // an MXFP8 matrix [N][K] as stored: E4M3 weights, a UE8M0 scale every 32
        const uint8_t *w = nullptr, *s = nullptr;
        int N = 0, K = 0;
    };
    struct L {
        MX q, k, v, o, g, u, d;
        float *in1 = nullptr, *post1 = nullptr, *qn1 = nullptr, *kn1 = nullptr;
        uint16_t *kc = nullptr, *vc = nullptr;
    };
    std::vector<L> L;
    uint16_t *fc = nullptr, *w1 = nullptr, *w2 = nullptr;
    float *hid1 = nullptr, *fin1 = nullptr, *cw = nullptr, *cb = nullptr, *inv = nullptr;
    // scratch
    float *ctx = nullptr, *xn = nullptr, *h = nullptr, *q = nullptr, *kk = nullptr, *vv = nullptr, *att = nullptr;
    float *ao = nullptr, *gt = nullptr, *up = nullptr, *hid = nullptr, *logit = nullptr, *bias = nullptr;
    float *embf = nullptr, *conf = nullptr;
    float* mxpart = nullptr;   // mx_rows' split partials
    uint16_t *x16 = nullptr, *hid16 = nullptr, *embb = nullptr;
    int *pos = nullptr, *lo = nullptr, *prev = nullptr, *out = nullptr;
    int* host = nullptr;   // pinned: drafts then confidences
    uint16_t* anc = nullptr;   // (mapped) the anchor's embedding row, BF16
    float* maskf = nullptr;    // the mask token's embedding row, F32 (device)
    k::QRow* qh = nullptr;   // (mapped) a sampled proposal's distributions, a row a position
    DraftProf prof;
    bool anc_busy = false;   // a proposal enqueued that may still read anc (one that failed before its own wait)

    ~Impl() {
        cudaSetDevice(dev);
        for (void* x : owned) cudaFree(x);
        if (host) cudaFreeHost(host);
        if (qh) cudaFreeHost(qh);
        if (anc) cudaFreeHost(anc);
    }

    template <class T>
    bool alloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaMalloc(&v, std::max<size_t>(n * sizeof(T), 16)) != cudaSuccess) {
            err = "dspark: out of device memory (" + std::to_string((n * sizeof(T)) >> 20) + " MB)";
            return false;
        }
        owned.push_back(v);
        bytes += n * sizeof(T);
        p = (T*) v;
        return true;
    }

    bool bf16(const Pack& pk, const std::string& name, uint16_t*& out, size_t want, std::string& err) {
        const Tensor* t = pk.get(name);
        if (!t || t->dtype != "BF16" || t->bytes != want * 2) {
            err = "dspark: " + name + ": not the BF16 tensor of " + std::to_string(want) + " values";
            return false;
        }
        if (!alloc(out, want, err)) return false;
        DK(cudaMemcpy(out, pk.data(*t), t->bytes, cudaMemcpyHostToDevice));
        return true;
    }

    // a float vector (BF16 stored) + add
    bool vec(const Pack& pk, const std::string& name, float add, float*& out, size_t want, std::string& err) {
        const Tensor* t = pk.get(name);
        if (!t || t->dtype != "BF16" || t->bytes != want * 2) {
            err = "dspark: " + name + ": not the BF16 vector of " + std::to_string(want);
            return false;
        }
        std::vector<float> hv(want);
        const uint16_t* b = (const uint16_t*) pk.data(*t);
        for (size_t i = 0; i < want; ++i) {
            const uint32_t u = (uint32_t) b[i] << 16;
            float f;
            std::memcpy(&f, &u, 4);
            hv[i] = f + add;
        }
        if (!alloc(out, want, err)) return false;
        DK(cudaMemcpy(out, hv.data(), want * 4, cudaMemcpyHostToDevice));
        return true;
    }

    // an MXFP8 matrix [N][K] on the device as it is stored (mx_rows reads it)
    bool mx(const Pack& pk, const std::string& pre, int N, int K, MX& m, std::string& err) {
        const Tensor* w = pk.get(pre + ".weight");
        const Tensor* sc = pk.get(pre + ".weight_scale_inv");
        if (!w || !sc || w->dtype != "F8_E4M3" || w->bytes != (uint64_t) N * K || sc->bytes != (uint64_t) N * (K / 32) ||
            N % 32 || K % 512) {
            err = "dspark: " + pre + ": not the MXFP8 matrix [" + std::to_string(N) + "][" + std::to_string(K) + "]";
            return false;
        }
        uint8_t *dw = nullptr, *ds = nullptr;
        if (!alloc(dw, w->bytes, err) || !alloc(ds, sc->bytes, err)) return false;
        DK(cudaMemcpy(dw, pk.data(*w), w->bytes, cudaMemcpyHostToDevice));
        DK(cudaMemcpy(ds, pk.data(*sc), sc->bytes, cudaMemcpyHostToDevice));
        m.w = dw;
        m.s = ds;
        m.N = N;
        m.K = K;
        return true;
    }
    // C [rows][N] F32 = X [rows][K] F32 . W^T (+ beta C), W MXFP8
    void mxg(const MX& m, const float* X, int rows, int ldx, float* C, int ldc, float beta) {
        k::mx_rows(m.w, m.s, m.N, m.K, X, rows, ldx, C, ldc, beta, mxpart, s);
    }

    // (the expansion to BF16, kept for reference)
    bool mx_bf16(const Pack& pk, const std::string& pre, int N, int K, uint16_t*& out, std::string& err) {
        const Tensor* w = pk.get(pre + ".weight");
        const Tensor* sc = pk.get(pre + ".weight_scale_inv");
        if (!w || !sc || w->dtype != "F8_E4M3" || w->bytes != (uint64_t) N * K || sc->bytes != (uint64_t) N * (K / 32)) {
            err = "dspark: " + pre + ": not the MXFP8 matrix [" + std::to_string(N) + "][" + std::to_string(K) + "]";
            return false;
        }
        uint8_t *dw = nullptr, *ds = nullptr;
        DK(cudaMalloc(&dw, w->bytes));
        DK(cudaMalloc(&ds, sc->bytes));
        DK(cudaMemcpy(dw, pk.data(*w), w->bytes, cudaMemcpyHostToDevice));
        DK(cudaMemcpy(ds, pk.data(*sc), sc->bytes, cudaMemcpyHostToDevice));
        if (!alloc(out, (size_t) N * K, err)) return false;
        k::mxfp8_to_bf16(dw, ds, N, K, out, s);
        DK(cudaStreamSynchronize(s));
        cudaFree(dw);
        cudaFree(ds);
        return true;
    }

    // C [rows][N] F32 = X [rows][K] BF16 . W [N][K]^T BF16 (+ beta C)
    bool gemm(const uint16_t* W, int N, int K, int ldw, const uint16_t* X, int rows, int ldx, float* C, int ldc,
              float beta, std::string& err) {
        const float one = 1.0f;
        DB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &one, W, CUDA_R_16BF, ldw, X, CUDA_R_16BF, ldx,
                        &beta, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        return true;
    }

    bool load(const Pack& pk, int dev_, cudaStream_t s_, cublasHandle_t blas_, const x3::Mat* head_,
              const uint16_t* emb_, int max_rows_, std::string& err) {
        dev = dev_;
        s = s_;
        blas = blas_;
        head = head_;
        emb = emb_;
        max_rows = std::max(max_rows_, 8);
        g.dim = (int) pk.num("hidden_size");
        g.n_head = (int) pk.num("num_attention_heads");
        g.n_kv = (int) pk.num("num_key_value_heads");
        g.hd = (int) pk.num("head_dim");
        g.n_rot = (int) pk.num("rotary_dim", g.hd * pk.num("partial_rotary_factor", 1.0));
        g.vocab = (int) pk.num("vocab_size");
        g.eps = (float) pk.num("rms_norm_eps");
        g.theta = (float) pk.num("rope_theta");
        g.alpha = (float) pk.num("swiglu_alpha");
        g.limit = (float) pk.num("swiglu_limit");
        g.n_layers = (int) pk.num("dspark.num_hidden_layers");
        g.ff = (int) pk.num("dspark.dspark_ffn_hidden_size");
        g.rank = (int) pk.num("dspark.dspark_markov_rank");
        g.block = (int) pk.num("dspark.dspark_block_size");
        g.mask_id = (int) pk.num("dspark.dspark_noise_token_id");
        g.window = (int) pk.num("dspark.dflash.sliding_window", 4096);
        g.taps = pk.ints("dspark.dspark_target_layer_ids");
        if (g.n_layers <= 0 || g.ff <= 0 || g.rank <= 0 || g.block <= 0 || g.block > 16 || g.taps.empty() ||
            g.hd != 128 || g.n_head != 16 * g.n_kv) {
            err = "dspark: the pack has no DSpark draft this engine covers (tools/mm3/pack.py with dspark/ beside "
                  "the model)";
            return false;
        }
        if (pk.str("dspark.dflash.causal", "0") == "1") {
            err = "dspark: a causal draft is not covered";
            return false;
        }
        ring = g.window + 64;
        cudaSetDevice(dev);
        const int D = g.dim, Q = g.n_head * g.hd, KV = g.n_kv * g.hd, F = g.ff, V = g.vocab, R = g.rank;
        const int NT = (int) g.taps.size();
        if (!bf16(pk, DP + "fc.weight", fc, (size_t) D * NT * D, err) ||
            !vec(pk, DP + "hidden_norm.weight", 1.0f, hid1, (size_t) D, err) ||
            !vec(pk, DP + "final_norm.weight", 1.0f, fin1, (size_t) D, err) ||
            !bf16(pk, DP + "markov_head.markov_w1.weight", w1, (size_t) V * R, err) ||
            !bf16(pk, DP + "markov_head.markov_w2.weight", w2, (size_t) V * R, err) ||
            !vec(pk, DP + "confidence_head.proj.weight", 0.0f, cw, (size_t) D + R, err) ||
            !vec(pk, DP + "confidence_head.proj.bias", 0.0f, cb, 1, err))
            return false;
        L.resize((size_t) g.n_layers);
        for (int i = 0; i < g.n_layers; ++i) {
            const std::string P = DP + "layers." + std::to_string(i) + ".decoder_layer.";
            auto& l = L[(size_t) i];
            if (!mx(pk, P + "self_attn.q_proj", Q, D, l.q, err) || !mx(pk, P + "self_attn.k_proj", KV, D, l.k, err) ||
                !mx(pk, P + "self_attn.v_proj", KV, D, l.v, err) || !mx(pk, P + "self_attn.o_proj", D, Q, l.o, err) ||
                !mx(pk, P + "mlp.gate_proj", F, D, l.g, err) || !mx(pk, P + "mlp.up_proj", F, D, l.u, err) ||
                !mx(pk, P + "mlp.down_proj", D, F, l.d, err) ||
                !vec(pk, P + "input_layernorm.weight", 1.0f, l.in1, (size_t) D, err) ||
                !vec(pk, P + "post_attention_layernorm.weight", 1.0f, l.post1, (size_t) D, err) ||
                !vec(pk, P + "self_attn.q_norm.weight", 1.0f, l.qn1, (size_t) g.hd, err) ||
                !vec(pk, P + "self_attn.k_norm.weight", 1.0f, l.kn1, (size_t) g.hd, err) ||
                !alloc(l.kc, (size_t) ring * KV, err) || !alloc(l.vc, (size_t) ring * KV, err))
                return false;
        }
        std::vector<float> inv_h((size_t) g.n_rot / 2);
        k::rope_inv_freq(g.theta, g.n_rot, inv_h.data());
        const int Rw = max_rows, B = g.block;
        if (!alloc(mxpart, (size_t) 16 * 16 * std::max(D, std::max(Q, F)), err)) return false;
        if (!alloc(inv, inv_h.size(), err) || !alloc(ctx, (size_t) Rw * D, err) || !alloc(xn, (size_t) Rw * D, err) ||
            !alloc(h, (size_t) B * D, err) || !alloc(q, (size_t) B * Q, err) || !alloc(kk, (size_t) Rw * KV, err) ||
            !alloc(vv, (size_t) Rw * KV, err) || !alloc(att, (size_t) B * Q, err) || !alloc(ao, (size_t) B * D, err) ||
            !alloc(gt, (size_t) B * F, err) || !alloc(up, (size_t) B * F, err) || !alloc(hid, (size_t) B * D, err) ||
            !alloc(logit, (size_t) B * V, err) || !alloc(bias, (size_t) V, err) || !alloc(embf, (size_t) R, err) ||
            !alloc(conf, (size_t) B, err) || !alloc(x16, (size_t) Rw * std::max(std::max(D, Q), F), err) ||
            !alloc(hid16, (size_t) B * D, err) || !alloc(embb, (size_t) R, err) || !alloc(pos, (size_t) Rw, err) ||
            !alloc(lo, (size_t) B, err) || !alloc(prev, 1, err) || !alloc(out, (size_t) B, err))
            return false;
        DK(cudaMemcpy(inv, inv_h.data(), inv_h.size() * 4, cudaMemcpyHostToDevice));
        DK(cudaMallocHost(&host, (size_t) 2 * B * sizeof(int)));
        DK(cudaHostAlloc((void**) &qh, (size_t) B * sizeof(k::QRow), cudaHostAllocMapped | cudaHostAllocPortable));
        // (a proposal's inputs without the copy engine, which the expert cache's promotions keep busy: the anchor's
        //  row through mapped memory, the mask token's on the device, the positions made there)
        DK(cudaHostAlloc((void**) &anc, (size_t) D * 2, cudaHostAllocMapped | cudaHostAllocPortable));
        {
            std::vector<float> mf((size_t) D);
            std::vector<uint16_t> mrow((size_t) D);
            row(g.mask_id, mrow.data());
            const uint16_t* r = mrow.data();
            for (int i = 0; i < D; ++i) {
                const uint32_t u = (uint32_t) r[i] << 16;
                std::memcpy(&mf[(size_t) i], &u, 4);
            }
            if (!alloc(maskf, (size_t) D, err)) return false;
            DK(cudaMemcpy(maskf, mf.data(), (size_t) D * 4, cudaMemcpyHostToDevice));
        }
        return true;
    }

    bool context(const float* taps, int T, int pos0, size_t stride, std::string& err) {
        const int NT = (int) g.taps.size();
        if (stride == 0) stride = (size_t) T;
        for (int a = 0; a < T; a += max_rows) {
            const int n = std::min(max_rows, T - a);
            for (int j = 0; j < NT; ++j)
                if (!context_tap(j, taps + ((size_t) j * stride + a) * g.dim, n, err)) return false;
            if (!context_finish(n, pos0 + a, err)) return false;
        }
        return true;
    }

    // ctx += tap_j . fc[:, j D:(j + 1) D]^T (= for j 0)
    bool context_tap(int j, const float* rows, int n, std::string& err) {
        const int D = g.dim, NT = (int) g.taps.size();
        cudaSetDevice(dev);
        k::to_bf16(rows, n, D, D, x16, D, s);
        return gemm(fc + (size_t) j * D, D, D, NT * D, x16, n, D, ctx, D, j == 0 ? 0.0f : 1.0f, err);
    }

    bool context_finish(int n, int pos0, std::string& err) {
        const int D = g.dim, KV = g.n_kv * g.hd;
        cudaSetDevice(dev);
        k::rms(ctx, hid1, n, D, g.eps, xn, nullptr, s);
        k::iota(pos, n, pos0, s);   // (on the device: an upload would wait behind the promotions' copies)
        for (auto& l : L) {
            mxg(l.k, xn, n, D, kk, KV, 0.0f);
            mxg(l.v, xn, n, D, vv, KV, 0.0f);
            k::qk_norm_rope(nullptr, kk, vv, nullptr, l.kn1, pos, n, 0, g.n_kv, g.hd, g.n_rot, inv, g.eps, l.kc,
                            l.vc, s, ring);
        }
        return true;
    }

    // sa (given): each token drawn from the biased logits' distribution (sa, u[i] the draw of position i), qd its rows;
    // else greedy
    bool propose(int anchor, int P, int n, std::vector<int>& drafts, std::vector<float>& cf, const k::SampleArgs* sa,
                 const float* u, std::vector<std::vector<std::pair<int, float>>>* qd, std::string& err) {
        const int D = g.dim, Q = g.n_head * g.hd, KV = g.n_kv * g.hd, F = g.ff, V = g.vocab, R = g.rank, B = g.block;
        n = std::max(1, std::min(n, B));
        cudaSetDevice(dev);
        // the block's embeddings: the anchor, then the mask token (the target's table, BF16 -> F32), and the
        // positions - made on the device (the anchor's row read where the host put it)
        // (the last proposal done with anc: it waited for its drafts at its end, unless it failed before that - no
        //  wait here otherwise, so the target's context rows queued before it go to the GPU with this proposal)
        if (anc_busy) DK(cudaStreamSynchronize(s));
        anc_busy = true;
        row(anchor, anc);
        prof.start(s);
        k::draft_block(anc, maskf, D, B, P, g.window, h, pos, lo, s);
        const float scale = 1.0f / std::sqrt((float) g.hd);
        for (auto& l : L) {
            k::rms(h, l.in1, B, D, g.eps, xn, nullptr, s);
            mxg(l.q, xn, B, D, q, Q, 0.0f);
            mxg(l.k, xn, B, D, kk, KV, 0.0f);
            mxg(l.v, xn, B, D, vv, KV, 0.0f);
            k::qk_norm_rope(q, kk, vv, l.qn1, l.kn1, pos, B, g.n_head, g.n_kv, g.hd, g.n_rot, inv, g.eps, l.kc, l.vc,
                            s, ring);
            k::window_attn(q, l.kc, l.vc, ring, lo, P + B - 1, B, g.n_head, g.n_kv, scale, att, s);
            mxg(l.o, att, B, Q, h, D, 1.0f);   // h += attn Wo
            k::rms(h, l.post1, B, D, g.eps, xn, nullptr, s);
            mxg(l.g, xn, B, D, gt, F, 0.0f);
            mxg(l.u, xn, B, D, up, F, 0.0f);
            k::swigluoai(gt, up, (size_t) B * F, g.alpha, g.limit, gt, nullptr, s);
            mxg(l.d, gt, B, F, h, D, 1.0f);   // h += mlp
        }
        k::rms(h, fin1, B, D, g.eps, hid, hid16, s);
        prof.mark(s, 0);
        // (the target's head on mv, F32 inputs: one pass for the block's rows - gemm_rows took ~1.8 ms on the 3090)
        for (int c0 = 0; c0 < n;) {
            x3::MvJob j;
            j.m = head;
            j.xf = hid + (size_t) c0 * D;
            j.y = logit + (size_t) c0 * V;
            j.ncol = std::min(std::max(1, x3::mv_cols(*head)), n - c0);
            j.xf_cs = (size_t) D;
            j.y_cs = (size_t) V;
            if (!x3::mv(&j, 1, s)) {
                err = "dspark: mv head";
                return false;
            }
            c0 += j.ncol;
        }
        prof.mark(s, 1);
        k::set_int(prev, anchor, s);
        for (int i = 0; i < n; ++i) {
            k::gather_row(w1, R, prev, embb, embf, s);
            const float one = 1.0f, zero = 0.0f;
            DB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, V, 1, R, &one, w2, CUDA_R_16BF, R, embb, CUDA_R_16BF, R,
                            &zero, bias, CUDA_R_32F, V, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            if (sa) {
                k::SampleArgs a = *sa;
                a.u = u[i];
                k::markov_sample(logit + (size_t) i * V, bias, V, prev, out + i, cw, cb, hid + (size_t) i * D, D, embf,
                                 R, conf + i, a, qh + i, s);
            } else {
                k::markov_pick(logit + (size_t) i * V, bias, V, prev, out + i, nullptr, cw, cb, hid + (size_t) i * D,
                               D, embf, R, conf + i, s);
            }
        }
        DK(cudaMemcpyAsync(host, out, (size_t) n * 4, cudaMemcpyDeviceToHost, s));
        DK(cudaMemcpyAsync(host + B, conf, (size_t) n * 4, cudaMemcpyDeviceToHost, s));
        DK(cudaStreamSynchronize(s));
        prof.mark(s, 2);
        ++prof.n;
        anc_busy = false;
        drafts.assign(host, host + n);
        cf.resize((size_t) n);
        std::memcpy(cf.data(), host + B, (size_t) n * 4);
        if (sa && qd) {
            qd->assign((size_t) n, {});
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < qh[i].n; ++j) (*qd)[(size_t) i].push_back({qh[i].id[j], qh[i].q[j]});
        }
        return true;
    }
};

Draft::Draft() : p(new Impl) {}
Draft::~Draft() { delete p; }
bool Draft::load(const Pack& pk, int dev, cudaStream_t s, cublasHandle_t blas, const x3::Mat* head,
                 const uint16_t* emb, int max_rows, std::string& err) {
    return p->load(pk, dev, s, blas, head, emb, max_rows, err);
}
void Draft::set_emb_rows(std::function<void(int, uint16_t*)> f) { p->emb_row = std::move(f); }
const DraftGeo& Draft::geo() const { return p->g; }
bool Draft::context(const float* taps, int T, int pos0, std::string& err, size_t stride) {
    return p->context(taps, T, pos0, stride, err);
}
bool Draft::context_tap(int j, const float* rows, int T, std::string& err) { return p->context_tap(j, rows, T, err); }
bool Draft::context_finish(int T, int pos0, std::string& err) { return p->context_finish(T, pos0, err); }
bool Draft::propose(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, float* logits,
                    std::string& err) {
    (void) logits;
    return p->propose(anchor, pos, n, drafts, conf, nullptr, nullptr, nullptr, err);
}
bool Draft::propose_sampled(int anchor, int pos, int n, const k::SampleArgs& sa, const std::vector<float>& u,
                            std::vector<int>& drafts, std::vector<float>& conf,
                            std::vector<std::vector<std::pair<int, float>>>& q, std::string& err) {
    if ((int) u.size() < std::max(1, std::min(n, p->g.block))) {
        err = "dspark: fewer draws than drafts";
        return false;
    }
    return p->propose(anchor, pos, n, drafts, conf, &sa, u.data(), &q, err);
}
size_t Draft::bytes() const { return p->bytes; }

}  // namespace mm3
