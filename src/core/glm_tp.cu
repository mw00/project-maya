// src/core/glm_tp.cu - GLM tensor parallel (MAYA_GLM_TP=N, opt-in): the members, the host all-reduce, the cuts.
//
// The members are Glm5Models on GPUs of one kind.  Each loads EVERY layer with a slice of it: heads [h0, h1) of the
// KDA mixer (q/k/v rows, conv channels, gates, beta, the recurrent state) and of the MLA mixer (q_b rows, k_b/v_b
// heads), the output projections' matching columns, FFN width [f0, f1) of each layer's routed experts, shared expert
// and dense FFN (gate/up rows, down columns), and a copy of the rest (the hyper-connections, the router, the MLA
// latent and the DSA indexer, the norms).  Member 0 drives: a token is queued on every member at once by one thread a
// member, each member's mixer part and FFN part (n_embd floats) go to pinned host memory with a flag, the driver's
// thread adds them IN MEMBER ORDER and releases the sum, which every member reads back into the same buffer - so the
// members' residual copies stay identical bit for bit and every run gives the same tokens.  Member 0 also holds the
// head.  This version: GLM-5.3-Flash GGUF packs, every expert slice in VRAM, the token path only (prompts too).
#include "glm_fast_state.hpp"
#include "glm_residual.hpp"
#include "maya/core/glm_model.hpp"
#include "maya/kernels/glm_fast.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

namespace maya::core {

namespace {
constexpr size_t kLine = 16;   // flags 64 bytes apart (unsigned)
inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void cpu_relax() {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

// a member's part (n4 float4s) into the host's slot, then its flag = v once every value is out
__global__ void tp_publish_kernel(const float4* __restrict__ src, float4* dst, int n4, unsigned* flag, unsigned v) {
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dst[i] = src[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile unsigned*) flag = v;
        __threadfence_system();
    }
}
// once the host's fin flag reaches v (wrapping compare; given up after ~2^37 cycles), the sum into dst
__global__ void tp_pull_kernel(const unsigned* fin, unsigned v, float* __restrict__ dst, const float* src, int n) {
    if (threadIdx.x == 0) {
        const long long t0 = clock64();
        while ((int) (*(const volatile unsigned*) fin - v) < 0)
            if (clock64() - t0 > (1ll << 37)) break;
    }
    __syncthreads();
    const volatile float* s = src;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = s[i];
}
}  // namespace

// The host side of a group: the mapped slots and flags, the members' enqueue threads, the counters.
struct GlmTpGroup {
    int n = 0;
    size_t E = 0, stride = 0;
    float* part = nullptr;      // n x stride (mapped, pinned)
    float* sum = nullptr;       // stride
    unsigned* flags = nullptr;  // n x kLine
    unsigned* fin = nullptr;    // kLine
    unsigned seq = 0;
    bool broken = false;
    std::atomic<int> fails{0};
    // the crew: one thread a member, each running job(i) per token
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable cv, done_cv;
    std::function<void(int)> job;
    unsigned gen = 0;
    int left = 0;
    bool quit = false;
    // MAYA_GLM_TP_PROF: per token, the driver's wait for the members' parts and its sums
    bool prof = false;
    double t_tok = 0, t_wait = 0, t_sum = 0, t_last = 0;
    uint64_t n_tok = 0, n_ar = 0;

    bool init(int n_members, size_t n_embd, std::string& err) {
        n = n_members;
        E = n_embd;
        stride = (E + 63) & ~(size_t) 63;
        const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
        if (cudaHostAlloc((void**) &part, sizeof(float) * stride * (size_t) n, fl) != cudaSuccess ||
            cudaHostAlloc((void**) &sum, sizeof(float) * stride, fl) != cudaSuccess ||
            cudaHostAlloc((void**) &flags, sizeof(unsigned) * kLine * (size_t) n, fl) != cudaSuccess ||
            cudaHostAlloc((void**) &fin, sizeof(unsigned) * kLine, fl) != cudaSuccess) {
            err = "glm tp: the host all-reduce buffers did not allocate";
            return false;
        }
        std::memset(part, 0, sizeof(float) * stride * (size_t) n);
        std::memset(sum, 0, sizeof(float) * stride);
        std::memset(flags, 0, sizeof(unsigned) * kLine * (size_t) n);
        std::memset(fin, 0, sizeof(unsigned) * kLine);
        prof = getenv("MAYA_GLM_TP_PROF") != nullptr;
        for (int i = 0; i < n; ++i)
            th.emplace_back([this, i] {
                unsigned seen = 0;
                std::unique_lock<std::mutex> lk(mu);
                for (;;) {
                    cv.wait(lk, [&] { return quit || gen != seen; });
                    if (quit) return;
                    seen = gen;
                    const std::function<void(int)> f = job;
                    lk.unlock();
                    f(i);
                    lk.lock();
                    if (--left == 0) done_cv.notify_all();
                }
            });
        return true;
    }
    void start(const std::function<void(int)>& f) {
        std::lock_guard<std::mutex> lk(mu);
        job = f;
        left = n;
        ++gen;
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        done_cv.wait(lk, [&] { return left == 0; });
    }
    // the driver's side of one all-reduce: every member's part (in member order), their sum, fin = v.  false: a member
    // failed or nothing came for a minute - every member's wait is let go (its later kernels then run on garbage, the
    // group is broken and refuses further tokens)
    bool reduce(unsigned v) {
        const double t0 = prof ? now_s() : 0.0;
        const double give_up = now_s() + 60.0;
        for (int j = 0; j < n; ++j) {
            const volatile unsigned* f = flags + (size_t) j * kLine;
            for (long spin = 0; (int) (*f - v) < 0; ++spin) {
                cpu_relax();
                if ((spin & 0xfff) == 0xfff && (fails.load() > 0 || now_s() > give_up)) {
                    *(volatile unsigned*) fin = v + 0x40000000u;
                    broken = true;
                    return false;
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const double t1 = prof ? now_s() : 0.0;
        for (size_t i = 0; i < E; ++i) {
            float s = part[i];
            for (int j = 1; j < n; ++j) s += part[(size_t) j * stride + i];
            sum[i] = s;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *(volatile unsigned*) fin = v;
        if (prof) {
            const double t2 = now_s();
            t_wait += t1 - t0;
            t_sum += t2 - t1;
            ++n_ar;
        }
        return true;
    }
    void destroy() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
            cv.notify_all();
        }
        for (auto& t : th) t.join();
        th.clear();
        for (void* p : {(void*) part, (void*) sum, (void*) flags, (void*) fin})
            if (p != nullptr) cudaFreeHost(p);
        part = sum = nullptr;
        flags = fin = nullptr;
    }
};

// ---------------------------------------------------------------- the cuts
Glm5Model::TpCut Glm5Model::tp_cut(const std::string& n) const {
    TpCut c;
    if (!tp_on()) return c;
    if (n == "output.weight" || n == "output_norm.weight") {
        if (!tp_.head) c.mode = 3;
        return c;
    }
    if (n.rfind("blk.", 0) != 0) return c;
    const int il = std::atoi(n.c_str() + 4);
    if (il < 0 || il >= g_.n_layers || il >= (int) tp_.f0.size()) return c;
    const std::string s = n.substr(n.find('.', 4) + 1);
    const int64_t h0 = tp_.h0, h1 = tp_.h1, f0 = tp_.f0[(size_t) il], f1 = tp_.f1[(size_t) il];
    const auto rows = [&](int64_t a, int64_t b) { c.mode = 1, c.a0 = a, c.a1 = b; };
    const auto cols = [&](int64_t a, int64_t b) { c.mode = 2, c.a0 = a, c.a1 = b; };
    if (s == "ffn_gate.weight" || s == "ffn_up.weight" || s == "ffn_gate_shexp.weight" || s == "ffn_up_shexp.weight")
        rows(f0, f1);
    else if (s == "ffn_down.weight" || s == "ffn_down_shexp.weight")
        cols(f0, f1);
    else if (g_.is_recr(il)) {   // KDA: heads of kda_head_dim channels
        const int64_t D = g_.kda_head_dim;
        if (s == "attn_q.weight" || s == "attn_k.weight" || s == "attn_v.weight" || s == "ssm_f_b.weight" ||
            s == "ssm_g_b.weight" || s.rfind("ssm_conv1d_", 0) == 0)
            rows(h0 * D, h1 * D);
        else if (s == "ssm_dt.bias")
            cols(h0 * D, h1 * D);
        else if (s == "ssm_beta.weight")
            rows(h0, h1);
        else if (s == "ssm_a")
            cols(h0, h1);
        else if (s == "attn_output.weight")
            cols(h0 * D, h1 * D);
    } else {   // MLA: q_b's heads of qk_nope rows, k_b / v_b one block a head (their last dimension), out's columns
        if (s == "attn_q_b.weight")
            rows(h0 * g_.qk_nope, h1 * g_.qk_nope);
        else if (s == "attn_k_b.weight" || s == "attn_v_b.weight")
            rows(h0, h1);
        else if (s == "attn_output.weight")
            cols(h0 * g_.v_head, h1 * g_.v_head);
    }
    return c;
}

void glmfast::tp_compact(uint8_t* b, const Glm5Model::FastState::Layer& Ly, int n_embd) {
    if (Ly.ff <= 0) return;
    const size_t gu_row = Ly.gu_bytes / (size_t) Ly.ff;
    std::memmove(b, b + (size_t) Ly.f0 * gu_row, Ly.gu_bytes);
    std::memmove(b + Ly.gu_bytes, b + Ly.src_gu + (size_t) Ly.f0 * gu_row, Ly.gu_bytes);
    const size_t o = Ly.f0 > 0 ? gf::row_bytes(Ly.d_type, Ly.f0) : 0;
    for (int r = 0; r < n_embd; ++r)
        std::memmove(b + Ly.down_off + (size_t) r * Ly.d_row, b + 2 * Ly.src_gu + (size_t) r * Ly.src_d_row + o, Ly.d_row);
}

// ---------------------------------------------------------------- the plan
// Heads: equal shares in units an output projection's quant blocks allow (the last members take the odd ones - member 0
// holds the head).  FFN width per layer: equal shares of the down projection's blocks; where they do not divide, the
// short shares go to the members holding the most bytes so far (the head counted), so the experts spread evenly.
bool Glm5Model::tp_plan(const std::string& pack_dir, int n, std::vector<TpSlice>& out, std::string& err) {
    const Glm5Geometry& g = g_;
    if (g.arch == GlmArch::GlmDsa || !g.mhc()) {
        err = "glm tp: tensor parallel covers GLM-5.3-Flash for now (not this architecture)";
        return false;
    }
    if (g.n_shared != 1) {
        err = "glm tp: one shared expert expected, the model has " + std::to_string(g.n_shared);
        return false;
    }
    if (std::ifstream(pack_dir + "/exl3.txt").good()) {
        err = "glm tp: EXL3 packs are not covered yet (GGUF packs only)";
        return false;
    }
    // every tensor's ggml type from the model's shards - the column cuts must fall on its quant blocks (the index's
    // native rows carry a fixed "8 0 32" marker, not the tensor's type: a Q6_K attn_output read as Q8_0 got cut
    // mid-block)
    std::map<std::string, int> qtype;
    {
        Glm5Geometry gtmp;
        std::vector<std::pair<std::string, int>> types;
        if (!glm_pack_geometry_full(pack_dir, gtmp, err, &types)) return false;
        for (const auto& t : types) qtype[t.first] = t.second;
    }
    const auto bw_of = [&](const std::string& name) -> int64_t {
        const auto it = qtype.find(name);
        return it == qtype.end() ? 1 : (int64_t) ggml_blck_size((ggml_type) it->second);
    };
    // the experts' types per layer (native_experts.txt)
    std::vector<int> gu_t((size_t) g.n_layers, -1), d_t((size_t) g.n_layers, -1);
    {
        std::ifstream ne(pack_dir + "/native_experts.txt");
        std::string line;
        while (std::getline(ne, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int il = -1, gu = -1, dt = -1;
            ss >> il >> gu >> dt;
            if (il >= 0 && il < g.n_layers) gu_t[(size_t) il] = gu, d_t[(size_t) il] = dt;
        }
    }
    // heads
    int64_t uh = 1;
    for (int il = 0; il < g.n_layers; ++il) {
        const int64_t hcols = g.is_recr(il) ? g.kda_head_dim : g.v_head;
        const int64_t bw = bw_of("blk." + std::to_string(il) + ".attn_output.weight");
        uh = std::lcm(uh, bw / std::gcd(bw, hcols));
    }
    if (g.n_head % uh != 0 || g.n_head / uh < n) {
        err = "glm tp: " + std::to_string(g.n_head) + " heads do not split over " + std::to_string(n) + " GPUs";
        return false;
    }
    out.assign((size_t) n, TpSlice{});
    {
        const int64_t U = g.n_head / uh, base = U / n, extra = U % n;
        int64_t at = 0;
        for (int m = 0; m < n; ++m) {
            const int64_t u = base + (m >= n - extra ? 1 : 0);
            out[(size_t) m].rank = m;
            out[(size_t) m].n = n;
            out[(size_t) m].h0 = (int) (at * uh);
            out[(size_t) m].h1 = (int) ((at + u) * uh);
            out[(size_t) m].head = m == 0;
            out[(size_t) m].f0.assign((size_t) g.n_layers, 0);
            out[(size_t) m].f1.assign((size_t) g.n_layers, 0);
            at += u;
        }
    }
    // FFN widths, balanced by bytes
    std::vector<double> bytes((size_t) n, 0.0);
    {
        const auto it = qtype.find("output.weight");
        bytes[0] += it == qtype.end() ? 2.0 * g.n_vocab * g.n_embd
                                      : (double) ggml_row_size((ggml_type) it->second, g.n_embd) * g.n_vocab;
    }
    for (int il = 0; il < g.n_layers; ++il) {
        const bool moe = il >= g.dense_lead;
        const std::string P = "blk." + std::to_string(il) + ".";
        const int64_t width = moe ? g.n_ff_exp : g.n_ff_dense;
        int64_t uf = moe ? bw_of(P + "ffn_down_shexp.weight") : bw_of(P + "ffn_down.weight");
        double unit_bytes = 0;   // the bytes one unit of width costs over the layer's experts (the planner's weight)
        if (moe) {
            if (d_t[(size_t) il] < 0) {
                err = "glm tp: no native experts for layer " + std::to_string(il);
                return false;
            }
            uf = std::lcm(uf, (int64_t) ggml_blck_size((ggml_type) d_t[(size_t) il]));
            unit_bytes = (double) g.n_expert *
                         ((double) 2 * uf * ggml_row_size((ggml_type) gu_t[(size_t) il], g.n_embd) +
                          (double) g.n_embd * ggml_row_size((ggml_type) d_t[(size_t) il], uf));
        } else {
            unit_bytes = 3.0 * uf * g.n_embd;
        }
        if (width % uf != 0 || width / uf < n) {
            err = "glm tp: layer " + std::to_string(il) + "'s FFN width " + std::to_string(width) +
                  " does not split over " + std::to_string(n) + " GPUs";
            return false;
        }
        const int64_t B = width / uf, base = B / n, extra = B % n;
        std::vector<int> order((size_t) n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return bytes[(size_t) a] > bytes[(size_t) b]; });
        std::vector<int64_t> units((size_t) n, base);
        for (int k = n - (int) extra; k < n; ++k) units[(size_t) order[(size_t) k]] += 1;
        int64_t at = 0;
        for (int m = 0; m < n; ++m) {
            out[(size_t) m].f0[(size_t) il] = (int) (at * uf);
            out[(size_t) m].f1[(size_t) il] = (int) ((at + units[(size_t) m]) * uf);
            at += units[(size_t) m];
            bytes[(size_t) m] += unit_bytes * (double) units[(size_t) m];
        }
    }
    std::string sh;
    for (int m = 0; m < n; ++m) {
        char b[96];
        std::snprintf(b, sizeof b, "%s%d heads, %.1f GB of experts%s", m ? "; " : "", out[(size_t) m].h1 - out[(size_t) m].h0,
                      bytes[(size_t) m] / 1e9, m == 0 ? " + the head" : "");
        sh += b;
    }
    std::fprintf(stderr, "glm tp: %d members - %s\n", n, sh.c_str());
    return true;
}

// ---------------------------------------------------------------- the load
bool Glm5Model::load_pack_tp(const std::string& pack_dir, int64_t max_ctx, const std::vector<int>& devs,
                             std::string& err) {
    const int n = (int) devs.size();
    if (n < 2 || n > 8) {
        err = "glm tp: 2 to 8 GPUs (MAYA_GLM_TP)";
        return false;
    }
    cudaDeviceProp p0{};
    if (cudaGetDeviceProperties(&p0, devs[0]) != cudaSuccess) {
        err = "glm tp: CUDA" + std::to_string(devs[0]) + " not found";
        return false;
    }
    for (int m = 1; m < n; ++m) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, devs[(size_t) m]) != cudaSuccess || std::strcmp(p.name, p0.name) != 0 ||
            p.major != p0.major || p.minor != p0.minor) {
            err = "glm tp: the members must be GPUs of one kind (CUDA" + std::to_string(devs[0]) + " " + p0.name +
                  ", CUDA" + std::to_string(devs[(size_t) m]) + " " + (p.name[0] ? p.name : "?") + ")";
            return false;
        }
    }
    for (const char* off : {"MAYA_GLM_EXPERT_HELPER", "MAYA_GLM_DFLASH"})
        if (const char* v = getenv(off); v != nullptr && v[0] != '\0') {
            err = std::string("glm tp: ") + off + " is not covered by tensor parallel yet";
            return false;
        }
    // GLM-5.3-Flash's geometry is read by load_pack, after this point: the plan needs it first
    if (!glm_pack_geometry_full(pack_dir, g_, err)) return false;
    std::vector<TpSlice> plan;
    if (!tp_plan(pack_dir, n, plan, err)) return false;
    auto* G = new GlmTpGroup();
    if (!G->init(n, (size_t) g_.n_embd, err)) {
        G->destroy();
        delete G;
        return false;
    }
    tpg_ = G;
    const double t0 = now_s();
    for (int m = 0; m < n; ++m) {
        Glm5Model* M = this;
        if (m > 0) {
            tp_peers_.emplace_back(new Glm5Model());
            M = tp_peers_.back().get();
            M->profile_path_ = profile_path_;
        }
        M->tp_ = plan[(size_t) m];
        M->tpg_ = G;
        std::string e;
        if (!M->load_pack(pack_dir, max_ctx, e, devs[(size_t) m], 0, 0)) {
            err = "glm tp: member " + std::to_string(m) + " (CUDA" + std::to_string(devs[(size_t) m]) + "): " + e;
            return false;
        }
        if (M->fast_ == nullptr) {
            err = "glm tp: the fast path is needed (unset MAYA_GLM_SLOW)";
            return false;
        }
    }
    std::fprintf(stderr, "glm tp: %d members loaded in %.0f s (CUDA%d", n, now_s() - t0, devs[0]);
    for (int m = 1; m < n; ++m) std::fprintf(stderr, ", CUDA%d", devs[(size_t) m]);
    std::fprintf(stderr, ") - host all-reduce, the token path for prompts, no drafting yet\n");
    return true;
}

void Glm5Model::tp_destroy() {
    if (tpg_ == nullptr) return;
    if (tp_.rank == 0 && tp_.n > 1) {
        tpg_->destroy();
        tp_peers_.clear();
        delete tpg_;
    }
    tpg_ = nullptr;
}

// ---------------------------------------------------------------- a token
void Glm5Model::tp_all_reduce(float* buf, cudaStream_t s) {
    GlmTpGroup* G = tpg_;
    const unsigned v = ++tp_seq_;
    const int E = (int) G->E;
    tp_publish_kernel<<<1, 256, 0, s>>>((const float4*) buf, (float4*) (G->part + (size_t) tp_.rank * G->stride), E / 4,
                                        G->flags + (size_t) tp_.rank * kLine, v);
    tp_pull_kernel<<<4, 256, 0, s>>>(G->fin, v, buf, G->sum, E);
}

bool Glm5Model::fast_token_tp(int32_t token, std::string& err) {
    GlmTpGroup* G = tpg_;
    if (G == nullptr || G->broken) {
        err = "glm tp: the group failed earlier - restart the engine";
        return false;
    }
    const Glm5Geometry& g = g_;
    const int64_t p = pos_++;
    std::vector<Glm5Model*> M{this};
    for (auto& m : tp_peers_) M.push_back(m.get());
    const int n = (int) M.size();
    for (Glm5Model* m : M) {
        cudaSetDevice(m->dev_);
        if (!m->fast_boundary(err)) return false;
    }
    cudaSetDevice(dev_);
    // the embedding row (host-dequantized, as one GPU does) -> every member's pinned row
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm fast: no embedding dequantizer";
        return false;
    }
    tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd), fast_->emb_h,
                 g.n_embd);
    if (const float* img = image_row(p)) std::memcpy(fast_->emb_h, img, (size_t) g.n_embd * sizeof(float));
    for (int m = 1; m < n; ++m) {
        std::memcpy(M[(size_t) m]->fast_->emb_h, fast_->emb_h, (size_t) g.n_embd * sizeof(float));
        M[(size_t) m]->pos_ = p;
    }
    const double t0 = G->prof ? now_s() : 0.0;
    std::vector<std::string> errs((size_t) n);
    G->fails.store(0);
    G->start([&](int i) {
        Glm5Model* m = M[(size_t) i];
        FastState* F = m->fast_;
        cudaSetDevice(m->dev_);
        cudaMemcpyAsync(F->emb, F->emb_h, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, F->cs);
        glmres::res_embed(m->g_, F->emb, m->state_, F->cs);
        bool ok = m->fast_layers(p, false, errs[(size_t) i]);
        if (ok && m->tp_.head) {
            glmres::res_head(m->g_, m->state_, m->w_.at("output_norm.weight"), F->head_x, F->head_xq, F->cs);
            const WSlot& ow = m->ws_map_.at("output.weight");
            gf::MvJob o = {ow.q, F->head_xq, F->head_x, m->sc_ + m->sc_logits, nullptr, 1.0f, ow.type, g.n_embd,
                           g.n_vocab};
            if (ow.type == 0) o.w = ow.f32;
            if (!gf::mv(&o, 1, F->cs)) {
                errs[(size_t) i] = "glm fast: output head (type " + std::to_string(ow.type) + ")";
                ok = false;
            } else {
                gf::argmax(m->sc_ + m->sc_logits, g.n_vocab, F->tok, F->cs);
                cudaMemcpyAsync(F->tok_h, F->tok, sizeof(int), cudaMemcpyDeviceToHost, F->cs);
                cudaEventRecord(F->ev_done, F->cs);
            }
        }
        if (!ok) G->fails.fetch_add(1);
    });
    // the all-reduces, in the order every member makes them: the mixer's and the FFN's of each layer
    bool ok = true;
    const int n_ar = 2 * (l1_ - l0_);
    for (int k = 0; k < n_ar && ok; ++k) ok = G->reduce(++G->seq);
    G->wait();
    for (int i = 0; i < n; ++i)
        if (!errs[(size_t) i].empty()) {
            err = "glm tp: member " + std::to_string(i) + ": " + errs[(size_t) i];
            G->broken = true;
            return false;
        }
    if (!ok) {
        err = "glm tp: a member's part did not come within a minute";
        return false;
    }
    FastState* FT = fast_;
    cudaSetDevice(dev_);
    cudaError_t done;
    while ((done = cudaEventQuery(FT->ev_done)) == cudaErrorNotReady) std::this_thread::yield();
    if (!glmfast::cuda_ok(done, "glm tp token completion", err)) return false;
    for (Glm5Model* m : M) {
        cudaSetDevice(m->dev_);
        if (!m->fast_->route_ok(err)) return false;
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            err = std::string("glm tp: CUDA") + std::to_string(m->dev_) + ": " + cudaGetErrorString(e);
            return false;
        }
    }
    cudaSetDevice(dev_);
    if (gf::launch_errors() > 0) {
        err = "glm fast: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    for (int m = 1; m < n; ++m) M[(size_t) m]->pos_ = pos_;
    last_tok_ = FT->tok_h[0];
    if (G->prof) {
        G->t_tok += now_s() - t0;
        if (++G->n_tok % 64 == 0) {
            std::fprintf(stderr, "glm tp: %llu tokens - %.2f ms a token, %.1f us waited a sync, %.1f us a sum "
                                 "(%d syncs a token)\n", (unsigned long long) G->n_tok, G->t_tok / 64 * 1e3,
                         G->t_wait / std::max<uint64_t>(1, G->n_ar) * 1e6, G->t_sum / std::max<uint64_t>(1, G->n_ar) * 1e6,
                         n_ar);
            G->t_tok = G->t_wait = G->t_sum = 0;
            G->n_ar = 0;
        }
    }
    return true;
}

}  // namespace maya::core
