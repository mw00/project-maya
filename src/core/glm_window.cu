// src/core/glm_window.cu - the speculative VERIFY window on the GLM trunk (Glm5Model members).
//
// T tokens at consecutive positions - the last accepted token and T-1 drafts - go through every layer in one pass,
// and the head's pick is produced for each of them.  Token t's pick is what token-at-a-time decode would produce
// after token t: the dense projections run as multi-column GEMVs (each column the single-column kernel's arithmetic,
// the weights read from DRAM once for all T), and every other kernel is the single-token one applied to each token
// in order on the same stream (the hyper-connections, the KDA convolution and recurrence, the DSA caches, selection
// and attention, the routes, the tiers and the CPU lane).  A sampled pick draws row t with the counter the token loop
// would have used for it.
// STATE: the KDA layers advance through all T tokens; the window keeps each KDA layer's per-token inputs, and
// window_commit(n) - fewer than T kept - restores the state saved before the window and replays the n kept tokens
// through the convolution and the recurrence alone.  The DSA caches take all T positions; a rejected one is
// rewritten when its position is processed again, before any query reads it (a pooled indexer key is built from the
// cached keys when its pool's last position is processed, and queries read only completed pools).
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/glm_fast.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace strata::core {

namespace {

float* take_f(uint8_t*& at, size_t floats) {
    float* p = (float*) at;
    at += (floats * sizeof(float) + 255u) & ~(size_t) 255u;
    return p;
}

void* take_b(uint8_t*& at, size_t bytes) {
    void* p = at;
    at += (bytes + 255u) & ~(size_t) 255u;
    return p;
}

// a KDA layer's inputs for token t of the window: q | k | v, f_a, g_a, beta
struct KdaIn {
    float *proj[3], *fa, *ga, *beta;
};

KdaIn kda_in(const Glm5Geometry& g, float* base, int tmax, int t) {
    const size_t DI = (size_t) g.d_inner(), T = (size_t) tmax;
    KdaIn k;
    for (int i = 0; i < 3; ++i) k.proj[i] = base + ((size_t) i * T + (size_t) t) * DI;
    float* rest = base + 3 * T * DI;
    k.fa = rest + (size_t) t * g.kda_head_dim;
    k.ga = rest + T * g.kda_head_dim + (size_t) t * g.kda_head_dim;
    k.beta = rest + 2 * T * g.kda_head_dim + (size_t) t * g.n_head;
    return k;
}

// one token of a KDA layer: its convolution (advancing the conv history) and recurrence (advancing S), the gated
// output to `gated_q` - the per-token half of fast_layers' KDA block
void kda_token(Glm5Model::FastState* F, const Glm5Model::FastState::Layer& Ly, const Glm5Geometry& g, const KdaIn& in,
               float* conv_state, float* S, void* gated_q, cudaStream_t s) {
    gf::KdaPrepArgs k;
    for (int i = 0; i < 3; ++i) {
        k.proj[i] = in.proj[i];
        k.conv_w[i] = Ly.conv[i];
        k.out[i] = F->conv[i];
    }
    k.conv_state = conv_state;
    k.fa = in.fa;
    k.ga = in.ga;
    k.f_b = Ly.f_b;
    k.g_b = Ly.g_b;
    k.dt_bias = Ly.dt_bias;
    k.ssm_a = Ly.ssm_a;
    k.lower_bound = g.kda_lb;
    k.g1 = F->g1;
    k.g2 = F->g2;
    k.n_head = g.n_head;
    k.head_dim = g.kda_head_dim;
    k.d_conv = g.d_conv;
    gf::kda_prep(k, s);
    gf::kda_rec(F->conv[0], F->conv[1], F->conv[2], F->g1, in.beta, S, F->g2, Ly.ssm_norm, g.norm_eps, g.n_head,
                g.kda_head_dim, gated_q, s);
}

}  // namespace

bool Glm5Model::win_alloc(int tmax, std::string& err) {
    FastState* F = fast_;
    auto& W = F->win;
    if (W.tmax >= tmax) return true;
    if (W.arena) cudaFree(W.arena);
    if (W.tok_h) cudaFreeHost(W.tok_h);
    if (W.emb_h) cudaFreeHost(W.emb_h);
    if (W.hop_h) cudaFreeHost(W.hop_h);
    if (W.scr_h) cudaFreeHost(W.scr_h);
    for (auto* a : W.ans_h)
        if (a) cudaFreeHost(a);
    cudaEvent_t ev = W.ev_hop;
    cudaEvent_t evt[kPipeSlots], evp[kPipeSlots];
    std::copy(std::begin(W.ev_tok), std::end(W.ev_tok), evt);
    std::copy(std::begin(W.ev_pick), std::end(W.ev_pick), evp);
    const int bank = W.bank;
    int* skip_d = W.skip_d;
    volatile int* from_h = W.from_h;
    std::vector<cudaEvent_t> pev = std::move(W.ev);
    std::vector<const char*> pnm = std::move(W.ev_name);
    W = FastState::Win{};
    W.ev_hop = ev;
    std::copy(evt, evt + kPipeSlots, W.ev_tok);
    std::copy(evp, evp + kPipeSlots, W.ev_pick);
    W.bank = bank;
    W.skip_d = skip_d;
    W.from_h = from_h;
    W.ev = std::move(pev);
    W.ev_name = std::move(pnm);
    const Glm5Geometry& g = g_;
    const size_t T = (size_t) tmax, E = (size_t) g.n_embd, DI = (size_t) g.d_inner(), q8 = 36;
    const size_t FF = (size_t) g.n_ff_exp * g.n_shared, hcE = (size_t) g.hc * E;
    // the rows a pipelined window's token crosses with (its embedding, its hop, its head row and pick) take a slot of
    // two banks (kPipeSlots), the batched window's tokens one each (T); the head's logits only where the head is
    const size_t TS = std::max(T, (size_t) kPipeSlots), TL = l1_ == g.n_layers ? TS : 1;
    W.xq_s = E / 32 * q8;
    W.gq_s = DI / 32 * q8;
    W.qrq_s = (size_t) g.q_lora / 32 * q8;
    W.aq_s = (size_t) g.n_head * g.v_head / 32 * q8;
    W.dhq_s = (size_t) g.n_ff_dense / 32 * q8;
    // the KDA layers here and their per-token inputs
    W.kin_of.assign((size_t) g.n_layers + 1, -1);
    int n_kda = 0;
    for (int il = l0_; il < l1_; ++il)
        if (F->L[(size_t) il].recr) W.kin_of[(size_t) il] = n_kda++;
    W.kin_layer = T * (3 * DI + 2 * (size_t) g.kda_head_dim + (size_t) g.n_head);
    const size_t f4 = sizeof(float);
    const auto al = [](size_t b) { return (b + 255u) & ~(size_t) 255u; };
    const size_t need = 2 * al(T * hcE * f4) + al(T * E * f4) * 3 + al(T * 8 * f4) * 2 + al(T * 16 * f4) +
                        al(T * W.xq_s) + al(T * W.gq_s) + 2 * al(T * (size_t) g.q_lora * f4) + al(T * W.qrq_s) +
                        al(T * (size_t) g.kv_lora * f4) + 2 * al(T * (size_t) g.idx_key * f4) +
                        al(T * (size_t) g.idx_heads * f4) + al(T * (size_t) g.n_head * g.qk_nope * f4) +
                        al(T * (size_t) g.idx_heads * g.idx_key * f4) + al(T * W.aq_s) +
                        al(T * (size_t) g.n_expert * f4) + 2 * al(T * FF * f4) + 2 * al(T * (size_t) g.n_ff_dense * f4) +
                        al(T * W.dhq_s) + al(TS * E * f4) + al(TS * W.xq_s) + al(TL * (size_t) g.n_vocab * f4) +
                        al(TS * sizeof(int)) + al(TS * E * f4) + al((size_t) n_kda * W.kin_layer * f4) +
                        // the absorbed MLA's rows and the tokens' cells
                        2 * al(T * (size_t) g.n_head * g.kv_lora * f4) + al(T * (size_t) g.n_sel_max() * sizeof(int)) +
                        // the batched MoE
                        2 * al(T * 8 * 8) + 2 * al(T * 8 * f4) + al(T * E * f4) + al(T * f4) +
                        al(T * ((size_t) FF / 32 * q8)) + al(T * E * f4) + al(T * 8 * ((size_t) g.n_ff_exp / 32 * q8)) +
                        al(T * 8 * E * f4) + al(T * 8 * 8) + 3 * al((T * 8 + 1) * f4) + al(64) + al(T * 64) +
                        (exl3_ ? al(T * 8 * (size_t) g.n_ff_exp * f4) : 0);
    if (cudaMalloc(&W.arena, need) != cudaSuccess) {
        cudaGetLastError();
        W.arena = nullptr;
        err = "glm window: " + std::to_string(need >> 20) + " MB of window buffers did not allocate";
        return false;
    }
    uint8_t* at = (uint8_t*) W.arena;
    for (auto& r : W.R) r = take_f(at, T * hcE);
    W.x = take_f(at, T * E);
    W.mixer = take_f(at, T * E);
    W.ffn = take_f(at, T * E);
    W.pre = take_f(at, T * 8);
    W.post = take_f(at, T * 8);
    W.comb = take_f(at, T * 16);
    W.xq = take_b(at, T * W.xq_s);
    W.gated_q = take_b(at, T * W.gq_s);
    W.qr_raw = take_f(at, T * g.q_lora);
    W.qr = take_f(at, T * g.q_lora);
    W.qr_q = take_b(at, T * W.qrq_s);
    W.kv_raw = take_f(at, T * g.kv_lora);
    W.ik_raw = take_f(at, T * g.idx_key);
    W.ig_raw = take_f(at, T * g.idx_key);
    W.iw = take_f(at, T * g.idx_heads);
    W.q = take_f(at, T * g.n_head * g.qk_nope);
    W.iq = take_f(at, T * g.idx_heads * g.idx_key);
    W.attn_q = take_b(at, T * W.aq_s);
    W.rlog = take_f(at, T * g.n_expert);
    W.sh_g = take_f(at, T * FF);
    W.sh_u = take_f(at, T * FF);
    W.dg = take_f(at, T * g.n_ff_dense);
    W.du = take_f(at, T * g.n_ff_dense);
    W.dhq = take_b(at, T * W.dhq_s);
    W.head_x = take_f(at, TS * E);
    W.head_xq = take_b(at, TS * W.xq_s);
    W.logits = take_f(at, TL * g.n_vocab);
    W.tok = (int*) take_b(at, TS * sizeof(int));
    W.emb = take_f(at, TS * E);
    W.kin = n_kda > 0 ? take_f(at, (size_t) n_kda * W.kin_layer) : nullptr;
    W.q_abs = take_f(at, T * (size_t) g.n_head * g.kv_lora);
    W.ctx = take_f(at, T * (size_t) g.n_head * g.kv_lora);
    W.cells = (int*) take_b(at, T * (size_t) g.n_sel_max() * sizeof(int));
    W.mplan_ptr = (unsigned long long*) take_b(at, T * 8 * 8);
    W.mfetch = (unsigned long long*) take_b(at, T * 8 * 8);
    W.mplan_w = take_f(at, T * 8);
    W.mplan_id = (int*) take_b(at, T * 8 * f4);
    W.mcpu_part = take_f(at, T * E);
    W.mcpu_flag = (int*) take_b(at, T * f4);
    W.sh_hq_s = (size_t) FF / 32 * q8;
    W.sh_hq = take_b(at, T * W.sh_hq_s);
    W.sh_out = take_f(at, T * E);
    W.whq = take_b(at, T * 8 * ((size_t) g.n_ff_exp / 32 * q8));
    W.wpart = take_f(at, T * 8 * E);
    W.grp_ptr = (unsigned long long*) take_b(at, T * 8 * 8);
    W.grp_start = (int*) take_b(at, (T * 8 + 1) * f4);
    W.grp_ent = (int*) take_b(at, (T * 8 + 1) * f4);
    W.n_groups = (int*) take_b(at, (T * 8 + 1) * f4);
    W.scr = (unsigned int*) take_b(at, 64);
    W.cpu_seq_w = (unsigned int*) take_b(at, T * 64);
    if (exl3_) W.wh = take_f(at, T * 8 * (size_t) g.n_ff_exp);
    cudaMemset(W.cpu_seq_w, 0, T * 64);
    cudaMemset(W.mcpu_flag, 0, T * f4);
    if ((size_t) (at - (uint8_t*) W.arena) > need) {
        err = "glm window: the window buffers overflowed their estimate";
        return false;
    }
    for (int t = 1; t < tmax && t < 16; ++t) {
        void* h = nullptr;
        if (cudaHostAlloc(&h, sizeof(gf::CpuAnswer), cudaHostAllocMapped) != cudaSuccess) {
            cudaGetLastError();
            err = "glm window: the CPU lane's answer slots did not allocate";
            return false;
        }
        std::memset(h, 0, sizeof(gf::CpuAnswer));
        W.ans_h[t] = (gf::CpuAnswer*) h;
        cudaHostGetDevicePointer(&W.ans_d[t], h, 0);
    }
    if (cudaHostAlloc((void**) &W.scr_h, 64, cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc((void**) &W.tok_h, TS * sizeof(int), cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc((void**) &W.emb_h, TS * E * f4, cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc((void**) &W.hop_h, TS * hcE * f4, cudaHostAllocPortable) != cudaSuccess ||
        (W.ev_hop == nullptr && cudaEventCreateWithFlags(&W.ev_hop, cudaEventDisableTiming) != cudaSuccess)) {
        cudaGetLastError();
        err = "glm window: the pinned window buffers did not allocate";
        return false;
    }
    if (W.skip_d == nullptr && cudaMalloc((void**) &W.skip_d, sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        W.skip_d = nullptr;
        err = "glm window: the skip flag did not allocate";
        return false;
    }
    cudaMemset(W.skip_d, 0, sizeof(int));
    if (W.from_h == nullptr) {
        int* h = nullptr;
        if (cudaHostAlloc((void**) &h, 64, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            err = "glm window: the skip word did not allocate";
            return false;
        }
        h[0] = h[1] = 1 << 30;
        W.from_h = h;
    }
    for (cudaEvent_t* set : {W.ev_tok, W.ev_pick})
        for (int i = 0; i < kPipeSlots; ++i) {
            cudaEvent_t& e = set[i];
            if (e != nullptr || cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess) continue;
            cudaGetLastError();
            e = nullptr;
            err = "glm window: the window's events did not create";
            return false;
        }
    if (getenv("STRATA_GLM_WIN_UNION") != nullptr) {
        const size_t nu = (size_t) (l1_ - l0_) * (size_t) tmax * 8;
        if (cudaMalloc((void**) &W.uid, nu * sizeof(int)) != cudaSuccess) {
            cudaGetLastError();
            W.uid = nullptr;
        }
        W.uid_h.assign(nu, -1);
    }
    W.tmax = tmax;
    return true;
}

// at setup, before the expert pool takes what is free: the window's buffers, the recurrent-state backup window_commit
// restores (the KDA states and conv histories of this part) and the drafter's capture rows
bool Glm5Model::win_reserve(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (!win_alloc(kWinMax, err)) return false;
    const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
    int n_rec = 0;
    for (int il = l0_; il < l1_; ++il) n_rec += g.is_recr(il);
    if (n_rec > 0 && kda_bak_ == nullptr && cudaMalloc(&kda_bak_, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        kda_bak_ = nullptr;
        err = "glm window: the recurrent-state backup did not allocate";
        return false;
    }
    if (n_rec > 0 && F->rep == nullptr &&
        cudaMalloc((void**) &F->rep, (size_t) gf::kKdaLayers * 5 * g.d_inner() * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        F->rep = nullptr;   // (the commit replays layer by layer)
    }
    if (F->cap == nullptr) {
        if (cudaMalloc((void**) &F->cap, (size_t) kWinMax * g.n_embd * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            F->cap = nullptr;
            err = "glm window: the drafter's capture rows did not allocate";
            return false;
        }
        F->cap_rows = kWinMax;
    }
    return true;
}

void Glm5Model::win_free() {
    win_crew_stop();
    if (fast_ == nullptr) return;
    auto& W = fast_->win;
    for (cudaEvent_t e : W.ev_tok)
        if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : W.ev_pick)
        if (e) cudaEventDestroy(e);
    if (W.skip_d) cudaFree(W.skip_d);
    if (W.from_h) cudaFreeHost((void*) W.from_h);
    if (W.arena) cudaFree(W.arena);
    if (W.tok_h) cudaFreeHost(W.tok_h);
    if (W.emb_h) cudaFreeHost(W.emb_h);
    if (W.hop_h) cudaFreeHost(W.hop_h);
    if (W.ev_hop) cudaEventDestroy(W.ev_hop);
    for (cudaEvent_t e : W.ev)
        if (e) cudaEventDestroy(e);
    if (W.uid) cudaFree(W.uid);
    if (W.scr_h) cudaFreeHost(W.scr_h);
    for (auto* a : W.ans_h)
        if (a) cudaFreeHost(a);
    W = FastState::Win{};
    if (fast_->cap) cudaFree(fast_->cap);
    fast_->cap = nullptr;
    fast_->cap_rows = 0;
    if (fast_->rep) cudaFree(fast_->rep);
    fast_->rep = nullptr;
    kda_bak_cur_ = false;
}

// tokens t0 .. t0 + T - 1 of the window (a staggered window's group; all of it: t0 0)
bool Glm5Model::fast_dsa_window(int il, int64_t p0, int t0, int T, std::string& err) {
    FastState* F = fast_;
    auto& W = F->win;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd;
    const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
    const auto cols = [&](gf::MvJob j, int64_t xq_s, int64_t xf_s, int64_t y_s) {
        j.ncol = T;
        j.xq_cstride = xq_s;
        j.xf_cstride = xf_s;
        j.y_cstride = y_s;
        return j;
    };
    gf::MvJob j[5];
    const void* xq0 = (const uint8_t*) W.xq + (size_t) t0 * W.xq_s;
    const float* x0 = W.x + (size_t) t0 * E;
    j[0] = cols({Ly.q_a.q, xq0, x0, W.qr_raw + (size_t) t0 * g.q_lora, nullptr, 1.0f, Ly.q_a.type, E, g.q_lora}, W.xq_s, E,
                g.q_lora);
    j[1] = cols({Ly.kv_a.q, xq0, x0, W.kv_raw + (size_t) t0 * g.kv_lora, nullptr, 1.0f, Ly.kv_a.type, E, g.kv_lora},
                W.xq_s, E, g.kv_lora);
    j[2] = cols({Ly.idx_k, nullptr, x0, W.ik_raw + (size_t) t0 * g.idx_key, nullptr, 1.0f, gf::kTypeBF16, E, g.idx_key}, 0,
                E, g.idx_key);
    j[3] = cols({Ly.idx_gate, nullptr, x0, W.ig_raw + (size_t) t0 * g.idx_key, nullptr, 1.0f, gf::kTypeBF16, E, g.idx_key},
                0, E, g.idx_key);
    j[4] = cols({Ly.idx_proj, nullptr, x0, W.iw + (size_t) t0 * g.idx_heads, nullptr, prescale, gf::kTypeBF16, E,
                 g.idx_heads}, 0, E, g.idx_heads);
    static const bool prof = getenv("STRATA_GLM_WIN_PROF") != nullptr;
    const auto mark = [&](const char* nm) {
        if (prof) W.mark(nm, s);
    };
    if (!gf::mv(j, 5, s)) { err = "glm window: dsa projections"; return false; }
    mark("dsa_proj_mv");
    // every token's caches first (a query reads only positions up to its own, and only completed pools)
    for (int t = t0; t < t0 + T; ++t) {
        gf::DsaPrepArgs d;
        d.qr_raw = W.qr_raw + (size_t) t * g.q_lora;
        d.q_a_norm = Ly.q_a_norm;
        d.qr = W.qr + (size_t) t * g.q_lora;
        d.qr_q = (uint8_t*) W.qr_q + (size_t) t * W.qrq_s;
        d.q_lora = g.q_lora;
        d.kv_raw = W.kv_raw + (size_t) t * g.kv_lora;
        d.kv_norm = Ly.kv_a_norm;
        kv_prep_targets(kv_, il, (uint16_t*) (state_ + dsa_lat_[(size_t) il]), d);   // (KV streaming: + host copy)
        d.kv_lora = g.kv_lora;
        d.lat_q8 = lat_q8_;
        d.ik_raw = W.ik_raw + (size_t) t * g.idx_key;
        d.k_norm_w = Ly.k_norm_w;
        d.k_norm_b = Ly.k_norm_b;
        d.ik_cache = state_ + dsa_ik_[(size_t) il];
        d.ig_raw = W.ig_raw + (size_t) t * g.idx_key;
        d.ig_cache = state_ + dsa_ig_[(size_t) il];
        d.ape = Ly.ape;
        d.pooled = state_ + dsa_pool_[(size_t) il];
        d.idx_key = g.idx_key;
        d.kpool = g.idx_kpool;
        d.ring = ik_ring_;
        d.p = (int) (p0 + t);
        d.eps = g.norm_eps;
        gf::dsa_prep(d, s);
    }
    mark("dsa_prep");
    gf::MvJob j2[2];
    const void* qrq0 = (const uint8_t*) W.qr_q + (size_t) t0 * W.qrq_s;
    const float* qr0 = W.qr + (size_t) t0 * g.q_lora;
    float* iq0 = W.iq + (size_t) t0 * g.idx_heads * g.idx_key;
    j2[0] = cols({Ly.q_b.q, qrq0, qr0, W.q + (size_t) t0 * g.n_head * g.qk_nope, nullptr, 1.0f, Ly.q_b.type, g.q_lora,
                  g.n_head * g.qk_nope}, W.qrq_s, g.q_lora, (int64_t) g.n_head * g.qk_nope);
    j2[1] = Ly.idx_q_b_x.q != nullptr
                ? cols({Ly.idx_q_b_x.q, qrq0, qr0, iq0, nullptr, 1.0f, Ly.idx_q_b_x.type, g.q_lora,
                        g.idx_heads * g.idx_key}, W.qrq_s, g.q_lora, (int64_t) g.idx_heads * g.idx_key)
                : cols({Ly.idx_q_b, nullptr, qr0, iq0, nullptr, 1.0f, gf::kTypeBF16, g.q_lora, g.idx_heads * g.idx_key},
                       0, g.q_lora, (int64_t) g.idx_heads * g.idx_key);
    if (!gf::mv(j2, 2, s)) { err = "glm window: dsa q"; return false; }
    mark("dsa_q_mv");
    // the absorbed MLA: wk_b . q of every token at once, each token's attention over its own selected cells, then
    // wv_b . ctx of every token at once (mla()'s arithmetic; the head-wise weights read once, not T times).
    // STRATA_GLM_WIN_MLA=0: mla() per token (A/B)
    static const bool mla_batch = [] {
        const char* v = getenv("STRATA_GLM_WIN_MLA");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const int64_t qd = (int64_t) g.n_head * g.qk_nope, ad = (int64_t) g.n_head * g.kv_lora;
    bool batched = mla_batch && W.q_abs != nullptr && T > 1;
    if (batched)
        gf::mla_q_abs(W.q + (size_t) t0 * qd, qd, Ly.k_b, g.n_head, g.qk_nope, g.kv_lora, T, W.q_abs + (size_t) t0 * ad,
                      ad, s);
    uint16_t* const lat = (uint16_t*) (state_ + dsa_lat_[(size_t) il]);
    // every token's selection into its own row of W.cells, then one batched attention (not under KV streaming: there a
    // token's blocks are made resident right before its own attention)
    const int64_t nsm = g.n_sel_max();
    thread_local std::vector<int> nsel;
    nsel.clear();
    const bool rows = batched && W.cells != nullptr && !kv_streamed(kv_, il);
    for (int t = t0; t < t0 + T; ++t) {
        const int64_t p = p0 + t;
        const float* iq = W.iq + (size_t) t * g.idx_heads * g.idx_key;
        const int pool_done = (int) ((p + 1) / g.idx_kpool);
        const bool all = pool_done <= g.top_pools_max() && gf::dsa_select_all_on();   // (as fast_dsa)
        if (pool_done > 0 && !all)
            gf::dsa_score(iq, state_ + dsa_pool_[(size_t) il], W.iw + (size_t) t * g.idx_heads, g.idx_key, g.idx_heads,
                          pool_done, F->score, s);
        const int top_pools = std::min(g.top_pools_max(), pool_done);
        const int n_sel = g.idx_kpool * top_pools + (g.idx_select_tail ? g.idx_kpool - 1 : 0);
        int* const sel = rows ? W.cells + (size_t) (t - t0) * nsm : F->cells;
        if (all) gf::dsa_select_all(pool_done, g.idx_kpool, n_sel, (int) p, sel, s);
        else gf::dsa_select(F->score, pool_done, g.idx_kpool, top_pools, n_sel, (int) p, sel, s);
        if (rows) {
            nsel.push_back(n_sel);
            continue;
        }
        // KV streaming: this token's blocks resident before its attention (the next token's resolve runs after it)
        const int* cells = kv_attend_cells(kv_, il, lat, F->cells, n_sel, F->slot_cells, s);
        if (batched && gf::mla_attend(W.q_abs + (size_t) t * ad, lat, cells, n_sel, g.n_head, g.qk_nope, g.kv_lora,
                                      W.ctx + (size_t) t * ad, s, lat_q8_))
            continue;
        if (batched) {   // (the first token: the split attention is not available here - every token per token)
            batched = false;
            if (t > t0) {
                err = "glm window: the batched MLA lost its attention mid-window";
                return false;
            }
        }
        gf::mla(W.q + (size_t) t * qd, Ly.k_b, Ly.v_b, lat, cells, n_sel, g.n_head, g.qk_nope, g.kv_lora, g.v_head,
                (uint8_t*) W.attn_q + (size_t) t * W.aq_s, s, lat_q8_);
    }
    if (rows && !gf::mla_attend_batch(W.q_abs + (size_t) t0 * ad, ad, lat, W.cells, nsm, nsel.data(), T, g.n_head,
                                      g.qk_nope, g.kv_lora, W.ctx + (size_t) t0 * ad, ad, s, lat_q8_)) {
        for (int t = t0; t < t0 + T; ++t) {   // (no tensor-core attention here: per token, as without the rows)
            const int* cells = W.cells + (size_t) (t - t0) * nsm;
            const int n_sel = nsel[(size_t) (t - t0)];
            if (batched && gf::mla_attend(W.q_abs + (size_t) t * ad, lat, cells, n_sel, g.n_head, g.qk_nope, g.kv_lora,
                                          W.ctx + (size_t) t * ad, s, lat_q8_))
                continue;
            if (batched) {
                batched = false;
                if (t > t0) {
                    err = "glm window: the batched MLA lost its attention mid-window";
                    return false;
                }
            }
            gf::mla(W.q + (size_t) t * qd, Ly.k_b, Ly.v_b, lat, cells, n_sel, g.n_head, g.qk_nope, g.kv_lora, g.v_head,
                    (uint8_t*) W.attn_q + (size_t) t * W.aq_s, s, lat_q8_);
        }
    }
    if (batched)
        gf::mla_out(W.ctx + (size_t) t0 * ad, ad, Ly.v_b, g.n_head, g.kv_lora, g.v_head, T,
                    (uint8_t*) W.attn_q + (size_t) t0 * W.aq_s, (int64_t) W.aq_s, s);
    mark("dsa_attn");
    gf::MvJob o = cols({Ly.out.q, (const uint8_t*) W.attn_q + (size_t) t0 * W.aq_s, nullptr, W.mixer + (size_t) t0 * E,
                        nullptr, 1.0f, Ly.out.type, g.n_head * g.v_head, E}, W.aq_s, 0, E);
    if (!gf::mv(&o, 1, s)) { err = "glm window: dsa output"; return false; }
    mark("dsa_out_mv");
    return true;
}

// the MoE FFN of every token: the router and the shared expert's gate/up for all T at once, then each token's route,
// tiers, routed experts and CPU lane in order - fast_moe's arithmetic without its prediction and look-ahead (they
// read ahead for the NEXT layer of the same token).  The plan, the misses' answer, the CPU lane's answer and the
// scratch slots are single: a token's are used up before the next token's route.
// tokens t0 .. t0 + T - 1; part 0: the whole FFN, 1: its front (router, routes, fetches), 2: its back (the experts and
// the CPU lane's answers) - a staggered window issues another group's attention between the two.  gid: the group
// (0 / 1: scratch slots 0-3 / 4-7 with a claim word each, so a group's next layer cannot reuse the other's slots)
bool Glm5Model::fast_moe_window(int il, int t0, int T, int part, int gid, std::string& err) {
    FastState* F = fast_;
    auto& W = F->win;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared;
    static const bool prof = getenv("STRATA_GLM_WIN_PROF") != nullptr;
    if (part != 2) {
        gf::MvJob j[3];
        const void* xq0 = (const uint8_t*) W.xq + (size_t) t0 * W.xq_s;
        j[0] = {Ly.router, nullptr, W.x + (size_t) t0 * E, W.rlog + (size_t) t0 * g.n_expert, nullptr, 1.0f, gf::kTypeBF16, E,
                g.n_expert};
        const float* x0 = W.x + (size_t) t0 * E;
        j[1] = {Ly.sh_gate.q, xq0, x0, W.sh_g + (size_t) t0 * FFs, nullptr, 1.0f, Ly.sh_gate.type, E, FFs};
        j[2] = {Ly.sh_up.q, xq0, x0, W.sh_u + (size_t) t0 * FFs, nullptr, 1.0f, Ly.sh_up.type, E, FFs};
        const int64_t ys[3] = {g.n_expert, FFs, FFs};
        for (int i = 0; i < 3; ++i) {
            j[i].ncol = T;
            j[i].xq_cstride = (int64_t) W.xq_s;
            j[i].xf_cstride = E;
            j[i].y_cstride = ys[i];
        }
        if (!gf::mv(j, 3, s)) { err = "glm window: router / shared expert"; return false; }
        if (prof) W.mark("moe_mv", s);
    }
    gf::MoeDev md = F->md;
    md.pf_src = F->pf_buf[il & 1];
    md.pf_dst = F->pf_buf[il & 1] + 8;
    md.pf_n = F->pf_n_buf[il & 1];
    namespace x3 = strata::kernels::exl3;
    const auto x3_args = [&](const gf::MoeDev& m) {
        x3::MoeArgs a;
        a.plan_ptr = m.plan_ptr;
        a.plan_w = m.plan_w;
        a.cpu_part = m.cpu_part;
        a.cpu_flag = m.cpu_flag;
        a.lay = Ly.xl;
        a.k = g.n_exp_used;
        a.n_embd = E;
        a.n_ff = g.n_ff_exp;
        a.limit = g.swiglu_exp;
        return a;
    };
    const bool lane = (F->cpu_plan != 0ull || F->helper.device() >= 0) && F->cpu_fmt[(size_t) il].n_ff > 0;
    // STRATA_GLM_TRUNK_KEEP / _MAP: a window token leaves out the same experts the token path would (8: none)
    const int keep = il == mtp_il_ ? 8 : glmfast::trunk_skip_from(il);
    static const int promote_min = [] {
        const char* v = getenv("STRATA_GLM_PROMOTE_MIN");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    const float* ah_bias[gf::kAhead] = {nullptr, nullptr, nullptr, nullptr};
    // the batched window's routes on two window lanes (both sockets' pools answer it, win_lane): every RAM-tier expert
    // to the CPU - the calibrated plan priced one pool against PCIe, and a merged layer has two (a 3090's 2.1 ms PCIe
    // fetch is several experts' CPU time there).  STRATA_GLM_WIN_ALLCPU=0: the calibrated plan
    static const bool all_cpu_env = [] {
        const char* v = getenv("STRATA_GLM_WIN_ALLCPU");
        return v == nullptr || std::atoi(v) != 0;
    }();
    unsigned long long win_plan = F->cpu_plan;
    if (all_cpu_env && F->cpu_plan != 0ull && F->win_lane[0] != nullptr && F->win_lane[1] != nullptr) {
        win_plan = 0ull;
        for (int f = 1; f <= 8; ++f) win_plan |= (unsigned long long) f << (4 * f);
    }
    // the batched MoE: every token's route, tiers and fetch first (each into its own plan, CPU answer slot and seq
    // word; the scratch slots once each), then each distinct expert once for all of them (gf::moe_win_experts), then
    // the CPU lane's answers - which ran while the GPU computed.  STRATA_GLM_WIN_BATCH=0: token by token (A/B).  Off by
    // default on AMD: the batched experts' kernels are the generic ones, the token path's are RDNA3's (the Halo APUs'
    // expert kernels) - a Radeon 8065S's MTP decode (Maya-S, 256-token answers) ran 26.7 / 25.7 tok/s token by token,
    // 22.7 / 22.3 batched (and 17.2 / 16.5 batched and staggered, below), the same tokens
    static const bool batch_env = glmfast::off_by_default_on_amd("STRATA_GLM_WIN_BATCH");
    static const bool prof_b = getenv("STRATA_GLM_WIN_PROF") != nullptr;
    // ... and with it off (AMD): the routes as the batched MoE takes them (each token its own plan), then every token's
    // experts on the token path's kernels in one gate/up and one down launch (gf::moe_experts_multi) - each token's
    // arithmetic its own, the blocks of all the tokens for a chunk of rows side by side, so an expert two of them route
    // is read from DRAM once (a Radeon 8065S's window layer: ~19 distinct experts of ~24 routed).
    // STRATA_GLM_WIN_MULTI=0: one token after another (A/B)
    static const bool multi_env = [] {
        const char* v = getenv("STRATA_GLM_WIN_MULTI");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool multi = !batch_env && multi_env && !Ly.exl3 && part == 0 && T > 1;
    if ((batch_env || multi) && !W.per_token && W.mplan_ptr != nullptr && (T > 1 || part != 0)) {
        const auto tok_dev = [&](int t) {
            gf::MoeDev mt = md;
            mt.plan_ptr = W.mplan_ptr + 8 * t;
            mt.plan_w = W.mplan_w + 8 * t;
            mt.plan_id = W.mplan_id + 8 * t;
            mt.fetch_src = W.mfetch + 8 * t;
            mt.cpu_part = W.mcpu_part + (size_t) E * t;
            mt.cpu_flag = W.mcpu_flag + t;
            if (t > 0 && md.cpu_seq != nullptr) {
                mt.cpu_seq = W.cpu_seq_w + 16 * t;
                mt.cpu_ans = W.ans_d[t];
            }
            return mt;
        };
        // this layer's scratch claims (word 1 stays sticky): all 8 slots free, or the group's 4 (words 8 / 9 hold the
        // claim words that leave a group only its own slots free)
        unsigned int* scr = W.scr + (gid == 1 ? 2 : 0);
        if (part != 2) {
        if (part == 0) cudaMemsetAsync(W.scr, 0, sizeof(unsigned int), s);
        else cudaMemcpyAsync(scr, W.scr + 8 + gid, sizeof(unsigned int), cudaMemcpyDeviceToDevice, s);
        for (int t = t0; t < t0 + T; ++t) {
            const gf::MoeDev mt = tok_dev(t);
            gf::moe_route(W.rlog + (size_t) t * g.n_expert, Ly.router_bias, g.n_expert, g.n_exp_used, g.w_scale,
                          g.norm_w != 0, il, W.x + (size_t) t * E, E, mt, W.sh_g + (size_t) t * FFs,
                          W.sh_u + (size_t) t * FFs, g.swiglu_shexp, FFs, (uint8_t*) W.sh_hq + (size_t) t * W.sh_hq_s, s,
                          nullptr, nullptr, 0, nullptr, ah_bias, 0, keep, lane ? win_plan : 0ull, promote_min, 0, 0, t,
                          scr, t0 + T);
            ++F->expected;
            // (every expert in VRAM: the wait and the fetch are no-ops - fast_moe's all_resident, 2 launches a token)
            if (!F->all_resident) {
                gf::moe_wait(mt, E, s);
                gf::moe_fetch(mt, g.n_exp_used, Ly.blob, s);
            }
            if (W.uid != nullptr)
                cudaMemcpyAsync(W.uid + ((size_t) (il - l0_) * kWinMax + t) * 8, mt.plan_id, 8 * sizeof(int),
                                cudaMemcpyDeviceToDevice, s);
        }
        if (prof_b) W.mark("moe_routes", s);
        }
        if (part == 1) return true;
        if (multi) {
            // (the shared expert's down rides the gate/up launch, from each token's h the route wrote - moe_gate_up's)
            constexpr size_t q8 = 36;   // (a q8_1 block's bytes, as win_alloc's)
            gf::MoeDev mm = tok_dev(t0);
            mm.mt_T = T;
            mm.mt_xq_s = (unsigned int) (W.xq_s / q8);
            mm.mt_hq_s = (unsigned int) (8 * (g.n_ff_exp / 32));
            mm.mt_sh_hq_s = (unsigned int) (W.sh_hq_s / q8);
            mm.mt_out_s = (unsigned int) E;
            mm.mt_cpu_s = (unsigned int) E;
            const size_t hq_t = 8 * ((size_t) g.n_ff_exp / 32) * q8;
            const bool sh = Ly.sh_down.q != nullptr;
            if (!gf::moe_experts_multi(Ly.gu_type, Ly.d_type, mm, g.n_exp_used, E, g.n_ff_exp, g.swiglu_exp,
                                       Ly.down_off, (const uint8_t*) W.xq + (size_t) t0 * W.xq_s,
                                       (uint8_t*) W.whq + (size_t) t0 * hq_t, Ly.sh_down.q, Ly.sh_down.type,
                                       sh ? (const uint8_t*) W.sh_hq + (size_t) t0 * W.sh_hq_s : nullptr, FFs,
                                       sh ? W.sh_out + (size_t) t0 * E : nullptr, W.ffn + (size_t) t0 * E, s)) {
                // (not here: each token's on its own launches, the same kernels)
                for (int t = t0; t < t0 + T; ++t) {
                    const gf::MoeDev mt = tok_dev(t);
                    uint8_t* hq = (uint8_t*) W.whq + (size_t) t * hq_t;
                    float* so = sh ? W.sh_out + (size_t) t * E : nullptr;
                    gf::moe_gate_up(Ly.gu_type, mt, g.n_exp_used, E, g.n_ff_exp, g.swiglu_exp,
                                    (const uint8_t*) W.xq + (size_t) t * W.xq_s, hq, Ly.sh_down.q, Ly.sh_down.type,
                                    sh ? (const uint8_t*) W.sh_hq + (size_t) t * W.sh_hq_s : nullptr, FFs, so, s);
                    gf::moe_down(Ly.d_type, mt, g.n_exp_used, E, g.n_ff_exp, Ly.down_off, hq, so,
                                 W.ffn + (size_t) t * E, s);
                }
            }
            if (prof_b) W.mark("moe_experts", s);
            if (lane)
                for (int t = t0; t < t0 + T; ++t) gf::moe_cpu_wait(tok_dev(t), E, W.ffn + (size_t) t * E, s);
            if (prof_b) W.mark("moe_cpu_wait", s);
            return true;
        }
        const float* sh_out = nullptr;
        if (Ly.sh_down.q != nullptr) {
            gf::MvJob jd{Ly.sh_down.q, (const uint8_t*) W.sh_hq + (size_t) t0 * W.sh_hq_s, nullptr, W.sh_out + (size_t) t0 * E,
                         nullptr, 1.0f, Ly.sh_down.type, FFs, E};
            jd.ncol = T;
            jd.xq_cstride = (int64_t) W.sh_hq_s;
            jd.y_cstride = E;
            if (!gf::mv(&jd, 1, s)) { err = "glm window: shared expert down"; return false; }
            sh_out = W.sh_out + (size_t) t0 * E;
        }
        // EXL3: each distinct expert once for all its tokens (exl3::win_experts, gf::moe_win_experts' groups);
        // STRATA_GLM_WIN_X3=0: each token's routed experts on the token path's kernels (A/B - the same bits)
        static const bool x3_batch = [] {
            const char* v = getenv("STRATA_GLM_WIN_X3");
            return v == nullptr || std::atoi(v) != 0;
        }();
        if (Ly.exl3 && x3_batch && W.wh != nullptr && x3::several_ok()) {
            gf::MoeWin mw;
            mw.plan_ptr = W.mplan_ptr + 8 * t0;
            mw.grp_ptr = W.grp_ptr;
            mw.grp_start = W.grp_start;
            mw.grp_ent = W.grp_ent;
            mw.n_groups = W.n_groups;
            mw.T = T;
            mw.k = g.n_exp_used;
            gf::moe_win_group(mw, s);
            x3::WinArgs xa;
            xa.grp_ptr = W.grp_ptr;
            xa.grp_start = W.grp_start;
            xa.grp_ent = W.grp_ent;
            xa.n_groups = W.n_groups;
            xa.plan_ptr = W.mplan_ptr + 8 * t0;
            xa.plan_w = W.mplan_w + 8 * t0;
            xa.cpu_part = W.mcpu_part + (size_t) E * t0;
            xa.cpu_flag = W.mcpu_flag + t0;
            xa.x = W.x + (size_t) t0 * E;
            xa.x_stride = (size_t) E;
            xa.h = W.wh;
            xa.part = W.wpart;
            xa.sh_out = sh_out;
            xa.out = W.ffn + (size_t) t0 * E;
            xa.out_stride = (size_t) E;
            xa.lay = Ly.xl;
            xa.T = T;
            xa.k = g.n_exp_used;
            xa.n_embd = E;
            xa.n_ff = g.n_ff_exp;
            xa.limit = g.swiglu_exp;
            if (!x3::win_experts(xa, s)) {
                err = "glm window: the batched EXL3 experts";
                return false;
            }
        } else if (Ly.exl3) {
            // EXL3: each token's routed experts on the token path's kernels (its own plan, F32 input and host part)
            for (int t = t0; t < t0 + T; ++t) {
                const x3::MoeArgs xa = x3_args(tok_dev(t));
                x3::moe_gate_up(xa, W.x + (size_t) t * E, nullptr, s);
                x3::moe_down(xa, nullptr, sh_out != nullptr ? sh_out + (size_t) (t - t0) * E : nullptr,
                             W.ffn + (size_t) t * E, s);
            }
        } else {
            gf::MoeWin mw;
            mw.plan_ptr = W.mplan_ptr + 8 * t0;
            mw.plan_w = W.mplan_w + 8 * t0;
            mw.cpu_part = W.mcpu_part + (size_t) E * t0;
            mw.cpu_flag = W.mcpu_flag + t0;
            mw.xq = (const uint8_t*) W.xq + (size_t) t0 * W.xq_s;
            mw.xq_stride = W.xq_s;
            mw.hq = W.whq;
            mw.part = W.wpart;
            mw.sh_out = sh_out;
            mw.out = W.ffn + (size_t) t0 * E;
            mw.out_stride = (size_t) E;
            mw.grp_ptr = W.grp_ptr;
            mw.grp_start = W.grp_start;
            mw.grp_ent = W.grp_ent;
            mw.n_groups = W.n_groups;
            mw.T = T;
            mw.k = g.n_exp_used;
            mw.n_embd = E;
            mw.n_ff = g.n_ff_exp;
            mw.limit = g.swiglu_exp;
            mw.down_off = Ly.down_off;
            if (!gf::moe_win_experts(Ly.gu_type, Ly.d_type, mw, s)) {
                err = "glm window: the batched experts (types " + std::to_string(Ly.gu_type) + " / " +
                      std::to_string(Ly.d_type) + ")";
                return false;
            }
        }
        if (prof_b) W.mark("moe_experts", s);
        if (lane)
            for (int t = t0; t < t0 + T; ++t) gf::moe_cpu_wait(tok_dev(t), E, W.ffn + (size_t) t * E, s);
        if (prof_b) W.mark("moe_cpu_wait", s);
        return true;
    }
    if (part != 0) {
        err = "glm window: a staggered group on the token-by-token path";
        return false;
    }
    for (int t = t0; t < t0 + T; ++t) {
        const float* x = W.x + (size_t) t * E;
        const void* xq = (const uint8_t*) W.xq + (size_t) t * W.xq_s;
        float* ffn = W.ffn + (size_t) t * E;
        gf::moe_route(W.rlog + (size_t) t * g.n_expert, Ly.router_bias, g.n_expert, g.n_exp_used, g.w_scale,
                      g.norm_w != 0, il, x, E, md, W.sh_g + (size_t) t * FFs, W.sh_u + (size_t) t * FFs,
                      g.swiglu_shexp, FFs, F->sh_hq, s, nullptr, nullptr, 0, nullptr, ah_bias, 0, keep,
                      lane ? F->cpu_plan : 0ull, promote_min);
        ++F->expected;
        if (!F->all_resident) {
            gf::moe_wait(md, E, s);
            gf::moe_fetch(md, g.n_exp_used, Ly.blob, s);
        }
        if (Ly.exl3) {
            // (fast_moe_experts' EXL3 order: the shared expert's down first, then the routed experts over the F32 input)
            if (Ly.sh_down.q != nullptr) {
                const gf::MvJob sd = {Ly.sh_down.q, F->sh_hq, nullptr, F->sh_out, nullptr, 1.0f, Ly.sh_down.type, FFs, E};
                if (!gf::mv(&sd, 1, s)) { err = "glm window: shared expert down"; return false; }
            }
            const x3::MoeArgs xa = x3_args(md);
            x3::moe_gate_up(xa, x, nullptr, s);
            x3::moe_down(xa, nullptr, Ly.sh_down.q != nullptr ? F->sh_out : nullptr, ffn, s);
        } else {
            gf::moe_gate_up(Ly.gu_type, md, g.n_exp_used, E, g.n_ff_exp, g.swiglu_exp, xq, F->hq, Ly.sh_down.q,
                            Ly.sh_down.type, F->sh_hq, FFs, F->sh_out, s);
            gf::moe_down(Ly.d_type, md, g.n_exp_used, E, g.n_ff_exp, Ly.down_off, F->hq,
                         Ly.sh_down.q != nullptr ? F->sh_out : nullptr, ffn, s);
        }
        if (lane) gf::moe_cpu_wait(md, E, ffn, s);
        if (W.uid != nullptr)
            cudaMemcpyAsync(W.uid + ((size_t) (il - l0_) * kWinMax + t) * 8, md.plan_id, 8 * sizeof(int),
                            cudaMemcpyDeviceToDevice, s);
    }
    if (prof) W.mark("moe_tokens", s);
    return true;
}

bool Glm5Model::fast_window_layers(int64_t p0, int T, std::string& err) {
    FastState* F = fast_;
    auto& W = F->win;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int E = g.n_embd, DI = g.d_inner();
    const size_t hcE = (size_t) g.hc * E;
    // W.R[rc] holds the rows a layer half reads, W.R[rc ^ 1] the rows it writes: each layer's indices, for every token
    // alike (rc_in: the attention half's read; rc_mid: what the FFN half reads; the next layer reads rc_mid ^ 1)
    std::vector<int> rc_in((size_t) (l1_ - l0_) + 1), rc_mid((size_t) (l1_ - l0_));
    {
        int rc = 0;
        for (int il = l0_; il < l1_; ++il) {
            rc_in[(size_t) (il - l0_)] = rc;
            if (il != l0_) rc ^= 1;
            rc_mid[(size_t) (il - l0_)] = rc;
            rc ^= 1;
        }
        rc_in[(size_t) (l1_ - l0_)] = rc;
    }
    const auto cols = [&](gf::MvJob& j, int n, int64_t xq_s, int64_t xf_s, int64_t y_s) {
        j.ncol = n;
        j.xq_cstride = xq_s;
        j.xf_cstride = xf_s;
        j.y_cstride = y_s;
    };
    const auto hc_rows = [&](int il, bool attn, int rc, int t0, int n) {
        const auto& Ly = F->L[(size_t) il];
        for (int t = t0; t < t0 + n; ++t) {
            gf::HcArgs h;
            h.block_out = attn ? (il == l0_ ? nullptr : W.ffn + (size_t) t * E) : W.mixer + (size_t) t * E;
            h.R_old = W.R[rc] + (size_t) t * hcE;
            h.R_new = W.R[rc ^ 1] + (size_t) t * hcE;
            h.post_in = W.post + (size_t) t * 8;
            h.comb_in = W.comb + (size_t) t * 16;
            h.pre = W.pre + (size_t) t * 8;
            h.post = W.post + (size_t) t * 8;
            h.comb = W.comb + (size_t) t * 16;
            h.w_fn = attn ? Ly.hc_attn_fn : Ly.hc_ffn_fn;
            h.w_scale = attn ? Ly.hc_attn_scale : Ly.hc_ffn_scale;
            h.w_base = attn ? Ly.hc_attn_base : Ly.hc_ffn_base;
            h.norm_w = attn ? Ly.attn_norm : Ly.ffn_norm;
            h.norm_eps = g.norm_eps;
            h.hc_eps = g.hc_eps;
            h.iters = g.sinkhorn_iters;
            h.n_embd = E;
            h.x = W.x + (size_t) t * E;
            h.xq = (uint8_t*) W.xq + (size_t) t * W.xq_s;
            h.part = F->part;
            h.counter = F->counter;
            gf::hc(h, s);
        }
    };
    static const bool prof = getenv("STRATA_GLM_WIN_PROF") != nullptr;
    const auto mark = [&](const char* nm) {
        if (prof) W.mark(nm, s);
    };
    // a layer's first half for tokens t0 .. t0 + n - 1: the attention side, the FFN-side read, and the FFN up to its
    // routes (moe part 1) - or all of it (moe part 0 / a dense FFN).  The second half (moe part 2): the experts and the
    // CPU lane's answers.
    const auto pre = [&](int il, int t0, int n, int moe_part, int gid) -> bool {
        const auto& Ly = F->L[(size_t) il];
        const int k = il - l0_;
        hc_rows(il, true, rc_in[(size_t) k], t0, n);
        mark("hc");
        // the layer's input rows - the drafter's features when it reads this layer
        dflash_capture(il, W.R[rc_mid[(size_t) k]] + (size_t) t0 * hcE, n, p0 + t0, F->cap, F->cap_rows, s);
        mark("capture");
        if (Ly.recr) {
            float* base = W.kin + (size_t) W.kin_of[(size_t) il] * W.kin_layer;
            const KdaIn k0 = kda_in(g, base, W.tmax, t0);
            const void* xq0 = (const uint8_t*) W.xq + (size_t) t0 * W.xq_s;
            const float* x0 = W.x + (size_t) t0 * E;
            gf::MvJob j[6];
            j[0] = {Ly.q.q, xq0, x0, k0.proj[0], nullptr, 1.0f, Ly.q.type, E, DI};
            j[1] = {Ly.k.q, xq0, x0, k0.proj[1], nullptr, 1.0f, Ly.k.type, E, DI};
            j[2] = {Ly.v.q, xq0, x0, k0.proj[2], nullptr, 1.0f, Ly.v.type, E, DI};
            j[3] = {Ly.f_a, nullptr, x0, k0.fa, nullptr, 1.0f, gf::kTypeBF16, E, g.kda_head_dim};
            j[4] = {Ly.g_a, nullptr, x0, k0.ga, nullptr, 1.0f, gf::kTypeBF16, E, g.kda_head_dim};
            j[5] = {Ly.beta, nullptr, x0, k0.beta, nullptr, 1.0f, gf::kTypeBF16, E, g.n_head};
            const int64_t ys[6] = {DI, DI, DI, g.kda_head_dim, g.kda_head_dim, g.n_head};
            for (int i = 0; i < 6; ++i) cols(j[i], n, (int64_t) W.xq_s, E, ys[i]);
            if (!gf::mv(j, 6, s)) { err = "glm window: kda projections"; return false; }
            mark("kda_proj_mv");
            for (int t = t0; t < t0 + n; ++t)
                kda_token(F, Ly, g, kda_in(g, base, W.tmax, t), state_ + kda_conv_[(size_t) il],
                          state_ + kda_S_[(size_t) il], (uint8_t*) W.gated_q + (size_t) t * W.gq_s, s);
            mark("kda_tokens");
            gf::MvJob o = {Ly.out.q, (const uint8_t*) W.gated_q + (size_t) t0 * W.gq_s, nullptr, W.mixer + (size_t) t0 * E,
                           nullptr, 1.0f, Ly.out.type, DI, E};
            cols(o, n, (int64_t) W.gq_s, 0, E);
            if (!gf::mv(&o, 1, s)) { err = "glm window: kda output"; return false; }
            mark("kda_out_mv");
        } else {
            if (!fast_dsa_window(il, p0, t0, n, err)) return false;
        }
        // ---- the FFN-side read (folds the mixer's write in)
        hc_rows(il, false, rc_mid[(size_t) k], t0, n);
        mark("hc");
        if (!Ly.moe) {
            const void* xq0 = (const uint8_t*) W.xq + (size_t) t0 * W.xq_s;
            const float* x0 = W.x + (size_t) t0 * E;
            gf::MvJob j[2];
            j[0] = {Ly.ffn_gate.q, xq0, x0, W.dg + (size_t) t0 * g.n_ff_dense, nullptr, 1.0f, Ly.ffn_gate.type, E,
                    g.n_ff_dense};
            j[1] = {Ly.ffn_up.q, xq0, x0, W.du + (size_t) t0 * g.n_ff_dense, nullptr, 1.0f, Ly.ffn_up.type, E,
                    g.n_ff_dense};
            for (auto& jj : j) cols(jj, n, (int64_t) W.xq_s, E, g.n_ff_dense);
            if (!gf::mv(j, 2, s)) { err = "glm window: dense ffn"; return false; }
            for (int t = t0; t < t0 + n; ++t)
                gf::swiglu_q8(W.dg + (size_t) t * g.n_ff_dense, W.du + (size_t) t * g.n_ff_dense, g.swiglu_shexp,
                              g.n_ff_dense, (uint8_t*) W.dhq + (size_t) t * W.dhq_s, s);
            gf::MvJob dn = {Ly.ffn_down.q, (const uint8_t*) W.dhq + (size_t) t0 * W.dhq_s, nullptr, W.ffn + (size_t) t0 * E,
                            nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense, E};
            cols(dn, n, (int64_t) W.dhq_s, 0, E);
            if (!gf::mv(&dn, 1, s)) { err = "glm window: dense ffn down"; return false; }
            mark("dense_ffn");
            return true;
        }
        return fast_moe_window(il, t0, n, moe_part, gid, err);
    };
    const auto post = [&](int il, int t0, int n, int gid) -> bool {
        if (!F->L[(size_t) il].moe) return true;
        return fast_moe_window(il, t0, n, 2, gid, err);
    };
    mark("start");
    // STAGGERED (a window of 2+ tokens, the batched MoE): the tokens in two groups A, B, one layer's halves
    // interleaved - pre(l, A) pre(l, B) post(l, A) pre(l + 1, A) post(l, B) pre(l + 1, B) .. - so while the CPU lane
    // answers one group's experts the device runs the other group's attention and routes, where it used to wait.
    // One stream: the order alone keeps every dependency (B's attention reads A's caches and recurrent state at the
    // layer; the routes reach the host in order A(l), B(l), A(l + 1), ..), and each token's arithmetic is the same.
    // STRATA_GLM_WIN_STAGGER=0: all the tokens a layer at a time (A/B)
    static const bool stagger_env = [] {
        const char* v = getenv("STRATA_GLM_WIN_STAGGER");
        return v == nullptr || std::atoi(v) != 0;
    }();
    static const bool batch_env = glmfast::off_by_default_on_amd("STRATA_GLM_WIN_BATCH");   // (fast_moe_window's)
    const bool stagger = stagger_env && batch_env && T >= 2 && !W.per_token && W.mplan_ptr != nullptr;
    if (stagger) {
        const int ka = (T + 1) / 2, kb = T - ka;
        if (!pre(l0_, 0, ka, 1, 0) || !pre(l0_, ka, kb, 1, 1)) return false;
        for (int il = l0_; il < l1_; ++il) {
            if (!post(il, 0, ka, 0)) return false;
            if (il + 1 < l1_ && !pre(il + 1, 0, ka, 1, 0)) return false;
            if (!post(il, ka, kb, 1)) return false;
            if (il + 1 < l1_ && !pre(il + 1, ka, kb, 1, 1)) return false;
        }
    } else {
        for (int il = l0_; il < l1_; ++il)
            if (!pre(il, 0, T, 0, 0)) return false;
    }
    // the last layer's write half: R (in the other buffer) = post x ffn + comb . R
    const int rc = rc_in[(size_t) (l1_ - l0_)];
    for (int t = 0; t < T; ++t)
        gf::hc_post(W.ffn + (size_t) t * E, W.R[rc] + (size_t) t * hcE, W.post + (size_t) t * 8,
                    W.comb + (size_t) t * 16, E, W.R[rc ^ 1] + (size_t) t * hcE, s);
    W.out = rc ^ 1;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm window: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Glm5Model::window(const int32_t* tokens, int T, int32_t* out, std::string& err,
                       const strata::kernels::SamplerParams* sp) {
    if (fast_ == nullptr || T < 1 || T > kWinMax) {
        err = "glm window: needs the fast path and 1.." + std::to_string(kWinMax) + " tokens";
        return false;
    }
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    const size_t hcE = (size_t) g.hc * E;
    const int64_t p0 = pos_;
    if (p0 + T > max_ctx_) {
        err = "glm window: past the context";
        return false;
    }
    static const bool hprof = getenv("STRATA_GLM_WIN_PROF") != nullptr;
    using hclk = std::chrono::steady_clock;
    auto h0 = hclk::now();
    const auto hlap = [&](double& acc) {
        if (!hprof) return;
        const auto t = hclk::now();
        acc += std::chrono::duration<double, std::milli>(t - h0).count();
        h0 = t;
    };
    auto& WH = fast_->win;
    // between tokens: every device idle, the tiers' edits in; the recurrent states saved (window_commit restores them)
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        cudaStreamSynchronize(m->fast_->cs);
        if (!m->win_alloc(kWinMax, err)) return false;
        if (!m->fast_boundary(err)) return false;
        // (a window of one token keeps it: nothing to restore)
        if (T > 1 && !m->spec_kda_copy(false)) {
            err = "glm window: the recurrent-state backup did not allocate";
            return false;
        }
        m->fast_->win.p0 = p0;
        m->fast_->win.T = T;
        m->fast_->win.piped = false;
        m->fast_->win.per_token = false;
        auto& Wm = m->fast_->win;
        if (Wm.scr != nullptr) {
            // the claim and sticky words of both groups; words 8 / 9: the claim words that leave group A slots 0-3
            // and group B slots 4-7 (a staggered window's layer halves: fast_moe_window)
            cudaMemsetAsync(Wm.scr, 0, 4 * sizeof(unsigned int), m->fast_->cs);
            if (Wm.scr_h != nullptr) {
                // (bits 8-15: the group's own slots, where a clash's fallback stays - moe_route)
                Wm.scr_h[8] = 0xF0u | (0x0Fu << 8);
                Wm.scr_h[9] = 0x0Fu | (0xF0u << 8);
                cudaMemcpyAsync(Wm.scr + 8, Wm.scr_h + 8, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice,
                                m->fast_->cs);
            }
        }
    }
    cudaSetDevice(dev_);
    hlap(WH.h_setup);
    // a batched pass whose tokens took more scratch slots than a layer has runs again token by token (rare: the tiers
    // land a fetched expert in a spare slot first)
    for (int attempt = 0;; ++attempt) {
    // the embedded rows (host-dequantized from the shard mapping) -> the 4 streams of each token
    {
        FastState* F = fast_;
        auto& W = F->win;
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
        if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
            err = "glm window: no embedding dequantizer";
            return false;
        }
        for (int t = 0; t < T; ++t) {
            float* row = W.emb_h + (size_t) t * E;
            tt->to_float(pack_emb_src_ + (size_t) tokens[t] * ggml_row_size((ggml_type) pack_emb_type_, E), row, E);
            if (const float* img = image_row(p0 + t)) std::memcpy(row, img, (size_t) E * sizeof(float));
        }
        cudaMemcpyAsync(W.emb, W.emb_h, (size_t) T * E * sizeof(float), cudaMemcpyHostToDevice, F->cs);
        for (int t = 0; t < T; ++t)
            gf::embed_streams(W.emb + (size_t) t * E, W.R[0] + (size_t) t * hcE, E, F->cs);
    }
    if (!fast_window_layers(p0, T, err)) return false;
    // each later part: the rows cross over through the previous part's pinned buffer, ordered on the devices
    Glm5Model* tail = this;
    for (Glm5Model* B = split_next_.get(); B != nullptr; B = B->split_next_.get()) {
        auto& WA = tail->fast_->win;
        auto& WB = B->fast_->win;
        const size_t bytes = (size_t) T * hcE * sizeof(float);
        cudaSetDevice(tail->dev_);
        cudaMemcpyAsync(WA.hop_h, WA.R[WA.out], bytes, cudaMemcpyDeviceToHost, tail->fast_->cs);
        cudaEventRecord(WA.ev_hop, tail->fast_->cs);
        cudaSetDevice(B->dev_);
        cudaStreamWaitEvent(B->fast_->cs, WA.ev_hop, 0);
        cudaMemcpyAsync(WB.R[0], WA.hop_h, bytes, cudaMemcpyHostToDevice, B->fast_->cs);
        if (!B->fast_window_layers(p0, T, err)) return false;
        tail = B;
    }
    // the head on the last part: every token's final hidden state, the output GEMV once for all, each pick
    FastState* FT = tail->fast_;
    auto& WT = FT->win;
    cudaSetDevice(tail->dev_);
    for (int t = 0; t < T; ++t)
        gf::head_prep(WT.R[WT.out] + (size_t) t * hcE, tail->w_.at("output_norm.weight"), g.norm_eps, E,
                      WT.head_x + (size_t) t * E, (uint8_t*) WT.head_xq + (size_t) t * WT.xq_s, FT->cs);
    const WSlot& ow = tail->ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, WT.head_xq, WT.head_x, WT.logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    o.ncol = T;
    o.xq_cstride = (int64_t) WT.xq_s;
    o.xf_cstride = E;
    o.y_cstride = g.n_vocab;
    if (!gf::mv(&o, 1, FT->cs)) {
        err = "glm window: output head (type " + std::to_string(ow.type) + ")";
        return false;
    }
    if (sp != nullptr && !sp->greedy) {
        // row t is drawn with the counter the token loop would use for the token after tokens[t]
        strata::kernels::sample_tokens(WT.logits, T, (int) g.n_vocab, nullptr, 0, *sp, WT.tok, FT->cs);
    } else {
        for (int t = 0; t < T; ++t) gf::argmax(WT.logits + (size_t) t * g.n_vocab, g.n_vocab, WT.tok + t, FT->cs);
    }
    cudaMemcpyAsync(WT.tok_h, WT.tok, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost, FT->cs);
    hlap(WH.h_enqueue);
    // every part done (their captures for the drafter too)
    bool clash = false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        auto& Wm = m->fast_->win;
        if (Wm.scr != nullptr && Wm.scr_h != nullptr)   // both groups' sticky words: [1], [3]
            cudaMemcpyAsync(Wm.scr_h, Wm.scr + 1, 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost, m->fast_->cs);
        cudaStreamSynchronize(m->fast_->cs);
        if (Wm.scr_h != nullptr && ((Wm.scr_h[0] | Wm.scr_h[2]) & 0x80000000u) != 0u) clash = true;
    }
    if (clash && attempt == 0) {
        static bool told = false;
        if (!told) {
            told = true;
            std::fprintf(stderr, "glm window: a window's tokens took more scratch slots than a layer has - that window "
                                 "runs again token by token (said once)\n");
        }
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            cudaSetDevice(m->dev_);
            m->spec_kda_copy(true);
            m->fast_->win.per_token = true;
            if (m->fast_->win.scr != nullptr) cudaMemsetAsync(m->fast_->win.scr, 0, 4 * sizeof(unsigned int), m->fast_->cs);
            m->fast_->win.ev_n = 0;
        }
        cudaSetDevice(dev_);
        continue;
    }
    break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm window: ") + cudaGetErrorString(e);
        return false;
    }
    if (gf::launch_errors() > 0) {
        err = "glm window: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    hlap(WH.h_wait);
    {
        const Glm5Model* last = this;
        while (last->split_next_) last = last->split_next_.get();
        for (int t = 0; t < T; ++t) out[t] = last->fast_->win.tok_h[t];
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        m->pos_ = p0 + T;
        auto& W = m->fast_->win;
        if (W.uid != nullptr) {
            cudaMemcpy(W.uid_h.data(), W.uid, W.uid_h.size() * sizeof(int), cudaMemcpyDeviceToHost);
            for (int il = m->l0_; il < m->l1_; ++il) {
                if (!m->fast_->L[(size_t) il].moe) continue;
                std::vector<int> ids;
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < g.n_exp_used; ++i)
                        ids.push_back(W.uid_h[((size_t) (il - m->l0_) * kWinMax + t) * 8 + i]);
                W.u_picks += (int64_t) ids.size();
                std::sort(ids.begin(), ids.end());
                W.u_distinct += (int64_t) (std::unique(ids.begin(), ids.end()) - ids.begin());
                ++W.u_layers;
            }
        }
        if (W.ev_n == 0) continue;
        for (size_t i = 1; i < W.ev_n; ++i) {
            float a = 0.0f;
            if (cudaEventElapsedTime(&a, W.ev[i - 1], W.ev[i]) == cudaSuccess) W.ev_ms[W.ev_name[i]] += a;
        }
        W.ev_n = 0;
        ++W.n_prof;
    }
    last_tok_ = out[T - 1];
    cudaSetDevice(dev_);
    hlap(WH.h_after);
    if (hprof) ++WH.h_n;
    return true;
}

void Glm5Model::win_kda_rows(int il, int t, float** proj, float** fa, float** ga, float** beta) {
    auto& W = fast_->win;
    if (W.kin == nullptr || il < 0 || il >= (int) W.kin_of.size() || W.kin_of[(size_t) il] < 0 || t < 0 ||
        t >= W.tmax)
        return;
    const KdaIn k = kda_in(g_, W.kin + (size_t) W.kin_of[(size_t) il] * W.kin_layer, W.tmax, t);
    for (int i = 0; i < 3; ++i) proj[i] = k.proj[i];
    *fa = k.fa;
    *ga = k.ga;
    *beta = k.beta;
}

// The window as a pipeline: token t's pass is the token path's (fast_layers on each part, the residual over the
// previous part's pinned rows), enqueued for every part before token t + 1's, so each part's stream takes token t + 1
// right after token t - the parts work on consecutive tokens at once, as a prompt's chunks do.  The KDA layers leave
// each token's projections in the window's input rows (window_commit replays the kept tokens from them after
// restoring the states saved at the start).  The last part picks each token as it finishes (its head row, argmax or
// sample, an event), so a caller can stop at the first wrong draft: win_pipe_settle + window_commit then queue the
// restore behind the tokens still in flight, and the next window's first token follows them (win_pipe_begin with
// fresh = false: no wait for the devices) instead of waiting for them to drain.  The rows a token crosses with (the
// hop slots and events, the embedding and head rows) alternate between two banks of kPipeMax a window, so the next
// window's tokens never write a slot a late token of this one still reads.
// Calls: win_pipe_begin(first token), win_pipe_push(each further token), win_pipe_pick(t) as each is needed, then
// win_pipe_settle (before window_commit) or win_pipe_end (every pick, the devices drained).
// window_pipe's crew: a thread a part, each running its part's share of the window's tokens in order (win_pipe_part)
// - the next part's thread waits until this one has RECORDED token t's hop event (`done`), then has its stream wait
// on it (each token its own event: the threads run out of turn)
struct Glm5Model::PipeCrew {
    struct Worker {
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        std::deque<int> jobs;
        std::atomic<int> done{0};   // this window's tokens this part has enqueued
        bool quit = false;
        std::chrono::steady_clock::time_point t_begun;   // its begin done (win_pipe_begin_part), the first token next
        std::atomic<unsigned> commits{0};                // window_commit jobs this part has enqueued (win_commit_part)
    };
    std::vector<std::unique_ptr<Worker>> w;
    std::atomic<bool> failed{false};
    std::atomic<bool> fresh{false};   // this window's begin waits for the devices (win_pipe_begin_part)
    std::vector<unsigned> commit_want;   // a commit not waited for (window_commit's wait = false): each part's count
    std::mutex emu;
    std::string err;
};

void Glm5Model::win_crew_stop() {
    if (crew_ == nullptr) return;
    for (auto& wk : crew_->w) {
        {
            std::lock_guard<std::mutex> lk(wk->mu);
            wk->quit = true;
        }
        wk->cv.notify_one();
    }
    for (auto& wk : crew_->w)
        if (wk->th.joinable()) wk->th.join();
    delete crew_;
    crew_ = nullptr;
}

// part k's share of window token t: its input (the embedding on the first part; else the previous part's rows, once
// its stream has them), its layers, its rows out (the hop slot and event, or on the last part the head row, the pick
// and its event)
bool Glm5Model::win_pipe_part(int k, int t, std::string& err) {
    Glm5Model* m = this;
    Glm5Model* prev = nullptr;
    for (int i = 0; i < k; ++i) {
        prev = m;
        m = m->split_next_.get();
    }
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    const size_t hcE = (size_t) g.hc * E;
    FastState* F = m->fast_;
    auto& W = F->win;
    const int64_t p = fast_->win.p0 + t;
    const int sl = W.bank * kPipeMax + t;   // this window's slot of token t
    cudaSetDevice(m->dev_);
    if (prev == nullptr) {
        cudaMemcpyAsync(W.emb + (size_t) sl * E, W.emb_h + (size_t) sl * E, (size_t) E * sizeof(float),
                        cudaMemcpyHostToDevice, F->cs);
        gf::embed_streams(W.emb + (size_t) sl * E, m->state_, E, F->cs);
    } else {
        cudaStreamWaitEvent(F->cs, prev->fast_->win.ev_tok[sl], 0);
        cudaMemcpyAsync(m->state_, prev->fast_->win.hop_h + (size_t) sl * hcE, hcE * sizeof(float),
                        cudaMemcpyHostToDevice, F->cs);
    }
    W.pipe_t = t;
    F->bnd_tokens.fetch_add(1, std::memory_order_relaxed);
    gf::set_skip(W.from != nullptr ? W.skip_d : nullptr);
    const bool ok = m->fast_layers(p, prev != nullptr, err);
    W.pipe_t = -1;
    if (!ok) {
        gf::set_skip(nullptr);
        return false;
    }
    if (m->split_next_) {
        gf::set_skip(nullptr);
        cudaMemcpyAsync(W.hop_h + (size_t) sl * hcE, m->state_, hcE * sizeof(float), cudaMemcpyDeviceToHost, F->cs);
        cudaEventRecord(W.ev_tok[sl], F->cs);
        return true;
    }
    // the last part: this token's pick now - its head row, the output GEMV, the argmax (or row t's draw at the
    // counter the token loop would use), copied out with an event the caller waits for
    float* hx = W.head_x + (size_t) sl * E;
    void* hq = (uint8_t*) W.head_xq + (size_t) sl * W.xq_s;
    float* lg = W.logits + (size_t) sl * g.n_vocab;
    gf::head_prep(m->state_, m->w_.at("output_norm.weight"), g.norm_eps, E, hx, hq, F->cs);
    const WSlot& ow = m->ws_map_.at("output.weight");
    if (m->head_split_ready()) {
        if (!m->win_head_split(t, sl, hx, lg, err)) return false;
    } else {
        gf::MvJob o = {ow.q, hq, hx, lg, nullptr, 1.0f, ow.type, E, g.n_vocab};
        if (ow.type == 0) o.w = ow.f32;
        if (!gf::mv(&o, 1, F->cs)) {
            err = "glm window: output head (type " + std::to_string(ow.type) + ")";
            return false;
        }
        if (W.pick_sampled) {
            strata::kernels::SamplerParams sp = W.pick_sp;
            sp.counter += (uint64_t) t;
            strata::kernels::sample_tokens(lg, 1, (int) g.n_vocab, nullptr, 0, sp, W.tok + sl, F->cs);
        } else {
            gf::argmax(lg, g.n_vocab, W.tok + sl, F->cs);
        }
    }
    gf::set_skip(nullptr);
    cudaMemcpyAsync(W.tok_h + sl, W.tok + sl, sizeof(int), cudaMemcpyDeviceToHost, F->cs);
    cudaEventRecord(W.ev_pick[sl], F->cs);
    return true;
}

// a pipelined window's pick on the last part with the head split (head_split_ready): the head input out to the
// helper, its columns [c0, V) there and [0, c0) here at once, then the two argmax pairs joined - or, sampling, the
// helper's logits copied in beside this half's - into W.tok + sl: the whole row's pick (see head_split_link)
bool Glm5Model::win_head_split(int t, int sl, const float* hx, float* lg, std::string& err) {
    FastState* F = fast_;
    auto& W = F->win;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd, V = (int) g.n_vocab, c0 = head_c0_, nc = V - c0;
    Glm5Model* h = head_helper_;
    cudaMemcpyAsync(ht_x_h_ + (size_t) sl * E, hx, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost, F->cs);
    cudaEventRecord(ht_ev_[sl], F->cs);
    // the helper's half on its own stream, skipped with the token past a wrong draft by its own flag (the skip word is
    // mapped for every device; this device's flag is not the helper's)
    gf::set_skip(nullptr);
    cudaSetDevice(h->dev_);
    cudaStreamWaitEvent(h->hh_s_, ht_ev_[sl], 0);
    if (W.from != nullptr && h->hh_skip_d_ != nullptr) {
        gf::skip_gate(W.from, t, h->hh_skip_d_, h->hh_s_);
        gf::set_skip(h->hh_skip_d_);
    }
    float* hxd = h->hh_x_ + (size_t) sl * E;
    float* hlg = h->hh_lg_ + (size_t) sl * nc;
    int* hpr = h->hh_pair_ + (size_t) sl * 2;
    cudaMemcpyAsync(hxd, ht_x_h_ + (size_t) sl * E, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, h->hh_s_);
    const WSlot& hw = h->ws_map_.at("output.weight.hi");
    const gf::MvJob oh = {hw.q, nullptr, hxd, hlg, nullptr, 1.0f, hw.type, E, nc};
    const bool ok_h = gf::mv(&oh, 1, h->hh_s_);
    if (W.pick_sampled) {
        cudaMemcpyAsync(ht_lg_h_ + (size_t) sl * nc, hlg, (size_t) nc * sizeof(float), cudaMemcpyDeviceToHost, h->hh_s_);
    } else {
        gf::argmax_pair(hlg, nc, hpr, h->hh_s_);
        cudaMemcpyAsync(ht_pair_h_ + (size_t) sl * 2, hpr, 2 * sizeof(int), cudaMemcpyDeviceToHost, h->hh_s_);
    }
    cudaEventRecord(h->hh_ev_[sl], h->hh_s_);
    gf::set_skip(nullptr);
    cudaSetDevice(dev_);
    gf::set_skip(W.from != nullptr ? W.skip_d : nullptr);
    if (!ok_h) {
        err = "glm window: the head's half on CUDA" + std::to_string(h->dev_);
        return false;
    }
    // this part's half, then the join
    const WSlot& lw = ws_map_.at("output.weight.lo");
    const gf::MvJob o = {lw.q, W.head_xq ? (const void*) ((uint8_t*) W.head_xq + (size_t) sl * W.xq_s) : nullptr, hx, lg,
                         nullptr, 1.0f, lw.type, E, c0};
    if (!gf::mv(&o, 1, F->cs)) {
        err = "glm window: the head's half";
        return false;
    }
    if (W.pick_sampled) {
        cudaStreamWaitEvent(F->cs, h->hh_ev_[sl], 0);
        cudaMemcpyAsync(lg + c0, ht_lg_h_ + (size_t) sl * nc, (size_t) nc * sizeof(float), cudaMemcpyHostToDevice, F->cs);
        strata::kernels::SamplerParams sp = W.pick_sp;
        sp.counter += (uint64_t) t;
        strata::kernels::sample_tokens(lg, 1, V, nullptr, 0, sp, W.tok + sl, F->cs);
    } else {
        int* own = ht_pair_d_ + (size_t) sl * 4;
        gf::argmax_pair(lg, c0, own, F->cs);
        cudaStreamWaitEvent(F->cs, h->hh_ev_[sl], 0);
        cudaMemcpyAsync(own + 2, ht_pair_h_ + (size_t) sl * 2, 2 * sizeof(int), cudaMemcpyHostToDevice, F->cs);
        gf::argmax_join(own, own + 2, c0, W.tok + sl, F->cs);
        // STRATA_GLM_HEAD_SPLIT_CHECK=1 (debug, slower): the whole row's GEMV and argmax too, the picks compared on the
        // host (the count printed at exit)
        static const bool check = getenv("STRATA_GLM_HEAD_SPLIT_CHECK") != nullptr;
        if (check) {
            static float* clg = nullptr;
            static int* ctok = nullptr;
            static int* pair_h = nullptr;
            if (clg == nullptr) {
                cudaMalloc((void**) &clg, (size_t) V * sizeof(float));
                cudaMalloc((void**) &ctok, sizeof(int));
                cudaHostAlloc((void**) &pair_h, 3 * sizeof(int), cudaHostAllocDefault);
            }
            const WSlot& ow = ws_map_.at("output.weight");
            const gf::MvJob of = {ow.q, nullptr, hx, clg, nullptr, 1.0f, ow.type, E, V};
            gf::set_skip(nullptr);
            gf::mv(&of, 1, F->cs);
            gf::argmax(clg, V, ctok, F->cs);
            gf::set_skip(W.from != nullptr ? W.skip_d : nullptr);
            cudaMemcpyAsync(pair_h, W.tok + sl, sizeof(int), cudaMemcpyDeviceToHost, F->cs);
            cudaMemcpyAsync(pair_h + 1, ctok, sizeof(int), cudaMemcpyDeviceToHost, F->cs);
            // (a token past a wrong draft skipped this half's GEMV: its pick is never read - not compared)
            if (W.from != nullptr && W.skip_d != nullptr)
                cudaMemcpyAsync(pair_h + 2, W.skip_d, sizeof(int), cudaMemcpyDeviceToHost, F->cs);
            else
                cudaMemsetAsync(ctok, 0, 0, F->cs), pair_h[2] = 0;
            static std::atomic<long> n_cmp{0}, n_diff{0};
            static struct Report {
                ~Report() { std::fprintf(stderr, "glm head split check: %ld picks, %ld differ\n", n_cmp.load(), n_diff.load()); }
            } report;
            (void) report;
            cudaLaunchHostFunc(F->cs, [](void* p) {
                const int* v = (const int*) p;
                if (v[2] != 0) return;
                n_cmp.fetch_add(1);
                if (v[0] != v[1]) n_diff.fetch_add(1);
            }, pair_h);
        }
    }
    return true;
}

// STRATA_GLM_WIN_THREADS=0: the window's tokens enqueued by the calling thread alone (A/B)
static bool win_threads_on() {
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_WIN_THREADS");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

// a part's own share of a pipelined window's begin: a fresh window waits for its device; the tiers' boundary (their
// moves go live; it waits for the part's routes so far) when the device is idle - after the first parts the late
// tokens skip their work, so most are; the later ones catch up on a fresh window - and the recurrent states' backup
bool Glm5Model::win_pipe_begin_part(bool fresh, std::string& err) {
    cudaSetDevice(dev_);
    if (fresh) cudaStreamSynchronize(fast_->cs);
    bool idle = fresh;
    if (!idle) {
        idle = cudaStreamQuery(fast_->cs) == cudaSuccess;
        if (!idle) cudaGetLastError();
    }
    if (idle && !fast_boundary(err)) return false;
    // (a fresh window - the first of a decode, every 8th - backs up whatever the last commit left)
    const bool cur = kda_bak_cur_ && !fresh;
    kda_bak_cur_ = false;
    if (!cur && !spec_kda_copy(false)) {
        err = "glm window: the recurrent-state backup did not allocate";
        return false;
    }
    return true;
}

bool Glm5Model::win_pipe_begin(int32_t first, std::string& err, const strata::kernels::SamplerParams* sp,
                               bool fresh) {
    if (fast_ == nullptr) {
        err = "glm window: needs the fast path";
        return false;
    }
    if (pos_ + 1 > max_ctx_) {
        err = "glm window: past the context";
        return false;
    }
    const int64_t p0 = pos_;
    int n_parts = 0;
    // a fresh window waits for the devices; one that follows a window's late tokens only queues behind them.  Each
    // part's begin (win_pipe_begin_part) runs on its crew thread ahead of its share of the first token, the parts at
    // once: here, one after the other, the idle parts' boundaries held the first part's anchor ~1.7 ms (and the
    // drafter's start behind it; EXL3-4.05bpw over 9 GPUs).  STRATA_GLM_WIN_THREADS=0: here, as before
    const bool crew_on = split_next_ != nullptr && win_threads_on();
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        ++n_parts;
        cudaSetDevice(m->dev_);
        if (!m->win_alloc(kWinMax, err)) return false;
        if (!crew_on && !m->win_pipe_begin_part(fresh, err)) return false;
        auto& W = m->fast_->win;
        W.p0 = p0;
        W.T = 0;
        W.per_token = false;
        W.piped = true;
        W.bank ^= 1;
        W.pick_sampled = sp != nullptr && !sp->greedy;
        if (W.pick_sampled) W.pick_sp = *sp;
    }
    // the tokens past a wrong draft skip their heavy kernels (STRATA_GLM_WIN_SKIP=0: they run anyway)
    static const bool skip_on = [] {
        const char* v = getenv("STRATA_GLM_WIN_SKIP");
        return v == nullptr || std::atoi(v) != 0;
    }();
    {
        auto& WH = fast_->win;
        WH.from_h[WH.bank] = 1 << 30;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
            m->fast_->win.from = skip_on ? WH.from_h + WH.bank : nullptr;
    }
    cudaSetDevice(dev_);
    if (n_parts > 1 && win_threads_on()) {
        if (crew_ == nullptr) {
            crew_ = new PipeCrew();
            for (int k = 0; k < n_parts; ++k) {
                crew_->w.emplace_back(new PipeCrew::Worker());
                PipeCrew::Worker* wk = crew_->w.back().get();
                PipeCrew::Worker* before = k > 0 ? crew_->w[(size_t) k - 1].get() : nullptr;
                wk->th = std::thread([this, k, wk, before] {
                    for (;;) {
                        int t = -1;
                        {
                            std::unique_lock<std::mutex> lk(wk->mu);
                            wk->cv.wait(lk, [&] { return wk->quit || !wk->jobs.empty(); });
                            if (wk->jobs.empty()) return;
                            t = wk->jobs.front();
                            wk->jobs.pop_front();
                        }
                        std::string e;
                        // a commit (window_commit: -1 - its kept tokens): this part's restore and replay
                        if (t < 0) {
                            Glm5Model* m = this;
                            for (int i = 0; i < k; ++i) m = m->split_next_.get();
                            m->win_commit_part(-1 - t);
                            wk->commits.fetch_add(1, std::memory_order_release);
                            continue;
                        }
                        // a window's first token: this part's begin first, beside the other parts'
                        if (t == 0 && !crew_->failed.load(std::memory_order_relaxed)) {
                            Glm5Model* m = this;
                            for (int i = 0; i < k; ++i) m = m->split_next_.get();
                            if (!m->win_pipe_begin_part(crew_->fresh.load(std::memory_order_acquire), e)) {
                                std::lock_guard<std::mutex> lk(crew_->emu);
                                if (crew_->err.empty()) crew_->err = e;
                                crew_->failed.store(true);
                            }
                            wk->t_begun = std::chrono::steady_clock::now();
                        }
                        // the previous part's event t is recorded (by its thread) before this stream waits on it
                        while (before != nullptr && before->done.load(std::memory_order_acquire) <= t &&
                               !crew_->failed.load(std::memory_order_relaxed))
                            std::this_thread::yield();
                        if (!crew_->failed.load(std::memory_order_relaxed) && !win_pipe_part(k, t, e)) {
                            std::lock_guard<std::mutex> lk(crew_->emu);
                            if (crew_->err.empty()) crew_->err = e;
                            crew_->failed.store(true);
                        }
                        wk->done.store(t + 1, std::memory_order_release);
                    }
                });
            }
        }
        crew_->failed.store(false);
        crew_->err.clear();
        crew_->fresh.store(fresh, std::memory_order_release);
        for (auto& wk : crew_->w) wk->done.store(0);
    }
    return win_pipe_push(first, err);
}

bool Glm5Model::win_pipe_push(int32_t token, std::string& err) {
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    FastState* F0 = fast_;
    auto& W0 = F0->win;
    const int t = W0.T;
    const int64_t p = W0.p0 + t;
    if (t >= kPipeMax || p + 1 > max_ctx_) {
        err = "glm window: past " + std::to_string(kPipeMax) + " pipelined tokens or the context";
        return false;
    }
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm window: no embedding dequantizer";
        return false;
    }
    float* row = W0.emb_h + (size_t) (W0.bank * kPipeMax + t) * E;
    tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, E), row, E);
    if (const float* img = image_row(p)) std::memcpy(row, img, (size_t) E * sizeof(float));
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->fast_->win.T = t + 1;
    if (crew_ != nullptr && win_threads_on() && split_next_ != nullptr) {
        // each part's thread takes it in turn
        for (auto& wk : crew_->w) {
            {
                std::lock_guard<std::mutex> lk(wk->mu);
                wk->jobs.push_back(t);
            }
            wk->cv.notify_one();
        }
        return true;
    }
    int k = 0;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get(), ++k)
        if (!win_pipe_part(k, t, err)) return false;
    cudaSetDevice(dev_);
    return true;
}

// every part's share of this window's tokens enqueued (a crew's threads done with them); false if one failed
bool Glm5Model::win_pipe_settle(std::string& err) {
    if (crew_ == nullptr || !win_threads_on() || split_next_ == nullptr) return true;
    const int T = fast_->win.T;
    for (auto& wk : crew_->w)
        while (wk->done.load(std::memory_order_acquire) < T) std::this_thread::yield();
    if (crew_->failed.load()) {
        std::lock_guard<std::mutex> lk(crew_->emu);
        err = crew_->err.empty() ? "glm window: a part's enqueue failed" : crew_->err;
        return false;
    }
    return true;
}

// (STRATA_GLM_DFLASH_TL) when part k's crew thread finished its begin for this window's first token
std::chrono::steady_clock::time_point Glm5Model::win_pipe_begun_at(int k) const {
    if (crew_ == nullptr || k < 0 || k >= (int) crew_->w.size()) return {};
    return crew_->w[(size_t) k]->t_begun;
}

// window tokens t.. follow a wrong draft: their kernels still queued skip their work
void Glm5Model::win_pipe_cancel_from(int t) {
    auto& WH = fast_->win;
    if (WH.from_h == nullptr) return;
    WH.from_h[WH.bank] = t;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Glm5Model::win_pipe_drain() {
    win_commit_wait();
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (m->fast_ != nullptr) cudaStreamSynchronize(m->fast_->cs);
    }
    cudaSetDevice(dev_);
}

// the pick after window token t (its row through the last part), waiting for it
bool Glm5Model::win_pipe_pick(int t, int32_t& pick, std::string& err) {
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    auto& WT = tail->fast_->win;
    if (t < 0 || t >= WT.T) {
        err = "glm window: no token " + std::to_string(t) + " in the window";
        return false;
    }
    if (crew_ != nullptr && win_threads_on() && split_next_ != nullptr) {
        PipeCrew::Worker* last = crew_->w.back().get();
        while (last->done.load(std::memory_order_acquire) <= t && !crew_->failed.load()) std::this_thread::yield();
        if (crew_->failed.load()) return win_pipe_settle(err);
    }
    const int sl = WT.bank * kPipeMax + t;
    cudaSetDevice(tail->dev_);
    cudaError_t e;
    while ((e = cudaEventQuery(WT.ev_pick[sl])) == cudaErrorNotReady) std::this_thread::yield();
    cudaSetDevice(dev_);
    if (e != cudaSuccess || gf::launch_errors() > 0) {
        err = std::string("glm window: ") + (e != cudaSuccess ? cudaGetErrorString(e) : "a kernel launch failed");
        return false;
    }
    pick = WT.tok_h[sl];
    return true;
}

bool Glm5Model::win_pipe_end(int32_t* out, std::string& err) {
    const int T = fast_->win.T;
    const int64_t p0 = fast_->win.p0;
    for (int t = 0; t < T; ++t)
        if (!win_pipe_pick(t, out[t], err)) return false;
    if (!win_pipe_settle(err)) return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        cudaStreamSynchronize(m->fast_->cs);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm window: ") + cudaGetErrorString(e);
        return false;
    }
    if (gf::launch_errors() > 0) {
        err = "glm window: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        m->pos_ = p0 + T;
        // STRATA_GLM_PROF: the T tokens' phases on each part (its "start" holds the wait for the previous part)
        if (m->fast_->prof_on) {
            m->fast_->collect();
            if (m->fast_->ptokens > 0) m->fast_->ptokens += (uint64_t) T - 1;
        }
    }
    last_tok_ = out[T - 1];
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::window_pipe(const int32_t* tokens, int T, int32_t* out, std::string& err,
                            const strata::kernels::SamplerParams* sp) {
    if (fast_ == nullptr || T < 1 || T > kPipeMax) {
        err = "glm window: needs the fast path and 1.." + std::to_string(kPipeMax) + " tokens";
        return false;
    }
    if (pos_ + T > max_ctx_) {
        err = "glm window: past the context";
        return false;
    }
    static const bool hprof = getenv("STRATA_GLM_WIN_PROF") != nullptr;
    using hclk = std::chrono::steady_clock;
    auto h0 = hclk::now();
    const auto hlap = [&](double& acc) {
        if (!hprof) return;
        const auto t = hclk::now();
        acc += std::chrono::duration<double, std::milli>(t - h0).count();
        h0 = t;
    };
    auto& WH = fast_->win;
    if (!win_pipe_begin(tokens[0], err, sp)) return false;
    hlap(WH.h_setup);
    for (int t = 1; t < T; ++t)
        if (!win_pipe_push(tokens[t], err)) return false;
    hlap(WH.h_enqueue);
    if (!win_pipe_end(out, err)) return false;
    hlap(WH.h_wait);
    if (hprof) ++WH.h_n;
    return true;
}

void Glm5Model::win_prof_report() {
    if (fast_ != nullptr && fast_->win.h_n > 0) {
        auto& W = fast_->win;
        const double n = (double) W.h_n;
        std::fprintf(stderr, "glm window host (%lld windows): setup %.2f ms | enqueue %.2f ms | wait for the GPUs %.2f ms | "
                             "after %.2f ms\n", (long long) W.h_n, W.h_setup / n, W.h_enqueue / n, W.h_wait / n,
                     W.h_after / n);
        W.h_setup = W.h_enqueue = W.h_wait = W.h_after = 0;
        W.h_n = 0;
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr) continue;
        auto& W = m->fast_->win;
        if (W.n_prof > 0) {
            double tot = 0;
            std::string line;
            for (const auto& kv : W.ev_ms) tot += kv.second;
            for (const auto& kv : W.ev_ms) {
                char b[64];
                std::snprintf(b, sizeof b, " %s %.2f", kv.first.c_str(), kv.second / W.n_prof);
                line += b;
            }
            std::fprintf(stderr, "glm window prof CUDA%d (%lld windows, layers [%d, %d)): %.2f ms a window -%s\n", m->dev_,
                         (long long) W.n_prof, m->l0_, m->l1_, tot / W.n_prof, line.c_str());
        }
        W.ev_ms.clear();
        W.n_prof = 0;
        if (W.u_layers > 0)
            std::fprintf(stderr, "glm window union CUDA%d: %.2f distinct experts of %.2f routed a window layer (%.0f%%)\n",
                         m->dev_, (double) W.u_distinct / W.u_layers, (double) W.u_picks / W.u_layers,
                         100.0 * (double) W.u_distinct / (double) std::max<int64_t>(1, W.u_picks));
        W.u_layers = W.u_distinct = W.u_picks = 0;
    }
}

// one part's share of window_commit for a window of more tokens than kept: the recurrent states as they were before
// the window, then the kept tokens again (their inputs as the window computed them; the outputs go to scratch)
void Glm5Model::win_commit_part(int n_keep) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    auto& W = F->win;
    cudaSetDevice(dev_);
    // the kept tokens replayed onto the backup (the states before the window) through every recurrent layer at once,
    // then the backup copied to the live states: one copy (not a restore and the next window's backup) and two
    // launches a token (not two a token and layer).  STRATA_GLM_WIN_REPLAY=0: restored, then layer by layer (A/B)
    static const bool batched = [] {
        const char* v = getenv("STRATA_GLM_WIN_REPLAY");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (batched && F->rep != nullptr && kda_bak_ != nullptr && g.kda_head_dim == 128) {
        const size_t sd = (size_t) g.d_inner() * g.kda_head_dim;
        const size_t run = sd + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
        const size_t di = (size_t) g.d_inner();
        std::vector<int> layers;
        for (int il = l0_; il < l1_; ++il)
            if (g.is_recr(il)) layers.push_back(il);
        for (int t = 0; t < n_keep; ++t)
            for (size_t c0 = 0; c0 < layers.size(); c0 += gf::kKdaLayers) {
                gf::KdaPrepArgs pa[gf::kKdaLayers];
                gf::KdaRecArgs ra[gf::kKdaLayers];
                const int n = (int) std::min<size_t>(gf::kKdaLayers, layers.size() - c0);
                for (int i = 0; i < n; ++i) {
                    const int il = layers[c0 + (size_t) i];
                    const auto& Ly = F->L[(size_t) il];
                    float* bak = kda_bak_ + (c0 + (size_t) i) * run;
                    float* sc = F->rep + (size_t) i * 5 * di;
                    const KdaIn in = kda_in(g, W.kin + (size_t) W.kin_of[(size_t) il] * W.kin_layer, W.tmax, t);
                    gf::KdaPrepArgs& k = pa[i];
                    for (int r = 0; r < 3; ++r) {
                        k.proj[r] = in.proj[r];
                        k.conv_w[r] = Ly.conv[r];
                        k.out[r] = sc + (size_t) r * di;
                    }
                    k.conv_state = bak + sd;
                    k.fa = in.fa;
                    k.ga = in.ga;
                    k.f_b = Ly.f_b;
                    k.g_b = Ly.g_b;
                    k.dt_bias = Ly.dt_bias;
                    k.ssm_a = Ly.ssm_a;
                    k.lower_bound = g.kda_lb;
                    k.g1 = sc + 3 * di;
                    k.g2 = sc + 4 * di;
                    k.n_head = g.n_head;
                    k.head_dim = g.kda_head_dim;
                    k.d_conv = g.d_conv;
                    gf::KdaRecArgs& r = ra[i];
                    r.q = k.out[0];
                    r.k = k.out[1];
                    r.v = k.out[2];
                    r.g1 = k.g1;
                    r.beta_raw = in.beta;
                    r.state = bak;
                    r.g2 = k.g2;
                    r.norm_w = Ly.ssm_norm;
                }
                gf::kda_layers(pa, ra, n, g.norm_eps, F->cs);
            }
        for (size_t i = 0; i < layers.size(); ++i)
            cudaMemcpyAsync(state_ + kda_S_[(size_t) layers[i]], kda_bak_ + i * run, run * sizeof(float),
                            cudaMemcpyDeviceToDevice, F->cs);
        kda_bak_cur_ = true;
        return;
    }
    spec_kda_copy(true);
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.recr) continue;
        float* base = W.kin + (size_t) W.kin_of[(size_t) il] * W.kin_layer;
        for (int t = 0; t < n_keep; ++t)
            kda_token(F, Ly, g, kda_in(g, base, W.tmax, t), state_ + kda_conv_[(size_t) il], state_ + kda_S_[(size_t) il],
                      F->gated_q, F->cs);
    }
}

void Glm5Model::win_commit_wait() {
    if (crew_ == nullptr || crew_->commit_want.empty()) return;
    for (size_t i = 0; i < crew_->w.size() && i < crew_->commit_want.size(); ++i)
        while (crew_->w[i]->commits.load(std::memory_order_acquire) < crew_->commit_want[i]) std::this_thread::yield();
    crew_->commit_want.clear();
}

bool Glm5Model::window_commit(int n_keep, std::string& err, bool wait) {
    if (fast_ == nullptr || fast_->win.T < 1 || n_keep < 1 || n_keep > fast_->win.T) {
        err = "glm window: commit of " + std::to_string(n_keep) + " without a window of as many tokens";
        return false;
    }
    const int T = fast_->win.T;
    const int64_t p0 = fast_->win.p0;
    // a pipelined window's parts each enqueue their own restore and replay on their crew thread, at once (one after
    // the other here, ~0.5 ms of the round on 9 GPUs between the last pick and the next anchor); this waits for the
    // enqueues, so the state is committed in stream order when it returns, as below
    const bool crewed = n_keep < T && fast_->win.piped && crew_ != nullptr && win_threads_on() &&
                        split_next_ != nullptr && !crew_->failed.load();
    if (crewed) {
        win_commit_wait();
        std::vector<unsigned> want;
        for (auto& wk : crew_->w) {
            want.push_back(wk->commits.load(std::memory_order_acquire) + 1);
            {
                std::lock_guard<std::mutex> lk(wk->mu);
                wk->jobs.push_back(-1 - n_keep);
            }
            wk->cv.notify_one();
        }
        crew_->commit_want = std::move(want);
        if (wait) win_commit_wait();
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (n_keep < T && !crewed) m->win_commit_part(n_keep);
        m->pos_ = p0 + n_keep;
        m->fast_->win.T = 0;
    }
    cudaSetDevice(dev_);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm window commit: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace strata::core
