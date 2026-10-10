// src/core/glm_mtp.cu - the MTP speculative decode: llama.cpp's draft-mtp (common/speculative.cpp, PR #29928's GLM5-Next
// NextN graph) on the fast path, on any number of devices.
//
// A ROUND, from the last emitted token y at position p0 (the trunk's hidden state of p0 - 1 in the tail's head_x):
//
//   1. draft   - the NextN block at p0 - 1 reads (h_{p0-1}, y) and proposes d1; its own output hidden state
//                (shared_head_norm's, which also feeds its head) and d1 make the next step at p0, ... up to n drafts
//                (STRATA_GLM_MTP_DRAFT, 3 like llama.cpp's --spec-draft-n-max; greedy: each draft is the head's argmax);
//   2. verify  - the window [y, d1 .. dn] at positions p0 .. p0 + n through every layer of every part as ONE batched
//                pass: the projections, the shared experts and the output head read their weights once for all rows
//                (mv_rows), the per-token steps (mHC, the convs, the recurrence, the DSA attention, the routed experts
//                through the tiers) row after row with the one-token kernels - each row's arithmetic is the one-token
//                path's, so a row's logits are the ones a token-at-a-time decode computes;
//   3. accept  - the target's sample at row t (the request's sampler, draw index counter + t) is the truth for position
//                p0 + t + 1: d_{t+1} stands while they match (llama.cpp's sample-and-match), the first mismatch is the
//                new token, a full match leaves the last row's sample as a bonus token;
//   4. commit  - the recurrent layers advance by the kept rows only (the verify read their states and wrote none: the
//                commit replays the kept rows' recurrence from the saved inputs, the conv histories come from per-row
//                copies); the attention caches are by position and need nothing;
//   5. cache   - the NextN block's cache takes the kept rows with the trunk's hidden states (its cache-writing half: the
//                entries the drafts wrote from their own hidden states are replaced), as llama.cpp's process() does.
//
// The tokens are exactly what the token-at-a-time loop produces for the same samples.  A length 0 round is a plain token
// (the token path, which keeps the NextN cache).  The length adapts per machine (STRATA_GLM_MTP_ADAPT, on): every
// round's wall time is measured per length and the draft positions' acceptance counted, and the length with the most
// tokens per second wins - where the routed experts stream from RAM a batched verify reads every row's experts and
// saves only the dense weights, so a short draft or none can be the faster one.
//
// The NextN cache's entry p pairs h_p with the token at p + 1 (the prompt path writes them so); llama.cpp keeps that
// pair at p + 1.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/sampler.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core {

namespace {
constexpr size_t kQ8 = 36;   // bytes per q8_1 block (32 values)

size_t q8_bytes(int64_t n) { return (size_t) (n / 32) * kQ8; }

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

// STRATA_GLM_MTP_PROMOTE_MIN is fast_moe's STRATA_GLM_PROMOTE_MIN (the verify's routes keep the same rule)
int promote_min_env() {
    static const int v = [] {
        const char* e = getenv("STRATA_GLM_PROMOTE_MIN");
        return e ? std::max(0, std::atoi(e)) : 0;
    }();
    return v;
}
}  // namespace

// the drafts per round at most: STRATA_GLM_MTP_DRAFT (llama.cpp's default 3; 0 turns the MTP decode off)
int glm_mtp_draft_cap() {
    static const int v = [] {
        const char* e = getenv("STRATA_GLM_MTP_DRAFT");
        return e ? std::max(0, std::min(gf::kMaxRows - 1, std::atoi(e))) : 3;
    }();
    return v;
}

// whether the MTP decode is wanted (the model must still carry the NextN block): not turned off, and not the earlier
// two-part pipelined decode (STRATA_GLM_MTP_PIPELINE=1)
bool glm_mtp_decode_wanted() {
    return getenv("STRATA_GLM_NO_SPEC") == nullptr && getenv("STRATA_GLM_NO_MTP") == nullptr &&
           getenv("STRATA_GLM_MTP_PIPELINE") == nullptr && glm_mtp_draft_cap() > 0;
}

// ---------------------------------------------------------------- the window's buffers (fast_setup, before the pool)
bool Glm5Model::mtp_rows_setup(int rows, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    auto& B = F->rows;
    const int64_t E = g.n_embd, DI = g.d_inner(), FF = (int64_t) g.n_ff_exp * g.n_shared, hcE = (int64_t) g.hc * E;
    const int64_t hist = g.d_conv - 1;
    const int R = rows;
    B.rec.assign((size_t) g.n_layers + 1, -1);
    int n_rec = 0;
    for (int il = l0_; il < l1_; ++il)
        if (g.is_recr(il)) B.rec[(size_t) il] = n_rec++;
    const bool head = split_next_ == nullptr;   // this part computes the head (and carries the NextN block, if any)
    const auto sizes = [&](uint8_t* at0, bool carve) -> size_t {
        uint8_t* at = at0;
        const auto f = [&](float*& p, size_t n) {
            float* q = take_f(at, n);
            if (carve) p = q;
        };
        const auto b = [&](void*& p, size_t n) {
            void* q = take_b(at, n);
            if (carve) p = q;
        };
        f(B.R, (size_t) R * 2 * hcE);
        f(B.x, (size_t) R * E);
        f(B.mixer, (size_t) R * E);
        f(B.ffn, (size_t) R * E);
        f(B.emb, (size_t) R * E);
        f(B.pre, (size_t) R * 8);
        f(B.post, (size_t) R * 8);
        f(B.comb, (size_t) R * 16);
        f(B.part, (size_t) R * FastState::Rows::kHcPart);
        void* ctr = nullptr;
        b(ctr, (size_t) R * 16 * sizeof(unsigned int));
        if (carve) B.counter = (unsigned int*) ctr;
        b(B.xq, (size_t) R * q8_bytes(E));
        for (int i = 0; i < 3; ++i) f(B.proj[i], (size_t) R * DI);
        f(B.fa, (size_t) R * g.kda_head_dim);
        f(B.ga, (size_t) R * g.kda_head_dim);
        f(B.kq, (size_t) R * DI);
        f(B.g2, (size_t) R * DI);
        b(B.gated_q, (size_t) R * q8_bytes(DI));
        f(B.qr_raw, (size_t) R * g.q_lora);
        f(B.qr, (size_t) R * g.q_lora);
        b(B.qr_q, (size_t) R * q8_bytes(g.q_lora));
        f(B.kv_raw, (size_t) R * g.kv_lora);
        f(B.ik_raw, (size_t) R * g.idx_key);
        f(B.ig_raw, (size_t) R * g.idx_key);
        f(B.iw, (size_t) R * g.idx_heads);
        f(B.q, (size_t) R * g.n_head * g.qk_nope);
        f(B.iq, (size_t) R * g.idx_heads * g.idx_key);
        b(B.attn_q, (size_t) R * q8_bytes((int64_t) g.n_head * g.v_head));
        f(B.rlog, (size_t) R * g.n_expert);
        f(B.sh_g, (size_t) R * FF);
        f(B.sh_u, (size_t) R * FF);
        f(B.dg, (size_t) R * g.n_ff_dense);
        f(B.du, (size_t) R * g.n_ff_dense);
        b(B.dhq, (size_t) R * q8_bytes(g.n_ff_dense));
        f(B.head_x, (size_t) R * E);
        b(B.head_xq, (size_t) R * q8_bytes(E));
        if (head) f(B.logits, (size_t) R * g.n_vocab);
        void* tk = nullptr;
        b(tk, (size_t) R * sizeof(int) + 64);
        if (carve) B.tok = (int*) tk;
        b(B.catq, (size_t) R * q8_bytes(2 * E));
        f(B.hid, (size_t) R * E);
        if (n_rec > 0) {
            f(B.keep_k, (size_t) n_rec * R * DI);
            f(B.keep_v, (size_t) n_rec * R * DI);
            f(B.keep_g1, (size_t) n_rec * R * DI);
            f(B.keep_beta, (size_t) n_rec * R * g.n_head);
            f(B.conv_after, (size_t) n_rec * R * 3 * DI * hist);
        }
        return (size_t) (at - at0);
    };
    const size_t need = sizes(nullptr, false) + 256;
    if (cudaMalloc(&B.arena, need) != cudaSuccess) {
        cudaGetLastError();
        B.arena = nullptr;
        err = "glm mtp: the verify window's buffers (" + std::to_string(need >> 20) + " MB) did not allocate";
        return false;
    }
    cudaMemset(B.arena, 0, need);
    B.bytes = need;
    sizes((uint8_t*) B.arena, true);
    if (cudaHostAlloc((void**) &B.emb_h, (size_t) R * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &B.hop_h, (size_t) R * hcE * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &B.tok_h, (size_t) R * sizeof(int) + 64, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        mtp_rows_free();
        err = "glm mtp: the verify window's pinned staging did not allocate";
        return false;
    }
    B.cap = R;
    mtp_rows_ = R;
    return true;
}

void Glm5Model::mtp_rows_free() {
    if (fast_ == nullptr) return;
    auto& B = fast_->rows;
    if (B.arena) cudaFree(B.arena);
    if (B.emb_h) cudaFreeHost(B.emb_h);
    if (B.hop_h) cudaFreeHost(B.hop_h);
    if (B.tok_h) cudaFreeHost(B.tok_h);
    B = FastState::Rows{};
    mtp_rows_ = 0;
}

bool Glm5Model::mtp_ready() const {
    if (fast_ == nullptr || !glm_mtp_decode_wanted()) return false;
    const Glm5Model* last = this;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr || m->fast_->rows.cap < 2) return false;
        last = m;
    }
    return last->mtp_il_ >= 0;
}

// ---------------------------------------------------------------- the window through the layers
// one DSA mixer over the rows (x / xq rows -> mixer rows) at positions p0 .. p0 + T - 1: fast_dsa's steps, the
// projections batched, the cache writes and the attention row after row (row t sees the positions up to its own)
bool Glm5Model::fast_dsa_rows(int il, int64_t p0, int T, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    auto& B = F->rows;
    const auto& Ly = F->L[(size_t) il];
    const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
    gf::MvJob j[5];
    j[0] = {Ly.q_a.q, B.xq, B.x, B.qr_raw, nullptr, 1.0f, Ly.q_a.type, g.n_embd, g.q_lora};
    j[1] = {Ly.kv_a.q, B.xq, B.x, B.kv_raw, nullptr, 1.0f, Ly.kv_a.type, g.n_embd, g.kv_lora};
    j[2] = {Ly.idx_k, nullptr, B.x, B.ik_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[3] = {Ly.idx_gate, nullptr, B.x, B.ig_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[4] = {Ly.idx_proj, nullptr, B.x, B.iw, nullptr, prescale, gf::kTypeBF16, g.n_embd, g.idx_heads};
    if (!gf::mv_rows(j, 5, T, s)) { err = "glm mtp: dsa projections"; return false; }
    if (F->prof_on) F->mark("v_dsa_proj_mv");
    for (int t = 0; t < T; ++t) {
        gf::DsaPrepArgs d;
        d.qr_raw = B.qr_raw + (size_t) t * g.q_lora;
        d.q_a_norm = Ly.q_a_norm;
        d.qr = B.qr + (size_t) t * g.q_lora;
        d.qr_q = (uint8_t*) B.qr_q + (size_t) t * q8_bytes(g.q_lora);
        d.q_lora = g.q_lora;
        d.kv_raw = B.kv_raw + (size_t) t * g.kv_lora;
        d.kv_norm = Ly.kv_a_norm;
        d.lat = (uint16_t*) (state_ + dsa_lat_[(size_t) il]);
        d.kv_lora = g.kv_lora;
        d.lat_q8 = lat_q8_;
        d.ik_raw = B.ik_raw + (size_t) t * g.idx_key;
        d.k_norm_w = Ly.k_norm_w;
        d.k_norm_b = Ly.k_norm_b;
        d.ik_cache = state_ + dsa_ik_[(size_t) il];
        d.ig_raw = B.ig_raw + (size_t) t * g.idx_key;
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
    if (F->prof_on) F->mark("v_dsa_prep");
    gf::MvJob j2[2];
    j2[0] = {Ly.q_b.q, B.qr_q, nullptr, B.q, nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * g.qk_nope};
    j2[1] = {Ly.idx_q_b, nullptr, B.qr, B.iq, nullptr, 1.0f, gf::kTypeBF16, g.q_lora, g.idx_heads * g.idx_key};
    if (!gf::mv_rows(j2, 2, T, s)) { err = "glm mtp: dsa q"; return false; }
    if (F->prof_on) F->mark("v_dsa_q_mv");
    for (int t = 0; t < T; ++t) {
        const int64_t p = p0 + t;
        const int pool_done = (int) ((p + 1) / g.idx_kpool);
        if (pool_done > 0)
            gf::dsa_score(B.iq + (size_t) t * g.idx_heads * g.idx_key, state_ + dsa_pool_[(size_t) il],
                          B.iw + (size_t) t * g.idx_heads, g.idx_key, g.idx_heads, pool_done, F->score, s);
        const int top_pools = std::min(g.top_pools_max(), pool_done);
        const int n_sel = g.idx_kpool * top_pools + (g.idx_select_tail ? g.idx_kpool - 1 : 0);
        gf::dsa_select(F->score, pool_done, g.idx_kpool, top_pools, n_sel, (int) p, F->cells, s);
        gf::mla(B.q + (size_t) t * g.n_head * g.qk_nope, Ly.k_b, Ly.v_b, (const uint16_t*) (state_ + dsa_lat_[(size_t) il]),
                F->cells, n_sel, g.n_head, g.qk_nope, g.kv_lora, g.v_head,
                (uint8_t*) B.attn_q + (size_t) t * q8_bytes((int64_t) g.n_head * g.v_head), s, lat_q8_);
    }
    if (F->prof_on) F->mark("v_mla");
    gf::MvJob o = {Ly.out.q, B.attn_q, nullptr, B.mixer, nullptr, 1.0f, Ly.out.type, g.n_head * g.v_head, g.n_embd};
    if (!gf::mv_rows(&o, 1, T, s)) { err = "glm mtp: dsa output"; return false; }
    if (F->prof_on) F->mark("v_dsa_out_mv");
    return true;
}

// one MoE FFN over the rows (x / xq rows -> ffn rows): the router and the shared expert's gate/up batched, then each
// row's route through the tiers as fast_moe takes it (no next-layer prediction or lookahead for the window's rows)
bool Glm5Model::fast_moe_rows(int il, int T, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    auto& B = F->rows;
    const auto& Ly = F->L[(size_t) il];
    const int FFs = g.n_ff_exp * g.n_shared;
    const int64_t E = g.n_embd;
    gf::MvJob j[3];
    j[0] = {Ly.router, nullptr, B.x, B.rlog, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.n_expert};
    j[1] = {Ly.sh_gate.q, B.xq, nullptr, B.sh_g, nullptr, 1.0f, Ly.sh_gate.type, g.n_embd, FFs};
    j[2] = {Ly.sh_up.q, B.xq, nullptr, B.sh_u, nullptr, 1.0f, Ly.sh_up.type, g.n_embd, FFs};
    if (!gf::mv_rows(j, 3, T, s)) { err = "glm mtp: router / shared expert"; return false; }
    if (F->prof_on) F->mark("v_router_shexp_mv");
    gf::MoeDev md = F->md;
    md.pf_src = F->pf_buf[il & 1];
    md.pf_dst = F->pf_buf[il & 1] + 8;
    md.pf_n = F->pf_n_buf[il & 1];
    const bool lane = F->cpu_plan != 0ull && F->cpu_fmt[(size_t) il].n_ff > 0;
    for (int t = 0; t < T; ++t) {
        float* xt = B.x + (size_t) t * E;
        const void* xqt = (const uint8_t*) B.xq + (size_t) t * q8_bytes(E);
        float* ft = B.ffn + (size_t) t * E;
        gf::moe_route(B.rlog + (size_t) t * g.n_expert, Ly.router_bias, g.n_expert, g.n_exp_used, g.w_scale,
                      g.norm_w != 0, il, xt, g.n_embd, md, B.sh_g + (size_t) t * FFs, B.sh_u + (size_t) t * FFs,
                      g.swiglu_shexp, FFs, F->sh_hq, s, nullptr, nullptr, 0, nullptr, nullptr, 0, 8,
                      lane ? F->cpu_plan : 0ull, promote_min_env());
        ++F->expected;
        // (every expert in VRAM: no route waits or fetches - fast_moe's all_resident; 2 launches a row and layer)
        if (!F->all_resident) {
            gf::moe_wait(md, g.n_embd, s);
            gf::moe_fetch(md, g.n_exp_used, Ly.blob, s);
        }
        gf::moe_gate_up(Ly.gu_type, md, g.n_exp_used, g.n_embd, g.n_ff_exp, g.swiglu_exp, xqt, F->hq, Ly.sh_down.q,
                        Ly.sh_down.type, F->sh_hq, FFs, F->sh_out, s);
        gf::moe_down(Ly.d_type, md, g.n_exp_used, g.n_embd, g.n_ff_exp, Ly.down_off, F->hq,
                     Ly.sh_down.q != nullptr ? F->sh_out : nullptr, ft, s);
        if (lane && !F->all_resident) gf::moe_cpu_wait(md, g.n_embd, ft, s);
    }
    if (F->prof_on) F->mark("v_moe_rows");
    return true;
}

// this part's layers over the window's rows: the residual of row t starts and ends in its first R buffer
bool Glm5Model::fast_layers_rows(int64_t p0, int T, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    auto& B = F->rows;
    const int64_t E = g.n_embd, DI = g.d_inner(), hcE = (int64_t) g.hc * E;
    const int64_t hist = g.d_conv - 1;
    const auto Rrow = [&](int t, int which) { return B.R + ((size_t) t * 2 + (size_t) which) * hcE; };
    const auto xq_t = [&](int t) { return (void*) ((uint8_t*) B.xq + (size_t) t * q8_bytes(E)); };
    int cur = 0;   // which of each row's two buffers holds the residual
    // an mHC read over the rows (fused with the write of block_out, row 0's; nullptr: none): one launch, else row by row
    const auto hc_all = [&](const float* block_out, const uint16_t* w_fn, const float* w_scale, const float* w_base,
                            const float* norm_w) {
        gf::HcArgs h;
        h.block_out = block_out;
        h.R_old = Rrow(0, cur);
        h.R_new = Rrow(0, 1 - cur);
        h.post_in = B.post;
        h.comb_in = B.comb;
        h.pre = B.pre;
        h.post = B.post;
        h.comb = B.comb;
        h.w_fn = w_fn;
        h.w_scale = w_scale;
        h.w_base = w_base;
        h.norm_w = norm_w;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = g.n_embd;
        h.x = B.x;
        h.xq = B.xq;
        h.part = B.part;
        h.counter = B.counter;
        if (gf::hc_rows(h, T, (int) (2 * hcE), FastState::Rows::kHcPart, s)) return;
        for (int t = 0; t < T; ++t) {
            gf::HcArgs r = h;
            if (block_out != nullptr) r.block_out = block_out + (size_t) t * E;
            r.R_old = Rrow(t, cur);
            r.R_new = Rrow(t, 1 - cur);
            r.post_in = r.post = B.post + (size_t) t * 8;
            r.comb_in = r.comb = B.comb + (size_t) t * 16;
            r.pre = B.pre + (size_t) t * 8;
            r.x = B.x + (size_t) t * E;
            r.xq = xq_t(t);
            r.part = F->part;
            r.counter = F->counter;
            gf::hc(r, s);
        }
    };
    if (F->prof_on) F->mark("v_start");
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        // ---- the attention-side read (fused with the previous FFN's write)
        hc_all(il == l0_ ? nullptr : B.ffn, Ly.hc_attn_fn, Ly.hc_attn_scale, Ly.hc_attn_base, Ly.attn_norm);
        if (il != l0_) cur = 1 - cur;
        if (F->prof_on) F->mark("v_hc_attn");
        if (Ly.recr) {
            const int li = B.rec[(size_t) il];
            const size_t kr = (size_t) li * B.cap;   // this layer's first row in the keep buffers
            gf::MvJob j[6];
            j[0] = {Ly.q.q, B.xq, nullptr, B.proj[0], nullptr, 1.0f, Ly.q.type, g.n_embd, (int) DI};
            j[1] = {Ly.k.q, B.xq, nullptr, B.proj[1], nullptr, 1.0f, Ly.k.type, g.n_embd, (int) DI};
            j[2] = {Ly.v.q, B.xq, nullptr, B.proj[2], nullptr, 1.0f, Ly.v.type, g.n_embd, (int) DI};
            j[3] = {Ly.f_a, nullptr, B.x, B.fa, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.kda_head_dim};
            j[4] = {Ly.g_a, nullptr, B.x, B.ga, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.kda_head_dim};
            j[5] = {Ly.beta, nullptr, B.x, B.keep_beta + kr * g.n_head, nullptr, 1.0f, gf::kTypeBF16, g.n_embd,
                    g.n_head};
            if (!gf::mv_rows(j, 6, T, s)) { err = "glm mtp: kda projections"; return false; }
            if (F->prof_on) F->mark("v_kda_proj_mv");
            // the convs and the gates of every row (each row's history: the live one, then the rows before it), each
            // row's history after it into conv_after (the live one is left as it was until the commit)
            gf::KdaPrepArgs k;
            for (int i = 0; i < 3; ++i) {
                k.proj[i] = B.proj[i];
                k.conv_w[i] = Ly.conv[i];
            }
            k.out[0] = B.kq;
            k.out[1] = B.keep_k + kr * DI;
            k.out[2] = B.keep_v + kr * DI;
            k.conv_state = state_ + kda_conv_[(size_t) il];
            k.conv_state_out = B.conv_after + kr * 3 * DI * hist;
            k.fa = B.fa;
            k.ga = B.ga;
            k.f_b = Ly.f_b;
            k.g_b = Ly.g_b;
            k.dt_bias = Ly.dt_bias;
            k.ssm_a = Ly.ssm_a;
            k.lower_bound = g.kda_lb;
            k.g1 = B.keep_g1 + kr * DI;
            k.g2 = B.g2;
            k.n_head = g.n_head;
            k.head_dim = g.kda_head_dim;
            k.d_conv = g.d_conv;
            gf::kda_prep_rows(k, T, s);
            if (F->prof_on) F->mark("v_kda_prep");
            // the recurrence over the rows from the live state, which stays as it was (the commit advances it)
            gf::kda_rec_rows(B.kq, B.keep_k + kr * DI, B.keep_v + kr * DI, B.keep_g1 + kr * DI,
                             B.keep_beta + kr * g.n_head, state_ + kda_S_[(size_t) il], B.g2, Ly.ssm_norm, g.norm_eps,
                             g.n_head, g.kda_head_dim, T, false, B.gated_q, s);
            if (F->prof_on) F->mark("v_kda_rec");
            gf::MvJob o = {Ly.out.q, B.gated_q, nullptr, B.mixer, nullptr, 1.0f, Ly.out.type, (int) DI, g.n_embd};
            if (!gf::mv_rows(&o, 1, T, s)) { err = "glm mtp: kda output"; return false; }
            if (F->prof_on) F->mark("v_kda_out_mv");
        } else {
            if (!fast_dsa_rows(il, p0, T, err)) return false;
        }
        // ---- the FFN-side read (fused with the mixer's write)
        hc_all(B.mixer, Ly.hc_ffn_fn, Ly.hc_ffn_scale, Ly.hc_ffn_base, Ly.ffn_norm);
        cur = 1 - cur;
        if (F->prof_on) F->mark("v_hc_ffn");
        if (!Ly.moe) {
            gf::MvJob j[2];
            j[0] = {Ly.ffn_gate.q, B.xq, nullptr, B.dg, nullptr, 1.0f, Ly.ffn_gate.type, g.n_embd, g.n_ff_dense};
            j[1] = {Ly.ffn_up.q, B.xq, nullptr, B.du, nullptr, 1.0f, Ly.ffn_up.type, g.n_embd, g.n_ff_dense};
            if (!gf::mv_rows(j, 2, T, s)) { err = "glm mtp: dense ffn"; return false; }
            for (int t = 0; t < T; ++t)
                gf::swiglu_q8(B.dg + (size_t) t * g.n_ff_dense, B.du + (size_t) t * g.n_ff_dense, g.swiglu_shexp,
                              g.n_ff_dense, (uint8_t*) B.dhq + (size_t) t * q8_bytes(g.n_ff_dense), s);
            gf::MvJob dn = {Ly.ffn_down.q, B.dhq, nullptr, B.ffn, nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense,
                            g.n_embd};
            if (!gf::mv_rows(&dn, 1, T, s)) { err = "glm mtp: dense ffn down"; return false; }
            if (F->prof_on) F->mark("v_dense_ffn");
        } else {
            if (!fast_moe_rows(il, T, err)) return false;
        }
        // Windows HIP: the queued layers go to the GPU now (the first at once, then every few: fast_layers')
        if (const int fe = glmfast::submit_every(); fe > 0 && (il - l0_) % fe == 0) glmfast::submit_queued(s);
    }
    // the last layer's write half; each row's residual back in its first buffer
    for (int t = 0; t < T; ++t) {
        gf::hc_post(B.ffn + (size_t) t * E, Rrow(t, cur), B.post + (size_t) t * 8, B.comb + (size_t) t * 16, g.n_embd,
                    Rrow(t, 1 - cur), s);
        if (1 - cur != 0)
            cudaMemcpyAsync(Rrow(t, 0), Rrow(t, 1), (size_t) hcE * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
    if (F->prof_on) F->mark("v_hc_post_end");
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm mtp: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

// the window [toks[0] .. toks[T-1]] at positions p0 .. through every part, the logits and the final hidden states of
// every row on the last part (rows.logits / rows.head_x); positions advance only at the commit
bool Glm5Model::fast_rows(const int32_t* toks, int T, int64_t p0, std::string& err) {
    const Glm5Geometry& g = g_;
    const int64_t E = g.n_embd, hcE = (int64_t) g.hc * E;
    // between tokens (every device idle): finished promotions go live in the device tables
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm mtp sync", err) || !m->fast_->route_ok(err) ||
            !m->fast_boundary(err)) {
            cudaSetDevice(dev_);
            return false;
        }
        if (m->fast_->prof_on) m->fast_->collect();
    }
    cudaSetDevice(dev_);
    FastState* F = fast_;
    auto& B = F->rows;
    // ---- the embedding rows (host-dequantized from the shard mapping) -> each row's 4 streams
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm mtp: no embedding dequantizer";
        return false;
    }
    const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, g.n_embd);
    for (int t = 0; t < T; ++t) {
        float* dst = B.emb_h + (size_t) t * E;
        tt->to_float(pack_emb_src_ + (size_t) toks[t] * row_b, dst, g.n_embd);
        if (const float* img = image_row(p0 + t)) std::memcpy(dst, img, (size_t) E * sizeof(float));
    }
    cudaMemcpyAsync(B.emb, B.emb_h, (size_t) T * E * sizeof(float), cudaMemcpyHostToDevice, F->cs);
    for (int t = 0; t < T; ++t) gf::embed_streams(B.emb + (size_t) t * E, B.R + (size_t) t * 2 * hcE, g.n_embd, F->cs);
    if (!fast_layers_rows(p0, T, err)) return false;
    // each later part: the rows' residuals cross over through the previous part's pinned rows (device-ordered by an
    // event, the host never waits between the parts)
    Glm5Model* tail = this;
    for (Glm5Model* Bm = split_next_.get(); Bm != nullptr; Bm = Bm->split_next_.get()) {
        FastState* FA = tail->fast_;
        FastState* FB = Bm->fast_;
        cudaSetDevice(tail->dev_);
        cudaMemcpy2DAsync(FA->rows.hop_h, (size_t) hcE * sizeof(float), FA->rows.R, (size_t) 2 * hcE * sizeof(float),
                          (size_t) hcE * sizeof(float), (size_t) T, cudaMemcpyDeviceToHost, FA->cs);
        cudaEventRecord(FA->ev_hop, FA->cs);
        cudaSetDevice(Bm->dev_);
        cudaStreamWaitEvent(FB->cs, FA->ev_hop, 0);
        cudaMemcpy2DAsync(FB->rows.R, (size_t) 2 * hcE * sizeof(float), FA->rows.hop_h, (size_t) hcE * sizeof(float),
                          (size_t) hcE * sizeof(float), (size_t) T, cudaMemcpyHostToDevice, FB->cs);
        if (!Bm->fast_layers_rows(p0, T, err)) return false;
        tail = Bm;
    }
    // ---- the head over the rows (one read of the output weight)
    FastState* FT = tail->fast_;
    auto& BT = FT->rows;
    cudaSetDevice(tail->dev_);
    for (int t = 0; t < T; ++t)
        gf::head_prep(BT.R + (size_t) t * 2 * hcE, tail->w_.at("output_norm.weight"), g.norm_eps, g.n_embd,
                      BT.head_x + (size_t) t * E, (uint8_t*) BT.head_xq + (size_t) t * q8_bytes(E), FT->cs);
    const WSlot& ow = tail->ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, BT.head_xq, BT.head_x, BT.logits, nullptr, 1.0f, ow.type, g.n_embd, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv_rows(&o, 1, T, FT->cs)) {
        err = "glm mtp: output head (type " + std::to_string(ow.type) + ")";
        return false;
    }
    if (FT->prof_on) FT->mark("v_head");
    cudaSetDevice(dev_);
    return true;
}

// every part: the recurrent layers' states advanced by the first `keep` rows of the last window (the verify left them
// as they were), their conv histories to the kept row's; the position after the kept rows
bool Glm5Model::mtp_commit(int T, int keep, std::string& err) {
    const Glm5Geometry& g = g_;
    const int64_t DI = g.d_inner(), hist = g.d_conv - 1;
    const int64_t p_end = pos_ + keep;
    (void) T;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        auto& B = F->rows;
        cudaSetDevice(m->dev_);
        for (int il = m->l0_; il < m->l1_; ++il) {
            const int li = B.rec[(size_t) il];
            if (li < 0) continue;
            const size_t kr = (size_t) li * B.cap;
            gf::kda_rec_rows(nullptr, B.keep_k + kr * DI, B.keep_v + kr * DI, B.keep_g1 + kr * DI,
                             B.keep_beta + kr * g.n_head, m->state_ + m->kda_S_[(size_t) il], nullptr,
                             F->L[(size_t) il].ssm_norm, g.norm_eps, g.n_head, g.kda_head_dim, keep, true, nullptr,
                             F->cs);
            cudaMemcpyAsync(m->state_ + m->kda_conv_[(size_t) il], B.conv_after + (kr + keep - 1) * 3 * DI * hist,
                            (size_t) 3 * DI * hist * sizeof(float), cudaMemcpyDeviceToDevice, F->cs);
        }
        m->pos_ = p_end;
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            cudaSetDevice(dev_);
            err = std::string("glm mtp commit: ") + cudaGetErrorString(e);
            return false;
        }
    }
    cudaSetDevice(dev_);
    return true;
}

// ---------------------------------------------------------------- the NextN block's cache
// entries p0 .. p0 + n - 1 of the draft block's DSA caches: entry p0 + i from the trunk's hidden state h + i * n_embd
// and the token next[i] (the one at p0 + i + 1) - the cache-writing half of the block (glm_prefill.cu's for a prompt):
// eh_proj, the attention norm, kv_a and the indexer's key and gate.  This part carries the block.
bool Glm5Model::mtp_cache_rows(int64_t p0, int n, const float* h, const int32_t* next, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (F == nullptr || mtp_il_ < 0 || n <= 0) return true;
    auto& B = F->rows;
    if (n > B.cap) {
        err = "glm mtp: more cache entries than the window holds";
        return false;
    }
    cudaStream_t s = F->cs;
    const int64_t E = g.n_embd;
    const auto& Ly = F->L[(size_t) mtp_il_];
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm mtp: no embedding dequantizer";
        return false;
    }
    cudaSetDevice(dev_);
    // the pinned rows may still feed an earlier copy of this stream
    if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "glm mtp cache sync", err)) return false;
    const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, g.n_embd);
    for (int i = 0; i < n; ++i) tt->to_float(pack_emb_src_ + (size_t) next[i] * row_b, B.emb_h + (size_t) i * E, g.n_embd);
    cudaMemcpyAsync(B.emb, B.emb_h, (size_t) n * E * sizeof(float), cudaMemcpyHostToDevice, s);
    for (int i = 0; i < n; ++i)
        gf::mtp_in(B.emb + (size_t) i * E, h + (size_t) i * E, Ly.enorm, Ly.hnorm, g.norm_eps, g.n_embd,
                   (uint8_t*) B.catq + (size_t) i * q8_bytes(2 * E), s);
    gf::MvJob eh = {Ly.eh.q, B.catq, nullptr, B.hid, nullptr, 1.0f, Ly.eh.type, (int) (2 * E), g.n_embd};
    if (!gf::mv_rows(&eh, 1, n, s)) {
        err = "glm mtp: eh_proj rows";
        return false;
    }
    for (int i = 0; i < n; ++i)
        gf::rms_q8(B.hid + (size_t) i * E, nullptr, Ly.attn_norm, g.norm_eps, g.n_embd, B.x + (size_t) i * E,
                   (uint8_t*) B.xq + (size_t) i * q8_bytes(E), s);
    gf::MvJob j[3];
    j[0] = {Ly.kv_a.q, B.xq, B.x, B.kv_raw, nullptr, 1.0f, Ly.kv_a.type, g.n_embd, g.kv_lora};
    j[1] = {Ly.idx_k, nullptr, B.x, B.ik_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[2] = {Ly.idx_gate, nullptr, B.x, B.ig_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    if (!gf::mv_rows(j, 3, n, s)) {
        err = "glm mtp: the cache rows' projections";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        gf::DsaPrepArgs d;
        d.qr_raw = nullptr;   // the caches only
        d.q_lora = g.q_lora;
        d.kv_raw = B.kv_raw + (size_t) i * g.kv_lora;
        d.kv_norm = Ly.kv_a_norm;
        d.lat = (uint16_t*) (state_ + dsa_lat_[(size_t) mtp_il_]);
        d.kv_lora = g.kv_lora;
        d.lat_q8 = lat_q8_;
        d.ik_raw = B.ik_raw + (size_t) i * g.idx_key;
        d.k_norm_w = Ly.k_norm_w;
        d.k_norm_b = Ly.k_norm_b;
        d.ik_cache = state_ + dsa_ik_[(size_t) mtp_il_];
        d.ig_raw = B.ig_raw + (size_t) i * g.idx_key;
        d.ig_cache = state_ + dsa_ig_[(size_t) mtp_il_];
        d.ape = Ly.ape;
        d.pooled = state_ + dsa_pool_[(size_t) mtp_il_];
        d.idx_key = g.idx_key;
        d.kpool = g.idx_kpool;
        d.ring = ik_ring_;
        d.p = (int) (p0 + i);
        d.eps = g.norm_eps;
        gf::dsa_prep(d, s);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm mtp cache: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

// the token path (fast_token at position p, this = the part with the NextN block): entry p - 1 from the trunk's hidden
// state of p - 1 (head_x, when it is that one) and the token at p - what llama.cpp's process() does for a decoded token
bool Glm5Model::mtp_fill_prev(int32_t token, int64_t p, std::string& err) {
    if (fast_ == nullptr || mtp_il_ < 0 || fast_->rows.cap < 1 || p < 1 || mtp_hx_pos_ != p - 1) return true;
    return mtp_cache_rows(p - 1, 1, fast_->head_x, &token, err);
}

// ---------------------------------------------------------------- the draft length
// The length that gives the most tokens per second on this machine: each length's round time (an EMA of the measured
// wall time) against the tokens a round of it yields - 1 + a1 + a1 a2 + ..., a_i the share of rounds that reached
// draft i (all before it accepted) and accepted it.  Every length is tried a few rounds first, then one round in 32
// tries a neighbour of the best (the estimates of lengths not in use go stale).  STRATA_GLM_MTP_ADAPT=0: always n_cap.
int Glm5Model::mtp_choose(int n_cap) {
    static const bool fixed = [] {
        const char* e = getenv("STRATA_GLM_MTP_ADAPT");
        return e != nullptr && std::atoi(e) == 0;
    }();
    // STRATA_GLM_MTP_FIXED=<n> (A/B and tests): every round drafts n, the buffers sized as usual - n = 0 is the
    // token path with the same memory layout as the drafting runs
    static const int forced_n = [] {
        const char* e = getenv("STRATA_GLM_MTP_FIXED");
        return e != nullptr ? std::max(0, std::atoi(e)) : -1;
    }();
    if (forced_n >= 0) return std::min(forced_n, n_cap);
    if (n_cap <= 0) return 0;
    if (fixed) return n_cap;
    ++mtp_round_no_;
    // a length runs in stints: a round's time holds the tiers' work left from the round before (its routes' answers
    // and moves), so the first two rounds after a change are not measured (decode_mtp) - a stint gives its length
    // measured rounds
    if (mtp_stint_left_ > 0) {
        --mtp_stint_left_;
        return mtp_stint_n_;
    }
    constexpr uint64_t kTry = 6;
    for (int n = n_cap; n >= 0; --n)
        if (mtp_seen_[n] < kTry) {
            mtp_stint_n_ = n;
            mtp_stint_left_ = (int) (kTry - mtp_seen_[n]) + 1;   // + 2 unmeasured, this one included
            return n;
        }
    double best = -1.0;
    int bn = 0;
    for (int n = 0; n <= n_cap; ++n) {
        double e = 1.0, run = 1.0;
        for (int i = 1; i <= n; ++i) {
            run *= mtp_reach_[i] > 0 ? (double) mtp_hit_[i] / (double) mtp_reach_[i] : 0.0;
            e += run;
        }
        const double rate = e / std::max(1e-3, mtp_ms_[n]);
        if (rate > best) {
            best = rate;
            bn = n;
        }
    }
    // a probe of a neighbour every mtp_gap_ rounds: the gap doubles (to 1024) while the best length stays, so a
    // machine where drafting does not pay spends ever fewer rounds finding that out again
    if (bn != mtp_best_) {
        mtp_best_ = bn;
        mtp_gap_ = 32;
        mtp_next_probe_ = mtp_round_no_ + mtp_gap_;
    }
    if (mtp_round_no_ >= mtp_next_probe_) {
        mtp_gap_ = std::min<uint64_t>(1024, mtp_gap_ * 2);
        mtp_next_probe_ = mtp_round_no_ + mtp_gap_;
        int alt = (mtp_probe_up_ = !mtp_probe_up_) ? bn + 1 : bn - 1;
        if (alt < 0 || alt > n_cap) alt = bn + 1 <= n_cap ? bn + 1 : bn - 1;
        if (alt >= 0 && alt <= n_cap) {
            mtp_stint_n_ = alt;
            mtp_stint_left_ = 3;   // a probe: four rounds, two of them measured
            return alt;
        }
    }
    return bn;
}

// ---------------------------------------------------------------- the decode
bool Glm5Model::decode_mtp(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit,
                           int64_t& produced, std::string& err) {
    produced = 0;
    if (!mtp_ready()) {
        err = "glm mtp: the model has no NextN block on its last part, or no verify window";
        return false;
    }
    Glm5Model* TL = this;
    int cap = fast_->rows.cap;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        TL = m;
        cap = std::min(cap, m->fast_->rows.cap);
    }
    FastState* FT = TL->fast_;
    const Glm5Geometry& g = g_;
    const int64_t E = g.n_embd;
    const int n_cap = std::min(glm_mtp_draft_cap(), cap - 1);
    static const bool trace = getenv("STRATA_GLM_MTP_TRACE") != nullptr;
    const auto check = [&]() -> bool {
        if (gf::launch_errors() > 0) {
            err = "glm mtp: a kernel launch failed (see stderr)";
            return false;
        }
        return true;
    };
    // the first token: the prompt's last forward left its logits (and greedy argmax) on the last part
    int y = -1;
    if (sp.greedy && last_tok_ >= 0) {
        y = last_tok_;
    } else {
        y = TL->fast_sample(sp, err);
        if (y < 0) return false;
    }
    y = forced(y);
    sp.counter += 1;
    last_tok_ = y;
    ++produced;
    if (!emit(y) || produced >= max_new || pos_ + 1 >= max_ctx_) return true;
    int64_t p0 = pos_;   // where y goes
    mtp_same_ = 0;       // (a request's first rounds carry its prompt's tier work: not measured)
    std::vector<int32_t> win((size_t) n_cap + 1);
    bool ok = true;
    // the drafting rounds' host time: drafts, verify (to the tokens), accept + commit + cache; [3]: the drafts' wait
    // for the device before their first step
    double ph[4] = {0, 0, 0, 0};
    int64_t n_ph = 0;
    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto ms_since = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    };
    for (;;) {
        const auto t0 = std::chrono::steady_clock::now();
        int n = mtp_choose(n_cap);
        n = (int) std::min<int64_t>(n, max_new - produced - 1);
        n = (int) std::min<int64_t>(n, max_ctx_ - 1 - p0);
        if (TL->mtp_hx_pos_ != p0 - 1) n = 0;   // (no trunk state to draft from: a plain token)
        n = std::max(n, 0);
        win[0] = y;
        int nd = 0;
        if (n > 0) {
            // ---- 1. the drafts: the NextN block at p0 - 1 + i on (its last hidden state, the last token)
            // (the tiers' boundary runs before the verify: the drafts route with the tables as they are -
            // STRATA_GLM_MTP_DRAFT_BOUNDARY=1 runs one before them too)
            static const bool draft_boundary = getenv("STRATA_GLM_MTP_DRAFT_BOUNDARY") != nullptr;
            cudaSetDevice(TL->dev_);
            if (!glmfast::cuda_ok(cudaStreamSynchronize(FT->cs), "glm mtp draft sync", err) || !FT->route_ok(err) ||
                (draft_boundary && !TL->fast_boundary(err))) {
                ok = false;
                break;
            }
            ph[3] += ms_since(t0);
            int32_t tok = y;
            for (int i = 0; i < n; ++i) {
                if (!TL->fast_mtp(p0 - 1 + i, tok, err) ||
                    !glmfast::wait_event(FT->ev_mtp, "glm mtp draft", err) || !FT->route_ok(err)) {
                    ok = false;
                    break;
                }
                tok = FT->mtp_tok_h[0];
                win[(size_t) i + 1] = tok;
                ++nd;
            }
            TL->mtp_hx_pos_ = -1;   // (head_x holds the block's hidden state now)
            cudaSetDevice(dev_);
            if (!ok) break;
        }
        const int T = 1 + nd;
        int keep = 1, acc = 0;
        bool more = true;
        const auto t1 = now();
        const double ph1_before = ph[1];
        if (T == 1) {
            // ---- a plain token (the token path keeps the NextN cache: entry p0 - 1)
            if (!fast_token(y, err)) {
                ok = false;
                break;
            }
            int y2 = -1;
            if (sp.greedy) {
                y2 = last_tok_;
            } else {
                y2 = TL->fast_sample(sp, err);
                if (y2 < 0) {
                    ok = false;
                    break;
                }
            }
            y2 = forced(y2);
            sp.counter += 1;
            last_tok_ = y2;
            ++produced;
            y = y2;
            more = emit(y2) && produced < max_new && p0 + 2 < max_ctx_;
        } else {
            // ---- 2. the verify: the window through every part, then every row's token
            if (!fast_rows(win.data(), T, p0, err)) {
                ok = false;
                break;
            }
            auto& BT = FT->rows;
            cudaSetDevice(TL->dev_);
            if (sp.greedy) {
                gf::argmax_rows(BT.logits, (int) g.n_vocab, T, BT.tok, FT->cs);
            } else {
                strata::kernels::sample_tokens(BT.logits, T, (int) g.n_vocab, nullptr, 0, sp, BT.tok, FT->cs);
            }
            cudaMemcpyAsync(BT.tok_h, BT.tok, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost, FT->cs);
            cudaEventRecord(FT->ev_done, FT->cs);
            if (!glmfast::wait_event(FT->ev_done, "glm mtp verify", err)) {
                ok = false;
                break;
            }
            for (Glm5Model* m = this; m != nullptr && ok; m = m->split_next_.get())
                if (!m->fast_->route_ok(err)) ok = false;
            cudaSetDevice(dev_);
            if (!ok || !check()) {
                ok = false;
                break;
            }
            const auto t2 = now();
            ph[0] += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ph[1] += std::chrono::duration<double, std::milli>(t2 - t1).count();
            // ---- 3. accept: row t's token is the truth for p0 + t + 1; draft t + 1 stands while they match
            for (int t = 0; t < T; ++t) {
                const int yy = forced(BT.tok_h[t]);
                sp.counter += 1;
                last_tok_ = yy;
                ++produced;
                keep = t + 1;
                y = yy;
                more = emit(yy) && produced < max_new && p0 + keep + 1 < max_ctx_;
                if (!more || t + 1 >= T) break;
                ++mtp_reach_[t + 1];   // draft t + 1 is compared (every draft before it stood)
                if (yy != win[(size_t) t + 1]) break;
                ++mtp_hit_[t + 1];
                ++acc;
            }
            // ---- 4. the commit: the recurrent states advanced by the kept rows (positions p0 .. p0 + keep - 1)
            if (!mtp_commit(T, keep, err)) {
                ok = false;
                break;
            }
            // ---- 5. the NextN cache: entries p0 .. p0 + keep - 2 from the kept rows' trunk states and the accepted
            //         drafts; entry p0 + keep - 1 is the next round's first draft step (or the token path's)
            if (keep > 1 && !TL->mtp_cache_rows(p0, keep - 1, BT.head_x, win.data() + 1, err)) {
                ok = false;
                break;
            }
            cudaSetDevice(TL->dev_);
            cudaMemcpyAsync(FT->head_x, BT.head_x + (size_t) (keep - 1) * E, (size_t) E * sizeof(float),
                            cudaMemcpyDeviceToDevice, FT->cs);
            TL->mtp_hx_pos_ = p0 + keep - 1;
            cudaSetDevice(dev_);
            ph[2] += ms_since(t1) - (ph[1] - ph1_before);
            ++n_ph;
        }
        // the statistics and the adaptive length's measurements
        ++mtp_rounds_;
        ++mtp_round_n_[nd];
        spec_steps_ += (uint64_t) nd;
        spec_hits_ += (uint64_t) acc;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        // measured from the third round of a run of one length on (mtp_choose's stints)
        if (nd == mtp_prev_n_) {
            ++mtp_same_;
        } else {
            mtp_prev_n_ = nd;
            mtp_same_ = 0;
        }
        if (mtp_same_ >= 2) {
            mtp_ms_[nd] = mtp_seen_[nd] == 0 ? ms : 0.9 * mtp_ms_[nd] + 0.1 * ms;
            ++mtp_seen_[nd];
        }
        if (trace)
            std::fprintf(stderr, "glm mtp: p0 %lld drafts %d kept %d accepted %d (%.1f ms)\n", (long long) p0, nd, keep,
                         acc, ms);
        p0 += keep;
        if (!more) break;
    }
    // drain: every part idle, the routes checked
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        std::string drain_err;
        if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm mtp drain", drain_err) ||
            !m->fast_->route_ok(drain_err)) {
            if (ok) err = drain_err;
            ok = false;
        }
    }
    cudaSetDevice(dev_);
    if (n_ph > 0)
        std::fprintf(stderr, "glm mtp: a drafting round's host time - drafts %.2f ms (%.2f of them waiting for the "
                             "device), verify %.2f ms, accept + commit + cache %.2f ms (%lld rounds)\n", ph[0] / n_ph,
                     ph[3] / n_ph, ph[1] / n_ph, ph[2] / n_ph, (long long) n_ph);
    if (ok && !check()) ok = false;
    return ok;
}

}  // namespace strata::core
