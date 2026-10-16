// src/core/glm_rows.cu - several conversations decoded together (Glm5Model::multi_rows / multi_issue_rows).
//
// A decode token reads its part's weights once.  The one-GPU path takes ONE token of each of n conversations through
// the model together, reading the dense weights once for all; the split path does the same on each part per group.
// The attention projections, shared experts, dense FFN and head go through mv_nt, fetching each weight row once for the
// n activations and giving each activation the very bits mv() gives it alone.  What
// depends on a conversation's own history runs per row, through the token path's own kernels: the residual mixing
// (mHC), the KDA recurrence, the DSA indexer and attention against that conversation's caches, and the routed
// experts (the tiers serve the n routes one after another, as they serve the layers of a token).  So each row's
// token is the one the token path gives that conversation alone - as long as the tiers hold the same experts
// (greedy with the CPU lane off: the multi bench compares them).
//
// Each part's row activations live in its own arena (rows_alloc, at load, before the expert pool takes the rest of the
// card); the per-row scratch the token path's kernels use in between (the conv outputs, the gates, the DSA scores and
// cells, the experts' h) is the token path's own, used by one row after the other on that part's stream.
#include "glm_fast_state.hpp"

#include "strata/kernels/sampler.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {
constexpr int kQ8 = 36;   // bytes per q8_1 block
size_t up256(size_t b) { return (b + 255u) & ~(size_t) 255u; }
const float* image_row_in(const std::vector<int64_t>& pos, const std::vector<float>& rows, int64_t p, int n_embd) {
    const auto it = std::lower_bound(pos.begin(), pos.end(), p);
    if (it == pos.end() || *it != p) return nullptr;
    return rows.data() + (size_t) (it - pos.begin()) * (size_t) n_embd;
}
float* row_at(float* b, int64_t per, int i) { return b + (size_t) i * (size_t) per; }
uint8_t* rowq_at(uint8_t* b, int64_t n_in, int i) { return b + (size_t) i * (size_t) (n_in / 32) * kQ8; }
}  // namespace

// rows_max() rows of each activation, one row after the other (mv_nt reads and writes them so)
struct Glm5Model::RowsState {
    void* arena = nullptr;
    size_t bytes = 0;
    float *x = nullptr, *mixer = nullptr, *ffn = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    uint8_t* xq = nullptr;
    float* proj[3] = {nullptr, nullptr, nullptr};
    float *fa = nullptr, *ga = nullptr, *beta = nullptr;
    uint8_t* gated_q = nullptr;
    float *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *ik_raw = nullptr, *ig_raw = nullptr, *iw = nullptr;
    float *q = nullptr, *iq = nullptr;
    uint8_t *qr_q = nullptr, *attn_q = nullptr;
    float *rlog = nullptr, *sh_g = nullptr, *sh_u = nullptr, *dg = nullptr, *du = nullptr;
    uint8_t* dhq = nullptr;
    float* head_x = nullptr;
    uint8_t* head_xq = nullptr;
    float* logits = nullptr;
    int* tok = nullptr;
    int* tok_h = nullptr;   // pinned
    cudaEvent_t ev_done = nullptr;
};
// one row's sequence on this part: state, position, caches, KV stream and image row
struct Glm5Model::RowSeq {
    int seq = 0;
    float* state = nullptr;
    float *Rc = nullptr, *Ro = nullptr;   // the residual pair (the token path's convention, per row)
    int64_t p = 0;
    int ring = 0;
    const std::vector<int64_t> *kda_S = nullptr, *kda_conv = nullptr, *dsa_lat = nullptr, *dsa_ik = nullptr,
                               *dsa_ig = nullptr, *dsa_pool = nullptr;
    const float* img = nullptr;
    const KvStream* kv = nullptr;   // KV streaming: that sequence's stream (null: not streamed)
};

int Glm5Model::rows_max() { return gf::kMaxMvNt; }

bool Glm5Model::rows_alloc(std::string& err) {
    if (rows_ != nullptr) return true;
    const Glm5Geometry& g = g_;
    const size_t R = (size_t) rows_max();
    const int64_t E = g.n_embd, DI = g.d_inner(), FFs = (int64_t) g.n_ff_exp * g.n_shared;
    auto* S = new RowsState();
    uint8_t* base = nullptr;
    size_t at = 0;
    const auto take = [&](size_t row_bytes) -> uint8_t* {   // R rows of row_bytes (256-aligned as a block)
        const size_t off = at;
        at += up256(row_bytes * R);
        return base == nullptr ? nullptr : base + off;
    };
    const auto f32 = [&](int64_t n) { return (float*) take((size_t) n * sizeof(float)); };
    const auto q8 = [&](int64_t n) { return take((size_t) n / 32 * kQ8); };
    const auto carve = [&] {
        S->x = f32(E);
        S->xq = q8(E);
        S->mixer = f32(E);
        S->ffn = f32(E);
        S->pre = f32(8);
        S->post = f32(8);
        S->comb = f32(16);
        for (auto& p : S->proj) p = f32(DI);
        S->fa = f32(g.kda_head_dim);
        S->ga = f32(g.kda_head_dim);
        S->beta = f32(g.n_head);
        S->gated_q = q8(DI);
        S->qr_raw = f32(g.q_lora);
        S->qr = f32(g.q_lora);
        S->qr_q = q8(g.q_lora);
        S->kv_raw = f32(g.kv_lora);
        S->ik_raw = f32(g.idx_key);
        S->ig_raw = f32(g.idx_key);
        S->iw = f32(g.idx_heads);
        S->q = f32((int64_t) g.n_head * g.qk_nope);
        S->iq = f32((int64_t) g.idx_heads * g.idx_key);
        S->attn_q = q8((int64_t) g.n_head * g.v_head);
        S->rlog = f32(g.n_expert);
        S->sh_g = f32(FFs);
        S->sh_u = f32(FFs);
        S->dg = f32(g.n_ff_dense);
        S->du = f32(g.n_ff_dense);
        S->dhq = q8(g.n_ff_dense);
        S->head_x = f32(E);
        S->head_xq = q8(E);
        S->logits = f32(g.n_vocab);
        S->tok = (int*) take(sizeof(int));
    };
    carve();   // the sizes
    cudaSetDevice(dev_);
    if (cudaMalloc(&base, at) != cudaSuccess) {
        cudaGetLastError();
        delete S;
        err = "glm rows: the activations of " + std::to_string(R) + " rows (" + std::to_string(at >> 20) +
              " MB) do not fit on CUDA" + std::to_string(dev_);
        return false;
    }
    cudaMemset(base, 0, at);
    S->arena = base;
    S->bytes = at;
    at = 0;
    carve();   // the pointers
    if (cudaHostAlloc((void**) &S->tok_h, R * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        cudaEventCreateWithFlags(&S->ev_done, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        rows_ = S;
        rows_destroy();
        err = "glm rows: the token buffers did not allocate";
        return false;
    }
    rows_ = S;
    return true;
}

void Glm5Model::rows_destroy() {
    RowsState* S = rows_;
    if (S == nullptr) return;
    cudaSetDevice(dev_);
    if (S->arena) cudaFree(S->arena);
    if (S->tok_h) cudaFreeHost(S->tok_h);
    if (S->ev_done) cudaEventDestroy(S->ev_done);
    delete S;
    rows_ = nullptr;
}

bool Glm5Model::rows_bind(const int* seqs, int n, RowSeq* rw, std::string& err) {
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    for (int i = 0; i < n; ++i) {
        const int sq = seqs[i];
        if (sq < 0 || sq >= seq_count()) {
            err = "glm rows: no sequence " + std::to_string(sq);
            return false;
        }
        for (int k = 0; k < i; ++k)
            if (seqs[k] == sq) {
                err = "glm rows: sequence " + std::to_string(sq) + " twice in one step";
                return false;
            }
        RowSeq& r = rw[i];
        r.seq = sq;
        int64_t max_ctx = 0;
        if (sq == cur_seq_) {
            r.state = state_;
            r.p = pos_;
            r.ring = ik_ring_;
            r.kda_S = &kda_S_;
            r.kda_conv = &kda_conv_;
            r.dsa_lat = &dsa_lat_;
            r.dsa_ik = &dsa_ik_;
            r.dsa_ig = &dsa_ig_;
            r.dsa_pool = &dsa_pool_;
            r.kv = kv_;
            r.img = image_row(r.p);
            max_ctx = max_ctx_;
        } else {
            const SeqCtx& c = seq_[(size_t) sq];
            r.state = c.state;
            r.p = c.pos;
            r.ring = c.ik_ring;
            r.kda_S = &c.kda_S;
            r.kda_conv = &c.kda_conv;
            r.dsa_lat = &c.dsa_lat;
            r.dsa_ik = &c.dsa_ik;
            r.dsa_ig = &c.dsa_ig;
            r.dsa_pool = &c.dsa_pool;
            r.kv = c.kv;
            r.img = image_row_in(c.img_pos, c.img_rows, r.p, E);
            max_ctx = c.max_ctx;
        }
        if (r.p + 1 >= max_ctx) {
            err = "glm rows: the context of sequence " + std::to_string(sq) + " is full";
            return false;
        }
        r.Rc = r.state;
        r.Ro = r.state + (int64_t) g.hc * E;
    }
    return true;
}

bool Glm5Model::rows_embed(const RowSeq* rw, const int32_t* tokens, int n, std::string& err) {
    FastState* F = fast_;
    const int E = g_.n_embd;
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm rows: no embedding dequantizer";
        return false;
    }
    cudaSetDevice(dev_);
    cudaStream_t s = F->cs;
    // ---- the embeddings: each row's token (or image row) into its own residual's 4 streams
    const size_t emb_row = ggml_row_size((ggml_type) pack_emb_type_, E);
    for (int i = 0; i < n; ++i) {
        SeqCtx& c = seq_[(size_t) rw[i].seq];
        // its own pinned row: the step before has completed, so no copy still reads it
        tt->to_float(pack_emb_src_ + (size_t) tokens[i] * emb_row, c.emb_h, E);
        if (rw[i].img != nullptr) std::memcpy(c.emb_h, rw[i].img, (size_t) E * sizeof(float));
        cudaMemcpyAsync(F->emb, c.emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
        gf::embed_streams(F->emb, rw[i].state, E, s);
    }
    return true;
}

bool Glm5Model::rows_layers(RowSeq* rw, int n, std::string& err) {
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd, DI = g.d_inner(), FFs = g.n_ff_exp * g.n_shared;
    cudaSetDevice(dev_);
    cudaStream_t s = F->cs;
    auto& rows = row_at;
    auto& rows_q = rowq_at;
    const auto mv = [&](gf::MvJob* j, int nj, const char* what) -> bool {
        if (gf::mv_nt(j, nj, n, s)) return true;
        err = std::string("glm rows: ") + what;
        return false;
    };
    const float* no_bias[gf::kAhead] = {nullptr, nullptr, nullptr, nullptr};
    bool pf_pending = false;   // (no prefetch in a batched step)
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        gf::HcArgs h;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = E;
        h.part = F->part;
        h.counter = F->counter;
        const auto hc_row = [&](int i, const float* block_out) {
            h.block_out = block_out;
            h.R_old = rw[i].Rc;
            h.R_new = rw[i].Ro;
            h.post_in = rows(S->post, 8, i);
            h.comb_in = rows(S->comb, 16, i);
            h.pre = rows(S->pre, 8, i);
            h.post = rows(S->post, 8, i);
            h.comb = rows(S->comb, 16, i);
            h.x = rows(S->x, E, i);
            h.xq = rows_q(S->xq, E, i);
            gf::hc(h, s);
        };
        // ---- the attention-side read (fused with the previous FFN's write), per row
        h.w_fn = Ly.hc_attn_fn;
        h.w_scale = Ly.hc_attn_scale;
        h.w_base = Ly.hc_attn_base;
        h.norm_w = Ly.attn_norm;
        for (int i = 0; i < n; ++i) {
            hc_row(i, il == l0_ ? nullptr : rows(S->ffn, E, i));
            if (il != l0_) std::swap(rw[i].Rc, rw[i].Ro);
        }

        if (Ly.recr) {
            gf::MvJob j[6];
            j[0] = {Ly.q.q, S->xq, S->x, S->proj[0], nullptr, 1.0f, Ly.q.type, E, DI};
            j[1] = {Ly.k.q, S->xq, S->x, S->proj[1], nullptr, 1.0f, Ly.k.type, E, DI};
            j[2] = {Ly.v.q, S->xq, S->x, S->proj[2], nullptr, 1.0f, Ly.v.type, E, DI};
            j[3] = {Ly.f_a, nullptr, S->x, S->fa, nullptr, 1.0f, gf::kTypeBF16, E, g.kda_head_dim};
            j[4] = {Ly.g_a, nullptr, S->x, S->ga, nullptr, 1.0f, gf::kTypeBF16, E, g.kda_head_dim};
            j[5] = {Ly.beta, nullptr, S->x, S->beta, nullptr, 1.0f, gf::kTypeBF16, E, g.n_head};
            if (!mv(j, 6, "kda projections")) return false;
            for (int i = 0; i < n; ++i) {
                gf::KdaPrepArgs k;
                for (int c = 0; c < 3; ++c) {
                    k.proj[c] = rows(S->proj[c], DI, i);
                    k.conv_w[c] = Ly.conv[c];
                    k.out[c] = F->conv[c];
                }
                k.conv_state = rw[i].state + (*rw[i].kda_conv)[(size_t) il];
                k.fa = rows(S->fa, g.kda_head_dim, i);
                k.ga = rows(S->ga, g.kda_head_dim, i);
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
                gf::kda_rec(F->conv[0], F->conv[1], F->conv[2], F->g1, rows(S->beta, g.n_head, i),
                            rw[i].state + (*rw[i].kda_S)[(size_t) il], F->g2, Ly.ssm_norm, g.norm_eps, g.n_head,
                            g.kda_head_dim, rows_q(S->gated_q, DI, i), s);
            }
            gf::MvJob o = {Ly.out.q, S->gated_q, nullptr, S->mixer, nullptr, 1.0f, Ly.out.type, DI, E};
            if (!mv(&o, 1, "kda output")) return false;
        } else {
            const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
            gf::MvJob j[5];
            j[0] = {Ly.q_a.q, S->xq, S->x, S->qr_raw, nullptr, 1.0f, Ly.q_a.type, E, g.q_lora};
            j[1] = {Ly.kv_a.q, S->xq, S->x, S->kv_raw, nullptr, 1.0f, Ly.kv_a.type, E, g.kv_lora};
            j[2] = {Ly.idx_k, nullptr, S->x, S->ik_raw, nullptr, 1.0f, gf::kTypeBF16, E, g.idx_key};
            j[3] = {Ly.idx_gate, nullptr, S->x, S->ig_raw, nullptr, 1.0f, gf::kTypeBF16, E, g.idx_key};
            j[4] = {Ly.idx_proj, nullptr, S->x, S->iw, nullptr, prescale, gf::kTypeBF16, E, g.idx_heads};
            if (!mv(j, 5, "dsa projections")) return false;
            for (int i = 0; i < n; ++i) {
                gf::DsaPrepArgs d;
                d.qr_raw = rows(S->qr_raw, g.q_lora, i);
                d.q_a_norm = Ly.q_a_norm;
                d.qr = rows(S->qr, g.q_lora, i);
                d.qr_q = rows_q(S->qr_q, g.q_lora, i);
                d.q_lora = g.q_lora;
                d.kv_raw = rows(S->kv_raw, g.kv_lora, i);
                d.kv_norm = Ly.kv_a_norm;
                kv_prep_targets(rw[i].kv, il, (uint16_t*) (rw[i].state + (*rw[i].dsa_lat)[(size_t) il]), d);
                d.kv_lora = g.kv_lora;
                d.lat_q8 = lat_q8_;
                d.ik_raw = rows(S->ik_raw, g.idx_key, i);
                d.k_norm_w = Ly.k_norm_w;
                d.k_norm_b = Ly.k_norm_b;
                d.ik_cache = rw[i].state + (*rw[i].dsa_ik)[(size_t) il];
                d.ig_raw = rows(S->ig_raw, g.idx_key, i);
                d.ig_cache = rw[i].state + (*rw[i].dsa_ig)[(size_t) il];
                d.ape = Ly.ape;
                d.pooled = rw[i].state + (*rw[i].dsa_pool)[(size_t) il];
                d.idx_key = g.idx_key;
                d.kpool = g.idx_kpool;
                d.ring = rw[i].ring;
                d.p = (int) rw[i].p;
                d.eps = g.norm_eps;
                gf::dsa_prep(d, s);
            }
            gf::MvJob j2[2];
            j2[0] = {Ly.q_b.q, S->qr_q, S->qr, S->q, nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * g.qk_nope};
            j2[1] = Ly.idx_q_b_x.q != nullptr   // (an EXL3 pack quantizes the indexer's q_b)
                        ? gf::MvJob{Ly.idx_q_b_x.q, S->qr_q, S->qr, S->iq, nullptr, 1.0f, Ly.idx_q_b_x.type, g.q_lora,
                                    g.idx_heads * g.idx_key}
                        : gf::MvJob{Ly.idx_q_b, nullptr, S->qr, S->iq, nullptr, 1.0f, gf::kTypeBF16, g.q_lora,
                                    g.idx_heads * g.idx_key};
            if (!mv(j2, 2, "dsa q")) return false;
            for (int i = 0; i < n; ++i) {
                const int64_t p = rw[i].p;
                const int pool_done = (int) ((p + 1) / g.idx_kpool);
                const float* pooled = rw[i].state + (*rw[i].dsa_pool)[(size_t) il];
                const bool all = pool_done <= g.top_pools_max() && gf::dsa_select_all_on();   // (as fast_dsa)
                if (pool_done > 0 && !all)
                    gf::dsa_score(rows(S->iq, (int64_t) g.idx_heads * g.idx_key, i), pooled,
                                  rows(S->iw, g.idx_heads, i), g.idx_key, g.idx_heads, pool_done, F->score, s);
                const int top_pools = std::min(g.top_pools_max(), pool_done);
                const int n_sel = g.idx_kpool * top_pools + (g.idx_select_tail ? g.idx_kpool - 1 : 0);
                if (all) gf::dsa_select_all(pool_done, g.idx_kpool, n_sel, (int) p, F->cells, s);
                else gf::dsa_select(F->score, pool_done, g.idx_kpool, top_pools, n_sel, (int) p, F->cells, s);
                uint16_t* const lat = (uint16_t*) (rw[i].state + (*rw[i].dsa_lat)[(size_t) il]);
                const int* cells = kv_attend_cells(rw[i].kv, il, lat, F->cells, n_sel, F->slot_cells, s);
                gf::mla(rows(S->q, (int64_t) g.n_head * g.qk_nope, i), Ly.k_b, Ly.v_b, lat, cells, n_sel, g.n_head,
                        g.qk_nope, g.kv_lora, g.v_head, rows_q(S->attn_q, (int64_t) g.n_head * g.v_head, i), s,
                        lat_q8_);
            }
            gf::MvJob o = {Ly.out.q, S->attn_q, nullptr, S->mixer, nullptr, 1.0f, Ly.out.type, g.n_head * g.v_head,
                           E};
            if (!mv(&o, 1, "dsa output")) return false;
        }

        // ---- the FFN-side read (fused with the mixer's write), per row
        h.w_fn = Ly.hc_ffn_fn;
        h.w_scale = Ly.hc_ffn_scale;
        h.w_base = Ly.hc_ffn_base;
        h.norm_w = Ly.ffn_norm;
        for (int i = 0; i < n; ++i) {
            hc_row(i, rows(S->mixer, E, i));
            std::swap(rw[i].Rc, rw[i].Ro);
        }

        if (!Ly.moe) {
            gf::MvJob j[2];
            j[0] = {Ly.ffn_gate.q, S->xq, S->x, S->dg, nullptr, 1.0f, Ly.ffn_gate.type, E, g.n_ff_dense};
            j[1] = {Ly.ffn_up.q, S->xq, S->x, S->du, nullptr, 1.0f, Ly.ffn_up.type, E, g.n_ff_dense};
            if (!mv(j, 2, "dense ffn")) return false;
            for (int i = 0; i < n; ++i)
                gf::swiglu_q8(rows(S->dg, g.n_ff_dense, i), rows(S->du, g.n_ff_dense, i), g.swiglu_shexp,
                              g.n_ff_dense, rows_q(S->dhq, g.n_ff_dense, i), s);
            gf::MvJob dn = {Ly.ffn_down.q, S->dhq, nullptr, S->ffn, nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense,
                            E};
            if (!mv(&dn, 1, "dense ffn down")) return false;
        } else {
            // the router and the shared expert's gate / up for every row, then each row's routed experts (and the
            // shared expert's down, which rides their launch) through the token path's own code
            gf::MvJob j[3];
            j[0] = {Ly.router, nullptr, S->x, S->rlog, nullptr, 1.0f, gf::kTypeBF16, E, g.n_expert};
            j[1] = {Ly.sh_gate.q, S->xq, S->x, S->sh_g, nullptr, 1.0f, Ly.sh_gate.type, E, FFs};
            j[2] = {Ly.sh_up.q, S->xq, S->x, S->sh_u, nullptr, 1.0f, Ly.sh_up.type, E, FFs};
            if (!mv(j, 3, "router / shared expert")) return false;
            for (int i = 0; i < n; ++i) {
                const MoeIo io = {rows(S->x, E, i), rows_q(S->xq, E, i), rows(S->rlog, g.n_expert, i),
                                  rows(S->sh_g, FFs, i), rows(S->sh_u, FFs, i), rows(S->ffn, E, i)};
                if (!fast_moe_experts(il, io, false, 0, no_bias, pf_pending, err)) return false;
            }
        }
    }
    // the last layer's write half, per row; the residual back at the arena's start (the token path's convention)
    for (int i = 0; i < n; ++i) {
        gf::hc_post(rows(S->ffn, E, i), rw[i].Rc, rows(S->post, 8, i), rows(S->comb, 16, i), E, rw[i].Ro, s);
        if (rw[i].Ro != rw[i].state)
            cudaMemcpyAsync(rw[i].state, rw[i].Ro, (size_t) g.hc * E * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
    return true;
}

bool Glm5Model::rows_head(const RowSeq* rw, const strata::kernels::SamplerParams* sps, int n, std::string& err) {
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    cudaSetDevice(dev_);
    cudaStream_t s = F->cs;
    auto& rows = row_at;
    auto& rows_q = rowq_at;
    const auto mv = [&](gf::MvJob* j, int nj, const char* what) -> bool {
        if (gf::mv_nt(j, nj, n, s)) return true;
        err = std::string("glm rows: ") + what;
        return false;
    };
    // ---- the head: one read of the output matrix for every row, then each row's draw
    for (int i = 0; i < n; ++i)
        gf::head_prep(rw[i].state, w_.at("output_norm.weight"), g.norm_eps, E, rows(S->head_x, E, i),
                      rows_q(S->head_xq, E, i), s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, S->head_xq, S->head_x, S->logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!mv(&o, 1, "output head")) return false;
    for (int i = 0; i < n; ++i) {
        const float* lg = rows(S->logits, g.n_vocab, i);
        if (sps[i].greedy || sps[i].temperature <= 0.0f)
            gf::argmax(lg, g.n_vocab, S->tok + i, s);
        else
            strata::kernels::sample_tokens(lg, 1, (int) g.n_vocab, nullptr, 0, sps[i], S->tok + i, s);
    }
    return true;
}

bool Glm5Model::multi_rows(const int* seqs, const int32_t* tokens, const strata::kernels::SamplerParams* sps, int n,
                           int* tokens_out, std::string& err) {
    if (!rows_ok() || n_parts_ != 1 || seq_.empty() || n < 1 || n > rows_max()) {
        err = "glm rows: not available here (one GPU, several sequences, 1.." + std::to_string(rows_max()) + " rows)";
        return false;
    }
    if (seq_[0].hop_h == nullptr && !seq_pipe_ready(err)) return false;   // (each sequence's pinned embedding row)
    cudaSetDevice(dev_);
    // between steps (the device idle): finished promotions go live and the layers get fresh spare slots, as the
    // token path does between tokens - every 32 steps instead, the spares ran dry and the VRAM tier stopped following
    // the routes (two conversations on a V100 streaming its experts from RAM: 27.1 tok/s in all against 34.2)
    if (!fast_boundary(err)) return false;
    FastState* F = fast_;
    RowsState* S = rows_;
    // (the token path's profiler collects one token's marks at a time: none from a batched step)
    struct ProfOff {
        FastState* F;
        bool was;
        ~ProfOff() { F->prof_on = was; }
    } prof_off{F, F->prof_on};
    F->prof_on = false;
    RowSeq rw[gf::kMaxMvNt];
    if (!rows_bind(seqs, n, rw, err) || !rows_embed(rw, tokens, n, err) || !rows_layers(rw, n, err) ||
        !rows_head(rw, sps, n, err))
        return false;
    cudaMemcpyAsync(S->tok_h, S->tok, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost, F->cs);
    cudaEventRecord(S->ev_done, F->cs);
    // the step back (the service threads answer the routes' misses meanwhile)
    if (!glmfast::wait_event(S->ev_done, "glm rows step", err) || !F->route_ok(err)) return false;
    if (gf::launch_errors() > 0) {
        err = "glm rows: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm rows: ") + cudaGetErrorString(e);
        return false;
    }
    for (int i = 0; i < n; ++i) {
        const int tok = S->tok_h[i];
        tokens_out[i] = tok;
        if (rw[i].seq == cur_seq_) {
            ++pos_;
            last_tok_ = tok;
        } else {
            SeqCtx& c = seq_[(size_t) rw[i].seq];
            ++c.pos;
            c.last_tok = tok;
        }
    }
    return true;
}

bool Glm5Model::split_rows_ok() const {
    if (n_parts_ <= 1 || seq_.empty()) return false;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ == nullptr || m->rows_ == nullptr) return false;
    return true;
}

bool Glm5Model::multi_issue_rows(const int* seqs, const int32_t* tokens,
                                 const strata::kernels::SamplerParams* sps, int n, std::string& err) {
    if (!split_rows_ok() || n < 1 || n > rows_max()) {
        err = "glm rows: the split batch is not available here (every part's row arena, 1.." +
              std::to_string(rows_max()) + " rows)";
        return false;
    }
    if (seq_[0].hop_h == nullptr && !seq_pipe_ready(err)) return false;

    struct ProfOff {
        FastState* fast[kMaxParts];
        bool was[kMaxParts];
        int n = 0;
        ~ProfOff() {
            for (int i = n; i-- > 0;) fast[i]->prof_on = was[i];
        }
    } prof_off;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        prof_off.fast[prof_off.n] = m->fast_;
        prof_off.was[prof_off.n] = m->fast_->prof_on;
        m->fast_->prof_on = false;
        ++prof_off.n;
    }

    RowSeq cur[gf::kMaxMvNt];
    if (!rows_bind(seqs, n, cur, err) || !rows_embed(cur, tokens, n, err) || !rows_layers(cur, n, err))
        return false;
    int64_t p0[8];
    for (int i = 0; i < n; ++i) p0[i] = cur[i].p;

    Glm5Model* tail = this;
    const size_t hop = (size_t) g_.hc * g_.n_embd * sizeof(float);
    for (Glm5Model* B = split_next_.get(); B != nullptr; B = B->split_next_.get()) {
        RowSeq nxt[8];
        if (!B->rows_bind(seqs, n, nxt, err)) return false;
        for (int i = 0; i < n; ++i)
            if (nxt[i].p != p0[i]) {
                err = "glm rows: sequence " + std::to_string(seqs[i]) + " is at a different position on CUDA" +
                      std::to_string(B->dev_);
                return false;
            }
        cudaSetDevice(tail->dev_);
        for (int i = 0; i < n; ++i) {
            SeqCtx& ca = tail->seq_[(size_t) seqs[i]];
            cudaMemcpyAsync(ca.hop_h, cur[i].state, hop, cudaMemcpyDeviceToHost, tail->fast_->cs);
            cudaEventRecord(ca.ev_hop, tail->fast_->cs);
        }
        cudaSetDevice(B->dev_);
        for (int i = 0; i < n; ++i) {
            SeqCtx& ca = tail->seq_[(size_t) seqs[i]];
            cudaStreamWaitEvent(B->fast_->cs, ca.ev_hop, 0);
            cudaMemcpyAsync(nxt[i].state, ca.hop_h, hop, cudaMemcpyHostToDevice, B->fast_->cs);
        }
        if (!B->rows_layers(nxt, n, err)) return false;
        for (int i = 0; i < n; ++i) cur[i] = nxt[i];
        tail = B;
    }

    if (!tail->rows_head(cur, sps, n, err)) return false;
    cudaSetDevice(tail->dev_);
    for (int i = 0; i < n; ++i) {
        SeqCtx& ct = tail->seq_[(size_t) seqs[i]];
        cudaMemcpyAsync(ct.tok_h, tail->rows_->tok + i, sizeof(int), cudaMemcpyDeviceToHost, tail->fast_->cs);
        cudaEventRecord(ct.ev_done, tail->fast_->cs);
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        for (int i = 0; i < n; ++i) {
            if (seqs[i] == m->cur_seq_)
                ++m->pos_;
            else
                ++m->seq_[(size_t) seqs[i]].pos;
        }
    cudaSetDevice(dev_);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm rows: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace strata::core
