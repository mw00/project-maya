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
// (greedy with the CPU lane off: the multi bench compares them).  STRATA_GLM_GROUP_EXPERTS=1 instead routes every row of
// the step first and runs each distinct routed expert once for all the rows that chose it (rows_moe_grouped, the verify
// window's batched MoE) - each row's output bit for bit the per-row kernels' for it.
//
// Each part's row activations live in its own arena (rows_alloc, at load, before the expert pool takes the rest of the
// card); the per-row scratch the token path's kernels use in between (the conv outputs, the gates, the DSA scores and
// cells, the experts' h) is the token path's own, used by one row after the other on that part's stream.
//
// STRATA_GLM_ROWS_BATCH=1: the residual mixing (mHC) and the KDA recurrence of all the rows in one launch each instead
// of one a row (gf::hc_rows, gf::kda_layers over the rows) - each row with its own scratch (rows_alloc) and the same
// device functions as the per-row kernels, so its arithmetic is meant to be theirs (glm_rows_kernels_test compares
// them bitwise).
//
// STRATA_GLM_ROWS_ATTN=1: a DSA layer's absorbed attention of all the rows in one launch instead of one a row: wk_b . q
// for every row at once (gf::mla_q_abs), each row's selection (and, streamed, its slot cells) into its own buffers, the
// split attention of all the rows over their own caches in one launch (gf::mla_attend_rows: a row's blocks on
// blockIdx.z, so its 64 blocks no longer leave half the card idle), then wv_b . ctx for every row at once
// (gf::mla_out) - the verify window's batched MLA, for rows of different conversations.  Each row's output is mla()'s
// bit for bit (glm_rows_attn_test compares them).
#include "glm_rows_state.hpp"

#include "strata/core/mtp_rows.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/sampler.hpp"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {
using glmrows::kQ8;
using glmrows::up256;
using glmrows::row_at;
using glmrows::rowq_at;
using glmrows::image_row_in;
bool rows_batch_on() {   // STRATA_GLM_ROWS_BATCH=1 (off by default)
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_ROWS_BATCH");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}
bool rows_small_on() {   // STRATA_GLM_ROWS_SMALL=1 (off by default)
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_ROWS_SMALL");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}
bool rows_attn_on() {   // STRATA_GLM_ROWS_ATTN=1 (off by default)
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_ROWS_ATTN");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}
}  // namespace

// 8 rows (one a conversation); with STRATA_GLM_MULTI_MTP a group's verify windows, multi_mtp_cap() rows
int Glm5Model::rows_max() { return multi_mtp_setting() > 0 ? multi_mtp_cap() : 8; }

int Glm5Model::split_rows_setting() {
    const char* v = getenv("STRATA_GLM_SPLIT_ROWS");
    return v != nullptr ? std::clamp((int) std::atof(v), 0, rows_max()) : 0;
}

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
    const auto take_all = [&](size_t bytes) -> uint8_t* {   // once, not a row
        const size_t off = at;
        at += up256(bytes);
        return base == nullptr ? nullptr : base + off;
    };
    cudaSetDevice(dev_);   // (mla_rows_ok asks the current device)
    const bool attn = rows_attn_on() && gf::mla_rows_ok(g.kv_lora);
    const int attn_tokens = (int) std::min<size_t>(R, (size_t) gf::kMlaTokMax);
    int64_t attn_ctx = max_ctx_;   // (the longest context a sequence has: its completed pools' scores)
    for (const SeqCtx& c : seq_) attn_ctx = std::max(attn_ctx, c.max_ctx);
    const int64_t attn_pools = attn_ctx / g.idx_kpool + 1;
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
        if (rows_batch_on()) {   // (STRATA_GLM_ROWS_BATCH's own per-row scratch: none with the setting off)
            for (auto& p : S->kconv) p = f32(DI);
            S->kg1 = f32(DI);
            S->kg2 = f32(DI);
            S->hc_part_n = (4 * E / 512) * 26 + 64;   // (as the token path's F->part)
            S->hc_part = f32(S->hc_part_n);
            S->hc_ctr = take(64);
        }
        if (attn) {   // (STRATA_GLM_ROWS_ATTN's buffers: none with the setting off)
            S->aq = f32((int64_t) g.n_head * g.kv_lora);
            S->actx = f32((int64_t) g.n_head * g.kv_lora);
            S->acells = (int*) take((size_t) g.n_sel_max() * sizeof(int));
            S->aslot = (int*) take((size_t) g.n_sel_max() * sizeof(int));
            S->apart = (float*) take_all(gf::mla_rows_part_floats(g.n_head, g.kv_lora) * attn_tokens * sizeof(float));
            S->apart_tokens = attn_tokens;
            S->ascore_n = attn_pools;
            S->ascore = f32(attn_pools);
        }
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
    if (attn)
        std::fprintf(stderr, "glm rows: CUDA%d the DSA attention of up to %d rows a launch (MAYA_GLM_ROWS_ATTN), %zu MB of "
                             "the rows' %zu MB\n", dev_, attn_tokens,
                     (size_t) ((2 * R * g.n_head * g.kv_lora * sizeof(float) + 2 * R * g.n_sel_max() * sizeof(int) +
                                gf::mla_rows_part_floats(g.n_head, g.kv_lora) * attn_tokens * sizeof(float) +
                                R * attn_pools * sizeof(float)) >> 20),
                     at >> 20);
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
        r.res = r.state;
        r.kin = nullptr;
        r.kin_t = 0;
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
        gf::embed_streams(F->emb, rw[i].res, E, s);
    }
    return true;
}

bool Glm5Model::rows_layers(RowSeq* rw, int n, std::string& err) {
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd, DI = g.d_inner();
    cudaSetDevice(dev_);
    cudaStream_t s = F->cs;
    auto& rows = row_at;
    auto& rows_q = rowq_at;
    const auto mv = [&](gf::MvJob* j, int nj, const char* what) -> bool {
        if (gf::mv_nt(j, nj, n, s)) return true;
        err = std::string("glm rows: ") + what;
        return false;
    };
    // a verify window's rows of one conversation advance its recurrent states one after the other: the per-row
    // kernels in row order then (rows_bind_win), never one launch for all of them
    bool dup = false, keep_in = false;
    for (int i = 0; i < n; ++i) {
        keep_in = keep_in || rw[i].kin != nullptr;
        for (int k = 0; k < i && !dup; ++k) dup = rw[k].seq == rw[i].seq;
    }
    const bool batch = rows_batch_on() && n > 1;
    // (kda_layers is built for head_dim 128, as the window commit's replay: else the KDA one row at a time)
    const bool kda_batch = batch && !dup && g.kda_head_dim == 128;
    gf::HcArgs hb[gf::kMaxMvNt];
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
            if (!batch) {
                gf::hc(h, s);
                return;
            }
            hb[i] = h;   // (launched for all the rows at once: hc_rows)
            hb[i].part = rows(S->hc_part, S->hc_part_n, i);
            hb[i].counter = (unsigned int*) (S->hc_ctr + (size_t) i * 64);   // (take(64): 64 bytes a row)
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
        if (batch) gf::hc_rows(hb, n, s);

        if (Ly.recr) {
            gf::MvJob j[6];
            j[0] = {Ly.q.q, S->xq, S->x, S->proj[0], nullptr, 1.0f, Ly.q.type, E, DI};
            j[1] = {Ly.k.q, S->xq, S->x, S->proj[1], nullptr, 1.0f, Ly.k.type, E, DI};
            j[2] = {Ly.v.q, S->xq, S->x, S->proj[2], nullptr, 1.0f, Ly.v.type, E, DI};
            j[3] = {Ly.f_a, nullptr, S->x, S->fa, nullptr, 1.0f, gf::kTypeBF16, E, g.kda_head_dim};
            j[4] = {Ly.g_a, nullptr, S->x, S->ga, nullptr, 1.0f, gf::kTypeBF16, E, g.kda_head_dim};
            j[5] = {Ly.beta, nullptr, S->x, S->beta, nullptr, 1.0f, gf::kTypeBF16, E, g.n_head};
            if (!mv(j, 6, "kda projections")) return false;
            if (keep_in) {   // (STRATA_GLM_MULTI_MTP: the window rows' inputs, for the commit's replay)
                gf::KdaGatherArgs ga;
                ga.src[0] = S->proj[0];
                ga.src[1] = S->proj[1];
                ga.src[2] = S->proj[2];
                ga.src[3] = S->fa;
                ga.src[4] = S->ga;
                ga.src[5] = S->beta;
                ga.n_rows = n;
                ga.tmax = mtpw_window();
                ga.di = DI;
                ga.hd = g.kda_head_dim;
                ga.nh = g.n_head;
                const size_t off = mtpw_kin_offset(il);
                for (int i = 0; i < n; ++i) {
                    ga.dst[i] = rw[i].kin != nullptr ? rw[i].kin + off : nullptr;
                    ga.t[i] = rw[i].kin_t;
                }
                gf::kda_gather(ga, s);
            }
            gf::KdaPrepArgs kp[gf::kMaxMvNt];
            gf::KdaRecArgs kr[gf::kMaxMvNt];
            for (int i = 0; i < n; ++i) {
                gf::KdaPrepArgs k;
                for (int c = 0; c < 3; ++c) {
                    k.proj[c] = rows(S->proj[c], DI, i);
                    k.conv_w[c] = Ly.conv[c];
                    k.out[c] = kda_batch ? rows(S->kconv[c], DI, i) : F->conv[c];
                }
                k.conv_state = rw[i].state + (*rw[i].kda_conv)[(size_t) il];
                k.fa = rows(S->fa, g.kda_head_dim, i);
                k.ga = rows(S->ga, g.kda_head_dim, i);
                k.f_b = Ly.f_b;
                k.g_b = Ly.g_b;
                k.dt_bias = Ly.dt_bias;
                k.ssm_a = Ly.ssm_a;
                k.lower_bound = g.kda_lb;
                k.g1 = kda_batch ? rows(S->kg1, DI, i) : F->g1;
                k.g2 = kda_batch ? rows(S->kg2, DI, i) : F->g2;
                k.n_head = g.n_head;
                k.head_dim = g.kda_head_dim;
                k.d_conv = g.d_conv;
                if (kda_batch) {   // (both kernels for all the rows below: kda_layers)
                    kp[i] = k;
                    kr[i].q = k.out[0];
                    kr[i].k = k.out[1];
                    kr[i].v = k.out[2];
                    kr[i].g1 = k.g1;
                    kr[i].beta_raw = rows(S->beta, g.n_head, i);
                    kr[i].state = rw[i].state + (*rw[i].kda_S)[(size_t) il];
                    kr[i].g2 = k.g2;
                    kr[i].norm_w = Ly.ssm_norm;
                    kr[i].out = rows_q(S->gated_q, DI, i);
                    continue;
                }
                gf::kda_prep(k, s);
                gf::kda_rec(F->conv[0], F->conv[1], F->conv[2], F->g1, rows(S->beta, g.n_head, i),
                            rw[i].state + (*rw[i].kda_S)[(size_t) il], F->g2, Ly.ssm_norm, g.norm_eps, g.n_head,
                            g.kda_head_dim, rows_q(S->gated_q, DI, i), s);
            }
            for (int i0 = 0; kda_batch && i0 < n; i0 += gf::kKdaLayers)
                gf::kda_layers(kp + i0, kr + i0, std::min(gf::kKdaLayers, n - i0), g.norm_eps, s);
            gf::MvJob o = {Ly.out.q, S->gated_q, nullptr, S->mixer, nullptr, 1.0f, Ly.out.type, DI, E};
            if (!mv(&o, 1, "kda output")) return false;
        } else {
            if (!rows_dsa(il, rw, n, err)) return false;
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
        if (batch) gf::hc_rows(hb, n, s);

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
            if (!rows_moe(il, rw, n, err)) return false;
        }
    }
    // the last layer's write half, per row; the residual back at the arena's start (the token path's convention)
    for (int i = 0; i < n; ++i) {
        gf::hc_post(rows(S->ffn, E, i), rw[i].Rc, rows(S->post, 8, i), rows(S->comb, 16, i), E, rw[i].Ro, s);
        if (rw[i].Ro != rw[i].res)
            cudaMemcpyAsync(rw[i].res, rw[i].Ro, (size_t) g.hc * E * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
    return true;
}

// a DSA layer's mixer for the rows (S->x / S->xq in, S->mixer out): the projections for every row at once, each row's
// caches, selection and attention through the token path's kernels against its own sequence (rw[i])
bool Glm5Model::rows_dsa(int il, RowSeq* rw, int n, std::string& err) {
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared;
    (void) FFs;
    cudaStream_t s = F->cs;
    auto& rows = row_at;
    auto& rows_q = rowq_at;
    const auto mv = [&](gf::MvJob* j, int nj, const char* what) -> bool {
        if (gf::mv_nt(j, nj, n, s)) return true;
        err = std::string("glm rows: ") + what;
        return false;
    };
    const auto& Ly = F->L[(size_t) il];
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
    // STRATA_GLM_ROWS_ATTN: the attention of every row in one launch (each row's selection into its own buffers).  A
    // verify window's rows of one conversation (consecutive rows) on a streamed layer share its slot pool: one resolve
    // over all their selections then (a resolve never evicts a block it names), not one a row - a row's resolve could
    // evict a block the row before it still has to read.  Rows of one conversation that are not consecutive: one by one
    const int64_t qd = (int64_t) g.n_head * g.qk_nope, ad = (int64_t) g.n_head * g.kv_lora, nsm = g.n_sel_max();
    bool attn = rows_attn_on() && n > 1 && S->apart != nullptr && g.v_head % 32 == 0 && gf::mla_rows_ok(g.kv_lora);
    int run0[gf::kMaxMvNt];   // the first row of each row's run of consecutive rows of its conversation
    for (int i = 0; i < n; ++i) {
        run0[i] = i > 0 && rw[i - 1].seq == rw[i].seq ? run0[i - 1] : i;
        for (int k = 0; k < run0[i] && attn; ++k)
            if (rw[k].seq == rw[i].seq) attn = false;
    }
    if (attn)   // (a run's selections are resolved as one list: the cells past a row's own count must read as padding)
        for (int i = 0; i < n; ++i)
            if (run0[i] == i && i + 1 < n && run0[i + 1] == i && kv_streamed(rw[i].kv, il)) {
                int len = 1;
                while (i + len < n && run0[i + len] == i) ++len;
                cudaMemsetAsync(S->acells + (size_t) i * nsm, 0xff, (size_t) len * nsm * sizeof(int), s);
            }
    const uint16_t* a_lat[gf::kMaxMvNt];
    const int* a_cells[gf::kMaxMvNt];
    int a_nsel[gf::kMaxMvNt];
    if (attn) gf::mla_q_abs(S->q, qd, Ly.k_b, g.n_head, g.qk_nope, g.kv_lora, n, S->aq, ad, s);
    gf::DsaSelectRows sr;   // (with the rows' attention: the rows' selections in one launch, then each one resolved)
    int n_sr = 0;
    const bool small = attn && rows_small_on();   // (STRATA_GLM_ROWS_SMALL: the rows' scores in one launch too)
    gf::DsaScoreRows sc;
    int n_sc = 0;
    for (int i = 0; i < n; ++i) {
        const int64_t p = rw[i].p;
        const int pool_done = (int) ((p + 1) / g.idx_kpool);
        const float* pooled = rw[i].state + (*rw[i].dsa_pool)[(size_t) il];
        const bool all = pool_done <= g.top_pools_max() && gf::dsa_select_all_on();   // (as fast_dsa)
        float* const score = attn ? S->ascore + (size_t) i * S->ascore_n : F->score;
        if (pool_done > 0 && !all && small) {
            sc.iq[n_sc] = rows(S->iq, (int64_t) g.idx_heads * g.idx_key, i);
            sc.pooled[n_sc] = pooled;
            sc.iw[n_sc] = rows(S->iw, g.idx_heads, i);
            sc.score[n_sc] = score;
            sc.n_pools[n_sc++] = pool_done;
        } else if (pool_done > 0 && !all) {
            gf::dsa_score(rows(S->iq, (int64_t) g.idx_heads * g.idx_key, i), pooled,
                          rows(S->iw, g.idx_heads, i), g.idx_key, g.idx_heads, pool_done, score, s);
        }
        const int top_pools = std::min(g.top_pools_max(), pool_done);
        const int n_sel = g.idx_kpool * top_pools + (g.idx_select_tail ? g.idx_kpool - 1 : 0);
        int* const sel = attn ? S->acells + (size_t) i * nsm : F->cells;
        uint16_t* const lat = (uint16_t*) (rw[i].state + (*rw[i].dsa_lat)[(size_t) il]);
        if (attn) {
            if (all) {
                gf::dsa_select_all(pool_done, g.idx_kpool, n_sel, (int) p, sel, s);
            } else {
                sr.score[n_sr] = score;
                sr.cells[n_sr] = sel;
                sr.n_vis[n_sr] = pool_done;
                sr.top_pools[n_sr] = top_pools;
                sr.n_sel[n_sr] = n_sel;
                sr.pos[n_sr] = (int) p;
                ++n_sr;
            }
            a_lat[i] = lat;
            a_nsel[i] = n_sel;
            continue;
        }
        if (all) gf::dsa_select_all(pool_done, g.idx_kpool, n_sel, (int) p, sel, s);
        else gf::dsa_select(F->score, pool_done, g.idx_kpool, top_pools, n_sel, (int) p, sel, s);
        const int* cells = kv_attend_cells(rw[i].kv, il, lat, F->cells, n_sel, F->slot_cells, s);
        gf::mla(rows(S->q, qd, i), Ly.k_b, Ly.v_b, lat, cells, n_sel, g.n_head,
                g.qk_nope, g.kv_lora, g.v_head, rows_q(S->attn_q, (int64_t) g.n_head * g.v_head, i), s,
                lat_q8_);
    }
    if (attn) {
        if (n_sc > 0) gf::dsa_score_rows(sc, n_sc, g.idx_key, g.idx_heads, s);
        gf::dsa_select_rows(sr, n_sr, g.idx_kpool, s);
        // KV streaming: each run's blocks resident, its rows' selections in slot rows (resident: the cells themselves);
        // STRATA_GLM_ROWS_SMALL: the runs' resolves in one launch (each run its own sequence's map)
        gf::KvResolveRows kr;
        int n_kr = 0, kr_row = 0, kr_page = 0;
        for (int i = 0; i < n;) {
            int len = 1;
            while (i + len < n && run0[i + len] == i) ++len;
            const int cnt = (int) ((len - 1) * nsm) + a_nsel[i + len - 1];
            const KvStream* K = rw[i].kv;
            const bool batch_it = small && kv_streamed(K, il) && n_kr < gf::kKvResolveRowsMax &&
                                  (n_kr == 0 || (K->row_bytes == kr_row && K->page == kr_page));
            if (batch_it) {
                kr.m[n_kr] = K->map[(size_t) il];
                kr.host[n_kr] = K->host[(size_t) il];
                kr.slots[n_kr] = (uint8_t*) a_lat[i];
                kr.cells[n_kr] = S->acells + (size_t) i * nsm;
                kr.slot_cells[n_kr] = S->aslot + (size_t) i * nsm;
                kr.n_sel[n_kr++] = cnt;
                kr_row = K->row_bytes;
                kr_page = K->page;
                for (int k = 0; k < len; ++k) a_cells[i + k] = S->aslot + (size_t) (i + k) * nsm;
            } else {
                const int* c = kv_attend_cells(K, il, const_cast<uint16_t*>(a_lat[i]), S->acells + (size_t) i * nsm, cnt,
                                               S->aslot + (size_t) i * nsm, s);
                for (int k = 0; k < len; ++k) a_cells[i + k] = c + (size_t) k * nsm;
            }
            i += len;
        }
        if (n_kr > 0) gf::glm_kv_resolve_rows(kr, n_kr, kr_row, kr_page, s);
        if (!gf::mla_attend_rows(S->aq, ad, a_lat, a_cells, a_nsel, n, g.n_head, g.qk_nope, g.kv_lora, S->actx, ad,
                                 S->apart, S->apart_tokens, s, lat_q8_)) {
            err = "glm rows: the rows' attention is not available here";   // (mla_rows_ok said it was)
            return false;
        }
        gf::mla_out(S->actx, ad, Ly.v_b, g.n_head, g.kv_lora, g.v_head, n, S->attn_q,
                    (int64_t) ((int64_t) g.n_head * g.v_head / 32) * kQ8, s);
    }
    gf::MvJob o = {Ly.out.q, S->attn_q, nullptr, S->mixer, nullptr, 1.0f, Ly.out.type, g.n_head * g.v_head,
                   E};
    if (!mv(&o, 1, "dsa output")) return false;
    return true;
}

// a MoE layer's FFN for the rows (S->x / S->xq in, S->ffn out)
bool Glm5Model::rows_moe(int il, RowSeq* rw, int n, std::string& err) {
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared;
    (void) FFs;
    cudaStream_t s = F->cs;
    auto& rows = row_at;
    auto& rows_q = rowq_at;
    const auto mv = [&](gf::MvJob* j, int nj, const char* what) -> bool {
        if (gf::mv_nt(j, nj, n, s)) return true;
        err = std::string("glm rows: ") + what;
        return false;
    };
    const auto& Ly = F->L[(size_t) il];
    (void) rw;
    const float* no_bias[gf::kAhead] = {nullptr, nullptr, nullptr, nullptr};
    bool pf_pending = false;   // (no prefetch in a batched step)
    // the router and the shared expert's gate / up for every row, then each row's routed experts (and the
    // shared expert's down, which rides their launch) through the token path's own code
    gf::MvJob j[3];
    j[0] = {Ly.router, nullptr, S->x, S->rlog, nullptr, 1.0f, gf::kTypeBF16, E, g.n_expert};
    j[1] = {Ly.sh_gate.q, S->xq, S->x, S->sh_g, nullptr, 1.0f, Ly.sh_gate.type, E, FFs};
    j[2] = {Ly.sh_up.q, S->xq, S->x, S->sh_u, nullptr, 1.0f, Ly.sh_up.type, E, FFs};
    if (!mv(j, 3, "router / shared expert")) return false;
    // its own error string: `err` is the scheduler's, and a cancelled or failed prompt leaves its text there
    // until the next prompt - read here, it failed every running request (with the setting off too)
    std::string gerr;
    const bool grouped = glmfast::group_experts_on() && rows_moe_grouped(il, n, gerr);
    if (!gerr.empty()) { err = gerr; return false; }
    for (int i = 0; !grouped && i < n; ++i) {
        const MoeIo io = {rows(S->x, E, i), rows_q(S->xq, E, i), rows(S->rlog, g.n_expert, i),
                          rows(S->sh_g, FFs, i), rows(S->sh_u, FFs, i), rows(S->ffn, E, i)};
        if (!fast_moe_experts(il, io, false, 0, no_bias, pf_pending, err, true)) return false;
    }
    return true;
}

// STRATA_GLM_GROUP_EXPERTS=1: the n rows' routed experts as the verify window's batched MoE takes a window's tokens
// (fast_moe_window).  Every row is routed first - into its own plan, fetch sources, disk answer, CPU-lane answer slot
// (the window's route state, its rows the window's tokens) and its own 8 scratch slots (rows_scr), each row's wait and
// fetch before the next row's route (the host's answer to a disk miss is one buffer) - then each distinct expert runs
// once for all the rows that routed it (gf::moe_win_experts, exl3::win_experts: each row's output is moe_gate_up +
// moe_down's for it alone, bit for bit), then the rows' CPU-lane answers in one launch.  An expert two rows route is
// read once instead of twice, and fast_service holds the group's routes and answers their CPU-lane experts as one
// merged request (win_cpu_merged on this part's pool: each RAM-tier expert read once for the group).  The shared
// expert's down rows of all n rows are one mv_nt (mv's bits for each row - the per-row path's fused down's too).
// Its route is the same per row: the spares a row's fetch takes are resident for the rows after it, as they are for
// the next row's route on the per-row path.  False with err empty: not here (the rows run one by one).
bool Glm5Model::rows_moe_grouped(int il, int n, std::string& err) {
    namespace x3 = strata::kernels::exl3;
    FastState* F = fast_;
    RowsState* S = rows_;
    auto& W = F->win;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared, K = g.n_exp_used;
    if (n < 2) return false;
    const bool wide = n > F->rows_group && n <= F->rows_scr_rows && F->all_in_vram;   // (STRATA_GLM_MULTI_MTP)
    if ((n > F->rows_group && !wide) || F->rows_scr == nullptr || W.mplan_ptr == nullptr || W.tmax < n ||
        (Ly.exl3 && (W.wh == nullptr || !x3::several_ok()))) {
        static std::atomic<bool> told{false};   // (a run with the setting on that never groups should say so; both
        if (!told.exchange(true)) {            //  parts' threads get here)
            std::fprintf(stderr, "glm rows: CUDA%d a group of %d rows runs its experts row by row (grouping set up for "
                                 "%d rows%s) - said once\n", dev_, n, F->rows_group,
                         Ly.exl3 && W.wh == nullptr ? ", no EXL3 rows" : "");
        }
        return false;
    }
    cudaStream_t s = F->cs;
    const bool lane = (F->cpu_plan != 0ull || F->helper.device() >= 0) && F->cpu_fmt[(size_t) il].n_ff > 0;
    const bool skip_empty = glmfast::skip_empty_on();
    const bool empty = F->all_resident || (skip_empty && F->layer_resident(il));
    gf::MoeDev md = F->md;
    md.pf_src = F->pf_buf[il & 1];
    md.pf_dst = F->pf_buf[il & 1] + 8;
    md.pf_n = F->pf_n_buf[il & 1];
    const auto row_dev = [&](int i) {   // (fast_moe_window's tok_dev, and the row's own scratch slots)
        gf::MoeDev m = md;
        m.plan_ptr = W.mplan_ptr + 8 * i;
        m.plan_w = W.mplan_w + 8 * i;
        m.plan_id = W.mplan_id + 8 * i;
        m.fetch_src = W.mfetch + 8 * i;
        m.cpu_part = W.mcpu_part + (size_t) E * i;
        m.cpu_flag = W.mcpu_flag + i;
        m.scratch = F->rows_scr + 8 * i;
        if (i > 0 && md.cpu_seq != nullptr) {
            m.cpu_seq = W.cpu_seq_w + 16 * i;
            m.cpu_ans = W.ans_d[i];
        }
        return m;
    };
    const float* no_bias[gf::kAhead] = {nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < n; ++i) {
        const gf::MoeDev m = row_dev(i);
        const MoeIo io = {row_at(S->x, E, i), rowq_at(S->xq, E, i), row_at(S->rlog, g.n_expert, i),
                          row_at(S->sh_g, FFs, i), row_at(S->sh_u, FFs, i), row_at(S->ffn, E, i)};
        fast_moe_route(il, io, m, (uint8_t*) W.sh_hq + (size_t) i * W.sh_hq_s, false, 0, no_bias, lane, i, n, 1);
        ++F->expected;
        if (empty) continue;
        if (skip_empty) {
            gf::moe_wait_fetch(m, E, K, Ly.blob, s);
        } else {
            gf::moe_wait(m, E, s);
            gf::moe_fetch(m, K, Ly.blob, s);
        }
    }
    const float* sh_out = nullptr;
    if (Ly.sh_down.q != nullptr) {
        gf::MvJob sd = {Ly.sh_down.q, W.sh_hq, nullptr, W.sh_out, nullptr, 1.0f, Ly.sh_down.type, FFs, E};
        if (!gf::mv_nt(&sd, 1, n, s)) {
            err = "glm rows: the grouped shared expert's down";
            return false;
        }
        sh_out = W.sh_out;
    }
    gf::MoeWin mw;
    mw.plan_ptr = W.mplan_ptr;
    mw.plan_w = W.mplan_w;
    mw.cpu_part = W.mcpu_part;
    mw.cpu_flag = W.mcpu_flag;
    mw.grp_ptr = W.grp_ptr;
    mw.grp_start = W.grp_start;
    mw.grp_ent = W.grp_ent;
    mw.n_groups = W.n_groups;
    mw.T = n;
    mw.k = K;
    mw.n_embd = E;
    mw.n_ff = g.n_ff_exp;
    mw.limit = g.swiglu_exp;
    if (Ly.exl3) {
        gf::moe_win_group(mw, s, rows_small_on());
        x3::WinArgs xa;
        xa.grp_ptr = W.grp_ptr;
        xa.grp_start = W.grp_start;
        xa.grp_ent = W.grp_ent;
        xa.n_groups = W.n_groups;
        xa.plan_ptr = W.mplan_ptr;
        xa.plan_w = W.mplan_w;
        xa.cpu_part = W.mcpu_part;
        xa.cpu_flag = W.mcpu_flag;
        xa.x = S->x;
        xa.x_stride = (size_t) E;
        xa.h = W.wh;
        xa.part = W.wpart;
        xa.sh_out = sh_out;
        xa.out = S->ffn;
        xa.out_stride = (size_t) E;
        xa.lay = Ly.xl;
        xa.T = n;
        xa.k = K;
        xa.n_embd = E;
        xa.n_ff = g.n_ff_exp;
        xa.limit = g.swiglu_exp;
        if (!x3::win_experts(xa, s)) {
            err = "glm rows: the grouped EXL3 experts";
            return false;
        }
    } else {
        mw.xq = S->xq;
        mw.xq_stride = (size_t) (E / 32) * kQ8;
        mw.hq = W.whq;
        mw.part = W.wpart;
        mw.sh_out = sh_out;
        mw.out = S->ffn;
        mw.out_stride = (size_t) E;
        mw.down_off = Ly.down_off;
        if (!gf::moe_win_experts(Ly.gu_type, Ly.d_type, mw, s, rows_small_on())) {
            err = "glm rows: the grouped experts (types " + std::to_string(Ly.gu_type) + " / " +
                  std::to_string(Ly.d_type) + ")";
            return false;
        }
    }
    // the CPU lane's answers last: the device's experts ran while the host computed the group's merged request
    if (lane && !empty && md.cpu_seq != nullptr) {
        if (n > gf::kMaxMvNt) {   // (rows_group keeps n at 8 or fewer; never leave the rows' CPU answers out quietly)
            err = "glm rows: " + std::to_string(n) + " rows are more than the CPU-lane wait takes";
            return false;
        }
        gf::CpuWaitRows cw;
        for (int i = 0; i < n; ++i) {
            const gf::MoeDev m = row_dev(i);
            cw.seq[i] = m.cpu_seq;
            cw.ans[i] = m.cpu_ans;
            cw.out[i] = row_at(S->ffn, E, i);
        }
        gf::moe_cpu_wait_rows(cw, n, E, s);
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
        gf::head_prep(rw[i].res, w_.at("output_norm.weight"), g.norm_eps, E, rows(S->head_x, E, i),
                      rows_q(S->head_xq, E, i), s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, S->head_xq, S->head_x, S->logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!mv(&o, 1, "output head")) return false;
    for (int i = 0; i < n; ++i) {
        const float* lg = rows(S->logits, g.n_vocab, i);
        if (strata::core::rows_pick_argmax(sps[i]))
            gf::argmax(lg, g.n_vocab, S->tok + i, s);
        else
            pick_tokens(lg, 1, sps[i], nullptr, 0, S->tok + i, s);
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
    int64_t p0[gf::kMaxMvNt];
    for (int i = 0; i < n; ++i) p0[i] = cur[i].p;

    Glm5Model* tail = this;
    const size_t hop = (size_t) g_.hc * g_.n_embd * sizeof(float);
    for (Glm5Model* B = split_next_.get(); B != nullptr; B = B->split_next_.get()) {
        RowSeq nxt[gf::kMaxMvNt];
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
