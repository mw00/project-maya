// src/core/glm_multi.cu - several conversations at once (STRATA_GLM_SEQS=<n>); see glm_model.hpp (SeqCtx).
//
// Every per-sequence thing a part keeps lives in its state arena (the residual, the KDA states and conv histories, the
// DSA latent / indexer caches) plus a handful of members (the position, the snapshot, the image rows).  A part holds n
// arenas and binds one: the token path, the prompt path, snapshots and slots all read `state_`, so they work on the
// bound sequence unchanged.  Activations and the tiers' machinery stay shared: a part runs one token at a time on its
// stream, whichever sequence it belongs to.
//
// THE PIPELINED DECODE.  multi_issue enqueues one token of one sequence through every part (each part's layers on its
// stream, the residual crossing over through that sequence's pinned hop slot, ordered by an event) and its head and
// sampler on the tail, and returns at once.  With several sequences in flight, the cards of a layer split work on
// different sequences at the same time: the first card on sequence C while the last one is on A.  The host only waits
// for the oldest sequence's token (multi_take), and issues that sequence's next one right away.
#include "glm_fast_state.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

void Glm5Model::seq_bind(int s) {
    if (seq_.empty() || s == cur_seq_ || s < 0 || s >= (int) seq_.size()) return;
    SeqCtx& o = seq_[(size_t) cur_seq_];
    o.state = state_;
    o.pos = pos_;
    o.snap = snap_;
    o.snap_pool = snap_pool_;
    o.snap_pos = snap_pos_;
    o.kda_bak = kda_bak_;
    o.last_tok = last_tok_;
    o.img_pos = std::move(img_pos_);
    o.img_rows = std::move(img_rows_);
    o.max_ctx = max_ctx_;
    o.bytes = state_bytes_;
    o.ik_ring = ik_ring_;
    std::swap(o.kda_S, kda_S_);
    std::swap(o.kda_conv, kda_conv_);
    std::swap(o.dsa_lat, dsa_lat_);
    std::swap(o.dsa_ik, dsa_ik_);
    std::swap(o.dsa_ig, dsa_ig_);
    std::swap(o.dsa_pool, dsa_pool_);
    SeqCtx& n = seq_[(size_t) s];
    state_ = n.state;
    pos_ = n.pos;
    snap_ = n.snap;
    snap_pool_ = n.snap_pool;
    snap_pos_ = n.snap_pos;
    kda_bak_ = n.kda_bak;
    last_tok_ = n.last_tok;
    img_pos_ = std::move(n.img_pos);
    img_rows_ = std::move(n.img_rows);
    n.img_pos.clear();
    n.img_rows.clear();
    max_ctx_ = n.max_ctx;
    state_bytes_ = n.bytes;
    ik_ring_ = n.ik_ring;
    std::swap(n.kda_S, kda_S_);
    std::swap(n.kda_conv, kda_conv_);
    std::swap(n.dsa_lat, dsa_lat_);
    std::swap(n.dsa_ik, dsa_ik_);
    std::swap(n.dsa_ig, dsa_ig_);
    std::swap(n.dsa_pool, dsa_pool_);
    cur_seq_ = s;
}

void Glm5Model::seq_bind_all(int s) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->seq_bind(s);
}

// load_pack's state layout (fast path) for a context of ctx tokens: the residual pair, then the NextN block's DSA
// caches, then this part's layers - the offsets into c's vectors, the bytes returned
uint64_t Glm5Model::seq_layout(int64_t ctx, SeqCtx& c) const {
    const Glm5Geometry& g = g_;
    const int64_t hc_dim = (int64_t) g.hc * g.n_embd;
    const int64_t max_pools = ctx / g.idx_kpool;
    int64_t floats = 2 * hc_dim;
    const int64_t lat_floats = lat_q8_ ? (int64_t) gf::lat8_rec_bytes(g.kv_lora) / 4 * ctx
                                       : (int64_t) g.kv_lora * ctx / 2;
    for (auto* v : {&c.kda_S, &c.kda_conv, &c.dsa_lat, &c.dsa_ik, &c.dsa_ig, &c.dsa_pool})
        v->assign((size_t) g.n_layers + 1, 0);
    c.ik_ring = (int) std::min<int64_t>(ctx, 8192 + 64);
    const auto dsa = [&](int il) {
        c.dsa_lat[(size_t) il] = floats;
        floats += lat_floats;
        c.dsa_ik[(size_t) il] = floats;
        floats += (int64_t) g.idx_key * c.ik_ring;
        c.dsa_ig[(size_t) il] = floats;
        floats += (int64_t) g.idx_key * c.ik_ring;
        c.dsa_pool[(size_t) il] = floats;
        floats += (int64_t) g.idx_key * max_pools;
    };
    if (mtp_il_ >= 0) dsa(mtp_il_);
    for (int il = l0_; il < l1_; ++il) {
        if (g.is_recr(il)) {
            c.kda_S[(size_t) il] = floats;
            floats += (int64_t) g.d_inner() * g.kda_head_dim;
            c.kda_conv[(size_t) il] = floats;
            floats += (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
        } else {
            dsa(il);
        }
    }
    c.max_ctx = ctx;
    c.bytes = (uint64_t) floats * sizeof(float);
    return c.bytes;
}

// STRATA_GLM_SEQS=<n> (1..8): n - 1 more state arenas on this part, zeroed - each for STRATA_GLM_SEQ_CTX tokens
// (default: the engine's context; at least 8320, at most the context).  Called by load_pack right after the first
// arena, so the expert pool is sized from what is left.
bool Glm5Model::seq_alloc_arenas(std::string& err) {
    const char* v = getenv("STRATA_GLM_SEQS");
    const int n = v ? std::max(1, std::min(8, std::atoi(v))) : 1;
    if (n <= 1) return true;
    int64_t ctx = max_ctx_;
    if (const char* c = getenv("STRATA_GLM_SEQ_CTX")) ctx = std::max<int64_t>(8320, std::min<int64_t>(max_ctx_, std::atoll(c)));
    cudaSetDevice(dev_);
    seq_.assign((size_t) n, SeqCtx{});
    // the first sequence's layout recomputed must be the one load_pack made, or the others' would be wrong
    SeqCtx check;
    if (seq_layout(max_ctx_, check) != state_bytes_ || check.dsa_pool != dsa_pool_ || check.kda_conv != kda_conv_ ||
        check.dsa_lat != dsa_lat_ || check.ik_ring != ik_ring_) {
        err = "glm multi: the state layout does not match load_pack's (this build cannot run STRATA_GLM_SEQS)";
        return false;
    }
    seq_[0].state = state_;
    uint64_t more = 0;
    for (int s = 1; s < n; ++s) {
        SeqCtx& c = seq_[(size_t) s];
        seq_layout(ctx, c);
        float* a = nullptr;
        if (cudaMalloc(&a, c.bytes) != cudaSuccess) {
            cudaGetLastError();
            err = "pack: sequence state " + std::to_string(s) + " of " + std::to_string(n) + " (" +
                  std::to_string(c.bytes >> 20) + " MB) does not fit on CUDA" + std::to_string(dev_) +
                  " - fewer STRATA_GLM_SEQS or a smaller STRATA_GLM_SEQ_CTX";
            return false;
        }
        cudaMemset(a, 0, c.bytes);
        c.state = a;
        more += c.bytes;
    }
    std::fprintf(stderr, "glm multi: CUDA%d %d sequence states: the first for %lld tokens (%.2f GB), %d more for %lld "
                 "(%.2f GB in all)\n", dev_, n, (long long) max_ctx_, (double) state_bytes_ / 1e9, n - 1,
                 (long long) ctx, (double) more / 1e9);
    return true;
}

bool Glm5Model::seq_pipe_ready(std::string& err) {
    const Glm5Geometry& g = g_;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->seq_.empty()) {
            err = "glm multi: one sequence state only (STRATA_GLM_SEQS)";
            return false;
        }
        cudaSetDevice(m->dev_);
        for (auto& c : m->seq_) {
            if (c.hop_h != nullptr) continue;
            if (cudaHostAlloc((void**) &c.hop_h, (size_t) g.hc * g.n_embd * sizeof(float), cudaHostAllocPortable) !=
                    cudaSuccess ||
                cudaHostAlloc((void**) &c.emb_h, (size_t) g.n_embd * sizeof(float), cudaHostAllocPortable) !=
                    cudaSuccess ||
                cudaHostAlloc((void**) &c.tok_h, sizeof(int), cudaHostAllocPortable) != cudaSuccess ||
                cudaEventCreateWithFlags(&c.ev_hop, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&c.ev_done, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                err = "glm multi: the pipeline buffers did not allocate";
                return false;
            }
        }
    }
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::multi_issue(int s, int32_t token, const strata::kernels::SamplerParams& sp, std::string& err) {
    if (s < 0 || s >= seq_count() || seq_.empty()) {
        err = "glm multi: no sequence " + std::to_string(s);
        return false;
    }
    if (seq_[0].hop_h == nullptr && !seq_pipe_ready(err)) return false;
    const Glm5Geometry& g = g_;
    seq_bind_all(s);
    FastState* F = fast_;
    SeqCtx& c = seq_[(size_t) s];
    const int64_t p = pos_++;
    if (p + 1 >= max_ctx_) {
        err = "glm multi: the context is full";
        return false;
    }
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm multi: no embedding dequantizer";
        return false;
    }
    // this sequence's own pinned row: its previous token has completed (multi_take), so no copy still reads it
    tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd), c.emb_h,
                 g.n_embd);
    if (const float* img = image_row(p)) std::memcpy(c.emb_h, img, (size_t) g.n_embd * sizeof(float));
    cudaSetDevice(dev_);
    cudaMemcpyAsync(F->emb, c.emb_h, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, F->cs);
    gf::embed_streams(F->emb, state_, g.n_embd, F->cs);
    if (!fast_layers(p, false, err)) return false;
    Glm5Model* tail = this;
    const size_t hop = (size_t) g.hc * g.n_embd * sizeof(float);
    for (Glm5Model* B = split_next_.get(); B != nullptr; B = B->split_next_.get()) {
        SeqCtx& ca = tail->seq_[(size_t) s];
        cudaSetDevice(tail->dev_);
        cudaMemcpyAsync(ca.hop_h, tail->state_, hop, cudaMemcpyDeviceToHost, tail->fast_->cs);
        cudaEventRecord(ca.ev_hop, tail->fast_->cs);
        cudaSetDevice(B->dev_);
        B->pos_ = pos_;
        cudaStreamWaitEvent(B->fast_->cs, ca.ev_hop, 0);
        cudaMemcpyAsync(B->state_, ca.hop_h, hop, cudaMemcpyHostToDevice, B->fast_->cs);
        if (!B->fast_layers(p, true, err)) return false;
        tail = B;
    }
    FastState* FT = tail->fast_;
    SeqCtx& ct = tail->seq_[(size_t) s];
    cudaSetDevice(tail->dev_);
    gf::head_prep(tail->state_, tail->w_.at("output_norm.weight"), g.norm_eps, g.n_embd, FT->head_x, FT->head_xq,
                  FT->cs);
    const WSlot& ow = tail->ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, FT->head_xq, FT->head_x, tail->sc_ + tail->sc_logits, nullptr, 1.0f, ow.type, g.n_embd,
                   g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, FT->cs)) {
        err = "glm multi: output head";
        return false;
    }
    // the logits are the tail's shared scratch: the draw goes on the same stream right behind them, before the next
    // sequence's head can write them again
    if (sp.greedy || sp.temperature <= 0.0f)
        gf::argmax(tail->sc_ + tail->sc_logits, g.n_vocab, FT->tok, FT->cs);
    else
        strata::kernels::sample_tokens(tail->sc_ + tail->sc_logits, 1, (int) g.n_vocab, nullptr, 0, sp, FT->tok,
                                       FT->cs);
    cudaMemcpyAsync(ct.tok_h, FT->tok, sizeof(int), cudaMemcpyDeviceToHost, FT->cs);
    cudaEventRecord(ct.ev_done, FT->cs);
    cudaSetDevice(dev_);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm multi: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

int Glm5Model::multi_take(int s, std::string& err) {
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    SeqCtx& ct = tail->seq_[(size_t) s];
    cudaSetDevice(tail->dev_);
    if (!glmfast::wait_event(ct.ev_done, "glm multi token", err)) return -1;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (!m->fast_->route_ok(err)) return -1;
    if (gf::launch_errors() > 0) {
        err = "glm multi: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return -1;
    }
    cudaSetDevice(dev_);
    const int tok = ct.tok_h[0];
    if (s == cur_seq_) last_tok_ = tok;
    else seq_[(size_t) s].last_tok = tok;
    return tok;
}

bool Glm5Model::multi_boundary(std::string& err) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!m->fast_boundary(err)) return false;
    }
    cudaSetDevice(dev_);
    return true;
}

}  // namespace strata::core
