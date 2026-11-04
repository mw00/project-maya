// src/core/glm_mtp_rows.cu - several conversations drafting with the NextN (MTP) block at once (STRATA_GLM_MULTI_MTP).
//
// The batched rows (glm_rows.cu) take one token of each of a group's conversations through the split.  Here each
// conversation brings a WINDOW instead: the token it feeds (the last one it wrote) and up to d drafts the NextN block
// made for the positions after it - consecutive rows of the group, at consecutive positions of that conversation.
// Every row goes through every layer as a row of the batched step does; a conversation's rows run its recurrent
// states and caches forward in row order (the per-row kernels, never the rows' one-launch KDA), each row with its own
// residual pair.  The last part's head picks a token for every row (row t of a window with the request's draw index
// + t, its penalties over the drafts before it - pick_tokens' window rule; argmax only where rows_head takes it,
// rows_pick_argmax), so row t's pick is meant to be the token the batched step without drafts makes at that position
// (the same other settings) whenever the drafts before it are the tokens written there.
//
// THE ROUND of a group, the host's calls in order (generate.cpp's scheduler):
//   1. mtpw_front / mtpw_back: the rows through the parts and the head (a group's front can run while another
//      group's back does: the split's GPUs work on different groups, as with the plain batched rows);
//   2. mtpw_picks: the picks; the host keeps each window's rows up to the first draft that differs from what was
//      written before it (strata/core/mtp_rows.hpp) - the kept rows' picks are the tokens written;
//   3. mtpw_commit: where a window kept fewer rows than it had, every part's recurrent states (KDA S and conv
//      history) go back to the copy taken before the window and the kept rows are replayed from their inputs (kept
//      by the forward pass); the positions become the window's start + kept rows.  The DSA caches' rows past that are
//      left as they are: a later token rewrites a position before any query can read it (as the verify window's
//      commit, glm_window.cu);
//   4. mtpw_draft_queue / mtpw_draft_take: on the last part, the NextN block's cache takes the kept rows' entries
//      from the trunk's hidden states (glm_mtp.cu's mtp_cache_rows: entry p pairs h_p with the token at p + 1), then
//      drafts the next window - its first step at the last kept row from that row's hidden state and the token just
//      written, each further step from its own hidden state and draft (one step of every conversation queued
//      together, the host feeding the next step its tokens).
// The kept tokens are the ones the plain batched rows would write for the same samples (greedy: the same text).
//
// Buffers (mtpw_alloc, before the expert pool): per conversation and part a copy of its recurrent states (the head
// group's spec backup where there is one) and its window rows' KDA inputs; per part the rows' private residual pairs;
// pinned per conversation: its embedding rows (first part), the hop rows (every part but the last), the picks and the
// NextN cache rows' embeddings (last part); on the last part the rows' trunk hidden states and a draft chain's state.
#include "glm_rows_state.hpp"

#include "strata/core/mtp_rows.hpp"
#include "strata/kernels/sampler.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

namespace {
using glmrows::image_row_in;
using glmrows::row_at;
using glmrows::rowq_at;
}  // namespace

struct Glm5Model::MtpRows {
    int W = 0;                       // a window's rows at most (1 + drafts)
    int n_seq = 0;
    int n_kda = 0;                   // this part's recurrent layers
    std::vector<int> kin_of;         // layer -> its record in a conversation's KDA inputs (-1: not recurrent here)
    size_t kin_layer = 0;            // floats a layer's record (W rows: q k v, f_a, g_a, beta - kda_in's layout)
    size_t run = 0;                  // floats of a recurrent layer's states (S, then the conv history)
    std::vector<float*> kin;         // per conversation: n_kda records (device)
    std::vector<uint8_t> own_bak;    // per conversation: the backup was allocated here (not the spec path's)
    float* res = nullptr;            // rows_max() residual pairs (device)
    // pinned per conversation
    std::vector<float*> emb_h;       // first part: W x n_embd
    std::vector<float*> hop_h;       // parts before the last: W x hc x n_embd
    std::vector<cudaEvent_t> ev_hop;
    std::vector<int*> tok_h;         // last part: W picks
    std::vector<cudaEvent_t> ev_done;
    std::vector<float*> cemb_h;      // last part: W x n_embd (the NextN cache rows' embeddings)
    std::vector<float*> demb_h;      // last part: n_embd (a draft step's embedding)
    // device per conversation, last part
    std::vector<float*> hx;          // W x n_embd: the window rows' trunk hidden states (head_prep's)
    std::vector<float*> dh;          // n_embd: a draft chain's hidden state between its steps
    // a draft step's rows (last part, rows_max() rows): the tokens' embeddings, the block's input and its residual
    float *demb = nullptr, *mh = nullptr;
    uint8_t* catq = nullptr;
    // the window in flight per conversation (this part)
    std::vector<int64_t> p0;
    std::vector<int> cnt;
    // drafting (last part)
    int d_n_ctx = 0, d_max = 0;
    std::vector<int> d_seq, d_n;
    std::vector<int64_t> d_p;
    std::vector<int32_t> d_tok, d_out;
    cudaEvent_t ev_draft = nullptr;
    bool d_pending = false;
};

int Glm5Model::multi_mtp_setting() {
    static const int v = [] {
        const char* e = getenv("STRATA_GLM_MULTI_MTP");
        return e != nullptr ? std::clamp(std::atoi(e), 0, 7) : 0;
    }();
    return v;
}

int Glm5Model::multi_mtp_cap() {
    static const int v = [] {
        const char* e = getenv("STRATA_GLM_MULTI_MTP_ROWS");
        return e != nullptr ? std::clamp(std::atoi(e), 8, gf::kMaxMvNt) : gf::kMaxMvNt;
    }();
    return v;
}

int Glm5Model::mtpw_window() const { return mtpw_ != nullptr ? mtpw_->W : 1; }

size_t Glm5Model::mtpw_kin_offset(int il) const {
    if (mtpw_ == nullptr || il < 0 || il >= (int) mtpw_->kin_of.size() || mtpw_->kin_of[(size_t) il] < 0) return 0;
    return (size_t) mtpw_->kin_of[(size_t) il] * mtpw_->kin_layer;
}

bool Glm5Model::mtpw_alloc(std::string& err) {
    if (mtpw_ != nullptr || multi_mtp_setting() <= 0 || seq_.empty()) return true;
    const Glm5Geometry& g = g_;
    auto* M = new MtpRows();
    mtpw_ = M;
    M->W = std::min(1 + multi_mtp_setting(), std::min(kWinMax, multi_mtp_cap()));
    M->n_seq = seq_count();
    const int n = M->n_seq;
    const size_t DI = (size_t) g.d_inner(), E = (size_t) g.n_embd, hcE = (size_t) g.hc * E;
    M->kin_of.assign((size_t) g.n_layers + 1, -1);
    for (int il = l0_; il < l1_; ++il)
        if (g.is_recr(il)) M->kin_of[(size_t) il] = M->n_kda++;
    M->kin_layer = (size_t) M->W * (3 * DI + 2 * (size_t) g.kda_head_dim + (size_t) g.n_head);
    M->run = DI * (size_t) g.kda_head_dim + 3 * DI * (size_t) (g.d_conv - 1);
    const bool first = l0_ == 0, last = l1_ == g.n_layers;
    cudaSetDevice(dev_);
    uint64_t dev_bytes = 0, pin_bytes = 0;
    const auto dmalloc = [&](void** p, size_t b) {
        if (cudaMalloc(p, b) != cudaSuccess) {
            cudaGetLastError();
            *p = nullptr;
            return false;
        }
        dev_bytes += b;
        return true;
    };
    const auto hmalloc = [&](void** p, size_t b) {
        if (cudaHostAlloc(p, b, cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            *p = nullptr;
            return false;
        }
        pin_bytes += b;
        return true;
    };
    const auto fail = [&](const std::string& what) {
        err = "glm multi mtp: " + what + " did not allocate on CUDA" + std::to_string(dev_) +
              " - fewer conversations at once (--parallel), a smaller MAYA_GLM_SEQ_CTX or fewer drafts";
        return false;
    };
    M->kin.assign((size_t) n, nullptr);
    M->own_bak.assign((size_t) n, 0);
    M->emb_h.assign((size_t) n, nullptr);
    M->hop_h.assign((size_t) n, nullptr);
    M->ev_hop.assign((size_t) n, nullptr);
    M->tok_h.assign((size_t) n, nullptr);
    M->ev_done.assign((size_t) n, nullptr);
    M->cemb_h.assign((size_t) n, nullptr);
    M->demb_h.assign((size_t) n, nullptr);
    M->hx.assign((size_t) n, nullptr);
    M->dh.assign((size_t) n, nullptr);
    M->p0.assign((size_t) n, 0);
    M->cnt.assign((size_t) n, 0);
    const size_t bak_bytes = (size_t) M->n_kda * M->run * sizeof(float);
    for (int s = 0; s < n; ++s) {
        // the recurrent states' copy: the spec path's own where it took one (the same layout), else here
        float*& bak = s == cur_seq_ ? kda_bak_ : seq_[(size_t) s].kda_bak;
        if (bak_bytes > 0 && bak == nullptr) {
            if (!dmalloc((void**) &bak, bak_bytes)) return fail("the recurrent-state copy of conversation " + std::to_string(s));
            M->own_bak[(size_t) s] = 1;
        }
        if (M->n_kda > 0 && !dmalloc((void**) &M->kin[(size_t) s], (size_t) M->n_kda * M->kin_layer * sizeof(float)))
            return fail("the window's recurrent inputs");
        if (first && !hmalloc((void**) &M->emb_h[(size_t) s], (size_t) M->W * E * sizeof(float)))
            return fail("the pinned embedding rows");
        if (!last && (!hmalloc((void**) &M->hop_h[(size_t) s], (size_t) M->W * hcE * sizeof(float)) ||
                      cudaEventCreateWithFlags(&M->ev_hop[(size_t) s], cudaEventDisableTiming) != cudaSuccess))
            return fail("the pinned hop rows");
        if (last && (!hmalloc((void**) &M->tok_h[(size_t) s], (size_t) M->W * sizeof(int)) ||
                     cudaEventCreateWithFlags(&M->ev_done[(size_t) s], cudaEventDisableTiming) != cudaSuccess ||
                     !hmalloc((void**) &M->cemb_h[(size_t) s], (size_t) M->W * E * sizeof(float)) ||
                     !hmalloc((void**) &M->demb_h[(size_t) s], E * sizeof(float)) ||
                     !dmalloc((void**) &M->hx[(size_t) s], (size_t) M->W * E * sizeof(float)) ||
                     !dmalloc((void**) &M->dh[(size_t) s], E * sizeof(float))))
            return fail("the head's window buffers");
    }
    if (!dmalloc((void**) &M->res, (size_t) rows_max() * 2 * hcE * sizeof(float))) return fail("the rows' residuals");
    if (last && (!dmalloc((void**) &M->demb, (size_t) rows_max() * E * sizeof(float)) ||
                 !dmalloc((void**) &M->mh, (size_t) rows_max() * E * sizeof(float)) ||
                 !dmalloc((void**) &M->catq, (size_t) rows_max() * (2 * E / 32) * 36)))
        return fail("the draft rows");
    if (last && cudaEventCreateWithFlags(&M->ev_draft, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        return fail("the drafts' event");
    }
    std::fprintf(stderr, "glm multi mtp: CUDA%d windows of %d rows (%d drafts) for %d conversations, %d rows a group: "
                         "%.0f MB of VRAM, %.1f MB pinned\n", dev_, M->W, M->W - 1, n, rows_max(),
                 (double) dev_bytes / 1048576.0, (double) pin_bytes / 1048576.0);
    return true;
}

bool Glm5Model::mtpw_setup(std::string& err) {
    if (mtpw_ == nullptr) return true;
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    // the commit's replay scratch (win_commit_part's)
    if (mtpw_->n_kda > 0 && F->rep == nullptr &&
        cudaMalloc((void**) &F->rep, (size_t) gf::kKdaLayers * 5 * g.d_inner() * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        F->rep = nullptr;
        err = "glm multi mtp: the commit's replay scratch did not allocate";
        return false;
    }
    return true;
}

void Glm5Model::mtpw_destroy() {
    MtpRows* M = mtpw_;
    if (M == nullptr) return;
    cudaSetDevice(dev_);
    for (int s = 0; s < M->n_seq; ++s) {
        if (M->kin[(size_t) s]) cudaFree(M->kin[(size_t) s]);   // (the recurrent-state copies: with the sequences')
        for (float* p : {M->emb_h[(size_t) s], M->hop_h[(size_t) s], M->cemb_h[(size_t) s], M->demb_h[(size_t) s]})
            if (p) cudaFreeHost(p);
        if (M->tok_h[(size_t) s]) cudaFreeHost(M->tok_h[(size_t) s]);
        for (cudaEvent_t e : {M->ev_hop[(size_t) s], M->ev_done[(size_t) s]})
            if (e) cudaEventDestroy(e);
        for (float* p : {M->hx[(size_t) s], M->dh[(size_t) s]})
            if (p) cudaFree(p);
    }
    if (M->res) cudaFree(M->res);
    for (void* p : {(void*) M->demb, (void*) M->mh, (void*) M->catq})
        if (p) cudaFree(p);
    if (M->ev_draft) cudaEventDestroy(M->ev_draft);
    delete M;
    mtpw_ = nullptr;
}

bool Glm5Model::mtpw_ok() const {
    if (multi_mtp_setting() <= 0 || seq_.empty()) return false;
    const Glm5Model* last = this;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr || m->mtpw_ == nullptr || m->rows_ == nullptr) return false;
        last = m;
    }
    return last->mtp_il_ >= 0 && last->fast_->mtpc.cap > 0 && last->fast_->mtp_tok != nullptr;
}

// a group's window rows on this part: conversation i's rows counts[i] in a row, at its positions p .. p + counts[i] - 1,
// each row with its own residual pair; the rows of a window of more than one keep their KDA inputs
bool Glm5Model::rows_bind_win(const int* seqs, const int* counts, int n, RowSeq* rw, std::string& err) {
    MtpRows* M = mtpw_;
    RowSeq base[gf::kMaxMvNt];
    if (n < 1 || n > gf::kMaxMvNt || !rows_bind(seqs, n, base, err)) {
        if (err.empty()) err = "glm multi mtp: " + std::to_string(n) + " conversations in a group";
        return false;
    }
    const Glm5Geometry& g = g_;
    const int64_t hcE = (int64_t) g.hc * g.n_embd;
    int row = 0;
    for (int c = 0; c < n; ++c) {
        const int k = counts[c];
        const int64_t max_ctx = seq_max_ctx(seqs[c]);
        if (k < 1 || k > M->W || row + k > rows_max()) {
            err = "glm multi mtp: a window of " + std::to_string(k) + " rows (" + std::to_string(row + k) +
                  " in the group; " + std::to_string(M->W) + " a window, " + std::to_string(rows_max()) + " a group)";
            return false;
        }
        if (base[c].p + k >= max_ctx) {
            err = "glm multi mtp: the context of sequence " + std::to_string(seqs[c]) + " is full";
            return false;
        }
        for (int t = 0; t < k; ++t, ++row) {
            RowSeq& r = rw[row];
            r = base[c];
            r.p = base[c].p + t;
            if (seqs[c] == cur_seq_) {
                r.img = image_row(r.p);
            } else {
                const SeqCtx& sc = seq_[(size_t) seqs[c]];
                r.img = image_row_in(sc.img_pos, sc.img_rows, r.p, g.n_embd);
            }
            r.res = M->res + (size_t) row * 2 * (size_t) hcE;
            r.Rc = r.res;
            r.Ro = r.res + hcE;
            r.kin = k > 1 ? M->kin[(size_t) seqs[c]] : nullptr;
            r.kin_t = t;
        }
    }
    return true;
}

// one part's share of a group's step (mtpw_front's parts, mtpw_back's last one)
bool Glm5Model::mtpw_part(const int* seqs, const int* counts, int n, const int32_t* tokens,
                          const strata::kernels::SamplerParams* sps, Glm5Model* prev, bool last, std::string& err) {
    MtpRows* M = mtpw_;
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    const size_t hcE = (size_t) g.hc * E;
    cudaSetDevice(dev_);
    cudaStream_t s = F->cs;
    // (the token path's profiler collects one token's marks at a time: none from a batched step)
    struct ProfOff {
        FastState* F;
        bool was;
        ~ProfOff() { F->prof_on = was; }
    } prof_off{F, F->prof_on};
    F->prof_on = false;
    RowSeq rw[gf::kMaxMvNt];
    if (!rows_bind_win(seqs, counts, n, rw, err)) return false;
    int nr = 0;
    for (int c = 0; c < n; ++c) nr += counts[c];
    // the recurrent states as they were before the window (the commit's restore point), windows of 2+ rows only
    for (int c = 0, row = 0; c < n; row += counts[c], ++c) {
        const int sq = seqs[c];
        M->p0[(size_t) sq] = rw[row].p;
        M->cnt[(size_t) sq] = counts[c];
        if (counts[c] < 2 || M->n_kda == 0) continue;
        float* bak = sq == cur_seq_ ? kda_bak_ : seq_[(size_t) sq].kda_bak;
        if (bak == nullptr) {
            err = "glm multi mtp: no recurrent-state copy for sequence " + std::to_string(sq);
            return false;
        }
        for (int il = l0_; il < l1_; ++il)
            if (M->kin_of[(size_t) il] >= 0)
                cudaMemcpyAsync(bak + (size_t) M->kin_of[(size_t) il] * M->run, rw[row].state + (*rw[row].kda_S)[(size_t) il],
                                M->run * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
    kda_bak_cur_ = false;
    // ---- the rows' input: the tokens' embeddings (first part), else the residual rows the part before handed over
    if (prev == nullptr) {
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
        if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
            err = "glm multi mtp: no embedding dequantizer";
            return false;
        }
        const size_t emb_row = ggml_row_size((ggml_type) pack_emb_type_, E);
        for (int c = 0, row = 0; c < n; row += counts[c], ++c)
            for (int t = 0; t < counts[c]; ++t) {
                // the conversation's own pinned rows: its window before has completed, so no copy still reads them
                float* e = M->emb_h[(size_t) seqs[c]] + (size_t) t * E;
                tt->to_float(pack_emb_src_ + (size_t) tokens[row + t] * emb_row, e, E);
                if (rw[row + t].img != nullptr) std::memcpy(e, rw[row + t].img, (size_t) E * sizeof(float));
                cudaMemcpyAsync(F->emb, e, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
                gf::embed_streams(F->emb, rw[row + t].res, E, s);
            }
    } else {
        for (int c = 0, row = 0; c < n; row += counts[c], ++c) {
            const int sq = seqs[c];
            cudaStreamWaitEvent(s, prev->mtpw_->ev_hop[(size_t) sq], 0);
            for (int t = 0; t < counts[c]; ++t)
                cudaMemcpyAsync(rw[row + t].res, prev->mtpw_->hop_h[(size_t) sq] + (size_t) t * hcE, hcE * sizeof(float),
                                cudaMemcpyHostToDevice, s);
        }
    }
    if (!rows_layers(rw, nr, err)) return false;
    if (!last) {
        // ---- the residual rows to the next part, through this conversation's pinned hop rows
        for (int c = 0, row = 0; c < n; row += counts[c], ++c) {
            const int sq = seqs[c];
            for (int t = 0; t < counts[c]; ++t)
                cudaMemcpyAsync(M->hop_h[(size_t) sq] + (size_t) t * hcE, rw[row + t].res, hcE * sizeof(float),
                                cudaMemcpyDeviceToHost, s);
            cudaEventRecord(M->ev_hop[(size_t) sq], s);
        }
    } else {
        // ---- the head: one read of the output matrix for every row, then each window's picks
        for (int i = 0; i < nr; ++i)
            gf::head_prep(rw[i].res, w_.at("output_norm.weight"), g.norm_eps, E, row_at(S->head_x, E, i),
                          rowq_at(S->head_xq, E, i), s);
        const WSlot& ow = ws_map_.at("output.weight");
        gf::MvJob o = {ow.q, S->head_xq, S->head_x, S->logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
        if (ow.type == 0) o.w = ow.f32;
        if (!gf::mv_nt(&o, 1, nr, s)) {
            err = "glm multi mtp: output head";
            return false;
        }
        for (int c = 0, row = 0; c < n; row += counts[c], ++c) {
            const float* lg = row_at(S->logits, g.n_vocab, row);
            if (strata::core::rows_pick_argmax(sps[c])) {
                for (int t = 0; t < counts[c]; ++t) gf::argmax(lg + (size_t) t * g.n_vocab, g.n_vocab, S->tok + row + t, s);
            } else {
                // row t: the request's draw index + t, its penalties over the window's tokens before it
                pick_tokens(lg, counts[c], sps[c], tokens + row, 0, S->tok + row, s);
            }
            const int sq = seqs[c];
            cudaMemcpyAsync(M->tok_h[(size_t) sq], S->tok + row, (size_t) counts[c] * sizeof(int),
                            cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(M->hx[(size_t) sq], row_at(S->head_x, E, row), (size_t) counts[c] * E * sizeof(float),
                            cudaMemcpyDeviceToDevice, s);
            cudaEventRecord(M->ev_done[(size_t) sq], s);
        }
    }
    for (int c = 0; c < n; ++c) {
        if (seqs[c] == cur_seq_) pos_ += counts[c];
        else seq_[(size_t) seqs[c]].pos += counts[c];
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm multi mtp: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Glm5Model::mtpw_front(const int* seqs, const int* counts, int n, const int32_t* tokens, std::string& err) {
    if (!mtpw_ok()) {
        err = "glm multi mtp: not set up (MAYA_GLM_MULTI_MTP with --parallel, the NextN block loaded)";
        return false;
    }
    if (seq_[0].hop_h == nullptr && !seq_pipe_ready(err)) return false;
    Glm5Model* prev = nullptr;
    for (Glm5Model* m = this; m != nullptr && m->split_next_; prev = m, m = m->split_next_.get())
        if (!m->mtpw_part(seqs, counts, n, tokens, nullptr, prev, false, err)) return false;
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::mtpw_back(const int* seqs, const int* counts, int n, const int32_t* tokens,
                          const strata::kernels::SamplerParams* sps, std::string& err) {
    Glm5Model *L = this, *prev = nullptr;
    while (L->split_next_) {
        prev = L;
        L = L->split_next_.get();
    }
    if (!L->mtpw_part(seqs, counts, n, tokens, sps, prev, true, err)) return false;
    // every part at the same positions (a conversation in one group at a time)
    for (int c = 0; c < n; ++c)
        if (L->mtpw_->p0[(size_t) seqs[c]] != mtpw_->p0[(size_t) seqs[c]]) {
            err = "glm multi mtp: sequence " + std::to_string(seqs[c]) + " is at a different position on CUDA" +
                  std::to_string(L->dev_);
            return false;
        }
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::mtpw_picks(const int* seqs, const int* counts, int n, int32_t* picks, std::string& err) {
    Glm5Model* L = this;
    while (L->split_next_) L = L->split_next_.get();
    cudaSetDevice(L->dev_);
    for (int c = 0, row = 0; c < n; row += counts[c], ++c) {
        const int sq = seqs[c];
        if (!glmfast::wait_event(L->mtpw_->ev_done[(size_t) sq], "glm multi mtp picks", err)) return false;
        for (int t = 0; t < counts[c]; ++t) picks[row + t] = L->mtpw_->tok_h[(size_t) sq][t];
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (!m->fast_->route_ok(err)) return false;
    if (gf::launch_errors() > 0) {
        err = "glm multi mtp: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    cudaSetDevice(dev_);
    ++mtpw_steps_;
    for (int c = 0; c < n; ++c) mtpw_rows_ += (uint64_t) counts[c];
    return true;
}

bool Glm5Model::mtpw_commit_part(const int* seqs, const int* counts, const int* keep, int n, std::string& err) {
    MtpRows* M = mtpw_;
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaSetDevice(dev_);
    cudaStream_t s = F->cs;
    const size_t DI = (size_t) g.d_inner(), sd = DI * (size_t) g.kda_head_dim;
    std::vector<int> layers;
    for (int il = l0_; il < l1_; ++il)
        if (M->kin_of[(size_t) il] >= 0) layers.push_back(il);
    for (int c = 0; c < n; ++c) {
        const int sq = seqs[c];
        if (M->cnt[(size_t) sq] != counts[c] || keep[c] < 1 || keep[c] > counts[c]) {
            err = "glm multi mtp: commit of " + std::to_string(keep[c]) + " rows of sequence " + std::to_string(sq) +
                  " without a window of as many";
            return false;
        }
        float* state = sq == cur_seq_ ? state_ : seq_[(size_t) sq].state;
        const std::vector<int64_t>& kda_S = sq == cur_seq_ ? kda_S_ : seq_[(size_t) sq].kda_S;
        if (keep[c] < counts[c] && !layers.empty()) {
            float* bak = sq == cur_seq_ ? kda_bak_ : seq_[(size_t) sq].kda_bak;
            float* kin = M->kin[(size_t) sq];
            // the kept rows replayed onto the copy (the states before the window) through up to kKdaLayers layers at
            // once - each layer's arithmetic the forward pass's per-row kernels' - then the copy back to the live
            // states (win_commit_part's batched replay)
            for (int t = 0; t < keep[c]; ++t)
                for (size_t c0 = 0; c0 < layers.size(); c0 += gf::kKdaLayers) {
                    gf::KdaPrepArgs pa[gf::kKdaLayers];
                    gf::KdaRecArgs ra[gf::kKdaLayers];
                    const int nl = (int) std::min<size_t>(gf::kKdaLayers, layers.size() - c0);
                    for (int i = 0; i < nl; ++i) {
                        const int il = layers[c0 + (size_t) i];
                        const auto& Ly = F->L[(size_t) il];
                        float* b = bak + (size_t) M->kin_of[(size_t) il] * M->run;
                        float* sc = F->rep + (size_t) i * 5 * DI;
                        const size_t T = (size_t) M->W, tt = (size_t) t;
                        float* base = kin + (size_t) M->kin_of[(size_t) il] * M->kin_layer;   // (kda_in's layout)
                        float* rest = base + 3 * T * DI;
                        gf::KdaPrepArgs& k = pa[i];
                        for (int r = 0; r < 3; ++r) {
                            k.proj[r] = base + ((size_t) r * T + tt) * DI;
                            k.conv_w[r] = Ly.conv[r];
                            k.out[r] = sc + (size_t) r * DI;
                        }
                        k.conv_state = b + sd;
                        k.fa = rest + tt * (size_t) g.kda_head_dim;
                        k.ga = rest + (T + tt) * (size_t) g.kda_head_dim;
                        k.f_b = Ly.f_b;
                        k.g_b = Ly.g_b;
                        k.dt_bias = Ly.dt_bias;
                        k.ssm_a = Ly.ssm_a;
                        k.lower_bound = g.kda_lb;
                        k.g1 = sc + 3 * DI;
                        k.g2 = sc + 4 * DI;
                        k.n_head = g.n_head;
                        k.head_dim = g.kda_head_dim;
                        k.d_conv = g.d_conv;
                        gf::KdaRecArgs& r = ra[i];
                        r.q = k.out[0];
                        r.k = k.out[1];
                        r.v = k.out[2];
                        r.g1 = k.g1;
                        r.beta_raw = rest + 2 * T * (size_t) g.kda_head_dim + tt * (size_t) g.n_head;
                        r.state = b;
                        r.g2 = k.g2;
                        r.norm_w = Ly.ssm_norm;
                    }
                    gf::kda_layers(pa, ra, nl, g.norm_eps, s);
                }
            for (const int il : layers)
                cudaMemcpyAsync(state + kda_S[(size_t) il], bak + (size_t) M->kin_of[(size_t) il] * M->run,
                                M->run * sizeof(float), cudaMemcpyDeviceToDevice, s);
        }
        const int64_t p = M->p0[(size_t) sq] + keep[c];
        if (sq == cur_seq_) pos_ = p;
        else seq_[(size_t) sq].pos = p;
        M->cnt[(size_t) sq] = 0;
    }
    kda_bak_cur_ = false;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm multi mtp commit: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Glm5Model::mtpw_commit(const int* seqs, const int* counts, const int* keep, int n, std::string& err) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (g_.kda_head_dim != 128 && m->mtpw_->n_kda > 0) {
            err = "glm multi mtp: the commit's replay is built for a KDA head_dim of 128";
            return false;
        }
        if (!m->mtpw_commit_part(seqs, counts, keep, n, err)) return false;
    }
    cudaSetDevice(dev_);
    return true;
}

// a conversation's binding on this part for the NextN block's rows (no context check: rows_bind's)
void Glm5Model::mtpw_bind(int sq, RowSeq& r) const {
    r = RowSeq{};
    r.seq = sq;
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
    }
    r.res = r.Rc = r.state;
    r.Ro = r.state + (int64_t) g_.hc * g_.n_embd;
}

// mtp_cache_rows for conversation rb (bound or not): entries p0 .. p0 + n - 1 of its NextN block's DSA caches from the
// trunk's hidden states h and the tokens after them (next[i]: the one at p0 + i + 1; an image row there stands in)
bool Glm5Model::mtpw_cache_rows(const RowSeq& rb, int64_t p0, int n, const float* h, const int32_t* next, float* emb_h,
                                std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    auto& C = F->mtpc;
    if (n <= 0) return true;
    if (n > C.cap) {
        err = "glm multi mtp: more cache entries than the window holds";
        return false;
    }
    cudaStream_t s = F->cs;
    const int64_t E = g.n_embd;
    const int il = mtp_il_;
    const auto& Ly = F->L[(size_t) il];
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm multi mtp: no embedding dequantizer";
        return false;
    }
    const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, g.n_embd);
    for (int i = 0; i < n; ++i) {
        float* row = emb_h + (size_t) i * E;
        const int64_t q = p0 + i + 1;
        const float* img = rb.seq == cur_seq_ ? image_row(q)
                                              : image_row_in(seq_[(size_t) rb.seq].img_pos, seq_[(size_t) rb.seq].img_rows,
                                                             q, g.n_embd);
        if (img != nullptr) std::memcpy(row, img, (size_t) E * sizeof(float));
        else tt->to_float(pack_emb_src_ + (size_t) next[i] * row_b, row, g.n_embd);
    }
    const size_t q2 = (size_t) (2 * E / 32) * 36, q1 = (size_t) (E / 32) * 36;
    cudaMemcpyAsync(C.emb, emb_h, (size_t) n * E * sizeof(float), cudaMemcpyHostToDevice, s);
    for (int i = 0; i < n; ++i)
        gf::mtp_in(C.emb + (size_t) i * E, h + (size_t) i * E, Ly.enorm, Ly.hnorm, g.norm_eps, g.n_embd,
                   (uint8_t*) C.catq + (size_t) i * q2, s);
    gf::MvJob eh = {Ly.eh.q, C.catq, nullptr, C.hid, nullptr, 1.0f, Ly.eh.type, (int) (2 * E), g.n_embd};
    eh.ncol = n;
    eh.xq_cstride = (int64_t) q2;
    eh.y_cstride = E;
    if (!gf::mv(&eh, 1, s)) {
        err = "glm multi mtp: eh_proj rows";
        return false;
    }
    for (int i = 0; i < n; ++i)
        gf::rms_q8(C.hid + (size_t) i * E, nullptr, Ly.attn_norm, g.norm_eps, g.n_embd, C.x + (size_t) i * E,
                   (uint8_t*) C.xq + (size_t) i * q1, s);
    gf::MvJob j[3];
    j[0] = {Ly.kv_a.q, C.xq, C.x, C.kv_raw, nullptr, 1.0f, Ly.kv_a.type, g.n_embd, g.kv_lora};
    j[1] = {Ly.idx_k, nullptr, C.x, C.ik_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[2] = {Ly.idx_gate, nullptr, C.x, C.ig_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    for (int k = 0; k < 3; ++k) {
        j[k].ncol = n;
        j[k].xq_cstride = (int64_t) q1;
        j[k].xf_cstride = E;
        j[k].y_cstride = k == 0 ? g.kv_lora : g.idx_key;
    }
    if (!gf::mv(j, 3, s)) {
        err = "glm multi mtp: the cache rows' projections";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        gf::DsaPrepArgs d;
        d.qr_raw = nullptr;   // the caches only
        d.q_lora = g.q_lora;
        d.kv_raw = C.kv_raw + (size_t) i * g.kv_lora;
        d.kv_norm = Ly.kv_a_norm;
        d.kv_lora = g.kv_lora;
        d.lat_q8 = lat_q8_;
        kv_prep_targets(rb.kv, il, (uint16_t*) (rb.state + (*rb.dsa_lat)[(size_t) il]), d);
        d.ik_raw = C.ik_raw + (size_t) i * g.idx_key;
        d.k_norm_w = Ly.k_norm_w;
        d.k_norm_b = Ly.k_norm_b;
        d.ik_cache = rb.state + (*rb.dsa_ik)[(size_t) il];
        d.ig_raw = C.ig_raw + (size_t) i * g.idx_key;
        d.ig_cache = rb.state + (*rb.dsa_ig)[(size_t) il];
        d.ape = Ly.ape;
        d.pooled = rb.state + (*rb.dsa_pool)[(size_t) il];
        d.idx_key = g.idx_key;
        d.kpool = g.idx_kpool;
        d.ring = rb.ring;
        d.p = (int) (p0 + i);
        d.eps = g.norm_eps;
        gf::dsa_prep(d, s);
    }
    return true;
}

// one draft step of nr conversations at once through the NextN block (rows rr[r]: their bindings at the step's
// position): the block's input from each one's token embedding (M->demb rows) and last hidden state (M->dh), eh_proj,
// the DSA mixer and the MoE FFN as the batched rows run them (rows_dsa / rows_moe on the block's layer), the head once
// for all, each row's argmax into F->mtp_tok[slot[r]]; each one's hidden state back to M->dh
bool Glm5Model::mtpw_draft_rows(RowSeq* rr, const int* slot, int nr, std::string& err) {
    MtpRows* M = mtpw_;
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    cudaStream_t s = F->cs;
    const int il = mtp_il_;
    const auto& Ly = F->L[(size_t) il];
    const size_t q2 = (size_t) (2 * E / 32) * 36;
    for (int r = 0; r < nr; ++r)
        gf::mtp_in(M->demb + (size_t) r * E, M->dh[(size_t) rr[r].seq], Ly.enorm, Ly.hnorm, g.norm_eps, E,
                   M->catq + (size_t) r * q2, s);
    gf::MvJob eh = {Ly.eh.q, M->catq, nullptr, M->mh, nullptr, 1.0f, Ly.eh.type, 2 * E, E};
    if (!gf::mv_nt(&eh, 1, nr, s)) {
        err = "glm multi mtp: the drafts' eh_proj";
        return false;
    }
    for (int r = 0; r < nr; ++r)
        gf::rms_q8(M->mh + (size_t) r * E, nullptr, Ly.attn_norm, g.norm_eps, E, row_at(S->x, E, r),
                   rowq_at(S->xq, E, r), s);
    if (!rows_dsa(il, rr, nr, err)) return false;
    for (int r = 0; r < nr; ++r)
        gf::rms_q8(M->mh + (size_t) r * E, row_at(S->mixer, E, r), Ly.ffn_norm, g.norm_eps, E, row_at(S->x, E, r),
                   rowq_at(S->xq, E, r), s);
    if (!rows_moe(il, rr, nr, err)) return false;
    for (int r = 0; r < nr; ++r) {
        gf::rms_q8(M->mh + (size_t) r * E, row_at(S->ffn, E, r), Ly.shnorm, g.norm_eps, E, row_at(S->head_x, E, r),
                   rowq_at(S->head_xq, E, r), s);
        cudaMemcpyAsync(M->dh[(size_t) rr[r].seq], row_at(S->head_x, E, r), (size_t) E * sizeof(float),
                        cudaMemcpyDeviceToDevice, s);
    }
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, S->head_xq, S->head_x, S->logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv_nt(&o, 1, nr, s)) {
        err = "glm multi mtp: the drafts' head";
        return false;
    }
    for (int r = 0; r < nr; ++r) gf::argmax(row_at(S->logits, g.n_vocab, r), g.n_vocab, F->mtp_tok + slot[r], s);
    return true;
}

bool Glm5Model::mtpw_draft_queue(const int* seqs, const int* keep, int n, const int32_t* written, const int32_t* last,
                                 const int* drafts, std::string& err) {
    Glm5Model* L = this;
    while (L->split_next_) L = L->split_next_.get();
    MtpRows* M = L->mtpw_;
    FastState* F = L->fast_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    cudaSetDevice(L->dev_);
    cudaStream_t s = F->cs;
    if (n > 16) {   // (F->mtp_tok holds 16 ids: one a conversation of the group)
        err = "glm multi mtp: more than 16 conversations draft at once";
        return false;
    }
    static const bool whole_head = [] {
        const char* e = getenv("STRATA_GLM_MULTI_MTP_VOCAB");
        return e != nullptr && std::atoi(e) == 0;
    }();
    const bool sub = !whole_head && F->mtp_dv > 0;
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm multi mtp: no embedding dequantizer";
        return false;
    }
    const size_t emb_row = ggml_row_size((ggml_type) pack_emb_type_, E);
    M->d_n_ctx = n;
    M->d_max = 0;
    M->d_seq.assign(seqs, seqs + n);
    M->d_n.assign((size_t) n, 0);
    M->d_p.assign((size_t) n, 0);
    M->d_tok.assign((size_t) n, -1);
    M->d_out.assign((size_t) n * (size_t) std::max(1, M->W - 1), -1);
    M->d_pending = false;
    // STRATA_GLM_MULTI_MTP_DRAFT_ROWS (on): the conversations' draft steps as batched rows of the NextN block, nothing
    // bound; 0: each conversation's steps through the token path's block (fast_mtp), the last part bound to it
    static const bool draft_rows = [] {
        const char* e = getenv("STRATA_GLM_MULTI_MTP_DRAFT_ROWS");
        return e == nullptr || std::atoi(e) != 0;
    }();
    RowSeq base[gf::kMaxMvNt];
    // ---- the NextN cache: entries p0 .. p0 + keep - 2 from the kept rows' trunk states and the tokens after them
    //      (entry p0 + keep - 1 is the first draft step's own), then the chain's start
    for (int c = 0, at = 0; c < n; at += std::max(0, keep[c] - 1), ++c) {
        const int sq = seqs[c];
        const int k = keep[c];
        const int64_t p0 = M->p0[(size_t) sq];   // (the window's start; the commit set the position to p0 + k)
        if (draft_rows) {
            L->mtpw_bind(sq, base[c]);
            if (k > 1 && !L->mtpw_cache_rows(base[c], p0, k - 1, M->hx[(size_t) sq], written + at,
                                             M->cemb_h[(size_t) sq], err))
                return false;
        } else {
            L->seq_bind(sq);
            if (k > 1 && !L->mtp_cache_rows(p0, k - 1, M->hx[(size_t) sq], written + at, err, false,
                                            M->cemb_h[(size_t) sq]))
                return false;
        }
        const int d = std::max(0, std::min(drafts[c], M->W - 1));
        M->d_n[(size_t) c] = d;
        M->d_p[(size_t) c] = p0 + k - 1;
        M->d_tok[(size_t) c] = last[c];
        M->d_max = std::max(M->d_max, d);
        if (d > 0)
            cudaMemcpyAsync(M->dh[(size_t) sq], M->hx[(size_t) sq] + (size_t) (k - 1) * E, (size_t) E * sizeof(float),
                            cudaMemcpyDeviceToDevice, s);
    }
    // ---- the drafts: step j of every conversation that drafts that far, queued together; the host reads the step's
    //      drafts back and feeds them to the next step (the last step's: mtpw_draft_take)
    // (fast_mtp's way: without the draft vocabulary the head - the step's largest read - runs once for every
    // conversation's row of the step)
    RowsState* S = L->rows_;
    const size_t xq_b = (size_t) (E / 32) * 36;
    for (int j = 0; j < M->d_max; ++j) {
        int rowc[gf::kMaxMvNt];
        RowSeq rr[gf::kMaxMvNt];
        int nr = 0;
        for (int c = 0; c < n; ++c) {
            if (M->d_n[(size_t) c] <= j) continue;
            const int sq = seqs[c];
            float* e = M->demb_h[(size_t) sq];   // (the step before it was waited for: no copy still reads it)
            tt->to_float(pack_emb_src_ + (size_t) M->d_tok[(size_t) c] * emb_row, e, E);
            if (draft_rows) {
                rr[nr] = base[c];
                rr[nr].p = M->d_p[(size_t) c] + j;
                cudaMemcpyAsync(M->demb + (size_t) nr * E, e, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
                rowc[nr++] = c;
                continue;
            }
            L->seq_bind(sq);
            cudaMemcpyAsync(F->head_x, M->dh[(size_t) sq], (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s);
            if (!L->fast_mtp(M->d_p[(size_t) c] + j, M->d_tok[(size_t) c], err, c, sub, false, e, sub)) return false;
            cudaMemcpyAsync(M->dh[(size_t) sq], F->head_x, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s);
            if (!sub) {
                cudaMemcpyAsync(row_at(S->head_x, E, nr), F->head_x, (size_t) E * sizeof(float),
                                cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(S->head_xq + (size_t) nr * xq_b, F->head_xq, xq_b, cudaMemcpyDeviceToDevice, s);
                rowc[nr++] = c;
            }
        }
        if (draft_rows && nr > 0) {
            if (!L->mtpw_draft_rows(rr, rowc, nr, err)) return false;
        } else if (nr > 0) {
            const WSlot& ow = L->ws_map_.at("output.weight");
            gf::MvJob o = {ow.q, S->head_xq, S->head_x, S->logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
            if (ow.type == 0) o.w = ow.f32;
            if (!gf::mv_nt(&o, 1, nr, s)) {
                err = "glm multi mtp: the drafts' head";
                return false;
            }
            for (int r = 0; r < nr; ++r)
                gf::argmax(row_at(S->logits, g.n_vocab, r), g.n_vocab, F->mtp_tok + rowc[r], s);
        }
        cudaMemcpyAsync(F->mtp_tok_h, F->mtp_tok, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost, s);
        cudaEventRecord(M->ev_draft, s);
        if (j + 1 == M->d_max) {
            M->d_pending = true;
            break;
        }
        if (!glmfast::wait_event(M->ev_draft, "glm multi mtp drafts", err) || !F->route_ok(err)) return false;
        for (int c = 0; c < n; ++c)
            if (M->d_n[(size_t) c] > j) {
                M->d_tok[(size_t) c] = F->mtp_tok_h[c];
                M->d_out[(size_t) c * (size_t) (M->W - 1) + (size_t) j] = F->mtp_tok_h[c];
            }
    }
    const cudaError_t e = cudaGetLastError();
    cudaSetDevice(dev_);
    if (e != cudaSuccess) {
        err = std::string("glm multi mtp drafts: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Glm5Model::mtpw_draft_take(int32_t* out, std::string& err) {
    Glm5Model* L = this;
    while (L->split_next_) L = L->split_next_.get();
    MtpRows* M = L->mtpw_;
    FastState* F = L->fast_;
    cudaSetDevice(L->dev_);
    if (M->d_pending) {
        M->d_pending = false;
        if (!glmfast::wait_event(M->ev_draft, "glm multi mtp drafts", err) || !F->route_ok(err)) return false;
        const int j = M->d_max - 1;
        for (int c = 0; c < M->d_n_ctx; ++c)
            if (M->d_n[(size_t) c] > j) M->d_out[(size_t) c * (size_t) (M->W - 1) + (size_t) j] = F->mtp_tok_h[c];
    }
    if (gf::launch_errors() > 0) {
        err = "glm multi mtp: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    const int D = std::max(1, M->W - 1);
    for (int c = 0; c < M->d_n_ctx; ++c)
        for (int j = 0; j < D; ++j)
            out[(size_t) c * D + (size_t) j] = j < M->d_n[(size_t) c] ? M->d_out[(size_t) c * D + (size_t) j] : -1;
    cudaSetDevice(dev_);
    return true;
}

}  // namespace strata::core
