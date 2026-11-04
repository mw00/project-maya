// src/core/glm_mtp.cu - the one-GPU MTP decode: the model's NextN (MTP) block drafts, the verify window
// (glm_window.cu) checks.
//
// A ROUND, from the last emitted token y at position p0 (the trunk's hidden state of p0 - 1 in head_x):
//
//   1. draft   - the NextN block at p0 - 1 reads (h_{p0-1}, y) and proposes d1; its own output hidden state
//                (shared_head_norm's, which also feeds its head) and d1 make the next step at p0, ... up to n drafts
//                (STRATA_GLM_MTP_DRAFT, 3; greedy: each draft is the head's argmax over the draft vocabulary);
//   2. verify  - window([y, d1 .. dn]): the rows through every layer in one batched pass, each row's pick the one the
//                token-at-a-time decode makes there (the request's sampler, draw index counter + t);
//   3. accept  - d_{t+1} stands while it matches the pick at row t; the first mismatch is the new token, a full match
//                leaves the last row's pick as a bonus token;
//   4. commit  - window_commit(keep): the recurrent states advanced by the kept rows;
//   5. cache   - the NextN block's cache takes the kept rows with the trunk's hidden states (its cache-writing half:
//                the entries the drafts wrote from their own hidden states are replaced).
//
// The tokens are exactly what the token-at-a-time loop produces for the same samples.  A length 0 round is a plain token
// (the token path, which keeps the NextN cache: mtp_fill_prev).  The length adapts per machine (STRATA_GLM_MTP_ADAPT,
// on): every round's wall time is measured per length and the draft positions' acceptance counted, and the length with
// the most tokens per second wins.
//
// Where it runs: one GPU that holds the routed experts (a verify row costs its experts in full where they stream from
// RAM) - by default the Strix / Gorgon Halo APUs (gfx115x), whose memory holds the whole model; STRATA_GLM_MTP_DECODE=1
// asks for it elsewhere, =0 turns it off.  Not with several conversations at once (--parallel) or a DFlash drafter.
//
// The NextN cache's entry p pairs h_p with the token at p + 1 (the prompt path writes them so).
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

namespace gf = strata::kernels::glmf;

namespace strata::core {

namespace {
constexpr size_t kQ8 = 36;   // bytes per q8_1 block (32 values)

size_t q8_bytes(int64_t n) { return (size_t) (n / 32) * kQ8; }

// the drafts per round at most: STRATA_GLM_MTP_DRAFT (3; 0 turns the MTP decode off)
int mtp_draft_cap() {
    static const int v = [] {
        const char* e = getenv("STRATA_GLM_MTP_DRAFT");
        return e ? std::max(0, std::min(Glm5Model::kWinMax - 1, std::atoi(e))) : 3;
    }();
    return v;
}

// STRATA_GLM_MTP_CHAIN: the drafts of a round go out as one chain on the device - each step's argmax feeds the next
// step's embedding there (token_embd kept on the device, embed_tok) - and the host reads them back once, instead of a
// wait, a host dequantization and a copy per step; nor does the round wait for the device before its drafts or its
// NextN cache rows (the host's own rows go through their own pinned buffers).  Unset: on where the device holds every
// routed expert and token_embd besides; 1 = on wherever token_embd is Q6_K or Q8_0; 0 = off (step by step).
bool mtp_chain_wanted() {
    static const bool v = [] {
        const char* e = getenv("STRATA_GLM_MTP_CHAIN");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return v;
}
}  // namespace

// load_pack / fast_setup: whether this model decodes with the MTP block on one GPU (see the top of this file)
bool Glm5Model::mtp_decode_wanted(int dev, int n_parts) {
    if (getenv("STRATA_GLM_NO_SPEC") != nullptr || getenv("STRATA_GLM_NO_MTP") != nullptr || mtp_draft_cap() <= 0)
        return false;
    if (const char* q = getenv("STRATA_GLM_SEQS"); q != nullptr && std::atoi(q) > 1) return false;   // (--parallel)
    if (const char* d = getenv("STRATA_GLM_DFLASH"); d != nullptr && d[0] != '\0') return false;
    if (const char* e = getenv("STRATA_GLM_MTP_DECODE")) return std::atoi(e) != 0 && n_parts == 1;
    if (n_parts != 1) return false;
#if defined(STRATA_USE_HIP)
    // the Strix / Gorgon Halo APUs (gfx1150 / gfx1151 / gfx1152): their memory holds every expert
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    return std::strncmp(prop.gcnArchName, "gfx115", 6) == 0;
#else
    (void) dev;
    return false;
#endif
}

bool Glm5Model::mtp_ready() const { return fast_ != nullptr && mtp_dec_ && mtp_il_ >= 0 && split_next_ == nullptr; }

// ---------------------------------------------------------------- the draft head
// A draft step ran the whole output head - 154880 x 4096 Q6_K, 0.52 GB, ~2.4 ms of the ~4 ms a step on the Radeon 8065S
// (Gorgon Halo) - to argmax one guess.  A draft only has to be a good guess: the verify decides every token, the output
// is the same whatever the drafts are.  So the draft head runs over a DRAFT VOCABULARY: the full head's own rows for its
// tokens (a draft that lies in it is the one the full head makes), the rest left out.
//
// Which tokens.  GLM's byte-level BPE lays its vocabulary out by script: ids [0, 98304) the Latin / code block (English,
// code, markup, the Latin-script languages), then the Han block (the most frequent first) with the multi-digit numbers
// in it, then Cyrillic, Arabic, Kana ..., the control tokens from 154820.  English or code output stays in the first
// block but for its numbers.  Maya-S's greedy answers to the bench's prompts (2304 tokens: Python, Rust, HTML/JS,
// TCP/UDP, photosynthesis, the history of computing) had 2 tokens outside the vocabulary below (98,304 + 437 rows, 64%
// of the head).  A miss costs the round its later drafts, ~0.9% of the tokens a round for each 1% missed, against the
// ~2% of a round the smaller head saves.  So the vocabulary is read from the GGUF's tokens:
//   - the leading block: the ids up to the first 1024 of which fewer than half are ASCII, and the ASCII ones that
//     begin that thousand;
//   - after it, the tokens that are digits only, and the control / user-defined ones (the roles, <think>, the tool-call
//     tags: a draft can still end a turn or open a call).
// STRATA_GLM_MTP_DRAFT_VOCAB=<K>: the first K ids as the leading block (the rest as above); 0 = the whole head.  Where
// the answer leaves the vocabulary (another script) the drafts go back to the whole head by themselves (decode_mtp:
// the share of the emitted tokens outside it).

// a GPT-2 byte-level token's text: all of its bytes ASCII (code points 0x21-0x7E stand for themselves, 0x100-0x120 for
// the bytes 0x00-0x20, 0x121 for 0x7F), and all of it digits
static void token_kind(const std::string& s, bool& ascii, bool& digits) {
    ascii = digits = !s.empty();
    for (size_t i = 0; i < s.size();) {
        uint32_t c = (uint8_t) s[i];
        int len = 1;
        if (c >= 0xF0) {
            c &= 0x07;
            len = 4;
        } else if (c >= 0xE0) {
            c &= 0x0F;
            len = 3;
        } else if (c >= 0xC0) {
            c &= 0x1F;
            len = 2;
        }
        for (int k = 1; k < len && i + (size_t) k < s.size(); ++k) c = (c << 6) | ((uint8_t) s[i + (size_t) k] & 0x3Fu);
        i += (size_t) len;
        if (!((c >= 0x21 && c <= 0x7E) || (c >= 0x100 && c <= 0x121))) ascii = false;
        if (c < '0' || c > '9') digits = false;
    }
}

// load_pack, the part with the NextN block: the draft vocabulary from the GGUF's tokenizer (mtp_vlead_, mtp_vextra_,
// mtp_vin_; left empty - the whole head - without a byte-level BPE vocabulary of n_vocab tokens, or where it would
// not pay: more than 90% of the head, or a leading block under an eighth of it)
void Glm5Model::mtp_vocab_scan(const std::vector<std::unique_ptr<strata::GgufFile>>& gfs) {
    mtp_vlead_ = 0;
    mtp_vextra_.clear();
    mtp_vin_.clear();
    static const int env = [] {
        const char* e = getenv("STRATA_GLM_MTP_DRAFT_VOCAB");
        return e != nullptr ? std::max(0, std::atoi(e)) : -1;
    }();
    if (env == 0) return;
    // (the reader's metadata keeps a sample of each array: the two arrays read whole)
    strata::MetaValue toks_v, types_v;
    const strata::MetaValue *toks = nullptr, *types = nullptr, *model = nullptr;
    for (const auto& gf : gfs) {
        if (toks == nullptr && gf->get_full("tokenizer.ggml.tokens", toks_v)) toks = &toks_v;
        if (types == nullptr && gf->get_full("tokenizer.ggml.token_type", types_v)) types = &types_v;
        if (model == nullptr) model = gf->get("tokenizer.ggml.model");
    }
    const int64_t N = g_.n_vocab;
    if (toks == nullptr || model == nullptr || model->s != "gpt2" || (int64_t) toks->items.size() != N) {
        std::fprintf(stderr, "glm mtp: no byte-level BPE vocabulary of %lld tokens in the GGUF - the drafts take the "
                             "whole head\n", (long long) N);
        return;
    }
    if (types != nullptr && (int64_t) types->items.size() != N) types = nullptr;
    std::vector<uint8_t> asc((size_t) N), dig((size_t) N);
    for (int64_t i = 0; i < N; ++i) {
        bool a = false, d = false;
        token_kind(toks->items[(size_t) i].s, a, d);
        asc[(size_t) i] = a;
        dig[(size_t) i] = d;
    }
    int64_t lead = 0;
    if (env > 0) {
        lead = std::min<int64_t>(env, N);
    } else {
        constexpr int64_t kBucket = 1024;
        while (lead < N) {
            const int64_t n = std::min(kBucket, N - lead);
            int64_t a = 0;
            for (int64_t i = lead; i < lead + n; ++i) a += asc[(size_t) i];
            if (2 * a < n) break;
            lead += n;
        }
        while (lead < N && asc[(size_t) lead]) ++lead;
    }
    std::vector<int32_t> extra;
    for (int64_t i = lead; i < N; ++i) {
        const int ty = types != nullptr ? (int) types->items[(size_t) i].u : 1;
        if (dig[(size_t) i] || ty == 3 || ty == 4) extra.push_back((int32_t) i);   // (3 control, 4 user-defined)
    }
    const int64_t n = lead + (int64_t) extra.size();
    if (n > N * 9 / 10 || (env < 0 && lead < N / 8)) return;
    mtp_vlead_ = (int) lead;
    mtp_vextra_ = std::move(extra);
    mtp_vin_.assign((size_t) N, 0);
    std::fill(mtp_vin_.begin(), mtp_vin_.begin() + lead, (uint8_t) 1);
    for (const int32_t i : mtp_vextra_) mtp_vin_[(size_t) i] = 1;
}

// fast_setup, the part with the NextN block: the draft head's gathered rows, and the chain's token_embd
void Glm5Model::mtp_draft_setup(bool chain) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const WSlot& ow = ws_map_.at("output.weight");
    const size_t hb = ow.type == 0 ? (size_t) g.n_embd * sizeof(float) : ggml_row_size((ggml_type) ow.type, g.n_embd);
    const void* head = ow.type == 0 ? (const void*) ow.f32 : ow.q;
    if (mtp_vlead_ > 0 && head != nullptr) {
        const int dx = (int) mtp_vextra_.size();
        bool ok = true;
        if (dx > 0) {
            ok = cudaMalloc(&F->mtp_dx_w, (size_t) dx * hb) == cudaSuccess &&
                 cudaMalloc((void**) &F->mtp_dx_ids, (size_t) dx * sizeof(int)) == cudaSuccess;
            for (int i = 0; i < dx && ok; ++i)
                ok = cudaMemcpy((uint8_t*) F->mtp_dx_w + (size_t) i * hb,
                                (const uint8_t*) head + (size_t) mtp_vextra_[(size_t) i] * hb, hb,
                                cudaMemcpyDeviceToDevice) == cudaSuccess;
            ok = ok && cudaMemcpy(F->mtp_dx_ids, mtp_vextra_.data(), (size_t) dx * sizeof(int),
                                  cudaMemcpyHostToDevice) == cudaSuccess;
        }
        if (ok) {
            F->mtp_dv = mtp_vlead_;
            F->mtp_dx = dx;
        } else {
            cudaGetLastError();
            if (F->mtp_dx_w) cudaFree(F->mtp_dx_w);
            if (F->mtp_dx_ids) cudaFree(F->mtp_dx_ids);
            F->mtp_dx_w = nullptr;
            F->mtp_dx_ids = nullptr;
            mtp_vin_.clear();
        }
    }
    // the chain: token_embd as stored, where the device has the room (or STRATA_GLM_MTP_CHAIN=1 says so)
    const char* ce = getenv("STRATA_GLM_MTP_CHAIN");
    const bool want = chain && mtp_chain_wanted(), forced = chain && ce != nullptr && std::atoi(ce) != 0;
    const size_t rb = ggml_row_size((ggml_type) pack_emb_type_, g.n_embd);
    const size_t tb = (size_t) g.n_vocab * rb;
    bool room = forced;
    if (want && !forced) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        size_t experts = 0;
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) experts += F->L[(size_t) il].blob * (size_t) g.n_expert;
        room = free_b > experts + tb + ((size_t) 4 << 30);
    }
    if (want && room && pack_emb_src_ != nullptr &&
        (pack_emb_type_ == GGML_TYPE_Q6_K || pack_emb_type_ == GGML_TYPE_Q8_0) && g.n_embd % 256 == 0) {
        bool ok = cudaMalloc(&F->mtp_emb, tb) == cudaSuccess &&
                  cudaMemcpy(F->mtp_emb, pack_emb_src_, tb, cudaMemcpyHostToDevice) == cudaSuccess &&
                  cudaHostAlloc((void**) &F->mtp_emb_h, (size_t) g.n_embd * sizeof(float), cudaHostAllocDefault) ==
                      cudaSuccess &&
                  cudaEventCreateWithFlags(&F->ev_mtp_emb, cudaEventDisableTiming) == cudaSuccess;
        F->mtp_emb_type = (int) pack_emb_type_;
        // the device's rows against the host's dequantization (the token path's): the same floats, or no chain
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
        int* d_i = nullptr;
        float* d_row = nullptr;
        ok = ok && tt != nullptr && tt->to_float != nullptr && cudaMalloc((void**) &d_i, sizeof(int)) == cudaSuccess &&
             cudaMalloc((void**) &d_row, (size_t) g.n_embd * sizeof(float)) == cudaSuccess;
        std::vector<float> want_r((size_t) g.n_embd), got((size_t) g.n_embd);
        for (const int i : {0, 1000, (int) g.n_vocab / 2, (int) g.n_vocab - 60}) {
            if (!ok) break;
            cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice);
            gf::embed_tok(F->mtp_emb, F->mtp_emb_type, (int) g.n_vocab, d_i, g.n_embd, d_row, nullptr);
            ok = cudaMemcpy(got.data(), d_row, got.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
            tt->to_float(pack_emb_src_ + (size_t) i * rb, want_r.data(), g.n_embd);
            ok = ok && std::memcmp(want_r.data(), got.data(), got.size() * sizeof(float)) == 0;
        }
        if (d_i) cudaFree(d_i);
        if (d_row) cudaFree(d_row);
        if (!ok) {
            cudaGetLastError();
            if (F->mtp_emb) cudaFree(F->mtp_emb);
            if (F->mtp_emb_h) cudaFreeHost(F->mtp_emb_h);
            if (F->ev_mtp_emb) cudaEventDestroy(F->ev_mtp_emb);
            F->mtp_emb = nullptr;
            F->mtp_emb_h = nullptr;
            F->ev_mtp_emb = nullptr;
            std::fprintf(stderr, "glm mtp: CUDA%d token_embd did not set up on the device - the drafts go step by step\n",
                         dev_);
        }
    }
    const int n = F->mtp_dv + F->mtp_dx;
    if (F->mtp_dv > 0)
        std::fprintf(stderr, "glm mtp: CUDA%d the MTP decode drafts over %d of %lld tokens (the ids below %d and %d "
                             "more: %.0f of %.0f MB a draft)", dev_, n, (long long) g.n_vocab, F->mtp_dv, F->mtp_dx,
                     (double) n * (double) hb / 1048576.0, (double) g.n_vocab * (double) hb / 1048576.0);
    else
        std::fprintf(stderr, "glm mtp: CUDA%d the MTP decode drafts over the whole vocabulary", dev_);
    if (F->mtp_emb)
        std::fprintf(stderr, ", chained on the device (token_embd there, %.0f MB)\n", (double) tb / 1048576.0);
    else
        std::fprintf(stderr, "\n");
}

// ---------------------------------------------------------------- the NextN block's cache
// fast_setup: the cache fill's row buffers for a whole window (kWinMax rows), before the expert pool is sized
bool Glm5Model::mtp_cache_setup(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    auto& C = F->mtpc;
    const int rows = kWinMax;
    const int64_t E = g.n_embd;
    const size_t a = 256;
    const auto up = [&](size_t b) { return (b + a - 1) / a * a; };
    const size_t f = sizeof(float);
    const size_t sz[8] = {up(rows * E * f), up(rows * E * f), up(rows * E * f), up(rows * g.kv_lora * f),
                          up(rows * g.idx_key * f), up(rows * g.idx_key * f), up(rows * q8_bytes(2 * E)),
                          up(rows * q8_bytes(E))};
    size_t total = 0;
    for (const size_t s : sz) total += s;
    if (cudaMalloc(&C.arena, total) != cudaSuccess ||
        cudaHostAlloc((void**) &C.emb_h, (size_t) rows * E * f, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        err = "the NextN cache rows' buffers did not allocate";
        mtp_cache_free();
        return false;
    }
    uint8_t* at = (uint8_t*) C.arena;
    void* p[8];
    for (int i = 0; i < 8; ++i) {
        p[i] = at;
        at += sz[i];
    }
    C.emb = (float*) p[0];
    C.hid = (float*) p[1];
    C.x = (float*) p[2];
    C.kv_raw = (float*) p[3];
    C.ik_raw = (float*) p[4];
    C.ig_raw = (float*) p[5];
    C.catq = p[6];
    C.xq = p[7];
    C.cap = rows;
    return true;
}

void Glm5Model::mtp_cache_free() {
    if (fast_ == nullptr) return;
    auto& C = fast_->mtpc;
    if (C.arena) cudaFree(C.arena);
    if (C.emb_h) cudaFreeHost(C.emb_h);
    C = FastState::MtpCacheRows{};
}

// entries p0 .. p0 + n - 1 of the draft block's DSA caches: entry p0 + i from the trunk's hidden state h + i * n_embd
// and the token next[i] (the one at p0 + i + 1; an image row there stands in for it) - the cache-writing half of the
// block (glm_prefill.cu's for a prompt): eh_proj, the attention norm, kv_a and the indexer's key and gate, written
// through KV streaming's hook where the block's cache is streamed.  This part carries the block.
bool Glm5Model::mtp_cache_rows(int64_t p0, int n, const float* h, const int32_t* next, std::string& err, bool sync,
                               float* emb_rows_h) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (F == nullptr || mtp_il_ < 0 || n <= 0) return true;
    auto& C = F->mtpc;
    if (n > C.cap) {
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
    // the pinned rows may still feed an earlier copy of this stream (sync = false: the caller knows they do not - the
    // MTP decode's round, after the verify it waited for)
    if (sync && !glmfast::cuda_ok(cudaStreamSynchronize(s), "glm mtp cache sync", err)) return false;
    const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, g.n_embd);
    float* const emb_h = emb_rows_h != nullptr ? emb_rows_h : C.emb_h;
    for (int i = 0; i < n; ++i) {
        float* row = emb_h + (size_t) i * E;
        if (const float* img = image_row(p0 + i + 1)) std::memcpy(row, img, (size_t) E * sizeof(float));
        else tt->to_float(pack_emb_src_ + (size_t) next[i] * row_b, row, g.n_embd);
    }
    cudaMemcpyAsync(C.emb, emb_h, (size_t) n * E * sizeof(float), cudaMemcpyHostToDevice, s);
    for (int i = 0; i < n; ++i)
        gf::mtp_in(C.emb + (size_t) i * E, h + (size_t) i * E, Ly.enorm, Ly.hnorm, g.norm_eps, g.n_embd,
                   (uint8_t*) C.catq + (size_t) i * q8_bytes(2 * E), s);
    gf::MvJob eh = {Ly.eh.q, C.catq, nullptr, C.hid, nullptr, 1.0f, Ly.eh.type, (int) (2 * E), g.n_embd};
    eh.ncol = n;
    eh.xq_cstride = (int64_t) q8_bytes(2 * E);
    eh.y_cstride = E;
    if (!gf::mv(&eh, 1, s)) {
        err = "glm mtp: eh_proj rows";
        return false;
    }
    for (int i = 0; i < n; ++i)
        gf::rms_q8(C.hid + (size_t) i * E, nullptr, Ly.attn_norm, g.norm_eps, g.n_embd, C.x + (size_t) i * E,
                   (uint8_t*) C.xq + (size_t) i * q8_bytes(E), s);
    gf::MvJob j[3];
    j[0] = {Ly.kv_a.q, C.xq, C.x, C.kv_raw, nullptr, 1.0f, Ly.kv_a.type, g.n_embd, g.kv_lora};
    j[1] = {Ly.idx_k, nullptr, C.x, C.ik_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[2] = {Ly.idx_gate, nullptr, C.x, C.ig_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    for (int k = 0; k < 3; ++k) {
        j[k].ncol = n;
        j[k].xq_cstride = (int64_t) q8_bytes(E);
        j[k].xf_cstride = E;
        j[k].y_cstride = k == 0 ? g.kv_lora : g.idx_key;
    }
    if (!gf::mv(j, 3, s)) {
        err = "glm mtp: the cache rows' projections";
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
        kv_prep_targets(kv_, mtp_il_, (uint16_t*) (state_ + dsa_lat_[(size_t) mtp_il_]), d);   // (KV streaming)
        d.ik_raw = C.ik_raw + (size_t) i * g.idx_key;
        d.k_norm_w = Ly.k_norm_w;
        d.k_norm_b = Ly.k_norm_b;
        d.ik_cache = state_ + dsa_ik_[(size_t) mtp_il_];
        d.ig_raw = C.ig_raw + (size_t) i * g.idx_key;
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
// state of p - 1 (head_x, when it is that one) and the token at p
bool Glm5Model::mtp_fill_prev(int32_t token, int64_t p, std::string& err) {
    if (!mtp_dec_ || fast_ == nullptr || mtp_il_ < 0 || fast_->mtpc.cap < 1 || p < 1 || mtp_hx_pos_ != p - 1)
        return true;
    return mtp_cache_rows(p - 1, 1, fast_->head_x, &token, err);
}

// ---------------------------------------------------------------- the draft length
// The length that gives the most tokens per second on this machine: each length's round time (measured wall time: the
// mean of its first rounds, then an EMA) against the tokens a round of it yields - 1 + a1 + a1 a2 + ..., a_i the share
// of rounds that reached draft i (all before it accepted) and accepted it.  STRATA_GLM_MTP_ADAPT=0: always n_cap.
//
// The search starts at STRATA_GLM_MTP_START (2) and climbs: the start runs a stint of measured rounds, then each
// neighbour of the best a short probe, until the best's neighbours are both measured and lose (Maya-S on the Radeon
// 8065S: trying every length first cost a machine's first answer 12.7 tok/s, against 21-27 after; length 2 wins
// there).  The token path (length 0) is only tried where length 1 is the best.  Then one neighbour of the best gets a
// probe now and then, the gap doubling while the best stays.
int Glm5Model::mtp_choose(int n_cap) {
    static const bool fixed = [] {
        const char* e = getenv("STRATA_GLM_MTP_ADAPT");
        return e != nullptr && std::atoi(e) == 0;
    }();
    // STRATA_GLM_MTP_FIXED=<n> (A/B and tests): every round drafts n - n = 0 is the token path with the same memory
    // layout as the drafting runs
    static const int forced_n = [] {
        const char* e = getenv("STRATA_GLM_MTP_FIXED");
        return e != nullptr ? std::max(0, std::atoi(e)) : -1;
    }();
    static const int start_n = [] {
        const char* e = getenv("STRATA_GLM_MTP_START");
        return e != nullptr ? std::max(0, std::atoi(e)) : 2;
    }();
    if (forced_n >= 0) return std::min(forced_n, n_cap);
    if (n_cap <= 0) return 0;
    if (fixed) return n_cap;
    ++mtp_round_no_;
    // a length runs in stints: a round's time holds the tiers' work left from the round before, so the first two
    // rounds after a change are not measured (decode_mtp) - a stint gives its length measured rounds
    if (mtp_stint_left_ > 0) {
        --mtp_stint_left_;
        return mtp_stint_n_;
    }
    const auto stint = [&](int n, int measured) {
        mtp_stint_n_ = n;
        mtp_stint_left_ = measured + 1;   // + 2 unmeasured, this one included
        return n;
    };
    constexpr uint64_t kTry = 6, kProbe = 2;   // the start's measured rounds, a probe's
    const int s0 = std::min(start_n, n_cap);
    if (mtp_seen_[s0] < kTry) return stint(s0, (int) (kTry - mtp_seen_[s0]));
    // the acceptance of draft i: its own count, pulled toward draft i - 1's (a few rounds of a long window say little
    // about its last draft; the share falls slowly with the depth)
    double a[9] = {};
    double prior = 0.75;
    for (int i = 1; i <= n_cap && i < 9; ++i) {
        constexpr double kPull = 4.0;
        a[i] = (mtp_hit_[i] + kPull * prior) / (mtp_reach_[i] + kPull);
        prior = a[i];
    }
    double best = -1.0;
    int bn = s0;
    for (int n = 0; n <= n_cap && n < 9; ++n) {
        if (mtp_seen_[n] < kProbe && n != s0) continue;   // (not measured yet)
        double e = 1.0, run = 1.0;
        for (int i = 1; i <= n; ++i) {
            run *= a[i];
            e += run;
        }
        const double rate = e / std::max(1e-3, mtp_ms_[n]);
        if (rate > best) {
            best = rate;
            bn = n;
        }
    }
    // the climb: the best's neighbours, the longer first, each probed until it is measured
    for (const int alt : {bn + 1, bn - 1})
        if (alt >= 0 && alt <= n_cap && alt < 9 && mtp_seen_[alt] < kProbe)
            return stint(alt, (int) (kProbe - mtp_seen_[alt]));
    // a probe of a neighbour every mtp_gap_ rounds: the gap doubles (to 1024) while the best length stays
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
        if (alt >= 0 && alt <= n_cap && alt < 9) return stint(alt, (int) kProbe);   // four rounds, two measured
    }
    return bn;
}

// ---------------------------------------------------------------- the decode
bool Glm5Model::decode_mtp(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit,
                           int64_t& produced, std::string& err) {
    produced = 0;
    if (!mtp_ready()) {
        err = "glm mtp: the MTP decode is not set up on this model";
        return false;
    }
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const int64_t E = g.n_embd;
    const int n_cap = std::min(mtp_draft_cap(), std::min(kWinMax - 1, 8));
    static const bool trace = getenv("STRATA_GLM_MTP_TRACE") != nullptr;
    // the first token: the prompt's last forward left its logits (and greedy argmax)
    int y = -1;
    if (sp.greedy && sp.penalty_last_n <= 0 && last_tok_ >= 0) {   // (penalties: picked through the sampler)
        y = last_tok_;
    } else {
        y = fast_sample(sp, err);
        if (y < 0) return false;
    }
    y = forced(y);
    sp.counter += 1;
    last_tok_ = y;
    ++produced;
    if (!emit(y) || produced >= max_new || pos_ + 1 >= max_ctx_) return true;
    int64_t p0 = pos_;   // where y goes
    mtp_same_ = 0;       // (a request's first rounds carry its prompt's tier work: not measured)
    std::vector<int32_t> win((size_t) n_cap + 1), out((size_t) n_cap + 1);
    bool ok = true;
    // the draft vocabulary's misses: every emitted token outside it moves mtp_out_ (an EMA over ~64 tokens) - above
    // kOutMax a round's drafts take the whole head (another script than the vocabulary's: a CJK answer, say, drafts as
    // before).  With ~82% of the drafts accepted a round loses ~0.9% of its tokens for each 1% missed, against the ~2%
    // the smaller head saves: 2% is about where the two meet.
    constexpr double kOutMax = 0.02;
    // the acceptance a length's choice weighs is the recent text's: each draft position's counts decay over its last
    // STRATA_GLM_MTP_ACC_WINDOW (64) comparisons (0: every comparison since the load).  A whole-session count let a
    // request's easy start speak for its hard part (Maya-S on the Radeon 8065S kept 2 drafts through a 1536-token essay,
    // 57% of the drafts accepted, 21.3 tok/s, where 1 gives ~22.5).
    static const double acc_keep = [] {
        const char* e = getenv("STRATA_GLM_MTP_ACC_WINDOW");
        const int w = e != nullptr ? std::max(0, std::atoi(e)) : 64;
        return w > 0 ? 1.0 - 1.0 / (double) w : 1.0;
    }();
    int64_t n_out = 0, n_full = 0, n_drafting = 0;
    const bool chain = F->mtp_emb != nullptr && mtp_chain_wanted();
    const auto note = [&](int tok) {
        const bool outside = !mtp_vin_.empty() && tok >= 0 && tok < (int) mtp_vin_.size() && !mtp_vin_[(size_t) tok];
        n_out += outside ? 1 : 0;
        mtp_out_ += ((outside ? 1.0 : 0.0) - mtp_out_) / 64.0;
    };
    note(y);
    // the drafting rounds' host time: drafts, verify (to the tokens), accept + commit + cache
    double ph[3] = {0, 0, 0};
    int64_t n_ph = 0;
    const auto ms_since = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    };
    for (;;) {
        const auto t0 = std::chrono::steady_clock::now();
        int n = mtp_choose(n_cap);
        n = (int) std::min<int64_t>(n, max_new - produced - 1);
        n = (int) std::min<int64_t>(n, max_ctx_ - 1 - p0);
        if (mtp_hx_pos_ != p0 - 1) n = 0;   // (no trunk state to draft from: a plain token)
        n = std::max(n, 0);
        win[0] = y;
        int nd = 0;
        if (n > 0) {
            // ---- 1. the drafts: the NextN block at p0 - 1 + i on (its last hidden state, the last token)
            // (the chain queues them behind the last round's commit and cache rows: nothing of the host's they read
            // is still in flight)
            cudaSetDevice(dev_);
            if ((!chain && !glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm mtp draft sync", err)) ||
                !F->route_ok(err)) {
                ok = false;
                break;
            }
            // the draft vocabulary while the answer stays in it, the whole head while it does not (mtp_out_)
            const bool sub = F->mtp_dv > 0 && mtp_out_ < kOutMax;
            if (chain) {
                // the chain: every step queued at once, each step's argmax the next one's token on the device; the
                // host reads the n drafts back once
                for (int i = 0; i < n && ok; ++i) ok = fast_mtp(p0 - 1 + i, i == 0 ? y : -1, err, i, sub, false);
                if (ok) {
                    cudaMemcpyAsync(F->mtp_tok_h, F->mtp_tok, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost, F->cs);
                    cudaEventRecord(F->ev_mtp, F->cs);
                    ok = glmfast::wait_event(F->ev_mtp, "glm mtp drafts", err) && F->route_ok(err);
                }
                for (int i = 0; i < n && ok; ++i) win[(size_t) i + 1] = F->mtp_tok_h[i];
                if (ok) nd = n;
            } else {
                int32_t tok = y;
                for (int i = 0; i < n; ++i) {
                    if (!fast_mtp(p0 - 1 + i, tok, err, i, sub) ||
                        !glmfast::wait_event(F->ev_mtp, "glm mtp draft", err) || !F->route_ok(err)) {
                        ok = false;
                        break;
                    }
                    tok = F->mtp_tok_h[i];
                    win[(size_t) i + 1] = tok;
                    ++nd;
                }
            }
            n_full += sub ? 0 : 1;
            ++n_drafting;
            mtp_hx_pos_ = -1;   // (head_x holds the block's hidden state now)
            if (!ok) break;
        }
        const int T = 1 + nd;
        int keep = 1, acc = 0;
        bool more = true;
        const auto t1 = std::chrono::steady_clock::now();
        if (T == 1) {
            // ---- a plain token (the token path keeps the NextN cache: entry p0 - 1)
            if (!fast_token(y, err)) {
                ok = false;
                break;
            }
            int y2 = -1;
            if (sp.greedy && sp.penalty_last_n <= 0) {
                y2 = last_tok_;
            } else {
                y2 = fast_sample(sp, err);
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
            note(y2);
            more = emit(y2) && produced < max_new && p0 + 2 < max_ctx_;
        } else {
            // ---- 2. the verify: the window through every layer, every row's pick
            if (!window(win.data(), T, out.data(), err, &sp)) {
                ok = false;
                break;
            }
            const auto t2 = std::chrono::steady_clock::now();
            ph[0] += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ph[1] += std::chrono::duration<double, std::milli>(t2 - t1).count();
            // ---- 3. accept: row t's pick is the truth for p0 + t + 1; draft t + 1 stands while they match
            for (int t = 0; t < T; ++t) {
                const int yy = forced(out[(size_t) t]);
                sp.counter += 1;
                last_tok_ = yy;
                ++produced;
                keep = t + 1;
                y = yy;
                note(yy);
                more = emit(yy) && produced < max_new && p0 + keep + 1 < max_ctx_;
                if (!more || t + 1 >= T) break;
                // draft t + 1 is compared (every draft before it stood): its counts decay first (acc_keep)
                mtp_hit_[t + 1] *= acc_keep;
                mtp_reach_[t + 1] = mtp_reach_[t + 1] * acc_keep + 1.0;
                if (yy != win[(size_t) t + 1]) break;
                mtp_hit_[t + 1] += 1.0;
                ++acc;
            }
            // ---- 4. the commit: the recurrent states advanced by the kept rows (positions p0 .. p0 + keep - 1)
            if (!window_commit(keep, err)) {
                ok = false;
                break;
            }
            // ---- 5. the NextN cache: entries p0 .. p0 + keep - 2 from the kept rows' trunk states and the accepted
            //         drafts; entry p0 + keep - 1 is the next round's first draft step (or the token path's)
            const float* rows_hx = F->win.head_x;
            if (keep > 1 && !mtp_cache_rows(p0, keep - 1, rows_hx, win.data() + 1, err, !chain)) {
                ok = false;
                break;
            }
            cudaSetDevice(dev_);
            cudaMemcpyAsync(F->head_x, rows_hx + (size_t) (keep - 1) * E, (size_t) E * sizeof(float),
                            cudaMemcpyDeviceToDevice, F->cs);
            mtp_hx_pos_ = p0 + keep - 1;
            ph[2] += ms_since(t2);
            ++n_ph;
        }
        // the statistics and the adaptive length's measurements
        ++mtp_rounds_;
        ++mtp_round_n_[std::min(nd, 8)];
        spec_steps_ += (uint64_t) nd;
        spec_hits_ += (uint64_t) acc;
        const double ms = ms_since(t0);
        // measured from the third round of a run of one length on (mtp_choose's stints)
        if (nd == mtp_prev_n_) {
            ++mtp_same_;
        } else {
            mtp_prev_n_ = nd;
            mtp_same_ = 0;
        }
        if (mtp_same_ >= 2 && nd < 9) {
            // (the mean of the first ten, then an EMA: one slow first round does not stand for the length)
            mtp_ms_[nd] += (ms - mtp_ms_[nd]) / (double) std::min<uint64_t>(mtp_seen_[nd] + 1, 10);
            ++mtp_seen_[nd];
        }
        if (trace)
            std::fprintf(stderr, "glm mtp: p0 %lld drafts %d kept %d accepted %d (%.1f ms)\n", (long long) p0, nd, keep,
                         acc, ms);
        p0 += keep;
        if (!more) break;
    }
    // drain: the device idle, the routes checked
    {
        cudaSetDevice(dev_);
        std::string drain_err;
        if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm mtp drain", drain_err) || !F->route_ok(drain_err)) {
            if (ok) err = drain_err;
            ok = false;
        }
    }
    win_prof_report();   // (STRATA_GLM_WIN_PROF: the verify windows' phases)
    if (n_ph > 0)
        std::fprintf(stderr, "glm mtp: a drafting round's host time - drafts %.2f ms, verify %.2f ms, commit + cache "
                             "%.2f ms (%lld rounds)\n", ph[0] / n_ph, ph[1] / n_ph, ph[2] / n_ph, (long long) n_ph);
    if (F->mtp_dv > 0 && produced > 0)
        std::fprintf(stderr, "glm mtp: the draft vocabulary (%d + %d tokens) - %lld of %lld emitted tokens outside it, "
                             "%lld of %lld drafting rounds on the whole head\n", F->mtp_dv, F->mtp_dx,
                     (long long) n_out, (long long) produced, (long long) n_full, (long long) n_drafting);
    if (ok && gf::launch_errors() > 0) {
        err = "glm mtp: a kernel launch failed (see stderr)";
        ok = false;
    }
    return ok;
}

}  // namespace strata::core
