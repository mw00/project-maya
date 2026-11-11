// src/core/glm_dsa_mixer.hpp - the glm-dsa decode attention mixer's orchestration (glm-dsa B4.6, B14.3, B14.4).
//
// One DSA layer of one decode token at position p, from the attention input x (attn_norm output, f32 + q8_1) to the
// mixer output (attn_output's), as a backend-independent template: the GPU path (Glm5Model::fast_dsa_glm,
// src/core/glm_dsa_mixer.cu) instantiates it with gf::mv + the glmd kernels, the CPU test
// (tests/glm_dsa/glm_dsa_mixer_cpu_test.cpp) with float GEMVs + the glmd::twin kernels, so the test runs the very
// sequence of calls and the very struct filling the engine runs.  Plain C++, no CUDA.
//
// Per layer (GLM-5.3 sizes):
//   1. GEMV batch   full:   {q_a x -> qr_raw [2048], kv_a x -> kv_raw [576], idx_k x -> ik_raw [128],
//                            idx_proj x -> iw [32] x 1/sqrt(idx_heads * idx_key)}
//                   shared: {q_a, kv_a}
//   2. dsa_prep     qr (f32 + q8_1), latent[p], k_pe[p]; full: index key[p]       (writes through the state layout)
//   3. GEMV batch   {q_b qr -> q [64 x 256]}, full: + {idx_q_b qr -> iq [32 x 128]}
//   4. dsa_rope_q   q_pe of every head at p; full: iq dims 0..idx_rope (B14.11: then each iq head Hadamard
//                   rotated when the layout's keys are, MixerCache::ik_had: the scores are unchanged)
//   5. selection    full: idx_score over cells 0..p (n = p + 1, skipped while n <= top_k) and topk (identity while
//                   n <= top_k, -1 padded) into `cells`, and the cell count min(n, top_k) on the device.
//                   shared: nothing - `cells` and the count of the previous full layer (same token, same stream,
//                   layers in order) are reused; the SelState check refuses a stale selection.
//                   MAYA_GLM_DSA_DENSE=1 (B14.3): every cell 0..p from a static iota buffer, no score / topk.
//   6. mla2         over the selected cells, scale 1/16 at GLM-5.3; f32 and q8_1 outputs
//   7. GEMV         attn_output -> mixer [6144]
//
// CUDA-graph replay safety (B14.4): every position the kernels use (cache writes at p, the rope angle, the score
// count, the top-k count, the attention's cell count) is read from the device word p_dev; the attention runs at the
// fixed capacity top_k with the device count n_sel_dev, and the scorer's grid is fixed by the cache capacity.  The
// only host-dependent launch is the dense switch's attention (n_sel = p + 1): a diagnostic mode, documented as not
// graph-safe.
//
// glm-dsa wave 5: the Volta kernels behind the decode policy (include/strata/kernels/glmd_decode_policy.hpp, the ONE
// dispatch table): MixerDims::pol says which scorer / selector / attention kernel the backend runs (the GPU backend
// calls glmd::decode_score / decode_topk / decode_mla, the twin backend their glmd::twin versions), MixerCache::ik_f16
// carries the state layout's index-key dtype (B10.8) into dsa_prep and the scorer, MixerLayer::w_type the binder's
// W_kb / W_vb type (B10.7: BF16 or native Q8_0), MixerBufs::n_grid a decode graph's fixed scorer / selector extent
// and n_sel_preset a device cell count the caller already wrote (glmd::DevPos, B10.1 / B14.4).
#pragma once

#include "strata/kernels/glmd_decode_policy.hpp"
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_mixer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace strata::core::glmdsa {

/// The geometry the mixer reads (from Glm5Geometry; the CPU test fills it directly).
struct MixerDims {
    int n_embd = 0, n_head = 0, q_lora = 0, kv_lora = 0, qk_nope = 0, qk_rope = 0, v_head = 0;
    int idx_heads = 0, idx_key = 0, idx_rope = 0, top_k = 2048;
    int64_t max_ctx = 0;
    float eps = 1e-5f, idx_ln_eps = 1e-6f, kq_scale = 0.0625f;
    glmd::DecodePolicy pol;         // the kernel dispatch (default: the wave-4 kernels, all switches off)
    int qk_head() const { return qk_nope + qk_rope; }
    int kv_row() const { return kv_lora + qk_rope; }
    float idx_w_scale() const { return 1.0f / std::sqrt((float) (idx_heads * idx_key)); }
};

/// The projections the mixer runs (the backend maps each to its weight slot).
enum class Proj : int { QA = 0, KVA, IdxK, IdxProj, QB, IdxQB, Out };
inline const char* proj_name(Proj w);

/// One GEMV of a batch: y = alpha * W x, with the float input xf and / or its q8_1 xq (a backend uses what the
/// weight's type needs).
struct Gemv {
    Proj w = Proj::QA;
    const float* xf = nullptr;
    const void* xq = nullptr;
    float* y = nullptr;
    float alpha = 1.0f;
    int n_in = 0, n_out = 0;
    // L1-4 (mixer_rows): ncol activations, each its column stride after the one before (xf / y in floats, xq in
    // bytes); mixer_layer leaves the one column
    int ncol = 1;
    int64_t xf_cs = 0, xq_cs = 0, y_cs = 0;
};

/// The buffers of one decode token (device pointers on the GPU, host pointers on the CPU).
struct MixerBufs {
    const float* x = nullptr;       // n_embd: the attention input (attn_norm output)
    const void* xq = nullptr;       // its q8_1
    float* qr_raw = nullptr;        // q_lora
    float* qr = nullptr;            // q_lora
    void* qr_q = nullptr;           // q_lora q8_1
    float* kv_raw = nullptr;        // kv_row
    float* ik_raw = nullptr;        // idx_key
    float* iw = nullptr;            // idx_heads
    float* q = nullptr;             // n_head * qk_head
    float* iq = nullptr;            // idx_heads * idx_key
    float* score = nullptr;         // max_ctx
    int* cells = nullptr;           // top_k
    const int* cells_dense = nullptr;   // max_ctx, cells_dense[i] = i (dense switch only)
    int* n_sel_dev = nullptr;       // one int: the attention's cell count (device)
    float* attn_out = nullptr;      // n_head * v_head
    void* attn_out_q = nullptr;     // its q8_1
    float* mixer = nullptr;         // n_embd: the output
    void* work = nullptr;           // glmd::mla2 scratch
    size_t work_bytes = 0;
    const float* inv_freq = nullptr;      // qk_rope / 2
    const float* idx_inv_freq = nullptr;  // idx_rope / 2
    const int* p_dev = nullptr;     // the position word (device; a host int on the CPU)
    int n_grid = 0;                 // a decode graph's scorer / selector extent (glmd::graph_n_cap); 0: p + 1
    bool n_sel_preset = false;      // *n_sel_dev is written by the caller for this token (glmd::DevPos::n_sel):
                                    // the scorer does not write it
};

/// One layer's caches at the state layout's offsets (GlmStateLayout dsa_lat / dsa_kpe / dsa_ik).
struct MixerCache {
    void* lat = nullptr;            // FP16 rows of kv_lora or INT8 records
    int lat_q8 = 0;
    uint16_t* k_pe = nullptr;       // FP16 [max_ctx][qk_rope]
    float* ik = nullptr;            // F32 [max_ctx][idx_key] (FP16 rows cast when ik_f16); null on shared layers
    int ik_f16 = 0;                 // B10.8: the state layout's key dtype (GlmStateLayout::ik_bytes == 2)
    int ik_had = 0;                 // B14.11: rotated keys and queries (GlmStateLayout::ik_had)
    // L1-6 KV streaming (Glm5Model::kv_stream_make; null kv_resolve: lat / k_pe are the identity caches): lat / k_pe
    // are the layer's VRAM slots, the prep writes the host copies too (DsaPrep lat_host ...), and before the attention
    // kv_resolve(kv_ctx, cells, cap, n_sel_dev) makes the selection resident and returns it in slot rows (null: failed)
    void* lat_host = nullptr;
    uint16_t* kpe_host = nullptr;
    const int* kv_table = nullptr;
    int kv_page = 1;
    const int* (*kv_resolve)(void* ctx, const int* cells, int n_cap, const int* n_sel_dev) = nullptr;
    void* kv_ctx = nullptr;
};

/// One layer's small weights (norms) and the BF16 k_b / v_b glmd::mla2 reads.
struct MixerLayer {
    bool full = false;              // runs its own indexer (Glm5Geometry::idx_full; the NextN block is full)
    int idx_src = -1;               // the layer whose selection a shared layer reuses
    const float* q_a_norm = nullptr;
    const float* kv_a_norm = nullptr;
    const float* ik_ln_w = nullptr;
    const float* ik_ln_b = nullptr;
    const uint16_t* wk_b = nullptr;   // BF16, or Q8_0 blocks (cast) when w_type == glmd::kWTypeQ8_0
    const uint16_t* wv_b = nullptr;
    int w_type = glmd::kWTypeBF16;    // B10.7: Mla2 / FusedMla w_type of both
};

/// The selection the shared layers reuse: which full layer made it, at which position, and its host-side count.
struct SelState {
    int layer = -1;
    int64_t pos = -1;
    int n_sel_last = 0;             // min(p + 1, top_k), or p + 1 under the dense switch (recorded per full layer)
    bool dense = false;
};

/// B11.2: forget a selection (a rejected speculative position is about to be redone): every shared layer must then
/// find its full layer's selection made again at the redo, never the rejected run's.
inline void sel_rollback(SelState& st) { st = SelState{}; }

/// The cells the attention reads at position p (host arithmetic; the kernels compute the same on the device).
inline int sel_count(int64_t p, int top_k, bool dense) {
    const int64_t n = p + 1;
    return (int) (dense || n <= (int64_t) top_k ? n : (int64_t) top_k);
}

/// One layer of the mixer through backend `o` (see the file comment for the calls).  Ops provides:
///   bool gemv(int il, const Gemv* jobs, int n);
///   bool prep(const glmd::DsaPrep&);  bool rope_q(const glmd::DsaRopeQ&);  bool score(const glmd::IdxScore&);
///   bool topk(const float* score, int ld, int n_kv, const int* pos, int top_k, int* sel);
///   bool mla2(const glmd::Mla2&);
/// The selection is recorded in / checked against st.  false + err on any refusal.
template <class Ops>
bool mixer_layer(Ops& o, const MixerDims& d, const MixerBufs& b, const MixerCache& c, const MixerLayer& w, int il,
                 int64_t p, bool dense, SelState& st, std::string& err) {
    const auto fail = [&](const char* what) {
        err = "glm-dsa mixer: layer " + std::to_string(il) + " at position " + std::to_string(p) + ": " + what;
        return false;
    };
    if (p < 0 || p >= d.max_ctx) return fail("position outside the context");
    const int E = d.n_embd, QH = d.qk_head();
    // 1. the input projections
    {
        Gemv j[4];
        int n = 0;
        j[n++] = {Proj::QA, b.x, b.xq, b.qr_raw, 1.0f, E, d.q_lora};
        j[n++] = {Proj::KVA, b.x, b.xq, b.kv_raw, 1.0f, E, d.kv_row()};
        if (w.full) {
            j[n++] = {Proj::IdxK, b.x, b.xq, b.ik_raw, 1.0f, E, d.idx_key};
            j[n++] = {Proj::IdxProj, b.x, b.xq, b.iw, d.idx_w_scale(), E, d.idx_heads};
        }
        if (!o.gemv(il, j, n)) return fail("input projections");
    }
    // 2. prep: qr, the latent / k_pe / index key at p
    {
        glmd::DsaPrep a;
        a.qr_raw = b.qr_raw;
        a.q_a_norm = w.q_a_norm;
        a.qr = b.qr;
        a.qr_q = b.qr_q;
        a.q_lora = d.q_lora;
        a.kv_raw = b.kv_raw;
        a.kv_norm = w.kv_a_norm;
        a.lat = c.lat;
        a.lat_q8 = c.lat_q8;
        a.kv_lora = d.kv_lora;
        a.k_pe = c.k_pe;
        a.lat_host = c.lat_host;   // L1-6 KV streaming (null: identity caches)
        a.kpe_host = c.kpe_host;
        a.kv_table = c.kv_table;
        a.kv_page = c.kv_page;
        a.qk_rope = d.qk_rope;
        a.inv_freq = b.inv_freq;
        if (w.full) {
            if (c.ik == nullptr) return fail("a full layer without an index-key cache");
            a.ik_raw = b.ik_raw;
            a.ik_ln_w = w.ik_ln_w;
            a.ik_ln_b = w.ik_ln_b;
            a.idx_ln_eps = d.idx_ln_eps;
            a.ik_cache = c.ik;
            a.ik_f16 = c.ik_f16;
            a.ik_had = c.ik_had;
            a.idx_dim = d.idx_key;
            a.idx_rope = d.idx_rope;
            a.idx_inv_freq = b.idx_inv_freq;
        }
        a.p = (int) p;
        a.p_dev = b.p_dev;
        a.eps = d.eps;
        if (!o.prep(a)) return fail("dsa_prep");
    }
    // 3. the query projections
    {
        Gemv j[2];
        int n = 0;
        j[n++] = {Proj::QB, b.qr, b.qr_q, b.q, 1.0f, d.q_lora, d.n_head * QH};
        if (w.full) j[n++] = {Proj::IdxQB, b.qr, b.qr_q, b.iq, 1.0f, d.q_lora, d.idx_heads * d.idx_key};
        if (!o.gemv(il, j, n)) return fail("query projections");
    }
    // 4. rope the query tails at p
    {
        glmd::DsaRopeQ r;
        r.q = b.q;
        r.n_head = d.n_head;
        r.q_stride = QH;
        r.rot_off = d.qk_nope;
        r.n_rot = d.qk_rope;
        r.inv_freq = b.inv_freq;
        if (w.full) {
            r.iq = b.iq;
            r.idx_heads = d.idx_heads;
            r.idx_dim = d.idx_key;
            r.idx_rope = d.idx_rope;
            r.idx_inv_freq = b.idx_inv_freq;
            r.iq_had = c.ik_had;
        }
        r.p = (int) p;
        r.p_dev = b.p_dev;
        if (!o.rope_q(r)) return fail("dsa_rope_q");
    }
    // 5. the selection
    if (w.full) {
        if (!dense) {
            glmd::IdxScore s;
            s.iq = b.iq;
            s.ik_cache = c.ik;
            s.w = b.iw;
            s.idx_heads = d.idx_heads;
            s.idx_dim = d.idx_key;
            s.n_cap = (int) d.max_ctx;
            s.p = (int) p;
            s.p_dev = b.p_dev;
            s.score = b.score;
            s.skip_le = d.top_k;
            s.n_sel_out = b.n_sel_preset ? nullptr : b.n_sel_dev;
            s.sel_cap = d.top_k;
            s.key_f16 = c.ik_f16;
            s.n_grid = b.n_grid;
            if (!o.score(s)) return fail("idx_score");
            // the selector's extent: the graph capacity, else the live cells (it reads the position on the device)
            const int n_ext = (int) std::min<int64_t>(d.max_ctx, b.n_grid > 0 ? (int64_t) b.n_grid : p + 1);
            if (!o.topk(b.score, (int) d.max_ctx, n_ext, b.p_dev, d.top_k, b.cells)) return fail("topk");
        }
        st.layer = il;
        st.pos = p;
        st.dense = dense;
        st.n_sel_last = sel_count(p, d.top_k, dense);
    } else if (st.layer != w.idx_src || st.pos != p || st.dense != dense) {
        return fail(("a shared layer whose selection (layer " + std::to_string(w.idx_src) +
                     ") was not made at this position on this stream")
                        .c_str());
    }
    // 6. the attention over the selected cells
    {
        glmd::Mla2 m;
        m.q = b.q;
        m.q_stride = QH;
        m.wk_b = w.wk_b;
        m.wv_b = w.wv_b;
        m.lat = c.lat;
        m.lat_q8 = c.lat_q8;
        m.k_pe = c.k_pe;
        if (dense) {
            m.cells = b.cells_dense;
            m.n_sel = (int) (p + 1);      // host count: the dense diagnostic is not graph-replay safe
        } else {
            m.cells = b.cells;
            m.n_sel = d.top_k;            // capacity; the device count says how many are live
            m.n_sel_dev = b.n_sel_dev;
        }
        if (c.kv_resolve != nullptr) {    // L1-6 KV streaming: the selection in slot rows
            if (dense) return fail("the dense switch with KV streaming");
            m.cells = c.kv_resolve(c.kv_ctx, m.cells, m.n_sel, m.n_sel_dev);
            if (m.cells == nullptr) return fail("KV streaming resolve");
        }
        m.n_head = d.n_head;
        m.qk_nope = d.qk_nope;
        m.qk_rope = d.qk_rope;
        m.kv_lora = d.kv_lora;
        m.v_head = d.v_head;
        m.scale = d.kq_scale;
        m.work = b.work;
        m.work_bytes = b.work_bytes;
        m.out = b.attn_out;
        m.out_q8_1 = b.attn_out_q;
        m.w_type = w.w_type;
        if (!o.mla2(m)) return fail("mla2");
    }
    // 7. attn_output
    {
        Gemv j = {Proj::Out, b.attn_out, b.attn_out_q, b.mixer, 1.0f, d.n_head * d.v_head, E};
        if (!o.gemv(il, &j, 1)) return fail("attn_output");
    }
    return true;
}

/// L1-4: the rows of a verify window (the one-GPU MTP decode, glm_window.cu): rows r = 0 .. n - 1 of tokens at
/// positions p0 + r.  mixer_layer's steps with its three GEMV batches run once for all rows (ncol = n: each column's
/// sums are the one-column GEMV's bit for bit) and the per-token steps (prep, rope, selection, attention) row by row in
/// order - row r's prep writes its caches before row r + 1 attends - so every row computes what mixer_layer computes
/// for its token (Ops also provides set_pos(p): the position the dispatch policy reads for the next score / topk).
/// b holds row 0's pointers, RowStrides each buffer's step to the next row; the selection of row r
/// goes to cells[r] with the cell count n_sel_dev[r] and the position word p_dev[r] (the caller wrote both), and its
/// record to st[r] (a shared layer reuses its row's full layer's).
struct RowStrides {
    int64_t x = 0, qr_raw = 0, qr = 0, kv_raw = 0, ik_raw = 0, iw = 0, q = 0, iq = 0, attn_out = 0, mixer = 0;   // floats
    int64_t xq = 0, qr_q = 0, attn_out_q = 0;                                                                 // bytes
};
template <class Ops>
bool mixer_rows(Ops& o, const MixerDims& d, const MixerBufs& b, const RowStrides& rs, int n, int* const* cells,
                int* const* n_sel_dev, const int* const* p_dev, const MixerCache& c, const MixerLayer& w, int il,
                int64_t p0, bool dense, SelState* st, std::string& err) {
    const auto fail = [&](const char* what) {
        err = "glm-dsa mixer rows: layer " + std::to_string(il) + " at position " + std::to_string(p0) + ": " + what;
        return false;
    };
    if (n < 1 || p0 < 0 || p0 + n > d.max_ctx) return fail("rows outside the context");
    const int E = d.n_embd, QH = d.qk_head();
    const auto cols = [&](Gemv j, int64_t xf_cs, int64_t xq_cs, int64_t y_cs) {
        j.ncol = n;
        j.xf_cs = xf_cs;
        j.xq_cs = xq_cs;
        j.y_cs = y_cs;
        return j;
    };
    const auto at = [](float* p, int64_t step, int r) { return p == nullptr ? p : p + step * r; };
    const auto atb = [](void* p, int64_t bytes, int r) { return p == nullptr ? p : (void*) ((uint8_t*) p + bytes * r); };
    // 1. the input projections of every row
    {
        Gemv j[4];
        int m = 0;
        j[m++] = cols({Proj::QA, b.x, b.xq, b.qr_raw, 1.0f, E, d.q_lora}, rs.x, rs.xq, rs.qr_raw);
        j[m++] = cols({Proj::KVA, b.x, b.xq, b.kv_raw, 1.0f, E, d.kv_row()}, rs.x, rs.xq, rs.kv_raw);
        if (w.full) {
            j[m++] = cols({Proj::IdxK, b.x, b.xq, b.ik_raw, 1.0f, E, d.idx_key}, rs.x, rs.xq, rs.ik_raw);
            j[m++] = cols({Proj::IdxProj, b.x, b.xq, b.iw, d.idx_w_scale(), E, d.idx_heads}, rs.x, rs.xq, rs.iw);
        }
        if (!o.gemv(il, j, m)) return fail("input projections");
    }
    // 2. prep, row by row: qr, the latent / k_pe / index key at p0 + r
    for (int r = 0; r < n; ++r) {
        glmd::DsaPrep a;
        a.qr_raw = at(b.qr_raw, rs.qr_raw, r);
        a.q_a_norm = w.q_a_norm;
        a.qr = at(b.qr, rs.qr, r);
        a.qr_q = atb(b.qr_q, rs.qr_q, r);
        a.q_lora = d.q_lora;
        a.kv_raw = at(b.kv_raw, rs.kv_raw, r);
        a.kv_norm = w.kv_a_norm;
        a.lat = c.lat;
        a.lat_q8 = c.lat_q8;
        a.kv_lora = d.kv_lora;
        a.k_pe = c.k_pe;
        a.lat_host = c.lat_host;   // L1-6 KV streaming (null: identity caches)
        a.kpe_host = c.kpe_host;
        a.kv_table = c.kv_table;
        a.kv_page = c.kv_page;
        a.qk_rope = d.qk_rope;
        a.inv_freq = b.inv_freq;
        if (w.full) {
            if (c.ik == nullptr) return fail("a full layer without an index-key cache");
            a.ik_raw = at(b.ik_raw, rs.ik_raw, r);
            a.ik_ln_w = w.ik_ln_w;
            a.ik_ln_b = w.ik_ln_b;
            a.idx_ln_eps = d.idx_ln_eps;
            a.ik_cache = c.ik;
            a.ik_f16 = c.ik_f16;
            a.ik_had = c.ik_had;
            a.idx_dim = d.idx_key;
            a.idx_rope = d.idx_rope;
            a.idx_inv_freq = b.idx_inv_freq;
        }
        a.p = (int) (p0 + r);
        a.p_dev = p_dev[r];
        a.eps = d.eps;
        if (!o.prep(a)) return fail("dsa_prep");
    }
    // 3. the query projections of every row
    {
        Gemv j[2];
        int m = 0;
        j[m++] = cols({Proj::QB, b.qr, b.qr_q, b.q, 1.0f, d.q_lora, d.n_head * QH}, rs.qr, rs.qr_q, rs.q);
        if (w.full)
            j[m++] = cols({Proj::IdxQB, b.qr, b.qr_q, b.iq, 1.0f, d.q_lora, d.idx_heads * d.idx_key}, rs.qr, rs.qr_q, rs.iq);
        if (!o.gemv(il, j, m)) return fail("query projections");
    }
    // 4.-6. row by row: rope, the selection, the attention
    for (int r = 0; r < n; ++r) {
        const int64_t p = p0 + r;
        float* q = at(b.q, rs.q, r);
        float* iq = at(b.iq, rs.iq, r);
        {
            glmd::DsaRopeQ x;
            x.q = q;
            x.n_head = d.n_head;
            x.q_stride = QH;
            x.rot_off = d.qk_nope;
            x.n_rot = d.qk_rope;
            x.inv_freq = b.inv_freq;
            if (w.full) {
                x.iq = iq;
                x.idx_heads = d.idx_heads;
                x.idx_dim = d.idx_key;
                x.idx_rope = d.idx_rope;
                x.idx_inv_freq = b.idx_inv_freq;
                x.iq_had = c.ik_had;
            }
            x.p = (int) p;
            x.p_dev = p_dev[r];
            if (!o.rope_q(x)) return fail("dsa_rope_q");
        }
        if (w.full) {
            if (!dense) {
                o.set_pos(p);   // (the dispatch policy picks the scorer / selector by the row's own position)
                glmd::IdxScore s;
                s.iq = iq;
                s.ik_cache = c.ik;
                s.w = at(b.iw, rs.iw, r);
                s.idx_heads = d.idx_heads;
                s.idx_dim = d.idx_key;
                s.n_cap = (int) d.max_ctx;
                s.p = (int) p;
                s.p_dev = p_dev[r];
                s.score = b.score;
                s.skip_le = d.top_k;
                s.n_sel_out = nullptr;   // (the caller wrote the row's count with its position)
                s.sel_cap = d.top_k;
                s.key_f16 = c.ik_f16;
                s.n_grid = 0;
                if (!o.score(s)) return fail("idx_score");
                const int n_ext = (int) std::min<int64_t>(d.max_ctx, p + 1);
                if (!o.topk(b.score, (int) d.max_ctx, n_ext, p_dev[r], d.top_k, cells[r])) return fail("topk");
            }
            st[r].layer = il;
            st[r].pos = p;
            st[r].dense = dense;
            st[r].n_sel_last = sel_count(p, d.top_k, dense);
        } else if (st[r].layer != w.idx_src || st[r].pos != p || st[r].dense != dense) {
            return fail("a shared layer whose row selection was not made at this position");
        }
        glmd::Mla2 m;
        m.q = q;
        m.q_stride = QH;
        m.wk_b = w.wk_b;
        m.wv_b = w.wv_b;
        m.lat = c.lat;
        m.lat_q8 = c.lat_q8;
        m.k_pe = c.k_pe;
        if (dense) {
            m.cells = b.cells_dense;
            m.n_sel = (int) (p + 1);
        } else {
            m.cells = cells[r];
            m.n_sel = d.top_k;
            m.n_sel_dev = n_sel_dev[r];
        }
        if (c.kv_resolve != nullptr) {    // L1-6 KV streaming: row r's selection in slot rows, right before its attention
            if (dense) return fail("the dense switch with KV streaming");
            m.cells = c.kv_resolve(c.kv_ctx, m.cells, m.n_sel, m.n_sel_dev);
            if (m.cells == nullptr) return fail("KV streaming resolve");
        }
        m.n_head = d.n_head;
        m.qk_nope = d.qk_nope;
        m.qk_rope = d.qk_rope;
        m.kv_lora = d.kv_lora;
        m.v_head = d.v_head;
        m.scale = d.kq_scale;
        m.work = b.work;
        m.work_bytes = b.work_bytes;
        m.out = at(b.attn_out, rs.attn_out, r);
        m.out_q8_1 = atb(b.attn_out_q, rs.attn_out_q, r);
        m.w_type = w.w_type;
        if (!o.mla2(m)) return fail("mla2");
    }
    // 7. attn_output of every row
    {
        Gemv j = cols({Proj::Out, b.attn_out, b.attn_out_q, b.mixer, 1.0f, d.n_head * d.v_head, E}, rs.attn_out,
                      rs.attn_out_q, rs.mixer);
        if (!o.gemv(il, &j, 1)) return fail("attn_output");
    }
    return true;
}

inline const char* proj_name(Proj w) {
    switch (w) {
    case Proj::QA: return "attn_q_a";
    case Proj::KVA: return "attn_kv_a_mqa";
    case Proj::IdxK: return "indexer.attn_k";
    case Proj::IdxProj: return "indexer.proj";
    case Proj::QB: return "attn_q_b";
    case Proj::IdxQB: return "indexer.attn_q_b";
    case Proj::Out: return "attn_output";
    }
    return "?";
}

/// The attention scratch the mixer needs (dense: the whole context; the fused kernel's when the policy runs it).
inline size_t mixer_work_bytes(const MixerDims& d, bool dense) {
    return glmd::decode_mla_work_bytes(d.pol, d.n_head, d.kv_lora, d.qk_rope, dense ? (int) d.max_ctx : d.top_k);
}

}  // namespace strata::core::glmdsa
