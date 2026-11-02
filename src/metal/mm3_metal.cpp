// src/metal/mm3_metal.cpp - Mm3Metal: the forward graph, following tools/mm3/ref.py op for op.
//
// One slice of T tokens runs one graph per layer:
//   graph 1   the attention projections of ref.py's attention(): q/k/v and the indexer's iq/ik, each
//             Gemma-normed and NeoX-roped, the keys and values written into the layer's BF16 caches, iq
//             kept (BF16) in a carry, and the indexer scores iq . ik over every cached position;
//   (in-graph) the kept blocks, exactly ref.py's msa_select, built by Builder::select from those
//             scores without leaving the graph: a block scores the max of its tokens' scores, the
//             query's own local blocks are forced in, the top blocks are kept - no host round-trip
//             per layer (the CUDA engine selects on-GPU too);
//   graph 2   ref.py's msa_attend per KV group (scores, the kept-block mask, softmax, the value
//             product), the output projection, then the MLP: dense swigluoai, or the MoE - sigmoid
//             router with the correction bias and renormalised weights in-graph, the shared expert, and
//             the routed experts (all resident, so mul_mat_id consumes the router's top-k directly:
//             no pool, no graph cut at the router, unlike the GLM engine).
// The last layer's graph also runs the head over the slice's last row.  Geometries the in-graph
// selection does not cover (see gpu_select_ok) keep the older two-graph path with the host's
// select_masks, as does MAYA_MM3_HOST_SELECT=1.
#include "mm3_metal_impl.hpp"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>

namespace maya::metal {

namespace {

constexpr int kGraphNodes = 32768;
constexpr int kFloatGroup = 16;   // tokens per mul_mat_id of float experts (as the GLM engine's mm_id)

struct Upload {
    ggml_tensor* t;
    std::vector<uint8_t> data;
};

// One graph under construction (the GLM engine's Builder, trimmed to what this model needs).
struct Builder {
    Mm3Metal::Impl& m;
    const Mm3Geometry& g;
    ggml_context* ctx;
    ggml_cgraph* gf;
    int T;          // tokens in the slice
    int64_t p0;     // position of the slice's first token
    std::vector<Upload> uploads;
    int logits_rows = 1;              // graph2 (last layer): logits for this many rows (a verify window)
    ggml_tensor** stream_out = nullptr;   // graph2: also hand back the layer's output stream (a draft tap)
    ggml_tensor* pos_i = nullptr;         // graph1's position input, kept for select()'s gathers

    ggml_tensor* input(ggml_type type, int64_t ne0, int64_t ne1, const void* data, size_t bytes,
                       int64_t ne2 = 1) {
        ggml_tensor* t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        ggml_set_input(t);
        uploads.push_back({t, std::vector<uint8_t>((const uint8_t*) data, (const uint8_t*) data + bytes)});
        return t;
    }
    void expand(ggml_tensor* t) { ggml_build_forward_expand(gf, t); }

    // the slice's columns of a carry [ne0, n_ubatch]
    ggml_tensor* cols(ggml_tensor* t) { return ggml_view_2d(ctx, t, t->ne[0], T, t->nb[1], 0); }
    ggml_tensor* cols3(ggml_tensor* t) { return ggml_view_3d(ctx, t, t->ne[0], t->ne[1], T, t->nb[1], t->nb[2], 0); }

    // y = w x, as the GLM engine's mm: a float weight runs as one f32 matvec per column (ggml-metal's
    // mul_mm rounds its tiles to half), a quantized weight keeps mul_mm
    static bool is_float(const ggml_tensor* w) {
        return w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16;
    }
    ggml_tensor* mm(ggml_tensor* w, ggml_tensor* x) {
        if (!is_float(w) || x->ne[1] == 1) return ggml_mul_mat(ctx, w, x);
        GGML_ASSERT(x->ne[3] == 1 && w->ne[3] == 1);
        if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
        const int64_t n = x->ne[1], h = x->ne[2];
        ggml_tensor* x4 = ggml_reshape_4d(ctx, x, x->ne[0], 1, n, h);
        ggml_tensor* w4 = w;
        if (w->ne[2] > 1) {
            GGML_ASSERT(w->ne[2] == h);
            w4 = ggml_view_4d(ctx, w, w->ne[0], w->ne[1], 1, h, w->nb[1], w->nb[2], w->nb[2], 0);
        }
        ggml_tensor* y = ggml_mul_mat(ctx, w4, x4);
        return h == 1 ? ggml_reshape_2d(ctx, y, y->ne[0], n) : ggml_reshape_3d(ctx, y, y->ne[0], n, h);
    }
    // mul_mat_id likewise: float experts go kFloatGroup tokens at a time below mul_mm's tile rounding
    ggml_tensor* mm_id(ggml_tensor* as, ggml_tensor* x, ggml_tensor* ids) {
        const int64_t n = ids->ne[1];
        if (!is_float(as) || n < 32) return ggml_mul_mat_id(ctx, as, x, ids);
        ggml_tensor* out = nullptr;
        for (int64_t c = 0; c < n; c += kFloatGroup) {
            const int64_t k = std::min<int64_t>(kFloatGroup, n - c);
            ggml_tensor* xs = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], k, x->nb[1], x->nb[2], (size_t) c * x->nb[2]);
            ggml_tensor* is = ggml_view_2d(ctx, ids, ids->ne[0], k, ids->nb[1], (size_t) c * ids->nb[1]);
            ggml_tensor* y = ggml_mul_mat_id(ctx, as, xs, is);
            out = out ? ggml_concat(ctx, out, y, 2) : y;
        }
        return out;
    }

    // Gemma RMSNorm: the GGUF's stored weight already holds 1 + w (llama.cpp's converter), so the norm
    // is rms_norm * w as it stands
    ggml_tensor* rms(ggml_tensor* x, ggml_tensor* weight) {
        x = ggml_rms_norm(ctx, x, g.rms_eps);
        return weight ? ggml_mul(ctx, x, weight) : x;
    }

    ggml_tensor* rope(ggml_tensor* x, ggml_tensor* pos) {   // NeoX, the first n_rot dims, theta from the GGUF
        return ggml_rope_ext(ctx, x, pos, nullptr, g.n_rot, GGML_ROPE_TYPE_NEOX,
                             (int) std::min<int64_t>(g.ctx_train, INT32_MAX), g.rope_theta, 1.0f, 0.0f, 1.0f,
                             32.0f, 1.0f);
    }

    ggml_tensor* pos_input() {
        std::vector<int32_t> p((size_t) T);
        for (int t = 0; t < T; ++t) p[(size_t) t] = (int32_t) (p0 + t);
        return input(GGML_TYPE_I32, T, 1, p.data(), p.size() * 4);
    }

    // graph 1 of layer il: the projections, the cache writes, the indexer scores
    ggml_tensor* graph1(int il, ggml_tensor* ids) {
        const Mm3LayerState& s = m.ls[(size_t) il];
        ggml_tensor* pos = pos_input();
        pos_i = pos;
        ggml_tensor* x;
        if (il == 0) {
            ggml_tensor* e = ggml_get_rows(ctx, m.w("token_embd.weight"), ids);
            if (e->type != GGML_TYPE_F32) e = ggml_cast(ctx, e, GGML_TYPE_F32);
            expand(ggml_cpy(ctx, e, cols(m.c_h)));
            x = cols(m.c_h);
        } else {
            x = cols(m.c_h);
        }
        ggml_tensor* xn = rms(x, m.wl(il, "attn_norm.weight"));

        ggml_tensor* q = ggml_reshape_3d(ctx, mm(m.wl(il, "attn_q.weight"), xn), g.head_dim, g.n_head, T);
        q = rope(rms(q, m.wl(il, "attn_q_norm.weight")), pos);
        expand(ggml_cpy(ctx, q, cols3(m.c_q)));
        ggml_tensor* k = ggml_reshape_3d(ctx, mm(m.wl(il, "attn_k.weight"), xn), g.head_dim, g.n_kv, T);
        k = rope(rms(k, m.wl(il, "attn_k_norm.weight")), pos);
        ggml_tensor* v = ggml_reshape_3d(ctx, mm(m.wl(il, "attn_v.weight"), xn), g.head_dim, g.n_kv, T);
        expand(ggml_cpy(ctx, k, ggml_view_3d(ctx, s.kc, g.head_dim, g.n_kv, T, s.kc->nb[1], s.kc->nb[2],
                                             (size_t) p0 * s.kc->nb[2])));
        expand(ggml_cpy(ctx, v, ggml_view_3d(ctx, s.vc, g.head_dim, g.n_kv, T, s.vc->nb[1], s.vc->nb[2],
                                             (size_t) p0 * s.vc->nb[2])));

        ggml_tensor* iq = ggml_reshape_3d(ctx, mm(m.wl(il, "indexer.q_proj.weight"), xn), g.ix_dim, g.ix_heads, T);
        iq = rope(rms(iq, m.wl(il, "indexer.q_norm.weight")), pos);
        expand(ggml_cpy(ctx, iq, cols3(m.c_iq)));
        ggml_tensor* ik = ggml_reshape_2d(ctx, mm(m.wl(il, "indexer.k_proj.weight"), xn), g.ix_dim, T);
        ik = rms(ik, m.wl(il, "indexer.k_norm.weight"));
        ik = ggml_reshape_2d(ctx, rope(ggml_reshape_3d(ctx, ik, g.ix_dim, 1, T), pos), g.ix_dim, T);
        expand(ggml_cpy(ctx, ik, ggml_view_2d(ctx, s.ikc, g.ix_dim, T, s.ikc->nb[1], (size_t) p0 * s.ikc->nb[1])));

        // the scores of every cached key against the slice's index queries: [Tk, ix_heads, T]
        const int64_t Tk = p0 + T;
        ggml_tensor* ik_all = ggml_view_2d(ctx, s.ikc, g.ix_dim, Tk, s.ikc->nb[1], 0);
        ggml_tensor* scores = ggml_mul_mat(ctx, ik_all, cols3(m.c_iq));
        ggml_set_output(scores);
        expand(scores);
        return scores;
    }

    // The kept-block masks, built in-graph from graph 1's scores - select_masks' semantics exactly,
    // with no host round-trip: pool the scores to block maxima (the pool skips out-of-range reads,
    // which is the causal -inf padding of a block's future tokens), suppress the blocks past the
    // query's own, boost the local blocks above every real score, take the top-k blocks per
    // (index head, query), and expand the kept set back to a token mask with the causal term folded
    // in (0 for a kept causal token, a large negative otherwise - the softmax's exp underflows to
    // the same exact 0 the host's -inf gives).  Needs ix_local >= 1 like the model ships: the query's
    // own block is then force-kept, so its pooled score (which includes that block's future tokens)
    // is never compared - for every other block the pooled max is over causal tokens only, exactly
    // the host's.  Returns one [Tk, T] F32 mask per index head.
    std::vector<ggml_tensor*> select(ggml_tensor* scores) {
        const int64_t Tk = p0 + T;
        const int H = g.ix_heads;
        const int64_t blk = g.ix_blk;
        const int64_t nblk = (Tk + blk - 1) / blk;
        const int k = (int) std::min<int64_t>(g.ix_topk, nblk);
        const float one_f = 1.0f;
        ggml_tensor* one = input(GGML_TYPE_F32, 1, 1, &one_f, 4);

        // block scores [nblk, H, T]: max over each block's tokens (pad the tail block with -inf)
        ggml_tensor* sc = scores;
        if (nblk * blk > Tk) {
            const float neg_f = -INFINITY;   // an input only when consumed: an unused upload gets no buffer
            ggml_tensor* negc = input(GGML_TYPE_F32, 1, 1, &neg_f, 4);
            ggml_tensor* first = ggml_view_3d(ctx, sc, 1, H, T, sc->nb[1], sc->nb[2], 0);
            ggml_tensor* donor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nblk * blk - Tk, H, T);
            ggml_tensor* neg = ggml_add(ctx, ggml_scale(ctx, ggml_repeat(ctx, first, donor), 0.0f), negc);
            sc = ggml_concat(ctx, sc, neg, 0);
        }
        ggml_tensor* bs = ggml_pool_1d(ctx, sc, GGML_OP_POOL_MAX, (int) blk, (int) blk, 0);

        // the query's block (gathered from the state table) and the block ids; ggml's binary ops only
        // broadcast the second operand into the first's shape, so both are repeated to common shapes
        ggml_tensor* qb = ggml_get_rows(ctx, ggml_view_2d(ctx, m.c_blk, 1, m.n_ctx, m.c_blk->nb[0], 0), pos_i);
        ggml_tensor* qbt = ggml_reshape_3d(ctx, qb, 1, 1, T);
        ggml_tensor* bidx = ggml_reshape_3d(ctx, ggml_arange(ctx, 0.0f, (float) nblk, 1.0f), nblk, 1, 1);
        ggml_tensor* shape_n1t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nblk, 1, T);
        ggml_tensor* qbt_r = ggml_repeat(ctx, qbt, shape_n1t);
        ggml_tensor* bidx_r = ggml_repeat(ctx, bidx, shape_n1t);
        ggml_tensor* bidx_h = ggml_repeat(ctx, bidx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nblk, H, T));
        auto one_minus = [&](ggml_tensor* x) { return ggml_add(ctx, ggml_scale(ctx, x, -1.0f), one); };
        // le1[b, t] = 1 iff b <= qb(t); inw adds the local window's lower edge b >= qb - ix_local + 1
        ggml_tensor* le1 = ggml_clamp(ctx, ggml_add(ctx, ggml_sub(ctx, qbt_r, bidx_r), one), 0.0f, 1.0f);
        ggml_tensor* inw = ggml_clamp(ctx, ggml_add(ctx, ggml_sub(ctx, bidx_r, qbt_r),
                                                  ggml_scale(ctx, one, (float) g.ix_local)), 0.0f, 1.0f);
        ggml_tensor* lb = ggml_mul(ctx, le1, inw);                                     // [nblk, 1, T]
        // future blocks drop to -1e38 (only picked when nothing real remains, and the causal term
        // below masks their tokens anyway); local blocks rise to +1e38, above every real score
        ggml_tensor* eff = ggml_add(ctx, ggml_add(ctx, bs, ggml_scale(ctx, one_minus(le1), -1e38f)),
                                    ggml_scale(ctx, lb, 1e38f));                       // [nblk, H, T]
        ggml_tensor* top = ggml_argsort_top_k(ctx, eff, k);                            // [k, H, T] I32
        // the kept set as a 0/1 block mask: the picks are distinct, so the per-pick indicators sum to 0/1
        ggml_tensor* kbm = nullptr;
        for (int j = 0; j < k; ++j) {
            ggml_tensor* pj = ggml_cast(ctx, ggml_view_3d(ctx, top, 1, H, T, top->nb[1], top->nb[2],
                                                          (size_t) j * top->nb[0]), GGML_TYPE_F32);
            ggml_tensor* eq = one_minus(ggml_clamp(ctx, ggml_abs(ctx, ggml_sub(ctx, bidx_h, pj)), 0.0f, 1.0f));
            kbm = kbm ? ggml_add(ctx, kbm, eq) : eq;
        }
        // expand blocks to tokens (each token takes its block's row) and fold in the causal term
        ggml_tensor* shape_tt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Tk, T);
        ggml_tensor* sidx = ggml_repeat(ctx, ggml_reshape_2d(ctx, ggml_arange(ctx, 0.0f, (float) Tk, 1.0f), Tk, 1), shape_tt);
        ggml_tensor* post = ggml_repeat(ctx, ggml_reshape_2d(ctx, ggml_arange(ctx, (float) p0, (float) (p0 + T), 1.0f), 1, T), shape_tt);
        ggml_tensor* fut = ggml_clamp(ctx, ggml_sub(ctx, sidx, post), 0.0f, 1.0f);      // [Tk, T], 1 where future
        ggml_tensor* rows = ggml_view_2d(ctx, m.c_blk, Tk, 1, m.c_blk->nb[0], 0);
        ggml_tensor* donor2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Tk, T);
        ggml_tensor* bidxT = ggml_cast(ctx, ggml_repeat(ctx, rows, donor2), GGML_TYPE_I32);
        std::vector<ggml_tensor*> masks;
        for (int h = 0; h < H; ++h) {
            ggml_tensor* ah = ggml_view_3d(ctx, kbm, 1, nblk, T, kbm->nb[0], kbm->nb[2],
                                           (size_t) h * kbm->nb[1]);
            ggml_tensor* tok = ggml_get_rows(ctx, ah, bidxT);                          // [1, Tk, T]
            ggml_tensor* tok2 = ggml_view_2d(ctx, tok, Tk, T, tok->nb[2], 0);
            masks.push_back(ggml_scale(ctx, ggml_add(ctx, one_minus(tok2), fut), -1e38f));
        }
        expand(masks.back());
        return masks;
    }

    // graph 2 of layer il: the sparse attention over the host's masks, the MLP, the residual writes;
    // the head too when last (its logits are the graph's output then)
    ggml_tensor* graph2(int il, const std::vector<ggml_tensor*>& masks, bool last, ggml_tensor** logits_out) {
        const Mm3LayerState& s = m.ls[(size_t) il];
        const int64_t Tk = p0 + T;
        const int G = g.n_head / g.n_kv;
        const float scale = 1.0f / std::sqrt((float) g.head_dim);
        ggml_tensor* q = cols3(m.c_q);
        std::vector<ggml_tensor*> groups;
        for (int h = 0; h < g.n_kv; ++h) {
            ggml_tensor* kf = ggml_cpy(ctx,
                                       ggml_view_2d(ctx, s.kc, g.head_dim, Tk, s.kc->nb[2], (size_t) h * s.kc->nb[1]),
                                       ggml_new_tensor_2d(ctx, GGML_TYPE_F32, g.head_dim, Tk));
            ggml_tensor* vf = ggml_cpy(ctx,
                                       ggml_view_2d(ctx, s.vc, g.head_dim, Tk, s.vc->nb[2], (size_t) h * s.vc->nb[1]),
                                       ggml_new_tensor_2d(ctx, GGML_TYPE_F32, g.head_dim, Tk));
            ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, vf));   // [Tk, head_dim], materialised
            std::vector<ggml_tensor*> chunks;
            for (int c0 = 0; c0 < T; c0 += m.o.attn_chunk) {
                const int tc = std::min(m.o.attn_chunk, T - c0);
                ggml_tensor* qc = ggml_view_3d(ctx, q, g.head_dim, G, tc, q->nb[1], q->nb[2],
                                               (size_t) h * G * q->nb[1] + (size_t) c0 * q->nb[2]);
                ggml_tensor* mk = ggml_view_3d(ctx, masks[(size_t) h], Tk, 1, tc, masks[(size_t) h]->nb[1],
                                               masks[(size_t) h]->nb[1], (size_t) c0 * masks[(size_t) h]->nb[1]);
                ggml_tensor* sc = ggml_mul_mat(ctx, kf, qc);                       // [Tk, G, tc]
                sc = ggml_soft_max(ctx, ggml_add(ctx, ggml_scale(ctx, sc, scale), mk));
                chunks.push_back(ggml_mul_mat(ctx, vt, sc));                       // [head_dim, G, tc]
            }
            ggml_tensor* o = chunks[0];
            for (size_t i = 1; i < chunks.size(); ++i) o = ggml_concat(ctx, o, chunks[i], 2);
            groups.push_back(o);
        }
        ggml_tensor* att = groups[0];
        for (size_t i = 1; i < groups.size(); ++i) att = ggml_concat(ctx, att, groups[i], 1);
        att = ggml_cont(ctx, ggml_reshape_2d(ctx, att, (int64_t) g.n_head * g.head_dim, T));
        ggml_tensor* h = ggml_add(ctx, cols(m.c_h), mm(m.wl(il, "attn_output.weight"), att));

        ggml_tensor* xn = rms(h, m.wl(il, "ffn_norm.weight"));
        ggml_tensor* f;
        if (il < g.n_dense_lead) {
            f = ggml_swiglu_oai(ctx, mm(m.wl(il, "ffn_gate.weight"), xn), mm(m.wl(il, "ffn_up.weight"), xn),
                                g.swiglu_alpha, g.swiglu_limit);
            f = mm(m.wl(il, "ffn_down.weight"), f);
        } else {
            ggml_tensor* sh = ggml_swiglu_oai(ctx, mm(m.wl(il, "ffn_gate_shexp.weight"), xn),
                                              mm(m.wl(il, "ffn_up_shexp.weight"), xn), g.swiglu_alpha,
                                              g.swiglu_limit);
            sh = mm(m.wl(il, "ffn_down_shexp.weight"), sh);
            // the router in-graph (the GLM engine's route()): sigmoid, the correction bias, top-k, the
            // chosen probabilities renormalised and scaled
            ggml_tensor* probs = ggml_sigmoid(ctx, mm(m.wl(il, "ffn_gate_inp.weight"), xn));   // [n_expert, T]
            ggml_tensor* biased = ggml_add(ctx, probs, m.wl(il, "exp_probs_b.bias"));
            ggml_tensor* top = ggml_cont(ctx, ggml_argsort_top_k(ctx, biased, g.n_expert_used));   // [k, T]
            ggml_tensor* wts = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, g.n_expert, T), top);
            ggml_tensor* w2 = ggml_reshape_2d(ctx, wts, g.n_expert_used, T);
            ggml_tensor* sum = ggml_clamp(ctx, ggml_sum_rows(ctx, w2), 6.103515625e-5f, INFINITY);
            wts = ggml_reshape_3d(ctx, ggml_div(ctx, w2, sum), 1, g.n_expert_used, T);
            if (g.routed_scale != 1.0f) wts = ggml_scale(ctx, wts, g.routed_scale);
            ggml_tensor* x3 = ggml_reshape_3d(ctx, xn, g.n_embd, 1, T);
            ggml_tensor* up = mm_id(m.wl(il, "ffn_up_exps.weight"), x3, top);
            ggml_tensor* gate = mm_id(m.wl(il, "ffn_gate_exps.weight"), x3, top);
            ggml_tensor* hh = ggml_swiglu_oai(ctx, gate, up, g.swiglu_alpha, g.swiglu_limit);
            ggml_tensor* down = ggml_mul(ctx, mm_id(m.wl(il, "ffn_down_exps.weight"), hh, top), wts);
            ggml_tensor* routed = ggml_view_2d(ctx, down, g.n_embd, T, down->nb[2], 0);
            for (int i = 1; i < g.n_expert_used; ++i)
                routed = ggml_add(ctx, routed,
                                   ggml_view_2d(ctx, down, g.n_embd, T, down->nb[2], (size_t) i * down->nb[1]));
            f = ggml_add(ctx, routed, sh);
        }
        ggml_tensor* h2 = ggml_add(ctx, h, f);
        if (stream_out) *stream_out = h2;
        expand(ggml_cpy(ctx, h2, cols(m.c_h)));
        if (last) {
            ggml_tensor* cols_in = logits_rows > 1
                                       ? ggml_view_2d(ctx, h2, g.n_embd, logits_rows, h2->nb[1],
                                                      (size_t) (T - logits_rows) * h2->nb[1])
                                       : ggml_view_2d(ctx, h2, g.n_embd, 1, h2->nb[1],
                                                      (size_t) (T - 1) * h2->nb[1]);
            ggml_tensor* logits = mm(m.w("output.weight"), rms(cols_in, m.w("output_norm.weight")));
            ggml_set_output(logits);
            expand(logits);
            *logits_out = logits;
        }
        return nullptr;
    }
};

}  // namespace

bool Mm3Metal::Impl::compute(ggml_cgraph* gf,
                             const std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>>& up,
                             std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        err = "cannot allocate the compute graph";
        return false;
    }
    for (const auto& [t, d] : up) ggml_backend_tensor_set(t, d.data(), 0, d.size());
    const auto t1 = std::chrono::steady_clock::now();
    st.host_seconds += std::chrono::duration<double>(t1 - t0).count();
    ++st.graphs;
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        err = "the compute graph failed";
        return false;
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    st.gpu_seconds += dt;
    if (std::getenv("MAYA_METAL_PROFILE"))
        std::fprintf(stderr, "graph %d nodes %.1f ms\n", ggml_graph_n_nodes(gf), dt * 1e3);
    return true;
}

// ref.py's msa_select on the host, from graph 1's scores [Tk, ix_heads, T]: per (group, query) the kept
// blocks become a 0 / -inf mask [Tk, T] per group (graph 2 broadcasts it over the group's query heads).
// The fallback path - Builder::select builds the same masks in-graph for the shipped geometry.
bool select_masks(const Mm3Geometry& g, const float* scores, int64_t Tk, int T, int64_t p0,
                  std::vector<std::vector<float>>& masks) {
    const int64_t nblk = (Tk + g.ix_blk - 1) / g.ix_blk;
    masks.assign((size_t) g.ix_heads, std::vector<float>((size_t) Tk * T, -INFINITY));
    std::vector<float> bs((size_t) nblk);
    std::vector<int> order((size_t) nblk);
    for (int h = 0; h < g.ix_heads; ++h) {
        float* mk = masks[(size_t) h].data();
        for (int t = 0; t < T; ++t) {
            const int64_t p = p0 + t;
            const int64_t qb = p / g.ix_blk;
            const float* sc = scores + (size_t) h * Tk + (size_t) t * Tk * g.ix_heads;
            for (int64_t b = 0; b <= qb; ++b) {
                const int64_t s0 = b * g.ix_blk, s1 = std::min(s0 + g.ix_blk, p + 1);
                float mx = -INFINITY;
                for (int64_t s = s0; s < s1; ++s) mx = std::max(mx, sc[s]);
                bs[(size_t) b] = mx;
            }
            // the local blocks are forced in, then the best of the rest up to topk in total
            std::vector<char> kept((size_t) nblk, 0);
            int n_kept = 0;
            for (int64_t b = std::max<int64_t>(0, qb - g.ix_local + 1); b <= qb; ++b) {
                kept[(size_t) b] = 1;
                ++n_kept;
            }
            if (n_kept < g.ix_topk) {
                std::iota(order.begin(), order.begin() + (qb + 1), 0);
                std::sort(order.begin(), order.begin() + (qb + 1),
                          [&](int a, int b) { return bs[(size_t) a] > bs[(size_t) b]; });
                for (int64_t i = 0; i <= qb && n_kept < g.ix_topk; ++i) {
                    const int b = order[(size_t) i];
                    if (!kept[(size_t) b]) {
                        kept[(size_t) b] = 1;
                        ++n_kept;
                    }
                }
            }
            for (int64_t s = 0; s <= p; ++s)
                if (kept[(size_t) (s / g.ix_blk)]) mk[(size_t) t * Tk + s] = 0.0f;
        }
    }
    return true;
}

// Whether Builder::select can build the masks in-graph for this geometry: one mask per index head is
// what graph 2 consumes (a mask per KV head), the boost arithmetic needs the local window inside the
// top-k, and ix_local >= 1 keeps the query's own block force-kept so its pooled score (the one block
// whose max runs over future tokens too) is never compared.  MAYA_MM3_HOST_SELECT forces the host path.
static bool gpu_select_ok(const Mm3Geometry& g) {
    if (std::getenv("MAYA_MM3_HOST_SELECT")) return false;
    return g.ix_heads == g.n_kv && g.ix_blk >= 1 && g.ix_topk >= 1 && g.ix_local >= 1 &&
           g.ix_local <= g.ix_topk;
}

bool Mm3Metal::Impl::run_slice(const int32_t* tok, int T, std::vector<float>& logits, std::string& err,
                               int logits_rows, std::vector<float>* rows_out) {
    const size_t mem = ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false);
    if (arena.size() < mem) arena.resize(mem);
    const int64_t Tk = pos + T;
    std::vector<int32_t> ids(tok, tok + T);
    std::vector<float> scores((size_t) Tk * g.ix_heads * T);
    std::vector<std::vector<float>> masks;
    // the draft's taps: this slice's output streams at the tap layers, tap-major [NT][T][dim]
    const std::vector<int>& dtaps = has_draft ? draft.geo().taps : std::vector<int>();
    std::vector<float> taps_stage;
    if (has_draft) taps_stage.assign((size_t) dtaps.size() * T * g.n_embd, 0.0f);
    auto tap_j = [&](int il) {
        for (size_t j = 0; j < dtaps.size(); ++j)
            if (dtaps[j] == il) return (int) j;
        return -1;
    };
    // the readbacks after a layer's attending graph: the last layer's logits and the draft's taps
    auto finish = [&](ggml_tensor* logits_t, ggml_tensor* tap_t, int tj, bool last) {
        if (last) {
            if (rows_out) {
                std::vector<float> all((size_t) g.n_vocab * logits_rows);
                ggml_backend_tensor_get(logits_t, all.data(), 0, all.size() * sizeof(float));
                rows_out->assign((size_t) T * g.n_vocab, 0.0f);
                for (int r = 0; r < logits_rows; ++r)
                    for (int v = 0; v < g.n_vocab; ++v)
                        (*rows_out)[(size_t) (T - logits_rows + r) * g.n_vocab + v] =
                            all[(size_t) v + (size_t) r * g.n_vocab];
                logits.assign(all.begin() + (size_t) (logits_rows - 1) * g.n_vocab, all.end());
            } else {
                logits.resize((size_t) g.n_vocab);
                ggml_backend_tensor_get(logits_t, logits.data(), 0, logits.size() * sizeof(float));
            }
        }
        if (tj >= 0 && tap_t) {
            std::vector<float> h2((size_t) g.n_embd * T);
            ggml_backend_tensor_get(tap_t, h2.data(), 0, h2.size() * sizeof(float));
            float* dst = taps_stage.data() + (size_t) tj * T * g.n_embd;
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < g.n_embd; ++d)
                    dst[(size_t) t * g.n_embd + d] = h2[(size_t) d + (size_t) t * g.n_embd];
        }
    };
    const bool gpu_sel = gpu_select_ok(g);
    for (int il = 0; il < g.n_layer; ++il) {
        const bool last = il == g.n_layer - 1;
        ggml_tensor* logits_t = nullptr;
        if (gpu_sel) {
            // one fused graph per layer: graph 1, the in-graph mask build, graph 2 - the scores and
            // the masks never leave the GPU, so the layer costs one pipeline drain instead of two
            ggml_init_params ip{mem, arena.data(), true};
            ggml_context* ctx = ggml_init(ip);
            Builder b{*this, g, ctx, ggml_new_graph_custom(ctx, kGraphNodes, false), T, pos, {}};
            std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> up;
            ggml_tensor* idt = nullptr;
            if (il == 0)
                idt = b.input(GGML_TYPE_I32, T, 1, ids.data(), ids.size() * 4);
            ggml_tensor* sc = b.graph1(il, idt);
            std::vector<ggml_tensor*> mt = b.select(sc);
            ggml_tensor* tap_t = nullptr;
            const int tj = tap_j(il);
            b.stream_out = tj >= 0 ? &tap_t : nullptr;
            if (last) b.logits_rows = logits_rows;
            b.graph2(il, mt, last, &logits_t);
            for (Upload& u : b.uploads) up.push_back({u.t, std::move(u.data)});
            if (!compute(b.gf, up, err)) {
                ggml_free(ctx);
                return false;
            }
            finish(logits_t, tap_t, tj, last);
            ggml_free(ctx);
            continue;
        }
        for (int part = 0; part < 2; ++part) {
            ggml_init_params ip{mem, arena.data(), true};
            ggml_context* ctx = ggml_init(ip);
            Builder b{*this, g, ctx, ggml_new_graph_custom(ctx, kGraphNodes, false), T, pos, {}};
            std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> up;
            if (part == 0) {
                ggml_tensor* idt = nullptr;
                if (il == 0)
                    idt = b.input(GGML_TYPE_I32, T, 1, ids.data(), ids.size() * 4);
                ggml_tensor* sc = b.graph1(il, idt);
                for (Upload& u : b.uploads) up.push_back({u.t, std::move(u.data)});
                if (!compute(b.gf, up, err)) {
                    ggml_free(ctx);
                    return false;
                }
                ggml_backend_tensor_get(sc, scores.data(), 0, scores.size() * sizeof(float));
                ggml_free(ctx);
                if (!select_masks(g, scores.data(), Tk, T, pos, masks)) {
                    err = "block selection failed";
                    return false;
                }
            } else {
                std::vector<ggml_tensor*> mt;
                for (int h = 0; h < g.ix_heads; ++h)
                    mt.push_back(b.input(GGML_TYPE_F32, Tk, T, masks[(size_t) h].data(),
                                         masks[(size_t) h].size() * sizeof(float)));
                ggml_tensor* tap_t = nullptr;
                const int tj = tap_j(il);
                b.stream_out = tj >= 0 ? &tap_t : nullptr;
                if (last) b.logits_rows = logits_rows;
                b.graph2(il, mt, last, &logits_t);
                for (Upload& u : b.uploads) up.push_back({u.t, std::move(u.data)});
                if (!compute(b.gf, up, err)) {
                    ggml_free(ctx);
                    return false;
                }
                finish(logits_t, tap_t, tj, last);
                ggml_free(ctx);
            }
        }
    }
    if (has_draft && !draft.context(taps_stage.data(), T, pos, err)) return false;
    return true;
}

// ---------------------------------------------------------------- public API

Mm3Metal::Mm3Metal() : p_(new Impl) {}
Mm3Metal::~Mm3Metal() = default;

void Mm3Metal::set_options(const Mm3Options& o) { p_->o = o; }

bool Mm3Metal::load(const std::string& path, int64_t max_ctx, std::string& err) {
    Impl& m = *p_;
    if (max_ctx < 1 || m.o.n_ubatch < 1 || m.o.attn_chunk < 1) {
        err = "the context, the slice and the attention chunk need at least one token";
        return false;
    }
    m.n_ctx = max_ctx;
    if (!m.open_files(path, err) || !m.read_geometry(err) || !m.init_backends(err) || !m.load_weights(err) ||
        !m.init_state(err))
        return false;
    reset();
    return true;
}

void Mm3Metal::reset() {
    p_->pos = 0;
    p_->snap_pos = -1;
    if (p_->has_draft) p_->draft.reset_context();
}

void Mm3Metal::snapshot_save() { p_->snap_pos = p_->pos; }

bool Mm3Metal::snapshot_restore() {
    if (p_->snap_pos < 0) return false;
    p_->pos = p_->snap_pos;
    return true;
}

int64_t Mm3Metal::snapshot_position() const { return p_->snap_pos; }
int Mm3Metal::ubatch() const { return p_->o.n_ubatch; }

bool Mm3Metal::forward(const std::vector<int32_t>& tokens, std::vector<float>& logits, std::string& err) {
    Impl& m = *p_;
    if (tokens.empty()) {
        err = "empty forward";
        return false;
    }
    if (m.pos + (int64_t) tokens.size() > m.n_ctx) {
        err = "context full (" + std::to_string(m.n_ctx) + " tokens)";
        return false;
    }
    for (size_t at = 0; at < tokens.size(); at += (size_t) m.o.n_ubatch) {
        const int n = (int) std::min(tokens.size() - at, (size_t) m.o.n_ubatch);
        if (!m.run_slice(tokens.data() + at, n, logits, err)) return false;
        m.pos += n;
        m.st.tokens += (uint64_t) n;
    }
    return true;
}

bool Mm3Metal::load_draft(const std::string& dir, std::string& err) {
    Impl& m = *p_;
    if (!m.draft.load(dir, m.w("token_embd.weight"), m.w("output.weight"), m.n_ctx, err)) return false;
    m.has_draft = true;
    return true;
}

int Mm3Metal::draft_block() const { return p_->has_draft ? p_->draft.geo().block : 0; }

bool Mm3Metal::draft_propose(int32_t anchor, int64_t pos, int n, std::vector<int32_t>& drafts,
                             std::vector<float>& confs, std::string& err) {
    if (!p_->has_draft) {
        drafts.clear();
        confs.clear();
        return true;
    }
    return p_->draft.propose(anchor, pos, n, drafts, confs, err);
}

bool Mm3Metal::forward_rows(const std::vector<int32_t>& tokens, std::vector<float>& rows, std::string& err) {
    Impl& m = *p_;
    if (tokens.empty()) {
        err = "empty forward";
        return false;
    }
    if ((int64_t) tokens.size() > m.o.n_ubatch) {
        err = "a verify window is wider than one slice";
        return false;
    }
    if (m.pos + (int64_t) tokens.size() > m.n_ctx) {
        err = "context full (" + std::to_string(m.n_ctx) + " tokens)";
        return false;
    }
    std::vector<float> logits;
    if (!m.run_slice(tokens.data(), (int) tokens.size(), logits, err, (int) tokens.size(), &rows))
        return false;
    m.pos += (int64_t) tokens.size();
    m.st.tokens += tokens.size();
    return true;
}

void Mm3Metal::rewind_to(int64_t pos) {
    if (pos < 0) pos = 0;
    if (pos > p_->n_ctx) pos = p_->n_ctx;
    p_->pos = pos;
}

int64_t Mm3Metal::position() const { return p_->pos; }
int64_t Mm3Metal::max_context() const { return p_->n_ctx; }
const Mm3Geometry& Mm3Metal::geometry() const { return p_->g; }
const Mm3Stats& Mm3Metal::stats() const { return p_->st; }

std::string Mm3Metal::describe() const {
    const Impl& m = *p_;
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "MiniMax-M3.1-Flash, %d layers (%d dense lead), %d experts top-%d; weights %.2f GB, state "
                  "%.2f GB (%lld ctx) on %s",
                  m.g.n_layer, m.g.n_dense_lead, m.g.n_expert, m.g.n_expert_used, m.st.weight_bytes / 1e9,
                  m.st.state_bytes / 1e9, (long long) m.n_ctx, m.gpu ? ggml_backend_name(m.gpu) : "CPU");
    return buf;
}

}  // namespace maya::metal
