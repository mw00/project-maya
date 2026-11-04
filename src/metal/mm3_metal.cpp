// src/metal/mm3_metal.cpp - Mm3Metal: the forward graph, following tools/mm3/ref.py op for op.
//
// One slice of T tokens runs two graphs per layer:
//   graph 1   the attention projections of ref.py's attention(): q/k/v and the indexer's iq/ik, each
//             Gemma-normed and NeoX-roped, the keys and values written into the layer's BF16 caches, iq
//             kept (BF16) in a carry, and the indexer scores iq . ik over every cached position read
//             back to the host;
//   (host)    the kept blocks, exactly ref.py's msa_select: a block scores the max of its tokens'
//             scores, the query's own local blocks are forced in, the top blocks are kept;
//   graph 2   ref.py's msa_attend per KV group (scores, the host's kept-block mask, softmax, the value
//             product), the output projection, then the MLP: dense swigluoai, or the MoE - sigmoid
//             router with the correction bias and renormalised weights in-graph, the shared expert, and
//             the routed experts (all resident, so mul_mat_id consumes the router's top-k directly:
//             no pool, no graph cut at the router, unlike the GLM engine).
// The last layer's second graph also runs the head over the slice's last row.
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

    // graph 1 of layer il: the projections, the cache writes, the indexer scores (the graph's output)
    ggml_tensor* graph1(int il, ggml_tensor* ids) {
        const Mm3LayerState& s = m.ls[(size_t) il];
        ggml_tensor* pos = pos_input();
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
        expand(ggml_cpy(ctx, h2, cols(m.c_h)));
        if (last) {
            ggml_tensor* last_col = ggml_view_2d(ctx, h2, g.n_embd, 1, h2->nb[1], (size_t) (T - 1) * h2->nb[1]);
            ggml_tensor* logits = mm(m.w("output.weight"), rms(last_col, m.w("output_norm.weight")));
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
// blocks become a 0 / -inf mask [Tk, T] per group (graph 2 broadcasts it over the group's query heads)
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

bool Mm3Metal::Impl::run_slice(const int32_t* tok, int T, std::vector<float>& logits, std::string& err) {
    const size_t mem = ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false);
    if (arena.size() < mem) arena.resize(mem);
    const int64_t Tk = pos + T;
    std::vector<int32_t> ids(tok, tok + T);
    std::vector<float> scores((size_t) Tk * g.ix_heads * T);
    std::vector<std::vector<float>> masks;
    for (int il = 0; il < g.n_layer; ++il) {
        const bool last = il == g.n_layer - 1;
        ggml_tensor* logits_t = nullptr;
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
                b.graph2(il, mt, last, &logits_t);
                for (Upload& u : b.uploads) up.push_back({u.t, std::move(u.data)});
                if (!compute(b.gf, up, err)) {
                    ggml_free(ctx);
                    return false;
                }
                if (last) {
                    logits.resize((size_t) g.n_vocab);
                    ggml_backend_tensor_get(logits_t, logits.data(), 0, logits.size() * sizeof(float));
                }
                ggml_free(ctx);
            }
        }
    }
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
