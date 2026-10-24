// src/metal/glm_metal.cpp - GlmMetal: the forward graph (ref/upstream-llama.cpp/glm5-next.cpp, op for op) cut
// at every MoE router so the host can make the selected experts resident between graphs.
//
// One slice of T tokens runs as a chain of graphs:
//   graph 0          embeddings -> layers up to the first MoE layer m, ending after m's router
//   graph for m      m's experts (pool slots) + combine -> next layers ... ending after the next router
//   last graph       the last MoE layer's experts -> head (the last token's logits)
// Between two graphs the host reads the router's top-k, ExpertPool::ensure() maps experts to slots (reading the
// misses from the SSD), and the slot ids are the next graph's input.  Everything a graph hands to the next one
// lives in the carry tensors (glm_metal_impl.hpp); everything a token hands to the next lives in LayerState.
#include "glm_metal_impl.hpp"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace maya::metal {

namespace {

constexpr int kGraphNodes = 32768;
constexpr int kFloatGroup = 16;     // tokens per mul_mat_id of float experts (mm_id)
constexpr int kChunkTensors = 48;   // a DSA layer's tensors per attn_chunk gather (35 measured)
constexpr int kGroupTensors = 16;   // a MoE layer's tensors per float-expert group (12: 4 each for gate, up, down)
constexpr int64_t kMmMinK = 64;     // ggml-metal takes mul_mm only from this inner length (ggml_metal_op_mul_mat_use_mm)
constexpr int kSparseMinT = 17;     // slices from this length take ggml_sparse_mla: its 8-head threadgroups leave the GPU
                                    // idle on a token (decode) or a verify window (at most 16), which keep mul_mm
constexpr size_t kSparseScoreBytes = (size_t) 256 << 20;   // ggml_sparse_mla gathers nothing: its chunks are sized by
                                                           // the indexer's scores ([n_pool, heads, tokens] f32) instead

struct Upload {
    ggml_tensor* t;
    std::vector<uint8_t> data;
};

// One graph under construction.
struct Builder {
    GlmMetal::Impl& m;
    const GlmGeometry& g;
    ggml_context* ctx;
    ggml_cgraph* gf;
    int T;          // tokens in the slice
    int64_t p0;     // position of the slice's first token
    bool window;    // a verify window: every row's state kept, every row's logits
    const float* emb = nullptr;   // the slice's token embeddings [n_embd, T] (the MTP cache fill pairs them with rows)
    std::vector<Upload> uploads;
    std::vector<std::pair<std::string, ggml_tensor*>> dumps;

    ggml_tensor* input(ggml_type type, int64_t ne0, int64_t ne1, const void* data, size_t bytes,
                       int64_t ne2 = 1, int64_t ne3 = 1) {
        ggml_tensor* t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        ggml_set_input(t);
        uploads.push_back({t, std::vector<uint8_t>((const uint8_t*) data, (const uint8_t*) data + bytes)});
        return t;
    }

    void cb(ggml_tensor* t, const char* name, int il) {
        if (m.o.dump_dir.empty()) return;
        std::string n = il >= 0 ? std::string(name) + "-" + std::to_string(il) : std::string(name);
        ggml_set_name(t, n.c_str());
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
        dumps.push_back({n, t});
    }

    void expand(ggml_tensor* t) { ggml_build_forward_expand(gf, t); }

    // columns [c0, c0 + n) of a [ne0, U] carry
    ggml_tensor* cols(ggml_tensor* t, int64_t n, int64_t c0 = 0) {
        return ggml_view_2d(ctx, t, t->ne[0], n, t->nb[1], (size_t) c0 * t->nb[1]);
    }
    ggml_tensor* cols3(ggml_tensor* t, int64_t n) {   // first n of the last axis of a 3D carry
        return ggml_view_3d(ctx, t, t->ne[0], t->ne[1], n, t->nb[1], t->nb[2], 0);
    }

    // y = w x.  ggml-metal's mul_mm (taken from 9 columns up) rounds its tiles to half, so a float weight (F32/F16:
    // the router, mHC, every fixture tensor) runs as one f32 matvec per column instead - the columns on a broadcast
    // axis, the same kernel a single token uses, so a prompt and a decode step give the same bits.  A quantized
    // weight keeps mul_mm, as llama.cpp does (its tiles are no coarser than the CUDA engine's q8_1 activations).
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
        if (w->ne[2] > 1) {   // one weight per x->ne[2] (heads): move it to the outer broadcast axis
            GGML_ASSERT(w->ne[2] == h);
            w4 = ggml_view_4d(ctx, w, w->ne[0], w->ne[1], 1, h, w->nb[1], w->nb[2], w->nb[2], 0);
        }
        ggml_tensor* y = ggml_mul_mat(ctx, w4, x4);   // [N, 1, n, h]
        return h == 1 ? ggml_reshape_2d(ctx, y, y->ne[0], n) : ggml_reshape_3d(ctx, y, y->ne[0], n, h);
    }
    // mul_mat_id likewise: below 32 tokens ggml-metal runs the f32 matvec; float experts go 16 tokens at a time
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

    ggml_tensor* rms(ggml_tensor* x, ggml_tensor* weight) {
        x = ggml_rms_norm(ctx, x, g.rms_eps);
        return weight ? ggml_mul(ctx, x, weight) : x;
    }

    // ---- mHC
    ggml_tensor* hc_pre(ggml_tensor* R, int il, bool ffn, ggml_tensor** post, ggml_tensor** comb) {
        const char* half = ffn ? "ffn" : "attn";
        ggml_tensor* fn = m.wl(il, ffn ? "hc_ffn_fn.weight" : "hc_attn_fn.weight");
        ggml_tensor* scale = m.wl(il, ffn ? "hc_ffn_scale.weight" : "hc_attn_scale.weight");
        ggml_tensor* base = m.wl(il, ffn ? "hc_ffn_base.weight" : "hc_attn_base.weight");
        const int hc = g.hc;
        ggml_tensor* flat = ggml_reshape_2d(ctx, R, (int64_t) hc * g.n_embd, T);
        ggml_tensor* mixes = mm(fn, ggml_rms_norm(ctx, flat, g.rms_eps));
        cb(mixes, ffn ? "hc_mixes_ffn" : "hc_mixes_attn", il);
        auto v1 = [&](ggml_tensor* t, int64_t n, int64_t i0) {
            return ggml_view_1d(ctx, t, n, ggml_row_size(t->type, i0));
        };
        auto v2 = [&](int64_t n, int64_t i0) {
            return ggml_view_2d(ctx, mixes, n, T, mixes->nb[1], ggml_row_size(mixes->type, i0));
        };
        ggml_tensor* pre = ggml_add(ctx, ggml_mul(ctx, v2(hc, 0), v1(scale, 1, 0)), v1(base, hc, 0));
        pre = ggml_scale_bias(ctx, ggml_sigmoid(ctx, pre), 1.0f, g.hc_eps);
        ggml_tensor* p = ggml_add(ctx, ggml_mul(ctx, v2(hc, hc), v1(scale, 1, 1)), v1(base, hc, hc));
        *post = ggml_scale(ctx, ggml_sigmoid(ctx, p), 2.0f);
        *comb = ggml_dsv4_hc_comb(ctx, mixes, scale, base, g.hc_eps, g.hc_iters);
        cb(pre, ffn ? "hc_pre_ffn" : "hc_pre_attn", il);
        cb(*post, ffn ? "hc_post_ffn" : "hc_post_attn", il);
        cb(*comb, ffn ? "hc_comb_ffn" : "hc_comb_attn", il);
        (void) half;
        return ggml_dsv4_hc_pre(ctx, R, pre);
    }

    ggml_tensor* hc_post(ggml_tensor* x, ggml_tensor* residual, ggml_tensor* post, ggml_tensor* comb) {
        return ggml_dsv4_hc_post(ctx, x, residual, post, comb);
    }

    // ---- KDA (linear attention) layer
    ggml_tensor* conv1d(int il, int qkv, ggml_tensor* x, const char* wname, const char* cname) {
        const LayerState& s = m.ls[(size_t) il];
        const int64_t d_inner = (int64_t) g.kda_head_dim * g.n_head;
        const int64_t dc = g.d_conv - 1;
        ggml_tensor* st = ggml_view_3d(ctx, s.conv, dc, d_inner, 1, s.conv->nb[1], s.conv->nb[2],
                                       (size_t) qkv * s.conv->nb[2]);
        ggml_tensor* proj = mm(m.wl(il, wname), x);
        ggml_tensor* cx = ggml_concat(ctx, st, ggml_transpose(ctx, ggml_reshape_3d(ctx, proj, d_inner, T, 1)), 0);
        // the newest d_conv-1 inputs are the next slice's history
        expand(ggml_cpy(ctx, ggml_view_3d(ctx, cx, dc, d_inner, 1, cx->nb[1], cx->nb[2], (size_t) T * cx->nb[0]),
                        st));
        if (window)   // every row's history: accept(n) takes columns n .. n + dc - 1
            expand(ggml_cpy(ctx, cx, ggml_view_3d(ctx, s.conv_win, dc + T, d_inner, 1, s.conv_win->nb[1],
                                                  s.conv_win->nb[2], (size_t) qkv * s.conv_win->nb[2])));
        ggml_tensor* w = ggml_reshape_2d(ctx, m.wl(il, cname), g.d_conv, d_inner);
        ggml_tensor* y = ggml_silu(ctx, ggml_ssm_conv(ctx, cx, w));
        return ggml_reshape_4d(ctx, y, g.kda_head_dim, g.n_head, T, 1);
    }

    ggml_tensor* l2(ggml_tensor* x) {   // build_gdn_l2_norm: x / sqrt(|x|^2 + eps)
        const float n = (float) x->ne[0];
        return ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-6f / n), 1.0f / std::sqrt(n));
    }

    ggml_tensor* kda(int il, ggml_tensor* x) {
        const LayerState& s = m.ls[(size_t) il];
        const int hd = g.kda_head_dim, nh = g.n_head;
        ggml_tensor* q = conv1d(il, 0, x, "attn_q.weight", "ssm_conv1d_q.weight");
        ggml_tensor* k = conv1d(il, 1, x, "attn_k.weight", "ssm_conv1d_k.weight");
        ggml_tensor* v = conv1d(il, 2, x, "attn_v.weight", "ssm_conv1d_v.weight");
        cb(q, "kda_q_conv", il);
        cb(k, "kda_k_conv", il);
        cb(v, "kda_v_conv", il);

        ggml_tensor* g1 = mm(m.wl(il, "ssm_f_b.weight"), mm(m.wl(il, "ssm_f_a.weight"), x));
        g1 = ggml_add(ctx, g1, m.wl(il, "ssm_dt.bias"));
        g1 = ggml_mul(ctx, ggml_reshape_3d(ctx, g1, hd, nh, T), ggml_reshape_3d(ctx, m.wl(il, "ssm_a"), 1, nh, 1));
        g1 = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, g1, -1.0f)), g.kda_lower_bound);
        cb(g1, "kda_g1", il);
        g1 = ggml_reshape_4d(ctx, g1, hd, nh, T, 1);

        ggml_tensor* beta = mm(m.wl(il, "ssm_beta.weight"), x);
        beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta, 1, nh, T, 1));
        cb(beta, "kda_beta", il);

        ggml_tensor* state = ggml_reshape_4d(ctx, s.ssm, hd, hd, nh, 1);
        ggml_tensor* out = ggml_gated_delta_net(ctx, l2(q), l2(k), v, g1, beta, state, window ? T : 1);
        const size_t es = ggml_element_size(out);
        ggml_tensor* o = ggml_view_4d(ctx, out, hd, nh, T, 1, (size_t) hd * es, (size_t) hd * nh * es,
                                      (size_t) hd * nh * T * es, 0);
        const int64_t S = (int64_t) hd * hd * nh;
        // the state copy follows the GDN directly: ggml-metal fuses the two (the kernel writes the state there and
        // skips its own output's state tail), so nothing may read that tail - a window's other copies read the copy
        if (!window) {
            expand(ggml_cpy(ctx, ggml_view_1d(ctx, out, S, (size_t) hd * nh * T * es), ggml_view_1d(ctx, s.ssm, S, 0)));
        } else {   // the state after every row, newest first (slot 0 = after the last row)
            ggml_tensor* slots = ggml_cpy(ctx, ggml_view_2d(ctx, out, S, T, (size_t) S * es, (size_t) hd * nh * T * es),
                                          ggml_view_2d(ctx, s.ssm_win, S, T, s.ssm_win->nb[1], 0));
            expand(slots);
            expand(ggml_cpy(ctx, ggml_view_1d(ctx, slots, S, 0), ggml_view_1d(ctx, s.ssm, S, 0)));
        }
        o = ggml_cont(ctx, o);
        cb(o, "kda_scan_out", il);

        ggml_tensor* g2 = mm(m.wl(il, "ssm_g_b.weight"), mm(m.wl(il, "ssm_g_a.weight"), x));
        g2 = ggml_reshape_3d(ctx, g2, hd, nh, T);
        ggml_tensor* normed = rms(ggml_reshape_3d(ctx, o, hd, nh, T), m.wl(il, "ssm_norm.weight"));
        cb(g2, "kda_g2", il);
        cb(normed, "kda_normed", il);
        ggml_tensor* gated = ggml_cont_2d(ctx, ggml_mul(ctx, normed, ggml_sigmoid(ctx, g2)), (int64_t) hd * nh, T);
        ggml_tensor* y = mm(m.wl(il, "attn_output.weight"), gated);
        cb(y, "kda_out", il);
        return y;
    }

    // ---- DSA layer: nope-MLA over the k-pool indexer's selection

    // the cache half: the latent, the indexer key and pool gate of positions pst .. pst+n-1 (x [n_embd, n]), and the
    // pools those positions complete
    void dsa_write(int il, ggml_tensor* x, int64_t pst, int64_t n) {
        const LayerState& s = m.ls[(size_t) il];
        const int kv = g.kv_lora, kd = g.idx_key, kp = g.idx_kpool;
        ggml_tensor* kvc = rms(mm(m.wl(il, "attn_kv_a_mqa.weight"), x), m.wl(il, "attn_kv_a_norm.weight"));
        cb(kvc, "kv_cmpr", il);
        expand(ggml_cpy(ctx, kvc, ggml_view_2d(ctx, s.lat, kv, n, s.lat->nb[1], (size_t) pst * s.lat->nb[1])));
        ggml_tensor* ik = ggml_norm(ctx, mm(m.wl(il, "indexer.attn_k.weight"), x), g.norm_eps);
        ik = ggml_add(ctx, ggml_mul(ctx, ik, m.wl(il, "indexer.k_norm.weight")), m.wl(il, "indexer.k_norm.bias"));
        ggml_tensor* ig = mm(m.wl(il, "indexer_compressor_gate.weight"), x);
        cb(ik, "indexer_k", il);
        cb(ig, "indexer_gate", il);
        const size_t cell = s.ikg->nb[1];
        expand(ggml_cpy(ctx, ik, ggml_view_2d(ctx, s.ikg, kd, n, cell, (size_t) pst * cell)));
        expand(ggml_cpy(ctx, ig, ggml_view_2d(ctx, s.ikg, kd, n, cell, (size_t) pst * cell + ggml_row_size(GGML_TYPE_F32, kd))));
        const int64_t pa = pst / kp, pb = (pst + n) / kp;   // pools completed by these positions
        if (pb > pa) {
            const int64_t nn = pb - pa;
            ggml_tensor* rows = ggml_view_3d(ctx, s.ikg, 2 * kd, kp, nn, cell, cell * kp, (size_t) pa * kp * cell);
            ggml_tensor* pk = ggml_view_3d(ctx, rows, kd, kp, nn, rows->nb[1], rows->nb[2], 0);
            ggml_tensor* pg = ggml_view_3d(ctx, rows, kd, kp, nn, rows->nb[1], rows->nb[2], ggml_row_size(GGML_TYPE_F32, kd));
            ggml_tensor* lg = ggml_cont(ctx, ggml_permute(ctx, ggml_add(ctx, pg, m.wl(il, "indexer_compressor_ape.weight")), 1, 0, 2, 3));
            ggml_tensor* pr = ggml_reshape_3d(ctx, ggml_soft_max(ctx, ggml_reshape_2d(ctx, lg, kp, kd * nn)), kp, kd, nn);
            ggml_tensor* pooled = ggml_sum_rows(ctx, ggml_mul(ctx, pr, ggml_cont(ctx, ggml_permute(ctx, pk, 1, 0, 2, 3))));
            expand(ggml_cpy(ctx, ggml_reshape_2d(ctx, pooled, kd, nn),
                            ggml_view_2d(ctx, s.pooled, kd, nn, s.pooled->nb[1], (size_t) pa * s.pooled->nb[1])));
        }
    }

    // the whole layer at the slice's positions p0 .. p0+T-1: the cache half, then the queries over the selection
    ggml_tensor* dsa(int il, ggml_tensor* x) {
        const LayerState& s = m.ls[(size_t) il];
        const int nh = g.n_head, kv = g.kv_lora, kd = g.idx_key, ih = g.idx_heads, kp = g.idx_kpool;
        const float kq_scale = 1.0f / std::sqrt((float) g.head_k_mla);

        ggml_tensor* qr = rms(mm(m.wl(il, "attn_q_a.weight"), x), m.wl(il, "attn_q_a_norm.weight"));
        cb(qr, "q_resid", il);
        ggml_tensor* q = ggml_reshape_3d(ctx, mm(m.wl(il, "attn_q_b.weight"), qr), g.head_k_mla, nh, T);
        dsa_write(il, x, p0, T);

        // absorb wk_b: the cache holds only the latent
        ggml_tensor* qa = mm(m.wl(il, "attn_k_b.weight"), ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)));
        qa = ggml_cont(ctx, ggml_permute(ctx, qa, 0, 2, 1, 3));   // [kv_lora, n_head, T]
        cb(qa, "q_absorbed", il);

        const int64_t n_pool = (p0 + T) / kp;
        const int64_t n_top = std::min<int64_t>(n_pool, g.idx_top_k / kp);
        const int64_t n_tail = g.idx_select_tail ? kp - 1 : 0;
        // + the null cell: a token with no visible cell reads 0.  dsa_mm pads the cells to kMmMinK with more null cells
        // (masked): below it the second product would switch to ggml-metal's f32 matvec, and a cold and a warm prompt,
        // sliced differently, would gather the same token with different kernels
        const int64_t n_core = kp * n_top + n_tail + 1;
#ifdef MAYA_METAL_SPARSE_MLA
        // ggml_sparse_mla (third_party/ggml/metal-sparse-mla.patch): its Metal kernel (no ggml-cpu one) takes these
        // head sizes and an F16 or F32 latent
        const bool fused = m.o.use_gpu && m.o.dsa_mm && m.o.dsa_kernel && T >= kSparseMinT &&
                           (kv == 64 || kv == 128 || kv == 256 || kv == 512);
#else
        const bool fused = false;
#endif
        const int64_t n_sel = m.o.dsa_mm && !fused ? std::max(n_core, kMmMinK) : n_core;
        const int64_t n_fill = n_sel - kp * n_top;   // the host-known cells: tail, null, padding

        ggml_tensor* iq = ggml_reshape_3d(ctx, mm(m.wl(il, "indexer.attn_q_b.weight"), qr), kd, ih, T);
        ggml_tensor* iw = ggml_scale(ctx, mm(m.wl(il, "indexer.proj.weight"), x), 1.0f / std::sqrt((float) (kd * ih)));
        cb(iq, "indexer_q", il);
        cb(iw, "indexer_weights", il);
        ggml_tensor* pooled_all = n_pool > 0
            ? ggml_view_3d(ctx, s.pooled, kd, n_pool, 1, s.pooled->nb[1], s.pooled->nb[1] * n_pool, 0) : nullptr;

        // the chunks' outputs, merged as they come like a binary counter (two of the same size become one): a running
        // concat copies everything before each chunk again, and holding every chunk's output to the end fragments the
        // graph allocator (a 2048-token slice in 4-token chunks ran out of its free blocks)
        std::vector<std::pair<ggml_tensor*, int>> outs;   // (output, merges)
        auto push = [&](ggml_tensor* o) {
            outs.push_back({o, 0});
            while (outs.size() >= 2 && outs[outs.size() - 1].second == outs[outs.size() - 2].second) {
                auto [b, lb] = outs.back();
                outs.pop_back();
                outs.back() = {ggml_concat(ctx, outs.back().first, b, 3), lb + 1};
            }
        };
        int chunk = std::max(1, m.o.attn_chunk);
        if (fused)
            chunk = (int) std::min<int64_t>(
                T, std::max<int64_t>(chunk, (int64_t) (kSparseScoreBytes / ((size_t) std::max<int64_t>(1, n_pool) * ih * 4))));
        for (int c0 = 0; c0 < T; c0 += chunk) {
            const int tc = std::min(chunk, T - c0);
            // host-known parts of the selection: the tail cells, the null cell, the live/dead mask
            std::vector<int32_t> tail((size_t) n_fill * tc);
            std::vector<float> mask((size_t) n_sel * tc, -INFINITY);
            std::vector<float> pmask;   // only when some token of the chunk cannot see every pool yet
            if (n_pool > 0 && (p0 + c0 + 1) / kp < n_pool) pmask.assign((size_t) n_pool * tc, 0.0f);
            const int64_t null_cell = m.n_ctx;
            for (int t = 0; t < tc; ++t) {
                const int64_t p = p0 + c0 + t;
                const int64_t nv = (p + 1) / kp;                 // pools whose last member is <= p
                const int64_t live_pools = std::min(nv, n_top);
                float* mk = mask.data() + (size_t) n_sel * t;
                for (int64_t i = 0; i < live_pools * kp; ++i) mk[i] = 0.0f;
                const int64_t tail_n = n_tail > 0 ? (p + 1) - nv * kp : 0;
                int32_t* tl = tail.data() + (size_t) n_fill * t;
                for (int64_t i = 0; i < n_tail; ++i) {
                    tl[i] = (int32_t) (i < tail_n ? nv * kp + i : null_cell);
                    if (i < tail_n) mk[kp * n_top + i] = 0.0f;
                }
                for (int64_t i = n_tail; i < n_fill; ++i) tl[i] = (int32_t) null_cell;
                if (live_pools == 0 && tail_n == 0) mk[kp * n_top + n_tail] = 0.0f;
                if (!pmask.empty())
                    for (int64_t q2 = nv; q2 < n_pool; ++q2) pmask[(size_t) n_pool * t + (size_t) q2] = -INFINITY;
            }
            ggml_tensor* tail_t = input(GGML_TYPE_I32, n_fill, tc, tail.data(), tail.size() * 4);
            ggml_tensor* mask_t = input(GGML_TYPE_F32, n_sel, 1, mask.data(), mask.size() * 4, 1, tc);

            ggml_tensor* idx = tail_t;
            if (n_top > 0) {
                ggml_tensor* iqc = ggml_view_3d(ctx, iq, kd, ih, tc, iq->nb[1], iq->nb[2], (size_t) c0 * iq->nb[2]);
                ggml_tensor* kq;
                if (m.o.dsa_mm) {   // the indexer heads are the columns: the same kernel whatever tc is
                    kq = ggml_mul_mat(ctx, pooled_all, iqc);                                       // [n_pool, ih, tc]
                    kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 0, 1, 3));                        // [ih, tc, n_pool]
                } else {
                    kq = mm(pooled_all, ggml_permute(ctx, iqc, 0, 2, 1, 3));                       // [n_pool, tc, ih]
                    kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 1, 0, 3));                        // [ih, tc, n_pool]
                }
                ggml_tensor* sc = ggml_mul(ctx, ggml_relu(ctx, kq), cols(iw, tc, c0));
                sc = ggml_cont(ctx, ggml_permute(ctx, ggml_sum_rows(ctx, sc), 2, 1, 0, 3));          // [n_pool, tc, 1]
                sc = ggml_reshape_2d(ctx, sc, n_pool, tc);
                if (!pmask.empty())
                    sc = ggml_add(ctx, sc, input(GGML_TYPE_F32, n_pool, tc, pmask.data(), pmask.size() * 4));
                if (c0 == 0) cb(sc, "indexer_score", il);
                ggml_tensor* top = ggml_cont(ctx, ggml_argsort_top_k(ctx, sc, (int) n_top));   // [n_top, tc], best first
                ggml_tensor* sel = ggml_get_rows(ctx, m.pool_tab, ggml_reshape_1d(ctx, top, n_top * tc));
                sel = ggml_reshape_2d(ctx, sel, kp * n_top, tc);
                idx = ggml_concat(ctx, sel, tail_t, 0);
            }
#ifdef MAYA_METAL_SPARSE_MLA
            if (fused) {   // every head of a token reads its selected cache rows in place, once per 8 heads
                ggml_tensor* qz = ggml_view_3d(ctx, qa, kv, nh, tc, qa->nb[1], qa->nb[2], (size_t) c0 * qa->nb[2]);
                ggml_tensor* o = ggml_sparse_mla(ctx, qz, s.lat, idx, ggml_reshape_2d(ctx, mask_t, n_sel, tc), kq_scale);
                push(ggml_reshape_4d(ctx, o, kv, nh, 1, tc));   // [kv, nh, 1, tc]
                continue;
            }
#endif
            ggml_tensor* kg = ggml_get_rows(ctx, s.lat, ggml_reshape_1d(ctx, idx, n_sel * tc));   // [kv, n_sel*tc]
            kg = ggml_reshape_4d(ctx, kg, kv, n_sel, 1, tc);
            ggml_tensor* qc = ggml_view_3d(ctx, qa, kv, nh, tc, qa->nb[1], qa->nb[2], (size_t) c0 * qa->nb[2]);
            ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, kg, 1, 0, 2, 3));               // [n_sel, kv, 1, tc]
            ggml_tensor* o;
            if (m.o.dsa_mm) {   // the heads are the columns: one mul_mm per token, reading the gathered rows once
                // (the cont stays: mul_mm reading the slice in place gave a prompt other bits than streaming)
                qc = ggml_reshape_4d(ctx, ggml_cont(ctx, qc), kv, nh, 1, tc);
                ggml_tensor* kq = ggml_mul_mat(ctx, kg, qc);                             // [n_sel, nh, 1, tc]
                ggml_prec_set_acc(kq, GGML_PREC_F32);
                kq = ggml_reshape_4d(ctx, kq, n_sel, 1, nh, tc);                         // the mask broadcasts over nh
                kq = ggml_soft_max_ext(ctx, kq, mask_t, kq_scale, 0.0f);
                o = ggml_mul_mat(ctx, vt, ggml_reshape_4d(ctx, kq, n_sel, nh, 1, tc));   // [kv, nh, 1, tc]
            } else {
                qc = ggml_reshape_4d(ctx, ggml_cont(ctx, qc), kv, 1, nh, tc);
                ggml_tensor* kq = mm(kg, qc);                                            // [n_sel, 1, nh, tc]
                ggml_prec_set_acc(kq, GGML_PREC_F32);
                kq = ggml_soft_max_ext(ctx, kq, mask_t, kq_scale, 0.0f);
                o = mm(vt, kq);                                                          // [kv, 1, nh, tc]
            }
            push(o);
        }
        while (outs.size() >= 2) {   // the unequal rest, newest first
            ggml_tensor* b = outs.back().first;
            outs.pop_back();
            outs.back().first = ggml_concat(ctx, outs.back().first, b, 3);
        }
        ggml_tensor* o = ggml_reshape_3d(ctx, outs[0].first, kv, nh, T);
        o = mm(m.wl(il, "attn_v_b.weight"), ggml_permute(ctx, o, 0, 2, 1, 3));   // [v, T, nh]
        o = ggml_cont_2d(ctx, ggml_permute(ctx, o, 0, 2, 1, 3), (int64_t) g.head_v_mla * nh, T);
        cb(o, "kqv_out", il);
        ggml_tensor* y = mm(m.wl(il, "attn_output.weight"), o);
        cb(y, "attn_out", il);
        return y;
    }

    ggml_tensor* swiglu(int il, ggml_tensor* x, const char* gate, const char* up, const char* down, float limit) {
        ggml_tensor* a = mm(m.wl(il, gate), x);
        ggml_tensor* b = mm(m.wl(il, up), x);
        ggml_tensor* h = limit > 1e-6f ? ggml_swiglu_clamp(ctx, a, b, limit) : ggml_swiglu_split(ctx, a, b);
        return mm(m.wl(il, down), h);
    }

    // layer il.  A dense layer, or a MoE layer whose experts are all resident, finishes here (returns the next R);
    // otherwise a MoE layer routes, runs its shared expert, writes the carries and returns nullptr (*sel = the top-k
    // the host reads back to make the experts resident).
    ggml_tensor* layer(int il, ggml_tensor* R, ggml_tensor** sel) {
        if (!m.feat_slot.empty() && m.feat_slot[(size_t) il] >= 0) {   // DFlash2: the streams' mean entering il
            ggml_tensor* acc = ggml_view_2d(ctx, R, g.n_embd, T, R->nb[2], 0);
            for (int s = 1; s < g.hc; ++s)
                acc = ggml_add(ctx, acc, ggml_view_2d(ctx, R, g.n_embd, T, R->nb[2], (size_t) s * R->nb[1]));
            expand(ggml_cpy(ctx, ggml_scale(ctx, acc, 1.0f / g.hc),
                            ggml_view_2d(ctx, m.feat, g.n_embd, T, m.feat->nb[1],
                                         (size_t) m.feat_slot[(size_t) il] * g.n_embd * sizeof(float))));
        }
        ggml_tensor *post, *comb;
        ggml_tensor* x = hc_pre(R, il, false, &post, &comb);
        x = rms(x, m.wl(il, "attn_norm.weight"));
        cb(x, "attn_norm", il);
        x = g.recurrent[(size_t) il] ? kda(il, x) : dsa(il, x);
        R = hc_post(x, R, post, comb);
        cb(R, "hc_attn_post", il);

        x = rms(hc_pre(R, il, true, &post, &comb), m.wl(il, "ffn_norm.weight"));
        cb(x, "ffn_norm", il);
        if (il < g.n_dense_lead) {
            x = swiglu(il, x, "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", g.clamp_shexp[(size_t) il]);
            cb(x, "ffn_out", il);
            R = hc_post(x, R, post, comb);
            cb(R, "l_out", il);
            return R;
        }
        ggml_tensor *top, *wts, *sh;
        route(il, x, &top, &wts, &sh);
        if (m.pool.all_resident()) {   // every expert has its fixed slot: the lookup is on the GPU, no cut
            ggml_tensor* out = experts_out(il, slot_ids(il, top), x, wts, sh);
            R = hc_post(out, R, post, comb);
            cb(R, "l_out", il);
            return R;
        }
        expand(ggml_cpy(ctx, x, cols(m.c_x, T)));
        expand(ggml_cpy(ctx, wts, cols3(m.c_w, T)));
        expand(ggml_cpy(ctx, sh, cols(m.c_shexp, T)));
        expand(ggml_cpy(ctx, R, cols3(m.c_res, T)));
        expand(ggml_cpy(ctx, post, cols(m.c_post, T)));
        expand(ggml_cpy(ctx, comb, cols3(m.c_comb, T)));
        ggml_set_output(top);
        expand(top);
        *sel = top;
        return nullptr;
    }

    // the router (sigmoid, the selection bias, top-k, normalized and scaled weights) and the shared expert of MoE
    // layer il on x: top [n_used, T], wts [1, n_used, T], sh [n_embd, T]
    void route(int il, ggml_tensor* x, ggml_tensor** top_out, ggml_tensor** wts_out, ggml_tensor** sh_out) {
        ggml_tensor* logits = mm(m.wl(il, "ffn_gate_inp.weight"), x);   // [n_expert, T]
        cb(logits, "ffn_moe_logits", il);
        ggml_tensor* probs = ggml_sigmoid(ctx, logits);
        ggml_tensor* biased = ggml_add(ctx, probs, m.wl(il, "exp_probs_b.bias"));
        cb(probs, "ffn_moe_probs", il);
        cb(biased, "ffn_moe_probs_biased", il);
        ggml_tensor* top = ggml_cont(ctx, ggml_argsort_top_k(ctx, biased, g.n_expert_used));   // [n_used, T]
        ggml_tensor* wts = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, g.n_expert, T), top);   // [1, n_used, T]
        if (g.expert_weights_norm) {
            ggml_tensor* w2 = ggml_reshape_2d(ctx, wts, g.n_expert_used, T);
            ggml_tensor* sum = ggml_clamp(ctx, ggml_sum_rows(ctx, w2), 6.103515625e-5f, INFINITY);
            wts = ggml_reshape_3d(ctx, ggml_div(ctx, w2, sum), 1, g.n_expert_used, T);
        }
        if (g.expert_weights_scale != 0.0f && g.expert_weights_scale != 1.0f)
            wts = ggml_scale(ctx, wts, g.expert_weights_scale);
        ggml_tensor* sh = swiglu(il, x, "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                                 g.clamp_shexp[(size_t) il]);
        cb(sh, "ffn_shexp", il);
        *top_out = top;
        *wts_out = wts;
        *sh_out = sh;
    }

    // all resident: expert e of layer il sits in slot slot_base(il) + e
    ggml_tensor* slot_ids(int il, ggml_tensor* top) {
        ggml_tensor* f = ggml_scale_bias(ctx, ggml_cast(ctx, top, GGML_TYPE_F32), 1.0f, (float) m.pool.slot_base(il));
        return ggml_cast(ctx, f, GGML_TYPE_I32);
    }

    // the routed experts (ids: their slots) weighted by w, plus the shared expert's sh
    ggml_tensor* experts_out(int il, ggml_tensor* ids, ggml_tensor* x, ggml_tensor* w, ggml_tensor* sh) {
        x = ggml_reshape_3d(ctx, x, g.n_embd, 1, T);
        ggml_tensor* up = mm_id(m.pool.tensor(il, 1), x, ids);     // [n_ff, n_used, T]
        ggml_tensor* gate = mm_id(m.pool.tensor(il, 0), x, ids);
        const float limit = g.clamp_exp[(size_t) il];
        ggml_tensor* h = limit > 1e-6f ? ggml_swiglu_clamp(ctx, gate, up, limit) : ggml_swiglu_split(ctx, gate, up);
        ggml_tensor* down = mm_id(m.pool.tensor(il, 2), h, ids);  // [n_embd, n_used, T]
        down = ggml_mul(ctx, down, w);
        ggml_tensor* out = ggml_view_2d(ctx, down, g.n_embd, T, down->nb[2], 0);
        for (int i = 1; i < g.n_expert_used; ++i)
            out = ggml_add(ctx, out, ggml_view_2d(ctx, down, g.n_embd, T, down->nb[2], (size_t) i * down->nb[1]));
        cb(out, "ffn_moe_out", il);
        out = ggml_add(ctx, out, sh);
        cb(out, "ffn_out", il);
        return out;
    }

    // the routed experts of MoE layer il from the previous graph's carries, combined into the next residual streams
    ggml_tensor* moe(int il, ggml_tensor* ids, ggml_tensor* x, ggml_tensor* w, ggml_tensor* sh, ggml_tensor* R,
                     ggml_tensor* post, ggml_tensor* comb) {
        R = hc_post(experts_out(il, ids, x, w, sh), R, post, comb);
        cb(R, "l_out", il);
        return R;
    }

    // the same from the carries the previous graph wrote, with the slots the host made resident
    ggml_tensor* experts(int il, const std::vector<int32_t>& slots) {
        ggml_tensor* ids = input(GGML_TYPE_I32, g.n_expert_used, T, slots.data(), slots.size() * 4);
        return moe(il, ids, cols(m.c_x, T), cols3(m.c_w, T), cols(m.c_shexp, T), cols3(m.c_res, T),
                   cols(m.c_post, T), cols3(m.c_comb, T));
    }

    // the NextN block's input: eh_proj . [enorm(e) | hnorm(h)] (e the next token's embedding, h a hidden state)
    ggml_tensor* mtp_in(ggml_tensor* e, ggml_tensor* h) {
        const int il = g.n_layers;
        return mm(m.wl(il, "nextn.eh_proj.weight"),
                  ggml_concat(ctx, rms(e, m.wl(il, "nextn.enorm.weight")), rms(h, m.wl(il, "nextn.hnorm.weight")), 0));
    }

    // the head over the last row, or every row of a window; the normed hidden state goes to m.hid as well.  With MTP
    // the block's cache gets the entries this slice makes known: position p pairs the hidden state of p with the
    // embedding of the token at p + 1 - from p0 - 1 (the previous forward's last row) to p0 + T - 2.
    ggml_tensor* head(ggml_tensor* R) {
        const int hc = g.hc;
        const int rows = (window || g.mtp) ? T : 1;
        ggml_tensor* last = ggml_view_3d(ctx, R, g.n_embd, hc, rows, R->nb[1], R->nb[2],
                                         (size_t) (T - rows) * R->nb[2]);
        ggml_tensor* acc = ggml_view_2d(ctx, last, g.n_embd, rows, last->nb[2], 0);
        for (int s = 1; s < hc; ++s)
            acc = ggml_add(ctx, acc, ggml_view_2d(ctx, last, g.n_embd, rows, last->nb[2], (size_t) s * last->nb[1]));
        ggml_tensor* h = rms(ggml_scale(ctx, acc, 1.0f / hc), m.w("output_norm.weight"));   // [n_embd, rows]
        if (g.mtp) {
            const bool prev = p0 > 0 && m.hx_pos == p0 - 1;
            const int64_t nf = T - 1 + (prev ? 1 : 0);
            if (nf > 0) {
                ggml_tensor* e = input(GGML_TYPE_F32, g.n_embd, T, emb, sizeof(float) * (size_t) g.n_embd * T);
                ggml_tensor* hs = T > 1 ? ggml_view_2d(ctx, h, g.n_embd, T - 1, h->nb[1], 0) : nullptr;
                if (prev) {
                    ggml_tensor* hp = input(GGML_TYPE_F32, g.n_embd, 1, m.hid_host.data() + (size_t) m.hx_col * g.n_embd,
                                            sizeof(float) * (size_t) g.n_embd);
                    hs = hs ? ggml_concat(ctx, hp, hs, 1) : hp;
                } else {
                    e = ggml_view_2d(ctx, e, g.n_embd, T - 1, e->nb[1], e->nb[1]);
                }
                const int il = g.n_layers;
                dsa_write(il, rms(mtp_in(e, hs), m.wl(il, "attn_norm.weight")), prev ? p0 - 1 : p0, nf);
            }
        }
        const int hrows = window ? T : 1;
        ggml_tensor* hout = ggml_view_2d(ctx, h, g.n_embd, hrows, h->nb[1], (size_t) (rows - hrows) * h->nb[1]);
        expand(ggml_cpy(ctx, hout, ggml_view_2d(ctx, m.hid, g.n_embd, hrows, m.hid->nb[1], 0)));
        m.hid_rows = hrows;
        ggml_tensor* logits = mm(m.w("output.weight"), hout);
        ggml_set_output(logits);
        expand(logits);
        return logits;
    }
};

}  // namespace

// the layers' fixed part plus what grows with T: the per-chunk gathers, the float-expert groups (a fixed budget
// aborted in ggml_new_object once --ubatch outgrew it)
size_t GlmMetal::Impl::graph_size(int T) const {
    const size_t dsa = (size_t) std::count(g.recurrent.begin(), g.recurrent.end(), 0);
    const size_t moe = g.recurrent.size() - (size_t) g.n_dense_lead;
    const size_t chunk = (size_t) std::max(1, o.attn_chunk);
    return kGraphNodes + dsa * (((size_t) T + chunk - 1) / chunk) * kChunkTensors +
           moe * (((size_t) T + kFloatGroup - 1) / kFloatGroup) * kGroupTensors;
}

bool GlmMetal::Impl::run_slice(const int32_t* tok, int T, std::vector<float>& logits, std::string& err, bool window) {
    std::vector<float> emb;
    if (!embed(tok, T, emb, err)) return false;
    const size_t nodes = graph_size(T);
    const size_t mem = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false);
    if (arena.size() < mem) arena.resize(mem);
    std::vector<int32_t> slots;
    int moe = -1;   // the MoE layer whose experts open the next graph
    for (;;) {
        const auto th = std::chrono::steady_clock::now();
        ggml_init_params ip{mem, arena.data(), true};
        ggml_context* ctx = ggml_init(ip);
        Builder b{*this, g, ctx, ggml_new_graph_custom(ctx, nodes, false), T, pos, window, emb.data(), {}, {}};
        ggml_tensor* R;
        int il;
        if (moe < 0) {
            ggml_tensor* x = b.input(GGML_TYPE_F32, g.n_embd, T, emb.data(), emb.size() * 4);
            R = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, x, g.n_embd, 1, T), g.n_embd, g.hc, T, 1);
            b.cb(R, "hc_init", -1);
            il = 0;
        } else {
            R = b.experts(moe, slots);
            il = moe + 1;
        }
        ggml_tensor* sel = nullptr;
        for (; il < g.n_layers && R; ++il) {
            ggml_tensor* next = b.layer(il, R, &sel);
            if (!next) break;
            R = next;
        }
        ggml_tensor* out = sel ? nullptr : b.head(R);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, b.gf)) {
            ggml_free(ctx);
            err = "cannot allocate the compute graph";
            return false;
        }
        if (gpu && std::getenv("MAYA_METAL_SPLITS")) {   // which nodes did not land on the GPU (debug)
            std::fprintf(stderr, "graph: %d nodes, %d splits\n", ggml_graph_n_nodes(b.gf), ggml_backend_sched_get_n_splits(sched));
            for (int i = 0; i < ggml_graph_n_nodes(b.gf); ++i) {
                ggml_tensor* t = ggml_graph_node(b.gf, i);
                if (ggml_backend_sched_get_tensor_backend(sched, t) != gpu)
                    std::fprintf(stderr, "  cpu: %s %s %s [%lld %lld %lld %lld]\n", ggml_op_desc(t), ggml_type_name(t->type),
                                 t->src[0] ? ggml_type_name(t->src[0]->type) : "-", (long long) t->ne[0], (long long) t->ne[1],
                                 (long long) t->ne[2], (long long) t->ne[3]);
            }
        }
        for (const Upload& u : b.uploads) ggml_backend_tensor_set(u.t, u.data.data(), 0, u.data.size());
        const auto t0 = std::chrono::steady_clock::now();
        st.host_seconds += std::chrono::duration<double>(t0 - th).count();
        ++st.graphs;
            if (ggml_backend_sched_graph_compute(sched, b.gf) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx);
            err = "the compute graph failed";
            return false;
        }
        st.gpu_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (std::getenv("MAYA_METAL_PROFILE"))
            std::fprintf(stderr, "slice T=%d%s nodes %d tensors %zu/%zu %.1f ms\n", T, window ? " window" : "",
                         ggml_graph_n_nodes(b.gf), ggml_used_mem(ctx) / ggml_tensor_overhead(), nodes,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        for (const auto& [name, t] : b.dumps) {
            std::vector<float> v((size_t) ggml_nelements(t));
            if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t))
                ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
            else
                continue;
            const std::string path = o.dump_dir + "/" + name + ".f32";
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(v.data(), 4, v.size(), f);
                std::fclose(f);
            }
        }
        if (out) {
            hid_host.resize((size_t) g.n_embd * hid_rows);
            ggml_backend_tensor_get(hid, hid_host.data(), 0, hid_host.size() * sizeof(float));
            hx_pos = pos + T - 1;
            hx_col = hid_rows - 1;
            logits.resize((size_t) g.n_vocab * (window ? T : 1));
            ggml_backend_tensor_get(out, logits.data(), 0, logits.size() * 4);
            ggml_free(ctx);
            // (after the reads: the drafter's graph takes the compute buffer the logits sit in)
            return !df || dflash_inject(pos, T, err);
        }
        std::vector<int32_t> experts((size_t) g.n_expert_used * T);
        ggml_backend_tensor_get(sel, experts.data(), 0, experts.size() * 4);
        ggml_free(ctx);
        moe = il;
        slots.resize(experts.size());
        if (!pool.ensure(moe, experts.data(), experts.size(), slots.data(), err)) return false;
    }
}

// ---------------------------------------------------------------- public API

GlmMetal::GlmMetal() : p_(new Impl) {}
GlmMetal::~GlmMetal() = default;

void GlmMetal::set_options(const GlmOptions& o) { p_->o = o; }

bool GlmMetal::load(const std::string& path, int64_t max_ctx, std::string& err) {
    Impl& m = *p_;
    if (max_ctx < 1 || m.o.n_ubatch < 1 || m.o.attn_chunk < 1) {
        err = "the context, the slice and the attention chunk need at least one token";
        return false;
    }
    m.n_ctx = max_ctx;
    if (!m.open_files(path, err) || !m.read_geometry(err) || !m.init_backends(err) || !m.load_weights(err) ||
        !m.init_state(err) || (!m.o.dflash.empty() && !m.load_dflash(err)) || !m.init_experts(err))
        return false;
    reset();
    return true;
}

bool GlmMetal::forward(const std::vector<int32_t>& tokens, std::vector<float>& logits, std::string& err) {
    Impl& m = *p_;
    m.win_T = 0;
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
    m.st.expert_hits = m.pool.hits;
    m.st.expert_misses = m.pool.misses;
    m.st.bytes_read = m.pool.bytes_read;
    m.st.io_seconds = m.pool.io_seconds;
    return true;
}

bool GlmMetal::forward_window(const std::vector<int32_t>& tokens, std::vector<float>& logits, std::string& err) {
    Impl& m = *p_;
    const int T = (int) tokens.size();
    if (T < 1 || T > m.o.max_window || T > m.o.n_ubatch) {
        err = "a verify window takes 1 to " + std::to_string(std::min(m.o.max_window, m.o.n_ubatch)) + " tokens";
        return false;
    }
    if (m.pos + T > m.n_ctx) {
        err = "context full (" + std::to_string(m.n_ctx) + " tokens)";
        return false;
    }
    m.win_p0 = m.pos;
    m.win_T = 0;
    if (!m.run_slice(tokens.data(), T, logits, err, true)) return false;
    m.win_T = T;
    m.pos += T;
    m.st.tokens += (uint64_t) T;
    m.st.expert_hits = m.pool.hits;
    m.st.expert_misses = m.pool.misses;
    m.st.bytes_read = m.pool.bytes_read;
    m.st.io_seconds = m.pool.io_seconds;
    return true;
}

bool GlmMetal::accept(int n, std::string& err) {
    Impl& m = *p_;
    const int T = m.win_T;
    if (T == 0 || n < 1 || n > T) {
        err = "accept(" + std::to_string(n) + ") without a window of at least that many rows";
        return false;
    }
    m.win_T = 0;
    m.pos = m.win_p0 + n;
    m.hx_pos = m.pos - 1;      // (hid_host holds every row of the window)
    m.dflash_forget(m.pos);
    m.hx_col = n - 1;
    if (n == T) return true;   // the state is already the last row's
    // the recurrent state after row n - 1: conv history columns n .. n + dc - 1, the GDN snapshot T - n back
    const size_t mem = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false);
    if (m.arena.size() < mem) m.arena.resize(mem);
    ggml_init_params ip{mem, m.arena.data(), true};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 512, false);
    const int64_t dc = m.g.d_conv - 1;
    for (const LayerState& s : m.ls) {
        if (!s.conv) continue;
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_3d(ctx, s.conv_win, dc, s.conv->ne[1], 3, s.conv_win->nb[1],
                                                                s.conv_win->nb[2], (size_t) n * s.conv_win->nb[0]),
                                               s.conv));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_1d(ctx, s.ssm_win, s.ssm_win->ne[0],
                                                                (size_t) (T - n) * s.ssm_win->nb[1]),
                                               ggml_view_1d(ctx, s.ssm, ggml_nelements(s.ssm), 0)));
    }
    ggml_backend_sched_reset(m.sched);
    const bool ok = ggml_backend_sched_alloc_graph(m.sched, gf) &&
                    ggml_backend_sched_graph_compute(m.sched, gf) == GGML_STATUS_SUCCESS;
    ggml_free(ctx);
    if (!ok) err = "the window's rollback failed";
    return ok;
}

void GlmMetal::hidden(std::vector<float>& out) const {
    const Impl& m = *p_;
    out.resize((size_t) m.g.n_embd * m.hid_rows);
    if (!out.empty()) ggml_backend_tensor_get(m.hid, out.data(), 0, out.size() * sizeof(float));
}

bool GlmMetal::has_mtp() const { return p_->g.mtp; }
int64_t GlmMetal::mtp_hidden_pos() const { return p_->hx_pos; }

bool GlmMetal::mtp_step(int32_t token, int64_t pos, bool from_trunk, int32_t& draft, std::string& err) {
    Impl& m = *p_;
    const GlmGeometry& g = m.g;
    const int il = g.n_layers;
    if (!g.mtp || pos < 0 || pos + 1 >= m.n_ctx || (from_trunk && m.hx_pos != pos) || (!from_trunk && m.mtp_hx.empty())) {
        err = "no MTP input for position " + std::to_string(pos);
        return false;
    }
    std::vector<float> e;
    if (!m.embed_raw(&token, 1, e, err)) return false;
    const std::vector<float> h = from_trunk ? std::vector<float>(m.hid_host.begin() + (long) m.hx_col * g.n_embd,
                                                                 m.hid_host.begin() + (long) (m.hx_col + 1) * g.n_embd)
                                            : m.mtp_hx;
    const size_t mem = ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false);
    if (m.arena.size() < mem) m.arena.resize(mem);
    std::vector<int32_t> slots;
    for (int part = 0; part < 2; ++part) {
        ggml_init_params ip{mem, m.arena.data(), true};
        ggml_context* ctx = ggml_init(ip);
        Builder b{m, g, ctx, ggml_new_graph_custom(ctx, kGraphNodes, false), 1, pos, false, e.data(), {}, {}};
        ggml_tensor *x, *out = nullptr, *top = nullptr;
        if (part == 0) {   // the attention half, the router
            x = b.mtp_in(b.input(GGML_TYPE_F32, g.n_embd, 1, e.data(), e.size() * 4),
                         b.input(GGML_TYPE_F32, g.n_embd, 1, h.data(), h.size() * 4));
            x = ggml_add(ctx, x, b.dsa(il, b.rms(x, m.wl(il, "attn_norm.weight"))));
            ggml_tensor* f = b.rms(x, m.wl(il, "ffn_norm.weight"));
            ggml_tensor *wts, *sh;
            b.route(il, f, &top, &wts, &sh);
            if (m.pool.all_resident()) {
                out = ggml_add(ctx, x, b.experts_out(il, b.slot_ids(il, top), f, wts, sh));
            } else {
                b.expand(ggml_cpy(ctx, f, b.cols(m.c_x, 1)));
                b.expand(ggml_cpy(ctx, wts, b.cols3(m.c_w, 1)));
                b.expand(ggml_cpy(ctx, sh, b.cols(m.c_shexp, 1)));
                b.expand(ggml_cpy(ctx, x, ggml_view_1d(ctx, m.c_mres, g.n_embd, 0)));
                ggml_set_output(top);
                b.expand(top);
            }
        } else {           // the experts the host made resident
            ggml_tensor* ids = b.input(GGML_TYPE_I32, g.n_expert_used, 1, slots.data(), slots.size() * 4);
            out = b.experts_out(il, ids, b.cols(m.c_x, 1), b.cols3(m.c_w, 1), b.cols(m.c_shexp, 1));
            out = ggml_add(ctx, ggml_reshape_2d(ctx, m.c_mres, g.n_embd, 1), out);
        }
        ggml_tensor *hx = nullptr, *arg = nullptr;
        if (out) {
            hx = b.rms(out, m.wl(il, "nextn.shared_head_norm.weight"));
            arg = ggml_argmax(ctx, b.mm(m.w("output.weight"), hx));
            ggml_set_output(hx);
            ggml_set_output(arg);
            b.expand(hx);
            b.expand(arg);
        }
        std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> up;
        for (Upload& u : b.uploads) up.push_back({u.t, std::move(u.data)});
        if (!m.compute(b.gf, up, err)) {
            ggml_free(ctx);
            return false;
        }
        if (out) {
            m.mtp_hx.resize((size_t) g.n_embd);
            ggml_backend_tensor_get(hx, m.mtp_hx.data(), 0, m.mtp_hx.size() * sizeof(float));
            ggml_backend_tensor_get(arg, &draft, 0, sizeof(int32_t));
            ggml_free(ctx);
            return true;
        }
        std::vector<int32_t> ex((size_t) g.n_expert_used);
        ggml_backend_tensor_get(top, ex.data(), 0, ex.size() * sizeof(int32_t));
        ggml_free(ctx);
        slots.resize(ex.size());
        if (!m.pool.ensure(il, ex.data(), ex.size(), slots.data(), err)) return false;
    }
    return false;
}

bool GlmMetal::Impl::compute(ggml_cgraph* gf, const std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>>& up,
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

void GlmMetal::set_image_rows(std::vector<int64_t> pos, std::vector<float> rows) {
    p_->img_pos = std::move(pos);
    p_->img_rows = std::move(rows);
}

void GlmMetal::reset() {
    Impl& m = *p_;
    m.pos = 0;
    m.snap_pos = -1;
    m.win_T = 0;
    m.hx_pos = -1;
    m.dflash_forget(0);
    // the recurrent state starts from zero; the caches are written before a position can see them, except the
    // latent's null cell (the last row), which must read as zero
    for (const LayerState& s : m.ls) {
        for (ggml_tensor* t : {s.conv, s.ssm})
            if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        if (s.lat) ggml_backend_tensor_memset(s.lat, 0, (size_t) m.n_ctx * s.lat->nb[1], s.lat->nb[1]);
    }
}

void GlmMetal::snapshot_save() {
    Impl& m = *p_;
    for (const LayerState& s : m.ls)
        if (s.conv) {
            ggml_backend_tensor_copy(s.conv, s.conv_snap);
            ggml_backend_tensor_copy(s.ssm, s.ssm_snap);
        }
    m.snap_pos = m.pos;
    m.hx_snap = m.hx_pos;
    if (m.hx_pos >= 0)
        m.hid_snap.assign(m.hid_host.begin() + (long) m.hx_col * m.g.n_embd,
                          m.hid_host.begin() + (long) (m.hx_col + 1) * m.g.n_embd);
}

bool GlmMetal::snapshot_restore() {
    Impl& m = *p_;
    if (m.snap_pos < 0) return false;
    for (const LayerState& s : m.ls)
        if (s.conv) {
            ggml_backend_tensor_copy(s.conv_snap, s.conv);
            ggml_backend_tensor_copy(s.ssm_snap, s.ssm);
        }
    m.pos = m.snap_pos;
    m.hx_pos = m.hx_snap;
    m.hx_col = 0;
    m.dflash_forget(m.pos);
    m.hid_host = m.hid_snap;
    m.win_T = 0;
    return true;
}

int64_t GlmMetal::snapshot_position() const { return p_->snap_pos; }

size_t GlmMetal::grow_pool() {
    Impl& m = *p_;
    if (m.pool.all_resident() || m.o.pool_slots > 0) return 0;
    const char* hv = std::getenv("MAYA_GLM_RAM_HEADROOM_GB");
    const double head_gb = hv && *hv ? std::atof(hv) : 6.0;
    const size_t avail = free_memory(), head = (size_t) (head_gb * (double) (1ull << 30));
    if (avail <= head + ((size_t) 1 << 30)) return 0;   // less than 1 GB to gain: not worth a NOTE
    const size_t got = m.pool.grow(avail - head);
    m.st.pool_slots = m.pool.slots_total();
    m.st.pool_bytes = m.pool.bytes();
    return got;
}

bool GlmMetal::save_profile(std::string& err) const {
    const Impl& m = *p_;
    const std::string& path = m.o.expert_profile_save;
    if (path.empty()) return true;
    const int E = m.g.n_expert, L = m.g.n_layers + 1;   // (+ the NextN block, as the shipped profile counts it)
    const std::vector<uint32_t>& use = m.pool.usage();
    std::vector<std::pair<int, int>> order;
    std::vector<char> taken((size_t) L * E, 0);
    for (int l = 0; l < m.g.n_layers + (m.g.mtp ? 1 : 0); ++l)
        for (int e = 0; e < E; ++e)
            if (use[(size_t) l * E + e] > 0) order.push_back({l, e});
    std::stable_sort(order.begin(), order.end(), [&](const auto& a, const auto& b) {
        return use[(size_t) a.first * E + a.second] > use[(size_t) b.first * E + b.second];
    });
    for (const auto& p : order) taken[(size_t) p.first * E + p.second] = 1;
    for (const auto& p : m.ranked)   // then the starting profile's order
        if (p.first < L && p.second < E && !taken[(size_t) p.first * E + p.second]) {
            taken[(size_t) p.first * E + p.second] = 1;
            order.push_back(p);
        }
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        err = "cannot write " + tmp;
        return false;
    }
    const uint32_t h[5] = {1, (uint32_t) L, (uint32_t) E, (uint32_t) order.size(), (uint32_t) order.size()};
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(h, 4, 5, f) == 5;
    for (const auto& p : order) {
        const uint16_t v[2] = {(uint16_t) p.first, (uint16_t) p.second};
        ok = ok && std::fwrite(v, 2, 2, f) == 2;
    }
    ok = std::fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        err = "cannot write " + path;
        return false;
    }
    return true;
}

bool GlmMetal::pool_holds_all() const { return p_->pool.holds_all(); }
const GlmGeometry& GlmMetal::geometry() const { return p_->g; }
int64_t GlmMetal::position() const { return p_->pos; }
int64_t GlmMetal::max_context() const { return p_->n_ctx; }
int GlmMetal::ubatch() const { return p_->o.n_ubatch; }
const GlmStats& GlmMetal::stats() const { return p_->st; }

std::string GlmMetal::describe() const {
    const Impl& m = *p_;
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "%s; dense %.2f GB, state %.2f GB (%lld ctx), expert pool %.2f GB in %zu slots%s",
                  m.gpu ? ggml_backend_name(m.gpu) : "CPU", m.st.dense_bytes / 1e9, m.st.state_bytes / 1e9,
                  (long long) m.n_ctx, m.st.pool_bytes / 1e9, m.st.pool_slots,
                  m.st.experts_all_resident ? " (every expert resident)" : "");
    return buf;
}

}  // namespace maya::metal
