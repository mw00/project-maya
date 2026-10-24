// src/metal/glm_metal_dflash.cpp - the DFlash2 drafter on maya-metal, as the CUDA engine runs it (src/core/glm_dflash.cu)
// and the pinned llama.cpp builds it (src/models/dflash.cpp):
//
//   inject  every trunk row: g = rms(fc . [the streams' mean entering each target layer]) * enc.output_norm, then
//           per drafter layer K = rope(rms(Wk g) * k_norm), V = Wv g into a KV ring (slot = position % ring)
//   draft   one non-causal pass over [anchor, mask x n] at positions pos .. pos + n, attending the ring's last
//           `window` positions and the block, each layer's attention and FFN wrapped in DFlash2's dynamic
//           convolution; the trunk's head gives each mask row's logits, the selector's lattice walk picks the drafts
//           (top-k candidates per row, a low-rank transition score from the previous pick) and their confidence
#include "glm_metal_impl.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace maya::metal {

namespace {

constexpr int kDraftNodes = 8192;

struct Up {
    ggml_tensor* t;
    std::vector<uint8_t> d;
};

ggml_tensor* input(ggml_context* ctx, std::vector<Up>& up, ggml_type type, int64_t ne0, int64_t ne1, const void* data,
                   size_t bytes) {
    ggml_tensor* t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_input(t);
    up.push_back({t, std::vector<uint8_t>((const uint8_t*) data, (const uint8_t*) data + bytes)});
    return t;
}

double gguf_num(const gguf_context* c, const char* key, double def) {
    const int64_t id = gguf_find_key(c, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(c, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(c, id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(c, id);
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(c, id);
        case GGUF_TYPE_BOOL: return gguf_get_val_bool(c, id) ? 1.0 : 0.0;
        case GGUF_TYPE_UINT64: return (double) gguf_get_val_u64(c, id);
        default: return def;
    }
}

// DFlash2's dynamic convolution over a block's rows (build_dflash2_conv): per channel group and tap a coefficient
// (dynamic [n_groups, taps, 2 sides, rows] + base [n_embd, taps, sides]) times the input `tap` rows back
ggml_tensor* dconv(ggml_context* ctx, const DFlash& d, ggml_tensor* x, ggml_tensor* dyn, ggml_tensor* base, int side) {
    const int64_t E = x->ne[0], B = x->ne[1], G = E / d.group;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    ggml_tensor* coef = ggml_reshape_4d(ctx, ggml_cont(ctx, dyn), G, d.taps, 2, B);
    coef = ggml_cont(ctx, ggml_view_3d(ctx, coef, G, d.taps, B, coef->nb[1], coef->nb[3], side * coef->nb[2]));
    coef = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, coef, 1, G, d.taps, B), d.group, G, d.taps, B);
    ggml_tensor* bs = ggml_reshape_4d(ctx, ggml_view_1d(ctx, base, E * d.taps, side * base->nb[2]), d.group, G, d.taps, 1);
    ggml_tensor* wt = ggml_add(ctx, coef, bs);
    ggml_tensor* out = nullptr;
    for (int64_t tap = 0; tap < d.taps; ++tap) {
        ggml_tensor* v = x;
        if (tap > 0) {
            ggml_tensor* z = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, std::min<int64_t>(tap, B)), 0.0f);
            v = tap < B ? ggml_concat(ctx, z, ggml_view_2d(ctx, x, E, B - tap, x->nb[1], 0), 1) : z;
        }
        ggml_tensor* w = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_view_4d(ctx, wt, d.group, G, 1, B, wt->nb[1], wt->nb[2],
                                                                          wt->nb[3], tap * wt->nb[2])),
                                         E, B);
        ggml_tensor* term = ggml_mul(ctx, w, v);
        out = out ? ggml_add(ctx, out, term) : term;
    }
    return out;
}

ggml_tensor* rms(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

ggml_tensor* rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos, const DFlash& d) {
    return ggml_rope_ext(ctx, x, pos, nullptr, d.head_dim, GGML_ROPE_TYPE_NEOX, 0, d.freq_base, 1.0f, 0.0f, 1.0f,
                         32.0f, 1.0f);
}

}  // namespace

DFlash::~DFlash() {
    if (sbuf) ggml_backend_buffer_free(sbuf);
    if (sctx) ggml_free(sctx);
    if (gguf) gguf_free(gguf);
    if (meta) ggml_free(meta);
    if (fd >= 0) close(fd);
}

bool GlmMetal::Impl::load_dflash(std::string& err) {
    df = std::make_unique<DFlash>();
    DFlash& d = *df;
    gguf_init_params ip{true, &d.meta};
    d.gguf = gguf_init_from_file(o.dflash.c_str(), ip);
    if (!d.gguf) {
        err = "cannot read the DFlash2 drafter " + o.dflash;
        return false;
    }
    const gguf_context* c = d.gguf;
    const int64_t a = gguf_find_key(c, "general.architecture");
    if (a < 0 || gguf_get_kv_type(c, a) != GGUF_TYPE_STRING || std::string(gguf_get_val_str(c, a)) != "dflash" || gguf_find_key(c, "dflash.selector_rank") < 0) {
        err = o.dflash + " is not a DFlash2 drafter";
        return false;
    }
    d.n_layer = (int) gguf_num(c, "dflash.block_count", 0);
    d.n_head = (int) gguf_num(c, "dflash.attention.head_count", 0);
    d.n_head_kv = (int) gguf_num(c, "dflash.attention.head_count_kv", d.n_head);
    d.head_dim = (int) gguf_num(c, "dflash.attention.key_length", 128);
    d.n_ff = (int) gguf_num(c, "dflash.feed_forward_length", 0);
    d.block = (int) gguf_num(c, "dflash.block_size", 16);
    d.taps = (int) gguf_num(c, "dflash.conv_kernel_size", 2);
    d.group = (int) gguf_num(c, "dflash.conv_group_size", 16);
    d.rank = (int) gguf_num(c, "dflash.selector_rank", 0);
    d.top_k = (int) gguf_num(c, "dflash.selector_top_k", 16);
    d.window = (int) gguf_num(c, "dflash.attention.sliding_window", 0);
    if (d.window <= 0) d.window = 4096;
    d.eps = (float) gguf_num(c, "dflash.attention.layer_norm_rms_epsilon", 1e-5);
    d.freq_base = (float) gguf_num(c, "dflash.rope.freq_base", 10000.0);
    d.mask_id = (int32_t) gguf_num(c, "tokenizer.ggml.mask_token_id", -1);
    if ((int) gguf_num(c, "dflash.embedding_length", 0) != g.n_embd || d.n_layer < 1 || d.head_dim != 128 ||
        gguf_num(c, "dflash.attention.causal", 0) != 0 || d.mask_id < 0 || d.mask_id >= g.n_vocab || d.top_k > 32 ||
        d.block < 2 || g.n_embd % d.group != 0) {
        err = o.dflash + " does not fit this model (width, head size, causality, mask token or block)";
        return false;
    }
    const int64_t tl = gguf_find_key(c, "dflash.target_layers");
    if (tl < 0 || gguf_get_arr_n(c, tl) == 0) {
        err = o.dflash + " names no target layers";
        return false;
    }
    feat_slot.assign((size_t) g.n_layers, -1);
    for (size_t i = 0; i < gguf_get_arr_n(c, tl); ++i) {
        const int l = ((const int32_t*) gguf_get_arr_data(c, tl))[i];
        if (l < 0 || l >= g.n_layers) {
            err = o.dflash + " reads layer " + std::to_string(l) + " of " + std::to_string(g.n_layers);
            return false;
        }
        d.targets.push_back(l);
        feat_slot[(size_t) l] = (int) i;
    }
    d.ring = d.window + 2 * 16;
    d.fd = open(o.dflash.c_str(), O_RDONLY);
    if (d.fd < 0) {
        err = "cannot open " + o.dflash;
        return false;
    }
    std::vector<std::pair<std::string, TensorLoc>> ts;
    const size_t base = gguf_get_data_offset(c);
    for (int64_t i = 0; i < gguf_get_n_tensors(c); ++i) {
        const char* name = gguf_get_tensor_name(c, i);
        ts.push_back({name, {d.fd, base + gguf_get_tensor_offset(c, i), ggml_get_tensor(d.meta, name)}});
    }
    if (!d.w.load(ts, gpu_dev, o.io_threads, err)) return false;
    std::vector<std::string> need = {"fc.weight", "enc.output_norm.weight", "output_norm.weight", "selector_hidden.weight",
                                     "selector_predecessor.weight", "selector_successor.weight"};
    for (int l = 0; l < d.n_layer; ++l)
        for (const char* s : {"attn_norm.weight", "ffn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                              "attn_conv_base", "ffn_conv_base", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                              "attn_output.weight", "attn_conv_proj.weight", "ffn_conv_proj.weight", "ffn_gate.weight",
                              "ffn_up.weight", "ffn_down.weight"})
            need.push_back("blk." + std::to_string(l) + "." + s);
    for (const std::string& n : need)
        if (!d.w.get(n)) {
            err = o.dflash + " has no " + n;
            return false;
        }
    // the ring and the trunk features the forward writes
    ggml_init_params sp{ggml_tensor_overhead() * (size_t) (2 * d.n_layer + 4), nullptr, true};
    d.sctx = ggml_init(sp);
    for (int l = 0; l < d.n_layer; ++l) {
        d.ring_k.push_back(ggml_new_tensor_3d(d.sctx, GGML_TYPE_F16, d.head_dim, d.n_head_kv, d.ring));
        d.ring_v.push_back(ggml_new_tensor_3d(d.sctx, GGML_TYPE_F16, d.head_dim, d.n_head_kv, d.ring));
    }
    feat = ggml_new_tensor_2d(d.sctx, GGML_TYPE_F32, (int64_t) g.n_embd * (int64_t) d.targets.size(),
                              std::max(o.n_ubatch, o.max_window));
    d.sbuf = ggml_backend_alloc_ctx_tensors_from_buft(
        d.sctx, gpu_dev ? ggml_backend_dev_buffer_type(gpu_dev) : ggml_backend_cpu_buffer_type());
    if (!d.sbuf) {
        err = "cannot allocate the DFlash2 drafter's ring";
        return false;
    }
    ggml_backend_buffer_clear(d.sbuf, 0);   // masked slots must hold numbers, not NaN
    d.kv_pos.assign((size_t) d.ring, -1);
    st.dense_bytes += d.w.bytes;
    st.state_bytes += ggml_backend_buffer_get_size(d.sbuf);
    return true;
}

bool GlmMetal::Impl::dflash_inject(int64_t p0, int n, std::string& err) {
    DFlash& d = *df;
    // a slice longer than the ring (--ubatch past the drafter's window): its first rows would share slots with its
    // last in one set_rows, whose GPU write order is undefined.  Only the last `ring` rows can stay anyway
    const int skip = std::max(0, n - d.ring);
    p0 += skip;
    n -= skip;
    const size_t mem = ggml_tensor_overhead() * kDraftNodes + ggml_graph_overhead_custom(kDraftNodes, false);
    std::vector<uint8_t> arena(mem);
    ggml_init_params ip{mem, arena.data(), true};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kDraftNodes, false);
    std::vector<Up> up;
    std::vector<int32_t> pos((size_t) n);
    std::vector<int64_t> slot((size_t) n);
    for (int i = 0; i < n; ++i) {
        pos[(size_t) i] = (int32_t) (p0 + i);
        slot[(size_t) i] = (p0 + i) % d.ring;
    }
    ggml_tensor* pt = input(ctx, up, GGML_TYPE_I32, n, 1, pos.data(), pos.size() * 4);
    pt = ggml_reshape_1d(ctx, pt, n);
    ggml_tensor* st_ = input(ctx, up, GGML_TYPE_I64, n, 1, slot.data(), slot.size() * 8);
    ggml_tensor* f = ggml_view_2d(ctx, feat, feat->ne[0], n, feat->nb[1], (size_t) skip * feat->nb[1]);
    ggml_tensor* gx = rms(ctx, ggml_mul_mat(ctx, d.w.get("fc.weight"), f), d.w.get("enc.output_norm.weight"), d.eps);
    const int64_t kvw = (int64_t) d.head_dim * d.n_head_kv;
    for (int l = 0; l < d.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        ggml_tensor* k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, d.w.get(b + "attn_k.weight"), gx), d.head_dim,
                                         d.n_head_kv, n);
        k = rope(ctx, rms(ctx, k, d.w.get(b + "attn_k_norm.weight"), d.eps), pt, d);
        ggml_tensor* v = ggml_mul_mat(ctx, d.w.get(b + "attn_v.weight"), gx);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, ggml_reshape_2d(ctx, d.ring_k[(size_t) l], kvw, d.ring),
                                                    ggml_reshape_2d(ctx, k, kvw, n), st_));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, ggml_reshape_2d(ctx, d.ring_v[(size_t) l], kvw, d.ring),
                                                    ggml_reshape_2d(ctx, v, kvw, n), st_));
    }
    std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> ups;
    for (Up& u : up) ups.push_back({u.t, std::move(u.d)});
    const bool ok = compute(gf, ups, err);
    ggml_free(ctx);
    if (!ok) return false;
    for (int i = 0; i < n; ++i) d.kv_pos[(size_t) slot[(size_t) i]] = p0 + i;
    return true;
}

void GlmMetal::Impl::dflash_forget(int64_t from) {
    if (!df) return;
    for (int64_t& p : df->kv_pos)
        if (p >= from) p = -1;
}

bool GlmMetal::has_dflash() const { return p_->df != nullptr; }
int GlmMetal::dflash_block() const { return p_->df ? p_->df->block : 0; }

bool GlmMetal::dflash_draft(int32_t token, int n, float p_min, int n_min, float p_cum, std::vector<int32_t>& drafts,
                            std::string& err) {
    Impl& m = *p_;
    drafts.clear();
    if (!m.df) {
        err = "no DFlash2 drafter";
        return false;
    }
    DFlash& d = *m.df;
    n = std::min(n, d.block - 1);
    if (n <= 0) return true;
    const GlmGeometry& g = m.g;
    const int B = n + 1, E = g.n_embd, R = d.ring, K = d.top_k;
    const int64_t np = m.pos;   // the anchor's position (not fed to the trunk yet)
    std::vector<int32_t> toks((size_t) B, d.mask_id);
    toks[0] = token;
    std::vector<float> emb;
    if (!m.embed_raw(toks.data(), B, emb, err)) return false;
    std::vector<int32_t> pos((size_t) B);
    for (int t = 0; t < B; ++t) pos[(size_t) t] = (int32_t) (np + t);
    // every row sees the ring's positions max(0, np - window + 1) .. np - 1, and the whole block
    std::vector<ggml_fp16_t> mask((size_t) (R + B) * B);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    const int64_t lo = std::max<int64_t>(0, np - d.window + 1);
    for (int t = 0; t < B; ++t)
        for (int s = 0; s < R + B; ++s) {
            const bool live = s >= R || (d.kv_pos[(size_t) s] >= lo && d.kv_pos[(size_t) s] < np);
            mask[(size_t) t * (R + B) + s] = live ? zero : ninf;
        }

    const size_t mem = ggml_tensor_overhead() * kDraftNodes + ggml_graph_overhead_custom(kDraftNodes, false);
    std::vector<uint8_t> arena(mem);
    ggml_init_params ip{mem, arena.data(), true};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kDraftNodes, false);
    std::vector<Up> up;
    ggml_tensor* x = input(ctx, up, GGML_TYPE_F32, E, B, emb.data(), emb.size() * 4);
    ggml_tensor* pt = ggml_reshape_1d(ctx, input(ctx, up, GGML_TYPE_I32, B, 1, pos.data(), pos.size() * 4), B);
    ggml_tensor* mk = input(ctx, up, GGML_TYPE_F16, R + B, B, mask.data(), mask.size() * sizeof(ggml_fp16_t));
    const float scale = 1.0f / std::sqrt((float) d.head_dim);
    for (int l = 0; l < d.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        auto W = [&](const char* s) { return d.w.get(b + s); };
        ggml_tensor* hn = rms(ctx, x, W("attn_norm.weight"), d.eps);
        ggml_tensor* dyn = ggml_mul_mat(ctx, W("attn_conv_proj.weight"), hn);
        ggml_tensor* xc = dconv(ctx, d, hn, dyn, W("attn_conv_base"), 0);
        ggml_tensor* q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W("attn_q.weight"), xc), d.head_dim, d.n_head, B);
        ggml_tensor* k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W("attn_k.weight"), xc), d.head_dim, d.n_head_kv, B);
        ggml_tensor* v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W("attn_v.weight"), xc), d.head_dim, d.n_head_kv, B);
        q = rope(ctx, rms(ctx, q, W("attn_q_norm.weight"), d.eps), pt, d);
        k = rope(ctx, rms(ctx, k, W("attn_k_norm.weight"), d.eps), pt, d);
        ggml_tensor* ka = ggml_concat(ctx, d.ring_k[(size_t) l], ggml_cast(ctx, k, GGML_TYPE_F16), 2);
        ggml_tensor* va = ggml_concat(ctx, d.ring_v[(size_t) l], ggml_cast(ctx, v, GGML_TYPE_F16), 2);
        ggml_tensor* a = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), ggml_permute(ctx, ka, 0, 2, 1, 3),
                                             ggml_permute(ctx, va, 0, 2, 1, 3), mk, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
        a = ggml_reshape_2d(ctx, a, (int64_t) d.head_dim * d.n_head, B);
        ggml_tensor* o = dconv(ctx, d, ggml_mul_mat(ctx, W("attn_output.weight"), a), dyn, W("attn_conv_base"), 1);
        x = ggml_add(ctx, o, x);
        ggml_tensor* fn = rms(ctx, x, W("ffn_norm.weight"), d.eps);
        ggml_tensor* dyn2 = ggml_mul_mat(ctx, W("ffn_conv_proj.weight"), fn);
        ggml_tensor* fc = dconv(ctx, d, fn, dyn2, W("ffn_conv_base"), 0);
        ggml_tensor* f = ggml_mul_mat(ctx, W("ffn_down.weight"),
                                      ggml_swiglu_split(ctx, ggml_mul_mat(ctx, W("ffn_gate.weight"), fc),
                                                        ggml_mul_mat(ctx, W("ffn_up.weight"), fc)));
        x = ggml_add(ctx, dconv(ctx, d, f, dyn2, W("ffn_conv_base"), 1), x);
    }
    ggml_tensor* y = rms(ctx, x, d.w.get("output_norm.weight"), d.eps);
    ggml_tensor* yr = ggml_view_2d(ctx, y, E, n, y->nb[1], y->nb[1]);   // the mask rows
    ggml_tensor* lg = ggml_mul_mat(ctx, m.w("output.weight"), yr);      // [V, n]
    ggml_tensor* cand = ggml_cont(ctx, ggml_argsort_top_k(ctx, lg, K));   // [K, n], best first
    ggml_tensor* unary = ggml_get_rows(ctx, ggml_reshape_3d(ctx, lg, 1, g.n_vocab, n), cand);   // [1, K, n]
    ggml_tensor* gate = ggml_mul_mat(ctx, d.w.get("selector_hidden.weight"), yr);              // [rank, n]
    ggml_tensor* succ = ggml_get_rows(ctx, d.w.get("selector_successor.weight"), ggml_reshape_1d(ctx, cand, K * n));
    ggml_tensor* anchor = ggml_reshape_1d(ctx, input(ctx, up, GGML_TYPE_I32, 1, 1, &token, 4), 1);
    ggml_tensor* pids = n > 1 ? ggml_concat(ctx, anchor, ggml_reshape_1d(ctx, ggml_view_2d(ctx, cand, K, n - 1, cand->nb[1], 0),
                                                                         K * (n - 1)), 0)
                              : anchor;
    ggml_tensor* pred = ggml_get_rows(ctx, d.w.get("selector_predecessor.weight"), pids);   // [rank, 1 + K(n-1)]
    for (ggml_tensor* t : {cand, unary, gate, succ, pred}) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }
    std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> ups;
    for (Up& u : up) ups.push_back({u.t, std::move(u.d)});
    if (!m.compute(gf, ups, err)) {
        ggml_free(ctx);
        return false;
    }
    std::vector<int32_t> c((size_t) K * n);
    std::vector<float> un((size_t) K * n), ga((size_t) d.rank * n), su((size_t) d.rank * K * n),
        pr((size_t) d.rank * (1 + K * (n - 1)));
    ggml_backend_tensor_get(cand, c.data(), 0, c.size() * 4);
    ggml_backend_tensor_get(unary, un.data(), 0, un.size() * 4);
    ggml_backend_tensor_get(gate, ga.data(), 0, ga.size() * 4);
    ggml_backend_tensor_get(succ, su.data(), 0, su.size() * 4);
    ggml_backend_tensor_get(pred, pr.data(), 0, pr.size() * 4);
    ggml_free(ctx);

    // the lattice walk: row i scores its candidates by their logit plus the transition from the previous pick
    const int rk = d.rank;
    const float* pv = pr.data();   // the anchor's predecessor row
    double cum = 1.0;
    for (int i = 0; i < n; ++i) {
        std::vector<double> s((size_t) K);
        double smax = -INFINITY;
        int best = 0;
        for (int j = 0; j < K; ++j) {
            const float* sv = su.data() + ((size_t) i * K + (size_t) j) * rk;
            const float* gv = ga.data() + (size_t) i * rk;
            double acc = un[(size_t) i * K + (size_t) j];
            for (int r = 0; r < rk; ++r) acc += (double) pv[r] * gv[r] * sv[r];
            s[(size_t) j] = acc;
            if (acc > smax) {
                smax = acc;
                best = j;
            }
        }
        double z = 0.0;
        for (int j = 0; j < K; ++j) z += std::exp(s[(size_t) j] - smax);
        const double conf = 1.0 / z;
        cum *= conf;
        if (i >= n_min && (conf < p_min || cum < p_cum)) break;
        drafts.push_back(c[(size_t) i * K + (size_t) best]);
        if (i + 1 < n) pv = pr.data() + (size_t) (1 + i * K + best) * rk;   // the pick's predecessor row
    }
    return true;
}

}  // namespace maya::metal
