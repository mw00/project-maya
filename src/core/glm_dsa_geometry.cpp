// src/core/glm_dsa_geometry.cpp - the glm-dsa port: the GLM architecture dispatch and the glm-dsa geometry reader.
//
// Glm5Geometry::from_gguf (glm_model.cu) calls glm_arch_route first: a glm5-next GGUF goes on through Maya's reader
// untouched, a glm-dsa GGUF is read here.  The glm-dsa reader is fail-closed where Maya's is lenient: every key the
// forward depends on is required, and every value this engine cannot serve is refused with the key named, rather
// than replaced by a Flash default (key_length_mla read as qk_nope, kpool 4, SwiGLU clamp 10 would each corrupt
// GLM-5.3 silently).  Values verified against llama.cpp src/models/glm-dsa.cpp and docs/strata/plan/GLM-5.3-SPEC.md.
//
// Also here: the small pack-level hooks the Maya-owned loader calls (pack geometry before the split bounds, split
// boundaries on full-indexer layers, the forward and prompt gates, the NextN lists, the dense split cost).
#include "strata/core/glm_env.hpp"
#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_dsa_split.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/glmd_mmq_pad.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>

namespace strata::core {

const char* glm_arch_name(GlmArch a) {
    switch (a) {
    case GlmArch::Glm5Next:
        return "glm5-next";
    case GlmArch::GlmDsa:
        return "glm-dsa";
    }
    return "?";
}

namespace {

// llama.cpp glm-dsa.cpp GLM_5_2_DEFAULT_INDEXER_TYPES: layers 0, 1 and every 4th from 2 are full (21 of 78); the
// table has 78 entries, layers past it are shared
bool glm52_default_full(int il) { return il < 78 && (il < 2 || (il - 2) % 4 == 0); }

// a scalar KV, or an array whose items all hold the same number (llama.cpp's get_key_or_arr writes per-layer arrays
// for some keys); false = absent or not one number
bool uniform_num(const strata::MetaValue* v, double& out, std::string& why) {
    if (v == nullptr) {
        why = "missing";
        return false;
    }
    if (v->type == strata::MetaType::ARRAY) {
        if (v->items.empty() || v->items.size() != v->count) {
            why = "an empty or truncated array";
            return false;
        }
        for (const auto& it : v->items)
            if (!it.is_num() || it.num() != v->items[0].num()) {
                why = "a per-layer array with differing values (not served)";
                return false;
            }
        out = v->items[0].num();
        return true;
    }
    if (!v->is_num()) {
        why = "not a number";
        return false;
    }
    out = v->num();
    return true;
}

// shard names of a pack: native_experts.txt's header names shard 1; siblings follow -NNNNN-of-MMMMM (the same rule
// as glm_model.cu's pack_shard_names)
bool pack_shards(const std::string& pack_dir, std::vector<std::string>& names, std::string& err) {
    names.clear();
    std::ifstream ne(pack_dir + "/native_experts.txt");
    if (!ne) {
        err = "pack: cannot open " + pack_dir + "/native_experts.txt";
        return false;
    }
    std::string line, first;
    const std::string key = "absolute offsets in ";
    while (first.empty() && std::getline(ne, line))
        if (line.rfind("#", 0) == 0 && line.find(key) != std::string::npos) {
            const size_t a = line.find(key) + key.size();
            const size_t b = line.find(',', a);
            first = line.substr(a, b == std::string::npos ? line.size() - a : b - a);
        }
    if (first.empty()) {
        err = "pack: native_experts.txt names no source shards";
        return false;
    }
    const size_t of = first.find("-of-");
    if (of == std::string::npos) {
        names.push_back(first);
        return true;
    }
    const size_t sdash = first.rfind('-', of - 1);
    const std::string stem = first.substr(0, sdash);
    const int total = std::atoi(first.c_str() + of + 4);
    for (int i = 1; i <= total; ++i) {
        char buf[512];
        std::snprintf(buf, sizeof buf, "%s-%05d-of-%05d.gguf", stem.c_str(), i, total);
        names.push_back(buf);
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------- the dispatch

int glm_arch_route(const strata::GgufFile& g, Glm5Geometry& out, std::string& err) {
    const strata::MetaValue* a = g.get("general.architecture");
    if (a == nullptr || a->type != strata::MetaType::STRING) {
        std::fprintf(stderr, "glm geometry: WARNING the GGUF has no general.architecture - reading it as glm5-next "
                             "(Maya's key probe)\n");
        out.arch = GlmArch::Glm5Next;
        return 0;
    }
    if (a->s == "glm5-next" || a->s == "glm5next") {
        out.arch = GlmArch::Glm5Next;
        return 0;
    }
    if (a->s == "glm-dsa") return from_gguf_glm_dsa(g, out, err) ? 1 : -1;
    err = "glm geometry: architecture '" + a->s + "' is not served (this engine reads glm5-next and glm-dsa)";
    return -1;
}

// ---------------------------------------------------------------- the glm-dsa reader

bool from_gguf_glm_dsa(const strata::GgufFile& g, Glm5Geometry& out_ref, std::string& err) {
    const std::string p = "glm-dsa.";
    Glm5Geometry out;   // from the defaults, never from a previous read
    // the reader keeps a 64-item sample of each array (metadata()); glm-dsa reads per-layer arrays of 78/79 items
    // (attention.indexer.types and the like), so an array the sample truncated is read again whole (get_full)
    std::map<std::string, strata::MetaValue> whole;
    const auto get = [&](const std::string& key) -> const strata::MetaValue* {
        const strata::MetaValue* v = g.get(key);
        if (v == nullptr || v->type != strata::MetaType::ARRAY || v->items.size() == v->count) return v;
        auto it = whole.find(key);
        if (it == whole.end()) {
            strata::MetaValue f;
            if (!g.get_full(key, f)) return v;
            it = whole.emplace(key, std::move(f)).first;
        }
        return &it->second;
    };
    out.arch = GlmArch::GlmDsa;
    bool ok = true;
    const auto fail = [&](const std::string& what) {
        if (ok) err = "glm-dsa: " + what;
        ok = false;
        return false;
    };
    const auto req = [&](const std::string& key, double& v) {
        std::string why;
        if (!uniform_num(get(p + key), v, why)) return fail("required key " + p + key + " is " + why);
        return true;
    };
    const auto req_i = [&](const std::string& key, int lo, int64_t& v) {
        double d = 0;
        if (!req(key, d)) return false;
        if (d != std::floor(d) || d < lo || d > 1e12) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%g", d);
            return fail(p + key + " = " + buf + " is not a valid count (>= " + std::to_string(lo) + ")");
        }
        v = (int64_t) d;
        return true;
    };
    int64_t block_count = 0, n_embd = 0, n_ff = 0, n_ctx = 0, n_head = 0, q_lora = 0, kv_lora = 0, key_len = 0,
            val_len = 0, key_mla = 0, val_mla = 0, n_rot = 0, n_expert = 0, n_used = 0, n_shared = 0, n_ff_exp = 0,
            dense_lead = 0, gating = 0, idx_heads = 0, idx_key = 0, idx_top_k = 0, hkv = 0;
    double rms_eps = 0, theta = 0, w_scale = 0, w_norm = 0;
    req_i("block_count", 1, block_count);
    req_i("embedding_length", 1, n_embd);
    req_i("feed_forward_length", 1, n_ff);
    req_i("context_length", 1, n_ctx);
    req_i("attention.head_count", 1, n_head);
    req_i("attention.head_count_kv", 1, hkv);   // glm-dsa: 1 (absorbed MLA); 0 would mark a KDA layer
    req_i("attention.q_lora_rank", 1, q_lora);
    req_i("attention.kv_lora_rank", 1, kv_lora);
    req_i("attention.key_length", 1, key_len);
    req_i("attention.value_length", 1, val_len);
    req_i("attention.key_length_mla", 1, key_mla);
    req_i("attention.value_length_mla", 1, val_mla);
    req("attention.layer_norm_rms_epsilon", rms_eps);
    req_i("rope.dimension_count", 1, n_rot);
    req("rope.freq_base", theta);
    req_i("expert_count", 1, n_expert);
    req_i("expert_used_count", 1, n_used);
    req_i("expert_shared_count", 0, n_shared);
    req_i("expert_feed_forward_length", 1, n_ff_exp);
    req_i("leading_dense_block_count", 0, dense_lead);
    req("expert_weights_scale", w_scale);
    req("expert_weights_norm", w_norm);
    req_i("expert_gating_func", 0, gating);
    req_i("attention.indexer.head_count", 1, idx_heads);
    req_i("attention.indexer.key_length", 1, idx_key);
    req_i("attention.indexer.top_k", 1, idx_top_k);
    if (!ok) return false;
    int64_t nextn = 0;
    if (get(p + "nextn_predict_layers") != nullptr && !req_i("nextn_predict_layers", 0, nextn)) return false;

    // the trunk and the NextN block (counted inside block_count: 79 = 78 + 1)
    if (nextn > 1) return fail("nextn_predict_layers " + std::to_string(nextn) + ": one NextN block is served");
    out.nextn = (int) nextn;
    out.n_layers = (int) (block_count - nextn);
    if (out.n_layers < 1) return fail("block_count " + std::to_string(block_count) + " leaves no trunk layer");
    out.n_embd = (int) n_embd;
    out.n_head = (int) n_head;
    out.n_head_kv.assign((size_t) out.n_layers, (int) hkv);   // every layer MLA, none recurrent
    out.q_lora = (int) q_lora;
    out.kv_lora = (int) kv_lora;
    out.n_ctx_train = n_ctx;
    out.norm_eps = (float) rms_eps;

    // MLA with a rope tail: key_length_mla is the FULL head (nope + rope), not qk_nope as Maya's reader takes it
    out.qk_rope = (int) n_rot;
    out.qk_nope = (int) (key_mla - n_rot);
    out.v_head = (int) val_mla;
    if (out.qk_nope <= 0)
        return fail("attention.key_length_mla " + std::to_string(key_mla) + " <= rope.dimension_count " +
                    std::to_string(n_rot) + " (no nope part)");
    if (key_len != kv_lora + n_rot)
        return fail("attention.key_length " + std::to_string(key_len) + " != kv_lora_rank + rope (" +
                    std::to_string(kv_lora + n_rot) + "): the latent row is [c_kv | k_pe]");
    if (val_len != kv_lora)
        return fail("attention.value_length " + std::to_string(val_len) + " != kv_lora_rank " + std::to_string(kv_lora));
    if (n_rot % 2 != 0) return fail("rope.dimension_count " + std::to_string(n_rot) + " is odd");

    // RoPE: NORM (interleaved pairs), theta from the file, no scaling of any kind
    out.rope_theta = theta;
    if (!(theta > 0) || !std::isfinite(theta)) return fail("rope.freq_base " + std::to_string(theta) + " is invalid");
    if (const strata::MetaValue* v = get(p + "rope.scaling.type"))
        if (v->type != strata::MetaType::STRING || (v->s != "none" && !v->s.empty()))
            return fail("rope scaling '" + v->s + "' is refused (GLM-5.3 uses plain RoPE)");
    for (const char* k : {"rope.scaling.factor", "rope.scaling.attn_factor", "rope.scale_linear"}) {
        double f = 0;
        std::string why;
        if (get(p + k) != nullptr && (!uniform_num(get(p + k), f, why) || (f != 0.0 && f != 1.0)))
            return fail(std::string(p) + k + " is set: rope scaling is refused (GLM-5.3 uses plain RoPE)");
    }
    {
        double f = 0;
        std::string why;
        if (get(p + "rope.scaling.yarn_log_multiplier") != nullptr &&
            (!uniform_num(get(p + "rope.scaling.yarn_log_multiplier"), f, why) || f != 0.0))
            return fail("rope.scaling.yarn_log_multiplier is set: YaRN mscale is refused");
    }

    // the token lightning indexer: [rope | nope] heads, rope on the first qk_rope dims, LayerNorm'd keys
    out.idx_kind = GlmIdxKind::Token;
    out.idx_heads = (int) idx_heads;
    out.idx_key = (int) idx_key;
    out.idx_top_k = (int) idx_top_k;
    out.idx_kpool = 1;          // no pooling
    out.idx_select_tail = 0;
    out.idx_rope = (int) n_rot;
    if (out.idx_rope > out.idx_key)
        return fail("indexer key_length " + std::to_string(idx_key) + " < its rope dims " + std::to_string(n_rot));
    // HF / vLLM: nn.LayerNorm(eps 1e-6); llama.cpp never loads f_norm_eps for glm-dsa (eps 0) -
    // STRATA_GLM_IDX_LN_EPS=0 reproduces it for parity runs
    out.idx_ln_eps = 1e-6f;
    if (const char* e = std::getenv("STRATA_GLM_IDX_LN_EPS"); e != nullptr && e[0] != '\0') {
        char* end = nullptr;
        const double v = std::strtod(e, &end);
        if (end == e || *end != '\0' || !(v >= 0) || !std::isfinite(v))
            return fail(std::string("STRATA_GLM_IDX_LN_EPS='") + e + "' is not a number >= 0");
        out.idx_ln_eps = (float) v;
    }

    // the indexer schedule: glm-dsa.attention.indexer.types (78 trunk entries, or 79 with the NextN block), else
    // llama.cpp's back-compat rule (glm-dsa.cpp:59-66): n_ctx_train < 1048576 (GLM-5/5.1) all full, else GLM-5.2's
    const int n_all = out.n_layers + out.nextn;
    out.indexer_full.assign((size_t) n_all, 1);
    if (const strata::MetaValue* v = get(p + "attention.indexer.types")) {
        if (v->type != strata::MetaType::ARRAY || (v->count != (uint64_t) out.n_layers && v->count != (uint64_t) n_all))
            return fail("attention.indexer.types has " + std::to_string(v->type == strata::MetaType::ARRAY ? v->count : 1) +
                        " entries, expected " + std::to_string(out.n_layers) +
                        (out.nextn ? " or " + std::to_string(n_all) : std::string()));
        if (v->items.size() != v->count) return fail("attention.indexer.types was truncated by the reader");
        for (size_t i = 0; i < v->items.size(); ++i) {
            if (!v->items[i].is_num()) return fail("attention.indexer.types holds a non-number");
            out.indexer_full[i] = v->items[i].u != 0 ? 1 : 0;
        }
    } else {
        for (int il = 0; il < out.n_layers; ++il)
            out.indexer_full[(size_t) il] = n_ctx < 1048576 ? 1 : (glm52_default_full(il) ? 1 : 0);
        std::fprintf(stderr, "glm-dsa: no attention.indexer.types - %s (llama.cpp's rule, context_length %lld)\n",
                     n_ctx < 1048576 ? "every layer full" : "the GLM-5.2 schedule", (long long) n_ctx);
    }
    for (int il = out.n_layers; il < n_all; ++il)
        if (!out.indexer_full[(size_t) il]) {
            std::fprintf(stderr, "glm-dsa: WARNING indexer.types marks the NextN block blk.%d shared - it carries "
                                 "its own (full) indexer\n", il);
            out.indexer_full[(size_t) il] = 1;
        }
    if (!out.indexer_full[0]) return fail("layer 0 is a shared-indexer layer: there is no earlier top-k to reuse");

    // MoE: sigmoid routing with the exp_probs_b bias for selection only, normalized weights x scale
    out.n_expert = (int) n_expert;
    out.n_exp_used = (int) n_used;
    out.n_shared = (int) n_shared;
    out.n_ff_exp = (int) n_ff_exp;
    out.n_ff_dense = (int) n_ff;
    out.dense_lead = (int) dense_lead;
    out.w_scale = (float) w_scale;
    out.norm_w = w_norm != 0 ? 1 : 0;
    if (gating != 2)
        return fail("expert_gating_func " + std::to_string(gating) + " is not sigmoid (2): GLM-5.3 routes by sigmoid");
    if (n_used > n_expert) return fail("expert_used_count > expert_count");
    if (dense_lead >= out.n_layers) return fail("leading_dense_block_count leaves no MoE layer");
    if (!(w_scale > 0) || !std::isfinite(w_scale)) return fail("expert_weights_scale is invalid");
    // no SwiGLU clamp in GLM-5.3: Maya's Flash default (10) would cut real activations
    out.swiglu_exp = std::numeric_limits<float>::infinity();
    out.swiglu_shexp = std::numeric_limits<float>::infinity();

    // one plain residual: no hyper-connections (mhc() == false, residual_floats() == n_embd)
    out.hc = 1;
    out.sinkhorn_iters = 0;

    // the vocabulary: vocab_size, else glm_geometry_check takes token_embd's rows (0 until then)
    out.n_vocab = 0;
    if (const strata::MetaValue* v = get(p + "vocab_size")) {
        double nv = 0;
        std::string why;
        if (!uniform_num(v, nv, why) || nv < 1) return fail("vocab_size is " + why);
        out.n_vocab = (int) nv;
    }
    out_ref = out;
    return true;
}

// ---------------------------------------------------------------- pack-level hooks

bool glm_pack_geometry(const std::string& pack_dir, Glm5Geometry& out, std::string& err) {
    // only a glm-dsa pack is read here: every other pack keeps Maya's defaults until load_pack reads it (v1.1.55's
    // order, no new failure point for Flash)
    std::vector<std::string> names;
    std::string why;
    if (!pack_shards(pack_dir, names, why)) return true;
    try {
        const strata::GgufFile g(pack_dir + "/../" + names[0]);
        const strata::MetaValue* a = g.get("general.architecture");
        if (a == nullptr || a->type != strata::MetaType::STRING || a->s != "glm-dsa") return true;
        Glm5Geometry geo;
        if (!Glm5Geometry::from_gguf(g, geo, err)) return false;
        out = geo;
        glmd::mmq_pad_enable(true);   // B15.2: before the split planner sizes the prompt buffers
    } catch (const std::exception& e) {
        (void) e;
        return true;   // load_pack reports an unreadable pack as before
    }
    return true;
}

bool glm_split_bounds_ok(const Glm5Geometry& g, const std::vector<int>& bounds, std::string& err) {
    if (g.arch != GlmArch::GlmDsa) return true;
    for (int b : bounds)
        if (b >= 1 && b < g.n_layers && !g.boundary_ok(b)) {
            std::string full;
            for (int il = 1; il < g.n_layers; ++il)
                if (g.boundary_ok(il)) full += (full.empty() ? "" : ",") + std::to_string(il);
            err = "glm split: layer " + std::to_string(b) + " shares the previous layer's indexer top-k, so a part "
                  "cannot start there; the nearest full-indexer layers are " + glm_nearest_boundaries_text(g, b) +
                  " (glm-dsa boundaries: " + full + ")";
            return false;
        }
    return true;
}

bool glm_forward_available(const Glm5Geometry& g, std::string& err) {
    if (g.arch != GlmArch::GlmDsa) return true;
    // glm-dsa B4.6 integration: both glm-dsa forwards exist - the F32 reference (STRATA_GLM_SLOW, B3.5,
    // src/core/glm_dsa_ref.cu) and the fast decode path (fast_layers_plain + the glm-dsa mixer, src/core/glm_dsa_mixer.cu).
    // A geometry the decode kernels cannot serve is refused by glmdsa::mixer_setup with the reason.
    // STRATA_GLM_CHECK_ONLY=1 stops a glm-dsa load here, after the geometry and tensor-shape checks and before any
    // weight upload (a pack check without a GPU; the loader test uses it).
    if (!strata::glmenv::env_flag("STRATA_GLM_CHECK_ONLY", false)) return true;
    int full = 0;
    for (int il = 0; il < g.n_layers; ++il) full += g.idx_full(il) ? 1 : 0;
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "glm-dsa: the pack's geometry and tensor shapes check out (%d layers + %d NextN, n_embd %d, %d "
                  "full-indexer layers, %d experts top-%d); check-only load (STRATA_GLM_CHECK_ONLY) - stopped before "
                  "any weight upload",
                  g.n_layers, g.nextn, g.n_embd, full, g.n_expert, g.n_exp_used);
    err = buf;
    return false;
}

void glm_mtp_lists(GlmArch a, std::vector<std::string>& raw, std::vector<std::string>& b16,
                   std::vector<std::string>& f32) {
    if (a != GlmArch::GlmDsa) return;
    // natively served as Flash's; BF16: no indexer compressor, and the router leaves this list; F32: the router
    // (kept F32 end to end), no compressor APE, shared_head_norm optional (glm_mtp_fallback)
    raw = {"attn_q_a.weight",       "attn_q_b.weight",     "attn_kv_a_mqa.weight",  "attn_output.weight",
           "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "nextn.eh_proj.weight"};
    b16 = {"indexer.attn_k.weight", "indexer.attn_q_b.weight", "indexer.proj.weight", "attn_k_b.weight",
           "attn_v_b.weight"};
    f32 = {"attn_norm.weight",      "ffn_norm.weight",      "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
           "indexer.k_norm.weight", "indexer.k_norm.bias",  "exp_probs_b.bias",     "ffn_gate_inp.weight",
           "nextn.enorm.weight",    "nextn.hnorm.weight",   "nextn.shared_head_norm.weight"};
}

const char* glm_mtp_fallback(GlmArch a, const std::string& name) {
    if (a == GlmArch::GlmDsa && name == "nextn.shared_head_norm.weight") return "output_norm.weight";
    return nullptr;
}

void glm_dense_layer_cost(const Glm5Geometry& g, const std::string& pack_dir, std::vector<double>& cost) {
    if (g.arch != GlmArch::GlmDsa) return;
    std::vector<std::string> names;
    std::string err;
    if (!pack_shards(pack_dir, names, err)) return;
    std::vector<std::unique_ptr<strata::GgufFile>> gfs;
    try {
        for (const auto& n : names) gfs.emplace_back(new strata::GgufFile(pack_dir + "/../" + n));
    } catch (const std::exception&) {
        return;
    }
    std::vector<double> c(cost.size(), 0.0);
    std::ifstream ix(pack_dir + "/index.txt");
    std::string line;
    int rows = 0;
    while (std::getline(ix, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("blk.", 0) != 0) continue;
        std::istringstream ss(line);
        std::string nm;
        ss >> nm;
        const int il = std::atoi(nm.c_str() + 4);
        if (il < 0 || il >= (int) c.size() || nm.find("_exps.") != std::string::npos) continue;
        for (const auto& gf : gfs)
            if (const strata::TensorInfo* t = gf->find(nm)) {
                c[(size_t) il] += (double) ggml_row_size((ggml_type) t->type, (int64_t) t->shape[0]) *
                                  (double) (t->elements() / t->shape[0]);
                ++rows;
                break;
            }
    }
    if (rows > 0) cost = c;
}

}  // namespace strata::core
