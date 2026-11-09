// src/core/glm_geometry_check.cpp - the glm-dsa port: tensor-shape validation of a glm-dsa GGUF against its geometry.
//
// The metadata says what the model is; the tensor directory says what the file holds.  A converter bug or a wrong
// key (key_length_mla read as qk_nope, a 64-entry truncated indexer.types) shows up here as a tensor whose shape
// disagrees, named with both shapes, before any weight is read.  Shapes in ggml ne order (dim 0 fastest), from
// llama.cpp src/models/glm-dsa.cpp load_arch_tensors.  Called once by Glm5Model::load and once by load_pack.
//
// Indexer tensors: required on full layers; on a shared layer they are tolerated with one warning (early GLM-5.2
// converters wrote them on every layer, llama.cpp #25407) and never read.
#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_model.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

namespace {

std::string dims(const std::vector<uint64_t>& s) {
    std::string r = "{";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "}";
}

// GGUF writers drop no dims, but a 1-D tensor may be stored as {n, 1}: compare without trailing 1s
std::vector<uint64_t> trimmed(std::vector<uint64_t> s) {
    while (s.size() > 1 && s.back() == 1) s.pop_back();
    return s;
}

}  // namespace

bool glm_geometry_check(Glm5Geometry& g, const GlmFindTensor& find, int l0, int l1, std::string& err) {
    if (g.arch != GlmArch::GlmDsa) return true;   // Flash: Maya's loader checks what it reads, unchanged
    const uint64_t E = (uint64_t) g.n_embd;
    if (l1 <= 0) l1 = g.n_layers;

    const strata::TensorInfo* emb = find("token_embd.weight");
    if (emb == nullptr) {
        err = "glm-dsa: tensor token_embd.weight is missing";
        return false;
    }
    if (g.n_vocab <= 0) {
        if (emb->shape.size() < 2) {
            err = "glm-dsa: token_embd.weight has shape " + dims(emb->shape) + ", expected {n_embd, n_vocab}";
            return false;
        }
        g.n_vocab = (int) emb->shape[1];   // no vocab_size key: the embedding's rows
    }
    const uint64_t V = (uint64_t) g.n_vocab;

    // a tensor that must exist with exactly this shape
    const auto need = [&](const std::string& name, std::vector<uint64_t> want) -> bool {
        const strata::TensorInfo* t = find(name);
        if (t == nullptr) {
            err = "glm-dsa: tensor " + name + " is missing (expected " + dims(want) + ")";
            return false;
        }
        if (trimmed(t->shape) != trimmed(want)) {
            err = "glm-dsa: tensor " + name + " has shape " + dims(t->shape) + ", expected " + dims(want);
            return false;
        }
        return true;
    };

    if (!need("token_embd.weight", {E, V}) || !need("output_norm.weight", {E}) || !need("output.weight", {E, V}))
        return false;

    const uint64_t H = (uint64_t) g.n_head, QL = (uint64_t) g.q_lora, KL = (uint64_t) g.kv_lora;
    const uint64_t QH = (uint64_t) g.qk_head(), NOPE = (uint64_t) g.qk_nope, VH = (uint64_t) g.v_head;
    const uint64_t IH = (uint64_t) g.idx_heads, IK = (uint64_t) g.idx_key;
    const uint64_t NX = (uint64_t) g.n_expert, FX = (uint64_t) g.n_ff_exp, FS = (uint64_t) g.n_ff_exp * g.n_shared;
    const uint64_t FD = (uint64_t) g.n_ff_dense;
    static const char* const kIdx[] = {"indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight",
                                       "indexer.attn_k.weight", "indexer.attn_q_b.weight"};

    // the trunk's layers of this part, and the NextN block when this part ends the trunk
    std::vector<int> layers;
    for (int il = l0; il < l1; ++il) layers.push_back(il);
    if (l1 == g.n_layers)
        for (int il = g.n_layers; il < g.n_layers + g.nextn; ++il) layers.push_back(il);

    std::vector<int> extra_idx;   // shared layers that still carry indexer tensors
    for (int il : layers) {
        const std::string P = "blk." + std::to_string(il) + ".";
        const bool mtp = il >= g.n_layers;
        if (mtp && find(P + "nextn.eh_proj.weight") == nullptr) {
            std::fprintf(stderr, "glm-dsa: WARNING the GGUF has no NextN block (blk.%d) - no speculative decode\n", il);
            continue;
        }
        if (!need(P + "attn_norm.weight", {E}) || !need(P + "attn_q_a_norm.weight", {QL}) ||
            !need(P + "attn_kv_a_norm.weight", {KL}) || !need(P + "attn_q_a.weight", {E, QL}) ||
            !need(P + "attn_q_b.weight", {QL, H * QH}) ||
            !need(P + "attn_kv_a_mqa.weight", {E, (uint64_t) g.kv_row()}) ||
            !need(P + "attn_k_b.weight", {NOPE, KL, H}) || !need(P + "attn_v_b.weight", {KL, VH, H}) ||
            !need(P + "attn_output.weight", {H * VH, E}) || !need(P + "ffn_norm.weight", {E}))
            return false;
        if (g.idx_full(il)) {
            if (!need(P + "indexer.k_norm.weight", {IK}) || !need(P + "indexer.k_norm.bias", {IK}) ||
                !need(P + "indexer.proj.weight", {E, IH}) || !need(P + "indexer.attn_k.weight", {E, IK}) ||
                !need(P + "indexer.attn_q_b.weight", {QL, IH * IK}))
                return false;
        } else {
            for (const char* n : kIdx)
                if (find(P + n) != nullptr) {
                    extra_idx.push_back(il);
                    break;
                }
        }
        if (il < g.dense_lead) {
            if (!need(P + "ffn_gate.weight", {E, FD}) || !need(P + "ffn_up.weight", {E, FD}) ||
                !need(P + "ffn_down.weight", {FD, E}))
                return false;
        } else {
            if (!need(P + "ffn_gate_inp.weight", {E, NX}) || !need(P + "exp_probs_b.bias", {NX}) ||
                !need(P + "ffn_gate_exps.weight", {E, FX, NX}) || !need(P + "ffn_up_exps.weight", {E, FX, NX}) ||
                !need(P + "ffn_down_exps.weight", {FX, E, NX}))
                return false;
            if (g.n_shared > 0 &&
                (!need(P + "ffn_gate_shexp.weight", {E, FS}) || !need(P + "ffn_up_shexp.weight", {E, FS}) ||
                 !need(P + "ffn_down_shexp.weight", {FS, E})))
                return false;
        }
        if (mtp) {
            if (!need(P + "nextn.eh_proj.weight", {2 * E, E}) || !need(P + "nextn.enorm.weight", {E}) ||
                !need(P + "nextn.hnorm.weight", {E}))
                return false;
            if (find(P + "nextn.shared_head_norm.weight") != nullptr && !need(P + "nextn.shared_head_norm.weight", {E}))
                return false;
        }
    }
    if (!extra_idx.empty()) {
        std::string s;
        for (size_t i = 0; i < extra_idx.size() && i < 8; ++i) s += (i ? "," : "") + std::to_string(extra_idx[i]);
        std::fprintf(stderr, "glm-dsa: WARNING %zu shared-indexer layers carry indexer tensors (%s%s) - ignored, they "
                             "reuse the previous full layer's top-k\n", extra_idx.size(), s.c_str(),
                     extra_idx.size() > 8 ? ",..." : "");
    }
    return true;
}

}  // namespace strata::core
