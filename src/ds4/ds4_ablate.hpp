// src/ds4/ds4_ablate.hpp - direction projection on Maya: a unit direction r ([dim]) that the layers project out of
// every residual write - y <- y - scale * r (r . y) on the attention output and on the MoE output, right before each
// hc_post (ds4_model.cu's ablate()).  The use it is written for is abliteration, r a refusal direction.
//
// The direction comes from one of two GGUF files:
//   a llama-cvector-generator file   direction.<layer> F32 [dim]            (--ablate FILE --ablate-layer N)
//   a rank-1 abliteration adapter    general.type "adapter"; blk.N.{attn_output_b,ffn_down_shexp,ffn_down_exps}.weight
//                                    .lora_a = s r^T W, .lora_b = -r on every block (--ablate FILE; layer / scale from
//                                    its abliterate.direction_layer / abliterate.scale)
// Of an adapter only r and the scale are taken, not its A halves (s r^T W of the weights it was computed against).
// On those weights the two agree: (W - s r r^T W) x = y - s r (r . y) for each writer, and the MoE output is the sum
// of the shared and routed writers.  On other weights they differ by s r r^T (W - W_source) x, so this is the model
// llama.cpp --lora runs only on the adapter's own source weights.  An adapter that does not carry the same -r on all
// three writers of every block (one made for some blocks only, or without the routed experts, or any other LoRA) is
// refused: projecting every write is not what it does.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"

namespace strata::ds4::ablate {

// the integer `key` of g, in [lo, hi]; absent -> true with `out` untouched
inline bool meta_int(const GgufFile& g, const std::string& path, const char* key, long long lo, long long hi,
                     long long& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (v == nullptr) return true;
    long long x = 0;
    switch (v->type) {
        case MetaType::U8: case MetaType::U16: case MetaType::U32: case MetaType::U64:
            if (v->u > (uint64_t) hi) { err = path + ": " + key + " out of range"; return false; }
            x = (long long) v->u;
            break;
        case MetaType::I8: case MetaType::I16: case MetaType::I32: case MetaType::I64:
            x = (long long) (int64_t) v->u;   // (the reader sign-extends into u)
            break;
        default:
            err = path + ": " + key + " is not an integer";
            return false;
    }
    if (x < lo || x > hi) { err = path + ": " + key + " out of range"; return false; }
    out = x;
    return true;
}

// the positive finite number `key` of g; absent -> true with `out` untouched
inline bool meta_pos(const GgufFile& g, const std::string& path, const char* key, double& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (v == nullptr) return true;
    const double x = v->is_num() ? v->num() : NAN;
    if (!std::isfinite(x) || !(x > 0.0) || x > 1e6) {
        err = path + ": " + key + " is not a positive finite number";
        return false;
    }
    out = x;
    return true;
}

// the F32 tensor `name` (its first `want` values; 0: all, up to 2^20), every byte of it inside the file
inline bool read_f32(const GgufFile& g, const std::string& path, const std::string& name, uint64_t want,
                     std::vector<float>& out, std::string& err) {
    const strata::TensorInfo* t = g.find(name);
    if (t == nullptr) { err = path + ": no " + name; return false; }
    if (t->type != 0) { err = path + ": " + name + " is not F32"; return false; }
    const uint64_t cap = want != 0 ? (1ull << 32) : (1ull << 20);   // (a routed lora_b is [1, dim, n_exp]: 2M at 384)
    uint64_t n = 1;
    for (uint64_t d : t->shape) {   // bound each dim and the running product first
        if (d == 0 || d > cap || n > cap / d) { err = path + ": " + name + " has an odd size"; return false; }
        n *= d;
    }
    if (want != 0) {
        if (n < want) { err = path + ": " + name + " has an odd size"; return false; }
        n = want;
    }
    const uint64_t size = g.file_size(), data = g.data_start(), bytes = n * 4;
    if (data > size || t->offset > size - data || bytes > size - data - t->offset) {   // (no sum that can wrap)
        err = path + ": " + name + " lies outside the file";
        return false;
    }
    out.resize((size_t) n);
    std::memcpy(out.data(), g.tensor_data(*t), (size_t) bytes);
    return true;
}

/// The direction of `path` (see the file comment): r unit length; `scale` 0 takes the file's (an adapter's
/// abliterate.scale x adapter.lora.alpha, else 1); `layer` 0 takes an adapter's own, or a one-layer control vector's.
/// An adapter must cover the model's `n_layer` blocks exactly.  `what` describes the pick for the start-up log.
inline bool load_direction(const std::string& path, int layer, int n_layer, float& scale, std::vector<float>& r,
                           std::string& what, std::string& err) {
    try {
        GgufFile g(path);
        const MetaValue* type = g.get("general.type");
        const bool adapter = type != nullptr && type->s == "adapter";
        if (adapter) {
            const std::string b0 = "blk.0.attn_output_b.weight.lora_b";
            if (!read_f32(g, path, b0, 0, r, err)) {
                err += " (not a rank-1 abliteration adapter)";
                return false;
            }
            const strata::TensorInfo* t0 = g.find(b0);
            if (t0->shape.size() != 2 || t0->shape[0] != 1) { err = path + ": " + b0 + " is not rank 1"; return false; }
            // the same -r on every writer of every block, or this is not a whole-model projection
            std::vector<float> b;
            for (int il = 0; il < n_layer; ++il)
                for (const char* w : {"attn_output_b", "ffn_down_shexp", "ffn_down_exps"}) {
                    const std::string base = "blk." + std::to_string(il) + "." + w + ".weight";
                    if (g.find(base + ".lora_a") == nullptr || g.find(base + ".lora_b") == nullptr) {
                        err = path + ": no " + base + " pair - only an adapter on every writer of every block is a "
                              "projection of every write";
                        return false;
                    }
                    if (!read_f32(g, path, base + ".lora_b", r.size(), b, err)) return false;
                    if (std::memcmp(b.data(), r.data(), r.size() * sizeof(float)) != 0) {
                        err = path + ": " + base + ".lora_b is not " + b0 + " - not a one-direction adapter";
                        return false;
                    }
                }
            if (g.find("blk." + std::to_string(n_layer) + ".attn_output_b.weight.lora_b") != nullptr) {
                err = path + ": an adapter for more than this model's " + std::to_string(n_layer) + " blocks";
                return false;
            }
            long long dl = layer;
            if (!meta_int(g, path, "abliterate.direction_layer", 1, 4096, dl, err)) return false;
            layer = (int) dl;
            if (scale == 0.0f) {
                double s = 1.0, alpha = 1.0;   // llama.cpp applies B A * alpha / rank, and the rank is 1
                if (!meta_pos(g, path, "abliterate.scale", s, err) || !meta_pos(g, path, "adapter.lora.alpha", alpha, err))
                    return false;
                scale = (float) (s * alpha);
            }
        } else {
            long long n_dirs = 0;
            if (!meta_int(g, path, "controlvector.layer_count", 0, 4096, n_dirs, err)) return false;
            if (layer <= 0) {
                if (n_dirs == 1) layer = 1;
                else {
                    err = path + ": a control vector with " + std::to_string(n_dirs) +
                          " directions: say which with --ablate-layer N (1-based)";
                    return false;
                }
            }
            const std::string tname = "direction." + std::to_string(layer);
            if (g.find(tname) == nullptr) {
                err = path + ": no " + tname + (n_dirs > 0 ? " (layers 1.." + std::to_string(n_dirs) + ")" : "");
                return false;
            }
            if (!read_f32(g, path, tname, 0, r, err)) return false;
            if (scale == 0.0f) scale = 1.0f;
        }
        if (!std::isfinite(scale) || !(scale > 0.0f)) {   // must not reach the GPU
            err = path + ": ablation scale is not a positive finite number";
            return false;
        }
        double ss = 0.0;
        for (float v : r) ss += (double) v * v;
        if (!(ss > 0.0) || !std::isfinite(ss)) {
            err = path + ": the direction is zero or not finite";
            return false;
        }
        const double inv = 1.0 / std::sqrt(ss);
        if (!std::isfinite(inv)) {   // a tiny-but-finite norm would overflow a float reciprocal; normalize in double
            err = path + ": the direction cannot be normalized (norm too small)";
            return false;
        }
        // an adapter's lora_b is -r; the projection r (r . y) does not care about the sign, r is kept as the direction
        for (float& v : r) v = (float) (adapter ? -(double) v * inv : (double) v * inv);
        char buf[200];
        if (adapter)
            std::snprintf(buf, sizeof buf, "adapter (direction layer %d), scale %.3g", layer, scale);
        else
            std::snprintf(buf, sizeof buf, "direction.%d, scale %.3g", layer, scale);
        what = "ds4: abliteration: " + path + " " + buf + ", projected out of every layer's attention and MoE writes";
        return true;
    } catch (const std::exception& e) {   // GgufFile refuses a malformed header by throwing
        err = path + ": " + e.what();
        return false;
    }
}

// --ablate-sites / --ablate-layers: which residual writes lose the direction.  The defaults - the attention's and the
// MoE's write of every layer, the DSpark drafter's blocks too - are --ablate's behaviour without them.  An edit made on
// some writers only (HuiHui's DeepSeek-V4.1-Flash-1017 abliteration: attn_output_b of blocks 11-28) is the projection
// of those writes alone: (W - s r r^T W) x = y - s r (r . y) for the writer W, y = W x.
struct Pick {
    bool attn = true, moe = true;   // the attention's write, the MoE's
    int first = -1, last = -1;      // the layers, 0-based, both included; -1: every layer
    bool takes(int il, bool moe_write) const {
        return (moe_write ? moe : attn) && (first < 0 || (il >= first && il <= last));
    }
    bool narrowed() const { return !(attn && moe) || first >= 0; }
};

// --ablate-sites all|attn|moe
inline bool parse_sites(const std::string& v, Pick& p, std::string& err) {
    if (v == "all") p.attn = p.moe = true;
    else if (v == "attn") p.attn = true, p.moe = false;
    else if (v == "moe") p.attn = false, p.moe = true;
    else {
        err = "--ablate-sites: all, attn or moe, not '" + v + "'";
        return false;
    }
    return true;
}

// --ablate-layers FIRST-LAST: 0-based, both included, FIRST <= LAST <= 4095, digits only
inline bool parse_layers(const std::string& v, Pick& p, std::string& err) {
    auto num = [](const std::string& t, int& out) {
        if (t.empty() || t.size() > 4 || t.find_first_not_of("0123456789") != std::string::npos) return false;
        out = std::stoi(t);
        return out <= 4095;
    };
    const size_t dash = v.find('-');
    int a = -1, b = -1;
    if (dash == std::string::npos || !num(v.substr(0, dash), a) || !num(v.substr(dash + 1), b) || b < a) {
        err = "--ablate-layers: FIRST-LAST, 0-based, both included (e.g. 11-28), not '" + v + "'";
        return false;
    }
    p.first = a;
    p.last = b;
    return true;
}

// at load: the layers inside the model's (its n_layer blocks; the drafter's are not named by --ablate-layers)
inline bool check_layers(const Pick& p, int n_layer, std::string& err) {
    if (p.first >= 0 && p.last >= n_layer) {
        err = "--ablate-layers " + std::to_string(p.first) + "-" + std::to_string(p.last) + ": the model has layers 0-" +
              std::to_string(n_layer - 1);
        return false;
    }
    return true;
}

// the start-up line's end: the writes the direction is projected out of
inline std::string describe(const Pick& p) {
    const std::string layers = p.first >= 0 ? "layers " + std::to_string(p.first) + "-" + std::to_string(p.last) + "'s"
                                            : std::string("every layer's");
    const std::string writes = p.attn && p.moe ? " attention and MoE writes"
                             : p.attn          ? " attention writes only"
                                               : " MoE writes only";
    return ", projected out of " + layers + writes;
}

}  // namespace strata::ds4::ablate
