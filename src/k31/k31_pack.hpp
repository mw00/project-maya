// src/k31/k31_pack.hpp - maya-k31 reads the text pack tools/k31/pack_gguf.py writes: k31.txt (the geometry, the
// expert classes and their source ids) and tensors.txt (every tensor's file, absolute offset, bytes and dims).
//
// Host only, no GGUF reader: the pack text is the contract (docs/K31-ENGINE.md §3 has the record grammar), checked and
// refused before any device call.  The runner maps the model file from tensors.txt; read_pack checks every tensor lies
// inside its file.  A header-only pack (`data header-only`: tensor data absent) is refused unless the caller allows it
// (--pack-info and tests).
//
// Every refusal reads "maya-k31: pack: ..." and names the record, key or layer.
#pragma once

#include "k31_geo.hpp"   // (kMaxPicks: k31_route's 16)

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace k31 {

struct PackLayer {
    int index = -1;
    std::string attn;   // kda | mla
    std::string ffn;    // dense | moe
    int n_exp = 0, k = 0, edim = 0, eff = 0, sff = 0, norm_div = 0;
};

// one expert class of a MoE layer: M1-M2 have exactly one a layer, slot 0 (plan §3.4; more only if D-FMT picks it)
struct PackClass {
    int layer = -1, slot = 0, count = 0;
    std::string format, gate, up, down;
};

struct PackTensor {
    std::string name, dtype;
    int file = 0;
    uint64_t offset = 0, bytes = 0;   // absolute in the file
    std::vector<uint64_t> dims;       // outermost first (ggml's ne reversed)
};

struct Pack {
    std::string arch, format, data, method, model_file;
    bool header_only = false;
    int n_layers = 0, n_embd = 0, vocab = 0, n_head = 0, context = 0;
    int attn_res_block = 0, kda_heads = 0, kda_head_dim = 0, d_conv = 0;
    int mla_heads = 0, kv_lora = 0, qk_nope = 0, qk_rope = 0, v_head = 0, q_lora = 0;
    int idx_heads = 0, idx_dim = 0, idx_topk = 0, idx_rope_dim = -1, idx_topk_freq = -1;
    int source_count = 0;
    double rms_eps = 0, lora_eps = 0, kda_l2_eps = 0, idx_ln_eps = 0, router_renorm = 0, gate_lower_bound = 0;
    std::vector<PackLayer> layers;
    std::vector<PackClass> classes;
    std::map<int, std::vector<int>> keep;
    // tensors.txt (read_tensors_text / read_pack)
    std::vector<std::string> files;   // as written (relative to the pack directory)
    std::vector<PackTensor> tensors;
    std::map<std::string, size_t> tensor_index;
};

struct PackOptions {
    bool allow_header_only = false;   // --pack-info and tests: a pack whose tensor data is not in the file
};

struct PackResult {
    bool ok = false;
    std::string err;
    Pack pack;
};

// ggml block formats by name: values and bytes a block (tools/gguf_reader.py BLOCK_GEOMETRY, ggml-common.h)
struct PackFormat {
    const char* name;
    int block_values, block_bytes;
};
inline const PackFormat* pack_format(const std::string& name) {
    static const PackFormat kFormats[] = {
        {"F32", 1, 4},       {"F16", 1, 2},       {"BF16", 1, 2},      {"F64", 1, 8},       {"Q4_0", 32, 18},
        {"Q4_1", 32, 20},    {"Q5_0", 32, 22},    {"Q5_1", 32, 24},    {"Q8_0", 32, 34},    {"Q8_1", 32, 36},
        {"Q2_K", 256, 84},   {"Q3_K", 256, 110},  {"Q4_K", 256, 144},  {"Q5_K", 256, 176},  {"Q6_K", 256, 210},
        {"Q8_K", 256, 292},  {"IQ2_XXS", 256, 66}, {"IQ2_XS", 256, 74}, {"IQ3_XXS", 256, 98}, {"IQ1_S", 256, 50},
        {"IQ4_NL", 32, 18},  {"IQ3_S", 256, 110}, {"IQ2_S", 256, 82},  {"IQ4_XS", 256, 136}, {"IQ1_M", 256, 56},
        {"Q2_0", 64, 18},    {"MXFP4", 32, 17},   {"NVFP4", 64, 36},
    };
    for (const auto& f : kFormats)
        if (name == f.name) return &f;
    return nullptr;
}

namespace pack_detail {

inline PackResult no(const std::string& why) {
    PackResult r;
    r.err = "maya-k31: pack: " + why;
    return r;
}

// a decimal int, all of the token, inside int's range
inline bool integer(const std::string& s, int& out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX) return false;
    out = (int) v;
    return true;
}

// a decimal uint64, all of the token (no sign)
inline bool u64(const std::string& s, uint64_t& out) {
    if (s.empty() || s[0] < '0' || s[0] > '9') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (*end != '\0' || errno == ERANGE) return false;
    out = (uint64_t) v;
    return true;
}

inline bool real(const std::string& s, double& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0' && std::isfinite(out);
}

inline std::vector<std::string> tokens(const std::string& line) {
    std::vector<std::string> t;
    std::istringstream ls(line);
    std::string w;
    while (ls >> w) t.push_back(w);
    return t;
}

inline std::string expert_name(int il, const char* role) {
    return "blk." + std::to_string(il) + ".ffn_" + role + "_exps.weight";
}

inline std::string dims_str(const std::vector<uint64_t>& d) {
    std::string s = "[";
    for (size_t i = 0; i < d.size(); ++i) s += (i ? ", " : "") + std::to_string(d[i]);
    return s + "]";
}

}  // namespace pack_detail

// Read k31.txt.  Every contract refusal comes back in PackResult::err.
inline PackResult read_pack_text(const std::string& text, const PackOptions& opt = {}) {
    using pack_detail::no;
    Pack p;
    struct IntKey {
        const char* name;
        int Pack::*dst;
    };
    static const IntKey kInts[] = {
        {"n_layers", &Pack::n_layers},     {"n_embd", &Pack::n_embd},           {"vocab", &Pack::vocab},
        {"n_head", &Pack::n_head},         {"context", &Pack::context},         {"attn_res_block", &Pack::attn_res_block},
        {"kda_heads", &Pack::kda_heads},   {"kda_head_dim", &Pack::kda_head_dim}, {"d_conv", &Pack::d_conv},
        {"mla_heads", &Pack::mla_heads},   {"kv_lora", &Pack::kv_lora},         {"qk_nope", &Pack::qk_nope},
        {"qk_rope", &Pack::qk_rope},       {"v_head", &Pack::v_head},           {"q_lora", &Pack::q_lora},
        {"idx_heads", &Pack::idx_heads},   {"idx_dim", &Pack::idx_dim},         {"idx_topk", &Pack::idx_topk},
        {"idx_rope_dim", &Pack::idx_rope_dim}, {"idx_topk_freq", &Pack::idx_topk_freq},
    };
    struct RealKey {
        const char* name;
        double Pack::*dst;
    };
    static const RealKey kReals[] = {
        {"rms_eps", &Pack::rms_eps},       {"lora_eps", &Pack::lora_eps},           {"kda_l2_eps", &Pack::kda_l2_eps},
        {"idx_ln_eps", &Pack::idx_ln_eps}, {"router_renorm", &Pack::router_renorm}, {"gate_lower_bound", &Pack::gate_lower_bound},
    };
    std::set<std::string> seen;   // the one-a-pack records
    bool saw_prune = false;
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::vector<std::string> t = pack_detail::tokens(line);
        if (t.empty()) continue;
        const std::string& key = t[0];
        const auto bad = [&](const std::string& why) { return no(why + " (line " + std::to_string(lineno) + ")"); };
        const auto once = [&]() { return seen.insert(key).second; };
        if (key == "layer") {
            PackLayer layer;
            static const char* const kNames[] = {"attn", "ffn", "n_exp", "k", "edim", "eff", "sff", "norm_div"};
            if (t.size() != 18) return bad("a layer record has " + std::to_string(t.size()) + " fields, not 18");
            for (int j = 0; j < 8; ++j)
                if (t[(size_t) (2 + 2 * j)] != kNames[j]) return bad("a layer record has " + t[(size_t) (2 + 2 * j)] +
                                                                    " where " + kNames[j] + " belongs");
            int* const ints[] = {&layer.n_exp, &layer.k, &layer.edim, &layer.eff, &layer.sff, &layer.norm_div};
            if (!pack_detail::integer(t[1], layer.index)) return bad("a layer record's index is not an integer");
            for (int j = 0; j < 6; ++j)
                if (!pack_detail::integer(t[(size_t) (7 + 2 * j)], *ints[j]))
                    return bad("layer " + t[1] + " " + kNames[2 + j] + " is not an integer");
            layer.attn = t[3];
            layer.ffn = t[5];
            if (layer.attn != "kda" && layer.attn != "mla")
                return bad("layer " + std::to_string(layer.index) + " attention is " + layer.attn);
            if (layer.ffn != "dense" && layer.ffn != "moe")
                return bad("layer " + std::to_string(layer.index) + " ffn is " + layer.ffn);
            p.layers.push_back(layer);
        } else if (key == "class") {
            PackClass c;
            if (t.size() != 8) return bad("a class record has " + std::to_string(t.size()) + " fields, not 8");
            if (!pack_detail::integer(t[1], c.layer) || !pack_detail::integer(t[2], c.slot) ||
                !pack_detail::integer(t[4], c.count))
                return bad("a class record's layer, slot or count is not an integer");
            c.format = t[3];
            c.gate = t[5];
            c.up = t[6];
            c.down = t[7];
            p.classes.push_back(c);
        } else if (key == "keep") {
            int il = 0;
            if (t.size() < 2 || !pack_detail::integer(t[1], il)) return bad("a keep record has no layer");
            if (p.keep.count(il)) return bad("layer " + std::to_string(il) + " has two keep records");
            std::vector<int> ids;
            for (size_t j = 2; j < t.size(); ++j) {
                int id = 0;
                if (!pack_detail::integer(t[j], id))
                    return bad("layer " + std::to_string(il) + " keep entry " + std::to_string(j - 2) + " '" + t[j] +
                               "' is not an integer");
                ids.push_back(id);
            }
            p.keep[il] = std::move(ids);
        } else if (key == "prune") {
            if (saw_prune) return bad("a second prune record");
            saw_prune = true;
            if (t.size() < 5 || t[1] != "source_count" || t[3] != "method")
                return bad("a prune record is not 'prune source_count N method M'");
            if (!pack_detail::integer(t[2], p.source_count)) return bad("prune source_count is not an integer");
            for (size_t j = 4; j < t.size(); ++j) p.method += (j > 4 ? " " : "") + t[j];
        } else if (key == "model_file" || key == "arch" || key == "format" || key == "data") {
            if (!once()) return bad("a second " + key + " record");
            if (t.size() != 2) return bad(key + " takes one value, not " + std::to_string(t.size() - 1));
            (key == "model_file" ? p.model_file : key == "arch" ? p.arch : key == "format" ? p.format : p.data) = t[1];
        } else {
            const IntKey* ik = nullptr;
            const RealKey* rk = nullptr;
            for (const auto& k : kInts)
                if (key == k.name) ik = &k;
            for (const auto& k : kReals)
                if (key == k.name) rk = &k;
            if (!ik && !rk) return bad("unknown record " + key);
            if (!once()) return bad("a second " + key + " record");
            if (t.size() != 2) return bad(key + " takes one value, not " + std::to_string(t.size() - 1));
            if (ik && !pack_detail::integer(t[1], p.*(ik->dst))) return bad(key + " is not an integer");
            if (rk && !pack_detail::real(t[1], p.*(rk->dst))) return bad(key + " is not a number");
        }
    }

    // ---- the one-a-pack records
    if (p.arch.empty()) return no("arch is missing");
    if (p.arch != "kimi-k31") return no("architecture " + p.arch + ", kimi-k31 only");
    if (p.format != "gguf") return no(p.format.empty() ? "format is missing" : "format " + p.format + ", gguf only");
    if (p.data == "header-only") {
        if (!opt.allow_header_only)
            return no("the pack is header-only (its tensor data is not in the file): --pack-info and tests only");
        p.header_only = true;
    } else if (p.data != "whole") {
        return no(p.data.empty() ? "data is missing (whole or header-only)" : "data " + p.data + " (whole or header-only)");
    }
    for (const auto& k : kInts)
        if (!seen.count(k.name)) return no(std::string(k.name) + " is missing");
    for (const auto& k : kReals)
        if (!seen.count(k.name))
            return no(std::string(k.name) == "idx_ln_eps" ? "indexer LayerNorm eps is missing"
                                                          : std::string(k.name) + " is missing");
    for (const auto& k : kInts) {
        const std::string n = k.name;
        if (n == "idx_rope_dim" || n == "idx_topk_freq") continue;
        if (p.*(k.dst) <= 0) return no(n + " " + std::to_string(p.*(k.dst)) + " (it must be above 0)");
    }
    if (p.idx_rope_dim != 0)
        return no("index_rope_dim " + std::to_string(p.idx_rope_dim) + " (K3.1's indexer rotates nothing)");
    if (p.idx_topk_freq != 1)
        return no("index_topk_freq " + std::to_string(p.idx_topk_freq) + " (every MLA layer runs its own indexer)");
    for (const auto& k : kReals) {
        const std::string n = k.name;
        const double v = p.*(k.dst);
        if (n == "gate_lower_bound") {
            if (!(v < 0.0)) return no("gate_lower_bound " + std::to_string(v) + " (it must be below 0)");
        } else if (!(v > 0.0 && v < 1.0)) {
            return no(n == "idx_ln_eps" ? "indexer LayerNorm eps is not in (0, 1)" : n + " is not in (0, 1)");
        }
    }

    // ---- layers: 0..n-1 in order, the dense ones first
    if ((int64_t) p.layers.size() != p.n_layers)
        return no(std::to_string(p.layers.size()) + " layer records for " + std::to_string(p.n_layers) + " layers");
    int moe_exp = -1;
    bool any_moe = false;
    for (int i = 0; i < p.n_layers; ++i) {
        const PackLayer& layer = p.layers[(size_t) i];
        const std::string L = "layer " + std::to_string(layer.index);
        if (layer.index != i) return no("layer records are not 0..n_layers-1 in order");
        if (layer.ffn == "dense") {
            if (any_moe) return no(L + " is dense after a MoE layer (the dense layers lead)");
            if (layer.n_exp != 0 || layer.k != 0 || layer.edim != 0 || layer.sff != 0)
                return no(L + " is dense but has experts (n_exp, k, edim and sff must be 0)");
            if (layer.eff <= 0) return no(L + " dense width " + std::to_string(layer.eff) + " (it must be above 0)");
            continue;
        }
        any_moe = true;
        if (layer.n_exp <= 0) return no(L + " keeps " + std::to_string(layer.n_exp) + " experts");
        if (moe_exp < 0) moe_exp = layer.n_exp;
        else if (layer.n_exp != moe_exp)
            return no(L + " keeps " + std::to_string(layer.n_exp) + " experts, an earlier layer keeps " +
                      std::to_string(moe_exp));
        if (layer.k <= 0 || layer.k > layer.n_exp || layer.k > kMaxPicks)
            return no(L + " picks k " + std::to_string(layer.k) + " of " + std::to_string(layer.n_exp) +
                      " experts (1 to " + std::to_string(kMaxPicks) + ", at most the kept count)");
        if (layer.edim <= 0 || layer.eff <= 0 || layer.sff <= 0 || layer.norm_div <= 0)
            return no(L + " has a width that is not above 0 (edim, eff, sff, norm_div)");
    }

    // ---- expert classes: exactly one a MoE layer, slot 0, a known format, the layer's own slabs
    std::map<int, const PackClass*> class_of;
    for (const auto& c : p.classes) {
        const std::string L = "layer " + std::to_string(c.layer);
        if (c.layer < 0 || c.layer >= p.n_layers || p.layers[(size_t) c.layer].ffn != "moe")
            return no("class for layer " + std::to_string(c.layer) + " is not a MoE layer");
        const PackLayer& layer = p.layers[(size_t) c.layer];
        if (class_of.count(c.layer)) return no(L + " has two expert classes (one a layer until D-FMT picks more)");
        if (c.slot != 0) return no(L + " class slot " + std::to_string(c.slot) + " (one class a layer, slot 0)");
        class_of[c.layer] = &c;
        if (c.count != layer.n_exp)
            return no(L + " class holds " + std::to_string(c.count) + " experts, the layer keeps " +
                      std::to_string(layer.n_exp));
        const PackFormat* f = pack_format(c.format);
        if (!f) return no("expert format " + c.format + " not built in (" + L + ")");
        for (int w : {layer.edim, layer.eff})
            if (w % f->block_values)
                return no(L + " expert width " + std::to_string(w) + " is not a multiple of " +
                          std::to_string(f->block_values) + " (" + c.format + ")");
        if (c.gate != pack_detail::expert_name(c.layer, "gate") || c.up != pack_detail::expert_name(c.layer, "up") ||
            c.down != pack_detail::expert_name(c.layer, "down"))
            return no(L + " class names " + c.gate + " " + c.up + " " + c.down + ", not the layer's own expert slabs");
    }
    for (const auto& layer : p.layers)
        if (layer.ffn == "moe" && !class_of.count(layer.index))
            return no("layer " + std::to_string(layer.index) + " has no expert class");

    // ---- the experts' source ids
    if (any_moe) {
        if (!saw_prune) return no("prune is missing (source_count and method)");
        if (p.source_count < moe_exp || p.source_count > (1 << 20))
            return no("prune source_count " + std::to_string(p.source_count) + " is below the kept " +
                      std::to_string(moe_exp) + " experts or implausible");
    }
    for (const auto& [il, ids] : p.keep)
        if (il < 0 || il >= p.n_layers || p.layers[(size_t) il].ffn != "moe")
            return no("keep list for layer " + std::to_string(il) + ", which is not a MoE layer");
    for (const auto& layer : p.layers) {
        if (layer.ffn != "moe") continue;
        const std::string L = "layer " + std::to_string(layer.index);
        auto it = p.keep.find(layer.index);
        if (it == p.keep.end()) return no(L + " has no keep list");
        const std::vector<int>& ids = it->second;
        if ((int) ids.size() != layer.n_exp)
            return no(L + " keep list has " + std::to_string(ids.size()) + " ids for " + std::to_string(layer.n_exp) +
                      " experts");
        std::vector<char> have((size_t) p.source_count, 0);
        for (size_t j = 0; j < ids.size(); ++j) {
            if (ids[j] < 0 || ids[j] >= p.source_count)
                return no(L + " keep entry " + std::to_string(j) + " = " + std::to_string(ids[j]) +
                          ", outside the source's " + std::to_string(p.source_count) + " experts");
            if (have[(size_t) ids[j]]++) return no(L + " keeps expert " + std::to_string(ids[j]) + " twice");
        }
    }
    if (p.model_file.empty()) return no("model_file is missing");
    PackResult ok;
    ok.ok = true;
    ok.pack = std::move(p);
    return ok;
}

// Read tensors.txt into a pack read_pack_text accepted: "" or the refusal.  Checks each record's form, the bytes against
// dtype x dims, unique names, no two tensors overlapping in a file, file 0 = model_file, and every expert class's
// slabs present at the layer's shapes and format.  That they lie inside their files is check_extent's.
inline std::string read_tensors_text(const std::string& text, Pack& p) {
    const auto no = [](const std::string& why) { return pack_detail::no("tensors.txt: " + why).err; };
    p.files.clear();
    p.tensors.clear();
    p.tensor_index.clear();
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::vector<std::string> t = pack_detail::tokens(line);
        if (t.empty()) continue;
        const std::string at = " (line " + std::to_string(lineno) + ")";
        if (t[0] == "file") {
            int i = -1;
            if (t.size() != 3 || !pack_detail::integer(t[1], i)) return no("a file record is not 'file I PATH'" + at);
            if (i != (int) p.files.size()) return no("file records are not 0, 1, ... in order" + at);
            p.files.push_back(t[2]);
            continue;
        }
        if (t[0] != "t") return no("unknown record " + t[0] + at);
        PackTensor x;
        if (t.size() < 7) return no("a t record has " + std::to_string(t.size()) + " fields (at least 7)" + at);
        x.name = t[1];
        x.dtype = t[2];
        if (!pack_detail::integer(t[3], x.file) || x.file < 0 || x.file >= (int) p.files.size())
            return no(x.name + " is in file " + t[3] + ", which no file record names" + at);
        if (!pack_detail::u64(t[4], x.offset) || !pack_detail::u64(t[5], x.bytes))
            return no(x.name + " offset or bytes is not an unsigned integer" + at);
        const PackFormat* f = pack_format(x.dtype);
        if (!f) return no(x.name + " has type " + x.dtype + ", which the pack does not read" + at);
        uint64_t elems = 1;
        for (size_t j = 6; j < t.size(); ++j) {
            uint64_t d = 0;
            if (!pack_detail::u64(t[j], d) || d == 0) return no(x.name + " dim '" + t[j] + "' is not above 0" + at);
            x.dims.push_back(d);
            if (elems > UINT64_MAX / d) return no(x.name + " has more elements than 2^64" + at);
            elems *= d;
        }
        if (x.dims.back() % (uint64_t) f->block_values)
            return no(x.name + " rows of " + std::to_string(x.dims.back()) + " are not whole " + x.dtype + " blocks" + at);
        const uint64_t blocks = elems / (uint64_t) f->block_values;
        if (blocks > UINT64_MAX / (uint64_t) f->block_bytes) return no(x.name + " has more bytes than 2^64" + at);
        const uint64_t want = blocks * (uint64_t) f->block_bytes;
        if (x.bytes != want)
            return no(x.name + " is " + std::to_string(x.bytes) + " bytes, " + x.dtype + " " +
                      pack_detail::dims_str(x.dims) + " is " + std::to_string(want) + at);
        if (x.offset > UINT64_MAX - x.bytes) return no(x.name + " ends past 2^64" + at);
        if (!p.tensor_index.emplace(x.name, p.tensors.size()).second) return no(x.name + " is listed twice" + at);
        p.tensors.push_back(std::move(x));
    }
    if (p.files.empty()) return no("no file record");
    if (p.files[0] != p.model_file) return no("file 0 is " + p.files[0] + ", k31.txt's model_file is " + p.model_file);
    if (p.tensors.empty()) return no("no tensors");
    // no two tensors share bytes of one file
    std::vector<const PackTensor*> order;
    for (const auto& x : p.tensors) order.push_back(&x);
    std::sort(order.begin(), order.end(), [](const PackTensor* a, const PackTensor* b) {
        return a->file != b->file ? a->file < b->file : a->offset < b->offset;
    });
    for (size_t j = 1; j < order.size(); ++j)
        if (order[j]->file == order[j - 1]->file && order[j]->offset < order[j - 1]->offset + order[j - 1]->bytes)
            return no(order[j - 1]->name + " and " + order[j]->name + " overlap in file " +
                      std::to_string(order[j]->file));
    // each class's slabs: gate / up [n_exp, eff, edim], down [n_exp, edim, eff] (outermost first), in its format
    for (const auto& c : p.classes) {
        const PackLayer& layer = p.layers[(size_t) c.layer];
        const uint64_t E = (uint64_t) layer.n_exp, D = (uint64_t) layer.eff, L = (uint64_t) layer.edim;
        const std::pair<const std::string*, std::vector<uint64_t>> slabs[] = {
            {&c.gate, {E, D, L}}, {&c.up, {E, D, L}}, {&c.down, {E, L, D}}};
        for (const auto& [name, want] : slabs) {
            auto it = p.tensor_index.find(*name);
            if (it == p.tensor_index.end()) return no("layer " + std::to_string(c.layer) + "'s " + *name + " is missing");
            const PackTensor& x = p.tensors[it->second];
            if (x.dims != want)
                return no(*name + " is " + pack_detail::dims_str(x.dims) + ", layer " + std::to_string(c.layer) +
                          " says " + pack_detail::dims_str(want));
            if (x.dtype != c.format) return no(*name + " is " + x.dtype + ", its class says " + c.format);
        }
    }
    return "";
}

// Every tensor inside its file (sizes[i]: bytes of file i): "" or the refusal.  A header-only pack (allowed by the
// caller) skips it: its data is not there.
inline std::string check_extent(const Pack& p, const std::vector<uint64_t>& sizes) {
    if (sizes.size() != p.files.size())
        return pack_detail::no("tensors.txt: " + std::to_string(p.files.size()) + " files, " +
                               std::to_string(sizes.size()) + " sizes")
            .err;
    if (p.header_only) return "";
    for (const auto& x : p.tensors)
        if (x.offset + x.bytes > sizes[(size_t) x.file])
            return pack_detail::no("tensors.txt: " + x.name + " ends at byte " + std::to_string(x.offset + x.bytes) +
                                   ", " + p.files[(size_t) x.file] + " has " + std::to_string(sizes[(size_t) x.file]) +
                                   " (not whole: a download or copy still running?)")
                .err;
    return "";
}

// The pack directory: k31_txt_path's k31.txt, the tensors.txt beside it, and the model file(s) it names (relative to
// the pack directory), each a regular file holding every tensor's bytes.
inline PackResult read_pack(const std::string& k31_txt_path, const PackOptions& opt = {}) {
    namespace fs = std::filesystem;
    std::string text;
    if (!detail::read_text(k31_txt_path, text)) return pack_detail::no("cannot read " + k31_txt_path);
    PackResult r = read_pack_text(text, opt);
    if (!r.ok) return r;
    const fs::path dir = fs::path(k31_txt_path).parent_path();
    const std::string tpath = (dir / "tensors.txt").string();
    if (!detail::read_text(tpath, text)) return pack_detail::no("cannot read " + tpath);
    std::string err = read_tensors_text(text, r.pack);
    if (!err.empty()) return {false, err, {}};
    std::vector<uint64_t> sizes;
    for (const auto& f : r.pack.files) {
        const fs::path full = fs::path(f).is_absolute() ? fs::path(f) : dir / f;
        std::error_code ec;
        if (!fs::is_regular_file(full, ec)) return pack_detail::no("model file " + full.string() + " is not a regular file");
        const uint64_t n = (uint64_t) fs::file_size(full, ec);
        if (ec) return pack_detail::no("model file " + full.string() + " cannot be read (" + ec.message() + ")");
        sizes.push_back(n);
    }
    err = check_extent(r.pack, sizes);
    if (!err.empty()) return {false, err, {}};
    return r;
}

}  // namespace k31
