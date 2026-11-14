// src/k31/k31_geo.hpp - maya-k31: Kimi K3.1's geometry, read once from the GGUF header, with every refusal.
//
// Host only (no CUDA/HIP headers, no allocation on any device), like the GLM runner's geometry reader: the runner reads every
// file through k31::read before it touches a GPU, and tests/k31_geo_test.cpp runs the same reading with g++ alone.
// It reads the header and the tensor directory through strata::GgufFile, never tensor data.
//
// The field names are those of ref/kimi_k31.py's KimiK31Config, the oracle the engine is held to; the test compares
// them on real files (tools/k31/geo_ref.py writes the oracle's side).  The file contract is docs/KIMI-K31.md §3.2,
// plus these keys, all optional (an absent one takes the default below and leaves a line in Geo::notes, which the
// engine prints):
//   kimi-k31.kda.head_count                  the KDA heads (default: attention.head_count, as upstream's graph reads it);
//                                            blk.N.ssm_a must have exactly this many entries (A_log sliced at conversion)
//   kimi-k31.attention.lora_norm_epsilon     q_a / kv_a RMSNorm eps (default 1e-6, Moonshot's code)
//   kimi-k31.kda.l2_norm_epsilon             KDA q/k l2-norm eps (default 1e-6, fla's)
//   kimi-k31.attention.indexer.rope_dim      must be 0 (default 0: a NoPE indexer)
//   kimi-k31.attention.indexer.topk_freq     must be 1 (default 1: every DSA layer runs its own indexer)
//   kimi-k31.expert_prune.source_count       a pruned file's provenance: the source's expert count (896) ...
//   kimi-k31.expert_prune.method             ... how the experts were chosen ("usage") ...
//   kimi-k31.expert_prune.keep               ... and, per MoE layer in order, the source id of each of the file's
//                                            experts (n_moe_layers x expert_count integers, layer-major)
// A pruned file without those keys has its ids in "<file>.prune.json" (the converter's and the prune tool's sidecar:
// {"arch", "source_expert_count", "expert_count", "layers": {"<layer>": [ids in the file's expert order]}}); with
// neither, the file's experts are its own ids 0..E-1 (Geo::keep_from "none").  When both are present they must
// agree.  A sidecar path that exists but is not a readable regular file is refused, never taken for "no sidecar".
//
// Per-layer arrays are read whole with GgufFile::get_full (metadata() keeps 64 items of an array: read from it, a
// 93-entry head_count_kv would lose layers 64-92).  expert_count may be a per-layer array only if every MoE
// layer has the same count; expert_feed_forward_length may differ per layer (narrower experts, a neuron trim: the
// routed norm runs over the latent and does not change), and each MoE layer's expert slabs must carry its own width.
// The rest of the thin geometry (a per-layer expert_latent_length, kimi-k31.routed_norm.divisor / .folded: a sliced
// latent whose norm divides by the source's width and whose norm weight is folded into the up-projection) is not
// built in, and a file carrying any of it is refused rather than run with the wrong routed norm.
//
// Widths that must be a multiple of an expert format's block (256 values for the IQ and K-quant formats) are the
// format's rule, checked where the format is known (k31_fmt / k31_pack), not here: the fixtures' 64-wide experts
// are a valid geometry.
//
// Every refusal reads "maya-k31: geometry not covered: <what>" and names the key, tensor or layer.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace k31 {

inline constexpr const char* kArch = "kimi-k31";
inline constexpr const char* kRefusal = "maya-k31: geometry not covered: ";

// what the engine's kernels hold (the widths have no kernel cap yet: k31_kernels sets one when it has one, and until
// then embedding_length and q_lora_rank take the same 1M plausibility bound as the FFN widths)
inline constexpr int kMaxPicks = 16;         // k31_route and every copied K3.1 structure (kPicks31)
inline constexpr int kMaxExperts = 1024;     // k31_route: one 1024-thread block a token, one thread an expert
inline constexpr int kLatentLora = 512;     // k31_mla_attention: the latent row is 512 values ...
inline constexpr int kLatentRope = 64;       // ... plus 64 un-rotated ones: key row 576, value row 512
inline constexpr int kMaxIndexHeads = 32;    // the indexer score kernels (glm_dsa_score; DeepSeek's index_score)
inline constexpr int kMaxIndexKey = 128;
inline constexpr int kMaxKdaHeadDim = 256;   // glm_kda_recurrence: one thread a state row, 256 threads
inline constexpr int kMinConv = 2, kMaxConv = 17;   // glm_kda_conv: KD_MAX_D_CONV 16 history rows
inline constexpr double kDefaultLoraEps = 1e-6, kDefaultKdaL2Eps = 1e-6;

// eps presets (ref/kimi_k31.py EPS_PRESETS): moonshot = the file's keys, else Moonshot's 1e-6; llama_cpp = the rms
// eps for both, as llama.cpp's kimi-k3 graph runs them
enum class EpsPreset { moonshot, llama_cpp };

struct Geo {
    int n_vocab = 0, n_embd = 0, n_layers = 0, n_head = 0, n_ctx_train = 0;
    std::vector<int> n_head_kv;       // per layer: 0 = KDA (recurrent), 1 = MLA (served as MQA)
    // MLA, NoPE
    int q_lora_rank = 0, kv_lora_rank = 0, n_rot = 0, n_head_k_mla = 0, n_head_v_mla = 0, qk_nope = 0;
    // KDA
    int ssm_d_conv = 0, kda_head_dim = 0, n_kda_head = 0;
    float kda_gate_lower_bound = 0;
    // FFN / MoE
    int n_ff = 0, n_layer_dense_lead = 0, n_expert = 0, n_expert_used = 0, n_expert_shared = 0, n_expert_latent = 0;
    std::vector<int> n_ff_exp;        // per layer, 0 on the dense layers
    // thin experts (expert_thin_latent_length / expert_thin_feed_forward_length, the llama.cpp k31/thin layout; per
    // layer, 0 = not thin): the experts read the top n_exp_in directions of the latent (routed_down is n_embd x
    // n_exp_in) and keep n_ff_exp neurons; down still writes the full latent, so routed_norm and routed_up are
    // unchanged.  n_exp_in = n_expert_latent on a layer that is not thin.
    std::vector<int> n_exp_in;
    float expert_weights_scale = 0;
    bool expert_weights_norm = true;
    float situ_beta = 0, situ_linear_beta = 0;
    int attn_res_block_size = 0;
    // the lightning indexer (MLA layers)
    int idx_n_head = 0, idx_key_dim = 0, idx_top_k = 0;
    // epsilons
    float eps_rms = 0, eps_indexer_ln = 0, eps_mla_latent = 0, eps_kda_l2 = 0, eps_router_renorm = 1e-20f;
    EpsPreset eps_preset = EpsPreset::moonshot;
    // a pruned file's provenance
    int expert_source_count = 0;              // the source model's experts (n_expert when the file says nothing)
    std::vector<std::vector<int>> keep;       // per layer (empty on the dense ones): source id of each file expert
    std::string keep_from;                    // "file", "sidecar" or "none"
    std::string prune_method;                 // kimi-k31.expert_prune.method, or the sidecar's "tool"
    std::vector<std::string> notes;           // defaults taken for absent keys

    bool is_recr(int il) const { return n_head_kv[(size_t) il] == 0; }
    bool is_moe(int il) const { return il >= n_layer_dense_lead; }
    int d_inner() const { return n_kda_head * kda_head_dim; }
    std::vector<int> mla_layers() const {
        std::vector<int> v;
        for (int il = 0; il < n_layers; ++il)
            if (!is_recr(il)) v.push_back(il);
        return v;
    }
    const char* eps_preset_name() const { return eps_preset == EpsPreset::moonshot ? "moonshot" : "llama.cpp"; }
};

// ---- a strict JSON reader, for the prune sidecar only.  Duplicate keys, trailing data and bad numbers are refused
// rather than resolved, so a damaged sidecar cannot read as a plausible id list.
namespace json {

struct Value {
    enum Kind { null, boolean, number, string, array, object } kind = null;
    bool b = false;
    double num = 0;
    bool is_int = false;
    int64_t i = 0;
    std::string s;
    std::vector<Value> items;
    std::vector<std::pair<std::string, Value>> members;
    const Value* get(const std::string& k) const {
        for (const auto& m : members)
            if (m.first == k) return &m.second;
        return nullptr;
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : t_(text) {}
    bool parse(Value& out, std::string& err) {
        if (!value(out, 0)) return fail_out(err);
        ws();
        if (p_ != t_.size()) {
            e_ = "trailing data";
            return fail_out(err);
        }
        return true;
    }

private:
    bool fail_out(std::string& err) {
        err = e_ + " at byte " + std::to_string(p_);
        return false;
    }
    bool bad(const char* what) {
        e_ = what;
        return false;
    }
    void ws() {
        while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\n' || t_[p_] == '\r')) ++p_;
    }
    bool lit(const char* w) {
        const size_t n = std::char_traits<char>::length(w);
        if (t_.compare(p_, n, w) != 0) return bad("bad literal");
        p_ += n;
        return true;
    }
    bool str(std::string& out) {
        if (p_ >= t_.size() || t_[p_] != '"') return bad("expected a string");
        ++p_;
        while (p_ < t_.size() && t_[p_] != '"') {
            const char c = t_[p_++];
            if ((unsigned char) c < 0x20) return bad("control character in a string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (p_ >= t_.size()) break;
            const char e = t_[p_++];
            switch (e) {
            case '"': case '\\': case '/': out += e; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                if (p_ + 4 > t_.size()) return bad("bad \\u escape");
                unsigned cp = 0;
                for (int k = 0; k < 4; ++k) {
                    const char h = t_[p_++];
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= (unsigned) (h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= (unsigned) (h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= (unsigned) (h - 'A' + 10);
                    else return bad("bad \\u escape");
                }
                // keys and values this reader compares are ASCII; anything else is kept as UTF-8 of the BMP point
                if (cp < 0x80) out += (char) cp;
                else if (cp < 0x800) { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 0x3F)); }
                else { out += (char) (0xE0 | (cp >> 12)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F)); }
                break;
            }
            default: return bad("bad escape");
            }
        }
        if (p_ >= t_.size()) return bad("unterminated string");
        ++p_;
        return true;
    }
    bool number(Value& v) {
        const size_t s = p_;
        if (p_ < t_.size() && t_[p_] == '-') ++p_;
        const size_t digits = p_;
        while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        if (p_ == digits) return bad("bad number");
        if (t_[digits] == '0' && p_ - digits > 1) return bad("leading zero");
        bool frac = false;
        if (p_ < t_.size() && t_[p_] == '.') {
            frac = true;
            ++p_;
            const size_t f = p_;
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
            if (p_ == f) return bad("bad number");
        }
        if (p_ < t_.size() && (t_[p_] == 'e' || t_[p_] == 'E')) {
            frac = true;
            ++p_;
            if (p_ < t_.size() && (t_[p_] == '+' || t_[p_] == '-')) ++p_;
            const size_t f = p_;
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
            if (p_ == f) return bad("bad number");
        }
        const std::string n = t_.substr(s, p_ - s);
        v.kind = Value::number;
        v.num = std::strtod(n.c_str(), nullptr);
        if (!frac && p_ - digits <= 18) {   // an integer that fits int64 exactly
            v.is_int = true;
            v.i = std::strtoll(n.c_str(), nullptr, 10);
        }
        return true;
    }
    bool value(Value& v, int depth) {
        if (depth > 32) return bad("nesting too deep");
        ws();
        if (p_ >= t_.size()) return bad("unexpected end");
        const char c = t_[p_];
        if (c == '{') {
            v.kind = Value::object;
            ++p_;
            ws();
            if (p_ < t_.size() && t_[p_] == '}') { ++p_; return true; }
            std::set<std::string> seen;
            for (;;) {
                ws();
                std::string k;
                if (!str(k)) return false;
                if (!seen.insert(k).second) return bad("duplicate key");
                ws();
                if (p_ >= t_.size() || t_[p_] != ':') return bad("expected ':'");
                ++p_;
                Value m;
                if (!value(m, depth + 1)) return false;
                v.members.emplace_back(std::move(k), std::move(m));
                ws();
                if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
                if (p_ < t_.size() && t_[p_] == '}') { ++p_; return true; }
                return bad("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.kind = Value::array;
            ++p_;
            ws();
            if (p_ < t_.size() && t_[p_] == ']') { ++p_; return true; }
            for (;;) {
                Value m;
                if (!value(m, depth + 1)) return false;
                v.items.push_back(std::move(m));
                ws();
                if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
                if (p_ < t_.size() && t_[p_] == ']') { ++p_; return true; }
                return bad("expected ',' or ']'");
            }
        }
        if (c == '"') { v.kind = Value::string; return str(v.s); }
        if (c == 't') { v.kind = Value::boolean; v.b = true; return lit("true"); }
        if (c == 'f') { v.kind = Value::boolean; v.b = false; return lit("false"); }
        if (c == 'n') { v.kind = Value::null; return lit("null"); }
        return number(v);
    }

    const std::string& t_;
    size_t p_ = 0;
    std::string e_;
};

}  // namespace json

namespace detail {

inline const char* meta_type_name(strata::MetaType t) {
    switch (t) {
    case strata::MetaType::U8: return "u8";
    case strata::MetaType::I8: return "i8";
    case strata::MetaType::U16: return "u16";
    case strata::MetaType::I16: return "i16";
    case strata::MetaType::U32: return "u32";
    case strata::MetaType::I32: return "i32";
    case strata::MetaType::F32: return "f32";
    case strata::MetaType::BOOL: return "bool";
    case strata::MetaType::STRING: return "string";
    case strata::MetaType::ARRAY: return "array";
    case strata::MetaType::U64: return "u64";
    case strata::MetaType::I64: return "i64";
    case strata::MetaType::F64: return "f64";
    }
    return "?";
}
inline bool is_integer(strata::MetaType t) {
    switch (t) {
    case strata::MetaType::U8: case strata::MetaType::I8: case strata::MetaType::U16: case strata::MetaType::I16:
    case strata::MetaType::U32: case strata::MetaType::I32: case strata::MetaType::U64: case strata::MetaType::I64:
        return true;
    default:
        return false;
    }
}
inline bool is_signed(strata::MetaType t) {
    return t == strata::MetaType::I8 || t == strata::MetaType::I16 || t == strata::MetaType::I32 ||
           t == strata::MetaType::I64;
}
// an integer item as int64 (a u64 above INT64_MAX reads as INT64_MAX: out of every range below)
inline int64_t as_int(const strata::MetaValue& v) {
    if (is_signed(v.type)) return (int64_t) v.u;
    return v.u > (uint64_t) INT64_MAX ? INT64_MAX : (int64_t) v.u;
}
inline std::string fmt(double x) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.9g", x);
    return b;
}
inline std::string dims(const std::vector<uint64_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? ", " : "") + std::to_string(s[i]);
    return o + "]";
}

template <class G>
class Reader {
public:
    Reader(const G& g, Geo& out, std::string& err) : g_(g), out_(out), err_(err) {}

    bool no(const std::string& what) {
        err_ = std::string(kRefusal) + what;
        return false;
    }
    static std::string key(const std::string& k) { return std::string(kArch) + "." + k; }
    const strata::MetaValue* get(const std::string& k) const { return g_.get(key(k)); }

    // a scalar integer key; required unless `def` is given (then an absent key takes it and leaves a note)
    bool integer(const std::string& k, int64_t& dst, const int64_t* def = nullptr, const char* why = nullptr) {
        const strata::MetaValue* v = get(k);
        if (!v) {
            if (!def) return no(key(k) + " is missing");
            dst = *def;
            out_.notes.push_back(key(k) + " absent: " + std::to_string(*def) + (why ? std::string(" (") + why + ")" : ""));
            return true;
        }
        if (!is_integer(v->type)) return no(key(k) + " is a " + meta_type_name(v->type) + ", not an integer");
        dst = as_int(*v);
        return true;
    }
    bool number(const std::string& k, double& dst, const double* def = nullptr, const char* why = nullptr) {
        const strata::MetaValue* v = get(k);
        if (!v) {
            if (!def) return no(key(k) + " is missing");
            dst = *def;
            out_.notes.push_back(key(k) + " absent: " + fmt(*def) + (why ? std::string(" (") + why + ")" : ""));
            return true;
        }
        if (v->type == strata::MetaType::F32 || v->type == strata::MetaType::F64) dst = v->f;
        else if (is_integer(v->type)) dst = (double) as_int(*v);
        else return no(key(k) + " is a " + meta_type_name(v->type) + ", not a number");
        if (!std::isfinite(dst)) return no(key(k) + " = " + fmt(dst) + " is not finite");
        return true;
    }
    bool boolean(const std::string& k, bool& dst) {
        const strata::MetaValue* v = get(k);
        if (!v) return no(key(k) + " is missing");
        if (v->type != strata::MetaType::BOOL) return no(key(k) + " is a " + meta_type_name(v->type) + ", not a bool");
        dst = v->u != 0;
        return true;
    }
    // a whole integer array (get_full); the caller checks its length
    bool int_array(const std::string& k, std::vector<int64_t>& dst) {
        strata::MetaValue v;
        if (!g_.get_full(key(k), v)) return no(key(k) + " is missing");
        if (v.type != strata::MetaType::ARRAY) return no(key(k) + " is a " + meta_type_name(v.type) + ", not an array");
        if (!is_integer(v.elem))
            return no(key(k) + " is an array of " + meta_type_name(v.elem) + ", not of integers");
        dst.clear();
        dst.reserve(v.items.size());
        for (const auto& it : v.items) dst.push_back(as_int(it));
        return true;
    }
    // a per-layer integer value: an array of n_layers entries, or (allow_scalar) one value for every layer
    bool per_layer(const std::string& k, int n_layers, std::vector<int64_t>& dst, bool allow_scalar) {
        const strata::MetaValue* v = get(k);
        if (!v) return no(key(k) + " is missing");
        if (v->type != strata::MetaType::ARRAY) {
            if (!allow_scalar) return no(key(k) + " is a " + meta_type_name(v->type) + ", not a per-layer array");
            int64_t x = 0;
            if (!integer(k, x)) return false;
            dst.assign((size_t) n_layers, x);
            return true;
        }
        if (!int_array(k, dst)) return false;
        if ((int64_t) dst.size() != n_layers)
            return no(key(k) + " has " + std::to_string(dst.size()) + " entries for block_count " +
                      std::to_string(n_layers));
        return true;
    }
    bool range(const std::string& k, int64_t x, int64_t lo, int64_t hi, const char* why) {
        if (x >= lo && x <= hi) return true;
        return no(key(k) + " = " + std::to_string(x) + " (" + why + ")");
    }

private:
    const G& g_;
    Geo& out_;
    std::string& err_;
};

inline bool read_text(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream s;
    s << f.rdbuf();
    out = s.str();
    return true;
}

// the ids of one layer: unique, each in [0, source)
template <class G>
inline bool check_ids(Reader<G>& r, const std::string& where, const std::vector<int>& ids, int source) {
    std::vector<char> seen((size_t) source, 0);
    for (size_t j = 0; j < ids.size(); ++j) {
        const int id = ids[j];
        if (id < 0 || id >= source)
            return r.no(where + " entry " + std::to_string(j) + " = " + std::to_string(id) + ", outside the source's " +
                        std::to_string(source) + " experts");
        if (seen[(size_t) id]++)
            return r.no(where + " keeps expert " + std::to_string(id) + " twice");
    }
    return true;
}

// the prune sidecar: <file>.prune.json.  false + err: refused; true with found = false: no sidecar (nothing at the
// path, not even a link).  Anything else at the path must be a regular file that reads: a pruned file read past a
// sidecar it cannot open would run with the wrong expert ids and no message.
template <class G>
inline bool read_sidecar(Reader<G>& r, const Geo& g, const std::string& path, bool& found, int& source,
                         std::vector<std::vector<int>>& keep, std::string& tool) {
    namespace fs = std::filesystem;
    found = false;
    std::error_code ec;
    if (fs::symlink_status(path, ec).type() == fs::file_type::not_found) return true;
    found = true;
    const std::string name = "the prune sidecar " + path;
    if (fs::status(path, ec).type() != fs::file_type::regular)
        return r.no(name + " is not a regular file (a directory, a link to nothing, or a path that cannot be read)");
    std::string text;
    if (!read_text(path, text)) return r.no(name + " cannot be read");
    json::Value root;
    std::string perr;
    if (!json::Parser(text).parse(root, perr)) return r.no(name + " is not valid JSON (" + perr + ")");
    if (root.kind != json::Value::object) return r.no(name + " is not a JSON object");
    const auto int_member = [&](const char* k, int64_t& dst) {
        const json::Value* v = root.get(k);
        if (!v) return r.no(name + " has no \"" + k + "\"");
        if (v->kind != json::Value::number || !v->is_int) return r.no(name + ": \"" + k + "\" is not an integer");
        dst = v->i;
        return true;
    };
    const json::Value* arch = root.get("arch");
    if (!arch || arch->kind != json::Value::string) return r.no(name + " has no \"arch\" string");
    if (arch->s != kArch) return r.no(name + " is for '" + arch->s + "', not '" + kArch + "'");
    int64_t src = 0, cnt = 0;
    if (!int_member("source_expert_count", src) || !int_member("expert_count", cnt)) return false;
    if (cnt != g.n_expert)
        return r.no(name + " keeps " + std::to_string(cnt) + " experts a layer, the file has " +
                    std::to_string(g.n_expert));
    if (src < g.n_expert || src > (int64_t) 1 << 20)
        return r.no(name + ": \"source_expert_count\" " + std::to_string(src) + " is below the file's " +
                    std::to_string(g.n_expert) + " experts or implausible");
    const json::Value* layers = root.get("layers");
    if (!layers || layers->kind != json::Value::object) return r.no(name + " has no \"layers\" object");
    keep.assign((size_t) g.n_layers, {});
    std::vector<char> have((size_t) g.n_layers, 0);
    for (const auto& m : layers->members) {
        int il = -1;
        bool digits = !m.first.empty() && m.first.size() <= 6;
        for (char c : m.first) digits = digits && c >= '0' && c <= '9';
        if (digits) il = std::stoi(m.first);
        if (il < 0 || il >= g.n_layers || !g.is_moe(il) || std::to_string(il) != m.first)
            return r.no(name + " lists layer \"" + m.first + "\", which is not a MoE layer of this file");
        if (m.second.kind != json::Value::array)
            return r.no(name + ": layer " + m.first + " is not an array");
        if ((int64_t) m.second.items.size() != g.n_expert)
            return r.no(name + ": layer " + m.first + " lists " + std::to_string(m.second.items.size()) +
                        " experts, the file has " + std::to_string(g.n_expert));
        std::vector<int>& ids = keep[(size_t) il];
        for (const auto& x : m.second.items) {
            if (x.kind != json::Value::number || !x.is_int || x.i < INT32_MIN || x.i > INT32_MAX)
                return r.no(name + ": layer " + m.first + " has an id that is not an integer");
            ids.push_back((int) x.i);
        }
        if (!check_ids(r, name + ": layer " + m.first, ids, (int) src)) return false;
        have[(size_t) il] = 1;
    }
    for (int il = g.n_layer_dense_lead; il < g.n_layers; ++il)
        if (!have[(size_t) il]) return r.no(name + " has no ids for layer " + std::to_string(il));
    source = (int) src;
    if (const json::Value* t = root.get("tool"))
        if (t->kind == json::Value::string) tool = t->s;
    return true;
}

}  // namespace detail

/// Reads `g` (opened from `path`; the prune sidecar is "<path>.prune.json", and an empty path skips it) into `out`.
/// false: the engine cannot run the file, and `err` says why ("maya-k31: geometry not covered: ...").
template <class G>
inline bool read(const G& g, const std::string& path, Geo& out, std::string& err,
                 EpsPreset eps = EpsPreset::moonshot) {
    out = Geo{};
    detail::Reader<G> r(g, out, err);
    using detail::fmt;

    // the architecture first: nothing below may read another model's keys as K3.1's
    const strata::MetaValue* arch = g.get("general.architecture");
    if (!arch) return r.no("general.architecture is missing");
    if (arch->type != strata::MetaType::STRING || arch->s != kArch)
        return r.no("general.architecture is '" + (arch->type == strata::MetaType::STRING ? arch->s : std::string("?")) +
                    "'; maya-k31 reads '" + kArch + "' only");

    int64_t v = 0;
    double f = 0;
    // ---- the trunk
    if (!r.integer("block_count", v) || !r.range("block_count", v, 1, 1 << 16, "1 to 65536 layers")) return false;
    out.n_layers = (int) v;
    if (!r.integer("vocab_size", v) || !r.range("vocab_size", v, 1, 1 << 24, "a vocabulary of 1 to 16M tokens"))
        return false;
    out.n_vocab = (int) v;
    if (!r.integer("context_length", v) || !r.range("context_length", v, 1, INT32_MAX, "a positive context"))
        return false;
    out.n_ctx_train = (int) v;
    if (!r.integer("embedding_length", v) || !r.range("embedding_length", v, 1, 1 << 20, "a width of 1 to 1M"))
        return false;
    out.n_embd = (int) v;
    if (!r.integer("attention.head_count", v) ||
        !r.range("attention.head_count", v, 1, 1024, "1 to 1024 attention heads"))
        return false;
    out.n_head = (int) v;

    // ---- the layer types: 0 = KDA, 1 = MLA
    std::vector<int64_t> arr;
    if (!r.per_layer("attention.head_count_kv", out.n_layers, arr, false)) return false;
    out.n_head_kv.resize((size_t) out.n_layers);
    for (int il = 0; il < out.n_layers; ++il) {
        if (arr[(size_t) il] != 0 && arr[(size_t) il] != 1)
            return r.no(r.key("attention.head_count_kv") + "[" + std::to_string(il) + "] = " +
                        std::to_string(arr[(size_t) il]) + " (0 marks a KDA layer, 1 an MLA layer; nothing else)");
        out.n_head_kv[(size_t) il] = (int) arr[(size_t) il];
    }

    // ---- MLA (NoPE): the latent row is kv_lora 512 + 64 un-rotated values (key 576), its values the first 512
    if (!r.integer("attention.q_lora_rank", v) ||
        !r.range("attention.q_lora_rank", v, 1, 1 << 20, "a q_a rank of 1 to 1M"))
        return false;
    out.q_lora_rank = (int) v;
    int64_t lora = 0, rope = 0, kl = 0, vl = 0;
    if (!r.integer("attention.kv_lora_rank", lora) || !r.integer("rope.dimension_count", rope) ||
        !r.integer("attention.key_length", kl) || !r.integer("attention.value_length", vl))
        return false;
    if (lora != kLatentLora || rope != kLatentRope || kl != kLatentLora + kLatentRope || vl != kLatentLora)
        return r.no("the MLA latent row is kv_lora_rank " + std::to_string(lora) + " + rope " + std::to_string(rope) +
                    " (key_length " + std::to_string(kl) + ", value_length " + std::to_string(vl) +
                    "); k31_mla_attention reads a 576-wide key row (512 + 64) and a 512-wide value row only");
    out.kv_lora_rank = (int) lora;
    out.n_rot = (int) rope;
    if (!r.integer("attention.key_length_mla", v)) return false;
    if (v <= rope || v > 1024)
        return r.no(r.key("attention.key_length_mla") + " = " + std::to_string(v) + " leaves no nope part after rope " +
                    std::to_string(rope) + " (or exceeds 1024)");
    out.n_head_k_mla = (int) v;
    out.qk_nope = out.n_head_k_mla - out.n_rot;
    if (!r.integer("attention.value_length_mla", v) ||
        !r.range("attention.value_length_mla", v, 1, 1024, "a value head of 1 to 1024"))
        return false;
    out.n_head_v_mla = (int) v;

    // ---- KDA
    if (!r.integer("ssm.conv_kernel", v) ||
        !r.range("ssm.conv_kernel", v, kMinConv, kMaxConv, "glm_kda_conv takes 2 to 17 taps"))
        return false;
    out.ssm_d_conv = (int) v;
    if (!r.integer("kda.head_dim", v) ||
        !r.range("kda.head_dim", v, 1, kMaxKdaHeadDim, "glm_kda_recurrence holds a head of at most 256"))
        return false;
    out.kda_head_dim = (int) v;
    const int64_t def_kda_heads = out.n_head;
    if (!r.integer("kda.head_count", v, &def_kda_heads, "the KDA layers take attention.head_count") ||
        !r.range("kda.head_count", v, 1, 1024, "1 to 1024 KDA heads"))
        return false;
    out.n_kda_head = (int) v;
    if (!r.number("kda.gate_lower_bound", f)) return false;
    if (!(f < 0.0)) return r.no(r.key("kda.gate_lower_bound") + " = " + fmt(f) + " (it must be below 0)");
    out.kda_gate_lower_bound = (float) f;

    // ---- FFN / MoE
    if (!r.integer("feed_forward_length", v) ||
        !r.range("feed_forward_length", v, 1, 1 << 20, "a dense FFN of 1 to 1M"))
        return false;
    out.n_ff = (int) v;
    if (!r.integer("leading_dense_block_count", v) ||
        !r.range("leading_dense_block_count", v, 0, out.n_layers - 1, "at least one MoE layer after the dense ones"))
        return false;
    out.n_layer_dense_lead = (int) v;
    // expert_count: one value, or a per-layer array whose MoE entries are all equal
    if (!r.per_layer("expert_count", out.n_layers, arr, true)) return false;
    {
        const int first = out.n_layer_dense_lead;
        for (int il = first; il < out.n_layers; ++il)
            if (arr[(size_t) il] != arr[(size_t) first])
                return r.no(r.key("expert_count") + " differs between layers (layer " + std::to_string(first) + ": " +
                            std::to_string(arr[(size_t) first]) + ", layer " + std::to_string(il) + ": " +
                            std::to_string(arr[(size_t) il]) + "); every MoE layer must keep the same count");
        v = arr[(size_t) first];
    }
    if (!r.range("expert_count", v, 1, kMaxExperts, "k31_route holds 1 to 1024 experts")) return false;
    out.n_expert = (int) v;
    if (!r.integer("expert_used_count", v) ||
        !r.range("expert_used_count", v, 1, kMaxPicks, "k31_route picks 1 to 16 experts a token"))
        return false;
    if (v > out.n_expert)
        return r.no(r.key("expert_used_count") + " = " + std::to_string(v) + " picks more experts than the " +
                    std::to_string(out.n_expert) + " a layer has");
    out.n_expert_used = (int) v;
    if (!r.integer("expert_shared_count", v) ||
        !r.range("expert_shared_count", v, 0, 16, "0 to 16 shared experts"))
        return false;
    out.n_expert_shared = (int) v;
    // thin experts' sliced latent (D-FMT arm B): its routed norm divides by the source's width and its norm weight is
    // folded into the up-projection, which Geo does not carry; run as a plain latent it would be normed wrongly
    {
        const std::string thin = "thin expert geometry (D-FMT B: a sliced latent, routed_norm.*) is not built in";
        const std::string pre = r.key("routed_norm.");
        for (const auto& kv : g.metadata())
            if (kv.first.compare(0, pre.size(), pre) == 0) return r.no(kv.first + ": " + thin);
        const strata::MetaValue* lat = r.get("expert_latent_length");
        if (lat && lat->type == strata::MetaType::ARRAY)
            return r.no(r.key("expert_latent_length") + " is per-layer: " + thin);
    }
    if (!r.integer("expert_latent_length", v) ||
        !r.range("expert_latent_length", v, 1, out.n_embd, "the latent is 1 to embedding_length wide"))
        return false;
    out.n_expert_latent = (int) v;
    if (!r.per_layer("expert_feed_forward_length", out.n_layers, arr, true)) return false;
    out.n_ff_exp.assign((size_t) out.n_layers, 0);
    for (int il = out.n_layer_dense_lead; il < out.n_layers; ++il) {
        if (arr[(size_t) il] < 1 || arr[(size_t) il] > (1 << 20))
            return r.no(r.key("expert_feed_forward_length") + " = " + std::to_string(arr[(size_t) il]) + " at layer " +
                        std::to_string(il) + " (an expert of 1 to 1M)");
        out.n_ff_exp[(size_t) il] = (int) arr[(size_t) il];
    }
    // thin experts: both keys or neither, scalar (every MoE layer) or per layer
    out.n_exp_in.assign((size_t) out.n_layers, 0);
    {
        const bool has_d = r.get("expert_thin_latent_length") != nullptr, has_f = r.get("expert_thin_feed_forward_length") != nullptr;
        if (has_d != has_f)
            return r.no(r.key(has_d ? "expert_thin_latent_length" : "expert_thin_feed_forward_length") +
                        " without its pair (thin experts carry both)");
        std::vector<int64_t> td, tf;
        if (has_d && (!r.per_layer("expert_thin_latent_length", out.n_layers, td, true) ||
                      !r.per_layer("expert_thin_feed_forward_length", out.n_layers, tf, true)))
            return false;
        for (int il = out.n_layer_dense_lead; il < out.n_layers; ++il) {
            const int64_t d = has_d ? td[(size_t) il] : 0, f = has_d ? tf[(size_t) il] : 0;
            if ((d == 0) != (f == 0))
                return r.no("thin experts at layer " + std::to_string(il) + ": latent " + std::to_string(d) + ", neurons " +
                            std::to_string(f) + " (both or neither)");
            if (d < 0 || d > out.n_expert_latent || f < 0 || f > out.n_ff_exp[(size_t) il])
                return r.no("thin experts at layer " + std::to_string(il) + ": latent " + std::to_string(d) + " of " +
                            std::to_string(out.n_expert_latent) + ", neurons " + std::to_string(f) + " of " +
                            std::to_string(out.n_ff_exp[(size_t) il]) + " (within the stock widths)");
            out.n_exp_in[(size_t) il] = d > 0 ? (int) d : out.n_expert_latent;
            if (f > 0) out.n_ff_exp[(size_t) il] = (int) f;
        }
    }
    if (!r.integer("expert_gating_func", v)) return false;
    if (v != 2)
        return r.no(r.key("expert_gating_func") + " = " + std::to_string(v) + " (only sigmoid routing, 2, is K3.1's)");
    for (const char* k : {"expert_group_count", "expert_group_used_count"}) {
        if (!r.get(k)) continue;   // absent: one group
        if (!r.integer(k, v)) return false;
        if (v > 1) return r.no(r.key(k) + " = " + std::to_string(v) + " (grouped expert routing is not K3.1's)");
    }
    if (!r.number("expert_weights_scale", f)) return false;
    if (!(f > 0.0)) return r.no(r.key("expert_weights_scale") + " = " + fmt(f) + " (it must be above 0)");
    out.expert_weights_scale = (float) f;
    if (!r.boolean("expert_weights_norm", out.expert_weights_norm)) return false;
    if (!r.number("activation.situ_beta", f)) return false;
    if (!(f > 0.0)) return r.no(r.key("activation.situ_beta") + " = " + fmt(f) + " (it must be above 0)");
    out.situ_beta = (float) f;
    if (!r.number("activation.situ_linear_beta", f)) return false;
    out.situ_linear_beta = (float) f;   // <= 0 leaves the up branch linear (llama.cpp's convention)

    // ---- Block Attention Residuals
    if (!r.integer("attn_res.block_size", v) ||
        !r.range("attn_res.block_size", v, 1, out.n_layers, "K3.1 always mixes blocks of 1 to block_count layers"))
        return false;
    out.attn_res_block_size = (int) v;

    // ---- the lightning indexer: per token, per MLA layer, no pools, no shared selections, no RoPE
    {
        const std::string pre = r.key("attention.indexer.");
        for (const auto& kv : g.metadata()) {
            if (kv.first.compare(0, pre.size(), pre) != 0) continue;
            const std::string k = kv.first.substr(pre.size());
            if (k != "head_count" && k != "key_length" && k != "top_k" && k != "rope_dim" && k != "topk_freq" &&
                k != "kpool" && k != "types")
                return r.no(kv.first + ": an indexer setting maya-k31 does not know (a per-token indexer only)");
        }
    }
    if (!r.integer("attention.indexer.head_count", v) ||
        !r.range("attention.indexer.head_count", v, 1, kMaxIndexHeads, "the indexer kernels take 1 to 32 heads"))
        return false;
    out.idx_n_head = (int) v;
    if (!r.integer("attention.indexer.key_length", v) ||
        !r.range("attention.indexer.key_length", v, 1, kMaxIndexKey, "the indexer kernels take keys of 1 to 128"))
        return false;
    out.idx_key_dim = (int) v;
    if (!r.integer("attention.indexer.top_k", v) ||
        !r.range("attention.indexer.top_k", v, 1, INT32_MAX, "the indexer keeps at least one key"))
        return false;
    out.idx_top_k = (int) v;
    {
        const int64_t zero = 0, one = 1;
        if (!r.integer("attention.indexer.rope_dim", v, &zero, "a NoPE indexer")) return false;
        if (v != 0)
            return r.no(r.key("attention.indexer.rope_dim") + " = " + std::to_string(v) +
                        " (K3.1's indexer rotates nothing: index_rope_dim must be 0)");
        if (!r.integer("attention.indexer.topk_freq", v, &one, "every MLA layer runs its own indexer")) return false;
        if (v != 1)
            return r.no(r.key("attention.indexer.topk_freq") + " = " + std::to_string(v) +
                        " (every MLA layer runs its own indexer: index_topk_freq must be 1)");
    }
    if (r.get("attention.indexer.kpool")) {
        if (!r.integer("attention.indexer.kpool", v)) return false;
        if (v != 0)
            return r.no(r.key("attention.indexer.kpool") + " = " + std::to_string(v) +
                        " (a pooled indexer; K3.1 scores every key)");
    }
    if (r.get("attention.indexer.types")) {
        if (!r.per_layer("attention.indexer.types", out.n_layers, arr, true)) return false;
        for (int il = 0; il < out.n_layers; ++il)
            if (!out.is_recr(il) && arr[(size_t) il] != 1)
                return r.no(r.key("attention.indexer.types") + "[" + std::to_string(il) + "] = " +
                            std::to_string(arr[(size_t) il]) +
                            " (MLA layer " + std::to_string(il) + " would share another layer's selection)");
    }

    // ---- epsilons
    if (!r.number("attention.layer_norm_rms_epsilon", f)) return false;
    if (!(f > 0.0 && f < 1.0))
        return r.no(r.key("attention.layer_norm_rms_epsilon") + " = " + fmt(f) + " (an eps in (0, 1))");
    out.eps_rms = (float) f;
    // the indexer key LayerNorm: never eps 0.  Upstream llama.cpp's converter leaves the key out and its graph takes
    // 1e-5, as ref/kimi_k31.py does (the source config's value)
    const double ln_def = 1e-5;
    if (!r.number("attention.layer_norm_epsilon", f, &ln_def, "the indexer LayerNorm's 1e-5, as ref and upstream llama.cpp"))
        return false;
    if (!(f > 0.0 && f < 1.0))
        return r.no(r.key("attention.layer_norm_epsilon") + " = " + fmt(f) + " (an eps in (0, 1))");
    out.eps_indexer_ln = (float) f;
    out.eps_preset = eps;
    if (eps == EpsPreset::llama_cpp) {
        out.eps_mla_latent = out.eps_kda_l2 = out.eps_rms;
        out.notes.push_back("eps preset llama.cpp: the MLA latent norms and the KDA l2 norm take the rms eps " +
                            fmt(out.eps_rms));
    } else {
        const auto eps_key = [&](const char* k, double def, float& dst) {
            if (!r.number(k, f, &def, "moonshot preset")) return false;
            if (!(f > 0.0 && f < 1.0)) return r.no(r.key(k) + " = " + fmt(f) + " (an eps in (0, 1))");
            dst = (float) f;
            return true;
        };
        if (!eps_key("attention.lora_norm_epsilon", kDefaultLoraEps, out.eps_mla_latent) ||
            !eps_key("kda.l2_norm_epsilon", kDefaultKdaL2Eps, out.eps_kda_l2))
            return false;
    }

    // ---- the tensor directory: A_log's length, and each MoE layer's expert count and widths
    std::map<std::string, const strata::TensorInfo*> by_name;
    for (const auto& t : g.tensors()) by_name.emplace(t.name, &t);
    const auto tensor = [&](int il, const char* what, const strata::TensorInfo*& t) {
        const std::string n = "blk." + std::to_string(il) + "." + what;
        auto it = by_name.find(n);
        if (it == by_name.end()) return r.no(n + " is missing");
        t = it->second;
        return true;
    };
    const auto shaped = [&](int il, const char* what, const std::vector<uint64_t>& want) {
        const strata::TensorInfo* t = nullptr;
        if (!tensor(il, what, t)) return false;
        if (t->shape != want)
            return r.no("blk." + std::to_string(il) + "." + what + " is " + detail::dims(t->shape) +
                        ", the geometry says " + detail::dims(want));
        return true;
    };
    for (int il = 0; il < out.n_layers; ++il) {
        if (out.is_recr(il)) {
            const strata::TensorInfo* t = nullptr;
            if (!tensor(il, "ssm_a", t)) return false;
            if (t->shape.size() != 1 || t->shape[0] != (uint64_t) out.n_kda_head)
                return r.no("blk." + std::to_string(il) + ".ssm_a is " + detail::dims(t->shape) + " for " +
                            std::to_string(out.n_kda_head) + " KDA heads (A_log is sliced to the head count)");
        }
        if (!out.is_moe(il)) continue;
        const uint64_t E = (uint64_t) out.n_expert, D = (uint64_t) out.n_ff_exp[(size_t) il],
                       L = (uint64_t) out.n_expert_latent, Lin = (uint64_t) out.n_exp_in[(size_t) il];
        const strata::TensorInfo* t = nullptr;
        if (!tensor(il, "ffn_gate_inp.weight", t)) return false;
        if (t->shape.size() != 2 || t->shape[1] != E)
            return r.no("blk." + std::to_string(il) + ".ffn_gate_inp.weight is " + detail::dims(t->shape) +
                        ": the layer does not route to the file's " + std::to_string(E) +
                        " experts (every MoE layer must keep the same count)");
        if (!shaped(il, "ffn_gate_inp.weight", {(uint64_t) out.n_embd, E}) || !shaped(il, "exp_probs_b.bias", {E}) ||
            !shaped(il, "ffn_gate_exps.weight", {Lin, D, E}) || !shaped(il, "ffn_up_exps.weight", {Lin, D, E}) ||
            !shaped(il, "ffn_down_exps.weight", {D, L, E}) ||
            !shaped(il, "ffn_routed_down.weight", {(uint64_t) out.n_embd, Lin}))
            return false;
    }

    // ---- a pruned file's expert ids: the file's keys, else the sidecar, else its own 0..E-1
    out.keep.assign((size_t) out.n_layers, {});
    std::vector<std::vector<int>> file_keep;
    int file_source = 0;
    bool in_file = false;
    if (r.get("expert_prune.keep")) {
        in_file = true;
        if (!r.int_array("expert_prune.keep", arr) || !r.integer("expert_prune.source_count", v)) return false;
        if (v < out.n_expert || v > (int64_t) 1 << 20)
            return r.no(r.key("expert_prune.source_count") + " = " + std::to_string(v) + " (below the file's " +
                        std::to_string(out.n_expert) + " experts, or implausible)");
        file_source = (int) v;
        const int64_t n_moe = out.n_layers - out.n_layer_dense_lead;
        if ((int64_t) arr.size() != n_moe * out.n_expert)
            return r.no(r.key("expert_prune.keep") + " has " + std::to_string(arr.size()) + " ids for " +
                        std::to_string(n_moe) + " MoE layers x " + std::to_string(out.n_expert) + " experts");
        file_keep.assign((size_t) out.n_layers, {});
        size_t at = 0;
        for (int il = out.n_layer_dense_lead; il < out.n_layers; ++il) {
            auto& ids = file_keep[(size_t) il];
            for (int j = 0; j < out.n_expert; ++j, ++at) {
                if (arr[at] < INT32_MIN || arr[at] > INT32_MAX)
                    return r.no(r.key("expert_prune.keep") + " entry " + std::to_string(at) + " = " +
                                std::to_string(arr[at]) + ", outside the source's " + std::to_string(file_source) +
                                " experts");
                ids.push_back((int) arr[at]);
            }
            if (!detail::check_ids(r, r.key("expert_prune.keep") + " layer " + std::to_string(il), ids, file_source))
                return false;
        }
        if (const strata::MetaValue* m = r.get("expert_prune.method")) {
            if (m->type != strata::MetaType::STRING)
                return r.no(r.key("expert_prune.method") + " is a " + detail::meta_type_name(m->type) + ", not a string");
            out.prune_method = m->s;
        }
    } else if (r.get("expert_prune.source_count")) {
        return r.no(r.key("expert_prune.source_count") + " without " + r.key("expert_prune.keep"));
    }
    bool side = false;
    std::vector<std::vector<int>> side_keep;
    int side_source = 0;
    std::string tool;
    if (!path.empty() && !detail::read_sidecar(r, out, path + ".prune.json", side, side_source, side_keep, tool))
        return false;
    if (in_file && side && side_keep != file_keep)
        return r.no("the prune sidecar " + path + ".prune.json disagrees with " + r.key("expert_prune.keep"));
    if (in_file && side && side_source != file_source)
        return r.no("the prune sidecar " + path + ".prune.json says \"source_expert_count\" " +
                    std::to_string(side_source) + ", " + r.key("expert_prune.source_count") + " " +
                    std::to_string(file_source));
    if (in_file) {
        out.keep = std::move(file_keep);
        out.expert_source_count = file_source;
        out.keep_from = "file";
    } else if (side) {
        out.keep = std::move(side_keep);
        out.expert_source_count = side_source;
        out.keep_from = "sidecar";
        out.prune_method = tool;
    } else {
        for (int il = out.n_layer_dense_lead; il < out.n_layers; ++il)
            for (int j = 0; j < out.n_expert; ++j) out.keep[(size_t) il].push_back(j);
        out.expert_source_count = out.n_expert;
        out.keep_from = "none";
        out.notes.push_back("no expert_prune keys and no prune sidecar: the file's experts are its own ids 0.." +
                            std::to_string(out.n_expert - 1));
    }
    return true;
}

/// One line a field, in the names of ref/kimi_k31.py's KimiK31Config (tools/k31/geo_ref.py writes the oracle's
/// side in the same form): what the test compares, and what --dump prints.
inline std::string dump(const Geo& g) {
    std::string o;
    const auto line = [&](const std::string& k, const std::string& v) { o += k + " " + v + "\n"; };
    const auto i = [&](const char* k, int x) { line(k, std::to_string(x)); };
    const auto fl = [&](const char* k, float x) { line(k, detail::fmt((double) x)); };
    const auto list = [&](const std::string& k, const std::vector<int>& xs) {
        std::string s;
        for (size_t j = 0; j < xs.size(); ++j) s += (j ? " " : "") + std::to_string(xs[j]);
        line(k, s);
    };
    i("n_vocab", g.n_vocab);
    i("n_embd", g.n_embd);
    i("n_layers", g.n_layers);
    i("n_head", g.n_head);
    list("n_head_kv", g.n_head_kv);
    list("mla_layers", g.mla_layers());
    i("q_lora_rank", g.q_lora_rank);
    i("kv_lora_rank", g.kv_lora_rank);
    i("n_rot", g.n_rot);
    i("n_head_k_mla", g.n_head_k_mla);
    i("n_head_v_mla", g.n_head_v_mla);
    i("qk_nope", g.qk_nope);
    i("ssm_d_conv", g.ssm_d_conv);
    i("kda_head_dim", g.kda_head_dim);
    fl("kda_gate_lower_bound", g.kda_gate_lower_bound);
    i("d_inner", g.d_inner());
    i("n_ff", g.n_ff);
    i("n_layer_dense_lead", g.n_layer_dense_lead);
    i("n_expert", g.n_expert);
    i("n_expert_used", g.n_expert_used);
    {
        std::vector<int> moe;
        for (int il = g.n_layer_dense_lead; il < g.n_layers; ++il) moe.push_back(g.n_ff_exp[(size_t) il]);
        list("n_ff_exp", moe);
    }
    i("n_expert_shared", g.n_expert_shared);
    i("n_expert_latent", g.n_expert_latent);
    fl("expert_weights_scale", g.expert_weights_scale);
    line("expert_weights_norm", g.expert_weights_norm ? "true" : "false");
    fl("situ_beta", g.situ_beta);
    fl("situ_linear_beta", g.situ_linear_beta);
    i("attn_res_block_size", g.attn_res_block_size);
    i("idx_n_head", g.idx_n_head);
    i("idx_key_dim", g.idx_key_dim);
    i("idx_top_k", g.idx_top_k);
    line("eps_preset", g.eps_preset_name());
    fl("eps_rms", g.eps_rms);
    fl("eps_indexer_ln", g.eps_indexer_ln);
    fl("eps_mla_latent", g.eps_mla_latent);
    fl("eps_kda_l2", g.eps_kda_l2);
    fl("eps_router_renorm", g.eps_router_renorm);
    line("keep_from", g.keep_from);
    i("expert_source_count", g.expert_source_count);
    for (int il = g.n_layer_dense_lead; il < g.n_layers; ++il) list("keep." + std::to_string(il), g.keep[(size_t) il]);
    return o;
}

}  // namespace k31
