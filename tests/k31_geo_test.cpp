// tests/k31_geo_test.cpp - maya-k31 reads Kimi K3.1's geometry once and refuses what it cannot run (src/k31/k31_geo.hpp).
//
// Host only, as CI builds it (no GPU, no CUDA headers):
//   c++ -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -Iinclude tests/k31_geo_test.cpp -o k31_geo_test
//
// With no arguments (the self-test): GGUF headers with a tensor directory and no tensor data are written to a temporary
// directory and read the way the engine reads them, GgufFile then k31::read.
//   * the released geometry (93 layers, 16 of 896 experts at 7168) is accepted, every field as released; its 93-entry
//     head_count_kv is read whole (GgufFile::metadata() keeps 64 items: layer 92, an MLA layer, would read as KDA);
//     the optional contract keys take their defaults and say so; the eps presets; a per-layer expert_count that is
//     equal everywhere; per-layer expert widths ("thin" experts) with matching slabs; expert ids from the
//     expert_prune keys and from a prune sidecar;
//   * every refusal, each with its exact message: another architecture, each required key missing, keys of the wrong
//     type, every out-of-range value, the latent row, the indexer settings, A_log's length, expert counts that differ
//     between layers (in the keys or in a layer's tensors), slabs of the wrong width, thin experts' sliced latent
//     (per-layer expert_latent_length, routed_norm.*), and every malformed expert-id source (keys, a sidecar that
//     is malformed or exists but cannot be read, and the two disagreeing on ids or on the source count).
// k31::read is host code and allocates on no device, so every refusal comes before any device allocation.
//
//   k31_geo_test --gguf FILE [--ref REF] [--eps moonshot|llama.cpp]
// reads a real file's header: with --ref, its k31::dump must equal REF line for line (REF from
// `python tools/k31/geo_ref.py FILE`, ref/kimi_k31.py's KimiK31Config and the sidecar read with Python's json);
// without, the dump is printed.  The defaults taken go to stderr, as the engine prints them.
#include "../src/k31/k31_geo.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define K31_GETPID _getpid
#else
#include <unistd.h>
#define K31_GETPID getpid
#endif

namespace fs = std::filesystem;

namespace {

int fails = 0, checks = 0;

void expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++fails;
    }
}

// ---- a GGUF v3 header: metadata, then a tensor directory whose data section is empty
template <class T> std::string raw(T v) { return std::string(reinterpret_cast<const char*>(&v), sizeof(v)); }
std::string gstr(const std::string& s) { return raw((uint64_t) s.size()) + s; }

struct Kv {
    std::string key;
    uint32_t type;
    std::string bytes;
};
Kv kv_u32(const std::string& k, uint32_t v) { return {k, 4, raw(v)}; }
Kv kv_i32(const std::string& k, int32_t v) { return {k, 5, raw(v)}; }
Kv kv_f32(const std::string& k, float v) { return {k, 6, raw(v)}; }
Kv kv_bool(const std::string& k, bool v) { return {k, 7, raw((uint8_t) v)}; }
Kv kv_str(const std::string& k, const std::string& s) { return {k, 8, gstr(s)}; }
template <class T> Kv kv_arr(const std::string& k, uint32_t elem, const std::vector<T>& xs) {
    std::string b = raw(elem) + raw((uint64_t) xs.size());
    for (const T& x : xs) b += raw(x);
    return {k, 9, b};
}
Kv kv_arr_i32(const std::string& k, const std::vector<int32_t>& xs) { return kv_arr(k, 5, xs); }
Kv kv_arr_u32(const std::string& k, const std::vector<uint32_t>& xs) { return kv_arr(k, 4, xs); }
Kv kv_arr_f32(const std::string& k, const std::vector<float>& xs) { return kv_arr(k, 6, xs); }
Kv kv_arr_i64(const std::string& k, const std::vector<int64_t>& xs) { return kv_arr(k, 11, xs); }

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;
};

const std::string P = "kimi-k31.";

struct Header {
    std::vector<Kv> kvs;
    std::vector<Tensor> tensors;
    Header& set(const Kv& kv) {
        for (Kv& k : kvs)
            if (k.key == kv.key) {
                k = kv;
                return *this;
            }
        kvs.push_back(kv);
        return *this;
    }
    Header& erase(const std::string& key) {
        for (size_t i = 0; i < kvs.size(); ++i)
            if (kvs[i].key == key) {
                kvs.erase(kvs.begin() + (long) i);
                return *this;
            }
        std::fprintf(stderr, "test bug: no key %s\n", key.c_str());
        std::exit(2);
    }
    Header& shape(const std::string& name, const std::vector<uint64_t>& s) {
        for (Tensor& t : tensors)
            if (t.name == name) {
                t.shape = s;
                return *this;
            }
        std::fprintf(stderr, "test bug: no tensor %s\n", name.c_str());
        std::exit(2);
    }
    Header& drop(const std::string& name) {
        for (size_t i = 0; i < tensors.size(); ++i)
            if (tensors[i].name == name) {
                tensors.erase(tensors.begin() + (long) i);
                return *this;
            }
        std::fprintf(stderr, "test bug: no tensor %s\n", name.c_str());
        std::exit(2);
    }
};

void write_gguf(const fs::path& path, const Header& h) {
    std::string out = "GGUF" + raw((uint32_t) 3) + raw((uint64_t) h.tensors.size()) + raw((uint64_t) h.kvs.size());
    for (const Kv& kv : h.kvs) out += gstr(kv.key) + raw(kv.type) + kv.bytes;
    for (const Tensor& t : h.tensors) {
        out += gstr(t.name) + raw((uint32_t) t.shape.size());
        for (uint64_t d : t.shape) out += raw(d);
        out += raw((uint32_t) 0) + raw((uint64_t) 0);   // F32 at offset 0: the reader never touches tensor data
    }
    out.resize((out.size() + 31) / 32 * 32, '\0');   // the (empty) data section starts at the aligned end
    std::ofstream(path, std::ios::binary).write(out.data(), (std::streamsize) out.size());
}

constexpr int kLayers = 93;
bool released_mla(int il) { return (il + 1) % 4 == 0 || il == kLayers - 1; }

// the released K3.1 header (docs/KIMI-K31.md §3.2) with `E` experts a MoE layer, and the tensors k31::read checks
Header released(uint32_t E = 896) {
    Header h;
    std::vector<int32_t> hkv;
    for (int il = 0; il < kLayers; ++il) hkv.push_back(released_mla(il) ? 1 : 0);
    h.kvs = {kv_str("general.architecture", "kimi-k31"),
             kv_u32("general.alignment", 32),
             kv_u32(P + "block_count", kLayers),
             kv_u32(P + "context_length", 1048576),
             kv_u32(P + "embedding_length", 7168),
             kv_u32(P + "feed_forward_length", 33792),
             kv_u32(P + "attention.head_count", 96),
             kv_arr_i32(P + "attention.head_count_kv", hkv),
             kv_f32(P + "attention.layer_norm_rms_epsilon", 1e-5f),
             kv_u32(P + "expert_count", E),
             kv_u32(P + "expert_used_count", 16),
             kv_u32(P + "expert_group_used_count", 1),
             kv_u32(P + "expert_gating_func", 2),
             kv_u32(P + "vocab_size", 163840),
             kv_u32(P + "ssm.conv_kernel", 4),
             kv_u32(P + "kda.head_dim", 128),
             kv_f32(P + "kda.gate_lower_bound", -5.0f),
             kv_u32(P + "attention.q_lora_rank", 1536),
             kv_u32(P + "attention.kv_lora_rank", 512),
             kv_u32(P + "rope.dimension_count", 64),
             kv_u32(P + "attention.key_length", 576),
             kv_u32(P + "attention.value_length", 512),
             kv_u32(P + "attention.key_length_mla", 192),
             kv_u32(P + "attention.value_length_mla", 128),
             kv_u32(P + "expert_feed_forward_length", 3072),
             kv_u32(P + "expert_shared_count", 2),
             kv_u32(P + "leading_dense_block_count", 1),
             kv_f32(P + "expert_weights_scale", 1.0f),
             kv_bool(P + "expert_weights_norm", true),
             kv_u32(P + "expert_latent_length", 3584),
             kv_f32(P + "activation.situ_beta", 4.0f),
             kv_f32(P + "activation.situ_linear_beta", 25.0f),
             kv_u32(P + "attn_res.block_size", 12),
             kv_u32(P + "attention.indexer.head_count", 32),
             kv_u32(P + "attention.indexer.key_length", 128),
             kv_u32(P + "attention.indexer.top_k", 2048),
             kv_f32(P + "attention.layer_norm_epsilon", 1e-5f),
             kv_str("tokenizer.ggml.pre", "kimi-k2")};
    for (int il = 0; il < kLayers; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        if (!released_mla(il)) h.tensors.push_back({b + "ssm_a", {96}});
        if (il < 1) continue;
        h.tensors.push_back({b + "ffn_gate_inp.weight", {7168, E}});
        h.tensors.push_back({b + "exp_probs_b.bias", {E}});
        h.tensors.push_back({b + "ffn_gate_exps.weight", {3584, 3072, E}});
        h.tensors.push_back({b + "ffn_up_exps.weight", {3584, 3072, E}});
        h.tensors.push_back({b + "ffn_down_exps.weight", {3072, 3584, E}});
    }
    return h;
}

fs::path tmp_dir;
int file_no = 0;

fs::path fresh_path() { return tmp_dir / ("h" + std::to_string(file_no++) + ".gguf"); }

struct Run {
    bool ok = false;
    std::string err;
    k31::Geo g;
    size_t kept_sample = 0;   // what GgufFile::metadata() kept of head_count_kv
};

Run run_file(const fs::path& path, k31::EpsPreset eps = k31::EpsPreset::moonshot) {
    Run r;
    const strata::GgufFile f(path.string());
    if (const strata::MetaValue* v = f.get(P + "attention.head_count_kv")) r.kept_sample = v->items.size();
    r.ok = k31::read(f, path.string(), r.g, r.err, eps);
    return r;
}

// writes `h` (and the sidecar text, when given) to a fresh file and reads it
Run run(const Header& h, const std::string* sidecar = nullptr, k31::EpsPreset eps = k31::EpsPreset::moonshot,
        fs::path* where = nullptr) {
    const fs::path p = fresh_path();
    write_gguf(p, h);
    if (sidecar) std::ofstream(p.string() + ".prune.json", std::ios::binary) << *sidecar;
    if (where) *where = p;
    return run_file(p, eps);
}

void refuse(const std::string& name, const Header& h, const std::string& want, const std::string* sidecar = nullptr) {
    const Run r = run(h, sidecar);
    const std::string full = std::string(k31::kRefusal) + want;
    expect(!r.ok && r.err == full, name + ": want \"" + full + "\", got " + (r.ok ? "accepted" : "\"" + r.err + "\""));
}

Run accept(const std::string& name, const Header& h, const std::string* sidecar = nullptr,
           k31::EpsPreset eps = k31::EpsPreset::moonshot) {
    Run r = run(h, sidecar, eps);
    expect(r.ok, name + ": refused: " + r.err);
    return r;
}

bool has_note(const k31::Geo& g, const std::string& part) {
    for (const auto& n : g.notes)
        if (n.find(part) != std::string::npos) return true;
    return false;
}

std::vector<int> iota_ids(int n) {
    std::vector<int> v;
    for (int j = 0; j < n; ++j) v.push_back(j);
    return v;
}

// ---------------------------------------------------------------- the released geometry
void test_released() {
    const Run r = accept("released", released());
    if (!r.ok) return;
    const k31::Geo& g = r.g;
    expect(r.kept_sample == 64, "GgufFile::metadata() keeps a 64-item sample of head_count_kv (the trap get_full avoids)");
    expect((int) g.n_head_kv.size() == 93 && g.n_head_kv[92] == 1 && g.n_head_kv[91] == 1 && g.n_head_kv[90] == 0,
           "released: head_count_kv read whole (layers 91 and 92 MLA, 90 KDA)");
    std::vector<int> mla;
    for (int il = 0; il < 93; ++il)
        if (released_mla(il)) mla.push_back(il);
    expect(g.mla_layers() == mla && mla.size() == 24, "released: the 24 MLA layers 3, 7, ..., 91, 92");
    expect(g.n_vocab == 163840 && g.n_embd == 7168 && g.n_layers == 93 && g.n_head == 96 && g.n_ctx_train == 1048576,
           "released: the trunk");
    expect(g.q_lora_rank == 1536 && g.kv_lora_rank == 512 && g.n_rot == 64 && g.n_head_k_mla == 192 &&
               g.n_head_v_mla == 128 && g.qk_nope == 128,
           "released: MLA");
    expect(g.ssm_d_conv == 4 && g.kda_head_dim == 128 && g.n_kda_head == 96 && g.d_inner() == 12288 &&
               g.kda_gate_lower_bound == -5.0f,
           "released: KDA");
    expect(g.n_ff == 33792 && g.n_layer_dense_lead == 1 && g.n_expert == 896 && g.n_expert_used == 16 &&
               g.n_expert_shared == 2 && g.n_expert_latent == 3584 && g.expert_weights_scale == 1.0f &&
               g.expert_weights_norm && g.situ_beta == 4.0f && g.situ_linear_beta == 25.0f,
           "released: FFN / MoE");
    bool widths = g.n_ff_exp.size() == 93 && g.n_ff_exp[0] == 0;
    for (int il = 1; il < 93; ++il) widths = widths && g.n_ff_exp[(size_t) il] == 3072;
    expect(widths, "released: expert width 3072 on every MoE layer, 0 on the dense one");
    expect(g.attn_res_block_size == 12 && g.idx_n_head == 32 && g.idx_key_dim == 128 && g.idx_top_k == 2048,
           "released: block residuals and the indexer");
    expect(g.eps_rms == 1e-5f && g.eps_indexer_ln == 1e-5f && g.eps_mla_latent == 1e-6f && g.eps_kda_l2 == 1e-6f &&
               g.eps_router_renorm == 1e-20f && g.eps_preset == k31::EpsPreset::moonshot,
           "released: the moonshot epsilons");
    bool ids = g.keep.size() == 93 && g.keep[0].empty() && g.keep_from == "none" && g.expert_source_count == 896;
    for (int il = 1; il < 93; ++il) ids = ids && g.keep[(size_t) il] == iota_ids(896);
    expect(ids, "released: no prune provenance - the file's experts are its own ids");
    for (const char* k : {"kimi-k31.kda.head_count absent", "kimi-k31.attention.lora_norm_epsilon absent: 1e-06",
                          "kimi-k31.kda.l2_norm_epsilon absent: 1e-06", "kimi-k31.attention.indexer.rope_dim absent: 0",
                          "kimi-k31.attention.indexer.topk_freq absent: 1", "no expert_prune keys and no prune sidecar"})
        expect(has_note(g, k), std::string("released: a note for the default taken: ") + k);
    expect(g.notes.size() == 6, "released: six notes, got " + std::to_string(g.notes.size()));
}

// ---------------------------------------------------------------- accepted variants
void test_variants() {
    // the contract's optional keys written out: no notes for them
    {
        Header h = released();
        h.set(kv_u32(P + "kda.head_count", 96))
            .set(kv_u32(P + "attention.indexer.rope_dim", 0))
            .set(kv_u32(P + "attention.indexer.topk_freq", 1))
            .set(kv_u32(P + "attention.indexer.kpool", 0))
            .set(kv_arr_i32(P + "attention.indexer.types", std::vector<int32_t>(93, 1)))
            .set(kv_u32(P + "expert_group_count", 1))
            .set(kv_f32(P + "attention.lora_norm_epsilon", 2e-6f))
            .set(kv_f32(P + "kda.l2_norm_epsilon", 3e-6f));
        const Run r = accept("explicit contract keys", h);
        expect(r.ok && r.g.eps_mla_latent == 2e-6f && r.g.eps_kda_l2 == 3e-6f, "the file's latent and l2 eps are used");
        expect(r.ok && r.g.notes.size() == 1, "explicit contract keys: only the expert-id note, got " +
                                                  std::to_string(r.g.notes.size()));
        // the llama.cpp preset takes the rms eps for both, whatever the file says
        const Run l = accept("llama.cpp preset", h, nullptr, k31::EpsPreset::llama_cpp);
        expect(l.ok && l.g.eps_mla_latent == 1e-5f && l.g.eps_kda_l2 == 1e-5f &&
                   l.g.eps_preset == k31::EpsPreset::llama_cpp && has_note(l.g, "eps preset llama.cpp"),
               "the llama.cpp preset: the rms eps for the latent and l2 norms");
    }
    // a KDA head count of its own
    {
        Header h = released();
        h.set(kv_u32(P + "kda.head_count", 64));
        for (int il = 0; il < 93; ++il)
            if (!released_mla(il)) h.shape("blk." + std::to_string(il) + ".ssm_a", {64});
        const Run r = accept("kda.head_count 64", h);
        expect(r.ok && r.g.n_kda_head == 64 && r.g.d_inner() == 8192 && r.g.n_head == 96, "kda.head_count is read");
    }
    // expert_count as a per-layer array, equal on every MoE layer (the dense layer's entry is not read)
    {
        Header h = released();
        std::vector<uint32_t> e(93, 896);
        e[0] = 0;
        h.set(kv_arr_u32(P + "expert_count", e));
        const Run r = accept("per-layer expert_count, all equal", h);
        expect(r.ok && r.g.n_expert == 896, "per-layer expert_count: 896");
    }
    // thin experts: a narrower expert width on some layers, the slabs to match
    {
        Header h = released();
        std::vector<uint32_t> d(93, 3072);
        d[40] = 2048;
        d[41] = 1024;
        h.set(kv_arr_u32(P + "expert_feed_forward_length", d));
        for (int il : {40, 41}) {
            const std::string b = "blk." + std::to_string(il) + ".";
            const uint64_t D = d[(size_t) il];
            h.shape(b + "ffn_gate_exps.weight", {3584, D, 896})
                .shape(b + "ffn_up_exps.weight", {3584, D, 896})
                .shape(b + "ffn_down_exps.weight", {D, 3584, 896});
        }
        const Run r = accept("thin experts", h);
        expect(r.ok && r.g.n_ff_exp[40] == 2048 && r.g.n_ff_exp[41] == 1024 && r.g.n_ff_exp[39] == 3072 &&
                   r.g.n_ff_exp[0] == 0,
               "thin experts: the per-layer widths");
    }
    // integer keys of other widths and signedness
    {
        Header h = released();
        h.set(kv_i32(P + "embedding_length", 7168)).set(kv_arr_u32(P + "attention.head_count_kv", [] {
            std::vector<uint32_t> v;
            for (int il = 0; il < 93; ++il) v.push_back(released_mla(il) ? 1u : 0u);
            return v;
        }()));
        const Run r = accept("i32 / u32 arrays", h);
        expect(r.ok && r.g.n_embd == 7168 && r.g.mla_layers().size() == 24, "i32 scalar and u32 array read");
    }
    // a linear up branch (situ_linear_beta <= 0, llama.cpp's convention) is a setting, not a refusal
    {
        Header h = released();
        h.set(kv_f32(P + "activation.situ_linear_beta", 0.0f));
        const Run r = accept("situ_linear_beta 0", h);
        expect(r.ok && r.g.situ_linear_beta == 0.0f, "situ_linear_beta 0 read");
    }
}

// ---------------------------------------------------------------- refusals of the header keys
void test_key_refusals() {
    refuse("no architecture", released().erase("general.architecture"), "general.architecture is missing");
    refuse("kimi-k3", released().set(kv_str("general.architecture", "kimi-k3")),
           "general.architecture is 'kimi-k3'; maya-k31 reads 'kimi-k31' only");
    refuse("glm5-next", released().set(kv_str("general.architecture", "glm5-next")),
           "general.architecture is 'glm5-next'; maya-k31 reads 'kimi-k31' only");
    refuse("architecture not a string", released().set(kv_u32("general.architecture", 7)),
           "general.architecture is '?'; maya-k31 reads 'kimi-k31' only");

    for (const char* k :
         {"block_count", "vocab_size", "context_length", "embedding_length", "attention.head_count",
          "attention.head_count_kv", "attention.q_lora_rank", "attention.kv_lora_rank", "rope.dimension_count",
          "attention.key_length", "attention.value_length", "attention.key_length_mla", "attention.value_length_mla",
          "ssm.conv_kernel", "kda.head_dim", "kda.gate_lower_bound", "feed_forward_length", "leading_dense_block_count",
          "expert_count", "expert_used_count", "expert_shared_count", "expert_latent_length",
          "expert_feed_forward_length", "expert_gating_func", "expert_weights_scale", "expert_weights_norm",
          "activation.situ_beta", "activation.situ_linear_beta", "attn_res.block_size", "attention.indexer.head_count",
          "attention.indexer.key_length", "attention.indexer.top_k", "attention.layer_norm_rms_epsilon",
          "attention.layer_norm_epsilon"})
        refuse(std::string("missing ") + k, released().erase(P + k), P + k + " is missing");

    // types
    refuse("embedding_length f32", released().set(kv_f32(P + "embedding_length", 7168.0f)),
           "kimi-k31.embedding_length is a f32, not an integer");
    refuse("expert_weights_norm u32", released().set(kv_u32(P + "expert_weights_norm", 1)),
           "kimi-k31.expert_weights_norm is a u32, not a bool");
    refuse("situ_beta string", released().set(kv_str(P + "activation.situ_beta", "4")),
           "kimi-k31.activation.situ_beta is a string, not a number");
    refuse("situ_beta bool", released().set(kv_bool(P + "activation.situ_beta", true)),
           "kimi-k31.activation.situ_beta is a bool, not a number");
    refuse("gate_lower_bound inf", released().set(kv_f32(P + "kda.gate_lower_bound", -INFINITY)),
           "kimi-k31.kda.gate_lower_bound = -inf is not finite");
    refuse("head_count_kv scalar", released().set(kv_u32(P + "attention.head_count_kv", 1)),
           "kimi-k31.attention.head_count_kv is a u32, not a per-layer array");
    refuse("head_count_kv of f32", released().set(kv_arr_f32(P + "attention.head_count_kv", std::vector<float>(93, 0))),
           "kimi-k31.attention.head_count_kv is an array of f32, not of integers");
    refuse("head_count_kv 92 entries",
           released().set(kv_arr_i32(P + "attention.head_count_kv", std::vector<int32_t>(92, 0))),
           "kimi-k31.attention.head_count_kv has 92 entries for block_count 93");
    {
        Header h = released();
        std::vector<int32_t> hkv(93, 0);
        hkv[5] = 2;
        refuse("head_count_kv 2", h.set(kv_arr_i32(P + "attention.head_count_kv", hkv)),
               "kimi-k31.attention.head_count_kv[5] = 2 (0 marks a KDA layer, 1 an MLA layer; nothing else)");
    }
    refuse("expert_count array of 64", released().set(kv_arr_u32(P + "expert_count", std::vector<uint32_t>(64, 896))),
           "kimi-k31.expert_count has 64 entries for block_count 93");

    // ranges
    const auto u = [](const char* k, uint32_t v) { return released().set(kv_u32(P + k, v)); };
    const auto fl = [](const char* k, float v) { return released().set(kv_f32(P + k, v)); };
    refuse("block_count 0", u("block_count", 0), "kimi-k31.block_count = 0 (1 to 65536 layers)");
    refuse("vocab_size 0", u("vocab_size", 0), "kimi-k31.vocab_size = 0 (a vocabulary of 1 to 16M tokens)");
    refuse("context_length 0", u("context_length", 0), "kimi-k31.context_length = 0 (a positive context)");
    refuse("width 1M + 1", u("embedding_length", (1u << 20) + 1),
           "kimi-k31.embedding_length = 1048577 (a width of 1 to 1M)");
    refuse("width 0", u("embedding_length", 0), "kimi-k31.embedding_length = 0 (a width of 1 to 1M)");
    refuse("head_count 0", u("attention.head_count", 0),
           "kimi-k31.attention.head_count = 0 (1 to 1024 attention heads)");
    refuse("q_lora 0", u("attention.q_lora_rank", 0), "kimi-k31.attention.q_lora_rank = 0 (a q_a rank of 1 to 1M)");
    refuse("q_lora 1M + 1", u("attention.q_lora_rank", (1u << 20) + 1),
           "kimi-k31.attention.q_lora_rank = 1048577 (a q_a rank of 1 to 1M)");
    const std::string latent = "; k31_mla_attention reads a 576-wide key row (512 + 64) and a 512-wide value row only";
    refuse("kv_lora 448", u("attention.kv_lora_rank", 448),
           "the MLA latent row is kv_lora_rank 448 + rope 64 (key_length 576, value_length 512)" + latent);
    refuse("key_length 640", u("attention.key_length", 640),
           "the MLA latent row is kv_lora_rank 512 + rope 64 (key_length 640, value_length 512)" + latent);
    refuse("value_length 576", u("attention.value_length", 576),
           "the MLA latent row is kv_lora_rank 512 + rope 64 (key_length 576, value_length 576)" + latent);
    refuse("rope 128", released().set(kv_u32(P + "rope.dimension_count", 128)).set(kv_u32(P + "attention.key_length", 640)),
           "the MLA latent row is kv_lora_rank 512 + rope 128 (key_length 640, value_length 512)" + latent);
    refuse("key_length_mla 64", u("attention.key_length_mla", 64),
           "kimi-k31.attention.key_length_mla = 64 leaves no nope part after rope 64 (or exceeds 1024)");
    refuse("value_length_mla 0", u("attention.value_length_mla", 0),
           "kimi-k31.attention.value_length_mla = 0 (a value head of 1 to 1024)");
    refuse("conv 1", u("ssm.conv_kernel", 1), "kimi-k31.ssm.conv_kernel = 1 (glm_kda_conv takes 2 to 17 taps)");
    refuse("conv 18", u("ssm.conv_kernel", 18), "kimi-k31.ssm.conv_kernel = 18 (glm_kda_conv takes 2 to 17 taps)");
    refuse("kda head_dim 512", u("kda.head_dim", 512),
           "kimi-k31.kda.head_dim = 512 (glm_kda_recurrence holds a head of at most 256)");
    refuse("kda.head_count 0", u("kda.head_count", 0), "kimi-k31.kda.head_count = 0 (1 to 1024 KDA heads)");
    refuse("gate_lower_bound 0", fl("kda.gate_lower_bound", 0.0f),
           "kimi-k31.kda.gate_lower_bound = 0 (it must be below 0)");
    refuse("dense ff 0", u("feed_forward_length", 0), "kimi-k31.feed_forward_length = 0 (a dense FFN of 1 to 1M)");
    refuse("no MoE layer", u("leading_dense_block_count", 93),
           "kimi-k31.leading_dense_block_count = 93 (at least one MoE layer after the dense ones)");
    refuse("E 1025", u("expert_count", 1025), "kimi-k31.expert_count = 1025 (k31_route holds 1 to 1024 experts)");
    refuse("E 0", u("expert_count", 0), "kimi-k31.expert_count = 0 (k31_route holds 1 to 1024 experts)");
    {
        std::vector<uint32_t> e(93, 896);
        e[40] = 895;
        refuse("expert_count differs", released().set(kv_arr_u32(P + "expert_count", e)),
               "kimi-k31.expert_count differs between layers (layer 1: 896, layer 40: 895); every MoE layer must keep "
               "the same count");
    }
    refuse("k 17", u("expert_used_count", 17), "kimi-k31.expert_used_count = 17 (k31_route picks 1 to 16 experts a token)");
    refuse("k 0", u("expert_used_count", 0), "kimi-k31.expert_used_count = 0 (k31_route picks 1 to 16 experts a token)");
    refuse("k > E", u("expert_count", 12),
           "kimi-k31.expert_used_count = 16 picks more experts than the 12 a layer has");
    refuse("shared 17", u("expert_shared_count", 17), "kimi-k31.expert_shared_count = 17 (0 to 16 shared experts)");
    refuse("latent > width", u("expert_latent_length", 8192),
           "kimi-k31.expert_latent_length = 8192 (the latent is 1 to embedding_length wide)");
    // thin experts' sliced latent: refused (a narrower expert width alone is accepted, test_variants)
    {
        const std::string thin = "thin expert geometry (D-FMT B: a sliced latent, routed_norm.*) is not built in";
        refuse("routed_norm.divisor", u("routed_norm.divisor", 3584), "kimi-k31.routed_norm.divisor: " + thin);
        refuse("routed_norm.folded", released().set(kv_bool(P + "routed_norm.folded", true)),
               "kimi-k31.routed_norm.folded: " + thin);
        // a sliced latent with its slabs to match and the norm keys: refused at the keys, not run as a plain latent
        Header h = released();
        h.set(kv_u32(P + "expert_latent_length", 2048))
            .set(kv_u32(P + "routed_norm.divisor", 3584))
            .set(kv_bool(P + "routed_norm.folded", true));
        for (int il = 1; il < 93; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            h.shape(b + "ffn_gate_exps.weight", {2048, 3072, 896})
                .shape(b + "ffn_up_exps.weight", {2048, 3072, 896})
                .shape(b + "ffn_down_exps.weight", {3072, 2048, 896});
        }
        refuse("sliced latent with routed_norm keys", h, "kimi-k31.routed_norm.divisor: " + thin);
        std::vector<uint32_t> lat(93, 3584);
        lat[40] = 2048;
        refuse("per-layer expert_latent_length", released().set(kv_arr_u32(P + "expert_latent_length", lat)),
               "kimi-k31.expert_latent_length is per-layer: " + thin);
    }
    refuse("expert width 0", u("expert_feed_forward_length", 0),
           "kimi-k31.expert_feed_forward_length = 0 at layer 1 (an expert of 1 to 1M)");
    refuse("gating 1", u("expert_gating_func", 1),
           "kimi-k31.expert_gating_func = 1 (only sigmoid routing, 2, is K3.1's)");
    refuse("groups 8", u("expert_group_count", 8), "kimi-k31.expert_group_count = 8 (grouped expert routing is not K3.1's)");
    refuse("groups used 2", u("expert_group_used_count", 2),
           "kimi-k31.expert_group_used_count = 2 (grouped expert routing is not K3.1's)");
    refuse("weights scale 0", fl("expert_weights_scale", 0.0f),
           "kimi-k31.expert_weights_scale = 0 (it must be above 0)");
    refuse("situ_beta 0", fl("activation.situ_beta", 0.0f), "kimi-k31.activation.situ_beta = 0 (it must be above 0)");
    refuse("attn_res 0", u("attn_res.block_size", 0),
           "kimi-k31.attn_res.block_size = 0 (K3.1 always mixes blocks of 1 to block_count layers)");
    refuse("attn_res 94", u("attn_res.block_size", 94),
           "kimi-k31.attn_res.block_size = 94 (K3.1 always mixes blocks of 1 to block_count layers)");

    // the indexer
    refuse("unknown indexer key", u("attention.indexer.kpool_select_tail", 1),
           "kimi-k31.attention.indexer.kpool_select_tail: an indexer setting maya-k31 does not know (a per-token "
           "indexer only)");
    refuse("indexer heads 33", u("attention.indexer.head_count", 33),
           "kimi-k31.attention.indexer.head_count = 33 (the indexer kernels take 1 to 32 heads)");
    refuse("indexer key 256", u("attention.indexer.key_length", 256),
           "kimi-k31.attention.indexer.key_length = 256 (the indexer kernels take keys of 1 to 128)");
    refuse("indexer top_k 0", u("attention.indexer.top_k", 0),
           "kimi-k31.attention.indexer.top_k = 0 (the indexer keeps at least one key)");
    refuse("indexer rope 64", u("attention.indexer.rope_dim", 64),
           "kimi-k31.attention.indexer.rope_dim = 64 (K3.1's indexer rotates nothing: index_rope_dim must be 0)");
    refuse("indexer topk_freq 2", u("attention.indexer.topk_freq", 2),
           "kimi-k31.attention.indexer.topk_freq = 2 (every MLA layer runs its own indexer: index_topk_freq must be 1)");
    refuse("pooled indexer", u("attention.indexer.kpool", 4),
           "kimi-k31.attention.indexer.kpool = 4 (a pooled indexer; K3.1 scores every key)");
    {
        std::vector<int32_t> types(93, 1);
        types[7] = 0;
        refuse("shared selection", released().set(kv_arr_i32(P + "attention.indexer.types", types)),
               "kimi-k31.attention.indexer.types[7] = 0 (MLA layer 7 would share another layer's selection)");
    }

    // epsilons
    refuse("rms eps 0", fl("attention.layer_norm_rms_epsilon", 0.0f),
           "kimi-k31.attention.layer_norm_rms_epsilon = 0 (an eps in (0, 1))");
    refuse("indexer ln eps 0", fl("attention.layer_norm_epsilon", 0.0f),
           "kimi-k31.attention.layer_norm_epsilon = 0 (an eps in (0, 1))");
    refuse("lora eps 0", fl("attention.lora_norm_epsilon", 0.0f),
           "kimi-k31.attention.lora_norm_epsilon = 0 (an eps in (0, 1))");
    refuse("l2 eps 1", fl("kda.l2_norm_epsilon", 1.0f), "kimi-k31.kda.l2_norm_epsilon = 1 (an eps in (0, 1))");
}

// ---------------------------------------------------------------- refusals from the tensor directory
void test_tensor_refusals() {
    refuse("A_log 128", released().shape("blk.0.ssm_a", {128}),
           "blk.0.ssm_a is [128] for 96 KDA heads (A_log is sliced to the head count)");
    refuse("A_log for other heads", released().set(kv_u32(P + "kda.head_count", 64)),
           "blk.0.ssm_a is [96] for 64 KDA heads (A_log is sliced to the head count)");
    refuse("A_log 2-d", released().shape("blk.4.ssm_a", {96, 1}),
           "blk.4.ssm_a is [96, 1] for 96 KDA heads (A_log is sliced to the head count)");
    refuse("A_log missing", released().drop("blk.5.ssm_a"), "blk.5.ssm_a is missing");
    refuse("layer routes 895", released().shape("blk.40.ffn_gate_inp.weight", {7168, 895}),
           "blk.40.ffn_gate_inp.weight is [7168, 895]: the layer does not route to the file's 896 experts (every MoE "
           "layer must keep the same count)");
    refuse("router width", released().shape("blk.40.ffn_gate_inp.weight", {4096, 896}),
           "blk.40.ffn_gate_inp.weight is [4096, 896], the geometry says [7168, 896]");
    refuse("router missing", released().drop("blk.92.ffn_gate_inp.weight"), "blk.92.ffn_gate_inp.weight is missing");
    refuse("bias 895", released().shape("blk.40.exp_probs_b.bias", {895}),
           "blk.40.exp_probs_b.bias is [895], the geometry says [896]");
    refuse("down slab 895", released().shape("blk.40.ffn_down_exps.weight", {3072, 3584, 895}),
           "blk.40.ffn_down_exps.weight is [3072, 3584, 895], the geometry says [3072, 3584, 896]");
    refuse("up slab missing", released().drop("blk.40.ffn_up_exps.weight"), "blk.40.ffn_up_exps.weight is missing");
    refuse("up slab 895", released().shape("blk.40.ffn_up_exps.weight", {3584, 3072, 895}),
           "blk.40.ffn_up_exps.weight is [3584, 3072, 895], the geometry says [3584, 3072, 896]");
    refuse("gate slab latent", released().shape("blk.1.ffn_gate_exps.weight", {7168, 3072, 896}),
           "blk.1.ffn_gate_exps.weight is [7168, 3072, 896], the geometry says [3584, 3072, 896]");
    {
        std::vector<uint32_t> d(93, 3072);
        d[40] = 2048;
        refuse("thin width without thin slabs", released().set(kv_arr_u32(P + "expert_feed_forward_length", d)),
               "blk.40.ffn_gate_exps.weight is [3584, 3072, 896], the geometry says [3584, 2048, 896]");
    }
}

// ---------------------------------------------------------------- a pruned file's expert ids
// E = 128 of 896: layer il keeps source ids (j * 7 + il) % 896, j = 0..127 (distinct: 7 * 127 < 896)
std::vector<int> ids_of(int il) {
    std::vector<int> v;
    for (int j = 0; j < 128; ++j) v.push_back((j * 7 + il) % 896);
    return v;
}
std::vector<int32_t> flat_keep() {
    std::vector<int32_t> v;
    for (int il = 1; il < 93; ++il)
        for (int x : ids_of(il)) v.push_back(x);
    return v;
}
Header pruned_with_keys() {
    return released(128)
        .set(kv_u32(P + "expert_prune.source_count", 896))
        .set(kv_str(P + "expert_prune.method", "usage"))
        .set(kv_arr_i32(P + "expert_prune.keep", flat_keep()));
}
std::string ids_json(const std::vector<int>& ids) {
    std::string s = "[";
    for (size_t j = 0; j < ids.size(); ++j) s += (j ? ", " : "") + std::to_string(ids[j]);
    return s + "]";
}
// the converter's / prune tool's sidecar, with one layer's list replaced when asked
std::string sidecar_text(const std::string& arch = "kimi-k31", int count = 128,
                         const std::function<std::string(int)>& layer = nullptr, int source = 896) {
    std::string s = "{\"tool\": \"prune v1\", \"arch\": \"" + arch + "\", \"source_expert_count\": " +
                    std::to_string(source) + ", \"input_expert_count\": 896, \"expert_count\": " +
                    std::to_string(count) + ", \"layers\": {";
    bool first = true;
    for (int il = 1; il < 93; ++il) {
        std::string body = layer ? layer(il) : ids_json(ids_of(il));
        if (body.empty()) continue;   // the layer left out
        s += (first ? "" : ", ") + body;
        first = false;
    }
    return s + "}, \"source_sizes\": [1, 2.5e3, null, true]}";
}
std::string entry(int il, const std::string& list) { return "\"" + std::to_string(il) + "\": " + list; }
std::string normal(int il) { return entry(il, ids_json(ids_of(il))); }

void test_keep() {
    // from the file's keys
    {
        const Run r = accept("keep from keys", pruned_with_keys());
        bool ok = r.ok && r.g.keep_from == "file" && r.g.expert_source_count == 896 && r.g.prune_method == "usage" &&
                  r.g.keep[0].empty();
        for (int il = 1; il < 93 && ok; ++il) ok = r.g.keep[(size_t) il] == ids_of(il);
        expect(ok, "keep from keys: every layer's ids in the file's order");
    }
    // from the sidecar
    {
        const std::string side = sidecar_text("kimi-k31", 128, normal);
        const Run r = accept("keep from sidecar", released(128), &side);
        bool ok = r.ok && r.g.keep_from == "sidecar" && r.g.expert_source_count == 896 && r.g.prune_method == "prune v1";
        for (int il = 1; il < 93 && ok; ++il) ok = r.g.keep[(size_t) il] == ids_of(il);
        expect(ok, "keep from sidecar: every layer's ids in the file's order");
        // both, agreeing
        const Run b = accept("keys and sidecar agree", pruned_with_keys(), &side);
        expect(b.ok && b.g.keep_from == "file", "keys and sidecar agree: the keys are taken");
    }
    // the keys, malformed
    {
        std::vector<int32_t> k = flat_keep();
        k.pop_back();
        refuse("keep short", pruned_with_keys().set(kv_arr_i32(P + "expert_prune.keep", k)),
               "kimi-k31.expert_prune.keep has 11775 ids for 92 MoE layers x 128 experts");
        k = flat_keep();
        k[0] = 896;
        refuse("keep id 896", pruned_with_keys().set(kv_arr_i32(P + "expert_prune.keep", k)),
               "kimi-k31.expert_prune.keep layer 1 entry 0 = 896, outside the source's 896 experts");
        k = flat_keep();
        k[128 + 5] = -1;
        refuse("keep id -1", pruned_with_keys().set(kv_arr_i32(P + "expert_prune.keep", k)),
               "kimi-k31.expert_prune.keep layer 2 entry 5 = -1, outside the source's 896 experts");
        k = flat_keep();
        k[1] = k[0];
        refuse("keep twice", pruned_with_keys().set(kv_arr_i32(P + "expert_prune.keep", k)),
               "kimi-k31.expert_prune.keep layer 1 keeps expert " + std::to_string(k[0]) + " twice");
        {
            const std::vector<int32_t> k32 = flat_keep();
            std::vector<int64_t> k64(k32.begin(), k32.end());
            k64[128 * 2 + 3] = (int64_t) 1 << 33;
            refuse("keep i64 past int32", pruned_with_keys().set(kv_arr_i64(P + "expert_prune.keep", k64)),
                   "kimi-k31.expert_prune.keep entry 259 = 8589934592, outside the source's 896 experts");
        }
        refuse("keep of f32", pruned_with_keys().set(kv_arr_f32(P + "expert_prune.keep", std::vector<float>(11776, 0))),
               "kimi-k31.expert_prune.keep is an array of f32, not of integers");
        refuse("keep scalar", pruned_with_keys().set(kv_u32(P + "expert_prune.keep", 3)),
               "kimi-k31.expert_prune.keep is a u32, not an array");
        refuse("keep without source", pruned_with_keys().erase(P + "expert_prune.source_count"),
               "kimi-k31.expert_prune.source_count is missing");
        refuse("source below E", pruned_with_keys().set(kv_u32(P + "expert_prune.source_count", 100)),
               "kimi-k31.expert_prune.source_count = 100 (below the file's 128 experts, or implausible)");
        refuse("source without keep", released(128).set(kv_u32(P + "expert_prune.source_count", 896)),
               "kimi-k31.expert_prune.source_count without kimi-k31.expert_prune.keep");
        refuse("method not a string", pruned_with_keys().set(kv_u32(P + "expert_prune.method", 1)),
               "kimi-k31.expert_prune.method is a u32, not a string");
    }
    // the sidecar, malformed: its path is the next fresh file's
    const auto side_refuse = [&](const std::string& name, const Header& h, const std::string& text,
                                 const std::string& want_after_path) {
        const std::string path = (tmp_dir / ("h" + std::to_string(file_no) + ".gguf")).string() + ".prune.json";
        refuse(name, h, "the prune sidecar " + path + want_after_path, &text);
    };
    const Header h128 = released(128);
    {
        const std::string t = "{\"arch\": \"kimi-k31";
        side_refuse("sidecar truncated", h128, t, " is not valid JSON (unterminated string at byte " +
                                                      std::to_string(t.size()) + ")");
        const std::string d = "{\"arch\": \"kimi-k31\", \"arch\": \"x\"}";
        side_refuse("sidecar duplicate key", h128, d, " is not valid JSON (duplicate key at byte " +
                                                          std::to_string(d.find("\"x\"") - 2) + ")");
        const std::string g = sidecar_text("kimi-k31", 128, normal) + " x";
        side_refuse("sidecar trailing data", h128, g, " is not valid JSON (trailing data at byte " +
                                                          std::to_string(g.size() - 1) + ")");
        const std::string z = "{\"expert_count\": 0128}";
        side_refuse("sidecar leading zero", h128, z, " is not valid JSON (leading zero at byte " +
                                                         std::to_string(z.find("0128") + 4) + ")");
    }
    side_refuse("sidecar not an object", h128, "[1, 2]", " is not a JSON object");
    side_refuse("sidecar without arch", h128, "{\"layers\": {}}", " has no \"arch\" string");
    side_refuse("sidecar other arch", h128, sidecar_text("glm5-next", 128, normal), " is for 'glm5-next', not 'kimi-k31'");
    side_refuse("sidecar count", h128, sidecar_text("kimi-k31", 147, normal), " keeps 147 experts a layer, the file has 128");
    side_refuse("sidecar without source count", h128, "{\"arch\": \"kimi-k31\", \"expert_count\": 128}",
                " has no \"source_expert_count\"");
    side_refuse("sidecar count not an integer", h128,
                "{\"arch\": \"kimi-k31\", \"source_expert_count\": 896, \"expert_count\": 128.0}",
                ": \"expert_count\" is not an integer");
    side_refuse("sidecar without layers", h128,
                "{\"arch\": \"kimi-k31\", \"source_expert_count\": 896, \"expert_count\": 128}",
                " has no \"layers\" object");
    side_refuse("sidecar source count below E", h128,
                "{\"arch\": \"kimi-k31\", \"source_expert_count\": 100, \"expert_count\": 128}",
                ": \"source_expert_count\" 100 is below the file's 128 experts or implausible");
    side_refuse("sidecar layer left out", h128,
                sidecar_text("kimi-k31", 128, [](int il) { return il == 40 ? std::string() : normal(il); }),
                " has no ids for layer 40");
    side_refuse("sidecar dense layer", h128,
                sidecar_text("kimi-k31", 128, [](int il) { return il == 1 ? entry(0, ids_json(ids_of(1))) + ", " + normal(1) : normal(il); }),
                " lists layer \"0\", which is not a MoE layer of this file");
    side_refuse("sidecar layer 01", h128,
                sidecar_text("kimi-k31", 128, [](int il) { return il == 1 ? "\"01\": " + ids_json(ids_of(1)) : normal(il); }),
                " lists layer \"01\", which is not a MoE layer of this file");
    side_refuse("sidecar layer 93", h128,
                sidecar_text("kimi-k31", 128, [](int il) { return il == 92 ? normal(92) + ", " + entry(93, "[]") : normal(il); }),
                " lists layer \"93\", which is not a MoE layer of this file");
    side_refuse("sidecar short layer", h128, sidecar_text("kimi-k31", 128, [](int il) {
                    std::vector<int> v = ids_of(il);
                    if (il == 40) v.pop_back();
                    return entry(il, ids_json(v));
                }),
                ": layer 40 lists 127 experts, the file has 128");
    side_refuse("sidecar id twice", h128, sidecar_text("kimi-k31", 128, [](int il) {
                    std::vector<int> v = ids_of(il);
                    if (il == 40) v[9] = v[2];
                    return entry(il, ids_json(v));
                }),
                ": layer 40 keeps expert " + std::to_string(ids_of(40)[2]) + " twice");
    side_refuse("sidecar id 900", h128, sidecar_text("kimi-k31", 128, [](int il) {
                    std::vector<int> v = ids_of(il);
                    if (il == 40) v[3] = 900;
                    return entry(il, ids_json(v));
                }),
                ": layer 40 entry 3 = 900, outside the source's 896 experts");
    side_refuse("sidecar id 1.5", h128, sidecar_text("kimi-k31", 128, [](int il) {
                    return il == 40 ? entry(il, "[1.5" + ids_json(ids_of(il)).substr(2)) : normal(il);
                }),
                ": layer 40 has an id that is not an integer");
    side_refuse("sidecar layer not a list", h128,
                sidecar_text("kimi-k31", 128, [](int il) { return il == 40 ? entry(il, "{}") : normal(il); }),
                ": layer 40 is not an array");
    // keys and sidecar disagreeing
    side_refuse("keys and sidecar disagree", pruned_with_keys(), sidecar_text("kimi-k31", 128, [](int il) {
                    std::vector<int> v = ids_of(il);
                    if (il == 40) std::swap(v[0], v[1]);
                    return entry(il, ids_json(v));
                }),
                " disagrees with kimi-k31.expert_prune.keep");
    side_refuse("keys and sidecar disagree on the source count", pruned_with_keys(),
                sidecar_text("kimi-k31", 128, normal, 900),
                " says \"source_expert_count\" 900, kimi-k31.expert_prune.source_count 896");
    // a sidecar path that exists but is not a readable regular file: refused, never taken for "no sidecar" (the
    // file would read as unpruned, its experts as ids 0..E-1)
    const auto odd_sidecar = [&](const std::string& name, const std::function<bool(const std::string&)>& make,
                                 const std::string& want_after_path) {
        const fs::path p = fresh_path();
        write_gguf(p, h128);
        const std::string side = p.string() + ".prune.json";
        if (!make(side)) {
            std::fprintf(stderr, "k31_geo_test: skipped \"%s\" (cannot set the sidecar up here)\n", name.c_str());
            return;
        }
        const Run r = run_file(p);
        const std::string full = std::string(k31::kRefusal) + "the prune sidecar " + side + want_after_path;
        expect(!r.ok && r.err == full, name + ": want \"" + full + "\", got " + (r.ok ? "accepted" : "\"" + r.err + "\""));
    };
    const std::string not_regular = " is not a regular file (a directory, a link to nothing, or a path that cannot be read)";
    odd_sidecar("sidecar a directory", [](const std::string& s) { return fs::create_directory(s); }, not_regular);
    odd_sidecar("sidecar a dangling link", [](const std::string& s) {
                    std::error_code ec;
                    fs::create_symlink(s + ".gone", s, ec);
                    return !ec;
                }, not_regular);
    odd_sidecar("sidecar unreadable", [](const std::string& s) {
                    std::ofstream(s, std::ios::binary) << sidecar_text("kimi-k31", 128, normal);
                    std::error_code ec;
                    fs::permissions(s, fs::perms::none, ec);
                    return !ec && !std::ifstream(s);   // (as root the file still opens: nothing to test)
                }, " cannot be read");
}

int self_test() {
    tmp_dir = fs::temp_directory_path() / ("k31_geo_test-" + std::to_string((long) K31_GETPID()));
    fs::create_directories(tmp_dir);
    test_released();
    test_variants();
    test_key_refusals();
    test_tensor_refusals();
    test_keep();
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
    if (fails) {
        std::fprintf(stderr, "k31_geo_test: %d of %d checks FAILED\n", fails, checks);
        return 1;
    }
    std::printf("k31_geo_test: %d checks passed (%d headers)\n", checks, file_no);
    return 0;
}

// ---------------------------------------------------------------- a real file against the oracle's dump
int file_mode(const std::string& gguf, const std::string& ref, k31::EpsPreset eps) {
    k31::Geo g;
    std::string err;
    try {
        const strata::GgufFile f(gguf);
        if (!k31::read(f, gguf, g, err, eps)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "k31_geo_test: %s: %s\n", gguf.c_str(), e.what());
        return 1;
    }
    for (const auto& n : g.notes) std::fprintf(stderr, "maya-k31: %s\n", n.c_str());
    const std::string mine = k31::dump(g);
    if (ref.empty()) {
        std::fputs(mine.c_str(), stdout);
        return 0;
    }
    std::string theirs;
    if (!k31::detail::read_text(ref, theirs)) {
        std::fprintf(stderr, "k31_geo_test: cannot read %s\n", ref.c_str());
        return 1;
    }
    std::istringstream a(mine), b(theirs);
    std::string la, lb;
    int n = 0, diff = 0;
    for (;;) {
        const bool ga = (bool) std::getline(a, la), gb = (bool) std::getline(b, lb);
        if (!ga && !gb) break;
        ++n;
        if (!ga || !gb || la != lb) {
            if (++diff <= 8)
                std::fprintf(stderr, "line %d:\n  maya-k31: %.200s\n  oracle:   %.200s\n", n, ga ? la.c_str() : "(none)",
                             gb ? lb.c_str() : "(none)");
        }
    }
    if (diff) {
        std::fprintf(stderr, "k31_geo_test: %s: %d of %d lines differ from %s\n", gguf.c_str(), diff, n, ref.c_str());
        return 1;
    }
    std::printf("k31_geo_test: %s: all %d lines equal the oracle's (%s, expert ids: %s)\n", gguf.c_str(), n,
                g.eps_preset_name(), g.keep_from.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, ref;
    k31::EpsPreset eps = k31::EpsPreset::moonshot;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--ref" && i + 1 < argc) ref = argv[++i];
        else if (a == "--eps" && i + 1 < argc) {
            const std::string e = argv[++i];
            if (e == "moonshot") eps = k31::EpsPreset::moonshot;
            else if (e == "llama.cpp") eps = k31::EpsPreset::llama_cpp;
            else {
                std::fprintf(stderr, "--eps moonshot|llama.cpp\n");
                return 2;
            }
        } else if (a == "--selftest") {
        } else {
            std::fprintf(stderr, "usage: k31_geo_test [--selftest] | --gguf FILE [--ref REF] [--eps moonshot|llama.cpp]\n");
            return 2;
        }
    }
    if (!gguf.empty()) return file_mode(gguf, ref, eps);
    return self_test();
}
