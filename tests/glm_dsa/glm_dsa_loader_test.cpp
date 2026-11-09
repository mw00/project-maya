// tests/glm_dsa/glm_dsa_loader_test.cpp - glm-dsa B1.10: the glm-dsa loader on synthetic GGUF headers (CPU only).
//
// The test writes its own GGUF v3 files: the full GLM-5.3 744B tensor directory (78 layers + the NextN blk.78, real
// shapes and types) with NO tensor data - the geometry reader, the shape check and the pack loader read headers and
// directories only, so a few hundred KB stand in for 238 GB.  Checked:
//
//   1. the GGUF reader keeps every item of numeric/bool arrays (a 79-entry bool array round-trips) and still samples
//      string arrays (B1.1);
//   2. the glm-dsa geometry: every value of a GLM-5.3 header, the helpers, the indexer schedule from a 78- or 79-entry
//      indexer.types and from llama.cpp's back-compat rule, the idx LayerNorm eps override (B1.2, B1.3);
//   3. refusals: each required key missing, key_length / value_length / gating / rope-scaling mutations, layer 0
//      shared, a truncated indexer.types, an unknown architecture (B1.3);
//   4. Flash (glm5-next) geometry unchanged: Maya's values and Flash-neutral helpers (B1.2);
//   5. a 3-shard pack with a metadata-only shard 1 (unsloth's shape, #83) through Glm5Model::load_pack: the shards
//      open, native_experts.txt agrees with the directories, the shapes check out, and the load stops cleanly at the
//      glm-dsa forward gate; a wrong dim or a missing full-layer indexer tensor fails naming the tensor (B1.4);
//   6. load_pack_env reads the pack's geometry before the split bounds: 26,54 is a valid 78-layer split, 25 is not
//      a full-indexer layer (B1.5);
//   7. the NextN lists, the prompt gate, the derived constants (B1.8, B1.9, B2.6).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_dsa_split.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace fs = std::filesystem;

namespace strata::core::glmpf {
bool f16_covers(int type, bool dsa);   // src/core/glm_prefill_dsa.hpp (an internal header: CUDA / cuBLAS includes)
}
using strata::core::Glm5Geometry;
using strata::core::GlmArch;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------- a GGUF v3 writer (header + directory only)

enum : uint32_t { kU32 = 4, kI32 = 5, kF32 = 6, kBool = 7, kStr = 8, kArr = 9, kU64 = 10 };

struct Writer {
    struct Kv {
        std::string key;
        std::vector<uint8_t> bytes;   // type id + payload
    };
    struct Tensor {
        std::string name;
        std::vector<uint64_t> shape;
        uint32_t type = 0;
        uint64_t offset = 0;
    };
    std::vector<Kv> kvs;
    std::vector<Tensor> tensors;

    template <class T> static void put(std::vector<uint8_t>& b, T v) {
        const size_t n = b.size();
        b.resize(n + sizeof(T));
        std::memcpy(b.data() + n, &v, sizeof(T));
    }
    static void put_str(std::vector<uint8_t>& b, const std::string& s) {
        put<uint64_t>(b, s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void set(const std::string& key, std::vector<uint8_t> bytes) {
        for (auto& kv : kvs)
            if (kv.key == key) {
                kv.bytes = std::move(bytes);
                return;
            }
        kvs.push_back({key, std::move(bytes)});
    }
    void erase(const std::string& key) {
        for (size_t i = 0; i < kvs.size(); ++i)
            if (kvs[i].key == key) {
                kvs.erase(kvs.begin() + (long) i);
                return;
            }
    }
    void u32(const std::string& k, uint32_t v) {
        std::vector<uint8_t> b;
        put<uint32_t>(b, kU32);
        put<uint32_t>(b, v);
        set(k, b);
    }
    void f32(const std::string& k, float v) {
        std::vector<uint8_t> b;
        put<uint32_t>(b, kF32);
        put<float>(b, v);
        set(k, b);
    }
    void boolean(const std::string& k, bool v) {
        std::vector<uint8_t> b;
        put<uint32_t>(b, kBool);
        put<uint8_t>(b, v ? 1 : 0);
        set(k, b);
    }
    void str(const std::string& k, const std::string& v) {
        std::vector<uint8_t> b;
        put<uint32_t>(b, kStr);
        put_str(b, v);
        set(k, b);
    }
    void arr_bool(const std::string& k, const std::vector<int>& v) {
        std::vector<uint8_t> b;
        put<uint32_t>(b, kArr);
        put<uint32_t>(b, kBool);
        put<uint64_t>(b, v.size());
        for (int x : v) put<uint8_t>(b, x ? 1 : 0);
        set(k, b);
    }
    void arr_i32(const std::string& k, const std::vector<int>& v) {
        std::vector<uint8_t> b;
        put<uint32_t>(b, kArr);
        put<uint32_t>(b, kI32);
        put<uint64_t>(b, v.size());
        for (int x : v) put<int32_t>(b, x);
        set(k, b);
    }
    void arr_str(const std::string& k, int n) {
        std::vector<uint8_t> b;
        put<uint32_t>(b, kArr);
        put<uint32_t>(b, kStr);
        put<uint64_t>(b, (uint64_t) n);
        for (int i = 0; i < n; ++i) put_str(b, "tok" + std::to_string(i));
        set(k, b);
    }
    static uint64_t nbytes(const Tensor& t) {
        int el = 1, by = 4;
        if (!strata::block_geometry(t.type, el, by)) return 0;
        uint64_t n = 1;
        for (auto d : t.shape) n *= d;
        return n / (uint64_t) el * (uint64_t) by;
    }
    void tensor(const std::string& name, std::vector<uint64_t> shape, uint32_t type) {
        tensors.push_back({name, std::move(shape), type, 0});
    }
    // writes the header and directory, then pads to the data start (no tensor bytes: nothing here reads them);
    // returns the data start
    uint64_t write(const fs::path& path) {
        uint64_t at = 0;
        for (auto& t : tensors) {
            t.offset = at;
            at += (nbytes(t) + 31) / 32 * 32;
        }
        std::vector<uint8_t> b;
        put<uint32_t>(b, 0x46554747u);
        put<uint32_t>(b, 3);
        put<uint64_t>(b, tensors.size());
        put<uint64_t>(b, kvs.size());
        for (const auto& kv : kvs) {
            put_str(b, kv.key);
            b.insert(b.end(), kv.bytes.begin(), kv.bytes.end());
        }
        for (const auto& t : tensors) {
            put_str(b, t.name);
            put<uint32_t>(b, (uint32_t) t.shape.size());
            for (auto d : t.shape) put<uint64_t>(b, d);
            put<uint32_t>(b, t.type);
            put<uint64_t>(b, t.offset);
        }
        const uint64_t data_start = (b.size() + 31) / 32 * 32;
        b.resize(tensors.empty() ? b.size() : data_start + 32, 0);
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
        return data_start;
    }
};

// ---------------------------------------------------------------- the GLM-5.3 744B header

constexpr int kLayers = 78, kE = 6144, kH = 64, kQL = 2048, kKL = 512, kRot = 64, kNope = 192, kV = 256;
constexpr int kNX = 256, kFX = 2048, kFD = 12288, kIH = 32, kIK = 128, kVocab = 154880, kLead = 3;
constexpr uint32_t F32 = 0, Q8_0 = 8, Q4_K = 12, Q6_K = 14, IQ2_XXS = 16, IQ3_XXS = 18;

std::vector<int> glm53_types(int n) {
    std::vector<int> v((size_t) n, 0);
    for (int il = 0; il < n; ++il) v[(size_t) il] = il < 2 || (il - 2) % 4 == 0 ? 1 : 0;
    if (n > kLayers) v[(size_t) kLayers] = 1;
    return v;
}

void glm53_meta(Writer& w) {
    const std::string p = "glm-dsa.";
    w.str("general.architecture", "glm-dsa");
    w.u32(p + "block_count", 79);
    w.u32(p + "nextn_predict_layers", 1);
    w.u32(p + "context_length", 1048576);
    w.u32(p + "embedding_length", kE);
    w.u32(p + "feed_forward_length", kFD);
    w.u32(p + "vocab_size", kVocab);
    w.u32(p + "attention.head_count", kH);
    w.u32(p + "attention.head_count_kv", 1);
    w.f32(p + "attention.layer_norm_rms_epsilon", 1e-5f);
    w.u32(p + "attention.q_lora_rank", kQL);
    w.u32(p + "attention.kv_lora_rank", kKL);
    w.u32(p + "attention.key_length", kKL + kRot);
    w.u32(p + "attention.value_length", kKL);
    w.u32(p + "attention.key_length_mla", kNope + kRot);
    w.u32(p + "attention.value_length_mla", kV);
    w.f32(p + "rope.freq_base", 8000000.0f);
    w.u32(p + "rope.dimension_count", kRot);
    w.u32(p + "expert_count", kNX);
    w.u32(p + "expert_used_count", 8);
    w.u32(p + "expert_group_count", 1);
    w.u32(p + "expert_group_used_count", 1);
    w.u32(p + "expert_gating_func", 2);
    w.u32(p + "leading_dense_block_count", kLead);
    w.u32(p + "expert_feed_forward_length", kFX);
    w.u32(p + "expert_shared_count", 1);
    w.f32(p + "expert_weights_scale", 2.5f);
    w.boolean(p + "expert_weights_norm", true);
    w.u32(p + "attention.indexer.head_count", kIH);
    w.u32(p + "attention.indexer.key_length", kIK);
    w.u32(p + "attention.indexer.top_k", 2048);
    w.arr_bool(p + "attention.indexer.types", glm53_types(kLayers));
    w.str("tokenizer.ggml.model", "gpt2");
    w.arr_str("tokenizer.ggml.tokens", 100);   // (the real one: 154,856 ids; the reader samples 64)
}

const std::vector<std::string> kRequired = {
    "block_count", "embedding_length", "feed_forward_length", "context_length", "attention.head_count",
    "attention.head_count_kv", "attention.q_lora_rank", "attention.kv_lora_rank", "attention.key_length",
    "attention.value_length", "attention.key_length_mla", "attention.value_length_mla",
    "attention.layer_norm_rms_epsilon", "rope.dimension_count", "rope.freq_base", "expert_count", "expert_used_count",
    "expert_shared_count", "expert_feed_forward_length", "leading_dense_block_count", "expert_weights_scale",
    "expert_weights_norm", "expert_gating_func", "attention.indexer.head_count", "attention.indexer.key_length",
    "attention.indexer.top_k"};

// the tensors of one layer (il == 78: the NextN block)
struct LayerOpts {
    bool indexer_everywhere = false;   // early GLM-5.2 converters: indexer tensors on every layer
    bool shared_head_norm = true;
    std::string bad_tensor;            // this tensor gets a wrong dim
    std::string drop_tensor;           // this tensor is left out
    bool full_index = false;           // index.txt lists every dense tensor (the split search's dense costs)
};

void layer_tensors(Writer& w, int il, const LayerOpts& o) {
    const std::string P = "blk." + std::to_string(il) + ".";
    const auto add = [&](const std::string& n, std::vector<uint64_t> s, uint32_t t) {
        if (P + n == o.drop_tensor) return;
        if (P + n == o.bad_tensor) s.back() -= 384;
        w.tensor(P + n, std::move(s), t);
    };
    add("attn_norm.weight", {kE}, F32);
    add("attn_q_a_norm.weight", {kQL}, F32);
    add("attn_kv_a_norm.weight", {kKL}, F32);
    add("attn_q_a.weight", {kE, kQL}, Q4_K);
    add("attn_q_b.weight", {kQL, (uint64_t) kH * (kNope + kRot)}, Q4_K);
    add("attn_kv_a_mqa.weight", {kE, kKL + kRot}, Q6_K);
    add("attn_k_b.weight", {kNope, kKL, kH}, Q8_0);
    add("attn_v_b.weight", {kKL, kV, kH}, Q8_0);
    add("attn_output.weight", {(uint64_t) kH * kV, kE}, Q4_K);
    add("ffn_norm.weight", {kE}, F32);
    const bool full = il < 2 || (il - 2) % 4 == 0 || il == kLayers;
    if (full || o.indexer_everywhere) {
        add("indexer.k_norm.weight", {kIK}, F32);
        add("indexer.k_norm.bias", {kIK}, F32);
        add("indexer.proj.weight", {kE, kIH}, F32);
        add("indexer.attn_k.weight", {kE, kIK}, Q8_0);
        add("indexer.attn_q_b.weight", {kQL, (uint64_t) kIH * kIK}, Q8_0);
    }
    if (il < kLead) {
        add("ffn_gate.weight", {kE, kFD}, Q4_K);
        add("ffn_up.weight", {kE, kFD}, Q4_K);
        add("ffn_down.weight", {kFD, kE}, Q6_K);
    } else {
        add("ffn_gate_inp.weight", {kE, kNX}, F32);
        add("exp_probs_b.bias", {kNX}, F32);
        add("ffn_gate_exps.weight", {kE, kFX, kNX}, IQ2_XXS);
        add("ffn_up_exps.weight", {kE, kFX, kNX}, IQ2_XXS);
        add("ffn_down_exps.weight", {kFX, kE, kNX}, IQ3_XXS);
        add("ffn_gate_shexp.weight", {kE, kFX}, Q4_K);
        add("ffn_up_shexp.weight", {kE, kFX}, Q4_K);
        add("ffn_down_shexp.weight", {kFX, kE}, Q6_K);
    }
    if (il == kLayers) {
        add("nextn.eh_proj.weight", {2 * kE, kE}, Q4_K);
        add("nextn.enorm.weight", {kE}, F32);
        add("nextn.hnorm.weight", {kE}, F32);
        if (o.shared_head_norm) add("nextn.shared_head_norm.weight", {kE}, F32);
    }
}

// a 3-shard GLM-5.3 GGUF beside <dir>/pack: shard 1 metadata only, layers 0-38 in shard 2, 39-78 in shard 3, and the
// pack's native_experts.txt (+ a small index.txt)
fs::path write_pack(const fs::path& dir_in, const LayerOpts& o) {
    fs::path dir = dir_in;
#if defined(_WIN32)
    // Windows: a refused load keeps the shard views it mapped (Maya's Glm5Model unmaps only in pack_release_views
    // after a finished load) and a mapped file cannot be deleted, so every pack after the first gets its own folder
    static int packs = 0;
    if (packs++ > 0) dir = fs::path(dir_in.string() + "." + std::to_string(packs));
#endif
    fs::remove_all(dir);
    fs::create_directories(dir / "pack");
    const std::string stem = "GLM-5.3-TEST";
    const auto shard = [&](int i) { return stem + "-0000" + std::to_string(i) + "-of-00003.gguf"; };
    Writer s1, s2, s3;
    glm53_meta(s1);
    for (Writer* w : {&s2, &s3}) {
        w->str("general.architecture", "glm-dsa");
        w->u32("split.count", 3);
    }
    s2.tensor("token_embd.weight", {kE, kVocab}, Q4_K);
    for (int il = 0; il < 39; ++il) layer_tensors(s2, il, o);
    for (int il = 39; il <= kLayers; ++il) layer_tensors(s3, il, o);
    s3.tensor("output_norm.weight", {kE}, F32);
    s3.tensor("output.weight", {kE, kVocab}, Q6_K);
    s1.write(dir / shard(1));
    const uint64_t ds2 = s2.write(dir / shard(2)), ds3 = s3.write(dir / shard(3));

    std::ofstream ne(dir / "pack" / "native_experts.txt");
    ne << "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
          "[shard | gate_shard up_shard down_shard] (n_expert 256, total 0; absolute offsets in "
       << shard(1) << ", or in the named shard(s) beside it)\n";
    uint64_t off = 0;
    for (int il = kLead; il < kLayers; ++il) {
        Writer& w = il < 39 ? s2 : s3;
        const uint64_t ds = il < 39 ? ds2 : ds3;
        const std::string P = "blk." + std::to_string(il) + ".";
        uint64_t go = 0, uo = 0, dof = 0, blob = 0;
        for (const auto& t : w.tensors) {
            if (t.name == P + "ffn_gate_exps.weight") go = ds + t.offset, blob += Writer::nbytes(t) / kNX;
            if (t.name == P + "ffn_up_exps.weight") uo = ds + t.offset, blob += Writer::nbytes(t) / kNX;
            if (t.name == P + "ffn_down_exps.weight") dof = ds + t.offset, blob += Writer::nbytes(t) / kNX;
        }
        ne << il << " " << IQ2_XXS << " " << IQ3_XXS << " " << off << " " << blob << " " << go << " " << uo << " "
           << dof << " " << shard(il < 39 ? 2 : 3) << "\n";
        off += blob * kNX;
    }
    std::ofstream ix(dir / "pack" / "index.txt");
    ix << "# strata pack index v3 -- synthetic\n";
    if (!o.full_index) {
        ix << "blk.0.attn_q_a.weight 0 0 0 0 0 0 6144 2048 8 0 32 0 0 0 0 0 0 0\n";
        ix << "blk.5.ffn_gate_exps.weight 0 0 0 0 0 0\n";
        return dir / "pack";
    }
    // every dense tensor of the trunk as tools/iq_pack.py lists it: quantized ones served natively (kind 0), the F32
    // ones from dense.bin (kind 2); the experts and the NextN block stay out of the index
    for (const Writer* w : {&s2, &s3})
        for (const auto& t : w->tensors) {
            if (t.name.find("_exps.") != std::string::npos || t.name.rfind("blk." + std::to_string(kLayers) + ".", 0) == 0)
                continue;
            ix << t.name << " 0 " << (t.type == F32 ? "2" : "0") << " 0 0 0 0 " << t.shape[0] << " "
               << (t.shape.size() > 1 ? t.shape[1] : 1) << " 8 0 32 0 0 0 0 0 0 0\n";
        }
    return dir / "pack";
}

std::string meta_file(const fs::path& dir, const std::function<void(Writer&)>& mutate, const char* name = "meta") {
    Writer w;
    glm53_meta(w);
    mutate(w);
    const fs::path p = dir / (std::string(name) + ".gguf");
    w.write(p);
    return p.string();
}

bool parse(const std::string& path, Glm5Geometry& g, std::string& err) {
    try {
        strata::GgufFile f(path);
        return Glm5Geometry::from_gguf(f, g, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool has(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

}  // namespace

int main() {
#if defined(_WIN32)
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // (MSVC's CRT has no line buffering, and _IOLBF with size 0 aborts)
#else
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // in step with the loader's stderr notes
#endif
    std::printf("glm_dsa_loader_test\n");
    const fs::path dir = fs::temp_directory_path() / ("strata_glm_dsa_loader_" + std::to_string((long) getpid()));
    fs::create_directories(dir);
    unsetenv("STRATA_GLM_IDX_LN_EPS");
    unsetenv("STRATA_GLM_SPLIT");
    setenv("STRATA_GLM_CHECK_ONLY", "1", 1);   // glm-dsa loads stop after the shape checks (no GPU work)

    // ---- 1. the reader keeps numeric/bool arrays whole (B1.1)
    std::printf("[B1.1 gguf arrays]\n");
    {
        const std::string path = meta_file(dir, [](Writer& w) {
            w.arr_bool("test.bool79", glm53_types(79));
            std::vector<int> big(65536);
            for (size_t i = 0; i < big.size(); ++i) big[i] = (int) i * 3 - 7;
            w.arr_i32("test.i32_65536", big);
            w.arr_i32("test.i32_154880", std::vector<int>(154880, 1));
        });
        strata::GgufFile f(path);
        // Maya v1.1.55 keeps a 64-item sample in metadata() and reads a whole array through get_full (the glm-dsa
        // geometry reader does that for every truncated array); the round trips go through it
        strata::MetaValue b79, t78, i65;
        const strata::MetaValue* b = f.get_full("test.bool79", b79) ? &b79 : nullptr;
        bool same = b && b->count == 79 && b->items.size() == 79;
        const auto want = glm53_types(79);
        for (size_t i = 0; same && i < 79; ++i) same = b->items[i].u == (uint64_t) want[i];
        check(same, "a 79-entry bool array round-trips (count 79, 79 items, every value)");
        const strata::MetaValue* it = f.get_full("glm-dsa.attention.indexer.types", t78) ? &t78 : nullptr;
        check(it && it->items.size() == 78, "indexer.types keeps its 78 entries (was cut to 64)");
        const strata::MetaValue* big = f.get_full("test.i32_65536", i65) ? &i65 : nullptr;
        check(big && big->items.size() == 65536 && (int64_t) big->items[65535].u == 65535LL * 3 - 7,
              "a 65536-entry i32 array keeps every item, signed values intact");
        const strata::MetaValue* voc = f.get("test.i32_154880");
        check(voc && voc->count == 154880 && voc->items.size() == 64,
              "a vocabulary-sized numeric array (154880) keeps the 64-item sample");
        const strata::MetaValue* toks = f.get("tokenizer.ggml.tokens");
        check(toks && toks->count == 100 && toks->items.size() == 64 && toks->items[63].s == "tok63",
              "string arrays keep their 64-item sample (count 100)");
    }

    // ---- 2. the glm-dsa geometry (B1.2, B1.3)
    std::printf("[B1.3 glm-dsa geometry]\n");
    {
        Glm5Geometry g;
        std::string err;
        const bool ok = parse(meta_file(dir, [](Writer&) {}), g, err);
        check(ok, "the GLM-5.3 header parses" + (err.empty() ? std::string() : " (" + err + ")"));
        check(g.arch == GlmArch::GlmDsa && std::string(strata::core::glm_arch_name(g.arch)) == "glm-dsa", "arch glm-dsa");
        check(g.n_layers == 78 && g.nextn == 1, "78 trunk layers + 1 NextN");
        check(g.n_embd == 6144 && g.n_head == 64 && g.n_vocab == 154880, "n_embd 6144, 64 heads, vocab 154880");
        check(g.q_lora == 2048 && g.kv_lora == 512, "q_lora 2048, kv_lora 512");
        check(g.qk_nope == 192 && g.qk_rope == 64 && g.v_head == 256, "qk_nope 192 (= key_length_mla - rope), rope 64, v 256");
        check(g.qk_head() == 256 && g.kv_row() == 576, "qk_head() 256, kv_row() 576");
        check(g.attn_scale() == 0.0625f, "attn_scale() 1/16");
        check(g.rope_theta == 8e6, "rope theta 8e6");
        check(g.idx_heads == 32 && g.idx_key == 128 && g.idx_top_k == 2048 && g.idx_rope == 64,
              "indexer 32 x 128, top-2048, rope on the first 64 dims");
        check(g.idx_kind == strata::core::GlmIdxKind::Token && g.idx_kpool == 1 && g.idx_select_tail == 0,
              "token indexer: no k-pool, no tail");
        check(g.idx_ln_eps == 1e-6f, "indexer LayerNorm eps 1e-6 (HF)");
        check(g.n_ctx_train == 1048576, "n_ctx_train 1048576");
        check(g.norm_eps == 1e-5f, "RMSNorm eps 1e-5");
        int full = 0;
        std::string fl;
        for (int il = 0; il < g.n_layers; ++il)
            if (g.idx_full(il)) ++full, fl += std::to_string(il) + ",";
        check(full == 21 && fl == "0,1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,",
              "21 full-indexer layers 0,1,2,6,...,74");
        check(g.idx_full(78), "the NextN block blk.78 is full");
        check(!g.mhc() && g.hc == 1 && g.residual_floats() == 6144, "plain residual: mhc() false, residual 6144 floats");
        check(std::isinf(g.swiglu_exp) && std::isinf(g.swiglu_shexp), "no SwiGLU clamp (+INF)");
        check(g.n_expert == 256 && g.n_exp_used == 8 && g.n_shared == 1 && g.n_ff_exp == 2048 && g.n_ff_dense == 12288,
              "256 experts top-8 + 1 shared, ff 2048, dense ff 12288");
        check(g.dense_lead == 3 && g.w_scale == 2.5f && g.norm_w == 1, "3 dense lead layers, weights normalized x 2.5");
        bool kv = (int) g.n_head_kv.size() == 78;
        for (int il = 0; kv && il < 78; ++il) kv = !g.is_recr(il);
        check(kv, "every layer MLA (none recurrent)");
        check(g.n_moe_layers() == 75, "n_moe_layers() 75 (protect 64 x 75)");
        check(g.boundary_ok(26) && g.boundary_ok(54) && !g.boundary_ok(25) && !g.boundary_ok(0) && !g.boundary_ok(78),
              "boundary_ok: 26, 54 yes; 25 (shared), 0, 78 no");
    }
    {
        Glm5Geometry g;
        std::string err;
        bool ok = parse(meta_file(dir, [](Writer& w) {
            w.arr_bool("glm-dsa.attention.indexer.types", glm53_types(79));
        }), g, err);
        int full = 0;
        for (int il = 0; il < 79; ++il) full += g.idx_full(il);
        check(ok && full == 22 && g.idx_full(78), "a 79-entry indexer.types (with the NextN block) parses: 22 full");
        ok = parse(meta_file(dir, [](Writer& w) { w.erase("glm-dsa.attention.indexer.types"); }), g, err);
        full = 0;
        for (int il = 0; il < 78; ++il) full += g.idx_full(il);
        check(ok && full == 21 && g.idx_full(6) && !g.idx_full(7),
              "no indexer.types, context 1048576: llama.cpp's GLM-5.2 schedule (21 full)");
        ok = parse(meta_file(dir, [](Writer& w) {
            w.erase("glm-dsa.attention.indexer.types");
            w.u32("glm-dsa.context_length", 202752);
        }), g, err);
        full = 0;
        for (int il = 0; il < 78; ++il) full += g.idx_full(il);
        check(ok && full == 78, "no indexer.types, context 202752 (GLM-5/5.1): every layer full");
        ok = parse(meta_file(dir, [](Writer& w) { w.erase("glm-dsa.vocab_size"); }), g, err);
        check(ok && g.n_vocab == 0, "no vocab_size: n_vocab left for token_embd (0)");
        setenv("STRATA_GLM_IDX_LN_EPS", "0", 1);
        ok = parse(meta_file(dir, [](Writer&) {}), g, err);
        check(ok && g.idx_ln_eps == 0.0f, "STRATA_GLM_IDX_LN_EPS=0 (llama.cpp parity) overrides the eps");
        setenv("STRATA_GLM_IDX_LN_EPS", "abc", 1);
        ok = parse(meta_file(dir, [](Writer&) {}), g, err);
        check(!ok && has(err, "STRATA_GLM_IDX_LN_EPS"), "STRATA_GLM_IDX_LN_EPS=abc refused");
        unsetenv("STRATA_GLM_IDX_LN_EPS");
    }

    // ---- 3. refusals (B1.3)
    std::printf("[B1.3 refusals]\n");
    {
        int refused = 0;
        std::string first_bad;
        for (const auto& k : kRequired) {
            Glm5Geometry g;
            std::string err;
            const bool ok = parse(meta_file(dir, [&](Writer& w) { w.erase("glm-dsa." + k); }), g, err);
            if (!ok && has(err, "glm-dsa." + k)) ++refused;
            else if (first_bad.empty()) first_bad = k + " (" + err + ")";
        }
        check(refused == (int) kRequired.size(), "each of the " + std::to_string(kRequired.size()) +
                                                     " required keys missing -> refused, key named" +
                                                     (first_bad.empty() ? "" : ": " + first_bad));
        struct Mut {
            const char* what;
            std::function<void(Writer&)> f;
            const char* expect;
        };
        const std::vector<Mut> muts = {
            {"key_length 512 (no rope tail)", [](Writer& w) { w.u32("glm-dsa.attention.key_length", 512); }, "key_length"},
            {"value_length 576", [](Writer& w) { w.u32("glm-dsa.attention.value_length", 576); }, "value_length"},
            {"key_length_mla 64 (no nope part)", [](Writer& w) { w.u32("glm-dsa.attention.key_length_mla", 64); },
             "key_length_mla"},
            {"softmax gating (1)", [](Writer& w) { w.u32("glm-dsa.expert_gating_func", 1); }, "sigmoid"},
            {"rope scaling yarn", [](Writer& w) { w.str("glm-dsa.rope.scaling.type", "yarn"); }, "rope scaling"},
            {"rope scaling factor 4", [](Writer& w) { w.f32("glm-dsa.rope.scaling.factor", 4.0f); }, "rope scaling"},
            {"yarn_log_multiplier 0.1", [](Writer& w) { w.f32("glm-dsa.rope.scaling.yarn_log_multiplier", 0.1f); },
             "yarn"},
            {"layer 0 shared", [](Writer& w) {
                 auto t = glm53_types(78);
                 t[0] = 0;
                 w.arr_bool("glm-dsa.attention.indexer.types", t);
             }, "layer 0"},
            {"indexer.types of 64 entries", [](Writer& w) {
                 w.arr_bool("glm-dsa.attention.indexer.types", glm53_types(64));
             }, "64 entries"},
            {"2 NextN blocks", [](Writer& w) { w.u32("glm-dsa.nextn_predict_layers", 2); }, "NextN"},
            {"head_count_kv 0 (recurrent)", [](Writer& w) { w.u32("glm-dsa.attention.head_count_kv", 0); },
             "head_count_kv"},
            {"per-layer expert ff array with two values", [](Writer& w) {
                 std::vector<int> v(79, 2048);
                 v[5] = 1024;
                 w.arr_i32("glm-dsa.expert_feed_forward_length", v);
             }, "expert_feed_forward_length"},
            {"architecture 'deepseek2'", [](Writer& w) { w.str("general.architecture", "deepseek2"); }, "deepseek2"},
        };
        for (const auto& m : muts) {
            Glm5Geometry g;
            std::string err;
            const bool ok = parse(meta_file(dir, m.f), g, err);
            check(!ok && has(err, m.expect), std::string("refused: ") + m.what + (ok ? "" : "  [" + err.substr(0, 60) + "]"));
        }
        Glm5Geometry g;
        std::string err;
        const bool ok = parse(meta_file(dir, [](Writer& w) {
            std::vector<int> v(79, 2048);
            w.arr_i32("glm-dsa.expert_feed_forward_length", v);
        }), g, err);
        check(ok && g.n_ff_exp == 2048, "a per-layer expert ff array with one value is accepted");
    }

    // ---- 4. Flash (glm5-next) unchanged (B1.2)
    std::printf("[B1.2 Flash geometry unchanged]\n");
    for (const std::string pre : {"glm5-next.", "glm5next."}) {
        Writer w;
        w.str("general.architecture", pre.substr(0, pre.size() - 1));
        w.u32(pre + "block_count", 46);
        w.u32(pre + "nextn_predict_layers", 1);
        w.u32(pre + "embedding_length", 4096);
        std::vector<int> hkv(46, 0);
        for (int il = 3; il < 46; il += 4) hkv[(size_t) il] = 64;
        w.arr_i32(pre + "attention.head_count_kv", hkv);
        w.arr_bool(pre + "attention.indexer.types", std::vector<int>(46, 1));
        w.u32(pre + "attention.key_length", 512);
        const fs::path p = dir / "flash.gguf";
        w.write(p);
        Glm5Geometry g, d;
        std::string err;
        const bool ok = parse(p.string(), g, err);
        bool same = ok && g.arch == GlmArch::Glm5Next && g.n_layers == 45 && g.nextn == 1 && g.n_embd == 4096 &&
                    g.n_head == d.n_head && g.q_lora == d.q_lora && g.kv_lora == d.kv_lora && g.qk_nope == 256 &&
                    g.v_head == 256 && g.idx_kpool == 4 && g.idx_select_tail == 1 && g.hc == 4 && g.n_expert == 288 &&
                    g.swiglu_exp == 10.0f && g.swiglu_shexp == 10.0f && g.w_scale == 1.0f && g.n_vocab == 154880 &&
                    g.qk_rope == 0 && g.rope_theta == 0 && g.idx_kind == strata::core::GlmIdxKind::KPool;
        for (int il = 0; same && il < 45; ++il) same = g.is_recr(il) == (hkv[(size_t) il] == 0) && g.idx_full(il);
        check(same, pre + " header: Maya's values (45 + 1 layers, kpool 4, hc 4, 288 experts, clamp 10)" +
                        (err.empty() ? "" : " [" + err + "]"));
        check(g.qk_head() == 256 && g.kv_row() == 512 && g.attn_scale() == 0.0625f && g.mhc() &&
                  g.residual_floats() == 4 * 4096 && g.n_moe_layers() == 42,
              pre + " helpers: qk_head 256, kv_row 512, mhc, residual 4 x 4096, n_moe_layers 42 (protect 64 x 42)");
        bool all = true;
        for (int il = 1; il < 45; ++il) all = all && g.boundary_ok(il);
        check(all && !g.boundary_ok(45), pre + " any trunk layer 1..44 may start a part");
        Glm5Geometry g2 = g;
        check(strata::core::glm_geometry_check(g2, [](const std::string&) { return nullptr; }, 0, 0, err),
              pre + " shape check is a no-op for Flash");
        std::string e2;
        check(strata::core::glm_forward_available(g, e2) && strata::core::glm_split_bounds_ok(g, {7}, e2),
              pre + " forward gate open, any split bound passes");
    }
    {
        Writer w;
        w.u32("glm5-next.block_count", 46);
        const fs::path p = dir / "noarch.gguf";
        w.write(p);
        Glm5Geometry g;
        std::string err;
        check(parse(p.string(), g, err) && g.arch == GlmArch::Glm5Next && g.n_layers == 46,
              "no general.architecture: a warning, then Maya's legacy key probe");
    }

    // ---- 5. a 3-shard pack, metadata-only shard 1, through load_pack (B1.4, B1.10)
    std::printf("[B1.4 / B1.10 pack load (metadata-only shard 1)]\n");
    const fs::path pdir = dir / "model";
    {
        const fs::path pack = write_pack(pdir, LayerOpts{});
        std::vector<std::unique_ptr<strata::GgufFile>> gfs;
        for (int i = 1; i <= 3; ++i)
            gfs.emplace_back(new strata::GgufFile((pdir / ("GLM-5.3-TEST-0000" + std::to_string(i) + "-of-00003.gguf")).string()));
        check(gfs[0]->tensors().empty() && gfs[0]->get("glm-dsa.block_count"), "shard 1 holds metadata only");
        const auto find = [&](const std::string& n) -> const strata::TensorInfo* {
            for (const auto& gf : gfs)
                if (const strata::TensorInfo* t = gf->find(n)) return t;
            return nullptr;
        };
        Glm5Geometry g;
        std::string err;
        bool ok = Glm5Geometry::from_gguf(*gfs[0], g, err);
        g.n_vocab = 0;   // as without vocab_size: the check takes token_embd's rows
        ok = ok && strata::core::glm_geometry_check(g, find, 0, 0, err);
        check(ok && g.n_vocab == 154880, "full 744B directory passes the shape check; n_vocab from token_embd" +
                                             (err.empty() ? "" : " [" + err + "]"));
        std::vector<double> cost(78, 0.2e9);
        strata::core::glm_dense_layer_cost(g, pack.string(), cost);
        check(cost[0] == 6144.0 * 2048 / 256 * 144 && cost[1] == 0 && cost[5] == 0,
              "dense split cost from index.txt + the shards (experts excluded)");

        strata::core::Glm5Model m;
        err.clear();
        ok = m.load_pack(pack.string(), 4096, err);
        check(!ok && has(err, "check-only load"),
              "load_pack: shards open, experts rows verified, shapes pass, refused at the forward gate");
        std::printf("    (%s)\n", err.c_str());
        check(m.geometry().arch == GlmArch::GlmDsa && m.geometry().n_layers == 78, "load_pack read the glm-dsa geometry");
    }
    const auto pack_err = [&](const LayerOpts& o) {
        const fs::path pack = write_pack(pdir, o);
        strata::core::Glm5Model m;
        std::string err;
        const bool ok = m.load_pack(pack.string(), 4096, err);
        return ok ? std::string("LOADED") : err;
    };
    {
        LayerOpts o;
        o.bad_tensor = "blk.5.attn_q_b.weight";
        std::string e = pack_err(o);
        check(has(e, "blk.5.attn_q_b.weight") && has(e, "16000") && has(e, "16384"),
              "a wrong dim fails naming the tensor and both shapes  [" + e.substr(0, 70) + "]");
        o = LayerOpts{};
        o.bad_tensor = "blk.40.attn_k_b.weight";
        e = pack_err(o);
        check(has(e, "blk.40.attn_k_b.weight"), "a wrong attn_k_b head count (3-D) fails, named");
        o = LayerOpts{};
        o.drop_tensor = "blk.6.indexer.attn_k.weight";
        e = pack_err(o);
        check(has(e, "blk.6.indexer.attn_k.weight") && has(e, "missing"), "a full layer without its indexer fails, named");
        o = LayerOpts{};
        o.drop_tensor = "blk.7.indexer.attn_k.weight";
        e = pack_err(o);
        check(has(e, "check-only load"), "a shared layer needs no indexer tensors");
        o = LayerOpts{};
        o.indexer_everywhere = true;
        e = pack_err(o);
        check(has(e, "check-only load"), "indexer tensors on every layer: a warning, then the forward gate");
        o = LayerOpts{};
        o.shared_head_norm = false;
        e = pack_err(o);
        check(has(e, "check-only load"), "a NextN block without shared_head_norm passes (falls back)");
        o = LayerOpts{};
        o.drop_tensor = "blk.78.nextn.hnorm.weight";
        e = pack_err(o);
        check(has(e, "blk.78.nextn.hnorm.weight"), "a NextN block missing hnorm fails, named");
    }

    // ---- 6. load_pack_env: the pack's geometry before the split bounds (B1.5)
    std::printf("[B1.5 split bounds from the pack's geometry]\n");
    {
        const fs::path pack = write_pack(pdir, LayerOpts{});
        std::string err;
        {
            setenv("STRATA_GLM_SPLIT", "26,54", 1);
            strata::core::Glm5Model m;
            m.load_pack_env(pack.string(), 4096, err);
            // (without GPUs Maya's glm_split_devices refuses the 3 parts after the bounds passed)
            check(!has(err, "1..44") && (has(err, "1..77") || has(err, "check-only load") || has(err, "devices")),
                  "STRATA_GLM_SPLIT=26,54 is in range of the 78-layer pack (not Flash's 45)  [" + err.substr(0, 60) + "]");
        }
        {
            setenv("STRATA_GLM_SPLIT", "25,54", 1);
            strata::core::Glm5Model m;
            err.clear();
            m.load_pack_env(pack.string(), 4096, err);
            check(has(err, "layer 25") && has(err, "full-indexer"), "25 is a shared-indexer layer: refused as a boundary");
        }
        {
            setenv("STRATA_GLM_SPLIT", "0", 1);
            strata::core::Glm5Model m;
            err.clear();
            m.load_pack_env(pack.string(), 4096, err);
            check(has(err, "check-only load"), "STRATA_GLM_SPLIT=0: one part, refused at the forward gate");
        }
        unsetenv("STRATA_GLM_SPLIT");
    }

    // ---- 6b. --layer-split auto on the 744B pack with simulated cards (B6.1 / B6.2; no GPU)
    std::printf("[B6.1 / B6.2 split search, simulated cards]\n");
    {
        LayerOpts full;
        full.full_index = true;
        const fs::path pack = write_pack(pdir, full);
        Glm5Geometry g;
        std::string err;
        {
            strata::GgufFile f((pdir / "GLM-5.3-TEST-00001-of-00003.gguf").string());
            Glm5Geometry::from_gguf(f, g, err);
        }
        const auto txt = [](const std::vector<int>& b) {
            std::string s;
            for (int k : b) s += (s.empty() ? "" : ",") + std::to_string(k);
            return s.empty() ? std::string("(none)") : s;
        };
        const auto search = [&](const char* cards, std::vector<int> devs) {
            setenv("STRATA_GLM_SPLIT_CARDS", cards, 1);
            const std::vector<int> b = strata::core::glm_split_search(pack.string(), 32768, devs);
            unsetenv("STRATA_GLM_SPLIT_CARDS");
            return b;
        };
        // the test PC: 3x V100-SXM2 32 GB (~31.5 GiB free, 898 GB/s); 217 GB of IQ2_XXS/IQ3_XXS experts, ~40% in VRAM
        const std::vector<int> v3 = search("31.5:898:V100,31.5:898:V100,31.5:898:V100", {0, 1, 2});
        // (BUILD-PLAN's {26, 54} and {30, 54} price within 0.2% of each other here - 27.79 / 27.74 ms with the default
        // coverage curve: 30 gives the three parts 27 / 24 / 24 MoE layers, 26 gives 23 / 28 / 24; the measured host
        // read speed picks between them)
        check(v3.size() == 2 && strata::core::glm_split_bounds_full(g, v3) && (v3[0] == 26 || v3[0] == 30) && v3[1] == 54,
              "3x V100 32 GB: full-indexer parts {26 or 30, 54}  [" + txt(v3) + "]");
        const std::vector<int> v2 = search("31.5:898,31.5:898", {0, 1});
        check(v2.size() == 1 && strata::core::glm_split_bounds_full(g, v2),
              "2 cards: one full-indexer boundary  [" + txt(v2) + "]");
        const std::vector<int> v4 = search("23:1008:4090,23:1008:4090,31.5:898,15:448", {0, 1, 2, 3});
        check(v4.size() == 3 && strata::core::glm_split_bounds_full(g, v4),
              "4 mixed cards: three full-indexer boundaries  [" + txt(v4) + "]");
        check(search("31.5:898", {0, 1}).empty(), "fewer simulated cards than devices: no plan (free-VRAM split)");
    }

    // ---- 7. NextN lists, gates (B1.8, B2.6)
    std::printf("[B1.8 / B2.6 lists and gates]\n");
    {
        std::vector<std::string> raw = {"flash"}, b16 = {"flash"}, f32 = {"flash"};
        strata::core::glm_mtp_lists(GlmArch::Glm5Next, raw, b16, f32);
        check(raw.size() == 1 && raw[0] == "flash" && b16[0] == "flash" && f32[0] == "flash",
              "Flash keeps Maya's NextN lists");
        strata::core::glm_mtp_lists(GlmArch::GlmDsa, raw, b16, f32);
        bool comp = false, gi_b16 = false, gi_f32 = false;
        for (const auto* v : {&raw, &b16, &f32})
            for (const auto& n : *v) comp = comp || has(n, "compressor");
        for (const auto& n : b16) gi_b16 = gi_b16 || n == "ffn_gate_inp.weight";
        for (const auto& n : f32) gi_f32 = gi_f32 || n == "ffn_gate_inp.weight";
        check(!comp && !gi_b16 && gi_f32, "glm-dsa NextN lists: no indexer_compressor_*, router kept F32");
        check(std::string(strata::core::glm_mtp_fallback(GlmArch::GlmDsa, "nextn.shared_head_norm.weight")) ==
                      "output_norm.weight" &&
                  strata::core::glm_mtp_fallback(GlmArch::Glm5Next, "nextn.shared_head_norm.weight") == nullptr,
              "glm-dsa shared_head_norm falls back to output_norm (Flash: none)");
        // the batched prompt's weight types (prefill_setup deq_ok): Flash keeps Maya v1.1.3's set (BF16 and the
        // dequant_bf16 table, no i-quant); glm-dsa adds the i-quants dequant_f16 routes (IQ2_XXS .. IQ1_M)
        bool flash_same = true;
        for (int t = 0; t < 64; ++t)
            flash_same = flash_same && strata::core::glmpf::f16_covers(t, false) ==
                                           (t == 30 || strata::kernels::dequant_bf16_supported(t));
        bool dsa_iq = true;
        for (int t : {16, 17, 18, 21, 22, 29})
            dsa_iq = dsa_iq && strata::core::glmpf::f16_covers(t, true) && !strata::core::glmpf::f16_covers(t, false);
        check(flash_same && dsa_iq, "prompt FP16 types: Flash = Maya's deq_ok set, glm-dsa adds the i-quants");
    }

    std::error_code ec;
    fs::remove_all(dir, ec);
    std::printf(g_fail ? "glm_dsa_loader_test: %d FAILED\n" : "glm_dsa_loader_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
