// tests/k31_pack_test.cpp - the text pack (k31.txt + tensors.txt) is accepted only when its contract holds
// (src/k31/k31_pack.hpp), and a pack tools/k31/pack_gguf.py wrote agrees with the GGUF it describes.
//
// Host only:
//   c++ -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -Iinclude tests/k31_pack_test.cpp -o k31_pack_test
//
// No arguments: a small pack (one dense layer, two MoE layers) is accepted, and each refusal has its own message.
//   k31_pack_test PACK/k31.txt [--header-only]
// reads a pack the Python tool wrote (k31::read_pack: tensors.txt and the model file too), then opens its model file
// and checks the pack against it: every tensor once, at its absolute offset, bytes, type and dims (outermost first),
// and the geometry, the expert ids and their source count equal to what k31::read takes from the same file (the
// engine's own reading; its prune sidecar is looked up beside the model path as the pack names it).  Prints one
// summary line.  --header-only allows a pack of a header-only file.
#include "../src/k31/k31_pack.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

int failures = 0;
int cases = 0;

std::string read_both(const std::string& k31, const std::string& tensors, const std::vector<uint64_t>* sizes,
                      const k31::PackOptions& opt = {}) {
    k31::PackResult r = k31::read_pack_text(k31, opt);
    if (!r.ok) return r.err;
    std::string err = k31::read_tensors_text(tensors, r.pack);
    if (!err.empty()) return err;
    if (sizes) return k31::check_extent(r.pack, *sizes);
    return "";
}

std::string good_tensors();
const std::vector<uint64_t> kSizes = {1u << 24};

void expect_ok(const char* name, const std::string& k31, const std::string& tensors = good_tensors(),
               const k31::PackOptions& opt = {}) {
    ++cases;
    const std::string err = read_both(k31, tensors, &kSizes, opt);
    if (!err.empty()) {
        std::fprintf(stderr, "FAIL %s: refused a good pack: %s\n", name, err.c_str());
        ++failures;
    }
}

void expect_no(const char* name, const std::string& k31, const std::string& needle,
               const std::string& tensors = good_tensors(), const std::vector<uint64_t>& sizes = kSizes,
               const k31::PackOptions& opt = {}) {
    ++cases;
    const std::string err = read_both(k31, tensors, &sizes, opt);
    if (err.empty() || err.find(needle) == std::string::npos || err.rfind("maya-k31: pack: ", 0) != 0) {
        std::fprintf(stderr, "FAIL %s: wanted \"%s\" in \"%s\"\n", name, needle.c_str(), err.c_str());
        ++failures;
    }
}

std::string ids(int n, int first = 0) {
    std::string s;
    for (int i = 0; i < n; ++i) s += " " + std::to_string(first + i);
    return s;
}

// three layers: 0 dense (KDA), 1 MoE (MLA), 2 MoE (KDA); 24 experts of 896, IQ1_S at edim 256 / eff 512
std::string good() {
    return "format gguf\n"
           "arch kimi-k31\n"
           "data whole\n"
           "n_layers 3\n"
           "n_embd 256\n"
           "vocab 128\n"
           "n_head 6\n"
           "context 256\n"
           "rms_eps 1e-05\n"
           "lora_eps 1e-06\n"
           "kda_l2_eps 1e-06\n"
           "idx_ln_eps 1e-05\n"
           "router_renorm 1e-20\n"
           "attn_res_block 4\n"
           "kda_heads 6\n"
           "kda_head_dim 32\n"
           "d_conv 4\n"
           "gate_lower_bound -5\n"
           "mla_heads 6\n"
           "kv_lora 32\n"
           "qk_nope 16\n"
           "qk_rope 16\n"
           "v_head 16\n"
           "q_lora 32\n"
           "idx_heads 4\n"
           "idx_dim 16\n"
           "idx_topk 8\n"
           "idx_rope_dim 0\n"
           "idx_topk_freq 1\n"
           "layer 0 attn kda ffn dense n_exp 0 k 0 edim 0 eff 448 sff 0 norm_div 256\n"
           "layer 1 attn mla ffn moe n_exp 24 k 10 edim 256 eff 512 sff 128 norm_div 256\n"
           "layer 2 attn kda ffn moe n_exp 24 k 10 edim 256 eff 512 sff 128 norm_div 256\n"
           "class 1 0 IQ1_S 24 blk.1.ffn_gate_exps.weight blk.1.ffn_up_exps.weight blk.1.ffn_down_exps.weight\n"
           "class 2 0 IQ1_S 24 blk.2.ffn_gate_exps.weight blk.2.ffn_up_exps.weight blk.2.ffn_down_exps.weight\n"
           "keep 1" + ids(24) + "\n"
           "keep 2" + ids(24, 100) + "\n"
           "prune source_count 896 method gguf_prune_experts.py v1\n"
           "model_file ../tiny.gguf\n";
}

// IQ1_S: 256 values in 50 bytes.  gate / up [24, 512, 256] = 24*512 blocks, down [24, 256, 512] the same
std::string good_tensors() {
    const uint64_t slab = 24ull * 512 * 50;
    std::string s = "file 0 ../tiny.gguf\n"
                    "t token_embd.weight F32 0 4096 131072 128 256\n";
    uint64_t at = 4096 + 131072;
    for (int il : {1, 2})
        for (const char* role : {"gate", "up", "down"}) {
            s += "t blk." + std::to_string(il) + ".ffn_" + role + "_exps.weight IQ1_S 0 " + std::to_string(at) + " " +
                 std::to_string(slab) + (std::string(role) == "down" ? " 24 256 512\n" : " 24 512 256\n");
            at += slab;
        }
    return s;
}

std::string swapped(const std::string& text, const std::string& from, const std::string& to) {
    std::string out = text;
    auto at = out.find(from);
    if (at == std::string::npos) {
        std::fprintf(stderr, "test setup: missing \"%s\"\n", from.c_str());
        std::exit(2);
    }
    out.replace(at, from.size(), to);
    return out;
}

void refusals() {
    const std::string ok = good();
    const std::string tt = good_tensors();
    expect_ok("fixture pack", ok);
    k31::PackOptions allow;
    allow.allow_header_only = true;
    expect_ok("header-only, allowed", swapped(ok, "data whole", "data header-only"), tt, allow);
    {
        // a header-only pack's data is not checked against the file's size
        ++cases;
        const std::vector<uint64_t> tiny = {100};
        const std::string err = read_both(swapped(ok, "data whole", "data header-only"), tt, &tiny, allow);
        if (!err.empty()) {
            std::fprintf(stderr, "FAIL header-only extent: %s\n", err.c_str());
            ++failures;
        }
    }

    // ---- the one-a-pack records
    expect_no("other architecture", swapped(ok, "arch kimi-k31", "arch kimi-k3"), "architecture kimi-k3, kimi-k31 only");
    expect_no("no architecture", swapped(ok, "arch kimi-k31\n", ""), "arch is missing");
    expect_no("format", swapped(ok, "format gguf", "format safetensors"), "format safetensors, gguf only");
    expect_no("no format", swapped(ok, "format gguf\n", ""), "format is missing");
    expect_no("header-only", swapped(ok, "data whole", "data header-only"), "the pack is header-only");
    expect_no("data", swapped(ok, "data whole", "data some"), "data some (whole or header-only)");
    expect_no("no data", swapped(ok, "data whole\n", ""), "data is missing");
    expect_no("two arch", swapped(ok, "arch kimi-k31\n", "arch kimi-k31\narch kimi-k31\n"), "a second arch record");
    expect_no("model_file value", swapped(ok, "model_file ../tiny.gguf", "model_file a b"), "model_file takes one value");
    expect_no("no model", swapped(ok, "model_file ../tiny.gguf\n", ""), "model_file is missing");
    expect_no("unknown record", ok + "alog_tail sliced\n", "unknown record alog_tail");

    // ---- scalars
    expect_no("missing scalar", swapped(ok, "vocab 128\n", ""), "vocab is missing");
    expect_no("missing indexer eps", swapped(ok, "idx_ln_eps 1e-05\n", ""), "indexer LayerNorm eps is missing");
    expect_no("missing real", swapped(ok, "router_renorm 1e-20\n", ""), "router_renorm is missing");
    expect_no("duplicate scalar", swapped(ok, "vocab 128\n", "vocab 128\nvocab 64\n"), "a second vocab record");
    expect_no("trailing junk", swapped(ok, "attn_res_block 4", "attn_res_block 4 garbage 9"),
              "attn_res_block takes one value, not 3");
    expect_no("int range", swapped(ok, "n_layers 3", "n_layers 4294967299"), "n_layers is not an integer");
    expect_no("not an integer", swapped(ok, "n_embd 256", "n_embd 2.5e2"), "n_embd is not an integer");
    expect_no("not a number", swapped(ok, "rms_eps 1e-05", "rms_eps tiny"), "rms_eps is not a number");
    expect_no("negative dim", swapped(ok, "n_embd 256", "n_embd -7"), "n_embd -7 (it must be above 0)");
    expect_no("zero heads", swapped(ok, "n_head 6", "n_head 0"), "n_head 0 (it must be above 0)");
    expect_no("rope", swapped(ok, "idx_rope_dim 0", "idx_rope_dim 64"), "index_rope_dim 64");
    expect_no("topk freq", swapped(ok, "idx_topk_freq 1", "idx_topk_freq 2"), "index_topk_freq 2");
    expect_no("indexer eps range", swapped(ok, "idx_ln_eps 1e-05", "idx_ln_eps 2"), "indexer LayerNorm eps is not in (0, 1)");
    expect_no("eps range", swapped(ok, "lora_eps 1e-06", "lora_eps 0"), "lora_eps is not in (0, 1)");
    expect_no("gate bound", swapped(ok, "gate_lower_bound -5", "gate_lower_bound 5"), "gate_lower_bound");

    // ---- layers
    expect_no("layer count", swapped(ok, "n_layers 3", "n_layers 4"), "3 layer records for 4 layers");
    expect_no("layer order",
              swapped(ok, "layer 1 attn mla ffn moe", "layer 2 attn mla ffn moe"), "layer records are not 0..n_layers-1");
    expect_no("layer fields", swapped(ok, "norm_div 256\nlayer 1", "\nlayer 1"), "a layer record has 16 fields, not 18");
    expect_no("layer field name", swapped(ok, "layer 1 attn mla ffn moe n_exp", "layer 1 attn mla ffn moe n_exq"),
              "a layer record has n_exq where n_exp belongs");
    expect_no("layer value", swapped(ok, "layer 1 attn mla ffn moe n_exp 24 k 10", "layer 1 attn mla ffn moe n_exp 24 k x"),
              "layer 1 k is not an integer");
    expect_no("attention", swapped(ok, "layer 1 attn mla", "layer 1 attn gqa"), "layer 1 attention is gqa");
    expect_no("ffn", swapped(ok, "layer 1 attn mla ffn moe", "layer 1 attn mla ffn wide"), "layer 1 ffn is wide");
    expect_no("dense after moe",
              swapped(ok, "layer 2 attn kda ffn moe n_exp 24 k 10 edim 256 eff 512 sff 128",
                      "layer 2 attn kda ffn dense n_exp 0 k 0 edim 0 eff 512 sff 0"),
              "layer 2 is dense after a MoE layer");
    expect_no("dense with experts", swapped(ok, "ffn dense n_exp 0", "ffn dense n_exp 4"),
              "layer 0 is dense but has experts");
    expect_no("dense width", swapped(ok, "edim 0 eff 448", "edim 0 eff 0"), "layer 0 dense width 0");
    expect_no("different expert counts",
              swapped(ok, "layer 2 attn kda ffn moe n_exp 24", "layer 2 attn kda ffn moe n_exp 16"),
              "layer 2 keeps 16 experts, an earlier layer keeps 24");
    expect_no("no experts", swapped(ok, "layer 1 attn mla ffn moe n_exp 24", "layer 1 attn mla ffn moe n_exp 0"),
              "layer 1 keeps 0 experts");
    expect_no("k above n_exp", swapped(ok, "n_exp 24 k 10 edim 256 eff 512 sff 128 norm_div 256\nlayer 2",
                                       "n_exp 24 k 99 edim 256 eff 512 sff 128 norm_div 256\nlayer 2"),
              "layer 1 picks k 99 of 24 experts");
    expect_no("k above 16", swapped(ok, "n_exp 24 k 10 edim 256 eff 512 sff 128 norm_div 256\nlayer 2",
                                    "n_exp 24 k 17 edim 256 eff 512 sff 128 norm_div 256\nlayer 2"),
              "layer 1 picks k 17 of 24 experts (1 to 16");
    expect_no("k zero", swapped(ok, "n_exp 24 k 10 edim 256 eff 512 sff 128 norm_div 256\nlayer 2",
                                "n_exp 24 k 0 edim 256 eff 512 sff 128 norm_div 256\nlayer 2"),
              "layer 1 picks k 0");
    expect_no("moe width", swapped(ok, "eff 512 sff 128 norm_div 256\nlayer 2", "eff 512 sff 0 norm_div 256\nlayer 2"),
              "layer 1 has a width that is not above 0");

    // ---- expert classes
    const std::string class1 =
        "class 1 0 IQ1_S 24 blk.1.ffn_gate_exps.weight blk.1.ffn_up_exps.weight blk.1.ffn_down_exps.weight\n";
    expect_no("no class", swapped(ok, class1, ""), "layer 1 has no expert class");
    expect_no("two classes", swapped(ok, class1, class1 + class1), "layer 1 has two expert classes");
    expect_no("class slot", swapped(ok, "class 1 0 IQ1_S", "class 1 1 IQ1_S"), "layer 1 class slot 1");
    expect_no("class on dense", ok + "class 0 0 IQ1_S 24 a b c\n", "class for layer 0 is not a MoE layer");
    expect_no("class on no layer", ok + "class 7 0 IQ1_S 24 a b c\n", "class for layer 7 is not a MoE layer");
    expect_no("class count", swapped(ok, "class 1 0 IQ1_S 24", "class 1 0 IQ1_S 20"),
              "layer 1 class holds 20 experts, the layer keeps 24");
    expect_no("class fields", swapped(ok, "class 1 0 IQ1_S 24 blk.1.ffn_gate_exps.weight", "class 1 0 IQ1_S 24"),
              "a class record has 7 fields, not 8");
    expect_no("class integer", swapped(ok, "class 1 0 IQ1_S 24", "class 1 0 IQ1_S many"),
              "a class record's layer, slot or count is not an integer");
    expect_no("unknown format", swapped(ok, "class 1 0 IQ1_S", "class 1 0 BOGUS"), "expert format BOGUS not built in");
    expect_no("width vs 256", swapped(ok, "n_exp 24 k 10 edim 256 eff 512 sff 128 norm_div 256\nlayer 2",
                                      "n_exp 24 k 10 edim 256 eff 320 sff 128 norm_div 256\nlayer 2"),
              "layer 1 expert width 320 is not a multiple of 256 (IQ1_S)");
    expect_no("latent width vs 256", swapped(ok, "n_exp 24 k 10 edim 256 eff 512 sff 128 norm_div 256\nlayer 2",
                                             "n_exp 24 k 10 edim 128 eff 512 sff 128 norm_div 256\nlayer 2"),
              "layer 1 expert width 128 is not a multiple of 256 (IQ1_S)");
    expect_no("width vs 32",
              swapped(swapped(ok, "class 1 0 IQ1_S", "class 1 0 IQ4_NL"),
                      "n_exp 24 k 10 edim 256 eff 512 sff 128 norm_div 256\nlayer 2",
                      "n_exp 24 k 10 edim 256 eff 80 sff 128 norm_div 256\nlayer 2"),
              "layer 1 expert width 80 is not a multiple of 32 (IQ4_NL)");
    expect_no("class names", swapped(ok, "class 1 0 IQ1_S 24 blk.1.ffn_gate_exps.weight", "class 1 0 IQ1_S 24 blk.0.attn_q.weight"),
              "layer 1 class names blk.0.attn_q.weight");
    expect_no("class names of another layer",
              swapped(ok, "blk.1.ffn_up_exps.weight blk.1.ffn_down_exps.weight\n",
                      "blk.2.ffn_up_exps.weight blk.1.ffn_down_exps.weight\n"),
              "not the layer's own expert slabs");

    // ---- expert ids
    expect_no("no keep", swapped(ok, "keep 1" + ids(24) + "\n", ""), "layer 1 has no keep list");
    expect_no("two keeps", swapped(ok, "keep 2" + ids(24, 100) + "\n", "keep 2" + ids(24, 100) + "\nkeep 2" + ids(24) + "\n"),
              "layer 2 has two keep records");
    expect_no("keep on dense", ok + "keep 0" + ids(24) + "\n", "keep list for layer 0, which is not a MoE layer");
    expect_no("keep on no layer", ok + "keep 77 1 2\n", "keep list for layer 77, which is not a MoE layer");
    expect_no("keep length", swapped(ok, "keep 1 0 1 2 ", "keep 1 1 2 "), "layer 1 keep list has 23 ids for 24 experts");
    expect_no("keep duplicate", swapped(ok, "keep 1 0 1 2 ", "keep 1 0 0 2 "), "layer 1 keeps expert 0 twice");
    expect_no("keep negative", swapped(ok, "keep 1 0 1 2 ", "keep 1 -5 1 2 "),
              "layer 1 keep entry 0 = -5, outside the source's 896 experts");
    expect_no("keep range", swapped(ok, "keep 1 0 1 2 ", "keep 1 999 1 2 "),
              "layer 1 keep entry 0 = 999, outside the source's 896 experts");
    expect_no("keep not an integer", swapped(ok, "keep 1 0 1 2 ", "keep 1 x 1 2 "), "layer 1 keep entry 0 'x' is not an integer");
    expect_no("keep no layer", ok + "keep\n", "a keep record has no layer");
    expect_no("no prune", swapped(ok, "prune source_count 896 method gguf_prune_experts.py v1\n", ""), "prune is missing");
    expect_no("two prunes", ok + "prune source_count 896 method usage\n", "a second prune record");
    expect_no("prune form", swapped(ok, "prune source_count 896 method", "prune count 896 method"),
              "a prune record is not 'prune source_count N method M'");
    expect_no("prune no method", swapped(ok, "method gguf_prune_experts.py v1", "method"),
              "a prune record is not 'prune source_count N method M'");
    expect_no("prune integer", swapped(ok, "source_count 896", "source_count lots"), "prune source_count is not an integer");
    expect_no("source zero", swapped(ok, "source_count 896", "source_count 0"),
              "prune source_count 0 is below the kept 24 experts");
    expect_no("source below kept", swapped(ok, "source_count 896", "source_count 3"),
              "prune source_count 3 is below the kept 24 experts");

    // ---- tensors.txt
    const auto tno = [&](const char* name, const std::string& tensors, const std::string& needle,
                         const std::vector<uint64_t>& sizes = kSizes) { expect_no(name, ok, needle, tensors, sizes); };
    tno("no file record", swapped(tt, "file 0 ../tiny.gguf\n", ""), "token_embd.weight is in file 0, which no file record names");
    tno("file order", swapped(tt, "file 0 ../tiny.gguf", "file 1 ../tiny.gguf"), "file records are not 0, 1, ... in order");
    tno("file record form", swapped(tt, "file 0 ../tiny.gguf", "file 0"), "a file record is not 'file I PATH'");
    tno("file 0 is the model", swapped(tt, "file 0 ../tiny.gguf", "file 0 ../other.gguf"),
        "file 0 is ../other.gguf, k31.txt's model_file is ../tiny.gguf");
    tno("unknown tensors record", tt + "x y\n", "tensors.txt: unknown record x");
    tno("t record fields", swapped(tt, "t token_embd.weight F32 0 4096 131072 128 256", "t token_embd.weight F32 0 4096 131072"),
        "a t record has 6 fields (at least 7)");
    tno("t file index", swapped(tt, "t token_embd.weight F32 0", "t token_embd.weight F32 3"), "token_embd.weight is in file 3");
    tno("t offset", swapped(tt, "F32 0 4096 131072", "F32 0 -4096 131072"), "token_embd.weight offset or bytes is not an unsigned integer");
    tno("t type", swapped(tt, "t token_embd.weight F32", "t token_embd.weight Q9_9"), "token_embd.weight has type Q9_9");
    tno("t dim", swapped(tt, "131072 128 256", "131072 128 0"), "token_embd.weight dim '0' is not above 0");
    tno("t bytes", swapped(tt, "F32 0 4096 131072", "F32 0 4096 131122"),
        "token_embd.weight is 131122 bytes, F32 [128, 256] is 131072");
    tno("t dims not reversed", swapped(tt, "IQ1_S 0 135168 614400 24 512 256", "IQ1_S 0 135168 614400 256 512 24"),
        "blk.1.ffn_gate_exps.weight rows of 24 are not whole IQ1_S blocks");
    tno("t dims swapped", swapped(tt, "IQ1_S 0 135168 614400 24 512 256", "IQ1_S 0 135168 614400 512 24 256"),
        "blk.1.ffn_gate_exps.weight is [512, 24, 256], layer 1 says [24, 512, 256]");
    tno("t elements", swapped(tt, "131072 128 256", "131072 4294967296 4294967296 2"), "has more elements than 2^64");
    tno("t twice", tt + "t token_embd.weight F32 0 99999999 131072 128 256\n", "token_embd.weight is listed twice");
    tno("t overlap", tt + "t extra F32 0 4100 1024 256\n", "token_embd.weight and extra overlap in file 0");
    tno("t class tensor missing",
        swapped(tt, "t blk.2.ffn_up_exps.weight IQ1_S 0 2592768 614400 24 512 256\n", ""),
        "layer 2's blk.2.ffn_up_exps.weight is missing");
    tno("t class format", swapped(tt, "blk.2.ffn_down_exps.weight IQ1_S 0 3207168 614400",
                                  "blk.2.ffn_down_exps.weight IQ2_XXS 0 3207168 811008"),
        "blk.2.ffn_down_exps.weight is IQ2_XXS, its class says IQ1_S");
    tno("no tensors", "file 0 ../tiny.gguf\n", "tensors.txt: no tensors");
    tno("past the end", tt, "blk.2.ffn_down_exps.weight ends at byte 3821568, ../tiny.gguf has 3821567",
        std::vector<uint64_t>{3821567});
    tno("sizes", tt, "1 files, 2 sizes", std::vector<uint64_t>{1u << 24, 1u << 24});
}

// --- a pack against its model file
int check_file(const std::string& k31_txt, bool header_only) {
    namespace fs = std::filesystem;
    k31::PackOptions opt;
    opt.allow_header_only = header_only;
    k31::PackResult r = k31::read_pack(k31_txt, opt);
    if (!r.ok) {
        std::fprintf(stderr, "%s\n", r.err.c_str());
        return 1;
    }
    const k31::Pack& p = r.pack;
    const std::string model = (fs::path(k31_txt).parent_path() / p.model_file).string();
    int bad = 0;
    const auto fail = [&](const std::string& why) {
        if (bad++ < 20) std::fprintf(stderr, "FAIL %s\n", why.c_str());
    };
    try {
        const strata::GgufFile g(model);
        // every tensor once, where and as the directory says
        size_t seen = 0;
        for (const auto& t : g.tensors()) {
            auto it = p.tensor_index.find(t.name);
            if (it == p.tensor_index.end()) {
                fail(t.name + " is not in tensors.txt");
                continue;
            }
            ++seen;
            const k31::PackTensor& x = p.tensors[it->second];
            const std::vector<uint64_t> dims(t.shape.rbegin(), t.shape.rend());
            if (x.file != 0) fail(t.name + " is in file " + std::to_string(x.file));
            if (x.offset != g.data_start() + t.offset)
                fail(t.name + " offset " + std::to_string(x.offset) + ", the file's " + std::to_string(g.data_start() + t.offset));
            if (x.dtype != t.type_name()) fail(t.name + " type " + x.dtype + ", the file's " + t.type_name());
            if (x.dims != dims) fail(t.name + " dims " + k31::pack_detail::dims_str(x.dims) + ", the file's reversed " +
                                     k31::pack_detail::dims_str(dims));
            const k31::PackFormat* f = k31::pack_format(t.type_name());
            if (f && x.bytes != t.elements() / (uint64_t) f->block_values * (uint64_t) f->block_bytes)
                fail(t.name + " bytes " + std::to_string(x.bytes));
        }
        if (seen != p.tensors.size())
            fail("tensors.txt lists " + std::to_string(p.tensors.size()) + " tensors, the file has " +
                 std::to_string(seen) + " of them");

        // the geometry and the expert ids: what the engine's own reading takes from the same path
        k31::Geo geo;
        std::string err;
        if (!k31::read(g, model, geo, err)) {
            std::fprintf(stderr, "k31::read refuses the pack's model: %s\n", err.c_str());
            return 1;
        }
        const auto eq = [&](const char* what, double a, double b) {
            if (a != b) fail(std::string(what) + ": pack " + std::to_string(a) + ", k31::read " + std::to_string(b));
        };
        eq("n_layers", p.n_layers, geo.n_layers);
        eq("n_embd", p.n_embd, geo.n_embd);
        eq("vocab", p.vocab, geo.n_vocab);
        eq("kda_heads", p.kda_heads, geo.n_kda_head);
        eq("kda_head_dim", p.kda_head_dim, geo.kda_head_dim);
        eq("d_conv", p.d_conv, geo.ssm_d_conv);
        eq("kv_lora", p.kv_lora, geo.kv_lora_rank);
        eq("q_lora", p.q_lora, geo.q_lora_rank);
        eq("qk_nope", p.qk_nope, geo.qk_nope);
        eq("qk_rope", p.qk_rope, geo.n_rot);
        eq("v_head", p.v_head, geo.n_head_v_mla);
        eq("idx_heads", p.idx_heads, geo.idx_n_head);
        eq("idx_dim", p.idx_dim, geo.idx_key_dim);
        eq("idx_topk", p.idx_topk, geo.idx_top_k);
        eq("attn_res_block", p.attn_res_block, geo.attn_res_block_size);
        eq("rms_eps", (float) p.rms_eps, geo.eps_rms);
        eq("lora_eps", (float) p.lora_eps, geo.eps_mla_latent);
        eq("kda_l2_eps", (float) p.kda_l2_eps, geo.eps_kda_l2);
        eq("idx_ln_eps", (float) p.idx_ln_eps, geo.eps_indexer_ln);
        eq("gate_lower_bound", (float) p.gate_lower_bound, geo.kda_gate_lower_bound);
        eq("source_count", p.source_count, geo.expert_source_count);
        for (const auto& layer : p.layers) {
            const int il = layer.index;
            const std::string L = "layer " + std::to_string(il) + " ";
            if ((layer.attn == "kda") != geo.is_recr(il)) fail(L + "attention " + layer.attn);
            if ((layer.ffn == "moe") != geo.is_moe(il)) fail(L + "ffn " + layer.ffn);
            if (layer.ffn != "moe") continue;
            eq((L + "n_exp").c_str(), layer.n_exp, geo.n_expert);
            eq((L + "k").c_str(), layer.k, geo.n_expert_used);
            eq((L + "edim").c_str(), layer.edim, geo.n_expert_latent);
            eq((L + "eff").c_str(), layer.eff, geo.n_ff_exp[(size_t) il]);
            auto it = p.keep.find(il);
            if (it == p.keep.end() || it->second != geo.keep[(size_t) il])
                fail(L + "keep ids differ from k31::read's (" + geo.keep_from + ")");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cannot read %s: %s\n", model.c_str(), e.what());
        return 1;
    }
    if (bad) {
        std::fprintf(stderr, "%d differences between the pack and %s\n", bad, model.c_str());
        return 1;
    }
    int kda = 0, mla = 0, experts = 0;
    for (const auto& layer : p.layers) {
        if (layer.attn == "kda") ++kda;
        if (layer.attn == "mla") ++mla;
        if (layer.ffn == "moe") experts = layer.n_exp;
    }
    std::printf("pack %d layers, KDA %d, MLA %d, experts %d, source %d, format %s, %zu tensors, data %s: agrees with "
                "the file and k31::read\n",
                p.n_layers, kda, mla, experts, p.source_count, p.classes.empty() ? "-" : p.classes[0].format.c_str(),
                p.tensors.size(), p.data.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        bool header_only = false;
        std::string path;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--header-only") header_only = true;
            else if (path.empty() && a[0] != '-') path = a;
            else {
                std::fprintf(stderr, "usage: k31_pack_test [PACK/k31.txt [--header-only]]\n");
                return 2;
            }
        }
        return check_file(path, header_only);
    }
    refusals();
    if (failures) {
        std::fprintf(stderr, "%d of %d pack cases did not match\n", failures, cases);
        return 1;
    }
    std::printf("k31_pack_test: %d cases ok\n", cases);
    return 0;
}
