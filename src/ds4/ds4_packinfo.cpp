// src/ds4/ds4_packinfo.cpp - `maya-ds4 --pack-info` (ds4_packinfo.hpp): the pack as its description says it is,
// before anything is loaded.  A pack is pointers - ds4.txt (the geometry and model_dir), tensors.txt (every tensor's
// type, file, offset, bytes and shape: tools/ds4/pack.py, pack_gguf.py) and the model's files where model_dir says -
// so a pack whose model_dir was cleaned up still reads perfectly and only fails at the load.  Here every file is
// looked at (there, and as long as its last tensor's end): such a pack is called a dead pack.  The checks follow the
// loader's (ds4_model.cu: Pack::load, Geometry::from, expert_layout, mat, the embedding and Engram tables).
#include "ds4_packinfo.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::ds4 {
namespace {

namespace fs = std::filesystem;

struct Tensor {
    std::string dtype;
    int file = -1;
    uint64_t off = 0, bytes = 0;
    std::vector<int64_t> shape;
};

struct Desc {
    std::unordered_map<std::string, std::string> kv;
    std::vector<std::string> files;
    std::unordered_map<std::string, Tensor> t;

    bool has(const char* k) const { return kv.count(k) != 0; }
    std::string str(const char* k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    long long num(const char* k, long long def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::atoll(it->second.c_str());
    }
    std::vector<int> list(const char* k) const {
        std::vector<int> v;
        std::stringstream ss(str(k));
        std::string x;
        while (std::getline(ss, x, ','))
            if (!x.empty()) v.push_back(std::atoi(x.c_str()));
        return v;
    }
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
};

bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}
bool starts_with(const std::string& s, const char* pre) { return s.rfind(pre, 0) == 0; }

// ds4.txt: a key and its value a line; tensors.txt: `file <i> <name>` and `t <name> <type> <file> <offset> <bytes>
// <shape ...>` lines (the loader's reading, without its strtok)
bool read_desc(const std::string& dir, Desc& d, std::FILE* out) {
    {
        std::ifstream f(dir + "/ds4.txt");
        if (!f) {
            std::fprintf(out, "pack-info: %s/ds4.txt: not found (not a maya-ds4 pack: tools/ds4/pack.py or pack_gguf.py "
                              "writes one)\n", dir.c_str());
            return false;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t sp = line.find(' ');
            if (sp != std::string::npos) d.kv[line.substr(0, sp)] = line.substr(sp + 1);
        }
    }
    std::ifstream f(dir + "/tensors.txt");
    if (!f) {
        std::fprintf(out, "pack-info: %s/tensors.txt: not found\n", dir.c_str());
        return false;
    }
    std::string line;
    std::vector<std::string> w;
    while (std::getline(f, line)) {
        w.clear();
        std::istringstream ss(line);
        for (std::string x; ss >> x;) w.push_back(std::move(x));
        if (w.size() >= 3 && w[0] == "file") {
            const int i = std::atoi(w[1].c_str());
            if (i < 0) continue;
            if ((int) d.files.size() <= i) d.files.resize((size_t) i + 1);
            d.files[(size_t) i] = w[2];
        } else if (w.size() >= 6 && w[0] == "t") {
            Tensor x;
            x.dtype = w[2];
            x.file = std::atoi(w[3].c_str());
            x.off = std::strtoull(w[4].c_str(), nullptr, 10);
            x.bytes = std::strtoull(w[5].c_str(), nullptr, 10);
            for (size_t i = 6; i < w.size(); ++i) x.shape.push_back(std::atoll(w[i].c_str()));
            d.t[w[1]] = std::move(x);
        }
    }
    if (d.t.empty()) {
        std::fprintf(out, "pack-info: %s/tensors.txt: no tensors\n", dir.c_str());
        return false;
    }
    return true;
}

std::string k_name(int k2) {   // an EXL3 bitrate: K 1..8, or 1.5 / 2.5 / 3.5
    char b[32];
    if (k2 % 2 == 0) std::snprintf(b, sizeof b, "EXL3 %d", k2 / 2);
    else std::snprintf(b, sizeof b, "EXL3 %.1f", k2 / 2.0);
    return b;
}

// one role (gate w1, up w3, down w2) of a routed expert: its type's name, whether a kernel takes it, its bytes
struct Role {
    std::string type;
    bool ok = false, found = false, exl3 = false;
    uint64_t bytes = 0;
};

struct Ctx {
    const Desc& d;
    const KernelTypes& kt;
    std::map<std::string, int> ids;   // a ggml type's name -> its id (the kernels' own names)

    int id_of(const std::string& name) const {
        auto it = ids.find(name);
        return it == ids.end() ? -1 : it->second;
    }
    // an EXL3 matrix (trellis, suh, svh, mul1) - the loader's rule: whole 16x16 tiles of k2 / 2 bits, mul1's codebook
    bool exl3_ok(const std::string& pre, const Tensor& tr, std::string& type) const {
        if (tr.shape.size() != 3 || tr.shape[2] % 8 != 0) {
            type = "EXL3 ?";
            return false;
        }
        const int k2 = (int) (tr.shape[2] / 8);
        type = k_name(k2);
        if (d.get(pre + ".mul1") == nullptr) {
            type += " (not mul1)";
            return false;
        }
        return kt.exl3_k2 != nullptr && kt.exl3_k2(k2);
    }
    Role role(const std::string& pre) const {
        Role r;
        if (const Tensor* w = d.get(pre + ".weight")) {
            r.found = true;
            r.type = w->dtype;
            r.bytes = w->bytes;
            const int id = id_of(w->dtype);
            r.ok = id >= 0 && kt.expert != nullptr && kt.expert(id);
        } else if (const Tensor* tr = d.get(pre + ".trellis")) {
            r.found = r.exl3 = true;
            r.ok = exl3_ok(pre, *tr, r.type);
            for (const char* s : {".trellis", ".suh", ".svh", ".mul1"})
                if (const Tensor* x = d.get(pre + s)) r.bytes += x->bytes;
        }
        return r;
    }
};

// a layer's routed experts: the types of their gate / up and down (experts each), their bytes
struct Layer {
    std::string label;
    int n_exp = 0, k = 0;
    std::map<std::string, int> gu, dn;
    std::vector<uint64_t> eb;   // each expert's bytes (its three matrices)
    double bytes = 0;
    bool gguf = false, exl3 = false, gate_up_differ = false;
    int missing = 0;            // experts with a matrix missing
    std::set<std::string> bad;  // types no kernel takes
};

Layer layer_map(const Ctx& c, const std::string& label, const std::string& prefix, int n_exp, int k) {
    Layer L;
    L.label = label;
    L.k = k;
    L.n_exp = n_exp;
    L.eb.assign((size_t) n_exp, 0);
    for (int e = 0; e < n_exp; ++e) {
        const std::string E = prefix + std::to_string(e) + ".";
        const Role g = c.role(E + "w1"), u = c.role(E + "w3"), dw = c.role(E + "w2");
        if (!g.found || !u.found || !dw.found) {
            ++L.missing;
            continue;
        }
        L.eb[(size_t) e] = g.bytes + u.bytes + dw.bytes;
        L.bytes += (double) L.eb[(size_t) e];
        ++L.gu[g.type == u.type ? g.type : g.type + "/" + u.type];
        ++L.dn[dw.type];
        L.gate_up_differ = L.gate_up_differ || g.type != u.type;
        (g.exl3 ? L.exl3 : L.gguf) = true;
        for (const Role* r : {&g, &u, &dw})
            if (!r->ok) L.bad.insert(r->type);
    }
    return L;
}

// the experts a layer prefix holds (expert e while its gate is in the pack)
int count_experts(const Desc& d, const std::string& prefix) {
    int n = 0;
    while (d.get(prefix + std::to_string(n) + ".w1.weight") != nullptr ||
           d.get(prefix + std::to_string(n) + ".w1.trellis") != nullptr)
        ++n;
    return n;
}

std::string types_text(const std::map<std::string, int>& m) {   // "IQ2_S", or "EXL3 3 x84 + EXL3 4 x300" (mixed)
    if (m.size() == 1) return m.begin()->first;
    std::string s;
    for (const auto& [ty, n] : m) s += (s.empty() ? "" : " + ") + ty + " x" + std::to_string(n);
    return s.empty() ? "-" : s;
}

double gb(double b) { return b / 1e9; }

}  // namespace

int pack_info(const std::string& dir, const std::string& profile_path, const KernelTypes& kt, std::FILE* out) {
    Desc d;
    if (!read_desc(dir, d, out)) return 1;
    Ctx c{d, kt, {}};
    for (int ty = 0; ty < 64 && kt.name != nullptr; ++ty) {
        const char* n = kt.name(ty);
        if (n != nullptr && std::strcmp(n, "?") != 0) c.ids[n] = ty;
    }
    std::vector<std::string> cannot, warn;   // what stops the load; what is suspicious
    const bool gguf = d.str("format") == "gguf";

    // ---- the geometry (Geometry::from and the load's checks)
    const long long nl = d.num("num_hidden_layers", 0), ne = d.num("n_routed_experts", 0);
    const long long k = d.num("num_experts_per_tok", 0), dim = d.num("hidden_size", 0);
    const long long ff = d.num("moe_intermediate_size", 0), vocab = d.num("vocab_size", 0);
    const std::vector<int> ratios = d.list("compress_ratios");
    if (dim == 0 || nl == 0 || (long long) ratios.size() < nl)
        cannot.push_back("ds4.txt: incomplete geometry (hidden_size, num_hidden_layers, compress_ratios for every layer)");
    if (d.num("head_dim", 0) != 512 || d.num("index_head_dim", 0) != 128 || k > 8 || d.num("hc_mult", 0) > 8)
        cannot.push_back("geometry the engine does not cover (head_dim " + d.str("head_dim", "?") + ", index_head_dim " +
                         d.str("index_head_dim", "?") + ", experts a token " + std::to_string(k) + ")");
    if (ne == 0 || k == 0 || ff == 0 || vocab == 0)
        cannot.push_back("ds4.txt: n_routed_experts, num_experts_per_tok, moe_intermediate_size or vocab_size missing");

    // ---- the drafter (DSpark): the engine loads its blocks only with a block size and their compress ratios
    const long long d_block = d.num("dspark_block_size", 0), d_noise = d.num("dspark_noise_token_id", 0);
    const long long d_rank = d.num("dspark_markov_rank", 0), d_ne = d.num("dspark_n_routed_experts", 0);
    const long long d_k = d.num("dspark_num_experts_per_tok", 0);
    const std::vector<int> d_targets = d.list("dspark_target_layer_ids");
    int mtp_in_pack = 0;   // mtp.<i> blocks the tensors hold
    while (d.get("mtp." + std::to_string(mtp_in_pack) + ".attn.wq_a.trellis") != nullptr ||
           d.get("mtp." + std::to_string(mtp_in_pack) + ".attn.wq_a.weight") != nullptr)
        ++mtp_in_pack;
    const int n_mtp =
        d_block > 0 && (long long) ratios.size() > nl && mtp_in_pack > 0 ? (int) ((long long) ratios.size() - nl) : 0;
    if (mtp_in_pack > 0 && d_block <= 0)
        warn.push_back("dspark_block_size " + d.str("dspark_block_size", "missing") + " beside " +
                       std::to_string(mtp_in_pack) + " drafter blocks in the tensors: the engine does not load the "
                       "drafter (no DSpark drafts; DeepSeek's checkpoint says 5)");
    if (mtp_in_pack > 0 && d_block > 0 && (long long) ratios.size() <= nl)
        warn.push_back("compress_ratios has no entries past the trunk's " + std::to_string(nl) + " layers: the engine does "
                       "not load the drafter's " + std::to_string(mtp_in_pack) + " blocks");
    if (n_mtp > 0) {
        if (n_mtp != mtp_in_pack)
            warn.push_back("compress_ratios gives " + std::to_string(n_mtp) + " drafter blocks, the tensors hold " +
                           std::to_string(mtp_in_pack));
        if (d_block == 1) warn.push_back("dspark_block_size 1: one draft a round at most");
        if (d_block > 16)
            warn.push_back("dspark_block_size " + std::to_string(d_block) + ": more than a window's 16 rows, the drafter "
                           "never drafts");
        if (d_block > 1 && (d_noise <= 0 || d_noise >= vocab))
            warn.push_back("dspark_noise_token_id " + d.str("dspark_noise_token_id", "missing") +
                           ": not a token of the vocabulary");
        if (d_ne <= 0 || d_k <= 0 || d_k > 8)
            warn.push_back("dspark_n_routed_experts " + d.str("dspark_n_routed_experts", "missing") +
                           ", dspark_num_experts_per_tok " + d.str("dspark_num_experts_per_tok", "missing") +
                           ": the drafter's routing is not described");
        if (d_targets.empty()) warn.push_back("dspark_target_layer_ids missing: the drafter reads no trunk layer");
        for (int il : d_targets)
            if (il < 0 || il >= nl)
                warn.push_back("dspark_target_layer_ids names layer " + std::to_string(il) + ", not a trunk layer");
        if (const Tensor* te = d.get("mtp." + std::to_string(n_mtp - 1) + ".markov_head.embed.weight"))
            if (te->dtype != "BF16" || te->bytes != (uint64_t) vocab * (uint64_t) std::max(0LL, d_rank) * 2)
                warn.push_back("dspark_markov_rank " + std::to_string(d_rank) + " does not match the markov embed (" +
                               te->dtype + ", " + std::to_string(te->bytes) +
                               " bytes): a tensor-parallel head stops the load at it");
    }

    // ---- the routed experts: the trunk's layers, then the drafter's blocks
    std::vector<Layer> lays;
    for (long long il = 0; il < nl; ++il)
        lays.push_back(
            layer_map(c, std::to_string(il), "layers." + std::to_string(il) + ".ffn.experts.", (int) ne, (int) k));
    const int d_have = mtp_in_pack > 0 ? count_experts(d, "mtp.0.ffn.experts.") : 0;
    for (int i = 0; i < mtp_in_pack; ++i)
        lays.push_back(layer_map(c, "mtp." + std::to_string(i), "mtp." + std::to_string(i) + ".ffn.experts.",
                                 (int) (d_ne > 0 ? d_ne : d_have), (int) d_k));
    if (n_mtp > 0 && d_ne > 0 && d_have != d_ne)
        warn.push_back("dspark_n_routed_experts " + std::to_string(d_ne) + ", the drafter's first block holds " +
                       std::to_string(d_have) + " experts");
    if (ne > 0 && nl > 0 && count_experts(d, "layers.0.ffn.experts.") != ne)
        warn.push_back("n_routed_experts " + std::to_string(ne) + ", layer 0 holds " +
                       std::to_string(count_experts(d, "layers.0.ffn.experts.")) + " experts");
    std::map<std::string, std::set<std::string>> no_kernel;   // type -> where
    std::map<std::string, long long> by_type;                 // expert matrices by type
    double trunk_b = 0, trunk_tok = 0, draft_b = 0;
    for (size_t i = 0; i < lays.size(); ++i) {
        const Layer& L = lays[i];
        const bool mtp = (long long) i >= nl;
        if (L.missing > 0)
            cannot.push_back((mtp ? "drafter block " : "layer ") + L.label + ": " + std::to_string(L.missing) + " of " +
                             std::to_string(L.n_exp) + " experts with a matrix missing");
        if (L.gguf && (L.gu.size() > 1 || L.dn.size() > 1))   // (expert_layout_mx: one type and size a role and layer)
            cannot.push_back("layer " + L.label + ": GGUF experts of several types in one role (" + types_text(L.gu) +
                             " / " + types_text(L.dn) + ")");
        if (L.gguf && L.gate_up_differ) cannot.push_back("layer " + L.label + ": gate and up of different types");
        if (L.gguf && L.exl3) cannot.push_back("layer " + L.label + ": EXL3 and GGUF experts mixed in one layer");
        for (const std::string& b : L.bad) no_kernel["routed experts in " + b].insert(L.label);
        for (const auto& [ty, n] : L.gu) by_type[ty] += 2LL * n;
        for (const auto& [ty, n] : L.dn) by_type[ty] += n;
        const double mean = L.n_exp > 0 ? L.bytes / (double) L.n_exp : 0.0;
        if (mtp) draft_b += L.bytes;
        else {
            trunk_b += L.bytes;
            trunk_tok += (double) L.k * mean;
        }
    }

    // ---- everything else, by where the engine puts it, and the types it reads it in
    double b_dense = 0, b_head = 0, b_mtp = 0, b_embed = 0, b_engram = 0, b_vision = 0, b_unread = 0;
    std::set<std::string> dense_types, eg_types;
    uint64_t embed_ram = 0;
    for (const auto& [name, x] : d.t) {
        if (name.find(".ffn.experts.") != std::string::npos && (starts_with(name, "layers.") || starts_with(name, "mtp.")))
            continue;
        const size_t dot = name.rfind('.');
        const std::string base = dot == std::string::npos ? name : name.substr(0, dot);
        const bool sib_trellis = d.get(base + ".trellis") != nullptr;
        auto bad = [&](const std::string& why) { no_kernel[why].insert(name); };
        if (name.find(".engram.embed.") != std::string::npos) {   // the Engram tables: the host's reader, from the files
            b_engram += (double) x.bytes;
            if (ends_with(name, ".weight")) eg_types.insert(x.dtype);
            if (ends_with(name, ".weight") && x.dtype != "Q6_K" && x.dtype != "IQ4_NL" && x.dtype != "MXFP8" &&
                x.dtype != "F8_E4M3")
                bad("an Engram table in " + x.dtype + " (the reader takes Q6_K, IQ4_NL, MXFP8 or FP8)");
            continue;
        }
        if (name == "embed.weight") {   // the token embedding: pinned RAM, decoded to BF16 rows at the load
            b_embed += (double) x.bytes;
            embed_ram = (uint64_t) std::max(0LL, vocab) * (uint64_t) std::max(0LL, dim) * 2;
            if (x.dtype != "BF16" && x.dtype != "Q8_0" && x.dtype != "Q4_K" && x.dtype != "Q5_K" && x.dtype != "Q6_K")
                bad("the token embedding in " + x.dtype + " (BF16, Q8_0, Q4_K, Q5_K or Q6_K)");
            if (x.shape.size() == 2 && (x.shape[0] != vocab || x.shape[1] != dim))
                cannot.push_back("embed.weight is [" + std::to_string(x.shape[0]) + "][" + std::to_string(x.shape[1]) +
                                 "], ds4.txt says [" + std::to_string(vocab) + "][" + std::to_string(dim) + "]");
            continue;
        }
        if (starts_with(name, "vision.") || starts_with(name, "aligner.") || starts_with(name, "image_")) {
            b_vision += (double) x.bytes;   // (the tower borrows its room while a picture is read)
            if (x.dtype != "BF16" && x.dtype != "F16" && x.dtype != "F32") bad("the vision tower in " + x.dtype);
            continue;
        }
        if (!ends_with(name, ".trellis") && !ends_with(name, ".suh") && !ends_with(name, ".svh") &&
            !ends_with(name, ".mul1") && sib_trellis) {   // (another copy of a matrix the engine reads as EXL3)
            b_unread += (double) x.bytes;
            continue;
        }
        double& part = starts_with(name, "mtp.") ? b_mtp : starts_with(name, "head.") ? b_head : b_dense;
        part += (double) x.bytes;
        if (ends_with(name, ".trellis")) {
            std::string ty;
            if (!c.exl3_ok(base, x, ty)) bad("a matrix in " + ty);
            dense_types.insert("EXL3");
            continue;
        }
        if (ends_with(name, ".suh") || ends_with(name, ".svh") || ends_with(name, ".mul1")) continue;
        dense_types.insert(x.dtype);
        if (x.dtype == "BF16" || x.dtype == "F16" || x.dtype == "F32" || x.dtype == "I32") continue;   // (vectors, norms)
        const int id = c.id_of(x.dtype);
        if (id < 0 || kt.dense == nullptr || !kt.dense(id)) bad("a dense tensor in " + x.dtype);
    }

    // ---- the expert profile: [layer][expert] uint32 counts, the trunk's layers first (then the drafter's)
    const std::string prof_path = profile_path.empty() ? dir + "/expert-profile.bin" : profile_path;
    std::vector<uint32_t> prof;
    std::error_code ec;
    const bool prof_there = fs::is_regular_file(prof_path, ec);
    uint64_t prof_size = prof_there ? fs::file_size(prof_path, ec) : 0;
    if (prof_there && !ec) {
        std::ifstream pf(prof_path, std::ios::binary);
        prof.resize((size_t) (prof_size / 4));
        pf.read(reinterpret_cast<char*>(prof.data()), (std::streamsize) (prof.size() * 4));
        if (!pf) prof.clear();
    }

    // ---- the files the pack points at: every one listed is opened at the load, each as long as its last tensor's end
    const std::string md = d.str("model_dir");
    std::vector<uint64_t> need(d.files.size(), 0);
    int bad_file_refs = 0;
    for (const auto& [name, x] : d.t) {
        if (x.file < 0 || (size_t) x.file >= d.files.size() || d.files[(size_t) x.file].empty()) {
            ++bad_file_refs;
            continue;
        }
        need[(size_t) x.file] = std::max<uint64_t>(need[(size_t) x.file], x.off + x.bytes);
    }
    std::vector<std::string> dead;
    double files_b = 0;
    const bool md_there = !md.empty() && fs::is_directory(md, ec);
    if (md.empty()) dead.push_back("ds4.txt names no model_dir: the pack points at no weights");
    else if (!md_there) dead.push_back("model_dir " + md + " is not there (moved, or cleaned up)");
    if (bad_file_refs > 0)
        dead.push_back(std::to_string(bad_file_refs) + " tensors name a file tensors.txt does not list");
    int missing_files = 0;
    for (size_t i = 0; i < d.files.size() && !md.empty(); ++i) {
        if (d.files[i].empty()) {
            dead.push_back("file " + std::to_string(i) + ": no name in tensors.txt");
            continue;
        }
        const std::string p = md + "/" + d.files[i];
        std::error_code e2;
        const bool there = fs::is_regular_file(p, e2);
        const uint64_t sz = there ? fs::file_size(p, e2) : 0;
        if (!there || e2) {
            ++missing_files;
            if (missing_files <= 8 && md_there) dead.push_back("MISSING " + p);
            continue;
        }
        files_b += (double) sz;
        if (sz < need[i]) {
            char b[96];
            std::snprintf(b, sizeof b, ": %.2f GB there, its tensors reach %.2f GB", gb((double) sz),
                          gb((double) need[i]));
            dead.push_back("SHORT " + p + b);
        }
    }
    if (missing_files > 8 && md_there) dead.push_back("... " + std::to_string(missing_files - 8) + " more");
    if (missing_files > 0)
        dead.insert(dead.begin() + (md_there ? 0 : 1), std::to_string(missing_files) + " of the " +
                                                          std::to_string(d.files.size()) + " files it lists are missing" +
                                                          (md_there ? ":" : ""));

    // ---- the pack's own files beside the description
    {
        std::ifstream eb(dir + "/engram.bin", std::ios::binary);
        char magic[4] = {};
        int32_t hdr[6] = {};
        if (!eb) cannot.push_back("engram.bin is not in the pack (the load reads it; tools/ds4/pack.py writes it)");
        else if (!eb.read(magic, 4) || std::memcmp(magic, "DSE1", 4) != 0 ||
                 !eb.read(reinterpret_cast<char*>(hdr), sizeof hdr))
            cannot.push_back("engram.bin: unreadable (not DSE1)");
        else {
            const std::vector<int> eg = d.list("engram_layer_ids");
            if (hdr[1] != (int32_t) eg.size())
                warn.push_back("engram.bin holds " + std::to_string(hdr[1]) + " Engram layers, engram_layer_ids names " +
                               std::to_string(eg.size()));
            if (hdr[0] != (int32_t) vocab)
                warn.push_back("engram.bin's vocabulary " + std::to_string(hdr[0]) + ", vocab_size " +
                               std::to_string(vocab));
            for (int il : eg)
                if (d.get("layers." + std::to_string(il) + ".engram.embed.weight") == nullptr)
                    cannot.push_back("Engram layer " + std::to_string(il) + ": no table (layers." + std::to_string(il) +
                                     ".engram.embed.weight)");
        }
    }
    if (!fs::is_regular_file(dir + "/tokenizer/vocab.json", ec))
        warn.push_back("no tokenizer/vocab.json in the pack: the server reads the tokenizer there");

    // ---- the report
    std::fprintf(out, "pack-info %s\n", dir.c_str());
    std::fprintf(out, "  %s, model_dir %s\n", gguf ? "GGUF" : "EXL3", md.empty() ? "(none)" : md.c_str());
    std::fprintf(out, "  %lld layers x %lld routed experts, %lld a token; dim %lld, ff %lld, vocab %lld; %s\n", nl, ne, k,
                 dim, ff, vocab,
                 n_mtp > 0 ? ("the DSpark drafter: " + std::to_string(n_mtp) + " blocks").c_str()
                 : mtp_in_pack > 0 ? "the DSpark drafter's blocks are in the tensors, not loaded (below)" : "no drafter");
    std::fprintf(out, "routed experts (gate, up / down; an expert's bytes, a token's picks):\n");
    std::fprintf(out, "  %6s  %-22s %-22s %12s %12s\n", "layer", "gate, up", "down", "an expert", "a token");
    for (const Layer& L : lays) {
        const double mean = L.n_exp > 0 ? L.bytes / (double) L.n_exp : 0.0;
        std::fprintf(out, "  %6s  %-22s %-22s %9.2f MB %9.2f MB%s\n", L.label.c_str(), types_text(L.gu).c_str(),
                     types_text(L.dn).c_str(), mean / 1e6, (double) L.k * mean / 1e6,
                     L.gu.size() > 1 || L.dn.size() > 1 ? "  (mixed)" : "");
    }
    std::fprintf(out, "  expert matrices by type:");
    for (auto it = by_type.begin(); it != by_type.end(); ++it)
        std::fprintf(out, "%s %s %lld", it == by_type.begin() ? "" : ",", it->first.c_str(), it->second);
    std::fprintf(out, "\n");
    const double params = (double) nl * (double) ne * 3.0 * (double) ff * (double) dim;
    std::fprintf(out, "  the trunk's %lld experts %.2f GB, %.2f bits a weight; a token reads %.3f GB of them (%lld a "
                      "layer)\n", nl * ne, gb(trunk_b), params > 0 ? trunk_b * 8.0 / params : 0.0, gb(trunk_tok), k);
    if (draft_b > 0) std::fprintf(out, "  the drafter's experts %.2f GB\n", gb(draft_b));
    std::fprintf(out, "where it goes (GB):\n");
    std::fprintf(out, "  the GPUs: dense %.2f (attention, shared experts, routers, norms), head %.2f%s",
                 gb(b_dense), gb(b_head), b_mtp + draft_b > 0 ? "" : "\n");
    if (b_mtp + draft_b > 0)
        std::fprintf(out, ", the drafter's blocks %.2f + their experts %.2f%s\n", gb(b_mtp), gb(draft_b),
                     n_mtp > 0 ? "" : " (not loaded)");
    std::fprintf(out, "  the routed experts %.2f: VRAM, RAM and the files, the most-routed first (the expert profile, "
                      "below)\n", gb(trunk_b));
    std::fprintf(out, "  pinned RAM: the token embedding %.2f (BF16 rows; %.2f in the files)\n", gb((double) embed_ram),
                 gb(b_embed));
    std::fprintf(out, "  read from the files as tokens need them: the Engram tables %.2f\n", gb(b_engram));
    if (b_vision > 0) std::fprintf(out, "  borrowed while a picture is read: the vision tower %.2f\n", gb(b_vision));
    if (b_unread > 0) std::fprintf(out, "  not read (another copy of a matrix read as EXL3): %.2f\n", gb(b_unread));
    std::fprintf(out, "  dense types:");
    for (const std::string& t : dense_types) std::fprintf(out, " %s", t.c_str());
    std::fprintf(out, "\n");

    // the tiers: how much of the profile's routing the most-routed experts take, and their bytes
    const size_t trunk_n = (size_t) std::max(0LL, nl) * (size_t) std::max(0LL, ne);
    if (!prof_there) {
        std::fprintf(out, "expert profile %s: none (the load places experts in no order until it has counted some)\n",
                     prof_path.c_str());
    } else if (prof_size % 4 != 0 || prof.size() < trunk_n || trunk_n == 0) {
        std::fprintf(out, "expert profile %s: %llu bytes\n", prof_path.c_str(), (unsigned long long) prof_size);
        warn.push_back("the expert profile holds " + std::to_string(prof_size / 4) + " counts, not " + std::to_string(nl) +
                       " x " + std::to_string(ne) + ": the engine ignores it and starts from none");
    } else {
        const size_t with_d = trunk_n + (size_t) n_mtp * (size_t) ne;
        if (prof.size() != trunk_n && prof.size() < with_d)
            warn.push_back("the expert profile holds " + std::to_string(prof.size()) + " counts, neither the trunk's " +
                           std::to_string(trunk_n) + " nor with the drafter's " + std::to_string(with_d) +
                           ": the engine ignores it when the drafter is loaded");
        else if (prof.size() > with_d)
            warn.push_back("the expert profile holds " + std::to_string(prof.size()) + " counts, more than this pack's " +
                           std::to_string(with_d) + " (another model's?): the engine reads the first ones");
        std::vector<std::pair<uint32_t, uint64_t>> v;   // (count, bytes) a trunk expert
        double total = 0;
        size_t zero = 0;
        double zero_b = 0;
        for (size_t il = 0; il < (size_t) nl; ++il)
            for (size_t e = 0; e < (size_t) ne; ++e) {
                const uint32_t n = prof[il * (size_t) ne + e];
                const uint64_t b = e < lays[il].eb.size() ? lays[il].eb[e] : 0;
                v.emplace_back(n, b);
                total += n;
                if (n == 0) {
                    ++zero;
                    zero_b += (double) b;
                }
            }
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::fprintf(out, "expert profile %s: %zu counts (%s), %.0f routes, about %.0f tokens' worth\n", prof_path.c_str(),
                     prof.size(), prof.size() >= with_d && n_mtp > 0 ? "the drafter's too" : "the trunk's", total,
                     nl > 0 && k > 0 ? total / (double) (nl * k) : 0.0);
        if (total <= 0) {
            warn.push_back("the expert profile counts nothing: the load places experts in no order");
        } else {
            std::fprintf(out, "  the most-routed experts holding");
            const double shares[] = {0.5, 0.8, 0.9, 0.95, 0.99};
            size_t j = 0;
            double cum = 0, cum_b = 0;
            for (double s : shares) {
                for (; j < v.size() && cum < s * total; ++j) {
                    cum += v[j].first;
                    cum_b += (double) v[j].second;
                }
                std::fprintf(out, "%s %.0f%%%s: %zu (%.1f GB)", s == shares[0] ? "" : ";", 100.0 * s,
                             s == shares[0] ? " of its routes" : "", j, gb(cum_b));
            }
            std::fprintf(out, "\n  never routed: %zu experts (%.1f GB)\n", zero, gb(zero_b));
        }
    }
    // the drafter's and the Engram's metadata, as the engine reads them
    if (mtp_in_pack > 0 || d.has("dspark_block_size"))
        std::fprintf(out, "drafter: %d blocks in the tensors, %d loaded; block size %lld, noise token %lld, markov rank "
                          "%lld, %lld experts (%lld a token), reading trunk layers %s\n",
                     mtp_in_pack, n_mtp, d_block, d_noise, d_rank, d_ne, d_k,
                     d.str("dspark_target_layer_ids", "(none)").c_str());
    std::string egt;
    for (const std::string& t : eg_types) egt += (egt.empty() ? "" : ", ") + t;
    std::fprintf(out, "Engram: layers %s, tables in %s\n", d.str("engram_layer_ids", "(none)").c_str(),
                 egt.empty() ? "(none)" : egt.c_str());
    if (no_kernel.empty()) {
        std::fprintf(out, "no GPU kernel: none (every tensor is of a type the engine reads)\n");
    } else {
        std::fprintf(out, "no GPU kernel (the load stops at these):\n");
        for (const auto& [why, where] : no_kernel) {
            std::string s;
            int shown = 0;
            for (const std::string& w : where) {
                if (++shown > 6) {
                    s += " ... (" + std::to_string(where.size()) + " in all)";
                    break;
                }
                s += (s.empty() ? "" : ", ") + w;
            }
            std::fprintf(out, "  %s: %s\n", why.c_str(), s.c_str());
        }
    }
    if (dead.empty())
        std::fprintf(out, "files: %zu in model_dir, all there and long enough (%.1f GB)\n", d.files.size(), gb(files_b));
    for (const std::string& s : warn) std::fprintf(out, "WARNING: %s\n", s.c_str());
    for (const std::string& s : cannot) std::fprintf(out, "CANNOT LOAD: %s\n", s.c_str());
    if (!dead.empty()) {
        std::fprintf(out, "DEAD PACK: the description is whole, the weights it points at are not:\n");
        for (const std::string& s : dead) std::fprintf(out, "  %s\n", s.c_str());
        std::fprintf(out, "  (pack the model again - tools/ds4/pack.py, pack_gguf.py - or point model_dir at the "
                          "weights)\n");
    }
    const bool ok = dead.empty() && cannot.empty() && no_kernel.empty();
    std::fprintf(out, "pack-info: %s\n", !ok ? "the pack cannot load"
                                       : warn.empty() ? "the pack can load" : "the pack can load (warnings above)");
    return ok ? 0 : 2;
}

}  // namespace strata::ds4
