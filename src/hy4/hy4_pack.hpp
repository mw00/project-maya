// src/hy4/hy4_pack.hpp - maya-hy4's pack (tools/hy4/pack_gguf.py): the geometry (hy4.txt), every tensor of the GGUF set
// read in place (tensors.txt, each shard mapped once) and the Engram n-gram hash (engram.bin).
//
//   hy4.txt      `key value` lines; lists comma-separated
//   tensors.txt  `file <i> <path>` (relative to the model's folder, model_dir in hy4.txt), then
//                `t <name> <type> <file> <offset> <bytes> <ne0> <ne1> ...` (ggml's dims: ne0 innermost)
//   engram.bin   'HYE1', int32 vocab, modules, ngram, heads_per_ngram, head_dim, pad, table_rows; int32 token_map[vocab];
//                int32 layers[modules]; uint64 multipliers[modules][ngram], primes[modules][cols], offsets[modules][cols]
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mm3_os.hpp"

namespace hy4 {

struct Tensor {
    std::string name, type;
    int file = -1;
    uint64_t off = 0, bytes = 0;
    std::vector<int64_t> ne;   // ggml order: ne[0] innermost
    int64_t n(int i) const { return i < (int) ne.size() ? ne[(size_t) i] : 1; }
};

// The model's shape, from hy4.txt
struct Geo {
    int n_layer = 0, n_mtp = 0, n_embd = 0, n_vocab = 0, n_ff = 0, n_dense_lead = 0;
    int64_t context = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_lat = 0, n_shared = 1;
    float w_scale = 1.0f;
    bool w_norm = true;
    std::vector<float> limit;   // a block's routed-expert SwiGLU clamp (0: none)
    float eps = 1e-5f;
    int q_lora = 0, kv_lora = 0, k_mla = 0, v_mla = 0, rope = 0;
    float rope_base = 1e4f, rope_base_swa = 1e4f;
    int n_swa = 0;
    std::vector<int> heads, swa, indexer;
    int ix_heads = 0, ix_dim = 0, ix_top_k = 0, ix_kpool = 0;
    bool ix_tail = false;
    int hc = 0;
    float hc_eps = 1e-6f, hc_mag = 1.0f;
    int bos = -1, eos = -1;
    int nope() const { return k_mla - rope; }
    int kv_row() const { return kv_lora + rope; }   // a latent cache row: the normed kv_cmpr, then the rotated k_pe
};

// Engram's n-gram hash (the hy_v4_flash graph's engram_hash)
struct Engram {
    int vocab = 0, modules = 0, ngram = 0, heads = 0, head_dim = 0, pad = 0;
    int64_t rows = 0;
    std::vector<int32_t> token_map, layers;
    std::vector<uint64_t> mult, primes, offsets;
    int cols() const { return (ngram - 1) * heads; }
    int module_of(int il) const {
        for (int e = 0; e < modules; ++e)
            if (layers[(size_t) e] == il) return e;
        return -1;
    }
};

struct Pack {
    std::string dir, model_dir;
    std::map<std::string, std::string> kv;
    std::map<std::string, Tensor> t;
    std::vector<std::string> files;
    std::vector<const uint8_t*> map;
    std::vector<size_t> map_bytes;
    Geo g;
    Engram eg;

    bool load(const std::string& pack_dir, std::string& err) {
        dir = pack_dir;
        std::ifstream gf(dir + "/hy4.txt");
        if (!gf) {
            err = dir + "/hy4.txt: cannot open (tools/hy4/pack_gguf.py --model <model> writes it)";
            return false;
        }
        for (std::string line; std::getline(gf, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t sp = line.find(' ');
            if (sp != std::string::npos) kv[line.substr(0, sp)] = line.substr(sp + 1);
        }
        if (str("arch") != "hy_v4_flash") {
            err = dir + "/hy4.txt: arch " + str("arch") + ", not hy_v4_flash";
            return false;
        }
        model_dir = str("model_dir");
        std::ifstream f(dir + "/tensors.txt");
        if (!f) {
            err = dir + "/tensors.txt: cannot open";
            return false;
        }
        for (std::string line; std::getline(f, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream is(line);
            std::string tag;
            is >> tag;
            if (tag == "file") {
                int i;
                std::string p;
                is >> i >> p;
                if (i != (int) files.size()) {
                    err = "tensors.txt: files out of order";
                    return false;
                }
                files.push_back(p);
            } else if (tag == "t") {
                Tensor x;
                is >> x.name >> x.type >> x.file >> x.off >> x.bytes;
                for (int64_t d; is >> d;) x.ne.push_back(d);
                t[x.name] = std::move(x);
            }
        }
        map.assign(files.size(), nullptr);
        map_bytes.assign(files.size(), 0);
        for (size_t i = 0; i < files.size(); ++i) {
            const std::string p = files[i].size() && files[i][0] == '/' ? files[i] : model_dir + "/" + files[i];
            size_t sz = 0;
            const uint8_t* m = mm3::os::map_file(p, sz);
            if (m == nullptr) {
                err = p + ": cannot open or map";
                return false;
            }
            map[i] = m;
            map_bytes[i] = sz;
        }
        for (const auto& [n, x] : t)
            if (x.file < 0 || x.file >= (int) files.size() || x.off + x.bytes > map_bytes[(size_t) x.file]) {
                err = n + ": outside its file";
                return false;
            }
        return geo(err) && engram(err);
    }

    bool geo(std::string& err) {
        g.n_layer = (int) num("n_layer");
        g.n_mtp = (int) num("n_mtp");
        g.n_embd = (int) num("n_embd");
        g.n_vocab = (int) num("n_vocab");
        g.context = (int64_t) num("context_length");
        g.n_ff = (int) num("n_ff");
        g.n_dense_lead = (int) num("n_dense_lead");
        g.n_expert = (int) num("n_expert");
        g.n_used = (int) num("n_expert_used");
        g.n_ff_exp = (int) num("n_ff_exp");
        g.n_lat = (int) num("n_expert_latent");
        g.n_shared = (int) num("n_expert_shared", 1);
        g.w_scale = (float) num("expert_weights_scale", 1.0);
        g.w_norm = num("expert_weights_norm", 1) != 0;
        g.limit = floats("swiglu_limit");
        g.eps = (float) num("rms_eps");
        g.q_lora = (int) num("q_lora");
        g.kv_lora = (int) num("kv_lora");
        g.k_mla = (int) num("k_mla");
        g.v_mla = (int) num("v_mla");
        g.rope = (int) num("rope_dim");
        g.rope_base = (float) num("rope_base");
        g.rope_base_swa = (float) num("rope_base_swa");
        g.n_swa = (int) num("n_swa");
        g.heads = ints("heads");
        g.swa = ints("swa");
        g.indexer = ints("indexer");
        g.ix_heads = (int) num("ix_heads");
        g.ix_dim = (int) num("ix_dim");
        g.ix_top_k = (int) num("ix_top_k");
        g.ix_kpool = (int) num("ix_kpool");
        g.ix_tail = num("ix_select_tail") != 0;
        g.hc = (int) num("hc");
        g.hc_eps = (float) num("hc_eps");
        g.hc_mag = (float) num("hc_magnitude", 1.0);
        g.bos = (int) num("bos", -1);
        g.eos = (int) num("eos", -1);
        const size_t all = (size_t) (g.n_layer + g.n_mtp);
        if (g.n_layer < 1 || g.n_embd < 1 || g.heads.size() != all || g.swa.size() != all || g.indexer.size() != all ||
            g.limit.size() != all || g.hc < 1 || g.rope < 2 || g.rope % 2 || g.k_mla <= g.rope || g.kv_lora % 32 ||
            g.n_lat % 256 || g.n_ff_exp % 256 || g.n_embd % 256 || g.n_used > 16 || g.ix_kpool < 2 || !g.ix_tail ||
            g.ix_dim <= g.rope || g.ix_top_k % g.ix_kpool) {
            err = dir + "/hy4.txt: a geometry maya-hy4 does not run";
            return false;
        }
        return true;
    }

    bool engram(std::string& err) {
        std::ifstream f(dir + "/engram.bin", std::ios::binary);
        if (!f) return true;   // (no Engram layers)
        char magic[4];
        int32_t h[7];
        if (!f.read(magic, 4) || std::memcmp(magic, "HYE1", 4) != 0 || !f.read((char*) h, sizeof h)) {
            err = dir + "/engram.bin: not an Engram hash (HYE1)";
            return false;
        }
        eg.vocab = h[0], eg.modules = h[1], eg.ngram = h[2], eg.heads = h[3], eg.head_dim = h[4], eg.pad = h[5];
        eg.rows = h[6];
        const int cols = eg.cols();
        eg.token_map.resize((size_t) eg.vocab);
        eg.layers.resize((size_t) eg.modules);
        eg.mult.resize((size_t) eg.modules * eg.ngram);
        eg.primes.resize((size_t) eg.modules * cols);
        eg.offsets.resize((size_t) eg.modules * cols);
        if (!f.read((char*) eg.token_map.data(), (std::streamsize) (eg.token_map.size() * 4)) ||
            !f.read((char*) eg.layers.data(), (std::streamsize) (eg.layers.size() * 4)) ||
            !f.read((char*) eg.mult.data(), (std::streamsize) (eg.mult.size() * 8)) ||
            !f.read((char*) eg.primes.data(), (std::streamsize) (eg.primes.size() * 8)) ||
            !f.read((char*) eg.offsets.data(), (std::streamsize) (eg.offsets.size() * 8))) {
            err = dir + "/engram.bin: cut short";
            return false;
        }
        if (eg.ngram < 2 || eg.ngram > 8 || cols < 1 || cols > 64 || eg.head_dim % 32) {
            err = dir + "/engram.bin: an Engram geometry maya-hy4 does not run";
            return false;
        }
        return true;
    }

    bool has(const std::string& k) const { return kv.count(k) != 0; }
    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    double num(const std::string& k, double def = 0) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::atof(it->second.c_str());
    }
    std::vector<int> ints(const std::string& k) const {
        std::vector<int> v;
        std::istringstream is(str(k));
        for (std::string x; std::getline(is, x, ',');) v.push_back(std::atoi(x.c_str()));
        return v;
    }
    std::vector<float> floats(const std::string& k) const {
        std::vector<float> v;
        std::istringstream is(str(k));
        for (std::string x; std::getline(is, x, ',');) v.push_back((float) std::atof(x.c_str()));
        return v;
    }
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
    const uint8_t* data(const Tensor& x) const { return map[(size_t) x.file] + x.off; }
};

}  // namespace hy4
