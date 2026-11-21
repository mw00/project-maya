// src/inkling2/inkling2_pack.hpp - maya-inkling2's pack (tools/inkling2/pack_gguf.py): the geometry (inkling2.txt) and
// every tensor of the GGUF set read in place (tensors.txt, each shard mapped once).
//
//   inkling2.txt  `key value` lines; lists comma-separated
//   tensors.txt   `file <i> <path>` (relative to the model's folder, model_dir in inkling2.txt), then
//                 `t <name> <type> <file> <offset> <bytes> <ne0> <ne1> ...` (ggml's dims: ne0 innermost)
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mm3_os.hpp"

namespace inkling2 {

struct Tensor {
    std::string name, type;
    int file = -1;
    uint64_t off = 0, bytes = 0;
    std::vector<int64_t> ne;   // ggml order: ne[0] innermost
    int64_t n(int i) const { return i < (int) ne.size() ? ne[(size_t) i] : 1; }
};

// The model's shape, from inkling2.txt (docs/INKLING2_AMD.md's constants)
struct Geo {
    int n_layer = 0, n_embd = 0, n_vocab = 0, n_vocab_unpadded = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;
    float eps = 1e-6f;
    int n_expert = 0, n_used = 0, n_ff_exp = 0;
    float w_scale = 0.0f, softcap = 0.0f;
    int n_swa = 0, d_rel = 0, rel_extent = 0, conv = 4, res_block = 6;
    std::vector<int> swa;
    double log_floor = 0.0;
    float log_alpha = 0.0f, qk_clip = 0.0f, rel_clip = 0.0f, situ_beta = 0.0f, situ_lb = 0.0f;
    int bos = 0, eos = 0;
};

struct Pack {
    std::string dir, model_dir;
    std::map<std::string, std::string> kv;
    std::map<std::string, Tensor> t;
    std::vector<std::string> files;
    std::vector<const uint8_t*> map;
    std::vector<size_t> map_bytes;
    Geo g;

    bool load(const std::string& pack_dir, std::string& err) {
        dir = pack_dir;
        std::ifstream gf(dir + "/inkling2.txt");
        if (!gf) {
            err = dir + "/inkling2.txt: cannot open (tools/inkling2/pack_gguf.py --model <model> writes it)";
            return false;
        }
        for (std::string line; std::getline(gf, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t sp = line.find(' ');
            if (sp != std::string::npos) kv[line.substr(0, sp)] = line.substr(sp + 1);
        }
        if (str("arch") != "inkling2") {
            err = dir + "/inkling2.txt: arch " + str("arch") + ", not inkling2";
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
                is >> i;
                std::getline(is >> std::ws, p);
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
        return geo(err);
    }

    bool geo(std::string& err) {
        // (keys the model's files always give - pack_gguf.py refuses a GGUF without them: no fallback here either)
        for (const char* k : {"expert_weights_scale", "final_logit_softcapping", "log_scaling_n_floor", "bos", "eos"})
            if (!kv.count(k)) {
                err = dir + "/inkling2.txt: no " + k + " (tools/inkling2/pack_gguf.py writes it)";
                return false;
            }
        g.n_layer = (int) num("n_layer");
        g.n_embd = (int) num("n_embd");
        g.n_vocab = (int) num("n_vocab");
        g.n_vocab_unpadded = (int) num("n_vocab_unpadded", g.n_vocab);
        g.n_head = (int) num("n_head");
        g.n_head_kv = (int) num("n_head_kv");
        g.head_dim = (int) num("head_dim");
        g.eps = (float) num("rms_eps", 1e-6);
        g.n_expert = (int) num("n_expert");
        g.n_used = (int) num("n_expert_used");
        g.n_ff_exp = (int) num("n_ff_exp");
        g.w_scale = (float) num("expert_weights_scale");
        g.softcap = (float) num("final_logit_softcapping");
        g.n_swa = (int) num("n_swa");
        g.swa = ints("swa");
        g.d_rel = (int) num("d_rel");
        g.rel_extent = (int) num("rel_extent");
        g.conv = (int) num("shortconv_kernel", 4);
        g.log_floor = num("log_scaling_n_floor");
        g.log_alpha = (float) num("log_scaling_alpha");
        g.qk_clip = (float) num("qk_clip");
        g.rel_clip = (float) num("rel_clip");
        g.res_block = (int) num("attn_res_block", 6);
        g.situ_beta = (float) num("situ_beta");
        g.situ_lb = (float) num("situ_linear_beta");
        g.bos = (int) num("bos");
        g.eos = (int) num("eos");
        // (the kernels' limits: n_embd and n_ff_exp whole 256-value blocks - ggml's k-quant super-blocks -, head_dim and
        //  d_rel <= 256 - clip2, attention_split -, the 4-tap short convolution, n_used <= 16 and n_expert <= 1024 -
        //  hy4's router -, the depth mix's states <= kMaxStates 16)
        if (g.n_layer < 1 || g.n_embd < 256 || g.n_embd % 256 || g.n_ff_exp % 256 || (int) g.swa.size() != g.n_layer ||
            g.n_head_kv < 1 || g.n_head % g.n_head_kv || g.head_dim < 1 || g.head_dim > 256 || g.d_rel < 1 ||
            g.d_rel > 256 || g.conv != 4 || g.n_used < 1 || g.n_used > 16 || g.n_expert > 1024 || g.res_block < 1 ||
            g.n_swa < 1 || g.rel_extent < 1 || g.log_floor < 1 || g.situ_beta <= 0 ||
            (g.n_layer + g.res_block - 1) / g.res_block + 1 > 16) {
            err = dir + "/inkling2.txt: a geometry maya-inkling2 does not run";
            return false;
        }
        return true;
    }

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
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
    const uint8_t* data(const Tensor& x) const { return map[(size_t) x.file] + x.off; }
};

}  // namespace inkling2
