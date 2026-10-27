// src/mm3/mm3_pack.hpp - maya-mm3's pack (tools/mm3/pack.py): the geometry (mm3.txt) and every tensor of the model's
// safetensors read in place (tensors.txt), each shard mapped once.
//
//   mm3.txt      `key value` lines; lists comma-separated
//   tensors.txt  `file <i> <path>` (relative to the model directory, model_dir in mm3.txt), then
//                `t <name> <dtype> <file> <offset> <bytes> <dims...>`
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mm3_os.hpp"

namespace mm3 {

struct Tensor {
    std::string name, dtype;
    int file = -1;
    uint64_t off = 0, bytes = 0;
    std::vector<int64_t> shape;
};

struct Pack {
    std::string dir, model_dir;
    std::map<std::string, std::string> kv;
    std::map<std::string, Tensor> t;
    std::vector<std::string> files;
    std::vector<const uint8_t*> map;
    std::vector<size_t> map_bytes;

    bool load(const std::string& pack_dir, std::string& err) {
        dir = pack_dir;
        std::ifstream g(dir + "/mm3.txt");
        if (!g) {
            err = dir + "/mm3.txt: cannot open (tools/mm3/pack.py --model <model> writes it)";
            return false;
        }
        for (std::string line; std::getline(g, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();   // (a pack written on Windows in text mode)
            const size_t sp = line.find(' ');
            if (sp != std::string::npos) kv[line.substr(0, sp)] = line.substr(sp + 1);
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
                is >> x.name >> x.dtype >> x.file >> x.off >> x.bytes;
                for (int64_t d; is >> d;) x.shape.push_back(d);
                t[x.name] = std::move(x);
            }
        }
        map.assign(files.size(), nullptr);
        map_bytes.assign(files.size(), 0);
        for (size_t i = 0; i < files.size(); ++i) {
            const std::string p = model_dir + "/" + files[i];
            size_t sz = 0;
            const uint8_t* m = os::map_file(p, sz);
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
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
    const uint8_t* data(const Tensor& x) const { return map[(size_t) x.file] + x.off; }
};

}  // namespace mm3
