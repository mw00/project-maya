// include/strata/artifact/safetensors_dir.hpp - a Hugging Face checkpoint directory read in place: config.json, the
// shard index (model.safetensors.index.json, or one model.safetensors) and every shard's tensor table, the shards
// memory-mapped.  A small JSON reader for those files (objects, arrays, strings, numbers, true / false / null).
#pragma once

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace strata {

struct Json {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json* get(const std::string& key) const {
        if (kind != Obj) return nullptr;
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    /// a dotted path ("text_config.hidden_size")
    const Json* path(const std::string& p) const {
        const Json* j = this;
        size_t i = 0;
        while (j && i <= p.size()) {
            const size_t e = p.find('.', i);
            j = j->get(p.substr(i, e == std::string::npos ? std::string::npos : e - i));
            if (e == std::string::npos) break;
            i = e + 1;
        }
        return j;
    }

    static Json parse(const std::string& s) {
        size_t i = 0;
        Json j = value(s, i);
        return j;
    }

private:
    static void ws(const std::string& s, size_t& i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
    }
    static std::string string_at(const std::string& s, size_t& i) {
        if (s[i] != '"') throw std::runtime_error("json: expected a string");
        ++i;
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                const char c = s[++i];
                if (c == 'n') out += '\n';
                else if (c == 't') out += '\t';
                else if (c == 'u') { out += '?'; i += 4; }   // (names and numbers only matter here)
                else out += c;
                ++i;
            } else {
                out += s[i++];
            }
        }
        ++i;
        return out;
    }
    static Json value(const std::string& s, size_t& i) {
        ws(s, i);
        if (i >= s.size()) throw std::runtime_error("json: unexpected end");
        Json j;
        const char c = s[i];
        if (c == '{') {
            j.kind = Obj;
            ++i;
            ws(s, i);
            if (s[i] == '}') { ++i; return j; }
            for (;;) {
                ws(s, i);
                std::string k = string_at(s, i);
                ws(s, i);
                if (s[i] != ':') throw std::runtime_error("json: expected ':'");
                ++i;
                j.obj.emplace_back(std::move(k), value(s, i));
                ws(s, i);
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == '}') { ++i; break; }
                throw std::runtime_error("json: expected ',' or '}'");
            }
        } else if (c == '[') {
            j.kind = Arr;
            ++i;
            ws(s, i);
            if (s[i] == ']') { ++i; return j; }
            for (;;) {
                j.arr.push_back(value(s, i));
                ws(s, i);
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == ']') { ++i; break; }
                throw std::runtime_error("json: expected ',' or ']'");
            }
        } else if (c == '"') {
            j.kind = Str;
            j.str = string_at(s, i);
        } else if (s.compare(i, 4, "true") == 0) {
            j.kind = Bool; j.b = true; i += 4;
        } else if (s.compare(i, 5, "false") == 0) {
            j.kind = Bool; i += 5;
        } else if (s.compare(i, 4, "null") == 0) {
            i += 4;
        } else {
            j.kind = Num;
            size_t e = i;
            while (e < s.size() && (std::strchr("+-0123456789.eE", s[e]) != nullptr)) ++e;
            j.num = std::stod(s.substr(i, e - i));
            i = e;
        }
        return j;
    }
};

inline std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct StTensor {
    std::string dtype;             // "BF16", "F16", "F32", "I16", "I32", ...
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr; // in the mapped shard
    uint64_t bytes = 0;
};

class SafetensorsDir {
public:
    explicit SafetensorsDir(const std::string& dir) : dir_(dir) {
        config_ = Json::parse(read_text(dir + "/config.json"));
        std::vector<std::string> shards;
        struct stat sb {};
        if (::stat((dir + "/model.safetensors.index.json").c_str(), &sb) == 0) {
            const Json idx = Json::parse(read_text(dir + "/model.safetensors.index.json"));
            const Json* wm = idx.get("weight_map");
            if (!wm) throw std::runtime_error(dir + ": index without weight_map");
            for (const auto& kv : wm->obj) {
                bool seen = false;
                for (const auto& s : shards) seen = seen || s == kv.second.str;
                if (!seen) shards.push_back(kv.second.str);
            }
        } else {
            shards.push_back("model.safetensors");
        }
        for (const auto& s : shards) map_shard(dir + "/" + s);
    }
    ~SafetensorsDir() {
        for (const auto& m : maps_) ::munmap(m.first, m.second);
    }
    SafetensorsDir(const SafetensorsDir&) = delete;
    SafetensorsDir& operator=(const SafetensorsDir&) = delete;

    const Json& config() const { return config_; }
    const StTensor* find(const std::string& name) const {
        const auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    const std::map<std::string, StTensor>& tensors() const { return tensors_; }
    const std::string& dir() const { return dir_; }

private:
    void map_shard(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat sb {};
        ::fstat(fd, &sb);
        void* p = ::mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("cannot map " + path);
        maps_.emplace_back(p, (size_t) sb.st_size);
        const uint8_t* base = (const uint8_t*) p;
        uint64_t hl = 0;
        std::memcpy(&hl, base, 8);
        const Json h = Json::parse(std::string((const char*) base + 8, (size_t) hl));
        const uint8_t* data0 = base + 8 + hl;
        for (const auto& kv : h.obj) {
            if (kv.first == "__metadata__") continue;
            StTensor t;
            const Json* dt = kv.second.get("dtype");
            const Json* sh = kv.second.get("shape");
            const Json* off = kv.second.get("data_offsets");
            if (!dt || !sh || !off || off->arr.size() != 2) throw std::runtime_error(path + ": bad entry " + kv.first);
            t.dtype = dt->str;
            for (const auto& d : sh->arr) t.shape.push_back((int64_t) d.num);
            const uint64_t a = (uint64_t) off->arr[0].num, b = (uint64_t) off->arr[1].num;
            if (8 + hl + b > (uint64_t) sb.st_size) throw std::runtime_error(path + ": truncated at " + kv.first);
            t.data = data0 + a;
            t.bytes = b - a;
            tensors_[kv.first] = t;
        }
    }

    std::string dir_;
    Json config_;
    std::map<std::string, StTensor> tensors_;
    std::vector<std::pair<void*, size_t>> maps_;
};

}  // namespace strata
