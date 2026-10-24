// src/ds4/ds4_ablate.hpp - abliteration on Maya: a refusal direction r (unit, [dim]) that the layers project out of
// every residual write - y <- y - scale * r (r . y) on the attention output and on the MoE output, right before each
// hc_post (ds4_model.cu's ablate()).  That is exactly what the rank-1 LoRA lora_from_dir.py builds for llama.cpp does
// (B A = -r (s r^T W) on attn_output_b, ffn_down_shexp and ffn_down_exps, which together ARE those two writes), so the
// two engines run the same abliterated model; here nothing of the pack changes and nothing is re-quantized.
//
// The direction comes from one of two GGUF files:
//   a llama-cvector-generator file   direction.<layer> F32 [dim]            (--ablate FILE --ablate-layer N)
//   a lora_from_dir.py adapter       blk.0.attn_output_b.weight.lora_b = -r (--ablate FILE; layer / scale from its
//                                    abliterate.* metadata)
// Only what those two need of GGUF is read here: the header, the key / values, the tensor infos, one F32 tensor.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace strata::ds4::ablate {

struct TensorInfo {
    std::vector<uint64_t> ne;
    uint32_t type = 0;   // GGML type (0 = F32)
    uint64_t off = 0;    // from the data section's start
};

struct Gguf {
    std::map<std::string, std::string> str_kv;
    std::map<std::string, double> num_kv;   // scalars (bool as 0 / 1)
    std::map<std::string, TensorInfo> tensors;
    uint64_t data = 0;   // the data section's start in the file
};

class Reader {
public:
    explicit Reader(const std::string& path) : f_(path, std::ios::binary) {}
    bool ok() const { return f_.good(); }
    template <class T>
    bool rd(T& v) {
        f_.read(reinterpret_cast<char*>(&v), sizeof(T));
        return f_.good();
    }
    bool rd_str(std::string& s) {
        uint64_t n = 0;
        if (!rd(n) || n > (1ull << 30)) return false;
        s.resize((size_t) n);
        if (n > 0) f_.read(&s[0], (std::streamsize) n);
        return f_.good();
    }
    uint64_t tell() { return (uint64_t) f_.tellg(); }
    bool seek(uint64_t p) {
        f_.seekg((std::streamoff) p, std::ios::beg);
        return f_.good();
    }
    bool rd_f32(float* out, size_t n) {
        f_.read(reinterpret_cast<char*>(out), (std::streamsize) (n * 4));
        return f_.good();
    }

private:
    std::ifstream f_;
};

// a scalar value of GGUF type t (0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 10 u64, 11 i64, 12 f64)
inline bool rd_scalar(Reader& r, uint32_t t, double& v) {
    switch (t) {
        case 0: { uint8_t x; if (!r.rd(x)) return false; v = x; return true; }
        case 1: { int8_t x; if (!r.rd(x)) return false; v = x; return true; }
        case 2: { uint16_t x; if (!r.rd(x)) return false; v = x; return true; }
        case 3: { int16_t x; if (!r.rd(x)) return false; v = x; return true; }
        case 4: { uint32_t x; if (!r.rd(x)) return false; v = x; return true; }
        case 5: { int32_t x; if (!r.rd(x)) return false; v = x; return true; }
        case 6: { float x; if (!r.rd(x)) return false; v = x; return true; }
        case 7: { uint8_t x; if (!r.rd(x)) return false; v = x != 0; return true; }
        case 10: { uint64_t x; if (!r.rd(x)) return false; v = (double) x; return true; }
        case 11: { int64_t x; if (!r.rd(x)) return false; v = (double) x; return true; }
        case 12: { double x; if (!r.rd(x)) return false; v = x; return true; }
        default: return false;
    }
}

inline bool read_gguf(const std::string& path, Gguf& g, std::string& err) {
    Reader r(path);
    if (!r.ok()) {
        err = path + ": cannot open";
        return false;
    }
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!r.rd(magic) || magic != 0x46554747u) {   // "GGUF"
        err = path + ": not a GGUF file";
        return false;
    }
    if (!r.rd(version) || version < 2 || version > 3 || !r.rd(n_tensors) || !r.rd(n_kv)) {
        err = path + ": GGUF header not understood";
        return false;
    }
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t t = 0;
        if (!r.rd_str(key) || !r.rd(t)) {
            err = path + ": key/value " + std::to_string(i) + " truncated";
            return false;
        }
        if (t == 8) {
            std::string s;
            if (!r.rd_str(s)) { err = path + ": " + key + " truncated"; return false; }
            g.str_kv[key] = s;
        } else if (t == 9) {   // array: element type, count, values (read past them; the direction files need none)
            uint32_t et = 0;
            uint64_t n = 0;
            if (!r.rd(et) || !r.rd(n)) { err = path + ": " + key + " truncated"; return false; }
            for (uint64_t k = 0; k < n; ++k) {
                if (et == 8) {
                    std::string s;
                    if (!r.rd_str(s)) { err = path + ": " + key + " truncated"; return false; }
                } else {
                    double v;
                    if (!rd_scalar(r, et, v)) { err = path + ": " + key + ": array type " + std::to_string(et); return false; }
                }
            }
        } else {
            double v = 0.0;
            if (!rd_scalar(r, t, v)) { err = path + ": " + key + ": value type " + std::to_string(t); return false; }
            g.num_kv[key] = v;
        }
    }
    for (uint64_t i = 0; i < n_tensors; ++i) {
        std::string name;
        uint32_t nd = 0;
        TensorInfo ti;
        if (!r.rd_str(name) || !r.rd(nd) || nd > 4) { err = path + ": tensor info " + std::to_string(i) + " truncated"; return false; }
        ti.ne.resize(nd);
        for (uint32_t d = 0; d < nd; ++d)
            if (!r.rd(ti.ne[d])) { err = path + ": tensor " + name + " truncated"; return false; }
        if (!r.rd(ti.type) || !r.rd(ti.off)) { err = path + ": tensor " + name + " truncated"; return false; }
        g.tensors[name] = ti;
    }
    uint64_t align = 32;
    auto a = g.num_kv.find("general.alignment");
    if (a != g.num_kv.end() && a->second >= 1) align = (uint64_t) a->second;
    const uint64_t pos = r.tell();
    g.data = (pos + align - 1) / align * align;
    return true;
}

/// The direction of `path` (see the file comment): r unit length; `scale` 0 takes the file's (an adapter's
/// abliterate.scale, else 1); `layer` 0 takes an adapter's own, or a one-layer control vector's.  `what` describes the
/// pick for the start-up log.
inline bool load_direction(const std::string& path, int layer, float& scale, std::vector<float>& r, std::string& what,
                           std::string& err) {
    Gguf g;
    if (!read_gguf(path, g, err)) return false;
    Reader rd(path);
    std::string tname;
    bool adapter = false;
    auto type = g.str_kv.find("general.type");
    if (type != g.str_kv.end() && type->second == "adapter") {
        adapter = true;
        tname = "blk.0.attn_output_b.weight.lora_b";
        auto t = g.tensors.find(tname);
        if (t == g.tensors.end()) {
            err = path + ": an adapter without " + tname + " (not a lora_from_dir.py abliteration adapter)";
            return false;
        }
        if (t->second.ne.size() != 2 || t->second.ne[0] != 1) {
            err = path + ": " + tname + " is not rank 1";
            return false;
        }
        auto l = g.num_kv.find("abliterate.direction_layer");
        if (l != g.num_kv.end()) layer = (int) l->second;
        if (scale == 0.0f) {
            auto s = g.num_kv.find("abliterate.scale");
            scale = s != g.num_kv.end() ? (float) s->second : 1.0f;
        }
    } else {
        auto lc = g.num_kv.find("controlvector.layer_count");
        const int n_layers = lc != g.num_kv.end() ? (int) lc->second : 0;
        if (layer <= 0) {
            if (n_layers == 1) layer = 1;
            else {
                err = path + ": a control vector with " + std::to_string(n_layers) +
                      " directions: say which with --ablate-layer N (1-based)";
                return false;
            }
        }
        tname = "direction." + std::to_string(layer);
        if (g.tensors.find(tname) == g.tensors.end()) {
            err = path + ": no " + tname + (n_layers > 0 ? " (layers 1.." + std::to_string(n_layers) + ")" : "");
            return false;
        }
        if (scale == 0.0f) scale = 1.0f;
    }
    if (!std::isfinite(scale) || scale <= 0.0f) {   // a junk --ablate-scale (atof -> 0/nan/inf) must not reach the GPU
        err = path + ": ablation scale is not a positive finite number";
        return false;
    }
    const TensorInfo& ti = g.tensors[tname];
    if (ti.type != 0) {
        err = path + ": " + tname + " is not F32";
        return false;
    }
    uint64_t n = 1;
    for (uint64_t d : ti.ne) {
        if (d == 0 || d > (1u << 20) || n > (1u << 20) / d) {   // bound each dim and the running product first
            err = path + ": " + tname + " has an odd size";
            return false;
        }
        n *= d;
    }
    if (n == 0 || n > (1u << 20)) {
        err = path + ": " + tname + " has an odd size";
        return false;
    }
    r.assign((size_t) n, 0.0f);
    if (!rd.seek(g.data + ti.off) || !rd.rd_f32(r.data(), (size_t) n)) {
        err = path + ": " + tname + ": read failed";
        return false;
    }
    double ss = 0.0;
    for (float v : r) ss += (double) v * v;
    if (!(ss > 0.0) || !std::isfinite(ss)) {
        err = path + ": " + tname + " is zero or not finite";
        return false;
    }
    const double inv = 1.0 / std::sqrt(ss);
    if (!std::isfinite(inv)) {   // a tiny-but-finite norm would overflow a float reciprocal; normalize in double
        err = path + ": " + tname + " cannot be normalized (norm too small)";
        return false;
    }
    // an adapter's lora_b is -r; the projection r (r . y) does not care about the sign, r is kept as the direction
    for (float& v : r) v = (float) (adapter ? -(double) v * inv : (double) v * inv);
    char buf[200];
    if (adapter)
        std::snprintf(buf, sizeof buf, "adapter (direction layer %d), scale %.3g", layer, scale);
    else
        std::snprintf(buf, sizeof buf, "%s, scale %.3g", tname.c_str(), scale);
    what = "ds4: abliteration: " + path + " " + buf + ", projected out of every layer's attention and MoE writes";
    return true;
}

}  // namespace strata::ds4::ablate
