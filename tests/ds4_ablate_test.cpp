// tests/ds4_ablate_test.cpp - ds4_ablate.hpp's load_direction on synthetic GGUFs (no model, no GPU).
//
// What --ablate accepts, and what it must refuse before anything reaches the GPU:
//   * a control vector: one direction found on its own, several needing --ablate-layer, r returned unit length;
//   * a whole-model rank-1 abliteration adapter: r = -lora_b (unit), scale = abliterate.scale x adapter.lora.alpha;
//   * refused: an adapter missing a writer (no routed experts) or a block, one for a deeper model, one whose
//     writers carry different directions; a tensor offset that wraps around to bytes elsewhere in the file;
//     out-of-range metadata; a zero direction; a non-F32 direction; a truncated file; a bad caller scale.
//
// The fixtures are minimal GGUF v3 files written here (32-byte alignment, F32 tensors).
#include "../src/ds4/ds4_ablate.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {
using strata::ds4::ablate::load_direction;

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

struct Kv {
    std::string key;
    uint32_t type;   // 4 u32, 5 i32, 6 f32, 8 string
    uint64_t u = 0;
    float f = 0.0f;
    std::string s;
};
Kv kv_u32(const std::string& k, uint32_t v) { return {k, 4, v, 0.0f, ""}; }
Kv kv_i32(const std::string& k, int32_t v) { return {k, 5, (uint64_t) (uint32_t) v, 0.0f, ""}; }
Kv kv_f32(const std::string& k, float v) { return {k, 6, 0, v, ""}; }
Kv kv_str(const std::string& k, const std::string& v) { return {k, 8, 0, 0.0f, v}; }

struct T {
    std::string name;
    std::vector<uint64_t> ne;
    std::vector<float> v;   // F32 payload (its size is the element count)
    uint32_t type = 0;
    uint64_t force_off = UINT64_MAX;   // a forged offset, else the next aligned slot
};

constexpr int kDim = 8;

std::filesystem::path write(const char* tag, const std::vector<Kv>& kvs, const std::vector<T>& ts, size_t cut = 0) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);
    put<uint32_t>(b, 3);
    put<uint64_t>(b, ts.size());
    put<uint64_t>(b, kvs.size());
    for (const Kv& k : kvs) {
        put_str(b, k.key);
        put<uint32_t>(b, k.type);
        if (k.type == 4) put<uint32_t>(b, (uint32_t) k.u);
        else if (k.type == 5) put<int32_t>(b, (int32_t) (uint32_t) k.u);
        else if (k.type == 6) put<float>(b, k.f);
        else put_str(b, k.s);
    }
    std::vector<uint64_t> offs;
    uint64_t off = 0;
    for (const T& t : ts) {
        put_str(b, t.name);
        put<uint32_t>(b, (uint32_t) t.ne.size());
        for (uint64_t d : t.ne) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, t.force_off != UINT64_MAX ? t.force_off : off);
        offs.push_back(off);
        off += (t.v.size() * 4 + 31) / 32 * 32;
    }
    const size_t data = (b.size() + 31) / 32 * 32;
    b.resize(data + off, 0);
    for (size_t i = 0; i < ts.size(); ++i) std::memcpy(b.data() + data + offs[i], ts[i].v.data(), ts[i].v.size() * 4);
    if (cut) b.resize(b.size() - cut);
    const auto path = std::filesystem::temp_directory_path() / (std::string("strata_ds4_ablate_") + tag + ".gguf");
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
    return path;
}

std::vector<float> dir(float base) {
    std::vector<float> v(kDim);
    for (int i = 0; i < kDim; ++i) v[i] = base + (float) i;
    return v;
}

// an abliteration adapter: blocks [0, n_blk), lora_b = -d on every writer (`skip` one, `odd` one differs)
std::vector<T> adapter_tensors(int n_blk, const std::string& skip = "", const std::string& odd = "") {
    std::vector<float> nb = dir(1.0f);
    for (float& x : nb) x = -x;
    std::vector<T> ts;
    for (int il = 0; il < n_blk; ++il)
        for (const char* w : {"attn_output_b", "ffn_down_shexp", "ffn_down_exps"}) {
            const std::string base = "blk." + std::to_string(il) + "." + w + ".weight";
            if (base == skip) continue;
            const bool exps = std::strcmp(w, "ffn_down_exps") == 0;
            const uint64_t n_exp = exps ? 2 : 1;
            std::vector<float> b;
            for (uint64_t e = 0; e < n_exp; ++e) b.insert(b.end(), nb.begin(), nb.end());
            if (base == odd) b[0] += 1.0f;
            std::vector<uint64_t> a_ne{4, 1}, b_ne{1, kDim};
            if (exps) a_ne.push_back(n_exp), b_ne.push_back(n_exp);
            ts.push_back({base + ".lora_a", a_ne, std::vector<float>(4 * n_exp, 0.5f)});
            ts.push_back({base + ".lora_b", b_ne, b});
        }
    return ts;
}
std::vector<Kv> adapter_kv(float scale = 0.5f, float alpha = 2.0f) {
    return {kv_str("general.type", "adapter"), kv_str("adapter.type", "lora"), kv_f32("adapter.lora.alpha", alpha),
            kv_u32("abliterate.direction_layer", 22), kv_f32("abliterate.scale", scale)};
}

bool unit_along(const std::vector<float>& r, const std::vector<float>& d) {
    if (r.size() != d.size()) return false;
    double nd = 0, dot = 0, nr = 0;
    for (size_t i = 0; i < d.size(); ++i) nd += (double) d[i] * d[i], dot += (double) r[i] * d[i], nr += (double) r[i] * r[i];
    return std::fabs(nr - 1.0) < 1e-6 && std::fabs(dot / std::sqrt(nd) - 1.0) < 1e-6;
}

bool loads(const std::filesystem::path& p, int layer, int n_layer, float scale_in, std::vector<float>* r_out = nullptr,
           float* scale_out = nullptr) {
    std::vector<float> r;
    std::string what, err;
    float s = scale_in;
    const bool ok = load_direction(p.string(), layer, n_layer, s, r, what, err);
    if (!ok) std::printf("      (%s)\n", err.c_str());
    if (r_out) *r_out = r;
    if (scale_out) *scale_out = s;
    return ok;
}
}  // namespace

int main() {
    std::vector<std::filesystem::path> made;
    auto mk = [&](std::filesystem::path p) { made.push_back(p); return p; };
    std::vector<float> r;
    float s = 0.0f;

    std::printf("control vectors\n");
    {
        const auto one = mk(write("cv1", {kv_i32("controlvector.layer_count", 1)}, {{"direction.1", {kDim}, dir(1.0f)}}));
        check(loads(one, 0, 2, 0.0f, &r, &s) && unit_along(r, dir(1.0f)) && s == 1.0f, "one direction: found, unit, scale 1");
        const auto three = mk(write("cv3", {kv_i32("controlvector.layer_count", 3)},
                                    {{"direction.1", {kDim}, dir(1.0f)}, {"direction.2", {kDim}, dir(5.0f)},
                                     {"direction.3", {kDim}, dir(9.0f)}}));
        check(!loads(three, 0, 2, 0.0f), "three directions without --ablate-layer: refused");
        check(loads(three, 2, 2, 0.0f, &r) && unit_along(r, dir(5.0f)), "--ablate-layer 2 picks direction.2");
        check(!loads(three, 4, 2, 0.0f), "--ablate-layer past the file: refused");
        loads(three, 2, 2, 0.25f, nullptr, &s);
        check(s == 0.25f, "a caller scale overrides the default");
        for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -1.0f})
            check(!loads(three, 2, 2, bad), "a NaN / inf / negative caller scale: refused");
        const auto neg = mk(write("cvneg", {kv_i32("controlvector.layer_count", -5)}, {{"direction.1", {kDim}, dir(1.0f)}}));
        check(!loads(neg, 1, 2, 0.0f), "a negative controlvector.layer_count: refused");
        const auto zero = mk(write("cvzero", {}, {{"direction.1", {kDim}, std::vector<float>(kDim, 0.0f)}}));
        check(!loads(zero, 1, 2, 0.0f), "a zero direction: refused");
        T f16{"direction.1", {kDim}, dir(1.0f)};
        f16.type = 1;
        check(!loads(mk(write("cvf16", {}, {f16})), 1, 2, 0.0f), "a non-F32 direction: refused");
        check(!loads(mk(write("cvcut", {}, {{"direction.1", {kDim}, dir(1.0f)}}, 20)), 1, 2, 0.0f),
              "a file cut short inside the tensor: refused");
        // an offset that wraps data_start + off back to the start of the file, where a forged direction could sit
        T wrap{"direction.1", {kDim}, dir(1.0f)};
        wrap.force_off = UINT64_MAX - 31;
        check(!loads(mk(write("cvwrap", {}, {wrap})), 1, 2, 0.0f), "a tensor offset that wraps around: refused");
        T past{"direction.1", {kDim}, dir(1.0f)};
        past.force_off = 1u << 20;
        check(!loads(mk(write("cvpast", {}, {past})), 1, 2, 0.0f), "a tensor offset past the end: refused");
        check(!loads("/nonexistent/strata_ds4_ablate.gguf", 1, 2, 0.0f), "a missing file: refused");
        std::vector<uint8_t> junk(64, 0x41);
        const auto jp = mk(std::filesystem::temp_directory_path() / "strata_ds4_ablate_junk.gguf");
        std::ofstream(jp, std::ios::binary).write(reinterpret_cast<const char*>(junk.data()), (std::streamsize) junk.size());
        check(!loads(jp, 1, 2, 0.0f), "not a GGUF: refused");
    }

    std::printf("adapters (2-block model)\n");
    {
        std::vector<float> d = dir(1.0f);
        const auto whole = mk(write("ad", adapter_kv(), adapter_tensors(2)));
        check(loads(whole, 0, 2, 0.0f, &r, &s) && unit_along(r, d), "whole-model adapter: r = -lora_b, unit");
        check(s == 1.0f, "scale = abliterate.scale 0.5 x adapter.lora.alpha 2");
        loads(whole, 0, 2, 0.3f, nullptr, &s);
        check(s == 0.3f, "a caller scale overrides the adapter's");
        check(!loads(mk(write("adnoexps", adapter_kv(), adapter_tensors(2, "blk.1.ffn_down_exps.weight"))), 0, 2, 0.0f),
              "a routed writer missing: refused");
        check(!loads(mk(write("adshort", adapter_kv(), adapter_tensors(1))), 0, 2, 0.0f),
              "a block missing: refused");
        check(!loads(mk(write("addeep", adapter_kv(), adapter_tensors(3))), 0, 2, 0.0f),
              "an adapter for a deeper model: refused");
        check(!loads(mk(write("adodd", adapter_kv(), adapter_tensors(2, "", "blk.1.ffn_down_shexp.weight"))), 0, 2, 0.0f),
              "a writer with a different direction: refused");
        std::vector<Kv> kv = adapter_kv();
        kv[3] = kv_u32("abliterate.direction_layer", 4000000000u);
        check(!loads(mk(write("adlayer", kv, adapter_tensors(2))), 0, 2, 0.0f),
              "abliterate.direction_layer out of int range: refused");
        check(!loads(mk(write("adscale", adapter_kv(-1.0f), adapter_tensors(2))), 0, 2, 0.0f),
              "a negative abliterate.scale: refused");
    }

    for (const auto& p : made) {
        std::error_code ec;
        std::filesystem::remove(p, ec);
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
