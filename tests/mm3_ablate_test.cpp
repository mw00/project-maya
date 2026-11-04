// tests/mm3_ablate_test.cpp - mm3_ablate.hpp's load on synthetic GGUFs (no model, no GPU).
//
// What maya-mm3 --ablate accepts, and what it must refuse before anything reaches the GPU:
//   * a control vector: one direction found on its own, several needing --ablate-layer, r returned unit length, the
//     scale 1 unless given;
//   * a one-direction file whose tensor is not direction.1 (direction.37): found by its name;
//   * refused: a direction whose size is not the model's hidden size, or of the right count but not one row ([2, 4]),
//     an abliteration adapter (maya-ds4's), a file that is not a control vector, a general.alignment that is not a
//     power of two (UINT64_MAX wraps the data section's start into the header), a missing direction.N, a missing or
//     malformed file.
//
// The fixtures are minimal GGUF v3 files written here (32-byte alignment, F32 tensors), as in ds4_ablate_test.cpp.
#include "../src/mm3/mm3_ablate.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

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
    uint32_t type;   // 4 u32, 8 string, 10 u64
    uint64_t u = 0;
    std::string s;
};
Kv kv_u32(const std::string& k, uint32_t v) { return {k, 4, v, ""}; }
Kv kv_str(const std::string& k, const std::string& v) { return {k, 8, 0, v}; }
Kv kv_u64(const std::string& k, uint64_t v) { return {k, 10, v, ""}; }

struct T {
    std::string name;
    std::vector<float> v;          // F32
    std::vector<uint64_t> shape;   // (empty: [v.size()])
    T(std::string n, std::vector<float> x, std::vector<uint64_t> sh = {})
        : name(std::move(n)), v(std::move(x)), shape(std::move(sh)) {}
};

constexpr int kDim = 8;

// cut: bytes off the end; header_only: the file ends with the tensor list (no data section)
std::filesystem::path write(const char* tag, const std::vector<Kv>& kvs, const std::vector<T>& ts, size_t cut = 0,
                            bool header_only = false) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);
    put<uint32_t>(b, 3);
    put<uint64_t>(b, ts.size());
    put<uint64_t>(b, kvs.size());
    for (const Kv& k : kvs) {
        put_str(b, k.key);
        put<uint32_t>(b, k.type);
        if (k.type == 4) put<uint32_t>(b, (uint32_t) k.u);
        else if (k.type == 10) put<uint64_t>(b, k.u);
        else put_str(b, k.s);
    }
    std::vector<uint64_t> offs;
    uint64_t off = 0;
    for (const T& t : ts) {
        put_str(b, t.name);
        const std::vector<uint64_t> shape = t.shape.empty() ? std::vector<uint64_t>{t.v.size()} : t.shape;
        put<uint32_t>(b, (uint32_t) shape.size());
        for (uint64_t d : shape) put<uint64_t>(b, d);
        put<uint32_t>(b, 0);
        put<uint64_t>(b, off);
        offs.push_back(off);
        off += (t.v.size() * 4 + 31) / 32 * 32;
    }
    if (!header_only) {
        const size_t data = (b.size() + 31) / 32 * 32;
        b.resize(data + off, 0);
        for (size_t i = 0; i < ts.size(); ++i)
            std::memcpy(b.data() + data + offs[i], ts[i].v.data(), ts[i].v.size() * 4);
    }
    if (cut) b.resize(b.size() - cut);
    const auto path = std::filesystem::temp_directory_path() / (std::string("strata_mm3_ablate_") + tag + ".gguf");
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
    return path;
}

std::vector<float> dir(float base, int n = kDim) {
    std::vector<float> v((size_t) n);
    for (int i = 0; i < n; ++i) v[(size_t) i] = base + (float) i;
    return v;
}
std::vector<Kv> cvec_kv(uint32_t layers) {
    return {kv_str("general.architecture", "controlvector"), kv_u32("controlvector.layer_count", layers)};
}

bool unit_along(const std::vector<float>& r, const std::vector<float>& d) {
    if (r.size() != d.size()) return false;
    double nd = 0, dot = 0, nr = 0;
    for (size_t i = 0; i < d.size(); ++i) nd += (double) d[i] * d[i], dot += (double) r[i] * d[i], nr += (double) r[i] * r[i];
    return std::fabs(nr - 1.0) < 1e-6 && std::fabs(dot / std::sqrt(nd) - 1.0) < 1e-6;
}

bool loads(const std::filesystem::path& p, int layer, float scale_in, std::vector<float>* r_out = nullptr,
           float* scale_out = nullptr, std::string* what_out = nullptr) {
    std::vector<float> r;
    std::string what, err;
    float s = scale_in;
    const bool ok = mm3::ablate::load(p.string(), layer, kDim, s, r, what, err);
    if (!ok) std::printf("      (%s)\n", err.c_str());
    if (r_out) *r_out = r;
    if (scale_out) *scale_out = s;
    if (what_out) *what_out = what;
    return ok;
}
}  // namespace

int main() {
    std::printf("mm3_ablate_test\n");
    std::vector<float> r;
    float s = 0.0f;
    std::string what;

    const auto one = write("one", cvec_kv(1), {{"direction.1", dir(1.0f)}});
    check(loads(one, 0, 0.0f, &r, &s, &what) && unit_along(r, dir(1.0f)) && s == 1.0f,
          "a one-direction control vector: found without --ablate-layer, unit, scale 1");
    check(what.find("direction.1, scale 1") != std::string::npos, "the start-up line names the pick");
    check(loads(one, 0, 2.5f, nullptr, &s) && s == 2.5f, "--ablate-scale replaces the default");

    const auto many = write("many", cvec_kv(3), {{"direction.1", dir(1.0f)}, {"direction.2", dir(5.0f)},
                                                 {"direction.3", dir(-3.0f)}});
    check(!loads(many, 0, 0.0f), "several directions without --ablate-layer: refused");
    check(loads(many, 2, 0.0f, &r) && unit_along(r, dir(5.0f)), "--ablate-layer 2 takes direction.2");
    check(!loads(many, 4, 0.0f), "--ablate-layer past the file's directions: refused");

    const auto wide = write("wide", cvec_kv(1), {{"direction.1", dir(1.0f, kDim * 2)}});
    check(!loads(wide, 0, 0.0f), "a direction of another hidden size: refused");
    const auto rows = write("rows", cvec_kv(1), {{"direction.1", dir(1.0f), {2, kDim / 2}}});
    check(!loads(rows, 0, 0.0f) && !loads(rows, 1, 0.0f), "the hidden size's count of values as [2, 4]: refused");

    const auto l37 = write("l37", cvec_kv(1), {{"direction.37", dir(2.0f)}});
    check(loads(l37, 0, 0.0f, &r, nullptr, &what) && unit_along(r, dir(2.0f)) &&
              what.find("direction.37,") != std::string::npos,
          "a one-direction file of direction.37: found without --ablate-layer");
    check(loads(l37, 37, 0.0f) && !loads(l37, 1, 0.0f), "... and with --ablate-layer 37 (1: refused)");

    const auto llama = write("llama", {kv_str("general.architecture", "llama"), kv_u32("controlvector.layer_count", 1)},
                             {{"direction.1", dir(1.0f)}});
    check(!loads(llama, 0, 0.0f), "general.architecture not controlvector: refused");

    std::vector<Kv> wrap = cvec_kv(1);
    wrap.push_back(kv_u64("general.alignment", UINT64_MAX));
    const auto align = write("align", wrap, {{"direction.1", dir(1.0f)}}, 0, true);
    check(!loads(align, 0, 0.0f), "general.alignment UINT64_MAX, no data section (the header as r): refused");

    const auto zero = write("zero", cvec_kv(1), {{"direction.1", std::vector<float>(kDim, 0.0f)}});
    check(!loads(zero, 0, 0.0f), "a zero direction: refused");

    const auto adapter = write("adapter", {kv_str("general.type", "adapter"), kv_str("adapter.type", "lora")},
                               {{"blk.0.attn_output_b.weight.lora_b", dir(1.0f)}});
    check(!loads(adapter, 0, 0.0f), "an abliteration adapter (maya-ds4's): refused");

    const auto cut = write("cut", cvec_kv(1), {{"direction.1", dir(1.0f)}}, 16);
    check(!loads(cut, 0, 0.0f), "a truncated file: refused");
    check(!loads(std::filesystem::temp_directory_path() / "strata_mm3_ablate_missing.gguf", 0, 0.0f),
          "a missing file: refused");

    for (const auto& p : {one, many, wide, rows, l37, llama, align, zero, adapter, cut}) {
        std::error_code ec;
        std::filesystem::remove(p, ec);
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
