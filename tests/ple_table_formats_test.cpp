// Synthetic GGUF Engram tables in every supported encoding, through mmap and direct I/O.
// The pinned ggml CPU dequantizer is the independent value reference.
#include "strata/kernels/ngram.hpp"

#include "ggml.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
namespace k = strata::kernels;

constexpr int kRows = 64;

template <typename T> void put(std::vector<uint8_t>& b, T v) {
    const size_t n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}

void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

bool near(float got, float want) {
    return std::isfinite(got) && std::isfinite(want) && std::fabs(got - want) <=
           1e-6f * (1.0f + std::fabs(want));
}

bool check_values(const float* got, const float* want, size_t n, int type, const char* stage) {
    for (size_t i = 0; i < n; ++i) if (!near(got[i], want[i])) {
        std::fprintf(stderr, "type %d %s element %zu: got %.9g, ggml %.9g\n", type, stage, i, got[i], want[i]);
        return false;
    }
    return true;
}

void fill_rows(int type, size_t row_bytes, std::vector<uint8_t>& raw) {
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = (uint8_t) ((i * 73 + i / 13 + type * 17) & 255);
    if (type == 30) {
        for (size_t i = 0; i < raw.size() / 2; ++i) {
            const float v = (float) ((int) (i % 101) - 50) / 64.0f;
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            const uint16_t bf = (uint16_t) (bits >> 16);
            std::memcpy(raw.data() + 2 * i, &bf, 2);
        }
        return;
    }
    const int block_bytes = type == 20 ? 18 : type == 7 ? 24 : 34;
    for (int r = 0; r < kRows; ++r) for (int b = 0; b < k::PLE_HEAD_DIM / 32; ++b) {
        uint8_t* p = raw.data() + (size_t) r * row_bytes + b * block_bytes;
        const uint16_t d = 0x3000;                 // 0.125 in FP16
        std::memcpy(p, &d, 2);
        if (type == 7) {
            const uint16_t m = 0xB800;             // -0.5 in FP16
            std::memcpy(p + 2, &m, 2);
        }
    }
}

fs::path write_gguf(int type, const std::vector<uint8_t>& raw) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 1);                            // one tensor
    put<uint64_t>(b, 0);                            // no metadata
    put_str(b, "per_layer_token_embd.weight");
    put<uint32_t>(b, 2);
    put<uint64_t>(b, k::PLE_HEAD_DIM);
    put<uint64_t>(b, kRows);
    put<uint32_t>(b, (uint32_t) type);
    put<uint64_t>(b, 0);                            // offset within the data section
    b.resize((b.size() + 31) / 32 * 32);
    b.insert(b.end(), raw.begin(), raw.end());
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path path = fs::temp_directory_path() /
                          ("maya_ple_formats_" + std::to_string(type) + "_" + std::to_string(id) + ".gguf");
    std::ofstream f(path, std::ios::binary);
    f.write((const char*) b.data(), (std::streamsize) b.size());
    return path;
}

bool run(int type) {
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    if (!tr || !tr->to_float) return false;
    const size_t row_bytes = (size_t) k::PLE_HEAD_DIM / ggml_blck_size((ggml_type) type) *
                             ggml_type_size((ggml_type) type);
    std::vector<uint8_t> raw(row_bytes * kRows);
    fill_rows(type, row_bytes, raw);
    std::vector<float> oracle((size_t) kRows * k::PLE_HEAD_DIM);
    for (int r = 0; r < kRows; ++r)
        tr->to_float(raw.data() + (size_t) r * row_bytes,
                     oracle.data() + (size_t) r * k::PLE_HEAD_DIM, k::PLE_HEAD_DIM);
    const fs::path path = write_gguf(type, raw);
    bool ok = true;
    for (const k::PleIo mode : {k::PleIo::Mmap, k::PleIo::Direct}) {
        k::PleTable table;
        k::PleIoOptions opts;
        opts.mode = mode;
        opts.cache_rows = 64;
        opts.max_inflight = 8;
        std::string err;
        if (!table.open(path.string(), err, opts)) {
            std::fprintf(stderr, "type %d %s open: %s\n", type, mode == k::PleIo::Direct ? "direct" : "mmap", err.c_str());
            ok = false;
            continue;
        }
        std::array<float, k::PLE_HEAD_DIM> row{};
        for (uint32_t r : {0u, 1u, 17u, 63u}) {
            table.read_row(r, row.data());
            ok = check_values(row.data(), oracle.data() + (size_t) r * k::PLE_HEAD_DIM,
                              k::PLE_HEAD_DIM, type, "single row") && ok;
        }
        table.read_row(kRows, row.data());
        for (float v : row) if (v != 0.0f) ok = false;

        std::array<uint32_t, 2 * k::PLE_N_HEADS> ids{};
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = (uint32_t) ((i * 19 + 3) % kRows);
        std::vector<float> want(ids.size() * k::PLE_HEAD_DIM);
        for (size_t i = 0; i < ids.size(); ++i)
            std::memcpy(want.data() + i * k::PLE_HEAD_DIM,
                        oracle.data() + (size_t) ids[i] * k::PLE_HEAD_DIM,
                        k::PLE_HEAD_DIM * sizeof(float));
        std::vector<float> out(want.size());
        if (!table.issue(ids.data()) || !table.collect(out.data(), err)) {
            std::fprintf(stderr, "type %d issue/collect: %s\n", type, err.c_str()); ok = false;
        } else ok = check_values(out.data(), want.data(), k::NG_N_EMBD, type, "token") && ok;
        if (!table.gather_batch(ids.data(), 2, out.data(), err)) {
            std::fprintf(stderr, "type %d batch: %s\n", type, err.c_str()); ok = false;
        } else ok = check_values(out.data(), want.data(), want.size(), type, "batch") && ok;
    }
    fs::remove(path);
    std::printf("  Engram type %2d: %s\n", type, ok ? "OK" : "FAILED");
    return ok;
}
}  // namespace

int main() {
    bool ok = true;
    for (int type : {20, 7, 8, 30}) ok = run(type) && ok;
    return ok ? 0 : 1;
}
