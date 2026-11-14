// A split Qwen GGUF may describe its embedded MTP only in shard 0. Both header
// discovery and upload must exclude that block in later shards, while still
// rejecting a genuinely missing canonical trunk matrix.
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
int fails = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "OK" : "FAIL", what.c_str());
    if (!ok) ++fails;
}
template <class T> void put(std::vector<uint8_t>& b, T v) {
    const size_t n = b.size(); b.resize(n + sizeof(T)); std::memcpy(b.data() + n, &v, sizeof(T));
}
void str(std::vector<uint8_t>& b, const std::string& v) {
    put<uint64_t>(b, v.size()); b.insert(b.end(), v.begin(), v.end());
}
void shard(const fs::path& path, int part, bool mtp, const std::vector<std::string>& names) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747); put<uint32_t>(b, 3); put<uint64_t>(b, names.size());
    put<uint64_t>(b, part == 0 ? 11 : 3);
    auto u32 = [&](const std::string& key, uint32_t v) { str(b, key); put<uint32_t>(b, 4); put<uint32_t>(b, v); };
    if (part == 0) {
        str(b, "general.architecture"); put<uint32_t>(b, 8); str(b, "qwen4exp");
        u32("qwen4exp.block_count", mtp ? 49 : 48); u32("qwen4exp.nextn_predict_layers", mtp ? 1 : 0);
        u32("qwen4exp.embedding_length", 2560); u32("qwen4exp.expert_count", 512);
        u32("qwen4exp.expert_used_count", 10); u32("qwen4exp.attention.head_count", 24);
        u32("qwen4exp.attention.head_count_kv", 2);
    }
    u32("split.count", 2); u32("split.no", part); u32("split.tensors.count", mtp ? 5 : 2);
    for (size_t i = 0; i < names.size(); ++i) {
        str(b, names[i]); put<uint32_t>(b, 2); put<uint64_t>(b, 32); put<uint64_t>(b, 1);
        put<uint32_t>(b, 8); put<uint64_t>(b, i * 64); // one Q8_0 block, 64-byte spacing
    }
    b.resize((b.size() + 31) / 32 * 32 + names.size() * 64, 0);
    std::ofstream(path, std::ios::binary).write((const char*) b.data(), b.size());
}
void index(const fs::path& d, const std::set<std::string>& names) {
    std::ofstream f(d / "index.txt");
    f << "# strata pack index v3\n# align 64 pool 64 tensors " << names.size() << '\n';
    for (const auto& n : names) f << n << " 0 0 0 0 0 0 32 1 8 0 32 0 0 0 0 0 0 0\n";
}
}
int main() {
    const fs::path root = fs::temp_directory_path() /
        ("maya-native-dense-split-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    int devices = 0;
    const bool gpu = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
    const std::set<std::string> expected = {"blk.0.attn_q.weight", "blk.47.attn_output.weight"};
    for (bool mtp : {false, true}) {
        const fs::path d = root / (mtp ? "embedded-mtp" : "trunk-only"); fs::create_directory(d);
        std::vector<std::string> last = {"blk.47.attn_output.weight"};
        if (mtp) for (const char* role : {"gate", "up", "down"}) last.push_back("blk.48.ffn_" + std::string(role) + "_shexp.weight");
        shard(d / "first.gguf", 0, mtp, {"blk.0.attn_q.weight"}); shard(d / "last.gguf", 1, mtp, last);
        const std::vector<std::string> paths = {(d / "first.gguf").string(), (d / "last.gguf").string()};
        std::set<std::string> names; std::string err;
        check(strata::core::NativeDense::served_names(paths, false, names, err) && names == expected,
              d.filename().string() + " discovery: only trunk matrices (" + err + ")");
        if (!gpu) continue;
        index(d, expected);
        {
            strata::core::WeightTable table; strata::core::NativeDense dense;
            const bool ok = table.load(d.string(), nullptr, 0, err, &expected) && dense.load(paths, table, err);
            check(ok && dense.tensor_count() == 2 && table.find(*expected.rbegin())->native_data,
                  d.filename().string() + " upload: late-shard trunk loaded, MTP excluded (" + err + ")");
        }
        index(d, {"blk.0.attn_q.weight"}); err.clear();
        {
            strata::core::WeightTable table; strata::core::NativeDense dense;
            check(table.load(d.string(), nullptr, 0, err, &expected) && !dense.load(paths, table, err) &&
                  err.find("absent from canonical table: blk.47.attn_output.weight") != std::string::npos,
                  d.filename().string() + " missing trunk matrix is still rejected");
        }
    }
    fs::remove_all(root);
    return fails ? 1 : gpu ? 0 : 77;
}
