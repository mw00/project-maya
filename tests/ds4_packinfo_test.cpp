// maya-ds4 --pack-info (src/ds4/ds4_packinfo.cpp) on small packs written here: a healthy GGUF-style pack and an EXL3
// one load (0, every file there); a pack whose model_dir is gone, or whose file is short, is a dead pack (2); a type no
// kernel takes stops it (2); a drafter its metadata leaves unloaded (dspark_block_size 0) and an expert profile of
// another size are warned about; the profile's most-routed experts are counted; no ds4.txt is not a pack (1).  The
// kernels' types come from a stand-in KernelTypes (the engine asks its own).  Host code only, no GPU.
#include "../src/ds4/ds4_packinfo.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::ds4::KernelTypes;

namespace {
int fails = 0;
void expect(bool ok, const std::string& what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++fails;
}

// a few ggml types, as ds4_iq / ds4_gguf name them: the experts' kernels take IQ2_S, IQ3_XXS and MXFP4, the dense
// GEMVs Q8_0, Q6_K and the low-bit ones; EXL3 K 1..8 and 1.5 / 2.5 / 3.5
KernelTypes kernels() {
    KernelTypes kt;
    kt.name = [](int t) -> const char* {
        switch (t) {
            case 8: return "Q8_0";
            case 14: return "Q6_K";
            case 18: return "IQ3_XXS";
            case 22: return "IQ2_S";
            case 39: return "MXFP4";
            default: return "?";
        }
    };
    kt.expert = [](int t) { return t == 18 || t == 22 || t == 39; };
    kt.dense = [](int t) { return t == 8 || t == 14 || t == 18 || t == 22; };
    kt.exl3_k2 = [](int k2) { return (k2 >= 2 && k2 <= 16 && k2 % 2 == 0) || k2 == 3 || k2 == 5 || k2 == 7; };
    return kt;
}

struct T {
    std::string name, dtype;
    uint64_t bytes = 0;
    std::vector<int64_t> shape;
};

// a pack: ds4.txt's keys, its tensors (laid out one after another in model/w.bin), engram.bin, a tokenizer
struct Pack {
    fs::path root;
    std::map<std::string, std::string> kv;
    std::vector<T> t;
    bool exl3 = false;

    explicit Pack(const fs::path& r) : root(r) {
        kv = {{"vocab_size", "64"}, {"hidden_size", "256"}, {"moe_intermediate_size", "512"},
              {"num_hidden_layers", "2"}, {"n_routed_experts", "4"}, {"num_experts_per_tok", "2"},
              {"head_dim", "512"}, {"index_head_dim", "128"}, {"hc_mult", "4"}, {"compress_ratios", "0,0,0"},
              {"engram_layer_ids", "1"}, {"dspark_block_size", "5"}, {"dspark_noise_token_id", "60"},
              {"dspark_n_routed_experts", "4"}, {"dspark_num_experts_per_tok", "2"},
              {"dspark_target_layer_ids", "1"}, {"format", "gguf"}};
    }
    void experts(const std::string& pre, int n, const char* gu, const char* dn, uint64_t b_gu, uint64_t b_dn) {
        for (int e = 0; e < n; ++e)
            for (const char* r : {"w1", "w3", "w2"}) {
                const bool down = std::strcmp(r, "w2") == 0;
                t.push_back({pre + std::to_string(e) + "." + r + ".weight", down ? dn : gu, down ? b_dn : b_gu, {}});
            }
    }
    // an EXL3 matrix's pieces (k x n, K = k2 / 2 bits)
    void exl3_mat(const std::string& pre, int k, int n, int k2) {
        t.push_back({pre + ".trellis", "I16", (uint64_t) (k / 16) * (n / 16) * 16 * k2, {k / 16, n / 16, 8 * k2}});
        t.push_back({pre + ".suh", "F16", (uint64_t) k * 2, {k}});
        t.push_back({pre + ".svh", "F16", (uint64_t) n * 2, {n}});
        t.push_back({pre + ".mul1", "I32", 4, {}});
    }
    void standard() {
        t.push_back({"embed.weight", "BF16", 64 * 256 * 2, {64, 256}});
        t.push_back({"norm.weight", "F32", 256 * 4, {256}});
        t.push_back({"head.weight", "Q8_0", 64 * 256 / 32 * 34, {64, 256}});
        t.push_back({"layers.0.attn.wq_a.weight", "Q8_0", 256 * 256 / 32 * 34, {256, 256}});
        t.push_back({"layers.1.engram.embed.weight", "Q6_K", 1000, {100, 256}});
        t.push_back({"mtp.0.attn.wq_a.weight", "Q8_0", 256 * 256 / 32 * 34, {256, 256}});
        for (int il = 0; il < 2; ++il) {
            const std::string pre = "layers." + std::to_string(il) + ".ffn.experts.";
            if (exl3)
                for (int e = 0; e < 4; ++e) {
                    const std::string E = pre + std::to_string(e) + ".";
                    exl3_mat(E + "w1", 256, 512, 8);
                    exl3_mat(E + "w3", 256, 512, 8);
                    exl3_mat(E + "w2", 512, 256, il == 1 && e == 3 ? 6 : 8);   // (a mixed layer: one expert at 3 bits)
                }
            else
                experts(pre, 4, "IQ2_S", "IQ3_XXS", 512 / 256 * 82 * 256, 256 / 256 * 98 * 512);
        }
        experts("mtp.0.ffn.experts.", 4, "MXFP4", "MXFP4", 512 * 256 / 32 * 17, 256 * 512 / 32 * 17);
    }
    // writes it all; the weights' file `cut` bytes short of its tensors' end
    void write(uint64_t cut = 0) const {
        fs::remove_all(root);
        fs::create_directories(root / "pack" / "tokenizer");
        fs::create_directories(root / "model");
        {
            std::ofstream f(root / "pack" / "ds4.txt");
            for (const auto& [k, v] : kv) f << k << " " << v << "\n";
            f << "model_dir " << (root / "model").string() << "\n";
        }
        uint64_t off = 0;
        {
            std::ofstream f(root / "pack" / "tensors.txt");
            f << "file 0 w.bin\n";
            for (const T& x : t) {
                f << "t " << x.name << " " << x.dtype << " 0 " << off << " " << x.bytes;
                for (int64_t s : x.shape) f << " " << s;
                f << "\n";
                off += x.bytes;
            }
        }
        { std::ofstream f(root / "model" / "w.bin", std::ios::binary); }
        fs::resize_file(root / "model" / "w.bin", off - cut);   // (sparse: nothing written)
        {
            std::ofstream f(root / "pack" / "engram.bin", std::ios::binary);
            const int32_t hdr[6] = {64, 1, 4, 8, 256, 2};
            f.write("DSE1", 4);
            f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
        }
        std::ofstream(root / "pack" / "tokenizer" / "vocab.json") << "{}";
    }
    void profile(const std::vector<uint32_t>& counts) const {
        std::ofstream f(root / "pack" / "expert-profile.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(counts.data()), (std::streamsize) (counts.size() * 4));
    }
};

// pack_info's code and its report
int run(const Pack& p, std::string& text) {
    std::FILE* f = std::tmpfile();
    const int rc = strata::ds4::pack_info((p.root / "pack").string(), "", kernels(), f);
    std::rewind(f);
    text.clear();
    char buf[4096];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, n);
    std::fclose(f);
    return rc;
}
bool has(const std::string& text, const char* s) { return text.find(s) != std::string::npos; }

}  // namespace

int main() {
    const fs::path base = fs::temp_directory_path() / ("ds4_packinfo_test_" + std::to_string((long long) std::time(nullptr)));
    std::string out;
    {   // healthy (GGUF experts, the drafter loaded): it can load, every file there
        Pack p(base / "a");
        p.standard();
        p.write();
        const int rc = run(p, out);
        expect(rc == 0, "a healthy pack: 0, got " + std::to_string(rc) + "\n" + out);
        expect(has(out, "files: 1 in model_dir, all there and long enough"), "a healthy pack: its file checked\n" + out);
        expect(has(out, "no GPU kernel: none"), "a healthy pack: every type taken\n" + out);
        expect(has(out, "the DSpark drafter: 1 blocks"), "a healthy pack: the drafter loads\n" + out);
        expect(has(out, "IQ2_S") && has(out, "IQ3_XXS"), "a healthy pack: its expert types in the map\n" + out);
        expect(has(out, "expert matrices by type: IQ2_S 16, IQ3_XXS 8, MXFP4 12\n"), "a healthy pack: the matrices "
               "counted by type\n" + out);
        expect(has(out, "Engram: layers 1, tables in Q6_K"), "a healthy pack: the Engram line\n" + out);
        expect(!has(out, "WARNING"), "a healthy pack: no warning\n" + out);
        expect(has(out, "expert profile") && has(out, ": none"), "a healthy pack: no profile said\n" + out);

        // the profile: 3 layers x 4 counts (the drafter's too); one expert takes more than half of the trunk's routes
        p.profile({100, 10, 10, 10, 10, 10, 10, 10, 5, 5, 5, 5});
        run(p, out);
        expect(has(out, "50% of its routes: 1 ("), "the profile: one expert takes half the routes\n" + out);
        expect(has(out, "170 routes"), "the profile: the trunk's routes counted\n" + out);
        p.profile({1, 2, 3, 4, 5});
        run(p, out);
        expect(has(out, "the engine ignores it"), "a profile of another size: warned\n" + out);
        p.profile({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});   // (more than the trunk's, fewer than with the drafter's)
        run(p, out);
        expect(has(out, "ignores it when the drafter is loaded"), "a profile between the sizes: warned\n" + out);
    }
    {   // model_dir cleaned up: a dead pack
        Pack p(base / "b");
        p.standard();
        p.write();
        fs::remove_all(base / "b" / "model");
        const int rc = run(p, out);
        expect(rc == 2, "model_dir gone: 2\n" + out);
        expect(has(out, "DEAD PACK") && has(out, "is not there") && has(out, "1 of the 1 files it lists are missing"),
               "model_dir gone: called a dead pack\n" + out);
    }
    {   // the weights' file shorter than its tensors
        Pack p(base / "c");
        p.standard();
        p.write(100);
        const int rc = run(p, out);
        expect(rc == 2 && has(out, "SHORT") && !has(out, "missing"), "a short file: 2 and SHORT\n" + out);
    }
    {   // dspark_block_size 0 beside the drafter's tensors: it can load, without the drafter - warned
        Pack p(base / "d");
        p.kv["dspark_block_size"] = "0";
        p.standard();
        p.write();
        const int rc = run(p, out);
        expect(rc == 0, "block size 0: still loads\n" + out);
        expect(has(out, "WARNING: dspark_block_size 0") && has(out, "does not load the drafter"),
               "block size 0: warned\n" + out);
        expect(has(out, "not loaded"), "block size 0: the drafter's blocks said not loaded\n" + out);
    }
    {   // a routed expert type no kernel takes (Q8_0), and an Engram table in a type the reader does not take
        Pack p(base / "e");
        p.standard();
        for (T& x : p.t)
            if (x.name == "layers.1.ffn.experts.2.w2.weight") x.dtype = "Q8_0";
            else if (x.name == "layers.1.engram.embed.weight") x.dtype = "BF16";
        p.write();
        const int rc = run(p, out);
        expect(rc == 2, "an expert in Q8_0: 2\n" + out);
        expect(has(out, "routed experts in Q8_0: 1"), "an expert in Q8_0: named with its layer\n" + out);
        expect(has(out, "GGUF experts of several types"), "a GGUF layer of two types: cannot load\n" + out);
        expect(has(out, "an Engram table in BF16"), "an Engram table in BF16: named\n" + out);
    }
    {   // EXL3 experts (a mixed layer: one expert's down at 3 bits) load; a bitrate the kernels lack (K 9) does not
        Pack p(base / "f");
        p.exl3 = true;
        p.kv.erase("format");
        p.standard();
        p.write();
        int rc = run(p, out);
        expect(rc == 0, "EXL3 experts: 0\n" + out);
        expect(has(out, "EXL3 4") && has(out, "EXL3 3 x1") && has(out, "(mixed)"), "EXL3 experts: K and the mixed layer\n" + out);
        for (T& x : p.t)
            if (x.name == "layers.0.ffn.experts.1.w1.trellis") x.shape[2] = 8 * 18;
        p.write();
        rc = run(p, out);
        expect(rc == 2 && has(out, "routed experts in EXL3 9"), "EXL3 K 9: no kernel\n" + out);
    }
    {   // not a pack
        Pack p(base / "g");
        p.standard();
        p.write();
        fs::remove(base / "g" / "pack" / "ds4.txt");
        expect(run(p, out) == 1, "no ds4.txt: 1\n" + out);
    }
    {   // geometry the engine does not cover, and no engram.bin
        Pack p(base / "h");
        p.kv["head_dim"] = "256";
        p.standard();
        p.write();
        fs::remove(base / "h" / "pack" / "engram.bin");
        const int rc = run(p, out);
        expect(rc == 2 && has(out, "geometry the engine does not cover") && has(out, "engram.bin is not in the pack"),
               "head_dim 256, no engram.bin: cannot load\n" + out);
    }
    std::error_code ec;
    fs::remove_all(base, ec);
    if (fails == 0) std::printf("ds4_packinfo_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
