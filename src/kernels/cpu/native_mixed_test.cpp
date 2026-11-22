// src/kernels/cpu/native_mixed_test.cpp - a native expert layer whose gate and up are of two types (unsloth's
// Qwen3.8 UD-IQ1_S keeps blk.18's gate IQ2_S and its up IQ2_XXS).  Synthetic, CPU only:
//   1. native_fmt_mixed: the blob layout [gate | up | down] with each role's own row size; the same-type call is
//      native_fmt field for field; a pair whose activation formats differ is refused
//   2. native_gu_rows on the mixed layer, one and three tokens, against ggml-cpu's own vec_dot per type (the
//      products llama.cpp computes): equal bit for bit
//   3. expert_layout_load reads a "gate:up" type column (and still the plain one); a malformed column is refused
#include "maya/kernels/cpu/native_expert.hpp"
#include "maya/kernels/cpu/expert.hpp"
#include "maya/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace cpu = maya::kernels::cpu;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "native_mixed_test: FAIL %s\n", what.c_str());
        ++failures;
    }
}

static bool same_fmt(const cpu::NativeFmt& a, const cpu::NativeFmt& b) {
    return a.gu_type == b.gu_type && a.d_type == b.d_type && a.gu_act == b.gu_act && a.d_act == b.d_act &&
           a.n_embd == b.n_embd && a.n_ff == b.n_ff && a.gu_row == b.gu_row && a.d_row == b.d_row &&
           a.up_off == b.up_off && a.down_off == b.down_off && a.bytes == b.bytes && a.act_bytes == b.act_bytes &&
           a.h_bytes == b.h_bytes && a.u_type == b.u_type && a.u_row == b.u_row;
}

int main() {
    const int G = GGML_TYPE_IQ2_S, U = GGML_TYPE_IQ2_XXS, D = GGML_TYPE_Q8_0;
    const int64_t H = 512, FF = 256;
    std::string err;

    // 1. the layout
    cpu::NativeFmt f, same, plain;
    check(cpu::native_fmt_mixed(G, U, D, H, FF, f, err), "mixed IQ2_S / IQ2_XXS format: " + err);
    check(f.mixed() && f.gu_type == G && f.u_type == U, "mixed types recorded");
    check(f.gu_row == ggml_row_size((ggml_type) G, H) && f.u_row == ggml_row_size((ggml_type) U, H), "row sizes");
    check(f.up_off == f.gu_row * FF && f.down_off == f.up_off + f.u_row * FF &&
          f.bytes == f.down_off + f.d_row * H, "blob offsets");
    check(f.u_row != f.gu_row, "the two types differ in size");
    for (int t : {G, U, (int) GGML_TYPE_Q4_K, (int) GGML_TYPE_IQ4_XS}) {
        check(cpu::native_fmt_mixed(t, t, D, H, FF, same, err) && cpu::native_fmt(t, D, H, FF, plain, err) &&
              same_fmt(same, plain) && !same.mixed(), "same-type call equals native_fmt for type " + std::to_string(t));
        check(same.down_off == 2 * same.up_off, "same-type layout unchanged");
    }
    check(!cpu::native_fmt_mixed(GGML_TYPE_Q8_0, G, D, H, FF, same, err), "Q8_0 gate with IQ2_S up refused");

    // 2. the products: random weights, quantized per type (the i-quants with a flat importance matrix)
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w((size_t) FF * H), imat((size_t) H, 1.0f), x((size_t) 3 * H);
    std::vector<uint8_t> blob(f.bytes);
    for (auto& v : w) v = nd(rng);
    ggml_quantize_init((ggml_type) G);
    ggml_quantize_init((ggml_type) U);
    ggml_quantize_chunk((ggml_type) G, w.data(), blob.data(), 0, FF, H, imat.data());
    for (auto& v : w) v = nd(rng);
    ggml_quantize_chunk((ggml_type) U, w.data(), blob.data() + f.up_off, 0, FF, H, imat.data());
    std::vector<float> wd((size_t) H * FF);
    for (auto& v : wd) v = nd(rng);
    ggml_quantize_chunk((ggml_type) D, wd.data(), blob.data() + f.down_off, 0, H, FF, nullptr);
    for (auto& v : x) v = nd(rng);
    std::vector<std::vector<uint8_t>> act(3, std::vector<uint8_t>(cpu::kNativeActBytes));
    const void* a[3];
    for (int t = 0; t < 3; ++t) {
        cpu::native_quant_act(f, x.data() + t * H, act[(size_t) t].data());
        a[t] = act[(size_t) t].data();
    }
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) G);
    const auto* tu = ggml_get_type_traits_cpu((ggml_type) U);
    std::vector<float> ref((size_t) 3 * FF);
    for (int t = 0; t < 3; ++t)
        for (int64_t r = 0; r < FF; ++r) {
            float g = 0.f, u = 0.f;
            tg->vec_dot((int) H, &g, 0, blob.data() + r * f.gu_row, 0, a[t], 0, 1);
            tu->vec_dot((int) H, &u, 0, blob.data() + f.up_off + r * f.u_row, 0, a[t], 0, 1);
            ref[(size_t) t * FF + r] = (g / (1.f + std::exp(-g))) * u;
        }
    for (int nt : {1, 3}) {
        std::vector<float> got((size_t) nt * FF, NAN);
        float* o[3];
        for (int t = 0; t < nt; ++t) o[t] = got.data() + (size_t) t * FF;
        // two row ranges, as the pool's tasks split them
        cpu::native_gu_rows(f, blob.data(), a, nt, o, 0, 100);
        cpu::native_gu_rows(f, blob.data(), a, nt, o, 100, (int) FF);
        size_t diff = 0;
        for (size_t i = 0; i < got.size(); ++i) diff += got[i] != ref[i];
        check(diff == 0, "mixed gate/up rows vs ggml vec_dot, " + std::to_string(nt) + " token(s): " +
                             std::to_string(diff) + " values differ");
    }

    // 3. the pack index
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("native_mixed_test_" + std::to_string((unsigned) rng()));
    fs::create_directories(dir);
    cpu::NativeFmt l0, l1;
    check(cpu::native_fmt(G, GGML_TYPE_IQ4_NL, cpu::H, cpu::FF, l0, err) &&
          cpu::native_fmt_mixed(G, U, GGML_TYPE_IQ4_NL, cpu::H, cpu::FF, l1, err), "pack geometry: " + err);
    const int64_t n_expert = 4;
    const auto write = [&](const std::string& col1) {
        std::ofstream o(dir / "native_experts.txt", std::ios::binary);
        o << "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off (n_expert "
          << n_expert << ", total 0; absolute offsets in x.gguf)\n";
        o << "0 " << G << " " << (int) GGML_TYPE_IQ4_NL << " 0 " << l0.bytes << " 64 128 256\n";
        o << "1 " << col1 << " " << (int) GGML_TYPE_IQ4_NL << " " << l0.bytes * n_expert << " " << l1.bytes
          << " 512 1024 2048\n";
    };
    write(std::to_string(G) + ":" + std::to_string(U));
    check(cpu::expert_layout_load(dir.string(), 2, n_expert, err), "load a gate:up row: " + err);
    const auto& L = cpu::expert_layout();
    check(L.native && !L.fmt[0].mixed() && same_fmt(L.fmt[0], l0), "plain row parsed as before");
    check(L.fmt[1].mixed() && same_fmt(L.fmt[1], l1) && L.bytes[1] == l1.bytes, "mixed row parsed");
    check(L.gguf_off.size() == 6 && L.gguf_off[4] == 1024 && L.gguf_off[5] == 2048, "mixed row offsets");
    for (const std::string bad : {std::to_string(G) + ":", ":" + std::to_string(U), std::to_string(G) + ":x",
                                  std::to_string(G) + "x"}) {
        write(bad);
        check(!cpu::expert_layout_load(dir.string(), 2, n_expert, err), "malformed type column refused: " + bad);
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    cpu::expert_layout_load((dir / "none").string(), 1, 1, err);   // back to the canonical layout

    if (failures) return 1;
    std::puts("native_mixed_test: ok");
    return 0;
}
