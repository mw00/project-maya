// tests/glm_dsa/iq_fixture_cpu_test.cpp - B13.1: the tools/iq_fixture.py expert fixtures through ggml-cpu, no GPU.
//
//   iq_fixture_cpu_test <fixture.gguf> <fixture.ref.f32>
//
// Per layer (one K-quant or i-quant type each), on expert 7 of blk.L.ffn_{gate,up,down}_exps:
//   (1) ggml's to_float of the first 16 gate rows and 16 down rows equals gguf-py's dequant (the --ref-out dump):
//       two independent decoders of the same blocks (max abs diff <= 1e-6 x the row scale);
//   (2) the CPU native expert (native_fmt / native_quant_act / native_gu_rows / native_down_rows: the decode lane's
//       arithmetic) against a double float SwiGLU reference over 3 tokens: rel < 3e-2 (activation rounding only), the
//       bound native_expert_parity uses;
//   (3) where the multi-token i-quant kernels serve the type (iq256 always, iq512 on an AVX-512 VBMI/VNNI CPU), their
//       gate rows against ggml's vec_dot on the same Q8_K activations: rel < 1e-5.
// The GPU third of the comparison is native_expert_parity on the same file (ctest native_expert_parity_fixture,
// label gpu).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) { std::fprintf(stderr, "usage: iq_fixture_cpu_test <fixture.gguf> <fixture.ref.f32>\n"); return 2; }
    if (!cpu::native_experts_available()) { std::printf("iq_fixture_cpu_test: built without native experts\n"); return 77; }
    strata::GgufFile gguf(argv[1]);
    std::FILE* rf = std::fopen(argv[2], "rb");
    if (!rf) { std::fprintf(stderr, "iq_fixture_cpu_test: cannot open %s\n", argv[2]); return 1; }
    constexpr int E = 7, R = 16, NT = 3;
    int failures = 0, layers = 0;
    for (int l = 0;; ++l) {
        const strata::TensorInfo* t[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& ti : gguf.tensors())
            for (int r = 0; r < 3; ++r)
                if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
        if (!t[0]) break;
        ++layers;
        if (!t[1] || !t[2]) { std::printf("layer %d: missing up/down\n", l); ++failures; continue; }
        const int64_t H = (int64_t) t[0]->shape[0], FF = (int64_t) t[0]->shape[1];
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
            std::printf("layer %d %s: %s\n", l, t[0]->type_name(), err.c_str());
            ++failures;
            continue;
        }
        std::vector<uint8_t> blob(f.bytes);
        std::memcpy(blob.data(), gguf.tensor_data(*t[0]) + (size_t) E * f.up_off, f.up_off);
        std::memcpy(blob.data() + f.up_off, gguf.tensor_data(*t[1]) + (size_t) E * f.up_off, f.up_off);
        std::memcpy(blob.data() + f.down_off, gguf.tensor_data(*t[2]) + (size_t) E * (f.bytes - f.down_off),
                    f.bytes - f.down_off);
        const auto* tg = ggml_get_type_traits((ggml_type) f.gu_type);
        const auto* td = ggml_get_type_traits((ggml_type) f.d_type);

        // (1) ggml to_float vs the gguf-py rows
        std::vector<float> ref_g((size_t) R * H), ref_d((size_t) R * FF);
        if (std::fread(ref_g.data(), 4, ref_g.size(), rf) != ref_g.size() ||
            std::fread(ref_d.data(), 4, ref_d.size(), rf) != ref_d.size()) {
            std::printf("layer %d: reference file too short\n", l);
            return 1;
        }
        std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
        for (int64_t r = 0; r < FF; ++r) {
            tg->to_float(blob.data() + r * f.gu_row, G.data() + r * H, H);
            tg->to_float(blob.data() + f.up_off + r * f.gu_row, U.data() + r * H, H);
        }
        for (int64_t r = 0; r < H; ++r) td->to_float(blob.data() + f.down_off + r * f.d_row, D.data() + r * FF, FF);
        double dmax = 0, scale = 0;
        bool finite = true;
        for (size_t i = 0; i < ref_g.size(); ++i) {
            dmax = std::max(dmax, std::fabs((double) G[i] - ref_g[i]));
            scale = std::max(scale, std::fabs((double) ref_g[i]));
            finite = finite && std::isfinite(G[i]);
        }
        for (size_t i = 0; i < ref_d.size(); ++i) {
            dmax = std::max(dmax, std::fabs((double) D[i] - ref_d[i]));
            scale = std::max(scale, std::fabs((double) ref_d[i]));
            finite = finite && std::isfinite(D[i]);
        }
        const bool dq_ok = finite && scale > 0 && dmax <= 1e-6 * scale;

        // (2) the CPU native expert vs the float reference
        std::mt19937 rng(11 + l);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) NT * H);
        for (auto& v : x) v = nd(rng);
        std::vector<float> ref((size_t) NT * H), got((size_t) NT * H);
        for (int k = 0; k < NT; ++k) {
            std::vector<double> h((size_t) FF);
            for (int64_t r = 0; r < FF; ++r) {
                double g = 0, u = 0;
                for (int64_t i = 0; i < H; ++i) {
                    g += (double) G[r * H + i] * x[k * H + i];
                    u += (double) U[r * H + i] * x[k * H + i];
                }
                h[(size_t) r] = g / (1.0 + std::exp(-g)) * u;
            }
            for (int64_t r = 0; r < H; ++r) {
                double o = 0;
                for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[(size_t) i];
                ref[k * H + r] = (float) o;
            }
        }
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes)),
            hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
        std::vector<std::vector<float>> ff(NT, std::vector<float>((size_t) FF));
        const void* a[NT];
        float* ffp[NT];
        const void* hp[NT];
        float* op[NT];
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_act(f, x.data() + k * H, act[k].data());
            a[k] = act[k].data();
            ffp[k] = ff[k].data();
        }
        cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_h(f, ff[k].data(), hq[k].data());
            hp[k] = hq[k].data();
            op[k] = got.data() + k * H;
        }
        cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, (int) H);
        const double ec = rel(got, ref);
        const bool cpu_ok = std::isfinite(ec) && ec < 3e-2;

        // (3) the multi-token i-quant kernels vs ggml vec_dot
        std::string mk;
        bool mk_ok = true;
        if (cpu::iq512_supported(f.gu_type)) {
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
            std::vector<float> gref((size_t) NT * FF);
            for (int k = 0; k < NT; ++k)
                for (int64_t r = 0; r < FF; ++r)
                    tc->vec_dot((int) H, &gref[k * FF + r], 0, blob.data() + r * f.gu_row, 0, a[k], 0, 1);
            auto run = [&](bool is512) {
                std::vector<float> g((size_t) NT * FF);
                float* gp[NT];
                for (int k = 0; k < NT; ++k) gp[k] = g.data() + k * FF;
                (is512 ? cpu::iq512_rows : cpu::iq256_rows)(f.gu_type, blob.data(), f.gu_row, (int) H, a, NT, gp, 0,
                                                             (int) FF);
                const double rg = rel(g, gref);
                char buf[64];
                std::snprintf(buf, sizeof buf, " %s %.1e", is512 ? "iq512" : "iq256", rg);
                mk += buf;
                if (!(rg < 1e-5)) mk_ok = false;
            };
            if (cpu::cpu_avx512_ok()) run(true);
            run(false);
        }
        const bool ok = dq_ok && cpu_ok && mk_ok;
        std::printf("layer %2d %-8s gate/up %6zu B/row  dequant vs gguf-py %.1e (scale %.1e)  cpu rel %.2e%s  %s\n", l,
                    t[0]->type_name(), f.gu_row, dmax, scale, ec, mk.empty() ? "" : ("  vec_dot" + mk).c_str(),
                    ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    }
    std::fclose(rf);
    if (layers == 0) { std::printf("iq_fixture_cpu_test: no expert layers in %s\n", argv[1]); return 1; }
    std::printf("iq_fixture_cpu_test: %d layers, %d failures\n", layers, failures);
    return failures ? 1 : 0;
}
