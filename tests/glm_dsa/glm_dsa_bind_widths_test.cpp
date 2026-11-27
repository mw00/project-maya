// tests/glm_dsa/glm_dsa_bind_widths_test.cpp - (label gpu) the full GLM-5.3 (glm-dsa) fast path binds every layer with
// the FFN widths its launches are sized from, and answers - on a small synthetic pack, no real model needed (#1200).
//
// v1.1.75 made the shared MoE / dense code read the layer's own widths (FastState::Layer::ff / ffd / f0).  The
// GLM-5.3-Flash loader set them, the full model's binder (Glm5Model::fast_bind_dsa) did not: every expert and dense
// launch had a width of 0, the driver refused them and every request failed - on every real pack, for three releases,
// because the release checks only ran Flash.  This is the check that fails on that build in a few seconds:
//
//   1. the pack ($MAYA_TEST_GLM_DSA_PACK, else the PACK9 preset as glm_dsa_pack_fast_test builds it) loads on one
//      part with the fast path;
//   2. every layer the part holds has ff == n_ff_exp (MoE layers) or ffd == n_ff_dense (dense ones), f0 == 0;
//   3. a prompt and 8 greedy tokens run with no refused launch (kernels::glmf::launch_errors() stays 0) and finite
//      logits that are not all one value.
// Without a device, or without the fixture tools, the test exits 77.
#include "maya/core/glm_arch.hpp"
#include "maya/core/glm_model.hpp"
#include "maya/kernels/glm_fast.hpp"

#include "glm_dsa_fixtures.hpp"
#include "glm_test_env.hpp"   // setenv / unsetenv (<unistd.h> on POSIX; mapped on Windows)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
constexpr int kSkip = 77;
constexpr int kTokens = 8;
constexpr int64_t kCtx = 1024;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("glm_dsa_bind_widths_test: the glm-dsa fast path's FFN widths on a synthetic pack (#1200)\n");
    std::string why, err;
    fs::path pack;
    if (const char* p = std::getenv("MAYA_TEST_GLM_DSA_PACK"); p != nullptr && p[0] != '\0') {
        pack = p;
    } else {
        const fs::path gguf = glmfx::synth("PACK9", why);
        pack = gguf.empty() ? fs::path() : glmfx::pack(gguf, why);
        if (pack.empty()) return glmfx::fixture_missing("glm_dsa_bind_widths_test", why);
    }
    maya::core::Glm5Geometry g;
    if (!maya::core::glm_pack_geometry(pack.string(), g, err) || g.arch != maya::core::GlmArch::GlmDsa) {
        std::fprintf(stderr, "glm_dsa_bind_widths_test: %s: %s\n", pack.string().c_str(),
                     err.empty() ? "not a glm-dsa pack" : err.c_str());
        return 1;
    }
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glm_dsa_bind_widths_test: no device - skipped\n");
        return kSkip;
    }
    unsetenv("MAYA_GLM_SLOW");
    unsetenv("MAYA_GLM_TP");
    unsetenv("MAYA_GLM_MTP");
    setenv("MAYA_GLM_SPLIT", "0", 1);

    auto m = std::make_unique<maya::core::Glm5Model>();
    const bool loaded = m->load_pack_env(pack.string(), kCtx, err);
    check(loaded && m->fast(), "one part loads with the fast path" + (err.empty() ? std::string() : ": " + err));
    if (!loaded || !m->fast()) {
        std::printf("glm_dsa_bind_widths_test: %d FAILED\n", g_fail);
        return 1;
    }

    int held = 0, bad = 0;
    std::string first_bad;
    for (int il = 0; il <= (int) g.n_layers; ++il) {
        int ff = -1, ffd = -1, f0 = -1;
        if (!m->fast_ffn_width(il, ff, ffd, f0)) continue;
        ++held;
        const bool moe = il >= (int) g.dense_lead;
        const bool ok = f0 == 0 && (moe ? ff == (int) g.n_ff_exp : ffd == (int) g.n_ff_dense);
        if (!ok && first_bad.empty())
            first_bad = "layer " + std::to_string(il) + (moe ? " (MoE)" : " (dense)") + ": ff " + std::to_string(ff) +
                        ", ffd " + std::to_string(ffd) + ", f0 " + std::to_string(f0) + " - the model's are " +
                        std::to_string((int) g.n_ff_exp) + " / " + std::to_string((int) g.n_ff_dense);
        bad += !ok;
    }
    check(held >= (int) g.n_layers, "the part holds every layer: " + std::to_string(held) + " of " +
                                        std::to_string((int) g.n_layers));
    check(bad == 0, bad == 0 ? "every layer's FFN width is the model's (" + std::to_string((int) g.n_ff_exp) +
                                   " per expert, " + std::to_string((int) g.n_ff_dense) + " dense)"
                             : std::to_string(bad) + " layer(s) with a wrong FFN width; " + first_bad);

    const int errors_before = maya::kernels::glmf::launch_errors();
    const std::vector<int32_t> prompt = {11, 220, 3, 47, 1000 % std::max(1, (int) g.n_vocab), 5, 9, 31};
    std::vector<float> logits;
    bool ran = m->forward(prompt, logits, err);
    bool finite = true, flat = false;
    for (int i = 0; ran && i < kTokens; ++i) {
        const auto mm = std::minmax_element(logits.begin(), logits.end());
        for (float v : logits) finite = finite && std::isfinite(v);
        flat = flat || logits.empty() || *mm.first == *mm.second;
        const int t = (int) (mm.second - logits.begin());
        if (i + 1 < kTokens) ran = m->forward({t}, logits, err);
    }
    check(ran, "a prompt and " + std::to_string(kTokens) + " greedy tokens run" +
                   (ran ? std::string() : ": " + err));
    const int refused = maya::kernels::glmf::launch_errors() - errors_before;
    check(refused == 0, "no refused kernel launch (" + std::to_string(refused) + ")");
    check(ran && finite && !flat, "the logits are finite and not all one value");
    unsetenv("MAYA_GLM_SPLIT");

    if (g_fail) {
        std::printf("glm_dsa_bind_widths_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_dsa_bind_widths_test: PASSED\n");
    return 0;
}
