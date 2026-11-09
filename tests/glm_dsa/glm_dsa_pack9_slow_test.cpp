// tests/glm_dsa/glm_dsa_pack9_slow_test.cpp - glm-dsa B3.7 (label gpu): the slow path's device expert pool with 9
// expert size classes, on the synthetic PACK9 preset (12 layers + NextN, 10 MoE layers whose (gate/up, down) expert
// types form 9 row-byte classes; tools/glm_dsa_synth_gguf.py --preset PACK9, packed by tools/iq_pack.py's glm-dsa
// mode).  Maya's four class arrays were [8] with no bound check; they are vectors now (glm_model.hpp) and
// STRATA_GLM_POOL_CLASSES folds the classes past a cap into one (src/core/glm_pool_classes.cpp), so the SAME weights
// run with 9, 8 and 1 classes.
//
// STRATA_GLM_SLOW=1, one GPU, 12 reference tokens ((i * 37 + 11) % 1024), token by token:
//   9 classes                         the pool really has 9 classes (glm_pool_classes_last), logits L9
//   STRATA_GLM_POOL_CLASSES=8         8 classes: max |logit - L9| < 1e-5 (the B3.7 gate: same kernels, other slots)
//   STRATA_GLM_NO_STAGE=1             the three-copy fill instead of the staging workers: < 1e-5
//   STRATA_GLM_POOL_CLASSES=1         Maya's uniform slots: < 1e-5
//   STRATA_GLM_NO_STAGE + _BOUNCE=1   the pinned bounce ring (it copied slot_bytes_ into smaller class slots before
//                                     B3.7, overrunning the next slot): < 1e-5
//   L9 vs ref/glm_dsa.py --mode hf    mean KL < 1e-2 and top-1 >= 11 / 12 (the slow path's GPU experts take q8_1
//                                     activations; glmd_layer_parity uses the same bound)
// In an ASan build (docs/strata/engine/asan.md) the same run checks the host side of the pool (class vectors, the
// plan, the staging buffers) for out-of-bounds accesses.  Exits 77 without a device or without the fixture tools.
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_pool_classes.hpp"

#include "glm_dsa_fixtures.hpp"
#include "glm_test_env.hpp"   // setenv / unsetenv (<unistd.h> on POSIX; mapped on Windows)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr int kSkip = 77;
constexpr int kTokens = 12;
constexpr int64_t kCtx = 64;
constexpr double kSameGate = 1e-5;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

const char* kEnv[] = {"STRATA_GLM_POOL_CLASSES", "STRATA_GLM_NO_STAGE", "STRATA_GLM_BOUNCE", "STRATA_GLM_CPU_EXPERTS",
                      "STRATA_GLM_POOL_SLOTS0"};

struct Case {
    std::string name;
    std::vector<std::pair<const char*, const char*>> env;
    int want_classes;
};

/// One slow-path run; logits per token.  false + err on a load / forward failure.
bool run(const fs::path& pack, const Case& c, const std::vector<int32_t>& tokens,
         std::vector<std::vector<float>>& out, int& classes, std::string& err) {
    for (const char* e : kEnv) unsetenv(e);
    for (const auto& kv : c.env) setenv(kv.first, kv.second, 1);
    setenv("STRATA_GLM_SLOW", "1", 1);
    setenv("STRATA_GLM_SPLIT", "0", 1);
    bool ok = false;
    {
        strata::core::Glm5Model m;
        ok = m.load_pack(pack.string(), kCtx, err);
        if (ok && m.fast()) {
            err = "STRATA_GLM_SLOW=1 loaded the fast path";
            ok = false;
        }
        classes = strata::core::glm_pool_classes_last();
        std::vector<float> lg;
        for (size_t t = 0; ok && t < tokens.size(); ++t) {
            ok = m.forward({tokens[t]}, lg, err);
            if (ok) out.push_back(lg);
        }
    }
    for (const char* e : kEnv) unsetenv(e);
    unsetenv("STRATA_GLM_SLOW");
    unsetenv("STRATA_GLM_SPLIT");
    return ok;
}

double max_abs(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0;
    for (size_t t = 0; t < a.size(); ++t) {
        if (a[t].size() != b[t].size()) return 1e30;
        for (size_t i = 0; i < a[t].size(); ++i) m = std::max(m, (double) std::fabs(a[t][i] - b[t][i]));
    }
    return m;
}

}  // namespace

int main() {
    std::printf("glm_dsa_pack9_slow_test: 9 expert size classes on the slow path (B3.7)\n");
    int n_dev = 0;   // first: a CPU box writes no fixtures
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glm_dsa_pack9_slow_test: SKIP - no CUDA device\n");
        return kSkip;
    }
    std::string why;
    const fs::path gguf = glmfx::synth("PACK9", why);
    const fs::path pack = gguf.empty() ? fs::path() : glmfx::pack(gguf, why);
    if (pack.empty()) {
        return glmfx::fixture_missing("glm_dsa_pack9_slow_test", why);
    }
    const int V = 1024;
    const std::vector<int32_t> tokens = glmfx::ref_tokens(kTokens, V);
    const fs::path olg = glmfx::fixture_dir() / "pack9-12.oracle.f32";
    std::vector<float> want;
    const bool have_oracle = glmfx::oracle(gguf, tokens, "hf", olg, fs::path(), why) && glmfx::read_all(olg, want) &&
                             want.size() == (size_t) kTokens * V;
    if (!have_oracle) std::printf("  (no oracle logits: %s - the oracle comparison is skipped)\n", why.c_str());

    // the order makes each load change glm_pool_classes_last(), so every case proves its pool was built
    const std::vector<Case> cases = {
        {"9 classes", {}, 9},
        {"STRATA_GLM_POOL_CLASSES=8", {{"STRATA_GLM_POOL_CLASSES", "8"}}, 8},
        {"9 classes, STRATA_GLM_NO_STAGE=1 (three-copy fill)", {{"STRATA_GLM_NO_STAGE", "1"}}, 9},
        {"STRATA_GLM_POOL_CLASSES=1 (uniform slots)", {{"STRATA_GLM_POOL_CLASSES", "1"}}, 1},
        {"9 classes, STRATA_GLM_BOUNCE=1 (pinned bounce ring)",
         {{"STRATA_GLM_NO_STAGE", "1"}, {"STRATA_GLM_BOUNCE", "1"}}, 9},
    };
    std::vector<std::vector<float>> base;
    for (const Case& c : cases) {
        std::vector<std::vector<float>> got;
        int classes = -1;
        std::string err;
        if (!run(pack, c, tokens, got, classes, err)) {
            check(false, c.name + ": " + err);
            continue;
        }
        check(classes == c.want_classes, c.name + ": the device expert pool has " + std::to_string(classes) +
                                             " size classes (want " + std::to_string(c.want_classes) + ")");
        if (base.empty()) {
            base = got;
            if (have_oracle) {
                const glmfx::LogitStats st = glmfx::compare_rows(got, want.data(), (size_t) V);
                check(st.kl_mean < 1e-2 && st.top1 >= (size_t) kTokens - 1,
                      c.name + " vs ref/glm_dsa.py (hf): " + st.line());
            }
            continue;
        }
        const double d = max_abs(got, base);
        char b[96];
        std::snprintf(b, sizeof b, "max |logit - 9 classes| %.3g (gate %.0e)", d, kSameGate);
        check(d < kSameGate, c.name + ": " + b);
    }
    std::printf("glm_dsa_pack9_slow_test: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
