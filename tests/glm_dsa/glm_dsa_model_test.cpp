// tests/glm_dsa/glm_dsa_model_test.cpp - the glm-dsa reference forward vs the Python oracle (glm-dsa B3.6).
//
// SMALL and LONG synthetic glm-dsa GGUFs (B3.2, tools/glm_dsa_synth_gguf.py) run token by token through the
// slow path (STRATA_GLM_SLOW=1: Glm5Model::load -> step_layers_dsa, src/core/glm_dsa_ref.cu), and every position's
// logits are compared with the oracle's (B3.1, ref/glm_dsa.py, numpy f64, HF semantics):
//
//   gate: KL(oracle || engine) p99 over the positions < 1e-4, plus the max abs logit error and the top-1 agreement
//         (reported; B3.5's own gate is max abs < 1e-4 and top-1 100% on SMALL/BIG).
//   LONG has more tokens than top_k (2048), so the indexer's selection is exercised, not only the dense regime.
//
// FIXTURES (B3.6 alignment with B3.1-B3.3): the GGUFs come from the real generator CLI and the references are the
// committed ones in data/glm-dsa/ (tests/glm_dsa/glm_dsa_fixtures.hpp):
//   python3 tools/glm_dsa_synth_gguf.py --preset SMALL|LONG --out glm-dsa-<preset>.gguf  (sha256 vs data/glm-dsa/sha256.txt)
//   SMALL, 48 tokens, HF semantics          vs data/glm-dsa/oracle-hf-small-48.f32 (ref/glm_dsa.py --mode hf)
//   SMALL, 48 tokens, STRATA_GLM_IDX_LN_EPS=0 (llama.cpp parity) vs data/glm-dsa/llama-small-48.f32
//   LONG, 2080 tokens, eps 0, rows 2048..2079 (top-2048 selection active) vs data/glm-dsa/llama-long-2080-tail32.f32
// Token i of every reference is (i * 37 + 11) % n_vocab.  Before any GPU work the test reruns the oracle's CLI
//   python3 ref/glm_dsa.py glm-dsa-small.gguf --tokens <48> --mode hf --logits-out X.f32 --mtp --mtp-out M.f32
// and requires X / M to equal the committed oracle-hf-small-48.f32 / oracle-hf-small-mtp-47.f32 byte for byte (the
// CLI, the generator and the references agree).  Without the tools, or without a GPU, the test says why and exits 77.
// The position offset 1,000,000 variant needs a start-position API on Glm5Model and is still open.
#include "strata/core/glm_model.hpp"

#include "glm_dsa_fixtures.hpp"
#include "glm_test_env.hpp"   // setenv / unsetenv (<unistd.h> on POSIX; mapped on Windows)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef STRATA_SOURCE_DIR
#define STRATA_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

namespace {

constexpr int kSkip = 77;

enum class Result { Pass, Fail, Skip };

/// One case: `n_tok` reference tokens through the slow path; rows [row0, n_tok) against `ref` (rows of n_vocab).
Result run_case(const char* name, const fs::path& gguf, int n_tok, int row0, const fs::path& ref, bool eps0) {
    std::vector<float> want;
    if (!glmfx::read_all(ref, want)) {
        std::printf("  %s: SKIP - %s is missing\n", name, ref.string().c_str());
        return Result::Skip;
    }
    if (eps0)
        setenv("STRATA_GLM_IDX_LN_EPS", "0", 1);   // read at load: the indexer LayerNorm as llama.cpp computes it
    else
        unsetenv("STRATA_GLM_IDX_LN_EPS");
    strata::core::Glm5Model m;
    std::string err;
    const bool loaded = m.load(gguf.string(), (int64_t) n_tok + 8, err);
    unsetenv("STRATA_GLM_IDX_LN_EPS");
    if (!loaded) {
        std::printf("  %s: FAIL - load: %s\n", name, err.c_str());
        return Result::Fail;
    }
    const size_t V = (size_t) m.geometry().n_vocab;
    if (m.geometry().arch != strata::core::GlmArch::GlmDsa || V == 0 || want.size() != (size_t) (n_tok - row0) * V) {
        std::printf("  %s: FAIL - not a glm-dsa model whose vocab (%zu) fits %s\n", name, V, ref.filename().string().c_str());
        return Result::Fail;
    }
    const std::vector<int32_t> tokens = glmfx::ref_tokens(n_tok, (int) V);
    std::vector<std::vector<float>> got;
    std::vector<float> lg;
    for (int t = 0; t < n_tok; ++t) {
        if (!m.forward({tokens[(size_t) t]}, lg, err)) {
            std::printf("  %s: FAIL - forward at %d: %s\n", name, t, err.c_str());
            return Result::Fail;
        }
        if (t >= row0) got.push_back(lg);
    }
    const glmfx::LogitStats st = glmfx::compare_rows(got, want.data(), V);
    const bool ok = st.kl_p99 < 1e-4;
    std::printf("  %s: %s - %s\n", name, ok ? "ok" : "FAIL", st.line().c_str());
    return ok ? Result::Pass : Result::Fail;
}

/// The oracle CLI reproduces the committed SMALL references (no GPU needed).
Result oracle_cli(const fs::path& small) {
    const fs::path dir = small.parent_path(), lg = dir / "small-48.oracle.f32", mt = dir / "small-mtp-47.oracle.f32";
    std::string why;
    if (!glmfx::oracle(small, glmfx::ref_tokens(48, 512), "hf", lg, mt, why)) {
        std::printf("  oracle CLI: SKIP - %s\n", why.c_str());
        return Result::Skip;
    }
    std::vector<uint8_t> a, b, c, d;
    const bool same = glmfx::read_all(lg, a) && glmfx::read_all(glmfx::data_dir() / "oracle-hf-small-48.f32", b) &&
                      a == b && glmfx::read_all(mt, c) &&
                      glmfx::read_all(glmfx::data_dir() / "oracle-hf-small-mtp-47.f32", d) && c == d;
    std::printf("  oracle CLI: %s - ref/glm_dsa.py --mode hf --mtp on SMALL %s the committed oracle-hf-small-48 / "
                "-mtp-47 references\n", same ? "ok" : "FAIL", same ? "reproduces" : "does NOT reproduce");
    return same ? Result::Pass : Result::Fail;
}

}  // namespace

int main() {
    std::printf("glm_dsa_model_test: synthetic glm-dsa GGUFs through the slow path vs the data/glm-dsa references (B3.6)\n");
    int pass = 0, fail = 0, skip = 0;
    const auto count = [&](Result r) {
        switch (r) {
        case Result::Pass: ++pass; break;
        case Result::Fail: ++fail; break;
        case Result::Skip: ++skip; break;
        }
    };
    std::string why;
    const fs::path small = glmfx::synth("SMALL", why);
    if (small.empty()) {
        return glmfx::fixture_missing("glm_dsa_model_test", why);
    }
    count(oracle_cli(small));
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glm_dsa_model_test: no CUDA device - the engine cases skipped\n");
        return fail > 0 ? 1 : kSkip;
    }
    setenv("STRATA_GLM_SLOW", "1", 1);   // the reference forward (glm_forward_available lets glm-dsa through)
    const fs::path data = glmfx::data_dir();
    count(run_case("SMALL hf, 48 tokens", small, 48, 0, data / "oracle-hf-small-48.f32", false));
    count(run_case("SMALL eps 0 vs llama.cpp, 48 tokens", small, 48, 0, data / "llama-small-48.f32", true));
    const fs::path lng = glmfx::synth("LONG", why);
    if (lng.empty()) {
        std::printf("  LONG: SKIP - %s\n", why.c_str());
        ++skip;
    } else {
        count(run_case("LONG eps 0 vs llama.cpp, 2080 tokens, rows 2048..2079", lng, 2080, 2048,
                       data / "llama-long-2080-tail32.f32", true));
    }
    std::printf("glm_dsa_model_test: %d passed, %d failed, %d skipped\n", pass, fail, skip);
    if (fail > 0) return 1;
    return pass == 0 ? kSkip : 0;
}
