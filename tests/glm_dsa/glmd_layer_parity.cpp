// tests/glm_dsa/glmd_layer_parity.cpp - glm-dsa B4.7 layer parity + B11.1 NextN (label gpu): the glm-dsa fast decode
// path against the F32 reference forward, layer by layer, on the synthetic PACK preset.
//
// Fixture (tests/glm_dsa/glm_dsa_fixtures.hpp): tools/glm_dsa_synth_gguf.py --preset PACK (6 layers + NextN, full
// indexer layers 0, 1, 2, dense layers 0, 1, community quant types), packed by tools/iq_pack.py (glm-dsa mode), and
// ref/glm_dsa.py --mode hf --mtp --mtp-out on the 12 reference tokens ((i * 37 + 11) % 1024).
//
// Both paths run the 12 tokens with GLM_CB_DIR seam dumps (the same llama.cpp cb names on both paths):
//   slow   STRATA_GLM_SLOW=1 load_pack -> step_layers_dsa (src/core/glm_dsa_ref.cu), F32 activations and caches;
//   fast   load_pack -> fast_layers_plain + fast_dsa_glm (src/core/glm_dsa_mixer.cu), q8_1 activations, FP16 latents,
//          with the NextN block (STRATA_GLM_MTP=1 on one part) drafting after every token.
// Gates (per seam, mean rel L2 over the positions; the fast path quantizes its activations to q8_1 as llama.cpp
// does, which the oracle README measures at a mean KL of 1e-3..5e-3 on this preset):
//   full + dense  (layer 0, 1)   attn_out, ffn_out, l_out  < 5e-2
//   full + MoE    (layer 2)      attn_out, ffn_out, l_out  < 5e-2
//   shared + MoE  (layer 3..5)   attn_out, ffn_out, l_out  < 5e-2
//   logits        fast vs slow and fast vs the oracle: mean KL < 1e-2, top-1 >= 11 / 12
//   NextN         the fast path's draft (mtp_draft) equals the oracle's draft argmax on >= 9 / 11 rows (the oracle
//                 feeds hnorm the post-output-norm hidden, as llama.cpp's t_h_nextn and Glm5Model's head_x do)
//   dsa_n_sel_last = p + 1 after every token (top_k 2048 > 12: the identity selection).
// Exits 77 without a device or without the fixture tools.
#include "strata/core/glm_model.hpp"

#include "glm_dsa_fixtures.hpp"
#include "glm_test_env.hpp"   // setenv / unsetenv (<unistd.h> on POSIX; mapped on Windows)

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

constexpr int kTokens = 12;
constexpr int64_t kCtx = 64;
const char* kSeams[] = {"attn_out", "ffn_out", "l_out"};

using Dumps = std::map<std::string, std::vector<float>>;   // "<seam>-<layer>" -> values

Dumps read_dumps(const fs::path& dir, int n_layers) {
    Dumps d;
    for (int il = 0; il < n_layers; ++il)
        for (const char* s : kSeams) {
            const std::string name = std::string(s) + "-" + std::to_string(il);
            std::vector<float> v;
            if (glmfx::read_all(dir / (name + ".f32"), v)) d[name] = std::move(v);
        }
    return d;
}

void clear_dumps(const fs::path& dir) {
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) fs::remove(e.path(), ec);
}

struct Run {
    std::vector<Dumps> dumps;                  // per token
    std::vector<std::vector<float>> logits;    // per token
    std::vector<int> drafts;                   // per token t < T-1: the NextN draft with token t+1
    std::vector<int> n_sel;                    // dsa_n_sel_last after each token
    std::string err;
};

bool run(bool slow, const fs::path& pack, const fs::path& cb, const std::vector<int32_t>& tokens, int n_layers, Run& r) {
    if (slow)
        setenv("STRATA_GLM_SLOW", "1", 1);
    else
        unsetenv("STRATA_GLM_SLOW");
    strata::core::Glm5Model m;
    const bool ok = m.load_pack(pack.string(), kCtx, r.err);
    unsetenv("STRATA_GLM_SLOW");
    if (!ok) return false;
    if (slow == m.fast()) {
        r.err = slow ? "STRATA_GLM_SLOW=1 still loaded the fast path" : "the fast path did not load";
        return false;
    }
    if (!slow && !m.has_mtp()) {
        r.err = "the fast path loaded without the NextN block (STRATA_GLM_MTP=1)";
        return false;
    }
    std::vector<float> lg;
    for (size_t t = 0; t < tokens.size(); ++t) {
        clear_dumps(cb);
        if (!m.forward({tokens[t]}, lg, r.err)) return false;
        r.dumps.push_back(read_dumps(cb, n_layers));
        r.logits.push_back(lg);
        if (!slow) {
            r.n_sel.push_back(m.dsa_n_sel_last());
            if (t + 1 < tokens.size()) {
                const int d = m.mtp_draft(tokens[t + 1], r.err);
                if (d < 0) return false;
                r.drafts.push_back(d);
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    std::printf("glmd_layer_parity: the glm-dsa fast path vs the F32 reference, per layer, on PACK (B4.7, B11.1)\n");
    std::string why;
    const fs::path gguf = glmfx::synth("PACK", why);
    const fs::path pack = gguf.empty() ? fs::path() : glmfx::pack(gguf, why);
    if (pack.empty()) return glmfx::fixture_missing("glmd_layer_parity", why);
    const int V = 1024, n_layers = 6;
    const std::vector<int32_t> tokens = glmfx::ref_tokens(kTokens, V);
    const fs::path dir = glmfx::fixture_dir(), olg = dir / "pack-12.oracle.f32", omtp = dir / "pack-mtp-11.oracle.f32";
    if (!glmfx::oracle(gguf, tokens, "hf", olg, omtp, why)) return glmfx::fixture_missing("glmd_layer_parity", why);
    std::vector<float> want, want_mtp;
    if (!glmfx::read_all(olg, want) || want.size() != (size_t) kTokens * V || !glmfx::read_all(omtp, want_mtp) ||
        want_mtp.size() != (size_t) (kTokens - 1) * V) {
        std::printf("glmd_layer_parity: FAIL - unreadable oracle output in %s\n", dir.string().c_str());
        return 1;
    }
    {   // the pack passes the glm-dsa loader's shape checks (no GPU work)
        setenv("STRATA_GLM_CHECK_ONLY", "1", 1);
        strata::core::Glm5Model m;
        std::string e;
        m.load_pack(pack.string(), kCtx, e);
        unsetenv("STRATA_GLM_CHECK_ONLY");
        check(e.find("check-only load") != std::string::npos, "PACK passes the loader's shape checks: " + e);
    }
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glmd_layer_parity: no CUDA device - the engine steps skipped\n");
        return g_fail ? 1 : 77;
    }
    const fs::path cb = dir / "layer_parity_cb";
    fs::create_directories(cb);
    setenv("GLM_CB_DIR", cb.string().c_str(), 1);   // (read once per translation unit: set before any model work)
    setenv("STRATA_GLM_MTP", "1", 1);       // the NextN block on a single part
    unsetenv("STRATA_GLM_DSA_DENSE");
    unsetenv("STRATA_GLMD_KQ_SCALE");

    Run slow, fast;
    if (!run(true, pack, cb, tokens, n_layers, slow)) {
        std::printf("glmd_layer_parity: FAIL - slow path: %s\n", slow.err.c_str());
        return 1;
    }
    if (!run(false, pack, cb, tokens, n_layers, fast)) {
        std::printf("glmd_layer_parity: FAIL - fast path: %s\n", fast.err.c_str());
        return 1;
    }
    unsetenv("GLM_CB_DIR");

    // ---- per layer
    struct Kind {
        const char* name;
        std::vector<int> layers;
    };
    const Kind kinds[] = {{"full + dense", {0, 1}}, {"full + MoE", {2}}, {"shared + MoE", {3, 4, 5}}};
    constexpr double kSeamTol = 5e-2;
    for (const Kind& k : kinds)
        for (int il : k.layers)
            for (const char* s : kSeams) {
                const std::string name = std::string(s) + "-" + std::to_string(il);
                double sum = 0, worst = 0;
                int n = 0, missing = 0;
                for (int t = 0; t < kTokens; ++t) {
                    const auto a = fast.dumps[(size_t) t].find(name), b = slow.dumps[(size_t) t].find(name);
                    if (a == fast.dumps[(size_t) t].end() || b == slow.dumps[(size_t) t].end()) {
                        ++missing;
                        continue;
                    }
                    const double e = glmfx::rel_l2(a->second, b->second);
                    sum += e;
                    worst = std::max(worst, e);
                    ++n;
                }
                char buf[256];
                std::snprintf(buf, sizeof buf, "%-13s layer %d %-8s: mean rel L2 %.2e, worst %.2e over %d positions%s",
                              k.name, il, s, n ? sum / n : 0.0, worst, n, missing ? " (dumps MISSING)" : "");
                check(missing == 0 && n > 0 && sum / n < kSeamTol, buf);
            }

    // ---- logits
    std::vector<float> slow_flat;
    for (const auto& r : slow.logits) slow_flat.insert(slow_flat.end(), r.begin(), r.end());
    const glmfx::LogitStats fs_ = glmfx::compare_rows(fast.logits, slow_flat.data(), V);
    check(fs_.kl_mean < 1e-2 && fs_.top1 + 1 >= (size_t) kTokens, "logits fast vs slow: " + fs_.line());
    const glmfx::LogitStats fo = glmfx::compare_rows(fast.logits, want.data(), V);
    check(fo.kl_mean < 1e-2 && fo.top1 + 1 >= (size_t) kTokens, "logits fast vs the oracle: " + fo.line());
    const glmfx::LogitStats so = glmfx::compare_rows(slow.logits, want.data(), V);
    std::printf("  (slow vs the oracle: %s)\n", so.line().c_str());

    // ---- the selection count and the NextN block
    int bad_sel = 0;
    for (int t = 0; t < kTokens; ++t) bad_sel += fast.n_sel[(size_t) t] == t + 1 ? 0 : 1;
    check(bad_sel == 0, "dsa_n_sel_last = p + 1 after every token (identity selection below top-2048)");
    int agree = 0;
    for (int t = 0; t + 1 < kTokens; ++t)
        agree += fast.drafts[(size_t) t] == (int) glmfx::argmax(want_mtp.data() + (size_t) t * V, V) ? 1 : 0;
    check(agree >= 9, "NextN draft (fast path) vs ref/glm_dsa.py --mtp argmax: " + std::to_string(agree) + " / " +
                          std::to_string(kTokens - 1));

    if (g_fail) {
        std::printf("glmd_layer_parity: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glmd_layer_parity: ok\n");
    return 0;
}
