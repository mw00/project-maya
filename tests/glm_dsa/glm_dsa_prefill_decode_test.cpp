// tests/glm_dsa/glm_dsa_prefill_decode_test.cpp - glm-dsa wave 5 (label gpu): the batched glm-dsa prompt path and the
// decode mixer write and read the SAME caches.
//
// On a synthetic pack (PACK9: 12 layers + NextN, real MLA head geometry, top-k 2048), two models load the same pack on
// one GPU with the NextN block (STRATA_GLM_MTP=1):
//   A  the batched prompt: prefill(prompt[:-1], next = prompt.back()) (glm_prefill_dsa.cu, including the NextN cache
//      fill mtp_fill), then forward(prompt.back());
//   B  the token path: forward(t) for every prompt token, each followed by mtp_draft(next) - the NextN block at that
//      position, as fast_mtp fills it - then forward(prompt.back()).
// Then both decode the same teacher-forced tokens (A's greedy choice) with mtp_draft after each.  The prompt is longer
// than top_k, so the decode crosses into the selecting regime with keys the prompt path wrote.  Checked:
//   * caches (dump_state, read through the bound state layout): for every DSA layer held (trunk + NextN), the rows of
//     every written position are non-zero in both runs and every later row is untouched (zero) in both - a cache
//     written at a wrong position, in a wrong dtype or by one path only fails here exactly; and the written rows agree
//     in value (rel L2 per layer and component < 5e-2: the prompt's FP16 GEMMs vs the token path's q8_1 GEMVs);
//   * logits: per step rel L2 < 0.15 and top-1 agreement >= 90% over the decode steps; NextN drafts agree >= 80%.
// Variants (ctest): the default; STRATA_GLM_IDX_FP16=1 (FP16 index keys end to end, B10.8); every Volta switch forced
// with decode graphs (STRATA_GLMD_FUSED_MLA / IDX_WMMA / TOPK_MB = 1, STRATA_GLM_DSA_GRAPHS=1, B10.1-B10.4); and
// --native32: the pack keeps attn_k_b / attn_v_b native Q8_0 (STRATA_PACK_NATIVE32=1, B10.7).
// Without a device or the fixture tools: exit 77.
#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_state_layout.hpp"

#include "glm_dsa_fixtures.hpp"
#include "glm_test_env.hpp"   // setenv / unsetenv (<unistd.h> on POSIX; mapped on Windows)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr int kSkip = 77;
constexpr int64_t kCtx = 4096;
constexpr int kDecode = 24;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

int argmax(const std::vector<float>& v) { return (int) (std::max_element(v.begin(), v.end()) - v.begin()); }

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0, n = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        e += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        n += (double) b[i] * b[i];
    }
    return std::sqrt(e / std::max(n, 1e-300));
}

float half_f(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        const float f = std::ldexp((float) m, -24);
        return s ? -f : f;
    }
    if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// <stem>.n32.pack: the pack with STRATA_PACK_NATIVE32=1 (native Q8_0 W_kb / W_vb).
fs::path pack_native32(const fs::path& gguf, std::string& why) {
    const fs::path out = gguf.parent_path() / (gguf.stem().string() + ".n32.pack"), log = gguf.parent_path() / "fixtures.log";
    const fs::path mark = out / ".from";
    {
        std::ifstream f(mark);
        std::string from;
        std::getline(f, from);
        if (from == gguf.string() && fs::exists(out / "index.txt") && fs::last_write_time(mark) >= fs::last_write_time(gguf))
            return out;
    }
    std::error_code ec;
    fs::remove_all(out, ec);
#ifdef _WIN32
    // (cmd.exe has no VAR=value command prefix: the tool inherits the switch from this process instead)
    setenv("STRATA_PACK_NATIVE32", "1", 1);
    const bool packed = glmfx::run(glmfx::python() + " " + glmfx::q(glmfx::src_dir() / "tools" / "iq_pack.py") +
                                       " --gguf " + glmfx::q(gguf) + " --out " + glmfx::q(out),
                                   log);
    unsetenv("STRATA_PACK_NATIVE32");
    if (!packed) {
#else
    if (!glmfx::run("STRATA_PACK_NATIVE32=1 " + glmfx::python() + " " + glmfx::q(glmfx::src_dir() / "tools" / "iq_pack.py") +
                        " --gguf " + glmfx::q(gguf) + " --out " + glmfx::q(out),
                    log)) {
#endif
        why = "tools/iq_pack.py (STRATA_PACK_NATIVE32=1) failed (see " + log.string() + ")";
        return {};
    }
    std::ofstream(mark) << gguf.string() << "\n";
    return out;
}

bool read_floats(const fs::path& p, std::vector<float>& out) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto n = (size_t) f.tellg();
    out.resize(n / 4);
    f.seekg(0);
    f.read((char*) out.data(), (std::streamsize) (out.size() * 4));
    return (bool) f;
}

struct Run {
    std::vector<std::vector<float>> logits;   // after the prompt, then after each decode token
    std::vector<int> drafts;
    std::vector<float> state;
    std::string err;
};

/// One cache component's rows [0, rows_total) as floats (FP16 / F32 / INT8 decoded), `width` values a row.
std::vector<float> rows_of(const std::vector<float>& st, int64_t off_floats, int64_t pp_floats, int width, int dtype,
                           int64_t n_rows) {
    // dtype: 0 F32, 1 FP16, 2 INT8 latent record (kv_lora int8 then kv_lora / 32 FP16 scales)
    std::vector<float> v((size_t) n_rows * width);
    const uint8_t* base = (const uint8_t*) (st.data() + off_floats);
    for (int64_t r = 0; r < n_rows; ++r) {
        const uint8_t* row = base + (size_t) r * pp_floats * 4;
        for (int c = 0; c < width; ++c) {
            float x;
            if (dtype == 0) std::memcpy(&x, row + 4 * c, 4);
            else if (dtype == 1) {
                uint16_t h;
                std::memcpy(&h, row + 2 * c, 2);
                x = half_f(h);
            } else {
                uint16_t d;
                std::memcpy(&d, row + width + 2 * (c / 32), 2);
                x = (float) ((const int8_t*) row)[c] * half_f(d);
            }
            v[(size_t) r * width + c] = x;
        }
    }
    return v;
}

bool row_written(const std::vector<float>& st, int64_t off_floats, int64_t pp_floats, int64_t r) {
    const float* p = st.data() + off_floats + r * pp_floats;
    for (int64_t i = 0; i < pp_floats; ++i)
        if (p[i] != 0.0f) return true;
    return false;
}

bool run_model(const fs::path& pack, bool batched, const std::vector<int32_t>& prompt, const std::vector<int>* forced,
               Run& out, strata::core::GlmStateLayout& lay, int64_t& pos_end) {
    if (batched) unsetenv("STRATA_GLM_NO_PREFILL");
    else setenv("STRATA_GLM_NO_PREFILL", "1", 1);
    auto m = std::make_unique<strata::core::Glm5Model>();
    std::string& err = out.err;
    if (!m->load_pack_env(pack.string(), kCtx, err) || !m->fast()) {
        if (err.empty()) err = "not on the fast path";
        return false;
    }
    if (!m->has_mtp()) {
        err = "no NextN block (STRATA_GLM_MTP=1)";
        return false;
    }
    m->reset();
    std::vector<float> logits;
    const std::vector<int32_t> head(prompt.begin(), prompt.end() - 1);
    if (batched) {
        if (!m->prefill(head, err, prompt.back())) {
            if (err.empty()) err = "the batched prompt path did not run (prefill returned false)";
            return false;
        }
    } else {
        for (size_t i = 0; i < head.size(); ++i) {
            if (!m->forward({head[i]}, logits, err)) return false;
            if (m->mtp_draft(i + 1 < head.size() ? head[i + 1] : prompt.back(), err) < 0 && !err.empty()) return false;
        }
    }
    if (!m->forward({prompt.back()}, logits, err)) return false;
    out.logits.push_back(logits);
    for (int k = 0; k < kDecode; ++k) {
        const int t = forced ? (*forced)[(size_t) k] : argmax(logits);
        const int d = m->mtp_draft(t, err);   // the NextN block at this position, then the next token
        if (d < 0 && !err.empty()) return false;
        out.drafts.push_back(d);
        if (!m->forward({t}, logits, err)) return false;
        out.logits.push_back(logits);
    }
    lay = m->state_layout();
    const fs::path dump = glmfx::fixture_dir() / (std::string("prefill_decode_") + (batched ? "a" : "b"));
    if (!m->dump_state(dump.string()) || !read_floats(dump.string() + ".0.bin", out.state)) {
        err = "dump_state failed";
        return false;
    }
    {
        std::ifstream t(dump.string() + ".0.txt");
        std::string w;
        long long mc = 0, pos = 0;
        t >> w >> mc >> w >> pos;
        pos_end = pos;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const bool native32 = argc > 1 && std::string(argv[1]) == "--native32";
    // --iq1m: PFIQ1M, a geometry the batched prompt path covers (PACK9's 4 heads run it token by token) with IQ1_M
    // experts in gate/up, in down and in both - the matrices MMQ has no instance for take the FP16 expert products
    const bool iq1m = argc > 1 && std::string(argv[1]) == "--iq1m";
    std::printf("glm_dsa_prefill_decode_test: the batched prompt vs the token path, then decode%s\n",
                native32 ? " (native Q8_0 W_kb / W_vb pack)" : iq1m ? " (IQ1_M experts, the batched geometry)" : "");
    std::string why, err;
    const fs::path gguf = glmfx::synth(iq1m ? "PFIQ1M" : "PACK9", why);
    fs::path pack = gguf.empty() ? fs::path() : (native32 ? pack_native32(gguf, why) : glmfx::pack(gguf, why));
    if (pack.empty()) {
        return glmfx::fixture_missing("glm_dsa_prefill_decode_test", why);
    }
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glm_dsa_prefill_decode_test: no CUDA device - SKIP\n");
        return kSkip;
    }
    strata::core::Glm5Geometry g;
    if (!strata::core::glm_pack_geometry(pack.string(), g, err) || g.arch != strata::core::GlmArch::GlmDsa) {
        std::fprintf(stderr, "glm_dsa_prefill_decode_test: %s: %s\n", pack.string().c_str(), err.c_str());
        return 1;
    }
    unsetenv("STRATA_GLM_SLOW");
    unsetenv("STRATA_GLM_DSA_DENSE");
    setenv("STRATA_GLM_SPLIT", "0", 1);
    setenv("STRATA_GLM_MTP", "1", 1);
    setenv("STRATA_GLM_PREFILL_MIN", "1", 1);
    // (PFIQ1M's pool holds every expert, so it lends the prompt path nothing by itself: a fixed 512 MB, two chunks)
    if (iq1m) setenv("STRATA_GLM_PREFILL_MB", "512", 0);
    for (const char* v : {"STRATA_GLM_IDX_FP16", "STRATA_GLMD_FUSED_MLA", "STRATA_GLMD_IDX_WMMA", "STRATA_GLMD_TOPK_MB",
                          "STRATA_GLM_DSA_GRAPHS", "STRATA_GLM_PREFILL_F16_ROWS"})
        if (const char* x = getenv(v)) std::printf("  (%s=%s)\n", v, x);
    // a prompt past top_k: the decode selects among keys the prompt path wrote
    const int L = std::min<int>((int) kCtx - kDecode - 8, g.idx_top_k + 64);
    std::vector<int32_t> prompt((size_t) L);
    for (int i = 0; i < L; ++i) prompt[(size_t) i] = (int32_t) ((i * 7919 + 13) % std::max<int64_t>(2, g.n_vocab));

    Run A, B;
    strata::core::GlmStateLayout la, lb;
    int64_t pa = 0, pb = 0;
    if (!run_model(pack, true, prompt, nullptr, A, la, pa)) {
        std::fprintf(stderr, "glm_dsa_prefill_decode_test: the batched run: %s\n", A.err.c_str());
        return 1;
    }
    std::vector<int> forced;
    for (int k = 0; k < kDecode; ++k) forced.push_back(argmax(A.logits[(size_t) k]));
    if (!run_model(pack, false, prompt, &forced, B, lb, pb)) {
        std::fprintf(stderr, "glm_dsa_prefill_decode_test: the token run: %s\n", B.err.c_str());
        return 1;
    }
    check(pa == pb && pa == L + kDecode, "both runs end at position " + std::to_string(pa) + " / " + std::to_string(pb));
    check(la.bytes == lb.bytes && la.ik_bytes == lb.ik_bytes && la.lat_pp == lb.lat_pp && la.mode == lb.mode,
          "the same state layout (index keys " + std::to_string(la.ik_bytes) + " B / element)");
    // ---- the caches
    const int64_t written = pa;                    // trunk rows 0 .. pos - 1
    int bad_pos = 0, bad_val = 0, layers = 0;
    double worst = 0;
    std::string worst_what;
    for (int il : la.dsa_layers()) {
        if (!la.holds(il)) continue;
        ++layers;
        // the NextN block's rows: mtp_draft ran at every position but the last decode token's
        const int64_t rows = il == g.n_layers ? written - 1 : written;
        struct Comp {
            const char* name;
            int64_t off, pp;
            int width, dtype;
        };
        std::vector<Comp> comps;
        comps.push_back({"latent", la.dsa_lat[(size_t) il], la.lat_pp, g.kv_lora, la.mode == strata::core::GlmStateMode::Int8 ? 2 : 1});
        if (la.has_kpe(il)) comps.push_back({"k_pe", la.dsa_kpe[(size_t) il], la.kpe_pp, g.qk_rope, 1});
        if (la.has_index_cache(il)) comps.push_back({"index keys", la.dsa_ik[(size_t) il], la.ik_pp, g.idx_key, la.ik_bytes == 2 ? 1 : 0});
        for (const Comp& c : comps) {
            for (int64_t r = 0; r < std::min<int64_t>(rows + 4, la.max_ctx); ++r) {
                const bool wa = row_written(A.state, c.off, c.pp, r), wb = row_written(B.state, c.off, c.pp, r);
                if (wa != (r < rows) || wb != (r < rows)) ++bad_pos;
            }
            const std::vector<float> va = rows_of(A.state, c.off, c.pp, c.width, c.dtype, rows),
                                     vb = rows_of(B.state, c.off, c.pp, c.width, c.dtype, rows);
            const double e = rel_l2(va, vb);
            if (!(e < 5e-2)) ++bad_val;
            if (!(e <= worst)) {
                worst = e;
                worst_what = "layer " + std::to_string(il) + " " + c.name;
            }
        }
    }
    check(layers > 0 && bad_pos == 0, "every DSA layer's latent / k_pe / index-key rows are written at exactly the "
                                      "positions 0..p by both paths (" + std::to_string(layers) + " layers, " +
                                      std::to_string(bad_pos) + " wrong rows)");
    char buf[256];
    std::snprintf(buf, sizeof buf, "cache values agree: worst rel L2 %.2e (%s), %d components over 5e-2", worst,
                  worst_what.c_str(), bad_val);
    check(bad_val == 0, buf);
    // ---- the logits and the drafts
    int top1 = 0, drafts = 0;
    double max_rel = 0;
    for (size_t k = 0; k < A.logits.size(); ++k) {
        max_rel = std::max(max_rel, rel_l2(B.logits[k], A.logits[k]));
        top1 += argmax(A.logits[k]) == argmax(B.logits[k]) ? 1 : 0;
    }
    for (size_t k = 0; k < A.drafts.size(); ++k) drafts += A.drafts[k] == B.drafts[k] ? 1 : 0;
    std::snprintf(buf, sizeof buf, "prompt + %d decode steps: logits max rel L2 %.3f, top-1 agree %d / %zu", kDecode,
                  max_rel, top1, A.logits.size());
    check(max_rel < 0.15 && top1 * 10 >= (int) A.logits.size() * 9, buf);
    std::snprintf(buf, sizeof buf, "NextN drafts agree %d / %zu", drafts, A.drafts.size());
    check(drafts * 10 >= (int) A.drafts.size() * 8, buf);
    if (iq1m) {   // the token path again would give the same logits bit for bit (prefill_tokens_dsa's fallback)
        const double e = rel_l2(B.logits[0], A.logits[0]);
        std::snprintf(buf, sizeof buf, "the batched prompt path ran (logits after the prompt differ from the token "
                                       "path's: rel L2 %.2e > 0)", e);
        check(e > 0, buf);
    }
    if (g_fail) {
        std::printf("glm_dsa_prefill_decode_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_dsa_prefill_decode_test: ok\n");
    return 0;
}
