// tests/glm_dsa/glm_dsa_kv_stream_test.cpp - KV streaming on glm-dsa (label gpu): a streamed model gives the resident
// model's logits BIT FOR BIT through a whole conversation.
//
// Synthetic packs (PFIQ1M: 4 layers + NextN on the batched prompt path; PACK9: 12 layers + NextN, token-path prompts,
// three indexer groups with shared layers), a context of 8192 and a VRAM window of 3072 positions
// (MAYA_GLM_KV_RESIDENT_TEST_MIN) - above the indexer's top-k of 2048, far below the context.  Every expert fits in
// VRAM, so the only difference between two loads is where the attention's cache rows live.  Each load runs one script:
//   1  a prompt of 2500 tokens (past top-k, inside the window), 24 decoded tokens with the NextN draft after each;
//   2  the snapshot there, saved to a slot file;
//   3  a second turn of 700 tokens - the conversation leaves the window - and 24 more tokens;
//   4  reset, the slot back (inside the window again), 16 tokens;
//   5  the snapshot restored (an edited last turn): another 300-token turn, 12 tokens;
//   6  reset, one prompt of 5000 tokens (through the window in one read), 16 tokens;
//   7  that conversation saved, reset, loaded (longer than the window: every block evicted), 12 tokens.
// The tokens are forced (the resident run's greedy picks), so every load computes the same positions.
// Loads: resident (the reference), then streamed with MAYA_GLM_KV_GROUP x MAYA_GLM_KV_WINDOW = 0/1 x 0/1 (0, 0 is
// v1.1.67's streaming), each with MAYA_GLM_KV_DEBUG=1; and resident / streamed again with MAYA_GLM_DSA_GRAPHS=1.
// Checked per streamed load: every logits vector equals the reference's bit for bit, the NextN drafts equal it, the
// debug counters saw rows and no mismatch, no resolve overflowed, and (window on) phase 1 resolved nothing.
// With 3 devices: PACK9 on a 3-part split, the speculative decode's tokens resident vs streamed.
// Without a device or the fixture tools: exit 77.  `--fixtures` only makes the synthetic packs (no device needed).
#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_dsa_split.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/sampler.hpp"

#include "glm_dsa_fixtures.hpp"
#include "glm_test_env.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using strata::core::Glm5Model;

constexpr int kSkip = 77;
constexpr int64_t kCtx = 8192;
constexpr const char* kWindow = "3072";

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++g_fail;
}

int argmax(const std::vector<float>& v) { return (int) (std::max_element(v.begin(), v.end()) - v.begin()); }

std::vector<int32_t> toks(int n, int salt, int64_t n_vocab) {
    std::vector<int32_t> t((size_t) n);
    for (int i = 0; i < n; ++i) t[(size_t) i] = (int32_t) (((int64_t) i * 7919 + 13 + (int64_t) salt * 104729) % std::max<int64_t>(2, n_vocab));
    return t;
}

struct Trace {
    std::vector<std::vector<float>> logits;   // every forward's, in script order
    std::vector<int> drafts;
    std::vector<int> phase_end;               // logits.size() after each phase
    Glm5Model::KvStreamStats kv, kv_phase1;
    int64_t window = 0;
    std::string err;
};

struct Cfg {
    const char* name;
    bool streamed, group, window, graphs;
};

void set_env(const Cfg& c) {
    if (c.streamed) {
        setenv("MAYA_GLM_KV_RESIDENT", kWindow, 1);
        setenv("MAYA_GLM_KV_RESIDENT_TEST_MIN", kWindow, 1);
        setenv("MAYA_GLM_KV_GROUP", c.group ? "1" : "0", 1);
        setenv("MAYA_GLM_KV_WINDOW", c.window ? "1" : "0", 1);
        setenv("MAYA_GLM_KV_DEBUG", "1", 1);
    } else {
        for (const char* v : {"MAYA_GLM_KV_RESIDENT", "MAYA_GLM_KV_RESIDENT_TEST_MIN", "MAYA_GLM_KV_GROUP",
                              "MAYA_GLM_KV_WINDOW", "MAYA_GLM_KV_DEBUG"})
            unsetenv(v);
    }
    if (c.graphs) setenv("MAYA_GLM_DSA_GRAPHS", "1", 1);
    else unsetenv("MAYA_GLM_DSA_GRAPHS");
}

// a prompt turn: all but its last token through the prompt path (the batched one where the pack's geometry has it, the
// token path else - prefill() falls back by itself), then the last token's forward
bool turn(Glm5Model& m, const std::vector<int32_t>& t, Trace& tr, int chunk) {
    std::vector<float> logits;
    const std::vector<int32_t> head(t.begin(), t.end() - 1);
    m.prefill_force_chunk = chunk;
    if (!m.prefill(head, tr.err, t.back())) {
        if (!tr.err.empty()) return false;
        for (size_t i = 0; i < head.size(); ++i) {   // (no batched prompt path here)
            if (!m.forward({head[i]}, logits, tr.err)) return false;
            if (m.mtp_draft(i + 1 < head.size() ? head[i + 1] : t.back(), tr.err) < 0 && !tr.err.empty()) return false;
        }
    }
    if (!m.forward({t.back()}, logits, tr.err)) return false;
    tr.logits.push_back(logits);
    return true;
}

// n decoded tokens: the reference's picks when given (forced), else greedy; the NextN draft after each
bool decode(Glm5Model& m, int n, const Trace* ref, Trace& tr) {
    std::vector<float> logits = tr.logits.back();
    for (int k = 0; k < n; ++k) {
        const int t = ref != nullptr ? argmax(ref->logits[tr.logits.size() - 1]) : argmax(logits);
        const int d = m.mtp_draft(t, tr.err);
        if (d < 0 && !tr.err.empty()) return false;
        tr.drafts.push_back(d);
        if (!m.forward({t}, logits, tr.err)) return false;
        tr.logits.push_back(logits);
    }
    return true;
}

bool script(const fs::path& pack, const Cfg& c, int64_t n_vocab, const Trace* ref, Trace& tr) {
    set_env(c);
    setenv("MAYA_GLM_SPLIT", "0", 1);
    auto m = std::make_unique<Glm5Model>();
    std::string& err = tr.err;
    if (!m->load_pack_env(pack.string(), kCtx, err) || !m->fast() || !m->has_mtp()) {
        if (err.empty()) err = "not on the fast path, or no NextN block";
        return false;
    }
    tr.window = m->kv_resident();
    const fs::path slot = glmfx::fixture_dir() / (std::string("kv_stream_test_") + c.name + ".slot");
    const fs::path slot2 = glmfx::fixture_dir() / (std::string("kv_stream_test_") + c.name + ".long.slot");
    const auto done = [&]() { tr.phase_end.push_back((int) tr.logits.size()); };
    m->reset();
    // 1
    if (!turn(*m, toks(2500, 1, n_vocab), tr, 1024) || !decode(*m, 24, ref, tr)) return false;
    done();
    tr.kv_phase1 = m->kv_stream_stats();
    // 2
    const int64_t n1 = m->position();
    if (!m->snapshot_save() || m->slot_save(slot.string(), err) == 0) {
        if (err.empty()) err = "snapshot / slot save";
        return false;
    }
    // 3
    if (!turn(*m, toks(700, 2, n_vocab), tr, 512) || !decode(*m, 24, ref, tr)) return false;
    done();
    // 4
    m->reset();
    if (!m->slot_load(slot.string(), n1, err)) return false;
    {
        std::vector<float> logits;
        if (!m->forward({(int32_t) (17 % n_vocab)}, logits, err)) return false;
        tr.logits.push_back(logits);
    }
    if (!decode(*m, 16, ref, tr)) return false;
    done();
    // 5
    if (!m->snapshot_restore() || m->position() != n1) {
        err = "snapshot restore";
        return false;
    }
    if (!turn(*m, toks(300, 3, n_vocab), tr, 0) || !decode(*m, 12, ref, tr)) return false;
    done();
    // 6
    m->reset();
    if (!turn(*m, toks(5000, 4, n_vocab), tr, 1024) || !decode(*m, 16, ref, tr)) return false;
    done();
    // 7
    const int64_t n6 = m->position();
    if (!m->snapshot_save() || m->slot_save(slot2.string(), err) == 0) {
        if (err.empty()) err = "snapshot / slot save (long)";
        return false;
    }
    m->reset();
    if (!m->slot_load(slot2.string(), n6, err)) return false;
    {
        std::vector<float> logits;
        if (!m->forward({(int32_t) (23 % n_vocab)}, logits, err)) return false;
        tr.logits.push_back(logits);
    }
    if (!decode(*m, 12, ref, tr)) return false;
    done();
    tr.kv = m->kv_stream_stats();
    std::error_code ec;
    fs::remove(slot, ec);
    fs::remove(slot2, ec);
    return true;
}

void compare(const Cfg& c, const Trace& ref, const Trace& tr, const std::string& pack_name) {
    const std::string tag = pack_name + " " + c.name;
    check(tr.window == std::atoll(kWindow), tag + ": streamed, " + std::to_string(tr.window) + " positions in VRAM");
    bool same_n = tr.logits.size() == ref.logits.size() && tr.phase_end == ref.phase_end;
    check(same_n, tag + ": " + std::to_string(tr.logits.size()) + " logits vectors (the reference: " +
                      std::to_string(ref.logits.size()) + ")");
    if (!same_n) return;
    int at = 0;
    for (size_t ph = 0; ph < ref.phase_end.size(); ++ph) {
        int bad = 0, first = -1;
        double worst = 0;
        for (; at < ref.phase_end[ph]; ++at) {
            const auto& a = tr.logits[(size_t) at];
            const auto& b = ref.logits[(size_t) at];
            if (a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0) continue;
            if (bad++ == 0) first = at;
            worst = std::max(worst, glmfx::rel_l2(a, b));
        }
        char buf[200];
        std::snprintf(buf, sizeof buf, ": phase %zu logits bit-identical to the resident run (%d differ, first %d, "
                                       "worst rel L2 %.3g)", ph + 1, bad, first, worst);
        check(bad == 0, tag + buf);
    }
    check(tr.drafts == ref.drafts, tag + ": the NextN drafts equal the resident run's");
    char buf[240];
    std::snprintf(buf, sizeof buf, ": the attentions read %llu rows, %llu differ from the host copy; %llu block reads, "
                                   "%llu from RAM%s", (unsigned long long) tr.kv.dbg_rows,
                  (unsigned long long) tr.kv.dbg_bad, (unsigned long long) tr.kv.lookups,
                  (unsigned long long) tr.kv.misses, tr.kv.overflow ? ", OVERFLOW" : "");
    check(tr.kv.dbg_on && tr.kv.dbg_rows > 0 && tr.kv.dbg_bad == 0 && !tr.kv.overflow, tag + buf);
    if (c.window)
        check(tr.kv_phase1.lookups == 0 && tr.kv_phase1.dbg_rows > 0,
              tag + ": inside the window nothing is resolved (" + std::to_string(tr.kv_phase1.lookups) + " block reads)");
    else
        check(tr.kv_phase1.lookups > 0, tag + ": with the window off the first phase resolves");
}

int run_pack(const char* preset) {
    std::string why, err;
    const fs::path gguf = glmfx::synth(preset, why);
    const fs::path pack = gguf.empty() ? fs::path() : glmfx::pack(gguf, why);
    if (pack.empty()) return glmfx::fixture_missing("glm_dsa_kv_stream_test", why);
    strata::core::Glm5Geometry g;
    if (!strata::core::glm_pack_geometry(pack.string(), g, err) || g.arch != strata::core::GlmArch::GlmDsa) {
        std::fprintf(stderr, "glm_dsa_kv_stream_test: %s: %s\n", pack.string().c_str(), err.c_str());
        return 1;
    }
    std::printf("%s: %d layers + NextN, top-k %d, context %lld, window %s\n", preset, g.n_layers, g.idx_top_k,
                (long long) kCtx, kWindow);
    const Cfg ref_cfg[2] = {{"resident", false, false, false, false}, {"resident+graphs", false, false, false, true}};
    const Cfg cfgs[6] = {{"stream g0 w0", true, false, false, false}, {"stream g1 w0", true, true, false, false},
                         {"stream g0 w1", true, false, true, false},  {"stream g1 w1", true, true, true, false},
                         {"stream g1 w1 graphs", true, true, true, true}, {"stream g0 w0 graphs", true, false, false, true}};
    Trace ref[2];
    for (int r = 0; r < 2; ++r)
        if (!script(pack, ref_cfg[r], g.n_vocab, r == 0 ? nullptr : &ref[0], ref[r])) {
            std::fprintf(stderr, "glm_dsa_kv_stream_test: %s %s: %s\n", preset, ref_cfg[r].name, ref[r].err.c_str());
            return 1;
        }
    check(ref[0].window == 0 && ref[1].window == 0, std::string(preset) + ": the reference runs are not streamed");
    for (const Cfg& c : cfgs) {
        Trace tr;
        const Trace& rf = ref[c.graphs ? 1 : 0];
        if (!script(pack, c, g.n_vocab, &ref[0], tr)) {
            check(false, std::string(preset) + " " + c.name + ": " + tr.err);
            continue;
        }
        compare(c, rf, tr, preset);
    }
    return 0;
}

// PACK9 on three devices: the speculative decode (the head group ahead of the tail, rejected drafts redone)
int run_split() {
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev < 3) {
        std::printf("PACK9 3-part split: %d devices - not run\n", n_dev);
        return 0;
    }
    std::string why, err;
    const fs::path gguf = glmfx::synth("PACK9", why);
    const fs::path pack = gguf.empty() ? fs::path() : glmfx::pack(gguf, why);
    if (pack.empty()) return glmfx::fixture_missing("glm_dsa_kv_stream_test", why);
    strata::core::Glm5Geometry g;
    if (!strata::core::glm_pack_geometry(pack.string(), g, err)) return 1;
    std::vector<int> b3 = strata::core::glm_snap_bounds(g, {g.n_layers / 3, 2 * g.n_layers / 3});
    if (b3.size() != 2 || b3[0] >= b3[1] || !strata::core::glm_split_bounds_ok(g, b3, err)) {
        std::printf("PACK9 3-part split: no legal bounds - not run\n");
        return 0;
    }
    unsetenv("MAYA_GLM_SPLIT");
    const std::vector<int32_t> prompt = toks(2900, 5, g.n_vocab);
    const int kNew = 400;   // from inside the window to past it
    std::vector<int> want;
    for (int pass = 0; pass < 3; ++pass) {
        const Cfg c = pass == 0 ? Cfg{"resident", false, false, false, true}
                      : pass == 1 ? Cfg{"stream g1 w1 graphs", true, true, true, true}
                                  : Cfg{"stream g0 w0", true, false, false, false};
        set_env(c);
        auto m = std::make_unique<Glm5Model>();
        if (!m->load_pack_split(pack.string(), kCtx, b3, {0, 1, 2}, err) || !m->fast() || !m->spec_ready()) {
            check(false, std::string("3 parts ") + c.name + ": " + (err.empty() ? "the speculative decode is not ready" : err));
            return 0;
        }
        m->reset();
        std::vector<float> logits;
        if (!m->forward(prompt, logits, err)) {
            check(false, std::string("3 parts ") + c.name + ": prompt: " + err);
            return 0;
        }
        strata::kernels::SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        std::vector<int> got;
        int64_t produced = 0;
        const uint64_t s0 = m->spec_steps_, h0 = m->spec_hits_;
        const bool ok = m->decode_spec(sp, kNew, [&](int t) { got.push_back(t); return true; }, produced, err);
        const auto kv = m->kv_stream_stats();
        char buf[320];
        std::snprintf(buf, sizeof buf, "3 parts at %d,%d %s: %zu tokens by the speculative decode (%llu positions, %llu "
                                       "accepted)", b3[0], b3[1], c.name, got.size(),
                      (unsigned long long) (m->spec_steps_ - s0), (unsigned long long) (m->spec_hits_ - h0));
        if (pass == 0) {
            want = got;
            check(ok && (int) got.size() == kNew, buf);
        } else {
            check(ok && got == want, std::string(buf) + " equal the resident run's");
            std::snprintf(buf, sizeof buf, "3 parts %s: the attentions read %llu rows, %llu differ from the host copy, "
                                           "%llu block reads%s", c.name, (unsigned long long) kv.dbg_rows,
                          (unsigned long long) kv.dbg_bad, (unsigned long long) kv.lookups, kv.overflow ? ", OVERFLOW" : "");
            check(kv.dbg_on && kv.dbg_rows > 0 && kv.dbg_bad == 0 && !kv.overflow && kv.lookups > 0, buf);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("glm_dsa_kv_stream_test: a streamed glm-dsa model against the resident one, bit for bit\n");
    if (argc > 1 && std::string(argv[1]) == "--fixtures") {   // the synthetic packs only (no device needed)
        for (const char* preset : {"PFIQ1M", "PACK9"}) {
            std::string why;
            const fs::path gguf = glmfx::synth(preset, why);
            const fs::path pack = gguf.empty() ? fs::path() : glmfx::pack(gguf, why);
            std::printf("%s: %s\n", preset, pack.empty() ? why.c_str() : pack.string().c_str());
            if (pack.empty()) return 1;
        }
        return 0;
    }
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glm_dsa_kv_stream_test: no CUDA device - SKIP\n");
        return kSkip;
    }
    unsetenv("MAYA_GLM_SLOW");
    unsetenv("MAYA_GLM_DSA_DENSE");
    unsetenv("MAYA_GLM_KV_AUTO");
    setenv("MAYA_GLM_MTP", "1", 1);
    setenv("MAYA_GLM_PREFILL_MIN", "1", 1);
    setenv("MAYA_GLM_KV_INT8", "1", 0);        // the served format: INT8 latents + FP16 k_pe
    setenv("MAYA_GLM_PREFILL_MB", "768", 0);   // (the pool holds every expert: a fixed loan for the prompt path)
    for (const char* preset : {"PFIQ1M", "PACK9"}) {
        const int rc = run_pack(preset);
        if (rc != 0) return rc;
    }
    const int rc = run_split();
    if (rc != 0) return rc;
    if (g_fail) {
        std::printf("glm_dsa_kv_stream_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_dsa_kv_stream_test: ok\n");
    return 0;
}
