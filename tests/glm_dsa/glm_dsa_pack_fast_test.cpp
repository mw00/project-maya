// tests/glm_dsa/glm_dsa_pack_fast_test.cpp - glm-dsa B6.4 (label gpu): the glm-dsa fast path on a synthetic pack, one
// part against three parts at full-indexer boundaries, the speculative decode on and off, slots and snapshots.
//
// The pack: $STRATA_TEST_GLM_DSA_PACK when set, else the PACK9 preset (12 layers + NextN, full indexer layers 0, 1, 2,
// 6, 10) made with tools/glm_dsa_synth_gguf.py --preset PACK9 and packed by tools/iq_pack.py's glm-dsa mode
// (tests/glm_dsa/glm_dsa_fixtures.hpp).  Without a device, or without the tools, the test exits 77.
//
//   1. greedy 32 tokens from a fixed prompt on one part (STRATA_GLM_SPLIT=0) and on three parts at full-indexer
//      bounds (glm_snap_bounds of the thirds): identical tokens (the hop carries the residual exactly; each part's
//      first layer is full, so no selection crosses a part);
//   2. the speculative decode (decode_spec: the head part runs the NextN draft's next token while the tail finishes;
//      needs a split whose last part carries the NextN block) emits the very tokens of the token-at-a-time greedy
//      decode, with STRATA_GLM_SPEC_DRAFT_EARLY off and on (B11.2: the draft beside the tail group's front parts);
//      tried on the 3-part split, then on a 2-part one;
//   3. slot round trip on the 3-part split (B14.5): snapshot after prompt[:-1], slot_save, 16 greedy tokens, reset,
//      slot_load, the same 16 tokens; the slot holds at least every layer's latent + k_pe rows and the full layers'
//      index keys (glm_dsa_state_row_bytes x positions), and a slot whose fingerprint names another latent row size
//      (the dtype / row-size word of MAYALST3) is refused;
//   4. snapshot round trip on the 3-part split: save after the prompt, 16 tokens, restore, the same 16 tokens.
//   5. B15.3 on a 2-part split with the NextN block: the first NextN draft (mtp_draft) after a reset that follows
//      another prompt of the same length, after a slot restore (the MTP1 trailer) and after a snapshot restore equals
//      the fresh run's first draft; the fingerprint names the NextN block (words 12, 13) and a slot whose block word
//      differs is refused.
//   6. B8.1 (warm): one part, and with 3 devices the parallel 3-part load, warmed in slot order
//      (STRATA_GLM_WARM_SEQ=0, Maya's) and in file order (glm-dsa's default): the same digest of every VRAM and
//      RAM-tier slot (which expert, its state, its bytes), with experts in both tiers; a usage file scrambles each
//      layer's warm order, so the file order must move experts and reads them in contiguous runs; the 3 parts'
//      file-order warm-ups run one at a time.
// With fewer than 3 devices the 3-part steps are skipped (2 devices: the 2-part speculative check still runs).
#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_dsa_split.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_warm_seq.hpp"
#include "strata/kernels/sampler.hpp"

#include "glm_dsa_fixtures.hpp"
#include "glm_test_env.hpp"   // setenv / unsetenv (<unistd.h> on POSIX; mapped on Windows)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr int kSkip = 77;
constexpr int kTokens = 32;
constexpr int64_t kCtx = 1024;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

int argmax(const std::vector<float>& v) { return (int) (std::max_element(v.begin(), v.end()) - v.begin()); }

// greedy `n` tokens after `prompt`; false + err on any failure
bool greedy(strata::core::Glm5Model& m, const std::vector<int32_t>& prompt, int n, std::vector<int>& out,
            std::string& err) {
    std::vector<float> logits;
    out.clear();
    if (!m.forward(prompt, logits, err)) return false;
    for (int i = 0; i < n; ++i) {
        const int t = argmax(logits);
        out.push_back(t);
        if (i + 1 < n && !m.forward({t}, logits, err)) return false;
    }
    return true;
}

std::string first_diff(const std::vector<int>& a, const std::vector<int>& b) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        if (a[i] != b[i]) return "first difference at token " + std::to_string(i);
    return a.size() == b.size() ? "identical" : "lengths differ";
}

std::unique_ptr<strata::core::Glm5Model> load_split(const fs::path& pack, const std::vector<int>& bounds,
                                                    const std::vector<int>& devs, std::string& err) {
    auto m = std::make_unique<strata::core::Glm5Model>();
    if (!m->load_pack_split(pack.string(), kCtx, bounds, devs, err) || !m->fast()) return nullptr;
    return m;
}

/// The speculative decode vs the plain greedy tokens, with STRATA_GLM_SPEC_DRAFT_EARLY off (today's step order) and on
/// (glm-dsa B11.2: the NextN draft runs while the tail group's front parts compute; on a one-part tail the redo goes in
/// before the draft); "" when it ran, else why it did not.
std::string spec_check(strata::core::Glm5Model& m, const std::vector<int32_t>& prompt, const std::vector<int>& want,
                       const std::string& what) {
    if (!m.spec_ready()) return what + ": the speculative decode is not ready (no NextN block on the last part?)";
    for (int early = 0; early < 2; ++early) {
        if (early) setenv("STRATA_GLM_SPEC_DRAFT_EARLY", "1", 1);
        else unsetenv("STRATA_GLM_SPEC_DRAFT_EARLY");
        const std::string label = what + (early ? ", STRATA_GLM_SPEC_DRAFT_EARLY=1" : "");
        m.reset();
        std::string err;
        std::vector<float> logits;
        if (!m.forward(prompt, logits, err)) {
            unsetenv("STRATA_GLM_SPEC_DRAFT_EARLY");
            return label + ": prompt: " + err;
        }
        strata::kernels::SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        std::vector<int> got;
        int64_t produced = 0;
        const uint64_t s0 = m.spec_steps_, h0 = m.spec_hits_;
        const bool ok = m.decode_spec(sp, (int64_t) want.size(), [&](int t) { got.push_back(t); return true; }, produced, err);
        char buf[200];
        std::snprintf(buf, sizeof buf, " (%llu speculative positions, %llu accepted)",
                      (unsigned long long) (m.spec_steps_ - s0), (unsigned long long) (m.spec_hits_ - h0));
        check(ok && got == want, label + ": spec on vs off: " + (ok ? first_diff(got, want) : "decode_spec: " + err) + buf);
    }
    unsetenv("STRATA_GLM_SPEC_DRAFT_EARLY");
    return "";
}

/// 6. B8.1 (warm): the pack loaded with STRATA_GLM_WARM_SEQ=0 (slot order, Maya's parallel reads) and =1 (file
/// order, one reader, the parts in turn) gives the same warm digest (STRATA_GLM_WARM_SUM: every part's tiers at the end
/// of its warm-up) - only the order of the reads differs.
void warm_order_check(const fs::path& pack, const strata::core::Glm5Geometry& g, const std::vector<int>& b3,
                      int n_dev) {
    namespace gw = strata::core::glmwarm;
    // the usage file (STRATA_GLM_USAGE) leads each layer's warm order: a scrambled one, so the VRAM slots take
    // experts from all over each layer and the slot order reads all over the files
    const fs::path usage = glmfx::fixture_dir() / "pack_fast_test_warm.usage";
    {
        std::ofstream f(usage, std::ios::trunc);
        std::mt19937 rng(7);
        for (int il = 0; il <= (int) g.n_layers; ++il) {
            std::vector<int> o((size_t) g.n_expert);
            std::iota(o.begin(), o.end(), 0);
            std::shuffle(o.begin(), o.end(), rng);
            f << il;
            for (size_t r = 0; r < o.size(); ++r) f << ' ' << o[r] << ':' << (o.size() - r);
            f << '\n';
        }
    }
    setenv("STRATA_GLM_USAGE", usage.string().c_str(), 1);
    setenv("STRATA_GLM_WARM_SUM", "1", 1);   // each part's digest at the end of its warm-up (glmwarm::warm_digest)
    const auto load = [&](const char* seq, bool three, gw::Digest& d, gw::Stats& st, std::string& err) {
        setenv("STRATA_GLM_WARM_SEQ", seq, 1);
        gw::stats_reset();
        std::unique_ptr<strata::core::Glm5Model> m;
        if (three) {
            m = load_split(pack, b3, {0, 1, 2}, err);
        } else {
            setenv("STRATA_GLM_SPLIT", "0", 1);
            m = std::make_unique<strata::core::Glm5Model>();
            if (!m->load_pack_env(pack.string(), kCtx, err) || !m->fast()) m.reset();
            unsetenv("STRATA_GLM_SPLIT");
        }
        if (!m) return false;
        d = gw::warm_digest();
        st = gw::stats();
        return true;
    };
    for (int pass = 0; pass < 2; ++pass) {
        const bool three = pass == 1;
        if (three && n_dev < 3) {
            std::printf("  (B8.1: %d device(s) - the 3-part warm order check is skipped)\n", n_dev);
            break;
        }
        const std::string what = three ? "B8.1, 3 parts (parallel load)" : "B8.1, 1 part";
        gw::Digest d0, d1;
        gw::Stats s0, s1;
        std::string e0, e1;
        const bool l0 = load("0", three, d0, s0, e0);
        const bool l1 = l0 && load("1", three, d1, s1, e1);
        if (!l1) {
            check(false, what + ": load: " + (l0 ? e1 : e0));
            continue;
        }
        // a part that owns no expert layer warms nothing and sends no digest: PACK9's two leading dense layers, with
        // b3 at 2, leave the first part holding only those. Count the parts with an MoE layer instead (the NextN
        // block belongs to the last part, hence the +1).
        int warm_parts = 1;
        if (three) {
            const int edge[4] = {0, b3[0], b3[1], (int) g.n_layers + 1};
            warm_parts = 0;
            for (int p = 0; p < 3; ++p)
                if (edge[p + 1] > std::max(edge[p], (int) g.dense_lead)) ++warm_parts;
        }
        char buf[256];
        std::snprintf(buf, sizeof buf, " (%016llx vs %016llx over %d / %d part(s) of %d with expert layers; "
                      "%llu VRAM + %llu RAM slots, %.2f MB)", (unsigned long long) d0.sum, (unsigned long long) d1.sum,
                      d0.parts, d1.parts, warm_parts, (unsigned long long) d1.vram, (unsigned long long) d1.ram,
                      (double) d1.bytes / 1e6);
        check(d0.parts == warm_parts && d1.parts == warm_parts && d0.ok && d1.ok && d0.sum == d1.sum &&
                  d0.vram == d1.vram && d0.ram == d1.ram && d0.bytes == d1.bytes,
              what + ": slot order and file order warm the same experts into the same slots, byte for byte" + buf);
        check(d1.vram > 0 && d1.ram > 0, what + ": experts in both tiers (the RAM tier's round-robin is covered)");
        check(s0.warms == 0 && s1.warms >= 1 && s1.moved > 0,
              what + ": WARM_SEQ=0 kept the slot order, =1 read in file order (" + std::to_string(s1.warms) +
                  " warm-ups, " + std::to_string(s1.moved) + " of " + std::to_string(s1.jobs) + " experts moved)");
        check(s1.reads > 0 && 2 * s1.runs <= s1.reads,
              what + ": file order, " + std::to_string(s1.reads) + " reads in " + std::to_string(s1.runs) + " runs (" +
                  std::to_string(s1.batches) + " batches)");
        if (three)
            check(s1.max_busy == 1, what + ": the parts' file-order warm-ups ran one at a time (slot order: " +
                                        std::to_string(s0.max_busy) + " at once)");
    }
    unsetenv("STRATA_GLM_WARM_SEQ");
    unsetenv("STRATA_GLM_WARM_SUM");
    unsetenv("STRATA_GLM_USAGE");
    std::error_code ec;
    fs::remove(usage, ec);
}

}  // namespace

int main() {
    std::printf("glm_dsa_pack_fast_test: the glm-dsa fast path on a synthetic pack (B6.4)\n");
    std::string why, err;
    fs::path pack;
    if (const char* p = std::getenv("STRATA_TEST_GLM_DSA_PACK"); p != nullptr && p[0] != '\0') {
        pack = p;
    } else {
        const fs::path gguf = glmfx::synth("PACK9", why);
        pack = gguf.empty() ? fs::path() : glmfx::pack(gguf, why);
        if (pack.empty()) {
            return glmfx::fixture_missing("glm_dsa_pack_fast_test", why);
        }
    }
    strata::core::Glm5Geometry g;
    if (!strata::core::glm_pack_geometry(pack.string(), g, err) || g.arch != strata::core::GlmArch::GlmDsa) {
        std::fprintf(stderr, "glm_dsa_pack_fast_test: %s: %s\n", pack.string().c_str(),
                     err.empty() ? "not a glm-dsa pack" : err.c_str());
        return 1;
    }
    if (!strata::core::glm_forward_available(g, err)) {
        std::fprintf(stderr, "glm_dsa_pack_fast_test: the glm-dsa forward is gated off: %s\n", err.c_str());
        return 1;
    }
    {   // the pack passes the glm-dsa loader's geometry and tensor-shape checks (no GPU work: STRATA_GLM_CHECK_ONLY)
        setenv("STRATA_GLM_CHECK_ONLY", "1", 1);
        setenv("STRATA_GLM_SPLIT", "0", 1);
        strata::core::Glm5Model m;
        std::string e;
        m.load_pack_env(pack.string(), kCtx, e);
        unsetenv("STRATA_GLM_CHECK_ONLY");
        unsetenv("STRATA_GLM_SPLIT");
        check(e.find("check-only load") != std::string::npos, "the pack passes the loader's shape checks: " + e);
    }
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glm_dsa_pack_fast_test: no CUDA device - the engine steps skipped\n");
        return g_fail ? 1 : kSkip;
    }
    std::printf("glm_dsa_pack_fast_test: the glm-dsa fast path on a synthetic pack (B6.4), %d device(s)\n", n_dev);
    unsetenv("STRATA_GLM_SLOW");
    unsetenv("STRATA_GLM_NO_SPEC");
    unsetenv("STRATA_GLM_SPEC_DRAFT_EARLY");   // spec_check sets it for its second run
    unsetenv("STRATA_GLM_NO_MTP");
    unsetenv("STRATA_GLM_DSA_DENSE");

    const std::vector<int32_t> prompt = {11, 220, 3, 47, 1000 % std::max(1, (int) g.n_vocab), 5, 9, 31};
    std::vector<int> one;
    {   // 1 part (no NextN block: STRATA_GLM_MTP unset)
        setenv("STRATA_GLM_SPLIT", "0", 1);
        unsetenv("STRATA_GLM_MTP");
        auto m = std::make_unique<strata::core::Glm5Model>();
        if (!m->load_pack_env(pack.string(), kCtx, err) || !m->fast() || !greedy(*m, prompt, kTokens, one, err)) {
            std::fprintf(stderr, "glm_dsa_pack_fast_test: 1 part: %s\n", err.c_str());
            return 1;
        }
        unsetenv("STRATA_GLM_SPLIT");
        check(m->dsa_n_sel_last() == std::min<int>((int) (prompt.size() + kTokens - 1), g.idx_top_k),
              "1 part: dsa_n_sel_last = min(p + 1, top_k) = " + std::to_string(m->dsa_n_sel_last()));
    }

    // the boundaries: the thirds snapped to full-indexer layers, else the last two full layers
    std::vector<int> fulls;
    for (int il = 1; il < g.n_layers; ++il)
        if (g.idx_full(il)) fulls.push_back(il);
    std::vector<int> b3 = strata::core::glm_snap_bounds(g, {g.n_layers / 3, 2 * g.n_layers / 3});
    if (b3.size() != 2 || b3[0] >= b3[1] || !strata::core::glm_split_bounds_ok(g, b3, err)) {
        if (fulls.size() < 2) {
            std::printf("glm_dsa_pack_fast_test: fewer than two full-indexer layers past 0 - no 3-part split\n");
            return g_fail ? 1 : kSkip;
        }
        b3 = {fulls[fulls.size() - 2], fulls.back()};
    }
    const std::vector<int> b2 = {fulls.empty() ? g.n_layers / 2 : fulls[fulls.size() / 2]};

    bool spec_ran = false;
    if (n_dev >= 3) {
        auto m = load_split(pack, b3, {0, 1, 2}, err);
        if (!m) {
            std::fprintf(stderr, "glm_dsa_pack_fast_test: 3 parts at %d,%d: %s\n", b3[0], b3[1], err.c_str());
            return 1;
        }
        std::vector<int> three;
        if (!greedy(*m, prompt, kTokens, three, err)) {
            std::fprintf(stderr, "glm_dsa_pack_fast_test: 3 parts: %s\n", err.c_str());
            return 1;
        }
        check(three == one, "1 part vs 3 parts at " + std::to_string(b3[0]) + "," + std::to_string(b3[1]) + ": " +
                                std::to_string(kTokens) + " greedy tokens " + first_diff(three, one));

        // the speculative decode on 3 parts
        const std::string s3 = spec_check(*m, prompt, one, "3 parts");
        spec_ran = s3.empty();
        if (!spec_ran) std::printf("  (%s)\n", s3.c_str());

        // the slot round trip (B14.5)
        {
            m->reset();
            std::vector<float> logits;
            const std::vector<int32_t> head(prompt.begin(), prompt.end() - 1);
            const fs::path slot = glmfx::fixture_dir() / "pack_fast_test.slot";
            uint64_t bytes = 0;
            if (!m->forward(head, logits, err) || !m->snapshot_save() || (bytes = m->slot_save(slot.string(), err)) == 0) {
                check(false, "slot save: " + err);
            } else {
                std::vector<int> a, b;
                const bool ra = greedy(*m, {prompt.back()}, 16, a, err);
                m->reset();
                const bool lb = m->slot_load(slot.string(), (int64_t) head.size(), err);
                const bool rb = lb && greedy(*m, {prompt.back()}, 16, b, err);
                check(ra && rb && a == b, "slot round trip on 3 parts: 16 tokens " +
                                              (ra && rb ? first_diff(a, b) : "not run: " + err));
                int64_t want = 0;
                for (int il = 0; il < g.n_layers; ++il)
                    want += strata::core::glm_dsa_state_row_bytes(g, il, m->kv_int8()) * (int64_t) head.size();
                check(bytes >= (uint64_t) want, "the slot (" + std::to_string(bytes) + " B) holds every layer's latent + "
                                                    "k_pe and the full layers' index keys (" + std::to_string(want) + " B)");
                // a slot whose fingerprint names another latent row size is refused
                {
                    std::fstream f(slot, std::ios::in | std::ios::out | std::ios::binary);
                    int32_t w = 0;
                    f.seekg(8 + 4 * 11);
                    f.read((char*) &w, 4);
                    w += 2;
                    f.seekp(8 + 4 * 11);
                    f.write((const char*) &w, 4);
                }
                m->reset();
                std::string e2;
                const bool refused = !m->slot_load(slot.string(), (int64_t) head.size(), e2);
                check(refused, "a slot with another latent row size in its fingerprint is refused (" + e2 + ")");
                std::error_code ec;
                fs::remove(slot, ec);
            }
        }
        // the snapshot round trip
        {
            m->reset();
            std::vector<float> logits;
            std::vector<int> a, b;
            if (!m->forward(prompt, logits, err) || !m->snapshot_save()) {
                check(false, "snapshot save: " + err);
            } else {
                const std::vector<float> at_save = logits;
                bool ok = true;
                for (int pass = 0; pass < 2 && ok; ++pass) {
                    auto& out = pass == 0 ? a : b;
                    logits = at_save;
                    for (int i = 0; i < 16 && ok; ++i) {
                        const int t = argmax(logits);
                        out.push_back(t);
                        ok = m->forward({t}, logits, err);
                    }
                    if (pass == 0 && ok) ok = m->snapshot_restore();
                }
                check(ok && a == b, "snapshot round trip on 3 parts: 16 tokens " + (ok ? first_diff(a, b) : err));
            }
        }
    } else {
        std::printf("  (%d device(s): the 3-part split steps are skipped)\n", n_dev);
    }
    if (!spec_ran && n_dev >= 2) {
        auto m = load_split(pack, b2, {0, 1}, err);
        if (!m) {
            check(false, "2 parts at " + std::to_string(b2[0]) + ": " + err);
        } else {
            const std::string s2 = spec_check(*m, prompt, one, "2 parts");
            spec_ran = s2.empty();
            if (!spec_ran) check(false, s2);
        }
    }
    if (n_dev >= 2) {   // 5. B15.3: the NextN carrier across reset, slot and snapshot restores
        auto m = load_split(pack, b2, {0, 1}, err);
        if (!m || !m->has_mtp()) {
            check(false, "B15.3: 2 parts with the NextN block: " + (m ? std::string("no NextN block") : err));
        } else {
            const std::vector<int32_t> other(prompt.rbegin(), prompt.rend());   // same length, other tokens
            std::vector<float> logits;
            const auto first_draft = [&](const std::vector<int32_t>& pr, int& y, int& d) {
                y = d = -1;
                if (!m->forward(pr, logits, err)) return false;
                y = argmax(logits);
                d = m->mtp_draft(y, err);
                return d >= 0;
            };
            int y0 = -1, d0 = -1, yo = -1, dd = -1, y1 = -1, d1 = -1;
            m->reset();
            const bool f0 = first_draft(prompt, y0, d0);
            m->reset();
            const bool fo = first_draft(other, yo, dd);
            m->reset();
            const bool f1 = first_draft(prompt, y1, d1);
            check(f0 && fo && f1 && y1 == y0 && d1 == d0,
                  "B15.3: the first draft after a reset that followed a same-length prompt == the fresh one (" +
                      std::to_string(d1) + " vs " + std::to_string(d0) + (f0 && fo && f1 ? "" : ": " + err) + ")");
            const strata::core::GlmStateFingerprint fp = m->state_fp();
            check(fp[12] != 0 && fp[13] == 4 * g.n_embd, "B15.3: the fingerprint names the NextN block (" +
                                                             std::to_string(fp[12]) + ") and the carrier bytes");
            // slot: the snapshot after the whole prompt holds h_{n-1}; the draft right after slot_load (no trunk
            // step in between) reads the trailer's row
            m->reset();
            const fs::path slot = glmfx::fixture_dir() / "pack_fast_test_mtp.slot";
            int ds0 = -1, ds1 = -1, ds2 = -1;
            if (!m->forward(prompt, logits, err) || !m->snapshot_save() || m->slot_save(slot.string(), err) == 0) {
                check(false, "B15.3: slot save: " + err);
            } else {
                ds0 = m->mtp_draft(y0, err);
                for (int t : {3, 5, 7}) m->forward({t}, logits, err);
                m->reset();
                std::vector<float> lo;
                m->forward(other, lo, err);
                m->mtp_draft(argmax(lo), err);
                m->reset();
                const uint64_t r0 = m->mtp_carrier_stats().restored;
                const bool lb = m->slot_load(slot.string(), (int64_t) prompt.size(), err);
                ds1 = lb ? m->mtp_draft(y0, err) : -1;
                check(lb && ds1 == ds0 && ds0 == d0 && m->mtp_carrier_stats().restored > r0,
                      "B15.3: the first draft after slot_load == the fresh one (" + std::to_string(ds1) + " vs " +
                          std::to_string(ds0) + ", fresh " + std::to_string(d0) + (lb ? ")" : "): " + err));
                // snapshot restore (slot_load set the snapshot): a few tokens, back, the same first draft
                for (int t : {2, 4}) m->forward({t}, logits, err);
                ds2 = m->snapshot_restore() ? m->mtp_draft(y0, err) : -1;
                check(ds2 == ds0, "B15.3: the first draft after snapshot_restore == the fresh one (" +
                                      std::to_string(ds2) + " vs " + std::to_string(ds0) + ")");
                // a slot whose NextN block word differs is refused (word 12 of the MAYALST3 fingerprint)
                {
                    std::fstream f(slot, std::ios::in | std::ios::out | std::ios::binary);
                    int32_t w = 0;
                    f.seekg(8 + 4 * 12);
                    f.read((char*) &w, 4);
                    w ^= 0x55;
                    f.seekp(8 + 4 * 12);
                    f.write((const char*) &w, 4);
                }
                m->reset();
                std::string e2;
                check(!m->slot_load(slot.string(), (int64_t) prompt.size(), e2),
                      "B15.3: a slot written with another NextN block is refused (" + e2 + ")");
                std::error_code ec;
                fs::remove(slot, ec);
            }
        }
    }
    warm_order_check(pack, g, b3, n_dev);   // 6. B8.1 (warm)
    if (g_fail) {
        std::printf("glm_dsa_pack_fast_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_dsa_pack_fast_test: ok\n");
    return 0;
}
