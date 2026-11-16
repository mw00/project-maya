// tests/glm_dsa/glm_dsa_prefill_stream_test.cpp - glm-dsa B9.9 on the CPU (no GPU): the prompt's expert streaming
// driver's planners and the node-split host share.
//
//   1. the switches: the driver on glm-dsa only (MAYA_GLM_PREFILL_STREAM=0 off), the prestage buffer on a glm-dsa
//      split's parts;
//   2. the early prestage schedule: on the serial schedule every later part pre-stages before every chunk, under the
//      chunk pipeline only before chunk 0; a single GPU never;
//   3. the early prestage registry: a record is taken exactly once (by the same model, chunk start and length), a
//      second early pass records twice, forget drops a model's unclaimed records, and the prestage-only flag is per
//      thread (a part's real pass on another thread never sees it);
//   4. the landing ring's NUMA node: the GPU's node on a 2-node host, none on one node or an unknown GPU, the env's
//      node when it exists, "off" none;
//   5. B14.7 in the prestage: a predicted expert under the plan's measured upload threshold is not prestaged
//      (the same rows* as glmpf::lane_first, the same PCIe estimate order), never on Flash / without a lane;
//      the share's job grid covers every (expert, row block) once;
//   6. the node-split host share (two nodes' share_node + share_add) on GLM-5.3 experts (6144 x 2048; IQ2_XXS /
//      IQ3_XXS and Q2_K / Q2_K, with and without the SwiGLU clamp) for 4 experts of 1, 5, 9 and 2 routed rows, in a
//      class layout (half 1 at half 0 + D), against Maya's unsplit multi-row share (native_gu_rows_split /
//      native_quant_h / native_down_rows_split over the contiguous blob): the gate/up values bit-identical, the
//      outputs within max rel 1e-5 (one float sum reassociated).
#include "glm_dsa_prefill_stream.hpp"
#include "glm_tier_lane.hpp"
#include "glm_tier_plan.hpp"
#include "strata/kernels/glmd_batch.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace gp = strata::core::glmpfs;
namespace gt = strata::core::glmtier;
namespace kc = strata::kernels::cpu;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

void test_switches() {
    std::printf("[switches]\n");
    unsetenv("MAYA_GLM_PREFILL_STREAM");
    check(gp::stream_on(true) && !gp::stream_on(false), "on for glm-dsa, off for Flash");
    check(gp::split_prestage(true, 3) && !gp::split_prestage(true, 1) && !gp::split_prestage(false, 3),
          "the prestage buffer on a glm-dsa split's parts only (a single GPU has Maya's)");
    setenv("MAYA_GLM_PREFILL_STREAM", "0", 1);
    check(!gp::stream_on(true) && !gp::split_prestage(true, 3), "MAYA_GLM_PREFILL_STREAM=0: off");
    unsetenv("MAYA_GLM_PREFILL_STREAM");
}

void test_schedule() {
    std::printf("[early prestage schedule]\n");
    bool serial = true;
    for (int c = 0; c < 4; ++c) serial = serial && gp::early_parts(3, c, false) == std::vector<int>{1, 2};
    check(serial, "serial, 3 parts: parts 1 and 2 before every chunk");
    check(gp::early_parts(3, 0, true) == std::vector<int>{1, 2} && gp::early_parts(3, 1, true).empty() &&
              gp::early_parts(3, 5, true).empty(),
          "chunk pipeline: parts 1 and 2 before chunk 0 only");
    check(gp::early_parts(1, 0, false).empty() && gp::early_parts(2, 0, true) == std::vector<int>{1},
          "one GPU: none; two: part 1");
}

void test_registry() {
    std::printf("[early prestage registry]\n");
    gp::early_clear();
    int a = 0, b = 0;
    check(!gp::early_take(&a, 0, 4096), "nothing recorded: the part prestages itself");
    gp::early_record(&a, 0, 4096);
    check(!gp::early_take(&b, 0, 4096) && !gp::early_take(&a, 4096, 4096) && !gp::early_take(&a, 0, 2048),
          "another model, chunk start or length does not take it");
    check(gp::early_take(&a, 0, 4096) && !gp::early_take(&a, 0, 4096), "taken exactly once");
    gp::early_record(&a, 8192, 4096);
    gp::early_record(&a, 8192, 4096);
    check(gp::early_take(&a, 8192, 4096) && gp::early_take(&a, 8192, 4096) && !gp::early_take(&a, 8192, 4096),
          "two records, two takes");
    gp::early_record(&a, 0, 512);
    gp::early_record(&b, 0, 512);
    gp::early_forget(&a);
    check(!gp::early_take(&a, 0, 512) && gp::early_take(&b, 0, 512), "forget drops one model's records only");
    gp::set_prestage_only(&a);
    bool other = true;
    std::thread t([&] { other = gp::prestage_only(&a); });
    t.join();
    check(gp::prestage_only(&a) && !gp::prestage_only(&b) && !other, "prestage-only: this thread, this model only");
    gp::set_prestage_only(nullptr);
    check(!gp::prestage_only(&a) && !gp::prestage_only(nullptr), "cleared");
}

void test_node() {
    std::printf("[landing ring node]\n");
    check(gp::staging_node(1, 2, nullptr) == 1 && gp::staging_node(0, 2, nullptr) == 0, "the GPU's node on 2 nodes");
    check(gp::staging_node(0, 1, nullptr) == -1 && gp::staging_node(-1, 2, nullptr) == -1,
          "one node or an unknown GPU: no binding");
    check(gp::staging_node(1, 2, "off") == -1 && gp::staging_node(1, 2, "0") == 0 && gp::staging_node(0, 2, "1") == 1 &&
              gp::staging_node(0, 2, "3") == -1,
          "env: off / a node / a node that does not exist");
    check(gp::staging_node(1, 2, "gpu") == 1 && gp::staging_node(1, 2, "") == 1, "env anything else: the GPU's node");
}

void test_prestage_skip() {
    std::printf("[B14.7 in the prestage]\n");
    unsetenv("MAYA_GLM_UPLOAD_MIN_ROWS");
    unsetenv("MAYA_GLM_PREFILL_CPU");
    unsetenv("MAYA_GLM_PREFILL_CPU_ROW_MS");
    // upload 1.3 ms, lane expert 2 x 0.15, 0.05 ms a row: rows* = ceil((1.3 - 0.3) / 0.05) = 20
    check(glmd::upload_rows_min_measured(1.3, 0.3, 0.05) == 20, "the plan's threshold: 20 rows");
    check(gp::prestage_skip(true, 19.5, 1.3, 0.0, 0.0, 0.15, 0.05, true) &&
              !gp::prestage_skip(true, 20.0, 1.3, 0.0, 0.0, 0.15, 0.05, true),
          "19.5 predicted rows stay on the lane, 20 are prestaged");
    check(gp::prestage_skip(true, 10, 0.0, 1.3, 9.0, 0.15, 0.05, true) && !gp::prestage_skip(true, 10, 0.0, 0.0, 0.2, 0.15, 0.05, true),
          "the plan's PCIe estimate order: measured copies, else the streamed calibration, else the single copy");
    check(!gp::prestage_skip(false, 1, 1.3, 0, 0, 0.15, 0.05, true) && !gp::prestage_skip(true, 1, 1.3, 0, 0, 0.15, 0.05, false) &&
              !gp::prestage_skip(true, 1, 1.3, 0, 0, 0.0, 0.05, true),
          "never on Flash, without a lane for the layer, or an uncalibrated lane");
    setenv("MAYA_GLM_PREFILL_CPU", "0", 1);
    check(!gp::prestage_skip(true, 1, 1.3, 0, 0, 0.15, 0.05, true), "MAYA_GLM_PREFILL_CPU=0: the lane takes nothing");
    unsetenv("MAYA_GLM_PREFILL_CPU");
    setenv("MAYA_GLM_UPLOAD_MIN_ROWS", "4", 1);
    check(!gp::prestage_skip(true, 5, 1.3, 0, 0, 0.15, 0.05, true) && gp::prestage_skip(true, 3, 1.3, 0, 0, 0.15, 0.05, true),
          "MAYA_GLM_UPLOAD_MIN_ROWS=4 moves the threshold for the prestage too");
    unsetenv("MAYA_GLM_UPLOAD_MIN_ROWS");
    // the plan's row cost honours MAYA_GLM_PREFILL_CPU_ROW_MS; so does the prestage (0.1 ms a row: rows* = 10)
    setenv("MAYA_GLM_PREFILL_CPU_ROW_MS", "0.1", 1);
    check(glmd::upload_rows_min_measured(1.3, 0.3, 0.1) == 10 && !gp::prestage_skip(true, 12, 1.3, 0, 0, 0.15, 0.05, true) &&
              gp::prestage_skip(true, 9, 1.3, 0, 0, 0.15, 0.05, true),
          "MAYA_GLM_PREFILL_CPU_ROW_MS=0.1 replaces the learned 0.05 ms a row, as in the plan");
    unsetenv("MAYA_GLM_PREFILL_CPU_ROW_MS");
}

void test_mtp_token_path() {
    std::printf("[B11.1 NextN rows on the token path]\n");
    check(gp::prefill_min_tokens(nullptr) == 32 && gp::prefill_min_tokens("8") == 8 && gp::prefill_min_tokens("0") == 1 &&
              gp::prefill_min_tokens("") == 1,
          "MAYA_GLM_PREFILL_MIN: default 32, at least 1 (prefill()'s rule)");
    check(gp::mtp_token_fill(true, true, true, 31, 32) && !gp::mtp_token_fill(true, true, true, 32, 32),
          "a short follow-up (31 < 32 tokens) fills the NextN rows; 32 take the batched path (mtp_fill)");
    check(gp::mtp_token_fill(true, true, false, 5000, 32),
          "no batched state (--prefill 0, MAYA_GLM_NO_PREFILL, 'not covered'): every prompt fills them");
    check(!gp::mtp_token_fill(false, true, false, 5, 32) && !gp::mtp_token_fill(true, false, false, 5, 32) &&
              !gp::mtp_token_fill(true, true, false, 0, 32),
          "never on Flash, without a NextN block, or for an empty prompt (Maya's prefill())");
    check(gp::mtp_token_chunk(true, true, true, true, false), "the head of a glm-dsa model with a NextN block, no pf_: 1");
    check(!gp::mtp_token_chunk(false, true, true, true, false) && !gp::mtp_token_chunk(true, false, true, true, false) &&
              !gp::mtp_token_chunk(true, true, false, true, false) && !gp::mtp_token_chunk(true, true, true, false, false) &&
              !gp::mtp_token_chunk(true, true, true, true, true),
          "Flash, a later part (the split's chunk floor), no fast path, no NextN, or a batched state: Maya's chunk");
}

void test_jobs() {
    std::printf("[share job grid]\n");
    bool ok = true;
    for (int n_exp : {1, 3, 7})
        for (int total : {1024, 6144, 100})
            for (int block : {32, 64}) {
                std::vector<int> seen((size_t) n_exp * total, 0);
                const int nj = gp::share_jobs(n_exp, total, block);
                for (int job = 0; job < nj; ++job) {
                    int j, r0, r1;
                    gp::share_job(job, n_exp, block, total, j, r0, r1);
                    for (int r = r0; r < r1; ++r) ++seen[(size_t) j * total + r];
                }
                ok = ok && std::all_of(seen.begin(), seen.end(), [](int v) { return v == 1; });
            }
    check(ok, "every (expert, row) exactly once for 1 / 3 / 7 experts, 1024 / 6144 / 100 rows, blocks 32 / 64");
}

// ---------------------------------------------------------------- the node-split share

std::vector<size_t> scale_offsets(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q2_K: return {80, 82};
        case GGML_TYPE_Q3_K: return {108};
        case GGML_TYPE_Q4_K: return {0, 2};
        default: return {0};
    }
}

void fill(ggml_type t, uint8_t* p, size_t bytes, std::mt19937& rng, float d_lo, float d_hi) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    const size_t bb = ggml_type_size(t);
    std::uniform_real_distribution<float> du(d_lo, d_hi);
    for (size_t b = 0; b + bb <= bytes; b += bb)
        for (size_t o : scale_offsets(t)) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(du(rng));
            std::memcpy(p + b + o, &h, sizeof h);
        }
}

const gp::RunFn serial = [](int n, const std::function<void(int)>& fn) {
    for (int i = 0; i < n; ++i) fn(i);
};

void share_case(ggml_type gu, ggml_type dn, float limit, std::mt19937& rng) {
    const int64_t E = 6144, FF = 2048;
    char lim[32] = "";
    if (!std::isinf(limit)) std::snprintf(lim, sizeof lim, ", swiglu clamp %.1f", limit);
    const std::string name = std::string(ggml_type_name(gu)) + " / " + ggml_type_name(dn) + lim;
    kc::NativeFmt f, hf;
    std::string why;
    if (!kc::native_fmt(gu, dn, E, FF, f, why) || !gt::half_fmt(f, hf, why)) {
        check(false, name + ": formats (" + why + ")");
        return;
    }
    gt::SplitLayout L;
    if (!gt::split_layout(f.gu_row, f.d_row, FF, E, ggml_blck_size(dn), ggml_type_size(dn), L, why)) {
        check(false, name + ": split_layout (" + why + ")");
        return;
    }
    const int cnt[] = {1, 5, 9, 2};
    const int ne = 4;
    int rows = 0;
    for (int c : cnt) rows += c;
    // a RAM class of 4 slots: half 0 of slot s at base + s * stride, half 1 at that + D
    const size_t stride = (L.half_bytes + 4095) / 4096 * 4096, D = (size_t) ne * stride;
    std::vector<uint8_t> cls(2 * D, 0);
    std::vector<std::vector<uint8_t>> blob((size_t) ne, std::vector<uint8_t>(f.bytes));
    std::vector<gp::ShareX> ex((size_t) ne);
    int r0 = 0;
    for (int i = 0; i < ne; ++i) {
        fill(gu, blob[(size_t) i].data(), 2 * f.up_off, rng, 0.002f, 0.004f);
        fill(dn, blob[(size_t) i].data() + f.down_off, f.bytes - f.down_off, rng, 0.002f, 0.004f);
        uint8_t* h0 = cls.data() + (size_t) i * stride;
        gt::scatter(L, blob[(size_t) i].data(), h0, h0 + D);
        ex[(size_t) i] = gp::ShareX{h0, cnt[i], r0};
        r0 += cnt[i];
    }
    // the rows' inputs and their quantized activations (what split_share quantizes)
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) rows * E);
    for (auto& v : x) v = nd(rng);
    constexpr size_t kA = kc::kNativeActBytes, kH = kc::kNativeHBytes;
    std::vector<uint8_t> act((size_t) rows * kA);
    for (int r = 0; r < rows; ++r) kc::native_quant_act(f, x.data() + (size_t) r * E, act.data() + (size_t) r * kA);
    // Maya's unsplit share (the non-kq path) on the contiguous blobs
    std::vector<float> ff_ref((size_t) rows * FF), out_ref((size_t) rows * E);
    std::vector<uint8_t> hq_ref((size_t) rows * kH);
    for (int i = 0; i < ne; ++i) {
        std::vector<const void*> a((size_t) cnt[i]), h((size_t) cnt[i]);
        std::vector<float*> o((size_t) cnt[i]), y((size_t) cnt[i]);
        for (int t = 0; t < cnt[i]; ++t) {
            const int r = ex[(size_t) i].r0 + t;
            a[(size_t) t] = act.data() + (size_t) r * kA;
            o[(size_t) t] = ff_ref.data() + (size_t) r * FF;
            h[(size_t) t] = hq_ref.data() + (size_t) r * kH;
            y[(size_t) t] = out_ref.data() + (size_t) r * E;
        }
        const uint8_t* b = blob[(size_t) i].data();
        kc::native_gu_rows_split(f, b, b + f.up_off, a.data(), cnt[i], o.data(), 0, (int) FF, limit);
        for (int t = 0; t < cnt[i]; ++t) kc::native_quant_h(f, o[(size_t) t], (void*) h[(size_t) t]);
        kc::native_down_rows_split(f, b + f.down_off, h.data(), cnt[i], y.data(), 0, (int) E);
    }
    // the node-split share: node 0 and node 1, then the add
    gp::ShareScratch sc[2];
    for (int h = 0; h < 2; ++h) gp::share_node(hf, L, ne, ex.data(), h ? D : 0, act.data(), rows, limit, sc[h], serial);
    std::vector<float> out((size_t) rows * E);
    gp::share_add(sc[0].part.data(), sc[1].part.data(), out.data(), out.size(), serial, rows);
    const int H = (int) L.H;
    bool gu_same = true;
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < 2; ++h)
            gu_same = gu_same && std::memcmp(sc[h].ff.data() + (size_t) r * H, ff_ref.data() + (size_t) r * FF + (size_t) h * H,
                                             (size_t) H * 4) == 0;
    check(gu_same, name + ": gate/up half values bit-identical to the unsplit share, every routed row");
    double worst = 0.0;
    for (int r = 0; r < rows; ++r) {
        double num = 0.0, den = 0.0;
        for (int c = 0; c < E; ++c) {
            num = std::max(num, (double) std::fabs(out[(size_t) r * E + c] - out_ref[(size_t) r * E + c]));
            den = std::max(den, (double) std::fabs(out_ref[(size_t) r * E + c]));
        }
        worst = std::max(worst, den > 0.0 ? num / den : num);
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, ": %d rows of 4 experts, max rel %.2e < 1e-5 against the unsplit share", rows, worst);
    check(worst < 1e-5, name + buf);
}

}  // namespace

int main() {
    test_switches();
    test_schedule();
    test_registry();
    test_node();
    test_prestage_skip();
    test_mtp_token_path();
    test_jobs();
    std::printf("[node-split host share]\n");
    if (!kc::native_experts_available()) {
        std::printf("(no ggml-cpu native path in this build - the share cases are skipped)\n");
    } else {
        std::mt19937 rng(7);
        share_case(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, std::numeric_limits<float>::infinity(), rng);
        share_case(GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, std::numeric_limits<float>::infinity(), rng);
        share_case(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, 7.0f, rng);
    }
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
