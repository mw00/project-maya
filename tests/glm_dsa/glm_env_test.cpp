// tests/glm_dsa/glm_env_test.cpp - glm-dsa's switch parser (include/strata/core/glm_env.hpp), CPU only:
//   * glmenv::parse: 1/0, on/off, true/false, yes/no in any case, integers, auto, and what it refuses;
//   * switch_value / env_flag / env_switch: unset and empty keep the default, an unknown value warns and keeps it,
//     "auto" only where the switch has an auto mode; env_int / env_num with their end checks;
//   * the real readers that use it: glmd::parse_switch and decode_policy_env (FUSED_MLA, IDX_WMMA, TOPK_MB, TOPK_HIST,
//     FUSED_SPLIT), glmd::random_routing_seed, the tier plan's thp_mode / numa_split_requested, the state layout's
//     IDX_FP16 / IDX_HADAMARD, and the prefill stream's PREFILL_STREAM (default on): the spellings that used to flip a
//     switch the wrong way ("off" / "false" / a JSON false passed as "False" -> on; "on" / "true" -> off);
//   * the decode graphs' host-side gate (glmdsa::host_off_reason): the switch that keeps MAYA_GLM_DSA_GRAPHS off,
//     and a request's graphs=0;
//   * a request's engine knobs (glm_req_knobs.hpp, the speed plan): the GEN keys parsed and range-checked, a knob
//     no request names gives its reader's start value, and a request that leaves it out puts it back;
//   * --random-routing (MAYA_GLM_RANDOM_ROUTING=1): spec decode gated off, the glm-dsa state fingerprint marked.
#include "strata/core/glm_dsa_spec.hpp"
#include "strata/core/glm_env.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_req_knobs.hpp"
#include "strata/core/glm_state_layout.hpp"
#include "strata/kernels/glmd_decode_policy.hpp"
#include "strata/kernels/glmd_random_route.hpp"

#include "glm_dsa_fast.hpp"
#include "glm_dsa_prefill_stream.hpp"
#include "glm_tier_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace ge = strata::glmenv;

static int g_fail = 0;
#define CHECK(c)                                                                \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)

static void test_parse() {
    for (const char* v : {"1", "on", "ON", "On", "true", "True", "TRUE", "yes", "Yes", "2", "-1", "+3"})
        CHECK(ge::parse(v) == ge::kOn);
    for (const char* v : {"0", "off", "OFF", "false", "False", "no", "NO", "00", "-0"}) CHECK(ge::parse(v) == ge::kOff);
    CHECK(ge::parse("auto") == ge::kAuto && ge::parse("AUTO") == ge::kAuto);
    for (const char* v : {"", "enable", "1x", "1 ", "onn", "of", "0.5", "y", "99999999999999999999999"})
        CHECK(ge::parse(v) == ge::kInvalid);
    CHECK(ge::parse(nullptr) == ge::kInvalid);
}

static void test_values() {
    std::printf("(the warnings below are expected)\n");
    std::fflush(stdout);
    CHECK(ge::switch_value("T", nullptr, ge::kOn) == ge::kOn && ge::switch_value("T", "", ge::kOff) == ge::kOff);
    CHECK(ge::switch_value("T", "off", ge::kOn) == ge::kOff && ge::switch_value("T", "False", ge::kOn) == ge::kOff);
    CHECK(ge::switch_value("T", "true", ge::kOff) == ge::kOn);
    CHECK(ge::switch_value("T", "garbage", ge::kOff) == ge::kOff && ge::switch_value("T", "garbage", ge::kOn) == ge::kOn);
    CHECK(ge::switch_value("T", "auto", ge::kOff) == ge::kOff);              // no auto mode: the default
    CHECK(ge::switch_value("T", "auto", ge::kOff, true) == ge::kAuto);
    CHECK(ge::int_value("N", nullptr, 7) == 7 && ge::int_value("N", "", 7) == 7 && ge::int_value("N", "12", 7) == 12);
    CHECK(ge::int_value("N", "-3", 7) == -3 && ge::int_value("N", "12x", 7) == 7 && ge::int_value("N", "x", 7) == 7);
    CHECK(ge::num_value("X", "0.25", 1.0) == 0.25 && ge::num_value("X", "1e-3", 1.0) == 1e-3);
    CHECK(ge::num_value("X", "0.25s", 1.0) == 1.0 && ge::num_value("X", "nan", 1.0) == 1.0 &&
          ge::num_value("X", nullptr, 1.0) == 1.0);
    setenv("GLM_ENV_TEST_FLAG", "no", 1);
    CHECK(!ge::env_flag("GLM_ENV_TEST_FLAG", true));
    setenv("GLM_ENV_TEST_FLAG", "Yes", 1);
    CHECK(ge::env_flag("GLM_ENV_TEST_FLAG", false));
    unsetenv("GLM_ENV_TEST_FLAG");
    CHECK(ge::env_flag("GLM_ENV_TEST_FLAG", true) && !ge::env_flag("GLM_ENV_TEST_FLAG", false));
}

static void test_readers() {
    // parse_switch (the decode policy's tri-state)
    using namespace glmd;
    CHECK(parse_switch("off", kSwitchOn) == kSwitchOff && parse_switch("false", kSwitchOn) == kSwitchOff);
    CHECK(parse_switch("on", kSwitchOff) == kSwitchOn && parse_switch("auto", kSwitchOff) == kSwitchAuto);
    CHECK(parse_switch("bogus", kSwitchAuto) == kSwitchAuto && parse_switch(nullptr, kSwitchOff) == kSwitchOff);
    setenv("MAYA_GLMD_FUSED_MLA", "off", 1);
    setenv("MAYA_GLMD_TOPK_HIST", "False", 1);
    setenv("MAYA_GLMD_IDX_WMMA", "no", 1);
    setenv("MAYA_GLMD_TOPK_MB", "AUTO", 1);
    setenv("MAYA_GLMD_FUSED_SPLIT", "abc", 1);
    DecodePolicy p = decode_policy_env();
    CHECK(!p.fused_mla && !p.topk_hist && p.idx_wmma == kSwitchOff && p.topk_mb == kSwitchAuto);
    CHECK(p.fused_split == kFusedSplitDefault);
    setenv("MAYA_GLMD_FUSED_MLA", "on", 1);
    setenv("MAYA_GLMD_TOPK_HIST", "true", 1);
    setenv("MAYA_GLMD_FUSED_SPLIT", "3", 1);
    p = decode_policy_env();
    CHECK(p.fused_mla && p.topk_hist && p.fused_split == 3);
    for (const char* n : {"MAYA_GLMD_FUSED_MLA", "MAYA_GLMD_TOPK_HIST", "MAYA_GLMD_IDX_WMMA", "MAYA_GLMD_TOPK_MB",
                          "MAYA_GLMD_FUSED_SPLIT"})
        unsetenv(n);

    // the random-routing bench mode: a JSON false ("False") must leave it off (read once per process), so spec decode
    // stays allowed and the state fingerprint unmarked
    setenv("MAYA_GLM_RANDOM_ROUTING", "False", 1);
    CHECK(random_routing_seed() == 0 && !strata::core::glmspec::spec_bench_off());
    unsetenv("MAYA_GLM_RANDOM_ROUTING");

    // the tier plan's string readers
    namespace gt = strata::core::glmtier;
    CHECK(gt::thp_mode("on") == 1 && gt::thp_mode("off") == 0 && gt::thp_mode("true") == 1 && gt::thp_mode("auto") == -1);
    CHECK(gt::thp_mode("bogus") == -1 && gt::thp_mode(nullptr) == -1);
    CHECK(gt::numa_split_requested(true, "on") && gt::numa_split_requested(true, "true"));
    CHECK(!gt::numa_split_requested(true, "off") && !gt::numa_split_requested(true, "false"));
    CHECK(!gt::numa_split_requested(false, "on"));

    // the state layout's index-key switches
    setenv("MAYA_GLM_IDX_FP16", "off", 1);
    setenv("MAYA_GLM_IDX_HADAMARD", "false", 1);
    CHECK(!strata::core::glm_state_idx_fp16() && !strata::core::glm_state_idx_hadamard());
    setenv("MAYA_GLM_IDX_FP16", "ON", 1);
    setenv("MAYA_GLM_IDX_HADAMARD", "yes", 1);
    CHECK(strata::core::glm_state_idx_fp16() && strata::core::glm_state_idx_hadamard());
    unsetenv("MAYA_GLM_IDX_FP16");
    unsetenv("MAYA_GLM_IDX_HADAMARD");

    // a default-on switch: "true" keeps it on (atoi read it as 0)
    namespace gs = strata::core::glmpfs;
    setenv("MAYA_GLM_PREFILL_STREAM", "true", 1);
    CHECK(gs::stream_on(true));
    setenv("MAYA_GLM_PREFILL_STREAM", "off", 1);
    CHECK(!gs::stream_on(true));
    unsetenv("MAYA_GLM_PREFILL_STREAM");
    CHECK(gs::stream_on(true) && !gs::stream_on(false));
}

// B15.7 isolation (run as `glm_env_test --random-routing` with MAYA_GLM_RANDOM_ROUTING=1 set by ctest): the seed is
// on, spec_ready's gate says no, and a glm-dsa state fingerprint carries kGlmFpRandomRoute (a Flash layout does not)
static int random_routing_mode() {
    CHECK(glmd::random_routing_seed() == 7);   // MAYA_GLM_RANDOM_ROUTING_SEED=7
    CHECK(strata::core::glmspec::spec_bench_off());
    strata::core::Glm5Geometry g;
    strata::core::GlmStateLayout lay;
    lay.arch = strata::core::GlmArch::GlmDsa;
    CHECK(strata::core::glm_state_fingerprint(g, lay)[0] == (strata::core::kGlmFpFormat | strata::core::kGlmFpRandomRoute));
    lay.arch = strata::core::GlmArch::Glm5Next;
    CHECK(strata::core::glm_state_fingerprint(g, lay)[0] == strata::core::kGlmFpFormat);
    std::printf("glm_env_test --random-routing: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

// MAYA_GLM_DSA_GRAPHS=1 meets a host-side switch: fast_layers_plain names it (graph_prepare logs it once a part and
// counts the part-tokens), instead of a load log that said "on" while every token ran eagerly
static void test_graph_host_off() {
    using strata::core::glmdsa::host_off_reason;
    CHECK(host_off_reason(false, false, 0, 0, false) == nullptr);
    CHECK(std::string(host_off_reason(true, true, 1, 1, true)).find("MAYA_GLM_PROF") != std::string::npos);
    CHECK(std::string(host_off_reason(false, true, 0, 0, false)).find("GLM_CB_DIR") != std::string::npos);
    CHECK(std::string(host_off_reason(false, false, 2, 0, false)).find("MAYA_GLM_PREFETCH_N") != std::string::npos);
    CHECK(std::string(host_off_reason(false, false, 0, 1, false)).find("MAYA_GLM_AHEAD") != std::string::npos);
    CHECK(std::string(host_off_reason(false, false, 0, 0, true)).find("GLM_DUMP_H") != std::string::npos);
    // a request that asks for no graphs (GEN graphs=0); a host switch still names itself first
    CHECK(host_off_reason(false, false, 0, 0, false, false) == nullptr);
    CHECK(std::string(host_off_reason(false, false, 0, 0, false, true)).find("graphs=0") != std::string::npos);
    const std::string first = host_off_reason(false, false, 1, 0, false, true);
    CHECK(first.find("MAYA_GLM_PREFETCH_N") != std::string::npos);
    CHECK(first.find("prefetch_n=") != std::string::npos);   // (the count may be the request's, not the switch's)
}

// The speed plan's per-request knobs: what generate.cpp's GEN line hands to Request::take, and the store apply()
// sets (apply itself needs a model; the store is what the readers see)
static void test_request_knobs() {
    namespace gk = strata::core::glmknob;
    gk::Request r;
    CHECK(!r.named() && r.line().empty());
    CHECK(r.take("promote_min", "6") && r.take("prefetch_n", "1") && r.take("prefetch_at", "fetch"));
    CHECK(r.take("prefetch_rank", "2") && r.take("prefetch_blocks", "4") && r.take("mtp_keep", "4"));
    CHECK(r.take("cpu_split", "2") && r.take("vram_evict", "lru") && r.take("ngram", "1"));
    CHECK(r.take("lane_row4", "1") && r.take("graphs", "0") && r.take("pcie_frac_head", "0.35"));
    CHECK(r.named() && r.value(gk::PromoteMin) == 6 && r.value(gk::PrefetchAt) == 1 && r.value(gk::VramLru) == 1);
    CHECK(r.value(gk::GraphsOff) == 1 && r.lane_row4 == 1 && r.pcie_frac_head == 0.35);
    CHECK(r.line() == "promote_min=6 prefetch_n=1 prefetch_at=fetch prefetch_rank=2 prefetch_blocks=4 mtp_keep=4 "
                      "cpu_split=2 vram_evict=lru ngram=1 graphs=0 lane_row4=1 pcie_frac_head=0.350");
    // not knobs: the GEN line's other keys stay with generate.cpp
    CHECK(!r.take("temperature", "0") && !r.take("pcie_frac", "0.2") && !r.take("spec", "1") && !r.take("rid", "3"));
    // out of range or misspelt: taken (a warning) but left out, so the knob keeps its start value
    std::printf("(the glm knobs warnings below are expected)\n");
    std::fflush(stdout);
    gk::Request bad;
    CHECK(bad.take("promote_min", "-1") && bad.take("promote_min", "65536") && bad.take("prefetch_n", "3"));
    CHECK(bad.take("prefetch_at", "later") && bad.take("vram_evict", "LRU") && bad.take("mtp_keep", "4x"));
    CHECK(bad.take("cpu_split", "65") && bad.take("graphs", "2") && bad.take("lane_row4", ""));
    CHECK(bad.take("ngram", "on") && bad.take("pcie_frac_head", "1.5") && bad.take("pcie_frac_head", "x"));
    CHECK(!bad.named());
    // the store: no request -> the reader's start value; a request's value, an explicit 0 included; back when left out
    const auto store = [](const gk::Request& q) {
        for (int k = 0; k < gk::kCount; ++k) gk::set((gk::Id) k, q.value((gk::Id) k));
    };
    CHECK(gk::get(gk::PromoteMin, 3) == 3 && gk::get(gk::PrefetchN, 0) == 0 && gk::raw(gk::Ngram) == gk::kStart);
    store(r);
    CHECK(gk::get(gk::PromoteMin, 3) == 6 && gk::get(gk::PrefetchN, 0) == 1 && gk::get(gk::GraphsOff, 0) == 1);
    CHECK(gk::get(gk::Ngram, 0) == 1 && gk::get(gk::CpuSplit, 0) == 2 && gk::get(gk::PrefetchAt, 0) == 1);
    gk::Request zero;
    CHECK(zero.take("promote_min", "0") && zero.take("mtp_keep", "0"));
    store(zero);
    CHECK(gk::get(gk::PromoteMin, 3) == 0 && gk::get(gk::MtpKeep, 2) == 0 && gk::get(gk::PrefetchN, 0) == 0);
    store(gk::Request());
    for (int k = 0; k < gk::kCount; ++k) CHECK(gk::raw((gk::Id) k) == gk::kStart);
    CHECK(gk::get(gk::PromoteMin, 3) == 3 && gk::get(gk::MtpKeep, 2) == 2);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--random-routing") return random_routing_mode();
    test_parse();
    test_values();
    test_readers();
    test_graph_host_off();
    test_request_knobs();
    if (g_fail != 0) {
        std::fprintf(stderr, "glm_env_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("glm_env_test: OK\n");
    return 0;
}
