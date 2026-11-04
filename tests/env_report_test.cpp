// The engine's start-up line of the GLM settings set in the environment (include/strata/platform/env_report.hpp,
// #6): only when one is set, each under the name it was set by, and the value used when the engine clamps it.
// Host code only: a made-up environment table, no GPU.
#include "strata/platform/env_report.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::platform::EnvRange;
using strata::platform::env_report;

static int failures = 0;

static void expect(std::vector<const char*> env, const std::string& want) {
    static const EnvRange ranges[] = {{"GLM_PREFETCH_N", 0, 2, "at most kSpares-1"}, {"GLM_PREFILL_SUB", 16, 8192, "16-8192"}};
    env.push_back(nullptr);
    const std::string got = env_report(const_cast<char**>(env.data()), "GLM_", "glm env", ranges, 2);
    if (got == want) return;
    ++failures;
    std::fprintf(stderr, "FAIL\n  got:  %s\n  want: %s\n", got.c_str(), want.c_str());
}

int main() {
    // nothing set (other settings and other groups do not count): no line
    expect({"PATH=/usr/bin", "STRATA_DS4_X=1", "MAYA_LOOP_GUARD=0"}, "");
    expect({}, "");
    // set, in the order of the setting's name past MAYA_ / STRATA_, each under the name used
    expect({"STRATA_GLM_SLOTS=0", "MAYA_GLM_PREFETCH_N=1"}, "glm env: MAYA_GLM_PREFETCH_N=1, STRATA_GLM_SLOTS=0");
    // both names (env_aliases copies MAYA_X to STRATA_X): said once, by the MAYA_ name, whichever comes first
    expect({"STRATA_GLM_SLOTS=4", "MAYA_GLM_SLOTS=4"}, "glm env: MAYA_GLM_SLOTS=4");
    expect({"MAYA_GLM_SLOTS=4", "STRATA_GLM_SLOTS=4"}, "glm env: MAYA_GLM_SLOTS=4");
    // clamped: the value used and why
    expect({"STRATA_GLM_PREFETCH_N=5"}, "glm env: STRATA_GLM_PREFETCH_N=5 (used: 2, at most kSpares-1)");
    expect({"STRATA_GLM_PREFETCH_N=2"}, "glm env: STRATA_GLM_PREFETCH_N=2");
    expect({"STRATA_GLM_PREFETCH_N=-1"}, "glm env: STRATA_GLM_PREFETCH_N=-1 (used: 0, at most kSpares-1)");
    expect({"MAYA_GLM_PREFILL_SUB=8", "STRATA_GLM_PREFETCH_N=0"},
           "glm env: STRATA_GLM_PREFETCH_N=0, MAYA_GLM_PREFILL_SUB=8 (used: 16, 16-8192)");
    // the server's own vision setting is not a user setting
    expect({"STRATA_GLM_VISION_LEND_MB=2048"}, "");
    expect({"STRATA_GLM_VISION_LEND_MB=2048", "MAYA_GLM_SLOTS=4"}, "glm env: MAYA_GLM_SLOTS=4");
    // a prefix alone, or an entry without '=', is not a setting
    expect({"STRATA_GLM", "MAYA_GLMX=1", "STRATA_GLM_RAM_GB"}, "");

    if (failures) {
        std::fprintf(stderr, "%d case(s) failed\n", failures);
        return 1;
    }
    std::printf("env_report_test: all cases passed\n");
    return 0;
}
