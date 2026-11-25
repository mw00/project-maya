// CPU-only unit tests for MAYA_GLM_NO_REUSE boolean flag parsing (include/maya/core/glm_env.hpp).
#include "maya/core/glm_env.hpp"
#include "glm_test_env.hpp"

#undef NDEBUG
#include <cassert>
#include <cstdio>

int main() {
    using maya::glmenv::no_reuse_env;

    unsetenv("MAYA_GLM_NO_REUSE");
    assert(!no_reuse_env());

    for (const char* off : {"", "0", "00", "-0", "off", "OFF", "false", "False", "FALSE", "no", "NO"}) {
        setenv("MAYA_GLM_NO_REUSE", off, 1);
        assert(!no_reuse_env());
    }

    for (const char* on : {"1", "2", "-1", "on", "ON", "true", "True", "TRUE", "yes", "Yes"}) {
        setenv("MAYA_GLM_NO_REUSE", on, 1);
        assert(no_reuse_env());
    }

    // Invalid spellings warn and keep the default (off).
    setenv("MAYA_GLM_NO_REUSE", "invalid", 1);
    assert(!no_reuse_env());

    unsetenv("MAYA_GLM_NO_REUSE");
    std::printf("glm_no_reuse_test: PASSED\n");
    return 0;
}
