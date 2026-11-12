// The NO_* on/off switches (include/strata/platform/env_switch.hpp, #860): unset, empty, 0, false, off and no are
// OFF; 1 and any other value ON, as before.  Host code only, no GPU.
#include "strata/platform/env_switch.hpp"

#include <cstdio>
#include <cstdlib>

using strata::platform::env_switch;
using strata::platform::env_switch_value;

static int failures = 0;

static void expect(const char* v, bool want) {
    const bool got = env_switch_value(v);
    if (got == want) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s%s%s -> %s, want %s\n", v ? "\"" : "", v ? v : "(unset)", v ? "\"" : "",
                 got ? "on" : "off", want ? "on" : "off");
}

static void set(const char* name, const char* v) {
#if defined(_WIN32)
    _putenv_s(name, v);   // (Windows: an empty value removes the variable - the same answer, off)
#else
    setenv(name, v, 1);
#endif
}

int main() {
    // unchanged: unset is off, 1 is on, and any other word or number still turns the switch on
    expect(nullptr, false);
    expect("1", true);
    expect("2", true);
    expect("yes", true);
    expect("true", true);
    expect("on", true);
    expect("00", true);
    expect("0 ", true);
    // off now (they all meant "on" when any value did): the empty value, 0, false, off, no - in any case
    expect("", false);
    expect("0", false);
    expect("false", false);
    expect("FALSE", false);
    expect("False", false);
    expect("off", false);
    expect("OFF", false);
    expect("no", false);
    expect("No", false);
    // from the environment itself
    set("MAYA_TEST_ENV_SWITCH", "0");
    if (env_switch("MAYA_TEST_ENV_SWITCH")) {
        ++failures;
        std::fprintf(stderr, "FAIL: MAYA_TEST_ENV_SWITCH=0 read as on\n");
    }
    set("MAYA_TEST_ENV_SWITCH", "1");
    if (!env_switch("MAYA_TEST_ENV_SWITCH")) {
        ++failures;
        std::fprintf(stderr, "FAIL: MAYA_TEST_ENV_SWITCH=1 read as off\n");
    }
    if (env_switch("MAYA_TEST_ENV_SWITCH_NEVER_SET")) {
        ++failures;
        std::fprintf(stderr, "FAIL: an unset switch read as on\n");
    }

    if (failures) {
        std::fprintf(stderr, "%d case(s) failed\n", failures);
        return 1;
    }
    std::printf("env_switch_test: all cases passed\n");
    return 0;
}
