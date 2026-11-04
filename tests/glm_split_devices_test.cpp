// The devices of a GLM layer split beside an expert helper (glm_split_devices): the helper's device left out of the
// default list, a named list (MAYA_GLM_DEVS) refused when it names the helper or repeats a device, pinned boundaries
// taking the first n + 1 devices beside the helper, `auto` taking up to the part limit - and no helper: the lists as
// before.  Host only, no GPU.
#include "strata/core/glm_split_devices.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::core::glm_split_devices;

static int fails = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}

static std::vector<int> run(const std::vector<int>* named, int n_dev, int helper, int n_parts, int max_parts,
                            bool want_ok, const char* what) {
    std::vector<int> out;
    std::string err;
    const bool ok = glm_split_devices(named, n_dev, helper, n_parts, max_parts, out, err);
    expect(ok == want_ok, what);
    if (!ok && want_ok) std::fprintf(stderr, "  %s\n", err.c_str());
    if (ok && !want_ok) std::fprintf(stderr, "  accepted\n");
    return out;
}

int main() {
    // no helper: every visible device in order, up to the part limit (auto) or the boundaries' count
    expect(run(nullptr, 3, -1, 0, 8, true, "auto, no helper") == std::vector<int>{0, 1, 2}, "auto lists every device");
    expect(run(nullptr, 6, -1, 0, 4, true, "auto, the part limit") == std::vector<int>{0, 1, 2, 3}, "auto stops at the limit");
    expect(run(nullptr, 3, -1, 2, 8, true, "one boundary") == std::vector<int>{0, 1}, "one boundary takes two devices");
    // the helper (the APU, visible last) beside two cards
    expect(run(nullptr, 3, 2, 0, 8, true, "auto beside the helper") == std::vector<int>{0, 1}, "the helper is left out");
    expect(run(nullptr, 3, 2, 2, 8, true, "one boundary beside the helper") == std::vector<int>{0, 1}, "pinned, helper out");
    expect(run(nullptr, 3, 0, 0, 8, true, "the helper visible first") == std::vector<int>{1, 2}, "a first helper is left out");
    run(nullptr, 3, 2, 3, 8, false, "two boundaries need three devices beside the helper");
    run(nullptr, 2, 1, 2, 8, false, "one card beside the helper is not a split");
    // a named list
    const std::vector<int> cards{0, 1}, with_helper{0, 1, 2}, twice{0, 0};
    expect(run(&cards, 3, 2, 0, 8, true, "named cards beside the helper") == cards, "a named list is kept");
    run(&with_helper, 3, 2, 0, 8, false, "a named list naming the helper is refused");
    run(&twice, 3, -1, 0, 8, false, "a named list naming a device twice is refused");
    run(&cards, 3, -1, 3, 8, false, "a named list shorter than the boundaries need is refused");
    if (fails == 0) std::puts("glm_split_devices: the helper's device is never a part of the split");
    return fails == 0 ? 0 : 1;
}
