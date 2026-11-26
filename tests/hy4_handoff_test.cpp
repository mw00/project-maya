// maya-hy4's layer-split hand-off switches (src/hy4/hy4_handoff.hpp): the default probes alternatives only when the
// destination-stream copy fails, and the IOMMU stays the safety gate.  No GPU needed.
#include "../src/hy4/hy4_handoff.hpp"

#include <cstdio>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

using hy4::handoff::Mode;

int main() {
    namespace h = hy4::handoff;
    // unset and the old behaviour's spellings
    CHECK(h::mode_from(nullptr) == Mode::kAuto);
    CHECK(h::mode_from("") == Mode::kAuto);
    CHECK(h::mode_from("1") == Mode::kAuto);
    CHECK(h::mode_from("on") == Mode::kAuto);
    CHECK(h::mode_from("auto") == Mode::kAuto);
    CHECK(h::mode_from("0") == Mode::kOff);
    CHECK(h::mode_from("off") == Mode::kOff);
    CHECK(h::mode_from("peer") == Mode::kPeer);
    CHECK(h::mode_from("src") == Mode::kSrc);
    CHECK(h::mode_from("sync") == Mode::kSync);
    CHECK(h::mode_from("host") == Mode::kHost);
    CHECK(h::mode_from("srk") == Mode::kAuto);   // a typo does not turn the feature off
    CHECK(h::name(Mode::kAuto) && !std::strcmp(h::name(Mode::kAuto), "auto"));

    // host wins, and a translating IOMMU keeps the direct paths off unless HY4_PEER=1
    CHECK(!h::host_forced(nullptr, Mode::kAuto));
    CHECK(h::host_forced("1", Mode::kAuto));
    CHECK(!h::host_forced("0", Mode::kAuto));
    CHECK(h::host_forced(nullptr, Mode::kHost));
    CHECK(h::direct_allowed(false, nullptr));
    CHECK(!h::direct_allowed(true, nullptr));
    CHECK(!h::direct_allowed(true, "0"));
    CHECK(h::direct_allowed(true, "1"));

    // which modes probe, and which skip the destination-stream test
    CHECK(h::probes_alternatives(Mode::kAuto));
    CHECK(h::probes_alternatives(Mode::kSrc));
    CHECK(h::probes_alternatives(Mode::kSync));
    CHECK(!h::probes_alternatives(Mode::kOff));
    CHECK(!h::probes_alternatives(Mode::kPeer));
    CHECK(!h::probes_alternatives(Mode::kHost));
    CHECK(h::only_alternative(Mode::kSrc));
    CHECK(h::only_alternative(Mode::kSync));
    CHECK(!h::only_alternative(Mode::kAuto));

    CHECK(!h::log_enabled(nullptr));
    CHECK(!h::log_enabled("0"));
    CHECK(h::log_enabled("1"));
    CHECK(h::check_count(nullptr) == 0);
    CHECK(h::check_count("4") == 4);
    CHECK(h::check_count("-3") == 0);
    return 0;
}
