// src/ling/ling_drafting.hpp - maya-ling's --drafting value (docs/LING.md, section 4).
//
//   auto          the draft block is loaded; the engine measures once on this PC whether drafting makes answers faster
//                 and keeps the faster (its calibration file)
//   on, mtp, 1    the draft block is loaded and drafting is used, without that measurement
//   off, none, 0  no drafting
// (the setup writes "mtp" for the setup's own "on".)  Host code only: tests/ling_unit_test.cpp runs it.
#pragma once

#include <string>

namespace ling::drafting {

struct Choice {
    bool known = false;    // a value this flag takes
    int spec_k = 0;        // 0: off; -1: drafts by the engine's own window (the auto rule's)
    bool load_mtp = false;
    bool measure = false;  // auto: the first-start measurement decides; on: no measurement, drafting used
};

inline Choice parse(const std::string& v) {
    Choice c;
    if (v == "none" || v == "off" || v == "0") {
        c.known = true;
    } else if (v == "auto") {
        c = {true, -1, true, true};
    } else if (v == "on" || v == "mtp" || v == "1") {
        c = {true, -1, true, false};
    }
    return c;
}

}  // namespace ling::drafting
