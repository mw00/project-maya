// include/strata/platform/env_switch.hpp - an on/off switch from the environment (the NO_* ones: MAYA_GLM_NO_SPEC,
// MAYA_GLM_NO_MTP, MAYA_GLM_NO_PREFILL): unset, empty, "0", "false", "off" or "no" (any case) is OFF; anything else -
// "1" as the docs give it - is ON.  A plain getenv() != nullptr took MAYA_GLM_NO_SPEC=0 for "no drafting" (#860).
// Header only, no dependencies: the CUDA engine and maya-metal both include it.
#pragma once

#include <cstdlib>

namespace strata::platform {

inline bool env_switch_value(const char* v) {
    if (v == nullptr) return false;
    // (each word compared without regard to case; the empty value is off too)
    const auto is = [v](const char* w) {
        int i = 0;
        for (; v[i] != '\0' && w[i] != '\0'; ++i) {
            const char a = (v[i] >= 'A' && v[i] <= 'Z') ? (char) (v[i] - 'A' + 'a') : v[i];
            if (a != w[i]) return false;
        }
        return v[i] == '\0' && w[i] == '\0';
    };
    return !(is("") || is("0") || is("false") || is("off") || is("no"));
}

inline bool env_switch(const char* name) {
    return env_switch_value(std::getenv(name));
}

}  // namespace strata::platform
