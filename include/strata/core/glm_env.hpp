// include/strata/core/glm_env.hpp - one parser for glm-dsa's environment switches.
//
// glm-dsa's on/off switches were read in three ways (a leading '0' = off and anything else on, so "off" / "false" / a
// JSON false that serve/server.py passes as "False" turned a switch ON; atoi() != 0, so "on" / "true" turned it OFF;
// a leading '1' only).  Every glm-dsa boolean switch now goes through env_switch() / env_flag():
//   1, on, true, yes (any case), or any other integer but 0      -> on
//   0, off, false, no (any case)                                   -> off
//   auto (any case)                                                -> kAuto, where the switch has an auto mode
//   unset or empty                                                 -> the switch's default
//   anything else: a warning on stderr naming the switch, and the default.
// Numbers: env_int() / env_num() parse the whole value (strtoll / strtod with an end check) and warn and keep the
// default for anything that is not a number.  Maya's own switches keep Maya's parsing.  Header-only, host code (no CUDA
// dependency), so the CPU tests check it directly (glm_env_test).
#pragma once

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::glmenv {

/// A switch value: off, on, auto (switches that have an auto mode), or a value parse() does not know.
enum : int { kInvalid = -2, kAuto = -1, kOff = 0, kOn = 1 };

/// ASCII case-insensitive equality (no locale).
inline bool ieq(const char* a, const char* b) {
    for (; *a != '\0' && *b != '\0'; ++a, ++b) {
        const char x = (*a >= 'A' && *a <= 'Z') ? (char) (*a - 'A' + 'a') : *a;
        const char y = (*b >= 'A' && *b <= 'Z') ? (char) (*b - 'A' + 'a') : *b;
        if (x != y) return false;
    }
    return *a == *b;
}

/// The value of a switch: kOn, kOff, kAuto ("auto") or kInvalid (null, empty, or an unknown word).
inline int parse(const char* v) {
    if (v == nullptr || v[0] == '\0') return kInvalid;
    if (ieq(v, "on") || ieq(v, "true") || ieq(v, "yes")) return kOn;
    if (ieq(v, "off") || ieq(v, "false") || ieq(v, "no")) return kOff;
    if (ieq(v, "auto")) return kAuto;
    char* end = nullptr;
    errno = 0;
    const long long x = std::strtoll(v, &end, 10);
    if (end == v || *end != '\0' || errno == ERANGE) return kInvalid;
    return x != 0 ? kOn : kOff;
}

/// A switch value as the engine reads it: null or "" -> dflt; a value parse() knows ("auto" only when allow_auto);
/// anything else -> a warning on stderr naming `name`, and dflt.
inline int switch_value(const char* name, const char* v, int dflt, bool allow_auto = false) {
    if (v == nullptr || v[0] == '\0') return dflt;
    int r = parse(v);
    if (r == kAuto && !allow_auto) r = kInvalid;
    if (r == kInvalid) {
        std::fprintf(stderr, "strata: %s=\"%s\" is not a switch value (1/0, on/off, true/false, yes/no%s): %s (the "
                             "default)\n", name, v, allow_auto ? ", auto" : "",
                     dflt == kAuto ? "auto" : dflt == kOff ? "off" : "on");
        return dflt;
    }
    return r;
}

/// The switch `name` from the environment (see switch_value).
inline int env_switch(const char* name, int dflt, bool allow_auto = false) {
    return switch_value(name, std::getenv(name), dflt, allow_auto);
}

/// An on/off switch from the environment.
inline bool env_flag(const char* name, bool dflt) { return env_switch(name, dflt ? kOn : kOff) == kOn; }

/// An integer from a value: null or "" -> dflt; a whole decimal integer; anything else -> a warning and dflt.
inline long long int_value(const char* name, const char* v, long long dflt) {
    if (v == nullptr || v[0] == '\0') return dflt;
    char* end = nullptr;
    errno = 0;
    const long long x = std::strtoll(v, &end, 10);
    if (end == v || *end != '\0' || errno == ERANGE) {
        std::fprintf(stderr, "strata: %s=\"%s\" is not an integer: %lld (the default)\n", name, v, dflt);
        return dflt;
    }
    return x;
}
inline long long env_int(const char* name, long long dflt) { return int_value(name, std::getenv(name), dflt); }

/// A finite number from a value: null or "" -> dflt; a whole number (strtod); anything else -> a warning and dflt.
inline double num_value(const char* name, const char* v, double dflt) {
    if (v == nullptr || v[0] == '\0') return dflt;
    char* end = nullptr;
    const double x = std::strtod(v, &end);
    if (end == v || *end != '\0' || !std::isfinite(x)) {
        std::fprintf(stderr, "strata: %s=\"%s\" is not a number: %g (the default)\n", name, v, dflt);
        return dflt;
    }
    return x;
}
inline double env_num(const char* name, double dflt) { return num_value(name, std::getenv(name), dflt); }

}  // namespace strata::glmenv
