#pragma once
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::ds4::cache_sub {
struct Config {
    float ratio = 0.0f;   // zero preserves native routing, without extra allocations or router work
    int candidates = 12;
    float budget = 0.6f;  // normalized source + replacement mass, NOT an error probability
};

// Parse before allocating the model. Failure leaves the caller's config unchanged.
inline bool parse(const char* ratio, const char* candidates, const char* norm, const char* budget,
                  Config& out, std::string& err) {
    Config next;
    auto number = [&](const char* text, float& value, const char* name) {
        if (text == nullptr) return true;
        char* end = nullptr;
        errno = 0;
        const float x = std::strtof(text, &end);
        if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(x)) {
            err = std::string(name) + ": expected a finite number";
            return false;
        }
        value = x;
        return true;
    };
    if (!number(ratio, next.ratio, "DS4_CACHE_SUB_RATIO") ||
        !number(budget, next.budget, "DS4_CACHE_SUB_RISK_BUDGET")) return false;
    if (next.ratio < 0 || next.ratio > 1 || next.budget < 0) {
        err = "DS4_CACHE_SUB_RATIO must be in [0,1]; DS4_CACHE_SUB_RISK_BUDGET must be nonnegative";
        return false;
    }
    if (candidates != nullptr) {
        char* end = nullptr;
        errno = 0;
        const long n = std::strtol(candidates, &end, 10);
        if (end == candidates || *end != '\0' || errno == ERANGE || (n != 12 && n != 16 && n != 32)) {
            err = "DS4_CACHE_SUB_N must be 12, 16 or 32";
            return false;
        }
        next.candidates = (int) n;
    }
    // Deliberately expose only the normalization validated with the mass gate.
    if (norm != nullptr && std::strcmp(norm, "reference") != 0) {
        err = "DS4_CACHE_SUB_NORM supports reference only (the original top-k score denominator)";
        return false;
    }
    out = next;
    return true;
}
}  // namespace strata::ds4::cache_sub
