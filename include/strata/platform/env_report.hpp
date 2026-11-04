// include/strata/platform/env_report.hpp - one start-up line naming the engine's tuning settings that came from the
// environment (#6): they are read with getenv where they are used and nothing else shows that they took effect.
// Each one under the name it was set by (MAYA_X, or the older STRATA_X when only that one is set), with the value the
// engine uses when it differs (a setting the engine clamps to its range).  Host code only.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace strata::platform {

struct EnvRange {
    const char* name;   // without the prefix, e.g. "GLM_PREFETCH_N"
    long lo, hi;        // what the engine clamps an integer value to
    const char* why;    // said with the clamped value, e.g. "at most 2"
};

// The settings `env` (a NULL-ended "NAME=value" table, the process's environ) holds under MAYA_<group>* or
// STRATA_<group>*, as "<what>: NAME=value, NAME=value (used: 2, at most 2), ..."; "" when none is set.
inline std::string env_report(char** env, const char* group, const char* what, const EnvRange* ranges = nullptr,
                              size_t n_ranges = 0) {
    if (env == nullptr) return "";
    std::map<std::string, std::pair<std::string, std::string>> set;   // the name past the prefix -> (full name, value)
    const size_t gl = std::strlen(group);
    for (char** e = env; *e != nullptr; ++e) {
        const char* s = *e;
        const char* eq = std::strchr(s, '=');
        if (eq == nullptr) continue;
        const bool maya = std::strncmp(s, "MAYA_", 5) == 0, strata = std::strncmp(s, "STRATA_", 7) == 0;
        const char* rest = maya ? s + 5 : strata ? s + 7 : nullptr;
        if (rest == nullptr || rest >= eq || std::strncmp(rest, group, gl) != 0) continue;
        const std::string key(rest, eq), full(s, eq);
        if (key == "GLM_VISION_LEND_MB") continue;   // the server sets it itself when vision is on, not the user
        auto it = set.find(key);
        if (it == set.end() || maya) set[key] = {full, std::string(eq + 1)};   // (MAYA_X wins, as env_aliases does)
    }
    std::string out;
    for (const auto& [key, nv] : set) {
        out += (out.empty() ? std::string(what) + ": " : std::string(", ")) + nv.first + "=" + nv.second;
        for (size_t i = 0; i < n_ranges; ++i) {
            if (key != ranges[i].name) continue;
            errno = 0;
            const long v = std::strtol(nv.second.c_str(), nullptr, 10);   // (atoi's reading: leading digits, else 0)
            const long used = std::clamp(v, ranges[i].lo, ranges[i].hi);
            if (used != v || errno == ERANGE)
                out += " (used: " + std::to_string(used) + ", " + ranges[i].why + ")";
        }
    }
    return out;
}

}  // namespace strata::platform
