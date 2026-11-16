// include/strata/platform/env_alias.hpp - Maya's settings are MAYA_X; the older STRATA_X still works as a fallback: every
// STRATA_X in the environment that has no MAYA_X also sets MAYA_X (the name the code reads; MAYA_X wins when both are
// set) - so a setting works under either name.  Each program calls it first thing in main(); a header that includes this
// one also runs it once at program start (the inline variable below), for the programs that never call it.
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <crt_externs.h>   // (macOS: the environment table is reached through _NSGetEnviron)
#elif !defined(_WIN32)
#include <unistd.h>
extern "C" char** environ;   // (POSIX's; declared again with the C linkage unistd.h gives it)
#endif

namespace strata::platform {

inline void env_aliases() {
#if defined(_WIN32)
    char** env = _environ;
#elif defined(__APPLE__)
    char** env = *_NSGetEnviron();
#else
    char** env = environ;
#endif
    if (env == nullptr) return;
    std::vector<std::pair<std::string, std::string>> add;   // (collected first: setting one may move the table)
    for (char** e = env; *e != nullptr; ++e) {
        const char* s = *e;
        const char* eq = std::strchr(s, '=');
        if (std::strncmp(s, "STRATA_", 7) != 0 || eq == nullptr || eq <= s + 7) continue;
        const std::string key = "MAYA_" + std::string(s + 7, eq);
        if (std::getenv(key.c_str()) != nullptr) continue;   // (MAYA_X is set: it counts)
        add.emplace_back(key, std::string(eq + 1));
    }
    for (const auto& kv : add) {
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 1);
#endif
    }
}

inline const bool env_aliases_at_start = (env_aliases(), true);   // (C++17 inline variable: one per program)

}  // namespace strata::platform
