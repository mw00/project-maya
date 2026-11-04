// include/strata/platform/env_alias.hpp - Maya's settings by the names its docs give them: every MAYA_X in the
// environment also sets the older STRATA_X, the name the code still reads (over an older value: MAYA_X counts) - so a
// setting works under either name.  Each program calls it first thing in main().
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
extern "C" char** environ;   // (POSIX's; declared again with the C linkage unistd.h gives it)
#endif

namespace strata::platform {

inline void env_aliases() {
#if defined(_WIN32)
    char** env = _environ;
#else
    char** env = environ;
#endif
    if (env == nullptr) return;
    std::vector<std::pair<std::string, std::string>> add;   // (collected first: setting one may move the table)
    for (char** e = env; *e != nullptr; ++e) {
        const char* s = *e;
        const char* eq = std::strchr(s, '=');
        if (std::strncmp(s, "MAYA_", 5) != 0 || eq == nullptr) continue;
        add.emplace_back("STRATA_" + std::string(s + 5, eq), std::string(eq + 1));
    }
    for (const auto& kv : add) {
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 1);
#endif
    }
}

}  // namespace strata::platform
