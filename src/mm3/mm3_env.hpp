// src/mm3/mm3_env.hpp - maya-mm3's environment settings: MAYA_<name> first (the other engines' prefix), then <name>
// itself (the MM3_* names #477 used, kept as aliases).
#pragma once

#include <cstdlib>
#include <string>

namespace mm3 {

inline const char* env(const char* name) {
    const std::string m = std::string("MAYA_") + name;
    if (const char* v = std::getenv(m.c_str())) return v;
    return std::getenv(name);
}

}  // namespace mm3
