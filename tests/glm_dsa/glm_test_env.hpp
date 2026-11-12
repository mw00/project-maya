// tests/glm_dsa/glm_test_env.hpp - the POSIX names the glm-dsa CPU tests use (setenv, unsetenv, getpid), so the same
// tests build on Windows (the HIP build).  POSIX: exactly <unistd.h> + <cstdlib>, nothing else changes.
// Windows: setenv/unsetenv through _putenv_s (an empty value removes the variable there, as Maya's own tests use it),
// getpid from <process.h>; and windef.h's `near` / `far` macros (the HIP headers pull in windows.h) are dropped, so a
// test's own near() helper compiles.
#pragma once

#include <cstdlib>

#ifdef _WIN32
#include <process.h>

#undef near
#undef far

inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite && std::getenv(name) != nullptr) return 0;
    return _putenv_s(name, value) == 0 ? 0 : -1;
}

inline int unsetenv(const char* name) { return _putenv_s(name, "") == 0 ? 0 : -1; }
#else
#include <unistd.h>
#endif
