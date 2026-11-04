// Host policy for the opt-in HIP scalar prompt tiles. No device or model state is changed here.
#pragma once
#include <climits>
#include <cstring>

namespace strata::kernels::exl3::hip_scalar {
// 8 is the existing path; -1 chooses the configurable batch-row policy. Invalid values stay on 8.
inline int mode(const char* value) {
    if (value == nullptr || std::strcmp(value, "0") == 0 || std::strcmp(value, "8") == 0) return 8;
    if (std::strcmp(value, "auto") == 0) return -1;
    if (std::strcmp(value, "12") == 0) return 12;
    if (std::strcmp(value, "16") == 0) return 16;
    if (std::strcmp(value, "24") == 0) return 24;
    return 8;
}
inline int cutoff(const char* value, int fallback) {
    if (value == nullptr || *value == 0) return fallback;
    unsigned n = 0;
    for (const char* p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || n > (unsigned(INT_MAX) - unsigned(*p - '0')) / 10) return fallback;
        n = n * 10 + unsigned(*p - '0');
    }
    return int(n);
}
inline int rows(int mode, bool gate_up, int max_rows, int gu_min, int down_min) {
    if (mode != -1) return mode;
    return gate_up ? (max_rows < gu_min ? 8 : 12) : (max_rows < down_min ? 12 : 24);
}
} // namespace strata::kernels::exl3::hip_scalar
