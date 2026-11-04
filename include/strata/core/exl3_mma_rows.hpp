#pragma once
// STRATA_GLM_EXL3_MMA_ROWS_SHORT=<rows>[,<max chunk tokens>]: in a prompt chunk of up to <max chunk tokens> (4096) an
// EXL3 expert with up to <rows> rows goes through the tensor-core kernels instead of STRATA_GLM_EXL3_MMA_ROWS's (64).
// A short chunk's experts have tens to a few hundred rows each, and reconstructing each one to FP16 once a chunk (its
// fixed cost) is what such a chunk pays most for; a long chunk keeps the default, where the FP16 product is cheaper.
// Used by glm_prefill.cu's x3_run_set; CPU-tested (tests/core/exl3_mma_rows_test.cpp).
#include <cstdio>

namespace strata::core {

struct MmaRowsShort {
    int rows = 0;            // 0: off
    int max_tokens = 4096;
};

/// The setting's text (null: unset) -> the setting; rows 0 or less: off.
inline MmaRowsShort mma_rows_short_parse(const char* v) {
    MmaRowsShort r;
    if (v == nullptr || v[0] == '\0') return r;
    int n = 0, t = 0;
    const int got = std::sscanf(v, "%d,%d", &n, &t);
    if (got < 1 || n <= 0) return r;
    r.rows = n;
    if (got >= 2 && t > 0) r.max_tokens = t;
    return r;
}

/// The most rows an expert of a chunk of t tokens takes through the tensor-core kernels: base (STRATA_GLM_EXL3_MMA_ROWS)
/// unless the chunk is short and the short setting is on
inline int mma_rows_for(int t, int base, const MmaRowsShort& s) {
    return s.rows > 0 && t > 0 && t <= s.max_tokens ? s.rows : base;
}

}  // namespace strata::core
