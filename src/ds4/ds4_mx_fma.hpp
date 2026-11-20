// Internal MXFP4 CPU helper. Build with floating-point contraction disabled.
#pragma once
#include <cfloat>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::ds4::gguf::detail {
inline bool fast_fma_environment_ok() {
#if (defined(__x86_64__) || defined(_M_X64)) && FLT_EVAL_METHOD == 0
    // Only RN with gradual underflow. Other environments retain the original library operation.
    return (_mm_getcsr() & 0xe040u) == 0 && std::fegetround() == FE_TONEAREST;
#else
    return false;
#endif
}

// Exact binary32 RN result with a rare fallback, not float mul+add or an unguarded double approximation.
// A product of two binary32 values is exactly representable in binary64 (at most 48 significant bits).
// A rounded binary64 sum can give a wrong binary32 result only when it lands on a binary32 midpoint.
// For NORMAL binary32 results that midpoint has low 29 bits exactly 0x10000000. Let std::fma handle it,
// and all zero/subnormal/overflow/nonfinite cases, preserving signs and NaN handling too.
inline float fma_guarded_rn(float a, float b, float c) {
    static_assert(sizeof(float) == 4 && sizeof(double) == 8 && std::numeric_limits<float>::digits == 24 &&
                  std::numeric_limits<double>::digits == 53, "requires IEEE binary32/binary64");
    const double product = (double) a * (double) b;
    const double sum = product + (double) c;
    uint64_t bits;
    std::memcpy(&bits, &sum, sizeof bits);
    const uint64_t magnitude = bits & 0x7fffffffffffffffull;
    if (magnitude >= 0x3810000000000000ull && magnitude <= 0x47efffffe0000000ull &&
        (bits & 0x1fffffffull) != 0x10000000ull)
        return (float) sum;
    return std::fma(a, b, c);
}
template <bool Fast> inline float mx_fma(float a, float b, float c) {
    if constexpr (Fast) return fma_guarded_rn(a, b, c);
    return std::fma(a, b, c);
}
}  // namespace strata::ds4::gguf::detail
