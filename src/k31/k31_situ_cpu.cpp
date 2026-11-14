// src/k31/k31_situ_cpu.cpp - SiTU-GLU on the host (the CPU lane).  See k31_situ.hpp.
// Build with -ffp-contract=off: the portable form must equal the device's bit for bit (ds4_par_exp.h's rule).
// (GCC's default contraction was measured harmless here - it only fuses exp_portable's exact 2^n scale - but the
// flag is the rule; -ffast-math is what breaks parity, and is refused below.)
#include "k31_situ.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "k31_situ_cpu.cpp must not be built with -ffast-math / -ffinite-math-only: the CPU lane's parity depends on IEEE semantics"
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace strata::k31 {

namespace {
// flush-to-zero / denormals-are-zero off for the loop, restored after: the portable forms make denormals, and a
// process (or a library it loaded) that set FTZ/DAZ would otherwise break the lane's parity with the device
struct NoFtz {
#if defined(__x86_64__) || defined(__i386__)
    unsigned saved = _mm_getcsr();
    NoFtz() { _mm_setcsr(saved & ~0x8040u); }   // bit 15 FTZ, bit 6 DAZ
    ~NoFtz() { _mm_setcsr(saved); }
#endif
};
}  // namespace

void situ_cpu(const float* gate, const float* up, float* out, int64_t n, float beta, float lbeta, bool portable) {
    const NoFtz guard;
    if (portable) {
        for (int64_t i = 0; i < n; ++i) out[i] = situ_portable(gate[i], up[i], beta, lbeta);
    } else {
        for (int64_t i = 0; i < n; ++i) out[i] = situ(gate[i], up[i], beta, lbeta);
    }
}

}  // namespace strata::k31
