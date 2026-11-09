// src/kernels/cuda/glmd_rope.cuh - THE glm-dsa rotation (B3.4): NORM (adjacent-pair) RoPE, host and device.
//
// GLM-5.3 rotates three things with one rule, theta 8e6, n_rot 64, no YaRN:
//   - the MLA query tail q_pe: dims 192..255 of each 256-wide head ([q_nope 192 | q_pe 64]);
//   - the MLA key tail k_pe: dims 512..575 of the 576 kv_a row ([c_kv 512 | k_pe 64]);
//   - the lightning indexer q (32 heads) and k (1 head): dims 0..63 of each 128 row ([rope 64 | nope 64]).
// llama.cpp: glm-dsa.cpp q_pe/k_pe ggml_rope_ext(rope_type) with LLM_ARCH_GLM_DSA -> LLAMA_ROPE_TYPE_NORM
// (llama-model.cpp), and the indexer calls name LLAMA_ROPE_TYPE_NORM explicitly.  HF: rope_interleave and
// indexer_rope_interleave are both true.
//
// THE PAIRING IS NORM: (2i, 2i+1) rotates by angle pos * inv_freq[i].  Maya's rope.hpp is NEOX (i, i + n_rot/2);
// using it here would produce well-shaped, wrong vectors (GLM-5.3-SPEC.md, RoPE row).
//
// Numerics, fixed here so every glm-dsa kernel and the CPU reference agree:
//   inv_freq[i] = float(pow(theta, -2i / n_rot))   computed in double ON THE HOST, rounded once to float
//   angle       = float(pos) * inv_freq[i]          one f32 multiply (HF's f32 outer(t, inv_freq))
//   (c, s)      = precise sincosf(angle)            device: sincosf, never __sincosf; host: cosf/sinf
//   rotation    = (x0*c - x1*s, x0*s + x1*c)       device: _rn intrinsics, so no FMA contraction
// llama.cpp builds theta by repeated multiplication (theta *= theta_scale); its angles differ from these by a few
// ulp at large positions.  The reference follows HF.
//
// Any file that includes this must be compiled WITHOUT --use_fast_math: fast math turns sincosf into __sincosf,
// whose error grows with |angle| and is far outside the 2e-6 gate at position 1,000,000.  The guard below makes
// that a compile error instead of a silent wrong answer.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GLMD_ROPE_HD __host__ __device__
#else
#define GLMD_ROPE_HD
#endif

#if defined(__CUDA_ARCH__) && defined(__USE_FAST_MATH__)
#error "glmd_rope.cuh needs precise sincosf: do not compile this file with --use_fast_math"
#endif
// HIP: __CUDA_ARCH__ is never defined there; its device pass is __HIP_DEVICE_COMPILE__, and -ffast-math (__FAST_MATH__)
// would make sincosf approximate the same way
#if defined(__HIP_DEVICE_COMPILE__) && defined(__FAST_MATH__)
#error "glmd_rope.cuh needs precise sincosf: do not compile this file with -ffast-math"
#endif

namespace glmd {

/// inv_freq[i] for i in [0, n_rot/2): double pow on the host, then one rounding to float.
inline void rope_inv_freq(float* inv_freq, int n_rot, double theta) {
    for (int i = 0; i < n_rot / 2; ++i)
        inv_freq[i] = (float) std::pow(theta, -2.0 * (double) i / (double) n_rot);
}

/// cos and sin of float(pos) * inv_freq.  pos is an int: 1,000,000 and every int32 position up to 2^24 are exact
/// in float, as in HF's f32 position arange.
GLMD_ROPE_HD inline void rope_cs(int pos, float inv_freq, float& c, float& s) {
    const float angle = (float) pos * inv_freq;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    sincosf(angle, &s, &c);
#else
    c = std::cos(angle);
    s = std::sin(angle);
#endif
}

/// ONE NORM PAIR: (x0, x1) -> (x0*c - x1*s, x0*s + x1*c), in place.
GLMD_ROPE_HD inline void rope_pair(float& x0, float& x1, float c, float s) {
#if defined(__CUDA_ARCH__)
    const float y0 = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
    const float y1 = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
#elif defined(__HIP_DEVICE_COMPILE__)
    // HIP's clang contracts a * b - c * d into an FMA by default (-ffp-contract=fast-honor-pragmas): the pragma keeps
    // both products and the sum separately rounded, as CUDA's _rn intrinsics and the host form do
    float y0, y1;
    {
#pragma clang fp contract(off)
        y0 = x0 * c - x1 * s;
        y1 = x0 * s + x1 * c;
    }
#else
    const float y0 = x0 * c - x1 * s;
    const float y1 = x0 * s + x1 * c;
#endif
    x0 = y0;
    x1 = y1;
}

/// Rotate x[0, n_rot) of one row in place at position pos (callers offset x to the rope part of the row).
GLMD_ROPE_HD inline void rope_row(float* x, int n_rot, int pos, const float* inv_freq) {
    for (int i = 0; i < n_rot / 2; ++i) {
        float c, s;
        rope_cs(pos, inv_freq[i], c, s);
        rope_pair(x[2 * i], x[2 * i + 1], c, s);
    }
}

}  // namespace glmd
