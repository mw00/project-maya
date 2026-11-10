// src/kernels/cpu/glm_lane_row4.cpp - the glm-dsa port B14.9: the CPU expert lane's lockstep 4-row decode GEMV - its
// switch, its CPU check and the two hooks native_expert.cpp calls at nt = 1.  See
// include/strata/kernels/cpu/glm_lane_row4.hpp for the contract; the kernels are in glm_lane_row4_avx2.cpp.
//
// Two files, because native_expert.cpp calls the hooks on every one-token step of every model whose rows its own
// kernels do not take, with the lane on or off - on a CPU without AVX, every such step.  This file is compiled like
// native_expert.cpp (no AVX flags), so there the hooks' checks and their "no" are plain x86-64 code.  (While the hooks
// sat in the AVX2-compiled file, GCC saved lane_row4_gu's float argument with vmovss before its first check: an illegal
// instruction on a CPU without AVX, with the lane off.)  glm_lane_row4_avx2.cpp is entered only through
// row4x86::kernel(), and only after cpu_ok() below has said yes.
//
// Builds (cmake/glm_dsa.d/70-glm-lane-row4.cmake): every native-experts build compiles this file.  x86-64 builds also
// get the kernels, with STRATA_GLM_LANE_ROW4_KERNELS defined on both files (a CMake option, ON by default).  Every other
// build - aarch64 (DGX Spark / Grace), ppc64le (IBM AC922), or x86-64 with -DSTRATA_GLM_LANE_ROW4_KERNELS=OFF - compiles
// the no-lane branch at the end: no type has a 4-row kernel, the lane is never on and both hooks return false, so
// native_expert.cpp's nt = 1 loops run ggml's vec_dot as they did before the port.
#include "strata/kernels/cpu/glm_lane_row4.hpp"

#if defined(STRATA_GLM_LANE_ROW4_KERNELS) && (defined(__x86_64__) || defined(_M_X64))

#include "glm_lane_row4_x86.hpp"
#include "strata/core/glm_env.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace strata::kernels::cpu {
namespace {

using row4x86::Kernel;
using row4x86::kQ8K;
using row4x86::kQK;

// F16C, from CPUID leaf 1 (the kernels convert their fp16 scales with VCVTPH2PS; cpu_avx2_ok() asks for AVX, AVX2, FMA
// and the OS's YMM state, not for F16C: every AVX2 CPU has it, a virtual machine can still hide it)
bool cpu_f16c_ok() {
    unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    int x[4];
    __cpuidex(x, 1, 0);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
    __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
    return ((r[2] >> 29) & 1u) != 0;
}

// The kernels may run: this CPU has AVX2 + FMA + F16C and the OS saves the YMM state (CPUID, asked from this file,
// which any x86-64 runs), and ggml-cpu was built with the __AVX2__ path the kernels reproduce bit for bit (its fmadd
// and F16C).  Nothing in glm_lane_row4_avx2.cpp runs before this has said yes.
bool cpu_ok() {
    static const bool ok =
        cpu_avx2_ok() && cpu_f16c_ok() && ggml_cpu_has_avx2() && ggml_cpu_has_fma() && ggml_cpu_has_f16c();
    return ok;
}

// Which form of IQ1_S's scalar delta term this ggml-cpu runs (2 = one fmaf, 1 = multiply then add, 0 = neither of
// k_iq1_s's forms matches, so IQ1_S stays on ggml's loop): ggml's C (accum1 += d * sumi1) is contracted or not by
// its compiler flags, not by anything this file can see.  Decided once, after cpu_ok() only, against ggml's own
// vec_dot on 64 random rows of 8 super-blocks (raw Q8_K bytes, -128 included), where the two forms differ.
int iq1s_form() {
    static const int form = [] {
        ggml_cpu_init();   // ggml's vec_dot reads x.d through ggml-cpu's fp16 table on x86
        constexpr int kRows = 64, kN = 2048, kNb = kN / kQK;
        constexpr size_t B = 50;
        std::vector<uint8_t> w((size_t) kRows * kNb * B), q((size_t) kNb * kQ8K);
        uint32_t s = 0x2545F491u;
        auto rnd = [&s] {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            return s;
        };
        for (uint8_t& b : w) b = (uint8_t) rnd();
        for (size_t o = 0; o < w.size(); o += B) {   // x.d: a normal fp16 of either sign, about 2^-8 .. 2^-4
            const uint32_t m = rnd();
            const uint16_t h = (uint16_t) (0x1C00u + (m & 0x0FFFu) + (m & 0x80000000u ? 0x8000u : 0u));
            std::memcpy(&w[o], &h, 2);
        }
        for (int b = 0; b < kNb; ++b) {
            uint8_t* p = &q[(size_t) b * kQ8K];
            const float d = 0.01f + (float) (rnd() & 0xFFFFu) * 1e-7f;
            std::memcpy(p, &d, 4);
            for (int j = 0; j < kQK; ++j) p[4 + j] = (uint8_t) rnd();
            for (int g = 0; g < 16; ++g) {
                int sum = 0;
                for (int j = 0; j < 16; ++j) sum += (int8_t) p[4 + 16 * g + j];
                const int16_t s16 = (int16_t) sum;
                std::memcpy(p + 260 + 2 * g, &s16, 2);
            }
        }
        const ggml_vec_dot_t dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ1_S)->vec_dot;
        const Kernel k_fused = row4x86::kernel(19, true), k_separate = row4x86::kernel(19, false);
        bool fused = true, separate = true;
        for (int r = 0; r < kRows; r += 4) {
            const uint8_t* rows[4];
            float ref[4], a[4], b[4];
            for (int k = 0; k < 4; ++k) {
                rows[k] = &w[(size_t) (r + k) * kNb * B];
                dot(kN, &ref[k], 0, rows[k], 0, q.data(), 0, 1);
            }
            k_fused(kNb, rows, q.data(), a);
            k_separate(kNb, rows, q.data(), b);
            fused = fused && std::memcmp(a, ref, sizeof ref) == 0;
            separate = separate && std::memcmp(b, ref, sizeof ref) == 0;
        }
        return fused ? 2 : separate ? 1 : 0;   // both (no row told them apart): the GCC form
    }();
    return form;
}

// the kernel to run once row4_supported(type) has held (so cpu_ok() has, and IQ1_S's probe has run)
Kernel kernel_for(int type) { return row4x86::kernel(type, type != 19 || iq1s_form() != 1); }

std::atomic<int> g_override{-1};
std::atomic<uint32_t> g_types{kRow4AllTypes};

bool type_on(int type) {
    return type >= 0 && type < 32 && ((g_types.load(std::memory_order_relaxed) >> type) & 1u) != 0;
}

}  // namespace

bool row4_supported(int type) {
    if (!cpu_ok() || row4x86::kernel(type, true) == nullptr) return false;
    const ggml_type_traits_cpu* t = ggml_get_type_traits_cpu((ggml_type) type);
    if (t == nullptr || t->vec_dot_type != GGML_TYPE_Q8_K) return false;
    return type != 19 || iq1s_form() != 0;
}

int row4_iq1s_form() { return cpu_ok() ? iq1s_form() : -1; }

void row4_dot(int type, int n, const uint8_t* const w[4], const void* q8, float out[4]) {
    kernel_for(type)(n / kQK, w, (const uint8_t*) q8, out);
}

void row4_rows(int type, int n, const uint8_t* w, size_t row_bytes, const void* q8, float* out, int r0, int r1) {
    const Kernel k = kernel_for(type);
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const uint8_t* rows[4] = {w + (size_t) r * row_bytes, w + (size_t) (r + 1) * row_bytes,
                                  w + (size_t) (r + 2) * row_bytes, w + (size_t) (r + 3) * row_bytes};
        k(n / kQK, rows, (const uint8_t*) q8, out + r);
    }
    if (r < r1) {
        const ggml_vec_dot_t dot = ggml_get_type_traits_cpu((ggml_type) type)->vec_dot;
        for (; r < r1; ++r) dot(n, out + r, 0, w + (size_t) r * row_bytes, 0, q8, 0, 1);
    }
}

bool lane_row4_on() {
    const int o = g_override.load(std::memory_order_relaxed);
    if (o >= 0) return o != 0;
    static const bool env = [] {
        return strata::glmenv::env_flag("STRATA_GLM_LANE_ROW4", true);   // (default on: bit-identical, faster)
    }();
    return env;
}

void lane_row4_override(int mode) noexcept { g_override.store(mode < 0 ? -1 : (mode ? 1 : 0)); }

void lane_row4_set(int mode, uint32_t types) noexcept {
    g_types.store(types, std::memory_order_relaxed);
    lane_row4_override(mode);
}

uint32_t lane_row4_types() noexcept { return g_types.load(std::memory_order_relaxed); }

bool lane_row4_gu(const NativeFmt& f, const uint8_t* gate_base, const uint8_t* up_base, const void* act, float* ff,
                  int r0, int r1, float swiglu_limit) {
    if (!lane_row4_on() || !type_on(f.gu_type) || !row4_supported(f.gu_type)) return false;
    const Kernel k = kernel_for(f.gu_type);
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu((ggml_type) f.gu_type)->vec_dot;
    const int n = (int) f.n_embd, nb = n / kQK;
    const bool clamp = swiglu_limit != std::numeric_limits<float>::infinity();
    // the same per-row finish as native_gu_rows_split
    auto finish = [&](float g, float u) {
        if (clamp) {
            g = std::fmin(g, swiglu_limit);
            u = std::fmin(std::fmax(u, -swiglu_limit), swiglu_limit);
        }
        return (g / (1.f + std::exp(-g))) * u;
    };
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const uint8_t* gr[4] = {gate_base + (size_t) r * f.gu_row, gate_base + (size_t) (r + 1) * f.gu_row,
                                gate_base + (size_t) (r + 2) * f.gu_row, gate_base + (size_t) (r + 3) * f.gu_row};
        const uint8_t* ur[4] = {up_base + (size_t) r * f.gu_row, up_base + (size_t) (r + 1) * f.gu_row,
                                up_base + (size_t) (r + 2) * f.gu_row, up_base + (size_t) (r + 3) * f.gu_row};
        float g[4], u[4];
        k(nb, gr, (const uint8_t*) act, g);
        k(nb, ur, (const uint8_t*) act, u);
        for (int q = 0; q < 4; ++q) ff[r + q] = finish(g[q], u[q]);
    }
    for (; r < r1; ++r) {
        float g = 0.f, u = 0.f;
        dot(n, &g, 0, gate_base + (size_t) r * f.gu_row, 0, act, 0, 1);
        dot(n, &u, 0, up_base + (size_t) r * f.gu_row, 0, act, 0, 1);
        ff[r] = finish(g, u);
    }
    return true;
}

bool lane_row4_down(const NativeFmt& f, const uint8_t* down_base, const void* hq, float* out, int r0, int r1) {
    if (!lane_row4_on() || !type_on(f.d_type) || !row4_supported(f.d_type)) return false;
    row4_rows(f.d_type, (int) f.n_ff, down_base, f.d_row, hq, out, r0, r1);
    return true;
}

}  // namespace strata::kernels::cpu

#else  // no 4-row lane in this build (not x86-64, or STRATA_GLM_LANE_ROW4_KERNELS=OFF): see the top

#include <cstdlib>

namespace strata::kernels::cpu {

bool row4_supported(int) { return false; }

int row4_iq1s_form() { return -1; }

// (both require row4_supported(type), which no type has here)
void row4_dot(int, int, const uint8_t* const[4], const void*, float[4]) { std::abort(); }
void row4_rows(int, int, const uint8_t*, size_t, const void*, float*, int, int) { std::abort(); }

bool lane_row4_on() { return false; }

void lane_row4_override(int) noexcept {}

void lane_row4_set(int, uint32_t) noexcept {}

uint32_t lane_row4_types() noexcept { return 0; }

bool lane_row4_gu(const NativeFmt&, const uint8_t*, const uint8_t*, const void*, float*, int, int, float) {
    return false;
}

bool lane_row4_down(const NativeFmt&, const uint8_t*, const void*, float*, int, int) { return false; }

}  // namespace strata::kernels::cpu

#endif
