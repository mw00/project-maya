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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
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

// AMD Zen / Zen+ (CPUID family 17h, models below 30h, where Zen 2 begins): the CPUs whose default is the auto mode
// (the numbers are further down)
bool cpu_zen() {
    static const bool zen = [] {
        unsigned v[4] = {0, 0, 0, 0}, r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, 0, 0);
        for (int i = 0; i < 4; ++i) v[i] = (unsigned) x[i];
        __cpuidex(x, 1, 0);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(0, 0, v[0], v[1], v[2], v[3]);
        __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
        if (v[1] != 0x68747541u || v[3] != 0x69746E65u || v[2] != 0x444D4163u) return false;   // "AuthenticAMD"
        const unsigned base = (r[0] >> 8) & 0xFu;
        const unsigned family = base + (base == 0xFu ? (r[0] >> 20) & 0xFFu : 0u);
        const unsigned model = (((r[0] >> 16) & 0xFu) << 4) | ((r[0] >> 4) & 0xFu);
        return family == 0x17u && model < 0x30u;
    }();
    return zen;
}

// MAYA_GLM_LANE_ROW4 as the program started: on (the kernel for every type that has one), off, or auto (each type by
// row4_faster).  Unset: on, as before the switch had an auto mode - except on AMD Zen / Zen+, where it is auto.
int env_mode() {
    static const int m = strata::glmenv::env_switch("MAYA_GLM_LANE_ROW4",
                                                    cpu_zen() ? strata::glmenv::kAuto : strata::glmenv::kOn, true);
    return m;
}

// ---- the kernel or ggml-cpu's own dot, on this CPU.  The two give the same bits, so only the time is at stake, and it
// goes either way.  glm_lane_row4_test --bench, one thread, rows read from RAM (~192 MB), the kernel's speed over
// ggml's dot:
//   Threadripper 1950X (Zen)          IQ2_XXS 0.75x, IQ2_XS 0.74x, IQ2_S 0.75x, IQ3_XXS 0.74x, Q2_K 0.63x, Q3_K 0.67x,
//                                     IQ4_XS 0.62x (IQ1_S 17.6x: ggml's takes pdep, microcode there)
//   Core i5-12600K, a performance     IQ2_XXS 1.07x, IQ2_XS 1.06x, IQ1_S 1.35x, IQ2_S 0.99x, IQ4_XS 0.94x,
//   core (medians of five runs)       IQ3_XXS 0.92x, Q2_K 0.89x, Q3_K 0.85x
//   the same CPU, an efficient core   IQ2_XXS 0.75x, IQ2_XS 0.94x, IQ1_S 0.79x, IQ2_S 0.80x, IQ4_XS 0.72x,
//                                     IQ3_XXS 0.71x, Q2_K 0.82x, Q3_K 0.90x
// With the hooks on for every type, as they were, the Threadripper's 16 threads took 0.407 ms for a DeepSeek IQ2_XXS
// expert's one-token call against 0.341 ms without them (three experts 1.058 against 0.885 ms).  The likely reason,
// not taken apart further: the kernels put a chunk's grid and sign vectors together from broadcast loads and dword
// blends, which keeps the one shuffle port of the Xeons they were written on free (glm_lane_row4_avx2.cpp); a core
// that runs a 256-bit operation as two 128-bit ones (Zen, Zen+, Intel's efficient cores) pays for each of them twice,
// and ggml's vmovq / vpinsrq / vinserti128 form stays in 128 bits until its last step.
// So a type takes its kernel where the kernel timed faster here, as the lane runs it: rows of 4096 values read in
// order from more memory than the caches hold (time_bytes), ggml's dot a row and the kernel four rows a call, whole
// passes in turns and the best of each kept (other work on the machine then hits both alike) - once a type on its
// first use: 40 to 130 ms a type on the Threadripper (0.9 s for IQ1_S, at its dot's 0.12 GB/s).
// The rows have to come from RAM, as an expert's do: in the cache the kernels look better against the dot than they
// are from RAM (the i5's performance core: IQ4_XS 1.15x in 1 MB, 0.94x from RAM).  So twice the largest L3
// where Linux names its size, within 32 .. 256 MB; 32 MB elsewhere.
// Where the default is this auto mode: on AMD Zen / Zen+ only (cpu_zen).  Everywhere else the switch stays on unless
// it is set to auto by hand, for what the timing showed while measuring.  It sees only the kind of core its thread
// runs on, and the i5's two kinds disagree.  Near a tie its pick turns from one start to the next (the i5's performance
// core: IQ2_XS the kernel in four runs of five).  And on the idle Threadripper the IQ2_XXS kernel's best pass came out
// 22% and 36% low in two of 16 fresh processes (1.70 and 1.39 GB/s, the median 2.17; none of the 52 timings of other
// types was 10% low on either side; the cause not found): there that only widens the dot's lead, on a CPU where the
// kernel wins by less than that it would turn the pick.
size_t time_bytes() {
    static const size_t bytes = [] {
        size_t l3 = 0;
#if defined(__linux__)
        for (int c = 0; c < 4096; ++c) {
            char path[96];
            std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index3/size", c);
            std::FILE* f = std::fopen(path, "r");
            if (f == nullptr) break;
            unsigned long long v = 0;
            char unit = 0;
            if (std::fscanf(f, "%llu%c", &v, &unit) >= 1) {
                if (unit == 'K') v <<= 10;
                else if (unit == 'M') v <<= 20;
                l3 = std::max(l3, (size_t) v);
            }
            std::fclose(f);
        }
#endif
        return std::min(std::max(2 * l3, (size_t) 32 << 20), (size_t) 256 << 20);
    }();
    return bytes;
}
double row4_now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Row4Time {
    double dot_gbs = 0.0, kernel_gbs = 0.0;   // GB/s of weight rows by ggml's dot / by the kernel (the best pass)
    bool faster = false;                      // the kernel is the one to take
};
Row4Time time_type(int type) {
    Row4Time rt;
    ggml_cpu_init();   // (ggml's vec_dot reads x.d through ggml-cpu's fp16 table on x86)
    constexpr int kN = 4096, kNb = kN / kQK, kRounds = 3;
    const ggml_type t = (ggml_type) type;
    const size_t rb = ggml_row_size(t, kN), bs = ggml_type_size(t);
    const int rows = (int) (time_bytes() / rb / 4 * 4);
    std::vector<uint8_t> w(rb * (size_t) rows), q((size_t) kNb * kQ8K);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    auto rnd = [&s] {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    };
    for (size_t i = 0; i + 8 <= w.size(); i += 8) {
        const uint64_t v = rnd();
        std::memcpy(&w[i], &v, 8);
    }
    // (each block's fp16 scale a small normal value - Q2_K's d and dmin at 80 / 82, Q3_K's d at 108, the others' d
    // first - so no sum runs into NaNs or subnormals)
    const ggml_fp16_t h = ggml_fp32_to_fp16(0.0078125f);
    const size_t at = type == GGML_TYPE_Q2_K ? 80 : type == GGML_TYPE_Q3_K ? 108 : 0;
    for (size_t o = 0; o + bs <= w.size(); o += bs) {
        std::memcpy(&w[o + at], &h, sizeof h);
        if (type == GGML_TYPE_Q2_K) std::memcpy(&w[o + 82], &h, sizeof h);
    }
    for (int b = 0; b < kNb; ++b) {   // (a Q8_K activation: its scale, its values, their sums of 16)
        uint8_t* p = &q[(size_t) b * kQ8K];
        const float d = 0.01f;
        std::memcpy(p, &d, 4);
        for (int j = 0; j < kQK; ++j) p[4 + j] = (uint8_t) (int8_t) ((int) (rnd() % 255u) - 127);
        for (int g = 0; g < 16; ++g) {
            int sum = 0;
            for (int j = 0; j < 16; ++j) sum += (int8_t) p[4 + 16 * g + j];
            const int16_t s16 = (int16_t) sum;
            std::memcpy(p + 260 + 2 * g, &s16, 2);
        }
    }
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu(t)->vec_dot;
    const Kernel k = kernel_for(type);
    std::vector<float> a((size_t) rows), b((size_t) rows);
    auto run_dot = [&](int n) {
        for (int r = 0; r < n; ++r) dot(kN, &a[(size_t) r], 0, &w[(size_t) r * rb], 0, q.data(), 0, 1);
    };
    auto run_kernel = [&](int n) {
        for (int r = 0; r < n; r += 4) {
            const uint8_t* rw[4] = {&w[(size_t) r * rb], &w[(size_t) (r + 1) * rb], &w[(size_t) (r + 2) * rb],
                                    &w[(size_t) (r + 3) * rb]};
            k(kNb, rw, q.data(), &b[(size_t) r]);
        }
    };
    run_dot(64);   // (the code and the tables into the caches)
    run_kernel(64);
    double td = 1e30, tk = 1e30;
    for (int round = 0; round < kRounds; ++round) {
        const double t0 = row4_now_us();
        run_dot(rows);
        const double t1 = row4_now_us();
        run_kernel(rows);
        const double t2 = row4_now_us();
        td = std::min(td, t1 - t0);
        tk = std::min(tk, t2 - t1);
    }
    rt.dot_gbs = (double) w.size() / td / 1e3;
    rt.kernel_gbs = (double) w.size() / tk / 1e3;
    // (the same rows either way - the kernel's are checked against the dot's here once more - so a tie may go to
    // either; it goes to the kernel, the lane's choice before this)
    rt.faster = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0 && tk <= td;
    return rt;
}
const Row4Time& type_time(int type) {   // (type: one row4_supported() holds for)
    static std::once_flag once[32];
    static Row4Time rt[32];
    std::call_once(once[type], [type] {
        rt[type] = time_type(type);
        if (std::getenv("MAYA_GLM_LANE_ROW4_VERBOSE") != nullptr)
            std::fprintf(stderr, "strata: CPU lane %s one-token rows from RAM (%d MB): ggml-cpu's dot %.2f GB/s, the "
                                 "4-row kernel %.2f GB/s -> %s\n", ggml_type_name((ggml_type) type),
                         (int) (time_bytes() >> 20), rt[type].dot_gbs, rt[type].kernel_gbs,
                         rt[type].faster ? "the kernel" : "ggml-cpu's dot");
    });
    return rt[type];
}

// the hooks take `type`: the switch, the mask and the kernel as before - and, in the auto mode (MAYA_GLM_LANE_ROW4=auto,
// the default on AMD Zen / Zen+, lane_row4_override(2)), only where the kernel is the faster of the two here
bool lane_takes(int type) {
    const int o = g_override.load(std::memory_order_relaxed);
    const int mode = o < 0 ? env_mode() : o == 0 ? strata::glmenv::kOff : o == 1 ? strata::glmenv::kOn
                                                                                 : strata::glmenv::kAuto;
    if (mode == strata::glmenv::kOff || !type_on(type) || !row4_supported(type)) return false;
    return mode == strata::glmenv::kOn || type_time(type).faster;
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
    return env_mode() != strata::glmenv::kOff;   // (on, or auto: each type where its kernel is the faster)
}

bool lane_row4_auto_default() { return cpu_ok() && cpu_zen(); }

bool row4_faster(int type) { return row4_supported(type) && type_time(type).faster; }

bool lane_row4_takes(int type) { return lane_takes(type); }

void lane_row4_override(int mode) noexcept { g_override.store(mode < 0 ? -1 : mode > 2 ? 1 : mode); }

void lane_row4_set(int mode, uint32_t types) noexcept {
    g_types.store(types, std::memory_order_relaxed);
    lane_row4_override(mode);
}

uint32_t lane_row4_types() noexcept { return g_types.load(std::memory_order_relaxed); }

bool lane_row4_gu(const NativeFmt& f, const uint8_t* gate_base, const uint8_t* up_base, const void* act, float* ff,
                  int r0, int r1, float swiglu_limit) {
    if (!lane_takes(f.gu_type)) return false;
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
    if (!lane_takes(f.d_type)) return false;
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

bool lane_row4_auto_default() { return false; }

bool row4_faster(int) { return false; }

bool lane_row4_takes(int) { return false; }

void lane_row4_override(int) noexcept {}

void lane_row4_set(int, uint32_t) noexcept {}

uint32_t lane_row4_types() noexcept { return 0; }

bool lane_row4_gu(const NativeFmt&, const uint8_t*, const uint8_t*, const void*, float*, int, int, float) {
    return false;
}

bool lane_row4_down(const NativeFmt&, const uint8_t*, const void*, float*, int, int) { return false; }

}  // namespace strata::kernels::cpu

#endif
