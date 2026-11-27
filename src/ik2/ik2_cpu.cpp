// src/ik2/ik2_cpu.cpp - maya-ik2's cache-path CPU lane (the interface: ik2_cpu.hpp).
#include "ik2_cpu.hpp"

#include "hy4_lane.hpp"
#include "maya/kernels/cpu/pool.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>

namespace ik2::cpu {
namespace {

enum Form { kScalar = 0, kAvx2 = 1, kAvx512 = 2 };

int detect() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int r[4];
    __cpuid(r, 0);
    const int max_leaf = r[0];
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osx = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!(fma && osx && avx) || max_leaf < 7) return kScalar;
    const unsigned long long xcr = _xgetbv(0);
    if ((xcr & 6) != 6) return kScalar;
    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] >> 5) & 1, f = (r[1] >> 16) & 1, bw = (r[1] >> 30) & 1, vl = (r[1] >> 31) & 1;
    if (f && bw && vl && (xcr & 0xE6) == 0xE6) return kAvx512;
    return avx2 ? kAvx2 : kScalar;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("fma"))
        return kAvx512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return kAvx2;
    return kScalar;
#else
    return kScalar;
#endif
}

int g_form = -1;
int form() {
    if (g_form < 0) g_form = detect();
    return g_form;
}

int available_physical_cores() {
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof mask, &mask) == 0) {
        std::vector<int> seen;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &mask)) continue;
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
            int first = c;
            if (f >> first) {}
            if (std::find(seen.begin(), seen.end(), first) == seen.end()) seen.push_back(first);
        }
        if (!seen.empty()) return (int) seen.size();
        return CPU_COUNT(&mask);
    }
#endif
    return (int) maya::kernels::cpu::physical_cores(false).size();
}

// ggml_ue4m3_to_fp32 of every byte
struct ScaleTable {
    float v[256];
    ScaleTable() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) {
                v[x] = 0.0f;
                continue;
            }
            const int e = (x >> 3) & 0xF, m = x & 7;
            v[x] = (e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7)) * 0.5f;
        }
    }
};
const ScaleTable g_scale;

#if defined(__x86_64__) || defined(_M_X64)
constexpr int kPrefetch = 2048;

#if defined(__GNUC__) || defined(__clang__)
#define IK2C_T512 __attribute__((target("avx512f,avx512bw,avx512vl,fma")))
#define IK2C_T256 __attribute__((target("avx2,fma")))
#else
#define IK2C_T512
#define IK2C_T256
#endif

// a 36-byte block: four UE4M3 scales, then 8 bytes a 16-value sub-block (value j < 8 the low nibble of byte j)
IK2C_T512 void nvfp4_rows_avx512(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 36, xp += 64) {
            _mm_prefetch((const char*) b + kPrefetch, _MM_HINT_T0);
#define IK2_SUB(sb, acc)                                                                                       \
    {                                                                                                          \
        const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + (sb) * 8));                              \
        const __m128i lo = _mm_and_si128(raw, m4);                                                             \
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);                                          \
        const __m128i val = _mm_shuffle_epi8(tab, _mm_unpacklo_epi64(lo, hi));                                 \
        const __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(val));                                        \
        const __m512 xs = _mm512_mul_ps(_mm512_loadu_ps(xp + (sb) * 16), _mm512_set1_ps(g_scale.v[b[sb]]));    \
        acc = _mm512_fmadd_ps(v, xs, acc);                                                                     \
    }
            IK2_SUB(0, acc0)
            IK2_SUB(1, acc1)
            IK2_SUB(2, acc0)
            IK2_SUB(3, acc1)
#undef IK2_SUB
        }
        out[r - r0] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

IK2C_T256 void nvfp4_rows_avx2(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 36, xp += 64) {
            _mm_prefetch((const char*) b + kPrefetch, _MM_HINT_T0);
            for (int sb = 0; sb < 4; ++sb) {
                const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + sb * 8));
                const __m128i ql = _mm_shuffle_epi8(tab, _mm_and_si128(raw, m4));
                const __m128i qh = _mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16(raw, 4), m4));
                const __m256 d = _mm256_set1_ps(g_scale.v[b[sb]]);
                const __m256 wl = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ql)), d);
                const __m256 wh = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qh)), d);
                acc0 = _mm256_fmadd_ps(wl, _mm256_loadu_ps(xp + sb * 16), acc0);
                acc1 = _mm256_fmadd_ps(wh, _mm256_loadu_ps(xp + sb * 16 + 8), acc1);
            }
        }
        const __m256 s8 = _mm256_add_ps(acc0, acc1);
        const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
        const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        out[r - r0] = _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
    }
}
#endif

void nvfp4_rows_scalar(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        double acc = 0.0;
        for (int i = 0; i < nb; ++i, b += 36)
            for (int sb = 0; sb < 4; ++sb) {
                float a = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t by = b[4 + sb * 8 + j];
                    a += kv[by & 0xF] * x[i * 64 + sb * 16 + j] + kv[by >> 4] * x[i * 64 + sb * 16 + j + 8];
                }
                acc += (double) g_scale.v[b[sb]] * a;
            }
        out[r - r0] = (float) acc;
    }
}

}  // namespace

bool available() { return form() != kScalar; }

int recommended_workers(int threads) {
    const int avail = available_physical_cores();
    const int cap = avail > 2 ? avail - 1 : avail;
    const int use = avail > 0 ? std::min(threads, cap) : threads;
    return std::max(0, use - 1);
}

const char* rows_form() {
    const int f = form();
    return f == kAvx512 ? "avx512" : f == kAvx2 ? "avx2" : "scalar";
}

bool rows_force(const char* name) {
    const int have = detect();
    const int want = !std::strcmp(name, "avx512") ? kAvx512 : !std::strcmp(name, "avx2") ? kAvx2 : kScalar;
    if (want > have) return false;
    g_form = want;
    return true;
}

void nvfp4_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
#if defined(__x86_64__) || defined(_M_X64)
    const int f = form();
    if (f == kAvx512) {
        nvfp4_rows_avx512(W, K, r0, r1, x, out);
        return;
    }
    if (f == kAvx2) {
        nvfp4_rows_avx2(W, K, r0, r1, x, out);
        return;
    }
#endif
    nvfp4_rows_scalar(W, K, r0, r1, x, out);
}

Lane::Lane(int workers) {
    (void) rows_form();   // initialize SIMD dispatch once before workers read it
    lane_ = std::make_unique<hy4::Lane>(workers);
}

Lane::~Lane() = default;

int Lane::threads() const { return lane_->threads() + 1; }

void Lane::run(const ExpertJob* jobs, int n, int C, int F, float gcap, float ucap) {
    if (n <= 0) return;
    const size_t need = (size_t) n * F;
    if (h_.size() < need) h_.resize(need);

    constexpr int kGuRows = 24, kDownRows = 48;
    const int pg = (F + kGuRows - 1) / kGuRows;
    const int pd = (C + kDownRows - 1) / kDownRows;

    lane_->parallel_for(n * pg, [&](int p) {
        const int k = p / pg;
        const int r0 = (p % pg) * kGuRows, r1 = std::min(F, r0 + kGuRows);
        const ExpertJob& J = jobs[k];
        float g[kGuRows], u[kGuRows];
        nvfp4_rows(J.gate, C, r0, r1, J.x, g);
        nvfp4_rows(J.up, C, r0, r1, J.x, u);
        float* h = h_.data() + (size_t) k * F;
        for (int r = r0; r < r1; ++r) {
            const float gv = g[r - r0] * J.sg, uv = u[r - r0] * J.su;
            h[r] = gcap * std::tanh(gv / gcap) * (1.0f / (1.0f + std::exp(-gv))) * ucap * std::tanh(uv / ucap);
        }
    });

    lane_->parallel_for(n * pd, [&](int p) {
        const int k = p / pd;
        const int r0 = (p % pd) * kDownRows, r1 = std::min(C, r0 + kDownRows);
        const ExpertJob& J = jobs[k];
        nvfp4_rows(J.down, F, r0, r1, h_.data() + (size_t) k * F, J.y + r0);
    });
}

}  // namespace ik2::cpu
