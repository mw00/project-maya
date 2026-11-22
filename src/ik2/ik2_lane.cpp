// src/ik2/ik2_lane.cpp - maya-ik2's CPU lane (ik2_lane.hpp).  Built without contraction (-ffp-contract=off, MSVC
// /fp:strict): every sum is the fused multiply-add or the rounding ik2_exact.h names, so the rows are the GPU's bits.
#include "ik2_lane.hpp"

#include "hy4_lane.hpp"
#include "ik2_exact.h"
#include "ik2_off.cuh"
#include "maya/kernels/cpu/pool.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace ik2::lane {
namespace {

#if defined(__GNUC__) || defined(__clang__)
#define IK2L_T512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
#define IK2L_T256 __attribute__((target("avx2,fma")))
#else
#define IK2L_T512
#define IK2L_T256
#endif

// ---- the CPU's forms
enum Form { kScalar = 0, kAvx2 = 1, kAvx512 = 2 };
int detect() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    const int max_leaf = r[0];
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osx = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!(fma && osx && avx) || max_leaf < 7) return kScalar;
    const unsigned long long xcr = _xgetbv(0);
    if ((xcr & 6) != 6) return kScalar;
    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] >> 5) & 1, f = (r[1] >> 16) & 1, dq = (r[1] >> 17) & 1, bw = (r[1] >> 30) & 1,
               vl = (r[1] >> 31) & 1;
    if (f && dq && bw && vl && (xcr & 0xE6) == 0xE6) return kAvx512;
    return avx2 ? kAvx2 : kScalar;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("avx512dq"))
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

// ---- the 16-lane sums (ik2_exact.h's tree16)
IK2L_T256 float tree256(__m256 lo, __m256 hi) {   // lanes 0-7 and 8-15
    const __m256 t8 = _mm256_add_ps(lo, hi);
    const __m128 t4 = _mm_add_ps(_mm256_castps256_ps128(t8), _mm256_extractf128_ps(t8, 1));
    const __m128 t2 = _mm_add_ps(t4, _mm_movehl_ps(t4, t4));
    return _mm_cvtss_f32(_mm_add_ss(t2, _mm_shuffle_ps(t2, t2, 1)));
}
IK2L_T512 float tree512(__m512 l) {
    const __m256 lo = _mm512_castps512_ps256(l), hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(l), 1));
    const __m256 t8 = _mm256_add_ps(lo, hi);
    const __m128 t4 = _mm_add_ps(_mm256_castps256_ps128(t8), _mm256_extractf128_ps(t8, 1));
    const __m128 t2 = _mm_add_ps(t4, _mm_movehl_ps(t4, t4));
    return _mm_cvtss_f32(_mm_add_ss(t2, _mm_shuffle_ps(t2, t2, 1)));
}

// an NVFP4 scale byte's value, ik2_exact.h's ue4m3 (exact), read from a table
struct ScaleLut {
    float v[256];
    ScaleLut() {
        for (int i = 0; i < 256; ++i) v[i] = x::ue4m3((uint8_t) i);
    }
};
const ScaleLut g_sl;

// ---- AVX-512: a sub-block = one register of 16 lanes
// the 16 exact weights of sub-block c of the 36-byte block at b
IK2L_T512 inline __m512 nvfp4_w512(const uint8_t* b, int c, __m128i tab, __m128i m4) {
    const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + 8 * c));
    const __m128i idx = _mm_unpacklo_epi64(_mm_and_si128(raw, m4), _mm_and_si128(_mm_srli_epi16(raw, 4), m4));
    return _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(tab, idx))), _mm512_set1_ps(g_sl.v[b[c]]));
}
IK2L_T512 void nvfp4_512(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    int r = r0;
    for (; r + 1 < r1; r += 2) {   // (two rows a pass: x read once, eight chains in flight)
        const uint8_t *b = W + (size_t) r * rb, *b2 = b + rb;
        __m512 a[4], c2[4];
        for (int c = 0; c < 4; ++c) a[c] = _mm512_setzero_ps(), c2[c] = _mm512_setzero_ps();
        for (int i = 0; i < nb; ++i, b += 36, b2 += 36) {
            const float* xp = x + 64 * i;
            for (int c = 0; c < 4; ++c) {
                const __m512 xv = _mm512_loadu_ps(xp + 16 * c);
                a[c] = _mm512_fmadd_ps(xv, nvfp4_w512(b, c, tab, m4), a[c]);
                c2[c] = _mm512_fmadd_ps(xv, nvfp4_w512(b2, c, tab, m4), c2[c]);
            }
        }
        out[r - r0] = tree512(_mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3])));
        out[r + 1 - r0] = tree512(_mm512_add_ps(_mm512_add_ps(c2[0], c2[1]), _mm512_add_ps(c2[2], c2[3])));
    }
    for (; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m512 a[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int i = 0; i < nb; ++i, b += 36) {
            const float* xp = x + 64 * i;
            for (int c = 0; c < 4; ++c) a[c] = _mm512_fmadd_ps(_mm512_loadu_ps(xp + 16 * c), nvfp4_w512(b, c, tab, m4), a[c]);
        }
        out[r - r0] = tree512(_mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3])));
    }
}
IK2L_T512 void q8_512(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const int nb = K / 32;
    const size_t rb = (size_t) nb * 34;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m512 a[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int i = 0; i < nb; ++i, b += 34) {   // (block i: sub-blocks 2i, 2i + 1 - chains (2i) % 4, (2i + 1) % 4)
            const __m512 d = _mm512_set1_ps(x::h2f((uint16_t) (b[0] | (b[1] << 8))));
            for (int h = 0; h < 2; ++h) {
                const int s = 2 * i + h;
                const __m512 w = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (b + 2 + 16 * h)))), d);
                a[s & 3] = _mm512_fmadd_ps(_mm512_loadu_ps(x + 16 * s), w, a[s & 3]);
            }
        }
        out[r - r0] = tree512(_mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3])));
    }
}

// ---- AVX2: a sub-block = two registers (lanes 0-7: the low nibbles, 8-15: the high ones)
IK2L_T256 void nvfp4_256(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m256 lo[4], hi[4];
        for (int c = 0; c < 4; ++c) lo[c] = _mm256_setzero_ps(), hi[c] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i, b += 36) {
            const float* xp = x + 64 * i;
            for (int c = 0; c < 4; ++c) {
                const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + 8 * c));
                const __m128i ql = _mm_shuffle_epi8(tab, _mm_and_si128(raw, m4));
                const __m128i qh = _mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16(raw, 4), m4));
                const __m256 d = _mm256_set1_ps(g_sl.v[b[c]]);
                const __m256 wl = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ql)), d);
                const __m256 wh = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qh)), d);
                lo[c] = _mm256_fmadd_ps(_mm256_loadu_ps(xp + 16 * c), wl, lo[c]);
                hi[c] = _mm256_fmadd_ps(_mm256_loadu_ps(xp + 16 * c + 8), wh, hi[c]);
            }
        }
        const __m256 l = _mm256_add_ps(_mm256_add_ps(lo[0], lo[1]), _mm256_add_ps(lo[2], lo[3]));
        const __m256 h = _mm256_add_ps(_mm256_add_ps(hi[0], hi[1]), _mm256_add_ps(hi[2], hi[3]));
        out[r - r0] = tree256(l, h);
    }
}
IK2L_T256 void q8_256(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const int nb = K / 32;
    const size_t rb = (size_t) nb * 34;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m256 lo[4], hi[4];
        for (int c = 0; c < 4; ++c) lo[c] = _mm256_setzero_ps(), hi[c] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i, b += 34) {
            const __m256 d = _mm256_set1_ps(x::h2f((uint16_t) (b[0] | (b[1] << 8))));
            for (int h = 0; h < 2; ++h) {
                const int s = 2 * i + h;
                const __m128i q = _mm_loadu_si128((const __m128i*) (b + 2 + 16 * h));
                const __m256 wl = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)), d);
                const __m256 wh = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8))), d);
                lo[s & 3] = _mm256_fmadd_ps(_mm256_loadu_ps(x + 16 * s), wl, lo[s & 3]);
                hi[s & 3] = _mm256_fmadd_ps(_mm256_loadu_ps(x + 16 * s + 8), wh, hi[s & 3]);
            }
        }
        const __m256 l = _mm256_add_ps(_mm256_add_ps(lo[0], lo[1]), _mm256_add_ps(lo[2], lo[3]));
        const __m256 hh = _mm256_add_ps(_mm256_add_ps(hi[0], hi[1]), _mm256_add_ps(hi[2], hi[3]));
        out[r - r0] = tree256(l, hh);
    }
}

void rows_scalar(int ty, const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const size_t rb = x::row_bytes(ty, K);
    for (int r = r0; r < r1; ++r) out[r - r0] = x::row_dot(ty, W + (size_t) r * rb, K, x);
}

}  // namespace

void rows(int ty, const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const int f = form();
    if (f == kAvx512) return ty == x::kNVFP4 ? nvfp4_512(W, K, r0, r1, x, out) : q8_512(W, K, r0, r1, x, out);
    if (f == kAvx2) return ty == x::kNVFP4 ? nvfp4_256(W, K, r0, r1, x, out) : q8_256(W, K, r0, r1, x, out);
    rows_scalar(ty, W, K, r0, r1, x, out);
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

// ---- the server
namespace {
// (a piece's rows: ~84 pieces a visit's expert, gate / up then down - small enough that a visit's 2-6 experts
//  spread evenly over a 38-thread lane)
constexpr int kGuRows = 32, kDownRows = 48;
}

Server::Server(int threads, int C, int F, float gcap, float ucap, std::vector<LayerW> layers, void* job, void* done)
    : C_(C), F_(F), gcap_(gcap), ucap_(ucap), L_(std::move(layers)), job_((uint8_t*) job), done_((uint8_t*) done) {
    (void) rows_form();   // (the CPU's form decided before the workers use it)
    lane_ = std::make_unique<hy4::Lane>(threads);
    h_.assign((size_t) k::kOffMaxK * F_, 0.0f);
    us_[0].store(0.0), us_[1].store(0.0);
    th_ = std::thread([this] { loop(); });
}
void Server::set_threads(int n) {
    lane_.reset();
    lane_ = std::make_unique<hy4::Lane>(n);
}
Server::~Server() {
    quit_.store(true);
    if (th_.joinable()) th_.join();
    if (std::getenv("IK2_LANE_LOG"))
        for (size_t i = 0; i < hist_.size(); ++i)
            if (hist_[i].n)
                std::printf("lane jobs of %zu experts: %llu, %.0f us on average, %.0f at most\n", i + 1, hist_[i].n,
                            hist_[i].sum / (double) hist_[i].n, hist_[i].max);
}
int Server::threads() const { return lane_->threads() + 1; }
double Server::us_expert(int fmt) const { return us_[fmt & 1].load(); }

void Server::compute(int layer, const int* e, const float* sg, const float* su, int n, const float* xin, float* y) {
    const LayerW& Ly = L_[(size_t) layer];
    const int pg = (F_ + kGuRows - 1) / kGuRows, pd = (C_ + kDownRows - 1) / kDownRows;
    lane_->parallel_for(n * pg, [&](int p) {
        const int k = p / pg, r0 = (p % pg) * kGuRows, r1 = std::min(F_, r0 + kGuRows);
        const uint8_t* base = Ly.base + (size_t) e[k] * Ly.stride;
        float g[kGuRows], u[kGuRows];
        rows(Ly.ty_gu, base, C_, r0, r1, xin, g);
        rows(Ly.ty_gu, base + Ly.off_up, C_, r0, r1, xin, u);
        float* h = h_.data() + (size_t) k * F_;
        for (int r = r0; r < r1; ++r)
            h[r] = x::swiglu(x::mul(g[r - r0], sg[k]), x::mul(u[r - r0], su[k]), gcap_, ucap_);
    });
    lane_->parallel_for(n * pd, [&](int p) {
        const int k = p / pd, r0 = (p % pd) * kDownRows, r1 = std::min(C_, r0 + kDownRows);
        const uint8_t* base = Ly.base + (size_t) e[k] * Ly.stride + Ly.off_down;
        rows(Ly.ty_d, base, F_, r0, r1, h_.data() + (size_t) k * F_, y + (size_t) k * C_ + r0);
    });
}

double Server::time_experts(int layer, const int* e, int n, const float* xin, int reps) {
    std::vector<float> sg((size_t) n, 1.0f), y((size_t) n * C_);
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        compute(layer, e, sg.data(), sg.data(), n, xin, y.data());
        best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    return best / n;
}

void Server::run(uint32_t seq) {
    const auto* J = (const k::LaneJob*) job_;
    const int n = (int) std::min<uint32_t>(J->n, (uint32_t) k::kOffMaxK);
    const int layer = J->layer;
    int e[k::kOffMaxK];
    float sg[k::kOffMaxK], su[k::kOffMaxK];
    for (int i = 0; i < n; ++i) e[i] = J->e[i], sg[i] = J->sg[i], su[i] = J->su[i];
    const auto t0 = std::chrono::steady_clock::now();
    compute(layer, e, sg, su, n, (const float*) (job_ + k::kLaneJobX), (float*) (done_ + k::kLaneDoneY));
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    const int fmt = L_[(size_t) layer].ty_gu == x::kNVFP4 ? 0 : 1;
    if (n > 0 && n <= k::kOffMaxK) {   // (IK2_LANE_LOG: a job's time by its size)
        Hist& hh = hist_[(size_t) n - 1];
        ++hh.n, hh.sum += us, hh.max = std::max(hh.max, us);
    }
    // (an expert's share: past the round trip the split already counts once a job - 30 us, LaneCost::lat - so small
    //  jobs do not make the lane look slower an expert than it is and push its share onto the bus)
    const double old = us_[fmt].load(), per = std::max(0.25 * us, us - 30.0) / std::max(1, n);
    us_[fmt].store(old > 0 ? 0.9 * old + 0.1 * per : per);
    n_done_.fetch_add((unsigned long long) n);
    std::atomic_thread_fence(std::memory_order_release);
    *(volatile uint32_t*) done_ = seq;
}

void Server::loop() {
    uint32_t last = *(volatile uint32_t*) job_;
    auto idle_since = std::chrono::steady_clock::now();
    while (!quit_.load(std::memory_order_relaxed)) {
        const uint32_t seq = *(volatile uint32_t*) job_;
        if (seq != last) {
            last = seq;
            std::atomic_thread_fence(std::memory_order_acquire);
            run(seq);
            idle_since = std::chrono::steady_clock::now();
            continue;
        }
        // (no job: spin for 20 ms after the last one - a token's visits come a fraction of a millisecond apart -
        //  then nap, longer once idle for a second)
        const double idle = std::chrono::duration<double>(std::chrono::steady_clock::now() - idle_since).count();
        if (idle < 0.02) {
            _mm_pause();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(idle < 1.0 ? 50 : 500));
        }
    }
}

}  // namespace ik2::lane
