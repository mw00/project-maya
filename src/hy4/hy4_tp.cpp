// src/hy4/hy4_tp.cpp - maya-hy4's tensor parallel: the host's threads (see hy4_tp.hpp).
#include "hy4_tp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__x86_64__)   // (GCC / Clang: the F16C sums below; MSVC takes the scalar loop)
#define HY4_X86 1
#endif
#define HY4_PAUSE() _mm_pause()
#else
#define HY4_PAUSE() std::this_thread::yield()
#endif

namespace hy4 {

// ---------------------------------------------------------------- Crew
Crew::Crew(int n) {
    for (int i = 0; i < n; ++i)
        th_.emplace_back([this, i] {
            unsigned seen = 0;
            for (;;) {
                std::function<void(int)> f;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [&] { return quit_ || gen_ != seen; });
                    if (quit_) return;
                    seen = gen_;
                    f = f_;
                }
                f(i);
                std::lock_guard<std::mutex> lk(mu_);
                if (--left_ == 0) done_cv_.notify_all();
            }
        });
}
Crew::~Crew() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
}
void Crew::start(const std::function<void(int)>& f) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        f_ = f;
        left_ = (int) th_.size();
        ++gen_;
    }
    cv_.notify_all();
}
void Crew::wait() {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return left_ == 0; });
}

// ---------------------------------------------------------------- Summer
namespace {
constexpr size_t kTile = 1024;   // (values a pass of the parts: an accumulator in L1)
constexpr size_t kPerThread = 2048;   // (values a thread takes at least: below it fewer threads join)
constexpr double kSpinS = 0.005;      // (a helper spins this long after its last sum, then sleeps)
constexpr size_t kAddTeam = 65536;    // (add(): values from which the helpers join)
}  // namespace

Summer::Summer(int helpers, const std::vector<int>& cpus) {
    for (int h = 1; h <= helpers; ++h)
        th_.emplace_back([this, h, cpus] {
            pin_to(cpus);
            unsigned seen = gen_.load();
            auto last = std::chrono::steady_clock::now();
            for (;;) {
                const unsigned gg = gen_.load(std::memory_order_acquire);
                if (quit_.load(std::memory_order_relaxed)) return;
                if (gg == seen) {
                    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - last).count() < kSpinS) {
                        for (int i = 0; i < 64; ++i) HY4_PAUSE();
                        continue;
                    }
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait_for(lk, std::chrono::milliseconds(50),
                                 [&] { return quit_.load() || gen_.load() != seen; });
                    continue;
                }
                seen = gg;
                // (every helper takes part in every threaded sum - an empty range past n - so the job's fields are
                //  only rewritten after each one has read them)
                run(std::min(n_, per_ * (size_t) h), std::min(n_, per_ * (size_t) (h + 1)));
                done_.fetch_add(1, std::memory_order_release);
                last = std::chrono::steady_clock::now();
            }
        });
}
Summer::~Summer() {
    quit_.store(true);
    {
        std::lock_guard<std::mutex> lk(mu_);
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
}
namespace {
inline float h2f(uint16_t h) {   // (IEEE half to float, exact)
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while (!(mm & 0x400));
            f = s | ((uint32_t) (127 - 15 - ee) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) {
        f = s | 0x7f800000u | (m << 13);
    } else {
        f = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}
inline uint16_t f2h(float x) {   // (float to IEEE half, round to nearest even; |x| < 65504 here)
    uint32_t b;
    std::memcpy(&b, &x, 4);
    const uint32_t s = (b >> 16) & 0x8000;
    const int e = (int) ((b >> 23) & 0xff) - 127 + 15;
    uint32_t m = b & 0x7fffff;
    if (((b >> 23) & 0xff) == 0xff && m) return (uint16_t) (s | 0x7e00);   // (a NaN stays one, as F16C's)
    if (e <= 0) {   // (a half's subnormal, or zero)
        if (e < -10) return (uint16_t) s;
        m |= 0x800000;
        const int sh = 14 - e;
        uint32_t hm = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (hm & 1))) ++hm;
        return (uint16_t) (s | hm);
    }
    uint32_t hm = m >> 13;
    const uint32_t rem = m & 0x1fff;
    uint32_t he = (uint32_t) e;
    if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) {
        if (++hm == 0x400) {
            hm = 0;
            ++he;
        }
    }
    if (he >= 31) return (uint16_t) (s | 0x7c00);
    return (uint16_t) (s | (he << 10) | hm);
}
#if HY4_X86
// (no fma in the target: the products and sums round as the scalar loop's do)
__attribute__((target("f16c,avx2"))) void row16_f16c(const uint16_t* const* p, const float* sc, int np, int E, float* acc) {
    for (int k = 0; k + 8 <= E; k += 8) {
        __m256 a = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (p[0] + k))), _mm256_set1_ps(sc[0]));
        for (int j = 1; j < np; ++j)
            a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (p[j] + k))),
                                               _mm256_set1_ps(sc[j])));
        _mm256_storeu_ps(acc + k, a);
    }
}
__attribute__((target("f16c,avx2"))) void pack16_f16c(const float* acc, int E, float inv, uint16_t* out) {
    const __m256 vi = _mm256_set1_ps(inv);
    for (int k = 0; k + 8 <= E; k += 8)
        _mm_storeu_si128((__m128i*) (out + k), _mm256_cvtps_ph(_mm256_mul_ps(_mm256_loadu_ps(acc + k), vi), 0));
}
const bool g_f16c = __builtin_cpu_supports("f16c") && __builtin_cpu_supports("avx2");
#endif
}  // namespace

void Summer::run16(size_t r0, size_t r1) const {
    std::vector<float> acc((size_t) E_);
    std::vector<const uint16_t*> p((size_t) n_parts_);
    std::vector<float> sc((size_t) n_parts_);
    for (size_t r = r0; r < r1; ++r) {
        for (int j = 0; j < n_parts_; ++j) {
            p[(size_t) j] = parts16_ + (size_t) j * stride_ + r * (size_t) E_;
            sc[(size_t) j] = scales_[(size_t) j * sstride_ + r];
        }
        int k0 = 0;
#if HY4_X86
        if (g_f16c && E_ % 8 == 0) {
            row16_f16c(p.data(), sc.data(), n_parts_, E_, acc.data());
            k0 = E_;
        }
#endif
        for (int k = k0; k < E_; ++k) {   // (the same products and order, without F16C)
            float a = h2f(p[0][k]) * sc[0];
            for (int j = 1; j < n_parts_; ++j) a += h2f(p[(size_t) j][k]) * sc[(size_t) j];
            acc[(size_t) k] = a;
        }
        float am = 0.0f;
        for (int k = 0; k < E_; ++k) am = std::max(am, std::fabs(acc[(size_t) k]));
        float f = 1.0f;
        while (am / f > 32768.0f && f < 1e30f) f *= 2.0f;
        osc_[r] = f;
        const float inv = 1.0f / f;
        uint16_t* o = out16_ + r * (size_t) E_;
#if HY4_X86
        if (g_f16c && E_ % 8 == 0) {
            pack16_f16c(acc.data(), E_, inv, o);
            continue;
        }
#endif
        for (int k = 0; k < E_; ++k) o[k] = f2h(acc[(size_t) k] * inv);
    }
}
void Summer::sum16(uint16_t* out, float* osc, const uint16_t* parts, const float* scales, int n_parts, size_t stride,
                   size_t sstride, size_t rows, int E) {
    mode_ = 3;
    out16_ = out;
    osc_ = osc;
    parts16_ = parts;
    scales_ = scales;
    n_parts_ = n_parts;
    stride_ = stride;
    sstride_ = sstride;
    E_ = E;
    n_ = rows;
    go(rows);
    mode_ = 0;
}
void Summer::run(size_t a0, size_t a1) const {
    if (mode_ == 3) {
        run16(a0, a1);
        return;
    }
    if (mode_ != 0) {   // (add(): dst = src, or dst += src)
        if (mode_ == 1) std::memcpy(dst_ + a0, parts_ + a0, (a1 - a0) * sizeof(float));
        else
            for (size_t i = a0; i < a1; ++i) dst_[i] += parts_[i];
        return;
    }
    float acc[kTile];
    for (size_t i = a0; i < a1; i += kTile) {
        const size_t m = std::min(kTile, a1 - i);
        const float* p0 = parts_ + i;
        for (size_t k = 0; k < m; ++k) acc[k] = p0[k];
        for (int j = 1; j < n_parts_; ++j) {
            const float* pj = parts_ + (size_t) j * stride_ + i;
            for (size_t k = 0; k < m; ++k) acc[k] += pj[k];
        }
        if (i < n_short_) {
            const size_t ms = std::min(m, n_short_ - i);
            for (int j = n_parts_; j < n_parts_ + n_extra_; ++j) {
                const float* pj = parts_ + (size_t) j * stride_ + i;
                for (size_t k = 0; k < ms; ++k) acc[k] += pj[k];
            }
        }
        std::memcpy(dst_ + i, acc, m * sizeof(float));
    }
}
void Summer::add(float* dst, const float* src, size_t n, bool first) {
    dst_ = dst;
    parts_ = src;
    n_parts_ = 1;
    n_extra_ = 0;
    n_short_ = 0;
    stride_ = 0;
    n_ = n;
    mode_ = first ? 1 : 2;
    if (n < kAddTeam) run(0, n);   // (a window's part: the caller alone - the team's hand-offs cost more than it)
    else go(n);
    mode_ = 0;
}
void Summer::sum(float* dst, const float* parts, int n_parts, size_t stride, size_t n, int n_extra, size_t n_short) {
    mode_ = 0;
    dst_ = dst;
    parts_ = parts;
    n_parts_ = n_parts;
    n_extra_ = n_extra;
    n_short_ = std::min(n_short, n);
    stride_ = stride;
    n_ = n;
    go(n);
}
void Summer::go(size_t n) {
    const int threads = (int) th_.size() + 1;
    if (th_.empty() || (mode_ != 3 && n < 2 * kPerThread) || (mode_ == 3 && n < 4)) {   // (a token's few thousand values: the caller alone)
        per_ = n;
        run(0, n);
        return;
    }
    per_ = ((n + (size_t) threads - 1) / (size_t) threads + 15) / 16 * 16;
    const unsigned d0 = done_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(mu_);   // (no helper between its check and its sleep)
        gen_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    run(0, std::min(n, per_));
    const unsigned want = d0 + (unsigned) th_.size();
    while ((int) (done_.load(std::memory_order_acquire) - want) < 0) HY4_PAUSE();
}

// ---------------------------------------------------------------- NUMA
int gpu_numa_node(int dev) {
#if defined(__linux__)
    char bus[64] = {0};
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    std::string id(bus);
    for (char& c : id) c = (char) std::tolower((unsigned char) c);
    // (CUDA gives an 8-digit domain: sysfs names 4)
    if (id.size() > 12 && id.compare(0, 4, "0000") == 0) id = id.substr(4);
    std::ifstream f("/sys/bus/pci/devices/" + id + "/numa_node");
    int n = -1;
    if (f >> n) return n;
#else
    (void) dev;
#endif
    return -1;
}
std::vector<int> node_cpus(int node) {
    std::vector<int> v;
#if defined(__linux__)
    if (node < 0) return v;
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string s;
    if (!std::getline(f, s)) return v;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        const size_t d = part.find('-');
        const int a = std::atoi(part.c_str()), b = d == std::string::npos ? a : std::atoi(part.c_str() + d + 1);
        for (int c = a; c <= b; ++c) v.push_back(c);
    }
#else
    (void) node;
#endif
    return v;
}
bool bind_memory_to(int node) {
#if defined(__linux__) && defined(SYS_set_mempolicy)
    if (node < 0 || node >= 128) return false;
    unsigned long mask[2] = {0, 0};
    mask[node / 64] = 1ul << (node % 64);
    return syscall(SYS_set_mempolicy, 2 /* MPOL_BIND */, mask, (unsigned long) 129) == 0;
#else
    (void) node;
    return false;
#endif
}
void unbind_memory() {
#if defined(__linux__) && defined(SYS_set_mempolicy)
    (void) syscall(SYS_set_mempolicy, 0 /* MPOL_DEFAULT */, nullptr, (unsigned long) 0);
#endif
}
void pin_to(const std::vector<int>& cpus) {
#if defined(__linux__)
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
        if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
    (void) pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#else
    (void) cpus;
#endif
}

}  // namespace hy4
