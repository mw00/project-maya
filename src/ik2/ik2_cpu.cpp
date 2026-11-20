// src/ik2/ik2_cpu.cpp - maya-ik2's CPU lane (the interface: ik2_cpu.hpp).
#include "ik2_cpu.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>

namespace ik2::cpu {

bool available() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
    return false;
#endif
}

Pool::Pool(int n) {
    for (int i = 0; i < n; ++i)
        th_.emplace_back([this]() {
            for (;;) {
                std::function<void()> f;
                {
                    // spin a while before sleeping: an expert's chunks come ~1 ms apart, a wake-up costs tens of us
                    std::unique_lock<std::mutex> lk(m_, std::defer_lock);
                    for (int spin = 0;; ++spin) {
                        lk.lock();
                        if (stop_ || !q_.empty()) break;
                        if (spin > 20000) {
                            cv_.wait(lk, [this]() { return stop_ || !q_.empty(); });
                            break;
                        }
                        lk.unlock();
                        _mm_pause();
                    }
                    if (stop_ && q_.empty()) return;
                    f = std::move(q_.front());
                    q_.pop_front();
                }
                f();
                {
                    std::lock_guard<std::mutex> lk(m_);
                    if (--pending_ == 0) done_.notify_all();
                }
            }
        });
}
Pool::~Pool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
}
void Pool::post(std::function<void()> f) {
    {
        std::lock_guard<std::mutex> lk(m_);
        q_.push_back(std::move(f));
        ++pending_;
    }
    cv_.notify_one();
}
void Pool::wait() {
    std::unique_lock<std::mutex> lk(m_, std::defer_lock);
    for (int spin = 0; spin < 200000; ++spin) {   // (the caller spins too: the work is short)
        lk.lock();
        if (pending_ == 0) return;
        lk.unlock();
        _mm_pause();
    }
    lk.lock();
    done_.wait(lk, [this]() { return pending_ == 0; });
}

namespace {

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

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// a 36-byte block: four UE4M3 scales, then 8 bytes a 16-value sub-block (value j < 8 the low nibble of byte j)
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
void nvfp4_rows_avx512(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 36, xp += 64) {
#define IK2_SUB(sb, acc)                                                                       \
    {                                                                                          \
        const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + (sb) * 8));             \
        const __m128i lo = _mm_and_si128(raw, m4);                                             \
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);                          \
        const __m128i val = _mm_shuffle_epi8(tab, _mm_unpacklo_epi64(lo, hi));                 \
        const __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(val));                        \
        const __m512 xs = _mm512_mul_ps(_mm512_loadu_ps(xp + (sb) * 16), _mm512_set1_ps(g_scale.v[b[sb]])); \
        acc = _mm512_fmadd_ps(v, xs, acc);                                                     \
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

void nvfp4_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool fast = available();
    if (fast) {
        nvfp4_rows_avx512(W, K, r0, r1, x, out);
        return;
    }
#endif
    nvfp4_rows_scalar(W, K, r0, r1, x, out);
}

void post_expert(Pool& pool, ExpertJob* job, int C, int F, float gcap, float ucap) {
    job->gu.resize((size_t) 2 * F);
    job->h.resize(F);
    const int n = std::max(1, pool.size());
    job->left = 2 * n;   // gate's chunks and up's
    auto down = [&pool, job, C, F, n]() {
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) C * c / n), r1 = (int) ((long) C * (c + 1) / n);
            pool.post([job, F, r0, r1]() { nvfp4_rows(job->down, F, r0, r1, job->h.data(), job->y + r0); });
        }
    };
    for (int part = 0; part < 2; ++part)
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) F * c / n), r1 = (int) ((long) F * (c + 1) / n);
            pool.post([job, C, F, gcap, ucap, part, r0, r1, down]() {
                nvfp4_rows(part ? job->up : job->gate, C, r0, r1, job->x, job->gu.data() + (size_t) part * F + r0);
                if (--job->left == 0) {   // the last chunk: the activation, then down's rows
                    for (int i = 0; i < F; ++i) {
                        const float g = job->gu[i] * job->sg, u = job->gu[(size_t) F + i] * job->su;
                        job->h[i] = gcap * std::tanh(g / gcap) * (1.0f / (1.0f + std::exp(-g))) * ucap * std::tanh(u / ucap);
                    }
                    down();
                }
            });
        }
}

}  // namespace ik2::cpu
