// src/ik2/ik2_cpu.cpp - maya-ik2's CPU lane (the interface: ik2_cpu.hpp).
#include "ik2_cpu.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace ik2::cpu {

bool available() { return nvfp4_ok(); }

bool nvfp4_ok() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
    return false;
#endif
}

bool native_ok() { return strata::kernels::cpu::native_experts_available(); }

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

// ---- IK2_CPU_FRAC: the GGUF forms through ggml-cpu's quantized dots (see the header)

bool native_layer(int ty0, int ty1, int ty2, int64_t C, int64_t F, size_t slot_bytes, NativeLayer& l) {
    std::string err;
    l.ok = strata::kernels::cpu::native_fmt_mixed(ty0, ty1, ty2, C, F, l.f, err) && l.f.bytes == slot_bytes;
    return l.ok;
}

void post_expert_native(Pool& pool, NativeJob* job, int C, int F, float gcap, float ucap) {
    namespace kn = strata::kernels::cpu;
    const kn::NativeFmt* fp = &job->lay->f;
    const uint8_t* gate = job->blob;
    const uint8_t* up = job->blob + fp->up_off;
    const uint8_t* down = job->blob + fp->down_off;
    const int up_ty = fp->mixed() ? fp->u_type : fp->gu_type;
    const size_t up_row = fp->mixed() ? fp->u_row : fp->gu_row;
    const int nF = (int) fp->n_ff, nC = (int) fp->n_embd;
    job->gu.resize((size_t) 2 * F);
    job->h.resize(F);
    job->act.resize(std::max<size_t>(fp->act_bytes, 16));
    job->hac.resize(std::max<size_t>(fp->h_bytes, 16));
    kn::native_quant_act(*fp, job->x, job->act.data());
    const int n = std::max(1, pool.size());
    job->left = 2 * n;   // gate's chunks and up's
    auto dchi = [&pool, job, fp, C, nC, nF, n, down]() {
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) C * c / n), r1 = (int) ((long) C * (c + 1) / n);
            pool.post([job, fp, r0, r1, nF, down]() {
                strata::kernels::cpu::native_dot_rows(fp->d_type, fp->d_row, down, job->hac.data(), nF, r0, r1,
                                                      job->y + r0);
            });
        }
    };
    for (int part = 0; part < 2; ++part)
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) F * c / n), r1 = (int) ((long) F * (c + 1) / n);
            pool.post([job, fp, part, r0, r1, nF, nC, up_ty, up_row, gate, up, gcap, ucap, dchi]() {
                strata::kernels::cpu::native_dot_rows(part ? up_ty : fp->gu_type, part ? up_row : fp->gu_row,
                                                      part ? up : gate, job->act.data(), nC, r0, r1,
                                                      job->gu.data() + (size_t) part * nF + r0);
                if (--job->left == 0) {   // (the last chunk: the activation, then down's rows)
                    for (int i = 0; i < nF; ++i) {
                        const float g = job->gu[i] * job->sg, u = job->gu[(size_t) nF + i] * job->su;
                        job->h[i] = gcap * std::tanh(g / gcap) * (1.0f / (1.0f + std::exp(-g))) * ucap *
                                    std::tanh(u / ucap);
                    }
                    strata::kernels::cpu::native_quant_h(*fp, job->h.data(), job->hac.data());
                    dchi();
                }
            });
        }
}

bool selftest(const std::vector<int>& types, std::string& out) { return strata::kernels::cpu::native_selftest(types, out); }

const char* type_name(int type) { return strata::kernels::cpu::native_type_name(type); }

bool make_test_row(int type, std::vector<uint8_t>& row, std::vector<float>& ref, size_t* row_bytes) {
    return strata::kernels::cpu::native_make_test_row(type, row, ref, row_bytes);
}

static double now_s() {
    using clk = std::chrono::steady_clock;
    static const clk::time_point t0 = clk::now();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

void bench(Pool& pool, const std::vector<int>& types, int rows, int reps, const char* which, std::string& out) {
    char line[256];
    for (int ty : types) {
        std::vector<uint8_t> one;
        std::vector<float> ref;
        size_t rb = 0;
        if (!make_test_row(ty, one, ref, &rb) || rb == 0) {
            std::snprintf(line, sizeof line, "  bench %-8s n/a\n", strata::kernels::cpu::native_type_name(ty));
            out += line;
            continue;
        }
        const int n = 256;
        std::vector<uint8_t> w((size_t) rb * rows);
        for (int r = 0; r < rows; ++r) std::memcpy(w.data() + (size_t) r * rb, one.data(), rb);
        std::vector<uint8_t> act(strata::kernels::cpu::native_act_bytes_for(ty), 0);
        std::vector<float> xf(256), af(256);
        std::memcpy(xf.data(), ref.data(), 256 * 4);
        for (int i = 0; i < 256; ++i) af[i] = std::sin(0.11f * (float) i);
        strata::kernels::cpu::native_quant_act_test(ty, af.data(), act.data());
        std::vector<float> out_v(rows);
        const int nw = std::max(1, pool.size());
        const bool hand = std::strcmp(which, "ggml") != 0;
        const double t0 = now_s();
        for (int rep = 0; rep < reps; ++rep) {
            for (int c = 0; c < nw; ++c) {
                const int r0 = rows * c / nw, r1 = rows * (c + 1) / nw;
                pool.post([&w, &act, &out_v, rb, r0, r1, ty, hand]() {
                    strata::kernels::cpu::native_dot_rows_ex(ty, rb, w.data(), act.data(), 256, r0, r1, out_v.data(),
                                                             hand);
                });
            }
            pool.wait();
        }
        const double dt = now_s() - t0;
        const double gb = (double) rows * rb * reps / dt / 1e9;
        std::snprintf(line, sizeof line, "  bench %-8s %6.2f GB/s  (%s, %d rows x %zu B, %d threads)\n",
                      strata::kernels::cpu::native_type_name(ty), gb, hand ? "hand" : "ggml", rows, rb, nw);
        out += line;
    }
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
