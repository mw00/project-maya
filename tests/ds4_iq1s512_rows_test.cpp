// tests/ds4_iq1s512_rows_test.cpp - the CPU lane's AVX-512 IQ1_S rows (src/ds4/ds4_iq512_kern.inc: rows_iq1s, vpdpbusd
// where the CPU has VNNI) against ggml-cpu's own dot, bit for bit, through the lane's calls as the engine makes them
// (iq_cpu_rows, iq_cpu_gate_up): random blocks at the experts' widths and others, 1 to 17 tokens a call (the rows take
// 8 a pass), whole matrices and row ranges that start mid-way (the rows outside untouched), block scales at their edges
// (zero, -0, subnormal, the largest F16), the largest integer sums (grid entries of all +-1 at scale 15 against
// activations of all -128 / 127) and activations scaled down until the sums are subnormal (the float steps' rounding).
// One thread, no GPU.
//
//   DS4_IQ1S512=1 ds4_iq1s512_rows_test   the rows forced on: on a CPU with AVX-512 F / BW / VL / DQ and BMI2 the lane
//                                         must take them (else FAIL)
//   ds4_iq1s512_rows_test                 this CPU's default choice (whichever rows serve, ggml-cpu's bits)
//   ds4_iq1s512_rows_test --bench [--threads N] [--runs R] [--tokens T]
//                                         GB/s of weights of the rows on the path this run takes (1, 2, 4 and 8 tokens,
//                                         64-row calls over ~96 MB of weights, N threads taking the calls in turn): the
//                                         median and best of R passes (--tokens: only T tokens a call); run it with
//                                         DS4_IQ1S512=1 and without to compare
//   ds4_iq1s512_rows_test --bench-ggml [--threads N] [--runs R]
//                                         the same for ggml-cpu's dot, a row and a token a call
#include "../src/ds4/ds4_gguf_cpu.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace gg = strata::ds4::gguf;

namespace {

constexpr int kT = GGML_TYPE_IQ1_S;
constexpr size_t kBB = 50;   // (block_iq1_s: F16 d, 32 qs, 8 qh words)

// rows of random blocks: every other block's scale one of the edge values, the rest small normal values
std::vector<uint8_t> rand_rows(int n, int rows, std::mt19937& rng, bool edges = true) {
    static const uint16_t edge[] = {0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x2400, 0xa400, 0x7bff};
    const size_t rb = ggml_row_size((ggml_type) kT, n);
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (auto& b : w) b = (uint8_t) rng();
    std::uniform_real_distribution<float> sc(0.0005f, 0.02f);
    for (size_t o = 0, k = 0; o < w.size(); o += kBB, ++k) {
        const uint16_t d = edges && k % 2 == 0 ? edge[(k / 2) % 8] : ggml_fp32_to_fp16(sc(rng));
        std::memcpy(&w[o], &d, 2);
    }
    return w;
}

// nt tokens of n values as Q8_K (ggml's quantizer), some with outliers; shift: their scales times 2^shift after
std::vector<uint8_t> rand_acts(int n, int nt, std::mt19937& rng, int shift = 0) {
    const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<uint8_t> a(ab * (size_t) nt);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) n);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng) * (rng() % 64 == 0 ? 20.0f : 1.0f) * (t % 3 == 2 ? 30.0f : 1.0f);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), a.data() + ab * (size_t) t, n);
    }
    if (shift != 0)
        for (size_t o = 0; o < a.size(); o += ggml_type_size(GGML_TYPE_Q8_K)) {   // (block_q8_K: the F32 d first)
            float d;
            std::memcpy(&d, &a[o], 4);
            d = std::ldexp(d, shift);
            std::memcpy(&a[o], &d, 4);
        }
    return a;
}

// the grid indices whose 8 values are all +1 / all -1 / all nonzero, from ggml's own dequantizer (a value of a block of
// scale 1, scale bits 0, no delta sign: grid + 0.125, so 1.125 for +1, -0.875 for -1, 0.125 for 0)
struct Extremes {
    std::vector<int> pos, neg, nz;
};
Extremes find_extremes() {
    Extremes e;
    const auto* tr = ggml_get_type_traits((ggml_type) kT);
    std::vector<uint8_t> b(kBB, 0);
    const uint16_t one = ggml_fp32_to_fp16(1.0f);
    float v[256];
    for (int g = 0; g < 2048; ++g) {
        std::memcpy(&b[0], &one, 2);
        for (int k = 0; k < 32; ++k) b[2 + k] = (uint8_t) (g & 255);
        for (int c = 0; c < 8; ++c) {
            const uint16_t h = (uint16_t) (((g >> 8) & 7) * 0x249);   // (the high 3 bits for all 4 groups of chunk c)
            std::memcpy(&b[34 + 2 * c], &h, 2);
        }
        tr->to_float(b.data(), v, 256);
        bool p = true, n = true, z = true;
        for (int k = 0; k < 8; ++k) {
            p = p && v[k] > 0.5f;
            n = n && v[k] < -0.5f;
            z = z && std::fabs(v[k]) > 0.5f;
        }
        if (p) e.pos.push_back(g);
        if (n) e.neg.push_back(g);
        if (z) e.nz.push_back(g);
    }
    return e;
}

// rows whose every group is one of `idx` (scale bits 7, random delta signs), scale `d`
std::vector<uint8_t> extreme_rows(int n, int rows, const std::vector<int>& idx, uint16_t d, std::mt19937& rng) {
    const size_t rb = ggml_row_size((ggml_type) kT, n);
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (size_t o = 0; o < w.size(); o += kBB) {
        std::memcpy(&w[o], &d, 2);
        for (int c = 0; c < 8; ++c) {
            uint16_t h = (uint16_t) (7u << 12 | (rng() & 1u) << 15);
            for (int k = 0; k < 4; ++k) {
                const int g = idx[rng() % idx.size()];
                w[o + 2 + 4 * c + k] = (uint8_t) (g & 255);
                h = (uint16_t) (h | ((g >> 8) & 7) << (3 * k));
            }
            std::memcpy(&w[o + 34 + 2 * c], &h, 2);
        }
    }
    return w;
}

// Q8_K activations of all `q` (or random bytes when q == 0, -128 included), their bsums the sums, scale 0.01
std::vector<uint8_t> const_acts(int n, int nt, int q, std::mt19937& rng) {
    const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n), bq = ggml_type_size(GGML_TYPE_Q8_K);
    std::vector<uint8_t> a(ab * (size_t) nt);
    for (size_t o = 0; o < a.size(); o += bq) {
        const float d = 0.01f;
        std::memcpy(&a[o], &d, 4);
        int8_t qs[256];
        for (int k = 0; k < 256; ++k) qs[k] = (int8_t) (q != 0 ? q : (int) (rng() & 255u) - 128);
        std::memcpy(&a[o + 4], qs, 256);
        for (int g = 0; g < 16; ++g) {
            int s = 0;
            for (int k = 0; k < 16; ++k) s += qs[16 * g + k];
            const int16_t s16 = (int16_t) s;
            std::memcpy(&a[o + 4 + 256 + 2 * g], &s16, 2);
        }
    }
    return a;
}

float dot(int n, const uint8_t* row, const void* act) {
    float s = 0.0f;
    ggml_get_type_traits_cpu((ggml_type) kT)->vec_dot(n, &s, 0, row, 0, act, 0, 1);
    return s;
}

// iq_cpu_rows for rows [r0, r1) of `rows` rows, nt tokens: the results that differ from ggml-cpu's dot in any bit
int check_rows(const std::vector<uint8_t>& w, const std::vector<uint8_t>& a, int n, int rows, int nt, int r0, int r1,
               const char* what) {
    const size_t rb = ggml_row_size((ggml_type) kT, n), ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<const void*> ap((size_t) nt);
    std::vector<float> got((size_t) nt * rows, -7.0f);
    std::vector<float*> op((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        ap[(size_t) t] = a.data() + ab * (size_t) t;
        op[(size_t) t] = got.data() + (size_t) t * rows;
    }
    gg::iq_cpu_rows(kT, w.data(), rb, n, ap.data(), nt, op.data(), r0, r1);
    int bad = 0;
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < rows; ++r) {
            const float want = r >= r0 && r < r1 ? dot(n, w.data() + (size_t) r * rb, ap[(size_t) t]) : -7.0f;
            const float g = got[(size_t) t * rows + r];
            if (std::memcmp(&g, &want, 4) != 0 && bad++ < 3)
                std::printf("  %s n=%d nt=%d token %d row %d: %.9g, ggml %.9g\n", what, n, nt, t, r, g, want);
        }
    return bad;
}

// iq_cpu_gate_up (the swiglu epilogue on the rows' values, clamped or not) against ggml's dots
int check_gate_up(int E, int FF, int nt, int r0, int r1, float limit, std::mt19937& rng) {
    gg::IqCpu f;
    std::string err;
    if (!gg::iq_cpu_fmt(kT, kT, E, FF, f, err)) {
        std::printf("FAIL: iq_cpu_fmt %d x %d: %s\n", E, FF, err.c_str());
        return 1;
    }
    const auto gate = rand_rows(E, FF, rng, false), up = rand_rows(E, FF, rng, false);
    const auto a = rand_acts(E, nt, rng);
    std::vector<const void*> ap((size_t) nt);
    std::vector<float> ff((size_t) nt * FF, -7.0f);
    std::vector<float*> fp((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        ap[(size_t) t] = a.data() + f.act_bytes * (size_t) t;
        fp[(size_t) t] = ff.data() + (size_t) t * FF;
    }
    gg::iq_cpu_gate_up(f, gate.data(), up.data(), ap.data(), nt, fp.data(), r0, r1, limit);
    int bad = 0;
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < FF; ++r) {
            float want = -7.0f;
            if (r >= r0 && r < r1) {
                float g = dot(E, gate.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                float u = dot(E, up.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                if (limit > 0.0f) {
                    g = std::fmin(g, limit);
                    u = std::fmin(std::fmax(u, -limit), limit);
                }
                want = (g / (1.f + std::exp(-g))) * u;
            }
            const float got = ff[(size_t) t * FF + r];
            if (std::memcmp(&got, &want, 4) != 0 && bad++ < 3)
                std::printf("  gate/up nt=%d token %d row %d: %.9g, ggml %.9g\n", nt, t, r, got, want);
        }
    return bad;
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// --bench: the rows on this run's path (or ggml-cpu's dot), nth threads taking 64-row calls (the lane's jobs) in turn
// over ~96 MB of rows (more than the caches), nt tokens; the median and best of `runs` passes, GB/s of weights
void bench(int nt, int nth, int runs, bool ggml) {
    const int n = 4096;
    const size_t rb = ggml_row_size((ggml_type) kT, n);
    const int rows = (int) (96e6 / (double) rb) / 64 * 64;
    std::mt19937 rng(7u);
    const auto w = rand_rows(n, rows, rng, false);
    const auto a = rand_acts(n, nt, rng);
    const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<const void*> ap((size_t) nt);
    std::vector<float> out((size_t) nt * rows);
    std::vector<float*> op((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        ap[(size_t) t] = a.data() + ab * (size_t) t;
        op[(size_t) t] = out.data() + (size_t) t * rows;
    }
    std::atomic<int> next{0};
    auto work = [&] {
        for (int r0; (r0 = next.fetch_add(64)) < rows;) {
            if (ggml) {
                for (int r = r0; r < r0 + 64; ++r)
                    for (int t = 0; t < nt; ++t) op[(size_t) t][r] = dot(n, w.data() + (size_t) r * rb, ap[(size_t) t]);
            } else {
                gg::iq_cpu_rows(kT, w.data(), rb, n, ap.data(), nt, op.data(), r0, r0 + 64);
            }
        }
    };
    if (!ggml) gg::iq_cpu_rows(kT, w.data(), rb, n, ap.data(), nt, op.data(), 0, 64);   // (the first-use check, untimed)
    std::barrier sync(nth);
    std::vector<double> us;
    std::vector<std::thread> th;
    std::atomic<bool> stop{false};
    for (int k = 1; k < nth; ++k)
        th.emplace_back([&] {
            for (;;) {
                sync.arrive_and_wait();   // (a pass starts)
                if (stop.load()) return;
                work();
                sync.arrive_and_wait();   // (a pass ends)
            }
        });
    for (int p = 0; p <= runs; ++p) {   // (pass 0: warm-up)
        next.store(0);
        const double t0 = now_us();
        sync.arrive_and_wait();
        work();
        sync.arrive_and_wait();
        if (p > 0) us.push_back(now_us() - t0);
    }
    stop.store(true);
    sync.arrive_and_wait();
    for (auto& t : th) t.join();
    std::sort(us.begin(), us.end());
    const double med = us[us.size() / 2], best = us[0];
    const double bytes = (double) rows * (double) rb;
    std::printf("%s IQ1_S %d token%s %d thread%s: median %7.2f GB/s (best %7.2f) of weights, %6.1f ns a row and token "
                "(%d rows of %d)\n",
                ggml ? "ggml-dot" : gg::iq_cpu_x512_rows(kT) ? "avx512  " : "before  ", nt, nt > 1 ? "s" : " ", nth,
                nth > 1 ? "s" : " ", bytes / med / 1e3, bytes / best / 1e3, med * 1e3 / rows / nt, rows, n);
}

bool cpu_x512() {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("bmi2");
#else
    return false;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    const char* e = std::getenv("DS4_IQ1S512");
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--bench" || mode == "--bench-ggml") {
        int nth = 1, runs = 9, only = 0;
        for (int k = 2; k + 1 < argc; k += 2) {
            if (std::string(argv[k]) == "--threads") nth = std::max(1, std::atoi(argv[k + 1]));
            if (std::string(argv[k]) == "--runs") runs = std::max(1, std::atoi(argv[k + 1]));
            if (std::string(argv[k]) == "--tokens") only = std::atoi(argv[k + 1]);
        }
        std::printf("ds4_iq1s512_rows_test %s: DS4_IQ1S512=%s DS4_IQ_OWN=%s, %d thread%s, %d passes\n", mode.c_str(),
                    e != nullptr ? e : "(unset)", std::getenv("DS4_IQ_OWN") != nullptr ? std::getenv("DS4_IQ_OWN") : "(unset)",
                    nth, nth > 1 ? "s" : "", runs);
        for (int nt : {1, 2, 4, 8})
            if (only <= 0 || nt == only) bench(nt, nth, runs, mode == "--bench-ggml");
        return 0;
    }
    const bool forced = e != nullptr && std::atoi(e) != 0;
    const bool x512 = gg::iq_cpu_x512_rows(kT);
    std::printf("ds4_iq1s512_rows_test: DS4_IQ1S512=%s, AVX-512 F/BW/VL/DQ + BMI2 %s; IQ1_S rows: %s\n",
                e != nullptr ? e : "(unset)", cpu_x512() ? "yes" : "no",
                x512 ? "AVX-512" : gg::iq_cpu_own_rows(kT) ? "own (AVX2)" : "ggml-cpu's / the GLM lane's");
    if (forced && cpu_x512() && !x512) {
        std::printf("FAIL: DS4_IQ1S512=1 on an AVX-512 CPU but the lane does not take the AVX-512 IQ1_S rows\n");
        return 1;
    }
    std::mt19937 rng(20261103u);
    int bad = 0, cases = 0;
    for (int n : {256, 2048, 2304, 4096, 7168})
        for (int nt : {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 17}) {
            const auto w1 = rand_rows(n, 17, rng), w2 = rand_rows(n, 23, rng);
            bad += check_rows(w1, rand_acts(n, nt, rng), n, 17, nt, 0, 17, "random");   // whole matrices
            bad += check_rows(w2, rand_acts(n, nt, rng), n, 23, nt, 5, 19, "range");    // a range mid-way
            cases += 2;
        }
    for (int nt : {1, 2, 5, 8})   // (sums near and in the subnormal range: the float steps' rounding there)
        for (int shift : {-96, -112, -120, -128, -136}) {
            bad += check_rows(rand_rows(2304, 17, rng), rand_acts(2304, nt, rng, shift), 2304, 17, nt, 0, 17, "subnormal");
            ++cases;
        }
    const Extremes ex = find_extremes();
    std::printf("  grid entries of all +1: %zu, all -1: %zu, all nonzero: %zu\n", ex.pos.size(), ex.neg.size(), ex.nz.size());
    if (ex.pos.empty() || ex.neg.empty() || ex.nz.empty()) {
        std::printf("FAIL: no extreme grid entries found (ggml's IQ1_S dequantizer changed?)\n");
        return 1;
    }
    const uint16_t dmax = 0x7bff, dmid = ggml_fp32_to_fp16(0.01f);
    for (const auto* idx : {&ex.pos, &ex.neg, &ex.nz})
        for (int q : {-128, 127, -127, 0})   // (0: random bytes, -128 included)
            for (uint16_t d : {dmid, dmax})
                for (int nt : {1, 3, 8, 9}) {
                    bad += check_rows(extreme_rows(4096, 9, *idx, d, rng), const_acts(4096, nt, q, rng), 4096, 9, nt, 0, 9,
                                      "extreme");
                    ++cases;
                }
    for (int nt : {1, 3, 4, 6, 8}) {
        bad += check_gate_up(4096, 512, nt, 0, 512, nt % 2 ? 10.0f : 0.0f, rng);
        bad += check_gate_up(4096, 512, nt, 37, 37 + 64, 0.0f, rng);
        cases += 2;
    }
    std::printf("ds4_iq1s512_rows_test: %d cases, %d results differ from ggml-cpu's dot - %s\n", cases, bad,
                bad == 0 ? "OK" : "FAILED");
    return bad == 0 ? 0 : 1;
}
