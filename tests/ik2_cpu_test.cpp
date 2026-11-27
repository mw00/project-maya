// tests/ik2_cpu_test.cpp - unit test and microbenchmark for maya-ik2's cache-path CPU lane (src/ik2/ik2_cpu.*).
#include "ik2_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_ok = 0, g_fail = 0;

void check(bool cond, const char* what) {
    if (cond) {
        ++g_ok;
    } else {
        ++g_fail;
        std::printf("FAIL %s\n", what);
    }
}

float ue4m3_ref(int x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    return (e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7)) * 0.5f;
}

void nvfp4_ref(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        double acc = 0.0;
        for (int i = 0; i < nb; ++i, b += 36)
            for (int sb = 0; sb < 4; ++sb) {
                double a = 0.0;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t by = b[4 + sb * 8 + j];
                    a += kv[by & 0xF] * (double) x[i * 64 + sb * 16 + j] +
                         kv[by >> 4] * (double) x[i * 64 + sb * 16 + j + 8];
                }
                acc += (double) ue4m3_ref(b[sb]) * a;
            }
        out[r - r0] = (float) acc;
    }
}

void expert_ref(const uint8_t* gate, const uint8_t* up, const uint8_t* down, const float* x, int C, int F,
                float sg, float su, float gcap, float ucap, float* y) {
    std::vector<float> g(F), u(F), h(F);
    nvfp4_ref(gate, C, 0, F, x, g.data());
    nvfp4_ref(up, C, 0, F, x, u.data());
    for (int i = 0; i < F; ++i) {
        const float gv = g[i] * sg, uv = u[i] * su;
        h[i] = gcap * std::tanh(gv / gcap) * (1.0f / (1.0f + std::exp(-gv))) * ucap * std::tanh(uv / ucap);
    }
    nvfp4_ref(down, F, 0, C, h.data(), y);
}

std::vector<uint8_t> random_nvfp4(int rows, int K, std::mt19937& rng) {
    const int nb = K / 64;
    std::vector<uint8_t> w((size_t) rows * nb * 36);
    std::uniform_int_distribution<int> byte(0, 255), sc(0x20, 0x48);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < nb; ++i) {
            uint8_t* b = &w[((size_t) r * nb + i) * 36];
            for (int sb = 0; sb < 4; ++sb) {
                const int pick = byte(rng);
                b[sb] = (uint8_t) (pick % 31 == 0 ? 0 : pick % 53 == 0 ? 0x7F : sc(rng));
            }
            for (int j = 4; j < 36; ++j) b[j] = (uint8_t) byte(rng);
        }
    }
    return w;
}

double rel_l2(const float* got, const float* ref, int n) {
    double n0 = 0.0, d = 0.0;
    for (int i = 0; i < n; ++i) {
        n0 += (double) ref[i] * ref[i];
        const double diff = (double) got[i] - ref[i];
        d += diff * diff;
    }
    return n0 > 0.0 ? std::sqrt(d / n0) : std::sqrt(d);
}

void test_rows() {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.5f);
    for (const char* form : {"scalar", "avx2", "avx512"}) {
        if (!ik2::cpu::rows_force(form)) {
            std::printf("  (%s: not on this CPU)\n", form);
            continue;
        }
        double worst = 0.0;
        for (int K : {64, 256, 2304, 4608}) {
            std::vector<float> x((size_t) K);
            for (int i = 0; i < K; ++i) x[(size_t) i] = (i % 29 == 0) ? 0.0f : nd(rng);
            for (int rows : {1, 2, 3, 7, 24, 25, 48, 49}) {
                const auto w = random_nvfp4(rows + 2, K, rng);
                std::vector<float> got(rows), ref(rows);
                const int r0 = 1, r1 = 1 + rows;
                ik2::cpu::nvfp4_rows(w.data(), K, r0, r1, x.data(), got.data());
                nvfp4_ref(w.data(), K, r0, r1, x.data(), ref.data());
                worst = std::max(worst, rel_l2(got.data(), ref.data(), rows));
            }
        }
        char msg[128];
        std::snprintf(msg, sizeof msg, "nvfp4_rows %s matches reference (worst rel L2 = %.2e)", form, worst);
        check(worst < 1e-5, msg);
    }
    ik2::cpu::rows_force("avx512") || ik2::cpu::rows_force("avx2") || ik2::cpu::rows_force("scalar");
}

void test_lane() {
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    struct Shape {
        int C, F;
    };
    for (const Shape sh : {Shape{128, 64}, Shape{192, 128}, Shape{2304, 1152}, Shape{4608, 2304}}) {
        const int n_exp = 8;
        struct Exp {
            std::vector<uint8_t> gate, up, down;
        };
        std::vector<Exp> ex(n_exp);
        for (int e = 0; e < n_exp; ++e) {
            ex[e].gate = random_nvfp4(sh.F, sh.C, rng);
            ex[e].up = random_nvfp4(sh.F, sh.C, rng);
            ex[e].down = random_nvfp4(sh.C, sh.F, rng);
        }
        std::vector<float> xs((size_t) n_exp * sh.C);
        for (float& v : xs) v = nd(rng);
        const float gcap = 4.0f, ucap = 25.0f;
        std::vector<float> sg(n_exp), su(n_exp), y_ref((size_t) n_exp * sh.C);
        for (int i = 0; i < n_exp; ++i) {
            sg[i] = 0.7f + 0.08f * i;
            su[i] = 1.4f - 0.06f * i;
            expert_ref(ex[i].gate.data(), ex[i].up.data(), ex[i].down.data(), xs.data() + (size_t) (i % 3) * sh.C,
                       sh.C, sh.F, sg[i], su[i], gcap, ucap, y_ref.data() + (size_t) i * sh.C);
        }

        for (int workers : {0, 1, 3, 7}) {
            ik2::cpu::Lane lane(workers);
            for (int n : {1, 2, 3, 8}) {
                std::vector<ik2::cpu::ExpertJob> jobs(n);
                std::vector<float> y((size_t) n * sh.C, 0.0f), y2((size_t) n * sh.C, 0.0f);
                for (int i = 0; i < n; ++i) {
                    jobs[i] = {ex[i].gate.data(), ex[i].up.data(), ex[i].down.data(),
                               xs.data() + (size_t) (i % 3) * sh.C, sg[i], su[i], y.data() + (size_t) i * sh.C};
                }
                lane.run(jobs.data(), n, sh.C, sh.F, gcap, ucap);
                for (int i = 0; i < n; ++i) jobs[i].y = y2.data() + (size_t) i * sh.C;
                lane.run(jobs.data(), n, sh.C, sh.F, gcap, ucap);

                double worst = 0.0;
                for (int i = 0; i < n; ++i)
                    worst = std::max(worst, rel_l2(y.data() + (size_t) i * sh.C, y_ref.data() + (size_t) i * sh.C, sh.C));
                const bool det = std::memcmp(y.data(), y2.data(), y.size() * sizeof(float)) == 0;
                char msg[160];
                std::snprintf(msg, sizeof msg, "Lane(w=%d) n=%d (%dx%d) rel=%.2e det=%d", workers, n, sh.C, sh.F,
                              worst, (int) det);
                check(worst < 1e-5 && det, msg);
            }
        }
    }
}

int run_bench(int argc, char** argv) {
    std::vector<int> tc;
    for (int i = 2; i < argc; ++i) tc.push_back(std::atoi(argv[i]));
    if (tc.empty()) tc = {8, 12, 16};

    const int C = 4608, F = 2304, n_exp = 96;   // 96 * 17.9 MB = 1.72 GB > LLC
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    struct Exp {
        std::vector<uint8_t> gate, up, down;
    };
    std::vector<Exp> ex(n_exp);
    for (int e = 0; e < n_exp; ++e) {
        ex[e].gate = random_nvfp4(F, C, rng);
        ex[e].up = random_nvfp4(F, C, rng);
        ex[e].down = random_nvfp4(C, F, rng);
    }
    const double mb = (double) (ex[0].gate.size() + ex[0].up.size() + ex[0].down.size()) / 1e6;
    std::vector<float> x(C), y((size_t) 4 * C);
    for (float& v : x) v = nd(rng);

    std::printf("ik2_cpu_test bench: form=%s, %d experts x %.2f MB (%.2f GB)\n", ik2::cpu::rows_form(), n_exp, mb,
                n_exp * mb / 1e3);
    for (int th : tc) {
        ik2::cpu::Lane lane(std::max(0, th - 1));
        for (int batch : {1, 2, 3}) {
            const int calls = n_exp / batch;
            double best = 1e30, sum = 0.0;
            const int reps = 3;
            for (int r = -1; r < reps; ++r) {   // r == -1: warm-up
                const auto t0 = std::chrono::steady_clock::now();
                for (int c = 0; c < calls; ++c) {
                    ik2::cpu::ExpertJob jobs[4];
                    for (int i = 0; i < batch; ++i) {
                        const int e = c * batch + i;
                        jobs[i] = {ex[e].gate.data(), ex[e].up.data(), ex[e].down.data(), x.data(),
                                   1.0f, 1.0f, y.data() + (size_t) i * C};
                    }
                    lane.run(jobs, batch, C, F, 4.0f, 25.0f);
                }
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (r >= 0) {
                    best = std::min(best, s);
                    sum += s;
                }
            }
            const double exp_best_ms = best / (calls * batch) * 1e3;
            const double exp_mean_ms = (sum / reps) / (calls * batch) * 1e3;
            std::printf("  threads=%2d (lane=%2d) batch=%d: best %.3f ms/exp (%.1f GB/s), mean %.3f ms/exp\n",
                        th, lane.threads(), batch, exp_best_ms, mb / exp_best_ms, exp_mean_ms);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "bench")) return run_bench(argc, argv);
    std::printf("ik2_cpu_test: available=%d form=%s rec(16)=%d\n", (int) ik2::cpu::available(),
                ik2::cpu::rows_form(), ik2::cpu::recommended_workers(16));
    check(ik2::cpu::recommended_workers(1) == 0, "recommended_workers(1) == 0");
    check(ik2::cpu::recommended_workers(1024) >= 0, "recommended_workers(1024) >= 0");
    test_rows();
    test_lane();
    std::printf("ik2_cpu_test: %d ok, %d FAIL\n", g_ok, g_fail);
    return g_fail ? 1 : 0;
}
