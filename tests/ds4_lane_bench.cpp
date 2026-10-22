// tests/ds4_lane_bench.cpp - the CPU lane's time for one routed expert of DeepSeek-V4.1-Flash, by format, as maya-ds4's
// PCIe lane times it at load (ds4_model.cu's lane_time): calls of 1 and 3 experts of one token - each call's gate / up
// rows in 64-row jobs on a pool of the physical cores (the caller helping), then its down rows, as cpu_experts_mx /
// cpu_experts_iq run them - the experts in turn from more than the caches hold; an expert's time c and a call's oc
// (the faster of 5 rounds, each the median of 4 calls).  Then the stream lane's read into pinned memory (a memcpy of an
// expert, 1 and 2 threads), and for a GPU on a link of --link GB/s the split the lane would plan (ds4_pcie.hpp: of n
// picks in RAM, how many it streams) - for a PC whose GPU cannot run the engine, what its CPU side would give.
// --check instead: each low-bit format's expert on the CPU lane against a double reference (no GPU needed).
//
//   ds4_lane_bench [--threads N] [--formats mxfp4,iq2_xxs,iq2_s] (also iq3_xxs, iq3_s, iq1_s, iq1_m) [--experts 24] [--link 12] [--gpu-ms 0.03]
#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "../src/ds4/ds4_pcie.hpp"
#include "../src/ds4/ds4_pool.hpp"

#include "ggml.h"   // (--check: ggml's dequantize_row_* for the reference)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace pcie = strata::ds4::pcie;
using strata::ds4::Pool;

namespace {

constexpr int E = 5120, FF = 2304, RB = 64;
constexpr float kLim = 10.0f;

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint16_t f2h(float f) {   // (normal values only: the blocks' scales)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    const uint32_t m = (u >> 13) & 0x3ffu;
    return (uint16_t) (s | (uint32_t) std::max(1, std::min(30, e)) << 10 | m);
}

struct Fmt {
    std::string name;
    int tg = 0, td = 0;   // ggml types: gate / up, down (0: MXFP4)
    size_t gr = 0, dr = 0, blob = 0;   // a gate row's bytes, a down row's, an expert's
    gg::IqCpu iq;
};

// experts of a format in anonymous memory, random blocks with weights' scales
struct Experts {
    Fmt f;
    int n = 0;
    std::vector<uint8_t> mem;
    const uint8_t* mat(int e, int r) const {
        const size_t g = (size_t) FF * f.gr;
        return mem.data() + (size_t) e * f.blob + (r == 0 ? 0 : r == 1 ? g : 2 * g);
    }
};

void fill(Experts& X, uint64_t seed) {
    std::mt19937_64 rng(seed);
    X.mem.resize((size_t) X.n * X.f.blob);
    for (auto& v : X.mem) v = (uint8_t) (rng() & 0xff);
    for (int e = 0; e < X.n; ++e)
        for (int r = 0; r < 3; ++r) {
            uint8_t* m = const_cast<uint8_t*>(X.mat(e, r));
            const int rows = r < 2 ? FF : E, k = r < 2 ? E : FF;
            const size_t rb = r < 2 ? X.f.gr : X.f.dr;
            const int t = r < 2 ? X.f.tg : X.f.td;
            if (t == 0) {   // MXFP4: an E8M0 scale a 17-byte block
                for (size_t b = 0; b + 17 <= (size_t) rows * rb; b += 17) m[b] = (uint8_t) (114 + rng() % 9);
            } else {        // the i-quants: an F16 d first in each 256-value block (IQ1_M: in its 4 scale words' top nibbles)
                const size_t bb = rb / (size_t) (k / 256);
                for (size_t b = 0; b + bb <= (size_t) rows * rb; b += bb) {
                    const uint16_t h = f2h(0.001f + 0.002f * (float) (rng() % 1000) / 1000.0f);
                    if (t == 29)
                        for (int i = 0; i < 4; ++i) {
                            uint16_t sw;
                            std::memcpy(&sw, m + b + 48 + 2 * i, 2);
                            sw = (uint16_t) ((sw & 0x0FFFu) | (((h >> (4 * i)) & 0xFu) << 12));
                            std::memcpy(m + b + 48 + 2 * i, &sw, 2);
                        }
                    else std::memcpy(m + b, &h, 2);
                }
            }
        }
}

// one call of the CPU lane: n experts (e0, e0 + 1, ..) of one token x - cpu_experts_mx / cpu_experts_iq's jobs
struct Lane {
    Pool* pool = nullptr;
    std::vector<uint8_t> xq, hq;
    std::vector<float> h, y;
    void call(const Experts& X, int e0, int n, const float* x) {
        const int GB = (FF + RB - 1) / RB, DB = (E + RB - 1) / RB;
        const Fmt& f = X.f;
        const bool mx = f.tg == 0;
        h.resize((size_t) 8 * FF);
        y.resize((size_t) 8 * E);
        if (mx) {
            const size_t xa = gg::mx_act_bytes(E), ha = gg::mx_act_bytes(FF);
            xq.resize(xa);
            hq.resize((size_t) 8 * ha);
            gg::mx_quant(x, E, xq.data());
            std::function<void(int)> gu = [&](int j) {
                const int e = (e0 + j / GB) % X.n, s = j / GB, r0 = (j % GB) * RB, r1 = std::min(FF, r0 + RB);
                const void* ap[1] = {xq.data()};
                float yg[RB], yu[RB];
                float* gp[1] = {yg};
                float* up[1] = {yu};
                gg::mx_rows(X.mat(e, 0), f.gr, E, r0, r1, ap, 1, gp);
                gg::mx_rows(X.mat(e, 1), f.gr, E, r0, r1, ap, 1, up);
                for (int r = r0; r < r1; ++r) {
                    const float gv = std::min(yg[r - r0], kLim), uv = std::min(std::max(yu[r - r0], -kLim), kLim);
                    yg[r - r0] = gv / (1.0f + std::exp(-gv)) * uv;
                }
                gg::mx_quant_blocks(yg, r0 / 32, r1 / 32, hq.data() + (size_t) s * ha, FF);
            };
            pool->run(n * GB, gu);
            std::function<void(int)> dn = [&](int j) {
                const int e = (e0 + j / DB) % X.n, s = j / DB, r0 = (j % DB) * RB, r1 = std::min(E, r0 + RB);
                const void* ap[1] = {hq.data() + (size_t) s * ha};
                float* yp[1] = {y.data() + (size_t) s * E + r0};
                gg::mx_rows(X.mat(e, 2), f.dr, FF, r0, r1, ap, 1, yp);
            };
            pool->run(n * DB, dn);
            return;
        }
        xq.resize((size_t) E * 4);
        hq.resize((size_t) 8 * E * 4);
        gg::iq_cpu_quant(f.iq.gu_act, x, E, xq.data());
        std::function<void(int)> gu = [&](int j) {
            const int e = (e0 + j / GB) % X.n, s = j / GB, r0 = (j % GB) * RB, r1 = std::min(FF, r0 + RB);
            const void* ap[1] = {xq.data()};
            float* hp[1] = {h.data() + (size_t) s * FF};
            gg::iq_cpu_gate_up(f.iq, X.mat(e, 0), X.mat(e, 1), ap, 1, hp, r0, r1, kLim);
        };
        pool->run(n * GB, gu);
        for (int s = 0; s < n; ++s) gg::iq_cpu_quant(f.iq.d_act, h.data() + (size_t) s * FF, FF, hq.data() + (size_t) s * E * 4);
        std::function<void(int)> dn = [&](int j) {
            const int e = (e0 + j / DB) % X.n, s = j / DB, r0 = (j % DB) * RB, r1 = std::min(E, r0 + RB);
            const void* hp[1] = {hq.data() + (size_t) s * E * 4};
            float* yp[1] = {y.data() + (size_t) s * E};
            gg::iq_cpu_down(f.iq, X.mat(e, 2), hp, 1, yp, r0, r1);
        };
        pool->run(n * DB, dn);
    }
};

// --check: a low-bit format's expert on the CPU lane (one token, gate / up then down, the pool's jobs) against a double
// reference on the Q8_K values the lane took (block_q8_K: F32 d, int8 qs[256], int16 bsums[16]) and ggml's own
// dequantize_row_* weights - the largest difference relative to the largest output; a CPU-only test (no GPU)
bool check_fmt(Lane& ln, const Experts& X, const float* x) {
    ln.call(X, 0, 1, x);
    auto q8k = [](const uint8_t* q, int k) {
        std::vector<double> v((size_t) k);
        for (int b = 0; b < k / 256; ++b) {
            float d;
            std::memcpy(&d, q + (size_t) b * 292, 4);
            for (int j = 0; j < 256; ++j) v[(size_t) b * 256 + j] = (double) d * (int8_t) q[(size_t) b * 292 + 4 + j];
        }
        return v;
    };
    auto dot = [](int type, const uint8_t* row, int k, const std::vector<double>& a) {
        static thread_local std::vector<float> w;
        w.resize((size_t) k);
        ggml_get_type_traits((ggml_type) type)->to_float(row, w.data(), k);
        double s = 0;
        for (int i = 0; i < k; ++i) s += (double) w[(size_t) i] * a[(size_t) i];
        return s;
    };
    const std::vector<double> xa = q8k(ln.xq.data(), E), ha = q8k(ln.hq.data(), FF);
    double hm = 1e-30, he = 0, ym = 1e-30, ye = 0;
    std::vector<double> hr((size_t) FF), yr((size_t) E);
    for (int f = 0; f < FF; ++f) {
        double g = dot(X.f.tg, X.mat(0, 0) + (size_t) f * X.f.gr, E, xa), u = dot(X.f.tg, X.mat(0, 1) + (size_t) f * X.f.gr, E, xa);
        g = std::min(g, (double) kLim);
        u = std::min(std::max(u, -(double) kLim), (double) kLim);
        hr[(size_t) f] = g / (1.0 + std::exp(-g)) * u;
        hm = std::max(hm, std::fabs(hr[(size_t) f]));
    }
    for (int d = 0; d < E; ++d) {
        yr[(size_t) d] = dot(X.f.td, X.mat(0, 2) + (size_t) d * X.f.dr, FF, ha);
        ym = std::max(ym, std::fabs(yr[(size_t) d]));
    }
    for (int f = 0; f < FF; ++f) he = std::max(he, std::fabs(ln.h[(size_t) f] - hr[(size_t) f]) / hm);
    for (int d = 0; d < E; ++d) ye = std::max(ye, std::fabs(ln.y[(size_t) d] - yr[(size_t) d]) / ym);
    const bool ok = he <= 1e-4 && ye <= 1e-4;
    std::printf("check %s (%s / %s): CPU lane vs the double reference: gate/up %.2e, down %.2e of the largest  %s\n",
                X.f.name.c_str(), ggml_type_name((ggml_type) X.f.tg), ggml_type_name((ggml_type) X.f.td), he, ye,
                ok ? "ok" : "FAIL");
    return ok;
}

// the logical CPUs, the first of each core first (Linux; elsewhere 0..n-1)
std::vector<int> core_cpus() {
    std::vector<int> first;
#if !defined(_WIN32)
    for (int c = 0; c < 1024; ++c) {
        std::FILE* f = std::fopen(("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list").c_str(), "r");
        if (f == nullptr) break;
        int s0 = -1;
        if (std::fscanf(f, "%d", &s0) != 1) s0 = c;
        std::fclose(f);
        if (s0 == c) first.push_back(c);
    }
#endif
    if (first.empty())
        for (unsigned c = 0; c < std::max(1u, std::thread::hardware_concurrency() / 2); ++c) first.push_back((int) c);
    return first;
}

}  // namespace

int main(int argc, char** argv) {
    int threads = 0, n_exp = 24;
    double link = 12.0, gpu_ms = 0.03;
    std::string formats = "mxfp4,iq2_xxs,iq2_s";
    bool check = false;
    int failed = 0;
    for (int i = 1; i < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--check") {   // (a flag: no value)
            check = true;
            --i;
            continue;
        }
        if (i + 1 >= argc) break;
        const std::string v = argv[i + 1];
        if (a == "--threads") threads = std::atoi(v.c_str());
        else if (a == "--formats") formats = v;
        else if (a == "--experts") n_exp = std::max(4, std::atoi(v.c_str()));
        else if (a == "--link") link = std::atof(v.c_str());
        else if (a == "--gpu-ms") gpu_ms = std::atof(v.c_str());
    }
    const std::vector<int> cores = core_cpus();
    if (threads <= 0) threads = (int) cores.size();
    // (the engine's pool: a worker on each physical core but the caller's, the caller helping)
    std::vector<int> pin(cores.begin() + 1, cores.end());
    Pool pool(threads - 1, pin, 20000);
    std::printf("CPU lane: %d threads (%zu physical cores), MXFP4 rows %s\n", threads, cores.size(),
                gg::mx_tier_name(gg::mx_tier()));
    std::vector<float> x((size_t) E);
    for (int i = 0; i < E; ++i) x[(size_t) i] = 0.02f * std::sin(0.37f * (float) i) + ((i % 97) == 0 ? 0.3f : 0.0f);
    size_t a = 0;
    while (a < formats.size()) {
        size_t b = formats.find(',', a);
        if (b == std::string::npos) b = formats.size();
        const std::string fn = formats.substr(a, b - a);
        a = b + 1;
        Experts X;
        X.n = n_exp;
        X.f.name = fn;
        if (fn == "mxfp4") {   // (17-byte blocks of 32 values)
            X.f.gr = (size_t) E / 32 * 17;
            X.f.dr = (size_t) FF / 32 * 17;
        } else {
            const int tg = fn == "iq2_xxs" ? 16 : fn == "iq2_s" ? 22 : fn == "iq3_xxs" ? 18 : fn == "iq3_s" ? 21
                           : fn == "iq1_s" ? 19 : fn == "iq1_m" ? 29 : -1;
            const int td = fn == "iq2_xxs" ? 18 : fn == "iq2_s" ? 21 : fn == "iq3_xxs" ? 21 : fn == "iq3_s" ? 21
                           : fn == "iq1_s" ? 29 : fn == "iq1_m" ? 29 : -1;   // (IQ1: an IQ1_M down)
            std::string err;
            if (tg < 0 || !gg::iq_cpu_fmt(tg, td, E, FF, X.f.iq, err)) {
                std::printf("%s: not run (%s)\n", fn.c_str(), tg < 0 ? "unknown format" : err.c_str());
                continue;
            }
            X.f.tg = tg;
            X.f.td = td;
            X.f.gr = X.f.iq.gu_row;
            X.f.dr = X.f.iq.d_row;
        }
        X.f.blob = 2 * (size_t) FF * X.f.gr + (size_t) E * X.f.dr;
        fill(X, 1234);
        Lane ln;
        ln.pool = &pool;
        if (check) {
            if (X.f.tg != 0 && !check_fmt(ln, X, x.data())) ++failed;
            continue;
        }
        int e0 = 0;
        auto timed = [&](int n, int reps) {
            std::vector<double> t;
            for (int k = 0; k < reps; ++k) {
                const double t0 = now_ms();
                ln.call(X, e0, n, x.data());
                t.push_back(now_ms() - t0);
                e0 = (e0 + n) % X.n;
            }
            std::sort(t.begin(), t.end());
            return t[t.size() / 2];
        };
        for (const double t0 = now_ms(); now_ms() - t0 < 50.0;) timed(3, 1);
        double t1 = 1e30, t3 = 1e30;
        for (int round = 0; round < 5; ++round) {
            t1 = std::min(t1, timed(1, 4));
            t3 = std::min(t3, timed(3, 4));
        }
        pcie::Costs k;
        k.c = std::max(0.5 * (t3 - t1), t3 / 6.0);
        k.oc = std::max(0.0, t1 - k.c);
        // the stream lane's read into pinned memory: an expert copied, by 1 and by 2 threads at once
        std::vector<uint8_t> stg(2 * X.f.blob);
        auto copy_ms = [&](int nt) {
            double best = 1e30;
            for (int r = 0; r < 6; ++r) {
                const double t0 = now_ms();
                std::vector<std::thread> th;
                for (int q = 0; q < nt; ++q)
                    th.emplace_back([&, q] { std::memcpy(stg.data() + q * X.f.blob, X.mat((r * 2 + q) % X.n, 0), X.f.blob); });
                for (auto& t : th) t.join();
                best = std::min(best, (now_ms() - t0) / nt);
            }
            return best;
        };
        const double r1 = copy_ms(1), r2 = copy_ms(2);
        // a link of `link` GB/s: an expert's copy overlaps its read but the last of its three matrices
        const double dma = (double) X.f.blob / (link * 1e9) * 1e3;
        k.p = std::max(r2, dma) + std::min(r2, dma) / 3.0;
        k.op = 0.05;
        k.g = gpu_ms;
        k.disk = (double) X.f.blob / 2e9 * 1e3;
        std::printf("%-8s expert %.1f MB: CPU lane %.3f ms an expert (%.1f GB/s) + %.3f a call | read into pinned "
                    "memory %.3f ms (1 thread), %.3f (2 at once) | a %.0f GB/s link: %.3f ms an expert streamed -> of "
                    "0..8 picks in RAM it streams %s\n",
                    fn.c_str(), X.f.blob / 1e6, k.c, X.f.blob / k.c / 1e6, k.oc, r1, r2, link, k.p,
                    pcie::table(k, 8).c_str());
    }
    if (check) std::printf("check: %s (%d failed)\n", failed ? "FAILED" : "ALL OK", failed);
    return failed ? 1 : 0;
}
