// tests/ds4_par_bench.cpp - DS4_LANE_PARITY's CPU rows (ds4_lane_parity.cpp) against the CPU lane's ggml rows
// (gg::iq_cpu_*): one thread, one expert of DeepSeek-V4.1-Flash's shape (5120 x 2304) with random weights, its gate /
// up then its down for nt tokens at once - ms an expert-token, the faster of several rounds (the expert's ~10 MB from
// L3 / RAM as the lane reads them, several experts in turn).  Also checks the fast rows against the scalar ones.
//
//   ds4_par_bench [--experts 8] [--rounds 5]
#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "../src/ds4/ds4_lane_parity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace par = strata::ds4::par;

namespace {
constexpr int E = 5120, FF = 2304;
constexpr float kLim = 10.0f;
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
int block_bytes(int t) { return t == 16 ? 66 : t == 22 ? 82 : t == 18 ? 98 : 110; }
std::vector<uint8_t> rand_mat(int type, int n, int k, std::mt19937& rng) {
    const size_t rb = (size_t) (k / 256) * block_bytes(type);
    std::vector<uint8_t> w(rb * (size_t) n);
    for (auto& v : w) v = (uint8_t) rng();
    for (int r = 0; r < n; ++r)
        for (int b = 0; b < k / 256; ++b) {
            uint8_t* p = w.data() + (size_t) r * rb + (size_t) b * block_bytes(type);
            const uint16_t h = (uint16_t) (0x1000u + (rng() % 0x1800u)) | (uint16_t) ((rng() & 1u) << 15);
            std::memcpy(p, &h, 2);
        }
    return w;
}
struct Expert {
    std::vector<uint8_t> g, u, d;
};
}  // namespace

int main(int argc, char** argv) {
    int n_exp = 8, rounds = 5;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--experts")) n_exp = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--rounds")) rounds = std::atoi(argv[i + 1]);
    }
    std::mt19937 rng(7);
    std::printf("par rows: %s\n", par::form());
    const int pairs[2][2] = {{16, 18}, {22, 21}};   // IQ2_XXS / IQ3_XXS, IQ2_S / IQ3_S
    for (const auto& pr : pairs) {
        const int tg = pr[0], td = pr[1];
        std::vector<Expert> X((size_t) n_exp);
        for (auto& x : X) x = {rand_mat(tg, FF, E, rng), rand_mat(tg, FF, E, rng), rand_mat(td, E, FF, rng)};
        gg::IqCpu fm;
        std::string err;
        if (!gg::iq_cpu_fmt(tg, td, E, FF, fm, err)) {
            std::printf("iq_cpu_fmt: %s\n", err.c_str());
            return 1;
        }
        for (int nt : {1, 2, 4}) {
            std::vector<std::vector<float>> x((size_t) nt, std::vector<float>(E)), h((size_t) nt, std::vector<float>(FF)),
                y((size_t) nt, std::vector<float>(E)), h2((size_t) nt, std::vector<float>(FF)), y2((size_t) nt, std::vector<float>(E));
            for (auto& v : x)
                for (auto& f : v) f = std::uniform_real_distribution<float>(-1, 1)(rng);
            std::vector<std::vector<uint8_t>> qa((size_t) nt, std::vector<uint8_t>(std::max(fm.act_bytes, par::act_bytes(E)))),
                qh((size_t) nt, std::vector<uint8_t>(std::max(fm.h_bytes, par::act_bytes(FF))));
            const void* ap[8];
            const void* hq[8];
            float *hp[8], *yp[8], *hp2[8], *yp2[8];
            for (int t = 0; t < nt; ++t) {
                ap[t] = qa[(size_t) t].data(), hq[t] = qh[(size_t) t].data();
                hp[t] = h[(size_t) t].data(), yp[t] = y[(size_t) t].data();
                hp2[t] = h2[(size_t) t].data(), yp2[t] = y2[(size_t) t].data();
            }
            auto run = [&](bool p) {
                double best = 1e30;
                for (int rd = 0; rd < rounds; ++rd) {
                    const double t0 = now_ms();
                    for (const Expert& ex : X) {
                        for (int t = 0; t < nt; ++t) {
                            if (p) par::quant(x[(size_t) t].data(), E, qa[(size_t) t].data());
                            else gg::iq_cpu_quant(fm.gu_act, x[(size_t) t].data(), E, qa[(size_t) t].data());
                        }
                        if (p) par::gate_up(tg, ex.g.data(), ex.u.data(), E, ap, nt, hp, 0, FF, kLim);
                        else gg::iq_cpu_gate_up(fm, ex.g.data(), ex.u.data(), ap, nt, hp, 0, FF, kLim);
                        for (int t = 0; t < nt; ++t) {
                            if (p) par::quant(h[(size_t) t].data(), FF, qh[(size_t) t].data());
                            else gg::iq_cpu_quant(fm.d_act, h[(size_t) t].data(), FF, qh[(size_t) t].data());
                        }
                        if (p) par::down(td, ex.d.data(), FF, hq, nt, yp, 0, E);
                        else gg::iq_cpu_down(fm, ex.d.data(), hq, nt, yp, 0, E);
                    }
                    best = std::min(best, (now_ms() - t0) / (double) X.size());
                }
                return best;
            };
            const double old_ms = run(false), new_ms = run(true);
            // the fast rows against the scalar ones, on the last expert
            const Expert& ex = X.back();
            par::gate_up(tg, ex.g.data(), ex.u.data(), E, ap, nt, hp, 0, FF, kLim);
            par::gate_up_ref(tg, ex.g.data(), ex.u.data(), E, ap, nt, hp2, 0, FF, kLim);
            for (int t = 0; t < nt; ++t) par::quant(h[(size_t) t].data(), FF, qh[(size_t) t].data());
            par::down(td, ex.d.data(), FF, hq, nt, yp, 0, E);
            par::down_ref(td, ex.d.data(), FF, hq, nt, yp2, 0, E);
            bool same = true;
            for (int t = 0; t < nt; ++t)
                same = same && !std::memcmp(h[(size_t) t].data(), h2[(size_t) t].data(), FF * 4) &&
                       !std::memcmp(y[(size_t) t].data(), y2[(size_t) t].data(), E * 4);
            std::printf("%-8s/%-8s nt %d: ggml rows %.3f ms, parity rows %.3f ms an expert (x%.2f), fast == scalar: %s\n",
                        tg == 16 ? "IQ2_XXS" : "IQ2_S", td == 18 ? "IQ3_XXS" : "IQ3_S", nt, old_ms, new_ms, new_ms / old_ms,
                        same ? "yes" : "NO");
        }
    }
    return 0;
}
