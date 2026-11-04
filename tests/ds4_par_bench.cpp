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
// MXFP4 (a GGUF's expert format): 32 values a block, an E8M0 scale then 16 code bytes
std::vector<uint8_t> rand_mx(int n, int k, std::mt19937& rng) {
    const size_t rb = (size_t) (k / 32) * 17;
    std::vector<uint8_t> w(rb * (size_t) n);
    for (auto& v : w) v = (uint8_t) rng();
    for (int r = 0; r < n; ++r)
        for (int b = 0; b < k / 32; ++b) w[(size_t) r * rb + (size_t) b * 17] = (uint8_t) (126 + (rng() % 5));
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
    // ---- MXFP4 down as a --tp 2 group splits it: the whole expert's rows (one member's sum) against the group's two FF
    // bands, each member's blocks of each row - what the lane pays a tensor-parallel layer, and how far the bands' F32 sum
    // is from the whole (the ulp the placement used to move)
    {
        const int TD = 39;   // MXFP4
        const size_t dr = (size_t) (FF / 32) * 17;   // (a whole row's bytes)
        std::vector<uint8_t> d = rand_mx(E, FF, rng);
        const int bands[2][2] = {{0, FF / 2}, {FF / 2, FF / 2}};
        for (int nt : {1, 2, 4}) {
            std::vector<std::vector<float>> xt((size_t) nt, std::vector<float>(FF)), y((size_t) nt, std::vector<float>(E)),
                yb((size_t) nt, std::vector<float>(E));
            std::vector<std::vector<uint8_t>> qh((size_t) nt, std::vector<uint8_t>(par::act_bytes(FF))),
                qb((size_t) nt, std::vector<uint8_t>(par::act_bytes(FF)));
            for (auto& v : xt)
                for (auto& f : v) f = std::uniform_real_distribution<float>(-1, 1)(rng);
            const void* hp[8];
            float* yp[8];
            float* ybp[8];
            for (int t = 0; t < nt; ++t) {
                par::quant(TD, xt[(size_t) t].data(), FF, qh[(size_t) t].data());
                hp[t] = qh[(size_t) t].data();
                yp[t] = y[(size_t) t].data();
                ybp[t] = yb[(size_t) t].data();
            }
            // a band's own activation (the member's h band: the same 32-value blocks, so the same int8 and scales)
            auto band_act = [&](const int* b) {
                for (int t = 0; t < nt; ++t) {
                    par::quant(TD, xt[(size_t) t].data() + b[0], b[1], qb[(size_t) t].data());
                    hp[t] = qb[(size_t) t].data();
                }
            };
            auto whole = [&] {
                double best = 1e30;
                for (int rd = 0; rd < rounds; ++rd) {
                    const double t0 = now_ms();
                    for (int e = 0; e < n_exp; ++e) par::down(TD, d.data(), FF, hp, nt, yp, 0, E);
                    best = std::min(best, (now_ms() - t0) / n_exp);
                }
                return best;
            };
            auto split = [&] {
                double best = 1e30;
                for (int rd = 0; rd < rounds; ++rd) {
                    const double t0 = now_ms();
                    for (int e = 0; e < n_exp; ++e)
                        for (const auto& b : bands) {
                            band_act(b);
                            par::down_band(TD, d.data() + (size_t) b[0] / 32 * 17, dr, b[1], hp, nt, ybp, 0, E);
                        }
                    best = std::min(best, (now_ms() - t0) / n_exp);
                }
                return best;
            };
            const double w_ms = whole(), s_ms = split();
            // the bands' F32 sum against the whole, each in its own order
            for (int t = 0; t < nt; ++t) hp[t] = qh[(size_t) t].data();
            par::down(TD, d.data(), FF, hp, nt, yp, 0, E);
            std::vector<float> sum((size_t) nt * E, 0.0f);
            for (int bi = 0; bi < 2; ++bi) {
                band_act(bands[bi]);
                par::down_band(TD, d.data() + (size_t) bands[bi][0] / 32 * 17, dr, bands[bi][1], hp, nt, ybp, 0, E);
                for (int t = 0; t < nt; ++t)
                    for (int c = 0; c < E; ++c) sum[(size_t) t * E + c] = bi == 0 ? yb[(size_t) t][c] : sum[(size_t) t * E + c] + yb[(size_t) t][c];
            }
            size_t diff = 0;
            double mx = 0.0, mxr = 0.0;
            for (size_t i = 0; i < sum.size(); ++i) {
                const float v = y[i / E][i % E];
                diff += std::memcmp(&sum[i], &v, 4) != 0;
                mx = std::max(mx, (double) std::fabs((double) sum[i] - (double) v));
                if (std::fabs((double) v) > 1.0) mxr = std::max(mxr, (double) std::fabs((double) sum[i] - (double) v) / std::fabs((double) v));
            }
            std::printf("MXFP4 down nt %d: whole row %.3f ms, two FF bands %.3f ms an expert (x%.2f); the bands' sum vs the "
                        "whole: %zu of %zu values differ, max |d| %.3g (rel %.2g)\n", nt, w_ms, s_ms, s_ms / w_ms, diff,
                        sum.size(), mx, mxr);
        }
    }
    return 0;
}
