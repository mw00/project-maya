// tests/mm3_lane_test.cpp - maya-mm3's CPU lane: run_multi's one-dispatch schedule against the four phases
// (MM3_LANE_OLD) on random EXL3 sub-experts (MiniMax-M3.1's at --tp 4: 6144 x 768 gate / up, 768 x 6144 down, 4 bits)
// - the same bits - and the time of each.
//     g++ -O3 -std=c++17 -mavx2 -mfma -mf16c -Iinclude -Isrc/mm3 tests/mm3_lane_test.cpp src/mm3/mm3_cpu.cpp \
//         src/kernels/cpu/exl3_cpu.cpp -lpthread -o build/mm3_lane_test && build/mm3_lane_test [threads]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "mm3_cpu.hpp"

// (expert_layout.cpp's, which the rest of that file would bring ggml with)
namespace strata::kernels::cpu {
bool cpu_avx2_ok() { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }
// (pool.cpp's, which would bring the rest of the CPU kernels: every logical CPU stands in for a core here)
std::vector<int> physical_cores(bool skip_first) {
    std::vector<int> c;
    const int n = (int) std::thread::hardware_concurrency();
    for (int i = skip_first && n > 1 ? 1 : 0; i < n; ++i) c.push_back(i);
    return c;
}
}

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 15;
    const int dim = 6144, ff = 768, k2 = 8, NE = 12;
    std::mt19937 rng(3);
    std::vector<std::vector<uint32_t>> tr;
    std::vector<std::vector<uint16_t>> sv;
    std::vector<mm3::CpuExpert> ex(NE);
    for (int e = 0; e < NE; ++e)
        for (int r = 0; r < 3; ++r) {
            const int k = r < 2 ? dim : ff, n = r < 2 ? ff : dim;
            tr.emplace_back((size_t) (k / 16) * (n / 16) * 4 * k2);
            for (uint32_t& w : tr.back()) w = rng();
            sv.emplace_back((size_t) k);
            for (uint16_t& h : sv.back()) h = (rng() & 1 ? 0x3c00 : 0xbc00) ^ (uint16_t) (rng() & 0x00ff);
            const uint16_t* suh = sv.back().data();
            sv.emplace_back((size_t) n);
            for (uint16_t& h : sv.back()) h = (rng() & 1 ? 0x3800 : 0xb800) ^ (uint16_t) (rng() & 0x00ff);
            auto& p = ex[(size_t) e].p[r];
            p.trellis = tr.back().data();
            p.suh = suh;
            p.svh = sv.back().data();
            p.k = k;
            p.n = n;
            p.k2 = k2;
            ex[(size_t) e].w = 1.0f;
        }
    const int T = 8;
    std::vector<float> x((size_t) T * dim);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng);
    // the users: decode (1 token, 4 experts), a window (8 tokens), a prompt's cold experts (up to 16 tokens each)
    struct Case {
        const char* name;
        int ne, tok;
    } cases[] = {{"decode, 4 sub-experts x 1 token", 4, 1}, {"window, 12 sub-experts x 1-8 tokens", 12, 8},
                 {"prompt, 6 sub-experts x 8 tokens + 2 x 16", 8, 16}};
    // the users of each case, fixed first
    std::vector<std::vector<std::vector<std::pair<int, float>>>> us;
    for (const Case& cs : cases) {
        std::vector<std::vector<std::pair<int, float>>> users((size_t) cs.ne);
        for (int i = 0; i < cs.ne; ++i) {
            const int n = cs.tok == 1 ? 1 : 1 + (int) (rng() % (unsigned) std::min(cs.tok, 8 + 8 * (i >= 6)));
            for (int u = 0; u < n; ++u) users[(size_t) i].push_back({(int) (rng() % T), 0.1f + 0.01f * (float) (rng() % 50)});
        }
        us.push_back(users);
    }
    // each schedule on a lane of its own, one lane at a time (two lanes' workers would share the cores they pin to)
    std::vector<std::vector<float>> out[2];
    double best[2][3];
    for (int v = 0; v < 2; ++v) {
        if (v == 0) setenv("MM3_LANE_OLD", "1", 1);
        else unsetenv("MM3_LANE_OLD");
        mm3::CpuLane lane(threads, dim, ff, 1.702f, 7.0f, 256);
        for (int c = 0; c < 3; ++c) {
            std::vector<mm3::CpuExpert> e(ex.begin(), ex.begin() + cases[c].ne);
            std::vector<float> o((size_t) T * dim);
            best[v][c] = 1e9;
            for (int rep = 0; rep < 40; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                lane.run_multi(e, us[(size_t) c], x.data(), T, o.data());
                best[v][c] = std::min(best[v][c], std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            out[v].push_back(o);
        }
    }
    bool all_ok = true;
    for (int c = 0; c < 3; ++c) {
        const bool same = std::memcmp(out[0][(size_t) c].data(), out[1][(size_t) c].data(), out[0][(size_t) c].size() * 4) == 0;
        all_ok &= same;
        std::printf("%-44s four phases %.3f ms, one dispatch %.3f ms (best of 40), bits %s\n", cases[c].name,
                    1e3 * best[0][c], 1e3 * best[1][c], same ? "same" : "DIFFER");
    }
    std::printf("%s\n", all_ok ? "OK" : "FAILED");
    return all_ok ? 0 : 1;
}
