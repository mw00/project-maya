// tests/ds4_cpu_flow_test.cpp - the CPU lane's EXL3 experts (src/ds4/ds4_cpu_exl3.hpp): flow against steps, bit for
// bit, then the time of a call of each (no GPU, no model).
//
//   ds4_cpu_flow_test [--threads N] [--experts N] [--cases N] [--iters N]
//
// DeepSeek-V4.1-Flash's geometry (5120 -> 2304 -> 5120, K 3), each expert in two NUMA parts (node X: the gate / up
// columns [X 1152, ..), the down columns [X 2560, ..)) of random trellis words in memory bound to its node, a pool of
// --threads a node pinned to its physical cores (CPU i on node i % 2 here, CPUs 0..43 the physical cores), the caller
// on CPU 0 helping node 0's pool as the engine's service thread does.  Random windows (1-12 entries of 1-6 tokens)
// for the parity; decode-like ones for the time, the two ways taking turns.
#include "../src/ds4/ds4_cpu_exl3.hpp"

#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace kc = strata::kernels::cpu;
using strata::ds4::Exl3Lane;
using strata::ds4::LaneEnt;
using strata::ds4::Pool;

static void* node_alloc(size_t bytes, int node) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    madvise(p, bytes, MADV_HUGEPAGE);
    unsigned long mask = 1ul << node;
    if (syscall(SYS_mbind, p, bytes, 2 /*MPOL_BIND*/, &mask, 64, 0) != 0) std::perror("mbind");
    return p;
}

int main(int argc, char** argv) {
    int threads = 22, n_exp = 24, cases = 200, iters = 200;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--threads") threads = std::atoi(argv[i + 1]);
        else if (a == "--experts") n_exp = std::atoi(argv[i + 1]);
        else if (a == "--cases") cases = std::atoi(argv[i + 1]);
        else if (a == "--iters") iters = std::atoi(argv[i + 1]);
    }
    constexpr int E = 5120, FF = 2304, NH = 2, K2 = 6, kWin = 16, kPicks = 6, S = kWin * kPicks;
    constexpr int FFh = FF / NH, Eh = E / NH;
    const size_t mat_gu = (size_t) E / 16 * FFh / 16 * 16 * K2, mat_d = (size_t) FF / 16 * Eh / 16 * 16 * K2;
    const size_t part_bytes = 2 * mat_gu + mat_d;
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(0, &set);
        sched_setaffinity(0, sizeof set, &set);
    }
    std::vector<std::unique_ptr<Pool>> pools;
    for (int X = 0; X < NH; ++X) {
        std::vector<int> cpus;
        for (int c = X; c < 44 && (int) cpus.size() < threads; c += 2) cpus.push_back(c);
        if (X == 0) cpus.erase(cpus.begin());   // (CPU 0: the caller's)
        pools.push_back(std::make_unique<Pool>((int) cpus.size(), cpus, 20000));
    }
    // the parts: [expert][node] gate | up | down trellis, the scales beside (FP16 near 1)
    uint8_t* mem[NH];
    for (int X = 0; X < NH; ++X) {
        mem[X] = (uint8_t*) node_alloc(part_bytes * n_exp, X);
        if (mem[X] == nullptr) {
            std::fprintf(stderr, "no memory\n");
            return 1;
        }
        pools[(size_t) X]->run(n_exp, [&, X](int e) {
            std::mt19937 r((unsigned) (e * 2 + X + 1));
            uint32_t* w = (uint32_t*) (mem[X] + part_bytes * e);
            for (size_t i = 0; i < part_bytes / 4; ++i) w[i] = r();
        });
    }
    std::mt19937 rs(7);
    auto scale = [&] { return (uint16_t) (0x3800 + (rs() % 0x0800)); };   // 0.5 .. 1.0
    std::vector<uint16_t> suh_gu((size_t) n_exp * 2 * E), svh_gu((size_t) n_exp * NH * 2 * FFh);
    std::vector<uint16_t> suh_d((size_t) n_exp * FF), svh_d((size_t) n_exp * NH * Eh);
    for (auto& v : suh_gu) v = scale();
    for (auto& v : svh_gu) v = scale();
    for (auto& v : suh_d) v = scale();
    for (auto& v : svh_d) v = scale();
    auto part = [&](int e, int X, int r) {
        kc::Exl3Part p;
        const uint8_t* b = mem[X] + part_bytes * e;
        p.k2 = K2;
        if (r < 2) {
            p.trellis = (const uint32_t*) (b + (size_t) r * mat_gu);
            p.suh = suh_gu.data() + ((size_t) e * 2 + r) * E;
            p.svh = svh_gu.data() + (((size_t) e * NH + X) * 2 + r) * FFh;
            p.k = E;
            p.n = FFh;
        } else {
            p.trellis = (const uint32_t*) (b + 2 * mat_gu);
            p.suh = suh_d.data() + (size_t) e * FF;
            p.svh = svh_d.data() + ((size_t) e * NH + X) * Eh;
            p.k = FF;
            p.n = Eh;
        }
        return p;
    };
    auto nodes = [&](int n, const std::function<void(int)>* f, int) {
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, f[X]);
        pools[0]->help();
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
    };
    auto nophase = [](int) {};
    std::vector<float> xh((size_t) NH * S * 2 * E), xsb((size_t) NH * S * 2 * (E / 128)),
        acc((size_t) NH * S * 2 * (E / 256) * FFh), xhd((size_t) S * FF), xsd((size_t) S * (FF / 128)),
        accd((size_t) NH * S * (FF / 128) * Eh), yd((size_t) S * E);
    Exl3Lane c;
    c.E = E;
    c.FF = FF;
    c.NH = NH;
    c.slots = S;
    c.lim = 10.0f;
    c.xh = xh.data();
    c.xsb = xsb.data();
    c.acc = acc.data();
    c.xhd = xhd.data();
    c.xsd = xsd.data();
    c.accd = accd.data();
    c.yd = yd.data();
    c.alloc_counters(NH);
    std::vector<float> x((size_t) kWin * E);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = nd(rs);
    // a window: ne entries (distinct experts) of n tokens each (distinct, < kWin), slots in order
    auto window = [&](int ne, int nmax, std::vector<LaneEnt>& ents) {
        ents.clear();
        std::vector<int> ex((size_t) n_exp);
        for (int e = 0; e < n_exp; ++e) ex[(size_t) e] = e;
        std::shuffle(ex.begin(), ex.end(), rs);
        int slot = 0;
        for (int i = 0; i < ne && slot < S; ++i) {
            LaneEnt en;
            en.e = ex[(size_t) i % n_exp];
            en.n = std::min(1 + (int) (rs() % (unsigned) nmax), S - slot);
            std::vector<int> tk(kWin);
            for (int t = 0; t < kWin; ++t) tk[(size_t) t] = t;
            std::shuffle(tk.begin(), tk.end(), rs);
            for (int t = 0; t < en.n; ++t) en.tok[t] = tk[(size_t) t];
            en.slot0 = slot;
            slot += en.n;
            ents.push_back(en);
        }
        return slot;
    };
    // parity
    std::vector<LaneEnt> ents;
    int bad = 0;
    for (int k = 0; k < cases; ++k) {
        const int rows = window(1 + (int) (rs() % 12), 1 + (int) (rs() % 6), ents);
        std::fill(yd.begin(), yd.end(), -1.0f);
        strata::ds4::exl3_experts(c, x.data(), E, ents.data(), (int) ents.size(), part, nodes, nophase, false);
        std::vector<float> a(yd.begin(), yd.begin() + (size_t) rows * E);
        std::fill(yd.begin(), yd.end(), -2.0f);
        strata::ds4::exl3_experts(c, x.data(), E, ents.data(), (int) ents.size(), part, nodes, nophase, true);
        if (std::memcmp(a.data(), yd.data(), a.size() * 4) != 0) {
            ++bad;
            size_t i = 0;
            while (i < a.size() && std::memcmp(&a[i], &yd[i], 4) == 0) ++i;
            std::fprintf(stderr, "case %d: %zu entries, row %zu col %zu: steps %.9g flow %.9g\n", k, ents.size(),
                         i / E, i % E, a[i], yd[i]);
        }
        bool finite = true;
        for (float v : a) finite = finite && std::isfinite(v);
        if (!finite && k == 0) std::fprintf(stderr, "(case 0 has non-finite outputs: the random weights' scale)\n");
    }
    std::printf("parity: %d of %d windows differ (flow against steps, bit for bit)\n", bad, cases);
    // time: decode-like windows, the two ways in turn
    struct Cfg {
        int ne, n;
    };
    for (const Cfg cf : {Cfg{1, 1}, Cfg{2, 1}, Cfg{4, 1}, Cfg{3, 2}, Cfg{6, 3}, Cfg{12, 4}}) {
        double ms[2] = {0, 0};
        for (int it = 0; it < iters; ++it)
            for (int w = 0; w < 2; ++w) {
                const int way = (it + w) % 2;
                ents.clear();
                int slot = 0;
                for (int i = 0; i < cf.ne; ++i) {
                    LaneEnt en;
                    en.e = (it * cf.ne + i) % n_exp;
                    en.n = cf.n;
                    for (int t = 0; t < cf.n; ++t) en.tok[t] = t;
                    en.slot0 = slot;
                    slot += cf.n;
                    ents.push_back(en);
                }
                const auto t0 = std::chrono::steady_clock::now();
                strata::ds4::exl3_experts(c, x.data(), E, ents.data(), cf.ne, part, nodes, nophase, way == 1);
                ms[way] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
        std::printf("%2d experts x %d tokens: steps %.3f ms, flow %.3f ms a call (%+.1f%%)\n", cf.ne, cf.n, ms[0] / iters,
                    ms[1] / iters, 100.0 * (ms[1] - ms[0]) / ms[0]);
    }
    return bad == 0 ? 0 : 1;
}
