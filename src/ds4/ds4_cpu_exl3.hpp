// src/ds4/ds4_cpu_exl3.hpp - the CPU lane's EXL3 experts: a window's entries (an entry: one expert for up to
// kExl3CpuMaxN of its tokens) over the NUMA nodes' pools, node X computing its share of every matrix - the gate / up
// columns [X n_ff / NH, ..), then (h whole) the down columns [X dim / NH, ..).
//
// Four steps, each cut into jobs: (0) the gate / up sums of 256-input pieces, (1) a 128-column band's sums over the
// pieces -> its outputs, the SwiGLU, the down's input block prepared, (2) the down's sums of 128-input pieces, (3) a
// band's sums over the pieces -> the outputs.  Steps: four batches, each started once the whole previous one is done.
// Flow: one batch a node, its jobs in step order, each job waiting only for the jobs whose sums it reads (counters by
// entry) - the same arithmetic in the same order, bit for bit, without three waits a layer for the slowest job.  A
// flow job waits only for jobs of its own node's batch claimed before it, or (step 2) for the other nodes' step-1 jobs,
// which wait for nothing but their own node's step 0: every claimed job runs on some thread, so the batch always ends.
#pragma once

#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "ds4_pool.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace strata::ds4 {

/// One expert of a window for n of its tokens: the CPU lane's rows slot0 .. slot0 + n - 1.  src[r]: a GGUF expert's
/// matrix r in a RAM store slot (--ssd-cache) for this call, else null (where the layer has it)
struct LaneEnt {
    int e = 0, n = 0, slot0 = 0;
    int tok[kernels::cpu::kExl3CpuMaxN] = {};
    const uint8_t* src[3] = {nullptr, nullptr, nullptr};
};

/// The CPU lane's EXL3 scratch (the engine's buffers) and geometry.
struct Exl3Lane {
    int E = 0, FF = 0, NH = 1, slots = 0;   // dim, n_ff, the nodes, the rows the buffers hold
    float lim = 0.0f;                       // the SwiGLU's clamp (0: none)
    float* xh = nullptr;     // [NH][slots][2][E]
    float* xsb = nullptr;    // [NH][slots][2][E / 128]
    float* acc = nullptr;    // [NH][slots][2][E / 256][FF / NH]
    float* xhd = nullptr;    // [slots][FF]
    float* xsd = nullptr;    // [slots][FF / 128]
    float* accd = nullptr;   // [NH][slots][FF / 128][E / NH]
    float* yd = nullptr;     // [slots][E]: the outputs
    std::unique_ptr<std::atomic<int>[]> cnt;   // flow: [NH][slots] step 0, [slots] step 1, [NH][slots] step 2
    void alloc_counters(int max_nh) { cnt.reset(new std::atomic<int>[(size_t) (2 * max_nh + 1) * (size_t) slots]); }
};

/// The entries' outputs into lane.yd (row slot0 + i: entry's i-th token, unweighted).  part(e, X, r): node X's piece
/// of expert e's matrix r (0 gate, 1 up, 2 down); nodes(n, f, step): f[X](0 .. n - 1) on node X's pool, every node at
/// once, returning when all are done (step: 0..3 the steps' batches, -1 flow's).  phase(i): called after step i (steps).
template <class Part, class Nodes, class Phase>
void exl3_experts(const Exl3Lane& c, const float* x, size_t ldx, const LaneEnt* ents, int ne, const Part& part,
                  const Nodes& nodes, const Phase& phase, bool flow) {
    namespace kc = strata::kernels::cpu;
    const int E = c.E, FF = c.FF, NH = c.NH, S = c.slots, FFh = FF / NH, Eh = E / NH, GQ = E / 256, DQ = FF / 128;
    const int GB = FFh / 128, DB = Eh / 128, EB = E / 128, FB = FF / 128;
    const float lim = c.lim;
    auto xh = [&](int X, int sl, int r) { return c.xh + ((size_t) (X * S + sl) * 2 + r) * E; };
    auto xsb = [&](int X, int sl, int r) { return c.xsb + ((size_t) (X * S + sl) * 2 + r) * EB; };
    auto acc = [&](int X, int sl, int r, int q) { return c.acc + (((size_t) (X * S + sl) * 2 + r) * GQ + q) * FFh; };
    auto xhd = [&](int sl) { return c.xhd + (size_t) sl * FF; };
    auto xsd = [&](int sl) { return c.xsd + (size_t) sl * FB; };
    auto accd = [&](int X, int sl, int q) { return c.accd + ((size_t) (X * S + sl) * DQ + q) * Eh; };
    // step 0: entry en's matrix r (gate / up), input piece q (256 inputs) over node X's columns
    auto step0 = [&](int X, int j) {
        const LaneEnt& en = ents[j / (2 * GQ)];
        const int r = (j / GQ) % 2, q = j % GQ;
        const kc::Exl3Part p = part(en.e, X, r);
        const float* xp[kc::kExl3CpuMaxN];
        float* ap[kc::kExl3CpuMaxN];
        for (int i = 0; i < en.n; ++i) {
            const int sl = en.slot0 + i;
            const float* xt = x + (size_t) en.tok[i] * ldx;
            for (int b = 2 * q; b < 2 * q + 2; ++b)
                xsb(X, sl, r)[b] = kc::exl3_cpu_prepare_block(p.suh + b * 128, xt + b * 128, xh(X, sl, r) + b * 128);
            xp[i] = xh(X, sl, r);
            ap[i] = acc(X, sl, r, q);
        }
        if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 16, (q + 1) * 16, ap[0]);
        else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 16, (q + 1) * 16, ap);
    };
    // step 1: band b of node X's gate / up columns -> h's block X GB + b, prepared for the down
    auto step1 = [&](int X, int j) {
        const LaneEnt& en = ents[j / GB];
        const int b = j % GB;
        const kc::Exl3Part pg = part(en.e, X, 0), pu = part(en.e, X, 1), pd = part(en.e, X, 2);
        for (int i = 0; i < en.n; ++i) {
            const int sl = en.slot0 + i;
            float y[2][128], a[128], hb[128];
            for (int r = 0; r < 2; ++r) {
                for (int cc = 0; cc < 128; ++cc) a[cc] = 0.0f;
                for (int q = 0; q < GQ; ++q) {
                    const float* aq = acc(X, sl, r, q) + b * 128;
                    for (int cc = 0; cc < 128; ++cc) a[cc] += aq[cc];
                }
                double xs = 0.0;
                for (int k = 0; k < EB; ++k) xs += xsb(X, sl, r)[k];
                kc::exl3_cpu_band_out(r == 0 ? pg : pu, a, (float) xs, b * 128, y[r]);
            }
            for (int cc = 0; cc < 128; ++cc) {
                float gv = y[0][cc], uv = y[1][cc];
                if (lim > 0.0f) {
                    gv = std::min(gv, lim);
                    uv = std::min(std::max(uv, -lim), lim);
                }
                hb[cc] = gv / (1.0f + std::exp(-gv)) * uv;
            }
            const int gb = X * GB + b;
            xsd(sl)[gb] = kc::exl3_cpu_prepare_block(pd.suh + gb * 128, hb, xhd(sl) + gb * 128);
        }
    };
    // step 2: the down's input piece q (128 inputs) over node X's columns
    auto step2 = [&](int X, int j) {
        const LaneEnt& en = ents[j / DQ];
        const int q = j % DQ;
        const kc::Exl3Part p = part(en.e, X, 2);
        const float* xp[kc::kExl3CpuMaxN];
        float* ap[kc::kExl3CpuMaxN];
        for (int i = 0; i < en.n; ++i) {
            xp[i] = xhd(en.slot0 + i);
            ap[i] = accd(X, en.slot0 + i, q);
        }
        if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 8, (q + 1) * 8, ap[0]);
        else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 8, (q + 1) * 8, ap);
    };
    // step 3: band b of node X's down columns -> the outputs
    auto step3 = [&](int X, int j) {
        const LaneEnt& en = ents[j / DB];
        const int b = j % DB;
        const kc::Exl3Part pd = part(en.e, X, 2);
        for (int i = 0; i < en.n; ++i) {
            const int sl = en.slot0 + i;
            float a[128];
            for (int cc = 0; cc < 128; ++cc) a[cc] = 0.0f;
            for (int q = 0; q < DQ; ++q) {
                const float* aq = accd(X, sl, q) + b * 128;
                for (int cc = 0; cc < 128; ++cc) a[cc] += aq[cc];
            }
            double xs = 0.0;
            for (int k = 0; k < FB; ++k) xs += xsd(sl)[k];
            kc::exl3_cpu_band_out(pd, a, (float) xs, b * 128, c.yd + (size_t) sl * E + (size_t) X * Eh + b * 128);
        }
    };
    std::function<void(int)> f[4];
    if (!flow) {
        for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { step0(X, j); };
        nodes(ne * 2 * GQ, f, 0);
        phase(0);
        for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { step1(X, j); };
        nodes(ne * GB, f, 1);
        phase(1);
        for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { step2(X, j); };
        nodes(ne * DQ, f, 2);
        phase(2);
        for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { step3(X, j); };
        nodes(ne * DB, f, 3);
        phase(3);
        return;
    }
    std::atomic<int>* cnt = c.cnt.get();
    for (int i = 0; i < (2 * NH + 1) * S; ++i) cnt[i].store(0, std::memory_order_relaxed);
    auto c0 = [&](int X, int en) -> std::atomic<int>& { return cnt[(size_t) X * S + en]; };
    auto c1 = [&](int en) -> std::atomic<int>& { return cnt[(size_t) NH * S + en]; };
    auto c2 = [&](int X, int en) -> std::atomic<int>& { return cnt[(size_t) (NH + 1) * S + (size_t) X * S + en]; };
    auto until = [](const std::atomic<int>& a, int n) {
        while (a.load(std::memory_order_acquire) < n) cpu_relax();
    };
    const int J0 = ne * 2 * GQ, J1 = ne * GB, J2 = ne * DQ, J3 = ne * DB;
    for (int X = 0; X < NH; ++X)
        f[X] = [&, X](int j) {
            if (j < J0) {
                step0(X, j);
                c0(X, j / (2 * GQ)).fetch_add(1, std::memory_order_release);
                return;
            }
            j -= J0;
            if (j < J1) {
                until(c0(X, j / GB), 2 * GQ);
                step1(X, j);
                c1(j / GB).fetch_add(1, std::memory_order_release);
                return;
            }
            j -= J1;
            if (j < J2) {
                until(c1(j / DQ), NH * GB);
                step2(X, j);
                c2(X, j / DQ).fetch_add(1, std::memory_order_release);
                return;
            }
            j -= J2;
            until(c2(X, j / DB), DQ);
            step3(X, j);
        };
    nodes(J0 + J1 + J2 + J3, f, -1);
}

}  // namespace strata::ds4
