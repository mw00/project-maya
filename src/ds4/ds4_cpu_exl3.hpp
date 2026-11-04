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
    // DS4_CPU_SPLIT_COLS: the threads a node's batch runs on (its pool's, the caller's included on node 0's), from
    // which a small call's pieces are cut by columns too (exl3_split_runs); 0: every piece whole, as before
    int split_w = 0;
};

/// DS4_CPU_SPLIT_COLS: the column runs each piece of a step is cut into, for `jobs` uncut jobs of a node's batch (the
/// step's pieces of every entry), `threads` threads running it (ds4_pool.hpp: each job claimed in turn by whichever
/// thread is free, the jobs of a step all the same work) and `bands` 128-column bands a piece covers.  A step of J
/// equal jobs on W threads takes ceil(J / W) jobs' time, so a call of one or two entries leaves threads idle for a
/// whole job: cut into s equal runs it takes ceil(J s / W) / s.  The runs: a divisor of the bands (equal runs - every
/// job of the step still the same work), at most kSplitJobs jobs a thread in all (each job one claim on the pool's
/// shared ticket, and a run of at least one band), the one taking the least time, the fewest on a tie; 1 (the step
/// as before) for a step of kSplitJobs jobs a thread already, for one thread, or where no cut takes less time.
/// Each column's sums are the uncut piece's whatever the runs (exl3_cpu_rows_acc_cols_n), so the outputs are too.
constexpr int kSplitJobs = 8;
inline int exl3_split_runs(int jobs, int threads, int bands) {
    if (jobs <= 0 || threads <= 1 || bands <= 1 || (long) jobs >= (long) kSplitJobs * threads) return 1;
    int best = 1;
    long best_t = ((long) jobs + threads - 1) / threads;   // rounds of the uncut step (a time of best_t / best)
    for (int s = 2; s <= bands && (long) jobs * s <= (long) kSplitJobs * threads; ++s) {
        if (bands % s != 0) continue;
        const long t = ((long) jobs * s + threads - 1) / threads;
        if (t * best < best_t * s) {   // t / s < best_t / best
            best = s;
            best_t = t;
        }
    }
    return best;
}

/// exl3_experts with each gate / up piece cut into R0 runs of node X's columns and each down piece into R2 runs
/// (DS4_CPU_SPLIT_COLS, exl3_split_runs): steps 0 and 2 have R0 / R2 times the jobs, each one run of its piece
/// (exl3_cpu_rows_acc_cols_n); steps 1 and 3, the flow's order and its waits are exl3_experts', the counters counting
/// the runs.  A gate / up run prepares its piece's two input blocks of each token on its own stack (the same function
/// of the same inputs: the same values) and reads the piece's slices from 0 there, the run-0 job alone storing the
/// blocks' sums for step 1, so no job reads what another writes but through the steps' order or the counters.  Each
/// column's sums are the uncut piece's in the same order, and steps 1 / 3 add the pieces in the same order: the
/// outputs are exl3_experts' uncut ones, bit for bit (tests/ds4_cpu_split_test.cpp)
template <class Part, class Nodes, class Phase>
void exl3_experts_cols(const Exl3Lane& c, const float* x, size_t ldx, const LaneEnt* ents, int ne, const Part& part,
                       const Nodes& nodes, const Phase& phase, bool flow, int R0, int R2) {
    namespace kc = strata::kernels::cpu;
    const int E = c.E, FF = c.FF, NH = c.NH, S = c.slots, FFh = FF / NH, Eh = E / NH, GQ = E / 256, DQ = FF / 128;
    const int GB = FFh / 128, DB = Eh / 128, EB = E / 128, FB = FF / 128;
    const int CG = 128 * (GB / R0), CD = 128 * (DB / R2);   // a run's columns
    const float lim = c.lim;
    auto xsb = [&](int X, int sl, int r) { return c.xsb + ((size_t) (X * S + sl) * 2 + r) * EB; };
    auto acc = [&](int X, int sl, int r, int q) { return c.acc + (((size_t) (X * S + sl) * 2 + r) * GQ + q) * FFh; };
    auto xhd = [&](int sl) { return c.xhd + (size_t) sl * FF; };
    auto xsd = [&](int sl) { return c.xsd + (size_t) sl * FB; };
    auto accd = [&](int X, int sl, int q) { return c.accd + ((size_t) (X * S + sl) * DQ + q) * Eh; };
    // step 0: run u of entry en's matrix r (gate / up), input piece q (256 inputs): node X's columns [CG u, CG (u + 1))
    auto step0 = [&](int X, int j) {
        const int u = j % R0;
        j /= R0;
        const LaneEnt& en = ents[j / (2 * GQ)];
        const int r = (j / GQ) % 2, q = j % GQ;
        const kc::Exl3Part p = part(en.e, X, r);
        kc::Exl3Part pq = p;   // the piece's 16 slices of tile rows alone ([k/16][n/16][4 k2] words), inputs from 0
        pq.trellis = p.trellis + (size_t) q * 16 * (size_t) (p.n / 16) * (size_t) (4 * p.k2);
        pq.k = 256;
        alignas(64) float xl[kc::kExl3CpuMaxN][256];
        const float* xp[kc::kExl3CpuMaxN];
        float* ap[kc::kExl3CpuMaxN];
        for (int i = 0; i < en.n; ++i) {
            const int sl = en.slot0 + i;
            const float* xt = x + (size_t) en.tok[i] * ldx;
            for (int b = 0; b < 2; ++b) {
                const int g = 2 * q + b;
                const float s = kc::exl3_cpu_prepare_block(p.suh + g * 128, xt + g * 128, xl[i] + b * 128);
                if (u == 0) xsb(X, sl, r)[g] = s;
            }
            xp[i] = xl[i];
            ap[i] = acc(X, sl, r, q);
        }
        kc::exl3_cpu_rows_acc_cols_n(pq, xp, en.n, 0, 16, CG * u, CG * (u + 1), ap);
    };
    // step 1: band b of node X's gate / up columns -> h's block X GB + b, prepared for the down (exl3_experts')
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
    // step 2: run u of the down's input piece q (128 inputs): node X's columns [CD u, CD (u + 1))
    auto step2 = [&](int X, int j) {
        const int u = j % R2;
        j /= R2;
        const LaneEnt& en = ents[j / DQ];
        const int q = j % DQ;
        const kc::Exl3Part p = part(en.e, X, 2);
        const float* xp[kc::kExl3CpuMaxN];
        float* ap[kc::kExl3CpuMaxN];
        for (int i = 0; i < en.n; ++i) {
            xp[i] = xhd(en.slot0 + i);
            ap[i] = accd(X, en.slot0 + i, q);
        }
        kc::exl3_cpu_rows_acc_cols_n(p, xp, en.n, q * 8, (q + 1) * 8, CD * u, CD * (u + 1), ap);
    };
    // step 3: band b of node X's down columns -> the outputs (exl3_experts')
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
    const int J0 = ne * 2 * GQ * R0, J1 = ne * GB, J2 = ne * DQ * R2, J3 = ne * DB;
    std::function<void(int)> f[4];
    if (!flow) {
        for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { step0(X, j); };
        nodes(J0, f, 0);
        phase(0);
        for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { step1(X, j); };
        nodes(J1, f, 1);
        phase(1);
        for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { step2(X, j); };
        nodes(J2, f, 2);
        phase(2);
        for (int X = 0; X < NH; ++X) f[X] = [&, X](int j) { step3(X, j); };
        nodes(J3, f, 3);
        phase(3);
        return;
    }
    // flow: exl3_experts' batch, a step's runs counted where its pieces were
    std::atomic<int>* cnt = c.cnt.get();
    for (int i = 0; i < (2 * NH + 1) * S; ++i) cnt[i].store(0, std::memory_order_relaxed);
    auto c0 = [&](int X, int en) -> std::atomic<int>& { return cnt[(size_t) X * S + en]; };
    auto c1 = [&](int en) -> std::atomic<int>& { return cnt[(size_t) NH * S + en]; };
    auto c2 = [&](int X, int en) -> std::atomic<int>& { return cnt[(size_t) (NH + 1) * S + (size_t) X * S + en]; };
    auto until = [](const std::atomic<int>& a, int n) {
        while (a.load(std::memory_order_acquire) < n) cpu_relax();
    };
    for (int X = 0; X < NH; ++X)
        f[X] = [&, X](int j) {
            if (j < J0) {
                step0(X, j);
                c0(X, j / (2 * GQ * R0)).fetch_add(1, std::memory_order_release);
                return;
            }
            j -= J0;
            if (j < J1) {
                until(c0(X, j / GB), 2 * GQ * R0);
                step1(X, j);
                c1(j / GB).fetch_add(1, std::memory_order_release);
                return;
            }
            j -= J1;
            if (j < J2) {
                until(c1(j / (DQ * R2)), NH * GB);
                step2(X, j);
                c2(X, j / (DQ * R2)).fetch_add(1, std::memory_order_release);
                return;
            }
            j -= J2;
            until(c2(X, j / DB), DQ * R2);
            step3(X, j);
        };
    nodes(J0 + J1 + J2 + J3, f, -1);
}

/// The entries' outputs into lane.yd (row slot0 + i: entry's i-th token, unweighted).  part(e, X, r): node X's piece
/// of expert e's matrix r (0 gate, 1 up, 2 down); nodes(n, f, step): f[X](0 .. n - 1) on node X's pool, every node at
/// once, returning when all are done (step: 0..3 the steps' batches, -1 flow's).  phase(i): called after step i (steps).
/// c.split_w > 1 (DS4_CPU_SPLIT_COLS) and a call small enough for a cut to pay: exl3_experts_cols, the same bits.
template <class Part, class Nodes, class Phase>
void exl3_experts(const Exl3Lane& c, const float* x, size_t ldx, const LaneEnt* ents, int ne, const Part& part,
                  const Nodes& nodes, const Phase& phase, bool flow) {
    if (c.split_w > 1) {
        const int R0 = exl3_split_runs(ne * 2 * (c.E / 256), c.split_w, c.FF / c.NH / 128);
        const int R2 = exl3_split_runs(ne * (c.FF / 128), c.split_w, c.E / c.NH / 128);
        if (R0 > 1 || R2 > 1) return exl3_experts_cols(c, x, ldx, ents, ne, part, nodes, phase, flow, R0, R2);
    }
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
