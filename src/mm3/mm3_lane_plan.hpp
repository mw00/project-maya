// src/mm3/mm3_lane_plan.hpp - maya-mm3's CPU lane, its arithmetic alone (no CUDA, no threads: tests/mm3_lane_plan_test):
// how many workers it takes on this host by default, and decode's own timing of it (the costs the split between the
// CPU lane and the staging slots plans with - mm3_model.cu's lane_split).
#pragma once

#include <algorithm>
#include <cmath>

namespace mm3 {

/// The lane's workers by default (no --cpu-threads): one a physical core but the caller's (`workers`, of `logical`
/// CPUs the process may run on) - and where that leaves fewer than spare + 1 logical CPUs to the rest (no SMT siblings
/// free: a VM's vCPUs, a CPU without SMT), `spare` cores fewer, as maya-ds4 leaves them.  The engine's other threads
/// (the GPU's launches and waits, the copies, the server) and the OS then find a core instead of preempting a spinning
/// worker, which holds the whole call up (#552: 32 vCPUs, no SMT - the lane on all 32 decoded 13.0 tok/s, on 28 15.8).
/// A host with SMT keeps every core: the siblings are free (a lane of fewer cores there cost decode - maya-ds4's #394)
inline int lane_default_workers(int workers, int logical, int spare) {
    if (workers <= 0 || logical <= 0 || spare <= 0) return std::max(0, workers);
    const bool leaves_none = logical - (workers + 1) < spare + 1;
    return leaves_none && workers > spare + 1 ? workers - spare : workers;
}

/// --cpu-threads N: the lane's threads, the caller included (as maya-ds4's, and the Models page's "CPU threads") ->
/// its workers besides the caller.  0: the caller alone; below 0 as given (-1: no CPU lane).  Before #552 N was the
/// workers, so N started N + 1 threads.
inline int lane_workers_of_threads(int n) { return n >= 1 ? n - 1 : n; }

/// Decode's timing of the CPU lane (maya-ds4's lane_note, for maya-mm3's two costs): the start-up figures (an expert
/// on the lane, and each more in the same call) are one moment's, each lane alone - a busy moment staged experts
/// the CPU lane would have taken faster for the whole run (#552: "+1.27 ms each more" staged 2 of 4, decode 9.6 tok/s
/// against 12.75).  Each decoded token's CPU-lane calls (n experts, ms) go into windows of kWindow experts (the first
/// only warms up); a window's calls fitted as ms = c1 + (n - 1) add (least squares where the calls' sizes vary, else
/// the costs scaled by its time against theirs); when two windows in a row took more than 15% longer, or shorter, than
/// the costs say, the split plans with their fits' mean from then on - at most kChanges times.  A start whose figures
/// hold is never changed.
struct LaneRetime {
    static constexpr int kWindow = 96;
    static constexpr int kChanges = 4;
    static constexpr double kOff = 1.15;
    double n = 0, sx = 0, sxx = 0, st = 0, sxt = 0, model = 0;
    int experts = 0, win = 0, dir = 0, changes = 0;
    double prev_c1 = 0, prev_add = 0;

    /// One call: ne experts in ms, against the costs c1 / add (changed in place).  True when they changed.
    bool note(int ne, double ms, double& c1, double& add) {
        if (changes >= kChanges || ne <= 0 || !(ms > 0.0) || !(c1 > 0.0)) return false;
        const double x = (double) (ne - 1);
        n += 1, sx += x, sxx += x * x, st += ms, sxt += x * ms, model += c1 + x * add;
        experts += ne;
        if (experts < kWindow) return false;
        const double ratio = model > 0.0 ? st / model : 1.0;
        double fc1 = c1 * ratio, fadd = add * ratio;   // (all calls of one size: the costs scaled)
        const double mx = sx / n, var = sxx / n - mx * mx;
        if (n >= 16 && var > 0.1) {
            const double b = (sxt / n - mx * (st / n)) / var, a = st / n - b * mx;
            if (b >= 0.0 && a > 0.0) fc1 = a, fadd = b;
        }
        n = sx = sxx = st = sxt = model = 0;
        experts = 0;
        if (++win == 1) return false;
        const int d = ratio > kOff ? 1 : ratio < 1.0 / kOff ? -1 : 0;
        if (d == 0 || d != dir) {
            dir = d;
            prev_c1 = fc1, prev_add = fadd;
            return false;
        }
        c1 = 0.5 * (fc1 + prev_c1);
        add = 0.5 * (fadd + prev_add);
        dir = 0;
        ++changes;
        return true;
    }
};

}  // namespace mm3
