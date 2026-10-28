// src/ds4/ds4_multi_attn.hpp - maya-ds4 --parallel, DS4_MULTI_ATTN: a multi step's attention for all its rows in one
// launch a step instead of one a conversation.  A run is one conversation's consecutive rows (a decoded token, a verify
// window, a drafter block); attn_core gives each run the window buffer alone (its ring's win rows, then its own rows),
// here every run has its block of one shared buffer, side by side, so one launch serves all the rows, each reading its
// own run's block.  The layout and the launches' spans alone, no model and no GPU (tests/ds4_multi_attn_plan_test.cpp).
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::ds4 {

constexpr int kMaRows = 16;   // a multi step's rows at most (kWin)

struct MaRun {
    int seq = 0, r0 = 0, n = 0;   // the conversation, its first row, its rows
    int64_t p0 = 0;               // its first row's position (row r0 + i at p0 + i)
    int e0 = 0;                   // its block's first row in the shared buffer: win ring rows, then its n rows
};

struct MaPlan {
    int nrun = 0, T = 0, rows = 0;   // the runs, the step's rows, the shared buffer's rows in use
    MaRun run[kMaRows];
    int row_run[kMaRows] = {};   // row t's run
    int pos[kMaRows] = {};       // row t's position: its run's p0 + its index in the run (attn_core's p0 + t)
    int dst[kMaRows] = {};       // row t's own KV row in the shared buffer (its block's row win + its index)
    // row t's window rows in the shared buffer (sparse_attn's AttnRows win0 / nwin): attn_core's for the row, moved to
    // its run's block
    int win0[kMaRows] = {}, nwin[kMaRows] = {};
};

// the plan of a step's T rows - row t conversation rseq[t] at position rpos[t], a run each conversation's consecutive
// rows - for a window of `win` rows.  A drafter block's rows (mtp) see the ring's rows as they lie (the first
// min(win, p0)) and their block through the selection.  false: T not 1 .. kMaRows, win < 1, or a conversation in two
// runs (its rows apart: one run's ring writes would then come before the other's reads)
inline bool ma_plan(const int* rseq, const int64_t* rpos, int T, int win, bool mtp, MaPlan& p) {
    p = MaPlan{};
    if (T < 1 || T > kMaRows || win < 1) return false;
    p.T = T;
    int e = 0;
    for (int t = 0; t < T;) {
        int u = t + 1;
        while (u < T && rseq[u] == rseq[t]) ++u;
        for (int k = 0; k < p.nrun; ++k)
            if (p.run[k].seq == rseq[t]) return false;
        MaRun& r = p.run[p.nrun];
        r.seq = rseq[t];
        r.r0 = t;
        r.n = u - t;
        r.p0 = rpos[t];
        r.e0 = e;
        for (int i = 0; i < r.n; ++i) {
            const int row = t + i;
            p.row_run[row] = p.nrun;
            p.pos[row] = (int) (r.p0 + i);
            p.dst[row] = e + win + i;
            if (mtp) {
                p.win0[row] = e;
                p.nwin[row] = (int) std::min<int64_t>(win, r.p0);
            } else {   // (attn_core: ext row of position max(0, p - win + 1), up to the row's own)
                const int first = (int) std::max<int64_t>(win - r.p0, i + 1);
                p.win0[row] = e + first;
                p.nwin[row] = win + i - first + 1;
            }
        }
        e += win + r.n;
        ++p.nrun;
        t = u;
    }
    p.rows = e;
    return true;
}

// the shared buffer's rows a step of up to kMaRows rows can take (each row a run of its own)
constexpr int ma_rows_max(int win) { return kMaRows * (win + 1); }

// a sparse-attention launch's runs: run0 .. run0 + nrun - 1, rows r0 .. r0 + n - 1, all of class cls
struct MaSpan {
    int run0 = 0, nrun = 0, r0 = 0, n = 0, cls = 0;
};

// the step's runs in launches: consecutive runs of the same class (cls[k], run k's) together while their rows fit in
// max_rows; class 0 (the kernels attn_core runs for that run take no shared launch), and a run longer than max_rows,
// a launch a run (class 0).  Returns the spans (at most p.nrun)
inline int ma_spans(const MaPlan& p, const int* cls, int max_rows, MaSpan* out) {
    int ns = 0;
    for (int k = 0; k < p.nrun; ++k) {
        const MaRun& r = p.run[k];
        if (ns > 0) {
            MaSpan& b = out[ns - 1];
            if (cls[k] != 0 && b.cls == cls[k] && b.n + r.n <= max_rows) {
                ++b.nrun;
                b.n += r.n;
                continue;
            }
        }
        MaSpan& s = out[ns++];
        s.run0 = k;
        s.nrun = 1;
        s.r0 = r.r0;
        s.n = r.n;
        s.cls = r.n <= max_rows ? cls[k] : 0;
    }
    return ns;
}

}  // namespace strata::ds4
