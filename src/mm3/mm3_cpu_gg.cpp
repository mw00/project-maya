// src/mm3/mm3_cpu_gg.cpp - maya-mm3's CPU lane for a GGUF's routed experts (mm3_cpu.hpp's gg_run): maya-ds4's CPU rows
// (src/ds4/ds4_iq_cpu.cpp - ggml-cpu's dots, or its own multi-token rows where they are their bits) on the lane's
// threads, with MiniMax's swigluoai between gate / up and down.
//
// A call: the tokens' x rounded once to each activation type the gate and up take (ggml's Q8_K for the i-quants: the
// device's experts round theirs the same way, mm3_gguf.cu), the gate and up rows in jobs of kRows rows (each
// expert's users at most kMaxN at a time, its weights read once for them), each entry's swigluoai and its h rounded to
// the down's activation type, the down rows, then each token's sum of its entries in `order`'s sequence - run()'s,
// slot by slot: a token's sum is the same bits in a window as alone.
#include "mm3_cpu.hpp"

#include "../ds4/ds4_gguf_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mm3 {

namespace dg = strata::ds4::gguf;

namespace {
constexpr int kRows = 64;   // rows a job
constexpr int kMaxN = 8;    // an expert's users a pass

// MiniMax-M3.1's swigluoai (alpha 1.702: its config's, the engine's Geo checks it), the EXL3 lane's operations
inline float oai(float g, float u, float limit) {
    g = std::min(g, limit);
    u = std::min(std::max(u, -limit), limit);
    return g / (1.0f + std::exp(-1.702f * g)) * (u + 1.0f);
}
}  // namespace

bool gg_fmt(int tg, int tu, int td, int dim, int ff, GgFmt& f, std::string& err) {
    dg::IqCpu a, b;
    if (!dg::iq_cpu_fmt(tg, td, dim, ff, a, err) || !dg::iq_cpu_fmt(tu, td, dim, ff, b, err)) return false;
    f.type[0] = tg;
    f.type[1] = tu;
    f.type[2] = td;
    f.rb[0] = a.gu_row;
    f.rb[1] = b.gu_row;
    f.rb[2] = a.d_row;
    f.act[0] = a.gu_act;
    f.act[1] = b.gu_act;
    f.act[2] = a.d_act;
    f.ab[0] = a.act_bytes;
    f.ab[1] = b.act_bytes;
    f.ab[2] = a.h_bytes;
    return true;
}

void gg_run(CpuLane& lane, int dim, int ff, float limit, const std::vector<CpuExpert>& ex, const float* x, float* out) {
    const size_t ne = std::min<size_t>(ex.size(), 8);
    std::vector<std::vector<std::pair<int, float>>> users(ne);
    std::vector<std::pair<int, int>> order;
    for (size_t e = 0; e < ne; ++e) {
        users[e].push_back({0, ex[e].w});
        order.push_back({(int) e, 0});
    }
    const std::vector<CpuExpert> head(ex.begin(), ex.begin() + (long) ne);
    gg_run_multi(lane, dim, ff, limit, head, users, x, 1, out, &order, false);
}

void gg_run_multi(CpuLane& lane, int dim, int ff, float limit, const std::vector<CpuExpert>& ex,
                  const std::vector<std::vector<std::pair<int, float>>>& users, const float* x, int T, float* out,
                  const std::vector<std::pair<int, int>>* order, bool accumulate) {
    if (!accumulate) std::fill(out, out + (size_t) T * dim, 0.0f);
    if (ex.empty() || ex[0].gf == nullptr) return;
    const GgFmt& f = *ex[0].gf;   // (one layer's experts: one format)
    struct Ent {
        int e, t;
        float w;
    };
    struct Chunk {
        int e, j0, n;
    };
    std::vector<Ent> ent;
    std::vector<Chunk> ch;
    std::vector<std::vector<int>> ent_of(ex.size());
    for (size_t e = 0; e < ex.size(); ++e)
        for (size_t a = 0; a < users[e].size(); a += kMaxN) {
            const int n = (int) std::min<size_t>(kMaxN, users[e].size() - a);
            ch.push_back({(int) e, (int) ent.size(), n});
            for (int i = 0; i < n; ++i) {
                ent_of[e].push_back((int) ent.size());
                ent.push_back({(int) e, users[e][a + (size_t) i].first, users[e][a + (size_t) i].second});
            }
        }
    const int N = (int) ent.size(), C = (int) ch.size();
    if (N == 0) return;
    // the scratch: the calling thread's (the model's host thread for its layers), reached by the workers through these
    // pointers - a thread_local named inside the jobs would be each worker's own
    thread_local std::vector<uint8_t> xg_, xu_, hq_;
    thread_local std::vector<float> gu_, yd_;
    const bool same_act = f.act[1] == f.act[0];
    xg_.resize((size_t) T * f.ab[0]);
    if (!same_act) xu_.resize((size_t) T * f.ab[1]);
    hq_.resize((size_t) N * f.ab[2]);
    gu_.resize((size_t) N * 2 * ff);
    yd_.resize((size_t) N * dim);
    uint8_t* const xg = xg_.data();
    uint8_t* const xu = same_act ? xg : xu_.data();
    uint8_t* const hq = hq_.data();
    float* const gu = gu_.data();
    float* const yd = yd_.data();
    const uint8_t* xa[2] = {xg, xu};
    std::vector<char> used((size_t) T, 0);
    for (const Ent& e : ent) used[(size_t) e.t] = 1;
    // x rounded to the gate's (and the up's) activation type, a token a job
    const int nq = same_act ? 1 : 2;
    lane.parallel_for(T * nq, [&](int j) {
        const int t = j / nq, r = j % nq;
        if (!used[(size_t) t]) return;
        dg::iq_cpu_quant(f.act[r], x + (size_t) t * dim, dim, (r == 0 ? xg : xu) + (size_t) t * f.ab[r]);
    });
    // gate / up rows: job (chunk, matrix, block of kRows rows)
    const int GB = (ff + kRows - 1) / kRows, DB = (dim + kRows - 1) / kRows;
    lane.parallel_for(C * 2 * GB, [&](int j) {
        const Chunk& c = ch[(size_t) (j / (2 * GB))];
        const int r = (j / GB) % 2, b = j % GB;
        const void* ap[kMaxN];
        float* op[kMaxN];
        for (int i = 0; i < c.n; ++i) {
            const Ent& e = ent[(size_t) (c.j0 + i)];
            ap[i] = xa[r] + (size_t) e.t * f.ab[r];
            op[i] = gu + ((size_t) (c.j0 + i) * 2 + r) * ff;
        }
        dg::iq_cpu_rows(f.type[r], ex[(size_t) c.e].gg[r], f.rb[r], dim, ap, c.n, op, b * kRows,
                        std::min(ff, (b + 1) * kRows));
    });
    // each entry's swigluoai, rounded to the down's activation type
    lane.parallel_for(N, [&](int j) {
        float* g = gu + (size_t) j * 2 * ff;
        const float* u = g + ff;
        for (int i = 0; i < ff; ++i) g[i] = oai(g[i], u[i], limit);
        dg::iq_cpu_quant(f.act[2], g, ff, hq + (size_t) j * f.ab[2]);
    });
    // down rows: job (chunk, block of kRows rows)
    lane.parallel_for(C * DB, [&](int j) {
        const Chunk& c = ch[(size_t) (j / DB)];
        const int b = j % DB;
        const void* hp[kMaxN];
        float* op[kMaxN];
        for (int i = 0; i < c.n; ++i) {
            hp[i] = hq + (size_t) (c.j0 + i) * f.ab[2];
            op[i] = yd + (size_t) (c.j0 + i) * dim;
        }
        dg::iq_cpu_rows(f.type[2], ex[(size_t) c.e].gg[2], f.rb[2], ff, hp, c.n, op, b * kRows,
                        std::min(dim, (b + 1) * kRows));
    });
    // the tokens' sums in order's sequence (each value's sum in the same order as entry by entry), a job a band
    std::vector<int> seq;
    if (order != nullptr) {
        for (const auto& [e, u] : *order) seq.push_back(ent_of[(size_t) e][(size_t) u]);
    } else {
        for (int j = 0; j < N; ++j) seq.push_back(j);
    }
    lane.parallel_for((dim + 127) / 128, [&](int b) {
        const int d0 = b * 128, d1 = std::min(dim, d0 + 128);
        for (int j : seq) {
            float* o = out + (size_t) ent[(size_t) j].t * dim;
            const float w = ent[(size_t) j].w;
            const float* y = yd + (size_t) j * dim;
            for (int d = d0; d < d1; ++d) o[d] += w * y[d];
        }
    });
}

}  // namespace mm3
