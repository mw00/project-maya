// src/ik2s/ik2_hip_lane.cpp - maya-ik2s's CPU lane (ik2_hip_lane.hpp)
#include "ik2_hip_lane.hpp"
#include "ik2_hip_situ_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ik2hip {

namespace iqc = maya::ds4::gguf;

namespace {
// IK2_HIP_LANE_PROF=1 (measurements): cpu_experts_rows' parts summed over the run, printed at the exit
struct LaneProf {
    bool on = std::getenv("IK2_HIP_LANE_PROF") != nullptr;
    long long calls = 0, ents = 0, experts = 0, rows = 0;
    double t[5] = {0, 0, 0, 0, 0};   // the inputs' int8 rows, gate + up, h's rows, down, the weighted sum
    ~LaneProf() {
        if (!on || calls == 0) return;
        const double tot = t[0] + t[1] + t[2] + t[3] + t[4];
        std::fprintf(stderr, "LANE_PROF %lld calls (layers with RAM picks), %.2f picks, %.2f experts, %.2f rows a call; us a call: "
                             "act %.1f, gate+up %.1f, h quant %.1f, down %.1f, sum %.1f, total %.1f; experts' bytes %.1f GB/s\n",
                     calls, (double) ents / calls, (double) experts / calls, (double) rows / calls, 1e6 * t[0] / calls,
                     1e6 * t[1] / calls, 1e6 * t[2] / calls, 1e6 * t[3] / calls, 1e6 * t[4] / calls, 1e6 * tot / calls,
                     bytes_per_expert * (double) experts / std::max(1e-9, t[1] + t[3]) / 1e9);
    }
    double bytes_per_expert = 0;
} g_lp;
// The int8 rows (Q8_K: a scale and sums a 256-value block, each block its own) a few blocks a work item over the lane's
// threads instead of a row a thread (IK2_HIP_LANE_OLD=1: as before; IK2_HIP_LANE_CHECK=1: each split row checked
// against the whole row's, the differing rows counted at the exit)
struct QuantCheck {
    bool on = std::getenv("IK2_HIP_LANE_CHECK") != nullptr;
    std::atomic<long long> rows{0}, bad{0};
    ~QuantCheck() {
        if (on) std::fprintf(stderr, "LANE_CHECK %lld rows split, %lld differing from the whole row's\n", rows.load(), bad.load());
    }
} g_qc;
constexpr int kQuantBlocks = 3;   // (256-value blocks a work item)
void quant_split(hy4::Lane& lane, int act, const float* const* src, void* const* dst, int nrows, int k, size_t bytes) {
    const int nbk = k / 256, nch = (nbk + kQuantBlocks - 1) / kQuantBlocks;
    const size_t bsz = bytes / (size_t) nbk;
    lane.parallel_for(nrows * nch, [&](int it) {
        const int r = it / nch, b0 = (it % nch) * kQuantBlocks, b1 = std::min(nbk, b0 + kQuantBlocks);
        iqc::iq_cpu_quant(act, src[r] + (size_t) b0 * 256, (b1 - b0) * 256, (uint8_t*) dst[r] + (size_t) b0 * bsz);
    });
    if (g_qc.on) {
        std::vector<uint8_t> whole(bytes);
        for (int r = 0; r < nrows; ++r) {
            iqc::iq_cpu_quant(act, src[r], k, whole.data());
            ++g_qc.rows;
            if (std::memcmp(whole.data(), dst[r], bytes) != 0) ++g_qc.bad;
        }
    }
}
bool quant_splits(int k, size_t bytes) {
    static const bool old = std::getenv("IK2_HIP_LANE_OLD") != nullptr;
    return !old && k % 256 == 0 && bytes % (size_t) (k / 256) == 0;
}
// A lane of several L3 groups (hy4's HY4_LANE_L3=1): each 128-row piece of an expert - a gate | up piece c, a down
// piece h_chunks + c - goes to the group piece_group gives it, the same in cpu_experts_prefetch and in the compute, so
// the compute finds what the prefetch read in its own L3 (IK2_HIP_LANE_AFFINE=0: the pieces to any thread, as before).
// Which thread does a piece never changes its arithmetic.
inline int piece_group(int slot, int c, int groups) { return (slot + c) % groups; }
bool lane_affine(const hy4::Lane& lane) {
    static const bool off = std::getenv("IK2_HIP_LANE_AFFINE") && std::atoi(std::getenv("IK2_HIP_LANE_AFFINE")) == 0;
    return !off && lane.nodes() > 1;
}
inline double lp_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int kRowChunk = 128;   // rows a work item
struct Scratch {
    std::vector<float> gu, y;
    std::vector<uint8_t> act, hq;
};
}  // namespace

[[maybe_unused]] static inline float situ(float g, float u, float beta, float lb) {
    if (lb > 0.0f) u = lb * std::tanh(u / lb);
    return beta * std::tanh(g / beta) * (1.0f / (1.0f + std::exp(-g))) * u;
}

bool CpuExperts::init(int tg, int td, int n_embd, int n_ff, std::string& err) {
    if (!iqc::iq_cpu_fmt(tg, td, n_embd, n_ff, f, err)) return false;
    iqc::iq_cpu_prepare_rows(tg);
    iqc::iq_cpu_prepare_rows(td);
    return true;
}

void cpu_experts(hy4::Lane& lane, const CpuExperts& ex, const std::vector<CpuEntry>& entries, const float* x, float* out) {
    const iqc::IqCpu& f = ex.f;
    const int E = f.n_embd, F = f.n_ff, n = (int) entries.size();
    std::fill(out, out + E, 0.0f);
    if (n == 0) return;
    static thread_local Scratch tls;
    Scratch& sc = tls;   // (the caller's: the workers use it through this reference)
    sc.gu.resize((size_t) n * 2 * F);
    sc.y.resize((size_t) n * E);
    sc.act.resize(f.act_bytes);
    sc.hq.resize((size_t) n * f.h_bytes);
    // the fused gate | up rows of every entry's expert: x quantized once (ggml's activation type for the weights')
    iqc::iq_cpu_quant(f.gu_act, x, E, sc.act.data());
    [[maybe_unused]] const int gu_chunks = (2 * F + kRowChunk - 1) / kRowChunk;
    const int dn_chunks = (E + kRowChunk - 1) / kRowChunk;
#if defined(IK2_HIP_IQ_SITU)
    const int h_chunks = (F + kRowChunk - 1) / kRowChunk;
    lane.parallel_for(n * h_chunks, [&](int it) {
        const int i = it / h_chunks, r0 = (it % h_chunks) * kRowChunk, r1 = std::min(F, r0 + kRowChunk);
        const void* act = sc.act.data();
        float* o = sc.gu.data() + (size_t) i * 2 * F;
        const uint8_t* gate = ex.gu(entries[(size_t) i].slot);
        situ_gate_up(f, ex.beta, ex.lb, gate, gate + (size_t) F * f.gu_row, &act, 1, &o, r0, r1);
    });
    lane.parallel_for(n, [&](int i) {
        iqc::iq_cpu_quant(f.d_act, sc.gu.data() + (size_t) i * 2 * F, F, sc.hq.data() + (size_t) i * f.h_bytes);
    });
#else
    lane.parallel_for(n * gu_chunks, [&](int it) {
        const int i = it / gu_chunks, r0 = (it % gu_chunks) * kRowChunk, r1 = std::min(2 * F, r0 + kRowChunk);
        const void* act = sc.act.data();
        float* o = sc.gu.data() + (size_t) i * 2 * F;
        iqc::iq_cpu_rows(f.tg, ex.gu(entries[(size_t) i].slot), f.gu_row, E, &act, 1, &o, r0, r1);
    });
    // SITU in place over the gate half, then h as the down's activation type
    lane.parallel_for(n, [&](int i) {
        float* gu = sc.gu.data() + (size_t) i * 2 * F;
        for (int r = 0; r < F; ++r) gu[r] = situ(gu[r], gu[F + r], ex.beta, ex.lb);
        iqc::iq_cpu_quant(f.d_act, gu, F, sc.hq.data() + (size_t) i * f.h_bytes);
    });
#endif
    lane.parallel_for(n * dn_chunks, [&](int it) {
        const int i = it / dn_chunks, r0 = (it % dn_chunks) * kRowChunk, r1 = std::min(E, r0 + kRowChunk);
        const void* hq = sc.hq.data() + (size_t) i * f.h_bytes;
        float* y = sc.y.data() + (size_t) i * E;
        iqc::iq_cpu_down(f, ex.dn(entries[(size_t) i].slot), &hq, 1, &y, r0, r1);
    });
    for (int i = 0; i < n; ++i) {
        const float w = entries[(size_t) i].w;
        const float* y = sc.y.data() + (size_t) i * E;
        for (int d = 0; d < E; ++d) out[d] += w * y[d];
    }
}

void cpu_experts_rows(hy4::Lane& lane, const CpuExperts& ex, const std::vector<CpuRowEntry>& entries, const float* x, int T,
                      float* out) {
    const iqc::IqCpu& f = ex.f;
    const int E = f.n_embd, F = f.n_ff, n = (int) entries.size();
    std::fill(out, out + (size_t) T * E, 0.0f);
    if (n == 0) return;
#if !defined(IK2_HIP_IQ_SITU)
    {   // (no fused SITU rows in this build: a row at a time)
        std::vector<CpuEntry> one;
        std::vector<float> o((size_t) E);
        for (int t = 0; t < T; ++t) {
            one.clear();
            for (const CpuRowEntry& e : entries)
                if (e.row == t) one.push_back({e.slot, e.w});
            if (one.empty()) continue;
            cpu_experts(lane, ex, one, x + (size_t) t * E, out + (size_t) t * E);
        }
        return;
    }
#else
    struct RowScratch {
        std::vector<float> gu, y;
        std::vector<uint8_t> act, hq;
        std::vector<int> uslot, ustart, uent, nof_gu, nof_dn;
        std::vector<const void*> ap, hp;
        std::vector<float*> gp, yp;
    };
    static thread_local RowScratch tls;
    RowScratch& sc = tls;
    sc.gu.resize((size_t) n * 2 * F);
    sc.y.resize((size_t) n * E);
    sc.act.resize((size_t) T * f.act_bytes);
    sc.hq.resize((size_t) n * f.h_bytes);
    // the experts, each with its entries (their rows ascending: the entries come row by row)
    sc.uslot.clear();
    sc.ustart.clear();
    sc.uent.clear();
    for (int i = 0; i < n; ++i) {
        bool seen = false;
        for (int u : sc.uslot) seen = seen || u == entries[(size_t) i].slot;
        if (!seen) sc.uslot.push_back(entries[(size_t) i].slot);
    }
    const int U = (int) sc.uslot.size();
    sc.ap.resize((size_t) n);
    sc.hp.resize((size_t) n);
    sc.gp.resize((size_t) n);
    sc.yp.resize((size_t) n);
    for (int u = 0; u < U; ++u) {
        sc.ustart.push_back((int) sc.uent.size());
        for (int i = 0; i < n; ++i)
            if (entries[(size_t) i].slot == sc.uslot[(size_t) u]) {
                const size_t q = sc.uent.size();
                sc.uent.push_back(i);
                sc.ap[q] = sc.act.data() + (size_t) entries[(size_t) i].row * f.act_bytes;
                sc.gp[q] = sc.gu.data() + (size_t) i * 2 * F;
                sc.hp[q] = sc.hq.data() + (size_t) i * f.h_bytes;
                sc.yp[q] = sc.y.data() + (size_t) i * E;
            }
    }
    sc.ustart.push_back((int) sc.uent.size());
    double tp = g_lp.on ? lp_now() : 0.0;
    auto lap = [&](int k) {
        if (!g_lp.on) return;
        const double t2 = lp_now();
        g_lp.t[k] += t2 - tp;
        tp = t2;
    };
    {
        bool used[64] = {};   // (at least kWin, 9 rows)
        for (const CpuRowEntry& e : entries)
            if (e.row >= 0 && e.row < 64) used[e.row] = true;
        if (quant_splits(E, f.act_bytes)) {
            const float* src[64];
            void* dst[64];
            int nr = 0;
            for (int t = 0; t < T && t < 64; ++t)
                if (used[t]) {
                    src[nr] = x + (size_t) t * E;
                    dst[nr++] = sc.act.data() + (size_t) t * f.act_bytes;
                }
            quant_split(lane, f.gu_act, src, dst, nr, E, f.act_bytes);
        } else {
            for (int t = 0; t < T && t < 64; ++t)
                if (used[t]) iqc::iq_cpu_quant(f.gu_act, x + (size_t) t * E, E, sc.act.data() + (size_t) t * f.act_bytes);
        }
        if (g_lp.on) {
            ++g_lp.calls;
            g_lp.ents += n;
            g_lp.experts += U;
            for (int t = 0; t < T && t < 64; ++t) g_lp.rows += used[t];
            g_lp.bytes_per_expert = (double) ex.blob;
        }
    }
    lap(0);
    const int h_chunks = (F + kRowChunk - 1) / kRowChunk, dn_chunks = (E + kRowChunk - 1) / kRowChunk;
    const bool aff = lane_affine(lane);
    if (aff) {
        const int G = lane.nodes();
        sc.nof_gu.resize((size_t) U * h_chunks);
        sc.nof_dn.resize((size_t) U * dn_chunks);
        for (int u = 0; u < U; ++u) {
            for (int c = 0; c < h_chunks; ++c) sc.nof_gu[(size_t) (u * h_chunks + c)] = piece_group(sc.uslot[(size_t) u], c, G);
            for (int c = 0; c < dn_chunks; ++c)
                sc.nof_dn[(size_t) (u * dn_chunks + c)] = piece_group(sc.uslot[(size_t) u], h_chunks + c, G);
        }
    }
    lane.parallel_for(U * h_chunks, [&](int it) {
        const int u = it / h_chunks, r0 = (it % h_chunks) * kRowChunk, r1 = std::min(F, r0 + kRowChunk);
        const int q0 = sc.ustart[(size_t) u], nt = sc.ustart[(size_t) u + 1] - q0;
        const uint8_t* gate = ex.gu(sc.uslot[(size_t) u]);
        situ_gate_up(f, ex.beta, ex.lb, gate, gate + (size_t) F * f.gu_row, sc.ap.data() + q0, nt, sc.gp.data() + q0, r0, r1);
    }, aff ? &sc.nof_gu : nullptr);
    lap(1);
    if (quant_splits(F, f.h_bytes) && n <= 256) {
        const float* src[256];
        void* dst[256];
        for (int i = 0; i < n; ++i) {
            src[i] = sc.gu.data() + (size_t) i * 2 * F;
            dst[i] = sc.hq.data() + (size_t) i * f.h_bytes;
        }
        quant_split(lane, f.d_act, src, dst, n, F, f.h_bytes);
    } else {
        lane.parallel_for(n, [&](int i) {
            iqc::iq_cpu_quant(f.d_act, sc.gu.data() + (size_t) i * 2 * F, F, sc.hq.data() + (size_t) i * f.h_bytes);
        });
    }
    lap(2);
    lane.parallel_for(U * dn_chunks, [&](int it) {
        const int u = it / dn_chunks, r0 = (it % dn_chunks) * kRowChunk, r1 = std::min(E, r0 + kRowChunk);
        const int q0 = sc.ustart[(size_t) u], nt = sc.ustart[(size_t) u + 1] - q0;
        iqc::iq_cpu_down(f, ex.dn(sc.uslot[(size_t) u]), sc.hp.data() + q0, nt,
                         sc.yp.data() + q0, r0, r1);
    }, aff ? &sc.nof_dn : nullptr);
    lap(3);
    for (int i = 0; i < n; ++i) {   // (a row's sum in its picks' order, as the row alone)
        const float w = entries[(size_t) i].w;
        const float* y = sc.y.data() + (size_t) i * E;
        float* o = out + (size_t) entries[(size_t) i].row * E;
        for (int d = 0; d < E; ++d) o[d] += w * y[d];
    }
    lap(4);
#endif
}

void cpu_experts_prefetch(hy4::Lane& lane, const CpuExperts& ex, const int* slots, int n, const volatile unsigned* flag,
                          unsigned v) {
    if (n <= 0) return;
    const iqc::IqCpu& f = ex.f;
    const int E = f.n_embd, F = f.n_ff;
    const int h_chunks = (F + kRowChunk - 1) / kRowChunk, dn_chunks = (E + kRowChunk - 1) / kRowChunk, per = h_chunks + dn_chunks;
    static thread_local std::vector<int> nof;
    const bool aff = lane_affine(lane);
    if (aff) {
        nof.resize((size_t) n * per);
        for (int u = 0; u < n; ++u)
            for (int c = 0; c < per; ++c) nof[(size_t) (u * per + c)] = piece_group(slots[u], c, lane.nodes());
    }
    std::atomic<bool> stop{false};
    lane.parallel_for(n * per, [&](int it) {
        const int u = it / per, c = it % per;
        auto stopped = [&] {
            if (stop.load(std::memory_order_relaxed)) return true;
            if (flag && *flag == v) {
                stop.store(true, std::memory_order_relaxed);
                return true;
            }
            return false;
        };
        auto touch = [&](const uint8_t* p, size_t bytes) {
            uint64_t acc = 0;
            for (size_t o = 0; o < bytes; o += 64) {
                if ((o & 16383) == 0 && stopped()) break;
                acc ^= *(const volatile uint64_t*) (p + o);
            }
            static thread_local volatile uint64_t sink;
            sink = acc;
        };
        if (stopped()) return;
        if (c < h_chunks) {
            const int r0 = c * kRowChunk, r1 = std::min(F, r0 + kRowChunk);
            const uint8_t* g = ex.gu(slots[u]);
            touch(g + (size_t) r0 * f.gu_row, (size_t) (r1 - r0) * f.gu_row);
            touch(g + (size_t) (F + r0) * f.gu_row, (size_t) (r1 - r0) * f.gu_row);
        } else {
            const int r0 = (c - h_chunks) * kRowChunk, r1 = std::min(E, r0 + kRowChunk);
            touch(ex.dn(slots[u]) + (size_t) r0 * f.d_row, (size_t) (r1 - r0) * f.d_row);
        }
    }, aff ? &nof : nullptr);
}

}  // namespace ik2hip
