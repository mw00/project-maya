// src/inkling2/inkling2_lane.cpp - maya-inkling2's CPU lane (inkling2_lane.hpp)
#include "inkling2_lane.hpp"

#include <algorithm>
#include <cmath>

namespace inkling2 {

namespace iqc = strata::ds4::gguf;

namespace {
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
#if defined(INKLING2_IQ_SITU)
    f.act = iqc::kIqActSitu;   // (ds4_iq_cpu's own SITU epilogue over the fused rows)
    f.situ_beta = beta;
    f.situ_lb = lb;
#endif
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
#if defined(INKLING2_IQ_SITU)
    const int h_chunks = (F + kRowChunk - 1) / kRowChunk;
    lane.parallel_for(n * h_chunks, [&](int it) {
        const int i = it / h_chunks, r0 = (it % h_chunks) * kRowChunk, r1 = std::min(F, r0 + kRowChunk);
        const void* act = sc.act.data();
        float* o = sc.gu.data() + (size_t) i * 2 * F;
        const uint8_t* gate = ex.base + (size_t) entries[(size_t) i].slot * ex.blob;
        iqc::iq_cpu_gate_up(f, gate, gate + (size_t) F * f.gu_row, &act, 1, &o, r0, r1, 0.0f);
    });
    lane.parallel_for(n, [&](int i) {
        iqc::iq_cpu_quant(f.d_act, sc.gu.data() + (size_t) i * 2 * F, F, sc.hq.data() + (size_t) i * f.h_bytes);
    });
#else
    lane.parallel_for(n * gu_chunks, [&](int it) {
        const int i = it / gu_chunks, r0 = (it % gu_chunks) * kRowChunk, r1 = std::min(2 * F, r0 + kRowChunk);
        const void* act = sc.act.data();
        float* o = sc.gu.data() + (size_t) i * 2 * F;
        iqc::iq_cpu_rows(f.tg, ex.base + (size_t) entries[(size_t) i].slot * ex.blob, f.gu_row, E, &act, 1, &o, r0, r1);
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
        iqc::iq_cpu_down(f, ex.base + (size_t) entries[(size_t) i].slot * ex.blob + ex.tr_down, &hq, 1, &y, r0, r1);
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
#if !defined(INKLING2_IQ_SITU)
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
        std::vector<int> uslot, ustart, uent;
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
    {
        bool used[64] = {};   // (at least kWin, 9 rows)
        for (const CpuRowEntry& e : entries)
            if (e.row >= 0 && e.row < 64) used[e.row] = true;
        for (int t = 0; t < T && t < 64; ++t)
            if (used[t]) iqc::iq_cpu_quant(f.gu_act, x + (size_t) t * E, E, sc.act.data() + (size_t) t * f.act_bytes);
    }
    const int h_chunks = (F + kRowChunk - 1) / kRowChunk, dn_chunks = (E + kRowChunk - 1) / kRowChunk;
    lane.parallel_for(U * h_chunks, [&](int it) {
        const int u = it / h_chunks, r0 = (it % h_chunks) * kRowChunk, r1 = std::min(F, r0 + kRowChunk);
        const int q0 = sc.ustart[(size_t) u], nt = sc.ustart[(size_t) u + 1] - q0;
        const uint8_t* gate = ex.base + (size_t) sc.uslot[(size_t) u] * ex.blob;
        iqc::iq_cpu_gate_up(f, gate, gate + (size_t) F * f.gu_row, sc.ap.data() + q0, nt, sc.gp.data() + q0, r0, r1, 0.0f);
    });
    lane.parallel_for(n, [&](int i) {
        iqc::iq_cpu_quant(f.d_act, sc.gu.data() + (size_t) i * 2 * F, F, sc.hq.data() + (size_t) i * f.h_bytes);
    });
    lane.parallel_for(U * dn_chunks, [&](int it) {
        const int u = it / dn_chunks, r0 = (it % dn_chunks) * kRowChunk, r1 = std::min(E, r0 + kRowChunk);
        const int q0 = sc.ustart[(size_t) u], nt = sc.ustart[(size_t) u + 1] - q0;
        iqc::iq_cpu_down(f, ex.base + (size_t) sc.uslot[(size_t) u] * ex.blob + ex.tr_down, sc.hp.data() + q0, nt,
                         sc.yp.data() + q0, r0, r1);
    });
    for (int i = 0; i < n; ++i) {   // (a row's sum in its picks' order, as the row alone)
        const float w = entries[(size_t) i].w;
        const float* y = sc.y.data() + (size_t) i * E;
        float* o = out + (size_t) entries[(size_t) i].row * E;
        for (int d = 0; d < E; ++d) o[d] += w * y[d];
    }
#endif
}


}  // namespace inkling2
