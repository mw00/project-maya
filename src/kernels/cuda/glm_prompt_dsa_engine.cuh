// src/kernels/cuda/glm_prompt_dsa_engine.cuh - dsa_indexer: glm_prompt_dsa.cuh for glm_prefill.cu, one plan per
// (device, stream). Included inside glm_batch.cu's namespace, after dsa_score / dsa_select. Same cells and n_sel.
// MAYA_GLM_PROMPT_DSA=0: the pair. MAYA_GLM_PROMPT_DSA_CHECK=n: the first n calls run both on the same inputs and
// compare cells.
#pragma once

namespace pdsa_engine {

constexpr int kGrowPools = 32768;   // the plan grows in steps of 128K positions (a 1M prompt: <= 8 regrowths)

struct Slot {
    int dev = -1;
    cudaStream_t s = nullptr;
    pdsa::Plan pl{};
    bool ready = false;
};

inline Slot* slot(cudaStream_t s) {
    static Slot tab[16];
    static std::atomic_flag lock = ATOMIC_FLAG_INIT;
    int dev = 0;
    cudaGetDevice(&dev);
    while (lock.test_and_set(std::memory_order_acquire)) {
    }
    Slot* x = nullptr;
    for (Slot& t : tab)
        if (t.dev == dev && t.s == s) { x = &t; break; }
    if (x == nullptr)
        for (Slot& t : tab)
            if (t.dev < 0) {
                t.dev = dev;
                t.s = s;
                x = &t;
                break;
            }
    lock.clear(std::memory_order_release);
    return x;
}

// a plan holding max_vis pools and T tokens (grown on demand); false: no memory for it (the pair runs)
inline bool ensure(Slot* x, int max_vis, int T) {
    if (x->ready && max_vis <= x->pl.max_pools && T <= x->pl.max_T) return true;
    const int pools = std::max((max_vis + kGrowPools - 1) / kGrowPools * kGrowPools, x->ready ? x->pl.max_pools : 0);
    const int toks = std::max(T, x->ready ? x->pl.max_T : 0);
    if (x->ready) {
        cudaStreamSynchronize(x->s);
        pdsa::release(x->pl);
        x->ready = false;
    }
    if (pdsa::init(x->pl, pools, toks) != cudaSuccess) {
        cudaGetLastError();
        pdsa::release(x->pl);
        static std::atomic<bool> told{false};
        if (!told.exchange(true))
            std::fprintf(stderr, "glm prefill: prompt DSA indexer: no VRAM for a %d-pool plan; dsa_score + dsa_select stay\n", pools);
        return false;
    }
    x->ready = true;
    return true;
}

struct Check {
    long left = 0, calls = 0, rows = 0, rows_eq = 0;
    int* cells = nullptr;   // the pair's cells / n_sel of the call being checked
    int* n_sel = nullptr;
    size_t cap = 0;
};
inline Check& check_state() {
    static Check c = [] {
        Check k;
        if (const char* v = std::getenv("MAYA_GLM_PROMPT_DSA_CHECK")) k.left = std::atol(v);
        return k;
    }();
    return c;
}

}   // namespace pdsa_engine

bool dsa_indexer(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int p0,
                     int kpool, int T, int max_vis, float* score, int score_ld, int top_pools_max, int tail,
                     int n_sel_max, int* cells, int* n_sel, cudaStream_t s) {
    namespace oe = pdsa_engine;
    static const bool on = [] {
        const char* e = std::getenv("MAYA_GLM_PROMPT_DSA");
        if (e != nullptr && e[0] == '0' && e[1] == 0) return false;
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        if (major < 8) return false;
        std::fprintf(stderr, "glm prefill: prompt DSA indexer in one pass (MAYA_GLM_PROMPT_DSA)%s\n",
                     oe::check_state().left > 0 ? ", checking its cells against dsa_score + dsa_select" : "");
        return true;
    }();
    if (!on || T <= 0 || key_dim != pdsa::kKey || idx_heads != pdsa::kHeads ||
        top_pools_max > pdsa::kMaxTop)
        return false;
    oe::Slot* x = oe::slot(s);
    if (x == nullptr || !oe::ensure(x, std::max(max_vis, 1), T)) return false;
    oe::Check& ck = oe::check_state();
    const bool chk = ck.left > 0;
    if (chk) {   // the pair first, its cells and n_sel kept aside
        const size_t need = (size_t) T * n_sel_max;
        if (ck.cap < need) {
            cudaFree(ck.cells);
            cudaFree(ck.n_sel);
            ck.cap = 0;
            if (cudaMalloc(&ck.cells, need * sizeof(int)) != cudaSuccess ||
                cudaMalloc(&ck.n_sel, (size_t) std::max(T, 1024) * sizeof(int)) != cudaSuccess) {
                cudaGetLastError();
                ck.left = 0;
                return false;
            }
            ck.cap = need;
        }
        if (max_vis > top_pools_max)
            dsa_score(iq, pooled, iw, key_dim, idx_heads, p0, kpool, T, max_vis, score, score_ld, s);
        dsa_select(score, score_ld, p0, kpool, top_pools_max, tail, T, n_sel_max, cells, n_sel, s);
        cudaMemcpyAsync(ck.cells, cells, need * sizeof(int), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(ck.n_sel, n_sel, (size_t) T * sizeof(int), cudaMemcpyDeviceToDevice, s);
    }
    if (pdsa::score_select(x->pl, iq, pooled, nullptr, iw, key_dim, idx_heads, p0, kpool, T, max_vis, score,
                                score_ld, top_pools_max, tail, n_sel_max, cells, n_sel, s) != cudaSuccess) {
        cudaGetLastError();   // not sticky: the caller runs the pair (it overwrites anything written)
        return false;
    }
    if (chk) {
        std::vector<int> a((size_t) T * n_sel_max), b(a.size()), na(T), nb(T);
        cudaStreamSynchronize(s);
        cudaMemcpy(a.data(), ck.cells, a.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), cells, b.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(na.data(), ck.n_sel, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(nb.data(), n_sel, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost);
        ++ck.calls;
        for (int t = 0; t < T; ++t) {
            bool eq = na[t] == nb[t];
            for (int i = 0; eq && i < na[t] && i < n_sel_max; ++i)
                eq = a[(size_t) t * n_sel_max + i] == b[(size_t) t * n_sel_max + i];
            ++ck.rows;
            ck.rows_eq += eq;
            if (!eq && ck.rows - ck.rows_eq == 1)
                std::fprintf(stderr, "glm prefill: prompt DSA check: first differing token: call %ld token %d (p0 %d)\n",
                             ck.calls, t, p0);
        }
        if (--ck.left == 0)
            std::fprintf(stderr, "glm prefill: prompt DSA check over %ld calls: tokens with identical cells %ld / %ld\n",
                         ck.calls, ck.rows_eq, ck.rows);
    }
    return true;
}
