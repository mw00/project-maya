// src/core/glm_dsa_prefill_share.cpp - the glm-dsa port B9.9: the prompt's host share on a node-split layer, one
// node's arithmetic (see glm_dsa_prefill_stream.hpp).  Plain C++ over ggml-cpu (strata_kernels_cpu); the node lanes
// that run it are driven by glm_dsa_prefill_stream.cpp.
//
// Maya's share (glm_prefill.cu, the CPU's share of a layer's staged experts) computes, for every routed row of each
// expert it takes, gate/up over the whole n_ff, the down activation, then down over n_embd from the contiguous blob.
// On a node-split layer (glm_tier_plan.hpp SplitLayout) node h holds gate/up rows [h*H, (h+1)*H) and the down rows'
// inner columns [h*H, (h+1)*H): it computes its H gate/up values per row, quantizes them as the half format (the
// blocks never straddle the halves), and dots each down row's half against them into a partial output; the two
// partials are added.  Gate/up values are the unsplit lane's bit for bit; the down output differs by the order of one
// float sum.
#include "glm_dsa_prefill_stream.hpp"

#include <algorithm>

namespace strata::core::glmpfs {

namespace {
constexpr int kGu = 32;   // gate/up rows a job (Maya's share: kGu)
constexpr int kDn = 64;   // down rows a job (Maya's share: kDn)
}  // namespace

void share_node(const kc::NativeFmt& hf, const glmtier::SplitLayout& L, int n_exp, const ShareX* ex, size_t half_off,
                const uint8_t* act, int rows, float swiglu_limit, ShareScratch& sc, const RunFn& run) {
    const int H = (int) L.H, E = (int) L.n_embd;
    constexpr size_t kA = kc::kNativeActBytes, kH = kc::kNativeHBytes;
    if (n_exp <= 0 || rows <= 0) return;
    if (sc.ff.size() < (size_t) rows * H) sc.ff.resize((size_t) rows * H);
    if (sc.hq.size() < (size_t) rows * kH) sc.hq.resize((size_t) rows * kH);
    if (sc.part.size() < (size_t) rows * E) sc.part.resize((size_t) rows * E);
    float* ff = sc.ff.data();
    uint8_t* hq = sc.hq.data();
    float* part = sc.part.data();
    // gate / up: rows [r0, r1) of the half for every routed row of expert j
    run(share_jobs(n_exp, H, kGu), [&](int job) {
        int j, r0, r1;
        share_job(job, n_exp, kGu, H, j, r0, r1);
        const ShareX& x = ex[j];
        if (x.rows <= 0) return;
        thread_local std::vector<const void*> a;
        thread_local std::vector<float*> o;
        a.resize((size_t) x.rows);
        o.resize((size_t) x.rows);
        for (int t = 0; t < x.rows; ++t) {
            a[(size_t) t] = act + (size_t) (x.r0 + t) * kA;
            o[(size_t) t] = ff + (size_t) (x.r0 + t) * H;
        }
        const uint8_t* blob = x.half0 + half_off;
        kc::native_gu_rows_split(hf, blob, blob + L.h_gu, a.data(), x.rows, o.data(), r0, r1, swiglu_limit);
    });
    // the half's down activation, one per routed row
    run(rows, [&](int r) { kc::native_quant_h(hf, ff + (size_t) r * H, hq + (size_t) r * kH); });
    // down: output rows [r0, r1) from the half's inner columns
    run(share_jobs(n_exp, E, kDn), [&](int job) {
        int j, r0, r1;
        share_job(job, n_exp, kDn, E, j, r0, r1);
        const ShareX& x = ex[j];
        if (x.rows <= 0) return;
        thread_local std::vector<const void*> h;
        thread_local std::vector<float*> o;
        h.resize((size_t) x.rows);
        o.resize((size_t) x.rows);
        for (int t = 0; t < x.rows; ++t) {
            h[(size_t) t] = hq + (size_t) (x.r0 + t) * kH;
            o[(size_t) t] = part + (size_t) (x.r0 + t) * E;
        }
        kc::native_down_rows_split(hf, x.half0 + half_off + L.h_down_off, h.data(), x.rows, o.data(), r0, r1);
    });
}

void share_add(const float* p0, const float* p1, float* out, size_t n, const RunFn& run, int rows) {
    if (rows <= 0 || n == 0) return;
    const size_t per = n / (size_t) rows;
    run(rows, [&](int r) {
        const size_t b = (size_t) r * per;
        for (size_t i = 0; i < per; ++i) out[b + i] = p0[b + i] + p1[b + i];
    });
}

}  // namespace strata::core::glmpfs
