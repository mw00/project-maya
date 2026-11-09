// src/core/glm_req_knobs.cpp - glm-dsa (speed plan, code change 2): a request's decode knobs, applied between
// requests (include/strata/core/glm_req_knobs.hpp), and the speculative head's own CPU-lane split.  generate.cpp's GEN
// line calls apply() after set_pcie_share / set_cpu_lane_threads / set_read_chunks (one-line hooks, STRATA-EDITS.txt).
#include "strata/core/glm_req_knobs.hpp"

#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glmd_graph.hpp"
#if defined(STRATA_NATIVE_EXPERTS)   // (strata_kernels_cpu's PUBLIC define: the row4 lane exists only with ggml-cpu)
#include "strata/kernels/cpu/glm_lane_row4.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <string>

namespace strata::core {

namespace {

// set_pcie_share's plan for a share s in [0, 1]: of f RAM-tier experts in a route, f - round(s f) go to the CPU
unsigned long long share_plan(double s) {
    s = std::min(1.0, std::max(0.0, s));
    unsigned long long plan = 0;
    for (int f = 0; f <= 8; ++f) plan |= (unsigned long long) (f - (int) std::lround(s * f)) << (4 * f);
    return plan;
}

std::atomic<bool> g_named{false};   // the last request named a knob (the next one without says they are back)
std::atomic<int> g_row4{-1};        // the lane_row4 value set last (-1: the switch, as the engine started)

}  // namespace

// The head: the parts the pipelined speculative decode runs ahead (spec_head_last, as it stands when the request
// starts; the head tuner may move it later), else the first part.  Each part has its own plan, so the head can pull
// more over its idle PCIe link while the tail keeps the shared CPU pool.  One part: nothing to split, false.
bool Glm5Model::set_pcie_share_head(double share) {
    if (share < 0.0 || split_next_ == nullptr) return false;   // (set_pcie_share has set every part already)
    Glm5Model* last = spec_ready() ? spec_head_last() : this;
    const unsigned long long plan = share_plan(share);
    bool any = false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        if (F != nullptr && F->cpu_pool != nullptr) {
            F->cpu_plan = plan;   // (the next route reads it, as set_pcie_share's)
            any = true;
        }
        if (m == last) break;
    }
    return any;
}

namespace glmknob {

void apply(Glm5Model& model, const Request& r) {
    // promote_min is an argument of the route kernel: a decode graph captured with the old value must not replay
    const bool regraph = raw(PromoteMin) != r.value(PromoteMin);
    for (int k = 0; k < kCount; ++k) set((Id) k, r.value((Id) k));
    if (regraph) ::glmd::graph_generation_bump("request knob promote_min");
    // the lane's row4 switch, only on a change (a request without it calls nothing): lane_row4_override(-1) is
    // STRATA_GLM_LANE_ROW4 as the engine started, 0 off, 1 on
    if (g_row4.exchange(r.lane_row4) != r.lane_row4) {
#if defined(STRATA_NATIVE_EXPERTS)
        strata::kernels::cpu::lane_row4_override(r.lane_row4);
        if (r.lane_row4 == 1 && !strata::kernels::cpu::lane_row4_on())   // (its kernels are x86-64 only)
            std::fprintf(stderr, "glm knobs: lane_row4 ignored (this build has no 4-row lane)\n");
#else
        if (r.lane_row4 >= 0) std::fprintf(stderr, "glm knobs: lane_row4 ignored (no native experts)\n");
#endif
    }
    if (r.pcie_frac_head >= 0.0 && !model.set_pcie_share_head(r.pcie_frac_head))
        std::fprintf(stderr, "glm knobs: pcie_frac_head ignored (one part, or no CPU lane)\n");
    const bool named = r.named();
    const bool was = g_named.exchange(named);
    if (named) std::fprintf(stderr, "glm knobs: this request %s\n", r.line().c_str());
    else if (was) std::fprintf(stderr, "glm knobs: back to the start values\n");
}

}  // namespace glmknob

}  // namespace strata::core
