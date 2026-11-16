// src/core/glm_ablate.hpp - refusal ablation on Maya's GLM engine (GLM-5.3-Flash's mHC and GLM-5.3's plain residual):
// a unit direction r ([n_embd], --ablate FILE: a control vector, the same files as maya-mm3's, mm3_ablate.hpp) that
// every residual write loses, y <- y - s r (r . y), on each layer's mixer output and FFN output right before it joins
// the residual (the mHC write half - gf::hc / hc_rows / hc_post / glm_hc_post / gb::hc_update - or the plain add -
// res_read / res_close / res_rows / add_rows).  Per conversation (Glm5Model::abl_on_, GEN ablate=0|1,
// --ablate-default): a step of several conversations' rows projects the runs of rows that ask, a launch a run, each
// row the same bits as one launch over all of them (a block a row, nothing shared between rows).  Nothing projects:
// no launch, the path as without --ablate.  The draft heads (the NextN / MTP block, DFlash) propose only and are left
// as they are: the main layers' verify projects, so the kept tokens are the ablated model's.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core::glmabl {

/// The runs [first, end) of consecutive rows whose on[i] is set, in row order (none: empty - no launch).
inline std::vector<std::pair<int, int>> runs(const uint8_t* on, int n) {
    std::vector<std::pair<int, int>> out;
    for (int i = 0; i < n;) {
        if (on[i] == 0) { ++i; continue; }
        int j = i + 1;
        while (j < n && on[j] != 0) ++j;
        out.emplace_back(i, j);
        i = j;
    }
    return out;
}

/// GPU (src/core/glm_ablate.cu): rows y[0 .. T) of a device buffer (row stride dim), each y -= scale r (r . y); a
/// 256-thread block a row, a fixed-order tree sum, so a row's bits do not depend on T or on the other rows.
/// `stream`: a cudaStream_t.  T <= 0: nothing.
void project_out(float* y, const float* r, float scale, int dim, int T, void* stream);

/// The CPU reference of one row: y -= s r (r . y), the dot accumulated in double (tests).
inline void project_row_ref(float* y, const float* r, float s, int dim) {
    double d = 0.0;
    for (int e = 0; e < dim; ++e) d += (double) r[e] * (double) y[e];
    for (int e = 0; e < dim; ++e) y[e] = (float) ((double) y[e] - (double) s * d * (double) r[e]);
}

}  // namespace strata::core::glmabl
