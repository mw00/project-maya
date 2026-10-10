#pragma once

// HIP-only launch controls for comparing decode expert kernels in one process.
// Baseline always launches the original kernels; production uses normal dispatch.
// Other values bypass dispatch. CUDA does not compile these variants.
#include "strata/kernels/glm_fast.hpp"

#if defined(STRATA_USE_HIP)
namespace strata::kernels::glmf::hip_expert_bench {
enum class Variant {
    baseline, production, signs, waves8, rows2, rows8, global_grid, global_plain,
    global_waves8, global_rows2, global_rows8, split_shared, direct_signs,
    vector_load, occupancy8, direct_lds, direct_rows4, direct_lds_rows4
};
void gate_up(Variant variant, int type, const MoeDev& d, int k, int n_embd, int n_ff,
             float limit, const void* xq, void* hq, const void* sh_down, int sh_type,
             const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t stream);
void down(Variant variant, int type, const MoeDev& d, int k, int n_embd, int n_ff,
          size_t down_off, const void* hq, const float* sh_out, float* out, cudaStream_t stream);
}
#endif

#if defined(STRATA_USE_HIP)
namespace strata::kernels::glmf::hip_expert_bench {
// The RDNA3 dense GEMV (mv / mv_rows on gfx11): r rows per wave, u super-blocks in flight, wpb waves per block,
// lds > 0 the Q6_K activations staged in LDS, lds < 0 the default's choice.  r == 0 launches the original kernels,
// r < 0 the default shape.
void mv_config(int r, int u, int wpb, int lds);
}
#endif
