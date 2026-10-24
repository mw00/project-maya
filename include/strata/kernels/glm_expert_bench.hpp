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
// The RDNA3 dense GEMV (mv on gfx11, by default the gfx115x APUs): r rows per wave, u super-blocks in flight, wpb waves per block,
// lds > 0 the Q6_K activations staged in LDS, lds < 0 the default's choice.  r == 0 launches the original kernels,
// r < 0 the default shape.
void mv_config(int r, int u, int wpb, int lds);
// glm_attn_bench: the decode attention's take-two kernels by label (glm_fast.cu - dsa_score2 [r|s][l|x][1|2]b<2|4|8|16>,
// mla_split2 h<16|32|64>[l|x][v|e][c|o]); "original" the original kernels; nullptr the engine's env again.  false: no label
bool dsa_score_config(const char* label);
bool mla_attn_config(const char* label);
// 1: this device's code reduces across the VALU's lanes (the RDNA3 DPP butterfly), 0: through the shuffles, -1: unknown
int lanes_kind(cudaStream_t s);
// every lane's warp_sum, warp_sum_lanes, warp_max and warp_max_lanes of n_warps x 32 inputs (device pointers)
void warp_reduce_check(const float* in, float* sum_shfl, float* sum_lanes, float* max_shfl, float* max_lanes, int n_warps,
                       cudaStream_t s);
}
#endif
