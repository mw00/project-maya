// src/prefill/moe_mmq_strata.cu - glm-dsa's view of the vendored MMQ (B9.8, B14.1, B14.2); see
// include/strata/prefill/moe_mmq_strata.hpp.  Compiled into strata_mmq (cmake/strata.d/50-glm-dsa-prefill.cmake)
// with moe_mmq.cu's include paths and flags, so the configs below are the ones the instances were built from.
#include "strata/prefill/moe_mmq_strata.hpp"

#include "common.cuh"
#include "mmq.cuh"

#include <climits>

namespace strata::prefill::mmq_strata {

int block_bytes() { return (int) sizeof(block_q8_1_mmq); }

int64_t blocks(int64_t rows, int64_t cols) { return rows * ((cols + 511) / 512 * 512) / (4 * QK8_1); }

int ds_halves(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_Q8_0: case GGML_TYPE_IQ1_S:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
            return mmq_get_q8_1_ds_layout((ggml_type) t) == MMQ_Q8_1_DS_LAYOUT_D4 ? 0 : 8;
        default:
            return 0;
    }
}

static bool same(const ggml_cuda_mmq_config& a, const ggml_cuda_mmq_config& b) {
    return a.type == b.type && a.J == b.J && a.I == b.I && a.nthreads == b.nthreads;
}

bool sm70_check(int t, int cc, int64_t smpbo, int max_cols, std::string& why) {
#if defined(GGML_USE_HIP)
    // (HIP's strata_mmq builds this file for the B14.1 layout above; the sm_70 instance rule is CUDA's, and the HIP
    // prompt path never asks it - glmpf::mmq_ok)
    (void) t;
    (void) cc;
    (void) smpbo;
    (void) max_cols;
    why = "the sm_70 MMQ instance check is CUDA's (a HIP build)";
    return false;
#else
    const ggml_type type = (ggml_type) t;
    // the device side of an sm_70 instance: __CUDA_ARCH__ 700 >= VOLTA selects the ampere table
    // (ggml_cuda_mmq_get_config's device overload); the host side goes through the build's compiled arch list
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_VOLTA) {
        why = "the build has no sm_70 code for cc " + std::to_string(cc) + " (highest compiled arch " +
              std::to_string(ggml_cuda_highest_compiled_arch(cc)) + ")";
        return false;
    }
    for (int fb = 0; fb < 2; ++fb) {
        const bool fallback = fb == 1;
        for (int J = 8; J <= 128; J += 8) {
            const ggml_cuda_mmq_config host = ggml_cuda_mmq_get_config(type, J, fallback, cc);
            if (host.type == GGML_TYPE_COUNT) continue;
            const ggml_cuda_mmq_config dev = ggml_cuda_mmq_get_config_ampere(type, J, fallback);
            if (!same(host, dev)) {
                why = "J " + std::to_string(J) + (fallback ? " fallback" : "") +
                      ": the host config is not the sm_70 device config (NO_DEVICE_CODE on V100)";
                return false;
            }
        }
        // the host switch over every column count: some J fits, and its instance is real
        for (int ncols = 1; ncols <= max_cols; ncols = ncols < 256 ? ncols + 1 : ncols + 37) {
            int J_best = 0, ntiles_best = INT_MAX;
            for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
                const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(type, J, fallback, cc);
                if (c.type == GGML_TYPE_COUNT || (int64_t) mmq_get_nbytes_shared(c, cc) > smpbo) continue;
                const int nt = (ncols + c.J - 1) / c.J;
                if (nt < ntiles_best) {
                    J_best = J;
                    ntiles_best = nt;
                }
            }
            if (J_best == 0) {
                why = "no tile width fits " + std::to_string(ncols) + " columns" + (fallback ? " (fallback)" : "");
                return false;
            }
        }
    }
    return true;
#endif
}

}  // namespace strata::prefill::mmq_strata
