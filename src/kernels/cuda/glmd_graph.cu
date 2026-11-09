// src/kernels/cuda/glmd_graph.cu - glm-dsa B10.1: the device position block of a captured glm-dsa decode part.
#include "strata/core/glm_env.hpp"
#include "strata/kernels/glmd_graph.hpp"

#include <cuda_runtime.h>

#include <cstdlib>

namespace glmd {
namespace {
__global__ void set_dev_pos_kernel(DevPos* dev, int p, int top_k) {
    DevPos v;
    v.p = p;
    v.n_kv = p + 1;
    v.n_sel = p + 1 < top_k ? p + 1 : top_k;
    v.pad = 0;
    *dev = v;
}
}  // namespace

bool set_dev_pos(DevPos* dev, int p, int top_k, void* stream) {
    if (!dev || p < 0 || top_k <= 0) return false;
    set_dev_pos_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(dev, p, top_k);
    return cudaGetLastError() == cudaSuccess;
}

bool dsa_graphs_enabled() {
    static const bool on = [] {
        return strata::glmenv::env_flag("STRATA_GLM_DSA_GRAPHS", false);
    }();
    return on;
}

}  // namespace glmd
