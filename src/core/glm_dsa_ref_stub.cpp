// src/core/glm_dsa_ref_stub.cpp - builds without a GPU backend (neither CUDA nor HIP): the glm-dsa reference forward
// (src/core/glm_dsa_ref.cu, glm-dsa B3.5) needs the glmd GPU reference kernels, so these builds refuse a glm-dsa model
// at load instead of failing to link.
#include "strata/core/glm_model.hpp"

namespace strata::core {

bool Glm5Model::dsa_ref_setup(std::string& err) {
    err = "glm-dsa: the reference forward is built for CUDA and HIP only";
    return false;
}

bool Glm5Model::step_layers_dsa(int64_t, std::string& err) {
    err = "glm-dsa: the reference forward is built for CUDA and HIP only";
    return false;
}

}  // namespace strata::core
