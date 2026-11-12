// src/core/glm_dsa_mixer_stub.cpp - builds without a GPU backend (neither CUDA nor HIP): the glm-dsa decode mixer
// (src/core/glm_dsa_mixer.cu, glm-dsa B4.6) needs the glmd GPU kernels, so these builds refuse a glm-dsa fast path at
// setup instead of failing to link.
#include "glm_dsa_fast.hpp"

namespace strata::core {

namespace glmdsa {
bool mixer_setup(GlmDsaFast&, const Glm5Geometry&, int64_t, std::string& err) {
    err = "glm-dsa: the fast decode mixer is built for CUDA and HIP only";
    return false;
}
// B10.1: no decode graphs without the GPU mixer (fast_layers_plain then runs its loop directly)
bool graph_prepare(GlmDsaFast&, const GlmStateLayout&, int, int, int64_t, unsigned long long, const void*, const void*,
                   const char*, uint64_t, cudaStream_t, PartGraph& pg, std::string&, int) {
    pg = PartGraph{};
    return true;
}
bool graph_run(GlmDsaFast&, PartGraph&, int64_t, cudaStream_t, uint64_t&, const std::function<bool()>& body,
               std::string&) {
    return body();
}
void graph_stats(const GlmDsaFast&, uint64_t& c, uint64_t& r, uint64_t& i) { c = r = i = 0; }
void mixer_rollback(GlmDsaFast& X) {
    X.n_sel_last = 0;
    X.sel_layer = -1;
    X.sel_pos = -1;
}
bool win_rows_begin(GlmDsaFast&, int64_t, int, int, cudaStream_t, std::string& err) {
    err = "glm-dsa: the fast decode mixer is built for CUDA only";
    return false;
}
}  // namespace glmdsa

bool Glm5Model::fast_dsa_glm(int, int64_t, std::string& err) {
    err = "glm-dsa: the fast decode mixer is built for CUDA and HIP only";
    return false;
}

bool Glm5Model::fast_dsa_glm_window(int, int64_t, int, int, const float*, const void*, size_t, float*,
                                    std::string& err) {
    err = "glm-dsa: the fast decode mixer is built for CUDA only";
    return false;
}

int Glm5Model::dsa_n_sel_last() const { return -1; }

bool Glm5Model::prefill_tokens_dsa(const std::vector<int32_t>&, int32_t, std::string&) { return false; }

}  // namespace strata::core
