// src/core/glm_ablate.cu - refusal ablation on Maya's GLM engine (see glm_ablate.hpp): the projection kernel, the
// direction's load onto every part of the split, and the per-conversation switch.
#include "glm_ablate.hpp"

#include "maya/core/glm_model.hpp"
#include "maya/core/glp.hpp"

#include "ablate_capture.hpp"         // (MAYA_ABLATE_CAPTURE: the direction fitting's read-out)
#include "../mm3/mm3_ablate.hpp"   // (the control vector loader maya-mm3 uses: the same direction files)

#include <algorithm>
#include <string>
#include <vector>

namespace maya::core {

namespace glmabl {

namespace {
constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads) project_out_kernel(float* __restrict__ y, const float* __restrict__ r,
                                                               float scale, int dim) {
    __shared__ float red[kThreads];
    y += (size_t) blockIdx.x * (size_t) dim;
    float d = 0.0f;
    for (int e = threadIdx.x; e < dim; e += kThreads) d += r[e] * y[e];
    red[threadIdx.x] = d;
    __syncthreads();
    for (int o = kThreads / 2; o > 0; o >>= 1) {   // (a fixed tree: the same bits on every device of a kind)
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const float a = scale * red[0];
    for (int e = threadIdx.x; e < dim; e += kThreads) y[e] -= a * r[e];
}
}  // namespace

void project_out(float* y, const float* r, float scale, int dim, int T, void* stream) {
    if (T <= 0 || y == nullptr || r == nullptr) return;
    project_out_kernel<<<T, kThreads, 0, (cudaStream_t) stream>>>(y, r, scale, dim);
}

}  // namespace glmabl

bool Glm5Model::ablate_load(const std::string& path, int layer, float scale, bool on, std::string& err) {
    std::vector<float> r;
    std::string what;
    if (!mm3::ablate::load(path, layer, g_.n_embd, scale, r, what, err, "maya's GLM engine")) return false;
    int dev0 = 0;
    cudaGetDevice(&dev0);
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {   // (a copy on every part's card)
        cudaSetDevice(m->dev_);
        if (m->abl_r_ == nullptr && cudaMalloc(&m->abl_r_, r.size() * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            m->abl_r_ = nullptr;
            cudaSetDevice(dev0);
            err = "--ablate: the direction (" + std::to_string(r.size() * sizeof(float)) + " bytes) did not fit on CUDA" +
                  std::to_string(m->dev_);
            return false;
        }
        if (cudaMemcpy(m->abl_r_, r.data(), r.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaSetDevice(dev0);
            err = "--ablate: the direction's copy to CUDA" + std::to_string(m->dev_) + " failed";
            return false;
        }
        m->abl_scale_ = scale;
        m->abl_on_.assign((size_t) std::max(1, m->seq_count()), on ? 1 : 0);
    }
    cudaSetDevice(dev0);
    abl_what_ = "glm: " + what;
    return true;
}

void Glm5Model::set_ablate_seq(int sq, bool on) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (sq >= 0 && sq < (int) m->abl_on_.size()) m->abl_on_[(size_t) sq] = on ? 1 : 0;
}

void Glm5Model::ablate(float* y, int T, cudaStream_t s) const {
    if (T <= 0 || !abl_seq_on(cur_seq_)) return;
    if (ablcap::on()) return ablcap::put(y, T, g_.n_embd, s);   // (a fitting run: read out, not projected)
    glmabl::project_out(y, abl_r_, abl_scale_, g_.n_embd, T, s);
}

void Glm5Model::ablate_runs(float* y, const int* seqs, int n, cudaStream_t s) const {
    if (abl_r_ == nullptr || n <= 0) return;
    uint8_t on[64] = {};
    const int k = std::min(n, 64);
    bool any = false;
    for (int i = 0; i < k; ++i) {
        on[i] = abl_seq_on(seqs[i]) ? 1 : 0;
        any = any || on[i] != 0;
    }
    if (!any) return;
    const int E = g_.n_embd;
    for (const auto& ru : glmabl::runs(on, k))
        glmabl::project_out(y + (size_t) ru.first * E, abl_r_, abl_scale_, E, ru.second - ru.first, s);
}

}  // namespace maya::core

// --ablate with a GLP file (glp.* metadata): the reader's per-layer directions on every part,
// the same per-conversation on/off as the single-direction path.  ablate_layer() applies the
// layer's own row at the same two writes (glm_model.cu).
bool Glm5Model::ablate_load_glp(const std::string& path, float alpha, bool on, std::string& err) {
    maya::core::glp::Vector v;
    std::vector<std::string> warnings;
    maya::core::glp::LoadOptions lo;
    lo.alpha = alpha > 0.0f ? alpha : -1.0f;   // --ablate-scale 0 = the file's glp.alpha_default (LoadOptions < 0)
    if (!maya::core::glp::load(path, g_.n_embd, g_.n_layers, lo, v, err, warnings)) return false;
    for (const std::string& w : warnings) std::fprintf(stderr, "maya generate: --ablate: %s\n", w.c_str());
    int dev0 = 0;
    cudaGetDevice(&dev0);
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {   // a copy on every part's card
        cudaSetDevice(m->dev_);
        const size_t bytes = v.dir.size() * sizeof(float);
        if (m->abl_glp_r_ == nullptr && cudaMalloc(&m->abl_glp_r_, bytes) != cudaSuccess) {
            cudaGetLastError();
            m->abl_glp_r_ = nullptr;
            cudaSetDevice(dev0);
            err = "--ablate: the GLP direction table (" + std::to_string(bytes) + " bytes) did not fit on CUDA" +
                  std::to_string(m->dev_);
            return false;
        }
        if (cudaMemcpy(m->abl_glp_r_, v.dir.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaSetDevice(dev0);
            err = "--ablate: the GLP direction table's copy to CUDA" + std::to_string(m->dev_) + " failed";
            return false;
        }
        m->abl_scale_ = v.alpha;
        m->abl_glp_s_host_ = v.s;
        m->abl_on_.assign((size_t) std::max(1, m->seq_count()), on ? 1 : 0);
    }
    cudaSetDevice(dev0);
    abl_what_ = "glm: glp project, layers " + std::to_string(v.layers.front()) + ".." +
                std::to_string(v.layers.back()) + " (" + std::to_string(v.layers.size()) + " steered), alpha " +
                std::to_string(v.alpha);
    return true;
}

// The layer's own GLP row on T rows (y -= scale * d (d . y) with the layer's unit row), or nothing
// when the file does not steer the layer.  The single-direction ablate() and this share abl_on_.
void Glm5Model::ablate_layer(float* y, int64_t layer, int T, cudaStream_t s) const {
    if (T <= 0 || !abl_seq_on(cur_seq_) || abl_glp_r_ == nullptr || layer < 0 || layer > g_.n_layers) return;
    if (layer >= (int64_t) abl_glp_s_host_.size() || abl_glp_s_host_[(size_t) layer] == 0.0f) return;
    glmabl::project_out(y, abl_glp_r_ + (size_t) layer * (size_t) g_.n_embd, abl_scale_, (int) g_.n_embd, T, s);
}
