// tests/glm_dsa/glm_dsa_spec_gpu_test.cu - glm-dsa B11.2: the rejected-draft rollback on the device kernels (label
// gpu; exits 77 without a device).  The same checks as the CPU twin run (glm_dsa_spec_rollback.hpp) with the decode
// kernels Glm5Model::fast_dsa_glm launches (dsa_prep, dsa_rope_q, idx_score, topk, mla2) in the GPU-shaped MID
// geometry, FP16 and INT8 latents: the draft writes only row p of each cache, the redo is bit-identical to the
// token-at-a-time run, and a shared layer after the rollback refuses the rejected selection.  Then decode_spec's step
// order (spec_rb::schedule) with STRATA_GLM_SPEC_DRAFT_EARLY off and on: every part's and every NextN draft's output
// bit-identical to the token-at-a-time run in both orders.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
int g_fail = 0;
}
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

#include "glm_dsa_spec_rollback.hpp"

#define CK(x)                                                                                            \
    do {                                                                                                 \
        cudaError_t e_ = (x);                                                                            \
        if (e_ != cudaSuccess) {                                                                         \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));      \
            std::exit(1);                                                                                \
        }                                                                                                \
    } while (0)

namespace {

struct GpuBe {
    cudaStream_t s = nullptr;   // a blocking stream: the harness's synchronous copies order with it
    std::vector<void*> mem;
    GpuBe() { CK(cudaStreamCreate(&s)); }
    ~GpuBe() {
        cudaStreamSynchronize(s);
        for (void* p : mem) cudaFree(p);
        cudaStreamDestroy(s);
    }
    void* alloc(size_t n) {
        void* p = nullptr;
        CK(cudaMalloc(&p, std::max<size_t>(n, 4)));
        CK(cudaMemset(p, 0, std::max<size_t>(n, 4)));
        mem.push_back(p);
        return p;
    }
    void put(void* d, const void* h, size_t n) {
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
    }
    void get(void* h, const void* d, size_t n) {
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost));
    }
    void sync() { CK(cudaStreamSynchronize(s)); }
    bool gemv(const float* W, const float* x, float* y, int in, int out, float alpha) {
        glmd::ref_gemv(W, x, y, in, out, 1, alpha, s);
        return true;
    }
    bool prep(const glmd::DsaPrep& a) { return glmd::dsa_prep(a, s); }
    bool rope_q(const glmd::DsaRopeQ& a) { return glmd::dsa_rope_q(a, s); }
    bool score(const glmd::IdxScore& a) { return glmd::idx_score(a, s); }
    bool topk(const float* sc, int ld, int n_kv, const int* pos, int top_k, int* sel) {
        return glmd::topk(sc, ld, n_kv, 1, pos, top_k, sel, s);
    }
    bool mla2(const glmd::Mla2& a) { return glmd::mla2(a, s); }
};

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glm_dsa_spec_gpu_test: no CUDA device - skipped\n");
        return 77;
    }
    spec_rb::suite<GpuBe>("MID", mixfx::mid_dims(64), 36, 0);
    spec_rb::suite<GpuBe>("MID", mixfx::mid_dims(64), 36, 1);
    // glm-dsa B11.2: decode_spec's step order with STRATA_GLM_SPEC_DRAFT_EARLY off (today's) and on (the draft after
    // the tail's front parts, plus the wrong-order control)
    spec_rb::schedule<GpuBe>("MID", mixfx::mid_dims(64), 36, 0, false);
    spec_rb::schedule<GpuBe>("MID", mixfx::mid_dims(64), 36, 0, true);
    spec_rb::schedule<GpuBe>("MID", mixfx::mid_dims(64), 36, 1, true);
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
