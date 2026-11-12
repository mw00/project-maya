// tests/glm_ablate_test.cu - the GLM engine's refusal-ablation projection (src/core/glm_ablate.hpp):
//   1. runs(): the rows that project, in runs; none -> no run (so no launch: the path as without --ablate)
//   2. project_out against the CPU reference (double dot) on rows of the model's widths (GLM-5.3-Flash 4096, GLM-5.3
//      6144), scales 1 and 0.35: each row within float rounding, and r . y after = (1 - s) r . y before
//   3. a step's rows projected by runs (rows 0, 2, 3 of 5) = the same rows of one launch over all 5, bit for bit, and
//      the rows that do not project untouched, bit for bit
// Needs a GPU (a few MB); none (or no memory): exit 77, ctest's skip.
#include "../src/core/glm_ablate.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace ga = strata::core::glmabl;

static int fails = 0;
#define CHECK(c, ...)                                                                                                  \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__);                                                            \
            std::printf(__VA_ARGS__);                                                                                  \
            std::printf("\n");                                                                                         \
            ++fails;                                                                                                   \
        }                                                                                                              \
    } while (0)

static void runs_test() {
    const uint8_t none[5] = {0, 0, 0, 0, 0};
    CHECK(ga::runs(none, 5).empty(), "all off: no run");
    CHECK(ga::runs(none, 0).empty(), "no rows: no run");
    const uint8_t mix[7] = {1, 0, 1, 1, 0, 0, 1};
    const auto r = ga::runs(mix, 7);
    CHECK(r.size() == 3 && r[0] == std::make_pair(0, 1) && r[1] == std::make_pair(2, 4) && r[2] == std::make_pair(6, 7),
          "mixed rows: [0,1) [2,4) [6,7)");
    const uint8_t all[3] = {1, 1, 1};
    CHECK(ga::runs(all, 3).size() == 1 && ga::runs(all, 3)[0] == std::make_pair(0, 3), "all on: one run");
}

static double dot(const float* a, const float* b, int n) {
    double d = 0.0;
    for (int i = 0; i < n; ++i) d += (double) a[i] * b[i];
    return d;
}

static bool gpu_test(int dim, float scale) {
    const int T = 5;
    std::mt19937 rng(1234u + (unsigned) dim);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> r((size_t) dim), y((size_t) T * dim);
    for (float& v : r) v = nd(rng);
    const double n = std::sqrt(dot(r.data(), r.data(), dim));
    for (float& v : r) v = (float) (v / n);
    for (int t = 0; t < T; ++t)
        for (int e = 0; e < dim; ++e)   // (rows with a strong component along r, as a refusal direction has)
            y[(size_t) t * dim + e] = nd(rng) * 0.05f + (float) (t + 1) * 3.0f * r[(size_t) e];
    float *dr = nullptr, *dy = nullptr, *dz = nullptr;
    if (cudaMalloc(&dr, r.size() * 4) != cudaSuccess || cudaMalloc(&dy, y.size() * 4) != cudaSuccess ||
        cudaMalloc(&dz, y.size() * 4) != cudaSuccess) {
        std::printf("SKIP: no GPU memory (%s)\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    cudaMemcpy(dr, r.data(), r.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dz, y.data(), y.size() * 4, cudaMemcpyHostToDevice);
    // all T rows in one launch
    ga::project_out(dy, dr, scale, dim, T, nullptr);
    // rows 0, 2, 3 by runs (as Glm5Model::ablate_runs does for a step whose rows 1 and 4 do not project)
    const uint8_t on[T] = {1, 0, 1, 1, 0};
    for (const auto& ru : ga::runs(on, T))
        ga::project_out(dz + (size_t) ru.first * dim, dr, scale, dim, ru.second - ru.first, nullptr);
    ga::project_out(dz, dr, scale, dim, 0, nullptr);   // (T 0: nothing)
    const cudaError_t e = cudaDeviceSynchronize();
    CHECK(e == cudaSuccess, "launch: %s", cudaGetErrorString(e));
    std::vector<float> got(y.size()), runs_out(y.size());
    cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(runs_out.data(), dz, runs_out.size() * 4, cudaMemcpyDeviceToHost);
    for (int t = 0; t < T; ++t) {
        std::vector<float> ref(y.begin() + (ptrdiff_t) t * dim, y.begin() + (ptrdiff_t) (t + 1) * dim);
        const double before = dot(r.data(), ref.data(), dim);
        ga::project_row_ref(ref.data(), r.data(), scale, dim);
        const float* g = got.data() + (size_t) t * dim;
        double worst = 0.0, ynorm = std::sqrt(dot(ref.data(), ref.data(), dim));
        for (int i = 0; i < dim; ++i) worst = std::max(worst, (double) std::fabs(g[i] - ref[(size_t) i]));
        CHECK(worst <= 1e-5 * std::max(1.0, std::fabs(before)), "dim %d scale %g row %d: max |gpu - ref| %.3g (r.y %.3g)",
              dim, scale, t, worst, before);
        const double after = dot(r.data(), g, dim);
        CHECK(std::fabs(after - (1.0 - scale) * before) <= 1e-5 * std::fabs(before) + 1e-6 * ynorm,
              "dim %d scale %g row %d: r.y after %.6g, want %.6g", dim, scale, t, after, (1.0 - scale) * before);
        const float* z = runs_out.data() + (size_t) t * dim;
        const float* src = on[t] ? g : y.data() + (size_t) t * dim;   // (projected: the one launch's bits; else as given)
        CHECK(std::memcmp(z, src, (size_t) dim * 4) == 0, "dim %d row %d: runs vs %s differ", dim, t,
              on[t] ? "one launch" : "untouched");
    }
    cudaFree(dr);
    cudaFree(dy);
    cudaFree(dz);
    return true;
}

int main() {
    runs_test();
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("runs: %s; SKIP the GPU part: no CUDA device\n", fails ? "FAIL" : "ok");
        return fails ? 1 : 77;
    }
    for (int dim : {4096, 6144})
        for (float s : {1.0f, 0.35f})
            if (!gpu_test(dim, s)) return fails ? 1 : 77;
    std::printf("glm_ablate_test: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
