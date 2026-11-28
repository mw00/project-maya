// tests/ik2_hip_mix_bench.cu - mix_rms_rows (the layer start's mix and norms) old against new, Inkling 2 Small's shapes
#include "ik2_hip_kernels.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)
namespace k = ik2hip::k;
int main() {
    const int E = 4608, Tm = 9, NB = 15;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> bl((size_t) NB * Tm * E), cur((size_t) Tm * E), qy(3 * E), g(3 * E);
    for (auto& v : bl) v = nd(rng); for (auto& v : cur) v = nd(rng); for (auto& v : qy) v = nd(rng) * 0.02f; for (auto& v : g) v = 1.0f + 0.1f * nd(rng);
    float *dbl, *dcur, *dq, *dg, *o1, *o2;
    CK(cudaMalloc(&dbl, bl.size() * 4)); CK(cudaMalloc(&dcur, cur.size() * 4)); CK(cudaMalloc(&dq, qy.size() * 4)); CK(cudaMalloc(&dg, g.size() * 4));
    CK(cudaMalloc(&o1, 3 * Tm * E * 4)); CK(cudaMalloc(&o2, 3 * Tm * E * 4));
    CK(cudaMemcpy(dbl, bl.data(), bl.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dcur, cur.data(), cur.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dq, qy.data(), qy.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s)); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int bad = 0;
    std::printf("%3s %3s %3s | %8s %8s | %9s\n", "nb", "nq", "T", "old us", "new us", "max/rms");
    for (int nb : { 1, 4, 8, 15 }) for (int nq : { 1, 3 }) for (int T : { 1, 2, 8 }) {
        const size_t ldb = (size_t) T * E;
        auto run = [&](int form, float* o) { k::mix_rms_rows_form(form, dbl, ldb, nb, dcur, T, E, dq, nq, dg, dg + E, dg + 2 * E, 1e-6f, o, s); };
        float us[2];
        for (int f = 0; f < 2; ++f) {
            for (int i = 0; i < 50; ++i) run(f, f ? o2 : o1);
            CK(cudaEventRecord(e0, s)); for (int i = 0; i < 500; ++i) run(f, f ? o2 : o1); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us[f] = 1e3f * ms / 500;
        }
        std::vector<float> a((size_t) nq * T * E), b(a.size());
        CK(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost));
        double mr = 0, rms = 0; for (size_t i = 0; i < a.size(); ++i) { rms += (double) a[i] * a[i]; mr = std::max(mr, std::fabs((double) a[i] - b[i])); } mr /= std::sqrt(rms / a.size());
        if (mr > 1e-4) ++bad;
        std::printf("%3d %3d %3d | %8.2f %8.2f | %9.2g\n", nb, nq, T, us[0], us[1], mr);
    }
    std::printf("%s\n", bad ? "MISMATCH" : "all within tolerance");
    return bad;
}
