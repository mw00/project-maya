// tests/dsa_topk_parity.cu - the O(n) DSA pool selection (src/kernels/cuda/dsa_topk.cuh) against the rank order the
// O(n^2) loop defined (score descending, ties by the lower index), on random scores, heavy ties (the ReLU sums are
// often exactly 0), one value everywhere and negative scores, n from 1 to 250k pools; and the old and new kernels'
// times at the context sizes that matter.
//   nvcc -O3 -std=c++17 -arch=sm_70 -I src/kernels/cuda tests/dsa_topk_parity.cu -o build/dsa_topk_parity
#include "dsa_topk.cuh"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

__global__ void new_kernel(const float* sc, int n, int top, int kpool, int* cells) {
    maya::kernels::dsa::select_top_pools(sc, n, top, kpool, cells);
}

// the loop it replaces (glm_fast.cu before 2026-10-07), scores from global memory
__global__ void old_kernel(const float* sc, int n_vis, int top, int kpool, int* cells) {
    for (int p = threadIdx.x; p < n_vis; p += blockDim.x) {
        const float v = sc[p];
        int rank = 0;
        for (int q = 0; q < n_vis; ++q) {
            const float u = sc[q];
            rank += u > v ? 1 : 0;
            rank += (q < p && u == v) ? 1 : 0;
        }
        if (rank < top)
            for (int m = 0; m < kpool; ++m) cells[rank * kpool + m] = p * kpool + m;
    }
}

static std::vector<int> reference(const std::vector<float>& s, int top, int kpool) {
    std::vector<int> idx(s.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return s[a] > s[b]; });   // ties keep index order
    std::vector<int> cells((size_t) top * kpool);
    for (int r = 0; r < top; ++r)
        for (int m = 0; m < kpool; ++m) cells[(size_t) r * kpool + m] = idx[r] * kpool + m;
    return cells;
}

int main() {
    const int kpool = 4, top_max = 512;
    std::mt19937 rng(7);
    const int sizes[] = {1, 2, 7, 511, 512, 513, 1000, 2000, 7500, 12288, 12289, 16000, 65536, 250000};
    const char* kinds[] = {"random", "ties", "constant", "negative"};
    float* d_s = nullptr;
    int* d_c = nullptr;
    cudaMalloc(&d_s, 250000 * sizeof(float));
    cudaMalloc(&d_c, top_max * kpool * sizeof(int));
    int bad = 0, cases = 0;
    for (int n : sizes)
        for (int kind = 0; kind < 4; ++kind) {
            std::vector<float> s(n);
            std::uniform_real_distribution<float> u(0.0f, 10.0f);
            std::uniform_int_distribution<int> q(0, 6);
            for (int i = 0; i < n; ++i)
                s[i] = kind == 0 ? u(rng) : kind == 1 ? (q(rng) < 4 ? 0.0f : 0.25f * (float) q(rng))
                     : kind == 2 ? 1.5f : -u(rng);
            const int top = std::min(top_max, n);
            cudaMemcpy(d_s, s.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_c, 0xff, top_max * kpool * sizeof(int));
            new_kernel<<<1, 1024>>>(d_s, n, top, kpool, d_c);
            std::vector<int> got((size_t) top * kpool);
            cudaMemcpy(got.data(), d_c, got.size() * sizeof(int), cudaMemcpyDeviceToHost);
            const cudaError_t e = cudaGetLastError();
            const bool ok = e == cudaSuccess && got == reference(s, top, kpool);
            ++cases;
            bad += !ok;
            if (!ok) std::printf("MISMATCH n=%d %s (%s)\n", n, kinds[kind], cudaGetErrorString(e));
        }
    std::printf("parity: %d of %d cases match the rank order\n", cases - bad, cases);
    // time at the context sizes of a chat: pools = context / 4
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int ctx : {8192, 32768, 65536, 131072, 262144}) {
        const int n = ctx / kpool;
        std::vector<float> s(n);
        std::uniform_real_distribution<float> u(0.0f, 10.0f);
        for (float& v : s) v = u(rng);
        cudaMemcpy(d_s, s.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        float t_new = 0, t_old = 0;
        for (int rep = 0; rep < 2; ++rep) {   // the first launch warms up
            cudaEventRecord(a);
            for (int i = 0; i < 10; ++i) new_kernel<<<1, 1024>>>(d_s, n, top_max, kpool, d_c);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            cudaEventElapsedTime(&t_new, a, b);
        }
        const int reps_old = ctx <= 65536 ? 3 : 1;
        cudaEventRecord(a);
        for (int i = 0; i < reps_old; ++i) old_kernel<<<1, 1024>>>(d_s, n, top_max, kpool, d_c);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&t_old, a, b);
        std::printf("context %6d (%5d pools): old %8.3f ms, new %6.3f ms per DSA layer per token\n", ctx, n,
                    t_old / reps_old, t_new / 10);
    }
    return bad ? 1 : 0;
}
