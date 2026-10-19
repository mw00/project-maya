// tests/ds4_topk_order.cu - strata-ds4's topk (the indexer's top 512 of each query's positions) against a reference on
// the host: the values above the k-th, then of those equal to it the first, each in position order - on random scores
// with heavy ties (the indexer's relu sums are often exactly 0), one value everywhere, and n from k + 1 to 200k; and the
// same output twice (it used to place them by atomics: other ties and another order run to run).
//   nvcc -O3 -std=c++17 -arch=native -I src/ds4 tests/ds4_topk_order.cu src/ds4/ds4_kernels.cu -o build/ds4_topk_order
#include "ds4_kernels.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ds4;

int main() {
    std::mt19937 rng(7);
    int fails = 0;
    const int k = 512;
    for (int n : {k + 1, 600, 4096, 70000, 200000})
        for (int kind = 0; kind < 3; ++kind) {
            std::vector<float> v((size_t) n);
            std::uniform_int_distribution<int> few(0, 20);
            std::normal_distribution<float> nd;
            for (float& x : v) x = kind == 0 ? nd(rng) : kind == 1 ? (float) (few(rng) / 4) : 1.5f;   // (ties: 1, all: 2)
            // the reference: positions by value descending, ties by position; the top k; then the run above the k-th
            // value in position order, then the k-th's ties in position order
            std::vector<int> ord((size_t) n);
            for (int i = 0; i < n; ++i) ord[(size_t) i] = i;
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return v[(size_t) a] > v[(size_t) b]; });
            const float kth = v[(size_t) ord[(size_t) k - 1]];
            std::vector<int> want;
            for (int i = 0; i < n; ++i)
                if (v[(size_t) i] > kth) want.push_back(i);
            for (int i = 0; i < n && (int) want.size() < k; ++i)
                if (v[(size_t) i] == kth) want.push_back(i);
            float* dv;
            int* dout;
            cudaMalloc(&dv, (size_t) n * 4);
            cudaMalloc(&dout, (size_t) 2 * k * 4);
            cudaMemcpy(dv, v.data(), (size_t) n * 4, cudaMemcpyHostToDevice);
            Rows r;
            r.n[0] = n;
            r.k[0] = k;
            topk(dv, n, 1, r, nullptr, 0, 1, dout, k, nullptr);
            topk(dv, n, 1, r, nullptr, 0, 1, dout + k, k, nullptr);
            std::vector<int> got((size_t) 2 * k);
            cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
            const bool ok = std::equal(want.begin(), want.end(), got.begin()) &&
                            std::equal(got.begin(), got.begin() + k, got.begin() + k);
            std::printf("n %6d, %s: %s\n", n, kind == 0 ? "random" : kind == 1 ? "heavy ties" : "one value",
                        ok ? "the reference's rows, in its order, twice" : "WRONG");
            fails += !ok;
            cudaFree(dv);
            cudaFree(dout);
        }
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
