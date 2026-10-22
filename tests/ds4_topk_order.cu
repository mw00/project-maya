// tests/ds4_topk_order.cu - strata-ds4's topk (the indexer's top 512 of each query's positions) against a reference on
// the host: the values above the k-th, then of those equal to it the first, each in position order - on random scores
// with heavy ties (the indexer's relu sums are often exactly 0), one value everywhere, and n from k + 1 to 200k; and the
// same output twice (it used to place them by atomics: other ties and another order run to run).  Each case also with
// the decode window's scratch (topk_seg + topk_merge: a long row over segments), three rows at once, and with the
// candidate positions' mapping (the layers that read the candidate source's blocks).
//   nvcc -O3 -std=c++17 -arch=native -I include -I src/ds4 tests/ds4_topk_order.cu src/ds4/ds4_kernels.cu \
//        src/ds4/ds4_host_ar.cpp -Xcompiler -mavx2,-mfma,-mf16c -o build/ds4_topk_order
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
            cudaMalloc(&dv, (size_t) 3 * n * 4);
            cudaMalloc(&dout, (size_t) 3 * 2 * k * 4);
            for (int t = 0; t < 3; ++t) cudaMemcpy(dv + (size_t) t * n, v.data(), (size_t) n * 4, cudaMemcpyHostToDevice);
            Rows r;
            r.n[0] = n;
            r.k[0] = k;
            topk(dv, n, 1, r, nullptr, 0, 1, dout, k, nullptr);
            topk(dv, n, 1, r, nullptr, 0, 1, dout + k, k, nullptr);
            std::vector<int> got((size_t) 2 * k);
            cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
            bool ok = std::equal(want.begin(), want.end(), got.begin()) &&
                      std::equal(got.begin(), got.begin() + k, got.begin() + k);
            // the scratch path: three rows (the same scores), twice; and with candidates (block b of the row at
            // position 7 * b + 3 of blk 4: the picks' positions mapped)
            TopkScratch sc;
            sc.T = 16, sc.P = kTopkSegMax, sc.K = 512;
            cudaMalloc(&sc.val, (size_t) sc.T * sc.P * sc.K * 4);
            cudaMalloc(&sc.idx, (size_t) sc.T * sc.P * sc.K * 4);
            cudaMalloc(&sc.cnt, (size_t) sc.T * sc.P * 4);
            cudaMalloc(&sc.kth, (size_t) sc.T * sc.P * 4);
            Rows r3;
            for (int t = 0; t < 3; ++t) r3.n[t] = n, r3.k[t] = k;
            bool seg_ok = true;
            for (int rep = 0; rep < 2; ++rep) {
                topk(dv, n, 3, r3, nullptr, 0, 1, dout, 2 * k, nullptr, &sc);
                std::vector<int> g3((size_t) 3 * 2 * k);
                cudaMemcpy(g3.data(), dout, g3.size() * 4, cudaMemcpyDeviceToHost);
                for (int t = 0; t < 3; ++t)
                    seg_ok = seg_ok && std::equal(want.begin(), want.end(), g3.begin() + (long) t * 2 * k);
            }
            const int blk = 4, nb = (n + blk - 1) / blk;
            std::vector<int> cand((size_t) nb);
            for (int b2 = 0; b2 < nb; ++b2) cand[(size_t) b2] = 7 * b2 + 3;
            int* dc;
            cudaMalloc(&dc, (size_t) nb * 4);
            cudaMemcpy(dc, cand.data(), (size_t) nb * 4, cudaMemcpyHostToDevice);
            topk(dv, n, 1, r, dc, 0, blk, dout, k, nullptr, &sc);
            std::vector<int> gc((size_t) k);
            cudaMemcpy(gc.data(), dout, (size_t) k * 4, cudaMemcpyDeviceToHost);
            for (int j = 0; j < k; ++j)
                seg_ok = seg_ok && gc[(size_t) j] == cand[(size_t) want[(size_t) j] / blk] * blk + want[(size_t) j] % blk;
            cudaFree(dc);
            cudaFree(sc.val);
            cudaFree(sc.idx);
            cudaFree(sc.cnt);
            cudaFree(sc.kth);
            std::printf("n %6d, %s: %s; segmented: %s\n", n, kind == 0 ? "random" : kind == 1 ? "heavy ties" : "one value",
                        ok ? "the reference's rows, in its order, twice" : "WRONG", seg_ok ? "the same" : "WRONG");
            ok = ok && seg_ok;
            fails += !ok;
            cudaFree(dv);
            cudaFree(dout);
        }
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
