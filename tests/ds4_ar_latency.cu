// tests/ds4_ar_latency.cu - a decode window's all-reduce latency (allreduce: T x 5120 values, the window's ArGroup of
// 4 slots), members 2 / 4 / 8, each member's host thread enqueuing `reps` all-reduces back to back on its own stream -
// the time an all-reduce takes in steady state, launches included.
//   nvcc -O3 -std=c++17 -arch=native -I src/ds4 tests/ds4_ar_latency.cu src/ds4/ds4_kernels.cu src/ds4/ds4_host_ar.cpp \
//        -o build/ds4_ar_latency
#include "ds4_kernels.cuh"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using namespace strata::ds4;

namespace {
template <class T>
T* hmap(size_t n) {
    void* v = nullptr;
    if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return nullptr;
    std::memset(v, 0, n * sizeof(T) + 64);
    return (T*) v;
}
}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    for (int nm : {2, 4, 8}) {
        if (nm > ndev) continue;
        for (int T : {1, 6}) {
            const int D = 5120, n = T * D;
            ArGroup a;
            a.nm = nm;
            a.slots = 4;
            a.stride = (size_t) 16 * D;
            a.data = hmap<float>((size_t) nm * a.slots * a.stride);
            a.flags = hmap<unsigned>((size_t) nm * a.slots * 32);
            if (nm > 2) {
                a.sums = hmap<float>((size_t) a.slots * a.stride);
                a.bflags = hmap<unsigned>((size_t) 2 * nm * a.slots * kArBlocks * 8);
            }
            std::vector<float*> x(nm);
            std::vector<cudaStream_t> st(nm);
            std::vector<unsigned*> cnt(nm);
            for (int m = 0; m < nm; ++m) {
                cudaSetDevice(m);
                cudaMalloc(&x[m], (size_t) n * 4);
                cudaMemset(x[m], 0, (size_t) n * 4);
                cudaStreamCreateWithFlags(&st[m], cudaStreamNonBlocking);
                cudaMalloc(&cnt[m], 64);
                cudaMemset(cnt[m], 0, 64);
                cudaDeviceSynchronize();
            }
            double best = 1e30;
            unsigned seq0 = 0;
            for (int trial = 0; trial < 3; ++trial) {
                std::vector<std::thread> th;
                const auto t0 = std::chrono::steady_clock::now();
                for (int m = 0; m < nm; ++m)
                    th.emplace_back([&, m] {
                        cudaSetDevice(m);
                        for (int r = 0; r < reps; ++r) allreduce(x[m], n, a, m, seq0 + 1 + (unsigned) r, cnt[m], st[m]);
                        cudaStreamSynchronize(st[m]);
                    });
                for (auto& t : th) t.join();
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                best = std::min(best, us / reps);
                seq0 += (unsigned) reps;
            }
            std::printf("allreduce, %d members, %d tokens (%d values): %.1f us each\n", nm, T, n, best);
        }
    }
}
