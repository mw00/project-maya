// tests/ds4_allreduce_dma.cu - strata-ds4's chunk all-reduce through the copy engines (allreduce_dma) against
// ar_rs16_kernel's (allreduce, three members or more): the same FP16 sums bit for bit on every member, a chunk's row
// (16..2048 tokens of 5120 values) at a time, several in a row on the two slots; and how long each takes.  One member a
// GPU (4, then 8 of them when there are).
//   nvcc -O3 -std=c++17 -arch=native -I include -I src/ds4 tests/ds4_allreduce_dma.cu src/ds4/ds4_kernels.cu src/ds4/ds4_host_ar.cpp \
//        -o build/ds4_allreduce_dma
#include "ds4_kernels.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ds4;

#define CK(x)                                                                         \
    do {                                                                              \
        const cudaError_t e_ = (x);                                                   \
        if (e_ != cudaSuccess) {                                                      \
            std::printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class T>
int hmap(T*& p, size_t n) {
    void* v = nullptr;
    CK(cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(v, 0, n * sizeof(T) + 64);
    p = (T*) v;
    return 0;
}

// nm members on GPUs 0..nm-1: rounds of all-reduces of n values (DMA or kernel), the results on the host
int run(int nm, int n, int rounds, int dma, unsigned& seq, ArGroup& a, std::vector<float*>& x, std::vector<void*>& scr,
        std::vector<cudaStream_t>& st, std::vector<unsigned*>& cnt, const std::vector<std::vector<float>>& in,
        std::vector<std::vector<float>>& out, double& ms, std::vector<ArAux>& aux, HostReducer* hr = nullptr) {
    ms = 0;
    for (int r = 0; r < rounds; ++r) {
        for (int m = 0; m < nm; ++m) {
            CK(cudaSetDevice(m));
            CK(cudaMemcpy(x[m], in[(size_t) (r * nm + m) % in.size()].data(), (size_t) n * 4, cudaMemcpyHostToDevice));
            CK(cudaDeviceSynchronize());
        }
        ++seq;
        const double t0 = now_ms();
        if (dma == 3) hr->submit(seq, n);
        for (int m = 0; m < nm; ++m) {
            CK(cudaSetDevice(m));
            if (dma == 3) allreduce_host(x[m], n, a, *hr, m, seq, scr[m], st[m], aux[m]);
            else if (dma) allreduce_dma(x[m], n, a, m, seq, scr[m], st[m], dma == 2 ? &aux[m] : nullptr);
            else allreduce(x[m], n, a, m, seq, cnt[m], st[m]);
        }
        for (int m = 0; m < nm; ++m) {
            CK(cudaSetDevice(m));
            CK(cudaStreamSynchronize(st[m]));
        }
        ms += now_ms() - t0;
        for (int m = 0; m < nm; ++m) {
            CK(cudaSetDevice(m));
            out[(size_t) (r * nm + m)].resize((size_t) n);
            CK(cudaMemcpy(out[(size_t) (r * nm + m)].data(), x[m], (size_t) n * 4, cudaMemcpyDeviceToHost));
        }
    }
    ms /= rounds;
    return 0;
}

}  // namespace

int main() {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    int fails = 0;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    for (int nm : {4, 8}) {
        if (nm > ndev) continue;
        const int D = 5120, Cmax = 2048, slots = 2, rounds = 3;
        const size_t stride = (size_t) Cmax * D;
        ArGroup a;
        a.nm = nm;
        a.slots = slots;
        a.stride = stride;
        if (hmap(a.data, (size_t) nm * slots * stride) || hmap(a.flags, (size_t) nm * slots * 32) ||
            hmap(a.sums, (size_t) slots * stride) || hmap(a.bflags, (size_t) 2 * nm * slots * kArBlocks * 8))
            return 1;
        std::vector<float*> x(nm);
        std::vector<void*> scr(nm);
        std::vector<cudaStream_t> st(nm);
        std::vector<unsigned*> cnt(nm);
        std::vector<ArAux> aux(nm);
        for (int m = 0; m < nm; ++m) {
            CK(cudaSetDevice(m));
            CK(cudaMalloc(&x[m], stride * 4));
            CK(cudaMalloc(&scr[m], stride * 4 + 64));
            CK(cudaStreamCreateWithFlags(&st[m], cudaStreamNonBlocking));
            CK(cudaMalloc(&cnt[m], 64));
            CK(cudaMemset(cnt[m], 0, 64));
            CK(cudaStreamCreateWithFlags(&aux[m].s2, cudaStreamNonBlocking));
            for (cudaEvent_t& e : aux[m].ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        }
        unsigned seq = 0;
        std::vector<int> ids;
        for (int m = 0; m < nm; ++m) ids.push_back(m);
        HostReducer hr(a, ids, stride, 32);
        for (int T : {16, 333, 2048}) {
            const int n = T * D;
            std::vector<std::vector<float>> in((size_t) rounds * nm, std::vector<float>((size_t) n));
            for (auto& v : in)
                for (float& f : v) f = nd(rng) * (rng() % 16 == 0 ? 300.0f : 1.0f);   // (a few past FP16's steps)
            std::vector<std::vector<float>> o1((size_t) rounds * nm), o2((size_t) rounds * nm), o3((size_t) rounds * nm),
                o4((size_t) rounds * nm);
            double ms1 = 0, ms2 = 0, ms3 = 0, ms4 = 0;
            if (run(nm, n, rounds, 0, seq, a, x, scr, st, cnt, in, o1, ms1, aux)) return 1;
            if (run(nm, n, rounds, 1, seq, a, x, scr, st, cnt, in, o2, ms2, aux)) return 1;
            if (run(nm, n, rounds, 2, seq, a, x, scr, st, cnt, in, o3, ms3, aux)) return 1;
            if (run(nm, n, rounds, 3, seq, a, x, scr, st, cnt, in, o4, ms4, aux, &hr)) return 1;
            bool same = true;
            for (size_t i = 0; i < o1.size(); ++i)
                same = same && std::memcmp(o1[i].data(), o2[i].data(), (size_t) n * 4) == 0 &&
                       std::memcmp(o1[i].data(), o3[i].data(), (size_t) n * 4) == 0 &&
                       std::memcmp(o1[i].data(), o4[i].data(), (size_t) n * 4) == 0 &&
                       std::memcmp(o2[i].data(), o2[i - i % nm].data(), (size_t) n * 4) == 0;   // (every member alike)
            std::printf("allreduce, %d members, %d x %d values (%.1f MB FP16): DMA / pieces / host %s the kernel's; kernel "
                        "%.2f ms, DMA %.2f ms, in pieces %.2f ms, summed by the host %.2f ms\n", nm, T, D, n * 2 / 1e6,
                        same ? "bit for bit" : "DIFFERENT from", ms1, ms2, ms3, ms4);
            fails += !same;
        }
    }
    std::printf(fails == 0 ? "PASS\n" : "FAIL\n");
    return fails != 0;
}
