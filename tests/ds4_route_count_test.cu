// tests/ds4_route_count_test.cu - route_publish's count (issue #501): a host-free layer's routes (no request to the host)
// counted on the device, one per kept pick, while the picks, weights and plan stay bit for bit those of a routing
// without the count.  Random logits for windows of 1..kWin tokens, a layer of 256 or 384 experts, 6 picks, some
// experts off the device (plan 0); run it with MAYA_DS4_RESIDENT_SKIP_W set too (dropped picks: weight 0, not counted).
// Needs a GPU, no model.
#include "../src/ds4/ds4_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ds4;

namespace {

#define CK(x)                                                                                                        \
    do {                                                                                                             \
        const cudaError_t e_ = (x);                                                                                  \
        if (e_ != cudaSuccess) {                                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));                \
            std::exit(2);                                                                                            \
        }                                                                                                            \
    } while (0)

template <class T> T* dev(size_t n) {
    void* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return (T*) p;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::printf("ds4_route_count_test: no GPU - skipped\n");
        return 77;
    }
    std::mt19937 rng(501);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    const int dim = 64, k = 6;
    int fails = 0, cases = 0;
    for (const int n : {256, 384})
        for (int T = 1; T <= kWin; T += (T < 4 ? 1 : 5)) {
            std::vector<float> lg((size_t) T * 1024, 0.0f), bias(n), x((size_t) T * dim);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < n; ++i) lg[(size_t) t * 1024 + i] = nd(rng);
            for (auto& b : bias) b = 0.1f * nd(rng);
            for (auto& v : x) v = nd(rng);
            std::vector<unsigned long long> vram(n);
            for (int i = 0; i < n; ++i) vram[i] = i % 7 == 3 ? 0ull : 0x10000ull + (unsigned long long) i * 256;
            float *d_lg = dev<float>(lg.size()), *d_b = dev<float>(n), *d_x = dev<float>(x.size());
            unsigned long long* d_v = dev<unsigned long long>(n);
            CK(cudaMemcpy(d_lg, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_b, bias.data(), (size_t) n * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_v, vram.data(), (size_t) n * 8, cudaMemcpyHostToDevice));
            std::vector<int> idx[2];
            std::vector<float> w[2];
            std::vector<unsigned long long> plan[2];
            std::vector<unsigned> cnt(n);
            unsigned* d_cnt = dev<unsigned>(n);
            for (int run = 0; run < 2; ++run) {   // 0: no count (the engine before), 1: with the count, twice
                int* d_idx = dev<int>((size_t) T * kPicks);
                float* d_w = dev<float>((size_t) T * kPicks);
                unsigned long long* d_p = dev<unsigned long long>((size_t) T * kPicks);
                for (int rep = 0; rep < (run ? 2 : 1); ++rep)
                    route_publish(d_lg, d_b, n, k, T, 2.5f, d_v, d_x, dim, d_idx, d_w, d_p, nullptr, 0, s, nullptr, 0,
                                  0, 0, nullptr, nullptr, nullptr, run ? d_cnt : nullptr);
                CK(cudaStreamSynchronize(s));
                idx[run].resize((size_t) T * kPicks), w[run].resize(idx[run].size()), plan[run].resize(idx[run].size());
                CK(cudaMemcpy(idx[run].data(), d_idx, idx[run].size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(w[run].data(), d_w, w[run].size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(plan[run].data(), d_p, plan[run].size() * 8, cudaMemcpyDeviceToHost));
                CK(cudaFree(d_idx));
                CK(cudaFree(d_w));
                CK(cudaFree(d_p));
            }
            CK(cudaMemcpy(cnt.data(), d_cnt, (size_t) n * 4, cudaMemcpyDeviceToHost));
            ++cases;
            bool ok = std::memcmp(idx[0].data(), idx[1].data(), idx[0].size() * 4) == 0 &&
                      std::memcmp(w[0].data(), w[1].data(), w[0].size() * 4) == 0 &&
                      std::memcmp(plan[0].data(), plan[1].data(), plan[0].size() * 8) == 0;
            // the expected counts: each kept pick (weight > 0; a dropped one has 0) twice (two routings)
            std::vector<unsigned> want(n, 0u);
            size_t picks = 0;
            for (int t = 0; t < T; ++t)
                for (int r = 0; r < k; ++r) {
                    const size_t o = (size_t) t * kPicks + r;
                    if (w[0][o] != 0.0f) want[(size_t) idx[0][o]] += 2u, ++picks;
                }
            size_t got = 0;
            for (int i = 0; i < n; ++i) {
                ok = ok && cnt[(size_t) i] == want[(size_t) i];
                got += cnt[(size_t) i];
            }
            if (!ok) {
                ++fails;
                std::printf("FAIL n %d T %d: picks/weights/plan equal %d, counts %zu (want %zu)\n", n, T,
                            (int) (std::memcmp(idx[0].data(), idx[1].data(), idx[0].size() * 4) == 0), got, 2 * picks);
            }
            CK(cudaFree(d_cnt));
            CK(cudaFree(d_lg));
            CK(cudaFree(d_b));
            CK(cudaFree(d_x));
            CK(cudaFree(d_v));
        }
    CK(cudaStreamDestroy(s));
    std::printf("ds4_route_count_test: %d of %d cases %s\n", cases - fails, cases, fails ? "FAILED" : "pass");
    return fails ? 1 : 0;
}
