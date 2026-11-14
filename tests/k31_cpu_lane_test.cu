// tests/k31_cpu_lane_test.cu - the CPU lane (src/k31/k31_cpu_lane.cpp) against the device: the runner's int8 rounding
// byte for byte, and IQ1_S rows bit for bit against iq::mv, over random blocks, column counts and input scales.
//   k31_cpu_lane_test        exit 0: every byte and bit equal
//   k31_cpu_lane_test --bench THREADS [TYPE]   the rows' speed: an expert's gate (2560 x 3584, IQ1_S 19 / IQ2_XXS 16), one input, GB/s
#include "k31_cpu_lane.hpp"

#include "../src/ds4/ds4_iq.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace iq = strata::ds4::gguf::iq;

namespace {

// the runner's q8_quant_kernel (src/k31/k31_model.cu), as it is there
__global__ void q8_quant_kernel(const float* __restrict__ x, int k, int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = blockIdx.y, b = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const float v = x[(size_t) r * k + (size_t) b * 32 + lane];
    float am = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[(size_t) r * (k / 32) + b] = sc;
}

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

uint16_t f2h(float f) {   // (a positive normal F16 near f: the test's block scales)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const int e = (int) ((u >> 23) & 255u) - 127 + 15;
    return (uint16_t) ((u >> 16 & 0x8000u) | ((uint32_t) e << 10) | ((u >> 13) & 1023u));
}

}  // namespace

int bench(int threads, int type) {
    constexpr int k = 3584, n = 2560, kMats = 64;   // 64 matrices (IQ1_S ~367 MB): past the caches
    const int B = type == 19 ? 50 : 66;
    const size_t ld = (size_t) (k / 256) * B, mb = (size_t) n * ld;
    std::vector<uint8_t> w(mb * kMats);
    std::mt19937 rng(7);
    for (size_t b = 0; b < w.size() / B; ++b) {
        uint8_t* p = w.data() + b * B;
        for (int j = 2; j < B; ++j) p[j] = (uint8_t) rng();
        const uint16_t h = f2h(0.01f);
        std::memcpy(p, &h, 2);
    }
    std::vector<int8_t> xq(k);
    std::vector<float> xd(k / 32, 0.01f);
    std::vector<std::vector<float>> ys(kMats, std::vector<float>((size_t) n));
    for (auto& v : xq) v = (int8_t) (rng() % 255 - 127);
    k31::cpu::Pool pool(threads);
    constexpr int kChunk = 128;
    const int ch = n / kChunk;
    for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        pool.run(kMats * ch, [&](int t) {   // (as the lane runs a layer: every expert's row chunks in one call)
            const int m = t / ch, c = t % ch;
            k31::cpu::rows(type, w.data() + (size_t) m * mb, ld, k, c * kChunk, (c + 1) * kChunk, xq.data(), xd.data(), 1,
                           ys[(size_t) m].data() + c * kChunk, (size_t) n);
        });
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("type %d, %d threads (%s): %.2f GB/s, %.3f ms a matrix\n", type, threads, k31::cpu::form(), (double) w.size() / s / 1e9,
                    1e3 * s / kMats);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--bench") return bench(std::atoi(argv[2]), argc >= 4 ? std::atoi(argv[3]) : 19);
    std::mt19937 rng(20261107);
    int bad = 0, cases = 0;
    for (int type : {19, 16}) {   // IQ1_S (50 bytes a block), IQ2_XXS (66)
    const int B = type == 19 ? 50 : 66;
    for (int k : {256, 2048, 3584}) {
        for (int n : {1, 37, 300}) {
            for (int ncol : {1, 5, 16, 20}) {
                for (float scale : {1e-3f, 1.0f, 300.0f}) {
                    ++cases;
                    const size_t ld = (size_t) (k / 256) * B;
                    std::vector<uint8_t> w((size_t) n * ld);
                    for (size_t b = 0; b < w.size() / B; ++b) {
                        uint8_t* p = w.data() + b * B;
                        for (int j = 2; j < B; ++j) p[j] = (uint8_t) rng();
                        std::uniform_real_distribution<float> ud(0.002f, 0.2f);
                        const uint16_t h = f2h(ud(rng));
                        std::memcpy(p, &h, 2);
                    }
                    std::normal_distribution<float> nd(0.0f, scale);
                    std::vector<float> x((size_t) ncol * k);
                    for (auto& v : x) v = nd(rng);
                    for (int i = 0; i < ncol; ++i) x[(size_t) i * k + (size_t) (rng() % k)] *= 40.0f;   // an outlier a column
                    if (k >= 64) for (int j = 0; j < 32; ++j) x[32 + j] = 0.0f;                        // an all-zero block
                    // device
                    float *dx = nullptr, *dxd = nullptr, *dy = nullptr;
                    int8_t* dq = nullptr;
                    uint8_t* dw = nullptr;
                    if (!ck(cudaMalloc(&dx, x.size() * 4), "malloc") || !ck(cudaMalloc(&dq, x.size()), "malloc") ||
                        !ck(cudaMalloc(&dxd, x.size() / 32 * 4), "malloc") || !ck(cudaMalloc(&dy, (size_t) ncol * n * 4), "malloc") ||
                        !ck(cudaMalloc(&dw, w.size()), "malloc"))
                        return 2;
                    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
                    q8_quant_kernel<<<dim3((unsigned) ((k / 32 + 7) / 8), (unsigned) ncol), 256>>>(dx, k, dq, dxd);
                    std::vector<iq::MvJob> jobs;
                    for (int c0 = 0; c0 < ncol; c0 += 16) {
                        iq::MvJob j;
                        j.w = dw; j.ld = ld; j.type = type; j.k = k; j.n = n; j.ncol = std::min(16, ncol - c0);
                        j.xq = dq + (size_t) c0 * k; j.xd = dxd + (size_t) c0 * (k / 32); j.y = dy + (size_t) c0 * n; j.ycs = (size_t) n;
                        jobs.push_back(j);
                    }
                    if (!iq::mv(jobs.data(), (int) jobs.size(), nullptr) || !ck(cudaDeviceSynchronize(), "mv")) return 2;
                    std::vector<int8_t> gq(x.size());
                    std::vector<float> gd(x.size() / 32), gy((size_t) ncol * n);
                    cudaMemcpy(gq.data(), dq, gq.size(), cudaMemcpyDeviceToHost);
                    cudaMemcpy(gd.data(), dxd, gd.size() * 4, cudaMemcpyDeviceToHost);
                    cudaMemcpy(gy.data(), dy, gy.size() * 4, cudaMemcpyDeviceToHost);
                    cudaFree(dx); cudaFree(dq); cudaFree(dxd); cudaFree(dy); cudaFree(dw);
                    // host
                    std::vector<int8_t> hq(x.size());
                    std::vector<float> hd(x.size() / 32), hy((size_t) ncol * n, -1.0f);
                    k31::cpu::q8(x.data(), k, ncol, hq.data(), hd.data());
                    const bool qsame = std::memcmp(hq.data(), gq.data(), hq.size()) == 0 && std::memcmp(hd.data(), gd.data(), hd.size() * 4) == 0;
                    // rows in two halves (the lane splits rows between threads)
                    const int mid = n / 2;
                    k31::cpu::rows(type, w.data(), ld, k, 0, mid, gq.data(), gd.data(), ncol, hy.data(), (size_t) n);
                    k31::cpu::rows(type, w.data(), ld, k, mid, n, gq.data(), gd.data(), ncol, hy.data() + mid, (size_t) n);
                    int diff = 0;
                    for (size_t i = 0; i < hy.size(); ++i) diff += std::memcmp(&hy[i], &gy[i], 4) != 0;
                    if (!qsame || diff) {
                        ++bad;
                        std::printf("type %d k %d n %d ncol %d scale %g: q8 %s, %d of %zu rows differ (e.g. %.9g vs %.9g)\n", type, k, n, ncol,
                                    scale, qsame ? "same" : "DIFFERS", diff, hy.size(), hy[0], gy[0]);
                    }
                }
            }
        }
    }
    }
    std::printf("k31_cpu_lane_test (%s): %d of %d cases bit for bit\n", k31::cpu::form(), cases - bad, cases);
    return bad ? 1 : 0;
}
