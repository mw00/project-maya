// tests/iq_mv_bench.cu - ds4_iq's Q6_K and Q8_0 GEMVs (iq::mv) alone on Hy4-flash's shapes (its attention, the DFly2
// drafter's, the shared expert's): time and GB/s by column count (random blocks, no model).  Built by hand: nvcc -O3 -arch=sm_70 -DMAYA_DS4_IQ_ONLY -DDS4_IQ_MIN_FORMS -Iinclude
// -Isrc tests/iq_mv_bench.cu src/ds4/ds4_iq.cu -o iq_mv_bench; MAYA_IQ_MV_ROWS etc. as for the engine.
#include "ds4/ds4_iq.cuh"

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace iq = maya::ds4::gguf::iq;

int main() {
    struct Shape {
        const char* name;
        int type, k, n;
    } shapes[] = {{"Q6_K attn_gate 3072-8192", iq::kQ6_K, 3072, 8192}, {"Q6_K attn_output 8192-3072", iq::kQ6_K, 8192, 3072},
                  {"Q6_K attn_q_b 1024-8192", iq::kQ6_K, 1024, 8192}, {"Q8_0 dfly o_proj 8192-3072", iq::kQ8_0, 8192, 3072},
                  {"Q8_0 dfly gate 3072-9216", iq::kQ8_0, 3072, 9216}, {"Q8_0 dfly down 9216-3072", iq::kQ8_0, 9216, 3072},
                  {"Q8_0 shexp gate 3072-1536", iq::kQ8_0, 3072, 1536}};
    std::mt19937 rng(1);
    const int reps = 200, kMaxC = 8;
    for (const Shape& sh : shapes) {
        const bool q6 = sh.type == iq::kQ6_K;
        const size_t ld = q6 ? (size_t) sh.k / 256 * 210 : (size_t) sh.k / 32 * 34, bytes = ld * sh.n;
        std::vector<uint8_t> h(bytes);
        for (auto& b : h) b = (uint8_t) rng();
        for (size_t r = 0; r < (size_t) sh.n; ++r)   // (each block's d a small F16: no infinities in the sums)
            for (int b = 0; b < sh.k / (q6 ? 256 : 32); ++b) {
                const __half d = __float2half(1e-3f * (float) (rng() % 1000 + 1));
                std::memcpy(&h[r * ld + (size_t) b * (q6 ? 210 : 34) + (q6 ? 208 : 0)], &d, 2);
            }
        // (as many copies as fill 256 MB: each rep reads a matrix the L2 does not hold)
        const int copies = (int) std::max<size_t>(1, ((size_t) 256 << 20) / bytes);
        uint8_t* w = nullptr;
        cudaMalloc(&w, bytes * copies);
        for (int c = 0; c < copies; ++c) cudaMemcpy(w + bytes * c, h.data(), bytes, cudaMemcpyHostToDevice);
        int8_t* xq = nullptr;
        float *xd = nullptr, *y = nullptr;
        cudaMalloc(&xq, (size_t) kMaxC * sh.k);
        cudaMalloc(&xd, sizeof(float) * kMaxC * sh.k / 32);
        cudaMalloc(&y, sizeof(float) * kMaxC * sh.n);
        std::vector<int8_t> hq((size_t) kMaxC * sh.k);
        for (auto& q : hq) q = (int8_t) (rng() % 255 - 127);
        std::vector<float> hd((size_t) kMaxC * sh.k / 32, 0.01f);
        cudaMemcpy(xq, hq.data(), hq.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(xd, hd.data(), sizeof(float) * hd.size(), cudaMemcpyHostToDevice);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int nc : {1, 2, 4, 6, 8}) {
            iq::MvJob j;
            j.ld = ld;
            j.type = sh.type;
            j.k = sh.k;
            j.n = sh.n;
            j.ncol = nc;
            j.xq = xq;
            j.xd = xd;
            j.y = y;
            j.ycs = (size_t) sh.n;
            for (int r = 0; r < 10; ++r) {
                j.w = w + bytes * (r % copies);
                iq::mv(&j, 1, 0);
            }
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) {
                j.w = w + bytes * (r % copies);
                if (!iq::mv(&j, 1, 0)) {
                    std::printf("mv refused\n");
                    return 1;
                }
            }
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = 1e3 * ms / reps;
            std::vector<float> out((size_t) sh.n);
            cudaMemcpy(out.data(), y, sizeof(float) * sh.n, cudaMemcpyDeviceToHost);
            double sum = 0;
            for (float v : out) sum += v;
            std::printf("%-28s %d col: %7.1f us  %6.0f GB/s  (checksum %.6e)\n", sh.name, nc, us, bytes / us / 1e3, sum);
        }
        cudaFree(w);
        cudaFree(xq);
        cudaFree(xd);
        cudaFree(y);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 0;
}
