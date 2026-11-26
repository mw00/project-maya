// tests/hy4_attn_tc_check.cu - maya-hy4's token-window attention (hy4::k::attention) on random queries and cache rows
// (F16 and Q8, picked keys and a window), its outputs written to a file: run once with HY4_ATTN_TC=0 (attn2_kernel)
// and once without (attn_tc_kernel), then compare the two files (argv[2] given: the compare).  Built by hand: nvcc -O3
// -arch=sm_70 -std=c++17 -Iinclude -Isrc/hy4 tests/hy4_attn_tc_check.cu src/hy4/hy4_kernels.cu -o hy4_attn_tc_check
#include "hy4_kernels.cuh"

#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = hy4::k;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s OUT [OTHER: compare OUT with it]\n", argv[0]);
        return 2;
    }
    if (argc == 3) {
        std::FILE* a = std::fopen(argv[1], "rb");
        std::FILE* b = std::fopen(argv[2], "rb");
        std::vector<float> x, y;
        float v;
        while (std::fread(&v, 4, 1, a) == 1) x.push_back(v);
        while (std::fread(&v, 4, 1, b) == 1) y.push_back(v);
        if (x.size() != y.size() || x.empty()) {
            std::printf("sizes differ: %zu %zu\n", x.size(), y.size());
            return 1;
        }
        double d2 = 0, r2 = 0, mx = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const double d = (double) x[i] - y[i];
            d2 += d * d;
            r2 += (double) y[i] * y[i];
            mx = std::max(mx, std::fabs(d));
        }
        std::printf("%zu values: relative RMS difference %.3e, largest %.3e\n", x.size(), std::sqrt(d2 / r2), mx);
        return std::sqrt(d2 / r2) < 1e-3 ? 0 : 1;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::FILE* out = std::fopen(argv[1], "wb");
    const int ctx = 4096, kvl = 512, rot = 64, rowv = kvl + rot;
    for (int q8 = 0; q8 < 2; ++q8)
        for (int H : {32, 64})
            for (int T : {1, 2, 4})
                for (int picked = 0; picked < 2; ++picked) {
                    const int nkeys = picked ? 2048 : 512;
                    const int rb = q8 ? (int) k::kv_q8_row_bytes(kvl, rot) : rowv * 2;
                    std::vector<uint8_t> cache((size_t) ctx * rb);
                    for (int r = 0; r < ctx; ++r) {
                        uint8_t* p = cache.data() + (size_t) r * rb;
                        if (!q8) {
                            for (int i = 0; i < rowv; ++i) ((__half*) p)[i] = __float2half(nd(rng));
                        } else {
                            for (int i = 0; i < kvl; ++i) p[i] = (uint8_t) (int8_t) (rng() % 255 - 127);
                            for (int i = 0; i < 16; ++i) ((__half*) (p + 512))[i] = __float2half(0.01f + 0.01f * (rng() % 4));
                            for (int i = 0; i < rot; ++i) ((__half*) (p + 544))[i] = __float2half(nd(rng));
                        }
                    }
                    std::vector<float> q((size_t) T * H * rowv);
                    for (auto& x : q) x = 0.05f * nd(rng);
                    std::vector<int> keys((size_t) T * nkeys), lo(T), nk(T);
                    for (int t = 0; t < T; ++t) {
                        nk[t] = nkeys - 7 * t;   // (a chunk partly filled)
                        lo[t] = 100 + t;
                        for (int i = 0; i < nkeys; ++i) keys[(size_t) t * nkeys + i] = (int) (rng() % ctx);
                    }
                    std::vector<float> sinks(H);
                    for (auto& x : sinks) x = nd(rng);
                    uint16_t* dc;
                    float *dq, *dws, *dout, *dsink;
                    int *dk, *dlo, *dnk;
                    const size_t wsf = k::attn_ws_floats(T, H, kvl, nkeys);
                    cudaMalloc(&dc, cache.size());
                    cudaMalloc(&dq, q.size() * 4);
                    cudaMalloc(&dws, wsf * 4);
                    cudaMalloc(&dout, (size_t) T * H * kvl * 4);
                    cudaMalloc(&dsink, H * 4);
                    cudaMalloc(&dk, keys.size() * 4);
                    cudaMalloc(&dlo, T * 4);
                    cudaMalloc(&dnk, T * 4);
                    cudaMemcpy(dc, cache.data(), cache.size(), cudaMemcpyHostToDevice);
                    cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemcpy(dsink, sinks.data(), H * 4, cudaMemcpyHostToDevice);
                    cudaMemcpy(dk, keys.data(), keys.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemcpy(dlo, lo.data(), T * 4, cudaMemcpyHostToDevice);
                    cudaMemcpy(dnk, nk.data(), T * 4, cudaMemcpyHostToDevice);
                    k::attention(dq, T, H, kvl, rot, dc, 0, picked ? dk : nullptr, nkeys, dlo, dnk, nkeys,
                                 1.0f / std::sqrt(192.0f), dsink, dws, dout, 0, q8 != 0, k::Q8Out{});
                    std::vector<float> o((size_t) T * H * kvl);
                    cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost);
                    std::fwrite(o.data(), 4, o.size(), out);
                    cudaFree(dc), cudaFree(dq), cudaFree(dws), cudaFree(dout), cudaFree(dsink), cudaFree(dk);
                    cudaFree(dlo), cudaFree(dnk);
                }
    std::fclose(out);
    const cudaError_t e = cudaDeviceSynchronize();
    std::printf("%s\n", e == cudaSuccess ? "ok" : cudaGetErrorString(e));
    return e == cudaSuccess ? 0 : 1;
}
