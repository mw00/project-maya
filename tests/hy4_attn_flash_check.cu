// tests/hy4_attn_flash_check.cu - maya-hy4's prompt attention in one pass (hy4::k::attn_flash) against the token
// path's kernel (hy4::k::attention, run with HY4_ATTN_TC=0: attn2_kernel's F32 sums) on random queries and cache rows:
// F16 and Q8 rows, 32 and 64 heads, a window over a ring and lists of picks.  Built by hand: nvcc -O3 -arch=sm_70
// -std=c++17 -Iinclude -Isrc/hy4 tests/hy4_attn_flash_check.cu src/hy4/hy4_kernels.cu -o hy4_attn_flash_check
#include "hy4_kernels.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace k = hy4::k;

int main() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int kvl = 512, rot = 64, rowv = kvl + rot, T = 64, W = 512, maxk = 2048 + 4;
    int bad = 0;
    for (int q8 = 0; q8 < 2; ++q8)
        for (int H : {32, 64})
            for (int list = 0; list < 2; ++list) {
                const int ring = list ? 0 : 1024, ctx = list ? 6000 : ring, P0 = list ? 5000 : 2900;
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
                std::vector<int> pos(T), lo(T), nk(T), keys((size_t) T * maxk, 0);
                for (int t = 0; t < T; ++t) {
                    pos[t] = P0 + t;
                    if (list) {   // (sorted picks of positions up to the token's own, the token's own among them)
                        std::vector<int> pk;
                        for (int p = 0; p <= pos[t]; ++p)
                            if (p == pos[t] || rng() % 3 == 0) pk.push_back(p);
                        if ((int) pk.size() > 2048) pk.erase(pk.begin(), pk.end() - 2048);
                        nk[t] = (int) pk.size();
                        std::copy(pk.begin(), pk.end(), keys.begin() + (size_t) t * maxk);
                    } else {
                        lo[t] = std::max(0, pos[t] - W + 1);
                        nk[t] = pos[t] - lo[t] + 1;
                    }
                }
                std::vector<float> sinks(H);
                for (auto& x : sinks) x = nd(rng);
                uint16_t* dc;
                float *dq, *dws, *dout, *dref, *dsink;
                int *dk, *dlo, *dnk, *dpos;
                const size_t wsf = k::attn_ws_floats(T, H, kvl, list ? maxk : W);
                cudaMalloc(&dc, cache.size());
                cudaMalloc(&dq, q.size() * 4);
                cudaMalloc(&dws, wsf * 4);
                cudaMalloc(&dout, (size_t) T * H * kvl * 4);
                cudaMalloc(&dref, (size_t) T * H * kvl * 4);
                cudaMalloc(&dsink, H * 4);
                cudaMalloc(&dk, keys.size() * 4);
                cudaMalloc(&dlo, T * 4);
                cudaMalloc(&dnk, T * 4);
                cudaMalloc(&dpos, T * 4);
                cudaMemcpy(dc, cache.data(), cache.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
                cudaMemcpy(dsink, sinks.data(), H * 4, cudaMemcpyHostToDevice);
                cudaMemcpy(dk, keys.data(), keys.size() * 4, cudaMemcpyHostToDevice);
                cudaMemcpy(dlo, lo.data(), T * 4, cudaMemcpyHostToDevice);
                cudaMemcpy(dnk, nk.data(), T * 4, cudaMemcpyHostToDevice);
                cudaMemcpy(dpos, pos.data(), T * 4, cudaMemcpyHostToDevice);
                const float scale = 1.0f / std::sqrt(192.0f);
                k::attention(dq, T, H, kvl, rot, dc, ring, list ? dk : nullptr, maxk, dlo, dnk, list ? maxk : W, scale,
                             dsink, dws, dref, 0, q8 != 0, k::Q8Out{});
                const bool took = k::attn_flash(dq, T, H, kvl, rot, dc, ring, list ? dk : nullptr, maxk, dnk, dpos,
                                                list ? 0 : W, scale, dsink, dout, 0, q8 != 0);
                std::vector<float> a((size_t) T * H * kvl), b(a.size());
                cudaMemcpy(a.data(), dout, a.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), dref, b.size() * 4, cudaMemcpyDeviceToHost);
                double d2 = 0, r2 = 0, mx = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    const double d = (double) a[i] - b[i];
                    d2 += d * d;
                    r2 += (double) b[i] * b[i];
                    mx = std::max(mx, std::fabs(d));
                }
                const double rel = std::sqrt(d2 / std::max(r2, 1e-30));
                const bool ok = took && rel < 3e-3;
                bad += !ok;
                std::printf("%s %2d heads %s: %s relative RMS %.2e, largest %.2e %s\n", q8 ? "Q8 " : "F16", H,
                            list ? "picks " : "window", took ? "" : "(not taken)", rel, mx, ok ? "ok" : "BAD");
                cudaFree(dc), cudaFree(dq), cudaFree(dws), cudaFree(dout), cudaFree(dref), cudaFree(dsink);
                cudaFree(dk), cudaFree(dlo), cudaFree(dnk), cudaFree(dpos);
            }
    // timing: a window layer's 1024 tokens x 64 heads, and an indexed layer's 1024 tokens x 32 heads over 2048 picks
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    std::printf("%s\n", bad || e != cudaSuccess ? "FAILED" : "PASSED");
    return bad || e != cudaSuccess ? 1 : 0;
}
