// tests/iq_gu_bench.cu - ds4_iq's token-window experts (iq::gate_up / iq::down, the grouped form) alone on Hy4-flash's
// routed experts (1536 x 1536, IQ2_XXS and IQ3_XXS): G experts of NE entries each, time and a checksum (random blocks,
// no model).  Built by hand: nvcc -O3 -std=c++17 -arch=sm_70 -DMAYA_DS4_IQ_ONLY -DDS4_IQ_MIN_FORMS -Iinclude -Isrc
// -I<ggml>/src -I<ggml>/include tests/iq_gu_bench.cu src/ds4/ds4_iq.cu -o iq_gu_bench
#include "ds4/ds4_iq.cuh"

#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace iq = maya::ds4::gguf::iq;

int main() {
    const int lat = 1536, ff = 1536, reps = 200;
    std::mt19937 rng(7);
    for (int type : {iq::kIQ2_XXS, iq::kIQ3_XXS}) {
        const size_t gb = iq::row_bytes(type, lat) * ff, db = iq::row_bytes(type, ff) * lat;
        auto al = [](size_t b) { return (b + 255) / 256 * 256; };
        const size_t tr1 = al(gb), tr2 = 2 * al(gb), blob = al(2 * al(gb) + db);
        const int nblob = 512;   // (a layer's experts: more than L2 holds)
        std::vector<uint8_t> h(blob * nblob);
        for (auto& b : h) b = (uint8_t) rng();
        const int bb = (int) iq::row_bytes(type, 256);
        for (size_t o = 0; o + bb <= h.size(); o += bb) {
            const __half d = __float2half(1e-4f * (float) (rng() % 100 + 1));
            std::memcpy(&h[o], &d, 2);
        }
        uint8_t* w;
        cudaMalloc(&w, h.size());
        cudaMemcpy(w, h.data(), h.size(), cudaMemcpyHostToDevice);
        for (int G : {16, 32, 48}) {
            for (int NE : {1, 2, 4}) {
                const int ent = 16 * 16;   // (entries' ids up to 16 tokens x 16)
                std::vector<unsigned long long> gp(G);
                std::vector<int> gs(G + 1), ge(G * NE);
                for (int g = 0; g < G; ++g) {
                    gp[g] = (unsigned long long) (w + (size_t) ((g * 37) % nblob) * blob);
                    gs[g] = g * NE;
                    for (int i = 0; i < NE; ++i) ge[g * NE + i] = (g * NE + i) % ent;
                }
                gs[G] = G * NE;
                unsigned long long* dgp;
                int *dgs, *dge, *dng;
                cudaMalloc(&dgp, 8 * G);
                cudaMalloc(&dgs, 4 * (G + 1));
                cudaMalloc(&dge, 4 * G * NE);
                cudaMalloc(&dng, 4);
                cudaMemcpy(dgp, gp.data(), 8 * G, cudaMemcpyHostToDevice);
                cudaMemcpy(dgs, gs.data(), 4 * (G + 1), cudaMemcpyHostToDevice);
                cudaMemcpy(dge, ge.data(), 4 * G * NE, cudaMemcpyHostToDevice);
                cudaMemcpy(dng, &G, 4, cudaMemcpyHostToDevice);
                int8_t *xq, *hq;
                float *xd, *hd, *hh, *part;
                const int xrows = ent / 8 + 1;
                cudaMalloc(&xq, (size_t) xrows * lat);
                cudaMalloc(&xd, sizeof(float) * xrows * lat / 32);
                cudaMalloc(&hq, (size_t) ent * ff);
                cudaMalloc(&hd, sizeof(float) * ent * ff / 32);
                cudaMalloc(&hh, sizeof(float) * ent * ff);
                cudaMalloc(&part, sizeof(float) * ent * lat);
                std::vector<int8_t> hx((size_t) xrows * lat);
                for (auto& x : hx) x = (int8_t) (rng() % 255 - 127);
                cudaMemcpy(xq, hx.data(), hx.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(hq, hx.data(), std::min(hx.size(), (size_t) ent * ff), cudaMemcpyHostToDevice);
                std::vector<float> d1((size_t) std::max(xrows * lat / 32, ent * ff / 32), 0.01f);
                cudaMemcpy(xd, d1.data(), sizeof(float) * xrows * lat / 32, cudaMemcpyHostToDevice);
                cudaMemcpy(hd, d1.data(), sizeof(float) * ent * ff / 32, cudaMemcpyHostToDevice);
                cudaMemset(hh, 0, sizeof(float) * ent * ff);
                cudaMemset(part, 0, sizeof(float) * ent * lat);
                iq::Args a;
                a.grp_ptr = dgp;
                a.grp_start = dgs;
                a.grp_ent = dge;
                a.n_groups = dng;
                a.tr[0] = 0;
                a.tr[1] = tr1;
                a.tr[2] = tr2;
                a.n_embd = lat;
                a.n_ff = ff;
                a.limit = 10.0f;
                a.xq = xq;
                a.xd = xd;
                a.hq = hq;
                a.hd = hd;
                a.h = hh;
                a.part = part;
                a.tg = a.td = type;
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                float t[2];
                for (int pass = 0; pass < 2; ++pass) {
                    auto run = [&] { return pass == 0 ? iq::gate_up(a, G, NE, 0) : iq::down(a, G, NE, 0); };
                    if (!run()) {
                        std::printf("refused\n");
                        return 1;
                    }
                    cudaEventRecord(e0);
                    for (int r = 0; r < reps; ++r) run();
                    cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    cudaEventElapsedTime(&t[pass], e0, e1);
                    t[pass] *= 1e3f / reps;
                }
                std::vector<float> o((size_t) ent * ff), q((size_t) ent * lat);
                cudaMemcpy(o.data(), hh, sizeof(float) * o.size(), cudaMemcpyDeviceToHost);
                cudaMemcpy(q.data(), part, sizeof(float) * q.size(), cudaMemcpyDeviceToHost);
                double s1 = 0, s2 = 0;
                for (float v : o) s1 += v;
                for (float v : q) s2 += v;
                const double gbs = (double) G * 2 * gb / (t[0] * 1e-6) / 1e9, dbs = (double) G * db / (t[1] * 1e-6) / 1e9;
                std::printf("%s G %2d NE %d: gate_up %6.1f us (%4.0f GB/s) down %6.1f us (%4.0f GB/s) ck %.6e %.6e\n",
                            type == iq::kIQ2_XXS ? "IQ2_XXS" : "IQ3_XXS", G, NE, t[0], gbs, t[1], dbs, s1, s2);
                cudaFree(dgp), cudaFree(dgs), cudaFree(dge), cudaFree(dng), cudaFree(xq), cudaFree(xd);
                cudaFree(hq), cudaFree(hd), cudaFree(hh), cudaFree(part);
            }
        }
        cudaFree(w);
    }
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 0;
}
