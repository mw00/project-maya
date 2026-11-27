// tests/iq_set_bench.cu - ds4_iq's prompt expert sets (iq::set_gate_up / set_down) alone on Hy4-flash's routed
// experts (1536 x 1536, IQ2_XXS and IQ3_XXS): time, TFLOP/s and a checksum by rows an expert (random blocks, no
// model).  Built by hand: nvcc -O3 -std=c++17 -arch=sm_70 -DMAYA_DS4_IQ_ONLY -DDS4_IQ_MIN_FORMS -Iinclude -Isrc
// -I<ggml>/src -I<ggml>/include tests/iq_set_bench.cu src/ds4/ds4_iq.cu -o iq_set_bench
#include "ds4/ds4_iq.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace iq = maya::ds4::gguf::iq;

int main() {
    const int lat = 1536, ff = 1536, ne = 32, reps = 20;
    std::mt19937 rng(5);
    for (int type : {iq::kIQ2_XXS, iq::kIQ3_XXS}) {
        const size_t gb = iq::row_bytes(type, lat) * ff, db = iq::row_bytes(type, ff) * lat;
        auto al = [](size_t b) { return (b + 255) / 256 * 256; };
        const size_t tr1 = al(gb), tr2 = 2 * al(gb), blob = al(2 * al(gb) + db);
        std::vector<uint8_t> h(blob * ne);
        for (auto& b : h) b = (uint8_t) rng();
        // (each block's d a small F16 - its first 2 bytes in both types)
        const int bb = (int) (iq::row_bytes(type, 256));
        for (size_t o = 0; o + bb <= h.size(); o += bb) {
            const __half d = __float2half(1e-4f * (float) (rng() % 100 + 1));
            std::memcpy(&h[o], &d, 2);
        }
        uint8_t* w;
        cudaMalloc(&w, h.size());
        cudaMemcpy(w, h.data(), h.size(), cudaMemcpyHostToDevice);
        for (int rows : {64, 128, 256, 512}) {
            const int nrow = rows * ne;
            std::vector<int> bounds(ne + 1), idx(nrow);
            for (int e = 0; e <= ne; ++e) bounds[e] = e * rows;
            for (int i = 0; i < nrow; ++i) idx[i] = (int) (rng() % (unsigned) nrow);
            std::vector<float> xf((size_t) nrow * lat);
            for (auto& x : xf) x = (float) ((int) (rng() % 2001) - 1000) * 1e-3f;
            int *db_, *di;
            float *dx, *dy;
            uint16_t* dh;
            cudaMalloc(&db_, sizeof(int) * (ne + 1));
            cudaMalloc(&di, sizeof(int) * nrow);
            cudaMalloc(&dx, sizeof(float) * xf.size());
            cudaMalloc(&dh, sizeof(uint16_t) * (size_t) nrow * ff);
            cudaMalloc(&dy, sizeof(float) * (size_t) nrow * lat);
            cudaMemcpy(db_, bounds.data(), sizeof(int) * (ne + 1), cudaMemcpyHostToDevice);
            cudaMemcpy(di, idx.data(), sizeof(int) * nrow, cudaMemcpyHostToDevice);
            cudaMemcpy(dx, xf.data(), sizeof(float) * xf.size(), cudaMemcpyHostToDevice);
            iq::SetArgs a;
            a.wbase = w;
            a.stride = blob;
            a.tr[0] = 0;
            a.tr[1] = tr1;
            a.tr[2] = tr2;
            a.bounds = db_;
            a.max_rows = rows;
            a.n_exp = ne;
            a.n_embd = lat;
            a.n_ff = ff;
            a.limit = 10.0f;
            a.xf = dx;
            a.idx = di;
            a.ldx = lat;
            a.h = dh;
            a.y = dy;
            a.tg = a.td = type;
            float tms[2] = {0, 0};
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            for (int pass = 0; pass < 2; ++pass) {   // gate / up, then down
                iq::SetArgs b = a;
                if (pass == 1) {
                    b.xh = dh;
                    b.xf = nullptr;
                    b.idx = nullptr;
                    b.ldx = ff;
                    b.ldo = lat;
                } else {
                    b.ldo = ff;
                }
                auto run = [&] { return pass == 0 ? iq::set_gate_up(b, 0) : iq::set_down(b, 0); };
                if (!run()) {
                    std::printf("set refused\n");
                    return 1;
                }
                cudaEventRecord(e0);
                for (int r = 0; r < reps; ++r) run();
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&tms[pass], e0, e1);
                tms[pass] /= reps;
            }
            {   // the same work as FP16 GEMMs: each expert's three matrices decoded to F16, gate|up then down
                cublasHandle_t bl;
                cublasCreate(&bl);
                uint16_t *wf, *xh, *gu;
                float* yo;
                cudaMalloc(&wf, sizeof(uint16_t) * (size_t) 3 * lat * ff);
                cudaMalloc(&xh, sizeof(uint16_t) * (size_t) rows * lat);
                cudaMalloc(&gu, sizeof(uint16_t) * (size_t) rows * 2 * ff);
                cudaMalloc(&yo, sizeof(float) * (size_t) rows * lat);
                cudaMemset(xh, 0, sizeof(uint16_t) * (size_t) rows * lat);
                const __half one = __float2half(1.0f), zero = __float2half(0.0f);
                const float onef = 1.0f, zerof = 0.0f;
                auto run = [&] {
                    for (int e = 0; e < ne; ++e) {
                        const uint8_t* b = w + (size_t) e * blob;
                        iq::f16_rows(type, b, lat, ff, wf, lat, 0);
                        iq::f16_rows(type, b + tr1, lat, ff, wf + (size_t) ff * lat, lat, 0);
                        iq::f16_rows(type, b + tr2, ff, lat, wf + (size_t) 2 * ff * lat, ff, 0);
                        cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N, 2 * ff, rows, lat, &one, wf, CUDA_R_16F, lat, xh,
                                     CUDA_R_16F, lat, &zero, gu, CUDA_R_16F, 2 * ff, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
                        cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N, lat, rows, ff, &onef, wf + (size_t) 2 * ff * lat,
                                     CUDA_R_16F, ff, gu, CUDA_R_16F, 2 * ff, &zerof, yo, CUDA_R_32F, lat, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
                    }
                };
                run();
                cudaEventRecord(e0);
                for (int r = 0; r < reps; ++r) run();
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms = 0, msd = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                cudaEventRecord(e0);
                for (int r = 0; r < reps; ++r)
                    for (int e = 0; e < ne; ++e) {
                        const uint8_t* b = w + (size_t) e * blob;
                        iq::f16_rows(type, b, lat, ff, wf, lat, 0);
                        iq::f16_rows(type, b + tr1, lat, ff, wf + (size_t) ff * lat, lat, 0);
                        iq::f16_rows(type, b + tr2, ff, lat, wf + (size_t) 2 * ff * lat, ff, 0);
                    }
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&msd, e0, e1);
                std::printf("   FP16 GEMMs: %.3f ms (decode %.3f ms) vs the tiles %.3f ms\n", ms / reps, msd / reps,
                            tms[0] + tms[1]);
                cudaFree(wf), cudaFree(xh), cudaFree(gu), cudaFree(yo);
                cublasDestroy(bl);
            }
            std::vector<float> y((size_t) nrow * lat);
            cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost);
            double sum = 0;
            for (float v : y) sum += v;
            const double fl = 2.0 * nrow * (2.0 * lat * ff + (double) ff * lat);
            std::printf("%s %3d rows x %d experts: gate/up %7.3f ms, down %7.3f ms, %5.2f TFLOP/s (checksum %.6e)\n",
                        type == iq::kIQ2_XXS ? "IQ2_XXS" : "IQ3_XXS", rows, ne, tms[0], tms[1],
                        fl / ((tms[0] + tms[1]) * 1e-3) / 1e12, sum);
            cudaFree(db_), cudaFree(di), cudaFree(dx), cudaFree(dh), cudaFree(dy);
        }
        cudaFree(w);
    }
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 0;
}
