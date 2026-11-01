// q35_mmt micro benchmark: one N x K matrix of each GGUF type (random blocks with sane scales), M = 1..8 rows,
// the time a call and the bandwidth; and a check of every M against a CPU dequant reference.
// build: nvcc -arch=sm_70 -O3 -I include tests/q35/mmt_bench.cu src/kernels/cuda/q35_mmt.cu -o mmt_bench
#include "strata/kernels/q35_kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static float h2f(uint16_t h) { __half x; memcpy(&x, &h, 2); return __half2float(x); }
static uint16_t f2h(float f) { __half x = __float2half(f); uint16_t h; memcpy(&h, &x, 2); return h; }

static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}
static void deq(int type, const uint8_t* b, float* y) {   // one 256-value super-block
    if (type == 12 || type == 13) {
        const float d = h2f(b[0] | b[1] << 8), dmin = h2f(b[2] | b[3] << 8);
        const uint8_t* sc = b + 4; const uint8_t* qh = b + 16; const uint8_t* ql = b + (type == 12 ? 16 : 48);
        int is = 0; uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t s, m; get_scale_min_k4(is, sc, &s, &m); const float d1 = d * s, m1 = dmin * m;
            get_scale_min_k4(is + 1, sc, &s, &m); const float d2 = d * s, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (type == 13 && (qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (type == 13 && (qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    } else if (type == 14) {
        const uint8_t* ql = b; const uint8_t* qh = b + 128; const int8_t* sc = (const int8_t*) (b + 192);
        const float d = h2f(b[208] | b[209] << 8);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32, q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32, q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * sc[is] * q1; y[l + 32] = d * sc[is + 2] * q2; y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    } else {
        for (int bb = 0; bb < 8; ++bb) {
            const uint8_t* p = b + 34 * bb; const float d = h2f(p[0] | p[1] << 8);
            for (int l = 0; l < 32; ++l) *y++ = d * (int8_t) p[2 + l];
        }
    }
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 17408, K = argc > 2 ? atoi(argv[2]) : 5120;
    std::mt19937 rng(1);
    for (int type : {12, 13, 14, 8}) {
        const int bytes = type == 12 ? 144 : type == 13 ? 176 : type == 14 ? 210 : 272;
        const size_t nb = (size_t) N * (K / 256);
        std::vector<uint8_t> W(nb * bytes + 64);
        for (auto& v : W) v = rng() & 0xff;
        for (size_t i = 0; i < nb; ++i) {   // sane scales
            uint8_t* b = W.data() + i * bytes;
            if (type == 12 || type == 13) { uint16_t d = f2h(0.002f), m = f2h(0.001f); memcpy(b, &d, 2); memcpy(b + 2, &m, 2); }
            else if (type == 14) { uint16_t d = f2h(0.0005f); memcpy(b + 208, &d, 2); }
            else for (int bb = 0; bb < 8; ++bb) { uint16_t d = f2h(0.002f); memcpy(b + 34 * bb, &d, 2); }
        }
        std::vector<float> X(8 * (size_t) K);
        for (auto& v : X) v = (float) ((int) (rng() % 2001) - 1000) / 500.0f;
        uint8_t* dW; float *dX, *dY, *dP; uint16_t* dXh;
        cudaMalloc(&dW, W.size()); cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dX, X.size() * 4); cudaMemcpy(dX, X.data(), X.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&dY, 8 * (size_t) N * 4); cudaMalloc(&dXh, 8 * (size_t) K * 2);
        cudaMalloc(&dP, strata::kernels::q35_mmt_part_floats(N) * 4);
        cudaStream_t s; cudaStreamCreate(&s);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        // reference for M = 8 on a sample of rows
        std::vector<float> row(K), Y(8 * (size_t) N);
        strata::kernels::q35_mmt_prep(dX, K, 8, dXh, s);
        strata::kernels::q35_mmt(type, dW, dXh, dY, K, N, 8, dP, s);
        cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost);
        double maxerr = 0, maxref = 0;
        for (int n = 0; n < N; n += 97) {
            for (int sb = 0; sb < K / 256; ++sb) deq(type, W.data() + ((size_t) n * (K / 256) + sb) * bytes, row.data() + sb * 256);
            for (int t = 0; t < 8; ++t) {
                double r = 0; for (int k = 0; k < K; ++k) r += (double) row[k] * (double) h2f(f2h(X[(size_t) t * K + k]));
                maxerr = fmax(maxerr, fabs(r - Y[(size_t) t * N + n])); maxref = fmax(maxref, fabs(r));
            }
        }
        printf("type %2d  N %d K %d: max err %.3g (max |y| %.3g)\n", type, N, K, maxerr, maxref);
        for (int M : {1, 2, 4, 8}) {
            strata::kernels::q35_mmt_prep(dX, K, M, dXh, s);
            for (int i = 0; i < 3; ++i) strata::kernels::q35_mmt(type, dW, dXh, dY, K, N, M, dP, s);
            cudaEventRecord(e0, s);
            const int reps = 50;
            for (int i = 0; i < reps; ++i) strata::kernels::q35_mmt(type, dW, dXh, dY, K, N, M, dP, s);
            cudaEventRecord(e1, s); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            printf("   M %d: %8.1f us  %6.0f GB/s\n", M, 1000 * ms / reps, nb * bytes / (ms / reps * 1e-3) / 1e9);
        }
        {   // the packed layout
            void* dQ; cudaMalloc(&dQ, strata::kernels::q35_mmt_packed_bytes(type, K, N));
            strata::kernels::q35_mmt_pack(type, dW, dQ, K, N, s);
            strata::kernels::q35_mmt_prep(dX, K, 8, dXh, s);
            strata::kernels::q35_mmtp(type, dQ, dXh, dY, K, N, 8, dP, s);
            std::vector<float> Y2(8 * (size_t) N);
            cudaMemcpy(Y2.data(), dY, Y2.size() * 4, cudaMemcpyDeviceToHost);
            double d = 0;
            for (size_t i = 0; i < Y2.size(); ++i) d = fmax(d, fabs(Y2[i] - Y[i]));
            printf("   packed: max diff to unpacked %.3g  %s\n", d, cudaGetErrorString(cudaGetLastError()));
            for (int M : {1, 2, 4, 8}) {
                strata::kernels::q35_mmt_prep(dX, K, M, dXh, s);
                for (int i = 0; i < 3; ++i) strata::kernels::q35_mmtp(type, dQ, dXh, dY, K, N, M, dP, s);
                cudaEventRecord(e0, s);
                for (int i = 0; i < 50; ++i) strata::kernels::q35_mmtp(type, dQ, dXh, dY, K, N, M, dP, s);
                cudaEventRecord(e1, s); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                printf("   packed M %d: %8.1f us  %6.0f GB/s", M, 1000 * ms / 50, nb * bytes / (ms / 50 * 1e-3) / 1e9);
                for (int pr = 1; pr <= 2; ++pr) {   // the loads alone / the math alone
                    strata::kernels::q35_mmtp_set_probe(pr);
                    strata::kernels::q35_mmtp(type, dQ, dXh, dY, K, N, M, dP, s);
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < 50; ++i) strata::kernels::q35_mmtp(type, dQ, dXh, dY, K, N, M, dP, s);
                    cudaEventRecord(e1, s); cudaEventSynchronize(e1);
                    float mp; cudaEventElapsedTime(&mp, e0, e1);
                    printf("  %s %7.1f us", pr == 1 ? "loads" : "math", 1000 * mp / 50);
                }
                strata::kernels::q35_mmtp_set_probe(0);
                printf("\n");
                printf("     splits:");   // (the heuristic's choice against a sweep)
                for (int sp : {1, 2, 3, 4, 5, 6, 8, 10, 16}) {
                    strata::kernels::q35_mmtp_set_probe(0, sp);
                    strata::kernels::q35_mmtp(type, dQ, dXh, dY, K, N, M, dP, s);
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < 50; ++i) strata::kernels::q35_mmtp(type, dQ, dXh, dY, K, N, M, dP, s);
                    cudaEventRecord(e1, s); cudaEventSynchronize(e1);
                    float mp; cudaEventElapsedTime(&mp, e0, e1);
                    printf("  %d: %.1f", sp, 1000 * mp / 50);
                }
                strata::kernels::q35_mmtp_set_probe(0, 0);
                printf(" us\n");
            }
            cudaFree(dQ);
        }
        cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaFree(dXh); cudaFree(dP);
    }
    return 0;
}
