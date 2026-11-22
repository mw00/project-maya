// tests/glm_mv_bench.cu - the fast path's GEMV (maya::kernels::glmf::mv) with several activation columns: each
// column's result against the single-column call (bit for bit), and the time of 1 - 8 and 16 columns.
//
//   glm_mv_bench [n_in n_out]      (default 4096 8192, the KDA projections' shape)
#include "maya/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace gf = maya::kernels::glmf;

namespace {

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// random raw blocks of a ggml type, with sane scales (the d / dmin fields as small FP16 values)
std::vector<uint8_t> random_rows(int type, int n_in, int n_out) {
    const size_t rb = gf::row_bytes(type, n_in);
    std::vector<uint8_t> w(rb * (size_t) n_out);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) mix(i);
    const auto put_half = [](uint8_t* p, float f) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        const uint16_t h = (uint16_t) (((u >> 16) & 0x8000) | ((((u >> 23) & 0xff) - 112) << 10) | ((u >> 13) & 0x3ff));
        std::memcpy(p, &h, 2);
    };
    for (int r = 0; r < n_out; ++r) {
        uint8_t* row = w.data() + rb * (size_t) r;
        if (type == 8)   // Q8_0: d + 32 int8
            for (int b = 0; b < n_in / 32; ++b) put_half(row + b * 34, 0.01f);
        if (type == 12)   // Q4_K: d, dmin first
            for (int b = 0; b < n_in / 256; ++b) put_half(row + b * 144, 0.01f), put_half(row + b * 144 + 2, 0.005f);
        if (type == 13)   // Q5_K: d, dmin first
            for (int b = 0; b < n_in / 256; ++b) put_half(row + b * 176, 0.01f), put_half(row + b * 176 + 2, 0.005f);
        if (type == 14)   // Q6_K: d last
            for (int b = 0; b < n_in / 256; ++b) put_half(row + b * 210 + 208, 0.01f);
        if (type == gf::kTypeBF16)
            for (int i = 0; i < n_in; ++i) {
                const float f = (float) ((int) (mix((uint64_t) r * n_in + i) % 2001) - 1000) * 1e-5f;
                uint32_t u;
                std::memcpy(&u, &f, 4);
                const uint16_t b = (uint16_t) (u >> 16);
                std::memcpy(row + 2 * i, &b, 2);
            }
    }
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    const int n_in = argc > 2 ? std::atoi(argv[1]) : 4096, n_out = argc > 2 ? std::atoi(argv[2]) : 8192;
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int C = gf::kMaxMvCols;
    std::vector<float> xh((size_t) C * n_in);
    for (size_t i = 0; i < xh.size(); ++i) xh[i] = (float) ((int) (mix(0x5000 + i) % 2001) - 1000) * 1e-3f;
    float *x = nullptr, *y = nullptr, *y1 = nullptr;
    void* xq = nullptr;
    const size_t xq_s = (size_t) n_in / 32 * 36;
    cudaMalloc(&x, xh.size() * 4);
    cudaMalloc(&xq, (size_t) C * xq_s);
    cudaMalloc(&y, (size_t) C * n_out * 4);
    cudaMalloc(&y1, (size_t) C * n_out * 4);
    cudaMemcpy(x, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice);
    gf::quantize_q8_1(x, xq, C * n_in, s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    bool all_ok = true;
    // MVB_APPROX=1: the several-column jobs as MvJob::approx (the tensor-core Q4_K path): their error against the
    // single column's, relative to the largest result
    const bool approx = getenv("MVB_APPROX") != nullptr && std::atoi(getenv("MVB_APPROX")) != 0;
    for (const int type : {8, 12, 13, 14, gf::kTypeBF16}) {
        const std::vector<uint8_t> wh = random_rows(type, n_in, n_out);
        void* w = nullptr;
        cudaMalloc(&w, wh.size());
        cudaMemcpy(w, wh.data(), wh.size(), cudaMemcpyHostToDevice);
        const auto job = [&](int ncol, float* out) {
            gf::MvJob j{w, xq, x, out, nullptr, 1.0f, type, n_in, n_out};
            j.ncol = ncol;
            j.xq_cstride = (int64_t) xq_s;
            j.xf_cstride = n_in;
            j.y_cstride = n_out;
            j.approx = approx && ncol > 1;
            return j;
        };
        // the reference: one column at a time
        for (int c = 0; c < C; ++c) {
            gf::MvJob j = job(1, y1 + (size_t) c * n_out);
            j.xq = (const uint8_t*) xq + (size_t) c * xq_s;
            j.xf = x + (size_t) c * n_in;
            gf::mv(&j, 1, s);
        }
        std::vector<float> r1((size_t) C * n_out), rc((size_t) C * n_out);
        cudaMemcpy(r1.data(), y1, r1.size() * 4, cudaMemcpyDeviceToHost);
        std::printf("type %2d, %d x %d (%.1f MB):", type, n_out, n_in, (double) wh.size() / 1048576.0);
        for (const int nc : {1, 2, 3, 4, 5, 6, 7, 8, 16}) {
            gf::MvJob j = job(nc, y);
            cudaMemset(y, 0, (size_t) C * n_out * 4);
            gf::mv(&j, 1, s);
            cudaMemcpy(rc.data(), y, rc.size() * 4, cudaMemcpyDeviceToHost);
            // bit for bit, or (the tensor-core Q8_0 path sums in another order) within float rounding of the scale
            const bool same = std::memcmp(rc.data(), r1.data(), (size_t) nc * n_out * 4) == 0;
            double md = 0, mx = 0;
            for (size_t i = 0; i < (size_t) nc * n_out; ++i) {
                md = std::max(md, (double) std::fabs(rc[i] - r1[i]));
                mx = std::max(mx, (double) std::fabs(r1[i]));
            }
            const bool close = md <= (approx ? 1e-2 : 1e-5) * mx;
            all_ok = all_ok && close;
            if (approx && !same) std::printf(" [%.1e]", md / std::max(mx, 1e-30));
            for (int i = 0; i < 5; ++i) gf::mv(&j, 1, s);
            cudaEventRecord(e0, s);
            const int reps = 50;
            for (int i = 0; i < reps; ++i) gf::mv(&j, 1, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            std::printf("  %2d cols %7.1f us%s", nc, 1000.0 * ms / reps, same ? "" : close ? "~" : " (DIFFERS)");
        }
        std::printf("\n");
        cudaFree(w);
    }
    std::printf("%s\n", all_ok ? "PASS: every column equals the single-column GEMV (~: to float rounding)" : "FAIL");
    return all_ok ? 0 : 1;
}
