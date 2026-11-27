// tests/hy4_vision_test.cu - unit & GPU kernel calculation tests for Hy4's vision FP16 fallback (src/hy4/hy4_vision.cu):
// 1. IEEE-754 FP16 host conversion & NaN-preserving +/-65504 clamp (+/-Inf -> +/-65504, NaN stays NaN).
// 2. GPU bf16_to_f16_inplace + cublasGemmEx(CUDA_R_16F, CUBLAS_COMPUTE_32F) matrix calculation against a double host ref.
// 3. Odd non-power-of-2 row count (M=49, K=1152, N=3456) matching 224x224 SigLIP patch projections.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

int fails = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                          \
        }                                                                     \
    } while (0)

inline uint16_t f32_bf16_ref(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline float bf16_f32_ref(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

__device__ __forceinline__ float clamp_f16_test(float v) {
    return v > 65504.0f ? 65504.0f : (v < -65504.0f ? -65504.0f : v);
}

__global__ void test_bf16_to_f16_kernel(const uint16_t* src, uint16_t* dst, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float f = __uint_as_float((uint32_t) src[i] << 16);
        dst[i] = __half_as_ushort(__float2half_rn(clamp_f16_test(f)));
    }
}

__global__ void test_pack_f16_kernel(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        dst[i] = __half_as_ushort(__float2half_rn(clamp_f16_test(src[i])));
    }
}

void test_clamp_and_nan_preservation() {
    std::vector<float> h_in = {
        0.0f, -0.0f, 1.5f, -3.25f, 65504.0f, -65504.0f,
        1.0e5f, -1.0e5f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    std::vector<uint16_t> h_bf(h_in.size());
    for (size_t i = 0; i < h_in.size(); ++i) h_bf[i] = f32_bf16_ref(h_in[i]);

    float* d_in = nullptr;
    uint16_t* d_bf = nullptr;
    uint16_t* d_f16_from_f32 = nullptr;
    uint16_t* d_f16_from_bf = nullptr;
    CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_bf, h_bf.size() * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_f16_from_f32, h_in.size() * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_f16_from_bf, h_bf.size() * sizeof(uint16_t)) == cudaSuccess);

    CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_bf, h_bf.data(), h_bf.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess);

    test_pack_f16_kernel<<<1, 64>>>(d_in, d_f16_from_f32, h_in.size());
    test_bf16_to_f16_kernel<<<1, 64>>>(d_bf, d_f16_from_bf, h_bf.size());
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uint16_t> out_f32(h_in.size()), out_bf(h_bf.size());
    CHECK(cudaMemcpy(out_f32.data(), d_f16_from_f32, out_f32.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
    CHECK(cudaMemcpy(out_bf.data(), d_f16_from_bf, out_bf.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) ==
          cudaSuccess);

    auto h2f = [](uint16_t bits) -> float {
        __half h;
        std::memcpy(&h, &bits, 2);
        return __half2float(h);
    };

    // Overflow & +/-Inf clamp to +/-65504.0f
    CHECK(h2f(out_f32[6]) == 65504.0f);
    CHECK(h2f(out_f32[7]) == -65504.0f);
    CHECK(h2f(out_f32[8]) == 65504.0f);
    CHECK(h2f(out_f32[9]) == -65504.0f);
    CHECK(h2f(out_bf[6]) == 65504.0f);
    CHECK(h2f(out_bf[7]) == -65504.0f);
    CHECK(h2f(out_bf[8]) == 65504.0f);
    CHECK(h2f(out_bf[9]) == -65504.0f);

    // NaN must stay NaN (never suppressed to +/-65504.0f)
    CHECK(std::isnan(h2f(out_f32[10])));
    CHECK(std::isnan(h2f(out_bf[10])));

    cudaFree(d_in);
    cudaFree(d_bf);
    cudaFree(d_f16_from_f32);
    cudaFree(d_f16_from_bf);
}

void test_fp16_gemm_parity() {
    // Non-power-of-2 rows M=49 (e.g. 7x7 patches), K=1152, N=3456 (wqkv shape in SigLIP)
    const int M = 49, K = 1152, N = 3456;
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 0.25f);

    std::vector<float> h_x((size_t) M * K);
    std::vector<uint16_t> h_w_bf((size_t) N * K);
    for (float& v : h_x) v = dist(rng);
    for (uint16_t& w : h_w_bf) w = f32_bf16_ref(dist(rng));

    std::vector<double> ref((size_t) M * N, 0.0);
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            double acc = 0.0;
            const float* xr = h_x.data() + (size_t) r * K;
            const uint16_t* wc = h_w_bf.data() + (size_t) c * K;
            for (int k = 0; k < K; ++k) acc += (double) xr[k] * (double) bf16_f32_ref(wc[k]);
            ref[(size_t) r * N + c] = acc;
        }
    }

    float* d_x = nullptr;
    uint16_t* d_x16 = nullptr;
    uint16_t* d_w_bf = nullptr;
    uint16_t* d_w16 = nullptr;
    float* d_c = nullptr;
    CHECK(cudaMalloc(&d_x, h_x.size() * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_x16, h_x.size() * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_w_bf, h_w_bf.size() * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_w16, h_w_bf.size() * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_c, (size_t) M * N * sizeof(float)) == cudaSuccess);

    CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_w_bf, h_w_bf.data(), h_w_bf.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess);

    test_pack_f16_kernel<<<(int) ((h_x.size() + 255) / 256), 256>>>(d_x, d_x16, h_x.size());
    test_bf16_to_f16_kernel<<<(int) ((h_w_bf.size() + 255) / 256), 256>>>(d_w_bf, d_w16, h_w_bf.size());

    cublasHandle_t blas = nullptr;
    CHECK(cublasCreate(&blas) == CUBLAS_STATUS_SUCCESS);
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t st = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, d_w16, CUDA_R_16F, K,
                                           d_x16, CUDA_R_16F, K, &zero, d_c, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT);
    CHECK(st == CUBLAS_STATUS_SUCCESS);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> h_c((size_t) M * N);
    CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    double d2 = 0.0, r2 = 0.0;
    for (size_t i = 0; i < h_c.size(); ++i) {
        CHECK(std::isfinite(h_c[i]));
        const double diff = (double) h_c[i] - ref[i];
        d2 += diff * diff;
        r2 += ref[i] * ref[i];
    }
    const double rel_l2 = std::sqrt(d2 / (r2 > 1e-30 ? r2 : 1e-30));
    std::printf("  hy4_vision_test FP16 GEMM (M=%d, K=%d, N=%d): status=%d rel_l2=%.6f\n", M, K, N, (int) st,
                rel_l2);
    CHECK(rel_l2 < 5e-3);

    cublasDestroy(blas);
    cudaFree(d_x);
    cudaFree(d_x16);
    cudaFree(d_w_bf);
    cudaFree(d_w16);
    cudaFree(d_c);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("hy4_vision_test: no GPU, skipped\n");
        return 0;
    }
    test_clamp_and_nan_preservation();
    test_fp16_gemm_parity();
    std::printf("hy4_vision_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
