// Every Q5_K selector entry against ggml's CPU decoder, including row/block boundaries.
#include "../src/core/glm_dflash_selector.cuh"
#include "ggml.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(call) do { \
    const cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
        return 1; \
    } \
} while (0)

__global__ void decode(const uint8_t* table, int rank, int rows, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rank * rows) out[i] = strata::core::dflash_detail::q5_k_at(table, rank, i / rank, i % rank);
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> random(-7.0f, 7.0f);
    constexpr int rows = 5;
    const auto type = GGML_TYPE_Q5_K;
    const auto* traits = ggml_get_type_traits(type);
    for (const int rank : {256, 512, 768, 1024}) {
        const size_t count = (size_t) rank * rows, row_bytes = ggml_row_size(type, rank);
        std::vector<float> values(count), ref(count), got(count);
        for (size_t i = 0; i < count; ++i) values[i] = random(rng);
        std::fill(values.begin(), values.begin() + rank, 0.0f);
        std::fill(values.begin() + rank, values.begin() + 2 * rank, -0.5f);
        std::vector<uint8_t> packed(row_bytes * rows);
        ggml_quantize_chunk(type, values.data(), packed.data(), 0, rows, rank, nullptr);
        // Also exercise every packed scale/high-bit pattern, including the
        // upper two bits shared between scales and minima in groups 4..7.
        auto* blocks = reinterpret_cast<strata::core::dflash_detail::BlkQ5K*>(packed.data());
        for (int i = 2 * rank / 256; i < rows * rank / 256; ++i) {
            blocks[i].dm = __floats2half2_rn(0.03125f, 0.015625f);
            for (auto& b : blocks[i].scales) b = (uint8_t) rng();
            for (auto& b : blocks[i].qh) b = (uint8_t) rng();
            for (auto& b : blocks[i].qs) b = (uint8_t) rng();
        }
        for (int row = 0; row < rows; ++row)
            traits->to_float(packed.data() + row * row_bytes, ref.data() + row * rank, rank);
        uint8_t* input = nullptr;
        float* output = nullptr;
        CK(cudaMalloc((void**) &input, packed.size()));
        CK(cudaMalloc((void**) &output, count * sizeof(float)));
        CK(cudaMemcpy(input, packed.data(), packed.size(), cudaMemcpyHostToDevice));
        decode<<<(count + 127) / 128, 128>>>(input, rank, rows, output);
        CK(cudaGetLastError());
        CK(cudaMemcpy(got.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaFree(input));
        CK(cudaFree(output));
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(got[i]) || std::fabs(got[i] - ref[i]) > 1e-6f + 2e-6f * std::fabs(ref[i])) {
                std::fprintf(stderr, "Q5_K rank %d row %zu entry %zu: GPU %.9g, ggml %.9g\n",
                             rank, i / rank, i % rank, got[i], ref[i]);
                return 1;
            }
        }
        std::printf("Q5_K rank %d: all %zu entries match ggml\n", rank, count);
    }
    return 0;
}
