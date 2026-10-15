// The real RAM-tier helper against the same experts computed from device memory.
#include "strata/core/remote_experts.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

int main() {
    namespace gf = strata::kernels::glmf;
    constexpr int E = 4096, FF = 2048, K = 8;
    int devices = 0, helper_device = -1;
    CHECK(hipGetDeviceCount(&devices));
    for (int d = 1; d < devices; ++d) {
        hipDeviceProp_t p{};
        CHECK(hipGetDeviceProperties(&p, d));
        if (p.integrated) { helper_device = d; break; }
    }
    if (helper_device < 0) {
        std::puts("SKIP: requires a second, unified-memory GPU");
        return 77;
    }
    std::mt19937 rng(47);
    std::vector<float> x(E), reference(E), actual(E);
    float weights[K];
    for (int i = 0; i < E; ++i) x[i] = float(int(rng() % 101) - 50) * 0.01f;
    for (int i = 0; i < K; ++i) weights[i] = float(i + 1) / 36.0f;
    std::string err;
    const auto block_bytes = [](int type) -> size_t {
        switch (type) {
            case 10: return sizeof(block_q2_K);
            case 11: return sizeof(block_q3_K);
            case 16: return sizeof(block_iq2_xxs);
            case 18: return sizeof(block_iq3_xxs);
            default: return sizeof(block_iq2_s);
        }
    };
    for (const auto [gu_type, d_type] : {std::pair{16, 22}, std::pair{18, 22}, std::pair{16, 18}, std::pair{10, 11}}) {
        const size_t gu_block = block_bytes(gu_type), d_block = block_bytes(d_type);
        const size_t gu_bytes = size_t(2) * FF * E / 256 * gu_block;
        const size_t down_bytes = size_t(E) * FF / 256 * d_block;
        const size_t blob_bytes = gu_bytes + down_bytes;
        uint8_t* host = nullptr;
        CHECK(hipSetDevice(0));
        CHECK(hipHostMalloc((void**) &host, K * blob_bytes, hipHostMallocMapped | hipHostMallocPortable));
        for (int i = 0; i < K; ++i) {
            for (int role = 0; role < 2; ++role) {
                const size_t block = role ? d_block : gu_block;
                const size_t bytes = role ? down_bytes : gu_bytes;
                uint8_t* p = host + size_t(i) * blob_bytes + (role ? gu_bytes : 0);
                for (size_t b = 0; b < bytes; b += block) {
                    for (size_t j = 0; j < block; ++j) p[b + j] = rng() & 255;
                    const uint16_t scale = std::bit_cast<uint16_t>(__float2half(0.0001f * float(1 + rng() % 4)));
                    const int type = role ? d_type : gu_type;
                    const size_t scale_at = type == 10 ? block - 4 : type == 11 ? block - 2 : 0;
                    std::memcpy(p + b + scale_at, &scale, 2);
                    if (type == 10) std::memcpy(p + b + block - 2, &scale, 2);   // finite Q2_K dmin too
                }
            }
        }
        strata::core::RamExpertHelper helper;
        if (!helper.open(helper_device, E, FF, {{host, K * blob_bytes}}, err)) {
            std::fprintf(stderr, "%s\n", err.c_str()); return 1;
        }
        int current = -1;
        CHECK(hipGetDevice(&current));
        if (current != 0) return 3;   // helper calls must restore the caller's device
        // Reference: all experts on the primary GPU, with the same GLM kernels.
        void *dev_blobs = nullptr, *dxq = nullptr, *dhq = nullptr;
        float *dx = nullptr, *dout = nullptr;
        gf::MoeDev plan;
        CHECK(hipMalloc(&dev_blobs, K * blob_bytes));
        CHECK(hipMemcpy(dev_blobs, host, K * blob_bytes, hipMemcpyHostToDevice));
        CHECK(hipMalloc((void**) &dx, E * 4));
        CHECK(hipMalloc((void**) &dout, E * 4));
        CHECK(hipMalloc(&dxq, E / 32 * sizeof(block_q8_1)));
        CHECK(hipMalloc(&dhq, K * FF / 32 * sizeof(block_q8_1)));
        CHECK(hipMalloc((void**) &plan.plan_ptr, K * 8));
        CHECK(hipMalloc((void**) &plan.plan_w, K * 4));
        CHECK(hipMalloc((void**) &plan.cpu_flag, 4));
        CHECK(hipMemset(plan.cpu_flag, 0, 4));
        CHECK(hipMemcpy(dx, x.data(), E * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(plan.plan_w, weights, K * 4, hipMemcpyHostToDevice));
        const uint8_t* blobs[K];
        unsigned long long pointers[K];
        // Repeated pointers and noncontiguous ordering must preserve the weights.
        for (int i = 0; i < K; ++i) {
            const int e = (i * 3) % 5;
            blobs[i] = host + size_t(e) * blob_bytes;
            pointers[i] = (unsigned long long) ((uint8_t*) dev_blobs + size_t(e) * blob_bytes);
        }
        CHECK(hipMemcpy(plan.plan_ptr, pointers, K * 8, hipMemcpyHostToDevice));
        gf::quantize_q8_1(dx, dxq, E, nullptr);
        for (int ne : {1, 3, 8}) {
            gf::moe_gate_up(gu_type, plan, ne, E, FF, 8.0f, dxq, dhq, nullptr, 0, nullptr, 0, nullptr, nullptr);
            gf::moe_down(d_type, plan, ne, E, FF, gu_bytes, dhq, nullptr, dout, nullptr);
            CHECK(hipMemcpy(reference.data(), dout, E * 4, hipMemcpyDeviceToHost));
            if (!helper.compute(gu_type, d_type, gu_bytes, 8.0f, ne, blobs, weights, x.data(), actual.data(), err)) {
                std::fprintf(stderr, "%s\n", err.c_str()); return 1;
            }
            double diff2 = 0, ref2 = 0, max_abs = 0;
            for (int i = 0; i < E; ++i) {
                if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) return 4;
                const double diff = double(actual[i]) - reference[i];
                diff2 += diff * diff; ref2 += double(reference[i]) * reference[i];
                max_abs = std::max(max_abs, std::abs(diff));
            }
            const double rel = std::sqrt(diff2 / std::max(ref2, 1e-30));
            std::printf("GLM RAM helper: types %d/%d, experts %d, max_abs %.3g rel_L2 %.3g\n",
                        gu_type, d_type, ne, max_abs, rel);
            if (max_abs > 1e-4 || rel > 1e-6) return 5;
            CHECK(hipGetDevice(&current));
            if (current != 0) return 6;
        }
        const uint8_t* outside = host + K * blob_bytes;
        if (helper.compute(gu_type, d_type, gu_bytes, 8.0f, 1, &outside, weights, x.data(), actual.data(), err)) return 7;
        helper.close();
        CHECK(hipFree(plan.cpu_flag)); CHECK(hipFree(plan.plan_w)); CHECK(hipFree(plan.plan_ptr));
        CHECK(hipFree(dhq)); CHECK(hipFree(dxq)); CHECK(hipFree(dout)); CHECK(hipFree(dx)); CHECK(hipFree(dev_blobs));
        CHECK(hipHostFree(host));
    }
    std::puts("GLM RAM expert helper parity passed");
}
