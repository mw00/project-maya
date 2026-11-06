#include "expert_abi.h"
#include "strata/kernels/glm_fast.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
struct Resources {
    cudaStream_t stream = nullptr;
    std::vector<void *> pointers;
    void *alloc(size_t bytes) {
        void *p = nullptr; check(cudaMalloc(&p, bytes));
        try { pointers.push_back(p); } catch (...) { cudaFree(p); throw; }
        return p;
    }
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        for (void *p : pointers) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
};
}
extern "C" __attribute__((visibility("default")))
int maya_expert_probe(uint32_t abi, int device, int gu_type, int down_type,
    int n_embd, int n_ff, int k, const void *blobs, size_t bytes, size_t stride,
    size_t down_offset, const float *routing, const float *input, float clamp,
    float *output, char *error, size_t error_size) {
    namespace gf = strata::kernels::glmf;
    try {
        // Deliberately restricted fixture shapes; generality needs its own tests.
        if (abi != 1 || n_embd != 4096 || n_ff != 2048 || k < 1 || k > 8 ||
            !blobs || !routing || !input || !output || !std::isfinite(clamp) || clamp <= 0 ||
            !gf::moe_supported(gu_type) || !gf::moe_supported(down_type))
            throw std::runtime_error("unsupported expert descriptor");
        const size_t gu_bytes = 2 * size_t(n_ff) * gf::row_bytes(gu_type, n_embd);
        const size_t d_bytes = size_t(n_embd) * gf::row_bytes(down_type, n_ff);
        if (down_offset != gu_bytes || stride != gu_bytes + d_bytes || bytes != size_t(k) * stride)
            throw std::runtime_error("invalid expert blob layout");
        for (int i = 0; i < k; ++i) if (!std::isfinite(routing[i])) throw std::runtime_error("nonfinite route");
        for (int i = 0; i < n_embd; ++i) if (!std::isfinite(input[i])) throw std::runtime_error("nonfinite input");
        check(cudaSetDevice(device));
        Resources r;
        check(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
        auto *data = static_cast<unsigned char *>(r.alloc(bytes));
        auto *x = static_cast<float *>(r.alloc(size_t(n_embd) * 4));
        auto *out = static_cast<float *>(r.alloc(size_t(n_embd) * 4));
        void *xq = r.alloc(size_t(n_embd) / 32 * 36);
        void *hq = r.alloc(size_t(k) * n_ff / 32 * 36);
        gf::MoeDev plan;
        plan.plan_ptr = static_cast<unsigned long long *>(r.alloc(size_t(k) * 8));
        plan.plan_w = static_cast<float *>(r.alloc(size_t(k) * 4));
        plan.cpu_flag = static_cast<int *>(r.alloc(4));
        unsigned long long pointers[8]{};
        for (int i = 0; i < k; ++i) pointers[i] = reinterpret_cast<unsigned long long>(data + size_t(i) * stride);
        check(cudaMemcpyAsync(data, blobs, bytes, cudaMemcpyHostToDevice, r.stream));
        check(cudaMemcpyAsync(x, input, size_t(n_embd) * 4, cudaMemcpyHostToDevice, r.stream));
        check(cudaMemcpyAsync(plan.plan_ptr, pointers, size_t(k) * 8, cudaMemcpyHostToDevice, r.stream));
        check(cudaMemcpyAsync(plan.plan_w, routing, size_t(k) * 4, cudaMemcpyHostToDevice, r.stream));
        check(cudaMemsetAsync(plan.cpu_flag, 0, 4, r.stream));
        gf::quantize_q8_1(x, xq, n_embd, r.stream);
        gf::moe_gate_up(gu_type, plan, k, n_embd, n_ff, clamp, xq, hq, nullptr, 0, nullptr, 0, nullptr, r.stream);
        gf::moe_down(down_type, plan, k, n_embd, n_ff, down_offset, hq, nullptr, out, r.stream);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(output, out, size_t(n_embd) * 4, cudaMemcpyDeviceToHost, r.stream));
        check(cudaStreamSynchronize(r.stream));
        return 0;
    } catch (const std::exception &e) {
        if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
        return 1;
    } catch (...) {
        if (error && error_size) std::snprintf(error, error_size, "unknown module failure");
        return 1;
    }
}
