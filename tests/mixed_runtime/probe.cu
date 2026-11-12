#include "helper_abi.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void cu(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__global__ void add(float *p, size_t n, float v) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] += v;
}
int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s /absolute/libmaya_hip_probe.so CUDA_DEVICE HIP_DEVICE\n", argv[0]);
        return 2;
    }
    void *lib = nullptr, *ctx = nullptr;
    const maya_probe_api *api = nullptr;
    float *device = nullptr, *host = nullptr;
    cudaStream_t stream = nullptr;
    char error[512] = {};
    int rc = 1;
    try {
        auto parse = [](const char *s) {
            char *end = nullptr;
            long v = std::strtol(s, &end, 10);
            if (!*s || *end || v < 0 || v > 1024) throw std::runtime_error("invalid device index");
            return int(v);
        };
        const int cuda_device = parse(argv[2]), hip_device = parse(argv[3]);
        cu(cudaSetDevice(cuda_device));
        cudaDeviceProp prop{}; cu(cudaGetDeviceProperties(&prop, cuda_device));
        std::printf("CUDA device %d: %s; HIP requested device %d\n", cuda_device, prop.name, hip_device);
        lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw std::runtime_error(dlerror());
        auto get = reinterpret_cast<maya_probe_get_api_fn>(dlsym(lib, "maya_probe_get_api"));
        if (!get) throw std::runtime_error("missing ABI entry point");
        if (get(MAYA_PROBE_ABI + 1)) throw std::runtime_error("unknown ABI accepted");
        api = get(MAYA_PROBE_ABI);
        if (!api || api->struct_size != sizeof(*api) || api->version != MAYA_PROBE_ABI ||
            !api->create || !api->twice || !api->destroy) throw std::runtime_error("bad ABI");
        constexpr size_t capacity = 1024 * 1024;
        if (api->create(hip_device, capacity, &ctx, error, sizeof(error))) throw std::runtime_error(error);
        cu(cudaMalloc(reinterpret_cast<void **>(&device), capacity * sizeof(float)));
        cu(cudaMallocHost(reinterpret_cast<void **>(&host), (capacity + 2) * sizeof(float)));
        cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::vector<float> hip_in(capacity), hip_out(capacity + 2);
        if (!api->twice(ctx, hip_in.data(), hip_out.data(), capacity + 1, error, sizeof(error)))
            throw std::runtime_error("oversized request accepted");
        for (int round = 0; round < 20; ++round) {
            for (size_t count : {size_t(1), size_t(255), size_t(257), size_t(4096), capacity}) {
                host[0] = host[count + 1] = -991;
                for (size_t i = 0; i < count; ++i) host[i + 1] = float(int(i % 511) - 255 + round);
                cu(cudaMemcpyAsync(device, host + 1, count * sizeof(float), cudaMemcpyHostToDevice, stream));
                add<<<(count + 255) / 256, 256, 0, stream>>>(device, count, 1);
                cu(cudaGetLastError());
                cu(cudaMemcpyAsync(host + 1, device, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
                cu(cudaStreamSynchronize(stream));
                std::memcpy(hip_in.data(), host + 1, count * sizeof(float));
                hip_out[0] = hip_out[count + 1] = -992;
                if (api->twice(ctx, hip_in.data(), hip_out.data() + 1, count, error, sizeof(error)))
                    throw std::runtime_error(error);
                if (hip_out[0] != -992 || hip_out[count + 1] != -992) throw std::runtime_error("HIP canary");
                std::memcpy(host + 1, hip_out.data() + 1, count * sizeof(float));
                cu(cudaMemcpyAsync(device, host + 1, count * sizeof(float), cudaMemcpyHostToDevice, stream));
                add<<<(count + 255) / 256, 256, 0, stream>>>(device, count, 3);
                cu(cudaGetLastError());
                cu(cudaMemcpyAsync(host + 1, device, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
                cu(cudaStreamSynchronize(stream));
                for (size_t i = 0; i < count; ++i)
                    if (host[i + 1] != 2 * float(int(i % 511) - 255 + round + 1) + 3)
                        throw std::runtime_error("mixed result mismatch");
                if (host[0] != -991 || host[count + 1] != -991) throw std::runtime_error("CUDA canary");
            }
        }
        rc = 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); }
    // Keep both runtimes loaded until every allocation and queue is released.
    if (stream && cudaStreamSynchronize(stream) != cudaSuccess) rc = 1;
    if (ctx && api->destroy(ctx, error, sizeof(error))) { std::fprintf(stderr, "%s\n", error); rc = 1; }
    if (stream && cudaStreamDestroy(stream) != cudaSuccess) rc = 1;
    if (device && cudaFree(device) != cudaSuccess) rc = 1;
    if (host && cudaFreeHost(host) != cudaSuccess) rc = 1;
    if (lib && dlclose(lib)) rc = 1;
    if (!rc) std::puts("PASS: 100 mixed-runtime round trips, exact results, boundary canaries, ABI and capacity rejection");
    return rc;
}
