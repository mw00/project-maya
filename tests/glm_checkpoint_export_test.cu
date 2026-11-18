#include "strata/kernels/glm_batch.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace gb = strata::kernels::glmb;
void ck(cudaError_t result) { if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result)); }
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class T> struct Device {
    T* ptr = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { ck(cudaMalloc(&ptr, n * sizeof(T))); }
    ~Device() { cudaFree(ptr); }
    void put(const std::vector<T>& data) { require(data.size() <= n, "device input size"); ck(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> out(n); ck(cudaMemcpy(out.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost)); return out; }
};
struct Result { std::vector<float> state, checkpoint; std::vector<__half> out; };

Result run(int split, int T, int after, bool export_state) {
    constexpr int HD = 128;
    const size_t values = size_t(T) * HD;
    Device<float> q(values), k(values), v(values), gate(values), out_gate(values), beta(T), dt(HD), decay(1), norm(HD);
    Device<float> state(HD * HD), saved(HD * HD);
    Device<__half> out(values);
    const auto wave = [](size_t n, float scale, float phase) {
        std::vector<float> values(n);
        for (size_t i = 0; i < n; ++i) values[i] = scale * std::sin(float(i) * 0.17f + phase);
        return values;
    };
    q.put(wave(values, .15f, 0)); k.put(wave(values, .13f, 1)); v.put(wave(values, .2f, 2));
    gate.put(wave(values, .1f, 3)); out_gate.put(wave(values, .1f, 4)); beta.put(wave(T, .1f, 5));
    dt.put(wave(HD, .1f, 6)); decay.put({-.1f}); norm.put(std::vector<float>(HD, 1));
    state.put(wave(HD * HD, .03f, 7)); saved.put(std::vector<float>(HD * HD, -999));
    float* checkpoint = export_state ? saved.ptr : nullptr;
    if (split)
        gb::kda_rec_split(q.ptr, k.ptr, v.ptr, gate.ptr, dt.ptr, decay.ptr, -5, beta.ptr, state.ptr,
                          out_gate.ptr, norm.ptr, 1e-5f, 1, T, out.ptr, nullptr, split, checkpoint, after);
    else
        gb::kda_rec(q.ptr, k.ptr, v.ptr, gate.ptr, dt.ptr, decay.ptr, -5, beta.ptr, state.ptr,
                    out_gate.ptr, norm.ptr, 1e-5f, 1, T, out.ptr, nullptr, checkpoint, after);
    ck(cudaDeviceSynchronize());
    return {state.get(), saved.get(), out.get()};
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        for (int split : {0, 4, 8}) {
            const auto full = run(split, 19, 0, false);
            for (int after : {1, 7, 16, 19}) {
                const auto exported = run(split, 19, after, true);
                const auto prefix = run(split, after, 0, false);
                require(exported.state == full.state, "export changed final live recurrent state");
                require(std::memcmp(exported.out.data(), full.out.data(), full.out.size() * sizeof(__half)) == 0,
                        "export changed full-call output");
                require(exported.checkpoint == prefix.state, "export differs from canonical prefix state");
            }
        }
        constexpr int DI = 128, T = 7, hist = 3;
        Device<float> a(DI * T), b(DI * T), c(DI * T), history(3 * DI * hist), saved(3 * DI * hist);
        std::vector<float> inputs(DI * T), original(3 * DI * hist);
        for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = float(i) / 100;
        for (size_t i = 0; i < original.size(); ++i) original[i] = -float(i) / 100;
        a.put(inputs); b.put(inputs); c.put(inputs); history.put(original);
        const float* proj[] = {a.ptr, b.ptr, c.ptr};
        for (int after : {1, 2, 3, 7}) {
            gb::kda_conv_checkpoint(proj, history.ptr, saved.ptr, after, DI, hist + 1, nullptr);
            ck(cudaDeviceSynchronize());
            require(history.get() == original, "export changed live convolution history");
            const auto checkpoint = saved.get();
            for (int role = 0; role < 3; ++role) for (int ch = 0; ch < DI; ++ch) for (int j = 0; j < hist; ++j) {
                const int token = after - hist + j;
                const auto at = size_t(role) * DI * hist + ch * hist + j;
                const float expected = token >= 0 ? inputs[size_t(token) * DI + ch] : original[size_t(role) * DI * hist + ch * hist + hist + token];
                require(checkpoint[at] == expected, "interior convolution checkpoint differs");
            }
        }
        std::puts("GLM checkpoint export: recurrent, split, convolution and full-call parity passed");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
