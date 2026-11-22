// src/kernels/cpu/native_split_test.cpp - GLM's split-base expert rows (a GGUF's own gate/up/down tensors, the CPU
// expert lane) against the contiguous [gate|up|down] blob path, and GLM's clamped SwiGLU.  Synthetic, CPU only.
#include "maya/kernels/cpu/native_expert.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "native_split_test: %s\n", what);
        std::exit(1);
    }
}

int main() {
    namespace cpu = maya::kernels::cpu;
    cpu::NativeFmt glm, f;
    std::string err;
    // GLM-5.3-Flash's experts (n_embd 4096, n_ff 2048, IQ1_S gate/up, Q2_K down) fit the pool's activation buffers
    check(cpu::native_fmt(GGML_TYPE_IQ1_S, GGML_TYPE_Q2_K, 4096, 2048, glm, err), "GLM geometry");
    check(cpu::native_fmt(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 32, 32, f, err), "small format");
    std::vector<unsigned char> blob(f.bytes), aq(f.act_bytes), hq(f.h_bytes);
    std::vector<float> weights(32 * 32, 1.0f), x(32, 1.0f), h(32), ordinary(32), out(32), separate(32);
    const auto quant = [&](const std::vector<float>& w, size_t off) {
        ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), blob.data() + off, 0, 32, 32, nullptr);
    };
    quant(weights, 0);   // gate: every dot 32
    for (auto& w : weights) w = 2.0f;
    quant(weights, f.up_off);   // up: 64
    for (auto& w : weights) w = 1.0f;
    quant(weights, f.down_off);
    cpu::native_quant_act(f, x.data(), aq.data());
    const void* act[] = {aq.data()};
    float* result[] = {h.data()};
    // limit 10: the gate is capped at 10 and the up clamped to [-10, 10]
    cpu::native_gu_rows_split(f, blob.data(), blob.data() + f.up_off, act, 1, result, 0, 32, 10.0f);
    const float expected = 100.0f / (1.0f + std::exp(-10.0f));
    for (float v : h) check(std::abs(v - expected) < 1e-4f, "clamped SwiGLU");
    cpu::native_quant_h(f, h.data(), hq.data());
    const void* down_act[] = {hq.data()};
    float* down_out[] = {out.data()};
    float* down_split[] = {separate.data()};
    cpu::native_down_rows(f, blob.data(), down_act, 1, down_out, 0, 32);
    cpu::native_down_rows_split(f, blob.data() + f.down_off, down_act, 1, down_split, 0, 32);
    check(out == separate, "split down equals contiguous down");
    float* plain_out[] = {ordinary.data()};
    cpu::native_gu_rows(f, blob.data(), act, 1, plain_out, 0, 32);
    cpu::native_gu_rows_split(f, blob.data(), blob.data() + f.up_off, act, 1, result, 0, 32,
                              std::numeric_limits<float>::infinity());
    check(h == ordinary, "unclamped split equals contiguous gate/up");
    std::puts("native_split_test: ok");
    return 0;
}
