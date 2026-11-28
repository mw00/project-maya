// src/ik2s/ik2_hip_situ_cpu.hpp - Inkling 2's SITU on the CPU lane (ik2_hip_lane.cpp; tests/ik2_hip_situ_test.cu):
// the gate's and the up's rows through maya-ds4's CPU rows for a GGUF's experts (ds4_gguf_cpu.hpp's iq_cpu_rows - the
// same dots iq_cpu_gate_up's SwiGLU takes), then the SITU of them.  ds4's lane keeps its own activations; this one is
// maya-ik2s's.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../ds4/ds4_gguf_cpu.hpp"

namespace ik2hip {

/// Inkling 2's SITU (llama.cpp's LLM_FFN_SITU, ggml's operations in its order; no multiply-add in it - ds4_iq.cu's
/// device situ): beta tanh(g / beta) sigmoid(g) * (lb > 0 ? lb tanh(u / lb) : u)
inline float situ_cpu(float g, float u, float beta, float lb) {
    const float a = std::tanh(g * (1.0f / beta)) * beta * (1.0f / (1.0f + std::exp(-g)));
    const float v = lb > 0.0f ? std::tanh(u * (1.0f / lb)) * lb : u;
    return a * v;
}

/// iq_cpu_gate_up with the SITU in place of the SwiGLU: ff[t][r] = situ(gate_r . act[t], up_r . act[t]) for rows
/// [r0, r1), nt tokens (f from iq_cpu_fmt; act as f.gu_act)
inline void situ_gate_up(const maya::ds4::gguf::IqCpu& f, float beta, float lb, const uint8_t* gate, const uint8_t* up,
                         const void* const* act, int nt, float* const* ff, int r0, int r1) {
    if (r1 <= r0 || nt <= 0) return;
    const int nr = r1 - r0;
    thread_local std::vector<float> buf;
    thread_local std::vector<float*> go, uo;
    buf.resize((size_t) 2 * nt * nr);
    go.resize((size_t) nt);
    uo.resize((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        go[(size_t) t] = buf.data() + (size_t) t * nr;
        uo[(size_t) t] = buf.data() + (size_t) (nt + t) * nr;
    }
    maya::ds4::gguf::iq_cpu_rows(f.tg, gate + (size_t) r0 * f.gu_row, f.gu_row, f.n_embd, act, nt, go.data(), 0, nr);
    maya::ds4::gguf::iq_cpu_rows(f.tg, up + (size_t) r0 * f.gu_row, f.gu_row, f.n_embd, act, nt, uo.data(), 0, nr);
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < nr; ++r) ff[t][r0 + r] = situ_cpu(go[(size_t) t][r], uo[(size_t) t][r], beta, lb);
}

}  // namespace ik2hip
