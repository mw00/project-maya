// src/core/glm_guards_host.cpp - glm-dsa wave 9 (B15.1, B15.2, B15.6): the host halves of the guards, plain C++ (no
// CUDA), in strata_engine on every backend and linked straight into the CPU tests.
//   B15.1  MAYA_GLM_F16_GUARD and the CPU twins of the FP16 range guard (include/strata/kernels/glmd_f16_guard.hpp)
//   B15.2  the gap / tail ranges of the MMQ expert banks (include/strata/kernels/glmd_mmq_pad.hpp)
//   B15.6  the process-wide CUDA-graph generation (include/strata/kernels/glmd_graph.hpp)
#include "strata/core/glm_env.hpp"
#include "strata/kernels/glmd_f16_guard.hpp"
#include "strata/kernels/glmd_graph.hpp"
#include "strata/kernels/glmd_mmq_pad.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace glmd {

namespace sk = strata::kernels;

// ---- B15.1 ----------------------------------------------------------------------------------------------------------

int f16_guard_mode() {
    return strata::glmenv::env_switch("MAYA_GLM_F16_GUARD", strata::glmenv::kAuto, true);
}

namespace guard_twin {

void pack_f16(const float* src, int64_t src_row, int64_t src_head, int n_head, int K, int T, uint16_t* dst) {
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < n_head; ++h)
            for (int k = 0; k < K; ++k)
                dst[((int64_t) t * n_head + h) * K + k] = sk::f16_from_f32(src[t * src_row + h * src_head + k]);
}

void guard_rows(const float* src, int64_t src_row, int64_t src_head, int n_head, int K, int T, uint16_t* dst,
                float* scale) {
    for (int t = 0; t < T; ++t) {
        float amax = 0.0f;
        for (int h = 0; h < n_head; ++h)
            for (int k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(src[t * src_row + h * src_head + k]));
        const float s = f16_guard_scale(amax);
        scale[t] = s;
        if (s == 1.0f) continue;
        const float inv = 1.0f / s;
        for (int h = 0; h < n_head; ++h)
            for (int k = 0; k < K; ++k)
                dst[((int64_t) t * n_head + h) * K + k] = sk::f16_from_f32(src[t * src_row + h * src_head + k] * inv);
    }
}

void swiglu_f16(const float* gate, const float* up, int64_t n, float limit, uint16_t* h16) {
    for (int64_t i = 0; i < n; ++i) h16[i] = sk::f16_from_f32(swiglu_value(gate[i], up[i], limit));
}

void guard_swiglu(const float* gate, const float* up, int n_ff, int T, float limit, uint16_t* h16, float* scale) {
    for (int t = 0; t < T; ++t) {
        const int64_t o = (int64_t) t * n_ff;
        float amax = 0.0f;
        for (int c = 0; c < n_ff; ++c) amax = std::fmax(amax, std::fabs(swiglu_value(gate[o + c], up[o + c], limit)));
        const float s = f16_guard_scale(amax);
        scale[t] = s;
        if (s == 1.0f) continue;
        const float inv = 1.0f / s;
        for (int c = 0; c < n_ff; ++c) h16[o + c] = sk::f16_from_f32(swiglu_value(gate[o + c], up[o + c], limit) * inv);
    }
}

void unscale_rows(float* Y, int64_t ldy, int n, int T, const float* scale) {
    for (int t = 0; t < T; ++t) {
        if (scale[t] == 1.0f) continue;
        for (int c = 0; c < n; ++c) Y[t * ldy + c] *= scale[t];
    }
}

void gemm_f16(const uint16_t* X16, int64_t ldx, const uint16_t* W16, float* Y, int64_t ldy, int T, int N, int K) {
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < N; ++o) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += sk::f32_from_f16(X16[t * ldx + k]) * sk::f32_from_f16(W16[(int64_t) o * K + k]);
            Y[t * ldy + o] = acc;
        }
}

}  // namespace guard_twin

// ---- B15.2 ----------------------------------------------------------------------------------------------------------

std::vector<MmqRun> mmq_gap_runs(int n, size_t stride, size_t blob, size_t room) {
    std::vector<MmqRun> out;
    if (n < 0 || stride == 0 || blob > stride) return out;
    const auto add = [&](size_t a, size_t b) {
        b = std::min(b, room);
        if (b > a) out.push_back(MmqRun{a, b - a});
    };
    for (int j = 0; j < n; ++j) add((size_t) j * stride + blob, (size_t) (j + 1) * stride);
    add((size_t) n * stride, (size_t) n * stride + kMmqTailPad);
    // adjacent ranges (the last slot's gap and the tail) merged: one memset each
    std::vector<MmqRun> merged;
    for (const MmqRun& r : out) {
        if (!merged.empty() && merged.back().off + merged.back().len == r.off) merged.back().len += r.len;
        else merged.push_back(r);
    }
    return merged;
}

// ---- B15.6 ----------------------------------------------------------------------------------------------------------

namespace {
std::atomic<uint64_t> g_gen{1};
std::mutex g_gen_mu;
std::map<std::string, uint64_t>& gen_counts() {
    static std::map<std::string, uint64_t> m;
    return m;
}
std::map<std::string, std::pair<const void*, size_t>>& gen_notes() {
    static std::map<std::string, std::pair<const void*, size_t>> m;
    return m;
}
}  // namespace

uint64_t graph_generation() { return g_gen.load(std::memory_order_acquire); }

uint64_t graph_generation_bump(const char* why) {
    std::lock_guard<std::mutex> lk(g_gen_mu);
    ++gen_counts()[why != nullptr ? why : "?"];
    return g_gen.fetch_add(1, std::memory_order_acq_rel) + 1;
}

uint64_t graph_generation_note(const char* name, const void* base, size_t bytes) {
    {
        std::lock_guard<std::mutex> lk(g_gen_mu);
        auto& notes = gen_notes();
        const std::string key = name != nullptr ? name : "?";
        const auto it = notes.find(key);
        if (it != notes.end() && it->second.first == base && it->second.second == bytes) return graph_generation();
        notes[key] = {base, bytes};
    }
    return graph_generation_bump(name);
}

uint64_t graph_generation_bumps(const char* why) {
    std::lock_guard<std::mutex> lk(g_gen_mu);
    if (why == nullptr) {
        uint64_t n = 0;
        for (const auto& kv : gen_counts()) n += kv.second;
        return n;
    }
    const auto it = gen_counts().find(why);
    return it == gen_counts().end() ? 0 : it->second;
}

}  // namespace glmd
