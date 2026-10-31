// src/core/glm_ablate.cuh - STRATA_GLM_ABLATE=<file>: directions projected out of the GLM path's residual streams
// (research; off unless the variable is set).  The file is raw float32.
//
// Outputs mode (default): K * n_embd values, K orthonormal vectors.  The token embedding and every block output (the
// mixer's and the FFN's) become x - scale (x . v) v before the mHC write.  The mHC streams are only ever linear
// mixes of those rows (R = post x block_out + comb . R), so every stream stays orthogonal to the vectors: the same
// effect as orthogonalizing every matrix that writes into the stream, without requantizing anything.
//
// Per-layer mode (STRATA_GLM_ABLATE_LAYERS=<n_layers>): n_layers * K * n_embd values, layer l's own K orthonormal
// vectors (all zero = the layer is not steered).  After layer l's FFN write every stream loses its components along
// layer l's vectors - done on the two operands of that write (the FFN output and the streams it is mixed into), so
// the fused read needs no change.  The same operation as exllamav3's EXL3_CVEC hook after each block.
//
// The NextN draft block is left alone (a draft only has to be a good guess).  STRATA_GLM_ABLATE_SCALE (default 1).
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace glm_ablate {

namespace {
__global__ void rows_kernel(float* __restrict__ X, const float* __restrict__ V, int E, int K, float scale) {
    float* x = X + (size_t) blockIdx.x * E;   // one block a row, 256 threads; the K directions one after another
    __shared__ float red[256];
    const int tid = threadIdx.x;
    for (int k = 0; k < K; ++k) {
        const float* v = V + (size_t) k * E;
        float acc = 0.0f;
        for (int e = tid; e < E; e += 256) acc += x[e] * v[e];
        red[tid] = acc;
        __syncthreads();
        for (int n = 128; n > 0; n >>= 1) {
            if (tid < n) red[tid] += red[tid + n];
            __syncthreads();
        }
        const float c = red[0] * scale;
        __syncthreads();
        for (int e = tid; e < E; e += 256) x[e] -= c * v[e];
        __syncthreads();
    }
}
}  // namespace

struct State {
    bool on = false;
    int layers = 0;               // > 0: per-layer mode, the file holds this many layers
    std::vector<char> steered;    // per-layer mode: the layer has a non-zero direction
    float scale = 1.0f;
    std::vector<float> host;
    std::mutex mu;
    float* dev[64] = {};
};

inline State& state() {
    static State s;
    static std::once_flag once;
    std::call_once(once, [] {
        const char* path = getenv("STRATA_GLM_ABLATE");
        if (path == nullptr || !path[0]) return;
        std::FILE* f = std::fopen(path, "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "glm ablate: cannot open %s - running the stock model\n", path);
            return;
        }
        std::fseek(f, 0, SEEK_END);
        const long bytes = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        s.host.resize((size_t) bytes / sizeof(float));
        const size_t got = std::fread(s.host.data(), sizeof(float), s.host.size(), f);
        std::fclose(f);
        if (got != s.host.size() || s.host.empty()) {
            std::fprintf(stderr, "glm ablate: %s is not a float32 vector - running the stock model\n", path);
            return;
        }
        if (const char* sc = getenv("STRATA_GLM_ABLATE_SCALE")) s.scale = (float) std::atof(sc);
        if (const char* ly = getenv("STRATA_GLM_ABLATE_LAYERS")) s.layers = std::atoi(ly);
        if (s.layers > 0) {
            if (s.host.size() % (size_t) s.layers != 0) {
                std::fprintf(stderr, "glm ablate: %s does not hold %d layers - running the stock model\n", path, s.layers);
                return;
            }
            const size_t per = s.host.size() / (size_t) s.layers;
            s.steered.assign((size_t) s.layers, 0);
            for (int l = 0; l < s.layers; ++l)
                for (size_t i = 0; i < per; ++i)
                    if (s.host[(size_t) l * per + i] != 0.0f) { s.steered[(size_t) l] = 1; break; }
        }
        double nn = 0.0;
        for (float x : s.host) nn += (double) x * x;
        s.on = true;
        std::fprintf(stderr, "glm ablate: %zu values from %s (norm^2 %.4f), %s, scale %.2f\n", s.host.size(), path, nn,
                     s.layers > 0 ? "per layer" : "block outputs", s.scale);
    });
    return s;
}

inline bool on() { return state().on; }

/// The table on the CURRENT device (uploaded on first use), or nullptr when off or of another width.
inline const float* vec(int E) {
    State& a = state();
    if (!a.on || E <= 0 || a.host.size() % (size_t) E != 0) return nullptr;
    if (a.layers > 0 && a.host.size() % ((size_t) a.layers * (size_t) E) != 0) return nullptr;
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 64) return nullptr;
    std::lock_guard<std::mutex> lk(a.mu);
    if (a.dev[d] == nullptr) {
        float* p = nullptr;
        if (cudaMalloc((void**) &p, a.host.size() * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(p, a.host.data(), a.host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "glm ablate: no memory for the direction on CUDA%d - ablation OFF\n", d);
            a.on = false;
            return nullptr;
        }
        a.dev[d] = p;
    }
    return a.dev[d];
}

/// Outputs mode: T rows of E values at X (row stride E): x <- x - scale (x . v) v for the file's vectors.
inline void rows(float* X, int T, int E, cudaStream_t s) {
    State& a = state();
    if (!a.on || a.layers > 0 || X == nullptr || T <= 0) return;
    const float* v = vec(E);
    if (v == nullptr) return;
    rows_kernel<<<T, 256, 0, s>>>(X, v, E, (int) (a.host.size() / (size_t) E), a.scale);
}

/// Per-layer mode: the same on T rows with layer `layer`'s vectors (no-op in outputs mode or for an unsteered layer).
inline void layer_rows(int layer, float* X, int T, int E, cudaStream_t s) {
    State& a = state();
    if (!a.on || a.layers <= 0 || layer < 0 || layer >= a.layers || !a.steered[(size_t) layer] || X == nullptr || T <= 0)
        return;
    const float* v = vec(E);
    if (v == nullptr) return;
    const int K = (int) (a.host.size() / ((size_t) a.layers * (size_t) E));
    rows_kernel<<<T, 256, 0, s>>>(X, v + (size_t) layer * K * E, E, K, a.scale);
}

}  // namespace glm_ablate
