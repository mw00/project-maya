// tests/glm_ablate_test.cu - glm_ablate.cuh's rows kernel against a CPU reference (GPU, no model).
//
// The settings are read once a process (the harness's state is a call_once), so each mode is a run of its own:
//   outputs: the file's K directions of n_embd values - every row becomes x - s (x . v) v, K times
//   scale:   the same at s = 0.5 - half the component along each direction is left
//   layers:  n_layers x K x n_embd values - layer 0's direction is removed, layer 1's (all zero) is not steered
//
// A device that cannot be opened (no card, or one the engine has taken) is a SKIP that passes, as the other GPU tests
// do: the numbers here need a kernel run, and nothing about the CPU side is worth checking on its own.
#include "../src/core/glm_ablate.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
int g_checks = 0;
void check(bool ok, const char* what) {
    std::printf("  %-68s %s\n", what, ok ? "ok" : "FAIL");
    ++g_checks;
    if (!ok) ++g_fail;
}

bool has_device() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { cudaGetLastError(); return false; }
    if (cudaSetDevice(0) != cudaSuccess) { cudaGetLastError(); return false; }
    void* probe = nullptr;
    if (cudaMalloc(&probe, (size_t) 1 << 20) != cudaSuccess) { cudaGetLastError(); return false; }
    cudaFree(probe);
    return true;
}

/// The reference: the same projection, in double, over rows of E values.
std::vector<double> project(const std::vector<float>& X, const std::vector<float>& V, int T, int E, int K,
                            long off, double scale) {
    std::vector<double> out(X.begin(), X.end());
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            const float* v = V.data() + (size_t) (off + k) * (size_t) E;
            double* x = out.data() + (size_t) t * (size_t) E;
            double acc = 0.0;
            for (int e = 0; e < E; ++e) acc += x[e] * (double) v[e];
            const double c = acc * scale;
            for (int e = 0; e < E; ++e) x[e] -= c * (double) v[e];
        }
    }
    return out;
}

/// The device result for the same input and the same vectors (nullptr: the harness is off, or the width is wrong).
bool run_device(const std::vector<float>& X, int T, int E, int layer, std::vector<float>& out) {
    float* d = nullptr;
    if (cudaMalloc(&d, X.size() * sizeof(float)) != cudaSuccess) return false;
    const bool up = cudaMemcpy(d, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    if (up) {
        if (layer < 0) {
            glm_ablate::rows(d, T, E, nullptr);
        } else {
            glm_ablate::layer_rows(layer, d, T, E, nullptr);
        }
    }
    const bool sync = cudaDeviceSynchronize() == cudaSuccess;
    out.assign(X.size(), 0.0f);
    const bool down = sync && cudaMemcpy(out.data(), d, X.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d);
    if (!up || !down) { cudaGetLastError(); return false; }
    return true;
}

bool close(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0.0, mag = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, std::fabs((double) got[i] - want[i]));
        mag = std::max(mag, std::fabs(want[i]));
    }
    const bool ok = worst <= 1e-4 * (mag > 1.0 ? mag : 1.0);
    if (!ok) std::printf("    worst difference %.6g against %.6g\n", worst, mag);
    return ok;
}

double dot(const std::vector<double>& x, const float* v, int E, int t) {
    double acc = 0.0;
    for (int e = 0; e < E; ++e) acc += x[(size_t) t * (size_t) E + (size_t) e] * (double) v[e];
    return acc;
}

std::vector<float> direction(int E, int kind) {   // two unit vectors that are orthogonal to each other
    std::vector<float> v((size_t) E, 0.0f);
    if (kind == 0) {
        const float a = (float) (1.0 / std::sqrt((double) E));
        for (int e = 0; e < E; ++e) v[(size_t) e] = a;
    } else {
        const float a = (float) (1.0 / std::sqrt((double) E));
        for (int e = 0; e < E; ++e) v[(size_t) e] = (e % 2 == 0) ? a : -a;
    }
    return v;
}

std::string write_file(const std::vector<float>& values, const char* name) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::FILE* f = std::fopen(p.string().c_str(), "wb");
    if (f == nullptr) return std::string();
    std::fwrite(values.data(), sizeof(float), values.size(), f);
    std::fclose(f);
    return p.string();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "outputs";
    if (!has_device()) {
        std::printf("SKIP: no usable CUDA device: the projection's kernel needs one\n");
        return 0;
    }
    const int E = 32, T = 4;
    std::vector<float> X((size_t) T * (size_t) E);
    for (int t = 0; t < T; ++t)
        for (int e = 0; e < E; ++e) X[(size_t) t * (size_t) E + (size_t) e] = (float) (0.5 * (e + 1) - 0.25 * t);

    if (mode == "layers") {
        // two layers, one direction each: layer 0 steered, layer 1 all zero
        const int L = 2, K = 1;
        const std::vector<float> v0 = direction(E, 0);
        std::vector<float> file(v0);
        file.resize((size_t) L * (size_t) K * (size_t) E, 0.0f);
        const std::string path = write_file(file, "maya-glm-ablate-test-layers.bin");
        if (path.empty()) { std::printf("FAIL: no temporary file\n"); return 1; }
        setenv("STRATA_GLM_ABLATE", path.c_str(), 1);
        setenv("STRATA_GLM_ABLATE_LAYERS", "2", 1);
        check(glm_ablate::on(), "the settings load: the harness is on");
        std::vector<float> out;
        check(run_device(X, T, E, 0, out), "layer 0 runs");
        check(close(out, project(X, file, T, E, K, 0, 1.0)), "layer 0: x - (x . v) v on every row");
        std::vector<float> out1;
        check(run_device(X, T, E, 1, out1), "layer 1 runs");
        check(out1 == X, "layer 1: all zero - the rows are untouched");
        std::remove(path.c_str());
    } else {
        const int K = 2;
        const std::vector<float> v0 = direction(E, 0), v1 = direction(E, 1);
        std::vector<float> file(v0);
        file.insert(file.end(), v1.begin(), v1.end());
        const char* name = mode == "scale" ? "maya-glm-ablate-test-scale.bin" : "maya-glm-ablate-test-outputs.bin";
        const std::string path = write_file(file, name);
        if (path.empty()) { std::printf("FAIL: no temporary file\n"); return 1; }
        const double scale = mode == "scale" ? 0.5 : 1.0;
        char sc[32] = {};
        std::snprintf(sc, sizeof sc, "%.6g", scale);
        setenv("STRATA_GLM_ABLATE", path.c_str(), 1);
        setenv("STRATA_GLM_ABLATE_SCALE", sc, 1);
        check(glm_ablate::on(), "the settings load: the harness is on");
        std::vector<float> out;
        check(run_device(X, T, E, -1, out), "the rows kernel runs");
        const std::vector<double> want = project(X, file, T, E, K, 0, scale);
        check(close(out, want), mode == "scale" ? "half the component along each direction is left"
                                               : "x - (x . v) v for every direction, on every row");
        // and the property the harness is for: no component along either direction survives, times the scale
        double worst = 0.0;
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                const float* v = file.data() + (size_t) k * (size_t) E;
                const double got = dot(std::vector<double>(out.begin(), out.end()), v, E, t);
                const double was = dot(std::vector<double>(X.begin(), X.end()), v, E, t);
                worst = std::max(worst, std::fabs(got - (1.0 - scale) * was));
            }
        }
        const bool ok = worst <= 2e-4;
        if (!ok) std::printf("    worst residual component %.6g (want %.6g)\n", worst, 2e-4);
        check(ok, scale == 1.0 ? "no component along a direction survives" : "the component along a direction is halved");
        std::remove(path.c_str());
    }

    std::printf("glm_ablate_test: %s, %d checks, %d failures\n", mode.c_str(), g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
