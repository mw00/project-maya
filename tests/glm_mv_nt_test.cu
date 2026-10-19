// tests/glm_mv_nt_test.cu - the dense GEMV for several activations at once (mv_nt) against the one-activation GEMV
// (mv), bit for bit: every weight type the decode reads, 2 to 32 activations, a launch of mixed types (the typed and
// the generic kernels), a scale and a bias. Synthetic weights from ggml's own quantizers; no model needed.
//
//   glm_mv_nt_test           the parity checks (ctest; with STRATA_GLM_MV_GENERIC=1 too: the generic kernel only)
//   glm_mv_nt_test --bench   the time of one Q6_K 4096 x 12288 GEMV (the dense FFN's gate) for 1..32 activations
#include "strata/kernels/glm_fast.hpp"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_mv_nt_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int kQ8 = 36;   // bytes per q8_1 block

struct Weights {
    int type = 0, n_in = 0, n_out = 0;
    void* d = nullptr;
};

// n_out rows of n_in weights ~ N(0, 0.05): F32 and BF16 as they are, the rest through ggml's quantizer (an importance
// matrix of ones for the types that need one)
Weights make_weights(int type, int n_in, int n_out, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> f((size_t) n_in * n_out);
    for (auto& v : f) v = nd(rng);
    const size_t bytes = gf::row_bytes(type, n_in) * (size_t) n_out;
    std::vector<uint8_t> host(bytes);
    if (type == gf::kTypeF32) {
        std::memcpy(host.data(), f.data(), bytes);
    } else if (type == gf::kTypeBF16) {
        for (size_t i = 0; i < f.size(); ++i) {
            uint32_t b;
            std::memcpy(&b, &f[i], 4);
            const uint16_t h = (uint16_t) (b >> 16);
            std::memcpy(host.data() + 2 * i, &h, 2);
        }
    } else {
        const std::vector<float> imatrix((size_t) n_in, 1.0f);
        const size_t wrote = ggml_quantize_chunk((ggml_type) type, f.data(), host.data(), 0, n_out, n_in,
                                                 ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data()
                                                                                                  : nullptr);
        if (wrote != bytes) {
            std::fprintf(stderr, "glm_mv_nt_test: type %d: ggml wrote %zu bytes, the kernels read %zu\n", type, wrote,
                         bytes);
            std::exit(1);
        }
    }
    Weights w{type, n_in, n_out, nullptr};
    check(cudaMalloc(&w.d, bytes), "weights");
    check(cudaMemcpy(w.d, host.data(), bytes, cudaMemcpyHostToDevice), "weights upload");
    return w;
}

// kMaxMvCols activations of n values ~ N(0, 1), one after another: the floats and, quantized one activation at a time
// by the engine's own quantizer, their q8_1 blocks
struct Acts {
    float* xf = nullptr;
    uint8_t* xq = nullptr;
};
Acts make_acts(int n, std::mt19937& rng, cudaStream_t s) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const int ncols = std::max(gf::kMaxMvCols, gf::kMaxMvNt);
    std::vector<float> h((size_t) n * ncols);
    for (auto& v : h) v = nd(rng);
    Acts a;
    check(cudaMalloc(&a.xf, h.size() * sizeof(float)), "activations");
    check(cudaMalloc(&a.xq, (size_t) n / 32 * kQ8 * ncols), "activations q8_1");
    check(cudaMemcpy(a.xf, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "activations upload");
    for (int t = 0; t < ncols; ++t)
        gf::quantize_q8_1(a.xf + (size_t) t * n, a.xq + (size_t) t * (n / 32) * kQ8, n, s);
    check(cudaStreamSynchronize(s), "quantize");
    return a;
}

struct Job {
    Weights w;
    Acts x;
    float alpha = 1.0f;
    float* bias = nullptr;
};

gf::MvJob mv_job(const Job& j, int t, float* y) {
    gf::MvJob m;
    m.w = j.w.d;
    m.xq = j.x.xq + (size_t) t * (j.w.n_in / 32) * kQ8;
    m.xf = j.x.xf + (size_t) t * j.w.n_in;
    m.y = y;
    m.bias = j.bias;
    m.alpha = j.alpha;
    m.type = j.w.type;
    m.n_in = j.w.n_in;
    m.n_out = j.w.n_out;
    return m;
}

// these jobs for nt activations, both ways the engine batches them - one mv_nt launch (several conversations, nt <=
// kMaxMvNt) and one mv() of nt-column jobs (a verify window's tokens) - against mv() for each activation alone;
// false: a mismatch (printed)
bool run_case(const char* name, const std::vector<Job>& jobs, int nt, cudaStream_t s) {
    const int n = (int) jobs.size();
    std::vector<float*> y_nt((size_t) n), y_col((size_t) n), y_one((size_t) n);
    std::vector<gf::MvJob> batch((size_t) n), cols((size_t) n);
    for (int i = 0; i < n; ++i) {
        const Job& j = jobs[(size_t) i];
        const size_t bytes = (size_t) nt * j.w.n_out * sizeof(float);
        check(cudaMalloc(&y_nt[(size_t) i], bytes), "outputs");
        check(cudaMalloc(&y_col[(size_t) i], bytes), "outputs");
        check(cudaMalloc(&y_one[(size_t) i], bytes), "outputs");
        check(cudaMemset(y_nt[(size_t) i], 0xff, bytes), "outputs");
        check(cudaMemset(y_col[(size_t) i], 0xff, bytes), "outputs");
        check(cudaMemset(y_one[(size_t) i], 0x7f, bytes), "outputs");
        batch[(size_t) i] = mv_job(j, 0, y_nt[(size_t) i]);
        cols[(size_t) i] = mv_job(j, 0, y_col[(size_t) i]);
        cols[(size_t) i].ncol = nt;
        cols[(size_t) i].xq_cstride = (int64_t) (j.w.n_in / 32) * kQ8;
        cols[(size_t) i].xf_cstride = j.w.n_in;
        cols[(size_t) i].y_cstride = j.w.n_out;
    }
    const bool with_nt = nt <= gf::kMaxMvNt;
    if (with_nt && !gf::mv_nt(batch.data(), n, nt, s)) {
        std::fprintf(stderr, "%s: mv_nt refused %d jobs x %d activations\n", name, n, nt);
        return false;
    }
    if (nt <= gf::kMaxMvCols && !gf::mv(cols.data(), n, s)) {
        std::fprintf(stderr, "%s: mv refused %d jobs of %d columns\n", name, n, nt);
        return false;
    }
    for (int t = 0; t < nt; ++t) {
        std::vector<gf::MvJob> one((size_t) n);
        for (int i = 0; i < n; ++i)
            one[(size_t) i] = mv_job(jobs[(size_t) i], t, y_one[(size_t) i] + (size_t) t * jobs[(size_t) i].w.n_out);
        if (!gf::mv(one.data(), n, s)) {
            std::fprintf(stderr, "%s: mv refused the jobs\n", name);
            return false;
        }
    }
    check(cudaStreamSynchronize(s), name);
    check(cudaGetLastError(), name);
    bool ok = gf::launch_errors() == 0;
    for (int i = 0; i < n; ++i) {
        const Job& j = jobs[(size_t) i];
        std::vector<float> b((size_t) nt * j.w.n_out);
        check(cudaMemcpy(b.data(), y_one[(size_t) i], b.size() * sizeof(float), cudaMemcpyDeviceToHost), "read back");
        for (const int form : {0, 1}) {
            if ((form == 0 && !with_nt) || (form == 1 && nt > gf::kMaxMvCols)) continue;
            std::vector<float> a(b.size());
            check(cudaMemcpy(a.data(), form == 0 ? y_nt[(size_t) i] : y_col[(size_t) i], a.size() * sizeof(float),
                             cudaMemcpyDeviceToHost), "read back");
            size_t bad = 0, first = 0;
            for (size_t k = 0; k < a.size(); ++k)
                if (std::memcmp(&a[k], &b[k], sizeof(float)) != 0 || !std::isfinite(b[k])) {
                    if (bad++ == 0) first = k;
                }
            if (bad > 0) {
                std::fprintf(stderr, "%s (%s): job %d (type %d, %d x %d), %d activations: %zu of %zu outputs differ - "
                             "first: activation %zu row %zu, %.9g against %.9g\n", name, form == 0 ? "mv_nt" : "columns",
                             i, j.w.type, j.w.n_in, j.w.n_out, nt, bad, a.size(), first / (size_t) j.w.n_out,
                             first % (size_t) j.w.n_out, a[first], b[first]);
                ok = false;
            }
        }
        cudaFree(y_nt[(size_t) i]);
        cudaFree(y_col[(size_t) i]);
        cudaFree(y_one[(size_t) i]);
    }
    return ok;
}

int bench(cudaStream_t s) {
    std::mt19937 rng(7);
    const Weights w = make_weights(14, 4096, 12288, rng);
    const Acts x = make_acts(4096, rng, s);
    float* y = nullptr;
    check(cudaMalloc(&y, (size_t) gf::kMaxMvCols * 12288 * sizeof(float)), "outputs");
    Job j{w, x, 1.0f, nullptr};
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    double one = 0;
    std::printf("Q6_K 4096 x 12288 (%.1f MB), one launch:\n", (double) gf::row_bytes(14, 4096) * 12288 / 1e6);
    for (int nt = 1; nt <= gf::kMaxMvNt; ++nt) {
        const gf::MvJob m = mv_job(j, 0, y);
        for (int i = 0; i < 5; ++i) gf::mv_nt(&m, 1, nt, s);
        constexpr int kCalls = 50;
        check(cudaEventRecord(a, s), "record");
        for (int i = 0; i < kCalls; ++i) gf::mv_nt(&m, 1, nt, s);
        check(cudaEventRecord(b, s), "record");
        check(cudaEventSynchronize(b), "sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, a, b), "elapsed");
        const double us = 1000.0 * ms / kCalls;
        if (nt == 1) one = us;
        std::printf("  %d activation%s: %7.1f us  (%.2f of one activation's time, %.2f per activation)\n", nt,
                    nt == 1 ? " " : "s", us, us / one, us / one / nt);
    }
    check(cudaGetLastError(), "the timed launches");
    if (gf::launch_errors() > 0) {
        std::fprintf(stderr, "glm_mv_nt_test: %d launch(es) failed: the times above are not real\n", gf::launch_errors());
        return 1;
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench(s);
    std::mt19937 rng(1234);
    // every type mv() serves: F32, BF16 (the pack's small and big float rows), the K-quants and Q8_0 (typed kernels),
    // the rest (the generic kernel).  Two row counts: a tail of 4 rows past the last full block of 8, and one block.
    // all[2 * i] is types[i] at 4096 x 300, all[2 * i + 1] at 1536 x 8
    const int types[] = {gf::kTypeF32, gf::kTypeBF16, 14, 8, 12, 13, 10, 11, 16, 17, 18, 19, 21, 22, 23, 29};
    const Acts x4096 = make_acts(4096, rng, s), x1536 = make_acts(1536, rng, s);
    int failed = 0, cases = 0;
    std::vector<Job> all;
    for (const int type : types) {
        for (const auto& shape : {std::pair<int, int>{4096, 300}, std::pair<int, int>{1536, 8}}) {
            Job j;
            j.w = make_weights(type, shape.first, shape.second, rng);
            j.x = shape.first == 4096 ? x4096 : x1536;
            all.push_back(j);
            for (int nt = 2; nt <= gf::kMaxMvCols; ++nt) {
                const std::string name = "type " + std::to_string(type) + " " + std::to_string(shape.first) + "x" +
                                         std::to_string(shape.second);
                ++cases;
                if (!run_case(name.c_str(), {j}, nt, s)) ++failed;
            }
        }
    }
    // a scale and a bias (the DSA indexer's weights projection; the BF16 rows with a bias)
    {
        float* bias = nullptr;
        std::vector<float> hb(300);
        for (size_t i = 0; i < hb.size(); ++i) hb[i] = 0.01f * (float) i - 1.0f;
        check(cudaMalloc(&bias, hb.size() * sizeof(float)), "bias");
        check(cudaMemcpy(bias, hb.data(), hb.size() * sizeof(float), cudaMemcpyHostToDevice), "bias");
        for (const int pick : {0, 4, 8, 16}) {   // F32, Q6_K, Q4_K, IQ2_XXS at 4096 x 300
            Job j = all[(size_t) pick];
            j.alpha = 0.0883883f;
            j.bias = bias;
            for (const int nt : {2, 5, 8}) {
                ++cases;
                if (!run_case("scaled, with a bias", {j}, nt, s)) ++failed;
            }
        }
    }
    // mixed launches as the decode issues them: the KDA projections (three quantized, three BF16), the DSA ones, the
    // router with the shared expert's gate and up, eight jobs at once
    {
        const std::vector<std::vector<int>> mixes = {{4, 4, 4, 2, 2, 2}, {4, 4, 2, 2, 2}, {2, 4, 4},
                                                     {4, 6, 8, 10, 16, 18, 26, 30}, {5, 9, 13, 17}};
        for (const auto& mix : mixes) {
            std::vector<Job> jobs;
            for (const int k : mix) jobs.push_back(all[(size_t) k]);
            for (int nt = 2; nt <= gf::kMaxMvCols; ++nt) {
                ++cases;
                if (!run_case("mixed launch", jobs, nt, s)) ++failed;
            }
        }
    }
    // one activation is mv() itself
    {
        ++cases;
        if (!run_case("one activation", {all[4], all[2]}, 1, s)) ++failed;
    }
    if (failed > 0) {
        std::printf("glm_mv_nt_test: %d of %d cases FAILED\n", failed, cases);
        return 1;
    }
    std::printf("glm_mv_nt_test: %d cases, every output bit for bit the one-activation GEMV's%s\n", cases,
                std::getenv("STRATA_GLM_MV_GENERIC") ? " (the generic kernel)" : "");
    return 0;
}
