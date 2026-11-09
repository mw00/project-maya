// tests/glm_dsa/glmd_mv_parity.cu - glm-dsa B10.6 / B10.7 (label gpu): gf::mv on the new 32-value block types and F16
// against the host run of the same dots (glmd_mv_dots.cuh), and on the file's dominant types (Q8_0, Q2_K..Q6_K,
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS) against ggml's dequantized rows dotted with the dequantized q8_1 activation in
// double.  ctest runs it twice: as built (the default kernels) and with STRATA_GLMD_MV_NW2=1 STRATA_GLMD_MV_TYPED=1
// (the B10.6 variants), with the same gates, and prints the time per GEMV (256 x 6144 rows).
//   new types   GPU vs host twin (same algorithm): err / sum|.| < 1e-6
//   others      GPU vs double reference:          err / sum|.| < 1e-5
// Needs a GPU; exits 77 without one.
#include <cstdint>
#include <cstring>

#include <cuda_fp16.h>

#include "../../src/kernels/cuda/glmd_mv_dots.cuh"
#include "strata/kernels/glm_fast.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace {

int g_fail = 0;
#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

void gate(const char* what, double err, double lim) {
    const bool ok = err < lim;
    std::printf("%-60s %.3e (gate %.1e) %s\n", what, err, lim, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

std::vector<glmd_mvq::BQ8_1> q8_1(const std::vector<float>& x) {
    std::vector<glmd_mvq::BQ8_1> q(x.size() / 32);
    for (size_t b = 0; b < q.size(); ++b) {
        float amax = 0, sum = 0;
        for (int l = 0; l < 32; ++l) {
            amax = std::max(amax, std::fabs(x[32 * b + l]));
            sum += x[32 * b + l];
        }
        const float d = std::min(amax / 127.0f, 65504.0f);
        for (int l = 0; l < 32; ++l)
            q[b].qs[l] = amax == 0.0f ? 0 : (int8_t) std::max(-127.0f, std::min(127.0f, std::round(x[32 * b + l] / d)));
        q[b].d = ggml_fp32_to_fp16(d);
        q[b].s = ggml_fp32_to_fp16(sum);
    }
    return q;
}
float tree(float* s) {
    for (int o = 16; o > 0; o >>= 1)
        for (int l = 0; l < 32; ++l)
            if ((l & o) == 0) s[l] = s[l] + s[l + o];
    return s[0];
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("glmd_mv_parity: no CUDA device, skipped\n");
        return 77;
    }
    const char* nw2 = std::getenv("STRATA_GLMD_MV_NW2");
    const char* typed = std::getenv("STRATA_GLMD_MV_TYPED");
    std::printf("glmd_mv_parity: STRATA_GLMD_MV_NW2=%s STRATA_GLMD_MV_TYPED=%s\n", nw2 ? nw2 : "-", typed ? typed : "-");
    const int n_in = 6144, rows = 256;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(n_in);
    for (auto& v : x) v = nd(rng);
    const auto xq = q8_1(x);
    std::vector<float> xd(n_in);
    for (int e = 0; e < n_in; ++e) xd[e] = xq[e / 32].qs[e % 32] * glmd_mvq::h2f(xq[e / 32].d);
    void *d_xq, *d_w;
    float *d_xf, *d_y;
    CK(cudaMalloc(&d_xq, xq.size() * sizeof(glmd_mvq::BQ8_1)));
    CK(cudaMemcpy(d_xq, xq.data(), xq.size() * sizeof(glmd_mvq::BQ8_1), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_xf, n_in * 4));
    CK(cudaMemcpy(d_xf, x.data(), n_in * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_y, rows * 4));
    CK(cudaMalloc(&d_w, (size_t) rows * n_in * 4));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::vector<float> imat(n_in, 1.0f);
    struct Case {
        int type;
        bool f32_in;
    };
    const Case cases[] = {{2, false}, {3, false}, {6, false}, {7, false}, {20, false}, {1, true}, {1, false},
                          {8, false}, {10, false}, {11, false}, {12, false}, {13, false}, {14, false},
                          {16, false}, {17, false}, {22, false}, {18, false}};
    for (const Case& c : cases) {
        const ggml_type t = (ggml_type) c.type;
        const bool ext = glmd_mvq::type_ok(c.type);
        if (!gf::mv_supported_ext(c.type)) {
            std::printf("type %d: not supported by mv\n", c.type);
            ++g_fail;
            continue;
        }
        std::vector<float> w((size_t) rows * n_in);
        for (auto& v : w) v = nd(rng) * 0.05f;
        const size_t rb = gf::row_bytes(c.type, n_in);
        if (rb != ggml_row_size(t, n_in)) {
            std::printf("type %d: row bytes %zu != ggml %zu\n", c.type, rb, ggml_row_size(t, n_in));
            ++g_fail;
            continue;
        }
        std::vector<uint8_t> q(rb * rows);
        ggml_quantize_chunk(t, w.data(), q.data(), 0, rows, n_in, imat.data());
        CK(cudaMemcpy(d_w, q.data(), q.size(), cudaMemcpyHostToDevice));
        gf::MvJob j;
        j.w = d_w;
        j.xq = c.f32_in ? nullptr : d_xq;
        j.xf = c.f32_in ? d_xf : nullptr;
        j.y = d_y;
        j.type = c.type;
        j.n_in = n_in;
        j.n_out = rows;
        if (!gf::mv(&j, 1, nullptr)) {
            std::printf("type %d: mv refused\n", c.type);
            ++g_fail;
            continue;
        }
        std::vector<float> y(rows);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(y.data(), d_y, rows * 4, cudaMemcpyDeviceToHost));
        std::vector<float> deq(n_in);
        double err = 0;
        for (int r = 0; r < rows; ++r) {
            const uint8_t* row = q.data() + rb * r;
            double ref, mag = 0;
            if (ext) {   // the same algorithm on the host
                float lanes[32];
                for (int l = 0; l < 32; ++l)
                    lanes[l] = c.type == 1 ? glmd_mvq::f16_lane((const uint16_t*) row, c.f32_in ? x.data() : nullptr,
                                                                xq.data(), n_in, l)
                                           : glmd_mvq::lane_part(c.type, row, xq.data(), n_in, l, 0, 1);
                ref = tree(lanes);
            }
            ggml_get_type_traits(t)->to_float(row, deq.data(), n_in);
            double dref = 0;
            for (int e = 0; e < n_in; ++e) {
                const double xv = c.f32_in ? x[e] : xd[e];
                dref += (double) deq[e] * xv;
                mag += std::fabs((double) deq[e] * xv);
            }
            if (!ext) ref = dref;
            err = std::max(err, std::fabs(y[r] - ref) / mag);
        }
        float ms = 0;
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 50; ++i) gf::mv(&j, 1, nullptr);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1));
        char what[96];
        std::snprintf(what, sizeof what, "%s%s 256 x 6144 (%.1f us): err / sum|.|", ggml_type_name(t),
                      c.type == 1 ? (c.f32_in ? " f32 in" : " q8_1 in") : "", ms * 20.0f);
        gate(what, err, ext ? 1e-6 : 1e-5);
    }
    if (gf::launch_errors() != 0) {
        std::printf("launch errors: %d\n", gf::launch_errors());
        ++g_fail;
    }
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
