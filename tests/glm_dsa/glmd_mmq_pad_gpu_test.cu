// tests/glm_dsa/glmd_mmq_pad_gpu_test.cu - glm-dsa B15.2 on a V100: MMQ over expert banks whose surroundings are NaN.
//
// A bank of 4 experts is laid out as the prompt path's staging groups are: slots `stride` apart (stride > blob, so
// every slot has a gap; whole blocks and 16 bytes, as glmfast::expert_stride rounds it: MMQ finds expert e at
// e * stride in whole blocks), inside a region whose every other byte is 0xFF (NaN in any FP16 / F32 field):
//   clean    the reference: the same experts in a zeroed allocation (mmq_malloc) with nothing else around them; its
//            product must be finite and not all zero, or the comparison below proves nothing
//   guarded  NaN everywhere, then the experts copied in and glmd::mmq_zero_gaps (each slot's gap and 512 bytes after
//            the last slot) - the product must be finite and bit-identical to the clean one
//   bare     informational: NaN gaps and NaN right after the last expert (what B15.2 removes); prints whether this
//            build's MMQ read any of it
// Each product is checked against mmq::launch_errors() (moe_mmq.cu counts every product it refuses and prints the
// first 8 on stderr): clean, guarded and bare must launch; the stride this test used to pass (blob + 1024: not whole
// blocks, so MMQ truncated it and read experts 1..3 off their block boundaries) must be refused, dst untouched.
// Per type the prompt path uses for routed experts on the test PC (the bring-up's UD-IQ1_S trunk: IQ1_S, IQ2_XXS, IQ3_XXS,
// IQ4_XS; other quants: IQ2_S, Q2_K, Q4_K) at the GLM-5.3 expert shapes (gate+up 2 x 2048 x 6144, down 6144 x 2048),
// each first through the prompt path's MMQ gate (mmq::supported, B14.2's sm_70 instance check): a refused type FAILs.
// Needs a GPU; exits 77 when there is none.
#include "strata/kernels/glmd_mmq_pad.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_mmq_strata.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;
namespace ma = strata::prefill::mmq_strata;

namespace {

int g_fail = 0;
void expect(bool ok, const char* what) {
    std::printf("%-110s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

std::vector<float> run(const void* w, size_t stride, int type, int N, int K, int n_exp, const std::vector<int>& bounds,
                       const void* d_xq, int rows, int max_rows) {
    int32_t *d_b, *d_ids;
    float* d_y;
    cudaMalloc(&d_b, bounds.size() * 4);
    cudaMalloc(&d_ids, (size_t) rows * 4);
    cudaMalloc(&d_y, (size_t) rows * N * 4);
    cudaMemcpy(d_b, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, (size_t) rows * N * 4);
    mmq::iota(d_ids, rows, nullptr);
    mmq::Context ctx;
    mmq::Product p;
    p.w = w;
    p.type = type;
    p.w_rows = N;
    p.w_cols = K;
    p.expert_bytes = stride;
    p.n = n_exp;
    p.xq = d_xq;
    p.bounds = d_b;
    p.ids = d_ids;
    p.total_rows = rows;
    p.max_rows = max_rows;
    p.dst = d_y;
    p.ld_dst = N;
    ctx.run(p, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> y((size_t) rows * N);
    cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(d_b);
    cudaFree(d_ids);
    cudaFree(d_y);
    return y;
}

void run_type(int type, int N, int K) {
    const ggml_type t = (ggml_type) type;
    const int n_exp = 4;
    const size_t blob = mmq::matrix_bytes(type, N, K), tsz = ggml_type_size(t);
    // a gap of at least 1000 bytes a slot, the stride rounded to whole blocks and 16 bytes (glmfast::expert_stride)
    const size_t a = std::lcm(tsz, (size_t) 16);
    const size_t stride = (blob + 1000 + a - 1) / a * a;
    std::mt19937 rng(type * 13 + N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint8_t> wq(blob * n_exp);
    std::vector<float> wf((size_t) N * K), imat((size_t) K, 1.0f);
    ggml_quantize_init(t);
    for (int e = 0; e < n_exp; ++e) {
        for (auto& v : wf) v = nd(rng) * 0.05f;
        ggml_quantize_chunk(t, wf.data(), wq.data() + e * blob, 0, N, K, ggml_quantize_requires_imatrix(t) ? imat.data() : nullptr);
    }
    std::vector<int> bounds = {0, 37, 37 + 1, 37 + 1 + 129, 37 + 1 + 129 + 64};   // rows per expert 37 / 1 / 129 / 64
    const int rows = bounds.back(), max_rows = 129;
    std::vector<float> x((size_t) rows * K);
    for (auto& v : x) v = nd(rng);
    float* d_x;
    void* d_xq;
    cudaMalloc(&d_x, x.size() * 4);
    cudaMalloc(&d_xq, mmq::q8_bytes(rows, K));
    cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    mmq::quantize(d_x, nullptr, d_xq, type, K, K, rows, nullptr);
    const size_t region = (size_t) n_exp * stride + 64 * 1024;   // the bank and NaN after it
    // mmq::launch_errors() before and after each product: moe_mmq.cu counts every product it refuses
    const int err0 = mmq::launch_errors();
    // clean: a zeroed allocation (mmq_malloc), experts at the slot starts
    void* clean = nullptr;
    const int e0 = glmd::mmq_malloc(&clean, region);
    for (int e = 0; e < n_exp; ++e)
        cudaMemcpy((uint8_t*) clean + (size_t) e * stride, wq.data() + e * blob, blob, cudaMemcpyHostToDevice);
    const auto y_clean = run(clean, stride, type, N, K, n_exp, bounds, d_xq, rows, max_rows);
    const int err1 = mmq::launch_errors();
    // guarded: NaN everywhere, the copies, then mmq_zero_gaps over the bank's room (its slots + the 512-B tail)
    void* dirty = nullptr;
    cudaMalloc(&dirty, region);
    cudaMemset(dirty, 0xFF, region);
    for (int e = 0; e < n_exp; ++e)
        cudaMemcpy((uint8_t*) dirty + (size_t) e * stride, wq.data() + e * blob, blob, cudaMemcpyHostToDevice);
    const bool zg = glmd::mmq_zero_gaps(dirty, n_exp, stride, blob, (size_t) n_exp * stride + glmd::kMmqTailPad, nullptr);
    const auto y_guard = run(dirty, stride, type, N, K, n_exp, bounds, d_xq, rows, max_rows);
    const int err2 = mmq::launch_errors();
    // bare: NaN in the gaps and right after the last expert
    cudaMemset(dirty, 0xFF, region);
    for (int e = 0; e < n_exp; ++e)
        cudaMemcpy((uint8_t*) dirty + (size_t) e * stride, wq.data() + e * blob, blob, cudaMemcpyHostToDevice);
    const auto y_bare = run(dirty, stride, type, N, K, n_exp, bounds, d_xq, rows, max_rows);
    const int err3 = mmq::launch_errors();
    // the old stride (blob + 1024, not whole blocks for any of these types): refused, so dst keeps run()'s zeros
    const size_t odd = blob + 1024;
    const auto y_odd = run(clean, odd, type, N, K, n_exp, bounds, d_xq, rows, max_rows);
    const int err4 = mmq::launch_errors();
    const bool clean_ran = err1 == err0, guard_ran = err2 == err1, bare_ran = err3 == err2;
    const bool refused = odd % tsz != 0 && err4 == err3 + 1;
    bool clean_finite = true, clean_nonzero = false, finite = true, bare_finite = true, odd_zero = true;
    for (float v : y_clean) {
        clean_finite = clean_finite && std::isfinite(v);
        clean_nonzero = clean_nonzero || v != 0.0f;
    }
    for (float v : y_guard) finite = finite && std::isfinite(v);
    for (float v : y_bare) bare_finite = bare_finite && std::isfinite(v);
    for (float v : y_odd) odd_zero = odd_zero && v == 0.0f;
    const bool same = std::memcmp(y_guard.data(), y_clean.data(), y_clean.size() * 4) == 0;
    char what[200];
    std::snprintf(what, sizeof what, "type %2d [%d x %d]: clean bank, %zu-block slots: launched, finite, not all zero",
                  type, N, K, stride / tsz);
    expect(e0 == 0 && stride % tsz == 0 && clean_ran && clean_finite && clean_nonzero, what);
    const char* bare = bare_finite ? "finite too" : "NOT finite: this MMQ reads outside the experts";
    std::snprintf(what, sizeof what,
                  "type %2d [%d x %d]: NaN past the 512-B tail and in zeroed gaps: launched, finite, == clean "
                  "(bare: %s)", type, N, K, bare_ran ? bare : "refused");
    expect(zg && guard_ran && bare_ran && finite && same && cudaGetLastError() == cudaSuccess, what);
    std::snprintf(what, sizeof what, "type %2d [%d x %d]: stride %zu B, not whole %zu-B blocks: refused, no launch",
                  type, N, K, odd, tsz);
    expect(refused && odd_zero, what);
    for (void* p : {clean, dirty, (void*) d_x, d_xq}) cudaFree(p);
}

}  // namespace

int main() {
    glmd::mmq_pad_enable(true);   // (load_pack does this for a glm-dsa model)
#ifdef _WIN32
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // (MSVC's CRT refuses _IOLBF with size 0)
#else
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // a case's lines reach ctest's log when it finishes, not at exit
#endif
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    // the prompt path's MMQ gate on this device (glmpf::mmq_ok: B14.2's sm_70 instance check with its cc, opt-in shared
    // memory and 32768 columns)
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    const int cc = 100 * prop.major + 10 * prop.minor;
    const int64_t optin = (int64_t) prop.sharedMemPerBlockOptin;
    for (int t : {(int) GGML_TYPE_IQ2_XXS, (int) GGML_TYPE_IQ2_S, (int) GGML_TYPE_IQ3_XXS, (int) GGML_TYPE_Q2_K,
                  (int) GGML_TYPE_Q4_K, (int) GGML_TYPE_IQ1_S, (int) GGML_TYPE_IQ4_XS}) {
        // a type the gate refuses is not multiplied with MMQ in production, and here it would hit NO_DEVICE_CODE (a
        // trap that ends the binary before the other types run): a FAIL with the reason, never a skip
        std::string why = "not covered by mmq::supported";
#if defined(STRATA_USE_HIP)
        // (a HIP build: glmpf::mmq_ok asks only mmq::supported - B14.2's instance rule is CUDA's sm_70 one)
        (void) cc;
        (void) optin;
        if (!mmq::supported(t)) {
#else
        if (!mmq::supported(t) || !ma::sm70_check(t, cc, optin, 32768, why)) {
#endif
            char what[200];
            std::snprintf(what, sizeof what, "type %2d: MMQ on CUDA%d (B14.2 instances): %s", t, dev, why.c_str());
            expect(false, what);
            continue;
        }
        run_type(t, 2 * 2048, 6144);
        run_type(t, 6144, 2048);
    }
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
