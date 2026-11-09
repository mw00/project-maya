// tests/glm_dsa/glmd_f16_guard_gpu_test.cu - glm-dsa B15.1 on a V100: the FP16 activation range guard on the device.
//   rows    glmd::f16_guard_rows after a plain FP16 conversion (plain rows and pack_f16's head-major layout): the
//           rewritten bits and the scales == glmd::guard_twin exactly; rows without an outlier untouched
//   swiglu  glm_batch.cu swiglu_f16 + glmd::f16_guard_swiglu == the twin within one FP16 ulp (host vs device expf)
//   ffn     the dense FFN as the prompt path runs it (cuBLAS FP16 GEMMs, F32 accumulate, the f16_guard_after
//           epilogue) over amax 1e5 rows: finite and within 4e-3 of the twin pipeline; with STRATA_GLM_F16_GUARD=0
//           the outlier rows are not finite and every other row is bit-identical to the guarded run
//   mla     glmd::mla_prompt_wmma (16 heads) and mla_prompt_h64 (64 heads) with a Q tile past 65504 (q_abs 1.5e6 x
//           the 0.0625 scale) vs the F32 twin
// Needs a GPU; exits 77 (ctest SKIP_RETURN_CODE) when there is none.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/glmd_batch.hpp"
#include "strata/kernels/glmd_f16_guard.hpp"
#include "strata/kernels/glmd_fast.hpp"

#include "glm_test_env.hpp"   // setenv (<unistd.h> on POSIX; mapped on Windows)

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace sk = strata::kernels;
namespace gt = glmd::guard_twin;

int g_fail = 0;
void expect(bool ok, const char* what) {
    std::printf("%-100s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

template <typename T>
T* up(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n);
    cudaDeviceSynchronize();
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}
std::vector<float> randn(size_t n, float sd, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

__global__ void convert_kernel(const float* src, int64_t src_row, int64_t src_head, int H, int K, int T, __half* dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * H * K) return;
    const int64_t t = i / ((int64_t) H * K), r = i - t * H * K, h = r / K, k = r - h * K;
    dst[i] = __float2half(src[t * src_row + h * src_head + k]);
}

void test_rows(bool heads) {
    const int T = 5, H = heads ? 4 : 1, K = heads ? 64 : 512;
    const int64_t src_row = heads ? 4 * 96 : 512, src_head = heads ? 96 : 0;
    auto src = randn((size_t) T * src_row, 3.0f, 3);
    src[(size_t) 1 * src_row + 7] = 1e5f;
    src[(size_t) 3 * src_row + (heads ? 2 * 96 + 5 : 300)] = -2.5e6f;
    float* d_src = up(src);
    uint16_t* d_dst = nullptr;
    cudaMalloc(&d_dst, (size_t) T * H * K * 2);
    cudaStream_t s;
    cudaStreamCreate(&s);
    convert_kernel<<<(T * H * K + 255) / 256, 256, 0, s>>>(d_src, src_row, src_head, H, K, T, (__half*) d_dst);
    const bool ok_call = glmd::f16_guard_rows(d_src, src_row, src_head, H, K, T, d_dst, s, true);
    const float* d_sc = glmd::f16_guard_scales_for(d_dst, s);
    std::vector<uint16_t> want((size_t) T * H * K);
    std::vector<float> sc_want(T);
    gt::pack_f16(src.data(), src_row, src_head, H, K, T, want.data());
    gt::guard_rows(src.data(), src_row, src_head, H, K, T, want.data(), sc_want.data());
    const auto got = down(d_dst, want.size());
    const auto sc = d_sc ? down(d_sc, (size_t) T) : std::vector<float>();
    char what[200];
    std::snprintf(what, sizeof what, "rows %s: bits == twin, scales %g %g %g %g %g == twin", heads ? "(heads)" : "(plain)",
                  sc.empty() ? -1.0 : sc[0], sc.empty() ? -1.0 : sc[1], sc.empty() ? -1.0 : sc[2],
                  sc.empty() ? -1.0 : sc[3], sc.empty() ? -1.0 : sc[4]);
    expect(ok_call && got == want && sc == sc_want, what);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaStreamDestroy(s);
}

void test_swiglu() {
    const int T = 4, FF = 768;
    auto g = randn((size_t) T * FF, 4.0f, 5), u = randn((size_t) T * FF, 4.0f, 6);
    g[(size_t) 2 * FF + 9] = 3e3f;
    u[(size_t) 2 * FF + 9] = 1e3f;   // h = 3e6
    float *d_g = up(g), *d_u = up(u);
    uint16_t* d_h = nullptr;
    cudaMalloc(&d_h, (size_t) T * FF * 2);
    cudaStream_t s;
    cudaStreamCreate(&s);
    sk::glmb::swiglu_f16(d_g, d_u, d_h, (int64_t) T * FF, INFINITY, s);
    glmd::f16_guard_swiglu(d_g, d_u, FF, T, INFINITY, d_h, s);
    std::vector<uint16_t> want((size_t) T * FF);
    std::vector<float> sc(T);
    gt::swiglu_f16(g.data(), u.data(), (int64_t) T * FF, INFINITY, want.data());
    gt::guard_swiglu(g.data(), u.data(), FF, T, INFINITY, want.data(), sc.data());
    const auto got = down(d_h, want.size());
    int worst = 0;
    for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs((int) got[i] - (int) want[i]));
    char what[160];
    std::snprintf(what, sizeof what, "swiglu: h16 within one FP16 ulp of the twin (worst %d), row 2 scale %g", worst,
                  (double) sc[2]);
    expect(worst <= 1 && sc[2] > 1.0f, what);
    cudaFree(d_g);
    cudaFree(d_u);
    cudaFree(d_h);
    cudaStreamDestroy(s);
}

// Y[t][N] = X16[t][K] . W16[N][K]^T, as hgemm_q calls cuBLAS, then the epilogue
void gemm(cublasHandle_t h, const uint16_t* X16, const uint16_t* W16, float* Y, int T, int N, int K, cudaStream_t s) {
    const float one = 1.0f, zero = 0.0f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W16, CUDA_R_16F, K, X16, CUDA_R_16F, K, &zero, Y,
                 CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    glmd::f16_guard_after(X16, K, Y, N, N, T, 0.0f, s);
}

std::vector<float> ffn_device(cublasHandle_t h, cudaStream_t s, const std::vector<float>& x, const uint16_t* wg,
                              const uint16_t* wu, const uint16_t* wd, int T, int E, int FF) {
    float* d_x = up(x);
    uint16_t *d_x16, *d_h16;
    float *d_g, *d_u, *d_y;
    cudaMalloc(&d_x16, (size_t) T * E * 2);
    cudaMalloc(&d_h16, (size_t) T * FF * 2);
    cudaMalloc(&d_g, (size_t) T * FF * 4);
    cudaMalloc(&d_u, (size_t) T * FF * 4);
    cudaMalloc(&d_y, (size_t) T * E * 4);
    convert_kernel<<<(T * E + 255) / 256, 256, 0, s>>>(d_x, E, 0, 1, E, T, (__half*) d_x16);
    glmd::f16_guard_rows(d_x, E, 0, 1, E, T, d_x16, s, true);
    gemm(h, d_x16, wg, d_g, T, FF, E, s);
    gemm(h, d_x16, wu, d_u, T, FF, E, s);
    sk::glmb::swiglu_f16(d_g, d_u, d_h16, (int64_t) T * FF, INFINITY, s);
    glmd::f16_guard_swiglu(d_g, d_u, FF, T, INFINITY, d_h16, s);
    gemm(h, d_h16, wd, d_y, T, E, FF, s);
    auto y = down(d_y, (size_t) T * E);
    for (void* p : {(void*) d_x, (void*) d_x16, (void*) d_h16, (void*) d_g, (void*) d_u, (void*) d_y}) cudaFree(p);
    return y;
}

void test_ffn() {
    const int T = 6, E = 512, FF = 384, hot = 2;
    auto wgf = randn((size_t) FF * E, 0.05f, 7), wuf = randn((size_t) FF * E, 0.05f, 8), wdf = randn((size_t) E * FF, 0.05f, 9);
    std::vector<uint16_t> wg(wgf.size()), wu(wuf.size()), wd(wdf.size());
    for (size_t i = 0; i < wg.size(); ++i) wg[i] = sk::f16_from_f32(wgf[i]);
    for (size_t i = 0; i < wu.size(); ++i) wu[i] = sk::f16_from_f32(wuf[i]);
    for (size_t i = 0; i < wd.size(); ++i) wd[i] = sk::f16_from_f32(wdf[i]);
    uint16_t *d_wg = up(wg), *d_wu = up(wu), *d_wd = up(wd);
    auto x = randn((size_t) T * E, 2.0f, 10);
    x[(size_t) hot * E + 17] = 1e5f;
    cublasHandle_t h;
    cublasCreate(&h);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cublasSetStream(h, s);
    setenv("STRATA_GLM_F16_GUARD", "1", 1);
    const auto y1 = ffn_device(h, s, x, d_wg, d_wu, d_wd, T, E, FF);
    setenv("STRATA_GLM_F16_GUARD", "0", 1);
    const auto y0 = ffn_device(h, s, x, d_wg, d_wu, d_wd, T, E, FF);
    setenv("STRATA_GLM_F16_GUARD", "1", 1);
    // the twin pipeline (guarded)
    std::vector<uint16_t> x16((size_t) T * E), h16((size_t) T * FF);
    std::vector<float> g((size_t) T * FF), u((size_t) T * FF), y((size_t) T * E), sx(T), sh(T);
    gt::pack_f16(x.data(), E, 0, 1, E, T, x16.data());
    gt::guard_rows(x.data(), E, 0, 1, E, T, x16.data(), sx.data());
    gt::gemm_f16(x16.data(), E, wg.data(), g.data(), FF, T, FF, E);
    gt::unscale_rows(g.data(), FF, FF, T, sx.data());
    gt::gemm_f16(x16.data(), E, wu.data(), u.data(), FF, T, FF, E);
    gt::unscale_rows(u.data(), FF, FF, T, sx.data());
    gt::swiglu_f16(g.data(), u.data(), (int64_t) T * FF, INFINITY, h16.data());
    gt::guard_swiglu(g.data(), u.data(), FF, T, INFINITY, h16.data(), sh.data());
    gt::gemm_f16(h16.data(), FF, wd.data(), y.data(), E, T, E, FF);
    gt::unscale_rows(y.data(), E, E, T, sh.data());
    bool finite = true, same = true, hot_bad = false;
    double num = 0.0, den = 0.0;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < E; ++c) {
            const size_t i = (size_t) t * E + c;
            finite = finite && std::isfinite(y1[i]);
            num += ((double) y1[i] - y[i]) * ((double) y1[i] - y[i]);
            den += (double) y[i] * y[i];
            if (t != hot) same = same && std::memcmp(&y1[i], &y0[i], 4) == 0;
            else hot_bad = hot_bad || !std::isfinite(y0[i]);
        }
    char what[200];
    std::snprintf(what, sizeof what, "ffn: guarded finite, rel %.2e vs the twin (< 4e-3); unguarded outlier row %s; others bit-identical",
                  std::sqrt(num / den), hot_bad ? "inf/NaN" : "finite (?)");
    expect(finite && std::sqrt(num / den) < 4e-3 && hot_bad && same, what);
    cublasDestroy(h);
    cudaStreamDestroy(s);
    for (void* p : {(void*) d_wg, (void*) d_wu, (void*) d_wd}) cudaFree(p);
}

void test_mla(bool h64) {
    if (!glmd::mla_prompt_wmma_ok()) {
        expect(true, "mla: the WMMA kernel is not available on this device (skipped)");
        return;
    }
    const int T = 4, H = h64 ? 64 : 16, K = 512, R = 64;
    const int64_t p0 = 60, n_kv = p0 + T;
    auto qa = randn((size_t) T * H * K, 0.5f, 12), q = randn((size_t) T * H * 256, 1.0f, 13);
    qa[((size_t) 1 * H + 3) * K + 100] = 1.5e6f;   // x 0.0625 = 93,750 > 65504: this tile's FP16 store overflows
    std::vector<uint16_t> lat((size_t) n_kv * K), kpe((size_t) n_kv * R);
    auto l = randn(lat.size(), 1.0f, 14), kp = randn(kpe.size(), 1.0f, 15);
    for (size_t i = 0; i < lat.size(); ++i) lat[i] = glmd::twin::half_bits(l[i]);
    for (size_t i = 0; i < kpe.size(); ++i) kpe[i] = glmd::twin::half_bits(kp[i]);
    glmd::MlaPrompt a;
    a.walk = glmd::AttnWalk::Dense;
    a.p0 = p0;
    a.T = T;
    a.n_head = H;
    a.kv_lora = K;
    a.qk_rope = R;
    std::vector<float> ref((size_t) T * H * K);
    a.q_abs = qa.data();
    a.q = q.data();
    a.lat = lat.data();
    a.k_pe = kpe.data();
    a.ctx = ref.data();
    glmd::twin::mla_prompt(a);
    glmd::MlaPrompt d = a;
    d.q_abs = up(qa);
    d.q = up(q);
    d.lat = up(lat);
    d.k_pe = up(kpe);
    float* d_ctx = nullptr;
    cudaMalloc(&d_ctx, ref.size() * 4);
    d.ctx = d_ctx;
    const bool ran = h64 ? glmd::mla_prompt_h64_fits(d) && glmd::mla_prompt_h64(d, nullptr)
                         : glmd::mla_prompt_wmma(d, nullptr);
    const auto got = down(d_ctx, ref.size());
    bool finite = true;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        finite = finite && std::isfinite(got[i]);
        num += ((double) got[i] - ref[i]) * ((double) got[i] - ref[i]);
        den += (double) ref[i] * ref[i];
    }
    char what[160];
    std::snprintf(what, sizeof what, "mla %s: an overflowing Q tile stays finite, rel L2 %.2e vs the F32 twin (< 2e-3)",
                  h64 ? "h64 (64 heads)" : "wmma", std::sqrt(num / den));
    expect(ran && finite && std::sqrt(num / den) < 2e-3, what);
    for (const void* p : {(const void*) d.q_abs, (const void*) d.q, d.lat, (const void*) d.k_pe, (const void*) d_ctx})
        cudaFree((void*) p);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    // the default (unset) and auto are on for every device (the prompt GEMMs take FP16 activations on all of them)
    unsetenv("STRATA_GLM_F16_GUARD");
    expect(glmd::f16_guard_active(), "auto (unset): the guard is on for this device");
    setenv("STRATA_GLM_F16_GUARD", "auto", 1);
    expect(glmd::f16_guard_active(), "auto: the guard is on for this device");
    setenv("STRATA_GLM_F16_GUARD", "0", 1);
    expect(!glmd::f16_guard_active(), "STRATA_GLM_F16_GUARD=0: off");
    setenv("STRATA_GLM_F16_GUARD", "1", 1);
    test_rows(false);
    test_rows(true);
    test_swiglu();
    test_ffn();
    test_mla(false);
    test_mla(true);
    char what[80];
    std::snprintf(what, sizeof what, "no guard errors (%llu)", (unsigned long long) glmd::f16_guard_errors());
    expect(glmd::f16_guard_errors() == 0, what);
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
