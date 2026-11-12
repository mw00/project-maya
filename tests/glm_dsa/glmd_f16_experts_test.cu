// tests/glm_dsa/glmd_f16_experts_test.cu - the prompt path's FP16 expert products (src/core/glm_prefill.cu,
// expert_f16): a routed-expert matrix whose type MMQ has no instance for - IQ1_M above all (the 170 GB GLM-5.3 quants
// carry it) - is dequantized to FP16 by mmq::to_f16 (llama.cpp's convert.cu) and multiplied by cuBLAS, instead of the
// whole prompt running token by token.  Checked, per type:
//   * the route: mmq::supported(t) is false and mmq::to_f16_supported(t) true (the prompt path's two tests);
//   * the FP16 operand, bit for bit: to_f16 of random blocks (finite scales, every code) == FP16(ggml's CPU
//     to_float) (round to nearest even) - the same dequantization the token path's tables and ggml's reference give;
//   * one expert's product: cuBLAS over the FP16 weights and FP16 activations, as the prompt path calls it
//     (CUBLAS_OP_T over the row-major matrix, F32 accumulation), against a double host product of the same FP16
//     values (rel L2 < 1e-4) - at 6144 -> 2 x 2048 (gate/up) and 2048 -> 6144 (down) shapes, 1..300 rows.
// The products inside the prompt path (their rows, the down's in-place order, the FP16 range guard) are checked end to
// end by glm_dsa_prefill_decode_test --iq1m against the token path.
// Needs a GPU; exits 77 (ctest SKIP_RETURN_CODE) when there is none.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// a finite half with a modest exponent (2^-10 .. 2^-3): the dequantized values stay normal FP16 numbers
uint16_t rand_scale(std::mt19937& rng) {
    const uint16_t exp = (uint16_t) (5 + rng() % 8);
    return (uint16_t) ((exp << 10) | (rng() & 0x3ffu));
}

// random blocks of type t with every scale field finite: IQ1_M keeps its FP16 scale in the top nibbles of its four
// 16-bit scale words (bytes 48..55); the legacy types keep d (and m) as FP16 at the block's start
std::vector<uint8_t> rand_blocks(ggml_type t, int64_t nb, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t);
    std::vector<uint8_t> b((size_t) nb * bs);
    for (auto& x : b) x = (uint8_t) rng();
    for (int64_t i = 0; i < nb; ++i) {
        uint8_t* blk = b.data() + (size_t) i * bs;
        if (t == GGML_TYPE_IQ1_M) {
            const uint16_t h = rand_scale(rng);
            uint16_t sc[4];
            std::memcpy(sc, blk + 48, 8);
            for (int k = 0; k < 4; ++k) sc[k] = (uint16_t) ((sc[k] & 0x0fff) | (((h >> (4 * k)) & 0xf) << 12));
            std::memcpy(blk + 48, sc, 8);
        } else {
            const uint16_t d = rand_scale(rng);
            std::memcpy(blk, &d, 2);
            if (t == GGML_TYPE_Q4_1 || t == GGML_TYPE_Q5_1) {
                const uint16_t m = (uint16_t) (rand_scale(rng) | 0x8000u);
                std::memcpy(blk + 2, &m, 2);
            }
        }
    }
    return b;
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "glmd_f16_experts_test: %s: %s\n", what, cudaGetErrorString(e));
    ++g_fail;
    return false;
}

float h2f(uint16_t h) { return ggml_fp16_to_fp32(h); }

// the FP16 operand of n values of type t: device to_f16 vs FP16(CPU to_float), bit for bit; returns the FP16 values
bool operand(ggml_type t, int64_t n, std::mt19937& rng, std::vector<uint16_t>& w16, cudaStream_t s) {
    const int64_t nb = n / ggml_blck_size(t);
    const std::vector<uint8_t> q = rand_blocks(t, nb, rng);
    std::vector<float> ref((size_t) n);
    ggml_get_type_traits(t)->to_float(q.data(), ref.data(), n);
    std::vector<uint16_t> want((size_t) n);
    for (int64_t i = 0; i < n; ++i) want[(size_t) i] = ggml_fp32_to_fp16(ref[(size_t) i]);
    void* dq = nullptr;
    uint16_t* d16 = nullptr;
    if (!ck(cudaMalloc(&dq, q.size()), "malloc") || !ck(cudaMalloc(&d16, (size_t) n * 2), "malloc")) return false;
    ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "upload");
    const bool ran = mmq::to_f16((int) t, dq, d16, n, s);
    w16.assign((size_t) n, 0);
    ck(cudaStreamSynchronize(s), "to_f16");
    ck(cudaMemcpy(w16.data(), d16, (size_t) n * 2, cudaMemcpyDeviceToHost), "download");
    (void) cudaFree(dq);
    (void) cudaFree(d16);
    int64_t bad = 0, nonfinite = 0, first = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isfinite(ref[(size_t) i])) ++nonfinite;
        if (w16[(size_t) i] != want[(size_t) i]) {
            if (first < 0) first = i;
            ++bad;
        }
    }
    std::string what = std::string(ggml_type_name(t)) + ": to_f16 of " + std::to_string(n) + " values == FP16(to_float)";
    if (bad) what += " (" + std::to_string(bad) + " differ, first at " + std::to_string(first) + ": " +
                     std::to_string(h2f(w16[(size_t) first])) + " vs " + std::to_string(h2f(want[(size_t) first])) + ")";
    check(ran && bad == 0 && nonfinite == 0, what);
    return ran && bad == 0;
}

// one expert matrix (out x in, row-major FP16 W) times rows x in FP16 activations, as the prompt path calls cuBLAS
void product(const char* name, ggml_type t, int out, int in, const std::vector<int>& row_counts, std::mt19937& rng,
             cublasHandle_t blas, cudaStream_t s) {
    std::vector<uint16_t> w16;
    if (!operand(t, (int64_t) out * in, rng, w16, s)) return;
    int max_rows = 0;
    for (int r : row_counts) max_rows = std::max(max_rows, r);
    std::vector<uint16_t> x16((size_t) max_rows * in);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x16) v = ggml_fp32_to_fp16(nd(rng));
    uint16_t *dw = nullptr, *dx = nullptr;
    float* dy = nullptr;
    if (!ck(cudaMalloc(&dw, w16.size() * 2), "malloc") || !ck(cudaMalloc(&dx, x16.size() * 2), "malloc") ||
        !ck(cudaMalloc(&dy, (size_t) max_rows * out * 4), "malloc"))
        return;
    ck(cudaMemcpy(dw, w16.data(), w16.size() * 2, cudaMemcpyHostToDevice), "upload w");
    ck(cudaMemcpy(dx, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice), "upload x");
    std::vector<float> wf(w16.size()), xf(x16.size());
    for (size_t i = 0; i < w16.size(); ++i) wf[i] = h2f(w16[i]);
    for (size_t i = 0; i < x16.size(); ++i) xf[i] = h2f(x16[i]);
    const float one = 1.0f, zero = 0.0f;
    double worst = 0;
    for (int n : row_counts) {
        const cublasStatus_t st = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, out, n, in, &one, dw, CUDA_R_16F, in, dx,
                                               CUDA_R_16F, in, &zero, dy, CUDA_R_32F, out, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            check(false, std::string(name) + ": cublasGemmEx status " + std::to_string((int) st));
            break;
        }
        std::vector<float> y((size_t) n * out);
        ck(cudaMemcpyAsync(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost, s), "download y");
        ck(cudaStreamSynchronize(s), "gemm");
        double e = 0, nn = 0;
        for (int r = 0; r < n; ++r)
            for (int o = 0; o < out; ++o) {
                double acc = 0;
                const float* wr = wf.data() + (size_t) o * in;
                const float* xr = xf.data() + (size_t) r * in;
                for (int k = 0; k < in; ++k) acc += (double) wr[k] * xr[k];
                const double d = (double) y[(size_t) r * out + o] - acc;
                e += d * d;
                nn += acc * acc;
            }
        worst = std::max(worst, std::sqrt(e / std::max(nn, 1e-300)));
    }
    char buf[160];
    // (F32 accumulation over 2048 / 6144 terms in cuBLAS's order: ~1e-5 on an RTX 3050)
    std::snprintf(buf, sizeof buf, "%s %s %d -> %d: cuBLAS FP16 product vs double, worst rel L2 %.2e (< 1e-4)", name,
                  ggml_type_name(t), in, out, worst);
    check(worst < 1e-4, buf);
    (void) cudaFree(dw);
    (void) cudaFree(dx);
    (void) cudaFree(dy);
}

}  // namespace

int main() {
    std::printf("glmd_f16_experts_test: the prompt path's FP16 expert products for types MMQ does not cover\n");
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glmd_f16_experts_test: no GPU - SKIP\n");
        return 77;
    }
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    if (!ck(cudaStreamCreate(&s), "stream") || cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "glmd_f16_experts_test: no stream / cuBLAS handle\n");
        return 1;
    }
    (void) cublasSetStream(blas, s);
    std::mt19937 rng(20261110);
    // the route the prompt path takes for these types (IQ1_M: the experts of the 170 GB GLM-5.3 quants)
    for (ggml_type t : {GGML_TYPE_IQ1_M, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1}) {
        check(!mmq::supported((int) t) && mmq::to_f16_supported((int) t),
              std::string(ggml_type_name(t)) + ": no MMQ instance, an FP16 dequantizer (the FP16 expert products)");
        std::vector<uint16_t> w16;
        operand(t, (int64_t) 64 * 4096, rng, w16, s);
    }
    // a covered type keeps MMQ (the products below never see it)
    check(mmq::supported((int) GGML_TYPE_IQ1_S) && mmq::supported((int) GGML_TYPE_IQ2_XXS),
          "IQ1_S / IQ2_XXS: MMQ instances (their layers unchanged)");
    // one expert's products at GLM-5.3's shapes: gate/up 6144 -> 2 x 2048, down 2048 -> 6144
    const std::vector<int> rows = {1, 7, 64, 300};
    product("gate/up", GGML_TYPE_IQ1_M, 2 * 2048, 6144, rows, rng, blas, s);
    product("down", GGML_TYPE_IQ1_M, 6144, 2048, rows, rng, blas, s);
    (void) cublasDestroy(blas);
    (void) cudaStreamDestroy(s);
    std::printf("glmd_f16_experts_test: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
