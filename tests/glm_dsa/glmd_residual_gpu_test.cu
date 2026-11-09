// tests/glm_dsa/glmd_residual_gpu_test.cu - glm-dsa B2.3 / B2.4 device checks (label gpu; exits 77 without a device):
//   * res_read (plain): R += block_out exactly, x = rms(R) * w within 1e-5 of a host double, xq = q8_1(x) within the
//     q8_1 bound, at n_embd 128, 4096 and 6144; res_close; res_embed (plain and mHC); res_head (plain); res_hop_bytes;
//   * the q8_1 -> f32 fallback dequantizer bit-equal to its host twin;
//   * glmdsa::mv with BF16 and F32 weights fed only a q8_1 input (the upcast set's call sites) against a host double of
//     the same dequantized input; the refusal without an ext; shexp_down_apart's two branches;
//   * B10.7: a Q4_1 shared down runs apart (bit for bit gf::mv), and gf::mv (2 columns), gf::mv_nt (2 activations)
//     and moe_gate_up's fused shared down refuse a Q4_1 weight instead of writing zeros.
#include "glm_dsa_fast.hpp"
#include "glm_residual.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::core;
namespace gf = strata::kernels::glmf;

static int g_fail = 0;
static int g_expected_launch_errors = 0;   // refusals a test provokes on purpose
#define CHECK(c)                                                                \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)

template<typename T>
static T* dev_copy(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template<typename T>
static std::vector<T> host_copy(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
static uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static float from_bf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static void test_residual(int n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> R((size_t) n), add((size_t) n), w((size_t) n);
    for (auto& v : R) v = nd(rng);
    for (auto& v : add) v = nd(rng) * 0.5f;
    for (auto& v : w) v = 1.0f + 0.1f * nd(rng);
    const float eps = 1e-5f;
    float* dR = dev_copy(R);
    float* dA = dev_copy(add);
    float* dW = dev_copy(w);
    float* dx = nullptr;
    void* dxq = nullptr;
    cudaMalloc(&dx, (size_t) n * 4);
    cudaMalloc(&dxq, glmdsa::q8_1_bytes(n));
    glmres::res_read(dR, dA, dW, eps, n, dx, dxq, nullptr);
    cudaDeviceSynchronize();
    const auto R1 = host_copy(dR, (size_t) n);
    const auto x = host_copy(dx, (size_t) n);
    const auto xq = host_copy((const uint8_t*) dxq, glmdsa::q8_1_bytes(n));
    double ss = 0.0;
    std::vector<float> Rs((size_t) n);
    for (int i = 0; i < n; ++i) {
        Rs[(size_t) i] = R[(size_t) i] + add[(size_t) i];
        ss += (double) Rs[(size_t) i] * Rs[(size_t) i];
    }
    const double inv = 1.0 / std::sqrt(ss / n + eps);
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        CHECK(R1[(size_t) i] == Rs[(size_t) i]);
        const double ref = Rs[(size_t) i] * inv * w[(size_t) i];
        worst = std::max(worst, std::fabs(x[(size_t) i] - ref) / std::max(1.0, std::fabs(ref)));
    }
    CHECK(worst < 1e-5);
    std::vector<float> xd((size_t) n);
    glmdsa::q8_1_to_f32_host(xq.data(), n, xd.data());
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(x[(size_t) (b * 32 + k)]));
        for (int k = 0; k < 32; ++k)
            CHECK(std::fabs(xd[(size_t) (b * 32 + k)] - x[(size_t) (b * 32 + k)]) <= amax / 127.0f * 0.63f + 1e-12f);
    }
    // the q8_1 fallback dequantizer: bit-equal to the host twin
    float* dy = nullptr;
    cudaMalloc(&dy, (size_t) n * 4);
    glmdsa::q8_1_to_f32(dxq, n, dy, nullptr);
    cudaDeviceSynchronize();
    CHECK(host_copy(dy, (size_t) n) == xd);

    // res_close: R += block_out
    glmres::res_close(dR, dA, n, nullptr);
    cudaDeviceSynchronize();
    const auto R2 = host_copy(dR, (size_t) n);
    for (int i = 0; i < n; ++i) CHECK(R2[(size_t) i] == Rs[(size_t) i] + add[(size_t) i]);

    // res_embed / res_head on a plain geometry
    Glm5Geometry g;
    g.hc = 1;
    g.n_embd = n;
    g.norm_eps = eps;
    CHECK(glmres::res_hop_bytes(g) == (size_t) n * 4);
    glmres::res_embed(g, dA, dR, nullptr);
    glmres::res_head(g, dR, dW, dx, dxq, nullptr);
    cudaDeviceSynchronize();
    CHECK(host_copy(dR, (size_t) n) == add);
    const auto xh = host_copy(dx, (size_t) n);
    ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double) add[(size_t) i] * add[(size_t) i];
    const double inv2 = 1.0 / std::sqrt(ss / n + eps);
    worst = 0.0;
    for (int i = 0; i < n; ++i) {
        const double ref = add[(size_t) i] * inv2 * w[(size_t) i];
        worst = std::max(worst, std::fabs(xh[(size_t) i] - ref) / std::max(1.0, std::fabs(ref)));
    }
    CHECK(worst < 1e-5);
    CHECK(host_copy(dA, (size_t) n) == add);   // the head never writes R / the input
    cudaFree(dR);
    cudaFree(dA);
    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dxq);
    cudaFree(dy);
    std::printf("res n_embd %d: x max rel err %.2e\n", n, worst);
}

static void test_embed_mhc() {
    Glm5Geometry g;   // Flash defaults: hc 4, n_embd 4096
    CHECK(g.mhc() && glmres::res_hop_bytes(g) == (size_t) 4 * 4096 * 4);
    std::vector<float> e(4096);
    for (int i = 0; i < 4096; ++i) e[(size_t) i] = (float) i * 0.25f - 7.0f;
    float* de = dev_copy(e);
    float* dR = nullptr;
    cudaMalloc(&dR, (size_t) 4 * 4096 * 4);
    glmres::res_embed(g, de, dR, nullptr);
    cudaDeviceSynchronize();
    const auto R = host_copy(dR, (size_t) 4 * 4096);
    for (int s = 0; s < 4; ++s)
        for (int i = 0; i < 4096; ++i) CHECK(R[(size_t) s * 4096 + i] == e[(size_t) i]);
    cudaFree(de);
    cudaFree(dR);
}

static void test_mv_fallback(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int n_in = 2048, n_out = 96;
    std::vector<float> xf((size_t) n_in), wf((size_t) n_in * n_out);
    for (auto& v : xf) v = nd(rng);
    for (auto& v : wf) v = nd(rng) * 0.05f;
    std::vector<uint8_t> xq(glmdsa::q8_1_bytes(n_in));
    glmdsa::q8_1_quantize_host(xf.data(), n_in, xq.data());
    std::vector<float> xd((size_t) n_in);
    glmdsa::q8_1_to_f32_host(xq.data(), n_in, xd.data());
    std::vector<uint16_t> wb(wf.size());
    for (size_t i = 0; i < wf.size(); ++i) wb[i] = to_bf16(wf[i]);
    uint8_t* dxq = dev_copy(xq);
    uint16_t* dwb = dev_copy(wb);
    float* dwf = dev_copy(wf);
    float *dy0 = nullptr, *dy1 = nullptr;
    cudaMalloc(&dy0, (size_t) n_out * 4);
    cudaMalloc(&dy1, (size_t) n_out * 4);

    glmdsa::GlmDsaFast ext;
    cudaGetDevice(&ext.dev);
    ext.plan.act_floats = 16384;
    cudaMalloc(&ext.arena, (size_t) glmdsa::kMaxMvJobs * ext.plan.act_floats * 4);
    ext.act = (float*) ext.arena;

    gf::MvJob j[2];
    j[0] = {dwb, dxq, nullptr, dy0, nullptr, 1.0f, gf::kTypeBF16, n_in, n_out};
    j[1] = {dwf, dxq, nullptr, dy1, nullptr, 1.0f, gf::kTypeF32, n_in, n_out};
    CHECK(!glmdsa::mv(nullptr, j, 2, nullptr));   // a fallback is needed and there is no scratch
    CHECK(glmdsa::mv(&ext, j, 2, nullptr));
    cudaDeviceSynchronize();
    const auto y0 = host_copy(dy0, (size_t) n_out), y1 = host_copy(dy1, (size_t) n_out);
    double worst = 0.0;
    for (int o = 0; o < n_out; ++o) {
        double r0 = 0.0, r1 = 0.0, mag = 0.0;
        for (int i = 0; i < n_in; ++i) {
            r0 += (double) from_bf16(wb[(size_t) o * n_in + i]) * xd[(size_t) i];
            r1 += (double) wf[(size_t) o * n_in + i] * xd[(size_t) i];
            mag += std::fabs((double) wf[(size_t) o * n_in + i] * xd[(size_t) i]);
        }
        worst = std::max({worst, std::fabs(y0[(size_t) o] - r0) / mag, std::fabs(y1[(size_t) o] - r1) / mag});
    }
    CHECK(worst < 1e-5);
    std::printf("glmdsa::mv fallback (BF16 + F32 on a q8_1 input): max err / sum|w x| %.2e\n", worst);

    // shexp_down_apart: BF16 down runs apart into sh_out; a native type is left to moe_gate_up
    WSlot sd;
    sd.type = gf::kTypeBF16;
    sd.q = dwb;
    bool apart = false;
    CHECK(glmdsa::shexp_down_apart(&ext, sd, dxq, n_in, n_out, dy1, nullptr, apart) && apart);
    cudaDeviceSynchronize();
    CHECK(host_copy(dy1, (size_t) n_out) == y0);
    sd.type = 14;
    CHECK(glmdsa::shexp_down_apart(&ext, sd, dxq, n_in, n_out, dy1, nullptr, apart) && !apart);

    // B10.7: a Q4_1 shared down (an STRATA_PACK_NATIVE32 pack) runs apart, bit for bit the plain one-column gf::mv
    // (ext_mv), also without an ext (glm5-next); the fused moe_gate_up dot would have read 0
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> w41((size_t) n_out * (n_in / 32) * 20);
    for (size_t b = 0; b < w41.size() / 20; ++b) {
        uint8_t* blk = w41.data() + b * 20;
        const uint16_t d = 0x2000 + (uint16_t) byte(rng), m = 0xa000 + (uint16_t) byte(rng);   // ~0.008, ~-0.008
        std::memcpy(blk, &d, 2);
        std::memcpy(blk + 2, &m, 2);
        for (int k = 4; k < 20; ++k) blk[k] = (uint8_t) byte(rng);
    }
    uint8_t* dw41 = dev_copy(w41);
    const gf::MvJob jq = {dw41, dxq, nullptr, dy0, nullptr, 1.0f, 3, n_in, n_out};
    CHECK(gf::mv(&jq, 1, nullptr));
    cudaDeviceSynchronize();
    const auto yq = host_copy(dy0, (size_t) n_out);
    float yq_abs = 0.0f;
    for (float v : yq) yq_abs += std::fabs(v);
    CHECK(yq_abs > 0.0f);
    sd.type = 3;
    sd.q = dw41;
    cudaMemset(dy1, 0, (size_t) n_out * 4);
    CHECK(glmdsa::shexp_down_apart(&ext, sd, dxq, n_in, n_out, dy1, nullptr, apart) && apart);
    cudaDeviceSynchronize();
    CHECK(host_copy(dy1, (size_t) n_out) == yq);
    cudaMemset(dy1, 0, (size_t) n_out * 4);
    CHECK(glmdsa::shexp_down_apart(nullptr, sd, dxq, n_in, n_out, dy1, nullptr, apart) && apart);
    cudaDeviceSynchronize();
    CHECK(host_copy(dy1, (size_t) n_out) == yq);
    // ... and the paths whose Maya kernels have no case for it refuse instead of writing zeros: several columns, several
    // activations (mv_nt), and the fused shared down (counted as a launch error)
    gf::MvJob jc = jq;
    jc.ncol = 2;
    jc.xq_cstride = (int64_t) glmdsa::q8_1_bytes(n_in);
    jc.y_cstride = n_out;
    CHECK(!gf::mv(&jc, 1, nullptr));
    CHECK(!gf::mv_nt(&jq, 1, 2, nullptr));
    const int err0 = gf::launch_errors();
    gf::MoeDev md;
    gf::moe_gate_up(12, md, 8, n_in, 256, 0.0f, dxq, nullptr, dw41, 3, dxq, n_in, dy1, nullptr);
    CHECK(gf::launch_errors() == err0 + 1);
    g_expected_launch_errors += 1;
    cudaFree(dw41);
    cudaFree(dxq);
    cudaFree(dwb);
    cudaFree(dwf);
    cudaFree(dy0);
    cudaFree(dy1);
}

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glmd_residual_gpu_test: no CUDA device, skipped\n");
        return 77;
    }
    std::mt19937 rng(53);
    for (int n : {128, 4096, 6144}) test_residual(n, rng);
    test_embed_mhc();
    test_mv_fallback(rng);
    CHECK(gf::launch_errors() == g_expected_launch_errors);
    CHECK(cudaGetLastError() == cudaSuccess);
    if (g_fail != 0) {
        std::fprintf(stderr, "glmd_residual_gpu_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("glmd_residual_gpu_test: OK\n");
    return 0;
}
