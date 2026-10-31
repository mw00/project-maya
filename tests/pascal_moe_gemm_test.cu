// tests/pascal_moe_gemm_test.cu - the Pascal prompt path's expert product (pascal::run: FP32 activations, weights
// dequantized into shared memory) against a double-precision product over ggml's own dequantizer, for the expert
// types of the GLM packs (IQ2_XXS, IQ2_S, IQ3_XXS), and its dense FP16 product (pascal::dense_run) against a double
// product of the same half inputs.  Synthetic weights from ggml's quantizers; no model needed.
//
//   pascal_moe_gemm_test           the parity checks (ctest)
//   pascal_moe_gemm_test --bench   Maya-S24's shapes: the time of pascal::run against ggml's MMQ on the same experts
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_pascal.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;
namespace pas = strata::prefill::pascal;

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "pascal_moe_gemm_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <class T> T* dev(size_t n) {
    void* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return (T*) p;
}

template <class T> T* upload(const std::vector<T>& h) {
    T* d = dev<T>(h.size());
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

// n_exp experts' [rows, cols] matrices, weights ~ N(0, 0.05) through ggml's quantizer (importance matrix of ones)
std::vector<uint8_t> make_experts(int type, int n_exp, int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.05f);
    const size_t eb = mmq::matrix_bytes(type, rows, cols);
    std::vector<uint8_t> host(eb * (size_t) n_exp);
    std::vector<float> f((size_t) rows * cols);
    const std::vector<float> imatrix((size_t) cols, 1.0f);
    for (int e = 0; e < n_exp; ++e) {
        for (auto& v : f) v = nd(rng);
        const size_t wrote = ggml_quantize_chunk((ggml_type) type, f.data(), host.data() + eb * (size_t) e, 0, rows, cols,
                                                 ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
        if (wrote != eb) {
            std::fprintf(stderr, "pascal_moe_gemm_test: %s: ggml wrote %zu bytes, expected %zu\n",
                         ggml_type_name((ggml_type) type), wrote, eb);
            std::exit(1);
        }
    }
    return host;
}

int parity(cudaStream_t s, bool f16) {
    const int w_rows = 256, w_cols = 512;
    std::mt19937 rng(20261016);
    // rows per expert: empty, one, short of a token tile, exactly one, one past, several tiles, and the edges of the
    // 8-token warp groups the FP16 kernel skips (8, 16, 17, 24, 31, 32, 33)
    const std::vector<int32_t> counts = {37, 0, 130, 1, 64, 9, 65, 200, 0, 5, 63, 128, 8, 16, 17, 24, 31, 32, 33};
    const int n_exp = (int) counts.size();
    std::vector<int32_t> hb(n_exp + 1, 0);
    for (int e = 0; e < n_exp; ++e) hb[e + 1] = hb[e] + counts[e];
    const int T = hb[n_exp], max_rows = *std::max_element(counts.begin(), counts.end());
    const int x_rows = T + 77;   // (a gather source larger than the set)
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t) x_rows * w_cols);
    for (auto& v : hx) v = nd(rng);
    std::vector<int32_t> hmap(T);
    for (int i = 0; i < T; ++i) hmap[i] = (int32_t) ((i * 7919L + 13) % x_rows);   // (7919 prime: a permutation-ish map)
    float* x = upload(hx);
    int32_t* map = upload(hmap);
    int32_t* bounds = upload(hb);
    float* dst = dev<float>((size_t) T * w_rows);

    int failures = 0;
    const int types[] = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS};
    for (int type : types) {
        const int distinct = 4;
        const auto some = make_experts(type, distinct, w_rows, w_cols, rng);
        const size_t eb = mmq::matrix_bytes(type, w_rows, w_cols);
        std::vector<uint8_t> hw(eb * (size_t) n_exp);
        for (int e = 0; e < n_exp; ++e) std::memcpy(hw.data() + eb * e, some.data() + eb * (e % distinct), eb);
        uint8_t* w = upload(hw);
        pas::Product p;
        p.w = w;
        p.type = type;
        p.w_rows = w_rows;
        p.w_cols = w_cols;
        p.expert_bytes = eb;
        p.n = n_exp;
        p.x = x;
        p.ld_x = w_cols;
        p.row_map = map;
        p.bounds = bounds;
        p.max_rows = max_rows;
        p.dst = dst;
        p.ld_dst = w_rows;
        p.f16 = f16;
        check(cudaMemsetAsync(dst, 0xff, (size_t) T * w_rows * 4, s), "memset");
        pas::run(p, s);
        std::vector<float> hd((size_t) T * w_rows);
        check(cudaMemcpyAsync(hd.data(), dst, hd.size() * 4, cudaMemcpyDeviceToHost, s), "download");
        check(cudaStreamSynchronize(s), "sync");

        // reference: ggml's dequantizer, double sums
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) type);
        std::vector<std::vector<float>> wf(distinct, std::vector<float>((size_t) w_rows * w_cols));
        for (int d = 0; d < distinct; ++d) tt->to_float(some.data() + eb * d, wf[d].data(), (int64_t) w_rows * w_cols);
        double worst = 0;
        long bad = 0;
        for (int e = 0; e < n_exp; ++e)
            for (int r = hb[e]; r < hb[e + 1]; ++r) {
                const float* xr = hx.data() + (size_t) hmap[r] * w_cols;
                double num = 0, den = 0;
                for (int m = 0; m < w_rows; ++m) {
                    const float* wr = wf[e % distinct].data() + (size_t) m * w_cols;
                    double ref = 0;
                    for (int k = 0; k < w_cols; ++k) ref += (double) wr[k] * xr[k];
                    const double got = hd[(size_t) r * w_rows + m];
                    num += (got - ref) * (got - ref);
                    den += ref * ref;
                    bad += !std::isfinite(got);
                }
                worst = std::max(worst, std::sqrt(num / std::max(den, 1e-30)));
            }
        const bool ok = bad == 0 && worst < (f16 ? 4e-3 : 2e-6);
        std::printf("  %-8s %s %d experts, %d rows: worst row rel L2 %.3g, %ld non-finite%s\n",
                    ggml_type_name((ggml_type) type), f16 ? "fp16" : "fp32", n_exp, T, worst, bad, ok ? "" : "  FAIL");
        failures += !ok;
        cudaFree(w);
    }
    cudaFree(dst);
    cudaFree(bounds);
    cudaFree(map);
    cudaFree(x);
    if (failures != 0) {
        std::printf("pascal_moe_gemm_test: %d FAILED\n", failures);
        return 1;
    }
    std::printf("pascal_moe_gemm_test: all pass\n");
    return 0;
}

// pascal::dense_run on tile edges: tokens short of / past a 64-token tile and of the 8-token warp groups, an N that
// is not a multiple of 128, a K that is not a multiple of 256 (the last flush), a long K.  `pieces` 2 takes the split
// path (K halves added into y: the output starts as garbage, and the pitch padding of x and y must stay untouched; a
// K of one 256 group stays whole), 0 lets the grid pick; `step` checks every step-th token (and the last) against the
// double product.
int dense_parity(cudaStream_t s) {
    struct Shape { int t, n, k, pieces, pad, step; };
    const Shape shapes[] = {{70, 256, 512, 1, 0, 1}, {65, 260, 256, 1, 0, 1}, {9, 128, 1536, 1, 0, 1}, {130, 512, 288, 1, 0, 1},
                            {1, 132, 64, 1, 0, 1}, {200, 384, 4096, 1, 0, 1},
                            {70, 256, 512, 2, 0, 1}, {65, 260, 256, 2, 0, 1}, {130, 512, 288, 2, 8, 1}, {9, 128, 1536, 2, 0, 1},
                            {70, 388, 4032, 2, 12, 1}, {64, 132, 1280, 2, 4, 1}, {37, 2052, 2336, 2, 0, 1}, {1, 132, 64, 2, 0, 1},
                            {256, 2048, 4096, 0, 8, 17}, {255, 2048, 4096, 0, 0, 17}};
    std::mt19937 rng(20261017);
    int failures = 0;
    double worst_all = 0;
    for (const Shape& sh : shapes) {
        std::normal_distribution<float> nw(0.f, 0.02f), nx(0.f, 1.f);
        const int ldx = sh.k + sh.pad, ldy = sh.n + sh.pad;
        std::vector<__half> hw((size_t) sh.n * sh.k), hx((size_t) sh.t * ldx);
        std::vector<float> fw(hw.size()), fx(hx.size());
        for (size_t i = 0; i < hw.size(); ++i) { hw[i] = __float2half(nw(rng)); fw[i] = __half2float(hw[i]); }
        for (size_t i = 0; i < hx.size(); ++i) { hx[i] = __float2half(nx(rng)); fx[i] = __half2float(hx[i]); }
        __half* w = upload(hw);
        __half* x = upload(hx);
        float* y = dev<float>((size_t) sh.t * ldy);
        check(cudaMemsetAsync(y, 0xff, (size_t) sh.t * ldy * 4, s), "memset");
        pas::Dense d;
        d.w = w;
        d.n = sh.n;
        d.k = sh.k;
        d.x = x;
        d.ld_x = ldx;
        d.t = sh.t;
        d.y = y;
        d.ld_y = ldy;
        d.pieces = sh.pieces;
        pas::dense_run(d, s);
        std::vector<float> hy((size_t) sh.t * ldy), hy2(hy.size());
        check(cudaMemcpyAsync(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost, s), "download");
        check(cudaMemsetAsync(y, 0x7f, hy.size() * 4, s), "memset");   // (another garbage)
        pas::dense_run(d, s);
        check(cudaMemcpyAsync(hy2.data(), y, hy2.size() * 4, cudaMemcpyDeviceToHost, s), "download");
        check(cudaStreamSynchronize(s), "sync");
        double worst = 0;
        long bad = 0;
        for (int t = 0; t < sh.t; ++t) {
            for (int n = sh.n; n < ldy; ++n) bad += !std::isnan(hy[(size_t) t * ldy + n]);   // (the pitch padding: still 0xff)
            const bool check_row = t % sh.step == 0 || t == sh.t - 1;
            double num = 0, den = 0;
            for (int n = 0; n < sh.n; ++n) {
                const double got = hy[(size_t) t * ldy + n];
                bad += !std::isfinite(got);
                if (!check_row) continue;
                double ref = 0;
                for (int k = 0; k < sh.k; ++k) ref += (double) fw[(size_t) n * sh.k + k] * fx[(size_t) t * ldx + k];
                num += (got - ref) * (got - ref);
                den += ref * ref;
            }
            if (check_row) worst = std::max(worst, std::sqrt(num / std::max(den, 1e-30)));
        }
        long same = 0;   // two runs over different garbage: the same bits
        for (int t = 0; t < sh.t; ++t)
            same += std::memcmp(&hy[(size_t) t * ldy], &hy2[(size_t) t * ldy], (size_t) sh.n * 4) != 0;
        const bool ok = bad == 0 && same == 0 && worst < 4e-3;
        worst_all = std::max(worst_all, worst);
        std::printf("  dense    fp16 t=%d n=%d k=%d pieces=%d pad=%d: worst row rel L2 %.3g, %ld bad, %ld rows differ run to run%s\n",
                    sh.t, sh.n, sh.k, sh.pieces, sh.pad, worst, bad, same, ok ? "" : "  FAIL");
        failures += !ok;
        cudaFree(y);
        cudaFree(x);
        cudaFree(w);
    }
    std::printf("  dense    fp16 worst over all shapes %.3g\n", worst_all);
    if (failures != 0) std::printf("pascal_moe_gemm_test: %d dense FAILED\n", failures);
    return failures != 0;
}

int bench(cudaStream_t s) {
    struct Shape { int type, w_rows, w_cols; const char* what; };
    const Shape shapes[] = {{GGML_TYPE_IQ2_XXS, 4096, 4096, "gate/up IQ2_XXS"}, {GGML_TYPE_IQ2_S, 4096, 2048, "down IQ2_S"},
                            {GGML_TYPE_IQ3_XXS, 4096, 2048, "down IQ3_XXS"}};
    const int n_exp = 100;
    std::mt19937 rng(5);
    // 100 experts of 33..160 rows (what the light kernels leave to the product)
    std::vector<int32_t> hb(n_exp + 1, 0);
    for (int e = 0; e < n_exp; ++e) hb[e + 1] = hb[e] + 33 + (int) (rng() % 128);
    const int T = hb[n_exp];
    int max_rows = 0;
    for (int e = 0; e < n_exp; ++e) max_rows = std::max(max_rows, hb[e + 1] - hb[e]);
    int32_t* bounds = upload(hb);
    int32_t* ids = dev<int32_t>(T);
    mmq::iota(ids, T, s);
    mmq::Context ctx;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const Shape& sh : shapes) {
        const size_t eb = mmq::matrix_bytes(sh.type, sh.w_rows, sh.w_cols);
        std::vector<uint8_t> hw(eb * n_exp);
        for (auto& b : hw) b = (uint8_t) rng();
        const size_t bsz = sh.type == GGML_TYPE_IQ2_XXS ? 66 : sh.type == GGML_TYPE_IQ2_S ? 82 : 98;
        for (size_t o = 0; o + bsz <= hw.size(); o += bsz) { hw[o] = 0x00; hw[o + 1] = 0x2c; }   // d = 0.0625 (half)
        uint8_t* w = upload(hw);
        std::vector<float> hx((size_t) T * sh.w_cols);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : hx) v = nd(rng);
        float* x = upload(hx);
        float* o1 = dev<float>((size_t) T * sh.w_rows);
        float* o2 = dev<float>((size_t) T * sh.w_rows);
        uint8_t* xq = dev<uint8_t>(mmq::q8_bytes(T, sh.w_cols));
        mmq::quantize(x, nullptr, xq, sh.type, sh.w_cols, sh.w_cols, T, s);
        mmq::Product mp;
        mp.w = w;
        mp.type = sh.type;
        mp.w_rows = sh.w_rows;
        mp.w_cols = sh.w_cols;
        mp.expert_bytes = eb;
        mp.n = n_exp;
        mp.xq = xq;
        mp.bounds = bounds;
        mp.ids = ids;
        mp.total_rows = T;
        mp.max_rows = max_rows;
        mp.dst = o1;
        mp.ld_dst = sh.w_rows;
        pas::Product pp;
        pp.w = w;
        pp.type = sh.type;
        pp.w_rows = sh.w_rows;
        pp.w_cols = sh.w_cols;
        pp.expert_bytes = eb;
        pp.n = n_exp;
        pp.x = x;
        pp.ld_x = sh.w_cols;
        pp.bounds = bounds;
        pp.max_rows = max_rows;
        pp.dst = o2;
        pp.ld_dst = sh.w_rows;
        float* o3 = nullptr;
        check(cudaMalloc(&o3, (size_t) T * sh.w_rows * 4), "o3");
        pas::Product ph = pp;
        ph.f16 = true;
        ph.dst = o3;
        float best_m = 1e9f, best_p = 1e9f, best_h = 1e9f;
        for (int rep = 0; rep < 6; ++rep) {
            float ms = 0;
            cudaEventRecord(e0, s);
            ctx.run(mp, s);
            cudaEventRecord(e1, s);
            check(cudaEventSynchronize(e1), "mmq");
            cudaEventElapsedTime(&ms, e0, e1);
            if (rep > 0) best_m = std::min(best_m, ms);
            cudaEventRecord(e0, s);
            pas::run(pp, s);
            cudaEventRecord(e1, s);
            check(cudaEventSynchronize(e1), "pascal");
            cudaEventElapsedTime(&ms, e0, e1);
            if (rep > 0) best_p = std::min(best_p, ms);
            cudaEventRecord(e0, s);
            pas::run(ph, s);
            cudaEventRecord(e1, s);
            check(cudaEventSynchronize(e1), "pascal16");
            cudaEventElapsedTime(&ms, e0, e1);
            if (rep > 0) best_h = std::min(best_h, ms);
        }
        std::vector<float> h3((size_t) T * sh.w_rows);
        check(cudaMemcpy(h3.data(), o3, h3.size() * 4, cudaMemcpyDeviceToHost), "d3");
        std::vector<float> h1((size_t) T * sh.w_rows), h2(h1.size());
        check(cudaMemcpy(h1.data(), o1, h1.size() * 4, cudaMemcpyDeviceToHost), "d1");
        check(cudaMemcpy(h2.data(), o2, h2.size() * 4, cudaMemcpyDeviceToHost), "d2");
        double num = 0, den = 0;
        for (size_t i = 0; i < h1.size(); ++i) {
            num += (double) (h1[i] - h2[i]) * (h1[i] - h2[i]);
            den += (double) h1[i] * h1[i];
        }
        double num3 = 0, den3 = 0, num4 = 0;
        for (size_t i = 0; i < h3.size(); ++i) {
            num3 += (double) (h2[i] - h3[i]) * (h2[i] - h3[i]);
            den3 += (double) h2[i] * h2[i];
            num4 += (double) (h1[i] - h3[i]) * (h1[i] - h3[i]);
        }
        const double macs = (double) T * sh.w_rows * sh.w_cols;
        std::printf("%-18s fp16: %7.2f ms (%.2f TMAC/s)  x%.2f over fp32  rel diff vs fp32 %.3g, vs MMQ %.3g\n", sh.what,
                    best_h, macs / best_h / 1e9, best_p / best_h, std::sqrt(num3 / std::max(den3, 1e-30)),
                    std::sqrt(num4 / std::max(den, 1e-30)));
        std::printf("%-18s %d experts %d rows: MMQ %7.2f ms (%.2f TMAC/s)  pascal %7.2f ms (%.2f TMAC/s)  x%.2f  rel diff %.3g\n",
                    sh.what, n_exp, T, best_m, macs / best_m / 1e9, best_p, macs / best_p / 1e9, best_m / best_p,
                    std::sqrt(num / std::max(den, 1e-30)));
        cudaFree(xq);
        cudaFree(o3);
        cudaFree(o2);
        cudaFree(o1);
        cudaFree(x);
        cudaFree(w);
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "device");
    std::printf("pascal_moe_gemm_test: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench(s);
    return parity(s, false) | parity(s, true) | dense_parity(s);
}
