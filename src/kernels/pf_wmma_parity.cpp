// src/kernels/pf_wmma_parity.cpp - gemm_iq_f16_grouped (the Volta prompt experts, STRATA_PF_WMMA) against a double
// reference over the same FP16 weights (iq_dequant_f16, the FP16 path's dequantization) and FP16 activations.
//
//     build/pf_wmma_parity            the checks (a ctest; needs a CUDA GPU with FP16 tensor cores: sm_70+)
//     build/pf_wmma_parity --bench    the benchmark only, one grouped launch per shape, timed (synthetic blobs)
//
// Every gate/up format the fork's native packs hand it (IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, and IQ2_XXS / IQ1_M through
// dq_dispatch: 1280 rows of 2560) and the down product (IQ4_NL and Q2_0: 2560 rows of 640).
// The wide group has 0, 1, 63, 64, 65, 127, 128, 129, 200, 255, 256, 257 and 300 rows per expert.
// A separate short group stops at 128 rows and exercises the short-row specialization of the gate/up formats most of
// the pack's experts use - IQ3_XXS and IQ2_S - as well as IQ3_S, and the IQ4_NL down product.
// Both groups start past a nonzero first bound and use input and output row strides wider than the matrices.
// Each type also uses a sparse `ids` gather and the by-value weight table the prompt path hands to the launch (`gu` in
// reverse expert order, `dn` in forward order, so a launch that reads the wrong half of the table mismatches
// instead of passing). Every output row outside the bounds and every column past n_out must remain at its sentinel.
// Results must be finite. The FP32 tensor-core sums use another order than the double reference: max |got - ref|
// <= 2e-5 * (max |ref| of the matrix) + 1e-6.
//
// --bench: the model's shapes (IQ3_XXS, IQ3_S and IQ2_S gate/up 1280 x 2560, IQ4_NL down 2560 x 640), 64 experts of 2,
// 64, 128 and 256 rows each - a short group, a moderate one, the short-row dispatch limit, and a wide group.
// Everything but the launch (the weights, the dequantization, the double reference) runs before the timed windows, and
// the launch is timed as a captured graph of 8 launches over 7 windows, the median per launch (native_grouped_parity's
// window convention: the host-side launch overhead is amortized, not measured).  Every shape is validated first -
// three rows of every expert against the double reference, every written output finite, every guard row and column at
// its sentinel - and a shape that fails is reported and not timed.  The bench assumes an idle GPU.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <vector>

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

const char* name_of(int t) {
    switch (t) {
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 29: return "IQ1_M";
        case 42: return "Q2_0";
        default: return "?";
    }
}
int block_values(int t) { return t == 20 || t == 7 || t == 8 ? 32 : t == 42 ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 29) {
            // IQ1_M: the fp16 scale is the top nibbles of its four scale words (bytes 48..55), the sign and the
            // exponent's top bits in the last: 0x1 / 0x2 there (0x9 / 0xA negative) keeps it in 2^-11 .. 2^-3
            const uint8_t nib = (uint8_t) ((sgn(rng) ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
        } else {   // the other formats start their block with the fp16 scale: 2^-13 .. 2^-6
            const int e = ex(rng);
            const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (e << 10) | man(rng));
            std::memcpy(&w[o], &h, 2);
        }
    }
    return w;
}

int run(int ty, int n_out, int K, std::mt19937& rng, bool use_ids, bool down, bool short_group = false) {
    static constexpr int short_rows[] = {0, 1, 63, 64, 65, 127, 128};
    static constexpr int wide_rows[] = {0, 1, 63, 64, 65, 127, 128, 129, 200, 255, 256, 257, 300};
    const auto rows = short_group ? std::span<const int>(short_rows) : std::span<const int>(wide_rows);
    const int E = (int) rows.size(), base = 5;
    std::vector<int32_t> bounds(E + 1);
    bounds[0] = base;
    for (int e = 0; e < E; ++e) bounds[e + 1] = bounds[e] + rows[e];
    const int total = bounds[E];
    const int ldx = K + 8, ldy = n_out + 8;      // row strides wider than the matrices (multiples of 8 / 4)
    const int xrows = total + 37;                // the ids gather may reach past the group's rows
    const int yrows = total + 16;
    const float SENT = -12345.5f;
    const size_t eb = k::iq_row_bytes(ty, K) * (size_t) n_out;
    std::vector<uint8_t> w;
    for (int e = 0; e < E; ++e) { auto r = random_rows(ty, n_out, K, rng); w.insert(w.end(), r.begin(), r.end()); }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> x((size_t) xrows * ldx);
    for (auto& v : x) v = __float2half(nd(rng));
    std::uniform_int_distribution<int> pick(0, xrows - 1);
    std::vector<int32_t> ids((size_t) total);
    for (int r = 0; r < total; ++r) ids[(size_t) r] = use_ids ? pick(rng) : r;
    std::vector<float> yh((size_t) yrows * ldy, SENT);
    uint8_t* dw; __half* dx; int32_t* db; int32_t* dids; float* dy; uint16_t* dq;
    ck(cudaMalloc(&dw, w.size() + 4096), "w");
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * 2), "x");
    ck(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&db, bounds.size() * 4), "b");
    ck(cudaMemcpy(db, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "b");
    ck(cudaMalloc(&dids, ids.size() * 4), "ids");
    ck(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMalloc(&dy, yh.size() * 4), "y");
    ck(cudaMemcpy(dy, yh.data(), yh.size() * 4, cudaMemcpyHostToDevice), "y");
    ck(cudaMalloc(&dq, (size_t) E * n_out * K * 2), "dq");
    const int maxr = *std::max_element(rows.begin(), rows.end());
    // The indirect pass: the by-value table the prompt path hands the grouped launches.  `gu` is the reverse
    // expert order (the existing check that the table, not the group's own order, selects the weights) and `dn`
    // the forward one, so a launch that reads the wrong half of the table mismatches instead of passing.
    k::GroupedWeightTable tbl{};
    tbl.n = E;
    for (int e = 0; e < E; ++e) { tbl.gu[e] = dw + (E - 1 - e) * eb; tbl.dn[e] = dw + e * eb; }
    if (!k::gemm_iq_f16_grouped(ty, use_ids ? nullptr : dw, eb, n_out, K, dx, ldx,
                                db, E, maxr, dy, ldy, nullptr, use_ids ? dids : nullptr,
                                use_ids ? &tbl : nullptr, down)) {
        std::printf("%-8s %5d x %4d: not launched\n", name_of(ty), n_out, K);
        return 1;
    }
    for (int e = 0; e < E; ++e) k::iq_dequant_f16(ty, dw + e * eb, (int64_t) n_out * K, dq + (size_t) e * n_out * K, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> y(yh.size());
    std::vector<uint16_t> q((size_t) E * n_out * K);
    ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "y");
    ck(cudaMemcpy(q.data(), dq, q.size() * 2, cudaMemcpyDeviceToHost), "dq");
    int bad = 0;   // rows outside the bounds and columns past n_out must be untouched
    for (int r = 0; r < yrows; ++r)
        for (int n = 0; n < ldy; ++n)
            if ((r < base || r >= total || n >= n_out) && y[(size_t) r * ldy + n] != SENT) ++bad;
    double worst = 0, scale = 0;
    for (int e = 0; e < E; ++e)
        for (int r = bounds[e]; r < bounds[e + 1]; ++r) {
            const __half* xr = &x[(size_t) (use_ids ? ids[(size_t) r] : r) * ldx];
            for (int n = 0; n < n_out; ++n) {
                double s = 0;
                const int we = use_ids ? (down ? e : E - 1 - e) : e;
                const uint16_t* wr = &q[((size_t) we * n_out + n) * K];
                for (int kk = 0; kk < K; ++kk) {
                    const __half h = __half_raw{wr[kk]};
                    s += (double) __half2float(h) * (double) __half2float(xr[kk]);
                }
                scale = std::max(scale, std::fabs(s));
                if (!std::isfinite(y[(size_t) r * ldy + n]) || !std::isfinite(s)) ++bad;
                worst = std::max(worst, std::fabs(s - (double) y[(size_t) r * ldy + n]));
            }
        }
    const bool ok = worst <= 2e-5 * scale + 1e-6 && bad == 0;
    std::printf("%-8s %5d x %4d, %d rows in %d experts%s: max |err| %.3g of max |ref| %.3g, %d stray %s\n", name_of(ty),
                n_out, K, total - base, E, use_ids ? (down ? " (ids, down)" : " (ids)") : "", worst, scale, bad, ok ? "ok" : "FAIL");
    cudaFree(dw); cudaFree(dx); cudaFree(db); cudaFree(dids); cudaFree(dy); cudaFree(dq);
    return ok ? 0 : 1;
}

// ------------------------------------------------------------------------------------------------------ --bench
// The model's expert shapes, a fixed group of experts each, timed with the launch alone inside the windows.
constexpr int kBenchExperts = 64;   // experts in the group (the launch's block count grows with it; the by-value
                                    // table is not used, the direct W + e * expert_bytes path is)
constexpr int kBenchRows[] = {2, 64, 128, 256};   // rows per expert: a decode-sized few, a moderate one, exactly
                                                  // one 128-row chunk (the shortest factor) and two of them
constexpr int kBenchRep = 8, kBenchWindows = 7;   // launches in the captured graph, timed windows per shape

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct BenchShape {
    int ty, n_out, K;
    const char* what;
};

// Validate one shape (its weights, activations and bounds) and time one launch of it per row count.  The weights,
// the dequantization and the double reference are outside the timed windows.  One expert's dequantized matrix is
// read back at a time, so the device allocation does not grow with the expert count.
int bench_shape(const BenchShape& c, std::mt19937& rng, cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1) {
    const int E = kBenchExperts;
    int maxr = 0;
    for (int r : kBenchRows) maxr = std::max(maxr, r);
    const size_t eb = k::iq_row_bytes(c.ty, c.K) * (size_t) c.n_out;
    const int ldx = c.K + 8, ldy = c.n_out + 8, yrows = E * maxr + 16;
    const float SENT = -12345.5f;
    std::vector<uint8_t> w;
    for (int e = 0; e < E; ++e) { auto r = random_rows(c.ty, c.n_out, c.K, rng); w.insert(w.end(), r.begin(), r.end()); }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> x((size_t) E * maxr * ldx);
    for (auto& v : x) v = __float2half(nd(rng));
    std::vector<int32_t> bounds(E + 1, 0);
    std::vector<float> yh((size_t) yrows * ldy, SENT);
    uint8_t* dw; __half* dx; int32_t* db; float* dy; uint16_t* dq;
    ck(cudaMalloc(&dw, w.size() + 4096), "w");
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * 2), "x");
    ck(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&db, bounds.size() * 4), "b");
    ck(cudaMalloc(&dy, yh.size() * 4), "y");
    ck(cudaMalloc(&dq, (size_t) c.n_out * c.K * 2), "dq");
    std::vector<float> y(yh.size());
    std::vector<uint16_t> q((size_t) c.n_out * c.K);
    int fails = 0;
    for (int rows : kBenchRows) {
        const int total = E * rows;
        for (int e = 0; e <= E; ++e) bounds[e] = e * rows;
        ck(cudaMemcpy(db, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "b");
        ck(cudaMemcpy(dy, yh.data(), yh.size() * 4, cudaMemcpyHostToDevice), "y");
        // the direct path: contiguous weights, no ids gather, no by-value table (down is then unused)
        if (!k::gemm_iq_f16_grouped(c.ty, dw, eb, c.n_out, c.K, dx, ldx, db, E, rows, dy, ldy, s, nullptr, nullptr,
                                    false)) {
            std::printf("%-15s %5d x %4d, %d rows: not launched\n", c.what, c.n_out, c.K, rows);
            ++fails;
            continue;
        }
        ck(cudaStreamSynchronize(s), "run");
        ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "y");
        int bad = 0;
        for (int r = 0; r < yrows; ++r)
            for (int n = 0; n < ldy; ++n) {
                const float v = y[(size_t) r * ldy + n];
                if (r >= total || n >= c.n_out) bad += v != SENT;
                else bad += !std::isfinite(v);
            }
        double worst = 0, scale = 0;   // three rows of every expert against the double reference
        for (int e = 0; e < E; ++e) {
            k::iq_dequant_f16(c.ty, dw + (size_t) e * eb, (int64_t) c.n_out * c.K, dq, nullptr);
            ck(cudaDeviceSynchronize(), "dq");
            ck(cudaMemcpy(q.data(), dq, q.size() * 2, cudaMemcpyDeviceToHost), "dq");
            for (int j = 0; j < 3; ++j) {
                const int r = bounds[e] + (j == 1 ? rows / 2 : j == 2 ? rows - 1 : 0);
                const __half* xr = &x[(size_t) r * ldx];
                for (int n = 0; n < c.n_out; ++n) {
                    double sum = 0;
                    const uint16_t* wr = &q[(size_t) n * c.K];
                    for (int kk = 0; kk < c.K; ++kk)
                        sum += (double) __half2float(__half_raw{wr[kk]}) * (double) __half2float(xr[kk]);
                    scale = std::max(scale, std::fabs(sum));
                    worst = std::max(worst, std::fabs(sum - (double) y[(size_t) r * ldy + n]));
                }
            }
        }
        if (!(bad == 0 && worst <= 2e-5 * scale + 1e-6)) {
            std::printf("%-15s %5d x %4d, %2d experts x %3d rows = %5d routed: max |err| %.3g of max |ref| %.3g, "
                        "%d stray: FAIL, not timed\n", c.what, c.n_out, c.K, E, rows, total, worst, scale, bad);
            ++fails;
            continue;
        }
        cudaGraph_t g;
        cudaGraphExec_t ge;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
        for (int i = 0; i < kBenchRep; ++i)
            k::gemm_iq_f16_grouped(c.ty, dw, eb, c.n_out, c.K, dx, ldx, db, E, rows, dy, ldy, s, nullptr, nullptr, false);
        ck(cudaStreamEndCapture(s, &g), "capture");
        ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
        ck(cudaGraphDestroy(g), "graph");
        for (int i = 0; i < 2; ++i) ck(cudaGraphLaunch(ge, s), "warm-up");
        ck(cudaStreamSynchronize(s), "warm-up");
        std::vector<float> ms;
        for (int i = 0; i < kBenchWindows; ++i) {
            ck(cudaEventRecord(e0, s), "record");
            ck(cudaGraphLaunch(ge, s), "launch");
            ck(cudaEventRecord(e1, s), "record");
            ck(cudaEventSynchronize(e1), "bench");
            float el = 0;
            ck(cudaEventElapsedTime(&el, e0, e1), "elapsed");
            ms.push_back(el / kBenchRep);
        }
        ck(cudaGraphExecDestroy(ge), "graph");
        std::printf("%-15s %5d x %4d, %2d experts x %3d rows = %5d routed: %8.3f ms/launch (median of %d x %d, "
                    "max |err| %.3g of max |ref| %.3g)\n", c.what, c.n_out, c.K, E, rows, total, median(ms),
                    kBenchWindows, kBenchRep, worst, scale);
    }
    cudaFree(dw); cudaFree(dx); cudaFree(db); cudaFree(dy); cudaFree(dq);
    return fails;
}

int bench(std::mt19937& rng) {
    std::printf("\n--bench: one grouped launch per shape, everything but the launch outside the timed windows (the\n"
                "weights, the dequantization and the double reference); the graph holds %d launches and the median of\n"
                "%d windows is reported per launch; an idle GPU is assumed\n", kBenchRep, kBenchWindows);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    const BenchShape shapes[] = {{18, 1280, 2560, "IQ3_XXS gate/up"}, {21, 1280, 2560, "IQ3_S gate/up"},
                                 {22, 1280, 2560, "IQ2_S gate/up"}, {20, 2560, 640, "IQ4_NL down"}};
    int fails = 0;
    for (const auto& c : shapes) fails += bench_shape(c, rng, s, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    ck(cudaStreamDestroy(s), "stream");
    return fails;
}
}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    if (argc > 2 || (argc == 2 && !do_bench)) {
        std::fprintf(stderr, "usage: pf_wmma_parity [--bench]\n");
        return 2;
    }
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 7) { std::printf("pf_wmma_parity: needs sm_70+, skipped\n"); return 77; }
    std::mt19937 rng(70);
    if (do_bench) {
        const int fails = bench(rng);
        std::printf("pf_wmma_parity --bench: %d failures\n", fails);
        return fails ? 1 : 0;
    }
    int fails = 0;
    for (int ty : {18, 21, 22, 23, 16, 29}) {
        fails += run(ty, 1280, 2560, rng, false, false);
        fails += run(ty, 1280, 2560, rng, true, false);
    }
    for (int ty : {20, 42}) {
        fails += run(ty, 2560, 640, rng, false, false);
        fails += run(ty, 2560, 640, rng, true, false);
        fails += run(ty, 2560, 640, rng, true, true);   // the prompt path's down launch: the table's dn half
    }
    // the gate/up formats most of the pack's experts use, over the same short-row bounds (direct and ids/table gu)
    for (int ty : {18, 22}) {
        fails += run(ty, 1280, 2560, rng, false, false, true);
        fails += run(ty, 1280, 2560, rng, true, false, true);
    }
    fails += run(21, 1280, 2560, rng, false, false, true);
    fails += run(21, 1280, 2560, rng, true, false, true);
    fails += run(20, 2560, 640, rng, false, false, true);
    fails += run(20, 2560, 640, rng, true, false, true);
    fails += run(20, 2560, 640, rng, true, true, true);
    std::printf("pf_wmma_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
