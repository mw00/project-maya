// tests/ds4_attn_mma_test.cu - sparse_attn_mma_kernel (Ampere tensor cores) against a double-precision host reference,
// next to the stock packed kernel on the same inputs: random FP4 compressed rows, random BF16 window rows, random
// queries, the sink.  Both kernels go through sparse_attn() - heads_per_block kAttnWmma picks the tensor-core kernel on
// sm_80+, 0 the packed one.  Also times both at a prompt chunk's shape.  Build (CUDA 13, sm_86):
//   nvcc -std=c++20 -O2 -arch=sm_86 -I src/ds4 tests/ds4_attn_mma_test.cu -o build/ds4_attn_mma_test
#include "../src/ds4/ds4_kernels.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace strata::ds4;

#define CK(x)                                                                                                   \
    do {                                                                                                        \
        cudaError_t e_ = (x);                                                                                   \
        if (e_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));                    \
            std::exit(2);                                                                                       \
        }                                                                                                       \
    } while (0)

static uint16_t h_f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static float h_bf2f(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// the compressed rows decoded to F32 by the engine's own unpack32 (the exact BF16 values both kernels see)
__global__ void decode_rows_kernel(const uint8_t* ckv, int n, float* out) {
    const int r = blockIdx.x, part = threadIdx.x;   // 16 threads: 32 values each
    if (part >= 16) return;
    __shared__ __align__(16) uint16_t row[512];
    unpack32(ckv + (size_t) r * kv_row_bytes(kKvFp4, 512), part, (uint4*) (row + part * 32));
    __syncthreads();
    for (int e = part; e < 512; e += 16) out[(size_t) r * 512 + e] = bf16f(row[e]);
}

struct Cfg {
    int T, win;
    long long p0;
    const char* name;
};

int main(int argc, char** argv) {
    const int heads = 64, d = 512, R = 4, kcap = 512, n_ckv = 4096;
    const float scale = 1.0f / std::sqrt((float) d);
    const bool time_only = argc > 1 && std::strcmp(argv[1], "--time") == 0;
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device %s sm_%d%d\n", p.name, p.major, p.minor);
    std::mt19937 rng(20261021);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> byte(0, 255), scode(8 * 1, 8 * 6 + 7), rowpick(0, n_ckv - 1);   // E4M3 exponents 1..6: 2^-6 .. ~2

    // compressed rows: random E2M1 codes (values 0 .. 6 times 1/14), E4M3 scales of exponent 1..6 (2^-6 .. ~2, times
    // 16384: the engine's scale) - rows of magnitude ~100 .. 10000, as the real ones; window rows BF16 of the same rms
    const size_t crow = kv_row_bytes(kKvFp4, d);
    std::vector<uint8_t> h_ckv((size_t) n_ckv * crow, 0);
    for (int r = 0; r < n_ckv; ++r) {
        uint8_t* row = h_ckv.data() + (size_t) r * crow;
        for (int i = 0; i < 256; ++i) row[i] = (uint8_t) byte(rng);
        for (int i = 0; i < 32; ++i) row[256 + i] = (uint8_t) scode(rng);
    }
    uint8_t* d_ckv;
    CK(cudaMalloc(&d_ckv, h_ckv.size()));
    CK(cudaMemcpy(d_ckv, h_ckv.data(), h_ckv.size(), cudaMemcpyHostToDevice));
    float* d_dec;
    CK(cudaMalloc(&d_dec, (size_t) n_ckv * d * sizeof(float)));
    decode_rows_kernel<<<n_ckv, 16>>>(d_ckv, n_ckv, d_dec);
    CK(cudaDeviceSynchronize());
    std::vector<float> h_dec((size_t) n_ckv * d);
    CK(cudaMemcpy(h_dec.data(), d_dec, h_dec.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double rms = 0;
    for (float v : h_dec) rms += (double) v * v;
    rms = std::sqrt(rms / h_dec.size());
    std::printf("compressed rows: %d, rms value %.1f\n", n_ckv, rms);
    if (!(rms > 1.0)) {
        std::fprintf(stderr, "the compressed rows decode to ~0: the test would not exercise them\n");
        return 2;
    }
    const float wmag = (float) rms;   // window rows of the same magnitude

    std::vector<Cfg> cfgs = {{1, 16, 2500, "T=1 win=16 deep (n_sel 512)"},
                             {8, 0, 0, "T=8 win=0 p0=0 (rows 0..2)"},
                             {7, 16, 50, "T=7 win=16 p0=50 (n_sel 12..14)"},
                             {16, 128, 100, "T=16 win=128 p0=100 (window clipped, n_sel 25..28)"},
                             {64, 128, 3000, "T=64 win=128 deep (n_sel 512)"},
                             {200, 16, 1, "T=200 win=16 p0=1 (n_sel 0..50)"}};
    if (time_only) cfgs = {{256, 128, 60000, "T=256 win=128 deep (timing)"}};
    // the query magnitude: scores of order a few units at the deep configs (512 rows ~ the model's)
    const float qmag = 3.0f / (scale * (float) rms * std::sqrt((float) d));
    bool ok = true;
    for (const Cfg& c : cfgs) {
        const int T = c.T;
        AttnRows ar{};
        ar.gen = 1;
        ar.t0 = 0;
        ar.win = c.win;
        ar.R = R;
        ar.kcap = kcap;
        ar.p0 = c.p0;
        const int nwin_rows = c.win + T;
        std::vector<uint16_t> h_win((size_t) std::max(nwin_rows, 1) * d);
        for (auto& v : h_win) v = h_f2bf(nd(rng) * wmag);
        std::vector<float> h_q((size_t) T * heads * d), h_sink(heads);
        for (auto& v : h_q) v = nd(rng) * qmag;
        for (auto& v : h_sink) v = nd(rng) * 2.0f;
        std::vector<int> h_sel((size_t) T * kcap, 0);
        for (auto& v : h_sel) v = rowpick(rng);
        uint16_t* d_win;
        float *d_q, *d_sink, *d_out0, *d_out1;
        int* d_sel;
        CK(cudaMalloc(&d_win, h_win.size() * 2));
        CK(cudaMalloc(&d_q, h_q.size() * 4));
        CK(cudaMalloc(&d_sink, heads * 4));
        CK(cudaMalloc(&d_sel, h_sel.size() * 4));
        CK(cudaMalloc(&d_out0, h_q.size() * 4));
        CK(cudaMalloc(&d_out1, h_q.size() * 4));
        CK(cudaMemcpy(d_win, h_win.data(), h_win.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_q, h_q.data(), h_q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sink, h_sink.data(), heads * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sel, h_sel.data(), h_sel.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_out0, 0xff, h_q.size() * 4));
        CK(cudaMemset(d_out1, 0xff, h_q.size() * 4));
        sparse_attn(d_q, d_win, d_ckv, d_sel, kcap, ar, d_sink, heads, d, T, scale, d_out0, 0, kKvFp4, 0, nullptr);
        CK(cudaGetLastError());
        sparse_attn(d_q, d_win, d_ckv, d_sel, kcap, ar, d_sink, heads, d, T, scale, d_out1, 0, kKvFp4, kAttnWmma, nullptr);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        if (time_only) {
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            for (int which = 0; which < 2; ++which) {
                const int hpb = which == 0 ? 0 : kAttnWmma;
                float* o = which == 0 ? d_out0 : d_out1;
                for (int i = 0; i < 3; ++i) sparse_attn(d_q, d_win, d_ckv, d_sel, kcap, ar, d_sink, heads, d, T, scale, o, 0, kKvFp4, hpb, nullptr);
                CK(cudaEventRecord(e0));
                const int iters = 20;
                for (int i = 0; i < iters; ++i) sparse_attn(d_q, d_win, d_ckv, d_sel, kcap, ar, d_sink, heads, d, T, scale, o, 0, kKvFp4, hpb, nullptr);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                std::printf("%s: %s kernel: %.3f ms per %d queries = %.1f us/query (%d rows a query)\n", c.name,
                            which == 0 ? "packed (stock)" : "mma (tensor cores)", ms / iters, T, ms / iters * 1000 / T,
                            c.win + kcap);
            }
            return 0;
        }
        std::vector<float> h_out0(h_q.size()), h_out1(h_q.size());
        CK(cudaMemcpy(h_out0.data(), d_out0, h_out0.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_out1.data(), d_out1, h_out1.size() * 4, cudaMemcpyDeviceToHost));
        // the host reference in double
        std::vector<double> ref(h_q.size());
        std::vector<double> sc;
        int nsel_min = 1 << 30, nsel_max = 0, nwin_min = 1 << 30, nwin_max = 0;
        for (int t = 0; t < T; ++t) {
            const long long ta = t, pa = c.p0 + ta;
            const long long first = std::max((long long) c.win - c.p0, ta + 1);
            const int w0 = (int) first, n_win = (int) (c.win + ta - first + 1);
            const int n_sel = R > 0 ? (int) std::min((long long) kcap, (pa + 1) / R) : 0;
            nsel_min = std::min(nsel_min, n_sel), nsel_max = std::max(nsel_max, n_sel);
            nwin_min = std::min(nwin_min, n_win), nwin_max = std::max(nwin_max, n_win);
            const int total = n_win + n_sel;
            for (int h = 0; h < heads; ++h) {
                const float* qh = h_q.data() + ((size_t) t * heads + h) * d;
                sc.assign(total, 0.0);
                double m = -1e300;
                for (int j = 0; j < total; ++j) {
                    double s = 0;
                    if (j < n_win) {
                        const uint16_t* row = h_win.data() + (size_t) (w0 + j) * d;
                        for (int e = 0; e < d; ++e) s += (double) qh[e] * h_bf2f(row[e]);
                    } else {
                        const float* row = h_dec.data() + (size_t) h_sel[(size_t) t * kcap + (j - n_win)] * d;
                        for (int e = 0; e < d; ++e) s += (double) qh[e] * row[e];
                    }
                    sc[j] = s * scale;
                    m = std::max(m, sc[j]);
                }
                m = std::max(m, (double) h_sink[h]);
                double den = std::exp((double) h_sink[h] - m);
                double* o = ref.data() + ((size_t) t * heads + h) * d;
                for (int e = 0; e < d; ++e) o[e] = 0;
                for (int j = 0; j < total; ++j) {
                    const double pj = std::exp(sc[j] - m);
                    den += pj;
                    if (j < n_win) {
                        const uint16_t* row = h_win.data() + (size_t) (w0 + j) * d;
                        for (int e = 0; e < d; ++e) o[e] += pj * h_bf2f(row[e]);
                    } else {
                        const float* row = h_dec.data() + (size_t) h_sel[(size_t) t * kcap + (j - n_win)] * d;
                        for (int e = 0; e < d; ++e) o[e] += pj * row[e];
                    }
                }
                for (int e = 0; e < d; ++e) o[e] /= den;
            }
        }
        double omax = 0;
        for (double v : ref) omax = std::max(omax, std::fabs(v));
        double e0 = 0, e1 = 0, e01 = 0;
        bool nan0 = false, nan1 = false;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!std::isfinite(h_out0[i])) nan0 = true;
            if (!std::isfinite(h_out1[i])) nan1 = true;
            e0 = std::max(e0, std::fabs(h_out0[i] - ref[i]));
            e1 = std::max(e1, std::fabs(h_out1[i] - ref[i]));
            e01 = std::max(e01, std::fabs((double) h_out1[i] - h_out0[i]));
        }
        const double r0 = e0 / omax, r1 = e1 / omax, r01 = e01 / omax;
        const bool pass = !nan0 && !nan1 && r1 <= 2e-3 && r0 <= 2e-3;
        ok = ok && pass;
        std::printf("%-52s n_win %d..%d n_sel %d..%d |ref|max %.3g  packed %.2e  mma %.2e  mma-packed %.2e  %s%s\n",
                    c.name, nwin_min, nwin_max, nsel_min, nsel_max, omax, r0, r1, r01, pass ? "ok" : "FAIL",
                    nan0 || nan1 ? " (NaN)" : "");
        CK(cudaFree(d_win));
        CK(cudaFree(d_q));
        CK(cudaFree(d_sink));
        CK(cudaFree(d_sel));
        CK(cudaFree(d_out0));
        CK(cudaFree(d_out1));
    }
    std::printf("%s\n", ok ? "ALL OK" : "FAILED");
    return ok ? 0 : 1;
}
