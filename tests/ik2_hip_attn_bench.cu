// tests/ik2_hip_attn_bench.cu - the decode attention at Inkling 2 Small's shapes (H 48, Hkv 8, D 128, rel 16 over
// 1024 positions, sliding window 1024 or global): attention_split_rows (16-key tiles) against attention_chunk_rows
// (chunks), on random F16 caches. Prints each form's time a call and a pass (40 window + 8 global layers), the largest
// difference between them, and both against a double-precision reference on sampled rows. No model needed.
//   ik2_hip_attn_bench [reps]
#include "ik2_hip_kernels.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

namespace k = ik2hip::k;
constexpr int H = 48, Hkv = 8, D = 128, NREL = 16, EXTENT = 1024, NSWA = 1024;

struct Case { int T; long long p0; bool swa; };

static double ref_row(const std::vector<float>& q, const std::vector<float>& rel, const std::vector<float>& sink,
                      const std::vector<float>& rb, const std::vector<uint16_t>& Kh, const std::vector<uint16_t>& Vh,
                      const std::vector<float>& gate, int ring, long long p0, int t, int h, int dd, bool swa, float scale) {
    const int g = h / (H / Hkv);
    const long long pf = p0 + t, lo = swa ? std::max<long long>(0, pf - (NSWA - 1)) : 0;
    std::vector<double> a; std::vector<long long> pos;
    for (long long p = lo; p <= pf; ++p) {
        const long long sl = ring > 0 ? p % ring : p;
        double acc = 0;
        for (int i = 0; i < D; ++i) acc += (double) q[((size_t) t * H + h) * D + i] * (double) __half2float(*(const __half*) &Kh[((size_t) sl * Hkv + g) * D + i]);
        double v = acc * scale;
        const long long d = pf - p;
        if (d < EXTENT) for (int i = 0; i < NREL; ++i) v += (double) rel[((size_t) t * H + h) * NREL + i] * rb[(size_t) d * NREL + i];
        v -= sink[(size_t) t * H + h];
        a.push_back(v); pos.push_back(sl);
    }
    double m = 0; for (double v : a) m = std::max(m, v);
    double num = 0, den = std::exp(-m);
    for (size_t j = 0; j < a.size(); ++j) {
        const double w = std::exp(a[j] - m);
        den += w;
        num += w * (double) __half2float(*(const __half*) &Vh[((size_t) pos[j] * Hkv + g) * D + dd]);
    }
    return num / den / (1.0 + std::exp(-(double) gate[((size_t) t * H + h) * D + dd]));
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int Tmax = 9;
    const long long cap = 70000;   // the global cache's rows
    const int ring_swa = NSWA + 32;
    std::vector<float> q((size_t) Tmax * H * D), rel((size_t) Tmax * H * NREL), sink((size_t) Tmax * H), gate((size_t) Tmax * H * D),
        rb((size_t) EXTENT * NREL);
    for (auto& v : q) v = nd(rng) * 0.25f;   // (logits of a few units: clipped q and k in the model)
    for (auto& v : rel) v = nd(rng) * 0.5f;
    for (auto& v : sink) v = nd(rng);
    for (auto& v : gate) v = nd(rng);
    for (auto& v : rb) v = nd(rng) * 0.2f;
    std::vector<uint16_t> Kg((size_t) cap * Hkv * D), Vg((size_t) cap * Hkv * D);
    for (size_t i = 0; i < Kg.size(); ++i) {
        __half a = __float2half(nd(rng) * 0.5f), b = __float2half(nd(rng));
        Kg[i] = *(uint16_t*) &a; Vg[i] = *(uint16_t*) &b;
    }
    float *dq, *drel, *dsink, *dgate, *drb, *dS, *dout1, *dout2; uint16_t *dK, *dV;
    const size_t S_floats = (size_t) Tmax * H * ((cap + 15) / 16 + 2) * (D + 2);
    CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&drel, rel.size() * 4)); CK(cudaMalloc(&dsink, sink.size() * 4));
    CK(cudaMalloc(&dgate, gate.size() * 4)); CK(cudaMalloc(&drb, rb.size() * 4)); CK(cudaMalloc(&dS, S_floats * 4));
    CK(cudaMalloc(&dout1, q.size() * 4)); CK(cudaMalloc(&dout2, q.size() * 4));
    CK(cudaMalloc(&dK, Kg.size() * 2)); CK(cudaMalloc(&dV, Vg.size() * 2));
    CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(drel, rel.data(), rel.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsink, sink.data(), sink.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dgate, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(drb, rb.data(), rb.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, Kg.data(), Kg.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dV, Vg.data(), Vg.size() * 2, cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const float scale = 1.0f / std::sqrt((float) D);

    auto run = [&](int form, const Case& c, float* out) {
        const int ring = c.swa ? ring_swa : 0;
        const long long lo = c.swa ? std::max<long long>(0, c.p0 - (NSWA - 1)) : 0;
        bool ok = form == 0 ? k::attention_split_rows(dq, drel, dsink, c.T, H, Hkv, D, NREL, drb, EXTENT, dK, dV, ring, c.p0, lo,
                                                      c.swa ? NSWA : 0, scale, dS, S_floats, dgate, out, s)
                            : k::attention_chunk_rows(dq, drel, dsink, c.T, H, Hkv, D, NREL, drb, EXTENT, dK, dV, ring, c.p0, lo,
                                                      c.swa ? NSWA : 0, scale, dS, S_floats, dgate, out, s);
        if (!ok) { std::fprintf(stderr, "form %d refused T %d p0 %lld\n", form, c.T, c.p0); std::exit(3); }
    };
    auto time_us = [&](int form, const Case& c, float* out) {
        for (int i = 0; i < 20; ++i) run(form, c, out);
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < reps; ++i) run(form, c, out);
        CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        return 1e3 * ms / reps;
    };
    std::printf("decode attention, H %d Hkv %d D %d, chunk %d keys; us a call (one layer); %d reps\n", H, Hkv, D, k::attention_chunk_keys(), reps);
    std::printf("%-6s %3s %8s | %9s %9s | %8s | %11s %11s %11s\n", "layer", "T", "position", "tiles us", "chunks us", "speedup",
                "max |diff|", "tiles-ref", "chunks-ref");
    const long long ps[] = { 256, 1600, 4096, 16384, 65536 };
    double pass_old[5] = {0}, pass_new[5] = {0};
    int bad = 0;
    for (int T : { 1, 2, 4, 8 })
        for (int pi = 0; pi < 5; ++pi)
            for (bool swa : { true, false }) {
                const Case c{ T, ps[pi], swa };
                const double u0 = time_us(0, c, dout1), u1 = time_us(1, c, dout2);
                std::vector<float> o1((size_t) T * H * D), o2(o1.size());
                CK(cudaMemcpy(o1.data(), dout1, o1.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(o2.data(), dout2, o2.size() * 4, cudaMemcpyDeviceToHost));
                double md = 0, mx = 0;
                for (size_t i = 0; i < o1.size(); ++i) { md = std::max(md, (double) std::fabs(o1[i] - o2[i])); mx = std::max(mx, (double) std::fabs(o1[i])); }
                double r0 = 0, r1 = 0;   // (sampled rows against the reference: every row t, a few heads and dims)
                if (c.p0 <= 16384)
                    for (int t = 0; t < T; ++t)
                        for (int h : { 0, 7, 23, 47 })
                            for (int dd : { 0, 63, 127 }) {
                                const double ref = ref_row(q, rel, sink, rb, Kg, Vg, gate, swa ? ring_swa : 0, c.p0, t, h, dd, swa, scale);
                                const size_t i = ((size_t) t * H + h) * D + dd;
                                r0 = std::max(r0, std::fabs(o1[i] - ref)); r1 = std::max(r1, std::fabs(o2[i] - ref));
                            }
                if (md > 1e-4 * std::max(1.0, mx) || r1 > 1e-4) ++bad;
                std::printf("%-6s %3d %8lld | %9.2f %9.2f | %7.2fx | %11.3g %11.3g %11.3g\n", swa ? "window" : "global", T, c.p0, u0, u1,
                            u0 / u1, md, r0, r1);
                if (T == 1) { pass_old[pi] += (swa ? 40 : 8) * u0; pass_new[pi] += (swa ? 40 : 8) * u1; }
                std::fflush(stdout);
            }
    std::printf("\nT 1, a pass (40 window + 8 global layers), ms: position / tiles / chunks\n");
    for (int pi = 0; pi < 5; ++pi) std::printf("  %6lld  %7.2f  %7.2f\n", ps[pi], pass_old[pi] / 1e3, pass_new[pi] / 1e3);
    std::printf("%s\n", bad ? "MISMATCH" : "all within tolerance");
    return bad ? 1 : 0;
}
