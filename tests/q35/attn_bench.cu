// q35_attn micro benchmark: a random K/V cache of N positions (FP16 and K8V4), T query tokens at the end,
// the time a call and the K/V bytes a second; and a check of the K8V4 output against the FP16 one.
// build: nvcc -arch=sm_70 -O3 -I include tests/q35/attn_bench.cu src/kernels/cuda/q35_kernels.cu src/kernels/cuda/dense_kernels.cu
#include "strata/kernels/q35_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#ifdef Q35_STREAM_PROBE
namespace strata::kernels { void q35_set_attn_stream_probe(int probe); void q35_stream_phases(unsigned long long* out, bool reset); }
#endif

using namespace strata::kernels;

// the GPU's streaming read rate over a buffer: every 16 bytes read once (uint4 loads, grid-stride), XOR-folded so the
// reads are not dropped - the ceiling the attention's K / V reads are measured against
__global__ void read_kernel(const uint4* __restrict__ a, size_t n, unsigned* out) {
    unsigned x = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = __ldg(a + i);
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (x == 0x12345678u) *out = x;
}
double read_gbs(const void* buf, size_t bytes, cudaStream_t s) {
    unsigned* o;
    cudaMalloc(&o, 4);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    double best = 0;
    for (int blocks : {sms * 4, sms * 8, sms * 16, sms * 32}) {
        read_kernel<<<blocks, 256, 0, s>>>((const uint4*) buf, bytes / 16, o);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a, s);
        for (int r = 0; r < 10; ++r) read_kernel<<<blocks, 256, 0, s>>>((const uint4*) buf, bytes / 16, o);
        cudaEventRecord(b, s); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        best = std::max(best, bytes / (ms / 10 * 1e-3) / 1e9);
    }
    cudaFree(o);
    return best;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 90000;
    const int H = 24, HK = 4, D = 256, max_ctx = N + 16;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    // K and V rows of the context, written through q35_attn_prep in chunks (both formats)
    Q35Kv kv[3];   // FP16, K8V4, K8V8
    for (int m = 0; m < 3; ++m) {
        kv[m].mode = m; kv[m].max_ctx = max_ctx;
        const size_t cells = (size_t) HK * max_ctx;
        cudaMalloc(&kv[m].k, m ? cells * D : cells * D * 2);
        cudaMalloc(&kv[m].v, m == 1 ? cells * D / 2 : m == 2 ? cells * D : cells * D * 2);
        if (m) { cudaMalloc(&kv[m].ks, cells * 8 * 2); cudaMalloc(&kv[m].vs, cells * 8 * 2); }
    }
    const int CH = 1024;
    std::vector<float> h((size_t) CH * H * 2 * D);
    float *qf, *kk, *vv, *qo, *nw, *out, *scr;
    cudaMalloc(&qf, h.size() * 4); cudaMalloc(&kk, (size_t) CH * HK * D * 4); cudaMalloc(&vv, (size_t) CH * HK * D * 4);
    cudaMalloc(&qo, (size_t) CH * H * D * 4); cudaMalloc(&nw, D * 4);
    std::vector<float> ones(D, 1.0f); cudaMemcpy(nw, ones.data(), D * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&out, (size_t) 2 * 8 * H * D * 4);
    cudaMalloc(&scr, q35_attn_scratch_floats(8, H, D, max_ctx) * 4);
    std::vector<float> kh((size_t) CH * HK * D);
    cudaStream_t s; cudaStreamCreate(&s);
    for (int p = 0; p < N; p += CH) {
        const int T = std::min(CH, N - p);
        for (auto& x : kh) x = nd(rng);
        cudaMemcpy(kk, kh.data(), kh.size() * 4, cudaMemcpyHostToDevice);
        for (auto& x : kh) x = nd(rng);
        cudaMemcpy(vv, kh.data(), kh.size() * 4, cudaMemcpyHostToDevice);
        for (int m = 0; m < 3; ++m)
            q35_attn_prep(qf, H * 2 * D, kk, vv, HK * D, nw, nw, 1e-6f, 1e7f, 64, p, T, H, HK, D, qo, kv[m], s);
    }
    for (auto& x : h) x = nd(rng) * 3.0f;   // the queries (sharper softmax)
    cudaMemcpy(qf, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    cudaStreamSynchronize(s);
    {
        const size_t kb = (size_t) HK * max_ctx * D * 2;
        printf("streaming read (the ceiling): K cache %.0f GB/s, V cache %.0f GB/s (%.0f MB each)\n", read_gbs(kv[0].k, kb, s),
               read_gbs(kv[0].v, kb, s), kb / 1e6);
    }
#ifdef Q35_STREAM_PROBE
    // `attn_bench N probe` (built with -DQ35_STREAM_PROBE): the stream kernel whole (0), its loads and stores alone (1),
    // its math alone over register-held tiles (2) - FP16 and K8V8
    if (argc > 2 && std::string(argv[2]) == "probe") {
        for (int m : {0, 2})
            for (int T : {1, 2, 4, 8})
                for (int sk : {32, 64}) {
                    printf("N %d %s T %d sk %d:", N, m ? "k8v8" : "fp16", T, sk);
                    for (int pr = 0; pr < 3; ++pr) {
                        q35_set_attn_stream_probe(pr);
                        q35_set_attn_stream(1, 0, sk);
                        const int p0 = N - T;
                        q35_attn(qo, kv[m], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
                        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                        cudaEventRecord(a, s);
                        for (int r = 0; r < 20; ++r) q35_attn(qo, kv[m], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
                        cudaEventRecord(b, s); cudaEventSynchronize(b);
                        float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 20;
                        const double gb = m == 0 ? (double) N * HK * D * 4 / 1e9 : (double) N * HK * (2 * D + 32) / 1e9;
                        printf("  %s %.3f ms (%.0f GB/s)", pr == 0 ? "whole" : pr == 1 ? "loads" : "math", ms, gb / ms * 1e3);
                    }
                    // the phases of one whole run (cycles summed over blocks, as a share)
                    q35_set_attn_stream_probe(0);
                    q35_attn(qo, kv[m], N - T, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
                    cudaStreamSynchronize(s);
                    unsigned long long ph[6];
                    q35_stream_phases(ph, true);
                    q35_attn(qo, kv[m], N - T, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
                    cudaStreamSynchronize(s);
                    q35_stream_phases(ph, true);
                    double tot = 0;
                    for (unsigned long long v : ph) tot += (double) v;
                    const char* nm[6] = {"Kst", "QK", "smax", "resc", "Vst", "PV"};
                    printf("  |");
                    for (int i = 0; i < 6; ++i) printf(" %s %.0f%%", nm[i], 100.0 * ph[i] / tot);
                    printf("  %s\n", cudaGetErrorString(cudaGetLastError()));
                }
        q35_set_attn_stream_probe(0);
        return 0;
    }
#endif
    // `attn_bench N tail`: a short prompt chunk (a continuation) over N cached keys - one key split (the prompt path's)
    // against the automatic choice (verify's), FP16; K8V8 direct against unpacked to FP16 first
    if (argc > 2 && std::string(argv[2]) == "tail") {
        float* pout;
        cudaMalloc(&pout, (size_t) CH * H * D * 4);
        uint16_t *k16, *v16;
        cudaMalloc(&k16, ((size_t) HK * max_ctx + 32) * D * 2);
        cudaMalloc(&v16, ((size_t) HK * max_ctx + 32) * D * 2);
        const Q35Kv kv16{k16, v16, nullptr, nullptr, 0, max_ctx};
        for (int T : {16, 32, 64, 128, 256, 512}) {
            const int p0 = N - T;
            auto t_of = [&](auto fn) {
                fn();
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventRecord(a, s);
                for (int r = 0; r < 10; ++r) fn();
                cudaEventRecord(b, s); cudaEventSynchronize(b);
                float ms = 0; cudaEventElapsedTime(&ms, a, b);
                return ms / 10;
            };
            std::vector<float> r1((size_t) T * H * D), r0(r1.size());
            const float m1 = t_of([&] { q35_attn(qo, kv[0], p0, T, H, HK, D, qf, H * 2 * D, pout, scr, 1, s); });
            cudaMemcpy(r1.data(), pout, r1.size() * 4, cudaMemcpyDeviceToHost);
            const float m0 = t_of([&] { q35_attn(qo, kv[0], p0, T, H, HK, D, qf, H * 2 * D, pout, scr, 0, s); });
            cudaMemcpy(r0.data(), pout, r0.size() * 4, cudaMemcpyDeviceToHost);
            double e = 0, r = 0;
            for (size_t i = 0; i < r1.size(); ++i) { e += pow(r0[i] - r1[i], 2); r += r1[i] * r1[i]; }
            const float k8d = t_of([&] { q35_attn(qo, kv[2], p0, T, H, HK, D, qf, H * 2 * D, pout, scr, 0, s); });
            const float k8x = t_of([&] {
                q35_kv_expand_f16(kv[2], p0 + T, HK, D, k16, v16, s);
                q35_attn(qo, kv16, p0, T, H, HK, D, qf, H * 2 * D, pout, scr, 0, s);
            });
            printf("tail T %3d at %d: fp16 1 split %.3f ms, auto splits %.3f ms (rel diff %.1e) | k8v8 direct %.3f ms, "
                   "unpacked first %.3f ms  %s\n", T, p0, m1, m0, sqrt(e / r), k8d, k8x, cudaGetErrorString(cudaGetLastError()));
        }
        return 0;
    }
    // `attn_bench N pp`: the prompt path - a 1024-row chunk (as the engine's, no splits) at a few depths, TFLOP/s of the
    // causal QK^T + PV work
    if (argc > 2 && std::string(argv[2]) == "pp") {
        float* pout;
        cudaMalloc(&pout, (size_t) CH * H * D * 4);
        uint16_t *k16, *v16;   // (K8V8's FP16 copy: m == 3 below)
        cudaMalloc(&k16, ((size_t) HK * max_ctx + 32) * D * 2);
        cudaMalloc(&v16, ((size_t) HK * max_ctx + 32) * D * 2);
        cudaMemset(k16, 0, ((size_t) HK * max_ctx + 32) * D * 2);
        cudaMemset(v16, 0, ((size_t) HK * max_ctx + 32) * D * 2);
        const Q35Kv kv16{k16, v16, nullptr, nullptr, 0, max_ctx};
        std::vector<float> ref8[3], got((size_t) CH * H * D);
        // the 8-warp prompt kernel against attn_tc_kernel (FP16): relative error and time
        for (int pi = 0; pi < 3; ++pi) {
            const int p0 = pi == 0 ? N / 4 : pi == 1 ? N / 2 : N - CH;
            std::vector<float> a((size_t) CH * H * D), b2(a.size());
            float tms[3];
            std::vector<float> c3(a.size());
            for (int k = 0; k < 3; ++k) {
                q35_set_attn_pp(k);
                cudaMemset(pout, 0xff, a.size() * 4);   // (NaN: an output not written shows)
                q35_attn(qo, kv[0], p0, CH, H, HK, D, qf, H * 2 * D, pout, scr, 1, s);
                cudaMemcpy(k == 2 ? c3.data() : k ? b2.data() : a.data(), pout, a.size() * 4, cudaMemcpyDeviceToHost);
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                cudaEventRecord(e0, s);
                for (int r = 0; r < 5; ++r) q35_attn(qo, kv[0], p0, CH, H, HK, D, qf, H * 2 * D, pout, scr, 1, s);
                cudaEventRecord(e1, s); cudaEventSynchronize(e1);
                cudaEventElapsedTime(&tms[k], e0, e1); tms[k] /= 5;
            }
            q35_set_attn_pp(-1);
            double e = 0, r = 0;
            size_t nan = 0, same = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::isnan(b2[i]) || std::isnan(a[i])) { ++nan; continue; }
                same += b2[i] == a[i];
                e += pow(b2[i] - a[i], 2); r += a[i] * a[i];
            }
            size_t same3 = 0;
            for (size_t i = 0; i < a.size(); ++i) same3 += c3[i] == a[i];
            printf("(%zu NaN, %zu / %zu of %zu equal; halves %.2f ms %.1f TFLOP/s) ", nan, same, same3, a.size(), tms[2],
                   4.0 * H * D * CH * (p0 + CH / 2.0) / tms[2] / 1e9);
            const double fl = 4.0 * H * D * CH * (p0 + CH / 2.0);
            printf("pp fp16 at %d: tc %.2f ms (%.1f TFLOP/s)  pp8 %.2f ms (%.1f TFLOP/s)  rel err %.2e  %s\n", p0, tms[0],
                   fl / tms[0] / 1e9, tms[1], fl / tms[1] / 1e9, sqrt(e / r), cudaGetErrorString(cudaGetLastError()));
        }
        for (int m : {0, 2, 3})   // (FP16, K8V8, K8V8 unpacked to FP16 first)
            for (int pi = 0; pi < 3; ++pi) {
                const int p0 = pi == 0 ? N / 4 : pi == 1 ? N / 2 : N - CH;
                auto run = [&] {
                    if (m == 3) {
                        q35_kv_expand_f16(kv[2], p0 + CH, HK, D, k16, v16, s);
                        q35_attn(qo, kv16, p0, CH, H, HK, D, qf, H * 2 * D, pout, scr, 1, s);
                    } else {
                        q35_attn(qo, kv[m], p0, CH, H, HK, D, qf, H * 2 * D, pout, scr, 1, s);
                    }
                };
                run();
                if (m >= 2) {   // (the copy's output against the K8V8 kernel's: the same bits expected)
                    cudaMemcpy(got.data(), pout, got.size() * 4, cudaMemcpyDeviceToHost);
                    if (m == 2) ref8[pi] = got;
                    if (m == 3) {
                        size_t diff = 0;
                        for (size_t i = 0; i < got.size(); ++i) diff += got[i] != ref8[pi][i];
                        printf("pp k8v8 FP16 copy at %d: %zu of %zu values differ from the K8V8 kernel's\n", p0, diff, got.size());
                    }
                }
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventRecord(a, s);
                for (int r = 0; r < 5; ++r) run();
                cudaEventRecord(b, s); cudaEventSynchronize(b);
                float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 5;
                const double fl = 4.0 * H * D * CH * (p0 + CH / 2.0);
                printf("pp %s chunk %d at %d: %.2f ms (%.1f TFLOP/s, %.3f ms a row)  %s\n", m == 3 ? "k8v8x" : m == 2 ? "k8v8" : m ? "k8v4" : "fp16", CH, p0, ms,
                       fl / ms / 1e9, ms / CH, cudaGetErrorString(cudaGetLastError()));
            }
        return 0;
    }
    // the streaming kernel (FP16 and K8V4): its tile choices, against the tile kernel's output in the same format
    for (int T : {1, 2, 4, 5, 8}) {
        const int p0 = N - T;
        for (int m : {0, 2}) {   // (FP16, K8V8)
            q35_set_attn_stream(0, 0, 0);
            q35_set_attn_dec_max_tokens(0);
            q35_set_attn_var(0);
            q35_attn_prep(qf, H * 2 * D, kk, vv, HK * D, nw, nw, 1e-6f, 1e7f, 64, p0, T, H, HK, D, qo, kv[m], s);
            q35_attn(qo, kv[m], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
            std::vector<float> ref((size_t) T * H * D), o2(ref.size());
            cudaMemcpy(ref.data(), out, ref.size() * 4, cudaMemcpyDeviceToHost);
            for (int cfg = 0; cfg < 3; ++cfg) {
                const int sk = cfg == 0 ? 0 : cfg == 1 ? 64 : 32;   // (0: the dispatch's choice)
                q35_set_attn_stream(1, 0, sk);
                cudaMemset(out, 0, ref.size() * 4);
                q35_attn(qo, kv[m], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
                cudaMemcpy(o2.data(), out, o2.size() * 4, cudaMemcpyDeviceToHost);
                double e = 0, r = 0;
                for (size_t i = 0; i < ref.size(); ++i) { e += pow(o2[i] - ref[i], 2); r += ref[i] * ref[i]; }
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventRecord(a, s);
                for (int rep = 0; rep < 20; ++rep) q35_attn(qo, kv[m], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
                cudaEventRecord(b, s); cudaEventSynchronize(b);
                float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 20;
                const double gb = m == 0 ? (double) N * HK * D * 2 * 2 / 1e9 : m == 2 ? (double) N * HK * (2 * D + 32) / 1e9
                                                                     : (double) N * HK * (D + D / 2 + 32) / 1e9;
                printf("N %d T %d stream %s sk %d: %.3f ms (%.0f GB/s)  vs tile rel err %.2e  %s\n", N, T, m == 2 ? "k8v8" : m ? "k8v4" : "fp16",
                       sk, ms, gb / ms * 1e3, sqrt(e / r), cudaGetErrorString(cudaGetLastError()));
            }
            q35_set_attn_stream(0, 0, 0);
        }
    }
    // the older kernels (stream off), FP16 and K8V8 (index 1 below)
    // the decode kernel's tuning: keys in flight a warp x target blocks, at T 1 and 2
    for (int T : {1}) {
        q35_set_attn_dec_max_tokens(8);
        q35_set_attn_var(0);
        const int p0 = N - T;
        for (int mb : {1})
        for (int pf : {2})
            for (int bl : {320}) {
                if (mb == 4 && pf == 8) continue;
                q35_set_attn_dec_tune(pf, bl, mb);
                float ms[2] = {0, 0};
                for (int m = 0; m < 2; ++m) {
                    q35_attn(qo, kv[m ? 2 : 0], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
                    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                    cudaEventRecord(a, s);
                    for (int r = 0; r < 20; ++r) q35_attn(qo, kv[m ? 2 : 0], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
                    cudaEventRecord(b, s); cudaEventSynchronize(b);
                    cudaEventElapsedTime(&ms[m], a, b); ms[m] /= 20;
                }
                const double gb16 = (double) N * HK * D * 2 * 2 / 1e9, gb8 = (double) N * HK * (2 * D + 32) / 1e9;
                printf("N %d T %d dec minb %d pf %d blocks %4d: fp16 %.3f ms (%.0f GB/s)  k8v8 %.3f ms (%.0f GB/s)  %s\n", N, T, mb, pf, bl,
                       ms[0], gb16 / ms[0] * 1e3, ms[1], gb8 / ms[1] * 1e3, cudaGetErrorString(cudaGetLastError()));
            }
        q35_set_attn_dec_tune(0, 0, 0);
    }
    for (int T : {1, 2, 4, 8}) {
        const int p0 = N - T;
        std::vector<float> ref;
        for (int dec : {0, 8, 101, 102}) {   // 0 the tile kernel, 8 the decode kernel, 10x tile variant x
            if (dec > 100 && T == 1) continue;
            q35_set_attn_dec_max_tokens(dec > 100 ? 0 : dec);
            q35_set_attn_var(dec > 100 ? dec - 100 : 0);
            float ms[2] = {0, 0};
            for (int m = 0; m < 2; ++m) {
                q35_attn_prep(qf, H * 2 * D, kk, vv, HK * D, nw, nw, 1e-6f, 1e7f, 64, p0, T, H, HK, D, qo, kv[m ? 2 : 0], s);
                q35_attn(qo, kv[m ? 2 : 0], p0, T, H, HK, D, qf, H * 2 * D, out + (size_t) m * 8 * H * D, scr, 0, s);
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventRecord(a, s);
                for (int r = 0; r < 20; ++r)
                    q35_attn(qo, kv[m ? 2 : 0], p0, T, H, HK, D, qf, H * 2 * D, out + (size_t) m * 8 * H * D, scr, 0, s);
                cudaEventRecord(b, s); cudaEventSynchronize(b);
                cudaEventElapsedTime(&ms[m], a, b); ms[m] /= 20;
            }
            std::vector<float> o((size_t) 2 * 8 * H * D);
            cudaMemcpy(o.data(), out, o.size() * 4, cudaMemcpyDeviceToHost);
            double e = 0, r = 0, e2 = 0;
            for (size_t i = 0; i < (size_t) T * H * D; ++i) { e += pow(o[i] - o[8 * H * D + i], 2); r += o[i] * o[i]; }
            if (dec == 0) ref = o;
            else for (size_t i = 0; i < (size_t) T * H * D; ++i) e2 += pow(o[i] - ref[i], 2) + pow(o[8 * H * D + i] - ref[8 * H * D + i], 2);
            const double gb16 = (double) N * HK * D * 2 * 2 / 1e9, gb8 = (double) N * HK * (2 * D + 32) / 1e9;
            printf("N %d T %d %s: fp16 %.3f ms (%.0f GB/s)  k8v8 %.3f ms (%.0f GB/s)  k8v8 rel err %.4f  dec vs tile %.2e  cuda %s\n",
                   N, T, dec == 8 ? "dec  " : dec > 100 ? (dec == 101 ? "var1 " : "var2 ") : "tile ", ms[0], gb16 / ms[0] * 1e3, ms[1], gb8 / ms[1] * 1e3, sqrt(e / r),
                   dec ? sqrt(e2 / (2 * r)) : 0.0, cudaGetErrorString(cudaGetLastError()));
        }
    }
    return 0;
}
