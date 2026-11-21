// tests/ik2_off_test.cu - maya-ik2's offload path (src/ik2/ik2_off.cuh, ik2_exact.h, ik2_lane.hpp):
//   * the exact lane: the GPU's off_gate_up_x / off_down_x against ik2_exact.h's reference and every CPU form this
//     machine runs (AVX-512, AVX2, scalar) - bit for bit, on random NVFP4 and Q8_0 experts at Mini's sizes;
//   * ik2_exact.h's exp / tanh / swiglu / FP16 rounding: the GPU's bits against the host's over a sweep;
//   * the GPU fetch: off_fetch_plan + off_fetch_copy fill staging with the host bytes, reuse a staged expert, and
//     never evict a slot the visit uses.
#include "ik2_exact.h"
#include "ik2_lane.hpp"
#include "ik2_off.cuh"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace k = ik2::k;
namespace x = ik2::x;

static int g_ok = 0, g_fail = 0;
static void check(bool c, const char* what) {
    if (c) {
        ++g_ok;
    } else {
        ++g_fail;
        std::printf("FAIL %s\n", what);
    }
}
#define CK(e)                                                                              \
    do {                                                                                   \
        cudaError_t r_ = (e);                                                              \
        if (r_ != cudaSuccess) {                                                           \
            std::printf("FAIL cuda %s: %s\n", #e, cudaGetErrorString(r_));                 \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

__global__ void math_kernel(const float* in, int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[5 * i + 0] = x::exp(in[i]);
    out[5 * i + 1] = x::tanh(in[i]);
    out[5 * i + 2] = x::sigmoid(in[i]);
    out[5 * i + 3] = x::round_f16(in[i] * 1000.0f);
    out[5 * i + 4] = x::swiglu(in[i] * 3.0f, in[(i * 7) % n] * 9.0f, 4.0f, 25.0f);
}

static std::vector<uint8_t> random_rows(int ty, int rows, int K, std::mt19937& g) {
    const size_t rb = x::row_bytes(ty, K);
    std::vector<uint8_t> b((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& v : b) v = (uint8_t) byte(g);
    for (int r = 0; r < rows; ++r) {
        uint8_t* row = b.data() + (size_t) r * rb;
        if (ty == x::kNVFP4) {
            for (int i = 0; i < K / 64; ++i)   // (scales: small E4M3 values, a zero now and then)
                for (int c = 0; c < 4; ++c) row[i * 36 + c] = (uint8_t) (byte(g) % 23 == 0 ? 0 : 0x20 + byte(g) % 40);
        } else {
            std::uniform_real_distribution<float> d(1e-4f, 4e-2f);
            for (int i = 0; i < K / 32; ++i) {
                const uint16_t h = x::f2h(d(g) * (byte(g) & 1 ? -1.0f : 1.0f));
                row[i * 34] = (uint8_t) (h & 0xFF), row[i * 34 + 1] = (uint8_t) (h >> 8);
            }
        }
    }
    return b;
}

// one expert format at C x F: K slots of random experts through the GPU and every CPU form
static void test_experts(int ty, int C, int F, int K, unsigned seed) {
    std::mt19937 g(seed);
    const size_t gb = x::row_bytes(ty, C) * F, db = x::row_bytes(ty, F) * C, eb = 2 * gb + db;
    std::vector<std::vector<uint8_t>> ex(K);
    for (int s = 0; s < K; ++s) {
        auto gu = random_rows(ty, 2 * F, C, g), dn = random_rows(ty, C, F, g);
        ex[s] = gu;
        ex[s].insert(ex[s].end(), dn.begin(), dn.end());
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> xv(C), sg(K), su(K);
    for (int i = 0; i < C; ++i) xv[i] = (i % 97 == 0) ? 0.0f : nd(g) * (i % 13 == 0 ? 8.0f : 1.0f);
    for (int s = 0; s < K; ++s) sg[s] = 0.5f + 0.1f * s, su[s] = 1.5f - 0.05f * s;
    const float gcap = 4.0f, ucap = 25.0f;
    // the reference
    std::vector<float> h_ref((size_t) K * F), y_ref((size_t) K * C);
    for (int s = 0; s < K; ++s) {
        const uint8_t* e = ex[s].data();
        for (int r = 0; r < F; ++r) {
            const float gd = x::row_dot(ty, e + r * x::row_bytes(ty, C), C, xv.data());
            const float ud = x::row_dot(ty, e + gb + r * x::row_bytes(ty, C), C, xv.data());
            h_ref[(size_t) s * F + r] = x::swiglu(x::mul(gd, sg[s]), x::mul(ud, su[s]), gcap, ucap);
        }
        for (int r = 0; r < C; ++r)
            y_ref[(size_t) s * C + r] = x::row_dot(ty, e + 2 * gb + r * x::row_bytes(ty, F), F, h_ref.data() + (size_t) s * F);
    }
    char what[160];
    // the CPU's forms
    for (const char* f : {"scalar", "avx2", "avx512"}) {
        if (!ik2::lane::rows_force(f)) {
            std::printf("  (%s: not on this CPU)\n", f);
            continue;
        }
        int bad = 0;
        std::vector<float> o(std::max(C, F));
        for (int s = 0; s < K && !bad; ++s) {
            ik2::lane::rows(ty, ex[s].data(), C, 0, F, xv.data(), o.data());
            for (int r = 0; r < F; ++r) {
                const float ref = x::row_dot(ty, ex[s].data() + r * x::row_bytes(ty, C), C, xv.data());
                bad += std::memcmp(&o[r], &ref, 4) != 0;
            }
            ik2::lane::rows(ty, ex[s].data() + 2 * gb, F, 0, C, h_ref.data() + (size_t) s * F, o.data());
            bad += std::memcmp(o.data(), y_ref.data() + (size_t) s * C, (size_t) C * 4) != 0;
        }
        std::snprintf(what, sizeof what, "CPU %s rows = the reference (ty %d, %d x %d)", f, ty, C, F);
        check(bad == 0, what);
    }
    ik2::lane::rows_force("avx512") || ik2::lane::rows_force("avx2");
    // the GPU
    std::vector<const uint8_t*> blob(K);
    uint8_t* dex = nullptr;
    CK(cudaMalloc(&dex, eb * K));
    for (int s = 0; s < K; ++s) {
        CK(cudaMemcpy(dex + eb * s, ex[s].data(), eb, cudaMemcpyHostToDevice));
        blob[s] = dex + eb * s;
    }
    const uint8_t** dblob;
    int* dcls;
    float *dx, *dsg, *dsu, *dh, *dy;
    CK(cudaMalloc(&dblob, K * sizeof(void*)));
    CK(cudaMalloc(&dcls, K * 4));
    CK(cudaMalloc(&dx, C * 4));
    CK(cudaMalloc(&dsg, K * 4));
    CK(cudaMalloc(&dsu, K * 4));
    CK(cudaMalloc(&dh, (size_t) K * F * 4));
    CK(cudaMalloc(&dy, (size_t) K * C * 4));
    std::vector<int> cls(K, 0);
    cls[K - 1] = 1;   // (a slot of the CPU's: left alone)
    CK(cudaMemcpy(dblob, blob.data(), K * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcls, cls.data(), K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, xv.data(), C * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsg, sg.data(), K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsu, su.data(), K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dh, 0x7f, (size_t) K * F * 4));
    CK(cudaMemset(dy, 0x7f, (size_t) K * C * 4));
    k::off_gate_up_x(dblob, dcls, K, ty, gb, dx, C, F, dsg, dsu, gcap, ucap, dh, nullptr);
    k::off_down_x(dblob, dcls, K, ty, 2 * gb, dh, C, F, dy, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> h((size_t) K * F), y((size_t) K * C);
    CK(cudaMemcpy(h.data(), dh, h.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    const size_t nh = (size_t) (K - 1) * F, ny = (size_t) (K - 1) * C;
    int dh_n = 0, dy_n = 0;
    for (size_t i = 0; i < nh; ++i) dh_n += std::memcmp(&h[i], &h_ref[i], 4) != 0;
    for (size_t i = 0; i < ny; ++i) dy_n += std::memcmp(&y[i], &y_ref[i], 4) != 0;
    std::snprintf(what, sizeof what, "GPU gate/up/SwiGLU = the reference bit for bit (ty %d: %d of %zu differ)", ty, dh_n, nh);
    check(dh_n == 0, what);
    std::snprintf(what, sizeof what, "GPU down = the reference bit for bit (ty %d: %d of %zu differ)", ty, dy_n, ny);
    check(dy_n == 0, what);
    uint32_t u;
    std::memcpy(&u, &y[ny], 4);
    check(u == 0x7f7f7f7fu, "GPU leaves the CPU's slot alone");
    cudaFree(dex), cudaFree(dblob), cudaFree(dcls), cudaFree(dx), cudaFree(dsg), cudaFree(dsu), cudaFree(dh), cudaFree(dy);
}

static void test_math() {
    std::vector<float> in;
    for (int i = -4000; i <= 4000; ++i) in.push_back(i * 0.0237f);
    for (float v : {0.0f, -0.0f, 1e-30f, -1e-30f, 1e-7f, 0.0624999f, 0.0625f, -0.0625f, 87.0f, -87.0f, 88.5f, -100.0f,
                    65519.0f / 1000.0f, 65520.0f / 1000.0f, 6.1e-8f, 3.0e-8f})
        in.push_back(v);
    const int n = (int) in.size();
    float *din, *dout;
    CK(cudaMalloc(&din, n * 4));
    CK(cudaMalloc(&dout, (size_t) n * 5 * 4));
    CK(cudaMemcpy(din, in.data(), n * 4, cudaMemcpyHostToDevice));
    math_kernel<<<(n + 127) / 128, 128>>>(din, n, dout);
    std::vector<float> out((size_t) n * 5);
    CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    double max_rel = 0;
    for (int i = 0; i < n; ++i) {
        const float h[5] = {x::exp(in[i]), x::tanh(in[i]), x::sigmoid(in[i]), x::round_f16(in[i] * 1000.0f),
                            x::swiglu(in[i] * 3.0f, in[(i * 7) % n] * 9.0f, 4.0f, 25.0f)};
        bad += std::memcmp(h, &out[(size_t) 5 * i], 20) != 0;
        if (in[i] > -80 && in[i] < 80) max_rel = std::max(max_rel, std::fabs(h[0] / std::exp((double) in[i]) - 1.0));
    }
    char what[128];
    std::snprintf(what, sizeof what, "exact math: GPU = host on %d inputs (%d differ)", n, bad);
    check(bad == 0, what);
    std::snprintf(what, sizeof what, "exact exp within 4e-7 of exp() (%.2e)", max_rel);
    check(max_rel < 4e-7, what);
    int tb = 0;
    for (float v = -12.0f; v < 12.0f; v += 0.01f) tb += std::fabs(x::tanh(v) - std::tanh((double) v)) > 3e-7;
    check(tb == 0, "exact tanh within 3e-7 of tanh()");
    int hb = 0;
    for (int i = 0; i < 65536; ++i) {   // every half: back and forth unchanged (NaNs: still NaNs)
        const float f = x::h2f((uint16_t) i);
        const uint16_t b = x::f2h(f);
        hb += (f == f) ? b != (uint16_t) i : !((b & 0x7c00) == 0x7c00 && (b & 0x3ff));
    }
    check(hb == 0, "FP16 round trip of every half");
    cudaFree(din), cudaFree(dout);
}

static void test_fetch() {
    const int E = 32, K = 8, NST = 12;
    const size_t sb = 4096;
    std::vector<uint8_t> host((size_t) E * sb);
    for (size_t i = 0; i < host.size(); ++i) host[i] = (uint8_t) (i * 131 + (i >> 12) * 7);
    uint8_t* hp = nullptr;
    CK(cudaHostAlloc(&hp, host.size(), cudaHostAllocMapped));
    std::memcpy(hp, host.data(), host.size());
    uint8_t* hdev = nullptr;
    CK(cudaHostGetDevicePointer((void**) &hdev, hp, 0));
    k::OffStage st;
    st.sb = sb, st.n = NST;
    CK(cudaMalloc(&st.data, NST * sb));
    CK(cudaMalloc(&st.tag, NST * 4));
    CK(cudaMalloc(&st.used, NST * 4));
    CK(cudaMalloc(&st.hand, 4));
    CK(cudaMemset(st.tag, 0xff, NST * 4));
    CK(cudaMemset(st.used, 0, NST * 4));
    CK(cudaMemset(st.hand, 0, 4));
    k::OffFetch* fl;
    k::OffCount* cnt;
    const uint8_t** blob;
    int* top;
    CK(cudaMalloc(&fl, sizeof(k::OffFetch)));
    CK(cudaMalloc(&cnt, sizeof(k::OffCount)));
    CK(cudaMemset(cnt, 0, sizeof(k::OffCount)));
    CK(cudaMalloc(&blob, K * sizeof(void*)));
    CK(cudaMalloc(&top, K * 4));
    uint8_t* tier = nullptr;
    CK(cudaMalloc(&tier, sb));
    int bad = 0;
    for (int v = 0; v < 6; ++v) {   // visits: picks overlapping the visit before
        int t[K];
        std::vector<const uint8_t*> b(K, nullptr);
        for (int i = 0; i < K; ++i) t[i] = (v * 2 + i) % E;
        b[0] = tier;   // (slot 0: "in the tier")
        CK(cudaMemcpy(top, t, K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(blob, b.data(), K * sizeof(void*), cudaMemcpyHostToDevice));
        k::off_fetch_plan(blob, top, K, 0, hdev, sb, st, fl, cnt, nullptr, nullptr);
        k::off_fetch_copy(fl, sb, 8, nullptr);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(b.data(), blob, K * sizeof(void*), cudaMemcpyDeviceToHost));
        std::vector<uint8_t> got(sb);
        for (int i = 1; i < K; ++i) {
            for (int j = 1; j < i; ++j) bad += b[i] == b[j];   // (two picks never share a slot)
            CK(cudaMemcpy(got.data(), b[i], sb, cudaMemcpyDeviceToHost));
            bad += std::memcmp(got.data(), host.data() + (size_t) t[i] * sb, sb) != 0;
        }
        bad += b[0] != tier;
    }
    k::OffCount c{};
    CK(cudaMemcpy(&c, cnt, sizeof c, cudaMemcpyDeviceToHost));
    check(bad == 0, "GPU fetch: staging holds each pick's host bytes, the tier's slot kept, no slot shared");
    check(c.picks == 48 && c.tier == 6 && c.stage > 0 && c.stage + c.fetched == 42, "GPU fetch: the counters");
    cudaFreeHost(hp), cudaFree(st.data), cudaFree(st.tag), cudaFree(st.used), cudaFree(st.hand), cudaFree(fl);
    cudaFree(cnt), cudaFree(blob), cudaFree(top), cudaFree(tier);
}

// `ik2_off_test bench [threads ...]`: the figures the offload path's ceiling rests on (no checks) - the CPU lane's rows
// per thread on rows in cache (compute) and streamed from memory (a 2 GiB buffer, every thread on its own experts),
// NVFP4 and Q8_0; the bus: cudaMemcpy from pinned memory and off_fetch_copy from mapped memory at a few grid sizes.
static double bench_rows(int ty, int K, const uint8_t* buf, size_t n_rows, int threads, bool hot, double secs) {
    const size_t rb = x::row_bytes(ty, K);
    const int chunk = 256;   // (rows a piece: hot - the same 256 rows again and again, ~330 KiB at 2304 NVFP4)
    std::vector<float> xv((size_t) K);
    for (int i = 0; i < K; ++i) xv[(size_t) i] = std::sin(0.37f * (float) i);
    std::atomic<bool> go{false}, stop{false};
    std::atomic<unsigned long long> rows_done{0};
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
        th.emplace_back([&, t] {
            std::vector<float> out(chunk);
            unsigned long long n = 0;
            size_t r = hot ? (size_t) t * chunk % (n_rows - chunk) : (size_t) t * (n_rows / threads);
            while (!go.load()) {}
            while (!stop.load(std::memory_order_relaxed)) {
                if (!hot && r + chunk > n_rows) r = 0;
                ik2::lane::rows(ty, buf + r * rb, K, 0, chunk, xv.data(), out.data());
                n += chunk;
                if (!hot) r += chunk;
            }
            rows_done.fetch_add(n);
        });
    const auto t0 = std::chrono::steady_clock::now();
    go.store(true);
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    stop.store(true);
    for (auto& t : th) t.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double) rows_done.load() * rb / s / 1e9;
}

static int bench(int argc, char** argv) {
    std::vector<int> tc;
    for (int i = 2; i < argc; ++i) tc.push_back(std::atoi(argv[i]));
    if (tc.empty()) tc = {1, (int) std::thread::hardware_concurrency() / 2};
    const size_t bytes = (size_t) 2 << 30;
    std::vector<uint8_t> buf(bytes);
    for (size_t i = 0; i < bytes; i += 4096) buf[i] = (uint8_t) i;   // (touched: on this process's node)
    {   // (scales and bytes: plausible values, 64 MiB of random bytes repeated)
        std::mt19937 g(7);
        auto blk = random_rows(x::kNVFP4, 64 * 1024 * 1024 / 1296, 2304, g);
        for (size_t o = 0; o < bytes; o += blk.size()) std::memcpy(buf.data() + o, blk.data(), std::min(blk.size(), bytes - o));
    }
    std::printf("bench: rows form %s\n", ik2::lane::rows_form());
    for (int ty : {x::kNVFP4, x::kQ8_0}) {
        const size_t rb = x::row_bytes(ty, 2304), n_rows = bytes / rb;
        for (int t : tc) {
            const double hot = bench_rows(ty, 2304, buf.data(), n_rows, t, true, 1.0);
            const double cold = bench_rows(ty, 2304, buf.data(), n_rows, t, false, 1.5);
            std::printf("bench: %s rows K 2304, %2d threads: in cache %6.1f GB/s (%.2f a thread), from memory %6.1f GB/s "
                        "(%.2f a thread)\n", ty == x::kNVFP4 ? "NVFP4" : "Q8_0", t, hot, hot / t, cold, cold / t);
        }
    }
    // the bus
    const size_t eb = 4478976, nb = 64;   // (an NVFP4 expert of Mini's: 4.48 MB; 64 of them)
    uint8_t* hp = nullptr;
    CK(cudaHostAlloc(&hp, eb * nb, cudaHostAllocMapped));
    std::memset(hp, 1, eb * nb);
    uint8_t *hd = nullptr, *dd = nullptr;
    CK(cudaHostGetDevicePointer((void**) &hd, hp, 0));
    CK(cudaMalloc(&dd, eb * 8));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    float ms = 0;
    for (int r = 0; r < 2; ++r) {
        CK(cudaEventRecord(e0));
        for (size_t j = 0; j < nb; ++j) CK(cudaMemcpyAsync(dd + (j % 8) * eb, hp + j * eb, eb, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1));
    }
    std::printf("bench: cudaMemcpy pinned -> VRAM, 4.48 MB copies: %.1f GB/s\n", eb * nb / (ms * 1e-3) / 1e9);
    k::OffFetch fh{};
    k::OffFetch* fl = nullptr;
    CK(cudaMalloc(&fl, sizeof fh));
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    for (int n : {1, 2, 8}) {
        for (int mul : {1, 2, 4, 8}) {
            const int blocks = std::max(8, mul * sms);
            double best = 0;
            for (int r = 0; r < 4; ++r) {
                fh.n = n;
                for (int i = 0; i < n; ++i) fh.src[i] = hd + (size_t) ((r * 8 + i) % nb) * eb, fh.dst[i] = dd + (size_t) i * eb;
                CK(cudaMemcpy(fl, &fh, sizeof fh, cudaMemcpyHostToDevice));
                CK(cudaEventRecord(e0));
                k::off_fetch_copy(fl, eb, blocks, nullptr);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms, e0, e1));
                best = std::max(best, n * eb / (ms * 1e-3) / 1e9);
            }
            std::printf("bench: off_fetch_copy, %d experts a launch, %4d blocks: %.1f GB/s (%.0f us an expert)\n", n,
                        blocks, best, eb / best / 1e3);
        }
    }
    cudaFreeHost(hp), cudaFree(dd), cudaFree(fl);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "bench")) return bench(argc, argv);
    std::printf("ik2_off_test: the CPU's rows here: %s\n", ik2::lane::rows_form());
    test_math();
    test_experts(x::kNVFP4, 2304, 1152, 8, 1);
    test_experts(x::kQ8_0, 2304, 1152, 8, 2);
    test_experts(x::kNVFP4, 256, 128, 3, 3);
    test_fetch();
    std::printf("ik2_off_test: %d ok, %d FAIL\n", g_ok, g_fail);
    return g_fail ? 1 : 0;
}
