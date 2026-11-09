// tests/glm_dsa/glmd_width_gpu_test.cpp - B2.1 / B2.2 / B2.5 on the device (label gpu): the norm kernels at glm-dsa's
// n_embd 6144 and at kMaxEmbd 8192 against an F64 reference (and glm5-next's 4096), the forced bad shapes that must
// count an error instead of returning silently, and a 6144-wide route into the host-mapped ring that must leave the
// next ring slot untouched (the canary: before B2.1 x[4096] spilled 8 KB into it).
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace gb = strata::kernels::glmb;

#define CHECK(call) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e_)); return 2; } } while (0)

static int g_fail = 0;
static void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "glmd_width_gpu_test: FAIL %s\n", what);
        ++g_fail;
    }
}

static float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, ex = (h >> 10) & 31, m = h & 1023;
    uint32_t bits;
    if (ex == 0) {
        if (m == 0) bits = s;
        else {
            float v = std::ldexp((float) m, -24);
            return s ? -v : v;
        }
    } else if (ex == 31) bits = s | 0x7f800000u | (m << 13);
    else bits = s | ((ex + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// FP16 round to nearest: |half(d) - d| <= 2^-11 d in FP16's normal range (the q8_1 scales here are ~0.02, > 2^-14)
constexpr double kHalfRound = 1.0 / 2048;

// rms_q8 (the decode pre-norm, with the residual add) and rms_rows / mtp_in at width n vs F64
static int norms_at(int n, cudaStream_t s) {
    std::mt19937 rng((uint32_t) n);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> h((size_t) n), add((size_t) n), w((size_t) n), e2((size_t) n);
    for (int i = 0; i < n; ++i) {
        h[(size_t) i] = nd(rng) * 3.0f;
        add[(size_t) i] = nd(rng);
        w[(size_t) i] = 0.5f + 0.01f * (float) (i % 97);
        e2[(size_t) i] = nd(rng) * 0.25f;
    }
    const float eps = 1e-5f;
    constexpr int T = 3;   // rms_rows rows
    float *dh, *dadd, *dw, *dx, *drow_in, *drow_x, *de2;
    void *dxq, *dx16, *dcat;
    CHECK(cudaMalloc(&dh, (size_t) n * 4));
    CHECK(cudaMalloc(&dadd, (size_t) n * 4));
    CHECK(cudaMalloc(&dw, (size_t) n * 4));
    CHECK(cudaMalloc(&dx, (size_t) n * 4));
    CHECK(cudaMalloc(&de2, (size_t) n * 4));
    CHECK(cudaMalloc(&dxq, (size_t) n / 32 * 36));
    CHECK(cudaMalloc(&dcat, (size_t) 2 * n / 32 * 36));
    CHECK(cudaMalloc(&drow_in, (size_t) T * n * 4));
    CHECK(cudaMalloc(&drow_x, (size_t) T * n * 4));
    CHECK(cudaMalloc(&dx16, (size_t) T * n * 2));
    CHECK(cudaMemcpy(dh, h.data(), (size_t) n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dadd, add.data(), (size_t) n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw, w.data(), (size_t) n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(de2, e2.data(), (size_t) n * 4, cudaMemcpyHostToDevice));
    std::vector<float> rows((size_t) T * n);
    for (auto& v : rows) v = nd(rng);
    CHECK(cudaMemcpy(drow_in, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
    const int e0f = gf::launch_errors(), e0b = gb::launch_errors();
    gf::rms_q8(dh, dadd, dw, eps, n, dx, dxq, s);
    gb::rms_rows(drow_in, dw, eps, T, n, drow_x, dx16, s);
    gf::mtp_in(de2, dh, dw, dw, eps, n, dcat, s);   // after rms_q8: h is the summed residual
    CHECK(cudaStreamSynchronize(s));
    expect(gf::launch_errors() == e0f && gb::launch_errors() == e0b, "no launch error at a valid width");
    std::vector<float> gh((size_t) n), gx((size_t) n), grx((size_t) T * n);
    std::vector<uint16_t> g16((size_t) T * n);
    std::vector<uint8_t> gq((size_t) n / 32 * 36), gcat((size_t) 2 * n / 32 * 36);
    CHECK(cudaMemcpy(gh.data(), dh, (size_t) n * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gx.data(), dx, (size_t) n * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gq.data(), dxq, gq.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gcat.data(), dcat, gcat.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(grx.data(), drow_x, grx.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(g16.data(), dx16, g16.size() * 2, cudaMemcpyDeviceToHost));
    // rms_q8: h = h + add exactly (one float add), x = rms(h) * w within 1e-6 relative of F64
    double ss = 0.0, worst = 0.0;
    bool h_ok = true;
    for (int i = 0; i < n; ++i) {
        const float hs = h[(size_t) i] + add[(size_t) i];
        h_ok &= gh[(size_t) i] == hs;
        ss += (double) hs * hs;
    }
    const double inv = 1.0 / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) {
        const double ref = (double) gh[(size_t) i] * inv * w[(size_t) i];
        worst = std::fmax(worst, std::fabs(gx[(size_t) i] - ref) / std::fmax(std::fabs(ref), 1e-30));
    }
    // its q8_1 (quantize.cu's arithmetic): d32 = max|x| / 127 per 32 in FP32, q = round(x / d32) with that d32, and
    // the block stores d16 = half(d32), within 2^-11 d32.  A reader's q d16 is then within d32 (0.5 + |q| 2^-11) of x,
    // half a step plus |q| times the scale's rounding: <= d16 (0.5 + |q| 2^-11) / (1 - 2^-11), 0.5623 d16 at |q| 127
    // (the 1.001 holds 1 / (1 - 2^-11) and the FP32 division's 127 * 2^-24).  A wrong scale fails qd_ok; a wrong
    // rounding fails q_ok (a truncation is off by up to a whole step).
    bool q_ok = true, qd_ok = true;
    double wq = 0.0;   // max |q d16 - x| / d16
    for (int b = 0; b < n / 32; ++b) {
        uint16_t dh16;
        std::memcpy(&dh16, gq.data() + (size_t) b * 36, 2);
        const float d = half_to_float(dh16);
        const int8_t* q = (const int8_t*) (gq.data() + (size_t) b * 36 + 4);
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(gx[(size_t) b * 32 + i]));
        const float d32 = amax / 127.0f;
        qd_ok &= std::fabs(d - d32) <= d32 * kHalfRound * 1.001;
        for (int i = 0; i < 32; ++i) {
            const double err = std::fabs(d * q[i] - gx[(size_t) b * 32 + i]);
            q_ok &= err <= d * (0.5 + std::abs((int) q[i]) * kHalfRound) * 1.001 + 1e-6;
            if (d > 0.0f) wq = std::fmax(wq, err / d);
        }
    }
    // rms_rows per row, F32 and FP16
    double worst_rows = 0.0;
    bool r16_ok = true;
    for (int t = 0; t < T; ++t) {
        double s2 = 0.0;
        for (int i = 0; i < n; ++i) s2 += (double) rows[(size_t) t * n + i] * rows[(size_t) t * n + i];
        const double iv = 1.0 / std::sqrt(s2 / n + eps);
        for (int i = 0; i < n; ++i) {
            const size_t k = (size_t) t * n + i;
            const double ref = rows[k] * iv * w[(size_t) i];
            worst_rows = std::fmax(worst_rows, std::fabs(grx[k] - ref) / std::fmax(std::fabs(ref), 1e-30));
            r16_ok &= std::fabs(half_to_float(g16[k]) - grx[k]) <= std::fabs(grx[k]) * 1e-3f + 1e-7f;
        }
    }
    // mtp_in: the two halves' q8_1 against the F64 values, the same bounds with d32 = max|ref| / 127.  The kernel's F32
    // halves are within ~1e-7 relative of F64 (rms_q8's x above is the same arithmetic); the bounds hold up to 4.8e-6
    // relative on the scale (its 1.01) and 2.1e-6 on the values (the 1.001's spare 2.8e-4 d16 over |ref| <= 127.5 d16),
    // plus the 1e-5
    double sa = 0.0, sb = 0.0;
    for (int i = 0; i < n; ++i) {
        sa += (double) e2[(size_t) i] * e2[(size_t) i];
        sb += (double) gh[(size_t) i] * gh[(size_t) i];
    }
    const double ia = 1.0 / std::sqrt(sa / n + eps), ib = 1.0 / std::sqrt(sb / n + eps);
    bool cat_ok = true, cd_ok = true;
    double wc = 0.0;   // max |q d16 - ref| / d16
    for (int b = 0; b < 2 * n / 32; ++b) {
        uint16_t dh16;
        std::memcpy(&dh16, gcat.data() + (size_t) b * 36, 2);
        const float d = half_to_float(dh16);
        const int8_t* q = (const int8_t*) (gcat.data() + (size_t) b * 36 + 4);
        double ref[32], amax = 0.0;
        for (int i = 0; i < 32; ++i) {
            const int e = b * 32 + i;
            ref[i] = e < n ? e2[(size_t) e] * ia * w[(size_t) e] : gh[(size_t) (e - n)] * ib * w[(size_t) (e - n)];
            amax = std::fmax(amax, std::fabs(ref[i]));
        }
        cd_ok &= std::fabs(d - amax / 127) <= amax / 127 * kHalfRound * 1.01;
        for (int i = 0; i < 32; ++i) {
            const double err = std::fabs(d * q[i] - ref[i]);
            cat_ok &= err <= d * (0.5 + std::abs((int) q[i]) * kHalfRound) * 1.001 + 1e-5;
            if (d > 0.0f) wc = std::fmax(wc, err / d);
        }
    }
    std::printf("glmd_width_gpu_test: n %d: rms_q8 max rel err %.2e, rms_rows %.2e; q8_1 max err %.3f d16 (rms_q8), "
                "%.3f (mtp_in), bound 0.563\n", n, worst, worst_rows, wq, wc);
    expect(h_ok, "rms_q8 residual add");
    expect(worst < 1e-6, "rms_q8 vs F64: max rel err < 1e-6");
    expect(qd_ok, "rms_q8 q8_1 scale = half(max|x| / 127)");
    expect(q_ok, "rms_q8 q8_1");
    expect(worst_rows < 1e-6, "rms_rows vs F64: max rel err < 1e-6");
    expect(r16_ok, "rms_rows FP16 copy");
    expect(cd_ok, "mtp_in q8_1 halves' scales");
    expect(cat_ok, "mtp_in q8_1 halves");
    for (void* p : {(void*) dh, (void*) dadd, (void*) dw, (void*) dx, (void*) de2, dxq, dcat, (void*) drow_in,
                    (void*) drow_x, dx16})
        cudaFree(p);
    return 0;
}

// bad shapes: refused before any launch, each counted in its family's launch_errors (the forward then fails)
static int bad_shapes(cudaStream_t s) {
    float* buf = nullptr;
    CHECK(cudaMalloc(&buf, (size_t) 4 * 9000 * 4));
    const int f0 = gf::launch_errors(), b0 = gb::launch_errors();
    gf::rms_q8(buf, nullptr, buf, 1e-5f, 8224, buf, buf, s);              // > kMaxEmbd
    gf::rms_q8(buf, nullptr, buf, 1e-5f, 6150, buf, buf, s);              // not a multiple of 32
    gf::mtp_in(buf, buf, buf, buf, 1e-5f, 8224, buf, s);
    gf::head_prep(buf, buf, 1e-5f, 6144, buf, buf, s);                    // the mHC head: 4096 at most
    expect(gf::launch_errors() == f0 + 4, "glmf: four refused shapes counted");
    gb::rms_rows(buf, buf, 1e-5f, 1, 6000, buf, buf, s);                  // not a multiple of 256
    gb::rms_rows(buf, buf, 1e-5f, 1, 8448, buf, buf, s);                  // > 32 x 256
    gb::mtp_in_rows(buf, buf, buf, buf, 1e-5f, 1, 8448, buf, s);
    gb::head_rows(buf, buf, 1e-5f, 1, 100, buf, s);
    gb::mla_attn(buf, nullptr, nullptr, nullptr, 0, 64, 256, 1.0f, 1, buf, s);   // kv_lora 256
    expect(gb::launch_errors() == b0 + 5, "glmb: five refused shapes counted");
    CHECK(cudaStreamSynchronize(s));
    cudaFree(buf);
    return 0;
}

// a 6144-wide CPU-lane route at E 256, k 8: request i carries x, request i + 1 stays the canary
static int ring_canary(cudaStream_t s) {
    constexpr int E = 256, N = 6144, K = 8;
    void* arena = nullptr;
    const size_t abytes = 1 << 20;
    CHECK(cudaMalloc(&arena, abytes));
    CHECK(cudaMemset(arena, 0, abytes));
    size_t off = 0;
    auto take = [&](size_t bytes) {
        off = (off + 255) & ~size_t(255);
        void* p = (char*) arena + off;
        off += bytes;
        return p;
    };
    gf::MoeDev d;
    d.n_keys = E;
    d.tab = (unsigned long long*) take((2 * E + gf::kSpares) * 8);
    d.scratch = (unsigned long long*) take(K * 8);
    d.plan_ptr = (unsigned long long*) take(K * 8);
    d.plan_w = (float*) take(K * 4);
    d.plan_id = (int*) take(K * 4);
    d.fetch_src = (unsigned long long*) take(K * 8);
    d.pf_src = (unsigned long long*) take(K * 8);
    d.pf_dst = (unsigned long long*) take(K * 8);
    d.pf_n = (int*) take(4);
    d.seq = (unsigned int*) take(4);
    d.wait_seq = (unsigned int*) take(4);
    d.cpu_seq = (unsigned int*) take(4);
    d.cpu_flag = (int*) take(4);
    d.cpu_part = (float*) take((size_t) gf::kMaxEmbd * 4);
    float* logits = (float*) take(E * 4);
    float* bias = (float*) take(E * 4);
    float* input = (float*) take((size_t) gf::kMaxEmbd * 4);
    if (off > abytes) return 3;
    const size_t ring_bytes = sizeof(gf::MoeRequest) * gf::kRingSize + sizeof(int);
    gf::MoeRequest* ring = nullptr;
    CHECK(cudaHostAlloc((void**) &ring, ring_bytes, cudaHostAllocMapped));
    std::memset(ring, 0xA5, ring_bytes);   // the canary everywhere
    void* mapped = nullptr;
    CHECK(cudaHostGetDevicePointer(&mapped, ring, 0));
    d.ring = mapped;
    auto* route_error = (volatile int*) (ring + gf::kRingSize);
    d.route_error = (volatile int*) ((gf::MoeRequest*) d.ring + gf::kRingSize);
    *route_error = 0;
    // every expert in the RAM tier only (rtab), all 8 to the host: the device publishes x
    std::vector<unsigned long long> table(2 * E + gf::kSpares, 0ull);
    for (int e = 0; e < E; ++e) table[(size_t) E + e] = 0x100000000ull + e;
    CHECK(cudaMemcpy(d.tab, table.data(), table.size() * 8, cudaMemcpyHostToDevice));
    std::vector<float> biases(E, -10.0f), x(N);
    for (int i = 0; i < K; ++i) biases[(size_t) (17 * i + 3) % E] = 8.0f - i;
    for (int e = 0; e < N; ++e) x[(size_t) e] = (float) (e + 1);
    CHECK(cudaMemcpy(bias, biases.data(), E * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(input, x.data(), N * 4, cudaMemcpyHostToDevice));
    // the route lands in slot seq % kRingSize; start at seq kRingSize - 2 so slot kRingSize - 1 is the route's and the
    // canary slot is the ring's last + the route_error int after it (the last slot spilled past the allocation)
    const unsigned int start = gf::kRingSize - 2;
    CHECK(cudaMemcpy(d.seq, &start, 4, cudaMemcpyHostToDevice));
    const unsigned long long plan = 8ull << 32;
    const int e0 = gf::launch_errors();
    for (int slot_pass = 0; slot_pass < 2; ++slot_pass) {
        gf::moe_route(logits, bias, E, K, 2.5f, true, 0, input, N, d, nullptr, nullptr, 1.0f, 0, nullptr, s, nullptr,
                      nullptr, 0, nullptr, nullptr, 0, K, plan);
        CHECK(cudaStreamSynchronize(s));
        const unsigned int seq = start + 1 + (unsigned) slot_pass;
        const int slot = (int) (seq % gf::kRingSize), next = (slot + 1) % gf::kRingSize;
        const gf::MoeRequest& rq = ring[slot];
        expect(rq.seq == seq && rq.error == 0 && rq.cpu_mask == 0xffu, "route published with all 8 on the CPU lane");
        bool x_ok = true;
        for (int e = 0; e < N; ++e) x_ok &= rq.x[e] == x[(size_t) e];
        expect(x_ok, "request x carries the 6144-wide input");
        // the request's own tail beyond n_embd and the whole next slot keep the canary
        const uint8_t* tail = (const uint8_t*) (rq.x + N);
        bool tail_ok = true;
        for (size_t b = 0; b < (size_t) (gf::kMaxEmbd - N) * 4; ++b) tail_ok &= tail[b] == 0xA5;
        expect(tail_ok, "x[6144..8191] untouched");
        if (slot_pass == 0) {   // slot kRingSize - 1: next is the route_error word (reset to 0 above)
            expect(*route_error == 0, "route_error after the last slot untouched");
        } else {
            const uint8_t* nb = (const uint8_t*) &ring[next];
            bool next_ok = true;
            for (size_t b = 0; b < sizeof(gf::MoeRequest); ++b) next_ok &= nb[b] == 0xA5;
            expect(next_ok, "ring slot i + 1 untouched (canary)");
        }
    }
    // the hard limits: refused, counted, nothing published
    unsigned int seq_before = 0;
    CHECK(cudaMemcpy(&seq_before, d.seq, 4, cudaMemcpyDeviceToHost));
    gf::moe_route(logits, bias, E, K, 2.5f, true, 0, input, gf::kMaxEmbd + 32, d, nullptr, nullptr, 1.0f, 0, nullptr,
                  s);
    gf::moe_route(logits, bias, E, 9, 2.5f, true, 0, input, N, d, nullptr, nullptr, 1.0f, 0, nullptr, s);
    gf::moe_route(logits, bias, 513, K, 2.5f, true, 0, input, N, d, nullptr, nullptr, 1.0f, 0, nullptr, s);
    gf::moe_wait(d, gf::kMaxEmbd + 32, s);
    CHECK(cudaStreamSynchronize(s));
    unsigned int seq_after = 0;
    CHECK(cudaMemcpy(&seq_after, d.seq, 4, cudaMemcpyDeviceToHost));
    expect(gf::launch_errors() == e0 + 4, "n_embd > kMaxEmbd, k > 8, n_expert > 512, wide wait: refused and counted");
    expect(seq_after == seq_before, "a refused route publishes nothing");
    cudaFreeHost(ring);
    cudaFree(arena);
    return 0;
}

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "glmd_width_gpu_test: no CUDA device - skipped\n");
        return 77;   // ctest SKIP_RETURN_CODE (cmake/strata.d/45-glm-dsa-mixer.cmake)
    }
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (int n : {4096, 6144, 8192})
        if (int rc = norms_at(n, s)) return rc;
    if (int rc = bad_shapes(s)) return rc;
    if (int rc = ring_canary(s)) return rc;
    cudaStreamDestroy(s);
    if (g_fail != 0) {
        std::fprintf(stderr, "glmd_width_gpu_test: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::puts("glmd_width_gpu_test: ok");
    return 0;
}
