// tests/hip/mm3_attn_bench.cu - Standalone microbenchmark and parity test for MiniMax-M3.1 Wave64 kernels on gfx906
// Verifies Stage 1 (CPU Reference Parity for block_scores, select_blocks, and rows_dot/rows_dot_q8)
// and Stage 2a (Measured GPU Wave64 vs. GPU Warp32 Baseline Microbenchmark, < 2 GiB VRAM).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "mm3_cands.hpp"
#include "mm3_kernels.cuh"

namespace k = mm3::k;

namespace {

static void check_hip(hipError_t e, const char* msg) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "[ERROR] HIP failure in %s: %s\n", msg, hipGetErrorString(e));
        std::exit(1);
    }
}

static uint16_t f2bf_host(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static float bf2f_host(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static uint16_t f2h_host(float f) {
    __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

static float h2f_host(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, 2);
    return __half2float(h);
}

template <class T>
T* dev_alloc_copy(const std::vector<T>& h) {
    T* d = nullptr;
    check_hip(hipMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)), "hipMalloc");
    if (!h.empty()) {
        check_hip(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "hipMemcpy H2D");
    }
    return d;
}

template <class T>
std::vector<T> host_copy(const T* d, size_t n) {
    std::vector<T> h(n);
    if (n > 0) {
        check_hip(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "hipMemcpy D2H");
    }
    return h;
}

double time_kernel_us(hipStream_t stream, int warmups, int iters, const std::function<void()>& fn) {
    for (int i = 0; i < warmups; ++i) fn();
    check_hip(hipStreamSynchronize(stream), "warmup sync");

    hipEvent_t start, stop;
    check_hip(hipEventCreate(&start), "event create");
    check_hip(hipEventCreate(&stop), "event create");

    check_hip(hipEventRecord(start, stream), "event record start");
    for (int i = 0; i < iters; ++i) fn();
    check_hip(hipEventRecord(stop, stream), "event record stop");
    check_hip(hipEventSynchronize(stop), "event sync");

    float ms = 0.0f;
    check_hip(hipEventElapsedTime(&ms, start, stop), "event elapsed time");

    check_hip(hipEventDestroy(start), "event destroy");
    check_hip(hipEventDestroy(stop), "event destroy");
    return (ms * 1000.0) / iters;
}

// ---- Stock Warp32 Baseline Reference Kernels (for live hardware timing comparison) ----
__device__ __forceinline__ bool better_ref(float a, int ia, float b, int ib) {
    return a > b || (a == b && ia < ib);
}

__global__ void __launch_bounds__(256) warp32_block_scores_tok_ref_kernel(
    const float* iq, const uint16_t* ic, const int* pos, int Hi, int nblk, int blk, float* bs) {
    __shared__ __align__(16) float qv[4][128];
    __shared__ float red[8];
    const int tid = threadIdx.x, b = blockIdx.x;
    const int p = pos[0];
    if (b > min(p / blk, nblk - 1)) return;
    for (int i = tid; i < Hi * 128; i += 256) qv[i / 128][i % 128] = iq[i];
    __syncthreads();
    const int v = tid >> 6, j0 = tid & 63;
    float best = -INFINITY;
    if (v < Hi) {
#pragma unroll 1
        for (int jj = 0; jj < 2; ++jj) {
            const int j = j0 + 64 * jj, kp = b * blk + j;
            if (j >= blk || kp > p) continue;
            const uint4* kr = (const uint4*)(ic + (size_t)kp * 128);
            float dot = 0.0f;
#pragma unroll 4
            for (int c = 0; c < 16; ++c) {
                const uint4 w = kr[c];
                const float4 qa = *(const float4*)&qv[v][c * 8], qb = *(const float4*)&qv[v][c * 8 + 4];
                dot = fmaf(qa.x, __uint_as_float(w.x << 16), dot);
                dot = fmaf(qa.y, __uint_as_float(w.x & 0xffff0000u), dot);
                dot = fmaf(qa.z, __uint_as_float(w.y << 16), dot);
                dot = fmaf(qa.w, __uint_as_float(w.y & 0xffff0000u), dot);
                dot = fmaf(qb.x, __uint_as_float(w.z << 16), dot);
                dot = fmaf(qb.y, __uint_as_float(w.z & 0xffff0000u), dot);
                dot = fmaf(qb.z, __uint_as_float(w.w << 16), dot);
                dot = fmaf(qb.w, __uint_as_float(w.w & 0xffff0000u), dot);
            }
            best = fmaxf(best, dot);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) best = fmaxf(best, __shfl_xor(best, o, 32));
    if ((tid & 31) == 0) red[tid >> 5] = best;
    __syncthreads();
    if (tid < Hi) bs[(size_t)tid * nblk + b] = fmaxf(red[2 * tid], red[2 * tid + 1]);
}

constexpr int kRefSelWarps = 4;

__global__ void warp32_select_ref_kernel(
    const float* bs, const int* pos, int T, int Hi, int nblk, int blk, int topk, int* kept, bool stage) {
    extern __shared__ float srow[];
    const int wl = threadIdx.x >> 5, wid = blockIdx.x * kRefSelWarps + wl, lane = threadIdx.x & 31;
    if (wid >= T * Hi) return;
    const int h = wid / T, t = wid % T;
    const int qb = pos[t] / blk;
    int* out = kept + ((size_t)h * T + t) * topk;
    const float* row = bs + ((size_t)h * T + t) * nblk;
    if (qb + 1 <= topk) {
        for (int i = lane; i < topk; i += 32) out[i] = i <= qb ? i : -1;
        return;
    }
    if (stage) {
        float* sr = srow + (size_t)wl * nblk;
        for (int b = lane; b < qb; b += 32) sr[b] = row[b];
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        row = sr;
    }
    if (lane == 0) out[0] = qb;
    float ps = INFINITY;
    int pi = -1;
    for (int r = 1; r < topk; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int b = lane; b < qb; b += 32) {
            const float sv = row[b];
            if (better_ref(ps, pi, sv, b) && better_ref(sv, b, bv, bi)) {
                bv = sv;
                bi = b;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, 32);
            const int oi = __shfl_xor(bi, o, 32);
            if (better_ref(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) out[r] = bi == 0x7fffffff ? -1 : bi;
        ps = bv;
        pi = bi;
    }
}

void launch_warp32_select_ref(
    const float* bs, const int* pos, int T, int Hi, int nblk, int blk, int topk, int* kept, hipStream_t s) {
    const int warps = T * Hi;
    const size_t smem = (size_t)kRefSelWarps * nblk * sizeof(float);
    const bool stage = smem <= 48 * 1024;
    warp32_select_ref_kernel<<<(warps + kRefSelWarps - 1) / kRefSelWarps, 32 * kRefSelWarps, stage ? smem : 0, s>>>(
        bs, pos, T, Hi, nblk, blk, topk, kept, stage);
}

// CPU reference helper for a single (h, t) row of block_scores and select_blocks
void compute_cpu_scores_and_kept(
    const std::vector<float>& iq, const std::vector<uint16_t>& ic, int pos_t, int T, int h, int t,
    int nblk, int blk, int topk, std::vector<float>& want_scores, std::vector<int>& want_kept) {
    const int qb = pos_t / blk;
    want_scores.assign(qb + 1, -INFINITY);
    std::vector<std::pair<float, int>> sc;
    for (int b = 0; b <= qb && b < nblk; ++b) {
        float best_cpu = -INFINITY;
        for (int j = 0; j < blk; ++j) {
            const int kp = b * blk + j;
            if (kp > pos_t) continue;
            float dot = 0.0f;
            for (int d = 0; d < 128; ++d) {
                dot = std::fmaf(iq[((size_t)t * (iq.size() / (T * 128)) + h) * 128 + d],
                                bf2f_host(ic[(size_t)kp * 128 + d]), dot);
            }
            best_cpu = std::fmax(best_cpu, dot);
        }
        want_scores[b] = best_cpu;
        if (b < qb) sc.push_back({best_cpu, b});
    }

    want_kept.assign(1, qb);
    std::vector<std::pair<float, int>> ok;
    for (auto& x : sc) {
        if (!std::isnan(x.first)) ok.push_back(x);
    }
    std::sort(ok.begin(), ok.end(), [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    for (int i = 0; i + 1 < topk; ++i) {
        want_kept.push_back(i < (int)ok.size() ? ok[(size_t)i].second : -1);
    }
    if (qb + 1 <= topk) {
        want_kept.resize(topk);
        for (int i = 0; i < topk; ++i) want_kept[(size_t)i] = i <= qb ? i : -1;
    }
}

} // namespace

int main(int argc, char** argv) {
    bool parity_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--parity-only") == 0) parity_only = true;
    }

    std::printf("========================================================================\n");
    std::printf("   MiniMax-M3.1 Wave64 Attention Microbenchmark (gfx906 / CDNA1)       \n");
    std::printf("========================================================================\n");

    int device = 0;
    check_hip(hipGetDevice(&device), "get device");
    hipDeviceProp_t prop;
    check_hip(hipGetDeviceProperties(&prop, device), "get device props");
    std::printf("[INFO] Running on Device %d: %s (gcnArch: %s, warpSize: %d)\n",
                device, prop.name, prop.gcnArchName, prop.warpSize);

    size_t free_mem = 0, total_mem = 0;
    check_hip(hipMemGetInfo(&free_mem, &total_mem), "hipMemGetInfo initial");
    const size_t initial_free = free_mem;
    size_t peak_allocated_bytes = 0;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int blk = 128;
    const int topk = 16;

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");

    bool all_parity_pass = true;

    std::printf("\n--- Stage 1: Bit-Exact CPU Reference Parity Validation ---\n");

    // 1A. Verify block_scores and select_blocks across Hi={1, 4}, T={1, 8} (including NaN query)
    for (int ctx : {8192, 32768}) {
        for (int Hi : {1, 4}) {
            for (int T : {1, 8}) {
                const int nblk = ctx / blk;
                const int p0 = ctx - T;
                std::vector<float> iq((size_t)T * Hi * 128);
                std::vector<uint16_t> ic((size_t)ctx * 128);
                for (float& x : iq) x = nd(rng);
                for (uint16_t& x : ic) x = f2bf_host(nd(rng));
                if (T == 8) iq[3 * Hi * 128] = NAN; // Query 3: NaN scores test

                std::vector<int> pos(T);
                for (int t = 0; t < T; ++t) pos[t] = p0 + t;

                float* d_iq = dev_alloc_copy(iq);
                uint16_t* d_ic = dev_alloc_copy(ic);
                int* d_pos = dev_alloc_copy(pos);
                float* d_bs = nullptr;
                int* d_kept = nullptr;
                check_hip(hipMalloc(&d_bs, (size_t)Hi * T * nblk * sizeof(float)), "malloc d_bs");
                check_hip(hipMalloc(&d_kept, (size_t)Hi * T * topk * sizeof(int)), "malloc d_kept");
                check_hip(hipMemsetAsync(d_kept, 0xcd, (size_t)Hi * T * topk * sizeof(int), stream), "memset d_kept");

                k::block_scores(d_iq, d_ic, d_pos, T, Hi, nblk, blk, d_bs, stream);
                k::select_blocks(d_bs, d_pos, T, Hi, nblk, blk, topk, d_kept, stream);
                check_hip(hipStreamSynchronize(stream), "sync after kernels");

                auto bs_gpu = host_copy(d_bs, (size_t)Hi * T * nblk);
                auto kept_gpu = host_copy(d_kept, (size_t)Hi * T * topk);

                bool scores_ok = true;
                bool kept_ok = true;
                for (int h = 0; h < Hi; ++h) {
                    for (int t = 0; t < T; ++t) {
                        const int qb = pos[t] / blk;
                        std::vector<float> want_scores;
                        std::vector<int> want_kept;
                        compute_cpu_scores_and_kept(iq, ic, pos[t], T, h, t, nblk, blk, topk, want_scores, want_kept);
                        for (int b = 0; b < qb; ++b) {
                            const float g = bs_gpu[((size_t)h * T + t) * nblk + b];
                            const float w = want_scores[b];
                            if (!(g == w || (std::isnan(g) && std::isnan(w)))) {
                                scores_ok = false;
                            }
                        }
                        for (int i = 0; i < topk; ++i) {
                            if (kept_gpu[((size_t)h * T + t) * topk + i] != want_kept[i]) {
                                kept_ok = false;
                            }
                        }
                    }
                }

                std::printf("  [PARITY] ctx=%5d Hi=%d T=%d | block_scores: %s | select_blocks: %s\n",
                            ctx, Hi, T, scores_ok ? "PASS" : "FAIL", kept_ok ? "PASS" : "FAIL");
                if (!scores_ok || !kept_ok) all_parity_pass = false;

                check_hip(hipFree(d_iq), "free d_iq");
                check_hip(hipFree(d_ic), "free d_ic");
                check_hip(hipFree(d_pos), "free d_pos");
                check_hip(hipFree(d_bs), "free d_bs");
                check_hip(hipFree(d_kept), "free d_kept");
            }
        }
    }

    // 1B. Verify rows_dot and rows_dot_q8 (32-lane sub-warp reduction: 2 rows per 64-lane wavefront)
    {
        const int Na = 19, Nb = 13, K = 6144;
        std::vector<float> Wa((size_t)Na * K), Wb((size_t)Nb * K);
        for (float& x : Wa) x = nd(rng) * 0.25f;
        for (float& x : Wb) x = nd(rng) * 0.25f;

        // Build Q8_0 representation (34 bytes per 32 elements: FP16 scale + 32 int8)
        auto pack_q8 = [](const std::vector<float>& W, int N, int K_dim, std::vector<uint8_t>& Wq8, std::vector<float>& W_deq) {
            const size_t nblocks = (size_t)N * (K_dim / 32);
            Wq8.resize(nblocks * 34);
            W_deq.resize((size_t)N * K_dim);
            for (size_t b = 0; b < nblocks; ++b) {
                float am = 0.0f;
                for (int i = 0; i < 32; ++i) am = std::fmax(am, std::fabs(W[b * 32 + i]));
                const uint16_t d16 = f2h_host(am / 127.0f);
                const float d = h2f_host(d16);
                uint8_t* dst = Wq8.data() + b * 34;
                dst[0] = (uint8_t)(d16 & 0xffu);
                dst[1] = (uint8_t)(d16 >> 8);
                for (int i = 0; i < 32; ++i) {
                    const int q = am > 0.0f ? (int)std::nearbyint(W[b * 32 + i] / (am / 127.0f)) : 0;
                    const int8_t qi = (int8_t)std::max(-127, std::min(127, q));
                    dst[2 + i] = (uint8_t)qi;
                    W_deq[b * 32 + i] = d * (float)qi;
                }
            }
        };

        std::vector<uint8_t> Wa_q8, Wb_q8;
        std::vector<float> Wa_deq, Wb_deq;
        pack_q8(Wa, Na, K, Wa_q8, Wa_deq);
        pack_q8(Wb, Nb, K, Wb_q8, Wb_deq);

        float* d_Wa = dev_alloc_copy(Wa);
        uint8_t* d_Wa_q8 = dev_alloc_copy(Wa_q8);
        uint8_t* d_Wb_q8 = dev_alloc_copy(Wb_q8);

        bool rows_dot_ok = true;
        for (int T : {1, 4, 9}) {
            std::vector<float> X((size_t)T * K);
            for (float& x : X) x = nd(rng);
            float* d_X = dev_alloc_copy(X);
            float *d_Ya = nullptr, *d_Ya_q8 = nullptr, *d_Yb_q8 = nullptr;
            check_hip(hipMalloc(&d_Ya, (size_t)T * Na * sizeof(float)), "malloc d_Ya");
            check_hip(hipMalloc(&d_Ya_q8, (size_t)T * Na * sizeof(float)), "malloc d_Ya_q8");
            check_hip(hipMalloc(&d_Yb_q8, (size_t)T * Nb * sizeof(float)), "malloc d_Yb_q8");

            k::rows_dot(d_Wa, Na, K, d_X, T, d_Ya, stream);
            k::rows_dot_q8(d_Wa_q8, Na, d_Wb_q8, Nb, K, d_X, T, d_Ya_q8, d_Yb_q8, stream);
            check_hip(hipStreamSynchronize(stream), "sync rows_dot");

            auto Ya_gpu = host_copy(d_Ya, (size_t)T * Na);
            auto Ya_q8_gpu = host_copy(d_Ya_q8, (size_t)T * Na);
            auto Yb_q8_gpu = host_copy(d_Yb_q8, (size_t)T * Nb);

            // Check against CPU reference (32-lane stride accumulation + 32-lane butterfly reduction)
            auto cpu_row_dot = [&](const std::vector<float>& W_mat, int r, int t) {
                float lanes[32] = {};
                for (int l = 0; l < 32; ++l) {
                    float acc = 0.0f;
                    for (int i = l; i < K; i += 32) {
                        acc = std::fmaf(W_mat[(size_t)r * K + i], X[(size_t)t * K + i], acc);
                    }
                    lanes[l] = acc;
                }
                for (int o = 16; o > 0; o >>= 1) {
                    float next[32];
                    for (int l = 0; l < 32; ++l) next[l] = lanes[l] + lanes[l ^ o];
                    std::memcpy(lanes, next, sizeof(lanes));
                }
                return lanes[0];
            };

            for (int t = 0; t < T; ++t) {
                for (int r = 0; r < Na; ++r) {
                    if (Ya_gpu[(size_t)t * Na + r] != cpu_row_dot(Wa, r, t)) rows_dot_ok = false;
                    if (Ya_q8_gpu[(size_t)t * Na + r] != cpu_row_dot(Wa_deq, r, t)) rows_dot_ok = false;
                }
                for (int r = 0; r < Nb; ++r) {
                    if (Yb_q8_gpu[(size_t)t * Nb + r] != cpu_row_dot(Wb_deq, r, t)) rows_dot_ok = false;
                }
            }

            check_hip(hipFree(d_X), "free d_X");
            check_hip(hipFree(d_Ya), "free d_Ya");
            check_hip(hipFree(d_Ya_q8), "free d_Ya_q8");
            check_hip(hipFree(d_Yb_q8), "free d_Yb_q8");
        }
        std::printf("  [PARITY] rows_dot & rows_dot_q8 (T=1,4,9, Na=19, Nb=13, K=6144): %s\n",
                    rows_dot_ok ? "PASS" : "FAIL");
        if (!rows_dot_ok) all_parity_pass = false;

        check_hip(hipFree(d_Wa), "free d_Wa");
        check_hip(hipFree(d_Wa_q8), "free d_Wa_q8");
        check_hip(hipFree(d_Wb_q8), "free d_Wb_q8");
    }

    bool all_bench_parity_pass = true;
    if (!parity_only) {
        const int Hi = 4;
        const int ctx_points[] = {32768, 65536, 131072};

        std::printf("\n--- Stage 2a: Standalone Microbenchmarks (Measured GPU Wave64 vs. GPU Warp32) ---\n");
        std::printf("  %-10s | %-16s | %-14s | %-14s | %-10s | %-6s\n",
                    "Context", "Kernel", "Wave64 Latency", "Warp32 Latency", "Speedup", "Parity");
        std::printf("  ---------------------------------------------------------------------------------\n");

        for (int ctx : ctx_points) {
            const int nblk = ctx / blk;
            const int pos_val = ctx - 1;
            std::vector<float> iq(Hi * 128);
            std::vector<uint16_t> ic((size_t)ctx * 128);
            for (float& x : iq) x = nd(rng);
            for (uint16_t& x : ic) x = f2bf_host(nd(rng));
            std::vector<int> pos = {pos_val};

            float* d_iq = dev_alloc_copy(iq);
            uint16_t* d_ic = dev_alloc_copy(ic);
            int* d_pos = dev_alloc_copy(pos);
            float *d_bs_w64 = nullptr, *d_bs_w32 = nullptr;
            int *d_kept_w64 = nullptr, *d_kept_w32 = nullptr;
            check_hip(hipMalloc(&d_bs_w64, (size_t)Hi * nblk * sizeof(float)), "malloc d_bs_w64");
            check_hip(hipMalloc(&d_bs_w32, (size_t)Hi * nblk * sizeof(float)), "malloc d_bs_w32");
            check_hip(hipMalloc(&d_kept_w64, (size_t)Hi * topk * sizeof(int)), "malloc d_kept_w64");
            check_hip(hipMalloc(&d_kept_w32, (size_t)Hi * topk * sizeof(int)), "malloc d_kept_w32");

            size_t cur_free = 0, cur_total = 0;
            check_hip(hipMemGetInfo(&cur_free, &cur_total), "hipMemGetInfo point");
            if (initial_free > cur_free) {
                peak_allocated_bytes = std::max(peak_allocated_bytes, initial_free - cur_free);
            }

            // 1. Time Wave64 block_scores vs Measured Warp32 block_scores
            double t_bs_w64_us = time_kernel_us(stream, 20, 100, [&]() {
                k::block_scores(d_iq, d_ic, d_pos, 1, Hi, nblk, blk, d_bs_w64, stream);
            });
            double t_bs_w32_us = time_kernel_us(stream, 20, 100, [&]() {
                warp32_block_scores_tok_ref_kernel<<<(unsigned)nblk, 256, 0, stream>>>(
                    d_iq, d_ic, d_pos, Hi, nblk, blk, d_bs_w32);
            });

            // 2. Time Wave64 select_blocks vs Measured Warp32 select_blocks
            double t_sel_w64_us = time_kernel_us(stream, 20, 100, [&]() {
                k::select_blocks(d_bs_w64, d_pos, 1, Hi, nblk, blk, topk, d_kept_w64, stream);
            });
            double t_sel_w32_us = time_kernel_us(stream, 20, 100, [&]() {
                launch_warp32_select_ref(d_bs_w32, d_pos, 1, Hi, nblk, blk, topk, d_kept_w32, stream);
            });

            auto h_bs_w64 = host_copy(d_bs_w64, (size_t)Hi * nblk);
            auto h_bs_w32 = host_copy(d_bs_w32, (size_t)Hi * nblk);
            auto h_kept_w64 = host_copy(d_kept_w64, (size_t)Hi * topk);
            auto h_kept_w32 = host_copy(d_kept_w32, (size_t)Hi * topk);

            const bool bs_match = std::memcmp(h_bs_w64.data(), h_bs_w32.data(), h_bs_w64.size() * sizeof(float)) == 0;
            const bool kept_match = (h_kept_w64 == h_kept_w32);
            if (!bs_match || !kept_match) all_bench_parity_pass = false;

            const double speedup_bs = t_bs_w32_us / t_bs_w64_us;
            const double speedup_sel = t_sel_w32_us / t_sel_w64_us;

            std::printf("  %6d tok | %-16s | %8.2f us    | %8.2f us    | %5.2fx     | %s\n",
                        ctx, "block_scores", t_bs_w64_us, t_bs_w32_us, speedup_bs, bs_match ? "PASS" : "FAIL");
            std::printf("  %6d tok | %-16s | %8.2f us    | %8.2f us    | %5.2fx     | %s\n",
                        ctx, "select_blocks", t_sel_w64_us, t_sel_w32_us, speedup_sel, kept_match ? "PASS" : "FAIL");

            check_hip(hipFree(d_iq), "free d_iq");
            check_hip(hipFree(d_ic), "free d_ic");
            check_hip(hipFree(d_pos), "free d_pos");
            check_hip(hipFree(d_bs_w64), "free d_bs_w64");
            check_hip(hipFree(d_bs_w32), "free d_bs_w32");
            check_hip(hipFree(d_kept_w64), "free d_kept_w64");
            check_hip(hipFree(d_kept_w32), "free d_kept_w32");
        }
    }

    const double vram_used_mib = (double)peak_allocated_bytes / (1024.0 * 1024.0);
    const bool vram_ok = vram_used_mib < 2048.0;
    const bool overall_ok = all_parity_pass && all_bench_parity_pass && vram_ok;

    std::printf("\n[INFO] Peak Benchmark VRAM Allocated: %.2f MiB (< 2048 MiB Gate 2a limit: %s)\n",
                vram_used_mib, vram_ok ? "PASS" : "FAIL");
    std::printf("[INFO] Numerical Parity Gate 1: %s\n", all_parity_pass ? "PASSED" : "FAILED");
    if (!parity_only) {
        std::printf("[INFO] Standalone Microbench Gate 2a: %s\n",
                    (all_bench_parity_pass && vram_ok) ? "PASSED" : "FAILED");
    }
    std::printf("========================================================================\n");

    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
    return overall_ok ? 0 : 1;
}
