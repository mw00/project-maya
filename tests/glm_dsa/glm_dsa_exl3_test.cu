// tests/glm_dsa/glm_dsa_exl3_test.cu - the EXL3 kernels at GLM-5.3's (glm-dsa) shapes against the scalar CPU
// reference (exl3_ref) and the CPU lane's own kernels (exl3_cpu).  Synthetic tensors only: random trellis words (any
// bit pattern is a valid encoding, a half-integer K too) and random FP16 channel scales, packed by the test - no
// model, no quant.  tests/exl3_test.cu covers the kernels at GLM-5.3-Flash's widths (4096 / 2048) and DeepSeek's
// (5120); GLM-5.3 adds n_embd 6144 (above 4096: every product over it is summed in pieces, and v1.1.8's expert kernels
// - a V100's default - do not take it), q_b 2048 -> 16384, attn_output 16384 -> 6144, kv_a 6144 -> 640 (512 + 64 =
// 576, padded to 640 by exllamav3, whose Linear pads to multiples of 128), the dense FFN's 12288, NextN's eh_proj
// 12288 -> 6144 and the 154880-row head.  Bitrates K 2, 2.5, 3, 4, 5, 8 (and the head's 6), codebooks 3inst, mcg, mul1.
//
//   glm_dsa_exl3_test [--only s1,s2,...] [--blas-old-layout] [--threads N]
//
// Sections (all by default):
//   mv     every trunk matrix x K x codebook: one activation (F32 and q8_1) against the reference, 2 / 8 / 16 at once
//          bit for bit one job each, a column view (and its 16)
//   batch  the engine's calls of several jobs: q_a + kv_a, the shared expert's gate + up, six bitrates in one call; a
//          verify window's 16 tokens of a two-job call bit for bit the token path's 16 calls
//   head   154880 x 6144 at K 6 and 8: column views [0, 1280) and across row 98304 against the reference (mv, and the
//          reconstruction); the whole head at mv_cols() activations bit for bit one job each, and what a
//          16-activation job of it does (exl3.cu's kernels refuse it: its input pieces' partials outgrow the scratch)
//   moe    the routed experts (6144 <-> 2048, top-8 with a skipped entry, the CPU lane's part on and off, the shared
//          expert's add, GLM-5.3's unclamped swiglu and a clamp), gate / up and down at different K, mixed-K layers
//   win    win_experts: 7 / 16 / 32 tokens bit for bit the token path's; 33 refused (its scratch)
//   set    the prompt path's expert sets (set_gate_up + set_down): 1 to 130 rows an expert, a set from a nonzero row,
//          mixed-K classes, guard bytes around h and the output
//   gemm   gemm_rows at the trunk's shapes, padded strides, beta 0 and 1
//   recon  reconstruct_inner / reconstruct_inner_rows (exact) and reconstruct_rows (F16, BF16) on column views
//   blas   the prompt path's cuBLAS routes rebuilt as src/core/glm_prefill.cu has them: a heavy expert
//          (x3_run_blas, both STRATA_GLM_EXL3_ROT values) with gate / up packed 2 n_ff wide at the end of the expert's
//          output rows - the layout glm-dsa needs (n_embd 6144 != 2 n_ff; GPU day 1's GGUF fix) - and the dense
//          weights' slices (reconstruct_rows of 128-row views + one GEMM a slice)
//   cpu    the CPU lane's band kernels at the experts' shapes: every tier this CPU has (scalar, AVX2, AVX2's 16-bit
//          lanes and byte shuffles, AVX-512, AVX-512 VNNI) bit for bit an exact model of their arithmetic, the lane
//          against the reference, n activations, split-K pieces, the band GEMM
//   --blas-old-layout  also the heavy expert in the layout x3_run_blas has today (gate / up written n_embd apart, read
//          2 n_ff apart): informational, its error is printed and not counted (item 4b fixes the engine)
//
// Exit 0 when every case is within exl3_test's tolerances (relative RMS 2e-3 for a product, 3e-3 for the experts' and
// the GEMMs' chains), 1 otherwise, 77 without a GPU (unless only the cpu section runs).  A call a kernel refuses
// fails, except the refusals listed as expected (printed at the end: they are what the engine has to cut around).
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/cpu/exl3_ref.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/glm_batch.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

// the CPU lane's AVX2 byte-shuffle switch (exl3_cpu_set_avx2_shuf) exists in the x86 kernels only: the portable
// build's stubs (src/kernels/cpu/portable.cpp) have no such function, and no lane at all
#if defined(__x86_64__) || defined(_M_X64)
#define GLM_DSA_EXL3_X86 1
#endif

namespace ex = strata::kernels::exl3;
namespace cpu = strata::kernels::cpu;
namespace gb = strata::kernels::glmb;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

namespace {

// GLM-5.3's widths (the pack index of GLM-5.3-UD-IQ2_M: blk.N.* and output.weight)
constexpr int kEmbd = 6144, kFfExp = 2048, kFfDense = 12288, kVocab = 154880, kTopK = 8;
// activations the CPU checks feed the lane at once: 11 (past kExl3CpuMaxN, two runs), the band GEMM 21 (3 tiles of 6
// and 3 over).  At namespace scope: nvcc's lambda rewrite turns a local constant a lambda reads into a capture, and
// MSVC then refuses it as an array bound (C2131)
constexpr int kCpuNT = 11, kGemmNT = 21;
const float kNoClamp = std::numeric_limits<float>::infinity();   // glm_dsa_geometry.cpp: swiglu_exp = +inf

int g_fail = 0, g_checks = 0;
std::mt19937_64 g_rng(530530);
int g_threads = 1;
bool g_legacy_mv = false;              // mv runs v1.1.8's kernels on this device (probe_family)
std::vector<std::string> g_refused;    // calls refused as expected (listed at the end)

#define CK(x)                                                                                        \
    do {                                                                                             \
        const cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(2);                                                                            \
        }                                                                                            \
    } while (0)
#define CB(x)                                                                                        \
    do {                                                                                             \
        const cublasStatus_t s_ = (x);                                                               \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                           \
            std::fprintf(stderr, "%s:%d %s: BLAS status %d\n", __FILE__, __LINE__, #x, (int) s_);    \
            std::exit(2);                                                                            \
        }                                                                                            \
    } while (0)

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// f(begin, end) over [0, n) in g_threads contiguous pieces
void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& f) {
    const int64_t T = std::max<int64_t>(1, std::min<int64_t>(g_threads, n));
    if (T <= 1) {
        if (n > 0) f(0, n);
        return;
    }
    std::vector<std::thread> th;
    th.reserve((size_t) T);
    for (int64_t t = 0; t < T; ++t) th.emplace_back(f, n * t / T, n * (t + 1) / T);
    for (auto& x : th) x.join();
}

// ---------------------------------------------------------------- random tensors
uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// n random words (splitmix64, one stream cut in 64K-word chunks: the same words for any thread count)
constexpr size_t kChunk = (size_t) 1 << 16;
std::vector<uint32_t> rand_words(size_t n) {
    std::vector<uint32_t> w(n);
    const uint64_t seed = g_rng();
    parallel_for((int64_t) ((n + kChunk - 1) / kChunk), [&](int64_t c0, int64_t c1) {
        for (int64_t c = c0; c < c1; ++c) {
            uint64_t s = seed + 0x9E3779B97F4A7C15ull * (uint64_t) c * kChunk;
            const size_t a = (size_t) c * kChunk, b = std::min(n, a + kChunk);
            for (size_t i = a; i < b; ++i) {
                s += 0x9E3779B97F4A7C15ull;
                w[i] = (uint32_t) (mix64(s) >> 32);
            }
        }
    });
    return w;
}
// channel scales: random sign, magnitude in [0.25, 2] (exl3_test's)
std::vector<uint16_t> rand_scales(int n) {
    std::uniform_real_distribution<float> u(0.25f, 2.0f);
    std::vector<uint16_t> v((size_t) n);
    for (auto& x : v) {
        const bool neg = (g_rng() & 1) != 0;
        const float a = u(g_rng);
        x = f16_from_f32(neg ? -a : a);
    }
    return v;
}
std::vector<float> rand_x(size_t n, float sd) {
    std::normal_distribution<float> d(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g_rng);
    return v;
}
// an activation scale that keeps an expert's gate / up near N(0, 2.5) over k inputs (the scales' RMS is ~1.23 and a
// decoded weight's ~1): the swiglu stays in FP16's range where the prompt path holds h as FP16, and a clamp of 7
// changes a few values, not most
float expert_x_sd(int k) { return 2.5f / (1.5f * std::sqrt((float) k)); }

// ---------------------------------------------------------------- device helpers
template <typename T>
T* dev_copy(const T* p, size_t n) {
    T* d = nullptr;
    CK(cudaMalloc((void**) &d, n * sizeof(T) + 256));
    CK(cudaMemcpy(d, p, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
T* dev_copy(const std::vector<T>& v) {
    return dev_copy(v.data(), v.size());
}
template <typename T>
T* dev_alloc(size_t n) {
    T* d = nullptr;
    CK(cudaMalloc((void**) &d, n * sizeof(T) + 256));
    CK(cudaMemset(d, 0, n * sizeof(T) + 256));
    return d;
}
template <typename T>
std::vector<T> host_copy(const T* d, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
void dfree(const void* p) {
    if (p != nullptr) CK(cudaFree(const_cast<void*>(p)));
}

// ---------------------------------------------------------------- comparisons
// exl3_test's measure: the relative RMS error, and the worst element against the reference's RMS (at most 8 tol)
struct Err {
    double rel = 0.0, worst = 0.0;
    bool finite = true;
};
Err err_of(const float* got, const float* ref, size_t n) {
    double se = 0.0, sr = 0.0, worst = 0.0;
    bool fin = true;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) fin = false;
        const double d = (double) got[i] - ref[i];
        se += d * d;
        sr += (double) ref[i] * ref[i];
        worst = std::max(worst, std::fabs(d));
    }
    Err e;
    const double rms = std::sqrt(sr / (double) std::max<size_t>(1, n));
    e.rel = std::sqrt(se / std::max(sr, 1e-30));
    e.worst = worst / std::max(rms, 1e-30);
    e.finite = fin;
    return e;
}
// one result against its reference: counted, a FAIL line when out of tolerance; the text for the case's line
std::string check(const std::string& what, const float* got, const float* ref, size_t n, double tol) {
    const Err e = err_of(got, ref, n);
    ++g_checks;
    const bool ok = e.finite && e.rel <= tol && e.worst <= 8.0 * tol;
    char b[64];
    std::snprintf(b, sizeof b, "%.1e%s", e.rel, ok ? "" : " FAIL");
    if (!ok) {
        ++g_fail;
        std::printf("  FAIL %s: rel %.2e worst/rms %.2e%s (tolerance %.0e)\n", what.c_str(), e.rel, e.worst,
                    e.finite ? "" : " NOT FINITE", tol);
    }
    return b;
}
size_t bits_differ(const float* a, const float* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    return d;
}
// bit for bit: counted, a FAIL line when any float differs
std::string check_bits(const std::string& what, size_t differ, size_t n) {
    ++g_checks;
    if (differ == 0 && n > 0) return "same bits";
    ++g_fail;
    std::printf("  FAIL %s: %zu of %zu floats differ\n", what.c_str(), differ, n);
    return std::to_string(differ) + " differ FAIL";
}
// a call the kernels refused: an expected refusal is listed (the engine has to cut around it), any other fails
void note_refusal(const std::string& what, bool expected, const std::string& why) {
    if (expected) {
        g_refused.push_back(what + ": " + why);
        std::printf("  refused as expected: %s (%s)\n", what.c_str(), why.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL refused: %s\n", what.c_str());
    }
}

std::string kname(int k2) { return (k2 & 1) ? std::to_string(k2 / 2) + ".5" : std::to_string(k2 / 2); }
std::string kcb(int k2, int cb) { return "K" + kname(k2) + " cb" + std::to_string(cb); }
std::string dims(int k, int n) { return std::to_string(k) + "x" + std::to_string(n); }

// ---------------------------------------------------------------- matrices and the reference
struct HostMat {
    std::vector<uint32_t> tr;
    std::vector<uint16_t> suh, svh;
    int k = 0, n = 0, k2 = 0, cb = 2;
};
HostMat make_mat(int k, int n, int k2, int cb) {
    HostMat m;
    m.tr = rand_words(ex::trellis_bytes_k2(k, n, k2) / 4);
    m.suh = rand_scales(k);
    m.svh = rand_scales(n);
    m.k = k;
    m.n = n;
    m.k2 = k2;
    m.cb = cb;
    return m;
}
ex::Mat upload(const HostMat& h) {
    ex::Mat m;
    m.trellis = dev_copy(h.tr);
    m.suh = dev_copy(h.suh);
    m.svh = dev_copy(h.svh);
    m.k = h.k;
    m.n = h.n;
    m.k2 = h.k2;
    m.cb = h.cb;
    return m;
}
void free_mat(ex::Mat& m) {
    dfree(m.trellis);
    dfree(m.suh);
    dfree(m.svh);
    m = ex::Mat{};
}
// W'[r][c] of the columns [c0, c0 + nc) (row-major k x nc): exl3_ref's exl3_inner_k2 over the 16-row slices in parallel
std::vector<float> decode(const HostMat& h, int c0 = 0, int nc = -1) {
    if (nc < 0) nc = h.n;
    std::vector<float> w((size_t) h.k * nc);
    const size_t W = 4 * (size_t) h.k2, ld = (size_t) h.n / 16;
    parallel_for(h.k / 16, [&](int64_t s0, int64_t s1) {
        cpu::exl3_inner_k2(h.tr.data() + ((size_t) s0 * ld + (size_t) c0 / 16) * W, (int) ld, (int) (s1 - s0) * 16, nc,
                           h.k2, h.cb, w.data() + (size_t) s0 * 16 * nc);
    });
    return w;
}
// y[r] = H(H(x_r * suh) W') * svh for rows x_r (ldx floats apart; y rows ldy apart): exl3_ref's exl3_mv itself, one
// 128-column block at a time (each block's outputs are exl3_mv's over the whole width bit for bit: its sums are per
// column and its output transform per 128), blocks and runs of rows spread over threads.  w: k x n (decode()), svh
// the n columns' scales
void ref_rows(const float* w, int k, int n, const uint16_t* suh, const uint16_t* svh, const float* x, size_t ldx,
              int rows, float* y, size_t ldy) {
    const int nb = n / 128;
    const int per_blk = std::max(1, std::min(rows, (4 * g_threads + nb - 1) / nb));
    const int rc = (rows + per_blk - 1) / per_blk, nch = (rows + rc - 1) / rc;
    parallel_for((int64_t) nb * nch, [&](int64_t t0, int64_t t1) {
        std::vector<float> wb((size_t) k * 128);
        int64_t have = -1;
        for (int64_t t = t0; t < t1; ++t) {
            const int b = (int) (t / nch), ch = (int) (t % nch);
            if (b != have) {
                for (int i = 0; i < k; ++i)
                    std::memcpy(&wb[(size_t) i * 128], w + (size_t) i * n + (size_t) b * 128, 128 * sizeof(float));
                have = b;
            }
            for (int r = ch * rc; r < std::min(rows, (ch + 1) * rc); ++r)
                cpu::exl3_mv(wb.data(), suh, svh + (size_t) b * 128, k, 128, x + (size_t) r * ldx,
                             y + (size_t) r * ldy + (size_t) b * 128);
        }
    });
}
// the full weight in reconstruct_rows' layout, as test_recon builds it: full[c][r] = suh[r] H_k(H_n(W'))[r][c] svh[c]
// for the view's nc columns (w: k x nc, svh: the view's scales)
std::vector<float> full_rows(const std::vector<float>& w, int k, int nc, const uint16_t* suh, const uint16_t* svh) {
    std::vector<float> t = w, full((size_t) nc * k);
    parallel_for(k, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) cpu::exl3_had128(t.data() + (size_t) r * nc, (size_t) nc);
    });
    parallel_for(nc, [&](int64_t c0, int64_t c1) {
        std::vector<float> col((size_t) k);
        for (int64_t c = c0; c < c1; ++c) {
            for (int r = 0; r < k; ++r) col[(size_t) r] = t[(size_t) r * nc + c];
            cpu::exl3_had128(col.data(), (size_t) k);
            for (int r = 0; r < k; ++r)
                full[(size_t) c * k + r] = col[(size_t) r] * f32_from_f16(suh[r]) * f32_from_f16(svh[c]);
        }
    });
    return full;
}
float swiglu_ref(float g, float u, float limit) {
    if (limit > 0.0f) {
        g = std::min(g, limit);
        u = std::min(std::max(u, -limit), limit);
    }
    return g / (1.0f + std::exp(-g)) * u;
}
float round16(float v) { return f32_from_f16(f16_from_f32(v)); }

// the engine's q8_1 of x (quantize_act.cu's arithmetic) and its dequantized values (exl3_test's)
void q8_1(const float* x, size_t n, uint8_t* q, float* deq) {
    for (size_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f, sum = 0.0f;
        for (int i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(x[b * 32 + i]));
            sum += x[b * 32 + i];
        }
        const float d = amax / 127.0f;
        const uint16_t dh = f16_from_f32(d), sh = f16_from_f32(sum);
        std::memcpy(&q[b * 36], &dh, 2);
        std::memcpy(&q[b * 36 + 2], &sh, 2);
        const float dd = f32_from_f16(dh);
        for (int i = 0; i < 32; ++i) {
            const int8_t v = amax == 0.0f ? 0 : (int8_t) std::lround(std::max(-127.0f, std::min(127.0f, x[b * 32 + i] / d)));
            q[b * 36 + 4 + i] = (uint8_t) v;
            deq[b * 32 + i] = dd * (float) v;
        }
    }
}

// which kernels mv runs here: v1.1.8's (exl3_legacy.cu - NVIDIA before sm_80 by default, STRATA_GLM_EXL3_DECODE=
// legacy) take a lone job's activations in runs of 4, exl3.cu's in runs of 8 (mv_cols of a one-piece matrix)
void probe_family() {
    HostMat h = make_mat(256, 256, 4, 2);
    ex::Mat m = upload(h);
    g_legacy_mv = ex::mv_cols(m) == 4;
    free_mat(m);
}
const char* family() { return g_legacy_mv ? "v1.1.8's (exl3_legacy.cu)" : "exl3.cu's"; }

// ---------------------------------------------------------------- activations of a matvec
// nc activations of k values for mv: F32 rows xs floats apart, their q8_1 blocks qs bytes apart, the q8_1 values
// dequantized (the reference's input for a q8_1 job)
struct Acts {
    int k = 0, nc = 0;
    size_t xs = 0, qs = 0;
    std::vector<float> x, deq;
    std::vector<uint8_t> q;
    float* d_x = nullptr;
    uint8_t* d_q = nullptr;
    void make(int k_, int nc_, float sd) {
        k = k_;
        nc = nc_;
        xs = (size_t) k + 128;
        qs = xs / 32 * 36;
        x.assign((size_t) nc * xs, 0.0f);
        deq.assign((size_t) nc * xs, 0.0f);
        q.assign((size_t) nc * qs, 0);
        for (int c = 0; c < nc; ++c) {
            const std::vector<float> v = rand_x((size_t) k, sd);
            std::copy(v.begin(), v.end(), x.begin() + (std::ptrdiff_t) (c * xs));
            q8_1(v.data(), (size_t) k, q.data() + c * qs, deq.data() + c * xs);
        }
        d_x = dev_copy(x);
        d_q = dev_copy(q);
    }
    void release() {
        dfree(d_x);
        dfree(d_q);
        d_x = nullptr;
        d_q = nullptr;
    }
};
ex::MvJob job_of(const ex::Mat* m, const Acts& a, bool q8, int c0, int ncol, float* y, size_t ys, const float* bias,
                 float alpha) {
    ex::MvJob j;
    j.m = m;
    j.xf = q8 ? nullptr : a.d_x + (size_t) c0 * a.xs;
    j.xq = q8 ? (const void*) (a.d_q + (size_t) c0 * a.qs) : nullptr;
    j.y = y;
    j.bias = bias;
    j.alpha = alpha;
    j.ncol = ncol;
    j.xf_cs = a.xs;
    j.xq_cs = a.qs;
    j.y_cs = ys;
    return j;
}

// ================================================================ mv: the trunk's matrices
struct Shape {
    const char* name;
    int k, n;
};
const Shape kTrunk[] = {
    {"attn_q_a (and the shared expert's gate / up)", kEmbd, 2048},
    {"attn_kv_a_mqa (576 + exllamav3's padding)", kEmbd, 640},
    {"attn_q_b (64 heads x 256)", 2048, 16384},
    {"attn_output", 16384, kEmbd},
    {"indexer q_b (32 heads x 128)", 2048, 4096},
    {"indexer k (and proj's 32 padded to 128)", kEmbd, 128},
    {"ffn_down_shexp", kFfExp, kEmbd},
    {"dense ffn_gate / ffn_up", kEmbd, kFfDense},
    {"dense ffn_down, nextn eh_proj", kFfDense, kEmbd},
};
struct KCb {
    int k2, cb;
};
// K 2, 2.5, 3, 4, 5, 8 (k2 4, 5, 6, 8, 10, 16) x the three codebooks; the half-integer K with mul1 only (exllamav3 1.6)
const KCb kAll[] = {{4, 0},  {4, 1},  {4, 2}, {5, 2},  {6, 0},  {6, 1},  {6, 2},  {8, 0},
                    {8, 1},  {8, 2},  {10, 0}, {10, 1}, {10, 2}, {16, 0}, {16, 1}, {16, 2}};

// one matrix: 16 activations through single jobs (the first F32 one twice - an input split's counters must come back
// to zero), activation 0 F32 and q8_1 against the reference, jobs of 2 / 8 / 16 activations bit for bit the singles, a
// column view against the reference and its 16-activation job bit for bit its singles
void mv_case(const Shape& sh, int k2, int cb) {
    const int k = sh.k, n = sh.n;
    constexpr int NC = 16;
    const std::string tag = std::string(sh.name) + " " + dims(k, n) + " " + kcb(k2, cb);
    HostMat h = make_mat(k, n, k2, cb);
    ex::Mat m = upload(h);
    Acts A;
    A.make(k, NC, 2.0f);
    const std::vector<float> bias = rand_x((size_t) n, 1.0f);
    const float alpha = 0.75f;
    const size_t ys = (size_t) n + 64;
    float* d_bias = dev_copy(bias);
    float* d_f1 = dev_alloc<float>(NC * ys);   // single jobs, F32
    float* d_q1 = dev_alloc<float>(NC * ys);   // single jobs, q8_1
    float* d_n = dev_alloc<float>(NC * ys);    // one job of several
    bool refused = false;
    {
        ex::MvJob j = job_of(&m, A, false, 0, 1, d_f1, ys, d_bias, alpha);
        refused |= !ex::mv(&j, 1, 0);
    }
    for (int c = 0; c < NC; ++c)
        for (int q8 = 0; q8 < 2; ++q8) {
            ex::MvJob j = job_of(&m, A, q8 != 0, c, 1, (q8 ? d_q1 : d_f1) + c * ys, ys, d_bias, alpha);
            refused |= !ex::mv(&j, 1, 0);
        }
    CK(cudaDeviceSynchronize());
    const std::vector<float> f1 = host_copy(d_f1, NC * ys), q1 = host_copy(d_q1, NC * ys);
    // the reference: activation 0 as F32 and as its q8_1 values
    const std::vector<float> w = decode(h);
    std::vector<float> ref((size_t) 2 * n);
    ref_rows(w.data(), k, n, h.suh.data(), h.svh.data(), A.x.data(), A.xs, 1, ref.data(), (size_t) n);
    ref_rows(w.data(), k, n, h.suh.data(), h.svh.data(), A.deq.data(), A.xs, 1, ref.data() + n, (size_t) n);
    for (int i = 0; i < 2 * n; ++i) ref[(size_t) i] = alpha * ref[(size_t) i] + bias[(size_t) (i % n)];
    std::string line = "f32 " + check("mv " + tag + " f32", f1.data(), ref.data(), (size_t) n, 2e-3);
    line += "  q8_1 " + check("mv " + tag + " q8_1", q1.data(), ref.data() + n, (size_t) n, 2e-3);
    // several activations in one job
    size_t differ = 0, cmp = 0;
    for (int nc : {2, 8, 16})
        for (int q8 = 0; q8 < 2; ++q8) {
            CK(cudaMemset(d_n, 0xff, NC * ys * sizeof(float)));
            ex::MvJob j = job_of(&m, A, q8 != 0, 0, nc, d_n, ys, d_bias, alpha);
            refused |= !ex::mv(&j, 1, 0);
            CK(cudaDeviceSynchronize());
            const std::vector<float> got = host_copy(d_n, NC * ys);
            const std::vector<float>& one = q8 ? q1 : f1;
            for (int c = 0; c < nc; ++c) differ += bits_differ(got.data() + c * ys, one.data() + c * ys, (size_t) n);
            cmp += (size_t) nc * n;
        }
    line += "  x2/x8/x16 " + check_bits("mv " + tag + " several activations", differ, cmp);
    // a column view: [c0, c0 + nv) (its own input split, so against the reference), and 16 activations through it
    const int c0 = (n / 128) / 2 * 128, nv = std::min(256, n - c0);
    ex::Mat v = ex::view(m, c0, nv);
    {
        float* d_v1 = dev_alloc<float>(NC * ys);
        for (int c = 0; c < NC; ++c) {
            ex::MvJob j = job_of(&v, A, false, c, 1, d_v1 + c * ys, ys, d_bias + c0, alpha);
            refused |= !ex::mv(&j, 1, 0);
        }
        CK(cudaMemset(d_n, 0xff, NC * ys * sizeof(float)));
        ex::MvJob j = job_of(&v, A, false, 0, NC, d_n, ys, d_bias + c0, alpha);
        refused |= !ex::mv(&j, 1, 0);
        CK(cudaDeviceSynchronize());
        const std::vector<float> v1 = host_copy(d_v1, NC * ys), vn = host_copy(d_n, NC * ys);
        line += "  view@" + std::to_string(c0) + "+" + std::to_string(nv) + " " +
                check("mv " + tag + " view", v1.data(), ref.data() + c0, (size_t) nv, 2e-3);
        differ = 0;
        for (int c = 0; c < NC; ++c) differ += bits_differ(vn.data() + c * ys, v1.data() + c * ys, (size_t) nv);
        line += " x16 " + check_bits("mv " + tag + " view, 16 activations", differ, (size_t) NC * nv);
        dfree(d_v1);
    }
    if (refused) note_refusal("mv " + tag, false, "");
    std::printf("  %-44s %-12s %s\n", (std::string(sh.name).substr(0, 44)).c_str(), (dims(k, n) + " " + kcb(k2, cb)).c_str(),
                line.c_str());
    A.release();
    free_mat(m);
    for (float* p : {d_bias, d_f1, d_q1, d_n}) dfree(p);
}

void sec_mv() {
    std::printf("mv: the trunk's matrices (mv runs %s kernels here)\n", family());
    for (const Shape& sh : kTrunk)
        for (const KCb& c : kAll) mv_case(sh, c.k2, c.cb);
}

// ================================================================ batch: the engine's calls of several jobs
struct Job {
    std::string name;
    HostMat h;
    ex::Mat m;
    std::vector<float> w;   // decoded (for the reference)
    float* d_y = nullptr;
    size_t ys = 0;
};
void sec_batch() {
    std::printf("batch: several jobs in one mv call (a launch per K / codebook, the input split counted over the call)\n");
    // (a) the MLA layer's q_a and kv_a over one x, and the shared expert's gate and up over one x: each job its reference
    struct Pair {
        const char* what;
        int n0, n1, k2, cb;
    };
    for (const Pair& p : {Pair{"q_a + kv_a", 2048, 640, 8, 2}, Pair{"q_a + kv_a", 2048, 640, 5, 2},
                          Pair{"q_a + kv_a", 2048, 640, 6, 1}, Pair{"shexp gate + up", kFfExp, kFfExp, 6, 2},
                          Pair{"shexp gate + up", kFfExp, kFfExp, 4, 0}}) {
        Acts A;
        A.make(kEmbd, 1, 2.0f);
        Job J[2];
        ex::MvJob mj[2];
        for (int i = 0; i < 2; ++i) {
            J[i].h = make_mat(kEmbd, i == 0 ? p.n0 : p.n1, p.k2, p.cb);
            J[i].m = upload(J[i].h);
            J[i].ys = (size_t) J[i].h.n;
            J[i].d_y = dev_alloc<float>(J[i].ys);
            mj[i] = job_of(&J[i].m, A, false, 0, 1, J[i].d_y, J[i].ys, nullptr, 1.0f);
        }
        const bool ok = ex::mv(mj, 2, 0);
        CK(cudaDeviceSynchronize());
        std::string line;
        for (int i = 0; i < 2; ++i) {
            const std::vector<float> w = decode(J[i].h);
            std::vector<float> ref(J[i].ys);
            ref_rows(w.data(), kEmbd, J[i].h.n, J[i].h.suh.data(), J[i].h.svh.data(), A.x.data(), A.xs, 1, ref.data(),
                     J[i].ys);
            const std::vector<float> got = host_copy(J[i].d_y, J[i].ys);
            line += " " + check(std::string("batch ") + p.what + " job " + std::to_string(i), got.data(), ref.data(),
                                J[i].ys, 2e-3);
            free_mat(J[i].m);
            dfree(J[i].d_y);
        }
        if (!ok) note_refusal(std::string("batch ") + p.what, false, "");
        std::printf("  %-20s %s:%s\n", p.what, kcb(p.k2, p.cb).c_str(), line.c_str());
        A.release();
    }
    // (b) six bitrates in one call, three inputs (6144, 16384, 2048 wide)
    {
        struct Spec {
            const char* name;
            int k, n, k2, cb;
        };
        const Spec spec[6] = {{"q_a", kEmbd, 2048, 8, 2},          {"kv_a", kEmbd, 640, 10, 2},
                              {"indexer k", kEmbd, 128, 16, 1},    {"shexp gate", kEmbd, kFfExp, 5, 2},
                              {"attn_output", 16384, kEmbd, 6, 0}, {"q_b", 2048, 16384, 4, 2}};
        Acts A6, A16, A2;
        A6.make(kEmbd, 1, 2.0f);
        A16.make(16384, 1, 2.0f);
        A2.make(2048, 1, 2.0f);
        std::vector<Job> J(6);
        ex::MvJob mj[6];
        for (int i = 0; i < 6; ++i) {
            J[(size_t) i].name = spec[i].name;
            J[(size_t) i].h = make_mat(spec[i].k, spec[i].n, spec[i].k2, spec[i].cb);
            J[(size_t) i].m = upload(J[(size_t) i].h);
            J[(size_t) i].ys = (size_t) spec[i].n;
            J[(size_t) i].d_y = dev_alloc<float>(J[(size_t) i].ys);
            const Acts& a = spec[i].k == kEmbd ? A6 : spec[i].k == 16384 ? A16 : A2;
            mj[i] = job_of(&J[(size_t) i].m, a, false, 0, 1, J[(size_t) i].d_y, J[(size_t) i].ys, nullptr, 1.0f);
        }
        const bool ok = ex::mv(mj, 6, 0);
        CK(cudaDeviceSynchronize());
        std::string line;
        for (int i = 0; i < 6; ++i) {
            Job& j = J[(size_t) i];
            const Acts& a = spec[i].k == kEmbd ? A6 : spec[i].k == 16384 ? A16 : A2;
            const std::vector<float> w = decode(j.h);
            std::vector<float> ref(j.ys);
            ref_rows(w.data(), j.h.k, j.h.n, j.h.suh.data(), j.h.svh.data(), a.x.data(), a.xs, 1, ref.data(), j.ys);
            const std::vector<float> got = host_copy(j.d_y, j.ys);
            line += " " + j.name + " " + kcb(spec[i].k2, spec[i].cb) + " " +
                    check("batch mixed " + j.name, got.data(), ref.data(), j.ys, 2e-3) + ";";
            free_mat(j.m);
            dfree(j.d_y);
        }
        if (!ok) note_refusal("batch of six bitrates", false, "");
        std::printf("  six bitrates:%s\n", line.c_str());
        A6.release();
        A16.release();
        A2.release();
    }
    // (c) a verify window: q_a + kv_a for 16 tokens in one call (one job of 16 activations each) bit for bit the
    // token path (the same two jobs for one token, 16 calls) - the input split is the call's, never the activations'
    for (const KCb& c : {KCb{8, 2}, KCb{5, 2}, KCb{6, 0}}) {
        constexpr int T = 16;
        Acts A;
        A.make(kEmbd, T, 2.0f);
        HostMat ha = make_mat(kEmbd, 2048, c.k2, c.cb), hb = make_mat(kEmbd, 640, c.k2, c.cb);
        ex::Mat ma = upload(ha), mb = upload(hb);
        const size_t ya = 2048 + 64, yb = 640 + 32;
        float *d_ta = dev_alloc<float>(T * ya), *d_tb = dev_alloc<float>(T * yb);
        float *d_wa = dev_alloc<float>(T * ya), *d_wb = dev_alloc<float>(T * yb);
        bool ok = true;
        for (int t = 0; t < T; ++t) {
            ex::MvJob j[2] = {job_of(&ma, A, false, t, 1, d_ta + t * ya, ya, nullptr, 1.0f),
                              job_of(&mb, A, false, t, 1, d_tb + t * yb, yb, nullptr, 1.0f)};
            ok &= ex::mv(j, 2, 0);
        }
        ex::MvJob j[2] = {job_of(&ma, A, false, 0, T, d_wa, ya, nullptr, 1.0f),
                          job_of(&mb, A, false, 0, T, d_wb, yb, nullptr, 1.0f)};
        ok &= ex::mv(j, 2, 0);
        CK(cudaDeviceSynchronize());
        size_t differ = 0;
        {
            const std::vector<float> a = host_copy(d_ta, T * ya), b = host_copy(d_wa, T * ya);
            for (int t = 0; t < T; ++t) differ += bits_differ(a.data() + t * ya, b.data() + t * ya, 2048);
        }
        {
            const std::vector<float> a = host_copy(d_tb, T * yb), b = host_copy(d_wb, T * yb);
            for (int t = 0; t < T; ++t) differ += bits_differ(a.data() + t * yb, b.data() + t * yb, 640);
        }
        if (!ok) note_refusal("batch window q_a + kv_a " + kcb(c.k2, c.cb), false, "");
        std::printf("  window q_a + kv_a x16 %s: %s\n", kcb(c.k2, c.cb).c_str(),
                    check_bits("batch window q_a + kv_a " + kcb(c.k2, c.cb), differ, (size_t) T * (2048 + 640)).c_str());
        A.release();
        free_mat(ma);
        free_mat(mb);
        for (float* p : {d_ta, d_tb, d_wa, d_wb}) dfree(p);
    }
}

// ================================================================ head: 154880 x 6144
void head_case(int k2, int cb) {
    constexpr int NC = 16;
    const std::string tag = "head " + dims(kEmbd, kVocab) + " " + kcb(k2, cb);
    const double t0 = now_s();
    HostMat h = make_mat(kEmbd, kVocab, k2, cb);
    ex::Mat m = upload(h);
    std::printf("  %s: %.0f MB of trellis\n", tag.c_str(), (double) h.tr.size() * 4 / 1e6);
    Acts A;
    A.make(kEmbd, NC, 2.0f);
    // the column views: their own input split, against the reference; 16 activations bit for bit their singles; the
    // reconstruction of the view's rows (the prompt path's slices) against the reference's full weight
    for (const int c0 : {0, 98304 - 640}) {
        const int nv = 1280;
        ex::Mat v = ex::view(m, c0, nv);
        const size_t ys = (size_t) nv + 64;
        float *d_1 = dev_alloc<float>(NC * ys), *d_n = dev_alloc<float>(NC * ys), *d_q = dev_alloc<float>(ys);
        bool ok = true;
        for (int c = 0; c < NC; ++c) {
            ex::MvJob j = job_of(&v, A, false, c, 1, d_1 + c * ys, ys, nullptr, 1.0f);
            ok &= ex::mv(&j, 1, 0);
        }
        {
            ex::MvJob j = job_of(&v, A, true, 0, 1, d_q, ys, nullptr, 1.0f);
            ok &= ex::mv(&j, 1, 0);
            ex::MvJob jn = job_of(&v, A, false, 0, NC, d_n, ys, nullptr, 1.0f);
            ok &= ex::mv(&jn, 1, 0);
        }
        CK(cudaDeviceSynchronize());
        const std::vector<float> w = decode(h, c0, nv);
        std::vector<float> ref((size_t) 2 * nv);
        ref_rows(w.data(), kEmbd, nv, h.suh.data(), h.svh.data() + c0, A.x.data(), A.xs, 1, ref.data(), (size_t) nv);
        ref_rows(w.data(), kEmbd, nv, h.suh.data(), h.svh.data() + c0, A.deq.data(), A.xs, 1, ref.data() + nv,
                 (size_t) nv);
        const std::vector<float> g1 = host_copy(d_1, NC * ys), gn = host_copy(d_n, NC * ys), gq = host_copy(d_q, ys);
        const std::string vt = tag + " view@" + std::to_string(c0);
        std::string line = "f32 " + check(vt + " f32", g1.data(), ref.data(), (size_t) nv, 2e-3);
        line += "  q8_1 " + check(vt + " q8_1", gq.data(), ref.data() + nv, (size_t) nv, 2e-3);
        size_t differ = 0;
        for (int c = 0; c < NC; ++c) differ += bits_differ(gn.data() + c * ys, g1.data() + c * ys, (size_t) nv);
        line += "  x16 " + check_bits(vt + " 16 activations", differ, (size_t) NC * nv);
        // the reconstruction: W' exactly, and the rotated rows (F16)
        uint16_t* d_w = dev_alloc<uint16_t>((size_t) kEmbd * nv);
        ex::reconstruct_inner(v, d_w, 0);
        CK(cudaDeviceSynchronize());
        {
            const std::vector<uint16_t> in = host_copy(d_w, (size_t) kEmbd * nv);
            size_t bad = 0;
            for (size_t i = 0; i < in.size(); ++i) bad += f32_from_f16(in[i]) != w[i];
            line += "  inner " + check_bits(vt + " reconstruct_inner", bad, in.size());
        }
        ex::reconstruct_rows(v, d_w, false, 0);
        CK(cudaDeviceSynchronize());
        {
            const std::vector<uint16_t> rows = host_copy(d_w, (size_t) kEmbd * nv);
            std::vector<float> got(rows.size());
            for (size_t i = 0; i < rows.size(); ++i) got[i] = f32_from_f16(rows[i]);
            const std::vector<float> full = full_rows(w, kEmbd, nv, h.suh.data(), h.svh.data() + c0);
            line += "  rows f16 " + check(vt + " reconstruct_rows", got.data(), full.data(), full.size(), 2e-3);
        }
        if (!ok) note_refusal(vt, false, "");
        std::printf("  view [%d, %d) (mv in %d pieces): %s\n", c0, c0 + nv, ex::mv_splits(v), line.c_str());
        for (const void* p : {(const void*) d_1, (const void*) d_n, (const void*) d_q, (const void*) d_w}) dfree(p);
    }
    // the whole head: mv_cols() activations in one job bit for bit as many single jobs (the decode's logits)
    const int S = ex::mv_splits(m), C = ex::mv_cols(m);
    const size_t ys = (size_t) kVocab + 64;
    float *d_1 = dev_alloc<float>(NC * ys), *d_n = dev_alloc<float>(NC * ys);
    bool ok = true;
    for (int c = 0; c < NC; ++c) {
        ex::MvJob j = job_of(&m, A, false, c, 1, d_1 + c * ys, ys, nullptr, 1.0f);
        ok &= ex::mv(&j, 1, 0);
    }
    if (!ok) note_refusal(tag + " one activation", false, "");
    {
        ex::MvJob j = job_of(&m, A, false, 0, C, d_n, ys, nullptr, 1.0f);
        const bool okc = ex::mv(&j, 1, 0);
        CK(cudaDeviceSynchronize());
        if (!okc) note_refusal(tag + " " + std::to_string(C) + " activations (mv_cols)", false, "");
        const std::vector<float> a = host_copy(d_1, NC * ys), b = host_copy(d_n, NC * ys);
        size_t differ = 0;
        for (int c = 0; c < C; ++c) differ += bits_differ(b.data() + c * ys, a.data() + c * ys, (size_t) kVocab);
        std::printf("  whole head: mv in %d pieces, mv_cols %d: %d activations in one job %s\n", S, C, C,
                    check_bits(tag + " mv_cols activations", differ, (size_t) C * kVocab).c_str());
    }
    {
        // 16 activations in one job (the engine's verify windows send up to exl3::kMaxMvCols): v1.1.8's kernels take
        // them in runs of 4; exl3.cu's in runs of 8, whose input pieces' partials (pieces x 154880 x 8 floats) outgrow
        // its 8 MB scratch when mv_cols() < 8 - refused there, and the caller must cut runs of mv_cols()
        CK(cudaMemset(d_n, 0xff, NC * ys * sizeof(float)));
        ex::MvJob j = job_of(&m, A, false, 0, NC, d_n, ys, nullptr, 1.0f);
        const bool ok16 = ex::mv(&j, 1, 0);
        CK(cudaDeviceSynchronize());
        const bool expect_refuse = !g_legacy_mv && C < 8;
        if (!ok16) {
            note_refusal(tag + " 16 activations in one mv job", expect_refuse,
                         "exl3.cu's runs of 8 need " + std::to_string(S) + " pieces x 154880 x 8 floats of partials, "
                         "over its 8 MB scratch: the head's verify-window GEMV must cut runs of exl3::mv_cols() (" +
                             std::to_string(C) + ") as maya-ds4 does");
        } else {
            if (expect_refuse) std::printf("  (16 activations taken although mv_cols() is %d)\n", C);
            const std::vector<float> a = host_copy(d_1, NC * ys), b = host_copy(d_n, NC * ys);
            size_t differ = 0;
            for (int c = 0; c < NC; ++c) differ += bits_differ(b.data() + c * ys, a.data() + c * ys, (size_t) kVocab);
            std::printf("  whole head: 16 activations in one job %s\n",
                        check_bits(tag + " 16 activations", differ, (size_t) NC * kVocab).c_str());
        }
    }
    dfree(d_1);
    dfree(d_n);
    A.release();
    free_mat(m);
    std::printf("  (%s: %.1f s)\n", tag.c_str(), now_s() - t0);
}
void sec_head() {
    std::printf("head: output.weight 6144 -> 154880 (mv runs %s kernels here)\n", family());
    head_case(12, 2);   // K 6: exllamav3's usual head bits
    head_case(16, 1);   // K 8, another codebook
}

// ================================================================ the routed experts
size_t align256(size_t v) { return (v + 255) / 256 * 256; }
// an expert's blob as the engine's layout describes it: gate, up, down, each [trellis | suh | svh] at 256-byte offsets
ex::ExpertLayout layout_of(int E, int F, const int k2[3], int cb, size_t& bytes) {
    ex::ExpertLayout lay;
    lay.cb = cb;
    size_t at = 0;
    for (int r = 0; r < 3; ++r) {
        const int k = r < 2 ? E : F, n = r < 2 ? F : E;
        lay.k2[r] = k2[r];
        lay.trellis[r] = at;
        at += align256(ex::trellis_bytes_k2(k, n, k2[r]));
        lay.suh[r] = at;
        at += align256(2 * (size_t) k);
        lay.svh[r] = at;
        at += align256(2 * (size_t) n);
    }
    bytes = at;
    return lay;
}
// NX experts of E <-> F on the device: one allocation of NX blobs `stride` apart (a set as the engine stores it), or a
// mixed layer's classes (each expert its own K per part and its own allocation); their host matrices for the
// reference, decoded on first use
struct Experts {
    int E = 0, F = 0, NX = 0, cb = 2;
    bool mixed = false;
    std::vector<std::array<HostMat, 3>> p;
    std::vector<std::array<std::vector<float>, 3>> w;
    std::vector<ex::ExpertLayout> lay;
    std::vector<size_t> bytes;
    uint8_t* base = nullptr;               // (not mixed) every blob, `stride` apart
    size_t stride = 0;
    std::vector<uint8_t*> blob;            // each expert's blob
    ex::MixedLayouts mix;                  // (mixed) the classes
    void make(int E_, int F_, int NX_, const std::function<std::array<int, 3>(int)>& k2_of, int cb_, bool mixed_) {
        E = E_;
        F = F_;
        NX = NX_;
        cb = cb_;
        mixed = mixed_;
        p.resize((size_t) NX);
        w.assign((size_t) NX, {});
        lay.resize((size_t) NX);
        bytes.resize((size_t) NX);
        blob.assign((size_t) NX, nullptr);
        for (int e = 0; e < NX; ++e) {
            const std::array<int, 3> k2 = k2_of(e);
            lay[(size_t) e] = layout_of(E, F, k2.data(), cb, bytes[(size_t) e]);
            for (int r = 0; r < 3; ++r) p[(size_t) e][(size_t) r] = make_mat(r < 2 ? E : F, r < 2 ? F : E, k2[(size_t) r], cb);
        }
        if (!mixed) {
            stride = bytes[0];
            CK(cudaMalloc((void**) &base, stride * NX + 256));
        } else {
            mix.n = NX;
        }
        for (int e = 0; e < NX; ++e) {
            std::vector<uint8_t> b(bytes[(size_t) e], 0);
            for (int r = 0; r < 3; ++r) {
                const HostMat& m = p[(size_t) e][(size_t) r];
                std::memcpy(&b[lay[(size_t) e].trellis[r]], m.tr.data(), m.tr.size() * 4);
                std::memcpy(&b[lay[(size_t) e].suh[r]], m.suh.data(), m.suh.size() * 2);
                std::memcpy(&b[lay[(size_t) e].svh[r]], m.svh.data(), m.svh.size() * 2);
            }
            if (!mixed) {
                blob[(size_t) e] = base + (size_t) e * stride;
                CK(cudaMemcpy(blob[(size_t) e], b.data(), b.size(), cudaMemcpyHostToDevice));
            } else {
                blob[(size_t) e] = dev_copy(b);
                mix.lay[e] = lay[(size_t) e];
                mix.lo[e] = (unsigned long long) blob[(size_t) e];
                mix.hi[e] = mix.lo[e] + bytes[(size_t) e];
            }
        }
    }
    void release() {
        if (!mixed) dfree(base);
        else
            for (uint8_t* b : blob) dfree(b);
        base = nullptr;
        blob.clear();
    }
    const std::vector<float>& W(int e, int r) {
        std::vector<float>& v = w[(size_t) e][(size_t) r];
        if (v.empty()) v = decode(p[(size_t) e][(size_t) r]);
        return v;
    }
    // expert e over rows x (E floats each, ldx apart): out rows (E floats, ldo apart); h16: the swiglu rounded to FP16
    // as the prompt path's sets hold it (the token path keeps it F32)
    void ref(int e, const float* x, size_t ldx, int rows, bool h16, float limit, float* out, size_t ldo) {
        const std::array<HostMat, 3>& P = p[(size_t) e];
        std::vector<float> g((size_t) rows * F), u((size_t) rows * F), h((size_t) rows * F);
        ref_rows(W(e, 0).data(), E, F, P[0].suh.data(), P[0].svh.data(), x, ldx, rows, g.data(), (size_t) F);
        ref_rows(W(e, 1).data(), E, F, P[1].suh.data(), P[1].svh.data(), x, ldx, rows, u.data(), (size_t) F);
        for (size_t i = 0; i < h.size(); ++i) {
            const float v = swiglu_ref(g[i], u[i], limit);
            h[i] = h16 ? round16(v) : v;
        }
        ref_rows(W(e, 2).data(), F, E, P[2].suh.data(), P[2].svh.data(), h.data(), (size_t) F, rows, out, ldo);
    }
};
std::string limit_name(float limit) { return std::isinf(limit) ? "no clamp" : "clamp " + std::to_string((int) limit); }
// a mixed layer's classes: expert e's gate / up / down K (K 1-4: the mixed kernels' range)
std::array<int, 3> mixed_k2(int e) {
    static const int ks[8][3] = {{2, 2, 4}, {4, 2, 6}, {6, 8, 2}, {8, 8, 8}, {4, 4, 4}, {2, 6, 8}, {8, 4, 2}, {6, 6, 6}};
    return {ks[e % 8][0], ks[e % 8][1], ks[e % 8][2]};
}

// ---------------------------------------------------------------- moe: one token's routed experts
struct MoeCase {
    int gu, dn, cb;
    float limit;
    bool mixed;
};
void moe_case(const MoeCase& mc) {
    const int E = kEmbd, F = kFfExp, NX = 7;
    const std::string tag = mc.mixed ? "mixed K1-4 cb" + std::to_string(mc.cb)
                                     : "K" + kname(mc.gu) + (mc.dn != mc.gu ? "/" + kname(mc.dn) : "") + " cb" +
                                           std::to_string(mc.cb);
    Experts X;
    X.make(E, F, NX, [&](int e) { return mc.mixed ? mixed_k2(e) : std::array<int, 3>{mc.gu, mc.gu, mc.dn}; }, mc.cb,
           mc.mixed);
    // top-8: entry 3 skipped (an expert another part of the engine computes), the others on experts 0..6
    std::vector<unsigned long long> plan((size_t) kTopK);
    std::vector<int> ent_exp((size_t) kTopK, -1);
    for (int i = 0, e = 0; i < kTopK; ++i) {
        if (i == 3) continue;
        ent_exp[(size_t) i] = e;
        plan[(size_t) i] = (unsigned long long) X.blob[(size_t) e++];
    }
    std::uniform_real_distribution<float> uw(0.05f, 0.3f);
    std::vector<float> pw((size_t) kTopK);
    for (auto& v : pw) v = uw(g_rng);
    const std::vector<float> x = rand_x((size_t) E, expert_x_sd(E)), cpu_part = rand_x((size_t) E, 20.0f),
                             sh = rand_x((size_t) E, 20.0f);
    unsigned long long* d_plan = dev_copy(plan);
    float *d_pw = dev_copy(pw), *d_x = dev_copy(x), *d_cpu = dev_copy(cpu_part), *d_sh = dev_copy(sh);
    int* d_flag = dev_alloc<int>(1);
    float* d_out = dev_alloc<float>((size_t) E);
    ex::MoeArgs a;
    a.plan_ptr = d_plan;
    a.plan_w = d_pw;
    a.cpu_part = d_cpu;
    a.cpu_flag = d_flag;
    if (mc.mixed) a.mix = X.mix;
    else a.lay = X.lay[0];
    a.k = kTopK;
    a.n_embd = E;
    a.n_ff = F;
    a.limit = mc.limit;
    // the CPU lane's part on, off, on again (the kernels' counters must come back to zero between tokens)
    std::vector<float> got[3];
    for (int rep = 0; rep < 3; ++rep) {
        const int flag = rep == 1 ? 0 : 1;
        CK(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemset(d_out, 0xff, (size_t) E * sizeof(float)));
        ex::moe_gate_up(a, d_x, nullptr, 0);
        ex::moe_down(a, nullptr, d_sh, d_out, 0);
        CK(cudaDeviceSynchronize());
        got[rep] = host_copy(d_out, (size_t) E);
    }
    std::vector<float> ref((size_t) E, 0.0f), o((size_t) E);
    for (int i = 0; i < kTopK; ++i) {
        if (ent_exp[(size_t) i] < 0) continue;
        X.ref(ent_exp[(size_t) i], x.data(), (size_t) E, 1, false, mc.limit, o.data(), (size_t) E);
        for (int c = 0; c < E; ++c) ref[(size_t) c] += pw[(size_t) i] * o[(size_t) c];
    }
    std::vector<float> ref_on = ref, ref_off = ref;
    for (int c = 0; c < E; ++c) {
        ref_off[(size_t) c] += sh[(size_t) c];
        ref_on[(size_t) c] += cpu_part[(size_t) c] + sh[(size_t) c];
    }
    const std::string what = "moe " + tag + " " + limit_name(mc.limit);
    std::string line = "CPU part off " + check(what + " CPU part off", got[1].data(), ref_off.data(), (size_t) E, 3e-3);
    line += "  on " + check(what + " CPU part on", got[2].data(), ref_on.data(), (size_t) E, 3e-3);
    line += "  first token " + check_bits(what + " first token against the third", bits_differ(got[0].data(), got[2].data(), (size_t) E), (size_t) E);
    std::printf("  %-22s %-9s %s\n", tag.c_str(), limit_name(mc.limit).c_str(), line.c_str());
    X.release();
    for (const void* p : {(const void*) d_plan, (const void*) d_pw, (const void*) d_x, (const void*) d_cpu,
                          (const void*) d_sh, (const void*) d_flag, (const void*) d_out})
        dfree(p);
}
void sec_moe() {
    std::printf("moe: one token's routed experts (6144 <-> 2048, top-8, entry 3 skipped, + the CPU part + the shared expert)\n");
    const MoeCase cases[] = {
        {4, 4, 2, kNoClamp, false},  {5, 5, 2, 7.0f, false},  {6, 6, 2, kNoClamp, false}, {8, 8, 2, 7.0f, false},
        {10, 10, 2, kNoClamp, false}, {16, 16, 2, kNoClamp, false}, {6, 6, 0, 7.0f, false}, {8, 8, 1, kNoClamp, false},
        {5, 8, 2, kNoClamp, false},  {4, 6, 2, 7.0f, false},  {0, 0, 2, kNoClamp, true},   {0, 0, 0, 7.0f, true},
    };
    for (const MoeCase& c : cases) moe_case(c);
}

// ---------------------------------------------------------------- win: a verify window's routed experts
void win_case(int gu, int dn, int cb, int T, bool mixed, float limit, bool expect_refuse = false) {
    const int E = kEmbd, F = kFfExp, NX = 12;
    const std::string tag = (mixed ? "mixed K1-4 cb" + std::to_string(cb)
                                   : "K" + kname(gu) + (dn != gu ? "/" + kname(dn) : "") + " cb" + std::to_string(cb)) +
                            " " + std::to_string(T) + " tokens";
    Experts X;
    X.make(E, F, NX, [&](int e) { return mixed ? mixed_k2(e) : std::array<int, 3>{gu, gu, dn}; }, cb, mixed);
    // plans [T][8]: token t's entry i on expert (3 t + 5 i) % 12 (8 distinct a token), entry t % 8 skipped on every
    // third token
    std::vector<unsigned long long> plan((size_t) T * 8, 0ull);
    std::vector<float> pw((size_t) T * 8, 0.0f);
    std::uniform_real_distribution<float> uw(0.05f, 0.3f);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < kTopK; ++i) {
            pw[(size_t) t * 8 + i] = uw(g_rng);
            if (t % 3 == 0 && i == t % 8) continue;
            plan[(size_t) t * 8 + i] = (unsigned long long) X.blob[(size_t) ((3 * t + 5 * i) % NX)];
        }
    // the groups, as glmf::moe_win_group makes them (experts in order of first appearance, entries in order)
    std::vector<unsigned long long> gp;
    std::vector<std::vector<int>> members;
    for (int e = 0; e < T * 8; ++e) {
        if (plan[(size_t) e] == 0ull) continue;
        size_t g = 0;
        while (g < gp.size() && gp[g] != plan[(size_t) e]) ++g;
        if (g == gp.size()) {
            gp.push_back(plan[(size_t) e]);
            members.emplace_back();
        }
        members[g].push_back(e);
    }
    std::vector<int> gstart(1, 0), gent;
    for (auto& mm : members) {
        for (int e : mm) gent.push_back(e);
        gstart.push_back((int) gent.size());
    }
    std::vector<int> ng = {(int) gp.size()}, flag((size_t) T);
    for (int t = 0; t < T; ++t) flag[(size_t) t] = t % 2;
    const std::vector<float> x = rand_x((size_t) T * E, expert_x_sd(E)), cpu_part = rand_x((size_t) T * E, 20.0f),
                             sh = rand_x((size_t) T * E, 20.0f);
    auto* d_plan = dev_copy(plan);
    float *d_pw = dev_copy(pw), *d_x = dev_copy(x), *d_cpu = dev_copy(cpu_part), *d_sh = dev_copy(sh);
    int* d_flag = dev_copy(flag);
    auto* d_gp = dev_copy(gp);
    int *d_gs = dev_copy(gstart), *d_ge = dev_copy(gent), *d_ng = dev_copy(ng);
    float* d_h = dev_alloc<float>((size_t) T * 8 * F);
    float* d_part = dev_alloc<float>((size_t) T * 8 * E);
    float* d_o1 = dev_alloc<float>((size_t) T * E);
    float* d_on = dev_alloc<float>((size_t) T * E);
    ex::WinArgs w;
    w.grp_ptr = d_gp;
    w.grp_start = d_gs;
    w.grp_ent = d_ge;
    w.n_groups = d_ng;
    w.plan_ptr = d_plan;
    w.plan_w = d_pw;
    w.cpu_part = d_cpu;
    w.cpu_flag = d_flag;
    w.x = d_x;
    w.x_stride = (size_t) E;
    w.h = d_h;
    w.part = d_part;
    w.sh_out = d_sh;
    w.out = d_on;
    w.out_stride = (size_t) E;
    if (mixed) w.mix = X.mix;
    else w.lay = X.lay[0];
    w.T = T;
    w.k = kTopK;
    w.n_embd = E;
    w.n_ff = F;
    w.limit = limit;
    const bool ok = ex::win_experts(w, 0);
    CK(cudaDeviceSynchronize());
    if (!ok) {
        note_refusal("win_experts " + tag, expect_refuse,
                     "a window's up values take T x 8 x n_ff floats of exl3.cu's 128 x 4096 scratch: at n_ff 2048 "
                     "at most 32 tokens a call");
    } else {
        if (expect_refuse) std::printf("  (win_experts took %d tokens: its scratch grew)\n", T);
        for (int t = 0; t < T; ++t) {
            ex::MoeArgs a;
            a.plan_ptr = d_plan + t * 8;
            a.plan_w = d_pw + t * 8;
            a.cpu_part = d_cpu + (size_t) t * E;
            a.cpu_flag = d_flag + t;
            if (mixed) a.mix = X.mix;
            else a.lay = X.lay[0];
            a.k = kTopK;
            a.n_embd = E;
            a.n_ff = F;
            a.limit = limit;
            ex::moe_gate_up(a, d_x + (size_t) t * E, nullptr, 0);
            ex::moe_down(a, nullptr, d_sh + (size_t) t * E, d_o1 + (size_t) t * E, 0);
        }
        CK(cudaDeviceSynchronize());
        const std::vector<float> a = host_copy(d_o1, (size_t) T * E), b = host_copy(d_on, (size_t) T * E);
        std::printf("  %-34s %-9s %zu experts: %s\n", tag.c_str(), limit_name(limit).c_str(), gp.size(),
                    check_bits("win " + tag, bits_differ(a.data(), b.data(), a.size()), a.size()).c_str());
    }
    X.release();
    for (const void* p : {(const void*) d_plan, (const void*) d_pw, (const void*) d_x, (const void*) d_cpu,
                          (const void*) d_sh, (const void*) d_flag, (const void*) d_gp, (const void*) d_gs,
                          (const void*) d_ge, (const void*) d_ng, (const void*) d_h, (const void*) d_part,
                          (const void*) d_o1, (const void*) d_on})
        dfree(p);
}
void sec_win() {
    std::printf("win: a verify window's routed experts, each read once for its tokens, bit for bit the token path\n");
    win_case(4, 4, 2, 16, false, kNoClamp);
    win_case(5, 5, 2, 16, false, 7.0f);
    win_case(6, 6, 0, 16, false, kNoClamp);
    win_case(8, 8, 1, 16, false, 7.0f);
    win_case(10, 10, 2, 7, false, kNoClamp);
    win_case(16, 16, 2, 16, false, kNoClamp);
    win_case(5, 8, 2, 16, false, kNoClamp);
    win_case(8, 8, 2, 32, false, 7.0f);   // the scratch's most at n_ff 2048
    win_case(0, 0, 2, 16, true, kNoClamp);
    win_case(0, 0, 0, 7, true, 7.0f);
    win_case(8, 8, 2, 33, false, kNoClamp, true);
}

// ---------------------------------------------------------------- set: the prompt path's expert sets
void set_case(int gu, int dn, int cb, const std::vector<int>& counts, int row0, float limit, bool mixed) {
    const int E = kEmbd, F = kFfExp, NE = (int) counts.size(), T = 256;
    std::string cs;
    for (int c : counts) cs += (cs.empty() ? "" : "/") + std::to_string(c);
    const std::string tag = (mixed ? "mixed K1-4 cb" + std::to_string(cb)
                                   : "K" + kname(gu) + (dn != gu ? "/" + kname(dn) : "") + " cb" + std::to_string(cb)) +
                            " rows " + cs + (row0 ? " from row " + std::to_string(row0) : "");
    Experts X;
    X.make(E, F, NE, [&](int e) { return mixed ? mixed_k2(e) : std::array<int, 3>{gu, gu, dn}; }, cb, mixed);
    const std::vector<float> x = rand_x((size_t) T * E, expert_x_sd(E));
    // expert e's rows are tokens (7 e + j) % T
    std::vector<int> bounds = {row0}, row_tok;
    int max_rows = 0;
    for (int e = 0; e < NE; ++e) {
        for (int j = 0; j < counts[(size_t) e]; ++j) row_tok.push_back((e * 7 + j) % T);
        bounds.push_back(row0 + (int) row_tok.size());
        max_rows = std::max(max_rows, counts[(size_t) e]);
    }
    const int rows = (int) row_tok.size();
    std::vector<unsigned long long> eptr((size_t) NE);
    for (int e = 0; e < NE; ++e) eptr[(size_t) e] = (unsigned long long) X.blob[(size_t) e];
    float* d_x = dev_copy(x);
    int *d_bounds = dev_copy(bounds), *d_rows = dev_copy(row_tok);
    unsigned long long* d_eptr = dev_copy(eptr);
    // h and the output with 32 guard bytes each side
    const size_t hbytes = (size_t) rows * F * 2, ybytes = (size_t) rows * E * 4;
    uint16_t* h_st = nullptr;
    float* y_st = nullptr;
    CK(cudaMalloc((void**) &h_st, hbytes + 64));
    CK(cudaMalloc((void**) &y_st, ybytes + 64));
    CK(cudaMemset(h_st, 0xa5, hbytes + 64));
    CK(cudaMemset(y_st, 0xa5, ybytes + 64));
    uint16_t* d_h = h_st + 16;
    float* d_y = y_st + 8;
    ex::SetArgs sa;
    if (mixed) {
        sa.mix = X.mix;
        sa.eptr = d_eptr;
    } else {
        sa.wbase = X.base;
        sa.stride = X.stride;
        sa.lay = X.lay[0];
    }
    sa.n_exp = NE;
    sa.bounds = d_bounds;
    sa.row0 = row0;
    sa.max_rows = max_rows;
    sa.n_embd = E;
    sa.n_ff = F;
    sa.limit = limit;
    ex::set_gate_up(sa, d_x, E, d_rows, d_h, F, 0);
    ex::set_down(sa, d_h, F, d_y, E, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t) rows * E);
    for (int e = 0, r = 0; e < NE; ++e) {
        if (counts[(size_t) e] == 0) continue;
        std::vector<float> xr((size_t) counts[(size_t) e] * E);
        for (int j = 0; j < counts[(size_t) e]; ++j)
            std::memcpy(&xr[(size_t) j * E], &x[(size_t) row_tok[(size_t) (r + j)] * E], (size_t) E * sizeof(float));
        X.ref(e, xr.data(), (size_t) E, counts[(size_t) e], true, limit, ref.data() + (size_t) r * E, (size_t) E);
        r += counts[(size_t) e];
    }
    const std::vector<float> got = host_copy(d_y, ref.size());
    std::string line = check("set " + tag + " " + limit_name(limit), got.data(), ref.data(), ref.size(), 3e-3);
    // nothing written outside h's rows and the output's
    const std::vector<uint8_t> hg = host_copy((const uint8_t*) h_st, hbytes + 64), yg = host_copy((const uint8_t*) y_st, ybytes + 64);
    size_t bad = 0;
    for (int i = 0; i < 32; ++i) bad += (hg[(size_t) i] != 0xa5) + (hg[hbytes + 32 + i] != 0xa5) + (yg[(size_t) i] != 0xa5) +
                                        (yg[ybytes + 32 + i] != 0xa5);
    line += "  guards " + check_bits("set " + tag + " guard bytes", bad, 128);
    std::printf("  %-44s %-9s %s\n", tag.c_str(), limit_name(limit).c_str(), line.c_str());
    X.release();
    for (const void* p : {(const void*) d_x, (const void*) d_bounds, (const void*) d_rows, (const void*) d_eptr,
                          (const void*) h_st, (const void*) y_st})
        dfree(p);
}
void sec_set() {
#if defined(STRATA_USE_HIP)
    const char* kind = ex::prompt_wmma() ? "the matrix-core kernels (EXL3_WMMA=0: the portable ones)" : "the portable kernels";
#else
    const char* kind = ex::mma_supported() ? "the sm_80+ tensor-core kernels"
                       : ex::prompt_wmma() ? "sm_7x's WMMA kernels (STRATA_EXL3_WMMA=0: the portable ones)"
                                           : "the portable kernels";
#endif
    std::printf("set: the prompt path's expert sets (set_gate_up + set_down; %s)\n", kind);
    const std::vector<int> c4 = {37, 0, 70, 1}, c2 = {130, 9};
    set_case(4, 4, 2, c4, 0, kNoClamp, false);
    set_case(5, 5, 2, c4, 19, 7.0f, false);
    set_case(6, 6, 2, c4, 0, kNoClamp, false);
    set_case(8, 8, 1, c4, 0, 7.0f, false);
    set_case(10, 10, 0, c4, 0, kNoClamp, false);
    set_case(16, 16, 2, c4, 0, kNoClamp, false);
    set_case(5, 8, 2, c4, 0, 7.0f, false);
    set_case(4, 4, 2, c2, 0, kNoClamp, false);   // 128+ rows an expert (gfx11's 64-row shaped blocks for K 2 / 2.5)
    set_case(5, 5, 2, c2, 7, 7.0f, false);
    set_case(6, 6, 2, c2, 0, kNoClamp, false);
    set_case(0, 0, 2, {37, 5, 70, 1}, 0, kNoClamp, true);
    set_case(0, 0, 0, {37, 5, 70, 1}, 3, 7.0f, true);
}

// ================================================================ gemm_rows
void sec_gemm() {
#if defined(STRATA_USE_HIP)
    const char* kind = "the portable kernels";
#else
    const char* kind = ex::mma_supported() ? "the sm_80+ tensor-core kernels" : "the portable kernels";
#endif
    std::printf("gemm: gemm_rows over FP16 rows (%s)\n", kind);
    struct G {
        const char* name;
        int k, n, k2, cb, rows, c0, nc;   // nc > 0: a column view [c0, c0 + nc)
    };
    const G cases[] = {
        {"attn_q_a", kEmbd, 2048, 8, 2, 45, 0, 0},       {"attn_kv_a_mqa", kEmbd, 640, 5, 2, 45, 0, 0},
        {"attn_q_b", 2048, 16384, 6, 1, 7, 0, 0},        {"attn_output", 16384, kEmbd, 10, 0, 33, 0, 0},
        {"dense ffn_down / eh_proj", kFfDense, kEmbd, 16, 2, 45, 0, 0}, {"ffn_down_shexp", kFfExp, kEmbd, 4, 2, 100, 0, 0},
        {"dense ffn_gate, a view", kEmbd, kFfDense, 6, 2, 37, 4096, 1024},
    };
    for (const G& g : cases) {
        HostMat h = make_mat(g.k, g.n, g.k2, g.cb);
        ex::Mat m = upload(h);
        const int c0 = g.nc > 0 ? g.c0 : 0, nc = g.nc > 0 ? g.nc : g.n;
        ex::Mat v = g.nc > 0 ? ex::view(m, c0, nc) : m;
        const size_t ldx = (size_t) g.k + 64, ldy = (size_t) nc + 32;
        std::vector<uint16_t> x16((size_t) g.rows * ldx, 0);
        std::vector<float> xr((size_t) g.rows * ldx, 0.0f);
        const std::vector<float> xs = rand_x((size_t) g.rows * g.k, 2.0f);
        for (int r = 0; r < g.rows; ++r)
            for (int i = 0; i < g.k; ++i) {
                x16[(size_t) r * ldx + i] = f16_from_f32(xs[(size_t) r * g.k + i]);
                xr[(size_t) r * ldx + i] = f32_from_f16(x16[(size_t) r * ldx + i]);
            }
        const std::vector<float> y0 = rand_x((size_t) g.rows * ldy, 5.0f);
        const std::vector<float> w = decode(h, c0, nc);
        std::vector<float> ref((size_t) g.rows * ldy, 0.0f);
        ref_rows(w.data(), g.k, nc, h.suh.data(), h.svh.data() + c0, xr.data(), ldx, g.rows, ref.data(), ldy);
        uint16_t* d_x = dev_copy(x16);
        float* d_y = dev_copy(y0);
        std::string line;
        for (int beta = 0; beta < 2; ++beta) {
            CK(cudaMemcpy(d_y, y0.data(), y0.size() * sizeof(float), cudaMemcpyHostToDevice));
            ex::gemm_rows(v, d_x, (int) ldx, g.rows, d_y, (int) ldy, (float) beta, 0);
            CK(cudaDeviceSynchronize());
            const std::vector<float> got = host_copy(d_y, y0.size());
            std::vector<float> gc((size_t) g.rows * nc), rc((size_t) g.rows * nc);
            for (int r = 0; r < g.rows; ++r)
                for (int c = 0; c < nc; ++c) {
                    gc[(size_t) r * nc + c] = got[(size_t) r * ldy + c];
                    rc[(size_t) r * nc + c] = ref[(size_t) r * ldy + c] + (beta ? y0[(size_t) r * ldy + c] : 0.0f);
                }
            line += "  beta " + std::to_string(beta) + " " +
                    check(std::string("gemm_rows ") + g.name + " " + kcb(g.k2, g.cb) + " beta " + std::to_string(beta),
                          gc.data(), rc.data(), gc.size(), 3e-3);
        }
        std::printf("  %-26s %-12s %3d rows%s\n", g.name,
                    (dims(g.k, nc) + " " + kcb(g.k2, g.cb)).c_str(), g.rows, line.c_str());
        free_mat(m);
        dfree(d_x);
        dfree(d_y);
    }
}

// ================================================================ the reconstructions
void sec_recon() {
    std::printf("recon: W' exactly (reconstruct_inner, reconstruct_inner_rows) and the rotated rows (reconstruct_rows) on views\n");
    struct R {
        const char* name;
        int k, n, k2, cb, c0, nc;
    };
    const R cases[] = {
        {"attn_kv_a_mqa, whole", kEmbd, 640, 5, 2, 0, 640},
        {"attn_output", 16384, kEmbd, 8, 2, 2048, 1024},
        {"attn_q_b", 2048, 16384, 6, 0, 8192, 1024},
        {"dense ffn_down / eh_proj", kFfDense, kEmbd, 10, 1, 0, 1024},
        {"dense ffn_gate", kEmbd, kFfDense, 16, 2, 11264, 1024},
        {"ffn_gate_shexp", kEmbd, kFfExp, 4, 2, 0, kFfExp},
    };
    for (const R& c : cases) {
        HostMat h = make_mat(c.k, c.n, c.k2, c.cb);
        ex::Mat m = upload(h);
        ex::Mat v = ex::view(m, c.c0, c.nc);
        const std::vector<float> w = decode(h, c.c0, c.nc);
        const size_t N = (size_t) c.k * c.nc;
        uint16_t* d_w = dev_alloc<uint16_t>(N);
        std::string line;
        ex::reconstruct_inner(v, d_w, 0);
        CK(cudaDeviceSynchronize());
        {
            const std::vector<uint16_t> in = host_copy(d_w, N);
            size_t bad = 0;
            for (size_t i = 0; i < N; ++i) bad += f32_from_f16(in[i]) != w[i];
            line += "inner " + check_bits(std::string("reconstruct_inner ") + c.name, bad, N);
        }
        ex::reconstruct_inner_rows(v, d_w, 0);
        CK(cudaDeviceSynchronize());
        {
            const std::vector<uint16_t> in = host_copy(d_w, N);   // [nc][k]
            size_t bad = 0;
            for (int col = 0; col < c.nc; ++col)
                for (int r = 0; r < c.k; ++r)
                    bad += f32_from_f16(in[(size_t) col * c.k + r]) != w[(size_t) r * c.nc + col];
            line += "  inner rows " + check_bits(std::string("reconstruct_inner_rows ") + c.name, bad, N);
        }
        const std::vector<float> full = full_rows(w, c.k, c.nc, h.suh.data(), h.svh.data() + c.c0);
        for (int bf = 0; bf < 2; ++bf) {
            ex::reconstruct_rows(v, d_w, bf == 1, 0);
            CK(cudaDeviceSynchronize());
            const std::vector<uint16_t> rows = host_copy(d_w, N);
            std::vector<float> got(N);
            for (size_t i = 0; i < N; ++i) {
                if (bf) {
                    const uint32_t u = (uint32_t) rows[i] << 16;
                    std::memcpy(&got[i], &u, 4);
                } else {
                    got[i] = f32_from_f16(rows[i]);
                }
            }
            line += std::string(bf ? "  rows bf16 " : "  rows f16 ") +
                    check(std::string("reconstruct_rows ") + (bf ? "bf16 " : "f16 ") + c.name, got.data(), full.data(), N,
                          bf ? 6e-3 : 2e-3);
        }
        std::printf("  %-26s %-12s [%d, %d): %s\n", c.name, (dims(c.k, c.n) + " " + kcb(c.k2, c.cb)).c_str(), c.c0,
                    c.c0 + c.nc, line.c_str());
        dfree(d_w);
        free_mat(m);
    }
}

// ================================================================ blas: the prompt path's cuBLAS routes
// one heavy expert as glm_prefill.cu's x3_run_blas computes it, its nrows rows (tokens row_tok[0..nrows) of x) into
// the expert's own OUTP rows (nrows x E from `out`).  Gate / up go 2 n_ff wide at the END of those rows (G2): the down's
// outputs (n_embd wide, from the start) then never reach a row it has not read - with 2 n_ff == n_embd (GLM-5.3-Flash)
// G2 is the rows themselves and every call is the one x3_run_blas makes today.  old_layout: x3_run_blas's layout as it
// is (gate / up written n_embd apart, read 2 n_ff apart by swiglu_rows), right only when 2 n_ff == n_embd
void blas_expert(cublasHandle_t blas, const ex::Mat mat[3], const float* x, const int* row_tok, int nrows, float* out,
                 uint16_t* w16, int64_t w16_elems, int E, int nff, float limit, bool rot_rows, bool old_layout,
                 cudaStream_t s) {
    const float one = 1.0f, zero = 0.0f;
    const int64_t wgu = (int64_t) 2 * nff * E, wdn = (int64_t) E * nff;
    const int gu_rows = (int) std::max<int64_t>(1, (w16_elems - wgu) / (rot_rows ? 2 * E : E));
    const int dn_rows = (int) std::max<int64_t>(1, (w16_elems - wdn) / nff);
    float* GU = out;
    const int ldg = old_layout ? E : 2 * nff;
    float* G2 = old_layout ? GU : GU + (size_t) nrows * (size_t) (E - 2 * nff);
    if (rot_rows) {
        for (int r = 0; r < 2; ++r) ex::reconstruct_inner_rows(mat[r], w16 + (size_t) r * nff * E, s);
        uint16_t* X16 = w16 + wgu;
        for (int c0 = 0; c0 < nrows; c0 += gu_rows) {
            const int n = std::min(gu_rows, nrows - c0);
            for (int r = 0; r < 2; ++r) {
                uint16_t* Xr = X16 + (size_t) r * gu_rows * E;
                ex::rows_rot_f16(x, E, row_tok + c0, n, E, mat[r].suh, Xr, s);
                CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, nff, n, E, &one, w16 + (size_t) r * nff * E,
                                CUDA_R_16F, E, Xr, CUDA_R_16F, E, &zero, G2 + (size_t) c0 * ldg + (size_t) r * nff,
                                CUDA_R_32F, ldg, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
        for (int r = 0; r < 2; ++r) ex::rows_rot_out(G2 + (size_t) r * nff, ldg, nrows, nff, mat[r].svh, s);
        gb::swiglu_rows(G2, G2, nrows, nff, limit, s, 2 * nff);
        ex::reconstruct_inner_rows(mat[2], w16, s);
        uint16_t* H16 = w16 + wdn;
        for (int c0 = 0; c0 < nrows; c0 += dn_rows) {
            const int n = std::min(dn_rows, nrows - c0);
            // (the old layout reads h n_embd apart where swiglu_rows wrote it 2 n_ff apart: x3_run_blas as it is)
            ex::rows_rot_f16(G2 + (size_t) c0 * ldg, ldg, nullptr, n, nff, mat[2].suh, H16, s);
            CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, E, n, nff, &one, w16, CUDA_R_16F, nff, H16, CUDA_R_16F, nff,
                            &zero, GU + (size_t) c0 * E, CUDA_R_32F, E, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        ex::rows_rot_out(GU, E, nrows, E, mat[2].svh, s);
        return;
    }
    ex::reconstruct_rows(mat[0], w16, false, s);
    ex::reconstruct_rows(mat[1], w16 + (size_t) nff * E, false, s);
    uint16_t* X16 = w16 + wgu;
    for (int c0 = 0; c0 < nrows; c0 += gu_rows) {
        const int n = std::min(gu_rows, nrows - c0);
        ex::rows_f16(x, E, row_tok + c0, n, E, X16, s);
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * nff, n, E, &one, w16, CUDA_R_16F, E, X16, CUDA_R_16F, E,
                        &zero, G2 + (size_t) c0 * ldg, CUDA_R_32F, ldg, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    gb::swiglu_rows(G2, G2, nrows, nff, limit, s, 2 * nff);
    ex::reconstruct_rows(mat[2], w16, false, s);
    uint16_t* H16 = w16 + wdn;
    for (int c0 = 0; c0 < nrows; c0 += dn_rows) {
        const int n = std::min(dn_rows, nrows - c0);
        ex::rows_f16(G2 + (size_t) c0 * ldg, ldg, nullptr, n, nff, H16, s);
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, E, n, nff, &one, w16, CUDA_R_16F, nff, H16, CUDA_R_16F, nff,
                        &zero, GU + (size_t) c0 * E, CUDA_R_32F, E, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}
// returns the relative error (for the old layout's informational line)
double blas_expert_case(cublasHandle_t blas, int E, int F, int gu, int dn, int cb, bool rot, int nrows, int gu_chunk,
                        float limit, bool old_layout) {
    const int T = nrows + 51, o = 13, W = nrows + 2 * o;   // the window's rows: the expert's at [o, o + nrows)
    const std::string tag = dims(E, F) + " K" + kname(gu) + (dn != gu ? "/" + kname(dn) : "") + " cb" +
                            std::to_string(cb) + " ROT=" + (rot ? "1" : "0") + " " + std::to_string(nrows) + " rows" +
                            (gu_chunk > 0 ? " in runs of " + std::to_string(gu_chunk) : "") +
                            (old_layout ? " (x3_run_blas's layout today)" : "");
    Experts X;
    X.make(E, F, 1, [&](int) { return std::array<int, 3>{gu, gu, dn}; }, cb, false);
    ex::Mat mat[3];
    for (int r = 0; r < 3; ++r) {
        mat[r].trellis = (const uint32_t*) (X.blob[0] + X.lay[0].trellis[r]);
        mat[r].suh = (const uint16_t*) (X.blob[0] + X.lay[0].suh[r]);
        mat[r].svh = (const uint16_t*) (X.blob[0] + X.lay[0].svh[r]);
        mat[r].k = r < 2 ? E : F;
        mat[r].n = r < 2 ? F : E;
        mat[r].k2 = X.lay[0].k2[r];
        mat[r].cb = cb;
    }
    const std::vector<float> x = rand_x((size_t) T * E, expert_x_sd(E));
    std::vector<int> row_tok((size_t) nrows);
    for (int i = 0; i < nrows; ++i) row_tok[(size_t) i] = (i * 7 + 3) % T;   // distinct (7 and T coprime)
    const std::vector<float> guard = rand_x((size_t) W * E, 1.0f);
    float* d_x = dev_copy(x);
    int* d_rt = dev_copy(row_tok);
    float* d_out = dev_copy(guard);
    const int64_t wgu = (int64_t) 2 * F * E;
    const int64_t w16_elems = gu_chunk > 0 ? wgu + (int64_t) gu_chunk * (rot ? 2 * E : E)
                                           : wgu + (int64_t) nrows * (rot ? 2 * E : E);
    uint16_t* d_w16 = dev_alloc<uint16_t>((size_t) w16_elems);
    blas_expert(blas, mat, d_x, d_rt, nrows, d_out + (size_t) o * E, d_w16, w16_elems, E, F, limit, rot, old_layout, 0);
    CK(cudaDeviceSynchronize());
    const std::vector<float> got = host_copy(d_out, (size_t) W * E);
    std::vector<float> xr((size_t) nrows * E), ref((size_t) nrows * E);
    for (int i = 0; i < nrows; ++i)
        std::memcpy(&xr[(size_t) i * E], &x[(size_t) row_tok[(size_t) i] * E], (size_t) E * sizeof(float));
    X.ref(0, xr.data(), (size_t) E, nrows, false, limit, ref.data(), (size_t) E);
    const Err e = err_of(got.data() + (size_t) o * E, ref.data(), ref.size());
    if (old_layout) {
        std::printf("  info: %-62s rel %.2e (not counted)\n", tag.c_str(), e.rel);
    } else {
        std::string line = check("blas expert " + tag, got.data() + (size_t) o * E, ref.data(), ref.size(), rot ? 2e-3 : 3e-3);
        // the window's other rows untouched (gate / up live inside the expert's own rows)
        const size_t before = bits_differ(got.data(), guard.data(), (size_t) o * E);
        const size_t after = bits_differ(got.data() + (size_t) (o + nrows) * E, guard.data() + (size_t) (o + nrows) * E,
                                         (size_t) (W - o - nrows) * E);
        line += "  other rows " + check_bits("blas expert " + tag + " rows outside the expert's", before + after,
                                             (size_t) (W - nrows) * E);
        std::printf("  %-62s %s\n", tag.c_str(), line.c_str());
    }
    X.release();
    for (const void* p : {(const void*) d_x, (const void*) d_rt, (const void*) d_out, (const void*) d_w16}) dfree(p);
    return e.rel;
}
// a dense EXL3 weight as glm_prefill.cu's hgemm_q takes it: 128-row slices of its outputs reconstructed (views) into
// the FP16 scratch, one GEMM a slice over the chunk's FP16 rows, beta 0 then 1
void blas_dense_case(cublasHandle_t blas, const char* name, int Kd, int N, int k2, int cb, int Tn, int64_t w16_elems) {
    HostMat h = make_mat(Kd, N, k2, cb);
    ex::Mat m = upload(h);
    const int ldx = Kd + 64, ldy = N + 32;
    int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, w16_elems / Kd));
    rows = std::max(128, rows / 128 * 128);
    std::vector<uint16_t> x16((size_t) Tn * ldx, 0);
    std::vector<float> xr((size_t) Tn * ldx, 0.0f);
    const std::vector<float> xs = rand_x((size_t) Tn * Kd, 2.0f);
    for (int t = 0; t < Tn; ++t)
        for (int i = 0; i < Kd; ++i) {
            x16[(size_t) t * ldx + i] = f16_from_f32(xs[(size_t) t * Kd + i]);
            xr[(size_t) t * ldx + i] = f32_from_f16(x16[(size_t) t * ldx + i]);
        }
    const std::vector<float> y0 = rand_x((size_t) Tn * ldy, 5.0f);
    const std::vector<float> w = decode(h);
    std::vector<float> ref((size_t) Tn * ldy, 0.0f);
    ref_rows(w.data(), Kd, N, h.suh.data(), h.svh.data(), xr.data(), (size_t) ldx, Tn, ref.data(), (size_t) ldy);
    uint16_t* d_x = dev_copy(x16);
    float* d_y = dev_copy(y0);
    uint16_t* d_w16 = dev_alloc<uint16_t>((size_t) std::max<int64_t>(w16_elems, (int64_t) rows * Kd));
    std::string line;
    int slices = 0;
    for (int beta_i = 0; beta_i < 2; ++beta_i) {
        const float one = 1.0f, beta = (float) beta_i;
        CK(cudaMemcpy(d_y, y0.data(), y0.size() * sizeof(float), cudaMemcpyHostToDevice));
        slices = 0;
        for (int r0 = 0; r0 < N; r0 += rows, ++slices) {
            const int n = std::min(rows, N - r0);
            ex::reconstruct_rows(ex::view(m, r0, n), d_w16, false, 0);
            CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, Tn, Kd, &one, d_w16, CUDA_R_16F, Kd, d_x, CUDA_R_16F,
                            ldx, &beta, d_y + r0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CK(cudaDeviceSynchronize());
        const std::vector<float> got = host_copy(d_y, y0.size());
        std::vector<float> gc((size_t) Tn * N), rc((size_t) Tn * N);
        for (int t = 0; t < Tn; ++t)
            for (int c = 0; c < N; ++c) {
                gc[(size_t) t * N + c] = got[(size_t) t * ldy + c];
                rc[(size_t) t * N + c] = ref[(size_t) t * ldy + c] + (beta_i ? y0[(size_t) t * ldy + c] : 0.0f);
            }
        line += "  beta " + std::to_string(beta_i) + " " +
                check(std::string("blas dense ") + name + " beta " + std::to_string(beta_i), gc.data(), rc.data(),
                      gc.size(), 3e-3);
    }
    std::printf("  dense %-26s %-12s %d tokens, %d slices of %d rows:%s\n", name,
                (dims(Kd, N) + " " + kcb(k2, cb)).c_str(), Tn, slices, rows, line.c_str());
    free_mat(m);
    for (const void* p : {(const void*) d_x, (const void*) d_y, (const void*) d_w16}) dfree(p);
}
void sec_blas(cublasHandle_t blas, bool old_layout) {
    std::printf("blas: the prompt path's cuBLAS routes, rebuilt (a heavy expert: x3_run_blas; the dense slices: hgemm_q)\n");
    // GLM-5.3 (n_embd 6144, n_ff 2048): gate / up in runs of 37 rows (the engine's w16 scratch cut small) and in one
    for (int rot = 1; rot >= 0; --rot) {
        blas_expert_case(blas, kEmbd, kFfExp, 8, 8, 2, rot != 0, 150, 37, kNoClamp, false);
        blas_expert_case(blas, kEmbd, kFfExp, 5, 6, 2, rot != 0, 150, 0, 7.0f, false);
        blas_expert_case(blas, kEmbd, kFfExp, 6, 6, 1, rot != 0, 97, 40, kNoClamp, false);
        blas_expert_case(blas, kEmbd, kFfExp, 4, 4, 0, rot != 0, 65, 0, kNoClamp, false);
        // GLM-5.3-Flash's widths (4096 = 2 x 2048): the packed rows are the expert's rows themselves
        blas_expert_case(blas, 4096, 2048, 8, 8, 2, rot != 0, 70, 30, kNoClamp, false);
    }
    if (old_layout)
        for (int rot = 1; rot >= 0; --rot) {
            blas_expert_case(blas, kEmbd, kFfExp, 8, 8, 2, rot != 0, 150, 37, kNoClamp, true);
            blas_expert_case(blas, 4096, 2048, 8, 8, 2, rot != 0, 70, 30, kNoClamp, true);
        }
    blas_dense_case(blas, "attn_output", 16384, kEmbd, 8, 2, 37, (int64_t) 16384 * 1024);
    blas_dense_case(blas, "attn_q_b", 2048, 16384, 6, 1, 37, (int64_t) 2048 * 3000);
    blas_dense_case(blas, "attn_kv_a_mqa", kEmbd, 640, 5, 2, 37, (int64_t) kEmbd * 300);
    blas_dense_case(blas, "dense ffn_down / eh_proj", kFfDense, kEmbd, 10, 0, 37, (int64_t) kFfDense * 2048);
}

// ================================================================ cpu: the CPU lane's band kernels
const char* tier_name(int tier) {
    return tier == 4 ? "avx2 16-bit" : tier == 3 ? "avx512vnni" : tier == 2 ? "avx512" : tier == 1 ? "avx2" : "scalar";
}
cpu::Exl3Part part_of(const HostMat& h) {
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = h.k;
    m.n = h.n;
    m.k2 = h.k2;
    return m;
}
// exl3_test's exact model of the vector tiers' arithmetic: column c's state at row r read bit by bit from the tile's
// MSB-first ring (the window that ends at bit floor((p + 1) k2 / 2) of its 128 k2 bits, p the tile's step there), its
// mul1 byte sum s, one fma(s, xh, acc) a weight in (slice, row) order
int g_step[16][16];
void init_steps() {
    for (int p = 0; p < 256; ++p) g_step[(p / 8 % 4) * 2 + (p & 1) + 8 * (p / 2 & 1)][p / 32 + 8 * (p / 4 & 1)] = p;
}
uint32_t model_state(const uint32_t* tile, int k2, int p) {
    const int bits = 128 * k2, end = ((p + 1) * k2) / 2 + bits;
    uint32_t st = 0;
    for (int q = end - 16; q < end; ++q) {
        const int b = q % bits;
        st = (st << 1) | ((tile[b >> 5] >> (31 - (b & 31))) & 1u);
    }
    return st;
}
float exact_band_col(const cpu::Exl3Part& m, const float* xh, int col, int ks0, int ks1) {
    const int t = col / 16, c = col % 16;
    float a = 0.0f;
    for (int ks = ks0; ks < ks1; ++ks) {
        const uint32_t* tile = m.trellis + ((size_t) ks * (m.n / 16) + t) * 4 * m.k2;
        for (int r = 0; r < 16; ++r) {
            const uint32_t x = model_state(tile, m.k2, g_step[r][c]) * 0x83DCD12Du;
            const int s = (int) ((x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24));
            a = std::fma((float) s, xh[ks * 16 + r], a);
        }
    }
    return a;
}
// every tier this CPU runs (and the AVX2 tier with its byte shuffles) against the model on two bands (the first and
// the last), the whole slice range and a partial one, 1..11 activations at once: the vector tiers bit for bit, the
// scalar one (no FMA) within rounding
void cpu_exact(const HostMat& h) {
    const cpu::Exl3Part m = part_of(h);
    std::vector<std::vector<float>> xh;
    const float* xp[kCpuNT];
    for (int i = 0; i < kCpuNT; ++i) xh.push_back(rand_x((size_t) h.k, 1.0f));
    for (int i = 0; i < kCpuNT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<float> want((size_t) kCpuNT * 128), got((size_t) kCpuNT * 128);
    float* gp[kCpuNT];
    for (int i = 0; i < kCpuNT; ++i) gp[i] = got.data() + (size_t) i * 128;
    const int S = h.k / 16;
    const int ranges[2][2] = {{0, S}, {3, S - 1}};
    size_t diff[6] = {}, cmp[6] = {};   // tiers 0..4, 5: tier 1 with the byte shuffles
    double se = 0.0, sr = 0.0;
    for (const int c0 : {0, h.n - 128})
        for (const auto& rg : ranges) {
            parallel_for((int64_t) kCpuNT * 128, [&](int64_t u0, int64_t u1) {
                for (int64_t u = u0; u < u1; ++u)
                    want[(size_t) u] = exact_band_col(m, xp[u / 128], c0 + (int) (u % 128), rg[0], rg[1]);
            });
            for (int tier = 0; tier <= 5; ++tier) {
                const int t = tier == 5 ? 1 : tier;
#if defined(GLM_DSA_EXL3_X86)
                if (tier == 5) cpu::exl3_cpu_set_avx2_shuf(true);
#else
                if (tier == 5) continue;
#endif
                for (int nt : {1, 2, 3, 5, 8, 11}) {
                    if (!cpu::exl3_cpu_band_acc_on(t, m, xp, nt, c0, rg[0], rg[1], gp)) break;
                    for (size_t i = 0; i < (size_t) nt * 128; ++i) {
                        ++cmp[tier];
                        diff[tier] += std::memcmp(&got[i], &want[i], sizeof(float)) != 0;
                        if (tier == 0) {
                            se += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
                            sr += (double) want[i] * want[i];
                        }
                    }
                }
#if defined(GLM_DSA_EXL3_X86)
                if (tier == 5) cpu::exl3_cpu_set_avx2_shuf(false);
#endif
            }
        }
    std::string line;
    for (int tier = 1; tier <= 5; ++tier) {
        if (cmp[tier] == 0) continue;
        const std::string name = tier == 5 ? "avx2 shuffles" : tier_name(tier);
        line += "  " + name + " " +
                check_bits("cpu exact " + name + " " + dims(h.k, h.n) + " K" + kname(h.k2), diff[tier], cmp[tier]);
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    ++g_checks;
    const bool ok = cmp[0] > 0 && rel <= 1e-5;
    if (!ok) {
        ++g_fail;
        std::printf("  FAIL cpu exact scalar %s K%s: rel %.2e\n", dims(h.k, h.n).c_str(), kname(h.k2).c_str(), rel);
    }
    char b[48];
    std::snprintf(b, sizeof b, "  scalar %.1e%s", rel, ok ? "" : " FAIL");
    std::printf("  exact model %-12s K%-3s%s%s\n", dims(h.k, h.n).c_str(), kname(h.k2).c_str(), line.c_str(), b);
}
// the lane over the whole matrix (the prepared input, every band) against the reference
std::string cpu_ref(const HostMat& h) {
    const cpu::Exl3Part m = part_of(h);
    const std::vector<float> x = rand_x((size_t) h.k, expert_x_sd(h.k) * 40.0f);
    std::vector<float> xh((size_t) h.k), got((size_t) h.n), ref((size_t) h.n);
    const float xs = cpu::exl3_cpu_prepare(m, x.data(), xh.data());
    parallel_for(h.n / 128, [&](int64_t b0, int64_t b1) {
        for (int64_t b = b0; b < b1; ++b) cpu::exl3_cpu_band(m, xh.data(), xs, (int) b * 128, got.data() + b * 128);
    });
    const std::vector<float> w = decode(h);
    ref_rows(w.data(), h.k, h.n, h.suh.data(), h.svh.data(), x.data(), (size_t) h.k, 1, ref.data(), (size_t) h.n);
    return check("cpu lane " + dims(h.k, h.n) + " K" + kname(h.k2), got.data(), ref.data(), (size_t) h.n, 2e-3);
}
// n activations at once (band_acc_n) against one call each, every band: the same bits
std::string cpu_n(const HostMat& h) {
    const cpu::Exl3Part m = part_of(h);
    std::vector<std::vector<float>> xh;
    const float* xp[kCpuNT];
    for (int i = 0; i < kCpuNT; ++i) xh.push_back(rand_x((size_t) h.k, 1.0f));
    for (int i = 0; i < kCpuNT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<size_t> diffs((size_t) (h.n / 128), 0);
    parallel_for(h.n / 128, [&](int64_t b0, int64_t b1) {
        std::vector<float> one((size_t) kCpuNT * 128), all((size_t) kCpuNT * 128);
        float* ap[kCpuNT];
        for (int i = 0; i < kCpuNT; ++i) ap[i] = all.data() + (size_t) i * 128;
        for (int64_t b = b0; b < b1; ++b)
            for (int nt : {2, 3, 8, 11}) {
                const int c0 = (int) b * 128, ks0 = b == 0 ? 0 : 3, ks1 = h.k / 16 - (nt % 2);
                for (int i = 0; i < nt; ++i) cpu::exl3_cpu_band_acc(m, xp[i], c0, ks0, ks1, one.data() + (size_t) i * 128);
                cpu::exl3_cpu_band_acc_n(m, xp, nt, c0, ks0, ks1, ap);
                diffs[(size_t) b] += bits_differ(one.data(), all.data(), (size_t) nt * 128);
            }
    });
    size_t d = 0;
    for (size_t v : diffs) d += v;
    return check_bits("cpu x2..11 " + dims(h.k, h.n) + " K" + kname(h.k2), d, (size_t) (h.n / 128) * 24 * 128);
}
// split-K pieces (every column, slices [ks0, ks1)) against the bands, several activations against one each, and the
// piece cut by columns into uneven runs: the same bits
std::string cpu_rows(const HostMat& h) {
    const cpu::Exl3Part m = part_of(h);
    std::vector<std::vector<float>> xh;
    const float* xp[kCpuNT];
    for (int i = 0; i < kCpuNT; ++i) xh.push_back(rand_x((size_t) h.k, 1.0f));
    for (int i = 0; i < kCpuNT; ++i) xp[i] = xh[(size_t) i].data();
    const int n = h.n;
    std::vector<float> band(128), one((size_t) kCpuNT * n), all((size_t) kCpuNT * n);
    float* ap[kCpuNT];
    for (int i = 0; i < kCpuNT; ++i) ap[i] = all.data() + (size_t) i * n;
    size_t diff = 0, cmp = 0;
    for (int nt : {1, 3, 8, 11}) {
        const int ks0 = nt % 3, ks1 = h.k / 16 - (nt % 2);
        for (int i = 0; i < nt; ++i) {
            cpu::exl3_cpu_rows_acc(m, xp[i], ks0, ks1, one.data() + (size_t) i * n);
            for (int c0 = 0; c0 < n; c0 += 128) {
                cpu::exl3_cpu_band_acc(m, xp[i], c0, ks0, ks1, band.data());
                diff += bits_differ(band.data(), one.data() + (size_t) i * n + c0, 128);
                cmp += 128;
            }
        }
        if (nt > 1) {
            cpu::exl3_cpu_rows_acc_n(m, xp, nt, ks0, ks1, ap);
            diff += bits_differ(one.data(), all.data(), (size_t) nt * n);
            cmp += (size_t) nt * n;
        }
        std::fill(all.begin(), all.end(), -1.0f);
        for (int c0 = 0; c0 < n;) {
            const int c1 = std::min(n, c0 + 128 * (1 + (c0 / 128 + nt) % 3));
            cpu::exl3_cpu_rows_acc_cols_n(m, xp, nt, ks0, ks1, c0, c1, ap);
            c0 = c1;
        }
        diff += bits_differ(one.data(), all.data(), (size_t) nt * n);
        cmp += (size_t) nt * n;
    }
    return check_bits("cpu rows " + dims(h.k, h.n) + " K" + kname(h.k2), diff, cmp);
}
// the prompt path's band GEMM (a band decoded once for many activations) against the band kernel: the same sums up to
// F32 rounding (another order)
std::string cpu_gemm(const HostMat& h) {
    const cpu::Exl3Part m = part_of(h);
    std::vector<std::vector<float>> xh;
    const float* xp[kGemmNT];
    for (int i = 0; i < kGemmNT; ++i) xh.push_back(rand_x((size_t) h.k, 1.0f));
    for (int i = 0; i < kGemmNT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<double> worsts((size_t) (h.n / 256), 0.0);
    parallel_for(h.n / 256, [&](int64_t p0, int64_t p1) {
        std::vector<float> wc(cpu::exl3_cpu_gemm_scratch() + 16), one(128), all((size_t) kGemmNT * 256);
        float* const w64 = (float*) (((uintptr_t) wc.data() + 63) & ~(uintptr_t) 63);
        float* ap[kGemmNT];
        for (int64_t p = p0; p < p1; ++p)
            for (int nb : {1, 2}) {
                const int c0 = (int) p * 256;
                for (int i = 0; i < kGemmNT; ++i) ap[i] = all.data() + (size_t) i * 128 * nb;
                cpu::exl3_cpu_band_gemm(m, xp, kGemmNT, c0, nb, ap, w64);
                for (int i = 0; i < kGemmNT; ++i)
                    for (int b = 0; b < nb; ++b) {
                        cpu::exl3_cpu_band_acc(m, xp[i], c0 + 128 * b, 0, h.k / 16, one.data());
                        double mx = 0.0, d = 0.0;
                        for (int j = 0; j < 128; ++j) {
                            mx = std::max(mx, (double) std::fabs(one[(size_t) j]));
                            d = std::max(d, (double) std::fabs(one[(size_t) j] - ap[i][128 * b + j]));
                        }
                        worsts[(size_t) p] = std::max(worsts[(size_t) p], d / std::max(mx, 1e-30));
                    }
            }
    });
    double worst = 0.0;
    for (double v : worsts) worst = std::max(worst, v);
    ++g_checks;
    const bool ok = worst < 1e-5;
    if (!ok) {
        ++g_fail;
        std::printf("  FAIL cpu gemm %s K%s: worst %.2e of the band's largest\n", dims(h.k, h.n).c_str(),
                    kname(h.k2).c_str(), worst);
    }
    char b[48];
    std::snprintf(b, sizeof b, "%.1e%s", worst, ok ? "" : " FAIL");
    return b;
}
void sec_cpu() {
    if (!cpu::exl3_cpu_available()) {
        std::printf("cpu: the CPU lane's kernels are not built here (x86 only): skipped\n");
        return;
    }
    std::printf("cpu: the CPU lane's band kernels at the experts' shapes (mul1; the lane runs the %s tier)\n",
                tier_name(cpu::exl3_cpu_tier()));
    init_steps();
    for (const int k2 : {4, 5, 6, 8, 10, 16})
        for (const auto& kn : {std::array<int, 2>{kEmbd, kFfExp}, std::array<int, 2>{kFfExp, kEmbd}}) {
            const HostMat h = make_mat(kn[0], kn[1], k2, 2);
            std::string line = "lane vs reference " + cpu_ref(h);
            if (k2 == 5 || k2 == 8 || k2 == 16) line += "  x2..11 " + cpu_n(h);
            if (k2 == 4 || k2 == 5 || k2 == 10) line += "  split-K " + cpu_rows(h);
            if (k2 == 6 || k2 == 8) line += "  band GEMM " + cpu_gemm(h);
            std::printf("  %-12s K%-3s %s\n", dims(h.k, h.n).c_str(), kname(k2).c_str(), line.c_str());
            if (k2 != 10) cpu_exact(h);
        }
}

// ================================================================ main
bool want(const std::set<std::string>& only, const char* s) { return only.empty() || only.count(s) != 0; }

}  // namespace

int main(int argc, char** argv) {
    static const char* const kSections[] = {"mv", "batch", "head", "moe", "win", "set", "gemm", "recon", "blas", "cpu"};
    std::set<std::string> only;
    bool old_layout = false;
    int threads = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--only" && i + 1 < argc) {
            std::string v = argv[++i];
            size_t at = 0;
            while (at <= v.size()) {
                const size_t c = v.find(',', at);
                const std::string s = v.substr(at, c == std::string::npos ? std::string::npos : c - at);
                bool known = false;
                for (const char* k : kSections) known = known || s == k;
                if (!known) {
                    std::fprintf(stderr, "glm_dsa_exl3_test: no section '%s'\n", s.c_str());
                    return 2;
                }
                only.insert(s);
                if (c == std::string::npos) break;
                at = c + 1;
            }
        } else if (a == "--blas-old-layout") {
            old_layout = true;
        } else if (a == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: glm_dsa_exl3_test [--only mv,batch,head,moe,win,set,gemm,recon,blas,cpu] "
                                 "[--blas-old-layout] [--threads N]\n");
            return 2;
        }
    }
    const unsigned hw = std::thread::hardware_concurrency();
    g_threads = threads > 0 ? threads : (int) std::max(1u, std::min(hw == 0 ? 8u : hw, 32u));
    const bool gpu = !(only.size() == 1 && only.count("cpu") == 1);
    const double t_all = now_s();
    cublasHandle_t blas = nullptr;
    if (gpu) {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
            std::printf("glm_dsa_exl3_test: no GPU - skipped\n");
            return 77;
        }
        int dev = 0;
        cudaDeviceProp p{};
        CK(cudaGetDevice(&dev));
        CK(cudaGetDeviceProperties(&p, dev));
        probe_family();
#if defined(STRATA_USE_HIP)
        std::printf("glm_dsa_exl3_test on %s (%s, %d CUs, %.1f GB)\n", p.name, p.gcnArchName, p.multiProcessorCount,
                    (double) p.totalGlobalMem / 1e9);
#else
        std::printf("glm_dsa_exl3_test on %s (sm_%d%d, %d SMs, %.1f GB)\n", p.name, p.major, p.minor,
                    p.multiProcessorCount, (double) p.totalGlobalMem / 1e9);
#endif
        std::printf("  mv: %s kernels; several activations at once: %s; prompt kernels: tensor-core (sm_80+) %s, "
                    "WMMA %s\n",
                    family(), ex::several_ok() ? "yes" : "no", ex::mma_supported() ? "yes" : "no",
                    ex::prompt_wmma() ? "yes" : "no");
        CB(cublasCreate(&blas));
        CB(cublasSetStream(blas, 0));
    }
    std::printf("  CPU lane tier: %s; %d threads for the reference\n",
                cpu::exl3_cpu_available() ? tier_name(cpu::exl3_cpu_tier()) : "none (not x86)", g_threads);
    const auto run = [&](const char* s, const std::function<void()>& f) {
        if (!want(only, s)) return;
        const double t0 = now_s();
        const int f0 = g_fail;
        f();
        std::printf("  [%s: %.1f s, %d failures]\n", s, now_s() - t0, g_fail - f0);
        std::fflush(stdout);
    };
    run("mv", sec_mv);
    run("batch", sec_batch);
    run("head", sec_head);
    run("moe", sec_moe);
    run("win", sec_win);
    run("set", sec_set);
    run("gemm", sec_gemm);
    run("recon", sec_recon);
    run("blas", [&] { sec_blas(blas, old_layout); });
    run("cpu", sec_cpu);
    if (blas != nullptr) cublasDestroy(blas);
    if (!g_refused.empty()) {
        std::printf("refused as expected (for the engine's EXL3 work):\n");
        for (const std::string& r : g_refused) std::printf("  - %s\n", r.c_str());
    }
    const int le = gpu ? ex::launch_errors() : 0;
    std::printf("glm_dsa_exl3_test: %s (%d checks, %d failures, %d launch errors, %.1f s)\n",
                g_fail || le ? "FAIL" : "ok", g_checks, g_fail, le, now_s() - t_all);
    return g_fail || le ? 1 : 0;
}
