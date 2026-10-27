// src/mm3/mm3_gguf.cu - maya-mm3's GGUF weights on the device (mm3_gguf.cuh): the activations' rounding, the dense
// GEMVs and GEMMs, the routed experts' passes and their combine around maya-ds4's kernels (src/ds4/ds4_iq.cu).
#include "mm3_gguf.cuh"

#include "mm3_env.hpp"
#include "../ds4/ds4_iq.cuh"
#include "../ds4/ds4_lt.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace mm3::gg {

namespace iq = strata::ds4::gguf::iq;

namespace {

std::atomic<int> g_errors{0};
std::atomic<int> g_gemm{0};   // set_gemm's backend: 0 the BLAS, 1 hipBLASLt (measured kernels) where it runs

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mm3 gguf %s: %s\n", what, cudaGetErrorString(e));
        g_errors.fetch_add(1, std::memory_order_relaxed);
    }
}
void fail(const char* what) {
    std::fprintf(stderr, "mm3 gguf %s\n", what);
    g_errors.fetch_add(1, std::memory_order_relaxed);
}

constexpr int kWarps = 8, kThreads = kWarps * 32;
constexpr int kMaxEnt = 16;          // a window's tokens at most (mm3_model.cu's kWin)
constexpr int kMaxEmbd = 8192;       // n_embd and n_ff at most (MiniMax-M3.1: 6144, 3072)
constexpr int kMaxQK = 16384;        // a GEMV's inputs at most (the dense MLP's down: 12288)
constexpr size_t kBand = 64u << 20;  // a GEMM's FP16 weight band (bytes): the rows decoded at a time

// ---- int8 activations
// per 32 values (llama.cpp's q8_1 scale, ds4_gguf.cu's q8_quant - the same operations): rows of k F32 values (row r
// at x + r ldx) -> q[r][k], d[r][k / 32] = max|x| / 127 (a warp a 32-block)
__global__ void __launch_bounds__(kThreads) q8_kernel(const float* __restrict__ x, size_t ldx, int k,
                                                      int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = blockIdx.y, b = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const float v = x[(size_t) r * ldx + (size_t) b * 32 + lane];
    float am = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[(size_t) r * (k / 32) + b] = sc;
}
// ggml's Q8_K, the CPU lane's (quantize_row_q8_K_ref; ds4_gguf.cu's q8k_quant - the same operations): per 256 values
// max = the signed value of the largest magnitude (the first one), iscale = -127 / max, q = min(127, nearest(iscale x)),
// d = 1 / iscale into each of the 256's eight 32-blocks
__device__ __forceinline__ int q8k_nearest(float fval) {
    const float val = __fadd_rn(fval, 12582912.0f);
    int i;
    memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}
__global__ void __launch_bounds__(256) q8k_kernel(const float* __restrict__ x, size_t ldx, int k, int8_t* __restrict__ q,
                                                  float* __restrict__ d) {
    __shared__ float s_am[8];
    __shared__ int s_at[8];
    const int r = blockIdx.y, j = (int) threadIdx.x, lane = j & 31, w = j >> 5;
    const size_t at = (size_t) blockIdx.x * 256 + (size_t) j;
    const float v = x[(size_t) r * ldx + at];
    float am = fabsf(v);
    int ix = j;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
        if (oa > am || (oa == am && oi < ix)) am = oa, ix = oi;
    }
    if (lane == 0) s_am[w] = am, s_at[w] = ix;
    __syncthreads();
    if (j == 0) {
        for (int i = 1; i < 8; ++i)
            if (s_am[i] > s_am[0] || (s_am[i] == s_am[0] && s_at[i] < s_at[0])) s_am[0] = s_am[i], s_at[0] = s_at[i];
    }
    __syncthreads();
    const float amax = s_am[0];
    int8_t qv = 0;
    float dv = 0.0f;
    if (amax > 0.0f) {
        const float mx = x[(size_t) r * ldx + (size_t) blockIdx.x * 256 + (size_t) s_at[0]];
        const float iscale = __fdiv_rn(-127.0f, mx);
        qv = (int8_t) min(127, q8k_nearest(__fmul_rn(iscale, v)));
        dv = __fdiv_rn(1.0f, iscale);
    }
    q[(size_t) r * k + at] = qv;
    if (lane == 0) d[(size_t) r * (k / 32) + at / 32] = dv;
}
void q8(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    q8_kernel<<<dim3((unsigned) ((k / 32 + kWarps - 1) / kWarps), (unsigned) rows), kThreads, 0, s>>>(x, ldx, k, q, d);
}
bool iq_q8k() {   // MAYA_MM3_IQ_Q8K=0: the experts' activations a scale a 32 (the dense GEMVs' rounding)
    static const bool on = [] {
        const char* v = ::mm3::env("MM3_IQ_Q8K");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}
void act(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    if (iq_q8k() && k % 256 == 0) q8k_kernel<<<dim3((unsigned) (k / 256), (unsigned) rows), 256, 0, s>>>(x, ldx, k, q, d);
    else q8(x, ldx, rows, k, q, d, s);
}

// ---- the routed experts' sum (mm3_exl3.cu's combine): the experts in plan order (one off the device adds 0), then
// the CPU lane's part and the shared expert's
struct Comb {
    const unsigned long long* plan_ptr;   // [T][8]
    const float* plan_w;                  // [T][8]
    const float* part;                    // [T * 8][n_embd]
    const float* cpu_part;
    size_t cpu_stride;
    const int* cpu_flag;
    const float* sh_out;                  // [T][n_embd]
    float* out;
    size_t out_stride;
    int k, n_embd;
};
__global__ void __launch_bounds__(128) comb_kernel(const Comb a) {
    const int t = blockIdx.y, r = blockIdx.x * 128 + threadIdx.x;
    if (r >= a.n_embd) return;
    float m = 0.0f;
    for (int i = 0; i < a.k; ++i) {
        const int e = t * 8 + i;
        const float p = a.plan_ptr[e] != 0ull ? a.part[(size_t) e * a.n_embd + r] : 0.0f;
        m = __fmaf_rn(a.plan_w[e], p, m);
    }
    if (a.cpu_flag != nullptr && a.cpu_flag[t]) m = __fadd_rn(m, a.cpu_part[(size_t) t * a.cpu_stride + r]);
    a.out[(size_t) t * a.out_stride + r] = __fadd_rn(m, a.sh_out != nullptr ? a.sh_out[(size_t) t * a.n_embd + r] : 0.0f);
}

// ---- the device's wait for the host's word (the CPU lane's parts), as mm3_exl3.cu's: it gives up after 120 s
__device__ unsigned g_timeouts = 0;
__global__ void wait_kernel(const unsigned* p, unsigned v, unsigned long long limit) {
    if (threadIdx.x != 0) return;
#if defined(STRATA_USE_HIP)
    const unsigned long long t0 = wall_clock64();   // (limit: 120 s of its ticks, the host's hipDeviceAttributeWallClockRate)
#else
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));   // (ns)
#endif
    while (*(volatile const unsigned*) p != v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
#if defined(STRATA_USE_HIP)
        const unsigned long long t = wall_clock64();
#else
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
        if (t - t0 > limit) {
            atomicAdd(&g_timeouts, 1u);
            break;
        }
    }
    __threadfence_system();
}
unsigned long long wait_limit() {   // 120 s in the wait kernel's clock
#if defined(STRATA_USE_HIP)
    static const unsigned long long l = [] {
        int dev = 0, khz = 0;
        if (hipGetDevice(&dev) != hipSuccess || hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) !=
                                                    hipSuccess || khz <= 0)
            khz = 100000;   // (100 MHz, RDNA's usual)
        return 120ull * 1000ull * (unsigned long long) khz;
    }();
    return l;
#else
    return 120ull * 1000000000ull;
#endif
}

// ---- per-stream scratch
struct Scratch {
    float* part = nullptr;   // [8][kMaxEmbd]: a token's experts' down outputs
    float* h = nullptr;      // [8][kMaxEmbd]: a token's experts' activations (moe_gate_up's h when the caller gives none)
    int8_t *xq = nullptr, *hq = nullptr, *mq = nullptr;   // int8 rows: x [kMaxEnt], h [kMaxEnt * 8], the GEMVs'
    float *xd = nullptr, *hd = nullptr, *md = nullptr;     //   [kMaxMvJobs * kMaxMvCols] inputs; their scales
    uint16_t* wb = nullptr;  // a GEMM's FP16 weight band (kBand bytes)
    cublasHandle_t blas = nullptr;
};
std::mutex g_mu;
std::unordered_map<cudaStream_t, Scratch> g_scratch;
constexpr size_t kNx = (size_t) kMaxEnt * kMaxEmbd, kNh = (size_t) kMaxEnt * 8 * kMaxEmbd,
                 kNm = (size_t) x3::kMaxMvJobs * x3::kMaxMvCols * kMaxQK;

Scratch* scratch(cudaStream_t s) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_scratch.find(s);
    if (it != g_scratch.end()) return &it->second;
    Scratch sc;
    if (cudaMalloc(&sc.part, (size_t) 8 * kMaxEmbd * 4) != cudaSuccess ||
        cudaMalloc(&sc.h, (size_t) 8 * kMaxEmbd * 4) != cudaSuccess || cudaMalloc(&sc.xq, kNx) != cudaSuccess ||
        cudaMalloc(&sc.xd, kNx / 32 * 4) != cudaSuccess || cudaMalloc(&sc.hq, kNh) != cudaSuccess ||
        cudaMalloc(&sc.hd, kNh / 32 * 4) != cudaSuccess || cudaMalloc(&sc.mq, kNm) != cudaSuccess ||
        cudaMalloc(&sc.md, kNm / 32 * 4) != cudaSuccess || cudaMalloc(&sc.wb, kBand) != cudaSuccess) {
        cudaGetLastError();
        fail("scratch: out of device memory");
        return nullptr;
    }
    if (cublasCreate(&sc.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(sc.blas, s) != CUBLAS_STATUS_SUCCESS) {
        fail("scratch: cuBLAS");
        return nullptr;
    }
    return &g_scratch.emplace(s, sc).first->second;
}

bool geometry(const char* who, int k, int n_embd, int n_ff) {
    if (k < 1 || k > 8 || n_embd % 256 != 0 || n_ff % 256 != 0 || n_embd > kMaxEmbd || n_ff > kMaxEmbd) {
        std::fprintf(stderr, "mm3 gguf %s: geometry not covered (k %d n_embd %d n_ff %d)\n", who, k, n_embd, n_ff);
        g_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

// a layer's experts as ds4_iq's arguments: the roles' offsets and types, MiniMax's swigluoai (false: types not covered)
bool iq_args(const x3::ExpertLayout& lay, int n_embd, int n_ff, float limit, iq::Args& ia, const char* who) {
    ia = iq::Args{};
    for (int r = 0; r < 3; ++r) ia.tr[r] = lay.trellis[r];
    ia.n_embd = n_embd;
    ia.n_ff = n_ff;
    ia.limit = limit;
    ia.tg = lay.gg_type[0];
    ia.td = lay.gg_type[2];
    ia.act = iq::kActOai;
    ia.tu = lay.gg_type[1] != lay.gg_type[0] ? lay.gg_type[1] : 0;
    if (lay.fmt != kFmtGG || !iq::supported(lay.gg_type[0]) || !iq::supported(lay.gg_type[1]) ||
        !iq::supported(lay.gg_type[2])) {
        std::fprintf(stderr, "mm3 gguf %s: expert types %s / %s / %s not covered\n", who, type_name(lay.gg_type[0]),
                     type_name(lay.gg_type[1]), type_name(lay.gg_type[2]));
        g_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

void combine(const x3::ExpertLayout&, const unsigned long long* plan_ptr, const float* plan_w, const float* part,
             const float* cpu_part, size_t cpu_stride, const int* cpu_flag, const float* sh_out, float* out,
             size_t out_stride, int T, int k, int n_embd, cudaStream_t s) {
    Comb c{plan_ptr, plan_w, part, cpu_part, cpu_stride, cpu_flag, sh_out, out, out_stride, k, n_embd};
    comb_kernel<<<dim3((unsigned) ((n_embd + 127) / 128), (unsigned) T), 128, 0, s>>>(c);
}

}  // namespace

// ---------------------------------------------------------------- types
int type_of(const std::string& dtype) {
    for (int t : {iq::kQ8_0, iq::kQ2_K, iq::kQ3_K, iq::kQ4_K, iq::kQ5_K, iq::kQ6_K, iq::kIQ2_XXS, iq::kIQ2_XS,
                  iq::kIQ3_XXS, iq::kIQ1_S, iq::kIQ3_S, iq::kIQ2_S, iq::kIQ4_XS, iq::kIQ1_M})
        if (dtype == iq::name(t)) return t;
    return -1;
}
bool expert_type(int t) { return iq::supported(t); }
bool dense_type(int t) { return iq::mv_supported(t); }
const char* type_name(int t) { return iq::name(t); }
int block_values(int t) { return iq::block_values(t); }
size_t row_bytes(int t, int64_t k) {
    return iq::mv_supported(t) && k % iq::block_values(t) == 0 ? iq::row_bytes(t, k) : 0;
}

bool prepare(cudaStream_t s) { return scratch(s) != nullptr; }
void set_gemm(int backend) { g_gemm.store(backend == 1 && gemm_lt_available() ? 1 : 0); }
int gemm_backend() { return g_gemm.load(); }
bool gemm_lt_available() {
#if defined(STRATA_USE_HIP)
    return strata::ds4::lt::available();
#else
    return false;
#endif
}
void set_lt_store(const char* path) { strata::ds4::lt::set_store(path); }
size_t scratch_bytes() {
    // (+ cuBLAS' workspace, and ds4_iq's light sets' int8 rows - 512 experts x 16 rows of up to 8192 values - which it
    //  takes on a stream's first prompt)
    const size_t n8 = kNx + kNh + kNm;
    return 2 * (size_t) 8 * kMaxEmbd * 4 + n8 + n8 / 32 * 4 + kBand + (16u << 20) + (72u << 20);
}
int launch_errors() { return g_errors.load(std::memory_order_relaxed); }
unsigned wait_timeouts() {
    unsigned n = 0;
    if (cudaMemcpyFromSymbol(&n, g_timeouts, sizeof n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
}

void act_quant(const float* x, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    act(x, (size_t) k, rows, k, q, d, s);
    check("act_quant");
}
void q8_quant(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    q8(x, ldx, rows, k, q, d, s);
    check("q8_quant");
}

// ---------------------------------------------------------------- dense matrices
bool mv(const x3::MvJob* jobs, int n, cudaStream_t s) {
    if (n <= 0 || n > x3::kMaxMvJobs) return false;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    iq::MvJob J[x3::kMaxMvJobs];
    for (int i = 0; i < n; ++i) {
        const x3::MvJob& q = jobs[i];
        const x3::Mat& m = *q.m;
        if (m.fmt != kFmtGG || m.raw == nullptr || q.xf == nullptr || q.y == nullptr || q.ncol < 1 ||
            q.ncol > x3::kMaxMvCols || m.k > kMaxQK || m.k % 32 != 0 || row_bytes(m.gg_type, m.k) == 0) {
            std::fprintf(stderr, "mm3 gguf mv: job %d not covered (%s, k %d n %d, %d activations)\n", i,
                         type_name(m.gg_type), m.k, m.n, q.ncol);
            g_errors.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const size_t at = (size_t) i * x3::kMaxMvCols * kMaxQK;
        q8(q.xf, q.ncol > 1 ? q.xf_cs : (size_t) m.k, q.ncol, m.k, sc->mq + at, sc->md + at / 32, s);
        iq::MvJob& b = J[i];
        b.w = (const uint8_t*) m.raw;
        b.ld = m.ld > 0 ? (size_t) m.ld : row_bytes(m.gg_type, m.k);
        b.type = m.gg_type;
        b.k = m.k;
        b.n = m.n;
        b.ncol = q.ncol;
        b.xq = sc->mq + at;
        b.xd = sc->md + at / 32;
        b.y = q.y;
        b.ycs = q.ncol > 1 ? q.y_cs : (size_t) m.n;
        b.bias = q.bias;
        b.alpha = q.alpha;
    }
    if (!iq::mv(J, n, s)) {
        fail("mv: a type not covered");
        return false;
    }
    check("mv");
    return true;
}

void gemm_rows(const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s) {
    if (rows <= 0) return;
    Scratch* sc = scratch(s);
    const size_t ld = m.ld > 0 ? (size_t) m.ld : row_bytes(m.gg_type, m.k);
    if (sc == nullptr || m.fmt != kFmtGG || m.raw == nullptr || ld == 0 || m.k % 256 != 0) {
        fail("gemm_rows: not a GGUF matrix the kernels take");
        return;
    }
    // the weight's rows (the outputs) in bands of the FP16 scratch, each decoded once and multiplied by every row
    const int band = (int) std::max<size_t>(256, kBand / ((size_t) m.k * 2) / 256 * 256);
    const float one = 1.0f;
    for (int c0 = 0; c0 < m.n; c0 += band) {
        const int nc = std::min(band, m.n - c0);
        if (!iq::f16_rows(m.gg_type, (const uint8_t*) m.raw + (size_t) c0 * ld, m.k, nc, sc->wb, m.k, s)) {
            fail("gemm_rows: f16_rows");
            return;
        }
        // (hipBLASLt: this shape's fastest kernel on this GPU, measured at its first product - ds4_lt)
        if (beta == 0.0f && g_gemm.load(std::memory_order_relaxed) == 1 &&
            strata::ds4::lt::gemm(s, sc->wb, nc, m.k, x16, ldx, rows, y + c0, ldy))
            continue;
        // Y^T [nc][rows] (column-major, ldy) = W [nc][k] (row-major: A^T) x X^T [k][rows] (ldx)
        if (cublasGemmEx(sc->blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, m.k, &one, sc->wb, CUDA_R_16F, m.k, x16, CUDA_R_16F,
                         ldx, &beta, y + c0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) !=
            CUBLAS_STATUS_SUCCESS) {
            fail("gemm_rows: cublasGemmEx");
            return;
        }
    }
    check("gemm_rows");
}

// ---------------------------------------------------------------- routed experts
void moe_gate_up(const x3::MoeArgs& a, const float* x, float* h, cudaStream_t s) {
    if (!geometry("moe_gate_up", a.k, a.n_embd, a.n_ff)) return;
    Scratch* sc = scratch(s);
    iq::Args ia;
    if (sc == nullptr || !iq_args(a.lay, a.n_embd, a.n_ff, a.limit, ia, "moe_gate_up")) return;
    act(x, (size_t) a.n_embd, 1, a.n_embd, sc->xq, sc->xd, s);
    ia.plan_ptr = a.plan_ptr;
    ia.h = h != nullptr ? h : sc->h;
    ia.xq = sc->xq;
    ia.xd = sc->xd;
    if (!iq::gate_up(ia, a.k, 1, s)) fail("moe_gate_up: not covered");
    check("moe_gate_up");
}

void moe_down(const x3::MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    if (!geometry("moe_down", a.k, a.n_embd, a.n_ff)) return;
    Scratch* sc = scratch(s);
    iq::Args ia;
    if (sc == nullptr || !iq_args(a.lay, a.n_embd, a.n_ff, a.limit, ia, "moe_down")) return;
    act(h != nullptr ? h : sc->h, (size_t) a.n_ff, a.k, a.n_ff, sc->hq, sc->hd, s);
    ia.plan_ptr = a.plan_ptr;
    ia.part = sc->part;
    ia.hq = sc->hq;
    ia.hd = sc->hd;
    if (!iq::down(ia, a.k, 1, s)) fail("moe_down: not covered");
    combine(a.lay, a.plan_ptr, a.plan_w, sc->part, a.cpu_part, (size_t) a.n_embd, a.cpu_flag, sh_out, out,
            (size_t) a.n_embd, 1, a.k, a.n_embd, s);
    check("moe_down");
}

bool win_experts(const x3::WinArgs& a, cudaStream_t s) {
    if (a.T < 1 || a.T > kMaxEnt || !geometry("win_experts", a.k, a.n_embd, a.n_ff) || a.h == nullptr ||
        a.part == nullptr || a.out == nullptr) {
        std::fprintf(stderr, "mm3 gguf win_experts: not covered (T %d)\n", a.T);
        return false;
    }
    Scratch* sc = scratch(s);
    iq::Args ia;
    if (sc == nullptr || !iq_args(a.lay, a.n_embd, a.n_ff, a.limit, ia, "win_experts")) return false;
    ia.grp_ptr = a.grp_ptr;
    ia.grp_start = a.grp_start;
    ia.grp_ent = a.grp_ent;
    ia.n_groups = a.n_groups;
    ia.h = a.h;
    ia.part = a.part;
    ia.xq = sc->xq;
    ia.xd = sc->xd;
    ia.hq = sc->hq;
    ia.hd = sc->hd;
    const int groups = a.T * a.k;   // at most (the blocks past the real count return at once)
    act(a.x, a.x_stride, a.T, a.n_embd, sc->xq, sc->xd, s);
    const bool ok1 = iq::gate_up(ia, groups, a.T, s);
    act(a.h, (size_t) a.n_ff, a.T * 8, a.n_ff, sc->hq, sc->hd, s);   // (the rows of picks not on the device: unused)
    const bool ok2 = iq::down(ia, groups, a.T, s);
    if (a.wait_ptr != nullptr) wait_kernel<<<1, 32, 0, s>>>(a.wait_ptr, a.wait_val, wait_limit());
    combine(a.lay, a.plan_ptr, a.plan_w, a.part, a.cpu_part, a.cpu_stride != 0 ? a.cpu_stride : (size_t) a.n_embd,
            a.cpu_flag, a.sh_out, a.out, a.out_stride, a.T, a.k, a.n_embd, s);
    check("win_experts");
    if (!ok1 || !ok2) fail("win_experts: not covered");
    return ok1 && ok2;   // (the wait was enqueued either way: the host's answer always lands)
}

namespace {
iq::SetArgs set_args(const x3::SetArgs& sa, const iq::Args& ia) {
    iq::SetArgs q;
    q.wbase = sa.wbase;
    q.stride = sa.stride;
    for (int r = 0; r < 3; ++r) q.tr[r] = sa.lay.trellis[r];
    q.bounds = sa.bounds;
    q.row0 = sa.row0;
    q.max_rows = sa.max_rows;
    q.n_exp = sa.n_exp;
    q.n_embd = sa.n_embd;
    q.n_ff = sa.n_ff;
    q.limit = sa.limit;
    q.skip_ge = sa.skip_ge;
    q.tg = ia.tg;
    q.td = ia.td;
    q.act = ia.act;
    q.tu = ia.tu;
    q.graw = sa.graw;
    return q;
}
}  // namespace

void set_gate_up(const x3::SetArgs& sa, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s) {
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    iq::Args ia;
    if (!geometry("set_gate_up", 1, sa.n_embd, sa.n_ff) || sa.wbase == nullptr ||
        !iq_args(sa.lay, sa.n_embd, sa.n_ff, sa.limit, ia, "set_gate_up"))
        return;
    if (ia.tu != 0 && sa.graw == nullptr) {
        fail("set_gate_up: gate and up of different types need the graw scratch");
        return;
    }
    iq::SetArgs q = set_args(sa, ia);
    q.xf = x;
    q.xh = sa.x16;   // (the tensor-core tiles' FP16 rows; the light kernels keep xf)
    q.idx = row_idx;
    q.ldx = ldx;
    q.h = h16;
    q.ldo = ldh;
    if (!iq::set_gate_up(q, s)) fail("set_gate_up: not covered");
    check("set_gate_up");
}

void set_down(const x3::SetArgs& sa, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s) {
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    iq::Args ia;
    if (!geometry("set_down", 1, sa.n_embd, sa.n_ff) || sa.wbase == nullptr ||
        !iq_args(sa.lay, sa.n_embd, sa.n_ff, sa.limit, ia, "set_down"))
        return;
    iq::SetArgs q = set_args(sa, ia);
    q.xh = h16;
    q.ldx = ldh;
    q.y = y;
    q.ldo = ldy;
    if (!iq::set_down(q, s)) fail("set_down: not covered");
    check("set_down");
}

}  // namespace mm3::gg
