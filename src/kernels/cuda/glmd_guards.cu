// src/kernels/cuda/glmd_guards.cu - glm-dsa wave 9: the device halves of B15.1 (the FP16 activation range guard of
// the prompt path) and B15.2 (mmq_malloc / mmq_zero_gaps of the MMQ expert banks).  The design is in
// include/strata/kernels/glmd_f16_guard.hpp and glmd_mmq_pad.hpp.  In strata_engine on every GPU backend because the
// Maya-owned glm_prefill.cu / glm_fast_path.cu call it.  HIP builds compile the same code (the glm-dsa prompt path runs
// there too).  The guard's auto mode is on for every device, AMD's included: see f16_guard_active.
#include "strata/kernels/glmd_f16_guard.hpp"
#include "strata/kernels/glmd_mmq_pad.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include <cuda_fp16.h>

namespace glmd {

namespace {
std::atomic<uint64_t> g_guard_errors{0};
void guard_error(const char* what) {
    const uint64_t n = g_guard_errors.fetch_add(1, std::memory_order_relaxed);
    if (n < 8) std::fprintf(stderr, "glm f16 guard: %s\n", what);
}
}  // namespace

uint64_t f16_guard_errors() { return g_guard_errors.load(std::memory_order_relaxed); }

namespace {

constexpr int kThreads = 256;
constexpr int kSlots = 16;   // live bindings per stream (a phase uses at most 4: x16, qr16, q16, attn16 / h16)

__device__ __forceinline__ float block_max(float v, float* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float m = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) m = fmaxf(m, sh[i]);
    return m;
}

// one block a row: amax over the row's heads, the scale, and (only when it is not 1) the row again from F32
__global__ void __launch_bounds__(kThreads) guard_rows_kernel(const float* __restrict__ src, int64_t src_row,
                                                              int64_t src_head, int n_head, int K,
                                                              __half* __restrict__ dst, float* __restrict__ scale) {
    __shared__ float sh[kThreads / 32];
    const int64_t t = blockIdx.x;
    const int n = n_head * K;
    const float* row = src + t * src_row;
    float m = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int h = i / K, k = i - h * K;
        m = fmaxf(m, fabsf(row[h * src_head + k]));
    }
    const float s = f16_guard_scale(block_max(m, sh));
    if (threadIdx.x == 0) scale[t] = s;
    if (s == 1.0f) return;
    const float inv = 1.0f / s;
    __half* d = dst + t * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int h = i / K, k = i - h * K;
        d[i] = __float2half(row[h * src_head + k] * inv);
    }
}

__global__ void __launch_bounds__(kThreads) guard_swiglu_kernel(const float* __restrict__ gate,
                                                                const float* __restrict__ up, int n_ff, float limit,
                                                                __half* __restrict__ h16, float* __restrict__ scale) {
    __shared__ float sh[kThreads / 32];
    const int64_t t = blockIdx.x, o = t * n_ff;
    float m = 0.0f;
    for (int c = threadIdx.x; c < n_ff; c += blockDim.x) m = fmaxf(m, fabsf(swiglu_value(gate[o + c], up[o + c], limit)));
    const float s = f16_guard_scale(block_max(m, sh));
    if (threadIdx.x == 0) scale[t] = s;
    if (s == 1.0f) return;
    const float inv = 1.0f / s;
    for (int c = threadIdx.x; c < n_ff; c += blockDim.x) h16[o + c] = __float2half(swiglu_value(gate[o + c], up[o + c], limit) * inv);
}

__global__ void __launch_bounds__(kThreads) unscale_kernel(float* __restrict__ Y, int64_t ldy, int n,
                                                           const float* __restrict__ scale) {
    const int64_t t = blockIdx.x;
    const float s = scale[t];
    if (s == 1.0f) return;
    float* y = Y + t * ldy;
    for (int c = threadIdx.x; c < n; c += blockDim.x) y[c] *= s;
}

// ---- the binding table ----------------------------------------------------------------------------------------------

struct Bind {
    const uint16_t* base = nullptr;
    int64_t ld = 0, rows = 0;
    int slot = -1;
    uint64_t order = 0;
};
struct StreamGuard {
    std::vector<Bind> binds;
    float* scales[kSlots] = {};
    int64_t cap[kSlots] = {};
    uint64_t order = 0;
};
std::mutex g_mu;
std::map<std::pair<int, void*>, StreamGuard>& table() {
    static auto* t = new std::map<std::pair<int, void*>, StreamGuard>();   // never destroyed (process-exit order)
    return *t;
}

StreamGuard& of(void* stream) {
    int dev = 0;
    cudaGetDevice(&dev);
    return table()[{dev, stream}];
}

// the scales for a new binding of dst (rows T): the slot dst had, else a free one; nullptr when the table is full
float* bind(StreamGuard& G, const uint16_t* dst, int64_t ld, int64_t T, bool phase) {
    if (phase) {
        std::vector<Bind> keep;
        for (const Bind& b : G.binds)
            if (b.base == dst) keep.push_back(b);
        G.binds.swap(keep);
    }
    int slot = -1;
    for (auto it = G.binds.begin(); it != G.binds.end(); ++it)
        if (it->base == dst) {
            slot = it->slot;
            G.binds.erase(it);
            break;
        }
    if (slot < 0) {
        bool used[kSlots] = {};
        for (const Bind& b : G.binds) used[b.slot] = true;
        for (int i = 0; i < kSlots && slot < 0; ++i)
            if (!used[i]) slot = i;
    }
    if (slot < 0) {
        guard_error("more than 16 guarded FP16 buffers live on one stream: this one runs unguarded");
        // STRATA_GLMD_GUARD_DEBUG=1: the live bindings at the first overflows (base, row stride, rows, age) - debug
        static int dumps = 0;
        static const bool dbg = std::getenv("STRATA_GLMD_GUARD_DEBUG") != nullptr;
        if (dbg && dumps < 3) {
            ++dumps;
            std::fprintf(stderr, "glm f16 guard: new %p ld %lld rows %lld phase %d; live:", (const void*) dst,
                         (long long) ld, (long long) T, (int) phase);
            for (const Bind& b : G.binds)
                std::fprintf(stderr, " [%p ld %lld rows %lld slot %d age %llu]", (const void*) b.base, (long long) b.ld,
                             (long long) b.rows, b.slot, (unsigned long long) (G.order - b.order));
            std::fprintf(stderr, "\n");
        }
        return nullptr;
    }
    if (G.cap[slot] < T) {
        if (G.scales[slot] != nullptr) cudaFree(G.scales[slot]);   // (synchronises: nothing reads the old scales)
        G.scales[slot] = nullptr;
        G.cap[slot] = 0;
        const int64_t cap = (T + 4095) / 4096 * 4096;
        if (cudaMalloc(&G.scales[slot], (size_t) cap * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            G.scales[slot] = nullptr;
            guard_error("the scale buffer did not allocate: this buffer runs unguarded");
            return nullptr;
        }
        G.cap[slot] = cap;
    }
    G.binds.push_back(Bind{dst, ld, T, slot, ++G.order});
    return G.scales[slot];
}

bool launched(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    guard_error(what);
    return false;
}

}  // namespace

// Auto is on for every device: what decides it is the operand, and the glm-dsa prompt path hands FP16 activations to
// all of its tensor-core products on every backend - cuBLAS / hipBLAS GemmEx with CUDA_R_16F (glmpf::gemm, hgemm_q),
// the absorbed MLA's strided-batched products, hipBLASLt's f16 GEMM - so a row above 65504 turns inf on an Ampere card
// or a Radeon as it does on the V100 (#27016 tied the rule to Volta because llama.cpp feeds F32 activations to FP16
// cuBLAS only there).  Every output stays bit-identical while no row overflows; the cost is one amax pass per FP16
// conversion.  Only glm-dsa calls the guard (glm_prefill_dsa.cu, and glm_prefill.cu behind the arch check).
bool f16_guard_active() { return f16_guard_mode() != 0; }

bool f16_guard_rows(const float* src, int64_t src_row, int64_t src_head, int n_head, int K, int T, uint16_t* dst,
                    void* stream, bool phase) {
    if (T <= 0 || !f16_guard_active()) return true;
    if (src == nullptr || dst == nullptr || n_head <= 0 || K <= 0 || T > 0x7fffffff) {
        guard_error("f16_guard_rows: refused shape");
        return false;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    float* sc = bind(of(stream), dst, (int64_t) n_head * K, T, phase);
    if (sc == nullptr) return true;
    guard_rows_kernel<<<T, kThreads, 0, (cudaStream_t) stream>>>(src, src_row, src_head, n_head, K, (__half*) dst, sc);
    return launched("guard_rows launch");
}

bool f16_guard_swiglu(const float* gate, const float* up, int n_ff, int T, float limit, uint16_t* h16, void* stream) {
    if (T <= 0 || !f16_guard_active()) return true;
    if (gate == nullptr || up == nullptr || h16 == nullptr || n_ff <= 0) {
        guard_error("f16_guard_swiglu: refused shape");
        return false;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    float* sc = bind(of(stream), h16, n_ff, T, false);
    if (sc == nullptr) return true;
    guard_swiglu_kernel<<<T, kThreads, 0, (cudaStream_t) stream>>>(gate, up, n_ff, limit, (__half*) h16, sc);
    return launched("guard_swiglu launch");
}

void f16_guard_reset(void* stream) {
    std::lock_guard<std::mutex> lk(g_mu);
    int dev = 0;
    cudaGetDevice(&dev);
    const auto it = table().find({dev, stream});
    if (it != table().end()) it->second.binds.clear();
}

namespace {
// the latest binding covering x16 (the caller holds g_mu)
const Bind* covering(StreamGuard& G, const void* x16) {
    const Bind* best = nullptr;
    const uint16_t* p = (const uint16_t*) x16;
    for (const Bind& b : G.binds)
        if (p >= b.base && p < b.base + b.rows * b.ld && (best == nullptr || b.order > best->order)) best = &b;
    return best;
}
}  // namespace

bool f16_guard_after(const void* x16, int64_t ldx, float* Y, int64_t ldy, int n, int T, float beta, void* stream) {
    if (T <= 0 || n <= 0 || x16 == nullptr || !f16_guard_active()) return true;   // (off: no binding is live)
    std::lock_guard<std::mutex> lk(g_mu);
    int dev = 0;
    cudaGetDevice(&dev);
    const auto it = table().find({dev, stream});
    if (it == table().end() || it->second.binds.empty()) return true;
    StreamGuard& G = it->second;
    const Bind* b = covering(G, x16);
    if (b == nullptr) return true;
    const int64_t d = (const uint16_t*) x16 - b->base;
    if (d % b->ld != 0 || (ldx != b->ld && T > 1)) {
        guard_error("a guarded FP16 operand read off its rows (unscaled)");
        return false;
    }
    const int64_t r0 = d / b->ld;
    if (r0 + T > b->rows) {
        guard_error("a guarded FP16 operand read past its rows (unscaled)");
        return false;
    }
    if (beta != 0.0f) {
        guard_error("a guarded FP16 operand in a GEMM with beta != 0 (unscaled)");
        return false;
    }
    unscale_kernel<<<T, kThreads, 0, (cudaStream_t) stream>>>(Y, ldy, n, G.scales[b->slot] + r0);
    return launched("unscale launch");
}

const float* f16_guard_scales_for(const void* x16, void* stream) {
    std::lock_guard<std::mutex> lk(g_mu);
    int dev = 0;
    cudaGetDevice(&dev);
    const auto it = table().find({dev, stream});
    if (it == table().end()) return nullptr;
    const Bind* b = covering(it->second, x16);
    return b == nullptr ? nullptr : it->second.scales[b->slot] + ((const uint16_t*) x16 - b->base) / b->ld;
}

// ---- B15.2 ----------------------------------------------------------------------------------------------------------

int mmq_malloc(void** p, size_t bytes) {
    *p = nullptr;
    if (!mmq_pad_on()) return (int) cudaMalloc(p, bytes);   // not glm-dsa: Maya's plain allocation
    void* q = nullptr;
    const cudaError_t e = cudaMalloc(&q, mmq_padded(bytes));
    if (e != cudaSuccess) return (int) e;
#if defined(STRATA_USE_HIP)
    // the legacy-stream zeroing must land before the tiers' non-blocking streams copy experts in: Windows' HIP runtime
    // submits legacy-stream work only when the host waits on it (#133), so wait for it here (allocation time only)
    cudaError_t z = cudaMemset(q, 0, mmq_padded(bytes));
    if (z == cudaSuccess) z = cudaStreamSynchronize(nullptr);
#else
    const cudaError_t z = cudaMemset(q, 0, mmq_padded(bytes));
#endif
    if (z != cudaSuccess) {
        cudaFree(q);
        return (int) z;
    }
    *p = q;
    return 0;
}

bool mmq_zero_gaps(void* base, int n, size_t stride, size_t blob, size_t room, void* stream) {
    if (base == nullptr || n < 0 || !mmq_pad_on()) return true;
    const std::vector<MmqRun> runs = mmq_gap_runs(n, stride, blob, room);
    if (runs.empty()) return true;
    // at most GE + 1 short runs a group: a memset each, on the copy stream before the group's ready event
    for (const MmqRun& r : runs)
        if (cudaMemsetAsync((uint8_t*) base + r.off, 0, r.len, (cudaStream_t) stream) != cudaSuccess) {
            cudaGetLastError();
            guard_error("mmq_zero_gaps memset");
            return false;
        }
    return true;
}

}  // namespace glmd
