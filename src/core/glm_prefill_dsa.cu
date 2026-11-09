// src/core/glm_prefill_dsa.cu - the glm-dsa prompt layer (glm-dsa B9.1, B9.6, B9.7, B14.1); see glm_prefill_dsa.hpp.
//
// One glm-dsa attention layer for a chunk of T tokens, in query sub-batches of PromptTiling::q_sub (2048):
//   projections   qr_raw = W_q_a x, kv_raw = W_kv_a x (576), full layers: ik_raw = W_idx_k x, iw = W_idx_proj x / 64
//                 (cuBLAS: native weights dequantized to FP16 in slices, BF16 narrowed; the indexer's BF16 widened
//                 to F32 - the selection is sensitive, as on the token path)
//   prep          glmd::dsa_prep_rows: qr (+FP16), latent / k_pe / index keys at p0 + t
//   queries       q = W_q_b qr (64 x 256), iq = W_idx_q_b qr (32 x 128), roped at p0 + t (dsa_rope_q_rows)
//   selection     full layers, once the sub-batch sees more than 2048 keys: index_score_tile_wmma over 16K-key blocks
//                 (FP16 queries / keys on tensor cores, F32 sums; STRATA_GLMD_IDX_WMMA=0 or a device without its 80 KB
//                 of opt-in shared memory: index_score_tile_keys, F32) merged into per-row running sets
//                 (topk_merge_tile), the final tile writing the chunk's selection rows `sel` (kept in io.keep), which
//                 the shared layers after it read as they are (A-96).  On a 7.7K prompt the two scorers' selections
//                 share 99.97% of their cells (mean of 63 layer sub-batches; the worst row 99.71%).
//                 STRATA_GLMD_IDX_CHECK=1 prints that agreement per layer and sub-batch (a debug aid)
//   q_abs         W_kb_h^T q_nope per head: FP16 strided-batched GEMM (B9.7; STRATA_GLMD_PF_F16=0: F32)
//   attention     glmd::mla_prompt_h64 (a block per row and 64 heads; STRATA_GLMD_PF_H64=0: mla_prompt_wmma, a block
//                 per row and 16 heads, B9.5; STRATA_GLMD_PF_WMMA=0 or no opt-in: mla_prompt_f32), the walk by
//                 glmd::attn_walk (B14.12: dense below 2049 keys, dense-masked below ratio x 2048, else gather; the
//                 ratio is 1 for the 64-head kernel - it gathers from 2049 keys on -, 4 for CUDA's 16-head ones).
//                 HIP: mla_prompt_rdna on gfx11 / gfx12 (B9.5's arithmetic on RDNA WMMA; STRATA_GLMD_PF_RDNA=0: off),
//                 else mla_prompt_f32 - both gather from 2049 keys on (ratio 1; STRATA_GLMD_GATHER_RATIO overrides)
//   out           W_vb ctx per head (FP16 strided-batched), then attn_output into the mixer rows
// The scale is glmd::attn_scale(qk_head) (1/16 for GLM-5.3), never Flash's 1/sqrt(qk_nope).
// glm-dsa wave 5: the caches are the decode mixer's, byte for byte in layout - latent / k_pe / index keys at the state
// layout's offsets, cell s = position s, the index keys in the layout's dtype (GlmStateLayout::ik_bytes: F32, or FP16
// with STRATA_GLM_IDX_FP16, B10.8) for both the write (dsa_prep_rows) and the score (index_score_tile_keys); a native
// Q8_0 W_kb / W_vb (B10.7, STRATA_PACK_NATIVE32 packs) is dequantized to FP16 for the absorbed products.
#include "strata/core/glm_env.hpp"
#include "glm_prefill_dsa.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/glm_kv_stream.hpp"
#include "glm_prefill_types.hpp"

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/glmd_batch.hpp"
#include "strata/kernels/glmd_f16_guard.hpp"
#include "strata/kernels/glmd_forward.hpp"
#include "strata/kernels/glmd_route.hpp"
#include "strata/kernels/iq_kernels.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/prefill/gemm.hpp"   // (the hipBLASLt projections, S7)
#endif
#include "strata/prefill/moe_mmq_strata.hpp"

#include "../kernels/cuda/glmd_rope.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#if defined(STRATA_USE_HIP)
#include <mutex>
#endif

namespace strata::core::glmpf {

namespace gf = strata::kernels::glmf;

bool f16_covers(int type, bool dsa) {
    // Maya v1.1.3's prefill_setup set: BF16 and the dequant_bf16 table
    if (type == gf::kTypeBF16 || strata::kernels::dequant_bf16_supported(type)) return true;
    if (!dsa) return false;                              // Flash: Maya's set, unchanged (no i-quant)
    // dequant_f16 (dequant_bf16.cu) also sends the i-quants to iq_dequant_f16: glm-dsa only
    if (type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 29) return true;
    if (built() && ext_f16_type(type)) return true;      // glm-dsa's extra types (glm_prefill_types.hpp)
    static bool told[256] = {};
    if (type >= 0 && type < 256 && !told[type]) {
        told[type] = true;
        std::fprintf(stderr, "glm prefill: %s\n", refusal(type).c_str());
    }
    return false;
}

std::string uncovered(const glmdsa::GlmDsaFast* dsa, int il) {
    if (dsa == nullptr || il < 0 || (size_t) il >= dsa->L.size()) return "blk." + std::to_string(il) + " (no glm-dsa weights) ";
    const glmdsa::GlmDsaLayerW& L = dsa->L[(size_t) il];
    std::string why;
    const auto ok = [&](const WSlot& w, const char* nm, bool need) {
        if (!need && w.q == nullptr && w.f32 == nullptr) return;
        if ((w.q == nullptr && w.f32 == nullptr) || !(w.type == gf::kTypeF32 || f16_covers(w.type, true)))
            why += "blk." + std::to_string(il) + "." + nm + " (type " + std::to_string(w.type) + " " +
                   type_name(w.type) + ") ";
    };
    ok(L.q_a, "attn_q_a", true);
    ok(L.q_b, "attn_q_b", true);
    ok(L.kv_a, "attn_kv_a_mqa", true);
    ok(L.out, "attn_output", true);
    const auto f = [&](const WSlot& w, const char* nm) {   // the absorbed products take BF16 / F32 only
        if ((w.q == nullptr && w.f32 == nullptr) || !glmdsa::float_type(w.type))
            why += "blk." + std::to_string(il) + "." + nm + " (type " + std::to_string(w.type) + ") ";
    };
    const auto fq = [&](const WSlot& w, const char* nm) {   // ... or native Q8_0 (B10.7: dequantized to FP16)
        if (w.type == glmdsa::kTypeQ8_0 && w.q != nullptr && f16_covers(w.type, true)) return;
        f(w, nm);
    };
    fq(L.k_b, "attn_k_b");
    fq(L.v_b, "attn_v_b");
    if (L.full) {
        f(L.idx_k, "indexer.attn_k");
        f(L.idx_q_b, "indexer.attn_q_b");
        f(L.idx_proj, "indexer.proj");
    }
    return why;
}

bool geometry_ok(const Glm5Geometry& g) {
    if (g.arch != GlmArch::GlmDsa) return g.kv_lora == 512 && g.n_head % 16 == 0 && g.kda_head_dim == 128;
    return g.kv_lora > 0 && g.kv_lora <= 512 && g.kv_lora % 32 == 0 && (g.qk_rope == 0 || g.qk_rope == 64) &&
           g.n_head > 0 && g.n_head % 16 == 0 && g.idx_key > 0 && g.idx_key <= 128 && g.idx_top_k > 0 &&
           g.q_lora % 32 == 0 && g.q_lora <= 2048 && g.v_head % 32 == 0 && g.n_embd % 32 == 0 && g.n_embd <= 8192;
}

bool lane_first(const Glm5Geometry& g, int rows, double upload_ms, double lane_expert_ms, double lane_row_ms) {
    if (g.arch != GlmArch::GlmDsa) return false;
    return rows < glmd::upload_rows_min_measured(upload_ms, lane_expert_ms, lane_row_ms);
}

namespace {

glmd::PromptDims dims(const Glm5Geometry& g, size_t T, int64_t max_ctx) {
    glmd::PromptDims d;
    d.T = (int64_t) T;
    d.max_ctx = std::max<int64_t>(1, max_ctx);
    d.n_head = g.n_head;
    d.kv_lora = g.kv_lora;
    d.qk_rope = g.qk_rope;
    d.qk_nope = g.qk_nope;
    d.v_head = g.v_head;
    d.q_lora = g.q_lora;
    d.idx_heads = g.idx_heads;
    d.idx_dim = g.idx_key;
    d.top_k = g.idx_top_k;
    d.n_embd = g.n_embd;
    return d;
}

}  // namespace

size_t scratch_bytes(const Glm5Geometry& g, size_t T, int64_t max_ctx) {
    return glmd::prompt_tiling(dims(g, T, max_ctx)).sub_bytes + 4096;
}
size_t keep_bytes(const Glm5Geometry& g, size_t T, int64_t max_ctx) {
    return glmd::prompt_tiling(dims(g, T, max_ctx)).keep_bytes + 4096;
}

// Every GPU backend runs the code below: HIP through hip_compat (hipBLAS for the cuBLAS calls, the F32 prompt attention
// - mla_prompt_wmma_ok() is false there - and the MMQ clamp on HIP's strata_mmq).

namespace {
// glm-dsa's extra FP16 types (glm_prefill_types.hpp): one thread per 32-value group, ggml's decode order
__global__ void dequant_ext_kernel(int type, const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0,
                                   int64_t rows, int64_t gpr, __half* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * gpr) return;
    const int64_t r = i / gpr, c = i % gpr;
    float y[32];
    ext_group(type, blocks + (size_t) (row0 + r) * row_bytes, c, y);
    __half* o = out + (size_t) r * gpr * 32 + c * 32;
#pragma unroll
    for (int k = 0; k < 32; ++k) o[k] = __float2half_rn(y[k]);
}
}  // namespace

void dequant16(int type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
               cudaStream_t stream) {
    if (!ext_f16_type(type)) {   // Maya's types: dequant_f16 unchanged (every Flash weight)
        strata::kernels::dequant_f16(type, blocks, row0, rows, cols, out, stream);
        return;
    }
    int be = 0, bb = 0;
    ext_geometry(type, be, bb);
    if (rows <= 0) return;
    if (cols % 32 != 0 || cols % be != 0) {
        std::fprintf(stderr, "glm prefill: dequant16: %s rows of %lld values\n", type_name(type), (long long) cols);
        std::exit(1);
    }
    const int64_t row_bytes = cols / be * bb;
    if (type == 19) {   // IQ1_S: Maya's i-quant dequantizer serves it (dequant_f16 does not route it)
        strata::kernels::iq_dequant_f16(type, (const uint8_t*) blocks + (size_t) row0 * row_bytes, rows * cols, out,
                                        stream);
        return;
    }
    const int64_t gpr = cols / 32, total = rows * gpr;
    dequant_ext_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, stream>>>(type, (const uint8_t*) blocks, row_bytes,
                                                                             row0, rows, gpr, (__half*) out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "glm prefill: dequant16 launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

bool built() { return true; }

bool mmq_ok(const Glm5Geometry& g, int type) {
    if (g.arch != GlmArch::GlmDsa) return true;
#if defined(STRATA_USE_HIP)
    // B14.2's check is about CUDA's sm_70 instances (NO_DEVICE_CODE stubs on Volta); a HIP build's MMQ instances are
    // the ones Maya's prompt path multiplies Flash's experts with (mmq::supported, which the caller checks first)
    (void) type;
    return true;
#else
    static int cache[256] = {};   // 0 unknown, 1 ok, 2 refused (one device type per process: the test PC's three V100s)
    if (type < 0 || type >= 256) return false;
    if (cache[type] == 0) {
        int dev = 0, major = 0, minor = 0, optin = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        std::string why;
        const bool ok = strata::prefill::mmq_strata::sm70_check(type, 100 * major + 10 * minor, optin, 32768, why);
        if (!ok) std::fprintf(stderr, "glm-dsa prefill: MMQ type %d on CUDA%d: %s\n", type, dev, why.c_str());
        cache[type] = ok ? 1 : 2;
    }
    return cache[type] == 1;
#endif
}

namespace {

void blas_ck(cublasStatus_t st, const char* what, bool& ok) {
    if (st == CUBLAS_STATUS_SUCCESS) return;
    std::fprintf(stderr, "glm-dsa prefill: %s: cuBLAS status %d\n", what, (int) st);
    ok = false;
}

bool env_on(const char* name, bool dflt) {
    return strata::glmenv::env_flag(name, dflt);
}

const float* f32_of(const WSlot& w) { return w.f32 ? w.f32 : (const float*) w.q; }

// W_kb / W_vb (rows x cols) as FP16 for the absorbed GEMMs: BF16 / F32 narrowed, native Q8_0 dequantized (B10.7)
bool weight16(const WSlot& w, int64_t rows, int64_t cols, uint16_t* dst, cudaStream_t s) {
    if (w.type == glmdsa::kTypeQ8_0) {
        if (w.q == nullptr || cols % 32) return false;
        strata::kernels::dequant_f16(w.type, w.q, 0, rows, cols, dst, s);
        return cudaPeekAtLastError() == cudaSuccess;
    }
    return glmd::weight_f16(w.type == gf::kTypeF32 ? (const void*) f32_of(w) : w.q, w.type, rows * cols, dst, s);
}

// a bump allocator over a device region (256-B aligned)
struct Bump {
    uint8_t* p;
    size_t off = 0;
    template <typename T>
    T* take(size_t n) {
        off = (off + 255) / 256 * 256;
        T* r = (T*) (p + off);
        off += n * sizeof(T);
        return r;
    }
};

// Y[t][N] (ldy) = alpha * X[t][K] . W[N][K]^T over T rows.  F32 weights: SGEMM on X32.  BF16 with f32_path: widened in
// slices, SGEMM on X32.  Otherwise (native / BF16): FP16 slices, tensor-core GEMM on X16 with F32 accumulation.
bool gemm(const Io& io, const WSlot& w, int N, int K, const float* X32, const uint16_t* X16, int ldx, float* Y, int ldy,
          int T, float alpha, bool f32_path) {
    bool ok = true;
    const float zero = 0.0f;
    if (T <= 0) return true;
    if (w.type == gf::kTypeF32) {
        blas_ck(cublasSgemm(io.blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &alpha, f32_of(w), K, X32, ldx, &zero, Y, ldy),
                "sgemm f32", ok);
        return ok;
    }
    if (w.type == gf::kTypeBF16 && f32_path) {
        const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, io.w32_elems / K));
        for (int r0 = 0; r0 < N; r0 += rows) {
            const int n = std::min(rows, N - r0);
            strata::kernels::glmb::bf16_to_f32((const uint16_t*) w.q + (size_t) r0 * K, io.w32, (int64_t) n * K, io.s);
            blas_ck(cublasSgemm(io.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, T, K, &alpha, io.w32, K, X32, ldx, &zero, Y + r0, ldy),
                    "sgemm bf16", ok);
        }
        return ok;
    }
    const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, io.w16_elems / K));
    for (int r0 = 0; r0 < N; r0 += rows) {
        const int n = std::min(rows, N - r0);
        if (w.type == gf::kTypeBF16)
            strata::kernels::glmb::bf16_to_f16((const uint16_t*) w.q + (size_t) r0 * K, io.w16, (int64_t) n * K, io.s);
        else
            dequant16(w.type, w.q, r0, n, K, io.w16, io.s);
#if defined(STRATA_USE_HIP)
        // HIP (S7): hipBLASLt with its tuning tables' row for the shape (glm_prefill.cu hands the prompt path's GEMM
        // over when a table is set; a shape without a row runs plain hipBLAS inside it) - plain hipBLAS here for a
        // strided X or a scaled product
        if (io.lt != nullptr && ldx == K && alpha == 1.0f) {
            io.lt->f16(X16, io.w16, Y + r0, T, n, K, ldy, 0.0f);
            continue;
        }
#endif
        blas_ck(cublasGemmEx(io.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, T, K, &alpha, io.w16, CUDA_R_16F, K, X16, CUDA_R_16F,
                             ldx, &zero, Y + r0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "gemm f16", ok);
    }
    // B15.1: rows of a guarded X16 were divided by their power-of-two scale: multiply Y's rows back (no-op otherwise)
    if (ok && !glmd::f16_guard_after(X16, ldx, Y, ldy, N, T, 0.0f, io.s)) ok = false;
    return ok;
}

// the device rope tables in io.keep, after the selection rows (filled once per chunk; cheap)
struct Keep {
    int* sel = nullptr;
    float* inv = nullptr;       // qk_rope / 2
    float* idx_inv = nullptr;   // idx_rope / 2
};
Keep keep_of(const Io& io, const glmd::PromptTiling& t) {
    const Glm5Geometry& g = *io.g;
    Bump b{io.keep};
    Keep k;
    k.sel = b.take<int>((size_t) io.T * g.idx_top_k);
    (void) t;
    k.inv = b.take<float>((size_t) std::max(1, g.qk_rope / 2));
    k.idx_inv = b.take<float>((size_t) std::max(1, g.idx_rope / 2));
    return k;
}
#if defined(STRATA_USE_HIP)
// HIP: the tables' host source is pinned and written once per geometry.  A copy from pageable memory makes HIP stage
// it (a host wait per copy, two a layer), and Windows' HIP runtime submits queued work only when the host waits, so a
// copy may read its source late: this one never changes once written.  (Never freed: a few hundred bytes a geometry.)
const float* rope_host(const Glm5Geometry& g, size_t n) {
    struct Tab {
        int qk, idx;
        double theta;
        float* p;
    };
    static std::mutex mu;
    static std::vector<Tab> tabs;
    std::lock_guard<std::mutex> lock(mu);
    for (const Tab& t : tabs)
        if (t.qk == g.qk_rope && t.idx == g.idx_rope && t.theta == g.rope_theta) return t.p;
    float* p = nullptr;
    if (cudaHostAlloc((void**) &p, n * sizeof(float), cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        p = new float[n];   // (pageable: correct, only slower)
    }
    std::fill(p, p + n, 0.0f);
    if (g.qk_rope > 0) glmd::rope_inv_freq(p, g.qk_rope, g.rope_theta);
    if (g.idx_rope > 0) glmd::rope_inv_freq(p + std::max(1, g.qk_rope / 2), g.idx_rope, g.rope_theta);
    tabs.push_back(Tab{g.qk_rope, g.idx_rope, g.rope_theta, p});
    return p;
}
#endif

void upload_rope(const Io& io, const Keep& k) {
    const Glm5Geometry& g = *io.g;
#if defined(STRATA_USE_HIP)
    const float* h = rope_host(g, (size_t) std::max(1, g.qk_rope / 2) + (size_t) std::max(1, g.idx_rope / 2));
    cudaMemcpyAsync(k.inv, h, (size_t) std::max(1, g.qk_rope / 2) * 4, cudaMemcpyHostToDevice, io.s);
    cudaMemcpyAsync(k.idx_inv, h + std::max(1, g.qk_rope / 2), (size_t) std::max(1, g.idx_rope / 2) * 4,
                    cudaMemcpyHostToDevice, io.s);
#else
    static thread_local std::vector<float> h;
    h.assign((size_t) std::max(1, g.qk_rope / 2) + (size_t) std::max(1, g.idx_rope / 2), 0.0f);
    if (g.qk_rope > 0) glmd::rope_inv_freq(h.data(), g.qk_rope, g.rope_theta);
    if (g.idx_rope > 0) glmd::rope_inv_freq(h.data() + std::max(1, g.qk_rope / 2), g.idx_rope, g.rope_theta);
    // (pageable source: the copy is staged before the call returns, so the thread_local buffer may be reused)
    cudaMemcpyAsync(k.inv, h.data(), (size_t) std::max(1, g.qk_rope / 2) * 4, cudaMemcpyHostToDevice, io.s);
    cudaMemcpyAsync(k.idx_inv, h.data() + std::max(1, g.qk_rope / 2), (size_t) std::max(1, g.idx_rope / 2) * 4,
                    cudaMemcpyHostToDevice, io.s);
#endif
}

// STRATA_GLMD_IDX_CHECK=1 (a debug aid): the other scorer's selection of the same sub-batch (F32 vs WMMA), compared on
// the host with the one the layer uses - the per-row overlap of the 2,048-cell sets, one line per layer and sub-batch.
// The layer goes on with its own selection.
template <typename Select>
bool check_selection(const Io& io, int il, int64_t pq, int tn, int64_t n_kv, int top, const int* sel, bool tc,
                     const Select& select) {
    int dev = 0;
    cudaGetDevice(&dev);
    static int* other[64] = {};
    static size_t cap[64] = {};
    const size_t n = (size_t) tn * top;
    if (cap[dev & 63] < n) {
        if (other[dev & 63]) cudaFree(other[dev & 63]);
        other[dev & 63] = nullptr;
        cap[dev & 63] = 0;
        if (cudaMalloc(&other[dev & 63], n * sizeof(int)) != cudaSuccess) return false;
        cap[dev & 63] = n;
    }
    if (!select(!tc, other[dev & 63])) return false;
    std::vector<int> a(n), b(n);
    cudaMemcpyAsync(a.data(), sel, n * sizeof(int), cudaMemcpyDeviceToHost, io.s);
    cudaMemcpyAsync(b.data(), other[dev & 63], n * sizeof(int), cudaMemcpyDeviceToHost, io.s);
    if (cudaStreamSynchronize(io.s) != cudaSuccess) return false;
    double sum = 0.0, mn = 1.0;
    int same = 0;
    for (int t = 0; t < tn; ++t) {
        const int* x = &a[(size_t) t * top];
        const int* y = &b[(size_t) t * top];
        int nx = 0, ny = 0, common = 0;
        while (nx < top && x[nx] >= 0) ++nx;
        while (ny < top && y[ny] >= 0) ++ny;
        for (int i = 0, j = 0; i < nx && j < ny;) {   // both ascending
            if (x[i] == y[j]) {
                ++common;
                ++i;
                ++j;
            } else if (x[i] < y[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        const double ov = nx > 0 ? (double) common / nx : 1.0;
        sum += ov;
        mn = std::min(mn, ov);
        same += nx == ny && common == nx;
    }
    std::fprintf(stderr,
                 "glm prefill idx-check: CUDA%d layer %d rows %lld..%lld (%lld keys): the %s selection vs the %s one: "
                 "overlap mean %.4f min %.4f, %.1f%% of rows identical\n",
                 dev, il, (long long) pq, (long long) (pq + tn - 1), (long long) n_kv, tc ? "WMMA" : "F32",
                 tc ? "F32" : "WMMA", sum / std::max(1, tn), mn, 100.0 * same / std::max(1, tn));
    return true;
}

}  // namespace

bool res_read(const Io& io, const float* block_out, const float* norm_w, std::string& err) {
    if (!glmd::res_rows(io.R, block_out, norm_w, io.g->norm_eps, io.T, io.g->n_embd, io.x, io.x16, io.s)) {
        err = "glm-dsa prefill: the plain residual read refused its shape";
        return false;
    }
    // B15.1: x16 opens the phase (the mixer's or the FFN's): guard it, drop the previous phase's bindings
    if (!glmd::f16_guard_rows(io.x, io.g->n_embd, 0, 1, io.g->n_embd, io.T, io.x16, io.s, true)) {
        err = "glm-dsa prefill: the FP16 range guard of x16 failed";
        return false;
    }
    return true;
}

bool res_close(const Io& io, const float* ffn, std::string& err) {
    if (!glmd::add_rows(io.R, ffn, (int64_t) io.T * io.g->n_embd, io.s)) {
        err = "glm-dsa prefill: the residual close failed";
        return false;
    }
    return true;
}

bool route(const Io& io, const void* router_w, int router_type, const float* bias, float* logits, int* ids, float* w,
           std::string& err) {
    const Glm5Geometry& g = *io.g;
    WSlot r;
    r.type = router_type;
    r.q = router_w;
    if (router_type == gf::kTypeF32) r.f32 = (const float*) router_w;
    if (router_w == nullptr || !glmdsa::float_type(router_type)) {
        err = "glm-dsa prefill: the router is neither F32 nor BF16";
        return false;
    }
    if (!gemm(io, r, g.n_expert, g.n_embd, io.x, io.x16, g.n_embd, logits, g.n_expert, io.T, 1.0f, true) ||
        !glmd::route_rows(logits, bias, g.n_expert, g.n_exp_used, g.w_scale, g.norm_w != 0, glmd::kRouteNormHf, io.T, ids,
                          w, io.s)) {
        err = "glm-dsa prefill: the router failed";
        return false;
    }
    return true;
}

// L1-6 KV streaming: layer il's caches on the stage - the prefix [0, p0) from the host copies (with_prefix), the
// stage as lat / k_pe (identity rows); kv_stage_out: rows [p0, p0 + n) back to the host copies, the map reset
bool kv_stage_in(const Io& io, int il, bool with_prefix, void*& lat, uint16_t*& k_pe, std::string& err) {
    const KvIo& k = *io.kv;
    if (io.p0 + io.T > k.stage_rows) {
        err = "glm-dsa prefill: the KV stage holds " + std::to_string(k.stage_rows) + " positions, the chunk ends at " +
              std::to_string(io.p0 + io.T);
        return false;
    }
    uint8_t* sl = k.stage;
    uint8_t* sk = k.stage + (size_t) k.stage_rows * k.lat_row;
    if (with_prefix && io.p0 > 0) {   // (zero-copy loads: a DMA copy would wait behind the expert copy lane)
        gf::glm_kv_pull(sl, k.lat_host[il], (size_t) io.p0 * k.lat_row, io.s);
        if (k.kpe_host != nullptr && k_pe != nullptr) gf::glm_kv_pull(sk, k.kpe_host[il], (size_t) io.p0 * k.kpe_row, io.s);
    }
    lat = sl;
    if (k.kpe_host != nullptr && k_pe != nullptr) k_pe = (uint16_t*) sk;
    return true;
}
bool kv_stage_out(const Io& io, int il, int64_t n, bool with_kpe, std::string& err) {
    const KvIo& k = *io.kv;
    const uint8_t* sl = k.stage;
    const uint8_t* sk = k.stage + (size_t) k.stage_rows * k.lat_row;
    if (n > 0 &&
        (cudaMemcpyAsync(k.lat_host[il] + (size_t) io.p0 * k.lat_row, sl + (size_t) io.p0 * k.lat_row,
                         (size_t) n * k.lat_row, cudaMemcpyDefault, io.s) != cudaSuccess ||
         (with_kpe && k.kpe_host != nullptr &&
          cudaMemcpyAsync(k.kpe_host[il] + (size_t) io.p0 * k.kpe_row, sk + (size_t) io.p0 * k.kpe_row,
                          (size_t) n * k.kpe_row, cudaMemcpyDefault, io.s) != cudaSuccess))) {
        cudaGetLastError();
        err = "glm-dsa prefill: the KV write-back";
        return false;
    }
    strata::kernels::kv_stream_reset(k.map[il], (void*) io.s);
    return true;
}

bool dsa_layer(const Io& io, int il, std::string& err) {
    const Glm5Geometry& g = *io.g;
    if (io.dsa == nullptr || (size_t) il >= io.dsa->L.size() || !io.dsa->L[(size_t) il].bound) {
        err = "glm-dsa prefill: layer " + std::to_string(il) + " has no glm-dsa weights on this part";
        return false;
    }
    const glmdsa::GlmDsaLayerW& L = io.dsa->L[(size_t) il];
    const GlmStateLayout& lay = *io.lay;
    if (!lay.holds(il) || !lay.is_dsa(il)) {
        err = "glm-dsa prefill: layer " + std::to_string(il) + " has no latent cache here";
        return false;
    }
    const int E = g.n_embd, H = g.n_head, QL = g.q_lora, KV = g.kv_lora, R = g.qk_rope, NOPE = g.qk_nope;
    const int QH = g.qk_head(), KVR = g.kv_row(), VH = g.v_head, IH = g.idx_heads, ID = g.idx_key, TOP = g.idx_top_k;
    const int64_t chunk_end = io.p0 + io.T;
    // the union was carved by scratch_bytes(g, min(T, sub), max_ctx): the same plan, so every buffer below fits
    const glmd::PromptTiling tl =
        glmd::prompt_tiling(dims(g, (size_t) std::min(io.T, io.sub), std::max<int64_t>(io.max_ctx, chunk_end)));
    const Keep keep = keep_of(io, tl);
    upload_rope(io, keep);
    // the selection of a shared layer is its source's (a full layer of this part, earlier in the chunk)
    const bool full = L.full;
    void* lat = (void*) (io.state + lay.dsa_lat[(size_t) il]);
    uint16_t* k_pe = lay.has_kpe(il) ? (uint16_t*) (io.state + lay.dsa_kpe[(size_t) il]) : nullptr;
    float* ik_cache = full && lay.has_index_cache(il) ? io.state + lay.dsa_ik[(size_t) il] : nullptr;
    const bool kv_on = io.kv != nullptr && io.kv->lat_host[il] != nullptr;   // L1-6 KV streaming: on the stage
    if (kv_on && !kv_stage_in(io, il, true, lat, k_pe, err)) return false;
    const int ik_f16 = lay.ik_bytes == 2 ? 1 : 0;   // B10.8: the layout's key dtype (the decode mixer reads the same)
    if ((R > 0 && k_pe == nullptr) || (full && ik_cache == nullptr)) {
        err = "glm-dsa prefill: layer " + std::to_string(il) + " lacks its k_pe or index-key cache";
        return false;
    }
    static const bool f16_env = env_on("STRATA_GLMD_PF_F16", true);
    // a native Q8_0 W_kb / W_vb has no F32 path here: its products always run on the FP16 dequant
    const bool f16_products = f16_env || L.k_b.type == glmdsa::kTypeQ8_0 || L.v_b.type == glmdsa::kTypeQ8_0;
    static const bool want_wmma = env_on("STRATA_GLMD_PF_WMMA", true);
    static const bool want_h64 = env_on("STRATA_GLMD_PF_H64", true);   // 0: the (row, 16 heads) WMMA kernel
    const bool wmma = want_wmma && KV == 512 && (R == 0 || R == 64) && glmd::mla_prompt_wmma_ok();
    // the indexer's score tiles on tensor cores (FP16 queries / keys, F32 sums; STRATA_GLMD_IDX_WMMA=0: the F32 scorer)
    static const bool want_idx_tc = env_on("STRATA_GLMD_IDX_WMMA", true);
    static const bool idx_check = env_on("STRATA_GLMD_IDX_CHECK", false);
    const bool idx_tc_ok = full && ((uintptr_t) ik_cache & 15u) == 0 && glmd::index_score_wmma_fits(ik_f16, IH, ID);
    const bool idx_tc = want_idx_tc && idx_tc_ok;
    // whether the 64-head kernel takes this layer (shape, the caches' alignment, the device): it sets the crossover
    bool h64 = false;
    if (wmma && want_h64) {
        glmd::MlaPrompt pa;
        pa.q_abs = pa.q = (const float*) io.uni;   // (the sub-batch rows come from io.uni: 256-B aligned)
        pa.ctx = (float*) io.uni;
        pa.q_stride = QH;
        pa.q_pe_off = NOPE;
        pa.lat = lat;
        pa.lat_q8 = io.lat_q8 ? 1 : 0;
        pa.k_pe = k_pe;
        pa.walk = glmd::AttnWalk::Dense;
        pa.T = 1;
        pa.n_head = H;
        pa.kv_lora = KV;
        pa.qk_rope = R;
        h64 = glmd::mla_prompt_h64_fits(pa);
        static bool told = false;
        if (!h64 && !told) {
            told = true;
            std::fprintf(stderr, "glm-dsa prefill: layer %d: the 64-head attention kernel does not take this layer "
                                 "(heads %d, latent %d, rope %d, caches 16-byte aligned: %s) - the 16-head kernel runs\n",
                         il, H, KV, R, (((uintptr_t) lat | (uintptr_t) k_pe) & 15u) == 0 ? "yes" : "no");
        }
    }
#if defined(STRATA_USE_HIP)
    // RDNA 3 / 3.5 / 4: the WMMA kernel (S7).  Every HIP kernel gathers from 2049 keys on, as the 64-head one: a tile
    // costs the same whichever cells fill it, and the gather walk's are the row's selection packed - never more tiles
    // than the dense-masked walk's (Radeon 8065S, 2,048 queries: RDNA 66-79 ms at any context against 75-898 dense-
    // masked; the F32 kernel 497-581 against 534-5,203)
    const bool rdna = KV == 512 && (R == 0 || R == 64) && glmd::mla_prompt_rdna_ok();
    constexpr bool gather_from_top = true;
#else
    constexpr bool rdna = false, gather_from_top = false;
#endif
    const double ratio = glmd::gather_ratio(h64 || gather_from_top ? glmd::kH64GatherRatio : 4.0);
    const float scale = glmd::attn_scale(QH);
    const float one = 1.0f, zero = 0.0f;
    bool ok = true;
    for (int t0 = 0; t0 < io.T && ok; t0 += tl.q_sub) {
        const int tn = std::min(tl.q_sub, io.T - t0);
        const int64_t pq = io.p0 + t0, n_kv = pq + tn;
        const float* xs = io.x + (size_t) t0 * E;
        const uint16_t* x16 = io.x16 + (size_t) t0 * E;
        Bump b{io.uni};
        float* qr_raw = b.take<float>((size_t) tn * QL);
        float* qr = b.take<float>((size_t) tn * QL);
        uint16_t* qr16 = b.take<uint16_t>((size_t) tn * QL);
        float* kv_raw = b.take<float>((size_t) tn * KVR);
        float* ik_raw = b.take<float>((size_t) tn * ID);
        float* iw = b.take<float>((size_t) tn * IH);
        float* q = b.take<float>((size_t) tn * H * QH);
        float* iq = b.take<float>((size_t) tn * IH * ID);
        float* tile = b.take<float>((size_t) tn * tl.key_block);
        int* run_c[2] = {b.take<int>((size_t) tn * TOP), b.take<int>((size_t) tn * TOP)};
        float* run_s[2] = {b.take<float>((size_t) tn * TOP), b.take<float>((size_t) tn * TOP)};
        int* run_n[2] = {b.take<int>((size_t) tn), b.take<int>((size_t) tn)};
        uint16_t* q16 = b.take<uint16_t>((size_t) tn * H * std::max(NOPE, KV));
        float* q_abs = b.take<float>((size_t) tn * H * KV);
        float* ctx = b.take<float>((size_t) tn * H * KV);
        float* attn = b.take<float>((size_t) tn * H * VH);
        uint16_t* attn16 = b.take<uint16_t>((size_t) tn * H * VH);
        // the mask last: its size grows with the context, so a sub-batch's FP16 operands above (q16, attn16) keep the
        // addresses of the one before - each stays ONE binding of the B15.1 guard (when they moved with the mask, a
        // layer of 7+ sub-batches - a 6.7K prompt in 1,024-token sub-batches - filled its 16 slots and ran unguarded)
        const int64_t words = (n_kv + 31) / 32;
        uint32_t* mask = b.take<uint32_t>((size_t) tn * words);

        // ---- projections
        ok = ok && gemm(io, L.q_a, QL, E, xs, x16, E, qr_raw, QL, tn, 1.0f, false);
        ok = ok && gemm(io, L.kv_a, KVR, E, xs, x16, E, kv_raw, KVR, tn, 1.0f, false);
        if (full) {
            ok = ok && gemm(io, L.idx_k, ID, E, xs, x16, E, ik_raw, ID, tn, 1.0f, true);
            ok = ok && gemm(io, L.idx_proj, IH, E, xs, x16, E, iw, IH, tn, 1.0f / std::sqrt((float) (ID * IH)), true);
        }
        // ---- prep: qr, latent, k_pe, keys at pq + t
        glmd::DsaPrepRows pr;
        pr.a.qr_raw = qr_raw;
        pr.a.q_a_norm = L.q_a_norm;
        pr.a.qr = qr;
        pr.a.q_lora = QL;
        pr.a.kv_raw = kv_raw;
        pr.a.kv_norm = L.kv_a_norm;
        pr.a.lat = lat;
        pr.a.lat_q8 = io.lat_q8 ? 1 : 0;
        pr.a.kv_lora = KV;
        pr.a.k_pe = k_pe;
        pr.a.qk_rope = R;
        pr.a.inv_freq = keep.inv;
        if (full) {
            pr.a.ik_raw = ik_raw;
            pr.a.ik_ln_w = L.idx_k_norm_w;
            pr.a.ik_ln_b = L.idx_k_norm_b;
            pr.a.idx_ln_eps = g.idx_ln_eps;
            pr.a.ik_cache = ik_cache;
            pr.a.ik_f16 = ik_f16;
            pr.a.ik_had = lay.ik_had;   // B14.11: rotated keys, as the decode mixer writes them
            pr.a.idx_dim = ID;
            pr.a.idx_rope = g.idx_rope;
            pr.a.idx_inv_freq = keep.idx_inv;
        }
        pr.a.p = (int) pq;
        pr.a.eps = g.norm_eps;
        pr.T = tn;
        pr.kv_ld = KVR;
        pr.qr16 = qr16;
        ok = ok && glmd::dsa_prep_rows(pr, io.s);
        ok = ok && glmd::f16_guard_rows(qr, QL, 0, 1, QL, tn, qr16, io.s);   // B15.1
        // ---- queries
        ok = ok && gemm(io, L.q_b, H * QH, QL, qr, qr16, QL, q, H * QH, tn, 1.0f, false);
        if (full) ok = ok && gemm(io, L.idx_q_b, IH * ID, QL, qr, qr16, QL, iq, IH * ID, tn, 1.0f, true);
        glmd::DsaRopeQRows rq;
        rq.a.q = q;
        rq.a.n_head = H;
        rq.a.q_stride = QH;
        rq.a.rot_off = NOPE;
        rq.a.n_rot = R;
        rq.a.inv_freq = keep.inv;
        if (full) {
            rq.a.iq = iq;
            rq.a.idx_heads = IH;
            rq.a.idx_dim = ID;
            rq.a.idx_rope = g.idx_rope;
            rq.a.idx_inv_freq = keep.idx_inv;
            rq.a.iq_had = lay.ik_had;   // B14.11: queries rotated like the keys they score
        }
        rq.a.p = (int) pq;
        rq.T = tn;
        rq.q_row = (int64_t) H * QH;
        rq.iq_row = (int64_t) IH * ID;
        ok = ok && glmd::dsa_rope_q_rows(rq, io.s);
        // ---- selection (full layers; dense below 2049 keys needs none)
        const glmd::AttnWalk walk = glmd::attn_walk(n_kv, TOP, ratio);
        int* sel = keep.sel + (size_t) t0 * TOP;
        // the tiled selection of this sub-batch into out (the F32 scorer, or the WMMA one)
        const auto select = [&](bool tc, int* out) {
            const int tiles = (int) ((n_kv + tl.key_block - 1) / tl.key_block);
            const int* ci = nullptr;
            const float* si = nullptr;
            const int* ni = nullptr;
            bool sok = true;
            for (int k = 0; k < tiles && sok; ++k) {
                const int64_t k0 = (int64_t) k * tl.key_block;
                const int nk = (int) std::min<int64_t>(tl.key_block, n_kv - k0);
                sok = tc ? glmd::index_score_tile_wmma(iq, iw, ik_cache, ik_f16, IH, ID, tn, k0, nk, tile, tl.key_block,
                                                       io.s)
                         : glmd::index_score_tile_keys(iq, iw, ik_cache, ik_f16, IH, ID, tn, k0, nk, tile, tl.key_block,
                                                       io.s);
                glmd::TopkMerge m;
                m.score = tile;
                m.ld = tl.key_block;
                m.T = tn;
                m.p0 = pq;
                m.k0 = k0;
                m.nk = nk;
                m.top_k = TOP;
                m.in_cell = ci;
                m.in_score = si;
                m.in_n = ni;
                m.out_cell = k == tiles - 1 ? out : run_c[k & 1];
                m.out_score = run_s[k & 1];
                m.out_n = run_n[k & 1];
                sok = sok && glmd::topk_merge_tile(m, io.s);
                ci = m.out_cell;
                si = m.out_score;
                ni = m.out_n;
            }
            return sok;
        };
        if (full && walk != glmd::AttnWalk::Dense) {
            ok = ok && select(idx_tc, sel);
            if (ok && idx_check && idx_tc_ok) ok = check_selection(io, il, pq, tn, n_kv, TOP, sel, idx_tc, select);
        }
        if (walk == glmd::AttnWalk::DenseMasked) ok = ok && glmd::sel_to_mask(sel, TOP, tn, n_kv, mask, words, io.s);
        // ---- q_abs[t][h] = W_kb_h^T q_nope[t][h] (B9.7: FP16 strided-batched on CUDA)
        if (f16_products) {
            ok = ok && weight16(L.k_b, (int64_t) H * KV, NOPE, io.w16, io.s);
            ok = ok && glmd::pack_f16(q, (int64_t) H * QH, QH, H, NOPE, tn, q16, io.s);
            ok = ok && glmd::f16_guard_rows(q, (int64_t) H * QH, QH, H, NOPE, tn, q16, io.s);   // B15.1
            blas_ck(cublasGemmStridedBatchedEx(io.blas, CUBLAS_OP_T, CUBLAS_OP_N, KV, tn, NOPE, &one, io.w16, CUDA_R_16F,
                                               NOPE, (long long) KV * NOPE, q16, CUDA_R_16F, H * NOPE, NOPE, &zero, q_abs,
                                               CUDA_R_32F, H * KV, KV, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "q_abs f16", ok);
            ok = ok && glmd::f16_guard_after(q16, (int64_t) H * NOPE, q_abs, (int64_t) H * KV, H * KV, tn, 0.0f, io.s);
        } else {
            const float* wkb = f32_of(L.k_b);
            if (L.k_b.type == gf::kTypeBF16) {
                strata::kernels::glmb::bf16_to_f32((const uint16_t*) L.k_b.q, io.w32, (int64_t) H * KV * NOPE, io.s);
                wkb = io.w32;
            }
            blas_ck(cublasSgemmStridedBatched(io.blas, CUBLAS_OP_T, CUBLAS_OP_N, KV, tn, NOPE, &one, wkb, NOPE,
                                              (long long) KV * NOPE, q, H * QH, QH, &zero, q_abs, H * KV, KV, H),
                    "q_abs", ok);
        }
        // ---- attention
        glmd::MlaPrompt a;
        a.q_abs = q_abs;
        a.q = q;
        a.q_stride = QH;
        a.q_pe_off = NOPE;
        a.lat = lat;
        a.lat_q8 = io.lat_q8 ? 1 : 0;
        a.k_pe = k_pe;
        a.walk = walk;
        a.sel = sel;
        a.mask = mask;
        a.mask_words = words;
        a.top_k = TOP;
        a.p0 = pq;
        a.T = tn;
        a.n_head = H;
        a.kv_lora = KV;
        a.qk_rope = R;
        a.scale = scale;
        a.ctx = ctx;
        if (ok)
            ok = (h64 && glmd::mla_prompt_h64_fits(a) && glmd::mla_prompt_h64(a, io.s)) ||
                 (wmma && glmd::mla_prompt_wmma(a, io.s)) || (rdna && glmd::mla_prompt_rdna(a, io.s)) ||
                 glmd::mla_prompt_f32(a, io.s);
        // ---- out[t][h] = W_vb_h ctx[t][h], then attn_output
        if (f16_products) {
            ok = ok && weight16(L.v_b, (int64_t) H * VH, KV, io.w16, io.s);
            ok = ok && glmd::pack_f16(ctx, (int64_t) H * KV, KV, H, KV, tn, q16, io.s);
            ok = ok && glmd::f16_guard_rows(ctx, (int64_t) H * KV, KV, H, KV, tn, q16, io.s);   // B15.1
            blas_ck(cublasGemmStridedBatchedEx(io.blas, CUBLAS_OP_T, CUBLAS_OP_N, VH, tn, KV, &one, io.w16, CUDA_R_16F, KV,
                                               (long long) VH * KV, q16, CUDA_R_16F, H * KV, KV, &zero, attn, CUDA_R_32F,
                                               H * VH, VH, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "mla out f16", ok);
            ok = ok && glmd::f16_guard_after(q16, (int64_t) H * KV, attn, (int64_t) H * VH, H * VH, tn, 0.0f, io.s);
        } else {
            const float* wvb = f32_of(L.v_b);
            if (L.v_b.type == gf::kTypeBF16) {
                strata::kernels::glmb::bf16_to_f32((const uint16_t*) L.v_b.q, io.w32, (int64_t) H * VH * KV, io.s);
                wvb = io.w32;
            }
            blas_ck(cublasSgemmStridedBatched(io.blas, CUBLAS_OP_T, CUBLAS_OP_N, VH, tn, KV, &one, wvb, KV,
                                              (long long) VH * KV, ctx, H * KV, KV, &zero, attn, H * VH, VH, H),
                    "mla out", ok);
        }
        if (!io.skip_output) {
            ok = ok && glmd::pack_f16(attn, (int64_t) H * VH, VH, H, VH, tn, attn16, io.s);
            ok = ok && glmd::f16_guard_rows(attn, (int64_t) H * VH, VH, H, VH, tn, attn16, io.s);   // B15.1
            ok = ok && gemm(io, L.out, E, H * VH, attn, attn16, H * VH, io.mixer + (size_t) t0 * E, E, tn, 1.0f, false);
        }
    }
    if (!ok) {
        err = "glm-dsa prefill: layer " + std::to_string(il) + " failed (a refused shape or a cuBLAS error, logged above)";
        return false;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm-dsa prefill: ") + cudaGetErrorString(e);
        return false;
    }
    if (kv_on && !kv_stage_out(io, il, io.T, k_pe != nullptr, err)) return false;   // L1-6
    return true;
}

bool mtp_fill(const Io& io, const MtpIo& m, std::string& err) {
    const Glm5Geometry& g = *io.g;
    if (m.Tv <= 0) return true;
    if (io.dsa == nullptr || m.il < 0 || (size_t) m.il >= io.dsa->L.size() || !io.dsa->L[(size_t) m.il].bound) {
        err = "glm-dsa prefill: the NextN block has no weights here";
        return false;
    }
    const glmdsa::GlmDsaLayerW& L = io.dsa->L[(size_t) m.il];
    const GlmStateLayout& lay = *io.lay;
    const int E = g.n_embd, KVR = g.kv_row(), ID = g.idx_key;
    const glmd::PromptTiling tl =
        glmd::prompt_tiling(dims(g, (size_t) std::min(io.T, io.sub), std::max<int64_t>(io.max_ctx, io.p0 + io.T)));
    const Keep keep = keep_of(io, tl);
    upload_rope(io, keep);
    void* lat = (void*) (io.state + lay.dsa_lat[(size_t) m.il]);
    uint16_t* k_pe = lay.has_kpe(m.il) ? (uint16_t*) (io.state + lay.dsa_kpe[(size_t) m.il]) : nullptr;
    float* ik_cache = L.full && lay.has_index_cache(m.il) ? io.state + lay.dsa_ik[(size_t) m.il] : nullptr;
    const bool kv_on = io.kv != nullptr && io.kv->lat_host[m.il] != nullptr;   // L1-6: writes only, no prefix
    if (kv_on && !kv_stage_in(io, m.il, false, lat, k_pe, err)) return false;
    if (m.eh.type == gf::kTypeF32 || m.emb_host == nullptr) {   // (the FP16 concat feeds native / BF16 eh_proj only)
        err = "glm-dsa prefill: the NextN eh_proj is F32 or no embeddings were given";
        return false;
    }
    bool ok = true;
    // in sub-batches: h, the FP16 concat, hid, x, x16, kv_raw, ik_raw (+ a dummy q side for prep)
    const int sub = std::max(1, std::min(m.Tv, tl.q_sub));
    for (int t0 = 0; t0 < m.Tv && ok; t0 += sub) {
        const int tn = std::min(sub, m.Tv - t0);
        Bump b{io.uni};
        float* emb = b.take<float>((size_t) tn * E);
        float* h = b.take<float>((size_t) tn * E);
        uint16_t* cat16 = b.take<uint16_t>((size_t) tn * 2 * E);
        float* hid = b.take<float>((size_t) tn * E);
        float* x = b.take<float>((size_t) tn * E);
        uint16_t* x16 = b.take<uint16_t>((size_t) tn * E);
        float* kv_raw = b.take<float>((size_t) tn * KVR);
        float* ik_raw = b.take<float>((size_t) tn * ID);
        float* qdummy = b.take<float>((size_t) tn * 32);
        float* qnorm1 = b.take<float>(32);
        cudaMemcpyAsync(emb, m.emb_host + (size_t) t0 * E, (size_t) tn * E * sizeof(float), cudaMemcpyHostToDevice, io.s);
        // B15.1: the concat below is produced unguarded (RMS-normed halves): no earlier binding may cover it
        glmd::f16_guard_reset(io.s);
        // h = RMS(R) * output_norm (R unchanged), [enorm(emb) | hnorm(h)] in FP16
        ok = ok && glmd::res_rows(io.R + (size_t) t0 * E, nullptr, m.output_norm, g.norm_eps, tn, E, h, nullptr, io.s);
        strata::kernels::glmb::mtp_in_rows(emb, h, m.enorm, m.hnorm, g.norm_eps, tn, E, cat16, io.s);
        ok = ok && gemm(io, m.eh, E, 2 * E, nullptr, cat16, 2 * E, hid, E, tn, 1.0f, false);
        ok = ok && glmd::res_rows(hid, nullptr, m.attn_norm, g.norm_eps, tn, E, x, x16, io.s);
        ok = ok && glmd::f16_guard_rows(x, E, 0, 1, E, tn, x16, io.s);   // B15.1
        ok = ok && gemm(io, L.kv_a, KVR, E, x, x16, E, kv_raw, KVR, tn, 1.0f, false);
        if (ik_cache) ok = ok && gemm(io, L.idx_k, ID, E, x, x16, E, ik_raw, ID, tn, 1.0f, true);
        // the cache half of the prep (a 32-wide dummy query side keeps dsa_prep_rows' shape rules)
        cudaMemsetAsync(qdummy, 0, (size_t) tn * 32 * 4, io.s);
        cudaMemsetAsync(qnorm1, 0, 32 * 4, io.s);
        glmd::DsaPrepRows pr;
        pr.a.qr_raw = qdummy;
        pr.a.q_a_norm = qnorm1;
        pr.a.q_lora = 32;
        pr.a.kv_raw = kv_raw;
        pr.a.kv_norm = L.kv_a_norm;
        pr.a.lat = lat;
        pr.a.lat_q8 = io.lat_q8 ? 1 : 0;
        pr.a.kv_lora = g.kv_lora;
        pr.a.k_pe = k_pe;
        pr.a.qk_rope = g.qk_rope;
        pr.a.inv_freq = keep.inv;
        if (ik_cache) {
            pr.a.ik_raw = ik_raw;
            pr.a.ik_ln_w = L.idx_k_norm_w;
            pr.a.ik_ln_b = L.idx_k_norm_b;
            pr.a.idx_ln_eps = g.idx_ln_eps;
            pr.a.ik_cache = ik_cache;
            pr.a.ik_f16 = lay.ik_bytes == 2 ? 1 : 0;   // B10.8: as the decode mixer's NextN block writes them
            pr.a.ik_had = lay.ik_had;                  // B14.11: likewise rotated
            pr.a.idx_dim = ID;
            pr.a.idx_rope = g.idx_rope;
            pr.a.idx_inv_freq = keep.idx_inv;
        }
        pr.a.p = (int) (io.p0 + t0);
        pr.a.eps = g.norm_eps;
        pr.T = tn;
        pr.kv_ld = KVR;
        ok = ok && glmd::dsa_prep_rows(pr, io.s);
    }
    if (!ok) {
        err = "glm-dsa prefill: the NextN cache fill failed";
        return false;
    }
    if (ok && kv_on && !kv_stage_out(io, m.il, m.Tv, k_pe != nullptr, err)) return false;   // L1-6
    return true;
}

void mmq_clamp(bool on, void* xq, int64_t rows, int64_t cols, int type, cudaStream_t s) {
    if (!on || rows <= 0) return;
    const int halves = strata::prefill::mmq_strata::ds_halves(type);
    if (halves <= 0) return;
    glmd::q8_mmq_clamp(xq, strata::prefill::mmq_strata::blocks(rows, cols), strata::prefill::mmq_strata::block_bytes(),
                       halves, s);
}

}  // namespace strata::core::glmpf
