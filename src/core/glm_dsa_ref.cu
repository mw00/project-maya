// src/core/glm_dsa_ref.cu - the glm-dsa F32 reference forward (glm-dsa B3.5): Glm5Model::step_layers_dsa.
//
// The correctness-first decode of GLM-5.3 (744B-A40B, GGUF arch glm-dsa) that the fast path (B4) and every parity
// gate are measured against.  STRATA_GLM_SLOW=1 selects it (glm_forward_available lets glm-dsa through only on that
// path); step_layers() dispatches here for glm-dsa with one line, so Flash (glm5-next) never reaches this file.
//
// One token at position p through this part's layers [l0_, l1_) (GLM-5.3-SPEC.md dataflow; llama.cpp
// src/models/glm-dsa.cpp), plain residual R (hc = 1, the first n_embd floats of state_):
//
//   x      = rms(R) * attn_norm
//   qr     = rms(W_q_a x) * q_a_norm                                         [6144 -> 2048]
//   full layer:  ik(p) = rope(LN(W_idx_k x)) -> the F32 index-key cache;  iq = rope(W_idx_q_b qr);
//                w = W_idx_proj x / sqrt(idx_heads * idx_key);  sel = every cell 0..p while p + 1 <= top_k,
//                else the top-k of sum_h w_h relu(iq_h . ik(s))
//   shared layer: sel = the previous full layer's selection for this token
//   q      = W_q_b qr, NORM rope on each head's last qk_rope dims            [2048 -> 64 x (192 + 64)]
//   kv(p)  = [ rms(c_kv) * kv_a_norm | rope(k_pe) ] = W_kv_a_mqa x, the F32 latent cache row (576)
//   attn   = absorbed MLA over sel (glmd::ref_mla_absorb_q + ref_mla_attention), scale 1/16 (glmd::kGlm53KqScale)
//   R     += W_o attn
//   x      = rms(R) * ffn_norm
//   dense lead layer: ffn = W_down (silu(W_gate x) * W_up x)
//   MoE layer: p = sigmoid(W_router x); top-8 by p + exp_probs_b (bias selects only); w = p / (sum + 1e-20) * 2.5;
//              ffn = sum_i w_i expert_i(x) + shared_expert(x) (unscaled); SwiGLU without a clamp
//   R     += ffn
// and on the last part: logits = W_out (rms(R) * output_norm).
//
// WEIGHTS STAY AS THE LOADER KEPT THEM.  Materialized F32 weights cannot fit the 744B model (BUILD-PLAN section 2.2
// slow-path table), so every projection reads its tensor in its stored form (F32 rows, the pack's BF16 rows in w16_,
// native GGUF blocks in ws_map_) and dequantizes it into one of two F32 scratch tiles, in row chunks for the big
// ones (the head in ~0.5 GB chunks), on a side stream so the next chunk's dequant overlaps the current chunk's GEMV.
// The math after the dequant is F32 throughout: F32 activations, F32 KV (latent 512 + k_pe 64 per position) and F32
// index keys on full layers.  The routed experts use the slow path's existing expert machinery (the device pool and
// its classes, or the CPU native path), exactly as Flash's slow path does; an F32 GGUF (Glm5Model::load) runs them
// as F32 GEMVs.
//
// GLM_CB_DIR=<dir> dumps the seams as <dir>/<name>-<layer>.f32 (GLM_CB_APPEND=1 appends per token, GLM_CB_ONLY=<seams>
// filters: src/core/glm_cb_dump.hpp) with llama.cpp's cb names (attn_norm, qr, indexer_q,
// indexer_k, indexer_weights, indexer_score, top_k, q, kv_cmpr_pe, Kcur, Qcur, kqv_out, attn_out, ffn_inp, ffn_norm,
// ffn_moe_logits, ffn_moe_topk, ffn_moe_weights_scaled, ffn_moe_out, ffn_shexp, ffn_out, l_out; result_norm and
// result_output without a layer suffix).  top_k and ffn_moe_topk hold int32 ids.
//
// Environment: STRATA_GLMD_TILE_MB (each of the two dequant tiles' cap, default 512), STRATA_GLMD_KQ_SCALE (a
// different attention scale, for mutation runs only).
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/glmd_forward.hpp"
#include "strata/kernels/glmd_ref.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "../kernels/cuda/glmd_rope.cuh"
#include "glm_cb_dump.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

namespace {

constexpr int kThreads = 256;
constexpr int kTypeF32 = 0, kTypeF16 = 1, kTypeBF16 = 30;

unsigned grid_for(int64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }

// y = x * rsqrt(mean(x^2) + eps) * w over n values, one block, double accumulation (in place allowed)
__global__ void rms_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int n,
                           float eps) {
    __shared__ double s_part[kThreads];
    double acc = 0.0;
    for (int i = threadIdx.x; i < n; i += kThreads) acc += (double) x[i] * (double) x[i];
    s_part[threadIdx.x] = acc;
    __syncthreads();
    for (int o = kThreads / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) s_part[threadIdx.x] += s_part[threadIdx.x + o];
        __syncthreads();
    }
    const float inv = (float) (1.0 / sqrt(s_part[0] / (double) n + (double) eps));
    for (int i = threadIdx.x; i < n; i += kThreads) y[i] = x[i] * inv * w[i];
}

__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void axpy_kernel(float* __restrict__ y, float a, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

// h = silu(g) * u, no clamp (glm-dsa's SwiGLU limit is +inf)
__global__ void silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h,
                                int n) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i < n) h[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

__global__ void iota_kernel(int* __restrict__ c, int n) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i < n) c[i] = i;
}

__global__ void bf16_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * kThreads + threadIdx.x;
    if (i < n) dst[i] = __uint_as_float((uint32_t) src[i] << 16);
}

__global__ void f16_kernel(const __half* __restrict__ src, float* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * kThreads + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// GLM_CB_DIR seam dumps (the same file convention as glm_model.cu's cb_dump)
const char* cb_dir() {
    static const char* dir = getenv("GLM_CB_DIR");
    return dir != nullptr && dir[0] != '\0' ? dir : nullptr;
}

// GLM_CB_APPEND / GLM_CB_ONLY (src/core/glm_cb_dump.hpp, B7.6): append per token, filter the seams; pad_n pads an
// appended selection row with -1 to a fixed width
void cb_write(const std::string& name, const void* host, int64_t n, int64_t pad_n = 0) {
    glmcb::write(name, host, (size_t) n * 4, ".f32", (size_t) std::max<int64_t>(pad_n, 0) * 4);
}

void cb_dev(const std::string& name, const void* dev, int64_t n, int64_t pad_n = 0) {
    if (cb_dir() == nullptr || dev == nullptr || n <= 0 || !glmcb::wanted(name)) return;
    std::vector<float> buf((size_t) n);
    if (cudaMemcpy(buf.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost) == cudaSuccess)
        cb_write(name, buf.data(), n, pad_n);
}

void cb_host(const std::string& name, const void* host, int64_t n) {
    if (cb_dir() != nullptr && host != nullptr && n > 0) cb_write(name, host, n);
}

// One weight as the loader kept it: F32 (type 0), the pack's BF16 rows (30) or native GGUF blocks (a ggml type).
struct WRef {
    const void* p = nullptr;
    int type = -1;
};

// The reference forward's per-part state (Glm5Model::dsa_ref_).
struct DsaRef {
    int dev = 0;
    float kq_scale = glmd::kGlm53KqScale;
    // the two dequant tiles and their ordering (see tile_acquire)
    cudaStream_t s_dq = nullptr;
    cudaEvent_t ev_issue[2] = {nullptr, nullptr}, ev_ready[2] = {nullptr, nullptr};
    float* tile[2] = {nullptr, nullptr};
    int64_t tile_floats = 0;
    int next_tile = 0;
    // rope tables (q_pe / k_pe and the indexer) and the position
    float* inv_q = nullptr;
    float* inv_idx = nullptr;
    int* d_pos = nullptr;
    // activations of one token
    float* scratch = nullptr;
    size_t scratch_bytes = 0;
    float *x = nullptr, *qr = nullptr, *q = nullptr, *Q = nullptr, *iq = nullptr, *iw = nullptr, *score = nullptr;
    float *attn = nullptr, *mix = nullptr, *gate = nullptr, *up = nullptr, *h = nullptr, *ffn = nullptr;
    float *sh = nullptr, *head = nullptr, *router = nullptr, *dn = nullptr;
    int* sel = nullptr;
    int64_t sel_n = -1;   // cells of the last full layer's selection for this token (-1: none yet in this part)
    // the F32 caches: per absolute layer, the [c_kv | k_pe] rows and the index keys (full layers)
    std::vector<float*> kv, ik;
    float* own_cache = nullptr;   // when the state arena's span could not hold them
    uint64_t kv_bytes = 0;
    bool kv_in_state = true;
    // small tensors (norms, biases) that came in a non-F32 form, widened once at setup
    std::map<std::string, float*> small;
    // the router bias per MoE layer, host side (the selection runs on the host in double)
    std::vector<float> bias;
    // the host fallback for block types without a device dequantizer
    std::vector<uint8_t> host_raw;
    std::vector<float> host_f32;
    bool host_dq_noted = false;
    bool logged = false;

    ~DsaRef() {
        cudaSetDevice(dev);
        for (float* t : tile)
            if (t) cudaFree(t);
        for (cudaEvent_t e : ev_issue)
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : ev_ready)
            if (e) cudaEventDestroy(e);
        if (s_dq) cudaStreamDestroy(s_dq);
        if (inv_q) cudaFree(inv_q);
        if (inv_idx) cudaFree(inv_idx);
        if (d_pos) cudaFree(d_pos);
        if (scratch) cudaFree(scratch);
        if (own_cache) cudaFree(own_cache);
        for (auto& kv2 : small) cudaFree(kv2.second);
    }
};

bool cuda_ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("glm-dsa ref: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

// rows [r0, r0 + nr) of a (rows, cols) tensor -> F32 at out, on stream st
bool dq_rows(DsaRef& R, const WRef& w, int64_t r0, int64_t nr, int64_t cols, float* out, cudaStream_t st,
             std::string& err) {
    const int64_t n = nr * cols;
    if (w.type == kTypeF32)
        return cuda_ok(cudaMemcpyAsync(out, (const float*) w.p + r0 * cols, (size_t) n * 4, cudaMemcpyDeviceToDevice, st),
                       "f32 copy", err);
    if (w.type == kTypeBF16) {
        bf16_kernel<<<grid_for(n), kThreads, 0, st>>>((const uint16_t*) w.p + r0 * cols, out, n);
        return cuda_ok(cudaGetLastError(), "bf16 widen", err);
    }
    if (w.type == kTypeF16) {
        f16_kernel<<<grid_for(n), kThreads, 0, st>>>((const __half*) w.p + r0 * cols, out, n);
        return cuda_ok(cudaGetLastError(), "f16 widen", err);
    }
    if (strata::kernels::dequant_bf16_supported(w.type)) {
        strata::kernels::dequant_f32(w.type, w.p, r0, nr, cols, out, st);
        return cuda_ok(cudaGetLastError(), "dequant_f32", err);
    }
    if (strata::kernels::iq_supported(w.type) && cols % 256 == 0) {
        const size_t rb = strata::kernels::iq_row_bytes(w.type, cols);
        strata::kernels::iq_dequant_f32(w.type, (const uint8_t*) w.p + (size_t) r0 * rb, n, out, st);
        return cuda_ok(cudaGetLastError(), "iq_dequant_f32", err);
    }
    // no device dequantizer (Q2_K, Q4_1, Q5_1, ...): ggml's own to_float on the host, synchronously
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) w.type);
    if (tt == nullptr || tt->to_float == nullptr) {
        err = "glm-dsa ref: no dequantizer for ggml type " + std::to_string(w.type);
        return false;
    }
    if (!R.host_dq_noted) {
        std::fprintf(stderr, "glm-dsa ref: CUDA%d ggml type %d has no device dequantizer; widened on the host (slow)\n",
                     R.dev, w.type);
        R.host_dq_noted = true;
    }
    const size_t rb = ggml_row_size((ggml_type) w.type, cols);
    R.host_raw.resize(rb * (size_t) nr);
    R.host_f32.resize((size_t) n);
    if (!cuda_ok(cudaStreamSynchronize(st), "host dequant wait", err) ||
        !cuda_ok(cudaMemcpy(R.host_raw.data(), (const uint8_t*) w.p + rb * (size_t) r0, R.host_raw.size(),
                            cudaMemcpyDeviceToHost), "host dequant read", err))
        return false;
    for (int64_t r = 0; r < nr; ++r) tt->to_float(R.host_raw.data() + rb * (size_t) r, R.host_f32.data() + r * cols, cols);
    return cuda_ok(cudaMemcpyAsync(out, R.host_f32.data(), (size_t) n * 4, cudaMemcpyHostToDevice, st), "host dequant write",
                   err) &&
           cuda_ok(cudaStreamSynchronize(st), "host dequant write wait", err);
}

// The tile ordering: tile b is handed out only after everything already queued on the main (legacy) stream, which
// includes the last kernel that read tile b, so the side stream's dequant cannot overwrite a tile still in use; the
// consumer then waits for ev_ready[b].  Every kernel reads at most one tile and is queued before the next acquire.
int tile_acquire(DsaRef& R, std::string& err) {
    const int b = R.next_tile;
    R.next_tile ^= 1;
    if (!cuda_ok(cudaEventRecord(R.ev_issue[b], 0), "tile issue event", err) ||
        !cuda_ok(cudaStreamWaitEvent(R.s_dq, R.ev_issue[b], 0), "tile issue wait", err))
        return -1;
    return b;
}

bool tile_fill(DsaRef& R, int b, const WRef& w, int64_t r0, int64_t nr, int64_t cols, std::string& err) {
    return dq_rows(R, w, r0, nr, cols, R.tile[b], R.s_dq, err) &&
           cuda_ok(cudaEventRecord(R.ev_ready[b], R.s_dq), "tile ready event", err);
}

bool tile_wait(DsaRef& R, int b, std::string& err) {
    return cuda_ok(cudaStreamWaitEvent(0, R.ev_ready[b], 0), "tile ready wait", err);
}

// the whole tensor (rows x cols) as F32: its own pointer when F32, else a dequantized tile
const float* f32_whole(DsaRef& R, const WRef& w, int64_t rows, int64_t cols, const std::string& name,
                       std::string& err) {
    if (w.type == kTypeF32) return (const float*) w.p;
    if (rows * cols > R.tile_floats) {
        err = "glm-dsa ref: " + name + " (" + std::to_string(rows * cols) + " floats) exceeds the dequant tile (" +
              std::to_string(R.tile_floats) + ")";
        return nullptr;
    }
    const int b = tile_acquire(R, err);
    if (b < 0 || !tile_fill(R, b, w, 0, rows, cols, err) || !tile_wait(R, b, err)) return nullptr;
    return R.tile[b];
}

// y[out] = alpha * W x, W (out rows of `in`) in any stored form; chunked through the two tiles
bool gemv(DsaRef& R, const WRef& w, const float* x, float* y, int64_t in, int64_t out, float alpha,
          const std::string& name, std::string& err) {
    if (w.type == kTypeF32) {
        glmd::ref_gemv((const float*) w.p, x, y, (int) in, (int) out, 1, alpha, nullptr);
        return cuda_ok(cudaGetLastError(), "ref_gemv", err);
    }
    const int64_t rows = std::min<int64_t>(out, glmd::tile_rows(in, R.tile_floats));
    if (rows <= 0) {
        err = "glm-dsa ref: one row of " + name + " (" + std::to_string(in) + " floats) exceeds the dequant tile";
        return false;
    }
    const int64_t n_chunks = (out + rows - 1) / rows;
    int buf[2] = {-1, -1};
    // chunk c + 1's dequant is queued before chunk c's GEMV, so the two overlap
    buf[0] = tile_acquire(R, err);
    if (buf[0] < 0 || !tile_fill(R, buf[0], w, 0, std::min(rows, out), in, err)) return false;
    for (int64_t c = 0; c < n_chunks; ++c) {
        const int64_t r0 = c * rows, nr = std::min(rows, out - r0);
        if (c + 1 < n_chunks) {
            const int64_t r1 = r0 + rows;
            buf[(c + 1) & 1] = tile_acquire(R, err);
            if (buf[(c + 1) & 1] < 0 || !tile_fill(R, buf[(c + 1) & 1], w, r1, std::min(rows, out - r1), in, err))
                return false;
        }
        if (!tile_wait(R, buf[c & 1], err)) return false;
        glmd::ref_gemv(R.tile[buf[c & 1]], x, y + r0, (int) in, (int) nr, 1, alpha, nullptr);
        if (!cuda_ok(cudaGetLastError(), "ref_gemv", err)) return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- setup

bool Glm5Model::dsa_ref_setup(std::string& err) {
    const Glm5Geometry& g = g_;
    cudaSetDevice(dev_);
    auto st = std::make_shared<DsaRef>();
    DsaRef& R = *st;
    R.dev = dev_;
    R.kq_scale = glmd::attn_scale(g.qk_head());
    if (const char* s = getenv("STRATA_GLMD_KQ_SCALE")) {
        R.kq_scale = (float) std::atof(s);
        std::fprintf(stderr, "glm-dsa ref: CUDA%d attention scale overridden to %g (STRATA_GLMD_KQ_SCALE)\n", dev_,
                     (double) R.kq_scale);
    }
    if (g.hc != 1 || g.idx_kpool != 1 || g.qk_rope <= 0 || g.qk_rope % 2 != 0 || g.idx_rope % 2 != 0 ||
        g.idx_rope > g.idx_key || g.idx_top_k <= 0 || g.idx_top_k > glmd::kRefMaxSel || max_ctx_ > INT32_MAX) {
        err = "glm-dsa ref: geometry outside the reference forward (hc 1, kpool 1, even rope dims, top_k <= " +
              std::to_string(glmd::kRefMaxSel) + ")";
        return false;
    }

    // the weight lookup (the same one step_layers_dsa uses)
    const auto wref = [&](const std::string& name) -> WRef {
        const auto b = w16_.find(name);
        if (b != w16_.end()) return {b->second, kTypeBF16};
        const auto s = ws_map_.find(name);
        if (s != ws_map_.end()) return s->second.q != nullptr ? WRef{s->second.q, s->second.type} : WRef{s->second.f32, 0};
        const auto f = w_.find(name);
        if (f != w_.end()) return {f->second, kTypeF32};
        return {};
    };

    // 1. the F32 KV: inside the span the state arena reserved per DSA layer when it fits, else its own buffer
    const int kvr = g.kv_row();
    R.kv.assign((size_t) g.n_layers, nullptr);
    R.ik.assign((size_t) g.n_layers, nullptr);
    std::vector<int> own;
    for (int il = l0_; il < l1_; ++il) {
        const bool full = g.idx_full(il);
        glmd::KvCarve c;
        if (glmd::kv_carve(dsa_lat_[(size_t) il], dsa_ik_[(size_t) il], dsa_ig_[(size_t) il], dsa_pool_[(size_t) il],
                           (int64_t) g.kv_lora * max_ctx_, ik_ring_, max_ctx_ / g.idx_kpool, g.idx_key, kvr, max_ctx_,
                           full, c)) {
            R.kv[(size_t) il] = state_ + c.kv;
            if (full) R.ik[(size_t) il] = state_ + c.ik;
        } else {
            own.push_back(il);
        }
        R.kv_bytes += (uint64_t) max_ctx_ * 4 * (uint64_t) (kvr + (full ? g.idx_key : 0));
    }
    if (!own.empty()) {
        R.kv_in_state = false;
        uint64_t floats = 0;
        for (int il : own) floats += (uint64_t) max_ctx_ * (uint64_t) (kvr + (g.idx_full(il) ? g.idx_key : 0));
        if (!cuda_ok(cudaMalloc(&R.own_cache, floats * 4), "own KV cache", err)) return false;
        uint64_t at = 0;
        for (int il : own) {
            R.kv[(size_t) il] = R.own_cache + at;
            at += (uint64_t) max_ctx_ * kvr;
            if (g.idx_full(il)) {
                R.ik[(size_t) il] = R.own_cache + at;
                at += (uint64_t) max_ctx_ * g.idx_key;
            }
        }
    }

    // 2. rope tables and the position
    {
        std::vector<float> inv((size_t) std::max(g.qk_rope, g.idx_rope) / 2 + 1);
        glmd::rope_inv_freq(inv.data(), g.qk_rope, g.rope_theta);
        if (!cuda_ok(cudaMalloc(&R.inv_q, (size_t) g.qk_rope / 2 * 4), "rope table", err) ||
            !cuda_ok(cudaMemcpy(R.inv_q, inv.data(), (size_t) g.qk_rope / 2 * 4, cudaMemcpyHostToDevice), "rope table", err))
            return false;
        const int nr = std::max(g.idx_rope, 2);
        glmd::rope_inv_freq(inv.data(), nr, g.rope_theta);
        if (!cuda_ok(cudaMalloc(&R.inv_idx, (size_t) nr / 2 * 4), "rope table", err) ||
            !cuda_ok(cudaMemcpy(R.inv_idx, inv.data(), (size_t) nr / 2 * 4, cudaMemcpyHostToDevice), "rope table", err) ||
            !cuda_ok(cudaMalloc(&R.d_pos, sizeof(int)), "position", err))
            return false;
    }

    // 3. one token's activations
    {
        const int64_t E = g.n_embd, H = g.n_head;
        const int64_t ff = std::max<int64_t>({(int64_t) g.n_ff_dense, (int64_t) g.n_ff_exp * g.n_shared, g.n_ff_exp});
        int64_t s = 0;
        std::vector<std::pair<float**, int64_t>> take = {
            {&R.x, E}, {&R.qr, g.q_lora}, {&R.q, H * g.qk_head()}, {&R.Q, H * kvr},
            {&R.iq, (int64_t) g.idx_heads * g.idx_key}, {&R.iw, g.idx_heads}, {&R.score, max_ctx_},
            {&R.attn, H * g.v_head}, {&R.mix, E}, {&R.gate, ff}, {&R.up, ff}, {&R.h, ff}, {&R.ffn, E}, {&R.sh, E},
            {&R.head, E}, {&R.router, g.n_expert}, {&R.dn, E}};
        std::vector<int64_t> at;
        for (auto& t : take) {
            at.push_back(s);
            s += (t.second + 63) & ~(int64_t) 63;
        }
        const int64_t sel_at = s;
        s += ((int64_t) std::max<int64_t>(g.idx_top_k, 1) + 63) & ~(int64_t) 63;
        R.scratch_bytes = (size_t) s * 4;
        if (!cuda_ok(cudaMalloc(&R.scratch, R.scratch_bytes), "scratch", err)) return false;
        for (size_t i = 0; i < take.size(); ++i) *take[i].first = R.scratch + at[i];
        R.sel = (int*) (R.scratch + sel_at);
    }

    // 4. the tensors this part reads, their shapes, and the dequant tile they need
    struct Need {
        std::string name;
        int64_t rows, cols;
        bool whole;   // read as one F32 tensor by a reference kernel (else row-chunked GEMV)
    };
    std::vector<Need> needs;
    const int64_t E = g.n_embd, QL = g.q_lora, KL = g.kv_lora, H = g.n_head, IK = g.idx_key, IH = g.idx_heads;
    const int64_t FS = (int64_t) g.n_ff_exp * g.n_shared;
    std::vector<std::pair<std::string, int64_t>> smalls;
    for (int il = l0_; il < l1_; ++il) {
        const std::string P = "blk." + std::to_string(il) + ".";
        needs.push_back({P + "attn_q_a.weight", QL, E, false});
        needs.push_back({P + "attn_q_b.weight", H * g.qk_head(), QL, false});
        needs.push_back({P + "attn_kv_a_mqa.weight", kvr, E, false});
        needs.push_back({P + "attn_k_b.weight", KL * H, g.qk_nope, true});
        needs.push_back({P + "attn_v_b.weight", (int64_t) g.v_head * H, KL, true});
        needs.push_back({P + "attn_output.weight", E, H * g.v_head, false});
        smalls.push_back({P + "attn_norm.weight", E});
        smalls.push_back({P + "attn_q_a_norm.weight", QL});
        smalls.push_back({P + "attn_kv_a_norm.weight", KL});
        smalls.push_back({P + "ffn_norm.weight", E});
        if (g.idx_full(il)) {
            needs.push_back({P + "indexer.attn_k.weight", IK, E, true});
            needs.push_back({P + "indexer.attn_q_b.weight", IH * IK, QL, true});
            needs.push_back({P + "indexer.proj.weight", IH, E, true});
            smalls.push_back({P + "indexer.k_norm.weight", IK});
            smalls.push_back({P + "indexer.k_norm.bias", IK});
        }
        if (il < g.dense_lead) {
            needs.push_back({P + "ffn_gate.weight", g.n_ff_dense, E, false});
            needs.push_back({P + "ffn_up.weight", g.n_ff_dense, E, false});
            needs.push_back({P + "ffn_down.weight", E, g.n_ff_dense, false});
        } else {
            needs.push_back({P + "ffn_gate_inp.weight", g.n_expert, E, false});
            needs.push_back({P + "ffn_gate_shexp.weight", FS, E, false});
            needs.push_back({P + "ffn_up_shexp.weight", FS, E, false});
            needs.push_back({P + "ffn_down_shexp.weight", E, FS, false});
            smalls.push_back({P + "exp_probs_b.bias", g.n_expert});
        }
    }
    if (l1_ == g.n_layers) {
        needs.push_back({"output.weight", g.n_vocab, E, false});
        smalls.push_back({"output_norm.weight", E});
    }
    int64_t cap = (int64_t) 512 << 20;
    if (const char* t = getenv("STRATA_GLMD_TILE_MB")) cap = std::max<int64_t>(1, std::atoll(t)) << 20;
    cap /= 4;   // floats
    int64_t tile = 0;
    for (const auto& n : needs) {
        const WRef w = wref(n.name);
        if (w.p == nullptr) {
            err = "glm-dsa ref: weight " + n.name + " is not loaded";
            return false;
        }
        if (w.type == kTypeF32) continue;
        tile = std::max(tile, n.whole ? n.rows * n.cols : std::min(n.rows * n.cols, std::max(cap, n.cols)));
    }
    R.tile_floats = tile;
    if (tile > 0) {
        for (int b = 0; b < 2; ++b)
            if (!cuda_ok(cudaMalloc(&R.tile[b], (size_t) tile * 4), "dequant tile", err) ||
                !cuda_ok(cudaEventCreateWithFlags(&R.ev_issue[b], cudaEventDisableTiming), "tile event", err) ||
                !cuda_ok(cudaEventCreateWithFlags(&R.ev_ready[b], cudaEventDisableTiming), "tile event", err))
                return false;
        if (!cuda_ok(cudaStreamCreateWithFlags(&R.s_dq, cudaStreamNonBlocking), "dequant stream", err)) return false;
    }

    // 5. the small vectors: F32 as loaded, or widened once now; the router bias also to the host
    R.bias.assign((size_t) g.n_layers * g.n_expert, 0.0f);
    for (const auto& s : smalls) {
        const WRef w = wref(s.first);
        if (w.p == nullptr) {
            err = "glm-dsa ref: weight " + s.first + " is not loaded";
            return false;
        }
        const float* f = (const float*) w.p;
        if (w.type != kTypeF32) {
            float* d = nullptr;
            if (!cuda_ok(cudaMalloc(&d, (size_t) s.second * 4), "small tensor", err)) return false;
            R.small[s.first] = d;
            if (!dq_rows(R, w, 0, 1, s.second, d, nullptr, err)) return false;
            f = d;
        }
        if (s.first.find("exp_probs_b") != std::string::npos) {
            const int il = std::atoi(s.first.c_str() + 4);
            if (!cuda_ok(cudaMemcpy(R.bias.data() + (size_t) il * g.n_expert, f, (size_t) g.n_expert * 4,
                                    cudaMemcpyDeviceToHost), "router bias", err))
                return false;
        }
    }
    if (!cuda_ok(cudaDeviceSynchronize(), "setup", err)) return false;
    std::fprintf(stderr, "glm-dsa ref: CUDA%d layers [%d, %d): F32 reference forward (STRATA_GLM_SLOW), KV %.3f GB F32 %s, "
                         "dequant tiles 2 x %.3f GB, scratch %.1f MB, scale %g\n",
                 dev_, l0_, l1_, (double) R.kv_bytes / 1e9, R.kv_in_state ? "in the state arena" : "(own buffer)",
                 (double) tile * 4 / 1e9, (double) R.scratch_bytes / 1e6, (double) R.kq_scale);
    dsa_ref_ = st;
    return true;
}

// ---------------------------------------------------------------------------------------------------- forward

bool Glm5Model::step_layers_dsa(int64_t p, std::string& err) {
    const Glm5Geometry& g = g_;
    DsaRef* Rp = static_cast<DsaRef*>(dsa_ref_.get());
    if (Rp == nullptr) {
        err = "glm-dsa ref: the reference state was not set up (load with STRATA_GLM_SLOW=1)";
        return false;
    }
    DsaRef& R = *Rp;
    if (p < 0 || p >= max_ctx_) {
        err = "glm-dsa ref: position " + std::to_string(p) + " outside the context";
        return false;
    }
    if (!R.logged) {
        // the per-GPU slow-path bytes (B3.5 / B7.1): everything this part holds, and what is left
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::fprintf(stderr,
                     "glm-dsa ref: CUDA%d slow-path bytes: dense weights %.2f GB (as stored), dequant tiles %.2f GB, "
                     "F32 KV %.2f GB, state %.2f GB, scratch %.2f GB, expert pool %.2f GB; free %.2f of %.2f GB\n",
                     dev_, (double) w_arena_bytes_ / 1e9, (double) R.tile_floats * 8 / 1e9, (double) R.kv_bytes / 1e9,
                     (double) state_bytes_ / 1e9, (double) (sc_bytes_ + R.scratch_bytes) / 1e9,
                     (double) pool_bytes_ / 1e9, (double) free_b / 1e9, (double) total_b / 1e9);
        R.logged = true;
    }
    const auto wref = [&](const std::string& name) -> WRef {
        const auto b = w16_.find(name);
        if (b != w16_.end()) return {b->second, kTypeBF16};
        const auto s = ws_map_.find(name);
        if (s != ws_map_.end()) return s->second.q != nullptr ? WRef{s->second.q, s->second.type} : WRef{s->second.f32, 0};
        const auto f = w_.find(name);
        if (f != w_.end()) return {f->second, kTypeF32};
        return {};
    };
    const auto vec = [&](const std::string& name) -> const float* {   // a small F32 vector (setup checked them)
        const auto s = R.small.find(name);
        if (s != R.small.end()) return s->second;
        return (const float*) wref(name).p;
    };
    const auto mv = [&](const std::string& name, const float* x, float* y, int64_t in, int64_t out, float alpha = 1.0f) {
        const WRef w = wref(name);
        if (w.p == nullptr) {
            err = "glm-dsa ref: weight " + name + " is not loaded";
            return false;
        }
        return gemv(R, w, x, y, in, out, alpha, name, err);
    };
    const auto whole = [&](const std::string& name, int64_t rows, int64_t cols) -> const float* {
        const WRef w = wref(name);
        if (w.p == nullptr) {
            err = "glm-dsa ref: weight " + name + " is not loaded";
            return nullptr;
        }
        return f32_whole(R, w, rows, cols, name, err);
    };
    const auto rms = [&](const float* x, const float* w, float* y, int n) {
        rms_kernel<<<1, kThreads>>>(x, w, y, n, g.norm_eps);
    };
    const auto add = [&](float* y, const float* x, int n) { add_kernel<<<grid_for(n), kThreads>>>(y, x, n); };

    const int E = g.n_embd, QL = g.q_lora, KL = g.kv_lora, H = g.n_head, IK = g.idx_key, IH = g.idx_heads;
    const int NOPE = g.qk_nope, ROPE = g.qk_rope, QH = g.qk_head(), VH = g.v_head, KVR = g.kv_row();
    const int FS = g.n_ff_exp * g.n_shared;
    const int64_t n_kv = p + 1;
    const int p32 = (int) p;
    if (!cuda_ok(cudaMemcpy(R.d_pos, &p32, sizeof(int), cudaMemcpyHostToDevice), "position", err)) return false;
    float* const Rs = state_;   // the plain residual (what step() hands to the next part)
    float* const A = sc_;
    R.sel_n = -1;

    for (int il = l0_; il < l1_; ++il) {
        const std::string L = std::to_string(il);
        const std::string P = "blk." + L + ".";
        const auto N = [&](const char* s) { return std::string(s) + "-" + L; };

        // ---- attention
        rms(Rs, vec(P + "attn_norm.weight"), R.x, E);
        cb_dev(N("attn_norm"), R.x, E);
        if (!mv(P + "attn_q_a.weight", R.x, R.qr, E, QL)) return false;
        rms(R.qr, vec(P + "attn_q_a_norm.weight"), R.qr, QL);
        cb_dev(N("qr"), R.qr, QL);

        if (g.idx_full(il)) {
            // the lightning indexer: this position's key into the cache, then the selection
            float* ik_p = R.ik[(size_t) il] + (int64_t) IK * p;
            const float* w_ik = whole(P + "indexer.attn_k.weight", IK, E);
            if (w_ik == nullptr) return false;
            glmd::ref_indexer_key(R.x, w_ik, vec(P + "indexer.k_norm.weight"), vec(P + "indexer.k_norm.bias"),
                                  g.idx_ln_eps, E, IK, 1, g.idx_rope, R.d_pos, R.inv_idx, ik_p, nullptr);
            const float* w_iq = whole(P + "indexer.attn_q_b.weight", (int64_t) IH * IK, QL);
            if (w_iq == nullptr) return false;
            glmd::ref_indexer_query(R.qr, w_iq, QL, IH, IK, 1, g.idx_rope, R.d_pos, R.inv_idx, R.iq, nullptr);
            const float* w_ip = whole(P + "indexer.proj.weight", IH, E);
            if (w_ip == nullptr) return false;
            glmd::ref_indexer_weights(R.x, w_ip, E, IH, IK, 1, R.iw, nullptr);
            cb_dev(N("indexer_q"), R.iq, (int64_t) IH * IK);
            cb_dev(N("indexer_k"), ik_p, IK);
            cb_dev(N("indexer_weights"), R.iw, IH);
            if (glmd::attend_dense(n_kv, g.idx_top_k)) {
                iota_kernel<<<grid_for(n_kv), kThreads>>>(R.sel, (int) n_kv);
                R.sel_n = n_kv;
            } else {
                glmd::ref_indexer_score(R.iq, R.ik[(size_t) il], R.iw, IH, IK, 1, (int) n_kv, R.d_pos, R.score,
                                        nullptr);
                cb_dev(N("indexer_score"), R.score, n_kv);
                glmd::ref_topk(R.score, (int) n_kv, 1, g.idx_top_k, R.sel, nullptr);
                R.sel_n = g.idx_top_k;
            }
        } else if (R.sel_n < 0) {
            err = "glm-dsa ref: shared-indexer layer " + L + " has no full layer before it in this part [" +
                  std::to_string(l0_) + ", " + std::to_string(l1_) + ")";
            return false;
        }
        cb_dev(N("top_k"), R.sel, R.sel_n, g.idx_top_k);

        if (!mv(P + "attn_q_b.weight", R.qr, R.q, QL, (int64_t) H * QH)) return false;
        cb_dev(N("q"), R.q, (int64_t) H * QH);
        glmd::ref_rope_norm(R.q, QH, H, 1, NOPE, ROPE, R.d_pos, R.inv_q, nullptr);
        float* kv_p = R.kv[(size_t) il] + (int64_t) KVR * p;
        if (!mv(P + "attn_kv_a_mqa.weight", R.x, kv_p, E, KVR)) return false;
        cb_dev(N("kv_cmpr_pe"), kv_p, KVR);
        glmd::ref_rope_norm(kv_p, KVR, 1, 1, KL, ROPE, R.d_pos, R.inv_q, nullptr);
        rms(kv_p, vec(P + "attn_kv_a_norm.weight"), kv_p, KL);
        cb_dev(N("Kcur"), kv_p, KVR);
        const float* wk_b = whole(P + "attn_k_b.weight", (int64_t) KL * H, NOPE);
        if (wk_b == nullptr) return false;
        glmd::ref_mla_absorb_q(R.q, wk_b, H, NOPE, ROPE, KL, 1, R.Q, nullptr);
        cb_dev(N("Qcur"), R.Q, (int64_t) H * KVR);
        const float* wv_b = whole(P + "attn_v_b.weight", (int64_t) VH * H, KL);
        if (wv_b == nullptr) return false;
        glmd::ref_mla_attention(R.Q, R.kv[(size_t) il], R.sel, wv_b, H, KL, ROPE, VH, 1, (int) R.sel_n, R.kq_scale,
                                R.attn, nullptr);
        cb_dev(N("kqv_out"), R.attn, (int64_t) H * VH);
        if (!mv(P + "attn_output.weight", R.attn, R.mix, (int64_t) H * VH, E)) return false;
        cb_dev(N("attn_out"), R.mix, E);
        ablate(R.mix, 1, nullptr);   // (--ablate: the write loses the direction)
        add(Rs, R.mix, E);
        cb_dev(N("ffn_inp"), Rs, E);

        // ---- FFN (x in sc_x: the slow path's expert tail reads it there)
        float* X = A + sc_x;
        rms(Rs, vec(P + "ffn_norm.weight"), X, E);
        cb_dev(N("ffn_norm"), X, E);
        if (il < g.dense_lead) {
            const int FD = g.n_ff_dense;
            if (!mv(P + "ffn_gate.weight", X, R.gate, E, FD) || !mv(P + "ffn_up.weight", X, R.up, E, FD)) return false;
            silu_mul_kernel<<<grid_for(FD), kThreads>>>(R.gate, R.up, R.h, FD);
            if (!mv(P + "ffn_down.weight", R.h, R.ffn, FD, E)) return false;
        } else {
            const int NX = g.n_expert, K = g.n_exp_used;
            if (!mv(P + "ffn_gate_inp.weight", X, R.router, E, NX)) return false;
            cb_dev(N("ffn_moe_logits"), R.router, NX);
            std::vector<float> lg((size_t) NX), rwv((size_t) K);
            std::vector<int> ids((size_t) K);
            if (!cuda_ok(cudaMemcpy(lg.data(), R.router, (size_t) NX * 4, cudaMemcpyDeviceToHost), "router logits", err) ||
                !cuda_ok(cudaMemcpy(A + sc_tmp, R.router, (size_t) NX * 4, cudaMemcpyDeviceToDevice), "router copy", err))
                return false;
            glmd::route_sigmoid(lg.data(), R.bias.data() + (size_t) il * NX, NX, K, g.norm_w != 0, g.w_scale,
                                ids.data(), rwv.data());
            cb_host(N("ffn_moe_topk"), ids.data(), K);
            cb_host(N("ffn_moe_weights_scaled"), rwv.data(), K);
            float* moe = A + sc_moe;
            if (pack_) {
                if (dev_experts_) {
                    if (!pack_moe_tail_device(err, ids, rwv, A, il)) return false;
                } else if (pack_moe_tail(err, ids, rwv, A, il) != cudaSuccess) {
                    return false;
                }
            } else {
                // an F32 GGUF (Glm5Model::load): the experts are F32 slabs, gate/up {E, F, NX}, down {F, E, NX}
                const WRef wg = wref(P + "ffn_gate_exps.weight"), wu = wref(P + "ffn_up_exps.weight"),
                           wd = wref(P + "ffn_down_exps.weight");
                if (wg.type != kTypeF32 || wu.type != kTypeF32 || wd.type != kTypeF32) {
                    err = "glm-dsa ref: layer " + L + ": routed experts must be F32 outside a pack";
                    return false;
                }
                const int F = g.n_ff_exp;
                const int64_t slab = (int64_t) E * F;
                if (!cuda_ok(cudaMemset(moe, 0, (size_t) E * 4), "moe clear", err)) return false;
                for (int i = 0; i < K; ++i) {
                    const int64_t e = ids[(size_t) i];
                    glmd::ref_gemv((const float*) wg.p + slab * e, X, R.gate, E, F, 1, 1.0f, nullptr);
                    glmd::ref_gemv((const float*) wu.p + slab * e, X, R.up, E, F, 1, 1.0f, nullptr);
                    silu_mul_kernel<<<grid_for(F), kThreads>>>(R.gate, R.up, R.h, F);
                    glmd::ref_gemv((const float*) wd.p + slab * e, R.h, R.dn, F, E, 1, 1.0f, nullptr);
                    axpy_kernel<<<grid_for(E), kThreads>>>(moe, rwv[(size_t) i], R.dn, E);
                }
            }
            cb_dev(N("ffn_moe_out"), moe, E);
            if (!mv(P + "ffn_gate_shexp.weight", X, R.gate, E, FS) || !mv(P + "ffn_up_shexp.weight", X, R.up, E, FS))
                return false;
            silu_mul_kernel<<<grid_for(FS), kThreads>>>(R.gate, R.up, R.h, FS);
            if (!mv(P + "ffn_down_shexp.weight", R.h, R.sh, FS, E)) return false;
            cb_dev(N("ffn_shexp"), R.sh, E);
            if (!cuda_ok(cudaMemcpy(R.ffn, moe, (size_t) E * 4, cudaMemcpyDeviceToDevice), "moe copy", err)) return false;
            add(R.ffn, R.sh, E);
        }
        cb_dev(N("ffn_out"), R.ffn, E);
        ablate(R.ffn, 1, nullptr);
        add(Rs, R.ffn, E);
        cb_dev(N("l_out"), Rs, E);
        if (!cuda_ok(cudaGetLastError(), ("layer " + L).c_str(), err)) return false;
    }

    if (l1_ == g.n_layers) {
        rms(Rs, vec("output_norm.weight"), R.head, E);
        cb_dev("result_norm", R.head, E);
        if (!mv("output.weight", R.head, A + sc_logits, E, g.n_vocab)) return false;
        cb_dev("result_output", A + sc_logits, g.n_vocab);
    }
    return cuda_ok(cudaGetLastError(), "step", err);
}

}  // namespace strata::core
