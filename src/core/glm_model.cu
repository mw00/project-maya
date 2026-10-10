// src/core/glm_model.cpp - the glm5-next runner.  See glm_model.hpp for the contract.
//
// Everything below is the parity test's PROVEN arithmetic (glm_layer_parity) promoted into a
// persistent-state runner, plus the small glue kernels the test did on the host:
//
//   gemv_bias   one thread per (out, token), an optional bias and an alpha (the indexer weights'
//               1/sqrt(key_dim * heads) prescale rides here)
//   rms/layer   one block per token for the reduction, then the weighted write
//   l2_heads    the GDN l2 norm, per (head, token) - q and k only, v is NOT normalized
//   sigmoid     in place, for beta
//   axpy/add    the MoE accumulate and the moe+shared sum
//   mean        the head's plain mean over the hc streams
//   q_absorb    q_h through wk_b into the latent space, one thread per (c, h, t)
//
// The streaming shape: ONE TOKEN AT A TIME through every layer, state carried in the state arena.
// The DSA caches are append-only (latent column per token; a pool's key when its last member
// lands), so a batched call is literally the same per-token steps as single-token calls - which
// is what glm_model_test's batched-vs-streamed equivalence pins.
#include "glm_fast_state.hpp"

#include "strata/kernels/glm_dsa.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/glm_ffn.hpp"
#include "strata/kernels/glm_hc.hpp"
#include "strata/kernels/glm_kda.hpp"
#include "strata/kernels/router_sigmoid.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <tuple>
#include <thread>
#include <map>
#include <atomic>
#include <functional>
#include <sstream>
#include <memory>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <numeric>

// the GGUF shards a pack serves from, in order: native_experts.txt's header names shard 1 (they sit NEXT TO the pack;
// the siblings follow its -NNNNN-of-MMMMM pattern)
static bool pack_shard_names(const std::string& pack_dir, std::vector<std::string>& shard_names, std::string& err) {
    shard_names.clear();
    std::ifstream ne(pack_dir + "/native_experts.txt");
    if (!ne) {
        err = "pack: cannot open " + pack_dir + "/native_experts.txt";
        return false;
    }
    std::string line;
    while (std::getline(ne, line)) {
        if (line.rfind("#", 0) == 0 && line.find("absolute offsets in ") != std::string::npos) {
            const std::string key = "absolute offsets in ";
            const size_t a = line.find(key) + key.size();
            const size_t b = line.find(',', a);
            shard_names.push_back(line.substr(a, b == std::string::npos ? line.size() : b - a));
        }
    }
    if (shard_names.empty()) {
        err = "pack: native_experts.txt names no source shards";
        return false;
    }
    // the header names only shard 1; the siblings follow the -NNNNN-of-MMMMM pattern
    const std::string first = shard_names[0];
    const size_t of = first.find("-of-");
    if (of != std::string::npos) {
        const size_t sdash = first.rfind("-", of - 1);   // the dash that starts this shard's number
        const std::string stem = first.substr(0, sdash);
        const int total = std::atoi(first.c_str() + of + 4);
        shard_names.clear();
        for (int i = 1; i <= total; ++i) {
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s-%05d-of-%05d.gguf", stem.c_str(), i, total);
            shard_names.push_back(buf);
        }
    }
    return true;
}

// a dense row's VRAM bytes as load_pack uploads it (0: not uploaded), and how: the fast path keeps the pack's big BF16
// rows (kind 4) BF16 and the natively served rows (kind 0) quantized, dequantizes the rest to F32, and leaves
// token_embd on the host (the embedding row is dequantized there from the shard mapping)
static uint64_t pack_row_vram(const std::string& name, const std::string& kind, const strata::TensorInfo* t, bool fast,
                              std::string* how = nullptr) {
    if (fast && name == "token_embd.weight") return 0;
    if (fast && kind == "4" && t->elements() >= 65536) {
        if (how) *how = " bf16";
        return ((uint64_t) t->elements() * 2 + 255u) & ~(uint64_t) 255u;
    }
    if (kind != "0" || name == "attn_k_b.weight" || name == "attn_v_b.weight") {
        if (how) *how = " f32";
        return ((uint64_t) t->elements() * 4 + 255u) & ~(uint64_t) 255u;
    }
    if (how) *how = " q" + std::to_string((int) t->type);
    return (strata::kernels::native_mmvq_weight_bytes(t->type, (int) t->shape[0], (int) t->shape[1]) + 255u) &
           ~(uint64_t) 255u;
}

// the NextN draft block's dense tensors (blk.<n_layers>.*) as load_mtp keeps them: natively served, BF16, F32
static const char* const kMtpRaw[] = {"attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight",
                                      "attn_output.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                      "ffn_down_shexp.weight", "nextn.eh_proj.weight"};
static const char* const kMtpB16[] = {"indexer.attn_k.weight", "indexer_compressor_gate.weight",
                                      "indexer.attn_q_b.weight", "indexer.proj.weight", "ffn_gate_inp.weight",
                                      "attn_k_b.weight", "attn_v_b.weight"};
static const char* const kMtpF32[] = {"attn_norm.weight", "ffn_norm.weight", "attn_q_a_norm.weight",
                                      "attn_kv_a_norm.weight", "indexer.k_norm.weight", "indexer.k_norm.bias",
                                      "indexer_compressor_ape.weight", "exp_probs_b.bias", "nextn.enorm.weight",
                                      "nextn.hnorm.weight", "nextn.shared_head_norm.weight"};

// ... and the VRAM they take
static bool mtp_dense_vram(const std::function<const strata::TensorInfo*(const std::string&)>& find, int il,
                           uint64_t& bytes, std::string& err) {
    const std::string P = "blk." + std::to_string(il) + ".";
    bytes = 0;
    for (const char* n : kMtpRaw) {
        const strata::TensorInfo* t = find(P + n);
        if (t == nullptr) { err = "mtp: " + P + n + " missing"; return false; }
        bytes += (strata::kernels::native_mmvq_weight_bytes(t->type, (int) t->shape[0], (int) t->shape[1]) + 255u) &
                 ~(uint64_t) 255u;
    }
    for (const char* n : kMtpB16) {
        const strata::TensorInfo* t = find(P + n);
        if (t == nullptr) { err = "mtp: " + P + n + " missing"; return false; }
        bytes += ((uint64_t) t->elements() * 2 + 255u) & ~(uint64_t) 255u;
    }
    for (const char* n : kMtpF32) {
        const strata::TensorInfo* t = find(P + n);
        if (t == nullptr) { err = "mtp: " + P + n + " missing"; return false; }
        bytes += ((uint64_t) t->elements() * 4 + 255u) & ~(uint64_t) 255u;
    }
    return true;
}

namespace strata::core {
namespace {

constexpr int GM_THREADS = 256;

/// gemv_bias_kernel is warp-per-output (one warp per row): the grid must cover `out` WARPS.
constexpr int gm_blocks(int64_t out) { return (int) ((out * 32 + GM_THREADS - 1) / GM_THREADS); }

#define GM_CHECK(call, err)                                                                  \
    do {                                                                                     \
        if ((call) != cudaSuccess) {                                                         \
            (err) = std::string("cuda: ") + #call + ": " + cudaGetErrorString(cudaGetLastError()); \
            return false;                                                                    \
        }                                                                                    \
    } while (0)

__global__ void gemv_bias_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                 const float* __restrict__ bias, float alpha, float* __restrict__ y, int out,
                                 int in, int T) {
    // One WARP per output row (per column), coalesced over k.  The first version put one thread on a
    // whole row: a serial `in`-element dot with a stride-`in` access pattern, ~3 G-MAC/s - measured
    // as 89 ms/token of the GLM decode (the F32-preserved rows: router, KDA gates, indexer).
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= out * T) return;
    const int t = wid % T;
    const int e = wid / T;
    const float* wr = w + (size_t) e * in;
    const float* xr = x + (size_t) in * t;
    float acc = 0.0f;
    for (int k = lane; k < in; k += 32) acc += wr[k] * xr[k];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[e + (size_t) out * t] = alpha * acc + (bias ? bias[e] : 0.0f);
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                float* __restrict__ y, int dim, int T, float eps) {
    const int t = blockIdx.x;
    if (t >= T) return;
    __shared__ float acc[GM_THREADS];
    const int tid = threadIdx.x;
    float local = 0.0f;
    for (int e = tid; e < dim; e += blockDim.x) local += x[e + (size_t) dim * t] * x[e + (size_t) dim * t];
    acc[tid] = local;
    __syncthreads();
    for (int span = blockDim.x / 2; span > 0; span >>= 1) {
        if (tid < span) acc[tid] += acc[tid + span];
        __syncthreads();
    }
    const float inv = rsqrtf(acc[0] / (float) dim + eps);
    for (int e = tid; e < dim; e += blockDim.x)
        y[e + (size_t) dim * t] = x[e + (size_t) dim * t] * inv * (w ? w[e] : 1.0f);
}

__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* __restrict__ y, int dim, int T,
                                  float eps) {
    const int t = blockIdx.x;
    if (t >= T) return;
    __shared__ float acc[2 * GM_THREADS];
    float* s_acc = acc;
    float* s_sq = acc + GM_THREADS;
    const int tid = threadIdx.x;
    float local = 0.0f, sq = 0.0f;
    for (int e = tid; e < dim; e += blockDim.x) {
        const float v = x[e + (size_t) dim * t];
        local += v;
        sq += v * v;
    }
    s_acc[tid] = local;
    s_sq[tid] = sq;
    __syncthreads();
    for (int span = blockDim.x / 2; span > 0; span >>= 1) {
        if (tid < span) {
            s_acc[tid] += s_acc[tid + span];
            s_sq[tid] += s_sq[tid + span];
        }
        __syncthreads();
    }
    const float mu = s_acc[0] / (float) dim;
    const float inv = rsqrtf(s_sq[0] / (float) dim - mu * mu + eps);
    for (int e = tid; e < dim; e += blockDim.x)
        y[e + (size_t) dim * t] = (x[e + (size_t) dim * t] - mu) * inv * w[e] + b[e];
}

__global__ void l2_heads_kernel(const float* __restrict__ x, float* __restrict__ y, int dk, int n_head, int T) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_head * T) return;
    const int h = idx % n_head;
    const int t = idx / n_head;
    const float* xt = x + ((size_t) t * n_head + h) * dk;
    float ss = 0.0f;
    for (int e = 0; e < dk; ++e) ss += xt[e] * xt[e];
    const float inv = rsqrtf(ss + 1e-6f);
    float* yt = y + ((size_t) t * n_head + h) * dk;
    for (int e = 0; e < dk; ++e) yt[e] = xt[e] * inv;
}

__global__ void sigmoid_kernel(float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void axpy_kernel(float* __restrict__ acc, float alpha, const float* __restrict__ v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] += alpha * v[i];
}

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + b[i];
}

__global__ void mean_streams_kernel(const float* __restrict__ R, float* __restrict__ y, int n_embd, int hc, int T) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_embd * T) return;
    const int t = idx / n_embd;
    const int e = idx % n_embd;
    float acc = 0.0f;
    for (int s = 0; s < hc; ++s) acc += R[e + (size_t) n_embd * s + (size_t) n_embd * hc * t];
    y[e + (size_t) n_embd * t] = acc / (float) hc;
}

__global__ void q_absorb_kernel(const float* __restrict__ wk_b, const float* __restrict__ q,
                                float* __restrict__ y, int qk_nope, int kv_lora, int n_head, int T) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kv_lora * n_head * T) return;
    const int t = idx / (kv_lora * n_head);
    const int h = (idx / kv_lora) % n_head;
    const int c = idx % kv_lora;
    const float* w = wk_b + (size_t) qk_nope * kv_lora * h;           // (e, c) at e + qk_nope*c
    const float* qh = q + (size_t) qk_nope * h + (size_t) qk_nope * n_head * t;
    float acc = 0.0f;
    for (int e = 0; e < qk_nope; ++e) acc += w[(size_t) e + (size_t) qk_nope * c] * qh[e];
    y[c + (size_t) kv_lora * h + (size_t) kv_lora * n_head * t] = acc;
}

int blocks(int n) { return (n + GM_THREADS - 1) / GM_THREADS; }

__global__ void bias_apply_kernel(float* __restrict__ y, const float* __restrict__ bias, float alpha,
                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = alpha * y[i] + (bias ? bias[i] : 0.0f);
}

float* q8_scratch(size_t bytes) {
    // PER DEVICE: the layer split runs two instances on two devices, and a single shared buffer
    // made the second device's kernels write into the first device's allocation (an out-of-bounds
    // illegal access that compute-sanitizer pinned to quantize_q8_1_kernel)
    static float* buf[64] = {};
    static size_t have[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return nullptr;
    if (have[dev] < bytes) {
        if (buf[dev]) cudaFree(buf[dev]);
        if (cudaMalloc(&buf[dev], bytes) != cudaSuccess) return nullptr;
        have[dev] = bytes;
    }
    return buf[dev];
}

cudaError_t gm_gemv(const strata::core::WSlot& w, const float* x, const float* bias, float alpha, float* y,
                    int out, int in) {
    if (w.type == 0) {
        // warp-per-row: `out` warps of work, not `out` threads
        gemv_bias_kernel<<<gm_blocks(out), GM_THREADS>>>(w.f32, x, bias, alpha, y, out, in, 1);
        return cudaGetLastError();
    }
    if (in % 32) return cudaErrorInvalidConfiguration;
    float* scratch = q8_scratch((size_t)(in / 32) * 64);
    if (!scratch) return cudaErrorMemoryAllocation;
    strata::kernels::quantize_q8_1_rows(x, 1, in, scratch, 0);
    strata::kernels::native_mmvq(w.type, w.q, scratch, y, in, out, 1, cudaStreamLegacy);
    bias_apply_kernel<<<blocks(out), GM_THREADS>>>(y, bias, alpha, out);
    return cudaGetLastError();
}

// dequantize one quantized row tensor to F32 using ggml's trait table
static bool pack_dequant_tensor(const strata::TensorInfo& t, const uint8_t* src, std::vector<float>& dst,
                                std::string& err) {
    const ggml_type ty = (ggml_type) t.type;
    const ggml_type_traits* tt = ggml_get_type_traits(ty);
    if (tt == nullptr || tt->to_float == nullptr) {
        err = "pack: no dequantizer for tensor type " + std::to_string(t.type) + " (" + t.name + ")";
        return false;
    }
    // ONE call for the whole tensor: a GGUF tensor's data is contiguous and to_float dequantizes
    // a run of k elements.  (A per-block loop striding by a full tensor ROW per block was the
    // first attempt - every block past the first read from the wrong offset, which showed up as
    // "token 0 works, token 1 is zeros".)
    dst.resize((size_t) t.elements());
    tt->to_float(src, dst.data(), (int64_t) t.elements());
    return true;
}

}  // namespace

// ================================ geometry ================================

bool Glm5Geometry::from_gguf(const strata::GgufFile& g, Glm5Geometry& out, std::string& err) {
    // two spellings in the wild: llama.cpp's canonical "glm5-next" and unsloth's "glm5next" -
    // the KV PREFIX follows the file's own arch string, so it is detected once here
    const std::string p = g.get("glm5next.block_count") ? "glm5next." : "glm5-next.";
    const auto num = [&](const std::string& key, double def, bool& ok) -> double {
        const auto* v = g.get(p + key);
        if (!v) return def;
        if (v->type == strata::MetaType::ARRAY) {
            if (v->items.empty() || !v->items[0].is_num()) return def;
            return v->items[0].num();
        }
        if (!v->is_num()) return def;
        (void) ok;
        return v->num();
    };
    const auto arr_i = [&](const std::string& key, std::vector<int>& dst, int n_layers) {
        const auto* v = g.get(p + key);
        if (v && v->type == strata::MetaType::ARRAY && v->items.size() >= (size_t) n_layers)
            // the per-layer arrays count ALL blocks (the NextN layer included); the trunk uses
            // the first n_layers entries
            for (int i = 0; i < n_layers; ++i) dst[(size_t) i] = (int) v->items[(size_t) i].u;
        else if (v && v->is_num())
            std::fill(dst.begin(), dst.end(), (int) v->u);
    };
    bool ok = true;
    out.n_embd = (int) num("embedding_length", out.n_embd, ok);
    const int block_count = (int) num("block_count", out.n_layers, ok);
    // the trunk vs the NextN/MTP block: some converters (antirez's) emit an explicit
    // trunk_block_count; others (llama.cpp's, unsloth's) count the MTP layer INSIDE block_count
    // (46 with nextn_predict_layers 1).  Prefer the explicit key, derive otherwise.
    const int trunk_explicit = (int) num("trunk_block_count", 0, ok);
    const int nextn = (int) num("nextn_predict_layers", 0, ok);
    out.nextn = nextn;
    out.n_layers = trunk_explicit > 0 ? trunk_explicit : block_count - nextn;
    out.n_head = (int) num("attention.head_count", out.n_head, ok);
    out.q_lora = (int) num("attention.q_lora_rank", out.q_lora, ok);
    out.kv_lora = (int) num("attention.kv_lora_rank", out.kv_lora, ok);
    out.qk_nope = (int) num("attention.key_length_mla", out.qk_nope, ok);
    out.v_head = (int) num("attention.value_length_mla", out.v_head, ok);
    out.kda_head_dim = (int) num("kda.head_dim", out.kda_head_dim, ok);
    out.d_conv = (int) num("ssm.conv_kernel", out.d_conv, ok);
    out.kda_lb = (float) num("kda.gate_lower_bound", out.kda_lb, ok);
    out.idx_heads = (int) num("attention.indexer.head_count", out.idx_heads, ok);
    out.idx_key = (int) num("attention.indexer.key_length", out.idx_key, ok);
    out.idx_top_k = (int) num("attention.indexer.top_k", out.idx_top_k, ok);
    out.idx_kpool = (int) num("attention.indexer.kpool", out.idx_kpool, ok);
    out.idx_select_tail = (int) num("attention.indexer.kpool_select_tail", out.idx_select_tail, ok);
    out.hc = (int) num("hyper_connection.count", out.hc, ok);
    out.sinkhorn_iters = (int) num("hyper_connection.sinkhorn_iterations", out.sinkhorn_iters, ok);
    out.hc_eps = (float) num("hyper_connection.epsilon", out.hc_eps, ok);
    out.norm_eps = (float) num("attention.layer_norm_rms_epsilon", out.norm_eps, ok);
    out.n_expert = (int) num("expert_count", out.n_expert, ok);
    out.n_exp_used = (int) num("expert_used_count", out.n_exp_used, ok);
    out.n_ff_exp = (int) num("expert_feed_forward_length", out.n_ff_exp, ok);
    out.n_shared = (int) num("expert_shared_count", out.n_shared, ok);
    out.n_ff_dense = (int) num("feed_forward_length", out.n_ff_dense, ok);
    out.dense_lead = (int) num("leading_dense_block_count", out.dense_lead, ok);
    out.w_scale = (float) num("expert_weights_scale", 1.0, ok);
    out.norm_w = (int) num("expert_weights_norm", 1, ok);
    out.swiglu_exp = (float) num("swiglu_clamp_exp", out.swiglu_exp, ok);
    out.swiglu_shexp = (float) num("swiglu_clamp_shexp", out.swiglu_shexp, ok);
    if (const auto* v = g.get(p + "vocab_size")) out.n_vocab = (int) v->u;

    // the layer-type array: 0 marks a KDA (recurrent) layer
    out.n_head_kv.assign((size_t) out.n_layers, 0);
    arr_i("attention.head_count_kv", out.n_head_kv, out.n_layers);

    // the per-layer indexer type (glm5-next.cpp L46-47: optional array, absent means every layer
    // is full; a non-full DSA layer carries no indexer tensors and reuses the previous full
    // layer's selection).  The released GLM-5.3-Flash config is "full" for all 45 layers and this
    // runner implements the full indexer only - a shared-indexer model must be refused, not
    // silently mis-served.  (The llama.cpp loader also asserts these two invariants.)
    out.indexer_full.assign((size_t) out.n_layers, 1);
    arr_i("attention.indexer.types", out.indexer_full, out.n_layers);
    for (int il = 0; il < out.n_layers; ++il) {
        if (!out.is_recr(il) && !out.indexer_full[(size_t) il]) {
            err = "glm5_model: layer " + std::to_string(il) +
                  " uses a shared (non-full) k-pool indexer; this runner implements the full "
                  "indexer only (the released GLM-5.3-Flash is full on every layer)";
            return false;
        }
    }
    if (out.idx_kpool <= 1 || out.idx_top_k % out.idx_kpool != 0) {
        err = "glm5_model: the k-pool indexer requires kpool > 1 and top_k % kpool == 0";
        return false;
    }
    if (!out.idx_select_tail) {
        err = "glm5_model: kpool_select_tail=false is not exercised by any fixture; refuse until "
              "that path is tested (the released config selects the tail)";
        return false;
    }
    // the MLA cache row width follows attention.key_length (the FULL head dim); this runner's
    // latent cache holds exactly kv_lora floats per cell, so a disagreeing key_length means the
    // GGUF's metadata contract is off (deepseek2 convention: key_length = kv_lora + rope)
    if (const auto* v = g.get(p + "attention.key_length")) {
        const double key_length = v->is_num() ? v->num() : (v->items.empty() ? -1 : v->items[0].num());
        if (key_length != (double) out.kv_lora) {
            err = "glm5_model: attention.key_length " + std::to_string(key_length) +
                  " != kv_lora_rank " + std::to_string(out.kv_lora) +
                  " (nope-only MLA holds the bare latent; a rope tail is not supported)";
            return false;
        }
    }
    if (out.qk_nope != out.v_head) {
        err = "glm5_model: this runner assumes the released nope-MLA (qk_nope == v_head)";
        return false;
    }
    if (out.hc != 4) {
        err = "glm5_model: hc != 4 is not supported (the mHC kernels pin 4 streams)";
        return false;
    }
    return true;
}

// ================================ the model ================================

Glm5Model::~Glm5Model() {
    if (split_next_) split_next_.reset();   // the tail half first (its service thread, its device)
    cudaSetDevice(dev_);   // every free below belongs to this half's device
    fast_destroy();
    if (!stage_workers_.empty()) {
        // stop the staging pool: without this the workers block in stage_cv_ forever and the
        // process hangs at exit (their std::thread destructors must see joined threads)
        {
            std::lock_guard<std::mutex> lk(stage_mu_);
            stage_quit_ = true;
        }
        stage_cv_.notify_all();
        for (auto& w : stage_workers_) w.join();
        stage_workers_.clear();
    }
    if (stream_hop_) {
        cudaStreamDestroy((cudaStream_t) stream_hop_);
        cudaEventDestroy((cudaEvent_t) ev_a_pipe_);
        cudaEventDestroy((cudaEvent_t) ev_h_pipe_);
    }
    for (void* ev : ev_h2d_)
        if (ev) cudaEventDestroy((cudaEvent_t) ev);
    if (hop_side_) cudaFree(hop_side_);
    if (timing_ && t_tokens_ > 0) {
        const double n = (double) t_tokens_;
        std::fprintf(stderr,
                     "glm timing: %lld tokens | dense %.1f ms/tok | expert tail %.1f ms/tok (pool host %.1f) "
                     "| rest %.1f ms/tok | pool hits %llu misses %llu\n",
                     (long long) t_tokens_, t_dense_ / n, t_tail_ / n, t_pool_host_ / n, t_rest_ / n,
                     (unsigned long long) pool_hits_, (unsigned long long) pool_misses_);
    }
    for (void* ev : {tev_a_, tev_b_, tev_c_, tev_d_})
        if (ev) cudaEventDestroy((cudaEvent_t) ev);
    if (d_tok_) cudaFree(d_tok_);
    if (w_arena_) cudaFree(w_arena_);
    if (mtp_arena_) cudaFree(mtp_arena_);
    if (state_) cudaFree(state_);
    if (sc_) cudaFree(sc_);
    if (d_pos_) cudaFree(d_pos_);
    if (pool_) cudaFree(pool_);
    if (dev_xq_) cudaFree(dev_xq_);
    if (dev_scratch_) cudaFree(dev_scratch_);
    if (dev_hit_out_) cudaFree(dev_hit_out_);
    if (bounce_) cudaFreeHost(bounce_);
    if (plan_host_) cudaFreeHost(plan_host_);
}

bool Glm5Model::load(const std::string& gguf_path, int64_t max_ctx, std::string& err) {
    strata::GgufFile gguf(gguf_path);
    if (!Glm5Geometry::from_gguf(gguf, g_, err)) return false;
    max_ctx_ = max_ctx;
    dev_ = 0;
    l0_ = 0;
    l1_ = g_.n_layers;

    // every tensor must be F32: quantized weights are the pack-integration step, not this runner's
    // (the NextN/MTP tensors are skipped entirely - they would waste ~2.5 GB of arena)
    auto is_nextn = [&](const std::string& name) {
        if (name.rfind("blk.", 0) != 0) return false;
        const std::string il = name.substr(4, name.find('.', 4) - 4);
        return std::all_of(il.begin(), il.end(), ::isdigit) && std::stoi(il) >= g_.n_layers;
    };
    uint64_t bytes = 0;
    for (const auto& t : gguf.tensors()) {
        if (is_nextn(t.name)) continue;
        if (t.type != 0) {
            err = "glm5_model: tensor " + t.name + " is not F32 - quantized glm5-next GGUFs need the packer "
                  "(docs/GLM5-FLASH.md Phase 3)";
            return false;
        }
        bytes += ((uint64_t) t.elements() * sizeof(float) + 255u) & ~(uint64_t) 255u;
    }
    GM_CHECK(cudaMalloc(&w_arena_, bytes), err);
    uint64_t at = 0;
    for (const auto& t : gguf.tensors()) {
        if (is_nextn(t.name)) continue;
        const uint64_t n = (uint64_t) t.elements() * sizeof(float);
        GM_CHECK(cudaMemcpy((uint8_t*) w_arena_ + at, gguf.tensor_data(t), n, cudaMemcpyHostToDevice), err);
        w_[t.name] = (const float*) ((uint8_t*) w_arena_ + at);
        ws_map_[t.name] = strata::core::WSlot{(const float*) ((uint8_t*) w_arena_ + at), 0, nullptr, 0, 0};
        at += (n + 255u) & ~(uint64_t) 255u;
    }

    // ---- the state arena
    const int64_t hc_dim = (int64_t) g_.hc * g_.n_embd;
    const int64_t max_pools = max_ctx / g_.idx_kpool;
    int64_t floats = 2 * hc_dim;   // R and its hc_post ping-pong
    kda_S_.assign(g_.n_layers, 0);
    kda_conv_.assign(g_.n_layers, 0);
    dsa_lat_.assign(g_.n_layers, 0);
    dsa_ik_.assign(g_.n_layers, 0);
    dsa_ig_.assign(g_.n_layers, 0);
    dsa_pool_.assign(g_.n_layers, 0);
    ik_ring_ = (int) max_ctx;   // the reference path keeps every position's indexer key / gate
    for (int il = 0; il < g_.n_layers; ++il) {
        if (il < l0_ || il >= l1_) continue;   // the other half's caches stay on the other device
        if (g_.is_recr(il)) {
            kda_S_[(size_t) il] = floats;
            floats += (int64_t) g_.d_inner() * g_.kda_head_dim;
            kda_conv_[(size_t) il] = floats;
            floats += (int64_t) 3 * g_.d_inner() * (g_.d_conv - 1);
        } else {
            dsa_lat_[(size_t) il] = floats;
            floats += (int64_t) g_.kv_lora * max_ctx;
            dsa_ik_[(size_t) il] = floats;
            floats += (int64_t) g_.idx_key * max_ctx;
            dsa_ig_[(size_t) il] = floats;
            floats += (int64_t) g_.idx_key * max_ctx;
            dsa_pool_[(size_t) il] = floats;
            floats += (int64_t) g_.idx_key * max_pools;
        }
    }
    state_bytes_ = (uint64_t) floats * sizeof(float);
    GM_CHECK(cudaMalloc(&state_, state_bytes_), err);

    // ---- the scratch arena (one token's activations; every region reused per token)
    const int64_t ff_max = std::max<long long>({g_.n_ff_dense, (int64_t) g_.n_ff_exp * g_.n_shared});
    int64_t s = 0;
    auto take = [&](int64_t n) {
        const int64_t at2 = s;
        s += n;
        return at2;
    };
    sc_mixed = take(g_.n_embd);
    sc_emb = take(g_.n_embd);
    sc_x = take(g_.n_embd);
    sc_pre = take(g_.hc);
    sc_post = take(g_.hc);
    sc_comb = take((int64_t) g_.hc * g_.hc);
    sc_inv = take(1);
    sc_proj = take(g_.d_inner());
    sc_conv[0] = take(g_.d_inner());
    sc_conv[1] = take(g_.d_inner());
    sc_conv[2] = take(g_.d_inner());
    sc_g = take(g_.d_inner());
    sc_g1 = take(g_.d_inner());
    sc_beta = take(g_.n_head);
    sc_tmp = take(std::max<long long>({g_.kda_head_dim, g_.q_lora, g_.idx_key, (int64_t) g_.n_expert}));
    sc_scan = take(g_.d_inner());
    sc_g2 = take(g_.d_inner());
    sc_gated = take(g_.d_inner());
    sc_qr = take(g_.q_lora);
    sc_kv = take(g_.kv_lora);
    sc_q = take((int64_t) g_.n_head * g_.qk_nope);
    sc_qabs = take((int64_t) g_.kv_lora * g_.n_head);
    sc_iq = take((int64_t) g_.idx_heads * g_.idx_key);
    sc_iw = take(g_.idx_heads);
    sc_score = take(max_pools);
    sc_cells = take(g_.n_sel_max());
    sc_attn = take((int64_t) g_.v_head * g_.n_head);
    sc_gate = take(ff_max);
    sc_up = take(ff_max);
    sc_h = take(ff_max);
    sc_dn = take(g_.n_embd);
    sc_moe = take(g_.n_embd);
    sc_sh = take(g_.n_embd);
    sc_ffn = take(g_.n_embd);
    sc_mixer = take(g_.n_embd);
    sc_head = take(g_.n_embd);
    sc_logits = take(g_.n_vocab);
    sc_ids = take(g_.n_exp_used);   // int32
    sc_rw = take(g_.n_exp_used);
    sc_bytes_ = (uint64_t) s * sizeof(float);
    GM_CHECK(cudaMalloc(&sc_, sc_bytes_), err);
    GM_CHECK(cudaMalloc(&d_pos_, sizeof(int)), err);
    loaded_ = true;
    reset();
    return true;
}

void Glm5Model::reset() {
    if (!loaded_) return;
    cudaSetDevice(dev_);
    cudaMemset(state_, 0, state_bytes_);
    pos_ = 0;
    mtp_hx_pos_ = -1;
    if (fast_ && fast_->route_error_h && cudaStreamSynchronize(fast_->cs) == cudaSuccess)
        *fast_->route_error_h = 0;   // a failed route belongs to the previous request
    if (split_next_) {
        split_next_->reset();
        cudaSetDevice(dev_);
    }
}

bool Glm5Model::forward(const std::vector<int32_t>& tokens, std::vector<float>& logits_out,
                        std::string& err) {
    if (!loaded_) {
        err = "glm5_model: not loaded";
        return false;
    }
    if (pos_ + (int64_t) tokens.size() > max_ctx_) {
        err = "glm5_model: the sequence exceeds max_ctx";
        return false;
    }
    if (fast_ != nullptr) return forward_fast(tokens, logits_out, err);
    // more than one token at a time + a split = the teacher-forced pipeline (STRATA_GLM_PIPE=0
    // forces the serial per-token path, for A/B and as a fallback)
    if (split_next_ && !split_next_->split_next_ && tokens.size() > 1 && getenv("STRATA_GLM_PIPE") == nullptr) {
        return forward_pipeline(tokens, logits_out, err);
    }
    for (int32_t t : tokens)
        if (!step(t, err)) return false;
    // the logits live on the LAST half's device (with a split, reading them from device 0 would
    // need peer access, which is closed on this driver)
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    cudaSetDevice(tail->dev_);
    logits_out.resize((size_t) g_.n_vocab);
    GM_CHECK(cudaMemcpy(logits_out.data(), tail->sc_ + tail->sc_logits, (size_t) g_.n_vocab * sizeof(float),
                        cudaMemcpyDeviceToHost), err);
    cudaSetDevice(dev_);
    return true;
}

int Glm5Model::sample_token(strata::kernels::SamplerParams& sp, std::string& err) {
    // the fast path already ran the greedy argmax on the device at the end of the forward
    if (fast_ != nullptr && sp.greedy && last_tok_ >= 0) return last_tok_;
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    if (tail->d_tok_ == nullptr) {
        err = "glm5_model: no sampler buffer";
        return -1;
    }
    if (tail->fast_ != nullptr) {
        const int tok = tail->fast_sample(sp, err);
        cudaSetDevice(dev_);
        return tok;
    }
    cudaSetDevice(tail->dev_);
    strata::kernels::sample_tokens(tail->sc_ + tail->sc_logits, 1, (int) g_.n_vocab, nullptr, 0, sp,
                                   tail->d_tok_, nullptr);
    int tok = -1;
    if (cudaMemcpy(&tok, tail->d_tok_, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = std::string("glm5_model: sampler copy: ") + cudaGetErrorString(cudaGetLastError());
        cudaSetDevice(dev_);
        return -1;
    }
    cudaSetDevice(dev_);
    return tok;
}

bool Glm5Model::load_pack_split(const std::string& pack_dir, int64_t max_ctx, const std::vector<int>& bounds,
                                const std::vector<int>& devs, std::string& err) {
    const int n = (int) devs.size();
    if (n < 2 || n > kMaxParts || (int) bounds.size() != n - 1) {
        err = "glm split: " + std::to_string(n) + " devices need " + std::to_string(std::max(0, n - 1)) +
              " boundaries (2.." + std::to_string(kMaxParts) + " devices)";
        return false;
    }
    Glm5Model* prev = nullptr;
    Glm5Model* m = this;
    std::string where;
    for (int i = 0; i < n; ++i) {
        const int l0 = i == 0 ? 0 : bounds[(size_t) i - 1], l1 = i + 1 < n ? bounds[(size_t) i] : 0;
        if (prev != nullptr) {
            prev->split_next_.reset(new Glm5Model());
            m = prev->split_next_.get();
        }
        m->n_parts_ = n;
        m->part_ = i;
        m->split_devs_ = devs;
        if (!m->load_pack(pack_dir, max_ctx, err, devs[(size_t) i], l0, l1)) {
            if (prev != nullptr) prev->split_next_.reset();
            return false;
        }
        where += (i ? ", [" : "[") + std::to_string(l0) + ", " + std::to_string(i + 1 < n ? l1 : g_.n_layers) +
                 ") on CUDA" + std::to_string(devs[(size_t) i]);
        prev = m;
    }
    // the chunks are the smallest part's (as Strata's split): the others keep their pinned staging to that, and a
    // later card that holds the first one's chunk down is named (Strata #448: a small card in a split caps every
    // stage's chunk - prompts read slower than on the first card alone)
    int tmin = 0;
    for (Glm5Model* p = this; p != nullptr; p = p->split_next_.get())
        if (p->prefill_chunk() > 0) tmin = tmin == 0 ? p->prefill_chunk() : std::min(tmin, p->prefill_chunk());
    if (tmin > 0) {
        const int alone = prefill_chunk();
        for (Glm5Model* p = split_next_.get(); p != nullptr; p = p->split_next_.get())
            if (alone > tmin && p->prefill_chunk() == tmin) {
                cudaDeviceProp prop{};
                if (cudaGetDeviceProperties(&prop, p->dev_) != cudaSuccess) {
                    cudaGetLastError();
                    prop.name[0] = 0;
                }
                std::fprintf(stderr, "glm prefill: WARNING: prompt chunk %d tokens, not %d: CUDA%d (%s) can lend only "
                                     "that much of its expert pool - prompts read slower than on CUDA%d alone\n",
                             tmin, alone, p->dev_, prop.name, dev_);
            }
        for (Glm5Model* p = this; p != nullptr; p = p->split_next_.get()) p->prefill_cap(tmin, "the split's smallest");
    }
    cudaSetDevice(dev_);
    std::fprintf(stderr, "glm split: layers %s (one 64 KB host hop per boundary per token)\n", where.c_str());
    return true;
}

// "auto" across more than two devices: each later part starts where the layers before it - weighed by their
// experts' bytes (native_experts.txt) plus an even share of the dense weights - reach the devices before it's share
// of the free VRAM, so each device's expert pool holds about the same fraction of its own layers' experts
static std::vector<int> glm_auto_bounds(const std::string& pack_dir, int n_layers, const std::vector<int>& devs) {
    std::vector<double> cost((size_t) n_layers, 0.2e9);   // ~8.6 GB of dense weights over 45 layers
    {
        std::ifstream ne(pack_dir + "/native_experts.txt");
        std::string line;
        double n_expert = 0;
        while (std::getline(ne, line)) {
            if (line.rfind("#", 0) == 0) {
                const size_t k = line.find("n_expert ");
                if (k != std::string::npos) n_expert = std::atof(line.c_str() + k + 9);
                continue;
            }
            std::istringstream ss(line);
            long long layer = -1, gu = 0, dt = 0, off = 0, blob = 0;
            if (ss >> layer >> gu >> dt >> off >> blob && layer >= 0 && layer < n_layers)
                cost[(size_t) layer] += (double) blob * n_expert;
        }
    }
    int cur = 0;
    cudaGetDevice(&cur);
    std::vector<double> cap;
    for (int d : devs) {
        size_t fr = 0, tot = 0;
        cudaSetDevice(d);
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) cudaGetLastError();
        cap.push_back(std::max(1.0e9, (double) fr - 2.0e9));   // less what a part needs besides its layers' weights
    }
    cudaSetDevice(cur);
    const int n = (int) devs.size();
    std::vector<double> cum((size_t) n_layers + 1, 0.0);
    for (int l = 0; l < n_layers; ++l) cum[(size_t) l + 1] = cum[(size_t) l] + cost[(size_t) l];
    const double cap_sum = std::accumulate(cap.begin(), cap.end(), 0.0);
    std::vector<int> bounds;
    double cap_before = 0;
    for (int i = 0; i + 1 < n; ++i) {
        cap_before += cap[(size_t) i];
        const double target = cum[(size_t) n_layers] * cap_before / cap_sum;
        const int lo = bounds.empty() ? 1 : bounds.back() + 1, hi = n_layers - (n - 1 - i);
        int best = lo;
        for (int l = lo; l <= hi; ++l)
            if (std::fabs(cum[(size_t) l] - target) < std::fabs(cum[(size_t) best] - target)) best = l;
        bounds.push_back(best);
    }
    return bounds;
}

// the host's RAM read speed with every CPU reading at once (bytes/s), each thread's slice first touched by that
// thread: what the CPU lanes compute the experts VRAM does not hold from - the split search's price of a miss
static double host_read_bps() {
    const int T = (int) std::max(1u, std::thread::hardware_concurrency());
    const size_t words = ((size_t) 512 << 20) / sizeof(uint64_t);
    std::unique_ptr<uint64_t[]> buf(new (std::nothrow) uint64_t[words]);
    if (!buf) return 0.0;
    constexpr int kReps = 4;   // the first pass touches the pages, the rest are timed
    std::atomic<int> phase{0}, arrived{0};
    std::atomic<uint64_t> sink{0};
    std::vector<std::thread> th;
    for (int i = 0; i < T; ++i)
        th.emplace_back([&, i] {
            const size_t a = words * (size_t) i / (size_t) T, b = words * (size_t) (i + 1) / (size_t) T;
            for (int ph = 1; ph <= kReps; ++ph) {
                while (phase.load(std::memory_order_acquire) < ph) std::this_thread::yield();
                uint64_t sum = 0;
                if (ph == 1)
                    for (size_t j = a; j < b; ++j) buf[j] = j;
                else
                    for (size_t j = a; j < b; ++j) sum += buf[j];
                sink.fetch_add(sum, std::memory_order_relaxed);
                arrived.fetch_add(1, std::memory_order_acq_rel);
            }
        });
    double best = 0.0;
    for (int ph = 1; ph <= kReps; ++ph) {
        arrived.store(0);
        const auto t0 = std::chrono::steady_clock::now();
        phase.store(ph, std::memory_order_release);
        while (arrived.load(std::memory_order_acquire) < T) std::this_thread::yield();
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (ph > 1 && sec > 0) best = std::max(best, (double) (words * sizeof(uint64_t)) / sec);
    }
    for (auto& t : th) t.join();
    return sink.load() == 1 ? best + 1.0 : best;   // (the sum keeps the reads)
}

// ---------------------------------------------------------------- --layer-split auto: Strata's split search
// Every placement of the layers on the GPUs is priced by the decode time it predicts, and the cheapest one that can
// start is taken - Strata's `--layer-split auto` (src/program/generate.cpp), with this engine's own measured terms
// where Strata's are fits to its model:
//   - a part's room for experts: its free VRAM less its layers' dense weights and state at the context (index.txt and
//     the GGUF directory, as load_pack sizes them), the head and the draft block on the last part, its scratch, and
//     the reserve fast_setup keeps (pool_avail); the pool gives every layer the same number of slots;
//   - the experts those slots hold: each layer's most routed, their share of its routes read off this machine's usage
//     counts (expert_usage.txt; the pack's expert_counts.txt without them; Strata's coverage curve 1 - (1 - f)^3,
//     STRATA_SPLIT_COVER_B, without either - that is what Strata has to assume, its profile carries only a ranking);
//   - a layer's time on its card: the bytes it reads (its dense weights and its VRAM experts' share of the routed
//     ones) over the card's memory bandwidth - Strata scales a fitted per-layer time by SMs x clock; and each miss:
//     the expert's bytes over the parts' share of the host's RAM read speed, measured here - the CPU lane computes
//     the experts VRAM does not hold (Strata prices a fitted miss over the PCIe link);
//   - a token: the parts' times added (Strata), the slowest part breaking a tie - except the pipelined decode of a
//     two-part split with the draft block (STRATA_GLM_MTP_PIPELINE=1), where the GPUs work on consecutive tokens:
//     there a token costs the slower part (the draft on the last one), and the sum breaks the tie;
//   - the startability gate (Strata #1094): every part keeps its pool's floor and can lend a 512-token prompt chunk.
// Two to four GPUs try every placement (memoised per part and range, as Strata's four-way search); more keep the
// proportional split (glm_auto_bounds).  STRATA_GLM_SPLIT_LOG=1 prints every placement's prediction.
static std::vector<int> glm_search_bounds(const std::string& pack_dir, int64_t max_ctx, const std::vector<int>& devs) {
    std::string err;
    std::vector<std::string> names;
    if (!pack_shard_names(pack_dir, names, err)) return {};
    std::vector<std::unique_ptr<strata::GgufFile>> gfs;
    Glm5Geometry g;
    try {
        for (const auto& nm : names) gfs.emplace_back(new strata::GgufFile(pack_dir + "/../" + nm));
        if (!Glm5Geometry::from_gguf(*gfs[0], g, err)) return {};
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm split auto: %s - the split by free VRAM instead\n", e.what());
        return {};
    }
    const int L = g.n_layers, n = (int) devs.size();
    if (n < 2 || L < n) return {};
    const bool fast = getenv("STRATA_GLM_SLOW") == nullptr;
    const auto find = [&](const std::string& nm) -> const strata::TensorInfo* {
        for (const auto& gf : gfs)
            if (const strata::TensorInfo* t = gf->find(nm)) return t;
        return nullptr;
    };
    const auto tensor_bytes = [](const strata::TensorInfo* t) {
        return (double) ggml_row_size((ggml_type) t->type, t->shape[0]) * (double) (t->elements() / t->shape[0]);
    };
    // the dense weights: per layer, the rows every part loads, the head's (the last part) and token_embd (the first,
    // off the fast path)
    std::vector<double> dense((size_t) L + 1, 0.0);
    double common = 0, head = 0, embd = 0;
    {
        std::ifstream ix(pack_dir + "/index.txt");
        std::string line;
        while (std::getline(ix, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string nm, served, kind;
            ss >> nm >> served >> kind;
            const strata::TensorInfo* t = find(nm);
            if (t == nullptr) continue;
            const double b = (double) pack_row_vram(nm, kind, t, fast);
            if (nm.rfind("blk.", 0) == 0) {
                const int il = std::atoi(nm.c_str() + 4);
                if (il >= 0 && il < L) dense[(size_t) il] += b;
            } else if (nm == "output.weight" || nm == "output_norm.weight") {
                head += b;
            } else if (nm == "token_embd.weight") {
                embd += b;
            } else {
                common += b;
            }
        }
    }
    // the state at the context, as load_pack lays it out
    const double max_pools = (double) (max_ctx / g.idx_kpool);
    const double lat = fast ? (double) g.kv_lora * (double) max_ctx / 2 : (double) g.kv_lora * (double) max_ctx;
    const double dsa_state = 4.0 * (lat + 2.0 * g.idx_key * (double) max_ctx + (double) g.idx_key * max_pools);
    const double kda_state = 4.0 * ((double) g.d_inner() * g.kda_head_dim + 3.0 * g.d_inner() * (g.d_conv - 1));
    // the experts: a layer's blob and its VRAM slot's stride (native_experts.txt)
    std::vector<double> blob((size_t) L + 1, 0.0), stride((size_t) L + 1, 0.0);
    {
        std::ifstream ne(pack_dir + "/native_experts.txt");
        std::string line;
        while (std::getline(ne, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            long long il = -1, gu = 0, dt = 0, off = 0, b = 0;
            if (!(ss >> il >> gu >> dt >> off >> b) || il < 0 || il >= L) continue;
            blob[(size_t) il] = (double) b;
            stride[(size_t) il] = (double) glmfast::expert_stride((size_t) b, (int) gu, (int) dt);
        }
    }
    // the NextN draft block: the last part of a two-part split carries it (load_pack), from the model's GGUF or
    // STRATA_GLM_MTP_GGUF's; the pipelined decode drafts with it once a token
    bool mtp = false;
    double mtp_dense = 0;
    std::unique_ptr<strata::GgufFile> extra;
    if (fast && n == 2 && getenv("STRATA_GLM_NO_MTP") == nullptr) {
        const std::string P = "blk." + std::to_string(L) + ".";
        const char* mg = getenv("STRATA_GLM_MTP_GGUF");
        if (find(P + "nextn.eh_proj.weight") == nullptr && mg != nullptr && mg[0] != '\0') try {
                extra.reset(new strata::GgufFile(mg));
            } catch (const std::exception&) {
                extra.reset();
            }
        const auto findm = [&](const std::string& nm) -> const strata::TensorInfo* {
            if (const strata::TensorInfo* t = find(nm)) return t;
            return extra ? extra->find(nm) : nullptr;
        };
        uint64_t b = 0;
        std::string e2;
        const strata::TensorInfo* tg = findm(P + "ffn_gate_exps.weight");
        const strata::TensorInfo* tu = findm(P + "ffn_up_exps.weight");
        const strata::TensorInfo* td = findm(P + "ffn_down_exps.weight");
        if ((g.nextn > 0 || extra) && findm(P + "nextn.eh_proj.weight") != nullptr && tg && tu && td &&
            mtp_dense_vram(findm, L, b, e2)) {
            blob[(size_t) L] = (tensor_bytes(tg) + tensor_bytes(tu) + tensor_bytes(td)) / g.n_expert;
            stride[(size_t) L] = (double) glmfast::expert_stride((size_t) blob[(size_t) L], (int) tg->type, (int) td->type);
            mtp_dense = (double) b;
            mtp = true;
        }
    }
    // (the MTP decode verifies a window through the parts in turn: a token costs them added, as without drafts)
    const bool pipelined = n == 2 && mtp && getenv("STRATA_GLM_NO_SPEC") == nullptr &&
                           getenv("STRATA_GLM_MTP_PIPELINE") != nullptr;
    // each layer's routes per expert: this machine's usage, else the pack's routing profile
    std::vector<std::vector<double>> cnt((size_t) L + 1);
    {
        const char* u = getenv("STRATA_GLM_USAGE");
        const std::string up = u != nullptr ? (std::string(u) == "0" ? std::string() : std::string(u))
                                            : pack_dir + "/expert_usage.txt";
        std::ifstream uf(up);
        std::string line;
        while (!up.empty() && std::getline(uf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int il = -1;
            ss >> il;
            if (il < 0 || il > L) continue;
            std::string kv;
            cnt[(size_t) il].assign((size_t) g.n_expert, 0.0);
            while (ss >> kv) {
                const size_t c = kv.find(':');
                const int e = c == std::string::npos ? -1 : std::atoi(kv.c_str());
                if (e >= 0 && e < g.n_expert) cnt[(size_t) il][(size_t) e] = std::atof(kv.c_str() + c + 1);
            }
        }
        std::ifstream cf(pack_dir + "/expert_counts.txt");
        while (std::getline(cf, line)) {
            std::istringstream ss(line);
            int il = -1;
            ss >> il;
            if (il < 0 || il > L || !cnt[(size_t) il].empty()) continue;
            double c = 0;
            while (ss >> c) cnt[(size_t) il].push_back(c);
        }
    }
    // cover[l][s]: the share of layer l's routes its s most routed experts take
    std::vector<std::vector<double>> cover((size_t) L + 1);
    for (int l = 0; l <= L; ++l) {
        std::vector<double> c = cnt[(size_t) l];
        const double tot = std::accumulate(c.begin(), c.end(), 0.0);
        if (tot <= 0) continue;
        std::sort(c.rbegin(), c.rend());
        cover[(size_t) l].assign(c.size() + 1, 0.0);
        for (size_t i = 0; i < c.size(); ++i) cover[(size_t) l][i + 1] = cover[(size_t) l][i] + c[i] / tot;
    }
    const double cover_b = getenv("STRATA_SPLIT_COVER_B") ? std::atof(getenv("STRATA_SPLIT_COVER_B")) : 3.0;
    const auto held_share = [&](int l, int64_t s) -> double {
        s = std::clamp<int64_t>(s, 0, g.n_expert);
        const auto& cv = cover[(size_t) l];
        if (!cv.empty()) return cv[(size_t) std::min<int64_t>(s, (int64_t) cv.size() - 1)];
        return 1.0 - std::pow(1.0 - (double) s / g.n_expert, cover_b);
    };
    // the cards: free VRAM, memory bandwidth (the bus: 2 x memory clock x width); the host's RAM read speed
    struct Card {
        int dev;
        double free, total, bw;
        std::string name;
    };
    std::vector<Card> cards;
    int cur = 0;
    cudaGetDevice(&cur);
    for (int d : devs) {
        Card c{d, 0, 0, 0, ""};
        size_t fr = 0, tot = 0;
        cudaSetDevice(d);
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) cudaGetLastError();
        int khz = 0, bits = 0;
        if (cudaDeviceGetAttribute(&khz, cudaDevAttrMemoryClockRate, d) != cudaSuccess) khz = 0;
        if (cudaDeviceGetAttribute(&bits, cudaDevAttrGlobalMemoryBusWidth, d) != cudaSuccess) bits = 0;
        cudaGetLastError();
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) c.name = prop.name;
        cudaGetLastError();
        c.free = (double) fr;
        c.total = (double) tot;
        c.bw = 2.0 * (double) khz * 1e3 * (double) bits / 8.0;
        cards.push_back(c);
    }
    cudaSetDevice(cur);
    for (auto& c : cards)
        if (c.bw <= 0) c.bw = 1e11;   // (no reading: every card the same, and the misses decide)
    const double host_bps = host_read_bps();
    const double miss_bps = std::max(1e9, (host_bps > 0 ? host_bps : 2e10) / n);   // each part's lane, all at once
    double max_stride = 0;
    for (double st : stride) max_stride = std::max(max_stride, st);
    const double fixed = 8.0 * max_stride + (double) (256u << 20);   // the scratch slots and the activations
    constexpr int kSparesHere = 3;   // gf::kSpares: slots that hold no resident expert
    const int64_t keep = g.n_exp_used + kSparesHere + 2;
    const auto moe_layer = [&](int l) { return l >= g.dense_lead && blob[(size_t) l] > 0; };
    struct Eval {
        double ms = 0, routes = 0, hits = 0;
        int64_t per = 0;
        bool ok = false;
    };
    // part i running layers [lb, le), memoised (a part's fill and time depend on its own range and card only)
    std::map<std::tuple<int, int, int>, Eval> memo;
    const auto stage = [&](int i, int lb, int le) -> Eval {
        const auto key = std::make_tuple(i, lb, le);
        if (const auto it = memo.find(key); it != memo.end()) return it->second;
        const bool last = i == n - 1;
        const Card& c = cards[(size_t) i];
        double used = common + fixed + 4.0 * 2 * g.hc * g.n_embd + (i == 0 ? embd : 0.0);
        double bsum = 0, ssum = 0, smax = 0;
        bool kda = false, dsa = false, dense_ffn = false, moe = false;
        for (int l = lb; l < le; ++l) {
            used += dense[(size_t) l] + (g.is_recr(l) ? kda_state : dsa_state);
            (g.is_recr(l) ? kda : dsa) = true;
            if (moe_layer(l)) {
                bsum += blob[(size_t) l];
                ssum += stride[(size_t) l];
                smax = std::max(smax, stride[(size_t) l]);
                moe = true;
            } else {
                dense_ffn = true;
            }
        }
        if (last) {
            used += head;
            if (mtp) {
                used += mtp_dense + dsa_state;
                bsum += blob[(size_t) L];
                ssum += stride[(size_t) L];
                smax = std::max(smax, stride[(size_t) L]);
                dsa = true;
            }
        }
        const double room =
            c.free > used ? (double) Glm5Model::pool_avail((size_t) (c.free - used), (size_t) c.total) : 0.0;
        Eval ev;
        ev.per = bsum > 0 ? std::min<int64_t>(g.n_expert, (int64_t) (room / bsum)) : 0;
        while (ev.per > 0 && (double) ev.per * ssum > room) --ev.per;
        // startable: the pool's floor, and a 512-token prompt chunk lent at --prefill auto's cap
        ev.ok = ssum == 0 || ev.per >= keep;
        if (ev.ok && ssum > 0) {
            const double need = (double) glm_prefill_lend_bytes(g, 512, kda, dsa, dense_ffn, moe, last && mtp, max_ctx,
                                                                (size_t) smax);
            const int64_t k = (int64_t) std::ceil(need / ssum);
            ev.ok = k + keep <= ev.per && k * 100 <= 90 * ev.per;
        }
        // a token on this part: each layer's reads over the card's bandwidth, its misses over the lane's share
        const int64_t held = std::max<int64_t>(0, ev.per - kSparesHere);
        const auto layer_ms = [&](int l, double dense_b) {
            double gpu = dense_b, cpu = 0;
            if (l == L || moe_layer(l)) {
                const double h = g.n_exp_used * held_share(l, held);
                gpu += h * blob[(size_t) l];
                cpu = (g.n_exp_used - h) * blob[(size_t) l];
                ev.hits += h;
                ev.routes += g.n_exp_used;
            }
            return 1e3 * (gpu / c.bw + cpu / miss_bps);
        };
        for (int l = lb; l < le; ++l) ev.ms += layer_ms(l, dense[(size_t) l]);
        if (last) {
            ev.ms += 1e3 * head / c.bw;
            if (pipelined) ev.ms += layer_ms(L, mtp_dense);   // the draft, once a token
        }
        memo[key] = ev;
        return ev;
    };
    // a placement: {objective, tie-break, startable}; parts' times in `ms`
    const auto price = [&](const std::vector<int>& b, std::vector<Eval>& ev, double& obj, double& tie) -> bool {
        ev.clear();
        double sum = 0, slow = 0;
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            const int lb = i == 0 ? 0 : b[(size_t) i - 1], le = i + 1 < n ? b[(size_t) i] : L;
            ev.push_back(stage(i, lb, le));
            sum += ev.back().ms;
            slow = std::max(slow, ev.back().ms);
            ok = ok && ev.back().ok;
        }
        obj = pipelined ? slow : sum;
        tie = pipelined ? sum : slow;
        return ok;
    };
    static const bool split_log = getenv("STRATA_GLM_SPLIT_LOG") != nullptr;
    std::vector<int> best, best_any, b((size_t) n - 1);
    double best_obj = 1e30, best_tie = 1e30, any_obj = 1e30, any_tie = 1e30;
    const double eps = 1e-6;
    std::vector<Eval> ev;
    const auto consider = [&]() {
        double obj = 0, tie = 0;
        const bool ok = price(b, ev, obj, tie);
        if (split_log) {
            std::string s;
            for (int k : b) s += (s.empty() ? "" : ",") + std::to_string(k);
            std::fprintf(stderr, "glm split auto: %s -> %.2f ms%s\n", s.c_str(), obj, ok ? "" : " (cannot start)");
        }
        const auto better = [&](double o, double t, double bo, double bt) {
            return o < bo - eps || (o <= bo + eps && t < bt - eps);
        };
        if (better(obj, tie, any_obj, any_tie)) { best_any = b; any_obj = obj; any_tie = tie; }
        if (ok && better(obj, tie, best_obj, best_tie)) { best = b; best_obj = obj; best_tie = tie; }
    };
    if (n == 2) {
        for (int k = 1; k < L; ++k) { b = {k}; consider(); }
    } else if (n == 3) {
        for (int k1 = 1; k1 + 1 < L; ++k1)
            for (int k2 = k1 + 1; k2 < L; ++k2) { b = {k1, k2}; consider(); }
    } else if (n == 4) {
        for (int k1 = 1; k1 + 2 < L; ++k1)
            for (int k2 = k1 + 1; k2 + 1 < L; ++k2)
                for (int k3 = k2 + 1; k3 < L; ++k3) { b = {k1, k2, k3}; consider(); }
    } else {
        return {};
    }
    for (const auto& c : cards)
        std::fprintf(stderr, "glm split auto: CUDA%d (%s) memory %.0f GB/s, %.2f GB free\n", c.dev, c.name.c_str(),
                     c.bw / 1e9, c.free / 1073741824.0);
    std::fprintf(stderr, "glm split auto: the host reads RAM at %.0f GB/s (a part's CPU lane: %.0f); the routes from %s; "
                         "%s\n", host_bps / 1e9, miss_bps / 1e9,
                 cover[3].empty() ? "the default coverage curve (no usage counts)" : "the usage counts",
                 pipelined ? "the pipelined decode: a token costs the slower part" : "a token costs the parts added");
    if (best.empty()) {
        if (best_any.empty()) return {};
        std::fprintf(stderr, "glm split auto: WARNING no placement leaves every part its pool's floor and a 512-token "
                             "prompt chunk - the fastest one anyway (a smaller --max-context leaves more room)\n");
        best = best_any;
    }
    double obj = 0, tie = 0;
    price(best, ev, obj, tie);
    std::string where, parts;
    double hits = 0, routes = 0;
    for (int i = 0; i < n; ++i) {
        const int lb = i == 0 ? 0 : best[(size_t) i - 1], le = i + 1 < n ? best[(size_t) i] : L;
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s[%d, %d) on CUDA%d", i ? ", " : "", lb, le, cards[(size_t) i].dev);
        where += buf;
        std::snprintf(buf, sizeof buf, "%sCUDA%d %.1f ms, %lld slots a layer", i ? " | " : "", cards[(size_t) i].dev,
                      ev[(size_t) i].ms, (long long) ev[(size_t) i].per);
        parts += buf;
        hits += ev[(size_t) i].hits;
        routes += ev[(size_t) i].routes;
    }
    std::fprintf(stderr, "glm split auto: layers %s - a token %.1f ms predicted (%s), %.0f%% of the routed experts in "
                         "VRAM\n", where.c_str(), obj, parts.c_str(), routes > 0 ? 100.0 * hits / routes : 0.0);
    return best;
}

bool Glm5Model::load_pack_env(const std::string& pack_dir, int64_t max_ctx, std::string& err,
                              const std::string& layer_split) {
    // the layer split is the DEFAULT on multi-GPU hosts (measured at steady state, 640-token GEN on
    // real text: 182 vs 840 MB/tok device bytes, decode 195 vs 506 ms/tok, hit 87-90 vs 64.4% - the
    // pools are per-part, so 2x capacity halves the demand reads).  STRATA_GLM_SPLIT=0 forces one
    // device; =<layer>[,<layer>..] pins the boundaries; STRATA_GLM_DEV1 picks the second device of a
    // two-part split (default 1), STRATA_GLM_DEVS the devices of any split.
    const char* sp = getenv("STRATA_GLM_SPLIT");
    const std::string spec = sp != nullptr ? std::string(sp) : layer_split;
    const auto ints = [](const std::string& s, std::vector<int>& out) {
        out.clear();
        std::stringstream ss(s);
        std::string t;
        while (std::getline(ss, t, ','))
            if (t.empty() || t.find_first_not_of("0123456789 ") != std::string::npos) return false;
            else out.push_back(std::atoi(t.c_str()));
        return !out.empty();
    };
    if (spec == "0" || (sp != nullptr && spec != "auto" && spec.find(',') == std::string::npos && std::atoi(sp) == 0))
        return load_pack(pack_dir, max_ctx, err);
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess) n_dev = 0;
    cudaGetLastError();
    std::vector<int> bounds, devs;
    if (const char* dv = getenv("STRATA_GLM_DEVS"); dv != nullptr && !ints(dv, devs)) {
        err = std::string("STRATA_GLM_DEVS: not a device list: ") + dv;
        return false;
    }
    if (!spec.empty() && spec != "auto") {
        if (!ints(spec, bounds)) {
            err = "glm split: not a layer list (K or K1,K2,..): " + spec;
            return false;
        }
        if (devs.empty())
            for (int i = 0; i <= (int) bounds.size(); ++i) devs.push_back(i);
    } else {
        if (devs.empty())
            for (int d = 0; d < std::min(n_dev, (int) kMaxParts); ++d) devs.push_back(d);
        if (devs.size() < 2) return load_pack(pack_dir, max_ctx, err, devs.empty() ? 0 : devs[0]);
        if (devs.size() == 2 && getenv("STRATA_GLM_DEVS") == nullptr)
            if (const char* d1 = getenv("STRATA_GLM_DEV1")) devs[1] = std::atoi(d1);
        // --layer-split auto: Strata's split search over two to four GPUs; more (or a pack it cannot read) share the
        // layers by free VRAM - two GPUs at the midpoint, plus two layers for the head half when the tail also runs
        // the NextN draft block (2x V100, split 22 / 23 / 24 / 25 / 26 -> 36.3 / 36.7 / 40.0 / 37.3 / 39.4 tok/s)
        if (devs.size() <= 4) bounds = glm_search_bounds(pack_dir, max_ctx, devs);
        if (bounds.empty() && devs.size() == 2)
            bounds = {g_.n_layers / 2 + (getenv("STRATA_GLM_NO_MTP") == nullptr && getenv("STRATA_GLM_NO_SPEC") == nullptr &&
                                                 getenv("STRATA_GLM_MTP_PIPELINE") != nullptr
                                             ? 2
                                             : 0)};
        if (bounds.empty()) bounds = glm_auto_bounds(pack_dir, g_.n_layers, devs);
    }
    if (devs.size() == 2 && getenv("STRATA_GLM_DEVS") == nullptr && !spec.empty() && spec != "auto")
        if (const char* d1 = getenv("STRATA_GLM_DEV1")) devs[1] = std::atoi(d1);
    bool ok = devs.size() == bounds.size() + 1;
    for (size_t i = 0; ok && i < bounds.size(); ++i)
        ok = bounds[i] >= 1 && bounds[i] < g_.n_layers && (i == 0 || bounds[i] > bounds[i - 1]);
    for (size_t i = 0; ok && i < devs.size(); ++i) {
        ok = devs[i] >= 0 && devs[i] < n_dev;
        for (size_t j = 0; ok && j < i; ++j) ok = devs[i] != devs[j];
    }
    if (!ok) {
        err = "glm split: " + std::to_string(bounds.size()) + " boundaries (rising, 1.." +
              std::to_string(g_.n_layers - 1) + ") need " + std::to_string(bounds.size() + 1) +
              " distinct devices of the " + std::to_string(n_dev) + " visible";
        return false;
    }
    return load_pack_split(pack_dir, max_ctx, bounds, devs, err);
}

// GLM_CB_DIR seam dump: when set, writes <dir>/<name>-<layer>.f32 for the named seams (the same
// nodes llama.cpp's cb_eval dumps) so the python diff can bisect the first divergent component.
// The MoE dumps (glm_moe_*, glm_exp*) name their own seams - see NEXT-STEPS.md for the diff recipe.
static void cb_dump(const std::string& name, const void* dev, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    if (!dir || !dir[0] || !dev) return;
    static std::vector<float> buf;
    buf.resize((size_t) n);
    cudaMemcpy(buf.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    std::FILE* f = std::fopen((std::string(dir) + "/" + name + ".f32").c_str(), "wb");
    if (f) { std::fwrite(buf.data(), 4, (size_t) n, f); std::fclose(f); }
}

// same, for values that already live on the host (the CPU expert path)
static void cb_dump_h(const std::string& name, const void* host, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    if (!dir || !dir[0] || !host) return;
    std::FILE* f = std::fopen((std::string(dir) + "/" + name + ".f32").c_str(), "wb");
    if (f) { std::fwrite(host, 4, (size_t) n, f); std::fclose(f); }
}

// STRATA_GLM_TIMING=1: record a phase event and surface a failure AT THE SITE (the error would
// otherwise latch and be blamed on whatever ran next)
static bool tev_rec(void* ev, const char* what) {
    if (cudaEventRecord((cudaEvent_t) ev, 0) == cudaSuccess) return true;
    std::fprintf(stderr, "glm timing: record %s failed: %s\n", what, cudaGetErrorString(cudaGetLastError()));
    return false;
}

bool Glm5Model::step_issue(int32_t token, int64_t& p, std::string& err) {
    cudaSetDevice(dev_);
    const Glm5Geometry& g = g_;
    p = pos_++;
    if (l0_ == 0) {
        // ---- the embedding row -> the hc stream copies (only the first half embeds)
        float* const S = state_;
        float* const A = sc_;
        float* R = S;   // the residual stack lives at the state arena's head
        float* emb = A + sc_emb;
        if (pack_) {
            // the pack keeps token_embd QUANTIZED (2.2 GB of F32 the expert pool gets instead): the
            // one row this token needs is dequantized from the host mapping with ggml's own traits
            static std::vector<float> emb_host;
            emb_host.resize((size_t) g.n_embd);
            const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
            if (tt == nullptr || tt->to_float == nullptr) {
                err = "pack: no dequantizer for the quantized token_embd";
                return false;
            }
            tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd),
                         emb_host.data(), g.n_embd);
            if (getenv("STRATA_GLM_EMB_PROBE") && token < 3) {
                const size_t rb = ggml_row_size((ggml_type) pack_emb_type_, g.n_embd);
                const uint8_t* rp = pack_emb_src_ + (size_t) token * rb;
                std::fprintf(stderr, "emb probe: token %d type %d row_bytes %zu src+0: %02x %02x %02x %02x "
                                     "vals %.6f %.6f %.6f %.6f\n",
                             token, pack_emb_type_, rb, rp[0], rp[1], rp[2], rp[3], emb_host[0], emb_host[1],
                             emb_host[2], emb_host[3]);
            }
            GM_CHECK(cudaMemcpy(emb, emb_host.data(), (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice), err);
        } else {
            GM_CHECK(cudaMemcpy(emb, w_.at("token_embd.weight") + (int64_t) g.n_embd * token,
                                (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice), err);
        }
        for (int s = 0; s < g.hc; ++s)
            GM_CHECK(cudaMemcpy(R + (int64_t) g.n_embd * s, emb, (size_t) g.n_embd * sizeof(float),
                                cudaMemcpyDeviceToDevice), err);
    }
    return step_layers(p, err);
}

bool Glm5Model::step(int32_t token, std::string& err) {
    cudaSetDevice(dev_);
    const Glm5Geometry& g = g_;
    int64_t p = 0;
    if (!step_issue(token, p, err)) return false;
    // hand the residual across the devices: one 64 KB host hop per boundary (peer access is closed on
    // this driver, and a single boundary copy per token does not need it)
    const size_t hop_floats = (size_t) g.hc * (size_t) g.n_embd;
    Glm5Model* prev = this;
    for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) {
        hop_.resize(hop_floats);
        cudaSetDevice(prev->dev_);
        GM_CHECK(cudaMemcpy(hop_.data(), prev->state_, hop_floats * sizeof(float), cudaMemcpyDeviceToHost), err);
        cudaSetDevice(m->dev_);
        GM_CHECK(cudaMemcpy(m->state_, hop_.data(), hop_floats * sizeof(float), cudaMemcpyHostToDevice), err);
        if (!m->step_layers(p, err)) return false;
        prev = m;
    }
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::forward_pipeline(const std::vector<int32_t>& tokens, std::vector<float>& logits_out,
                                 std::string& err) {
    const Glm5Geometry& g = g_;
    const size_t n = tokens.size();
    if (stream_hop_ == nullptr) {
        cudaSetDevice(dev_);
        cudaStream_t s = nullptr;
        cudaEvent_t ea = nullptr, eh = nullptr;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ea, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&eh, cudaEventDisableTiming) != cudaSuccess) {
            err = "glm5_model: the pipeline stream/events did not create";
            return false;
        }
        stream_hop_ = s;
        ev_a_pipe_ = ea;
        ev_h_pipe_ = eh;
    }
    const size_t hop_floats = (size_t) g.hc * (size_t) g.n_embd;
    // double-buffered host hops: the H2D into device 1 is ASYNC (a blocking one made the host wait
    // for the whole previous B-half every token and the pipeline degraded to serial), so slot t&1
    // is not reused until device 1 has finished reading it (a cross-device event wait)
    hop_.resize(hop_floats * 2);
    if (ev_h2d_[0] == nullptr) {
        cudaSetDevice(split_next_->dev_);
        for (int i = 0; i < 2; ++i)
            if (cudaEventCreateWithFlags((cudaEvent_t*) &ev_h2d_[i], cudaEventDisableTiming) != cudaSuccess) {
                err = "glm5_model: pipeline H2D events did not create";
                return false;
            }
        cudaSetDevice(dev_);
    }
    // the end-copy side buffer: the hop's D2H must not read state_ while A(t+1) (queued right behind
    // the event) overwrites it - so A(t)'s residual is copied to side[t&1] ON DEV0'S OWN STREAM
    // before the event, and the D2H reads the side buffer; double-buffered so A(t+1)'s end-copy (the
    // other slot) cannot race A(t)'s D2H either
    if (hop_side_ == nullptr) {
        if (cudaMalloc(&hop_side_, hop_floats * 2 * sizeof(float)) != cudaSuccess) {
            err = "glm5_model: the pipeline side buffer did not allocate";
            return false;
        }
    }
    float* side = (float*) hop_side_;
    // A(0) first; then per token: end-copy this token's residual, mark it complete, QUEUE THE NEXT
    // TOKEN'S FIRST HALF BEHIND THAT MARK, and only then wait for this token's residual - so device 0
    // computes token t+1 while device 1 computes token t (and the host's pool work overlaps both)
    int64_t p_cur = 0, p_next = 0;
    if (!step_issue(tokens[0], p_cur, err)) return false;
    for (size_t t = 0; t < n; ++t) {
        const size_t slot = t & 1;
        cudaSetDevice(dev_);
        GM_CHECK(cudaMemcpyAsync(side + slot * hop_floats, state_, hop_floats * sizeof(float),
                                 cudaMemcpyDeviceToDevice, 0), err);
        if (cudaEventRecord((cudaEvent_t) ev_a_pipe_, 0) != cudaSuccess) {
            err = "glm5_model: pipeline event record failed";
            return false;
        }
        if (t + 1 < n && !step_issue(tokens[t + 1], p_next, err)) return false;
        if (t >= 2) cudaStreamWaitEvent((cudaStream_t) stream_hop_, (cudaEvent_t) ev_h2d_[slot], 0);
        cudaStreamWaitEvent((cudaStream_t) stream_hop_, (cudaEvent_t) ev_a_pipe_, 0);
        cudaMemcpyAsync(hop_.data() + slot * hop_floats, side + slot * hop_floats,
                        hop_floats * sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t) stream_hop_);
        cudaEventRecord((cudaEvent_t) ev_h_pipe_, (cudaStream_t) stream_hop_);
        GM_CHECK(cudaEventSynchronize((cudaEvent_t) ev_h_pipe_), err);
        cudaSetDevice(split_next_->dev_);
        GM_CHECK(cudaMemcpyAsync(split_next_->state_, hop_.data() + slot * hop_floats,
                                 hop_floats * sizeof(float), cudaMemcpyHostToDevice, 0), err);
        GM_CHECK(cudaEventRecord((cudaEvent_t) ev_h2d_[slot], 0), err);
        if (!split_next_->step_layers(p_cur, err)) return false;
        cudaSetDevice(dev_);
        p_cur = p_next;
    }
    cudaSetDevice(split_next_->dev_);
    logits_out.resize((size_t) g_.n_vocab);
    GM_CHECK(cudaMemcpy(logits_out.data(), split_next_->sc_ + split_next_->sc_logits,
                        (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost), err);
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::step_layers(int64_t p, std::string& err) {
    const Glm5Geometry& g = g_;
    float* const S = state_;
    float* const A = sc_;
    float* R = S;   // the residual stack lives at the state arena's head
    strata::kernels::GlmHcShapes hcs;
    hcs.n_embd = g.n_embd;
    hcs.hc = g.hc;
    hcs.sinkhorn_iters = g.sinkhorn_iters;
    strata::kernels::GlmHcWorkspace hcws;
    strata::kernels::glm_hc_workspace_init(hcs, A + sc_inv, hcws);
    const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));

    for (int il = l0_; il < l1_; ++il) {
        if (timing_ && !tev_rec(tev_a_, "a")) timing_ = false;
        const std::string L = std::to_string(il);
        const auto W = [&](const char* suffix) { return w_.at("blk." + L + "." + suffix); };
        const auto ws_ = [&](const std::string& suffix) -> WSlot& { return ws_map_.at("blk." + L + "." + suffix); };

        // ---- hc read + norm
        strata::kernels::glm_hc_pre(R, W("hc_attn_fn.weight"), W("hc_attn_scale.weight"),
                                    W("hc_attn_base.weight"), g.norm_eps, g.hc_eps, hcs, hcws, A + sc_mixed,
                                    A + sc_pre, A + sc_post, A + sc_comb, nullptr);
        cb_dump("hc_pre-" + L, A + sc_pre, g.hc);
        cb_dump("hc_comb-" + L, A + sc_comb, g.hc * g.hc);
        cb_dump("hc_post-" + L, A + sc_post, g.hc);
        cb_dump("hc_attn_pre-" + L, A + sc_mixed, g.n_embd);
        rms_norm_kernel<<<1, GM_THREADS>>>(A + sc_mixed, W("attn_norm.weight"), A + sc_x, g.n_embd, 1,
                                           g.norm_eps);
        cb_dump("attn_norm-" + L, A + sc_x, g.n_embd);

        if (g.is_recr(il)) {
            // ---- the KDA mixer
            const char* streams[] = {"q", "k", "v"};
            for (int si = 0; si < 3; ++si) {
                const std::string attn = std::string("attn_") + streams[si] + ".weight";
                const std::string conv = std::string("ssm_conv1d_") + streams[si] + ".weight";
                GM_CHECK(gm_gemv(ws_(attn), A + sc_x, nullptr, 1.0f,
                                 A + sc_proj, g.d_inner(), g.n_embd), err);
                strata::kernels::glm_kda_conv(A + sc_proj, W(conv.c_str()),
                                              S + kda_conv_[(size_t) il] + (int64_t) si * g.d_inner() * (g.d_conv - 1),
                                              g.d_inner(), g.d_conv, 1, A + sc_conv[si], nullptr);
            }
            l2_heads_kernel<<<blocks(g.n_head), GM_THREADS>>>(A + sc_conv[0], A + sc_conv[0], g.kda_head_dim,
                                                              g.n_head, 1);
            l2_heads_kernel<<<blocks(g.n_head), GM_THREADS>>>(A + sc_conv[1], A + sc_conv[1], g.kda_head_dim,
                                                              g.n_head, 1);
            GM_CHECK(gm_gemv(ws_("ssm_f_a.weight"), A + sc_x, nullptr, 1.0f,
                                                                     A + sc_tmp, g.kda_head_dim, g.n_embd), err);
            GM_CHECK(gm_gemv(ws_("ssm_f_b.weight"), A + sc_tmp, nullptr, 1.0f,
                                                                 A + sc_g, g.d_inner(), g.kda_head_dim), err);
            axpy_kernel<<<blocks(g.d_inner()), GM_THREADS>>>(A + sc_g, 1.0f, W("ssm_dt.bias"), g.d_inner());
            strata::kernels::glm_kda_gate(A + sc_g, W("ssm_a"), g.kda_lb, g.kda_head_dim, g.n_head, 1,
                                          A + sc_g1, nullptr);
            GM_CHECK(gm_gemv(ws_("ssm_beta.weight"), A + sc_x, nullptr, 1.0f,
                                                               A + sc_beta, g.n_head, g.n_embd), err);
            sigmoid_kernel<<<blocks(g.n_head), GM_THREADS>>>(A + sc_beta, g.n_head);
            strata::kernels::glm_kda_recurrence(A + sc_conv[0], A + sc_conv[1], A + sc_conv[2], A + sc_g1,
                                                A + sc_beta, S + kda_S_[(size_t) il], g.n_head, g.kda_head_dim, 1,
                                                A + sc_scan, nullptr);
            GM_CHECK(gm_gemv(ws_("ssm_g_a.weight"), A + sc_x, nullptr, 1.0f,
                                                                     A + sc_tmp, g.kda_head_dim, g.n_embd), err);
            GM_CHECK(gm_gemv(ws_("ssm_g_b.weight"), A + sc_tmp, nullptr, 1.0f,
                                                                 A + sc_g2, g.d_inner(), g.kda_head_dim), err);
            strata::kernels::glm_kda_out_gate(A + sc_scan, W("ssm_norm.weight"), A + sc_g2, g.kda_head_dim,
                                              g.n_head, 1, g.norm_eps, A + sc_gated, nullptr);
            GM_CHECK(gm_gemv(ws_("attn_output.weight"), A + sc_gated, nullptr, 1.0f,
                                                               A + sc_mixer, g.n_embd, g.d_inner()), err);
        } else {
            // ---- the DSA mixer
            GM_CHECK(gm_gemv(ws_("attn_q_a.weight"), A + sc_x, nullptr, 1.0f,
                                                               A + sc_qr, g.q_lora, g.n_embd), err);
            rms_norm_kernel<<<1, GM_THREADS>>>(A + sc_qr, W("attn_q_a_norm.weight"), A + sc_qr, g.q_lora, 1,
                                               g.norm_eps);
            GM_CHECK(gm_gemv(ws_("attn_kv_a_mqa.weight"), A + sc_x, nullptr, 1.0f,
                                                                A + sc_kv, g.kv_lora, g.n_embd), err);
            rms_norm_kernel<<<1, GM_THREADS>>>(A + sc_kv, W("attn_kv_a_norm.weight"), A + sc_kv, g.kv_lora, 1,
                                               g.norm_eps);
            float* lat = S + dsa_lat_[(size_t) il];
            GM_CHECK(cudaMemcpy(lat + (int64_t) g.kv_lora * p, A + sc_kv, (size_t) g.kv_lora * sizeof(float),
                                cudaMemcpyDeviceToDevice), err);

            GM_CHECK(gm_gemv(ws_("attn_q_b.weight"), A + sc_qr, nullptr, 1.0f, A + sc_q, g.n_head * g.qk_nope, g.q_lora), err);
            q_absorb_kernel<<<blocks((int64_t) g.kv_lora * g.n_head), GM_THREADS>>>(
                W("attn_k_b.weight"), A + sc_q, A + sc_qabs, g.qk_nope, g.kv_lora, g.n_head, 1);

            // the indexer: per-token key/gate into their buffers; a pool's key when it completes
            GM_CHECK(gm_gemv(ws_("indexer.attn_k.weight"), A + sc_x, nullptr,
                                                                1.0f, A + sc_tmp, g.idx_key, g.n_embd), err);
            layer_norm_kernel<<<1, GM_THREADS>>>(A + sc_tmp, W("indexer.k_norm.weight"), W("indexer.k_norm.bias"),
                                                 A + sc_tmp, g.idx_key, 1, g.norm_eps);
            GM_CHECK(cudaMemcpy(S + dsa_ik_[(size_t) il] + (int64_t) g.idx_key * p, A + sc_tmp,
                                (size_t) g.idx_key * sizeof(float), cudaMemcpyDeviceToDevice), err);
            GM_CHECK(gm_gemv(ws_("indexer_compressor_gate.weight"), A + sc_x, nullptr,
                                                                1.0f, A + sc_tmp, g.idx_key, g.n_embd), err);
            GM_CHECK(cudaMemcpy(S + dsa_ig_[(size_t) il] + (int64_t) g.idx_key * p, A + sc_tmp,
                                (size_t) g.idx_key * sizeof(float), cudaMemcpyDeviceToDevice), err);
            const int64_t pool_done = (p + 1) / g.idx_kpool;   // pools whose last member is <= p
            if ((p + 1) % g.idx_kpool == 0) {
                const int64_t pi = pool_done - 1;
                strata::kernels::glm_dsa_pool(S + dsa_ik_[(size_t) il] + (int64_t) g.idx_key * pi * g.idx_kpool,
                                              S + dsa_ig_[(size_t) il] + (int64_t) g.idx_key * pi * g.idx_kpool,
                                              W("indexer_compressor_ape.weight"), g.idx_key, g.idx_kpool, 1,
                                              S + dsa_pool_[(size_t) il] + (int64_t) g.idx_key * pi, nullptr);
            }
            GM_CHECK(gm_gemv(ws_("indexer.attn_q_b.weight"), A + sc_qr, nullptr, 1.0f, A + sc_iq, g.idx_heads * g.idx_key,
                g.q_lora), err);
            GM_CHECK(gm_gemv(ws_("indexer.proj.weight"), A + sc_x, nullptr,
                                                                  prescale, A + sc_iw, g.idx_heads, g.n_embd), err);
            // the incomplete tail (select_tail) makes the CURRENT token visible before the first
            // pool completes - the oracle attends from t = 0 through the tail cells, so the
            // selection runs whenever pools exist OR the tail is on; only tail-off with no pools
            // has nothing to attend to
            if (pool_done > 0 || g.idx_select_tail) {
                if (pool_done > 0)
                    strata::kernels::glm_dsa_score(A + sc_iq, S + dsa_pool_[(size_t) il], A + sc_iw, g.idx_key,
                                                   g.idx_heads, 1, (int) pool_done, A + sc_score, nullptr);
                const int top_pools = (int) std::min<int64_t>(g.top_pools_max(), pool_done);
                const int n_sel = g.idx_kpool * top_pools + (g.idx_select_tail ? g.idx_kpool - 1 : 0);
                const int p32 = (int) p;
                GM_CHECK(cudaMemcpy(d_pos_, &p32, sizeof(int), cudaMemcpyHostToDevice), err);
                strata::kernels::glm_dsa_select(A + sc_score, (int) pool_done, g.idx_kpool, top_pools,
                                                g.idx_select_tail, 1, n_sel, d_pos_, (int*) (A + sc_cells),
                                                nullptr);
                strata::kernels::glm_mla_attention(A + sc_qabs, lat, (const int*) (A + sc_cells),
                                                   W("attn_v_b.weight"), g.kv_lora, g.n_head, g.v_head,
                                                   g.qk_nope, 1, n_sel, A + sc_attn, nullptr);
            } else {
                GM_CHECK(cudaMemset(A + sc_attn, 0, (size_t) g.v_head * g.n_head * sizeof(float)), err);
            }
            GM_CHECK(gm_gemv(ws_("attn_output.weight"), A + sc_attn, nullptr, 1.0f,
                             A + sc_mixer, g.n_embd, (int64_t) g.n_head * g.v_head), err);
        }

        // ---- hc write (ping-pong), then the FFN half
        strata::kernels::glm_hc_post(A + sc_mixer, R, A + sc_post, A + sc_comb, hcs, S + g.hc * g.n_embd, nullptr);
        R = S + g.hc * g.n_embd;
        cb_dump("hc_attn_post-" + L, R, g.hc * g.n_embd);
        strata::kernels::glm_hc_pre(R, W("hc_ffn_fn.weight"), W("hc_ffn_scale.weight"), W("hc_ffn_base.weight"),
                                    g.norm_eps, g.hc_eps, hcs, hcws, A + sc_mixed, A + sc_pre, A + sc_post,
                                    A + sc_comb, nullptr);
        cb_dump("hc_ffn_pre-" + L, A + sc_mixed, g.n_embd);
        rms_norm_kernel<<<1, GM_THREADS>>>(A + sc_mixed, W("ffn_norm.weight"), A + sc_x, g.n_embd, 1, g.norm_eps);
        cb_dump("ffn_norm-" + L, A + sc_x, g.n_embd);

        if (il < g.dense_lead) {
            GM_CHECK(gm_gemv(ws_("ffn_gate.weight"), A + sc_x, nullptr, 1.0f,
                                                                   A + sc_gate, g.n_ff_dense, g.n_embd), err);
            GM_CHECK(gm_gemv(ws_("ffn_up.weight"), A + sc_x, nullptr, 1.0f,
                                                                   A + sc_up, g.n_ff_dense, g.n_embd), err);
            strata::kernels::glm_swiglu_clamp(A + sc_gate, A + sc_up, g.swiglu_shexp, g.n_ff_dense, A + sc_h,
                                              nullptr);
            GM_CHECK(gm_gemv(ws_("ffn_down.weight"), A + sc_h, nullptr, 1.0f,
                                                               A + sc_ffn, g.n_embd, g.n_ff_dense), err);
        } else {
            GM_CHECK(gm_gemv(ws_("ffn_gate_inp.weight"), A + sc_x, nullptr, 1.0f,
                                                                 A + sc_tmp, g.n_expert, g.n_embd), err);
            cb_dump("glm_moe_logits-" + L, A + sc_tmp, g.n_expert);
            strata::kernels::router_sigmoid(A + sc_tmp, W("exp_probs_b.bias"), 1, g.n_expert, g.n_exp_used,
                                            g.w_scale, g.norm_w != 0, (int*) (A + sc_ids), A + sc_rw, nullptr);
            cb_dump("glm_moe_topk-" + L, A + sc_ids, g.n_exp_used);
            cb_dump("glm_moe_rw-" + L, A + sc_rw, g.n_exp_used);
            std::vector<int> ids((size_t) g.n_exp_used);
            std::vector<float> rw((size_t) g.n_exp_used);
            if (timing_ && !tev_rec(tev_b_, "b")) timing_ = false;   // dense/mixer/router done; the tail
            tail_rec_ = timing_;                                     // includes the D2H sync + the pool
            GM_CHECK(cudaMemcpy(ids.data(), A + sc_ids, ids.size() * sizeof(int), cudaMemcpyDeviceToHost), err);
            GM_CHECK(cudaMemcpy(rw.data(), A + sc_rw, rw.size() * sizeof(float), cudaMemcpyDeviceToHost), err);
            GM_CHECK(cudaMemset(A + sc_moe, 0, (size_t) g.n_embd * sizeof(float)), err);
            const int64_t slab = (int64_t) g.n_embd * g.n_ff_exp;
            if (pack_) {
                // the routed experts ride the native path (quantized blobs, per-token);
                // docs/GLM5-FLASH.md §9 - device pool with LRU (M2b), CPU native fallback
                if (dev_experts_) {
                    if (!pack_moe_tail_device(err, ids, rw, A, il)) return false;
                } else {
                    GM_CHECK(pack_moe_tail(err, ids, rw, A, il), err);
                }
            } else {
                for (int i = 0; i < g.n_exp_used; ++i) {
                    const int e = ids[(size_t) i];
                    gemv_bias_kernel<<<gm_blocks(g.n_ff_exp), GM_THREADS>>>(W("ffn_gate_exps.weight") + slab * e,
                                                                         A + sc_x, nullptr, 1.0f, A + sc_gate,
                                                                         g.n_ff_exp, g.n_embd, 1);
                    gemv_bias_kernel<<<gm_blocks(g.n_ff_exp), GM_THREADS>>>(W("ffn_up_exps.weight") + slab * e,
                                                                         A + sc_x, nullptr, 1.0f, A + sc_up,
                                                                         g.n_ff_exp, g.n_embd, 1);
                    strata::kernels::glm_swiglu_clamp(A + sc_gate, A + sc_up, g.swiglu_exp, g.n_ff_exp, A + sc_h,
                                                      nullptr);
                    gemv_bias_kernel<<<gm_blocks(g.n_embd), GM_THREADS>>>(W("ffn_down_exps.weight") + slab * e,
                                                                       A + sc_h, nullptr, 1.0f, A + sc_dn, g.n_embd,
                                                                       g.n_ff_exp, 1);
                    axpy_kernel<<<blocks(g.n_embd), GM_THREADS>>>(A + sc_moe, rw[(size_t) i], A + sc_dn, g.n_embd);
                }
            }
            if (timing_ && !tev_rec(tev_c_, "c")) timing_ = false;   // the expert tail is done
            GM_CHECK(gm_gemv(ws_("ffn_gate_shexp.weight"), A + sc_x, nullptr, 1.0f, A + sc_gate, g.n_ff_exp * g.n_shared,
                g.n_embd), err);
            GM_CHECK(gm_gemv(ws_("ffn_up_shexp.weight"), A + sc_x, nullptr, 1.0f, A + sc_up, g.n_ff_exp * g.n_shared, g.n_embd), err);
            strata::kernels::glm_swiglu_clamp(A + sc_gate, A + sc_up, g.swiglu_shexp, g.n_ff_exp * g.n_shared,
                                              A + sc_h, nullptr);
            GM_CHECK(gm_gemv(ws_("ffn_down_shexp.weight"), A + sc_h, nullptr,
                                                               1.0f, A + sc_sh, g.n_embd,
                                                               (int64_t) g.n_ff_exp * g.n_shared), err);
            add_kernel<<<blocks(g.n_embd), GM_THREADS>>>(A + sc_moe, A + sc_sh, A + sc_ffn, g.n_embd);
            cb_dump("ffn_moe_weighted-" + L, A + sc_moe, g.n_embd);
            cb_dump("ffn_shexp-" + L, A + sc_sh, g.n_embd);
        }
        cb_dump("ffn_out-" + L, A + sc_ffn, g.n_embd);

        strata::kernels::glm_hc_post(A + sc_ffn, R, A + sc_post, A + sc_comb, hcs, S, nullptr);
        R = S;
        cb_dump("l_out-" + L, R, g.hc * g.n_embd);
        if (timing_) {
            if (!tev_rec(tev_d_, "d")) timing_ = false;
            cudaEventSynchronize((cudaEvent_t) tev_d_);
            bool el_ok = true;
            const auto el = [&](void* p, void* q, double& acc) {
                float m2 = 0.0f;
                if (cudaEventElapsedTime(&m2, (cudaEvent_t) p, (cudaEvent_t) q) == cudaSuccess) acc += m2;
                else el_ok = false;
            };
            if (tail_rec_) {   // an MoE layer: b and c were recorded this layer
                el(tev_a_, tev_b_, t_dense_);
                el(tev_b_, tev_c_, t_tail_);
                el(tev_c_, tev_d_, t_rest_);
            } else {   // a dense lead layer: no tail to separate (b/c were never recorded here)
                el(tev_a_, tev_d_, t_dense_);
            }
            tail_rec_ = false;
            if (!el_ok) {
                std::fprintf(stderr, "glm timing: elapsed failed: %s\n", cudaGetErrorString(cudaGetLastError()));
                timing_ = false;   // never let a broken measurement disturb the engine
            }
        }
    }

    if (timing_) ++t_tokens_;
    if (l1_ == g.n_layers) {
        // ---- the head: mean over streams, norm, logits (the LAST half owns it)
        mean_streams_kernel<<<blocks(g.n_embd), GM_THREADS>>>(R, A + sc_head, g.n_embd, g.hc, 1);
        rms_norm_kernel<<<1, GM_THREADS>>>(A + sc_head, w_.at("output_norm.weight"), A + sc_head, g.n_embd, 1,
                                           g.norm_eps);
        GM_CHECK(gm_gemv(ws_map_.at("output.weight"), A + sc_head, nullptr, 1.0f,
                                     A + sc_logits, g.n_vocab, g.n_embd), err);
    }
    return true;
}

}  // namespace strata::core

// ================================ the pack (docs/GLM5-FLASH.md §9) ================================

// the routed half of one MoE layer, on the CPU: x (device) -> host, the selected experts'
// quantized blobs dotted by native_expert, weighted sum uploaded into A + sc_moe
cudaError_t strata::core::Glm5Model::pack_moe_tail(std::string& err, const std::vector<int>& ids, const std::vector<float>& rw,
                              float* A, int layer) {
    const auto& g = g_;
    const auto& nl = pack_layers_[(size_t) layer];
    if (nl.layer < 0) {
        err = "pack: layer " + std::to_string(layer) + " has no routed experts";
        return cudaErrorUnknown;
    }

    { cudaError_t e_ = cudaMemcpy(host_x_.data(), A + sc_x, (size_t) g.n_embd * sizeof(float),
                        cudaMemcpyDeviceToHost); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return cudaErrorUnknown; } }

    const void* act[1] = {host_act_.data()};
    float* ff1[1] = {host_ff_.data()};
    float* dn1[1] = {host_dn_.data()};
    const void* hq[1] = {host_hq_.data()};
    std::memset(host_moe_.data(), 0, (size_t) g.n_embd * sizeof(float));
    for (size_t i = 0; i < ids.size(); ++i) {
        const int e = ids[i];
        const Shard& gs = pack_shards_[(size_t) nl.gate_shard];
        const Shard& us = pack_shards_[(size_t) nl.up_shard];
        const Shard& ds = pack_shards_[(size_t) nl.down_shard];
        const uint8_t* gb = gs.base + nl.gate_off + (uint64_t) e * nl.fmt.gu_row * (uint64_t) g.n_ff_exp;
        const uint8_t* ub = us.base + nl.up_off + (uint64_t) e * nl.fmt.gu_row * (uint64_t) g.n_ff_exp;
        const uint8_t* db = ds.base + nl.down_off + (uint64_t) e * nl.fmt.d_row * (uint64_t) g.n_embd;
        strata::kernels::cpu::native_quant_act(nl.fmt, host_x_.data(), host_act_.data());
        strata::kernels::cpu::native_gu_rows_split(nl.fmt, gb, ub, act, 1, ff1, 0, (int) g.n_ff_exp, g.swiglu_exp);
        cb_dump_h("glm_exp" + std::to_string(e) + "-h-" + std::to_string(layer), host_ff_.data(), g.n_ff_exp);
        strata::kernels::cpu::native_quant_h(nl.fmt, host_ff_.data(), host_hq_.data());
        strata::kernels::cpu::native_down_rows_split(nl.fmt, db, hq, 1, dn1, 0, (int) g.n_embd);
        cb_dump_h("glm_exp" + std::to_string(e) + "-dn-" + std::to_string(layer), host_dn_.data(), g.n_embd);
        const float w = rw[i];
        for (int j = 0; j < g.n_embd; ++j) host_moe_[(size_t) j] += w * host_dn_[(size_t) j];
    }
    { cudaError_t e_ = cudaMemcpy(A + sc_moe, host_moe_.data(), (size_t) g.n_embd * sizeof(float),
                        cudaMemcpyHostToDevice); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return cudaErrorUnknown; } }
    return cudaSuccess;
}

// ---- M2b: the device expert pool -----------------------------------------------

void strata::core::Glm5Model::stage_worker() {
    for (;;) {
        StageJob job;
        {
            std::unique_lock<std::mutex> lk(stage_mu_);
            stage_cv_.wait(lk, [&] { return stage_quit_ || !stage_q_.empty(); });
            if (stage_q_.empty()) return;   // quit with nothing left
            job = stage_q_.back();
            stage_q_.pop_back();
        }
        uint8_t* dst = job.buf;
        for (int i = 0; i < 3; ++i) {
            if (job.len[i]) std::memcpy(dst, job.src[i], (size_t) job.len[i]);
            dst += job.len[i];
        }
        {
            std::lock_guard<std::mutex> lk(stage_mu_);
            if (--stage_pending_ == 0) stage_done_cv_.notify_all();
        }
    }
}

void strata::core::Glm5Model::stage_join() {
    std::unique_lock<std::mutex> lk(stage_mu_);
    stage_done_cv_.wait(lk, [&] { return stage_pending_ == 0; });
}

int32_t strata::core::Glm5Model::pool_evict(int cls) {
    // LFU with aging, not plain LRU: the routing traces show a heavy hitter set that recency
    // churns.  The least-used slot goes first, recency breaks ties; every 4096 evictions all
    // counts halve, so the pool tracks a phase change instead of fossilising the warmup.
    // A class evicts only within its own id range (its slots are size-homogeneous).
    const int64_t lo = class_base_[cls], hi = lo + class_slots_[cls];
    int32_t best = (int32_t) lo;
    uint32_t best_count = UINT32_MAX;
    uint64_t best_tick = ~0ull;
    for (int64_t s2 = lo; s2 < hi; ++s2) {
        const uint32_t c = pool_count_[(size_t) s2];
        if (c < best_count || (c == best_count && pool_tick_[(size_t) s2] < best_tick)) {
            best_count = c;
            best_tick = pool_tick_[(size_t) s2];
            best = (int32_t) s2;
        }
    }
    if (++pool_evictions_ % 4096 == 0)
        for (auto& c : pool_count_) c >>= 1;
    const int64_t key = pool_key_of_[(size_t) best];
    if (key >= 0) pool_slot_of_[(size_t) key] = -1;
    pool_key_of_[(size_t) best] = -1;
    return best;
}

void strata::core::Glm5Model::pool_ensure(int layer, const int* ids, int k) {
    const auto& nl = pack_layers_[(size_t) layer];
    const uint64_t gu_bytes = (uint64_t) nl.fmt.gu_row * (uint64_t) g_.n_ff_exp;   // one role
    const uint64_t dn_bytes = (uint64_t) nl.fmt.d_row * (uint64_t) g_.n_embd;
    const Shard& gs = pack_shards_[(size_t) nl.gate_shard];
    const Shard& us = pack_shards_[(size_t) nl.up_shard];
    const Shard& ds = pack_shards_[(size_t) nl.down_shard];
    static const bool stats = getenv("STRATA_GLM_POOL_STATS") != nullptr;
    if (bounce_)   // the staging ring is reused every layer: drain queued miss copies first
        cudaStreamSynchronize(0);
    // STRATA_GLM_TRACE=<file>: append "layer expert" per routed expert (the routing distribution
    // study - what a static/staged tier could hold)
    static std::FILE* trace = nullptr;
    static bool trace_init = false;
    if (!trace_init) {
        trace_init = true;
        if (const char* tp = getenv("STRATA_GLM_TRACE")) trace = std::fopen(tp, "w");
    }
    if (trace)
        for (int i = 0; i < k; ++i) std::fprintf(trace, "%d %d\n", layer, ids[i]);
    const int cls = layer_class_[(size_t) layer];
    const size_t cls_stride = class_stride_[cls];
    uint8_t* cls_arena = pool_ + class_off_[cls];
    for (int i = 0; i < k; ++i) {
        const int64_t key = (int64_t) layer * g_.n_expert + ids[i];
        int32_t slot = pool_slot_of_[(size_t) key];
        if (slot < 0) {
            slot = pool_evict(cls);
            pool_slot_of_[(size_t) key] = slot;
            pool_key_of_[(size_t) slot] = key;
            pool_count_[(size_t) slot] = 1;
            uint8_t* dst = cls_arena + (size_t) (slot - class_base_[cls]) * cls_stride;
            if (stage_on_) {
                // stage on a worker in parallel (the NVMe read is the cost; the main thread would
                // serialize it); the DMA is issued after stage_join() at the end of pool_ensure
                StageJob j;
                j.src[0] = gs.base + nl.gate_off + (uint64_t) ids[i] * gu_bytes;
                j.len[0] = gu_bytes;
                j.src[1] = us.base + nl.up_off + (uint64_t) ids[i] * gu_bytes;
                j.len[1] = gu_bytes;
                j.src[2] = ds.base + nl.down_off + (uint64_t) ids[i] * dn_bytes;
                j.len[2] = dn_bytes;
                j.buf = stage_bufs_[(size_t) i];
                {
                    std::lock_guard<std::mutex> lk(stage_mu_);
                    stage_q_.push_back(j);
                    ++stage_pending_;
                    stage_cv_.notify_one();
                }
                staged_dst_[i] = dst;
                staged_len_[i] = 2 * gu_bytes + dn_bytes;
            } else if (bounce_) {
                // pinned staging (STRATA_GLM_BOUNCE=1): CPU into the ring, then one stream-ordered
                // device copy per miss.  Left of the default path after two driver wedges on this
                // host; the ring drains via the sync at the head of every pool_ensure.
                uint8_t* b = bounce_ + (size_t) (bounce_i_++ % 24) * slot_bytes_;
                std::memcpy(b, gs.base + nl.gate_off + (uint64_t) ids[i] * gu_bytes, gu_bytes);
                std::memcpy(b + gu_bytes, us.base + nl.up_off + (uint64_t) ids[i] * gu_bytes, gu_bytes);
                std::memcpy(b + 2 * gu_bytes, ds.base + nl.down_off + (uint64_t) ids[i] * dn_bytes, dn_bytes);
                cudaMemcpyAsync(dst, b, slot_bytes_, cudaMemcpyHostToDevice);
            } else {
                // the verified phase-1 default: three stream-ordered copies out of the mmap'd
                // shards (the roles share a shard but an expert's gate/up rows are not adjacent)
                cudaMemcpyAsync(dst, gs.base + nl.gate_off + (uint64_t) ids[i] * gu_bytes, gu_bytes,
                                cudaMemcpyHostToDevice);
                cudaMemcpyAsync(dst + gu_bytes, us.base + nl.up_off + (uint64_t) ids[i] * gu_bytes, gu_bytes,
                                cudaMemcpyHostToDevice);
                cudaMemcpyAsync(dst + 2 * gu_bytes, ds.base + nl.down_off + (uint64_t) ids[i] * dn_bytes, dn_bytes,
                                cudaMemcpyHostToDevice);
            }
            ++pool_misses_;
        } else {
            ++pool_hits_;
            if (pool_count_[(size_t) slot] < (1u << 20)) ++pool_count_[(size_t) slot];
        }
        pool_tick_[(size_t) slot] = ++pool_clock_;
        plan_host_->ptr[i] = (unsigned long long) (cls_arena + (size_t) (slot - class_base_[cls]) * cls_stride);
        plan_host_->start[i] = i;
        plan_host_->dst[i] = i;
        plan_host_->tok[i] = 0;
    }
    plan_host_->start[k] = k;
    plan_host_->n_groups[0] = k;
    if (stage_on_) {
        // the workers have staged this layer's misses; now one pinned DMA per miss (stream-ordered,
        // so the grouped kernel below reads complete slots)
        stage_join();
        for (int i = 0; i < k; ++i)
            if (staged_len_[(size_t) i]) {
                cudaMemcpyAsync(staged_dst_[(size_t) i], stage_bufs_[(size_t) i], (size_t) staged_len_[(size_t) i],
                                cudaMemcpyHostToDevice);
                staged_len_[(size_t) i] = 0;
            }
    }
    if (stats && (pool_hits_ + pool_misses_) % 420 == 0) {
        std::fprintf(stderr, "pool: hits %llu misses %llu (%.1f%%)\n", (unsigned long long) pool_hits_,
                     (unsigned long long) pool_misses_,
                     100.0 * (double) pool_hits_ / (double) (pool_hits_ + pool_misses_));
    }
}

bool strata::core::Glm5Model::pack_moe_tail_device(std::string& err, const std::vector<int>& ids,
                                                   const std::vector<float>& rw, float* A, int layer) {
    const auto& g = g_;
    const auto& nl = pack_layers_[(size_t) layer];
    const auto tp0 = std::chrono::steady_clock::now();
    pool_ensure(layer, ids.data(), (int) ids.size());
    if (timing_)
        t_pool_host_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
#ifndef _WIN32
    // prefetch the router's NEXT-BEST experts: ranks 8..8+R are what the next token likely routes,
    // so posix_fadvise(WILLNEED) streams their shard pages into the OS cache during this token -
    // fire-and-forget, no buffers, and the demand read of token t+1 then faults from RAM instead
    // of the NVMe (STRATA_GLM_PREFETCH=0 disables, STRATA_GLM_PREFETCH_RANKS=N bounds it)
    {
        // STRATA_GLM_PREFETCH=1 opts IN: measured ~6% SLOWER on Mercury (the demand stream already
        // saturates the 2.2 GB/s drive; extra readahead steals from it), kept for machines with a
        // faster SSD where the idle bandwidth exists
        static const bool pf_on = getenv("STRATA_GLM_PREFETCH") != nullptr &&
                                  std::atoi(getenv("STRATA_GLM_PREFETCH")) != 0;
        static const int pf_ranks = getenv("STRATA_GLM_PREFETCH_RANKS") != nullptr
                                        ? std::atoi(getenv("STRATA_GLM_PREFETCH_RANKS"))
                                        : 3;
        if (pf_on && pf_ranks > 0 && !host_probs_b_.empty() && (int) host_probs_b_.size() >= (layer + 1) * g.n_expert) {
            static std::vector<float> lg_host;
            static std::vector<int> order;
            lg_host.resize((size_t) g.n_expert);
            order.resize((size_t) g.n_expert);
            cudaMemcpy(lg_host.data(), A + sc_tmp, (size_t) g.n_expert * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (int e2 = 0; e2 < (int) g.n_expert; ++e2) order[(size_t) e2] = e2;
            const float* bias = host_probs_b_.data() + (size_t) layer * g.n_expert;
            const int top = std::min((int) g.n_expert, (int) g.n_exp_used + pf_ranks);
            std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int a2, int b2) {
                const float va = 1.0f / (1.0f + std::exp(-lg_host[(size_t) a2])) + bias[a2];
                const float vb = 1.0f / (1.0f + std::exp(-lg_host[(size_t) b2])) + bias[b2];
                return va > vb;
            });
            const uint64_t gu_b = (uint64_t) nl.fmt.gu_row * (uint64_t) g.n_ff_exp;
            const uint64_t dn_b = (uint64_t) nl.fmt.d_row * (uint64_t) g.n_embd;
            const Shard& s1 = pack_shards_[(size_t) nl.gate_shard];
            const Shard& s2 = pack_shards_[(size_t) nl.up_shard];
            const Shard& s3 = pack_shards_[(size_t) nl.down_shard];
            for (int r2 = (int) g.n_exp_used; r2 < top; ++r2) {
                const int e2 = order[(size_t) r2];
                if (pool_slot_of_[(size_t) ((int64_t) layer * g.n_expert + e2)] >= 0) continue;  // resident
                if (s1.fd >= 0)
                    posix_fadvise(s1.fd, (off_t) (nl.gate_off + (uint64_t) e2 * gu_b), (off_t) gu_b,
                                  POSIX_FADV_WILLNEED);
                if (s2.fd >= 0)
                    posix_fadvise(s2.fd, (off_t) (nl.up_off + (uint64_t) e2 * gu_b), (off_t) gu_b,
                                  POSIX_FADV_WILLNEED);
                if (s3.fd >= 0)
                    posix_fadvise(s3.fd, (off_t) (nl.down_off + (uint64_t) e2 * dn_b), (off_t) dn_b,
                                  POSIX_FADV_WILLNEED);
            }
        }
    }
#endif
    strata::kernels::quantize_q8_1_rows(A + sc_x, 1, g.n_embd, dev_xq_, nullptr);
    const strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout(
        nl.fmt.gu_type, nl.fmt.d_type, g.n_embd, g.n_ff_exp);
    strata::kernels::native_expert_grouped(L, (const unsigned long long*) (plan_dev_ + offsetof(DevPlan, ptr)),
                                           (const int32_t*) (plan_dev_ + offsetof(DevPlan, start)),
                                           (const int32_t*) (plan_dev_ + offsetof(DevPlan, n_groups)),
                                           (const int32_t*) (plan_dev_ + offsetof(DevPlan, dst)),
                                           (const int32_t*) (plan_dev_ + offsetof(DevPlan, tok)),
                                           (int64_t) ids.size(), (int64_t) ids.size(), dev_xq_, dev_scratch_,
                                           dev_hit_out_, nullptr, g.swiglu_exp);
    { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return false; } }
    GM_CHECK(cudaMemset(A + sc_moe, 0, (size_t) g.n_embd * sizeof(float)), err);
    for (int i = 0; i < (int) ids.size(); ++i)
        axpy_kernel<<<blocks(g.n_embd), GM_THREADS>>>(A + sc_moe, rw[(size_t) i],
                                                      dev_hit_out_ + (size_t) i * g.n_embd, g.n_embd);
    { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return false; } }
    return true;
}

// ================================ the pack (docs/GLM5-FLASH.md §9) ================================

static bool pack_shard_mmap(const std::string& path, strata::core::Glm5Model::Shard& s, uint64_t data_start,
                            std::string& err) {
#ifdef _WIN32
    // the same as below with Windows' calls: a read-only view of the whole shard, and a second, unbuffered handle
    // (FILE_FLAG_NO_BUFFERING, Windows' O_DIRECT) for the fast path's expert reads.  That one is OVERLAPPED: Windows
    // runs the reads of a synchronous handle one at a time, and the fast path reads from many threads at once
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        err = "pack: open failed for " + path;
        return false;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(h, &sz)) {
        CloseHandle(h);
        err = "pack: the size of " + path + " is unknown";
        return false;
    }
    s.size = (uint64_t) sz.QuadPart;
    HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    void* v = m ? MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (m) CloseHandle(m);   // the view keeps the mapping
    CloseHandle(h);
    if (v == nullptr) {
        err = "pack: mapping failed for " + path;
        return false;
    }
    s.base = (uint8_t*) v;
    s.path = path;
    s.data_start = data_start;
    HANDLE hd = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    s.h_direct = hd == INVALID_HANDLE_VALUE ? nullptr : (void*) hd;
    return true;
#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "pack: open failed for " + path;
        return false;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        close(fd);
        err = "pack: fstat failed for " + path;
        return false;
    }
    s.size = (uint64_t) st.st_size;
    void* m = mmap(nullptr, s.size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) {
        close(fd);
        err = "pack: mmap failed for " + path;
        return false;
    }
    s.base = (uint8_t*) m;
    s.path = path;
    s.data_start = data_start;
    s.fd = fd;   // kept open: the prefetch fadvises its ranges (the mapping itself does not need it)
    // the fast path reads experts with O_DIRECT (no page cache: on a host whose RAM is mostly the pinned
    // expert tier, buffered reads pushed the system into swap); absent where the filesystem refuses it
    s.fd_direct = open(path.c_str(), O_RDONLY | O_DIRECT);
    return true;
#endif
}

// Windows: while a file has a mapped view, every unbuffered read of it takes NTFS's cache-coherency path - the
// decode's expert reads ran 3x as long (an expert in 24 parallel reads: 5.4 ms with the shards mapped, 1.8 ms
// without; RTX 5090 Laptop, Intel RST RAID 0 of two NVMe drives, Maya-S24).  Once the load is done the fast path
// reads the shards through h_direct only, and token_embd through pack_emb_src_: the table moves to the heap
// (~0.7 GB for GLM-5.3-Flash's Q8_0 rows) and the views go.  Kept when a shard has no unbuffered handle (its reads
// fall back to the view) or with STRATA_GLM_KEEP_MAP=1.
void strata::core::Glm5Model::pack_release_views() {
#ifdef _WIN32
    const char* km = getenv("STRATA_GLM_KEEP_MAP");
    if (km != nullptr && std::atoi(km) != 0) return;
    for (const Shard& s : pack_shards_)
        if (s.base != nullptr && s.h_direct == nullptr) return;
    for (const Shard& s : pack_shards_)
        if (pack_emb_src_ != nullptr && pack_emb_src_ >= s.base && pack_emb_src_ < s.base + s.size) {
            const size_t bytes = (size_t) g_.n_vocab * ggml_row_size((ggml_type) pack_emb_type_, g_.n_embd);
            pack_emb_copy_.assign(pack_emb_src_, pack_emb_src_ + bytes);
            pack_emb_src_ = pack_emb_copy_.data();
            break;
        }
    for (Shard& s : pack_shards_)
        if (s.base != nullptr) {
            UnmapViewOfFile(s.base);
            s.base = nullptr;
        }
#endif
}

bool strata::core::Glm5Model::load_pack(const std::string& pack_dir, int64_t max_ctx, std::string& err, int dev,
                                        int l0, int l1) {
    dev_ = dev;
    pack_dir_ = pack_dir;
    cudaSetDevice(dev_);
    // 1. the shard names come from native_experts.txt's header; the shards sit NEXT TO the pack
    std::vector<std::string> shard_names;
    if (!pack_shard_names(pack_dir, shard_names, err)) return false;
    const std::string dir2 = pack_dir + "/..";   // the shards live beside the pack directory
    strata::GgufFile gguf(dir2 + "/" + shard_names[0]);
    if (!Glm5Geometry::from_gguf(gguf, g_, err)) return false;
    max_ctx_ = max_ctx;
    l0_ = l0;
    l1_ = l1 > 0 ? l1 : g_.n_layers;
    if (l0_ < 0 || l1_ > g_.n_layers || l1_ <= l0_) {
        err = "pack: bad layer range [" + std::to_string(l0_) + ", " + std::to_string(l1_) + ")";
        return false;
    }
    // the layer split: this instance carries only ITS layers' rows (token_embd lives on the first
    // half, the head's tensors on the last); every per-layer array stays indexed by the ABSOLUTE
    // layer id, so all offsets and kernels are unchanged
    const auto row_in_range = [&](const std::string& n) {
        if (n.rfind("blk.", 0) == 0) {
            const int il = std::atoi(n.c_str() + 4);
            return il >= l0_ && il < l1_;
        }
        if (n == "token_embd.weight") return l0_ == 0;
        if (n == "output.weight" || n == "output_norm.weight") return l1_ == g_.n_layers;
        return true;
    };

    // 2. mmap the shards AND keep their GGUF directories alive: unsloth's shard 1 carries only
    //    metadata - the tensor directories live in shards 2/3, so lookups must scan them all
    std::vector<std::unique_ptr<strata::GgufFile>> gfs;
    for (const auto& n : shard_names) {
        gfs.emplace_back(new strata::GgufFile(dir2 + "/" + n));
        Shard s;
        if (!pack_shard_mmap(dir2 + "/" + n, s, gfs.back()->data_start(), err)) return false;
        pack_shards_.push_back(s);
    }
    const auto find_tensor = [&](const std::string& name) -> const strata::TensorInfo* {
        for (const auto& gf : gfs) {
            const strata::TensorInfo* t = gf->find(name);
            if (t) return t;
        }
        return nullptr;
    };
    const auto find_shard_of = [&](const std::string& name) -> int {
        for (size_t i = 0; i < gfs.size(); ++i)
            if (gfs[i]->find(name)) return (int) i;
        return -1;
    };
    const auto shard_by_name = [&](const std::string& n) -> int {
        for (size_t i = 0; i < shard_names.size(); ++i)
            if (shard_names[i] == n) return (int) i;
        return -1;
    };

    // 3. native_experts.txt: per MoE layer, the expert types and blob offsets
    pack_layers_.assign((size_t) g_.n_layers, NativeLayer{});
    {
        std::ifstream ne(pack_dir + "/native_experts.txt");
        std::string line;
        while (std::getline(ne, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int layer = -1, gu = -1, dty = -1;
            uint64_t off = 0, blob = 0, go = 0, uo = 0, dof = 0;
            ss >> layer >> gu >> dty >> off >> blob >> go >> uo >> dof;
            std::vector<std::string> names;
            std::string w;
            while (ss >> w) names.push_back(w);
            if (layer < 0 || layer >= g_.n_layers) continue;
            NativeLayer nl;
            nl.layer = layer;
            if (!strata::kernels::cpu::native_fmt(gu, dty, g_.n_embd, g_.n_ff_exp, nl.fmt, err)) return false;
            nl.gate_off = go;
            nl.up_off = uo;
            nl.down_off = dof;
            if (names.empty()) {   // v3: a row without a shard lies in shard 1 (the --gguf file)
                nl.gate_shard = nl.up_shard = nl.down_shard = 0;
            } else if (names.size() == 1) {
                const int si = shard_by_name(names[0]);
                if (si < 0) {
                    err = "pack: unknown shard " + names[0];
                    return false;
                }
                nl.gate_shard = nl.up_shard = nl.down_shard = si;
            } else if (names.size() == 3) {   // v4 straddle row
                nl.gate_shard = names[0].empty() ? 0 : shard_by_name(names[0]);
                nl.up_shard = names[1].empty() ? 0 : shard_by_name(names[1]);
                nl.down_shard = names[2].empty() ? 0 : shard_by_name(names[2]);
                if (nl.gate_shard < 0 || nl.up_shard < 0 || nl.down_shard < 0) {
                    err = "pack: unknown shard name in native_experts row " + std::to_string(layer);
                    return false;
                }
            } else {
                err = "pack: native_experts row " + std::to_string(layer) + " has " +
                      std::to_string(names.size()) + " shard fields";
                return false;
            }
            // fail-closed: every row offset must match the shard directories exactly.  (The
            // v4 packer once computed down_off with the gate role's data_start; the 7616-byte
            // drift dequanted layer 25's experts into NaN - silent at load, fatal at run.)
            const char* role_tn[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
            const int role_sh[3] = {nl.gate_shard, nl.up_shard, nl.down_shard};
            const uint64_t role_of[3] = {nl.gate_off, nl.up_off, nl.down_off};
            const int role_ty[3] = {gu, gu, dty};   // gate and up share one type; down is its own
            for (int ri = 0; ri < 3; ++ri) {
                const std::string rn = "blk." + std::to_string(layer) + "." + role_tn[ri];
                const strata::TensorInfo* ti = gfs[(size_t) role_sh[ri]]->find(rn);
                if (ti == nullptr || (int) ti->type != role_ty[ri] ||
                    (uint64_t) gfs[(size_t) role_sh[ri]]->data_start() + ti->offset != role_of[ri]) {
                    err = "pack: native_experts row " + std::to_string(layer) +
                          " disagrees with the shard directory (" + rn + "; row " +
                          std::to_string(role_of[ri]) + " type " + std::to_string(role_ty[ri]) + " vs " +
                          (ti ? "dir " + std::to_string((uint64_t) gfs[(size_t) role_sh[ri]]->data_start() +
                                                          ti->offset) + " type " + std::to_string((int) ti->type)
                              : "NOT FOUND") + ")";
                    return false;
                }
            }
            pack_layers_[(size_t) layer] = std::move(nl);
        }
    }

    // 4. the dense tensors: dequantize everything into the F32 device arena.  F32 rows come
    //    straight out of dense.bin; every other kind (BF16 rows and the natively-served
    //    quantized rows) is resolved through the GGUF shards and dequantized by ggml's traits.
    struct Row {
        std::string name, kind;
        uint64_t offset = 0, bytes = 0;
    };
    std::vector<Row> rows;
    {
        std::ifstream ix(pack_dir + "/index.txt");
        std::string line;
        while (std::getline(ix, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            Row r;
            std::string f;
            ss >> r.name >> f >> r.kind >> r.offset >> r.bytes;   // name served kind offset bytes ...
            row_kind_[r.name] = r.kind;
            rows.push_back(r);
        }
    }
    // dequant-at-load ONLY for the small tensors; everything the device MMVQ serves stays
    // quantized in VRAM (glm5-next's dense side is ~30B params - 129.8 GiB in F32, never fits)
    // token_embd stays QUANTIZED too (Q4_K, 0.36 GB instead of 2.54 GB F32): step() dequantizes
    // the one row it needs per token from the host mapping (pack_emb_*), and the 2.2 GB go to
    // the expert pool instead
    const auto dequant_at_load = [&](const std::string& name) {
        return row_kind_[name] != "0" || name == "attn_k_b.weight" || name == "attn_v_b.weight";
    };
    // the fast path keeps the pack's BIG BF16 rows as BF16 on the device (dense.bin stores them BF16,
    // so this is lossless; F32 expansion doubled their bytes and their read time) and does not upload
    // token_embd at all (the embedding row is dequantized on the host from the shard mapping)
    fast_mode_ = getenv("STRATA_GLM_SLOW") == nullptr;
    const auto keep16 = [&](const Row& r, const strata::TensorInfo* t) {
        return fast_mode_ && r.kind == "4" && t->elements() >= 65536;
    };
    const auto skip_upload = [&](const Row& r) { return fast_mode_ && r.name == "token_embd.weight"; };
    uint64_t bytes = 0;
    std::map<std::string, uint64_t> by_kind;   // STRATA_GLM_TIMING: the dense weights' VRAM by kind and storage
    const auto kind_of = [](const std::string& n) -> std::string {
        if (n == "output.weight") return "output head";
        if (n.find("_shexp") != std::string::npos) return "shared experts";
        if (n.find("ffn_gate_inp") != std::string::npos) return "routers";
        if (n.find("attn_k_b") != std::string::npos || n.find("attn_v_b") != std::string::npos) return "mla k_b/v_b";
        if (n.find("attn_q_a") != std::string::npos || n.find("attn_q_b") != std::string::npos ||
            n.find("attn_kv_a") != std::string::npos) return "mla q/kv";
        if (n.find("indexer") != std::string::npos || n.find("idx") != std::string::npos) return "dsa indexer";
        if (n.find("attn_q.") != std::string::npos || n.find("attn_k.") != std::string::npos ||
            n.find("attn_v.") != std::string::npos) return "kda q/k/v";
        if (n.find("attn_output") != std::string::npos) return "attn output";
        if (n.find("ssm_") != std::string::npos) return "kda small";
        if (n.find("hc_") != std::string::npos) return "mhc";
        if (n.find("ffn_") != std::string::npos) return "dense ffn";
        if (n.find("norm") != std::string::npos) return "norms";
        return "other";
    };
    for (const auto& r : rows) {
        if (!row_in_range(r.name)) continue;   // the other half's rows live on the other device
        const strata::TensorInfo* t = find_tensor(r.name);
        if (!t) {
            err = "pack: tensor " + r.name + " is in index.txt but in no shard";
            return false;
        }
        if (skip_upload(r)) continue;
        std::string how;
        const uint64_t b = pack_row_vram(r.name, r.kind, t, fast_mode_, &how);
        bytes += b;
        by_kind[kind_of(r.name.substr(r.name.find('.', 4) + 1)) + how] += b;
    }
    if (getenv("STRATA_GLM_TIMING") != nullptr) {
        std::vector<std::pair<uint64_t, std::string>> v;
        for (auto& kv : by_kind) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        std::string s;
        for (auto& e : v) s += " | " + e.second + " " + std::to_string((int) (e.first >> 20)) + " MB";
        std::fprintf(stderr, "glm pack: CUDA%d dense weights %.2f GB:%s\n", dev_, (double) bytes / 1073741824.0, s.c_str());
    }
    { cudaError_t e_ = cudaMalloc(&w_arena_, bytes); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return false; } }
    w_arena_bytes_ = bytes;
    std::ifstream dense(pack_dir + "/dense.bin", std::ios::binary);
    if (!dense) {
        err = "pack: cannot open dense.bin";
        return false;
    }
    uint64_t at = 0;
    std::vector<float> host;
    std::vector<uint8_t> raw;
    for (const auto& r : rows) {
        if (!row_in_range(r.name)) continue;   // the other half's rows live on the other device
        const strata::TensorInfo* t = find_tensor(r.name);
        if (skip_upload(r)) {
            const int si = find_shard_of(r.name);
            const strata::TensorInfo* ti = si >= 0 ? gfs[(size_t) si]->find(r.name) : nullptr;
            if (ti == nullptr) {
                err = "pack: token_embd not found in any shard";
                return false;
            }
            pack_emb_src_ = pack_shards_[(size_t) si].base + gfs[(size_t) si]->data_start() + ti->offset;
            pack_emb_type_ = (int) ti->type;
            continue;
        }
        if (keep16(r, t)) {
            if (r.bytes != (uint64_t) t->elements() * 2) {
                err = "pack: BF16 row " + r.name + " has " + std::to_string(r.bytes) + " bytes";
                return false;
            }
            raw.resize((size_t) r.bytes);
            dense.seekg((long long) r.offset);
            dense.read((char*) raw.data(), (long long) r.bytes);
            if (!dense) {
                err = "pack: short read on dense.bin for " + r.name;
                return false;
            }
            { cudaError_t e_ = cudaMemcpy((uint8_t*) w_arena_ + at, raw.data(), (size_t) r.bytes, cudaMemcpyHostToDevice); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return false; } }
            w16_[r.name] = (const uint16_t*) ((uint8_t*) w_arena_ + at);
            at += (r.bytes + 255u) & ~(uint64_t) 255u;
            continue;
        }
        if (!dequant_at_load(r.name)) {
            // quantized rows: upload the raw GGUF bytes and serve them by device MMVQ
            const int si = find_shard_of(r.name);
            if (si < 0) {
                err = "pack: native tensor " + r.name + " not found in any shard";
                return false;
            }
            const strata::TensorInfo* ti = gfs[(size_t) si]->find(r.name);
            const size_t nbytes = strata::kernels::native_mmvq_weight_bytes(ti->type, (int) ti->shape[0],
                                                                            (int) ti->shape[1]);
            const uint8_t* src = pack_shards_[(size_t) si].base + gfs[(size_t) si]->data_start() + ti->offset;
            { cudaError_t e_ = cudaMemcpy((uint8_t*) w_arena_ + at, src, nbytes, cudaMemcpyHostToDevice); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return false; } }
            ws_map_[r.name] = strata::core::WSlot{nullptr, (int) ti->type, (const void*) ((uint8_t*) w_arena_ + at),
                                    (int64_t) ti->shape[0], (int64_t) ti->shape[1]};
            if (r.name == "token_embd.weight") {
                pack_emb_src_ = src;
                pack_emb_type_ = (int) ti->type;
            }
            at += (nbytes + 255u) & ~(uint64_t) 255u;
            continue;
        }
        const uint64_t n_f32 = (uint64_t) t->elements();
        host.resize((size_t) n_f32);
        if (r.kind == "0") {
            // natively served but dequantized at load (the embedding and the 3-D absorbed
            // projections, which the runner indexes by row)
            const int si = find_shard_of(r.name);
            if (si < 0) {
                err = "pack: native tensor " + r.name + " not found in any shard";
                return false;
            }
            const strata::TensorInfo* ti = gfs[(size_t) si]->find(r.name);
            if (!ti) {
                err = "pack: tensor " + r.name + " missing from shard " + std::to_string(si);
                return false;
            }
            if (ti->elements() != t->elements()) {
                err = "pack: element count mismatch for " + r.name + " (dir " +
                      std::to_string(t->elements()) + " vs shard " + std::to_string(ti->elements()) + ")";
                return false;
            }
            if (!pack_dequant_tensor(*ti, pack_shards_[(size_t) si].base + gfs[(size_t) si]->data_start() +
                                               ti->offset, host, err)) {
                return false;
            }
        } else if (r.kind == "4") {
            // BF16 rows in dense.bin: expand in place (bf16 bits << 16 = f32 bits)
            raw.resize((size_t) r.bytes);
            dense.seekg((long long) r.offset);
            dense.read((char*) raw.data(), (long long) r.bytes);
            if (!dense) {
                err = "pack: short read on dense.bin for " + r.name;
                return false;
            }
            const uint16_t* src = (const uint16_t*) raw.data();
            uint32_t* dst = (uint32_t*) host.data();
            for (size_t i = 0; i < (size_t) n_f32; ++i) dst[i] = (uint32_t) src[i] << 16;
        } else if (r.kind == "2") {
            dense.seekg((long long) r.offset);
            dense.read((char*) host.data(), (long long) r.bytes);
            if (!dense) {
                err = "pack: short read on dense.bin for " + r.name;
                return false;
            }
        } else {
            err = "pack: unsupported dense kind '" + r.kind + "' for " + r.name;
            return false;
        }
        { cudaError_t e_ = cudaMemcpy((uint8_t*) w_arena_ + at, host.data(), (size_t) n_f32 * 4,
                            cudaMemcpyHostToDevice); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return false; } }
        w_[r.name] = (const float*) ((uint8_t*) w_arena_ + at);
        ws_map_[r.name] = strata::core::WSlot{(const float*) ((uint8_t*) w_arena_ + at), 0, nullptr, 0, 0};
        at += (n_f32 * 4 + 255u) & ~(uint64_t) 255u;
    }

    // 4b. the NextN (MTP) block on the LAST half of the fast path: its own weights from the shards (the pack never
    //     carried blk.<n_layers>.*), its experts through the tiers like any MoE layer
    lt_ = l1_;
    // (the MTP decode drafts with it on any device count - src/core/glm_mtp.cu; without that decode only the earlier
    // pipelined one of a split reads it, and one device loads it with STRATA_GLM_MTP=1 only)
    // STRATA_GLM_MTP_GGUF=<a GGUF holding blk.<n_layers>.* of the same model>: the draft block from that file - for a
    // quant published without one (the block reads the trunk's hidden state and the embedding, so any quant's block
    // fits), or a more precise block than the model's own: load_mtp looks there first.
    bool mtp_extra = false;
    const char* mtp_gguf = getenv("STRATA_GLM_MTP_GGUF");
    if (fast_mode_ && l1_ == g_.n_layers && mtp_gguf != nullptr && mtp_gguf[0] != '\0') {
        const bool own = find_tensor("blk." + std::to_string(g_.n_layers) + ".nextn.eh_proj.weight") != nullptr;
        auto extra = std::make_unique<strata::GgufFile>(mtp_gguf);
        if (extra->find("blk." + std::to_string(g_.n_layers) + ".nextn.eh_proj.weight") == nullptr) {
            err = std::string("STRATA_GLM_MTP_GGUF: ") + mtp_gguf + " has no blk." + std::to_string(g_.n_layers) +
                  " draft block";
            return false;
        }
        Shard s;
        if (!pack_shard_mmap(mtp_gguf, s, extra->data_start(), err)) return false;
        mtp_src_ = (int) gfs.size();
        gfs.push_back(std::move(extra));   // after the model's own shards: only the draft block is looked up in it
        pack_shards_.push_back(s);
        mtp_extra = true;
        std::fprintf(stderr, "glm mtp: CUDA%d the draft block from %s%s\n", dev_, mtp_gguf,
                     own ? " (instead of the model's own)" : "");
    }
    if (fast_mode_ && l1_ == g_.n_layers && (g_.nextn > 0 || mtp_extra) && getenv("STRATA_GLM_NO_MTP") == nullptr &&
        (l0_ > 0 || getenv("STRATA_GLM_MTP") != nullptr || glm_mtp_decode_wanted())) {
        if (!load_mtp(gfs, err)) return false;
        if (mtp_il_ >= 0) lt_ = l1_ + 1;
    }

    // 5. the router bias, host-side: the prefetch ranks the next-best experts per layer, and the
    //    ranking needs sigmoid(logits) + this per-layer bias (device-side it stays as it was)
    host_probs_b_.assign((size_t) g_.n_layers * g_.n_expert, 0.0f);
    for (int il = l0_; il < l1_; ++il) {
        const auto it = w_.find("blk." + std::to_string(il) + ".exp_probs_b.bias");
        if (it != w_.end())
            cudaMemcpy(host_probs_b_.data() + (size_t) il * g_.n_expert, it->second,
                       (size_t) g_.n_expert * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // 6. the state and scratch arenas (identical to load()), then the CPU MoE buffers
    const int64_t hc_dim = (int64_t) g_.hc * g_.n_embd;
    const int64_t max_pools = max_ctx / g_.idx_kpool;
    int64_t floats = 2 * hc_dim;
    // the fast path keeps the DSA latent cache in FP16 (half the bytes: at 128k context ~0.8 GB more for experts on
    // each card, and the prompt attention reads half as much); the reference path (STRATA_GLM_SLOW) keeps F32
    lat_q8_ = fast_mode_ && g_.kv_lora % 32 == 0 && getenv("STRATA_GLM_KV_INT8") != nullptr &&
              std::atoi(getenv("STRATA_GLM_KV_INT8")) != 0;
    // (INT8: lat8_rec_bytes per position, a multiple of 16 bytes at kv_lora 512)
    const int64_t lat_floats = lat_q8_      ? (int64_t) strata::kernels::glmf::lat8_rec_bytes(g_.kv_lora) / 4 * max_ctx
                               : fast_mode_ ? (int64_t) g_.kv_lora * max_ctx / 2
                                            : (int64_t) g_.kv_lora * max_ctx;
    if (lat_q8_) std::fprintf(stderr, "glm pack: CUDA%d latent cache INT8 (%d bytes a position and layer)\n", dev_,
                              strata::kernels::glmf::lat8_rec_bytes(g_.kv_lora));
    // (one entry past the trunk: the NextN block's DSA caches, when this half carries it)
    kda_S_.assign((size_t) g_.n_layers + 1, 0);
    kda_conv_.assign((size_t) g_.n_layers + 1, 0);
    dsa_lat_.assign((size_t) g_.n_layers + 1, 0);
    dsa_ik_.assign((size_t) g_.n_layers + 1, 0);
    dsa_ig_.assign((size_t) g_.n_layers + 1, 0);
    dsa_pool_.assign((size_t) g_.n_layers + 1, 0);
    // the indexer key / gate caches: a ring on the fast path (the prompt path's largest sub-batch + 64 positions; see
    // ik_ring_), the whole context on the reference path (whose kernels index them by position)
    ik_ring_ = fast_mode_ ? (int) std::min<int64_t>(max_ctx, 8192 + 64) : (int) max_ctx;
    if (mtp_il_ >= 0) {
        dsa_lat_[(size_t) mtp_il_] = floats;
        floats += lat_floats;
        dsa_ik_[(size_t) mtp_il_] = floats;
        floats += (int64_t) g_.idx_key * ik_ring_;
        dsa_ig_[(size_t) mtp_il_] = floats;
        floats += (int64_t) g_.idx_key * ik_ring_;
        dsa_pool_[(size_t) mtp_il_] = floats;
        floats += (int64_t) g_.idx_key * max_pools;
    }
    for (int il = 0; il < g_.n_layers; ++il) {
        if (il < l0_ || il >= l1_) continue;   // the other half's caches stay on the other device
        if (g_.is_recr(il)) {
            kda_S_[(size_t) il] = floats;
            floats += (int64_t) g_.d_inner() * g_.kda_head_dim;
            kda_conv_[(size_t) il] = floats;
            floats += (int64_t) 3 * g_.d_inner() * (g_.d_conv - 1);
        } else {
            dsa_lat_[(size_t) il] = floats;
            floats += lat_floats;
            dsa_ik_[(size_t) il] = floats;
            floats += (int64_t) g_.idx_key * ik_ring_;
            dsa_ig_[(size_t) il] = floats;
            floats += (int64_t) g_.idx_key * ik_ring_;
            dsa_pool_[(size_t) il] = floats;
            floats += (int64_t) g_.idx_key * max_pools;
        }
    }
    state_bytes_ = (uint64_t) floats * sizeof(float);
    { cudaError_t e_ = cudaMalloc(&state_, state_bytes_); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return false; } }

    const int64_t ff_max = std::max<long long>({g_.n_ff_dense, (int64_t) g_.n_ff_exp * g_.n_shared});
    int64_t s = 0;
    auto take = [&](int64_t n) {
        const int64_t at2 = s;
        s += n;
        return at2;
    };
    sc_mixed = take(g_.n_embd);
    sc_emb = take(g_.n_embd);
    sc_x = take(g_.n_embd);
    sc_pre = take(g_.hc);
    sc_post = take(g_.hc);
    sc_comb = take((int64_t) g_.hc * g_.hc);
    sc_inv = take(1);
    sc_proj = take(g_.d_inner());
    sc_conv[0] = take(g_.d_inner());
    sc_conv[1] = take(g_.d_inner());
    sc_conv[2] = take(g_.d_inner());
    sc_g = take(g_.d_inner());
    sc_g1 = take(g_.d_inner());
    sc_beta = take(g_.n_head);
    sc_tmp = take(std::max<long long>({g_.kda_head_dim, g_.q_lora, g_.idx_key, (int64_t) g_.n_expert}));
    sc_scan = take(g_.d_inner());
    sc_g2 = take(g_.d_inner());
    sc_gated = take(g_.d_inner());
    sc_qr = take(g_.q_lora);
    sc_kv = take(g_.kv_lora);
    sc_q = take((int64_t) g_.n_head * g_.qk_nope);
    sc_qabs = take((int64_t) g_.kv_lora * g_.n_head);
    sc_iq = take((int64_t) g_.idx_heads * g_.idx_key);
    sc_iw = take(g_.idx_heads);
    sc_score = take(max_pools);
    sc_cells = take(g_.n_sel_max());
    sc_attn = take((int64_t) g_.v_head * g_.n_head);
    sc_gate = take(ff_max);
    sc_up = take(ff_max);
    sc_h = take(ff_max);
    sc_dn = take(g_.n_embd);
    sc_moe = take(g_.n_embd);
    sc_sh = take(g_.n_embd);
    sc_ffn = take(g_.n_embd);
    sc_mixer = take(g_.n_embd);
    sc_head = take(g_.n_embd);
    sc_logits = take(g_.n_vocab);
    sc_ids = take(g_.n_exp_used);
    sc_rw = take(g_.n_exp_used);
    sc_bytes_ = (uint64_t) s * sizeof(float);
    { cudaError_t e_ = cudaMalloc(&sc_, sc_bytes_); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return false; } }
    { cudaError_t e_ = cudaMalloc(&d_pos_, sizeof(int)); if (e_ != cudaSuccess) { err = std::string("pack: ") + cudaGetErrorString(e_); return false; } }

    // STRATA_GLM_TIMING=1: the phase events (step() records them; ~10 us/token)
    if (getenv("STRATA_GLM_TIMING")) {
        cudaEvent_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
        if (cudaEventCreate(&a) == cudaSuccess && cudaEventCreate(&b) == cudaSuccess &&
            cudaEventCreate(&c) == cudaSuccess && cudaEventCreate(&d) == cudaSuccess) {
            tev_a_ = a; tev_b_ = b; tev_c_ = c; tev_d_ = d;
            timing_ = true;
        }
    }

    // 6. the device expert pool (M2b): uniform slots sized for the largest per-layer blob, filled
    //    on miss straight from the mmap'd shards, LRU residency.  The grouped kernels read a
    //    host-mapped plan (UVA), so a token needs no plan copies.  STRATA_GLM_CPU_EXPERTS forces
    //    the CPU native path (fallback + A/B).
    dev_experts_ = !fast_mode_ && getenv("STRATA_GLM_CPU_EXPERTS") == nullptr && g_.n_exp_used <= 8;
    if (dev_experts_) {
        size_t gu_row_max = 0, d_row_max = 0;
        for (const auto& nl : pack_layers_) {
            if (nl.layer < 0) continue;
            gu_row_max = std::max(gu_row_max, nl.fmt.gu_row);
            d_row_max = std::max(d_row_max, nl.fmt.d_row);
        }
        slot_bytes_ = 2 * gu_row_max * (size_t) g_.n_ff_exp + d_row_max * (size_t) g_.n_embd;
        // SIZE CLASSES: one class per distinct (gu_row, d_row).  Uniform slots at the largest blob
        // pad ~20% of the arena (8.78 MB max vs 7.03 MB average on this pack); per-class sub-arenas
        // sized by layer-count-weighted demand recover it (~+25% slots).
        std::map<std::pair<size_t, size_t>, int> class_ids;
        layer_class_.assign((size_t) g_.n_layers, -1);
        std::vector<size_t> c_stride, c_weight;
        std::vector<int> c_layers;
        for (const auto& nl : pack_layers_) {
            if (nl.layer < 0) continue;
            const auto key = std::make_pair(nl.fmt.gu_row, nl.fmt.d_row);
            auto it = class_ids.find(key);
            if (it == class_ids.end()) {
                it = class_ids.emplace(key, (int) c_stride.size()).first;
                c_stride.push_back(2 * nl.fmt.gu_row * (size_t) g_.n_ff_exp + nl.fmt.d_row * (size_t) g_.n_embd);
                c_weight.push_back(0);
                c_layers.push_back(0);
            }
            layer_class_[(size_t) nl.layer] = it->second;
            c_weight[(size_t) it->second] += 1;
            c_layers[(size_t) it->second] += 1;
        }
        pool_nclasses_ = (int) c_stride.size();
        double weight_sum = 0;   // the byte demand: layers x stride, summed over the classes
        for (size_t c = 0; c < c_weight.size(); ++c) weight_sum += (double) c_weight[c] * (double) c_stride[c];
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t reserve = (size_t) 1536 << 20;   // context growth + the driver's share
        const size_t pool_avail = free_b > reserve ? free_b - reserve : 0;
        // STRATA_GLM_POOL_SLOTS0=n caps the pool at n largest-class-equivalent slots (debug/tuning)
        size_t pool_bytes_cap = pool_avail;
        if (const char* ps = getenv("STRATA_GLM_POOL_SLOTS0"))
            pool_bytes_cap = std::min<size_t>(pool_avail, (size_t) std::atoi(ps) * slot_bytes_);
        // per-class bytes proportional to that class's byte demand
        size_t off = 0;
        int64_t base = 0;
        int64_t slots_total = 0;
        for (int c = 0; c < pool_nclasses_; ++c) {
            class_stride_[c] = c_stride[(size_t) c];
            const size_t bytes_c = c_weight[(size_t) c] ? (size_t) ((double) pool_bytes_cap *
                                                ((double) c_weight[(size_t) c] * (double) c_stride[(size_t) c]) /
                                                weight_sum) : 0;
            class_slots_[c] = std::min<int64_t>((int64_t) (bytes_c / class_stride_[c]),
                                                (int64_t) c_layers[(size_t) c] * g_.n_expert);
            class_off_[c] = off;
            class_base_[c] = base;
            off += (size_t) class_slots_[c] * class_stride_[c];
            base += class_slots_[c];
            slots_total += class_slots_[c];
        }
        pool_bytes_ = off;
        dev_experts_ = slots_total >= g_.n_exp_used;
    }
    if (dev_experts_) {
        cudaError_t e_ = cudaMalloc(&pool_, pool_bytes_);
        if (e_ != cudaSuccess) { pool_ = nullptr; pool_bytes_ = 0; dev_experts_ = false; }
    }
    if (dev_experts_) {
        pool_slot_of_.assign((size_t) g_.n_layers * g_.n_expert, -1);
        pool_key_of_.assign((size_t) std::accumulate(class_slots_, class_slots_ + 8, (int64_t) 0), -1);
        pool_tick_.assign(pool_key_of_.size(), 0);
        pool_count_.assign(pool_key_of_.size(), 0);
        if (cudaHostAlloc(&plan_host_, sizeof(DevPlan), cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &plan_dev_, plan_host_, 0) != cudaSuccess ||
            cudaMalloc(&dev_xq_, (size_t) g_.n_embd / 32 * 36) != cudaSuccess ||
            cudaMalloc(&dev_scratch_, strata::kernels::native_expert_scratch_bytes(g_.n_exp_used, g_.n_ff_exp)) !=
                cudaSuccess ||
            cudaMalloc(&dev_hit_out_, (size_t) g_.n_exp_used * (size_t) g_.n_embd * sizeof(float)) != cudaSuccess) {
            err = "pack: the device expert pool could not allocate its scratch";
            return false;
        }
        // pinned staging ring (STRATA_GLM_BOUNCE=1): a pageable-src H2D copy whose destination
        // lives on a PEER device deadlocked the driver here (D-state in os_acquire_mutex) - and
        // pinned staging is faster than the driver's pageable bounce even for local slots.  24
        // slots drain every layer.
        constexpr int kBounceSlots = 24;
        if (getenv("STRATA_GLM_BOUNCE") &&
            cudaHostAlloc(&bounce_, (size_t) kBounceSlots * slot_bytes_, cudaHostAllocDefault) != cudaSuccess) {
            err = "pack: the device expert pool could not allocate its pinned staging ring";
            return false;
        }
        // the staging pool: one pinned buffer + one worker per possible miss (n_exp_used <= 8)
        stage_on_ = getenv("STRATA_GLM_NO_STAGE") == nullptr;
        if (stage_on_) {
            stage_bufs_.assign((size_t) g_.n_exp_used, nullptr);
            for (auto& b : stage_bufs_) {
                if (cudaHostAlloc(&b, slot_bytes_, cudaHostAllocDefault) != cudaSuccess) {
                    stage_on_ = false;
                    break;
                }
            }
        }
        if (stage_on_) {
            for (size_t i = 0; i < stage_bufs_.size(); ++i)
                stage_workers_.emplace_back([this] { stage_worker(); });
        }
    }
    // the sampler's output buffer lives on THIS half's device (sample_token switches devices for it)
    if (cudaMalloc(&d_tok_, sizeof(int)) != cudaSuccess) {
        err = "pack: sampler buffer";
        return false;
    }

    host_x_.resize((size_t) g_.n_embd);
    host_moe_.resize((size_t) g_.n_embd);
    host_ff_.resize((size_t) g_.n_ff_exp);
    host_dn_.resize((size_t) g_.n_embd);
    host_act_.resize(strata::kernels::cpu::kNativeActBytes);
    host_hq_.resize(strata::kernels::cpu::kNativeHBytes);
    pack_ = true;
    loaded_ = true;
    if (fast_mode_) pack_release_views();   // before fast_setup: its RAM tier sees the embedding's heap copy
    if (fast_mode_ && !fast_setup(err)) return false;
    reset();
    return true;
}

// ---------------------------------------------------------------- the NextN (MTP) block
// blk.<n_layers>.* is the draft block a speculative decode proposes the token after next with: a DSA + MoE layer
// without hyper-connections, read through eh_proj from [enorm(embedding of the next token), hnorm(final hidden
// state)].  The pack never carried it, so its tensors come straight from the shards: the quantized projections stay
// quantized, the small rows the trunk's kernels expect as BF16 (indexer, router, absorbed k/v) are widened at load,
// and its routed experts join the expert tiers as one more MoE layer (pack_layers_[n_layers]).
bool strata::core::Glm5Model::load_mtp(const std::vector<std::unique_ptr<strata::GgufFile>>& gfs, std::string& err) {
    const int il = g_.n_layers;
    const std::string P = "blk." + std::to_string(il) + ".";
    const auto find = [&](const std::string& n, int& si) -> const strata::TensorInfo* {
        if (mtp_src_ >= 0 && n.compare(0, P.size(), P) == 0)   // the block from STRATA_GLM_MTP_GGUF wins
            if (const strata::TensorInfo* t = gfs[(size_t) mtp_src_]->find(n)) {
                si = mtp_src_;
                return t;
            }
        for (size_t i = 0; i < gfs.size(); ++i)
            if (const strata::TensorInfo* t = gfs[i]->find(n)) {
                si = (int) i;
                return t;
            }
        si = -1;
        return nullptr;
    };
    int si = -1;
    if (find(P + "nextn.eh_proj.weight", si) == nullptr) return true;   // a GGUF without the draft block
    const auto& raw_names = kMtpRaw;
    const auto& b16_names = kMtpB16;
    const auto& f32_names = kMtpF32;
    uint64_t bytes = 0;
    if (!mtp_dense_vram([&](const std::string& n) { return find(n, si); }, il, bytes, err)) return false;
    if (cudaMalloc(&mtp_arena_, bytes) != cudaSuccess) {
        cudaGetLastError();
        err = "mtp: the draft block's weights (" + std::to_string(bytes >> 20) + " MB) did not allocate";
        return false;
    }
    uint64_t at = 0;
    uint8_t* base = (uint8_t*) mtp_arena_;
    std::vector<float> host;
    const auto src_of = [&](const strata::TensorInfo* t, int s) {
        return pack_shards_[(size_t) s].base + gfs[(size_t) s]->data_start() + t->offset;
    };
    const auto to_host = [&](const strata::TensorInfo* t, int s) -> bool {
        if (t->type == 0) {   // F32
            host.resize((size_t) t->elements());
            std::memcpy(host.data(), src_of(t, s), (size_t) t->elements() * 4);
            return true;
        }
        return pack_dequant_tensor(*t, src_of(t, s), host, err);
    };
    for (const char* n : raw_names) {
        const strata::TensorInfo* t = find(P + n, si);
        const size_t nb = strata::kernels::native_mmvq_weight_bytes(t->type, (int) t->shape[0], (int) t->shape[1]);
        if (cudaMemcpy(base + at, src_of(t, si), nb, cudaMemcpyHostToDevice) != cudaSuccess) { err = "mtp: upload"; return false; }
        ws_map_[P + n] = strata::core::WSlot{nullptr, (int) t->type, (const void*) (base + at), (int64_t) t->shape[0],
                                             (int64_t) t->shape[1]};
        at += (nb + 255u) & ~(uint64_t) 255u;
    }
    std::vector<uint16_t> b16;
    for (const char* n : b16_names) {
        const strata::TensorInfo* t = find(P + n, si);
        if (!to_host(t, si)) return false;
        b16.resize(host.size());
        for (size_t i = 0; i < host.size(); ++i) {   // round to nearest even
            uint32_t u;
            std::memcpy(&u, &host[i], 4);
            u += 0x7fffu + ((u >> 16) & 1u);
            b16[i] = (uint16_t) (u >> 16);
        }
        if (cudaMemcpy(base + at, b16.data(), b16.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) { err = "mtp: upload"; return false; }
        w16_[P + n] = (const uint16_t*) (base + at);
        at += ((uint64_t) b16.size() * 2 + 255u) & ~(uint64_t) 255u;
    }
    for (const char* n : f32_names) {
        const strata::TensorInfo* t = find(P + n, si);
        if (!to_host(t, si)) return false;
        if (cudaMemcpy(base + at, host.data(), host.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) { err = "mtp: upload"; return false; }
        w_[P + n] = (const float*) (base + at);
        ws_map_[P + n] = strata::core::WSlot{(const float*) (base + at), 0, nullptr, 0, 0};
        at += ((uint64_t) host.size() * 4 + 255u) & ~(uint64_t) 255u;
    }
    // the routed experts: gate/up/down by absolute offset, like native_experts.txt rows
    NativeLayer nl;
    nl.layer = il;
    int sg = -1, su = -1, sd = -1;
    const strata::TensorInfo* tg = find(P + "ffn_gate_exps.weight", sg);
    const strata::TensorInfo* tu = find(P + "ffn_up_exps.weight", su);
    const strata::TensorInfo* td = find(P + "ffn_down_exps.weight", sd);
    if (tg == nullptr || tu == nullptr || td == nullptr || tg->type != tu->type) {
        err = "mtp: the draft block's routed experts are missing or mixed";
        return false;
    }
    if (!strata::kernels::cpu::native_fmt((int) tg->type, (int) td->type, g_.n_embd, g_.n_ff_exp, nl.fmt, err)) return false;
    nl.gate_shard = sg;
    nl.up_shard = su;
    nl.down_shard = sd;
    nl.gate_off = gfs[(size_t) sg]->data_start() + tg->offset;
    nl.up_off = gfs[(size_t) su]->data_start() + tu->offset;
    nl.down_off = gfs[(size_t) sd]->data_start() + td->offset;
    if (pack_layers_.size() < (size_t) il + 1) pack_layers_.resize((size_t) il + 1);
    pack_layers_[(size_t) il] = nl;
    // the draft block embeds the next token: the embedding rows come from the shard mapping on this half too
    if (pack_emb_src_ == nullptr) {
        int se = -1;
        const strata::TensorInfo* te = find("token_embd.weight", se);
        if (te == nullptr) { err = "mtp: token_embd not found"; return false; }
        pack_emb_src_ = src_of(te, se);
        pack_emb_type_ = (int) te->type;
    }
    mtp_il_ = il;
    std::fprintf(stderr, "glm mtp: CUDA%d the NextN block (blk.%d) loaded: %.0f MB dense, experts %s/%s\n", dev_, il,
                 (double) bytes / 1048576.0, ggml_type_name((ggml_type) tg->type), ggml_type_name((ggml_type) td->type));
    return true;
}
