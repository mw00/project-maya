// src/ling/ling_vision.cu - Ling-3.2-air qwen3_moe_vit + linear_proj CUDA vision tower
#include "ling_vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "maya/artifact/gguf_reader.hpp"

namespace ling {

#define VK(call)                                                                           \
    do {                                                                                   \
        const cudaError_t e_ = (call);                                                     \
        if (e_ != cudaSuccess) {                                                           \
            err = std::string("ling_vision: ") + #call + ": " + cudaGetErrorString(e_);    \
            return false;                                                                  \
        }                                                                                  \
    } while (0)
#define VB(call)                                                                           \
    do {                                                                                   \
        const cublasStatus_t b_ = (call);                                                  \
        if (b_ != CUBLAS_STATUS_SUCCESS) {                                                 \
            err = std::string("ling_vision: ") + #call + ": cublas " + std::to_string((int) b_); \
            return false;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

using bf16_t = __nv_bfloat16;

constexpr size_t kScoreBudget = (size_t) 64 << 20;   // <= 64 MiB attention score tile
constexpr int kFfnChunk = 2048;                      // MLP tile rows
constexpr size_t kCublasWs = (size_t) 16 << 20;      // 16 MiB cuBLAS workspace

inline size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }

__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286536f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

__device__ __forceinline__ float warp_sum_v(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max_v(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ float block_sum_v(float v, float* red) {
    v = warp_sum_v(v);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) red[w] = v;
    __syncthreads();
    float s = 0.0f;
    const int nw = (int) (blockDim.x >> 5);
    for (int i = 0; i < nw; ++i) s += red[i];
    __syncthreads();
    return s;
}

// Pack RGB uint8 [H][W][3] into merged-block patch matrix [N][1536] BF16 matching BailingMM2ImageProcessor:
// patch idx p = ((by * mw + bx) * 2 + sy) * 2 + sx; feat idx = ((c * 2 + t) * 16 + dy) * 16 + dx (t=0,1 identical)
__global__ void patch_pack_kernel(const uint8_t* __restrict__ rgb, int H, int W, int mh, int mw,
                                  float m0, float m1, float m2, float s0, float s1, float s2,
                                  bf16_t* __restrict__ out) {
    const int p = blockIdx.x;
    const int tid = threadIdx.x;
    const int sub = p & 3;
    const int blk = p >> 2;
    const int bx = blk % mw;
    const int by = blk / mw;
    const int sy = sub >> 1;
    const int sx = sub & 1;
    const int y0 = (by * 2 + sy) * 16;
    const int x0 = (bx * 2 + sx) * 16;
    const float mean[3] = {m0, m1, m2};
    const float invs[3] = {1.0f / s0, 1.0f / s1, 1.0f / s2};
    bf16_t* dst = out + (size_t) p * 1536;

    // 256 threads = 16 x 16 pixels in patch; each thread writes 3 channels x 2 temporal copies = 6 values
    const int dy = tid >> 4;
    const int dx = tid & 15;
    const int y = y0 + dy;
    const int x = x0 + dx;
    const uint8_t* px = rgb + ((size_t) y * W + x) * 3;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float v = ((float) px[c] * (1.0f / 255.0f) - mean[c]) * invs[c];
        const bf16_t hv = __float2bfloat16(v);
        dst[(c * 2 + 0) * 256 + tid] = hv;
        dst[(c * 2 + 1) * 256 + tid] = hv;
    }
}

// Add patch_embed bias + bilinearly interpolated 48x48 pos_embed (matching fast_pos_embed_interpolate exactly!)
__global__ void add_bias_pos_kernel(float* __restrict__ x, const float* __restrict__ bias,
                                    const bf16_t* __restrict__ pos_emb, int N, int D, int gh, int gw, int mw,
                                    int grid_side) {
    const int p = blockIdx.x;
    const int sub = p & 3;
    const int blk = p >> 2;
    const int bx = blk % mw;
    const int by = blk / mw;
    const int py = by * 2 + (sub >> 1);
    const int px = bx * 2 + (sub & 1);

    const float hf = (gh > 1) ? ((float) py * (float) (grid_side - 1) / (float) (gh - 1)) : 0.0f;
    const float wf = (gw > 1) ? ((float) px * (float) (grid_side - 1) / (float) (gw - 1)) : 0.0f;
    const int h0 = (int) floorf(hf);
    const int w0 = (int) floorf(wf);
    const int h1 = min(h0 + 1, grid_side - 1);
    const int w1 = min(w0 + 1, grid_side - 1);
    const float dh = hf - (float) h0;
    const float dw = wf - (float) w0;
    const float w11 = dh * dw;
    const float w10 = dh - w11;
    const float w01 = dw - w11;
    const float w00 = 1.0f - dh - w01;
    // Match HF weights.to(bf16) precision
    const float bw00 = __bfloat162float(__float2bfloat16(w00));
    const float bw01 = __bfloat162float(__float2bfloat16(w01));
    const float bw10 = __bfloat162float(__float2bfloat16(w10));
    const float bw11 = __bfloat162float(__float2bfloat16(w11));

    const bf16_t* e00 = pos_emb + (size_t) (h0 * grid_side + w0) * D;
    const bf16_t* e01 = pos_emb + (size_t) (h0 * grid_side + w1) * D;
    const bf16_t* e10 = pos_emb + (size_t) (h1 * grid_side + w0) * D;
    const bf16_t* e11 = pos_emb + (size_t) (h1 * grid_side + w1) * D;
    float* row = x + (size_t) p * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float pe = bw00 * __bfloat162float(e00[d]) + bw01 * __bfloat162float(e01[d]) +
                         bw10 * __bfloat162float(e10[d]) + bw11 * __bfloat162float(e11[d]);
        row[d] = row[d] + bias[d] + pe;
    }
}

__global__ void layernorm_bf16_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                      const float* __restrict__ b, int D, float eps, bf16_t* __restrict__ y16) {
    __shared__ float red[16];
    const float* r = x + (size_t) blockIdx.x * D;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += r[i];
    const float mean = block_sum_v(s, red) / (float) D;
    float q = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float d = r[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(block_sum_v(q, red) / (float) D + eps);
    bf16_t* dst = y16 + (size_t) blockIdx.x * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = __float2bfloat16((r[i] - mean) * inv * w[i] + b[i]);
    }
}

// Split QKV [N][3*D] + bias -> Q, K, V in [H][N][HD] (F32) and apply 2D Vision RoPE (rotary_dim=36, theta=10000, NeoX half-rot)
__global__ void qkv_split_rope2d_kernel(const float* __restrict__ qkv, const float* __restrict__ bias, int N, int H,
                                        int HD, int mw, float theta, float* __restrict__ Q, float* __restrict__ K,
                                        float* __restrict__ V) {
    const int p = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;   // 0 .. HD-1 (72 threads, launch 96 or 128)
    if (tid >= HD) return;
    const int D = H * HD;
    const int half = HD >> 1;      // 36
    const int qtr = half >> 1;     // 18

    const int sub = p & 3;
    const int blk = p >> 2;
    const int bx = blk % mw;
    const int by = blk / mw;
    const int py = by * 2 + (sub >> 1);
    const int px = bx * 2 + (sub & 1);

    const float* src = qkv + (size_t) p * 3 * D;
    const size_t dst_off = ((size_t) h * N + p) * HD;

    // V needs no RoPE
    V[dst_off + tid] = src[2 * D + h * HD + tid] + bias[2 * D + h * HD + tid];

    // Q and K: NeoX half-rotation pairs (i, i + 36) for i in 0..35
    if (tid < half) {
        const int i = tid;
        const int k = i < qtr ? i : (i - qtr);
        const float pos = (i < qtr) ? (float) py : (float) px;
        const float inv_f = 1.0f / powf(theta, (float) (2 * k) / (float) half);
        const float ang = pos * inv_f;
        float s_val, c_val;
        __sincosf(ang, &s_val, &c_val);

        const float q0 = src[0 * D + h * HD + i] + bias[0 * D + h * HD + i];
        const float q1 = src[0 * D + h * HD + i + half] + bias[0 * D + h * HD + i + half];
        Q[dst_off + i] = q0 * c_val - q1 * s_val;
        Q[dst_off + i + half] = q1 * c_val + q0 * s_val;

        const float k0 = src[1 * D + h * HD + i] + bias[1 * D + h * HD + i];
        const float k1 = src[1 * D + h * HD + i + half] + bias[1 * D + h * HD + i + half];
        K[dst_off + i] = k0 * c_val - k1 * s_val;
        K[dst_off + i + half] = k1 * c_val + k0 * s_val;
    }
}

// Row-wise softmax in-place over S [H * BQ][N]
__global__ void softmax_rows_kernel(float* __restrict__ S, int N) {
    __shared__ float red[16];
    float* row = S + (size_t) blockIdx.x * N;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = warp_max_v(mx);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) red[w] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = red[0];
        const int nw = (int) (blockDim.x >> 5);
        for (int i = 1; i < nw; ++i) m = fmaxf(m, red[i]);
        red[0] = m;
    }
    __syncthreads();
    const float m = red[0];
    float sm = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        const float v = expf(row[j] - m);
        row[j] = v;
        sm += v;
    }
    const float inv = 1.0f / block_sum_v(sm, red);
    for (int j = threadIdx.x; j < N; j += blockDim.x) row[j] *= inv;
}

// Transpose attention output [H][BQ][HD] (F32) -> [BQ][H*HD] (BF16) at rows [q0 .. q0+BQ)
__global__ void attn_out_to_bf16_kernel(const float* __restrict__ O, int H, int BQ, int HD, bf16_t* __restrict__ dst) {
    const int q = blockIdx.x;
    const int D = H * HD;
    bf16_t* out_row = dst + (size_t) q * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const int h = d / HD;
        const int c = d - h * HD;
        out_row[d] = __float2bfloat16(O[((size_t) h * BQ + q) * HD + c]);
    }
}

__global__ void add_bias_res_kernel(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ b,
                                    size_t total, int D) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += y[i] + b[i % (size_t) D];
}

__global__ void gelu_tanh_bias_bf16_kernel(const float* __restrict__ in, const float* __restrict__ b, size_t total,
                                           int FF, bf16_t* __restrict__ out16) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out16[i] = __float2bfloat16(gelu_tanh(in[i] + b[i % (size_t) FF]));
}

__global__ void gelu_erf_bias_bf16_kernel(const float* __restrict__ in, const float* __restrict__ b, size_t total,
                                          int D, bf16_t* __restrict__ out16) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out16[i] = __float2bfloat16(gelu_erf(in[i] + b[i % (size_t) D]));
}

__global__ void add_bias_inplace_kernel(float* __restrict__ y, const float* __restrict__ b, size_t total, int D) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] += b[i % (size_t) D];
}

// In-place BF16 -> FP16 conversion (identical 2-byte storage) for sm_70/sm_75 portability where cuBLAS lacks BF16.
__global__ void bf16_to_f16_inplace_kernel(uint16_t* __restrict__ p, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const __nv_bfloat16 bv = *reinterpret_cast<const __nv_bfloat16*>(p + i);
        *reinterpret_cast<__half*>(p + i) = __float2half_rn(__bfloat162float(bv));
    }
}

}  // namespace

struct VisionTower::Impl {
    std::unique_ptr<maya::GgufFile> gf;
    VisionGeo g;
    int dev = 0;
    bool ok = false;
    size_t last_b = 0;
    bool last_l = false;

    // Pointers directly into mmap'd mmproj GGUF on host (zero extra host RAM copy!)
    struct BlockPtrs {
        const void *n1_w, *n1_b, *n2_w, *n2_b;
        const void *qkv_w, *qkv_b, *proj_w, *proj_b;
        const void *fc1_w, *fc1_b, *fc2_w, *fc2_b;
    };
    const void *patch_w = nullptr, *patch_b = nullptr, *pos_w = nullptr;
    const void *m_norm_w = nullptr, *m_norm_b = nullptr;
    const void *lp0_w = nullptr, *lp0_b = nullptr, *lp2_w = nullptr, *lp2_b = nullptr;
    std::vector<BlockPtrs> blocks;

    // Single-block weight staging bytes on GPU
    size_t block_w_bytes() const {
        const size_t D = (size_t) g.dim, FF = (size_t) g.ff;
        size_t s = 0;
        s += align256(D * 4) * 4;              // n1_w, n1_b, n2_w, n2_b (F32)
        s += align256(3 * D * D * 2);          // qkv_w (BF16)
        s += align256(3 * D * 4);              // qkv_b (F32)
        s += align256(D * D * 2);              // proj_w (BF16)
        s += align256(D * 4);                  // proj_b (F32)
        s += align256(FF * D * 2);             // fc1_w (BF16)
        s += align256(FF * 4);                 // fc1_b (F32)
        s += align256(D * FF * 2);             // fc2_w (BF16)
        s += align256(D * 4);                  // fc2_b (F32)
        // Also covers projector + patch_embed + pos_embed when reused before/after blocks
        const size_t p_init = align256(D * 1536 * 2) + align256(D * 4) + align256((size_t) g.grid * g.grid * D * 2);
        const size_t p_proj = align256(D * 4) * 2 + align256((size_t) g.out_dim * 4 * D * 2) +
                              align256((size_t) g.out_dim * 4) + align256((size_t) g.out_dim * g.out_dim * 2) +
                              align256((size_t) g.out_dim * 4);
        return std::max({s, p_init, p_proj});
    }

    size_t room_for(int width, int height) const {
        const int tok = g.tokens(width, height);
        if (tok <= 0) return 0;
        const size_t N = (size_t) tok * 4;
        const size_t D = (size_t) g.dim, FF = (size_t) g.ff, OD = (size_t) g.out_dim;
        size_t s = kCublasWs;
        s += align256((size_t) height * width * 3);                    // d_rgb
        s += align256(N * D * 4);                                      // x (F32)
        s += align256(std::max(N * 1536 * 2, N * D * 2));              // x16 / patches16 (BF16)
        s += align256(block_w_bytes());                                // 1-block / proj weights
        // Shared working scratch for QKV/attn OR MLP OR projector
        const size_t qkv_b = align256(N * 3 * D * 4) + align256(N * D * 4) * 3 + align256(kScoreBudget) +
                             align256((size_t) g.heads * 512 * g.hd * 4);
        const size_t mlp_b = align256((size_t) kFfnChunk * FF * 4) + align256((size_t) kFfnChunk * FF * 2) +
                             align256((size_t) kFfnChunk * D * 4);
        const size_t prj_b = align256((size_t) tok * OD * 4) * 2 + align256((size_t) tok * OD * 2);
        s += std::max({qkv_b, mlp_b, prj_b}) + 4096;
        return s;
    }
};

VisionTower::VisionTower() : p_(std::make_unique<Impl>()) {}
VisionTower::~VisionTower() = default;
bool VisionTower::loaded() const { return p_->ok; }
const VisionGeo& VisionTower::geo() const { return p_->g; }
size_t VisionTower::last_bytes() const { return p_->last_b; }
bool VisionTower::last_lent() const { return p_->last_l; }
int VisionTower::device() const { return p_->dev; }

size_t VisionTower::room_of(int width, int height) const { return p_->room_for(width, height); }
size_t VisionTower::max_room_bytes() const {
    const int f = p_->g.factor();
    const int side = ((int) std::sqrt((double) p_->g.max_pixels) / f) * f;
    return p_->room_for(side, side);
}

bool VisionTower::load(const std::string& mmproj_path, int dev, std::string& err) {
    try {
        p_->gf = std::make_unique<maya::GgufFile>(mmproj_path);
    } catch (const std::exception& e) {
        err = std::string("ling_vision: open failed: ") + e.what();
        return false;
    }
    const maya::GgufFile& gf = *p_->gf;
    auto opt_num = [&](const std::string& k, double dflt) -> double {
        const maya::MetaValue* v = gf.get(k);
        return (v != nullptr && v->is_num()) ? v->num() : dflt;
    };
    auto opt_arr = [&](const std::string& k) -> std::vector<double> {
        maya::MetaValue v;
        if (!gf.get_full(k, v)) return {};
        std::vector<double> out;
        out.reserve((size_t) v.count);
        for (const auto& e : v.items) out.push_back(e.num());
        return out;
    };
    p_->dev = dev;
    VisionGeo& g = p_->g;
    g.layers = (int) opt_num("clip.vision.block_count", 27);
    g.dim = (int) opt_num("clip.vision.embedding_length", 1152);
    g.ff = (int) opt_num("clip.vision.feed_forward_length", 4304);
    g.heads = (int) opt_num("clip.vision.attention.head_count", 16);
    g.hd = g.dim / g.heads;
    g.patch = (int) opt_num("clip.vision.patch_size", 16);
    g.merge = (int) opt_num("clip.vision.spatial_merge_size", 2);
    g.temporal_patch = (int) opt_num("clip.vision.temporal_patch_size", 2);
    const int npos = (int) opt_num("clip.vision.num_position_embeddings", 2304);
    g.grid = (int) std::lround(std::sqrt((double) npos));
    g.out_dim = (int) opt_num("clip.vision.projection_dim", 2560);
    g.eps = (float) opt_num("clip.vision.attention.layer_norm_epsilon", 1e-6);
    g.rope_theta = (float) opt_num("clip.vision.rope_theta", 10000.0);
    g.min_pixels = (int) opt_num("clip.vision.min_pixels", 4096);
    g.max_pixels = (int) opt_num("clip.vision.max_pixels", 4194304);
    g.image_token = (int) opt_num("clip.vision.image_token_id", 157157);
    g.image_start = (int) opt_num("clip.vision.vision_start_token_id", 157158);
    g.image_end = (int) opt_num("clip.vision.vision_end_token_id", 157159);
    const auto mean_v = opt_arr("clip.vision.image_mean");
    if (mean_v.size() == 3) for (int c = 0; c < 3; ++c) g.mean[c] = (float) mean_v[(size_t) c];
    const auto std_v = opt_arr("clip.vision.image_std");
    if (std_v.size() == 3) for (int c = 0; c < 3; ++c) g.stdv[c] = (float) std_v[(size_t) c];
    const auto mr_v = opt_arr("clip.vision.mrope_section");
    if (mr_v.size() == 3) for (int c = 0; c < 3; ++c) g.mrope_section[c] = (int) mr_v[(size_t) c];

    auto get_t = [&](const std::string& name, int expect_type) -> const void* {
        const maya::TensorInfo* ti = gf.find(name);
        if (ti == nullptr) {
            err = "ling_vision: missing tensor " + name;
            return nullptr;
        }
        if ((int) ti->type != expect_type) {
            err = "ling_vision: bad type for " + name + " (got " + std::to_string((int) ti->type) + ")";
            return nullptr;
        }
        return gf.tensor_data(*ti);
    };

    p_->patch_w = get_t("model.visual.patch_embed.proj.weight", 30);
    p_->patch_b = get_t("model.visual.patch_embed.proj.bias", 0);
    p_->pos_w = get_t("model.visual.pos_embed.weight", 30);
    p_->m_norm_w = get_t("model.visual.merger.norm.weight", 0);
    p_->m_norm_b = get_t("model.visual.merger.norm.bias", 0);
    p_->lp0_w = get_t("linear_proj.0.weight", 30);
    p_->lp0_b = get_t("linear_proj.0.bias", 0);
    p_->lp2_w = get_t("linear_proj.2.weight", 30);
    p_->lp2_b = get_t("linear_proj.2.bias", 0);
    if (!p_->patch_w || !p_->patch_b || !p_->pos_w || !p_->m_norm_w || !p_->m_norm_b || !p_->lp0_w || !p_->lp0_b ||
        !p_->lp2_w || !p_->lp2_b)
        return false;

    p_->blocks.resize((size_t) g.layers);
    for (int i = 0; i < g.layers; ++i) {
        const std::string pre = "model.visual.blocks." + std::to_string(i) + ".";
        auto& b = p_->blocks[(size_t) i];
        b.n1_w = get_t(pre + "norm1.weight", 0);
        b.n1_b = get_t(pre + "norm1.bias", 0);
        b.n2_w = get_t(pre + "norm2.weight", 0);
        b.n2_b = get_t(pre + "norm2.bias", 0);
        b.qkv_w = get_t(pre + "attn.qkv.weight", 30);
        b.qkv_b = get_t(pre + "attn.qkv.bias", 0);
        b.proj_w = get_t(pre + "attn.proj.weight", 30);
        b.proj_b = get_t(pre + "attn.proj.bias", 0);
        b.fc1_w = get_t(pre + "mlp.linear_fc1.weight", 30);
        b.fc1_b = get_t(pre + "mlp.linear_fc1.bias", 0);
        b.fc2_w = get_t(pre + "mlp.linear_fc2.weight", 30);
        b.fc2_b = get_t(pre + "mlp.linear_fc2.bias", 0);
        if (!b.n1_w || !b.n1_b || !b.n2_w || !b.n2_b || !b.qkv_w || !b.qkv_b || !b.proj_w || !b.proj_b ||
            !b.fc1_w || !b.fc1_b || !b.fc2_w || !b.fc2_b)
            return false;
    }
    p_->ok = true;
    return true;
}

bool VisionTower::encode(const uint8_t* rgb, int width, int height, std::vector<float>& out_rows, std::string& err,
                         uint8_t* lent_room, size_t lent_bytes) {
    if (!p_->ok) {
        err = "ling_vision: tower not loaded";
        return false;
    }
    const VisionGeo& g = p_->g;
    const int tok = g.tokens(width, height);
    if (tok <= 0) {
        err = "ling_vision: invalid image dimensions " + std::to_string(width) + "x" + std::to_string(height);
        return false;
    }
    const int gh = height / g.patch, gw = width / g.patch;
    const int mh = gh / g.merge, mw = gw / g.merge;
    const int N = gh * gw;
    const int D = g.dim, H = g.heads, HD = g.hd, FF = g.ff, OD = g.out_dim;

    const size_t need = room_of(width, height);
    VK(cudaSetDevice(p_->dev));

    uint8_t* owned = nullptr;
    uint8_t* arena = lent_room;
    const bool use_lent = (lent_room != nullptr && lent_bytes >= need);
    if (!use_lent) {
        VK(cudaMalloc(&owned, need));
        arena = owned;
    }
    p_->last_b = need;
    p_->last_l = use_lent;

    struct Cleanup {
        uint8_t* mem = nullptr;
        cublasHandle_t blas = nullptr;
        cudaStream_t s = nullptr;
        ~Cleanup() {
            if (blas != nullptr) cublasDestroy(blas);
            if (s != nullptr) cudaStreamDestroy(s);
            if (mem != nullptr) cudaFree(mem);
        }
    } cl;
    cl.mem = owned;

    VK(cudaStreamCreateWithFlags(&cl.s, cudaStreamNonBlocking));
    VB(cublasCreate(&cl.blas));
    VB(cublasSetStream(cl.blas, cl.s));

    size_t off = 0;
    auto take = [&](size_t bytes) -> uint8_t* {
        uint8_t* ptr = arena + off;
        off += align256(bytes);
        return ptr;
    };
    int cc_maj = 8;
    VK(cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, p_->dev));
    const bool use_bf16 = (cc_maj >= 8);
#ifndef MAYA_USE_HIP
    const cudaDataType_t gemm_dt = use_bf16 ? CUDA_R_16BF : CUDA_R_16F;
    const cublasComputeType_t attn_compute = use_bf16 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

    uint8_t* ws_ptr = take(kCublasWs);
    VB(cublasSetWorkspace(cl.blas, ws_ptr, kCublasWs));
    if (use_bf16) VB(cublasSetMathMode(cl.blas, CUBLAS_TF32_TENSOR_OP_MATH));
#else
    const auto gemm_dt = CUDA_R_16F;
    const cublasComputeType_t attn_compute = CUBLAS_COMPUTE_32F;
    uint8_t* ws_ptr = take(kCublasWs);
    VB(cublasSetWorkspace(cl.blas, ws_ptr, kCublasWs));
#endif

    uint8_t* d_rgb = take((size_t) height * width * 3);
    float* d_x = (float*) take((size_t) N * D * 4);
    bf16_t* d_x16 = (bf16_t*) take(std::max((size_t) N * 1536 * 2, (size_t) N * D * 2));
    uint8_t* d_wbuf = take(p_->block_w_bytes());
    uint8_t* d_work = arena + off;

    auto cvt_if_f16 = [&](void* ptr, size_t elems) {
        if (!use_bf16 && elems > 0) {
            bf16_to_f16_inplace_kernel<<<(unsigned) ((elems + 255) / 256), 256, 0, cl.s>>>((uint16_t*) ptr, elems);
        }
    };

    // Helper: row-major Y[M, N_out] (F32) = X[M, K] (BF16 or FP16) @ W[N_out, K]^T (BF16 or FP16)
    auto gemm_16 = [&](const bf16_t* X, const bf16_t* W, int M, int N_out, int K, float* Y) -> bool {
        const float alpha = 1.0f, beta = 0.0f;
        VB(cublasGemmEx(cl.blas, CUBLAS_OP_T, CUBLAS_OP_N, N_out, M, K, &alpha, W, gemm_dt, K, X, gemm_dt, K,
                        &beta, Y, CUDA_R_32F, N_out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return true;
    };

    // 1. Upload RGB + Patch Embed + Bilinear Pos Embed
    VK(cudaMemcpyAsync(d_rgb, rgb, (size_t) height * width * 3, cudaMemcpyHostToDevice, cl.s));
    patch_pack_kernel<<<(unsigned) N, 256, 0, cl.s>>>(d_rgb, height, width, mh, mw, g.mean[0], g.mean[1], g.mean[2],
                                                      g.stdv[0], g.stdv[1], g.stdv[2], d_x16);
    cvt_if_f16(d_x16, (size_t) N * 1536);
    {
        size_t woff = 0;
        auto wcopy = [&](const void* src, size_t bytes, bool is_w16 = false) -> uint8_t* {
            uint8_t* dst = d_wbuf + woff;
            woff += align256(bytes);
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, cl.s);
            if (is_w16) cvt_if_f16(dst, bytes / 2);
            return dst;
        };
        const bf16_t* d_pw = (const bf16_t*) wcopy(p_->patch_w, (size_t) D * 1536 * 2, true);
        const float* d_pb = (const float*) wcopy(p_->patch_b, (size_t) D * 4);
        const bf16_t* d_pe = (const bf16_t*) wcopy(p_->pos_w, (size_t) g.grid * g.grid * D * 2, false);
        if (!gemm_16(d_x16, d_pw, N, D, 1536, d_x)) return false;
        add_bias_pos_kernel<<<(unsigned) N, 256, 0, cl.s>>>(d_x, d_pb, d_pe, N, D, gh, gw, mw, g.grid);
    }

    // 2. 27 Transformer Blocks (stream 1 block of weights at a time into d_wbuf)
    const int BQ_max = std::max(1, std::min(N, (int) (kScoreBudget / ((size_t) H * (size_t) N * 4))));
    const int BQ_cap = std::min(BQ_max, 512);
    const float attn_scale = 1.0f / std::sqrt((float) HD);

    for (int il = 0; il < g.layers; ++il) {
        const auto& b = p_->blocks[(size_t) il];
        size_t woff = 0;
        auto wcopy = [&](const void* src, size_t bytes, bool is_w16 = false) -> uint8_t* {
            uint8_t* dst = d_wbuf + woff;
            woff += align256(bytes);
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, cl.s);
            if (is_w16) cvt_if_f16(dst, bytes / 2);
            return dst;
        };
        const float* n1_w = (const float*) wcopy(b.n1_w, (size_t) D * 4);
        const float* n1_b = (const float*) wcopy(b.n1_b, (size_t) D * 4);
        const float* n2_w = (const float*) wcopy(b.n2_w, (size_t) D * 4);
        const float* n2_b = (const float*) wcopy(b.n2_b, (size_t) D * 4);
        const bf16_t* qkv_w = (const bf16_t*) wcopy(b.qkv_w, (size_t) 3 * D * D * 2, true);
        const float* qkv_b = (const float*) wcopy(b.qkv_b, (size_t) 3 * D * 4);
        const bf16_t* proj_w = (const bf16_t*) wcopy(b.proj_w, (size_t) D * D * 2, true);
        const float* proj_b = (const float*) wcopy(b.proj_b, (size_t) D * 4);
        const bf16_t* fc1_w = (const bf16_t*) wcopy(b.fc1_w, (size_t) FF * D * 2, true);
        const float* fc1_b = (const float*) wcopy(b.fc1_b, (size_t) FF * 4);
        const bf16_t* fc2_w = (const bf16_t*) wcopy(b.fc2_w, (size_t) D * FF * 2, true);
        const float* fc2_b = (const float*) wcopy(b.fc2_b, (size_t) D * 4);

        // Attention sub-block
        layernorm_bf16_kernel<<<(unsigned) N, 256, 0, cl.s>>>(d_x, n1_w, n1_b, D, g.eps, d_x16);
        cvt_if_f16(d_x16, (size_t) N * D);

        size_t aoff = 0;
        auto atake = [&](size_t bytes) -> float* {
            float* ptr = (float*) (d_work + aoff);
            aoff += align256(bytes);
            return ptr;
        };
        float* d_qkv = atake((size_t) N * 3 * D * 4);
        float* d_Q = atake((size_t) N * D * 4);
        float* d_K = atake((size_t) N * D * 4);
        float* d_V = atake((size_t) N * D * 4);
        float* d_S = atake((size_t) H * BQ_cap * N * 4);
        float* d_O = atake((size_t) H * BQ_cap * HD * 4);

        if (!gemm_16(d_x16, qkv_w, N, 3 * D, D, d_qkv)) return false;
        qkv_split_rope2d_kernel<<<dim3((unsigned) N, (unsigned) H), 128, 0, cl.s>>>(
            d_qkv, qkv_b, N, H, HD, mw, g.rope_theta, d_Q, d_K, d_V);

        // Tiled full bidirectional attention over query rows [q0 .. q0+BQ)
        for (int q0 = 0; q0 < N; q0 += BQ_cap) {
            const int BQ = std::min(BQ_cap, N - q0);
            const float beta0 = 0.0f, alpha1 = 1.0f;
            // S[h, BQ, N] = attn_scale * Q[h, q0..q0+BQ, HD] @ K[h, 0..N, HD]^T
            VB(cublasGemmStridedBatchedEx(
                cl.blas, CUBLAS_OP_T, CUBLAS_OP_N, N, BQ, HD, &attn_scale, d_K, CUDA_R_32F, HD, (long long) N * HD,
                d_Q + (size_t) q0 * HD, CUDA_R_32F, HD, (long long) N * HD, &beta0, d_S, CUDA_R_32F, N,
                (long long) BQ * N, H, attn_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            softmax_rows_kernel<<<(unsigned) (H * BQ), 256, 0, cl.s>>>(d_S, N);
            // O[h, BQ, HD] = S[h, BQ, N] @ V[h, 0..N, HD]
            VB(cublasGemmStridedBatchedEx(
                cl.blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, BQ, N, &alpha1, d_V, CUDA_R_32F, HD, (long long) N * HD,
                d_S, CUDA_R_32F, N, (long long) BQ * N, &beta0, d_O, CUDA_R_32F, HD, (long long) BQ * HD, H,
                attn_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            attn_out_to_bf16_kernel<<<(unsigned) BQ, 256, 0, cl.s>>>(d_O, H, BQ, HD, d_x16 + (size_t) q0 * D);
            cvt_if_f16(d_x16 + (size_t) q0 * D, (size_t) BQ * D);
        }
        // Output projection + residual (reuse d_qkv for output F32)
        if (!gemm_16(d_x16, proj_w, N, D, D, d_qkv)) return false;
        const size_t tot_nd = (size_t) N * D;
        add_bias_res_kernel<<<(unsigned) ((tot_nd + 255) / 256), 256, 0, cl.s>>>(d_x, d_qkv, proj_b, tot_nd, D);

        // MLP sub-block (tiled by kFfnChunk)
        layernorm_bf16_kernel<<<(unsigned) N, 256, 0, cl.s>>>(d_x, n2_w, n2_b, D, g.eps, d_x16);
        cvt_if_f16(d_x16, (size_t) N * D);
        size_t moff = 0;
        float* d_mid = (float*) (d_work + moff);
        moff += align256((size_t) kFfnChunk * FF * 4);
        bf16_t* d_mid16 = (bf16_t*) (d_work + moff);
        moff += align256((size_t) kFfnChunk * FF * 2);
        float* d_mlp_out = (float*) (d_work + moff);

        for (int r0 = 0; r0 < N; r0 += kFfnChunk) {
            const int nr = std::min(kFfnChunk, N - r0);
            if (!gemm_16(d_x16 + (size_t) r0 * D, fc1_w, nr, FF, D, d_mid)) return false;
            const size_t n_mid = (size_t) nr * FF;
            gelu_tanh_bias_bf16_kernel<<<(unsigned) ((n_mid + 255) / 256), 256, 0, cl.s>>>(d_mid, fc1_b, n_mid, FF,
                                                                                           d_mid16);
            cvt_if_f16(d_mid16, n_mid);
            if (!gemm_16(d_mid16, fc2_w, nr, D, FF, d_mlp_out)) return false;
            const size_t n_out = (size_t) nr * D;
            add_bias_res_kernel<<<(unsigned) ((n_out + 255) / 256), 256, 0, cl.s>>>(
                d_x + (size_t) r0 * D, d_mlp_out, fc2_b, n_out, D);
        }
    }

    // 3. Merger LayerNorm (before 2x2 concat) + linear_proj (4608 -> 2560 -> exact GELU -> 2560)
    {
        size_t woff = 0;
        auto wcopy = [&](const void* src, size_t bytes, bool is_w16 = false) -> uint8_t* {
            uint8_t* dst = d_wbuf + woff;
            woff += align256(bytes);
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, cl.s);
            if (is_w16) cvt_if_f16(dst, bytes / 2);
            return dst;
        };
        const float* mn_w = (const float*) wcopy(p_->m_norm_w, (size_t) D * 4);
        const float* mn_b = (const float*) wcopy(p_->m_norm_b, (size_t) D * 4);
        const bf16_t* l0_w = (const bf16_t*) wcopy(p_->lp0_w, (size_t) OD * (4 * D) * 2, true);
        const float* l0_b = (const float*) wcopy(p_->lp0_b, (size_t) OD * 4);
        const bf16_t* l2_w = (const bf16_t*) wcopy(p_->lp2_w, (size_t) OD * OD * 2, true);
        const float* l2_b = (const float*) wcopy(p_->lp2_b, (size_t) OD * 4);

        // LayerNorm(1152) on each of the N patches -> d_x16[N][1152], which is ALREADY [tok][4608] contiguous!
        layernorm_bf16_kernel<<<(unsigned) N, 256, 0, cl.s>>>(d_x, mn_w, mn_b, D, g.eps, d_x16);
        cvt_if_f16(d_x16, (size_t) N * D);

        size_t poff = 0;
        float* d_p0 = (float*) (d_work + poff);
        poff += align256((size_t) tok * OD * 4);
        bf16_t* d_p0_16 = (bf16_t*) (d_work + poff);
        poff += align256((size_t) tok * OD * 2);
        float* d_p2 = (float*) (d_work + poff);

        if (!gemm_16(d_x16, l0_w, tok, OD, 4 * D, d_p0)) return false;
        const size_t n_tok_od = (size_t) tok * OD;
        gelu_erf_bias_bf16_kernel<<<(unsigned) ((n_tok_od + 255) / 256), 256, 0, cl.s>>>(d_p0, l0_b, n_tok_od, OD,
                                                                                         d_p0_16);
        cvt_if_f16(d_p0_16, n_tok_od);
        if (!gemm_16(d_p0_16, l2_w, tok, OD, OD, d_p2)) return false;
        add_bias_inplace_kernel<<<(unsigned) ((n_tok_od + 255) / 256), 256, 0, cl.s>>>(d_p2, l2_b, n_tok_od, OD);

        out_rows.resize(n_tok_od);
        VK(cudaMemcpyAsync(out_rows.data(), d_p2, n_tok_od * sizeof(float), cudaMemcpyDeviceToHost, cl.s));
        VK(cudaStreamSynchronize(cl.s));
    }

    return true;
}

}  // namespace ling
