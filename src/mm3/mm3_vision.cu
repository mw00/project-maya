// maya-mm3: MiniMax-M3.1-Flash's vision tower (see mm3_vision.hpp)
#include "mm3_vision.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <sstream>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <cuda_bf16.h>
#endif
#include "../ds4/ds4_numerics.cuh"   // (BF16 on HIP: the same bits in plain arithmetic)

#include "mm3_env.hpp"

namespace mm3 {

#define VK(call)                                                                       \
    do {                                                                               \
        const cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                       \
            err = std::string("vision: ") + #call + ": " + cudaGetErrorString(e_);     \
            return false;                                                          \
        }                                                                              \
    } while (0)
#define VB(call)                                                                       \
    do {                                                                               \
        const cublasStatus_t b_ = (call);                                              \
        if (b_ != CUBLAS_STATUS_SUCCESS) {                                             \
            err = std::string("vision: ") + #call + ": cublas " + std::to_string((int) b_); \
            return false;                                                          \
        }                                                                              \
    } while (0)

namespace {

// the GEMMs' BF16 inputs: CUDA's type, or on AMD (HIP) its 16 bits as they are (hip_bf16.h's own type clashes with
// the hip_compat warp functions; hipBLAS reads HIP_R_16BF from the same bits), rounded to nearest even either way
#if defined(STRATA_USE_HIP)
using bf16_t = uint16_t;
__device__ __forceinline__ bf16_t f2bf(float f) { return strata::ds4::bf16_rn(f); }
#else
using bf16_t = __nv_bfloat16;
__device__ __forceinline__ bf16_t f2bf(float f) { return __float2bfloat16(f); }
#endif

constexpr int kHeadChunk = 4;   // the heads whose scores are held at once ([4][2304][2304] F32: 85 MB)

__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float s = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) s += red[i];
    return s;
}

// one block a row: y16 = LayerNorm(x) (biased variance) . w + b, BF16
__global__ void layernorm_kernel(const float* x, const float* w, const float* b, int D, float eps,
                                 bf16_t* y16) {
    __shared__ float red[32];
    const float* r = x + (size_t) blockIdx.x * D;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += r[i];
    const float mean = block_sum(s, red) / (float) D;
    float q = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float d = r[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(block_sum(q, red) / (float) D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        y16[(size_t) blockIdx.x * D + i] = f2bf((r[i] - mean) * inv * w[i] + b[i]);
}

// x [rows][D] in place: LayerNorm (the pre-layernorm of the residual stream itself), F32
__global__ void layernorm_f32_kernel(float* x, const float* w, const float* b, int D, float eps) {
    __shared__ float red[32];
    float* r = x + (size_t) blockIdx.x * D;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += r[i];
    const float mean = block_sum(s, red) / (float) D;
    float q = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float d = r[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(block_sum(q, red) / (float) D + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += blockDim.x) r[i] = (r[i] - mean) * inv * w[i] + b[i];
}

// y [n] (rows of M) += bias; then GELU (act); into F32 (yf, may be y) and / or BF16 (y16)
__global__ void bias_act_kernel(const float* y, const float* bias, size_t n, int M, int act, float* yf,
                                bf16_t* y16) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        float v = y[i] + bias[i % (size_t) M];
        if (act) v = gelu(v);
        if (yf) yf[i] = v;
        if (y16) y16[i] = f2bf(v);
    }
}

// h [n] += t + bias (rows of M)
__global__ void residual_kernel(float* h, const float* t, const float* bias, size_t n, int M) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        h[i] += t[i] + bias[i % (size_t) M];
}

__global__ void to_bf16_kernel(const float* x, size_t n, bf16_t* y) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        y[i] = f2bf(x[i]);
}

// qkv [N][3 dim] += bias, then the 3D RoPE on q and k: per head, dims [0, rot / 2) and [rot / 2, rot) a rotate-half
// pair, dim i's angle pos[i / nf] * inv[i % nf] (axes t, h, w; nf frequencies each), dims past rot unrotated
__global__ void qkv_rope_kernel(float* qkv, const float* bias, const int* pos, const float* inv, int dim, int heads,
                                int hd, int nf) {
    const int t = blockIdx.x;
    float* r = qkv + (size_t) t * 3 * dim;
    for (int i = threadIdx.x; i < 3 * dim; i += blockDim.x) r[i] += bias[i];
    __syncthreads();
    const int half = 3 * nf;   // rot / 2
    for (int j = threadIdx.x; j < 2 * heads * half; j += blockDim.x) {
        const int which = j / (heads * half), hh = (j / half) % heads, i = j % half;
        float* v = r + which * dim + hh * hd;
        const float a = (float) pos[t * 3 + i / nf] * inv[i % nf];
        float sn, cs;
        sincosf(a, &sn, &cs);
        const float x1 = v[i], x2 = v[i + half];
        v[i] = x1 * cs - x2 * sn;
        v[i + half] = x2 * cs + x1 * sn;
    }
}

// one block a row of n: softmax in place
__global__ void softmax_kernel(float* s, int n) {
    __shared__ float red[32];
    float* r = s + (size_t) blockIdx.x * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, r[i]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __shared__ float mw[32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) mw[w] = m;
    __syncthreads();
    m = -INFINITY;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) m = fmaxf(m, mw[i]);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(r[i] - m);
        r[i] = e;
        sum += e;
    }
    const float inv = 1.0f / block_sum(sum, red);
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] *= inv;
}

int blocks_for(size_t n) { return (int) std::min<size_t>((n + 255) / 256, 65535); }

}  // namespace

struct Vision::Impl {
    VisionGeo g;
    int dev = -1;
    cudaStream_t s = nullptr, cs = nullptr;   // the encode's stream, the weights' copy stream
    cublasHandle_t blas = nullptr;
    cudaEvent_t ev_up[2] = {nullptr, nullptr};     // a slot's stage copied in (its bounce buffer free again)
    cudaEvent_t ev_free[2] = {nullptr, nullptr};   // a slot's stage used (the slot free for the next one)
    // The weights stay in host memory between pictures - the BF16 matrices where the pack maps them (the page cache
    // while the RAM holds it, else the SSD), the F32 vectors converted here - and an encode copies them to the GPU a
    // stage at a time (a layer, or a projector matrix's rows) into two slots in turn, the next one's copy under this
    // one's work.  The GPU memory of a picture (the slots, its scratch, cuBLAS's workspace) is taken for that encode
    // and given back after it: freed, or the lender's room (its staging slots) handed back.
    struct Piece {
        const uint8_t* src;
        size_t bytes, off;
    };
    struct Stage {
        std::vector<Piece> pc;
        size_t bytes = 0;
        int mat = -1, n0 = 0, nc = 0;   // (a projector matrix's rows [n0, n0 + nc); -1: a layer)
    };
    struct LOff {
        size_t ln1w, ln1b, ln2w, ln2b, wqkv, bqkv, wo, bo, w1, b1, w2, b2;
    };
    std::vector<Stage> st;   // the layers' stages, then the projector's
    std::vector<LOff> lo;
    std::deque<std::vector<float>> hostf;
    std::vector<uint8_t> pre_h;   // the patch embedding, the pre-layernorm, the RoPE's frequencies, the projector's biases
    size_t o_wpatch = 0, o_prew = 0, o_preb = 0, o_inv = 0, o_pb1 = 0, o_pb2 = 0, o_pmb1 = 0, o_pmb2 = 0;
    int mat_n[4] = {}, mat_k[4] = {};   // pl1, pl2, pm1, pm2: BF16 [N][K]
    size_t slot = 0;                    // a stage's room
    uint8_t* bounce[2] = {nullptr, nullptr};   // pinned: a stage gathered for its copy
    std::vector<uint8_t> bounce_v[2];          // (pageable, when no pinned memory was had)
    bool pinned = false;
    size_t host_b = 0, cur_b = 0, last_b = 0;
    std::function<void*(size_t)> lend;
    std::function<void()> give_back;
    bool lent_once = false;
    static constexpr size_t kWs = (size_t) 32 << 20;   // cuBLAS's workspace (its default's largest)

    static size_t al(size_t n) { return (n + 255) & ~(size_t) 255; }

    ~Impl() {
        if (dev >= 0) cudaSetDevice(dev);
        if (s) cudaStreamSynchronize(s);
        if (cs) cudaStreamSynchronize(cs);
        for (int b = 0; b < 2; ++b) {
            if (ev_up[b]) cudaEventDestroy(ev_up[b]);
            if (ev_free[b]) cudaEventDestroy(ev_free[b]);
            if (pinned && bounce[b]) cudaFreeHost(bounce[b]);
        }
        if (blas) cublasDestroy(blas);
        if (s) cudaStreamDestroy(s);
        if (cs) cudaStreamDestroy(cs);
    }

    const Tensor* need(const Pack& pk, const std::string& n, size_t elems, std::string& err) const {
        const Tensor* t = pk.get(n);
        if (!t) {
            err = "vision: " + n + " missing";
            return nullptr;
        }
        const size_t es = t->dtype == "F32" ? 4 : 2;
        if ((t->dtype != "F32" && t->dtype != "BF16") || t->bytes != elems * es) {
            err = "vision: " + n + " is not " + std::to_string(elems) + " F32 / BF16 values";
            return nullptr;
        }
        return t;
    }

    // a tensor as F32 in host memory (kept here)
    const uint8_t* f32(const Pack& pk, const std::string& n, size_t elems, std::string& err) {
        const Tensor* t = need(pk, n, elems, err);
        if (!t) return nullptr;
        hostf.emplace_back(elems);
        std::vector<float>& hv = hostf.back();
        const uint8_t* d = pk.data(*t);
        for (size_t i = 0; i < elems; ++i) {
            if (t->dtype == "F32") {
                std::memcpy(&hv[i], d + 4 * i, 4);
            } else {
                uint16_t b;
                std::memcpy(&b, d + 2 * i, 2);
                const uint32_t u = (uint32_t) b << 16;
                std::memcpy(&hv[i], &u, 4);
            }
        }
        host_b += elems * 4;
        return (const uint8_t*) hv.data();
    }

    // a BF16 tensor where the pack maps it
    const uint8_t* bf16(const Pack& pk, const std::string& n, size_t elems, std::string& err) {
        const Tensor* t = need(pk, n, elems, err);
        if (!t) return nullptr;
        if (t->dtype != "BF16") {
            err = "vision: " + n + " is not BF16";
            return nullptr;
        }
        host_b += elems * 2;
        return pk.data(*t);
    }

    // a piece into a stage: its offset (aligned, or right after the one before - wqkv's k and v, bqkv's)
    static size_t add(Stage& S, const uint8_t* src, size_t bytes, bool align = true) {
        if (align) S.bytes = al(S.bytes);
        const size_t off = S.bytes;
        S.pc.push_back({src, bytes, off});
        S.bytes += bytes;
        return off;
    }

    // C [rows][ldc] F32 (columns [0, N) of it) = X [rows][K] BF16 . W [N][K]^T BF16
    bool gemm(const uint8_t* W, int N, int K, const bf16_t* X, int rows, float* C, int ldc, std::string& err) {
        const float one = 1.0f, zero = 0.0f;
        VB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero,
                        C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        return true;
    }

    bool load(const Pack& pk, int dev_, std::string& err) {
        dev = dev_;
        VK(cudaSetDevice(dev));
        g.dim = (int) pk.num("vision.hidden_size", 1280);
        g.heads = (int) pk.num("vision.num_attention_heads", 16);
        g.hd = g.dim / g.heads;
        g.ff = (int) pk.num("vision.intermediate_size", 5120);
        g.layers = (int) pk.num("vision.num_hidden_layers", 32);
        g.patch = (int) pk.num("vision.patch_size", 14);
        g.tps = (int) pk.num("vision.temporal_patch_size", 2);
        g.merge = (int) pk.num("vision.spatial_merge_size", 2);
        g.proj = (int) pk.num("vision.projector_hidden_size", 6144);
        g.out_dim = (int) pk.num("hidden_size", 6144);
        g.eps = (float) pk.num("vision.layer_norm_eps", 1e-5);
        g.theta = (float) pk.num("vision.rope_theta", 10000.0);
        g.max_pixels = (int) pk.num("vision.max_pixels", 672 * 672);
        g.min_pixels = (int) pk.num("vision.min_pixels", 4 * 28 * 28);
        g.image_token = (int) pk.num("vision.image_token", -1);
        g.image_start = (int) pk.num("vision.image_start", -1);
        g.image_end = (int) pk.num("vision.image_end", -1);
        for (const char* k : {"vision.image_mean", "vision.image_std"}) {
            std::stringstream ss(pk.str(k));
            float* dst = std::strcmp(k, "vision.image_mean") == 0 ? g.mean : g.stdv;
            std::string x;
            for (int i = 0; i < 3 && std::getline(ss, x, ','); ++i) dst[i] = std::strtof(x.c_str(), nullptr);
        }
        if (g.dim % g.heads || g.hd < 6 || g.image_token < 0 || g.image_start < 0 || g.image_end < 0 || g.tps != 2 ||
            g.merge != 2) {
            err = "vision: an unexpected geometry in mm3.txt";
            return false;
        }
        // (streams, events and the cuBLAS handle: no room of the GPU's to speak of - its workspace is set per encode)
        VK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        VK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            VK(cudaEventCreateWithFlags(&ev_up[b], cudaEventDisableTiming));
            VK(cudaEventCreateWithFlags(&ev_free[b], cudaEventDisableTiming));
        }
        VB(cublasCreate(&blas));
        VB(cublasSetStream(blas, s));
        const std::string V = "vision_tower.vision_model.";
        const int D = g.dim, F = g.ff, P = 3 * g.tps * g.patch * g.patch;
        const int Pj = g.proj, O = g.out_dim, M = g.merge * g.merge;
        // the fixed part (uploaded first): F32 vectors one after another
        auto pre = [&](const uint8_t* src, size_t n, size_t& off) {
            if (!src) return false;
            off = al(pre_h.size());
            pre_h.resize(off + n * 4);
            std::memcpy(pre_h.data() + off, src, n * 4);
            return true;
        };
        const int nf = ((g.hd / 3) / 2);   // (the RoPE's frequencies: 2 nf dims an axis, theta^(-2j / that))
        std::vector<float> iv((size_t) nf);
        for (int j = 0; j < nf; ++j) iv[(size_t) j] = 1.0f / std::pow(g.theta, (float) (2 * j) / (float) (2 * nf));
        if (!pre(f32(pk, V + "embeddings.patch_embedding.weight", (size_t) D * P, err), (size_t) D * P, o_wpatch) ||
            !pre(f32(pk, V + "pre_layrnorm.weight", D, err), D, o_prew) ||
            !pre(f32(pk, V + "pre_layrnorm.bias", D, err), D, o_preb) ||
            !pre((const uint8_t*) iv.data(), (size_t) nf, o_inv) ||
            !pre(f32(pk, "multi_modal_projector.linear_1.bias", Pj, err), Pj, o_pb1) ||
            !pre(f32(pk, "multi_modal_projector.linear_2.bias", O, err), O, o_pb2) ||
            !pre(f32(pk, "patch_merge_mlp.linear_1.bias", Pj, err), Pj, o_pmb1) ||
            !pre(f32(pk, "patch_merge_mlp.linear_2.bias", O, err), O, o_pmb2))
            return false;
        // a layer a stage
        const size_t DD = (size_t) D * D;
        for (int i = 0; i < g.layers; ++i) {
            const std::string N = V + "encoder.layers." + std::to_string(i) + ".";
            Stage S;
            LOff f{};
            const uint8_t *a[16] = {};
            int k = 0;
            for (const char* n : {"layer_norm1.weight", "layer_norm1.bias", "layer_norm2.weight", "layer_norm2.bias"})
                if (!(a[k++] = f32(pk, N + n, D, err))) return false;
            if (!(a[4] = bf16(pk, N + "self_attn.q_proj.weight", DD, err)) ||
                !(a[5] = bf16(pk, N + "self_attn.k_proj.weight", DD, err)) ||
                !(a[6] = bf16(pk, N + "self_attn.v_proj.weight", DD, err)) ||
                !(a[7] = f32(pk, N + "self_attn.q_proj.bias", D, err)) ||
                !(a[8] = f32(pk, N + "self_attn.k_proj.bias", D, err)) ||
                !(a[9] = f32(pk, N + "self_attn.v_proj.bias", D, err)) ||
                !(a[10] = bf16(pk, N + "self_attn.out_proj.weight", DD, err)) ||
                !(a[11] = f32(pk, N + "self_attn.out_proj.bias", D, err)) ||
                !(a[12] = bf16(pk, N + "mlp.fc1.weight", (size_t) F * D, err)) ||
                !(a[13] = f32(pk, N + "mlp.fc1.bias", F, err)) ||
                !(a[14] = bf16(pk, N + "mlp.fc2.weight", (size_t) D * F, err)) ||
                !(a[15] = f32(pk, N + "mlp.fc2.bias", D, err)))
                return false;
            f.ln1w = add(S, a[0], D * 4);
            f.ln1b = add(S, a[1], D * 4);
            f.ln2w = add(S, a[2], D * 4);
            f.ln2b = add(S, a[3], D * 4);
            f.wqkv = add(S, a[4], DD * 2);
            add(S, a[5], DD * 2, false);
            add(S, a[6], DD * 2, false);
            f.bqkv = add(S, a[7], D * 4);
            add(S, a[8], D * 4, false);
            add(S, a[9], D * 4, false);
            f.wo = add(S, a[10], DD * 2);
            f.bo = add(S, a[11], D * 4);
            f.w1 = add(S, a[12], (size_t) F * D * 2);
            f.b1 = add(S, a[13], (size_t) F * 4);
            f.w2 = add(S, a[14], (size_t) D * F * 2);
            f.b2 = add(S, a[15], D * 4);
            slot = std::max(slot, al(S.bytes));
            st.push_back(std::move(S));
            lo.push_back(f);
        }
        // the projector's matrices in stages of rows that fit a slot
        const char* mn[4] = {"multi_modal_projector.linear_1.weight", "multi_modal_projector.linear_2.weight",
                             "patch_merge_mlp.linear_1.weight", "patch_merge_mlp.linear_2.weight"};
        const int nn[4] = {Pj, O, Pj, O}, kk[4] = {D, Pj, O * M, Pj};
        for (int m = 0; m < 4; ++m) {
            const uint8_t* src = bf16(pk, mn[m], (size_t) nn[m] * kk[m], err);
            if (!src) return false;
            mat_n[m] = nn[m];
            mat_k[m] = kk[m];
            const size_t row = (size_t) kk[m] * 2;
            if (m == 0) slot = std::max(slot, al((size_t) nn[m] * row));
            const int rc = (int) std::max<size_t>(1, std::min<size_t>((size_t) nn[m], slot / row));
            for (int n0 = 0; n0 < nn[m]; n0 += rc) {
                Stage S;
                S.mat = m;
                S.n0 = n0;
                S.nc = std::min(rc, nn[m] - n0);
                add(S, src + (size_t) n0 * row, (size_t) S.nc * row);
                st.push_back(std::move(S));
            }
        }
        for (int b = 0; b < 2; ++b) {
            if (cudaHostAlloc((void**) &bounce[b], slot, cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                bounce[b] = nullptr;
            }
        }
        pinned = bounce[0] && bounce[1];
        if (!pinned) {
            for (int b = 0; b < 2; ++b) {
                if (bounce[b]) cudaFreeHost(bounce[b]);
                bounce_v[b].resize(slot);
                bounce[b] = bounce_v[b].data();
            }
        }
        return true;
    }

    int tokens(int w, int hgt) const {
        const int f = g.factor();
        if (w <= 0 || hgt <= 0 || w % f || hgt % f || (long long) w * hgt > g.max_pixels) return 0;
        return (hgt / f) * (w / f);
    }

    // a picture of N patches: its room of the GPU, carved
    struct Lay {
        size_t pre, s0, s1, px, h, qkv, sc, att, big, out, x16, pos, ws, total;
    };
    Lay layout(size_t N) const {
        const size_t D = (size_t) g.dim, P = (size_t) 3 * g.tps * g.patch * g.patch;
        const size_t W = (size_t) std::max(std::max(g.ff, g.proj), g.out_dim), NB = N / (size_t) (g.merge * g.merge);
        Lay y{};
        size_t off = 0;
        auto take = [&](size_t b) {
            const size_t o = off;
            off += al(b);
            return o;
        };
        y.pre = take(pre_h.size());
        y.s0 = take(slot);
        y.s1 = take(slot);
        y.px = take(N * P * 4);
        y.h = take(N * D * 4);
        y.qkv = take(N * 3 * D * 4);
        y.sc = take((size_t) kHeadChunk * N * N * 4);
        y.att = take(N * D * 4);
        y.big = take(N * W * 4);
        y.out = take(NB * (size_t) g.out_dim * 4);
        y.x16 = take(N * W * 2);
        y.pos = take(N * 3 * 4);
        y.ws = take(kWs);
        y.total = off;
        return y;
    }

    bool encode(const uint8_t* rgb, int w, int hgt, std::vector<float>& rows, std::string& err) {
        const int n_tok = tokens(w, hgt);
        if (n_tok == 0) {
            err = "vision: a picture of " + std::to_string(w) + " x " + std::to_string(hgt) +
                  " pixels (sides multiples of " + std::to_string(g.factor()) + ", at most " +
                  std::to_string(g.max_pixels) + " pixels)";
            return false;
        }
        VK(cudaSetDevice(dev));
        const int ps = g.patch, gh = hgt / ps, gw = w / ps, N = gh * gw, D = g.dim, F = g.ff, M = g.merge;
        const int P = 3 * g.tps * ps * ps;
        // the patches, image_processor.py's order: merge blocks row-major, the 2 x 2 patches of a block row-major;
        // a patch [channel][frame][y][x], its frames the same picture
        std::vector<float> hp((size_t) N * P);
        std::vector<int> hpos((size_t) N * 3);
        for (int p = 0; p < N; ++p) {
            const int b = p / (M * M), in = p % (M * M), bh = b / (gw / M), bw = b % (gw / M);
            const int py = bh * M + in / M, pxx = bw * M + in % M;
            hpos[(size_t) p * 3 + 0] = 0;
            hpos[(size_t) p * 3 + 1] = py;
            hpos[(size_t) p * 3 + 2] = pxx;
            float* dst = hp.data() + (size_t) p * P;
            for (int c = 0; c < 3; ++c)
                for (int t = 0; t < g.tps; ++t)
                    for (int y = 0; y < ps; ++y)
                        for (int x = 0; x < ps; ++x) {
                            const int Y = py * ps + y, X = pxx * ps + x;
                            const float v = (float) rgb[((size_t) Y * w + X) * 3 + c] * (1.0f / 255.0f);
                            dst[((c * g.tps + t) * ps + y) * ps + x] = (v - g.mean[c]) / g.stdv[c];
                        }
        }
        // the GPU's room for this picture: its own, else the lender's (the engine's staging slots, handed back after)
        const Lay y = layout((size_t) N);
        uint8_t* base = nullptr;
        const char* fl = env("MM3_VISION_LEND");   // (1: the lender's room even when there is free memory - a test)
        bool own = !(lend && fl && std::atoi(fl) == 1) && cudaMalloc((void**) &base, y.total) == cudaSuccess;
        if (!own) {
            cudaGetLastError();
            base = lend ? (uint8_t*) lend(y.total) : nullptr;
            if (!base) {
                err = "vision: " + std::to_string(y.total >> 20) + " MB of CUDA" + std::to_string(dev) +
                      " for the picture: not free" + (lend ? ", and more than the engine's staging slots" : "");
                return false;
            }
            if (!lent_once)
                std::fprintf(stderr, "maya-mm3: NOTE: vision: CUDA%d had not %zu MB free for the picture - the "
                             "experts' staging slots are lent to it meanwhile (their experts copied in again after)\n",
                             dev, y.total >> 20);
            lent_once = true;
        }
        cur_b = y.total;
        last_b = y.total;
        const bool ok = run(base, y, hp, hpos, N, D, F, P, rows, err);
        cudaStreamSynchronize(s);
        cudaStreamSynchronize(cs);
        if (own) cudaFree(base);
        else give_back();
        cur_b = 0;
        if (!ok) cudaGetLastError();
        return ok;
    }

    bool run(uint8_t* base, const Lay& y, const std::vector<float>& hp, const std::vector<int>& hpos, int N, int D,
             int F, int P, std::vector<float>& rows, std::string& err) {
        auto fp = [&](size_t o) { return (float*) (base + o); };
        float *px = fp(y.px), *h = fp(y.h), *qkv = fp(y.qkv), *sc = fp(y.sc), *att = fp(y.att), *big = fp(y.big);
        float* out = fp(y.out);
        bf16_t* x16 = (bf16_t*) (base + y.x16);
        int* pos = (int*) (base + y.pos);
        float* pre = fp(y.pre);
        auto pv = [&](size_t o) { return (float*) ((uint8_t*) pre + o); };
        VB(cublasSetWorkspace(blas, base + y.ws, kWs));
        VK(cudaMemcpyAsync(pre, pre_h.data(), pre_h.size(), cudaMemcpyHostToDevice, s));
        VK(cudaMemcpyAsync(px, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice, s));
        VK(cudaMemcpyAsync(pos, hpos.data(), hpos.size() * 4, cudaMemcpyHostToDevice, s));
        // the stages through the two slots
        uint8_t* sl[2] = {base + y.s0, base + y.s1};
        for (int b = 0; b < 2; ++b) {
            VK(cudaEventRecord(ev_free[b], s));
            VK(cudaEventRecord(ev_up[b], cs));
        }
        const int ns = (int) st.size();
        auto upload = [&](int k) -> bool {
            const int b = k & 1;
            const Stage& S = st[(size_t) k];
            VK(cudaEventSynchronize(ev_up[b]));
            for (const Piece& q : S.pc) std::memcpy(bounce[b] + q.off, q.src, q.bytes);
            VK(cudaStreamWaitEvent(cs, ev_free[b], 0));
            VK(cudaMemcpyAsync(sl[b], bounce[b], S.bytes, cudaMemcpyHostToDevice, cs));
            VK(cudaEventRecord(ev_up[b], cs));
            return true;
        };
        auto acquire = [&](int k, uint8_t*& at) -> bool {
            VK(cudaStreamWaitEvent(s, ev_up[k & 1], 0));
            at = sl[k & 1];
            return true;
        };
        auto done = [&](int k) -> bool {
            VK(cudaEventRecord(ev_free[k & 1], s));
            return k + 2 < ns ? upload(k + 2) : true;
        };
        if (!upload(0) || (ns > 1 && !upload(1))) return false;
        const float one = 1.0f, zero = 0.0f;
        // the patch embedding (F32) and the pre-layernorm: the residual stream
        VB(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, D, N, P, &one, pv(o_wpatch), P, px, P, &zero, h, D));
        layernorm_f32_kernel<<<N, 256, 0, s>>>(h, pv(o_prew), pv(o_preb), D, g.eps);
        const size_t ND = (size_t) N * D;
        const float scale = 1.0f / std::sqrt((float) g.hd);
        const int nf = (g.hd / 3) / 2;
        for (int li = 0; li < g.layers; ++li) {
            uint8_t* sb = nullptr;
            if (!acquire(li, sb)) return false;
            const LOff& l = lo[(size_t) li];
            auto lf = [&](size_t o) { return (float*) (sb + o); };
            layernorm_kernel<<<N, 256, 0, s>>>(h, lf(l.ln1w), lf(l.ln1b), D, g.eps, x16);
            if (!gemm(sb + l.wqkv, 3 * D, D, x16, N, qkv, 3 * D, err)) return false;
            qkv_rope_kernel<<<N, 256, 0, s>>>(qkv, lf(l.bqkv), pos, pv(o_inv), D, g.heads, g.hd, nf);
            // attention, kHeadChunk heads at a time: S = scale Q K^T (q, k, v read in place from qkv), softmax, S V
            for (int h0 = 0; h0 < g.heads; h0 += kHeadChunk) {
                const int nh = std::min(kHeadChunk, g.heads - h0);
                VB(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, N, g.hd, &scale, qkv + D + h0 * g.hd,
                                             3 * D, g.hd, qkv + h0 * g.hd, 3 * D, g.hd, &zero, sc, N, (long long) N * N,
                                             nh));
                softmax_kernel<<<N * nh, 256, 0, s>>>(sc, N);
                VB(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, g.hd, N, N, &one,
                                             qkv + 2 * D + h0 * g.hd, 3 * D, g.hd, sc, N, (long long) N * N, &zero,
                                             att + h0 * g.hd, D, g.hd, nh));
            }
            to_bf16_kernel<<<blocks_for(ND), 256, 0, s>>>(att, ND, x16);
            if (!gemm(sb + l.wo, D, D, x16, N, big, D, err)) return false;
            residual_kernel<<<blocks_for(ND), 256, 0, s>>>(h, big, lf(l.bo), ND, D);
            layernorm_kernel<<<N, 256, 0, s>>>(h, lf(l.ln2w), lf(l.ln2b), D, g.eps, x16);
            if (!gemm(sb + l.w1, F, D, x16, N, big, F, err)) return false;
            bias_act_kernel<<<blocks_for((size_t) N * F), 256, 0, s>>>(big, lf(l.b1), (size_t) N * F, F, 1, nullptr, x16);
            if (!gemm(sb + l.w2, D, F, x16, N, att, D, err)) return false;
            residual_kernel<<<blocks_for(ND), 256, 0, s>>>(h, att, lf(l.b2), ND, D);
            if (!done(li)) return false;
        }
        // the projector: per patch Pj -> out_dim, then each block's M x M rows (consecutive) as one - each matrix's
        // stages of rows into their columns of C
        int k = g.layers;
        auto mat = [&](int m, const bf16_t* X, int rws, float* C) -> bool {
            for (; k < ns && st[(size_t) k].mat == m; ++k) {
                uint8_t* sb = nullptr;
                if (!acquire(k, sb) ||
                    !gemm(sb, st[(size_t) k].nc, mat_k[m], X, rws, C + st[(size_t) k].n0, mat_n[m], err) || !done(k))
                    return false;
            }
            return true;
        };
        const int Pj = g.proj, O = g.out_dim, NB = N / (g.merge * g.merge);
        to_bf16_kernel<<<blocks_for(ND), 256, 0, s>>>(h, ND, x16);
        if (!mat(0, x16, N, big)) return false;
        bias_act_kernel<<<blocks_for((size_t) N * Pj), 256, 0, s>>>(big, pv(o_pb1), (size_t) N * Pj, Pj, 1, nullptr, x16);
        if (!mat(1, x16, N, big)) return false;
        bias_act_kernel<<<blocks_for((size_t) N * O), 256, 0, s>>>(big, pv(o_pb2), (size_t) N * O, O, 0, nullptr, x16);
        bf16_t* m16 = (bf16_t*) qkv;   // (free now: [NB][Pj] BF16)
        if (!mat(2, x16, NB, big)) return false;
        bias_act_kernel<<<blocks_for((size_t) NB * Pj), 256, 0, s>>>(big, pv(o_pmb1), (size_t) NB * Pj, Pj, 1, nullptr,
                                                                   m16);
        if (!mat(3, m16, NB, out)) return false;
        bias_act_kernel<<<blocks_for((size_t) NB * O), 256, 0, s>>>(out, pv(o_pmb2), (size_t) NB * O, O, 0, out, nullptr);
        rows.resize((size_t) NB * O);
        VK(cudaMemcpyAsync(rows.data(), out, rows.size() * 4, cudaMemcpyDeviceToHost, s));
        VK(cudaStreamSynchronize(s));
        VK(cudaGetLastError());
        return true;
    }
};

Vision::Vision() : p(new Impl) {}
Vision::~Vision() = default;

bool Vision::in_pack(const Pack& pk) {
    return pk.has("vision.hidden_size") && pk.get("vision_tower.vision_model.pre_layrnorm.weight") != nullptr;
}
bool Vision::load(const Pack& pk, int dev, std::string& err) { return p->load(pk, dev, err); }
const VisionGeo& Vision::geo() const { return p->g; }
int Vision::tokens(int w, int h) const { return p->tokens(w, h); }
bool Vision::encode(const uint8_t* rgb, int w, int h, std::vector<float>& rows, std::string& err) {
    return p->encode(rgb, w, h, rows, err);
}
size_t Vision::bytes() const { return p->cur_b; }
size_t Vision::host_bytes() const { return p->host_b; }
size_t Vision::room_bytes() const { return p->layout((size_t) p->g.max_patches()).total; }
size_t Vision::last_bytes() const { return p->last_b; }
void Vision::set_lender(std::function<void*(size_t)> lend, std::function<void()> give_back) {
    p->lend = std::move(lend);
    p->give_back = std::move(give_back);
}
int Vision::device() const { return p->dev; }

}  // namespace mm3
