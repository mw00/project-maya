// maya-mm3: MiniMax-M3.1-Flash's vision tower (see mm3_vision.hpp)
#include "mm3_vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
                                 __nv_bfloat16* y16) {
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
        y16[(size_t) blockIdx.x * D + i] = __float2bfloat16((r[i] - mean) * inv * w[i] + b[i]);
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
                                __nv_bfloat16* y16) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        float v = y[i] + bias[i % (size_t) M];
        if (act) v = gelu(v);
        if (yf) yf[i] = v;
        if (y16) y16[i] = __float2bfloat16(v);
    }
}

// h [n] += t + bias (rows of M)
__global__ void residual_kernel(float* h, const float* t, const float* bias, size_t n, int M) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        h[i] += t[i] + bias[i % (size_t) M];
}

__global__ void to_bf16_kernel(const float* x, size_t n, __nv_bfloat16* y) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        y[i] = __float2bfloat16(x[i]);
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
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    std::vector<void*> owned;
    size_t nbytes = 0;
    struct Layer {
        float *ln1w, *ln1b, *ln2w, *ln2b, *bqkv, *bo, *b1, *b2;
        __nv_bfloat16 *wqkv, *wo, *w1, *w2;
    };
    std::vector<Layer> L;
    float *wpatch = nullptr, *prew = nullptr, *preb = nullptr;
    __nv_bfloat16 *pl1 = nullptr, *pl2 = nullptr, *pm1 = nullptr, *pm2 = nullptr;
    float *pb1 = nullptr, *pb2 = nullptr, *pmb1 = nullptr, *pmb2 = nullptr;
    float* inv = nullptr;
    // scratch, for max_patches rows
    float *px = nullptr, *h = nullptr, *qkv = nullptr, *sc = nullptr, *att = nullptr, *big = nullptr, *out = nullptr;
    __nv_bfloat16* x16 = nullptr;
    int* pos = nullptr;

    ~Impl() {
        if (dev >= 0) cudaSetDevice(dev);
        if (s) cudaStreamSynchronize(s);
        for (void* p : owned) cudaFree(p);
        if (blas) cublasDestroy(blas);
        if (s) cudaStreamDestroy(s);
    }

    template <class T>
    bool alloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaMalloc(&v, std::max<size_t>(n * sizeof(T), 16)) != cudaSuccess) {
            err = "vision: out of device memory on CUDA" + std::to_string(dev) + " (" +
                  std::to_string((n * sizeof(T)) >> 20) + " MB)";
            return false;
        }
        owned.push_back(v);
        nbytes += n * sizeof(T);
        p = (T*) v;
        return true;
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

    // a tensor as F32 on the device
    bool f32(const Pack& pk, const std::string& n, size_t elems, float*& out, std::string& err) {
        const Tensor* t = need(pk, n, elems, err);
        if (!t || !alloc(out, elems, err)) return false;
        std::vector<float> hv(elems);
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
        VK(cudaMemcpy(out, hv.data(), elems * 4, cudaMemcpyHostToDevice));
        return true;
    }

    // BF16 tensors (as stored) one after another into dst
    bool bf16_into(const Pack& pk, const std::string& n, size_t elems, __nv_bfloat16* dst, std::string& err) {
        const Tensor* t = need(pk, n, elems, err);
        if (!t) return false;
        if (t->dtype != "BF16") {
            err = "vision: " + n + " is not BF16";
            return false;
        }
        VK(cudaMemcpy(dst, pk.data(*t), elems * 2, cudaMemcpyHostToDevice));
        return true;
    }

    bool bf16(const Pack& pk, const std::string& n, size_t elems, __nv_bfloat16*& out, std::string& err) {
        return alloc(out, elems, err) && bf16_into(pk, n, elems, out, err);
    }

    // C [rows][N] F32 = X [rows][K] BF16 . W [N][K]^T BF16
    bool gemm(const __nv_bfloat16* W, int N, int K, const __nv_bfloat16* X, int rows, float* C, std::string& err) {
        const float one = 1.0f, zero = 0.0f;
        VB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero,
                        C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
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
        VK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        VB(cublasCreate(&blas));
        VB(cublasSetStream(blas, s));
        const std::string V = "vision_tower.vision_model.";
        const int D = g.dim, F = g.ff, P = 3 * g.tps * g.patch * g.patch;
        if (!f32(pk, V + "embeddings.patch_embedding.weight", (size_t) D * P, wpatch, err) ||
            !f32(pk, V + "pre_layrnorm.weight", D, prew, err) || !f32(pk, V + "pre_layrnorm.bias", D, preb, err))
            return false;
        L.resize((size_t) g.layers);
        for (int i = 0; i < g.layers; ++i) {
            Layer& l = L[(size_t) i];
            const std::string N = V + "encoder.layers." + std::to_string(i) + ".";
            const size_t DD = (size_t) D * D;
            if (!f32(pk, N + "layer_norm1.weight", D, l.ln1w, err) || !f32(pk, N + "layer_norm1.bias", D, l.ln1b, err) ||
                !f32(pk, N + "layer_norm2.weight", D, l.ln2w, err) || !f32(pk, N + "layer_norm2.bias", D, l.ln2b, err) ||
                !alloc(l.wqkv, 3 * DD, err) || !bf16_into(pk, N + "self_attn.q_proj.weight", DD, l.wqkv, err) ||
                !bf16_into(pk, N + "self_attn.k_proj.weight", DD, l.wqkv + DD, err) ||
                !bf16_into(pk, N + "self_attn.v_proj.weight", DD, l.wqkv + 2 * DD, err) ||
                !alloc(l.bqkv, 3 * (size_t) D, err) || !bf16(pk, N + "self_attn.out_proj.weight", DD, l.wo, err) ||
                !f32(pk, N + "self_attn.out_proj.bias", D, l.bo, err) ||
                !bf16(pk, N + "mlp.fc1.weight", (size_t) F * D, l.w1, err) ||
                !f32(pk, N + "mlp.fc1.bias", F, l.b1, err) || !bf16(pk, N + "mlp.fc2.weight", (size_t) D * F, l.w2, err) ||
                !f32(pk, N + "mlp.fc2.bias", D, l.b2, err))
                return false;
            float *bq = nullptr, *bk = nullptr, *bv = nullptr;
            if (!f32(pk, N + "self_attn.q_proj.bias", D, bq, err) || !f32(pk, N + "self_attn.k_proj.bias", D, bk, err) ||
                !f32(pk, N + "self_attn.v_proj.bias", D, bv, err))
                return false;
            VK(cudaMemcpy(l.bqkv, bq, D * 4, cudaMemcpyDeviceToDevice));
            VK(cudaMemcpy(l.bqkv + D, bk, D * 4, cudaMemcpyDeviceToDevice));
            VK(cudaMemcpy(l.bqkv + 2 * D, bv, D * 4, cudaMemcpyDeviceToDevice));
        }
        const int Pj = g.proj, O = g.out_dim, M = g.merge * g.merge;
        if (!bf16(pk, "multi_modal_projector.linear_1.weight", (size_t) Pj * D, pl1, err) ||
            !f32(pk, "multi_modal_projector.linear_1.bias", Pj, pb1, err) ||
            !bf16(pk, "multi_modal_projector.linear_2.weight", (size_t) O * Pj, pl2, err) ||
            !f32(pk, "multi_modal_projector.linear_2.bias", O, pb2, err) ||
            !bf16(pk, "patch_merge_mlp.linear_1.weight", (size_t) Pj * O * M, pm1, err) ||
            !f32(pk, "patch_merge_mlp.linear_1.bias", Pj, pmb1, err) ||
            !bf16(pk, "patch_merge_mlp.linear_2.weight", (size_t) O * Pj, pm2, err) ||
            !f32(pk, "patch_merge_mlp.linear_2.bias", O, pmb2, err))
            return false;
        // the RoPE's frequencies: 2 * ((hd / 3) / 2) dims an axis, theta^(-2j / that)
        const int nf = ((g.hd / 3) / 2);
        std::vector<float> iv((size_t) nf);
        for (int j = 0; j < nf; ++j) iv[(size_t) j] = 1.0f / std::pow(g.theta, (float) (2 * j) / (float) (2 * nf));
        if (!alloc(inv, (size_t) nf, err)) return false;
        VK(cudaMemcpy(inv, iv.data(), iv.size() * 4, cudaMemcpyHostToDevice));
        const size_t N = (size_t) g.max_patches();
        if (!alloc(px, N * P, err) || !alloc(h, N * D, err) || !alloc(qkv, N * 3 * D, err) ||
            !alloc(sc, (size_t) kHeadChunk * N * N, err) || !alloc(att, N * D, err) ||
            !alloc(big, N * (size_t) std::max(std::max(F, Pj), O), err) || !alloc(out, N / M * (size_t) O, err) ||
            !alloc(x16, N * (size_t) std::max(std::max(F, Pj), O), err) || !alloc(pos, N * 3, err))
            return false;
        VK(cudaStreamSynchronize(s));
        return true;
    }

    int tokens(int w, int hgt) const {
        const int f = g.factor();
        if (w <= 0 || hgt <= 0 || w % f || hgt % f || (long long) w * hgt > g.max_pixels) return 0;
        return (hgt / f) * (w / f);
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
        VK(cudaMemcpyAsync(px, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice, s));
        VK(cudaMemcpyAsync(pos, hpos.data(), hpos.size() * 4, cudaMemcpyHostToDevice, s));
        const float one = 1.0f, zero = 0.0f;
        // the patch embedding (F32) and the pre-layernorm: the residual stream
        VB(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, D, N, P, &one, wpatch, P, px, P, &zero, h, D));
        layernorm_f32_kernel<<<N, 256, 0, s>>>(h, prew, preb, D, g.eps);
        const size_t ND = (size_t) N * D;
        const float scale = 1.0f / std::sqrt((float) g.hd);
        const int nf = (g.hd / 3) / 2;
        for (const Layer& l : L) {
            layernorm_kernel<<<N, 256, 0, s>>>(h, l.ln1w, l.ln1b, D, g.eps, x16);
            if (!gemm(l.wqkv, 3 * D, D, x16, N, qkv, err)) return false;
            qkv_rope_kernel<<<N, 256, 0, s>>>(qkv, l.bqkv, pos, inv, D, g.heads, g.hd, nf);
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
            if (!gemm(l.wo, D, D, x16, N, big, err)) return false;
            residual_kernel<<<blocks_for(ND), 256, 0, s>>>(h, big, l.bo, ND, D);
            layernorm_kernel<<<N, 256, 0, s>>>(h, l.ln2w, l.ln2b, D, g.eps, x16);
            if (!gemm(l.w1, F, D, x16, N, big, err)) return false;
            bias_act_kernel<<<blocks_for((size_t) N * F), 256, 0, s>>>(big, l.b1, (size_t) N * F, F, 1, nullptr, x16);
            if (!gemm(l.w2, D, F, x16, N, att, err)) return false;
            residual_kernel<<<blocks_for(ND), 256, 0, s>>>(h, att, l.b2, ND, D);
        }
        // the projector: per patch Pj -> out_dim, then each block's M x M rows (consecutive) as one
        const int Pj = g.proj, O = g.out_dim, NB = N / (M * M);
        to_bf16_kernel<<<blocks_for(ND), 256, 0, s>>>(h, ND, x16);
        if (!gemm(pl1, Pj, D, x16, N, big, err)) return false;
        bias_act_kernel<<<blocks_for((size_t) N * Pj), 256, 0, s>>>(big, pb1, (size_t) N * Pj, Pj, 1, nullptr, x16);
        if (!gemm(pl2, O, Pj, x16, N, big, err)) return false;
        bias_act_kernel<<<blocks_for((size_t) N * O), 256, 0, s>>>(big, pb2, (size_t) N * O, O, 0, nullptr, x16);
        __nv_bfloat16* m16 = (__nv_bfloat16*) qkv;   // (free now: [NB][Pj] BF16)
        if (!gemm(pm1, Pj, O * M * M, x16, NB, big, err)) return false;
        bias_act_kernel<<<blocks_for((size_t) NB * Pj), 256, 0, s>>>(big, pmb1, (size_t) NB * Pj, Pj, 1, nullptr, m16);
        if (!gemm(pm2, O, Pj, m16, NB, out, err)) return false;
        bias_act_kernel<<<blocks_for((size_t) NB * O), 256, 0, s>>>(out, pmb2, (size_t) NB * O, O, 0, out, nullptr);
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
size_t Vision::bytes() const { return p->nbytes; }
int Vision::device() const { return p->dev; }

}  // namespace mm3
