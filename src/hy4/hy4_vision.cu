// src/hy4/hy4_vision.cu - maya-hy4: Hy4-flash's vision tower (see hy4_vision.hpp)
#include "hy4_vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <sstream>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "hy4_pack.hpp"

namespace hy4 {

#define VK(call)                                                                       \
    do {                                                                               \
        const cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                       \
            err = std::string("vision: ") + #call + ": " + cudaGetErrorString(e_);     \
            return false;                                                              \
        }                                                                              \
    } while (0)
#define VB(call)                                                                       \
    do {                                                                               \
        const cublasStatus_t b_ = (call);                                              \
        if (b_ != CUBLAS_STATUS_SUCCESS) {                                             \
            err = std::string("vision: ") + #call + ": cublas " + std::to_string((int) b_); \
            return false;                                                              \
        }                                                                              \
    } while (0)

namespace {

using bf16_t = __nv_bfloat16;

constexpr size_t kScoreBytes = (size_t) 128 << 20;   // attention's scores held at once (query rows x keys x heads)
constexpr int kFfnRows = 4096;                       // the MLP's rows a pass (its 4304-wide middle held for them)
constexpr size_t kWs = (size_t) 32 << 20;            // cuBLAS's workspace

__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286536f * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

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
__global__ void layernorm_kernel(const float* x, const float* w, const float* b, int D, float eps, bf16_t* y16) {
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

// one block a row, in place: x = x / rms(x) . w
__global__ void rmsnorm_kernel(float* x, const float* w, int D, float eps) {
    __shared__ float red[32];
    float* r = x + (size_t) blockIdx.x * D;
    float q = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) q += r[i] * r[i];
    const float inv = rsqrtf(block_sum(q, red) / (float) D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) r[i] = r[i] * inv * w[i];
}

// y [n] += bias (rows of M)
__global__ void add_bias_kernel(float* y, const float* bias, size_t n, int M) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        y[i] += bias[i % (size_t) M];
}

// y16 [n] = act(y + bias) (rows of M; act 1: tanh GELU, 2: exact GELU)
__global__ void bias_act_kernel(const float* y, const float* bias, size_t n, int M, int act, bf16_t* y16) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        float v = y[i] + bias[i % (size_t) M];
        v = act == 1 ? gelu_tanh(v) : act == 2 ? gelu_erf(v) : v;
        y16[i] = __float2bfloat16(v);
    }
}

// h [n] += t + bias (rows of M)
__global__ void residual_kernel(float* h, const float* t, const float* bias, size_t n, int M) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        h[i] += t[i] + bias[i % (size_t) M];
}

__global__ void to_bf16_kernel(const float* x, size_t n, bf16_t* y) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        y[i] = __float2bfloat16(x[i]);
}

// one block a row of n: softmax in place
__global__ void softmax_kernel(float* s, int n) {
    __shared__ float red[32];
    __shared__ float mw[32];
    float* r = s + (size_t) blockIdx.x * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, r[i]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
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

// the 2 x 2 convolution's input: block b (row-major over ow x oh) -> m16 [b][ic * 4 + ky * 2 + kx] (the kernel's
// [OC][IC][KH][KW] order), from the normed patch rows xn [gh * gw][D] (raster order)
__global__ void merge_gather_kernel(const float* xn, int gw, int D, int ow, bf16_t* m16) {
    const int b = blockIdx.x, by = b / ow, bx = b % ow;
    bf16_t* dst = m16 + (size_t) b * 4 * D;
    for (int i = threadIdx.x; i < 4 * D; i += blockDim.x) {
        const int ic = i >> 2, ky = (i >> 1) & 1, kx = i & 1;
        dst[i] = __float2bfloat16(xn[((size_t) (2 * by + ky) * gw + 2 * bx + kx) * D + ic]);
    }
}

// the picture's rows: block (by, bx)'s y + bias at by * (ow + 1) + bx, the newline row at by * (ow + 1) + ow
__global__ void place_rows_kernel(const float* y, const float* bias, const float* nl, int ow, int O, float* out) {
    const int r = blockIdx.x, by = r / (ow + 1), bx = r % (ow + 1);
    float* dst = out + (size_t) r * O;
    if (bx == ow) {
        for (int c = threadIdx.x; c < O; c += blockDim.x) dst[c] = nl[c];
    } else {
        const float* src = y + ((size_t) by * ow + bx) * O;
        for (int c = threadIdx.x; c < O; c += blockDim.x) dst[c] = src[c] + bias[c];
    }
}

int blocks_for(size_t n) { return (int) std::min<size_t>((n + 255) / 256, 65535); }

size_t al(size_t n) { return (n + 255) & ~(size_t) 255; }

// A layer's matrices in the file's other types (a quantized mmproj: Q8_0 blocks - an F16 scale and 32 int8 -, F16 where
// a row is no multiple of 32) are made BF16 as a stage is gathered for its copy: the file stays the only copy in host
// memory, the GPU's side is the same for every type.
enum : int { kRaw = 0, kF16 = 1, kQ8 = 2 };

inline float half_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else {   // (subnormal)
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 0x400u)) {
                mm <<= 1;
                ++sh;
            }
            u = s | ((uint32_t) (113 - sh) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_bf16(float f) {   // (round to nearest, ties to even)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// n values of src (kF16: n F16; kQ8: n / 32 blocks of 34 bytes) -> BF16
void to_bf16_host(const uint8_t* src, int kind, size_t n, uint8_t* dst8) {
    uint16_t* dst = (uint16_t*) dst8;
    if (kind == kF16) {
        for (size_t i = 0; i < n; ++i) {
            uint16_t h;
            std::memcpy(&h, src + 2 * i, 2);
            dst[i] = f32_bf16(half_f32(h));
        }
        return;
    }
    for (size_t b = 0; b < n / 32; ++b) {
        const uint8_t* blk = src + b * 34;
        uint16_t h;
        std::memcpy(&h, blk, 2);
        const float d = half_f32(h);
        const int8_t* q = (const int8_t*) (blk + 2);
        uint16_t* o = dst + b * 32;
        for (int j = 0; j < 32; ++j) o[j] = f32_bf16(d * (float) q[j]);
    }
}

}  // namespace

int VisionGeo::tokens(int w, int h) const {
    const int f = factor();
    if (w <= 0 || h <= 0 || w % f || h % f || (long long) w * h > max_pixels) return 0;
    return (h / f) * (w / f + 1);
}

struct Vision::Impl {
    VisionGeo g;
    int dev = -1;
    cudaStream_t s = nullptr, cs = nullptr;   // the encode's stream, the weights' copy stream
    cublasHandle_t blas = nullptr;
    cudaEvent_t ev_up[2] = {nullptr, nullptr};     // a slot's stage copied in (its bounce buffer free again)
    cudaEvent_t ev_free[2] = {nullptr, nullptr};   // a slot's stage used (the slot free for the next one)
    // The weights stay in host memory between pictures (the pack's mapping: the page cache while the RAM holds it,
    // else the SSD; the projector's convolutions as BF16 here), and an encode copies them to the GPU a stage at a time
    // (a layer, or a projector matrix's rows) into two slots in turn, the next one's copy under this one's work.
    struct Piece {
        const uint8_t* src;
        size_t bytes, off;
        int kind;   // (kRaw: copied; kF16, kQ8: made BF16 - bytes is the BF16's)
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
    std::deque<std::vector<uint16_t>> hostb;   // (the convolutions' BF16)
    std::vector<uint8_t> pre_h;   // the patch embedding, its bias, the projector's norm, biases and newline row
    size_t o_wpatch = 0, o_bpatch = 0, o_norm = 0, o_b0 = 0, o_b2 = 0, o_nl = 0;
    const float* pos_tab = nullptr;   // the position grid [grid * grid][dim] F32 (the pack's mapping)
    int mat_n[2] = {}, mat_k[2] = {};   // mm.0, mm.2: BF16 [N][K]
    size_t slot = 0;
    uint8_t* bounce[2] = {nullptr, nullptr};   // pinned: a stage gathered for its copy
    std::vector<uint8_t> bounce_v[2];          // (pageable, when no pinned memory was had)
    bool pinned = false;
    size_t host_b = 0, last_b = 0;
    bool tf32 = std::getenv("HY4_VISION_F32") == nullptr;   // (attention's products in TF32; =1: plain F32)

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

    // a tensor of the mmproj file: its type and element count checked, where the pack maps it
    const uint8_t* need(const Pack& pk, const std::string& n, const char* type, size_t elems, std::string& err) {
        const Tensor* t = pk.get(n);
        if (!t) {
            err = "vision: " + n + " missing (pack the model again with its mmproj file)";
            return nullptr;
        }
        const size_t es = std::strcmp(type, "F32") == 0 ? 4 : 2;
        if (t->type != type || t->bytes != elems * es) {
            err = "vision: " + n + " is " + t->type + " of " + std::to_string(t->bytes) + " bytes, not " +
                  std::to_string(elems) + " " + type + " values";
            return nullptr;
        }
        host_b += t->bytes;
        return pk.data(*t);
    }

    // a layer's matrix: BF16 (copied as it is), or F16 / Q8_0 (a quantized mmproj file: made BF16 at each copy) - its
    // element count checked against its type's bytes
    const uint8_t* mat16(const Pack& pk, const std::string& n, size_t elems, int& kind, std::string& err) {
        const Tensor* t = pk.get(n);
        if (!t) {
            err = "vision: " + n + " missing (pack the model again with its mmproj file)";
            return nullptr;
        }
        if (t->type == "BF16" && t->bytes == elems * 2) kind = kRaw;
        else if (t->type == "F16" && t->bytes == elems * 2) kind = kF16;
        else if (t->type == "Q8_0" && elems % 32 == 0 && t->n(0) % 32 == 0 && t->bytes == elems / 32 * 34) kind = kQ8;
        else {
            err = "vision: " + n + " is " + t->type + " of " + std::to_string(t->bytes) + " bytes, not " +
                  std::to_string(elems) + " BF16, F16 or Q8_0 values";
            return nullptr;
        }
        host_b += t->bytes;
        return pk.data(*t);
    }

    // an F32 tensor whose values are BF16 values (the file's convolutions) -> BF16 here (exact; refused otherwise)
    const uint8_t* bf16_of_f32(const Pack& pk, const std::string& n, size_t elems, std::string& err) {
        const uint8_t* src = need(pk, n, "F32", elems, err);
        if (!src) return nullptr;
        hostb.emplace_back(elems);
        std::vector<uint16_t>& v = hostb.back();
        for (size_t i = 0; i < elems; ++i) {
            uint32_t u;
            std::memcpy(&u, src + 4 * i, 4);
            if (u & 0xffffu) {
                err = "vision: " + n + " holds values BF16 cannot (a converter that kept more precision)";
                return nullptr;
            }
            v[i] = (uint16_t) (u >> 16);
        }
        host_b += elems * 2;
        return (const uint8_t*) v.data();
    }

    static size_t add(Stage& S, const uint8_t* src, size_t bytes, bool align = true, int kind = kRaw) {
        if (align) S.bytes = al(S.bytes);
        const size_t off = S.bytes;
        S.pc.push_back({src, bytes, off, kind});
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
        if (!Vision::geometry(pk, g, err)) return false;
        dev = dev_;
        VK(cudaSetDevice(dev));
        VK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        VK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            VK(cudaEventCreateWithFlags(&ev_up[b], cudaEventDisableTiming));
            VK(cudaEventCreateWithFlags(&ev_free[b], cudaEventDisableTiming));
        }
        VB(cublasCreate(&blas));
        VB(cublasSetStream(blas, s));
        slot = slot_bytes(g);
        const int D = g.dim, F = g.ff, P = 3 * g.patch * g.patch, O = g.out_dim, M4 = g.merge * g.merge * D;
        auto pre = [&](const uint8_t* src, size_t n, size_t& off) {
            if (!src) return false;
            off = al(pre_h.size());
            pre_h.resize(off + n * 4);
            std::memcpy(pre_h.data() + off, src, n * 4);
            return true;
        };
        if (!pre(need(pk, "v.patch_embd.weight", "F32", (size_t) D * P, err), (size_t) D * P, o_wpatch) ||
            !pre(need(pk, "v.patch_embd.bias", "F32", D, err), D, o_bpatch) ||
            !pre(need(pk, "mm.pre_norm.weight", "F32", D, err), D, o_norm) ||
            !pre(need(pk, "mm.0.bias", "F32", M4, err), M4, o_b0) ||
            !pre(need(pk, "mm.2.bias", "F32", O, err), O, o_b2) ||
            !pre(need(pk, "v.image_newline", "F32", O, err), O, o_nl))
            return false;
        pos_tab = (const float*) need(pk, "v.position_embd.weight", "F32", (size_t) g.grid * g.grid * D, err);
        if (!pos_tab) return false;
        const size_t DD = (size_t) D * D;
        for (int i = 0; i < g.layers; ++i) {
            const std::string N = "v.blk." + std::to_string(i) + ".";
            const uint8_t* a[16] = {};
            int kd[16] = {};
            int k = 0;
            for (const char* n : {"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"})
                if (!(a[k++] = need(pk, N + n, "F32", D, err))) return false;
            if (!(a[4] = mat16(pk, N + "attn_q.weight", DD, kd[4], err)) ||
                !(a[5] = mat16(pk, N + "attn_k.weight", DD, kd[5], err)) ||
                !(a[6] = mat16(pk, N + "attn_v.weight", DD, kd[6], err)) ||
                !(a[7] = need(pk, N + "attn_q.bias", "F32", D, err)) ||
                !(a[8] = need(pk, N + "attn_k.bias", "F32", D, err)) ||
                !(a[9] = need(pk, N + "attn_v.bias", "F32", D, err)) ||
                !(a[10] = mat16(pk, N + "attn_out.weight", DD, kd[10], err)) ||
                !(a[11] = need(pk, N + "attn_out.bias", "F32", D, err)) ||
                !(a[12] = mat16(pk, N + "ffn_up.weight", (size_t) F * D, kd[12], err)) ||
                !(a[13] = need(pk, N + "ffn_up.bias", "F32", F, err)) ||
                !(a[14] = mat16(pk, N + "ffn_down.weight", (size_t) D * F, kd[14], err)) ||
                !(a[15] = need(pk, N + "ffn_down.bias", "F32", D, err)))
                return false;
            Stage S;
            LOff f{};
            f.ln1w = add(S, a[0], D * 4);
            f.ln1b = add(S, a[1], D * 4);
            f.ln2w = add(S, a[2], D * 4);
            f.ln2b = add(S, a[3], D * 4);
            f.wqkv = add(S, a[4], DD * 2, true, kd[4]);   // (q, k and v one [3D][D] matrix)
            add(S, a[5], DD * 2, false, kd[5]);
            add(S, a[6], DD * 2, false, kd[6]);
            f.bqkv = add(S, a[7], D * 4);
            add(S, a[8], D * 4, false);
            add(S, a[9], D * 4, false);
            f.wo = add(S, a[10], DD * 2, true, kd[10]);
            f.bo = add(S, a[11], D * 4);
            f.w1 = add(S, a[12], (size_t) F * D * 2, true, kd[12]);
            f.b1 = add(S, a[13], (size_t) F * 4);
            f.w2 = add(S, a[14], (size_t) D * F * 2, true, kd[14]);
            f.b2 = add(S, a[15], D * 4);
            if (al(S.bytes) > slot) {
                err = "vision: layer " + std::to_string(i) + "'s weights are more than a slot";
                return false;
            }
            st.push_back(std::move(S));
            lo.push_back(f);
        }
        // the projector's convolutions (BF16 here) in stages of rows that fit a slot
        const char* mn[2] = {"mm.0.weight", "mm.2.weight"};
        const int nn[2] = {M4, O}, kk[2] = {M4, M4};
        for (int m = 0; m < 2; ++m) {
            const uint8_t* src = bf16_of_f32(pk, mn[m], (size_t) nn[m] * kk[m], err);
            if (!src) return false;
            mat_n[m] = nn[m];
            mat_k[m] = kk[m];
            const size_t row = (size_t) kk[m] * 2;
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

    // a picture of N patches: its room of the GPU, carved
    struct Lay {
        size_t pre, s0, s1, px, h, qkv, att, big, x16, sc, out, ws, total;
    };
    // (oh: the rows of blocks - a newline row each; the largest room takes oh = N / 4, a picture one block wide)
    static Lay layout(const VisionGeo& g, size_t N, size_t oh, size_t slot, size_t pre_bytes) {
        const size_t D = (size_t) g.dim, P = (size_t) 3 * g.patch * g.patch, F = (size_t) g.ff;
        const size_t rc = std::min<size_t>(N, kFfnRows);
        const size_t NB = N / ((size_t) g.merge * g.merge);
        Lay y{};
        size_t off = 0;
        auto take = [&](size_t b) {
            const size_t o = off;
            off += al(b);
            return o;
        };
        y.pre = take(pre_bytes);
        y.s0 = take(slot);
        y.s1 = take(slot);
        y.px = take(N * P * 4);
        y.h = take(N * D * 4);
        y.qkv = take(N * 3 * D * 4);
        y.att = take(N * D * 4);
        y.big = take(rc * F * 4);
        y.x16 = take(std::max(N * D, rc * F) * 2);
        y.sc = take(std::min(kScoreBytes, N * N * (size_t) g.heads * 4));
        y.out = take((NB + oh) * (size_t) g.out_dim * 4);   // (a row a block, and a newline a row of blocks)
        y.ws = take(kWs);
        y.total = off;
        return y;
    }
    static size_t max_patches(const VisionGeo& g) { return (size_t) g.max_pixels / ((size_t) g.patch * g.patch); }
    static size_t slot_bytes(const VisionGeo& g) {
        const size_t D = (size_t) g.dim, F = (size_t) g.ff;
        return al(8 * D * 4 + 6 * D * 4 + 4 * D * D * 2 + 256 * 8 + 2 * F * D * 2 + F * 4 + 4096);
    }
    static size_t pre_bytes(const VisionGeo& g) {
        const size_t D = (size_t) g.dim, P = (size_t) 3 * g.patch * g.patch, M4 = 4 * D;
        return al((D * P + 2 * D + M4 + 2 * (size_t) g.out_dim) * 4 + 6 * 256);
    }

    bool encode(const uint8_t* rgb, int w, int hgt, std::vector<float>& rows, std::string& err) {
        const int n_tok = g.tokens(w, hgt);
        if (n_tok == 0) {
            err = "vision: a picture of " + std::to_string(w) + " x " + std::to_string(hgt) +
                  " pixels (sides multiples of " + std::to_string(g.factor()) + ", at most " +
                  std::to_string(g.max_pixels) + " pixels)";
            return false;
        }
        VK(cudaSetDevice(dev));
        const int ps = g.patch, gh = hgt / ps, gw = w / ps, N = gh * gw, D = g.dim, P = 3 * ps * ps;
        // the patches (raster order), each [channel][y][x], normalised
        std::vector<float> hp((size_t) N * P);
        for (int p = 0; p < N; ++p) {
            const int py = p / gw, pxx = p % gw;
            float* dst = hp.data() + (size_t) p * P;
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < ps; ++y)
                    for (int x = 0; x < ps; ++x) {
                        const float v = (float) rgb[((size_t) (py * ps + y) * w + pxx * ps + x) * 3 + c] * (1.0f / 255.0f);
                        dst[(c * ps + y) * ps + x] = (v - g.mean[c]) / g.stdv[c];
                    }
        }
        // the position grid resampled to gh x gw (the reference's bilinear sampling: scale (out + 0.1) / grid, pixel
        // centres, edges clamped) - the residual stream's start, the patch embedding added on the GPU
        std::vector<float> hpos((size_t) N * D);
        {
            const int n = g.grid;
            const float sx = ((float) gw + 0.1f) / (float) n, sy = ((float) gh + 0.1f) / (float) n;
            for (int y = 0; y < gh; ++y) {
                const float fy = ((float) y + 0.5f) / sy - 0.5f;
                int y0 = (int) std::floor(fy);
                const int y1 = std::clamp(y0 + 1, 0, n - 1);
                y0 = std::clamp(y0, 0, n - 1);
                const float wy1 = std::clamp(fy - (float) y0, 0.0f, 1.0f), wy0 = 1.0f - wy1;
                for (int x = 0; x < gw; ++x) {
                    const float fx = ((float) x + 0.5f) / sx - 0.5f;
                    int x0 = (int) std::floor(fx);
                    const int x1 = std::clamp(x0 + 1, 0, n - 1);
                    x0 = std::clamp(x0, 0, n - 1);
                    const float wx1 = std::clamp(fx - (float) x0, 0.0f, 1.0f), wx0 = 1.0f - wx1;
                    const float w00 = wy0 * wx0, w01 = wy0 * wx1, w10 = wy1 * wx0, w11 = wy1 * wx1;
                    const float* s00 = pos_tab + ((size_t) y0 * n + x0) * D;
                    const float* s01 = pos_tab + ((size_t) y0 * n + x1) * D;
                    const float* s10 = pos_tab + ((size_t) y1 * n + x0) * D;
                    const float* s11 = pos_tab + ((size_t) y1 * n + x1) * D;
                    float* d = hpos.data() + ((size_t) y * gw + x) * D;
                    for (int c = 0; c < D; ++c) d[c] = w00 * s00[c] + w01 * s01[c] + w10 * s10[c] + w11 * s11[c];
                }
            }
        }
        const Lay y = layout(g, (size_t) N, (size_t) (gh / g.merge), slot, pre_h.size());
        uint8_t* base = nullptr;
        if (cudaMalloc((void**) &base, y.total) != cudaSuccess) {
            cudaGetLastError();
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            err = "vision: " + std::to_string(y.total >> 20) + " MB of CUDA" + std::to_string(dev) + " for the picture, " +
                  std::to_string(fr >> 20) + " MB free";
            return false;
        }
        last_b = y.total;
        const bool ok = run(base, y, hp, hpos, gh, gw, rows, err);
        cudaStreamSynchronize(s);
        cudaStreamSynchronize(cs);
        cudaFree(base);
        if (!ok) cudaGetLastError();
        return ok;
    }

    bool run(uint8_t* base, const Lay& y, const std::vector<float>& hp, const std::vector<float>& hpos, int gh, int gw,
             std::vector<float>& rows, std::string& err) {
        const int N = gh * gw, D = g.dim, F = g.ff, P = 3 * g.patch * g.patch, O = g.out_dim, hd = g.hd;
        auto fp = [&](size_t o) { return (float*) (base + o); };
        float *px = fp(y.px), *h = fp(y.h), *qkv = fp(y.qkv), *att = fp(y.att), *big = fp(y.big), *sc = fp(y.sc);
        float* out = fp(y.out);
        bf16_t* x16 = (bf16_t*) (base + y.x16);
        float* pre = fp(y.pre);
        auto pv = [&](size_t o) { return (float*) ((uint8_t*) pre + o); };
        VB(cublasSetWorkspace(blas, base + y.ws, kWs));
        VK(cudaMemcpyAsync(pre, pre_h.data(), pre_h.size(), cudaMemcpyHostToDevice, s));
        VK(cudaMemcpyAsync(px, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice, s));
        VK(cudaMemcpyAsync(h, hpos.data(), hpos.size() * 4, cudaMemcpyHostToDevice, s));
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
            for (const Piece& q : S.pc) {
                if (q.kind == kRaw) std::memcpy(bounce[b] + q.off, q.src, q.bytes);
                else to_bf16_host(q.src, q.kind, q.bytes / 2, bounce[b] + q.off);
            }
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
        // the patch embedding (F32) onto the resampled positions, + its bias: the residual stream
        VB(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, D, N, P, &one, pv(o_wpatch), P, px, P, &one, h, D));
        const size_t ND = (size_t) N * D;
        add_bias_kernel<<<blocks_for(ND), 256, 0, s>>>(h, pv(o_bpatch), ND, D);
        // attention's passes: nh heads x qb query rows of scores held at once
        const size_t row_b = (size_t) N * 4;
        const size_t fit = std::max<size_t>(1, kScoreBytes / row_b);   // query rows (of all keys) at once
        const int nh = (int) std::max<size_t>(1, std::min<size_t>((size_t) g.heads, fit / (size_t) N));
        const int qb = nh > 1 ? N : (int) std::min<size_t>((size_t) N, fit);
        const float scale = 1.0f / std::sqrt((float) hd);
        const cublasComputeType_t ct = tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        const int rc = std::min(N, kFfnRows);
        for (int li = 0; li < g.layers; ++li) {
            uint8_t* sb = nullptr;
            if (!acquire(li, sb)) return false;
            const LOff& l = lo[(size_t) li];
            auto lf = [&](size_t o) { return (float*) (sb + o); };
            layernorm_kernel<<<N, 256, 0, s>>>(h, lf(l.ln1w), lf(l.ln1b), D, g.eps, x16);
            if (!gemm(sb + l.wqkv, 3 * D, D, x16, N, qkv, 3 * D, err)) return false;
            add_bias_kernel<<<blocks_for((size_t) N * 3 * D), 256, 0, s>>>(qkv, lf(l.bqkv), (size_t) N * 3 * D, 3 * D);
            // S = scale Q K^T for nh heads x qb queries (q, k, v read in place from qkv), softmax, S V
            for (int h0 = 0; h0 < g.heads; h0 += nh) {
                const int nhh = std::min(nh, g.heads - h0);
                for (int q0 = 0; q0 < N; q0 += qb) {
                    const int nq = std::min(qb, N - q0);
                    VB(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, nq, hd, &scale,
                                                  qkv + D + h0 * hd, CUDA_R_32F, 3 * D, hd,
                                                  qkv + (size_t) q0 * 3 * D + h0 * hd, CUDA_R_32F, 3 * D, hd, &zero,
                                                  sc, CUDA_R_32F, N, (long long) N * nq, nhh, ct, CUBLAS_GEMM_DEFAULT));
                    softmax_kernel<<<nq * nhh, 256, 0, s>>>(sc, N);
                    VB(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, nq, N, &one,
                                                  qkv + 2 * D + h0 * hd, CUDA_R_32F, 3 * D, hd, sc, CUDA_R_32F, N,
                                                  (long long) N * nq, &zero, att + (size_t) q0 * D + h0 * hd,
                                                  CUDA_R_32F, D, hd, nhh, ct, CUBLAS_GEMM_DEFAULT));
                }
            }
            to_bf16_kernel<<<blocks_for(ND), 256, 0, s>>>(att, ND, x16);
            if (!gemm(sb + l.wo, D, D, x16, N, att, D, err)) return false;
            residual_kernel<<<blocks_for(ND), 256, 0, s>>>(h, att, lf(l.bo), ND, D);
            // the MLP, rc rows a pass
            for (int r0 = 0; r0 < N; r0 += rc) {
                const int nr = std::min(rc, N - r0);
                float* hr = h + (size_t) r0 * D;
                layernorm_kernel<<<nr, 256, 0, s>>>(hr, lf(l.ln2w), lf(l.ln2b), D, g.eps, x16);
                if (!gemm(sb + l.w1, F, D, x16, nr, big, F, err)) return false;
                bias_act_kernel<<<blocks_for((size_t) nr * F), 256, 0, s>>>(big, lf(l.b1), (size_t) nr * F, F, 1, x16);
                if (!gemm(sb + l.w2, D, F, x16, nr, att, D, err)) return false;
                residual_kernel<<<blocks_for((size_t) nr * D), 256, 0, s>>>(hr, att, lf(l.b2), (size_t) nr * D, D);
            }
            if (!done(li)) return false;
        }
        // the projector: RMSNorm, each 2 x 2 block's patches -> the strided convolution (4608), exact GELU, the 1 x 1
        // convolution (out_dim) - each matrix's stages of rows into their columns
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
        const int ow = gw / g.merge, oh = gh / g.merge, NB = ow * oh, M4 = g.merge * g.merge * D;
        rmsnorm_kernel<<<N, 256, 0, s>>>(h, pv(o_norm), D, g.eps);
        merge_gather_kernel<<<NB, 256, 0, s>>>(h, gw, D, ow, x16);
        if (!mat(0, x16, NB, qkv)) return false;   // ([NB][4D] F32: qkv's room, free now)
        bf16_t* g16 = (bf16_t*) att;                // ([NB][4D] BF16: att's room)
        bias_act_kernel<<<blocks_for((size_t) NB * M4), 256, 0, s>>>(qkv, pv(o_b0), (size_t) NB * M4, M4, 2, g16);
        if (!mat(1, g16, NB, qkv)) return false;
        const int R = (ow + 1) * oh;
        place_rows_kernel<<<R, 256, 0, s>>>(qkv, pv(o_b2), pv(o_nl), ow, O, out);
        rows.resize((size_t) R * O);
        VK(cudaMemcpyAsync(rows.data(), out, rows.size() * 4, cudaMemcpyDeviceToHost, s));
        VK(cudaStreamSynchronize(s));
        VK(cudaGetLastError());
        return true;
    }
};

Vision::Vision() : p(new Impl) {}
Vision::~Vision() = default;

bool Vision::in_pack(const Pack& pk) { return pk.has("vision.dim") && pk.get("v.patch_embd.weight") != nullptr; }

bool Vision::geometry(const Pack& pk, VisionGeo& g, std::string& err) {
    g.dim = (int) pk.num("vision.dim", 1152);
    g.heads = (int) pk.num("vision.heads", 16);
    g.hd = g.heads > 0 ? g.dim / g.heads : 0;
    g.ff = (int) pk.num("vision.ff", 4304);
    g.layers = (int) pk.num("vision.layers", 27);
    g.patch = (int) pk.num("vision.patch", 16);
    g.merge = (int) pk.num("vision.merge", 2);
    g.grid = (int) pk.num("vision.grid", 128);
    g.out_dim = (int) pk.num("vision.out_dim", 3072);
    g.eps = (float) pk.num("vision.eps", 1e-5);
    g.min_pixels = (int) pk.num("vision.min_pixels", 4096);
    g.max_pixels = (int) pk.num("vision.max_pixels", 4194304);
    g.image_token = (int) pk.num("vision.image_token", -1);
    g.image_start = (int) pk.num("vision.image_start", -1);
    g.image_end = (int) pk.num("vision.image_end", -1);
    for (const char* k : {"vision.mean", "vision.std"}) {
        std::stringstream ss(pk.str(k, "0.5,0.5,0.5"));
        float* dst = std::strcmp(k, "vision.mean") == 0 ? g.mean : g.stdv;
        std::string x;
        for (int i = 0; i < 3 && std::getline(ss, x, ','); ++i) dst[i] = std::strtof(x.c_str(), nullptr);
    }
    if (g.heads < 1 || g.dim % g.heads || g.merge != 2 || g.patch < 1 || g.grid < 2 || g.out_dim != pk.g.n_embd ||
        g.image_token < 0 || g.image_start < 0 || g.image_end < 0 || g.max_pixels < g.factor() * g.factor() ||
        g.stdv[0] == 0 || g.stdv[1] == 0 || g.stdv[2] == 0) {
        err = "vision: a geometry in hy4.txt maya-hy4 does not run";
        return false;
    }
    return true;
}

size_t Vision::room_for(const VisionGeo& g) {
    const size_t N = Impl::max_patches(g);
    return Impl::layout(g, N, N / ((size_t) g.merge * g.merge), Impl::slot_bytes(g), Impl::pre_bytes(g)).total;
}
bool Vision::load(const Pack& pk, int dev, std::string& err) {
    if (!p->load(pk, dev, err)) return false;
    if (p->pre_h.size() > Impl::pre_bytes(p->g)) {
        err = "vision: the patch embedding and the projector's vectors are more than their room";
        return false;
    }
    return true;
}
const VisionGeo& Vision::geo() const { return p->g; }
bool Vision::encode(const uint8_t* rgb, int w, int h, std::vector<float>& rows, std::string& err) {
    return p->encode(rgb, w, h, rows, err);
}
size_t Vision::host_bytes() const { return p->host_b; }
size_t Vision::room_bytes() const { return room_for(p->g); }
size_t Vision::last_bytes() const { return p->last_b; }
int Vision::device() const { return p->dev; }

}  // namespace hy4
