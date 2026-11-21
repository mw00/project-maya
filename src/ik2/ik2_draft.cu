// src/ik2/ik2_draft.cu - maya-ik2's drafter (the design: ik2_draft.hpp).
#include "ik2_draft.hpp"
#include "ik2_kernels.cuh"

#include <cuda_fp16.h>
#include <fcntl.h>

#ifdef _WIN32
#include <io.h>
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#define O_RDONLY (_O_RDONLY | _O_BINARY)
#define open _open
#define close _close
static inline ssize_t pread(int fd, void* buf, size_t count, int64_t offset) {
    if (_lseeki64(fd, offset, SEEK_SET) == -1) return -1;
    return _read(fd, buf, (unsigned int)count);
}
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

namespace ik2::draft {
namespace {

namespace k = ik2::k;

void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("drafter: ") + what + ": " + cudaGetErrorString(e));
}
void cb(cublasStatus_t e, const char* what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("drafter: ") + what + ": cuBLAS " +
                                                             std::to_string((int) e));
}
template <class T>
T* dalloc(size_t n) {
    T* p = nullptr;
    cu(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)), "cudaMalloc");
    return p;
}
float bf(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- safetensors: the header's tensors (name -> dtype, shape, byte range)
struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;
};
struct StFile {
    std::string path;
    uint64_t data0 = 0;
    std::map<std::string, StTensor> t;
};

StFile st_open(const std::string& path) {
    StFile f;
    f.path = path;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("drafter: cannot open " + path);
    uint64_t n = 0;
    if (std::fread(&n, 8, 1, fp) != 1 || n > (64u << 20)) throw std::runtime_error("drafter: not a safetensors file");
    std::string h(n, '\0');
    if (std::fread(&h[0], 1, n, fp) != n) throw std::runtime_error("drafter: short header");
    std::fclose(fp);
    f.data0 = 8 + n;
    // {"name":{"dtype":"BF16","shape":[a,b],"data_offsets":[x,y]}, ...}: a flat scan is enough for this header
    size_t i = h.find('{') + 1;
    while (true) {
        const size_t q0 = h.find('"', i);
        if (q0 == std::string::npos) break;
        const size_t q1 = h.find('"', q0 + 1);
        const std::string name = h.substr(q0 + 1, q1 - q0 - 1);
        const size_t ob = h.find('{', q1), oe = h.find('}', ob);
        const std::string body = h.substr(ob, oe - ob);
        i = oe + 1;
        if (name == "__metadata__") continue;
        StTensor t;
        const size_t d0 = body.find("\"dtype\"");
        const size_t d1 = body.find('"', body.find(':', d0) + 1), d2 = body.find('"', d1 + 1);
        t.dtype = body.substr(d1 + 1, d2 - d1 - 1);
        auto ints = [&](const char* key) {
            std::vector<int64_t> v;
            const size_t a = body.find(key);
            const size_t b0 = body.find('[', a), b1 = body.find(']', b0);
            const std::string list = body.substr(b0 + 1, b1 - b0 - 1);
            for (const char* p = list.c_str(); *p;) {
                while (*p == ' ' || *p == ',') ++p;
                if (!*p) break;
                v.push_back(std::strtoll(p, (char**) &p, 10));
            }
            return v;
        };
        t.shape = ints("\"shape\"");
        const auto off = ints("\"data_offsets\"");
        if (off.size() != 2) throw std::runtime_error("drafter: bad entry " + name);
        t.begin = (uint64_t) off[0], t.end = (uint64_t) off[1];
        f.t[name] = t;
    }
    return f;
}

std::vector<uint16_t> st_bf16(const StFile& f, const std::string& name, const std::vector<int64_t>& shape) {
    auto it = f.t.find(name);
    if (it == f.t.end()) throw std::runtime_error("drafter: no tensor " + name);
    if (it->second.dtype != "BF16") throw std::runtime_error("drafter: " + name + " is " + it->second.dtype);
    if (it->second.shape != shape) throw std::runtime_error("drafter: " + name + ": unexpected shape");
    const size_t n = (it->second.end - it->second.begin) / 2;
    std::vector<uint16_t> v(n);
    const int fd = ::open(f.path.c_str(), O_RDONLY);
    size_t done = 0;
    while (done < n * 2) {
        const ssize_t r = ::pread(fd, (char*) v.data() + done, n * 2 - done, (off_t) (f.data0 + it->second.begin + done));
        if (r <= 0) throw std::runtime_error("drafter: read failed");
        done += (size_t) r;
    }
    ::close(fd);
    return v;
}

}  // namespace

struct DLayer {
    bool local = false;
    float *in_norm, *post_norm, *dflare, *q_norm, *k_norm, *k_norm_ctx;
    uint16_t *wq, *wk, *wv, *wo, *wkc, *wvc, *wgu, *wd;   // BF16 [out][in]
    float *kc = nullptr, *vc = nullptr;                    // the context caches, F32 [cap][KV * 128] (a window's
    uint16_t *kc16 = nullptr, *vc16 = nullptr;             //  layer) or FP16 (the layer that sees the whole context)
    int cap = 0;                                           // their rows: a ring of the last cap positions
};

struct Drafter {
    int taps = 8, C = 4608, H = 48, KV = 8, F = 18432, R = 256, V = 0, ctx = 0;
    int cap = 0;   // the rows of the layer without a window's context cache (IK2_DRAFT_CTX, default the context)
    float* ctx_tmp = nullptr;   // write_context's K and V rows before an FP16 cache, [2][kMixRows][KV * 128]
    float eps = 1e-6f, theta = 1e6f;
    std::vector<DLayer> L;
    float *alpha, *norm, *mask, *mk_dflare, *rnn_b;
    uint16_t *w1, *rnn_w;
    std::vector<uint16_t> hout_w, hout_b;   // host BF16, for the fold
    uint16_t *fold_w = nullptr, *fold_b = nullptr;   // FP16 [R][n_rows], [n_rows]
    int n_rows = 0;
    const int8_t* head_q8 = nullptr;      // the target's head as planar Q8_0 (set_head_q8): the base logits through it
    const uint16_t* head_q8d = nullptr;
    int* shortlist = nullptr;
    float* shortlist_scores = nullptr;
    cublasHandle_t blas;
    cudaStream_t s;
    size_t vram = 0;
    // work
    float *x, *res, *h, *q, *kk, *vv, *o, *gu, *f_rnn, *base, *state, *latent, *proj, *scratch, *mix;
    uint16_t *hb, *hb2, *rnn_in;
    __half* x16;
    int *pos, *ctx_pos, *toks;
    int* h_out = nullptr;   // pinned host [kBlock - 1]
};

namespace {

constexpr int kMixRows = 512;
constexpr int kFoldChunk = 4096;

__global__ void iota_kernel(int* out, int start, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = start + i;
}

__global__ void init_block_kernel(const float* anchor_h, const float* mask, float* res, int* pos, int* toks, int T,
                                  int C, int P, int anchor) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0 && blockIdx.x == 0) {
        if (threadIdx.x < T) pos[threadIdx.x] = P + threadIdx.x;
        if (threadIdx.x == 0) toks[0] = anchor;
    }
    if (c < C) res[(size_t) t * C + c] = (t == 0 ? anchor_h : mask)[c];
}

__global__ void f16_to_f32_kernel(const uint16_t* src, size_t n, float* dst) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(__ushort_as_half(src[i]));
}

__global__ void pack_f16_strided_kernel(const float* src, int m, int n_rows, int r0, uint16_t* dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i < m) dst[(size_t) r * n_rows + r0 + i] = __half_as_ushort(__float2half_rn(src[(size_t) r * m + i]));
}

// Each warp keeps the strongest base candidate among 32 vocabulary entries.
// Only the drafter uses this shortlist; the target still verifies the full vocabulary.
__global__ void draft_shortlist(const float* base, int ld, int vocab, int stride, int* picks) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    float val = token < vocab ? base[(size_t)row * ld + token] : -INFINITY;
    int idx = token;
    for (int step = 16; step; step >>= 1) {
        float v = __shfl_down_sync(0xffffffff, val, step);
        int j = __shfl_down_sync(0xffffffff, idx, step);
        if (v > val || (v == val && j < idx)) val = v, idx = j;
    }
    if (!(threadIdx.x & 31) && token < vocab) picks[row * stride + token / 32] = idx;
}
__global__ void fold_vocab_rows(const float* src, int m, int r0, int R, uint16_t* dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * R) {
        const int i = idx / R, r = idx % R;
        dst[(size_t)(r0 + i) * R + r] = __half_as_ushort(__float2half_rn(src[(size_t)r * m + i]));
    }
}
__global__ void shortlist_gemv(const int* ids, int n, const uint16_t* W, const float* x, const float* base,
                               const uint16_t* bias, int R, float* scores) {
    const int lane = threadIdx.x & 31;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (i >= n) return;
    const int token = ids[i];
    float dot = 0.0f;
    for (int r = lane; r < R; r += 32)
        dot = fmaf(__half2float(__ushort_as_half(W[(size_t)token * R + r])), x[r], dot);
    for (int step = 16; step; step >>= 1) dot += __shfl_down_sync(0xffffffff, dot, step);
    if (lane == 0) scores[i] = dot + base[token] + __half2float(__ushort_as_half(bias[token]));
}
__global__ void shortlist_pick(int* out, const int* ids) { *out = ids[*out]; }

uint16_t* up16(Drafter* d, const std::vector<uint16_t>& v) {
    uint16_t* p = dalloc<uint16_t>(v.size());
    cu(cudaMemcpy(p, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "upload");
    d->vram += v.size() * 2;
    return p;
}
float* up32(Drafter* d, const std::vector<uint16_t>& v) {
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i) f[i] = bf(v[i]);
    float* p = dalloc<float>(f.size());
    cu(cudaMemcpy(p, f.data(), f.size() * 4, cudaMemcpyHostToDevice), "upload");
    d->vram += f.size() * 4;
    return p;
}

// Y [T][N] (ldy) = X [T][K] BF16 . W [N][K]^T BF16, F32 sums
void gemm_bf(Drafter* d, const uint16_t* W, int N, int K, const uint16_t* X, int T, float* Y, int ldy) {
    const float one = 1.0f, zero = 0.0f;
    cb(cublasGemmEx(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "gemm_bf");
}
void gemm_h(Drafter* d, const uint16_t* W, int N, int K, const void* X, int T, float* Y, int ldy) {
    const float one = 1.0f, zero = 0.0f;
    cb(cublasGemmEx(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16F, K, X, CUDA_R_16F, K, &zero, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "gemm_h");
}

void layer_attn(Drafter* d, DLayer& l, int P, float* out) {
    const int T = kBlock, HD = d->H * 128, KD = d->KV * 128;
    k::to_bf16(d->h, (size_t) T * d->C, d->hb, d->s);
    gemm_bf(d, l.wq, HD, d->C, d->hb, T, d->q, HD);
    gemm_bf(d, l.wk, KD, d->C, d->hb, T, d->kk, KD);
    gemm_bf(d, l.wv, KD, d->C, d->hb, T, d->vv, KD);
    k::rms(d->q, l.q_norm, T * d->H, 128, d->eps, d->q, d->s);
    k::rms(d->kk, l.k_norm, T * d->KV, 128, d->eps, d->kk, d->s);
    k::rope_neox(d->q, T, d->H, d->pos, d->theta, d->s);
    k::rope_neox(d->kk, T, d->KV, d->pos, d->theta, d->s);
    // Local layers (0, 1, 3, 4) match SGLang inkling2_mtp.py's window_anchor_at_start=True with window=1024:
    // every query in the block [P .. P+7] sees context [max(0, P - 1023) .. P-1] plus all 8 block tokens.
    const int k_lo = l.local ? std::max(0, P - 1023) : std::max(0, P - l.cap);
    if (l.kc16)
        k::draft_attn_f16(d->q, l.kc16, l.vc16, k_lo, P, d->kk, d->vv, T, d->H, d->KV, 1.0f / std::sqrt(128.0f),
                          d->scratch, d->o, d->s, l.cap);
    else
        k::draft_attn(d->q, l.kc, l.vc, k_lo, P, d->kk, d->vv, T, d->H, d->KV, 1.0f / std::sqrt(128.0f), d->scratch,
                      d->o, d->s, l.cap);
    k::to_bf16(d->o, (size_t) T * HD, d->hb, d->s);
    gemm_bf(d, l.wo, d->C, HD, d->hb, T, out, d->C);
}

void layer_mlp(Drafter* d, DLayer& l, float* out) {
    const int T = kBlock;
    k::to_bf16(d->h, (size_t) T * d->C, d->hb, d->s);
    gemm_bf(d, l.wgu, 2 * d->F, d->C, d->hb, T, d->gu, 2 * d->F);
    k::silu_mul_bf16(d->gu, T, d->F, d->hb2, d->s);
    gemm_bf(d, l.wd, d->C, d->F, d->hb2, T, out, d->C);
}

}  // namespace

const std::vector<int>& tap_layers() {
    static const std::vector<int> v = {5, 11, 17, 23, 29, 35, 41};
    return v;
}

Drafter* load(const std::string& path, int C, int ctx, cublasHandle_t blas, cudaStream_t s) {
    const StFile f = st_open(path);
    auto* d = new Drafter;
    d->C = C, d->ctx = ctx, d->blas = blas, d->s = s;
    {
        // (IK2_DRAFT_CTX: the positions the layer without a window sees, the last ones; default the whole context)
        const int want = getenv("IK2_DRAFT_CTX") ? std::atoi(getenv("IK2_DRAFT_CTX")) : ctx;
        d->cap = std::min(ctx, std::max(want, 2048));
    }
    const bool mini = C == 2304;
    if (mini) {
        d->H = 24;
        d->F = 9216;
        d->taps = 4;
    } else if (C != 4608) {
        throw std::runtime_error("drafter: unsupported target width");
    }
    const int layers = mini ? 3 : 5;
    const int64_t H = d->H, KV = d->KV, F = d->F, R = d->R;
    {
        auto it = f.t.find("markov_head.markov_w1.weight");
        if (it == f.t.end() || it->second.shape.size() != 2) throw std::runtime_error("drafter: no markov_w1");
        d->V = (int) it->second.shape[0];
    }
    // fuse_weight [6][8] -> softmax over each row
    {
        const auto fw = st_bf16(f, "fuse_weight", {layers + 1, d->taps});
        std::vector<float> a(fw.size());
        for (int r = 0; r <= layers; ++r) {
            float m = -1e30f, sum = 0;
            for (int t = 0; t < d->taps; ++t) m = std::max(m, bf(fw[r * d->taps + t]));
            for (int t = 0; t < d->taps; ++t) sum += (a[r * d->taps + t] = std::exp(bf(fw[r * d->taps + t]) - m));
            for (int t = 0; t < d->taps; ++t) a[r * d->taps + t] /= sum;
        }
        d->alpha = dalloc<float>(a.size());
        cu(cudaMemcpy(d->alpha, a.data(), a.size() * 4, cudaMemcpyHostToDevice), "alpha");
    }
    const int local_ids[] = {0, 1, 3, 4};
    d->L.resize(layers);
    for (int i = 0; i < layers; ++i) {
        DLayer& l = d->L[i];
        const std::string b = "layers." + std::to_string(i) + ".";
        l.local = mini ? i != 1 : std::find(std::begin(local_ids), std::end(local_ids), i) != std::end(local_ids);
        l.in_norm = up32(d, st_bf16(f, b + "input_layernorm.weight", {C}));
        l.post_norm = up32(d, st_bf16(f, b + "post_attention_layernorm.weight", {C}));
        l.dflare = up32(d, st_bf16(f, b + "dflare_norm.weight", {C}));
        l.q_norm = up32(d, st_bf16(f, b + "self_attn.q_norm.weight", {128}));
        l.k_norm = up32(d, st_bf16(f, b + "self_attn.k_norm.weight", {128}));
        l.k_norm_ctx = up32(d, st_bf16(f, b + "self_attn.k_norm_ctx.weight", {128}));
        l.wq = up16(d, st_bf16(f, b + "self_attn.q_proj.weight", {H * 128, C}));
        l.wk = up16(d, st_bf16(f, b + "self_attn.k_proj.weight", {KV * 128, C}));
        l.wv = up16(d, st_bf16(f, b + "self_attn.v_proj.weight", {KV * 128, C}));
        l.wo = up16(d, st_bf16(f, b + "self_attn.o_proj.weight", {C, H * 128}));
        l.wkc = up16(d, st_bf16(f, b + "self_attn.k_proj_ctx.weight", {KV * 128, C}));
        l.wvc = up16(d, st_bf16(f, b + "self_attn.v_proj_ctx.weight", {KV * 128, C}));
        {
            auto g = st_bf16(f, b + "mlp.gate_proj.weight", {F, C});
            const auto u = st_bf16(f, b + "mlp.up_proj.weight", {F, C});
            g.insert(g.end(), u.begin(), u.end());   // gate's rows, then up's
            l.wgu = up16(d, g);
        }
        l.wd = up16(d, st_bf16(f, b + "mlp.down_proj.weight", {C, F}));
        if (l.local) {   // (a window of 1024: a ring of 2048 rows, F32)
            l.cap = std::min(ctx, 2048);
            l.kc = dalloc<float>((size_t) l.cap * KV * 128);
            l.vc = dalloc<float>((size_t) l.cap * KV * 128);
            d->vram += (size_t) l.cap * KV * 128 * 8;
        } else {         // (the whole context, or IK2_DRAFT_CTX's last positions: FP16)
            l.cap = d->cap;
            l.kc16 = dalloc<uint16_t>((size_t) l.cap * KV * 128);
            l.vc16 = dalloc<uint16_t>((size_t) l.cap * KV * 128);
            d->vram += (size_t) l.cap * KV * 128 * 4;
        }
    }
    d->norm = up32(d, st_bf16(f, "norm.weight", {C}));
    d->mask = up32(d, st_bf16(f, "mask_embed", {C}));
    d->mk_dflare = up32(d, st_bf16(f, "markov_head.dflare_norm.weight", {C}));
    d->w1 = up16(d, st_bf16(f, "markov_head.markov_w1.weight", {d->V, R}));
    d->rnn_w = up16(d, st_bf16(f, "markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * C}));
    d->rnn_b = up32(d, st_bf16(f, "markov_head.rnn_proj.bias", {3 * R}));
    d->hout_w = st_bf16(f, "markov_head.hidden_out.weight", {C, R});
    d->hout_b = st_bf16(f, "markov_head.hidden_out.bias", {C});
    // work buffers
    const size_t T = kBlock;
    d->x = dalloc<float>(T * C), d->res = dalloc<float>(T * C), d->h = dalloc<float>(T * C);
    d->q = dalloc<float>(T * H * 128), d->o = dalloc<float>(T * H * 128);
    d->kk = dalloc<float>(std::max<size_t>(T, kMixRows) * KV * 128), d->vv = dalloc<float>(T * KV * 128);
    d->gu = dalloc<float>(T * 2 * F);
    d->hb = dalloc<uint16_t>(std::max<size_t>(T * std::max<int64_t>(C, H * 128), (size_t) kMixRows * C));
    d->hb2 = dalloc<uint16_t>(T * F);
    // (draft_attn's: the logits [H][8][<= cap + T + chunk padding], the chunks' outputs [64][H][8][128], (max, sum))
    d->ctx_tmp = dalloc<float>((size_t) 2 * kMixRows * KV * 128);
    d->scratch = dalloc<float>((size_t) H * 8 * (d->cap + T + 512) + (size_t) 64 * H * 8 * 130);
    d->mix = dalloc<float>((size_t) kMixRows * C);
    d->f_rnn = dalloc<float>(C);
    d->state = dalloc<float>(R), d->latent = dalloc<float>(R), d->proj = dalloc<float>(3 * R);
    d->rnn_in = dalloc<uint16_t>(2 * R + 2 * C);
    d->x16 = dalloc<__half>((kBlock - 1) * (size_t) C);
    d->pos = dalloc<int>(T), d->ctx_pos = dalloc<int>(kMixRows), d->toks = dalloc<int>(kBlock);
    cu(cudaHostAlloc(&d->h_out, (kBlock - 1) * sizeof(int), cudaHostAllocDefault), "h_out");
    return d;
}

void fold(Drafter* d, const uint16_t* lm_f16, int n_rows) {
    const int C = d->C, R = d->R;
    d->n_rows = n_rows;
    if (getenv("IK2_DRAFT_SHORTLIST") && atoi(getenv("IK2_DRAFT_SHORTLIST"))) {
        const int stride = (n_rows + 31) / 32;
        d->shortlist = dalloc<int>((kBlock - 1) * stride);
        d->shortlist_scores = dalloc<float>(stride);
    }
    // fold_w [R][n_rows] = (lm . hidden_out)^T, fold_b [n_rows] = lm . b (stored as FP16 to halve Markov traffic).
    // Compute chunked in F32 (like inkling2_mtp.py's 4096-row double() fold) so BF16 hout_w/b never underflow FP16
    // before the C=4608 inner product, then round once to FP16.
    std::vector<float> ht((size_t) R * C), hb(C);
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r) ht[(size_t) r * C + c] = bf(d->hout_w[(size_t) c * R + r]);
    for (int c = 0; c < C; ++c) hb[c] = bf(d->hout_b[c]);
    float *dht = dalloc<float>(ht.size()), *dhb = dalloc<float>(hb.size());
    cu(cudaMemcpy(dht, ht.data(), ht.size() * sizeof(float), cudaMemcpyHostToDevice), "fold");
    cu(cudaMemcpy(dhb, hb.data(), hb.size() * sizeof(float), cudaMemcpyHostToDevice), "fold");
    const int chunk = std::min(kFoldChunk, std::max(1, n_rows));
    float* dlm = dalloc<float>((size_t) chunk * C);
    float* dfw = dalloc<float>((size_t) R * chunk);
    float* dfb = dalloc<float>(chunk);
    d->fold_w = dalloc<uint16_t>((size_t) R * n_rows);
    d->fold_b = dalloc<uint16_t>(n_rows);
    d->base = dalloc<float>((size_t) (kBlock - 1) * n_rows);
    const float one = 1.0f, zero = 0.0f;
    for (int r0 = 0; r0 < n_rows; r0 += chunk) {
        const int m = std::min(chunk, n_rows - r0);
        const size_t elems = (size_t) m * C;
        f16_to_f32_kernel<<<(unsigned) ((elems + 255) / 256), 256, 0, d->s>>>(lm_f16 + (size_t) r0 * C, elems, dlm);
        cb(cublasSgemm(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, R, C, &one, dlm, C, dht, C, &zero, dfw, m), "fold w");
        cb(cublasSgemv(d->blas, CUBLAS_OP_T, C, m, &one, dlm, C, dhb, 1, &zero, dfb, 1), "fold b");
        if (d->shortlist)
            fold_vocab_rows<<<(m * R + 255) / 256, 256, 0, d->s>>>(dfw, m, r0, R, d->fold_w);
        else
            pack_f16_strided_kernel<<<dim3((unsigned) ((m + 255) / 256), (unsigned) R), 256, 0, d->s>>>(
            dfw, m, n_rows, r0, d->fold_w);
        k::to_f16(dfb, (size_t) m, d->fold_b + r0, d->s);
    }
    cu(cudaStreamSynchronize(d->s), "fold");
    cudaFree(dht), cudaFree(dhb), cudaFree(dlm), cudaFree(dfw), cudaFree(dfb);
    d->vram += ((size_t) R + 1) * n_rows * 2 + (size_t) (kBlock - 1) * n_rows * 4;
}

void write_context(Drafter* d, const float* taps, int ld, int r0, int m, int pos0) {
    const int C = d->C, KD = d->KV * 128;
    for (auto& l : d->L) {
        const int li = (int) (&l - d->L.data());
        for (int a = 0; a < m;) {
            const int at = (pos0 + a) % l.cap;
            const int n = std::min({kMixRows, m - a, l.cap - at});   // (a ring: a chunk ends where it wraps)
            iota_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, d->s>>>(d->ctx_pos, pos0 + a, n);
            k::mix_rms(taps + (size_t) (r0 + a) * C, d->taps, (size_t) ld * C, n, C, d->alpha + li * d->taps, l.dflare,
                       d->eps, d->mix, d->s);
            k::to_bf16(d->mix, (size_t) n * C, d->hb, d->s);
            float* kd = l.kc16 ? d->ctx_tmp : l.kc + (size_t) at * KD;
            float* vd = l.kc16 ? d->ctx_tmp + (size_t) kMixRows * KD : l.vc + (size_t) at * KD;
            gemm_bf(d, l.wkc, KD, C, d->hb, n, kd, KD);
            gemm_bf(d, l.wvc, KD, C, d->hb, n, vd, KD);
            k::rms(kd, l.k_norm_ctx, n * d->KV, 128, d->eps, kd, d->s);
            k::rope_neox(kd, n, d->KV, d->ctx_pos, d->theta, d->s);
            if (l.kc16) {
                k::to_f16(kd, (size_t) n * KD, l.kc16 + (size_t) at * KD, d->s);
                k::to_f16(vd, (size_t) n * KD, l.vc16 + (size_t) at * KD, d->s);
            }
            a += n;
        }
    }
}

void set_f_rnn(Drafter* d, const float* taps, int ld, int row) {
    k::mix_rms(taps + (size_t) row * d->C, d->taps, (size_t) ld * d->C, 1, d->C, d->alpha + d->L.size() * d->taps, d->mk_dflare,
               d->eps, d->f_rnn, d->s);
}

void propose(Drafter* d, int anchor, const float* anchor_h, int P, const uint16_t* lm_f16, int n_rows, int n_vocab,
             int* out, int proposals) {
    const int C = d->C, T = kBlock;
    // Initialize the residual stream [anchor_h; mask x 7], block positions, and anchor token in one kernel.
    // Layer 0 reads d->res and overwrites d->x, so d->x does not need pre-filling.
    init_block_kernel<<<dim3((unsigned) ((C + 255) / 256), (unsigned) T), 256, 0, d->s>>>(
        anchor_h, d->mask, d->res, d->pos, d->toks, T, C, P, anchor);
    const size_t TC = (size_t) T * C;
    // pre-norm layers, the residual carried apart (res = x + res before each norm)
    for (size_t i = 0; i < d->L.size(); ++i) {
        DLayer& l = d->L[i];
        if (i > 0) k::add(d->x, d->res, TC, d->res, d->s);
        k::rms(d->res, l.in_norm, T, C, d->eps, d->h, d->s);
        layer_attn(d, l, P, d->x);
        k::add(d->x, d->res, TC, d->res, d->s);
        k::rms(d->res, l.post_norm, T, C, d->eps, d->h, d->s);
        layer_mlp(d, l, d->x);
    }
    k::add(d->x, d->res, TC, d->res, d->s);
    k::rms(d->res, d->norm, T, C, d->eps, d->h, d->s);   // the draft hidden, rows 1 .. 7 used
    // base logits of rows 1 .. 7 through the target's head
    k::to_f16(d->h + C, (size_t) proposals * C, (uint16_t*) d->x16, d->s);
    if (d->head_q8)
        k::q8_tile(d->head_q8, d->head_q8d, n_rows, C, (const uint16_t*) d->x16, proposals, d->base, n_rows, d->s);
    else
        gemm_h(d, lm_f16, n_rows, C, d->x16, proposals, d->base, n_rows);
    const int shortlist_n = (n_vocab + 31) / 32;
    if (d->shortlist)
        draft_shortlist<<<dim3((n_vocab + 127) / 128, proposals), 128, 0, d->s>>>(
            d->base, n_rows, n_vocab, shortlist_n, d->shortlist);
    // the Markov head, a token at a time (note: k::rnn_gates keeps state/latent in F32 whereas SGLang mtp.py
    // rounds intermediate gate/tanh ops to BF16; fused argmax_gemv_f16 reads FP16 fold_w/fold_b + F32 base/latent)
    cu(cudaMemsetAsync(d->state, 0, d->R * 4, d->s), "state");
    for (int t = 0; t < proposals; ++t) {
        k::rnn_inputs(d->state, d->w1, d->toks + t, d->h + (size_t) (t + 1) * C, d->f_rnn, d->R, C, d->rnn_in, d->s);
        gemm_bf(d, d->rnn_w, 3 * d->R, 2 * d->R + 2 * C, d->rnn_in, 1, d->proj, 3 * d->R);
        k::rnn_gates(d->proj, d->rnn_b, d->R, d->state, d->latent, d->s);
        if (d->shortlist) {
            const int* ids = d->shortlist + t * shortlist_n;
            shortlist_gemv<<<(shortlist_n + 3) / 4, 128, 0, d->s>>>(ids, shortlist_n, d->fold_w, d->latent,
                d->base + (size_t)t * n_rows, d->fold_b, d->R, d->shortlist_scores);
            k::argmax_sum(d->shortlist_scores, nullptr, nullptr, shortlist_n, d->toks + t + 1, d->s);
            shortlist_pick<<<1, 1, 0, d->s>>>(d->toks + t + 1, ids);
        } else {
            k::argmax_gemv_f16(d->fold_w, n_rows, d->latent, d->base + (size_t) t * n_rows, d->fold_b, d->R, n_vocab,
                           d->toks + t + 1, d->s);
        }
    }
    cu(cudaMemcpyAsync(d->h_out, d->toks + 1, proposals * 4, cudaMemcpyDeviceToHost, d->s), "tokens");
    cu(cudaStreamSynchronize(d->s), "propose");
    std::memcpy(out, d->h_out, proposals * sizeof(int));
}

size_t bytes(const Drafter* d) { return d->vram; }
void set_head_q8(Drafter* d, const int8_t* q8, const uint16_t* q8d) { d->head_q8 = q8, d->head_q8d = q8d; }

}  // namespace ik2::draft
