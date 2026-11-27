// src/ik2/ik2_draft.cu - maya-ik2's drafter (the design: ik2_draft.hpp).
#include "ik2_draft.hpp"
#include "ik2_kernels.cuh"
#include "maya/artifact/gguf_reader.hpp"

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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
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
std::vector<void*>* g_track = nullptr;   // (the drafter being loaded: its allocations, which destroy() frees)
template <class T>
T* dalloc(size_t n) {
    T* p = nullptr;
    cu(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)), "cudaMalloc");
    if (g_track) g_track->push_back(p);
    return p;
}
struct Track {   // (allocations in this scope: into t, or none - a temporary freed in the scope)
    std::vector<void*>* prev;
    explicit Track(std::vector<void*>* t) : prev(g_track) { g_track = t; }
    ~Track() { g_track = prev; }
};
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

// ---- the drafter's file: thinkingmachines' mtp.safetensors (BF16), or a GGUF of it - llama.cpp's "dflash" arch,
// e.g. unsloth's dspark-Inkling-2-Mini-Q8_0.gguf: the same tensors renamed (convert_hf_to_gguf's Inkling2DSparkModel),
// the layers' matrices Q8_0 (kept planar here, for q8_tile), the norms, biases and fuse weights F32, the Markov head
// BF16 as the original's.  Tensors are asked for by their safetensors name and shape ([out][in]).
float h2f_host(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 31) u = s | 0x7f800000u | (m << 13);
    else if (e) u = s | ((e + 112) << 23) | (m << 13);
    else if (!m) u = s;
    else {   // (subnormal)
        int k = -1;
        uint32_t mm = m;
        do { mm <<= 1, ++k; } while (!(mm & 1024));
        u = s | ((uint32_t) (112 - k) << 23) | ((mm & 1023) << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f2bf_host(float f) {   // (round to nearest even: exact for a value that was BF16)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}

struct Src {
    StFile st;                              // mtp.safetensors
    std::unique_ptr<maya::GgufFile> g;    // or the GGUF
    bool verbatim = false;                  // (a GGUF that keeps the safetensors names, under "mtp.": arch inkling2-mtp)
};

bool is_gguf(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("drafter: cannot open " + path);
    char m[4] = {};
    const bool yes = std::fread(m, 1, 4, fp) == 4 && std::memcmp(m, "GGUF", 4) == 0;
    std::fclose(fp);
    return yes;
}

std::string gguf_name(const std::string& n, bool verbatim) {
    if (verbatim) return "mtp." + n;
    static const std::map<std::string, std::string> top = {
        {"fuse_weight", "dspark_fuse.weight"},
        {"mask_embed", "mask_embd.weight"},
        {"norm.weight", "output_norm.weight"},
        {"markov_head.dflare_norm.weight", "dspark_norm.weight"},
        {"markov_head.markov_w1.weight", "markov_w1.weight"},
        {"markov_head.rnn_proj.weight", "joint_proj.weight"},
        {"markov_head.rnn_proj.bias", "joint_proj.bias"},
        {"markov_head.hidden_out.weight", "dspark_hidden_out.weight"},
        {"markov_head.hidden_out.bias", "dspark_hidden_out.bias"}};
    static const std::map<std::string, std::string> layer = {
        {"input_layernorm.weight", "attn_norm.weight"},
        {"post_attention_layernorm.weight", "ffn_norm.weight"},
        {"dflare_norm.weight", "dflare_norm.weight"},
        {"self_attn.q_norm.weight", "attn_q_norm.weight"},
        {"self_attn.k_norm.weight", "attn_k_norm.weight"},
        {"self_attn.k_norm_ctx.weight", "attn_k_ctx_norm.weight"},
        {"self_attn.q_proj.weight", "attn_q.weight"},
        {"self_attn.k_proj.weight", "attn_k.weight"},
        {"self_attn.v_proj.weight", "attn_v.weight"},
        {"self_attn.o_proj.weight", "attn_output.weight"},
        {"self_attn.k_proj_ctx.weight", "attn_k_ctx.weight"},
        {"self_attn.v_proj_ctx.weight", "attn_v_ctx.weight"},
        {"mlp.gate_proj.weight", "ffn_gate.weight"},
        {"mlp.up_proj.weight", "ffn_up.weight"},
        {"mlp.down_proj.weight", "ffn_down.weight"}};
    if (const auto it = top.find(n); it != top.end()) return it->second;
    if (n.rfind("layers.", 0) == 0) {
        const size_t dot = n.find('.', 7);
        if (dot != std::string::npos)
            if (const auto it = layer.find(n.substr(dot + 1)); it != layer.end())
                return "blk." + n.substr(7, dot - 7) + "." + it->second;
    }
    throw std::runtime_error("drafter: no GGUF name for " + n);
}

// the GGUF tensor of a safetensors name, its shape checked (GGUF dims run the other way: [in, out])
const maya::TensorInfo& g_tensor(const Src& s, const std::string& n, const std::vector<int64_t>& shape) {
    const std::string gn = gguf_name(n, s.verbatim);
    const maya::TensorInfo* t = s.g->find(gn);
    if (!t) throw std::runtime_error("drafter: no tensor " + gn + " in the GGUF");
    bool same = t->shape.size() == shape.size();
    for (size_t i = 0; same && i < shape.size(); ++i) same = (int64_t) t->shape[shape.size() - 1 - i] == shape[i];
    if (!same) throw std::runtime_error("drafter: " + gn + ": unexpected shape");
    return *t;
}

// the first dimension (safetensors order) of a tensor
int64_t src_dim0(const Src& s, const std::string& n) {
    if (s.g) {
        const maya::TensorInfo* t = s.g->find(gguf_name(n, s.verbatim));
        if (!t || t->shape.size() != 2) throw std::runtime_error("drafter: no 2-D " + gguf_name(n, s.verbatim));
        return (int64_t) t->shape[1];
    }
    const auto it = s.st.t.find(n);
    if (it == s.st.t.end() || it->second.shape.size() != 2) throw std::runtime_error("drafter: no 2-D " + n);
    return it->second.shape[0];
}

// a tensor's values as F32 (BF16, F16 or F32 in the file; a Q8_0 one as its values times their scales)
std::vector<float> src_f32(const Src& s, const std::string& n, const std::vector<int64_t>& shape) {
    if (!s.g) {
        const auto b = st_bf16(s.st, n, shape);
        std::vector<float> v(b.size());
        for (size_t i = 0; i < b.size(); ++i) v[i] = bf(b[i]);
        return v;
    }
    const auto& t = g_tensor(s, n, shape);
    const uint8_t* p = s.g->tensor_data(t);
    std::vector<float> v((size_t) t.elements());
    if (t.type == 0) {
        std::memcpy(v.data(), p, v.size() * 4);
    } else if (t.type == 1 || t.type == 30) {
        for (size_t i = 0; i < v.size(); ++i) {
            const uint16_t h = (uint16_t) (p[2 * i] | (p[2 * i + 1] << 8));
            v[i] = t.type == 1 ? h2f_host(h) : bf(h);
        }
    } else if (t.type == 8) {   // Q8_0: 2 bytes of FP16 scale, 32 int8 values
        for (size_t b = 0; b < v.size() / 32; ++b, p += 34) {
            const float sc = h2f_host((uint16_t) (p[0] | (p[1] << 8)));
            for (int j = 0; j < 32; ++j) v[b * 32 + j] = sc * (float) (int8_t) p[2 + j];
        }
    } else {
        throw std::runtime_error("drafter: " + t.name + " is " + t.type_name() + " (BF16, F16, F32 or Q8_0 read)");
    }
    return v;
}

// ... as BF16 (the original's BF16 tensors as they are; others rounded)
std::vector<uint16_t> src_bf16(const Src& s, const std::string& n, const std::vector<int64_t>& shape) {
    if (!s.g) return st_bf16(s.st, n, shape);
    const auto& t = g_tensor(s, n, shape);
    if (t.type == 30) {
        std::vector<uint16_t> v((size_t) t.elements());
        std::memcpy(v.data(), s.g->tensor_data(t), v.size() * 2);
        return v;
    }
    const auto f = src_f32(s, n, shape);
    std::vector<uint16_t> v(f.size());
    for (size_t i = 0; i < f.size(); ++i) v[i] = f2bf_host(f[i]);
    return v;
}

// a GGUF of the drafter: llama.cpp's dflash metadata checked against what this drafter is (a file of another model
// or another layout stops the load instead of drafting garbage)
void check_gguf(const maya::GgufFile& g, int C, int layers, int H, int KV, int F, int taps, const std::vector<bool>& local) {
    auto num = [&](const char* key) -> double {
        const auto* v = g.get(key);
        if (!v || !v->is_num()) throw std::runtime_error(std::string("drafter: the GGUF has no ") + key);
        return v->num();
    };
    const auto* arch = g.get("general.architecture");
    if (arch && arch->s == "inkling2-mtp") {   // (the original's config under its own names, the tensors BF16)
        auto want = [&](const char* key, double v) {
            if (num(key) != v)
                throw std::runtime_error(std::string("drafter: the GGUF's ") + key + " is " + std::to_string(num(key)) +
                                         ", this model's drafter has " + std::to_string(v));
        };
        want("inkling2-mtp.block_size", kBlock);
        want("inkling2-mtp.hidden_size", C);
        want("inkling2-mtp.num_hidden_layers", layers);
        want("inkling2-mtp.num_attention_heads", H);
        want("inkling2-mtp.num_key_value_heads", KV);
        want("inkling2-mtp.intermediate_size", F);
        want("inkling2-mtp.head_dim", 128);
        want("inkling2-mtp.sliding_window_size", 1024);
        want("inkling2-mtp.rope_theta", 1e6);
        want("inkling2-mtp.markov_rank", 256);
        if (std::fabs(num("inkling2-mtp.rms_norm_eps") - 1e-6) > 1e-9)
            throw std::runtime_error("drafter: the GGUF's RMS epsilon is not 1e-6");
        std::vector<uint64_t> expect;
        if (taps == 4) expect = {5, 23, 41, 48};
        else for (int l : {5, 11, 17, 23, 29, 35, 41, 48}) expect.push_back((uint64_t) l);
        const auto* tl = g.get("inkling2-mtp.target_layer_ids");
        bool ok = tl && tl->items.size() == expect.size();
        for (size_t i = 0; ok && i < expect.size(); ++i) ok = tl->items[i].u == expect[i];
        if (!ok) throw std::runtime_error("drafter: the GGUF's target_layer_ids are not the taps this engine gives");
        const auto* loc = g.get("inkling2-mtp.local_layer_ids");
        std::vector<bool> is_local(local.size(), false);
        ok = loc != nullptr;
        for (size_t i = 0; ok && i < loc->items.size(); ++i) {
            ok = loc->items[i].u < local.size();
            if (ok) is_local[loc->items[i].u] = true;
        }
        if (!ok || is_local != local) throw std::runtime_error("drafter: the GGUF's local_layer_ids are not this drafter's");
        for (const auto& t : g.tensors())
            if (t.type != 30) throw std::runtime_error("drafter: " + t.name + " is " + t.type_name() + " (an inkling2-mtp GGUF is read as BF16)");
        return;
    }
    if (!arch || arch->s != "dflash") throw std::runtime_error("drafter: the GGUF is not a DSpark drafter (architecture " +
                                                               (arch ? arch->s : std::string("?")) + ", dflash expected)");
    if (!g.find("dspark_hidden_out.weight"))
        throw std::runtime_error("drafter: the GGUF is not Inkling 2's DSpark (no dspark_hidden_out)");
    auto want = [&](const char* key, double v) {
        if (num(key) != v)
            throw std::runtime_error(std::string("drafter: the GGUF's ") + key + " is " + std::to_string(num(key)) +
                                     ", this model's drafter has " + std::to_string(v));
    };
    want("dflash.block_size", kBlock);
    want("dflash.embedding_length", C);
    want("dflash.block_count", layers);
    want("dflash.attention.head_count", H);
    want("dflash.attention.head_count_kv", KV);
    want("dflash.feed_forward_length", F);
    want("dflash.attention.key_length", 128);
    want("dflash.attention.sliding_window", 1024);
    want("dflash.rope.freq_base", 1e6);
    if (std::fabs(num("dflash.attention.layer_norm_rms_epsilon") - 1e-6) > 1e-9)
        throw std::runtime_error("drafter: the GGUF's RMS epsilon is not 1e-6");
    const auto* tl = g.get("dflash.target_layers");
    // (the target's taps: a looped model's visits 5, 23, 41 and its final hidden - 48 visits; Inkling-2-Small's
    //  layers 5, 11, .., 41 and its final hidden)
    std::vector<uint64_t> expect;
    if (taps == 4) expect = {5, 23, 41, 48};
    else for (int l : {5, 11, 17, 23, 29, 35, 41, 48}) expect.push_back((uint64_t) l);
    bool ok = tl && tl->items.size() == expect.size();
    for (size_t i = 0; ok && i < expect.size(); ++i) ok = tl->items[i].u == expect[i];
    if (!ok) throw std::runtime_error("drafter: the GGUF's dflash.target_layers are not the taps this engine gives");
    const auto* pat = g.get("dflash.attention.sliding_window_pattern");
    ok = pat && pat->items.size() == local.size();
    for (size_t i = 0; ok && i < local.size(); ++i) ok = (pat->items[i].u != 0) == local[i];
    if (!ok) throw std::runtime_error("drafter: the GGUF's sliding window pattern is not this drafter's");
}

}  // namespace

// a layer's matrix [out][in]: BF16 (cuBLAS, BF16 inputs), or a GGUF's Q8_0 kept planar - q its int8 values, qd its
// blocks' FP16 scales - for q8_tile (the tensor cores, FP16 inputs)
struct DMat {
    uint16_t* bf = nullptr;
    int8_t* q = nullptr;
    uint16_t* qd = nullptr;
};

struct DLayer {
    bool local = false;
    float *in_norm, *post_norm, *dflare, *q_norm, *k_norm, *k_norm_ctx;
    DMat wq, wk, wv, wo, wkc, wvc, wgu, wd;
    float *kc = nullptr, *vc = nullptr;                    // the context caches, F32 [cap][KV * 128] (a window's
    uint16_t *kc16 = nullptr, *vc16 = nullptr;             //  layer) or FP16 (the layer that sees the whole context)
    uint8_t *kc8 = nullptr, *vc8 = nullptr;                //  ... or FP8 E4M3 (load's kv8), times k_scale / v_scale
    int cap = 0;                                           // their rows: a ring of the last cap positions
};

struct Drafter {
    int taps = 8, C = 4608, H = 48, KV = 8, F = 18432, R = 256, V = 0, ctx = 0;
    int cap = 0;   // the rows of the layer without a window's context cache (IK2_DRAFT_CTX, default the context)
    float* ctx_tmp = nullptr;   // write_context's K and V rows before an FP16 cache, [2][kMixRows][KV * 128]
    float eps = 1e-6f, theta = 1e6f;
    bool q8 = false;   // the layers' matrices Q8_0 (a GGUF's): their inputs FP16, else BF16
    bool kv8 = false;  // the layer without a window's context rows FP8 E4M3, K times k_scale, V times v_scale
    // (K is RMS-normed a head - |k| <= sqrt(128) x its norm's weight, 12.4 at most measured - V not: 17.7 at most
    //  measured; 448 is E4M3's largest value - powers of two, so the scaling itself rounds nothing)
    float k_scale = 16.0f, v_scale = 4.0f;
    std::vector<void*> mem;   // its device allocations (destroy)
    std::vector<DLayer> L;
    float *alpha, *norm, *mask, *mk_dflare, *rnn_b;
    uint16_t *w1, *rnn_w;
    std::vector<uint16_t> hout_w;   // host BF16, for the fold
    std::vector<float> hout_b;      // host, for the fold (the original's BF16 values)
    uint16_t *fold_w = nullptr, *fold_b = nullptr;   // FP16 [R][n_rows], [n_rows]
    // the fold as int8 [R][n_rows], a row's scale apart (fold_sc [R]; xs = latent * fold_sc a draft): half the bytes
    // a draft reads - fold_w is then gone after the load.  IK2_DRAFT_FOLD=f16: the FP16 table
    int8_t* fold_q = nullptr;
    float *fold_sc = nullptr, *fold_xs = nullptr;
    int n_rows = 0;
    const int8_t* head_q8 = nullptr;      // the target's head as planar Q8_0 (set_head_q8): the base logits through it
    const uint16_t* head_q8d = nullptr;
    bool head_tiles = true;               // its product on the tensor cores (q8_tile: sm_80 on), else mv_q8p_cols
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
    // a step's launches as a recorded CUDA graph, one a draft count (a GGUF drafter - no cuBLAS call in it): the
    // block's position and anchor go through par (pinned par_h copied in by the graph), the attention reads them there
    int* par = nullptr;     // device [8]: P, anchor, the noise's salt (2), the bits of 1 / temperature
    int* par_h = nullptr;   // pinned host
    float noise_cap = 30.0f;
    cudaGraphExec_t graph[kBlock] = {};
    const float* graph_anchor = nullptr;   // (the anchor row the graphs were recorded with)
    int graph_rows = 0, graph_vocab = 0;
    float graph_cap = 30.0f;   // (the softcap recorded in the graphs' launches)
};

namespace {

constexpr int kMixRows = 512;
constexpr int kFoldChunk = 4096;

__global__ void iota_kernel(int* out, int start, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = start + i;
}

__global__ void init_block_kernel(const float* anchor_h, const float* mask, float* res, int* pos, int* toks, int T,
                                  int C, int P, int anchor, const int* par) {
    if (par) P = par[0], anchor = par[1];   // (a recorded graph: read here)
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
__global__ void scale_rows_kernel(const float* x, const float* sc, int n, float* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * sc[i];
}

uint16_t* up16(Drafter* d, const std::vector<uint16_t>& v) {
    uint16_t* p = dalloc<uint16_t>(v.size());
    cu(cudaMemcpy(p, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "upload");
    d->vram += v.size() * 2;
    return p;
}
float* up32f(Drafter* d, const std::vector<float>& f) {
    float* p = dalloc<float>(f.size());
    cu(cudaMemcpy(p, f.data(), f.size() * 4, cudaMemcpyHostToDevice), "upload");
    d->vram += f.size() * 4;
    return p;
}

// rows of a planar Q8_0 matrix as F32: each value times its block's scale rounded to FP16, as decode_q8p_f16 gives
// them (the fold through a Q8_0 head the same as through its FP16 copy, without the copy: 0.9 GiB for Inkling 2's)
__global__ void q8p_f32_kernel(const int8_t* q, const uint16_t* sc, size_t n, float* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__float2half_rn(__half2float(__ushort_as_half(sc[i >> 5])) * (float) q[i]));
}

// Y [T][N] (ldy) = X [T][K] BF16 . W [N][K]^T BF16, F32 sums
void gemm_bf(Drafter* d, const uint16_t* W, int N, int K, const uint16_t* X, int T, float* Y, int ldy) {
#ifdef MAYA_USE_HIP   // (AMD: the BLAS's kernels for 8 rows read the weights at a fifth of the bus - the drafter's
                        //  block on a kernel of ours, the weights read once; the drafts only, never an answer)
    if (k::gemm_bf16_cols(W, N, K, X, T, Y, ldy, d->s)) return;
#endif
    const float one = 1.0f, zero = 0.0f;
    cb(cublasGemmEx(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "gemm_bf");
}
void gemm_h(Drafter* d, const uint16_t* W, int N, int K, const void* X, int T, float* Y, int ldy) {
#ifdef MAYA_USE_HIP   // (AMD: no FP16-in F32-out product in the BLAS - the kernel of gemm_bf, reading FP16; drafts only)
    if (k::gemm_f16_cols(W, N, K, (const uint16_t*) X, T, Y, ldy, d->s)) return;
#endif
    const float one = 1.0f, zero = 0.0f;
    cb(cublasGemmEx(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16F, K, X, CUDA_R_16F, K, &zero, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "gemm_h");
}
// Y [T][N] (ldy) = X [T][K] . W^T, X as act() made it: BF16 for a BF16 matrix, FP16 for a Q8_0 one
void gemm(Drafter* d, const DMat& W, int N, int K, const uint16_t* X, int T, float* Y, int ldy) {
    if (W.q) k::q8_tile(W.q, W.qd, N, K, X, T, Y, ldy, d->s);
    else gemm_bf(d, W.bf, N, K, X, T, Y, ldy);
}
void act(Drafter* d, const float* x, size_t n, uint16_t* y) {
    if (d->q8) k::to_f16(x, n, y, d->s);
    else k::to_bf16(x, n, y, d->s);
}

// y [N] = W [N][K] . x [K], both BF16, F32 sums - the Markov head's projection as a kernel of ours (a warp a row, a
// lane 8 values a step), so that a step's launches can be a recorded graph.  K a multiple of 256.
__global__ void rnn_proj_bf_kernel(const uint16_t* __restrict__ W, const uint16_t* __restrict__ x, int N, int K,
                                   float* __restrict__ y) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint4* w4 = (const uint4*) (W + (size_t) row * K) + lane;
    const uint4* x4 = (const uint4*) x + lane;
    float acc = 0.0f;
#pragma unroll 4
    for (int i = 0, n = K >> 8; i < n; ++i) {
        const uint4 w = __ldg(w4 + i * 32), v = __ldg(x4 + i * 32);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w}, vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc = fmaf(__uint_as_float((ww[j] & 0xFFFFu) << 16), __uint_as_float((vv[j] & 0xFFFFu) << 16), acc);
            acc = fmaf(__uint_as_float(ww[j] & 0xFFFF0000u), __uint_as_float(vv[j] & 0xFFFF0000u), acc);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

void layer_attn(Drafter* d, DLayer& l, int P, float* out, bool dyn = false) {
    const int T = kBlock, HD = d->H * 128, KD = d->KV * 128;
    act(d, d->h, (size_t) T * d->C, d->hb);
    gemm(d, l.wq, HD, d->C, d->hb, T, d->q, HD);
    gemm(d, l.wk, KD, d->C, d->hb, T, d->kk, KD);
    gemm(d, l.wv, KD, d->C, d->hb, T, d->vv, KD);
    k::rms(d->q, l.q_norm, T * d->H, 128, d->eps, d->q, d->s);
    k::rms(d->kk, l.k_norm, T * d->KV, 128, d->eps, d->kk, d->s);
    k::rope_neox(d->q, T, d->H, d->pos, d->theta, d->s);
    k::rope_neox(d->kk, T, d->KV, d->pos, d->theta, d->s);
    // Local layers (0, 1, 3, 4): a window of 1024 positions anchored at the block's start -
    // every query in the block [P .. P+7] sees context [max(0, P - 1023) .. P-1] plus all 8 block tokens.
    const int k_lo = l.local ? std::max(0, P - 1023) : std::max(0, P - l.cap);
    const float sc = 1.0f / std::sqrt(128.0f);
    if (dyn) {   // (a recorded graph: the position from d->par, the keys the layer's span before it)
        const int span = l.local ? 1023 : l.cap;
        if (l.kc8)
            k::draft_attn_dyn_fp8(d->q, l.kc8, l.vc8, d->par, span, d->kk, d->vv, T, d->H, d->KV, sc, 1.0f / d->k_scale,
                                  1.0f / d->v_scale, d->scratch, d->o, d->s, l.cap);
        else if (l.kc16)
            k::draft_attn_dyn_f16(d->q, l.kc16, l.vc16, d->par, span, d->kk, d->vv, T, d->H, d->KV, sc, d->scratch, d->o,
                                  d->s, l.cap);
        else
            k::draft_attn_dyn(d->q, l.kc, l.vc, d->par, span, d->kk, d->vv, T, d->H, d->KV, sc, d->scratch, d->o, d->s,
                              l.cap);
    } else if (l.kc8)
        k::draft_attn_fp8(d->q, l.kc8, l.vc8, k_lo, P, d->kk, d->vv, T, d->H, d->KV, 1.0f / std::sqrt(128.0f),
                          1.0f / d->k_scale, 1.0f / d->v_scale, d->scratch, d->o, d->s, l.cap);
    else if (l.kc16)
        k::draft_attn_f16(d->q, l.kc16, l.vc16, k_lo, P, d->kk, d->vv, T, d->H, d->KV, 1.0f / std::sqrt(128.0f),
                          d->scratch, d->o, d->s, l.cap);
    else
        k::draft_attn(d->q, l.kc, l.vc, k_lo, P, d->kk, d->vv, T, d->H, d->KV, 1.0f / std::sqrt(128.0f), d->scratch,
                      d->o, d->s, l.cap);
    act(d, d->o, (size_t) T * HD, d->hb);
    gemm(d, l.wo, d->C, HD, d->hb, T, out, d->C);
}

void layer_mlp(Drafter* d, DLayer& l, float* out) {
    const int T = kBlock;
    act(d, d->h, (size_t) T * d->C, d->hb);
    gemm(d, l.wgu, 2 * d->F, d->C, d->hb, T, d->gu, 2 * d->F);
    if (d->q8) k::silu_mul_f16(d->gu, T, d->F, d->hb2, d->s);
    else k::silu_mul_bf16(d->gu, T, d->F, d->hb2, d->s);
    gemm(d, l.wd, d->C, d->F, d->hb2, T, out, d->C);
}

// a matrix from the file: [rows][K], the rows of `names` one after another (gate's then up's).  Q8_0 in a GGUF:
// planar on the device when the drafter reads Q8_0 (d->q8), else its values in BF16, as a BF16 file's.
DMat load_mat(Drafter* d, const Src& s, const std::vector<std::string>& names, const std::vector<int64_t>& rows,
              int64_t K) {
    DMat m;
    if (d->q8) {
        size_t n = 0;
        for (int64_t r : rows) n += (size_t) r * K;
        std::vector<int8_t> q(n);
        std::vector<uint16_t> sc(n / 32);
        size_t b = 0;
        for (size_t i = 0; i < names.size(); ++i) {
            const auto& t = g_tensor(s, names[i], {rows[i], K});
            if (t.type != 8) throw std::runtime_error("drafter: " + t.name + " is " + t.type_name() + ", Q8_0 expected");
            const uint8_t* p = s.g->tensor_data(t);
            for (size_t j = 0; j < (size_t) rows[i] * K / 32; ++j, ++b, p += 34) {
                std::memcpy(&sc[b], p, 2);
                std::memcpy(&q[b * 32], p + 2, 32);
            }
        }
        m.q = dalloc<int8_t>(n);
        m.qd = dalloc<uint16_t>(n / 32);
        cu(cudaMemcpy(m.q, q.data(), n, cudaMemcpyHostToDevice), "upload");
        cu(cudaMemcpy(m.qd, sc.data(), n / 32 * 2, cudaMemcpyHostToDevice), "upload");
        d->vram += n + n / 16;
        return m;
    }
    std::vector<uint16_t> v;
    for (size_t i = 0; i < names.size(); ++i) {
        const auto part = src_bf16(s, names[i], {rows[i], K});
        v.insert(v.end(), part.begin(), part.end());
    }
    m.bf = up16(d, v);
    return m;
}

}  // namespace

const std::vector<int>& tap_layers() {
    static const std::vector<int> v = {5, 11, 17, 23, 29, 35, 41};
    return v;
}

// (the engine's plan: the rows a long context's layout leaves room for - the last positions; 0: the whole context)
static int g_plan_rows = 0;
void plan_context_rows(int rows) { g_plan_rows = rows; }

int context_rows(int ctx) {
    // (IK2_DRAFT_CTX: the positions the layer without a window sees, the last ones; default the plan's, else the
    //  whole context)
    const int want = getenv("IK2_DRAFT_CTX") ? std::atoi(getenv("IK2_DRAFT_CTX")) : g_plan_rows > 0 ? g_plan_rows : ctx;
    return std::min(ctx, std::max(want, 2048));
}

Drafter* load(const std::string& path, int C, int ctx, cublasHandle_t blas, cudaStream_t s, bool kv8) {
    Src f;
    if (is_gguf(path)) f.g = std::make_unique<maya::GgufFile>(path);
    else f.st = st_open(path);
    auto* d = new Drafter;
    Track track(&d->mem);
    d->kv8 = kv8;
    d->C = C, d->ctx = ctx, d->blas = blas, d->s = s;
    d->cap = context_rows(ctx);
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
    const int local_ids[] = {0, 1, 3, 4};
    std::vector<bool> local(layers);
    for (int i = 0; i < layers; ++i)
        local[i] = mini ? i != 1 : std::find(std::begin(local_ids), std::end(local_ids), i) != std::end(local_ids);
    if (f.g) {
        const auto* arch = f.g->get("general.architecture");
        f.verbatim = arch && arch->s == "inkling2-mtp";
        check_gguf(*f.g, C, layers, d->H, d->KV, d->F, d->taps, local);
        // (the layers' matrices on the tensor cores straight from their Q8_0 blocks - q8_tile, sm_75 on; else their
        //  values in BF16, as mtp.safetensors' are read)
        bool all_q8 = true;
        for (const auto& t : f.g->tensors())
            if (t.name.rfind("blk.", 0) == 0 && t.shape.size() == 2) all_q8 = all_q8 && t.type == 8;
        d->q8 = all_q8 && !f.verbatim && k::arch_class() >= 75;
    }
    d->V = (int) src_dim0(f, "markov_head.markov_w1.weight");
    // fuse_weight [layers + 1][taps] -> softmax over each row
    {
        const auto fw = src_f32(f, "fuse_weight", {layers + 1, d->taps});
        std::vector<float> a(fw.size());
        for (int r = 0; r <= layers; ++r) {
            float m = -1e30f, sum = 0;
            for (int t = 0; t < d->taps; ++t) m = std::max(m, fw[r * d->taps + t]);
            for (int t = 0; t < d->taps; ++t) sum += (a[r * d->taps + t] = std::exp(fw[r * d->taps + t] - m));
            for (int t = 0; t < d->taps; ++t) a[r * d->taps + t] /= sum;
        }
        d->alpha = dalloc<float>(a.size());
        cu(cudaMemcpy(d->alpha, a.data(), a.size() * 4, cudaMemcpyHostToDevice), "alpha");
    }
    auto vec = [&](const std::string& n, int64_t len) { return up32f(d, src_f32(f, n, {len})); };
    d->L.resize(layers);
    for (int i = 0; i < layers; ++i) {
        DLayer& l = d->L[i];
        const std::string b = "layers." + std::to_string(i) + ".";
        l.local = local[i];
        l.in_norm = vec(b + "input_layernorm.weight", C);
        l.post_norm = vec(b + "post_attention_layernorm.weight", C);
        l.dflare = vec(b + "dflare_norm.weight", C);
        l.q_norm = vec(b + "self_attn.q_norm.weight", 128);
        l.k_norm = vec(b + "self_attn.k_norm.weight", 128);
        l.k_norm_ctx = vec(b + "self_attn.k_norm_ctx.weight", 128);
        l.wq = load_mat(d, f, {b + "self_attn.q_proj.weight"}, {H * 128}, C);
        l.wk = load_mat(d, f, {b + "self_attn.k_proj.weight"}, {KV * 128}, C);
        l.wv = load_mat(d, f, {b + "self_attn.v_proj.weight"}, {KV * 128}, C);
        l.wo = load_mat(d, f, {b + "self_attn.o_proj.weight"}, {C}, H * 128);
        l.wkc = load_mat(d, f, {b + "self_attn.k_proj_ctx.weight"}, {KV * 128}, C);
        l.wvc = load_mat(d, f, {b + "self_attn.v_proj_ctx.weight"}, {KV * 128}, C);
        l.wgu = load_mat(d, f, {b + "mlp.gate_proj.weight", b + "mlp.up_proj.weight"}, {F, F}, C);   // gate's, up's
        l.wd = load_mat(d, f, {b + "mlp.down_proj.weight"}, {C}, F);
        if (l.local) {   // (a window of 1024: a ring of 2048 rows, F32)
            l.cap = std::min(ctx, 2048);
            l.kc = dalloc<float>((size_t) l.cap * KV * 128);
            l.vc = dalloc<float>((size_t) l.cap * KV * 128);
            d->vram += (size_t) l.cap * KV * 128 * 8;
        } else if (kv8) {   // (the whole context, or IK2_DRAFT_CTX's last positions: FP8 - 2 KiB a position)
            l.cap = d->cap;
            l.kc8 = dalloc<uint8_t>((size_t) l.cap * KV * 128);
            l.vc8 = dalloc<uint8_t>((size_t) l.cap * KV * 128);
            d->vram += (size_t) l.cap * KV * 128 * 2;
        } else {            // (... FP16 - 4 KiB a position)
            l.cap = d->cap;
            l.kc16 = dalloc<uint16_t>((size_t) l.cap * KV * 128);
            l.vc16 = dalloc<uint16_t>((size_t) l.cap * KV * 128);
            d->vram += (size_t) l.cap * KV * 128 * 4;
        }
    }
    d->norm = vec("norm.weight", C);
    d->mask = vec("mask_embed", C);
    d->mk_dflare = vec("markov_head.dflare_norm.weight", C);
    d->w1 = up16(d, src_bf16(f, "markov_head.markov_w1.weight", {d->V, R}));
    d->rnn_w = up16(d, src_bf16(f, "markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * C}));
    d->rnn_b = vec("markov_head.rnn_proj.bias", 3 * R);
    d->hout_w = src_bf16(f, "markov_head.hidden_out.weight", {C, R});
    d->hout_b = src_f32(f, "markov_head.hidden_out.bias", {C});
    // work buffers
    const size_t T = kBlock;
    d->x = dalloc<float>(T * C), d->res = dalloc<float>(T * C), d->h = dalloc<float>(T * C);
    d->q = dalloc<float>(T * H * 128), d->o = dalloc<float>(T * H * 128);
    d->kk = dalloc<float>(std::max<size_t>(T, kMixRows) * KV * 128), d->vv = dalloc<float>(T * KV * 128);
    d->gu = dalloc<float>(T * 2 * F);
    d->hb = dalloc<uint16_t>(std::max<size_t>(T * std::max<int64_t>(C, H * 128), (size_t) kMixRows * C));
    d->hb2 = dalloc<uint16_t>(T * F);
    // (draft_attn's: the chunks' outputs [64][H][8][128] and (max, sum) - nothing that grows with the context)
    d->ctx_tmp = dalloc<float>((size_t) 2 * kMixRows * KV * 128);
    d->scratch = dalloc<float>(k::draft_attn_scratch((int) H));
    d->mix = dalloc<float>((size_t) kMixRows * C);
    d->f_rnn = dalloc<float>(C);
    d->state = dalloc<float>(R), d->latent = dalloc<float>(R), d->proj = dalloc<float>(3 * R);
    d->rnn_in = dalloc<uint16_t>(2 * R + 2 * C);
    d->x16 = dalloc<__half>((kBlock - 1) * (size_t) C);
    d->pos = dalloc<int>(T), d->ctx_pos = dalloc<int>(kMixRows), d->toks = dalloc<int>(kBlock);
    cu(cudaHostAlloc(&d->h_out, (kBlock - 1) * sizeof(int), cudaHostAllocDefault), "h_out");
    d->par = dalloc<int>(8);
    cu(cudaHostAlloc(&d->par_h, 8 * sizeof(int), cudaHostAllocDefault), "par_h");
    std::memset(d->par_h, 0, 8 * sizeof(int));
    return d;
}

namespace {
// the fold, the head's rows a chunk at a time as F32 from lm_f16 (FP16 [n_rows][C]) or from planar Q8_0 (q8, q8d)
void fold_impl(Drafter* d, const uint16_t* lm_f16, const int8_t* q8, const uint16_t* q8d, int n_rows) {
    const int C = d->C, R = d->R;
    d->n_rows = n_rows;
    if (getenv("IK2_DRAFT_SHORTLIST") && atoi(getenv("IK2_DRAFT_SHORTLIST"))) {
        const int stride = (n_rows + 31) / 32;
        Track kept(&d->mem);
        d->shortlist = dalloc<int>((kBlock - 1) * stride);
        d->shortlist_scores = dalloc<float>(stride);
    }
    // fold_w [R][n_rows] = (lm . hidden_out)^T, fold_b [n_rows] = lm . b (stored as FP16 to halve Markov traffic).
    // Compute chunked in F32 (like inkling2_mtp.py's 4096-row double() fold) so BF16 hout_w/b never underflow FP16
    // before the C=4608 inner product, then round once to FP16.
    std::vector<float> ht((size_t) R * C), hb(C);
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r) ht[(size_t) r * C + c] = bf(d->hout_w[(size_t) c * R + r]);
    for (int c = 0; c < C; ++c) hb[c] = d->hout_b[c];
    Track temps(nullptr);   // (the fold's temporaries, freed below)
    float *dht = dalloc<float>(ht.size()), *dhb = dalloc<float>(hb.size());
    cu(cudaMemcpy(dht, ht.data(), ht.size() * sizeof(float), cudaMemcpyHostToDevice), "fold");
    cu(cudaMemcpy(dhb, hb.data(), hb.size() * sizeof(float), cudaMemcpyHostToDevice), "fold");
    const int chunk = std::min(kFoldChunk, std::max(1, n_rows));
    float* dlm = dalloc<float>((size_t) chunk * C);
    float* dfw = dalloc<float>((size_t) R * chunk);
    float* dfb = dalloc<float>(chunk);
    {
        Track kept(&d->mem);
        d->fold_w = dalloc<uint16_t>((size_t) R * n_rows);
        d->fold_b = dalloc<uint16_t>(n_rows);
        d->base = dalloc<float>((size_t) (kBlock - 1) * n_rows);
    }
    const float one = 1.0f, zero = 0.0f;
    for (int r0 = 0; r0 < n_rows; r0 += chunk) {
        const int m = std::min(chunk, n_rows - r0);
        const size_t elems = (size_t) m * C;
        if (lm_f16)
            f16_to_f32_kernel<<<(unsigned) ((elems + 255) / 256), 256, 0, d->s>>>(lm_f16 + (size_t) r0 * C, elems, dlm);
        else
            q8p_f32_kernel<<<(unsigned) ((elems + 255) / 256), 256, 0, d->s>>>(q8 + (size_t) r0 * C,
                                                                               q8d + (size_t) r0 * C / 32, elems, dlm);
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
    const bool want_i8 = !(getenv("IK2_DRAFT_FOLD") && std::string(getenv("IK2_DRAFT_FOLD")) == "f16");
    if (want_i8 && !d->shortlist && n_rows % 8 == 0) {
        // (each row over the vocabulary rounded to int8 of its largest value: the drafts only)
        std::vector<uint16_t> hw((size_t) R * n_rows);
        cu(cudaMemcpy(hw.data(), d->fold_w, hw.size() * 2, cudaMemcpyDeviceToHost), "fold (int8)");
        std::vector<int8_t> hq(hw.size());
        std::vector<float> sc(R);
        for (int r = 0; r < R; ++r) {
            const uint16_t* row = hw.data() + (size_t) r * n_rows;
            float mx = 0.0f;
            for (int v = 0; v < n_rows; ++v) mx = std::max(mx, std::fabs(h2f_host(row[v])));
            const float scale = mx > 0.0f ? mx / 127.0f : 1.0f;
            sc[r] = scale;
            int8_t* q = hq.data() + (size_t) r * n_rows;
            for (int v = 0; v < n_rows; ++v)
                q[v] = (int8_t) std::lrintf(std::max(-127.0f, std::min(127.0f, h2f_host(row[v]) / scale)));
        }
        {
            Track kept(&d->mem);
            d->fold_q = dalloc<int8_t>(hq.size());
            d->fold_sc = dalloc<float>(R), d->fold_xs = dalloc<float>(R);
        }
        cu(cudaMemcpy(d->fold_q, hq.data(), hq.size(), cudaMemcpyHostToDevice), "fold (int8)");
        cu(cudaMemcpy(d->fold_sc, sc.data(), (size_t) R * 4, cudaMemcpyHostToDevice), "fold (int8)");
        for (auto it = d->mem.begin(); it != d->mem.end(); ++it)
            if (*it == d->fold_w) {
                d->mem.erase(it);
                break;
            }
        cudaFree(d->fold_w);
        d->fold_w = nullptr;
        d->vram -= (size_t) R * n_rows;
    }
}
}  // namespace

void fold(Drafter* d, const uint16_t* lm_f16, int n_rows) { fold_impl(d, lm_f16, nullptr, nullptr, n_rows); }
void fold_q8(Drafter* d, const int8_t* q8, const uint16_t* q8d, int n_rows) { fold_impl(d, nullptr, q8, q8d, n_rows); }

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
            act(d, d->mix, (size_t) n * C, d->hb);
            const bool packed = l.kc16 || l.kc8;   // (F32 rows made in ctx_tmp, then stored as FP16 / FP8)
            float* kd = packed ? d->ctx_tmp : l.kc + (size_t) at * KD;
            float* vd = packed ? d->ctx_tmp + (size_t) kMixRows * KD : l.vc + (size_t) at * KD;
            gemm(d, l.wkc, KD, C, d->hb, n, kd, KD);
            gemm(d, l.wvc, KD, C, d->hb, n, vd, KD);
            k::rms(kd, l.k_norm_ctx, n * d->KV, 128, d->eps, kd, d->s);
            k::rope_neox(kd, n, d->KV, d->ctx_pos, d->theta, d->s);
            if (l.kc16) {
                k::to_f16(kd, (size_t) n * KD, l.kc16 + (size_t) at * KD, d->s);
                k::to_f16(vd, (size_t) n * KD, l.vc16 + (size_t) at * KD, d->s);
            } else if (l.kc8) {
                k::to_e4m3(kd, (size_t) n * KD, d->k_scale, l.kc8 + (size_t) at * KD, d->s);
                k::to_e4m3(vd, (size_t) n * KD, d->v_scale, l.vc8 + (size_t) at * KD, d->s);
            }
            a += n;
        }
    }
}

void set_f_rnn(Drafter* d, const float* taps, int ld, int row) {
    k::mix_rms(taps + (size_t) row * d->C, d->taps, (size_t) ld * d->C, 1, d->C, d->alpha + d->L.size() * d->taps, d->mk_dflare,
               d->eps, d->f_rnn, d->s);
}

double g_prop_submit = 0, g_prop_wait = 0;
uint64_t g_prop_n = 0;
void propose_times(double* submit_ms, double* wait_ms) {
    *submit_ms = g_prop_n ? g_prop_submit * 1e3 / g_prop_n : 0, *wait_ms = g_prop_n ? g_prop_wait * 1e3 / g_prop_n : 0;
}
static double pnow() {   // (steady_clock: clock_gettime is not in MSVC's runtime)
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
namespace {
// a step's launches; dyn: for a recorded graph - the position and the anchor read from d->par on the device
void propose_launches(Drafter* d, int anchor, const float* anchor_h, int P, const uint16_t* lm_f16, int n_rows,
                      int n_vocab, int proposals, bool dyn) {
    const int C = d->C, T = kBlock;
    cu(cudaMemcpyAsync(d->par, d->par_h, 8 * sizeof(int), cudaMemcpyHostToDevice, d->s), "par");
    // Initialize the residual stream [anchor_h; mask x 7], block positions, and anchor token in one kernel.
    // Layer 0 reads d->res and overwrites d->x, so d->x does not need pre-filling.
    init_block_kernel<<<dim3((unsigned) ((C + 255) / 256), (unsigned) T), 256, 0, d->s>>>(
        anchor_h, d->mask, d->res, d->pos, d->toks, T, C, P, anchor, dyn ? d->par : nullptr);
    const size_t TC = (size_t) T * C;
    // pre-norm layers, the residual carried apart (res = x + res before each norm)
    for (size_t i = 0; i < d->L.size(); ++i) {
        DLayer& l = d->L[i];
        if (i > 0) k::add(d->x, d->res, TC, d->res, d->s);
        k::rms(d->res, l.in_norm, T, C, d->eps, d->h, d->s);
        layer_attn(d, l, P, d->x, dyn);
        k::add(d->x, d->res, TC, d->res, d->s);
        k::rms(d->res, l.post_norm, T, C, d->eps, d->h, d->s);
        layer_mlp(d, l, d->x);
    }
    k::add(d->x, d->res, TC, d->res, d->s);
    k::rms(d->res, d->norm, T, C, d->eps, d->h, d->s);   // the draft hidden, rows 1 .. 7 used
    // base logits of rows 1 .. 7 through the target's head
    k::to_f16(d->h + C, (size_t) proposals * C, (uint16_t*) d->x16, d->s);
    if (d->head_q8 && !d->head_tiles)
        k::mv_q8p_cols(d->head_q8, d->head_q8d, n_rows, C, d->h + C, C, proposals, d->base, n_rows, d->s);
    else if (d->head_q8)
        k::q8_tile(d->head_q8, d->head_q8d, n_rows, C, (const uint16_t*) d->x16, proposals, d->base, n_rows, d->s);
    else
        gemm_h(d, lm_f16, n_rows, C, d->x16, proposals, d->base, n_rows);
    const int shortlist_n = (n_vocab + 31) / 32;
    if (d->shortlist)
        draft_shortlist<<<dim3((n_vocab + 127) / 128, proposals), 128, 0, d->s>>>(
            d->base, n_rows, n_vocab, shortlist_n, d->shortlist);
    // the Markov head, a token at a time (note: k::rnn_gates keeps state/latent in F32, its gate/tanh steps not
    // rounded to BF16 between them; fused argmax_gemv_f16 reads FP16 fold_w/fold_b + F32 base/latent)
    cu(cudaMemsetAsync(d->state, 0, d->R * 4, d->s), "state");
    const int rk = 2 * d->R + 2 * C;
    for (int t = 0; t < proposals; ++t) {
        k::rnn_inputs(d->state, d->w1, d->toks + t, d->h + (size_t) (t + 1) * C, d->f_rnn, d->R, C, d->rnn_in, d->s);
        if (dyn)
            rnn_proj_bf_kernel<<<(unsigned) ((3 * d->R + 3) / 4), 128, 0, d->s>>>(d->rnn_w, d->rnn_in, 3 * d->R, rk,
                                                                                  d->proj);
        else
            gemm_bf(d, d->rnn_w, 3 * d->R, rk, d->rnn_in, 1, d->proj, 3 * d->R);
        k::rnn_gates(d->proj, d->rnn_b, d->R, d->state, d->latent, d->s);
        if (d->shortlist) {
            const int* ids = d->shortlist + t * shortlist_n;
            shortlist_gemv<<<(shortlist_n + 3) / 4, 128, 0, d->s>>>(ids, shortlist_n, d->fold_w, d->latent,
                d->base + (size_t)t * n_rows, d->fold_b, d->R, d->shortlist_scores);
            k::argmax_sum(d->shortlist_scores, nullptr, nullptr, shortlist_n, d->toks + t + 1, d->s);
            shortlist_pick<<<1, 1, 0, d->s>>>(d->toks + t + 1, ids);
        } else if (d->fold_q) {
            scale_rows_kernel<<<1, 256, 0, d->s>>>(d->latent, d->fold_sc, d->R, d->fold_xs);
            k::argmax_gemv_i8(d->fold_q, n_rows, d->fold_xs, d->base + (size_t) t * n_rows, d->fold_b, d->R, n_vocab,
                              d->toks + t + 1, d->s, d->par, t + 1, d->noise_cap);
        } else {
            k::argmax_gemv_f16(d->fold_w, n_rows, d->latent, d->base + (size_t) t * n_rows, d->fold_b, d->R, n_vocab,
                           d->toks + t + 1, d->s);
        }
    }
    cu(cudaMemcpyAsync(d->h_out, d->toks + 1, proposals * 4, cudaMemcpyDeviceToHost, d->s), "tokens");
}
}  // namespace

void propose(Drafter* d, int anchor, const float* anchor_h, int P, const uint16_t* lm_f16, int n_rows, int n_vocab,
             int* out, int proposals, const Noise* noise) {
    const double t_p0 = pnow();
    {   // (the launches' parameters, read on the device: the block's position and anchor, the sampled drafts' noise)
        d->par_h[0] = P, d->par_h[1] = anchor;
        const uint64_t salt = noise ? noise->salt : 0;
        const float inv_t = noise ? noise->inv_temp : 0.0f;
        d->par_h[2] = (int) (uint32_t) salt, d->par_h[3] = (int) (uint32_t) (salt >> 32);
        std::memcpy(&d->par_h[4], &inv_t, 4);
        if (noise) d->noise_cap = noise->cap;
    }
    // (IK2_DRAFT_GRAPH=0: its launches one by one.  A drafter of ~75 short launches a step: queueing them took the
    //  host 0.75 ms of a 2.7 ms step, the GPU waiting in between; recorded, a step is its kernels' 2.1 ms)
    static const bool graph_on = !(getenv("IK2_DRAFT_GRAPH") && !atoi(getenv("IK2_DRAFT_GRAPH")));
    const bool can = graph_on && d->q8 && d->head_q8 && d->head_tiles && !d->shortlist && proposals >= 1 &&
                     proposals < kBlock && (2 * d->R + 2 * d->C) % 256 == 0 &&
                     (!d->graph_anchor || (d->graph_anchor == anchor_h && d->graph_rows == n_rows &&
                                           d->graph_vocab == n_vocab)) &&
                     (!noise || !d->graph_anchor || noise->cap == d->graph_cap);
    if (can) {
        if (!d->graph[proposals]) {
            cudaGraph_t g = nullptr;
            cu(cudaStreamBeginCapture(d->s, cudaStreamCaptureModeThreadLocal), "the drafter's graph");
            propose_launches(d, anchor, anchor_h, P, lm_f16, n_rows, n_vocab, proposals, true);
            cu(cudaStreamEndCapture(d->s, &g), "the drafter's graph");
            cu(cudaGraphInstantiate(&d->graph[proposals], g, 0), "the drafter's graph");
            cudaGraphDestroy(g);
            d->graph_anchor = anchor_h, d->graph_rows = n_rows, d->graph_vocab = n_vocab, d->graph_cap = d->noise_cap;
        }
        cu(cudaGraphLaunch(d->graph[proposals], d->s), "the drafter's graph");
    } else {
        propose_launches(d, anchor, anchor_h, P, lm_f16, n_rows, n_vocab, proposals, false);
    }
    const double t_p1 = pnow();
    cu(cudaStreamSynchronize(d->s), "propose");
    g_prop_submit += t_p1 - t_p0, g_prop_wait += pnow() - t_p1, ++g_prop_n;
    std::memcpy(out, d->h_out, proposals * sizeof(int));
}

size_t bytes(const Drafter* d) { return d->vram; }
bool matrices_q8(const Drafter* d) { return d->q8; }
bool context_fp8(const Drafter* d) { return d->kv8; }
void destroy(Drafter* d) {
    if (!d) return;
    cudaStreamSynchronize(d->s);
    for (auto& g : d->graph)
        if (g) cudaGraphExecDestroy(g);
    for (void* p : d->mem) cudaFree(p);
    if (d->h_out) cudaFreeHost(d->h_out);
    if (d->par_h) cudaFreeHost(d->par_h);
    delete d;
}
void set_head_q8(Drafter* d, const int8_t* q8, const uint16_t* q8d) {
    d->head_q8 = q8, d->head_q8d = q8d;
    int dev = 0, major = 0;
    cu(cudaGetDevice(&dev), "cudaGetDevice");
    cu(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cudaDeviceGetAttribute");
#ifdef MAYA_USE_HIP
    d->head_tiles = false;        // (AMD: mma.sync is emulated on the shuffles - the row sums read the head once)
#else
    d->head_tiles = major >= 8;   // (no mma.sync m16n8k16 before Ampere)
#endif
}

namespace {
// save_context / load_context: each layer's K then V rows of positions pos - w .. pos - 1 (w = min(pos, cap)), in
// ring order from the oldest - the two pieces either side of the ring's end
template <class F>
void ctx_rows(const Drafter* d, int pos, F&& copy) {
    const size_t KD = (size_t) d->KV * 128;
    for (const auto& l : d->L) {
        const int w = std::min(pos, l.cap);
        if (w <= 0) continue;
        const int s0 = (pos - w) % l.cap, n1 = std::min(w, l.cap - s0), n2 = w - n1;
        // (rows FP8, FP16 or F32: load's kv8, the whole-context layer's FP16, a window's F32)
        const size_t row = KD * (l.kc8 ? 1 : l.kc16 ? 2 : 4);
        uint8_t* kc = l.kc8 ? l.kc8 : l.kc16 ? (uint8_t*) l.kc16 : (uint8_t*) l.kc;
        uint8_t* vc = l.kc8 ? l.vc8 : l.kc16 ? (uint8_t*) l.vc16 : (uint8_t*) l.vc;
        for (uint8_t* c : {kc, vc}) {
            copy(c + (size_t) s0 * row, (size_t) n1 * row);
            if (n2 > 0) copy(c, (size_t) n2 * row);
        }
    }
}
}  // namespace

void save_context(const Drafter* d, int pos, std::vector<uint8_t>& out) {
    size_t n = 0;
    ctx_rows(d, pos, [&](uint8_t*, size_t b) { n += b; });
    out.resize(n);
    size_t at = 0;
    ctx_rows(d, pos, [&](uint8_t* dev, size_t b) {
        cu(cudaMemcpyAsync(out.data() + at, dev, b, cudaMemcpyDeviceToHost, d->s), "save_context");
        at += b;
    });
    cu(cudaStreamSynchronize(d->s), "save_context");
}

void load_context(Drafter* d, int pos, const std::vector<uint8_t>& in) {
    size_t at = 0;
    ctx_rows(d, pos, [&](uint8_t* dev, size_t b) {
        if (at + b > in.size()) return;   // (not this drafter's rows: what is there stays, the drafts only worse)
        cu(cudaMemcpyAsync(dev, in.data() + at, b, cudaMemcpyHostToDevice, d->s), "load_context");
        at += b;
    });
    cu(cudaStreamSynchronize(d->s), "load_context");
}

}  // namespace ik2::draft
