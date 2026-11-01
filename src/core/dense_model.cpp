// src/core/dense_model.cpp - see include/strata/core/dense_model.hpp.
#include "strata/core/dense_model.hpp"

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/safetensors_dir.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/dense_kernels.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/g4_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q35_dflash.hpp"
#include "strata/kernels/q35_kernels.hpp"
#include "strata/kernels/q35_moe.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_tc_kq.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_runtime.h>

#include <deque>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <regex>
#include <set>
#include <thread>

namespace strata::core {
namespace {

using namespace strata::kernels;

constexpr uint32_t kF32 = 0, kF16 = 1, kBF16 = 30;
constexpr int kMaxCols = 8;                         // native_mmvq's columns; more tokens take the GEMM path

// The GGUF types a quantized weight may have: exactly what native_mmvq takes.
const char* kSupported = "Q4_0, Q5_0, Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS";

// defined below: dequantizes contiguous blocks of any type the host knows (F32/F16/BF16 and the quantized ones)
bool dequant_row(uint32_t type, const uint8_t* src, int n, float* out);

struct QMat {
    void* dev = nullptr;
    int type = -1;
    int n_in = 0, n_out = 0;
    bool packed = false;                            // in q35_mmtp's packed layout (else GGUF rows)
    bool fp16_act = false;                          // FP16 activations at every width (no Q8_1: an unnormed input)
    const strata::kernels::exl3::Mat* x3 = nullptr; // an EXL3 matrix (an exllamav3 checkpoint): dev / type unused
};

struct Layer {
    bool attn = false;
    float* attn_norm = nullptr;
    float* post_norm = nullptr;
    QMat ffn_gate, ffn_up, ffn_down;
    // GDN mixer
    QMat qkv, z, ssm_out;
    float *conv_w = nullptr, *dt = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
    float *ab_w = nullptr;                         // F32/F16/BF16 files: [alpha rows | beta rows] (2 h_v x n_embd)
    QMat alpha_q, beta_q;                          // quantized files
    bool ab_quant = false;
    float *state = nullptr, *conv_state = nullptr;
    // a verify pass's inputs of the recurrence (kMaxCols tokens): raw conv input, conv output (q, k normed), g, b
    float *st_raw = nullptr, *st_conv = nullptr, *st_ab = nullptr;   // st_ab rows: [g (h_v) | b (h_v)]
    float* st_d = nullptr;                          // the verify pass's recurrence updates [kMaxCols][h_v][128] (commit)
    // attention mixer
    QMat q, k, v, o;
    float *q_norm = nullptr, *k_norm = nullptr;
    Q35Kv kv;                                      // the K / V cache (FP16 or K8V4)
    // a MoE FFN: the router (F32, n_expert rows of n_embd), the stacked experts in their GGUF rows (MMQ), the shared
    // expert and its gate (F32 row)
    float *router = nullptr, *sh_gate = nullptr;   // router: n_expert rows (+1: the shared expert's gate row)
    void *exp_gate = nullptr, *exp_up = nullptr, *exp_down = nullptr;
    int type_gate = -1, type_up = -1, type_down = -1;
    QMat sh_gate_m, sh_up, sh_down;
    QMat sh_gu;                                     // [gate rows | up rows] of the shared expert (one GEMV)
    // Gemma 4: sliding-window (head 256) or full (head 512) attention, KV of its own or of layer kv_src; the FFN's
    // pre / post norms; the per-layer input's gate and projection; the layer's output scale
    int D = 0;
    int HK = 0;                                     // its K/V heads (Gemma 4 31B: 16 sliding, 4 full)
    bool swa = false;
    int kv_src = -1;
    G4Kv g4kv;
    float *ffn_pre_norm = nullptr, *post_ffw_norm = nullptr, *ple_norm = nullptr;
    QMat inp_gate, ple_proj;
    QMat attn_gate;                                 // Muse Glimmer: the attention output gate (sigmoid)
    float *inp_gate_f = nullptr, *ple_proj_f = nullptr;   // (STRATA_G4_F32: the F32 rows, exact)
    float out_scale = 1.0f;
    // the MTP (nextn) block's own
    QMat eh_proj;
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
};

struct DfLayer {
    float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    float *attn_base = nullptr, *ffn_base = nullptr;   // [side][tap][E]
    QMat q, k, v, o, attn_conv, ffn_conv, gate, up, down;
    uint16_t *Kr = nullptr, *Vr = nullptr;            // the context ring, [ring][n_kv * 128] FP16
};

struct Dflash {
    bool on = false;
    int n_layer = 0, n_head = 0, n_kv = 0, block = 0, taps = 0, group = 0, rank = 0, top_k = 0, window = 2048;
    int mask_tok = 0, n_cand = 0;                     // candidates come from ids [0, n_cand)
    int n_ff = 0;                                     // the drafter's MLP width (its own: a MoE target's differs)
    // DSpark (LithosAI): no convolutions, no selector; its own embedding and head, a Markov head
    // d_k = argmax(head(h_k) + W2 . W1[d_(k-1)]); every block row drafts (row 0: the anchor's)
    bool dspark = false;
    bool plain = false;                               // plain DFlash (arch "dflash": no conv, no selector; the target's
                                                      // head on rows 1..K, as llama.cpp's dflash.cpp)
    QMat ds_embd, ds_head, ds_w2;
    uint16_t* ds_w1 = nullptr;                        // [vocab][rank] FP16
    float *ds_x = nullptr, *ds_lg = nullptr;          // the chain's W1 row and one row's logits
    int* ds_tok = nullptr;                            // the anchor, then the drafts (device)
    float* attn_part = nullptr;                       // the split-K attention's partials
    float *kc = nullptr, *vc = nullptr;               // its K / V rows (chunk x kvd: wider than the target's on a MoE)
    float *g = nullptr, *u = nullptr, *hh = nullptr;  // its MLP rows (block x n_ff)
    float eps = 1e-6f, theta = 1e7f;
    std::vector<int> target_layers;
    QMat fc, hproj;
    float *hidden_norm = nullptr, *out_norm = nullptr;
    uint16_t *pred = nullptr, *succ = nullptr;        // selector codebooks, FP16 rows of rank
    std::vector<DfLayer> layers;
    int64_t* ring_pos = nullptr;
    std::vector<float> conf_h;                        // the last draft's confidences (host)
    // a sampled request's draft (dflash_sample): the walk draws with u_h's uniforms at temp; q_h / cand_h the
    // drawn positions' proposal distributions [K][top_k] and their candidates' ids
    bool sample = false;
    float temp = 1.0f;
    std::vector<float> u_h, q_h;
    std::vector<int32_t> cand_h;
    float *u_dev = nullptr, *q_dev = nullptr;
    // buffers
    float *feat = nullptr, *h = nullptr, *dyn = nullptr, *a = nullptr, *hf = nullptr, *logits = nullptr, *gate = nullptr;
    float* cand_v = nullptr;
    int* cand_i = nullptr;
    int* tok = nullptr;
    float* conf = nullptr;
    void* topk_scratch = nullptr;
};

std::string upper_type(uint32_t t) { return strata::ggml_type_name(t); }

uint16_t f32_to_f16(float f) {   // round to nearest even (normal range; subnormals flush, overflow saturates)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    const uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 31) return (uint16_t) (sign | 0x7bffu);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

// F16 rows (n_out of n_in) -> Q8_0 blocks (ggml's quantize_row_q8_0_ref), on the host, a few threads
std::vector<uint8_t> f16_to_q8_0(const uint16_t* src, int64_t n_in, int64_t n_out) {
    std::vector<uint8_t> out((size_t) (n_in / 32) * n_out * 34);
    const int64_t nb = n_in / 32 * n_out;
    const int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            for (int64_t b = t; b < nb; b += nt) {
                float v[32], amax = 0.0f;
                for (int i = 0; i < 32; ++i) { v[i] = strata::fp16_to_fp32(src[b * 32 + i]); amax = std::max(amax, std::fabs(v[i])); }
                const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
                uint8_t* o = out.data() + b * 34;
                const uint16_t dh = f32_to_f16(d);
                std::memcpy(o, &dh, 2);
                for (int i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) std::lround(v[i] * id);
            }
        });
    for (auto& x : th) x.join();
    return out;
}

// F32 rows (n_out of n_in) -> Q8_0 blocks, the same rounding
std::vector<uint8_t> f32_to_q8_0(const float* src, int64_t n_in, int64_t n_out) {
    std::vector<uint8_t> out((size_t) (n_in / 32) * n_out * 34);
    const int64_t nb = n_in / 32 * n_out;
    const int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            for (int64_t b = t; b < nb; b += nt) {
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(src[b * 32 + i]));
                const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
                uint8_t* o = out.data() + b * 34;
                const uint16_t dh = f32_to_f16(d);
                std::memcpy(o, &dh, 2);
                for (int i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) std::lround(src[b * 32 + i] * id);
            }
        });
    for (auto& x : th) x.join();
    return out;
}

// A tensor of one GGUF file, found by name across the shards.
struct Found {
    const strata::GgufFile* file = nullptr;
    const strata::TensorInfo* tensor = nullptr;
    // (an exllamav3 checkpoint's tensor under its GGUF name: `tensor` a stand-in with the GGUF shape)
    const strata::StTensor* st = nullptr;           // a float tensor (BF16 / F16 / F32)
    int xf = 0;                                     // 1: + 1 (the checkpoint's norms are offsets), 2: -exp (A_log -> ssm_a)
    const strata::StTensor *trellis = nullptr, *suh = nullptr, *svh = nullptr;   // an EXL3 matrix
};

}  // namespace

struct DenseModel::Impl {
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::map<std::string, Found> index;
    // an exllamav3 (EXL3) checkpoint directory instead of a GGUF file: its tensors under GGUF names (index_exl3)
    std::unique_ptr<strata::SafetensorsDir> st;
    std::deque<strata::TensorInfo> synth;
    std::deque<strata::kernels::exl3::Mat> x3mats;
    std::vector<Layer> layers;
    Layer mtp;                                      // blk.<n_layer>: the nextn block (attn == true when loaded)
    Dflash df;
    std::unique_ptr<strata::GgufFile> df_file;
    std::vector<uint8_t*> ckpt;                      // pinned host checkpoint slots (allocated on first use)
    std::vector<void*> allocations;                 // everything cudaMalloc'ed here, freed in the destructor
    std::vector<void*> host_allocs;                 // weights that did not fit in VRAM: mapped pinned host memory
    uint64_t budget = UINT64_MAX;                   // bytes of weights that may go to VRAM
    uint64_t dev_used = 0, host_bytes = 0;
    int host_tensors = 0;
    cudaStream_t stream = nullptr;
    // decode graphs (STRATA_DENSE_GRAPH=0 disables): the segments between attention cores of a pass of up to 8 rows,
    // per (rows, verify, logits); dropped when a buffer they hold moves
    bool pairs_mma = false;   // STRATA_DENSE_PAIRS_MMA: the experts' tensor-core GEMVs at 1-3 rows too
    bool graphs = false;   // (measured: no gain on the 35B / 27B decode - the kernels, not the launches, cost)
    std::map<int, std::vector<cudaGraphExec_t>> graph_sets;
    void drop_graphs() {
        for (auto& [k, v] : graph_sets)
            for (cudaGraphExec_t g : v) cudaGraphExecDestroy(g);
        graph_sets.clear();
    }
    int chunk = 512;                                // the largest T of a forward pass
    int max_in = 0;                                 // the widest input the activation conversions are sized for
    // the K / V cache: 0 FP16, 1 K8V4, 2 K8V8 (STRATA_DENSE_KV / --kv).  K8V8 by default: Qwen3.8-27B, 512 greedy
    // tokens, the same tokens as FP16 at 35K / 68K, TG 61.6 -> 62.6 (35K MTP), 35.9 -> 38.8 (68K MTP) at half the memory
    int kv_mode = 2;

    // Gemma 4: the per-layer embeddings (the token table, the model projection and its norm), rope bases, window
    bool gemma = false, muse = false;
    float logit_scale = 1.0f;
    float* attn_gate_out = nullptr;
    int ple_dim = 0, n_swa = 0;
    float rope_swa = 1e4f, softcap = 0.0f;
    QMat ple_tok, ple_model;
    float *ple_proj_norm = nullptr, *rope_freqs = nullptr;
    float* ple_model_f = nullptr;                   // (STRATA_G4_F32)
    bool g4_f32 = false;
    // the Gemma 4 assistant (gemma4-assistant GGUF, STRATA_DENSE_MTP): 4 small layers on the target's last K/V of
    // each kind, its own tied 256-wide head; a draft step at the anchor's position
    struct G4ALayer {
        float *attn_norm = nullptr, *q_norm = nullptr, *post_attn = nullptr, *ffn_norm = nullptr, *post_ffw = nullptr;
        QMat q, o, gate, up, down;
        float out_scale = 1.0f;
        int D = 0;
        bool swa = false;
    };
    std::unique_ptr<strata::GgufFile> asst_file;
    std::vector<G4ALayer> asst;
    QMat asst_pre, asst_post, asst_embd;
    float* asst_norm = nullptr;
    int asst_E = 0, asst_H = 0, asst_ff = 0;
    float *a_h = nullptr, *a_n = nullptr, *a_q = nullptr, *a_qr = nullptr, *a_att = nullptr, *a_o = nullptr,
          *a_g = nullptr, *a_u = nullptr;
    float *ple_rows = nullptr, *ple_proj_out = nullptr, *ple = nullptr, *ple_g = nullptr;

    QMat embd;                                      // token_embd, on the device (rows dequantized per token)
    float* output_norm = nullptr;
    QMat head;
    QMat head_draft;                                // the head's first draft_vocab rows (the drafts' vocabulary)
    // the MTP draft head (packed rows of the head gathered: slot j = token mtp_ids[j]) - built by mtp_vocab_select
    QMat head_mtp;
    int32_t* mtp_ids_dev = nullptr;
    std::vector<int32_t> mtp_ids;
    int mtp_extra = 0;
    void* q8_1 = nullptr;                           // the GEMVs' quantized activations (kMaxCols columns)
    uint16_t* xh = nullptr;                         // the GEMMs' FP16 activations (chunk rows of the widest input)
    strata::prefill::Gemm gemm;
    int cur_T = 0;                                  // the T the prepared activation (q8_1 / xh) is for

    // activations (device, f32, chunk rows each)
    float *x = nullptr, *xn = nullptr, *mix = nullptr, *hn = nullptr;
    float *qkv = nullptr, *conv = nullptr, *ab = nullptr;
    float *o = nullptr, *z = nullptr, *y = nullptr, *ab_tmp = nullptr;
    float *q_full = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr;
    float *ffn_g = nullptr, *ffn_u = nullptr, *ffn_h = nullptr;
    float* logits = nullptr;                        // kMaxCols rows of n_vocab
    float* attn_scratch = nullptr;
    // (K8V8) one layer's cache unpacked to FP16 for the prompt path's attention (STRATA_DENSE_KV_EXPAND, default on)
    uint16_t *kv16_k = nullptr, *kv16_v = nullptr;
    float* conv_dummy = nullptr;
    int swa_ring = 0;          // Gemma: the sliding-window layers' ring cells (0: they hold the whole context)
    int64_t ring_hi = -1;      // the highest position written into the rings since the last reset
    strata::kernels::Q35CommitLayer* commit_tab = nullptr;   // device: the GDN layers' commit inputs (built once)
    int commit_n = 0;
    int32_t* tok_dev = nullptr;
    // MoE (chunk x k pairs)
    float *moe_logits = nullptr, *moe_w = nullptr, *moe_g = nullptr, *moe_u = nullptr, *moe_y = nullptr, *moe_sh = nullptr;
    float* moe_shl = nullptr;
    int32_t *moe_ids = nullptr, *moe_bounds = nullptr, *moe_tok = nullptr, *moe_pos = nullptr, *moe_max = nullptr;
    int32_t *moe_cnt = nullptr, *moe_iota = nullptr, *moe_pairtok = nullptr;
    void* moe_xq = nullptr;
    int32_t* moe_max_host = nullptr;   // pinned
    std::unique_ptr<strata::prefill::mmq::Context> mmq;
    // MTP
    float *mtp_x = nullptr, *mtp_cat = nullptr, *mtp_h = nullptr, *mtp_logits = nullptr;
    const int32_t* mtp_dev_tok = nullptr;   // (mtp_forward_dev: the row's token on the device)

    // ---- STRATA_DENSE_PROF=1: GPU time a stage (events around each), printed every 64 forward passes
    struct Prof {
        bool on = false;
        std::vector<cudaEvent_t> ev;
        std::vector<int> ev_name;                  // the stage an event pair closes (-1: an opening event)
        std::vector<std::string> names;
        std::vector<double> ms;
        std::vector<long long> calls;
        int used = 0, passes = 0;
        cudaEvent_t next(cudaStream_t s) {
            if (used == (int) ev.size()) { cudaEvent_t e; cudaEventCreate(&e); ev.push_back(e); ev_name.push_back(-1); }
            cudaEventRecord(ev[used], s);
            return ev[used++];
        }
        int id(const char* n) {
            for (size_t i = 0; i < names.size(); ++i) if (names[i] == n) return (int) i;
            names.push_back(n); ms.push_back(0); calls.push_back(0);
            return (int) names.size() - 1;
        }
    } prof;
    void mark(cudaStream_t s) { if (prof.on) { prof.next(s); prof.ev_name[prof.used - 1] = -1; } }
    void stage(const char* n, cudaStream_t s) {    // closes the stage begun at the last mark/stage
        if (!prof.on) return;
        prof.next(s);
        prof.ev_name[prof.used - 1] = prof.id(n);
    }
    void prof_flush(int T) {
        if (!prof.on) return;
        cudaEventSynchronize(prof.ev[prof.used - 1]);
        for (int i = 1; i < prof.used; ++i) {
            if (prof.ev_name[i] < 0) continue;
            float m = 0;
            cudaEventElapsedTime(&m, prof.ev[i - 1], prof.ev[i]);
            prof.ms[prof.ev_name[i]] += m;
            prof.calls[prof.ev_name[i]] += 1;
        }
        prof.used = 0;
        // (STRATA_DENSE_PROF_N: passes a report, 64 - e.g. a 28K prompt in 2048-token chunks is 14)
        static const int every = std::getenv("STRATA_DENSE_PROF_N") ? std::max(1, std::atoi(std::getenv("STRATA_DENSE_PROF_N"))) : 64;
        if (++prof.passes % every == 0) {
            double tot = 0;
            for (double m : prof.ms) tot += m;
            std::fprintf(stderr, "dense prof (last %d passes, T=%d): %.2f ms a pass\n", every, T, tot / every);
            for (size_t i = 0; i < prof.names.size(); ++i)
                std::fprintf(stderr, "  %-14s %8.3f ms a pass %5.1f%%\n", prof.names[i].c_str(), prof.ms[i] / every,
                             tot > 0 ? 100.0 * prof.ms[i] / tot : 0.0);
            std::fill(prof.ms.begin(), prof.ms.end(), 0.0);
        }
    }

    // ---- helpers
    template <class T> bool alloc(T** out, uint64_t bytes, std::string& err, const char* what) {
        void* p = nullptr;
        const cudaError_t s = cudaMalloc(&p, bytes ? bytes : 16);
        if (s != cudaSuccess) {
            err = std::string("dense model: cannot allocate ") + what + " (" + std::to_string(bytes >> 20) + " MiB): " +
                  cudaGetErrorString(s);
            return false;
        }
        allocations.push_back(p);
        *out = (T*) p;
        return true;
    }

    const Found* find(const std::string& name) const {
        auto it = index.find(name);
        return it == index.end() ? nullptr : &it->second;
    }
    const Found* find_any(std::initializer_list<const char*> names, const std::string& prefix) const {
        for (const char* n : names)
            if (const Found* f = find(prefix + n)) return f;
        return nullptr;
    }

    // An exllamav3 checkpoint's tensors entered under the names the GGUF path looks up (llama.cpp's convention):
    // its norms (stored as offsets from 1) and A_log converted as they are read, its quantized matrices EXL3
    void index_exl3(int n_layer, const strata::Json& tc) {
        const strata::SafetensorsDir& D = *st;
        auto stand_in = [&](const std::string& gname, std::vector<uint64_t> shape, uint32_t type) {
            synth.emplace_back();
            synth.back().name = gname;
            synth.back().shape = std::move(shape);
            synth.back().type = type;
            return &synth.back();
        };
        auto flt = [&](const std::string& gname, const std::string& hname, int xf) {
            const strata::StTensor* t = D.find(hname);
            if (!t) return;
            std::vector<uint64_t> shape;
            for (auto it = t->shape.rbegin(); it != t->shape.rend(); ++it) shape.push_back((uint64_t) *it);   // (GGUF order)
            Found f;
            f.tensor = stand_in(gname, shape, 0 /* F32 */);
            f.st = t;
            f.xf = xf;
            index[gname] = f;
        };
        auto mat = [&](const std::string& gname, const std::string& hbase) {
            const strata::StTensor* tr = D.find(hbase + ".trellis");
            if (!tr || tr->shape.size() != 3) return;
            Found f;
            f.tensor = stand_in(gname, {(uint64_t) tr->shape[0] * 16, (uint64_t) tr->shape[1] * 16},
                                (uint32_t) strata::kernels::exl3::kTypeEXL3);
            f.trellis = tr;
            f.suh = D.find(hbase + ".suh");
            f.svh = D.find(hbase + ".svh");
            index[gname] = f;
        };
        const strata::Json* lt = tc.get("layer_types");
        for (int l = 0; l <= n_layer; ++l) {   // (l == n_layer: the MTP block as GGUF's blk.<n_layer>)
            const bool m = l == n_layer;
            const std::string g = "blk." + std::to_string(l) + ".";
            const std::string h = m ? "mtp.layers.0." : "model.language_model.layers." + std::to_string(l) + ".";
            flt(g + "attn_norm.weight", h + "input_layernorm.weight", 1);
            flt(g + "post_attention_norm.weight", h + "post_attention_layernorm.weight", 1);
            mat(g + "ffn_gate.weight", h + "mlp.gate_proj");
            mat(g + "ffn_up.weight", h + "mlp.up_proj");
            mat(g + "ffn_down.weight", h + "mlp.down_proj");
            const bool full = m || (lt && l < (int) lt->arr.size() && lt->arr[(size_t) l].str == "full_attention");
            if (full) {
                mat(g + "attn_q.weight", h + "self_attn.q_proj");
                mat(g + "attn_k.weight", h + "self_attn.k_proj");
                mat(g + "attn_v.weight", h + "self_attn.v_proj");
                mat(g + "attn_output.weight", h + "self_attn.o_proj");
                flt(g + "attn_q_norm.weight", h + "self_attn.q_norm.weight", 1);
                flt(g + "attn_k_norm.weight", h + "self_attn.k_norm.weight", 1);
            } else {
                mat(g + "attn_qkv.weight", h + "linear_attn.in_proj_qkv");
                mat(g + "attn_gate.weight", h + "linear_attn.in_proj_z");
                mat(g + "ssm_out.weight", h + "linear_attn.out_proj");
                flt(g + "ssm_conv1d.weight", h + "linear_attn.conv1d.weight", 0);
                flt(g + "ssm_dt.bias", h + "linear_attn.dt_bias", 0);
                flt(g + "ssm_a", h + "linear_attn.A_log", 2);
                flt(g + "ssm_norm.weight", h + "linear_attn.norm.weight", 0);
                flt(g + "ssm_alpha.weight", h + "linear_attn.in_proj_a.weight", 0);
                flt(g + "ssm_beta.weight", h + "linear_attn.in_proj_b.weight", 0);
            }
            if (m) {
                mat(g + "nextn.eh_proj.weight", "mtp.fc");
                flt(g + "nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight", 1);
                flt(g + "nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight", 1);
                flt(g + "nextn.shared_head_norm.weight", "mtp.norm.weight", 1);
            }
        }
        flt("token_embd.weight", "model.language_model.embed_tokens.weight", 0);
        flt("output_norm.weight", "model.language_model.norm.weight", 1);
        mat("output.weight", "lm_head");
    }

    // An EXL3 matrix (Found with a trellis) onto the device; out.x3 points at it (exl3::mv / reconstruct_rows read it)
    bool upload_exl3(const Found* f, const std::string& name, int n_in, int n_out, QMat& out, uint64_t& total,
                     std::string& err) {
        namespace ex = strata::kernels::exl3;
        const strata::StTensor& tr = *f->trellis;
        const int k = (int) tr.shape[0] * 16, n = (int) tr.shape[1] * 16, k2 = (int) tr.shape[2] / 8;
        if (k != n_in || n != n_out || !f->suh || !f->svh || f->suh->bytes != (uint64_t) k * 2 || f->svh->bytes != (uint64_t) n * 2) {
            err = "dense model: " + name + " (EXL3) is " + std::to_string(k) + " x " + std::to_string(n) + ", expected " +
                  std::to_string(n_in) + " x " + std::to_string(n_out);
            return false;
        }
        void *t = nullptr, *su = nullptr, *sv = nullptr;
        if (!alloc(&t, tr.bytes + 16, err, name.c_str()) || !alloc(&su, (uint64_t) k * 2, err, name.c_str()) ||
            !alloc(&sv, (uint64_t) n * 2, err, name.c_str()))
            return false;
        cudaMemcpy(t, tr.data, tr.bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(su, f->suh->data, (size_t) k * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(sv, f->svh->data, (size_t) n * 2, cudaMemcpyHostToDevice);
        ex::Mat m;
        m.trellis = (const uint32_t*) t;
        m.suh = (const uint16_t*) su;
        m.svh = (const uint16_t*) sv;
        m.k = k; m.n = n; m.k2 = k2; m.cb = 2;   // (the checkpoints' mul1 is the kernels' multiplier: 0x83DCD12D)
        if (!ex::supported(m)) { err = "dense model: " + name + ": an EXL3 shape / bitrate the kernels do not take"; return false; }
        x3mats.push_back(m);
        out = QMat{};
        out.type = ex::kTypeEXL3;
        out.n_in = n_in;
        out.n_out = n_out;
        out.x3 = &x3mats.back();
        dev_used += tr.bytes + (uint64_t) (k + n) * 2;
        total += tr.bytes + (uint64_t) (k + n) * 2;
        return true;
    }

    // A checkpoint's float tensor as F32 on the host, converted as its index entry says
    static std::vector<float> st_floats(const Found& f) {
        const strata::StTensor& t = *f.st;
        uint64_t n = 1;
        for (int64_t d : t.shape) n *= (uint64_t) d;
        std::vector<float> h(n);
        if (t.dtype == "F32") std::memcpy(h.data(), t.data, n * 4);
        else if (t.dtype == "F16") strata::dequantize_f16(t.data, h.data(), (int) n);
        else strata::dequantize_bf16(t.data, h.data(), (int) n);
        if (f.xf == 1) for (float& v : h) v += 1.0f;
        else if (f.xf == 2) for (float& v : h) v = -std::exp(v);
        return h;
    }

    // A float vector or matrix (F32, F16 or BF16 in the file) uploaded as F32.
    bool upload_floats(const Found* f, const std::string& name, uint64_t expect, float** out, std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        if (f->st) {
            const std::vector<float> host = st_floats(*f);
            if (expect && host.size() != expect) {
                err = "dense model: " + name + " has " + std::to_string(host.size()) + " values, expected " + std::to_string(expect);
                return false;
            }
            if (!alloc(out, host.size() * 4, err, name.c_str())) return false;
            cudaMemcpy(*out, host.data(), host.size() * 4, cudaMemcpyHostToDevice);
            return true;
        }
        uint64_t n = 1;
        for (uint64_t d : f->tensor->shape) n *= d;
        if (expect && n != expect) {
            err = "dense model: " + name + " has " + std::to_string(n) + " values, expected " + std::to_string(expect);
            return false;
        }
        const uint8_t* src = f->file->tensor_data(*f->tensor);
        std::vector<float> host(n);
        switch (f->tensor->type) {
        case kF32: strata::dequantize_f32(src, host.data(), (int) n); break;
        case kF16: strata::dequantize_f16(src, host.data(), (int) n); break;
        case kBF16: strata::dequantize_bf16(src, host.data(), (int) n); break;
        default:
            if (n > (uint64_t) INT_MAX || !dequant_row(f->tensor->type, src, (int) n, host.data())) {
                err = "dense model: " + name + " is " + upper_type(f->tensor->type) +
                      ", which cannot be read as floats (expected F32, F16, BF16 or a quantized type)";
                return false;
            }
        }
        if (!alloc(out, n * 4, err, name.c_str())) return false;
        const cudaError_t s = cudaMemcpy(*out, host.data(), n * 4, cudaMemcpyHostToDevice);
        if (s != cudaSuccess) { err = "dense model: upload " + name + ": " + cudaGetErrorString(s); return false; }
        return true;
    }

    // A quantized matrix, kept in its GGUF blocks.
    // q35_mmtp's packed layout for the GEMVs (v100-skinny / NInfer's prepacking; STRATA_DENSE_PACK=0: GGUF rows);
    // a prompt chunk's GEMM dequantizes straight from it (q35_mmt_dequant_packed).
    bool pack_on = true;
    bool pack(QMat& m, bool wide, std::string& err) {
        (void) wide;
        if (!pack_on || !q35_mmt_supported(m.type) || m.n_in % 256) return true;
        const uint64_t pb = q35_mmt_packed_bytes(m.type, m.n_in, m.n_out);
        void* p = nullptr;
        if (cudaMalloc(&p, pb) != cudaSuccess) { cudaGetLastError(); return true; }   // (no room: GGUF rows)
        q35_mmt_pack(m.type, m.dev, p, m.n_in, m.n_out, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) { err = "dense model: packing failed"; return false; }
        const uint64_t ob = native_mmvq_weight_bytes(m.type, m.n_in, m.n_out);
        cudaFree(m.dev);
        allocations.erase(std::find(allocations.begin(), allocations.end(), m.dev));
        allocations.push_back(p);
        dev_used += pb - ob;
        m.dev = p;
        m.packed = true;
        return true;
    }

    bool upload_mat(const Found* f, const std::string& name, int n_in, int n_out, QMat& out, uint64_t& total,
                    std::string& err, bool packable = true, bool wide = true, bool host_only = false) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        if (f->trellis) return upload_exl3(f, name, n_in, n_out, out, total, err);
        if (f->st) {   // a checkpoint's float matrix (the embedding): Q8_0 rows, as the row lookups and GEMVs take
            const strata::StTensor& t = *f->st;
            if (t.shape.size() != 2 || t.shape[0] != n_out || t.shape[1] != n_in || n_in % 32) {
                err = "dense model: " + name + " is not [" + std::to_string(n_out) + ", " + std::to_string(n_in) + "]";
                return false;
            }
            const uint64_t blocks = (uint64_t) n_out * (n_in / 32), bytes = blocks * 34;
            std::vector<uint8_t> q(bytes);
            const bool bf = t.dtype == "BF16";
            auto work = [&](uint64_t b0, uint64_t b1) {
                for (uint64_t b = b0; b < b1; ++b) {
                    float v[32], amax = 0.0f;
                    for (int i = 0; i < 32; ++i) {
                        const uint16_t u = ((const uint16_t*) t.data)[b * 32 + i];
                        if (bf) { const uint32_t w = (uint32_t) u << 16; std::memcpy(&v[i], &w, 4); }
                        else { v[i] = strata::kernels::f32_from_f16(u); }
                        amax = std::max(amax, std::fabs(v[i]));
                    }
                    const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
                    uint8_t* o = q.data() + b * 34;
                    const uint16_t dh = strata::kernels::f16_from_f32(d);
                    std::memcpy(o, &dh, 2);
                    for (int i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) std::lround(v[i] * id);
                }
            };
            const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
            std::vector<std::thread> th;
            for (unsigned i = 0; i < nt; ++i) th.emplace_back(work, blocks * i / nt, blocks * (i + 1) / nt);
            for (auto& x : th) x.join();
            void* dev = nullptr;
            if (host_only) {
                void* host = nullptr;
                if (cudaHostAlloc(&host, bytes, cudaHostAllocMapped) != cudaSuccess || cudaHostGetDevicePointer(&dev, host, 0) != cudaSuccess) {
                    err = "dense model: " + name + ": pinned host memory"; return false;
                }
                std::memcpy(host, q.data(), bytes);
                host_allocs.push_back(host);
            } else {
                if (!alloc(&dev, bytes, err, name.c_str())) return false;
                cudaMemcpy(dev, q.data(), bytes, cudaMemcpyHostToDevice);
            }
            out = QMat{dev, 8 /* Q8_0 */, n_in, n_out};
            total += bytes;
            return !(packable && !host_only) || pack(out, wide, err);
        }
        const strata::TensorInfo& t = *f->tensor;
        if (t.shape.size() != 2 || t.shape[0] != (uint64_t) n_in || t.shape[1] != (uint64_t) n_out) {
            err = "dense model: " + name + " has shape [" + std::to_string(t.shape.empty() ? 0 : t.shape[0]) + ", " +
                  std::to_string(t.shape.size() > 1 ? t.shape[1] : 0) + "], expected [" + std::to_string(n_in) + ", " +
                  std::to_string(n_out) + "]";
            return false;
        }
        if (!native_mmvq_supported((int) t.type) || !dequant_f16_supported((int) t.type)) {
            err = "dense model: " + name + " is " + upper_type(t.type) + ", which the native GEMVs do not take. "
                  "Supported: " + kSupported + "";
            return false;
        }
        int elems = 0, block_bytes = 0;
        if (!strata::block_geometry(t.type, elems, block_bytes) || (uint64_t) n_in % (uint64_t) elems) {
            err = "dense model: " + name + ": invalid block geometry"; return false;
        }
        const uint64_t bytes = native_mmvq_weight_bytes((int) t.type, n_in, n_out);
        const uint64_t in_file = (uint64_t) n_in * n_out / (uint64_t) elems * (uint64_t) block_bytes;
        const uint64_t payload = f->file->file_size() - f->file->data_start();
        if (bytes != in_file || t.offset > payload || bytes > payload - t.offset) {
            err = "dense model: " + name + ": payload size mismatch or truncated file"; return false;
        }
        void* dev = nullptr;
        cudaError_t s = cudaSuccess;
        if (!host_only && dev_used + bytes <= budget) {
            s = cudaMalloc(&dev, bytes);
            if (s == cudaSuccess) s = cudaMemcpy(dev, f->file->tensor_data(t), bytes, cudaMemcpyHostToDevice);
            if (s != cudaSuccess) {                 // the card is fuller than cudaMemGetInfo said: spill this one
                if (dev) cudaFree(dev);
                dev = nullptr;
                cudaGetLastError();
            } else {
                allocations.push_back(dev);
                dev_used += bytes;
            }
        }
        if (!dev) {
            // Host-resident: pinned + mapped, so the GEMV kernel reads the blocks over PCIe on every token.
            void* host = nullptr;
            s = cudaHostAlloc(&host, bytes, cudaHostAllocMapped);
            if (s == cudaSuccess) s = cudaHostGetDevicePointer(&dev, host, 0);
            if (s != cudaSuccess) {
                if (host) cudaFreeHost(host);
                err = "dense model: " + name + " fits neither in VRAM nor in pinned host memory (" +
                      std::to_string(bytes >> 20) + " MiB): " + cudaGetErrorString(s);
                return false;
            }
            std::memcpy(host, f->file->tensor_data(t), bytes);
            host_allocs.push_back(host);
            if (!host_only) {   // (the warning below is for matrices that did not fit, not ones placed there)
                host_bytes += bytes;
                ++host_tensors;
            }
        }
        const bool in_vram = std::find(allocations.begin(), allocations.end(), dev) != allocations.end();
        out = QMat{dev, (int) t.type, n_in, n_out};
        total += bytes;
        return !(packable && in_vram) || pack(out, wide, err);
    }

    // ---- the projections of T token rows (x dense rows of n_in): prep() once per input, then mm() per matrix
    // up to 8 rows: the tensor-core GEMV (q35_mmt) for the formats it takes - from mmt_min rows, or from 1 with
    // STRATA_DENSE_MMT1 - else native_mmvq's dp4a GEMV (its q8_1 activations made on first use)
    bool mmt_on = true, mmt1 = false, packed_dp4a = true;   // (STRATA_DENSE_PACKED_DP4A=0: the tensor-core form at 1-3 rows)
    int mmt_min = 4;                                // (dp4a is faster up to 3 rows on a V100)
    uint16_t* xh8 = nullptr;
    float* mmt_part = nullptr;
    const float* cur_x = nullptr;
    int cur_n = 0;
    bool q8_ready = false, h8_ready = false;
    void prep(const float* x, int n_in, int T) {
        cur_T = T; cur_x = x; cur_n = n_in; q8_ready = h8_ready = false;
        if (T > kMaxCols) q35_to_half(x, xh, (int64_t) T * n_in, stream);
    }
    // an EXL3 matrix: up to 8 rows by exl3::mv (F32 rows in, the weights decoded once for all of them); a prompt
    // chunk by FP16 GEMMs over the weight reconstructed a column slice at a time into the GEMM's scratch
    void mm_exl3(const QMat& w, float* y) {
        namespace ex = strata::kernels::exl3;
        const ex::Mat& m = *w.x3;
        if (cur_T <= kMaxCols) {
            ex::MvJob j;
            j.m = &m; j.xf = cur_x; j.y = y; j.ncol = cur_T;
            j.xf_cs = (size_t) cur_n; j.y_cs = (size_t) w.n_out;
            if (!ex::mv(&j, 1, stream)) throw std::runtime_error("exl3 mv");
            return;
        }
        const int64_t step = std::max<int64_t>(128, gemm.scratch_elems() / m.k / 128 * 128);
        for (int64_t c0 = 0; c0 < m.n; c0 += step) {
            const int nc = (int) std::min<int64_t>(step, m.n - c0);
            const ex::Mat v = ex::view(m, (int) c0, nc);
            ex::reconstruct_rows(v, gemm.scratch(), false, stream);
            gemm.f16(xh, gemm.scratch(), y + c0, cur_T, nc, m.k, w.n_out);
        }
    }

    // (decode / verify) the GEMV input's FP16 rows written by the kernel that made x: no conversion pass
    // (STRATA_DENSE_FUSE=0: the separate passes); rows past T keep stale finite values - their products are dropped
    bool fuse = true;
    // commit from the verify pass's recorded recurrence updates (STRATA_DENSE_COMMIT_REC=0: recomputed)
    bool commit_rec = true;
    // images (STRATA_DENSE_VISION=1): the (t, h, w) rotary position of every cell, on the device for q35_attn_prep and
    // on the host (what was uploaded), and this request's image cells (ascending) with their embedding rows
    int32_t* mrope_dev = nullptr;
    std::vector<int32_t> mrope_host;
    std::vector<int64_t> img_cells;
    std::vector<float> img_rows;
    uint16_t* h8_out(int T) const { return fuse && T <= kMaxCols ? xh8 : nullptr; }
    void prep_h8(const float* x, int n_in, int T) {
        prep(x, n_in, T);
        if (h8_out(T)) h8_ready = true;
    }
    void mm(const QMat& w, float* y) {
        if (w.x3) { mm_exl3(w, y); return; }
        if (cur_T > kMaxCols) {
            if (w.packed) {   // dequantized to FP16 straight from the packed layout, row slices that fit the scratch
                const int64_t step = std::max<int64_t>(32, gemm.scratch_elems() / w.n_in / 32 * 32);
                for (int64_t r0 = 0; r0 < w.n_out; r0 += step) {
                    const int rows = (int) std::min<int64_t>(step, w.n_out - r0);
                    // (STRATA_DENSE_PP_NODQ=1: a measurement only - the GEMM over a stale scratch, wrong output - the
                    // weights' decode left out, what reusing decoded weights across chunks could save at most)
                    static const bool nodq = std::getenv("STRATA_DENSE_PP_NODQ") && std::atoi(std::getenv("STRATA_DENSE_PP_NODQ")) != 0;
                    if (!nodq) q35_mmt_dequant_packed(w.type, w.dev, w.n_in, w.n_out, (int) r0, rows, gemm.scratch(), stream);
                    gemm.f16(xh, gemm.scratch(), y + r0, cur_T, rows, w.n_in, w.n_out);
                }
            } else {
                gemm.native(xh, w.type, w.dev, y, cur_T, w.n_out, w.n_in);
            }
            return;
        }
        if (w.packed && cur_T < mmt_min && packed_dp4a && !w.fp16_act && q35_mmtp_dp4a_supported(w.type)) {
            if (!q8_ready) { native_quantize_q8_1(cur_x, q8_1, cur_n, cur_T, stream); q8_ready = true; }
            q35_mmtp_dp4a(w.type, w.dev, q8_1, y, w.n_in, w.n_out, cur_T, mmt_part, stream);
            return;
        }
        if (w.packed) {
            if (!h8_ready) { q35_mmt_prep(cur_x, cur_n, cur_T, xh8, stream); h8_ready = true; }
            q35_mmtp(w.type, w.dev, xh8, y, w.n_in, w.n_out, cur_T, mmt_part, stream);
            return;
        }
        if (mmt_on && (cur_T >= mmt_min || mmt1 || w.fp16_act) && q35_mmt_supported(w.type)) {
            if (!h8_ready) { q35_mmt_prep(cur_x, cur_n, cur_T, xh8, stream); h8_ready = true; }
            q35_mmt(w.type, w.dev, xh8, y, w.n_in, w.n_out, cur_T, mmt_part, stream);
            return;
        }
        if (!q8_ready) { native_quantize_q8_1(cur_x, q8_1, cur_n, cur_T, stream); q8_ready = true; }
        native_mmvq(w.type, w.dev, q8_1, y, w.n_in, w.n_out, cur_T, stream);
    }
};

namespace {

// Host-side dequantization of one row (n values, whole blocks).
bool dequant_row(uint32_t type, const uint8_t* src, int n, float* out) {
    switch (type) {
    case kF32: strata::dequantize_f32(src, out, n); return true;
    case kF16: strata::dequantize_f16(src, out, n); return true;
    case kBF16: strata::dequantize_bf16(src, out, n); return true;
    default: break;
    }
    int elems = 0, bytes = 0;
    if (!strata::block_geometry(type, elems, bytes) || n % elems) return false;
    for (int b = 0; b < n / elems; ++b) {
        const uint8_t* blk = src + (size_t) b * bytes;
        float* o = out + (size_t) b * elems;
        switch (type) {
        case 2: strata::dequantize_q4_0(blk, o); break;
        case 6: strata::dequantize_q5_0(blk, o); break;
        case 8: strata::dequantize_q8_0(blk, o); break;
        case 11: strata::dequantize_q3_K(blk, o); break;
        case 12: strata::dequantize_q4_K(blk, o); break;
        case 13: strata::dequantize_q5_K(blk, o); break;
        case 14: strata::dequantize_q6_K(blk, o); break;
        case 20: strata::dequantize_iq4_nl(blk, o); break;
        case 23: strata::dequantize_iq4_xs(blk, o); break;
        case 42: strata::dequantize_q2_0(blk, o); break;
        default: return false;
        }
    }
    return true;
}

// llama.cpp names a split file <base>-00001-of-0000N.gguf
std::vector<std::string> shard_paths(const std::string& path) {
    static const std::regex re(R"((.*-)(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (!std::regex_match(path, m, re)) return {path};
    const int n = std::stoi(m[3].str());
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%05d", i);
        out.push_back(m[1].str() + buf + "-of-" + m[3].str() + ".gguf");
    }
    return out;
}

// integer metadata that may be a scalar or (per layer) an array: the first element
bool meta_int(const strata::GgufFile& g, const std::string& key, int64_t& out) {
    const strata::MetaValue* v = g.get(key);
    if (!v) return false;
    if (v->type == strata::MetaType::ARRAY) {
        if (v->items.empty()) return false;
        out = (int64_t) v->items[0].u;
        return true;
    }
    out = (int64_t) v->u;
    return true;
}

}  // namespace

DenseModel::DenseModel() : impl_(new Impl) {}

DenseModel::~DenseModel() {
    if (!impl_) return;
    if (impl_->stream) cudaStreamSynchronize(impl_->stream);
    impl_->drop_graphs();
    for (void* p : impl_->allocations) cudaFree(p);
    for (void* p : impl_->host_allocs) cudaFreeHost(p);
    for (uint8_t* p : impl_->ckpt) if (p) cudaFreeHost(p);
    if (impl_->stream) cudaStreamDestroy(impl_->stream);
}

float* DenseModel::logits() const { return impl_->logits; }
void* DenseModel::stream() const { return impl_->stream; }
const float* DenseModel::hidden_norm() const { return impl_->hn; }
int DenseModel::chunk() const { return impl_->chunk; }
int DenseModel::draft_vocab() const { return impl_->head_mtp.dev ? impl_->head_mtp.n_out : impl_->head_draft.n_out; }
const int32_t* DenseModel::draft_ids() const { return impl_->head_mtp.dev ? impl_->mtp_ids_dev : nullptr; }
int32_t DenseModel::draft_id(int slot) const { return impl_->head_mtp.dev ? impl_->mtp_ids[(size_t) slot] : slot; }

bool DenseModel::mtp_vocab_select(const int32_t* prompt, size_t n, const int32_t* gen, size_t n_gen,
                                  std::string& err) {
    Impl& I = *impl_;
    if (I.mtp_extra <= 0 || !(I.mtp.attn || I.df.on) || I.gemma || !I.head.packed) return true;
    const int N = I.head_draft.n_out, X = std::min(I.mtp_extra, N / 2), base = N - X, V = cfg_.n_vocab;
    // the prompt's tokens at or beyond `base`, the most frequent first (ties: the lower id)
    std::unordered_map<int32_t, int> cnt;
    for (size_t i = 0; i < n; ++i)
        if (prompt[i] >= base && prompt[i] < V) ++cnt[prompt[i]];
    for (size_t i = 0; i < n_gen; ++i)
        if (gen[i] >= base && gen[i] < V) cnt[gen[i]] += 4;
    std::vector<std::pair<int, int32_t>> by;
    by.reserve(cnt.size());
    for (const auto& kv : cnt) by.push_back({-kv.second, kv.first});
    std::sort(by.begin(), by.end());
    std::vector<int32_t> ids((size_t) N);
    for (int j = 0; j < base; ++j) ids[(size_t) j] = j;
    int j = base;
    for (size_t k = 0; k < by.size() && j < N; ++k) ids[(size_t) j++] = by[k].second;
    std::vector<int32_t> extra(ids.begin() + base, ids.begin() + j);
    std::sort(extra.begin(), extra.end());
    for (int t = base; j < N; ++t)   // (the rest: the next ids in order, skipping the ones already in)
        if (!std::binary_search(extra.begin(), extra.end(), t)) ids[(size_t) j++] = t;
    if (I.head_mtp.dev && ids == I.mtp_ids) return true;   // (the same rows as the last request's)
    try {
        if (!I.head_mtp.dev) {
            I.head_mtp = I.head;
            I.head_mtp.n_out = N;
            void* p = nullptr;
            if (!I.alloc(&p, q35_mmt_packed_bytes(I.head.type, I.head.n_in, N), err, "MTP draft head") ||
                !I.alloc(&I.mtp_ids_dev, (uint64_t) N * 4, err, "MTP draft ids"))
                return false;
            I.head_mtp.dev = p;
        }
        cudaMemcpyAsync(I.mtp_ids_dev, ids.data(), (size_t) N * 4, cudaMemcpyHostToDevice, I.stream);
        q35_mmt_gather_packed(I.head.type, I.head.dev, I.head.n_in, I.mtp_ids_dev, N, I.head_mtp.dev, I.stream);
        I.mtp_ids.swap(ids);
        return sync(err);
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}
bool DenseModel::has_mtp() const { return impl_->mtp.attn; }
float* DenseModel::mtp_logits() const { return impl_->mtp_logits; }
const float* DenseModel::mtp_hidden() const { return impl_->mtp_h; }

bool DenseModel::load(const std::string& path, int64_t max_context, std::string& err, int chunk) {
    Impl& I = *impl_;
    // chunk <= 0: chosen here - the Qwen3.5/3.8 path takes the largest of 4096 / 2048 / 1024 whose activations still
    // leave VRAM for all the weights (Qwen3.8-27B, 35K prompt: PP 755 tok/s at 512, 888 at 2048, 903 at 4096);
    // the other paths 512
    const bool auto_chunk = chunk <= 0;
    I.chunk = auto_chunk ? 512 : std::max(kMaxCols + 1, chunk);
    if (const char* e = std::getenv("STRATA_DENSE_PACK")) I.pack_on = std::atoi(e) != 0;
    if (const char* e = std::getenv("STRATA_DENSE_KV")) {
        if (std::string(e) == "k8v4") I.kv_mode = 1;
        else if (std::string(e) == "k8v8") I.kv_mode = 2;
        else if (std::string(e) == "fp16") I.kv_mode = 0;
        else { err = std::string("dense model: STRATA_DENSE_KV=") + e + " (fp16 | k8v4 | k8v8)"; return false; }
    }
    I.prof.on = std::getenv("STRATA_DENSE_PROF") != nullptr && std::atoi(std::getenv("STRATA_DENSE_PROF")) != 0;
    try {
        if (std::filesystem::is_directory(path)) {
            // ---- 1-2. an exllamav3 (EXL3) checkpoint directory: config.json, its tensors under GGUF names
            I.st = std::make_unique<strata::SafetensorsDir>(path);
            const strata::Json* tcp = I.st->config().get("text_config");
            const strata::Json& tc = tcp ? *tcp : I.st->config();
            auto num = [&](const char* k, double d) { const strata::Json* j = tc.get(k); return j && j->kind == strata::Json::Num ? j->num : d; };
            cfg_.arch = "qwen35";
            cfg_.n_layer = (int) num("num_hidden_layers", 0);
            cfg_.n_embd = (int) num("hidden_size", 0);
            cfg_.n_ff = (int) num("intermediate_size", 0);
            cfg_.n_head = (int) num("num_attention_heads", 0);
            cfg_.n_head_kv = (int) num("num_key_value_heads", 0);
            cfg_.head_dim = (int) num("head_dim", 0);
            const strata::Json* rp = tc.get("rope_parameters");
            const double prf = rp && rp->get("partial_rotary_factor") ? rp->get("partial_rotary_factor")->num : num("partial_rotary_factor", 1.0);
            cfg_.n_rot = (int) std::lround(cfg_.head_dim * prf);
            cfg_.rope_base = (float) (rp && rp->get("rope_theta") ? rp->get("rope_theta")->num : num("rope_theta", 10000.0));
            cfg_.ssm_d_conv = (int) num("linear_conv_kernel_dim", 4);
            cfg_.ssm_state = (int) num("linear_key_head_dim", 128);
            cfg_.ssm_k_heads = (int) num("linear_num_key_heads", 0);
            cfg_.ssm_v_heads = (int) num("linear_num_value_heads", 0);
            cfg_.rms_eps = (float) num("rms_norm_eps", 1e-6);
            cfg_.eos_id = (int32_t) num("eos_token_id", -1);
            cfg_.n_nextn = (int) num("mtp_num_hidden_layers", 0);
            if (cfg_.n_layer < 1 || cfg_.n_embd < 1 || num("num_experts", 0) > 0 ||
                num("linear_value_head_dim", 128) != cfg_.ssm_state) {
                err = "dense model: " + path + ": not a dense Qwen3.5 / 3.8 checkpoint this engine reads";
                return false;
            }
            I.index_exl3(cfg_.n_layer, tc);
            // the value heads in the checkpoint's own (grouped) order: value head h reads key head h / (h_v / h_k)
            q35_set_gdn_grouped(cfg_.ssm_v_heads / std::max(1, cfg_.ssm_k_heads));
        } else {
            q35_set_gdn_grouped(0);
            // ---- 1. the files and the tensor index
            for (const std::string& p : shard_paths(path)) I.files.emplace_back(new strata::GgufFile(p));
            for (const auto& f : I.files)
                for (const auto& t : f->tensors()) I.index[t.name] = Found{f.get(), &t};
            const strata::GgufFile& g0 = *I.files.front();

            // ---- 2. the configuration, from the model's own metadata
            const strata::MetaValue* arch = g0.get("general.architecture");
            if (!arch) { err = "dense model: missing general.architecture"; return false; }
            cfg_.arch = arch->s;
            const std::string a = cfg_.arch + ".";
            if (cfg_.arch == "gemma4") return load_gemma4(max_context, err);
            if (cfg_.arch == "muse-glimmer") return load_muse(max_context, err);
            if (cfg_.arch.rfind("qwen3", 0) != 0) {
                err = "dense model: architecture '" + cfg_.arch + "' is not a qwen35 dense model";
                return false;
            }
            int64_t v = 0;
            if (g0.get(a + "expert_count") && g0.get(a + "expert_count")->u > 0) {   // qwen35moe: a MoE FFN in every layer
                if (!strata::prefill::mmq::built()) { err = "dense model: a MoE model needs the MMQ build (STRATA_GGML_DIR)"; return false; }
                cfg_.n_expert = (int) g0.get(a + "expert_count")->u;
                if (meta_int(g0, a + "expert_used_count", v)) cfg_.n_expert_used = (int) v;
                if (meta_int(g0, a + "expert_feed_forward_length", v)) cfg_.n_ff_exp = (int) v;
                if (meta_int(g0, a + "expert_shared_feed_forward_length", v)) cfg_.n_ff_shexp = (int) v;
                if (const auto* e = g0.get(a + "expert_weights_scale")) cfg_.expert_weights_scale = (float) e->num();
                if (cfg_.n_expert_used < 1 || cfg_.n_ff_exp < 1 || cfg_.n_expert > 1024) { err = "dense model: MoE metadata"; return false; }
                cfg_.n_ff = std::max(cfg_.n_ff_exp, cfg_.n_ff_shexp);
            }
            auto need = [&](const char* key, int& dst) {
                if (!meta_int(g0, a + key, v)) { err = std::string("dense model: missing ") + a + key; return false; }
                dst = (int) v;
                return true;
            };
            if (!need("block_count", cfg_.n_layer) || !need("embedding_length", cfg_.n_embd) ||
                (!cfg_.n_expert && !need("feed_forward_length", cfg_.n_ff)) || !need("attention.head_count", cfg_.n_head) ||
                !need("attention.head_count_kv", cfg_.n_head_kv) || !need("attention.key_length", cfg_.head_dim) ||
                !need("rope.dimension_count", cfg_.n_rot) || !need("ssm.conv_kernel", cfg_.ssm_d_conv) ||
                !need("ssm.state_size", cfg_.ssm_state) || !need("ssm.group_count", cfg_.ssm_k_heads) ||
                !need("ssm.time_step_rank", cfg_.ssm_v_heads))
                return false;
            // block_count includes the MTP/nextn layer(s) stored at the end of the file: they are not part of the main
            // forward pass (llama.cpp uses n_layer_all - n_layer_nextn), running them corrupts the hidden state.
            if (meta_int(g0, a + "nextn_predict_layers", v) && v > 0 && v < cfg_.n_layer) {
                cfg_.n_nextn = (int) v;
                cfg_.n_layer -= (int) v;
            }
            if (const auto* e = g0.get(a + "attention.layer_norm_rms_epsilon")) cfg_.rms_eps = (float) e->num();
            if (const auto* e = g0.get(a + "rope.freq_base")) cfg_.rope_base = (float) e->num();
            if (const auto* e = g0.get("tokenizer.ggml.eos_token_id")) cfg_.eos_id = (int32_t) e->u;
            if (meta_int(g0, a + "ssm.inner_size", v) && v != cfg_.value_dim()) {
                err = "dense model: ssm.inner_size " + std::to_string(v) + " != state_size * time_step_rank";
                return false;
            }
        }
        if (cfg_.ssm_state != 128 || cfg_.ssm_d_conv != 4 || cfg_.ssm_v_heads % cfg_.ssm_k_heads != 0) {
            err = "dense model: the GDN kernels are written for state_size 128 and conv_kernel 4";
            return false;
        }
        if (cfg_.head_dim != 256 || cfg_.n_head % cfg_.n_head_kv != 0 || cfg_.n_head / cfg_.n_head_kv > 8 ||
            cfg_.n_rot > cfg_.head_dim || cfg_.n_rot % 2 != 0) {
            err = "dense model: unsupported attention geometry (head size 256, up to 8 query heads a KV head)";
            return false;
        }
        if (cfg_.n_embd % 256 != 0 || cfg_.n_ff % 256 != 0 || cfg_.value_dim() % 256 != 0 ||
            (cfg_.n_head * cfg_.head_dim) % 256 != 0) {
            err = "dense model: every GEMV input width must be a multiple of 256 (K-quant super-block)";
            return false;
        }
        const Found* tok = I.find("token_embd.weight");
        if (!tok || tok->tensor->shape.size() != 2 || tok->tensor->shape[0] != (uint64_t) cfg_.n_embd) {
            err = "dense model: token_embd.weight missing or not [n_embd, n_vocab]";
            return false;
        }
        cfg_.n_vocab = (int) tok->tensor->shape[1];
        max_context_ = max_context;
        if (max_context < 16) { err = "dense model: context must be at least 16"; return false; }

        cudaError_t cs = cudaStreamCreate(&I.stream);
        if (cs != cudaSuccess) { err = std::string("dense model: CUDA setup: ") + cudaGetErrorString(cs); return false; }
        if (I.st && !strata::kernels::exl3::prepare(I.stream)) { err = "dense model: EXL3 kernel setup"; return false; }

        // ---- 3. the layers
        uint64_t total = 0;
        const int C = cfg_.conv_channels(), V = cfg_.value_dim(), H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;
        const int E = cfg_.n_embd, NL = cfg_.n_layer;
        const bool want_mtp = cfg_.n_nextn > 0 && I.find("blk." + std::to_string(NL) + ".nextn.eh_proj.weight") &&
                              std::getenv("STRATA_DENSE_NO_MTP") == nullptr;
        {   // How much of the weights VRAM can take: what is free, minus the KV caches, the activations and a margin.
            int n_attn = want_mtp ? 1 : 0;
            for (int l = 0; l < NL; ++l)
                if (I.find("blk." + std::to_string(l) + ".attn_q.weight")) ++n_attn;
            const uint64_t kv_need = (uint64_t) n_attn * q35_kv_bytes(I.kv_mode, HK, D, (int) max_context) +
                                     (I.kv_mode == 2 ? q35_kv_bytes(0, HK, D, (int) max_context) : 0);   // (+ the FP16 copy)
            auto act_of = [&](int ch) {
                return (uint64_t) ch * (uint64_t) (8 * E + 3 * C + 3 * V + 4 * H * D + 3 * cfg_.n_ff) * 4 +
                       (uint64_t) cfg_.n_ff * E * 2 + (uint64_t) kMaxCols * cfg_.n_vocab * 4 * 2 +
                       (uint64_t) q35_attn_scratch_floats(kMaxCols, H, D, (int) max_context) * 4;
            };
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const uint64_t margin = 768ull << 20;
            if (auto_chunk && !std::getenv("STRATA_DENSE_GPU_MIB")) {
                uint64_t file_bytes = 0;   // (the weights' upper bound: the files, the host-side embedding included)
                if (I.st) {
                    for (const auto& kv : I.st->tensors()) file_bytes += kv.second.bytes;
                } else {
                    for (const std::string& p : shard_paths(path)) {
                        std::error_code ec;
                        file_bytes += (uint64_t) std::filesystem::file_size(p, ec);
                    }
                }
                for (const int ch : {4096, 2048, 1024})
                    if ((uint64_t) free_b > kv_need + act_of(ch) + margin + file_bytes) { I.chunk = ch; break; }
            }
            const uint64_t act = act_of(I.chunk);
            if (const char* e = std::getenv("STRATA_DENSE_GPU_MIB")) {       // explicit cap for the weights, in MiB
                I.budget = (uint64_t) std::atoll(e) << 20;
            } else if ((uint64_t) free_b > kv_need + act + margin) {
                I.budget = (uint64_t) free_b - kv_need - act - margin;
            } else {
                err = "dense model: the KV cache for a context of " + std::to_string(max_context) + " needs " +
                      std::to_string(kv_need >> 20) + " MiB, but only " + std::to_string(free_b >> 20) +
                      " MiB of VRAM are free. Use a shorter --context.";
                return false;
            }
            std::fprintf(stderr, "strata-dense: VRAM %zu MiB free of %zu; KV cache %llu MiB (context %lld), activations "
                         "%llu MiB (prompt chunk %d); %llu MiB of weights may go to the GPU\n",
                         free_b >> 20, total_b >> 20, (unsigned long long) (kv_need >> 20), (long long) max_context,
                         (unsigned long long) (act >> 20), I.chunk, (unsigned long long) (I.budget >> 20));
        }
        auto load_attn = [&](Layer& L, const std::string& p) {
            if (!I.upload_mat(I.find(p + "attn_q.weight"), p + "attn_q.weight", E, H * 2 * D, L.q, total, err) ||
                !I.upload_mat(I.find(p + "attn_k.weight"), p + "attn_k.weight", E, HK * D, L.k, total, err) ||
                !I.upload_mat(I.find(p + "attn_v.weight"), p + "attn_v.weight", E, HK * D, L.v, total, err) ||
                !I.upload_mat(I.find(p + "attn_output.weight"), p + "attn_output.weight", H * D, E, L.o, total, err))
                return false;
            if (!I.upload_floats(I.find(p + "attn_q_norm.weight"), p + "attn_q_norm.weight", D, &L.q_norm, err) ||
                !I.upload_floats(I.find(p + "attn_k_norm.weight"), p + "attn_k_norm.weight", D, &L.k_norm, err))
                return false;
            L.kv.mode = I.kv_mode;
            L.kv.max_ctx = (int) max_context;
            const uint64_t cells = (uint64_t) HK * (uint64_t) max_context;
            if (I.kv_mode == 0) {
                // (+ 32 cells, all zero: the prompt attention reads whole 32-key tiles of K / V in place - the cells past
                // a tile's last key must be finite, and the last head's tile may reach past its region)
                if (!I.alloc(&L.kv.k, (cells + 32) * D * 2, err, "K cache") || !I.alloc(&L.kv.v, (cells + 32) * D * 2, err, "V cache"))
                    return false;
                cudaMemset(L.kv.k, 0, (cells + 32) * D * 2);
                cudaMemset(L.kv.v, 0, (cells + 32) * D * 2);
            } else if (!I.alloc(&L.kv.k, cells * D, err, "K cache") ||
                       !I.alloc(&L.kv.v, I.kv_mode == 2 ? cells * D : cells * D / 2, err, "V cache") ||
                       !I.alloc(&L.kv.ks, cells * (D / 32) * 2, err, "K scales") ||
                       !I.alloc(&L.kv.vs, cells * (D / 32) * 2, err, "V scales")) {
                return false;
            }
            kv_bytes_ += q35_kv_bytes(I.kv_mode, HK, D, (int) max_context);
            static const bool expand = !std::getenv("STRATA_DENSE_KV_EXPAND") || std::atoi(std::getenv("STRATA_DENSE_KV_EXPAND")) != 0;
            if (I.kv_mode == 2 && expand && !I.kv16_k) {   // (one layer's FP16 copy, shared by the layers: + 32 zero cells)
                if (!I.alloc(&I.kv16_k, (cells + 32) * D * 2, err, "K cache FP16 copy") ||
                    !I.alloc(&I.kv16_v, (cells + 32) * D * 2, err, "V cache FP16 copy"))
                    return false;
                cudaMemset(I.kv16_k, 0, (cells + 32) * D * 2);
                cudaMemset(I.kv16_v, 0, (cells + 32) * D * 2);
            }
            return true;
        };
        auto load_common = [&](Layer& L, const std::string& p) {
            if (!I.upload_floats(I.find(p + "attn_norm.weight"), p + "attn_norm.weight", E, &L.attn_norm, err)) return false;
            if (!I.upload_floats(I.find_any({"post_attention_norm.weight", "ffn_norm.weight"}, p), p + "post_attention_norm.weight",
                                 E, &L.post_norm, err)) return false;
            if (cfg_.n_expert) {
                const int NE = cfg_.n_expert, FF = cfg_.n_ff_exp;
                {   // the router's rows, then the shared expert's gate row (one GEMV for both)
                    float* r = nullptr;
                    if (!I.upload_floats(I.find(p + "ffn_gate_inp.weight"), p + "ffn_gate_inp.weight", (uint64_t) NE * E, &r, err) ||
                        !I.alloc(&L.router, (uint64_t) (NE + 1) * E * 4, err, "router"))
                        return false;
                    cudaMemcpy(L.router, r, (size_t) NE * E * 4, cudaMemcpyDeviceToDevice);
                    cudaMemset(L.router + (size_t) NE * E, 0, (size_t) E * 4);
                    cudaFree(r);
                    I.allocations.erase(std::find(I.allocations.begin(), I.allocations.end(), (void*) r));
                }
                auto raw3 = [&](const char* n, int n_in, int n_out, void** dev, int& type) {
                    const Found* f = I.find(p + n);
                    if (!f || f->tensor->shape.size() != 3 || f->tensor->shape[0] != (uint64_t) n_in ||
                        f->tensor->shape[1] != (uint64_t) n_out || f->tensor->shape[2] != (uint64_t) NE ||
                        !strata::prefill::mmq::supported((int) f->tensor->type)) {
                        err = "dense model: " + p + n + " missing, not [" + std::to_string(n_in) + ", " + std::to_string(n_out) +
                              ", experts] or a type MMQ does not take";
                        return false;
                    }
                    type = (int) f->tensor->type;
                    const uint64_t b = strata::prefill::mmq::matrix_bytes(type, n_out, n_in) * NE;
                    if (!I.alloc(dev, b, err, n)) return false;
                    cudaMemcpy(*dev, f->file->tensor_data(*f->tensor), b, cudaMemcpyHostToDevice);
                    I.dev_used += b;
                    total += b;
                    return true;
                };
                if (!raw3("ffn_gate_exps.weight", E, FF, &L.exp_gate, L.type_gate) ||
                    !raw3("ffn_up_exps.weight", E, FF, &L.exp_up, L.type_up) ||
                    !raw3("ffn_down_exps.weight", FF, E, &L.exp_down, L.type_down))
                    return false;
                if (cfg_.n_ff_shexp > 0) {
                    const int SF = cfg_.n_ff_shexp;
                    if (!I.upload_mat(I.find(p + "ffn_gate_shexp.weight"), p + "ffn_gate_shexp.weight", E, SF, L.sh_gate_m, total, err) ||
                        !I.upload_mat(I.find(p + "ffn_up_shexp.weight"), p + "ffn_up_shexp.weight", E, SF, L.sh_up, total, err) ||
                        !I.upload_mat(I.find(p + "ffn_down_shexp.weight"), p + "ffn_down_shexp.weight", SF, E, L.sh_down, total, err))
                        return false;
                    if (I.find(p + "ffn_gate_inp_shexp.weight")) {
                        float* g = nullptr;
                        if (!I.upload_floats(I.find(p + "ffn_gate_inp_shexp.weight"), p + "ffn_gate_inp_shexp.weight", E, &g, err))
                            return false;
                        cudaMemcpy(L.router + (size_t) NE * E, g, (size_t) E * 4, cudaMemcpyDeviceToDevice);
                        cudaFree(g);
                        I.allocations.erase(std::find(I.allocations.begin(), I.allocations.end(), (void*) g));
                        L.sh_gate = L.router + (size_t) NE * E;   // (a marker: the gate logit is the router's last row)
                    }
                    // gate and up stacked in one packed matrix when both are packed and alike
                    if (L.sh_gate_m.packed && L.sh_up.packed && L.sh_gate_m.type == L.sh_up.type) {
                        const uint64_t b1 = q35_mmt_packed_bytes(L.sh_gate_m.type, E, SF);
                        void* gu = nullptr;
                        if (SF % 32 == 0 && I.alloc(&gu, 2 * b1, err, "shared gate/up")) {
                            cudaMemcpy(gu, L.sh_gate_m.dev, b1, cudaMemcpyDeviceToDevice);
                            cudaMemcpy((uint8_t*) gu + b1, L.sh_up.dev, b1, cudaMemcpyDeviceToDevice);
                            L.sh_gu = QMat{gu, L.sh_gate_m.type, E, 2 * SF, true};
                            for (QMat* m : {&L.sh_gate_m, &L.sh_up}) {
                                cudaFree(m->dev);
                                I.allocations.erase(std::find(I.allocations.begin(), I.allocations.end(), m->dev));
                                m->dev = nullptr;
                            }
                        }
                    }
                }
                return true;
            }
            return I.upload_mat(I.find(p + "ffn_gate.weight"), p + "ffn_gate.weight", E, cfg_.n_ff, L.ffn_gate, total, err) &&
                   I.upload_mat(I.find(p + "ffn_up.weight"), p + "ffn_up.weight", E, cfg_.n_ff, L.ffn_up, total, err) &&
                   I.upload_mat(I.find(p + "ffn_down.weight"), p + "ffn_down.weight", cfg_.n_ff, E, L.ffn_down, total, err);
        };
        I.layers.assign((size_t) NL, Layer{});
        for (int l = 0; l < NL; ++l) {
            Layer& L = I.layers[(size_t) l];
            const std::string p = "blk." + std::to_string(l) + ".";
            const std::string at = "layer " + std::to_string(l) + ": ";
            const bool has_gdn = I.find(p + "attn_qkv.weight") != nullptr;
            const bool has_attn = I.find(p + "attn_q.weight") != nullptr;
            if (has_gdn == has_attn) { err = "dense model: " + at + "needs exactly one of attn_qkv.weight / attn_q.weight"; return false; }
            L.attn = has_attn;
            if (!load_common(L, p)) return false;
            if (!L.attn) {
                if (!I.upload_mat(I.find(p + "attn_qkv.weight"), p + "attn_qkv.weight", E, C, L.qkv, total, err) ||
                    !I.upload_mat(I.find(p + "attn_gate.weight"), p + "attn_gate.weight", E, V, L.z, total, err) ||
                    !I.upload_mat(I.find(p + "ssm_out.weight"), p + "ssm_out.weight", V, E, L.ssm_out, total, err))
                    return false;
                if (!I.upload_floats(I.find(p + "ssm_conv1d.weight"), p + "ssm_conv1d.weight", (uint64_t) C * cfg_.ssm_d_conv, &L.conv_w, err) ||
                    !I.upload_floats(I.find_any({"ssm_dt.bias", "ssm_dt"}, p), p + "ssm_dt.bias", cfg_.ssm_v_heads, &L.dt, err) ||
                    !I.upload_floats(I.find_any({"ssm_a", "ssm_a.weight"}, p), p + "ssm_a", cfg_.ssm_v_heads, &L.ssm_a, err) ||
                    !I.upload_floats(I.find(p + "ssm_norm.weight"), p + "ssm_norm.weight", cfg_.ssm_state, &L.ssm_norm, err))
                    return false;
                {   // alpha / beta: a float matrix in some files, Q8_0 (or another block type) in others
                    const Found* fa = I.find(p + "ssm_alpha.weight");
                    const Found* fb = I.find(p + "ssm_beta.weight");
                    if (!fa || !fb) { err = "dense model: " + at + "missing ssm_alpha.weight / ssm_beta.weight"; return false; }
                    auto is_float = [](uint32_t t) { return t == kF32 || t == kF16 || t == kBF16; };
                    if (is_float(fa->tensor->type) != is_float(fb->tensor->type)) {
                        err = "dense model: " + at + "ssm_alpha and ssm_beta have different kinds of type"; return false;
                    }
                    L.ab_quant = !is_float(fa->tensor->type);
                    if (L.ab_quant) {
                        if (!I.upload_mat(fa, p + "ssm_alpha.weight", E, cfg_.ssm_v_heads, L.alpha_q, total, err) ||
                            !I.upload_mat(fb, p + "ssm_beta.weight", E, cfg_.ssm_v_heads, L.beta_q, total, err))
                            return false;
                    } else {   // one matrix for both: [alpha rows | beta rows]
                        float *wa = nullptr, *wb = nullptr;
                        const uint64_t n1 = (uint64_t) E * cfg_.ssm_v_heads;
                        if (!I.upload_floats(fa, p + "ssm_alpha.weight", n1, &wa, err) ||
                            !I.upload_floats(fb, p + "ssm_beta.weight", n1, &wb, err) ||
                            !I.alloc(&L.ab_w, 2 * n1 * 4, err, "alpha/beta"))
                            return false;
                        cudaMemcpy(L.ab_w, wa, n1 * 4, cudaMemcpyDeviceToDevice);
                        cudaMemcpy(L.ab_w + n1, wb, n1 * 4, cudaMemcpyDeviceToDevice);
                        for (void* t : {(void*) wa, (void*) wb}) {
                            cudaFree(t);
                            I.allocations.erase(std::find(I.allocations.begin(), I.allocations.end(), t));
                        }
                    }
                }
                const uint64_t sbytes = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * kQ35StateBytes;
                const uint64_t cbytes = (uint64_t) C * (cfg_.ssm_d_conv - 1) * 4;
                if (!I.alloc(&L.state, sbytes, err, "GDN state") || !I.alloc(&L.conv_state, cbytes, err, "conv state") ||
                    !I.alloc(&L.st_raw, (uint64_t) kMaxCols * C * 4, err, "verify stash") ||
                    !I.alloc(&L.st_conv, (uint64_t) kMaxCols * C * 4, err, "verify stash") ||
                    !I.alloc(&L.st_ab, (uint64_t) kMaxCols * 2 * cfg_.ssm_v_heads * 4, err, "verify stash") ||
                    !I.alloc(&L.st_d, (uint64_t) kMaxCols * cfg_.ssm_v_heads * cfg_.ssm_state * 4, err, "verify stash"))
                    return false;
            } else if (!load_attn(L, p)) {
                return false;
            }
        }
        if (want_mtp) {   // the nextn block: a full-attention layer behind eh_proj . [enorm(emb) | hnorm(h)]
            Layer& M = I.mtp;
            const std::string p = "blk." + std::to_string(NL) + ".";
            if (!load_common(M, p) || !load_attn(M, p) ||
                !I.upload_mat(I.find(p + "nextn.eh_proj.weight"), p + "nextn.eh_proj.weight", 2 * E, E, M.eh_proj, total, err) ||
                !I.upload_floats(I.find(p + "nextn.enorm.weight"), p + "nextn.enorm.weight", E, &M.enorm, err) ||
                !I.upload_floats(I.find(p + "nextn.hnorm.weight"), p + "nextn.hnorm.weight", E, &M.hnorm, err))
                return false;
            const Found* hn = I.find(p + "nextn.shared_head_norm.weight");
            if (!I.upload_floats(hn ? hn : I.find("output_norm.weight"), "mtp head norm", E, &M.head_norm, err)) return false;
            M.attn = true;
        }

        // ---- 4. the input and output side
        // the embedding rows are read a token at a time (one row of E): pinned host memory (read over PCIe) when the
        // head has its own matrix - 0.7-1.3 GB of VRAM for the KV cache (STRATA_DENSE_EMBD_HOST=0: in VRAM)
        static const bool embd_host = !std::getenv("STRATA_DENSE_EMBD_HOST") || std::atoi(std::getenv("STRATA_DENSE_EMBD_HOST")) != 0;
        const bool untied = I.find("output.weight") != nullptr;
        if (!I.upload_mat(tok, "token_embd.weight", E, cfg_.n_vocab, I.embd, total, err, false, true, embd_host && untied))
            return false;
        if (!I.upload_floats(I.find("output_norm.weight"), "output_norm.weight", E, &I.output_norm, err)) return false;
        const Found* head = I.find("output.weight");
        if (!head) head = tok;                                   // tied embeddings
        if (!I.upload_mat(head, head == tok ? "token_embd.weight (tied head)" : "output.weight", E, cfg_.n_vocab,
                          I.head, total, err, head != tok, false))
            return false;
        I.head_draft = I.head;
        // the draft head scores the first N vocabulary rows (STRATA_DENSE_DRAFT_VOCAB; 0: all): 32768 by default on the
        // dense model (27B MTP3, two cooled rounds: 61.3 / 61.2 -> 63.4 / 63.4 tok/s, 2.56 -> 2.43 tokens a round;
        // 65536 no gain; DFlash2 unchanged); the MoE keeps the whole vocabulary (not measured there)
        {
            const char* e = std::getenv("STRATA_DENSE_DRAFT_VOCAB");
            const int dv = e ? std::atoi(e) : cfg_.n_expert == 0 ? 32768 : 0;
            if (dv > 0 && dv < cfg_.n_vocab) {
                I.head_draft.n_out = dv / 128 * 128;   // (rows are contiguous blocks)
                if (I.head.x3) {   // (EXL3: a view of the head's first columns)
                    I.x3mats.push_back(strata::kernels::exl3::view(*I.head.x3, 0, I.head_draft.n_out));
                    I.head_draft.x3 = &I.x3mats.back();
                }
            }
            const char* x = std::getenv("STRATA_DENSE_MTP_VOCAB_EXTRA");
            I.mtp_extra = I.head_draft.n_out < cfg_.n_vocab ? (x ? std::max(0, std::atoi(x)) / 32 * 32 : 4096) : 0;
        }
        weight_bytes_ = total;
        host_bytes_ = I.host_bytes;
        if (I.host_tensors)
            std::fprintf(stderr, "strata-dense: %d tensors (%.2f GiB of %.2f) live in host memory and are read over PCIe "
                         "on every token - a shorter --context or a smaller quant puts more on the GPU\n",
                         I.host_tensors, I.host_bytes / 1073741824.0, total / 1073741824.0);

        // ---- 5. activations and scratch
        const uint64_t R = (uint64_t) I.chunk, f = 4;
        const int max_in = std::max({E, cfg_.n_ff, V, H * D, 2 * E, 5 * E});   // (5 E: the DFlash2 drafter's fc)
        I.max_in = max_in;
        bool ok = I.alloc(&I.q8_1, native_q8_1_bytes(max_in, kMaxCols), err, "q8_1 scratch") &&
                  I.alloc(&I.xh, R * max_in * 2, err, "fp16 activations") &&
                  I.alloc(&I.x, R * E * f, err, "x") && I.alloc(&I.xn, R * E * f, err, "xn") &&
                  I.alloc(&I.mix, R * E * f, err, "mix") && I.alloc(&I.hn, R * E * f, err, "hidden") &&
                  I.alloc(&I.qkv, R * C * f, err, "qkv") && I.alloc(&I.conv, R * C * f, err, "conv") &&
                  I.alloc(&I.ab, R * 2 * cfg_.ssm_v_heads * f, err, "alpha/beta") && I.alloc(&I.o, R * V * f, err, "o") &&
                  I.alloc(&I.z, R * V * f, err, "z") && I.alloc(&I.y, R * V * f, err, "y") &&
                  I.alloc(&I.q_full, R * H * 2 * D * f, err, "q_full") && I.alloc(&I.qcur, R * H * D * f, err, "qcur") &&
                  I.alloc(&I.kcur, R * HK * D * f, err, "kcur") && I.alloc(&I.vcur, R * HK * D * f, err, "vcur") &&
                  I.alloc(&I.attn, R * H * D * f, err, "attn") &&
                  I.alloc(&I.ffn_g, R * std::max(cfg_.n_ff, 2 * cfg_.n_ff_shexp) * f, err, "ffn_g")   /* (the shared expert's [gate | up]) */ && I.alloc(&I.ffn_u, R * cfg_.n_ff * f, err, "ffn_u") &&
                  I.alloc(&I.ffn_h, R * cfg_.n_ff * f, err, "ffn_h") &&
                  I.alloc(&I.logits, (uint64_t) kMaxCols * cfg_.n_vocab * f, err, "logits") &&
                  I.alloc(&I.attn_scratch, (uint64_t) q35_attn_scratch_floats(kMaxCols, H, D, (int) max_context) * f, err,
                          "attention scratch") &&
                  I.alloc(&I.conv_dummy, (uint64_t) kMaxCols * C * f, err, "conv scratch") &&
                  I.alloc(&I.ab_tmp, R * 2 * cfg_.ssm_v_heads * f, err, "alpha/beta") &&
                  I.alloc(&I.tok_dev, R * 4, err, "tokens") &&
                  I.alloc(&I.mtp_x, (uint64_t) kMaxCols * E * f, err, "mtp") &&
                  I.alloc(&I.mtp_cat, (uint64_t) R * 2 * E * f, err, "mtp") &&
                  I.alloc(&I.mtp_h, R * E * f, err, "mtp") &&
                  I.alloc(&I.mtp_logits, (uint64_t) kMaxCols * cfg_.n_vocab * f, err, "mtp logits") &&
                  I.alloc(&I.xh8, (uint64_t) 8 * max_in * 2, err, "fp16 rows") &&
                  I.alloc(&I.mmt_part, (uint64_t) q35_mmt_part_floats(std::max({cfg_.n_vocab, cfg_.n_ff, H * 2 * D})) * f,
                          err, "GEMV partials");
        if (!ok) return false;
        if (cfg_.n_expert) {   // the MoE buffers: chunk x k (token, expert) pairs
            namespace mmq = strata::prefill::mmq;
            const uint64_t K = (uint64_t) cfg_.n_expert_used, PR = R * K, FF = (uint64_t) cfg_.n_ff_exp;
            ok = I.alloc(&I.moe_logits, R * (cfg_.n_expert + 1) * f, err, "moe") && I.alloc(&I.moe_w, PR * f, err, "moe") &&
                 I.alloc(&I.moe_ids, PR * 4, err, "moe") && I.alloc(&I.moe_bounds, (uint64_t) (cfg_.n_expert + 1) * 4, err, "moe") &&
                 I.alloc(&I.moe_tok, PR * 4, err, "moe") && I.alloc(&I.moe_pos, PR * 4, err, "moe") &&
                 I.alloc(&I.moe_max, 4, err, "moe") && I.alloc(&I.moe_cnt, (uint64_t) (cfg_.n_expert + 1) * 4, err, "moe") &&
                 I.alloc(&I.moe_iota, PR * 4, err, "moe") && I.alloc(&I.moe_pairtok, PR * 4, err, "moe") &&
                 I.alloc(&I.moe_g, PR * FF * f, err, "moe") &&
                 I.alloc(&I.moe_u, PR * FF * f, err, "moe") && I.alloc(&I.moe_y, PR * E * f, err, "moe") &&
                 I.alloc(&I.moe_sh, R * E * f, err, "moe") && I.alloc(&I.moe_shl, R * f, err, "moe") &&
                 I.alloc(&I.moe_xq, mmq::q8_bytes((int64_t) PR, std::max<int64_t>(E, (int64_t) FF)), err, "moe");
            if (!ok) return false;
            if (cudaHostAlloc((void**) &I.moe_max_host, 4, cudaHostAllocDefault) != cudaSuccess) { err = "moe: pinned"; return false; }
            mmq::iota(I.moe_iota, (int64_t) PR, I.stream);
            {   // pair p's token: p / k
                std::vector<int32_t> pt((size_t) PR);
                for (size_t i = 0; i < pt.size(); ++i) pt[i] = (int32_t) (i / K);
                cudaMemcpy(I.moe_pairtok, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice);
            }
            I.mmq.reset(new mmq::Context());
        }
        if (!I.gemm.init(I.stream, (int64_t) cfg_.n_ff * E, err)) return false;
        q35_set_gdn_scratch((float*) I.gemm.scratch(), I.gemm.scratch_elems() / 2);   // (the chunked GDN prompt form)
        if (std::getenv("STRATA_DENSE_VISION") && std::atoi(std::getenv("STRATA_DENSE_VISION")) != 0) {
            // the identity until a request has images; set before any decode graph is captured
            I.mrope_host.resize((size_t) max_context_ * 3);
            for (int64_t c = 0; c < max_context_; ++c)
                I.mrope_host[(size_t) c * 3] = I.mrope_host[(size_t) c * 3 + 1] = I.mrope_host[(size_t) c * 3 + 2] = (int32_t) c;
            if (!I.alloc(&I.mrope_dev, I.mrope_host.size() * 4, err, "image positions") ||
                cudaMemcpy(I.mrope_dev, I.mrope_host.data(), I.mrope_host.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
                return false;
            q35_set_mrope_table(I.mrope_dev);
        }
        if (const char* e = std::getenv("STRATA_DENSE_MULTI_EXACT")) native_mmvq_set_multi_exact(std::atoi(e) != 0);
        if (const char* e = std::getenv("STRATA_DENSE_MMT")) I.mmt_on = std::atoi(e) != 0;
        if (const char* e = std::getenv("STRATA_DENSE_MMT1")) I.mmt1 = std::atoi(e) != 0;
        if (const char* e = std::getenv("STRATA_DENSE_MMT_MIN")) I.mmt_min = std::max(1, std::atoi(e));
        if (const char* e = std::getenv("STRATA_DENSE_FUSE")) I.fuse = std::atoi(e) != 0;
        if (const char* e = std::getenv("STRATA_DENSE_COMMIT_REC")) I.commit_rec = std::atoi(e) != 0;
        if (const char* e = std::getenv("STRATA_DENSE_GDN_PIPE"); e && std::atoi(e) == 0) I.commit_rec = false;   // (records: pipelined kernel)
        if (const char* e = std::getenv("STRATA_DENSE_GDN_OLD"); e && std::atoi(e) != 0) I.commit_rec = false;
        if (const char* e = std::getenv("STRATA_DENSE_GRAPH")) I.graphs = std::atoi(e) != 0;
        if (const char* e = std::getenv("STRATA_DENSE_PACKED_DP4A")) I.packed_dp4a = std::atoi(e) != 0;
        I.pairs_mma = std::getenv("STRATA_DENSE_PAIRS_MMA") != nullptr;

        reset();
        std::string serr;
        if (!sync(serr)) { err = serr; return false; }
        std::fprintf(stderr, "strata-dense: %d layers%s, prompt chunks of %d tokens\n", NL,
                     I.mtp.attn ? " + the MTP block" : "", I.chunk);
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

void DenseModel::reset() {
    Impl& I = *impl_;
    const uint64_t sbytes = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * kQ35StateBytes;
    const uint64_t cbytes = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    for (Layer& L : I.layers) {
        if (L.state) cudaMemsetAsync(L.state, 0, sbytes, I.stream);
        if (L.conv_state) cudaMemsetAsync(L.conv_state, 0, cbytes, I.stream);
    }
    if (I.df.on) cudaMemsetAsync(I.df.ring_pos, 0xff, (size_t) I.df.window * 8, I.stream);   // the drafter's context
    I.ring_hi = -1;
    pos_ = 0;
    pending_ = 0;
}

bool DenseModel::sync(std::string& err) {
    const cudaError_t s = cudaStreamSynchronize(impl_->stream);
    if (s != cudaSuccess) { err = std::string("dense model: ") + cudaGetErrorString(s); return false; }
    return true;
}

namespace {

// The attention layer's mixer for T rows at p0 (xn prepared): mix = attn_output(gated attention), in three parts:
// the projections and the output projection read no position (a decode graph holds them), the core does
void attn_proj(DenseModel::Impl& I, Layer& L) {
    I.mm(L.q, I.q_full);
    I.mm(L.k, I.kcur);
    I.mm(L.v, I.vcur);
    I.stage("attn_qkv", I.stream);
}
void attn_core(DenseModel::Impl& I, const DenseConfig& c, Layer& L, int64_t p0, int T) {
    const int H = c.n_head, HK = c.n_head_kv, D = c.head_dim;
    q35_attn_prep(I.q_full, H * 2 * D, I.kcur, I.vcur, HK * D, L.q_norm, L.k_norm, c.rms_eps, c.rope_base, c.n_rot,
                  (int) p0, T, H, HK, D, I.qcur, L.kv, I.stream);
    Q35Kv kv = L.kv;
    // (a prompt chunk of 64+ tokens: its keys unpacked once, not once a query tile; a shorter one reads K8V8 directly -
    // 16 tokens at 68K 1.5 vs 2.1 ms)
    if (T >= 64 && L.kv.mode == 2 && I.kv16_k) {
        q35_kv_expand_f16(L.kv, (int) p0 + T, HK, D, I.kv16_k, I.kv16_v, I.stream);
        kv = Q35Kv{I.kv16_k, I.kv16_v, nullptr, nullptr, 0, L.kv.max_ctx};
    }
    // (key splits: the automatic choice for every width - a short prompt chunk over a long cache fills the GPU with
    // them, a long one gets 1; ideas.md 38)
    q35_attn(I.qcur, kv, (int) p0, T, H, HK, D, I.q_full, H * 2 * D, I.attn, I.attn_scratch, 0, I.stream);
    I.stage("attn_core", I.stream);
}
void attn_out(DenseModel::Impl& I, const DenseConfig& c, Layer& L, int T) {
    I.prep(I.attn, c.n_head * c.head_dim, T);
    I.mm(L.o, I.mix);
    I.stage("attn_out", I.stream);
}

// x += ffn(rms(x, post_norm)), T rows
// the MoE FFN of T rows (xn prepared): x += sum_j w_j expert_j(xn) + sigmoid(g . xn) shared(xn)
void moe_ffn(DenseModel::Impl& I, const DenseConfig& c, Layer& L, float* x, int T) {
    namespace mmq = strata::prefill::mmq;
    cudaStream_t s = I.stream;
    const int E = c.n_embd, K = c.n_expert_used, NE = c.n_expert, FF = c.n_ff_exp, P = T * K;
    const int NR = NE + (L.sh_gate ? 1 : 0);   // (the last row: the shared expert's gate logit)
    q35_gemv_f32_cols(L.router, I.xn, E, I.moe_logits, E, NR, T, s);
    q35_moe_route_ld(I.moe_logits, NR, T, NE, K, c.expert_weights_scale, I.moe_ids, I.moe_w, s);
    const bool pairs = P <= 64 && q35_mmt_supported(L.type_gate) && q35_mmt_supported(L.type_up) &&
                       q35_mmt_supported(L.type_down) && FF % 256 == 0;
    const bool dp4a = pairs && T < I.mmt_min && native_mmvq_pairs_supported(L.type_gate) &&
                      native_mmvq_pairs_supported(L.type_up) && native_mmvq_pairs_supported(L.type_down) &&
                      (int64_t) P * FF <= (int64_t) kMaxCols * I.max_in && !I.pairs_mma;
    if (dp4a) {   // a window of 1-3 tokens: dp4a GEMVs a (token, expert) pair (the tensor-core form pads to 8 columns)
        native_quantize_q8_1(I.xn, I.q8_1, E, T, s);
        I.stage("moe_route", s);
        if (L.type_up == L.type_gate) {   // one launch for both
            native_mmvq_pairs(L.type_gate, L.exp_gate, mmq::matrix_bytes(L.type_gate, FF, E), I.moe_ids, I.moe_pairtok,
                              I.q8_1, I.moe_g, E, FF, P, s, L.exp_up, I.moe_u);
        } else {
            native_mmvq_pairs(L.type_gate, L.exp_gate, mmq::matrix_bytes(L.type_gate, FF, E), I.moe_ids, I.moe_pairtok,
                              I.q8_1, I.moe_g, E, FF, P, s);
            native_mmvq_pairs(L.type_up, L.exp_up, mmq::matrix_bytes(L.type_up, FF, E), I.moe_ids, I.moe_pairtok,
                              I.q8_1, I.moe_u, E, FF, P, s);
        }
        I.stage("ex_gate_up", s);
        dense_swiglu(I.moe_g, I.moe_u, I.moe_g, (int64_t) P * FF, s);
        native_quantize_q8_1(I.moe_g, I.q8_1, P * FF, 1, s);   // (the pairs' rows back to back: one long row)
        I.stage("ex_swiglu", s);
        native_mmvq_pairs(L.type_down, L.exp_down, mmq::matrix_bytes(L.type_down, E, FF), I.moe_ids, I.moe_iota,
                          I.q8_1, I.moe_y, FF, E, P, s);
        I.q8_ready = false;   // (the shared expert quantizes its rows again)
        I.stage("moe_experts", s);
    } else if (pairs) {   // a decode / verify window: each (token, expert) pair its own GEMV, no grouping
        q35_to_half(I.xn, I.xh, (int64_t) T * E, s);
        if (T <= 8) {   // (the shared expert's GEMV reads the same FP16 rows)
            cudaMemcpyAsync(I.xh8, I.xh, (size_t) T * E * 2, cudaMemcpyDeviceToDevice, s);
            if (T < 8) cudaMemsetAsync(I.xh8 + (size_t) T * E, 0, (size_t) (8 - T) * E * 2, s);
        }
        I.stage("moe_route", s);
        q35_mmt_pairs(L.type_gate, L.exp_gate, strata::prefill::mmq::matrix_bytes(L.type_gate, FF, E), I.moe_ids,
                      I.moe_pairtok, I.xh, I.moe_g, E, FF, P, I.mmt_part, s);
        I.stage("ex_gate", s);
        q35_mmt_pairs(L.type_up, L.exp_up, strata::prefill::mmq::matrix_bytes(L.type_up, FF, E), I.moe_ids,
                      I.moe_pairtok, I.xh, I.moe_u, E, FF, P, I.mmt_part, s);
        I.stage("ex_up", s);
        dense_swiglu(I.moe_g, I.moe_u, I.moe_g, (int64_t) P * FF, s);
        q35_to_half(I.moe_g, I.xh, (int64_t) P * FF, s);
        I.stage("ex_swiglu", s);
        q35_mmt_pairs(L.type_down, L.exp_down, strata::prefill::mmq::matrix_bytes(L.type_down, E, FF), I.moe_ids,
                      I.moe_iota, I.xh, I.moe_y, FF, E, P, I.mmt_part, s);
        I.stage("moe_experts", s);
    } else {
    q35_moe_group(I.moe_ids, T, K, NE, I.moe_bounds, I.moe_tok, I.moe_pos, I.moe_max, I.moe_cnt, s);
    int max_rows = P;   // (a launch grid's bound: the pairs when few, else the busiest expert's - read back)
    if (P > 64) {
        cudaMemcpyAsync(I.moe_max_host, I.moe_max, 4, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        max_rows = std::max(1, *I.moe_max_host);
    }
    I.stage("moe_route", s);
    // gate and up: the pairs' token rows quantized once (gate and up share a type in these files)
    mmq::quantize(I.xn, I.moe_tok, I.moe_xq, L.type_gate, E, E, P, s);
    mmq::Product pr;
    // the K-quant experts on the FP16 tensor cores on sm_70 (moe_tc_kq, register fragments; STRATA_DENSE_MOE_TC=0: MMQ) -
    // 35B prompt, two cooled rounds: 1.6K 702 / 691 -> 664 / 654 ms, 6K 2,329 / 2,325 -> 2,241 / 2,242, 28K 13.17 / 13.16 ->
    // 12.84 / 12.85 s; needles 3/3, greedy 128/128 vs llama.cpp
    static const bool tc_env = !(std::getenv("STRATA_DENSE_MOE_TC") && std::atoi(std::getenv("STRATA_DENSE_MOE_TC")) == 0);
    const auto run_pr = [&](const mmq::Product& q) {
        if (tc_env && strata::prefill::tckq::available() && strata::prefill::tckq::supported(q.type, q.w_cols, q.w_rows))
            strata::prefill::tckq::product(q.type, q.w, q.expert_bytes, q.w_cols, q.w_rows, q.n, q.bounds, q.max_rows, q.xq,
                                           q.total_rows, q.dst, q.ld_dst, s);
        else I.mmq->run(q, s);
    };
    pr.w = L.exp_gate; pr.type = L.type_gate; pr.w_rows = FF; pr.w_cols = E;
    pr.expert_bytes = mmq::matrix_bytes(L.type_gate, FF, E); pr.n = NE; pr.xq = I.moe_xq; pr.bounds = I.moe_bounds;
    pr.ids = I.moe_iota; pr.total_rows = P; pr.max_rows = max_rows; pr.dst = I.moe_g; pr.ld_dst = FF;
    run_pr(pr);
    if (L.type_up != L.type_gate) mmq::quantize(I.xn, I.moe_tok, I.moe_xq, L.type_up, E, E, P, s);
    pr.w = L.exp_up; pr.type = L.type_up; pr.expert_bytes = mmq::matrix_bytes(L.type_up, FF, E); pr.dst = I.moe_u;
    run_pr(pr);
    dense_swiglu(I.moe_g, I.moe_u, I.moe_g, (int64_t) P * FF, s);
    mmq::quantize(I.moe_g, nullptr, I.moe_xq, L.type_down, FF, FF, P, s);
    pr.w = L.exp_down; pr.type = L.type_down; pr.w_rows = E; pr.w_cols = FF;
    pr.expert_bytes = mmq::matrix_bytes(L.type_down, E, FF); pr.dst = I.moe_y; pr.ld_dst = E;
    run_pr(pr);
    I.stage("moe_experts", s);
    }
    // the shared expert and its gate
    const bool sh = L.sh_up.dev != nullptr || L.sh_gu.dev != nullptr;
    const int SF = c.n_ff_shexp;
    if (sh) {
        I.prep(I.xn, E, T);
        if (pairs && !dp4a && T <= 8) I.h8_ready = true;   // (xh8 already holds these rows)
        if (L.sh_gu.dev) {   // gate and up in one GEMV: rows [g | u] per token
            I.mm(L.sh_gu, I.ffn_g);
            I.stage("sh_gu", s);
            q35_swiglu_rows(I.ffn_g, I.ffn_g + SF, 2 * SF, I.ffn_h, SF, T, s);
            I.stage("sh_swiglu", s);
        } else {
            I.mm(L.sh_gate_m, I.ffn_g);
            I.mm(L.sh_up, I.ffn_u);
            dense_swiglu(I.ffn_g, I.ffn_u, I.ffn_h, (int64_t) T * SF, s);
        }
        I.prep(I.ffn_h, SF, T);
        I.mm(L.sh_down, I.moe_sh);
        I.stage("sh_down", s);
    }
    q35_moe_combine_ld(I.moe_y, pairs ? I.moe_iota : I.moe_pos, I.moe_w, T, K, E,
                       sh && L.sh_gate ? I.moe_logits + NE : nullptr, NR, sh ? I.moe_sh : nullptr,
                    x, true, s);
    I.stage("moe_shared", s);
}

// add: x += add first (the mixer's output), in the norm's kernel
void ffn(DenseModel::Impl& I, const DenseConfig& c, Layer& L, float* x, int T, const float* add = nullptr) {
    const int E = c.n_embd;
    if (add) q35_add_rms_rows(x, E, add, L.post_norm, I.xn, I.h8_out(T), T, E, c.rms_eps, I.stream);
    else q35_rms_rows(x, E, L.post_norm, I.xn, I.h8_out(T), T, E, c.rms_eps, I.stream);
    I.prep_h8(I.xn, E, T);
    I.stage("norm_prep", I.stream);
    if (c.n_expert) { moe_ffn(I, c, L, x, T); return; }
    I.mm(L.ffn_gate, I.ffn_g);
    I.mm(L.ffn_up, I.ffn_u);
    I.stage("ffn_gate_up", I.stream);
    if (I.h8_out(T)) {
        q35_swiglu_h(I.ffn_g, I.ffn_u, I.ffn_h, I.xh8, (int64_t) T * c.n_ff, I.stream);
        I.prep_h8(I.ffn_h, c.n_ff, T);
    } else {
        dense_swiglu(I.ffn_g, I.ffn_u, I.ffn_h, (int64_t) T * c.n_ff, I.stream);
        I.prep(I.ffn_h, c.n_ff, T);
    }
    I.mm(L.ffn_down, I.mix);
    q35_add(x, I.mix, (int64_t) T * E, I.stream);
}

}  // namespace

bool DenseModel::forward(const int32_t* tokens, int T, bool verify, int want, std::string& err) {
    Impl& I = *impl_;
    if (T < 1 || T > I.chunk) { err = "dense model: a forward pass takes 1.." + std::to_string(I.chunk) + " tokens"; return false; }
    if (verify && T > kMaxCols) { err = "dense model: a verify pass takes up to 8 tokens"; return false; }
    if (want == kAll && T > kMaxCols) { err = "dense model: logits of every row up to 8 rows"; return false; }
    if (pending_) { err = "dense model: a verify pass waits for commit()"; return false; }
    if (pos_ + T > max_context_) { err = "dense model: the context is full"; return false; }
    for (int t = 0; t < T; ++t)
        if (tokens[t] < 0 || tokens[t] >= cfg_.n_vocab) { err = "dense model: token id outside the vocabulary"; return false; }
    if (I.gemma) return forward_gemma4(tokens, T, verify, want, err);
    if (I.muse) return forward_muse(tokens, T, verify, want, err);
    cudaStream_t s = I.stream;
    const int C = cfg_.conv_channels(), V = cfg_.value_dim(), E = cfg_.n_embd;
    const int KH = cfg_.ssm_k_heads, VH = cfg_.ssm_v_heads;
    const float eps = cfg_.rms_eps;
    const int64_t p0 = pos_;
    try {
        // ---- the embedding rows, dequantized on the device (a prompt chunk's in one launch, its ids copied over)
        I.mark(s);
        if (T > kMaxCols && dequant_f32_dev_supported(I.embd.type)) {
            cudaMemcpyAsync(I.tok_dev, tokens, (size_t) T * 4, cudaMemcpyHostToDevice, s);
            dequant_f32_rows_dev(I.embd.type, I.embd.dev, I.tok_dev, T, E, I.x, s);
        } else {
            for (int t = 0; t < T; ++t)
                dequant_f32(I.embd.type, I.embd.dev, tokens[t], 1, E, I.x + (size_t) t * E, s);
        }
        if (!I.img_cells.empty()) {   // the images' rows over their tokens' embeddings, a copy a run of cells
            auto it = std::lower_bound(I.img_cells.begin(), I.img_cells.end(), p0);
            while (it != I.img_cells.end() && *it < p0 + T) {
                auto run = it;
                while (run + 1 != I.img_cells.end() && *(run + 1) == *run + 1 && *(run + 1) < p0 + T) ++run;
                const size_t r0 = (size_t) (it - I.img_cells.begin()), n = (size_t) (run - it) + 1;
                cudaMemcpyAsync(I.x + (size_t) (*it - p0) * E, I.img_rows.data() + r0 * E, n * E * 4,
                                cudaMemcpyHostToDevice, s);
                it = run + 1;
            }
        }
        I.stage("embed", s);

        auto post = [&](int l) {   // the layer from the mixer's output: the residual add, the FFN, the drafter taps
            Layer& L = I.layers[(size_t) l];
            if (L.attn) attn_out(I, cfg_, L, T);
            if (I.fuse && !cfg_.n_expert) {
                ffn(I, cfg_, L, I.x, T, I.mix);   // (the residual add in the norm's kernel)
            } else {
                q35_add(I.x, I.mix, (int64_t) T * E, s);
                ffn(I, cfg_, L, I.x, T);
            }
            I.stage("ffn", s);
            if (I.df.on)   // the drafter's features: the residual after its target layers, in their order
                for (size_t j = 0; j < I.df.target_layers.size(); ++j)
                    if (I.df.target_layers[j] == l)
                        cudaMemcpy2DAsync(I.df.feat + j * E, I.df.target_layers.size() * E * 4, I.x, (size_t) E * 4,
                                          (size_t) E * 4, T, cudaMemcpyDeviceToDevice, s);
        };
        // the pass as steps: the ones that read no position run as CUDA graphs (a segment between two attention
        // cores, one graph set per (rows, verify, logits) - the launches of a small model's decode cost more than
        // its kernels); the attention cores (their position, their grid) run as they are
        auto pre = [&](int l) {   // the layer up to the attention core (all of a GDN layer)
            Layer& L = I.layers[(size_t) l];
            q35_rms_rows(I.x, E, L.attn_norm, I.xn, I.h8_out(T), T, E, eps, s);
            I.prep_h8(I.xn, E, T);
            I.stage("norm_prep", s);
            if (L.attn) { attn_proj(I, L); return; }
            {
                // ---- gated delta net: a verify pass keeps the recurrence's inputs for commit()
                float* raw = verify ? L.st_raw : I.qkv;
                float* cv = verify ? L.st_conv : I.conv;
                float* ab = verify ? L.st_ab : I.ab;    // rows [alpha | beta] -> [g | b]
                I.mm(L.qkv, raw);
                I.mm(L.z, I.z);
                I.stage("gdn_qkvz", s);
                if (L.ab_quant) {   // (the quantized pair: two GEMVs into the halves of a row)
                    I.mm(L.alpha_q, I.ab_tmp);
                    I.mm(L.beta_q, I.ab_tmp + (size_t) T * VH);
                    cudaMemcpy2DAsync(ab, (size_t) 2 * VH * 4, I.ab_tmp, (size_t) VH * 4, (size_t) VH * 4, T,
                                      cudaMemcpyDeviceToDevice, s);
                    cudaMemcpy2DAsync(ab + VH, (size_t) 2 * VH * 4, I.ab_tmp + (size_t) T * VH, (size_t) VH * 4,
                                      (size_t) VH * 4, T, cudaMemcpyDeviceToDevice, s);
                } else if (T > kMaxCols) {   // (a prompt chunk: one F32 GEMM - the GEMV re-read the rows 8 tokens at a time)
                    I.gemm.f32(I.xn, E, L.ab_w, ab, T, 2 * VH, E, 2 * VH);
                } else {
                    q35_gemv_f32_cols(L.ab_w, I.xn, E, ab, E, 2 * VH, T, s);
                }
                I.stage("gdn_ab", s);
                // (decode / verify windows: one launch for conv + q / k L2 + gate / beta - its L2 is serial over the
                // window's tokens, so a prompt chunk keeps the three; STRATA_DENSE_GDN_FUSE=0: three always)
                static const bool fuse = !(std::getenv("STRATA_DENSE_GDN_FUSE") && std::atoi(std::getenv("STRATA_DENSE_GDN_FUSE")) == 0);
                if (!fuse || T > kMaxCols || !q35_gdn_prep(L.conv_state, raw, C, L.conv_w, cv, T, C, !verify, 2 * KH, eps, ab, ab + VH, 2 * VH,
                                           L.dt, L.ssm_a, VH, s)) {
                    q35_conv_seq(L.conv_state, raw, C, L.conv_w, cv, T, C, !verify, s);
                    q35_qk_l2(cv, C, T, 2 * KH, eps, s);
                    q35_gate_beta(ab, ab + VH, 2 * VH, L.dt, L.ssm_a, T, VH, s);
                }
                I.stage("gdn_conv", s);
                q35_gdn_seq(L.state, cv, C, ab, ab + VH, 2 * VH, I.o, T, KH, VH, !verify, s, verify && I.commit_rec ? L.st_d : nullptr);
                I.stage("gdn_rec", s);
                q35_gdn_out(I.o, I.z, L.ssm_norm, I.y, T * VH, eps, s);
                I.prep(I.y, V, T);
                I.mm(L.ssm_out, I.mix);
                I.stage("gdn_out", s);
            }
            post(l);
        };
        auto head = [&]() {   // the final norm of every row (the MTP block reads them), then the logits asked for
            q35_rms_rows(I.x, E, I.output_norm, I.hn, nullptr, T, E, eps, s);
            if (want != kNone) {
                const int rows = want == kAll ? T : 1;
                I.prep(want == kAll ? I.hn : I.hn + (size_t) (T - 1) * E, E, rows);
                I.mm(I.head, I.logits);
            }
        };
        const bool graphs = I.graphs && T <= kMaxCols && !I.prof.on;
        if (!graphs) {
            for (int l = 0; l < cfg_.n_layer; ++l) {
                pre(l);
                if (I.layers[(size_t) l].attn) { attn_core(I, cfg_, I.layers[(size_t) l], p0, T); post(l); }
            }
            head();
        } else {
            auto& set = I.graph_sets[(T * 2 + (verify ? 1 : 0)) * 4 + want];
            size_t seg = 0;
            int l = 0;
            bool after_core = false;   // the segment starts with the post part of layer l
            while (true) {
                // one segment: [post(l) if after_core], then pre() of the layers up to and including the next attention
                // layer's (or all, then the head)
                const int l0 = l;
                const bool ac0 = after_core;
                auto body = [&]() {
                    int k = l0;
                    if (ac0) post(k++);
                    while (k < cfg_.n_layer) {
                        pre(k);
                        if (I.layers[(size_t) k].attn) return;
                        ++k;
                    }
                    head();
                };
                int k = l0 + (ac0 ? 1 : 0);
                while (k < cfg_.n_layer && !I.layers[(size_t) k].attn) ++k;
                if (seg == set.size()) {
                    cudaGraph_t g = nullptr;
                    cudaGraphExec_t ge = nullptr;
                    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
                        throw std::runtime_error("decode graph: capture");
                    body();
                    if (cudaStreamEndCapture(s, &g) != cudaSuccess || cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0) != cudaSuccess)
                        throw std::runtime_error("decode graph: instantiate");
                    cudaGraphDestroy(g);
                    set.push_back(ge);
                }
                if (cudaGraphLaunch(set[seg++], s) != cudaSuccess) throw std::runtime_error("decode graph: launch");
                if (k >= cfg_.n_layer) break;
                attn_core(I, cfg_, I.layers[(size_t) k], p0, T);
                l = k;
                after_core = true;
            }
        }
        I.stage("head", s);
        I.prof_flush(T);
        if (verify) pending_ = T;
        else pos_ += T;
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

bool DenseModel::vision_enabled() const { return impl_->mrope_dev != nullptr; }

bool DenseModel::set_images(const std::vector<ImagePlace>& places, std::vector<float> rows, std::string& err) {
    Impl& I = *impl_;
    const int64_t E = cfg_.n_embd;
    if (!places.empty() && !I.mrope_dev) { err = "dense model: images need STRATA_DENSE_VISION=1 at load"; return false; }
    if (!I.mrope_dev) return true;
    int64_t n_img = 0;
    for (const ImagePlace& p : places) n_img += (int64_t) p.nx * p.ny;
    if ((int64_t) rows.size() != n_img * E) { err = "dense model: image rows do not match the image tokens"; return false; }
    std::vector<int64_t> cells;
    cells.reserve((size_t) n_img);
    std::vector<int32_t> tab(I.mrope_host.size());
    int64_t r = 0;   // the rotary position of the next text cell
    size_t k = 0;
    for (int64_t c = 0; c < max_context_;) {
        if (k < places.size() && places[k].cell == c) {
            const ImagePlace& p = places[k++];
            const int64_t n = (int64_t) p.nx * p.ny;
            if (c + n > max_context_) { err = "dense model: an image past the context"; return false; }
            for (int64_t j = 0; j < n; ++j) {
                tab[(size_t) (c + j) * 3] = (int32_t) r;
                tab[(size_t) (c + j) * 3 + 1] = (int32_t) (r + j / p.nx);
                tab[(size_t) (c + j) * 3 + 2] = (int32_t) (r + j % p.nx);
                cells.push_back(c + j);
            }
            c += n;
            r += std::max(p.nx, p.ny);
        } else {
            if (k < places.size() && places[k].cell < c) { err = "dense model: overlapping images"; return false; }
            tab[(size_t) c * 3] = tab[(size_t) c * 3 + 1] = tab[(size_t) c * 3 + 2] = (int32_t) r++;
            ++c;
        }
    }
    // upload from the first cell that changed (text only after images: back to the identity)
    size_t first = 0;
    while (first < tab.size() && tab[first] == I.mrope_host[first]) ++first;
    if (first < tab.size()) {
        first -= first % 3;
        if (cudaMemcpyAsync(I.mrope_dev + first, tab.data() + first, (tab.size() - first) * 4, cudaMemcpyHostToDevice,
                            I.stream) != cudaSuccess) { err = "dense model: image positions upload"; return false; }
        cudaStreamSynchronize(I.stream);
        I.mrope_host.swap(tab);
    }
    I.img_cells.swap(cells);
    I.img_rows.swap(rows);
    return true;
}

bool DenseModel::commit(int n, std::string& err) {
    Impl& I = *impl_;
    if (pending_ == 0 || n < 1 || n > pending_) { err = "dense model: commit without a matching verify pass"; return false; }
    const int C = cfg_.conv_channels(), KH = cfg_.ssm_k_heads, VH = cfg_.ssm_v_heads;
    try {
        // STRATA_DENSE_COMMIT_OLD=1: a conv + recurrence launch a layer (96 on the 27B) instead of two for all
        static const bool old = std::getenv("STRATA_DENSE_COMMIT_OLD") && std::atoi(std::getenv("STRATA_DENSE_COMMIT_OLD")) != 0;
        if (!old && !I.commit_tab && I.commit_n >= 0) {
            std::vector<strata::kernels::Q35CommitLayer> tab;
            for (Layer& L : I.layers)
                if (!L.attn) tab.push_back({L.conv_state, L.st_raw, L.state, L.st_conv, L.st_ab, L.st_d});
            if (tab.empty()) I.commit_n = -1;   // (no GDN layer: nothing to commit)
            else if (cudaMalloc(&I.commit_tab, tab.size() * sizeof(tab[0])) != cudaSuccess ||
                cudaMemcpy(I.commit_tab, tab.data(), tab.size() * sizeof(tab[0]), cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("commit table");
            if (!tab.empty()) I.commit_n = (int) tab.size();
        }
        if (!old) {
            if (I.commit_n > 0) q35_commit_all(I.commit_tab, I.commit_n, n, C, KH, VH, I.stream, I.commit_rec);
        } else {
            for (Layer& L : I.layers) {
                if (L.attn) continue;
                q35_conv_seq(L.conv_state, L.st_raw, C, L.conv_w, I.conv_dummy, n, C, true, I.stream);
                q35_gdn_seq(L.state, L.st_conv, C, L.st_ab, L.st_ab + VH, 2 * VH, I.o, n, KH, VH, true, I.stream);
            }
        }
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
    pos_ += n;
    pending_ = 0;
    return true;
}

bool DenseModel::mtp_forward(const int32_t* toks, const float* hidden, int64_t p0, int T, bool all_logits,
                             std::string& err) {
    Impl& I = *impl_;
    Layer& M = I.mtp;
    if (!M.attn) { err = "dense model: no MTP block"; return false; }
    if (I.gemma) {   // the assistant: the last row only (it keeps no cache), at the anchor's position
        if (T < 1 || pos_ < 1) { err = "gemma4: an assistant step needs context"; return false; }
        cudaStream_t s = I.stream;
        const int E = cfg_.n_embd, AE = I.asst_E, AH = I.asst_H, NL = cfg_.n_layer, HK = cfg_.n_head_kv;
        const float eps = cfg_.rms_eps;
        try {
            if (I.mtp_dev_tok) dequant_f32_dev(I.embd.type, I.embd.dev, I.mtp_dev_tok, E, I.mtp_cat, s);
            else dequant_f32(I.embd.type, I.embd.dev, toks[T - 1], 1, E, I.mtp_cat, s);
            g4_scale(I.mtp_cat, sqrtf((float) E), E, s);
            cudaMemcpyAsync(I.mtp_cat + E, hidden + (size_t) (T - 1) * E, (size_t) E * 4, cudaMemcpyDeviceToDevice, s);
            I.prep(I.mtp_cat, 2 * E, 1);
            I.mm(I.asst_pre, I.a_h);
            const int64_t pa = pos_;   // the anchor's position; keys [0, pa)
            for (auto& L : I.asst) {
                const Layer& src = I.layers[(size_t) (NL - (L.swa ? 2 : 1))];
                const Layer& KV = src.kv_src >= 0 ? I.layers[(size_t) src.kv_src] : src;
                q35_rms_rows(I.a_h, AE, L.attn_norm, I.a_n, nullptr, 1, AE, eps, s);
                I.prep(I.a_n, AE, 1);
                I.mm(L.q, I.a_q);
                const int KHK = KV.HK > 0 ? KV.HK : HK;   // (31B: 16 KV heads on the sliding layers, 4 on the full ones)
                g4_attn_prep(I.a_q, nullptr, nullptr, 1, AH, KHK, L.D, L.q_norm, nullptr, eps,
                             L.swa ? I.rope_swa : cfg_.rope_base, L.swa ? nullptr : I.rope_freqs, (int) pa, I.a_qr, G4Kv{}, s);
                // the query at pa over keys < pa: the attention's "position" pa - 1 (the window one shorter to match)
                g4_attn(I.a_qr, KV.g4kv, (int) pa - 1, 1, AH, KHK, L.D, L.swa ? I.n_swa - 1 : 0, 1.0f, I.a_att,
                        I.attn_scratch, s);
                I.prep(I.a_att, AH * L.D, 1);
                I.mm(L.o, I.a_o);
                q35_rms_rows(I.a_o, AE, L.post_attn, I.a_o, nullptr, 1, AE, eps, s);
                q35_add(I.a_h, I.a_o, AE, s);
                q35_rms_rows(I.a_h, AE, L.ffn_norm, I.a_n, nullptr, 1, AE, eps, s);
                I.prep(I.a_n, AE, 1);
                I.mm(L.gate, I.a_g);
                I.mm(L.up, I.a_u);
                g4_gelu_mul(I.a_g, I.a_u, I.a_g, I.asst_ff, s);
                I.prep(I.a_g, I.asst_ff, 1);
                I.mm(L.down, I.a_o);
                q35_rms_rows(I.a_o, AE, L.post_ffw, I.a_o, nullptr, 1, AE, eps, s);
                g4_add_scale(I.a_h, I.a_o, L.out_scale, AE, s);
            }
            q35_rms_rows(I.a_h, AE, I.asst_norm, I.a_n, nullptr, 1, AE, eps, s);
            I.prep(I.a_n, AE, 1);
            I.mm(I.asst_embd, I.mtp_logits);
            I.mm(I.asst_post, I.mtp_h);
            (void) p0; (void) all_logits;
            return true;
        } catch (const std::exception& e) {
            err = std::string("gemma4 assistant: ") + e.what();
            return false;
        }
    }
    if (T < 1 || T > I.chunk || p0 + T > max_context_) { err = "dense model: MTP rows outside the context"; return false; }
    if (all_logits && T > kMaxCols) { err = "dense model: MTP logits of every row up to 8 rows"; return false; }
    cudaStream_t s = I.stream;
    const int E = cfg_.n_embd;
    const float eps = cfg_.rms_eps;
    try {
        I.mark(s);
        // [enorm(emb(tok)) | hnorm(h)] -> eh_proj
        for (int t = 0; t < T; ++t) {
            if (I.mtp_dev_tok) dequant_f32_dev(I.embd.type, I.embd.dev, I.mtp_dev_tok, E, I.y + (size_t) t * E, s);
            else dequant_f32(I.embd.type, I.embd.dev, toks[t], 1, E, I.y + (size_t) t * E, s);
        }
        float* cat = I.mtp_cat;
        q35_rms_rows(I.y, E, M.enorm, I.xn, nullptr, T, E, eps, s);
        cudaMemcpy2DAsync(cat, (size_t) 2 * E * 4, I.xn, (size_t) E * 4, (size_t) E * 4, T, cudaMemcpyDeviceToDevice, s);
        q35_rms_rows(hidden, E, M.hnorm, I.xn, nullptr, T, E, eps, s);
        cudaMemcpy2DAsync(cat + E, (size_t) 2 * E * 4, I.xn, (size_t) E * 4, (size_t) E * 4, T, cudaMemcpyDeviceToDevice, s);
        I.prep(cat, 2 * E, T);
        float* x = T <= kMaxCols ? I.mtp_x : I.x;   // (a prompt's rows reuse the trunk's buffer: the trunk is done)
        I.mm(M.eh_proj, x);
        q35_rms_rows(x, E, M.attn_norm, I.xn, nullptr, T, E, eps, s);
        I.prep(I.xn, E, T);
        attn_proj(I, M);
        attn_core(I, cfg_, M, p0, T);
        attn_out(I, cfg_, M, T);
        q35_add(x, I.mix, (int64_t) T * E, s);
        ffn(I, cfg_, M, x, T);
        q35_rms_rows(x, E, M.head_norm, I.mtp_h, nullptr, T, E, eps, s);
        const int rows = all_logits ? T : 1;
        I.prep(all_logits ? I.mtp_h : I.mtp_h + (size_t) (T - 1) * E, E, rows);
        I.mm(I.head_mtp.dev ? I.head_mtp : I.head_draft, I.mtp_logits);
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

bool DenseModel::mtp_fill(const int32_t* toks, const float* hidden, int64_t p0, int T, std::string& err) {
    Impl& I = *impl_;
    Layer& M = I.mtp;
    // STRATA_DENSE_MTP_FILL=0: the full block (for A/B)
    static const bool fill = !std::getenv("STRATA_DENSE_MTP_FILL") || std::atoi(std::getenv("STRATA_DENSE_MTP_FILL")) != 0;
    if (I.gemma || !M.attn || !fill || I.mtp_dev_tok) return mtp_forward(toks, hidden, p0, T, false, err);
    if (T < 1 || T > I.chunk || p0 + T > max_context_) { err = "dense model: MTP rows outside the context"; return false; }
    cudaStream_t s = I.stream;
    const int E = cfg_.n_embd, H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;
    const float eps = cfg_.rms_eps;
    try {
        I.mark(s);
        // as mtp_forward up to the K / V projections; no Q, attention, output, FFN or logits
        if (T > kMaxCols && dequant_f32_dev_supported(I.embd.type)) {
            cudaMemcpyAsync(I.tok_dev, toks, (size_t) T * 4, cudaMemcpyHostToDevice, s);
            dequant_f32_rows_dev(I.embd.type, I.embd.dev, I.tok_dev, T, E, I.y, s);
        } else {
            for (int t = 0; t < T; ++t) dequant_f32(I.embd.type, I.embd.dev, toks[t], 1, E, I.y + (size_t) t * E, s);
        }
        float* cat = I.mtp_cat;
        q35_rms_rows(I.y, E, M.enorm, I.xn, nullptr, T, E, eps, s);
        cudaMemcpy2DAsync(cat, (size_t) 2 * E * 4, I.xn, (size_t) E * 4, (size_t) E * 4, T, cudaMemcpyDeviceToDevice, s);
        q35_rms_rows(hidden, E, M.hnorm, I.xn, nullptr, T, E, eps, s);
        cudaMemcpy2DAsync(cat + E, (size_t) 2 * E * 4, I.xn, (size_t) E * 4, (size_t) E * 4, T, cudaMemcpyDeviceToDevice, s);
        I.prep(cat, 2 * E, T);
        float* x = T <= kMaxCols ? I.mtp_x : I.x;
        I.mm(M.eh_proj, x);
        q35_rms_rows(x, E, M.attn_norm, I.xn, nullptr, T, E, eps, s);
        I.prep(I.xn, E, T);
        I.mm(M.k, I.kcur);
        I.mm(M.v, I.vcur);
        q35_attn_prep(nullptr, H * 2 * D, I.kcur, I.vcur, HK * D, M.q_norm, M.k_norm, eps, cfg_.rope_base, cfg_.n_rot,
                      (int) p0, T, H, HK, D, I.qcur, M.kv, s);
        I.stage("mtp_fill", s);
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

bool DenseModel::mtp_dev_ok() const { return dequant_f32_dev_supported(impl_->embd.type); }

bool DenseModel::mtp_forward_dev(const int32_t* d_tok, const float* hidden, int64_t p0, std::string& err) {
    const int32_t dummy = 0;
    impl_->mtp_dev_tok = d_tok;
    const bool ok = mtp_forward(&dummy, hidden, p0, 1, false, err);
    impl_->mtp_dev_tok = nullptr;
    return ok;
}

bool DenseModel::has_dflash() const { return impl_->df.on; }
int DenseModel::dflash_max() const { return impl_->df.on ? impl_->df.block - 1 : 0; }

bool DenseModel::dflash_load(const std::string& path, std::string& err) {
    Impl& I = *impl_;
    Dflash& F = I.df;
    try {
        I.df_file.reset(new strata::GgufFile(path));
        const strata::GgufFile& G = *I.df_file;
        const auto* arch = G.get("general.architecture");
        if (!arch || (arch->s != "qwen35-dflash-draft" && arch->s != "qwen35-dspark-draft" && arch->s != "dflash")) {
            err = "dflash: " + path + " is not a qwen35-dflash-draft / qwen35-dspark-draft / dflash GGUF";
            return false;
        }
        F.dspark = arch->s == "qwen35-dspark-draft";
        F.plain = arch->s == "dflash";
        const std::string a = arch->s + ".";
        auto num = [&](const std::string& k, double def) -> double {
            const auto* v = G.get(a + k);
            return v ? v->num() : def;
        };
        const int E = cfg_.n_embd;
        if ((int) num("embedding_length", 0) != E || (int) num("vocab_size", cfg_.n_vocab) != cfg_.n_vocab) {
            err = "dflash: the drafter's hidden size / vocabulary do not match the model";
            return false;
        }
        F.n_layer = (int) num("block_count", 0);
        F.n_ff = (int) num("feed_forward_length", cfg_.n_ff);
        F.n_head = (int) num("attention.head_count", 0);
        F.n_kv = (int) num("attention.head_count_kv", 0);
        F.block = (int) num("dflash.block_size", 8);
        F.taps = (int) num("dflash.dflash2.conv_kernel_size", 0);
        F.group = (int) num("dflash.dflash2.conv_group_size", 16);
        F.rank = (int) num(F.dspark ? "dspark.markov_rank" : "dflash.dflash2.selector_rank", 0);
        F.top_k = (int) num("dflash.dflash2.selector_top_k", 16);
        F.mask_tok = (int) num("dflash.mask_token_id", -1);
        F.eps = (float) num("attention.layer_norm_rms_epsilon", 1e-6);
        F.theta = (float) num("rope.freq_base", 1e7);
        F.window = (int) num("attention.sliding_window", 2048);
        if (const char* e = std::getenv("STRATA_DENSE_DFLASH_CTX")) F.window = std::max(64, std::atoi(e));
        F.n_cand = std::min(cfg_.n_vocab, (int) num("dflash.candidate_vocab", 248077));
        if (const auto* tl = G.get(a + "dflash.target_layer_ids"))
            for (const auto& it : tl->items) F.target_layers.push_back((int) it.u);
        if (F.plain) {   // its own key names
            F.block = (int) num("block_size", 16);
            if (const auto* m = G.get("tokenizer.ggml.mask_token_id")) F.mask_tok = (int) m->u;
            if (const auto* tl = G.get(a + "target_layers"))
                for (const auto& it : tl->items) F.target_layers.push_back((int) it.u);
            F.taps = 2;
            F.rank = 32;   // (unused)
        }
        if (F.dspark) {   // trained on the whole context, but a ring of 8K positions drafts faster at 28K (162.7 tok/s
                          // vs 134.5 with 32K: the longer attention costs more than the extra kept drafts gain)
            F.taps = 2;
            F.window = (int) std::min<int64_t>(max_context_, std::getenv("STRATA_DENSE_DFLASH_CTX") ? F.window : 8192);
        }
        if (F.n_layer < 1 || F.n_head < 1 || F.n_kv < 1 || F.taps != 2 || F.rank < 32 || F.rank > 1024 ||
            F.rank % 32 || F.top_k < 1 || F.top_k > 32 || F.mask_tok < 0 || F.target_layers.empty() ||
            F.block < 2 || F.block > (F.plain || F.dspark ? 16 : 9) || (int) num("attention.key_length", 128) != 128) {
            err = "dflash: unsupported drafter geometry (DFlash2 with a selector, head size 128, block 2..9)";
            return false;
        }
        for (int l : F.target_layers)
            if (l < 0 || l >= cfg_.n_layer) { err = "dflash: a target layer outside the model"; return false; }
        const int NF = (int) F.target_layers.size();
        uint64_t total = 0;
        auto find = [&](const std::string& n) -> const strata::TensorInfo* { return G.find(n); };
        auto f32 = [&](const std::string& n, uint64_t want, float** out) {
            const auto* t = find(n);
            if (!t) { err = "dflash: missing " + n; return false; }
            Found f{&G, t};
            return I.upload_floats(&f, n, want, out, err);
        };
        // an F16 / BF16 / F32 matrix -> Q8_0 on the device (the GEMVs and GEMMs take Q8_0); a quantized one as is
        auto mat = [&](const std::string& n, int n_in, int n_out, QMat& out, bool packable = true) {
            const auto* t = find(n);
            if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_in || t->shape[1] != (uint64_t) n_out) {
                err = "dflash: " + n + " missing or not " + std::to_string(n_in) + " x " + std::to_string(n_out);
                return false;
            }
            if (t->type != kF16) {
                Found f{&G, t};
                return I.upload_mat(&f, n, n_in, n_out, out, total, err);
            }
            const std::vector<uint8_t> q = f16_to_q8_0((const uint16_t*) G.tensor_data(*t), n_in, n_out);
            void* dev = nullptr;
            if (!I.alloc(&dev, q.size(), err, n.c_str())) return false;
            cudaMemcpy(dev, q.data(), q.size(), cudaMemcpyHostToDevice);
            out = QMat{dev, 8, n_in, n_out};
            total += q.size();
            if (packable && !I.pack(out, true, err)) return false;
            return true;
        };
        const int qd = F.n_head * 128, kvd = F.n_kv * 128, cdyn = 2 * F.taps * (E / F.group);
        if (!mat(F.plain ? "fc.weight" : "dflash.fc.weight", NF * E, E, F.fc) ||
            (!F.dspark && !F.plain && !mat("dflash.selector.hproj.weight", E, F.rank, F.hproj)) ||
            !f32(F.plain ? "enc.output_norm.weight" : "dflash.hidden_norm.weight", E, &F.hidden_norm) ||
            !f32("output_norm.weight", E, &F.out_norm))
            return false;
        if (F.dspark) {
            const auto* t1 = find("dspark.markov_w1");
            if (!mat("token_embd.weight", E, cfg_.n_vocab, F.ds_embd, false) || !mat("output.weight", E, cfg_.n_vocab, F.ds_head) ||
                !mat("dspark.markov_w2.weight", F.rank, cfg_.n_vocab, F.ds_w2) || !t1 || t1->type != kF16)
                return false;
            const size_t b1 = (size_t) F.rank * cfg_.n_vocab * 2;
            if (!I.alloc(&F.ds_w1, b1, err, "markov w1")) return false;
            cudaMemcpy(F.ds_w1, G.tensor_data(*t1), b1, cudaMemcpyHostToDevice);
            total += b1;
            if (!I.alloc(&F.ds_x, (uint64_t) F.rank * 4, err, "dspark") ||
                !I.alloc(&F.ds_lg, (uint64_t) cfg_.n_vocab * 4, err, "dspark") || !I.alloc(&F.ds_tok, 16 * 4, err, "dspark"))
                return false;
        }
        for (const char* n : {"dflash.selector.pred_cb", "dflash.selector.succ_cb"}) {
            if (F.dspark || F.plain) break;
            const auto* t = find(n);
            if (!t || t->type != kF16 || t->shape.size() != 2 || t->shape[0] != (uint64_t) F.rank ||
                t->shape[1] != (uint64_t) cfg_.n_vocab) {
                err = std::string("dflash: ") + n + " missing or not F16 " + std::to_string(F.rank) + " x vocabulary";
                return false;
            }
            uint16_t*& dst = std::string(n).find("pred") != std::string::npos ? F.pred : F.succ;
            const size_t b = (size_t) F.rank * cfg_.n_vocab * 2;
            if (!I.alloc(&dst, b, err, n)) return false;
            cudaMemcpy(dst, G.tensor_data(*t), b, cudaMemcpyHostToDevice);
            total += b;
        }
        F.layers.assign((size_t) F.n_layer, DfLayer{});
        for (int il = 0; il < F.n_layer; ++il) {
            DfLayer& L = F.layers[(size_t) il];
            const std::string P = "blk." + std::to_string(il) + ".";
            if (!f32(P + "attn_norm.weight", E, &L.attn_norm) || !f32(P + "ffn_norm.weight", E, &L.ffn_norm) ||
                !f32(P + "attn_q_norm.weight", 128, &L.q_norm) || !f32(P + "attn_k_norm.weight", 128, &L.k_norm) ||
                (!F.dspark && !F.plain && !f32(P + "attn_conv.base", (uint64_t) 2 * F.taps * E, &L.attn_base)) ||
                (!F.dspark && !F.plain && !f32(P + "ffn_conv.base", (uint64_t) 2 * F.taps * E, &L.ffn_base)) ||
                !mat(P + "attn_q.weight", E, qd, L.q) || !mat(P + "attn_k.weight", E, kvd, L.k) ||
                !mat(P + "attn_v.weight", E, kvd, L.v) || !mat(P + "attn_output.weight", qd, E, L.o) ||
                (!F.dspark && !F.plain && !mat(P + "attn_conv.proj.weight", E, cdyn, L.attn_conv)) ||
                (!F.dspark && !F.plain && !mat(P + "ffn_conv.proj.weight", E, cdyn, L.ffn_conv)) ||
                !mat(P + "ffn_gate.weight", E, F.n_ff, L.gate) || !mat(P + "ffn_up.weight", E, F.n_ff, L.up) ||
                !mat(P + "ffn_down.weight", F.n_ff, E, L.down))
                return false;
            const uint64_t rb = (uint64_t) F.window * kvd * 2;
            if (!I.alloc(&L.Kr, rb, err, "dflash ring") || !I.alloc(&L.Vr, rb, err, "dflash ring")) return false;
            total += 2 * rb;
        }
        const uint64_t R = (uint64_t) I.chunk, W = (uint64_t) F.block;
        if (!I.alloc(&F.ring_pos, (uint64_t) F.window * 8, err, "dflash ring") ||
            !I.alloc(&F.feat, R * NF * E * 4, err, "dflash features") || !I.alloc(&F.h, W * E * 4, err, "dflash") ||
            !I.alloc(&F.dyn, W * cdyn * 4, err, "dflash") || !I.alloc(&F.a, W * E * 4, err, "dflash") ||
            !I.alloc(&F.hf, W * E * 4, err, "dflash") ||
            !I.alloc(&F.logits, W * cfg_.n_vocab * 4, err, "dflash logits") ||
            !I.alloc(&F.gate, W * F.rank * 4, err, "dflash") ||
            !I.alloc(&F.cand_v, W * F.top_k * 4, err, "dflash") || !I.alloc(&F.cand_i, W * F.top_k * 4, err, "dflash") ||
            !I.alloc(&F.tok, W * 4, err, "dflash") || !I.alloc(&F.conf, W * 4, err, "dflash") ||
            !I.alloc(&F.u_dev, W * 4, err, "dflash") || !I.alloc(&F.q_dev, W * F.top_k * 4, err, "dflash") ||
            !I.alloc(&F.kc, std::max(R, W) * kvd * 4, err, "dflash k") || !I.alloc(&F.vc, std::max(R, W) * kvd * 4, err, "dflash v") ||
            !I.alloc(&F.attn_part, (uint64_t) q35_df_attn_scratch(F.n_head, (int) W, F.window), err, "dflash attn") ||
            !I.alloc(&F.topk_scratch, (uint64_t) q35_df_topk_scratch((int) W, F.top_k), err, "dflash"))
            return false;
        if (!I.alloc(&F.g, W * F.n_ff * 4, err, "dflash mlp") || !I.alloc(&F.u, W * F.n_ff * 4, err, "dflash mlp") ||
            !I.alloc(&F.hh, W * F.n_ff * 4, err, "dflash mlp"))
            return false;
        {   // the activation conversions sized for the widest input: the drafter's fc (taps x n_embd), its MLP
            const int need = std::max(NF * E, F.n_ff);
            if (need > I.max_in) {
                for (void* old : {(void*) I.xh, (void*) I.xh8, I.q8_1}) {
                    cudaFree(old);
                    I.allocations.erase(std::find(I.allocations.begin(), I.allocations.end(), old));
                }
                I.max_in = need;
                I.drop_graphs();
                if (!I.alloc(&I.xh, (uint64_t) I.chunk * need * 2, err, "fp16 activations") ||
                    !I.alloc(&I.xh8, (uint64_t) 8 * need * 2, err, "fp16 rows") ||
                    !I.alloc(&I.q8_1, native_q8_1_bytes(need, kMaxCols), err, "q8_1 scratch"))
                    return false;
            }
        }
        cudaMemset(F.ring_pos, 0xff, (size_t) F.window * 8);   // -1: no position
        if (cudaDeviceSynchronize() != cudaSuccess) { err = "dflash: upload failed"; return false; }
        F.on = true;
        std::fprintf(stderr, "strata-dense: %s drafter %s: %d layers, block %d, %.2f GiB (Q8_0), window %d, "
                     "target layers", F.dspark ? "DSpark" : F.plain ? "DFlash" : "DFlash2", path.c_str(), F.n_layer, F.block, total / 1073741824.0, F.window);
        for (int l : F.target_layers) std::fprintf(stderr, " %d", l);
        std::fprintf(stderr, "\n");
        return true;
    } catch (const std::exception& e) {
        err = std::string("dflash: ") + e.what();
        return false;
    }
}

bool DenseModel::dflash_inject(int n, int64_t p0, std::string& err) {
    Impl& I = *impl_;
    Dflash& F = I.df;
    if (!F.on || n <= 0) return true;
    cudaStream_t s = I.stream;
    const int E = cfg_.n_embd, NF = (int) F.target_layers.size(), kvd = F.n_kv * 128;
    try {
        // c = rms(fc . features, hidden_norm); per layer K = rope(rms_head(Wk c) * k_norm) at the row's position, V = Wv c
        I.prep(F.feat, NF * E, n);
        I.mm(F.fc, I.mix);
        q35_rms_rows(I.mix, E, F.hidden_norm, I.xn, nullptr, n, E, F.eps, s);
        I.prep(I.xn, E, n);
        for (DfLayer& L : F.layers) {
            I.mm(L.k, F.kc);
            I.mm(L.v, F.vc);
            q35_df_head_norm_rope(F.kc, n, F.n_kv, L.k_norm, F.eps, p0, F.theta, s);
            // rows beyond the window would overwrite each other's slots: keep the last `window`
            const int skip = std::max(0, n - F.window);
            q35_df_ring_store(F.kc + (size_t) skip * kvd, F.vc + (size_t) skip * kvd, n - skip, kvd, p0 + skip,
                              F.window, L.Kr, L.Vr, F.ring_pos, s);
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dflash: ") + e.what();
        return false;
    }
}

const std::vector<float>& DenseModel::dflash_conf() const { return impl_->df.conf_h; }
bool DenseModel::dflash_samples() const { return impl_->df.on && !impl_->df.dspark && !impl_->df.plain; }
void DenseModel::dflash_sample(const std::vector<float>& u, float temperature) {
    impl_->df.sample = true;
    impl_->df.u_h = u;
    impl_->df.temp = temperature;
}
void DenseModel::dflash_proposals(std::vector<float>& q, std::vector<int32_t>& cand, int& k) const {
    q = impl_->df.q_h;
    cand = impl_->df.cand_h;
    k = impl_->df.top_k;
}

const char* DenseModel::dflash_kind() const { return impl_->df.dspark ? "DSpark" : impl_->df.plain ? "DFlash" : "DFlash2"; }

bool DenseModel::dflash_draft(int32_t anchor, int K, std::vector<int32_t>& out, std::string& err) {
    Impl& I = *impl_;
    Dflash& F = I.df;
    out.clear();
    F.conf_h.clear();
    if (!F.on || K < 1) return true;
    K = std::min(K, F.block - 1);
    // DSpark: every row drafts (row k the token after d_(k-1)); the block as trained, its first K rows' drafts kept
    K = std::min(K, kMaxCols - 1);   // (a verify pass takes up to 8 rows)
    // DSpark: the block as trained; plain DFlash: up to 8 rows (STRATA_DENSE_DFLASH_ROWS; more rows leave the GEMVs
    // for the dequantizing GEMM - 16 rows cost 20 ms a draft on Muse Glimmer's 6656-wide drafter)
    static const int plain_rows = std::getenv("STRATA_DENSE_DFLASH_ROWS") ? std::atoi(std::getenv("STRATA_DENSE_DFLASH_ROWS")) : kMaxCols;
    const int W = F.dspark ? F.block : F.plain ? std::max(K + 1, std::min(F.block, plain_rows)) : K + 1;
    const int E = cfg_.n_embd, kvd = F.n_kv * 128, qd = F.n_head * 128;
    const int cdyn = 2 * F.taps * (E / F.group);
    const int64_t P0 = pos_;
    cudaStream_t s = I.stream;
    try {
        // h = emb([anchor, mask x K])
        const QMat& emb = F.dspark ? F.ds_embd : I.embd;
        for (int t = 0; t < W; ++t)
            dequant_f32(emb.type, emb.dev, t == 0 ? anchor : F.mask_tok, 1, E, F.h + (size_t) t * E, s);
        if (F.dspark || F.plain) {
            for (DfLayer& L : F.layers) {   // plain pre-norm layers: no convolutions
                q35_rms_rows(F.h, E, L.attn_norm, I.xn, nullptr, W, E, F.eps, s);
                I.prep(I.xn, E, W);
                I.mm(L.q, I.q_full);
                I.mm(L.k, F.kc);
                I.mm(L.v, F.vc);
                q35_df_head_norm_rope(I.q_full, W, F.n_head, L.q_norm, F.eps, P0, F.theta, s);
                q35_df_head_norm_rope(F.kc, W, F.n_kv, L.k_norm, F.eps, P0, F.theta, s);
                q35_df_attn(I.q_full, L.Kr, L.Vr, F.ring_pos, F.kc, F.vc, W, F.n_head, F.n_kv, P0, F.window,
                            F.window, I.attn, s, F.attn_part);
                I.prep(I.attn, qd, W);
                I.mm(L.o, F.a);
                q35_add(F.h, F.a, (int64_t) W * E, s);
                q35_rms_rows(F.h, E, L.ffn_norm, I.xn, nullptr, W, E, F.eps, s);
                I.prep(I.xn, E, W);
                I.mm(L.gate, F.g);
                I.mm(L.up, F.u);
                dense_swiglu(F.g, F.u, F.hh, (int64_t) W * F.n_ff, s);
                I.prep(F.hh, F.n_ff, W);
                I.mm(L.down, F.a);
                q35_add(F.h, F.a, (int64_t) W * E, s);
            }
            q35_rms_rows(F.h, E, F.out_norm, F.hf, nullptr, W, E, F.eps, s);
            if (F.plain) {   // rows 1..K through the target's head, their arg-max
                const int V = cfg_.n_vocab;
                I.prep(F.hf + E, E, K);
                I.mm(I.head, F.logits);
                q35_argmax_rows(F.logits, V, V, K, F.tok, s);
                out.resize((size_t) K);
                cudaMemcpyAsync(out.data(), F.tok, (size_t) K * 4, cudaMemcpyDeviceToHost, s);
                if (cudaStreamSynchronize(s) != cudaSuccess) { err = "dflash: draft failed"; return false; }
                return true;
            }
            // base logits of the K rows through the drafter's own head; the Markov chain on the device:
            // d_k = argmax(base_k + W2 . W1[d_(k-1)]), d_(-1) = the anchor
            const int V = cfg_.n_vocab;
            I.prep(F.hf, E, K);
            I.mm(F.ds_head, F.logits);
            cudaMemcpyAsync(F.ds_tok, &anchor, 4, cudaMemcpyHostToDevice, s);
            for (int k = 0; k < K; ++k) {
                q35_df_gather_f16(F.ds_w1, F.rank, F.ds_tok + k, F.ds_x, s);
                I.prep(F.ds_x, F.rank, 1);
                I.mm(F.ds_w2, F.ds_lg);
                q35_add(F.ds_lg, F.logits + (size_t) k * V, V, s);
                q35_argmax_rows(F.ds_lg, V, V, 1, F.ds_tok + k + 1, s);
            }
            out.resize((size_t) K);
            cudaMemcpyAsync(out.data(), F.ds_tok + 1, (size_t) K * 4, cudaMemcpyDeviceToHost, s);
            if (cudaStreamSynchronize(s) != cudaSuccess) { err = "dspark: draft failed"; return false; }
            return true;
        }
        for (DfLayer& L : F.layers) {
            // attention, wrapped in the attention conv
            q35_rms_rows(F.h, E, L.attn_norm, I.xn, nullptr, W, E, F.eps, s);
            I.prep(I.xn, E, W);
            I.mm(L.attn_conv, F.dyn);
            q35_df_conv(I.xn, F.dyn, cdyn, L.attn_base, 0, W, E, F.taps, F.group, F.a, s);
            I.prep(F.a, E, W);
            I.mm(L.q, I.q_full);
            I.mm(L.k, F.kc);
            I.mm(L.v, F.vc);
            q35_df_head_norm_rope(I.q_full, W, F.n_head, L.q_norm, F.eps, P0, F.theta, s);
            q35_df_head_norm_rope(F.kc, W, F.n_kv, L.k_norm, F.eps, P0, F.theta, s);
            q35_df_attn(I.q_full, L.Kr, L.Vr, F.ring_pos, F.kc, F.vc, W, F.n_head, F.n_kv, P0, F.window,
                        F.window, I.attn, s, F.attn_part);
            I.prep(I.attn, qd, W);
            I.mm(L.o, I.mix);
            q35_df_conv(I.mix, F.dyn, cdyn, L.attn_base, 1, W, E, F.taps, F.group, F.a, s);
            q35_add(F.h, F.a, (int64_t) W * E, s);
            // the MLP, wrapped in the MLP conv
            q35_rms_rows(F.h, E, L.ffn_norm, I.xn, nullptr, W, E, F.eps, s);
            I.prep(I.xn, E, W);
            I.mm(L.ffn_conv, F.dyn);
            q35_df_conv(I.xn, F.dyn, cdyn, L.ffn_base, 0, W, E, F.taps, F.group, F.a, s);
            I.prep(F.a, E, W);
            I.mm(L.gate, F.g);
            I.mm(L.up, F.u);
            dense_swiglu(F.g, F.u, F.hh, (int64_t) W * F.n_ff, s);
            I.prep(F.hh, F.n_ff, W);
            I.mm(L.down, I.mix);
            q35_df_conv(I.mix, F.dyn, cdyn, L.ffn_base, 1, W, E, F.taps, F.group, F.a, s);
            q35_add(F.h, F.a, (int64_t) W * E, s);
        }
        (void) kvd;
        q35_rms_rows(F.h, E, F.out_norm, F.hf, nullptr, W, E, F.eps, s);
        // the mask rows' candidates through the target's head, their gates, the walk
        I.prep(F.hf + E, E, K);
        // (the draft head: the picked vocabulary when there is one - its slots mapped back to ids for the walk)
        const QMat& dh = I.head_mtp.dev ? I.head_mtp : I.head_draft;
        I.mm(dh, F.logits);
        q35_df_topk(F.logits, dh.n_out, std::min(F.n_cand, dh.n_out), K, F.top_k, F.cand_i, F.cand_v, F.topk_scratch, s);
        if (I.head_mtp.dev) q35_map_ids(F.cand_i, I.mtp_ids_dev, K * F.top_k, s);
        I.mm(F.hproj, F.gate);
        if (F.sample && (int) F.u_h.size() >= K) {   // (a sampled request: drawn, with the proposal distributions)
            cudaMemcpyAsync(F.u_dev, F.u_h.data(), (size_t) K * 4, cudaMemcpyHostToDevice, s);
            q35_df_walk_sample(F.cand_i, F.cand_v, F.gate, F.top_k, F.rank, F.pred, F.succ, anchor, K, F.u_dev, F.temp,
                               F.tok, F.conf, F.q_dev, s);
            F.q_h.resize((size_t) K * F.top_k);
            F.cand_h.resize((size_t) K * F.top_k);
            cudaMemcpyAsync(F.q_h.data(), F.q_dev, F.q_h.size() * 4, cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(F.cand_h.data(), F.cand_i, F.cand_h.size() * 4, cudaMemcpyDeviceToHost, s);
        } else {
            q35_df_walk(F.cand_i, F.cand_v, F.gate, F.top_k, F.rank, F.pred, F.succ, anchor, K, F.tok, F.conf, s);
        }
        F.sample = false;
        out.resize((size_t) K);
        cudaMemcpyAsync(out.data(), F.tok, (size_t) K * 4, cudaMemcpyDeviceToHost, s);
        F.conf_h.resize((size_t) K);
        cudaMemcpyAsync(F.conf_h.data(), F.conf, (size_t) K * 4, cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) { err = "dflash: draft failed"; return false; }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dflash: ") + e.what();
        return false;
    }
}

uint64_t DenseModel::ckpt_bytes() const {
    const Impl& I = *impl_;
    const uint64_t sb = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * kQ35StateBytes;
    const uint64_t cb = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    uint64_t n = 0;
    for (const Layer& L : I.layers)
        if (!L.attn) n += sb + cb;
    if (I.df.on) n += (uint64_t) I.df.n_layer * 2 * I.df.window * I.df.n_kv * 128 * 2 + (uint64_t) I.df.window * 8;
    return n;
}

namespace {
// every buffer of the recurrent state, in a fixed order: (device pointer, bytes)
std::vector<std::pair<void*, uint64_t>> ckpt_parts(DenseModel::Impl& I, const DenseConfig& c) {
    std::vector<std::pair<void*, uint64_t>> v;
    const uint64_t sb = (uint64_t) c.ssm_state * c.ssm_v_heads * c.ssm_state * kQ35StateBytes;
    const uint64_t cb = (uint64_t) c.conv_channels() * (c.ssm_d_conv - 1) * 4;
    for (Layer& L : I.layers)
        if (!L.attn) { v.push_back({L.state, sb}); v.push_back({L.conv_state, cb}); }
    if (I.df.on) {
        const uint64_t rb = (uint64_t) I.df.window * I.df.n_kv * 128 * 2;
        for (DfLayer& L : I.df.layers) { v.push_back({L.Kr, rb}); v.push_back({L.Vr, rb}); }
        v.push_back({I.df.ring_pos, (uint64_t) I.df.window * 8});
    }
    return v;
}
}  // namespace

bool DenseModel::ckpt_save(int slot, std::string& err) {
    Impl& I = *impl_;
    if (slot < 0 || pending_) { err = "dense model: checkpoint slot / pending verify"; return false; }
    if ((int) I.ckpt.size() <= slot) I.ckpt.resize((size_t) slot + 1, nullptr);
    if (!I.ckpt[(size_t) slot] &&
        cudaHostAlloc((void**) &I.ckpt[(size_t) slot], ckpt_bytes(), cudaHostAllocDefault) != cudaSuccess) {
        I.ckpt[(size_t) slot] = nullptr;
        err = "dense model: no pinned memory for a checkpoint";
        return false;
    }
    uint8_t* h = I.ckpt[(size_t) slot];
    for (auto& [p, b] : ckpt_parts(I, cfg_)) {
        cudaMemcpyAsync(h, p, b, cudaMemcpyDeviceToHost, I.stream);
        h += b;
    }
    return sync(err);
}

bool DenseModel::ckpt_restore(int slot, int64_t pos, std::string& err) {
    Impl& I = *impl_;
    if (slot < 0 || slot >= (int) I.ckpt.size() || !I.ckpt[(size_t) slot] || pos < 0 || pos > max_context_) {
        err = "dense model: no such checkpoint";
        return false;
    }
    // a sliding-window ring keeps positions (ring_hi - swa_ring, ring_hi]: the window behind `pos` must be among them
    if (I.swa_ring > 0 && pos - I.n_swa <= I.ring_hi - I.swa_ring) {
        err = "dense model: the checkpoint's window has left the sliding-window ring";
        return false;
    }
    const uint8_t* h = I.ckpt[(size_t) slot];
    for (auto& [p, b] : ckpt_parts(I, cfg_)) {
        cudaMemcpyAsync(p, h, b, cudaMemcpyHostToDevice, I.stream);
        h += b;
    }
    pos_ = pos;
    pending_ = 0;
    return sync(err);
}

// ---------------------------------------------------------------------------------------------------- Gemma 4
// llama.cpp src/models/gemma4.cpp (E2B / E4B: dense FFN, per-layer embeddings, shared K/V in the last layers)
bool DenseModel::load_gemma4(int64_t max_context, std::string& err) {
    Impl& I = *impl_;
    I.gemma = true;
    // GGUF rows by default: llama.cpp's dp4a GEMV at one token is faster here than the packed form and matches
    // llama.cpp's greedy tokens (128 / 128); STRATA_DENSE_PACK=1 packs
    if (!std::getenv("STRATA_DENSE_PACK")) I.pack_on = false;
    const strata::GgufFile& g0 = *I.files.front();
    const std::string a = "gemma4.";
    int64_t v = 0;
    auto need = [&](const char* key, int& dst) {
        if (!meta_int(g0, a + key, v)) { err = std::string("gemma4: missing ") + a + key; return false; }
        dst = (int) v;
        return true;
    };
    int D_full = 0, D_swa = 0, shared = 0;
    if (!need("block_count", cfg_.n_layer) || !need("embedding_length", cfg_.n_embd) ||
        !need("feed_forward_length", cfg_.n_ff) || !need("attention.head_count", cfg_.n_head) ||
        !need("attention.head_count_kv", cfg_.n_head_kv) || !need("attention.key_length", D_full) ||
        !need("attention.key_length_swa", D_swa) || !need("attention.sliding_window", I.n_swa))
        return false;
    // a sliding-window layer's cells: the window, a prompt chunk and a verify window (a chunk writes its rows before it
    // reads the window behind its first one), in steps of 256; the whole context when that is no smaller
    // (STRATA_DENSE_SWA_RING=0: always the whole context)
    auto swa_cells = [&](int64_t max_context) -> int64_t {
        static const bool ring_on = !std::getenv("STRATA_DENSE_SWA_RING") || std::atoi(std::getenv("STRATA_DENSE_SWA_RING")) != 0;
        const int64_t r = ((int64_t) I.n_swa + I.chunk + kMaxCols + 255) / 256 * 256;
        return ring_on && r < max_context ? r : max_context;
    };
    // (the E-series' per-layer inputs; the 31B has none)
    I.ple_dim = meta_int(g0, a + "embedding_length_per_layer_input", v) ? (int) v : 0;
    if (meta_int(g0, a + "attention.shared_kv_layers", v)) shared = (int) v;
    if (const auto* e = g0.get(a + "attention.layer_norm_rms_epsilon")) cfg_.rms_eps = (float) e->num();
    if (const auto* e = g0.get(a + "rope.freq_base")) cfg_.rope_base = (float) e->num();
    if (const auto* e = g0.get(a + "rope.freq_base_swa")) I.rope_swa = (float) e->num();
    if (const auto* e = g0.get(a + "final_logit_softcapping")) I.softcap = (float) e->num();
    if (const auto* e = g0.get("tokenizer.ggml.eos_token_id")) cfg_.eos_id = (int32_t) e->u;
    const auto* pat = g0.get(a + "attention.sliding_window_pattern");
    const int NL = cfg_.n_layer, E = cfg_.n_embd, H = cfg_.n_head, HK = cfg_.n_head_kv, P = I.ple_dim;
    if (!pat || (int) pat->items.size() < NL) {
        err = "gemma4: attention.sliding_window_pattern missing or short (the GGUF reader keeps 64 array items)";
        return false;
    }
    // K/V heads a layer: one number, or an array (the 31B: 16 on the sliding layers, 4 on the full ones)
    std::vector<int> hk_of((size_t) NL, HK);
    if (const auto* hkv = g0.get(a + "attention.head_count_kv"); hkv && (int) hkv->items.size() >= NL)
        for (int l = 0; l < NL; ++l) hk_of[(size_t) l] = (int) hkv->items[(size_t) l].u;
    int HK_max = 0;
    for (int l = 0; l < NL; ++l) {
        HK_max = std::max(HK_max, hk_of[(size_t) l]);
        if (hk_of[(size_t) l] <= 0 || H % hk_of[(size_t) l] || H / hk_of[(size_t) l] > 8) {
            err = "gemma4: layer " + std::to_string(l) + " has an unsupported K/V head count";
            return false;
        }
    }
    if ((P != 0 && P != 256) || (D_full != 256 && D_full != 512) || D_swa != 256 || E % 256 || cfg_.n_ff % 256) {
        err = "gemma4: unsupported geometry (per-layer inputs of 256 or none, heads 256 / 512, widths of 256)";
        return false;
    }
    cfg_.head_dim = D_full;
    cfg_.n_rot = D_full;
    const Found* tok = I.find("token_embd.weight");
    if (!tok || tok->tensor->shape.size() != 2 || tok->tensor->shape[0] != (uint64_t) E) {
        err = "gemma4: token_embd.weight missing or not [n_embd, n_vocab]";
        return false;
    }
    cfg_.n_vocab = (int) tok->tensor->shape[1];
    max_context_ = max_context;
    if (cudaStreamCreate(&I.stream) != cudaSuccess) { err = "gemma4: CUDA stream"; return false; }
    const int kv_from = NL - shared;   // layers from here use the K/V of the last earlier layer of their kind
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    {
        uint64_t kv_need = 0;
        for (int l = 0; l < kv_from; ++l) {
            const bool sw = pat->items[(size_t) l].u != 0;
            kv_need += (uint64_t) 2 * hk_of[(size_t) l] * (sw ? D_swa : D_full) * (sw ? swa_cells(max_context) : max_context) * 2;
        }
        const uint64_t act = (uint64_t) I.chunk * (8ull * E + 4ull * cfg_.n_ff + 4ull * NL * P + 4ull * H * D_full) * 4 +
                             (uint64_t) kMaxCols * cfg_.n_vocab * 4 + 1024ull * 1024 * 1024;
        if ((uint64_t) free_b < kv_need + act) {
            err = "gemma4: the K/V cache for a context of " + std::to_string(max_context) + " needs " +
                  std::to_string(kv_need >> 20) + " MiB; " + std::to_string(free_b >> 20) + " MiB are free";
            return false;
        }
        I.budget = (uint64_t) free_b - kv_need - act;
    }
    uint64_t total = 0;
    // a float matrix (F32 / F16 / BF16) as Q8_0 rows
    auto floatmat = [&](const std::string& n, int n_in, int n_out, QMat& out) {
        const Found* f = I.find(n);
        if (!f || f->tensor->shape.size() != 2 || f->tensor->shape[0] != (uint64_t) n_in || f->tensor->shape[1] != (uint64_t) n_out) {
            err = "gemma4: " + n + " missing or not [" + std::to_string(n_in) + ", " + std::to_string(n_out) + "]";
            return false;
        }
        if (f->tensor->type != kF32 && f->tensor->type != kF16 && f->tensor->type != kBF16)
            return I.upload_mat(f, n, n_in, n_out, out, total, err);
        const uint64_t cnt = (uint64_t) n_in * n_out;
        std::vector<float> host(cnt);
        const uint8_t* src = f->file->tensor_data(*f->tensor);
        if (f->tensor->type == kF32) std::memcpy(host.data(), src, cnt * 4);
        else if (f->tensor->type == kF16) strata::dequantize_f16(src, host.data(), (int) cnt);
        else strata::dequantize_bf16(src, host.data(), (int) cnt);
        const std::vector<uint8_t> q = f32_to_q8_0(host.data(), n_in, n_out);
        void* dev = nullptr;
        if (!I.alloc(&dev, q.size(), err, n.c_str())) return false;
        cudaMemcpy(dev, q.data(), q.size(), cudaMemcpyHostToDevice);
        out = QMat{dev, 8, n_in, n_out};
        total += q.size();
        return I.pack(out, true, err);
    };
    I.g4_f32 = std::getenv("STRATA_G4_F32") != nullptr;
    I.layers.resize((size_t) NL);
    for (int l = 0; l < NL; ++l) {
        Layer& L = I.layers[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        L.attn = true;
        L.swa = pat->items[(size_t) l].u != 0;
        L.D = L.swa ? D_swa : D_full;
        L.HK = hk_of[(size_t) l];
        const int D = L.D, HK = L.HK;
        if (l >= kv_from) {
            L.kv_src = kv_from - (L.swa ? 2 : 1);
            if (L.kv_src < 0 || I.layers[(size_t) L.kv_src].swa != L.swa) { err = "gemma4: shared K/V layer"; return false; }
        }
        if (!I.upload_floats(I.find(p + "attn_norm.weight"), p + "attn_norm.weight", E, &L.attn_norm, err) ||
            !I.upload_floats(I.find(p + "post_attention_norm.weight"), p + "post_attention_norm.weight", E, &L.post_norm, err) ||
            !I.upload_floats(I.find(p + "ffn_norm.weight"), p + "ffn_norm.weight", E, &L.ffn_pre_norm, err) ||
            !I.upload_floats(I.find(p + "post_ffw_norm.weight"), p + "post_ffw_norm.weight", E, &L.post_ffw_norm, err) ||
            !I.upload_floats(I.find(p + "attn_q_norm.weight"), p + "attn_q_norm.weight", D, &L.q_norm, err) ||
            !I.upload_mat(I.find(p + "attn_q.weight"), p + "attn_q.weight", E, H * D, L.q, total, err) ||
            !I.upload_mat(I.find(p + "attn_output.weight"), p + "attn_output.weight", H * D, E, L.o, total, err) ||
            !I.upload_mat(I.find(p + "ffn_gate.weight"), p + "ffn_gate.weight", E, cfg_.n_ff, L.ffn_gate, total, err) ||
            !I.upload_mat(I.find(p + "ffn_up.weight"), p + "ffn_up.weight", E, cfg_.n_ff, L.ffn_up, total, err) ||
            !I.upload_mat(I.find(p + "ffn_down.weight"), p + "ffn_down.weight", cfg_.n_ff, E, L.ffn_down, total, err))
            return false;
        if (P > 0 && (!I.upload_floats(I.find(p + "post_norm.weight"), p + "post_norm.weight", E, &L.ple_norm, err) ||
                      !floatmat(p + "inp_gate.weight", E, P, L.inp_gate) || !floatmat(p + "proj.weight", P, E, L.ple_proj)))
            return false;
        L.inp_gate.fp16_act = L.ple_proj.fp16_act = true;   // (the residual stream: Q8_1 rows lose too much, measured)
        if (P > 0 && I.g4_f32 && (!I.upload_floats(I.find(p + "inp_gate.weight"), p + "inp_gate.weight", (uint64_t) E * P, &L.inp_gate_f, err) ||
                         !I.upload_floats(I.find(p + "proj.weight"), p + "proj.weight", (uint64_t) E * P, &L.ple_proj_f, err)))
            return false;
        if (L.kv_src < 0) {
            if (!I.upload_floats(I.find(p + "attn_k_norm.weight"), p + "attn_k_norm.weight", D, &L.k_norm, err) ||
                !I.upload_mat(I.find(p + "attn_k.weight"), p + "attn_k.weight", E, HK * D, L.k, total, err))
                return false;
            const Found* fv = I.find(p + "attn_v.weight");
            if (fv && !I.upload_mat(fv, p + "attn_v.weight", E, HK * D, L.v, total, err)) return false;
            if (!fv) L.v = L.k;   // (no v_proj: V is K's projection)
            // a sliding-window layer keeps a ring of the window + a prompt chunk + a verify window (swa_cells)
            const int64_t per = L.swa ? swa_cells(max_context) : max_context;
            const uint64_t cells = (uint64_t) HK * per;
            // (+ 16 cells, all zero: the head-512 prompt kernel reads whole 16-key tiles of K / V in place - the cells
            // past a tile's last key must be finite, and the last head's tile may reach past its region)
            if (!I.alloc(&L.g4kv.k, (cells + 16) * D * 2, err, "K cache") || !I.alloc(&L.g4kv.v, (cells + 16) * D * 2, err, "V cache"))
                return false;
            cudaMemset(L.g4kv.k, 0, (cells + 16) * D * 2);
            cudaMemset(L.g4kv.v, 0, (cells + 16) * D * 2);
            L.g4kv.max_ctx = (int) per;
            L.g4kv.ring = per < max_context ? (int) per : 0;
            if (L.g4kv.ring) I.swa_ring = L.g4kv.ring;
            kv_bytes_ += cells * D * 4;
        }
        if (const Found* f = I.find(p + "layer_output_scale.weight")) L.out_scale = *(const float*) f->file->tensor_data(*f->tensor);
    }
    // the input and output side: the tied head reads token_embd's GGUF rows (the embedding rows are dequantized)
    if (!I.upload_mat(tok, "token_embd.weight", E, cfg_.n_vocab, I.embd, total, err, false) ||
        !I.upload_floats(I.find("output_norm.weight"), "output_norm.weight", E, &I.output_norm, err))
        return false;
    I.head = I.find("output.weight") ? QMat{} : I.embd;
    if (const Found* h = I.find("output.weight"))
        if (!I.upload_mat(h, "output.weight", E, cfg_.n_vocab, I.head, total, err, true, false)) return false;
    I.head_draft = I.head;
    if (const Found* rf = I.find("rope_freqs.weight"))
        if (!I.upload_floats(rf, "rope_freqs.weight", D_full / 2, &I.rope_freqs, err)) return false;
    if (P > 0) {   // per-layer token embeddings: GGUF rows of NL * P (dequantized per token), the model projection, its norm
        const Found* pt = I.find("per_layer_token_embd.weight");
        if (!pt || !I.upload_mat(pt, "per_layer_token_embd.weight", NL * P, cfg_.n_vocab, I.ple_tok, total, err, false) ||
            !floatmat("per_layer_model_proj.weight", E, NL * P, I.ple_model) ||
            !I.upload_floats(I.find("per_layer_proj_norm.weight"), "per_layer_proj_norm.weight", P, &I.ple_proj_norm, err))
            return false;
        I.ple_model.fp16_act = true;
        if (I.g4_f32 && !I.upload_floats(I.find("per_layer_model_proj.weight"), "per_layer_model_proj.weight",
                                         (uint64_t) E * NL * P, &I.ple_model_f, err))
            return false;
    }
    weight_bytes_ = total;
    host_bytes_ = I.host_bytes;
    // activations
    const uint64_t R = (uint64_t) I.chunk, f = 4;
    const int max_in = std::max({E, cfg_.n_ff, H * D_full, NL * P});
    I.max_in = max_in;
    bool ok = I.alloc(&I.q8_1, native_q8_1_bytes(max_in, kMaxCols), err, "q8_1 scratch") &&
              I.alloc(&I.xh, R * max_in * 2, err, "fp16 activations") &&
              I.alloc(&I.x, R * E * f, err, "x") && I.alloc(&I.xn, R * E * f, err, "xn") &&
              I.alloc(&I.mix, R * E * f, err, "mix") && I.alloc(&I.hn, R * E * f, err, "hidden") &&
              I.alloc(&I.q_full, R * H * D_full * f, err, "q") && I.alloc(&I.qcur, R * H * D_full * f, err, "qcur") &&
              I.alloc(&I.kcur, R * HK_max * D_full * f, err, "kcur") && I.alloc(&I.vcur, R * HK_max * D_full * f, err, "vcur") &&
              I.alloc(&I.attn, R * H * D_full * f, err, "attn") &&
              I.alloc(&I.ffn_g, R * cfg_.n_ff * f, err, "ffn_g") && I.alloc(&I.ffn_u, R * cfg_.n_ff * f, err, "ffn_u") &&
              I.alloc(&I.ffn_h, R * cfg_.n_ff * f, err, "ffn_h") &&
              (P == 0 || (I.alloc(&I.ple_rows, R * NL * P * f, err, "ple") && I.alloc(&I.ple_proj_out, R * NL * P * f, err, "ple") &&
                          I.alloc(&I.ple, R * NL * P * f, err, "ple") && I.alloc(&I.ple_g, R * P * f, err, "ple"))) &&
              I.alloc(&I.logits, (uint64_t) kMaxCols * cfg_.n_vocab * f, err, "logits") &&
              I.alloc(&I.attn_scratch, (uint64_t) g4_attn_scratch_floats(kMaxCols, H, D_full, (int) max_context) * f, err,
                      "attention scratch") &&
              I.alloc(&I.tok_dev, R * 4, err, "tokens") &&
              I.alloc(&I.xh8, (uint64_t) 8 * max_in * 2, err, "fp16 rows") &&
              I.alloc(&I.mmt_part, (uint64_t) q35_mmt_part_floats(std::max({cfg_.n_vocab, cfg_.n_ff, NL * P})) * f, err,
                      "GEMV partials");
    if (!ok) return false;
    if (!I.gemm.init(I.stream, std::max<int64_t>((int64_t) cfg_.n_ff * E, (int64_t) NL * P * E), err)) return false;
    if (const char* e = std::getenv("STRATA_DENSE_MMT_MIN")) I.mmt_min = std::max(1, std::atoi(e));
    if (const char* e = std::getenv("STRATA_DENSE_PACKED_DP4A")) I.packed_dp4a = std::atoi(e) != 0;
    if (const char* mp = std::getenv("STRATA_DENSE_MTP")) {   // the assistant
        I.asst_file.reset(new strata::GgufFile(mp));
        const strata::GgufFile& A = *I.asst_file;
        const auto* aa = A.get("general.architecture");
        if (!aa || aa->s != "gemma4-assistant") { err = std::string("gemma4: ") + mp + " is not a gemma4-assistant GGUF"; return false; }
        const std::string q = "gemma4-assistant.";
        auto anum = [&](const std::string& k) { const auto* m = A.get(q + k); return m ? (int) m->num() : 0; };
        I.asst_E = anum("embedding_length");
        I.asst_H = anum("attention.head_count");
        I.asst_ff = anum("feed_forward_length");
        const int AL = anum("block_count"), AD_f = anum("attention.key_length"), AD_s = anum("attention.key_length_swa");
        const auto* apat = A.get(q + "attention.sliding_window_pattern");
        // the assistant's KV head counts: one number (E4B) or one a layer (31B: [16, 16, 16, 4]) - each must be the
        // target's for that kind of layer (it reads their K/V)
        bool hk_ok = true;
        {
            int hk_swa = HK, hk_full = HK;
            for (int l = 0; l < NL; ++l) (pat->items[(size_t) l].u ? hk_swa : hk_full) = hk_of[(size_t) l];
            const auto* hkv = A.get(q + "attention.head_count_kv");
            const auto* ap = A.get(q + "attention.sliding_window_pattern");
            if (!hkv || !ap) hk_ok = false;
            else if (hkv->items.empty()) hk_ok = (int) hkv->num() == hk_swa && (int) hkv->num() == hk_full;
            else
                for (size_t l = 0; l < hkv->items.size() && l < ap->items.size(); ++l)
                    hk_ok = hk_ok && (int) hkv->items[l].u == (ap->items[l].u ? hk_swa : hk_full);
        }
        if (anum("embedding_length_out") != E || !hk_ok || AD_f != D_full || AD_s != D_swa ||
            !apat || (int) apat->items.size() < AL || I.asst_E % 256 || I.asst_ff % 256) {
            err = "gemma4: the assistant's geometry does not match the model (out " + std::to_string(anum("embedding_length_out")) +
                  " vs " + std::to_string(E) + ", kv heads " + (hk_ok ? "ok" : "differ") + ", head " + std::to_string(AD_f) + "/" +
                  std::to_string(AD_s) + " vs " + std::to_string(D_full) + "/" + std::to_string(D_swa) + ", layers " +
                  std::to_string(AL) + ", embd " + std::to_string(I.asst_E) + ", ff " + std::to_string(I.asst_ff) + ")";
            return false;
        }
        auto af = [&](const std::string& n) -> const Found* {
            static std::vector<std::unique_ptr<Found>> keep;
            const auto* t = A.find(n);
            if (!t) return nullptr;
            keep.emplace_back(new Found{&A, t});
            return keep.back().get();
        };
        const int AE = I.asst_E, AH = I.asst_H;
        I.asst.resize((size_t) AL);
        for (int l = 0; l < AL; ++l) {
            auto& L = I.asst[(size_t) l];
            const std::string p = "blk." + std::to_string(l) + ".";
            L.swa = apat->items[(size_t) l].u != 0;
            L.D = L.swa ? D_swa : D_full;
            if (!I.upload_floats(af(p + "attn_norm.weight"), p + "attn_norm.weight", AE, &L.attn_norm, err) ||
                !I.upload_floats(af(p + "attn_q_norm.weight"), p + "attn_q_norm.weight", L.D, &L.q_norm, err) ||
                !I.upload_floats(af(p + "post_attention_norm.weight"), p + "post_attention_norm.weight", AE, &L.post_attn, err) ||
                !I.upload_floats(af(p + "ffn_norm.weight"), p + "ffn_norm.weight", AE, &L.ffn_norm, err) ||
                !I.upload_floats(af(p + "post_ffw_norm.weight"), p + "post_ffw_norm.weight", AE, &L.post_ffw, err) ||
                !I.upload_mat(af(p + "attn_q.weight"), p + "attn_q.weight", AE, AH * L.D, L.q, total, err) ||
                !I.upload_mat(af(p + "attn_output.weight"), p + "attn_output.weight", AH * L.D, AE, L.o, total, err) ||
                !I.upload_mat(af(p + "ffn_gate.weight"), p + "ffn_gate.weight", AE, I.asst_ff, L.gate, total, err) ||
                !I.upload_mat(af(p + "ffn_up.weight"), p + "ffn_up.weight", AE, I.asst_ff, L.up, total, err) ||
                !I.upload_mat(af(p + "ffn_down.weight"), p + "ffn_down.weight", I.asst_ff, AE, L.down, total, err))
                return false;
            if (const auto* t = A.find(p + "layer_output_scale.weight")) L.out_scale = *(const float*) A.tensor_data(*t);
        }
        if (!I.upload_mat(af("nextn.pre_projection.weight"), "nextn.pre_projection.weight", 2 * E, AE, I.asst_pre, total, err) ||
            !I.upload_mat(af("nextn.post_projection.weight"), "nextn.post_projection.weight", AE, E, I.asst_post, total, err) ||
            !I.upload_mat(af("token_embd.weight"), "assistant token_embd.weight", AE, cfg_.n_vocab, I.asst_embd, total, err, false) ||
            !I.upload_floats(af("output_norm.weight"), "assistant output_norm.weight", AE, &I.asst_norm, err))
            return false;
        const int AQ = AH * D_full;
        if (!I.alloc(&I.a_h, (uint64_t) AE * 4, err, "asst") || !I.alloc(&I.a_n, (uint64_t) AE * 4, err, "asst") ||
            !I.alloc(&I.a_q, (uint64_t) AQ * 4, err, "asst") || !I.alloc(&I.a_qr, (uint64_t) AQ * 4, err, "asst") ||
            !I.alloc(&I.a_att, (uint64_t) AQ * 4, err, "asst") || !I.alloc(&I.a_o, (uint64_t) AE * 4, err, "asst") ||
            !I.alloc(&I.a_g, (uint64_t) I.asst_ff * 4, err, "asst") || !I.alloc(&I.a_u, (uint64_t) I.asst_ff * 4, err, "asst") ||
            !I.alloc(&I.mtp_cat, (uint64_t) 2 * E * 4, err, "asst") || !I.alloc(&I.mtp_h, (uint64_t) kMaxCols * E * 4, err, "asst") ||
            !I.alloc(&I.mtp_logits, (uint64_t) kMaxCols * cfg_.n_vocab * 4, err, "asst"))
            return false;
        I.mtp.attn = true;   // (has_mtp)
        weight_bytes_ = total;
        std::fprintf(stderr, "strata-dense: gemma4 assistant %s: %d layers of %d\n", mp, AL, AE);
    }
    reset();
    std::string serr;
    if (!sync(serr)) { err = serr; return false; }
    std::fprintf(stderr, "strata-dense: gemma4, %d layers (%d with their own K/V), heads %d / %d, window %d, %.2f GiB "
                 "of weights, K/V %.2f GiB, prompt chunks of %d tokens\n", NL, kv_from, D_swa, D_full, I.n_swa,
                 total / 1073741824.0, kv_bytes_ / 1073741824.0, I.chunk);
    return true;
}

bool DenseModel::forward_gemma4(const int32_t* tokens, int T, bool verify, int want, std::string& err) {
    Impl& I = *impl_;
    cudaStream_t s = I.stream;
    const int E = cfg_.n_embd, NL = cfg_.n_layer, H = cfg_.n_head, HK = cfg_.n_head_kv, P = I.ple_dim;
    const float eps = cfg_.rms_eps;
    const int64_t p0 = pos_;
    try {
        I.mark(s);
        // embeddings * sqrt(E); the per-layer inputs: (rms(W_proj x / sqrt E) * w + ple_tok * sqrt P) / sqrt 2
        for (int t = 0; t < T; ++t) {
            dequant_f32(I.embd.type, I.embd.dev, tokens[t], 1, E, I.x + (size_t) t * E, s);
            if (P > 0)
                dequant_f32(I.ple_tok.type, I.ple_tok.dev, tokens[t], 1, (int64_t) NL * P, I.ple_rows + (size_t) t * NL * P, s);
        }
        g4_scale(I.x, sqrtf((float) E), (int64_t) T * E, s);
        if (P > 0) {
            if (I.ple_model_f) q35_gemv_f32_cols(I.ple_model_f, I.x, E, I.ple_proj_out, E, NL * P, T, s);
            else {
                I.prep(I.x, E, T);
                I.mm(I.ple_model, I.ple_proj_out);
            }
            g4_ple(I.ple_proj_out, I.ple_rows, I.ple_proj_norm, eps, E, NL, T, I.ple, s);
        }
        I.stage("embed", s);
        for (int l = 0; l < NL; ++l) {
            Layer& L = I.layers[(size_t) l];
            const int D = L.D;
            const Layer& KV = L.kv_src >= 0 ? I.layers[(size_t) L.kv_src] : L;
            // attention: q (and k / v when the layer keeps its own), norms + rope, the core, the output projection
            q35_rms_rows(I.x, E, L.attn_norm, I.xn, nullptr, T, E, eps, s);
            I.prep(I.xn, E, T);
            I.mm(L.q, I.q_full);
            G4Kv kvw;   // (no k / v rows: the shared layers read their source's cache)
            if (L.kv_src < 0) {
                I.mm(L.k, I.kcur);
                I.mm(L.v, I.vcur);
                kvw = L.g4kv;
            }
            I.stage("attn_qkv", s);
            g4_attn_prep(I.q_full, I.kcur, I.vcur, T, H, KV.HK, D, L.q_norm, L.k_norm, eps, L.swa ? I.rope_swa : cfg_.rope_base,
                         L.swa ? nullptr : I.rope_freqs, (int) p0, I.qcur, kvw, s);
            g4_attn(I.qcur, KV.g4kv, (int) p0, T, H, KV.HK, D, L.swa ? I.n_swa : 0, 1.0f, I.attn, I.attn_scratch, s);
            I.stage(L.swa ? "attn_core_swa" : "attn_core_full", s);
            I.prep(I.attn, H * D, T);
            I.mm(L.o, I.mix);
            q35_rms_rows(I.mix, E, L.post_norm, I.mix, nullptr, T, E, eps, s);
            q35_add(I.x, I.mix, (int64_t) T * E, s);
            I.stage("attn_out", s);
            // the FFN: x += post_ffw_norm(down(gelu(gate) * up)) of ffn_norm(x)
            q35_rms_rows(I.x, E, L.ffn_pre_norm, I.xn, nullptr, T, E, eps, s);
            I.prep(I.xn, E, T);
            I.mm(L.ffn_gate, I.ffn_g);
            I.mm(L.ffn_up, I.ffn_u);
            g4_gelu_mul(I.ffn_g, I.ffn_u, I.ffn_h, (int64_t) T * cfg_.n_ff, s);
            I.prep(I.ffn_h, cfg_.n_ff, T);
            I.mm(L.ffn_down, I.mix);
            q35_rms_rows(I.mix, E, L.post_ffw_norm, I.mix, nullptr, T, E, eps, s);
            q35_add(I.x, I.mix, (int64_t) T * E, s);
            I.stage("ffn", s);
            // the per-layer input: x = (x + post_norm(proj(gelu(inp_gate x) * ple_l))) * out_scale (none: x *= out_scale)
            if (P == 0) {
                if (L.out_scale != 1.0f) g4_scale(I.x, L.out_scale, (int64_t) T * E, s);
                continue;
            }
            if (L.inp_gate_f) q35_gemv_f32_cols(L.inp_gate_f, I.x, E, I.ple_g, E, P, T, s);
            else {
                I.prep(I.x, E, T);
                I.mm(L.inp_gate, I.ple_g);
            }
            g4_gelu_ple(I.ple_g, I.ple, l, NL, T, s);
            if (L.ple_proj_f) q35_gemv_f32_cols(L.ple_proj_f, I.ple_g, P, I.mix, P, E, T, s);
            else {
                I.prep(I.ple_g, P, T);
                I.mm(L.ple_proj, I.mix);
            }
            q35_rms_rows(I.mix, E, L.ple_norm, I.mix, nullptr, T, E, eps, s);
            g4_add_scale(I.x, I.mix, L.out_scale, (int64_t) T * E, s);
            I.stage("ple", s);
        }
        q35_rms_rows(I.x, E, I.output_norm, I.hn, nullptr, T, E, eps, s);
        if (want != kNone) {
            const int rows = want == kAll ? T : 1;
            I.prep(want == kAll ? I.hn : I.hn + (size_t) (T - 1) * E, E, rows);
            I.mm(I.head, I.logits);
            if (I.softcap > 0.0f) g4_softcap(I.logits, cfg_.n_vocab, cfg_.n_vocab, rows, I.softcap, s);
        }
        I.stage("head", s);
        I.prof_flush(T);
        I.ring_hi = std::max<int64_t>(I.ring_hi, p0 + T - 1);
        if (verify) pending_ = T;
        else pos_ += T;
        return true;
    } catch (const std::exception& e) {
        err = std::string("gemma4: ") + e.what();
        return false;
    }
}

// ---------------------------------------------------------------------------------------------------- Muse Glimmer
// llama.cpp src/models/muse-glimmer.cpp: a dense transformer, 3 sliding-window (rope) + 1 global (NoPE) layers,
// gated attention (32 / 2 heads of 128), sandwich norms, SwiGLU, a logit scale and soft cap
bool DenseModel::load_muse(int64_t max_context, std::string& err) {
    Impl& I = *impl_;
    I.muse = true;
    const strata::GgufFile& g0 = *I.files.front();
    const std::string a = "muse-glimmer.";
    int64_t v = 0;
    auto need = [&](const char* key, int& dst) {
        if (!meta_int(g0, a + key, v)) { err = std::string("muse: missing ") + a + key; return false; }
        dst = (int) v;
        return true;
    };
    int pattern = 4;
    if (!need("block_count", cfg_.n_layer) || !need("embedding_length", cfg_.n_embd) ||
        !need("feed_forward_length", cfg_.n_ff) || !need("attention.head_count", cfg_.n_head) ||
        !need("attention.head_count_kv", cfg_.n_head_kv) || !need("attention.key_length", cfg_.head_dim) ||
        !need("attention.sliding_window", I.n_swa))
        return false;
    if (meta_int(g0, a + "attention.sliding_window_pattern", v)) pattern = (int) v;
    if (const auto* e = g0.get(a + "attention.layer_norm_rms_epsilon")) cfg_.rms_eps = (float) e->num();
    if (const auto* e = g0.get(a + "rope.freq_base")) cfg_.rope_base = (float) e->num();
    I.rope_swa = cfg_.rope_base;
    if (const auto* e = g0.get(a + "rope.freq_base_swa")) I.rope_swa = (float) e->num();
    if (const auto* e = g0.get(a + "final_logit_softcapping")) I.softcap = (float) e->num();
    if (const auto* e = g0.get(a + "logit_scale")) I.logit_scale = (float) e->num();
    if (const auto* e = g0.get("tokenizer.ggml.eos_token_id")) cfg_.eos_id = (int32_t) e->u;
    const int NL = cfg_.n_layer, E = cfg_.n_embd, H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;
    cfg_.n_rot = D;
    if (D != 128 || H % HK || H / HK > 16 || E % 256 || cfg_.n_ff % 256 || pattern < 1) {
        err = "muse: unsupported geometry (head 128, up to 16 query heads a KV head, widths of 256)";
        return false;
    }
    const Found* tok = I.find("token_embd.weight");
    if (!tok || tok->tensor->shape.size() != 2 || tok->tensor->shape[0] != (uint64_t) E) {
        err = "muse: token_embd.weight missing or not [n_embd, n_vocab]";
        return false;
    }
    cfg_.n_vocab = (int) tok->tensor->shape[1];
    max_context_ = max_context;
    if (cudaStreamCreate(&I.stream) != cudaSuccess) { err = "muse: CUDA stream"; return false; }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    {   // (every layer keeps the whole context's K/V: the sliding ones only read their window)
        const uint64_t kv_need = (uint64_t) NL * 2 * HK * D * max_context * 2;
        const uint64_t act = (uint64_t) I.chunk * (8ull * E + 4ull * cfg_.n_ff + 6ull * H * D) * 4 +
                             (uint64_t) kMaxCols * cfg_.n_vocab * 4 + 1024ull * 1024 * 1024;
        if ((uint64_t) free_b < kv_need + act) {
            err = "muse: the K/V cache for a context of " + std::to_string(max_context) + " needs " +
                  std::to_string(kv_need >> 20) + " MiB; " + std::to_string(free_b >> 20) + " MiB are free";
            return false;
        }
        I.budget = (uint64_t) free_b - kv_need - act;
    }
    uint64_t total = 0;
    I.layers.resize((size_t) NL);
    for (int l = 0; l < NL; ++l) {
        Layer& L = I.layers[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        L.attn = true;
        L.swa = l % pattern < pattern - 1;
        L.D = D;
        if (!I.upload_floats(I.find(p + "attn_norm.weight"), p + "attn_norm.weight", E, &L.attn_norm, err) ||
            !I.upload_floats(I.find(p + "post_attention_norm.weight"), p + "post_attention_norm.weight", E, &L.post_norm, err) ||
            !I.upload_floats(I.find(p + "ffn_norm.weight"), p + "ffn_norm.weight", E, &L.ffn_pre_norm, err) ||
            !I.upload_floats(I.find(p + "post_ffw_norm.weight"), p + "post_ffw_norm.weight", E, &L.post_ffw_norm, err) ||
            !I.upload_floats(I.find(p + "attn_q_norm.weight"), p + "attn_q_norm.weight", D, &L.q_norm, err) ||
            !I.upload_floats(I.find(p + "attn_k_norm.weight"), p + "attn_k_norm.weight", D, &L.k_norm, err) ||
            !I.upload_mat(I.find(p + "attn_q.weight"), p + "attn_q.weight", E, H * D, L.q, total, err) ||
            !I.upload_mat(I.find(p + "attn_k.weight"), p + "attn_k.weight", E, HK * D, L.k, total, err) ||
            !I.upload_mat(I.find(p + "attn_v.weight"), p + "attn_v.weight", E, HK * D, L.v, total, err) ||
            !I.upload_mat(I.find(p + "attn_gate.weight"), p + "attn_gate.weight", E, H * D, L.attn_gate, total, err) ||
            !I.upload_mat(I.find(p + "attn_output.weight"), p + "attn_output.weight", H * D, E, L.o, total, err) ||
            !I.upload_mat(I.find(p + "ffn_gate.weight"), p + "ffn_gate.weight", E, cfg_.n_ff, L.ffn_gate, total, err) ||
            !I.upload_mat(I.find(p + "ffn_up.weight"), p + "ffn_up.weight", E, cfg_.n_ff, L.ffn_up, total, err) ||
            !I.upload_mat(I.find(p + "ffn_down.weight"), p + "ffn_down.weight", cfg_.n_ff, E, L.ffn_down, total, err))
            return false;
        const uint64_t cells = (uint64_t) HK * max_context;
        if (!I.alloc(&L.g4kv.k, cells * D * 2, err, "K cache") || !I.alloc(&L.g4kv.v, cells * D * 2, err, "V cache"))
            return false;
        L.g4kv.max_ctx = (int) max_context;
        kv_bytes_ += cells * D * 4;
    }
    if (!I.upload_mat(tok, "token_embd.weight", E, cfg_.n_vocab, I.embd, total, err, false) ||
        !I.upload_floats(I.find("output_norm.weight"), "output_norm.weight", E, &I.output_norm, err))
        return false;
    if (const Found* h = I.find("output.weight")) {
        if (!I.upload_mat(h, "output.weight", E, cfg_.n_vocab, I.head, total, err, true, false)) return false;
    } else {
        I.head = I.embd;
    }
    I.head_draft = I.head;
    weight_bytes_ = total;
    host_bytes_ = I.host_bytes;
    const uint64_t R = (uint64_t) I.chunk, f = 4;
    const int max_in = std::max({E, cfg_.n_ff, H * D});
    I.max_in = std::max(max_in, 5 * E);   // (a DFlash drafter's fc reads 5 target rows)
    bool ok = I.alloc(&I.q8_1, native_q8_1_bytes(I.max_in, kMaxCols), err, "q8_1 scratch") &&
              I.alloc(&I.xh, R * I.max_in * 2, err, "fp16 activations") &&
              I.alloc(&I.x, R * E * f, err, "x") && I.alloc(&I.xn, R * E * f, err, "xn") &&
              I.alloc(&I.mix, R * E * f, err, "mix") && I.alloc(&I.hn, R * E * f, err, "hidden") &&
              I.alloc(&I.q_full, R * H * D * f, err, "q") && I.alloc(&I.qcur, R * H * D * f, err, "qcur") &&
              I.alloc(&I.kcur, R * HK * D * f, err, "kcur") && I.alloc(&I.vcur, R * HK * D * f, err, "vcur") &&
              I.alloc(&I.attn, R * H * D * f, err, "attn") && I.alloc(&I.attn_gate_out, R * H * D * f, err, "gate") &&
              I.alloc(&I.ffn_g, R * cfg_.n_ff * f, err, "ffn_g") && I.alloc(&I.ffn_u, R * cfg_.n_ff * f, err, "ffn_u") &&
              I.alloc(&I.ffn_h, R * cfg_.n_ff * f, err, "ffn_h") &&
              I.alloc(&I.logits, (uint64_t) kMaxCols * cfg_.n_vocab * f, err, "logits") &&
              I.alloc(&I.attn_scratch, (uint64_t) g4_attn_scratch_floats(kMaxCols, H, D, (int) max_context) * f, err,
                      "attention scratch") &&
              I.alloc(&I.tok_dev, R * 4, err, "tokens") &&
              I.alloc(&I.xh8, (uint64_t) 8 * I.max_in * 2, err, "fp16 rows") &&
              I.alloc(&I.mmt_part, (uint64_t) q35_mmt_part_floats(std::max({cfg_.n_vocab, cfg_.n_ff, H * D})) * f, err,
                      "GEMV partials");
    if (!ok) return false;
    if (!I.gemm.init(I.stream, (int64_t) cfg_.n_ff * E, err)) return false;
    if (const char* e = std::getenv("STRATA_DENSE_MMT_MIN")) I.mmt_min = std::max(1, std::atoi(e));
    if (const char* e = std::getenv("STRATA_DENSE_PACKED_DP4A")) I.packed_dp4a = std::atoi(e) != 0;
    reset();
    std::string serr;
    if (!sync(serr)) { err = serr; return false; }
    std::fprintf(stderr, "strata-dense: muse-glimmer, %d layers (%d of %d with a window of %d), %d / %d heads of %d, "
                 "%.2f GiB of weights, K/V %.2f GiB, prompt chunks of %d tokens\n", NL, pattern - 1, pattern, I.n_swa,
                 H, HK, D, total / 1073741824.0, kv_bytes_ / 1073741824.0, I.chunk);
    return true;
}

bool DenseModel::forward_muse(const int32_t* tokens, int T, bool verify, int want, std::string& err) {
    Impl& I = *impl_;
    cudaStream_t s = I.stream;
    const int E = cfg_.n_embd, NL = cfg_.n_layer, H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;
    const float eps = cfg_.rms_eps, post_eps = 1e-8f;
    const int64_t p0 = pos_;
    try {
        I.mark(s);
        for (int t = 0; t < T; ++t) dequant_f32(I.embd.type, I.embd.dev, tokens[t], 1, E, I.x + (size_t) t * E, s);
        q35_rms_rows(I.x, E, nullptr, I.x, nullptr, T, E, eps, s);   // the embedding's norm (no weight)
        I.stage("embed", s);
        for (int l = 0; l < NL; ++l) {
            Layer& L = I.layers[(size_t) l];
            q35_rms_rows(I.x, E, L.attn_norm, I.xn, nullptr, T, E, eps, s);
            I.prep(I.xn, E, T);
            I.mm(L.q, I.q_full);
            I.mm(L.k, I.kcur);
            I.mm(L.v, I.vcur);
            I.mm(L.attn_gate, I.attn_gate_out);
            I.stage("attn_qkv", s);
            g4_attn_prep(I.q_full, I.kcur, I.vcur, T, H, HK, D, L.q_norm, L.k_norm, eps, I.rope_swa, nullptr, (int) p0,
                         I.qcur, L.g4kv, s, L.swa ? 2 : 0, false);
            g4_attn(I.qcur, L.g4kv, (int) p0, T, H, HK, D, L.swa ? I.n_swa : 0, 1.0f / sqrtf((float) D), I.attn,
                    I.attn_scratch, s);
            g4_sigmoid_mul(I.attn, I.attn_gate_out, (int64_t) T * H * D, s);
            I.stage("attn_core", s);
            I.prep(I.attn, H * D, T);
            I.mm(L.o, I.mix);
            q35_rms_rows(I.mix, E, L.post_norm, I.mix, nullptr, T, E, post_eps, s);
            q35_add(I.x, I.mix, (int64_t) T * E, s);
            I.stage("attn_out", s);
            q35_rms_rows(I.x, E, L.ffn_pre_norm, I.xn, nullptr, T, E, eps, s);
            I.prep(I.xn, E, T);
            I.mm(L.ffn_gate, I.ffn_g);
            I.mm(L.ffn_up, I.ffn_u);
            dense_swiglu(I.ffn_g, I.ffn_u, I.ffn_h, (int64_t) T * cfg_.n_ff, s);
            I.prep(I.ffn_h, cfg_.n_ff, T);
            I.mm(L.ffn_down, I.mix);
            q35_rms_rows(I.mix, E, L.post_ffw_norm, I.mix, nullptr, T, E, post_eps, s);
            q35_add(I.x, I.mix, (int64_t) T * E, s);
            I.stage("ffn", s);
            if (I.df.on)   // the drafter's features: the residual at the input of its target layers
                for (size_t j = 0; j < I.df.target_layers.size(); ++j)
                    if (I.df.target_layers[j] == l + 1)
                        cudaMemcpy2DAsync(I.df.feat + j * E, I.df.target_layers.size() * E * 4, I.x, (size_t) E * 4,
                                          (size_t) E * 4, T, cudaMemcpyDeviceToDevice, s);
        }
        q35_rms_rows(I.x, E, I.output_norm, I.hn, nullptr, T, E, eps, s);
        if (want != kNone) {
            const int rows = want == kAll ? T : 1;
            I.prep(want == kAll ? I.hn : I.hn + (size_t) (T - 1) * E, E, rows);
            I.mm(I.head, I.logits);
            g4_scale(I.logits, I.logit_scale, (int64_t) rows * cfg_.n_vocab, s);
            if (I.softcap > 0.0f) g4_softcap(I.logits, cfg_.n_vocab, cfg_.n_vocab, rows, I.softcap, s);
        }
        I.stage("head", s);
        I.prof_flush(T);
        if (verify) pending_ = T;
        else pos_ += T;
        return true;
    } catch (const std::exception& e) {
        err = std::string("muse: ") + e.what();
        return false;
    }
}

}  // namespace strata::core
